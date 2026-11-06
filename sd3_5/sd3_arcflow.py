import copy
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps
from diffusers.utils import BaseOutput, apply_lora_scale
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class SD3ArcFlowTransformer2DOutput(BaseOutput):
    sample: torch.FloatTensor
    basic_velocity: torch.FloatTensor
    gating_probability: torch.FloatTensor
    momentum_factor: torch.FloatTensor


@dataclass
class ArcFlowStepOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    effective_velocity: torch.FloatTensor
    basic_velocity: torch.FloatTensor
    gating_probability: torch.FloatTensor
    momentum_factor: torch.FloatTensor


def _filter_config_for_class(config, cls):
    valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {key: value for key, value in dict(config).items() if key in valid_keys}


class SD3ArcFlowTransformer2DModel(SD3Transformer2DModel):
    """SD3/SD3.5 transformer with K trainable ArcFlow output heads.

    The backbone and the optional copied tail blocks keep the original SD3.5
    joint image/text attention implementation, including dual-attention blocks.
    Only the ArcFlow heads and LoRA adapters are intended to be trained.
    """

    def __init__(
        self,
        *args,
        num_arc_heads: int = 8,
        num_arc_head_layers: int = 0,
        momentum_init_bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if num_arc_heads <= 0:
            raise ValueError("`num_arc_heads` must be positive.")
        if not 0 <= num_arc_head_layers <= len(self.transformer_blocks):
            raise ValueError(
                "`num_arc_head_layers` must be between 0 and the number of transformer blocks "
                f"({len(self.transformer_blocks)}), got {num_arc_head_layers}."
            )

        self.register_to_config(
            num_arc_heads=num_arc_heads,
            num_arc_head_layers=num_arc_head_layers,
            momentum_init_bias=momentum_init_bias,
        )
        self.num_arc_heads = num_arc_heads
        self.num_arc_head_layers = num_arc_head_layers

        velocity_dim = self.config.patch_size**2 * self.out_channels
        self.arc_basic_velocity_heads = nn.ModuleList(
            [nn.Linear(self.inner_dim, velocity_dim) for _ in range(num_arc_heads)]
        )
        self.arc_gate_heads = nn.ModuleList(
            [nn.Linear(self.inner_dim, 1) for _ in range(num_arc_heads)]
        )
        self.arc_momentum_heads = nn.ModuleList(
            [nn.Linear(self.inner_dim, 1) for _ in range(num_arc_heads)]
        )
        self.arc_tail_blocks = nn.ModuleList(
            [
                nn.ModuleList(copy.deepcopy(self.transformer_blocks[-num_arc_head_layers:]))
                for _ in range(num_arc_heads)
            ]
            if num_arc_head_layers
            else [nn.ModuleList() for _ in range(num_arc_heads)]
        )
        self._init_arc_heads(momentum_init_bias)

    def _init_arc_heads(self, momentum_init_bias: float = 0.0):
        if self.num_arc_head_layers:
            source_tail = self.transformer_blocks[-self.num_arc_head_layers:]
            for head_tail in self.arc_tail_blocks:
                for target, source in zip(head_tail, source_tail):
                    target.load_state_dict(source.state_dict())

        with torch.no_grad():
            for head in self.arc_basic_velocity_heads:
                head.weight.copy_(self.proj_out.weight)
                if head.bias is not None and self.proj_out.bias is not None:
                    head.bias.copy_(self.proj_out.bias)
            for head in self.arc_gate_heads:
                head.weight.zero_()
                head.bias.zero_()
            for head in self.arc_momentum_heads:
                head.weight.zero_()
                head.bias.fill_(momentum_init_bias)

    @classmethod
    def from_sd3_transformer(
        cls,
        transformer: SD3Transformer2DModel,
        num_arc_heads: int = 8,
        num_arc_head_layers: int = 0,
    ):
        config = _filter_config_for_class(transformer.config, SD3Transformer2DModel)
        model = cls(
            **config,
            num_arc_heads=num_arc_heads,
            num_arc_head_layers=num_arc_head_layers,
        )
        missing, unexpected = model.load_state_dict(transformer.state_dict(), strict=False)
        non_arc_missing = [key for key in missing if not key.startswith("arc_")]
        if non_arc_missing or unexpected:
            raise ValueError(
                f"Could not convert SD3 transformer: missing={non_arc_missing}, unexpected={unexpected}"
            )
        model._init_arc_heads()
        try:
            reference = next(transformer.parameters())
            model.to(device=reference.device, dtype=reference.dtype)
        except StopIteration:
            pass
        model.train(transformer.training)
        return model

    def _unpatchify(self, hidden_states, height, width):
        patch_size = self.config.patch_size
        height //= patch_size
        width //= patch_size
        channels = hidden_states.shape[-1] // (patch_size * patch_size)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], height, width, patch_size, patch_size, channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0], channels, height * patch_size, width * patch_size
        )

    def _run_blocks(
        self,
        blocks,
        start_index,
        hidden_states,
        encoder_hidden_states,
        temb,
        joint_attention_kwargs,
        block_controlnet_hidden_states,
    ):
        for offset, block in enumerate(blocks):
            index_block = start_index + offset
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            if block_controlnet_hidden_states is not None and not block.context_pre_only:
                interval = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval)]
        return encoder_hidden_states, hidden_states

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: list = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[list[int]] = None,
        return_arcflow: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput, SD3ArcFlowTransformer2DOutput]:
        if skip_layers is not None:
            if return_arcflow:
                raise ValueError("`skip_layers` is not supported for ArcFlow outputs.")
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                block_controlnet_hidden_states=block_controlnet_hidden_states,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                skip_layers=skip_layers,
            )

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states).contiguous()
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        shared_count = len(self.transformer_blocks) - self.num_arc_head_layers
        encoder_hidden_states, hidden_states = self._run_blocks(
            self.transformer_blocks[:shared_count],
            0,
            hidden_states,
            encoder_hidden_states,
            temb,
            joint_attention_kwargs,
            block_controlnet_hidden_states,
        )

        if return_arcflow:
            normalized_heads = []
            for tail in self.arc_tail_blocks:
                _, head_hidden = self._run_blocks(
                    tail,
                    shared_count,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    block_controlnet_hidden_states,
                )
                normalized_heads.append(self.norm_out(head_hidden, temb))

            basic_velocity = torch.stack(
                [self._unpatchify(head(state), height, width) for head, state in zip(self.arc_basic_velocity_heads, normalized_heads)],
                dim=1,
            )
            # Gate and log-gamma are global scalars for each sample/head. SD3
            # has no dedicated image CLS token, so pool all image tokens before
            # applying the scalar heads. Keep singleton C/H/W dimensions so the
            # values broadcast over the complete component velocity field.
            pooled_heads = [state.mean(dim=1) for state in normalized_heads]
            gate_logits = torch.stack(
                [head(state) for head, state in zip(self.arc_gate_heads, pooled_heads)],
                dim=1,
            ).unsqueeze(-1).unsqueeze(-1)
            momentum_factor = torch.stack(
                [head(state) for head, state in zip(self.arc_momentum_heads, pooled_heads)],
                dim=1,
            ).unsqueeze(-1).unsqueeze(-1)
            gating_probability = torch.softmax(gate_logits, dim=1)
            sample = (gating_probability * basic_velocity).sum(dim=1)
            if not return_dict:
                return sample, basic_velocity, gating_probability, momentum_factor
            return SD3ArcFlowTransformer2DOutput(
                sample=sample,
                basic_velocity=basic_velocity,
                gating_probability=gating_probability,
                momentum_factor=momentum_factor,
            )

        _, hidden_states = self._run_blocks(
            self.transformer_blocks[shared_count:],
            shared_count,
            hidden_states,
            encoder_hidden_states,
            temb,
            joint_attention_kwargs,
            block_controlnet_hidden_states,
        )
        output = self._unpatchify(self.proj_out(self.norm_out(hidden_states, temb)), height, width)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def split_guided_arcflow_outputs(outputs, guidance_scale: float):
    basic_velocity, gating_probability, momentum_factor = outputs
    velocity_uncond, velocity_text = basic_velocity.chunk(2)
    gate_uncond, gate_text = gating_probability.chunk(2)
    momentum_uncond, momentum_text = momentum_factor.chunk(2)
    velocity = velocity_uncond + guidance_scale * (velocity_text - velocity_uncond)
    gate_logits = torch.log(gate_uncond.clamp_min(1e-8)) + guidance_scale * (
        torch.log(gate_text.clamp_min(1e-8)) - torch.log(gate_uncond.clamp_min(1e-8))
    )
    gate = torch.softmax(gate_logits, dim=1)
    momentum = momentum_uncond + guidance_scale * (momentum_text - momentum_uncond)
    return velocity, gate, momentum


def arcflow_step(
    basic_velocity,
    gating_probability,
    momentum_factor,
    timestep,
    sample,
    sigma_next,
    num_train_timesteps=1000,
    momentum_scale=1.0,
    momentum_clamp=20.0,
    eps=1e-4,
):
    t_start = timestep.to(device=sample.device, dtype=torch.float32) / float(num_train_timesteps)
    t_end = sigma_next.to(device=sample.device, dtype=torch.float32)
    while t_start.ndim < sample.ndim:
        t_start = t_start.unsqueeze(-1)
    while t_end.ndim < sample.ndim:
        t_end = t_end.unsqueeze(-1)
    t_start_heads = t_start.unsqueeze(1)
    t_end_heads = t_end.unsqueeze(1)
    delta_t = t_start_heads - t_end_heads
    log_gamma = (momentum_factor.float() * momentum_scale).clamp(-momentum_clamp, momentum_clamp)
    z = log_gamma * delta_t
    safe_z = torch.where(z.abs() < eps, torch.ones_like(z), z)
    phi = torch.where(z.abs() < eps, 1.0 + 0.5 * z + z.square() / 6.0, torch.expm1(z) / safe_z)
    coefficient = torch.exp(log_gamma * (1.0 - t_start_heads)) * delta_t * phi
    displacement = (gating_probability.float() * basic_velocity.float() * coefficient).sum(dim=1)
    interval = t_start - t_end
    effective_velocity = displacement / torch.where(interval.abs() < eps, torch.ones_like(interval), interval)
    effective_velocity = torch.where(interval.abs() < eps, torch.zeros_like(effective_velocity), effective_velocity)
    return ArcFlowStepOutput(
        prev_sample=(sample.float() - displacement).to(basic_velocity.dtype),
        effective_velocity=effective_velocity.to(basic_velocity.dtype),
        basic_velocity=basic_velocity,
        gating_probability=gating_probability,
        momentum_factor=momentum_factor,
    )


def set_flowmatch_timesteps(scheduler, device, num_inference_steps=None, timesteps=None, **scheduler_kwargs):
    """Configure a FlowMatch scheduler, optionally with exact model timesteps."""
    if timesteps is None:
        scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
        return scheduler.timesteps

    timesteps = torch.as_tensor(timesteps, device=device, dtype=torch.float32)
    if timesteps.ndim != 1 or timesteps.numel() == 0:
        raise ValueError("`timesteps` must be a non-empty one-dimensional sequence.")
    if not torch.all(timesteps[:-1] > timesteps[1:]):
        raise ValueError("`timesteps` must be strictly decreasing.")
    num_train_timesteps = float(scheduler.config.num_train_timesteps)
    if timesteps.max() > num_train_timesteps or timesteps.min() <= 0:
        raise ValueError(
            f"`timesteps` must be in (0, {num_train_timesteps:g}], got "
            f"[{timesteps.min().item():g}, {timesteps.max().item():g}]."
        )
    if not torch.isclose(timesteps[0], timesteps.new_tensor(num_train_timesteps)):
        raise ValueError(
            f"The first custom timestep must be {num_train_timesteps:g}, because sampling starts from pure noise."
        )

    # Treat the supplied values as the exact timesteps seen by the model. Do
    # not apply the scheduler's static/dynamic shift a second time.
    scheduler.timesteps = timesteps
    scheduler.sigmas = torch.cat(
        [timesteps / num_train_timesteps, torch.zeros(1, device=device, dtype=torch.float32)]
    )
    scheduler.num_inference_steps = timesteps.numel()
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler.timesteps


class StableDiffusion3ArcFlowPipeline(StableDiffusion3Pipeline):
    """Stable Diffusion 3 pipeline using ArcFlow analytical integration."""

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        prompt_2=None,
        prompt_3=None,
        height=None,
        width=None,
        num_inference_steps=2,
        timesteps=None,
        guidance_scale=1.0,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        max_sequence_length=256,
        momentum_scale=1.0,
        momentum_clamp=20.0,
        consistency_sampling=True,
    ):
        if not isinstance(self.transformer, SD3ArcFlowTransformer2DModel):
            raise TypeError("StableDiffusion3ArcFlowPipeline requires SD3ArcFlowTransformer2DModel.")
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds, pooled, negative_pooled = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_cfg,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_cfg:
            model_prompt = torch.cat([negative_prompt_embeds, prompt_embeds])
            model_pooled = torch.cat([negative_pooled, pooled])
        else:
            model_prompt, model_pooled = prompt_embeds, pooled
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", False):
            image_seq_len = (latents.shape[-2] // self.transformer.config.patch_size) * (
                latents.shape[-1] // self.transformer.config.patch_size
            )
            scheduler_kwargs["mu"] = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
        if timesteps is None:
            timesteps, _ = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, **scheduler_kwargs
            )
        else:
            timesteps = set_flowmatch_timesteps(self.scheduler, device, timesteps=timesteps)
        for index, timestep in enumerate(timesteps):
            model_input = torch.cat([latents, latents]) if do_cfg else latents
            model_timestep = timestep.expand(model_input.shape[0])
            output = self.transformer(
                hidden_states=model_input,
                timestep=model_timestep,
                encoder_hidden_states=model_prompt,
                pooled_projections=model_pooled,
                return_dict=False,
                return_arcflow=True,
            )[1:]
            if do_cfg:
                output = split_guided_arcflow_outputs(output, guidance_scale)
            sigma_next = torch.zeros((), device=device) if consistency_sampling else self.scheduler.sigmas[index + 1]
            x0 = arcflow_step(
                *output,
                timestep,
                latents,
                sigma_next,
                self.scheduler.config.num_train_timesteps,
                momentum_scale,
                momentum_clamp,
            ).prev_sample
            if consistency_sampling and index + 1 < len(timesteps):
                latents = consistency_renoise_pipeline(
                    x0, self.scheduler.sigmas[index + 1], randn_tensor(x0.shape, generator=generator, device=device, dtype=x0.dtype)
                )
            else:
                latents = x0
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)


def consistency_renoise_pipeline(x0, sigma, noise):
    sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    while sigma.ndim < x0.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * x0 + sigma * noise
