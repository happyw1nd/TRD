import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers import SanaPipeline, SanaTransformer2DModel
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import SanaLoraLoaderMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.pipelines.sana.pipeline_sana import (
    ASPECT_RATIO_4096_BIN,
    retrieve_timesteps,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers.utils import BaseOutput, USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import get_device, randn_tensor


try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


@dataclass
class ArcFlowTransformer2DOutput(BaseOutput):
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
    signature = inspect.signature(cls.__init__)
    valid_keys = set(signature.parameters.keys()) - {"self"}
    return {key: value for key, value in dict(config).items() if key in valid_keys}


class SanaArcFlowTransformer2DModel(SanaTransformer2DModel):
    """Sana transformer with K ArcFlow output heads.

    Each component predicts a basic velocity, a gating probability, and a
    momentum factor. Each head can optionally own a copied suffix of the base
    transformer followed by its output projections. The base `proj_out` is kept
    so plain Sana checkpoints can still be loaded and used for conservative
    initialization.
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
            raise ValueError("`num_arc_heads` must be a positive integer.")
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
        self.arc_output_channels = (
            self.config.in_channels
            if self.config.out_channels // 2 == self.config.in_channels
            else self.config.out_channels
        )

        inner_dim = self.proj_out.in_features
        patch_area = self.config.patch_size * self.config.patch_size
        velocity_dim = patch_area * self.arc_output_channels
        gate_dim = patch_area

        self.arc_basic_velocity_heads = nn.ModuleList(
            [nn.Linear(inner_dim, velocity_dim) for _ in range(num_arc_heads)]
        )
        self.arc_gate_heads = nn.ModuleList([nn.Linear(inner_dim, gate_dim) for _ in range(num_arc_heads)])
        self.arc_momentum_heads = nn.ModuleList(
            [nn.Linear(inner_dim, velocity_dim) for _ in range(num_arc_heads)]
        )
        self.arc_tail_blocks = nn.ModuleList(
            [
                nn.ModuleList(copy.deepcopy(self.transformer_blocks[-num_arc_head_layers:]))
                for _ in range(num_arc_heads)
            ]
            if num_arc_head_layers > 0
            else [nn.ModuleList() for _ in range(num_arc_heads)]
        )

        self._init_arc_heads(momentum_init_bias=momentum_init_bias)

    def _init_arc_heads(self, momentum_init_bias: float = 0.0):
        if self.num_arc_head_layers > 0:
            source_tail = self.transformer_blocks[-self.num_arc_head_layers:]
            for head_tail in self.arc_tail_blocks:
                for target_block, source_block in zip(head_tail, source_tail):
                    target_block.load_state_dict(source_block.state_dict())

        with torch.no_grad():
            source_weight = self.proj_out.weight[: self.arc_basic_velocity_heads[0].out_features]
            source_bias = self.proj_out.bias
            if source_bias is not None:
                source_bias = source_bias[: self.arc_basic_velocity_heads[0].out_features]

            for velocity_head in self.arc_basic_velocity_heads:
                velocity_head.weight.copy_(source_weight)
                if velocity_head.bias is not None and source_bias is not None:
                    velocity_head.bias.copy_(source_bias)

            for gate_head in self.arc_gate_heads:
                gate_head.weight.zero_()
                gate_head.bias.zero_()

            for momentum_head in self.arc_momentum_heads:
                momentum_head.weight.zero_()
                momentum_head.bias.fill_(momentum_init_bias)

    @classmethod
    def from_sana_transformer(
        cls,
        transformer: SanaTransformer2DModel,
        num_arc_heads: int = 8,
        num_arc_head_layers: int = 0,
        copy_device: bool = True,
        copy_dtype: bool = True,
    ) -> "SanaArcFlowTransformer2DModel":
        config = _filter_config_for_class(transformer.config, SanaTransformer2DModel)
        config["num_arc_heads"] = num_arc_heads
        config["num_arc_head_layers"] = num_arc_head_layers
        model = cls(**config)
        missing_keys, unexpected_keys = model.load_state_dict(transformer.state_dict(), strict=False)
        model._init_arc_heads()

        if unexpected_keys:
            logger.warning("Unexpected Sana keys while building ArcFlow transformer: %s", unexpected_keys)
        missing_arc_keys = [key for key in missing_keys if not key.startswith("arc_")]
        if missing_arc_keys:
            logger.warning("Missing non-ArcFlow keys while building ArcFlow transformer: %s", missing_arc_keys)

        try:
            reference_param = next(transformer.parameters())
            device = reference_param.device if copy_device else None
            dtype = reference_param.dtype if copy_dtype and torch.is_floating_point(reference_param) else None
            if device is not None or dtype is not None:
                model.to(device=device, dtype=dtype)
        except StopIteration:
            pass

        model.train(transformer.training)
        return model

    def _unpatchify(self, hidden_states: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
        p = self.config.patch_size
        hidden_states = hidden_states.reshape(batch_size, height, width, p, p, -1)
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        return hidden_states.reshape(batch_size, -1, height * p, width * p)

    def _run_backbone(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using PEFT has no effect.")

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size, _, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        shared_block_count = len(self.transformer_blocks) - self.num_arc_head_layers
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks[:shared_block_count]):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
        else:
            for index_block, block in enumerate(self.transformer_blocks[:shared_block_count]):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        return (
            hidden_states,
            lora_scale,
            batch_size,
            post_patch_height,
            post_patch_width,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            embedded_timestep,
            controlnet_block_samples,
            shared_block_count,
        )

    def _run_tail_blocks(
        self,
        hidden_states,
        blocks,
        start_index,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
        post_patch_height,
        post_patch_width,
        controlnet_block_samples,
    ):
        for offset, block in enumerate(blocks):
            index_block = start_index + offset
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
            if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
        return_arcflow: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput, ArcFlowTransformer2DOutput]:
        (
            hidden_states,
            lora_scale,
            batch_size,
            post_patch_height,
            post_patch_width,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            embedded_timestep,
            controlnet_block_samples,
            shared_block_count,
        ) = self._run_backbone(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            attention_kwargs=attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
        )

        if return_arcflow:
            head_hidden_states = []
            for head_tail in self.arc_tail_blocks:
                head_hidden = self._run_tail_blocks(
                    hidden_states,
                    head_tail,
                    shared_block_count,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    controlnet_block_samples,
                )
                head_hidden_states.append(
                    self.norm_out(head_hidden, embedded_timestep, self.scale_shift_table)
                )
            basic_velocity = torch.stack(
                [
                    self._unpatchify(head(head_hidden), batch_size, post_patch_height, post_patch_width)
                    for head, head_hidden in zip(self.arc_basic_velocity_heads, head_hidden_states)
                ],
                dim=1,
            )
            gating_logits = torch.stack(
                [
                    self._unpatchify(head(head_hidden), batch_size, post_patch_height, post_patch_width)
                    for head, head_hidden in zip(self.arc_gate_heads, head_hidden_states)
                ],
                dim=1,
            )
            momentum_factor = torch.stack(
                [
                    self._unpatchify(head(head_hidden), batch_size, post_patch_height, post_patch_width)
                    for head, head_hidden in zip(self.arc_momentum_heads, head_hidden_states)
                ],
                dim=1,
            )
            gating_probability = torch.softmax(gating_logits, dim=1)
            sample = (gating_probability * basic_velocity).sum(dim=1)

            if USE_PEFT_BACKEND:
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return sample, basic_velocity, gating_probability, momentum_factor

            return ArcFlowTransformer2DOutput(
                sample=sample,
                basic_velocity=basic_velocity,
                gating_probability=gating_probability,
                momentum_factor=momentum_factor,
            )

        hidden_states = self._run_tail_blocks(
            hidden_states,
            self.transformer_blocks[shared_block_count:],
            shared_block_count,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            post_patch_height,
            post_patch_width,
            controlnet_block_samples,
        )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)
        output = self._unpatchify(hidden_states, batch_size, post_patch_height, post_patch_width)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def _split_guided_arcflow_outputs(outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], guidance_scale: float):
    basic_velocity, gating_probability, momentum_factor = outputs
    velocity_uncond, velocity_text = basic_velocity.chunk(2)
    gate_uncond, gate_text = gating_probability.chunk(2)
    momentum_uncond, momentum_text = momentum_factor.chunk(2)

    guided_velocity = velocity_uncond + guidance_scale * (velocity_text - velocity_uncond)
    guided_gate_logits = torch.log(gate_uncond.clamp_min(1e-8)) + guidance_scale * (
        torch.log(gate_text.clamp_min(1e-8)) - torch.log(gate_uncond.clamp_min(1e-8))
    )
    guided_gate = torch.softmax(guided_gate_logits, dim=1)
    guided_momentum = momentum_uncond + guidance_scale * (momentum_text - momentum_uncond)
    return guided_velocity, guided_gate, guided_momentum


def arcflow_step(
    basic_velocity: torch.Tensor,
    gating_probability: torch.Tensor,
    momentum_factor: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    sigma_next: torch.Tensor,
    num_train_timesteps: int = 1000,
    momentum_scale: float = 1.0,
    momentum_clamp: float = 20.0,
    eps: float = 1e-4,
    return_dict: bool = True,
) -> Union[ArcFlowStepOutput, Tuple[torch.Tensor, ...]]:
    """Analytical ArcFlow integration for one flow-matching step.

    This implements the paper coefficient
    `C(gamma, t_s, t_e) = (gamma ** (1 - t_e) - gamma ** (1 - t_s)) / log(gamma)`,
    with the smooth `gamma -> 1` limit `t_s - t_e`. The model output
    `momentum_factor` is interpreted as `log(gamma)`.
    """

    t_start = timestep.to(device=sample.device, dtype=torch.float32) / float(num_train_timesteps)
    t_end = sigma_next.to(device=sample.device, dtype=torch.float32)
    while t_start.ndim < sample.ndim:
        t_start = t_start.unsqueeze(-1)
    while t_end.ndim < sample.ndim:
        t_end = t_end.unsqueeze(-1)

    t_start_for_heads = t_start.unsqueeze(1)
    t_end_for_heads = t_end.unsqueeze(1)
    delta_t = t_start_for_heads - t_end_for_heads

    log_gamma = (momentum_factor.float() * momentum_scale).clamp(
        min=-momentum_clamp, max=momentum_clamp
    )
    z = log_gamma * delta_t
    safe_z = torch.where(z.abs() < eps, torch.ones_like(z), z)
    phi = torch.where(z.abs() < eps, 1.0 + 0.5 * z + z * z / 6.0, torch.expm1(z) / safe_z)
    coefficient = torch.exp(log_gamma * (1.0 - t_start_for_heads)) * delta_t * phi

    component_displacement = basic_velocity.float() * coefficient
    displacement = (gating_probability.float() * component_displacement).sum(dim=1)

    interval = t_start - t_end
    safe_interval = torch.where(interval.abs() < eps, torch.ones_like(interval), interval)
    effective_velocity = torch.where(
        interval.abs() < eps,
        torch.zeros_like(displacement),
        displacement / safe_interval,
    )
    prev_sample = sample.float() - displacement
    prev_sample = prev_sample.to(dtype=basic_velocity.dtype)

    if not return_dict:
        return prev_sample, effective_velocity, basic_velocity, gating_probability, momentum_factor

    return ArcFlowStepOutput(
        prev_sample=prev_sample,
        effective_velocity=effective_velocity.to(dtype=basic_velocity.dtype),
        basic_velocity=basic_velocity,
        gating_probability=gating_probability,
        momentum_factor=momentum_factor,
    )


class SanaArcFlowPipeline(SanaPipeline, SanaLoraLoaderMixin):
    """Sana pipeline with ArcFlow's non-linear analytical denoising step."""

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "basic_velocity",
        "gating_probability",
        "momentum_factor",
        "effective_velocity",
    ]

    @classmethod
    def from_sana_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes],
        num_arc_heads: int = 8,
        num_arc_head_layers: int = 0,
        **kwargs,
    ) -> "SanaArcFlowPipeline":
        base_pipe = SanaPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        arc_transformer = SanaArcFlowTransformer2DModel.from_sana_transformer(
            base_pipe.transformer,
            num_arc_heads=num_arc_heads,
            num_arc_head_layers=num_arc_head_layers,
        )
        pipe = cls(
            tokenizer=base_pipe.tokenizer,
            text_encoder=base_pipe.text_encoder,
            vae=base_pipe.vae,
            transformer=arc_transformer,
            scheduler=base_pipe.scheduler,
        )
        return pipe

    def _get_aspect_ratio_bin(self):
        if self.transformer.config.sample_size == 128:
            return ASPECT_RATIO_4096_BIN
        if self.transformer.config.sample_size == 64:
            return ASPECT_RATIO_2048_BIN
        if self.transformer.config.sample_size == 32:
            return ASPECT_RATIO_1024_BIN
        if self.transformer.config.sample_size == 16:
            return ASPECT_RATIO_512_BIN
        raise ValueError("Invalid sample size")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 2,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: List[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
        momentum_scale: float = 1.0,
        momentum_clamp: float = 20.0,
        consistency_sampling: bool = False,
    ) -> Union[SanaPipelineOutput, Tuple]:
        if not isinstance(self.transformer, SanaArcFlowTransformer2DModel):
            raise TypeError(
                "`SanaArcFlowPipeline` requires `SanaArcFlowTransformer2DModel`. "
                "Use `SanaArcFlowPipeline.from_sana_pretrained(...)` or wrap the transformer with "
                "`SanaArcFlowTransformer2DModel.from_sana_transformer(...)`."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if use_resolution_binning:
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=self._get_aspect_ratio_bin())

        self.check_inputs(
            prompt,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        self._num_timesteps = len(timesteps)
        transformer_dtype = self.transformer.dtype

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep * self.transformer.config.timestep_scale

                arc_outputs = self.transformer(
                    latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    return_arcflow=True,
                    attention_kwargs=self.attention_kwargs,
                )
                _, basic_velocity, gating_probability, momentum_factor = arc_outputs

                if self.do_classifier_free_guidance:
                    basic_velocity, gating_probability, momentum_factor = _split_guided_arcflow_outputs(
                        (basic_velocity, gating_probability, momentum_factor), guidance_scale
                    )

                sigma_next = (
                    torch.zeros((), device=latents.device, dtype=torch.float32)
                    if consistency_sampling
                    else self.scheduler.sigmas[i + 1]
                )
                step_output = arcflow_step(
                    basic_velocity=basic_velocity,
                    gating_probability=gating_probability,
                    momentum_factor=momentum_factor,
                    timestep=t,
                    sample=latents,
                    sigma_next=sigma_next,
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    momentum_scale=momentum_scale,
                    momentum_clamp=momentum_clamp,
                )
                x0 = step_output.prev_sample
                effective_velocity = step_output.effective_velocity
                if consistency_sampling and i + 1 < len(timesteps):
                    renoise = randn_tensor(
                        x0.shape,
                        generator=generator,
                        device=x0.device,
                        dtype=x0.dtype,
                    )
                    sigma_next = self.scheduler.sigmas[i + 1].to(
                        device=x0.device, dtype=x0.dtype
                    )
                    while sigma_next.ndim < x0.ndim:
                        sigma_next = sigma_next.unsqueeze(-1)
                    latents = (1.0 - sigma_next) * x0 + sigma_next * renoise
                else:
                    latents = x0

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for key in callback_on_step_end_tensor_inputs:
                        callback_kwargs[key] = locals()[key]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            torch_accelerator_module = getattr(torch, get_device(), torch.cuda)
            oom_error = (
                torch.OutOfMemoryError
                if is_torch_version(">=", "2.5.0")
                else torch_accelerator_module.OutOfMemoryError
            )
            try:
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            except oom_error as e:
                warnings.warn(
                    f"{e}. \n"
                    f"Try VAE tiling for large images, for example: "
                    f"pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)"
                )
                raise
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

        if output_type != "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)
