#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sd3_arcflow import (
    SD3ArcFlowTransformer2DModel,
    StableDiffusion3ArcFlowPipeline,
    arcflow_step,
    set_flowmatch_timesteps,
    split_guided_arcflow_outputs,
)


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)
ARCFLOW_OUTPUT_HEADS_FILE = "arcflow_output_heads.pt"
ARCFLOW_OUTPUT_HEAD_PREFIXES = (
    "arc_tail_blocks",
    "arc_basic_velocity_heads",
    "arc_gate_heads",
    "arc_momentum_heads",
)


def is_arcflow_output_head(name):
    return name.startswith(ARCFLOW_OUTPUT_HEAD_PREFIXES)


def enable_student_adapter(transformer):
    transformer.set_adapter("student")
    if hasattr(transformer, "enable_adapters"):
        transformer.enable_adapters()


def disable_lora_adapters(transformer):
    if hasattr(transformer, "disable_adapters"):
        transformer.disable_adapters()
    else:
        transformer.set_adapters([])


def initialize_arcflow_momentum_heads(transformer, gamma_min=0.8, gamma_max=1.25):
    count = len(transformer.arc_momentum_heads)
    log_gammas = torch.zeros(1) if count == 1 else torch.linspace(math.log(gamma_min), math.log(gamma_max), count)
    if count > 1:
        log_gammas[count // 2] = 0.0
    with torch.no_grad():
        for head, log_gamma in zip(transformer.arc_momentum_heads, log_gammas):
            head.weight.zero_()
            if head.bias is not None:
                head.bias.fill_(log_gamma.item())


def get_arcflow_output_head_state_dict(transformer):
    return {name: value.detach().cpu() for name, value in transformer.state_dict().items() if is_arcflow_output_head(name)}


def get_trainable_parameter_state_dict(transformer):
    return {name: value.detach().clone() for name, value in transformer.named_parameters() if value.requires_grad}


def swap_trainable_parameter_state_(transformer, state_dict):
    parameters = dict(transformer.named_parameters())
    with torch.no_grad():
        for name, reference in state_dict.items():
            current = parameters[name].detach().clone()
            parameters[name].copy_(reference.to(parameters[name]))
            reference.copy_(current.to(reference))


def refresh_trainable_parameter_state_dict_(transformer, state_dict=None, ema_decay=0.0):
    if state_dict is None:
        return get_trainable_parameter_state_dict(transformer)
    parameters = dict(transformer.named_parameters())
    with torch.no_grad():
        for name, reference in state_dict.items():
            current = parameters[name].detach().to(reference)
            reference.mul_(ema_decay).add_(current, alpha=1.0 - ema_decay)
    return state_dict


def predict_arcflow_output(transformer, **kwargs):
    output = transformer(return_dict=False, return_arcflow=True, **kwargs)
    return output[1], output[2], output[3]


def guide_arcflow_output(output, guidance_scale):
    return split_guided_arcflow_outputs(output, guidance_scale)


def arcflow_instantaneous_velocity(output, timesteps, momentum_clamp):
    basic_velocity, gating_probability, log_gamma = output
    exponent = (1.0 - timesteps.float()).view(-1, 1, 1, 1, 1)
    momentum = torch.exp(log_gamma.float().clamp(-momentum_clamp, momentum_clamp) * exponent)
    return (gating_probability.float() * basic_velocity.float() * momentum).sum(dim=1).to(basic_velocity.dtype)


def fixed_policy_velocity(output, policy_timesteps, query_timesteps, momentum_scale, momentum_clamp):
    basic_velocity, gating_probability, log_gamma = output
    delta = (policy_timesteps.float() - query_timesteps.float()).view(-1, 1, 1, 1, 1)
    momentum = torch.exp((log_gamma.float() * momentum_scale).clamp(-momentum_clamp, momentum_clamp) * delta)
    return (gating_probability.float() * basic_velocity.float() * momentum).sum(dim=1).to(basic_velocity.dtype)


def summarize_arcflow_heads(accelerator, output, momentum_scale, momentum_clamp):
    """Return distributed means of each head's gate probability and base gamma."""
    _, gating_probability, log_gamma = output
    reduce_dims = (0, 2, 3, 4)
    gate_means = gating_probability.detach().float().mean(dim=reduce_dims)
    gamma = torch.exp(
        (log_gamma.detach().float() * float(momentum_scale)).clamp(
            min=-float(momentum_clamp), max=float(momentum_clamp)
        )
    )
    gamma_means = gamma.mean(dim=reduce_dims)
    gate_means = accelerator.reduce(gate_means, reduction="mean")
    gamma_means = accelerator.reduce(gamma_means, reduction="mean")
    metrics = {}
    for head_index, (gate_mean, gamma_mean) in enumerate(zip(gate_means, gamma_means)):
        metrics[f"arcflow/head_{head_index}/gate"] = gate_mean
        metrics[f"arcflow/head_{head_index}/gamma"] = gamma_mean
    return metrics


def sample_synchronized_rollout_steps(scheduler, timestep_sampling_clip_range, device):
    """Sample a timestep uniformly, then select the rollout depth containing it."""
    num_train_timesteps = float(scheduler.config.num_train_timesteps)
    clip_upper_timestep, clip_lower_timestep = timestep_sampling_clip_range
    clip_lower = torch.tensor(clip_lower_timestep, device=device, dtype=torch.float32)
    clip_upper = torch.tensor(clip_upper_timestep, device=device, dtype=torch.float32)

    # Rollout k owns the interval between its anchor timestep and the next
    # scheduler timestep. The final rollout owns the interval down to zero.
    boundaries = scheduler.sigmas.float() * num_train_timesteps
    interval_uppers = torch.minimum(boundaries[:-1], clip_upper)
    interval_lowers = torch.maximum(boundaries[1:], clip_lower)
    interval_widths = (interval_uppers - interval_lowers).clamp_min(0.0)
    total_width = interval_widths.sum()
    if total_width <= 0:
        raise ValueError(
            "No rollout interval overlaps --timestep_sampling_clip_range: "
            f"clip=[{clip_lower_timestep:g}, {clip_upper_timestep:g}]."
        )

    # Sampling proportional to interval width is equivalent to uniformly
    # sampling a raw timestep from the clipped range and mapping it to the
    # rollout interval that contains it.
    sampled_offset = torch.rand((), device=device) * total_width
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(sampled_offset, src=0)
    rollout_index = torch.searchsorted(interval_widths.cumsum(dim=0), sampled_offset, right=True)
    return int(rollout_index.item()) + 1


def consistency_renoise(x0, sigma, noise):
    sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    while sigma.ndim < x0.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * x0 + sigma * noise


def expand_conditioning(prompt, pooled, uncond_prompt, uncond_pooled, guidance_scale):
    if guidance_scale > 1.0:
        return torch.cat([uncond_prompt, prompt]), torch.cat([uncond_pooled, pooled])
    return prompt, pooled


def teacher_velocity_at_xt(
    transformer, transformer_unwrapped, xt, timesteps, prompt_embeds, pooled_prompt_embeds,
    uncond_prompt_embeds, uncond_pooled_prompt_embeds, guidance_scale, num_train_timesteps,
):
    teacher_prompt, teacher_pooled = expand_conditioning(
        prompt_embeds, pooled_prompt_embeds, uncond_prompt_embeds, uncond_pooled_prompt_embeds, guidance_scale
    )
    if guidance_scale > 1.0:
        xt = torch.cat([xt, xt])
        timesteps = torch.cat([timesteps, timesteps])
    velocity = transformer(
        hidden_states=xt,
        timestep=timesteps.float() * num_train_timesteps,
        encoder_hidden_states=teacher_prompt,
        pooled_projections=teacher_pooled,
        return_dict=False,
    )[0]
    if guidance_scale > 1.0:
        uncond, cond = velocity.chunk(2)
        velocity = uncond + guidance_scale * (cond - uncond)
    return velocity


def compute_trajectory_alignment_loss(
    transformer, transformer_unwrapped, x_t_hat, policy_timesteps, prompt_embeds, pooled_prompt_embeds,
    uncond_prompt_embeds, uncond_pooled_prompt_embeds, teacher_guidance_scale, num_train_timesteps,
    ode_step_size, max_integration_length, momentum_scale, momentum_clamp, student_guidance_scale,
):
    student_prompt, student_pooled = expand_conditioning(
        prompt_embeds, pooled_prompt_embeds, uncond_prompt_embeds, uncond_pooled_prompt_embeds,
        student_guidance_scale,
    )
    student_xt = torch.cat([x_t_hat, x_t_hat]) if student_guidance_scale > 1.0 else x_t_hat
    student_timestep = (
        torch.cat([policy_timesteps, policy_timesteps])
        if student_guidance_scale > 1.0 else policy_timesteps
    )
    student_output = predict_arcflow_output(
        transformer,
        hidden_states=student_xt,
        timestep=student_timestep * num_train_timesteps,
        encoder_hidden_states=student_prompt,
        pooled_projections=student_pooled,
    )
    if student_guidance_scale > 1.0:
        student_output = guide_arcflow_output(student_output, student_guidance_scale)
    sample = x_t_hat
    query_t = policy_timesteps
    target_t = (policy_timesteps - max_integration_length).clamp_min(0.0)
    active = torch.ones_like(query_t, dtype=torch.bool)
    total_error = torch.zeros((), device=x_t_hat.device, dtype=torch.float32)
    total_count = torch.zeros((), device=x_t_hat.device, dtype=torch.float32)
    disable_lora_adapters(transformer_unwrapped)
    try:
        for _ in range(math.ceil(max_integration_length / ode_step_size) + 1):
            with torch.no_grad():
                teacher_velocity = teacher_velocity_at_xt(
                    transformer, transformer_unwrapped, sample, query_t, prompt_embeds, pooled_prompt_embeds,
                    uncond_prompt_embeds, uncond_pooled_prompt_embeds, teacher_guidance_scale, num_train_timesteps,
                )
            student_velocity = fixed_policy_velocity(
                student_output, policy_timesteps, query_t, momentum_scale, momentum_clamp
            )
            error = (student_velocity.float() - teacher_velocity.float()).square()
            total_error += (error * active.view(-1, 1, 1, 1)).sum()
            total_count += active.sum() * error[0].numel()
            reached = query_t <= target_t
            next_t = torch.maximum(query_t - ode_step_size, target_t)
            sample = (sample + (next_t - query_t).view(-1, 1, 1, 1).to(sample) * teacher_velocity).detach()
            query_t = next_t
            active = active & ~reached
    finally:
        enable_student_adapter(transformer_unwrapped)
    return total_error / total_count.clamp_min(1.0)


def compute_trd_loss(
    accelerator, transformer, scheduler, latents, prompt_embeds, pooled_prompt_embeds,
    uncond_prompt_embeds, uncond_pooled_prompt_embeds, args, reference_trainable_state,
):
    transformer_unwrapped = accelerator.unwrap_model(transformer)
    scheduler_kwargs = {}
    if scheduler.config.get("use_dynamic_shifting", False):
        image_seq_len = (latents.shape[-2] // transformer_unwrapped.config.patch_size) * (
            latents.shape[-1] // transformer_unwrapped.config.patch_size
        )
        scheduler_kwargs["mu"] = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.16),
        )
    full_timesteps = set_flowmatch_timesteps(
        scheduler,
        latents.device,
        num_inference_steps=args.num_inference_steps,
        timesteps=args.denoising_timesteps,
        **scheduler_kwargs,
    )
    schedule_length = len(full_timesteps)
    rollout_steps = sample_synchronized_rollout_steps(
        scheduler,
        args.timestep_sampling_clip_range,
        latents.device,
    )
    timesteps = scheduler.timesteps[:rollout_steps]
    batch_size = latents.shape[0]
    num_train_timesteps = float(scheduler.config.num_train_timesteps)
    student_prompt, student_pooled = expand_conditioning(
        prompt_embeds, pooled_prompt_embeds, uncond_prompt_embeds, uncond_pooled_prompt_embeds,
        args.student_guidance_scale,
    )
    renoise_noises = [torch.randn_like(latents) for _ in range(rollout_steps - 1)]
    reference_x0 = None
    anchor_output = None

    for index, anchor_timestep in enumerate(timesteps):
        latents = latents.detach()
        enable_student_adapter(transformer_unwrapped)
        if args.student_guidance_scale > 1.0:
            model_input = torch.cat([latents, latents])
            model_timestep = anchor_timestep.expand(batch_size).repeat(2)
        else:
            model_input = latents
            model_timestep = anchor_timestep.expand(batch_size)
        is_anchor = index == rollout_steps - 1
        if is_anchor and args.distribution_preservation_weight > 0.0:
            swap_trainable_parameter_state_(transformer_unwrapped, reference_trainable_state)
            try:
                with torch.no_grad():
                    reference_output = predict_arcflow_output(
                        transformer, hidden_states=model_input, timestep=model_timestep,
                        encoder_hidden_states=student_prompt, pooled_projections=student_pooled,
                    )
                    if args.student_guidance_scale > 1.0:
                        reference_output = guide_arcflow_output(reference_output, args.student_guidance_scale)
                    reference_x0 = arcflow_step(
                        *reference_output, anchor_timestep, latents,
                        torch.zeros((), device=latents.device), scheduler.config.num_train_timesteps,
                        args.arc_momentum_scale, args.arc_momentum_clamp,
                    ).prev_sample.detach()
            finally:
                swap_trainable_parameter_state_(transformer_unwrapped, reference_trainable_state)
        with (torch.enable_grad() if is_anchor else torch.no_grad()):
            output = predict_arcflow_output(
                transformer, hidden_states=model_input, timestep=model_timestep,
                encoder_hidden_states=student_prompt, pooled_projections=student_pooled,
            )
            if args.student_guidance_scale > 1.0:
                output = guide_arcflow_output(output, args.student_guidance_scale)
            if is_anchor:
                anchor_output = output
            x0 = arcflow_step(
                *output, anchor_timestep, latents, torch.zeros((), device=latents.device),
                scheduler.config.num_train_timesteps, args.arc_momentum_scale, args.arc_momentum_clamp,
            ).prev_sample
        if index + 1 < rollout_steps:
            latents = consistency_renoise(x0.detach(), scheduler.sigmas[index + 1], renoise_noises[index])
        else:
            latents = x0

    if anchor_output is None:
        raise RuntimeError("No ArcFlow anchor output was produced.")
    head_metrics = summarize_arcflow_heads(
        accelerator, anchor_output, args.arc_momentum_scale, args.arc_momentum_clamp
    )
    head_metrics["arcflow/anchor_timestep"] = timesteps[-1].detach().float()
    student_x0 = latents
    # Match the FM/DMD query time to the interval that selected the rollout
    # depth. A k-step rollout uses [t_{k+1}, t_k], while the full rollout uses
    # [0, t_k].
    interval_upper = scheduler.sigmas[rollout_steps - 1].float()
    interval_lower = (
        scheduler.sigmas[rollout_steps].float()
        if rollout_steps < schedule_length
        else torch.zeros((), device=latents.device, dtype=torch.float32)
    )

    clip_upper_timestep, clip_lower_timestep = args.timestep_sampling_clip_range
    if clip_upper_timestep > num_train_timesteps:
        raise ValueError(
            "The upper value of --timestep_sampling_clip_range cannot exceed "
            f"the scheduler's num_train_timesteps ({num_train_timesteps:g})."
        )
    clip_lower = torch.tensor(
        clip_lower_timestep / num_train_timesteps, device=latents.device, dtype=torch.float32
    )
    clip_upper = torch.tensor(
        clip_upper_timestep / num_train_timesteps, device=latents.device, dtype=torch.float32
    )
    sampling_lower = torch.maximum(interval_lower, clip_lower)
    sampling_upper = torch.minimum(interval_upper, clip_upper)
    if sampling_lower >= sampling_upper:
        raise ValueError(
            "The timestep sampling interval does not overlap --timestep_sampling_clip_range: "
            f"rollout interval=[{interval_lower.item() * num_train_timesteps:g}, "
            f"{interval_upper.item() * num_train_timesteps:g}], "
            f"clip=[{clip_lower_timestep:g}, {clip_upper_timestep:g}]."
        )
    shared_t = sampling_lower + torch.rand(
        batch_size, device=latents.device, dtype=torch.float32
    ) * (sampling_upper - sampling_lower)
    shared_noise = torch.randn_like(student_x0)
    t_view = shared_t.view(-1, 1, 1, 1).to(student_x0)
    x_t_hat = ((1.0 - t_view) * student_x0.detach() + t_view * shared_noise).detach()
    raw_timestep = shared_t.to(student_x0.dtype) * num_train_timesteps
    velocity_input = torch.cat([x_t_hat, x_t_hat]) if args.student_guidance_scale > 1.0 else x_t_hat
    velocity_timestep = torch.cat([raw_timestep, raw_timestep]) if args.student_guidance_scale > 1.0 else raw_timestep

    if args.self_velocity_weight > 0.0:
        student_output = predict_arcflow_output(
            transformer, hidden_states=velocity_input, timestep=velocity_timestep,
            encoder_hidden_states=student_prompt, pooled_projections=student_pooled,
        )
        if args.student_guidance_scale > 1.0:
            student_output = guide_arcflow_output(student_output, args.student_guidance_scale)
        student_velocity = arcflow_instantaneous_velocity(student_output, shared_t, args.arc_momentum_clamp)
        self_velocity_loss = F.mse_loss(student_velocity.float(), (shared_noise - student_x0.detach()).float())
    else:
        self_velocity_loss = torch.zeros((), device=latents.device)

    distribution_loss = (
        F.mse_loss(student_x0.float(), reference_x0.float())
        if reference_x0 is not None else torch.zeros((), device=latents.device)
    )

    if args.dmd_weight > 0.0:
        with torch.no_grad():
            disable_lora_adapters(transformer_unwrapped)
            teacher_velocity = teacher_velocity_at_xt(
                transformer, transformer_unwrapped, x_t_hat, shared_t, prompt_embeds, pooled_prompt_embeds,
                uncond_prompt_embeds, uncond_pooled_prompt_embeds, args.teacher_guidance_scale,
                num_train_timesteps,
            )
            enable_student_adapter(transformer_unwrapped)
            student_output = predict_arcflow_output(
                transformer, hidden_states=velocity_input, timestep=velocity_timestep,
                encoder_hidden_states=student_prompt, pooled_projections=student_pooled,
            )
            if args.student_guidance_scale > 1.0:
                student_output = guide_arcflow_output(student_output, args.student_guidance_scale)
            student_velocity = arcflow_instantaneous_velocity(student_output, shared_t, args.arc_momentum_clamp)
            velocity_delta = (teacher_velocity - student_velocity).detach()
        dmd_loss = (velocity_delta.float() * student_x0.float()).mean()
    else:
        dmd_loss = torch.zeros((), device=latents.device)
    enable_student_adapter(transformer_unwrapped)

    trajectory_loss = (
        compute_trajectory_alignment_loss(
            transformer, transformer_unwrapped, x_t_hat, shared_t, prompt_embeds, pooled_prompt_embeds,
            uncond_prompt_embeds, uncond_pooled_prompt_embeds, args.teacher_guidance_scale,
            num_train_timesteps, args.ode_step_size, args.trajectory_max_integration_length,
            args.arc_momentum_scale, args.arc_momentum_clamp, args.student_guidance_scale,
        )
        if args.trajectory_alignment_weight > 0.0 else torch.zeros((), device=latents.device)
    )
    loss = (
        args.self_velocity_weight * self_velocity_loss
        + args.distribution_preservation_weight * distribution_loss
        + args.dmd_weight * dmd_loss
        + args.trajectory_alignment_weight * trajectory_loss
    )
    return (
        loss,
        self_velocity_loss,
        distribution_loss,
        dmd_loss,
        trajectory_loss,
        student_x0.detach(),
        rollout_steps,
        head_metrics,
    )

def to_01(img):
    # img: -1..1 -> 0..1
    return (img + 1.0) / 2.0

def latent_to_pixel(latents, vae):
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    return image


def save_decoded_student_samples(latents, vae, output_dir, step, rollout_steps):
    save_dir = Path(output_dir) / "train_samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        images = latent_to_pixel(latents.detach().to(device=next(vae.parameters()).device, dtype=torch.float32), vae)
        images = to_01(images).clamp(0.0, 1.0).cpu()
    for index, image in enumerate(images):
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array).save(save_dir / f"step-{step:08d}_sample-{index:02d}_rollout-{rollout_steps}.png")

def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train SD3/SD3.5 TRD LoRA and ArcFlow heads.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--prompt_database",
        type=str,
        required=True,
        help="Path to a tab-separated prompt dataset.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10,
        help=(
            "Run validation every X steps."
        ),
    )
    parser.add_argument(
        "--val_resolution",
        type=int,
        default=512,
        help=(
            "val_resolution."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help=(
            "Inference steps for validation."
        ),
    )
    parser.add_argument(
        "--denoising_timesteps",
        type=str,
        default=None,
        help=(
            "Comma-separated exact SD3 denoising timesteps in strictly decreasing order, for example "
            "'1000,500'. The first value must equal the scheduler's training timestep count (normally 1000). "
            "When set, this replaces the scheduler-generated training and validation schedule."
        ),
    )
    parser.add_argument(
        "--timestep_sampling_clip_range",
        type=str,
        default="950,50",
        help=(
            "Upper and lower raw SD3 timestep bounds for FM/DMD noising-time sampling, in 'upper,lower' "
            "order. The default '950,50' avoids the extreme endpoints."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help=("lora alpha."),
    )

    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--student_learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gate_learning_rate",
        type=float,
        default=None,
        help="Learning rate for ArcFlow gate heads. Defaults to --student_learning_rate.",
    )
    parser.add_argument(
        "--gamma_learning_rate",
        type=float,
        default=None,
        help="Learning rate for ArcFlow momentum/gamma heads. Defaults to --student_learning_rate.",
    )
    parser.add_argument(
        "--gamma_min",
        type=float,
        default=0.8,
        help="Minimum gamma used to initialize the ArcFlow momentum heads.",
    )
    parser.add_argument(
        "--gamma_max",
        type=float,
        default=1.25,
        help="Maximum gamma used to initialize the ArcFlow momentum heads.",
    )
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--sample_saving_steps", type=int, default=10)
    parser.add_argument("--num_arc_heads", type=int, default=4)
    parser.add_argument("--num_arc_head_layers", type=int, default=1)
    parser.add_argument("--arc_momentum_scale", type=float, default=1.0)
    parser.add_argument("--arc_momentum_clamp", type=float, default=20.0)
    parser.add_argument("--student_guidance_scale", type=float, default=1.0)
    parser.add_argument("--teacher_guidance_scale", type=float, default=4.5)
    parser.add_argument("--validation_guidance_scale", type=float, default=1.0)
    parser.add_argument("--self_velocity_weight", type=float, default=1.0)
    parser.add_argument("--distribution_preservation_weight", type=float, default=1.0)
    parser.add_argument("--dmd_weight", type=float, default=1.0)
    parser.add_argument("--trajectory_alignment_weight", type=float, default=1.0)
    parser.add_argument("--ode_step_size", type=float, default=0.05)
    parser.add_argument("--trajectory_max_integration_length", type=float, default=0.3)
    parser.add_argument("--preservation_reference_update_steps", type=int, default=30)
    parser.add_argument("--preservation_reference_ema_decay", type=float, default=0.0)
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="Comma-separated transformer layer suffixes to receive LoRA adapters.",
    )
    parser.add_argument(
        "--lora_blocks",
        type=str,
        default=None,
        help="Comma-separated transformer block indices to receive LoRA adapters.",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.denoising_timesteps is not None:
        try:
            args.denoising_timesteps = [
                float(value.strip()) for value in args.denoising_timesteps.split(",") if value.strip()
            ]
        except ValueError as error:
            raise ValueError("`--denoising_timesteps` must be a comma-separated list of numbers.") from error
        if not args.denoising_timesteps:
            raise ValueError("`--denoising_timesteps` must contain at least one timestep.")
        if any(
            earlier <= later
            for earlier, later in zip(args.denoising_timesteps, args.denoising_timesteps[1:])
        ):
            raise ValueError("`--denoising_timesteps` must be strictly decreasing.")

    try:
        timestep_clip_values = [
            float(value.strip()) for value in args.timestep_sampling_clip_range.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "`--timestep_sampling_clip_range` must contain two comma-separated numbers in 'upper,lower' order."
        ) from error
    if len(timestep_clip_values) != 2:
        raise ValueError(
            "`--timestep_sampling_clip_range` must contain exactly two values in 'upper,lower' order."
        )
    clip_upper, clip_lower = timestep_clip_values
    if not (clip_upper > clip_lower >= 0):
        raise ValueError(
            "`--timestep_sampling_clip_range` must satisfy upper > lower >= 0."
        )
    args.timestep_sampling_clip_range = (clip_upper, clip_lower)

    if min(args.self_velocity_weight, args.distribution_preservation_weight, args.dmd_weight,
           args.trajectory_alignment_weight) < 0:
        raise ValueError("TRD loss weights must be non-negative.")
    if args.num_arc_heads <= 0 or args.num_arc_head_layers < 0:
        raise ValueError("Invalid ArcFlow head configuration.")
    if not 0 < args.gamma_min <= args.gamma_max:
        raise ValueError("`--gamma_min` and `--gamma_max` must satisfy 0 < gamma_min <= gamma_max.")
    if args.num_inference_steps <= 0:
        raise ValueError("`--num_inference_steps` must be positive.")
    if args.sample_saving_steps < 0:
        raise ValueError("`--sample_saving_steps` must be non-negative.")
    if args.validation_steps <= 0:
        raise ValueError("`--validation_steps` must be positive.")
    if args.arc_momentum_clamp <= 0:
        raise ValueError("`--arc_momentum_clamp` must be positive.")
    if not 0 < args.ode_step_size <= 1 or not 0 < args.trajectory_max_integration_length <= 1:
        raise ValueError("TRD integration lengths must be in (0, 1].")
    if args.preservation_reference_update_steps <= 0:
        raise ValueError("`preservation_reference_update_steps` must be positive.")
    if not 0 <= args.preservation_reference_ema_decay < 1:
        raise ValueError("`preservation_reference_ema_decay` must be in [0, 1).")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args




def collate_fn(examples):
    return {"prompts": [example["prompt"] for example in examples]}


def tracker_config(args):
    """Convert argparse values to TensorBoard-compatible hparam scalars."""
    scalar_types = (bool, int, float, str)
    return {
        name: value if isinstance(value, scalar_types) else str(value)
        for name, value in vars(args).items()
    }


class PromptTSVDataset(Dataset):
    def __init__(self, path):
        self.prompts = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[1].strip():
                    continue
                self.prompts.append(parts[1])
        if not self.prompts:
            raise ValueError(f"Prompt TSV contains no valid rows: {path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    student_scheduler = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    transformer = SD3ArcFlowTransformer2DModel.from_sd3_transformer(
        transformer,
        num_arc_heads=args.num_arc_heads,
        num_arc_head_layers=args.num_arc_head_layers,
    )
    initialize_arcflow_momentum_heads(
        transformer,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_suffixes = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_suffixes = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
    if args.lora_blocks is not None:
        target_blocks = [int(block.strip()) for block in args.lora_blocks.split(",")]
    else:
        target_blocks = list(range(len(transformer.transformer_blocks)))
    target_modules = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.startswith(f"transformer_blocks.{block}.") for block in target_blocks)
        and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    if not target_modules:
        raise ValueError("No SD3 transformer LoRA target modules were found.")

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config, adapter_name="student")
    enable_student_adapter(transformer)
    for name, parameter in transformer.named_parameters():
        if is_arcflow_output_head(name):
            parameter.requires_grad = True

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    if args.upcast_before_saving:
                        model = model.to(torch.float32)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="student")
                    arcflow_output_heads_to_save = get_arcflow_output_head_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusion3Pipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
            torch.save(arcflow_output_heads_to_save, os.path.join(output_dir, ARCFLOW_OUTPUT_HEADS_FILE))

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                transformer_ = unwrap_model(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="student")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        heads_path = os.path.join(input_dir, ARCFLOW_OUTPUT_HEADS_FILE)
        if os.path.exists(heads_path):
            transformer_.load_state_dict(torch.load(heads_path, map_location="cpu"), strict=False)
        else:
            logger.warning("No ArcFlow output heads found at %s", heads_path)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gate_learning_rate is None:
        args.gate_learning_rate = args.student_learning_rate
    if args.gamma_learning_rate is None:
        args.gamma_learning_rate = args.student_learning_rate
    if min(args.student_learning_rate, args.gate_learning_rate, args.gamma_learning_rate) < 0:
        raise ValueError("Student, gate, and gamma learning rates must be non-negative.")

    if args.scale_lr:
        lr_scale = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        args.student_learning_rate *= lr_scale
        args.gate_learning_rate *= lr_scale
        args.gamma_learning_rate *= lr_scale

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    student_lora_parameters = [
        p for name, p in transformer.named_parameters()
        if "student" in name and p.requires_grad and not is_arcflow_output_head(name)
    ]
    student_gate_parameters = [
        p for name, p in transformer.named_parameters()
        if name.startswith("arc_gate_heads") and p.requires_grad
    ]
    student_gamma_parameters = [
        p for name, p in transformer.named_parameters()
        if name.startswith("arc_momentum_heads") and p.requires_grad
    ]
    student_arc_parameters = [
        p for name, p in transformer.named_parameters()
        if is_arcflow_output_head(name)
        and not name.startswith(("arc_gate_heads", "arc_momentum_heads"))
        and p.requires_grad
    ]
    print(f"Number of trainable parameters in student LoRA: {sum(p.numel() for p in student_lora_parameters)}")
    print(f"Number of trainable parameters in ArcFlow basic/tail heads: {sum(p.numel() for p in student_arc_parameters)}")
    print(f"Number of trainable parameters in ArcFlow gate heads: {sum(p.numel() for p in student_gate_parameters)}")
    print(f"Number of trainable parameters in ArcFlow gamma heads: {sum(p.numel() for p in student_gamma_parameters)}")

    # Optimization parameters
    student_params_to_optimize = [
        {"params": student_lora_parameters, "lr": args.student_learning_rate, "name": "student_lora"},
        {"params": student_arc_parameters, "lr": args.student_learning_rate, "name": "arcflow_basic_tail"},
        {"params": student_gate_parameters, "lr": args.gate_learning_rate, "name": "arcflow_gate"},
        {"params": student_gamma_parameters, "lr": args.gamma_learning_rate, "name": "arcflow_gamma"},
    ]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as error:
                raise ImportError("Install bitsandbytes to use --use_8bit_adam.") from error
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        student_optimizer = optimizer_class(
            student_params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError as error:
            raise ImportError("Install prodigyopt to use --optimizer=prodigy.") from error
        if args.use_8bit_adam:
            logger.warning("`--use_8bit_adam` is ignored when using Prodigy.")
        student_optimizer = prodigyopt.Prodigy(
            student_params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = PromptTSVDataset(args.prompt_database)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)


            # 无条件（uncond）——用空字符串编码，而不是零向量
            uncond_prompts = [""] * (len(prompt) if isinstance(prompt, (list, tuple)) else prompt.shape[0])
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, uncond_prompts, args.max_sequence_length
            )
            uncond_prompt_embeds = uncond_prompt_embeds.to(accelerator.device)
            uncond_pooled_prompt_embeds = uncond_pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, uncond_prompt_embeds, uncond_pooled_prompt_embeds


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    student_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=student_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, student_optimizer, train_dataloader, student_lr_scheduler = accelerator.prepare(
        transformer, student_optimizer, train_dataloader, student_lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sd3-trd"
        accelerator.init_trackers(tracker_name, config=tracker_config(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    student_loss = torch.zeros(1, device=accelerator.device)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = Path(args.resume_from_checkpoint).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(args.output_dir) / checkpoint_path
            checkpoint_path = checkpoint_path.resolve()
        else:
            candidates = [
                path for path in Path(args.output_dir).glob("checkpoint-*")
                if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
            ]
            checkpoint_path = max(
                candidates,
                key=lambda path: int(path.name.split("-")[1]),
                default=None,
            )

        if checkpoint_path is None or not checkpoint_path.is_dir():
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            accelerator.load_state(str(checkpoint_path))
            global_step = int(checkpoint_path.name.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    if args.resume_from_checkpoint is None:
        initial_checkpoint_path = os.path.join(args.output_dir, "checkpoint-0")
        should_save_initial = not os.path.exists(initial_checkpoint_path)
        accelerator.wait_for_everyone()
        if should_save_initial:
            accelerator.save_state(initial_checkpoint_path)
            logger.info("Saved initial student state to %s", initial_checkpoint_path)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    preservation_reference_state = refresh_trainable_parameter_state_dict_(accelerator.unwrap_model(transformer))
    logger.info(
        "Initialized old-student snapshot; refreshing every %s optimizer steps with EMA decay %s.",
        args.preservation_reference_update_steps,
        args.preservation_reference_ema_decay,
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                prompt_embeds, pooled_prompt_embeds, uncond_prompt_embeds, uncond_pooled_prompt_embeds = compute_text_embeddings(
                    batch["prompts"], text_encoders, tokenizers
                )
                conditioning = [prompt_embeds, pooled_prompt_embeds, uncond_prompt_embeds, uncond_pooled_prompt_embeds]
                conditioning = [value.to(dtype=weight_dtype) for value in conditioning]
                latent_size = args.val_resolution // 8
                noise = torch.randn(
                    len(batch["prompts"]),
                    accelerator.unwrap_model(transformer).config.in_channels,
                    latent_size,
                    latent_size,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                loss_values = compute_trd_loss(
                    accelerator, transformer, student_scheduler, noise, *conditioning, args, preservation_reference_state
                )
                (
                    student_loss,
                    self_loss,
                    preservation_loss,
                    dmd_loss,
                    trajectory_loss,
                    student_x0,
                    rollout_steps,
                    head_metrics,
                ) = loss_values
                accelerator.backward(student_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                student_optimizer.step()
                student_lr_scheduler.step()
                student_optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if global_step % args.preservation_reference_update_steps == 0:
                    preservation_reference_state = refresh_trainable_parameter_state_dict_(
                        accelerator.unwrap_model(transformer), preservation_reference_state,
                        args.preservation_reference_ema_decay,
                    )
                if accelerator.is_main_process and args.sample_saving_steps > 0 and global_step % args.sample_saving_steps == 0:
                    save_decoded_student_samples(student_x0, vae, args.output_dir, global_step, rollout_steps)
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = [
                            name for name in os.listdir(args.output_dir)
                            if name.startswith("checkpoint-") and name != "checkpoint-0"
                        ]
                        checkpoints.sort(key=lambda name: int(name.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint))
                    accelerator.wait_for_everyone()
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                if (
                    accelerator.is_main_process
                    and args.validation_prompt is not None
                    and global_step % args.validation_steps == 0
                ):
                    pipeline = StableDiffusion3ArcFlowPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        text_encoder_3=accelerator.unwrap_model(text_encoder_three),
                        transformer=accelerator.unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    log_validation(
                        pipeline,
                        args,
                        accelerator,
                        {
                            "prompt": args.validation_prompt,
                            "num_inference_steps": args.num_inference_steps,
                            "timesteps": args.denoising_timesteps,
                            "guidance_scale": args.validation_guidance_scale,
                            "height": args.val_resolution,
                            "width": args.val_resolution,
                            "momentum_scale": args.arc_momentum_scale,
                            "momentum_clamp": args.arc_momentum_clamp,
                        },
                        epoch,
                    )
                    del pipeline
                    free_memory()

            logs = {
                "loss_stu": student_loss.detach().item(),
                "self_velocity_loss": self_loss.detach().item(),
                "distribution_preservation_loss": preservation_loss.detach().item(),
                "dmd_loss": dmd_loss.detach().item(),
                "trajectory_alignment_loss": trajectory_loss.detach().item(),
                "rollout_steps": rollout_steps,
                "lr": student_lr_scheduler.get_last_lr()[0],
            }
            current_lrs = {
                group.get("name", f"group_{index}"): group["lr"]
                for index, group in enumerate(student_optimizer.param_groups)
            }
            logs.update(
                {
                    "lr/student": current_lrs["student_lora"],
                    "lr/gate": current_lrs["arcflow_gate"],
                    "lr/gamma": current_lrs["arcflow_gamma"],
                }
            )
            logs.update({name: value.item() for name, value in head_metrics.items()})
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer, adapter_name="student")

        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        torch.save(get_arcflow_output_head_state_dict(transformer), os.path.join(args.output_dir, ARCFLOW_OUTPUT_HEADS_FILE))
        if args.validation_prompt and args.num_validation_images > 0:
            enable_student_adapter(transformer)
            pipeline = StableDiffusion3ArcFlowPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline_args = {
                "prompt": args.validation_prompt,
                "num_inference_steps": args.num_inference_steps,
                "timesteps": args.denoising_timesteps,
                "guidance_scale": args.validation_guidance_scale,
                "height": args.val_resolution,
                "width": args.val_resolution,
                "momentum_scale": args.arc_momentum_scale,
                "momentum_clamp": args.arc_momentum_clamp,
            }
            log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
