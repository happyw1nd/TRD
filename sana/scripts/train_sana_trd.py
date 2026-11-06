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

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=1.0.0",
#     "transformers>=4.47.0",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.14.0",
#     "sentencepiece",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import logging
import math
import os
import shutil
import sys
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
from transformers import AutoTokenizer, Gemma2Model

import diffusers
from diffusers import (
    AutoencoderDC,
    DPMSolverMultistepScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sana_arcflow import (
    SanaArcFlowPipeline,
    SanaArcFlowTransformer2DModel,
    _split_guided_arcflow_outputs,
    arcflow_step,
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
ARCFLOW_SLOW_OUTPUT_HEAD_PREFIX = "arc_momentum_heads"

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def is_arcflow_output_head(name: str) -> bool:
    return name.startswith(ARCFLOW_OUTPUT_HEAD_PREFIXES)


def is_slow_arcflow_output_head(name: str) -> bool:
    return name.startswith(ARCFLOW_SLOW_OUTPUT_HEAD_PREFIX)


def enable_student_adapter(transformer):
    transformer.set_adapter("default")
    if hasattr(transformer, "enable_adapters"):
        transformer.enable_adapters()


def disable_lora_adapters(transformer):
    if hasattr(transformer, "disable_adapters"):
        transformer.disable_adapters()


def initialize_arcflow_momentum_heads(transformer, gamma_min: float = 0.4, gamma_max: float = 5.0):
    num_heads = len(transformer.arc_momentum_heads)
    if num_heads == 1:
        log_gammas = torch.zeros(1)
    else:
        log_gammas = torch.linspace(math.log(gamma_min), math.log(gamma_max), num_heads)
        log_gammas[num_heads // 2] = 0.0

    with torch.no_grad():
        for head, log_gamma in zip(transformer.arc_momentum_heads, log_gammas):
            head.weight.zero_()
            if head.bias is not None:
                head.bias.fill_(log_gamma.item())


def get_arcflow_output_head_state_dict(transformer):
    return {
        name: param.detach().cpu()
        for name, param in transformer.state_dict().items()
        if is_arcflow_output_head(name)
    }



def get_trainable_parameter_state_dict(transformer):
    return {
        name: param.detach().clone()
        for name, param in transformer.named_parameters()
        if param.requires_grad
    }


def swap_trainable_parameter_state_(transformer, state_dict):
    params = dict(transformer.named_parameters())
    missing = [name for name in state_dict if name not in params]
    if missing:
        raise ValueError(f"Reference trainable state has parameters not found in the model: {missing}")
    with torch.no_grad():
        for name, value in state_dict.items():
            param = params[name]
            current = param.detach().clone()
            param.copy_(value.to(device=param.device, dtype=param.dtype))
            value.copy_(current.to(device=value.device, dtype=value.dtype))


def refresh_trainable_parameter_state_dict_(transformer, state_dict=None, ema_decay=0.0):
    if state_dict is None:
        return get_trainable_parameter_state_dict(transformer)
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("`ema_decay` must be in [0, 1).")
    params = dict(transformer.named_parameters())
    missing = [name for name in state_dict if name not in params]
    if missing:
        raise ValueError(f"Reference trainable state has parameters not found in the model: {missing}")
    with torch.no_grad():
        for name, value in state_dict.items():
            current = params[name].detach().to(device=value.device, dtype=value.dtype)
            if ema_decay == 0.0:
                value.copy_(current)
            else:
                value.mul_(ema_decay).add_(current, alpha=1.0 - ema_decay)
    return state_dict

def arcflow_instantaneous_velocity(basic_velocity, gating_probability, log_gamma, timesteps, log_gamma_clip=20.0):
    exponent = (1.0 - timesteps.float()).view(-1, 1, 1, 1, 1)
    log_gamma = log_gamma.float().clamp(min=-log_gamma_clip, max=log_gamma_clip)
    momentum_factor = torch.exp(log_gamma * exponent)
    velocity = (gating_probability.float() * basic_velocity.float() * momentum_factor).sum(dim=1)
    return velocity.to(dtype=basic_velocity.dtype)


def predict_arcflow_output(transformer, **model_kwargs):
    outputs = transformer(return_dict=False, return_arcflow=True, **model_kwargs)
    return outputs[1], outputs[2], outputs[3]


def guide_arcflow_output(output, guidance_scale):
    return _split_guided_arcflow_outputs(output, guidance_scale)


def detach_arcflow_output(output):
    return tuple(value.detach() for value in output)


def arcflow_output_to_instantaneous_velocity(output, timesteps, momentum_clamp):
    return arcflow_instantaneous_velocity(*output, timesteps=timesteps, log_gamma_clip=momentum_clamp)


def arcflow_velocity_from_fixed_policy(
    output,
    policy_timesteps,
    query_timesteps,
    momentum_scale,
    momentum_clamp,
):
    basic_velocity, gating_probability, log_gamma = output
    delta = (policy_timesteps.float() - query_timesteps.float()).view(-1, 1, 1, 1, 1)
    log_gamma = (log_gamma.float() * float(momentum_scale)).clamp(
        min=-float(momentum_clamp), max=float(momentum_clamp)
    )
    momentum = torch.exp(log_gamma * delta)
    velocity = (gating_probability.float() * basic_velocity.float() * momentum).sum(dim=1)
    return velocity.to(dtype=basic_velocity.dtype)


def teacher_velocity_at_xt(
    transformer,
    transformer_unwrapped,
    xt,
    timesteps,
    prompt_embeds,
    prompt_attention_mask,
    uncond_prompt_embeds,
    uncond_prompt_attention_mask,
    guidance_scale,
    num_train_timesteps,
):
    if guidance_scale > 1.0:
        teacher_xt = torch.cat([xt, xt], dim=0)
        teacher_timesteps = torch.cat([timesteps, timesteps], dim=0)
        teacher_prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
        teacher_prompt_mask = torch.cat([uncond_prompt_attention_mask, prompt_attention_mask], dim=0)
    else:
        teacher_xt = xt
        teacher_timesteps = timesteps
        teacher_prompt_embeds = prompt_embeds
        teacher_prompt_mask = prompt_attention_mask

    teacher_velocity = transformer(
        hidden_states=teacher_xt,
        timestep=(
            teacher_timesteps.float()
            * float(num_train_timesteps)
            * transformer_unwrapped.config.timestep_scale
        ),
        encoder_hidden_states=teacher_prompt_embeds,
        encoder_attention_mask=teacher_prompt_mask,
        return_dict=False,
    )[0]
    if guidance_scale > 1.0:
        teacher_uncond, teacher_text = teacher_velocity.chunk(2)
        teacher_velocity = teacher_uncond + guidance_scale * (teacher_text - teacher_uncond)
    return teacher_velocity.to(dtype=xt.dtype)


def compute_trajectory_alignment_loss(
    transformer,
    transformer_unwrapped,
    x_t_hat,
    policy_timesteps,
    prompt_embeds,
    prompt_attention_mask,
    uncond_prompt_embeds,
    uncond_prompt_attention_mask,
    teacher_guidance_scale,
    num_train_timesteps,
    ode_step_size,
    max_integration_length,
    momentum_scale,
    momentum_clamp,
):
    """Align teacher ODE velocities with one fixed ArcFlow policy predicted at x_t_hat."""
    student_output = predict_arcflow_output(
        transformer,
        hidden_states=x_t_hat,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=(
            policy_timesteps
            * float(num_train_timesteps)
            * transformer_unwrapped.config.timestep_scale
        ),
    )

    trajectory_sample = x_t_hat
    query_timesteps = policy_timesteps
    target_timesteps = (policy_timesteps - float(max_integration_length)).clamp_min(0.0)
    active = torch.ones_like(query_timesteps, dtype=torch.bool)
    total_squared_error = torch.zeros((), device=x_t_hat.device, dtype=torch.float32)
    total_active_elements = torch.zeros((), device=x_t_hat.device, dtype=torch.float32)

    disable_lora_adapters(transformer_unwrapped)
    try:
        # Keep the number of teacher forwards identical across distributed ranks.
        max_trajectory_points = math.ceil(float(max_integration_length) / float(ode_step_size)) + 1
        for _ in range(max_trajectory_points):
            with torch.no_grad():
                teacher_velocity = teacher_velocity_at_xt(
                    transformer=transformer,
                    transformer_unwrapped=transformer_unwrapped,
                    xt=trajectory_sample,
                    timesteps=query_timesteps,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    uncond_prompt_embeds=uncond_prompt_embeds,
                    uncond_prompt_attention_mask=uncond_prompt_attention_mask,
                    guidance_scale=teacher_guidance_scale,
                    num_train_timesteps=num_train_timesteps,
                )

            student_velocity = arcflow_velocity_from_fixed_policy(
                output=student_output,
                policy_timesteps=policy_timesteps,
                query_timesteps=query_timesteps,
                momentum_scale=momentum_scale,
                momentum_clamp=momentum_clamp,
            )
            active_mask = active.view(-1, 1, 1, 1)
            squared_error = (student_velocity.float() - teacher_velocity.float()).square()
            total_squared_error = total_squared_error + (squared_error * active_mask).sum()
            total_active_elements = total_active_elements + active.sum() * squared_error[0].numel()

            reached_endpoint = query_timesteps <= target_timesteps
            next_timesteps = torch.maximum(
                query_timesteps - float(ode_step_size), target_timesteps
            )
            dt = (next_timesteps - query_timesteps).view(-1, 1, 1, 1).to(dtype=trajectory_sample.dtype)
            trajectory_sample = (trajectory_sample + dt * teacher_velocity).detach()
            query_timesteps = next_timesteps
            active = active & ~reached_endpoint
    finally:
        enable_student_adapter(transformer_unwrapped)

    return total_squared_error / total_active_elements.clamp_min(1.0)


def take_arcflow_step(
    output,
    timestep,
    sample,
    sigma_next,
    num_train_timesteps,
    momentum_scale,
    momentum_clamp,
):
    return arcflow_step(
        basic_velocity=output[0],
        gating_probability=output[1],
        momentum_factor=output[2],
        timestep=timestep,
        sample=sample,
        sigma_next=sigma_next,
        num_train_timesteps=num_train_timesteps,
        momentum_scale=momentum_scale,
        momentum_clamp=momentum_clamp,
    )


def sample_synchronized_rollout_steps(num_inference_steps, device):
    """Sample one prefix length shared by every distributed training rank."""
    if num_inference_steps < 1:
        raise ValueError("`num_inference_steps` must be at least 1.")
    rollout_steps = torch.randint(1, num_inference_steps + 1, (), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(rollout_steps, src=0)
    return int(rollout_steps.item())


def consistency_renoise(x0, sigma, noise):
    """Sample x_t=(1-t)x_0+t*epsilon on the flow-matching probability path."""
    sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    while sigma.ndim < x0.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * x0 + sigma * noise


def validation_pipeline_args(args):
    return {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.validation_guidance_scale,
        "height": args.val_resolution,
        "width": args.val_resolution,
        "momentum_scale": args.arc_momentum_scale,
        "momentum_clamp": args.arc_momentum_clamp,
        "consistency_sampling": True,
    }

def to_01(img):
    # img: -1..1 -> 0..1
    return (img + 1.0) / 2.0

def latent_to_pixel(latents, vae):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    return image

def save_decoded_student_samples(latents, vae, output_dir, step):
    save_dir = Path(output_dir) / "train_samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    vae_device = next(vae.parameters()).device

    with torch.no_grad():
        images = latent_to_pixel(latents.detach().to(device=vae_device, dtype=torch.float32), vae)
        images = to_01(images).clamp(0.0, 1.0).cpu()

    for sample_index, image in enumerate(images):
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255.0).round().astype(np.uint8)
        image_path = save_dir / f"step-{step:08d}_sample-{sample_index:02d}.png"
        Image.fromarray(image).save(image_path)

    logger.info("Saved %s decoded student sample(s) to %s", len(images), save_dir)


def compute_student_random_t_anchor_loss(
    accelerator, transformer, scheduler, latents, prompt_embeds, prompt_attention_mask,
    uncond_prompt_embeds, uncond_prompt_attention_mask,
    guidance_scale=1.0, teacher_guidance_scale=4.5, num_inference_steps=4,
    momentum_scale=1.0, momentum_clamp=20.0,
    self_velocity_weight=1.0, distribution_preservation_weight=1.0,
    dmd_weight=1.0, trajectory_alignment_weight=1.0, ode_step_size=0.02,
    trajectory_max_integration_length=0.3,
    reference_trainable_state=None,
):
    scheduler.set_timesteps(num_inference_steps)
    full_timesteps = scheduler.timesteps.to(device=latents.device)
    rollout_steps = sample_synchronized_rollout_steps(num_inference_steps, latents.device)
    timesteps = full_timesteps[:rollout_steps]
    if len(timesteps) == 0:
        raise ValueError("`num_inference_steps` must produce at least one scheduler timestep.")

    batch_size = latents.shape[0]
    initial_latents = latents.detach()
    transformer_unwrapped = accelerator.unwrap_model(transformer)
    num_train_timesteps = float(scheduler.config.num_train_timesteps)
    renoise_noises = [torch.randn_like(initial_latents) for _ in range(rollout_steps - 1)]

    if guidance_scale > 1.0:
        student_prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
        student_prompt_mask = torch.cat([uncond_prompt_attention_mask, prompt_attention_mask], dim=0)
    else:
        student_prompt_embeds = prompt_embeds
        student_prompt_mask = prompt_attention_mask
    teacher_prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
    teacher_prompt_mask = torch.cat([uncond_prompt_attention_mask, prompt_attention_mask], dim=0)

    selected_output = None
    selected_anchor_latents = None
    selected_anchor_timestep = None
    reference_student_sample_x0 = None

    for index, anchor_timestep in enumerate(timesteps):
        anchor_timestep = anchor_timestep.to(device=latents.device)
        latents = latents.detach()
        enable_student_adapter(transformer_unwrapped)
        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latents] * 2)
            timestep = anchor_timestep.expand(batch_size).repeat(2)
        else:
            latent_model_input = latents
            timestep = anchor_timestep.expand(batch_size)
        model_timestep = timestep.to(latent_model_input.device) * transformer_unwrapped.config.timestep_scale
        grad_enabled = index == rollout_steps - 1
        if grad_enabled and distribution_preservation_weight > 0.0:
            if reference_trainable_state is None:
                raise ValueError(
                    "Distribution preservation requires `reference_trainable_state` for the old student."
                )
            swap_trainable_parameter_state_(transformer_unwrapped, reference_trainable_state)
            try:
                with torch.no_grad():
                    reference_output = predict_arcflow_output(
                        transformer,
                        hidden_states=latent_model_input,
                        encoder_hidden_states=student_prompt_embeds,
                        encoder_attention_mask=student_prompt_mask,
                        timestep=model_timestep,
                    )
                    if guidance_scale > 1.0:
                        reference_output = guide_arcflow_output(reference_output, guidance_scale)
                    reference_student_sample_x0 = take_arcflow_step(
                        reference_output,
                        anchor_timestep,
                        latents,
                        torch.zeros((), device=latents.device, dtype=torch.float32),
                        scheduler.config.num_train_timesteps,
                        momentum_scale,
                        momentum_clamp,
                    ).prev_sample.detach()
            finally:
                swap_trainable_parameter_state_(transformer_unwrapped, reference_trainable_state)
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            output = predict_arcflow_output(
                transformer,
                hidden_states=latent_model_input,
                encoder_hidden_states=student_prompt_embeds,
                encoder_attention_mask=student_prompt_mask,
                timestep=model_timestep,
            )
        if guidance_scale > 1.0:
            output = guide_arcflow_output(output, guidance_scale)
        if grad_enabled:
            selected_output = output
            selected_anchor_latents = latents.detach()
            selected_anchor_timestep = anchor_timestep
        if grad_enabled:
            x0 = take_arcflow_step(
                output, anchor_timestep, latents,
                torch.zeros((), device=latents.device, dtype=torch.float32),
                scheduler.config.num_train_timesteps, momentum_scale, momentum_clamp,
            ).prev_sample.to(dtype=latents.dtype)
        else:
            with torch.no_grad():
                x0 = take_arcflow_step(
                    detach_arcflow_output(output), anchor_timestep,
                    latents, torch.zeros((), device=latents.device, dtype=torch.float32),
                    scheduler.config.num_train_timesteps,
                    momentum_scale, momentum_clamp,
                ).prev_sample.detach()
        if index + 1 < rollout_steps:
            latents = consistency_renoise(
                x0.detach(), scheduler.sigmas[index + 1], renoise_noises[index]
            ).detach()
        else:
            latents = x0

    student_sample_x0 = latents.to(dtype=latents.dtype)
    if selected_output is None:
        raise RuntimeError("No nearest anchor was selected for ArcFlow velocity matching.")

    shared_t = sample_t(
        batch_size=batch_size,
        device=latents.device,
        dtype=torch.float32,
    )
    shared_noise = torch.randn_like(student_sample_x0)
    shared_t_view = shared_t.view(-1, 1, 1, 1).to(
        device=student_sample_x0.device, dtype=student_sample_x0.dtype
    )
    x_t_hat = (
        (1.0 - shared_t_view) * student_sample_x0.detach()
        + shared_t_view * shared_noise
    ).detach()

    if self_velocity_weight > 0.0:
        shared_raw_timestep = shared_t.to(dtype=selected_anchor_latents.dtype) * num_train_timesteps
        self_velocity_target = (shared_noise - student_sample_x0.detach()).to(dtype=latents.dtype)
        enable_student_adapter(transformer_unwrapped)
        if guidance_scale > 1.0:
            student_x_t = torch.cat([x_t_hat] * 2)
            student_timestep = torch.cat([shared_raw_timestep] * 2).to(student_x_t.device)
        else:
            student_x_t = x_t_hat
            student_timestep = shared_raw_timestep.to(student_x_t.device)
        student_output = predict_arcflow_output(
            transformer,
            hidden_states=student_x_t,
            encoder_hidden_states=student_prompt_embeds,
            encoder_attention_mask=student_prompt_mask,
            timestep=student_timestep * transformer_unwrapped.config.timestep_scale,
        )
        if guidance_scale > 1.0:
            student_output = guide_arcflow_output(student_output, guidance_scale)
        student_velocity = arcflow_output_to_instantaneous_velocity(
            student_output, shared_t, momentum_clamp
        )
        self_velocity_loss = F.mse_loss(student_velocity.float(), self_velocity_target.float())
    else:
        self_velocity_loss = torch.zeros((), device=latents.device, dtype=torch.float32)

    if reference_student_sample_x0 is None:
        distribution_preservation_loss = torch.zeros((), device=latents.device, dtype=torch.float32)
    else:
        reference_student_sample_x0 = reference_student_sample_x0.to(
            device=student_sample_x0.device, dtype=student_sample_x0.dtype
        )
        distribution_preservation_loss = F.mse_loss(
            student_sample_x0.float(), reference_student_sample_x0.float()
        )
    dmd_x0 = student_sample_x0

    if dmd_weight > 0.0:
        dmd_raw_timestep = shared_t.to(dtype=selected_anchor_latents.dtype) * num_train_timesteps
        with torch.no_grad():
            disable_lora_adapters(transformer_unwrapped)
            teacher_x_t = torch.cat([x_t_hat] * 2)
            teacher_timestep = torch.cat([dmd_raw_timestep] * 2).to(teacher_x_t.device) * transformer_unwrapped.config.timestep_scale
            teacher_velocity = transformer(
                hidden_states=teacher_x_t, timestep=teacher_timestep,
                encoder_hidden_states=teacher_prompt_embeds,
                encoder_attention_mask=teacher_prompt_mask, return_dict=False,
            )[0]
            teacher_uncond, teacher_text = teacher_velocity.chunk(2)
            teacher_velocity = teacher_uncond + teacher_guidance_scale * (teacher_text - teacher_uncond)
            enable_student_adapter(transformer_unwrapped)
            if guidance_scale > 1.0:
                dmd_student_x_t = torch.cat([x_t_hat] * 2)
                dmd_student_timestep = torch.cat([dmd_raw_timestep] * 2).to(dmd_student_x_t.device)
            else:
                dmd_student_x_t = x_t_hat
                dmd_student_timestep = dmd_raw_timestep.to(dmd_student_x_t.device)
            dmd_output = predict_arcflow_output(
                transformer,
                hidden_states=dmd_student_x_t,
                encoder_hidden_states=student_prompt_embeds,
                encoder_attention_mask=student_prompt_mask,
                timestep=dmd_student_timestep * transformer_unwrapped.config.timestep_scale,
            )
            if guidance_scale > 1.0:
                dmd_output = guide_arcflow_output(dmd_output, guidance_scale)
            dmd_student_velocity = arcflow_output_to_instantaneous_velocity(
                dmd_output, shared_t, momentum_clamp
            )
            dmd_velocity_delta = (teacher_velocity - dmd_student_velocity).detach()
        dmd_loss = (dmd_velocity_delta.float() * dmd_x0.float()).mean()
    else:
        dmd_loss = torch.zeros((), device=latents.device, dtype=torch.float32)
    enable_student_adapter(transformer_unwrapped)

    if trajectory_alignment_weight > 0.0:
        trajectory_alignment_loss = compute_trajectory_alignment_loss(
            transformer=transformer,
            transformer_unwrapped=transformer_unwrapped,
            x_t_hat=x_t_hat,
            policy_timesteps=shared_t,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            uncond_prompt_embeds=uncond_prompt_embeds,
            uncond_prompt_attention_mask=uncond_prompt_attention_mask,
            teacher_guidance_scale=teacher_guidance_scale,
            num_train_timesteps=int(num_train_timesteps),
            ode_step_size=ode_step_size,
            max_integration_length=trajectory_max_integration_length,
            momentum_scale=momentum_scale,
            momentum_clamp=momentum_clamp,
        )
    else:
        trajectory_alignment_loss = torch.zeros((), device=latents.device, dtype=torch.float32)

    loss = (
        self_velocity_weight * self_velocity_loss
        + distribution_preservation_weight * distribution_preservation_loss
        + dmd_weight * dmd_loss
        + trajectory_alignment_weight * trajectory_alignment_loss
    )
    return {
        "loss": loss,
        "self_velocity_loss": self_velocity_loss,
        "distribution_preservation_loss": distribution_preservation_loss,
        "dmd_loss": dmd_loss,
        "trajectory_alignment_loss": trajectory_alignment_loss,
        "student_sample_x0": student_sample_x0.detach(),
        "rollout_steps": rollout_steps,
    }

def sample_t(batch_size, device, dtype=torch.float32):
    return torch.rand(batch_size, device=device, dtype=dtype)

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
    if args.enable_vae_tiling:
        pipeline.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_stride_width=1024)

    pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train a Sana TRD ArcFlow LoRA model from prompt-only TSV data.")
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
        help="Path to a UTF-8 TSV file whose second column contains prompts.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300,
        help="Maximum sequence length to use with with the Gemma model",
    )
    parser.add_argument(
        "--complex_human_instruction",
        type=str,
        default=None,
        help="Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
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
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana-trd-lora",
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
        "--learning_rate_stu",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--ode_step_size",
        type=float,
        default=0.02,
        help="Fixed normalized-time Euler step used for teacher trajectory alignment.",
    )
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
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
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
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only'
        ),
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
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_vae_tiling", action="store_true", help="Enabla vae tiling in log validation")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
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
        "--sample_saving_steps",
        type=int,
        default=10,
        help="Save decoded student_sample_x0 images locally every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--num_arc_heads",
        type=int,
        default=8,
        help="Number of ArcFlow mixture heads used by the student output head.",
    )
    parser.add_argument(
        "--num_arc_head_layers",
        type=int,
        default=0,
        help="Number of final transformer blocks copied into each ArcFlow output head; 0 keeps projection-only heads.",
    )
    parser.add_argument(
        "--gamma_min",
        type=float,
        default=0.4,
        help="Minimum gamma used to initialize the ArcFlow momentum heads.",
    )
    parser.add_argument(
        "--gamma_max",
        type=float,
        default=5.0,
        help="Maximum gamma used to initialize the ArcFlow momentum heads.",
    )
    parser.add_argument(
        "--momentum_head_learning_rate_scale",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for the slower ArcFlow momentum heads.",
    )
    parser.add_argument(
        "--arc_momentum_scale",
        type=float,
        default=1.0,
        help="Scale applied to ArcFlow momentum factors during student generation.",
    )
    parser.add_argument(
        "--arc_momentum_clamp",
        type=float,
        default=20.0,
        help="Clamp range used for ArcFlow momentum integration during student generation.",
    )
    parser.add_argument("--self_velocity_weight", type=float, default=1.0)
    parser.add_argument("--distribution_preservation_weight", type=float, default=1.0)
    parser.add_argument("--dmd_weight", type=float, default=1.0)
    parser.add_argument(
        "--trajectory_alignment_weight",
        type=float,
        default=1.0,
        help="Weight of teacher/student velocity MSE over the teacher ODE trajectory.",
    )
    parser.add_argument(
        "--trajectory_max_integration_length",
        type=float,
        default=0.3,
        help="Maximum normalized-time length of the teacher ODE trajectory.",
    )
    parser.add_argument("--preservation_reference_update_steps", type=int, default=30)
    parser.add_argument("--preservation_reference_ema_decay", type=float, default=0.0)
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale used by the selected validation pipeline.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale used by the student.",
    )
    parser.add_argument(
        "--teacher_guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale used by the frozen teacher.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.sample_saving_steps < 0:
        raise ValueError("`--sample_saving_steps` must be non-negative.")
    if args.validation_steps < 0:
        raise ValueError("`--validation_steps` must be non-negative.")
    if args.checkpointing_steps <= 0:
        raise ValueError("`--checkpointing_steps` must be positive.")
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit <= 0:
        raise ValueError("`--checkpoints_total_limit` must be positive when provided.")
    if args.num_validation_images < 0:
        raise ValueError("`--num_validation_images` must be non-negative.")
    if min(
        args.self_velocity_weight,
        args.distribution_preservation_weight,
        args.dmd_weight,
        args.trajectory_alignment_weight,
    ) < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    if not 0.0 < args.ode_step_size <= 1.0:
        raise ValueError("`--ode_step_size` must be in (0, 1].")
    if not 0.0 < args.trajectory_max_integration_length <= 1.0:
        raise ValueError("`--trajectory_max_integration_length` must be in (0, 1].")
    if args.preservation_reference_update_steps <= 0:
        raise ValueError("`--preservation_reference_update_steps` must be positive.")
    if not 0.0 <= args.preservation_reference_ema_decay < 1.0:
        raise ValueError("`--preservation_reference_ema_decay` must be in [0, 1).")

    if args.num_arc_heads <= 0:
        raise ValueError("`--num_arc_heads` must be a positive integer.")
    if args.num_arc_head_layers < 0:
        raise ValueError("`--num_arc_head_layers` must be non-negative.")
    if not 0.0 < args.gamma_min <= args.gamma_max:
        raise ValueError("`--gamma_min` and `--gamma_max` must satisfy 0 < gamma_min <= gamma_max.")
    if args.momentum_head_learning_rate_scale <= 0.0:
        raise ValueError("`--momentum_head_learning_rate_scale` must be positive.")
    if args.arc_momentum_clamp <= 0.0:
        raise ValueError("`--arc_momentum_clamp` must be positive.")

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
                if len(parts) < 2:
                    continue
                prompt = parts[1]
                if prompt:
                    self.prompts.append(prompt)
        if not self.prompts:
            raise ValueError(f"No prompts were found in TSV file: {path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


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
        Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load scheduler and models
    student_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    transformer = SanaArcFlowTransformer2DModel.from_sana_transformer(
        transformer,
        num_arc_heads=args.num_arc_heads,
        num_arc_head_layers=args.num_arc_head_layers,
    )
    initialize_arcflow_momentum_heads(
        transformer,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
    )
    logger.info(
        "Using ArcFlow pipeline with %s heads and %s copied transformer layer(s) per head",
        args.num_arc_heads,
        args.num_arc_head_layers,
    )

    # Train student LoRA layers plus the ArcFlow output heads.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
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

    # VAE should always be kept in fp32 for SANA (?)
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    # because Gemma2 is particularly suited for bfloat16.
    text_encoder.to(dtype=torch.bfloat16)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            for block in transformer.transformer_blocks:
                block.attn2.set_use_npu_flash_attention(True)
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            name
            for name, module in transformer.named_modules()
            if isinstance(module, nn.Linear)
            and not is_arcflow_output_head(name)
            and name != "proj_out"
        ]

    if not target_modules:
        raise ValueError("No LoRA target modules were found for the non-output-head transformer layers.")

    forbidden_lora_targets = [
        layer for layer in target_modules
        if layer.startswith("arc_") or layer == "proj_out"
    ]
    if forbidden_lora_targets:
        raise ValueError(
            "Output heads are trained as full parameters and must not be LoRA targets: "
            f"{forbidden_lora_targets}"
        )

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    enable_student_adapter(transformer)

    for name, param in transformer.named_parameters():
        if is_arcflow_output_head(name):
            param.requires_grad = True

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Create custom saving and loading hooks for LoRA plus the ArcFlow output heads.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}
            arcflow_output_heads_to_save = None
            for model in models:
                model_to_save = unwrap_model(model)
                if isinstance(model_to_save, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model_to_save, adapter_name="default")
                    arcflow_output_heads_to_save = get_arcflow_output_head_state_dict(model_to_save)
                    modules_to_save["transformer"] = model_to_save
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            SanaPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
            torch.save(arcflow_output_heads_to_save, os.path.join(output_dir, ARCFLOW_OUTPUT_HEADS_FILE))

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()
            model_unwrapped = unwrap_model(model)

            if isinstance(model_unwrapped, type(unwrap_model(transformer))):
                transformer_ = model_unwrapped
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = SanaPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        arcflow_output_heads_path = os.path.join(input_dir, ARCFLOW_OUTPUT_HEADS_FILE)
        if os.path.exists(arcflow_output_heads_path):
            arcflow_output_heads_state_dict = torch.load(arcflow_output_heads_path, map_location="cpu")
            missing_keys, unexpected_keys = transformer_.load_state_dict(arcflow_output_heads_state_dict, strict=False)
            unexpected_keys = [key for key in unexpected_keys if is_arcflow_output_head(key)]
            missing_flow_keys = [key for key in missing_keys if is_arcflow_output_head(key)]
            if unexpected_keys:
                logger.warning("Unexpected ArcFlow output head keys while loading checkpoint: %s", unexpected_keys)
            if missing_flow_keys:
                logger.warning("Missing ArcFlow output head keys while loading checkpoint: %s", missing_flow_keys)
        else:
            logger.warning("No ArcFlow output head checkpoint found at %s", arcflow_output_heads_path)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # Only upcast trainable parameters (LoRA and ArcFlow heads) into fp32.
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate_stu = (
            args.learning_rate_stu * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # Only upcast trainable parameters (LoRA and ArcFlow heads) into fp32.
        cast_training_params(models, dtype=torch.float32)

    student_lora_parameters = [
        p for name, p in transformer.named_parameters()
        if "default" in name and p.requires_grad and not is_arcflow_output_head(name)
    ]
    student_flow_main_parameters = [
        p for name, p in transformer.named_parameters()
        if is_arcflow_output_head(name)
        and not is_slow_arcflow_output_head(name)
        and p.requires_grad
    ]
    student_flow_slow_parameters = [
        p for name, p in transformer.named_parameters()
        if is_slow_arcflow_output_head(name) and p.requires_grad
    ]
    print(f"Number of trainable parameters in student LoRA: {sum(p.numel() for p in student_lora_parameters)}")
    print(f"Number of trainable parameters in ArcFlow main heads: {sum(p.numel() for p in student_flow_main_parameters)}")
    print(f"Number of trainable parameters in ArcFlow slow heads: {sum(p.numel() for p in student_flow_slow_parameters)}")
    params_to_optimize_stu = [
        {"params": student_lora_parameters, "lr": args.learning_rate_stu},
        {"params": student_flow_main_parameters, "lr": args.learning_rate_stu},
        {
            "params": student_flow_slow_parameters,
            "lr": args.learning_rate_stu * args.momentum_head_learning_rate_scale,
        },
    ]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer_stu = optimizer_class(
            params_to_optimize_stu,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate_stu <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer_stu = optimizer_class(
            params_to_optimize_stu,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    train_dataset = PromptTSVDataset(args.prompt_database)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                prompt,
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=args.complex_human_instruction,
            )
        prompt_embeds = prompt_embeds.to(accelerator.unwrap_model(transformer).dtype)

        uncond_prompts = [""] * (len(prompt) if isinstance(prompt, (list, tuple)) else prompt.shape[0])
        with torch.no_grad():
            uncond_prompt_embeds, uncond_prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                uncond_prompts,
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=args.complex_human_instruction,
            )
        if args.offload:
            text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        uncond_prompt_embeds = uncond_prompt_embeds.to(accelerator.unwrap_model(transformer).dtype)
        return prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_stu = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_stu,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer_stu, train_dataloader, lr_scheduler_stu = accelerator.prepare(
        transformer, optimizer_stu, train_dataloader, lr_scheduler_stu
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
        tracker_name = "sana-trd-lora"
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

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            candidates = []
            for checkpoint_path in Path(args.output_dir).glob("checkpoint-*"):
                try:
                    step_number = int(checkpoint_path.name.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                if checkpoint_path.is_dir():
                    candidates.append((step_number, checkpoint_path))
            resume_path = max(candidates, default=(None, None))[1]
        else:
            resume_path = Path(args.resume_from_checkpoint).expanduser()
            if not resume_path.is_absolute():
                resume_path = Path(args.output_dir) / resume_path
            resume_path = resume_path.resolve()

        if resume_path is None or not resume_path.is_dir():
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            try:
                global_step = int(resume_path.name.split("-", 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Checkpoint directory must be named checkpoint-N, got: {resume_path}"
                ) from error
            accelerator.print(f"Resuming from checkpoint {resume_path}")
            accelerator.load_state(str(resume_path))

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    preservation_reference_state = refresh_trainable_parameter_state_dict_(
        accelerator.unwrap_model(transformer)
    )
    logger.info(
        "Initialized old student trainable-weight snapshot; refreshing every %s optimizer steps with EMA decay %s.",
        args.preservation_reference_update_steps,
        args.preservation_reference_ema_decay,
    )

    if args.resume_from_checkpoint is None:
        initial_checkpoint_path = os.path.join(args.output_dir, "checkpoint-0")
        if accelerator.is_main_process and os.path.exists(initial_checkpoint_path):
            incomplete_checkpoint_path = os.path.join(
                args.output_dir, f".checkpoint-0.incomplete-{os.getpid()}"
            )
            os.replace(initial_checkpoint_path, incomplete_checkpoint_path)
            logger.warning(
                "Moved the existing checkpoint-0 to %s before recreating the initial checkpoint.",
                incomplete_checkpoint_path,
            )
        accelerator.wait_for_everyone()
        accelerator.save_state(initial_checkpoint_path)
        logger.info("Saved initial state to %s", initial_checkpoint_path)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    loss_stu = torch.tensor(0.0, device=accelerator.device)
    self_velocity_loss = torch.tensor(0.0, device=accelerator.device)
    distribution_preservation_loss = torch.tensor(0.0, device=accelerator.device)
    dmd_loss = torch.tensor(0.0, device=accelerator.device)
    trajectory_alignment_loss = torch.tensor(0.0, device=accelerator.device)
    student_sample_x0 = None
    rollout_steps = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask = compute_text_embeddings(prompts, text_encoding_pipeline)
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=weight_dtype)
                prompt_attention_mask = prompt_attention_mask.to(dtype=weight_dtype)
                uncond_prompt_attention_mask = uncond_prompt_attention_mask.to(dtype=weight_dtype)

                noise_shape = (
                    len(prompts),
                    accelerator.unwrap_model(transformer).config.in_channels,
                    args.val_resolution // (2 ** (len(vae.config.encoder_block_out_channels) - 1)),
                    args.val_resolution // (2 ** (len(vae.config.encoder_block_out_channels) - 1)),
                )
                noise = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype)
                loss_outputs = compute_student_random_t_anchor_loss(
                    accelerator=accelerator,
                    transformer=transformer,
                    scheduler=student_scheduler,
                    latents=noise,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    uncond_prompt_embeds=uncond_prompt_embeds,
                    uncond_prompt_attention_mask=uncond_prompt_attention_mask,
                    guidance_scale=args.guidance_scale,
                    teacher_guidance_scale=args.teacher_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    momentum_scale=args.arc_momentum_scale,
                    momentum_clamp=args.arc_momentum_clamp,
                    self_velocity_weight=args.self_velocity_weight,
                    distribution_preservation_weight=args.distribution_preservation_weight,
                    dmd_weight=args.dmd_weight,
                    trajectory_alignment_weight=args.trajectory_alignment_weight,
                    ode_step_size=args.ode_step_size,
                    trajectory_max_integration_length=args.trajectory_max_integration_length,
                    reference_trainable_state=preservation_reference_state,
                )
                loss_stu = loss_outputs["loss"]
                self_velocity_loss = loss_outputs["self_velocity_loss"]
                distribution_preservation_loss = loss_outputs["distribution_preservation_loss"]
                dmd_loss = loss_outputs["dmd_loss"]
                trajectory_alignment_loss = loss_outputs["trajectory_alignment_loss"]
                student_sample_x0 = loss_outputs["student_sample_x0"]
                rollout_steps = loss_outputs["rollout_steps"]

                accelerator.backward(loss_stu)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer_stu.step()
                lr_scheduler_stu.step()
                optimizer_stu.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.preservation_reference_update_steps == 0:
                    preservation_reference_state = refresh_trainable_parameter_state_dict_(
                        accelerator.unwrap_model(transformer),
                        preservation_reference_state,
                        ema_decay=args.preservation_reference_ema_decay,
                    )

                if accelerator.is_main_process:
                    if args.sample_saving_steps > 0 and global_step % args.sample_saving_steps == 0:
                        save_decoded_student_samples(
                            student_sample_x0, vae, args.output_dir, global_step
                        )

                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints
                                if d.startswith("checkpoint-") and d != "checkpoint-0"
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        args.validation_prompt is not None
                        and args.validation_steps > 0
                        and global_step % args.validation_steps == 0
                    ):
                        # create pipeline
                        enable_student_adapter(accelerator.unwrap_model(transformer))
                        pipeline = SanaArcFlowPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=accelerator.unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=torch.float32,
                        )
                        pipeline_args = {"prompt": args.validation_prompt, **validation_pipeline_args(args)}
                        log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                        )
                        free_memory()

                        del pipeline

            logs = {
                "loss_stu": loss_stu.detach().item(),
                "self_velocity_loss": self_velocity_loss.detach().item(),
                "distribution_preservation_loss": distribution_preservation_loss.detach().item(),
                "dmd_loss": dmd_loss.detach().item(),
                "trajectory_alignment_loss": trajectory_alignment_loss.detach().item(),
                "rollout_steps": rollout_steps,
                "lr": lr_scheduler_stu.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        modules_to_save = {}
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer, adapter_name="default")
        modules_to_save["transformer"] = transformer

        SanaPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
        torch.save(
            get_arcflow_output_head_state_dict(transformer),
            os.path.join(args.output_dir, ARCFLOW_OUTPUT_HEADS_FILE),
        )

        # Final inference with the selected in-memory flow student.
        enable_student_adapter(transformer)
        pipeline = SanaArcFlowPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=torch.float32,
        )

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {
                "prompt": args.validation_prompt,
                "complex_human_instruction": args.complex_human_instruction,
                **validation_pipeline_args(args),
            }
            log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
            )

        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
