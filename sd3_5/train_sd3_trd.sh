#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Paths
MODEL_PATH="stabilityai/stable-diffusion-3.5-medium"
PROMPT_DATABASE="datasets/labels_cleaned_train.tsv"
OUTPUT_DIR="outputs/sd3-trd"
LOGGING_DIR="logs"

accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --main_process_port=39502 \
  "${PROJECT_ROOT}/scripts/train_sd3_trd.py" \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --prompt_database "${PROMPT_DATABASE}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --mixed_precision= bf16 \
  --report_to tensorboard \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 0 \
  --max_train_steps 10000 \
  --checkpointing_steps 500 \
  --sample_saving_steps 10 \
  --max_grad_norm 1.0 \
  --rank 32 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --optimizer AdamW \
  --student_learning_rate 1e-4 \
  --gate_learning_rate 1e-4 \
  --gamma_learning_rate 1e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-4 \
  --adam_epsilon 1e-8 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --self_velocity_weight 1.0 \
  --distribution_preservation_weight 1.0 \
  --dmd_weight 30.0 \
  --trajectory_alignment_weight 0.1 \
  --ode_step_size 0.025 \
  --trajectory_max_integration_length 0.2 \
  --preservation_reference_update_steps 1 \
  --preservation_reference_ema_decay 0.9 \
  --num_arc_heads 4 \
  --num_arc_head_layers 2 \
  --gamma_min 0.5 \
  --gamma_max 1.0 \
  --arc_momentum_scale 1.0 \
  --arc_momentum_clamp 20.0 \
  --student_guidance_scale 1.0 \
  --teacher_guidance_scale 4.5 \
  --num_inference_steps 2 \
  --denoising_timesteps "1000,600" \
  --timestep_sampling_clip_range "950,50" \
  --validation_prompt "a bear riding a skateboard, 4k, high resolution" \
  --validation_steps 200 \
  --num_validation_images 4 \
  --validation_guidance_scale 1.0 \
  --val_resolution 1024 \
  --seed 42 \
  --use_8bit_adam \
  "$@"
