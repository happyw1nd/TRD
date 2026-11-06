#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Paths
MODEL_PATH="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
PROMPT_DATABASE="datasets/labels_cleaned_train.tsv"
OUTPUT_DIR="outputs/sana-trd"
LOGGING_DIR="logs"

accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --main_process_port=29500 \
  "${PROJECT_ROOT}/scripts/train_sana_trd.py" \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --variant bf16 \
  --prompt_database "${PROMPT_DATABASE}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --max_sequence_length 300 \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 4 \
  --num_train_epochs 1 \
  --max_train_steps 5000 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 3 \
  --sample_saving_steps 100 \
  --rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.0 \
  --optimizer AdamW \
  --learning_rate_stu 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-4 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --self_velocity_weight 1.0 \
  --distribution_preservation_weight 1.0 \
  --dmd_weight 10.0 \
  --trajectory_alignment_weight 0.5 \
  --ode_step_size 0.05 \
  --trajectory_max_integration_length 0.3 \
  --preservation_reference_update_steps 1 \
  --preservation_reference_ema_decay 0.9 \
  --num_arc_heads 4 \
  --num_arc_head_layers 1 \
  --gamma_min 0.4 \
  --gamma_max 5.0 \
  --momentum_head_learning_rate_scale 0.1 \
  --arc_momentum_scale 1.0 \
  --arc_momentum_clamp 20.0 \
  --num_inference_steps 1 \
  --guidance_scale 1.0 \
  --teacher_guidance_scale 4.5 \
  --validation_prompt "a bear riding a skateboard, 4k, high resolution" \
  --validation_steps 500 \
  --num_validation_images 4 \
  --val_resolution 1024 \
  --validation_guidance_scale 0.0 \
  --report_to tensorboard \
  --seed 42 \
  --use_8bit_adam \
  --allow_tf32 \
  "$@"