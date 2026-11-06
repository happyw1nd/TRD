#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Paths
MODEL_PATH="stabilityai/stable-diffusion-3.5-medium"
PROMPT_DATABASE="datasets/labels_cleaned_test_10k.tsv"
CHECKPOINT_DIR="outputs/sd3-trd"
SAVE_DIR="${CHECKPOINT_DIR}/samples"
PROMPT_MAP_PATH="${SAVE_DIR}/prompts.json"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=39503 \
  "${PROJECT_ROOT}/scripts/infer_sd3_trd.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --prompt_database "${PROMPT_DATABASE}" \
  --save_dir "${SAVE_DIR}" \
  --prompt_map_path "${PROMPT_MAP_PATH}" \
  --num_gen_pics 2500 \
  --num_images_per_prompt 4 \
  --batch_size 4 \
  --num_workers 2 \
  --height 1024 \
  --width 1024 \
  --max_sequence_length 256 \
  --mixed_precision= bf16 \
  --guidance_scale 1.0 \
  --num_inference_steps 2 \
  --denoising_timesteps "1000,780" \
  --num_arc_heads 4 \
  --num_arc_head_layers 1 \
  --momentum_scale 1.0 \
  --momentum_clamp 20.0 \
  --consistency_sampling \
  --seed 42 \
  --dataset_seed 666 \
  "$@"
