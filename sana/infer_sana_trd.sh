#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Paths
MODEL_NAME="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
PROMPT_DATABASE="datasets/labels_cleaned_test_10k.tsv"
CHECKPOINT_DIR="outputs/sana-trd/checkpoint-1000"
SAVE_DIR="${CHECKPOINT_DIR}/samples"
PROMPT_MAP_PATH="${SAVE_DIR}/prompts.json"

torchrun \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  "${PROJECT_ROOT}/scripts/infer_sana_trd.py" \
  --model_name_or_path "${MODEL_NAME}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --prompt_database "${PROMPT_DATABASE}" \
  --save_dir "${SAVE_DIR}" \
  --prompt_map_path "${PROMPT_MAP_PATH}" \
  --variant bf16 \
  --dtype bfloat16 \
  --num_gen_pics 2500 \
  --num_images_per_prompt 4 \
  --batch_size 4 \
  --num_workers 2 \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 1 \
  --guidance_scale 0.0 \
  --seed 42 \
  --dataset_seed 666 \
  --num_arc_heads 4 \
  --num_arc_head_layers 1 \
  --momentum_scale 1.0 \
  --momentum_clamp 20.0 \
  --consistency_sampling \
  "$@"
