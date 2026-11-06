#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Paths
METRICS_MODEL_ROOT="${PROJECT_ROOT}/models"
TEACHER_SAMPLES_DIR="${PROJECT_ROOT}/samples/teacher"
STUDENT_SAMPLES_DIR="${PROJECT_ROOT}/samples/student"
PROMPT_JSON_PATH="${STUDENT_SAMPLES_DIR}/prompts.json"

export HF_HOME="${METRICS_MODEL_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${METRICS_MODEL_ROOT}/torch"
export HPS_ROOT="${METRICS_MODEL_ROOT}/hps"
export XDG_CACHE_HOME="${METRICS_MODEL_ROOT}/xdg_cache"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  "${PROJECT_ROOT}/calculate_metrics.py" \
  --teacher_samples_dir "${TEACHER_SAMPLES_DIR}" \
  --student_samples_dir "${STUDENT_SAMPLES_DIR}" \
  --prompt_json_path "${PROMPT_JSON_PATH}" \
  --aesthetic_checkpoint "${METRICS_MODEL_ROOT}/aesthetic/sac+logos+ava1-l14-linearMSE.pth" \
  --clip_cache_dir "${METRICS_MODEL_ROOT}/openai_clip" \
  --image_reward_dir "${METRICS_MODEL_ROOT}/image_reward" \
  --dinov2_repo "${METRICS_MODEL_ROOT}/dinov2/repo" \
  --batch_size 16 \
  --fid_batch_size 32 \
  --dinov2_recall_k 3 \
  --dinov2_distance_chunk_size 1024 \
  --num_workers 4 \
  "$@"
