#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

CHECKPOINTS_DIR="$ROOT/outputs/dreamdojo/jokeru_posttrain/jokeru_2b_8gpu_posttrain_20260829/checkpoints"
CHECKPOINT_PATH="$CHECKPOINTS_DIR/iter_000003000/model_ema_bf16.pt"
RUNTIME_RESULTS="$ROOT/results/jokeru_iter3000_long_horizon_61f"
VERSIONED_RESULTS="$ROOT/inference_results/iter_000003000/long_horizon_61f"
DATASETS="datasets/jokeru/continuous_shelf_organizing,datasets/jokeru/arrange_orange_juice_and_green_tea_3,datasets/jokeru/pick_right_purple_box_and_place_it_in_the_middle"

source "$ROOT/.venv/bin/activate"

python examples/action_conditioned.py \
  -o outputs/inference_runtime/jokeru_iter3000_long_horizon_61f \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --experiment dreamdojo_2b_480_640_jokeru \
  --save-dir "$RUNTIME_RESULTS" \
  --num-frames 61 \
  --num-samples 3 \
  --dataset-path "$DATASETS" \
  --data-split test \
  --single-base-index \
  --deterministic-uniform-sampling

mkdir -p "$VERSIONED_RESULTS"
cp -a "$RUNTIME_RESULTS/iter_000003000/." "$VERSIONED_RESULTS/"
