#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PHASE="${1:-status}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-12431}"
TEACHER_EXPERIMENT="${TEACHER_EXPERIMENT:-dreamdojo_2b_480_640_jokeru}"
TEACHER_ITER="${TEACHER_ITER:-$ROOT/outputs/dreamdojo/jokeru_posttrain/jokeru_2b_8gpu_posttrain_20260829/checkpoints/iter_000003000}"
TEACHER_MODEL="${TEACHER_MODEL:-$TEACHER_ITER/model}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/outputs/self_forcing/jokeru/teacher_cache}"
WARMUP_OUTPUT="${WARMUP_OUTPUT:-$ROOT/outputs/self_forcing/jokeru/warmup}"
WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-$WARMUP_OUTPUT/checkpoints/iter_000020000}"
SELF_FORCING_OUTPUT="${SELF_FORCING_OUTPUT:-$ROOT/outputs/self_forcing/jokeru/student}"
SELF_FORCING_CHECKPOINT="${SELF_FORCING_CHECKPOINT:-$SELF_FORCING_OUTPUT/checkpoints/iter_000003000}"
TEACHER_START="${TEACHER_START:-0}"
TEACHER_END="${TEACHER_END:-10000}"
WARMUP_ITERS="${WARMUP_ITERS:-20000}"
SELF_FORCING_ITERS="${SELF_FORCING_ITERS:-3000}"
SAVE_ITER="${SAVE_ITER:-1000}"
INPUT_JSON="${INPUT_JSON:-$ROOT/outputs/self_forcing/jokeru/eval/pi05_parallel_x8.json}"
VAE_PATH="${VAE_PATH:-$ROOT/checkpoints/Wan2.1/Wan2.1_VAE.pth}"
CR1_EMBEDDING="${CR1_EMBEDDING:-$ROOT/datasets/cr1_empty_string_text_embeddings.pt}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_API_KEY="${WANDB_API_KEY:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export CUDA_MODULE_LOADING=LAZY

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

require_local_assets() {
  if [[ ! -f "$VAE_PATH" ]]; then
    echo "Wan2.1 VAE missing: $VAE_PATH" >&2
    echo "Run: python scripts/download_jokeru_assets.py --skip-datasets" >&2
    exit 2
  fi
  if [[ ! -f "$CR1_EMBEDDING" ]]; then
    echo "CR1 empty-prompt embedding missing: $CR1_EMBEDDING" >&2
    echo "Run: python scripts/download_jokeru_assets.py --skip-datasets" >&2
    exit 2
  fi
}

discover_checkpoint() {
  local requested="$1"
  local search_root="$2"
  local metadata=""
  if [[ -f "$requested/model/.metadata" ]]; then
    printf '%s\n' "$requested"
    return
  fi
  metadata="$(find "$search_root" -type f -path '*/checkpoints/iter_*/model/.metadata' 2>/dev/null | sort | tail -n 1)"
  if [[ -n "$metadata" ]]; then
    dirname "$(dirname "$metadata")"
    return
  fi
  printf '%s\n' "$requested"
}

teacher_cache() {
  mkdir -p "$CACHE_ROOT"
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m cosmos_predict2._src.predict2.action.inference.inference_gr00t_warmup \
    -- \
    --experiment="$TEACHER_EXPERIMENT" \
    --ckpt_path="$TEACHER_MODEL" \
    --save_root="$CACHE_ROOT" \
    --guidance=0 \
    --chunk_size=12 \
    --start="$TEACHER_START" \
    --end="$TEACHER_END" \
    --query_steps=0,9,18,27,34
}

warmup() {
  require_local_assets
  mkdir -p "$WARMUP_OUTPUT"
  IMAGINAIRE_OUTPUT_ROOT="$WARMUP_OUTPUT" torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
    -- experiment=dreamdojo_2b_action_jokeru_warmup_no_s3 \
    "checkpoint.load_path=$TEACHER_ITER" \
    "+model.config.tokenizer.vae_pth=$VAE_PATH" \
    "dataloader_train.num_workers=0" \
    "dataloader_val.num_workers=0" \
    "job.wandb_mode=disabled" \
    "trainer.max_iter=$WARMUP_ITERS" \
    "checkpoint.save_iter=$SAVE_ITER"
}

self_forcing() {
  require_local_assets
  WARMUP_CHECKPOINT="$(discover_checkpoint "$WARMUP_CHECKPOINT" "$WARMUP_OUTPUT")"
  if [[ ! -f "$WARMUP_CHECKPOINT/model/.metadata" ]]; then
    echo "Warmup checkpoint missing: $WARMUP_CHECKPOINT/model/.metadata" >&2
    exit 2
  fi
  mkdir -p "$SELF_FORCING_OUTPUT"
  IMAGINAIRE_OUTPUT_ROOT="$SELF_FORCING_OUTPUT" torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
    -- experiment=dreamdojo_2b_action_jokeru_self_forcing_no_s3 \
    "checkpoint.load_path=$WARMUP_CHECKPOINT" \
    "model.config.teacher_load_from.load_path=$WARMUP_CHECKPOINT/model" \
    "model.config.teacher_load_from.credentials=" \
    "+model.config.tokenizer.vae_pth=$VAE_PATH" \
    "job.wandb_mode=disabled" \
    "trainer.max_iter=$SELF_FORCING_ITERS" \
    "checkpoint.save_iter=$SAVE_ITER"
}

student_inference() {
  SELF_FORCING_CHECKPOINT="$(discover_checkpoint "$SELF_FORCING_CHECKPOINT" "$SELF_FORCING_OUTPUT")"
  if [[ ! -f "$SELF_FORCING_CHECKPOINT/model/.metadata" ]]; then
    echo "Self-Forcing checkpoint missing: $SELF_FORCING_CHECKPOINT/model/.metadata" >&2
    exit 2
  fi
  python integrations/self_forcing/prepare_inference_json.py \
    --output "$INPUT_JSON"
  torchrun --nproc_per_node="${INFERENCE_NPROC:-1}" --master_port="$MASTER_PORT" \
    -m cosmos_predict2._src.predict2.interactive.inference.action_video2world \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
    --experiment=dreamdojo_2b_action_jokeru_self_forcing_no_s3 \
    --ckpt_path="$SELF_FORCING_CHECKPOINT" \
    --input_json="$INPUT_JSON" \
    --fps=7.25 \
    --max_frames=49 \
    --num_steps=4 \
    --torch_compile
}

case "$PHASE" in
  status)
    python integrations/self_forcing/check_readiness.py
    ;;
  teacher-cache)
    teacher_cache
    ;;
  warmup)
    warmup
    ;;
  self-forcing)
    self_forcing
    ;;
  inference)
    student_inference
    ;;
  all)
    teacher_cache
    warmup
    self_forcing
    student_inference
    ;;
  *)
    echo "Usage: $0 {status|teacher-cache|warmup|self-forcing|inference|all}" >&2
    exit 2
    ;;
esac
