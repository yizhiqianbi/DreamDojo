#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$ROOT/../openpi}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/inference_results/pi05_parallel_worlds_48}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

python "$ROOT/integrations/pi05/prepare_jokeru_lerobot.py"
python "$ROOT/integrations/pi05/compute_norm_stats.py"

uv run --project "$OPENPI_ROOT" python "$ROOT/integrations/pi05/run_openpi.py" \
  --openpi-root "$OPENPI_ROOT" \
  "$ROOT/integrations/pi05/sample_pi05_actions.py" \
  --checkpoint "$PI05_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

source "$ROOT/.venv/bin/activate"
python "$ROOT/integrations/pi05/generate_parallel_worlds.py" \
  --manifest "$OUTPUT_DIR/manifest.json" \
  --overwrite
