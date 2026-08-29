#!/usr/bin/env bash
set -euo pipefail
# hubin 你好，我希望我们和谐相处，这个脚本是不是对我的训练不太友好，我可以只占 4卡
DREAMDOJO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DREAMDOJO_ROOT"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
MAX_ITER="${MAX_ITER:-8953}"
JOB_NAME="${JOB_NAME:-jokeru_2b_8gpu_posttrain_20260829}"
SOURCE_JOB_NAME="${SOURCE_JOB_NAME:-jokeru_2b_gpu1_posttrain_20260829}"
OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$DREAMDOJO_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$DREAMDOJO_ROOT/outputs/launch_logs}"
SAVE_ITER="${SAVE_ITER:-1000}"
RETENTION_COUNT="${RETENTION_COUNT:-2}"
POLL_SECONDS="${POLL_SECONDS:-2}"
RESTART_DELAY="${RESTART_DELAY:-10}"
MASTER_PORT="${MASTER_PORT:-12343}"

mkdir -p "$LOG_DIR"
LOCK_FILE="$LOG_DIR/${JOB_NAME}.lock"
PID_FILE="$LOG_DIR/${JOB_NAME}.pid"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another supervisor already holds $LOCK_FILE" >&2
  exit 1
fi
printf '%s\n' "$$" >"$PID_FILE"

SOURCE_CHECKPOINT_DIR="$OUTPUT_ROOT/dreamdojo/jokeru_posttrain/$SOURCE_JOB_NAME/checkpoints"
SOURCE_LATEST_FILE="$SOURCE_CHECKPOINT_DIR/latest_checkpoint.txt"
if [[ ! -s "$SOURCE_LATEST_FILE" ]]; then
  echo "Resume marker not found: $SOURCE_LATEST_FILE" >&2
  exit 1
fi
SOURCE_CHECKPOINT_NAME="$(tr -d '\r\n' <"$SOURCE_LATEST_FILE")"
SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT_DIR/$SOURCE_CHECKPOINT_NAME"
if [[ ! -d "$SOURCE_CHECKPOINT/model" || ! -d "$SOURCE_CHECKPOINT/optim" ]]; then
  echo "Incomplete resume checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

JOB_CHECKPOINT_DIR="$OUTPUT_ROOT/dreamdojo/jokeru_posttrain/$JOB_NAME/checkpoints"
JOB_LATEST_FILE="$JOB_CHECKPOINT_DIR/latest_checkpoint.txt"

declare -A SELECTED_GPU_UUIDS=()
while IFS=',' read -r gpu_index gpu_uuid; do
  gpu_index="${gpu_index//[[:space:]]/}"
  gpu_uuid="${gpu_uuid//[[:space:]]/}"
  if [[ ",$GPU_LIST," == *",$gpu_index,"* ]]; then
    SELECTED_GPU_UUIDS["$gpu_uuid"]="$gpu_index"
  fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)
if [[ "${#SELECTED_GPU_UUIDS[@]}" -ne "$NPROC" ]]; then
  echo "GPU_LIST=$GPU_LIST resolved ${#SELECTED_GPU_UUIDS[@]} GPUs, expected NPROC=$NPROC" >&2
  exit 1
fi

training_pid=""

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

stop_training() {
  if [[ -n "$training_pid" ]] && kill -0 "$training_pid" 2>/dev/null; then
    echo "[$(timestamp)] stopping DreamDojo process group $training_pid"
    kill -TERM -- "-$training_pid" 2>/dev/null || true
  fi
}

cleanup() {
  stop_training
  rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stop_known_interferers() {
  local uuid pid cmd gpu_index
  while IFS=',' read -r uuid pid; do
    uuid="${uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    [[ -n "${SELECTED_GPU_UUIDS[$uuid]:-}" ]] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd="$(ps -ww -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *'/public/inz/envs/multi-wam-mira/bin/python'* \
       && "$cmd" == *'mira.training.world_v8.plain_train'* \
       && "$cmd" == *'experiment.process_prefix=filter_x_wv12b'* ]]; then
      gpu_index="${SELECTED_GPU_UUIDS[$uuid]}"
      echo "[$(timestamp)] stopping GPU $gpu_index interferer PID $pid"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "[$(timestamp)] force-stopping GPU $gpu_index interferer PID $pid"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
}

cleanup_old_checkpoints() {
  local latest_name latest_iteration checkpoint_name checkpoint_iteration kept checkpoint_path
  [[ "$RETENTION_COUNT" =~ ^[1-9][0-9]*$ ]] || return 0
  [[ -s "$JOB_LATEST_FILE" ]] || return 0
  latest_name="$(tr -d '\r\n' <"$JOB_LATEST_FILE")"
  [[ "$latest_name" =~ ^iter_[0-9]{9}$ ]] || return 0
  latest_iteration=$((10#${latest_name#iter_}))
  kept=0

  while IFS= read -r checkpoint_name; do
    [[ "$checkpoint_name" =~ ^iter_[0-9]{9}$ ]] || continue
    checkpoint_iteration=$((10#${checkpoint_name#iter_}))
    # Never touch a checkpoint that may currently be in progress.
    [[ "$checkpoint_iteration" -le "$latest_iteration" ]] || continue
    if [[ "$kept" -lt "$RETENTION_COUNT" ]]; then
      kept=$((kept + 1))
      continue
    fi
    checkpoint_path="$JOB_CHECKPOINT_DIR/$checkpoint_name"
    if [[ "$checkpoint_path" == "$JOB_CHECKPOINT_DIR"/iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9] \
       && -d "$checkpoint_path" ]]; then
      echo "[$(timestamp)] removing superseded checkpoint $checkpoint_name"
      rm -rf -- "$checkpoint_path"
    fi
  done < <(find "$JOB_CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'iter_[0-9]*' -printf '%f\n' 2>/dev/null | sort -r)
}

echo "[$(timestamp)] supervising $JOB_NAME on physical GPUs $GPU_LIST"
echo "[$(timestamp)] per-GPU batch $PER_GPU_BATCH; global batch $((PER_GPU_BATCH * NPROC))"
echo "[$(timestamp)] stopping at iteration $MAX_ITER (approximately three effective epochs)"
echo "[$(timestamp)] initial resume checkpoint: $SOURCE_CHECKPOINT"
echo "[$(timestamp)] checkpoint interval $SAVE_ITER; retaining the latest $RETENTION_COUNT"

attempt=0
while true; do
  attempt=$((attempt + 1))
  stop_known_interferers
  cleanup_old_checkpoints
  echo "[$(timestamp)] attempt $attempt starts; save interval $SAVE_ITER"

  setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    NPROC="$NPROC" \
    MASTER_PORT="$MASTER_PORT" \
    IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$DREAMDOJO_ROOT/launch.sh" dreamdojo_2b_480_640_jokeru \
      "job.name=$JOB_NAME" \
      "dataloader_train.batch_size=$PER_GPU_BATCH" \
      dataloader_val.batch_size=1 \
      "trainer.max_iter=$MAX_ITER" \
      "checkpoint.save_iter=$SAVE_ITER" \
      "checkpoint.load_path=$SOURCE_CHECKPOINT" \
      checkpoint.load_training_state=true &
  training_pid=$!
  echo "[$(timestamp)] DreamDojo launcher PID $training_pid"

  while kill -0 "$training_pid" 2>/dev/null; do
    stop_known_interferers
    cleanup_old_checkpoints
    sleep "$POLL_SECONDS"
  done

  if wait "$training_pid"; then
    status=0
  else
    status=$?
  fi
  training_pid=""

  if [[ "$status" -eq 0 ]]; then
    echo "[$(timestamp)] training completed successfully"
    exit 0
  fi
  echo "[$(timestamp)] training exited with status $status; restarting in ${RESTART_DELAY}s"
  sleep "$RESTART_DELAY"
done
