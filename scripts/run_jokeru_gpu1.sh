#!/usr/bin/env bash
set -euo pipefail

DREAMDOJO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DREAMDOJO_ROOT"

GPU_INDEX="${GPU_INDEX:-1}"
JOB_NAME="${JOB_NAME:-jokeru_2b_gpu1_posttrain_20260829}"
OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$DREAMDOJO_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$DREAMDOJO_ROOT/outputs/launch_logs}"
BOOTSTRAP_SAVE_ITER="${BOOTSTRAP_SAVE_ITER:-10}"
SAVE_ITER="${SAVE_ITER:-1000}"
POLL_SECONDS="${POLL_SECONDS:-2}"
RESTART_DELAY="${RESTART_DELAY:-5}"
MASTER_PORT="${MASTER_PORT:-12342}"

mkdir -p "$LOG_DIR"
LOCK_FILE="$LOG_DIR/${JOB_NAME}.lock"
PID_FILE="$LOG_DIR/${JOB_NAME}.pid"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another supervisor already holds $LOCK_FILE" >&2
  exit 1
fi
printf '%s\n' "$$" >"$PID_FILE"

CHECKPOINT_DIR="$OUTPUT_ROOT/dreamdojo/jokeru_posttrain/$JOB_NAME/checkpoints"
LATEST_CHECKPOINT="$CHECKPOINT_DIR/latest_checkpoint.txt"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', *' -v target_index="$GPU_INDEX" '$1 == target_index { print $2 }')"
if [[ -z "$GPU_UUID" ]]; then
  echo "Physical GPU $GPU_INDEX was not found" >&2
  exit 1
fi

training_pid=""
bootstrap_restart=0

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

stop_gpu1_interferers() {
  local uuid pid cmd
  while IFS=',' read -r uuid pid; do
    uuid="${uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    [[ "$uuid" == "$GPU_UUID" ]] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd="$(ps -ww -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *'/public/inz/envs/multi-wam-mira/bin/python'* \
       && "$cmd" == *'mira.training.world_v8.plain_train'* \
       && "$cmd" == *'experiment.process_prefix=filter_x_wv12b'* ]]; then
      echo "[$(timestamp)] stopping GPU $GPU_INDEX interferer PID $pid"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "[$(timestamp)] force-stopping GPU $GPU_INDEX interferer PID $pid"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
}

echo "[$(timestamp)] supervising $JOB_NAME on physical GPU $GPU_INDEX ($GPU_UUID)"
echo "[$(timestamp)] checkpoints: $CHECKPOINT_DIR"

attempt=0
while true; do
  attempt=$((attempt + 1))
  stop_gpu1_interferers

  if [[ -s "$LATEST_CHECKPOINT" ]]; then
    checkpoint_name="$(tr -d '\r\n' <"$LATEST_CHECKPOINT")"
    save_iter="$SAVE_ITER"
    echo "[$(timestamp)] attempt $attempt resumes $checkpoint_name; save interval $save_iter"
  else
    checkpoint_name=""
    save_iter="$BOOTSTRAP_SAVE_ITER"
    echo "[$(timestamp)] attempt $attempt starts from the 2B base; bootstrap save interval $save_iter"
  fi

  setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    NPROC=1 \
    MASTER_PORT="$MASTER_PORT" \
    IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$DREAMDOJO_ROOT/launch.sh" dreamdojo_2b_480_640_jokeru \
      "job.name=$JOB_NAME" \
      dataloader_train.batch_size=1 \
      dataloader_val.batch_size=1 \
      "checkpoint.save_iter=$save_iter" &
  training_pid=$!
  echo "[$(timestamp)] DreamDojo launcher PID $training_pid"

  bootstrap_restart=0
  while kill -0 "$training_pid" 2>/dev/null; do
    stop_gpu1_interferers
    if [[ -z "$checkpoint_name" && -s "$LATEST_CHECKPOINT" ]]; then
      checkpoint_name="$(tr -d '\r\n' <"$LATEST_CHECKPOINT")"
      echo "[$(timestamp)] bootstrap checkpoint $checkpoint_name is complete; restarting with save interval $SAVE_ITER"
      bootstrap_restart=1
      stop_training
      break
    fi
    sleep "$POLL_SECONDS"
  done

  if wait "$training_pid"; then
    status=0
  else
    status=$?
  fi
  training_pid=""

  if [[ "$bootstrap_restart" -eq 1 ]]; then
    echo "[$(timestamp)] bootstrap restart complete; waiting ${RESTART_DELAY}s for the rendezvous port"
    sleep "$RESTART_DELAY"
  elif [[ "$status" -eq 0 ]]; then
    echo "[$(timestamp)] training completed successfully"
    exit 0
  else
    echo "[$(timestamp)] training exited with status $status; restarting in ${RESTART_DELAY}s"
    sleep "$RESTART_DELAY"
  fi
done
