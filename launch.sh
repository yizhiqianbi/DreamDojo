#!/usr/bin/env bash
set -euo pipefail
set -x

DREAMDOJO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$DREAMDOJO_ROOT"

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

# CUDA 12 fix: Force PyTorch to use its bundled CUDA libraries
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""  # Clear any preloaded libraries

echo "Running on $NNODES nodes with $NPROC processes per node. This node rank is $NODE_RANK."

export PYTHONPATH="$DREAMDOJO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$DREAMDOJO_ROOT/.cache/huggingface}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$DREAMDOJO_ROOT/outputs}"
# export WANDB_API_KEY=  # Set your key before removing job.wandb_mode=disabled

source "$DREAMDOJO_ROOT/.venv/bin/activate"

config_name=${1:?"usage: bash launch.sh EXPERIMENT_NAME [HYDRA_OVERRIDE ...]"}
shift

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=$config_name \
  job.wandb_mode=disabled \
  ~dataloader_train.dataloaders \
  "$@"
