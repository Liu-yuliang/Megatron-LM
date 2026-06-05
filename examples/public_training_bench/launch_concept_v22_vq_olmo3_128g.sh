#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

export REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"
export OPTIMIZER="${OPTIMIZER:-adam}"
export RUN_BASE="${RUN_BASE:-/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_${OPTIMIZER}_128g_$STAMP}"
export JOB_NAME="${JOB_NAME:-concept-olmo3-v22vq-${OPTIMIZER}-128g-$(date +%m%d%H%M)}"

# 16 replicas * 8 GPUs per replica = 128 GPUs.
export REPLICAS="${REPLICAS:-16}"
export GPUS="${GPUS:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$GPUS}"
export CPUS="${CPUS:-96}"
export MEMORY_MB="${MEMORY_MB:-786432}"

export CHARGED_GROUP="${CHARGED_GROUP:-plm_gpu}"
export NAMESPACE="${NAMESPACE:-ailab-plm}"
export IMAGE="${IMAGE:-registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab}"
export PRIORITY="${PRIORITY:-9}"
export HOST_NETWORK="${HOST_NETWORK:-true}"

# Formal ConceptLM V2.2-VQ + OLMo3 defaults. The lower-level script passes
# Megatron optimizer defaults and enables --use-distributed-optimizer.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1.0e-8}"

export USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$JOB_NAME}"

echo "Launching ConceptLM V2.2-VQ + ${OPTIMIZER} on ${REPLICAS}x${GPUS} GPUs"
echo "  repo_root=$REPO_ROOT"
echo "  run_base=$RUN_BASE"
echo "  job_name=$JOB_NAME"
echo "  optimizer=$OPTIMIZER"
echo "  conda_env=default from train_olmo3_7b.sh"

bash "$REPO_ROOT/examples/public_training_bench/train_concept_v22_vq_olmo3_7b.sh"
