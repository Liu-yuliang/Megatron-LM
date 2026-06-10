#!/usr/bin/env bash
set -Eeuo pipefail

# Re-submit the ConceptLM V2.2-VQ OLMo3 7B 64-GPU Muon scaled-init run.
# This delegates to the jyhuang repo because that is the repo_root recorded
# by the running job manifest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%m%d%H%M%S)}"

export REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR/repo}"
export RUN_BASE="${RUN_BASE:-$SCRIPT_DIR/runs/concept_olmo3_v22_vq_muon_64g_special8_lr6e5_scaledtrunc_opg0_$STAMP}"
export JOB_NAME="${JOB_NAME:-c22vq-muon64-s8-lr6e5-scaled-opg0-$STAMP}"

export MODEL=olmo3
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-pretrain_conceptlm_v22_vq.py}"

export REPLICAS="${REPLICAS:-8}"
export GPUS="${GPUS:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export CPUS="${CPUS:-96}"
export MEMORY_MB="${MEMORY_MB:-786432}"
export NAMESPACE="${NAMESPACE:-ailab-plm}"
export CHARGED_GROUP="${CHARGED_GROUP:-plm_gpu}"
export IMAGE="${IMAGE:-registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab}"
export PRIORITY="${PRIORITY:-9}"
export HOST_NETWORK="${HOST_NETWORK:-true}"

export OPTIMIZER="${OPTIMIZER:-muon}"
export LR="${LR:-6.0e-5}"
export MIN_LR="${MIN_LR:-6.0e-6}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-1430512}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2000}"
export TRAIN_ITERS="${TRAIN_ITERS:-1430512}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1.0e-8}"

export INIT_METHOD_VARIANT="${INIT_METHOD_VARIANT:-scaled_truncated_normal}"
export CONCEPTLM_SPECIAL_LAYERS="${CONCEPTLM_SPECIAL_LAYERS:-8}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-0}"
export GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-0}"
export SAVE_FULL_STATE="${SAVE_FULL_STATE:-1}"
export OLMO3_Z_LOSS_MULTIPLIER="${OLMO3_Z_LOSS_MULTIPLIER:-1e-5}"
export OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO="${OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO:-1}"

export USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ConceptLM}"
export WANDB_ENTITY="${WANDB_ENTITY:-iammi-nanjing-university}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$JOB_NAME}"

echo "Submitting $JOB_NAME"
echo "  repo_root=$REPO_ROOT"
echo "  run_base=$RUN_BASE"
echo "  optimizer=$OPTIMIZER lr=$LR min_lr=$MIN_LR wd=$WEIGHT_DECAY"
echo "  init_method_variant=$INIT_METHOD_VARIANT"

bash "$REPO_ROOT/examples/public_training_bench/train_concept_v22_vq_olmo3_7b.sh"
