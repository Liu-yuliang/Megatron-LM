#!/usr/bin/env bash
set -Eeuo pipefail

RUN_BASE="/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/runs/concept_pythia_20260515"
REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/Megatron-LM}"
SUBMIT_SCRIPT="$REPO_ROOT/examples/public_training_bench/submit_rjob.sh"

export MODEL=pythia
export PYTHIA_SIZE="${PYTHIA_SIZE:-6.9b}"
export JOB_NAME="${JOB_NAME:-concept-pythia-${PYTHIA_SIZE}-$(date +%m%d%H%M)}"
export RUN_ROOT="${RUN_ROOT:-$RUN_BASE/run}"
export CKPT_DIR="${CKPT_DIR:-$RUN_BASE/checkpoints}"
export REPLICAS="${REPLICAS:-32}"
export GPUS="${GPUS:-8}"
export CPUS="${CPUS:-96}"
export MEMORY_MB="${MEMORY_MB:-786432}"
export NAMESPACE="${NAMESPACE:-ailab-plm}"
export CHARGED_GROUP="${CHARGED_GROUP:-plm_gpu}"
export IMAGE="${IMAGE:-registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab}"
export PRIORITY="${PRIORITY:-9}"
export HOST_NETWORK="${HOST_NETWORK:-true}"

concept_num_codebooks_for_pythia_size() {
  case "$1" in
    160m) echo 12 ;;
    410m) echo 16 ;;
    1b) echo 8 ;;
    1.4b) echo 16 ;;
    2.8b) echo 32 ;;
    6.9b) echo 32 ;;
    12b) echo 40 ;;
    *)
      echo "Unsupported PYTHIA_SIZE=$1. Use one of: 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b." >&2
      return 2
      ;;
  esac
}

export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-pretrain_conceptlm_v1_pythia.py}"
export CALCULATE_PER_TOKEN_LOSS=0
export CONCEPTLM_BACKBONE="${CONCEPTLM_BACKBONE:-pythia}"
export CONCEPTLM_CHUNK_SIZE="${CONCEPTLM_CHUNK_SIZE:-4}"
export CONCEPTLM_CODEBOOK_SIZE="${CONCEPTLM_CODEBOOK_SIZE:-128}"
export CONCEPTLM_NUM_CODEBOOKS="${CONCEPTLM_NUM_CODEBOOKS:-$(concept_num_codebooks_for_pythia_size "$PYTHIA_SIZE")}"
export CONCEPTLM_SPECIAL_LAYERS="${CONCEPTLM_SPECIAL_LAYERS:-2}"
export CONCEPTLM_LOSS_TYPE="${CONCEPTLM_LOSS_TYPE:-hlm_MSE_loss}"
export CONCEPTLM_MERGE_MODE="${CONCEPTLM_MERGE_MODE:-raw_logits}"
export CONCEPTLM_NCP_LOSS_WEIGHT="${CONCEPTLM_NCP_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_VQ_LOSS_WEIGHT="${CONCEPTLM_VQ_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_VQ_COMMITMENT_COST="${CONCEPTLM_VQ_COMMITMENT_COST:-0.25}"
export CONCEPTLM_HLM_FFN_HIDDEN_SIZE="${CONCEPTLM_HLM_FFN_HIDDEN_SIZE:-}"
export CONCEPTLM_DETACH_NCP_TARGET="${CONCEPTLM_DETACH_NCP_TARGET:-1}"

export USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/hnet/model}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-$MODEL_ROOT/pythia-$PYTHIA_SIZE}"
export DATA_PATH="${DATA_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/megatron_pile_from_hfds_v2/the_pile_text_document}"
export SPLIT="${SPLIT:-100,0,0}"
export MMAP_BIN_FILES="${MMAP_BIN_FILES:-1}"

export TRANSFORMER_IMPL=transformer_engine
export ATTENTION_BACKEND=flash
export SEQ_LENGTH="${SEQ_LENGTH:-2048}"
export MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-2048}"
# Pythia official training uses 2,097,152 tokens/step. At seq=2048 this is
# 1024 sequences per optimizer step.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export TP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
if [[ -n "${TRAIN_ITERS:-}" ]]; then
  # Short smoke/OOM tests can explicitly set TRAIN_ITERS. Formal training keeps
  # sample-based scheduling below so it consumes the full 330B-token corpus.
  unset TRAIN_SAMPLES
  unset LR_DECAY_SAMPLES
  unset LR_WARMUP_SAMPLES
  export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
  export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$(( TRAIN_ITERS > 1 ? 1 : 0 ))}"
else
  unset TRAIN_ITERS
  # 330B tokens at seq=2048. Megatron derives the effective iteration count from
  # train samples and global batch size, so this avoids hard-coding max steps.
  export TRAIN_SAMPLES="${TRAIN_SAMPLES:-161132812}"
  unset LR_DECAY_ITERS
  unset LR_WARMUP_ITERS
  export LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-$TRAIN_SAMPLES}"
  export LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-2048000}"
fi
export SAVE_INTERVAL="${SAVE_INTERVAL:-7500}"
export SAVE_STEP0="${SAVE_STEP0:-1}"
export SAVE_FULL_STATE="${SAVE_FULL_STATE:-1}"
export LR="${LR:-1.2e-4}"
export MIN_LR="${MIN_LR:-1.2e-5}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1.0e-8}"
export EVAL_ITERS="${EVAL_ITERS:-0}"
unset EVAL_INTERVAL
export EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-0}"
export MANUAL_GC_INTERVAL="${MANUAL_GC_INTERVAL:-10}"
export DDP_BUCKET_SIZE="${DDP_BUCKET_SIZE:-240000000}"

export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ConceptLM}"
export WANDB_ENTITY="${WANDB_ENTITY:-iammi-nanjing-university}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$JOB_NAME}"
export HIDDEN_RANK_LOG_INTERVAL="${HIDDEN_RANK_LOG_INTERVAL:-500}"
export HIDDEN_RANK_LOG_TOKENS="${HIDDEN_RANK_LOG_TOKENS:-1024}"
export HIDDEN_RANK_RTOL="${HIDDEN_RANK_RTOL:-1e-3}"
export HIDDEN_RANK_CENTER="${HIDDEN_RANK_CENTER:-1}"

export EXTRA_ARGS="${EXTRA_ARGS:-} \
  --conceptlm-backbone ${CONCEPTLM_BACKBONE} \
  --conceptlm-chunk-size ${CONCEPTLM_CHUNK_SIZE} \
  --conceptlm-codebook-size ${CONCEPTLM_CODEBOOK_SIZE} \
  --conceptlm-num-codebooks ${CONCEPTLM_NUM_CODEBOOKS} \
  --conceptlm-special-layers ${CONCEPTLM_SPECIAL_LAYERS} \
  --conceptlm-loss-type ${CONCEPTLM_LOSS_TYPE} \
  --conceptlm-merge-mode ${CONCEPTLM_MERGE_MODE} \
  --conceptlm-ncp-loss-weight ${CONCEPTLM_NCP_LOSS_WEIGHT} \
  --conceptlm-vq-loss-weight ${CONCEPTLM_VQ_LOSS_WEIGHT} \
  --conceptlm-vq-commitment-cost ${CONCEPTLM_VQ_COMMITMENT_COST}"
if [[ -n "${CONCEPTLM_HLM_FFN_HIDDEN_SIZE}" ]]; then
  EXTRA_ARGS+=" --conceptlm-hlm-ffn-hidden-size ${CONCEPTLM_HLM_FFN_HIDDEN_SIZE}"
fi
if [[ "${CONCEPTLM_DETACH_NCP_TARGET}" == "1" || "${CONCEPTLM_DETACH_NCP_TARGET}" == "true" ]]; then
  EXTRA_ARGS+=" --conceptlm-detach-ncp-target"
elif [[ "${CONCEPTLM_DETACH_NCP_TARGET}" == "0" || "${CONCEPTLM_DETACH_NCP_TARGET}" == "false" ]]; then
  EXTRA_ARGS+=" --conceptlm-no-detach-ncp-target"
fi

mkdir -p "$RUN_ROOT"
echo "Submitting Concept Pythia job: $JOB_NAME"
bash "$SUBMIT_SCRIPT" pythia
