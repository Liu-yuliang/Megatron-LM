#!/usr/bin/env bash
set -Eeuo pipefail

RUN_BASE="/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/runs/concept_olmo3_20260515"
REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/Megatron-LM}"
SUBMIT_SCRIPT="$REPO_ROOT/examples/public_training_bench/submit_rjob.sh"

export MODEL=olmo3
export JOB_NAME="${JOB_NAME:-concept-olmo3-$(date +%m%d%H%M)}"
export RUN_ROOT="${RUN_ROOT:-$RUN_BASE/run}"
export REPLICAS=1
export GPUS=8
export CPUS="${CPUS:-96}"
export MEMORY_MB="${MEMORY_MB:-786432}"
export NAMESPACE="${NAMESPACE:-ailab-plm}"
export CHARGED_GROUP="${CHARGED_GROUP:-plm_gpu}"
export IMAGE="${IMAGE:-registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab}"
export PRIORITY="${PRIORITY:-9}"
export HOST_NETWORK=true

export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-pretrain_conceptlm_v1_olmo3.py}"
export CALCULATE_PER_TOKEN_LOSS=0
export CONCEPTLM_BACKBONE="${CONCEPTLM_BACKBONE:-olmo3}"
export CONCEPTLM_CHUNK_SIZE="${CONCEPTLM_CHUNK_SIZE:-4}"
export CONCEPTLM_CODEBOOK_SIZE="${CONCEPTLM_CODEBOOK_SIZE:-128}"
export CONCEPTLM_NUM_CODEBOOKS="${CONCEPTLM_NUM_CODEBOOKS:-32}"
export CONCEPTLM_SPECIAL_LAYERS="${CONCEPTLM_SPECIAL_LAYERS:-2}"
export CONCEPTLM_LOSS_TYPE="${CONCEPTLM_LOSS_TYPE:-hlm_MSE_loss}"
export CONCEPTLM_MERGE_MODE="${CONCEPTLM_MERGE_MODE:-softmax}"
export CONCEPTLM_NCP_LOSS_WEIGHT="${CONCEPTLM_NCP_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_VQ_LOSS_WEIGHT="${CONCEPTLM_VQ_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_VQ_COMMITMENT_COST="${CONCEPTLM_VQ_COMMITMENT_COST:-0.25}"
export CONCEPTLM_HLM_FFN_HIDDEN_SIZE="${CONCEPTLM_HLM_FFN_HIDDEN_SIZE:-}"
export CONCEPTLM_DETACH_NCP_TARGET="${CONCEPTLM_DETACH_NCP_TARGET:-1}"

export USE_MOCK_DATA=0
export DATA_ARGS_PATH="${DATA_ARGS_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-6T/megatron_olmo3_8192/data_args_path.txt}"
export PER_DATASET_SEQUENCES_PATH="${PER_DATASET_SEQUENCES_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-6T/megatron_olmo3_8192/per_dataset_sequences.json}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/model/Olmo-3-1025-7B-stage1-step1413814}"
export SPLIT="${SPLIT:-100,0,0}"
export MMAP_BIN_FILES="${MMAP_BIN_FILES:-0}"

export TRANSFORMER_IMPL=transformer_engine
export OLMO3_LAYER_SPEC_NAME=olmo3_layer_spec
export OLMO3_QK_NORM_IMPL=te
export ATTENTION_BACKEND=flash
export FUSED_RESIDUAL_RMSNORM=1
export SEQ_LENGTH=8192
export MAX_POSITION_EMBEDDINGS=8192
export GLOBAL_BATCH_SIZE=512
export MICRO_BATCH_SIZE=1
export TP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
export TRAIN_ITERS=1430512
export SAVE_INTERVAL=7153
export LR_DECAY_ITERS=1430512
export LR_WARMUP_ITERS=2000
export LR=3.0e-4
export MIN_LR=3.0e-5
export CLIP_GRAD=1.0
export WEIGHT_DECAY=0.1
export ADAM_BETA1=0.9
export ADAM_BETA2=0.95
export ADAM_EPS=1.0e-8
export GRAD_REDUCE_IN_BF16=0
export SAVE_FULL_STATE=1
export EMPTY_UNUSED_MEMORY_LEVEL=0
export MANUAL_GC_INTERVAL=10
export RERUN_MODE=disabled
export DDP_BUCKET_SIZE=240000000

export OLMO3_Z_LOSS_MULTIPLIER=1e-5
export OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO=1
export HIDDEN_RANK_LOG_INTERVAL=500
export HIDDEN_RANK_LOG_TOKENS=1024
export HIDDEN_RANK_RTOL=1e-3
export HIDDEN_RANK_CENTER=1

export ENABLE_WANDB=1
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-ConceptLM}"
export WANDB_ENTITY="${WANDB_ENTITY:-iammi-nanjing-university}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$JOB_NAME}"

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
echo "Submitting Concept OLMo 3 job: $JOB_NAME"
bash "$SUBMIT_SCRIPT" olmo3
