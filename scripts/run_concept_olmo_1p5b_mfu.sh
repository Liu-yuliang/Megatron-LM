#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"

TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data/ConceptLM/model/olmo3}"
DATA_PATH="${DATA_PATH:-}"
USE_MOCK_DATA="${USE_MOCK_DATA:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/checkpoints/concept_olmo_1p5b_mfu}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
RUN_NAME="${RUN_NAME:-concept-olmo-1p5b-mfu}"

mkdir -p "${CHECKPOINT_PATH}" "${LOG_DIR}"

EXTRA_ARGS=("$@")

TRAIN_ARGS=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  pretrain_concept_olmo.py \
  --save "${CHECKPOINT_PATH}" \
  --load "${CHECKPOINT_PATH}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --tokenizer-type HuggingFaceTokenizer \
  --distributed-backend nccl \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-512}" \
  --train-iters "${TRAIN_ITERS:-200}" \
  --lr "${LR:-3e-4}" \
  --min-lr "${MIN_LR:-3e-5}" \
  --lr-decay-style cosine \
  --lr-warmup-iters "${LR_WARMUP_ITERS:-200}" \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --use-flash-attn \
  --normalization RMSNorm \
  --position-embedding-type rope \
  --rotary-base 500000 \
  --no-position-embedding \
  --swiglu \
  --disable-bias-linear \
  --num-layers 16 \
  --hidden-size 2048 \
  --ffn-hidden-size 8192 \
  --num-attention-heads 16 \
  --num-query-groups 16 \
  --untie-embeddings-and-output-weights \
  --transformer-impl local \
  --attention-backend flash \
  --tensor-model-parallel-size "${TP_SIZE:-1}" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --concept-use-olmo-defaults \
  --concept-encoder-num-layers 0 \
  --concept-decoder-num-layers 16 \
  --concept-hlm-num-layers 2 \
  --concept-chunk-size 4 \
  --concept-vq-patch-ratio 1 \
  --concept-codebook-size 128 \
  --concept-include-loss \
  --concept-commit-loss-weight 1.0 \
  --concept-hlm-loss-weight 1.0 \
)

if [[ -n "${DATA_PATH}" ]]; then
  TRAIN_ARGS+=(--train-data-path "${DATA_PATH}")
  echo "[data] using Megatron indexed dataset prefix: ${DATA_PATH}"
elif [[ "${USE_MOCK_DATA}" == "1" || "${USE_MOCK_DATA}" == "true" || "${USE_MOCK_DATA}" == "yes" ]]; then
  TRAIN_ARGS+=(--mock-data)
  echo "[data] DATA_PATH not set, falling back to --mock-data for MFU / throughput smoke test"
else
  echo "DATA_PATH is required unless USE_MOCK_DATA=1"
  exit 1
fi

TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

"${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
