#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6000}"

export TP_SIZE="${TP_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export TRAIN_ITERS="${TRAIN_ITERS:-200}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-200}"
export RUN_NAME="${RUN_NAME:-concept-olmo-1p5b-mfu-8gpu}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data/ConceptLM/model/olmo3}"
export USE_MOCK_DATA="${USE_MOCK_DATA:-1}"

bash "${ROOT_DIR}/scripts/run_concept_olmo_1p5b_mfu.sh" "$@"
