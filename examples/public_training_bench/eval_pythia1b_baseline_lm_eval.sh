#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="${BASE_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu}"
LM_EVAL_ROOT="${LM_EVAL_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/works/lm-evaluation-harness}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$BASE_DIR/miniconda3/envs/megatron-h200-cu128-py312}"
CONDA_BASE="${CONDA_BASE:-$(dirname "$(dirname "$CONDA_ENV_PREFIX")")}"

TARGET_UID="${TARGET_UID:-10250}"
TARGET_GID="${TARGET_GID:-10250}"
TARGET_GROUPS="${TARGET_GROUPS:-10250,27,44}"
TARGET_USER="${TARGET_USER:-linzhouhan}"

if [[ "$(id -u)" == "0" && "${PUBLIC_TRAIN_ALREADY_SETUID:-0}" != "1" ]]; then
  export PUBLIC_TRAIN_ALREADY_SETUID=1
  if command -v setpriv >/dev/null 2>&1; then
    exec setpriv --reuid "$TARGET_UID" --regid "$TARGET_GID" --groups "$TARGET_GROUPS" \
      env HOME="$BASE_DIR" USER="$TARGET_USER" LOGNAME="$TARGET_USER" bash "$0" "$@"
  fi
  echo "WARN: running as root because setpriv is unavailable" >&2
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/runs/pythia1b_token_baseline_pondering_ref_20260518/checkpoints/pythia1b-token-baseline-64g-mbs16-0518}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/hnet/model/pythia-1b}"
TASKS="${TASKS:-lambada_openai,hellaswag,piqa,winogrande}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MASTER_PORT="${MASTER_PORT:-29500}"

RUN_BASE="${RUN_BASE:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/runs/pythia1b_token_baseline_pondering_ref_20260518/eval}"
RUN_NAME="${RUN_NAME:-pythia1b-baseline-lmeval-${JOB_ID:-local}-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUN_BASE/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
RESULTS_DIR="${RESULTS_DIR:-$RUN_DIR/results}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

MAIN_LOG="$LOG_DIR/lm_eval_${JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

cleanup() {
  if [[ "$(id -u)" == "0" ]]; then
    chown -R "$TARGET_UID:$TARGET_GID" "$RUN_DIR" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

export HOME="${HOME:-$BASE_DIR}"
export TMPDIR="${TMPDIR:-/tmp/lmeval_${JOB_ID:-local}_$$}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export LM_EVAL_RANK_ISOLATE_DATASETS_CACHE="${LM_EVAL_RANK_ISOLATE_DATASETS_CACHE:-1}"
export LM_EVAL_DATASETS_CACHE_ROOT="${LM_EVAL_DATASETS_CACHE_ROOT:-$XDG_CACHE_HOME/hf_datasets_by_rank}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:$LM_EVAL_ROOT:${PYTHONPATH:-}"
export MEGATRON_PATH="${MEGATRON_PATH:-$REPO_ROOT}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$LM_EVAL_DATASETS_CACHE_ROOT"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
conda activate "$CONDA_ENV_PREFIX"
set -u

cd "$LM_EVAL_ROOT"

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
PROC_PER_NODE="${PROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TOTAL_DEVICES="${TOTAL_DEVICES:-$((NODE_COUNT * PROC_PER_NODE))}"
MODEL_TYPE="${MODEL_TYPE:-gpt}"
MODEL_ARGS="load=${CHECKPOINT_ROOT},tokenizer_model=${TOKENIZER_MODEL},tokenizer_type=HuggingFaceTokenizer,devices=${TOTAL_DEVICES},micro_batch_size=${MICRO_BATCH_SIZE},seq_length=${SEQ_LENGTH},model_type=${MODEL_TYPE}"
if [[ -n "${MODEL_EXTRA_ARGS:-}" ]]; then
  MODEL_ARGS+=",extra_args=${MODEL_EXTRA_ARGS}"
fi

log "RUN_DIR=$RUN_DIR"
log "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
log "TOKENIZER_MODEL=$TOKENIZER_MODEL"
log "TASKS=$TASKS"
log "TOTAL_DEVICES=$TOTAL_DEVICES NODE_COUNT=$NODE_COUNT NODE_RANK=$NODE_RANK PROC_PER_NODE=$PROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
log "LM_EVAL_DATASETS_CACHE_ROOT=$LM_EVAL_DATASETS_CACHE_ROOT"
log "MODEL_ARGS=$MODEL_ARGS"

torchrun \
  --nnodes="$NODE_COUNT" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$PROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --local-ranks-filter=0 \
  --tee=3 \
  --redirects=3 \
  --log-dir="$LOG_DIR/torchrun" \
  -m lm_eval run \
  --model megatron_lm \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --model_args "$MODEL_ARGS" \
  --batch_size "$MICRO_BATCH_SIZE" \
  --output_path "$RESULTS_DIR" \
  --log_samples \
  --include_path "$LM_EVAL_ROOT/lm_eval/tasks"
