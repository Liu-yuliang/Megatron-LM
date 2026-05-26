#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu}"
REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/public_training_bench/Megatron-LM}"
RUN_DIR="${RUN_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/public_training_bench/runs/olmo3-qknorm-check-${JOB_ID:-local}}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$BASE_DIR/miniconda3/envs/megatron-h200-cu128-py312}"
CONDA_BASE="${CONDA_BASE:-$(dirname "$(dirname "$CONDA_ENV_PREFIX")")}"

mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/tmp" "$RUN_DIR/cache"

export HOME="${HOME:-$BASE_DIR}"
export TMPDIR="${TMPDIR:-$RUN_DIR/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$RUN_DIR/torch_extensions}"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
conda activate "$CONDA_ENV_PREFIX"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python "$REPO_ROOT/examples/public_training_bench/check_olmo3_qk_norm_precision.py" \
  --output "$RUN_DIR/results/qk_norm_precision.json" \
  "$@" 2>&1 | tee "$RUN_DIR/logs/qk_norm_precision.log"
