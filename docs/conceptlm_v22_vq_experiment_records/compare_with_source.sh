#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${SOURCE_REPO:-/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2}"
TARGET_REPO="${TARGET_REPO:-$SCRIPT_DIR/repo}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/comparisons}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$REPORT_DIR/source_vs_target_$STAMP.txt"

mkdir -p "$REPORT_DIR"

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='runs/'
  --exclude='run/'
  --exclude='checkpoints/'
  --exclude='checkpoint/'
  --exclude='wandb/'
  --exclude='wandb_cache/'
  --exclude='wandb_config/'
  --exclude='tensorboard/'
  --exclude='data_cache/'
  --exclude='cache/'
  --exclude='tmp/'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='.venv/'
  --exclude='venv/'
)

{
  echo "date=$(date -Is)"
  echo "source_repo=$SOURCE_REPO"
  echo "target_repo=$TARGET_REPO"
  echo
  echo "== du =="
  du -sh "$SOURCE_REPO" "$TARGET_REPO" 2>/dev/null || true
  echo
  echo "== largest source files over 50M =="
  find "$SOURCE_REPO" -type f -size +50M -printf '%s %p\n' 2>/dev/null | sort -nr | head -50 || true
  echo
  echo "== largest target files over 50M =="
  find "$TARGET_REPO" -type f -size +50M -printf '%s %p\n' 2>/dev/null | sort -nr | head -50 || true
  echo
  echo "== rsync dry-run delta after excludes =="
  rsync -ani --delete "${RSYNC_EXCLUDES[@]}" "$SOURCE_REPO/" "$TARGET_REPO/" | sed -n '1,200p' || true
} | tee "$REPORT"

echo "report=$REPORT"
