#!/usr/bin/env bash
set -Eeuo pipefail

AB_ROOT="/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step4_paramorder_0610015751"
STAMP="0610015751"

BASE_REPO="/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/ab_worktrees/conceptlm_step4_base_1ca63d1cc_0610015751"
NEW_REPO="/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/ab_worktrees/conceptlm_step4_new_6ef82efbf_0610015751"
BASE_COMMIT="1ca63d1cc17d2f69afe0ae8588a8c0da8dcfdadc"
NEW_COMMIT="6ef82efbfc93fad97f9692bbee6c1d75f0893b2b"

mkdir -p "$AB_ROOT/logs" "$AB_ROOT/jobs"

submit_one() {
  local label="$1"
  local repo="$2"
  local commit="$3"
  local job_name="c22vq-s4p-${label}-16g-${STAMP}"
  local run_base="/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step4p_${label}_16g_${STAMP}"
  local submit_log="$AB_ROOT/logs/submit_${label}.log"
  local metadata_file="$AB_ROOT/jobs/${label}.metadata"

  {
    echo "date=$(date -Is)"
    echo "label=$label"
    echo "job_name=$job_name"
    echo "repo=$repo"
    echo "expected_commit=$commit"
    echo "actual_commit=$(git -C "$repo" rev-parse HEAD)"
    echo "run_base=$run_base"
    echo "ab_root=$AB_ROOT"
    echo "train_iters=60"
    echo "save_interval=1000000"
    echo "overlap_param_gather=0"
    echo "wandb=0"
    echo "step4_change=ConceptLM V2.2 named_parameters forward-order default"
    echo
  } | tee "$submit_log"

  env \
    REPO_ROOT="$repo" \
    RUN_BASE="$run_base" \
    RUN_ROOT="$run_base/run" \
    CKPT_DIR="$run_base/checkpoints" \
    JOB_NAME="$job_name" \
    REPLICAS=2 \
    GPUS=8 \
    GPUS_PER_NODE=8 \
    CPUS=96 \
    MEMORY_MB=786432 \
    NAMESPACE=ailab-plm \
    CHARGED_GROUP=plm_gpu \
    IMAGE=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
    PRIORITY=9 \
    HOST_NETWORK=true \
    USE_MOCK_DATA=0 \
    TRAIN_ITERS=60 \
    SAVE_INTERVAL=1000000 \
    EVAL_ITERS=0 \
    EVAL_INTERVAL=1000000 \
    LOG_INTERVAL=1 \
    ENABLE_WANDB=0 \
    WANDB_MODE=offline \
    OVERLAP_PARAM_GATHER=0 \
    GRAD_REDUCE_IN_BF16=0 \
    SAVE_FULL_STATE=0 \
    CONCEPTLM_V21_SELF_DD_UNSTACKED_FASTPATH=1 \
    CONCEPTLM_V21_ROUTE_FIRST_PLUS_LAST_N=0 \
    bash "$repo/examples/public_training_bench/train_concept_v22_vq_olmo3_7b.sh" \
    2>&1 | tee -a "$submit_log"

  local metadata_name
  metadata_name="$(sed -n 's/^metadata_name=//p' "$submit_log" | tail -n 1)"
  if [[ -z "$metadata_name" ]]; then
    metadata_name="$job_name"
  fi
  echo "$metadata_name" > "$metadata_file"
  rjob get "$metadata_name" --namespace ailab-plm > "$AB_ROOT/jobs/${label}.status.txt" 2>&1 || true
  echo "metadata_name=$metadata_name"
}

submit_one base "$BASE_REPO" "$BASE_COMMIT"
submit_one new "$NEW_REPO" "$NEW_COMMIT"

{
  echo "date=$(date -Is)"
  echo "base_commit=$BASE_COMMIT"
  echo "new_commit=$NEW_COMMIT"
  echo "base_metadata=$(cat "$AB_ROOT/jobs/base.metadata" 2>/dev/null || true)"
  echo "new_metadata=$(cat "$AB_ROOT/jobs/new.metadata" 2>/dev/null || true)"
  echo "push_status=failed: github https credentials unavailable"
  echo "previous_pass=/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step3_decoderroute_0610012914/RESULT.md"
  echo "step4_change=forward-order named_parameters without module tree, state_dict, or forward math changes"
} > "$AB_ROOT/jobs/summary.txt"
