#!/usr/bin/env bash
set -Eeuo pipefail

RUN_BASE="${RUN_BASE:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/runs/concept_v21_pythia_20260518}"
REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/works/megatron_ConceptlmV2}"
SUBMIT_SCRIPT="$REPO_ROOT/examples/public_training_bench/submit_rjob.sh"

export MODEL=pythia
export PYTHIA_SIZE="${PYTHIA_SIZE:-6.9b}"
export JOB_NAME="${JOB_NAME:-concept-v21-pythia-${PYTHIA_SIZE}-$(date +%m%d%H%M)}"
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

export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-pretrain_conceptlm_v21_pythia.py}"
export CALCULATE_PER_TOKEN_LOSS=0
export CONCEPTLM_BACKBONE="${CONCEPTLM_BACKBONE:-pythia}"
export CONCEPTLM_ENCODER_LAYERS="${CONCEPTLM_ENCODER_LAYERS:-6}"
export CONCEPTLM_SPECIAL_LAYERS="${CONCEPTLM_SPECIAL_LAYERS:-6}"
export CONCEPTLM_DECODER_LAYERS="${CONCEPTLM_DECODER_LAYERS:-6}"
export CONCEPTLM_CHUNK_SIZE="${CONCEPTLM_CHUNK_SIZE:-4}"
export CONCEPTLM_CHUNK_MERGE_METHOD="${CONCEPTLM_CHUNK_MERGE_METHOD:-meanpooling}"
export CONCEPTLM_SHIFT_FEATURE="${CONCEPTLM_SHIFT_FEATURE:-1}"
export CONCEPTLM_ENABLE_CHUNK_DUALPATH_SMOOTHING="${CONCEPTLM_ENABLE_CHUNK_DUALPATH_SMOOTHING:-1}"
export CONCEPTLM_CHUNK_DUALPATH_ALPHA_INIT="${CONCEPTLM_CHUNK_DUALPATH_ALPHA_INIT:-0.5}"
export CONCEPTLM_BOTTLENECK_TYPE="${CONCEPTLM_BOTTLENECK_TYPE:-mlp}"
export CONCEPTLM_VQ_PATCH_RATIO="${CONCEPTLM_VQ_PATCH_RATIO:-1}"
export CONCEPTLM_MLP_BOTTLENECK_RATIO="${CONCEPTLM_MLP_BOTTLENECK_RATIO:-0.25}"
export CONCEPTLM_MLP_BOTTLENECK_ACTIVATION="${CONCEPTLM_MLP_BOTTLENECK_ACTIVATION:-none}"
export CONCEPTLM_MLP_RECON_LOSS_WEIGHT="${CONCEPTLM_MLP_RECON_LOSS_WEIGHT:-0.0}"
export CONCEPTLM_MLP_HLM_LOSS_WEIGHT="${CONCEPTLM_MLP_HLM_LOSS_WEIGHT:-0.0}"
export CONCEPTLM_MLP_HIDDEN_LOSS_WEIGHT="${CONCEPTLM_MLP_HIDDEN_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_MLP_HLM_LOSS_TYPE="${CONCEPTLM_MLP_HLM_LOSS_TYPE:-mse}"
export CONCEPTLM_LAYER_NORM_OPTION="${CONCEPTLM_LAYER_NORM_OPTION:-normed_add}"
export CONCEPTLM_FUSION_NORM_ALPHA_INIT="${CONCEPTLM_FUSION_NORM_ALPHA_INIT:-0.1}"
export CONCEPTLM_HLM_FFN_HIDDEN_SIZE="${CONCEPTLM_HLM_FFN_HIDDEN_SIZE:-}"
export CONCEPTLM_HLM_LOSS_WEIGHT="${CONCEPTLM_HLM_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_VQ_LOSS_WEIGHT="${CONCEPTLM_VQ_LOSS_WEIGHT:-1.0}"

export CONCEPTLM_V21_DD_TWO_ROUTE_ADD="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_SOURCE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_SOURCE:-final}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_RAW_CONCEPT_ROUTE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_RAW_CONCEPT_ROUTE:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_FINAL_CONCEPT_ROUTE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_FINAL_CONCEPT_ROUTE:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_BETA_INIT="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_BETA_INIT:-0.05}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_EVERY_N_LAYERS="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_EVERY_N_LAYERS:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_USE_SOFTMAX="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_USE_SOFTMAX:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_LAYERNORM="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_LAYERNORM:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_SOFTMAX="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_SOFTMAX:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_USE_LAYERNORM="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_USE_LAYERNORM:-1}"
export CONCEPTLM_V21_DD_ENCODER_SELF_DD="${CONCEPTLM_V21_DD_ENCODER_SELF_DD:-1}"
export CONCEPTLM_V21_DD_ENCODER_SELF_DD_USE_LAYERNORM="${CONCEPTLM_V21_DD_ENCODER_SELF_DD_USE_LAYERNORM:-0}"
export CONCEPTLM_V21_DD_CONCEPT_SELF_DD="${CONCEPTLM_V21_DD_CONCEPT_SELF_DD:-1}"
export CONCEPTLM_V21_DD_CONCEPT_SELF_DD_USE_LAYERNORM="${CONCEPTLM_V21_DD_CONCEPT_SELF_DD_USE_LAYERNORM:-0}"
export CONCEPTLM_V21_ENABLE_CONCEPT_READ_ENCODER="${CONCEPTLM_V21_ENABLE_CONCEPT_READ_ENCODER:-1}"
export CONCEPTLM_V21_ENABLE_DECODER_READ_ENCODER="${CONCEPTLM_V21_ENABLE_DECODER_READ_ENCODER:-1}"
export CONCEPTLM_V21_ENABLE_DECODER_READ_CONCEPT="${CONCEPTLM_V21_ENABLE_DECODER_READ_CONCEPT:-1}"
export CONCEPTLM_V21_RESIDUAL_FLOW_BETA_INIT="${CONCEPTLM_V21_RESIDUAL_FLOW_BETA_INIT:-0.02}"
export CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_HIDDEN_SIZE="${CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_HIDDEN_SIZE:-0}"
export CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_USE_SOFTMAX="${CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_USE_SOFTMAX:-1}"
export CONCEPTLM_V21_RESIDUAL_FLOW_SOURCE_USE_LAYERNORM="${CONCEPTLM_V21_RESIDUAL_FLOW_SOURCE_USE_LAYERNORM:-1}"
export CONCEPTLM_V21_RESIDUAL_FLOW_SHARED_SOURCE_NORM="${CONCEPTLM_V21_RESIDUAL_FLOW_SHARED_SOURCE_NORM:-1}"
export CONCEPTLM_V21_COMPILE_RESIDUAL_FLOW_ROUTES="${CONCEPTLM_V21_COMPILE_RESIDUAL_FLOW_ROUTES:-1}"
export CONCEPTLM_V21_COMPILE_DD_ROUTES="${CONCEPTLM_V21_COMPILE_DD_ROUTES:-1}"

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
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export TP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
if [[ -n "${TRAIN_ITERS:-}" ]]; then
  unset TRAIN_SAMPLES
  unset LR_DECAY_SAMPLES
  unset LR_WARMUP_SAMPLES
  export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
  export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$(( TRAIN_ITERS > 1 ? 1 : 0 ))}"
else
  unset TRAIN_ITERS
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
export WANDB_PROJECT="${WANDB_PROJECT:-public-training-bench}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$JOB_NAME}"
export HIDDEN_RANK_LOG_INTERVAL="${HIDDEN_RANK_LOG_INTERVAL:-500}"
export HIDDEN_RANK_LOG_TOKENS="${HIDDEN_RANK_LOG_TOKENS:-1024}"
export HIDDEN_RANK_RTOL="${HIDDEN_RANK_RTOL:-1e-3}"
export HIDDEN_RANK_CENTER="${HIDDEN_RANK_CENTER:-1}"

export EXTRA_ARGS="${EXTRA_ARGS:-} \
  --conceptlm-backbone ${CONCEPTLM_BACKBONE} \
  --conceptlm-encoder-layers ${CONCEPTLM_ENCODER_LAYERS} \
  --conceptlm-special-layers ${CONCEPTLM_SPECIAL_LAYERS} \
  --conceptlm-decoder-layers ${CONCEPTLM_DECODER_LAYERS} \
  --conceptlm-chunk-size ${CONCEPTLM_CHUNK_SIZE} \
  --conceptlm-chunk-merge-method ${CONCEPTLM_CHUNK_MERGE_METHOD} \
  --conceptlm-bottleneck-type ${CONCEPTLM_BOTTLENECK_TYPE} \
  --conceptlm-vq-patch-ratio ${CONCEPTLM_VQ_PATCH_RATIO} \
  --conceptlm-mlp-bottleneck-ratio ${CONCEPTLM_MLP_BOTTLENECK_RATIO} \
  --conceptlm-mlp-bottleneck-activation ${CONCEPTLM_MLP_BOTTLENECK_ACTIVATION} \
  --conceptlm-mlp-recon-loss-weight ${CONCEPTLM_MLP_RECON_LOSS_WEIGHT} \
  --conceptlm-mlp-hlm-loss-weight ${CONCEPTLM_MLP_HLM_LOSS_WEIGHT} \
  --conceptlm-mlp-hidden-loss-weight ${CONCEPTLM_MLP_HIDDEN_LOSS_WEIGHT} \
  --conceptlm-mlp-hlm-loss-type ${CONCEPTLM_MLP_HLM_LOSS_TYPE} \
  --conceptlm-layer-norm-option ${CONCEPTLM_LAYER_NORM_OPTION} \
  --conceptlm-fusion-norm-alpha-init ${CONCEPTLM_FUSION_NORM_ALPHA_INIT} \
  --conceptlm-hlm-loss-weight ${CONCEPTLM_HLM_LOSS_WEIGHT} \
  --conceptlm-vq-loss-weight ${CONCEPTLM_VQ_LOSS_WEIGHT} \
  --conceptlm-v21-dd-two-route-add-concept-source ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_SOURCE} \
  --conceptlm-v21-dd-two-route-add-beta-init ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_BETA_INIT} \
  --conceptlm-v21-dd-two-route-add-every-n-layers ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_EVERY_N_LAYERS} \
  --conceptlm-v21-residual-flow-beta-init ${CONCEPTLM_V21_RESIDUAL_FLOW_BETA_INIT} \
  --conceptlm-v21-residual-flow-route-hidden-size ${CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_HIDDEN_SIZE}"

[[ "$CONCEPTLM_SHIFT_FEATURE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-shift-feature" || EXTRA_ARGS+=" --no-conceptlm-shift-feature"
[[ "$CONCEPTLM_ENABLE_CHUNK_DUALPATH_SMOOTHING" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-enable-chunk-dualpath-smoothing" || EXTRA_ARGS+=" --no-conceptlm-enable-chunk-dualpath-smoothing"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_RAW_CONCEPT_ROUTE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-enable-raw-concept-route" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-enable-raw-concept-route"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_FINAL_CONCEPT_ROUTE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-enable-final-concept-route" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-enable-final-concept-route"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_USE_SOFTMAX" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-use-softmax" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-use-softmax"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-decoder-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-decoder-use-layernorm"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_SOFTMAX" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-decoder-use-softmax" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-decoder-use-softmax"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-concept-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-concept-use-layernorm"
[[ "$CONCEPTLM_V21_DD_ENCODER_SELF_DD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-encoder-self-dd" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-encoder-self-dd"
[[ "$CONCEPTLM_V21_DD_ENCODER_SELF_DD_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-encoder-self-dd-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-encoder-self-dd-use-layernorm"
[[ "$CONCEPTLM_V21_DD_CONCEPT_SELF_DD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-concept-self-dd" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-concept-self-dd"
[[ "$CONCEPTLM_V21_DD_CONCEPT_SELF_DD_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-concept-self-dd-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-concept-self-dd-use-layernorm"
[[ "$CONCEPTLM_V21_ENABLE_CONCEPT_READ_ENCODER" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-concept-read-encoder" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-concept-read-encoder"
[[ "$CONCEPTLM_V21_ENABLE_DECODER_READ_ENCODER" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-decoder-read-encoder" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-decoder-read-encoder"
[[ "$CONCEPTLM_V21_ENABLE_DECODER_READ_CONCEPT" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-decoder-read-concept" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-decoder-read-concept"
[[ "$CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_USE_SOFTMAX" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-residual-flow-route-use-softmax" || EXTRA_ARGS+=" --no-conceptlm-v21-residual-flow-route-use-softmax"
[[ "$CONCEPTLM_V21_RESIDUAL_FLOW_SOURCE_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-residual-flow-source-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-residual-flow-source-use-layernorm"
[[ "$CONCEPTLM_V21_RESIDUAL_FLOW_SHARED_SOURCE_NORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-residual-flow-shared-source-norm" || EXTRA_ARGS+=" --no-conceptlm-v21-residual-flow-shared-source-norm"
[[ "$CONCEPTLM_V21_COMPILE_RESIDUAL_FLOW_ROUTES" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-compile-residual-flow-routes" || EXTRA_ARGS+=" --no-conceptlm-v21-compile-residual-flow-routes"
[[ "$CONCEPTLM_V21_COMPILE_DD_ROUTES" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-compile-dd-routes" || EXTRA_ARGS+=" --no-conceptlm-v21-compile-dd-routes"

if [[ -n "${CONCEPTLM_HLM_FFN_HIDDEN_SIZE}" ]]; then
  EXTRA_ARGS+=" --conceptlm-hlm-ffn-hidden-size ${CONCEPTLM_HLM_FFN_HIDDEN_SIZE}"
fi

mkdir -p "$RUN_ROOT"
echo "Submitting ConceptLM V2.1 Pythia job: $JOB_NAME"
bash "$SUBMIT_SCRIPT" pythia
