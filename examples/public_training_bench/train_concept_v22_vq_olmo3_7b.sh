#!/usr/bin/env bash
set -Eeuo pipefail

RUN_BASE="${RUN_BASE:-/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_muon_64g_special8_lr6e5_scaledtrunc_opg0_$(date +%m%d%H%M)}"
REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2}"
SUBMIT_SCRIPT="$REPO_ROOT/examples/public_training_bench/submit_rjob.sh"

export MODEL=olmo3
export JOB_NAME="${JOB_NAME:-c22vq-muon64-s8-lr6e5-scaled-opg0-$(date +%m%d%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-$RUN_BASE/run}"
export CKPT_DIR="${CKPT_DIR:-$RUN_BASE/checkpoints}"
export REPLICAS="${REPLICAS:-8}"
export GPUS="${GPUS:-8}"
export CPUS="${CPUS:-96}"
export MEMORY_MB="${MEMORY_MB:-786432}"
export NAMESPACE="${NAMESPACE:-ailab-plm}"
export CHARGED_GROUP="${CHARGED_GROUP:-plm_gpu}"
export IMAGE="${IMAGE:-registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab}"
export PRIORITY="${PRIORITY:-9}"
export HOST_NETWORK="${HOST_NETWORK:-true}"

export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-pretrain_conceptlm_v22_vq.py}"
export CALCULATE_PER_TOKEN_LOSS=0
export CONCEPTLM_BACKBONE="${CONCEPTLM_BACKBONE:-olmo3}"
export CONCEPTLM_ENCODER_LAYERS="${CONCEPTLM_ENCODER_LAYERS:-16}"
export CONCEPTLM_DECODER_LAYERS="${CONCEPTLM_DECODER_LAYERS:-16}"
export CONCEPTLM_SPECIAL_LAYERS="${CONCEPTLM_SPECIAL_LAYERS:-8}"
export CONCEPTLM_CHUNK_SIZE="${CONCEPTLM_CHUNK_SIZE:-4}"
export CONCEPTLM_CHUNK_MERGE_METHOD="${CONCEPTLM_CHUNK_MERGE_METHOD:-meanpooling}"
export CONCEPTLM_SHIFT_FEATURE="${CONCEPTLM_SHIFT_FEATURE:-1}"
export CONCEPTLM_ENABLE_CHUNK_DUALPATH_SMOOTHING="${CONCEPTLM_ENABLE_CHUNK_DUALPATH_SMOOTHING:-0}"
export CONCEPTLM_CHUNK_DUALPATH_ALPHA_INIT="${CONCEPTLM_CHUNK_DUALPATH_ALPHA_INIT:-0.5}"
export CONCEPTLM_LAYER_NORM_OPTION="${CONCEPTLM_LAYER_NORM_OPTION:-normed_add}"
export CONCEPTLM_FUSION_NORM_ALPHA_INIT="${CONCEPTLM_FUSION_NORM_ALPHA_INIT:-0.1}"
export CONCEPTLM_FUSION_ALPHA_INIT="${CONCEPTLM_FUSION_ALPHA_INIT:-1.0}"
export CONCEPTLM_HLM_FFN_HIDDEN_SIZE="${CONCEPTLM_HLM_FFN_HIDDEN_SIZE:-}"
export CONCEPTLM_HLM_LOSS_WEIGHT="${CONCEPTLM_HLM_LOSS_WEIGHT:-1.0}"
export CONCEPTLM_VQ_LOSS_WEIGHT="${CONCEPTLM_VQ_LOSS_WEIGHT:-1.0}"

export CONCEPTLM_V22_VQ_CODEBOOK_SIZE="${CONCEPTLM_V22_VQ_CODEBOOK_SIZE:-128}"
export CONCEPTLM_V22_VQ_NUM_CODEBOOKS="${CONCEPTLM_V22_VQ_NUM_CODEBOOKS:-32}"
export CONCEPTLM_V22_VQ_COMMITMENT_COST="${CONCEPTLM_V22_VQ_COMMITMENT_COST:-0.25}"
export CONCEPTLM_V22_VQ_MERGE_MODE="${CONCEPTLM_V22_VQ_MERGE_MODE:-raw_logits}"
export CONCEPTLM_V22_VQ_HLM_LOSS_TYPE="${CONCEPTLM_V22_VQ_HLM_LOSS_TYPE:-mse}"
export CONCEPTLM_V22_VQ_DETACH_HLM_TARGET="${CONCEPTLM_V22_VQ_DETACH_HLM_TARGET:-1}"
export CONCEPTLM_V22_VQ_LR_MULT="${CONCEPTLM_V22_VQ_LR_MULT:-1.0}"
export CONCEPTLM_V22_VQ_HIGHLEVEL_LR_MULT="${CONCEPTLM_V22_VQ_HIGHLEVEL_LR_MULT:-1.0}"

export CONCEPTLM_V21_DD_TWO_ROUTE_ADD="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_SOURCE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_SOURCE:-final}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_RAW_CONCEPT_ROUTE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_RAW_CONCEPT_ROUTE:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_FINAL_CONCEPT_ROUTE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_FINAL_CONCEPT_ROUTE:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_BETA_INIT="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_BETA_INIT:-0.05}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_EVERY_N_LAYERS="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_EVERY_N_LAYERS:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_ROUTE_FIRST_N="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_ROUTE_FIRST_N:--1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_HIDDEN_SIZE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_HIDDEN_SIZE:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_HIDDEN_SIZE="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_HIDDEN_SIZE:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_USE_SOFTMAX="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_USE_SOFTMAX:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DISABLE_DECODER_DD="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DISABLE_DECODER_DD:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_LAYERNORM="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_LAYERNORM:-0}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_SOFTMAX="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_SOFTMAX:-1}"
export CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_USE_LAYERNORM="${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_USE_LAYERNORM:-1}"
export CONCEPTLM_V21_DD_ENCODER_SELF_DD="${CONCEPTLM_V21_DD_ENCODER_SELF_DD:-1}"
export CONCEPTLM_V21_DD_ENCODER_SELF_DD_EVERY_N_LAYERS="${CONCEPTLM_V21_DD_ENCODER_SELF_DD_EVERY_N_LAYERS:-1}"
export CONCEPTLM_V21_DD_ENCODER_SELF_DD_HIDDEN_SIZE="${CONCEPTLM_V21_DD_ENCODER_SELF_DD_HIDDEN_SIZE:-0}"
export CONCEPTLM_V21_DD_ENCODER_SELF_DD_USE_LAYERNORM="${CONCEPTLM_V21_DD_ENCODER_SELF_DD_USE_LAYERNORM:-0}"
export CONCEPTLM_V21_DD_CONCEPT_SELF_DD="${CONCEPTLM_V21_DD_CONCEPT_SELF_DD:-1}"
export CONCEPTLM_V21_DD_CONCEPT_SELF_DD_EVERY_N_LAYERS="${CONCEPTLM_V21_DD_CONCEPT_SELF_DD_EVERY_N_LAYERS:-1}"
export CONCEPTLM_V21_DD_CONCEPT_SELF_DD_HIDDEN_SIZE="${CONCEPTLM_V21_DD_CONCEPT_SELF_DD_HIDDEN_SIZE:-0}"
export CONCEPTLM_V21_DD_CONCEPT_SELF_DD_USE_LAYERNORM="${CONCEPTLM_V21_DD_CONCEPT_SELF_DD_USE_LAYERNORM:-0}"
export CONCEPTLM_V21_ENABLE_FULL_RESIDUAL_FLOW="${CONCEPTLM_V21_ENABLE_FULL_RESIDUAL_FLOW:-0}"
export CONCEPTLM_V21_ENABLE_CONCEPT_READ_ENCODER="${CONCEPTLM_V21_ENABLE_CONCEPT_READ_ENCODER:-1}"
export CONCEPTLM_V21_ENABLE_DECODER_READ_ENCODER="${CONCEPTLM_V21_ENABLE_DECODER_READ_ENCODER:-1}"
export CONCEPTLM_V21_ENABLE_DECODER_READ_CONCEPT="${CONCEPTLM_V21_ENABLE_DECODER_READ_CONCEPT:-1}"
export CONCEPTLM_V21_RESIDUAL_FLOW_BETA_INIT="${CONCEPTLM_V21_RESIDUAL_FLOW_BETA_INIT:-0.02}"
export CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_HIDDEN_SIZE="${CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_HIDDEN_SIZE:-0}"
export CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_USE_SOFTMAX="${CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_USE_SOFTMAX:-1}"
export CONCEPTLM_V21_RESIDUAL_FLOW_SOURCE_USE_LAYERNORM="${CONCEPTLM_V21_RESIDUAL_FLOW_SOURCE_USE_LAYERNORM:-1}"
export CONCEPTLM_V21_RESIDUAL_FLOW_SHARED_SOURCE_NORM="${CONCEPTLM_V21_RESIDUAL_FLOW_SHARED_SOURCE_NORM:-1}"
export CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE="${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE:-1}"
export CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_INIT_FINAL="${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_INIT_FINAL:-0.5}"
export CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_TARGET_FINAL="${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_TARGET_FINAL:-0.5}"
export CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_REG_WEIGHT="${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_REG_WEIGHT:-0.0}"
export CONCEPTLM_V21_COMPILE_RESIDUAL_FLOW_ROUTES="${CONCEPTLM_V21_COMPILE_RESIDUAL_FLOW_ROUTES:-1}"
export CONCEPTLM_V21_COMPILE_DD_ROUTES="${CONCEPTLM_V21_COMPILE_DD_ROUTES:-1}"
export ENABLE_MUDD_TORCH_COMPILE="${ENABLE_MUDD_TORCH_COMPILE:-0}"
export ENABLE_CONCEPTLM_V21_DECODER_TORCH_COMPILE="${ENABLE_CONCEPTLM_V21_DECODER_TORCH_COMPILE:-0}"
export MUDD_TORCH_COMPILE_BACKEND="${MUDD_TORCH_COMPILE_BACKEND:-inductor}"
export MUDD_TORCH_COMPILE_MODE="${MUDD_TORCH_COMPILE_MODE:-default}"
export MUDD_TORCH_COMPILE_FULLGRAPH="${MUDD_TORCH_COMPILE_FULLGRAPH:-0}"
export MUDD_TORCH_COMPILE_DYNAMIC="${MUDD_TORCH_COMPILE_DYNAMIC:-0}"
export CONCEPTLM_V21_ROUTE_TORCH_COMPILE_BACKEND="${CONCEPTLM_V21_ROUTE_TORCH_COMPILE_BACKEND:-inductor}"
export CONCEPTLM_V21_ROUTE_TORCH_COMPILE_MODE="${CONCEPTLM_V21_ROUTE_TORCH_COMPILE_MODE:-default}"
export CONCEPTLM_V21_ROUTE_TORCH_COMPILE_FULLGRAPH="${CONCEPTLM_V21_ROUTE_TORCH_COMPILE_FULLGRAPH:-0}"
export CONCEPTLM_V21_ROUTE_TORCH_COMPILE_DYNAMIC="${CONCEPTLM_V21_ROUTE_TORCH_COMPILE_DYNAMIC:-0}"
export CONCEPTLM_V21_SELF_DD_UNSTACKED_FASTPATH="${CONCEPTLM_V21_SELF_DD_UNSTACKED_FASTPATH:-1}"

export USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
export DATA_ARGS_PATH="${DATA_ARGS_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-6T/megatron_olmo3_8192/data_args_path.txt}"
export PER_DATASET_SEQUENCES_PATH="${PER_DATASET_SEQUENCES_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-6T/megatron_olmo3_8192/per_dataset_sequences.json}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/model/Olmo-3-1025-7B-stage1-step1413814}"
export SPLIT="${SPLIT:-100,0,0}"
export MMAP_BIN_FILES="${MMAP_BIN_FILES:-0}"

export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-transformer_engine}"
export OLMO3_LAYER_SPEC_NAME="${OLMO3_LAYER_SPEC_NAME:-olmo3_layer_spec}"
export OLMO3_QK_NORM_IMPL="${OLMO3_QK_NORM_IMPL:-te}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export FUSED_RESIDUAL_RMSNORM="${FUSED_RESIDUAL_RMSNORM:-1}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-8192}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export TRAIN_ITERS="${TRAIN_ITERS:-1430512}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-1430512}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2000}"
export LR="${LR:-6.0e-5}"
export MIN_LR="${MIN_LR:-6.0e-6}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export OPTIMIZER="${OPTIMIZER:-muon}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1.0e-8}"
export GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-0}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-0}"
export SAVE_FULL_STATE="${SAVE_FULL_STATE:-1}"
export EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-0}"
export MANUAL_GC_INTERVAL="${MANUAL_GC_INTERVAL:-10}"
export RERUN_MODE="${RERUN_MODE:-disabled}"
export DDP_BUCKET_SIZE="${DDP_BUCKET_SIZE:-240000000}"

export OLMO3_Z_LOSS_MULTIPLIER="${OLMO3_Z_LOSS_MULTIPLIER:-1e-5}"
export OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO="${OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO:-1}"
export HIDDEN_RANK_LOG_INTERVAL="${HIDDEN_RANK_LOG_INTERVAL:-500}"
export HIDDEN_RANK_LOG_TOKENS="${HIDDEN_RANK_LOG_TOKENS:-1024}"
export HIDDEN_RANK_RTOL="${HIDDEN_RANK_RTOL:-1e-3}"
export HIDDEN_RANK_CENTER="${HIDDEN_RANK_CENTER:-1}"

export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ConceptLM}"
export WANDB_ENTITY="${WANDB_ENTITY:-iammi-nanjing-university}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$JOB_NAME}"

export EXTRA_ARGS="${EXTRA_ARGS:-} \
  --conceptlm-backbone ${CONCEPTLM_BACKBONE} \
  --conceptlm-encoder-layers ${CONCEPTLM_ENCODER_LAYERS} \
  --conceptlm-decoder-layers ${CONCEPTLM_DECODER_LAYERS} \
  --conceptlm-special-layers ${CONCEPTLM_SPECIAL_LAYERS} \
  --conceptlm-chunk-size ${CONCEPTLM_CHUNK_SIZE} \
  --conceptlm-chunk-merge-method ${CONCEPTLM_CHUNK_MERGE_METHOD} \
  --conceptlm-chunk-dualpath-alpha-init ${CONCEPTLM_CHUNK_DUALPATH_ALPHA_INIT} \
  --conceptlm-layer-norm-option ${CONCEPTLM_LAYER_NORM_OPTION} \
  --conceptlm-fusion-norm-alpha-init ${CONCEPTLM_FUSION_NORM_ALPHA_INIT} \
  --conceptlm-fusion-alpha-init ${CONCEPTLM_FUSION_ALPHA_INIT} \
  --conceptlm-hlm-loss-weight ${CONCEPTLM_HLM_LOSS_WEIGHT} \
  --conceptlm-vq-loss-weight ${CONCEPTLM_VQ_LOSS_WEIGHT} \
  --conceptlm-v22-vq-codebook-size ${CONCEPTLM_V22_VQ_CODEBOOK_SIZE} \
  --conceptlm-v22-vq-num-codebooks ${CONCEPTLM_V22_VQ_NUM_CODEBOOKS} \
  --conceptlm-v22-vq-commitment-cost ${CONCEPTLM_V22_VQ_COMMITMENT_COST} \
  --conceptlm-v22-vq-merge-mode ${CONCEPTLM_V22_VQ_MERGE_MODE} \
  --conceptlm-v22-vq-hlm-loss-type ${CONCEPTLM_V22_VQ_HLM_LOSS_TYPE} \
  --conceptlm-v22-vq-lr-mult ${CONCEPTLM_V22_VQ_LR_MULT} \
  --conceptlm-v22-vq-highlevel-lr-mult ${CONCEPTLM_V22_VQ_HIGHLEVEL_LR_MULT} \
  --conceptlm-v21-dd-two-route-add-concept-source ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_SOURCE} \
  --conceptlm-v21-dd-two-route-add-beta-init ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_BETA_INIT} \
  --conceptlm-v21-dd-two-route-add-every-n-layers ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_EVERY_N_LAYERS} \
  --conceptlm-v21-dd-two-route-add-concept-route-first-n ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_ROUTE_FIRST_N} \
  --conceptlm-v21-dd-two-route-add-decoder-hidden-size ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_HIDDEN_SIZE} \
  --conceptlm-v21-dd-two-route-add-concept-hidden-size ${CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_HIDDEN_SIZE} \
  --conceptlm-v21-dd-encoder-self-dd-every-n-layers ${CONCEPTLM_V21_DD_ENCODER_SELF_DD_EVERY_N_LAYERS} \
  --conceptlm-v21-dd-encoder-self-dd-hidden-size ${CONCEPTLM_V21_DD_ENCODER_SELF_DD_HIDDEN_SIZE} \
  --conceptlm-v21-dd-concept-self-dd-every-n-layers ${CONCEPTLM_V21_DD_CONCEPT_SELF_DD_EVERY_N_LAYERS} \
  --conceptlm-v21-dd-concept-self-dd-hidden-size ${CONCEPTLM_V21_DD_CONCEPT_SELF_DD_HIDDEN_SIZE} \
  --conceptlm-v21-residual-flow-beta-init ${CONCEPTLM_V21_RESIDUAL_FLOW_BETA_INIT} \
  --conceptlm-v21-residual-flow-route-hidden-size ${CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_HIDDEN_SIZE} \
  --conceptlm-v21-final-read-concept-gate-init-final ${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_INIT_FINAL} \
  --conceptlm-v21-final-read-concept-gate-target-final ${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_TARGET_FINAL} \
  --conceptlm-v21-final-read-concept-gate-reg-weight ${CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE_REG_WEIGHT}"

if [[ -n "${CONCEPTLM_HLM_FFN_HIDDEN_SIZE}" ]]; then
  EXTRA_ARGS+=" --conceptlm-hlm-ffn-hidden-size ${CONCEPTLM_HLM_FFN_HIDDEN_SIZE}"
fi

[[ "$CONCEPTLM_SHIFT_FEATURE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-shift-feature" || EXTRA_ARGS+=" --no-conceptlm-shift-feature"
[[ "$CONCEPTLM_ENABLE_CHUNK_DUALPATH_SMOOTHING" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-enable-chunk-dualpath-smoothing" || EXTRA_ARGS+=" --no-conceptlm-enable-chunk-dualpath-smoothing"
[[ "$CONCEPTLM_V22_VQ_DETACH_HLM_TARGET" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v22-vq-detach-hlm-target" || EXTRA_ARGS+=" --no-conceptlm-v22-vq-detach-hlm-target"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_RAW_CONCEPT_ROUTE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-enable-raw-concept-route" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-enable-raw-concept-route"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_ENABLE_FINAL_CONCEPT_ROUTE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-enable-final-concept-route" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-enable-final-concept-route"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_USE_SOFTMAX" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-use-softmax" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-use-softmax"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DISABLE_DECODER_DD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-disable-decoder-dd" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-disable-decoder-dd"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-decoder-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-decoder-use-layernorm"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_DECODER_USE_SOFTMAX" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-decoder-use-softmax" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-decoder-use-softmax"
[[ "$CONCEPTLM_V21_DD_TWO_ROUTE_ADD_CONCEPT_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-two-route-add-concept-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-two-route-add-concept-use-layernorm"
[[ "$CONCEPTLM_V21_DD_ENCODER_SELF_DD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-encoder-self-dd" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-encoder-self-dd"
[[ "$CONCEPTLM_V21_DD_ENCODER_SELF_DD_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-encoder-self-dd-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-encoder-self-dd-use-layernorm"
[[ "$CONCEPTLM_V21_DD_CONCEPT_SELF_DD" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-concept-self-dd" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-concept-self-dd"
[[ "$CONCEPTLM_V21_DD_CONCEPT_SELF_DD_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-dd-concept-self-dd-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-dd-concept-self-dd-use-layernorm"
[[ "$CONCEPTLM_V21_ENABLE_FULL_RESIDUAL_FLOW" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-full-residual-flow" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-full-residual-flow"
[[ "$CONCEPTLM_V21_ENABLE_CONCEPT_READ_ENCODER" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-concept-read-encoder" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-concept-read-encoder"
[[ "$CONCEPTLM_V21_ENABLE_DECODER_READ_ENCODER" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-decoder-read-encoder" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-decoder-read-encoder"
[[ "$CONCEPTLM_V21_ENABLE_DECODER_READ_CONCEPT" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-enable-decoder-read-concept" || EXTRA_ARGS+=" --no-conceptlm-v21-enable-decoder-read-concept"
[[ "$CONCEPTLM_V21_RESIDUAL_FLOW_ROUTE_USE_SOFTMAX" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-residual-flow-route-use-softmax" || EXTRA_ARGS+=" --no-conceptlm-v21-residual-flow-route-use-softmax"
[[ "$CONCEPTLM_V21_RESIDUAL_FLOW_SOURCE_USE_LAYERNORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-residual-flow-source-use-layernorm" || EXTRA_ARGS+=" --no-conceptlm-v21-residual-flow-source-use-layernorm"
[[ "$CONCEPTLM_V21_RESIDUAL_FLOW_SHARED_SOURCE_NORM" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-residual-flow-shared-source-norm" || EXTRA_ARGS+=" --no-conceptlm-v21-residual-flow-shared-source-norm"
[[ "$CONCEPTLM_V21_FINAL_READ_CONCEPT_GATE" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-final-read-concept-gate" || EXTRA_ARGS+=" --no-conceptlm-v21-final-read-concept-gate"
[[ "$CONCEPTLM_V21_COMPILE_RESIDUAL_FLOW_ROUTES" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-compile-residual-flow-routes" || EXTRA_ARGS+=" --no-conceptlm-v21-compile-residual-flow-routes"
[[ "$CONCEPTLM_V21_COMPILE_DD_ROUTES" =~ ^(1|true|yes|on)$ ]] && EXTRA_ARGS+=" --conceptlm-v21-compile-dd-routes" || EXTRA_ARGS+=" --no-conceptlm-v21-compile-dd-routes"

mkdir -p "$RUN_ROOT"
cat <<EOF
Submitting ConceptLM V2.2-VQ OLMo3 job: $JOB_NAME
  layers: olmo3_backbone=${CONCEPTLM_BACKBONE} num_layers=32 encoder=${CONCEPTLM_ENCODER_LAYERS} decoder=${CONCEPTLM_DECODER_LAYERS} special=${CONCEPTLM_SPECIAL_LAYERS}
  hidden: hidden_size=4096 ffn_hidden_size=11008 heads=32 kv_channels=128 num_codebooks=${CONCEPTLM_V22_VQ_NUM_CODEBOOKS}
  optimizer=$OPTIMIZER
  checkpoint_dir=$CKPT_DIR
EOF
bash "$SUBMIT_SCRIPT" olmo3
