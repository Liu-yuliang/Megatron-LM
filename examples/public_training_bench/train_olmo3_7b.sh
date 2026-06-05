#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_DIR="${BASE_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/ylliu}"
RUN_ROOT="${RUN_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/public_training_bench/runs}"
RUN_NAME="${RUN_NAME:-olmo3-7b-${JOB_ID:-local}-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_NAME}"
HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/model/Olmo-3-1025-7B-stage1-step1413814}"
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

LOG_DIR="$RUN_DIR/logs"
RESULTS_DIR="$RUN_DIR/results"
CKPT_DIR="${CKPT_DIR:-$RUN_DIR/checkpoints}"
TB_DIR="${TB_DIR:-$RUN_DIR/tensorboard}"
WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$RUN_DIR/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUN_DIR/wandb_cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb_config}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$RUN_DIR/data_cache}"
TMP_ROOT="${TMP_ROOT:-$RUN_DIR/tmp}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$RUN_DIR/torch_extensions}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR" "$TB_DIR" "$WANDB_SAVE_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$DATA_CACHE_DIR" "$TMP_ROOT" "$TORCH_EXTENSIONS_DIR"

export HOME="${HOME:-$BASE_DIR}"
export TMPDIR="${TMPDIR:-/tmp/public_olmo3_${JOB_ID:-local}_$$}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TORCH_EXTENSIONS_DIR
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$WANDB_SAVE_DIR"
export WANDB_CACHE_DIR
export WANDB_CONFIG_DIR
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export HIDDEN_RANK_LOG_INTERVAL="${HIDDEN_RANK_LOG_INTERVAL:-500}"
export HIDDEN_RANK_LOG_TOKENS="${HIDDEN_RANK_LOG_TOKENS:-1024}"
export HIDDEN_RANK_RTOL="${HIDDEN_RANK_RTOL:-1e-3}"
export HIDDEN_RANK_CENTER="${HIDDEN_RANK_CENTER:-1}"
export HIDDEN_RANK_LOG_PATH="${HIDDEN_RANK_LOG_PATH:-$RESULTS_DIR/exit_hidden_rank.jsonl}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-pretrain_gpt.py}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"

MAIN_LOG="$LOG_DIR/train_${JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

cleanup() {
  rm -rf "$TMPDIR" 2>/dev/null || true
  if [[ "$(id -u)" == "0" ]]; then
    chown -R "$TARGET_UID:$TARGET_GID" "$RUN_DIR" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

setup_cuda_env() {
  local site nvidia_libs nvidia_includes nvidia_include_flags nvidia_lib_flags conda_target
  site="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

  if [[ -d "$site/nvidia/cuda_runtime" ]]; then
    if [[ ! -e "$site/nvidia/cudart" ]]; then
      ln -s cuda_runtime "$site/nvidia/cudart" 2>/dev/null || true
    fi
    export CUDART_HOME="$site/nvidia/cuda_runtime"
    export CUDART_PATH="$site/nvidia/cuda_runtime"
    export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-$site/nvidia/cuda_runtime/include}"
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
    export CUDA_PATH="$CUDA_HOME"
    export CUDART_HOME="${CUDART_HOME:-$CONDA_PREFIX}"
    export CUDART_PATH="${CUDART_PATH:-$CONDA_PREFIX}"
    export CUDACXX="$CONDA_PREFIX/bin/nvcc"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH:-}"
    export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
    conda_target="$CONDA_PREFIX/targets/x86_64-linux"
    if [[ -d "$conda_target" ]]; then
      export LD_LIBRARY_PATH="$conda_target/lib:$conda_target/lib64:${LD_LIBRARY_PATH:-}"
      export LIBRARY_PATH="$conda_target/lib:$conda_target/lib64:${LIBRARY_PATH:-}"
      export CPATH="$conda_target/include:${CPATH:-}"
    fi
    if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
      export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    fi
    if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]]; then
      export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    fi
  fi

  nvidia_libs="$(python - <<'PY'
import pathlib, site
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
paths = []
if root.exists():
    for child in sorted(root.iterdir()):
        for name in ("lib", "lib64"):
            p = child / name
            if p.exists():
                paths.append(str(p))
print(":".join(paths))
PY
)"
  if [[ -n "$nvidia_libs" ]]; then
    export LD_LIBRARY_PATH="$nvidia_libs:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$nvidia_libs:${LIBRARY_PATH:-}"
    nvidia_lib_flags="$(python - "$nvidia_libs" <<'PY'
import sys
print(" ".join(f"-L{p}" for p in sys.argv[1].split(":") if p))
PY
)"
    export LDFLAGS="${nvidia_lib_flags} ${LDFLAGS:-}"
  fi

  nvidia_includes="$(python - <<'PY'
import pathlib, site
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
paths = []
if root.exists():
    for p in sorted(root.glob("*/include")):
        paths.append(str(p))
print(":".join(paths))
PY
)"
  if [[ -n "$nvidia_includes" ]]; then
    export CPATH="$nvidia_includes:${CPATH:-}"
    export C_INCLUDE_PATH="$nvidia_includes:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="$nvidia_includes:${CPLUS_INCLUDE_PATH:-}"
    nvidia_include_flags="$(python - "$nvidia_includes" <<'PY'
import sys
print(" ".join(f"-I{p}" for p in sys.argv[1].split(":") if p))
PY
)"
    export CPPFLAGS="${nvidia_include_flags} ${CPPFLAGS:-}"
    export CFLAGS="${nvidia_include_flags} ${CFLAGS:-}"
    export CXXFLAGS="${nvidia_include_flags} ${CXXFLAGS:-}"
  fi

  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  export CUDAARCHS="${CUDAARCHS:-90}"
  export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"
  export MAX_JOBS="${MAX_JOBS:-8}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
  export NVCC_THREADS="${NVCC_THREADS:-4}"
}

activate_env() {
  if [[ -f "$BASE_DIR/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "$BASE_DIR/.bashrc" >/dev/null 2>&1 || true
    export HOME="${HOME:-$BASE_DIR}"
    if declare -F proxy_on >/dev/null 2>&1; then
      proxy_on || true
    fi
  fi
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
  export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
  conda activate "$CONDA_ENV_PREFIX"
  setup_cuda_env
}

probe() {
  log "Probe node"
  {
    echo "date=$(date -Is)"
    echo "repo_root=$REPO_ROOT"
    echo "run_dir=$RUN_DIR"
    echo "id=$(id)"
    echo "hostname=$(hostname)"
    echo "job_id=${JOB_ID:-}"
    echo "node_count=${NODE_COUNT:-}"
    echo "node_rank=${NODE_RANK:-}"
    echo "master_addr=${MASTER_ADDR:-}"
    echo "proc_per_node=${PROC_PER_NODE:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi || true
    df -h / /tmp "$RUN_DIR" 2>/dev/null || true
    free -h || true
  } | tee "$RESULTS_DIR/probe_${JOB_ID:-local}.txt"
}

verify_env() {
  activate_env
  cd "$REPO_ROOT"
  python - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
for name in ("transformer_engine", "apex", "flash_attn", "flash_attn_3", "flash_attn_interface"):
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "FAIL", repr(exc))
PY
}

write_arch_summary() {
  python - "$HF_MODEL_DIR/config.json" "$RESULTS_DIR/olmo3_arch_summary.txt" <<'PY'
import json, sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
full_layers = [i + 1 for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
expected = {
    "model_type": cfg["model_type"],
    "num_layers": cfg["num_hidden_layers"],
    "hidden_size": cfg["hidden_size"],
    "ffn_hidden_size": cfg["intermediate_size"],
    "num_attention_heads": cfg["num_attention_heads"],
    "num_key_value_heads": cfg["num_key_value_heads"],
    "vocab_size": cfg["vocab_size"],
    "max_position_embeddings": cfg["max_position_embeddings"],
    "rope_theta": cfg["rope_theta"],
    "rms_norm_eps": cfg["rms_norm_eps"],
    "sliding_window": cfg["sliding_window"],
    "full_attention_layers_1_indexed": full_layers,
    "megatron_window_args": "--window-size 4096,0 --window-attn-skip-freq 4",
}
lines = [f"{k}={v}" for k, v in expected.items()]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

build_data_args() {
  local use_mock="${USE_MOCK_DATA:-1}"
  if [[ "$use_mock" == "1" || ( -z "${DATA_PATH:-}" && -z "${DATA_ARGS_PATH:-}" ) ]]; then
    DATA_ARGS=(
      --mock-data
      --tokenizer-type NullTokenizer
      --vocab-size "${VOCAB_SIZE:-100278}"
      --null-tokenizer-eod-id "${EOS_TOKEN_ID:-100257}"
      --null-tokenizer-pad-id "${PAD_TOKEN_ID:-100277}"
      --data-cache-path "$DATA_CACHE_DIR/mock"
      --split 99,1,0
      --no-create-attention-mask-in-dataloader
      --num-workers 1
    )
  else
    DATA_ARGS=(
      --tokenizer-type HuggingFaceTokenizer
      --tokenizer-model "$HF_MODEL_DIR"
      --vocab-size "${VOCAB_SIZE:-100278}"
      --data-cache-path "${DATA_CACHE_PATH:-$DATA_CACHE_DIR/real}"
      --split "${SPLIT:-99,1,0}"
      --no-create-attention-mask-in-dataloader
      --num-workers "${NUM_WORKERS:-2}"
    )
    if [[ -n "${DATA_ARGS_PATH:-}" ]]; then
      DATA_ARGS+=(--data-args-path "$DATA_ARGS_PATH")
    else
      DATA_ARGS+=(--data-path "$DATA_PATH")
    fi
    if [[ -n "${PER_DATASET_SEQUENCES_PATH:-}" ]]; then
      DATA_ARGS+=(--per-dataset-sequences-path "$PER_DATASET_SEQUENCES_PATH")
    fi
    if [[ "${MMAP_BIN_FILES:-1}" == "0" ]]; then
      DATA_ARGS+=(--no-mmap-bin-files)
    fi
    if [[ "${DATALOADER_FAST_CACHE_LOAD:-0}" == "1" ]]; then
      DATA_ARGS+=(--dataloader-fast-cache-load)
    fi
    if [[ "${DATALOADER_DEFER_NPY_INDEX_MMAP:-0}" == "1" ]]; then
      DATA_ARGS+=(--dataloader-defer-npy-index-mmap)
    fi
  fi
}

run_train() {
  activate_env
  cd "$REPO_ROOT"

  local gpus_per_node node_count node_rank master_addr master_port world_size
  gpus_per_node="${GPUS_PER_NODE:-${PROC_PER_NODE:-8}}"
  node_count="${NODE_COUNT:-1}"
  node_rank="${NODE_RANK:-0}"
  master_addr="${MASTER_ADDR:-127.0.0.1}"
  master_port="${MASTER_PORT:-6028}"
  world_size=$((gpus_per_node * node_count))

  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export NVTE_DEBUG="${NVTE_DEBUG:-0}"
  export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-0}"

  if [[ "${NCCL_AUTO_CONFIG:-1}" == "1" ]]; then
    eval "$(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export)" || true
  fi

  local tp pp cp mbs gbs seq train_iters save_interval lr min_lr lr_decay_iters lr_warmup_iters attention_backend transformer_impl layer_spec_module layer_spec_name
  tp="${TP_SIZE:-1}"
  pp="${PP_SIZE:-1}"
  cp="${CP_SIZE:-1}"
  mbs="${MICRO_BATCH_SIZE:-1}"
  gbs="${GLOBAL_BATCH_SIZE:-128}"
  seq="${SEQ_LENGTH:-8192}"
  # OLMo 3 7B official-scaled defaults.
  # Official stage-1 uses GBS=512 sequences of 8192 tokens and 2000 warmup steps.
  # For the 150B-token Dolma3 pre-experiment we keep the same token/step and scale
  # warmup from the 6T-token reference budget: 2000 * 150B / 6T = 50 steps.
  train_iters="${TRAIN_ITERS:-35763}"
  save_interval="${SAVE_INTERVAL:-7153}"
  lr="${LR:-3.0e-4}"
  min_lr="${MIN_LR:-3.0e-5}"
  lr_decay_iters="${LR_DECAY_ITERS:-$train_iters}"
  if [[ -n "${LR_WARMUP_ITERS:-}" ]]; then
    lr_warmup_iters="$LR_WARMUP_ITERS"
  elif [[ "$train_iters" -le 1 ]]; then
    lr_warmup_iters=0
  else
    lr_warmup_iters=50
  fi
  transformer_impl="${TRANSFORMER_IMPL:-transformer_engine}"
  layer_spec_module="${OLMO3_LAYER_SPEC_MODULE:-megatron.core.models.gpt.olmo3_layer_specs}"
  if [[ "$transformer_impl" == "local" ]]; then
    layer_spec_name="${OLMO3_LAYER_SPEC_NAME:-olmo3_local_layer_spec}"
    attention_backend="${ATTENTION_BACKEND:-local}"
    export OLMO3_QK_NORM_IMPL="${OLMO3_QK_NORM_IMPL:-torch}"
  else
    layer_spec_name="${OLMO3_LAYER_SPEC_NAME:-olmo3_layer_spec}"
    attention_backend="${ATTENTION_BACKEND:-flash}"
    export OLMO3_QK_NORM_IMPL="${OLMO3_QK_NORM_IMPL:-te}"
  fi

  build_data_args
  mkdir -p "$CKPT_DIR" "$TB_DIR" "$DATA_CACHE_DIR/mock" "$DATA_CACHE_DIR/real"

  local -a distributed_args model_args train_args parallel_args logging_args checkpoint_args
  distributed_args=(
    --nproc_per_node "$gpus_per_node"
    --nnodes "$node_count"
    --node_rank "$node_rank"
    --master_addr "$master_addr"
    --master_port "$master_port"
  )

  model_args=(
    --use-mcore-models
    --transformer-impl "$transformer_impl"
    --spec "$layer_spec_module" "$layer_spec_name"
    --num-layers "${NUM_LAYERS:-32}"
    --hidden-size "${HIDDEN_SIZE:-4096}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE:-11008}"
    --num-attention-heads "${NUM_ATTENTION_HEADS:-32}"
    --kv-channels "${KV_CHANNELS:-128}"
    --seq-length "$seq"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-8192}"
    --position-embedding-type rope
    --rotary-base "${ROTARY_BASE:-500000}"
    --rotary-percent "${ROTARY_PERCENT:-1.0}"
    --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-1}"
    --window-size "${WINDOW_SIZE:-4096,0}"
    --window-attn-skip-freq "${WINDOW_ATTN_SKIP_FREQ:-4}"
    --qk-layernorm
    --swiglu
    --normalization RMSNorm
    --norm-epsilon "${NORM_EPSILON:-1e-6}"
    --attention-dropout "${ATTENTION_DROPOUT:-0.0}"
    --hidden-dropout "${HIDDEN_DROPOUT:-0.0}"
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --init-method-std "${INIT_METHOD_STD:-0.02}"
    --init-method-variant "${INIT_METHOD_VARIANT:-scaled_truncated_normal}"
  )

  train_args=(
    --micro-batch-size "$mbs"
    --global-batch-size "$gbs"
    --train-iters "$train_iters"
    --lr-decay-iters "$lr_decay_iters"
    --lr-warmup-iters "$lr_warmup_iters"
    --lr "$lr"
    --min-lr "$min_lr"
    --lr-decay-style cosine
    --clip-grad "${CLIP_GRAD:-1.0}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --optimizer "${OPTIMIZER:-adam}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.95}"
    --adam-eps "${ADAM_EPS:-1.0e-8}"
    --bf16
    --cross-entropy-loss-fusion
  )
  if [[ "${CALCULATE_PER_TOKEN_LOSS:-1}" == "1" ]]; then
    train_args+=(--calculate-per-token-loss)
  fi
  if [[ "${GRAD_REDUCE_IN_BF16:-1}" == "1" ]]; then
    train_args+=(--grad-reduce-in-bf16)
  fi

  parallel_args=(
    --tensor-model-parallel-size "$tp"
    --pipeline-model-parallel-size "$pp"
    --context-parallel-size "$cp"
    --attention-backend "$attention_backend"
    --use-distributed-optimizer
    --overlap-grad-reduce
    --ddp-bucket-size "${DDP_BUCKET_SIZE:-$([[ "$node_count" -ge 16 ]] && echo 480000000 || echo 240000000)}"
    --ddp-pad-buckets-for-high-nccl-busbw
  )
  if [[ "${OVERLAP_PARAM_GATHER:-1}" == "1" ]]; then
    parallel_args+=(--overlap-param-gather)
  fi
  if [[ "${FUSED_RESIDUAL_RMSNORM:-$([[ "$transformer_impl" == "local" ]] && echo 0 || echo 1)}" == "1" ]]; then
    parallel_args+=(--fused-residual-rmsnorm)
  fi

  if [[ "${SEQUENCE_PARALLEL:-0}" == "1" || "$tp" -gt 1 ]]; then
    parallel_args+=(--sequence-parallel)
  fi

  logging_args=(
    --log-interval "${LOG_INTERVAL:-1}"
    --eval-iters "${EVAL_ITERS:-0}"
    --eval-interval "${EVAL_INTERVAL:-1000}"
    --rerun-mode "${RERUN_MODE:-disabled}"
    --log-throughput
    --log-timers-to-tensorboard
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
    --tensorboard-dir "$TB_DIR"
    --manual-gc
    --manual-gc-interval "${MANUAL_GC_INTERVAL:-10}"
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL:-0}"
  )
  if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
    logging_args+=(
      --wandb-project "${WANDB_PROJECT:-ConceptLM}"
      --wandb-exp-name "${WANDB_EXP_NAME:-$RUN_NAME}"
      --wandb-save-dir "$WANDB_SAVE_DIR"
    )
    if [[ -n "${WANDB_ENTITY:-iammi-nanjing-university}" ]]; then
      logging_args+=(--wandb-entity "${WANDB_ENTITY:-iammi-nanjing-university}")
    fi
  fi

  checkpoint_args=(
    --save "$CKPT_DIR"
    --save-interval "$save_interval"
    --ckpt-format "${CKPT_FORMAT:-torch_dist}"
  )
  if [[ -n "${LOAD_DIR:-}" ]]; then
    checkpoint_args+=(--load "$LOAD_DIR")
  fi
  if [[ "${SAVE_FULL_STATE:-0}" != "1" ]]; then
    checkpoint_args+=(--no-save-optim --no-save-rng)
  fi

  local manifest="$RESULTS_DIR/launch_manifest_${JOB_ID:-local}.txt"
  {
    echo "date=$(date -Is)"
    echo "repo_root=$REPO_ROOT"
    echo "run_dir=$RUN_DIR"
    echo "hf_model_dir=$HF_MODEL_DIR"
    echo "world_size=$world_size"
    echo "gpus_per_node=$gpus_per_node"
    echo "node_count=$node_count"
    echo "node_rank=$node_rank"
    echo "master_addr=$master_addr"
    echo "master_port=$master_port"
    echo "precision=bf16"
    echo "transformer_impl=$transformer_impl"
    echo "layer_spec=$layer_spec_module $layer_spec_name"
    echo "olmo3_qk_norm_impl=$OLMO3_QK_NORM_IMPL"
    echo "attention_backend=$attention_backend"
    echo "init_method_variant=${INIT_METHOD_VARIANT:-scaled_truncated_normal}"
    echo "init_method_std=${INIT_METHOD_STD:-0.02}"
    echo "wandb_enabled=${ENABLE_WANDB:-1}"
    echo "wandb_mode=$WANDB_MODE"
    echo "wandb_project=${WANDB_PROJECT:-ConceptLM}"
    echo "wandb_exp_name=${WANDB_EXP_NAME:-$RUN_NAME}"
    echo "wandb_save_dir=$WANDB_SAVE_DIR"
    echo "optimizer=${OPTIMIZER:-adam}"
    echo "hidden_rank_log_interval=$HIDDEN_RANK_LOG_INTERVAL"
    echo "hidden_rank_log_tokens=$HIDDEN_RANK_LOG_TOKENS"
    echo "hidden_rank_rtol=$HIDDEN_RANK_RTOL"
    echo "hidden_rank_center=$HIDDEN_RANK_CENTER"
    echo "hidden_rank_log_path=$HIDDEN_RANK_LOG_PATH"
    echo "rerun_mode=${RERUN_MODE:-disabled}"
    echo "olmo3_z_loss_multiplier=${OLMO3_Z_LOSS_MULTIPLIER:-0}"
    echo "olmo3_embedding_weight_decay_zero=${OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO:-0}"
    echo "grad_reduce_in_bf16=${GRAD_REDUCE_IN_BF16:-1}"
    echo "save_full_state=${SAVE_FULL_STATE:-0}"
    echo "tp=$tp pp=$pp cp=$cp mbs=$mbs gbs=$gbs seq=$seq train_iters=$train_iters"
    echo "checkpoint_dir=$CKPT_DIR"
    echo "train_entrypoint=$TRAIN_ENTRYPOINT"
    echo "load_dir=${LOAD_DIR:-}"
    echo "save_interval=$save_interval"
    printf 'torchrun_args=%q ' "${distributed_args[@]}"; echo
    printf 'model_args=%q ' "${model_args[@]}"; echo
    printf 'train_args=%q ' "${train_args[@]}"; echo
    printf 'parallel_args=%q ' "${parallel_args[@]}"; echo
    printf 'data_args=%q ' "${DATA_ARGS[@]}"; echo
    printf 'logging_args=%q ' "${logging_args[@]}"; echo
    printf 'checkpoint_args=%q ' "${checkpoint_args[@]}"; echo
  } | tee "$manifest"

  local train_log="$LOG_DIR/${TRAIN_ENTRYPOINT%.py}_rank${node_rank}_${JOB_ID:-local}.log"
  log "Launching OLMo 3 7B training, log=$train_log"
  set +e
  torchrun "${distributed_args[@]}" "$TRAIN_ENTRYPOINT" \
    "${model_args[@]}" \
    "${train_args[@]}" \
    "${parallel_args[@]}" \
    "${DATA_ARGS[@]}" \
    "${logging_args[@]}" \
    "${checkpoint_args[@]}" \
    ${EXTRA_ARGS:-} > "$train_log" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$RESULTS_DIR/train_exit_code_rank${node_rank}.txt"
  ln -sfn "$(basename "$train_log")" "$LOG_DIR/latest_train_rank${node_rank}.log"
  if [[ "$node_rank" == "0" ]]; then
    find "$CKPT_DIR" -maxdepth 5 -type f | sort > "$RESULTS_DIR/checkpoint_files.txt" || true
  fi
  return "$rc"
}

main() {
  probe
  verify_env
  write_arch_summary
  run_train
}

main "$@"
