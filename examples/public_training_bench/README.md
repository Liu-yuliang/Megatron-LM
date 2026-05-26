# Public Training Bench

This directory contains shared Megatron launchers for H200 BF16 training benchmarks.

Scripts:

- `train_olmo3_7b.sh`: OLMo 3 7B, aligned to `/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/model/Olmo-3-1025-7B-stage1-step1413814`.
- `train_pythia.sh`: Pythia sizes `160m`, `410m`, `1b`, `1.4b`, `2.8b`, `6.9b`, `12b`, aligned to `/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/hnet/model`. It uses `megatron.core.models.gpt.pythia_layer_specs` to match GPT-NeoX/Pythia parallel residual blocks.
- `train_concept_olmo3_7b.sh`: ConceptLM V1 on the OLMo 3 7B backbone, inserted after embeddings before decoder layer 0.
- `train_concept_pythia_7b.sh`: ConceptLM V1 on the Pythia 6.9B backbone, inserted after embeddings before decoder layer 0.
- `submit_rjob.sh`: RJob wrapper for allocated GPU nodes.
- `validate_model_config.py`: lightweight HF config and Megatron log checker.

Defaults are mixed BF16, Transformer Engine, FlashAttention via `--attention-backend flash`, distributed optimizer, overlap grad/param communication, offline W&B logging, and `EMPTY_UNUSED_MEMORY_LEVEL=0`. OLMo defaults are explicitly pinned to the aligned OLMo 3 reference config: `seq-length=8192`, `max-position-embeddings=8192`, `TP_SIZE=1`, `--spec megatron.core.models.gpt.olmo3_layer_specs olmo3_layer_spec`, `--qk-layernorm`, `--window-size 4096,0 --window-attn-skip-freq 4`, and fused residual RMSNorm.

ConceptLM V1 launchers default to `chunk_size=4`, `special_layers=2`, `codebook_size=128`, and `num_codebooks=num_attention_heads`. They keep baseline W&B/logging keys and add `conceptlm ncp loss`, `conceptlm vq loss`, `conceptlm aux loss`, and `conceptlm total loss`.

W&B is local/offline by default: `WANDB_MODE=offline`, project `public-training-bench`, and files under each run's `wandb/`, `wandb_cache/`, and `wandb_config/` directories. Set `ENABLE_WANDB=0` to disable it. With the current Megatron logger it records train loss keys, learning rate, batch size, grad norm, loss scale, iteration time, throughput TFLOP/s/GPU, `tps-per-gpu`, `mfu-h200-percent`, timers, and checkpoint artifacts when checkpoints are saved.

The launcher also logs decoder-exit hidden-state rank every 500 steps by default. It samples up to `HIDDEN_RANK_LOG_TOKENS=1024` local tokens on the final rank before the output projection, computes centered singular-value diagnostics, writes W&B/TensorBoard metrics under `exit_hidden/*`, and appends JSONL records to `results/exit_hidden_rank.jsonl`. Override with `HIDDEN_RANK_LOG_INTERVAL`, `HIDDEN_RANK_LOG_TOKENS`, `HIDDEN_RANK_RTOL`, or set `HIDDEN_RANK_LOG_INTERVAL=0` to disable.

Examples:

```bash
# Submit OLMo 3 7B, 8xH200, 50 steps, mock data, save one checkpoint at step 50.
MODEL=olmo3 TRAIN_ITERS=50 SAVE_INTERVAL=50 \
  bash examples/public_training_bench/submit_rjob.sh

# Submit Pythia 160M.
MODEL=pythia PYTHIA_SIZE=160m TRAIN_ITERS=50 SAVE_INTERVAL=50 \
  bash examples/public_training_bench/submit_rjob.sh

# Submit ConceptLM V1 on Pythia 6.9B.
TRAIN_ITERS=50 SAVE_INTERVAL=50 USE_MOCK_DATA=1 \
  bash examples/public_training_bench/train_concept_pythia_7b.sh

# Run inside an already allocated container.
TRAIN_ITERS=50 SAVE_INTERVAL=50 USE_MOCK_DATA=1 \
  bash examples/public_training_bench/train_olmo3_7b.sh

# Real indexed Megatron data prefix.
USE_MOCK_DATA=0 DATA_PATH=/path/to/megatron_text_document \
  bash examples/public_training_bench/train_olmo3_7b.sh
```

Outputs are written under `${RUN_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/public_training_bench/runs}` with `logs/`, `results/`, `checkpoints/`, and `tensorboard/` subdirectories.
