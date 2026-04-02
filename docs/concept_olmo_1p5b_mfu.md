# Concept-OLMo 1.5B MFU Prototype

## 改了什么

这次改动只发生在 `/data/ConceptLM/Megatron-LM`，没有碰你原来的 `Concept-OLMo-core` 目录。

新增了一套独立实验实现：

- `megatron/core/models/concept_olmo/model.py`
  - 新增 `ConceptOLMoModel`
  - 结构按你当前训练链路对齐为：
    - `encoder_layers=0`
    - `decoder_layers=16`
    - `hlm_layers=2`
    - `chunk_size=4`
    - `vq_patch_ratio=1`
    - `codebook_size=128`
  - 主链路是：
    - token embedding
    - main hidden states
    - chunk mean-pooling
    - per-head VQ
    - HLM transformer
    - softmax over codebook and weighted code reconstruction
    - repeat 回 token 级别
    - 和 encoder hidden states 做 RMSNorm 后相加
    - 进入 decoder blocks
    - vocab projection
  - loss 为：
    - token CE
    - VQ commit loss
    - HLM MSE loss

- `concept_olmo_builder.py`
  - 新增 Megatron builder
  - 增加了 `--concept-*` 一组参数
  - 可以直接走 Megatron 的 `pretrain()` 框架

- `pretrain_concept_olmo.py`
  - 新增独立训练入口
  - 复用了 Megatron 原来的 GPT batch / dataset provider
  - 自定义了 Concept-OLMo 的 loss 聚合与 metric 上报

- `scripts/run_concept_olmo_1p5b_mfu.sh`
  - 新增启动脚本
  - 用环境变量传数据、tokenizer、batch 等配置

## 当前限制

这版是“先跑通、先看 MFU”的原型，不是把你原来的工程一比一搬过来。

当前我显式限制了：

- `pipeline-model-parallel-size=1`
- `context-parallel-size=1`
- `sequence-parallel=false`
- 暂不支持 Megatron 的 PP/CP 复杂切分

原因很直接：Concept 分支里有 chunk 聚合、VQ、HLM、repeat 回 token 级别，这部分如果直接做 PP/CP 拆分，先要把跨 stage / 跨 shard 的 hidden state 语义理顺，不然结果会错。

所以这版适合先做：

- 单机多卡
- 纯 DP / 可选 TP
- 先看 Megatron 框架下的吞吐和 MFU

## 启动方式

先准备环境变量：

```bash
cd /data/ConceptLM/Megatron-LM

export TOKENIZER_MODEL=/path/to/tokenizer
export DATA_PATH=/path/to/data
export NPROC_PER_NODE=8
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=512
```

然后启动：

```bash
bash /data/ConceptLM/Megatron-LM/scripts/run_concept_olmo_1p5b_mfu.sh
```

常用可调项：

- `TP_SIZE`
- `TRAIN_ITERS`
- `LR`
- `MIN_LR`
- `LR_WARMUP_ITERS`
- `CHECKPOINT_PATH`
- `LOG_DIR`
- `RUN_NAME`

## 我保留的命名和参数

为了和你现在的 Concept-OLMo 习惯保持一致，默认参数就是你当前 64 卡脚本里的这一组：

- `encoder=0`
- `decoder=16`
- `hlm=2`
- `chunk=4`
- `codebook=128`
- `distribution_merge` 等细节没有原样复刻成老逻辑，而是改成了更稳定的 `softmax -> weighted code reconstruction`

这部分是有意的。你现在目标是先看 Megatron 下的 MFU 和整体吞吐，不是先去复刻旧实现里的所有历史细节和已知问题。
