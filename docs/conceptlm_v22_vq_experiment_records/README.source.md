# ConceptLM V2.2-VQ OLMo3 7B Scale-Up Run

This directory organizes the 64-GPU Muon scale-up run:

`c22vq-muon64-s8-lr6e5-scaled-opg0-0608234919`

## Verdict

Your chain is basically correct:

1. ConceptLM V2.2-VQ wrapper:
   `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2/examples/public_training_bench/train_concept_v22_vq_olmo3_7b.sh`
2. RJob submit wrapper:
   `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2/examples/public_training_bench/submit_rjob.sh`
3. Container torchrun entry:
   `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2/examples/public_training_bench/train_olmo3_7b.sh`

There is also a convenience launch entry:

`/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2/examples/public_training_bench/launch_concept_v22_vq_olmo3_128g.sh`

The concrete run record is:

`/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_muon_64g_special8_lr6e5_scaledtrunc_opg0_retry_0608234919/run/rjob_submissions/submit_c22vq-muon64-s8-lr6e5-scaled-opg0-0608234919.log`

## Important correction

The scaled init for this run is confirmed in the launch manifest:

`init_method_variant=scaled_truncated_normal`

That value is consumed by `train_olmo3_7b.sh` through `INIT_METHOD_VARIANT`. It is not hard-coded as the default inside `train_concept_v22_vq_olmo3_7b.sh`, so the reusable submit script here sets it explicitly.

## Organized Files

- `submit_scaledtrunc_64g_muon_lr6e5.sh`: reusable submit entry for the scaled-truncated-normal 64-GPU run. It launches through the copied `repo/` tree in this directory.
- `repo/`: full runnable code/script snapshot copied from the jyhuang repo, excluding `.git`, pycache, run outputs, checkpoints, W&B files, caches, and virtual environments.
- `source_snapshot/`: small historical source/script snapshot from the jyhuang repo.
- `run_records/`: copied submit log, status file if present, and launch manifest for the specific run.

## Key Run Settings

- Model: ConceptLM V2.2-VQ on OLMo3 7B
- GPUs: `8 replicas x 8 GPUs = 64 GPUs`
- Optimizer: `muon`
- LR: `6.0e-5`
- Min LR: `6.0e-6`
- Weight decay: `0.1`
- Global batch size: `512`
- Micro batch size: `1`
- Sequence length: `8192`
- Train iters: `1430512`
- Special layers: `8`
- Encoder/special/decoder layers: `16 / 8 / 16`
- Init: `scaled_truncated_normal`
- `OVERLAP_PARAM_GATHER=0`
- `GRAD_REDUCE_IN_BF16=0`
- `SAVE_FULL_STATE=1`
- `OLMO3_EMBEDDING_WEIGHT_DECAY_ZERO=1`

## 监控计划

当前这次运行对应的集群 RJob 是：

`c22vq-muon64-s8-lr6e5-scaled-opg0-0608234919-152d6`

这个 live job 实际运行用的是：

`/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2`

本目录下的 `repo/` 是用于分析和后续重提任务的本地工作快照。这里的代码改动不会影响已经在跑的 job，除非之后从这个快照重启或重提任务。当前正在跑的任务仍然可以加外部日志/文件监控。

### 自建监控系统

本地快照已经加了独立于 W&B/TensorBoard 的文件监控系统。V2.2-VQ wrapper 默认开启，统一由 `motitor_step` / `monitor_step` 控制记录频率，默认值是 `1000`；16 卡时延测评脚本会传 `2`，所以每 2 step 写一次。

兼容的参数名如下，优先级从高到低：

- `CONCEPTLM_MONITOR_STEP`
- `MONITOR_STEP`
- `monitor_step`
- `MOTITOR_STEP`
- `motitor_step`
- `CONCEPTLM_MONITOR_INTERVAL`，旧名字，仅作为兼容别名

所有新增的自建监控都应该走同一个 monitor helper，由这个 step 参数统一控制频率。`monitor_step <= 0` 时不写 step 监控；`CONCEPTLM_MONITOR_ENABLE=0` 时关闭整套自建监控。

运行目录结构：

```text
<CONCEPTLM_MONITOR_ROOT>/<model_name>_<startup_time>_run/
  training_context_rank*.txt
  python_runtime_rank*.json
  latency.jsonl
  rank_latency.jsonl
  route_latency.jsonl
  grad_update_latency.jsonl
  step_0000001000/
    vq_summary.json
    tensor_health.json
    codebook_counts.json
    latency.json
    rank_summary.json
    rank_latency.json
    route_summary.json
    route_latency.json
    grad_update_summary.json
    grad_update_latency.json
```

当前 VQ/HLM MSE 监控入口在：

`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/repo/megatron/core/models/gpt/conceptlm_custom_monitor.py`

### 当前已有监控

- RJob 状态：`rjob list` / `rjob get` 可以看 replica 状态、节点分布，以及 8 个 replica 是否都保持 active。
- 训练 stdout 日志：运行目录的 `logs/` 下有各 rank 日志，final rank 会打印 iteration、ETA、LR、loss、throughput、TPS、MFU、grad norm、skipped iterations、nan iterations。
- TensorBoard：`tensorboard/` 下有 event file；当前 job 已经在写 event。
- Offline W&B：`ENABLE_WANDB=1` 时，会在 run 目录下写 `wandb/`、`wandb_cache/`、`wandb_config/`。
- Megatron 基础指标：`lm loss`、learning rate、batch size、loss scale、grad norm、num zeros、可选 params norm、iteration time、throughput、`tps-per-gpu`、global token throughput、consumed tokens、`mfu-h200-percent`。
- ConceptLM V2.2-VQ loss：`conceptlm_v22_vq total loss`、VQ/HLM raw loss、VQ/HLM effective loss、aux loss、route regularization loss。
- ConceptLM V2.2-VQ 标量指标：usable sequence length、concept chunks、codebook size、num codebooks、head dimension、`code_id_mean`、`code_id_max`。
- route/gate 指标：route projection norm/diagonal mean，以及 final/read-concept gate 的 mean。
- debug-only 保护：non-finite loss summary，以及 `CONCEPTLM_V22_VQ_CE_DEBUG*` 这类重型 CE/logit 扫描开关。
- 自建 VQ/HLM MSE 监控：每 `motitor_step` step 写一次 `vq_summary.json`、`tensor_health.json`、`codebook_counts.json`、`latency.json`，并在 run 根目录追加 `latency.jsonl`。
- 自建 rank/各向异性监控：每 `motitor_step` step 写一次 `rank_summary.json`、`rank_latency.json`，并在 run 根目录追加 `rank_latency.jsonl`。这个监控和 checkpoint 保存频率不绑定。
- 自建 route/gate 监控：每 `motitor_step` step 写一次 `route_summary.json`、`route_latency.json`，并在 run 根目录追加 `route_latency.jsonl`。这个监控和 checkpoint 保存频率不绑定。
- 自建 grad/update 监控：每 `motitor_step` step 写一次 `grad_update_summary.json`、`grad_update_latency.json`，并在 run 根目录追加 `grad_update_latency.jsonl`。这个监控在 optimizer step 前采样参数和梯度，在 optimizer step 后计算实际 update。

### Rank 监控

NITP 里讨论的 rank 这里按 last hidden state 的 effective rank 实现：对有效 token hidden states 做 mean-center，计算协方差谱，把特征值归一化成概率分布后取 entropy effective rank。同时记录 paired cosine similarity，用来辅助判断表征是否过度各向异性。MSE 训练下这里没有 top-k accuracy；`embed_in` 是离散 token id，所以记录 token 分布摘要，不做 SVD rank。

当前 `rank_summary.json` 包含：

- `embed_in`：输入 token id 的 shape、unique token 数、unique ratio、token id min/max，以及 labels 的 token id 摘要。
- `activation_effective_rank`：`embed_out`、`encoder_final`、`concept_vq_hidden`、`decoder_input_after_fusion`、`decoder_final_hidden` 等关键激活的 effective rank、stable rank、top eigenvalue share、finite/nan/inf 计数、sample token 数。
- `paired_cosine`：上述关键激活的 token pair cosine mean/std/min/max。
- `layerwise_effective_rank`：分三组记录层级 rank：`encoder`、`highlevel`、`decoder`。
- `layerwise_paired_cosine`：同样分三组记录层级 paired cosine。
- `primary`：默认指向 `decoder_final_hidden` 的 effective rank 和 paired cosine mean，方便后续画主曲线。

层选择由环境变量控制，支持逗号列表、区间、`all`/`*` 和负数索引：

- `CONCEPTLM_MONITOR_RANK_ENCODER_LAYERS`，默认 `0,4,8,12,15`
- `CONCEPTLM_MONITOR_RANK_HIGHLEVEL_LAYERS`，默认 `0,2,4,6,7`
- `CONCEPTLM_MONITOR_RANK_DECODER_LAYERS`，默认 `0,4,8,12,15`
- `CONCEPTLM_MONITOR_RANK_LAYERS` 是兼容用的全局 fallback；如果上面某一组没有单独设置，就用它。

rank 计算开销相关参数：

- `CONCEPTLM_MONITOR_RANK_MAX_TOKENS`：每个 activation 最多抽样 token 数，当前 16 卡测评用 `512`。
- `CONCEPTLM_MONITOR_RANK_TOPK`：记录前多少个归一化特征值占比，当前默认 `16`。
- `CONCEPTLM_MONITOR_RANK_EPS`：epsilon rank 阈值，当前默认 `1e-12`。
- `CONCEPTLM_MONITOR_COSINE_PAIRS`：paired cosine 最多抽样 pair 数，当前测评用 `1024`。

16 卡 layerwise rank 监控测评：

- RJob：`c22vq-layerank16-0609223750-73129281`，状态 `Succeeded`。
- run 目录：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/monitor16_0609223750/custom_monitor/c22vq-layerank16-0609223750_20260609_223750_run`
- checkpoint 目录：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/monitor16_0609223750/checkpoints/iter_0000008`
- 测评设置：`motitor_step=2`，`CONCEPTLM_MONITOR_RANK_MAX_TOKENS=512`，每组 layer 选 5 层，三组一共 15 层。
- step 2/4/6/8 都写出了 `rank_summary.json`，每个 step 都有 `encoder/highlevel/decoder` 三组 layerwise rank 和 cosine。
- rank log 时延：step 2 为 `926.82 ms`，step 4/6/8 为 `619.89 / 584.95 / 584.65 ms`，去掉第一次 warmup 后平均 `596.50 ms`。
- checkpoint 只在 `iter_0000008` 出现，说明自建 log 频率和 checkpoint 保存频率没有绑定。

### Route/Gate 监控

route/gate 监控用于看 ConceptLM 里不同路径的读写权重是否塌缩、是否只读某一路、以及 route 相关参数矩阵是否健康。它和 VQ、rank 一样由同一个 `motitor_step` / `monitor_step` 控制频率。

当前 `route_summary.json` 包含：

- `route_config`：本次模型里 concept route、decoder read encoder/concept、final read concept gate 等开关和结构配置。
- `final_read_concept_gate`：final hidden 与 read-concept 的 gate 权重，包含 mean/std/min/max、接近 0/1 的比例、target MSE 和逐层权重。
- `dynamic_route_records`：训练 forward 中实际经过的 route 权重，包含 DD route、highlevel read encoder、decoder read encoder/concept、dd_two_route、concept_route 等记录。
- `route_parameter_health`：route/gate 相关参数的 finite、nan/inf、norm、RMS、max_abs 等健康摘要。
- `route_matrix_spectrum`：可选 SVD，对匹配到的二维大矩阵记录 singular value 谱、effective rank、stable rank、epsilon rank、condition number、top singular energy shares。
- `route_matrix_spectrum_settings`：本次 SVD 是否开启、匹配到多少矩阵、实际算了几个、跳过几个。

route 权重摘要里重点看：

- `entropy_mean` / `perplexity_mean`：越低说明越集中到少数 source。
- `max_share_mean` / `max_share_max`：越高说明越接近单一路由。
- `collapse_ratio_max_share_gt_0_95`：超过 0.95 的 route 占比，直接看塌缩比例。
- `argmax_counts` / `argmax_fractions`：看最终主要选了哪一路 source。

route 监控开销相关参数：

- `CONCEPTLM_MONITOR_ROUTE_WEIGHT_MAX_TOKENS`：每个 route weight 最多抽样多少行，默认 `4096`。
- `CONCEPTLM_MONITOR_ROUTE_PARAM_PATTERNS`：哪些参数名进入 route 参数健康检查，默认匹配 route/gate 相关模块名。
- `CONCEPTLM_MONITOR_ROUTE_SVD`：是否对匹配到的二维参数矩阵做 SVD，默认开启。
- `CONCEPTLM_MONITOR_ROUTE_SVD_MAX_MATRICES`：每次最多做几个矩阵 SVD，默认 `8`；16 卡短测默认 `2`。
- `CONCEPTLM_MONITOR_ROUTE_SVD_TOPK`：记录前多少个归一化 singular energy share，默认 `16`。
- `CONCEPTLM_MONITOR_ROUTE_SVD_EPS`：epsilon rank 阈值，默认 `1e-12`。
- `CONCEPTLM_MONITOR_ROUTE_SVD_NAME_PATTERNS`：SVD 矩阵名过滤；16 卡短测默认 `route,proj`。

16 卡 route/gate 监控测评：

- 动态 route 权重验证 RJob：`c22vq-routemon16-0609233749-71756741`，手动 stop 前已写出 step 2/4。
- 动态 route run 目录：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/monitor16_0609233749/custom_monitor/c22vq-routemon16-0609233749_20260609_233749_run`
- step 2/4 都写出 `route_summary.json` 和 `route_latency.json`，每步 `dynamic_route_record_count=96`。
- route 动态记录时延：step 2 为 `88.45 ms`，step 4 为 `81.79 ms`。这轮当时参数 SVD 未命中，因为空字符串 env 覆盖了默认参数匹配；代码已修成空字符串按默认匹配处理。
- 手动 stop 前 checkpoint 目录为空，但 step 2/4 route log 已存在，说明自建 route log 不依赖 checkpoint 保存频率。

16 卡 route/gate 大矩阵 SVD 测评：

- RJob：`c22vq-routesvd16-0609235104-67443519`，状态 `Succeeded`。
- run 目录：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/monitor16_0609235104/custom_monitor/c22vq-routesvd16-0609235104_20260609_235104_run`
- 测评设置：`TRAIN_ITERS=2`，`motitor_step=2`，`CONCEPTLM_MONITOR_RANK_MAX_TOKENS=64`，每组 rank layer 只选第 0 层，`CONCEPTLM_MONITOR_ROUTE_SVD_MAX_MATRICES=2`。
- step 2 写出 VQ、rank、route 三组文件，根目录追加 `latency.jsonl`、`rank_latency.jsonl`、`route_latency.jsonl`。
- `route_summary.json` 中 `dynamic_route_record_count=96`，`route_parameter_health` 记录 `217` 个参数，SVD 匹配到 `152` 个二维 route 参数矩阵，按上限实际计算 `2` 个。
- 实际 SVD 矩阵：`concept_predictor.concept_read_encoder_routes.0.residual_proj.weight` 和 `concept_predictor.concept_read_encoder_routes.1.residual_proj.weight`，shape 都是 `[4096, 4096]`。
- route/SVD log 时延：route 总耗时 `203.32 ms`，其中 route compute `196.80 ms`，写文件 `6.20 ms`。同一个 step 的 rank 总耗时 `64.26 ms`，VQ 总耗时 `237.93 ms`。
- checkpoint 目录出现 `iter_0000002`，这是 `TRAIN_ITERS=2` 到训练结束后的 final checkpoint；不是由 `motitor_step=2` 触发的周期性 checkpoint。

### Grad/Update 监控

grad/update 监控用于看哪些模块真的收到梯度、哪些模块实际发生更新，以及 update/param ratio 是否异常。它在训练循环里接入，而不是 forward 里接入：`optimizer.step()` 前记录参数抽样和梯度健康，`optimizer.step()` 后计算参数 update。

当前 `grad_update_summary.json` 包含：

- `optimizer`：`update_successful`、Megatron 全局 `grad_norm`、`num_zeros_in_grad`。
- `groups`：按模块分组的参数、梯度、update 摘要。
- `per_parameter_samples`：每组抽样到的具体参数记录，包含 shape、dtype、sampled elements、param/grad/update norm、max_abs、nan/inf 计数。
- `counts`：本 rank 看到的参数数、按 pattern 跳过数、按 group limit 跳过数、实际抽样参数数。

当前分组规则：

- `route_gate`：route、read encoder/concept、final read concept gate、DD route 等。
- `vq_codebook`：VQ、codebook、quantizer 相关参数。
- `fusion`：fusion/fuse 相关参数。
- `embed_in` / `embed_out`：输入 embedding 和输出 head。
- `encoder` / `highlevel` / `decoder` / `backbone` / `other`。

重点看：

- `parameters_with_grad_count`：该组有多少参数真的有梯度。
- `grad_l2_norm_sampled` / `grad_max_abs_sampled`：该组梯度量级。
- `update_l2_norm_sampled` / `update_max_abs_sampled`：该组实际参数更新量级。
- `grad_to_param_norm_ratio_sampled`：梯度相对参数量级。
- `update_to_param_norm_ratio_sampled`：实际更新相对参数量级。
- `grad_nan_count` / `grad_inf_count` / `update_nan_count` / `update_inf_count`：反向和更新是否出现非有限值。
- `grad_zero_ratio_sampled` / `update_zero_ratio_sampled`：梯度或更新是否大面积为 0。

grad/update 监控开销相关参数：

- `CONCEPTLM_MONITOR_GRAD_UPDATE_PARAM_PATTERNS`：参数名 include 过滤；空值表示不过滤。
- `CONCEPTLM_MONITOR_GRAD_UPDATE_EXCLUDE_PATTERNS`：参数名 exclude 过滤；空值表示不过滤。
- `CONCEPTLM_MONITOR_GRAD_UPDATE_SNAPSHOT_TENSORS_PER_GROUP`：每组最多抽样多少个参数张量，默认 `4`；16 卡压测默认 `2`。
- `CONCEPTLM_MONITOR_GRAD_UPDATE_SNAPSHOT_MAX_ELEMENTS`：每个参数张量最多抽样多少个元素，默认 `65536`；16 卡压测默认 `16384`。
- `CONCEPTLM_MONITOR_GRAD_UPDATE_MAX_PARAM_RECORDS`：`per_parameter_samples` 最多写多少条，默认 `128`；16 卡压测默认 `64`。

16 卡 grad/update 监控测评：

- RJob：`c22vq-gradupd2-0610005002-5293050`，状态 `Succeeded`。
- run 目录：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/monitor16_0610005002/custom_monitor/c22vq-gradupd2-0610005002_20260610_005002_run`
- checkpoint 目录：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/monitor16_0610005002/checkpoints/iter_0000002`
- 测评设置：`TRAIN_ITERS=2`，`motitor_step=2`，`CONCEPTLM_MONITOR_RANK_MAX_TOKENS=64`，每组 rank layer 只选第 0 层，`CONCEPTLM_MONITOR_ROUTE_SVD_MAX_MATRICES=1`，每个 grad/update 组抽样 2 个参数张量、每张量最多 16384 个元素。
- step 2 写出 VQ、rank、route、grad/update 四组文件：`vq_summary.json`、`tensor_health.json`、`codebook_counts.json`、`latency.json`、`rank_summary.json`、`rank_latency.json`、`route_summary.json`、`route_latency.json`、`grad_update_summary.json`、`grad_update_latency.json`。
- run 根目录追加 `latency.jsonl`、`rank_latency.jsonl`、`route_latency.jsonl`、`grad_update_latency.jsonl`。
- grad/update 采样覆盖：本 writer rank 看到 `755` 个参数张量，实际抽样 `14` 个；`decoder/embed_in/embed_out/encoder/fusion/highlevel/route_gate/vq_codebook` 都有梯度，`grad_nan_count`、`grad_inf_count`、`update_nan_count`、`update_inf_count` 全部为 0。
- grad/update log 时延：包含 optimizer 前采样总耗时 `39.35 ms`，其中 `pre_optimizer_collect=27.85 ms`，`post_optimizer_compute=4.04 ms`，`write_files=7.07 ms`。
- 同一步其他自建监控时延：VQ `503.68 ms`，rank `63.32 ms`，route/SVD `195.67 ms`。
- checkpoint 出现 `iter_0000002` 是 `TRAIN_ITERS=2` 到训练结束后的 final checkpoint；不是由 `motitor_step=2` 触发的周期性 checkpoint。自建 log 仍然只由 `motitor_step` 控制频率，和 checkpoint 保存频率不绑定。

补充压测记录：

- 4 iter 版本 RJob `c22vq-gradupd16-0610004107-69400648` 在 step 2 已成功写出 `grad_update_summary.json` 和 `grad_update_latency.json`，grad/update 总耗时 `41.90 ms`。
- 该 4 iter 任务后续在 step 4 forward 的 fused CE 分配约 3.06 GiB 时 OOM，已手动 stop；这个失败发生在训练 forward/CE 路径，不是 grad/update 监控写入失败。

### 已知缺口

当前任务配置了 `HIDDEN_RANK_LOG_INTERVAL=500` 和 `HIDDEN_RANK_LOG_PATH=results/exit_hidden_rank.jsonl`，但 V2.2-VQ 的 labels path 里直接计算 logits 和 LM loss，看起来没有调用传入的 `output_processor`。因此 live run 目前没有写出 `exit_hidden_rank.jsonl`。同一路径大概率也绕过了 `olmo3_z_loss_output_processor` 里的 `OLMO3_Z_LOSS_MULTIPLIER=1e-5` 处理。后续在依赖 exit hidden rank 或 z-loss 指标前，要优先修这个问题。

### 监控 Backlog

Priority 0：当前 job 正在跑时也能加

- 外部 progress parser：解析 final-rank training log，落成 JSONL/CSV，字段包括 iteration、consumed tokens、LR、LM loss、VQ/HLM loss、aux loss、route reg、throughput、MFU、grad norm、skipped/nan iterations、ETA。
- 外部 health probe：周期性记录 RJob 状态、active replica 数、最新日志时间戳、TensorBoard/W&B 文件更新时间、checkpoint 目录增长、GPFS 剩余空间。
- 告警条件：固定时间内没有新训练日志、任一 replica 不是 Running、nan iterations 增长、throughput 明显下降、checkpoint lag 超过预期 save interval、GPFS 剩余空间过低。

Priority 1：需要改代码，下次重启/重提后生效

- 修 V2.2-VQ 的 `output_processor` 使用，让 labels path 也能启用 exit hidden rank 和 OLMo z-loss。
- 增加结构化 VQ codebook 健康指标：每个 codebook 的 usage histogram summary、entropy/perplexity、dead-code ratio、max usage share、per-head utilization。
- 增加 concept 表征健康指标：`concept_vq_hidden`、predicted vectors、target concepts、repeated concepts、fusion output、final decoder input 的 finite count、RMS/norm/max_abs。
- 增加 HLM 预测质量指标：MSE 模式记录 predicted/target concepts 的 cosine similarity 和 norm ratio；`conceptlm_v22_vq_hlm_loss_type=ce` 时记录 top-1/top-k accuracy。

Priority 2：Priority 1 稳定后再加

- route/gate 指标后续可以做跨 step 聚合和可视化，把 entropy、max-share、collapse ratio、SVD rank 曲线单独画出来。
- 在关键边界加低频 activation finite/norm probe。
- 增加可选短窗口 profiler，用来定位 iteration-time regression，不做 always-on profiling。
- 增加 checkpoint metadata summary：latest iteration、checkpoint 写入时间、大小、保存耗时、full-state/sharded-state 确认。

## Re-submit

From this directory:

```bash
bash submit_scaledtrunc_64g_muon_lr6e5.sh
```

By default, new run outputs go under this scale-up directory:

`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/runs/`

Override `RUN_BASE` or `JOB_NAME` if you want an exact target path/name.

The copied repo entrypoints have only their default storage paths changed:

- `REPO_ROOT` defaults to this directory's `repo/`
- `RUN_BASE` defaults to this directory's `runs/`

Training hyperparameters and model arguments are kept aligned with the original run manifest.
The target wrapper also defaults `INIT_METHOD_VARIANT=scaled_truncated_normal`, matching the recorded run.
