# Step4 forward-order named_parameters A/B 结果 - 2026-06-10

结论：通过。step4 在 `6ef82efbfc93fad97f9692bbee6c1d75f0893b2b` 上通过 16 卡 A/B，`lm loss` 与 step3 baseline 对齐。

## 对比设置

- baseline commit: `1ca63d1cc17d2f69afe0ae8588a8c0da8dcfdadc`
- new commit: `6ef82efbfc93fad97f9692bbee6c1d75f0893b2b`
- base job: `c22vq-s4p-base-16g-0610015751-63204750`
- new job: `c22vq-s4p-new-16g-0610015751-68934843`
- 卡数：16 卡，`REPLICAS=2`，每节点 8 卡
- `TRAIN_ITERS=60`，loss 已对齐后提前停止
- `OVERLAP_PARAM_GATHER=0`
- `ENABLE_WANDB=0`
- `CONCEPTLM_V21_SELF_DD_UNSTACKED_FASTPATH=1`
- `CONCEPTLM_V21_ROUTE_FIRST_PLUS_LAST_N=0`
- `CONCEPTLM_V21_FORWARD_ORDER_NAMED_PARAMETERS=1`，默认开启；可用 `=0` 回退原始 named parameter 顺序

## 实现说明

step4 只改 `ConceptLMV21Model.named_parameters()` 的输出顺序，让参数遍历顺序尽量贴近真实 forward dispatch：

1. embedding / rotary；
2. encoder layer + encoder self-DD；
3. chunk smoothing / VQ norm / quantizer / MLP bottleneck；
4. concept/HLM layer + concept self-DD + concept read encoder route；
5. concept final LN / prediction heads；
6. fusion；
7. decoder layer + decoder DD / concept route + decoder read encoder / concept route；
8. decoder final LN；
9. output layer；
10. fallback 补齐未覆盖参数。

这个改动不改变 module tree、不改变 `state_dict` key、不改变 forward 数学逻辑，目标是给后续 `overlap-param-gather` 提供更接近执行顺序的参数顺序。

## loss 对齐

重叠步数：4 步，first=1，last=4。

```text
lm max_abs=0
concept total max_abs=0
vq_raw max_abs=0
hlm_raw max_abs=1e-06
aux_raw max_abs=0
vq_eff max_abs=0
hlm_eff max_abs=0
aux_eff max_abs=0
grad_norm max_abs=0
```

第 4 步 `hlm_raw` 有 `1e-06` 的日志打印差异，其余关键项包括 `lm loss` 完全一致。未观察到逻辑变化导致的 loss drift。

## 速度观察

step4 的目标是参数顺序对齐，不把这组短跑速度作为最终性能结论。首轮 compile 后已有样本：

```text
baseline: n=3 first=3 last=5 avg_ms=31446.5 avg_tps_gpu=8336.3 avg_global_tps=133380.9 avg_mfu=40.27
new:      n=2 first=3 last=4 avg_ms=30921.2 avg_tps_gpu=8477.8 avg_global_tps=135644.8 avg_mfu=40.95
```

## 任务状态

loss 对齐后两个任务均已停止：

```text
base: Stopped, stopped replicas=2
new:  Stopped, stopped replicas=2
```

## 复现路径

- 提交脚本：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step4_paramorder_0610015751/scripts/submit_step4_ab.sh`
- baseline rank1 log: `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step4p_base_16g_0610015751/run/c22vq-s4p-base-16g-0610015751/logs/pretrain_conceptlm_v22_vq_rank1_c22vq-s4p-base-16g-0610015751-63204750.log`
- new rank1 log: `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step4p_new_16g_0610015751/run/c22vq-s4p-new-16g-0610015751/logs/pretrain_conceptlm_v22_vq_rank1_c22vq-s4p-new-16g-0610015751-68934843.log`
- base status: `/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step4_paramorder_0610015751/jobs/base.status.txt`
- new status: `/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step4_paramorder_0610015751/jobs/new.status.txt`

## git 同步

已提交本地 git commit：`6ef82efbf Order ConceptLM V2.2 parameters by forward dispatch`。

已按要求尝试 push 到 `fork conceptlmv2-linear-resflow-20260526`，失败原因仍为 GitHub HTTPS 凭据不可用：

```text
fatal: could not read Username for 'https://github.com': No such device or address
```
