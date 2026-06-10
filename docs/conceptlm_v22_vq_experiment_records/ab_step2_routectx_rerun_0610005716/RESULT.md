# Step2 route source context A/B 结果 - 2026-06-10

结论：通过。step2 在 `e8c0e9e3246820da5133485b60bfc58c7e4292d0` 上通过 16 卡 A/B，`lm loss` 与 baseline 对齐。

## 对比设置

- baseline commit: `365cd0744005126b1288b0e044a6d66d8d2992dd`
- new commit: `e8c0e9e3246820da5133485b60bfc58c7e4292d0`
- base job: `c22vq-s2r-base-16g-0610005716-37147556`
- new job: `c22vq-s2r-new-16g-0610005716-44432979`
- 卡数：16 卡，`REPLICAS=2`，每节点 8 卡
- `TRAIN_ITERS=60`，但 loss 已对齐后提前停止
- `OVERLAP_PARAM_GATHER=0`
- `ENABLE_WANDB=0`
- `CONCEPTLM_V21_SELF_DD_UNSTACKED_FASTPATH=1`
- `CONCEPTLM_V21_ROUTE_FIRST_PLUS_LAST_N=0`

## loss 对齐

重叠步数：21 步，first=1，last=21。

```text
lm max_abs=1e-05 max_rel=8.62005801e-07 at=18 mean_abs=5.71428571e-06
concept total max_abs=2e-05 max_rel=1.71280536e-06 at=11 mean_abs=6.66666667e-06
vq_raw max_abs=0
hlm_raw max_abs=2e-06
aux_raw max_abs=2e-06
vq_eff max_abs=0
hlm_eff max_abs=1e-10
aux_eff max_abs=1e-10
grad_norm max_abs=0.002
```

这些差异属于 bf16/打印精度范围，未观察到逻辑变化导致的 loss drift。

## 速度粗略统计

统计 step 3 起的已有样本：

```text
baseline: n=18 first=3 last=20 avg_ms=31288.2 avg_tps_gpu=8379.8 avg_global_tps=134076.1 avg_mfu=40.47
new:      n=18 first=3 last=20 avg_ms=30983.4 avg_tps_gpu=8462.0 avg_global_tps=135391.5 avg_mfu=40.87
```

step2 主要目标是等价复用 route source context，不把这组短跑速度作为最终结论。

## 复现路径

- 提交脚本：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step2_routectx_rerun_0610005716/scripts/submit_step2_ab.sh`
- baseline rank1 log: `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step2r_base_16g_0610005716/run/c22vq-s2r-base-16g-0610005716/logs/pretrain_conceptlm_v22_vq_rank1_c22vq-s2r-base-16g-0610005716-37147556.log`
- new rank1 log: `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step2r_new_16g_0610005716/run/c22vq-s2r-new-16g-0610005716/logs/pretrain_conceptlm_v22_vq_rank1_c22vq-s2r-new-16g-0610005716-44432979.log`
