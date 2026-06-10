# Step3 decoder route dispatch A/B 结果 - 2026-06-10

结论：通过。step3 在 `1ca63d1cc17d2f69afe0ae8588a8c0da8dcfdadc` 上通过 16 卡 A/B，`lm loss` 与 step2 baseline 对齐。

## 对比设置

- baseline commit: `e8c0e9e3246820da5133485b60bfc58c7e4292d0`
- new commit: `1ca63d1cc17d2f69afe0ae8588a8c0da8dcfdadc`
- base job: `c22vq-s3d-base-16g-0610012914-93361608`
- new job: `c22vq-s3d-new-16g-0610012914-99352712`
- 卡数：16 卡，`REPLICAS=2`，每节点 8 卡
- `TRAIN_ITERS=60`，loss 已对齐后提前停止
- `OVERLAP_PARAM_GATHER=0`
- `ENABLE_WANDB=0`
- `CONCEPTLM_V21_SELF_DD_UNSTACKED_FASTPATH=1`
- `CONCEPTLM_V21_ROUTE_FIRST_PLUS_LAST_N=0`

## loss 对齐

重叠步数：5 步，first=1，last=5。

```text
lm max_abs=1e-05 max_rel=8.55260795e-07 at=3 mean_abs=6e-06
concept total max_abs=1e-05 max_rel=8.55240314e-07 at=3 mean_abs=2e-06
vq_raw max_abs=0
hlm_raw max_abs=1e-06
aux_raw max_abs=1e-06
vq_eff max_abs=0
hlm_eff max_abs=1e-10
aux_eff max_abs=0
grad_norm max_abs=0
```

这些差异属于 bf16/日志打印精度范围，未观察到逻辑变化导致的 loss drift。

## 速度观察

step3 主要验证 decoder route dispatch 的实现等价性，未用短跑速度作最终结论。首轮 compile 后，step 3-5 的 new 侧 tps/GPU 略高于 baseline，但样本太少，只作为参考。

## 复现路径

- 提交脚本：`/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step3_decoderroute_0610012914/scripts/submit_step3_ab.sh`
- baseline rank1 log: `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step3d_base_16g_0610012914/run/c22vq-s3d-base-16g-0610012914/logs/pretrain_conceptlm_v22_vq_rank1_c22vq-s3d-base-16g-0610012914-93361608.log`
- new rank1 log: `/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/runs/concept_olmo3_v22_vq_step3d_new_16g_0610012914/run/c22vq-s3d-new-16g-0610012914/logs/pretrain_conceptlm_v22_vq_rank1_c22vq-s3d-new-16g-0610012914-99352712.log`

## git 同步

已提交本地 git commit：`1ca63d1cc Compose ConceptLM V2.2 decoder route dispatch`。

已按要求尝试 push 到 `fork conceptlmv2-linear-resflow-20260526`，失败原因仍为 GitHub HTTPS 凭据不可用：

```text
fatal: could not read Username for 'https://github.com': No such device or address
```
