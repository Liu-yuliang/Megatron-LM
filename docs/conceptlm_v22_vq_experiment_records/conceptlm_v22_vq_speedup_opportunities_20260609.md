# ConceptLM V2.2-VQ 加速机会记录 - 2026-06-09

这份文档记录在排查当前 ConceptLM V2.2-VQ + Muon 任务的
`overlap-param-gather` 问题时看到的实现层加速机会。重点是区分：

- 不改变数学逻辑、只优化实现和调度的改动；
- 会改变有效路径或 source 集合、需要单独做 ablation 的改动。

## 当前现象

修复 Muon layer-wise `overlap-param-gather` 兼容性后，16 卡验证任务可以正常跑完，
但没有带来吞吐提升：

| 设置 | 统计步数 | 平均 ms/iter | 平均 tps/GPU | 平均 global tps | 平均 MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| OPG off baseline | 3-60 | 8487.1 | 7725.5 | 123608.6 | 37.31% |
| OPG on with compatibility fix | 3-60 | 8515.5 | 7700.2 | 123203.1 | 37.19% |

这说明兼容性修复只是让 OPG 能跑起来，并没有解决当前模型结构导致的预取命中率问题。

核心原因是：ConceptLM 当前把 backbone 层和 DD/residual route 模块分组注册，
但 forward 里是交错执行的。参数注册顺序和真实执行顺序不一致，会让参数预取顺序偏离
forward 顺序，导致已经预取的参数变成 stale handle，或者在真正用到前就被 flush 掉。

## 最高优先级：让模块注册顺序对齐 forward 顺序

当前真实执行顺序大致是：

```text
encoder layer 0 -> encoder self-DD 0
encoder layer 1 -> encoder self-DD 1
...
concept/HLM layer i -> concept self-DD/read-encoder i
...
decoder layer i -> decoder DD/final concept route i
                -> decoder read encoder i
                -> decoder read concept i
```

但当前注册顺序更接近：

```text
先注册所有 backbone encoder/decoder layers
再注册 DD modules
再注册 residual route modules
```

建议方向：

- 引入 encoder、concept/HLM、decoder 的 per-layer wrapper。
- 每个 wrapper 内部注册该层 backbone block，以及紧跟该层执行的 DD/residual routes。
- 保持现有数学和调用顺序不变，只改变 PyTorch module tree 的参数注册顺序。
- 让 Megatron DDP bucket 构建时看到更接近真实 forward 的参数顺序。

预期收益：

- 提高 OPG 参数预取命中率。
- 减少 stale prefetch flush。
- 改善 Muon layer-wise optimizer 下的参数 locality。
- 让 bucket 调度更稳定、更可预测。

风险：

- checkpoint key 兼容性需要处理。要么保留旧 attribute 名称，要么加 load/save remapping。

## 避免反复 materialize history stack

`V21DepthDD` 已经有 unstacked 路径：

- `V21DepthDD.forward_unstacked(...)`

但 encoder self-DD 当前仍然会在 active layer 上先构造：

```python
torch.stack(dd_history, dim=2)
```

对应位置：

- `conceptlm_v21.py:_run_v21_encoder`

建议方向：

- 给 `V21SelfDD` 增加 `forward_unstacked(...)` helper。
- encoder self-DD 和 concept self-DD 在可用时直接传 list 走 unstacked 路径。
- 只有在必须使用 stacked tensor 的情况下才 fallback 到原路径。

预期收益：

- 减少 `[seq, batch, history, hidden]` 大 tensor 的写入和读取。
- 减少 DD 路径上的临时 tensor。
- 降低 activation memory bandwidth 压力。

风险：

- 低。只要做成 guarded fast path，并保留原始 fallback，数学逻辑可以完全不变。

## 合并 decoder 每层的 route 工作

decoder 每层之后现在可能连续执行多段 route-add：

```text
decoder DD / final concept route
decoder read encoder
decoder read concept
```

每一段都会读写完整 hidden state，并可能启动独立的小 kernel。可以把这些操作合并到
每层一个 composite helper 里，保持顺序不变，但减少 hidden state 往返和 kernel
fragmentation。

建议方向：

- 构造 decoder per-layer route helper。
- 输入包括当前 layer output、decoder history、encoder source、concept source、
  final concept state、gate scale 等。
- 保持现有顺序：
  1. decoder DD/final concept route；
  2. decoder read encoder；
  3. decoder read concept。
- 对当前常用配置做 specialize 或 torch.compile。

预期收益：

- 减少 hidden-state round trip。
- 减少小 kernel 数量。
- 给 torch.compile 更大的融合空间。

风险：

- 中等。需要和现有路径做等价测试，尤其是 final/read-concept gate scale。

## 共享 route source context

当前多处会分别构建 source stack：

- encoder-to-concept source stack；
- encoder-to-decoder source stack；
- concept-to-decoder source stack；
- DD concept candidate stack。

这些路径里会重复做 layer selection、`torch.stack`、`torch.cat`、chunk merge、
shared layernorm 等操作。

建议方向：

- 每个 forward 构建一个 route context。
- 缓存选中的 source layer list、chunk-merged states、stacked states、
  shared-normalized states。
- 多个 route consumer 复用同一个 context。

预期收益：

- 减少重复 stack/cat/chunk merge/layernorm。
- 更清晰地管理大 source tensor 的生命周期。

风险：

- 低到中等。要避免 context 保留 tensor 太久，否则可能增加 activation memory。

## routes 稀疏时使用 active-only 注册

代码里已经支持：

```bash
CONCEPTLM_V21_ACTIVE_ONLY_ROUTES=1
```

当 `first_n` 或 `every_n` 让 route 只在部分层生效时，active-only 注册可以避免创建
不会被真正执行的 route 模块，从而减少参数 bucket 噪声。

预期收益：

- routes 稀疏时有用。
- 如果当前默认设置是每层都 active，收益有限。

风险：

- 需要确认 checkpoint 兼容性和提交脚本中的 flag 对齐。

## 低优先级选项

2026-05-29 的 overhead note 已经测过 full route projection、scalar route、
diag route。结果是 scalar/diag 明显减少参数量，但 wall-clock 没有变快。

这说明当前主要瓶颈不是 `H x H` projection 本身，而是：

- route path 被打开后的额外读写；
- source stack 构建；
- normalization；
- kernel fragmentation；
- compile/fusion 效果不足。

`CONCEPTLM_V21_ROUTE_FIRST_PLUS_LAST_N=2` 这种 source truncation 之前能带来小幅加速，
但它改变了实际使用的 source 集合，应当算 ablation，不是纯实现优化。

## 建议实施顺序

1. 给 encoder/concept self-DD 增加 unstacked fast path。
2. 增加 per-forward route source context 共享。
3. 把 decoder route 工作合成 per-layer composite helper。
4. 把 module registration 重构成和 forward 顺序一致的 per-layer wrapper。
5. 在 registration-order 对齐后，重新测试 Muon layer-wise OPG。

前两项风险较低，适合先做短 benchmark。第四项对 OPG 和参数 locality 最关键，
但需要更认真处理 checkpoint key 兼容性。
