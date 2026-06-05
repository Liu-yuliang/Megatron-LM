# ConceptLM V2.2-VQ Overhead Notes - 2026-05-29

This note records the 8-GPU H200 RJob overhead experiments for the Pythia-1B
ConceptLM V2.2-VQ branch. All timings below use rank0 training logs from RJob
runs, not local GPU execution.

## Baseline Timing

Common setup:

- Model backbone: Pythia-1B, 16 layers, hidden size 2048, FFN size 8192.
- Sequence length: 2048.
- Global batch size: 256.
- Micro batch size: 16.
- 8 H200 GPUs, TP/PP/CP all 1.
- Timing metric: median `elapsed time per iteration (ms)` over iter 10-80.
- Compile/autotune iter 1-2 are ignored.

| Setting | Job | Params | Median ms/iter | Relative to Pythia |
| --- | --- | ---: | ---: | ---: |
| Pythia-1B | `pythia1b-overhead-8g-05291540-82444430` | 1.0118B | 794.9 | 1.00x |
| V2.2-VQ, no residual routes | `c22vqabl-none-8g-05292048-3050967` | 1.4212B | 887.4 | 1.12x |
| V2.2-VQ, all residual routes | `c22vqallres-full-8g-05291958-50991347` | 1.5562B | 1174.8 | 1.48x |

The clean V2.2-VQ overhead without residual routes is about +11.6% wall-clock.
All residual routes increase the overhead to about +47.8%.

## Theoretical Estimate

For Pythia-1B with `H=2048`, `L=2048`, `chunk=4`, concept length is 512.

The high-level tower has 8 layers at 1/4 sequence length:

- Approx short/full layer ratio: 0.223.
- 8 short layers versus 16 full layers: about +11.2% FLOPs.

All-residual full `H x H` route projections add approximately:

- 24 full-sequence `H x H` projections.
- 8 chunk-sequence `H x H` projections.
- Approx +11.6% FLOPs versus Pythia-1B.

So all-residual full-projection theoretical FLOPs are roughly `1.23x` Pythia,
while measured wall-clock is `1.48x`. The gap is attributed to route/read path
overheads: stack/norm/einsum/projection, kernel fragmentation, memory movement,
and compile/fusion behavior.

## Full Projection vs Scalar vs Diag

All-residual routes were tested in three projection modes:

| Mode | Job | Params | Median ms/iter | Relative to Pythia |
| --- | --- | ---: | ---: | ---: |
| full `H x H` | `c22vqallres-full-8g-05291958-50991347` | 1.556B | 1174.8 | 1.478x |
| scalar | `c22vqallres-scalar-8g-05291958-54891687` | 1.422B | 1179.4 | 1.484x |
| diag | `c22vqallres-diag-8g-05291958-54959050` | 1.422B | 1189.7 | 1.497x |

Observation: scalar/diag reduce parameters by about 134M but did not improve
wall-clock in this implementation. Full projection likely benefits from the
compiled/fused route path, while scalar/diag currently take less fused paths.
The dominant cost appears to be opening route/read paths, not the projection
matrix form alone.

## Two Residual Waves

The residual system has two different waves:

1. Intra-module DD:
   - encoder self-DD,
   - high-level/concept self-DD,
   - decoder DD.
2. Inter-module read/flow:
   - concept reads encoder layers,
   - decoder reads encoder layers,
   - decoder reads concept/high-level layers,
   - decoder final concept route.

8-GPU ablation:

| Setting | Job | Params | Median ms/iter | Relative to no residual |
| --- | --- | ---: | ---: | ---: |
| no residual | `c22vqabl-none-8g-05292048-3050967` | 1.421B | 887.4 | 1.00x |
| intra only | `c22vqabl-intra-8g-05292048-2928705` | 1.422B | 998.5 | +12.5% |
| inter only | `c22vqabl-inter-8g-05292032-88347541` | 1.556B | 1062.9 | +19.8% |
| both | `c22vqallres-full-8g-05291958-50991347` | 1.556B | 1174.8 | +32.4% |

Conclusion: inter-module read/flow is more expensive than intra-module DD.
If reducing overhead is the priority, first consider reducing or sparsifying
`decoder_read_encoder`, `decoder_read_concept`, `concept_read_encoder`, and
`final_concept_route`.

## Source Truncation: First Plus Last Two

An optional runtime switch was added:

```bash
CONCEPTLM_V21_ROUTE_FIRST_PLUS_LAST_N=2
```

When unset or zero, behavior is unchanged. When set to `2`, route/DD sources are
truncated to source `0` plus the most recent 2 sources.

Code touched:

- `megatron/core/models/gpt/conceptlm_v21.py`
- `examples/public_training_bench/submit_rjob.sh`

The source truncation is applied to:

- `V21DepthDD` history sources,
- `V21ResidualFlowRouteAdd` token-level sources,
- `V21ResidualFlowRouteAdd.forward_repeated_chunks` concept chunk sources,
- encoder-to-concept source stacks,
- encoder-to-decoder source stacks,
- concept-to-decoder source stacks,
- DD concept candidate stacks.

8-GPU all-residual result:

| Setting | Job | Median ms/iter | Delta vs all-source |
| --- | --- | ---: | ---: |
| all sources | `c22vqallres-full-8g-05291958-50991347` | 1174.8 | baseline |
| source 0 + recent 2 | `c22vqallres-full-8g-05292109-18159082` | 1143.8 | -2.6% |

The truncation is correct and runs successfully, but speedup is modest. This
suggests that simply reducing the number of source layers is not enough; the
main overhead is still the existence of route/read paths and their associated
operations.

## Practical Takeaways

- No-residual V2.2-VQ is close to the theoretical high-level tower cost.
- All-residual V2.2-VQ is substantially slower than the FLOP estimate.
- Scalar/diag route projection reduces parameter count but not current
  wall-clock.
- Inter-module read/flow is the larger residual overhead contributor.
- `0 + recent 2` source truncation gives only a small speedup in the current
  implementation.
- Better overhead reductions likely need path-level sparsification or fusion,
  not only smaller projection parameterization.

