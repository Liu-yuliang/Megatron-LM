# ConceptLM V2.2 VQ Experiment Records

This directory keeps lightweight reproducibility records for the ConceptLM V2.2 VQ staged optimization and training setup checks done on 2026-06-09 and 2026-06-10.

Large runtime artifacts are intentionally not committed here. Full rank logs, checkpoints, dataset caches, TensorBoard events, and W&B offline runs remain on shared storage and are referenced by absolute paths inside the copied result files.

## Main Training Setup Records

- `README.source.md`: Chinese task notes for the moved default ConceptLM V2.2 training setup.
- `run_records/`: original `c22vq-muon64-s8-lr6e5-scaled-opg0-0608234919` submit log, launch manifest, and RJob status.
- `source_snapshot/`: training and RJob submission scripts captured when the default setup was inspected.
- `comparisons/`: source-vs-target setup comparison report.
- `dryruns/`: lightweight local/RJob dry-run submit records.
- `align16_0609172152/`: 16-card submission alignment reports.
- `loss20_0609175814/`: 20-step loss alignment report.

## Optimization Steps

All optimization steps use the current training branch:

```text
conceptlmv2-linear-resflow-20260526
```

Step commits:

```text
365cd0744 Enable ConceptLM V2.2 self-DD fastpath by default
88902b7d4 Share ConceptLM V2.2 route source context
e8c0e9e32 Fix ConceptLM V2.2 route context wiring
1ca63d1cc Compose ConceptLM V2.2 decoder route dispatch
6ef82efbf Order ConceptLM V2.2 parameters by forward dispatch
```

The intermediate `88902b7d4` route-context commit failed its first A/B because the V2.2 VQ forward still passed a raw tuple into helpers expecting stacked context. The fix is `e8c0e9e32`; the passing step2 result is recorded under `ab_step2_routectx_rerun_0610005716/`.

## A/B Result Directories

- `ab_step2_routectx_0610004443/`: failed step2 attempt; includes `FAILED_ATTEMPT.md`, job metadata/status, submit logs, and submit script.
- `ab_step2_routectx_rerun_0610005716/`: passing step2 rerun; includes `RESULT.md`, job metadata/status, submit logs, and submit script.
- `ab_step3_decoderroute_0610012914/`: passing step3 A/B; includes `RESULT.md`, job metadata/status, submit logs, and submit script.
- `ab_step4_paramorder_0610015751/`: passing step4 A/B; includes `RESULT.md`, job metadata/status, submit logs, and submit script.

## Speedup Planning

- `conceptlm_v22_vq_speedup_opportunities_20260609.md`: Chinese notes on implementation-level speedup opportunities, including self-DD unstacked fast path, route source context sharing, decoder route dispatch composition, parameter order alignment, and OPG retesting.

## Git Sync Notes

Each step was committed locally and a push was attempted. Earlier attempts failed because GitHub HTTPS credentials were unavailable in the execution environment. This directory was added later so the code commits and lightweight experiment records can be pushed together.
