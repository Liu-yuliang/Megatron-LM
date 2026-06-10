# Step2 route-source-context A/B failed attempt - 2026-06-10

AB root: /mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/ab_step2_routectx_0610004443

Baseline job:
- c22vq-s2-base-16g-0610004443-49074180
- stopped manually after new job failed; no loss comparison used from this failed attempt.

New job:
- c22vq-s2-new-16g-0610004443-55509325
- failed on first forward before producing training loss.

Failure:
- file: megatron/core/models/gpt/conceptlm_v21.py:_build_decoder_concept_states
- error: TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not tuple
- cause: Step2 changed helpers to accept a shared concept_layer_stack tensor, but ConceptLM V2.2 VQ forward still passed raw concept_layer_states tuple directly to _build_dd_concept_candidates() and _build_decoder_concept_states().

Fix plan:
- Build route_source_context in ConceptLM V2.2 VQ forward.
- Pass route_source_context.concept_layer_stack to both DD concept candidate and decoder concept state builders.
- This preserves the same selected HLM layer stack as before and only removes duplicate stack construction.
