# Copyright (c) 2026, ConceptLM contributors.

"""Builder for the isolated ConceptLM V1 GPT prototype."""

from megatron.core.models.gpt.conceptlm_v1 import ConceptLMV1Model
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from gpt_builders import _get_transformer_layer_spec


def conceptlm_v1_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build ConceptLM V1 without changing the baseline GPT builder."""

    print_rank_0(f"building ConceptLM V1 model with {args.conceptlm_backbone} backbone ...")
    if args.mtp_num_layers is not None:
        raise NotImplementedError("ConceptLM V1 and MTP are not wired together in this prototype")
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    elif args.conceptlm_backbone == "olmo3":
        from megatron.core.models.gpt.olmo3_layer_specs import (
            olmo3_layer_spec,
            olmo3_local_layer_spec,
        )

        transformer_layer_spec = (
            olmo3_local_layer_spec
            if args.transformer_impl == "local"
            else olmo3_layer_spec
        )
    elif args.conceptlm_backbone == "pythia":
        if args.transformer_impl != "transformer_engine":
            raise NotImplementedError("ConceptLM V1 Pythia backbone currently expects transformer_engine")
        from megatron.core.models.gpt.pythia_layer_specs import (
            pythia_parallel_residual_layer_spec,
        )

        transformer_layer_spec = pythia_parallel_residual_layer_spec
    else:
        use_te = args.transformer_impl == "transformer_engine"
        if args.experimental_attention_variant is not None:
            transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config=config, vp_stage=vp_stage
            )
        elif args.num_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )
        elif args.heterogeneous_layers_config_path is not None:
            if config.transformer_impl == "inference_optimized":
                raise AssertionError("heterogeneous layers do not support inference_optimized")
            transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
        else:
            transformer_layer_spec = _get_transformer_layer_spec(use_te, config)

    model = ConceptLMV1Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=None,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        concept_chunk_size=args.conceptlm_chunk_size,
        concept_codebook_size=args.conceptlm_codebook_size,
        concept_num_codebooks=args.conceptlm_num_codebooks,
        concept_special_layers=args.conceptlm_special_layers,
        concept_loss_type=args.conceptlm_loss_type,
        concept_merge_mode=args.conceptlm_merge_mode,
        concept_vq_commitment_cost=args.conceptlm_vq_commitment_cost,
        concept_hlm_ffn_hidden_size=args.conceptlm_hlm_ffn_hidden_size,
        concept_detach_ncp_target=args.conceptlm_detach_ncp_target,
    )

    return model
