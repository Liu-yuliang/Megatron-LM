# Copyright (c) 2026, ConceptLM contributors.

"""Builder for the Megatron-Core ConceptLM V2.2-VQ prototype."""

from megatron.core.models.gpt.conceptlm_v22_vq import ConceptLMV22VQModel
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


def conceptlm_v22_vq_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    """Build ConceptLM V2.2-VQ without changing V2.1."""

    print_rank_0(f"building ConceptLM V2.2-VQ model with {args.conceptlm_backbone} backbone ...")
    if args.mtp_num_layers is not None:
        raise NotImplementedError("ConceptLM V2.2-VQ and MTP are not wired together")
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
            raise NotImplementedError(
                "ConceptLM V2.2-VQ Pythia backbone currently expects transformer_engine"
            )
        from megatron.core.models.gpt.pythia_layer_specs import (
            pythia_parallel_residual_layer_spec,
        )

        transformer_layer_spec = pythia_parallel_residual_layer_spec
    else:
        use_te = args.transformer_impl == "transformer_engine"
        if args.experimental_attention_variant is not None:
            transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config=config,
                vp_stage=vp_stage,
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

    return ConceptLMV22VQModel(
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
        concept_encoder_layers=args.conceptlm_encoder_layers,
        concept_decoder_layers=args.conceptlm_decoder_layers,
        concept_special_layers=args.conceptlm_special_layers,
        concept_chunk_size=args.conceptlm_chunk_size,
        concept_shift_feature=args.conceptlm_shift_feature,
        concept_chunk_merge_method=args.conceptlm_chunk_merge_method,
        concept_enable_chunk_dualpath_smoothing=args.conceptlm_enable_chunk_dualpath_smoothing,
        concept_chunk_dualpath_alpha_init=args.conceptlm_chunk_dualpath_alpha_init,
        concept_codebook_size=args.conceptlm_v22_vq_codebook_size,
        concept_num_codebooks=args.conceptlm_v22_vq_num_codebooks,
        concept_vq_commitment_cost=args.conceptlm_v22_vq_commitment_cost,
        concept_vq_merge_mode=args.conceptlm_v22_vq_merge_mode,
        concept_hlm_loss_type=args.conceptlm_v22_vq_hlm_loss_type,
        concept_detach_hlm_target=args.conceptlm_v22_vq_detach_hlm_target,
        concept_layer_norm_option=args.conceptlm_layer_norm_option,
        concept_fusion_norm_alpha_init=args.conceptlm_fusion_norm_alpha_init,
        concept_fusion_alpha_init=args.conceptlm_fusion_alpha_init,
        concept_hlm_ffn_hidden_size=args.conceptlm_hlm_ffn_hidden_size,
        concept_dd_two_route_add=args.conceptlm_v21_dd_two_route_add,
        concept_dd_two_route_add_concept_source=args.conceptlm_v21_dd_two_route_add_concept_source,
        concept_dd_two_route_add_enable_raw_concept_route=args.conceptlm_v21_dd_two_route_add_enable_raw_concept_route,
        concept_dd_two_route_add_enable_final_concept_route=args.conceptlm_v21_dd_two_route_add_enable_final_concept_route,
        concept_dd_two_route_add_beta_init=args.conceptlm_v21_dd_two_route_add_beta_init,
        concept_dd_two_route_add_every_n_layers=args.conceptlm_v21_dd_two_route_add_every_n_layers,
        concept_dd_two_route_add_concept_route_first_n=args.conceptlm_v21_dd_two_route_add_concept_route_first_n,
        concept_dd_two_route_add_decoder_hidden_size=args.conceptlm_v21_dd_two_route_add_decoder_hidden_size,
        concept_dd_two_route_add_concept_hidden_size=args.conceptlm_v21_dd_two_route_add_concept_hidden_size,
        concept_dd_two_route_add_use_softmax=args.conceptlm_v21_dd_two_route_add_use_softmax,
        concept_dd_two_route_add_disable_decoder_dd=args.conceptlm_v21_dd_two_route_add_disable_decoder_dd,
        concept_dd_two_route_add_decoder_use_layernorm=args.conceptlm_v21_dd_two_route_add_decoder_use_layernorm,
        concept_dd_two_route_add_decoder_use_softmax=args.conceptlm_v21_dd_two_route_add_decoder_use_softmax,
        concept_dd_two_route_add_concept_use_layernorm=args.conceptlm_v21_dd_two_route_add_concept_use_layernorm,
        concept_dd_encoder_self_dd=args.conceptlm_v21_dd_encoder_self_dd,
        concept_dd_encoder_self_dd_every_n_layers=args.conceptlm_v21_dd_encoder_self_dd_every_n_layers,
        concept_dd_encoder_self_dd_hidden_size=args.conceptlm_v21_dd_encoder_self_dd_hidden_size,
        concept_dd_encoder_self_dd_use_layernorm=args.conceptlm_v21_dd_encoder_self_dd_use_layernorm,
        concept_dd_concept_self_dd=args.conceptlm_v21_dd_concept_self_dd,
        concept_dd_concept_self_dd_every_n_layers=args.conceptlm_v21_dd_concept_self_dd_every_n_layers,
        concept_dd_concept_self_dd_hidden_size=args.conceptlm_v21_dd_concept_self_dd_hidden_size,
        concept_dd_concept_self_dd_use_layernorm=args.conceptlm_v21_dd_concept_self_dd_use_layernorm,
        concept_enable_full_residual_flow=args.conceptlm_v21_enable_full_residual_flow,
        concept_enable_concept_read_encoder=args.conceptlm_v21_enable_concept_read_encoder,
        concept_enable_decoder_read_encoder=args.conceptlm_v21_enable_decoder_read_encoder,
        concept_enable_decoder_read_concept=args.conceptlm_v21_enable_decoder_read_concept,
        concept_read_encoder_first_n=args.conceptlm_v21_concept_read_encoder_first_n,
        concept_decoder_read_encoder_first_n=args.conceptlm_v21_decoder_read_encoder_first_n,
        concept_residual_flow_beta_init=args.conceptlm_v21_residual_flow_beta_init,
        concept_residual_flow_route_hidden_size=args.conceptlm_v21_residual_flow_route_hidden_size,
        concept_residual_flow_route_use_softmax=args.conceptlm_v21_residual_flow_route_use_softmax,
        concept_residual_flow_source_use_layernorm=args.conceptlm_v21_residual_flow_source_use_layernorm,
        concept_residual_flow_shared_source_norm=args.conceptlm_v21_residual_flow_shared_source_norm,
        concept_final_read_concept_gate=args.conceptlm_v21_final_read_concept_gate,
        concept_final_read_concept_gate_init_final=args.conceptlm_v21_final_read_concept_gate_init_final,
        concept_final_read_concept_gate_target_final=args.conceptlm_v21_final_read_concept_gate_target_final,
        concept_compile_residual_flow_routes=args.conceptlm_v21_compile_residual_flow_routes,
        concept_compile_dd_routes=args.conceptlm_v21_compile_dd_routes,
    )
