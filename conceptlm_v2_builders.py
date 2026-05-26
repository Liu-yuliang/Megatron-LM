# Copyright (c) 2026, ConceptLM contributors.

"""Builder for the Megatron-Core ConceptLM V2 prototype."""

from megatron.core.models.gpt.conceptlm_v2 import ConceptLMV2Model
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


def conceptlm_v2_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build ConceptLM V2 without changing the baseline GPT or V1 builders."""

    print_rank_0(f"building ConceptLM V2 model with {args.conceptlm_v2_backbone} backbone ...")
    if args.mtp_num_layers is not None:
        raise NotImplementedError("ConceptLM V2 and MTP are not wired together in this prototype")
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    elif args.conceptlm_v2_backbone == "pythia":
        if args.transformer_impl != "transformer_engine":
            raise NotImplementedError("ConceptLM V2 Pythia backbone currently expects transformer_engine")
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

    return ConceptLMV2Model(
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
        concept_encoder_layers=args.conceptlm_v2_encoder_layers,
        concept_decoder_layers=args.conceptlm_v2_decoder_layers,
        concept_special_layers=args.conceptlm_v2_special_layers,
        concept_chunk_size=args.conceptlm_v2_chunk_size,
        concept_shift_feature=args.conceptlm_v2_shift_feature,
        concept_chunk_merge_method=args.conceptlm_v2_chunk_merge_method,
        concept_enable_chunk_dualpath_smoothing=args.conceptlm_v2_enable_chunk_dualpath_smoothing,
        concept_chunk_dualpath_alpha_init=args.conceptlm_v2_chunk_dualpath_alpha_init,
        concept_bottleneck_type=args.conceptlm_v2_bottleneck_type,
        concept_vq_patch_ratio=args.conceptlm_v2_vq_patch_ratio,
        concept_mlp_bottleneck_ratio=args.conceptlm_v2_mlp_bottleneck_ratio,
        concept_mlp_bottleneck_activation=args.conceptlm_v2_mlp_bottleneck_activation,
        concept_mlp_recon_loss_weight=args.conceptlm_v2_mlp_recon_loss_weight,
        concept_mlp_hlm_loss_weight=args.conceptlm_v2_mlp_hlm_loss_weight,
        concept_mlp_hidden_loss_weight=args.conceptlm_v2_mlp_hidden_loss_weight,
        concept_mlp_hlm_loss_type=args.conceptlm_v2_mlp_hlm_loss_type,
        concept_layer_norm_option=args.conceptlm_v2_layer_norm_option,
        concept_fusion_norm_alpha_init=args.conceptlm_v2_fusion_norm_alpha_init,
        concept_fusion_alpha_init=args.conceptlm_v2_fusion_alpha_init,
        concept_hlm_ffn_hidden_size=args.conceptlm_v2_hlm_ffn_hidden_size,
    )
