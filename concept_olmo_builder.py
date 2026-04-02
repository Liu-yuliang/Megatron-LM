from __future__ import annotations

from argparse import ArgumentParser
from typing import Optional

import torch.nn.functional as F

from megatron.core.models.concept_olmo import ConceptOLMoModel
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml


def add_concept_olmo_extra_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("concept-olmo")
    group.add_argument("--concept-encoder-num-layers", type=int, default=0)
    group.add_argument("--concept-decoder-num-layers", type=int, default=-1)
    group.add_argument("--concept-hlm-num-layers", type=int, default=2)
    group.add_argument("--concept-chunk-size", type=int, default=4)
    group.add_argument("--concept-vq-patch-ratio", type=int, default=1)
    group.add_argument("--concept-codebook-size", type=int, default=128)
    group.add_argument("--concept-include-loss", action="store_true")
    group.add_argument("--concept-commit-loss-weight", type=float, default=1.0)
    group.add_argument("--concept-hlm-loss-weight", type=float, default=1.0)
    group.add_argument("--concept-use-olmo-defaults", action="store_true")
    return parser


def _apply_olmo_defaults(config, args) -> None:
    if not args.concept_use_olmo_defaults:
        return
    config.hidden_size = 2048
    config.num_attention_heads = 16
    config.ffn_hidden_size = 8192
    config.num_query_groups = 16
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.normalization = "RMSNorm"
    config.layernorm_epsilon = 1e-6
    config.gated_linear_unit = True
    config.activation_func = F.silu
    config.add_bias_linear = False
    config.add_qkv_bias = False
    config.num_layers = 16


def _get_transformer_layer_spec(args, config):
    use_te = args.transformer_impl == "transformer_engine"
    if args.experimental_attention_variant is not None:
        return get_transformer_block_with_experimental_attention_variant_spec(config=config)
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            qk_l2_norm=config.qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
        )
    if config.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_spec(
            config.qk_layernorm,
            config.multi_latent_attention,
            qk_l2_norm=config.qk_l2_norm,
        )
    return get_gpt_layer_local_spec(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        config.experimental_attention_variant,
        normalization=config.normalization,
        use_kitchen=config.use_kitchen,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )


def concept_olmo_builder(
    args,
    pre_process: bool,
    post_process: bool,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    del vp_stage
    print_rank_0("building Concept-OLMo model ...")
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
    _apply_olmo_defaults(config, args)
    transformer_layer_spec = _get_transformer_layer_spec(args, config)
    decoder_num_layers = (
        args.concept_decoder_num_layers
        if args.concept_decoder_num_layers >= 0
        else config.num_layers
    )
    return ConceptOLMoModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        encoder_num_layers=args.concept_encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
        hlm_num_layers=args.concept_hlm_num_layers,
        chunk_size=args.concept_chunk_size,
        vq_patch_ratio=args.concept_vq_patch_ratio,
        codebook_size=args.concept_codebook_size,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        include_concept_loss=args.concept_include_loss,
        commit_loss_weight=args.concept_commit_loss_weight,
        hlm_loss_weight=args.concept_hlm_loss_weight,
        pg_collection=pg_collection,
    )
