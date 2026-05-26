# Copyright (c) 2026, Shanghai AI Laboratory. All rights reserved.

"""GPT-NeoX/Pythia layer spec for public training benchmarks.

Pythia uses GPT-NeoX parallel residual blocks: attention and MLP are both fed by
the same normalized layer input and their outputs are added to the residual in a
single residual update. Megatron's default GPT layer applies attention then MLP
sequentially, so the public Pythia launcher selects this spec explicitly.
"""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push


class PythiaParallelResidualTransformerLayer(TransformerLayer, BaseTransformerLayer):
    """Transformer layer matching GPT-NeoX/Pythia parallel residual ordering."""

    def forward(self, *args, **kwargs):
        hidden_states, context = self._forward_attention(*args, **kwargs)
        output = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )
        return output, context

    def _forward_attention(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        # Keep the same signature and call shape as TransformerLayer so the
        # standard TransformerBlock can drive this layer unchanged.
        inference_context = inference_context if inference_context is not None else inference_params
        residual = hidden_states
        if self.config.fp32_residual_connection:
            residual = residual.float()

        nvtx_range_push(suffix="pythia_self_attention")
        self._pythia_layer_input = hidden_states
        self._pythia_residual = residual
        self._pythia_attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="pythia_self_attention")
        return hidden_states, context

    def _forward_mlp(
        self,
        hidden_states,
        inference_context=None,
        padding_mask=None,
    ):
        del hidden_states, inference_context
        layer_input = self._pythia_layer_input
        residual = self._pythia_residual
        attention_output_with_bias = self._pythia_attention_output_with_bias

        nvtx_range_push(suffix="pythia_mlp")
        mlp_output_with_bias = apply_module(self.mlp)(layer_input, padding_mask=padding_mask)
        nvtx_range_pop(suffix="pythia_mlp")

        attn_output, attn_bias = attention_output_with_bias
        mlp_output, mlp_bias = mlp_output_with_bias
        combined_output = attn_output + mlp_output
        combined_bias: Optional[torch.Tensor]
        if attn_bias is None and mlp_bias is None:
            combined_bias = None
        elif attn_bias is None:
            combined_bias = mlp_bias
        elif mlp_bias is None:
            combined_bias = attn_bias
        else:
            combined_bias = attn_bias + mlp_bias

        nvtx_range_push(suffix="pythia_parallel_residual_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                (combined_output, combined_bias), residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="pythia_parallel_residual_bda")

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        self._pythia_layer_input = None
        self._pythia_residual = None
        self._pythia_attention_output_with_bias = None
        return output


def get_pythia_parallel_residual_layer_with_transformer_engine_spec(*args, **kwargs) -> ModuleSpec:
    """Return a TE-backed GPT-NeoX/Pythia parallel-residual transformer layer spec."""

    return ModuleSpec(
        module=PythiaParallelResidualTransformerLayer,
        submodules=get_pythia_parallel_residual_layer_with_transformer_engine_submodules(
            *args, **kwargs
        ),
    )


def get_pythia_parallel_residual_layer_with_transformer_engine_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    **kwargs,
) -> TransformerLayerSubmodules:
    if num_experts is not None:
        raise ValueError("Pythia dense models should not set num_experts")
    if qk_layernorm:
        raise ValueError("Pythia dense models do not use qk_layernorm")
    if multi_latent_attention:
        raise ValueError("Pythia dense models do not use multi_latent_attention")

    backend = TESpecProvider()
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_activation_func=kwargs.get("use_te_activation_func", False),
    )

    return TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_layer_norm_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=IdentityOp,
        mlp=mlp,
        mlp_bda=get_bias_dropout_add,
    )


pythia_parallel_residual_layer_spec = (
    get_pythia_parallel_residual_layer_with_transformer_engine_spec()
)
"""Import target used by `--spec megatron.core.models.gpt.pythia_layer_specs pythia_parallel_residual_layer_spec`."""
