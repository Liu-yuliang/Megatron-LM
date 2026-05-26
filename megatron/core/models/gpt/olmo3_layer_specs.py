# Copyright (c) 2026, Shanghai AI Laboratory. All rights reserved.

"""OLMo 3 layer spec for public training benchmarks.

OLMo 3 differs from Megatron's default GPT block in two places:

* Q/K RMSNorm is applied to the full Q/K projection vectors before splitting
  them into heads.
* Attention and MLP outputs are RMSNormed before the residual add.

The launchers use this spec to keep the Megatron model structure and parameter
count aligned with the Hugging Face OLMo 3 checkpoint.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import divide, make_viewless_tensor, nvtx_range_pop, nvtx_range_push
from megatron.core.transformer.utils import is_layer_window_attention

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TENorm, te
else:
    TENorm = None
    te = None


class Olmo3FullHiddenRMSNorm(torch.nn.Module):
    """RMSNorm over the full Q/K projection dimension, matching OLMo 3."""

    def __init__(self, *, config, hidden_size: int, eps: float):
        super().__init__()
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=config.params_dtype, device=device)
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class Olmo3FullHiddenTERMSNorm(te.pytorch.RMSNorm if HAVE_TE else torch.nn.Module):
    """Transformer Engine RMSNorm with the same parameter layout as the reference path."""

    def __init__(self, *, config, hidden_size: int, eps: float):
        if not HAVE_TE:
            raise ImportError("Transformer Engine is required for fused OLMo 3 Q/K RMSNorm")

        if config.use_cpu_initialization:
            device = "cpu"
        elif getattr(config, "init_model_with_meta_device", False):
            device = "meta"
        else:
            device = torch.cuda.current_device()

        super().__init__(
            normalized_shape=hidden_size,
            eps=eps,
            sequence_parallel=config.sequence_parallel,
            params_dtype=config.params_dtype,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            device=device,
    )


_FLASH_ATTN_FUNC = None
_FLASH_ATTN_IMPL = None


def _get_flash_attn_func():
    """Load FA3/FA2 lazily so CPU-only imports do not touch CUDA extensions."""

    global _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPL
    if _FLASH_ATTN_FUNC is not None:
        return _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPL

    impl = os.getenv("OLMO3_LOCAL_FLASH_IMPL", "fa3").strip().lower()
    errors = []

    if impl in ("fa3", "flash3", "flash_attn_3", "auto"):
        try:
            from flash_attn_3.flash_attn_interface import flash_attn_func

            _FLASH_ATTN_FUNC = flash_attn_func
            _FLASH_ATTN_IMPL = "fa3"
            return _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPL
        except Exception as exc:  # pragma: no cover - depends on cluster image
            errors.append(f"fa3: {exc!r}")

    if impl in ("fa2", "flash2", "flash_attn", "auto"):
        try:
            from flash_attn import flash_attn_func

            _FLASH_ATTN_FUNC = flash_attn_func
            _FLASH_ATTN_IMPL = "fa2"
            return _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPL
        except Exception as exc:  # pragma: no cover - depends on cluster image
            errors.append(f"fa2: {exc!r}")

    raise ImportError(
        "OLMo 3 local fallback requires flash attention training kernels. "
        f"Set OLMO3_LOCAL_FLASH_IMPL=fa3/fa2/auto. Import errors: {'; '.join(errors)}"
    )


class Olmo3FlashAttentionCore(torch.nn.Module):
    """Flash-attn core used by the non-TE OLMo 3 fallback path."""

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type,
        attention_type,
        attention_dropout=None,
        softmax_scale=None,
        cp_comm_type=None,
        pg_collection=None,
    ):
        super().__init__()
        del attention_type, cp_comm_type, pg_collection
        if config.context_parallel_size != 1:
            raise ValueError("OLMo 3 local flash attention fallback requires CP_SIZE=1")
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_dropout = (
            config.attention_dropout if attention_dropout is None else attention_dropout
        )
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / (config.kv_channels**0.5)
        )
        if is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, self.layer_number
        ):
            self.window_size = tuple(config.window_size)
        else:
            self.window_size = (-1, -1)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        if packed_seq_params is not None:
            raise ValueError("OLMo 3 local flash attention fallback does not support packed seq")
        if attention_bias is not None:
            raise ValueError("OLMo 3 local flash attention fallback does not support attention bias")
        if self.attention_dropout and self.attention_dropout != 0.0:
            raise ValueError("OLMo 3 local flash attention fallback expects attention_dropout=0")
        if query.size(2) != key.size(2):
            repeat = divide(query.size(2), key.size(2))
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        q = query.permute(1, 0, 2, 3).contiguous()
        k = key.permute(1, 0, 2, 3).contiguous()
        v = value.permute(1, 0, 2, 3).contiguous()

        flash_attn_func, flash_impl = _get_flash_attn_func()
        common_kwargs = {
            "softmax_scale": self.softmax_scale,
            "causal": True,
            "window_size": self.window_size,
        }
        if flash_impl == "fa3":
            output = flash_attn_func(
                q,
                k,
                v,
                deterministic=self.config.deterministic_mode,
                **common_kwargs,
            )
        else:
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                deterministic=self.config.deterministic_mode,
                **common_kwargs,
            )
        if isinstance(output, tuple):
            output = output[0]
        context = output.permute(1, 0, 2, 3).contiguous()
        return context.view(context.size(0), context.size(1), -1)


def _get_olmo3_qk_norm_cls():
    impl = os.getenv("OLMO3_QK_NORM_IMPL", "torch").strip().lower()
    if impl in ("torch", "pytorch", "reference", "ref"):
        return Olmo3FullHiddenRMSNorm, "torch"
    if impl in ("te", "fused", "te_fused", "transformer_engine"):
        return Olmo3FullHiddenTERMSNorm, "te"
    raise ValueError(
        "Unsupported OLMO3_QK_NORM_IMPL="
        f"{impl!r}; expected torch/reference or te/fused/transformer_engine"
    )


class Olmo3SelfAttention(SelfAttention):
    """Self-attention with full-hidden Q/K RMSNorm before head reshape."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.qk_layernorm:
            raise ValueError("OLMo 3 attention requires --qk-layernorm")
        if self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError("This OLMo 3 spec expects full multi-head attention, not GQA")
        qk_norm_cls, qk_norm_impl = _get_olmo3_qk_norm_cls()
        self.olmo3_qk_norm_impl = qk_norm_impl
        self.q_layernorm = qk_norm_cls(
            config=self.config,
            hidden_size=self.query_projection_size,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm = qk_norm_cls(
            config=self.config,
            hidden_size=self.kv_projection_size,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ):
        if key_value_states is not None:
            raise ValueError("OLMo 3 self-attention does not use cross-attention states")
        if output_gate:
            raise ValueError("OLMo 3 attention does not use attention output gates")
        if not split_qkv:
            raise ValueError("OLMo 3 full-hidden Q/K RMSNorm requires split QKV")
        if self.world_size != 1:
            raise ValueError("OLMo 3 full-hidden Q/K RMSNorm currently requires TP_SIZE=1")

        mixed_qkv, _ = apply_module(self.linear_qkv)(hidden_states)
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            num_query_heads_per_group * self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)

        query = query.reshape(query.size(0), query.size(1), self.query_projection_size)
        key = key.reshape(key.size(0), key.size(1), self.kv_projection_size)
        query = apply_module(self.q_layernorm)(query)
        key = apply_module(self.k_layernorm)(key)

        query = query.view(
            query.size(0),
            query.size(1),
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        key = key.view(
            key.size(0),
            key.size(1),
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
        )
        return query, key, value


class Olmo3TransformerLayer(TransformerLayer, BaseTransformerLayer):
    """Transformer layer matching OLMo 3 post-attention/post-MLP RMSNorm ordering."""

    def __init__(self, *args, post_norm_cls=None, **kwargs):
        if post_norm_cls is None:
            post_norm_cls = TENorm
        if post_norm_cls is None:
            raise ImportError("OLMo 3 Megatron spec requires Transformer Engine")
        super().__init__(*args, **kwargs)
        self.post_attention_layernorm = post_norm_cls(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm = post_norm_cls(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def _add_residual(self, output, residual):
        output, bias = output
        if bias is not None:
            output = output + bias
        if self.hidden_dropout:
            output = F.dropout(output, p=self.hidden_dropout, training=self.training)
        return residual + output

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
        del context_mask, kwargs
        if inference_params is not None and inference_context is None:
            inference_context = inference_params

        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states

        nvtx_range_push(suffix="olmo3_self_attention")
        attention_output_with_bias = self.self_attention(
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
        nvtx_range_pop(suffix="olmo3_self_attention")

        attention_output, attention_bias = attention_output_with_bias
        attention_output = apply_module(self.post_attention_layernorm)(attention_output)
        hidden_states = self._add_residual((attention_output, attention_bias), residual)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return hidden_states, context

    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
        del inference_context
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states

        nvtx_range_push(suffix="olmo3_mlp")
        mlp_output_with_bias = apply_module(self.mlp)(hidden_states, padding_mask=padding_mask)
        nvtx_range_pop(suffix="olmo3_mlp")

        mlp_output, mlp_bias = mlp_output_with_bias
        mlp_output = apply_module(self.post_feedforward_layernorm)(mlp_output)
        hidden_states = self._add_residual((mlp_output, mlp_bias), residual)
        return make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )


def get_olmo3_layer_with_transformer_engine_spec(*args, **kwargs) -> ModuleSpec:
    """Return a TE-backed OLMo 3 transformer layer spec."""

    return ModuleSpec(
        module=Olmo3TransformerLayer,
        submodules=get_olmo3_layer_with_transformer_engine_submodules(*args, **kwargs),
    )


def get_olmo3_layer_with_transformer_engine_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = True,
    multi_latent_attention: Optional[bool] = False,
    **kwargs,
) -> TransformerLayerSubmodules:
    if num_experts is not None:
        raise ValueError("OLMo 3 dense models should not set num_experts")
    if not qk_layernorm:
        raise ValueError("OLMo 3 dense models require qk_layernorm")
    if multi_latent_attention:
        raise ValueError("OLMo 3 dense models do not use multi_latent_attention")

    backend = TESpecProvider()
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_op_fuser=kwargs.get("use_te_op_fuser", False),
        use_te_activation_func=kwargs.get("use_te_activation_func", False),
    )
    mlp.keywords["submodules"].linear_fc1 = backend.column_parallel_linear()

    return TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=ModuleSpec(
            module=Olmo3SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
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


def get_olmo3_layer_local_spec(*args, **kwargs) -> ModuleSpec:
    """Return a non-TE OLMo 3 transformer layer spec using local linear/MLP/norm modules."""

    return ModuleSpec(
        module=Olmo3TransformerLayer,
        params={"post_norm_cls": Olmo3FullHiddenRMSNorm},
        submodules=get_olmo3_layer_local_submodules(*args, **kwargs),
    )


def get_olmo3_layer_local_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = True,
    multi_latent_attention: Optional[bool] = False,
    **kwargs,
) -> TransformerLayerSubmodules:
    if num_experts is not None:
        raise ValueError("OLMo 3 dense models should not set num_experts")
    if not qk_layernorm:
        raise ValueError("OLMo 3 dense models require qk_layernorm")
    if multi_latent_attention:
        raise ValueError("OLMo 3 dense models do not use multi_latent_attention")

    backend = LocalSpecProvider()
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    return TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=ModuleSpec(
            module=Olmo3SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=Olmo3FlashAttentionCore,
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


olmo3_layer_spec = get_olmo3_layer_with_transformer_engine_spec()
"""Import target used by `--spec megatron.core.models.gpt.olmo3_layer_specs olmo3_layer_spec`."""

olmo3_local_layer_spec = get_olmo3_layer_local_spec()
"""Import target used for the non-TE OLMo 3 fallback path."""
