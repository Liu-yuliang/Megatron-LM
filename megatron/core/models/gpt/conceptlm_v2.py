# Copyright (c) 2026, ConceptLM contributors.

"""ConceptLM V2 prototype for Megatron-Core GPT-style training.

This module implements the main V2 path from the HuggingFace/Pythia checkout:

    token embeddings -> token encoder -> chunk compression -> concept/HLM tower
    -> MLP concept bottleneck prediction -> repeat/shift concept states
    -> token/concept fusion -> token decoder -> LM head

The implementation is intentionally scoped to the current V2 best line:
single-stage training, no tensor/pipeline/context parallel split inside the
ConceptLM branch, MLP concept bottleneck, and standard Megatron GPT data/loss
plumbing. Baseline GPT and ConceptLM V1 files are left untouched.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.conceptlm_v1 import ConceptRoPE
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import init_method_truncated_normal


@dataclass
class ConceptLMV2Output:
    """Training output consumed by ``pretrain_conceptlm_v2.loss_func``."""

    lm_loss: Tensor
    vq_loss: Tensor
    hlm_loss: Tensor
    concept_metrics: Dict[str, Tensor]


def _uses_scaled_truncated_init(config: Any) -> bool:
    return getattr(config, "init_method_variant", "normal") == "scaled_truncated_normal"


def _scaled_truncated_init_stds(hidden_size: int, num_layers: int) -> tuple[float, float]:
    base_std = math.sqrt(2.0 / (5.0 * int(hidden_size)))
    output_std = base_std / math.sqrt(2.0 * max(1, int(num_layers)))
    return base_std, output_std


def _init_linear_truncated_normal(module: nn.Linear, std: float) -> None:
    nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _reset_layernorm(module: nn.LayerNorm) -> None:
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)


def _total_concept_layers(config: Any, fallback: int) -> int:
    return (
        int(getattr(config, "conceptlm_encoder_layers", 0) or 0)
        + int(getattr(config, "conceptlm_decoder_layers", 0) or 0)
        + int(getattr(config, "conceptlm_special_layers", 0) or 0)
    ) or int(fallback)


class ConceptMLPBottleneck(nn.Module):
    """Per-split two-layer MLP bottleneck used by the V2 non-VQ best setting."""

    def __init__(self, input_dim: int, latent_dim: int, eps: float, activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim, eps=eps),
            nn.Linear(input_dim, latent_dim),
        ]
        if activation == "gelu":
            layers.append(nn.GELU())
        elif activation != "none":
            raise ValueError(f"unsupported ConceptLM V2 MLP activation: {activation}")
        layers.append(nn.Linear(latent_dim, latent_dim))
        if activation == "gelu":
            layers.append(nn.GELU())
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, hidden_states: Tensor) -> Tensor:
        return self.encoder(hidden_states)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(hidden_states)
        reconstructed = self.decode(latents)
        return latents, reconstructed

    def reset_scaled_truncated_parameters(self, hidden_size: int) -> None:
        base_std, _ = _scaled_truncated_init_stds(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear_truncated_normal(module, base_std)
            elif isinstance(module, nn.LayerNorm):
                _reset_layernorm(module)


class ConceptCausalBlock(nn.Module):
    """Small full-causal Transformer block for the concept/HLM tower."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: Optional[int],
        dropout: float,
        eps: float,
        rope: Optional[ConceptRoPE],
        activation: Literal["gelu", "silu"] = "gelu",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = float(dropout)
        self.rope = rope
        self.activation = activation

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.pre_mlp_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size or hidden_size * 4)
        self.fc2 = nn.Linear(ffn_hidden_size or hidden_size * 4, hidden_size)
        self.mlp_dropout = nn.Dropout(dropout)

    def reset_scaled_truncated_parameters(self, total_layers: int) -> None:
        base_std, output_std = _scaled_truncated_init_stds(self.hidden_size, total_layers)
        _reset_layernorm(self.input_layernorm)
        _init_linear_truncated_normal(self.qkv, base_std)
        _init_linear_truncated_normal(self.proj, output_std)
        _reset_layernorm(self.pre_mlp_layernorm)
        _init_linear_truncated_normal(self.fc1, base_std)
        _init_linear_truncated_normal(self.fc2, output_std)

    def _activate(self, hidden_states: Tensor) -> Tensor:
        if self.activation == "silu":
            return F.silu(hidden_states)
        return F.gelu(hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        qkv = self.qkv(hidden_states).view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        if self.rope is not None:
            query = self.rope(query)
            key = self.rope(key)

        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(
            residual.shape[0], residual.shape[1], self.hidden_size
        )
        hidden_states = residual + self.attn_dropout(self.proj(attn_output))

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.fc2(self._activate(self.fc1(hidden_states)))
        return residual + self.mlp_dropout(hidden_states)


class ConceptPredictorV2(nn.Module):
    """Concept/HLM tower that predicts the next MLP latent for every split."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        num_splits: int,
        latent_dim: int,
        ffn_hidden_size: Optional[int],
        dropout: float,
        eps: float,
        rotary_percent: float,
        rotary_base: int,
        rotary_interleaved: bool,
        position_stride: int,
        use_rope: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("ConceptLM V2 special_layers must be positive")
        rope = (
            ConceptRoPE(
                head_dim=hidden_size // num_heads,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
                rotary_interleaved=rotary_interleaved,
                position_stride=position_stride,
            )
            if use_rope
            else None
        )
        self.layers = nn.ModuleList(
            [
                ConceptCausalBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    dropout=dropout,
                    eps=eps,
                    rope=rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(hidden_size, latent_dim) for _ in range(num_splits)]
        )

    def reset_scaled_truncated_parameters(self, hidden_size: int, total_layers: int) -> None:
        base_std, _ = _scaled_truncated_init_stds(hidden_size, total_layers)
        for layer in self.layers:
            layer.reset_scaled_truncated_parameters(total_layers)
        _reset_layernorm(self.final_layernorm)
        for head in self.prediction_heads:
            _init_linear_truncated_normal(head, base_std)

    def forward(self, concept_hidden: Tensor) -> Tensor:
        hidden_states = concept_hidden
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)
        return torch.stack([head(hidden_states) for head in self.prediction_heads], dim=2)


def _clone_transformer_config(config, num_layers: int, init_num_layers: Optional[int] = None):
    """Return a shallow config clone with a different local block depth."""

    block_config = copy.copy(config)
    block_config.num_layers = int(num_layers)
    if _uses_scaled_truncated_init(config) and init_num_layers is not None:
        _, output_std = _scaled_truncated_init_stds(config.hidden_size, init_num_layers)
        block_config.output_layer_init_method = init_method_truncated_normal(output_std)
    return block_config


class ConceptLMV2Model(GPTModel):
    """Megatron GPTModel subclass implementing the ConceptLM V2 forward path."""

    def __init__(
        self,
        *args: Any,
        concept_encoder_layers: int = 6,
        concept_decoder_layers: int = 6,
        concept_special_layers: int = 6,
        concept_chunk_size: int = 4,
        concept_shift_feature: bool = True,
        concept_chunk_merge_method: Literal["meanpooling", "first", "last"] = "meanpooling",
        concept_enable_chunk_dualpath_smoothing: bool = False,
        concept_chunk_dualpath_alpha_init: float = 0.5,
        concept_bottleneck_type: Literal["mlp"] = "mlp",
        concept_vq_patch_ratio: int = 1,
        concept_mlp_bottleneck_ratio: float = 0.25,
        concept_mlp_bottleneck_activation: Literal["gelu", "none"] = "gelu",
        concept_mlp_recon_loss_weight: float = 1.0,
        concept_mlp_hlm_loss_weight: float = 1.0,
        concept_mlp_hidden_loss_weight: float = 1.0,
        concept_mlp_hlm_loss_type: Literal["mse", "cosine"] = "mse",
        concept_layer_norm_option: Literal[
            "rawadd", "rawadd_scalar_gate", "normed_add"
        ] = "normed_add",
        concept_fusion_norm_alpha_init: float = 1.0,
        concept_fusion_alpha_init: float = 1.0,
        concept_hlm_ffn_hidden_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if concept_encoder_layers <= 0:
            raise ValueError("concept_encoder_layers must be positive")
        if concept_decoder_layers <= 0:
            raise ValueError("concept_decoder_layers must be positive")
        if concept_chunk_size <= 0:
            raise ValueError("concept_chunk_size must be positive")
        if concept_bottleneck_type != "mlp":
            raise NotImplementedError(
                "This Megatron V2 port currently implements the best non-VQ MLP bottleneck path."
            )
        if concept_chunk_merge_method not in ("meanpooling", "first", "last"):
            raise ValueError(f"unsupported chunk merge method: {concept_chunk_merge_method}")
        if concept_layer_norm_option not in ("rawadd", "rawadd_scalar_gate", "normed_add"):
            raise ValueError(f"unsupported V2 fusion option: {concept_layer_norm_option}")
        if concept_vq_patch_ratio <= 0:
            raise ValueError("concept_vq_patch_ratio must be positive")
        if self.config.num_attention_heads % concept_vq_patch_ratio != 0:
            raise ValueError(
                "num_attention_heads must be divisible by concept_vq_patch_ratio"
            )

        self.transformer_layer_spec = kwargs.get("transformer_layer_spec", self.transformer_layer_spec)
        hidden_size = self.config.hidden_size
        num_splits = self.config.num_attention_heads // concept_vq_patch_ratio
        if hidden_size % num_splits != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by {num_splits}")
        split_dim = hidden_size // num_splits
        latent_dim = max(1, int(split_dim * concept_mlp_bottleneck_ratio))
        eps = self.config.layernorm_epsilon

        self.concept_encoder_layers = int(concept_encoder_layers)
        self.concept_decoder_layers = int(concept_decoder_layers)
        self.concept_special_layers = int(concept_special_layers)
        self.concept_chunk_size = int(concept_chunk_size)
        self.concept_shift_feature = bool(concept_shift_feature)
        self.concept_chunk_merge_method = concept_chunk_merge_method
        self.concept_layer_norm_option = concept_layer_norm_option
        self.concept_num_splits = int(num_splits)
        self.concept_split_dim = int(split_dim)
        self.concept_latent_dim = int(latent_dim)
        self.concept_mlp_recon_loss_weight = float(concept_mlp_recon_loss_weight)
        self.concept_mlp_hlm_loss_weight = float(concept_mlp_hlm_loss_weight)
        self.concept_mlp_hidden_loss_weight = float(concept_mlp_hidden_loss_weight)
        self.concept_mlp_hlm_loss_type = concept_mlp_hlm_loss_type
        self.concept_enable_chunk_dualpath_smoothing = bool(
            concept_enable_chunk_dualpath_smoothing
        )

        self.encoder = TransformerBlock(
            config=_clone_transformer_config(
                self.config,
                self.concept_encoder_layers,
                init_num_layers=(
                    self.concept_encoder_layers
                    + self.concept_decoder_layers
                    + self.concept_special_layers
                ),
            ),
            spec=self.transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=False,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )
        self.decoder = TransformerBlock(
            config=_clone_transformer_config(
                self.config,
                self.concept_decoder_layers,
                init_num_layers=(
                    self.concept_encoder_layers
                    + self.concept_decoder_layers
                    + self.concept_special_layers
                ),
            ),
            spec=self.transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=True,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

        if self.concept_enable_chunk_dualpath_smoothing:
            self.chunk_dualpath_b_proj = nn.Linear(hidden_size, hidden_size)
            self.chunk_dualpath_a_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.chunk_dualpath_b_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.chunk_dualpath_alpha = nn.Parameter(
                torch.tensor(float(concept_chunk_dualpath_alpha_init))
            )

        self.mlp_bottlenecks = nn.ModuleList(
            [
                ConceptMLPBottleneck(
                    input_dim=split_dim,
                    latent_dim=latent_dim,
                    eps=eps,
                    activation=concept_mlp_bottleneck_activation,
                )
                for _ in range(num_splits)
            ]
        )
        self.concept_predictor = ConceptPredictorV2(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            num_layers=self.concept_special_layers,
            num_splits=num_splits,
            latent_dim=latent_dim,
            ffn_hidden_size=concept_hlm_ffn_hidden_size,
            dropout=self.config.hidden_dropout,
            eps=eps,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rotary_interleaved=self.config.rotary_interleaved,
            position_stride=self.concept_chunk_size,
            use_rope=self.position_embedding_type == "rope",
        )
        self._reset_plain_concept_parameters_if_needed(hidden_size)

        if self.concept_layer_norm_option == "normed_add":
            self.fusion_tok_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.fusion_hl_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.fusion_norm_alpha = nn.Parameter(
                torch.tensor(float(concept_fusion_norm_alpha_init))
            )
        elif self.concept_layer_norm_option == "rawadd_scalar_gate":
            self.fusion_alpha = nn.Parameter(torch.tensor(float(concept_fusion_alpha_init)))

    def _reset_plain_concept_parameters_if_needed(self, hidden_size: int) -> None:
        if not _uses_scaled_truncated_init(self.config):
            return
        total_layers = (
            self.concept_encoder_layers + self.concept_decoder_layers + self.concept_special_layers
        )
        if self.concept_enable_chunk_dualpath_smoothing:
            base_std, _ = _scaled_truncated_init_stds(hidden_size, total_layers)
            _init_linear_truncated_normal(self.chunk_dualpath_b_proj, base_std)
            _reset_layernorm(self.chunk_dualpath_a_norm)
            _reset_layernorm(self.chunk_dualpath_b_norm)
        for bottleneck in self.mlp_bottlenecks:
            bottleneck.reset_scaled_truncated_parameters(hidden_size)
        self.concept_predictor.reset_scaled_truncated_parameters(hidden_size, total_layers)

    def _assert_supported_v2_runtime(
        self,
        decoder_input: Tensor,
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> None:
        if not self.pre_process or not self.post_process:
            raise NotImplementedError("ConceptLM V2 currently requires one local model stage")
        if inference_context is not None and not self.training:
            raise NotImplementedError("ConceptLM V2 inference/generation is not implemented")
        if packed_seq_params is not None:
            raise NotImplementedError("ConceptLM V2 does not support packed sequences yet")
        if self.config.tensor_model_parallel_size != 1:
            raise NotImplementedError("ConceptLM V2 currently requires tensor_model_parallel_size=1")
        if self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                "ConceptLM V2 currently requires pipeline_model_parallel_size=1"
            )
        if self.config.context_parallel_size != 1:
            raise NotImplementedError("ConceptLM V2 currently requires context_parallel_size=1")
        if self.config.sequence_parallel:
            raise NotImplementedError("ConceptLM V2 currently does not support sequence_parallel")
        if self.config.calculate_per_token_loss:
            raise NotImplementedError("ConceptLM V2 currently requires calculate_per_token_loss=False")
        if decoder_input is None:
            raise NotImplementedError("ConceptLM V2 needs local embedding hidden states")

    def _merge_token_chunks(self, token_states: Tensor) -> tuple[Tensor, int]:
        seq_len, batch_size, hidden_size = token_states.shape
        num_chunks = seq_len // self.concept_chunk_size
        usable_seq_len = num_chunks * self.concept_chunk_size
        if num_chunks < 2:
            raise ValueError(
                "ConceptLM V2 needs at least two concept chunks; "
                f"got seq_len={seq_len}, chunk_size={self.concept_chunk_size}"
            )
        chunked = token_states[:usable_seq_len].reshape(
            num_chunks, self.concept_chunk_size, batch_size, hidden_size
        )
        if self.concept_chunk_merge_method == "meanpooling":
            merged = chunked.mean(dim=1)
        elif self.concept_chunk_merge_method == "first":
            merged = chunked[:, 0]
        else:
            merged = chunked[:, -1]

        if self.concept_enable_chunk_dualpath_smoothing:
            b_path = self.chunk_dualpath_b_proj(merged)
            b_prev = torch.cat((torch.zeros_like(b_path[:1]), b_path[:-1]), dim=0)
            merged = self.chunk_dualpath_a_norm(merged) + self.chunk_dualpath_alpha * (
                self.chunk_dualpath_b_norm(b_prev)
            )
        return merged, usable_seq_len

    def _split_concepts(self, concept_hidden: Tensor) -> Tensor:
        return concept_hidden.reshape(
            concept_hidden.shape[0],
            concept_hidden.shape[1],
            self.concept_num_splits,
            self.concept_split_dim,
        )

    def _latent_prediction_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        if self.concept_mlp_hlm_loss_type == "cosine":
            return 1.0 - F.cosine_similarity(
                predicted.float(), target.detach().float(), dim=-1
            ).mean()
        return F.mse_loss(predicted, target.detach())

    def _decode_predicted_concepts(self, predicted_latents: Tensor) -> Tensor:
        decoded_splits = [
            self.mlp_bottlenecks[i].decode(predicted_latents[:, :, i, :])
            for i in range(self.concept_num_splits)
        ]
        return torch.stack(decoded_splits, dim=2).reshape(
            predicted_latents.shape[0],
            predicted_latents.shape[1],
            self.config.hidden_size,
        )

    def _repeat_shift_concepts(self, concept_states: Tensor, seq_len: int) -> Tensor:
        shifted = torch.cat((torch.zeros_like(concept_states[:1]), concept_states), dim=0)
        shifted[:1] = 0
        repeated = shifted.repeat_interleave(self.concept_chunk_size, dim=0)[: seq_len + 1]
        if self.concept_shift_feature:
            repeated = repeated[1:]
        else:
            repeated = repeated[:seq_len]
        if repeated.shape[0] < seq_len:
            pad = repeated.new_zeros(seq_len - repeated.shape[0], *repeated.shape[1:])
            repeated = torch.cat((repeated, pad), dim=0)
        return repeated[:seq_len]

    def _apply_fusion(self, token_states: Tensor, concept_states: Tensor) -> Tensor:
        if self.concept_layer_norm_option == "rawadd":
            return token_states + concept_states
        if self.concept_layer_norm_option == "rawadd_scalar_gate":
            return token_states + self.fusion_alpha.to(dtype=token_states.dtype) * concept_states
        return self.fusion_tok_norm(token_states) + self.fusion_norm_alpha.to(
            dtype=token_states.dtype
        ) * self.fusion_hl_norm(concept_states)

    def _concept_branch(self, encoder_hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        concept_hidden, usable_seq_len = self._merge_token_chunks(encoder_hidden_states)
        concept_target_hidden = concept_hidden.detach()
        split_hidden = self._split_concepts(concept_target_hidden)

        latents = []
        recon_losses = []
        for split_idx in range(self.concept_num_splits):
            latent, reconstructed = self.mlp_bottlenecks[split_idx](
                split_hidden[:, :, split_idx, :]
            )
            latents.append(latent)
            recon_losses.append(
                F.mse_loss(reconstructed, split_hidden[:, :, split_idx, :].detach())
            )
        target_latents = torch.stack(latents, dim=2)
        recon_loss = torch.stack(recon_losses).mean()
        vq_loss = self.concept_mlp_recon_loss_weight * recon_loss

        predicted_latents = self.concept_predictor(concept_hidden)
        pred_latent_loss = self._latent_prediction_loss(
            predicted_latents[:-1],
            target_latents[1:],
        )

        predicted_hidden = self._decode_predicted_concepts(predicted_latents)
        pred_hidden_loss = self._latent_prediction_loss(
            predicted_hidden[:-1],
            concept_target_hidden[1:],
        )
        hlm_loss = self.concept_mlp_hlm_loss_weight * (
            pred_latent_loss + self.concept_mlp_hidden_loss_weight * pred_hidden_loss
        )

        repeated_concepts = self._repeat_shift_concepts(predicted_hidden, encoder_hidden_states.shape[0])
        decoder_input = self._apply_fusion(
            encoder_hidden_states,
            repeated_concepts.to(dtype=encoder_hidden_states.dtype),
        )

        metrics = {
            "conceptlm_v2/usable_seq_len": torch.tensor(
                usable_seq_len, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v2/concept_chunks": torch.tensor(
                concept_hidden.shape[0], device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v2/latent_dim": torch.tensor(
                self.concept_latent_dim, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v2/recon_loss_raw": recon_loss.detach().float(),
            "conceptlm_v2/pred_latent_loss_raw": pred_latent_loss.detach().float(),
            "conceptlm_v2/pred_hidden_loss_raw": pred_hidden_loss.detach().float(),
        }
        return decoder_input, vq_loss, hlm_loss, metrics

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        output_processor: Optional[Callable[..., Tensor]] = None,
        output_processor_context: Optional[Any] = None,
    ) -> Tensor | ConceptLMV2Output:
        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        from megatron.core.utils import deprecate_inference_params

        inference_context = deprecate_inference_params(inference_context, inference_params)
        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preproc_output[:6]
        rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None

        self._assert_supported_v2_runtime(decoder_input, inference_context, packed_seq_params)

        encoder_hidden_states = self.encoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            **(extra_block_kwargs or {}),
        )
        decoder_hidden_states, vq_loss, hlm_loss, concept_metrics = self._concept_branch(
            encoder_hidden_states
        )
        hidden_states = self.decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            **(extra_block_kwargs or {}),
        )

        if labels is None:
            return self._postprocess(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                mtp_in_postprocess=self.mtp_process,
                loss_mask=loss_mask,
                decoder_input=decoder_hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                runtime_gather_output=runtime_gather_output,
                extra_block_kwargs=extra_block_kwargs,
                inference_context=inference_context,
                output_processor=output_processor,
                output_processor_context=output_processor_context,
            )

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        logits = self._scale_logits(logits)
        lm_output = self.compute_language_model_loss(labels, logits)

        return ConceptLMV2Output(
            lm_loss=lm_output,
            vq_loss=vq_loss,
            hlm_loss=hlm_loss,
            concept_metrics=concept_metrics,
        )
