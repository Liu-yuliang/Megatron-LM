# Copyright (c) 2026, ConceptLM contributors.

"""ConceptLM V1 prototype for GPT-style Megatron models.

This module intentionally keeps the baseline GPTModel path untouched.  It
implements the minimal architecture described in conceptlm_V1.md:

* use the normal GPT embedding output as token-level encoder hidden states;
* mean-pool every ``chunk_size`` tokens into concept hidden states;
* quantize concepts with a SimVQ-style per-head product codebook;
* predict the next concept with a small causal Transformer;
* inject predicted concept vectors back into the token path by addition.

The first version is deliberately conservative.  It is meant for single-stage
experiments before wiring the concept branch into Megatron pipeline/packed
sequence paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class ConceptLMV1Output:
    """Training output consumed by ``pretrain_conceptlm_v1.loss_func``."""

    lm_loss: Tensor
    ncp_loss: Tensor
    vq_loss: Tensor
    concept_metrics: Dict[str, Tensor]


class SimVQProductQuantizer(nn.Module):
    """Per-head product quantizer with a direct product codebook."""

    def __init__(
        self,
        hidden_size: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_size % num_codebooks != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_codebooks ({num_codebooks})"
            )
        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.head_dim = hidden_size // num_codebooks
        self.commitment_cost = commitment_cost

        self.codebook = nn.Parameter(torch.empty(num_codebooks, codebook_size, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.codebook, mean=0.0, std=0.02)

    def transformed_codebook(self) -> Tensor:
        return self.codebook

    def forward(self, concept_hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize concept hidden states.

        Args:
            concept_hidden: Tensor with shape [concept_seq, batch, hidden].

        Returns:
            quantized: [concept_seq, batch, hidden]
            code_ids: [concept_seq, batch, num_codebooks]
            vq_loss: scalar tensor
        """

        concept_seq, batch_size, hidden_size = concept_hidden.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        inputs = concept_hidden.reshape(
            concept_seq, batch_size, self.num_codebooks, self.head_dim
        )
        codebook = self.transformed_codebook()

        distances = (
            inputs.float().square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.einsum("cbhd,hnd->cbhn", inputs.float(), codebook.float())
            + codebook.float().square().sum(dim=-1).view(1, 1, self.num_codebooks, self.codebook_size)
        )
        code_ids = torch.argmin(distances, dim=-1)

        gather_index = code_ids.permute(2, 0, 1).reshape(self.num_codebooks, -1)
        quantized_heads = []
        for head_idx in range(self.num_codebooks):
            quantized_heads.append(codebook[head_idx].index_select(0, gather_index[head_idx]))
        quantized = torch.stack(quantized_heads, dim=0)
        quantized = quantized.reshape(
            self.num_codebooks, concept_seq, batch_size, self.head_dim
        ).permute(1, 2, 0, 3)
        quantized = quantized.reshape(concept_seq, batch_size, hidden_size).to(concept_hidden.dtype)

        codebook_loss = F.mse_loss(quantized, concept_hidden.detach())
        commitment_loss = F.mse_loss(quantized.detach(), concept_hidden)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        return quantized, code_ids, vq_loss


class ConceptRoPE(nn.Module):
    """RoPE over chunk positions, using token-scale position ids."""

    def __init__(
        self,
        head_dim: int,
        rotary_percent: float,
        rotary_base: int,
        rotary_interleaved: bool,
        position_stride: int,
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * rotary_percent)
        rotary_dim -= rotary_dim % 2
        if rotary_dim <= 0:
            raise ValueError(
                f"rotary_percent={rotary_percent} gives invalid rotary_dim={rotary_dim}"
            )
        if rotary_dim > head_dim:
            raise ValueError(f"rotary_dim={rotary_dim} exceeds head_dim={head_dim}")
        self.rotary_dim = rotary_dim
        self.rotary_interleaved = rotary_interleaved
        self.position_stride = position_stride
        inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        positions = (
            torch.arange(seq_len, device=device, dtype=torch.float32) * self.position_stride
        )
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, tensor: Tensor) -> Tensor:
        seq_len = tensor.shape[0]
        cos, sin = self._cos_sin(seq_len, tensor.device)
        cos = cos[:, None, None, :].to(dtype=tensor.dtype)
        sin = sin[:, None, None, :].to(dtype=tensor.dtype)

        tensor_rot = tensor[..., : self.rotary_dim]
        tensor_pass = tensor[..., self.rotary_dim :]
        if self.rotary_interleaved:
            x1 = tensor_rot[..., ::2]
            x2 = tensor_rot[..., 1::2]
            rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
            tensor_rot = rotated.flatten(start_dim=-2)
        else:
            x1, x2 = torch.chunk(tensor_rot, 2, dim=-1)
            tensor_rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return torch.cat((tensor_rot, tensor_pass), dim=-1)


class ConceptCausalSelfAttentionBlock(nn.Module):
    """Full causal self-attention block used by the concept HLM."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: Optional[int],
        dropout: float,
        rope: Optional[ConceptRoPE],
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout
        self.rope = rope

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.pre_mlp_layernorm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size or hidden_size * 4)
        self.fc2 = nn.Linear(ffn_hidden_size or hidden_size * 4, hidden_size)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        qkv = self.qkv(hidden_states).view(
            hidden_states.shape[0], hidden_states.shape[1], 3, self.num_heads, self.head_dim
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
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        return residual + self.mlp_dropout(hidden_states)


class ConceptPredictorV1(nn.Module):
    """Small full-causal HLM that predicts the next concept."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        num_codebooks: int,
        codebook_size: int,
        ffn_hidden_size: Optional[int],
        dropout: float,
        rotary_percent: float,
        rotary_base: int,
        rotary_interleaved: bool,
        position_stride: int,
        use_rope: bool,
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        rope = (
            ConceptRoPE(
                head_dim=head_dim,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
                rotary_interleaved=rotary_interleaved,
                position_stride=position_stride,
            )
            if use_rope
            else None
        )
        self.use_rope = use_rope
        self.position_stride = position_stride
        self.layers = nn.ModuleList(
            [
                ConceptCausalSelfAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    dropout=dropout,
                    rope=rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_size)
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size) for _ in range(num_codebooks)]
        )

    def forward(self, concept_hidden: Tensor) -> Tensor:
        hidden = concept_hidden
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_layernorm(hidden)
        return torch.stack([head(hidden) for head in self.prediction_heads], dim=2)


class ConceptLMV1Model(GPTModel):
    """GPTModel subclass with an isolated ConceptLM V1 branch."""

    def __init__(
        self,
        *args: Any,
        concept_chunk_size: int = 4,
        concept_codebook_size: int = 128,
        concept_num_codebooks: Optional[int] = None,
        concept_special_layers: int = 2,
        concept_loss_type: Literal["hlm_MSE_loss", "hlm_CE_loss"] = "hlm_MSE_loss",
        concept_merge_mode: Literal["raw_logits", "softmax", "hard_top1"] = "raw_logits",
        concept_vq_commitment_cost: float = 0.25,
        concept_hlm_ffn_hidden_size: Optional[int] = None,
        concept_detach_ncp_target: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if concept_chunk_size <= 0:
            raise ValueError("concept_chunk_size must be positive")
        if concept_special_layers <= 0:
            raise ValueError("concept_special_layers must be positive")
        if concept_loss_type not in ("hlm_MSE_loss", "hlm_CE_loss"):
            raise ValueError(f"unsupported concept_loss_type: {concept_loss_type}")
        if concept_merge_mode not in ("raw_logits", "softmax", "hard_top1"):
            raise ValueError(f"unsupported concept_merge_mode: {concept_merge_mode}")

        hidden_size = self.config.hidden_size
        num_codebooks = concept_num_codebooks or self.config.num_attention_heads
        self.concept_chunk_size = concept_chunk_size
        self.concept_codebook_size = concept_codebook_size
        self.concept_num_codebooks = num_codebooks
        self.concept_loss_type = concept_loss_type
        self.concept_merge_mode = concept_merge_mode
        self.concept_detach_ncp_target = concept_detach_ncp_target
        norm_eps = self.config.layernorm_epsilon
        self.concept_vq_input_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.concept_encoder_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.concept_hlm_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.concept_injection_proj = nn.Linear(hidden_size, hidden_size)

        self.concept_quantizer = SimVQProductQuantizer(
            hidden_size=hidden_size,
            num_codebooks=num_codebooks,
            codebook_size=concept_codebook_size,
            commitment_cost=concept_vq_commitment_cost,
        )
        self.concept_predictor = ConceptPredictorV1(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            num_layers=concept_special_layers,
            num_codebooks=num_codebooks,
            codebook_size=concept_codebook_size,
            ffn_hidden_size=concept_hlm_ffn_hidden_size,
            dropout=self.config.hidden_dropout,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rotary_interleaved=self.config.rotary_interleaved,
            position_stride=concept_chunk_size,
            use_rope=self.position_embedding_type == "rope",
        )

    def _assert_supported_v1_runtime(
        self,
        decoder_input: Tensor,
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> None:
        if not self.pre_process:
            raise NotImplementedError("ConceptLM V1 currently requires pre_process=True")
        if inference_context is not None and not self.training:
            raise NotImplementedError("ConceptLM V1 inference path is not implemented")
        if packed_seq_params is not None:
            raise NotImplementedError("ConceptLM V1 does not support packed sequences yet")
        if self.config.tensor_model_parallel_size != 1:
            raise NotImplementedError("ConceptLM V1 currently supports tensor_model_parallel_size=1")
        if self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError("ConceptLM V1 currently supports pipeline_model_parallel_size=1")
        if self.config.context_parallel_size != 1:
            raise NotImplementedError("ConceptLM V1 currently supports context_parallel_size=1")
        if self.config.calculate_per_token_loss:
            raise NotImplementedError("ConceptLM V1 currently requires calculate_per_token_loss=False")
        if self.config.sequence_parallel:
            raise NotImplementedError("ConceptLM V1 currently does not support sequence_parallel")
        if decoder_input is None:
            raise NotImplementedError("ConceptLM V1 needs local embedding hidden states")

    def _mean_pool_concepts(self, token_hidden: Tensor) -> tuple[Tensor, int]:
        seq_len, batch_size, hidden_size = token_hidden.shape
        num_chunks = seq_len // self.concept_chunk_size
        usable_seq_len = num_chunks * self.concept_chunk_size
        if num_chunks < 2:
            raise ValueError(
                "ConceptLM V1 needs at least two concept chunks; "
                f"got seq_len={seq_len}, chunk_size={self.concept_chunk_size}"
            )
        pooled = token_hidden[:usable_seq_len].reshape(
            num_chunks, self.concept_chunk_size, batch_size, hidden_size
        )
        return pooled.mean(dim=1), usable_seq_len

    def _predict_vectors_from_logits(self, logits: Tensor) -> Tensor:
        codebook = self.concept_quantizer.transformed_codebook().to(logits.dtype)
        if self.concept_merge_mode == "raw_logits":
            vectors = torch.einsum("cbhn,hnd->cbhd", logits, codebook)
        elif self.concept_merge_mode == "softmax":
            weights = torch.softmax(logits, dim=-1)
            vectors = torch.einsum("cbhn,hnd->cbhd", weights, codebook)
        else:
            code_ids = torch.argmax(logits, dim=-1)
            gather_index = code_ids.permute(2, 0, 1).reshape(self.concept_num_codebooks, -1)
            vectors_per_head = []
            for head_idx in range(self.concept_num_codebooks):
                vectors_per_head.append(codebook[head_idx].index_select(0, gather_index[head_idx]))
            concept_seq, batch_size = logits.shape[0], logits.shape[1]
            vectors = torch.stack(vectors_per_head, dim=0)
            vectors = vectors.reshape(
                self.concept_num_codebooks,
                concept_seq,
                batch_size,
                self.concept_quantizer.head_dim,
            ).permute(1, 2, 0, 3)
        return vectors.reshape(logits.shape[0], logits.shape[1], self.config.hidden_size)

    def _concept_branch(self, encoder_hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        concept_hidden, usable_seq_len = self._mean_pool_concepts(encoder_hidden_states)
        concept_vq_hidden = self.concept_vq_input_norm(concept_hidden)
        # Match ZIP-style VQ: codebook learning can see detached encoder states,
        # but the VQ commitment term must not pull on the encoder hidden states.
        _, code_ids, vq_loss = self.concept_quantizer(concept_vq_hidden.detach())
        concept_logits = self.concept_predictor(concept_vq_hidden)
        predicted_vectors = self._predict_vectors_from_logits(concept_logits).to(
            encoder_hidden_states.dtype
        )

        target_concepts = concept_vq_hidden[1:]
        if self.concept_detach_ncp_target:
            target_concepts = target_concepts.detach()

        if self.concept_loss_type == "hlm_CE_loss":
            ncp_loss = F.cross_entropy(
                concept_logits[:-1].reshape(-1, self.concept_codebook_size).float(),
                code_ids[1:].reshape(-1),
            )
        else:
            ncp_loss = F.mse_loss(predicted_vectors[:-1].float(), target_concepts.float())

        shifted_vectors = torch.cat((
            predicted_vectors.new_zeros(
                1,
                predicted_vectors.shape[1],
                predicted_vectors.shape[2],
            ),
            predicted_vectors,
        ), dim=0)
        repeated = shifted_vectors.repeat_interleave(
            self.concept_chunk_size, dim=0
        )[: encoder_hidden_states.shape[0] + 1]
        repeated = repeated[1:]
        if repeated.shape[0] < encoder_hidden_states.shape[0]:
            pad = repeated.new_zeros(
                encoder_hidden_states.shape[0] - repeated.shape[0],
                repeated.shape[1],
                repeated.shape[2],
            )
            repeated = torch.cat((repeated, pad), dim=0)
        repeated = repeated.to(encoder_hidden_states.dtype)

        metrics = {
            "conceptlm/usable_seq_len": torch.tensor(
                usable_seq_len, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm/concept_chunks": torch.tensor(
                concept_hidden.shape[0], device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm/chunk_rope_stride": torch.tensor(
                self.concept_chunk_size,
                device=encoder_hidden_states.device,
                dtype=torch.float32,
            ),
        }
        decoder_hidden_states = self.concept_encoder_norm(
            encoder_hidden_states
        ) + self.concept_injection_proj(self.concept_hlm_norm(repeated))
        return decoder_hidden_states, ncp_loss, vq_loss, metrics

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
    ) -> Tensor | ConceptLMV1Output:
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

        self._assert_supported_v1_runtime(decoder_input, inference_context, packed_seq_params)
        decoder_input, ncp_loss, vq_loss, concept_metrics = self._concept_branch(decoder_input)

        hidden_states = self.decoder(
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
                decoder_input=decoder_input,
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

        from pretrain_gpt import (
            _current_train_step,
            _is_exit_hidden_rank_logger_rank,
            _log_scalar_metrics,
            _maybe_log_exit_hidden_rank,
        )
        from megatron.training import get_args

        _maybe_log_exit_hidden_rank(hidden_states)
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        logits = self._scale_logits(logits)

        lm_output = self.compute_language_model_loss(labels, logits)
        z_loss_multiplier = float(os.environ.get("OLMO3_Z_LOSS_MULTIPLIER", "0") or 0.0)
        if z_loss_multiplier > 0.0:
            z_losses = torch.logsumexp(logits, dim=-1).square() * z_loss_multiplier
            lm_output = lm_output + z_losses.transpose(0, 1).contiguous()
            if loss_mask is not None and _is_exit_hidden_rank_logger_rank():
                with torch.no_grad():
                    mask = loss_mask.float()
                    denom = mask.sum().clamp_min(1.0)
                    _log_scalar_metrics(
                        {"olmo3_z_loss": float((z_losses.transpose(0, 1).detach() * mask).sum().item() / denom.item())},
                        _current_train_step(get_args()),
                    )

        return ConceptLMV1Output(
            lm_loss=lm_output,
            ncp_loss=ncp_loss,
            vq_loss=vq_loss,
            concept_metrics=concept_metrics,
        )
