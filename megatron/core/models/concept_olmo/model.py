from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _clone_config(config: TransformerConfig, *, num_layers: int) -> TransformerConfig:
    cloned = copy.deepcopy(config)
    cloned.num_layers = max(int(num_layers), 1)
    return cloned


def _zeros_like_ref(ref: Tensor) -> Tensor:
    return ref.new_zeros(())


@dataclass
class ConceptOutput:
    token_loss: Tensor
    aux_loss: Tensor
    metrics: Dict[str, Tensor]


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook = nn.Parameter(torch.empty(codebook_size, dim))
        nn.init.normal_(self.codebook, mean=0.0, std=dim ** -0.5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        codebook_fp32 = self.codebook.float()
        x2 = (x_fp32 * x_fp32).sum(dim=-1, keepdim=True)
        c2 = (codebook_fp32 * codebook_fp32).sum(dim=-1).unsqueeze(0).unsqueeze(0)
        dots = torch.einsum("bnd,cd->bnc", x_fp32, codebook_fp32)
        distances = x2 + c2 - 2.0 * dots
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook).to(orig_dtype)
        commit_loss = F.mse_loss(x, quantized.detach()) + F.mse_loss(x.detach(), quantized)
        quantized = x + (quantized - x).detach()
        return quantized, indices, commit_loss

    def codes_from_probs(self, probs: Tensor) -> Tensor:
        return torch.einsum("bnc,cd->bnd", probs, self.codebook.to(probs.dtype))


class ConceptHLMHead(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        hidden_size: int,
        num_layers: int,
        num_vq_heads: int,
        codebook_size: int,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__()
        self.transformer = TransformerBlock(
            config=_clone_config(config, num_layers=max(num_layers, 1)),
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            pg_collection=pg_collection,
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size, bias=True) for _ in range(num_vq_heads)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tensor:
        hidden_states = self.transformer(
            hidden_states,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        hidden_states_bsh = hidden_states.transpose(0, 1).contiguous()
        logits = [head(hidden_states_bsh) for head in self.heads]
        return torch.stack(logits, dim=2)


class ConceptOLMoModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        hlm_num_layers: int,
        chunk_size: int,
        vq_patch_ratio: int,
        codebook_size: int,
        pre_process: bool = True,
        post_process: bool = True,
        position_embedding_type: str = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 500000,
        share_embeddings_and_output_weights: bool = False,
        include_concept_loss: bool = True,
        commit_loss_weight: float = 1.0,
        hlm_loss_weight: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.encoder_num_layers = int(encoder_num_layers)
        self.decoder_num_layers = int(decoder_num_layers)
        self.hlm_num_layers = int(hlm_num_layers)
        self.chunk_size = max(int(chunk_size), 1)
        self.vq_patch_ratio = max(int(vq_patch_ratio), 1)
        self.codebook_size = int(codebook_size)
        self.include_concept_loss = include_concept_loss
        self.commit_loss_weight = float(commit_loss_weight)
        self.hlm_loss_weight = float(hlm_loss_weight)
        self.num_vq_heads = max(1, config.num_attention_heads // self.vq_patch_ratio)
        if config.hidden_size % self.num_vq_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_vq_heads "
                f"({self.num_vq_heads})"
            )
        self.vq_head_dim = config.hidden_size // self.num_vq_heads

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=not config.sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        self.rotary_pos_emb = None
        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=None,
                rotary_base=rotary_base,
                rope_scaling=False,
                rope_scaling_factor=1.0,
                use_cpu_initialization=config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        self.encoder = None
        if self.encoder_num_layers > 0:
            self.encoder = TransformerBlock(
                config=_clone_config(config, num_layers=self.encoder_num_layers),
                spec=transformer_layer_spec,
                pre_process=True,
                post_process=True,
                pg_collection=self.pg_collection,
            )

        self.decoder = None
        if self.decoder_num_layers > 0:
            self.decoder = TransformerBlock(
                config=_clone_config(config, num_layers=self.decoder_num_layers),
                spec=transformer_layer_spec,
                pre_process=True,
                post_process=True,
                pg_collection=self.pg_collection,
            )

        self.vq_input_norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.concept_encoder_norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.concept_hlm_norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self.vqs = nn.ModuleList(
            [VectorQuantizer(self.vq_head_dim, self.codebook_size) for _ in range(self.num_vq_heads)]
        )
        self.hlm = ConceptHLMHead(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            hidden_size=config.hidden_size,
            num_layers=max(self.hlm_num_layers, 1),
            num_vq_heads=self.num_vq_heads,
            codebook_size=self.codebook_size,
            pg_collection=self.pg_collection,
        )

        if self.post_process:
            self.output_layer = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=None,
                grad_output_buffer=None,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if self.encoder is not None:
            self.encoder.set_input_tensor(input_tensor)
        if self.decoder is not None:
            self.decoder.set_input_tensor(input_tensor)

    def _get_rotary_pos_emb(
        self, input_ids: Tensor, packed_seq_params: Optional[PackedSeqParams]
    ) -> Optional[Tensor]:
        if self.rotary_pos_emb is None:
            return None
        rotary_seq_len = input_ids.shape[1]
        packed = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        return self.rotary_pos_emb(
            rotary_seq_len,
            packed_seq=packed,
            cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
        )

    def _get_rotary_pos_emb_for_length(self, seq_len: int) -> Optional[Tensor]:
        if self.rotary_pos_emb is None:
            return None
        return self.rotary_pos_emb(seq_len, packed_seq=False, cp_group=None)

    def _run_vq(self, hidden_states_bsh: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        vq_input = self.vq_input_norm(hidden_states_bsh)
        head_inputs = vq_input.view(
            vq_input.shape[0],
            vq_input.shape[1],
            self.num_vq_heads,
            self.vq_head_dim,
        )
        quantized_per_head = []
        indices_per_head = []
        commit_loss = _zeros_like_ref(hidden_states_bsh)
        for head_idx, vq in enumerate(self.vqs):
            quantized, indices, this_loss = vq(head_inputs[:, :, head_idx, :])
            quantized_per_head.append(quantized)
            indices_per_head.append(indices)
            commit_loss = commit_loss + this_loss
        quantized = torch.cat(quantized_per_head, dim=-1)
        indices = torch.stack(indices_per_head, dim=2)
        return quantized, indices, commit_loss

    def _concept_branch(
        self,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        encoder_hidden_states_bsh = encoder_hidden_states.transpose(0, 1).contiguous()
        batch_size, seq_len, hidden_size = encoder_hidden_states_bsh.shape
        usable_tokens = (seq_len // self.chunk_size) * self.chunk_size
        if usable_tokens == 0:
            zero = _zeros_like_ref(encoder_hidden_states_bsh)
            return encoder_hidden_states, zero, zero, {}

        pooled = encoder_hidden_states_bsh[:, :usable_tokens, :].reshape(
            batch_size,
            usable_tokens // self.chunk_size,
            self.chunk_size,
            hidden_size,
        ).mean(dim=2)
        hlm_attention_mask = None
        if attention_mask is not None:
            hlm_attention_mask = attention_mask[
                ...,
                :usable_tokens:self.chunk_size,
                :usable_tokens:self.chunk_size,
            ].contiguous()

        _, vq_indices, commit_loss = self._run_vq(pooled)
        pooled_sbh = pooled.transpose(0, 1).contiguous()
        hlm_logits = self.hlm(
            pooled_sbh,
            attention_mask=hlm_attention_mask,
            rotary_pos_emb=self._get_rotary_pos_emb_for_length(pooled.shape[1]),
            packed_seq_params=None,
        )

        decoded_heads = []
        head_accs = []
        for head_idx, vq in enumerate(self.vqs):
            head_logits = hlm_logits[:, :, head_idx, :]
            head_probs = torch.softmax(head_logits, dim=-1)
            decoded_heads.append(vq.codes_from_probs(head_probs))
            head_pred = head_logits.argmax(dim=-1)
            head_accs.append((head_pred == vq_indices[:, :, head_idx]).float().mean())
        decoded = torch.cat(decoded_heads, dim=-1)

        hlm_loss = _zeros_like_ref(decoded)
        if decoded.shape[1] > 1:
            hlm_loss = F.mse_loss(decoded[:, :-1, :], pooled[:, 1:, :].detach())

        shifted = torch.cat((decoded.new_zeros(batch_size, 1, hidden_size), decoded), dim=1)
        repeated = shifted.repeat_interleave(self.chunk_size, dim=1)[:, : seq_len + 1, :]
        repeated = repeated[:, 1:, :]

        decoder_hidden_states_bsh = (
            self.concept_encoder_norm(encoder_hidden_states_bsh)
            + self.concept_hlm_norm(repeated)
        )
        decoder_hidden_states = decoder_hidden_states_bsh.transpose(0, 1).contiguous()
        metrics = {
            "concept/commit_loss": commit_loss.detach(),
            "concept/hlm_mse_loss": hlm_loss.detach(),
            "concept/vq_head_acc": torch.stack(head_accs).mean().detach(),
        }
        return decoder_hidden_states, commit_loss, hlm_loss, metrics

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context=None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params=None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_spec_decode: Optional[bool] = None,
    ):
        del decoder_input
        del inference_context
        del extra_block_kwargs
        del inference_params
        del loss_mask
        del padding_mask
        del is_spec_decode

        if not self.pre_process:
            raise NotImplementedError("ConceptOLMoModel currently requires pre_process=True")
        if self.config.context_parallel_size > 1:
            raise NotImplementedError("ConceptOLMoModel currently does not support context parallelism")

        hidden_states = self.embedding(input_ids=input_ids, position_ids=position_ids)
        rotary_pos_emb = self._get_rotary_pos_emb(input_ids, packed_seq_params)

        if self.encoder is not None:
            encoder_hidden_states = self.encoder(
                hidden_states,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
        else:
            encoder_hidden_states = hidden_states

        commit_loss = _zeros_like_ref(encoder_hidden_states)
        hlm_loss = _zeros_like_ref(encoder_hidden_states)
        metrics: Dict[str, Tensor] = {}
        if self.hlm_num_layers > 0 and len(self.vqs) > 0:
            decoder_hidden_states, commit_loss, hlm_loss, metrics = self._concept_branch(
                encoder_hidden_states, attention_mask, packed_seq_params
            )
        else:
            decoder_hidden_states = encoder_hidden_states

        if self.decoder is not None:
            decoder_hidden_states = self.decoder(
                decoder_hidden_states,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )

        if not self.post_process:
            return decoder_hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            decoder_hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        logits = self._scale_logits(logits)

        if labels is None:
            return logits.transpose(0, 1).contiguous()

        token_loss = self.compute_language_model_loss(labels, logits)
        aux_loss = (
            self.commit_loss_weight * commit_loss + self.hlm_loss_weight * hlm_loss
            if self.include_concept_loss
            else _zeros_like_ref(token_loss)
        )
        metrics = dict(metrics)
        metrics["concept/aux_loss"] = aux_loss.detach()
        return ConceptOutput(token_loss=token_loss, aux_loss=aux_loss, metrics=metrics)
