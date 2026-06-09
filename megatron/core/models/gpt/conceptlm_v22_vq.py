# Copyright (c) 2026, ConceptLM contributors.

"""ConceptLM V2.2-VQ prototype.

V2.2-VQ keeps the V2.1 encoder/decoder DD and residual-flow plumbing, but
replaces the MLP bottleneck concept path with the V1-style product VQ codebook
and an explicit high-level concept prediction loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.conceptlm_v1 import SimVQProductQuantizer
from megatron.core.models.gpt.conceptlm_v2 import (
    _reset_layernorm,
    _scaled_truncated_init_stds,
    _uses_scaled_truncated_init,
)
from megatron.core.models.gpt.conceptlm_v21 import (
    ConceptLMV21Model,
    ConceptPredictorV21,
    _env_flag,
    _env_int,
    _v21_check_finite,
    _v21_current_iteration_for_debug,
    _v21_rank,
    _v21_tensor_summary,
)
from megatron.core.packed_seq_params import PackedSeqParams


@dataclass
class ConceptLMV22VQOutput:
    """Training output consumed by ``pretrain_conceptlm_v22_vq.loss_func``."""

    lm_loss: Tensor
    vq_loss: Tensor
    hlm_loss: Tensor
    concept_metrics: Dict[str, Tensor]


def _maybe_print_v22_vq_ce_debug(
    hidden_states: Tensor,
    logits: Tensor,
    labels: Tensor,
    lm_output: Tensor,
    input_ids: Optional[Tensor] = None,
) -> None:
    if not _env_flag("CONCEPTLM_V22_VQ_CE_DEBUG", False):
        return
    force_every_n = _env_int("CONCEPTLM_V22_VQ_CE_DEBUG_EVERY_N", 0)
    force_scan_logits = force_every_n > 0
    if force_every_n > 0:
        if not hasattr(_maybe_print_v22_vq_ce_debug, "_step"):
            _maybe_print_v22_vq_ce_debug._step = 0
        _maybe_print_v22_vq_ce_debug._step += 1
        force_scan_logits = (_maybe_print_v22_vq_ce_debug._step % force_every_n) == 0

    lm_output_detached = lm_output.detach()
    lm_is_finite = torch.isfinite(lm_output_detached)
    if bool(lm_is_finite.all().item()) and not force_scan_logits:
        return

    rank = _v21_rank()
    current_iter = _v21_current_iteration_for_debug()
    iter_text = "unknown" if current_iter is None else str(current_iter)
    print(
        f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
        f"triggered: iter={iter_text}, "
        f"lm_output_finite={bool(lm_is_finite.all().item())}, "
        f"force_scan_logits={force_scan_logits}",
        flush=True,
    )
    print(
        f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
        f"{_v21_tensor_summary('lm_output', lm_output)}",
        flush=True,
    )
    print(
        f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
        f"{_v21_tensor_summary('labels', labels)}",
        flush=True,
    )
    print(
        f"[ConceptLM V2.2-VQ CE debug][rank {rank}] labels_range: "
        f"min={int(labels.detach().min().item())}, max={int(labels.detach().max().item())}, "
        f"vocab={int(logits.shape[-1])}",
        flush=True,
    )

    # These scans touch full logits and are intentionally gated behind the debug flag.
    print(
        f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
        f"{_v21_tensor_summary('hidden_states', hidden_states)}",
        flush=True,
    )
    print(
        f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
        f"{_v21_tensor_summary('logits', logits)}",
        flush=True,
    )

    bad_positions = torch.nonzero(~lm_is_finite, as_tuple=False)
    if bad_positions.numel() == 0:
        return
    max_bad = max(1, _env_int("CONCEPTLM_V22_VQ_CE_DEBUG_MAX_BAD_TOKENS", 8))
    for pos in bad_positions[:max_bad]:
        b = int(pos[0].item())
        s = int(pos[1].item())
        label = int(labels.detach()[b, s].item())
        logit_row = logits.detach()[s, b]
        hidden_row = hidden_states.detach()[s, b]
        context_start = max(0, s - 4)
        context_end = min(int(labels.shape[1]), s + 5)
        label_context = labels.detach()[b, context_start:context_end].cpu().tolist()
        loss_context = lm_output_detached[b, context_start:context_end].float().cpu().tolist()
        if input_ids is not None and input_ids.dim() >= 2:
            input_context = input_ids.detach()[b, context_start:context_end].cpu().tolist()
        else:
            input_context = None
        target_logit = logit_row[label].float()
        logsumexp = torch.logsumexp(logit_row.float(), dim=-1)
        manual_ce = logsumexp - target_logit
        print(
            f"[ConceptLM V2.2-VQ CE debug][rank {rank}] bad_token b={b}, s={s}, "
            f"label={label}, loss={float(lm_output_detached[b, s].float().item())}, "
            f"target_logit={float(target_logit.item())}, "
            f"logsumexp={float(logsumexp.item())}, "
            f"manual_ce={float(manual_ce.item())}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.2-VQ CE debug][rank {rank}] bad_token_context "
            f"b={b}, s={s}, range=[{context_start},{context_end}), "
            f"input_ids={input_context}, labels={label_context}, lm_output={loss_context}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
            f"{_v21_tensor_summary(f'logits[s={s},b={b}]', logit_row)}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.2-VQ CE debug][rank {rank}] "
            f"{_v21_tensor_summary(f'hidden_states[s={s},b={b}]', hidden_row)}",
            flush=True,
        )


class ConceptLMV22VQModel(ConceptLMV21Model):
    """ConceptLM V2.2 with a V1-style true VQ concept bottleneck."""

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
        concept_codebook_size: int = 128,
        concept_num_codebooks: Optional[int] = None,
        concept_vq_commitment_cost: float = 0.25,
        concept_vq_merge_mode: Literal["raw_logits", "softmax", "hard_top1"] = "raw_logits",
        concept_hlm_loss_type: Literal["mse", "ce"] = "mse",
        concept_detach_hlm_target: bool = True,
        concept_layer_norm_option: Literal[
            "rawadd", "rawadd_scalar_gate", "normed_add"
        ] = "normed_add",
        concept_fusion_norm_alpha_init: float = 0.1,
        concept_fusion_alpha_init: float = 1.0,
        concept_hlm_ffn_hidden_size: Optional[int] = None,
        concept_dd_two_route_add: bool = False,
        concept_dd_two_route_add_concept_source: Literal[
            "final", "hlm_layers", "hlm_layers_plus_final"
        ] = "final",
        concept_dd_two_route_add_enable_raw_concept_route: bool = True,
        concept_dd_two_route_add_enable_final_concept_route: bool = True,
        concept_dd_two_route_add_beta_init: float = 0.3,
        concept_dd_two_route_add_every_n_layers: int = 1,
        concept_dd_two_route_add_concept_route_first_n: int = -1,
        concept_dd_two_route_add_decoder_hidden_size: int = 0,
        concept_dd_two_route_add_concept_hidden_size: int = 0,
        concept_dd_two_route_add_use_softmax: bool = True,
        concept_dd_two_route_add_disable_decoder_dd: bool = False,
        concept_dd_two_route_add_decoder_use_layernorm: bool = False,
        concept_dd_two_route_add_decoder_use_softmax: bool = True,
        concept_dd_two_route_add_concept_use_layernorm: bool = True,
        concept_dd_encoder_self_dd: bool = False,
        concept_dd_encoder_self_dd_every_n_layers: int = 1,
        concept_dd_encoder_self_dd_hidden_size: int = 0,
        concept_dd_encoder_self_dd_use_layernorm: bool = False,
        concept_dd_concept_self_dd: bool = False,
        concept_dd_concept_self_dd_every_n_layers: int = 1,
        concept_dd_concept_self_dd_hidden_size: int = 0,
        concept_dd_concept_self_dd_use_layernorm: bool = False,
        concept_enable_full_residual_flow: bool = False,
        concept_enable_concept_read_encoder: bool = False,
        concept_enable_decoder_read_encoder: bool = False,
        concept_enable_decoder_read_concept: bool = False,
        concept_residual_flow_beta_init: float = 0.02,
        concept_residual_flow_route_hidden_size: int = 0,
        concept_residual_flow_route_use_softmax: bool = True,
        concept_residual_flow_source_use_layernorm: bool = True,
        concept_residual_flow_shared_source_norm: bool = False,
        concept_final_read_concept_gate: bool = False,
        concept_final_read_concept_gate_init_final: float = 0.5,
        concept_final_read_concept_gate_target_final: float = 0.5,
        concept_compile_residual_flow_routes: bool = False,
        concept_compile_dd_routes: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            concept_encoder_layers=concept_encoder_layers,
            concept_decoder_layers=concept_decoder_layers,
            concept_special_layers=concept_special_layers,
            concept_chunk_size=concept_chunk_size,
            concept_shift_feature=concept_shift_feature,
            concept_chunk_merge_method=concept_chunk_merge_method,
            concept_enable_chunk_dualpath_smoothing=concept_enable_chunk_dualpath_smoothing,
            concept_chunk_dualpath_alpha_init=concept_chunk_dualpath_alpha_init,
            concept_bottleneck_type="mlp",
            concept_vq_patch_ratio=1,
            concept_mlp_recon_loss_weight=0.0,
            concept_mlp_hlm_loss_weight=0.0,
            concept_layer_norm_option=concept_layer_norm_option,
            concept_fusion_norm_alpha_init=concept_fusion_norm_alpha_init,
            concept_fusion_alpha_init=concept_fusion_alpha_init,
            concept_hlm_ffn_hidden_size=concept_hlm_ffn_hidden_size,
            concept_dd_two_route_add=concept_dd_two_route_add,
            concept_dd_two_route_add_concept_source=concept_dd_two_route_add_concept_source,
            concept_dd_two_route_add_enable_raw_concept_route=concept_dd_two_route_add_enable_raw_concept_route,
            concept_dd_two_route_add_enable_final_concept_route=concept_dd_two_route_add_enable_final_concept_route,
            concept_dd_two_route_add_beta_init=concept_dd_two_route_add_beta_init,
            concept_dd_two_route_add_every_n_layers=concept_dd_two_route_add_every_n_layers,
            concept_dd_two_route_add_concept_route_first_n=concept_dd_two_route_add_concept_route_first_n,
            concept_dd_two_route_add_decoder_hidden_size=concept_dd_two_route_add_decoder_hidden_size,
            concept_dd_two_route_add_concept_hidden_size=concept_dd_two_route_add_concept_hidden_size,
            concept_dd_two_route_add_use_softmax=concept_dd_two_route_add_use_softmax,
            concept_dd_two_route_add_disable_decoder_dd=concept_dd_two_route_add_disable_decoder_dd,
            concept_dd_two_route_add_decoder_use_layernorm=concept_dd_two_route_add_decoder_use_layernorm,
            concept_dd_two_route_add_decoder_use_softmax=concept_dd_two_route_add_decoder_use_softmax,
            concept_dd_two_route_add_concept_use_layernorm=concept_dd_two_route_add_concept_use_layernorm,
            concept_dd_encoder_self_dd=concept_dd_encoder_self_dd,
            concept_dd_encoder_self_dd_every_n_layers=concept_dd_encoder_self_dd_every_n_layers,
            concept_dd_encoder_self_dd_hidden_size=concept_dd_encoder_self_dd_hidden_size,
            concept_dd_encoder_self_dd_use_layernorm=concept_dd_encoder_self_dd_use_layernorm,
            concept_dd_concept_self_dd=concept_dd_concept_self_dd,
            concept_dd_concept_self_dd_every_n_layers=concept_dd_concept_self_dd_every_n_layers,
            concept_dd_concept_self_dd_hidden_size=concept_dd_concept_self_dd_hidden_size,
            concept_dd_concept_self_dd_use_layernorm=concept_dd_concept_self_dd_use_layernorm,
            concept_enable_full_residual_flow=concept_enable_full_residual_flow,
            concept_enable_concept_read_encoder=concept_enable_concept_read_encoder,
            concept_enable_decoder_read_encoder=concept_enable_decoder_read_encoder,
            concept_enable_decoder_read_concept=concept_enable_decoder_read_concept,
            concept_residual_flow_beta_init=concept_residual_flow_beta_init,
            concept_residual_flow_route_hidden_size=concept_residual_flow_route_hidden_size,
            concept_residual_flow_route_use_softmax=concept_residual_flow_route_use_softmax,
            concept_residual_flow_source_use_layernorm=concept_residual_flow_source_use_layernorm,
            concept_residual_flow_shared_source_norm=concept_residual_flow_shared_source_norm,
            concept_final_read_concept_gate=concept_final_read_concept_gate,
            concept_final_read_concept_gate_init_final=concept_final_read_concept_gate_init_final,
            concept_final_read_concept_gate_target_final=concept_final_read_concept_gate_target_final,
            concept_compile_residual_flow_routes=concept_compile_residual_flow_routes,
            concept_compile_dd_routes=concept_compile_dd_routes,
            **kwargs,
        )
        if hasattr(self, "mlp_bottlenecks"):
            del self.mlp_bottlenecks

        hidden_size = self.config.hidden_size
        num_codebooks = concept_num_codebooks or self.config.num_attention_heads
        if concept_codebook_size <= 0:
            raise ValueError("concept_codebook_size must be positive")
        if num_codebooks <= 0:
            raise ValueError("concept_num_codebooks must be positive")
        if hidden_size % num_codebooks != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by concept_num_codebooks ({num_codebooks})"
            )
        if concept_vq_merge_mode not in ("raw_logits", "softmax", "hard_top1"):
            raise ValueError(f"unsupported V2.2-VQ merge mode: {concept_vq_merge_mode}")
        if concept_hlm_loss_type not in ("mse", "ce"):
            raise ValueError(f"unsupported V2.2-VQ HLM loss type: {concept_hlm_loss_type}")

        eps = self.config.layernorm_epsilon
        self.concept_codebook_size = int(concept_codebook_size)
        self.concept_num_codebooks = int(num_codebooks)
        self.concept_vq_merge_mode = concept_vq_merge_mode
        self.concept_hlm_loss_type = concept_hlm_loss_type
        self.concept_detach_hlm_target = bool(concept_detach_hlm_target)
        self.concept_vq_input_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.concept_quantizer = SimVQProductQuantizer(
            hidden_size=hidden_size,
            num_codebooks=self.concept_num_codebooks,
            codebook_size=self.concept_codebook_size,
            commitment_cost=concept_vq_commitment_cost,
        )
        self.concept_predictor = ConceptPredictorV21(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            num_layers=self.concept_special_layers,
            num_splits=self.concept_num_codebooks,
            latent_dim=self.concept_codebook_size,
            ffn_hidden_size=concept_hlm_ffn_hidden_size,
            dropout=self.config.hidden_dropout,
            eps=eps,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rotary_interleaved=self.config.rotary_interleaved,
            position_stride=self.concept_chunk_size,
            use_rope=self.position_embedding_type == "rope",
            use_concept_self_dd=concept_dd_concept_self_dd,
            concept_self_dd_every_n_layers=concept_dd_concept_self_dd_every_n_layers,
            concept_self_dd_hidden_size=concept_dd_concept_self_dd_hidden_size,
            concept_self_dd_use_layernorm=concept_dd_concept_self_dd_use_layernorm,
            concept_self_dd_compile=concept_compile_dd_routes,
            enable_concept_read_encoder=self.concept_enable_concept_read_encoder,
            num_concept_read_encoder_sources=max(0, self.concept_encoder_layers - 1),
            residual_flow_beta_init=concept_residual_flow_beta_init,
            residual_flow_route_hidden_size=concept_residual_flow_route_hidden_size,
            residual_flow_route_use_softmax=concept_residual_flow_route_use_softmax,
            residual_flow_source_use_layernorm=concept_residual_flow_source_use_layernorm,
            residual_flow_shared_source_norm=concept_residual_flow_shared_source_norm,
            residual_flow_compile_routes=concept_compile_residual_flow_routes,
        )
        self._reset_v22_vq_parameters_if_needed(hidden_size)

    def _reset_v22_vq_parameters_if_needed(self, hidden_size: int) -> None:
        if not _uses_scaled_truncated_init(self.config):
            return
        total_layers = (
            self.concept_encoder_layers + self.concept_decoder_layers + self.concept_special_layers
        )
        base_std, _ = _scaled_truncated_init_stds(hidden_size, total_layers)
        for codebook in self.concept_quantizer.codebook:
            nn.init.trunc_normal_(
                codebook,
                mean=0.0,
                std=base_std,
                a=-3.0 * base_std,
                b=3.0 * base_std,
            )
        _reset_layernorm(self.concept_vq_input_norm)
        self.concept_predictor.reset_scaled_truncated_parameters(hidden_size, total_layers)

    def _predict_vectors_from_logits(self, logits: Tensor) -> Tensor:
        codebook = self.concept_quantizer.transformed_codebook().to(logits.dtype)
        if self.concept_vq_merge_mode == "raw_logits":
            vectors = torch.einsum("cbhk,hkd->cbhd", logits, codebook)
        elif self.concept_vq_merge_mode == "softmax":
            weights = torch.softmax(logits, dim=-1)
            vectors = torch.einsum("cbhk,hkd->cbhd", weights, codebook)
        else:
            code_ids = torch.argmax(logits, dim=-1)
            gather_index = code_ids.permute(2, 0, 1).reshape(self.concept_num_codebooks, -1)
            vectors_per_head = []
            for head_idx in range(self.concept_num_codebooks):
                vectors_per_head.append(codebook[head_idx].index_select(0, gather_index[head_idx]))
            concept_seq, batch_size = logits.shape[0], logits.shape[1]
            vectors = torch.stack(vectors_per_head, dim=0).reshape(
                self.concept_num_codebooks,
                concept_seq,
                batch_size,
                self.concept_quantizer.head_dim,
            ).permute(1, 2, 0, 3)
        return vectors.reshape(logits.shape[0], logits.shape[1], self.config.hidden_size)

    def _v22_route_metrics(self, device: torch.device) -> Dict[str, Tensor]:
        return {
            name.replace("conceptlm_v21/", "conceptlm_v22_vq/"): value
            for name, value in self._collect_v21_route_metrics(device).items()
        }

    def _concept_branch_v21(
        self,
        encoder_hidden_states: Tensor,
        encoder_raw_layer_states: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, Optional[tuple[Tensor, ...]]]:
        concept_hidden, usable_seq_len = self._merge_token_chunks(encoder_hidden_states)
        _v21_check_finite("v22_vq.concept_hidden", concept_hidden)
        concept_vq_hidden = self.concept_vq_input_norm(concept_hidden)
        _v21_check_finite("v22_vq.concept_vq_hidden", concept_vq_hidden)
        _, code_ids, vq_loss = self.concept_quantizer(concept_vq_hidden.detach())
        _v21_check_finite("v22_vq.vq_loss", vq_loss)

        encoder_concept_states = self._build_encoder_concept_states(encoder_raw_layer_states)
        _v21_check_finite("v22_vq.encoder_concept_states", encoder_concept_states)
        need_concept_layer_states = (
            self.dd_two_route_add is not None
            and self.concept_dd_two_route_add_concept_source
            in ("hlm_layers", "hlm_layers_plus_final")
        ) or self.decoder_read_concept_routes is not None
        concept_logits, concept_layer_states = self.concept_predictor(
            concept_vq_hidden,
            encoder_concept_states=encoder_concept_states,
            return_layer_states=need_concept_layer_states,
        )
        _v21_check_finite("v22_vq.concept_logits", concept_logits)
        predicted_vectors = self._predict_vectors_from_logits(concept_logits).to(
            encoder_hidden_states.dtype
        )
        _v21_check_finite("v22_vq.predicted_vectors", predicted_vectors)

        if self.concept_hlm_loss_type == "ce":
            hlm_loss = F.cross_entropy(
                concept_logits[:-1].reshape(-1, self.concept_codebook_size).float(),
                code_ids[1:].reshape(-1),
            )
        else:
            target_concepts = concept_vq_hidden[1:]
            if self.concept_detach_hlm_target:
                target_concepts = target_concepts.detach()
            hlm_loss = F.mse_loss(predicted_vectors[:-1].float(), target_concepts.float())
        _v21_check_finite("v22_vq.hlm_loss", hlm_loss)

        repeated_concepts = self._repeat_shift_concepts(
            predicted_vectors,
            encoder_hidden_states.shape[0],
        )
        _v21_check_finite("v22_vq.repeated_concepts", repeated_concepts)
        decoder_input = self._apply_fusion(
            encoder_hidden_states,
            repeated_concepts.to(dtype=encoder_hidden_states.dtype),
        )
        _v21_check_finite("v22_vq.decoder_input", decoder_input)

        metrics = {
            "conceptlm_v22_vq/usable_seq_len": torch.tensor(
                usable_seq_len, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v22_vq/concept_chunks": torch.tensor(
                concept_hidden.shape[0], device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v22_vq/codebook_size": torch.tensor(
                self.concept_codebook_size, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v22_vq/num_codebooks": torch.tensor(
                self.concept_num_codebooks, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v22_vq/head_dim": torch.tensor(
                self.concept_quantizer.head_dim,
                device=encoder_hidden_states.device,
                dtype=torch.float32,
            ),
            "conceptlm_v22_vq/vq_loss_raw": vq_loss.detach().float(),
            "conceptlm_v22_vq/hlm_loss_raw": hlm_loss.detach().float(),
            "conceptlm_v22_vq/code_id_mean": code_ids.detach().float().mean(),
            "conceptlm_v22_vq/code_id_max": code_ids.detach().float().max(),
        }
        metrics.update(self._v22_route_metrics(encoder_hidden_states.device))
        return (
            decoder_input,
            vq_loss,
            hlm_loss,
            metrics,
            predicted_vectors,
            repeated_concepts,
            concept_layer_states,
        )

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
    ) -> Tensor | ConceptLMV22VQOutput:
        del extra_block_kwargs
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

        self._assert_supported_v21_runtime(decoder_input, inference_context, packed_seq_params)
        encoder_hidden_states, encoder_raw_layer_states = self._run_v21_encoder(
            decoder_input,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_mask,
        )
        _v21_check_finite("v22_vq.encoder_hidden_states", encoder_hidden_states)
        (
            decoder_hidden_states,
            vq_loss,
            hlm_loss,
            concept_metrics,
            final_concept_chunk_states,
            repeated_final_concept_states,
            concept_layer_states,
        ) = self._concept_branch_v21(encoder_hidden_states, encoder_raw_layer_states)

        route_source_context = self._build_route_source_context(concept_layer_states)
        final_concept_state, dd_concept_states = self._build_dd_concept_candidates(
            final_concept_chunk_states,
            repeated_final_concept_states,
            route_source_context.concept_layer_stack,
            encoder_hidden_states.shape[0],
        )
        _v21_check_finite("v22_vq.final_concept_state", final_concept_state)
        _v21_check_finite("v22_vq.dd_concept_states", dd_concept_states)
        decoder_encoder_states = self._build_decoder_encoder_states(encoder_raw_layer_states)
        _v21_check_finite("v22_vq.decoder_encoder_states", decoder_encoder_states)
        decoder_concept_states = self._build_decoder_concept_states(
            route_source_context.concept_layer_stack
        )
        _v21_check_finite("v22_vq.decoder_concept_states", decoder_concept_states)

        hidden_states = self._run_v21_decoder_compiled(
            decoder_hidden_states,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_mask,
            dd_concept_states,
            final_concept_state,
            decoder_encoder_states,
            decoder_concept_states,
        )
        _v21_check_finite("v22_vq.decoder.final_hidden", hidden_states)

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
                extra_block_kwargs=None,
                inference_context=inference_context,
                output_processor=output_processor,
                output_processor_context=output_processor_context,
            )

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        logits = self._scale_logits(logits)
        lm_output = self.compute_language_model_loss(labels, logits)
        _maybe_print_v22_vq_ce_debug(
            hidden_states,
            logits,
            labels,
            lm_output,
            input_ids=input_ids,
        )
        return ConceptLMV22VQOutput(
            lm_loss=lm_output,
            vq_loss=vq_loss,
            hlm_loss=hlm_loss,
            concept_metrics=concept_metrics,
        )
