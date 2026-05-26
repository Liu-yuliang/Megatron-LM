# Copyright (c) 2026, ConceptLM contributors.

"""ConceptLM V2.1 prototype for Megatron-Core GPT-style training.

V2.1 keeps the existing Megatron V2 training path intact and adds an
independent model class with the newer DD/residual-flow design:

* optional encoder self-DD over encoder layer history;
* optional concept self-DD over HLM/concept layer history;
* optional decoder DD/two-route-add from decoder history and concept states;
* optional encoder->concept, encoder->decoder, and concept->decoder route-add.

This file intentionally uses a separate ``ConceptLMV21Model`` instead of
mutating ``ConceptLMV2Model`` so V2 and V2.1 can be trained side by side.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.conceptlm_v2 import (
    ConceptCausalBlock,
    ConceptLMV2Model,
    ConceptMLPBottleneck,
    ConceptRoPE,
    _init_linear_truncated_normal,
    _reset_layernorm,
    _scaled_truncated_init_stds,
    _uses_scaled_truncated_init,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


@dataclass
class ConceptLMV21Output:
    """Training output consumed by ``pretrain_conceptlm_v21.loss_func``."""

    lm_loss: Tensor
    vq_loss: Tensor
    hlm_loss: Tensor
    concept_metrics: Dict[str, Tensor]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _configure_v21_dynamo_cache() -> None:
    try:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit),
            128,
        )
    except Exception:
        pass


def _compile_v21(fn: Callable) -> Callable:
    _configure_v21_dynamo_cache()
    return torch.compile(
        fn,
        backend="inductor",
        mode="max-autotune-no-cudagraphs",
        fullgraph=True,
        dynamic=False,
    )


def _compile_v21_configurable(
    fn: Callable,
    *,
    backend: str,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
) -> Callable:
    _configure_v21_dynamo_cache()
    kwargs: Dict[str, Any] = {
        "backend": backend,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if mode.strip().lower() not in ("", "none", "default"):
        kwargs["mode"] = mode
    return torch.compile(fn, **kwargs)


def _v21_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _v21_tensor_summary(name: str, tensor: Tensor) -> str:
    detached = tensor.detach()
    flat = detached.float().reshape(-1)
    numel = flat.numel()
    if numel == 0:
        return f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, numel=0"
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    parts = [
        f"{name}: shape={tuple(detached.shape)}",
        f"dtype={detached.dtype}",
        f"finite={finite_count}/{numel}",
        f"nan={nan_count}",
        f"inf={inf_count}",
    ]
    if finite_count:
        finite_flat = flat[finite]
        parts.extend(
            [
                f"min={finite_flat.min().item():.6e}",
                f"max={finite_flat.max().item():.6e}",
                f"mean={finite_flat.mean().item():.6e}",
            ]
        )
    return ", ".join(parts)


def _v21_current_iteration_for_debug() -> Optional[int]:
    try:
        from megatron.core.rerun_state_machine import get_rerun_state_machine

        return int(get_rerun_state_machine().current_iteration) + 1
    except Exception:
        return None


def _v21_finite_debug_enabled() -> bool:
    if not _env_flag("CONCEPTLM_V21_FINITE_DEBUG", False):
        return False
    current_iter = _v21_current_iteration_for_debug()
    if current_iter is None:
        return True
    start_iter = _env_int("CONCEPTLM_V21_FINITE_DEBUG_START_ITER", -1)
    end_iter = _env_int("CONCEPTLM_V21_FINITE_DEBUG_END_ITER", -1)
    if start_iter >= 0 and current_iter < start_iter:
        return False
    if end_iter >= 0 and current_iter > end_iter:
        return False
    return True


def _v21_debug_iter_window_enabled(prefix: str) -> bool:
    current_iter = _v21_current_iteration_for_debug()
    if current_iter is None:
        return True
    start_iter = _env_int(f"{prefix}_START_ITER", -1)
    end_iter = _env_int(f"{prefix}_END_ITER", -1)
    if start_iter < 0:
        start_iter = _env_int("CONCEPTLM_V21_FINITE_DEBUG_START_ITER", -1)
    if end_iter < 0:
        end_iter = _env_int("CONCEPTLM_V21_FINITE_DEBUG_END_ITER", -1)
    if start_iter >= 0 and current_iter < start_iter:
        return False
    if end_iter >= 0 and current_iter > end_iter:
        return False
    return True


def _v21_activation_debug_enabled(tag: str) -> bool:
    if not _env_flag("CONCEPTLM_V21_ACTIVATION_DEBUG", False):
        return False
    tag_filter = os.environ.get("CONCEPTLM_V21_ACTIVATION_DEBUG_TAGS", "").strip()
    if tag_filter:
        tag_parts = [part.strip() for part in tag_filter.split(",") if part.strip()]
        if tag_parts and not any(part in tag for part in tag_parts):
            return False
    return _v21_debug_iter_window_enabled("CONCEPTLM_V21_ACTIVATION_DEBUG")


def _v21_unravel_index(flat_idx: int, shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    idx = []
    for size in reversed(tuple(shape)):
        idx.append(flat_idx % int(size))
        flat_idx //= int(size)
    return tuple(reversed(idx))


def _v21_activation_debug(tag: str, tensor: Optional[Tensor]) -> None:
    if tensor is None or not _v21_activation_debug_enabled(tag):
        return
    detached = tensor.detach()
    if detached.numel() == 0:
        return
    threshold = _env_float("CONCEPTLM_V21_ACTIVATION_DEBUG_ABS_THRESHOLD", 1024.0)
    score = torch.nan_to_num(
        detached.abs(),
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("inf"),
    )
    max_abs = score.max()
    if threshold > 0 and bool(torch.isfinite(max_abs).item()) and max_abs.item() <= threshold:
        return

    rank = _v21_rank()
    current_iter = _v21_current_iteration_for_debug()
    iter_text = "unknown" if current_iter is None else str(current_iter)
    flat_idx = int(score.reshape(-1).argmax().item())
    max_idx = _v21_unravel_index(flat_idx, detached.shape)
    max_value = detached[max_idx]
    print(
        f"[ConceptLM V2.1 activation debug][rank {rank}] "
        f"iter={iter_text} tag={tag} threshold={threshold:.6e} "
        f"max_abs={float(max_abs.float().item()):.6e} max_idx={max_idx} "
        f"max_value={float(max_value.float().item()):.6e}",
        flush=True,
    )
    print(
        f"[ConceptLM V2.1 activation debug][rank {rank}] "
        f"{_v21_tensor_summary(tag, detached)}",
        flush=True,
    )

    if detached.dim() < 2:
        return
    row_score = score.amax(dim=-1)
    large_rows = torch.nonzero(row_score > threshold, as_tuple=False)
    max_rows = max(1, _env_int("CONCEPTLM_V21_ACTIVATION_DEBUG_MAX_ROWS", 4))
    for pos in large_rows[:max_rows]:
        idx = tuple(int(x.item()) for x in pos)
        row = detached[idx]
        row_score_flat = torch.nan_to_num(
            row.detach().abs(),
            nan=float("inf"),
            posinf=float("inf"),
            neginf=float("inf"),
        ).reshape(-1)
        row_feature_idx = int(row_score_flat.argmax().item())
        row_value = row.reshape(-1)[row_feature_idx]
        print(
            f"[ConceptLM V2.1 activation debug][rank {rank}] "
            f"large_row tag={tag} idx={idx} row_max_abs="
            f"{float(row_score_flat[row_feature_idx].float().item()):.6e} "
            f"feature={row_feature_idx} value={float(row_value.float().item()):.6e} "
            f"{_v21_tensor_summary('row', row)}",
            flush=True,
        )


def _v21_check_finite(tag: str, tensor: Optional[Tensor]) -> None:
    _v21_activation_debug(tag, tensor)
    if tensor is None or not _v21_finite_debug_enabled():
        return
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    if bool(finite.all().item()):
        return

    rank = _v21_rank()
    current_iter = _v21_current_iteration_for_debug()
    iter_text = "unknown" if current_iter is None else str(current_iter)
    print(
        f"[ConceptLM V2.1 finite debug][rank {rank}] "
        f"non-finite tensor at iter={iter_text} tag={tag}",
        flush=True,
    )
    print(
        f"[ConceptLM V2.1 finite debug][rank {rank}] "
        f"{_v21_tensor_summary(tag, detached)}",
        flush=True,
    )

    bad = torch.nonzero(~finite, as_tuple=False)
    max_bad = max(1, _env_int("CONCEPTLM_V21_FINITE_DEBUG_MAX_BAD", 8))
    for pos in bad[:max_bad]:
        idx = tuple(int(x.item()) for x in pos)
        print(
            f"[ConceptLM V2.1 finite debug][rank {rank}] bad_element tag={tag} idx={idx}",
            flush=True,
        )
    if detached.dim() >= 3:
        bad_rows = torch.nonzero((~finite).any(dim=-1), as_tuple=False)
        for pos in bad_rows[:max_bad]:
            idx = tuple(int(x.item()) for x in pos)
            row = detached[idx]
            print(
                f"[ConceptLM V2.1 finite debug][rank {rank}] "
                f"bad_row tag={tag} idx={idx} {_v21_tensor_summary('row', row)}",
                flush=True,
            )

    if _env_flag("CONCEPTLM_V21_FINITE_DEBUG_FATAL", True):
        raise RuntimeError(
            f"ConceptLM V2.1 finite debug found non-finite tensor at iter={iter_text} "
            f"tag={tag}"
        )


def _v21_trace_enabled() -> bool:
    return _env_flag("CONCEPTLM_V21_TRACE_DEBUG", False) and _v21_finite_debug_enabled()


def _v21_trace_tensor(tag: str, tensor: Optional[Tensor], *, force: bool = False) -> None:
    if tensor is None or (not force and not _v21_trace_enabled()):
        return
    threshold = _env_float("CONCEPTLM_V21_TRACE_DEBUG_ABS_THRESHOLD", 64.0)
    detached = tensor.detach()
    flat = detached.float().reshape(-1)
    if flat.numel() == 0:
        return
    finite = torch.isfinite(flat)
    if bool(finite.all().item()):
        if not force and threshold > 0:
            finite_abs = flat.abs()
            if not bool((finite_abs > threshold).any().item()):
                return
        elif not force:
            return

    rank = _v21_rank()
    current_iter = _v21_current_iteration_for_debug()
    iter_text = "unknown" if current_iter is None else str(current_iter)
    print(
        f"[ConceptLM V2.1 trace debug][rank {rank}] iter={iter_text} tag={tag} "
        f"{_v21_tensor_summary(tag, detached)}",
        flush=True,
    )
    max_bad = max(1, _env_int("CONCEPTLM_V21_TRACE_DEBUG_MAX_POS", 4))
    bad_mask = ~finite
    if bool(bad_mask.any().item()):
        bad = torch.nonzero(bad_mask.reshape(detached.shape), as_tuple=False)
        for pos in bad[:max_bad]:
            idx = tuple(int(x.item()) for x in pos)
            print(
                f"[ConceptLM V2.1 trace debug][rank {rank}] bad_element "
                f"tag={tag} idx={idx}",
                flush=True,
            )
        return
    if threshold > 0:
        big_mask = flat.abs() > threshold
        if bool(big_mask.any().item()):
            shaped_mask = big_mask.reshape(detached.shape)
            big = torch.nonzero(shaped_mask, as_tuple=False)
            for pos in big[:max_bad]:
                idx = tuple(int(x.item()) for x in pos)
                value = float(detached[idx].float().item())
                print(
                    f"[ConceptLM V2.1 trace debug][rank {rank}] large_element "
                    f"tag={tag} idx={idx} value={value:.6e}",
                    flush=True,
                )


def _maybe_print_v21_ce_debug(
    hidden_states: Tensor,
    logits: Tensor,
    labels: Tensor,
    lm_output: Tensor,
    input_ids: Optional[Tensor] = None,
) -> None:
    if not _env_flag("CONCEPTLM_V21_CE_DEBUG", False):
        return
    force_every_n = _env_int("CONCEPTLM_V21_CE_DEBUG_EVERY_N", 0)
    force_scan_logits = force_every_n > 0
    if force_every_n > 0:
        if not hasattr(_maybe_print_v21_ce_debug, "_step"):
            _maybe_print_v21_ce_debug._step = 0
        _maybe_print_v21_ce_debug._step += 1
        force_scan_logits = (_maybe_print_v21_ce_debug._step % force_every_n) == 0

    lm_output_detached = lm_output.detach()
    lm_is_finite = torch.isfinite(lm_output_detached)
    if bool(lm_is_finite.all().item()) and not force_scan_logits:
        return

    rank = _v21_rank()
    current_iter = _v21_current_iteration_for_debug()
    iter_text = "unknown" if current_iter is None else str(current_iter)
    print(
        f"[ConceptLM V2.1 CE debug][rank {rank}] "
        f"triggered: iter={iter_text}, "
        f"lm_output_finite={bool(lm_is_finite.all().item())}, "
        f"force_scan_logits={force_scan_logits}",
        flush=True,
    )
    print(f"[ConceptLM V2.1 CE debug][rank {rank}] {_v21_tensor_summary('lm_output', lm_output)}", flush=True)
    print(f"[ConceptLM V2.1 CE debug][rank {rank}] {_v21_tensor_summary('labels', labels)}", flush=True)
    print(
        f"[ConceptLM V2.1 CE debug][rank {rank}] labels_range: "
        f"min={int(labels.detach().min().item())}, max={int(labels.detach().max().item())}, "
        f"vocab={int(logits.shape[-1])}",
        flush=True,
    )

    # The following scans are intentionally gated because logits are very large.
    print(
        f"[ConceptLM V2.1 CE debug][rank {rank}] {_v21_tensor_summary('hidden_states', hidden_states)}",
        flush=True,
    )
    print(f"[ConceptLM V2.1 CE debug][rank {rank}] {_v21_tensor_summary('logits', logits)}", flush=True)

    bad_positions = torch.nonzero(~lm_is_finite, as_tuple=False)
    if bad_positions.numel() == 0:
        return
    max_bad = max(1, _env_int("CONCEPTLM_V21_CE_DEBUG_MAX_BAD_TOKENS", 8))
    for pos in bad_positions[:max_bad]:
        b = int(pos[0].item())
        s = int(pos[1].item())
        label = int(labels.detach()[b, s].item())
        vocab_size = int(logits.shape[-1])
        label_is_valid = 0 <= label < vocab_size
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
        if label_is_valid:
            label_logit_text = f"{float(logit_row[label].float().item()):.6e}"
        else:
            label_logit_text = "invalid-label"
        finite_logits = torch.isfinite(logit_row)
        top_logit_text = "no-finite-logits"
        if bool(finite_logits.any().item()):
            finite_row = logit_row.float().masked_fill(~finite_logits, float("-inf"))
            k = min(5, finite_row.numel())
            top_values, top_indices = torch.topk(finite_row, k=k)
            top_logit_text = ", ".join(
                f"{int(idx.item())}:{float(value.item()):.6e}"
                for value, idx in zip(top_values, top_indices)
            )
        print(
            f"[ConceptLM V2.1 CE debug][rank {rank}] bad_token b={b}, s={s}, "
            f"label={label}, label_is_valid={label_is_valid}, "
            f"label_logit={label_logit_text}, "
            f"loss={float(lm_output_detached[b, s].float().item())}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.1 CE debug][rank {rank}] bad_token_context "
            f"b={b}, s={s}, range=[{context_start},{context_end}), "
            f"input_ids={input_context}, labels={label_context}, lm_output={loss_context}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.1 CE debug][rank {rank}] "
            f"{_v21_tensor_summary(f'hidden_states[s={s},b={b}]', hidden_row)}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.1 CE debug][rank {rank}] "
            f"{_v21_tensor_summary(f'logits[s={s},b={b}]', logit_row)}",
            flush=True,
        )
        print(
            f"[ConceptLM V2.1 CE debug][rank {rank}] top_finite_logits "
            f"b={b}, s={s}: {top_logit_text}",
            flush=True,
        )


_V21_DEPTH_DD_COMPILED_CACHE: Dict[tuple[int, bool], Callable] = {}
_V21_DECODER_DD_FINAL_CONCEPT_COMPILED_CACHE: Dict[tuple[int, bool], Callable] = {}


def _get_compiled_v21_depth_dd(num_prev: int, use_softmax: bool) -> Callable:
    num_prev = int(num_prev)
    use_softmax = bool(use_softmax)
    cache_key = (num_prev, use_softmax)
    compiled = _V21_DEPTH_DD_COMPILED_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    # Dynamo's recompile accounting is keyed by Python code object. Generate one
    # code object per fixed history width so 24 DD layers do not fight one cache.
    namespace = {"F": F, "torch": torch}
    softmax_line = "    weights = weights.softmax(dim=-1)\n" if use_softmax else ""
    suffix = "softmax" if use_softmax else "linear"
    exec(
        f"""
def _v21_depth_dd_{num_prev}_{suffix}(
    history_states,
    current_hidden,
    w1_weight,
    w2_weight,
    static_a,
):
    weights = F.linear(F.gelu(F.linear(current_hidden, w1_weight)), w2_weight)
    weights = weights + static_a.view(1, 1, {num_prev})
{softmax_line.rstrip()}
    return torch.einsum("sbm,sbmh->sbh", weights, history_states)
        """,
        namespace,
    )
    compiled = _compile_v21(namespace[f"_v21_depth_dd_{num_prev}_{suffix}"])
    _V21_DEPTH_DD_COMPILED_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_v21_decoder_dd_final_concept(num_prev: int, use_softmax: bool) -> Callable:
    num_prev = int(num_prev)
    use_softmax = bool(use_softmax)
    cache_key = (num_prev, use_softmax)
    compiled = _V21_DECODER_DD_FINAL_CONCEPT_COMPILED_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    namespace = {"F": F, "torch": torch}
    softmax_line = "    weights = weights.softmax(dim=-1)\n" if use_softmax else ""
    suffix = "softmax" if use_softmax else "linear"
    exec(
        f"""
def _v21_decoder_dd_final_concept_{num_prev}_{suffix}(
    history_states,
    current_hidden,
    w1_weight,
    w2_weight,
    static_a,
    final_concept_state,
    final_proj_weight,
):
    weights = F.linear(F.gelu(F.linear(current_hidden, w1_weight)), w2_weight)
    weights = weights + static_a.view(1, 1, {num_prev})
{softmax_line.rstrip()}
    hidden = torch.einsum("sbm,sbmh->sbh", weights, history_states)
    residual_update = F.linear(final_concept_state, final_proj_weight.to(dtype=final_concept_state.dtype))
    return hidden + residual_update.to(dtype=hidden.dtype)
	""",
	        namespace,
	    )
    compiled = _compile_v21(namespace[f"_v21_decoder_dd_final_concept_{num_prev}_{suffix}"])
    _V21_DECODER_DD_FINAL_CONCEPT_COMPILED_CACHE[cache_key] = compiled
    return compiled


class V21DepthDD(nn.Module):
    """Single-stream DD router over same-sequence layer history.

    Shapes are Megatron seq-major:
      current_hidden: [seq, batch, hidden]
      history_states: [seq, batch, num_states, hidden]
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        layer_idx: int,
        route_hidden_size: int = 0,
        use_layernorm: bool = False,
        use_softmax: bool = False,
        compile_dd: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_prev = self.layer_idx + 2
        hidden = int(route_hidden_size) or self.num_prev
        self.use_softmax = bool(use_softmax)
        self.compile_dd = bool(compile_dd)
        self.history_norm = nn.LayerNorm(hidden_size, eps=eps) if use_layernorm else nn.Identity()
        self.w1 = nn.Linear(hidden_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, self.num_prev, bias=False)
        self.static_a = nn.Parameter(torch.zeros(self.num_prev))
        self.act = nn.GELU()
        self._compiled_forward = (
            _get_compiled_v21_depth_dd(self.num_prev, self.use_softmax)
            if self.compile_dd
            else None
        )
        self.reset_parameters(hidden_size)

    def reset_parameters(self, hidden_size: int) -> None:
        nn.init.normal_(self.w1.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        nn.init.zeros_(self.w2.weight)
        with torch.no_grad():
            self.static_a.zero_()
            self.static_a[-1] = 1.0

    def forward(self, history_states: Tensor, current_hidden: Tensor) -> Tensor:
        active_prev = history_states.shape[2]
        normed_history = self.history_norm(history_states)
        if self.compile_dd:
            if active_prev != self.num_prev:
                raise RuntimeError(
                    f"compiled V21DepthDD expects {self.num_prev} history states, "
                    f"got {active_prev}"
                )
            return self._compiled_forward(
                normed_history,
                current_hidden,
                self.w1.weight,
                self.w2.weight,
                self.static_a,
            )
        weights = self.w2(self.act(self.w1(current_hidden)))
        weights = weights[:, :, -active_prev:]
        weights = weights + self.static_a[-active_prev:].view(1, 1, active_prev)
        if self.use_softmax:
            weights = weights.softmax(dim=-1)
        _v21_trace_tensor(f"depth_dd.layer{self.layer_idx}.weights", weights)
        _v21_trace_tensor(f"depth_dd.layer{self.layer_idx}.history", normed_history)
        return (weights.unsqueeze(-1) * normed_history).sum(dim=2)


class V21SelfDD(nn.Module):
    """Depth DD over encoder/concept layer history."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        num_layers: int,
        every_n_layers: int = 1,
        route_hidden_size: int = 0,
        use_layernorm: bool = False,
        use_softmax: bool = False,
        compile_dd: bool = False,
    ) -> None:
        super().__init__()
        self.every_n_layers = max(1, int(every_n_layers))
        self.depth_dds = nn.ModuleList(
            [
                V21DepthDD(
                    hidden_size=hidden_size,
                    eps=eps,
                    layer_idx=layer_idx,
                    route_hidden_size=route_hidden_size,
                    use_layernorm=use_layernorm,
                    use_softmax=use_softmax,
                    compile_dd=compile_dd,
                )
                for layer_idx in range(int(num_layers))
            ]
        )

    def forward(self, layer_idx: int, current_hidden: Tensor, history_states: Tensor) -> Tensor:
        if (layer_idx + 1) % self.every_n_layers != 0:
            return current_hidden
        return self.depth_dds[layer_idx](history_states, current_hidden)


@_compile_v21
def _compiled_v21_residual_route_add(
    target_hidden: Tensor,
    source_states: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
    residual_weight: Tensor,
    use_softmax: bool,
) -> Tensor:
    active_sources = source_states.shape[2]
    route_logits = F.linear(F.gelu(F.linear(target_hidden, w1_weight)), w2_weight)
    route_logits = route_logits[:, :, -active_sources:]
    weights = route_logits.softmax(dim=-1) if use_softmax else route_logits
    source_mix = torch.einsum("sbm,sbmh->sbh", weights, source_states)
    residual_update = F.linear(source_mix, residual_weight.to(dtype=source_mix.dtype))
    return target_hidden + residual_update.to(dtype=target_hidden.dtype)


@_compile_v21
def _compiled_v21_residual_route_add_repeated_chunks(
    target_hidden: Tensor,
    chunk_source_states: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
    residual_weight: Tensor,
    chunk_size: int,
    shift_feature: bool,
    use_softmax: bool,
) -> Tensor:
    token_len, batch_size, hidden_size = target_hidden.shape
    num_chunks = chunk_source_states.shape[0]
    active_sources = chunk_source_states.shape[2]
    route_logits = F.linear(F.gelu(F.linear(target_hidden, w1_weight)), w2_weight)
    route_logits = route_logits[:, :, -active_sources:]
    weights = route_logits.softmax(dim=-1) if use_softmax else route_logits

    left_pad = 1 if shift_feature else 0
    if left_pad:
        weights = torch.cat(
            [weights.new_zeros(left_pad, batch_size, active_sources), weights],
            dim=0,
        )

    repeated_len = num_chunks * chunk_size
    right_pad = repeated_len - weights.shape[0]
    if right_pad > 0:
        weights = torch.cat(
            [weights, weights.new_zeros(right_pad, batch_size, active_sources)],
            dim=0,
        )
    elif right_pad < 0:
        weights = weights[:repeated_len]

    chunk_weights = weights.reshape(num_chunks, chunk_size, batch_size, active_sources)
    source_mix = torch.einsum("ckbm,cbmh->ckbh", chunk_weights, chunk_source_states)
    source_mix = source_mix.reshape(repeated_len, batch_size, hidden_size)
    if left_pad:
        source_mix = source_mix[left_pad : left_pad + token_len]
    else:
        source_mix = source_mix[:token_len]
    if source_mix.shape[0] < token_len:
        pad = source_mix.new_zeros(token_len - source_mix.shape[0], batch_size, hidden_size)
        source_mix = torch.cat((source_mix, pad), dim=0)
    residual_update = F.linear(source_mix, residual_weight.to(dtype=source_mix.dtype))
    return target_hidden + residual_update.to(dtype=target_hidden.dtype)


class V21ResidualFlowRouteAdd(nn.Module):
    """DD-style additive route over cross-module source states."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        num_source_states: int,
        beta_init: float = 0.02,
        route_hidden_size: int = 0,
        use_softmax: bool = True,
        source_use_layernorm: bool = True,
        compile_route: bool = False,
    ) -> None:
        super().__init__()
        self.num_source_states = max(0, int(num_source_states))
        self.hidden_size = int(hidden_size)
        hidden = int(route_hidden_size) or max(1, self.num_source_states)
        self.use_softmax = bool(use_softmax)
        self.compile_route = bool(compile_route)
        self.source_norm = (
            nn.LayerNorm(self.hidden_size, eps=eps) if source_use_layernorm else nn.Identity()
        )
        self.w1 = nn.Linear(self.hidden_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, max(1, self.num_source_states), bias=False)
        self.residual_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act = nn.GELU()
        self.reset_parameters(beta_init)

    def reset_parameters(self, beta_init: float = 0.02) -> None:
        nn.init.normal_(self.w1.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
        nn.init.zeros_(self.w2.weight)
        nn.init.eye_(self.residual_proj.weight)
        with torch.no_grad():
            self.residual_proj.weight.mul_(float(beta_init))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        old_beta_key = prefix + "beta"
        new_weight_key = prefix + "residual_proj.weight"
        if old_beta_key in state_dict and new_weight_key not in state_dict:
            beta = state_dict.pop(old_beta_key)
            weight = torch.zeros(
                self.hidden_size,
                self.hidden_size,
                dtype=beta.dtype,
                device=beta.device,
            )
            beta_value = beta.detach().to(dtype=weight.dtype, device=weight.device).view(-1)[0]
            weight.diagonal().copy_(beta_value.expand(self.hidden_size))
            state_dict[new_weight_key] = weight
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, target_hidden: Tensor, source_states: Optional[Tensor], source_is_normed: bool = False) -> Tensor:
        if source_states is None or source_states.shape[2] == 0:
            return target_hidden
        active_sources = source_states.shape[2]
        normed_sources = source_states if source_is_normed else self.source_norm(source_states)
        if self.compile_route:
            return _compiled_v21_residual_route_add(
                target_hidden,
                normed_sources,
                self.w1.weight,
                self.w2.weight,
                self.residual_proj.weight,
                self.use_softmax,
            )
        route_logits = self.w2(self.act(self.w1(target_hidden)))
        route_logits = route_logits[:, :, -active_sources:]
        _v21_trace_tensor("residual_route.route_logits", route_logits)
        weights = route_logits.softmax(dim=-1) if self.use_softmax else route_logits
        _v21_trace_tensor("residual_route.weights", weights)
        _v21_trace_tensor("residual_route.sources", normed_sources)
        source_mix = torch.einsum("sbm,sbmh->sbh", weights, normed_sources)
        _v21_trace_tensor("residual_route.source_mix", source_mix)
        residual_update = F.linear(
            source_mix,
            self.residual_proj.weight.to(dtype=source_mix.dtype),
        )
        return target_hidden + residual_update.to(dtype=target_hidden.dtype)

    def forward_repeated_chunks(
        self,
        target_hidden: Tensor,
        chunk_source_states: Optional[Tensor],
        chunk_size: int,
        shift_feature: bool = False,
        source_is_normed: bool = False,
    ) -> Tensor:
        if chunk_source_states is None or chunk_source_states.shape[2] == 0:
            return target_hidden

        token_len, batch_size, hidden_size = target_hidden.shape
        num_chunks = chunk_source_states.shape[0]
        active_sources = chunk_source_states.shape[2]
        chunk_size = int(chunk_size)
        normed_sources = (
            chunk_source_states
            if source_is_normed
            else self.source_norm(chunk_source_states)
        )
        if self.compile_route:
            return _compiled_v21_residual_route_add_repeated_chunks(
                target_hidden,
                normed_sources,
                self.w1.weight,
                self.w2.weight,
                self.residual_proj.weight,
                chunk_size,
                shift_feature,
                self.use_softmax,
            )
        route_logits = self.w2(self.act(self.w1(target_hidden)))
        route_logits = route_logits[:, :, -active_sources:]
        _v21_trace_tensor("residual_route.repeated.route_logits", route_logits)
        weights = route_logits.softmax(dim=-1) if self.use_softmax else route_logits
        _v21_trace_tensor("residual_route.repeated.weights", weights)
        _v21_trace_tensor("residual_route.repeated.sources", normed_sources)

        left_pad = 1 if shift_feature else 0
        if left_pad:
            weights = torch.cat(
                [weights.new_zeros(left_pad, batch_size, active_sources), weights],
                dim=0,
            )

        repeated_len = num_chunks * chunk_size
        right_pad = repeated_len - weights.shape[0]
        if right_pad > 0:
            weights = torch.cat(
                [weights, weights.new_zeros(right_pad, batch_size, active_sources)],
                dim=0,
            )
        elif right_pad < 0:
            weights = weights[:repeated_len]

        chunk_weights = weights.reshape(num_chunks, chunk_size, batch_size, active_sources)
        source_mix = torch.einsum("ckbm,cbmh->ckbh", chunk_weights, normed_sources)
        source_mix = source_mix.reshape(repeated_len, batch_size, hidden_size)
        if left_pad:
            source_mix = source_mix[left_pad : left_pad + token_len]
        else:
            source_mix = source_mix[:token_len]
        if source_mix.shape[0] < token_len:
            pad = source_mix.new_zeros(token_len - source_mix.shape[0], batch_size, hidden_size)
            source_mix = torch.cat((source_mix, pad), dim=0)
        _v21_trace_tensor("residual_route.repeated.source_mix", source_mix)
        residual_update = F.linear(
            source_mix,
            self.residual_proj.weight.to(dtype=source_mix.dtype),
        )
        return target_hidden + residual_update.to(dtype=target_hidden.dtype)


class V21ConceptRouteAdd(nn.Module):
    """Separate final-concept and raw-concept projection adds after decoder DD."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        num_concept_states: int,
        beta_init: float,
        route_hidden_size: int,
        use_softmax: bool,
        enable_raw_concept_route: bool,
        enable_final_concept_route: bool,
        concept_use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_concept_states = max(0, int(num_concept_states))
        self.enable_raw_concept_route = bool(enable_raw_concept_route) and self.num_concept_states > 0
        self.enable_final_concept_route = bool(enable_final_concept_route)
        self.use_dynamic_route = self.enable_raw_concept_route and self.num_concept_states > 1
        hidden = int(route_hidden_size) or max(1, self.num_concept_states)
        self.concept_norm = nn.LayerNorm(self.hidden_size, eps=eps) if concept_use_layernorm else nn.Identity()
        if self.use_dynamic_route:
            self.w1 = nn.Linear(self.hidden_size, hidden, bias=False)
            self.w2 = nn.Linear(hidden, self.num_concept_states, bias=False)
        else:
            self.w1 = None
            self.w2 = None
        self.raw_proj = (
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            if self.enable_raw_concept_route
            else None
        )
        self.final_proj = (
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            if self.enable_final_concept_route
            else None
        )
        self.use_softmax = bool(use_softmax)
        self.act = nn.GELU()
        self.reset_parameters(beta_init)

    def reset_parameters(self, beta_init: float) -> None:
        if self.use_dynamic_route:
            nn.init.normal_(self.w1.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
            nn.init.zeros_(self.w2.weight)
        for proj in (self.raw_proj, self.final_proj):
            if proj is None:
                continue
            nn.init.eye_(proj.weight)
            with torch.no_grad():
                proj.weight.mul_(float(beta_init))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        def convert_old_beta(old_name: str, proj: Optional[nn.Linear]) -> None:
            old_key = prefix + old_name
            if old_key not in state_dict:
                return
            new_key = prefix + old_name.replace("_beta", "_proj.weight")
            if proj is None or new_key in state_dict:
                state_dict.pop(old_key)
                return
            beta = state_dict.pop(old_key)
            weight = torch.zeros(
                self.hidden_size,
                self.hidden_size,
                dtype=beta.dtype,
                device=beta.device,
            )
            beta_value = beta.detach().to(dtype=weight.dtype, device=weight.device).view(-1)[0]
            weight.diagonal().copy_(beta_value.expand(self.hidden_size))
            state_dict[new_key] = weight

        convert_old_beta("raw_beta", self.raw_proj)
        convert_old_beta("final_beta", self.final_proj)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        decoder_hidden: Tensor,
        raw_concept_states: Optional[Tensor],
        final_concept_state: Optional[Tensor],
    ) -> Tensor:
        hidden = decoder_hidden
        if self.final_proj is not None and final_concept_state is not None:
            final_concept = self.concept_norm(final_concept_state.unsqueeze(2))[:, :, 0, :]
            _v21_trace_tensor("concept_route.final_concept", final_concept)
            final_update = F.linear(
                final_concept,
                self.final_proj.weight.to(dtype=final_concept.dtype),
            )
            hidden = hidden + final_update.to(dtype=hidden.dtype)

        if self.raw_proj is None or raw_concept_states is None:
            return hidden

        active_concepts = raw_concept_states.shape[2]
        normed_concepts = self.concept_norm(raw_concept_states)
        _v21_trace_tensor("concept_route.raw_concepts", normed_concepts)
        if active_concepts == 1:
            concept_mix = normed_concepts[:, :, 0, :]
        else:
            route_logits = self.w2(self.act(self.w1(decoder_hidden)))
            route_logits = route_logits[:, :, -active_concepts:]
            _v21_trace_tensor("concept_route.route_logits", route_logits)
            weights = route_logits.softmax(dim=-1) if self.use_softmax else route_logits
            _v21_trace_tensor("concept_route.weights", weights)
            concept_mix = torch.einsum("sbc,sbch->sbh", weights, normed_concepts)
        _v21_trace_tensor("concept_route.concept_mix", concept_mix)
        raw_update = F.linear(
            concept_mix,
            self.raw_proj.weight.to(dtype=concept_mix.dtype),
        )
        return hidden + raw_update.to(dtype=hidden.dtype)


class V21DDTwoRouteAdd(nn.Module):
    """Per-layer decoder DD plus concept route add."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        num_layers: int,
        num_concept_states: int,
        every_n_layers: int = 1,
        concept_route_first_n: int = -1,
        decoder_route_hidden_size: int = 0,
        concept_route_hidden_size: int = 0,
        beta_init: float = 0.3,
        use_softmax: bool = True,
        disable_decoder_dd: bool = False,
        decoder_use_layernorm: bool = False,
        decoder_use_softmax: bool = False,
        concept_use_layernorm: bool = True,
        enable_raw_concept_route: bool = True,
        enable_final_concept_route: bool = True,
        compile_routes: bool = False,
    ) -> None:
        super().__init__()
        self.every_n_layers = max(1, int(every_n_layers))
        self.concept_route_first_n = int(concept_route_first_n)
        self.disable_decoder_dd = bool(disable_decoder_dd)
        self.decoder_dds = nn.ModuleList(
            [
                V21DepthDD(
                    hidden_size=hidden_size,
                    eps=eps,
                    layer_idx=layer_idx,
                    route_hidden_size=decoder_route_hidden_size,
                    use_layernorm=decoder_use_layernorm,
                    use_softmax=decoder_use_softmax,
                    compile_dd=compile_routes,
                )
                for layer_idx in range(int(num_layers))
            ]
        )
        self.concept_routes = nn.ModuleList(
            [
                V21ConceptRouteAdd(
                    hidden_size=hidden_size,
                    eps=eps,
                    num_concept_states=num_concept_states,
                    beta_init=beta_init,
                    route_hidden_size=concept_route_hidden_size,
                    use_softmax=use_softmax,
                    enable_raw_concept_route=enable_raw_concept_route,
                    enable_final_concept_route=enable_final_concept_route,
                    concept_use_layernorm=concept_use_layernorm,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        layer_idx: int,
        current_hidden: Tensor,
        history_states: Tensor,
        concept_states: Optional[Tensor],
        final_concept_state: Optional[Tensor] = None,
    ) -> Tensor:
        hidden = current_hidden
        route = self.concept_routes[layer_idx]
        apply_dd = (
            (not self.disable_decoder_dd)
            and ((layer_idx + 1) % self.every_n_layers == 0)
        )
        apply_concept_route = (
            (concept_states is not None or final_concept_state is not None)
            and (self.concept_route_first_n < 0 or layer_idx < self.concept_route_first_n)
        )
        can_compile_final_route = (
            route.raw_proj is None
            and final_concept_state is not None
            and route.final_proj is not None
            and isinstance(route.concept_norm, nn.Identity)
        )
        if (
            apply_dd
            and apply_concept_route
            and can_compile_final_route
        ):
            depth_dd = self.decoder_dds[layer_idx]
            return _get_compiled_v21_decoder_dd_final_concept(
                depth_dd.num_prev, depth_dd.use_softmax
            )(
                depth_dd.history_norm(history_states),
                current_hidden,
                depth_dd.w1.weight,
                depth_dd.w2.weight,
                depth_dd.static_a,
                final_concept_state,
                route.final_proj.weight,
            )
        if apply_dd:
            hidden = self.decoder_dds[layer_idx](history_states, current_hidden)
            _v21_check_finite(f"dd_two_route.layer{layer_idx}.decoder_dd", hidden)
        elif self.disable_decoder_dd and self.training:
            dummy = current_hidden.new_zeros(())
            for parameter in self.decoder_dds[layer_idx].parameters():
                dummy = dummy + parameter.float().sum() * 0.0
            hidden = hidden + dummy.to(dtype=hidden.dtype)
        if apply_concept_route:
            hidden = route(hidden, concept_states, final_concept_state)
            _v21_check_finite(f"dd_two_route.layer{layer_idx}.concept_route", hidden)
        return hidden


class V21ConceptCandidateBuilder(nn.Module):
    """Build shifted token-length concept candidates in seq-major layout."""

    def __init__(self, chunk_size: int, shift_feature: bool, concept_source: str) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.shift_feature = bool(shift_feature)
        self.concept_source = str(concept_source).strip().lower()
        if self.concept_source not in {"final", "hlm_layers", "hlm_layers_plus_final"}:
            raise ValueError(
                "concept_dd_two_route_add_concept_source must be one of "
                "'final', 'hlm_layers', or 'hlm_layers_plus_final'"
            )

    def _repeat_shift_one(self, chunk_states: Tensor, seq_len: int) -> Tensor:
        shifted = torch.cat((torch.zeros_like(chunk_states[:1]), chunk_states), dim=0)
        repeated = shifted.repeat_interleave(self.chunk_size, dim=0)[: seq_len + 1]
        if self.shift_feature:
            repeated = repeated[1:]
        else:
            repeated = repeated[:seq_len]
        if repeated.shape[0] < seq_len:
            pad = repeated.new_zeros(seq_len - repeated.shape[0], *repeated.shape[1:])
            repeated = torch.cat((repeated, pad), dim=0)
        return repeated[:seq_len]

    def _repeat_shift_many(self, chunk_states: Tensor, seq_len: int) -> Tensor:
        num_chunks, batch_size, num_concepts, hidden_size = chunk_states.shape
        flat_states = chunk_states.permute(0, 2, 1, 3).reshape(
            num_chunks, batch_size * num_concepts, hidden_size
        )
        flat_repeated = self._repeat_shift_one(flat_states, seq_len)
        return flat_repeated.reshape(seq_len, num_concepts, batch_size, hidden_size).permute(
            0, 2, 1, 3
        ).contiguous()

    def forward(
        self,
        final_concept_states: Tensor,
        layer_concept_states: Optional[Tensor],
        seq_len: int,
    ) -> tuple[Tensor, Optional[Tensor]]:
        final_repeated = self._repeat_shift_one(final_concept_states, seq_len)
        if self.concept_source == "final":
            return final_repeated, None
        if layer_concept_states is None:
            raise ValueError(f"{self.concept_source} requires HLM layer concept states.")
        layer_repeated = self._repeat_shift_many(layer_concept_states, seq_len)
        return final_repeated, layer_repeated


class ConceptPredictorV21(nn.Module):
    """Concept/HLM tower with optional self-DD and encoder-read routes."""

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
        use_concept_self_dd: bool = False,
        concept_self_dd_every_n_layers: int = 1,
        concept_self_dd_hidden_size: int = 0,
        concept_self_dd_use_layernorm: bool = False,
        concept_self_dd_compile: bool = False,
        enable_concept_read_encoder: bool = False,
        num_concept_read_encoder_sources: int = 0,
        residual_flow_beta_init: float = 0.02,
        residual_flow_route_hidden_size: int = 0,
        residual_flow_route_use_softmax: bool = True,
        residual_flow_source_use_layernorm: bool = True,
        residual_flow_shared_source_norm: bool = False,
        residual_flow_compile_routes: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("ConceptLM V2.1 special_layers must be positive")
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
        self.concept_self_dd = (
            V21SelfDD(
                hidden_size=hidden_size,
                eps=eps,
                num_layers=num_layers,
                every_n_layers=concept_self_dd_every_n_layers,
                route_hidden_size=concept_self_dd_hidden_size,
                use_layernorm=concept_self_dd_use_layernorm,
                compile_dd=concept_self_dd_compile,
            )
            if use_concept_self_dd
            else None
        )
        self.concept_read_encoder_routes = (
            nn.ModuleList(
                [
                    V21ResidualFlowRouteAdd(
                        hidden_size=hidden_size,
                        eps=eps,
                        num_source_states=num_concept_read_encoder_sources,
                        beta_init=residual_flow_beta_init,
                        route_hidden_size=residual_flow_route_hidden_size,
                        use_softmax=residual_flow_route_use_softmax,
                        source_use_layernorm=(
                            False if residual_flow_shared_source_norm else residual_flow_source_use_layernorm
                        ),
                        compile_route=residual_flow_compile_routes,
                    )
                    for _ in range(num_layers)
                ]
            )
            if enable_concept_read_encoder and num_concept_read_encoder_sources > 0
            else None
        )
        self.concept_read_encoder_shared_source_norm = (
            nn.LayerNorm(hidden_size, eps=eps)
            if self.concept_read_encoder_routes is not None
            and residual_flow_shared_source_norm
            and residual_flow_source_use_layernorm
            else None
        )

    def reset_scaled_truncated_parameters(self, hidden_size: int, total_layers: int) -> None:
        for layer in self.layers:
            layer.reset_scaled_truncated_parameters(total_layers)
        _reset_layernorm(self.final_layernorm)
        pred_std, _ = _scaled_truncated_init_stds(hidden_size, total_layers)
        for head in self.prediction_heads:
            _init_linear_truncated_normal(head, pred_std)
        if self.concept_read_encoder_shared_source_norm is not None:
            _reset_layernorm(self.concept_read_encoder_shared_source_norm)

    def forward(
        self,
        concept_hidden: Tensor,
        encoder_concept_states: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, ...]]]:
        hidden_states = concept_hidden
        concept_history = []
        dd_history = [hidden_states] if self.concept_self_dd is not None else None
        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(hidden_states)
            _v21_check_finite(f"concept.layer{layer_idx}.raw", layer_out)
            raw_out = layer_out
            if return_layer_states:
                concept_history.append(raw_out)
            if self.concept_self_dd is not None:
                dd_history.append(raw_out)
                layer_out = self.concept_self_dd(
                    layer_idx,
                    layer_out,
                    torch.stack(dd_history, dim=2),
                )
                _v21_check_finite(f"concept.layer{layer_idx}.self_dd", layer_out)
            if self.concept_read_encoder_routes is not None and encoder_concept_states is not None:
                layer_out = self.concept_read_encoder_routes[layer_idx](
                    layer_out,
                    encoder_concept_states,
                    self.concept_read_encoder_shared_source_norm is not None,
                )
                _v21_check_finite(f"concept.layer{layer_idx}.read_encoder", layer_out)
            hidden_states = layer_out
        hidden_states = self.final_layernorm(hidden_states)
        _v21_check_finite("concept.final_layernorm", hidden_states)
        predicted_latents = torch.stack(
            [head(hidden_states) for head in self.prediction_heads],
            dim=2,
        )
        _v21_check_finite("concept.prediction_heads", predicted_latents)
        return predicted_latents, tuple(concept_history) if return_layer_states else None


class ConceptLMV21Model(ConceptLMV2Model):
    """V2.1 model class with independent DD/residual-flow switches."""

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
        concept_mlp_recon_loss_weight: float = 0.0,
        concept_mlp_hlm_loss_weight: float = 0.0,
        concept_mlp_hidden_loss_weight: float = 1.0,
        concept_mlp_hlm_loss_type: Literal["mse", "cosine"] = "mse",
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
            concept_bottleneck_type=concept_bottleneck_type,
            concept_vq_patch_ratio=concept_vq_patch_ratio,
            concept_mlp_bottleneck_ratio=concept_mlp_bottleneck_ratio,
            concept_mlp_bottleneck_activation=concept_mlp_bottleneck_activation,
            concept_mlp_recon_loss_weight=concept_mlp_recon_loss_weight,
            concept_mlp_hlm_loss_weight=concept_mlp_hlm_loss_weight,
            concept_mlp_hidden_loss_weight=concept_mlp_hidden_loss_weight,
            concept_mlp_hlm_loss_type=concept_mlp_hlm_loss_type,
            concept_layer_norm_option=concept_layer_norm_option,
            concept_fusion_norm_alpha_init=concept_fusion_norm_alpha_init,
            concept_fusion_alpha_init=concept_fusion_alpha_init,
            concept_hlm_ffn_hidden_size=concept_hlm_ffn_hidden_size,
            **kwargs,
        )

        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon
        self.concept_dd_two_route_add = bool(concept_dd_two_route_add)
        self.concept_dd_two_route_add_concept_source = str(
            concept_dd_two_route_add_concept_source
        ).strip().lower()
        self.concept_enable_full_residual_flow = bool(concept_enable_full_residual_flow)
        self.concept_enable_concept_read_encoder = (
            self.concept_enable_full_residual_flow or bool(concept_enable_concept_read_encoder)
        )
        self.concept_enable_decoder_read_encoder = (
            self.concept_enable_full_residual_flow or bool(concept_enable_decoder_read_encoder)
        )
        self.concept_enable_decoder_read_concept = (
            self.concept_enable_full_residual_flow or bool(concept_enable_decoder_read_concept)
        )
        self.concept_residual_flow_shared_source_norm = bool(
            concept_residual_flow_shared_source_norm
        )
        self.concept_residual_flow_source_use_layernorm = bool(
            concept_residual_flow_source_use_layernorm
        )

        self.dd_encoder_self_dd = (
            V21SelfDD(
                hidden_size=hidden_size,
                eps=eps,
                num_layers=self.concept_encoder_layers,
                every_n_layers=concept_dd_encoder_self_dd_every_n_layers,
                route_hidden_size=concept_dd_encoder_self_dd_hidden_size,
                use_layernorm=concept_dd_encoder_self_dd_use_layernorm,
                compile_dd=concept_compile_dd_routes,
            )
            if concept_dd_encoder_self_dd
            else None
        )

        self.concept_predictor = ConceptPredictorV21(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            num_layers=self.concept_special_layers,
            num_splits=self.concept_num_splits,
            latent_dim=self.concept_latent_dim,
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
        if _uses_scaled_truncated_init(self.config):
            self.concept_predictor.reset_scaled_truncated_parameters(
                hidden_size,
                self.concept_encoder_layers
                + self.concept_decoder_layers
                + self.concept_special_layers,
            )

        dd_concept_counts = {
            "final": 0,
            "hlm_layers": self.concept_special_layers,
            "hlm_layers_plus_final": self.concept_special_layers,
        }
        if self.concept_dd_two_route_add_concept_source not in dd_concept_counts:
            raise ValueError(
                "concept_dd_two_route_add_concept_source must be one of "
                "'final', 'hlm_layers', or 'hlm_layers_plus_final'."
            )
        if self.concept_dd_two_route_add:
            self.dd_concept_builder = V21ConceptCandidateBuilder(
                chunk_size=self.concept_chunk_size,
                shift_feature=self.concept_shift_feature,
                concept_source=self.concept_dd_two_route_add_concept_source,
            )
            self.dd_two_route_add = V21DDTwoRouteAdd(
                hidden_size=hidden_size,
                eps=eps,
                num_layers=self.concept_decoder_layers,
                num_concept_states=dd_concept_counts[
                    self.concept_dd_two_route_add_concept_source
                ],
                every_n_layers=concept_dd_two_route_add_every_n_layers,
                concept_route_first_n=concept_dd_two_route_add_concept_route_first_n,
                decoder_route_hidden_size=concept_dd_two_route_add_decoder_hidden_size,
                concept_route_hidden_size=concept_dd_two_route_add_concept_hidden_size,
                beta_init=concept_dd_two_route_add_beta_init,
                use_softmax=concept_dd_two_route_add_use_softmax,
                disable_decoder_dd=concept_dd_two_route_add_disable_decoder_dd,
                decoder_use_layernorm=concept_dd_two_route_add_decoder_use_layernorm,
                decoder_use_softmax=concept_dd_two_route_add_decoder_use_softmax,
                concept_use_layernorm=concept_dd_two_route_add_concept_use_layernorm,
                enable_raw_concept_route=concept_dd_two_route_add_enable_raw_concept_route,
                enable_final_concept_route=concept_dd_two_route_add_enable_final_concept_route,
                compile_routes=concept_compile_dd_routes,
            )
        else:
            self.dd_concept_builder = None
            self.dd_two_route_add = None

        self.decoder_read_encoder_routes = (
            nn.ModuleList(
                [
                    V21ResidualFlowRouteAdd(
                        hidden_size=hidden_size,
                        eps=eps,
                        num_source_states=self.concept_encoder_layers,
                        beta_init=concept_residual_flow_beta_init,
                        route_hidden_size=concept_residual_flow_route_hidden_size,
                        use_softmax=concept_residual_flow_route_use_softmax,
                        source_use_layernorm=(
                            False
                            if concept_residual_flow_shared_source_norm
                            else concept_residual_flow_source_use_layernorm
                        ),
                        compile_route=concept_compile_residual_flow_routes,
                    )
                    for _ in range(self.concept_decoder_layers)
                ]
            )
            if self.concept_enable_decoder_read_encoder and self.concept_encoder_layers > 0
            else None
        )
        self.decoder_read_encoder_shared_source_norm = (
            nn.LayerNorm(hidden_size, eps=eps)
            if self.decoder_read_encoder_routes is not None
            and concept_residual_flow_shared_source_norm
            and concept_residual_flow_source_use_layernorm
            else None
        )
        self.decoder_read_concept_routes = (
            nn.ModuleList(
                [
                    V21ResidualFlowRouteAdd(
                        hidden_size=hidden_size,
                        eps=eps,
                        num_source_states=self.concept_special_layers,
                        beta_init=concept_residual_flow_beta_init,
                        route_hidden_size=concept_residual_flow_route_hidden_size,
                        use_softmax=concept_residual_flow_route_use_softmax,
                        source_use_layernorm=(
                            False
                            if concept_residual_flow_shared_source_norm
                            else concept_residual_flow_source_use_layernorm
                        ),
                        compile_route=concept_compile_residual_flow_routes,
                    )
                    for _ in range(self.concept_decoder_layers)
                ]
            )
            if self.concept_enable_decoder_read_concept and self.concept_special_layers > 0
            else None
        )
        self.decoder_read_concept_shared_source_norm = (
            nn.LayerNorm(hidden_size, eps=eps)
            if self.decoder_read_concept_routes is not None
            and concept_residual_flow_shared_source_norm
            and concept_residual_flow_source_use_layernorm
            else None
        )
        self._v21_decoder_torch_compile_enabled = _env_flag(
            "ENABLE_MUDD_TORCH_COMPILE"
        ) or _env_flag("ENABLE_CONCEPTLM_V21_DECODER_TORCH_COMPILE")
        self._compiled_run_v21_decoder: Optional[Callable[..., Tensor]] = None

    def _assert_supported_v21_runtime(
        self,
        decoder_input: Tensor,
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> None:
        self._assert_supported_v2_runtime(decoder_input, inference_context, packed_seq_params)
        if self.config.recompute_granularity == "full" and self.training:
            raise NotImplementedError(
                "ConceptLM V2.1 per-layer DD/residual-flow routing does not support "
                "Megatron full activation recompute yet."
            )
        if self.config.fp8 or self.config.fp4:
            raise NotImplementedError(
                "ConceptLM V2.1 per-layer routing currently requires fp8=False and fp4=False."
            )

    def _run_v21_block(
        self,
        block,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[Tensor],
        padding_mask: Optional[Tensor],
        after_layer: Optional[Callable[[int, Tensor], Tensor]] = None,
    ) -> Tensor:
        if not block.pre_process:
            hidden_states = block.input_tensor
        hidden_states = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=True,
            keep_graph=True,
        )
        context = None
        for layer_idx, layer in enumerate(block.layers):
            with block.offload_context:
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    rotary_pos_cos_sin=rotary_pos_cos_sin,
                    attention_bias=None,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    padding_mask=padding_mask,
                )
            if (
                torch.is_grad_enabled()
                and block.config.cpu_offloading
                and block.group_prefetch_offload_commit_async is not None
            ):
                hidden_states = block.group_prefetch_offload_commit_async(hidden_states)
            if after_layer is not None:
                hidden_states = after_layer(layer_idx, hidden_states)

        if block.final_layernorm is not None:
            hidden_states = apply_module(block.final_layernorm)(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states,
                requires_grad=True,
                keep_graph=True,
            )
        return hidden_states

    def _run_v21_encoder(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[Tensor],
        padding_mask: Optional[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        encoder_raw_layer_states: list[Tensor] = []
        dd_history = [hidden_states] if self.dd_encoder_self_dd is not None else None

        def after_layer(layer_idx: int, layer_out: Tensor) -> Tensor:
            _v21_check_finite(f"encoder.layer{layer_idx}.raw", layer_out)
            raw_out = layer_out
            encoder_raw_layer_states.append(raw_out)
            if self.dd_encoder_self_dd is None:
                return layer_out
            dd_history.append(raw_out)
            out = self.dd_encoder_self_dd(layer_idx, layer_out, torch.stack(dd_history, dim=2))
            _v21_check_finite(f"encoder.layer{layer_idx}.self_dd", out)
            return out

        hidden_states = self._run_v21_block(
            self.encoder,
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_mask,
            after_layer=after_layer,
        )
        return hidden_states, encoder_raw_layer_states

    def _build_encoder_concept_states(self, encoder_raw_layer_states: list[Tensor]) -> Optional[Tensor]:
        routes = getattr(self.concept_predictor, "concept_read_encoder_routes", None)
        if routes is None or len(encoder_raw_layer_states) <= 1:
            return None
        chunk_states = [self._merge_token_chunks(state)[0] for state in encoder_raw_layer_states[:-1]]
        encoder_concept_states = torch.stack(chunk_states, dim=2)
        shared_norm = getattr(self.concept_predictor, "concept_read_encoder_shared_source_norm", None)
        if shared_norm is not None:
            encoder_concept_states = shared_norm(encoder_concept_states)
        return encoder_concept_states

    def _build_decoder_encoder_states(self, encoder_raw_layer_states: list[Tensor]) -> Optional[Tensor]:
        if self.decoder_read_encoder_routes is None or len(encoder_raw_layer_states) == 0:
            return None
        decoder_encoder_states = torch.stack(encoder_raw_layer_states, dim=2)
        if self.decoder_read_encoder_shared_source_norm is not None:
            decoder_encoder_states = self.decoder_read_encoder_shared_source_norm(
                decoder_encoder_states
            )
        return decoder_encoder_states

    def _concept_branch_v21(
        self,
        encoder_hidden_states: Tensor,
        encoder_raw_layer_states: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, Optional[tuple[Tensor, ...]]]:
        concept_hidden, usable_seq_len = self._merge_token_chunks(encoder_hidden_states)
        _v21_check_finite("concept_branch.concept_hidden", concept_hidden)
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

        encoder_concept_states = self._build_encoder_concept_states(encoder_raw_layer_states)
        need_concept_layer_states = (
            self.dd_two_route_add is not None
            and self.concept_dd_two_route_add_concept_source
            in ("hlm_layers", "hlm_layers_plus_final")
        ) or self.decoder_read_concept_routes is not None
        predicted_latents, concept_layer_states = self.concept_predictor(
            concept_hidden,
            encoder_concept_states=encoder_concept_states,
            return_layer_states=need_concept_layer_states,
        )
        _v21_check_finite("concept_branch.predicted_latents", predicted_latents)
        pred_latent_loss = self._latent_prediction_loss(
            predicted_latents[:-1],
            target_latents[1:],
        )

        predicted_hidden = self._decode_predicted_concepts(predicted_latents)
        _v21_check_finite("concept_branch.predicted_hidden", predicted_hidden)
        pred_hidden_loss = self._latent_prediction_loss(
            predicted_hidden[:-1],
            concept_target_hidden[1:],
        )
        hlm_loss = self.concept_mlp_hlm_loss_weight * (
            pred_latent_loss + self.concept_mlp_hidden_loss_weight * pred_hidden_loss
        )

        repeated_concepts = self._repeat_shift_concepts(
            predicted_hidden,
            encoder_hidden_states.shape[0],
        )
        _v21_check_finite("concept_branch.repeated_concepts", repeated_concepts)
        decoder_input = self._apply_fusion(
            encoder_hidden_states,
            repeated_concepts.to(dtype=encoder_hidden_states.dtype),
        )
        _v21_check_finite("concept_branch.decoder_input", decoder_input)

        metrics = {
            "conceptlm_v21/usable_seq_len": torch.tensor(
                usable_seq_len, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v21/concept_chunks": torch.tensor(
                concept_hidden.shape[0], device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v21/latent_dim": torch.tensor(
                self.concept_latent_dim, device=encoder_hidden_states.device, dtype=torch.float32
            ),
            "conceptlm_v21/recon_loss_raw": recon_loss.detach().float(),
            "conceptlm_v21/pred_latent_loss_raw": pred_latent_loss.detach().float(),
            "conceptlm_v21/pred_hidden_loss_raw": pred_hidden_loss.detach().float(),
        }
        metrics.update(self._collect_v21_route_metrics(encoder_hidden_states.device))
        return (
            decoder_input,
            vq_loss,
            hlm_loss,
            metrics,
            predicted_hidden,
            repeated_concepts,
            concept_layer_states,
        )

    def _build_dd_concept_candidates(
        self,
        final_concept_chunk_states: Tensor,
        repeated_final_concept_states: Tensor,
        concept_layer_states: Optional[tuple[Tensor, ...]],
        seq_len: int,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.dd_concept_builder is None:
            return repeated_final_concept_states, None
        layer_stack = None
        if concept_layer_states is not None:
            layer_stack = torch.stack(concept_layer_states, dim=2)
        return self.dd_concept_builder(final_concept_chunk_states, layer_stack, seq_len)

    def _build_decoder_concept_states(
        self,
        concept_layer_states: Optional[tuple[Tensor, ...]],
    ) -> Optional[Tensor]:
        if self.decoder_read_concept_routes is None or concept_layer_states is None:
            return None
        layer_stack = torch.stack(concept_layer_states, dim=2)
        zero_chunk = torch.zeros_like(layer_stack[:1])
        decoder_concept_states = torch.cat((zero_chunk, layer_stack), dim=0)
        if self.decoder_read_concept_shared_source_norm is not None:
            decoder_concept_states = self.decoder_read_concept_shared_source_norm(
                decoder_concept_states
            )
        return decoder_concept_states

    def _run_v21_decoder_compiled(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[Tensor],
        padding_mask: Optional[Tensor],
        dd_concept_states: Optional[Tensor],
        final_concept_state: Optional[Tensor],
        decoder_encoder_states: Optional[Tensor],
        decoder_concept_states: Optional[Tensor],
    ) -> Tensor:
        if not self._v21_decoder_torch_compile_enabled:
            return self._run_v21_decoder(
                hidden_states,
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
        if self._compiled_run_v21_decoder is None:
            self._compiled_run_v21_decoder = _compile_v21_configurable(
                self._run_v21_decoder,
                backend=_env_str("MUDD_TORCH_COMPILE_BACKEND", "inductor"),
                mode=_env_str(
                    "MUDD_TORCH_COMPILE_MODE",
                    "max-autotune-no-cudagraphs",
                ),
                fullgraph=_env_flag("MUDD_TORCH_COMPILE_FULLGRAPH", False),
                dynamic=_env_flag("MUDD_TORCH_COMPILE_DYNAMIC", False),
            )
        return self._compiled_run_v21_decoder(
            hidden_states,
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

    def _run_v21_decoder(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[Tensor],
        padding_mask: Optional[Tensor],
        dd_concept_states: Optional[Tensor],
        final_concept_state: Optional[Tensor],
        decoder_encoder_states: Optional[Tensor],
        decoder_concept_states: Optional[Tensor],
    ) -> Tensor:
        dd_history = [hidden_states]

        def after_layer(layer_idx: int, layer_out: Tensor) -> Tensor:
            _v21_check_finite(f"decoder.layer{layer_idx}.raw", layer_out)
            hidden = layer_out
            dd_history.append(layer_out)
            if self.dd_two_route_add is not None:
                hidden = self.dd_two_route_add(
                    layer_idx,
                    hidden,
                    torch.stack(dd_history, dim=2),
                    dd_concept_states,
                    final_concept_state,
                )
                _v21_check_finite(f"decoder.layer{layer_idx}.dd_two_route_add", hidden)
            if self.decoder_read_encoder_routes is not None and decoder_encoder_states is not None:
                hidden = self.decoder_read_encoder_routes[layer_idx](
                    hidden,
                    decoder_encoder_states,
                    self.decoder_read_encoder_shared_source_norm is not None,
                )
                _v21_check_finite(f"decoder.layer{layer_idx}.read_encoder", hidden)
            if self.decoder_read_concept_routes is not None and decoder_concept_states is not None:
                hidden = self.decoder_read_concept_routes[layer_idx].forward_repeated_chunks(
                    hidden,
                    decoder_concept_states,
                    self.concept_chunk_size,
                    self.concept_shift_feature,
                    self.decoder_read_concept_shared_source_norm is not None,
                )
                _v21_check_finite(f"decoder.layer{layer_idx}.read_concept", hidden)
            return hidden

        return self._run_v21_block(
            self.decoder,
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_mask,
            after_layer=after_layer,
        )

    def _mean_tensor_metric(self, values: list[Tensor], device: torch.device) -> Optional[Tensor]:
        if not values:
            return None
        return torch.stack([value.detach().float().view(()) for value in values]).mean().to(device)

    def _collect_v21_route_metrics(self, device: torch.device) -> Dict[str, Tensor]:
        metrics: Dict[str, Tensor] = {}

        def add_mean(name: str, values: list[Tensor]) -> None:
            value = self._mean_tensor_metric(values, device)
            if value is not None:
                metrics[name] = value

        if self.dd_two_route_add is not None:
            final_diag_values = []
            final_norm_values = []
            raw_diag_values = []
            raw_norm_values = []
            for route in self.dd_two_route_add.concept_routes:
                if route.final_proj is not None:
                    final_diag_values.append(route.final_proj.weight.diagonal().mean())
                    final_norm_values.append(route.final_proj.weight.norm())
                if route.raw_proj is not None:
                    raw_diag_values.append(route.raw_proj.weight.diagonal().mean())
                    raw_norm_values.append(route.raw_proj.weight.norm())
            add_mean("conceptlm_v21/dd_final_concept_proj_diag_mean", final_diag_values)
            add_mean("conceptlm_v21/dd_final_concept_proj_norm_mean", final_norm_values)
            add_mean("conceptlm_v21/dd_raw_concept_proj_diag_mean", raw_diag_values)
            add_mean("conceptlm_v21/dd_raw_concept_proj_norm_mean", raw_norm_values)

        def collect_resflow(prefix: str, routes: Optional[nn.ModuleList]) -> None:
            if routes is None:
                return
            diag_values = [route.residual_proj.weight.diagonal().mean() for route in routes]
            norm_values = [route.residual_proj.weight.norm() for route in routes]
            add_mean(f"conceptlm_v21/{prefix}_residual_proj_diag_mean", diag_values)
            add_mean(f"conceptlm_v21/{prefix}_residual_proj_norm_mean", norm_values)

        collect_resflow("resflow_decoder_read_encoder", self.decoder_read_encoder_routes)
        collect_resflow("resflow_decoder_read_concept", self.decoder_read_concept_routes)
        collect_resflow(
            "resflow_concept_read_encoder",
            getattr(self.concept_predictor, "concept_read_encoder_routes", None),
        )
        return metrics

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
    ) -> Tensor | ConceptLMV21Output:
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
        (
            decoder_hidden_states,
            vq_loss,
            hlm_loss,
            concept_metrics,
            final_concept_chunk_states,
            repeated_final_concept_states,
            concept_layer_states,
        ) = self._concept_branch_v21(encoder_hidden_states, encoder_raw_layer_states)

        final_concept_state, dd_concept_states = self._build_dd_concept_candidates(
            final_concept_chunk_states,
            repeated_final_concept_states,
            concept_layer_states,
            encoder_hidden_states.shape[0],
        )
        decoder_encoder_states = self._build_decoder_encoder_states(encoder_raw_layer_states)
        decoder_concept_states = self._build_decoder_concept_states(concept_layer_states)

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
        _v21_check_finite("decoder.final_hidden", hidden_states)

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
        _v21_check_finite("decoder.logits", logits)
        lm_output = self.compute_language_model_loss(labels, logits)
        _v21_check_finite("decoder.lm_output", lm_output)
        _maybe_print_v21_ce_debug(hidden_states, logits, labels, lm_output, input_ids=input_ids)
        return ConceptLMV21Output(
            lm_loss=lm_output,
            vq_loss=vq_loss,
            hlm_loss=hlm_loss,
            concept_metrics=concept_metrics,
        )
