#!/usr/bin/env python3
# Copyright (c) 2026, ConceptLM contributors.

"""Pretrain entrypoint for the Megatron-Core ConceptLM V2.1 prototype."""

from __future__ import annotations

import os
import time
from functools import partial
from typing import Optional

import torch

from conceptlm_v21_builders import conceptlm_v21_builder
from megatron.core.enums import ModelType
from megatron.core.models.gpt.conceptlm_v21 import ConceptLMV21Output
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_timers, inprocess_restart, pretrain, set_startup_timestamps
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from pretrain_gpt import (
    SPIKY_LOSS_FACTOR,
    _env_float,
    _should_monitor_exit_hidden_rank,
    get_batch,
    get_embedding_ranks,
    has_nvidia_modelopt,
    olmo3_z_loss_output_processor,
    stimer,
    train_valid_test_datasets_provider,
)

if has_nvidia_modelopt:
    from megatron.post_training.arguments import add_modelopt_args

_PROGRAM_START_TIME = time.time()


def _tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    flat = detached.float().reshape(-1)
    numel = flat.numel()
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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _print_v21_loss_debug(
    output_tensor: ConceptLMV21Output,
    loss_mask: torch.Tensor,
    losses: torch.Tensor,
    masked_losses: torch.Tensor,
    token_loss: torch.Tensor,
    raw_vq: torch.Tensor,
    raw_hlm: torch.Tensor,
    weighted_vq: torch.Tensor,
    weighted_hlm: torch.Tensor,
    aux_loss: torch.Tensor,
    total_loss: torch.Tensor,
) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    print(f"[ConceptLM V2.1 loss debug][rank {rank}] non-finite total_loss detected", flush=True)
    for name, tensor in (
        ("total_loss", total_loss),
        ("token_loss", token_loss),
        ("lm_loss_per_token", losses),
        ("loss_mask", loss_mask),
        ("masked_lm_loss_per_token", masked_losses),
        ("raw_vq", raw_vq),
        ("raw_hlm", raw_hlm),
        ("weighted_vq", weighted_vq),
        ("weighted_hlm", weighted_hlm),
        ("aux_loss", aux_loss),
    ):
        print(
            f"[ConceptLM V2.1 loss debug][rank {rank}] {_tensor_debug_summary(name, tensor)}",
            flush=True,
        )
    bad_loss_mask = ~torch.isfinite(losses.detach())
    bad_masked_mask = ~torch.isfinite(masked_losses.detach())
    bad = torch.nonzero(bad_loss_mask | bad_masked_mask, as_tuple=False).view(-1)
    if bad.numel():
        max_bad = max(1, _env_int("CONCEPTLM_V21_LOSS_DEBUG_MAX_BAD", 16))
        seq_length = getattr(get_args(), "seq_length", None)
        print(
            f"[ConceptLM V2.1 loss debug][rank {rank}] bad_loss_positions "
            f"count={int(bad.numel())}, seq_length={seq_length}",
            flush=True,
        )
        for pos in bad[:max_bad]:
            flat_idx = int(pos.item())
            if seq_length:
                sample_idx = flat_idx // int(seq_length)
                token_idx = flat_idx % int(seq_length)
                token_text = f", sample={sample_idx}, token={token_idx}"
            else:
                token_text = ""
            print(
                f"[ConceptLM V2.1 loss debug][rank {rank}] bad_loss "
                f"flat={flat_idx}{token_text}, "
                f"loss={float(losses.detach()[flat_idx].float().item())}, "
                f"mask={float(loss_mask.detach()[flat_idx].float().item())}, "
                f"masked={float(masked_losses.detach()[flat_idx].float().item())}, "
                f"loss_finite={bool(torch.isfinite(losses.detach()[flat_idx]).item())}, "
                f"masked_finite={bool(torch.isfinite(masked_losses.detach()[flat_idx]).item())}",
                flush=True,
            )
    for name, tensor in sorted(output_tensor.concept_metrics.items()):
        print(
            f"[ConceptLM V2.1 loss debug][rank {rank}] {_tensor_debug_summary(name, tensor)}",
            flush=True,
        )


def _add_bool_pair(group, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")


def _add_conceptlm_v21_args(parser):
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)

    group = parser.add_argument_group(title="ConceptLM V2.1")
    group.add_argument(
        "--conceptlm-backbone",
        type=str,
        default="pythia",
        choices=["olmo3", "pythia", "gpt"],
    )
    group.add_argument("--conceptlm-encoder-layers", type=int, default=6)
    group.add_argument("--conceptlm-special-layers", type=int, default=6)
    group.add_argument("--conceptlm-decoder-layers", type=int, default=6)
    group.add_argument("--conceptlm-chunk-size", type=int, default=4)
    _add_bool_pair(group, "conceptlm-shift-feature", True, "Shift repeated concept features by one token.")
    group.add_argument(
        "--conceptlm-chunk-merge-method",
        type=str,
        default="meanpooling",
        choices=["meanpooling", "first", "last"],
    )
    _add_bool_pair(
        group,
        "conceptlm-enable-chunk-dualpath-smoothing",
        False,
        "Enable chunk dual-path smoothing before the concept tower.",
    )
    group.add_argument("--conceptlm-chunk-dualpath-alpha-init", type=float, default=0.5)
    group.add_argument(
        "--conceptlm-bottleneck-type",
        type=str,
        default="mlp",
        choices=["mlp"],
    )
    group.add_argument("--conceptlm-vq-patch-ratio", type=int, default=1)
    group.add_argument("--conceptlm-mlp-bottleneck-ratio", type=float, default=0.25)
    group.add_argument(
        "--conceptlm-mlp-bottleneck-activation",
        type=str,
        default="none",
        choices=["gelu", "none"],
    )
    group.add_argument("--conceptlm-mlp-recon-loss-weight", type=float, default=0.0)
    group.add_argument("--conceptlm-mlp-hlm-loss-weight", type=float, default=0.0)
    group.add_argument("--conceptlm-mlp-hidden-loss-weight", type=float, default=1.0)
    group.add_argument(
        "--conceptlm-mlp-hlm-loss-type",
        type=str,
        default="mse",
        choices=["mse", "cosine"],
    )
    group.add_argument(
        "--conceptlm-layer-norm-option",
        type=str,
        default="normed_add",
        choices=["rawadd", "rawadd_scalar_gate", "normed_add"],
    )
    group.add_argument("--conceptlm-fusion-norm-alpha-init", type=float, default=0.1)
    group.add_argument("--conceptlm-fusion-alpha-init", type=float, default=1.0)
    group.add_argument("--conceptlm-hlm-ffn-hidden-size", type=int, default=None)
    group.add_argument("--conceptlm-hlm-loss-weight", type=float, default=1.0)
    group.add_argument("--conceptlm-vq-loss-weight", type=float, default=1.0)

    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add",
        False,
        "Enable decoder DD plus concept two-route add.",
    )
    group.add_argument(
        "--conceptlm-v21-dd-two-route-add-concept-source",
        type=str,
        default="final",
        choices=["final", "hlm_layers", "hlm_layers_plus_final"],
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-enable-raw-concept-route",
        True,
        "Enable raw concept layer candidates in decoder concept routing.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-enable-final-concept-route",
        True,
        "Enable final concept route into the decoder.",
    )
    group.add_argument("--conceptlm-v21-dd-two-route-add-beta-init", type=float, default=0.3)
    group.add_argument("--conceptlm-v21-dd-two-route-add-every-n-layers", type=int, default=1)
    group.add_argument("--conceptlm-v21-dd-two-route-add-concept-route-first-n", type=int, default=-1)
    group.add_argument("--conceptlm-v21-dd-two-route-add-decoder-hidden-size", type=int, default=0)
    group.add_argument("--conceptlm-v21-dd-two-route-add-concept-hidden-size", type=int, default=0)
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-use-softmax",
        True,
        "Use softmax for decoder concept routing.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-disable-decoder-dd",
        False,
        "Disable decoder self-DD while keeping concept routes.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-decoder-use-layernorm",
        False,
        "Normalize decoder DD history candidates.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-decoder-use-softmax",
        True,
        "Use softmax for decoder DD history routing.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-concept-use-layernorm",
        True,
        "Normalize decoder concept route candidates.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-encoder-self-dd",
        False,
        "Enable encoder self-DD over layer history.",
    )
    group.add_argument("--conceptlm-v21-dd-encoder-self-dd-every-n-layers", type=int, default=1)
    group.add_argument("--conceptlm-v21-dd-encoder-self-dd-hidden-size", type=int, default=0)
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-encoder-self-dd-use-layernorm",
        False,
        "Normalize encoder DD history candidates.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-concept-self-dd",
        False,
        "Enable concept self-DD over HLM layer history.",
    )
    group.add_argument("--conceptlm-v21-dd-concept-self-dd-every-n-layers", type=int, default=1)
    group.add_argument("--conceptlm-v21-dd-concept-self-dd-hidden-size", type=int, default=0)
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-concept-self-dd-use-layernorm",
        False,
        "Normalize concept DD history candidates.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-enable-full-residual-flow",
        False,
        "Enable all V2.1 cross-module residual routes.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-enable-concept-read-encoder",
        False,
        "Enable encoder-to-concept residual routes.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-enable-decoder-read-encoder",
        False,
        "Enable encoder-to-decoder residual routes.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-enable-decoder-read-concept",
        False,
        "Enable concept-to-decoder residual routes.",
    )
    group.add_argument("--conceptlm-v21-residual-flow-beta-init", type=float, default=0.02)
    group.add_argument("--conceptlm-v21-residual-flow-route-hidden-size", type=int, default=0)
    _add_bool_pair(
        group,
        "conceptlm-v21-residual-flow-route-use-softmax",
        True,
        "Use softmax for same-source residual-flow routing.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-residual-flow-source-use-layernorm",
        True,
        "Normalize residual-flow source candidates.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-residual-flow-shared-source-norm",
        False,
        "Share one source norm per residual-flow source group.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-compile-residual-flow-routes",
        True,
        "Compile V2.1 residual-flow route-add tensor helpers with torch.compile.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-compile-dd-routes",
        True,
        "Compile V2.1 DD route tensor helpers with torch.compile.",
    )
    return parser


def conceptlm_v21_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    if getattr(args, "modelopt_enabled", False):
        raise NotImplementedError("ConceptLM V2.1 currently does not support modelopt_enabled")
    return conceptlm_v21_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: ConceptLMV21Output, model=None):
    """Combine token CE, bottleneck loss, and HLM prediction loss."""

    del model
    if not isinstance(output_tensor, ConceptLMV21Output):
        raise TypeError(f"ConceptLM V2.1 loss expected ConceptLMV21Output, got {type(output_tensor)}")

    args = get_args()
    losses = output_tensor.lm_loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    num_tokens_float = loss_mask.sum()
    masked_losses = losses * loss_mask
    token_loss = torch.sum(masked_losses)

    raw_vq = output_tensor.vq_loss.float()
    raw_hlm = output_tensor.hlm_loss.float()
    weighted_vq = raw_vq * args.conceptlm_vq_loss_weight
    weighted_hlm = raw_hlm * args.conceptlm_hlm_loss_weight
    aux_loss = weighted_vq + weighted_hlm
    total_loss = token_loss + aux_loss

    local_num_tokens = num_tokens_float.clone().detach().to(torch.int)
    normalizer = torch.clamp(num_tokens_float.detach().float(), min=1.0)
    total_loss_avg = total_loss.detach() / normalizer

    report = {
        "lm loss": torch.cat([token_loss.detach().view(1), num_tokens_float.detach().view(1)]),
        "conceptlm_v21 total loss": torch.cat(
            [total_loss.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v21 vq loss raw": raw_vq.detach().view(1),
        "conceptlm_v21 hlm loss raw": raw_hlm.detach().view(1),
        "conceptlm_v21 aux loss raw": aux_loss.detach().view(1),
        "conceptlm_v21 vq loss effective": torch.cat(
            [weighted_vq.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v21 hlm loss effective": torch.cat(
            [weighted_hlm.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v21 aux loss effective": torch.cat(
            [aux_loss.detach().view(1), num_tokens_float.detach().view(1)]
        ),
    }
    for name, value in output_tensor.concept_metrics.items():
        report[name] = value.detach().view(1)

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        if not torch.isfinite(total_loss.detach()).all():
            _print_v21_loss_debug(
                output_tensor,
                loss_mask,
                losses,
                masked_losses,
                token_loss,
                raw_vq,
                raw_hlm,
                weighted_vq,
                weighted_hlm,
                aux_loss,
                total_loss,
            )
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isnan,
            message="found NaN in local ConceptLM V2.1 forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isinf,
            message="found Inf in local ConceptLM V2.1 forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=total_loss_avg,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )
    return total_loss, local_num_tokens, report


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward step for ConceptLM V2.1."""

    if return_schedule_plan:
        raise NotImplementedError("ConceptLM V2.1 does not support schedule-plan mode yet")

    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator,
            vp_stage,
        )
    timers("batch-generator").stop()

    with stimer:
        output_processor = None
        if _should_monitor_exit_hidden_rank(get_args()) or _env_float("OLMO3_Z_LOSS_MULTIPLIER", 0.0) > 0.0:
            output_processor = olmo3_z_loss_output_processor
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            output_processor=output_processor,
        )
    return output_tensor, partial(loss_func, loss_mask, model=model)


def main(argv: Optional[list[str]] = None):
    del argv
    main_entry_time = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=main_entry_time)
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(
        extra_args_provider=_add_conceptlm_v21_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    full_config = pretrain_cfg_container_from_args(args)
    wrapped_pretrain(
        full_config,
        train_valid_test_datasets_provider,
        conceptlm_v21_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
