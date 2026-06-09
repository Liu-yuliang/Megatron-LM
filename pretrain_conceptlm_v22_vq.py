#!/usr/bin/env python3
# Copyright (c) 2026, ConceptLM contributors.

"""Pretrain entrypoint for ConceptLM V2.2-VQ."""

from __future__ import annotations

import time
from functools import partial
from typing import Optional

import torch

from conceptlm_v22_vq_builders import conceptlm_v22_vq_builder
from megatron.core.enums import ModelType
from megatron.core.models.gpt.conceptlm_v22_vq import ConceptLMV22VQOutput
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


def _print_v22_vq_loss_debug(
    output_tensor: ConceptLMV22VQOutput,
    loss_mask: torch.Tensor,
    losses: torch.Tensor,
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
    print(f"[ConceptLM V2.2-VQ loss debug][rank {rank}] non-finite total_loss detected", flush=True)
    for name, tensor in (
        ("total_loss", total_loss),
        ("token_loss", token_loss),
        ("lm_loss_per_token", losses),
        ("loss_mask", loss_mask),
        ("raw_vq", raw_vq),
        ("raw_hlm", raw_hlm),
        ("weighted_vq", weighted_vq),
        ("weighted_hlm", weighted_hlm),
        ("aux_loss", aux_loss),
    ):
        print(
            f"[ConceptLM V2.2-VQ loss debug][rank {rank}] {_tensor_debug_summary(name, tensor)}",
            flush=True,
        )
    for name, tensor in sorted(output_tensor.concept_metrics.items()):
        print(
            f"[ConceptLM V2.2-VQ loss debug][rank {rank}] {_tensor_debug_summary(name, tensor)}",
            flush=True,
        )


def _add_bool_pair(group, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")


def _add_conceptlm_v22_vq_args(parser):
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)

    group = parser.add_argument_group(title="ConceptLM V2.2-VQ")
    group.add_argument(
        "--conceptlm-backbone",
        type=str,
        default="olmo3",
        choices=["olmo3", "pythia", "gpt"],
    )
    group.add_argument("--conceptlm-encoder-layers", type=int, default=16)
    group.add_argument("--conceptlm-special-layers", type=int, default=8)
    group.add_argument("--conceptlm-decoder-layers", type=int, default=16)
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

    group.add_argument("--conceptlm-v22-vq-codebook-size", type=int, default=128)
    group.add_argument(
        "--conceptlm-v22-vq-num-codebooks",
        type=int,
        default=32,
        help="Number of VQ codebooks.",
    )
    group.add_argument("--conceptlm-v22-vq-commitment-cost", type=float, default=0.25)
    group.add_argument(
        "--conceptlm-v22-vq-merge-mode",
        type=str,
        default="raw_logits",
        choices=["raw_logits", "softmax", "hard_top1"],
    )
    group.add_argument(
        "--conceptlm-v22-vq-hlm-loss-type",
        type=str,
        default="mse",
        choices=["mse", "ce"],
    )
    _add_bool_pair(
        group,
        "conceptlm-v22-vq-detach-hlm-target",
        True,
        "Detach concept targets for high-level MSE loss.",
    )
    group.add_argument("--conceptlm-v22-vq-lr-mult", type=float, default=1.0)
    group.add_argument("--conceptlm-v22-vq-highlevel-lr-mult", type=float, default=1.0)

    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add",
        True,
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
        False,
        "Enable raw concept layer candidates in decoder concept routing.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-enable-final-concept-route",
        True,
        "Enable final concept route into the decoder.",
    )
    group.add_argument("--conceptlm-v21-dd-two-route-add-beta-init", type=float, default=0.05)
    group.add_argument("--conceptlm-v21-dd-two-route-add-every-n-layers", type=int, default=1)
    group.add_argument("--conceptlm-v21-dd-two-route-add-concept-route-first-n", type=int, default=-1)
    group.add_argument("--conceptlm-v21-dd-two-route-add-decoder-hidden-size", type=int, default=0)
    group.add_argument("--conceptlm-v21-dd-two-route-add-concept-hidden-size", type=int, default=0)
    _add_bool_pair(
        group,
        "conceptlm-v21-dd-two-route-add-use-softmax",
        False,
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
        "Normalize decoder DD depth weights with softmax.",
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
        True,
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
        True,
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
        True,
        "Enable encoder-to-concept residual routes.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-enable-decoder-read-encoder",
        True,
        "Enable encoder-to-decoder residual routes.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-enable-decoder-read-concept",
        True,
        "Enable concept-to-decoder residual routes.",
    )
    group.add_argument(
        "--conceptlm-v21-concept-read-encoder-first-n",
        type=int,
        default=-1,
        help="Apply encoder-to-concept residual routes only to the first N concept layers; negative means all enabled layers.",
    )
    group.add_argument(
        "--conceptlm-v21-decoder-read-encoder-first-n",
        type=int,
        default=-1,
        help="Apply encoder-to-decoder residual routes only to the first N decoder layers; negative means all enabled layers.",
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
        True,
        "Share one source norm per residual-flow source group.",
    )
    _add_bool_pair(
        group,
        "conceptlm-v21-final-read-concept-gate",
        True,
        "Use a per-layer softmax gate whose final-concept and decoder-read-concept weights sum to 1.",
    )
    group.add_argument(
        "--conceptlm-v21-final-read-concept-gate-init-final",
        type=float,
        default=0.5,
        help="Initial final-concept share for the final/read-concept softmax gate.",
    )
    group.add_argument(
        "--conceptlm-v21-final-read-concept-gate-target-final",
        type=float,
        default=0.5,
        help="Target final-concept share used by the optional final/read-concept gate regularizer.",
    )
    group.add_argument(
        "--conceptlm-v21-final-read-concept-gate-reg-weight",
        type=float,
        default=0.0,
        help="Weight for the final/read-concept gate target regularizer.",
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


def conceptlm_v22_vq_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    if getattr(args, "modelopt_enabled", False):
        raise NotImplementedError("ConceptLM V2.2-VQ currently does not support modelopt_enabled")
    return conceptlm_v22_vq_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: ConceptLMV22VQOutput, model=None):
    """Combine token CE, VQ loss, and high-level concept prediction loss."""

    del model
    if not isinstance(output_tensor, ConceptLMV22VQOutput):
        raise TypeError(
            f"ConceptLM V2.2-VQ loss expected ConceptLMV22VQOutput, got {type(output_tensor)}"
        )

    args = get_args()
    losses = output_tensor.lm_loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    num_tokens_float = loss_mask.sum()
    token_loss = torch.sum(losses * loss_mask)

    raw_vq = output_tensor.vq_loss.float()
    raw_hlm = output_tensor.hlm_loss.float()
    weighted_vq = raw_vq * args.conceptlm_vq_loss_weight
    weighted_hlm = raw_hlm * args.conceptlm_hlm_loss_weight
    aux_loss = weighted_vq + weighted_hlm
    route_reg_raw = output_tensor.concept_metrics.get(
        "conceptlm_v22_vq/final_read_concept_gate_reg_raw",
    )
    if route_reg_raw is None:
        route_reg_raw = token_loss.new_zeros(())
    else:
        route_reg_raw = route_reg_raw.float()
    local_num_tokens = num_tokens_float.clone().detach().to(torch.int)
    normalizer = torch.clamp(num_tokens_float.detach().float(), min=1.0)
    route_reg_loss = (
        route_reg_raw
        * args.conceptlm_v21_final_read_concept_gate_reg_weight
        * normalizer
    )
    total_loss = token_loss + aux_loss + route_reg_loss
    total_loss_avg = total_loss.detach() / normalizer

    report = {
        "lm loss": torch.cat([token_loss.detach().view(1), num_tokens_float.detach().view(1)]),
        "conceptlm_v22_vq total loss": torch.cat(
            [total_loss.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v22_vq vq loss raw": raw_vq.detach().view(1),
        "conceptlm_v22_vq hlm loss raw": raw_hlm.detach().view(1),
        "conceptlm_v22_vq aux loss raw": aux_loss.detach().view(1),
        "conceptlm_v22_vq vq loss effective": torch.cat(
            [weighted_vq.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v22_vq hlm loss effective": torch.cat(
            [weighted_hlm.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v22_vq aux loss effective": torch.cat(
            [aux_loss.detach().view(1), num_tokens_float.detach().view(1)]
        ),
        "conceptlm_v22_vq route reg loss raw": route_reg_raw.detach().view(1),
        "conceptlm_v22_vq route reg loss effective": torch.cat(
            [route_reg_loss.detach().view(1), num_tokens_float.detach().view(1)]
        ),
    }
    for name, value in output_tensor.concept_metrics.items():
        report[name] = value.detach().view(1)

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        if not torch.isfinite(total_loss.detach()).all():
            _print_v22_vq_loss_debug(
                output_tensor,
                loss_mask,
                losses,
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
            message="found NaN in local ConceptLM V2.2-VQ forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isinf,
            message="found Inf in local ConceptLM V2.2-VQ forward loss calculation",
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
    """Forward step for ConceptLM V2.2-VQ."""

    if return_schedule_plan:
        raise NotImplementedError("ConceptLM V2.2-VQ does not support schedule-plan mode yet")

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
        extra_args_provider=_add_conceptlm_v22_vq_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    full_config = pretrain_cfg_container_from_args(args)
    wrapped_pretrain(
        full_config,
        train_valid_test_datasets_provider,
        conceptlm_v22_vq_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
