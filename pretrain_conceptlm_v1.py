# Copyright (c) 2026, ConceptLM contributors.

"""Pretrain entrypoint for the isolated ConceptLM V1 prototype."""

import time
from functools import partial
from typing import Optional

import torch

from conceptlm_v1_builders import conceptlm_v1_builder
from megatron.core.enums import ModelType
from megatron.core.models.gpt.conceptlm_v1 import ConceptLMV1Output
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


def _add_conceptlm_v1_args(parser):
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)

    group = parser.add_argument_group(title="ConceptLM V1")
    group.add_argument("--conceptlm-chunk-size", type=int, default=4)
    group.add_argument(
        "--conceptlm-backbone",
        type=str,
        default="olmo3",
        choices=["olmo3", "pythia", "gpt"],
        help="Backbone layer spec to use when --spec is not provided.",
    )
    group.add_argument("--conceptlm-codebook-size", type=int, default=128)
    group.add_argument(
        "--conceptlm-num-codebooks",
        type=int,
        default=None,
        help="Defaults to num_attention_heads when omitted.",
    )
    group.add_argument("--conceptlm-special-layers", type=int, default=2)
    group.add_argument(
        "--conceptlm-loss-type",
        type=str,
        default="hlm_MSE_loss",
        choices=["hlm_MSE_loss", "hlm_CE_loss"],
    )
    group.add_argument(
        "--conceptlm-merge-mode",
        type=str,
        default="raw_logits",
        choices=["raw_logits", "softmax", "hard_top1"],
    )
    group.add_argument("--conceptlm-ncp-loss-weight", type=float, default=1.0)
    group.add_argument("--conceptlm-vq-loss-weight", type=float, default=1.0)
    group.add_argument("--conceptlm-vq-commitment-cost", type=float, default=0.25)
    group.add_argument("--conceptlm-hlm-ffn-hidden-size", type=int, default=None)
    group.add_argument(
        "--conceptlm-detach-ncp-target",
        dest="conceptlm_detach_ncp_target",
        action="store_true",
        default=True,
    )
    group.add_argument(
        "--conceptlm-no-detach-ncp-target",
        dest="conceptlm_detach_ncp_target",
        action="store_false",
    )
    return parser


def conceptlm_v1_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    """Build ConceptLM V1 directly, without the shared model_provider indirection."""

    args = get_args()
    if getattr(args, "modelopt_enabled", False):
        raise NotImplementedError("ConceptLM V1 currently does not support modelopt_enabled")
    return conceptlm_v1_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: ConceptLMV1Output, model=None):
    """Combine token CE, VQ loss, and NCP loss using the zip-style scaling.

    Megatron's three-return loss path divides the returned scalar by the local
    token count. Keep aux losses in the summed numerator so their effective
    weight matches the reference zip implementation.
    """

    del model
    if not isinstance(output_tensor, ConceptLMV1Output):
        raise TypeError(f"ConceptLM V1 loss expected ConceptLMV1Output, got {type(output_tensor)}")

    args = get_args()
    losses = output_tensor.lm_loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    num_tokens = loss_mask.sum()
    token_loss = torch.sum(losses * loss_mask)

    raw_ncp = output_tensor.ncp_loss.float()
    raw_vq = output_tensor.vq_loss.float()
    weighted_ncp = raw_ncp * args.conceptlm_ncp_loss_weight
    weighted_vq = raw_vq * args.conceptlm_vq_loss_weight
    aux_loss = weighted_ncp + weighted_vq
    total_loss = token_loss + aux_loss

    local_num_tokens = num_tokens.clone().detach().to(torch.int)
    normalizer = torch.clamp(num_tokens.detach().float(), min=1.0)
    total_loss_avg = total_loss.detach() / normalizer

    report = {
        "lm loss": torch.cat([token_loss.detach().view(1), num_tokens.detach().view(1)]),
        "conceptlm total loss": torch.cat([total_loss.detach().view(1), num_tokens.detach().view(1)]),
        "conceptlm ncp loss raw": raw_ncp.detach().view(1),
        "conceptlm vq loss raw": raw_vq.detach().view(1),
        "conceptlm aux loss raw": aux_loss.detach().view(1),
        "conceptlm ncp loss effective": torch.cat([weighted_ncp.detach().view(1), num_tokens.detach().view(1)]),
        "conceptlm vq loss effective": torch.cat([weighted_vq.detach().view(1), num_tokens.detach().view(1)]),
        "conceptlm aux loss effective": torch.cat([aux_loss.detach().view(1), num_tokens.detach().view(1)]),
    }

    for name, value in output_tensor.concept_metrics.items():
        report[name] = value.detach().view(1)

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isnan,
            message="found NaN in local ConceptLM V1 forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isinf,
            message="found Inf in local ConceptLM V1 forward loss calculation",
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
    """Forward step for ConceptLM V1."""

    if return_schedule_plan:
        raise NotImplementedError("ConceptLM V1 does not support schedule-plan mode yet")

    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator, vp_stage
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
        extra_args_provider=_add_conceptlm_v1_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    full_config = pretrain_cfg_container_from_args(args)
    wrapped_pretrain(
        full_config,
        train_valid_test_datasets_provider,
        conceptlm_v1_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
