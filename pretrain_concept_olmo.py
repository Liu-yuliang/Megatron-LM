# Copyright (c) 2026

"""Pretrain Concept-OLMo in Megatron-LM."""

import time
from functools import partial
from typing import Optional

import torch

from concept_olmo_builder import add_concept_olmo_extra_args, concept_olmo_builder
from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from model_provider import model_provider
from pretrain_gpt import (
    _PROGRAM_START_TIME,
    SPIKY_LOSS_FACTOR,
    get_batch,
    get_embedding_ranks,
    get_rerun_state_machine,
    get_timers as _unused_get_timers,
    has_nvidia_modelopt,
    is_dataset_built_on_rank,
    loss_func as gpt_loss_func,
    mtp_on_this_rank,
    set_startup_timestamps as _unused_startup,
    stimer,
    train_valid_test_datasets_provider,
)


def loss_func(
    loss_mask: torch.Tensor, output_tensor, model=None
):
    args = get_args()
    losses = output_tensor.token_loss.view(-1).float()
    flat_loss_mask = loss_mask.view(-1).float()
    lm_loss = torch.sum(losses * flat_loss_mask)
    num_tokens = flat_loss_mask.sum().clone().detach().to(torch.int)
    total_loss = lm_loss + output_tensor.aux_loss.float() * num_tokens.clamp_min(1).float()

    report = {"lm loss": torch.cat([lm_loss.detach().view(1), num_tokens.view(1)])}
    for name, value in output_tensor.metrics.items():
        report[name] = torch.cat(
            [
                value.detach().float().view(1),
                torch.ones_like(num_tokens, dtype=torch.int).view(1),
            ]
        )

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=total_loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )
    return total_loss, num_tokens, report


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    if return_schedule_plan:
        raise NotImplementedError("Concept-OLMo does not support overlap schedule planning yet")

    args = get_args()
    if args.pipeline_model_parallel_size > 1:
        raise NotImplementedError("Concept-OLMo Megatron prototype currently requires PP=1")
    if args.context_parallel_size > 1:
        raise NotImplementedError("Concept-OLMo Megatron prototype currently requires CP=1")
    if args.sequence_parallel:
        raise NotImplementedError("Concept-OLMo Megatron prototype currently requires sequence_parallel=False")

    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = None
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator, vp_stage
        )
    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )
    return output_tensor, partial(loss_func, loss_mask, model=model)


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, concept_olmo_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_concept_olmo_extra_args,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
