# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import Any, List, Optional, Tuple

import torch
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_hybrid_cp_rank, StragglerDetector
from megatron.training import (
    get_args,
    get_tensorboard_writer,
    get_timers,
    get_wandb_writer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch.

    Packed sequence support (SFT / ``--sft`` flag):
        When ``args.sft`` is True, the dataset emits THD-format batches where
        multiple sequences are concatenated into a single flat token tensor.
        The batch includes ``cu_seqlens`` (cumulative sequence lengths, shape
        ``[1, S+1]``) and ``max_seqlen`` (shape ``[1]``) that describe the
        individual sequence boundaries.

        This function validates and squeezes those fields:
          - ``cu_seqlens``:  asserted to have shape ``[1, S+1]`` (micro-batch
            size must be 1 for packing), then squeezed to ``[S+1]``.
          - ``max_seqlen``:  asserted to be 1-D; kept as a tensor and passed
            to ``get_thd_batch_on_this_cp_rank`` which performs the final
            scalar conversion internally.

        Pipeline stage handling:
          - First/last PP stages: fetch the full batch (tokens + labels) and
            route through ``get_thd_batch_on_this_cp_rank`` to produce a
            ``PackedSeqParams`` object that carries ``cu_seqlens`` and
            ``max_seqlen`` to the attention kernel.
          - Middle PP stages: only ``cu_seqlens`` and ``max_seqlen`` are
            needed for attention masking; all other fields are returned as
            ``None`` with a ``PackedSeqParams`` built directly here.
          - MTP ranks (``mtp_on_this_rank``) also receive the full batch,
            regardless of pipeline stage.

        Difference from ``pretrain_hybrid.py``:
          - Return format: GPT returns a 6-tuple
            ``(tokens, labels, loss_mask, attention_mask, position_ids,
            packed_seq_params)`` where ``packed_seq_params`` is a
            ``PackedSeqParams`` dataclass.  Mamba returns 7 values via
            ``batch.values()`` with ``cu_seqlens`` and ``max_seqlen`` as
            separate dict entries (no ``PackedSeqParams`` wrapper).
          - Middle-stage return: GPT returns ``(None×5, PackedSeqParams)``;
            Mamba returns an ``empty_batch`` dict with ``cu_seqlens`` and
            ``max_seqlen`` set.
          - CP with packed sequences: GPT delegates to
            ``get_thd_batch_on_this_cp_rank`` (MCore utility); Mamba
            implements the ``tex.thd_get_partitioned_indices`` CP slicing
            inline and does not call that helper.
          - MTP: GPT passes ``mtp_on_this_rank`` to ``get_batch_on_this_tp_rank``
            and uses it to gate the early-return; Mamba has no MTP support.
          - ``max_seqlen`` conversion: Mamba converts to a Python int scalar
            before returning (``int(max_seqlen[0].item())``); GPT keeps it as
            a tensor and lets ``get_thd_batch_on_this_cp_rank`` convert it,
            except for the middle-stage ``PackedSeqParams`` where conversion
            is done inline.
    """
    args = get_args()
    config = core_transformer_config_from_args(args)
    # TODO: this is pretty hacky, find a better way
    is_packed_sequence = get_args().sft  # SFT always uses packed sequence
    if not is_first_or_last_pipeline_stage(vp_stage) and not is_packed_sequence and (
    (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_seqlens = cu_seqlens[0]
        assert max_seqlen.dim() == 1

    # For middle pipeline stages with packed sequences, only cu_seqlens and
    # max_seqlen are needed (for attention masking); skip the full batch.
    if not is_first_or_last_pipeline_stage(vp_stage) and is_packed_sequence:
        return None, None, None, None, None, PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=int(max_seqlen[0].item()),
            max_seqlen_kv=int(max_seqlen[0].item()),
            qkv_format='thd',
        )

    if cu_seqlens is None and local_cp_size is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else: # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size)

    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10
_EXIT_HIDDEN_RANK_LAST_LOGGED_STEP = None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _current_train_step(args) -> int:
    iteration = getattr(args, "curr_iteration", None)
    if iteration is None:
        iteration = getattr(args, "iteration", -1)
    return int(iteration) + 1


def _should_monitor_exit_hidden_rank(args) -> bool:
    interval = _env_int("HIDDEN_RANK_LOG_INTERVAL", 0)
    if interval <= 0:
        return False
    step = _current_train_step(args)
    return step > 0 and step % interval == 0


def _dist_rank_info() -> dict:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return {
            "rank": 0,
            "world_size": 1,
            "tp_rank": 0,
            "pp_rank": 0,
            "dp_rank": 0,
            "cp_rank": 0,
        }
    info = {
        "rank": torch.distributed.get_rank(),
        "world_size": torch.distributed.get_world_size(),
    }
    for key, getter in (
        ("tp_rank", parallel_state.get_tensor_model_parallel_rank),
        ("pp_rank", parallel_state.get_pipeline_model_parallel_rank),
        ("dp_rank", parallel_state.get_data_parallel_rank),
        ("cp_rank", parallel_state.get_context_parallel_rank),
    ):
        try:
            info[key] = getter()
        except Exception:
            info[key] = None
    return info


def _is_exit_hidden_rank_logger_rank() -> bool:
    rank_info = _dist_rank_info()
    return rank_info["rank"] == rank_info["world_size"] - 1


def _append_exit_hidden_rank_json(record: dict) -> None:
    path = os.environ.get("HIDDEN_RANK_LOG_PATH")
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _log_exit_hidden_rank_metrics(metrics: dict, step: int) -> None:
    writer = get_tensorboard_writer()
    if writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)
    wandb_writer = get_wandb_writer()
    if wandb_writer is not None:
        wandb_writer.log(metrics, step)


def _log_scalar_metrics(metrics: dict, step: int) -> None:
    writer = get_tensorboard_writer()
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
    wandb_writer = get_wandb_writer()
    if wandb_writer is not None:
        wandb_writer.log(metrics, step)


def _maybe_log_exit_hidden_rank(hidden_states: torch.Tensor) -> None:
    global _EXIT_HIDDEN_RANK_LAST_LOGGED_STEP
    args = get_args()
    if not _should_monitor_exit_hidden_rank(args):
        return
    step = _current_train_step(args)
    if _EXIT_HIDDEN_RANK_LAST_LOGGED_STEP == step:
        return
    if not _is_exit_hidden_rank_logger_rank():
        return
    _EXIT_HIDDEN_RANK_LAST_LOGGED_STEP = step

    max_tokens = max(1, _env_int("HIDDEN_RANK_LOG_TOKENS", 1024))
    rtol = max(0.0, _env_float("HIDDEN_RANK_RTOL", 1.0e-3))
    center = os.environ.get("HIDDEN_RANK_CENTER", "1") != "0"
    rank_info = _dist_rank_info()
    metrics = {}

    with torch.no_grad():
        hidden = hidden_states.detach()
        hidden_size = int(hidden.shape[-1])
        matrix = hidden.reshape(-1, hidden_size)
        local_tokens = int(matrix.shape[0])
        if local_tokens > max_tokens:
            indices = torch.linspace(
                0, local_tokens - 1, steps=max_tokens, device=matrix.device
            ).long()
            matrix = matrix.index_select(0, indices)
        matrix = matrix.float()

        finite = torch.isfinite(matrix)
        nan_count = int(torch.isnan(matrix).sum().item())
        inf_count = int(torch.isinf(matrix).sum().item())
        finite_rows = finite.all(dim=-1)
        finite_row_count = int(finite_rows.sum().item())
        sample_tokens = int(matrix.shape[0])

        metrics.update(
            {
                "exit_hidden/local_tokens": local_tokens,
                "exit_hidden/sample_tokens": sample_tokens,
                "exit_hidden/finite_sample_tokens": finite_row_count,
                "exit_hidden/hidden_size": hidden_size,
                "exit_hidden/nan_count": nan_count,
                "exit_hidden/inf_count": inf_count,
            }
        )

        if finite_row_count == 0:
            metrics["exit_hidden/numerical_rank"] = 0
            record = {"step": step, **rank_info, **metrics}
            _append_exit_hidden_rank_json(record)
            _log_exit_hidden_rank_metrics(metrics, step)
            return

        matrix = matrix[finite_rows]
        metrics.update(
            {
                "exit_hidden/rms": float(torch.sqrt(matrix.square().mean()).item()),
                "exit_hidden/mean_abs": float(matrix.abs().mean().item()),
                "exit_hidden/max_abs": float(matrix.abs().max().item()),
                "exit_hidden/token_norm_mean": float(torch.linalg.vector_norm(matrix, dim=-1).mean().item()),
            }
        )

        if center and matrix.shape[0] > 1:
            matrix = matrix - matrix.mean(dim=0, keepdim=True)

        try:
            singular_values = torch.linalg.svdvals(matrix)
            singular_values = singular_values[torch.isfinite(singular_values)]
            if singular_values.numel() == 0 or singular_values[0].item() <= 0.0:
                numeric_rank = 0
                effective_rank = 0.0
                stable_rank = 0.0
                top_sv = 0.0
                median_sv = 0.0
                min_sv = 0.0
            else:
                threshold = singular_values[0] * rtol
                numeric_rank = int((singular_values > threshold).sum().item())
                spectrum_sum = singular_values.sum()
                probabilities = singular_values / spectrum_sum.clamp_min(1.0e-30)
                entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum()
                effective_rank = float(torch.exp(entropy).item())
                stable_rank = float(
                    (singular_values.square().sum() / singular_values[0].square()).item()
                )
                top_sv = float(singular_values[0].item())
                median_sv = float(singular_values[singular_values.numel() // 2].item())
                min_sv = float(singular_values[-1].item())
            metrics.update(
                {
                    "exit_hidden/numerical_rank": numeric_rank,
                    "exit_hidden/effective_rank": effective_rank,
                    "exit_hidden/stable_rank": stable_rank,
                    "exit_hidden/top_singular_value": top_sv,
                    "exit_hidden/median_singular_value": median_sv,
                    "exit_hidden/min_singular_value": min_sv,
                    "exit_hidden/rank_rtol": rtol,
                    "exit_hidden/centered": int(center),
                }
            )
        except RuntimeError as exc:
            metrics["exit_hidden/rank_error"] = str(exc)[:200]

    record = {"step": step, **rank_info, **metrics}
    _append_exit_hidden_rank_json(record)
    _log_exit_hidden_rank_metrics(metrics, step)


def exit_hidden_rank_output_processor(
    hidden_states,
    output_layer,
    output_weight,
    labels,
    runtime_gather_output,
    compute_language_model_loss,
    scale_logits,
    **_kwargs,
):
    _maybe_log_exit_hidden_rank(hidden_states)
    logits, _ = output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )
    logits = scale_logits(logits)
    if labels is None:
        return logits.transpose(0, 1).contiguous()
    return compute_language_model_loss(labels, logits)


def olmo3_z_loss_output_processor(
    hidden_states,
    output_layer,
    output_weight,
    labels,
    runtime_gather_output,
    compute_language_model_loss,
    scale_logits,
    loss_mask=None,
    **_kwargs,
):
    _maybe_log_exit_hidden_rank(hidden_states)
    logits, _ = output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )
    logits = scale_logits(logits)
    if labels is None:
        return logits.transpose(0, 1).contiguous()

    z_loss_multiplier = _env_float("OLMO3_Z_LOSS_MULTIPLIER", 0.0)
    if z_loss_multiplier > 0.0:
        # OLMo applies dense LM z-loss: square(logsumexp(logits)).
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            gathered = differentiable_all_gather(logits, group=parallel_state.get_tensor_model_parallel_group())
            logits_for_z_loss = torch.cat(tuple(gathered), dim=-1)
        else:
            logits_for_z_loss = logits
        z_losses = torch.logsumexp(logits_for_z_loss, dim=-1).square() * z_loss_multiplier
        z_losses = z_losses.transpose(0, 1).contiguous()
    else:
        z_losses = None

    lm_losses = compute_language_model_loss(labels, logits)
    if z_losses is None:
        return lm_losses

    step = _current_train_step(get_args())
    if loss_mask is not None and _is_exit_hidden_rank_logger_rank():
        with torch.no_grad():
            mask = loss_mask.float()
            denom = mask.sum().clamp_min(1.0)
            _log_scalar_metrics(
                {"olmo3_z_loss": float((z_losses.detach() * mask).sum().item() / denom.item())},
                step,
            )
    return lm_losses + z_losses


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        else:
            output_processor = None
            if _should_monitor_exit_hidden_rank(args) or _env_float("OLMO3_Z_LOSS_MULTIPLIER", 0.0) > 0.0:
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

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size * args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)


    is_packed_sequence = False
    if args.sft:
        dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
