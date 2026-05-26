#!/usr/bin/env python3
"""Compare OLMo3 reference and TE fused full-hidden Q/K RMSNorm."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from megatron.core.models.gpt.olmo3_layer_specs import (
    Olmo3FullHiddenRMSNorm,
    Olmo3FullHiddenTERMSNorm,
)


@dataclass
class _Config:
    params_dtype: torch.dtype
    use_cpu_initialization: bool = False
    init_model_with_meta_device: bool = False
    sequence_parallel: bool = False
    layernorm_zero_centered_gamma: bool = False


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_f = a.detach().float()
    b_f = b.detach().float()
    diff = (a_f - b_f).abs()
    denom = torch.maximum(a_f.abs(), b_f.abs()).clamp_min(1e-8)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def run_case(seq: int, batch: int, hidden: int, eps: float, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")
    cfg = _Config(params_dtype=dtype)

    ref = Olmo3FullHiddenRMSNorm(config=cfg, hidden_size=hidden, eps=eps).to(device)
    fused = Olmo3FullHiddenTERMSNorm(config=cfg, hidden_size=hidden, eps=eps).to(device)
    with torch.no_grad():
        fused.weight.copy_(ref.weight)

    x = torch.randn(seq, batch, hidden, device=device, dtype=dtype)
    x_ref = x.detach().clone().requires_grad_(True)
    x_fused = x.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref)
    y_fused = fused(x_fused)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_fused.backward(grad)

    return {
        "shape": [seq, batch, hidden],
        "dtype": str(dtype).removeprefix("torch."),
        "eps": eps,
        "forward": _metrics(y_ref, y_fused),
        "input_grad": _metrics(x_ref.grad, x_fused.grad),
        "weight_grad": _metrics(ref.weight.grad, fused.weight.grad),
        "finite": bool(
            torch.isfinite(y_ref).all()
            and torch.isfinite(y_fused).all()
            and torch.isfinite(x_ref.grad).all()
            and torch.isfinite(x_fused.grad).all()
            and torch.isfinite(ref.weight.grad).all()
            and torch.isfinite(fused.weight.grad).all()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-max-abs", type=float, default=2.0e-2)
    parser.add_argument("--grad-max-abs", type=float, default=2.5e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TE fused RMSNorm precision check")

    result = run_case(
        seq=args.seq,
        batch=args.batch,
        hidden=args.hidden,
        eps=args.eps,
        dtype=torch.bfloat16,
        seed=args.seed,
    )
    result["thresholds"] = {
        "forward_max_abs": args.forward_max_abs,
        "grad_max_abs": args.grad_max_abs,
    }
    result["ok"] = bool(
        result["finite"]
        and result["forward"]["max_abs"] <= args.forward_max_abs
        and result["input_grad"]["max_abs"] <= args.grad_max_abs
        and result["weight_grad"]["max_abs"] <= args.grad_max_abs
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
