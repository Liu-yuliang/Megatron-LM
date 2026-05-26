#!/usr/bin/env python3
"""Lightweight config and parameter-count checks for public Megatron runs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def padded_vocab_size(vocab_size: int, make_divisible_by: int) -> int:
    if make_divisible_by <= 1:
        return vocab_size
    remainder = vocab_size % make_divisible_by
    return vocab_size if remainder == 0 else vocab_size + make_divisible_by - remainder


def dense_gpt_param_count(
    *,
    vocab_size: int,
    hidden_size: int,
    ffn_hidden_size: int,
    num_layers: int,
    tie_word_embeddings: bool,
    attention_bias: bool,
    mlp_bias: bool,
    gated_mlp: bool,
    qk_norm: bool,
    post_norm: bool,
    norm: str,
) -> int:
    attn_weights = 4 * hidden_size * hidden_size
    attn_biases = 4 * hidden_size if attention_bias else 0
    mlp_weights = (3 if gated_mlp else 2) * hidden_size * ffn_hidden_size
    mlp_biases = ffn_hidden_size + hidden_size if mlp_bias else 0
    norm_params_per_norm = hidden_size if norm == "rmsnorm" else 2 * hidden_size
    qk_norm_params = 2 * hidden_size if qk_norm else 0
    post_norm_params = 2 * norm_params_per_norm if post_norm else 0
    pre_norm_params = 0 if post_norm else 2 * norm_params_per_norm
    per_layer = (
        attn_weights
        + attn_biases
        + mlp_weights
        + mlp_biases
        + qk_norm_params
        + post_norm_params
        + pre_norm_params
    )
    embeddings = vocab_size * hidden_size
    output = 0 if tie_word_embeddings else vocab_size * hidden_size
    final_norm = norm_params_per_norm
    return num_layers * per_layer + embeddings + output + final_norm


def parse_manifest(path: Path) -> dict:
    data: dict[str, str] = {}
    if not path or not path.exists():
        return data
    arg_prefixes = (
        "model_args=",
        "train_args=",
        "parallel_args=",
        "data_args=",
        "logging_args=",
        "checkpoint_args=",
    )
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line and not line.startswith(arg_prefixes):
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
        if line.startswith(arg_prefixes):
            tokens = []
            for token in shlex.split(line):
                if "=" in token:
                    _, value = token.split("=", 1)
                    tokens.append(value)
            for idx, token in enumerate(tokens):
                if not token.startswith("--"):
                    continue
                name = token[2:]
                next_value = tokens[idx + 1] if idx + 1 < len(tokens) else None
                if next_value is None or next_value.startswith("--"):
                    data[name] = "true"
                else:
                    data[name] = next_value
            data[line.split("=", 1)[0]] = " ".join(tokens)
    text = path.read_text(errors="ignore")
    for name in [
        "num-layers",
        "hidden-size",
        "ffn-hidden-size",
        "num-attention-heads",
        "kv-channels",
        "seq-length",
        "max-position-embeddings",
        "rotary-base",
        "rotary-percent",
        "norm-epsilon",
        "make-vocab-size-divisible-by",
        "window-size",
        "window-attn-skip-freq",
        "normalization",
        "attention-backend",
        "spec",
    ]:
        match = re.search(rf"--{re.escape(name)}\\ ([^\\\n ]+)", text)
        if match:
            data[name] = match.group(1)
    return data


def safetensors_parameter_info(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"error": f"safetensors import failed: {exc!r}"}

    excluded_suffixes = (
        ".attention.bias",
        ".attention.masked_bias",
        ".attention.rotary_emb.inv_freq",
    )
    info: dict[str, object] = {
        "total_elements": 0,
        "excluded_buffer_elements": 0,
        "trainable_elements": 0,
        "files": [],
    }
    for file in sorted(path.glob("*.safetensors")):
        file_total = 0
        file_excluded = 0
        with safe_open(file, framework="numpy", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                count = 1
                for dim in shape:
                    count *= dim
                file_total += count
                if key.endswith(excluded_suffixes):
                    file_excluded += count
        info["files"].append(
            {"file": file.name, "total_elements": file_total, "excluded_buffer_elements": file_excluded}
        )
        info["total_elements"] += file_total
        info["excluded_buffer_elements"] += file_excluded
    info["trainable_elements"] = info["total_elements"] - info["excluded_buffer_elements"]
    return info


def values_match(observed: object, expected: object) -> bool:
    if observed is None:
        return True
    if isinstance(expected, float):
        try:
            return abs(float(str(observed)) - expected) <= max(abs(expected), 1.0) * 1e-12
        except ValueError:
            return False
    return str(observed) == str(expected)


def parse_megatron_log(path: Path | None) -> dict:
    result: dict[str, object] = {}
    if not path or not path.exists():
        return result
    text = path.read_text(errors="ignore")
    matches = re.findall(
        r"number of parameters on \(tensor, pipeline\) model parallel rank \((\d+), (\d+)\): (\d+)",
        text,
    )
    result["rank_parameter_counts"] = [
        {"tp_rank": int(tp), "pp_rank": int(pp), "parameters": int(count)}
        for tp, pp, count in matches
    ]
    if matches:
        # TP rank 0, PP rank 0 is the whole model for TP=PP=1. For other layouts this is a shard.
        result["first_reported_parameters"] = int(matches[0][2])
    return result


def validate_olmo(args: argparse.Namespace, cfg: dict, manifest: dict) -> tuple[dict, list[str]]:
    expected_layers = cfg["num_hidden_layers"]
    expected_hidden = cfg["hidden_size"]
    expected_ffn = cfg["intermediate_size"]
    expected_heads = cfg["num_attention_heads"]
    expected_kv_heads = cfg["num_key_value_heads"]
    expected_vocab = padded_vocab_size(cfg["vocab_size"], int(manifest.get("make-vocab-size-divisible-by", 1)))
    expected = {
        "num_layers": expected_layers,
        "hidden_size": expected_hidden,
        "ffn_hidden_size": expected_ffn,
        "num_attention_heads": expected_heads,
        "kv_channels": expected_hidden // expected_heads,
        "seq_length": cfg["max_position_embeddings"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "rotary_base": cfg["rope_theta"],
        "rotary_percent": 1.0,
        "norm_epsilon": cfg["rms_norm_eps"],
        "window_size": f"{cfg['sliding_window']},0",
        "window_attn_skip_freq": 4,
        "vocab_size": expected_vocab,
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "attention_bias": cfg["attention_bias"],
        "qk_norm": True,
        "post_norm": True,
        "norm": "rmsnorm",
    }
    expected["dense_parameters"] = dense_gpt_param_count(
        vocab_size=expected_vocab,
        hidden_size=expected_hidden,
        ffn_hidden_size=expected_ffn,
        num_layers=expected_layers,
        tie_word_embeddings=cfg["tie_word_embeddings"],
        attention_bias=cfg["attention_bias"],
        mlp_bias=False,
        gated_mlp=True,
        qk_norm=True,
        post_norm=True,
        norm="rmsnorm",
    )

    problems: list[str] = []
    checks = {
        "num-layers": expected_layers,
        "hidden-size": expected_hidden,
        "ffn-hidden-size": expected_ffn,
        "num-attention-heads": expected_heads,
        "kv-channels": expected_hidden // expected_heads,
        "seq-length": cfg["max_position_embeddings"],
        "max-position-embeddings": cfg["max_position_embeddings"],
        "rotary-base": cfg["rope_theta"],
        "rotary-percent": "1.0",
        "norm-epsilon": cfg["rms_norm_eps"],
        "window-size": f"{cfg['sliding_window']},0",
        "window-attn-skip-freq": 4,
        "qk-layernorm": "true",
    }
    for key, expected_value in checks.items():
        observed = manifest.get(key)
        if manifest and observed is None:
            problems.append(f"{key}: missing, expected {expected_value}")
            continue
        if not values_match(observed, expected_value):
            problems.append(f"{key}: observed {observed}, expected {expected_value}")
    if expected_kv_heads != expected_heads:
        problems.append("OLMo grouped-query attention is not represented by current script defaults")
    spec_text = f"{manifest.get('layer_spec', '')} {manifest.get('model_args', '')}"
    if (
        manifest
        and "olmo3_layer_specs" not in spec_text
        and "olmo3_layer_spec" not in spec_text
    ):
        problems.append(
            "Reference OLMo 3 uses full-hidden Q/K RMSNorm and post-attention/post-MLP RMSNorm; launch manifest does not show the OLMo 3 Megatron spec."
        )
    layer_types = cfg.get("layer_types", [])
    if layer_types and layer_types != ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 8:
        problems.append("OLMo layer_types are not the expected 3 sliding + 1 full repeating pattern")
    log_info = parse_megatron_log(args.log)
    reported = log_info.get("first_reported_parameters")
    if reported is not None and int(reported) != int(expected["dense_parameters"]):
        problems.append(
            f"megatron reported {reported} parameters, expected {expected['dense_parameters']}"
        )
    return expected, problems


def validate_pythia(args: argparse.Namespace, cfg: dict, manifest: dict) -> tuple[dict, list[str]]:
    expected_vocab = padded_vocab_size(cfg["vocab_size"], int(manifest.get("make-vocab-size-divisible-by", 1)))
    expected = {
        "num_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "ffn_hidden_size": cfg["intermediate_size"],
        "num_attention_heads": cfg["num_attention_heads"],
        "kv_channels": cfg["hidden_size"] // cfg["num_attention_heads"],
        "seq_length": cfg["max_position_embeddings"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "rotary_base": cfg["rotary_emb_base"],
        "rotary_percent": cfg["rotary_pct"],
        "norm_epsilon": cfg["layer_norm_eps"],
        "vocab_size": expected_vocab,
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "hidden_act": cfg["hidden_act"],
        "use_parallel_residual": cfg.get("use_parallel_residual"),
        "norm": "layernorm",
    }
    expected["dense_parameters_with_bias"] = dense_gpt_param_count(
        vocab_size=expected_vocab,
        hidden_size=cfg["hidden_size"],
        ffn_hidden_size=cfg["intermediate_size"],
        num_layers=cfg["num_hidden_layers"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        attention_bias=True,
        mlp_bias=True,
        gated_mlp=False,
        qk_norm=False,
        post_norm=False,
        norm="layernorm",
    )

    problems: list[str] = []
    checks = {
        "num-layers": cfg["num_hidden_layers"],
        "hidden-size": cfg["hidden_size"],
        "ffn-hidden-size": cfg["intermediate_size"],
        "num-attention-heads": cfg["num_attention_heads"],
        "kv-channels": cfg["hidden_size"] // cfg["num_attention_heads"],
        "seq-length": cfg["max_position_embeddings"],
        "max-position-embeddings": cfg["max_position_embeddings"],
        "rotary-base": cfg["rotary_emb_base"],
        "rotary-percent": cfg["rotary_pct"],
        "norm-epsilon": cfg["layer_norm_eps"],
        "normalization": "LayerNorm",
    }
    for key, expected_value in checks.items():
        observed = manifest.get(key)
        if manifest and observed is None:
            problems.append(f"{key}: missing, expected {expected_value}")
            continue
        if not values_match(observed, expected_value):
            problems.append(f"{key}: observed {observed}, expected {expected_value}")
    if cfg.get("use_parallel_residual") is True:
        spec_text = f"{manifest.get('layer_spec', '')} {manifest.get('model_args', '')}"
        if (
            manifest
            and "pythia_layer_specs" not in spec_text
            and "pythia_parallel_residual_layer_spec" not in spec_text
        ):
            problems.append(
                "Reference Pythia uses GPT-NeoX parallel residual; launch manifest does not show the Pythia parallel-residual Megatron spec."
            )
    log_info = parse_megatron_log(args.log)
    reported = log_info.get("first_reported_parameters")
    if reported is not None and int(reported) != int(expected["dense_parameters_with_bias"]):
        problems.append(
            f"megatron reported {reported} parameters, expected {expected['dense_parameters_with_bias']}"
        )
    return expected, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["olmo3", "pythia"], required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    ns = parser.parse_args()

    cfg = read_json(ns.hf_config)
    manifest = parse_manifest(ns.manifest) if ns.manifest else {}
    log_info = parse_megatron_log(ns.log)
    if ns.model == "olmo3":
        expected, problems = validate_olmo(ns, cfg, manifest)
    else:
        expected, problems = validate_pythia(ns, cfg, manifest)
    reference_info = safetensors_parameter_info(ns.reference_dir)
    reference_trainable = reference_info.get("trainable_elements")
    expected_params = expected.get("dense_parameters") or expected.get("dense_parameters_with_bias")
    if reference_trainable is not None and expected_params is not None:
        if int(reference_trainable) != int(expected_params):
            problems.append(
                f"reference safetensors trainable elements {reference_trainable}, expected {expected_params}"
            )

    report = {
        "model": ns.model,
        "hf_config": str(ns.hf_config),
        "manifest": str(ns.manifest) if ns.manifest else None,
        "log": str(ns.log) if ns.log else None,
        "expected": expected,
        "megatron_log": log_info,
        "reference_safetensors": reference_info,
        "problems": problems,
        "ok": not problems,
    }
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
