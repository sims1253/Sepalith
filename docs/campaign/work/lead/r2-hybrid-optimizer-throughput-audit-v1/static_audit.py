#!/usr/bin/env python3
"""Static, model-free audit of the frozen full-weight optimizer.

The script reads only source and small JSON manifests/receipts.  It does not
import the model, open a checkpoint, allocate CUDA tensors, or run training.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def call_name(node: ast.Call) -> str | None:
    value = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node  # type: ignore[return-value]
    raise KeyError(name)


def function_stats(node: ast.FunctionDef) -> dict:
    calls: dict[str, int] = {}
    methods: dict[str, int] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = call_name(child)
            if name:
                calls[name] = calls.get(name, 0) + 1
            if isinstance(child.func, ast.Attribute):
                methods[child.func.attr] = methods.get(child.func.attr, 0) + 1
    return {
        "lineno": node.lineno,
        "end_lineno": node.end_lineno,
        "for_loops": sum(isinstance(x, ast.For) for x in ast.walk(node)),
        "matmul_operators": sum(
            isinstance(x, ast.BinOp) and isinstance(x.op, ast.MatMult)
            for x in ast.walk(node)
        ),
        "torch_calls": dict(sorted(calls.items())),
        "method_calls": dict(sorted(methods.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    source = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/full_weight_optimizer.py"
    expected = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/expected-aurora-mix-manifest.json"
    configs = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/smoke-configs.json"
    timing = root / "docs/campaign/receipts/SFT-11-native-update-component-timing-root.json"
    closure = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/source-closure.json"

    source_text = source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source))
    manifest = json.loads(expected.read_text(encoding="utf-8"))
    config = json.loads(configs.read_text(encoding="utf-8"))
    timing_receipt = json.loads(timing.read_text(encoding="utf-8"))
    closure_doc = json.loads(closure.read_text(encoding="utf-8"))
    rows = manifest["ordered_rows"]
    chunk = int(config["common"]["stochastic_round_chunk_elements"])
    counts = manifest["counts"]
    by_dispatch = {}
    for kind in ("adamw", "muon", "aurora"):
        selected = [row for row in rows if row["dispatch"] == kind]
        by_dispatch[kind] = {
            "rows": len(selected),
            "parameters": sum(row["parameters"] for row in selected),
            "chunks_at_configured_rounding_chunk": sum(
                math.ceil(row["parameters"] / chunk) for row in selected
            ),
            "dtypes": sorted({row["dtype"] for row in selected}),
        }
    hidden_rows = counts["muon"] + counts["aurora"]
    side_chunks = by_dispatch["adamw"]["chunks_at_configured_rounding_chunk"]
    all_chunks = sum(item["chunks_at_configured_rounding_chunk"] for item in by_dispatch.values())
    apply_calls = hidden_rows + side_chunks
    zero_inf_allocations = apply_calls * 2
    ns = function_stats(function_node(tree, "zeropower_fp32"))
    aurora = function_stats(function_node(tree, "aurora_fp32"))
    rounding = function_stats(function_node(tree, "apply_fp32_update_"))
    hidden_step = function_stats(function_node(tree, "_step_hidden"))
    adam_step = function_stats(function_node(tree, "_step_adamw"))

    compile_patterns = [
        r"torch\.compile",
        r"torch\._dynamo",
        r"foreach",
        r"multi_tensor",
        r"fused",
        r"capturable",
    ]
    compile_hits = {
        pattern: re.findall(pattern, source_text, flags=re.IGNORECASE)
        for pattern in compile_patterns
    }
    compile_hits = {key: len(value) for key, value in compile_hits.items()}

    result = {
        "schema": "sepalith.sft11.hybrid-optimizer-throughput-static-audit.v1",
        "scope": {
            "model_free": True,
            "cuda_used": False,
            "checkpoint_loaded": False,
            "training_launched": False,
            "source_edited": False,
        },
        "frozen_inputs": {
            "optimizer_source": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            },
            "dispatch_manifest": {
                "path": str(expected),
                "bytes": expected.stat().st_size,
                "sha256": sha256(expected),
                "ordered_rows_sha256": manifest["ordered_rows_sha256"],
                "parameter_objects": manifest["parameter_objects"],
                "parameters": manifest["parameters"],
                "counts": counts,
                "dtypes": sorted({row["dtype"] for row in rows}),
            },
            "optimizer_config": {
                "path": str(configs),
                "sha256": sha256(configs),
                "common": config["common"],
            },
            "source_closure": {
                "path": str(closure),
                "sha256": sha256(closure),
                "optimizer_source_sha256": next(
                    item["sha256"] for item in closure_doc["files"]
                    if item["path"].endswith("full_weight_optimizer.py")
                ),
            },
            "timing_receipt": {
                "path": str(timing),
                "sha256": sha256(timing),
                "first_step": timing_receipt["first_step"],
                "last_step": timing_receipt["last_step"],
                "completed_updates": timing_receipt["completed_updates"],
                "mean_update_interval_seconds": timing_receipt["mean_update_interval_seconds"],
                "mean_pre_optimizer_to_log_seconds": timing_receipt["mean_pre_optimizer_to_log_seconds"],
                "median_pre_optimizer_to_log_seconds": timing_receipt["median_pre_optimizer_to_log_seconds"],
                "interpretation": timing_receipt["interpretation"],
            },
        },
        "dispatch_cost_census": by_dispatch,
        "static_step_cost": {
            "configured_ns_steps": config["common"]["ns_steps"],
            "configured_aurora_K": config["common"]["aurora_K"],
            "matmul_operators_per_newton_schulz_iteration": ns["matmul_operators"],
            "muon_matmul_operator_dispatches_per_step": counts["muon"] * config["common"]["ns_steps"] * ns["matmul_operators"],
            "aurora_matmul_operator_dispatches_per_step": counts["aurora"] * config["common"]["aurora_K"] * config["common"]["ns_steps"] * ns["matmul_operators"],
            "total_custom_matmul_operator_dispatches_per_step": (
                counts["muon"] * config["common"]["ns_steps"] * ns["matmul_operators"]
                + counts["aurora"] * config["common"]["aurora_K"] * config["common"]["ns_steps"] * ns["matmul_operators"]
            ),
            "hidden_parameter_python_iterations": hidden_rows,
            "side_parameter_python_iterations": counts["adamw"],
            "side_chunk_python_iterations": side_chunks,
            "apply_fp32_update_invocations_per_step_for_bf16_manifest": apply_calls,
            "stochastic_round_chunks_per_step_for_bf16_manifest": all_chunks,
            "rand_like_invocations_per_step_for_bf16_manifest": all_chunks,
            "pos_neg_inf_scalar_tensor_allocations_per_step_for_bf16_manifest": zero_inf_allocations,
            "note": "Counts are source/manifest arithmetic, not isolated kernel timing.",
        },
        "ast_function_stats": {
            "zeropower_fp32": ns,
            "aurora_fp32": aurora,
            "apply_fp32_update_": rounding,
            "_step_hidden": hidden_step,
            "_step_adamw": adam_step,
        },
        "existing_fusion_hooks_in_frozen_optimizer_source": compile_hits,
        "source_semantics_observed": [
            "Muon and Aurora matrix work is a Python loop over parameters; each Newton-Schulz iteration creates gram, gram@gram, pointwise combination, and final matrix-product temporaries.",
            "Aurora adds K row-norm/damping/row-scaling rounds before calling the same Newton-Schulz routine.",
            "AdamW updates each parameter and then each configured flat chunk serially; it cannot use a whole embedding/lm_head temporary under the current bounded-memory contract.",
            "BF16 stochastic rounding uses the default torch RNG once per element per chunk and creates nearest/adjacent/probability/rounded tensors; RNG state is part of exact resume identity.",
            "FP32 parameters bypass stochastic rounding and subtract in place; the large native manifest is entirely BF16.",
            "Optimizer state restoration deep-copies and explicitly preserves FP32 state; this is checkpoint-path work, not an ordinary step hot path.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
