#!/usr/bin/env python3
"""Reconcile optimizer rounder counts with a live checkpoint header.

This intentionally reads only safetensors' eight-byte length prefix and JSON
header.  It never reads model payload bytes, imports a model, opens a
checkpoint through safetensors, or touches CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path


CHECKPOINT_MANIFEST_SHA256 = "81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf"
CHUNK_ELEMENTS = 1_048_576


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def header_only(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"short safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 0 < header_bytes < 16 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {header_bytes}")
        header = stream.read(header_bytes)
        if len(header) != header_bytes:
            raise ValueError(f"short safetensors header: {path}")
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "header_bytes": header_bytes,
        "prefix_hex": prefix.hex(),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "header": json.loads(header),
    }


def elements(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    optimizer = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/full_weight_optimizer.py"
    dispatch_path = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/expected-aurora-mix-manifest.json"
    config_path = root / "docs/campaign/work/lead/r2-full-weight-optimizer-v1/smoke-configs.json"
    trainer = root / "docs/campaign/work/lead/r2-native-cpt194-continuation-preparation-v1/source/experiments/training/full_weight_cpt_trainer.py"
    saved_precision = root / "docs/campaign/work/lead/r2-native-cpt194-continuation-preparation-v1/source/experiments/training/saved_precision.py"
    checkpoint_roots = [
        Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-CPT-prefix-extension-v3-from194/runtime/checkpoint-322"),
        Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-prefix-extension-native-v3-from194/full/checkpoint-322"),
    ]

    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dispatch_rows = {row["name"]: row for row in dispatch["ordered_rows"]}
    if len(dispatch_rows) != 381:
        raise ValueError("unexpected dispatch row count")
    chunk = int(config["common"]["stochastic_round_chunk_elements"])
    if chunk != CHUNK_ELEMENTS:
        raise ValueError(f"unexpected rounding chunk: {chunk}")

    observed = []
    header_summaries = []
    campaign_manifests = []
    for checkpoint_root in checkpoint_roots:
        campaign_path = checkpoint_root / "campaign-manifest.json"
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if sha256(campaign_path) != CHECKPOINT_MANIFEST_SHA256:
            raise ValueError(f"checkpoint manifest pin differs: {campaign_path}")
        model_record = campaign["files"]["model.safetensors"]
        model_path = checkpoint_root / "model.safetensors"
        if model_path.stat().st_size != model_record["bytes"]:
            raise ValueError(f"model size differs from manifest: {model_path}")
        header = header_only(model_path)
        rows = {name: value for name, value in header["header"].items() if name != "__metadata__"}
        if set(rows) != set(dispatch_rows):
            raise ValueError(f"checkpoint header names differ: {model_path}")
        header_summaries.append({key: value for key, value in header.items() if key != "header"})
        campaign_manifests.append({
            "path": str(campaign_path),
            "bytes": campaign_path.stat().st_size,
            "sha256": sha256(campaign_path),
            "step": campaign["step"],
            "model_safetensors": model_record,
        })
        for name, value in rows.items():
            row = dispatch_rows[name]
            if list(value["shape"]) != row["shape"]:
                raise ValueError(f"shape differs for {name}")
            dtype = value["dtype"]
            if dtype not in ("F32", "BF16"):
                raise ValueError(f"unexpected live dtype {dtype} for {name}")
            observed.append({
                "name": name,
                "dispatch": row["dispatch"],
                "shape": list(value["shape"]),
                "dtype": dtype,
                "elements": elements(value["shape"]),
            })

    # Both runtime and durable copies must expose the same header identity.
    header_identity = ("file_bytes", "header_bytes", "prefix_hex", "header_sha256")
    if any(header_summaries[0][key] != header_summaries[1][key] for key in header_identity):
        raise ValueError("runtime and durable checkpoint headers differ")

    # The two copies contain the same 381 names.  Count once, not twice.
    if len(observed) != 762:
        raise AssertionError(f"expected two header observations, got {len(observed)}")
    observed = observed[:381]
    dtype_counts = Counter(item["dtype"] for item in observed)
    dispatch_dtype_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"rows": 0, "elements": 0, "chunks": 0}))
    for item in observed:
        bucket = dispatch_dtype_counts[item["dispatch"]][item["dtype"]]
        bucket["rows"] += 1
        bucket["elements"] += item["elements"]
        bucket["chunks"] += math.ceil(item["elements"] / chunk)
    dispatch_dtype_counts = {
        dispatch_name: dict(dtype_map)
        for dispatch_name, dtype_map in dispatch_dtype_counts.items()
    }
    for dispatch_name in ("adamw", "muon", "aurora"):
        for dtype in ("F32", "BF16"):
            dispatch_dtype_counts.setdefault(dispatch_name, {}).setdefault(dtype, {"rows": 0, "elements": 0, "chunks": 0})

    f32 = [item for item in observed if item["dtype"] == "F32"]
    bf16 = [item for item in observed if item["dtype"] == "BF16"]
    hidden = [item for item in observed if item["dispatch"] in ("muon", "aurora")]
    side = [item for item in observed if item["dispatch"] == "adamw"]
    bf16_hidden = [item for item in hidden if item["dtype"] == "BF16"]
    bf16_side = [item for item in side if item["dtype"] == "BF16"]
    f32_side = [item for item in side if item["dtype"] == "F32"]
    bf16_chunks = sum(math.ceil(item["elements"] / chunk) for item in bf16)
    bf16_apply_calls = len(bf16_hidden) + sum(math.ceil(item["elements"] / chunk) for item in bf16_side)
    f32_apply_calls = sum(math.ceil(item["elements"] / chunk) for item in f32_side)

    result = {
        "schema": "sepalith.sft11.hybrid-optimizer-live-dtype-correction.v1",
        "status": "prepared_not_admitted",
        "read_policy": {
            "safetensors_payload_read": False,
            "safetensors_header_only": True,
            "model_imported": False,
            "checkpoint_loaded": False,
            "cuda_used": False,
            "training_launched": False,
        },
        "source_pins": {
            "optimizer": {"path": str(optimizer), "bytes": optimizer.stat().st_size, "sha256": sha256(optimizer)},
            "trainer": {"path": str(trainer), "bytes": trainer.stat().st_size, "sha256": sha256(trainer)},
            "saved_precision": {"path": str(saved_precision), "bytes": saved_precision.stat().st_size, "sha256": sha256(saved_precision)},
            "dispatch_manifest": {"path": str(dispatch_path), "bytes": dispatch_path.stat().st_size, "sha256": sha256(dispatch_path), "ordered_rows_sha256": dispatch["ordered_rows_sha256"]},
            "optimizer_config": {"path": str(config_path), "bytes": config_path.stat().st_size, "sha256": sha256(config_path)},
        },
        "checkpoint_header_pins": {
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "campaign_manifests": campaign_manifests,
            "model_headers": header_summaries,
        },
        "live_inventory": {
            "parameter_tensors": len(observed),
            "parameters": sum(item["elements"] for item in observed),
            "dtype_rows": dict(dtype_counts),
            "dtype_elements": {
                "F32": sum(item["elements"] for item in f32),
                "BF16": sum(item["elements"] for item in bf16),
            },
            "dispatch_dtype_counts": dispatch_dtype_counts,
            "fp32_restore_contract": {
                "fp32_tensors": len(f32),
                "fp32_elements": sum(item["elements"] for item in f32),
                "source_behavior": "saved_precision.restore_saved_fp32 selects F32 header entries, verifies each payload, changes parameter storage to float32 when needed, and copies exact values before optimizer construction.",
                "trainer_callsite": "full_weight_cpt_trainer calls restore_saved_fp32 before inventory_full_weight/build_full_weight_optimizer.",
            },
        },
        "corrected_step_census": {
            "configured_rounding_chunk_elements": chunk,
            "hidden_parameter_rows": len(hidden),
            "side_parameter_rows": len(side),
            "side_chunk_iterations_all_dtypes": sum(math.ceil(item["elements"] / chunk) for item in side),
            "bf16_stochastic_round_rows": len(bf16),
            "bf16_stochastic_round_chunks": bf16_chunks,
            "bf16_apply_fp32_update_invocations": bf16_apply_calls,
            "bf16_rand_like_invocations": bf16_chunks,
            "bf16_pos_neg_inf_scalar_tensor_allocations": bf16_apply_calls * 2,
            "f32_direct_update_rows": len(f32_side),
            "f32_direct_apply_fp32_update_invocations": f32_apply_calls,
            "f32_rand_like_invocations": 0,
            "f32_pos_neg_inf_scalar_tensor_allocations": 0,
            "all_apply_fp32_update_invocations": bf16_apply_calls + f32_apply_calls,
            "correction": "The prior v1 static manifest treated all 381 pre-restore rows as BF16. For the live restored checkpoint, only the 296 BF16 rows enter stochastic rounding; 85 F32 norms use the direct param.sub_ branch.",
        },
        "reconciliation": {
            "v1_static_bf16_rows": 381,
            "live_bf16_rows": len(bf16),
            "live_f32_rows": len(f32),
            "v1_stochastic_round_chunks": 2527,
            "corrected_stochastic_round_chunks": bf16_chunks,
            "v1_rand_like_invocations": 2527,
            "corrected_rand_like_invocations": bf16_chunks,
            "v1_pos_neg_inf_scalar_tensor_allocations": 1778,
            "corrected_pos_neg_inf_scalar_tensor_allocations": bf16_apply_calls * 2,
            "v1_status": "superseded_for_live_native_dtype accounting; frozen v1 packet remains unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
