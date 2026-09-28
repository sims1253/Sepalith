#!/usr/bin/env python3
"""CPU-only full-weight MiniCPM5-2B memory and loader evidence audit.

This module deliberately does not import torch, Transformers, Unsloth, or load
model weights.  It reads the pinned config/tensor header and package source
files, then emits arithmetic that the lead can compare with a real CUDA smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path


BYTES_PER_GIB = 2**30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gib(n: int | float) -> float:
    return n / BYTES_PER_GIB


def gb(n: int | float) -> float:
    return n / 1_000_000_000


def estimate_bytes(params: int, hidden_matrix_params: int, side_params: int) -> dict:
    """Return persistent storage estimates, with assumptions stated in names."""

    # BF16 weights and BF16 gradients are two bytes each.  BF16 optimizer state
    # is what the CPU PyTorch probe observed; CUDA fused AdamW must still be
    # measured.  FP32 moments/master weights are conservative alternate cases.
    weights = params * 2
    grads = params * 2
    base = weights + grads
    adam_bf16_state = params * 4
    adam_fp32_state = params * 8
    fp32_master = params * 4

    # Paged 8-bit AdamW has two uint8 moments.  BNB's block metadata is an
    # estimate: two FP32 scales per block.  It is reported as a range because
    # block size is implementation dependent and must be checked in the smoke.
    paged_state = params * 2
    paged_metadata = {
        str(block): 2 * math.ceil(params / block) * 4 for block in (2048, 4096)
    }

    # Muon/Aurora owns one BF16 momentum for hidden 2-D matrices.  The side
    # group (embedding, lm_head, norms and any scalar parameters) owns AdamW's
    # two BF16 moments.  No parameter is allowed in both groups.
    muon_state = hidden_matrix_params * 2 + side_params * 4
    muon_side_fp32_state = hidden_matrix_params * 2 + side_params * 8
    # The currently prepared FullWeightCompositeOptimizer uses FP32 hidden
    # momentum and FP32 side moments for numerical stability.  This is a
    # separate, larger estimate than the BF16 vendored POC arithmetic above.
    muon_fp32_state = hidden_matrix_params * 4 + side_params * 8

    def row(n: int, state: int, extras: int = 0) -> dict:
        total = base + state + extras
        return {"bytes": total, "GB": gb(total), "GiB": gib(total)}

    return {
        "parameters": params,
        "hidden_matrix_parameters": hidden_matrix_params,
        "side_parameters": side_params,
        "weights_bf16": {"bytes": weights, "GB": gb(weights), "GiB": gib(weights)},
        "gradients_bf16": {"bytes": grads, "GB": gb(grads), "GiB": gib(grads)},
        "weights_plus_gradients": {"bytes": base, "GB": gb(base), "GiB": gib(base)},
        "adamw_bf16_moments": row(params, adam_bf16_state),
        "adamw_fp32_moments": row(params, adam_fp32_state),
        "adamw_fp32_moments_plus_master_weights": row(
            params, adam_fp32_state, fp32_master
        ),
        "paged_adamw_8bit_estimate": {
            "state_bytes": paged_state,
            "state_GB": gb(paged_state),
            "state_GiB": gib(paged_state),
            "block_metadata_bytes": paged_metadata,
            "totals": {
                str(block): {
                    "bytes": base + paged_state + metadata,
                    "GB": gb(base + paged_state + metadata),
                    "GiB": gib(base + paged_state + metadata),
                }
                for block, metadata in paged_metadata.items()
            },
            "assumption": "two uint8 moments plus two FP32 scales per block; verify BNB state layout on CUDA",
        },
        "muon_or_aurora_bf16_hidden_adamw_bf16_side": row(
            params, muon_state
        ),
        "muon_or_aurora_bf16_hidden_adamw_fp32_side": row(
            params, muon_side_fp32_state
        ),
        "muon_or_aurora_fp32_hidden_adamw_fp32_side": row(
            params, muon_fp32_state
        ),
        "optimizer_state_components": {
            "muon_or_aurora_hidden_momentum_bf16_bytes": hidden_matrix_params * 2,
            "muon_or_aurora_hidden_momentum_fp32_bytes": hidden_matrix_params * 4,
            "side_adamw_two_bf16_moments_bytes": side_params * 4,
            "side_adamw_two_fp32_moments_bytes": side_params * 8,
            "aurora_fp32_row_norm_is_transient": True,
        },
        "important_exclusions": [
            "Activation/workspace memory is not included in persistent arithmetic.",
            "BF16 CUDA AdamW state and BNB paged metadata require the lead smoke.",
            "An FP32 master copy is included only in the named conservative scenario.",
        ],
    }


def inspect_tensors(header: dict) -> dict:
    groups = {"all": 0, "hidden_matrix": 0, "embedding_or_head": 0, "norm_or_other": 0}
    tensor_counts = {key: 0 for key in groups}
    dtypes = set()
    shapes = {}
    for name, item in header.items():
        shape = item["shape"]
        count = math.prod(shape)
        dtypes.add(item["dtype"])
        groups["all"] += count
        tensor_counts["all"] += 1
        shapes[name] = shape
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            bucket = "embedding_or_head"
        elif name.startswith("model.layers.") and (".mlp." in name or ".self_attn." in name) and len(shape) == 2:
            bucket = "hidden_matrix"
        else:
            bucket = "norm_or_other"
        groups[bucket] += count
        tensor_counts[bucket] += 1
    return {
        "tensor_count": tensor_counts["all"],
        "dtype_values": sorted(dtypes),
        "parameter_counts": groups,
        "tensor_counts_by_group": tensor_counts,
        "shape_samples": {
            key: shapes[key]
            for key in (
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.input_layernorm.weight",
            )
            if key in shapes
        },
    }


def command_output(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (completed.stdout or "").strip()
    return text or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.model / "config.json"
    parent_manifest = args.model / "parent-manifest.preparation.json"
    config = json.loads(config_path.read_text())
    header = json.loads(args.base_header.read_text())
    inspected = inspect_tensors(header)
    counts = inspected["parameter_counts"]
    params = counts["all"]
    assert params == 2_516_756_480, params
    assert counts["hidden_matrix"] == 1_981_808_640, counts
    assert counts["embedding_or_head"] == 534_773_760, counts
    assert counts["norm_or_other"] == 174_080, counts
    assert inspected["dtype_values"] == ["BF16"], inspected["dtype_values"]

    source_root = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages")
    source_files = [
        source_root / "unsloth/models/loader.py",
        source_root / "unsloth/models/_utils.py",
        source_root / "unsloth/models/llama.py",
        source_root / "unsloth_zoo/loss_utils.py",
        Path("experiments/training/train_sft.py"),
    ]
    source_hashes = {
        str(path): sha256(path) for path in source_files if path.exists()
    }
    versions = {}
    for package in ("torch", "transformers", "unsloth", "trl", "accelerate", "bitsandbytes", "safetensors", "peft"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    report = {
        "schema_version": "sepalith.sft11.full_weight_memory_feasibility.v1",
        "status": "cpu_static_feasibility_ready_cuda_smoke_pending",
        "model": {
            "path": str(args.model),
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "parent_manifest_path": str(parent_manifest),
            "parent_manifest_sha256": sha256(parent_manifest) if parent_manifest.exists() else None,
            "config_selected": {
                key: config.get(key)
                for key in (
                    "architectures", "model_type", "hidden_size", "intermediate_size",
                    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "head_dim", "vocab_size", "max_position_embeddings", "torch_dtype",
                    "bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings",
                )
            },
        },
        "base_tensor_header": {
            "path": str(args.base_header),
            "sha256": sha256(args.base_header),
            "bytes": args.base_header.stat().st_size,
            "inspection": inspected,
        },
        "memory": estimate_bytes(params, counts["hidden_matrix"], params - counts["hidden_matrix"]),
        "activation_reference": {
            "checkpoint_boundary_b1_s4096_bf16_bytes": 42 * 4096 * 2048 * 2,
            "checkpoint_boundary_b1_s4096_bf16_GiB": gib(42 * 4096 * 2048 * 2),
            "full_logits_b1_s4096_vocab130560_bf16_bytes": 4096 * 130560 * 2,
            "full_logits_b1_s4096_vocab130560_bf16_GiB": gib(4096 * 130560 * 2),
            "interpretation": "Boundary and logits values are lower-bound/transient references; checkpointed attention/MLP workspaces and backward buffers require CUDA measurement.",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": versions,
            "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"]),
            "loader_path": "unsloth.FastLanguageModel.from_pretrained",
            "loader_arguments": {
                "model_name": "merged SFT-500 model path",
                "max_seq_length": 4096,
                "dtype": "torch.bfloat16",
                "load_in_4bit": False,
                "full_finetuning": True,
                "float32_mixed_precision": False,
                "fast_inference": False,
            },
            "source_hashes": source_hashes,
        },
        "method": {
            "persistent_memory": "BF16 weights + BF16 gradients + named optimizer state; decimal GB=10^9 and GiB=2^30",
            "hidden_matrix_rule": "all model.layers.*.mlp/self_attn 2-D weights; excludes model.embed_tokens, lm_head and norms",
            "side_rule": "embedding, lm_head, norms and any scalar parameters use AdamW side group",
            "no_cuda_model_load": True,
            "no_weight_payload_read": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "parameters": params, "memory": report["memory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
