#!/usr/bin/env python3
"""Metadata-only audit of the stopped SFT step-1000 theta0 F16 export.

The GGUFReader is used for headers, metadata, tensor names, shapes, types, byte
counts, and offsets.  No parent-model tensor buffer is read.  The F16 output is
hashed in 1 MiB blocks and asks Linux to drop each completed read range.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from gguf import GGUFReader

ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models")
F16 = ROOT / "SFT-primary-step1000-runtime-gguf/model-F16.gguf"
OLD_F16 = ROOT / "SFT-primary-step500-runtime-gguf/model-F16.gguf"
THETA0 = ROOT / "SFT-primary-step1000-theta0"
WORK = Path(__file__).parent
EXPECTED_F16_BYTES = 5039006496
EXPECTED_F16_SHA256 = "ee7ba2fd7e7b6446d74224e1e1f9e4724fbcfa54e56e60c4b623981d1e4f45ed"
EXPECTED_OLD_F16_SHA256 = "50f523af997f2f77dcbd187adde8a36dbc41529932703fd016e506b172761725"
EXPECTED_PARENT_MANIFEST_SHA256 = "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12"
EXPECTED_PARENT_WEIGHTS_SHA256 = "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d"
EXPECTED_CONFIG_SHA256 = "f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991"
EXPECTED_GENERATION_CONFIG_SHA256 = "7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269"
EXPECTED_TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"
EXPECTED_NATIVE_EOG = [1, 130073]


def sha256_file(path: Path, drop_cache: bool = False) -> tuple[int, str, float]:
    import time
    fd = os.open(path, os.O_RDONLY)
    digest = hashlib.sha256()
    total = 0
    block = 1 << 20
    started = time.monotonic()
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", 4)
    try:
        while True:
            chunk = os.read(fd, block)
            if not chunk:
                break
            offset = total
            total += len(chunk)
            digest.update(chunk)
            if drop_cache and advise is not None:
                try:
                    advise(fd, offset, len(chunk), dontneed)
                except OSError:
                    pass
    finally:
        os.close(fd)
    return total, digest.hexdigest(), time.monotonic() - started


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def tensor_inventory(reader: GGUFReader) -> list[dict[str, Any]]:
    return [
        {
            "name": tensor.name,
            "shape": [int(value) for value in tensor.shape],
            "tensor_type": tensor.tensor_type.name,
            "n_elements": int(tensor.n_elements),
            "n_bytes": int(tensor.n_bytes),
            "data_offset": int(tensor.data_offset),
        }
        for tensor in reader.tensors
    ]


def metadata_summary(reader: GGUFReader) -> dict[str, Any]:
    return {name: field.contents() for name, field in reader.fields.items()}


def scalar_or_shape(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "sha256": canonical_hash(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value)}
    return {"type": type(value).__name__}


def expected_names_shapes(config: dict[str, Any]) -> list[tuple[str, list[int]]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads_kv = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    layers = int(config["num_hidden_layers"])
    kv = heads_kv * head_dim
    names: list[tuple[str, list[int]]] = [("output.weight", [hidden, int(config["vocab_size"])])]
    names.append(("token_embd.weight", [hidden, int(config["vocab_size"])]))
    for layer in range(layers):
        prefix = f"blk.{layer}"
        names.extend([
            (f"{prefix}.attn_norm.weight", [hidden]),
            (f"{prefix}.attn_q.weight", [hidden, hidden]),
            (f"{prefix}.attn_k.weight", [hidden, kv]),
            (f"{prefix}.attn_v.weight", [hidden, kv]),
            (f"{prefix}.attn_output.weight", [hidden, hidden]),
            (f"{prefix}.ffn_norm.weight", [hidden]),
            (f"{prefix}.ffn_gate.weight", [hidden, intermediate]),
            (f"{prefix}.ffn_down.weight", [intermediate, hidden]),
            (f"{prefix}.ffn_up.weight", [hidden, intermediate]),
        ])
    names.append(("output_norm.weight", [hidden]))
    return names


def parent_token_ids(tokenizer: dict[str, Any]) -> dict[int, str]:
    values = {int(token_id): str(piece) for piece, token_id in tokenizer["model"]["vocab"].items()}
    for item in tokenizer.get("added_tokens", []):
        values[int(item["id"])] = str(item["content"])
    return values


def audit() -> dict[str, Any]:
    terminal_path = F16.parent / "command-0-terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    config = json.loads((THETA0 / "config.json").read_text(encoding="utf-8"))
    generation_config = json.loads((THETA0 / "generation_config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((THETA0 / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((THETA0 / "tokenizer_config.json").read_text(encoding="utf-8"))
    parent_manifest = json.loads((THETA0 / "parent-manifest.json").read_text(encoding="utf-8"))

    f16_bytes, f16_sha, f16_seconds = sha256_file(F16, drop_cache=True)
    # Hashing the old artifact is unnecessary; its accepted hash is recorded from
    # the existing quantization preparation receipt.  GGUFReader reads metadata
    # and maps tensor ranges, but this code never indexes tensor.data.
    new_reader = GGUFReader(F16)
    old_reader = GGUFReader(OLD_F16)
    new_meta = metadata_summary(new_reader)
    old_meta = metadata_summary(old_reader)
    new_inventory = tensor_inventory(new_reader)
    old_inventory = tensor_inventory(old_reader)

    sorted_offsets = sorted((item["data_offset"], item["n_bytes"], item["name"]) for item in new_inventory)
    overlaps = []
    gaps = []
    for current, following in zip(sorted_offsets, sorted_offsets[1:]):
        end = current[0] + current[1]
        if end > following[0]:
            overlaps.append({"name": current[2], "end": end, "next_offset": following[0]})
        if end < following[0]:
            gaps.append(following[0] - end)
    data_start = min(item["data_offset"] for item in new_inventory)
    data_end = max(item["data_offset"] + item["n_bytes"] for item in new_inventory)

    new_names_shapes = [(item["name"], item["shape"]) for item in new_inventory]
    old_names_shapes = [(item["name"], item["shape"]) for item in old_inventory]
    expected_shapes = expected_names_shapes(config)
    new_types = [item["tensor_type"] for item in new_inventory]
    old_types = [item["tensor_type"] for item in old_inventory]
    new_tokens = new_meta["tokenizer.ggml.tokens"]
    parent_ids = parent_token_ids(tokenizer)
    token_mismatches = [
        {"id": index, "gguf": new_tokens[index] if index < len(new_tokens) else None, "parent": parent_ids.get(index)}
        for index in range(max(len(new_tokens), (max(parent_ids) + 1)))
        if (new_tokens[index] if index < len(new_tokens) else None) != parent_ids.get(index)
    ]
    parent_merges = [" ".join(pair) if isinstance(pair, list) else str(pair) for pair in tokenizer["model"]["merges"]]
    gguf_merges = [str(value) for value in new_meta["tokenizer.ggml.merges"]]

    metadata_differences = []
    for key in sorted(set(new_meta) | set(old_meta)):
        if key not in new_meta or key not in old_meta:
            metadata_differences.append({"key": key, "kind": "missing_in_other"})
        elif new_meta[key] != old_meta[key]:
            metadata_differences.append({"key": key, "step1000": scalar_or_shape(new_meta[key]), "step500": scalar_or_shape(old_meta[key])})

    config_checks = {
        "architecture": new_meta.get("general.architecture") == config["model_type"],
        "vocab_size": new_meta.get("llama.vocab_size") == config["vocab_size"],
        "context_length": new_meta.get("llama.context_length") == config["max_position_embeddings"],
        "embedding_length": new_meta.get("llama.embedding_length") == config["hidden_size"],
        "block_count": new_meta.get("llama.block_count") == config["num_hidden_layers"],
        "head_count": new_meta.get("llama.attention.head_count") == config["num_attention_heads"],
        "head_count_kv": new_meta.get("llama.attention.head_count_kv") == config["num_key_value_heads"],
        "feed_forward_length": new_meta.get("llama.feed_forward_length") == config["intermediate_size"],
        "rope_dimension": new_meta.get("llama.rope.dimension_count") == config["head_dim"],
        "rope_theta": new_meta.get("llama.rope.freq_base") == config["rope_parameters"]["rope_theta"],
        "bos_id": new_meta.get("tokenizer.ggml.bos_token_id") == config["bos_token_id"],
        "pad_id": new_meta.get("tokenizer.ggml.padding_token_id") == config["pad_token_id"],
        "add_bos": new_meta.get("tokenizer.ggml.add_bos_token") == tokenizer_config["add_bos_token"],
        "add_eos": new_meta.get("tokenizer.ggml.add_eos_token") == tokenizer_config["add_eos_token"],
    }
    # The native GGUF field is scalar EOS=1. The HF parent contract retains
    # native EOG 130073 as a separate generation stop ID.
    eos_contract = {
        "gguf_eos_id": new_meta.get("tokenizer.ggml.eos_token_id"),
        "hf_config_eos_ids": config.get("eos_token_id"),
        "generation_config_eos_ids": generation_config.get("eos_token_id"),
        "native_eog_ids": EXPECTED_NATIVE_EOG,
        "eog_token_present_in_gguf_vocab": len(new_tokens) > 130073 and new_tokens[130073] == "<|im_end|>",
        "scalar_metadata_matches_canonical_only": new_meta.get("tokenizer.ggml.eos_token_id") == 1,
        "runtime_eog_contract_requires_verification": new_meta.get("tokenizer.ggml.eos_token_id") != EXPECTED_NATIVE_EOG,
    }

    parent_files = {
        "parent_manifest.json": (THETA0 / "parent-manifest.json", EXPECTED_PARENT_MANIFEST_SHA256),
        "config.json": (THETA0 / "config.json", EXPECTED_CONFIG_SHA256),
        "generation_config.json": (THETA0 / "generation_config.json", EXPECTED_GENERATION_CONFIG_SHA256),
        "tokenizer.json": (THETA0 / "tokenizer.json", EXPECTED_TOKENIZER_JSON_SHA256),
        "tokenizer_config.json": (THETA0 / "tokenizer_config.json", EXPECTED_TOKENIZER_CONFIG_SHA256),
    }
    parent_hashes = {}
    for name, (path, expected) in parent_files.items():
        size, digest, _ = sha256_file(path)
        parent_hashes[name] = {"bytes": size, "sha256": digest, "expected_sha256": expected, "matches": digest == expected}
    weight_entry = parent_manifest["weight_inventory"][0]
    parent_weight_path = THETA0 / weight_entry["path"]

    checks = {
        "terminal_exit_zero": terminal.get("exit_code") == 0,
        "terminal_output_size": terminal.get("seconds", 0) > 0 and f16_bytes == EXPECTED_F16_BYTES,
        "f16_sha256": f16_sha == EXPECTED_F16_SHA256,
        "gguf_version_3": new_meta.get("GGUF.version") == 3,
        "header_tensor_count_matches_reader": new_meta.get("GGUF.tensor_count") == len(new_inventory),
        "tensor_offsets_nonoverlap": not overlaps,
        "tensor_offsets_within_file": data_start >= 0 and data_end <= f16_bytes,
        "tensor_data_reaches_exact_file_end": data_end == f16_bytes,
        "derived_architecture_inventory_matches": sorted(new_names_shapes) == sorted(expected_shapes),
        "old_step500_names_shapes_match": new_names_shapes == old_names_shapes,
        "old_step500_tensor_types_match": new_types == old_types,
        "parent_token_ids_exact": not token_mismatches,
        "parent_merges_exact_after_gguf_join_normalization": parent_merges == gguf_merges,
        "chat_template_exact": (THETA0 / "chat_template.jinja").read_text(encoding="utf-8") == new_meta.get("tokenizer.chat_template"),
        "parent_metadata_hashes": all(item["matches"] for item in parent_hashes.values()),
        "parent_weight_inventory_stat": parent_weight_path.is_file() and parent_weight_path.stat().st_size == weight_entry["bytes"],
        "config_architecture_mapping": all(config_checks.values()),
    }
    report = {
        "schema_version": "sepalith.run01.theta0-f16-metadata-review.v1",
        "status": "pass_metadata_extent_runtime_stop_gate" if all(checks.values()) else "fail_metadata_extent",
        "scope": {
            "cuda_visible_devices": "",
            "model_loaded": False,
            "parent_tensor_values_read": False,
            "gguf_tensor_values_read": False,
            "hash_block_bytes": 1048576,
            "posix_fadvise_dontneed": hasattr(os, "posix_fadvise"),
        },
        "export": {
            "path": str(F16), "bytes": f16_bytes, "sha256": f16_sha,
            "expected_bytes": EXPECTED_F16_BYTES, "expected_sha256": EXPECTED_F16_SHA256,
            "hash_seconds": round(f16_seconds, 3), "terminal": terminal,
        },
        "parent": {
            "path": str(THETA0), "manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
            "manifest_status": parent_manifest.get("status"), "kind": parent_manifest.get("kind"),
            "base_model_revision": parent_manifest.get("base_model_revision"),
            "merged_weights_sha256_declared": parent_manifest.get("merged_weights_sha256"),
            "weight_inventory": {"path": weight_entry["path"], "bytes": weight_entry["bytes"], "sha256_declared": weight_entry["sha256"]},
            "files": parent_hashes,
            "config": {key: config[key] for key in ("architectures", "model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings", "eos_token_id", "bos_token_id", "pad_token_id", "dtype")},
            "generation": {key: generation_config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id", "do_sample", "temperature", "top_p")},
            "tokenizer": {"vocab_size": len(parent_ids), "merge_count": len(parent_merges), "tokenizer_config_add_bos": tokenizer_config.get("add_bos_token"), "tokenizer_config_add_eos": tokenizer_config.get("add_eos_token"), "native_eog_ids": EXPECTED_NATIVE_EOG},
        },
        "gguf": {
            "fields": len(new_meta), "field_keys": sorted(new_meta), "gguf_version": new_meta.get("GGUF.version"), "header_tensor_count": new_meta.get("GGUF.tensor_count"),
            "tensor_count": len(new_inventory), "tensor_type_counts": dict(sorted(Counter(item["tensor_type"] for item in new_inventory).items())),
            "total_elements": sum(item["n_elements"] for item in new_inventory), "total_tensor_bytes": sum(item["n_bytes"] for item in new_inventory),
            "inventory_sha256": canonical_hash([{key: item[key] for key in ("name", "shape", "tensor_type", "n_elements", "n_bytes")} for item in new_inventory]),
            "names_shapes_sha256": canonical_hash([(item["name"], item["shape"]) for item in new_inventory]),
            "data_extent": {"first_offset": data_start, "last_end": data_end, "overlaps": overlaps, "gap_count": len(gaps), "gap_bytes": sum(gaps)},
            "derived_expected_tensor_count": len(expected_shapes), "derived_expected_names_shapes_sha256": canonical_hash(expected_shapes),
            "config_mapping": config_checks,
            "tokenizer": {"token_count": len(new_tokens), "token_id_mismatches": len(token_mismatches), "merge_count": len(gguf_merges), "merges_exact": parent_merges == gguf_merges, "chat_template_exact": checks["chat_template_exact"], "bos_id": new_meta.get("tokenizer.ggml.bos_token_id"), "eos_id": new_meta.get("tokenizer.ggml.eos_token_id"), "padding_id": new_meta.get("tokenizer.ggml.padding_token_id")},
            "eos_contract": eos_contract,
            "step500_comparison": {"old_path": str(OLD_F16), "old_sha256_declared": EXPECTED_OLD_F16_SHA256, "new_names_shapes_exact": checks["old_step500_names_shapes_match"], "new_types_exact": checks["old_step500_tensor_types_match"], "metadata_differences": metadata_differences},
        },
        "checks": checks,
        "limits": [
            "Parent model.safetensors is bound by the accepted parent manifest and stat-checked only; no parent tensor bytes were read or compared.",
            "GGUF tensor values were not traversed; names, shapes, types, byte extents, and the complete F16 file hash were checked.",
            "The GGUF scalar EOS field is 1 while the HF config/generation contract is [1, 130073]. The 130073 token is present as <|im_end|>; native runtime EOG/stop handling requires a separate verification gate. This audit does not prescribe an API change.",
            "The Q8 companion is partial and not accepted; no Q6 candidate was reviewed. No model load, inference, quantized quality, or semantic evaluation was performed.",
        ],
    }
    output = WORK / "theta0-f16-metadata-review.json"
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "f16_bytes": f16_bytes, "f16_sha256": f16_sha, "tensor_count": len(new_inventory), "checks_failed": [key for key, value in checks.items() if not value], "eos_contract": eos_contract}, sort_keys=True))
    return report


if __name__ == "__main__":
    audit()
