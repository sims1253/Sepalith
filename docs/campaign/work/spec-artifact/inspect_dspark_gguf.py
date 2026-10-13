#!/usr/bin/env python3
"""CPU-only, metadata/header audit for the pinned DSpark and campaign Q8 GGUFs.

The audit never reads tensor payloads. It records tensor names/shapes/types and
hashes the complete embedded tokenizer fields so HF-file hashes are not used as
an in-GGUF equivalence test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

GGUF_PY = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/gguf-py")
sys.path.insert(0, str(GGUF_PY))
from gguf import GGUFReader  # noqa: E402

DSPARK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-bf16/MiniCPM5-2.6B-DSpark.gguf")
TARGET = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step500-runtime-gguf/model-Q8_0.gguf")
WORK = Path(__file__).resolve().parent
OUTPUT = WORK / "dspark-gguf-audit.json"
EXPECTED_DSPARK_SHA = "57df08640f0534a1aac075d1c8bdacdb2b7e5815da6f4e5cfd39ecac3a3f0c26"
EXPECTED_DSPARK_BYTES = 652730240
EXPECTED_TARGET_SHA = "f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256"
EXPECTED_TARGET_BYTES = 2679710464
EXPECTED_DSPARK_TARGET_LAYERS = [2, 11, 21, 31, 40]
EXPECTED_BLOCK_FIELDS = {
    "dflash.block_count": 5,
    "dflash.block_size": 7,
    "dflash.embedding_length": 2048,
    "dflash.feed_forward_length": 6144,
    "dflash.attention.head_count": 16,
    "dflash.attention.head_count_kv": 2,
    "dflash.attention.key_length": 128,
    "dflash.attention.value_length": 128,
    "dflash.context_length": 131072,
    "dflash.rope.freq_base": 5000000.0,
    "tokenizer.ggml.mask_token_id": 75982,
}
TOKENIZER_KEYS = {
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_sep_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.chat_template",
}
TOKEN_ARRAY_KEYS = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
}
KNOWN_TOKENIZER_DIFFERENCES = {
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.mask_token_id",
}
EXPECTED_BLOCK_TENSORS = {
    "attn_norm.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_norm.weight",
    "attn_k_norm.weight",
    "attn_k.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_q.weight",
    "attn_v.weight",
}
EXPECTED_TOP_TENSORS = {
    "conf_proj.bias",
    "conf_proj.weight",
    "fc.weight",
    "enc.output_norm.weight",
    "markov_w1.weight",
    "markov_w2.weight",
    "output_norm.weight",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError(f"unsupported GGUF metadata value {type(value)!r}")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preview(value: Any) -> dict[str, Any]:
    value = jsonable(value)
    result: dict[str, Any] = {
        "value_type": type(value).__name__,
        "canonical_sha256": canonical_hash(value),
    }
    if isinstance(value, list):
        result["length"] = len(value)
        result["first"] = value[:4]
        result["last"] = value[-4:]
    elif isinstance(value, str):
        result["length"] = len(value)
    else:
        result["value"] = value
    return result


def read_fields(reader: GGUFReader) -> dict[str, Any]:
    return {name: jsonable(field.contents()) for name, field in reader.fields.items()}


def inspect_file(path: Path, expected_bytes: int, expected_sha: str) -> dict[str, Any]:
    reader = GGUFReader(str(path))
    fields = read_fields(reader)
    inventory = [
        {
            "name": tensor.name,
            "shape": [int(value) for value in tensor.shape],
            "tensor_type": tensor.tensor_type.name,
            "n_elements": int(tensor.n_elements),
            "n_bytes": int(tensor.n_bytes),
        }
        for tensor in reader.tensors
    ]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha,
        "gguf_header": {
            "version": int(fields["GGUF.version"]),
            "kv_count": int(fields["GGUF.kv_count"]),
            "tensor_count": int(fields["GGUF.tensor_count"]),
            "reader_data_offset": int(reader.data_offset),
        },
        "metadata_count": len(fields),
        "metadata": {key: preview(value) for key, value in sorted(fields.items())},
        "fields": fields,
        "tensor_count": len(inventory),
        "tensor_inventory_sha256": canonical_hash(inventory),
        "tensor_type_counts": dict(sorted(Counter(item["tensor_type"] for item in inventory).items())),
        "tensor_inventory": inventory,
    }


def compare_tokenizers(draft: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    draft_fields = draft["fields"]
    target_fields = target["fields"]
    draft_keys = sorted(key for key in draft_fields if key.startswith("tokenizer."))
    target_keys = sorted(key for key in target_fields if key.startswith("tokenizer."))
    union = sorted(set(draft_keys) | set(target_keys))
    differences = []
    exact_common = []
    for key in union:
        if key not in draft_fields or key not in target_fields:
            differences.append({
                "key": key,
                "kind": "missing_from_draft" if key not in draft_fields else "missing_from_target",
                "draft": preview(draft_fields[key]) if key in draft_fields else None,
                "target": preview(target_fields[key]) if key in target_fields else None,
            })
        elif draft_fields[key] != target_fields[key]:
            differences.append({
                "key": key,
                "kind": "value_mismatch",
                "draft": preview(draft_fields[key]),
                "target": preview(target_fields[key]),
            })
        else:
            exact_common.append(key)
    arrays = {}
    for key in sorted(TOKEN_ARRAY_KEYS):
        left = draft_fields.get(key)
        right = target_fields.get(key)
        arrays[key] = {
            "draft": preview(left) if left is not None else None,
            "target": preview(right) if right is not None else None,
            "exact": left == right,
        }
    common_keys = sorted(set(draft_keys) & set(target_keys))
    common_differences = [item for item in differences if item["key"] in common_keys]
    non_known = [item for item in differences if item["key"] not in KNOWN_TOKENIZER_DIFFERENCES]
    return {
        "draft_tokenizer_keys": draft_keys,
        "target_tokenizer_keys": target_keys,
        "exact_common_keys": exact_common,
        "differences": differences,
        "known_difference_keys": sorted(KNOWN_TOKENIZER_DIFFERENCES),
        "non_known_differences": non_known,
        "common_value_mismatches": common_differences,
        "token_arrays": arrays,
        "common_metadata_exact_except_known_runtime_fields": not non_known,
        "tokens_types_merges_exact": all(item["exact"] for item in arrays.values()),
        "chat_template_exact": draft_fields.get("tokenizer.chat_template") == target_fields.get("tokenizer.chat_template"),
        "special_ids_exact": all(
            draft_fields.get(key) == target_fields.get(key)
            for key in (
                "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.unknown_token_id",
                "tokenizer.ggml.padding_token_id",
            )
        ),
    }


def audit_geometry(draft: dict[str, Any]) -> dict[str, Any]:
    fields = draft["fields"]
    field_checks = {
        key: {"expected": expected, "actual": fields.get(key), "equal": fields.get(key) == expected}
        for key, expected in EXPECTED_BLOCK_FIELDS.items()
    }
    target_layers = fields.get("dflash.target_layers")
    tensor_names = {item["name"] for item in draft["tensor_inventory"]}
    block_matches = []
    for block in range(5):
        prefix = f"blk.{block}."
        found = {name[len(prefix):] for name in tensor_names if name.startswith(prefix)}
        block_matches.append({"block": block, "found": sorted(found), "exact": found == EXPECTED_BLOCK_TENSORS})
    tops = {name for name in tensor_names if not re.match(r"blk\.\d+\.", name)}
    shape_by_name = {item["name"]: item["shape"] for item in draft["tensor_inventory"]}
    expected_shapes = {
        "conf_proj.bias": [1],
        "conf_proj.weight": [2304, 1],
        "fc.weight": [10240, 2048],
        "enc.output_norm.weight": [2048],
        "markov_w1.weight": [256, 130560],
        "markov_w2.weight": [256, 130560],
        "output_norm.weight": [2048],
    }
    shape_checks = {
        key: {"expected": value, "actual": shape_by_name.get(key), "equal": shape_by_name.get(key) == value}
        for key, value in expected_shapes.items()
    }
    names_expected = EXPECTED_TOP_TENSORS | {f"blk.{block}.{suffix}" for block in range(5) for suffix in EXPECTED_BLOCK_TENSORS}
    return {
        "metadata": field_checks,
        "target_layers_exact": target_layers == EXPECTED_DSPARK_TARGET_LAYERS,
        "target_layers": target_layers,
        "block_tensor_sets": block_matches,
        "all_block_tensor_sets_exact": all(item["exact"] for item in block_matches),
        "top_tensor_names": sorted(tops),
        "top_tensor_names_exact": tops == EXPECTED_TOP_TENSORS,
        "expected_tensor_names_exact": tensor_names == names_expected,
        "top_shapes": shape_checks,
        "markov_tensors_present": all(name in tensor_names for name in ("markov_w1.weight", "markov_w2.weight")),
        "confidence_projection_present": all(name in tensor_names for name in ("conf_proj.bias", "conf_proj.weight")),
        "draft_has_no_target_embedding_or_lm_head": not any(
            name in tensor_names for name in ("token_embd.weight", "output.weight", "lm_head.weight")
        ),
        "all_metadata_checks_pass": all(item["equal"] for item in field_checks.values()),
        "all_shape_checks_pass": all(item["equal"] for item in shape_checks.values()),
    }


def main() -> int:
    draft = inspect_file(DSPARK, EXPECTED_DSPARK_BYTES, EXPECTED_DSPARK_SHA)
    target = inspect_file(TARGET, EXPECTED_TARGET_BYTES, EXPECTED_TARGET_SHA)
    tokenizer = compare_tokenizers(draft, target)
    geometry = audit_geometry(draft)
    checks = {
        "draft_size_hash": draft["bytes"] == EXPECTED_DSPARK_BYTES and draft["sha256"] == EXPECTED_DSPARK_SHA,
        "target_size_hash": target["bytes"] == EXPECTED_TARGET_BYTES and target["sha256"] == EXPECTED_TARGET_SHA,
        "draft_header_tensor_count": draft["gguf_header"]["tensor_count"] == 62 and draft["tensor_count"] == 62,
        # GGUF.kv_count excludes the three synthetic reader fields
        # (GGUF.version/kv_count/tensor_count); the reader therefore exposes
        # 37 on-disk KVs as 40 fields.
        "draft_header_kv_count": draft["gguf_header"]["kv_count"] == 37 and draft["metadata_count"] == 40,
        "geometry_metadata": geometry["all_metadata_checks_pass"],
        "geometry_tensors": geometry["expected_tensor_names_exact"] and geometry["all_shape_checks_pass"],
        "markov_and_confidence_tensors": geometry["markov_tensors_present"] and geometry["confidence_projection_present"],
        "token_arrays_exact": tokenizer["tokens_types_merges_exact"],
        "tokenizer_common_exact_except_known_runtime_fields": tokenizer["common_metadata_exact_except_known_runtime_fields"],
        "chat_template_exact": tokenizer["chat_template_exact"],
        "special_ids_exact": tokenizer["special_ids_exact"],
    }
    report = {
        "schema_version": "sepalith.run06.released-dspark-gguf-audit.v1",
        "reader": str(GGUF_PY),
        "comparison_scope": "GGUF header/metadata/tensor inventory only; no model load or forward pass",
        "files": {"draft": {key: value for key, value in draft.items() if key != "fields"}, "campaign_target_q8": {key: value for key, value in target.items() if key != "fields"}},
        "draft_geometry": geometry,
        "tokenizer_comparison": tokenizer,
        "checks": checks,
        "complete_structural_screen": all(checks.values()),
        "interpretation": {
            "tokenizer_sha_policy": "Embedded tokenizer fields and full token/type/merge arrays are compared directly. A HF tokenizer.json SHA is external provenance and is not expected as a GGUF field.",
            "draft_model_identity": "The DSpark GGUF is a five-block draft with dflash target taps and Markov/confidence tensors; it intentionally has no target embedding or lm_head.",
            "filename_note": "The official filename says 2.6B while embedded general.size_label is 324M and the source card/config describe a 2B target plus draft. The pinned file hash is the binding identity.",
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "checks": checks, "complete": report["complete_structural_screen"]}, sort_keys=True))
    return 0 if report["complete_structural_screen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
