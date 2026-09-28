#!/usr/bin/env python3
"""Audit intermediate GGUF quantizations without loading tensor values."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from gguf import GGUFReader


INPUT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step500-runtime-gguf/model-F16.gguf")
OUTPUT_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step500-quant-candidates")
WORK_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/quant-preparation")
EXPECTED_INPUT_SHA256 = "50f523af997f2f77dcbd187adde8a36dbc41529932703fd016e506b172761725"
EXPECTED_OUTPUTS = {
    "Q6_K": ("model-Q6_K.gguf", "b11ffcc093b78261af1c5eb450feefcbca6ee35c0cecc22133236506410e143e"),
    "Q5_K_M": ("model-Q5_K_M.gguf", "84764de53d128e589627dadcceb50ad4c18c1705b55e7477816bd66a293f4b1f"),
    "Q4_K_M": ("model-Q4_K_M.gguf", "f6f005d6fdd92142debcdf7766efd16698a4a3b37077891268c2fef79208b828"),
}
PROTECTED = ("output.weight", "token_embd.weight")
ALLOWED_METADATA_DIFFERENCES = ("general.file_type",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_inventory(reader: GGUFReader) -> list[dict[str, object]]:
    return [
        {
            "name": tensor.name,
            "shape": [int(value) for value in tensor.shape],
            "tensor_type": tensor.tensor_type.name,
            "n_elements": int(tensor.n_elements),
            "n_bytes": int(tensor.n_bytes),
        }
        for tensor in reader.tensors
    ]


def metadata_values(reader: GGUFReader) -> dict[str, object]:
    return {name: field.contents() for name, field in reader.fields.items()}


def main() -> int:
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA256:
        raise SystemExit("verified F16 input SHA256 changed")
    source = GGUFReader(INPUT)
    source_inventory = tensor_inventory(source)
    source_metadata = metadata_values(source)
    source_by_name = {item["name"]: item for item in source_inventory}
    report: dict[str, object] = {
        "schema_version": "sepalith.run09.gguf-quantization-audit.v1",
        "input": {
            "path": str(INPUT),
            "sha256": EXPECTED_INPUT_SHA256,
            "bytes": INPUT.stat().st_size,
            "tensor_count": len(source_inventory),
            "metadata_count": len(source_metadata),
            "tensor_inventory_sha256": canonical_hash(source_inventory),
            "metadata_keys": sorted(source_metadata),
        },
        "candidates": {},
    }
    candidates: dict[str, object] = {}
    for requested_type, (filename, expected_sha256) in EXPECTED_OUTPUTS.items():
        path = OUTPUT_ROOT / filename
        actual_sha256 = sha256_file(path)
        reader = GGUFReader(path)
        inventory = tensor_inventory(reader)
        metadata = metadata_values(reader)
        differences = []
        for key in sorted(set(source_metadata) | set(metadata)):
            if key not in source_metadata or key not in metadata:
                differences.append({"key": key, "kind": "missing"})
            elif source_metadata[key] != metadata[key]:
                differences.append({
                    "key": key,
                    "source": source_metadata[key],
                    "candidate": metadata[key],
                })
        source_names_shapes = [(item["name"], item["shape"]) for item in source_inventory]
        candidate_names_shapes = [(item["name"], item["shape"]) for item in inventory]
        tensor_by_name = {item["name"]: item for item in inventory}
        type_counts = dict(sorted(Counter(item["tensor_type"] for item in inventory).items()))
        protected_types = {
            name: tensor_by_name.get(name, {}).get("tensor_type") for name in PROTECTED
        }
        primary_types = sorted({
            item["tensor_type"] for item in inventory
            if item["name"] not in PROTECTED and item["tensor_type"] not in ("F32",)
        })
        candidate_report = {
            "requested_quant_type": requested_type,
            "path": str(path),
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
            "sha256_matches_terminal_record": actual_sha256 == expected_sha256,
            "bytes": path.stat().st_size,
            "gguf_version": int(reader.fields["GGUF.version"].contents()),
            "tensor_count": len(inventory),
            "metadata_count": len(metadata),
            "tensor_inventory_sha256": canonical_hash(inventory),
            "tensor_inventory": inventory,
            "tensor_names_and_shapes_match_f16": sorted(candidate_names_shapes) == sorted(source_names_shapes),
            "tensor_order_matches_f16": candidate_names_shapes == source_names_shapes,
            "tensor_names_sha256": canonical_hash([item["name"] for item in inventory]),
            "tensor_shapes_sha256": canonical_hash([item["shape"] for item in inventory]),
            "tensor_type_counts": type_counts,
            "protected_tensor_types": protected_types,
            "protected_tensors_are_q8_0": protected_types == {name: "Q8_0" for name in PROTECTED},
            "non_protected_primary_types": primary_types,
            "metadata_differences": differences,
            "metadata_parity_except_file_type": (
                all(item["key"] in ALLOWED_METADATA_DIFFERENCES for item in differences)
                and all(key in metadata for key in source_metadata)
            ),
        }
        candidates[requested_type] = candidate_report
    report["candidates"] = candidates
    report["checks"] = {
        "input_sha256": report["input"]["sha256"] == EXPECTED_INPUT_SHA256,
        "all_candidates_sha256_match": all(
            item["sha256_matches_terminal_record"] for item in candidates.values()
        ),
        "all_tensor_counts_381": all(item["tensor_count"] == 381 for item in candidates.values()),
        "all_tensor_names_and_shapes_match_f16": all(
            item["tensor_names_and_shapes_match_f16"] for item in candidates.values()
        ),
        "all_metadata_parity_except_file_type": all(
            item["metadata_parity_except_file_type"] for item in candidates.values()
        ),
        "all_protected_tensors_q8_0": all(
            item["protected_tensors_are_q8_0"] for item in candidates.values()
        ),
        "all_metadata_keys_preserved": all(
            item["metadata_count"] == report["input"]["metadata_count"] for item in candidates.values()
        ),
    }
    report["checks"]["complete"] = all(report["checks"].values())
    output = WORK_ROOT / "gguf-quantization-audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "checks": report["checks"]}, sort_keys=True))
    return 0 if report["checks"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
