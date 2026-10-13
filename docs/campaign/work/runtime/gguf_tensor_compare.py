#!/usr/bin/env python3
"""Bounded, read-only tensor-payload comparison for the two protected b4 GGUFs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GGUF_PY = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/gguf-py")
LEFT = Path("/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf")
RIGHT = Path("/home/m0hawk/Documents/Sepalith/experiments/models/b4_qwen35_2b-Q8_0.gguf")
CHUNK_BYTES = 8 * 1024 * 1024

sys.path.insert(0, str(GGUF_PY))
from gguf import GGUFReader  # noqa: E402


def tensor_hash(data) -> str:
    view = memoryview(data)
    if view.format != "B":
        view = view.cast("B")
    digest = hashlib.sha256()
    for offset in range(0, view.nbytes, CHUNK_BYTES):
        digest.update(view[offset:offset + CHUNK_BYTES])
    return digest.hexdigest()


def inspect(path: Path) -> tuple[dict, list[dict]]:
    reader = GGUFReader(str(path))
    metadata = {}
    for field in reader.fields.values():
        if field.name in {"general.name", "general.architecture", "general.file_type"}:
            metadata[field.name] = field.contents()
    tensors = []
    for tensor in reader.tensors:
        tensors.append({
            "name": tensor.name,
            "type": int(tensor.tensor_type),
            "shape": [int(value) for value in tensor.shape],
            "elements": int(tensor.n_elements),
            "bytes": int(tensor.n_bytes),
            "sha256": tensor_hash(tensor.data),
        })
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "metadata": metadata,
        "tensor_count": len(tensors),
        "data_offset": int(reader.data_offset),
    }, tensors


def main() -> None:
    left_meta, left_tensors = inspect(LEFT)
    right_meta, right_tensors = inspect(RIGHT)
    inventory_equal = [
        {key: value for key, value in tensor.items() if key != "sha256"}
        for tensor in left_tensors
    ] == [
        {key: value for key, value in tensor.items() if key != "sha256"}
        for tensor in right_tensors
    ]
    differences = [
        {
            "name": left["name"],
            "left_sha256": left["sha256"],
            "right_sha256": right["sha256"],
        }
        for left, right in zip(left_tensors, right_tensors)
        if left["sha256"] != right["sha256"]
    ]
    combined = hashlib.sha256()
    for tensor in left_tensors:
        combined.update(tensor["name"].encode())
        combined.update(tensor["sha256"].encode())
    print(json.dumps({
        "reader": str(GGUF_PY),
        "chunk_bytes": CHUNK_BYTES,
        "left": left_meta,
        "right": right_meta,
        "inventory_equal": inventory_equal,
        "tensor_count_compared": min(len(left_tensors), len(right_tensors)),
        "tensor_hash_matches": sum(
            left["sha256"] == right["sha256"]
            for left, right in zip(left_tensors, right_tensors)
        ),
        "tensor_hash_differences": differences,
        "combined_left_tensor_hash": combined.hexdigest(),
        "conclusion": (
            "tensor payloads identical; file-level metadata differs"
            if inventory_equal and not differences
            else "tensor payload identity unresolved"
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
