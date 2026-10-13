#!/usr/bin/env python3
"""CPU-only contract audit for the proposed MiniCPM5 DSpark cache.

This file deliberately does not import torch/transformers, open model weights,
or create a target cache.  It checks the already admitted token rows, the
target config metadata, the canonical v2 cache byte geometry, and the index
record packing used by DeepSpec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
ROWS = REPO / "docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl"
TARGET = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native/config.json")
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
VOCAB_SIZE = 130560
TARGET_LAYER_IDS = (1, 10, 20, 30, 39)
HIDDEN_SIZE = 2048
BLOCK_SIZE = 7
MASK_TOKEN_ID = 75982
INDEX_RECORD = struct.Struct("<QIIQQQQQ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_row(row: dict, line_number: int) -> None:
    ident = row.get("id")
    ids = row.get("input_ids")
    start = row.get("target_start")
    body = row.get("target_body_tokens")
    terminal = row.get("target_terminal_tokens")
    if row.get("split") != "train":
        raise AssertionError(f"line {line_number}: split is not train")
    if not isinstance(ident, str) or not ident:
        raise AssertionError(f"line {line_number}: missing id")
    if row.get("tokenizer_json_sha256") != TOKENIZER_SHA256:
        raise AssertionError(f"line {line_number}: tokenizer identity differs")
    if not isinstance(ids, list) or not ids:
        raise AssertionError(f"line {line_number}: input_ids is not a nonempty list")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids):
        raise AssertionError(f"line {line_number}: invalid token id")
    if ids[0] != 0 or ids[-1] != 1 or ids.count(0) != 1 or ids.count(1) != 1:
        raise AssertionError(f"line {line_number}: BOS/EOS geometry differs")
    if type(start) is not int or not 1 < start < len(ids) - 1:
        raise AssertionError(f"line {line_number}: invalid target_start")
    if start != row.get("prompt_token_count", -1) + 1:
        raise AssertionError(f"line {line_number}: prompt count/start mismatch")
    if not isinstance(body, list) or not isinstance(terminal, list) or not terminal:
        raise AssertionError(f"line {line_number}: incomplete target arrays")
    if ids[start:] != body + terminal + [1]:
        raise AssertionError(f"line {line_number}: target tail does not match input_ids")
    if row.get("target_body_token_count") != len(body):
        raise AssertionError(f"line {line_number}: body count mismatch")
    if row.get("target_terminal_token_count") != len(terminal):
        raise AssertionError(f"line {line_number}: terminal count mismatch")
    if row.get("target_token_count") != len(body) + len(terminal):
        raise AssertionError(f"line {line_number}: target count mismatch")
    if len(ids) > 4096 or len(ids) - start > 192:
        raise AssertionError(f"line {line_number}: cache sequence/target cap exceeded")


def cache_bytes(token_count: int, *, hidden_size: int = HIDDEN_SIZE, taps: int = 5) -> int:
    # int32 IDs + uint8 attention/loss masks + BF16 tapped states + BF16 final state.
    return int(token_count) * (4 + 1 + 1 + 2 * hidden_size * (taps + 1))


def audit(rows_path: Path, target_path: Path, selected_rows: int) -> dict:
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    config = json.loads(target_path.read_text())
    expected_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 42,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": VOCAB_SIZE,
        "bos_token_id": 0,
        "pad_token_id": 1,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise AssertionError(f"target config {key} differs: {config.get(key)!r}")

    digest = hashlib.sha256()
    rows = []
    split_counts: dict[str, int] = {}
    total_tokens = 0
    total_target_tokens = 0
    with rows_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            row = json.loads(raw)
            validate_row(row, line_number)
            rows.append(row)
            split = row["split"]
            split_counts[split] = split_counts.get(split, 0) + 1
            total_tokens += len(row["input_ids"])
            total_target_tokens += len(row["input_ids"]) - row["target_start"]
    if len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("duplicate row id")
    if split_counts != {"train": len(rows)}:
        raise AssertionError(f"unexpected split counts: {split_counts}")
    if selected_rows < 1 or selected_rows > len(rows):
        raise ValueError("selected_rows must be in [1, total rows]")

    # Hash-order selection is stable if source row order is regenerated while
    # preserving IDs. It is a cache/adaptation smoke, never a promotion set.
    chosen = sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())[:selected_rows]
    chosen_tokens = sum(len(row["input_ids"]) for row in chosen)
    chosen_target_tokens = sum(len(row["input_ids"]) - row["target_start"] for row in chosen)
    for row in chosen:
        loss_mask = [0] * row["target_start"] + [1] * (len(row["input_ids"]) - row["target_start"])
        if sum(loss_mask) != row["target_token_count"] + 1:
            raise AssertionError("loss mask does not include target and protocol EOS")

    return {
        "rows_path": str(rows_path),
        "rows_sha256": digest.hexdigest(),
        "row_count": len(rows),
        "split_counts": split_counts,
        "unique_ids": len(rows),
        "total_input_tokens": total_tokens,
        "total_target_tokens_excluding_protocol_eos": total_target_tokens,
        "total_cache_bytes_if_all_rows": cache_bytes(total_tokens),
        "selected_rows": selected_rows,
        "selected_ids": [row["id"] for row in chosen],
        "selected_input_tokens": chosen_tokens,
        "selected_target_tokens_including_protocol_eos": chosen_target_tokens,
        "selected_cache_bytes": cache_bytes(chosen_tokens),
        "selected_index_bytes": selected_rows * INDEX_RECORD.size,
        "cache_bytes_per_token": cache_bytes(1),
        "cache_protocol": {
            "version": 2,
            "index_record_format": INDEX_RECORD.format,
            "index_record_size": INDEX_RECORD.size,
            "hidden_dtype": "bfloat16",
            "token_dtype": "int32",
            "mask_dtype": "uint8",
            "target_layer_ids": list(TARGET_LAYER_IDS),
            "hidden_size": HIDDEN_SIZE,
            "block_size": BLOCK_SIZE,
            "mask_token_id": MASK_TOKEN_ID,
        },
        "target_config_sha256": sha256(target_path),
        "target_config": expected_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--target-config", type=Path, default=TARGET)
    parser.add_argument("--selected-rows", type=int, default=128)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = audit(args.rows, args.target_config, args.selected_rows)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
