#!/usr/bin/env python3
"""Materialize a tiny deterministic selected-R2 TRAIN quality panel.

The source is streamed and hash checked.  Selected lines are copied verbatim;
no model, DEV, final set, tokenizer, or generated output is read or created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe"
FAMILY_ORDER = ("format_propagation", "na_rm_propagation", "pipe_rewrite", "rename_propagation")


def key(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rows-sha256", default=SOURCE_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    noops: list[tuple[str, int, str, dict[str, Any]]] = []
    edits: dict[str, list[tuple[str, int, str, dict[str, Any]]]] = {family: [] for family in FAMILY_ORDER}
    total = 0
    eligible = 0
    rejected = 0
    with args.rows.resolve().open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            total += 1
            row = json.loads(raw)
            if not isinstance(row, dict) or row.get("split") != "train":
                continue
            if not isinstance(row.get("prompt_token_count"), int) or not isinstance(row.get("target_token_count"), int):
                raise SystemExit(f"line {line_number}: missing token counts")
            if row["prompt_token_count"] > 2048 or row["target_token_count"] + 1 > 192:
                rejected += 1
                continue
            eligible += 1
            item = (key(row), line_number, raw.decode("utf-8"), row)
            if row.get("target_operation") == "no_op":
                noops.append(item)
            elif row.get("target_operation") in {"replace", "delete"} and row.get("family") in edits:
                edits[row["family"]].append(item)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != args.rows_sha256:
        raise SystemExit(f"source SHA mismatch: expected {args.rows_sha256}, got {actual_sha256}")
    noops.sort(key=lambda value: (value[0], value[1]))
    chosen = noops[:4]
    chosen_families: dict[str, tuple[str, int, str, dict[str, Any]]] = {}
    for family in FAMILY_ORDER:
        candidates = sorted(edits[family], key=lambda value: (value[0], value[1]))
        if not candidates:
            raise SystemExit(f"no eligible edit row for family {family}")
        chosen_families[family] = candidates[0]
        chosen.append(candidates[0])
    if len(chosen) != 8:
        raise SystemExit(f"expected 8 selected rows, got {len(chosen)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.out.open("w", encoding="utf-8", newline="") as output:
        for ordinal, (stable_key, line_number, raw_text, row) in enumerate(chosen):
            output.write(raw_text)
            if not raw_text.endswith("\n"):
                output.write("\n")
            records.append({
                "ordinal": ordinal,
                "row_id": row["id"],
                "source_line": line_number,
                "row_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                "stable_selection_key": stable_key,
                "family": row.get("family"),
                "package_id": row.get("package_id"),
                "target_operation": row.get("target_operation"),
                "prompt_token_count": row.get("prompt_token_count"),
                "target_token_count": row.get("target_token_count"),
                "split": row.get("split"),
            })
    manifest = {
        "schema_version": "sepalith.r2.serving.train-panel.v1",
        "status": "pass",
        "source": {
            "path": str(args.rows.resolve()),
            "sha256": actual_sha256,
            "expected_sha256": args.rows_sha256,
            "rows_seen": total,
            "split_required": "train",
            "prompt_limit": 2048,
            "target_plus_protocol_eos_limit": 192,
            "eligible_rows_after_limits": eligible,
            "rejected_rows": rejected,
        },
        "panel": {
            "path": str(args.out),
            "row_count": len(records),
            "records": records,
            "contains_final_or_dev": False,
            "contains_authored_padding_or_eos": False,
        },
        "selection": {
            "method": "stable row-id SHA-256",
            "no_op_rows": 4,
            "edit_rows": 4,
            "edit_family_order": list(FAMILY_ORDER),
            "contains_dev_or_final": False,
            "contains_authored_padding_or_eos": False,
        },
        "selected_r2_identity": {
            "hf_alias": "NATIVE/models/SFT11-task-global-b-500-merged",
            "hf_weights_sha256": "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c",
            "f16_path": "/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf",
            "f16_bytes": 5039006496,
            "f16_sha256": "fe1c38a2b53519fb15eeb4ac36efdd7b6f60ad5a58451475b51308a93cb8a8ee",
            "q8_path": "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf",
            "q8_bytes": 2679710496,
            "q8_sha256": "d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db",
            "f16_relocation_receipt": "docs/campaign/work/lead/r2-storage-reserve-step500-v1/SFT11-task-global-b-500-quant-relocation.json",
            "integrity_receipt": "docs/campaign/work/lead/r2-task-global500-native-selection/q8-integrity.json",
            "parent_manifest_sha256": "92157a4a52a4ed928f76df9f1f77a6aa39235eefd6a859eb9c2b3d4118ee18db",
            "renderer_id": "zeta2-prm03-v1",
            "tokenizer_json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
            "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
            "vocab_size": 130560,
            "bos_id": 0,
            "eos_id": 1,
            "pad_id": 1,
            "native_eog_ids": [1, 130073],
        },
        "runtime_contract": {
            "context_size": 4096,
            "completion_cap": 192,
            "ordinary_decoding": {"temperature": 0, "greedy": True, "seed": 0},
            "server_args_same_for_both": ["-c", "4096", "-b", "256", "-ub", "256", "--parallel", "1", "-ngl", "99", "-ngld", "99"],
            "no_authored_target_tail": True,
            "score_each_row_once_on_cold_response": True,
        },
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(records), "source_rows": total, "source_sha256": actual_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
