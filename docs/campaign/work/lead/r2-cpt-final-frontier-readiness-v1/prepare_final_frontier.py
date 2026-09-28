#!/usr/bin/env python3
"""Prepare a lossless 16K input manifest after all CPT groups finish.

The historical frontier preparer accepts only an in-progress prefix.  This
fresh helper accepts the terminal state (all 8,092 groups) and preserves any
repair queues as explicit ledger entries.  It validates committed artifact
hashes and writes only a manifest/bindings packet; the rechunker is a separate
command in ``final-commands.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


START = 775
GROUPS = 8092
END = START + GROUPS
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
RAW_SHA = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
RECHUNK_SHA = "0530b93885e4db9f4c8f87acc6e5efd9bd735f82b5b2c24b8006d145acb5fa13"
CAMPAIGN_CPT_SHA = "8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa"
TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output_must_be_fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    progress_raw = args.progress.read_bytes()
    progress = json.loads(progress_raw)
    if progress.get("groups_committed") != GROUPS or progress.get("groups_remaining") != 0:
        raise ValueError("main_groups_not_terminal")
    if progress.get("status") not in {"complete", "all_groups_inventoried_repairs_pending"}:
        raise ValueError("main_status_not_terminal")
    if progress.get("first_uncommitted_seeded_index") not in {None, END}:
        raise ValueError("main_frontier_not_terminal")
    source_pins = {
        "raw_cpt_broader.py": RAW_SHA,
        "lossless_rechunk.py": RECHUNK_SHA,
        "campaign_cpt_data.py": CAMPAIGN_CPT_SHA,
    }
    for name, expected in source_pins.items():
        path = args.source / name
        if sha_file(path) != expected:
            raise ValueError(f"source_pin_mismatch:{name}")

    inputs: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    seen_groups: set[str] = set()
    for index in range(START, END):
        matches = list(args.groups.glob(f"{index:06d}-*"))
        if len(matches) != 1 or not matches[0].is_dir():
            raise ValueError(f"group_directory_missing_or_ambiguous:{index}")
        folder = matches[0]
        receipt_path = folder / "receipt.json"
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        if receipt.get("schema") != "sepalith.cpt.all-eligible-group.v1":
            raise ValueError(f"group_schema_mismatch:{index}")
        if receipt.get("seeded_index") != index:
            raise ValueError(f"group_index_mismatch:{index}")
        group_id = receipt.get("group_id")
        if not isinstance(group_id, str) or folder.name != f"{index:06d}-{group_id}" or group_id in seen_groups:
            raise ValueError(f"group_identity_mismatch:{index}")
        seen_groups.add(group_id)
        status = receipt.get("status")
        if status not in {"complete", "complete_with_repairs_pending"}:
            raise ValueError(f"group_not_committed:{index}")
        statuses[status] += 1
        counts = receipt.get("counts", {})
        for key in ("rows", "documents", "payload_tokens"):
            if not isinstance(counts.get(key, 0), int) or counts.get(key, 0) < 0:
                raise ValueError(f"invalid_group_count:{index}:{key}")
            totals[key] += counts[key]
        artifacts = receipt.get("artifacts", {})
        for name in ("cpt_train.jsonl", "documents.jsonl", "exclusions.jsonl", "repair-queue.jsonl", "source-inventory.jsonl"):
            path = folder / name
            pin = artifacts.get(name)
            if not path.is_file() or not isinstance(pin, dict):
                raise ValueError(f"group_artifact_missing:{index}:{name}")
            if path.stat().st_size != pin.get("bytes") or sha_file(path) != pin.get("sha256"):
                raise ValueError(f"group_artifact_hash_mismatch:{index}:{name}")
        queue_path = folder / "repair-queue.jsonl"
        queue_rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines() if line]
        if len(queue_rows) != counts.get("repair_items", 0):
            raise ValueError(f"repair_count_mismatch:{index}")
        for repair in queue_rows:
            repairs.append({"seeded_index": index, "group_id": group_id, "package": receipt.get("package"),
                            "receipt_path": str(receipt_path), "repair": repair})
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        cpt_pin = artifacts["cpt_train.jsonl"]
        inputs.append({
            "seeded_index": index,
            "group_id": group_id,
            "package": receipt.get("package"),
            "committed_receipt": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "receipt_status": status,
            "path": str(folder / "cpt_train.jsonl"),
            "source_path": str(folder / "cpt_train.jsonl"),
            "bytes": cpt_pin["bytes"],
            "sha256": cpt_pin["sha256"],
            "rows": counts["rows"],
            "documents": counts["documents"],
            "payload_tokens": counts["payload_tokens"],
            "cpt_partition": "cpt_train",
        })
        bindings.append({"seeded_index": index, "group_id": group_id, "package": receipt.get("package"),
                         "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha,
                         "status": status, "counts": counts, "artifacts": artifacts})

    manifest = {
        "schema": "sepalith.cpt.lossless-rechunk-input.v2",
        "status": "terminal_main_candidate_pending_root_admission",
        "context_sizes": [16384],
        "frontier": {"start_inclusive": START, "end_exclusive": END, "groups": GROUPS,
                      "frozen_progress_path": str(args.progress), "frozen_progress_sha256": hashlib.sha256(progress_raw).hexdigest()},
        "expected_totals": dict(totals),
        "inputs": inputs,
        "raw_chunks": {"path": str(args.source / "raw_cpt_broader.py"), "sha256": RAW_SHA,
                        "bos": 0, "eos": 1, "source_chunk_size": 2048},
        "source": {"lossless_rechunk_sha256": RECHUNK_SHA, "campaign_cpt_data_sha256": CAMPAIGN_CPT_SHA},
        "original_tokenizer": {"path": "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json",
                                "sha256": TOKENIZER_SHA, "retokenized": False},
        "repair_items_pending": len(repairs),
        "receipt_statuses": dict(statuses),
        "training_admission": False,
        "truncation": False,
        "heldout_content_read": False,
    }
    write_json(args.output / "input-manifest.json", manifest)
    with (args.output / "receipt-bindings.jsonl").open("x", encoding="utf-8") as stream:
        for row in bindings:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with (args.output / "repair-ledger.jsonl").open("x", encoding="utf-8") as stream:
        for row in repairs:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    audit = {"schema": "sepalith.cpt.lossless-rechunk-terminal-preparation.v1", "status": "complete_pending_repair_admission",
             "input_manifest": str(args.output / "input-manifest.json"), "input_manifest_sha256": sha_file(args.output / "input-manifest.json"),
             "receipt_bindings": str(args.output / "receipt-bindings.jsonl"), "repair_ledger": str(args.output / "repair-ledger.jsonl"),
             "groups": GROUPS, "totals": dict(totals), "repairs": repairs, "receipt_statuses": dict(statuses),
             "training_admission": False, "truncation": False}
    write_json(args.output / "terminal-preparation.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
