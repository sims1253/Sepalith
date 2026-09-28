#!/usr/bin/env python3
"""Freeze a read-only inventory of the live all-CPT frontier.

This script reads progress, group receipts, and repair queues only.  It does
not read candidate payload JSONL files and never changes the live materializer
directory.  A live worker may commit another group while the snapshot is
being made; the receipt records that race and binds every observation to the
captured progress and run-manifest hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


START = 775
EXPECTED = 8092
ORDER = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cpt-long-stage-review-v1/next-global-package-order.json")
MAIN = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1")
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
GROUP_RE = re.compile(r"^(\d{6})-(g-[0-9a-f]+)$")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_write(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def process_snapshot() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        output = subprocess.check_output(["pgrep", "-af", "materialize_all_eligible.py"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return rows
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            rows.append({"pid": parts[0], "command": parts[1]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output_must_be_fresh")
    args.output.mkdir(parents=True, exist_ok=True)

    progress_path = MAIN / "progress.json"
    run_manifest_path = MAIN / "run-manifest.json"
    progress_raw = progress_path.read_bytes()
    run_manifest_raw = run_manifest_path.read_bytes()
    progress = json.loads(progress_raw)
    run_manifest = json.loads(run_manifest_raw)
    if run_manifest.get("order_start") != START or run_manifest.get("groups") != EXPECTED:
        raise ValueError("main_schedule_guard_mismatch")
    order_doc = json_read(ORDER)
    order = order_doc.get("packages", [])
    if len(order) < START + EXPECTED:
        raise ValueError("global_order_shorter_than_main_schedule")
    schedule = order[START : START + EXPECTED]
    expected_by_index = {START + offset: row for offset, row in enumerate(schedule)}
    if len(expected_by_index) != EXPECTED or len({row["group_id"] for row in schedule}) != EXPECTED:
        raise ValueError("global_order_identity_not_unique")

    groups_dir = MAIN / "groups"
    discovered: dict[int, Path] = {}
    malformed_dirs: list[str] = []
    for path in groups_dir.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        match = GROUP_RE.match(path.name)
        if match is None:
            malformed_dirs.append(path.name)
            continue
        index = int(match.group(1))
        if START <= index < START + EXPECTED:
            if index in discovered:
                raise ValueError(f"duplicate_group_index:{index}")
            discovered[index] = path

    statuses: Counter[str] = Counter()
    category_inventory: dict[str, dict[str, int]] = defaultdict(lambda: {"groups": 0, "inventory_units": 0})
    aggregate_counts: Counter[str] = Counter()
    group_records: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    committed_indices: set[int] = set()
    for index, folder in sorted(discovered.items()):
        receipt_path = folder / "receipt.json"
        if not receipt_path.is_file():
            statuses["missing_receipt"] += 1
            continue
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        status = str(receipt.get("status", "missing_status"))
        statuses[status] += 1
        if receipt.get("seeded_index") != index:
            raise ValueError(f"receipt_index_mismatch:{index}")
        group_id = receipt.get("group_id")
        expected = expected_by_index[index]
        if group_id != expected["group_id"] or folder.name != f"{index:06d}-{group_id}":
            raise ValueError(f"schedule_identity_mismatch:{index}")
        if status in {"complete", "complete_with_repairs_pending"}:
            committed_indices.add(index)
        counts = receipt.get("counts", {})
        for key, value in counts.items():
            if isinstance(value, int):
                aggregate_counts[key] += value
        for category, value in receipt.get("source_categories", {}).items():
            if isinstance(value, int):
                category_inventory[category]["groups"] += 1
                category_inventory[category]["inventory_units"] += value
        artifacts = receipt.get("artifacts", {})
        queue_path = folder / "repair-queue.jsonl"
        queue_rows: list[dict[str, Any]] = []
        if queue_path.is_file() and queue_path.stat().st_size:
            queue_rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines() if line]
            for repair in queue_rows:
                repair_rows.append({
                    "seeded_index": index,
                    "group_id": group_id,
                    "package": receipt.get("package"),
                    "group_dir": str(folder),
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                    "repair_queue_path": str(queue_path),
                    "repair_queue_sha256": sha_file(queue_path),
                    "repair": repair,
                })
        group_records.append({
            "seeded_index": index,
            "group_id": group_id,
            "package": receipt.get("package"),
            "status": status,
            "receipt_path": str(receipt_path),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "counts": counts,
            "source_categories": receipt.get("source_categories", {}),
            "repair_items": len(queue_rows),
            "repair_queue_sha256": sha_file(queue_path) if queue_path.is_file() else None,
        })

    remaining = []
    for index, entry in expected_by_index.items():
        if index not in committed_indices:
            remaining.append({
                "seeded_index": index,
                "group_id": entry["group_id"],
                "package": entry["name"],
                "split": entry.get("split"),
                "flags": entry.get("flags", []),
                "status": "uncommitted_at_snapshot",
            })
    contiguous_end = START
    while contiguous_end in committed_indices:
        contiguous_end += 1
    progress_counts = progress.get("counts", {})
    count_deltas = {
        key: aggregate_counts.get(key, 0) - value
        for key, value in progress_counts.items()
        if isinstance(value, int) and aggregate_counts.get(key, 0) != value
    }
    inventory = {
        "schema": "sepalith.dat10.cpt_frontier_inventory.v1",
        "status": "read_only_snapshot_live_materializer",
        "main_output": str(MAIN),
        "progress": {
            "path": str(progress_path),
            "sha256": hashlib.sha256(progress_raw).hexdigest(),
            "observed": progress,
        },
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": hashlib.sha256(run_manifest_raw).hexdigest(),
            "materializer_sha256": run_manifest.get("materializer_sha256"),
            "order_start": run_manifest.get("order_start"),
            "groups": run_manifest.get("groups"),
        },
        "global_order": {"path": str(ORDER), "sha256": sha_file(ORDER), "scheduled_start": START,
                          "scheduled_end_exclusive": START + EXPECTED, "scheduled_groups": EXPECTED},
        "directory_observation": {
            "group_dirs_in_schedule": len(discovered),
            "committed_receipts": len(committed_indices),
            "contiguous_committed_end_exclusive": contiguous_end,
            "remaining_groups": len(remaining),
            "malformed_or_out_of_schedule_dirs": sorted(malformed_dirs),
        },
        "receipt_statuses": dict(statuses),
        "aggregate_receipt_counts": dict(aggregate_counts),
        "progress_vs_receipt_count_deltas": count_deltas,
        "source_category_inventory": {key: category_inventory[key] for key in sorted(category_inventory)},
        "processes": process_snapshot(),
        "training_admission": False,
        "payloads_read": False,
        "main_output_modified": False,
    }
    canonical_write(args.output / "frontier-inventory.json", inventory)
    with (args.output / "remaining-groups.jsonl").open("x", encoding="utf-8") as stream:
        for row in remaining:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    canonical_write(args.output / "repair-ledger.json", {
        "schema": "sepalith.dat10.cpt_repair_ledger.v1",
        "status": "observed_pending_review",
        "records": repair_rows,
        "counts": {"records": len(repair_rows), "groups": len({row["group_id"] for row in repair_rows})},
        "source_payloads_read": False,
        "training_admission": False,
    })
    canonical_write(args.output / "source-category-inventory.json", {
        "schema": "sepalith.dat10.cpt_source_category_inventory.v1",
        "scope": "committed group receipts at one live snapshot; remaining groups have no committed source category yet",
        "categories": {key: category_inventory[key] for key in sorted(category_inventory)},
        "receipt_statuses": dict(statuses),
        "committed_groups": len(committed_indices),
        "remaining_groups": len(remaining),
        "training_admission": False,
    })
    print(json.dumps({"committed_receipts": len(committed_indices), "remaining_groups": len(remaining),
                      "repair_records": len(repair_rows), "contiguous_end_exclusive": contiguous_end,
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
