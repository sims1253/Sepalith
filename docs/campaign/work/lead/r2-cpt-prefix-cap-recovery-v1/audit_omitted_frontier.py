#!/usr/bin/env python3
"""Audit the small pre-existing omission categories without opening CPT rows.

The output identifies the four broader package-cap records and the fourteen
global empty/over-4MiB records.  It only reads exclusion/registry metadata,
DESCRIPTION files, source stat records, and terminal document provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
GLOBAL_SPLIT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
PROVENANCE = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1/document-provenance.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def description_fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    key: str | None = None
    for line in text.splitlines():
        if line[:1].isspace() and key:
            result[key] += " " + line.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            result[key] = value.strip()
    return result


def source_stat(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path), "bytes": None}
    value = path.stat()
    return {"exists": True, "path": str(path), "bytes": value.st_size,
            "inode": value.st_ino, "device": value.st_dev,
            "mtime_ns": value.st_mtime_ns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", type=Path,
                    default=PLAN / "docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/global-cap-exclusion-frontier.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    terminal_paths = {
        row["source_path"] for row in jsonl(PROVENANCE)
        if isinstance(row.get("source_path"), str)
    }
    cap_frontier = jsonl(args.frontier)
    cap_paths = {row["path"] for row in cap_frontier}
    global_exclusion_path = PLAN / "docs/campaign/work/r2-cpt-global-shard-v1/shard/exclusions.jsonl"
    broad_exclusion_path = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/broader-shard-v1-2k/exclusions.jsonl"
    global_exclusions = [row for row in jsonl(global_exclusion_path)
                         if row.get("reason") == "R_file_empty_or_over_4MiB"]
    broad_exclusions = [row for row in jsonl(broad_exclusion_path)
                        if row.get("reason") == "package_code_token_cap"]
    global_receipts = {
        row["package"]: row for row in jsonl(
            PLAN / "docs/campaign/work/r2-cpt-global-shard-v1/shard/package-receipts.jsonl"
        )
    }
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))["groups"]
    split_groups = {row["group_id"]: row for row in json.loads(GLOBAL_SPLIT.read_text(encoding="utf-8"))["groups"]}
    broad_meta_by_package: dict[str, dict[str, Any]] = {}
    broad_meta_path = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/broader-shard-v1-2k/selected-source-metadata.jsonl"
    for row in jsonl(broad_meta_path):
        broad_meta_by_package.setdefault(row["package"], row)

    broad_records: list[dict[str, Any]] = []
    for excluded in broad_exclusions:
        path = Path(excluded["path"])
        # The older exclusion record has no package field.  The package is the
        # normalized path component and is cross-bound to selected metadata.
        package = path.parts[5]
        meta = broad_meta_by_package.get(package)
        if meta is None:
            raise AssertionError(f"broad cap package missing metadata: {package}")
        if meta["cpt_partition"] != "cpt_train" or meta["split"] != "train_group":
            raise AssertionError(f"broad cap is not TRAIN: {path}")
        group = meta["group_id"]
        if partition.get(group) != "cpt_train" or split_groups.get(group, {}).get("split") != "train_group":
            raise AssertionError(f"broad cap group is not TRAIN: {group}")
        broad_records.append({
            "scope": "base_broader_shard",
            "package": package,
            "group_id": group,
            "path": str(path),
            "reason": excluded["reason"],
            "code_tokens_recorded_by_old_builder": excluded["code_tokens"],
            "source_stat": source_stat(path),
            "license": meta.get("license"),
            "description_path": meta.get("description_path"),
            "description_sha256": meta.get("description_sha256"),
            "split": meta["split"],
            "cpt_partition": meta["cpt_partition"],
            "terminal_exact_source_path_hit": str(path) in terminal_paths,
            "source_sha256": None,
            "disposition": "recoverable_pending_source_rehash_without_package_cap",
        })

    empty_records: list[dict[str, Any]] = []
    for excluded in global_exclusions:
        path = Path(excluded["path"])
        package = excluded["package"]
        receipt = global_receipts.get(package)
        if receipt is None:
            raise AssertionError(f"global empty package missing receipt: {package}")
        group = receipt["group_id"]
        if partition.get(group) != "cpt_train" or split_groups.get(group, {}).get("split") != "train_group":
            raise AssertionError(f"global empty group is not TRAIN: {group}")
        description = path.parent.parent / "DESCRIPTION"
        description_value: dict[str, Any] = {"path": str(description), "exists": description.is_file()}
        if description.is_file():
            raw = description.read_bytes()
            description_value.update({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                     "fields": description_fields(raw.decode("utf-8"))})
        st = source_stat(path)
        if st.get("bytes") != 0:
            raise AssertionError(f"global empty record is no longer zero-byte: {path}")
        empty_records.append({
            "scope": "base_global_shard",
            "package": package,
            "group_id": group,
            "seeded_index": receipt["seeded_index"],
            "path": str(path),
            "reason": excluded["reason"],
            "source_stat": st,
            "description": description_value,
            "split": split_groups[group]["split"],
            "cpt_partition": partition[group],
            "terminal_exact_source_path_hit": str(path) in terminal_paths,
            "source_sha256": EMPTY_SHA256,
            "disposition": "closed_exclude_zero_byte_no_payload",
        })

    records = broad_records + empty_records
    if len(cap_frontier) != 1999 or len(broad_records) != 4 or len(empty_records) != 14:
        raise AssertionError("omission denominators changed")
    if any(row["terminal_exact_source_path_hit"] for row in records):
        raise AssertionError("an omitted source path is already in terminal provenance")
    output = {
        "schema": "sepalith.dat10.omitted-frontier-audit.v1",
        "status": "verified_metadata_and_stat_only_no_payload_opened",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "terminal_provenance": {
            "path": str(PROVENANCE),
            "documents_indexed": len(terminal_paths),
            "exact_source_path_set": True,
            "pin": pin(PROVENANCE),
        },
        "global_cap_frontier": {
            "documents": len(cap_frontier),
            "groups": len({row["group_id"] for row in cap_frontier}),
            "seeded_index_min": min(row["seeded_index"] for row in cap_frontier),
            "seeded_index_max": max(row["seeded_index"] for row in cap_frontier),
            "terminal_exact_source_path_hits": sum(row["path"] in terminal_paths for row in cap_frontier),
            "input_pin": pin(args.frontier),
        },
        "supplemental_omissions": {
            "base_broader_package_cap": {
                "records": len(broad_records),
                "packages": len({row["package"] for row in broad_records}),
                "groups": len({row["group_id"] for row in broad_records}),
                "code_tokens_recorded": sum(row["code_tokens_recorded_by_old_builder"] for row in broad_records),
                "source_bytes_current_stat": sum(row["source_stat"]["bytes"] for row in broad_records),
                "terminal_exact_source_path_hits": sum(row["terminal_exact_source_path_hit"] for row in broad_records),
            },
            "base_global_empty_or_over_4MiB": {
                "records": len(empty_records),
                "packages": len({row["package"] for row in empty_records}),
                "groups": len({row["group_id"] for row in empty_records}),
                "zero_byte_records": sum(row["source_stat"]["bytes"] == 0 for row in empty_records),
                "over_4MiB_records": sum(row["source_stat"]["bytes"] > 4 * 1024 * 1024 for row in empty_records),
                "terminal_exact_source_path_hits": sum(row["terminal_exact_source_path_hit"] for row in empty_records),
            },
        },
        "records": records,
        "content_dedup_limit": "The old omission ledgers do not carry source hashes for the cap records. Recovery must hash source bytes after stat stability and deduplicate against terminal/protected/reserved identities.",
        "inputs": {
            "global_exclusions": pin(global_exclusion_path),
            "broad_exclusions": pin(broad_exclusion_path),
            "global_package_receipts": pin(PLAN / "docs/campaign/work/r2-cpt-global-shard-v1/shard/package-receipts.jsonl"),
            "broad_selected_source_metadata": pin(broad_meta_path),
            "global_split": pin(GLOBAL_SPLIT),
            "cpt_partition": pin(PARTITION),
        },
        "payload_opened": False,
        "training_admission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "cap": len(cap_frontier), "broad_cap": len(broad_records), "global_empty": len(empty_records)}, sort_keys=True))


if __name__ == "__main__":
    main()
