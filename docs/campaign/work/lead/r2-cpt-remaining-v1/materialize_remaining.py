#!/usr/bin/env python3
"""Materialize every unconsumed row from the two admitted CPT shards."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PINS = {
    "broad_rows": "c0ef25c474dda17b464c3f412b12cc1b64f1538085dae8c8603c44293ced4b35",
    "broad_schedule": "39e02ec24cd51d27d8eb42bb4bbc4bd5c1bbf3a72b82a6d502f9a71dc4a9ae48",
    "global_rows": "aac7e1b6140b011043f25a22970c0e8c300fde65316f7e5fbcbda7d747d4f703",
    "global_schedule": "b8a756534875ebb629db03213fa6155140ac9aaeaf75fa4bf3b9f63c33df88e0",
    "root_audit": "dd20c359b0f977bfacaec5a843a26767cbadf3c3a4e66bfc30f937814762bdf3",
    "reproduce_audit": "4517decfa5c9ff9a92733db90cfd18346cc482343427928f7b71b50e13f22a23",
}
SPLIT_ID = "DAT-02-global-v2-285001f3d93e9f1871df"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_pin(path: Path, name: str) -> None:
    actual = digest(path)
    if actual != PINS[name]:
        raise ValueError(f"{name} hash differs: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-rows", type=Path, required=True)
    parser.add_argument("--broad-schedule", type=Path, required=True)
    parser.add_argument("--global-rows", type=Path, required=True)
    parser.add_argument("--global-schedule", type=Path, required=True)
    parser.add_argument("--root-audit", type=Path, required=True)
    parser.add_argument("--reproduce-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "broad_rows": args.broad_rows,
        "broad_schedule": args.broad_schedule,
        "global_rows": args.global_rows,
        "global_schedule": args.global_schedule,
        "root_audit": args.root_audit,
        "reproduce_audit": args.reproduce_audit,
    }
    for name, path in sources.items():
        require_pin(path, name)
    audit = json.loads(args.root_audit.read_text())
    if audit["combined_materialized_pool"]["remaining_rows"] != 30421:
        raise ValueError("root audit remaining-row count differs")

    broad_schedule = json.loads(args.broad_schedule.read_text())
    global_schedule = json.loads(args.global_schedule.read_text())
    consumed = set(broad_schedule["row_ids"][:12000]) | set(global_schedule["row_ids"][:4000])
    if len(consumed) != 16000:
        raise ValueError("selected lineage must contain 16000 distinct row IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    output_path = args.output / "cpt_train.jsonl"
    output_hash = hashlib.sha256()
    row_ids = []
    seen = set()
    totals = Counter()
    packages = set()
    documents = set()
    groups = set()
    shard_counts = Counter()
    with output_path.open("wb") as output:
        for shard, source in (("broad750", args.broad_rows), ("global250", args.global_rows)):
            with source.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    row_id = row["row_id"]
                    if row_id in consumed:
                        continue
                    if row_id in seen:
                        raise ValueError(f"duplicate remaining row {row_id}")
                    seen.add(row_id)
                    row_ids.append(row_id)
                    output.write(line)
                    output_hash.update(line)
                    shard_counts[shard] += 1
                    packages.add(row["package"])
                    documents.add(row["document_id"])
                    groups.add(row["group_id"])
                    totals["input_tokens"] += len(row["input_ids"])
                    totals["loss_tokens"] += sum(value != -100 for value in row["labels"])
                    totals["payload_tokens"] += row["source_token_end"] - row["source_token_start"]
                    totals["overlap_context_tokens"] += row["overlap_context_tokens"]
    if len(row_ids) != 30421 or len(seen) != 30421:
        raise ValueError(f"expected 30421 distinct remaining rows, got {len(seen)}")
    expected = audit["combined_materialized_pool"]
    for key, audit_key in (("input_tokens", "remaining_input_tokens"),
                           ("loss_tokens", "remaining_loss_tokens"),
                           ("payload_tokens", "remaining_payload_tokens")):
        if totals[key] != expected[audit_key]:
            raise ValueError(f"remaining {key} differs from root audit")

    replay_ids = row_ids[:11]
    draws = row_ids + replay_ids
    if len(draws) != 30432 or len(draws) % 16:
        raise ValueError("remaining schedule is not batch aligned")
    rows_sha = output_hash.hexdigest()
    schedule = {
        "schema": "sepalith.cpt.remaining-draws.v1",
        "split_id": SPLIT_ID,
        "effective_batch": 16,
        "max_steps": 1902,
        "method": "one_pass_source_order_plus_named_replay_v1",
        "token_rows_sha256": rows_sha,
        "row_ids": draws,
        "unique_rows": 30421,
        "replay_rows": 11,
        "replay_row_ids": replay_ids,
        "every_unique_row_named": True,
    }
    schedule_path = args.output / "draws-1902.json"
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "sepalith.cpt.remaining-materialization.v1",
        "status": "all_eligible_remaining_rows_materialized",
        "policy": "docs/campaign/TRAINING-DATA-POLICY.md",
        "source_pins": {name: {"path": str(path), "sha256": PINS[name], "bytes": path.stat().st_size}
                        for name, path in sources.items()},
        "selection": {
            "selected_lineage_distinct_rows_excluded": 16000,
            "remaining_distinct_rows": 30421,
            "broad_remaining_rows": shard_counts["broad750"],
            "global_remaining_rows": shard_counts["global250"],
            "documents": len(documents),
            "packages": len(packages),
            "groups": len(groups),
            **dict(totals),
        },
        "schedule": {
            "draws": 30432,
            "updates": 1902,
            "effective_batch": 16,
            "unique_rows": 30421,
            "replay_rows": 11,
            "replay_row_ids": replay_ids,
        },
        "artifacts": {
            "cpt_train.jsonl": {"sha256": rows_sha, "bytes": output_path.stat().st_size},
            "draws-1902.json": {"sha256": digest(schedule_path), "bytes": schedule_path.stat().st_size},
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "selection": manifest["selection"],
                      "schedule": manifest["schedule"], "artifacts": manifest["artifacts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
