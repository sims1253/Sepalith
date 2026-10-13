#!/usr/bin/env python3
"""Independent structural verification for the DAT10 15,008-row candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["status"] == "candidate_only_root_admission_pending"
    for name, meta in manifest["artifacts"].items():
        path = args.output / name
        assert path.stat().st_size == meta["bytes"]
        assert sha(path) == meta["sha256"]

    candidate = list(rows(args.output / "candidate-token-rows.jsonl"))
    combined = list(rows(args.output / "combined-token-rows.jsonl"))
    receipts = list(rows(args.output / "candidate-per-example-receipts.jsonl"))
    geometry = list(rows(args.output / "tokenizer-render-geometry-audit.jsonl"))
    dispositions = list(rows(args.output / "review-disposition-ledger.jsonl"))
    origins = list(rows(args.output / "combined-row-origin.jsonl"))
    assert (len(candidate), len(combined), len(receipts), len(geometry), len(dispositions), len(origins)) == (3503, 15008, 3503, 3503, 5245, 15008)
    assert combined[11505:] == candidate
    ids = [x["id"] for x in combined]
    assert len(ids) == len(set(ids)) == 15008
    assert [x["row_id"] for x in receipts] == [x["id"] for x in candidate]
    assert [x["row_id"] for x in geometry] == [x["id"] for x in candidate]
    assert all(not x["repair_reasons"] and x["license_evidence_present"] and x["tokenizer_reencode_exact"] for x in receipts)
    assert all(all(x["checks"].values()) and not x["selection_overflow"] and not x["selection_omissions"] for x in geometry)
    assert sum(x["decision"] == "admissible" for x in dispositions) == 3503
    assert sum(x["decision"] == "repair_required" for x in dispositions) == 1572
    assert sum(x["decision"] == "exclude_duplicate" for x in dispositions) == 170
    for row in combined:
        assert row["input_ids"][0] == row["bos_token_id"] == 0
        assert row["input_ids"][-1] == row["eos_token_id"] == 1
        assert row["target_start"] == row["prompt_token_count"] + 1
        assert len(row["input_ids"]) == row["target_start"] + row["target_token_count"] + 1
    schedule = json.loads((args.output / "schedule-proposal.json").read_text())
    assert schedule["pool"]["unique_rows"] == 15008
    assert schedule["recommended_preserve_25pct_no_op"]["length_bucket_plan"]["bucket_4096_unique_long_rows"] == 1454
    assert schedule["recommended_preserve_25pct_no_op"]["length_bucket_plan"]["bucket_4096_batches"] == 121
    assert schedule["recommended_preserve_25pct_no_op"]["minimum_updates_covering_all_unique_edits"] == 1160
    print("PASS: 15008 combined rows; 3503 exact re-encoded candidates; lineage, repair, length schedule coherent")


if __name__ == "__main__":
    main()
