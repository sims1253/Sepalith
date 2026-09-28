#!/usr/bin/env python3
import argparse, collections, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--review", type=Path, required=True)
ap.add_argument("--ledger", type=Path, required=True)
ap.add_argument("--noop-metadata", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

noop_ids = {json.loads(line)["row_id"] for line in a.noop_metadata.open()}
by_reason = collections.Counter()
by_combo = collections.Counter()
held_by_family = collections.Counter()
for line in a.ledger.open():
    row = json.loads(line)
    family = "no_op" if row["row_id"] in noop_ids else "roxygen_drafting"
    held_by_family[family] += 1
    for reason in row["reasons"]:
        by_reason[(reason, family)] += 1
    by_combo[(family, tuple(row["reasons"]))] += 1

review = json.loads(a.review.read_text())
out = {
    "schema": "sepalith.dat10.sourcewalk_sft_hold_family_summary.v1",
    "sourcewalk_rows": review["converted_audit"]["rows"],
    "held_union_by_family": dict(held_by_family),
    "frontier_by_family": review["converted_audit"]["mechanically_unique_frontier_by_family"],
    "reason_by_family": [
        {"reason": reason, "family": family, "rows": rows}
        for (reason, family), rows in sorted(by_reason.items())
    ],
    "exclusive_reason_combinations": [
        {"family": family, "reasons": list(reasons), "rows": rows}
        for (family, reasons), rows in sorted(by_combo.items())
    ],
    "interpretation": {
        "target_body_in_prompt": "leakage-suspicion hold pending occurrence review, not asserted contradiction",
        "prompt_conflict": "same rendered prompt token sequence with distinct target token sequence; actual label contradiction until resolved",
        "exact_duplicate": "deduplicate deterministically; does not invalidate the canonical row",
        "replaced_by_roxy10017_review_path": "older cropped-context row excluded from union; safe full-context row is the replacement when present",
    },
}
a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
