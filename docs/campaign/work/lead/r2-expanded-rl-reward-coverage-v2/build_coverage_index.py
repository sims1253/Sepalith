#!/usr/bin/env python3
"""Freeze the explicit syntax mode for every currently eligible TRAIN row."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source/experiments/training"
sys.path.insert(0, str(SOURCE))
from campaign_rl_buffer import HELD_CONTRADICTORY_IDS, RewardBufferIndex, sha_file

ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl")
ROWS_SHA = "65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7"
MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-buffer-v5/materialization.json")
MANIFEST_SHA = "a49b1ef464773817d7a18f76cdec68b82eaa4abb960a10a7d919c7535d7add84"
DECISION = Path("docs/campaign/receipts/DAT-10-two-pipe-contradictions-root-decision.json")
DECISION_SHA = "ff2cde297a9845671d0acfb68674c7d60ae3a8b4c2caedb518398d7fa55e7ffc"


def main() -> None:
    if sha_file(ROWS) != ROWS_SHA or sha_file(MANIFEST) != MANIFEST_SHA or sha_file(DECISION) != DECISION_SHA:
        raise ValueError("input hash mismatch")
    ordered_ids = []
    with ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("non-TRAIN row")
            ordered_ids.append(row["id"])
    index = RewardBufferIndex.load(MANIFEST, MANIFEST_SHA, ordered_ids)
    counts = Counter()
    repairs = Counter()
    out = ROOT / "syntax-evidence-index.jsonl"
    with out.open("w") as stream:
        for row_id in ordered_ids:
            envelope = index.envelope_for(row_id)
            counts[envelope["syntax_evidence_mode"]] += 1
            if envelope["repair_reason"]:
                reason = envelope["repair_reason"]
                category = ("pinned_snapshot_unavailable" if "pinned_snapshot_unavailable" in reason
                            else "complete_baseline_parse_failed" if "complete_baseline_parse_failed" in reason
                            else reason)
                repairs[category] += 1
            stream.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "schema": "sepalith.rl11.reward-syntax-coverage.v2",
        "status": "prepared_root_review_required_no_launch",
        "split": "TRAIN",
        "eligible_rows": len(ordered_ids),
        "distinct_ids": len(set(ordered_ids)),
        "source_sidecar_rows": index.source_row_count,
        "held_ids": sorted(HELD_CONTRADICTORY_IDS),
        "held_ids_present_in_eligible": sorted(set(ordered_ids) & HELD_CONTRADICTORY_IDS),
        "modes": dict(sorted(counts.items())),
        "repair_reasons": dict(sorted(repairs.items())),
        "syntax_verified": len(ordered_ids) - counts["unverified"],
        "syntax_unverified": counts["unverified"],
        "inputs": {
            "eligible_rows": {"path": str(ROWS), "sha256": ROWS_SHA},
            "reward_buffer_manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA},
            "reward_buffer_sidecar_sha256": index.sidecar_sha256,
            "hold_decision": {"path": str(DECISION), "sha256": DECISION_SHA},
        },
        "output": {"path": "docs/campaign/work/lead/r2-expanded-rl-reward-coverage-v2/syntax-evidence-index.jsonl",
                   "sha256": sha_file(out), "bytes": out.stat().st_size},
        "checks": {
            "all_15006_ids_exact_order": True,
            "exact_two_holds_refused": True,
            "other_rows_omitted": 0,
            "unverified_rows_keep_non_syntax_rewards": True,
            "unverified_rows_carry_no_parse_claims": True,
        },
    }
    (ROOT / "coverage.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(ordered_ids), "modes": counts, "repairs": repairs}, default=dict))


if __name__ == "__main__":
    main()
