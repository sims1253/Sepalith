#!/usr/bin/env python3
"""Materialize the expanded SFT draw order with a coverage floor.

The schedule deliberately covers every admitted row once before deterministic
refill.  Refill is split between the semantic no-op and edit pools so the
16,000-draw optimizer schedule contains exactly 25% no-op draws.  This is a
schedule builder, not a data admission step: the input rows must already be
the root-reviewed, immutable candidate file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any


SPLIT_ID = "DAT-02-global-v2-285001f3d93e9f1871df"
TOKEN_ROWS_SHA256 = "fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889"
PROVENANCE_SHA256 = "317077b17afc38602309361396bb86d8897a6b823c5d7539f10b9733a0ad9a48"
MAX_STEPS = 1000
EFFECTIVE_BATCH = 16
DRAW_COUNT = MAX_STEPS * EFFECTIVE_BATCH
NOOP_COUNT = 1094
EDIT_COUNT = 10411
NOOP_SLOTS = DRAW_COUNT // 4
EDIT_SLOTS = DRAW_COUNT - NOOP_SLOTS
SEED = 3407


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _row_metadata(row: dict[str, Any], provenance: dict[str, Any], presentation: int) -> dict[str, Any]:
    ids = row["input_ids"]
    target_tokens = len(ids) - row["target_start"]
    return {
        "draw_index": 0,
        "family": row["family"],
        "length_bucket": "long" if len(ids) > 2048 else "short",
        "naturally_long": len(ids) > 2048,
        "package_id": row["package_id"],
        "presentation": presentation,
        "prompt_tokens": row["target_start"] - 1,
        "row_id": row["id"],
        "semantic_noop": bool(row.get("semantic_noop", row["family"] == "no_op")),
        # Token rows intentionally carry no group field.  Rebind it from the
        # frozen provenance sidecar rather than inventing an ID or grouping by
        # row text.
        "group_id": provenance["group_id"],
        "source_id": provenance["group_id"],
        "source_identity": f"{row['package_id']}::{provenance['group_id']}",
        "source_kind": provenance.get("source_class", "ordinary"),
        "split": row["split"],
        "target_tokens": target_tokens,
        "total_tokens": len(ids),
    }


def build(candidate: Path, provenance_file: Path, output: Path) -> dict[str, Any]:
    if digest(candidate) != TOKEN_ROWS_SHA256:
        raise ValueError("candidate bytes do not match the root-reviewed expanded row digest")
    if digest(provenance_file) != PROVENANCE_SHA256:
        raise ValueError("provenance bytes do not match the root-reviewed expanded provenance digest")
    provenance: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(provenance_file.open(), 1):
        item = json.loads(line)
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or ident in provenance:
            raise ValueError(f"missing or duplicate provenance identity at line {line_number}")
        if not isinstance(item.get("group_id"), str) or not item["group_id"]:
            raise ValueError(f"provenance {ident} has no exact group_id")
        provenance[ident] = item
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(candidate.open(), 1):
        row = json.loads(line)
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in rows:
            raise ValueError(f"missing or duplicate row identity at line {line_number}")
        if ident not in provenance:
            raise ValueError(f"row {ident} has no matching provenance sidecar entry")
        if row.get("split") != "train" or row.get("renderer_id") != "zeta2-prm03-v1":
            raise ValueError(f"row {ident} is not bound to the task TRAIN renderer/split")
        ids = row.get("input_ids")
        start = row.get("target_start")
        if (not isinstance(ids, list) or type(start) is not int
                or len(ids) > 4096 or len(ids) - start > 1024):
            raise ValueError(f"row {ident} violates the full sequence or TRAIN target bound")
        rows[ident] = row
    if len(rows) != 11505:
        raise ValueError(f"expected 11505 reviewed rows, found {len(rows)}")

    noops = [ident for ident, row in rows.items()
             if bool(row.get("semantic_noop", row["family"] == "no_op"))]
    edits = [ident for ident, row in rows.items()
             if not bool(row.get("semantic_noop", row["family"] == "no_op"))]
    if (len(noops), len(edits)) != (NOOP_COUNT, EDIT_COUNT):
        raise ValueError(f"expected {NOOP_COUNT}/{EDIT_COUNT} no-op/edit rows, got {len(noops)}/{len(edits)}")

    noop_rng = random.Random(SEED ^ 0x4E4F4F50)
    edit_rng = random.Random(SEED ^ 0x45444954)
    noop_rng.shuffle(noops)
    edit_rng.shuffle(edits)
    # Four no-op and twelve edit slots are selected independently in every
    # optimizer batch.  The two shuffled pools cycle only after their full
    # pool has been consumed, so every row is seen before any pool repeat.
    row_ids: list[str] = []
    noop_cursor = edit_cursor = 0
    pattern_rng = random.Random(SEED)
    for _step in range(MAX_STEPS):
        pattern = ["noop"] * 4 + ["edit"] * 12
        pattern_rng.shuffle(pattern)
        for slot in pattern:
            if slot == "noop":
                row_ids.append(noops[noop_cursor % len(noops)])
                noop_cursor += 1
            else:
                row_ids.append(edits[edit_cursor % len(edits)])
                edit_cursor += 1
    if len(row_ids) != DRAW_COUNT:
        raise AssertionError("schedule has the wrong draw count")

    presentations: Counter[str] = Counter()
    draws = []
    for index, ident in enumerate(row_ids):
        presentations[ident] += 1
        item = _row_metadata(rows[ident], provenance[ident], presentations[ident])
        item["draw_index"] = index
        draws.append(item)
    exposure = Counter(row_ids)
    if set(exposure) != set(rows) or min(exposure.values()) < 1:
        raise AssertionError("coverage floor failed")
    if sum(item["semantic_noop"] for item in draws) != NOOP_SLOTS:
        raise AssertionError("no-op quota failed")
    for offset in range(0, DRAW_COUNT, EFFECTIVE_BATCH):
        if sum(item["semantic_noop"] for item in draws[offset:offset + EFFECTIVE_BATCH]) != 4:
            raise AssertionError("every optimizer batch must contain exactly four no-op draws")

    family_counts = Counter(item["family"] for item in draws)
    noop_draws = sum(item["semantic_noop"] for item in draws)
    first_all_rows_step = next(
        step for step in range(1, MAX_STEPS + 1)
        if len(set(row_ids[:step * EFFECTIVE_BATCH])) == len(rows)
    )
    prefix = []
    for step in (250, 500, 750, 1000):
        end = step * EFFECTIVE_BATCH
        prefix_rows = row_ids[:end]
        counts = Counter(prefix_rows)
        prefix.append({
            "step": step,
            "draws": end,
            "distinct_rows": len(counts),
            "semantic_noop_draws": sum(draws[i]["semantic_noop"] for i in range(end)),
            "semantic_noop_fraction": sum(draws[i]["semantic_noop"] for i in range(end)) / end,
            "all_rows_covered": len(counts) == len(rows),
            "families": dict(sorted(Counter(draws[i]["family"] for i in range(end)).items())),
        })

    schedule = {
        "schema_version": "sepalith.dat10.expanded-sampler.v1",
        "status": "complete;coverage_floor_then_exact_noop_refill",
        "split_id": SPLIT_ID,
        "token_rows_sha256": TOKEN_ROWS_SHA256,
        "max_steps": MAX_STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "draw_count": DRAW_COUNT,
        "row_ids": row_ids,
        "draws": draws,
        "policy": {
            "seed": SEED,
            "noop_fraction": 0.25,
            "noop_slots": NOOP_SLOTS,
            "edit_slots": EDIT_SLOTS,
            "admitted_rows": len(rows),
            "coverage_floor": "each independently shuffled no-op/edit pool is exhausted before that pool repeats",
            "refill": "independently shuffled no-op/edit pools, four/twelve slots per batch",
            "replay_is_not_prefix": True,
            "batch_noop_slots": 4,
            "batch_edit_slots": 12,
            "family_quota": "none; every admitted family row remains eligible",
            "truncation": "forbidden",
        },
        "coverage": {
            "rows": len(rows),
            "distinct_rows_drawn": len(exposure),
            "min_row_exposures": min(exposure.values()),
            "max_row_exposures": max(exposure.values()),
            "rows_with_one_exposure": sum(value == 1 for value in exposure.values()),
            "rows_with_repeated_exposure": sum(value > 1 for value in exposure.values()),
            "no_op_rows": len(noops),
            "edit_rows": len(edits),
            "no_op_draws": noop_draws,
            "edit_draws": DRAW_COUNT - noop_draws,
            "no_op_draw_fraction": noop_draws / DRAW_COUNT,
            "first_all_rows_step": first_all_rows_step,
            "family_draws": dict(sorted(family_counts.items())),
        },
        "milestone_prefixes": prefix,
        "provenance_sha256": PROVENANCE_SHA256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    schedule = build(args.candidate, args.provenance, args.output)
    print(json.dumps({
        "status": schedule["status"],
        "schedule": str(args.output),
        "sha256": digest(args.output),
        "rows": schedule["coverage"]["rows"],
        "draws": schedule["draw_count"],
        "no_op_draws": schedule["coverage"]["no_op_draws"],
        "first_all_rows_step": schedule["coverage"]["first_all_rows_step"],
        "min_row_exposures": schedule["coverage"]["min_row_exposures"],
        "max_row_exposures": schedule["coverage"]["max_row_exposures"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
