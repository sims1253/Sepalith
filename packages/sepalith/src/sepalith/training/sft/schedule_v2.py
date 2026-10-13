#!/usr/bin/env python3
"""Editing-SFT draw schedule v2: 13 edits plus 3 no-ops per update.

The plan (card C04) fixes the batch at 16 draws with a constant 13:3
composition. Every edit row appears once; only the last update may hold edit
replays (at most ``edit_per_batch - 1``). No-op rows are drawn in passes: each
pass draws every row of its pool once before the next pass starts, so a row's
exposures differ from any other row's by at most one. The final, partial pass
draws a seeded subset, which caps every no-op row at ``ceil(noop_draws /
noop_rows)`` exposures. For the 15,006-row cohort that is 1,071 updates,
3,213 no-op draws (18.75%), 1,025 no-op rows seen 3 times and 69 seen twice.

The output keeps the v1 schedule format (``sepalith.sft.full-coverage-batchmix-draws.v2``)
so ``campaign_edit_data.inspect`` and the trainer read it unchanged.
"""
from __future__ import annotations

import argparse
import collections
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path

from sepalith.training.sft.full_coverage_schedule import (
    bucket_for,
    denominator,
    edit_batch_queue,
    load_rows,
    ordered,
    stable,
)

Row = dict[str, object]

SCHEDULE_VERSION = "sft-schedule-v2-13edit-3noop"
EDIT_REPLAY = "edit_batch_completion_alignment_replay"
NOOP_REPLAY = "noop_pass_replay"


def gate_updates(updates: int, gates: int = 4) -> list[int]:
    """Gates at each quarter of the horizon, rounded half to even."""
    return [round(updates * index / gates) for index in range(1, gates + 1)]


def _int(row: Row, key: str) -> int:
    value = row[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _tokens(row: Row) -> int:
    return _int(row, "total_tokens")


def _closest(pool: list[Row], capacity: int, seed: int, label: str, slot: int) -> Row:
    """Closest length fit at or below the batch's longest edit, else closest overall."""
    fitting = [row for row in pool if _tokens(row) <= capacity] or pool
    return min(fitting, key=lambda row: (abs(capacity - _tokens(row)), stable(seed, label, slot, row["row_id"])))


class NoopPasses:
    """Draw no-op rows pass by pass with a seeded subset for the last pass."""

    def __init__(self, noops: list[Row], total_draws: int, seed: int) -> None:
        if not noops:
            raise ValueError("no-op pool is empty")
        self.noops, self.seed, self.total = noops, seed, total_draws
        self.passes = math.ceil(total_draws / len(noops))
        self.last_pass_rows = total_draws - (self.passes - 1) * len(noops)
        self.pass_index = 0
        self.remaining: list[Row] = []

    def _refill(self) -> None:
        self.pass_index += 1
        if self.pass_index > self.passes:
            raise AssertionError("no-op passes exhausted")
        if self.pass_index < self.passes:
            self.remaining = list(self.noops)
        else:
            subset = ordered(self.noops, self.seed, "noop-final-pass-subset")[: self.last_pass_rows]
            self.remaining = subset

    def draw(self, capacity: int, slot: int) -> tuple[Row, int]:
        if not self.remaining:
            self._refill()
        row = _closest(self.remaining, capacity, self.seed, f"noop-pass-{self.pass_index}", slot)
        self.remaining.remove(row)
        return row, self.pass_index


def build(rows: list[Row], excluded: list[Row], seed: int = 3407, effective_batch: int = 16,
          noop_per_batch: int = 3, buckets: Sequence[int] = (512, 1024, 2048, 4096, 8192, 16384)) -> dict[str, object]:
    if excluded:
        raise ValueError("schedule v2 keeps every eligible row; nothing may be excluded")
    edit_per = effective_batch - noop_per_batch
    if not 0 < noop_per_batch < effective_batch:
        raise ValueError("schedule v2 needs both edits and no-ops in every batch")
    noops = ordered([r for r in rows if r["semantic_noop"]], seed, "noop-unique")
    edits = ordered([r for r in rows if not r["semantic_noop"]], seed, "edit-unique")
    updates = math.ceil(len(edits) / edit_per)
    chunks = edit_batch_queue(edits, seed, edit_per)
    if len(chunks) != updates:
        raise AssertionError("edit chunks do not cover the horizon")
    passes = NoopPasses(noops, updates * noop_per_batch, seed)
    exposure: collections.Counter[object] = collections.Counter()
    draws: list[Row] = []
    reasons: collections.Counter[str] = collections.Counter()
    chunk_kinds: collections.Counter[str] = collections.Counter()
    batch_buckets: collections.Counter[str] = collections.Counter()
    for update, (chunk, chunk_kind) in enumerate(chunks, 1):
        chunk_kinds[chunk_kind] += 1
        selected: list[tuple[Row, str | None, int, str | None]] = []
        for row in chunk:
            exposure[row["row_id"]] += 1
            selected.append((row, None, exposure[row["row_id"]], chunk_kind))
        while len(selected) < edit_per:
            if update != updates:
                raise AssertionError("edit replays are allowed only in the last update")
            capacity = max(_tokens(r) for r, _, _, _ in selected)
            fresh = [r for r in edits if exposure[r["row_id"]] == 1 and r["row_id"] not in {s["row_id"] for s, _, _, _ in selected}]
            row = _closest(fresh, capacity, seed, EDIT_REPLAY, len(draws) + len(selected))
            exposure[row["row_id"]] += 1
            selected.append((row, EDIT_REPLAY, exposure[row["row_id"]], chunk_kind))
        capacity = max(_tokens(r) for r, _, _, _ in selected)
        for _ in range(noop_per_batch):
            row, pass_index = passes.draw(capacity, len(draws) + len(selected))
            exposure[row["row_id"]] += 1
            selected.append((row, NOOP_REPLAY if pass_index > 1 else None, exposure[row["row_id"]], None))
        # Within an update, lower presentations come first, so draw order never shows pass k+1 before pass k.
        selected.sort(key=lambda item: (item[2], stable(seed, "within-batch", update, item[0]["row_id"], item[2])))
        cap = bucket_for(max(_tokens(r) for r, _, _, _ in selected), list(buckets))
        batch_buckets[str(cap)] += 1
        for row, reason, presentation, row_chunk_kind in selected:
            if reason:
                reasons[reason] += 1
            draws.append({"draw_index": len(draws), "update": update, "row_id": row["row_id"], "family": row["family"],
                          "package_id": row["package_id"], "semantic_noop": row["semantic_noop"],
                          "presentation": presentation, "replay_reason": reason, "edit_chunk_kind": row_chunk_kind,
                          "sequence_bucket": row["sequence_bucket"], "batch_sequence_bucket": cap,
                          "prompt_tokens": row["prompt_tokens"], "supervised_target_tokens": row["supervised_target_tokens"],
                          "target_body_tokens": row["target_body_tokens"],
                          "target_protocol_tokens": row["target_protocol_tokens"], "total_tokens": row["total_tokens"]})
    if passes.remaining or passes.pass_index != passes.passes or set(exposure) != {r["row_id"] for r in rows}:
        raise AssertionError("full coverage not achieved")
    return _summary(rows, edits, noops, draws, exposure, reasons, chunk_kinds, batch_buckets,
                    seed=seed, effective_batch=effective_batch, noop_per_batch=noop_per_batch,
                    buckets=list(buckets), updates=updates, passes=passes)


def _summary(rows: list[Row], edits: list[Row], noops: list[Row], draws: list[Row],
             exposure: collections.Counter[object], reasons: collections.Counter[str],
             chunk_kinds: collections.Counter[str], batch_buckets: collections.Counter[str], *, seed: int,
             effective_batch: int, noop_per_batch: int, buckets: list[int], updates: int,
             passes: NoopPasses) -> dict[str, object]:
    first: dict[object, int] = {}
    first_replay: dict[str, int | None] = {"noop": None, "edit": None}
    for draw in draws:
        index = _int(draw, "draw_index")
        first.setdefault(draw["row_id"], index)
        pool = "noop" if draw["semantic_noop"] else "edit"
        if draw["presentation"] != 1 and first_replay[pool] is None:
            first_replay[pool] = index
    last_first = {"noop": max(first[r["row_id"]] for r in noops), "edit": max(first[r["row_id"]] for r in edits)}
    for pool, replay in first_replay.items():
        if replay is not None and replay <= last_first[pool]:
            raise AssertionError("pool replay preceded unique coverage")
    by_pool = {pool: dict(sorted(collections.Counter(exposure[r["row_id"]] for r in members).items()))
               for pool, members in (("edit", edits), ("noop", noops))}
    noop_draws = sum(1 for d in draws if d["semantic_noop"])
    per_update = collections.Counter(d["update"] for d in draws if d["semantic_noop"])
    first_draws = set(first.values())
    return {
        "schema": "sepalith.sft.full-coverage-batchmix-draws.v2",
        "status": "plan_c04_schedule_v2_bound_by_recipe",
        "policy": {
            "schedule_version": SCHEDULE_VERSION, "seed": seed, "effective_batch": effective_batch,
            "noop_per_batch": noop_per_batch, "edit_per_batch": effective_batch - noop_per_batch,
            "no_op_fraction": noop_per_batch / effective_batch, "sequence_buckets": buckets, "target_cap": None,
            "target_policy": "retain complete target body, protocol terminal, and terminal EOS whenever full sequence fits",
            "edit_batch_order": "deterministic seeded mixing of whole actual-length-bucket edit chunks; terminal partial unique chunk last",
            "edit_replay": "only the last update is padded with edit replays, each a distinct row seen once before",
            "noop_selection": "pass by pass: every no-op row once per pass, closest length fit within the pass; the last pass draws a seeded subset",
            "unique_before_replay": "no-op pass k+1 starts only after pass k drew every row of its pool; edit replays follow every unique edit",
            "max_noop_exposures": passes.passes,
            "replay_reasons": [NOOP_REPLAY, EDIT_REPLAY],
            "gate_updates": gate_updates(updates),
        },
        "max_steps": updates,
        "draw_count": len(draws),
        "row_ids": sorted(str(key) for key in exposure),
        "excluded": [],
        "coverage": {
            "eligible_rows": len(rows), "eligible_edit_rows": len(edits), "eligible_noop_rows": len(noops),
            "distinct_rows_drawn": len(exposure),
            "first_all_rows_step": max(_int(d, "update") for d in draws if d["draw_index"] in first_draws),
            "min_row_exposures": min(exposure.values()), "max_row_exposures": max(exposure.values()),
            "last_unique_draw_by_pool": last_first, "first_replay_draw_by_pool": first_replay,
            "replay_draws_by_reason": dict(sorted(reasons.items())),
            "edit_chunk_batches": dict(sorted(chunk_kinds.items())),
            "exposure_histogram": dict(sorted(collections.Counter(exposure.values()).items())),
            "exposure_histogram_by_pool": by_pool,
            "noop_passes": {"passes": passes.passes, "rows_per_pass": [len(noops)] * (passes.passes - 1) + [passes.last_pass_rows]},
            "noop_draws": noop_draws, "noop_share": noop_draws / len(draws),
            "noop_draws_per_update": sorted(set(per_update.values())),
            "edit_replays": reasons[EDIT_REPLAY],
        },
        "sequence_buckets": {
            "unique_rows": dict(sorted(collections.Counter(str(r["sequence_bucket"]) for r in rows).items())),
            "draws": dict(sorted(collections.Counter(str(d["sequence_bucket"]) for d in draws).items())),
            "batches": dict(sorted(batch_buckets.items())),
        },
        "token_denominators": {"unique": denominator(rows), "scheduled": denominator(draws)},
        "checks": {
            "all_eligible_ids_drawn": len(exposure) == len(rows),
            "no_target_truncation": all(
                _int(d, "supervised_target_tokens") == _int(d, "target_body_tokens") + _int(d, "target_protocol_tokens") + 1
                for d in draws),
            "targets_over_1024_retained": sum(1 for d in draws if _int(d, "supervised_target_tokens") > 1024),
            "every_unique_noop_before_noop_replay": first_replay["noop"] is None or first_replay["noop"] > last_first["noop"],
            "every_unique_edit_before_edit_replay": first_replay["edit"] is None or first_replay["edit"] > last_first["edit"],
            "edits_seen_once_except_last_update_replays": by_pool["edit"].get(1, 0) + by_pool["edit"].get(2, 0) == len(edits)
            and by_pool["edit"].get(2, 0) <= effective_batch - noop_per_batch - 1,
            "noop_exposures_at_most_cap": max(by_pool["noop"]) <= passes.passes,
            "constant_batch_composition": sorted(set(per_update.values())) == [noop_per_batch] and len(per_update) == updates,
        },
        "draws": draws,
    }


def write_schedule(rows_path: Path, rows_sha256: str, output: Path, *, seed: int = 3407, effective_batch: int = 16,
                   noop_per_batch: int = 3, max_sequence: int = 16384) -> dict[str, object]:
    if output.exists():
        raise ValueError("fresh output required")
    buckets = sorted({value for value in (512, 1024, 2048, 4096, 8192, 16384) if value < max_sequence} | {max_sequence})
    rows, excluded = load_rows(rows_path, rows_sha256, buckets)
    result = build(rows, excluded, seed, effective_batch, noop_per_batch, buckets)
    row_ids = result["row_ids"]
    if not isinstance(row_ids, list):
        raise TypeError("row_ids must be a list")
    result["input"] = {"path": str(rows_path.resolve()), "sha256": rows_sha256, "rows_seen": len(rows) + len(excluded),
                       "eligible_row_ids_sha256": hashlib.sha256(("\n".join(str(x) for x in row_ids) + "\n").encode()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rows-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--noop-per-batch", type=int, default=3)
    parser.add_argument("--max-sequence", type=int, default=16384)
    args = parser.parse_args(argv)
    result = write_schedule(args.rows, args.rows_sha256, args.output, seed=args.seed, effective_batch=args.effective_batch,
                            noop_per_batch=args.noop_per_batch, max_sequence=args.max_sequence)
    print(json.dumps({"output": str(args.output), "updates": result["max_steps"], "draws": result["draw_count"],
                      "coverage": result["coverage"], "checks": result["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
