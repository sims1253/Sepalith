#!/usr/bin/env python3
"""Create the deterministic all-row CPT draw schedule after 16K rechunking.

This reads only the generated 16K token-row stream. It never reads source
files, the 2K union, DEV/final inputs, or model data. The first R draws are a
seeded permutation of every unique row; only the minimal named tail replay
needed for effective batch 16 is appended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

BATCH = 16
CONTEXT = 16384
RESULT_SCHEMA = "sepalith.cpt.lossless-rechunk-result.v1"
SCHEDULE_SCHEMA = "sepalith.sft11.cpt-draw-schedule.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def make_schedule(rows: Path, rechunk_result: Path, output: Path, *, seed: int, split_id: str) -> dict[str, Any]:
    rows_arg = Path(rows)
    result_arg = Path(rechunk_result)
    output_arg = Path(output)
    require(rows_arg.is_file() and not rows_arg.is_symlink(), f"rows_missing_or_symlink:{rows_arg}")
    require(result_arg.is_file() and not result_arg.is_symlink(), f"rechunk_result_missing_or_symlink:{result_arg}")
    require(not output_arg.exists() and not output_arg.is_symlink(), f"schedule_output_must_be_fresh:{output_arg}")
    rows = rows_arg.resolve()
    rechunk_result = result_arg.resolve()
    output = output_arg.resolve()
    result = json.loads(rechunk_result.read_text(encoding="utf-8"))
    require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "complete", "rechunk_result_not_complete")
    output_stats = result.get("outputs", {}).get(str(CONTEXT))
    artifact = result.get("artifacts", {}).get(f"cpt_train_ctx{CONTEXT}.jsonl")
    require(isinstance(output_stats, dict) and isinstance(artifact, dict), "rechunk_16k_artifact_missing")
    totals = result.get("totals", {})
    require(type(output_stats.get("rows")) is int and output_stats["rows"] > 0, "rechunk_16k_rows_invalid")
    require(output_stats.get("payload_tokens") == totals.get("payload_tokens"), "rechunk_16k_payload_not_conserved")
    require(output_stats.get("terminal_eos") == totals.get("documents"), "rechunk_16k_terminal_coverage_invalid")
    row_ids: list[str] = []
    seen: set[str] = set()
    rows_stat_before = rows.stat()
    digest = hashlib.sha256()
    with rows.open("rb", buffering=4 * 1024 * 1024) as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            require(raw.endswith(b"\n"), f"rows_final_line_missing_newline:{line_number}")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"rows_json_invalid:{line_number}") from exc
            require(isinstance(row, dict), f"rows_not_object:{line_number}")
            require(row.get("schema") == 1 and row.get("cpt_partition") == "cpt_train", f"row_partition_or_schema_invalid:{line_number}")
            row_id = row.get("row_id")
            require(isinstance(row_id, str) and row_id, f"row_id_invalid:{line_number}")
            require(row_id not in seen, f"duplicate_row_id:{row_id}")
            seen.add(row_id)
            row_ids.append(row_id)
    rows_stat_after = rows.stat()
    require((rows_stat_before.st_dev, rows_stat_before.st_ino, rows_stat_before.st_size, rows_stat_before.st_mtime_ns) == (rows_stat_after.st_dev, rows_stat_after.st_ino, rows_stat_after.st_size, rows_stat_after.st_mtime_ns), "rechunk_16k_rows_changed_during_schedule_read")
    actual_sha = digest.hexdigest()
    require(actual_sha == artifact.get("sha256") and rows_stat_after.st_size == artifact.get("bytes"), "rechunk_16k_artifact_hash_mismatch")
    require(len(row_ids) == output_stats.get("rows"), "rechunk_16k_observed_row_count_mismatch")
    require(row_ids, "rechunk_16k_rows_empty")
    original_order = list(row_ids)
    random.Random(seed).shuffle(row_ids)
    replay_count = (-len(row_ids)) % BATCH
    replay = row_ids[:replay_count]
    draw_ids = row_ids + replay
    require(len(draw_ids) % BATCH == 0, "draws_not_batch_aligned")
    require(len(set(draw_ids[:len(original_order)])) == len(original_order), "unique_first_draw_contract_failed")
    schedule = {
        "schema": SCHEDULE_SCHEMA,
        "status": "complete_candidate_pending_root_admission",
        "split_id": split_id,
        "method": "one_pass_plus_named_replay_v1",
        "seed": seed,
        "effective_batch": BATCH,
        "max_steps": len(draw_ids) // BATCH,
        "token_rows_sha256": actual_sha,
        "row_ids": draw_ids,
        "replay_count": replay_count,
        "replay_row_ids": replay,
        "coverage": {
            "unique_rows": len(original_order),
            "first_unique_draw_position_exclusive": len(original_order),
            "draws": len(draw_ids),
            "updates": len(draw_ids) // BATCH,
            "all_unique_rows_before_replay": True,
            "all_final_union_rows_required": True,
            "minimal_named_tail_replay": True,
        },
        "source": {
            "rows": {"path": str(rows), "bytes": rows.stat().st_size, "sha256": actual_sha},
            "lossless_rechunk_result": {"path": str(rechunk_result), "sha256": sha256(rechunk_result)},
            "context_tokens": CONTEXT,
        },
        "stage_transition": {
            "destination_sampler": "fresh_cursor_zero",
            "initial_stage_cursor": 0,
            "global_optimizer_step_offset": 66,
            "global_max_step": 66 + len(draw_ids) // BATCH,
        },
        "training_admission": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    require(not temporary.exists(), f"schedule_temp_must_be_fresh:{temporary}")
    temporary.write_text(json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size, "counts": schedule["coverage"], "split_id": split_id, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rechunk-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--split-id", default="cpt_train_final_union_ctx16384_v1")
    args = parser.parse_args()
    print(json.dumps(make_schedule(args.rows, args.rechunk_result, args.output, seed=args.seed, split_id=args.split_id), sort_keys=True))


if __name__ == "__main__":
    main()
