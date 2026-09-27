#!/usr/bin/env python3
"""Merge the two disjoint held-denominator recovery replays.

The merger verifies coverage and disjointness before writing a source-only
ledger. It does not read or emit source/target payload text and cannot admit
TRAIN rows.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-roxy-full-context-recovery-v1"
PRIOR = PLAN / "docs/campaign/work/lead/r2-roxy-context-admission-review-v1/full/audited-rows.jsonl"
SHARDS = (PACKET / "materialization-shard-0000", PACKET / "materialization-shard-0001")
# v2 is a superseded interrupted single-process replay and v3 is the first
# merged ledger.  Keep this final merge in a fresh directory so no partial or
# pre-reason output can be mistaken for the authoritative result.
OUT = PACKET / "materialization-v4"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recovery_reason(status: str) -> str:
    if status == "recoverable_by_scope_correction":
        return "prior free-reference hold removed by current nearest-function scope/member-label replay; root semantic, license, dedup, and API review remains required"
    if status == "recoverable_ordinary_r_or_imported_reference_pending_root_api_review":
        return "all remaining free names have standard-R, exact-import, package-namespace, or observed-call evidence; root API/package review remains required"
    if status == "hold_residual_noncall_reference_repair_or_semantic_evidence":
        return "residual non-call free references remain after current scope/member replay; retain for source repair or semantic evidence"
    raise ValueError("unknown_recovery_status:" + status)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("merged_output_must_be_fresh")
    prior = load_jsonl(PRIOR)
    prior_held = {
        str(row["row_id"])
        for row in prior
        if row.get("recommendation") == "hold_true_omitted_or_external_global_evidence"
    }
    if len(prior) != 10017 or len(prior_held) != 5942:
        raise ValueError("prior_scope_mismatch")
    all_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    shard_ids: list[set[str]] = []
    for shard in SHARDS:
        summary_path = shard / "recovery-summary.json"
        ledger_path = shard / "recovery-ledger.jsonl"
        errors_path = shard / "recovery-errors.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = load_jsonl(ledger_path)
        errors = load_jsonl(errors_path)
        if summary.get("status") != "complete" or summary.get("error_rows") != 0 or errors:
            raise ValueError("incomplete_or_error_shard:" + str(shard))
        ids = [str(row["row_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_rows_in_shard:" + str(shard))
        if set(ids) - prior_held:
            raise ValueError("shard_row_outside_held_denominator:" + str(shard))
        all_rows.extend(rows)
        shard_summaries.append(summary)
        shard_ids.append(set(ids))
    if shard_ids[0] & shard_ids[1]:
        raise ValueError("shard_overlap")
    merged_ids = shard_ids[0] | shard_ids[1]
    if merged_ids != prior_held or len(all_rows) != len(prior_held):
        raise ValueError(f"coverage_mismatch:{len(merged_ids)}:{len(prior_held)}")
    all_rows.sort(key=lambda row: row["row_id"])
    for row in all_rows:
        row["recovery_reason"] = recovery_reason(str(row["recovery_status"]))
    counts = Counter(str(row["recovery_status"]) for row in all_rows)
    classes: Counter[str] = Counter()
    for row in all_rows:
        classes.update(row.get("free_reference_class_counts", {}))
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "recovery-ledger.jsonl", all_rows)
    errors_path = OUT / "recovery-errors.jsonl"
    write_jsonl(errors_path, [])
    recoverable = sorted(
        str(row["row_id"])
        for row in all_rows
        if row["recovery_status"] != "hold_residual_noncall_reference_repair_or_semantic_evidence"
    )
    held = sorted(
        str(row["row_id"])
        for row in all_rows
        if row["recovery_status"] == "hold_residual_noncall_reference_repair_or_semantic_evidence"
    )
    write_json(OUT / "recoverable-ids.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_ids.v1",
        "status": "recommendation_only_not_training_admission",
        "scope": "prior_5942_held_rows",
        "recoverable_count": len(recoverable),
        "recoverable_ids": recoverable,
    })
    write_json(OUT / "held-ids.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_held_ids.v1",
        "status": "repair_or_semantic_evidence_required",
        "scope": "prior_5942_held_rows",
        "held_count": len(held),
        "held_ids": held,
    })
    representatives: list[dict[str, Any]] = []
    for status in sorted(counts):
        representatives.extend([row for row in all_rows if row["recovery_status"] == status][:5])
    write_json(OUT / "representative-evidence.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_representatives.v1",
        "status": "review_only_no_source_or_target_text",
        "rows": representatives,
    })
    summary = {
        "schema": "sepalith.dat10.roxy_full_context_recovery_summary.v1",
        "status": "complete",
        "prior_held_denominator": len(prior_held),
        "rows_audited": len(all_rows),
        "error_rows": 0,
        "recoverable_count": len(recoverable),
        "residual_held_count": len(held),
        "recovery_status_counts": dict(sorted(counts.items())),
        "free_reference_class_counts": dict(sorted(classes.items())),
        "full_file_within_131072_rows": sum(bool(row["full_file_within_131072"]) for row in all_rows),
        "full_file_fallback_useful_rows": sum(bool(row["full_file_fallback_useful"]) for row in all_rows),
        "same_file_definition_recovery_rows": sum(bool(row["same_file_top_level_names"]) for row in all_rows),
        "source_files_seen": len({row["source_path"] for row in all_rows}),
        "package_metadata_roots_seen": len({row["namespace_import_evidence"].get("package_root") for row in all_rows}),
        "target_gt_1024_rows": sum(bool(row["target_body_tokens"] > 1024) for row in all_rows),
        "semantic_support_claims": 0,
        "source_text_written": False,
        "target_text_written": False,
        "training_admission": False,
        "prior_review_outputs_unchanged": True,
        "coverage": {
            "shards": [
                {
                    "path": str(shard),
                    "rows": len(ids),
                    "ledger_sha256": sha_file(shard / "recovery-ledger.jsonl"),
                    "summary_sha256": sha_file(shard / "recovery-summary.json"),
                }
                for shard, ids in zip(SHARDS, shard_ids)
            ],
            "disjoint": True,
            "union_equals_prior_held": True,
        },
        "input_pins": {
            "prior_full_audited_rows": {"path": str(PRIOR), "rows": len(prior), "sha256": sha_file(PRIOR)},
            "merge_script": {"path": str(Path(__file__)), "sha256": sha_file(Path(__file__))},
        },
    }
    write_json(OUT / "recovery-summary.json", summary)
    write_json(OUT / "status.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_status.v1",
        "status": "complete",
        "scope_rows": len(all_rows),
        "rows_audited": len(all_rows),
        "error_rows": 0,
        "recoverable_count": len(recoverable),
        "residual_held_count": len(held),
        "cpu_threads_max": 2,
        "cuda": False,
        "training_admission": False,
        "source_text_written": False,
        "target_text_written": False,
        "prior_review_outputs_unchanged": True,
        "summary_path": str(OUT / "recovery-summary.json"),
    })


if __name__ == "__main__":
    main()
