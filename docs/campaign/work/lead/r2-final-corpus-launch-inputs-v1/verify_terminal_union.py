#!/usr/bin/env python3
"""Check final-union terminal metadata without opening the bulk row stream.

The final-union builder writes its manifest only after all source streams have
been consumed. This gate checks that manifest and its small input manifest;
it deliberately never opens cpt_train.jsonl or any other payload stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "sepalith.dat10.cpt_final_union_candidate.v1"
EXPECTED_INPUT_SCHEMA = "sepalith.cpt.lossless-rechunk-input.v1"
EXPECTED_GROUPS = 8092
EXPECTED_ALIAS_CAPTURE = 8092
EXPECTED_MAIN_END = 8867


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def regular_file(path: Path, label: str) -> None:
    require(path.is_file(), f"{label}_missing:{path}")
    require(not path.is_symlink(), f"{label}_symlink:{path}")


def verify(union: Path, report: Path | None = None) -> dict[str, Any]:
    union_arg = Path(union)
    require(union_arg.is_dir() and not union_arg.is_symlink(), f"union_directory_missing_or_symlink:{union_arg}")
    union = union_arg.resolve()
    manifest_path = union / "manifest.json"
    input_manifest_path = union / "input-manifest.json"
    source_manifest_path = union / "source-manifest.jsonl"
    regular_file(manifest_path, "union_manifest")
    regular_file(input_manifest_path, "union_input_manifest")
    regular_file(source_manifest_path, "union_source_manifest")
    row_path = union / "cpt_train.jsonl"
    regular_file(row_path, "union_rows")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == EXPECTED_SCHEMA, "union_schema_mismatch")
    require(manifest.get("mode") == "final", "union_mode_not_final")
    require(manifest.get("status") == "complete_candidate_pending_root_global_dedup_review_and_training_admission", "union_status_not_complete")
    require(manifest.get("training_admission") is False, "union_already_claims_training_admission")
    require(manifest.get("truncation") is False and manifest.get("retokenized") is False, "union_transform_contract_changed")
    require(manifest.get("heldout_content_read") is False, "union_heldout_content_read")
    main_gate = manifest.get("main_terminal_gate")
    require(isinstance(main_gate, dict), "main_terminal_gate_missing")
    main_progress = main_gate.get("progress")
    require(isinstance(main_progress, dict), "main_terminal_progress_missing")
    require(main_progress.get("groups_committed") == EXPECTED_GROUPS, "main_groups_not_terminal")
    require(main_progress.get("groups_remaining") == 0, "main_groups_remaining")
    require(main_progress.get("first_uncommitted_seeded_index") in (None, EXPECTED_MAIN_END), "main_frontier_not_terminal")
    require(main_progress.get("status") in {"complete", "all_groups_inventoried_repairs_pending"}, "main_status_not_terminal")
    alias_gate = manifest.get("alias_terminal_capture_gate")
    require(isinstance(alias_gate, dict), "alias_terminal_gate_missing")
    alias_progress = alias_gate.get("progress")
    require(isinstance(alias_progress, dict), "alias_terminal_progress_missing")
    require(alias_progress.get("captured_main_groups") == EXPECTED_ALIAS_CAPTURE, "alias_capture_not_terminal")
    require(alias_progress.get("admitted") is not True, "alias_producer_claims_admission")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "union_artifacts_missing")
    row_artifact = artifacts.get("cpt_train.jsonl")
    require(isinstance(row_artifact, dict), "union_rows_artifact_missing")
    require(type(row_artifact.get("bytes")) is int and row_artifact["bytes"] > 0, "union_rows_bytes_missing")
    require(isinstance(row_artifact.get("sha256"), str) and len(row_artifact["sha256"]) == 64, "union_rows_sha_missing")
    require(manifest.get("source_manifest") == str(source_manifest_path), "union_source_manifest_pointer_mismatch")
    require(manifest.get("source_manifest_sha256") == sha256(source_manifest_path), "union_source_manifest_hash_mismatch")
    require(manifest.get("input_manifest") == str(input_manifest_path), "union_input_manifest_pointer_mismatch")
    require(manifest.get("input_manifest_sha256") == sha256(input_manifest_path), "union_input_manifest_hash_mismatch")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    require(input_manifest.get("schema") == EXPECTED_INPUT_SCHEMA, "rechunk_input_schema_mismatch")
    require(input_manifest.get("status") == "final_candidate_pending_root_admission", "rechunk_input_status_mismatch")
    require(input_manifest.get("context_sizes") == [16384], "rechunk_context_must_be_16384")
    require(input_manifest.get("training_admission") is False, "rechunk_input_claims_admission")
    require(input_manifest.get("truncation") is False, "rechunk_input_allows_truncation")
    require(input_manifest.get("tokenizer", {}).get("retokenized") is False, "rechunk_input_allows_retokenization")
    inputs = input_manifest.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 1, "rechunk_input_source_count_mismatch")
    source = inputs[0]
    require(source.get("path") == str(union / "cpt_train.jsonl"), "rechunk_input_rows_pointer_mismatch")
    require(source.get("sha256") == row_artifact["sha256"] and source.get("bytes") == row_artifact["bytes"], "rechunk_input_rows_pin_mismatch")
    counts = manifest.get("counts", {})
    expected = input_manifest.get("expected_totals", {})
    require(expected == {"rows": counts.get("rows"), "documents": counts.get("documents"), "payload_tokens": counts.get("payload_tokens")}, "rechunk_totals_do_not_match_union")
    result = {
        "schema": "sepalith.dat10.final-union-terminal-gate.v1",
        "status": "pass",
        "union": str(union),
        "union_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "union_input_manifest": {"path": str(input_manifest_path), "sha256": sha256(input_manifest_path)},
        "source_manifest": {"path": str(source_manifest_path), "sha256": sha256(source_manifest_path)},
        "row_artifact": {"path": str(row_path), "bytes": source["bytes"], "sha256": source["sha256"]},
        "counts": counts,
        "gates": {
            "main_groups_committed": main_progress["groups_committed"],
            "main_groups_remaining": main_progress["groups_remaining"],
            "main_first_uncommitted_seeded_index": main_progress.get("first_uncommitted_seeded_index"),
            "alias_captured_main_groups": alias_progress["captured_main_groups"],
        },
        "bulk_payload_opened": False,
        "training_admission": False,
    }
    if report is not None:
        report = Path(report)
        require(str(report.resolve()).startswith("/mnt/e/"), "terminal_gate_report_must_be_on_E")
        require(not report.exists(), f"terminal_gate_report_must_be_fresh:{report}")
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_name("." + report.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, report)
        result["report"] = {"path": str(report), "sha256": sha256(report)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.union, args.report), sort_keys=True))


if __name__ == "__main__":
    main()
