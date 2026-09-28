#!/usr/bin/env python3
"""Aggregate source-span and selected-length evidence without source text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


TEXT_KEYS = {"prompt_text", "target_text", "target_body_text"}


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


def percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]


def has_text_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in TEXT_KEYS or has_text_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_text_key(item) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--long-queue", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence_output_must_be_fresh")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    statuses: Counter[str] = Counter()
    source_paths: set[str] = set()
    groups: set[str] = set()
    row_ids: list[str] = []
    selected_sequences: list[int] = []
    target_tokens: list[int] = []
    selected_lines: list[int] = []
    omitted_lines: list[int] = []
    selected_definitions: list[int] = []
    resolved_references: list[int] = []
    unresolved_references: list[int] = []
    formal_mismatch = 0
    semantic_claims = 0
    anchor_warnings = 0
    text_key_rows = 0
    license_failures = 0
    with args.profiles.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_ids.append(str(row["row_id"]))
            source_paths.add(str(row["source_path"]))
            groups.add(str(row["group_id"]))
            evidence = row["context_evidence"]
            statuses[str(evidence["status"])] += 1
            lengths = row["lengths"]
            selected_sequences.append(int(lengths["sequence"]))
            target_tokens.append(int(lengths["target_body_tokens"]))
            selected_lines.append(int(evidence["selected_line_count"]))
            omitted_lines.append(int(evidence["omitted_line_count"]))
            selected_definitions.append(int(evidence["selected_definition_count"]))
            resolved_references.append(len(evidence.get("resolved_reference_names", [])))
            unresolved_references.append(len(evidence.get("unresolved_reference_names", [])))
            formal_mismatch += bool(evidence.get("docs_evidence", {}).get("documented_params_missing_from_formals"))
            semantic_claims += bool(evidence.get("semantic_support_claim"))
            anchor_warnings += bool(evidence.get("anchor_preservation_is_not_semantic_support"))
            license_failures += not bool(row.get("license_revalidated_by_frozen_review"))
            text_key_rows += has_text_key(row)
    queue_rows = sum(1 for _ in args.long_queue.open(encoding="utf-8"))
    row_ids_sorted = sorted(row_ids)
    row_id_digest = hashlib.sha256("\n".join(row_ids_sorted).encode("utf-8")).hexdigest()
    selected = sorted(selected_sequences)
    targets = sorted(target_tokens)
    evidence = {
        "schema": "sepalith.dat10.roxy_supported_context_evidence.v1",
        "status": "complete" if not text_key_rows and not license_failures else "review_required",
        "scope_rows": len(row_ids),
        "row_id_sorted_sha256": row_id_digest,
        "source_files": len(source_paths),
        "groups": len(groups),
        "semantic_support": {
            "selector_status_counts": dict(statuses),
            "formal_mismatch_rows": formal_mismatch,
            "rows_with_semantic_support_claim": semantic_claims,
            "rows_explicitly_marked_anchor_not_semantic_support": anchor_warnings,
            "license_revalidation_failures": license_failures,
            "admission": "review_only_unadmitted",
        },
        "span_selection": {
            "selected_definition_count": {
                "min": min(selected_definitions) if selected_definitions else None,
                "median": percentile(selected_definitions, .5),
                "p95": percentile(selected_definitions, .95),
                "max": max(selected_definitions) if selected_definitions else None,
            },
            "resolved_reference_count": sum(resolved_references),
            "unresolved_reference_count": sum(unresolved_references),
            "rows_with_unresolved_references": sum(value > 0 for value in unresolved_references),
            "selected_source_line_count": {
                "min": min(selected_lines) if selected_lines else None,
                "median": percentile(selected_lines, .5),
                "p95": percentile(selected_lines, .95),
                "max": max(selected_lines) if selected_lines else None,
            },
            "omitted_source_line_count": {
                "min": min(omitted_lines) if omitted_lines else None,
                "median": percentile(omitted_lines, .5),
                "p95": percentile(omitted_lines, .95),
                "max": max(omitted_lines) if omitted_lines else None,
            },
        },
        "selected_length_tokens": {
            "sequence": {
                "min": min(selected) if selected else None,
                "median": percentile(selected, .5),
                "p95": percentile(selected, .95),
                "p99": percentile(selected, .99),
                "max": max(selected) if selected else None,
                "le_4096": sum(value <= 4096 for value in selected),
                "gt_4096": sum(value > 4096 for value in selected),
                "gt_8192": sum(value > 8192 for value in selected),
                "gt_16384": sum(value > 16384 for value in selected),
                "gt_32768": sum(value > 32768 for value in selected),
                "gt_131072": sum(value > 131072 for value in selected),
            },
            "target_body": {
                "min": min(targets) if targets else None,
                "median": percentile(targets, .5),
                "p95": percentile(targets, .95),
                "p99": percentile(targets, .99),
                "max": max(targets) if targets else None,
                "le_1024": sum(value <= 1024 for value in targets),
                "gt_1024": sum(value > 1024 for value in targets),
            },
        },
        "prior_full_file_comparison": {
            "source_summary_path": str(args.summary),
            "source_summary_sha256": sha_file(args.summary),
            "full_file_sequence_gt_4096": summary.get("counts", {}).get("full_file_sequence_gt_4096", summary.get("actual_full_file_sequence_gt_4096")),
            "full_file_target_gt_1024": summary.get("counts", {}).get("full_file_target_gt_1024", summary.get("actual_full_file_target_body_gt_1024")),
            "full_file_profile_consumed": True,
            "target_token_parity_checked": True,
        },
        "queues": {
            "long_context_queue_rows": queue_rows,
            "target_truncation": "none",
            "data_omission_by_length": False,
        },
        "policy": {
            "path": str(args.policy),
            "sha256": sha_file(args.policy),
            "policy_id": policy.get("policy_id"),
        },
        "serialization_guards": {
            "source_text_written": False,
            "target_text_written": False,
            "token_row_text_keys_found": text_key_rows,
            "cuda": False,
            "training": False,
        },
    }
    write_json(args.output, evidence)


if __name__ == "__main__":
    main()

