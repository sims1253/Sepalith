#!/usr/bin/env python3
"""Orchestrate the occurrence-based TRAIN-only roxygen NSE review."""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-roxy-nse-semantic-review-v1"
INPUT = PLAN / "docs/campaign/work/lead/r2-roxy-full-context-recovery-v1/materialization-v4/recovery-ledger.jsonl"
OUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/Roxy-NSE-semantic-review-v1")
EXPECTED_ROWS = 2273
SHAP_ID = "018df81bd4773e47ca8cfe2e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main() -> None:
    if OUT.exists():
        raise FileExistsError("fresh review output required")
    OUT.mkdir(parents=True)
    held = []
    with INPUT.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["recovery_status"] == "hold_residual_noncall_reference_repair_or_semantic_evidence":
                held.append(row)
    if len(held) != EXPECTED_ROWS or len({row["row_id"] for row in held}) != EXPECTED_ROWS:
        raise ValueError("residual held denominator differs")
    groups = collections.defaultdict(list)
    source_hashes = {}
    for row in held:
        path = Path(row["source_path"])
        expected = row["source_sha256"]
        prior = source_hashes.setdefault(str(path), expected)
        if prior != expected:
            raise ValueError("one source path has conflicting hashes")
        groups[(str(path), expected)].append({
            "row_id": row["row_id"],
            "target_definition_name": row["target_definition_name"],
            "target_definition_span": row["target_definition_span"],
            "residual_noncall_names": row["residual_noncall_names"],
            "selected_context_tokens": row["selected_context_tokens"],
            "target_body_tokens": row["target_body_tokens"],
        })
    verified = {}
    for (path_text, expected), _ in sorted(groups.items()):
        path = Path(path_text)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"source hash differs: {path}")
        verified[path_text] = {"sha256": actual, "bytes": path.stat().st_size}
    compact = {
        "schema": "sepalith.dat10.roxy-nse-review-input.v1",
        "source_groups": [
            {"source_path": path, "source_sha256": digest, "rows": sorted(rows, key=lambda row: row["row_id"])}
            for (path, digest), rows in sorted(groups.items())
        ],
    }
    input_path = OUT / "r-input.json"
    input_path.write_text(json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n")
    raw_path = OUT / "r-occurrence-output.jsonl"
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(
        ["Rscript", "--vanilla", str(PACKET / "nse_occurrence_audit.R"), str(input_path), str(raw_path)],
        text=True, capture_output=True, timeout=1200, env=env,
    )
    (OUT / "r-stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"R occurrence audit failed: {completed.returncode}")
    audited = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    if len(audited) != EXPECTED_ROWS or {row["row_id"] for row in audited} != {row["row_id"] for row in held}:
        raise ValueError("R occurrence result ID parity differs")
    audited.sort(key=lambda row: row["row_id"])
    status_counts = collections.Counter(row["status"] for row in audited)
    recoverable = [row for row in audited if row["status"] == "recoverable_occurrence_proven_nse"]
    retained = [row for row in audited if row["status"] != "recoverable_occurrence_proven_nse"]
    if not any(row["row_id"] == SHAP_ID for row in recoverable):
        raise ValueError("root-provided SHAP NSE control was not recovered")
    class_counts = collections.Counter()
    for row in audited:
        evidence = row.get("evidence") or {}
        for details in (evidence.get("residuals") or {}).values():
            class_counts.update(details.get("class_counts") or {})
    final_path = OUT / "semantic-ledger.jsonl"
    with final_path.open("x") as stream:
        for row in audited:
            stream.write(canonical(row) + "\n")
    recoverable_path = OUT / "recoverable-ids.json"
    recoverable_path.write_text(json.dumps({
        "schema": "sepalith.dat10.roxy-nse-recoverable-ids.v1",
        "status": "review_only_not_training_admission",
        "count": len(recoverable),
        "ids": [row["row_id"] for row in recoverable],
    }, indent=2, sort_keys=True) + "\n")
    retained_path = OUT / "retained-hold-ids.json"
    retained_path.write_text(json.dumps({
        "schema": "sepalith.dat10.roxy-nse-retained-hold-ids.v1",
        "status": "unsupported_reference_or_more_semantic_evidence_required",
        "count": len(retained),
        "ids": [row["row_id"] for row in retained],
    }, indent=2, sort_keys=True) + "\n")
    representative_ids = [SHAP_ID]
    representative_ids += sorted((row["row_id"] for row in recoverable if row["row_id"] != SHAP_ID), key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:9]
    representative_ids += sorted((row["row_id"] for row in retained), key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:10]
    by_id = {row["row_id"]: row for row in audited}
    representative_path = OUT / "representative-evidence.json"
    representative_path.write_text(json.dumps({
        "schema": "sepalith.dat10.roxy-nse-representative-evidence.v1",
        "method": "root-named positive control plus deterministic SHA-256 ordering within recovered and retained outcomes",
        "rows": [by_id[row_id] for row_id in representative_ids],
        "source_or_target_text_copied": False,
    }, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "sepalith.dat10.roxy-nse-semantic-review.v1",
        "status": "complete_review_only_root_admission_required",
        "input": {"path": str(INPUT), "sha256": sha256(INPUT), "residual_rows": EXPECTED_ROWS},
        "source_files": len(groups),
        "all_source_hashes_verified": True,
        "rows_audited": len(audited),
        "status_counts": dict(sorted(status_counts.items())),
        "recoverable_count": len(recoverable),
        "retained_hold_count": len(retained),
        "occurrence_class_counts": dict(sorted(class_counts.items())),
        "shap_positive_control_recovered": True,
        "policy": {
            "names_only_allowlist": False,
            "recovery_rule": "Every occurrence of every residual name must be inside the target function and have data.table NSE evidence tied to a local colnames(receiver) schema expression; ggforce::facet_zoom x/y is accepted only after the same name has that direct column proof.",
            "true_globals_default": "hold",
            "ambiguous_tidy_eval_formula_or_subset": "hold",
            "target_truncation": False,
            "training_admission": False,
        },
        "selected_context_adequacy": {
            "basis": "Only occurrences inside the exact named target function are considered; no same-file occurrence outside that target can support recovery.",
            "all_recovered_rows_selected_context_tokens_positive": all(row["selected_context_tokens"] > 0 for row in recoverable),
            "all_recovered_occurrences_have_expression_hashes": all(
                occurrence.get("expression_sha256")
                for row in recoverable
                for details in row["evidence"]["residuals"].values()
                for occurrence in details["occurrence_evidence"]
            ),
        },
        "artifacts": {},
    }
    for path in (input_path, raw_path, final_path, recoverable_path, retained_path, representative_path, OUT / "r-stderr.log"):
        summary["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(canonical(summary))


if __name__ == "__main__":
    main()
