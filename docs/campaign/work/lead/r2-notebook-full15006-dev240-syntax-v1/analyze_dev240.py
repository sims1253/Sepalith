#!/usr/bin/env python3
"""Join full15006 step-240 DEV identities to notebook parse-only results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
RESULTS = ROOT / "notebook-results"
E750 = ROOT.parent / "r2-notebook-e750-dev-syntax-v1" / "independent-readout.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def read_tsv(path: Path) -> list[dict[str, str]]:
    # R write.table escapes embedded newlines and quotes with backslashes.
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", escapechar="\\"))


def concise_error(value: str) -> str:
    lines = value.splitlines()
    return " | ".join([lines[0], lines[-1]]) if lines else ""


def counts(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "protocol_valid": sum(row["protocol_valid"] for row in rows),
        "protocol_invalid": sum(not row["protocol_valid"] for row in rows),
        "applied_parse_ok": sum(row["candidate_parse_status"] == "parse_ok" for row in rows),
        "applied_parse_failed": sum(row["candidate_parse_status"] == "parse_failed" for row in rows),
        "not_attempted_invalid_protocol": sum(
            row["candidate_parse_status"] == "not_attempted_invalid_protocol" for row in rows
        ),
        "edit_exact": sum(row["edit_exact"] for row in rows),
        "strict_noop_correct": sum(row["strict_noop_correct"] for row in rows),
        "noop_false_positive": sum(row["noop_false_positive"] for row in rows),
        "cap_hit": sum(row["cap_hit"] for row in rows),
    }


def main() -> None:
    inputs = [json.loads(line) for line in (PACKET / "replay-inputs.jsonl").open()]
    require(len(inputs) == 75, "DEV240 case denominator differs")
    require([row["case_index"] for row in inputs] == list(range(75)), "case order differs")
    require(len({row["id"] for row in inputs}) == 75, "case IDs are not unique")

    baseline_rows = read_tsv(RESULTS / "results-baseline.tsv")
    applied_rows = read_tsv(RESULTS / "results-full15006-step240.tsv")
    require(len(baseline_rows) == 75, "baseline parse denominator differs")
    require(len(applied_rows) == sum(row["protocol_valid"] for row in inputs),
            "applied parse denominator differs")
    baseline = {row["id"]: row for row in baseline_rows}
    applied = {row["id"]: row for row in applied_rows}
    require(len(baseline) == 75 and len(applied) == len(applied_rows), "parse IDs differ")

    joined = []
    for item in inputs:
        before = baseline[item["id"]]
        require(before["family"] == item["family"] and before["sha256"] == item["baseline_sha256"],
                f"{item['id']}: baseline identity differs")
        after = applied.get(item["id"])
        require((after is not None) == item["protocol_valid"],
                f"{item['id']}: applied presence differs from protocol validity")
        if after:
            require(after["family"] == item["family"] and after["sha256"] == item["applied_sha256"],
                    f"{item['id']}: applied identity differs")
        joined.append({
            "arm": "full15006-step240",
            "case_index": item["case_index"],
            "id": item["id"],
            "family": item["family"],
            "expected_operation": item["expected_operation"],
            "prompt_sha256": item["prompt_sha256"],
            "raw_text_sha256": item["raw_text_sha256"],
            "generated_ids_sha256": item["generated_ids_sha256"],
            "baseline_sha256": item["baseline_sha256"],
            "baseline_parse_ok": before["parse_ok"] == "true",
            "baseline_parse_error": before["error"],
            "protocol_valid": item["protocol_valid"],
            "protocol_status": item["protocol_status"],
            "protocol_operation": item["protocol_operation"],
            "protocol_reason": item["protocol_reason"],
            "applied_sha256": item["applied_sha256"],
            "candidate_parse_status": ("parse_ok" if after and after["parse_ok"] == "true" else
                                       "parse_failed" if after else "not_attempted_invalid_protocol"),
            "candidate_parse_error": after["error"] if after else None,
            "edit_exact": item["edit_exact"],
            "strict_noop_correct": item["strict_noop_correct"],
            "noop_false_positive": item["noop_false_positive"],
            "cap_hit": item["cap_hit"],
            "semantic_correctness": "not_measured_by_parse_replay",
        })

    with (ROOT / "parse-results.jsonl").open("w") as handle:
        for row in joined:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    families = {
        family: counts([row for row in joined if row["family"] == family])
        for family in sorted({row["family"] for row in joined})
    }
    e750 = json.loads(E750.read_text())
    comparison = {
        "E750": e750["E750"],
        "full15006-step240": counts(joined),
    }
    timings = {
        name: json.loads((RESULTS / f"timing-{name}.json").read_text())
        for name in ("baseline", "full15006-step240")
    }
    baseline_failures = [
        {"id": row["id"], "family": row["family"], "parse_error": concise_error(row["error"])}
        for row in baseline_rows if row["parse_ok"] != "true"
    ]
    invalid = [
        {key: row[key] for key in ("id", "family", "protocol_reason", "raw_text_sha256", "cap_hit")}
        for row in joined if not row["protocol_valid"]
    ]
    syntax_failures = [
        {key: row[key] for key in (
            "case_index", "id", "family", "candidate_parse_error", "noop_false_positive",
            "edit_exact", "cap_hit", "applied_sha256"
        )}
        for row in joined if row["candidate_parse_status"] == "parse_failed"
    ]
    readout = {
        "schema": "sepalith.run10.notebook-full15006-dev240-syntax-replay.v1",
        "status": "verified_parse_only_dev_replay_complete",
        "inputs": json.loads((PACKET / "manifest.json").read_text()),
        "identity_checks": {
            "case_rows_exact": 75,
            "case_ids_exact_order": True,
            "prompt_family_operation_and_quality_identity_exact": 75,
            "full_baseline_content_sha256_exact": 75,
            "protocol_reparse_matches_recorded": 75,
            "context_aware_raw_wire_parse": True,
            "decoded_body_text_used": False,
        },
        "baseline": {"cases": 75, "parse_ok": 69, "parse_failed": 6,
                     "failures": baseline_failures},
        "full15006-step240": counts(joined),
        "comparison": comparison,
        "transitions": dict(sorted(Counter(
            ("baseline_parse_ok" if row["baseline_parse_ok"] else "baseline_parse_failed")
            + "__" + row["candidate_parse_status"] for row in joined
        ).items())),
        "families": families,
        "focus_families": {family: families[family] for family in ("finish_block", "roxygen_drafting")},
        "invalid_protocol_no_application": invalid,
        "candidate_syntax_failures": syntax_failures,
        "timings": timings,
        "notebook": {
            "host": "m0pad",
            "address": "192.168.178.40",
            "r_version": "4.6.1 (2026-06-24)",
            "remote_directory": "/home/m0hawk/.local/state/sepalith/campaign-20260915/notebook-full15006-dev240-syntax-v1-attempt2",
            "affinity": [0, 2],
            "maximum_threads": 2,
            "timeout_seconds_per_batch": 300,
            "owned_parse_processes_after": 0,
            "remote_owned_prefix_removed_after_copy": True,
        },
        "execution_note": {
            "authoritative_attempt": "attempt2",
            "discarded_attempt": "The first fresh attempt bound desktop paths into the TSV and was discarded without analysis; its remote directory was removed. The corrected attempt2 bound the remote packet paths and completed both batches.",
        },
        "interpretation": [
            "R parse success establishes syntax only; it does not establish behavioral or semantic correctness.",
            "Native exact target matches and strict no-op correctness are separate recorded metrics and were not inferred from parsing.",
            "The six finish_block baselines are intentionally incomplete. Three DEV240 finish outputs are protocol-valid: one completed buffer parses and two remain syntactically incomplete; three invalid outputs were not applied.",
            "All eight DEV240 roxygen outputs are protocol-valid and their applied buffers parse; this does not establish documentation correctness.",
            "Two other DEV240 syntax failures are expected no-op cases with false suggestions.",
            "The E750 reference had 68 parse successes and two failures among 70 protocol-valid cases; DEV240 has 66 successes and four failures at the same denominator.",
        ],
        "safety": {
            "generated_r_executed": False,
            "harness_primitive": "base::parse(file=..., keep.source=FALSE)",
            "source_or_eval_used": False,
            "gpu_model_server_editor_touched": False,
            "final_content_accessed": False,
            "dev_material_used_for_train_pilot": False,
        },
        "artifacts": {
            "parse_results_jsonl_sha256": sha(ROOT / "parse-results.jsonl"),
            "e750_reference_readout_sha256": sha(E750),
            "remote_results": {
                path.name: sha(path) for path in sorted(RESULTS.iterdir()) if path.is_file()
            },
        },
    }
    require(counts(joined)["protocol_valid"] == 70, "protocol-valid denominator differs")
    require(counts(joined)["applied_parse_ok"] == 66, "parse-ok count differs")
    require(counts(joined)["applied_parse_failed"] == 4, "parse-failure count differs")
    (ROOT / "independent-readout.json").write_text(json.dumps(readout, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
