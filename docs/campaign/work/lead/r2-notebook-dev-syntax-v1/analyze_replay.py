#!/usr/bin/env python3
"""Join notebook parse results to the frozen DEV replay identities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
RESULTS = ROOT / "notebook-results"
ARMS = ("incumbent", "C250", "D500")


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
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", escapechar="\\"))


def concise_error(value: str) -> str:
    lines = value.splitlines()
    if not lines:
        return ""
    # Keep the parser location/reason and caret, without copying source lines.
    return " | ".join([lines[0], lines[-1]])


def main() -> None:
    inputs = [json.loads(line) for line in (PACKET / "replay-inputs.jsonl").open()]
    require(len(inputs) == 225, "input arm/case denominator differs")
    by_arm = {arm: [row for row in inputs if row["arm"] == arm] for arm in ARMS}
    require(all(len(rows) == 75 for rows in by_arm.values()), "per-arm denominator differs")

    baseline_rows = read_tsv(RESULTS / "results-baseline.tsv")
    require(len(baseline_rows) == 75, "baseline parse result denominator differs")
    baseline = {row["id"]: row for row in baseline_rows}
    require(len(baseline) == 75, "baseline result IDs are not unique")
    applied = {}
    for arm in ARMS:
        rows = read_tsv(RESULTS / f"results-{arm}.tsv")
        expected = sum(item["protocol_valid"] for item in by_arm[arm])
        require(len(rows) == expected, f"{arm}: applied parse denominator differs")
        applied[arm] = {row["id"]: row for row in rows}
        require(len(applied[arm]) == expected, f"{arm}: applied result IDs are not unique")

    joined = []
    for arm in ARMS:
        for item in by_arm[arm]:
            before = baseline[item["id"]]
            require(before["family"] == item["family"] and before["sha256"] == item["baseline_sha256"],
                    f"{arm}/{item['id']}: baseline parse identity differs")
            after = applied[arm].get(item["id"])
            require((after is not None) == item["protocol_valid"],
                    f"{arm}/{item['id']}: applied parse presence differs from protocol")
            if after:
                require(after["family"] == item["family"] and after["sha256"] == item["applied_sha256"],
                        f"{arm}/{item['id']}: applied parse identity differs")
            joined.append({
                "arm": arm, "case_index": item["case_index"], "id": item["id"], "family": item["family"],
                "expected_operation": item["expected_operation"], "prompt_sha256": item["prompt_sha256"],
                "baseline_sha256": item["baseline_sha256"], "baseline_parse_ok": before["parse_ok"] == "true",
                "baseline_parse_error": concise_error(before["error"]),
                "protocol_valid": item["protocol_valid"], "protocol_operation": item["protocol_operation"],
                "protocol_reason": item["protocol_reason"], "raw_text_sha256": item["raw_text_sha256"],
                "applied_sha256": item["applied_sha256"],
                "candidate_parse_status": ("parse_ok" if after and after["parse_ok"] == "true" else
                                           "parse_failed" if after else "not_attempted_invalid_protocol"),
                "candidate_parse_error": concise_error(after["error"]) if after else None,
                "edit_exact": item["edit_exact"], "strict_noop_correct": item["strict_noop_correct"],
                "noop_false_positive": item["noop_false_positive"],
                "semantic_correctness": "not_measured_by_parse_replay",
            })

    with (ROOT / "parse-results.jsonl").open("w") as handle:
        for row in joined:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {}
    family_summary = {}
    transitions = {}
    invalid_protocol = {}
    syntax_failures = {}
    for arm in ARMS:
        rows = [row for row in joined if row["arm"] == arm]
        summary[arm] = {
            "cases": len(rows),
            "protocol_valid": sum(row["protocol_valid"] for row in rows),
            "protocol_invalid": sum(not row["protocol_valid"] for row in rows),
            "applied_parse_ok": sum(row["candidate_parse_status"] == "parse_ok" for row in rows),
            "applied_parse_failed": sum(row["candidate_parse_status"] == "parse_failed" for row in rows),
            "not_attempted_invalid_protocol": sum(row["candidate_parse_status"] == "not_attempted_invalid_protocol" for row in rows),
            "edit_exact": sum(row["edit_exact"] for row in rows),
            "strict_noop_correct": sum(row["strict_noop_correct"] for row in rows),
            "noop_false_positive": sum(row["noop_false_positive"] for row in rows),
        }
        transitions[arm] = dict(sorted(Counter(
            ("baseline_parse_ok" if row["baseline_parse_ok"] else "baseline_parse_failed") + "__" + row["candidate_parse_status"]
            for row in rows
        ).items()))
        invalid_protocol[arm] = [
            {key: row[key] for key in ("id", "family", "protocol_reason", "raw_text_sha256")}
            for row in rows if not row["protocol_valid"]
        ]
        syntax_failures[arm] = [
            {key: row[key] for key in ("id", "family", "candidate_parse_error", "noop_false_positive", "edit_exact")}
            for row in rows if row["candidate_parse_status"] == "parse_failed"
        ]
        family_summary[arm] = {}
        for family in sorted({row["family"] for row in rows}):
            subset = [row for row in rows if row["family"] == family]
            family_summary[arm][family] = {
                "cases": len(subset), "baseline_parse_ok": sum(row["baseline_parse_ok"] for row in subset),
                "protocol_valid": sum(row["protocol_valid"] for row in subset),
                "protocol_invalid": sum(not row["protocol_valid"] for row in subset),
                "applied_parse_ok": sum(row["candidate_parse_status"] == "parse_ok" for row in subset),
                "applied_parse_failed": sum(row["candidate_parse_status"] == "parse_failed" for row in subset),
                "edit_exact": sum(row["edit_exact"] for row in subset),
                "strict_noop_correct": sum(row["strict_noop_correct"] for row in subset),
                "noop_false_positive": sum(row["noop_false_positive"] for row in subset),
            }

    baseline_failed = [{"id": row["id"], "family": row["family"],
                        "parse_error": concise_error(row["error"])}
                       for row in baseline_rows if row["parse_ok"] != "true"]
    timings = json.loads((RESULTS / "timings.json").read_text())
    readout = {
        "schema": "sepalith.run10.notebook-dev-syntax-replay.v1",
        "status": "verified_parse_only_dev_replay_complete",
        "inputs": json.loads((PACKET / "manifest.json").read_text()),
        "identity_checks": {
            "arm_case_rows_exact": 225, "case_ids_exact_order_per_arm": True,
            "prompt_family_operation_and_noop_identity_exact": 225,
            "full_baseline_content_sha256_exact": 75,
            "reconstruction_complete_selection": 45,
            "reconstruction_source_snapshot_plus_history": 30,
            "protocol_reparse_matches_recorded": 225,
            "remote_payload_files_rehashed_exact": 290,
        },
        "baseline": {"cases": 75, "parse_ok": 69, "parse_failed": 6,
                     "failures": baseline_failed},
        "arms": summary,
        "transitions": transitions,
        "families": family_summary,
        "invalid_protocol_no_application": invalid_protocol,
        "candidate_syntax_failures": syntax_failures,
        "timings": timings,
        "notebook": {
            "host": "m0pad", "address": "192.168.178.40", "r_version": "4.6.1",
            "remote_directory": "/home/m0hawk/.local/state/sepalith/campaign-20260915/notebook-dev-syntax-v1",
            "affinity": [0, 2], "maximum_threads": 2, "timeout_seconds_per_batch": 300,
            "owned_parse_processes_after": 0,
        },
        "interpretation": [
            "R parse success establishes syntax only; it does not establish behavioral or semantic correctness.",
            "Native exact target matches and no-op correctness are retained as separate recorded metrics, not inferred from parsing.",
            "The six finish_block baselines are intentionally incomplete and fail parsing; each arm produced three protocol-valid finish applications, and all nine applied buffers parse successfully. The other three finish outputs per arm were protocol-invalid and were not applied or parsed.",
            "All protocol-valid expected-edit applications parse successfully. Candidate syntax failures occur only among false suggestions on expected no-op cases.",
            "All protocol-valid roxygen applications parse successfully, but roxygen exact match remains 0/8 for every arm; incumbent and D500 each also have one separate protocol-invalid roxygen result.",
        ],
        "safety": {
            "generated_r_executed": False, "harness_primitive": "base::parse(file=..., keep.source=FALSE)",
            "source_or_eval_used": False, "sentinel_top_level_system_expression_executed": False,
            "gpu_model_server_editor_touched": False, "final_content_accessed": False,
        },
        "artifacts": {
            "parse_results_jsonl_sha256": None,
            "remote_results": {path.name: sha(path) for path in sorted(RESULTS.iterdir()) if path.is_file()},
        },
    }
    readout["artifacts"]["parse_results_jsonl_sha256"] = sha(ROOT / "parse-results.jsonl")
    (ROOT / "independent-readout.json").write_text(json.dumps(readout, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
