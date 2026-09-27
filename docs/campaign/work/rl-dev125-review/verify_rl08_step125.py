#!/usr/bin/env python3
"""Independent, model-free reparse of the RL-08 DEV125 artifact.

The checker reads one named completed cases file and the pinned 75-row DEV
panel.  It uses only the frozen pure PRM-03 parser; it does not import a
framework, load weights, read final data, or inspect live training telemetry.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


PANEL = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl")
CASES = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
    "RL-primary-p2-mb4-full5-e/archive/evaluations/cases-step-125.json"
)
STEP100_RECEIPT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/"
    "RL-08-step100-lead-decision.json"
)
THETA0_RECEIPT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/"
    "RL-01-theta0-lead-review.json"
)
RECEIPT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/"
    "RL-08-step125-independent-review.json"
)
PROTOCOL_SRC = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/"
    "snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/"
    "source/packages/sepalith/src"
)
EXPECTED_PANEL_SHA = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
EXPECTED_CASES_SHA = "878c8d8983b5f0903a36f6bef6d24c15067878526380ce4f1abf835c676be930"
EXPECTED_CASES = 75
EXPECTED_EDITS = 43
EXPECTED_NOOP = 32
DEV_PROMPT_CAP = 4096
DEV_OUTPUT_CAP = 512
COUNT_FIELDS = (
    "protocol_valid",
    "exact_region",
    "predicted_noop",
    "suggestion",
    "cap_hit",
    "strict_noop_correct",
    "strict_noop_false_suggestions",
    "edit_exact",
)


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def ordered_ids_sha(ids: list[str]) -> str:
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def normalized_package(value: object) -> str:
    # This is the panel's documented display-to-normalized identity mapping:
    # package IDs may carry a `pkg:` display prefix and case is insignificant.
    text = str(value)
    return text[4:].lower() if text.startswith("pkg:") else text.lower()


def check_record(
    row: dict[str, Any],
    panel_row: dict[str, Any],
    *,
    PromptContext: Any,
    parse_output: Any,
    valid_generation_tokens: Any,
) -> tuple[dict[str, Any], dict[str, bool]]:
    context = PromptContext.from_mapping(panel_row["context"])
    parsed = parse_output(row.get("raw_output", ""), context)
    token_ids = row.get("generated_ids", [])
    token_valid = valid_generation_tokens(token_ids)
    valid = bool(token_valid and parsed.status == "accepted")
    predicted_noop = bool(valid and parsed.operation == "no_op")
    actual = list(context.region_old) if predicted_noop else list(parsed.body)
    exact = bool(valid and actual == list(panel_row.get("region_new", [])))
    failure = None if valid else (parsed.reason or "noncanonical_or_missing_EOS")
    expected_noop = list(context.region_old) == list(panel_row.get("region_new", []))
    expected = {
        "protocol_valid": valid,
        "predicted_noop": predicted_noop,
        "exact_region": exact,
        "failure": failure,
        "expected_noop": expected_noop,
    }
    fields = {key: row.get(key) == value for key, value in expected.items()}
    return expected, fields


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, details: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "status": "pass" if ok else "fail"}
    if details is not None:
        item["details"] = details
    checks.append(item)


def main() -> int:
    checks: list[dict[str, Any]] = []
    panel = read_jsonl(PANEL)
    panel_size, panel_sha = sha256_file(PANEL)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    cases_size, cases_sha = sha256_file(CASES)
    results = cases.get("results", []) if isinstance(cases, dict) else []
    summary = cases.get("summary", {}) if isinstance(cases, dict) else {}
    panel_ids = [str(row.get("id")) for row in panel]
    result_ids = [str(row.get("id")) for row in results if isinstance(row, dict)]

    sys.path.insert(0, str(PROTOCOL_SRC))
    from sepalith.campaign_protocol import PromptContext, parse_output, valid_generation_tokens  # type: ignore

    add_check(
        checks,
        "panel.identity",
        len(panel) == EXPECTED_CASES and len(set(panel_ids)) == EXPECTED_CASES and panel_sha == EXPECTED_PANEL_SHA,
        {"rows": len(panel), "bytes": panel_size, "sha256": panel_sha},
    )
    add_check(
        checks,
        "cases.identity",
        cases.get("status") == "complete"
        and cases.get("step") == 125
        and len(results) == EXPECTED_CASES
        and len(set(result_ids)) == EXPECTED_CASES
        and result_ids == panel_ids
        and summary.get("panel_sha256") == EXPECTED_PANEL_SHA
        and summary.get("case_ids") == panel_ids,
        {
            "status": cases.get("status"),
            "step": cases.get("step"),
            "rows": len(results),
            "ordered_ids_exact": result_ids == panel_ids,
            "ordered_ids_sha256": ordered_ids_sha(result_ids),
            "artifact_sha256": cases_sha,
        },
    )
    add_check(checks, "cases.pinned_artifact_hash", cases_sha == EXPECTED_CASES_SHA, {"bytes": cases_size, "sha256": cases_sha})
    add_check(
        checks,
        "panel.dev_split_and_labels",
        all(row.get("split") == "dev" for row in panel)
        and sum(list(PromptContext.from_mapping(row["context"]).region_old) == list(row.get("region_new", [])) for row in panel) == EXPECTED_NOOP,
        {"split_counts": dict(collections.Counter(str(row.get("split")) for row in panel))},
    )

    denominators = summary.get("denominators", {})
    denom_ok = {
        "cases": denominators.get("cases") == EXPECTED_CASES,
        "edits": denominators.get("edits") == EXPECTED_EDITS,
        "strict_noop": denominators.get("strict_noop") == EXPECTED_NOOP,
    }
    add_check(checks, "development.denominators", all(denom_ok.values()), {key: denominators.get(key) for key in denom_ok})

    panel_by_id = {str(row.get("id")): row for row in panel}
    counts: collections.Counter[str] = collections.Counter()
    family_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    parser_errors: list[str] = []
    cap_errors: list[str] = []
    token_errors: list[str] = []
    prompt_lengths = [int(row.get("prompt_tokens", -1)) for row in results]
    output_lengths = [int(row.get("generated_tokens", -1)) for row in results]

    for row in results:
        row_id = str(row.get("id"))
        panel_row = panel_by_id.get(row_id)
        if panel_row is None:
            parser_errors.append(f"{row_id}:missing_panel")
            continue
        try:
            computed, field_checks = check_record(
                row,
                panel_row,
                PromptContext=PromptContext,
                parse_output=parse_output,
                valid_generation_tokens=valid_generation_tokens,
            )
        except Exception as error:  # parser failures are evidence, not fatal checker crashes
            parser_errors.append(f"{row_id}:parser_exception:{type(error).__name__}")
            continue
        if not all(field_checks.values()):
            parser_errors.extend(f"{row_id}:{key}" for key, ok in field_checks.items() if not ok)
        valid = bool(computed["protocol_valid"])
        expected_noop = bool(computed["expected_noop"])
        predicted_noop = bool(computed["predicted_noop"])
        exact = bool(computed["exact_region"])
        counts.update(
            {
                "protocol_valid": int(valid),
                "exact_region": int(exact),
                "predicted_noop": int(predicted_noop),
                "suggestion": int(valid and not predicted_noop),
                "cap_hit": int(bool(row.get("cap_hit"))),
                "strict_noop_correct": int(expected_noop and predicted_noop),
                "strict_noop_false_suggestions": int(expected_noop and valid and not predicted_noop),
                "edit_exact": int(not expected_noop and exact),
            }
        )
        family_counts[str(panel_row.get("family"))].update(
            {
                "cases": 1,
                "protocol_valid": int(valid),
                "exact_region": int(exact),
                "predicted_noop": int(predicted_noop),
                "strict_noop_correct": int(expected_noop and predicted_noop),
                "strict_noop_false_suggestions": int(expected_noop and valid and not predicted_noop),
                "cap_hit": int(bool(row.get("cap_hit"))),
            }
        )

        ids = row.get("generated_ids")
        if not isinstance(ids, list) or row.get("generated_tokens") != len(ids):
            token_errors.append(f"{row_id}:generated_length")
        elif any(type(token) is not int or not 0 <= token < 130560 for token in ids):
            token_errors.append(f"{row_id}:token_range")
        elif ids and ids[0] == 0:
            token_errors.append(f"{row_id}:leading_bos")
        if row.get("prompt_tokens", -1) > DEV_PROMPT_CAP or row.get("prompt_tokens", -1) < 0:
            cap_errors.append(f"{row_id}:prompt_cap")
        if row.get("generated_tokens", -1) > DEV_OUTPUT_CAP or row.get("generated_tokens", -1) < 0:
            cap_errors.append(f"{row_id}:output_cap")
        if bool(row.get("cap_hit")) != (row.get("generated_tokens") == DEV_OUTPUT_CAP):
            cap_errors.append(f"{row_id}:cap_flag")

    stored_counts = summary.get("counts", {})
    count_ok = all(stored_counts.get(key) == counts.get(key) for key in COUNT_FIELDS)
    add_check(checks, "development.pinned_parser_reparse", not parser_errors, {"error_count": len(parser_errors), "errors": parser_errors[:12]})
    add_check(checks, "development.counts", count_ok, {"independent": dict(counts), "stored": {key: stored_counts.get(key) for key in COUNT_FIELDS}})
    add_check(
        checks,
        "development.token_and_cap_gates",
        not token_errors and not cap_errors,
        {
            "token_errors": token_errors[:12],
            "cap_errors": cap_errors[:12],
            "prompt_tokens": {"min": min(prompt_lengths), "max": max(prompt_lengths), "cap": DEV_PROMPT_CAP},
            "generated_tokens": {"min": min(output_lengths), "max": max(output_lengths), "cap": DEV_OUTPUT_CAP},
            "eos_valid": counts.get("protocol_valid", 0),
        },
    )

    result_packages = [str(row.get("package_id")) for row in results]
    panel_packages = [str(row.get("package_id")) for row in panel]
    package_ok = (
        len(set(result_packages)) == 69
        and len({normalized_package(value) for value in result_packages}) == 60
        and result_packages == panel_packages
        and denominators.get("packages") == 69
    )
    add_check(
        checks,
        "development.package_denominators",
        package_ok,
        {
            "formatted_unique": len(set(result_packages)),
            "normalized_unique": len({normalized_package(value) for value in result_packages}),
            "stored_packages": denominators.get("packages"),
        },
    )

    loss_sums = {
        key: sum(float(row.get("loss", {}).get(key, 0.0)) for row in results)
        for key in ("prompt_nll_sum", "target_nll_sum", "prompt_tokens", "target_tokens")
    }
    nlls_finite = all(math.isfinite(float(row.get("loss", {}).get(key, float("nan")))) for row in results for key in loss_sums)
    weighted_prompt = loss_sums["prompt_nll_sum"] / loss_sums["prompt_tokens"]
    weighted_target = loss_sums["target_nll_sum"] / loss_sums["target_tokens"]
    nll_ok = (
        nlls_finite
        and denominators.get("prompt_loss_tokens") == int(loss_sums["prompt_tokens"]) == 91180
        and denominators.get("target_loss_tokens") == int(loss_sums["target_tokens"]) == 3450
        and math.isclose(weighted_prompt, float(summary.get("prompt_nll")), rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(weighted_target, float(summary.get("target_nll")), rel_tol=0.0, abs_tol=1e-12)
    )
    add_check(
        checks,
        "development.weighted_nll_denominators",
        nll_ok,
        {
            "prompt_loss_tokens": int(loss_sums["prompt_tokens"]),
            "target_loss_tokens": int(loss_sums["target_tokens"]),
            "prompt_nll": weighted_prompt,
            "target_nll": weighted_target,
            "finite": nlls_finite,
        },
    )

    step100 = json.loads(STEP100_RECEIPT.read_text(encoding="utf-8"))
    old_counts = step100.get("counts", {})
    old_denominators = step100.get("denominators", {})
    delta = {key: counts.get(key, 0) - int(old_counts.get(key, 0)) for key in COUNT_FIELDS}
    comparison_ok = all(old_denominators.get(key) == denominators.get(key) for key in ("cases", "edits", "strict_noop", "packages", "prompt_loss_tokens", "target_loss_tokens"))
    add_check(
        checks,
        "development.compare_dev100",
        comparison_ok,
        {
            "step100_artifact_sha256": step100.get("artifact_sha256"),
            "step100_receipt_sha256": sha256_file(STEP100_RECEIPT)[1],
            "step100_counts": {key: old_counts.get(key) for key in COUNT_FIELDS},
            "step125_counts": dict(counts),
            "delta_step125_minus_step100": delta,
            "nll_delta": {
                "prompt": float(summary.get("prompt_nll")) - float(step100.get("prompt_nll")),
                "target": float(summary.get("target_nll")) - float(step100.get("target_nll")),
            },
        },
    )

    theta0 = json.loads(THETA0_RECEIPT.read_text(encoding="utf-8"))
    theta0_ok = theta0.get("status") == "accepted_live_theta0_gate" and theta0.get("decision", "").startswith("Admit")
    add_check(
        checks,
        "theta0.baseline_gate_context",
        theta0_ok,
        {
            "status": theta0.get("status"),
            "counts": theta0.get("counts"),
            "quality_score_present": any(key in theta0 for key in ("prompt_nll", "target_nll", "edit_exact")),
            "interpretation": "theta0 receipt is a native protocol gate, so no directly comparable DEV quality score is claimed",
        },
    )

    artifact_size, artifact_sha = sha256_file(CASES)
    overall = all(item["status"] == "pass" for item in checks)
    receipt = {
        "task": "RL-08",
        "status": "pass_independent_dev125_reparse" if overall else "fail_independent_dev125_reparse",
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owner": "worker-rl-contexts",
        "scope": "DEV125 quality reparse only; no promotion or sealed-final read",
        "artifacts": {
            "cases": {"path": str(CASES), "bytes": artifact_size, "sha256": artifact_sha, "step": cases.get("step"), "status": cases.get("status")},
            "panel": {"path": str(PANEL), "bytes": panel_size, "sha256": panel_sha, "rows": len(panel), "split": "dev"},
            "step100_receipt": {"path": str(STEP100_RECEIPT), "sha256": sha256_file(STEP100_RECEIPT)[1]},
            "theta0_receipt": {"path": str(THETA0_RECEIPT), "sha256": sha256_file(THETA0_RECEIPT)[1]},
            "protocol_source": str(PROTOCOL_SRC),
        },
        "ordered_case_ids": {"rows": len(result_ids), "exact_panel_order": result_ids == panel_ids, "sha256": ordered_ids_sha(result_ids)},
        "denominators": {key: denominators.get(key) for key in ("cases", "edits", "strict_noop", "packages", "prompt_loss_tokens", "target_loss_tokens")},
        "package_id_counts": {"formatted_unique": len(set(result_packages)), "normalized_unique": len({normalized_package(value) for value in result_packages})},
        "counts": dict(counts),
        "family_counts": {family: dict(values) for family, values in sorted(family_counts.items())},
        "weighted_nll": {"prompt": weighted_prompt, "target": weighted_target},
        "checks": checks,
        "decision_context": "DEV125 remains diagnostic evidence; theta0 remains the selected parent and no quality promotion is made.",
    }
    out = RECEIPT
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "artifact_sha256": artifact_sha, "counts": dict(counts), "denominators": receipt["denominators"], "package_id_counts": receipt["package_id_counts"], "failed_checks": [item["name"] for item in checks if item["status"] == "fail"]}, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
