#!/usr/bin/env python3
"""Summarize tokenizer lengths from the review-only repaired token profiles.

The profile JSONL contains lengths and hashes only.  This postprocessor never
reads or writes source/target text and keeps the source rows out of the
summary.  Percentiles use the lower order statistic at floor(p * (n - 1));
this is stated in the output so downstream admission does not infer an
interpolation rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_lower(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def stats(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": percentile_lower(ordered, 0.95),
        "p99": percentile_lower(ordered, 0.99),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    length_names = (
        "prompt_without_bos",
        "sequence",
        "target_body_tokens",
        "response_with_terminal_eos",
    )
    by_budget: dict[str, dict[str, list[int]]] = {}
    row_ids: set[str] = set()
    for line_number, line in enumerate(args.profiles.open(), 1):
        row = json.loads(line)
        row_id = row["row_id"]
        if row_id in row_ids:
            raise ValueError(f"duplicate row_id at line {line_number}: {row_id}")
        row_ids.add(row_id)
        for profile in row["context_profiles"]:
            budget = str(profile["budget"])
            values = by_budget.setdefault(
                budget,
                {name: [] for name in (*length_names, "used_utf16_units")},
            )
            lengths = profile["lengths"]
            for name in length_names:
                values[name].append(int(lengths[name]))
            values["used_utf16_units"].append(int(profile["used_utf16_units"]))

    if len(row_ids) != 10017:
        raise ValueError(f"expected 10017 unique profiles, got {len(row_ids)}")

    summary: dict[str, object] = {
        "schema": "DAT-10-source-walk-support-repair-token-length-summary-v1",
        "profile_path": str(args.profiles),
        "profile_sha256": sha256(args.profiles),
        "profiles_rows": len(row_ids),
        "metrics": {
            "prompt_without_bos": "tokenizer tokens",
            "sequence": "tokenizer tokens",
            "target_body_tokens": "tokenizer tokens",
            "response_with_terminal_eos": "tokenizer tokens",
            "used_utf16_units": "UTF-16 code units plus selected line separators",
        },
        "percentile_method": "lower order statistic at floor(p * (n - 1))",
        "target_policy": "complete target body; no target truncation",
        "old_cropped_metric": {
            "sequence_gt_4096": 0,
            "meaning": "OLD cropped repair rows only; not the repaired full-file token profile",
        },
        "budgets": {},
    }
    budget_output: dict[str, object] = {}

    # Omission and overflow flags are kept separately from length arrays so
    # the summary can distinguish context-budget coverage from token overflow.
    omission_rows: dict[str, int] = {budget: 0 for budget in by_budget}
    overflow_rows: dict[str, int] = {budget: 0 for budget in by_budget}
    target_gt_1024: dict[str, int] = {budget: 0 for budget in by_budget}
    for line in args.profiles.open():
        row = json.loads(line)
        for profile in row["context_profiles"]:
            budget = str(profile["budget"])
            if int(profile["omission_count"]) > 0:
                omission_rows[budget] += 1
            if bool(profile["overflow"]):
                overflow_rows[budget] += 1
            if int(profile["lengths"]["target_body_tokens"]) > 1024:
                target_gt_1024[budget] += 1

    for budget, values in by_budget.items():
        sequence = values["sequence"]
        budget_output[budget] = {
            "rows": len(sequence),
            "omission_rows": omission_rows[budget],
            "overflow_rows": overflow_rows[budget],
            "target_body_gt_1024_rows": target_gt_1024[budget],
            "lengths": {name: stats(items) for name, items in values.items()},
            "sequence_gt_4096_rows": sum(value > 4096 for value in sequence),
        }
    summary["budgets"] = budget_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(row_ids), "output": str(args.output), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
