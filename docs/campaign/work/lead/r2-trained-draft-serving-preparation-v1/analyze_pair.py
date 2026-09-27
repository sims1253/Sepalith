#!/usr/bin/env python3
"""Analyze native probe JSON without loading models or contacting a server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        raise ValueError(f"{path} does not contain a probe request list")
    return value


def key(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    return record.get("row_id"), record.get("phase"), record.get("rep")


def records(value: dict[str, Any]) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    indexed: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for record in value["requests"]:
        if not isinstance(record, dict):
            raise ValueError("probe request is not an object")
        item_key = key(record)
        if item_key in indexed:
            raise ValueError(f"duplicate request key: {item_key!r}")
        indexed[item_key] = record
    return indexed


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 3)


def timing_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    wall = [float(item.get("combined_case_wall_ms", item.get("wall_ms")))
            for item in items
            if isinstance(item.get("combined_case_wall_ms", item.get("wall_ms")), (int, float))]
    ttft = [float(item["ttft_ms"]) for item in items if isinstance(item.get("ttft_ms"), (int, float))]
    return {
        "requests": len(items),
        "wall_ms": {"p50": percentile(wall, 0.50), "p95": percentile(wall, 0.95)},
        "ttft_ms": {"p50": percentile(ttft, 0.50), "p95": percentile(ttft, 0.95)},
    }


def arm_summary(value: dict[str, Any], indexed: dict[tuple[Any, Any, Any], dict[str, Any]]) -> dict[str, Any]:
    items = list(indexed.values())
    statuses: dict[str, int] = {}
    for item in items:
        status = str(item.get("protocol_status", "missing"))
        statuses[status] = statuses.get(status, 0) + 1
    by_band: dict[str, dict[str, Any]] = {}
    for band in ("short", "long"):
        by_band[band] = timing_summary([item for item in items if item.get("ctx_band") == band])
    draft_pairs = [
        (item.get("draft_n"), item.get("draft_n_accepted"))
        for item in items
    ]
    has_all_counters = bool(items) and all(
        isinstance(drafted, (int, float)) and isinstance(accepted, (int, float))
        and drafted >= 0 and accepted >= 0 and accepted <= drafted
        for drafted, accepted in draft_pairs
    )
    if has_all_counters:
        drafted_total = sum(float(drafted) for drafted, _ in draft_pairs)
        accepted_total = sum(float(accepted) for _, accepted in draft_pairs)
        acceptance: dict[str, Any] = {
            "status": "present",
            "drafted": int(drafted_total),
            "accepted": int(accepted_total),
            "rate": None if drafted_total == 0 else round(accepted_total / drafted_total, 6),
        }
    else:
        acceptance = {"status": "missing", "drafted": None, "accepted": None, "rate": None}
    return {
        "arm": value.get("arm"),
        "status": value.get("status"),
        "protocol_status_counts": statuses,
        "timing": timing_summary(items),
        "timing_by_band": by_band,
        "acceptance": acceptance,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any], label: str) -> dict[str, Any]:
    left, right = records(baseline), records(candidate)
    all_keys = sorted(set(left) | set(right), key=repr)
    mismatches: list[dict[str, Any]] = []
    comparable = 0
    for item_key in all_keys:
        before, after = left.get(item_key), right.get(item_key)
        if before is None or after is None:
            mismatches.append({"key": list(item_key), "reason": "missing_pair"})
            continue
        if before.get("protocol_status") != "accepted" or after.get("protocol_status") != "accepted":
            mismatches.append({
                "key": list(item_key),
                "reason": "unaccepted_pair",
                "baseline_status": before.get("protocol_status"),
                "candidate_status": after.get("protocol_status"),
            })
            continue
        comparable += 1
        if (before.get("returned_token_ids") != after.get("returned_token_ids")
                or before.get("raw_text") != after.get("raw_text")):
            mismatches.append({"key": list(item_key), "reason": "greedy_output_mismatch"})
    baseline_wall = arm_summary(baseline, left)["timing"]["wall_ms"]
    candidate_wall = arm_summary(candidate, right)["timing"]["wall_ms"]
    speedup = {
        "p50": None,
        "p95": None,
    }
    if baseline_wall.get("p50") and candidate_wall.get("p50"):
        speedup["p50"] = round(baseline_wall["p50"] / candidate_wall["p50"], 6)
    if baseline_wall.get("p95") and candidate_wall.get("p95"):
        speedup["p95"] = round(baseline_wall["p95"] / candidate_wall["p95"], 6)
    return {
        "candidate": label,
        "status": "pass" if comparable == len(all_keys) and not mismatches else "fail",
        "requests_compared": comparable,
        "requests_expected": len(all_keys),
        "greedy_parity": not mismatches,
        "mismatches": mismatches,
        "speedup_vs_ordinary_baseline": speedup,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", action="append", default=[], metavar="LABEL=JSON")
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    baseline_path = Path(args.baseline)
    baseline = read_json(baseline_path)
    baseline_records = records(baseline)
    output: dict[str, Any] = {
        "schema_version": "sepalith.r2.native-pair-analysis.v1",
        "status": "pending_gate",
        "baseline": arm_summary(baseline, baseline_records),
        "candidates": [],
    }
    for specification in args.candidate:
        if "=" not in specification:
            raise ValueError("--candidate must be LABEL=JSON")
        label, path_text = specification.split("=", 1)
        candidate_path = Path(path_text)
        candidate = read_json(candidate_path)
        candidate_records = records(candidate)
        output["candidates"].append({
            "summary": arm_summary(candidate, candidate_records),
            "comparison": compare(baseline, candidate, label),
        })
    output["status"] = "pass" if output["candidates"] and all(
        item["comparison"]["status"] == "pass"
        and item["summary"]["acceptance"]["status"] == "present"
        for item in output["candidates"]
    ) else "fail"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "out": str(out)}, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
