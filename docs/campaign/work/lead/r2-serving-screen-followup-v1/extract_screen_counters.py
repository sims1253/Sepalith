#!/usr/bin/env python3
"""Extract and cross-check speculative counters from the four-arm screen.

The native adapter writes DSpark counters inside each request's ``timings``
object.  The old pair analyzer looked for top-level fields and therefore
reported missing acceptance.  This script reads only the completed screen
JSON/logs, checks the per-request values against server cumulative statistics,
and never infers acceptance from returned text or output length.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ARM_FILES = {
    "ordinary-baseline": ("ordinary-baseline.json", "ordinary-baseline-server.log", "ordinary"),
    "model-free-ngram": ("model-free-ngram.json", "model-free-ngram-server.log", "ngram"),
    "released-dspark": ("released-dspark.json", "released-dspark-server.log", "dspark"),
    "trained-dspark": ("trained-dspark.json", "trained-dspark-server.log", "dspark"),
}

PRINT_RE = re.compile(
    r"draft acceptance\s*=\s*[0-9.]+\s*\(\s*(?P<accepted>\d+)\s+accepted\s*/\s*(?P<drafted>\d+)\s+generated\)"
)
STATS_RE = re.compile(
    r"statistics\s+(?P<kind>draft-dspark|ngram-mod):.*?"
    r"#gen tokens\s*=\s*(?P<drafted>\d+)\s*,\s*#acc tokens\s*=\s*(?P<accepted>\d+)"
)


class CheckError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CheckError(f"{path} is not a JSON object")
    return value


def _pctl(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # Match the simple linear interpolation convention used by the screen
    # analyzer without importing a numerical package.
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _timing_summary(requests: list[dict[str, Any]]) -> dict[str, Any]:
    wall = [float(r["combined_case_wall_ms"]) for r in requests]
    ttft = [float(r["ttft_ms"]) for r in requests]
    return {
        "requests": len(requests),
        "combined_case_wall_ms": {"p50": _pctl(wall, 0.50), "p95": _pctl(wall, 0.95)},
        "ttft_ms": {"p50": _pctl(ttft, 0.50), "p95": _pctl(ttft, 0.95)},
    }


def _parse_log(path: Path) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    print_rows: list[dict[str, int]] = []
    stats_rows: list[dict[str, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = PRINT_RE.search(line)
        if match:
            print_rows.append(
                {
                    "line": line_number,
                    "drafted": int(match.group("drafted")),
                    "accepted": int(match.group("accepted")),
                }
            )
        match = STATS_RE.search(line)
        if match:
            stats_rows.append(
                {
                    "line": line_number,
                    "drafted": int(match.group("drafted")),
                    "accepted": int(match.group("accepted")),
                }
            )
    return print_rows, stats_rows


def _protocol_summary(requests: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for request in requests:
        status = request.get("protocol_status")
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _extract_arm(screen_dir: Path, arm_name: str) -> dict[str, Any]:
    json_name, log_name, kind = ARM_FILES[arm_name]
    data = _read_json(screen_dir / json_name)
    requests = data.get("requests")
    if not isinstance(requests, list) or len(requests) != 8:
        raise CheckError(f"{arm_name}: expected exactly 8 requests")
    requests = [r for r in requests if isinstance(r, dict)]
    if len(requests) != 8:
        raise CheckError(f"{arm_name}: request is not an object")
    protocol_counts = _protocol_summary(requests)
    if protocol_counts != {"accepted": 8}:
        raise CheckError(f"{arm_name}: protocol statuses {protocol_counts}")

    print_rows, stats_rows = _parse_log(screen_dir / log_name)
    rows: list[dict[str, Any]] = []
    if kind == "dspark":
        if len(print_rows) != 8 or len(stats_rows) != 8:
            raise CheckError(
                f"{arm_name}: expected 8 print_timing and 8 statistics rows, "
                f"got {len(print_rows)} and {len(stats_rows)}"
            )
        cumulative_drafted = 0
        cumulative_accepted = 0
        for index, (request, printed, cumulative) in enumerate(zip(requests, print_rows, stats_rows)):
            timings = request.get("timings")
            if not isinstance(timings, dict):
                raise CheckError(f"{arm_name} request {index}: missing timings object")
            drafted = timings.get("draft_n")
            accepted = timings.get("draft_n_accepted")
            if not isinstance(drafted, int) or not isinstance(accepted, int):
                raise CheckError(f"{arm_name} request {index}: missing timings draft counters")
            if drafted < 0 or accepted < 0 or accepted > drafted:
                raise CheckError(f"{arm_name} request {index}: invalid counters {drafted}/{accepted}")
            if {"drafted": drafted, "accepted": accepted} != {
                "drafted": printed["drafted"],
                "accepted": printed["accepted"],
            }:
                raise CheckError(f"{arm_name} request {index}: JSON/log counter mismatch")
            cumulative_drafted += drafted
            cumulative_accepted += accepted
            if cumulative["drafted"] != cumulative_drafted or cumulative["accepted"] != cumulative_accepted:
                raise CheckError(f"{arm_name} request {index}: cumulative statistics mismatch")
            rows.append(
                {
                    "ordinal": index,
                    "row_id": request.get("row_id"),
                    "phase": request.get("phase"),
                    "rep": request.get("rep"),
                    "drafted_tokens": drafted,
                    "accepted_tokens": accepted,
                    "acceptance_rate": accepted / drafted if drafted else None,
                    "json_source": f"{json_name}:requests[{index}].timings",
                    "log_print_line": printed["line"],
                    "log_statistics_line": cumulative["line"],
                }
            )
        total_drafted = cumulative_drafted
        total_accepted = cumulative_accepted
        counter_status = "verified_json_timing_and_server_log"
        rate = total_accepted / total_drafted if total_drafted else None
    elif kind == "ngram":
        if print_rows or len(stats_rows) != 8:
            raise CheckError(f"{arm_name}: unexpected print_timing or statistics row count")
        for index, (request, cumulative) in enumerate(zip(requests, stats_rows)):
            if cumulative["drafted"] != 0 or cumulative["accepted"] != 0:
                raise CheckError(f"{arm_name} request {index}: nonzero ngram draft counters")
            rows.append(
                {
                    "ordinal": index,
                    "row_id": request.get("row_id"),
                    "phase": request.get("phase"),
                    "rep": request.get("rep"),
                    "drafted_tokens": 0,
                    "accepted_tokens": 0,
                    "acceptance_rate": None,
                    "log_statistics_line": cumulative["line"],
                }
            )
        total_drafted = 0
        total_accepted = 0
        rate = None
        counter_status = "verified_zero_denominator"
    else:
        if print_rows or stats_rows:
            raise CheckError(f"{arm_name}: ordinary baseline unexpectedly has speculative statistics")
        rows = [
            {
                "ordinal": index,
                "row_id": request.get("row_id"),
                "phase": request.get("phase"),
                "rep": request.get("rep"),
                "drafted_tokens": None,
                "accepted_tokens": None,
                "acceptance_rate": None,
            }
            for index, request in enumerate(requests)
        ]
        total_drafted = None
        total_accepted = None
        rate = None
        counter_status = "not_applicable"

    return {
        "arm": arm_name,
        "source_json": str(screen_dir / json_name),
        "source_server_log": str(screen_dir / log_name),
        "screen_status": data.get("status"),
        "protocol_status_counts": protocol_counts,
        "protocol_all_accepted": True,
        "request_order_used": "JSON requests order; log sequence cross-checked",
        "counter_status": counter_status,
        "requests": rows,
        "drafted_tokens_total": total_drafted,
        "accepted_tokens_total": total_accepted,
        "acceptance_rate": rate,
        "timing": _timing_summary(requests),
    }


def _pair_summary(screen_dir: Path) -> dict[str, Any]:
    data = _read_json(screen_dir / "pair-analysis.json")
    candidates = data.get("candidates", [])
    result: dict[str, Any] = {
        "source": str(screen_dir / "pair-analysis.json"),
        "prior_status": data.get("status"),
        "prior_status_interpretation": "acceptance fields were missing from analyzer summaries; per-request JSON timings contain the counters",
        "comparisons": [],
    }
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        comparison = candidate.get("comparison", {})
        result["comparisons"].append(
            {
                "candidate": comparison.get("candidate"),
                "status": comparison.get("status"),
                "requests_compared": comparison.get("requests_compared"),
                "requests_expected": comparison.get("requests_expected"),
                "greedy_parity": comparison.get("greedy_parity"),
                "mismatches": comparison.get("mismatches"),
                "speedup_vs_ordinary_baseline": comparison.get("speedup_vs_ordinary_baseline"),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    screen_dir = args.screen_dir.resolve()
    arms = {name: _extract_arm(screen_dir, name) for name in ARM_FILES}
    pair = _pair_summary(screen_dir)
    comparisons_pass = all(
        c["status"] == "pass"
        and c["requests_compared"] == c["requests_expected"] == 8
        and c["greedy_parity"] is True
        and c["mismatches"] == []
        for c in pair["comparisons"]
    )
    output = {
        "schema_version": "sepalith.r2.serving-screen-counter-extraction.v1",
        "status": "pass" if comparisons_pass else "fail",
        "screen_dir": str(screen_dir),
        "arms": arms,
        "pair_analysis": pair,
        "interpretation": {
            "dspark_acceptance_source": "requests[*].timings.draft_n and requests[*].timings.draft_n_accepted, cross-checked with print_timing and cumulative statistics",
            "ngram_acceptance": "not defined: server generated zero draft tokens, so denominator is zero",
            "ordinary_acceptance": "not applicable",
            "promotion_decision": "screen evidence alone does not promote an arm; trained/released latency and acceptance require the larger paired panel",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "out": str(args.out), "arms": sorted(arms)}))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
