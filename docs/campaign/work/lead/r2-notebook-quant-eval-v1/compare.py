#!/usr/bin/env python3
"""Compare four hash-bound V1 arm results on identical cases."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ARMS = ("q8", "q4_calibrated", "iq3", "iq2")


def pctl(values: list[float], percentile: float):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (rank - lo), 3)


def timings(records: list[dict], phase: str, deadline_ms: int) -> dict:
    selected = [r for r in records if r["phase"] == phase]
    def collect(field):
        return [float(r[field]) for r in selected if isinstance(r.get(field), (int, float))]
    def server(field):
        return [float(r.get("final", {}).get("timings", {}).get(field)) for r in selected
                if isinstance(r.get("final", {}).get("timings", {}).get(field), (int, float))]
    return {
        "requests": len(selected),
        "accepted": sum(r.get("protocol_status") == "accepted" for r in selected),
        "deadline_ms": deadline_ms,
        "deadline_success": sum(float(r.get("combined_case_wall_ms", 1e12)) <= deadline_ms and r.get("protocol_status") == "accepted" for r in selected),
        "end_to_end_ms": {"p50": pctl(collect("combined_case_wall_ms"), .5), "p95": pctl(collect("combined_case_wall_ms"), .95)},
        "ttft_ms": {"p50": pctl(collect("ttft_ms"), .5), "p95": pctl(collect("ttft_ms"), .95)},
        "prefill_ms": {"p50": pctl(server("prompt_ms"), .5), "p95": pctl(server("prompt_ms"), .95)},
        "decode_ms": {"p50": pctl(server("predicted_ms"), .5), "p95": pctl(server("predicted_ms"), .95)}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--diagnostic", action="append", type=Path)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    expected_arms = tuple(args.arms.split(","))
    if not expected_arms or expected_arms[0] != "q8" or any(arm not in ARMS for arm in expected_arms):
        raise SystemExit("arms must be a comma-separated subset beginning with q8")
    by_arm = {result["arm"]: result for result in results}
    if set(by_arm) != set(expected_arms):
        raise SystemExit("exactly one result for each requested arm is required")
    case_sets = [{(r["row_id"], r["phase"], r["rep"]) for r in result["requests"]} for result in results]
    if len({frozenset(cases) for cases in case_sets}) != 1:
        raise SystemExit("arm case identities are not paired")
    summary = {"schema": "sepalith.r2.notebook-quant-comparison.v1", "paired": True,
               "diagnostic_latency_included": False, "arms": {}}
    for arm in expected_arms:
        result = by_arm[arm]
        summary["arms"][arm] = {"model": result["model"], "quality": result["quality"],
                                 "cold": timings(result["requests"], "cold", 5000),
                                 "warm": timings(result["requests"], "warm", 5000)}
    q8 = summary["arms"]["q8"]
    for arm in expected_arms[1:]:
        current = summary["arms"][arm]
        current["relative_to_q8"] = {
            "strict_exact_target_text_delta": current["quality"]["strict_exact_target_text"] - q8["quality"]["strict_exact_target_text"],
            "strict_no_edit_target_exact_delta": current["quality"]["strict_no_edit_target_exact"] - q8["quality"]["strict_no_edit_target_exact"],
            "cold_deadline_success_delta": current["cold"]["deadline_success"] - q8["cold"]["deadline_success"],
            "warm_deadline_success_delta": current["warm"]["deadline_success"] - q8["warm"]["deadline_success"]
        }
    if args.diagnostic:
        diagnostics = [json.loads(path.read_text(encoding="utf-8")) for path in args.diagnostic]
        diagnostics_by_arm = {result["arm"]: result for result in diagnostics}
        if set(diagnostics_by_arm) != set(expected_arms):
            raise SystemExit("exactly one diagnostic for each requested arm is required")
        diagnostic_case_sets = [
            {(r["row_id"], r["phase"], r["rep"]) for r in result["requests"]}
            for result in diagnostics
        ]
        diagnostics_paired = len({frozenset(cases) for cases in diagnostic_case_sets}) == 1
        summary["diagnostic_60000"] = {
            "paired": diagnostics_paired,
            "case_ids_by_arm": {
                arm: sorted({record["row_id"] for record in diagnostics_by_arm[arm]["requests"]})
                for arm in expected_arms
            },
            "excluded_from_five_second_metrics": True,
            "purpose": "decompose prefill/decode and inspect quality after each arm's first real-deadline failure; compare arms only when paired is true",
            "arms": {
                arm: {
                    "quality": diagnostics_by_arm[arm].get("quality"),
                    "cold": timings(diagnostics_by_arm[arm]["requests"], "cold", 60000),
                    "warm": timings(diagnostics_by_arm[arm]["requests"], "warm", 60000),
                }
                for arm in expected_arms
            }
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "compared", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
