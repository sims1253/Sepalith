#!/usr/bin/env python3
"""Compare two retained RUN-04 trace JSON files without contacting a server."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(row: dict, name: str):
    timings = row["responseMetrics"]["timings"]
    top = row["responseMetrics"]["topLevel"]
    if name == "eval":
        return top["tokens_evaluated"]
    if name == "prefill":
        return timings["prompt_n"]
    if name == "decode":
        return timings["predicted_n"]
    if name == "prompt_ms":
        return timings["prompt_ms"]
    if name == "decode_ms":
        return timings["predicted_ms"]
    if name == "elapsed_ms":
        return row["elapsedMs"]
    raise KeyError(name)


def summary(rows: list[dict]) -> dict:
    result = {}
    for name in ("elapsed_ms", "prompt_ms", "decode_ms"):
        values = [metric(row, name) for row in rows]
        result[name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} THETA0_TRACE PRIOR_TRACE")
    theta, prior = read(sys.argv[1]), read(sys.argv[2])
    theta_rows, prior_rows = theta["rows"], prior["rows"]
    assert len(theta_rows) == len(prior_rows) == 9
    assert [row["eventId"] for row in theta_rows] == [row["eventId"] for row in prior_rows]
    assert theta["nativeProfile"] == prior["nativeProfile"]
    assert theta["source"] == prior["source"]

    rows = []
    for current, baseline in zip(theta_rows, prior_rows):
        assert current["promptTextSha256"] == baseline["promptTextSha256"]
        assert current["promptTokenIdsSha256"] == baseline["promptTokenIdsSha256"]
        for name in ("eval", "prefill"):
            assert metric(current, name) == metric(baseline, name)
        rows.append(
            {
                "event": current["eventId"],
                "eval": metric(current, "eval"),
                "prefill": metric(current, "prefill"),
                "theta0_generated": metric(current, "decode"),
                "prior_generated": metric(baseline, "decode"),
                "theta0_elapsed_ms": metric(current, "elapsed_ms"),
                "prior_elapsed_ms": metric(baseline, "elapsed_ms"),
                "theta0_prompt_ms": metric(current, "prompt_ms"),
                "prior_prompt_ms": metric(baseline, "prompt_ms"),
                "theta0_decode_ms": metric(current, "decode_ms"),
                "prior_decode_ms": metric(baseline, "decode_ms"),
                "generated_hash_equal": current["generatedTokenIdsSha256"]
                == baseline["generatedTokenIdsSha256"],
            }
        )

    output = {
        "assertions": "passed",
        "rows": rows,
        "theta0_summary": summary(theta_rows),
        "prior_summary": summary(prior_rows),
        "elapsed_delta_mean_ms": statistics.mean(
            [metric(a, "elapsed_ms") - metric(b, "elapsed_ms") for a, b in zip(theta_rows, prior_rows)]
        ),
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
