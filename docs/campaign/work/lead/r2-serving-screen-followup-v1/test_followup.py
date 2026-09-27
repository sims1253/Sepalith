#!/usr/bin/env python3
"""Small CPU contract tests for the packet-local extractor and panel."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import extract_screen_counters as counters


SCREEN = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-four-arm-screen-v1")
PACKET = Path(__file__).resolve().parent


def main() -> None:
    released = counters._extract_arm(SCREEN, "released-dspark")
    trained = counters._extract_arm(SCREEN, "trained-dspark")
    ngram = counters._extract_arm(SCREEN, "model-free-ngram")
    ordinary = counters._extract_arm(SCREEN, "ordinary-baseline")
    assert (released["drafted_tokens_total"], released["accepted_tokens_total"]) == (392, 102)
    assert (trained["drafted_tokens_total"], trained["accepted_tokens_total"]) == (462, 92)
    assert ngram["counter_status"] == "verified_zero_denominator"
    assert ordinary["counter_status"] == "not_applicable"

    # A changed JSON counter must fail against the immutable server log.
    with tempfile.TemporaryDirectory(prefix="r2-screen-counter-test-") as tmp:
        root = Path(tmp)
        for name in ("released-dspark.json", "released-dspark-server.log"):
            shutil.copy2(SCREEN / name, root / name)
        changed = json.loads((root / "released-dspark.json").read_text())
        changed["requests"][0]["timings"]["draft_n"] += 1
        (root / "released-dspark.json").write_text(json.dumps(changed))
        try:
            counters._extract_arm(root, "released-dspark")
        except counters.CheckError:
            pass
        else:
            raise AssertionError("counter mismatch was accepted")

    manifest = json.loads((PACKET / "train-serving-panel.manifest.json").read_text())
    rows = (PACKET / "train-serving-panel.jsonl").read_bytes().splitlines()
    assert manifest["status"] == "pass"
    assert len(rows) == manifest["panel"]["row_count"] == 40
    assert manifest["panel"]["contains_final_or_dev"] is False
    assert manifest["eight_k_stress"]["status"] == "unsupported_daily_context4096"
    print("screen counter mismatch, denominator, and panel contract tests PASS")


if __name__ == "__main__":
    main()
