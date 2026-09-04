#!/usr/bin/env python3
"""summarize.py — bake-off summary: per-arm tables, held-out margins, budget
accounting (rollouts + proposer tokens), for H1_RESULTS.md and the board.

  python3 experiments/harness_search/summarize.py [--results results] [--md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def arm_summary(results: Path, arm: str) -> dict:
    state = json.loads((results / arm / "state.json").read_text())
    ev = state["evaluated"]
    led = [json.loads(l) for l in
           open(results / arm / "proposer_ledger.jsonl")] if (
        results / arm / "proposer_ledger.jsonl").exists() else []
    ok = [l for l in led if l.get("ok")]
    return dict(
        arm=arm, n_candidates=len(ev),
        completions=sum(e["completions"] for e in ev),
        pair_reuses=sum(e["pair_reuses"] for e in ev),
        best_fp=state["best_entry"]["fp"],
        best_cfg=state["best_entry"]["cfg"], best_texts=state["best_entry"]["texts"],
        best_dh_exact=state["best_entry"]["exact_pass"],
        proposer_calls=len(led), proposer_ok=len(ok),
        prompt_tokens=sum(l.get("prompt_tokens", 0) for l in ok),
        completion_tokens=sum(l.get("completion_tokens", 0) for l in ok),
        total_tokens=sum(l.get("total_tokens", 0) for l in ok),
        reasoning_tokens=sum(l.get("reasoning_tokens", 0) for l in ok),
        fallback_iters=len({e["iter"] for e in ev if e["origin"] == "fallback"}),
        proposer_candidates=sum(1 for e in ev if e["origin"] == "proposer"),
        evaluated=[dict(iter=e["iter"], fp=e["fp"], origin=e["origin"],
                        exact=e["exact_pass"], unstable=e["unstable_frac"],
                        p95=e["p95_latency_s"], noop=e["noop"]["proposal_rate"],
                        guard_ok=bool(e["guard"] and e["guard"]["noop_ok"]
                                      and e["guard"]["latency_ok"]),
                        completions=e["completions"],
                        cfg=e["cfg"], texts=e["texts"]) for e in ev],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    results = Path(args.results)
    out = {"baseline": None, "arms": {}}
    if (results / "baseline.json").exists():
        b = json.loads((results / "baseline.json").read_text())
        out["baseline"] = dict(exact=b["exact_pass"], unstable=b["unstable_frac"],
                               p95=b["p95_latency_s"],
                               noop=b["noop"]["proposal_rate"],
                               completions=b["completions"],
                               families=b["families"])
    for arm in ("hill", "population", "gepa"):
        if (results / arm / "state.json").exists():
            out["arms"][arm] = arm_summary(results, arm)
    if (results / "verdict.json").exists():
        out["verdict"] = json.loads((results / "verdict.json").read_text())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
