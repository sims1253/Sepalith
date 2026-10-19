#!/usr/bin/env python3
"""Replay cascade routing over a finished results.jsonl.

The 2026-10-16 full run recorded `route: run` for all 27 rollouts because
d3's score answers surfaced their numeric EXPECTATION ("2.45...") instead
of the criteria label, so `cascade_decision` never matched good/elegant.
The normalization is fixed in d3_client; rather than re-paying 28 minutes
of generation, this script re-derives each row's quality label from the
logged d3 probs/expectation and computes the route the FIXED router would
have taken — into a separate `route_replay` field, leaving the recorded
`skip_reason` (what actually happened) untouched.

Usage:
  python3 route_replay.py [--inp results.jsonl] [--regen]
  --regen rewrites RESULTS.md / dashboard.html with the replay column.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from d3_client import TERMINAL_QUESTIONS
from env import cascade_decision, glm_style_score
from reports import write_dashboard, write_results_md

HERE = Path(__file__).resolve().parent
CRIT = TERMINAL_QUESTIONS["quality"]["criteria"]


def quality_label(d3_q: dict) -> tuple[str, float | None]:
    """Fixed normalization over a logged d3 quality answer."""
    probs = d3_q.get("probs", {})
    exp = d3_q.get("expectation")
    if exp is None:
        for cand in (d3_q.get("label"),):
            try:
                exp = float(cand)
                break
            except (TypeError, ValueError):
                continue
    if exp is not None:
        return CRIT[max(0, min(len(CRIT) - 1, int(round(exp))))], float(exp)
    # fall back to probability-weighted expectation over criteria levels
    if probs and set(map(str, probs)) <= set(CRIT):
        return max(probs, key=probs.get), sum(
            CRIT.index(k) * v for k, v in probs.items())  # type: ignore
    return "acceptable", None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", default=str(HERE / "results.jsonl"))
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    routes = {}
    for r in rows:
        q = r.get("d3_terminal", {}).get("quality", {})
        label, exp = quality_label(q)
        q["label_fixed"] = label
        if exp is not None:
            q["expectation"] = exp
        rr = cascade_decision(
            r["correctness"], {
                "quality": {"label": label},
                "hacky": r["d3_terminal"].get("hacky", {}),
                "looks_correct": r["d3_terminal"].get("looks_correct", {}),
            })
        r["route_replay"] = rr
        routes[rr] = routes.get(rr, 0) + 1

    out = Path(args.inp).with_name(
        Path(args.inp).stem + "_rerouted.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    saved = routes.get("clean", 0) + routes.get("dead", 0)
    print(f"replayed routes over {n} rollouts: {routes}")
    print(f"big-judge calls the fixed router would have skipped: "
          f"{saved}/{n} ({100 * saved / n:.0f}%)")
    # sanity: quality-fixed label vs judge style correlation
    pairs = [(r["d3_terminal"]["quality"].get("label_fixed"),
              glm_style_score(r.get("glm_judge") or {})) for r in rows]
    by = {}
    for lbl, s in pairs:
        if s is not None:
            by.setdefault(lbl, []).append(s)
    for lbl in sorted(by, key=lambda x: CRIT.index(x)):
        print(f"  d3 quality={lbl:10s} n={len(by[lbl])} "
              f"mean glm style {statistics.mean(by[lbl]):.2f}")

    if args.regen:
        comp_s = round(sum(
            v for r in rows
            for v in (r.get("timings") or {}).values()
            if isinstance(v, (int, float))), 1)
        meta = {
            "rounds": max(r.get("round", 0) for r in rows) + 1,
            "rollouts": max(r.get("rollout", 0) for r in rows) + 1,
            "glm_off": False, "d3_backend": rows[0].get("backend", "d3"),
            "wall_s": comp_s,
            "generated_at": "route-replay of the 2026-10-16 run "
                            "(component-time sum; run wall was 1705s)",
            "glm_usage": {"prompt_tokens": 32682,
                          "completion_tokens": 14904, "calls": 79},
        }
        write_results_md(rows, meta, HERE / "RESULTS.md")
        write_dashboard(rows, meta, HERE / "dashboard.html")
        print("regenerated RESULTS.md + dashboard.html with route_replay")


if __name__ == "__main__":
    main()
