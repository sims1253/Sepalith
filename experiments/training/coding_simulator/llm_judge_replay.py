#!/usr/bin/env python3
"""Panel-judge acceptance, replayed over recorded judge-loop records.

USER DIRECTIVES (2026-08-24): (1) gemma-free is too small a judge — the
panel is ox-alpha + gemini (agy) + muse (glm rejoins at quota reset),
majority vote, per-judge verdicts recorded; (2) acceptance baseline is
VERBATIM-ONLY: accept iff the developer would take the suggestion exactly
as proposed with NO EDITS needed — "usefully starts" is not acceptance.

Replays the RECORDED proposals from judge_loop output (no model server
needed). NOOP points keep the rule-based accounting.

Usage:
  OPENCODE/OPENROUTER keys exported; then:
  python3 llm_judge_replay.py --inp judged_v7.jsonl --out judged_v7_panel.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_judge import Panel, ACCEPT_PROMPT  # noqa: E402


def context_tail(prompt: str, n: int = 12) -> str:
    lines = prompt.split("\n")
    try:
        i = lines.index("<<<<<<< CURRENT")
    except ValueError:
        i = len(lines)
    return "\n".join(lines[max(0, i - n):i])[-1200:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2,
                    help="panel votes are sequential inside; keep small")
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.inp)]
    jobs = []
    for ti, tr in enumerate(trajs):
        for pi, p in enumerate(tr["points"]):
            if p["label"] == "typing" and p.get("pred_head") \
                    and p.get("gt") and len(jobs) < args.limit:
                jobs.append((ti, pi))
    print(f"typing points to panel-judge: {len(jobs)}", flush=True)

    panel = Panel()

    def work(job):
        ti, pi = job
        tr = trajs[ti]
        p = tr["points"][pi]
        prompt = ACCEPT_PROMPT.format(
            goal=tr.get("goal") or "(unknown goal)",
            ctx=context_tail(p["prompt"]),
            gt=p["gt"][:300], prop=p["pred_head"][:300])
        v, per = panel.vote(prompt)
        return job, v, per

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for (ti, pi), v, per in ex.map(work, jobs):
            trajs[ti]["points"][pi]["panel"] = v
            trajs[ti]["points"][pi]["panel_votes"] = per
    counts = {}
    for ti, pi in jobs:
        counts[trajs[ti]["points"][pi].get("panel", "bad")] = \
            counts.get(trajs[ti]["points"][pi].get("panel", "bad"), 0) + 1
    with open(args.out, "w") as fh:
        for tr in trajs:
            fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
    n = sum(counts.values())
    print(json.dumps(dict(judged=n, **counts,
                          panel_accept_rate=round(
                              counts.get("accept", 0) / max(1, n), 3),
                          elapsed_s=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
