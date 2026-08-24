#!/usr/bin/env python3
"""LLM-judge acceptance, replayed over recorded judge-loop records (the
stage-2 rung from coding_simulator/design.md — user design: acceptance
judged by the commit's SHAPE/GOAL, not literal match).

Input: judged_*.jsonl from judge_loop.py (per-point: prompt tail, proposal
head, gt, label, plus the trajectory's goal card). For TYPING points with
a non-empty proposal, a free-tier judge (gemma-4-31b-it:free via
openrouter; ox fallback) sees: the commit goal card, the code context
around the cursor, the developer's ACTUAL next chunk (gt), and the
model's proposal — and picks:

    accept  — the developer would have accepted this suggestion
    reject  — plausible but not what they wanted
    wrong   — wrong shape/content for this position

NOOP points keep the rule-based accounting (any proposal = false
suggestion; nothing for a judge to weigh).

Output: judged_*_llm.jsonl (records + judge verdicts) + a summary line.

Usage:
  OPENROUTER_API_KEY=... python3 llm_judge_replay.py \
    --inp /mnt/.../judged_v7.jsonl --out /mnt/.../judged_v7_llm.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "google/gemma-4-31b-it:free"


def call(payload: dict, timeout: int = 120) -> str:
    req = urllib.request.Request(
        OR_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def context_tail(prompt: str, n: int = 12) -> str:
    """The last n lines before the CURRENT marker — where the cursor is."""
    lines = prompt.split("\n")
    try:
        i = lines.index("<<<<<<< CURRENT")
    except ValueError:
        i = len(lines)
    return "\n".join(lines[max(0, i - n):i])[-1200:]


def judge_point(goal: str, ctx: str, gt: str, prop: str) -> str:
    p = (
        "You are simulating a developer's acceptance decision for a code "
        "suggestion in their editor.\n\n"
        f"THEIR GOAL (this commit): {goal}\n\n"
        f"CODE AT THE CURSOR (they just paused here):\n```\n{ctx}\n```\n\n"
        f"WHAT THEY ACTUALLY TYPED NEXT:\n```\n{gt[:300]}\n```\n\n"
        f"THE SUGGESTION SHOWN:\n```\n{prop[:300]}\n```\n\n"
        "Would they accept the suggestion (it matches or usefully starts "
        "what they were about to type), reject it (plausible but not their "
        "intent), or is it wrong for this position? Reply with ONLY one "
        "word: accept, reject, or wrong.")
    for attempt in range(2):
        try:
            t = call({"model": JUDGE_MODEL, "max_tokens": 300,
                      "messages": [{"role": "user", "content": p}]}
                     ).strip().lower()
            for v in ("accept", "reject", "wrong"):
                if v in t:
                    return v
        except Exception:
            time.sleep(3)
    return "bad"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400,
                    help="max typing points to judge (cost cap)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.inp)]
    jobs = []          # (traj_idx, pt_idx)
    for ti, tr in enumerate(trajs):
        for pi, p in enumerate(tr["points"]):
            if p["label"] == "typing" and p.get("pred_head") \
                    and p.get("gt") and len(jobs) < args.limit:
                jobs.append((ti, pi))
    print(f"typing points to judge: {len(jobs)}", flush=True)

    def work(job):
        ti, pi = job
        tr = trajs[ti]
        p = tr["points"][pi]
        v = judge_point(tr.get("goal") or "(unknown goal)",
                        context_tail(p["prompt"]), p["gt"], p["pred_head"])
        return job, v

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for (ti, pi), v in ex.map(work, jobs):
            trajs[ti]["points"][pi]["llm_judge"] = v
    counts = {"accept": 0, "reject": 0, "wrong": 0, "bad": 0}
    for job, _ in []:
        pass
    for ti, pi in jobs:
        counts[trajs[ti]["points"][pi].get("llm_judge", "bad")] += 1
    with open(args.out, "w") as fh:
        for tr in trajs:
            fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
    n = sum(counts.values())
    print(json.dumps(dict(judged=n, **counts,
                          llm_accept_rate=round(counts["accept"] /
                                                max(1, n), 3),
                          elapsed_s=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
