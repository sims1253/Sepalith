#!/usr/bin/env python3
"""Coding simulator, stage 2 — the judge loop (user design 2026-08-23).

Walks a trajectory's suggestion points IN ORDER with a serving model
(the product's proposal engine). At each point:

  1. build the prompt WITH PROPOSAL HISTORY (the user's sequential
     conditioning): past proposals and their accept/reject outcomes
     rendered into the prompt's history channel as event lines —
         proposal dismissed: <first ~60 chars>...
         proposal accepted: <first ~60 chars>...
     (stage-2 v1 renders them into the suffix-window head as plain
     comment lines after a marked separator; the PSM <|history|> channel
     is the eventual home once the base model trains on it)
  2. the model proposes (midtyping-style completion, stop at UPDATED)
  3. ACCEPTANCE:
     - typing points: stage-1 rule — accept iff the proposal's first
       non-empty line prefix-matches the ground-truth chunk (the product
       accepts word-by-word); else dismissed
     - noop points: any non-empty proposal = FALSE SUGGESTION (recorded,
       feeds the episode metrics); empty = correct restraint
     - (LLM-judged acceptance vs the goal card is the next rung; the
       recorder below already emits everything the judge needs)
  4. the decision + proposal append to the history for SUBSEQUENT points

Episode outputs per trajectory: shown / accepted / dismissed / false-
suggestions / keystrokes-saved (accepted chars / typed chars), plus the
per-point records for RL consumption (prompt-with-history, proposal,
decision, label) — the acceptance/rejection history becomes training
signal exactly as the user specced.

Usage:
  uv run python judge_loop.py --model experiments/models/sft_v7_minicpm5-Q8_0.gguf \
    --traj /mnt/h/sepalith/datasets/sim_trajectories_v1/trajectories.jsonl \
    --n 100 --out /mnt/h/sepalith/datasets/sim_trajectories_v1/judged_v7.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "eval"))
from eval_noop_fp import parse_prediction  # noqa: E402  extension-faithful

CUDA_LLAMA = Path("/tmp/llamacpp-cuda-build/bin/llama-server")
UPDATED = ">>>>>>> UPDATED"
HIST_MARK = "#~ proposal history (this session):"


def complete(port: int, prompt: str, max_tokens: int = 160) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(dict(prompt=prompt, max_tokens=max_tokens,
                             temperature=0.0)).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["text"]


def with_history(prompt: str, history: list[dict]) -> str:
    """Render proposal history into the prompt (stage-2 v1: inject the
    event lines right after the <[fim-prefix]><filename> header — the
    region the extension controls and the model has seen comment-style
    content in). History rows: {proposal_head, accepted}."""
    if not history:
        return prompt
    lines = [HIST_MARK]
    for h in history[-8:]:     # cap at 8 events (context economy)
        verb = "accepted" if h["accepted"] else "dismissed"
        lines.append(f"#~ proposal {verb}: {h['proposal_head']}")
    # insert after the filename header line
    parts = prompt.split("\n")
    for i, l in enumerate(parts):
        if l.startswith("<[fim-prefix]>"):
            return "\n".join(parts[:i + 1] + lines + parts[i + 1:])
    return prompt


def first_content_line(pred_lines: list[str]) -> str:
    for l in pred_lines:
        if l.strip():
            return l
    return ""


def run_trajectory(traj: dict, port: int, cap_points: int = 30) -> dict:
    history: list[dict] = []
    records = []
    stats = dict(shown=0, accepted=0, dismissed=0, false_sug=0,
                 correct_stop=0, chars_saved=0, chars_typed=0)
    for p in traj["points"][:cap_points]:
        prompt = with_history(p["prompt"], history)
        try:
            comp = complete(port, prompt)
        except Exception:
            continue
        pred = parse_prediction(comp)
        head = first_content_line(pred)
        stats["shown"] += 1
        if p["label"] == "noop":
            if pred and head:
                decision, stats["false_sug"] = "false_suggestion", stats["false_sug"] + 1
            else:
                decision, stats["correct_stop"] = "correct_stop", stats["correct_stop"] + 1
            accepted = False
        else:
            gt = (p["gt"] or "").lstrip()
            ph = head.strip()
            accepted = bool(ph) and bool(gt) and \
                (gt.startswith(ph[:max(1, len(ph))])
                 or ph.startswith(gt.split("\n")[0][:len(ph)]))
            if accepted:
                decision, stats["accepted"] = "accepted", stats["accepted"] + 1
                stats["chars_saved"] += len(ph)
            else:
                decision, stats["dismissed"] = "dismissed", stats["dismissed"] + 1
        if p["label"] == "typing" and p.get("gt"):
            stats["chars_typed"] += len(p["gt"])
        history.append(dict(proposal_head=head[:60] if head else "(empty)",
                            accepted=bool(accepted)))
        records.append(dict(
            label=p["label"], ctx=p["ctx"], t_ms=p["t_ms"],
            prompt=prompt, proposal=comp[:600],
            pred_head=head[:120], decision=decision,
            gt=(p["gt"] or "")[:200]))
    return dict(key=traj["key"], variant=traj.get("variant", 1),
                goal=traj.get("goal"),
                stats=stats, points=records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18130)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    trajs = []
    for line in open(args.traj):
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if t.get("n_points", 0) >= 5:
            trajs.append(t)
    random.Random(3).shuffle(trajs)
    trajs = trajs[: args.n]
    print(f"trajectories: {len(trajs)}", flush=True)

    srv = None
    import subprocess
    # spawn our own server unless one already answers on the port
    # (external mode: never spawn, never kill)
    try:
        complete(args.port, "x", 1)
        print(f"[serve] external server :{args.port} answering", flush=True)
    except Exception:
        srv = subprocess.Popen(
            [str(CUDA_LLAMA), "-m", args.model, "--port", str(args.port),
             "--host", "127.0.0.1", "-c", "4096", "--parallel", "2",
             "-t", "8", "-ngl", "99"],
            stdout=open("/tmp/llama-server-judgeloop.log", "w"),
            stderr=subprocess.STDOUT)
    t0 = time.time()
    ready = False
    while time.time() - t0 < 300:
        try:
            complete(args.port, "x", 1)
            ready = True
            break
        except Exception:
            time.sleep(4)
    if not ready:
        if srv:
            srv.kill()
        sys.exit("server never ready")

    agg = dict(shown=0, accepted=0, dismissed=0, false_sug=0,
               correct_stop=0, chars_saved=0, chars_typed=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    t0 = time.time()
    with open(out, "w") as fh, ThreadPoolExecutor(max_workers=2) as ex:
        for res in ex.map(lambda t: run_trajectory(t, args.port), trajs):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            for k in agg:
                agg[k] += res["stats"][k]
            n_done += 1
            if n_done % 10 == 0:
                print(f"[{n_done}] elapsed={time.time()-t0:.0f}s", flush=True)
    if srv:
        srv.terminate()
    print(json.dumps(dict(n=n_done, **agg,
                          accept_rate=round(agg["accepted"] /
                                            max(1, agg["shown"]), 3),
                          fp_rate=round(agg["false_sug"] /
                                        max(1, agg["false_sug"] +
                                            agg["correct_stop"]), 3),
                          saved_ratio=round(agg["chars_saved"] /
                                            max(1, agg["chars_typed"]), 3))),
          flush=True)


if __name__ == "__main__":
    main()
