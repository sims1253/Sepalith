#!/usr/bin/env python3
"""Ox-alpha goal-card writer (user directive 2026-08-23 night): sol quota
is parked for the reset; ox-alpha (openrouter stealth) takes over the
commit-goal corpus — first the ~3.2k transitions that errored when the
sol quota died, then fresh commits beyond the sol sweep.

Same card schema as mine_commit_goals.py (goal/changes/domain/difficulty
+ repo/sha provenance) with a `model` field; lenient JSON extraction
(ox gets no format pressure — the OxCalpha12k lesson). Output:
commit_goals_v1/cards_ox.jsonl + done_ox.txt (ox's own resume keys; sol
successes in done.txt are skipped so ox only writes NEW cards).

Blind-quality round (the user's ask): --judge mode samples commits with
BOTH sol and ox cards, asks a FREE-tier judge (gemma-4-31b-it:free via
openrouter) a randomized blind pairwise "which card better captures the
developer's goal", and logs win rates. Never uses gpt-5.6-sol.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_commit_goals import (  # noqa: E402  shared diff machinery
    GIT, commit_diff, PROMPT)

OUT = Path("/mnt/h/sepalith/datasets/commit_goals_v1")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OX_MODEL = "stealth/ox-alpha"
JUDGE_MODEL = "google/gemma-4-31b-it:free"


def or_call(payload: dict, timeout: int = 240) -> str:
    req = urllib.request.Request(
        OR_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        card = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(card.get("goal"), str) or not card["goal"].strip():
        return None
    if not isinstance(card.get("changes"), list):
        return None
    return card


def ox_card(repo: str, date: str, blob: str, nfiles: int, nch: int):
    txt = or_call({"model": OX_MODEL, "max_tokens": 2000,
                   "messages": [{"role": "user", "content": PROMPT.format(
                       repo=repo, date=date, nfiles=nfiles, nch=nch,
                       diff=blob)}]})
    card = extract_json(txt)
    if card is None:
        # one retry with an explicit JSON nudge (lenient, not format pressure)
        txt2 = or_call({"model": OX_MODEL, "max_tokens": 2000,
                        "messages": [
                            {"role": "user", "content": PROMPT.format(
                                repo=repo, date=date, nfiles=nfiles,
                                nch=nch, diff=blob)},
                            {"role": "assistant", "content": txt},
                            {"role": "user", "content":
                             "Reply again with ONLY the JSON object."}]})
        card = extract_json(txt2)
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge", action="store_true",
                    help="run the blind sol-vs-ox quality round instead")
    ap.add_argument("--judge-n", type=int, default=120)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sol_done = set()
    dp = OUT / "done.txt"
    if dp.exists():
        sol_done = {l.strip() for l in dp.read_text().splitlines() if l.strip()}

    if args.judge:
        judge_round(args.judge_n)
        return

    cards_p = OUT / "cards_ox.jsonl"
    done_p = OUT / "done_ox.txt"
    done = set()
    if done_p.exists():
        done = {l.strip() for l in done_p.read_text().splitlines() if l.strip()}

    repo_dirs = sorted(p for p in GIT.iterdir() if p.is_dir())
    print(f"repos {len(repo_dirs)} | sol-done {len(sol_done)} | "
          f"ox-done {len(done)}", flush=True)
    t0, n_ok, n_err = time.time(), 0, 0

    def work(job):
        key, repo_name, sha = job
        d = commit_diff(GIT / repo_name, sha)
        if d is None:
            return None
        blob, nfiles, nch, date = d
        try:
            card = ox_card(repo_name, date, blob, nfiles, nch)
        except Exception:
            return ("err", key)
        if card is None:
            return ("err", key)
        return ("ok", dict(key=key, repo=repo_name, sha=sha, date=date,
                           n_files=nfiles, changed_lines=nch,
                           model="ox-alpha", **card))

    for repo in repo_dirs:
        if args.limit and n_ok >= args.limit:
            break
        try:
            shas = subprocess.run(
                ["git", "-C", str(repo), "log", "--format=%H", "-n", "40",
                 "--", "*.R", "*.r"], capture_output=True, text=True,
                errors="replace", timeout=30).stdout.split()
        except (subprocess.TimeoutExpired, OSError):
            continue
        random.Random(f"cgo:{repo.name}").shuffle(shas)
        jobs = [(f"{repo.name}@{sha[:12]}", repo.name, sha)
                for sha in shas[:8]
                if f"{repo.name}@{sha[:12]}" not in sol_done
                and f"{repo.name}@{sha[:12]}" not in done]
        if not jobs:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(work, jobs):
                if res is None:
                    continue
                kind, payload = res
                if kind == "err":
                    n_err += 1
                    continue
                with open(cards_p, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                with open(done_p, "a") as fh:
                    fh.write(payload["key"] + "\n")
                n_ok += 1
        if n_ok and n_ok % 100 == 0:
            print(f"[{n_ok}] err={n_err} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    print(f"DONE ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s",
          flush=True)


def judge_round(n: int):
    """Blind pairwise: sol card vs ox card for the same commit; gemma
    (free tier) judges which better captures the developer's goal."""
    sol = {}
    for line in open(OUT / "cards.jsonl"):
        try:
            c = json.loads(line)
            sol.setdefault(c["key"], c)
        except ValueError:
            pass
    ox = {}
    p = OUT / "cards_ox.jsonl"
    if p.exists():
        for line in open(p):
            try:
                c = json.loads(line)
                ox.setdefault(c["key"], c)
            except ValueError:
                pass
    both = sorted(set(sol) & set(ox))
    print(f"commits with both cards: {len(both)}", flush=True)
    if not both:
        return
    random.Random(31).shuffle(both)
    both = both[: n]
    wins = {"sol": 0, "ox": 0, "tie": 0, "bad": 0}

    def one(key):
        a, b = sol[key], ox[key]
        flip = random.Random(f"j:{key}").random() < 0.5
        first, second = (b, a) if flip else (a, b)
        prompt = (
            "Two analysts wrote a goal card for the same git commit "
            "(what the developer was trying to achieve). Judge which card "
            "better captures the goal: more specific, more accurate to the "
            "changes, more useful for deciding whether a proposed code "
            "suggestion advances the commit.\n\n"
            f"Commit: {a['repo']} ({a.get('date', '')}), "
            f"{a.get('changed_lines', '?')} changed lines\n\n"
            f"CARD A:\n{json.dumps({k: first.get(k) for k in ('goal', 'changes')}, indent=1)}\n\n"
            f"CARD B:\n{json.dumps({k: second.get(k) for k in ('goal', 'changes')}, indent=1)}\n\n"
            "Reply with ONLY one word: A, B, or TIE.")
        try:
            txt = or_call({"model": JUDGE_MODEL, "max_tokens": 200,
                           "messages": [{"role": "user", "content": prompt}]},
                          timeout=120).strip().upper()[:4]
        except Exception:
            return "bad"
        if txt.startswith("TIE"):
            return "tie"
        pick = "A" if txt.startswith("A") else "B" if txt.startswith("B") else None
        if pick is None:
            return "bad"
        winner = "sol" if (pick == "A") != flip else "ox"
        return winner

    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(one, both):
            wins[r if r in wins else "bad"] += 1
    total = wins["sol"] + wins["ox"] + wins["tie"]
    print(json.dumps(dict(
        n=total, sol_wins=wins["sol"], ox_wins=wins["ox"], ties=wins["tie"],
        unjudgeable=wins["bad"],
        sol_winrate=round(wins["sol"] / max(1, total), 3),
        ox_winrate=round(wins["ox"] / max(1, total), 3))), flush=True)
    (OUT / "judge_round1.json").write_text(json.dumps(wins, indent=1))


if __name__ == "__main__":
    main()
