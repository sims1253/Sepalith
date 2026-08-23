#!/usr/bin/env python3
"""Commit-goal corpus builder (the judged-acceptance RL asset — user idea
2026-08-23: work streams from the PARENT COMMIT, acceptance judged by the
commit's shape/goal rather than literal match).

Source: the license-audited git mirror /mnt/h/sepalith/git (2,586 R repos,
real commit history — one commit ≈ one work session by construction). For
each repo, sample up to --per-repo commits touching R sources; for each
commit, diff against its parent and ask gpt-5.6-sol (9router, codex-reset
quota burn) for a GOAL CARD:

    {"goal": "<one sentence: what the developer was trying to achieve>",
     "changes": ["<3-6 short bullet phrases>"],
     "domain": "<seed domain or 'general'>",
     "difficulty": "trivial|small|medium|large"}

The card conditions (a) the coding simulator's judge-based acceptance
policy (coding_simulator/design.md), (b) summary-network RL (workspace
context at the parent commit judged against the goal), (c) future
goal-conditioned SFT rows.

Resume: done.txt lists repo@sha keys. Output:
/mnt/h/sepalith/datasets/commit_goals_v1/cards.jsonl.

Usage:
  python3 mine_commit_goals.py --workers 6 [--per-repo 8] [--repos N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GIT = Path("/mnt/h/sepalith/git")
OUT = Path("/mnt/h/sepalith/datasets/commit_goals_v1")
URL = "http://localhost:20128/v1/chat/completions"
MAX_DIFF_CHARS = 7000
MIN_CHANGED_LINES = 3
MAX_FILES = 15

PROMPT = """Below is a unified diff of one commit in the R project `{repo}` (date {date}). Write the developer's goal card for this commit, as if it was the work session they were developing.

Reply with ONLY this JSON:
{{"goal": "<one sentence, imperative: what they were trying to achieve>",
  "changes": ["<3-6 short phrases, each one concrete change>"],
  "domain": "<one of: gis/spatial, ecology, epidemiology, genomics, finance, econometrics/forecasting, web/api, weather/climate, psychometrics, pharma/clinical, sports/analytics, agriculture, energy/grid, transport/traffic, insurance/actuarial, imaging/microscopy, audio/synthesis/music, chemometrics/spectroscopy, geotechnics/tunnel, ocean/marine, general>",
  "difficulty": "trivial|small|medium|large"}}

Diff (files: {nfiles}, changed lines: {nch}):

```diff
{diff}
```"""


def git(repo: Path, *args: str, timeout: int = 30) -> str:
    # errors="replace": repo diffs can carry Windows-1252 bytes (0x92
    # smart quotes killed a run); the card model tolerates replacements
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True,
                          errors="replace", timeout=timeout
                          ).stdout


def commit_diff(repo: Path, sha: str) -> tuple[str, int, int, str] | None:
    """(capped diff, nfiles, nch, date) or None if not worth a card."""
    try:
        meta = git(repo, "show", "--format=%ad", "--date=short", "-s", sha)
        date = meta.strip() or ""
        raw = git(repo, "show", "--format=", "--unified=2",
                  f"{sha}^", sha, "--", "*.R", "*.r", timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None
    files = [l for l in raw.splitlines() if l.startswith("+++ ")]
    if not files or len(files) > MAX_FILES:
        return None
    nch = sum(1 for l in raw.splitlines()
              if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    if nch < MIN_CHANGED_LINES or nch > 400:
        return None
    blob = raw[:MAX_DIFF_CHARS]
    if len(raw) > MAX_DIFF_CHARS:
        blob += "\n... (truncated)"
    return blob, len(files), nch, date


def sol_card(repo: str, date: str, blob: str, nfiles: int,
             nch: int) -> dict | None:
    payload = {
        "model": "cx/gpt-5.6-sol", "max_tokens": 700,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": PROMPT.format(
            repo=repo, date=date, nfiles=nfiles, nch=nch, diff=blob)}],
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('NINE_ROUTER_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=240) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    card = json.loads(txt)
    if not isinstance(card.get("goal"), str) or not card["goal"].strip():
        return None
    if not isinstance(card.get("changes"), list):
        return None
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-repo", type=int, default=8,
                    help="commits sampled per repo")
    ap.add_argument("--repos", type=int, default=0,
                    help="limit repo count (0 = all)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cards_p = OUT / "cards.jsonl"
    done_p = OUT / "done.txt"

    done = set()
    if done_p.exists():
        done = {l.strip() for l in done_p.read_text().splitlines() if l.strip()}

    repo_dirs = sorted(p for p in GIT.iterdir() if p.is_dir())
    if args.repos:
        random.Random(11).shuffle(repo_dirs)
        repo_dirs = repo_dirs[: args.repos]
    print(f"repos: {len(repo_dirs)} | done keys: {len(done)}", flush=True)

    t0 = time.time()
    n_ok = n_err = n_scan = 0

    def repo_jobs(repo: Path):
        """(key, repo, sha) candidates for one repo — sampled commits."""
        try:
            shas = git(repo, "log", "--format=%H", "--date=short", "-n",
                       str(max(40, args.per_repo * 5)), "--", "*.R", "*.r"
                       ).split()
        except (subprocess.TimeoutExpired, OSError):
            return []
        if not shas:
            return []
        random.Random(f"cg:{repo.name}").shuffle(shas)
        return [(f"{repo.name}@{sha[:12]}", repo.name, sha)
                for sha in shas[: args.per_repo]]

    def work(job):
        key, repo_name, sha = job
        d = commit_diff(GIT / repo_name, sha)
        if d is None:
            return None
        blob, nfiles, nch, date = d
        try:
            card = sol_card(repo_name, date, blob, nfiles, nch)
        except Exception:
            return ("err", key)
        if card is None:
            return ("err", key)
        return ("ok", dict(key=key, repo=repo_name, sha=sha, date=date,
                           n_files=nfiles, changed_lines=nch, **card))

    def append(line: str):
        for attempt in range(20):
            try:
                with open(cards_p, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                return True
            except OSError:
                time.sleep(10)
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        # repo-at-a-time job generation keeps the git scans off the quote
        for repo in repo_dirs:
            jobs = [j for j in repo_jobs(repo) if j[0] not in done]
            if not jobs:
                continue
            n_scan += 1
            for res in ex.map(work, jobs):
                if res is None:
                    continue
                kind, payload = res
                if kind == "err":
                    n_err += 1
                    continue
                if append(json.dumps(payload, ensure_ascii=False) + "\n"):
                    n_ok += 1
                    with open(done_p, "a") as fh:
                        fh.write(payload["key"] + "\n")
            if n_ok and n_ok % 100 == 0:
                el = time.time() - t0
                print(f"[{n_ok}] repos_scanned={n_scan} err={n_err} "
                      f"elapsed={el:.0f}s", flush=True)
    print(f"DONE ok={n_ok} err={n_err} repos={n_scan} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
