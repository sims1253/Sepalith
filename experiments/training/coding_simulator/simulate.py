#!/usr/bin/env python3
"""Coding simulator, stage 1 — the trajectory generator (design.md).

Simulates a developer executing ONE commit (a real work session, sourced
from the goal-card corpus / git mirror): types the new hunks chunk by
chunk (with typos + corrections), deletes-then-retypes edited regions,
navigates to other code to read context, pauses to think. At every
quiesce point (>= the extension's debounce) the simulator emits the
SUGGESTION POINT the product would have faced:

  - the byte-faithful extension prompt (eval_noop_fp.build_prompt port —
    the product v0.0.6 render)
  - ground truth = the text typed at the cursor before the next pause/
    navigation (ACCEPTABLE/COMPETING decided at metric time), or None
    when nothing is typed within the action window (NO-OP — by
    construction, since navigation/think events type nothing)

Labels are TRUE by construction — no judge needed for stage 1; the goal
card rides along for the stage-2 judged acceptance.

Timing model (seeded per trajectory; the sim-to-real knobs, see
design.md mitigations): per-chunk gaps lognormal(μ 250-500ms per-dev,
σ 0.6), EOL pause +300-900ms, typo prob 0.04/chunk (type corrupted chunk,
~400ms, backspace-fix, retype), navigation dwell 2-8s, pre-task think
5-45s (p=0.5), post-commit review pause 10-60s. Debounce 1500ms (the
extension default), action window 3000ms.

Usage:
  python3 simulate.py --n 30 --out /mnt/h/sepalith/datasets/sim_trajectories_v1/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "eval"))
from eval_noop_fp import build_prompt  # noqa: E402  product-faithful render

GIT = Path("/mnt/h/sepalith/git")
CARDS = Path("/mnt/h/sepalith/datasets/commit_goals_v1/cards.jsonl")
DEBOUNCE_MS = 1500
ACTION_WINDOW_MS = 3000
MAX_POINTS = 60
MAX_HUNKS = 8
MAX_LINES_PER_HUNK = 40


def git(repo: str, *args: str, timeout: int = 30) -> str:
    return subprocess.run(["git", "-C", str(GIT / repo), *args],
                          capture_output=True, text=True, errors="replace",
                          timeout=timeout).stdout


def hunks_for(repo: str, sha: str) -> list[dict]:
    """Per-file edit tasks from the unified diff (old-line coordinates)."""
    raw = git(repo, "show", "--format=", "--unified=0", f"{sha}^", sha,
              "--", "*.R", "*.r", timeout=60)
    tasks, cur_file = [], None
    for line in raw.splitlines():
        if line.startswith("+++ "):
            cur_file = line[4:].lstrip("b/")
        elif line.startswith("@@") and cur_file:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                continue
            tasks.append(dict(file=cur_file,
                              old_start=int(m.group(1)),
                              old_n=int(m.group(2) or 1),
                              new_start=int(m.group(3)),
                              new_n=int(m.group(4) or 1)))
    return tasks[:MAX_HUNKS]


class Trajectory:
    """Event stream + document states for one simulated commit."""

    def __init__(self, repo: str, sha: str, card: dict | None,
                 rng: random.Random):
        self.repo, self.sha, self.card = repo, sha, card
        self.rng = rng
        self.t = 0                     # simulated ms
        self.events = []               # (kind, t_ms, payload)
        self.docs: dict[str, list[str]] = {}
        self.pending_shift: dict[str, int] = {}

    def add(self, kind: str, dt: int, payload=None):
        self.t += dt
        self.events.append((kind, self.t, payload))


def word_chunks(line: str) -> list[str]:
    return re.findall(r"\S+\s*", line) or ["\n"]


def corrupt(chunk: str, rng: random.Random) -> str:
    if len(chunk) >= 3 and rng.random() < 0.5:
        i = rng.randrange(1, len(chunk) - 1)
        return chunk[:i] + chunk[i + 1] + chunk[i] + chunk[i + 2:]
    return chunk + rng.choice("xer.")


def simulate_commit(repo: str, sha: str, card: dict | None,
                    seed: str) -> dict | None:
    """One trajectory. DETERMINISTIC per seed; the seed embeds a variant
    index (sim@2:..., sim@3:...) so the same goal-card commit yields
    multiple distinct-but-reproducible developer behaviors — the user's
    'randomly generate trajectories' knob."""
    rng = random.Random(seed)
    tasks = hunks_for(repo, sha)
    if not tasks:
        return None
    traj = Trajectory(repo, sha, card, rng)
    dev_speed = rng.uniform(250, 500)   # ms per chunk, per-developer

    def gap():
        return max(60, int(rng.lognormvariate(0, 0.6) * dev_speed))

    points = []

    def maybe_point(doc_lines, cursor_line, cursor_char, rel_path,
                    ctx_kind, next_event_kind, gt_chunks):
        """Suggestion point at a quiesce: gap to the next event >= debounce.
        Emits (prompt, gt, label)."""
        if len(points) >= MAX_POINTS:
            return
        if next_event_kind in ("nav", "review"):
            label, gt = "noop", None
        elif gt_chunks:
            label, gt = "typing", "".join(gt_chunks)
        else:
            label, gt = "noop", None
        try:
            prompt, _ = build_prompt(doc_lines, cursor_line, cursor_char,
                                     rel_path)
        except Exception:
            return
        points.append(dict(prompt=prompt, gt=gt, label=label,
                           ctx=ctx_kind, t_ms=traj.t,
                           cursor=[cursor_line, cursor_char]))

    # load parent file states
    files = sorted({t["file"] for t in tasks})
    for f in files:
        raw = git(repo, "show", f"{sha}^:{f}", timeout=30)
        if raw or raw == "":
            traj.docs[f] = raw.split("\n")
        else:
            return None
    new_content = {f: git(repo, "show", f"{sha}:{f}").split("\n")
                   for f in files}

    # work plan: hunks per file in document order, files in diff order
    traj.add("open", 800)
    cur = [0, 0]        # cursor line, char (in the CURRENT file)
    cur_file = files[0]

    for ti, task in enumerate(tasks):
        f = task["file"]
        if f != cur_file:
            traj.add("switch_file", rng.randint(1500, 4000))
            cur_file, cur = f, [0, 0]
        # pre-task think (a no-op window)
        if rng.random() < 0.5:
            traj.add("think", rng.randint(5000, 45000))
            maybe_point(traj.docs[f], cur[0], cur[1], f, "pre_task_think",
                        "delete", [])
        # navigation to view context (no-op window), p=0.35
        if rng.random() < 0.35:
            lines = traj.docs[f]
            nav_line = rng.randrange(max(1, len(lines)))
            traj.add("nav", rng.randint(2000, 8000))
            cur = [nav_line, len(lines[nav_line]) if nav_line < len(lines) else 0]
            maybe_point(traj.docs[f], cur[0], cur[1], f, "navigation",
                        "nav", [])
        # hunk target in NEW coordinates
        start = task["old_start"] - 1 + traj.pending_shift.get(f, 0)
        start = max(0, min(start, len(traj.docs[f])))
        old_n = task["old_n"]
        # DELETE the old region (selection + backspace as one action)
        # this is the removed_block geometry: suggestion point where the
        # block was removed and retyping is about to start
        doc = traj.docs[f]
        removed = doc[start:start + old_n]
        traj.add("delete", rng.randint(400, 900), (f, start, old_n))
        # ground truth for the post-delete point: the upcoming first line
        first_new = new_content[f][start:start + 1]
        gt_preview = first_new[0].strip()[:80] if first_new else ""
        # cursor sits at start (beginning of the deleted region)
        cur = [start, 0]
        # the post-delete quiesce: the extension would fire here; gt is
        # what gets retyped first (built below from the actual chunks)
        # apply the deletion to the doc state
        doc = doc[:start] + doc[start + old_n:]
        traj.docs[f] = doc

        # TYPE the new region: exactly the diff's inserted lines
        ins = new_content[f][task["new_start"] - 1:
                             task["new_start"] - 1 + task["new_n"]]
        chunks_pending = []
        for li, line in enumerate(ins[:MAX_LINES_PER_HUNK]):
            for ch in word_chunks(line):
                if rng.random() < 0.04:
                    bad = corrupt(ch, rng)
                    traj.add("chunk", gap(), bad)
                    cur[1] += len(bad)
                    # mid-typing quiesce after the typo (mid-identifier
                    # cursor): the fix is coming — gt = the correction
                    g = gap()
                    if g >= DEBOUNCE_MS:
                        maybe_point(traj.docs[f], cur[0], cur[1], f,
                                    "mid_typing", "chunk", [ch])
                    traj.add("backspace", rng.randint(200, 500), len(bad))
                    cur[1] -= len(bad)
                traj.add("chunk", gap(), ch)
                cur[1] += len(ch)
                chunks_pending.append(ch)
            traj.add("newline", rng.randint(300, 900))
            cur = [cur[0] + 1, 0]
            # post-line quiesce: gt = the next line's leading chunks
            g = rng.randint(400, 2600)
            if g >= DEBOUNCE_MS and li + 1 < len(ins):
                nxt = word_chunks(ins[li + 1])[:4]
                maybe_point(traj.docs[f], cur[0], cur[1], f, "post_line",
                            "chunk", nxt)
        # splice the typed region into the doc
        traj.docs[f] = traj.docs[f][:start] + list(ins) + traj.docs[f][start:]
        traj.pending_shift[f] = traj.pending_shift.get(f, 0) + \
            (len(ins) - old_n)
        cur = [start + len(ins), 0] if ins else [start, 0]
        # post-task pause; gt = the NEXT task's first inserted line
        traj.add("pause", rng.randint(1000, 6000))
        if ti + 1 < len(tasks):
            nxt_task = tasks[ti + 1]
            nf = new_content[nxt_task["file"]]
            first = nf[nxt_task["new_start"] - 1:
                       nxt_task["new_start"] - 1 + nxt_task["new_n"]]
            gt = word_chunks(first[0])[:4] if first else []
            maybe_point(traj.docs[f], cur[0], cur[1], f, "post_task",
                        "chunk", gt)
        else:
            maybe_point(traj.docs[f], cur[0], cur[1], f, "post_task",
                        "review", [])

    # review pause at the end
    traj.add("review", rng.randint(10000, 60000))
    maybe_point(traj.docs[cur_file], cur[0], cur[1], cur_file, "review",
                "review", [])

    if not points:
        return None
    from collections import Counter
    return dict(key=f"{repo}@{sha[:12]}",
                goal=card.get("goal") if card else None,
                domain=card.get("domain") if card else None,
                n_events=len(traj.events), n_points=len(points),
                ctx_hist=dict(Counter(p["ctx"] for p in points)),
                label_hist=dict(Counter(p["label"] for p in points)),
                points=points)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--variants", type=int, default=1,
                    help="trajectories per commit (variant 1..N seeds)")
    ap.add_argument("--out", default="/mnt/h/sepalith/datasets/"
                                      "sim_trajectories_v1/trajectories.jsonl")
    args = ap.parse_args()
    cards = {}
    if CARDS.exists():
        for line in open(CARDS):
            try:
                c = json.loads(line)
                cards[c["key"]] = c
            except ValueError:
                pass
    keys = sorted(cards)
    random.Random(7).shuffle(keys)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    with open(out_p, "w") as fh:
        for key in keys:
            if n_ok >= args.n:
                break
            card = cards[key]
            repo, sha = card["key"].split("@")
            if not (GIT / repo).exists():
                continue
            try:
                traj = None
                for v in range(1, args.variants + 1):
                    traj = simulate_commit(
                        repo, sha, card, f"sim@{v}:{repo}@{sha[:12]}")
                    if traj:
                        traj["variant"] = v
                        break
            except Exception:
                continue
            if traj:
                fh.write(json.dumps(traj, ensure_ascii=False) + "\n")
                n_ok += 1
                if n_ok % 10 == 0:
                    print(f"[{n_ok}] elapsed={time.time()-t0:.0f}s",
                          flush=True)
    print(f"DONE {n_ok} trajectories -> {out_p} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
