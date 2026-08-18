#!/usr/bin/env python3
"""Construct next-edit-suggestion examples from real git commits (generalized
scale version of experiments/stage0b-niche/build_examples.py).

For each selected hunk: the parent file state around the hunk becomes
prefix/region/suffix; the hunk's result becomes the ground-truth region;
another hunk from the same commit becomes the "recent edit" event.

Behavior (filters, validators, leak-filtering, region alignment, constants) is
kept IDENTICAL to the prototype except:
  - the event hunk can no longer be the target hunk itself when the target is
    pure-deletion (prototype's `h2 is h` object-identity check never fired for
    re-parsed hunks; we compare absolute changed-line spans instead, which is
    context-invariant) -- this closes an answer-leak edge case
  - commits per repo examined is a --max-commits parameter (prototype: 30)

Output rows keep the prototype's exact field set plus provenance/audit fields
(repo_url, sha_full, author_date, default_branch, license_file, flags).

Resumable: completed repos are recorded in <spool>/_progress.jsonl and skipped
on restart; examples are written atomically per repo into <spool>/<slug>.jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

GIT_TIMEOUT = 20        # seconds per git call; NAS-hung reads on binary blobs
REPO_BUDGET = 300       # seconds per repo; data-heavy search repos can stall

CTX_BEFORE, CTX_AFTER = 4, 4          # editable-region padding around changes
PREFIX_LINES, SUFFIX_LINES = 20, 20   # excerpt beyond the region
MAX_REGION = 18
MAX_EVENT_LINES = 14
MAX_CHG = 15
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md", "COPYING")
MIN_DATE = "2026-04-15"               # temporal hygiene floor (postdates zeta-2.1 training)


def run(repo, *args, **kw):
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True,
                              timeout=GIT_TIMEOUT, **kw).stdout
    except subprocess.TimeoutExpired:
        return ""


def parse_hunks(diff_text):
    """Parse unified diff for ONE file into [{old_start,old_count,new_start,new_count,lines}]."""
    hunks, cur = [], None
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if cur: hunks.append(cur)
            hdr = line.split("@@")[1]
            old, new = hdr.split()[0], hdr.split()[1]
            os_, oc = old[1:].split(",") if "," in old else (old[1:], "1")
            ns_, nc = new[1:].split(",") if "," in new else (new[1:], "1")
            cur = dict(old_start=int(os_), old_count=int(oc),
                       new_start=int(ns_), new_count=int(nc), lines=[])
        elif cur is not None and (line.startswith(("+", "-", " ")) or line == ""):
            if line == "":
                line = " "  # empty context line lost its leading space
            cur["lines"].append(line)
    if cur: hunks.append(cur)
    return hunks


def hunk_stats(h):
    adds = sum(1 for l in h["lines"] if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in h["lines"] if l.startswith("-") and not l.startswith("---"))
    return adds, dels


def changed_span(h):
    """First/last CHANGED old-line index (context excluded) for a hunk."""
    pi, chg = h["old_start"] - 1, []
    for l in h["lines"]:
        if l.startswith(" "):
            pi += 1
        elif l.startswith("-"):
            chg.append(pi); pi += 1
        elif l.startswith("+"):
            chg.append(pi)
    return (chg[0], chg[-1]) if chg else (pi, pi)


def build_event_diff(repo, sha, path, h):
    body = "\n".join(h["lines"][:MAX_EVENT_LINES])
    return f'User edited "{path}":\n\n```diff\n@@ -{h["old_start"]},{h["old_count"]} +{h["new_start"]},{h["new_count"]} @@\n{body}\n```'


# --- audit auto-flags (same rules as experiments/stage1-data/audit.py) -------
def old_ratio(e):
    o = [l for l in e["region_old"] if l.strip()]
    n = [l for l in e["region_new"] if l.strip()]
    same = sum(1 for a, b in zip(o, n) if a == b)
    return same / max(len(o), len(n), 1)


def auto_flags(e):
    f = []
    path = e["path"].lower()
    if path.endswith((".md", ".rmd", ".qmd", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")):
        f.append("docsfile")
    if any(s in path for s in ("snapshot", "_snaps", "news", "description", " changelog", "cran-comments")):
        f.append("churnfile")
    o, n = [l for l in e["region_old"] if l.strip()], [l for l in e["region_new"] if l.strip()]
    o_s = [l.strip() for l in e["region_old"]]
    n_s = [l.strip() for l in e["region_new"]]
    if o_s == n_s:
        f.append("ws-only")
    comments_o = sum(1 for l in o if l.strip().startswith(("#", "//", "*")))
    comments_n = sum(1 for l in n if l.strip().startswith(("#", "//", "*")))
    if o and n and comments_o == len(o) and comments_n == len(n):
        f.append("comment-only")
    if len(e["region_new"]) - len(e["region_old"]) > 12:
        f.append("large-grow")
    if not e["prefix"] and not e["suffix"]:
        f.append("no-context")
    if old_ratio(e) > 0.9:
        f.append("mostly-unchanged")
    return f


def license_at(repo, sha, cache):
    if sha not in cache:
        root = run(repo, "ls-tree", "--name-only", sha).split()
        cache[sha] = next((f for f in LICENSE_NAMES if f in root), "none")
    return cache[sha]


def mine_repo(repo_dir, per_repo, since, max_commits):
    """Yield (example, counters) for one cloned repo dir. Mirrors prototype main loop."""
    slug = repo_dir.name.replace("__", "/")
    stats = {"commits": 0, "files_seen": 0, "no_hunk": 0, "too_big": 0, "ok": 0,
             "skipped_old": 0}
    lic_cache = {}
    examples = []
    t_start = time.time()
    # fast probe: if no non-merge commit since `since` touches an R file, the
    # main loop provably yields nothing (identical outcome, one cheap call)
    if not run(repo_dir, "log", "--no-merges", f"--since={since}", "-1",
               "--format=%H", "--", "*.R", "*.Rmd").strip():
        return examples, stats
    # one log call for shas + dates (replaces per-example `show -s` calls)
    log = run(repo_dir, "log", "--no-merges", f"--since={since}",
              f"-{max_commits}", "--format=%H %cs %as")
    for row in log.splitlines():
        if len(examples) >= per_repo or time.time() - t_start > REPO_BUDGET:
            break
        parts = row.split(" ")
        if len(parts) != 3:
            continue
        sha, cdate, adate = parts
        if cdate < MIN_DATE:
            stats["skipped_old"] += 1
            continue
        stats["commits"] += 1
        files = run(repo_dir, "show", "--format=", "--numstat", sha)
        picked = []
        for line in files.splitlines():
            a, d, p = (line.split("\t") + [""])[:3]
            if "{" in p: continue  # rename
            if not (p.endswith(".R") or p.endswith(".Rmd")): continue
            picked.append(p)
        for path in picked[:4]:
            if time.time() - t_start > REPO_BUDGET:
                stats["budget_hit"] = 1
                break
            stats["files_seen"] += 1
            diff = run(repo_dir, "diff", "-U6", f"{sha}^", sha, "--", path)
            if not diff: continue
            hunks = [h for h in parse_hunks(diff) if h["lines"]]
            usable = [h for h in hunks
                      if 1 <= sum(hunk_stats(h)) <= MAX_CHG and hunk_stats(h)[0] >= 1]
            if not usable:
                stats["no_hunk"] += 1; continue
            h = usable[0]
            try:
                parent = run(repo_dir, "show", f"{sha}^:{path}").splitlines()
                child = run(repo_dir, "show", f"{sha}:{path}").splitlines()
            except Exception:
                continue
            first_chg, last_chg = changed_span(h)
            rs = max(0, first_chg - CTX_BEFORE)
            re_old = min(len(parent), last_chg + 1 + CTX_AFTER)
            region_old = parent[rs:re_old]
            # map the SAME window into the child file: hunks whose changes
            # lie fully above the window shift everything below them;
            # only the target hunk's changes intersect the window.
            def shift(line):
                return sum(x["new_count"] - x["old_count"] for x in hunks
                           if changed_span(x)[1] < line)
            region_new = child[rs + shift(rs): re_old + shift(re_old)]
            if not (1 <= len(region_old) <= MAX_REGION and 1 <= len(region_new) <= MAX_REGION + 2):
                stats["too_big"] += 1; continue
            # event: a different hunk from the same commit, whose added lines
            # share no content with the target's added lines (no answer leaks)
            ev = ""
            target_added = set(l.strip() for l in region_new
                               if l.strip() and l.strip() not in
                               set(x.strip() for x in region_old))
            others = [x for x in hunks if x is not h and sum(hunk_stats(x)) >= 1]
            if others:
                allf = run(repo_dir, "show", "--format=", "--name-only", sha).splitlines()
                for alt in allf:
                    d2 = run(repo_dir, "diff", "-U3", f"{sha}^", sha, "--", alt)
                    for h2 in parse_hunks(d2):
                        if not h2["lines"] or sum(hunk_stats(h2)) < 1:
                            continue
                        if h2 is h:
                            continue
                        if alt == path and changed_span(h2) == changed_span(h):
                            continue  # the target hunk re-parsed (leak guard)
                        added2 = [l[1:].strip() for l in h2["lines"]
                                  if l.startswith("+") and l[1:].strip()]
                        if any(a in target_added for a in added2):
                            continue  # leak risk: same content as the answer
                        ev = build_event_diff(repo_dir, sha, alt, h2)
                        break
                    if ev:
                        break
            cursor = min(max(first_chg - rs, 0), len(region_old) - 1) if region_old else 0
            ex = dict(
                lang="r", repo=slug, sha=sha[:10], path=path, date=cdate,
                prefix=parent[max(0, rs - PREFIX_LINES):rs],
                region_old=region_old, cursor_idx=cursor, region_new=region_new,
                suffix=parent[re_old:re_old + SUFFIX_LINES],
                event_diff=ev, is_test="test" in path.lower() or "spec" in path.lower())
            # provenance + audit tags (additive only; core format unchanged)
            ex.update(repo_url=REPO_URL, sha_full=sha, author_date=adate,
                      default_branch=DEFAULT_BRANCH,
                      license_file=license_at(repo_dir, sha, lic_cache),
                      flags=auto_flags(ex))
            examples.append(ex)
            stats["ok"] += 1
            break
    return examples, stats


def main():
    global REPO_URL, DEFAULT_BRANCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", required=True, help="dir of cloned repos (<owner>__<repo>)")
    ap.add_argument("--spool", required=True, help="dir for per-repo jsonl outputs + progress")
    ap.add_argument("--per-repo", type=int, default=30)
    ap.add_argument("--since", default="2026-05-01")
    ap.add_argument("--max-repos", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max-commits", type=int, default=120)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    repos_dir, spool = Path(args.repos_dir), Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    progress_file = spool / "_progress.jsonl"
    done = set()
    if progress_file.exists():
        for line in progress_file.read_text().splitlines():
            try:
                done.add(json.loads(line)["repo"])
            except Exception:
                pass

    repo_dirs = sorted(p for p in repos_dir.iterdir() if p.is_dir() and "__" in p.name)
    repo_dirs = repo_dirs[args.shard::args.shards]
    if args.max_repos:
        repo_dirs = repo_dirs[:args.max_repos]
    todo = [p for p in repo_dirs if p.name not in done]
    print(f"shard {args.shard}/{args.shards}: {len(repo_dirs)} repos, {len(todo)} to do", flush=True)

    with progress_file.open("a") as prog:
        for i, rd in enumerate(todo):
            t0 = time.time()
            REPO_URL = run(rd, "remote", "get-url", "origin").strip()
            DEFAULT_BRANCH = run(rd, "symbolic-ref", "--short", "HEAD").strip()
            try:
                examples, st = mine_repo(rd, args.per_repo, args.since, args.max_commits)
            except Exception as e:
                prog.write(json.dumps({"repo": rd.name, "status": "error", "err": str(e)[:200]}) + "\n")
                prog.flush()
                print(f"[{i+1}/{len(todo)}] {rd.name} ERROR {e}", flush=True)
                continue
            tmp = spool / f".{rd.name}.tmp"
            tmp.write_text("".join(json.dumps(e) + "\n" for e in examples))
            tmp.replace(spool / f"{rd.name}.jsonl")
            prog.write(json.dumps({"repo": rd.name, "status": "ok", **st,
                                   "repo_url": REPO_URL, "default_branch": DEFAULT_BRANCH,
                                   "secs": round(time.time() - t0, 1)}) + "\n")
            prog.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"[{i+1}/{len(todo)}] {rd.name} -> {st['ok']} examples "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
