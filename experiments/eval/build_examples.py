#!/usr/bin/env python3
"""Construct next-edit-suggestion examples from real git commits.

For each selected hunk: the parent file state around the hunk becomes
prefix/region/suffix; the hunk's result becomes the ground-truth region;
another hunk from the same commit becomes the "recent edit" event.
Language-neutral JSONL out; prompt rendering per model happens at eval time.
"""
import json
import subprocess
import sys
from pathlib import Path

REPOS = Path(__file__).resolve().parent.parent / ".cache" / "repos"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "examples.jsonl"
PER_REPO = int(sys.argv[2]) if len(sys.argv) > 2 else 8

CTX_BEFORE, CTX_AFTER = 4, 4          # editable-region padding around changes
PREFIX_LINES, SUFFIX_LINES = 20, 20   # excerpt beyond the region
MAX_REGION = 18
MAX_EVENT_LINES = 14
MAX_CHG = 15


def run(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, **kw).stdout


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


def main():
    examples, stats = [], {"commits": 0, "files_seen": 0, "no_hunk": 0, "too_big": 0, "ok": 0}
    for lang in ("r", "python"):
        for repo in sorted((REPOS / lang).iterdir()):
            shas = run(repo, "log", "--no-merges", "--since=2026-04-15",
                       "-30", "--format=%H").split()
            got = 0
            for sha in shas:
                if got >= PER_REPO: break
                stats["commits"] += 1
                files = run(repo, "show", "--format=", "--numstat", sha)
                picked = []
                for row in files.splitlines():
                    a, d, p = (row.split("\t") + [""])[:3]
                    if "{" in p: continue  # rename
                    ext_ok = (lang == "r" and (p.endswith(".R") or p.endswith(".Rmd"))) or \
                             (lang == "python" and p.endswith(".py"))
                    if not ext_ok: continue
                    picked.append(p)
                for path in picked[:4]:
                    stats["files_seen"] += 1
                    diff = run(repo, "diff", "-U6", f"{sha}^", sha, "--", path)
                    if not diff: continue
                    hunks = [h for h in parse_hunks(diff) if h["lines"]]
                    usable = [h for h in hunks
                              if 1 <= sum(hunk_stats(h)) <= MAX_CHG and hunk_stats(h)[0] >= 1]
                    if not usable:
                        stats["no_hunk"] += 1; continue
                    h = usable[0]
                    try:
                        parent = run(repo, "show", f"{sha}^:{path}").splitlines()
                        child = run(repo, "show", f"{sha}:{path}").splitlines()
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
                        allf = run(repo, "show", "--format=", "--name-only", sha).splitlines()
                        for alt in allf:
                            d2 = run(repo, "diff", "-U3", f"{sha}^", sha, "--", alt)
                            for h2 in parse_hunks(d2):
                                if not h2["lines"] or sum(hunk_stats(h2)) < 1:
                                    continue
                                if h2 is h:
                                    continue
                                added2 = [l[1:].strip() for l in h2["lines"]
                                          if l.startswith("+") and l[1:].strip()]
                                if any(a in target_added for a in added2):
                                    continue  # leak risk: same content as the answer
                                ev = build_event_diff(repo, sha, alt, h2)
                                break
                            if ev:
                                break
                    cursor = min(max(first_chg - rs, 0), len(region_old) - 1) if region_old else 0
                    date = run(repo, "show", "-s", "--format=%cs", sha).strip()
                    examples.append(dict(
                        lang={"r": "r", "python": "python"}[lang], repo=repo.name, sha=sha[:10],
                        path=path, date=date,
                        prefix=parent[max(0, rs - PREFIX_LINES):rs],
                        region_old=region_old, cursor_idx=cursor, region_new=region_new,
                        suffix=parent[re_old:re_old + SUFFIX_LINES],
                        event_diff=ev, is_test="test" in path.lower() or "spec" in path.lower()))
                    stats["ok"] += 1; got += 1
                    break
    OUT.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    by_lang = {}
    for e in examples: by_lang[e["lang"]] = by_lang.get(e["lang"], 0) + 1
    print(json.dumps({"stats": stats, "by_lang": by_lang, "out": str(OUT)}, indent=1))


if __name__ == "__main__":
    main()
