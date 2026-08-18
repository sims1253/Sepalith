#!/usr/bin/env python3
"""Stage 1 audit: score constructed examples for 'would a developer plausibly
have accepted this edit at this moment'. Emits a worksheet with auto-flags;
human verdicts go into verdicts.tsv (id \t verdict \t reason).

verdict: OK | DROP with reason codes:
  fmt    degenerate construction (region/window misaligned, empty pieces)
  gen    generated/churn content (roxygen regen, snapshots, config/version)
  docs   prose/docs-only change (hard to call an 'edit suggestion')
  big    edit is a fragment of a larger refactor, region_new unmotivated alone
  ctx    unpredictable from available context (needs unseen files/discussion)
  ws     whitespace/style-only change
"""
import json, sys, re

EX = sys.argv[1] if len(sys.argv) > 1 else "stage0b-niche/examples.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "stage1-data/worksheet.txt"

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

exs = [json.loads(l) for l in open(EX)]
lines = []
for i, e in enumerate(exs):
    fl = auto_flags(e)
    lines.append(f"### [{i}] {e['lang']}/{e['repo']} {e['path']} @{e['sha']} {e['date']} flags={','.join(fl) or '-'}")
    if e["event_diff"]:
        lines.append("EVT|" + e["event_diff"].replace("\n", "\nEVT|")[:400])
    lines.append("OLD|" + "\nOLD|".join(e["region_old"][:16]))
    lines.append("NEW|" + "\nNEW|".join(e["region_new"][:18]))
    lines.append("")
open(OUT, "w").write("\n".join(lines))
flagged = sum(1 for e in exs if auto_flags(e))
print(json.dumps({"n": len(exs), "auto_flagged": flagged,
                  "flag_counts": {f: sum(1 for e in exs if f in auto_flags(e))
                                  for f in ("docsfile","churnfile","ws-only","comment-only","large-grow","no-context","mostly-unchanged")}}, indent=1))
