#!/usr/bin/env python3
"""Strict cut of the stage-2 CodeX harvest.

Stage 2 (r_detect.py) keeps anything whose token mass looks R-ish, but
Bluespec and Verilog also assign with '<-' and define 'function'-shaped
blocks, so they leak through the shared-token path. The strict filter keeps a
row only when the R evidence is distinctive:

  keep iff the output carries an R fence tag (```r / ```R / rlang / ...),
  OR at least two strong-R-token occurrences in input+output.

Strong tokens are the R-specific markers of r_detect._R_MARKERS; '<-',
'function(', and cat()/head()/str()/summary() do not count (they are the
leak vectors).

RECONSTRUCTION CAVEAT: the strict cut actually used for the v2 mixture
(13,380 -> 9,661 rows, codex_r_strict.jsonl of 2026-08-18) was made by a
script that was never committed, and this implementation of the documented
rule does not reproduce it exactly: it keeps 11,246 rows. The NAS file
remains the authoritative artifact; this script defaults to writing
codex_r_strict.rebuilt.jsonl so a rerun cannot silently replace it. If you
need a fresh strict cut, review the delta before renaming.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r_detect import R_FENCE_RE, R_FENCE_TAGS

NAS = Path("/mnt/h/sepalith/datasets/hidden_r_instruction_v1")

# R-specific markers from r_detect._R_MARKERS, weak/shared ones dropped
_STRONG_TOKEN_RES = [
    re.compile(r"\blibrary\("),
    re.compile(r"\brequire\("),
    re.compile(r"%>%"),
    re.compile(r"%in%"),
    re.compile(r"\|\>"),
    re.compile(r"\bdata\.frame\("),
    re.compile(r"\bstopifnot\("),
    re.compile(r"\bread\.csv\(|\bread_csv\(|\bwrite\.csv\(|\bwrite_csv\("),
    re.compile(r"\bsaveRDS\(|\breadRDS\(|\bread\.rds\("),
    re.compile(r"\bggplot\(|\bggplot2::|\bdplyr::|\btidyverse|\btidyr::|\bstringr::|\bpurrr::|\bforcats::|\btibble\("),
    re.compile(r"^#'[ \t]", re.M),
    re.compile(r"\bmtcars\b|\biris\b|\blm\(|\baov\(|\bglm\("),
]


def has_r_fence(text: str) -> bool:
    return any(t.lower() in R_FENCE_TAGS for t in R_FENCE_RE.findall(text or ""))


def strong_token_count(text: str) -> int:
    return sum(len(rx.findall(text)) for rx in _STRONG_TOKEN_RES)


def strict_keep(inp: str, out: str) -> bool:
    if has_r_fence(out):
        return True
    return strong_token_count((inp or "") + "\n" + (out or "")) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, default=NAS / "codex_r.jsonl")
    ap.add_argument("--out", type=Path, default=NAS / "codex_r_strict.rebuilt.jsonl")
    args = ap.parse_args()

    kept = dropped = 0
    with open(args.out, "w") as fh:
        for line in open(args.inp):
            row = json.loads(line)
            if strict_keep(row.get("input") or "", row.get("output") or ""):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    print(json.dumps({"kept": kept, "dropped": dropped,
                      "original_2026_08_18_kept": 9661}))


if __name__ == "__main__":
    main()
