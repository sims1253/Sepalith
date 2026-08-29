"""
R pattern-token coverage counter (T1 shortlist tool).

Candidate forced-token patterns for the A2 tokenizer refit: R's symbol soup
that a generic/multilingual BPE splits wastefully (pipes, assignment,
access, namespace, tidy-eval, %infix%, comparisons). Counts occurrences
over raw R text (line-streamed), reports per-pattern counts and byte share
against total bytes, and emits a min-frequency shortlist of patterns that
have earned a forced vocab slot.

Coverage is an UPPER BOUND: counting is substring-based (`.data` matches
inside `.database`; `$` matches in strings/docs). It ranks; it does not
measure post-refit tokenization — that check happens in T1 proper against
the built tokenizers.

CPU-only (≤8 threads, nice'd per resource policy); no GPU claim needed.
Line plan: docs/research/2026-08-29-papers-recon-poc-plan.md (zcode-stabtok).

Usage:
  python token_patterns.py 'corpus/**/*.R' ['more/*.Rmd'] \
      [--min-count 1000] [--json OUT.json]
Globs are expanded internally — quote them.
"""
import argparse
import glob
import json
import sys

# (tag, pattern) — kept symbol-only on purpose: common words are already
# cheap for BPE; the win is multi-char symbol soup.
PATTERNS = [
    ("pipe", "%>%"), ("pipe", "%T>%"), ("pipe", "%<>%"), ("pipe", "|>"),
    ("assign", "<-"), ("assign", "<<-"), ("assign", "->"), ("assign", "->>"),
    ("access", "$"), ("access", "[["), ("access", "]]"), ("access", "@"),
    ("namespace", "::"), ("namespace", ":::"),
    ("tidyeval", "!!"), ("tidyeval", "!!!"), ("tidyeval", ":="),
    ("tidyeval", ".data"), ("tidyeval", ".env"),
    ("infix", "%in%"), ("infix", "%like%"), ("infix", "%o%"),
    ("infix", "%*%"), ("infix", "%d%"),
    ("compare", "=="), ("compare", "!="), ("compare", "<="), ("compare", ">="),
]


def iter_files(glob_args):
    seen, files = set(), []
    for g in glob_args:
        for p in sorted(glob.glob(g, recursive=True)):
            if p not in seen:
                seen.add(p)
                files.append(p)
    if not files:
        raise ValueError(f"no files matched {glob_args}")
    return files


def coverage(files):
    counts = {pat: 0 for _, pat in PATTERNS}
    total_bytes = 0
    for path in files:
        with open(path, "r", errors="replace") as f:
            for line in f:
                b = len(line.encode("utf-8", errors="replace"))
                total_bytes += b
                for _, pat in PATTERNS:
                    n = line.count(pat)
                    if n:
                        counts[pat] += n
    rows = []
    for tag, pat in PATTERNS:
        c = counts[pat]
        rows.append({
            "pattern": pat, "tag": tag, "count": c,
            "pattern_bytes": c * len(pat.encode()),
            "byte_share": (c * len(pat.encode()) / total_bytes) if total_bytes else 0.0,
        })
    rows.sort(key=lambda r: (-r["byte_share"], r["pattern"]))
    return {"files": len(files), "total_bytes": total_bytes, "rows": rows}


def shortlist(cov, min_count):
    return [r["pattern"] for r in cov["rows"] if r["count"] >= min_count]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("globs", nargs="+", help="quoted glob patterns of R text files")
    ap.add_argument("--min-count", type=int, default=1000,
                    help="occurrences needed to earn a forced-slot nomination")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="optional path for the machine-readable report")
    args = ap.parse_args(argv)

    cov = coverage(iter_files(args.globs))
    sl = shortlist(cov, args.min_count)
    print(f"files {cov['files']}  total bytes {cov['total_bytes']:,}")
    print(f"{'pattern':<8} {'tag':<10} {'count':>12} {'byte_share':>11}")
    for r in cov["rows"]:
        print(f"{r['pattern']:<8} {r['tag']:<10} {r['count']:>12,}"
              f" {100 * r['byte_share']:>10.4f}%")
    print(f"\nshortlist (count >= {args.min_count}): {sl if sl else '(none)'}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({**cov, "min_count": args.min_count,
                       "shortlist": sl}, f, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
