#!/usr/bin/env python3
"""Code->comment drafting family: the reverse of comment-to-code.

comment_to_code mines real (comment, code-block) pairs and trains
comment->code. The product also needs the other direction: cursor where a
comment belongs, code visible BELOW it, target = the comment the author
wrote. This script reverses every comment_to_code record into such an
example and writes scenarios_v1/comment_drafting.jsonl in the same
edit-pair schema the assembler already consumes.

No-op negatives (code that never had a comment) are NOT emitted here: the
assembler drops empty-target rows, and unconditional comment-emission is
better countered by the planned no-op scenario family (training playbook,
eagerness defenses).

Usage:
  uv run python experiments/synthetic-data/comment_drafting.py
  (add --with-synthetic once comment_to_code_synthetic.jsonl exists)
"""
import argparse
import json
from pathlib import Path

NAS = Path("/mnt/h/sepalith/datasets/scenarios_v1")
MAX_COMMENT_LINES = 8


def split_trailing_comments(prefix):
    """Split prefix into (head, comment_lines) at the trailing comment run."""
    i = len(prefix)
    while i > 0 and prefix[i - 1].strip().startswith("#") and len(prefix) - i < MAX_COMMENT_LINES:
        i -= 1
    return prefix[:i], prefix[i:]


def reverse_record(r):
    head, comments = split_trailing_comments(r["prefix"])
    if not comments or not head:
        return None
    return dict(
        prefix=head,
        region_old=[""],
        cursor_idx=0,
        region_new=comments,
        suffix=r["region_new"],           # the code being commented, below the cursor
        event_diff=r.get("event_diff") or "",
        family="comment_drafting",
        package=r["package"],
        path=r["path"],
        note=f"reversed {r['family']} ({r.get('note', '')})",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-synthetic", action="store_true",
                    help="also reverse comment_to_code_synthetic.jsonl when it exists")
    args = ap.parse_args()

    srcs = ["comment_to_code_real.jsonl"]
    if args.with_synthetic:
        srcs.append("comment_to_code_synthetic.jsonl")

    out_rows, n_in, n_skip = [], 0, 0
    for fname in srcs:
        path = NAS / fname
        if not path.exists():
            print(f"skip (missing): {fname}")
            continue
        for line in open(path):
            r = json.loads(line)
            n_in += 1
            rev = reverse_record(r)
            if rev is None:
                n_skip += 1
                continue
            out_rows.append(rev)

    out = NAS / "comment_drafting.jsonl"
    with open(out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    pkgs = {r["package"] for r in out_rows}
    print(json.dumps({"source_rows": n_in, "written": len(out_rows),
                      "skipped_no_comment_run": n_skip,
                      "packages": len(pkgs), "file": str(out)}))


if __name__ == "__main__":
    main()
