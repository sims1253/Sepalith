#!/usr/bin/env python3
"""comment_insert: comment typed MID-FUNCTION, existing code below.

The third view of the mined (comment, block) triples — the live case the
user tested in the extension: existing function code, the user adds a
comment in the middle, and the model proposes only the code that comment
introduces while the REST of the function stays visible below the cursor.

Construction from comment_to_code records: the block following the comment
splits at its midpoint — the head is the target (the "added" code), the
tail becomes the visible suffix. Suffix convention throughout.

Usage: uv run python experiments/synthetic-data/comment_insert.py
"""
import argparse, json, random
from pathlib import Path

NAS = Path("/mnt/h/sepalith/datasets/scenarios_v1")


def insert_record(r, rng):
    block = r.get("region_new") or []
    if len(block) < 4:
        return None  # need head>=1 and tail>=2 lines to look like insertion
    # split toward the first third so the target reads as "the addition"
    k = max(1, min(len(block) - 2, round(len(block) * rng.uniform(0.25, 0.5))))
    comment_lines = [l for l in (r.get("prefix") or []) if l.strip().startswith("#")]
    if not comment_lines:
        return None
    # region_old carries the freshly typed comment + cursor (midtyping
    # convention: typed partial sits in the region with the marker)
    region_old = [comment_lines[-1] + "<|user_cursor|>"]
    return dict(
        prefix=[l for l in r["prefix"] if l not in comment_lines[-1:]],
        region_old=region_old,
        cursor_idx=len(region_old) - 1,
        region_new=block[:k],
        suffix=block[k:],
        event_diff=r.get("event_diff") or "",
        family="comment_insert",
        package=r["package"],
        path=r["path"],
        note=f"mid-body comment insertion (split {k}/{len(block)}): {r.get('note', '')[:60]}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-gemini", action="store_true")
    args = ap.parse_args()
    srcs = ["comment_to_code_real.jsonl"]
    if args.with_gemini:
        srcs.append("comment_to_code_gemini.jsonl")

    rng = random.Random(31)
    rows, n_in = [], 0
    for fname in srcs:
        p = NAS / fname
        if not p.exists():
            print(f"skip (missing): {fname}")
            continue
        for line in open(p):
            r = json.loads(line)
            n_in += 1
            rec = insert_record(r, rng)
            if rec is not None:
                rows.append(rec)

    out = NAS / "comment_insert.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows": n_in, "written": len(rows), "file": str(out)}))


if __name__ == "__main__":
    main()
