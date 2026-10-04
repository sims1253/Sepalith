"""C05: pick the authored finish_block cases for DEV250.

Takes author_dev_finish.py output and picks at most 3 functions per domain
(one DEV group each), preferring distinct backends within a domain and an
even backend mix overall, then one derived cut row per function. The
functions are newly authored, so they cannot be CPT documents; the caps
are the only limits.

    python3 select_authored_finish.py AUTHOR_DIR OUT.jsonl [--n 55]
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MAX_PER_GROUP = 3
# The fb_ctx_* variants add a synthetic outline or diagnostics comment to the
# file, which an editor would never show; DEV250 uses plain cuts only.
TRANSFORMS = {"fb_cut_signature", "fb_cut_after_first", "fb_cut_mid_nested",
              "fb_cut_before_return", "fb_cut_random", "fb_docstring_strip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=55)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    bases = [json.loads(line) for line in (args.author_dir / "bases.jsonl").open()]
    rows_path = args.author_dir / "rows.jsonl"
    rows_by_base = defaultdict(list)
    for line_no, line in enumerate(rows_path.open("rb"), 1):
        r = json.loads(line)
        if r["transform"] not in TRANSFORMS:
            continue
        rows_by_base[r["base_id"]].append((line_no, hashlib.sha256(line).hexdigest(), r))

    # Unique functions only (identical base ids can recur across jobs).
    seen, by_domain = set(), defaultdict(list)
    for b in sorted(bases, key=lambda b: b["job"]):
        if b["base_id"] in seen or not rows_by_base.get(b["base_id"]):
            continue
        seen.add(b["base_id"])
        by_domain[b["cell"]["domain"]].append(b)

    rng = random.Random(args.seed)
    backend_use = Counter()
    picked = defaultdict(list)
    # Round-robin over domains; in each round every domain takes the
    # candidate whose backend is least used overall and new to the domain.
    for _ in range(MAX_PER_GROUP):
        for domain in sorted(by_domain):
            if sum(len(v) for v in picked.values()) >= args.n:
                break
            used_here = {b["backend"] for b in picked[domain]}
            pool = [b for b in by_domain[domain] if b not in picked[domain]]
            if not pool:
                continue
            pool.sort(key=lambda b: (b["backend"] in used_here,
                                     backend_use[b["backend"]], b["job"]))
            choice = pool[0]
            picked[domain].append(choice)
            backend_use[choice["backend"]] += 1

    out = []
    for domain in sorted(picked):
        for b in picked[domain]:
            line_no, line_sha, row = rng.choice(sorted(
                rows_by_base[b["base_id"]], key=lambda x: x[0]))
            out.append(dict(
                domain=domain, group=row["package"], job=b["job"],
                backend=b["backend"], model=b["model"], base_id=b["base_id"],
                rows_file=str(rows_path), rows_line=line_no,
                raw_line_sha256=line_sha, transform=row["transform"],
                cut=row["cut"]))
    args.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    summary = dict(
        selected=len(out), groups=len(picked),
        per_group=dict(Counter(x["group"] for x in out)),
        per_backend=dict(Counter(x["backend"] for x in out)),
        per_transform=dict(Counter(x["transform"] for x in out)),
        rows_file_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
