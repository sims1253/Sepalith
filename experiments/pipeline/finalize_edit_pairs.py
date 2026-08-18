#!/usr/bin/env python3
"""Finalize edit-pair dataset: merge per-repo spool files into examples.jsonl,
split package-level eval (5% of repos), write stats.json + repos.json.

- temporal hygiene: hard floor 2026-04-15 on every emitted example
- audit auto-flags (already tagged per row) aggregated into stats
- flagged rows are NOT dropped; downstream filters choose
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

FLAGS = ("docsfile", "churnfile", "ws-only", "comment-only", "large-grow",
         "no-context", "mostly-unchanged")
MIN_DATE = "2026-04-15"
EVAL_FRAC = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True)
    ap.add_argument("--out", required=True, help="dataset dir for examples.jsonl etc.")
    args = ap.parse_args()
    spool, out = Path(args.spool), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, per_repo = [], {}
    for f in sorted(spool.glob("*__*.jsonl")):
        n = 0
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e["date"] < MIN_DATE:
                continue  # temporal hygiene floor
            rows.append(e)
            n += 1
        per_repo[f.stem] = n

    repos = sorted(per_repo)
    eval_repos = {r for r in repos
                  if int(hashlib.md5(r.encode()).hexdigest()[:8], 16) / 0xffffffff < EVAL_FRAC}
    train, evals = [], []
    for e in rows:
        stem = e["repo"].replace("/", "__")
        (evals if stem in eval_repos else train).append(e)

    def dump(path, data):
        path.write_text("".join(json.dumps(e) + "\n" for e in data))

    dump(out / "examples.jsonl", train)
    dump(out / "eval.jsonl", evals)

    flag_counts = Counter(fl for e in rows for fl in e.get("flags", []))
    dist = Counter(e["repo"] for e in rows)
    repo_meta = {}
    prog = spool / "_progress.jsonl"
    if prog.exists():
        for line in prog.read_text().splitlines():
            try:
                r = json.loads(line)
                repo_meta[r["repo"]] = r
            except Exception:
                pass
    repos_json = {r: {"examples": per_repo.get(r, 0),
                      "url": (repo_meta.get(r) or {}).get("repo_url"),
                      "default_branch": (repo_meta.get(r) or {}).get("default_branch"),
                      "split": "eval" if r in eval_repos else "train",
                      **{k: (repo_meta.get(r) or {}).get(k) for k in
                         ("commits", "files_seen", "no_hunk", "too_big", "ok", "secs")}}
                 for r in repos}

    stats = {
        "total_examples": len(rows),
        "train_examples": len(train),
        "eval_examples": len(evals),
        "repos_with_examples": len(repos),
        "repos_ge_5_examples": sum(1 for v in per_repo.values() if v >= 5),
        "eval_repos": sorted(eval_repos),
        "examples_per_repo": {
            "min": min(per_repo.values()) if per_repo else 0,
            "max": max(per_repo.values()) if per_repo else 0,
            "median": sorted(per_repo.values())[len(per_repo) // 2] if per_repo else 0,
            "histogram_0": sum(1 for v in per_repo.values() if v == 0),
            "histogram_1_4": sum(1 for v in per_repo.values() if 1 <= v <= 4),
            "histogram_5_15": sum(1 for v in per_repo.values() if 5 <= v <= 15),
            "histogram_16_29": sum(1 for v in per_repo.values() if 16 <= v <= 29),
            "histogram_30": sum(1 for v in per_repo.values() if v >= 30),
        },
        "flag_distribution": {f: flag_counts.get(f, 0) for f in FLAGS},
        "examples_with_any_flag": sum(1 for e in rows if e.get("flags")),
        "flag_clean_frac": round(1 - sum(1 for e in rows if e.get("flags")) / max(len(rows), 1), 4),
        "date_min": min((e["date"] for e in rows), default=None),
        "date_max": max((e["date"] for e in rows), default=None),
        "with_event_diff": sum(1 for e in rows if e["event_diff"]),
        "is_test_frac": round(sum(1 for e in rows if e["is_test"]) / max(len(rows), 1), 4),
        "top_repos_by_examples": dist.most_common(15),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "repos.json").write_text(json.dumps(repos_json, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
