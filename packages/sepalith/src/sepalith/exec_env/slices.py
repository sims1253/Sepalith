"""Frozen execution hold-out and the author slice manifest.

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.exec_env.slices \\
        --pool OUT/pool.jsonl --out OUT/slices.json [--holdout 800] [--slice-size 100] [--materialize ROOT]

1. **Hold-out first** (before any RL cut exists): whole packages, in a fixed
   hash order, are reserved until at least ``--holdout`` functions are held out
   (packages with more than ``HOLDOUT_PACKAGE_MAX`` functions are skipped so
   the set spans many packages). Pool functions outside the hold-out that are
   near-duplicates of a held-out function (Jaccard >= ``LEAK_JACCARD`` on 5-line
   shingles, a looser bar than the pool's 0.8 dedup) are dropped from the
   training side. The hold-out is a frozen execution-eval set with frozen
   tests; its tasks never enter a training pool.
2. **Slices**: the remaining packages are packed greedily (largest first) into
   slices of about ``--slice-size`` functions; a package is never split, so two
   authors never share a package. Hold-out packages get their own slices
   (role ``holdout``), written and reviewed like training slices.
3. ``--materialize ROOT`` writes ``ROOT/<slice>/<function id>/source.R`` and
   ``meta.json`` for every function (the author's working layout).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

MANIFEST_SCHEMA = "sepalith.exec-env.slices.v1"
HOLDOUT_PACKAGE_MAX = 12
LEAK_JACCARD = 0.6


def _order(name: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()


def function_text(row: dict) -> str:
    lines = row["unit"].rstrip("\n").split("\n")
    marker = f"{row['name']} <- function"
    starts = [i for i, line in enumerate(lines) if line.startswith((marker, f"{row['name']} = function"))]
    return "\n".join(lines[starts[-1]:]) if starts else row["unit"]


def build(rows: list[dict], holdout_target: int, slice_size: int) -> dict:
    from sepalith.data.synth import dedup

    by_package: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_package[row["package"]].append(row)
    holdout: list[str] = []
    count = 0
    for package in sorted(by_package, key=lambda p: _order(p, "exec-env-holdout-v1")):
        if count >= holdout_target:
            break
        if len(by_package[package]) > HOLDOUT_PACKAGE_MAX:
            continue
        holdout.append(package)
        count += len(by_package[package])
    held = set(holdout)
    held_rows = [r for r in rows if r["package"] in held]
    train_rows = [r for r in rows if r["package"] not in held]
    keys = [dedup.shingles(function_text(r)) for r in held_rows] + [dedup.shingles(function_text(r)) for r in train_rows]
    matches = dedup.near_duplicates(keys, LEAK_JACCARD)
    leaked = {train_rows[i - len(held_rows)]["id"] for i, m in enumerate(matches)
              if i >= len(held_rows) and m is not None and m < len(held_rows)}
    train_by_package: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        if row["id"] not in leaked:
            train_by_package[row["package"]].append(row)

    def pack(groups: dict[str, list[dict]], role: str, prefix: str) -> list[dict]:
        bins: list[dict] = []
        for package in sorted(groups, key=lambda p: (-len(groups[p]), _order(p, "exec-env-slices-v1"))):
            n = len(groups[package])
            target = next((b for b in bins if b["functions"] + n <= slice_size), None)
            if target is None:
                target = {"packages": [], "functions": 0, "function_ids": []}
                bins.append(target)
            target["packages"].append(package)
            target["functions"] += n
            target["function_ids"] += [r["id"] for r in sorted(groups[package], key=lambda r: (r["path"], r["name"]))]
        # claim order: hash order, so the first slices taken are a representative mix, not the biggest packages
        bins.sort(key=lambda b: _order(",".join(sorted(b["packages"])), "exec-env-slice-order-v1"))
        out = []
        for index, b in enumerate(bins):
            out.append({"slice": f"{prefix}{index:03d}", "role": role, "packages": sorted(b["packages"]),
                        "functions": b["functions"], "function_ids": b["function_ids"]})
        return out

    held_groups: dict[str, list[dict]] = defaultdict(list)
    for row in held_rows:
        held_groups[row["package"]].append(row)
    slices = pack(held_groups, "holdout", "H") + pack(train_by_package, "train", "S")
    train_sizes = [s["functions"] for s in slices if s["role"] == "train"]
    return {"schema": MANIFEST_SCHEMA, "pool_functions": len(rows),
            "holdout": {"packages": len(holdout), "functions": len(held_rows),
                        "train_near_duplicates_dropped": len(leaked), "leak_jaccard": LEAK_JACCARD},
            "train": {"packages": len(train_by_package), "functions": sum(train_sizes), "slices": len(train_sizes),
                      "slice_functions_min": min(train_sizes, default=0), "slice_functions_max": max(train_sizes, default=0)},
            "slice_size_target": slice_size, "slices": slices}


def materialize(manifest: dict, rows: list[dict], root: Path, only: set[str] | None = None) -> int:
    by_id = {r["id"]: r for r in rows}
    written = 0
    for entry in manifest["slices"]:
        if only and entry["slice"] not in only:
            continue
        for fid in entry["function_ids"]:
            row = by_id[fid]
            directory = root / entry["slice"] / fid
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "source.R").write_text(row["unit"], encoding="utf-8")
            meta = {k: row[k] for k in ("id", "package", "path", "name", "helpers", "formals", "body_nonblank",
                                        "shape", "exposure", "unit_sha256")}
            meta.update(slice=entry["slice"], role=entry["role"])
            (directory / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=800)
    parser.add_argument("--slice-size", type=int, default=100)
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.pool.open()]
    if args.out.exists():
        manifest = json.loads(args.out.read_text())
    else:
        manifest = build(rows, args.holdout, args.slice_size)
        args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "slices"}, indent=1))
    if args.materialize:
        only = set(args.only.split(",")) if args.only else None
        print(json.dumps({"materialized": materialize(manifest, rows, args.materialize, only)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
