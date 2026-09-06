#!/usr/bin/env python3
"""Paired cluster bootstrap: sample repositories, retaining all rows in each."""

import argparse
import collections
import json
import math
import random
from pathlib import Path

from b8b import audit_pair
from common import dump


def cluster_ci(deltas, groups, seed=20260906, repeats=10000):
    if type(repeats) is not int or repeats < 40:
        raise ValueError("At least 40 bootstrap repeats are required")
    if (
        len(deltas) != len(groups)
        or not deltas
        or not all(math.isfinite(d) for d in deltas)
    ):
        raise ValueError("Invalid grouped deltas")
    clusters = collections.defaultdict(list)
    for d, g in zip(deltas, groups):
        clusters[g].append(d)
    values = [clusters[g] for g in sorted(clusters)]
    if len(values) < 2:
        return {"clusters": len(values), "ci": None}
    rng = random.Random(seed)
    samples = []
    stats = [(sum(v), len(v)) for v in values]
    for _ in range(repeats):
        selected = rng.choices(stats, k=len(stats))
        samples.append(sum(s for s, n in selected) / sum(n for s, n in selected))
    samples.sort()
    return {
        "clusters": len(values),
        "ci": [
            samples[int(0.025 * repeats)],
            samples[min(repeats - 1, int(0.975 * repeats))],
        ],
        "mean_delta": sum(deltas) / len(deltas),
        "seed": seed,
        "repeats": repeats,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b8b", required=True)
    p.add_argument("--loc1", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    b = json.loads(Path(a.b8b).read_text())["rows"]
    l = json.loads(Path(a.loc1).read_text())["rows"]
    out = {"b8b": {}, "loc1": {}}
    for name, rs in [
        ("all", b),
        ("format", [r for r in b if r["family"] == "format_propagation"]),
        ("evaluation", [r for r in b if r["split"] == "evaluation"]),
    ]:
        out["b8b"][name] = {
            k: cluster_ci(
                [r[k] - r["control_exact"] for r in rs], [r["package"] for r in rs]
            )
            for k in ("treatment_exact", "selected_exact")
        }
    for name, rs in [
        ("all", l),
        ("evaluation", [r for r in l if r["split"] == "evaluation"]),
    ]:
        out["loc1"][name] = {}
        for arm in ("muninn", "muninn-small", "hybrid-small", "hybrid-large"):
            aa = [r["metrics"][arm]["hit@10"] for r in rs]
            bb = [r["metrics"]["bm25"]["hit@10"] for r in rs]
            out["loc1"][name][arm] = {
                "paired_hit": audit_pair(arm + " minus bm25", aa, bb),
                "cluster_hit": cluster_ci(
                    [x - y for x, y in zip(aa, bb)], [r["repo"] for r in rs]
                ),
                "cluster_recall": cluster_ci(
                    [
                        r["metrics"][arm]["recall@10"]
                        - r["metrics"]["bm25"]["recall@10"]
                        for r in rs
                    ],
                    [r["repo"] for r in rs],
                ),
            }
    dump(a.out, out)


if __name__ == "__main__":
    main()
