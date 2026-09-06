#!/usr/bin/env python3
"""Reconstruct cached retrieval rankings without loading any model."""

import argparse
import hashlib
import io

import numpy as np
from b8b import split_group
from common import Evidence, dump, keyed


def ranking(scores):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector")
    return np.argsort(-scores, kind="stable").tolist()


def metrics(order, gold, k, gold_groups=None):
    if type(k) is not int or k <= 0:
        raise ValueError("Cutoff must be a positive integer")
    if len(set(order)) != len(order) or not gold or not set(gold) <= set(order):
        raise ValueError("Invalid ranking or gold")
    groups = gold_groups if gold_groups is not None else [{g} for g in gold]
    if (
        not groups
        or any(not group for group in groups)
        or set().union(*groups) != set(gold)
    ):
        raise ValueError("Gold groups must cover the specified gold indices")
    hits = sum(bool(set(order[:k]) & set(group)) for group in groups)
    denominator = len(groups)
    return {
        f"hit@{k}": int(hits > 0),
        f"recall@{k}": hits / denominator,
        f"all_gold@{k}": int(hits == denominator),
        "rr": 1 / (min(order.index(g) for g in gold) + 1),
    }


def rrf(a, b, constant=60):
    if not np.isfinite(constant) or constant < 0:
        raise ValueError("Fusion constant must be finite and nonnegative")
    if not a or set(a) != set(b) or len(a) != len(set(a)) or len(b) != len(set(b)):
        raise ValueError("Fusion requires identical, unique candidate pools")
    rank_a = {idx: rank for rank, idx in enumerate(a, 1)}
    rank_b = {idx: rank for rank, idx in enumerate(b, 1)}
    return sorted(
        a,
        key=lambda idx: (
            -(1 / (constant + rank_a[idx]) + 1 / (constant + rank_b[idx])),
            idx,
        ),
    )


def cache_array(e, suffix):
    r = e.record(suffix)
    if not r["exists"]:
        raise FileNotFoundError(r["path"])
    data = (e.snapshot / r["sha256"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != r["sha256"]:
        raise ValueError("Cache hash mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        a = z["emb"]
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Invalid embedding array")
    return a


def run(e, cache):
    runner = e.module("/experiments/data-mining/loc1_run_eval.py")
    saved = keyed(e.json("/loc1_s0_results/results.json")["rows"], "qid")
    source = keyed(e.rows("/loc1_s0_r/set.jsonl"), "qid")
    corpora = keyed(e.rows("/loc1_s0_r/corpus.jsonl"), "qid")
    if len(saved) != 60:
        raise ValueError("Expected the frozen 60-row comparison")
    rows = []
    private = []
    for qid, original in sorted(saved.items()):
        row = source[qid]
        chunks = corpora[qid]["chunks"]
        if (
            any(row[k] != original[k] for k in ("query", "repo", "rule"))
            or len(chunks) != original["n_corpus"]
            or len(row["gold"]) != original["n_gold"]
        ):
            raise ValueError("Query or corpus metadata drift")
        keys = [(c["path"], c["name"]) for c in chunks]
        duplicate_keys = len(keys) - len(set(keys))
        gold_keys = {(g["path"], g["name"]) for g in row["gold"]}
        if not gold_keys <= set(keys):
            raise ValueError("Gold outside pool")
        gold = [i for i, c in enumerate(keys) if c in gold_keys]
        gold_groups = [
            {i for i, c in enumerate(keys) if c == g} for g in sorted(gold_keys)
        ]
        docs = [runner.doc_text(c) for c in chunks]
        h = hashlib.sha1("\0".join(docs).encode()).hexdigest()[:16]
        lexical_scores = runner.BM25(docs).score(row["query"])
        ranks = {"bm25": ranking(lexical_scores)}
        for model in ("muninn", "muninn-small"):
            d = cache_array(cache, f"/{model}/{qid}.{h}.npz")
            q = cache_array(cache, f"/{model}/q.{qid}.{h}.npz")
            if len(d) != len(chunks) or q.shape != (1, d.shape[1]):
                raise ValueError("Cache shape mismatch")
            ranks[model] = ranking(q[0] @ d.T)
        for model, order in ranks.items():
            m = metrics(order, gold, 10) | metrics(order, gold, 5)
            for k in ("hit@10", "hit@5", "rr"):
                if abs(m[k] - original[model][k]) > 1e-7:
                    raise ValueError(f"Saved metric mismatch: {qid} {model} {k}")
        ranks["hybrid-small"] = rrf(ranks["bm25"], ranks["muninn-small"])
        ranks["hybrid-large"] = rrf(ranks["bm25"], ranks["muninn"])
        result = {
            "qid": qid,
            "repo": row["repo"],
            "sha": row["sha"],
            "rule": row["rule"],
            "split": split_group(row["repo"]),
            "n_gold": len(gold_keys),
            "n_gold_chunks": len(gold),
            "duplicate_candidate_keys": duplicate_keys,
            "n_candidates": len(chunks),
            "bm25_nonzero_candidates": int(np.count_nonzero(lexical_scores)),
            "names_target": any(
                g["name"].lower() in row["query"].lower() for g in row["gold"]
            ),
            "metrics": {
                k: metrics(v, gold, 5, gold_groups) | metrics(v, gold, 10, gold_groups)
                for k, v in ranks.items()
            },
        }
        rows.append(result)
        private.append(
            result
            | {
                "query": row["query"],
                "gold": row["gold"],
                "rankings": {k: [keys[i] for i in v] for k, v in ranks.items()},
            }
        )
    groups = {}
    subsets = {
        "all": rows,
        "development": [r for r in rows if r["split"] == "development"],
        "evaluation": [r for r in rows if r["split"] == "evaluation"],
        "multi_gold": [r for r in rows if r["n_gold"] > 1],
        "names_target": [r for r in rows if r["names_target"]],
        "no_target_name": [r for r in rows if not r["names_target"]],
    }
    for name, rs in subsets.items():
        groups[name] = {"n": len(rs), "repos": len({r["repo"] for r in rs}), "arms": {}}
        for arm in ranks:
            groups[name]["arms"][arm] = (
                {
                    k: sum(r["metrics"][arm][k] for r in rs) / len(rs)
                    for k in rows[0]["metrics"][arm]
                }
                if rs
                else {}
            )
    for r in rows:
        r["neural_only"] = (
            r["metrics"]["muninn"]["hit@10"] == 1
            and r["metrics"]["bm25"]["hit@10"] == 0
        )
        r["lexical_only"] = (
            r["metrics"]["muninn"]["hit@10"] == 0
            and r["metrics"]["bm25"]["hit@10"] == 1
        )
    return {
        "retrospective": True,
        "parity_rows": len(rows),
        "groups": groups,
        "rows": rows,
    }, private


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "inventory",
        "snapshot",
        "cache-inventory",
        "cache-snapshot",
        "out",
        "private-out",
    ):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    result, private = run(
        Evidence(a.inventory, a.snapshot), Evidence(a.cache_inventory, a.cache_snapshot)
    )
    dump(a.out, result)
    dump(a.private_out, private)


if __name__ == "__main__":
    main()
