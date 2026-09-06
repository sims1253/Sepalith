"""Exact paired McNemar test + paired bootstrap CI for small eval batteries.

Motivation: our RESULTS.md before/after claims are often at n~=20-260 and
rest on raw counts. This module stamps them with an exact two-sided
McNemar (binomial on the discordant pairs) and a paired bootstrap CI on
the rate delta, so "0/24 -> 16/24"-style claims carry their uncertainty
(pattern after glm-5.3-flash-from-scratch's eval reporting).

Library core takes in-memory aligned outcome lists; the __main__ block is
the retrospective audit runner over experiments/eval/results_*.jsonl pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Preserve the historical standalone command without installing the research stack.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/sepalith/src"))
from sepalith.evaluation import (
    PairedVerdict, audit_pair, mcnemar_exact, paired_bootstrap_ci,
)


# ---------------------------------------------------------------- audit IO


def jsonl_rows(path: str, must_have: tuple[str, ...]) -> list[dict]:
    """Rows from a JSONL file that carry the id fields.

    Tolerates trailing pretty-printed summary blobs (parsed and dropped:
    they are aggregates, not per-example rows).
    """
    rows = []
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and all(k in rec for k in must_have):
            rows.append(rec)
    return rows


def pair_on_key(
    rows_a: list[dict],
    rows_b: list[dict],
    key_fn,
    outcome_fn,
    name: str,
) -> tuple[list[int], list[int]]:
    """Pair two row lists on a shared key; loud failure on dups/gaps."""
    def index(rows: list[dict]) -> dict:
        idx = {}
        for r in rows:
            k = key_fn(r)
            if k in idx:
                raise ValueError(f"{name}: duplicate key {k!r}")
            idx[k] = int(outcome_fn(r))
        return idx

    ia, ib = index(rows_a), index(rows_b)
    shared = sorted(set(ia) & set(ib))
    only_a, only_b = set(ia) - set(ib), set(ib) - set(ia)
    if only_a or only_b:
        print(
            f"  note [{name}]: {len(only_a)} keys only in A, "
            f"{len(only_b)} only in B -> paired on {len(shared)}"
        )
    if not shared:
        raise ValueError(f"{name}: zero shared keys")
    return [ia[k] for k in shared], [ib[k] for k in shared]


def _noop_pair(a_path: str, b_path: str, mode: str):
    key = lambda r: r["id"]
    if mode == "fp":  # false-positive rate at no-proposal geometries
        filt = lambda r: r.get("expectation") == "no_proposal"
        outcome = lambda r: int(r.get("proposal") == 1)
        higher_better = False
    else:  # hit rate at judgment (mid-typing) points
        filt = lambda r: r.get("expectation") == "judgment"
        outcome = lambda r: int(r.get("proposal") == 1)
        higher_better = True
    a = [r for r in jsonl_rows(a_path, ("id",)) if filt(r)]
    b = [r for r in jsonl_rows(b_path, ("id",)) if filt(r)]
    return pair_on_key(a, b, key, outcome, f"{a_path}:{mode}") + (higher_better,)


def run_audit() -> list[PairedVerdict]:
    import os

    here = os.path.dirname(os.path.abspath(__file__))

    def p(f):
        return os.path.join(here, f)

    results: list[PairedVerdict] = []
    skips: list[tuple[str, str]] = []

    def mid_key(r):
        return (r["i"], r["repo"], r["sha"])

    def attempt(label, fn):
        try:
            results.append(fn())
        except ValueError as e:
            skips.append((label, str(e)))

    def noop(label, b_name, mode):
        def run():
            oa, ob, hb = _noop_pair(
                p("results_noop_fp_sft_v7_minicpm5.jsonl"), p(b_name), mode
            )
            return audit_pair(label, oa, ob, higher_better=hb)
        return run

    attempt("noopFP v7 vs v8.2", noop("noopFP v7 vs v8.2", "results_noop_fp_sft_v8_2_minicpm5.jsonl", "fp"))
    attempt("noopHIT v7 vs v8.2", noop("noopHIT v7 vs v8.2", "results_noop_fp_sft_v8_2_minicpm5.jsonl", "hit"))
    attempt("noopFP v7 vs RL-v2c", noop("noopFP v7 vs RL-v2c", "results_noop_fp_rl_grpo_v2c.jsonl", "fp"))
    attempt("noopHIT v7 vs RL-v2c", noop("noopHIT v7 vs RL-v2c", "results_noop_fp_rl_grpo_v2c.jsonl", "hit"))

    def midtyping():
        oa, ob = pair_on_key(
            jsonl_rows(p("results_sft_v2_minicpm5_midtyping_suffix.jsonl"), ("i",)),
            jsonl_rows(p("results_sft_v5_minicpm5_midtyping_suffix.jsonl"), ("i",)),
            mid_key,
            lambda r: r["exact"],
            "midtyping v2 vs v5",
        )
        return audit_pair("midtyping v2 vs v5 (exact)", oa, ob)

    attempt("midtyping v2 vs v5 (exact)", midtyping)

    def abl():
        oa, ob = pair_on_key(
            jsonl_rows(p("results_abl_v4_on_plain.jsonl"), ("i",)),
            jsonl_rows(p("results_ablation_v6_on_plain.jsonl"), ("i",)),
            lambda r: (r["i"], r["package"], r["kind"]),
            lambda r: r["exact"],
            "abl v4 vs v6 on plain",
        )
        return audit_pair("abl v4 vs v6 on plain (exact)", oa, ob)

    attempt("abl v4 vs v6 on plain (exact)", abl)

    def intent():
        # id alone is not unique: calibration 'anchor' rows and real suite
        # rows share ids — the (id, family) pair is the suite entry key.
        oa, ob = pair_on_key(
            jsonl_rows(p("results_intent_abl_dropout-Q8_0.jsonl"), ("id",)),
            jsonl_rows(p("results_intent_sft_v8_minicpm5-Q8_0.jsonl"), ("id",)),
            lambda r: (r["id"], r["family"]),
            # anchor rows carry their own bar in `expected`; suite rows are
            # judged on the 0/1/2 scale where 2 is a full pass.
            lambda r: int(r["score"] >= r.get("expected", 2)),
            "intent dropout vs v8",
        )
        return audit_pair("intent suite: dropout vs v8 (pass)", oa, ob)

    attempt("intent suite: dropout vs v8 (pass)", intent)

    skips.append(
        (
            "scenarios battery, RL-v2 vs pre-RL base",
            "only one scenarios results file per model exists "
            "(results_scenarios_rl_grpo_v2.jsonl); no aligned pre-RL arm "
            "was ever run on the same prompts",
        )
    )
    skips.append(
        (
            "poc_diff Task-6 md vs AR (216 triples)",
            "experiments/training/poc_diff/eval_results.json is "
            "aggregate-only; per-triple rows were not persisted",
        )
    )

    print("| pair | n | rate A | rate B | discord A/B | exact p | 95% CI(dA-dB) | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for v in results:
        print(v.row())
    print()
    for name, why in skips:
        print(f"SKIP [{name}]: {why}")
    return results


if __name__ == "__main__":
    run_audit()
