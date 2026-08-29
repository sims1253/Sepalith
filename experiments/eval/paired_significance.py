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
import math
import random
from dataclasses import dataclass


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts.

    b = pairs where A wins (a=1, b_out=0), c = pairs where B wins.
    Under H0 the discordant split is Binomial(b+c, 0.5); the exact
    two-sided p is 2 * P(X <= min(b, c)), capped at 1.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be >= 0, got b={b} c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    deltas: list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1273,
) -> tuple[float, float]:
    """Percentile CI for the mean of paired deltas (resampling pairs)."""
    if not deltas:
        raise ValueError("need at least one paired delta")
    rng = random.Random(seed)
    n = len(deltas)
    stats = []
    for _ in range(n_boot):
        sample = (deltas[rng.randrange(n)] for _ in range(n))
        stats.append(math.fsum(sample) / n)
    stats.sort()
    lo = stats[max(0, int(math.floor(alpha / 2 * n_boot)) - 1)]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return lo, hi


@dataclass
class PairedVerdict:
    name: str
    n: int
    rate_a: float
    rate_b: float
    b_wins: int  # a=1, b=0
    c_wins: int  # a=0, b=1
    p: float
    ci: tuple[float, float]
    mean_delta: float
    verdict: str

    def row(self) -> str:
        return (
            f"| {self.name} | {self.n} | {self.rate_a:.3f} | {self.rate_b:.3f} "
            f"| {self.b_wins}/{self.c_wins} | {self.p:.4f} "
            f"| [{self.ci[0]:+.3f}, {self.ci[1]:+.3f}] | {self.verdict} |"
        )


def audit_pair(
    name: str,
    outcomes_a: list[int],
    outcomes_b: list[int],
    alpha: float = 0.05,
    higher_better: bool = True,
) -> PairedVerdict:
    """McNemar + bootstrap on two aligned 0/1 outcome lists.

    Verdict vocabulary: WINNER-A / WINNER-B (p < alpha and CI excludes 0
    in the matching direction), SIGNIFICANT-AMBIGUOUS (p < alpha, CI
    straddles 0), TIE-UNDERPOWERED otherwise. With higher_better=False
    (e.g. false-positive rates) the winner direction flips.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(
            f"{name}: misaligned outcome lists ({len(outcomes_a)} vs {len(outcomes_b)})"
        )
    if any(o not in (0, 1) for o in outcomes_a + outcomes_b):
        raise ValueError(f"{name}: outcomes must be 0/1")
    n = len(outcomes_a)
    if n == 0:
        raise ValueError(f"{name}: no paired rows")
    b_wins = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 1 and b == 0)
    c_wins = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 0 and b == 1)
    p = mcnemar_exact(b_wins, c_wins)
    deltas = [float(a) - float(b) for a, b in zip(outcomes_a, outcomes_b)]
    ci = paired_bootstrap_ci(deltas)
    mean_delta = math.fsum(deltas) / n
    decisive = p < alpha and (ci[0] > 0 or ci[1] < 0)
    if not decisive:
        verdict = (
            "SIGNIFICANT-AMBIGUOUS" if p < alpha else "TIE-UNDERPOWERED"
        )
    else:
        a_ahead = mean_delta > 0
        verdict = "WINNER-A" if a_ahead == higher_better else "WINNER-B"
    return PairedVerdict(
        name=name,
        n=n,
        rate_a=sum(outcomes_a) / n,
        rate_b=sum(outcomes_b) / n,
        b_wins=b_wins,
        c_wins=c_wins,
        p=p,
        ci=ci,
        mean_delta=mean_delta,
        verdict=verdict,
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
