"""Paired binary-outcome tests with deterministic pair bootstrap intervals.

Extracted from the historical evaluation audit. These are row-level intervals;
correlated trajectories require a separately specified cluster analysis.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts.

    b = pairs where A wins (a=1, b_out=0), c = pairs where B wins.
    Under H0 the discordant split is Binomial(b+c, 0.5); the exact
    two-sided p is 2 * P(X <= min(b, c)), capped at 1.
    """
    if type(b) is not int or type(c) is not int or b < 0 or c < 0:
        raise ValueError(f"discordant counts must be nonnegative integers, got b={b} c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n <= 67 or k <= 1:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (1 << n)
    else:
        term = total = 1
        for i in range(1, k + 1):
            term = term * (n - i + 1) // i
            total += term
        tail = total / (1 << n)
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
    if type(n_boot) is not int or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in deltas):
        raise ValueError("paired deltas must be finite numbers")
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
    b_wins = sum(1 for a, b in zip(outcomes_a, outcomes_b, strict=True) if a == 1 and b == 0)
    c_wins = sum(1 for a, b in zip(outcomes_a, outcomes_b, strict=True) if a == 0 and b == 1)
    p = mcnemar_exact(b_wins, c_wins)
    deltas = [float(a) - float(b) for a, b in zip(outcomes_a, outcomes_b, strict=True)]
    ci = paired_bootstrap_ci(deltas, alpha=alpha)
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
