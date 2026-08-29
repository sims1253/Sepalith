"""Tests for paired_significance (run: python3 test_paired_significance.py)."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paired_significance import (
    audit_pair,
    mcnemar_exact,
    paired_bootstrap_ci,
)


def test_mcnemar_known_values():
    assert math.isclose(mcnemar_exact(0, 0), 1.0)
    assert math.isclose(mcnemar_exact(5, 5), 1.0)
    # b=0, c=10: p = 2 * (0.5)^10
    assert math.isclose(mcnemar_exact(0, 10), 2 * 0.5**10)
    # b=3, c=10, n=13: 2 * sum_{k<=3} C(13,k)/2^13
    expected = 2 * (1 + 13 + 78 + 286) / 2.0**13
    assert math.isclose(mcnemar_exact(3, 10), expected)
    # symmetric in its arguments
    assert math.isclose(mcnemar_exact(10, 3), mcnemar_exact(3, 10))


def test_mcnemar_bounds_and_errors():
    for b, c in [(0, 100), (57, 3), (1, 1)]:
        p = mcnemar_exact(b, c)
        assert 0.0 < p <= 1.0
    try:
        mcnemar_exact(-1, 5)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_bootstrap_constant_deltas():
    lo, hi = paired_bootstrap_ci([0.5] * 40, n_boot=500, seed=1)
    assert math.isclose(lo, 0.5) and math.isclose(hi, 0.5)


def test_bootstrap_ci_contains_mean_and_shrinks():
    deltas = [1.0] * 15 + [0.0] * 45  # mean 0.15
    lo, hi = paired_bootstrap_ci(deltas, n_boot=2000, seed=7)
    mean = 0.15
    assert lo <= mean <= hi
    tight = paired_bootstrap_ci(deltas, n_boot=2000, seed=7)
    assert tight == (lo, hi)  # deterministic under fixed seed
    wider = paired_bootstrap_ci(deltas[:20], n_boot=2000, seed=7)
    assert (wider[1] - wider[0]) >= (hi - lo) - 1e-9


def test_audit_pair_verdicts():
    # Decisive win for A: 18 wins vs 2 losses -> McNemar p < 0.001
    a = [1] * 18 + [0] * 2 + [1] * 30 + [0] * 30
    b = [0] * 18 + [1] * 2 + [1] * 30 + [0] * 30
    v = audit_pair("testA", a, b)
    assert v.p < 0.05 and v.verdict == "WINNER-A"

    # Same discordants but B is the better outcome under higher_better=False
    v = audit_pair("testB", a, b, higher_better=False)
    assert v.verdict == "WINNER-B"

    # Dead tie is underpowered, not significant
    a = [1, 0] * 25
    b = [0, 1] * 25
    v = audit_pair("tie", a, b)
    assert math.isclose(v.p, 1.0) and v.verdict == "TIE-UNDERPOWERED"

    # 9 vs 1 discordant out of 10: significant by exact test
    a = [1] * 9 + [0] * 1
    b = [0] * 9 + [1] * 1
    v = audit_pair("small", a, b)
    assert v.p < 0.05 and v.verdict == "WINNER-A"


def test_audit_pair_rejects_misalignment_and_nonbinary():
    for bad in ([[1, 0], [1]], [[1, 2], [0, 1]], [[], []]):
        try:
            audit_pair("bad", *bad)
            raised = False
        except ValueError:
            raised = True
        assert raised


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all green")
