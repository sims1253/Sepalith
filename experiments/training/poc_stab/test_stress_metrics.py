"""
Tests for stress_metrics.py — hand-computable cases, loud failures.
Run: uv run python test_stress_metrics.py  (or pytest).
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stress_metrics import (compare, grad_stats, load_jsonl, rolling_median,
                            score_run, spike_stats)


def _write_rows(tmp, rows):
    path = os.path.join(tmp, "log.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_rolling_median_fallback():
    med, w = rolling_median([3.0, 1.0, 2.0], window=101)
    assert w == 3  # window >= n → global-median fallback
    assert np.allclose(med, 2.0)


def test_single_spike_hand_computed():
    # window 3: local median at the spike is the neighbor value 1.0,
    # so 1.2 > 1.0 + 0.1 is the ONLY spike (1.2 at neighbors: median 1.0,
    # loss 1.0 not > 1.1).
    steps = list(range(10))
    losses = [1.0] * 10
    losses[5] = 1.2
    s = spike_stats(steps, losses, window=3, threshold=0.1)
    assert s["count"] == 1
    assert s["first_spike_step"] == 5 == s["last_spike_step"]
    assert s["rate"] == 1 / 10


def test_no_spike_flat():
    s = spike_stats(list(range(50)), [2.0 + 0.001 * i for i in range(50)],
                    window=5)
    assert s["count"] == 0


def test_skip_first_excludes_warmup_blowup():
    steps = list(range(20))
    losses = [50.0] + [1.0] * 19  # step-0 warmup value would dominate
    s_all = spike_stats(steps, losses, window=3)
    s_skip = spike_stats(steps, losses, window=3, skip_first=1)
    assert s_all["count"] >= 1  # the 50.0 itself spikes
    assert s_skip["count"] == 0
    assert s_skip["n_steps"] == 19


def test_grad_stats_exact_bounds():
    g = grad_stats(list(range(1, 11)), clip=1.0)  # grads 1..10
    assert g["max"] == 10.0 and g["max_frac"] == 10.0
    assert 9.9 < g["p999"] <= 10.0  # linear interpolation just under max
    assert g["n"] == 10


def test_grad_stats_rejects_bad_clip():
    try:
        grad_stats([1.0], clip=0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_load_rejects_missing_keys_and_nan():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"step": 1}) + "\n")
        try:
            load_jsonl(path)
            raise AssertionError("expected ValueError for missing loss")
        except ValueError as e:
            assert "loss" in str(e)
        path2 = os.path.join(tmp, "nan.jsonl")
        with open(path2, "w") as f:
            f.write(json.dumps({"step": 1, "loss": float("nan")}) + "\n")
        try:
            load_jsonl(path2)
            raise AssertionError("expected ValueError for NaN loss")
        except ValueError as e:
            assert "non-finite" in str(e)


def test_partial_grad_column_is_loud():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_rows(tmp, [
            {"step": 1, "loss": 1.0, "grad_norm": 0.5},
            {"step": 2, "loss": 1.0},
        ])
        try:
            score_run(path)
            raise AssertionError("expected ValueError for partial grad column")
        except ValueError as e:
            assert "all-or-nothing" in str(e)


def test_compare_verdicts():
    base = {"path": "A", "spike": {"count": 2}, "grad": {"p999_frac": 0.3}}
    better = {"path": "B", "spike": {"count": 1}, "grad": {"p999_frac": 0.2}}
    worse = {"path": "C", "spike": {"count": 3}, "grad": {"p999_frac": 0.4}}
    mixed = {"path": "D", "spike": {"count": 0}, "grad": {"p999_frac": 0.9}}
    assert compare(base, better)["b_at_least_as_stable_as_a"] is True
    v = compare(base, worse)
    assert v["b_at_least_as_stable_as_a"] is False and "veto" in v["verdict"]
    # both axes must hold: fewer spikes but worse grad p99.9 → veto
    assert compare(base, mixed)["b_at_least_as_stable_as_a"] is False
    # grad axis absent → spike axis decides alone
    nograd = {"path": "E", "spike": {"count": 1}, "grad": None}
    assert compare(base, nograd)["grad_axis"] == "skipped"
    assert compare(base, nograd)["b_at_least_as_stable_as_a"] is True


def test_score_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        rows = [{"step": 100 * (i + 1), "loss": 1.0, "grad_norm": 0.5}
                for i in range(30)]
        rows[10]["loss"] = 1.5  # one spike
        rows[10]["grad_norm"] = 2.0
        s = score_run(_write_rows(tmp, rows), clip=1.0, window=5)
        assert s["spike"]["count"] == 1
        assert s["grad"]["max_frac"] == 2.0
        assert 0.5 <= s["grad"]["p999_frac"] <= 2.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
