#!/usr/bin/env python3
"""Self-contained tests for ot_coupling.py (run: python3 -m pytest
experiments/training/poc_ddot/test_ot_coupling.py -q).

Implements the failing-test list from POC-DDOT Task 2
(docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md), verbatim:
  (a) uniform case — coupling ≈ identity when noised == target positions;
  (b) known shift — 5 positions shifted by +2 aligns to the shifted targets
      (argmax of coupling rows);
  (c) gradients flow (autograd through Sinkhorn iters, non-zero grad on
      inputs);
  (d) length change — a 4-token noised span against a 6-token target
      produces a valid row-stochastic plan (relaxed to column-normalized in
      the unbalanced case);
  (e) uniformity/row-stoch sanity at eps in {0.01, 0.05, 0.1};
plus the numerics check (fp32 coupling, no NaN in 10k random cases) and the
frozen-decision check that the 1-D ground cost on a uniform slot grid equals
|i-j|/S exactly.

Shift-test interpretation (documented because the plan sentence is terse):
noised slots and target slots are uniform grids over the same window offset
by +2 slot widths — x_i = i/S, y_j = (j+2)/S. Position-only OT aligns each
noised token to the target at the same coordinate, so argmax(row i) must be
max(i - 2, 0): the alignment shifts by exactly the known offset.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ot_coupling as OT


def uniform_grid(n, batch=False):
    t = torch.arange(n, dtype=torch.float32) / n
    return t.unsqueeze(0) if batch else t


# --- frozen-decision ground cost -------------------------------------------

def test_position_cost_is_abs_ij_over_s_on_uniform_grid():
    # The pre-registered 1-D ground cost |i-j|/S must be exactly what the
    # module computes on Task 1's uniform slot grids (slot = index / S).
    n = m = 8
    C = OT.position_cost(uniform_grid(n), uniform_grid(m))
    assert C.shape == (n, m)
    for i in range(n):
        for j in range(m):
            assert math.isclose(C[i, j].item(), abs(i - j) / n, rel_tol=1e-6)


def test_position_cost_batched_and_positive():
    x = torch.tensor([[0.0, 0.5, 1.0], [0.1, 0.2, 0.3]])
    y = torch.tensor([[0.25, 0.75], [0.0, 1.0]])
    C = OT.position_cost(x, y)
    assert C.shape == (2, 3, 2)
    assert (C >= 0).all() and torch.isfinite(C).all()
    assert math.isclose(C[0, 1, 0].item(), 0.25, rel_tol=1e-6)


# --- (a) identity coupling when positions coincide --------------------------

def test_identity_coupling_when_positions_equal():
    n = 32
    x = uniform_grid(n)
    for eps in OT.EPS_SWEEP:
        plan = OT.sinkhorn_coupling(x, x, eps=eps).plan[0]
        argmax = plan.argmax(dim=-1)
        assert torch.equal(argmax, torch.arange(n)), f"eps={eps}"
    # At the tight end of the sweep the plan is numerically the identity.
    plan = OT.sinkhorn_coupling(x, x, eps=0.01).plan[0]
    diag_mass = plan.diagonal().sum().item()
    assert diag_mass >= 0.8, f"diag mass {diag_mass:.3f} at eps=0.01"
    # Entropy telemetry agrees: sharp plan has low entropy (uniform row
    # would be ln 32 = 3.47).
    assert OT.plan_entropy(plan.unsqueeze(0)).item() < 1.0


# --- (b) known +2 shift ------------------------------------------------------

def test_known_shift_alignment():
    S = 7  # window size: 5 noised slots, targets anchored +2 slots later
    x = torch.arange(5, dtype=torch.float32) / S
    y = (torch.arange(5, dtype=torch.float32) + 2.0) / S
    for eps in OT.EPS_SWEEP:
        plan = OT.sinkhorn_coupling(x, y, eps=eps).plan[0]
        argmax = plan.argmax(dim=-1)
        expected = torch.clamp(torch.arange(5) - 2, min=0)
        assert torch.equal(argmax, expected), f"eps={eps}: {argmax.tolist()}"
        assert (argmax.diff() >= 0).all(), "1-D OT alignment must be monotone"


# --- (c) gradients flow ------------------------------------------------------

def test_gradients_flow_through_sinkhorn():
    torch.manual_seed(0)
    n = m = 16
    x = (torch.arange(n, dtype=torch.float32) / n).requires_grad_(True)
    y = ((torch.arange(m, dtype=torch.float32) / m) + 0.13).clamp(0, 1)
    y = y.requires_grad_(True)
    out = OT.sinkhorn_coupling(x, y, eps=0.05)
    weights = torch.randn_like(out.plan)
    loss = (out.plan * weights).sum()
    loss.backward()
    for t, name in ((x, "x"), (y, "y")):
        assert t.grad is not None, name
        assert torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum().item() > 0, f"zero grad on {name}"


# --- (d) length change: 4-token noised vs 6-token target ---------------------

def test_length_change_balanced_row_stochastic():
    n, m = 4, 6
    x = uniform_grid(n)
    y = uniform_grid(m)
    out = OT.sinkhorn_coupling(x, y, eps=0.05, n_iters=200)
    plan = out.plan[0]
    assert plan.shape == (n, m)
    # Balanced: total mass 1, both marginals uniform.
    assert math.isclose(plan.sum().item(), 1.0, rel_tol=1e-3)
    row_sums = plan.sum(dim=-1)
    assert torch.allclose(row_sums, torch.full((n,), 1.0 / n), rtol=1e-2)
    col_sums = plan.sum(dim=0)
    assert torch.allclose(col_sums, torch.full((m,), 1.0 / m), rtol=1e-2)
    # Row-normalized plan is a valid distribution over target slots for
    # every noised token (this is what Task 3 uses as loss weights).
    rw = OT.row_weights(plan.unsqueeze(0))[0]
    assert torch.allclose(rw.sum(dim=-1), torch.ones(n), rtol=1e-5)
    assert (rw >= 0).all()


def test_length_change_unbalanced_column_normalized():
    n, m = 4, 6
    x = uniform_grid(n)
    y = uniform_grid(m)
    # lam = kappa/(kappa+eps) = 0.5: strong one-sided relaxation.
    out = OT.sinkhorn_coupling(x, y, eps=0.05, n_iters=200, kappa=0.05)
    plan = out.plan[0]
    # Unbalanced relaxes the ROW marginal; the hard guarantee moves to the
    # columns (plan wording: "relaxed to column-normalized").
    col_sums = plan.sum(dim=0)
    assert torch.allclose(col_sums, torch.full((m,), 1.0 / m), rtol=1e-2)
    row_sums = plan.sum(dim=-1)
    assert (row_sums > 0).all() and torch.isfinite(row_sums).all()
    # Rows are genuinely relaxed, not silently balanced.
    assert not torch.allclose(row_sums, torch.full((n,), 1.0 / n), rtol=1e-2)


# --- (e) sweep sanity --------------------------------------------------------

def test_epsilon_sweep_uniformity_and_row_stoch():
    torch.manual_seed(1)
    n = 64
    for eps in OT.EPS_SWEEP:
        x, _ = torch.sort(torch.rand(n))
        y, _ = torch.sort(torch.rand(n))
        plan = OT.sinkhorn_coupling(x, y, eps=eps).plan[0]
        assert torch.isfinite(plan).all(), f"eps={eps}"
        assert (plan >= 0).all(), f"eps={eps}"
        assert math.isclose(plan.sum().item(), 1.0, rel_tol=5e-2), f"eps={eps}"
        # At the frozen default of 50 iterations, eps=0.01 (the sharp end)
        # leaves ~10% row-marginal slack on random 64-point supports — the
        # known slow-convergence regime of entropic Sinkhorn. The slack
        # cancels in row_weights() (Task 3 normalizes rows before use);
        # the converged check below pins the solver itself.
        assert torch.allclose(
            plan.sum(dim=-1), torch.full((n,), 1.0 / n), rtol=1.2e-1
        ), f"eps={eps}"
        assert torch.allclose(
            plan.sum(dim=0), torch.full((n,), 1.0 / n), rtol=1.2e-1
        ), f"eps={eps}"
        deep = OT.sinkhorn_coupling(x, y, eps=eps, n_iters=1000).plan[0]
        assert torch.allclose(
            deep.sum(dim=-1), torch.full((n,), 1.0 / n), rtol=2e-2
        ), f"eps={eps} @1000 iters"
    # Entropy telemetry is monotone in eps (sharper coupling -> lower H).
    x, _ = torch.sort(torch.rand(n))
    h = [OT.plan_entropy(OT.sinkhorn_coupling(x, x, eps=e).plan).item()
         for e in OT.EPS_SWEEP]
    assert h[0] < h[1] < h[2], f"entropy not monotone in eps: {h}"


# --- numerics -----------------------------------------------------------------

def test_fp32_coupling_from_bf16_inputs():
    x = torch.arange(8, dtype=torch.bfloat16) / 8
    y = (torch.arange(8, dtype=torch.bfloat16) + 1) / 9
    plan = OT.sinkhorn_coupling(x, y, eps=0.05).plan
    assert plan.dtype == torch.float32
    assert torch.isfinite(plan).all()


def test_no_nan_in_10k_random_cases():
    torch.manual_seed(1273)
    cases = 0
    for trial in range(100):
        n = int(torch.randint(1, 65, (1,)))
        m = int(torch.randint(1, 65, (1,)))
        b = 100
        eps = float(torch.tensor([0.005, 0.01, 0.05, 0.1, 0.2])[
            torch.randint(0, 5, (1,))])
        kappa = None
        if trial % 3 == 0:
            kappa = float(torch.tensor([0.05, 0.5, 5.0])[
                torch.randint(0, 3, (1,))])
        kind = trial % 10
        x, _ = torch.sort(torch.rand(b, n))
        y, _ = torch.sort(torch.rand(b, m))
        if kind == 2:      # ties / duplicate coordinates
            x = (x * 5).round() / 5
            y = (y * 5).round() / 5
        elif kind == 5:    # degenerate: all mass at one coordinate
            x = torch.full((b, n), 0.5)
        elif kind == 7:    # extremes only
            x = torch.randint(0, 2, (b, n)).float()
            y = torch.randint(0, 2, (b, m)).float()
        out = OT.sinkhorn_coupling(x, y, eps=eps, kappa=kappa)
        assert torch.isfinite(out.plan).all(), f"trial={trial} eps={eps}"
        assert (out.plan >= 0).all(), f"trial={trial}"
        cases += b
    assert cases == 10_000


def test_convergence_telemetry():
    x = uniform_grid(32)
    out = OT.sinkhorn_coupling(x, x, eps=0.05, n_iters=500, tol=1e-5)
    assert out.converged
    assert out.iters_ran <= 500
    assert out.row_err.max().item() < 1e-4
    assert out.col_err.max().item() < 1e-4
    # Frozen default is 50 iterations, no early stop.
    default = OT.sinkhorn_coupling(x, x, eps=0.05)
    assert default.iters_ran == 50


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
