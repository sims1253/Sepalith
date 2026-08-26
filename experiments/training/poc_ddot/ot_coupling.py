#!/usr/bin/env python3
"""Differentiable OT coupling of span slot positions — POC-DDOT Task 2.

Implements the entropic (Sinkhorn) OT alignment between the noised span's
slot coordinates and the target span's slot coordinates, per
docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md (Task 2) and the
pre-registered decisions frozen there:

  - solver: log-domain stabilized Sinkhorn, 50 iterations by default;
  - eps sweep {0.01, 0.05, 0.1} at device precision (EPS_SWEEP);
  - ground cost: 1-D position distance only, no content cost in v1 — on the
    uniform slot grids emitted by Task 1's renderer this is exactly the
    frozen |i-j|/S; on general slot coordinates it is the direct
    generalization |x_i - y_j| (both in [0, 1], so the cost is too);
  - unbalanced variant behind a flag (kappa) for unequal support sizes:
    the ROW marginal is KL-relaxed with strength kappa (damping
    lam = kappa / (kappa + eps) on the row update), the COLUMN marginal
    stays hard — so the unbalanced plan is column-normalized by
    construction (the plan's wording: "relaxed to column-normalized");
  - batched: (B, N, M) plans, N and M may differ (the plan's B x 1 x N x N
    window case is N == M);
  - numerics: the coupling is computed and returned in fp32 regardless of
    input dtype (plan: "fp32 for the coupling, bf16 elsewhere" — training
    will feed bf16 slots), and must be NaN-free (10k-case check in the
    tests).

Design notes for later tasks:
  - Task 3 consumes row_weights(plan): the per-noised-token distribution
    over target slots that reweights the value CE loss.
  - Task 4 logs plan_entropy(plan) per batch as the collapse-to-identity
    watch (entropy ~0 means the coupling is inert; that IS the finding) and
    iters_ran when tol is set.
  - The plan's Risks section moots detaching the plan every k iterations
    (memory) for Task 4; deliberately not built here — add it at the
    training integration if memory forces it, not before.
  - DDOT couples at sample level; we couple at window level because
    context is fixed in NSE — a deliberate design difference to note in
    the Task 6 write-up, not a bug.

CPU-only; ~50 iters on (B<=32, N, M<=256) is negligible next to the forward
pass (plan's own estimate).
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

# Frozen by the plan's pre-registered decisions.
EPS_SWEEP = (0.01, 0.05, 0.1)
DEFAULT_ITERS = 50


@dataclass
class CouplingResult:
    """Sinkhorn output. `plan[b]` is an (N, M) transport plan between the
    noised slots (rows) and target slots (columns) of batch element b.

    row_err / col_err: per-element max absolute deviation of the plan's
    row / column sums from the uniform marginals (for the unbalanced
    variant, row_err reports the relaxed marginal gap on purpose — it does
    not go to zero).
    """

    plan: Tensor
    iters_ran: int
    converged: bool
    row_err: Tensor
    col_err: Tensor


def _as_batch(t: Tensor, name: str) -> Tensor:
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"{name} must be 1-D (N,) or 2-D (B, N), got {tuple(t.shape)}")
    if not t.is_floating_point():
        raise TypeError(f"{name} must be a float tensor, got {t.dtype}")
    return t


def position_cost(x: Tensor, y: Tensor) -> Tensor:
    """1-D ground cost |x_i - y_j| as an fp32 tensor.

    Preserves input rank: 1-D inputs give (N, M), 2-D inputs give (B, N, M).
    On uniform slot grids (slot = index / S) this equals the frozen |i-j|/S
    exactly; the renderer may emit non-uniform slots, which this accepts.
    """
    was_1d = x.dim() == 1 and y.dim() == 1
    xf = _as_batch(x, "x").float().unsqueeze(-1)  # (B, N, 1)
    yf = _as_batch(y, "y").float().unsqueeze(-2)  # (B, 1, M)
    if xf.size(0) != yf.size(0):
        raise ValueError(f"batch mismatch: x has {xf.size(0)}, y has {yf.size(0)}")
    C = (xf - yf).abs()
    return C[0] if was_1d else C


def sinkhorn_coupling(
    x: Tensor,
    y: Tensor,
    eps: float = 0.05,
    n_iters: int = DEFAULT_ITERS,
    kappa: float | None = None,
    tol: float | None = None,
) -> CouplingResult:
    """Entropic OT between two sets of slot coordinates.

    Args:
      x: (N,) or (B, N) noised slot coordinates in [0, 1] (Task 1 layout).
      y: (M,) or (B, M) target slot coordinates in [0, 1].
      eps: entropic regularization (sweep candidates: EPS_SWEEP).
      n_iters: Sinkhorn iterations (plan default 50).
      kappa: unbalanced row-marginal strength; None = balanced OT.
        lam = kappa / (kappa + eps) damps the row update; lam -> 1 (kappa
        -> inf) recovers the balanced solver, smaller kappa relaxes rows
        harder. Columns are always hard-normalized.
      tol: if set, stop early once both marginal errors' batch max < tol.

    Returns CouplingResult; gradients flow through x and y.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if kappa is not None and kappa <= 0:
        raise ValueError(f"kappa must be > 0 when set, got {kappa}")

    xf = _as_batch(x, "x")
    yf = _as_batch(y, "y")
    B, N = xf.shape
    M = yf.size(1)
    if xf.size(0) != yf.size(0):
        raise ValueError(f"batch mismatch: x has {B}, y has {yf.size(0)}")

    # Everything coupling-related in fp32 (plan's numerics decision), even
    # when the caller feeds bf16 slots from the training loop.
    C = position_cost(xf, yf)                              # (B, N, M) fp32
    log_a = torch.full((B, N), -math.log(N), device=C.device, dtype=torch.float32)
    log_b = torch.full((B, M), -math.log(M), device=C.device, dtype=torch.float32)

    lam = 1.0 if kappa is None else kappa / (kappa + eps)
    f = torch.zeros(B, N, device=C.device, dtype=torch.float32)
    g = torch.zeros(B, M, device=C.device, dtype=torch.float32)

    a = log_a.exp()
    b = log_b.exp()
    iters_ran, converged = n_iters, False
    for _ in range(n_iters):
        # Row update (soft when unbalanced): f = lam * eps * (log a - LSE_j).
        f = lam * (eps * log_a - eps * torch.logsumexp(
            (g.unsqueeze(1) - C) / eps, dim=-1))
        # Column update (always hard): g = eps * (log b - LSE_i).
        g = eps * log_b - eps * torch.logsumexp(
            (f.unsqueeze(2) - C) / eps, dim=1)
        if tol is not None:
            plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / eps)
            row_err = (plan.sum(-1) - a).abs().amax(-1)
            col_err = (plan.sum(-2) - b).abs().amax(-1)
            if max(row_err.max(), col_err.max()) < tol:
                converged = True
                break

    plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / eps)
    if tol is None:
        row_err = (plan.sum(-1) - a).abs().amax(-1)
        col_err = (plan.sum(-2) - b).abs().amax(-1)
    return CouplingResult(
        plan=plan, iters_ran=iters_ran, converged=converged,
        row_err=row_err, col_err=col_err,
    )


def row_weights(plan: Tensor) -> Tensor:
    """Per-noised-token distribution over target slots: plan rows normalized.

    Task 3's loss reweighting: weight[i, j] = P[i, j] / sum_j P[i, j].
    Rows with vanishing mass (possible only far from convergence in the
    unbalanced variant) yield all-zero weights by the clamp — callers
    should treat that as "no alignment signal", not as a distribution.
    """
    return plan / plan.sum(-1, keepdim=True).clamp_min(torch.finfo(plan.dtype).tiny)


def plan_entropy(plan: Tensor) -> Tensor:
    """Mean Shannon entropy (nats) of the row-normalized plan.

    Task 4 telemetry: per-batch watch for collapse-to-identity. Entropy
    near 0 = each noised token maps to exactly one target slot; entropy
    near ln(M) = the plan is uninformative.
    """
    w = row_weights(plan)
    return -(w * (w + 1e-12).log()).sum(-1).mean()
