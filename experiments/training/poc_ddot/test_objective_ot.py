#!/usr/bin/env python3
"""Self-contained tests for objective_ot.py (run: python3 -m pytest
experiments/training/poc_ddot/test_objective_ot.py -q).

Implements the failing-test list from POC-DDOT Task 3
(docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md), verbatim:
  (a) with identity coupling, loss reduces exactly to POC-DIFF's
      objective.py loss (regression against the existing estimator —
      identity coupling is injected as an explicit plan; the entropic
      Sinkhorn on equal slots smears ~15% mass off-diagonal, which is the
      coupling signal, not an exact identity);
  (b) coupling from a known shift concentrates loss weight on misaligned
      tokens;
  (c) [EMPTY] handling;
plus: position MSE zero at zero noise and correctly 1/t-weighted
otherwise; total = value CE + lambda * position MSE with lambda = 1.0;
gradients flow to both fields.

Interpretation note (documented in objective_ot.py): under uniform
marginals the plan's literal "row mass" is constant 1/N (inert as a
per-token weight), so the operative mechanism is the coupling as a soft
routing of the value CE. Row mass is still logged (telemetry).
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import objective_ot as OTO
import ot_coupling as OT

from experiments.training.poc_diff import EMPTY_ID
from experiments.training.poc_diff.objective import (
    T_MIN, bernoulli_mask, mdlm_loss)


def toy_batch(B=2, T=6, vocab=5, seed=0, span=2):
    """Mirrors poc_diff's test_objective.toy_batch so the identity-coupling
    regression runs against the same fixtures their tests use."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, 8, generator=g)
    x = torch.randint(0, vocab, (B, T), generator=g)
    span_pos = torch.zeros(B, T, dtype=torch.bool)
    span_pos[:, -span:] = True
    t = torch.full((B,), 0.5)
    span_len = span_pos.sum(dim=1)
    return h, x, span_pos, t, span_len


def identity_plan(span_pos):
    """(B, T, T) routing weights = identity on each example's span."""
    B, T = span_pos.shape
    W = torch.zeros(B, T, T)
    for b in range(B):
        idx = span_pos[b].nonzero(as_tuple=True)[0]
        W[b, idx[:, None], idx[None, :]] = torch.eye(idx.numel())
    return W


def slots_for(span_pos):
    """Uniform slot grid over each example's span region (Task 1 layout)."""
    B, T = span_pos.shape
    slots = torch.zeros(B, T)
    for b in range(B):
        idx = span_pos[b].nonzero(as_tuple=True)[0]
        n = idx.numel()
        slots[b, idx] = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
    return slots


# --- (a) identity coupling: exact reduction to poc_diff --------------------

def test_identity_coupling_reduces_to_poc_diff():
    torch.manual_seed(0)
    h, x, span_pos, t, span_len = toy_batch()
    head_w = torch.randn(5, 8)
    g = torch.Generator().manual_seed(42)
    m = bernoulli_mask(span_pos, t, generator=g)
    assert m.any()
    slots = slots_for(span_pos)
    ref = mdlm_loss(h, x, m, t, span_len, head_w, chunk=3).item()
    out = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w, plan=identity_plan(
        span_pos), chunk=3)
    assert abs(out.value.item() - ref) < 1e-5, f"{out.value.item()} vs {ref}"
    # No position head -> position term exactly zero -> total == value.
    assert out.position.item() == 0.0
    assert abs(out.total.item() - ref) < 1e-5


# --- (b) known shift concentrates loss weight on misaligned tokens ----------

def test_shift_coupling_concentrates_loss_on_misaligned():
    # Span of 5 tokens; predictions are CORRECT for their own positions
    # (logit 8 on the true token). Identity coupling -> ~zero loss; noised
    # positions shifted +2 slots -> the Sinkhorn plan routes each
    # prediction to the token at the same coordinate (2 slots earlier) ->
    # the CE lands on the misaligned pairs.
    n = 5
    h = torch.eye(n, 8).unsqueeze(0)             # feature i -> e_i
    head_w = torch.zeros(n, 8)                   # (vocab, d) — F.linear convention
    for i in range(n):
        head_w[i, i] = 8.0                       # CE(pred_i, x_j) ~ 0 iff i==j
    x = torch.arange(n).unsqueeze(0)
    span_pos = torch.ones(1, n, dtype=torch.bool)
    t = torch.ones(1)
    m = bernoulli_mask(span_pos, t, generator=torch.Generator().manual_seed(0))
    assert m.all()
    span_len = torch.tensor([n])
    slots_true = slots_for(span_pos)

    ident = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w,
                             plan=identity_plan(span_pos))
    assert ident.value.item() < 0.01, (
        f"identity coupling must ~zero the loss on self-correct preds: "
        f"{ident.value.item()}")

    S = 7
    shifted = (torch.arange(n, dtype=torch.float32) / S).unsqueeze(0)
    truth = ((torch.arange(n, dtype=torch.float32) + 2) / S).unsqueeze(0)
    out = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w,
                           slots_noised=shifted, slots_true=truth,
                           span_pos=span_pos, eps=0.01)
    assert out.value.item() > 1.0, (
        f"shifted coupling must concentrate loss on misaligned tokens: "
        f"{out.value.item()}")
    # The routed pair-weight matrix carries the concentration signal. Note:
    # balanced OT with uniform marginals FORCES spreading on this clamped
    # many-to-one shift (targets 3,4 need their column mass from rows 2-4),
    # so mass is ~75% off-diagonal, not ~100% — identity would be <15%.
    w = out.pair_weights()[0]
    off_diag = w.sum() - w.diagonal().sum()
    assert off_diag / w.sum() > 0.5, "routing mass stayed on the diagonal"


# --- (c) [EMPTY] handling -----------------------------------------------------

def test_empty_span_single_sentinel():
    # One row, span = [EMPTY] (1 token), everything masked.
    h = torch.randn(1, 3, 8)
    x = torch.tensor([[3, 4, EMPTY_ID]])
    span_pos = torch.zeros(1, 3, dtype=torch.bool)
    span_pos[0, 2] = True
    t = torch.ones(1)
    m = span_pos.clone()
    span_len = torch.tensor([1])
    head_w = torch.randn(EMPTY_ID + 1, 8)
    slots = torch.zeros(1, 3)  # sentinel slot 0.0 at the span position
    out = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w,
                           slots_noised=slots, slots_true=slots,
                           span_pos=span_pos, eps=0.05)
    assert torch.isfinite(out.total)
    # 1x1 plan -> weight 1 -> plain CE on the EMPTY token (per-token-nats).
    ce = torch.nn.functional.cross_entropy(
        (head_w @ h[0, 2]).unsqueeze(0), x[0, 2].unsqueeze(0)).item()
    assert abs(out.value.item() - ce) < 1e-5


# --- position term -------------------------------------------------------------

def test_position_mse_zero_at_zero_noise():
    torch.manual_seed(1)
    slots = torch.rand(2, 6)
    pred = slots + 0.1
    m = torch.zeros(2, 6, dtype=torch.bool)
    m[:, 2:5] = True
    t = torch.full((2,), 0.5)
    assert OTO.position_mse(pred, slots, m, t, torch.tensor([3, 3]),
                            sigma=torch.zeros(2)).item() == 0.0


def test_position_mse_weighted_by_inverse_t():
    slots = torch.zeros(1, 4)
    pred = torch.full((1, 4), 0.25)     # squared error 1/16 per position
    m = torch.zeros(1, 4, dtype=torch.bool)
    m[0, 1] = True                       # exactly one masked position
    S = torch.tensor([4])
    t = torch.tensor([0.25])
    # contribution = (1/t) * mse / S = 4 * (1/16) / 4
    got = OTO.position_mse(pred, slots, m, t, S, sigma=t).item()
    assert math.isclose(got, (1 / 0.25) * (0.0625) / 4, rel_tol=1e-6)


def test_noise_positions_perturbs_only_masked():
    torch.manual_seed(2)
    slots = torch.zeros(1, 4)
    m = torch.zeros(1, 4, dtype=torch.bool)
    m[0, 1] = True
    t = torch.full((1,), 1.0)
    noised, sigma = OTO.noise_positions(slots, m, t,
                                        generator=torch.Generator().manual_seed(3))
    assert sigma.item() == 1.0
    assert noised[0, 0] == 0.0 and noised[0, 2] == 0.0 and noised[0, 3] == 0.0
    assert noised[0, 1] != 0.0


# --- composition ---------------------------------------------------------------

def test_total_composition_lambda():
    torch.manual_seed(2)
    h, x, span_pos, t, span_len = toy_batch()
    head_w = torch.randn(5, 8)
    g = torch.Generator().manual_seed(7)
    m = bernoulli_mask(span_pos, t, generator=g)
    assert m.any()
    slots = slots_for(span_pos)
    noised, sigma = OTO.noise_positions(
        slots, m, t, generator=torch.Generator().manual_seed(9))
    pos_pred = slots + 0.02
    a = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w,
                         slots_noised=noised, slots_true=slots,
                         span_pos=span_pos, eps=0.05, lam=1.0,
                         pos_pred=pos_pred, sigma=sigma)
    b = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w,
                         slots_noised=noised, slots_true=slots,
                         span_pos=span_pos, eps=0.05, lam=0.3,
                         pos_pred=pos_pred, sigma=sigma)
    assert math.isclose(a.total.item(),
                        a.value.item() + a.position.item(), rel_tol=1e-6)
    # `position` is reported unscaled; lambda enters the total only.
    assert math.isclose(b.total.item(),
                        b.value.item() + 0.3 * b.position.item(), rel_tol=1e-6)
    assert math.isclose(a.value.item(), b.value.item(), rel_tol=1e-6)


def test_gradients_flow_value_and_position():
    torch.manual_seed(3)
    h, x, span_pos, t, span_len = toy_batch()
    h = h.requires_grad_(True)
    head_w = torch.randn(5, 8)
    g = torch.Generator().manual_seed(11)
    m = bernoulli_mask(span_pos, t, generator=g)
    assert m.any()
    slots = slots_for(span_pos)
    noised, sigma = OTO.noise_positions(
        slots, m, t, generator=torch.Generator().manual_seed(13))
    pos_pred = (slots + 0.05).requires_grad_(True)
    out = OTO.ot_mdlm_loss(h, x, m, t, span_len, head_w,
                           slots_noised=noised, slots_true=slots,
                           span_pos=span_pos, eps=0.05,
                           pos_pred=pos_pred, sigma=sigma)
    out.total.backward()
    assert h.grad is not None and h.grad[m].abs().sum() > 0
    assert pos_pred.grad is not None and pos_pred.grad[m].abs().sum() > 0
    assert out.telemetry["plan_entropy"] >= 0.0


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
            except Exception as e:  # noqa: BLE001 - standalone runner reports all
                failures += 1
                print(f"ERROR {name}: {e!r}")
    sys.exit(1 if failures else 0)
