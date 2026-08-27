#!/usr/bin/env python3
"""Task 3: joint (value, position) MDLM objective with OT coupling — POC-DDOT.

Extends poc_diff's Task-3 estimator (objective.py) per
docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md: positions get their
own mask + Gaussian noise schedule; the OT plan couples noised and target
span slots; total = value CE + lambda * position MSE, lambda = 1.0 frozen
(sweep 0.3/1.0/3.0 only if v1 underperforms on position-MSE).

Interpretation decision (documented, load-bearing): the plan says "value
loss per token weighted by its OT-plan row mass". Under the frozen uniform
marginals the row mass is constant 1/N — inert as a per-token weight. The
operative DDOT mechanism, and the one that satisfies the plan's own tests,
is the coupling as a SOFT ROUTING of the value CE: each noised span token
i carries CE(pred_i, target_j) weighted by the row-normalized plan
P~_ij. Identity coupling reduces exactly to poc_diff's estimator (test a);
a shifted coupling routes each prediction's CE to the coupled target,
concentrating loss on misaligned tokens (test b). Row mass is still
computed and logged (telemetry row_mass_min/max): in training, a plan
stuck at identity means the coupling is inert — which per the plan IS the
finding (Task 4's entropy watch).

Position term: positions are a continuous field on the span region;
noise_positions() applies p~ = p + sigma(t) * eps at masked positions
(sigma(t) = t, linear v1). position_mse() is the denoising MSE on noised
positions only (sigma == 0 -> nothing was noised -> no position loss),
with the same (1/t)/S per-example normalization as the value term, so
both terms live on comparable per-token scales. The plan's risk note
applies: log the position-loss scale early; if it dominates the value
loss, lambda drops to 0.3 in a pre-registered single adjustment.

The coupling is applied per-example on the span window (frozen decision)
via a Sinkhorn loop over the batch — B <= 32 windows of <= 256 slots is
negligible next to the trunk forward (Task 2 measurement). An explicit
`plan` (B, T, T) bypasses Sinkhorn: tests, and Task 5's sampler side.

span_pos contract: the coupling window is the FULL span region, not just
masked positions. Callers on the Sinkhorn path must pass span_pos
(occam: training always knows it; the eval/sampler path passes the
explicit plan instead).
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from . import ot_coupling as OT          # package import (trainer path)
except ImportError:                          # script/test path (dir on sys.path)
    import ot_coupling as OT

from experiments.training.poc_diff.objective import T_MIN

PAIR_THRESH = 1e-3   # routed pairs below this weight are dropped (sparsity)
LAMBDA = 1.0         # frozen; 0.3/3.0 only if v1 underperforms position-MSE


@dataclass
class OTLossResult:
    total: torch.Tensor
    value: torch.Tensor
    position: torch.Tensor
    telemetry: dict
    pairs: tuple = ()      # flat (b, i, j, w) routed pairs — the actual routing
    pairs_map: torch.Tensor | None = None   # packed-index -> sequence-position

    def pair_weights(self):
        """(B, T, T) routing-weight matrix (zeros outside recorded pairs).
        Pairs are recorded in packed span coordinates; `pairs_map`
        translates them back to sequence positions when present."""
        if not self.pairs:
            raise ValueError("no pairs recorded (zero-mask batch)")
        b, i, j, w = self.pairs
        if self.pairs_map is not None:
            tk = self.pairs_map
            T = int(tk.max().item()) + 1
            out = torch.zeros(int(b.max().item()) + 1, T, T)
            out[b.long(), tk[b.long(), i.long()], tk[b.long(), j.long()]] = \
                w.float()
            return out
        T = int(max(i.max(), j.max()).item()) + 1
        out = torch.zeros(int(b.max().item()) + 1, T, T)
        out[b.long(), i.long(), j.long()] = w.float()
        return out


def noise_positions(slots, m, t, generator=None):
    """p~ = p + sigma(t)*eps at masked positions, sigma(t) = t (linear v1).

    Returns (noised copy, sigma) — sigma is the per-example noise scale
    actually applied (position-loss gate + telemetry)."""
    sigma = t.clone()
    eps = torch.randn(slots.shape, generator=generator, device=slots.device)
    noised = slots.clone()
    m_sigma = m & (sigma > 0)[:, None]
    noised[m_sigma] = (slots + eps * sigma[:, None])[m_sigma]
    return noised, sigma


def position_mse(pred, true, m, t, span_len, sigma):
    """Denoising MSE on noised span positions, poc_diff's estimator shape:
    per example (1/t) * sum_{i noised} (pred_i - true_i)^2 / S, batch mean.
    sigma == 0 examples contribute nothing (nothing was noised)."""
    m_eff = m & (sigma > 0)[:, None]
    if not m_eff.any():
        return pred.sum() * 0.0
    sq = (pred - true).pow(2)[m_eff]
    bidx = m_eff.nonzero(as_tuple=False)[:, 0]
    w = (1.0 / t.clamp(min=T_MIN))[bidx]
    ex_sum = torch.zeros(pred.size(0), device=pred.device, dtype=torch.float32)
    ex_sum.scatter_add_(0, bidx, (sq * w).float())
    return (ex_sum / span_len.clamp(min=1).float()).mean()


def _explicit_plan_entropy(W):
    rows = W[W.sum(-1) > 0]
    if rows.numel() == 0:
        return 0.0
    return OT.plan_entropy(rows.unsqueeze(0)).item()


def ot_mdlm_loss(h, x, m, t, span_len, head_w,
                 slots_noised=None, slots_true=None, plan=None,
                 span_pos=None, eps=0.05, kappa=None, n_iters=50,
                 lam=LAMBDA, pos_pred=None, sigma=None,
                 pair_thresh=PAIR_THRESH, chunk=4096):
    """Joint (value, position) MDLM loss with OT-routed value CE.

    Args mirror poc_diff.objective.mdlm_loss (h/x/m/t/span_len/head_w) plus:
      slots_noised/slots_true: (B, T) slot coordinates (Task 1 layout);
        required on the Sinkhorn path, unused when `plan` is given.
      plan: optional (B, T, T) explicit row-normalized routing weights,
        bypassing Sinkhorn (tests, Task 5 sampler path).
      span_pos: (B, T) bool span-region mask — the coupling window;
        required on the Sinkhorn path.
      pos_pred: (B, T) predicted clean slots (Task 4 wires the regression
        head); None -> position term is a graph-safe zero.
      sigma: per-example position-noise scale; defaults to t.
    """
    B, T = h.shape[:2]
    if plan is None and (span_pos is None or slots_noised is None
                         or slots_true is None):
        raise ValueError(
            "Sinkhorn path needs slots_noised, slots_true AND span_pos "
            "(or pass an explicit `plan`)")

    # --- pack the span window: (B, Nmax) coords instead of (B, T, T) grids --
    # The plan only lives on span positions (<=256); the full-sequence grid
    # was 162MB+/micro of dead traffic and the throughput killer.
    if span_pos is not None:
        span_rank = (span_pos.cumsum(dim=1) - 1).clamp(min=0)     # (B, T)
        bk, tk = span_pos.nonzero(as_tuple=True)                  # span coords
        kk = span_rank[bk, tk]
        lens = span_pos.sum(dim=1)
        Nmax = max(1, int(lens.max().item()))
        t_of_k = torch.zeros(B, Nmax, dtype=torch.long, device=h.device)
        t_of_k[bk, kk] = tk
        x_px = torch.zeros(B, Nmax, dtype=torch.long, device=h.device)
        x_px[bk, kk] = x[bk, tk]
        m_px = torch.zeros(B, Nmax, dtype=torch.bool, device=h.device)
        m_px[bk, kk] = m[bk, tk]
    else:
        # explicit plan, no span mask: packed == absolute coordinates
        Nmax, t_of_k, x_px, m_px = T, None, x, m
        bk = kk = None

    entropies, iters, row_mass = [], [], []
    if plan is None:
        # ONE batched Sinkhorn with zero-mass padded marginals (the whole
        # micro is a single call; per-example loops were kernel-launch-
        # bound — the plan's Risks section called this). Under no_grad —
        # exact, not an approximation: the plan's inputs (noised/target
        # slots) are data+noise constants w.r.t. model parameters.
        with torch.no_grad():
            src = torch.zeros(B, Nmax, device=h.device)
            dst = torch.zeros(B, Nmax, device=h.device)
            src[bk, kk] = slots_noised[bk, tk].float()
            dst[bk, kk] = slots_true[bk, tk].float()
            ar = torch.arange(Nmax, device=h.device)
            log_a = torch.where(ar[None, :] < lens[:, None],
                                -torch.log(lens.clamp(min=1).float())[:, None],
                                torch.full_like(lens.float()[:, None], -1e9))
            out = OT.sinkhorn_coupling(src, dst, eps=eps, kappa=kappa,
                                       n_iters=n_iters, log_a=log_a,
                                       log_b=log_a)
            rw = OT.row_weights(out.plan)          # pads -> ~0 rows
            entropies.append(_explicit_plan_entropy(rw))
            iters.append(out.iters_ran)
            row_mass.append(out.plan.sum(-1).amax(-1))
    else:
        # explicit (B, T, T) plan (tests, sampler path) -> packed; when
        # no span mask was given, packed == absolute: no gather needed
        if t_of_k is None:
            rw = plan
        else:
            rw = plan[torch.arange(B, device=h.device)[:, None, None],
                      t_of_k[:, :, None], t_of_k[:, None, :]]
        entropies = [_explicit_plan_entropy(rw)]
        iters = 0
        row_mass = [rw[b].sum(-1) for b in range(B)]

    # --- value term: routed, chunked CE (poc_diff estimator shape) --------
    # pairs: every (masked span slot i -> span slot j) with weight >= thresh
    pairs_map = None
    sel = (rw >= pair_thresh) & m_px[:, :, None]
    if not sel.any():
        value = h.sum() * 0.0
        pairs = ()
    else:
        b, i, j = sel.nonzero(as_tuple=True)
        w = rw[b, i, j].float()
        if bk is None:                    # packed == absolute
            h_sel = h[b, i]
        else:
            h_px = h.new_zeros(B, Nmax, h.size(-1))
            h_px = h_px.index_put((bk, kk), h[bk, tk])  # differentiable gather
            h_sel = h_px[b, i]
        tgt = x_px[b, j]
        w_pos = w * (1.0 / t.clamp(min=T_MIN))[b]
        n = w.size(0)

        def _ce_pair(hc, wc, tc):
            return F.cross_entropy(F.linear(hc, head_w), tc,
                                   reduction="none") * wc

        parts = []
        for c in range(0, n, chunk):
            parts.append(checkpoint(_ce_pair, h_sel[c:c + chunk],
                                    w_pos[c:c + chunk], tgt[c:c + chunk],
                                    use_reentrant=False))
        per_pair = torch.cat(parts).float()
        ex_sum = torch.zeros(B, device=h.device, dtype=torch.float32)
        ex_sum.scatter_add_(0, b, per_pair)
        contrib = ex_sum / span_len.clamp(min=1).float()
        value = contrib.mean()
        pairs = (b.detach(), i.detach(), j.detach(), w.detach())
        pairs_map = t_of_k

    # --- position term -------------------------------------------------------
    if pos_pred is not None:
        position = position_mse(pos_pred, slots_true, m, t, span_len,
                                sigma=t if sigma is None else sigma)
    else:
        position = h.sum() * 0.0

    total_loss = value + lam * position
    telemetry = dict(
        plan_entropy=(sum(entropies) / len(entropies) if entropies else 0.0),
        iters_mean=(iters[0] if iters else 0),
        row_mass_min=(float(torch.stack(row_mass).min())
                      if row_mass and len(row_mass[0].shape) == 0
                      else float("nan")),
        row_mass_max=(float(torch.stack(row_mass).max())
                      if row_mass and len(row_mass[0].shape) == 0
                      else float("nan")),
    )
    return OTLossResult(total=total_loss, value=value, position=position,
                        telemetry=telemetry, pairs=pairs,
                        pairs_map=pairs_map)
