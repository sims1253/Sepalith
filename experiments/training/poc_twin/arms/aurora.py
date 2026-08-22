"""
Aurora optimizer (arm C, sweep OPT-3) — Muon with a leverage-aware
orthogonalization on TALL matrices.

Method source (fetched 2026-08-21, zero LLM calls):
  arXiv:2606.27715 "Aurora: A Leverage-Aware Spectral Optimizer" (Tilde),
  Algorithm 3 (vanilla Aurora), + blog.tilderesearch.com/blog/aurora.

WHAT IS IMPLEMENTED (paper-faithful core):
  Algorithm 3, verbatim:
    X_0 <- M / ||M||_F                      (M = momentum buffer)
    D_{-1} <- I_m                           (diagonal)
    for k in 0..K-1:
        r_k     <- row norms of X_k
        D_k     <- D_{k-1}^beta * diag(r_k)^(1-beta)     (damped accumulation)
        Xtilde_k <- sqrt(n/m) * D_k^{-1} X_k             (rows toward uniform)
        X_{k+1} <- polar(Xtilde_k)                       (NS iteration)
    return X_K
  - K=2, beta=0.5 (the paper's speedrun setting; their Fig. 9 shows the
    projection converges by ~2-3 iterations).
  - polar() = the vendored 5-step quintic Newton-Schulz from ../muon.py
    (speedrun coefficients 3.4445/-4.7750/2.0315, bf16, Frobenius eps 1e-7).
    Like Muon, the whole scheme is STATELESS across steps (D is intra-step
    only) — restore-exact resume, same memory as Muon (momentum buffer only).
  - Applied ONLY to the MLP up/gate projections (Wg/Wu, 3072x768 tall) —
    the paper's own scope: "The Aurora update rule is specifically for the
    up and gate projections in the MLP layers"; their Appendix A found
    row-normalizing other tall matrices gave nothing (U-NorMuon) or hurt.
    Everything else (Wq/Wk/Wv/Wo/Wd) gets the unmodified vendored Muon
    path. Wq is 1024x768 (mildly tall) — intentionally NOT Aurora'd,
    following the paper.
  - Update scale unchanged: 0.2*sqrt(max(m,n)) Moonlight/K2 RMS matching.
    Valid because X_K has the same Frobenius norm (~sqrt(n)) and hence the
    same entry RMS (1/sqrt(m)) as Muon's polar factor — the RMS-matched
    update magnitude carries over 1:1. (Uniform row norm = sqrt(n/m); the
    entry RMS is sqrt(n/m/n) = 1/sqrt(m) either way.)
  - Damping math (r_k, D_k) in fp32 (8-bit-mantissa bf16 is enough for NS
    but the repeated pow/accumulate is cheap to keep exact); X stays bf16.

WHAT IS SKIPPED (documented divergence from the full paper):
  - Riemannian-Aurora (their Algorithm 4: SVD-anchored Riemannian gradient
    descent on the joint manifold). The paper itself reports vanilla
    Aurora (this file) matched it to ~0.1-0.2% train loss at K>=2 with
    1.3-2.3x lower cost; the sweep only requires the low-overhead core.
  - Their momentum-buffer pre-scaling by D (Sec. 5 discussion) — we keep
    the plain EMA momentum identical to the Muon baseline arm so the ONLY
    studied variable is the orthogonalization.
  - per-matrix K/beta tuning — K=2, beta=0.5 fixed for all Wg/Wu.
"""
import torch
from torch import Tensor

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from muon import zeropower_via_newtonschulz5  # noqa: E402


def aurora_orthonorm(G: Tensor, ns_steps: int = 5, K: int = 2, beta: float = 0.5,
                     eps: float = 1e-7) -> Tensor:
    """Aurora orthogonalization for a TALL matrix G (m x n, m > n).
    Returns X_K: approximately column-orthonormal (Stiefel) with
    approximately-uniform row norms sqrt(n/m)."""
    m, n = G.shape
    assert m > n, "Aurora path is for tall matrices only"
    target = (n / m) ** 0.5
    X = G.bfloat16()
    X = X / (X.norm() + eps)                      # X_0 = M / ||M||_F
    d = torch.ones(m, device=G.device, dtype=torch.float32)  # D_{-1} = I
    for _ in range(K):
        r = X.float().norm(dim=1)                 # r_k: row norms of X_k
        d = d.pow(beta) * r.clamp_min(eps).pow(1.0 - beta)   # D_k
        Xt = (target / d.to(X.dtype)).unsqueeze(1) * X       # sqrt(n/m) D^-1 X
        X = zeropower_via_newtonschulz5(Xt, steps=ns_steps)  # polar
    return X


class Aurora(torch.optim.Optimizer):
    """Muon + Aurora on tall MLP up/gate projections. Params: list of
    (name, param) for ALL 2D hidden matrices; names in `aurora_names`
    (exact suffix match) take the Aurora path, the rest the Muon path.
    Everything else (momentum, wd-before-update, RMS-matched scale) is
    bit-identical to the vendored Muon in ../muon.py."""

    def __init__(self, named_params, lr=0.02, momentum=0.95, ns_steps=5,
                 weight_decay=0.1, rms_scale=0.2, aurora_beta=0.5, aurora_K=2,
                 aurora_names=("Wg.weight", "Wu.weight")):
        params = [p for _, p in named_params]
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps,
                                      weight_decay=weight_decay,
                                      rms_scale=rms_scale,
                                      aurora_beta=aurora_beta,
                                      aurora_K=aurora_K))
        self._aurora_flags = [name.endswith(tuple(aurora_names))
                              for name, _ in named_params]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            rms_scale = group["rms_scale"]
            a_beta = group["aurora_beta"]
            a_K = group["aurora_K"]
            for p, is_aurora in zip(group["params"], self._aurora_flags):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - momentum)
                if is_aurora and p.size(0) > p.size(1):
                    u = aurora_orthonorm(buf, ns_steps=ns_steps, K=a_K,
                                         beta=a_beta)
                else:
                    u = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                scale = rms_scale * max(p.size(0), p.size(1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss
