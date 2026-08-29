"""MuonH: Muon + hidden-radius hygiene (decay/CMA POC arm H, plan
docs/research/2026-08-28-decay-cma-muonh-poc-plan.md).

Motivation (Puro-2B, arXiv:2608.27370): Muon's update scale grows with
sqrt(max(m,n)); over long runs hidden-matrix Frobenius radii drift, which
acts as an effective-LR change. MuonH captures each 2D hidden matrix's
initial Frobenius radius R0=||W0||_F at construction and, after every
step, projects the matrix back onto that radius (W *= R0/||W||_F). Puro's
170M isolation: MuonH 3.029 vs Muon 3.073, with ELR-matched Muon
recovering to 3.030 — the win is effective-LR control, so ELR telemetry
(load-bearing for the H verdict) comes free via track_updates.

Usage: same param groups as Muon (2D hidden matrices only); pair with
wd=0 on the Muon group (--wd-muon 0) per the plan. Set project=False to
disable the radius projection (regression mode: must equal plain Muon).
"""
import torch

from muon import Muon


class MuonH(Muon):
    def __init__(self, params, *a, project=True, **kw):
        super().__init__(params, *a, **kw)
        self.project_radius = project
        self._r0 = {}  # id(param) -> initial Frobenius radius

    @torch.no_grad()
    def _capture(self):
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in self._r0:
                    self._r0[id(p)] = float(p.float().norm())

    @torch.no_grad()
    def step(self, closure=None):
        if self.project_radius:
            self._capture()
        loss = super().step(closure)
        if self.project_radius:
            for group in self.param_groups:
                for p in group["params"]:
                    r0 = self._r0.get(id(p))
                    if r0 is None or r0 == 0.0:
                        continue
                    n = float(p.float().norm())
                    if n > 0:
                        p.mul_(r0 / n)
        return loss

    def radius_error(self):
        """max_i | ||W_i||/R0_i - 1 | over tracked params (telemetry/tests)."""
        errs = []
        for group in self.param_groups:
            for p in group["params"]:
                r0 = self._r0.get(id(p))
                if r0:
                    errs.append(abs(float(p.float().norm()) / r0 - 1.0))
        return max(errs) if errs else 0.0
