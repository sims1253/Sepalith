"""
P1 Muon-hygiene variants — Qwen3.8-Flash-Next §3.1 deltas on our pinned Muon.

Plan doc: docs/research/2026-08-29-papers-recon-poc-plan.md (P1). The pinned
recipe (poc_twin/muon.py + the Aug sweep) is the control; arms change ONE
axis each:

  control  — poc_twin.muon.Muon verbatim (NS-5 quintic, plain EMA, eps 1e-7)
  split    — per-head chunks of Wq/Wk/Wv/Wo orthogonalized independently
             (Qwen: per-head qkv splitting improved loss AND benchmarks)
  polar    — Polar Express per-step schedule, NS-8, Frobenius eps 1e-14
             (Qwen: 8 steps reduced grad-spike magnitude and frequency)
  nesterov — NS applied to Nesterov-accelerated momentum g + m·buf
             (Qwen uses Nesterov µ=0.95; our pin cites NVIDIA contra)

Everything else (RMS scale 0.2·√max, wd-before-update, side AdamW for
embed/norms) is shared with the pinned recipe. The trainer wrapper
(train_p1.py) swaps ONLY build_optim in train_ladder, so arms are paired by
construction: identical data order, schedule, seed, logging.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.join(os.path.dirname(HERE), "poc_twin")
sys.path.insert(0, POC_TWIN)

from muon import Muon, zeropower_via_newtonschulz5  # noqa: E402
from ns_polar import zeropower_via_polar            # noqa: E402

ARMS = ("control", "split", "polar", "nesterov")


class MuonHygiene(torch.optim.Optimizer):
    """Pinned Muon with single-axis deltas. `split_map` maps a Parameter to
    (dim, [chunk sizes]) — chunks are orthogonalized and RMS-scaled
    independently (scale from the CHUNK shape, per Qwen's per-head rule)."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5,
                 weight_decay=0.1, rms_scale=0.2, nesterov=False,
                 polar=False, frob_eps=1e-7, split_map=None):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps,
                                      weight_decay=weight_decay,
                                      rms_scale=rms_scale))
        self.nesterov = nesterov
        self.polar = polar
        self.frob_eps = frob_eps
        self.split_map = split_map or {}

    def _ortho(self, m):
        if self.polar:
            return zeropower_via_polar(m, steps=self._ns_steps_cache,
                                       eps=self.frob_eps, bound="power")
        return zeropower_via_newtonschulz5(m, steps=self._ns_steps_cache)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            self._ns_steps_cache = group["ns_steps"]
            wd = group["weight_decay"]
            rms_scale = group["rms_scale"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - momentum)
                d = g.add(buf, alpha=momentum) if self.nesterov else buf
                if wd > 0:
                    p.mul_(1.0 - lr * wd)   # decoupled wd before the update
                spec = self.split_map.get(p)
                if spec is None:
                    u = self._ortho(d)
                    scale = rms_scale * max(p.size(0), p.size(1)) ** 0.5
                    p.add_(u.to(p.dtype), alpha=-lr * scale)
                else:
                    dim, sizes = spec
                    offs, acc = [0], 0
                    for s in sizes:
                        acc += s
                        offs.append(acc)
                    assert acc == d.size(dim), (
                        f"split chunks ({acc}) != extent ({d.size(dim)})")
                    for lo, hi in zip(offs[:-1], offs[1:]):
                        dch = d[lo:hi] if dim == 0 else d[:, lo:hi]
                        uch = self._ortho(dch)
                        scale = rms_scale * max(dch.size(0), dch.size(1)) ** 0.5
                        tgt = p[lo:hi] if dim == 0 else p[:, lo:hi]
                        tgt.add_(uch.to(p.dtype), alpha=-lr * scale)
        return loss


def build_optim(model, lr, lr_embed, wd, arm="control"):
    """Drop-in replacement for train_ladder.build_optim (same partition:
    hidden 2D non-embed on Muon, everything else on side AdamW)."""
    hidden_named, other = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n:
            hidden_named.append((n, p))
        else:
            other.append(p)
    if arm == "control":
        muon = Muon([p for _, p in hidden_named], lr=lr, momentum=0.95,
                    ns_steps=5, weight_decay=wd)
        return muon, _side_adamw(other, lr_embed, wd)

    kwargs = dict(lr=lr, momentum=0.95, weight_decay=wd)
    if arm == "split":
        cfg = model.cfg
        hd, n_q, n_kv = cfg["head_dim"], cfg["n_q"], cfg["n_kv"]
        split_map = {}
        for n, p in hidden_named:
            if n.endswith("attn.Wq.weight"):
                split_map[p] = (0, [hd] * n_q)
            elif n.endswith(("attn.Wk.weight", "attn.Wv.weight")):
                split_map[p] = (0, [hd] * n_kv)
            elif n.endswith("attn.Wo.weight"):
                split_map[p] = (1, [hd] * n_q)
        kwargs["split_map"] = split_map
    elif arm == "polar":
        kwargs.update(polar=True, ns_steps=8, frob_eps=1e-14)
    elif arm == "nesterov":
        kwargs["nesterov"] = True
    else:
        raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")
    muon = MuonHygiene([p for _, p in hidden_named], **kwargs)
    return muon, _side_adamw(other, lr_embed, wd)


def _side_adamw(other, lr_embed, wd):
    # fused AdamW is CUDA-only; CPU (tests) falls back to the default loop
    fused = bool(other) and other[0].is_cuda
    return [torch.optim.AdamW(other, lr=lr_embed, betas=(0.9, 0.95),
                              eps=1e-8, weight_decay=wd, fused=fused)]
