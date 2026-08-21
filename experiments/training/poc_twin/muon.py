"""
Vendored single-file Muon optimizer (Newton-Schulz orthogonalized momentum).

Reference implementation: Keller Jordan, https://github.com/KellerJordan/Muon
(MIT license). zeropower_via_newtonschulz5 is the well-known speedrun-class
implementation (5 NS steps, coefficients 3.4445/-4.7750/2.0315).

Adaptations for the Sepalith twin POC, pinned by
docs/research/optimizer-sweep-2026-08.md (the pinned default recipe):
  - update scale uses Moonlight/K2 RMS matching: 0.2 * sqrt(max(m,n)),
    so the RMS of the parameter update is ~0.2*lr and transfers 1:1 in RMS
    terms against an AdamW arm with lr_adamw (i.e. lr_muon ~= 5 * lr_adamw
    for matched update magnitude);
  - momentum 0.95, plain EMA (no Nesterov; NVIDIA found Nesterov adds
    nothing at 1T tokens);
  - decoupled weight decay (default 0.1) applied BEFORE the update —
    the load-bearing Moonlight trick;
  - Frobenius epsilon 1e-7; NS iterations in bf16 (standard).

Usage: apply ONLY to 2D hidden matrices (not embeddings/lm_head/scalars/
norms) — those go to a side AdamW in the Muon arm.
"""
import torch
from torch import Tensor


@torch.compile(dynamic=False)
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Computes the orthogonal component G (msign) via quintic Newton-Schulz
    iteration in bf16. The output has all singular values ~ equalized."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Frobenius normalization (eps 1e-7 per the pinned recipe)
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5,
                 weight_decay=0.1, rms_scale=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps,
                                      weight_decay=weight_decay,
                                      rms_scale=rms_scale))

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
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - momentum)
                # orthogonalize the (plain-EMA) momentum buffer
                u = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                # decoupled weight decay, applied before the update
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                # RMS-matched update: RMS(O) ~ 1/sqrt(max(m,n)), so
                # 0.2*sqrt(max(m,n)) * O has RMS ~ 0.2 (Moonlight/K2 Alg. 1)
                scale = rms_scale * max(p.size(0), p.size(1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss
