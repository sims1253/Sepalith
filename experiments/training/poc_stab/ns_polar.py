"""
Polar Express per-step Newton-Schulz schedule for Muon orthogonalization.

Source: Amsel et al., "The Polar Express: Optimal Matrix Sign Methods and
Their Hardware-Aware Implementation for Muon" (arXiv:2505.16932), Appendix A
`coeffs_list` (github.com/NoahAmsel/PolarExpress) — degree-5 polynomials
p(x) = a·x + b·x³ + c·x⁵ with per-step coefficients. The released code
folds the §3.4 safety factor into the Frobenius normalization
X/(‖X‖·1.01 + eps); the raw table alone still escapes its [ℓ,u] domain
for some spectra and diverges in bf16 (verified: torch seed 4, 256×128 →
inf), so we additionally enforce the §3.4 upper bound u=1 per step with a
spectral-norm clamp — without it the per-step polynomials leave the domain
they were optimized for.

Paper recommendation for Muon: d=5, T=5–6, bf16, ℓ=1e-3. Qwen3.8-Flash-Next
§3.1 runs T=8, citing reduced grad-norm spike magnitude/frequency under
stress — P1 arm C tests T=8 for that reason (plan doc 2026-08-29).

Drop-in alternative to poc_twin.muon.zeropower_via_newtonschulz5 — same
call signature and Frobenius-normalize/transposed-layout convention, so the
P1 arm swaps one function. CPU-testable: orthogonalization error vs the
classic quintic on random matrices.
"""
import torch
from torch import Tensor

# Raw Appendix-A table (8 steps; steps >8 repeat the last row numerically).
_POLAR_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_SAFETY = 1.01


def polar_coeffs(steps: int):
    """Raw per-step coefficients for a T=`steps` schedule: the Appendix-A
    table in order, extended beyond 8 steps with the exact degree-5 NS
    polynomial (1.875, −1.25, 0.375). Safety lives in the normalization
    (released-code convention), not here."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return [_POLAR_RAW[i] if i < len(_POLAR_RAW) else _POLAR_RAW[-1]
            for i in range(steps)]


def orthogonalization_error(O: Tensor) -> float:
    """Frobenius distance of the smaller-side Gram matrix from identity,
    ‖G − I‖_F/√r with r = min(m, n) (0 = perfectly orthogonal)."""
    Of = O.float()
    if Of.size(1) <= Of.size(0):
        M, r = Of.T @ Of, Of.size(1)
    else:
        M, r = Of @ Of.T, Of.size(0)
    I = torch.eye(r, dtype=Of.dtype, device=Of.device)
    return float((M - I).norm() / r ** 0.5)


def zeropower_via_polar(G: Tensor, steps: int = 8, eps: float = 1e-7) -> Tensor:
    """Polar Express variant of zeropower_via_newtonschulz5: same interface
    (bf16 iteration, transpose convention), per-step coefficients, released-
    code normalization X/(‖X‖·1.01 + eps)."""
    assert len(G.shape) == 2
    coeffs = polar_coeffs(steps)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() * _SAFETY + eps)
    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        # §3.4 upper bound u=1, enforced per step: overshoot past the
        # polynomial's design domain otherwise diverges in bf16.
        sig = torch.linalg.matrix_norm(X.float(), ord=2)
        if sig > 1.0:
            X = (X.float() * (1.0 / (sig * _SAFETY))).bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    return X


def classic_quintic(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Reference = poc_twin.muon.zeropower_via_newtonschulz5, reimplemented
    locally so this module is self-contained for tests and P1 arms."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


if __name__ == "__main__":
    torch.manual_seed(0)
    G = torch.randn(256, 128)
    for name, fn, kw in [("classic-5", classic_quintic, {}),
                         ("polar-6", zeropower_via_polar, {"steps": 6}),
                         ("polar-8", zeropower_via_polar, {"steps": 8})]:
        O = fn(G, **kw)
        print(f"{name:<10} ortho-error {orthogonalization_error(O):.4f}")
