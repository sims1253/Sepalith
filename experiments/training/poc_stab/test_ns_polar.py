"""
Tests for ns_polar.py — coefficient-table integrity + orthogonalization
quality vs the classic quintic. CPU-only.
Run: uv run python test_ns_polar.py  (or pytest).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ns_polar import (_POLAR_RAW, classic_quintic, orthogonalization_error,
                      polar_coeffs, zeropower_via_polar)


def test_table_shape_and_tail():
    assert len(_POLAR_RAW) == 8
    assert _POLAR_RAW[7] == (1.875, -1.25, 0.375)
    # extension beyond 8 repeats the exact polynomial, raw
    c10 = polar_coeffs(10)
    assert c10[8] == (1.875, -1.25, 0.375) and c10[9] == (1.875, -1.25, 0.375)


def test_released_code_convention():
    # coefficients are used RAW (safety folded into normalization instead)
    c8 = polar_coeffs(8)
    assert c8 == list(_POLAR_RAW)
    c6 = polar_coeffs(6)
    assert c6 == list(_POLAR_RAW[:6])


def test_rejects_bad_steps():
    try:
        polar_coeffs(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _rand(shape, seed):
    torch.manual_seed(seed)
    return torch.randn(*shape)


def test_polar_not_worse_than_classic():
    for shape, seed in [((256, 128), 0), ((128, 256), 1), ((64, 64), 2)]:
        G = _rand(shape, seed)
        e_classic = orthogonalization_error(classic_quintic(G, steps=5))
        e_polar8 = orthogonalization_error(zeropower_via_polar(G, steps=8))
        e_polar6 = orthogonalization_error(zeropower_via_polar(G, steps=6))
        assert e_classic < 0.5, (shape, e_classic)  # sanity: both work
        assert e_polar8 <= e_classic + 0.05, (shape, e_polar8, e_classic)
        assert e_polar6 <= e_classic + 0.05, (shape, e_polar6, e_classic)


def test_transpose_convention_preserved():
    # the output preserves the input's layout; tall-vs-wide paths of the
    # same matrix must agree up to the transpose
    G = _rand((128, 256), 3)  # wide
    O_from_tall = zeropower_via_polar(G.T.contiguous(), steps=6)  # (256,128)
    O_from_wide = zeropower_via_polar(G, steps=6)                 # (128,256)
    assert O_from_tall.shape == (256, 128) and O_from_wide.shape == (128, 256)
    assert torch.allclose(O_from_tall.float(), O_from_wide.float().T, atol=2e-2)


def test_output_norm_sane():
    # orthogonalized output must not blow up: ‖O‖_F ≈ sqrt(max(m,n));
    # seed 4 diverged to inf before the per-step u=1 clamp was added.
    G = _rand((256, 128), 4)
    O = zeropower_via_polar(G, steps=8)
    n = O.float().norm()
    assert torch.isfinite(n)
    assert 10.0 < n < 20.0, n  # sqrt(256) = 16


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
