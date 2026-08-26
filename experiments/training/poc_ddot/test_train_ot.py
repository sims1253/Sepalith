#!/usr/bin/env python3
"""Self-contained tests for train_ot.py (run: python3 -m pytest
experiments/training/poc_ddot/test_train_ot.py -q).

DDOT Task 4 trainer: the loss-step composition, the slots-bin alignment
with poc_diff's flat token stream, checkpoint round-trip through
eval_ot.load_ot, and the plan-mandated telemetry fields (OT-plan entropy,
Sinkhorn iters, row mass, position-loss scale). CPU, tiny MDGQA — the GPU
is pocdiff's until ~17:00 per the ledger.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ot as T

from experiments.training.poc_diff.model_md import MDGQA, model_config_md


def tiny_md():
    torch.manual_seed(0)
    return MDGQA(model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                                 head_dim=16, ffn_hidden=64, max_seq=1024,
                                 vocab=130560)).eval()


def toy_micro(B=2, T=10, device="cpu", seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, 100, (B, T), generator=g)
    span_pos = torch.zeros(B, T, dtype=torch.bool)
    span_pos[:, -4:] = True
    valid = torch.ones(B, T, dtype=torch.bool)
    slots = torch.zeros(B, T)
    slots[:, -4:] = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]).unsqueeze(0)
    return x.to(device), span_pos.to(device), valid.to(device), slots.to(device)


# --- slots bin alignment (real artifacts) --------------------------------------

def test_slots_data_aligned_with_flat_ids():
    tmp = Path("/tmp/poc_ddot")
    diff = Path("/tmp/poc_diff")
    if not (tmp / "train_slots.bin").exists() or \
            not (diff / "train_row_lens.bin").exists():
        pytest.skip("train artifacts not present (tmpfs)")
    sd = T.SlotsData()
    assert sd.n_rows == 112_630
    total = int(sd.region_len.sum())
    import numpy as np
    got = np.frombuffer(open(tmp / "train_slots.bin", "rb").read(),
                        dtype=np.float16).size
    assert got == total, f"{got} slots vs {total} region tokens"
    # spot-check row 0's slots: strictly increasing, in [0,1]
    s0 = sd.slots_for(0)
    assert all(0.0 <= v <= 1.0 for v in s0)
    assert all(b > a for a, b in zip(s0, s0[1:]))


# --- loss-step composition --------------------------------------------------------

def test_ot_step_computes_and_backprops_tiny():
    import math
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1)
    x, span_pos, valid, slots = toy_micro()
    gen = torch.Generator().manual_seed(7)
    out = T.ot_step(md, pos_head, x, span_pos, valid, slots,
                    generator=gen, eps=0.05)
    for key in ("total", "value", "position"):
        assert math.isfinite(float(out[key])), key
    assert out["telemetry"]["plan_entropy"] >= 0.0
    assert out["telemetry"]["iters_mean"] > 0
    assert "row_mass_min" in out["telemetry"] and \
        "row_mass_max" in out["telemetry"]
    assert float(out["position"]) > 0, "random pos_head must give positive MSE"
    out["total"].backward()
    assert pos_head.weight.grad is not None
    assert pos_head.weight.grad.abs().sum() > 0


def test_ot_step_identity_noise_zero_position_loss():
    # t=0 draw: nothing masked, nothing noised -> both terms are graph-safe
    # zeros (their t=0 "untouched" property, extended to the position field)
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1)
    x, span_pos, valid, slots = toy_micro()
    out = T.ot_step(md, pos_head, x, span_pos, valid, slots,
                    generator=torch.Generator().manual_seed(0),
                    eps=0.05, t_override=torch.zeros(2))
    assert float(out["position"]) == 0.0
    assert float(out["total"]) == 0.0
    assert torch.isfinite(out["total"])


def test_ot_step_lam_scales_position_only():
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1)
    x, span_pos, valid, slots = toy_micro()
    # fixed t + a fresh same-seeded generator per call: identical masks and
    # noise across the two calls, so only lam differs
    a = T.ot_step(md, pos_head, x, span_pos, valid, slots, lam=1.0,
                  eps=0.05, t_override=torch.full((2,), 0.5),
                  generator=torch.Generator().manual_seed(3))
    b = T.ot_step(md, pos_head, x, span_pos, valid, slots, lam=0.3,
                  eps=0.05, t_override=torch.full((2,), 0.5),
                  generator=torch.Generator().manual_seed(3))
    assert a["mask"].any(), "seed must mask something for a live check"
    assert abs(a["total"].item() -
               (a["value"] + a["position"]).item()) < 1e-6
    assert abs(b["total"].item() -
               (b["value"] + 0.3 * b["position"]).item()) < 1e-6
    assert abs(a["value"].item() - b["value"].item()) < 1e-6


# --- checkpoint round-trip ----------------------------------------------------------

def test_ckpt_roundtrip_through_load_ot(tmp_path):
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1)
    ck = dict(cfg=md.cfg, model=md.state_dict(),
              pos_head=pos_head.state_dict(), step=123, opt=[])
    p = tmp_path / "ot_final.pt"
    torch.save(ck, p)
    from eval_ot import load_ot
    loaded = load_ot(str(p), torch.device("cpu"))
    assert isinstance(loaded, T.PosModel) or hasattr(loaded, "pos_head")
    for k, v in md.state_dict().items():
        assert torch.equal(loaded.md.state_dict()[k].float(), v.float()), k
    assert torch.equal(loaded.pos_head.weight, pos_head.weight)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                failures += 1
                import traceback
                traceback.print_exc()
                print(f"ERROR {name}")
                continue
            print(f"PASS {name}")
    sys.exit(1 if failures else 0)
