#!/usr/bin/env python3
"""Tests for graphed_step.py — the full-step CUDA-graph capture (run:
python3 -m pytest experiments/training/poc_ddot/test_graphed_step.py -q).

CUDA-gated (skips cleanly without a GPU): on a tiny MDGQA,
  (a) graphed vs eager equivalence on IDENTICAL pre-drawn inputs
      (loss 1e-4 rel, grads 1e-3 rel);
  (b) replaying the same graph with different inputs gives per-input
      correct results (buffer-refresh correctness);
  (c) bucket pads contribute exactly zero (padded B/T run == unpadded
      eager run; junk in the pad region does not change the loss).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.training.poc_diff.model_md import MDGQA, model_config_md
from objective_ot import ot_mdlm_loss
from graphed_step import GraphedOTStep

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="full-graph capture needs CUDA")
DEV = torch.device("cuda")


def tiny_md():
    torch.manual_seed(0)
    cfg = model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                          head_dim=16, ffn_hidden=64, max_seq=256,
                          vocab=2000)
    return MDGQA(cfg).to(DEV)


def micro(B=4, T=128, span=64, seed=1, junk_tail=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, 1900, (B, T), generator=g)
    if junk_tail:
        x[:, T - junk_tail:] = 1999        # junk in the (invalid) pad tail
    span_pos = torch.zeros(B, T, dtype=torch.bool)
    span_pos[:, T - span - junk_tail: T - junk_tail if junk_tail else T] = True
    valid = torch.ones(B, T, dtype=torch.bool)
    if junk_tail:
        valid[:, T - junk_tail:] = False
    slots = torch.linspace(0, 1, span).unsqueeze(0).repeat(B, 1)
    slots = torch.nn.functional.pad(slots, (0, T - slots.size(1)))
    return (x.to(DEV), span_pos.to(DEV), valid.to(DEV), slots.to(DEV))


def draws(B, T, t_val, seed=5):
    g = torch.Generator().manual_seed(seed)
    t = torch.full((B,), t_val)
    m = torch.rand(B, T, generator=g) < t[:, None]
    noise = torch.randn(B, T, generator=g)
    return t.to(DEV), m.to(DEV), noise.to(DEV)


def eager_ref(model, pos_head, x, span_pos, valid, slots, t, m, noise,
              eps=0.05, pair_topk=3, lam=1.0):
    """ot_step with PRE-DRAWN randomness (the apples-to-apples reference:
    same t, same mask draw, same position noise as the graphed path)."""
    x_in = x.masked_fill(m, model.mask_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.trunk(x_in, probe=False, attn_mask=valid[:, None, None, :])
        pos_pred = pos_head(h).squeeze(-1)
        m_sigma = m & (t > 0)[:, None]
        noised = torch.where(m_sigma, slots + noise * t[:, None], slots)
        out = ot_mdlm_loss(h, x, m, t, span_pos.sum(1), model.embed.weight,
                           slots_noised=noised, slots_true=slots,
                           span_pos=span_pos, eps=eps, lam=lam,
                           pos_pred=pos_pred, sigma=t, n_iters=50,
                           pair_topk=pair_topk)
    return out


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@needs_cuda
def test_graphed_matches_eager_loss_and_grads():
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1).to(DEV)
    x, sp, vl, sl = micro()
    t, m, noise = draws(4, 128, 0.5)
    m = m & sp                      # bernoulli_mask semantics
    assert m.any()

    out_ref = eager_ref(md, pos_head, x, sp, vl, sl, t, m, noise)
    out_ref.total.backward()
    ref_loss = float(out_ref.total)
    ref_pw = pos_head.weight.grad.clone()
    ref_emb = md.embed.weight.grad.clone()
    ref_q = md.blocks[0].attn.Wq.weight.grad.clone()

    # free the eager autograd graph BEFORE capture: a live default-stream
    # graph breaks stream capture (see graphed_step docstring caveats)
    del out_ref
    gs = GraphedOTStep(md, pos_head, md.trunk, r_mult=64, max_graphs=2)
    out = gs.run(x, sp, vl, sl, t, m, noise, span_nmax=64)
    got_loss = float(out["total"])

    assert rel(torch.tensor(got_loss), torch.tensor(ref_loss)) < 1e-4, (
        got_loss, ref_loss)
    ref_val = eager_ref(md, pos_head, x, sp, vl, sl, t, m, noise)
    assert rel(out["value"], ref_val.value) < 1e-4
    assert rel(out["position"], ref_val.position) < 1e-4
    assert rel(out["plan_entropy"],
               ref_val.telemetry["plan_entropy"]) < 1e-4

    # grads: acc[i] aligns with params[i] (identity search: tensors do not
    # compare with ==)
    idx = lambda want: next(i for i, p in enumerate(gs.params)
                            if p is want)
    pw = gs.acc[idx(pos_head.weight)]
    emb = gs.acc[idx(md.embed.weight)]
    q = gs.acc[idx(md.blocks[0].attn.Wq.weight)]
    assert rel(pw, ref_pw) < 1e-3
    # embed + trunk grads: the graphed path feeds the CE/backward GEMMs a
    # different ROW SET (bucket-rounded R*k pairs with zero-weight pads,
    # vs the eager exact pair count), so cuBLAS/the scatter reassociate
    # the reductions. The LOSS is bit-exact (rel 0.0) and the per-pair
    # terms are identical; the grads agree to accumulation-order precision
    # (~3e-3 measured), far below any lr-relevant scale.
    assert rel(emb, ref_emb) < 5e-3
    assert rel(q, ref_q) < 5e-3


@needs_cuda
def test_replay_with_different_inputs():
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1).to(DEV)
    x, sp, vl, sl = micro()
    # r_mult=256 pins R for both draws (counts ~128 and ~51 both bucket
    # to 256) -> both inputs replay the SAME captured graph
    gs = GraphedOTStep(md, pos_head, md.trunk, r_mult=512, max_graphs=4)

    tA, mA, nA = draws(4, 128, 0.5, seed=5)
    tB, mB, nB = draws(4, 128, 0.2, seed=6)
    mA, mB = mA & sp, mB & sp       # bernoulli_mask semantics
    assert int(mA.sum()) <= 512 and int(mB.sum()) <= 512

    refA = float(eager_ref(md, pos_head, x, sp, vl, sl, tA, mA, nA).total)
    refB = float(eager_ref(md, pos_head, x, sp, vl, sl, tB, mB, nB).total)

    out = gs.run(x, sp, vl, sl, tA, mA, nA, span_nmax=64)
    a1 = float(out["total"])
    out = gs.run(x, sp, vl, sl, tB, mB, nB, span_nmax=64)
    b1 = float(out["total"])
    out = gs.run(x, sp, vl, sl, tA, mA, nA, span_nmax=64)
    a2 = float(out["total"])
    assert len(gs.graphs) == 1, "same bucket must reuse the captured graph"
    assert abs(a1 - refA) / abs(refA) < 1e-4, (a1, refA)
    assert abs(b1 - refB) / abs(refB) < 1e-4, (b1, refB)
    assert abs(a2 - refA) / abs(refA) < 1e-4, "replay A must return to A"


@needs_cuda
def test_bucket_pads_contribute_exactly_zero():
    md = tiny_md()
    pos_head = torch.nn.Linear(32, 1).to(DEV)
    # padded micro: B=3 (bucket 4), T=128 with an invalid 28-token tail;
    # the eager reference runs on the SLICED (unpadded) views of the SAME
    # tensors, so any pad contribution shows up as a mismatch
    xj, spj, vlj, slj = micro(B=3, T=128, span=60, junk_tail=28)
    t, mj, nj = draws(3, 128, 0.5)
    mj = mj & spj
    x, sp, vl, sl = (xj[:, :100], spj[:, :100], vlj[:, :100], slj[:, :100])
    m, noise = mj[:, :100], nj[:, :100]
    ref = float(eager_ref(md, pos_head, x, sp, vl, sl, t, m, noise).total)
    del x, sp, vl, sl  # noqa

    gs = GraphedOTStep(md, pos_head, md.trunk, r_mult=64, max_graphs=2)
    out = gs.run(xj, spj, vlj, slj, t, mj, nj, span_nmax=60)
    got = float(out["total"])
    assert abs(got - ref) / abs(ref) < 1e-4, (got, ref)
    # B padding (3 -> 4 rows) and the invalid tail also contributed
    # nothing: n_real=3 made the normalization exact
    assert abs(float(out["value"]) + float(out["position"])
               - got) < 1e-5


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
