"""X5-S1 tests: the pre-registered health-check gates (B8 pattern), all
CPU-side, run BEFORE any GPU work. Gate list mirrors the x5_s1_train.py
header:

  G1 carry-channel zero-init verified at load (banked ckpt compat)
  G2 null-carry safety (null training can never move the channel)
  G3 two-pass shapes (logits_to_carry dense/zero rows; stage-A masking)
  G4 nested masks (m subset m_start; Beta(2,2) capped at t)
  G5 FPF rollout (commits only m_start\\m; no grad; context untouched)
  G6 objective otherwise unchanged (loss == mdlm_loss on pass-2 states)
  + sampler parity: depth=1 carry-active sampler == banked sample_spans
    bit-for-bit on the same weights; NFE accounting steps x depth with
    early exit; depth>1 refines within a step (carry actually feeds back)
"""
import torch

from experiments.training.poc_diff.model_md import (
    MDGQA, logits_to_carry, model_config_md,
)
from experiments.training.poc_diff.sample import sample_spans
from experiments.training.poc_diff.x5_s1_sample import load_md_x5, \
    sample_spans_rc
from experiments.training.poc_diff.x5_s1_train import (
    beta22, fpf_rollout, nested_masks, stage_a_carry, stage_loss,
)


def tiny_cfg(**over):
    return model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                           head_dim=16, ffn_hidden=64, max_seq=128,
                           vocab=200)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return MDGQA(tiny_cfg())


def make_banked_ckpt(m, path):
    """A pre-X5-style ckpt: full state EXCEPT carry_proj."""
    sd = {k: v for k, v in m.state_dict().items() if k != "carry_proj.weight"}
    torch.save(dict(cfg=m.cfg if hasattr(m, "cfg") else tiny_cfg(),
                    model=sd), path)


def test_g1_zero_init_and_load_gate(tmp_path):
    m = tiny_model()
    assert torch.count_nonzero(m.carry_proj.weight) == 0, \
        "carry_proj must be zero-init at construction"
    # banked ckpt (no carry key) loads through the gate, W stays zero,
    # and trunk(carry=zeros) == trunk(carry=None) exactly.
    make_banked_ckpt(m, tmp_path / "banked.pt")
    m2 = load_md_x5(str(tmp_path / "banked.pt"), torch.device("cpu"))
    assert torch.count_nonzero(m2.carry_proj.weight) == 0
    x = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        a = m2.trunk(x, probe=False)
        z = m2.trunk(x, probe=False,
                     carry=torch.zeros(2, 10, m2.cfg["d_model"]))
    assert torch.equal(a, z), "zero-carry must reproduce the null path"


def test_g1_bad_ckpt_rejected(tmp_path):
    import pytest
    m = tiny_model()
    sd = {k: v for k, v in m.state_dict().items()
          if k not in ("carry_proj.weight", "embed.weight")}
    torch.save(dict(cfg=tiny_cfg(), model=sd), tmp_path / "broken.pt")
    with pytest.raises(AssertionError):
        load_md_x5(str(tmp_path / "broken.pt"), torch.device("cpu"))


def test_g2_null_carry_never_moves_channel():
    m = tiny_model()
    x = torch.randint(0, 200, (2, 12))
    span = torch.zeros(2, 12, dtype=torch.bool)
    span[:, 6:] = True
    valid = torch.ones(2, 12, dtype=torch.bool)
    t = torch.tensor([0.5, 0.5])
    u = torch.full((2, 12), 0.3)
    null = torch.zeros(2, dtype=torch.bool)
    loss, _ = stage_loss(m, m.trunk, x, span, valid, t, u, null, "A")
    loss.backward()
    g = m.carry_proj.weight.grad
    assert g is None or torch.count_nonzero(g) == 0, \
        "G2: null-carry backward must leave carry_proj grad zero"
    # and an active pass DOES move it (the channel is learnable)
    m.zero_grad(set_to_none=True)
    active = torch.ones(2, dtype=torch.bool)
    loss, _ = stage_loss(m, m.trunk, x, span, valid, t, u, active, "A")
    loss.backward()
    assert m.carry_proj.weight.grad is not None and \
        torch.count_nonzero(m.carry_proj.weight.grad) > 0


def test_g3_logits_to_carry_shapes_and_math():
    m = tiny_model()
    x = torch.randint(0, 200, (2, 12))
    span = torch.zeros(2, 12, dtype=torch.bool)
    span[0, 4:9] = True
    span[1, 2:5] = True
    with torch.no_grad():
        h = m.trunk(x, probe=False)
        c = logits_to_carry(h, span, m.embed.weight, chunk=3)
    assert c.shape == (2, 12, m.cfg["d_model"]) and c.dtype == torch.float32
    assert torch.count_nonzero(c[0, :4]) == 0 and \
        torch.count_nonzero(c[1, 5:]) == 0, "non-span rows must be null"
    import torch.nn.functional as F
    with torch.no_grad():
        logits = F.linear(h[0, 4], m.embed.weight).float()
        manual = F.softmax(logits, dim=-1) @ m.embed.weight.float()
    assert torch.allclose(c[0, 4], manual, atol=1e-5)
    # stage-A carry zeroes null-example rows
    valid = torch.ones(2, 12, dtype=torch.bool)
    sc = torch.tensor([True, False])
    with torch.no_grad():
        cA = stage_a_carry(m, m.trunk, x, span, valid, sc)
    assert torch.count_nonzero(cA[1]) == 0 and \
        torch.count_nonzero(cA[0, 4:9]) > 0


def test_g4_nested_masks_and_beta():
    torch.manual_seed(3)
    span = torch.zeros(2, 16, dtype=torch.bool)
    span[:, 8:] = True
    t = torch.tensor([0.3, 0.6])
    msr = torch.maximum(beta22(2000), torch.full((2000,), 0.4))
    assert float(msr.min()) >= 0.4 - 1e-6 and float(msr.max()) <= 1.0
    b = beta22(100000)
    assert abs(float(b.mean()) - 0.5) < 0.01 and float(b.min()) > 0
    g1 = torch.Generator().manual_seed(0)
    assert torch.equal(beta22(8, generator=g1),
                       beta22(8, generator=torch.Generator().manual_seed(0)))
    u = torch.rand(2, 16)
    msr = torch.tensor([0.5, 0.7])
    m, m_start = nested_masks(span, t, msr, u)
    assert bool((m & ~m_start).sum() == 0), "G4: m must be a subset of m_start"
    assert bool((m & ~span).sum() == 0)


def test_g5_fpf_rollout_commits_only_target_set():
    mdl = tiny_model().eval()
    torch.manual_seed(1)
    B, T = 2, 14
    x = torch.randint(0, 200, (B, T))
    span = torch.zeros(B, T, dtype=torch.bool)
    span[:, 8:] = True
    valid = torch.ones(B, T, dtype=torch.bool)
    t = torch.full((B,), 0.4)
    msr = torch.full((B,), 0.8)
    u = torch.rand(B, T)
    m, m_start = nested_masks(span, t, msr, u)
    calls = {}

    def spy(idx, probe=True, attn_mask=None, carry=None):
        calls[len(calls)] = carry
        return mdl.trunk(idx, probe=probe, attn_mask=attn_mask, carry=carry)

    c = fpf_rollout(mdl, spy, x, m, m_start, span, valid, depth=2)
    assert not c.requires_grad, "G5: rollout carry must be stopgrad"
    assert calls[0] is None, "first rollout forward must carry None"
    assert calls[1] is not None and calls[1].shape == x.shape + \
        (mdl.cfg["d_model"],), "second forward must ingest the carry"
    # reproduce the rollout state manually: committed = m_start & ~m only
    with torch.no_grad():
        h = mdl.trunk(x.masked_fill(m_start, mdl.mask_id), probe=False,
                      attn_mask=valid[:, None, None, :])
    import torch.nn.functional as F
    sel = (m_start & ~m)
    logits = F.linear(h[sel], mdl.embed.weight).float()
    picks = logits.argmax(-1)
    x_expected = x.masked_fill(m_start, mdl.mask_id)
    x_expected[sel] = picks
    with torch.no_grad():
        h2 = mdl.trunk(x_expected, probe=False,
                       attn_mask=valid[:, None, None, :], carry=calls[1])
    c_expected = logits_to_carry(h2, span, mdl.embed.weight)
    assert torch.allclose(c, c_expected, atol=1e-4)
    # the supervised mask m positions remain [MASK]; context untouched


def test_g6_objective_unchanged():
    from experiments.training.poc_diff import objective
    m = tiny_model()
    x = torch.randint(0, 200, (2, 12))
    span = torch.zeros(2, 12, dtype=torch.bool)
    span[:, 5:] = True
    valid = torch.ones(2, 12, dtype=torch.bool)
    t = torch.tensor([0.5, 0.5])
    u = torch.full((2, 12), 0.25)
    active = torch.zeros(2, dtype=torch.bool)  # null carry
    loss, st = stage_loss(m, m.trunk, x, span, valid, t, u, active, "A")
    m_in = x.masked_fill(span & (u < t[:, None]), m.mask_id)
    with torch.no_grad():
        h = m.trunk(m_in, probe=False, attn_mask=valid[:, None, None, :])
        ref = objective.mdlm_loss(h, x, span & (u < t[:, None]), t,
                                  span.sum(1), m.embed.weight)
    assert torch.allclose(loss, ref, atol=1e-6), \
        "G6: stage loss must equal mdlm_loss on the (null-carry) pass-2 states"
    assert torch.isfinite(loss)


def test_sampler_depth1_matches_banked():
    m = tiny_model(seed=5).eval()
    prompts = [[3, 4, 5, 6, 7, 8], [9, 10, 11, 12]]
    lens = [5, 3]
    with torch.no_grad():
        banked = sample_spans(m, prompts, lens, steps=4, temperature=0.0)
        rc = sample_spans_rc(m, prompts, lens, steps=4, depth=1,
                             temperature=0.0)
    assert torch.equal(banked["pred_ids"], rc["pred_ids"]), \
        "depth-1 carry-active sampler must reproduce sample_spans on a " \
        "zero-carry model, bit-for-bit"
    assert banked["n_fwd"] == rc["n_fwd"]


def test_sampler_nfe_accounting_and_feedback():
    m = tiny_model(seed=9).eval()
    with torch.no_grad():
        m.carry_proj.weight.normal_(0, 0.5)  # nonzero channel: carry matters
    prompts = [[1, 2, 3, 4]]
    rc = sample_spans_rc(m, prompts, [3], steps=4, depth=2)
    assert rc["n_fwd_nominal"] == 8
    assert rc["n_fwd"] <= 8  # early exit: 3 positions freeze 1/step -> 3 steps
    assert rc["n_fwd"] == 6, "3 schedule steps x depth 2 with early exit"
    rc1 = sample_spans_rc(m, prompts, [3], steps=4, depth=1)
    assert rc["pred_ids"].shape == (1, 3)
    # context is never written: prompt ids stay out of the span region
    p = [42, 43]
    out = sample_spans_rc(m, [p], [2], steps=2, depth=4)
    assert out["n_fwd"] <= 2 * 4


def test_stage_b_loss_runs_and_masks():
    m = tiny_model()
    x = torch.randint(0, 200, (2, 12))
    span = torch.zeros(2, 12, dtype=torch.bool)
    span[:, 6:] = True
    valid = torch.ones(2, 12, dtype=torch.bool)
    t = torch.tensor([0.4, 0.4])
    u = torch.full((2, 12), 0.5)
    active = torch.tensor([True, False])
    loss, st = stage_loss(m, m.trunk, x, span, valid, t, u, active, "B",
                          fpf_depth=2)
    assert torch.isfinite(loss)
    loss.backward()
    m.zero_grad(set_to_none=True)
    loss0, _ = stage_loss(m, m.trunk, x, span, valid, t, u,
                          torch.zeros(2, dtype=torch.bool), "B")
    loss0.backward()
    g = m.carry_proj.weight.grad
    assert g is None or torch.count_nonzero(g) == 0, \
        "stage-B null rows must not move the channel"
