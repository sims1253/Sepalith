"""
Tests for muon_hygiene.py (P1 arms) — CPU-only.
Run: uv run python test_muon_hygiene.py  (or pytest).
"""
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.join(os.path.dirname(HERE), "poc_twin")
for p in (HERE, POC_TWIN):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import TinyGQA, model_config  # noqa: E402
from muon import Muon                    # noqa: E402
from muon_hygiene import MuonHygiene, build_optim  # noqa: E402

TINY = dict(vocab=256, d_model=64, n_layers=2, n_q=4, n_kv=2, head_dim=16,
            ffn_hidden=128, max_seq=32)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return TinyGQA(model_config(**TINY))


def _hidden_named(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.ndim == 2 and "embed" not in n]


def _feed_grads(model, step):
    torch.manual_seed(100 + step)
    for _, p in _hidden_named(model):
        p.grad = torch.randn_like(p)


def _params_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_defaults_match_pinned_muon():
    ma, mb = _tiny_model(7), _tiny_model(7)
    mb.load_state_dict(copy.deepcopy(ma.state_dict()))
    oa = Muon([p for _, p in _hidden_named(ma)], lr=0.01, momentum=0.95,
              ns_steps=5, weight_decay=0.1)
    ob = MuonHygiene([p for _, p in _hidden_named(mb)], lr=0.01, momentum=0.95,
                     ns_steps=5, weight_decay=0.1)
    for step in range(3):
        _feed_grads(ma, step)
        _feed_grads(mb, step)
        oa.step()
        ob.step()
    for (n, pa), (_, pb) in zip(_hidden_named(ma), _hidden_named(mb)):
        assert torch.allclose(pa, pb, atol=1e-6), f"drift on {n}"


def test_split_single_step_semantics():
    m_plain = _tiny_model(7)
    m_split = _tiny_model(7)
    m_split.load_state_dict(copy.deepcopy(m_plain.state_dict()))
    n_q, n_kv, hd = TINY["n_q"], TINY["n_kv"], TINY["head_dim"]
    split_map = {}
    for n, p in _hidden_named(m_split):
        if n.endswith("attn.Wq.weight"):
            split_map[p] = (0, [hd] * n_q)
        elif n.endswith(("attn.Wk.weight", "attn.Wv.weight")):
            split_map[p] = (0, [hd] * n_kv)
        elif n.endswith("attn.Wo.weight"):
            split_map[p] = (1, [hd] * n_q)
    o_plain = MuonHygiene([p for _, p in _hidden_named(m_plain)], lr=0.01,
                          weight_decay=0.1)
    o_split = MuonHygiene([p for _, p in _hidden_named(m_split)], lr=0.01,
                          weight_decay=0.1, split_map=split_map)
    _feed_grads(m_plain, 0)
    _feed_grads(m_split, 0)
    before = _params_snapshot(m_plain)
    o_plain.step()
    o_split.step()
    for (n, pp), (_, ps) in zip(_hidden_named(m_plain),
                                _hidden_named(m_split)):
        delta_plain = (pp - before[n]).abs().max()
        delta_split = (ps - before[n]).abs().max()
        if n.endswith(("attn.Wq.weight", "attn.Wk.weight",
                       "attn.Wv.weight", "attn.Wo.weight")):
            # chunked ortho + per-chunk scale must differ from whole-matrix
            assert not torch.allclose(pp, ps, atol=1e-6), f"{n} identical"
            assert delta_split > 0 and delta_plain > 0
        else:
            # non-split params get identical updates from identical grads
            assert torch.allclose(pp, ps, atol=1e-6), f"{n} drifted"


def test_nesterov_differs_from_step2():
    m_plain = _tiny_model(7)
    m_nest = _tiny_model(7)
    m_nest.load_state_dict(copy.deepcopy(m_plain.state_dict()))
    o_plain = MuonHygiene([p for _, p in _hidden_named(m_plain)], lr=0.01)
    o_nest = MuonHygiene([p for _, p in _hidden_named(m_nest)], lr=0.01,
                         nesterov=True)
    _feed_grads(m_plain, 0)
    _feed_grads(m_nest, 0)
    o_plain.step()
    o_nest.step()
    # step 1: buf=(1-m)g, so d_nesterov = g + m·buf = c·g — NS of a positive
    # multiple of g equals NS(g); both arms coincide after step 1.
    for (_, pp), (_, pn) in zip(_hidden_named(m_plain),
                                _hidden_named(m_nest)):
        assert torch.allclose(pp, pn, atol=1e-6)
    _feed_grads(m_plain, 1)
    _feed_grads(m_nest, 1)
    o_plain.step()
    o_nest.step()
    drifted = [n for (n, pp), (_, pn) in zip(_hidden_named(m_plain),
                                             _hidden_named(m_nest))
               if not torch.allclose(pp, pn, atol=1e-6)]
    assert drifted, "nesterov never diverged from plain EMA"


def test_polar_arm_finite_update():
    m = _tiny_model(7)
    o = MuonHygiene([p for _, p in _hidden_named(m)], lr=0.01,
                    polar=True, ns_steps=8, frob_eps=1e-14)
    for step in range(3):
        _feed_grads(m, step)
        o.step()
        for _, p in _hidden_named(m):
            assert torch.isfinite(p).all()


def test_build_optim_arms():
    m = _tiny_model(7)
    muon, adam = build_optim(m, 0.01, 0.004, 0.1, arm="control")
    assert isinstance(muon, Muon)          # control IS the pinned optimizer
    assert len(adam) == 1
    m2 = _tiny_model(7)
    muon_s, _ = build_optim(m2, 0.01, 0.004, 0.1, arm="split")
    assert isinstance(muon_s, MuonHygiene)
    n_q, n_kv, hd = TINY["n_q"], TINY["n_kv"], TINY["head_dim"]
    for n, p in _hidden_named(m2):
        spec = muon_s.split_map.get(p)
        if n.endswith("attn.Wq.weight"):
            assert spec == (0, [hd] * n_q) and sum(spec[1]) == p.size(0)
        elif n.endswith("attn.Wo.weight"):
            assert spec == (1, [hd] * n_q) and sum(spec[1]) == p.size(1)
        elif n.endswith(("attn.Wk.weight", "attn.Wv.weight")):
            assert spec == (0, [hd] * n_kv) and sum(spec[1]) == p.size(0)
        else:
            assert spec is None
    try:
        build_optim(m2, 0.01, 0.004, 0.1, arm="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_split_chunk_extent_mismatch_is_loud():
    m = _tiny_model(7)
    bad = {p: (0, [7] * 999) for n, p in _hidden_named(m)
           if n.endswith("attn.Wq.weight")}
    o = MuonHygiene([p for _, p in _hidden_named(m)], lr=0.01,
                    split_map=bad)
    _feed_grads(m, 0)
    try:
        o.step()
        raise AssertionError("expected AssertionError on chunk extent")
    except AssertionError as e:
        assert "split chunks" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
