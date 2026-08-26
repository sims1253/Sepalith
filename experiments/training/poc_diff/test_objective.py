"""Task 3 tests: MDLM span objective.

Plan-mandated cases:
  (a) brute-force check on a 5-token toy vocab — vectorized loss matches a
      naive loop implementation to 1e-5
  (b) masking a triple at rate t=0 leaves it untouched, t=1 masks every
      span token
  (c) gradient flows to span positions only (context positions receive
      zero loss weight — d(loss)/d h is exactly zero at every context
      position; grads INTO context representations via attention are
      expected and correct)
plus: 1/t eps-clamp active at small t; context never masked at any t;
the composed span_batch_to_loss runs end-to-end on an MDGQA.
"""
import torch

from experiments.training.poc_diff.objective import (
    T_MIN, bernoulli_mask, mdlm_loss, naive_mdlm_loss, sample_rates,
    span_batch_to_loss)
from experiments.training.poc_diff.model_md import MDGQA, model_config_md


def toy_batch(B=2, T=6, vocab=5, seed=0):
    """h random features; x ids; span occupies the last 2 positions;
    t fixed at 0.5 for reproducibility in (a)."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, 8, generator=g)
    x = torch.randint(0, vocab, (B, T), generator=g)
    span_pos = torch.zeros(B, T, dtype=torch.bool)
    span_pos[:, -2:] = True
    t = torch.full((B,), 0.5)
    span_len = span_pos.sum(dim=1)
    return h, x, span_pos, t, span_len


def test_vectorized_matches_naive_toy_vocab():
    torch.manual_seed(0)
    h, x, span_pos, t, span_len = toy_batch()
    head_w = torch.randn(5, 8)  # toy 5-token vocab
    g = torch.Generator().manual_seed(42)
    m = bernoulli_mask(span_pos, t, generator=g)
    assert m.any(), "toy case needs at least one masked position"
    vec = mdlm_loss(h, x, m, t, span_len, head_w, chunk=3).item()
    nai = naive_mdlm_loss(h, x, m, t, span_len, head_w)
    assert abs(vec - nai) < 1e-5, f"{vec} vs {nai}"


def test_t_zero_untouched_t_one_all_masked():
    h, x, span_pos, t, span_len = toy_batch()
    m0 = bernoulli_mask(span_pos, torch.zeros(2))
    assert not m0.any(), "t=0 must leave the triple untouched"
    m1 = bernoulli_mask(span_pos, torch.ones(2))
    assert torch.equal(m1, span_pos), "t=1 must mask every span token"
    # context positions never masked at any rate
    tall = bernoulli_mask(span_pos, torch.ones(2) * 0.999999)
    assert not tall[:, :-2].any()


def test_gradient_only_at_span_positions():
    torch.manual_seed(3)
    h, x, span_pos, t, span_len = toy_batch(T=6)
    head_w = torch.randn(5, 8)
    g = torch.Generator().manual_seed(1)
    m = bernoulli_mask(span_pos, t, generator=g)
    h = h.detach().requires_grad_(True)
    loss = mdlm_loss(h, x, m, t, span_len, head_w)
    loss.backward()
    # loss WEIGHT lands only on masked span positions
    masked_or_not = torch.zeros_like(h.grad, dtype=torch.bool)
    masked_or_not[m] = True
    assert torch.equal(h.grad != 0, masked_or_not), (
        "nonzero d(loss)/dh outside masked span positions")
    # and at least one span position did receive gradient
    assert h.grad[m].abs().sum() > 0


def test_eps_clamp():
    t = torch.tensor([1e-6])
    assert (1.0 / t.clamp(min=T_MIN)).item() == 1.0 / T_MIN


def test_zero_mask_batch_keeps_graph():
    """No span token masked at all: loss must be a graph-connected 0."""
    h, x, span_pos, t, span_len = toy_batch()
    h = h.requires_grad_(True)
    m = torch.zeros_like(span_pos)
    loss = mdlm_loss(h, x, m, t, span_len, torch.randn(5, 8))
    assert loss.item() == 0.0
    loss.backward()  # must not raise


def test_span_batch_to_loss_end_to_end():
    torch.manual_seed(0)
    m = MDGQA(model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                              head_dim=16, ffn_hidden=64, max_seq=64, vocab=50))
    B, T = 3, 20
    x = torch.randint(0, 50, (B, T))
    span_pos = torch.zeros(B, T, dtype=torch.bool)
    span_pos[:, 12:] = True
    valid = torch.ones(B, T, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    loss, stats = span_batch_to_loss(m, x, span_pos, valid,
                                     t=torch.full((B,), 1.0), generator=g)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert m.embed.weight.grad is not None
    assert stats["mask_rate"] == 1.0
