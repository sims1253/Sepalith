"""Task 2 tests: bidirectional TinyGQA-MD.

Plan-mandated cases:
  (a) NO causal leakage — perturbing tokens AFTER position i changes the
      output AT position i (impossible under a causal mask). Stated in the
      plan as "permutation equivariance"; RoPE makes literal permutation
      equivariance false, so the equivalent falsifiable property is tested
      (see model_md.py docstring).
  (b) mask embedding id = 130,560, [EMPTY] = 130,561, table resized, tied
      output head resized to match.
  (c) forward on a (prefix, masked-span, suffix) batch produces logits of
      the extended shape.
Plus: param count == TinyGQA + 2*d_model exactly (and within ±1% of the
plan's 206.5M + 2*768 anchor at full size), [MASK]/[EMPTY] row init
discipline, padding attn_mask isolates pad keys.
"""
import torch
import torch.nn.functional as F

from experiments.training.poc_twin.model import TinyGQA, model_config
from experiments.training.poc_diff.model_md import MDGQA, model_config_md


def small_cfg(**over):
    return model_config_md(d_model=64, n_layers=2, n_q=4, n_kv=2,
                           head_dim=16, ffn_hidden=128, max_seq=128, vocab=200)


def test_no_causal_leakage_suffix_perturbation():
    torch.manual_seed(0)
    m = MDGQA(small_cfg()).eval()
    x = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        out1 = m(x, probe=False)
        x2 = x.clone()
        x2[0, 6:] = (x2[0, 6:] + 7) % 200  # perturb positions AFTER 5
        out2 = m(x2, probe=False)
    delta_before = (out1[0, :6] - out2[0, :6]).abs().max().item()
    assert delta_before > 1e-4, (
        "outputs at positions before the perturbation did not change — "
        "attention is (partially) causal")


def test_identity_control():
    torch.manual_seed(0)
    m = MDGQA(small_cfg()).eval()
    x = torch.randint(0, 200, (1, 10))
    with torch.no_grad():
        a = m(x, probe=False)
        b = m(x.clone(), probe=False)
    assert torch.equal(a, b)


def test_vocab_extension_and_tied_head():
    cfg = model_config_md()  # full-size defaults
    assert cfg["mask_id"] == 130_560 and cfg["empty_id"] == 130_561
    assert cfg["vocab"] == 130_562
    m = MDGQA(small_cfg())   # small model, same convention
    assert m.embed.weight.shape == (202, 64)
    assert m.mask_id == 200 and m.empty_id == 201
    x = torch.randint(0, 200, (2, 16))
    logits = m(x, probe=False)
    assert logits.shape == (2, 16, 202)  # tied head resized with the table
    h = m.trunk(x, probe=False)
    assert torch.allclose(logits, F.linear(h, m.embed.weight), atol=1e-5)


def test_new_row_init_discipline():
    torch.manual_seed(7)
    m = MDGQA(small_cfg())
    base_mean = m.embed.weight[:200].mean(dim=0)
    assert torch.allclose(m.embed.weight[200], base_mean, atol=1e-6), \
        "[MASK] row must equal the mean base embedding"
    diff = (m.embed.weight[201] - base_mean)
    assert 0.0 < diff.abs().max().item() < 0.2, \
        "[EMPTY] row = mean + small noise"


def test_param_audit_full_size():
    """Exact delta vs TinyGQA and the plan's ±1% absolute band."""
    cfg_full = model_config_md()
    m = MDGQA(cfg_full)
    n_md = sum(p.numel() for p in m.parameters())
    n_base = sum(p.numel() for p in TinyGQA(model_config()).parameters())
    assert n_md - n_base == 2 * 768
    anchor = 206.5e6 + 2 * 768
    assert abs(n_md - anchor) / anchor < 0.01


def test_padding_mask_blocks_pad_keys():
    """Pad keys must not influence real positions: appending pads behind a
    padding attn_mask leaves real outputs unchanged."""
    torch.manual_seed(1)
    m = MDGQA(small_cfg()).eval()
    x = torch.randint(0, 200, (1, 10))
    pads = torch.full((1, 5), 199, dtype=torch.long)
    xp = torch.cat([x, pads], dim=1)
    valid = torch.ones(1, 15, dtype=torch.bool)
    valid[0, 10:] = False
    mask = valid[:, None, None, :]  # (B,1,1,T): True = attend
    with torch.no_grad():
        out_short = m(x, probe=False)
        out_padded = m(xp, attn_mask=mask, probe=False)
    assert torch.allclose(out_short, out_padded[:, :10], atol=1e-5), \
        "pad keys leaked into real-position outputs"
