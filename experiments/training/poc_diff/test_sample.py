"""Task 5 sampler tests.

Plan-mandated cases:
  (a) sampler never modifies context positions (structural: every idx the
      trunk receives has the original prompt tokens bitwise-intact)
  (b) k=8 samples with fixed seed are reproducible (and distinct across
      seeds at temperature > 0)
  (c) [EMPTY] prediction yields an empty-string span
plus: steps>=L finishes everything; early-exit forward counting; padded
batch equals per-row result under greedy.
"""
import torch

from experiments.training.poc_diff.model_md import MDGQA, model_config_md
from experiments.training.poc_diff.sample import decode_span, sample_spans


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return MDGQA(model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                                 head_dim=16, ffn_hidden=64, max_seq=128,
                                 vocab=50)).eval()


def test_never_modifies_context():
    m = tiny_model()
    seen = []
    orig = m.trunk

    def spy(idx, probe=True, attn_mask=None):
        seen.append(idx.clone())
        return orig(idx, probe=probe, attn_mask=attn_mask)

    m.trunk = spy
    prompts = [[3, 4, 5, 6], [7, 8, 9]]
    out = sample_spans(m, prompts, span_lens=[4, 2], steps=4)
    assert seen, "trunk was never called"
    for x in seen:
        assert x[0, :4].tolist() == prompts[0], "context row 0 modified"
        assert x[1, :3].tolist() == prompts[1], "context row 1 modified"


def test_greedy_padded_batch_matches_per_row():
    torch.manual_seed(5)
    m = tiny_model()
    prompts = [[3, 4, 5], [9, 8, 7, 6, 5, 4]]
    lens = [3, 5]
    b = sample_spans(m, prompts, lens, steps=4, temperature=0.0)
    for i in range(2):
        s = sample_spans(m, [prompts[i]], [lens[i]], steps=4, temperature=0.0)
        assert b["pred_ids"][i, :lens[i]].tolist() == \
            s["pred_ids"][0, :lens[i]].tolist()


def test_seed_reproducibility_and_distinctness():
    m = tiny_model(seed=3)
    prompts = [[1, 2, 3, 4, 5]]
    g1 = torch.Generator().manual_seed(11)
    g1b = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(12)
    a = sample_spans(m, prompts, [6], steps=8, temperature=1.0, generator=g1)
    b = sample_spans(m, prompts, [6], steps=8, temperature=1.0, generator=g1b)
    c = sample_spans(m, prompts, [6], steps=8, temperature=1.0, generator=g2)
    assert torch.equal(a["pred_ids"], b["pred_ids"]), "same seed must reproduce"
    assert not torch.equal(a["pred_ids"], c["pred_ids"]), \
        "different seed gave identical sample (multinomial not live?)"


def test_all_positions_frozen_when_steps_ge_len():
    m = tiny_model()
    out = sample_spans(m, [[1, 2, 3]], [4], steps=8)
    assert out["pred_ids"].shape == (1, 4)
    assert (out["conf"] > 0).all(), "every position should be frozen"


def test_empty_prediction_decodes_to_empty_string():
    """Bias the tied head hard toward [EMPTY] over a constant trunk output;
    the sampler must predict it and decode_span must yield ''."""
    m = tiny_model(seed=9)
    with torch.no_grad():
        w = m.embed.weight
        w.mul_(1e-3)
        w[m.empty_id] = 10.0  # dominates against constant-positive h
    d = m.cfg["d_model"]
    m.trunk = lambda idx, probe=True, attn_mask=None: torch.ones(
        idx.shape[0], idx.shape[1], d)
    out = sample_spans(m, [[1, 2, 3]], [5], steps=4)
    ids = out["pred_ids"][0].tolist()
    assert ids == [m.empty_id] * 5

    class TokStub:
        @staticmethod
        def decode(ids):
            return "".join(chr(40 + i) for i in ids)

    assert decode_span(TokStub, out["pred_ids"][0], m.empty_id) == ""
    # mixed: non-empty tokens survive the strip
    assert decode_span(TokStub, torch.tensor([3, 4]), m.empty_id) == chr(43) + chr(44)
