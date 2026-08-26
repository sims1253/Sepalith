#!/usr/bin/env python3
"""Self-contained tests for sample_ot.py (run: python3 -m pytest
experiments/training/poc_ddot/test_sample_ot.py -q).

Implements the failing-test list from POC-DDOT Task 5
(docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md), verbatim:
  (a) snapping round-trips (positions -> token layout -> string);
  (b) sampler never touches context;
  (c) reproducible with seed;
plus the length-emergence property (positions clustering -> shorter span,
spreading -> longer; [EMPTY] -> empty string) and cal_length.py's
first-step-confidence length pick.

The sampler targets the OT twin (Task 4's checkpoint, position head
included). Tests use a tiny mock MDGQA-shaped model with a fixed linear
position head so the geometry is hand-computable — no real checkpoint
needed.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_ot as SO
import cal_length as CAL

from experiments.training.poc_diff import EMPTY_ID, MASK_ID


class MockModel(torch.nn.Module):
    """Minimal duck-typed stand-in for the OT twin: `trunk` returns one
    hot-ish feature per position, `embed.weight` maps feature f to a
    one-hot vocab row with logits bias, `predict_positions` returns the
    position recorded at forward time (geometry set by pos_fn). Toy vocab:
    64 real tokens + local mask (64) + local empty (65)."""

    def __init__(self, T, vocab=64, pos_fn=None, conf_bias=8.0):
        super().__init__()
        d = 32
        self.mask_id = vocab
        self.empty_id = vocab + 1
        self.embed = torch.nn.Embedding(vocab + 2, d)
        with torch.no_grad():  # feature f -> strong logit on token f
            self.embed.weight.zero_()
            for v in range(vocab):
                self.embed.weight[v, v % d] = conf_bias
        self._pos_fn = pos_fn or (lambda slot, step: slot)
        self._T = T
        self._step = 0
        self.calls = 0
        self.frames = []              # x snapshot per trunk call

    def trunk(self, x, probe=False, attn_mask=None):
        self.calls += 1
        self._step += 1
        self.frames.append(x.clone())   # for the never-touches-context check
        B, T = x.shape
        step = self._step
        h = torch.zeros(B, T, self.embed.weight.size(1))
        for b in range(B):
            masked = (x[b] == self.mask_id).nonzero(as_tuple=True)[0]
            # masked features are SLOT-relative (window slot index), so the
            # confidence geometry doesn't shift with prompt length
            first = int(masked[0]) if masked.numel() else T
            for t in range(T):
                if t >= first:
                    h[b, t, ((t - first) * (b + 1)) %
                      self.embed.weight.size(1)] = 1.0
                else:
                    h[b, t, t % self.embed.weight.size(1)] = 1.0
        # positions: the mock predicts pos_fn(slot, step) for masked slots
        self._last_pos = torch.zeros(B, T)
        for b in range(B):
            for t in range(T):
                self._last_pos[b, t] = self._pos_fn(t, step)
        return h

    def predict_positions(self, h):
        """Mock position head: recover the position recorded at forward
        time (geometry is set by pos_fn, not by features)."""
        return self._last_pos


# --- (a) snapping round-trips ------------------------------------------------

def test_snap_round_trips_layout_and_string():
    # 5 tokens at spread positions -> snapped back to distinct slots,
    # ordered, decoded. slot = round(p * (window-1)).
    values = [11, 12, 13, 14, 15]
    positions = [0.0, 0.26, 0.5, 0.74, 1.0]
    conf = [0.9] * 5
    snapped = SO.snap_slots(values, positions, conf, window=16)
    assert snapped["ids"] == [11, 12, 13, 14, 15]
    assert snapped["slots"] == [0, 4, 8, 11, 15]


def test_snap_collides_to_shorter_span():
    # Two tokens snapping to the same slot merge (higher confidence wins):
    # length EMERGES from position clustering. window=4: slot=round(p*3).
    values = [21, 22, 23]
    positions = [0.0, 0.34, 0.36]
    conf = [0.5, 0.9, 0.4]          # 22 beats 23 on the shared slot
    snapped = SO.snap_slots(values, positions, conf, window=4)
    assert len(snapped["ids"]) == 2
    assert snapped["ids"] == [21, 22]  # 23 lost the collision, order kept


def test_snap_empty_span():
    assert SO.snap_slots([], [], [], window=8)["ids"] == []
    # all-[EMPTY] predictions decode to the empty string via poc_diff's
    # decode_span convention
    ids = [EMPTY_ID, EMPTY_ID]
    assert SO.decode_ot_span(ids) == ""


# --- (b) sampler never touches context ---------------------------------------

def test_sampler_never_touches_context():
    model = MockModel(T=12)
    # equal-length prompts: no pad tokens in the context zone, so the
    # context block is exactly checkable in every trunk-call snapshot
    prompt = [[5, 6, 7], [8, 9, 10]]
    window = 4
    out = SO.sample_ot_spans(model, prompt, window, steps=2, temperature=0.0)
    assert model.frames, "mock never saw a forward pass"
    want = torch.tensor(prompt)
    for frame in model.frames:
        assert torch.equal(frame[:, :3], want), (
            "context positions were written during sampling")
    assert out["pred_ids"].shape[0] == 2


# --- (c) reproducible with seed ------------------------------------------------

def test_reproducible_with_seed():
    m1 = MockModel(T=12)
    m2 = MockModel(T=12)
    prompt = [[5, 6, 7]]
    g1 = torch.Generator().manual_seed(1273)
    g2 = torch.Generator().manual_seed(1273)
    a = SO.sample_ot_spans(m1, prompt, window=6, steps=3, temperature=1.0,
                           generator=g1)
    b = SO.sample_ot_spans(m2, prompt, window=6, steps=3, temperature=1.0,
                           generator=g2)
    assert torch.equal(a["pred_ids"], b["pred_ids"])
    assert torch.allclose(a["slots"], b["slots"])


# --- joint sampling geometry ----------------------------------------------------

def test_positions_drive_layout_after_snap():
    # Mock predicts positions squeezed into the first half of the window:
    # after snapping, tokens occupy early slots; ordering follows positions.
    def pos_fn(slot, step):
        return (slot % 4) * 0.1          # cluster into [0, 0.3]
    model = MockModel(T=12, pos_fn=pos_fn)
    out = SO.sample_ot_spans(model, [[5]], window=8, steps=4, temperature=0.0)
    assert out["slots"].numel() and (out["slots"] <= 6).all(), (
        "clustered positions must snap to early slots")


# --- cal_length -----------------------------------------------------------------

class ConfMock(MockModel):
    """Confidence profile: high on the first `true_len` window slots, low
    after — the first-step signal CAL-style search keys on."""

    def __init__(self, T, true_len, vocab=64):
        super().__init__(T, vocab)
        self.true_len = true_len
        with torch.no_grad():
            self.embed.weight.zero_()
            for v in range(vocab):
                self.embed.weight[v, v % 32] = 8.0 if v < true_len else 0.1
            if true_len == 0:  # empty candidate: EMPTY logit must dominate
                self.embed.weight[self.empty_id, 0] = 8.0


def test_cal_length_picks_confident_prefix():
    m = ConfMock(T=16, true_len=5)
    picked = CAL.pick_length(m, [[3, 4]], window=16)
    assert picked[0] == 5


def test_cal_length_empty_candidate():
    # All-low confidence + a high-confidence [EMPTY] logit at slot 0 ->
    # the empty span (length 0) wins.
    m = ConfMock(T=16, true_len=0)
    picked = CAL.pick_length(m, [[3, 4]], window=16)
    assert picked[0] == 0


def test_cal_length_deterministic():
    m = ConfMock(T=16, true_len=7)
    assert CAL.pick_length(m, [[3, 4]], window=16) == \
        CAL.pick_length(m, [[3, 4]], window=16)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {e!r}")
    sys.exit(1 if failures else 0)
