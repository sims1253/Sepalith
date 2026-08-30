"""Tests for e2_sft.py (Task 6) and e3_orders.py (Task 7) — CPU only."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))


class FakeTok:
    """Deterministic char-pair tokenizer stand-in: 1 token per 2 chars."""
    def get_vocab_size(self):
        return 32768

    def encode(self, text, add_special_tokens=False):
        ids = [min(30000, 2 * ord(c)) for c in text[::2]] or [1]
        return type("E", (), {"ids": ids})()


def test_e3_orders_ramp(tmp_path):
    import e3_orders
    strata = [dict(name=f"s{i}", path=f"/nonexistent/{i}.npy", blocks=1000 + 100 * i,
                   draw_share=0.1 + 0.05 * i, normalized_share=None,
                   eval_slice=None) for i in range(5)]
    tot = sum(s["draw_share"] for s in strata)
    for s in strata:
        s["normalized_share"] = s["draw_share"] / tot
    order, names = e3_orders.make_order(strata, 500, 7, cand_slot=2)
    assert order.shape == (500, 2) and order.dtype == np.int32
    # first window has ~0 candidate share; last ~0.8
    w = 50
    first = (order[:w, 0] == 2).mean()
    last = (order[-w:, 0] == 2).mean()
    assert first < 0.15, first
    assert 0.65 < last < 0.95, last
    # all indices in avail range (slot-2 stratum has 1200 blocks,
    # holdout 256 -> idx < 944)
    m = order[:, 0] == 2
    assert order[m, 1].max() < 944
    # control: no ramp
    ctrl, _ = e3_orders.make_order(strata, 500, 8, cand_slot=None)
    frac = (ctrl[:, 0] == 2).mean()
    assert abs(frac - strata[2]["normalized_share"]) < 0.05, frac


def test_e2_row_render():
    import e2_sft
    pre, comp, suf = e2_sft._row_parts(
        dict(prefix="['a(', 'b)']", region_new="['# c']", suffix="['d']"))
    assert pre == "a(\nb)" and comp == "# c" and suf == "d"
    # plain-string fields pass through
    p2, c2, s2 = e2_sft._row_parts(dict(prefix="x", region_new="y", suffix=""))
    assert (p2, c2, s2) == ("x", "y", "")


def test_e2_build_mask_alignment(tmp_path, monkeypatch):
    import e2_sft
    monkeypatch.setattr(e2_sft, "TOK_PATH", "fake")
    import e2_sft as E
    # rows: prompt 4 tok, completion 6 tok, suffix 4 tok
    rows = [dict(prefix="abcdefgh", region_new="IJMNOPQRST", suffix="uvwxyz")]
    src = tmp_path / "scen"
    src.mkdir()
    (src / "f.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows * 3) + "\n")
    import tokenizers  # noqa: F401
    # monkeypatch the Tokenizer.from_file used inside build()
    class FakeTokWrap(FakeTok):
        pass
    import types
    fake_mod = types.ModuleType("tokenizers")
    fake_mod.Tokenizer = types.SimpleNamespace(
        from_file=lambda p: FakeTokWrap())
    monkeypatch.setitem(sys.modules, "tokenizers", fake_mod)
    out = tmp_path / "e2"
    rc = E.build(["--src", str(src), "--files", "f.jsonl",
                  "--out", str(out), "--max-tokens", "1e9",
                  "--holdout-mod", "1000"])  # everything to train
    assert rc == 0
    b = np.load(out / "train_blocks.npy")
    m = np.load(out / "train_mask.npy")
    assert b.shape[1] == 1025 and m.shape[1] == 1024
    # rows are PACKED: 3 identical rows share the block. FakeTok yields
    # 4 prompt tok + 5 completion tok + 3 suffix tok = 12/row (mask =
    # token-slot flags of slots 1..1024; completion slots 4..8 -> idx 3..7)
    assert m[0, :3].sum() == 0
    assert m[0].sum() == 15  # 3 completions x 5 masked tok
    for k in range(3):
        assert m[0, 12 * k + 3:12 * k + 8].sum() == 5
    # suffix tokens unmasked (3 tok at slots 9..11, nonzero ids)
    assert b[0, 9:12].tolist() != [0] * 3


def test_e2_left_truncation_keeps_completion(tmp_path, monkeypatch):
    import e2_sft as E, sys, types, json as J
    rows = [dict(prefix="p" * 4000, region_new="c" * 10, suffix="")]
    src = tmp_path / "scen"
    src.mkdir()
    (src / "f.jsonl").write_text(J.dumps(rows[0]) + "\n")
    fake_mod = types.ModuleType("tokenizers")
    fake_mod.Tokenizer = types.SimpleNamespace(
        from_file=lambda p: FakeTok())
    monkeypatch.setitem(sys.modules, "tokenizers", fake_mod)
    out = tmp_path / "e2"
    E.build(["--src", str(src), "--files", "f.jsonl", "--out", str(out),
             "--holdout-mod", "1000"])
    m = np.load(out / "train_mask.npy")
    # completion (5 masked tok) fully inside, at the right edge of the
    # token stream (one pad slot may follow)
    assert m[0].sum() == 5
    first = int(np.argmax(m[0]))
    assert m[0, first:first + 5].all()
