#!/usr/bin/env python3
"""Self-contained tests for eval_ot.py (run: python3 -m pytest
experiments/training/poc_ddot/test_eval_ot.py -q).

DDOT Task 5 step 3 — the three-way eval runner's new metric layer and arm
plumbing, per docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md:
  - length-MAE and per-bucket breakdown;
  - position-MSE (insertion-point error) — operationalized honestly: in
    the astfim anatomy the GT insertion point is always the cursor, so the
    metric is the OT arm's spurious re-anchoring, mean squared
    (first-snapped-slot / window); fixed-window arms anchor exactly and
    score 0. LOWER is better for everyone — the OT arm "beats" only by
    matching 0, which is the correct reading under the frozen kill test;
  - [EMPTY] precision/recall (astfim has zero GT empties — recall is
    reported degenerate-with-n, precision live);
  - CAL arm plumbing: pick_length feeds sample_spans (tiny-model smoke);
  - OT arm plumbing: PosModel wrapper + sample_ot_spans (tiny-model smoke).

The smokes follow eval_spans' --tiny pattern: tiny random MDGQA on CPU,
2 eval rows, GPU untouched (RL-run-4 is live per the board).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ot as EO

from experiments.training.poc_diff import EMPTY_ID


# --- length-MAE ---------------------------------------------------------------

def test_length_mae_tokens():
    rows = [dict(pred_len=10, gt_len=8), dict(pred_len=0, gt_len=3),
            dict(pred_len=12, gt_len=12)]
    assert EO.length_mae(rows) == pytest.approx((2 + 3 + 0) / 3)


def test_length_mae_empty_rows():
    assert EO.length_mae([]) == 0.0


# --- position-MSE (spurious re-anchoring) --------------------------------------

def test_position_mse_zero_for_window_arms():
    # Fixed-window arms anchor at the cursor by construction.
    rows = [dict(first_slot=0), dict(first_slot=0)]
    assert EO.position_mse(rows, window=256) == 0.0


def test_position_mse_penalizes_reanchoring():
    rows = [dict(first_slot=0), dict(first_slot=25)]  # 25/256 off-cursor
    want = (25 / 256) ** 2 / 2
    assert EO.position_mse(rows, window=256) == pytest.approx(want, rel=1e-6)


# --- [EMPTY] precision / recall -------------------------------------------------

def test_empty_precision_recall():
    rows = [dict(pred_empty=True, gt_empty=False),   # FP
            dict(pred_empty=True, gt_empty=False),   # FP
            dict(pred_empty=False, gt_empty=False),
            dict(pred_empty=False, gt_empty=False)]
    p, r, n_gt = EO.empty_precision_recall(rows)
    assert p == pytest.approx(0.0)          # no TP, 2 FP
    assert r == 0.0 and n_gt == 0           # recall degenerate on astfim


def test_empty_precision_recall_with_truth():
    rows = [dict(pred_empty=True, gt_empty=True),    # TP
            dict(pred_empty=True, gt_empty=False),  # FP
            dict(pred_empty=False, gt_empty=True)]  # FN
    p, r, n_gt = EO.empty_precision_recall(rows)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert n_gt == 2


def test_pred_empty_detection():
    assert EO.is_pred_empty([]) is True
    assert EO.is_pred_empty([EMPTY_ID, EMPTY_ID]) is True
    assert EO.is_pred_empty([5, 6]) is False


# --- arm plumbing smokes (tiny models, CPU, 2 rows) ------------------------------

def _tiny_md():
    from experiments.training.poc_diff.model_md import MDGQA, model_config_md
    torch.manual_seed(0)
    m = MDGQA(model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                              head_dim=16, ffn_hidden=64, max_seq=1024,
                              vocab=130560))
    return m.eval()


def _eval_rows(n=2):
    p = Path("/tmp/poc_diff/eval_triples.jsonl")
    if not p.exists():
        pytest.skip("/tmp/poc_diff/eval_triples.jsonl not present (tmpfs)")
    import json
    return [json.loads(l) for l in open(p, encoding="utf-8")][:n]


def test_cal_arm_smoke():
    """pick_length -> sample_spans with the picked length; shapes + metrics."""
    md = _tiny_md()
    rows = _eval_rows()
    prompts = [r["prompt_ids"] for r in rows]
    lens = EO.pick_length(md, prompts, window=32)
    assert len(lens) == len(rows) and all(0 <= l <= 32 for l in lens)
    from experiments.training.poc_diff.sample import sample_spans
    for r, L in zip(rows, lens):
        if L == 0:
            continue
        out = sample_spans(md, [r["prompt_ids"]], [L], steps=4,
                           temperature=0.0)
        assert out["pred_ids"].shape == (1, L)


def test_ot_arm_smoke():
    """PosModel wrapper + sample_ot_spans end-to-end on a tiny MDGQA."""
    md = _tiny_md()
    rows = _eval_rows()
    pos = torch.nn.Linear(32, 1)
    with torch.no_grad():
        pos.bias.fill_(0.5)             # all slots predict coordinate 0.5
    model = EO.PosModel(md, pos)
    out = EO.sample_ot_spans(model, [r["prompt_ids"] for r in rows],
                             window=16, steps=4, temperature=0.0)
    assert out["pred_ids"].shape[0] == len(rows)
    # all positions 0.5 -> every token snaps to slot round(0.5*15)=8 ->
    # collisions merge to a single-token span
    lengths = (out["pred_ids"] != 1).sum(dim=1).tolist()
    assert all(l == 1 for l in lengths), lengths
    assert (out["slots"] <= 15).all()


def test_posmodel_delegates():
    md = _tiny_md()
    pos = torch.nn.Linear(32, 1)
    m = EO.PosModel(md, pos)
    assert m.mask_id == md.mask_id
    assert m.empty_id == md.empty_id
    x = torch.randint(0, 100, (1, 12))
    x[0, 8:] = m.mask_id
    h = m.trunk(x, probe=False)
    assert h.shape == (1, 12, 32)
    p = m.predict_positions(h)
    assert p.shape == (1, 12)


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
