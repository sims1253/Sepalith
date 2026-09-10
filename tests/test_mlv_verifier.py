"""Tests for experiments/eval/mlv_verifier.py (MLV EBT verifier v1).

Run: cd <worktree root> && .venv/bin/python -m pytest tests/test_mlv_verifier.py -q

Pure-CPU, deterministic, no network, no serving, no NAS. Covers:
  byte tokenizer, encoder/nesting shapes, MRL-NCE loss, metric identities,
  group-disjoint split, episode loader, synthetic train smoke, ckpt roundtrip,
  and the --smoke CLI entry point.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "experiments" / "eval" / "mlv_verifier.py"
_spec = importlib.util.spec_from_file_location("mlv_verifier", _SCRIPT)
mlv = importlib.util.module_from_spec(_spec)
sys.modules["mlv_verifier"] = mlv
_spec.loader.exec_module(mlv)


def test_tokenize_bytes_and_pad():
    ids = mlv.tokenize("ab", max_len=8)
    assert ids[:2] == [97, 98]
    assert ids[2:] == [mlv.PAD_ID] * 6
    assert all(0 <= i < mlv.N_TOKENS for i in ids)


def test_encoder_and_nesting_shapes():
    m = mlv.MLVVerifier(d_model=32)
    p_ids, p_m = mlv.batch_encode(["mean(x)", "filter(y)"], max_len=16)
    c_ids, c_m = mlv.batch_encode(["mean", "filter"], max_len=16)
    zp, zc = m.reps(p_ids, p_m, c_ids, c_m)
    assert zp.shape == (2, 512) and zc.shape == (2, 512)
    en = m.energies(zp, zc)
    assert set(en) == {128, 256, 512}
    assert all(v.shape == (2,) for v in en.values())


def test_mrl_nce_loss_finite_and_learns():
    torch.manual_seed(0)
    m = mlv.MLVVerifier(d_model=32)
    rows = mlv.synthetic_rows(n=32, seed=0)
    p_ids, p_m, c_ids, c_m, y = mlv.collate(rows, max_len=32)
    zp, zc = m.reps(p_ids, p_m, c_ids, c_m)
    l0, _ = mlv.mrl_nce_loss(m, zp, zc, y)
    assert torch.isfinite(l0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        zp, zc = m.reps(p_ids, p_m, c_ids, c_m)
        l, _ = mlv.mrl_nce_loss(m, zp, zc, y)
        opt.zero_grad()
        l.backward()
        opt.step()
    assert float(l) < float(l0)


def test_auroc_identities():
    assert mlv.auroc([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]) == 1.0
    assert mlv.auroc([0.0, 0.0, 0.0, 0.0], [1, 1, 0, 0]) == 0.5
    assert mlv.auroc([0.0, 1.0, 2.0, 3.0], [1, 1, 0, 0]) == 0.0


def test_pairwise_accuracy_and_ece():
    assert mlv.pairwise_accuracy([2.0, 2.0, 0.0, 0.0], [1, 1, 0, 0]) == 1.0
    assert mlv.ece([10.0] * 4, [1, 1, 1, 1]) < 0.01


def test_retention_gate():
    tr_s = [float(i) for i in range(100)]
    g = mlv.retention_gate(tr_s[:50], [1] * 50, [5.0] * 10 + [-5.0] * 10,
                           [1] * 10 + [0] * 10)
    assert g["retention"] == 1.0
    assert g["fp_suppression"] == 1.0


def test_group_split_disjoint_and_deterministic():
    rows = mlv.synthetic_rows(n=32, seed=1)
    tr1, te1 = mlv.group_split(rows, 0.25, seed=0)
    tr2, te2 = mlv.group_split(rows, 0.25, seed=0)
    assert {r["group"] for r in tr1}.isdisjoint({r["group"] for r in te1})
    assert [r["completion"] for r in te1] == [r["completion"] for r in te2]
    assert len(tr1) + len(te1) == len(rows)


def test_load_episode_pairs_fixture(tmp_path):
    eps = [{"key": "pkg@1", "points": [
        {"prompt": "p1", "proposal": "good code", "decision": "accepted"},
        {"prompt": "p2", "proposal": "bad code", "decision": "false_suggestion"},
        {"prompt": "p3", "proposal": "meh", "decision": "dismissed"},
        {"prompt": "p4", "proposal": "", "decision": "accepted"},
        {"prompt": "p5", "proposal": "stop", "decision": "correct_stop"},
    ]}]
    ep = tmp_path / "ep.jsonl"
    ep.write_text("\n".join(json.dumps(e) for e in eps))
    rows = mlv.load_episode_pairs(ep)
    assert [(r["label"]) for r in rows] == [0, 0, 1]
    assert all(r["group"] == "pkg@1" for r in rows)


def test_train_smoke_beats_chance():
    rows = mlv.synthetic_rows(n=64, seed=0)
    rep = mlv.run_train(rows, out=None, epochs=3, seed=0, d_model=32,
                        max_len=32, batch=16)
    assert rep["n_train"] + rep["n_test"] == 64
    assert rep["metrics_test"]["auroc"] > 0.5


def test_ckpt_roundtrip(tmp_path):
    rows = mlv.synthetic_rows(n=32, seed=2)
    ckpt = tmp_path / "m.pt"
    mlv.run_train(rows, out=ckpt, epochs=1, seed=0, d_model=16, max_len=32)
    assert ckpt.exists()
    rep = mlv.run_eval(ckpt, rows)
    assert rep["n"] == 32
    assert 0.0 <= rep["metrics"]["auroc"] <= 1.0


def test_cli_smoke_train(tmp_path):
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "train", "--smoke"],
        capture_output=True, text=True, cwd=REPO, timeout=600)
    assert r.returncode in (0, 2), r.stderr[-2000:]
    rep = json.loads(r.stdout)
    assert rep["metrics_test"]["auroc"] > 0.5
