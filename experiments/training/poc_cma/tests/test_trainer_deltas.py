"""Tests for the decay/CMA POC trainer deltas (Tasks 0 + 3).

Run:  .venv/bin/python3 -m unittest discover -s experiments/training/poc_cma/tests
All CPU-only. Covers:
  - LR schedule: new args default to the historical curve BIT-EXACTLY
    (regression gate, plan Task 0), and decay_frac/floor_ratio/const tail
    behave as specified.
  - OrderFileData: consumes the exact index sequence (Task 3).
  - MuonH: radius preserved +-1e-3 over 100 random steps; projection
    disabled == plain Muon bit-exactly (Task 3 tests, plan-verbatim).
  - ELR telemetry: opt-in flag records ||dW||/||W|| and is off by default.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.abspath(os.path.join(HERE, "..", "..", "poc_twin"))
sys.path.insert(0, POC_TWIN)

import train as T  # noqa: E402
from muon import Muon  # noqa: E402
from muonh import MuonH  # noqa: E402


class TestLRSchedule(unittest.TestCase):
    def test_defaults_bit_exact(self):
        """New args must reproduce the frozen historical curve bit-exactly
        for every (step, total_steps, peak) the trainers use."""
        for total in (200, 480, 668, 1600, 1900, 3800):
            for peak in (0.01, 0.004, 0.02):
                for step in range(0, total, max(1, total // 37)):
                    legacy = _legacy_lr_at(step, total, peak)
                    new = T.lr_at(step, total, peak)
                    self.assertEqual(legacy, new,
                                     f"step={step} total={total} peak={peak}")

    def test_decay_frac(self):
        # D arm: decay_frac 0.5 — constant until 50%, then linear to floor
        self.assertEqual(T.lr_at(500, 1000, 1.0, decay_frac=0.5), 1.0)
        self.assertEqual(T.lr_at(750, 1000, 1.0, decay_frac=0.5), 0.55)
        # last step of a 0-idx loop: frac = 499/500 -> just above floor
        self.assertLess(T.lr_at(999, 1000, 1.0, decay_frac=0.5), 0.102)

    def test_floor_ratio(self):
        # matches the legacy formula at the same step (never exactly floor
        # at the last 0-idx step)
        expect = 1.0 * (0.01 + 0.99 * (1 - (999 - 800) / 200))
        self.assertAlmostEqual(T.lr_at(999, 1000, 1.0, floor_ratio=0.01),
                               expect, places=9)

    def test_const_tail(self):
        # KT: 0.2 decay + 5% const tail at floor*peak
        lr = T.lr_at(960, 1000, 1.0, const_tail_frac=0.05)
        self.assertEqual(lr, 0.1)  # inside tail -> floor, not mid-decay
        lr = T.lr_at(799, 1000, 1.0, const_tail_frac=0.05)
        self.assertEqual(lr, 1.0)  # before decay start


def _legacy_lr_at(step, total_steps, peak, warmup_frac=0.015,
                  decay_frac=0.2, floor_ratio=0.1):
    """Verbatim copy of the pre-POC lr_at (the bit-exactness reference)."""
    w = max(1, int(total_steps * warmup_frac))
    if step < w:
        return peak * (step + 1) / w
    d_start = int(total_steps * (1 - decay_frac))
    if step < d_start:
        return peak
    frac = (step - d_start) / max(1, total_steps - d_start)
    return peak * (floor_ratio + (1 - floor_ratio) * (1 - frac))


class TestOrderFileData(unittest.TestCase):
    def _make_draw(self, d):
        os.makedirs(d, exist_ok=True)
        rng = np.random.RandomState(0)
        strs = []
        blocks_per = []
        for si in range(2):
            b = rng.randint(0, 1000, size=(32, 8)).astype(np.int32)
            b[:, 0] = si * 1000 + np.arange(32)  # identifiable rows
            np.save(os.path.join(d, f"s{si}.npy"), b)
            strs.append(dict(name=f"s{si}", path=os.path.join(d, f"s{si}.npy"),
                             blocks=32))
            blocks_per.append(b)
        order = np.stack([rng.randint(0, 2, 64),
                          rng.randint(0, 32, 64)], axis=1).astype(np.int32)
        np.save(os.path.join(d, "uniform_order.idx.npy"), order)
        json = {"draw_id": "test", "strata": strs}
        with open(os.path.join(d, "draw_manifest.json"), "w") as f:
            f.write(__import__("json").dumps(json))
        return order, blocks_per

    def test_exact_sequence(self):
        import json as _json  # noqa: F401
        with tempfile.TemporaryDirectory() as d:
            order, blocks_per = self._make_draw(d)
            data = T.OrderFileData(d, seq_per_step=8)
            for step in range(8):
                xb = data.batch(step)
                idx = order[step * 8:(step + 1) * 8]
                for row, (si, bi) in zip(xb, idx):
                    self.assertTrue((row == blocks_per[si][bi]).all())


class TestMuonH(unittest.TestCase):
    def _params(self, seed=0):
        torch.manual_seed(seed)
        return [torch.nn.Parameter(torch.randn(16, 8)),
                torch.nn.Parameter(torch.randn(8, 16))]

    def test_radius_preserved(self):
        p = self._params()
        r0 = [float(w.norm()) for w in p]
        opt = MuonH(p, lr=0.01, weight_decay=0.0)
        g = torch.Generator().manual_seed(123)
        for i in range(100):
            for w in p:
                w.grad = torch.randn(w.shape, generator=g)
            opt.step()
        opt._capture()
        self.assertLess(opt.radius_error(), 1e-3)

    def test_projection_disabled_equals_muon(self):
        torch.manual_seed(7)
        pa = [torch.nn.Parameter(w.clone()) for w in self._params(7)]
        torch.manual_seed(7)
        pb = [torch.nn.Parameter(w.clone()) for w in self._params(7)]
        oa = MuonH(pa, lr=0.01, weight_decay=0.1, project=False)
        ob = Muon(pb, lr=0.01, weight_decay=0.1)
        g = torch.Generator().manual_seed(99)
        for _ in range(20):
            for wa, wb in zip(pa, pb):
                gr = torch.randn(wa.shape, generator=g)
                wa.grad, wb.grad = gr.clone(), gr.clone()
            oa.step()
            ob.step()
        for wa, wb in zip(pa, pb):
            self.assertTrue(torch.equal(wa, wb))

    def test_elr_telemetry(self):
        p = self._params(3)
        opt = Muon(p, lr=0.01, weight_decay=0.1, track_updates=True)
        self.assertTrue(opt.track_updates)
        for w in p:
            w.grad = torch.randn(w.shape)
        w_pre = [w.detach().clone() for w in p]
        opt.step()
        for w, wp, st in zip(p, w_pre, [opt.state[w] for w in p]):
            d = w.detach() - wp
            self.assertAlmostEqual(float(d.norm()),
                                   st["last_update_norm"], places=5)
            self.assertAlmostEqual(float(w.norm()),
                                   st["last_weight_norm"], places=5)

    def test_elr_off_by_default(self):
        p = self._params(5)
        opt = Muon(p, lr=0.01)
        self.assertFalse(opt.track_updates)
        for w in p:
            w.grad = torch.randn(w.shape)
        opt.step()
        self.assertFalse(any("last_update_norm" in s for s in opt.state.values()))


class TestMPLFit(unittest.TestCase):
    def test_powerlaw_roundtrip(self):
        from mpl_fit import mpl_fit, decay_ratio_estimate
        a, b, c = 500.0, 0.35, 1.5
        ts = [1e7 * (1.3 ** i) for i in range(25)]
        pts = [(t, c + a * t ** (-b)) for t in ts]
        fit = mpl_fit(pts)
        self.assertAlmostEqual(fit["b"], b, places=2)
        self.assertAlmostEqual(fit["c"], c, places=2)
        frac, _ = decay_ratio_estimate(fit)
        self.assertTrue(0.0 <= frac <= 1.0)

    def test_stream_reader(self):
        from mpl_fit import read_loss_series
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as f:
            for i in range(10):
                f.write(__import__("json").dumps(
                    dict(step=i, tokens=(i + 1) * 524288,
                         loss=3.0 - 0.5 * (i / 10))) + "\n")
            f.write("not json\n")
            path = f.name
        try:
            pts = read_loss_series(path)
            self.assertEqual(len(pts), 10)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
