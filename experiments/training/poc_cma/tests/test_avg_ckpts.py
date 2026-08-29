"""Tests for poc_cma/avg_ckpts.py (Task 5). CPU-only."""
import os
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from avg_ckpts import average  # noqa: E402


class TestAvgCkpts(unittest.TestCase):
    def _save(self, d, name, scale, step):
        ck = dict(step=step, cfg=dict(vocab=32768),
                  model={"w": (torch.arange(6, dtype=torch.float32)
                               .reshape(2, 3) * scale).bfloat16()})
        p = os.path.join(d, name)
        torch.save(ck, p)
        return p

    def test_uniform_sma(self):
        with tempfile.TemporaryDirectory() as d:
            ps = [self._save(d, f"tail_{i}.pt", float(i + 1), 100 + i)
                  for i in range(3)]
            out = average(ps)
            w = out["model"]["w"].float()
            expect = torch.arange(6, dtype=torch.float32).reshape(2, 3) * 2.0
            self.assertTrue(torch.allclose(w, expect))
            self.assertEqual(out["step"], 102)  # last ckpt metadata
            self.assertEqual(out["avg_of"], ["tail_0.pt", "tail_1.pt",
                                             "tail_2.pt"])

    def test_weighted(self):
        with tempfile.TemporaryDirectory() as d:
            ps = [self._save(d, "a.pt", 1.0, 10), self._save(d, "b.pt", 3.0, 11)]
            out = average(ps, weights=[3, 1])
            w = out["model"]["w"].float()
            expect = torch.arange(6, dtype=torch.float32).reshape(2, 3) * 1.5
            self.assertTrue(torch.allclose(w, expect))

    def test_rejects_single(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._save(d, "a.pt", 1.0, 1)
            with self.assertRaises(ValueError):
                average([p])


if __name__ == "__main__":
    unittest.main()
