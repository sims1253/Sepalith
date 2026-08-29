"""Tests for poc_cma/data_prep.py (Task 1). CPU-only, synthetic strata —
never touches the NAS (paths in the synthetic manifest are dummy but
load_strata must not stat them; only main() does the existence check)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import data_prep as dp  # noqa: E402


def _synthetic_manifest(d):
    strata = {
        "alpha": dict(path=f"{d}/alpha.npy", draw_share=0.6, blocks=1000),
        "beta": dict(path=f"{d}/beta.npy", draw_share=0.3, blocks=500),
        "gamma": dict(path=f"{d}/gamma.npy", draw_share=0.09, blocks=200),
        # adaptive rule: not packed -> excluded, share re-cut pro-rata
        "missing": dict(path=f"{d}/missing.npy", draw_share=0.01, blocks=0),
    }
    m = dict(total_draw_tokens=1e9, seq=1025, strata=strata)
    p = os.path.join(d, "manifest.json")
    with open(p, "w") as f:
        json.dump(m, f)
    return p


class TestDataPrep(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            mp = _synthetic_manifest(d)
            strata = dp.load_strata(mp)
            self.assertEqual([s["name"] for s in strata],
                             ["alpha", "beta", "gamma"])
            # pro-rata normalization over packed strata only
            self.assertAlmostEqual(sum(s["normalized_share"] for s in strata),
                                   1.0, places=9)
            order, strata = dp.make_draw(strata, 10_000, seed=1273)
            self.assertEqual(order.shape, (10_000, 2))
            self.assertEqual(order.dtype, np.int32)
            cnt = np.bincount(order[:, 0], minlength=3)
            # shares within 3% absolute of the normalized targets
            for i, s in enumerate(strata):
                self.assertAlmostEqual(cnt[i] / 10_000,
                                       s["normalized_share"], delta=0.03)
            # holdout tail disjoint from draw indices
            for i, s in enumerate(strata):
                m = order[:, 0] == i
                self.assertLess(int(order[m, 1].max()), s["avail_blocks"])
                self.assertEqual(s["blocks"] - s["holdout_start"],
                                 s["holdout_blocks"])

            out = os.path.join(d, "draw")
            meta = dp.write_outputs(out, order, strata, mp, 1273, 10_000, 256)
            # manifest + order + eval sets + csv all present and consistent
            man = json.load(open(os.path.join(out, "draw_manifest.json")))
            self.assertEqual(len(man["strata"]), 3)
            self.assertEqual(man["tokens"], 10_000 * 1024)
            o2 = np.load(os.path.join(out, "uniform_order.idx.npy"))
            self.assertTrue((o2 == order).all())
            ev = np.load(os.path.join(out, "eval_sets", "alpha.npy"))
            self.assertEqual(ev[0], strata[0]["holdout_start"])
            # every drawn alpha index is disjoint from its eval holdout
            self.assertFalse(np.intersect1d(ev, order[order[:, 0] == 0, 1]).size)
            # metadata csv parses and package column is empty (deviation)
            import gzip
            import csv as _csv
            with gzip.open(os.path.join(out, "blocks_meta.csv.gz"), "rt") as f:
                rows = list(_csv.DictReader(f))
            self.assertEqual(len(rows), 10_000)
            self.assertEqual(rows[0]["package"], "")
            self.assertEqual(int(rows[0]["tokens"]), 1024)
            self.assertEqual(meta["n_blocks"], 10_000)

    def test_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            mp = _synthetic_manifest(d)
            s1 = dp.load_strata(mp)
            s2 = dp.load_strata(mp)
            o1, _ = dp.make_draw(s1, 5_000, seed=1273)
            o2, _ = dp.make_draw(s2, 5_000, seed=1273)
            self.assertTrue((o1 == o2).all())
            o3, _ = dp.make_draw(dp.load_strata(mp), 5_000, seed=1274)
            self.assertFalse((o1 == o3).all())


if __name__ == "__main__":
    unittest.main()


class TestCurriculum(unittest.TestCase):
    def test_multiset_and_trend(self):
        import score_blocks as sb
        rng = np.random.RandomState(0)
        order = np.stack([rng.randint(0, 3, 600),
                          rng.randint(0, 50, 600)], axis=1).astype(np.int32)
        # stratum 0 easy (low CE), stratum 2 hard — CE correlated to block
        ce = (order[:, 0] * 2.0 + rng.rand(600) * 0.1).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            np.save(f"{d}/ce.npy", ce)
            np.save(f"{d}/uniform_order.idx.npy", order)
            man = dict(strata=[dict(name=f"s{i}") for i in range(3)])
            with open(f"{d}/draw_manifest.json", "w") as f:
                json.dump(man, f)
            out = f"{d}/cur.npy"
            sb.cmd_curriculum(d, f"{d}/ce.npy", 10, out, seed=1273)
            cur = np.load(out)
            self.assertEqual(len(cur), 600)
            # block multiset preserved exactly
            a = np.array(sorted(map(tuple, order.tolist())))
            b = np.array(sorted(map(tuple, cur.tolist())))
            self.assertTrue((a == b).all())
            # global CE trend is ascending (hard/noisy-last)
            self.assertLess(cur[:300, 0].mean(), cur[300:, 0].mean())
            # per-bucket mixture shares ~ production (1/3 each bucket)
            for bk in range(10):
                seg = cur[bk * 60:(bk + 1) * 60]
                for si in range(3):
                    self.assertAlmostEqual((seg[:, 0] == si).mean(),
                                           1 / 3, delta=0.12)
