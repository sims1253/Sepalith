"""Numeric and compatibility checks for the extracted paired scorer."""
from pathlib import Path
import subprocess
import sys
import unittest

from sepalith.evaluation import audit_pair, mcnemar_exact, paired_bootstrap_ci


class EvaluationTests(unittest.TestCase):
    def test_exact_tail_and_symmetry(self):
        self.assertEqual(mcnemar_exact(0, 10), 2 / 1024)
        self.assertEqual(mcnemar_exact(3, 10), 2 * (1 + 13 + 78 + 286) / 8192)
        self.assertEqual(mcnemar_exact(10, 3), mcnemar_exact(3, 10))
        self.assertEqual(mcnemar_exact(0, 0), 1)

    def test_large_counts_do_not_overflow(self):
        self.assertEqual(mcnemar_exact(600, 600), 1)
        self.assertTrue(0 <= mcnemar_exact(10, 1200) <= 1)

    def test_invalid_numeric_inputs_fail_clearly(self):
        for counts in ((-1, 0), (1.5, 2), (True, 1)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                mcnemar_exact(*counts)
        for kwargs in ({"n_boot": 0}, {"n_boot": True}, {"alpha": 0},
                       {"alpha": 1}, {"alpha": float("nan")}, {"alpha": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paired_bootstrap_ci([0, 1], **kwargs)
        for deltas in ([], [float("nan")], [float("inf")], ["1"]):
            with self.subTest(deltas=deltas), self.assertRaises(ValueError):
                paired_bootstrap_ci(deltas)

    def test_confidence_level_applies_to_interval_and_test(self):
        a, b = [1] * 9 + [0], [0] * 9 + [1]
        verdict = audit_pair("alpha", a, b, alpha=0.5)
        deltas = [float(x-y) for x, y in zip(a, b)]
        self.assertEqual(verdict.ci, paired_bootstrap_ci(deltas, alpha=0.5))
        self.assertNotEqual(verdict.ci, paired_bootstrap_ci(deltas, alpha=0.05))
        self.assertEqual(verdict.verdict, "WINNER-A")
        self.assertEqual(audit_pair("reversed", a, b, higher_better=False).verdict, "WINNER-B")

    def test_historical_checks_work_through_compatibility_entrypoint(self):
        root = Path(__file__).resolve().parents[3]
        path = root / "experiments/eval/test_paired_significance.py"
        result = subprocess.run([sys.executable, str(path)], cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all green", result.stdout)
