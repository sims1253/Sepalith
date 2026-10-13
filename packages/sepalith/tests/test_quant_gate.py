"""C09 quant gate statistics and checks (sepalith.quant.gate)."""
import math
import unittest

from sepalith.quant import gate


def case(case_id, exact, *, noop=False, valid=True, suggestion=None, ids=(5, 1), cap=False):
    suggestion = (valid and not noop and True) if suggestion is None else suggestion
    return {"id": case_id, "exact_region": exact, "expected_noop": noop, "protocol_valid": valid,
            "suggestion": suggestion, "predicted_noop": valid and not suggestion, "cap_hit": cap,
            "generated_ids": list(ids)}


class StatisticsTest(unittest.TestCase):
    def test_binomial_cdf(self):
        self.assertAlmostEqual(gate.binomial_cdf(2, 10), 56 / 1024)
        self.assertEqual(gate.binomial_cdf(-1, 10), 0.0)
        self.assertEqual(gate.binomial_cdf(10, 10), 1.0)

    def test_mcnemar_exact(self):
        self.assertEqual(gate.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(gate.mcnemar_exact(5, 0), 2 / 32)
        self.assertAlmostEqual(gate.mcnemar_exact(6, 0), 2 / 64)
        self.assertAlmostEqual(gate.mcnemar_exact(8, 2), 2 * 56 / 1024)
        self.assertEqual(gate.mcnemar_exact(2, 8), gate.mcnemar_exact(8, 2))
        self.assertEqual(gate.mcnemar_exact(3, 3), 1.0)
        with self.assertRaises(ValueError):
            gate.mcnemar_exact(-1, 2)

    def test_net_loss_upper_bound(self):
        # Bonett-Price: no discordant pairs still leaves about 0.92 pp on 250 cases.
        self.assertAlmostEqual(gate.net_loss_upper_bound_pp(0, 0, 250), 100 * 1.6448536269514722 * math.sqrt(2 / 252) / 252 ** 0.5, places=9)
        self.assertAlmostEqual(gate.net_loss_upper_bound_pp(0, 0, 250), 0.9232, places=3)
        self.assertAlmostEqual(gate.net_loss_upper_bound_pp(10, 2, 250), 5.595, places=2)
        self.assertLess(gate.net_loss_upper_bound_pp(2, 10, 250), 0)
        with self.assertRaises(ValueError):
            gate.net_loss_upper_bound_pp(200, 100, 250)

    def test_tiers(self):
        self.assertEqual(gate.tier_of("Q8_0"), "q8")
        self.assertEqual(gate.tier_of("Q5_K_M"), "q6_q5")
        self.assertEqual(gate.tier_of("Q4_K_M"), "q4")
        with self.assertRaises(ValueError):
            gate.tier_of("Q2_K")


class PairedTest(unittest.TestCase):
    def setUp(self):
        self.ref = [case("a", True), case("b", True), case("c", False), case("n1", True, noop=True, suggestion=False),
                    case("n2", True, noop=True, suggestion=False)]
        self.cand = [case("a", True), case("b", False, ids=(6, 1)), case("c", True, ids=(7, 1)),
                     case("n1", False, noop=True, suggestion=True, ids=(9, 1)),
                     case("n2", False, noop=True, valid=False, suggestion=False, ids=(9, 9))]

    def test_paired_metrics(self):
        m = gate.paired_metrics(self.ref, self.cand)
        self.assertEqual(m["cases"], 5)
        self.assertEqual((m["exact"]["lost"], m["exact"]["gained"]), (3, 1))
        self.assertAlmostEqual(m["exact"]["delta_pp"], -40.0)
        self.assertEqual((m["nofp"]["reference"], m["nofp"]["candidate"]), (0, 1))
        self.assertAlmostEqual(m["nofp"]["rise_pp"], 20.0)
        self.assertAlmostEqual(m["valid"]["drop_pp"], 20.0)
        self.assertEqual(m["token_identity"]["identical"], 1)
        self.assertEqual(m["lost_case_ids"], ["b", "n1", "n2"])
        self.assertEqual(m["gained_case_ids"], ["c"])

    def test_ids_must_match(self):
        with self.assertRaises(ValueError):
            gate.paired_metrics(self.ref, self.cand[:-1])

    def test_identical_run_passes_every_tier(self):
        m = gate.paired_metrics(self.ref, self.ref)
        kld = {"mean_kld": 0.001, "same_top_p_pct": 99.9}
        for tier in gate.THRESHOLDS:
            checks = gate.gate_checks(tier, m, kld)
            self.assertEqual(gate.verdict(checks), gate.PASS, tier)
        self.assertEqual(gate.gate_checks("q4", m, kld)["token_identity_pct"]["outcome"], gate.RECORDED)

    def test_failures_and_missing(self):
        m = gate.paired_metrics(self.ref, self.cand)
        checks = gate.gate_checks("q8", m, None)
        self.assertEqual(checks["exact_delta_pp"]["outcome"], gate.FAIL)
        self.assertEqual(checks["mean_kld"]["outcome"], gate.NOT_MEASURED)
        self.assertEqual(gate.verdict(checks), gate.FAIL)
        clean = gate.gate_checks("q8", gate.paired_metrics(self.ref, self.ref), None)
        self.assertEqual(gate.verdict(clean), "incomplete")
        self.assertEqual(gate.verdict(gate.gate_checks("q8", None, None)), "incomplete")

    def test_kld_thresholds(self):
        m = gate.paired_metrics(self.ref, self.ref)
        self.assertEqual(gate.gate_checks("q8", m, {"mean_kld": 0.0021, "same_top_p_pct": 99.6})["mean_kld"]["outcome"], gate.FAIL)
        self.assertEqual(gate.gate_checks("q4", m, {"mean_kld": 0.03, "same_top_p_pct": 96.9})["top1_agreement_pct"]["outcome"], gate.RECORDED)
        self.assertEqual(gate.gate_checks("q8", m, {"mean_kld": 0.001, "same_top_p_pct": 98.6})["top1_agreement_pct"]["outcome"], gate.RECORDED)
        self.assertEqual(gate.gate_checks("q6_q5", m, {"mean_kld": 0.01, "same_top_p_pct": 98.5})["mean_kld"]["outcome"], gate.PASS)

    def test_mcnemar_fails_only_on_significant_loss(self):
        n = 250
        ref = [case(f"c{i}", i < 20) for i in range(n)]
        worse = [case(f"c{i}", False) for i in range(n)]
        better = [case(f"c{i}", True) for i in range(n)]
        self.assertEqual(gate.gate_checks("q4", gate.paired_metrics(ref, worse), None)["mcnemar"]["outcome"], gate.FAIL)
        improving = gate.gate_checks("q4", gate.paired_metrics(ref, better), None)["mcnemar"]
        self.assertLess(improving["value"], 0.05)
        self.assertEqual(improving["outcome"], gate.PASS)

    def test_ship_bound(self):
        n = 250
        ref = [case(f"c{i}", True) for i in range(n)]
        self.assertEqual(gate.ship_check(gate.paired_metrics(ref, ref))["outcome"], gate.PASS)
        cand = [case(f"c{i}", i >= 5) for i in range(n)]
        bound = gate.ship_check(gate.paired_metrics(ref, cand))
        self.assertGreater(bound["value"], 3.0)
        self.assertEqual(bound["outcome"], gate.FAIL)

    def test_dev75_parity(self):
        parity = gate.dev75_parity(self.ref, self.cand)
        self.assertEqual(parity["difference"]["nofp"], 1)
        self.assertEqual(parity["difference"]["valid"], -1)
        self.assertTrue(parity["within_tolerance"])
        self.assertFalse(gate.dev75_parity(self.ref, self.cand, tolerance=0)["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
