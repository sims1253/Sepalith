"""C09 GGUF DEV evaluator: llama-server response checks, request shape and summary keys."""
import unittest

from sepalith.quant import gguf_dev as gd
from sepalith.quant.llama_tools import parse_kld_log


class ResponseTest(unittest.TestCase):
    def response(self, tokens, stop_type="eos", evaluated=10):
        return {"tokens": tokens, "stop_type": stop_type, "tokens_evaluated": evaluated}

    def test_eos_stops(self):
        self.assertEqual(gd.generated_ids(self.response([5, 6, 1]), 10, 8), [5, 6, 1])
        self.assertEqual(gd.generated_ids(self.response([5, 130073]), 10, 8), [5, 130073])

    def test_cap(self):
        self.assertEqual(gd.generated_ids(self.response([5] * 4, "limit"), 10, 4), [5] * 4)
        with self.assertRaises(gd.GenerationError):
            gd.generated_ids(self.response([5] * 3, "limit"), 10, 4)

    def test_rejections(self):
        for response in (self.response([5, 130074]),  # a stop id the PC evaluator would not stop on
                         self.response([5, 1, 6, 1]),  # continued past a stop id
                         self.response([5, 1], evaluated=11),  # prompt changed on the way
                         self.response([5, 1], "word"),
                         {"stop_type": "eos", "tokens_evaluated": 10}):
            with self.assertRaises(gd.GenerationError):
                gd.generated_ids(response, 10, 8)

    def test_request_is_greedy_and_uncached(self):
        body = gd.completion_request([0, 5, 6], 1024)
        self.assertEqual(body["prompt"], [0, 5, 6])
        self.assertEqual((body["n_predict"], body["temperature"], body["top_k"]), (1024, 0.0, 1))
        self.assertFalse(body["cache_prompt"])
        self.assertTrue(body["return_tokens"])
        self.assertEqual(body["stop"], [])


class SummaryTest(unittest.TestCase):
    def record(self, case_id, *, noop, valid, exact, suggestion, complete=True, cap=False):
        return {"id": case_id, "package_id": "p" + case_id, "expected_noop": noop, "protocol_valid": valid,
                "exact_region": exact, "predicted_noop": valid and not suggestion, "suggestion": suggestion,
                "cap_hit": cap, "response_complete": complete}

    def test_summary_keys_and_counts(self):
        results = [self.record("a", noop=False, valid=True, exact=True, suggestion=True),
                   self.record("b", noop=True, valid=True, exact=False, suggestion=True),
                   self.record("c", noop=True, valid=True, exact=True, suggestion=False),
                   self.record("d", noop=False, valid=False, exact=False, suggestion=False, complete=False, cap=True)]
        summary = gd.summarize(results, 1024, "sha", "/x/cases.json")
        self.assertEqual(set(summary), {"case_ids", "denominators", "counts", "complete_case_counts",
                                        "generation_max_new_tokens", "prompt_nll", "target_nll", "panel_sha256",
                                        "cases_path", "interpretation"})
        self.assertEqual(summary["counts"], {"protocol_valid": 3, "exact_region": 2, "predicted_noop": 1, "suggestion": 2,
                                             "cap_hit": 1, "strict_noop_correct": 1, "strict_noop_false_suggestions": 1,
                                             "edit_exact": 1})
        self.assertEqual(summary["denominators"]["incomplete_cases"], 1)
        self.assertIsNone(summary["prompt_nll"])


class KldLogTest(unittest.TestCase):
    LOG = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   3.123456 ±   0.012345
Mean PPL(base)                :   3.120000 ±   0.012000

====== KL divergence statistics ======
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   1.234567
99.9%   KLD:   0.900000
99.0%   KLD:   0.200000
Median  KLD:   0.004000

====== Token probability statistics ======
RMS Δp    :  2.345 ± 0.012 %
Same top p: 97.654 ± 0.045 %
"""

    def test_parse(self):
        stats = parse_kld_log(self.LOG)
        self.assertAlmostEqual(stats["mean_kld"], 0.012345)
        self.assertAlmostEqual(stats["mean_kld_uncertainty"], 0.000123)
        self.assertAlmostEqual(stats["same_top_p_pct"], 97.654)
        self.assertAlmostEqual(stats["p99_kld"], 0.2)
        with self.assertRaises(ValueError):
            parse_kld_log("nothing here")


if __name__ == "__main__":
    unittest.main()
