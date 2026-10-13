#!/usr/bin/env python3
"""Focused integrity tests for the retained DEV counterfactual audit."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("rescore_finish_sensitivity.py")
REPORT = Path(__file__).with_name("sensitivity-report.json")


def load_audit_module():
    spec = importlib.util.spec_from_file_location("finish_rescore_sensitivity", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinishRescoreSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_audit_module()
        cls.audit = cls.module.audit()

    def test_all_retained_artifacts_and_selected_rows_are_pinned(self) -> None:
        self.assertEqual(len(self.audit["checkpoints"]), 9)
        self.assertEqual(
            [item["label"] for item in self.audit["checkpoints"]],
            [
                "SFT500",
                "SFT1000",
                "SFT2000",
                "SFT2500",
                "RL25",
                "RL50",
                "RL75",
                "RL100",
                "RL125",
            ],
        )
        for checkpoint in self.audit["checkpoints"]:
            self.assertEqual(checkpoint["artifact"]["result_count"], 75)
            self.assertEqual(checkpoint["denominators"]["all_cases"], 75)
            self.assertEqual(checkpoint["denominators"]["finish_cases"], 6)
            self.assertEqual(checkpoint["denominators"]["other_cases_unchanged"], 69)
            self.assertGreaterEqual(len(checkpoint["pin_receipts"]), 1)
            self.assertEqual(
                [case["id"] for case in checkpoint["cases"]],
                list(self.module.FINISH_IDS),
            )

    def test_only_counterfactual_target_changes_and_six_boundaries_are_fixed(self) -> None:
        for checkpoint in self.audit["checkpoints"]:
            counts = checkpoint["counts"]
            self.assertEqual(counts["old_finish_exact_region"], 0)
            self.assertEqual(counts["counterfactual_finish_exact_region"], 0)
            self.assertEqual(checkpoint["denominators"]["other_cases_unchanged"], 69)
            self.assertEqual(counts["counterfactual_delta_total"], 0)
            self.assertEqual(counts["old_total_exact_region"], counts["counterfactual_total_exact_region"])
            for case in checkpoint["cases"]:
                self.assertFalse(case["old_exact_region"])
                self.assertFalse(case["target_reconstruction_r_parse"])
                self.assertTrue(case["corrected_target_reconstruction_r_parse"])
                self.assertFalse(case["counterfactual_exact_region_after_one_brace"])
        self.assertTrue(self.audit["protocol_and_cap_policy"]["other_69_scores_unchanged"])

    def test_protocol_cap_and_prediction_reconstruction_are_reported_separately(self) -> None:
        for checkpoint in self.audit["checkpoints"]:
            counts = checkpoint["counts"]
            denominators = checkpoint["denominators"]
            self.assertEqual(denominators["protocol_consistency"], "6/6")
            self.assertEqual(
                int(denominators["protocol_valid"].split("/", 1)[0]),
                counts["protocol_valid_finish"],
            )
            self.assertLessEqual(
                counts["prediction_r_parse_after_one_brace"],
                counts["protocol_valid_finish"],
            )
            self.assertEqual(counts["prediction_r_parse_before_brace"], 0)
            self.assertEqual(counts["corrected_target_r_parse"], 6)
            self.assertEqual(
                sum(case["artifact_protocol_valid"] for case in checkpoint["cases"]),
                counts["protocol_valid_finish"],
            )
            self.assertEqual(
                sum(case["cap_hit"] for case in checkpoint["cases"]),
                counts["cap_hit_finish"],
            )

    def test_ordering_is_explicitly_insensitive(self) -> None:
        ordering = self.audit["ordering"]
        self.assertFalse(ordering["sensitive"])
        for item in ordering["old_vs_counterfactual"]:
            self.assertEqual(
                item["old_total_exact_region"],
                item["counterfactual_total_exact_region"],
            )
        self.assertEqual(
            [item["old_total_exact_region"] for item in ordering["old_vs_counterfactual"]],
            [45, 51, 46, 47, 50, 50, 50, 51, 51],
        )

    def test_materialized_report_matches_fresh_audit(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report, self.audit)


if __name__ == "__main__":
    unittest.main()
