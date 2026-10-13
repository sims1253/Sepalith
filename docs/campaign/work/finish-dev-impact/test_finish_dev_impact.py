#!/usr/bin/env python3
"""Bounded regression checks for the six-row DEV finish audit."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "dat07_finish_impact_test_module", HERE / "audit_finish_dev_impact.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinishDevImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = MODULE.audit()

    def test_exact_panel_and_six_finish_denominator(self) -> None:
        self.assertEqual(self.report["panel_sha256"], MODULE.EXPECTED_PANEL_SHA256)
        self.assertEqual(self.report["selected_panel_lines"], [70, 71, 72, 73, 74, 75])
        self.assertEqual(self.report["selected_count"], 6)
        self.assertEqual(self.report["finish_operation_counts"], {"replace": 6, "no_op": 0})
        self.assertEqual(len({case["id"] for case in self.report["cases"]}), 6)

    def test_utf16_splices_and_source_target_identities_are_exact(self) -> None:
        for case in self.report["cases"]:
            self.assertEqual(
                case["geometry"]["before_sha256"],
                case["source_provenance"]["pre_edit_content_sha256"],
            )
            self.assertEqual(len(case["geometry"]["selected_text_sha256"]), 64)
            self.assertNotEqual(case["geometry"]["before_sha256"], case["geometry"]["after_sha256"])
            self.assertEqual(case["source_provenance"]["target_convention"], "suffix")
            self.assertEqual(case["source_provenance"]["target_framing"]["finish_target_bytes_preserved"], True)
            self.assertEqual(case["source_provenance"]["outer_closing_brace_in_label"], False)

    def test_every_finish_post_edit_fragment_needs_one_outer_brace(self) -> None:
        for case in self.report["cases"]:
            shape = case["fragment_shape"]
            parser = case["parser"]
            self.assertEqual(shape["visible_suffix_lines"], 0)
            self.assertFalse(shape["visible_suffix_has_standalone_closing_brace"])
            self.assertFalse(shape["target_has_outer_closing_brace"])
            self.assertFalse(parser["post_edit_ok"])
            self.assertTrue(parser["post_edit_plus_one_outer_brace_ok"])

    def test_fragment_parse_failure_is_not_called_source_failure(self) -> None:
        self.assertTrue(
            self.report["interpretation"]["fragment_limit"].startswith(
                "The panel pre_edit_document is source-derived simulated"
            )
        )
        self.assertIn("does not claim every DEV row", self.report["interpretation"]["truth_scope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
