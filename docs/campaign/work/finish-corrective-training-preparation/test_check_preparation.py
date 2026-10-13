"""Meaningful CPU-only contract tests for the SFT-06 preparation packet."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_preparation import PreparationError, load_skeleton, validate_skeleton


class PreparationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recipe = load_skeleton()

    def test_template_has_overlay_evidence_and_stage_geometry(self) -> None:
        evidence = validate_skeleton(copy.deepcopy(self.recipe))
        self.assertEqual(evidence["effective_batch"], 16)
        self.assertEqual(evidence["draws"], 3200)
        self.assertEqual(evidence["overlay_eligible"], 4051)
        self.assertEqual(evidence["overlay_rejected"], 949)
        self.assertFalse(evidence["overlay_tokenized"])

    def test_launch_authorization_cannot_be_smuggled_into_template(self) -> None:
        candidate = copy.deepcopy(self.recipe)
        candidate["launch_authorized"] = True
        with self.assertRaisesRegex(PreparationError, "launch_authorized"):
            validate_skeleton(candidate)

    def test_rejected_overlay_rows_and_tokenization_gate_are_bound(self) -> None:
        candidate = copy.deepcopy(self.recipe)
        candidate["overlay_preparation"]["eligible_strict_v2_rows"] = 4289
        with self.assertRaisesRegex(PreparationError, "eligible_strict_v2_rows"):
            validate_skeleton(candidate)

        candidate = copy.deepcopy(self.recipe)
        candidate["overlay_preparation"]["tokenization_performed"] = True
        with self.assertRaisesRegex(PreparationError, "tokenization_performed"):
            validate_skeleton(candidate)

    def test_identity_and_milestone_resume_contract_cannot_drift(self) -> None:
        candidate = copy.deepcopy(self.recipe)
        candidate["parameters"]["gradient_accumulation"] = 2
        with self.assertRaisesRegex(PreparationError, "identity.schedule_matches_parameters.gradient_accumulation|effective_batch"):
            validate_skeleton(candidate)

        candidate = copy.deepcopy(self.recipe)
        candidate["milestone_plan"]["attempts"][1]["resume_from"] = None
        with self.assertRaisesRegex(PreparationError, "milestone_plan.attempts"):
            validate_skeleton(candidate)


if __name__ == "__main__":
    unittest.main()
