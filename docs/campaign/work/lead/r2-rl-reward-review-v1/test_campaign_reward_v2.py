#!/usr/bin/env python3
"""TRAIN-only regression tests for the prepared reward-v2 core."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from campaign_reward_v2 import BufferEvidence, apply_region, score_candidate, severe_repetition, sha_text


ROOT = Path(__file__).resolve().parent
FIXTURES = json.loads((ROOT / "fixtures/train-fixtures.json").read_text())["fixtures"]
R_PARSER_IDENTITY = "R-4.6.1-base-parse-file-keep.source.FALSE"
FROZEN_SOURCE = ROOT.parent / "r2-step500-rl-gate-v2/source"
sys.path.insert(0, str(FROZEN_SOURCE / "experiments/training"))
sys.path.insert(0, str(FROZEN_SOURCE / "packages/sepalith/src"))
from campaign_rl_train import CampaignPRM03Reward as FrozenCampaignPRM03Reward  # noqa: E402


def parse_only(text: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".R", encoding="utf-8") as source:
        source.write(text)
        source.flush()
        result = subprocess.run(
            ["Rscript", "--vanilla", "-e",
             "parse(file=commandArgs(trailingOnly=TRUE)[1], keep.source=FALSE)", source.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
    return result.returncode == 0


def evidence(fixture: dict) -> BufferEvidence:
    context = fixture["context"]
    baseline = fixture["baseline_text"]
    if fixture["buffer_mode"] == "unverified":
        return BufferEvidence("unverified", None,
                              context["replacement_range"]["content_sha256"], None, None, None)
    gold = apply_region(baseline, context["replacement_range"], fixture["target_body_text"],
                        context["document_eol"])
    return BufferEvidence(
        fixture["buffer_mode"], baseline, sha_text(baseline), parse_only(baseline),
        parse_only(gold), R_PARSER_IDENTITY,
    )


def score(fixture: dict, *, operation: str | None, body: list[str], valid: bool = True,
          failure: str | None = None):
    context = fixture["context"]
    buffer = evidence(fixture)
    return score_candidate(
        target_operation=fixture["target_operation"], target_body_text=fixture["target_body_text"],
        region_old=context["region_old"], protocol_valid=valid, protocol_failure=failure,
        operation=operation, body_lines=body, replacement_range=context["replacement_range"],
        document_eol=context["document_eol"], buffer=buffer,
        parse_probe=parse_only if buffer.mode != "unverified" else None,
    )


class RewardV2TrainFixtureTest(unittest.TestCase):
    def test_frozen_reward_collapses_distinct_defects_to_zero(self):
        fixture = FIXTURES["no_op"]
        raw = {100: "x <- 1L\n>>>>>>> UPDATED"}
        reward = FrozenCampaignPRM03Reward(decoder=lambda ids, **_: raw[ids[0]])
        false_edit, false_record = reward.score_one(
            fixture["context"], fixture["target_operation"], fixture["target_body_text"], [100, 1],
            row_id=fixture["id"], family=fixture["family"], package_id=fixture["package_id"],
        )
        missing, missing_record = reward.score_one(
            fixture["context"], fixture["target_operation"], fixture["target_body_text"], [100],
            row_id=fixture["id"], family=fixture["family"], package_id=fixture["package_id"],
        )
        self.assertEqual((false_edit, missing), (0.0, 0.0))
        self.assertTrue(false_record["protocol_valid"])
        self.assertEqual(missing_record["failure"], "missing_canonical_eos")

    def test_fixtures_are_train_only_with_source_ids(self):
        self.assertEqual(set(FIXTURES), {"complete_edit", "finish_prefix", "no_op", "roxygen_window"})
        for fixture in FIXTURES.values():
            self.assertEqual(fixture["split"], "train")
            self.assertEqual(fixture["source_ref"]["split"], "train_group")
            self.assertEqual(fixture["id"], fixture["source_ref"]["row_id"])

    def test_exact_train_edit_scores_highest(self):
        fixture = FIXTURES["complete_edit"]
        value, record = score(fixture, operation="replace", body=fixture["target_body_text"].split("\n"))
        self.assertEqual(value, 1.2)
        self.assertTrue(record["exact_region"])
        self.assertEqual(record["semantic_correctness"], "not_measured_beyond_exact_region")

    def test_false_edit_on_train_noop_is_explicitly_negative(self):
        value, record = score(FIXTURES["no_op"], operation="replace", body=["x <- 1L"])
        self.assertEqual(value, -1.0)
        self.assertTrue(record["false_noop_edit"])
        self.assertEqual(record["outcome"], "false_noop_edit")

    def test_missing_termination_and_cap_are_negative(self):
        fixture = FIXTURES["complete_edit"]
        for failure in ("missing_canonical_eos", "missing_exact_terminal", "invalid_generation_tokens"):
            with self.subTest(failure=failure):
                value, record = score(fixture, operation=None, body=[], valid=False, failure=failure)
                self.assertEqual(value, -1.0)
                self.assertEqual(record["outcome"], "invalid_or_unterminated")

    def test_severe_repetition_is_below_wrong_parseable_content(self):
        fixture = FIXTURES["complete_edit"]
        repeated = ["x <- x + 1L", "y <- y + 1L"] * 3
        value, record = score(fixture, operation="replace", body=repeated)
        self.assertEqual(value, -0.75)
        self.assertTrue(record["repetition"]["detected"])
        wrong, wrong_record = score(fixture, operation="replace", body=["  xpos <- 0"])
        self.assertEqual(wrong, 0.0)
        self.assertEqual(wrong_record["candidate_parse"], "passed")

    def test_two_similar_lines_do_not_trigger_runaway_heuristic(self):
        self.assertFalse(severe_repetition(["x <- 1L", "x <- 1L"])["detected"])

    def test_parse_broken_applied_full_buffer_is_negative(self):
        value, record = score(FIXTURES["complete_edit"], operation="replace", body=["if ("])
        self.assertEqual(value, -0.5)
        self.assertEqual(record["candidate_parse"], "failed")

    def test_incomplete_finish_baseline_is_not_blindly_penalized(self):
        fixture = FIXTURES["finish_prefix"]
        prepared = evidence(fixture)
        self.assertFalse(prepared.baseline_parse_ok)
        self.assertTrue(prepared.gold_applied_parse_ok)
        exact, exact_record = score(fixture, operation="replace", body=fixture["target_body_text"].split("\n"))
        broken, broken_record = score(fixture, operation="replace", body=["if ("])
        self.assertEqual((exact, exact_record["outcome"]), (1.2, "exact"))
        self.assertEqual((broken, broken_record["outcome"]), (-0.5, "applied_syntax_invalid"))

    def test_unverified_source_window_gets_no_parse_claim(self):
        value, record = score(FIXTURES["roxygen_window"], operation="replace", body=["#' Alternate title"])
        self.assertEqual(value, 0.0)
        self.assertEqual(record["candidate_parse"], "unavailable_unverified_buffer")

    def test_order_is_exact_then_wrong_then_parse_broken_then_repetition_then_invalid(self):
        fixture = FIXTURES["complete_edit"]
        values = [
            score(fixture, operation="replace", body=fixture["target_body_text"].split("\n"))[0],
            score(fixture, operation="replace", body=["  xpos <- 0"])[0],
            score(fixture, operation="replace", body=["if ("])[0],
            score(fixture, operation="replace", body=["x <- 1", "y <- 2"] * 3)[0],
            score(fixture, operation=None, body=[], valid=False, failure="missing_canonical_eos")[0],
        ]
        self.assertEqual(values, [1.2, 0.0, -0.5, -0.75, -1.0])

    def test_scores_are_deterministic(self):
        fixture = FIXTURES["complete_edit"]
        first = score(fixture, operation="replace", body=["  xpos <- 0"])
        second = score(fixture, operation="replace", body=["  xpos <- 0"])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
