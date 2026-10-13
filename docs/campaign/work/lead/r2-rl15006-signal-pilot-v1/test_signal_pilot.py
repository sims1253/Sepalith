#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "source" / "experiments" / "training"))
import signal_pilot_harness as pilot
from campaign_r_parse_probe import EXPECTED_BUFFER_PARSER_IDENTITY, ParseProbeError, RParseOnlyProbe


class Result:
    def __init__(self, code: int, stdout: str = "") -> None:
        self.returncode = code
        self.stdout = stdout
        self.stderr = ""


class SequenceRunner:
    def __init__(self, parse_code: int, version: str = EXPECTED_BUFFER_PARSER_IDENTITY["r_version"]) -> None:
        self.parse_code = parse_code
        self.version = version
        self.calls = 0

    def __call__(self, command, **kwargs):
        self.calls += 1
        if "-e" in command:
            return Result(0, self.version)
        return Result(self.parse_code)


def generation(group: dict, index: int) -> dict:
    ids = [100 + index, 1]
    return {
        "row_id": group["row_id"], "group_index": group["group_index"],
        "candidate_index": index, "generated_ids": ids,
        "generated_token_count": len(ids),
        "generated_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest(),
        "cap_hit": False, "terminal_reason": "canonical_eos",
    }


def reward(group: dict, index: int) -> dict:
    exact = index == 0
    unverified = group["syntax_evidence_mode"] == "unverified"
    return {
        "row_id": group["row_id"], "group_index": group["group_index"],
        "candidate_index": index, "reward": 1.2 if exact else 0.0,
        "protocol_valid": True, "protocol_failure": None, "operation": "replace",
        "exact_region": exact, "false_noop_edit": False,
        "repetition": {"detected": False},
        "parser_infrastructure_failure": False,
        "candidate_parse": "unavailable_unverified_buffer" if unverified else ("not_checked" if exact else "passed"),
        "syntax_evidence_mode": group["syntax_evidence_mode"],
    }


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec, cls.spec_sha, cls.groups = pilot.load_spec(HERE / "pilot-spec.json")

    def test_exact_pool_and_strata_contract(self) -> None:
        self.assertEqual(len(self.groups), 112)
        self.assertEqual(len({row["row_id"] for row in self.groups}), 112)
        self.assertEqual(len({row["family"] for row in self.groups}), 7)
        self.assertTrue(any(row["target_band"] == "long_gt_192" for row in self.groups))
        self.assertTrue(any(row["syntax_evidence_mode"] == "unverified" for row in self.groups))
        self.assertTrue(self.spec["pool"]["eventual_rl_pool_preserved"])

    def test_unbound_model_refuses_runnable_path(self) -> None:
        with self.assertRaisesRegex(pilot.PilotError, "model snapshot"):
            pilot.require_runnable(self.spec)

    def test_atomic_commit_resume_and_summary(self) -> None:
        group = self.groups[0]
        model = {"manifest_path": "/frozen/model.json", "manifest_sha256": "a" * 64,
                 "merged_weights_sha256": "b" * 64, "tokenizer_json_sha256": "c" * 64}
        generations = [generation(group, index) for index in range(4)]
        rewards = [reward(group, index) for index in range(4)]
        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            pilot.commit_group(output, spec_sha256=self.spec_sha, model_identity=model,
                               group=group, generations=generations, rewards=rewards)
            committed = pilot.validate_committed(output, spec_sha256=self.spec_sha,
                                                 model_identity=model, groups=self.groups)
            self.assertEqual(set(committed), {0})
            summary = pilot.summarize(committed, self.groups)
            self.assertEqual(summary["denominators"]["raw_outputs"], 4)
            self.assertEqual(summary["denominators"]["nonzero_variance_groups"], 1)
            self.assertEqual(summary["denominators"]["finish_candidates"], 4)
            self.assertEqual(summary["denominators"]["finish_valid_nonexact_syntax_passed"], 3)
            with self.assertRaisesRegex(pilot.PilotError, "already committed"):
                pilot.commit_group(output, spec_sha256=self.spec_sha, model_identity=model,
                                   group=group, generations=generations, rewards=rewards)

    def test_attempt_directory_is_not_a_committed_group(self) -> None:
        model = {"manifest_path": "/frozen/model.json", "manifest_sha256": "a" * 64,
                 "merged_weights_sha256": "b" * 64, "tokenizer_json_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as value:
            root = Path(value) / "groups"
            (root / ".attempt-000000-crash").mkdir(parents=True)
            committed = pilot.validate_committed(Path(value), spec_sha256=self.spec_sha,
                                                 model_identity=model, groups=self.groups)
            self.assertEqual(committed, {})

    def test_unverified_row_cannot_claim_parse(self) -> None:
        group = next(row for row in self.groups if row["syntax_evidence_mode"] == "unverified")
        generations = [generation(group, index) for index in range(4)]
        rewards = [reward(group, index) for index in range(4)]
        rewards[1]["candidate_parse"] = "passed"
        with self.assertRaisesRegex(pilot.PilotError, "unverified syntax"):
            pilot._validate_candidate_records(group, generations, rewards)

    def test_corrupt_generated_ids_hash_rejected(self) -> None:
        group = self.groups[0]
        generations = [generation(group, index) for index in range(4)]
        rewards = [reward(group, index) for index in range(4)]
        generations[2]["generated_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(pilot.PilotError, "generated ID hash"):
            pilot._validate_candidate_records(group, generations, rewards)

    def test_parser_exit_codes_are_three_way(self) -> None:
        harness = HERE / "source" / "experiments" / "training" / "reward_parse_only.R"
        harness_sha = pilot.sha256(harness)
        self.assertTrue(RParseOnlyProbe(harness, harness_sha, runner=SequenceRunner(0))("x <- 1\n"))
        self.assertFalse(RParseOnlyProbe(harness, harness_sha, runner=SequenceRunner(1))("x <-\n"))
        with self.assertRaisesRegex(ParseProbeError, "infrastructure_exit:2"):
            RParseOnlyProbe(harness, harness_sha, runner=SequenceRunner(2))("x <- 1\n")
        with self.assertRaisesRegex(ParseProbeError, "signal:9"):
            RParseOnlyProbe(harness, harness_sha, runner=SequenceRunner(-9))("x <- 1\n")

    def test_parser_identity_is_exact(self) -> None:
        harness = HERE / "source" / "experiments" / "training" / "reward_parse_only.R"
        probe = RParseOnlyProbe(harness, pilot.sha256(harness), runner=SequenceRunner(0))
        probe.assert_buffer_identity(EXPECTED_BUFFER_PARSER_IDENTITY)
        wrong = dict(EXPECTED_BUFFER_PARSER_IDENTITY, r_version="R version unknown")
        with self.assertRaisesRegex(ParseProbeError, "row_parser_identity_mismatch"):
            probe.assert_buffer_identity(wrong)

    def test_actual_pinned_r_parse_only(self) -> None:
        harness = HERE / "source" / "experiments" / "training" / "reward_parse_only.R"
        probe = RParseOnlyProbe(harness, pilot.sha256(harness))
        self.assertTrue(probe("x <- 1\n"))
        self.assertFalse(probe("x <-\n"))

    def test_parser_infrastructure_failure_is_separate_and_group_uncommitted(self) -> None:
        group = self.groups[0]
        model = {"manifest_path": "/frozen/model.json", "manifest_sha256": "a" * 64,
                 "merged_weights_sha256": "b" * 64, "tokenizer_json_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            path = pilot.record_infrastructure_failure(
                output, spec_sha256=self.spec_sha, model_identity=model, group=group,
                candidate_index=2, error=ParseProbeError("parse_probe_infrastructure_exit:2"),
            )
            record = json.loads(path.read_text())
            self.assertFalse(record["reward_emitted"])
            self.assertFalse(record["group_committed"])
            self.assertEqual(pilot.validate_committed(
                output, spec_sha256=self.spec_sha, model_identity=model, groups=self.groups,
            ), {})


if __name__ == "__main__":
    unittest.main()
