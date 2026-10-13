"""RL length policy (card C11). Ported from the RL-11 packet tests, plus loader integration."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.training.rl import campaign_rl_data
from sepalith.training.rl.length_policy import (
    LEGACY_POLICY,
    REVIEWED_POLICY,
    SCHEMA,
    LengthPolicy,
    LengthPolicyError,
    assess_row,
    audit_rows,
    require_row,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(prompt: int = 100, completion: int = 20) -> dict[str, object]:
    # Completion includes EOS; stored target_token_count excludes EOS.
    ids = [0] + [9] * (prompt - 1) + [10] * (completion - 1) + [1]
    return {"id": "r", "split": "train", "family": "finish_block", "input_ids": ids,
            "target_start": prompt, "prompt_token_count": prompt - 1,
            "target_token_count": completion - 1}


def policy(rows_hash: str = "1" * 64, prompt: int = 3072, completion: int = 1024, context: int = 4096) -> LengthPolicy:
    return LengthPolicy.from_mapping({
        "schema": SCHEMA, "policy_id": "p",
        "input_rows_sha256": rows_hash, "model_context_tokens": context,
        "prompt_max_tokens": prompt, "completion_max_tokens": completion,
        "disposition": "candidate_pending_resource_admission"})


class TestLengthPolicy(unittest.TestCase):
    def test_reviewed_policy_replaces_legacy_limits(self) -> None:
        self.assertEqual((REVIEWED_POLICY.prompt_max_tokens, REVIEWED_POLICY.completion_max_tokens,
                          REVIEWED_POLICY.model_context_tokens), (3072, 1024, 4096))
        self.assertEqual((campaign_rl_data.PROMPT_MAX_TOKENS, campaign_rl_data.COMPLETION_MAX_TOKENS,
                          campaign_rl_data.CONTEXT_MAX_TOKENS), (3072, 1024, 4096))
        self.assertEqual(LEGACY_POLICY.disposition, "audit_only")

    def test_complete_target_above_192_is_accepted_by_candidate(self) -> None:
        value = row(prompt=3000, completion=933)
        self.assertIs(require_row(value, policy()), value)
        self.assertTrue(assess_row(value, policy())["accepted_complete"])

    def test_legacy_rejects_without_mutating(self) -> None:
        value = row(prompt=100, completion=193)
        original = copy.deepcopy(value)
        with self.assertRaisesRegex(LengthPolicyError, "rejected without truncation"):
            require_row(value, policy(prompt=2048, completion=192, context=2240))
        self.assertEqual(value, original)

    def test_over_prompt_and_context_are_named(self) -> None:
        result = assess_row(row(prompt=3073, completion=1024), policy())
        self.assertEqual(result["reasons"], ["prompt_over_limit", "sequence_over_limit"])

    def test_malformed_and_truncated_rows_fail(self) -> None:
        ids = [0] + [9] * 99 + [10] * 19
        for value in (
            {**row(), "target_start": 99},
            {**row(), "input_ids": ids},
            {**row(), "input_ids": [*ids, 9]},
            {**row(), "split": "dev"},
        ):
            with self.assertRaises(LengthPolicyError):
                assess_row(value, policy())

    def test_policy_rejects_bool_and_impossible_envelope(self) -> None:
        raw = {"schema": SCHEMA, "policy_id": "p",
               "input_rows_sha256": "1" * 64, "model_context_tokens": 4096,
               "prompt_max_tokens": 3072, "completion_max_tokens": 1024,
               "disposition": "candidate_pending_resource_admission"}
        for field, value in (("prompt_max_tokens", True), ("completion_max_tokens", 2000)):
            bad = dict(raw)
            bad[field] = value
            with self.assertRaises(LengthPolicyError):
                LengthPolicy.from_mapping(bad)

    def test_streaming_audit_binds_bytes_and_all_rejected_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            values = [row(100, 20), dict(row(100, 193), id="long")]
            path.write_text("".join(json.dumps(v, separators=(",", ":")) + "\n" for v in values))
            legacy = policy(digest(path), prompt=2048, completion=192, context=2240)
            reviewed = dataclasses.replace(policy(digest(path)), policy_id="q")
            report = json.loads(json.dumps(audit_rows(path, [legacy, reviewed])))
            policies = report["policies"]
            self.assertEqual(policies["p"]["accepted_complete_rows"], 1)
            self.assertEqual(policies["p"]["rejected_without_truncation_rows"], 1)
            self.assertEqual(policies["q"]["accepted_complete_rows"], 2)
            self.assertTrue(report["no_tokens_truncated"])
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(LengthPolicyError, "invalid JSON|hash mismatch"):
                audit_rows(path, [legacy])

    def test_loader_binds_the_policy_to_its_census_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row()) + "\n")
            other = Path(directory) / "missing"
            arguments = (path, digest(path), other, "2" * 64, other, "3" * 64)
            with self.assertRaisesRegex(campaign_rl_data.RLDataError, "census the new rows"):
                campaign_rl_data.load_training_records(
                    *arguments, reward_buffer_manifest_path=other, reward_buffer_manifest_sha256="4" * 64)
            with self.assertRaisesRegex(campaign_rl_data.RLDataError, "audit-only"):
                campaign_rl_data.load_training_records(
                    *arguments, reward_buffer_manifest_path=other, reward_buffer_manifest_sha256="4" * 64,
                    length_policy=LEGACY_POLICY)


if __name__ == "__main__":
    unittest.main()
