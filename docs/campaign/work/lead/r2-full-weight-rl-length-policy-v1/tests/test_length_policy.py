import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_length_policy import LengthPolicy, LengthPolicyError, assess_row, audit_rows, require_row


def digest(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(prompt=100, completion=20):
    # Completion includes EOS; stored target_token_count excludes EOS.
    ids = [0] + [9] * (prompt - 1) + [10] * (completion - 1) + [1]
    return {"id": "r", "split": "train", "family": "finish_block", "input_ids": ids,
            "target_start": prompt, "prompt_token_count": prompt - 1,
            "target_token_count": completion - 1}


def policy(rows_hash="1" * 64, prompt=3072, completion=1024, context=4096):
    return LengthPolicy.from_mapping({
        "schema": "sepalith.rl11.full-weight-rl-length-policy.v1", "policy_id": "p",
        "input_rows_sha256": rows_hash, "model_context_tokens": context,
        "prompt_max_tokens": prompt, "completion_max_tokens": completion,
        "disposition": "candidate_pending_resource_admission"})


class TestLengthPolicy(unittest.TestCase):
    def test_complete_target_above_192_is_accepted_by_candidate(self):
        value = row(prompt=3000, completion=933)
        self.assertIs(require_row(value, policy()), value)
        self.assertTrue(assess_row(value, policy())["accepted_complete"])

    def test_legacy_rejects_without_mutating(self):
        value = row(prompt=100, completion=193); original = copy.deepcopy(value)
        with self.assertRaisesRegex(LengthPolicyError, "rejected without truncation"):
            require_row(value, policy(prompt=2048, completion=192, context=2240))
        self.assertEqual(value, original)

    def test_over_prompt_and_context_are_named(self):
        result = assess_row(row(prompt=3073, completion=1024), policy())
        self.assertEqual(result["reasons"], ["prompt_over_limit", "sequence_over_limit"])

    def test_malformed_and_truncated_rows_fail(self):
        for mutation in (
            lambda r: r.update(target_start=99),
            lambda r: r["input_ids"].pop(),
            lambda r: r["input_ids"].__setitem__(-1, 9),
            lambda r: r.update(split="dev"),
        ):
            value = row(); mutation(value)
            with self.assertRaises(LengthPolicyError): assess_row(value, policy())

    def test_policy_rejects_bool_and_impossible_envelope(self):
        raw = {"schema":"sepalith.rl11.full-weight-rl-length-policy.v1", "policy_id":"p",
               "input_rows_sha256":"1"*64, "model_context_tokens":4096,
               "prompt_max_tokens":3072, "completion_max_tokens":1024,
               "disposition":"candidate_pending_resource_admission"}
        for field, value in (("prompt_max_tokens", True), ("completion_max_tokens", 2000)):
            bad = dict(raw); bad[field] = value
            with self.assertRaises(LengthPolicyError): LengthPolicy.from_mapping(bad)

    def test_streaming_audit_binds_bytes_and_all_rejected_ids(self):
        with tempfile.TemporaryDirectory(dir="/mnt/e") as td:
            path = Path(td) / "rows.jsonl"
            values = [row(100, 20), dict(row(100, 193), id="long")]
            path.write_text("".join(json.dumps(v, separators=(",", ":"))+"\n" for v in values))
            p = policy(digest(path), prompt=2048, completion=192, context=2240)
            report = audit_rows(path, [p])
            got = report["policies"]["p"]
            self.assertEqual(got["accepted_complete_rows"], 1)
            self.assertEqual(got["rejected_without_truncation_rows"], 1)
            self.assertTrue(report["no_tokens_truncated"])
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(LengthPolicyError, "invalid JSON|hash mismatch"):
                audit_rows(path, [p])


if __name__ == "__main__": unittest.main()
