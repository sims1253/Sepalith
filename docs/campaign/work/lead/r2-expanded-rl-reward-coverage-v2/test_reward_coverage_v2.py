from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "source/experiments/training"
PKG = ROOT / "source/packages/sepalith/src"
sys.path[:0] = [str(TRAIN), str(PKG)]

from campaign_r_parse_probe import RParseOnlyProbe, sha_file
from campaign_rl_buffer import BindingError, HELD_CONTRADICTORY_IDS, RewardBufferIndex
from campaign_rl_train import CampaignPRM03Reward, EOS_ID
from campaign_reward_v2 import apply_region
from sepalith.campaign_protocol import Cursor, Position, PromptContext, ReplacementRange, SCHEMA_VERSION


class CharEncoder:
    def __init__(self):
        self.forward = {}
        self.reverse = {}
        self.next_id = 200
    def encode(self, text):
        result = []
        for char in text:
            if char not in self.forward:
                self.forward[char] = self.next_id
                self.reverse[self.next_id] = char
                self.next_id += 1
            result.append(self.forward[char])
        return result
    def decode(self, ids, **_):
        return "".join(self.reverse[int(token)] for token in ids)


class FakeIndex:
    def baseline_text(self, envelope):
        return envelope["_baseline_text"]


class CountingProbe:
    def __init__(self, answer=True):
        self.answer = answer
        self.inputs = []
    def __call__(self, text):
        self.inputs.append(text)
        return self.answer


def context_for(baseline="old\n", region="old", start=(0, 0), end=(0, 3)):
    digest = hashlib.sha256(baseline.encode()).hexdigest()
    return PromptContext(
        path="src/test.R", prefix=(), selected_references=(), history=(), diagnostics=(), retrieval=(),
        scope_mode="outline", scope_lines=(), suffix_lines=(), region_old=(region,),
        cursor=Cursor(0, 0, 0),
        replacement_range=ReplacementRange(
            uri="file:///workspace/test.R", document_version=0, content_sha256=digest,
            start=Position(*start), end=Position(*end),
        ),
        document_eol="lf", schema_version=SCHEMA_VERSION,
    )


def envelope(row_id, context, mode="complete_document", baseline="old\n"):
    digest = context.replacement_range.content_sha256
    if mode == "unverified":
        return {"row_id": row_id, "context_sha256": digest, "syntax_evidence_mode": "unverified",
                "syntax_available": False, "repair_reason": "pinned_snapshot_unavailable"}
    return {
        "row_id": row_id, "context_sha256": digest, "syntax_evidence_mode": mode,
        "syntax_available": True, "repair_reason": None, "baseline_sha256": digest,
        "baseline_parse_ok": True, "gold_applied_parse_ok": True,
        "diagnostic_suffix": "\n}" if mode == "framed_fragment" else "",
        "framed_projection_parse_ok": True if mode == "framed_fragment" else None,
        "parser_identity": {"operation": "base::parse", "generated_r_executed": False},
        "_baseline_text": baseline,
    }


class RewardCoverageV2Test(unittest.TestCase):
    def setUp(self):
        self.encoder = CharEncoder()
        self.probe = CountingProbe(True)
        self.reward = CampaignPRM03Reward(
            decoder=self.encoder.decode, reward_buffer_index=FakeIndex(), parse_probe=self.probe,
        )
        self.context = context_for()
        self.env = envelope("row", self.context)
    def generated(self, text, terminal=EOS_ID):
        return self.encoder.encode(text) + [terminal]
    def score(self, text, *, operation="replace", target="new", env=None, terminal=EOS_ID, context=None):
        return self.reward.score_one(
            context or self.context, operation, target, self.generated(text, terminal), env or self.env,
            row_id="row", family="fixture", package_id="train-only",
        )
    def test_exact_gold_is_only_positive_semantic_credit(self):
        value, record = self.score("new\n>>>>>>> UPDATED")
        self.assertEqual(value, 1.2)
        self.assertEqual(record["outcome"], "exact")
        self.assertEqual(self.probe.inputs, [])
    def test_wrong_parseable_has_zero_semantic_credit(self):
        value, record = self.score("wrong\n>>>>>>> UPDATED")
        self.assertEqual((value, record["candidate_parse"], record["outcome"]),
                         (0.0, "passed", "wrong_content_no_semantic_credit"))
        self.assertEqual(len(self.probe.inputs), 1)
    def test_unavailable_does_not_call_parse_but_keeps_zero_wrong_reward(self):
        self.probe.inputs.clear()
        value, record = self.score("wrong\n>>>>>>> UPDATED", env=envelope("row", self.context, "unverified"))
        self.assertEqual((value, record["candidate_parse"], record["syntax_evidence_mode"]),
                         (0.0, "unavailable_unverified_buffer", "unverified"))
        self.assertEqual(self.probe.inputs, [])
    def test_framed_candidate_uses_same_prefix_and_suffix_as_gold_scope(self):
        baseline = "f <- function() {\n  old\n"
        context = context_for(baseline, "  old", (1, 0), (1, 5))
        env = envelope("row", context, "framed_fragment", baseline)
        value, record = self.score("  wrong\n>>>>>>> UPDATED", context=context, env=env)
        expected = apply_region(baseline, context.replacement_range.to_dict(), "  wrong", "lf") + "\n}"
        self.assertEqual(value, 0.0)
        self.assertEqual(record["parse_scope"], "framed_fragment")
        self.assertEqual(self.probe.inputs[-1], expected)
    def test_false_noop_and_invalid_eos_are_distinct_negative_outcomes(self):
        value, record = self.score("changed\n>>>>>>> UPDATED", operation="no_op", target="[NO_EDIT]")
        self.assertEqual((value, record["outcome"]), (-1.0, "false_noop_edit"))
        self.probe.inputs.clear()
        value, record = self.score("new\n>>>>>>> UPDATED", terminal=999)
        self.assertEqual((value, record["outcome"], record["failure"]),
                         (-1.0, "invalid_or_unterminated", "missing_canonical_eos"))
        self.assertEqual(self.probe.inputs, [])
    def test_repetition_precedes_syntax_and_is_penalized(self):
        value, record = self.score("x\ny\nx\ny\nx\ny\n>>>>>>> UPDATED")
        self.assertEqual((value, record["outcome"]), (-0.75, "severe_repetition"))
        self.assertEqual(self.probe.inputs, [])


class BindingAndHarnessTest(unittest.TestCase):
    def test_all_15006_modes_and_no_parse_claims_on_unverified(self):
        coverage = json.loads((ROOT / "coverage.json").read_text())
        self.assertEqual(coverage["modes"], {"complete_document": 4488, "completion_prefix": 4282,
                                             "framed_fragment": 3503, "unverified": 2733})
        ids = set()
        unverified = 0
        with (ROOT / "syntax-evidence-index.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line); ids.add(row["row_id"])
                if row["syntax_evidence_mode"] == "unverified":
                    unverified += 1
                    self.assertNotIn("baseline_blob", row)
                    self.assertNotIn("parser_identity", row)
        self.assertEqual((len(ids), unverified), (15006, 2733))
        self.assertFalse(ids & HELD_CONTRADICTORY_IDS)
    def test_held_id_refusal_is_explicit(self):
        index = RewardBufferIndex(Path("."), {}, "0" * 64, 15008, HELD_CONTRADICTORY_IDS)
        for row_id in HELD_CONTRADICTORY_IDS:
            with self.assertRaisesRegex(BindingError, "held_contradictory_id_refused"):
                index.envelope_for(row_id)
    def test_fixed_r_harness_parses_without_executing_generated_expression(self):
        harness = TRAIN / "reward_parse_only.R"
        probe = RParseOnlyProbe(harness, sha_file(harness), timeout_seconds=5)
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "must-not-exist"
            generated = f"writeLines('executed', {str(sentinel)!r})\n"
            self.assertTrue(probe(generated))
            self.assertFalse(sentinel.exists())
            self.assertFalse(probe("if (\n"))


if __name__ == "__main__":
    unittest.main()
