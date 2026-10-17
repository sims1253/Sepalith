"""Reward v2.1 fixtures (card C11). Needs Rscript and air on PATH."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import unittest
from unittest import mock

from sepalith.campaign_protocol import Cursor, Position, PromptContext, ReplacementRange
from sepalith.training.rl.air_canonical import AirFormatter, AirInfrastructureError
from sepalith.training.rl.campaign_r_parse_probe import (
    DEFAULT_HARNESS,
    ParseInfrastructureError,
    RParseOnlyProbe,
    sha_file,
)
from sepalith.training.rl.length_policy import REVIEWED_POLICY, REVIEWED_POLICY_V4
from sepalith.training.rl.reward_v21 import (
    TRAIN_INPUTS,
    CampaignPRM03RewardV21,
    Evidence,
    ReferenceSpec,
    RewardV21Config,
    RewardV21Error,
    RewardV21Scorer,
    RowReference,
    Sample,
    load_train_specs,
)

HAVE_TOOLS = shutil.which("Rscript") is not None and shutil.which("air") is not None
KILL_SELF = ("bash", "-c", "kill -KILL $$", "killed")

DOCUMENT = """scale_values <- function(x, center = TRUE) {
  m <- mean(x)
  s <- sd(x)
  y <- (x - m) / s
  y
}
"""
REGION = ("  m <- mean(x)", "  s <- sd(x)", "  y <- (x - m) / s")
TARGET = "  mu <- mean(x)\n  s <- sd(x)\n  y <- (x - mu) / s"
FORMAT_DOCUMENT = """f <- function(x) {
  y<-(x-1)/2
  y
}
"""


def context_for(document: str, start_line: int, region: tuple[str, ...], *, eol: str = "lf") -> PromptContext:
    lines = document.replace("\r\n", "\n").split("\n")
    end_line = start_line + len(region) - 1
    return PromptContext(
        path="R/fixture.R",
        prefix=tuple(lines[:start_line]),
        selected_references=(),
        history=(),
        diagnostics=(),
        retrieval=(),
        scope_mode="off",
        scope_lines=(),
        suffix_lines=tuple(line for line in lines[end_line + 1:] if line),
        region_old=region,
        cursor=Cursor(0, 0, 0),
        replacement_range=ReplacementRange(
            uri="file:///sepalith/R/fixture.R",
            document_version=0,
            content_sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
            start=Position(start_line, 0),
            end=Position(end_line, len(region[-1])),
        ),
        document_eol=eol,
    )


def wire(*body: str) -> str:
    return "\n".join((*body, ">>>>>>> UPDATED", ""))


@unittest.skipUnless(HAVE_TOOLS, "reward v2.1 needs Rscript and air")
class RewardV21Tests(unittest.TestCase):
    scorer: RewardV21Scorer
    references: dict[str, RowReference]

    @classmethod
    def setUpClass(cls) -> None:
        probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS), max_attempts=2,
                                retry_delay_seconds=0)
        cls.scorer = RewardV21Scorer(probe, AirFormatter(), workers=2, chunk_size=4)
        evidence = Evidence("complete_document", DOCUMENT, "", probe.r_version())
        specs = [
            ReferenceSpec("edit", "rename_propagation", context_for(DOCUMENT, 1, REGION), "replace", TARGET,
                          evidence),
            ReferenceSpec("noop", "no_op", context_for(DOCUMENT, 1, REGION), "no_op", "[NO_EDIT]", evidence),
            ReferenceSpec("format", "format_propagation", context_for(FORMAT_DOCUMENT, 1, ("  y<-(x-1)/2",)),
                          "replace", "  y <- (x - 1) / 2",
                          Evidence("complete_document", FORMAT_DOCUMENT, "", probe.r_version())),
            ReferenceSpec("window", "rename_propagation", context_for(DOCUMENT, 1, REGION), "replace", TARGET,
                          Evidence("unverified")),
            ReferenceSpec("crlf", "rename_propagation",
                          context_for(DOCUMENT.replace("\n", "\r\n"), 1, REGION, eol="crlf"), "replace", TARGET,
                          Evidence("complete_document", DOCUMENT.replace("\n", "\r\n"), "", probe.r_version())),
        ]
        cls.references = cls.scorer.build_references(specs)

    def score(self, row_id: str, text: str, *, stop: str = "eos",
              scorer: RewardV21Scorer | None = None) -> dict[str, object]:
        sample = Sample(row_id, completion_text=text, stop=stop)
        return (scorer or self.scorer).score(self.references, [sample])[0]

    def test_exact_edit_scores_one(self) -> None:
        record = self.score("edit", wire(*TARGET.split("\n")))
        self.assertEqual((record["exact"], record["reward"], record["outcome"], record["comparison"]),
                         (True, 1.0, "exact", "air"))
        self.assertTrue(record["exact_raw"])

    def test_alternative_formatting_is_exact_after_canonicalization(self) -> None:
        record = self.score("edit", wire("  mu<-mean( x )", "  s <- sd(x)", "  y <- (x-mu)/s"))
        self.assertTrue(record["exact"])
        self.assertFalse(record["exact_raw"])
        self.assertEqual((record["reward"], record["comparison"], record["parse"]), (1.0, "air", "passed"))

    def test_semantically_different_edit_gets_bounded_partial_credit(self) -> None:
        record = self.score("edit", wire("  mu <- mean(x)", "  s <- sd(x)", "  y <- (x - m) / s"))
        self.assertFalse(record["exact"])
        self.assertEqual(record["outcome"], "near_miss")
        self.assertEqual(record["similarity_space"], "canonical")
        self.assertAlmostEqual(float(str(record["similarity"])), 2 / 3, places=5)
        self.assertAlmostEqual(float(str(record["baseline_similarity"])), 1 / 3, places=5)
        # gain = (2/3 - 1/3) / (1 - 1/3) = 0.5, reward = 0.5 x 0.5
        self.assertAlmostEqual(float(str(record["reward"])), 0.25, places=5)

    def test_partial_credit_never_reaches_a_correct_edit(self) -> None:
        with self.assertRaises(RewardV21Error):
            RewardV21Config(partial_weight=0.6)
        plain = RewardV21Scorer(self.scorer.probe, self.scorer.formatter,
                                RewardV21Config(partial_credit="plain"), workers=1)
        record = self.score("edit", wire("  mu <- mean(x)", "  s <- sd(x)", "  y <- (x - m) / s"), scorer=plain)
        self.assertAlmostEqual(float(str(record["reward"])), 0.5 * 2 / 3, places=5)
        self.assertLess(float(str(record["reward"])), 1.0)

    def test_partial_credit_can_be_switched_off(self) -> None:
        off = RewardV21Scorer(self.scorer.probe, self.scorer.formatter,
                              RewardV21Config(partial_credit="off"), workers=1)
        record = self.score("edit", wire("  mu <- mean(x)", "  s <- sd(x)", "  y <- (x - m) / s"), scorer=off)
        self.assertEqual((record["exact"], record["reward"], record["outcome"]), (False, 0.0, "near_miss"))

    def test_cosmetic_echo_of_the_old_region_earns_nothing(self) -> None:
        record = self.score("edit", wire("  m <- mean( x )", "  s <- sd(x)", "  y <- (x-m)/s"))
        self.assertEqual((record["exact"], record["outcome"], record["reward"]), (False, "near_miss", 0.0))

    def test_parse_failure_scores_zero(self) -> None:
        record = self.score("edit", wire("  mu <- mean(x", "  s <- sd(x)", "  y <- (x - mu) / s"))
        self.assertEqual((record["exact"], record["reward"], record["outcome"], record["parse"]),
                         (False, 0.0, "applied_syntax_invalid", "failed"))

    def test_missed_edit_scores_zero(self) -> None:
        record = self.score("edit", wire("[NO_EDIT]"))
        self.assertEqual((record["exact"], record["reward"], record["outcome"]), (False, 0.0, "missed_edit"))

    def test_no_op_true_positive_and_false_positive(self) -> None:
        record = self.score("noop", wire("[NO_EDIT]"))
        self.assertEqual((record["exact"], record["reward"], record["outcome"]), (True, 1.0, "correct_no_edit"))
        record = self.score("noop", wire(*TARGET.split("\n")))
        self.assertEqual((record["exact"], record["reward"], record["outcome"]),
                         (False, -0.5, "false_positive_edit"))
        # A whitespace-only rewrite is still a suggested edit on a no-op case.
        record = self.score("noop", wire("  m <- mean( x )", "  s <- sd(x)", "  y <- (x - m) / s"))
        self.assertEqual(record["reward"], -0.5)

    def test_protocol_invalid_and_cap_hit(self) -> None:
        record = self.score("edit", "  mu <- mean(x)\n")
        self.assertEqual((record["exact"], record["reward"], record["outcome"]), (False, -0.5, "protocol_invalid"))
        record = self.score("edit", wire(*TARGET.split("\n")), stop="length")
        self.assertEqual((record["reward"], record["outcome"], record["cap_hit"]), (-0.5, "cap_hit", True))
        capped = RewardV21Scorer(self.scorer.probe, self.scorer.formatter,
                                 RewardV21Config(max_completion_tokens=4), workers=1)
        record = capped.score(self.references, [Sample("edit", completion_ids=(40, 41, 42, 43))],
                              lambda ids: "x")[0]
        self.assertEqual((record["reward"], record["outcome"], record["failure"]),
                         (-0.5, "cap_hit", "missing_canonical_eos"))
        record = capped.score(self.references, [Sample("edit", completion_ids=(40, 1, 41, 1))],
                              lambda ids: "x")[0]
        self.assertEqual((record["reward"], record["outcome"], record["failure"]),
                         (-0.5, "protocol_invalid", "premature_canonical_eos"))

    def test_formatting_only_reference_is_compared_raw(self) -> None:
        reference = self.references["format"]
        self.assertTrue(reference.formatting_target)
        record = self.score("format", wire("  y <- (x - 1) / 2"))
        self.assertEqual((record["exact"], record["comparison"]), (True, "raw_formatting_target"))
        # Canonically identical to the unchanged region, but not the formatting asked for.
        record = self.score("format", wire("  y <-(x-1) / 2"))
        self.assertEqual((record["exact"], record["comparison"], record["reward"]),
                         (False, "raw_formatting_target", 0.0))

    def test_context_window_and_crlf_documents(self) -> None:
        self.assertEqual(self.references["window"].evidence_mode, "context_window")
        self.assertTrue(self.references["window"].parse_available)
        for row_id in ("window", "crlf"):
            record = self.score(row_id, wire("  mu<-mean(x)", "  s <- sd(x)", "  y <- (x - mu) / s"))
            self.assertEqual((record["exact"], record["comparison"]), (True, "air"), row_id)

    def test_killed_r_process_raises_instead_of_scoring(self) -> None:
        probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS), max_attempts=2,
                                retry_delay_seconds=0)
        probe._r_version = self.scorer.r_version  # skip the version call
        probe.command = KILL_SELF
        with self.assertRaisesRegex(ParseInfrastructureError, "signal_9"):
            probe("x <- 1\n")
        self.assertEqual(probe.infrastructure_failures, 2)
        killed = RewardV21Scorer(probe, self.scorer.formatter, workers=1)
        with self.assertRaises(ParseInfrastructureError):
            self.score("edit", wire("  mu <- mean(x)", "  s <- sd(x)", "  y <- (x - m) / s"), scorer=killed)

    def test_undesignated_exit_and_timeout_raise(self) -> None:
        probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS), max_attempts=1,
                                retry_delay_seconds=0, timeout_seconds=0.5)
        probe.command = ("bash", "-c", "exit 1", "r")
        with self.assertRaisesRegex(ParseInfrastructureError, "exit_1"):
            probe("x <- 1\n")
        probe.command = ("bash", "-c", "sleep 5", "r")
        with self.assertRaisesRegex(ParseInfrastructureError, "timeout"):
            probe("x <- 1\n")

    def test_designated_syntax_exit_and_batch_agree(self) -> None:
        probe = self.scorer.probe
        texts = ["x <- 1\n", "x <- (\n", "f <- function() {\n  1\n}\n", "}\n"]
        self.assertEqual([probe(text) for text in texts], [True, False, True, False])
        self.assertEqual(probe.parse_many(texts), [True, False, True, False])
        self.assertEqual(probe.r_version()[:2], "R ")

    def test_air_infrastructure_failure_raises(self) -> None:
        self.assertIsNone(self.scorer.formatter.format("x <- (\n"))
        broken = AirFormatter(KILL_SELF, max_attempts=2, retry_delay_seconds=0)
        with self.assertRaisesRegex(AirInfrastructureError, "signal_9"):
            broken.format("x <- 1\n")

    def test_parser_identity_must_match_buffer_evidence(self) -> None:
        spec = ReferenceSpec("edit", "rename_propagation", context_for(DOCUMENT, 1, REGION), "replace", TARGET,
                             Evidence("complete_document", DOCUMENT, "", "R 3.6.0 (2019-04-26)"))
        with self.assertRaisesRegex(RewardV21Error, "differs from the probe"):
            self.scorer.build_references([spec])

    def test_trl_callable_matches_batch_scores(self) -> None:
        class Index:
            @staticmethod
            def baseline_text(envelope: dict[str, object]) -> str:
                return DOCUMENT

        context = context_for(DOCUMENT, 1, REGION)
        envelope = {"row_id": "edit", "context_sha256": context.replacement_range.content_sha256,
                    "syntax_evidence_mode": "complete_document", "diagnostic_suffix": "",
                    "parser_identity": {"r_version": self.scorer.r_version}}
        texts = {40: wire(*TARGET.split("\n")), 41: wire("[NO_EDIT]")}
        reward = CampaignPRM03RewardV21(self.scorer, Index(), decoder=lambda ids: texts[ids[0]])
        values = reward(completion_ids=[[40, 1], [41, 1]], context=[context.to_dict()] * 2,
                        target_operation=["replace"] * 2, target_body_text=[TARGET] * 2,
                        reward_buffer=[envelope] * 2, id=["edit"] * 2, family=["rename_propagation"] * 2)
        self.assertEqual(values, [1.0, 0.0])
        self.assertEqual([record["exact"] for record in reward.last_records], [True, False])


class TrainInputsTests(unittest.TestCase):
    """The census scores against the rows arm A was trained on (DAT10 v4-braceindent)."""

    def test_train_inputs_are_the_v4_braceindent_rows(self) -> None:
        rows_path, rows_sha = TRAIN_INPUTS["rows"]
        context_path, context_sha = TRAIN_INPUTS["context"]
        self.assertTrue(rows_path.startswith("DAT10-finish-source-repair-v4-braceindent/"))
        self.assertTrue(context_path.startswith("DAT10-finish-source-repair-v4-braceindent/"))
        self.assertEqual(rows_sha, "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef")
        # v4 leaves prompts and ranges alone: the sidecar is byte-identical to v3's.
        self.assertEqual(context_sha, "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a")
        self.assertEqual(REVIEWED_POLICY_V4.input_rows_sha256, rows_sha)

    def test_loader_binds_the_v4_length_policy(self) -> None:
        class Stop(Exception):
            pass

        target = "sepalith.training.rl.campaign_rl_data.load_training_records"
        with mock.patch(target, side_effect=Stop) as loader, self.assertRaises(Stop):
            load_train_specs(Path("/nonexistent"))
        args, kwargs = loader.call_args
        self.assertEqual(args[0], Path("/nonexistent") / TRAIN_INPUTS["rows"][0])
        self.assertEqual(args[1], TRAIN_INPUTS["rows"][1])
        self.assertIs(kwargs["length_policy"], REVIEWED_POLICY_V4)
        self.assertEqual(RewardV21Config().max_completion_tokens, REVIEWED_POLICY.completion_max_tokens)


if __name__ == "__main__":
    unittest.main()
