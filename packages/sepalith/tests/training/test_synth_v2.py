"""SYNTH-v2: decontamination fingerprints, verification helpers, path guard,
no-op exposure rule and novelty (needs numpy; R tests skip without Rscript)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sepalith.data.synth import dedup
from sepalith.data.synth.build import guard_path
from sepalith.data.synth.cases import Case
from sepalith.data.synth.cohort import assert_noop_cap
from sepalith.data.synth.common import lexical_items, novelty, structural_support
from sepalith.data.synth.verify import family_checks, parse_status

HAS_R = shutil.which("Rscript") is not None
CODE = ["f <- function(x, y) {", "  out <- vapply(x, function(v) v + y, numeric(1)) # add", "  out", "}"]
REFORMATTED = ["f <- function(x,", "              y) {", "  out <- vapply(", "    x,", "    function(v) v + y,",
               "    numeric(1)", ")  # ADD", "  out", "}"]


class TokenFingerprintTests(unittest.TestCase):
    def index(self, lines, masked=False):
        idx = dedup.TokenIndex("t", masked=masked)
        idx.add("\n".join(lines))
        idx.freeze()
        return idx

    def test_air_style_reformatting_does_not_hide_a_copy(self):
        self.assertEqual(dedup.line_4grams("\n".join(CODE)) & dedup.line_4grams("\n".join(REFORMATTED)), set())
        containment, run = dedup.overlap(self.index(REFORMATTED).hits(dedup.span_hashes(CODE[:1], CODE[1:])))
        self.assertEqual(containment, 1.0)
        self.assertGreaterEqual(run, 24)

    def test_identifier_renaming_is_caught_by_the_masked_check(self):
        renamed = [line.replace("out", "res").replace("x", "z") for line in CODE]
        exact = dedup.overlap(self.index(REFORMATTED).hits(dedup.span_hashes(renamed[:1], renamed[1:])))[0]
        masked = dedup.overlap(self.index(REFORMATTED, masked=True).hits(
            dedup.span_hashes(renamed[:1], renamed[1:], masked=True)))[0]
        self.assertLess(exact, 0.5)
        self.assertEqual(masked, 1.0)

    def test_short_targets_are_covered_through_context(self):
        hashes = dedup.span_hashes(CODE[:3], ["}"])
        self.assertEqual(hashes.size, 1)  # one 12-gram ending in the target's single token
        self.assertTrue(self.index(CODE).hits(hashes).all())

    def test_unrelated_code_does_not_match(self):
        other = ["g <- function(a) {", "  stopifnot(is.character(a))", "  toupper(a)", "}"]
        self.assertEqual(dedup.overlap(self.index(CODE).hits(dedup.span_hashes(other[:1], other[1:])))[0], 0.0)


class NoveltyTests(unittest.TestCase):
    def test_roxygen_apostrophe_does_not_open_a_string(self):
        idents, words, _ = lexical_items(["#' Don't count 'EUR' as code", "x <- 1"])
        self.assertIn("eur", words)
        self.assertEqual(idents, {"x"})

    def test_string_words_and_numbers_count(self):
        novel = novelty(['  stop("Currency must be USD", call. = FALSE)', "  x * 365"], ["f <- function(x) {"])
        self.assertIn("str:Currency", novel)
        self.assertIn("num:365", novel)
        self.assertNotIn("call.", novel)
        self.assertNotIn("str:USD", novelty(['  stop("USD")'], ["#' Amounts in USD", "f <- function() {"]))

    def test_structural_support_sees_siblings(self):
        sibling = ["read_a <- function(path) {", "  df <- utils::read.csv(path, stringsAsFactors = FALSE)",
                   "  df[order(df$id), , drop = FALSE]", "}", "read_b <- function(path) {"]
        target = ["  df <- utils::read.csv(path, stringsAsFactors = FALSE)", "  df[order(df$id), , drop = FALSE]", "}"]
        self.assertGreater(structural_support(target, sibling), 0.9)
        self.assertLess(structural_support(["  sum(abs(path))", "}"], sibling), 0.5)


class GuardPathTests(unittest.TestCase):
    def test_refuses_final_and_sealed_strings(self):
        for path in ("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-dev-candidates.jsonl",
                     "~/.local/state/sepalith/campaign-20260915/sealed-final/x.json", "/tmp/a/FINAL/b"):
            with self.assertRaises(SystemExit):
                guard_path(path)

    def test_refuses_a_symlink_into_a_final_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "final-set"
            target.mkdir()
            (target / "rows.jsonl").write_text("{}\n")
            link = Path(tmp) / "innocent.jsonl"
            os.symlink(target / "rows.jsonl", link)
            with self.assertRaises(SystemExit):
                guard_path(link)

    def test_allows_dev_paths(self):
        path = "/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/dev250-v1.jsonl"
        self.assertEqual(guard_path(path), path)


class NoopCapTests(unittest.TestCase):
    def schedule(self, exposures):
        draws = [{"row_id": f"noop{i}", "semantic_noop": True} for i, n in enumerate(exposures) for _ in range(n)]
        return {"draws": draws + [{"row_id": "edit", "semantic_noop": False}] * 50}

    def test_passes_at_three(self):
        self.assertEqual(assert_noop_cap(self.schedule([3, 3, 2]))["max_noop_exposures"], 3)

    def test_fails_at_four(self):
        with self.assertRaises(AssertionError):
            assert_noop_cap(self.schedule([3, 4]))


@unittest.skipUnless(HAS_R, "Rscript not available")
class VerifyTests(unittest.TestCase):
    def test_parse_status_classes(self):
        self.assertEqual(parse_status(["x <- 1\n", "f <- function() {\n  1\n", "x <- )\n"]),
                         ["ok", "incomplete", "error"])

    def test_finish_block_family_checks(self):
        good = Case("finish_block", "synth:p", "R/a.R", ("f <- function() {",), 1, 1, ("  1", "}"))
        bad = Case("finish_block", "synth:p", "R/a.R", ("f <- function() {",), 1, 1, ("  1", "  }"))
        open_end = Case("finish_block", "synth:p", "R/a.R", ("f <- function() {",), 1, 1, ("  1",))
        self.assertEqual(family_checks(good), [])
        self.assertIn("closing_brace_not_at_header_column", family_checks(bad))
        self.assertIn("target_does_not_end_with_close", family_checks(open_end))


if __name__ == "__main__":
    unittest.main()
