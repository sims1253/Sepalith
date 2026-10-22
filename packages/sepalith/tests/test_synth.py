"""SYNTH-v1 generator: geometry, cut selection, roxygen checks and collapses."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from sepalith.campaign_protocol import PromptContext, render_prompt
from sepalith.data.synth import finish_block, format_propagation, roxygen
from sepalith.data.synth.cases import Case, Event
from sepalith.data.synth.common import novelty
from sepalith.data.synth.normalize import apply_blank_policy, style_for

HAS_R = shutil.which("Rscript") is not None

SOURCE = '''#' Scale a vector
#'
#' Centre and scale a numeric vector.
#'
#' @param x A numeric vector.
#' @param center Whether to centre.
#' @return The scaled vector.
#' @export
scale_vec <- function(x, center = TRUE) {
  stopifnot(is.numeric(x))
  if (center) {
    x <- x - mean(x)
  }
  s <- sd(x)
  x / s
}
'''


class EventDiffTests(unittest.TestCase):
    def test_single_line_header_matches_train(self):
        event = Event("R/a.R", ("a", "x <- 1", "b"), 1, ("x <- 1",), ("x <- 2",))
        self.assertEqual(event.event_diff(), 'User edited "R/a.R":\n\n```diff\n@@ -2 +2 @@\n-x <- 1\n+x <- 2\n```')

    def test_multi_line_header_has_counts(self):
        event = Event("R/a.R", ("f(a, b)",), 0, ("f(a, b)",), ("f(", "  a,", "  b", ")"))
        self.assertIn("@@ -1,1 +1,4 @@", event.event_diff())
        record = event.to_dict("file:///synth-v1/p/R/a.R")
        self.assertEqual(record["range_utf16"]["end"], {"character": 7, "line": 0})


class CaseGeometryTests(unittest.TestCase):
    def test_insertion_case_round_trips_through_prompt_context(self):
        case = Case("finish_block", "synth:p", "R/a.R", ("f <- function(x) {", "  y <- x"), 2, 2, ("  y", "}"))
        context = PromptContext.from_dict(case.to_context())
        self.assertEqual(context.cursor.region_line_index, -1)
        prompt = render_prompt(context)
        self.assertTrue(prompt.endswith("<<<<<<< CURRENT\n=======\n<[fim-middle]>\n"))
        self.assertEqual(len(case.row_id()), 24)

    def test_region_case_has_cursor_and_range(self):
        case = Case("format_propagation", "synth:p", "R/a.R", ("a <- 1", "f(a, b)"), 1, 2, ("f(", "  a,", "  b", ")"),
                    cursor=(0, 0))
        context = case.to_context()
        self.assertEqual(context["replacement_range"]["end"], {"character": 7, "line": 1})
        self.assertEqual(case.after_doc(), ("a <- 1", "f(", "  a,", "  b", ")"))


class NormalisationTests(unittest.TestCase):
    def test_style_is_deterministic(self):
        self.assertEqual(style_for("b01/x/R/a.R"), style_for("b01/x/R/a.R"))

    @unittest.skipUnless(HAS_R, "Rscript not available")
    def test_dense_and_sectioned_policies(self):
        from sepalith.data.synth.structure import describe
        text = "f <- function(x) {\n  a <- 1\n\n  # step two\n  b <- 2\n  s <- \"line\n\nend\"\n  b\n}\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.R"
            path.write_text(text)
            structure = describe([path])[str(path)]
        lines = text.rstrip("\n").split("\n")
        dense = apply_blank_policy(lines, structure, "dense")
        self.assertNotIn("", dense[1:4])
        self.assertIn("", dense[5:8])  # blank line inside the string is kept
        sectioned = apply_blank_policy(lines, structure, "sectioned")
        self.assertEqual(sectioned[2:4], ["", "  # step two"])


@unittest.skipUnless(HAS_R, "Rscript not available")
class FinishBlockTests(unittest.TestCase):
    def setUp(self):
        from sepalith.data.synth.structure import describe
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "a.R"
        path.write_text(SOURCE)
        self.structure = describe([path])[str(path)]
        self.lines = SOURCE.rstrip("\n").split("\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cuts_cover_statements_nested_close_and_final_close(self):
        unit = self.structure.units[0]
        self.assertEqual(unit.roxygen, (1, 8))
        cuts = {c.line: c.kind for c in finish_block.cuts_for(unit, self.lines, self.structure)}
        self.assertEqual(cuts, {10: "statement", 11: "statement", 12: "statement", 13: "nested_close",
                                14: "statement", 15: "statement", 16: "final_close"})

    def test_diag_header_counts_typed_code(self):
        unit = self.structure.units[0]
        cut = finish_block.Cut(13, "nested_close", 4)
        header = finish_block.header_lines("diag", unit, self.lines, cut)
        self.assertEqual(header[1], "# typed statements: 1 of 4")
        self.assertEqual(header[2], "# args not yet referenced: none")
        self.assertEqual(header[3], "# unclosed braces in typed code: 2")
        outline = finish_block.header_lines("outline", unit, self.lines, cut)
        self.assertEqual(outline[1], "# fn: scale_vec(x, center)")

    def test_case_target_is_remainder(self):
        unit = self.structure.units[0]
        case = finish_block.build_case(unit, finish_block.Cut(14, "statement", 3), "strip", self.lines,
                                       self.structure, "synth:p", "R/a.R")
        self.assertEqual(case.target, ("  s <- sd(x)", "  x / s", "}"))
        self.assertEqual(case.doc[0], "scale_vec <- function(x, center = TRUE) {")
        self.assertEqual(case.meta["novel_tokens"], ["sd"])  # one-character names are ignored


@unittest.skipUnless(HAS_R, "Rscript not available")
class RoxygenTests(unittest.TestCase):
    def test_block_checks_signature(self):
        from sepalith.data.synth.structure import describe
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.R"
            path.write_text(SOURCE)
            unit = describe([path])[str(path)].units[0]
        lines = SOURCE.split("\n")
        self.assertEqual(roxygen.check_block(lines[:8], unit), [])
        swapped = [lines[0], lines[1], lines[2], lines[3], lines[5], lines[4], lines[6], lines[7]]
        self.assertIn("param_order_differs_from_signature", roxygen.check_block(swapped, unit))
        self.assertIn("formal_not_documented", roxygen.check_block([lines[0], lines[4], lines[7]], unit))
        self.assertIn("unknown_tag:exprt", roxygen.check_block([lines[0], lines[4], lines[5], "#' @exprt"], unit))

    def test_determined_share(self):
        self.assertEqual(roxygen.determined_share(["#' @export", "#' new prose"], ["x"]), 0.5)


class FormatCollapseTests(unittest.TestCase):
    def test_one_line_and_aligned(self):
        lines = ["out <- list(", "  alpha = 1,", "  beta = c(1, 2)", ")"]
        self.assertEqual(format_propagation.one_line(lines), ["out <- list(alpha = 1, beta = c(1, 2))"])
        self.assertEqual(format_propagation.aligned(lines), ["out <- list(alpha = 1,", "            beta = c(1, 2))"])


class NoveltyTests(unittest.TestCase):
    def test_comment_tokens_ignored(self):
        self.assertEqual(novelty(["  yy <- foo(x) # bar baz"], "x <- 1"), ["foo", "yy"])


if __name__ == "__main__":
    unittest.main()
