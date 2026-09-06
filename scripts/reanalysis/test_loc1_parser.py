"""Real R grammar checks on synthetic inputs; never execute dataset code."""

import shutil
import unittest

from loc1_parser import parse_sources


@unittest.skipUnless(shutil.which("Rscript"), "Rscript is required for parser checks")
class RParserTests(unittest.TestCase):
    def test_single_function_with_braces_in_strings(self):
        result = parse_sources(['f <- function(x) { "}"; x }'])[0]
        self.assertEqual(result, {"parse_ok": True, "expressions": 1, "names": ["f"]})

    def test_absorbed_following_function(self):
        result = parse_sources(["f <- function(x) x\ny <- 2\ng <- function() 3"])[0]
        self.assertEqual(result["expressions"], 3)
        self.assertEqual(result["names"], ["f", "g"])

    def test_nested_callback_is_not_top_level(self):
        result = parse_sources(["tryCatch(f(), error = function(e) NULL)"])[0]
        self.assertEqual(result["names"], [])

    def test_fragment_failure_is_local(self):
        a, b = parse_sources(["error = function(e) NULL)", "f = function() 1"])
        self.assertFalse(a["parse_ok"])
        self.assertEqual(b["names"], ["f"])

    def test_parsing_does_not_evaluate(self):
        result = parse_sources(['stop("Must never execute")\n`a b` <- function() 1'])[0]
        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["names"], ["a b"])


if __name__ == "__main__":
    unittest.main()
