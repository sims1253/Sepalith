#!/usr/bin/env python3
"""Small CPU-only checks for the SFT-06 diagnostic adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sft06_dev_error_diagnostic as diagnostic  # noqa: E402


class DiagnosticUnitTests(unittest.TestCase):
    def test_protocol_decoder_removes_only_terminal_and_trailing_newline(self) -> None:
        self.assertEqual(
            diagnostic.decoded_output("x\n>>>>>>> UPDATED\nignored"),
            "x",
        )

    def test_repeated_block_requires_contiguous_nonblank_repetition(self) -> None:
        self.assertEqual(
            diagnostic.repeated_block(["a", "b", "a", "b", "a", "b"]),
            {"block_lines": ["a", "b"], "width": 2, "repeats": 3},
        )
        self.assertIsNone(diagnostic.repeated_block(["}", "x", "}", "x", "}"]))

    def test_locked_scope_and_parser_projection(self) -> None:
        evidence = diagnostic.analyse()
        self.assertEqual(evidence["scope"]["rows"], 14)
        self.assertEqual(evidence["scope"]["finish_block"], 6)
        self.assertEqual(evidence["scope"]["roxygen_drafting"], 8)
        self.assertEqual(evidence["counts"]["exact_region"], 0)
        self.assertEqual(evidence["counts"]["strict_textual_mismatch"], 14)
        self.assertEqual(evidence["counts"]["protocol_valid"], 10)
        self.assertEqual(evidence["counts"]["protocol_invalid_or_capped"], 4)
        self.assertEqual(evidence["counts"]["repeated_block"], 3)
        self.assertFalse(evidence["inputs"]["step1000"]["admitted"])
        for row in evidence["rows"]:
            self.assertFalse(row["projected_parse"]["target"]["has_error"])
            self.assertFalse(row["projected_parse"]["generated"]["has_error"])
            if row["family"] == "finish_block":
                self.assertTrue(row["projected_parse"]["synthetic_close_added"])


if __name__ == "__main__":
    unittest.main()
