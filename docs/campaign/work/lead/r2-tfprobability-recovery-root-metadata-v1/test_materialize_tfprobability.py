#!/usr/bin/env python3
"""Synthetic contract tests for the tfprobability recovery preparer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "materialize_tfprobability.py"
RAW = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py")


def load_module():
    spec = importlib.util.spec_from_file_location("tfprobability_materializer_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("materializer_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.raw = cls.module.load_module("tfprobability_test_raw", RAW)

    def test_multiline_description_fields(self):
        result = self.module.fields(
            "Package: tfprobability\n"
            "Description: A distribution package\n"
            " for R.\n"
            "License: Apache License (>= 2.0)\n"
        )
        self.assertEqual(result["Package"], "tfprobability")
        self.assertEqual(result["Description"], "A distribution package for R.")
        self.assertEqual(result["License"], "Apache License (>= 2.0)")

    def test_chunk_coverage_preserves_source_tokens_and_terminal_eos(self):
        ids = list(range(2, 19))
        chunks = list(self.raw.chunks(ids, 7))
        supervised = [
            token
            for chunk in chunks
            for token, label in zip(chunk["input_ids"], chunk["labels"])
            if label != -100
        ]
        self.assertEqual(supervised, ids + [self.raw.EOS])
        self.assertEqual(sum(chunk["source_token_end"] - chunk["source_token_start"] for chunk in chunks), len(ids))
        self.assertEqual(sum(chunk["supervised_tokens"] for chunk in chunks), len(ids) + 1)
        self.assertTrue(chunks[-1]["is_document_end"])
        self.assertTrue(all(len(chunk["input_ids"]) <= 7 for chunk in chunks))

    def test_seen_snapshot_requires_unique_lowercase_sha256_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.txt"
            valid.write_text("a" * 64 + "\n" + "b" * 64 + "\n", encoding="utf-8")
            values, digest = self.module.read_seen_snapshot(valid)
            self.assertEqual(values, {"a" * 64, "b" * 64})
            self.assertEqual(len(digest), 64)
            invalid = Path(directory) / "invalid.txt"
            invalid.write_text("A" * 64 + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.read_seen_snapshot(invalid)

    def test_upstream_metadata_constants_are_pinned(self):
        self.assertEqual(self.module.TAR_SHA, "9268e148396a4231aa11bee32ddd755e879abe243b84c3064e010db1d21bd46c")
        self.assertEqual(self.module.DESCRIPTION_SHA, "f8a3ba18f403feb221352417c643ab22bf2c4e65846bed30e974a37778a45163")
        self.assertEqual(self.module.GROUP_ID, "g-1037a9f3b52fac791d3d")
        self.assertEqual(self.module.CONTEXT_SIZE, 16384)


if __name__ == "__main__":
    unittest.main(verbosity=2)
