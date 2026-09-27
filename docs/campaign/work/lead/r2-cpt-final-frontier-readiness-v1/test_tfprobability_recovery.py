#!/usr/bin/env python3
"""Small source-only tests for the fail-closed tfprobability recovery packet."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "prepare_tfprobability_recovery.py"


def load_recovery():
    spec = importlib.util.spec_from_file_location("tfprobability_recovery_test_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("recovery_script_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recovery = load_recovery()

    def test_dcf_multiline_fields_are_folded(self):
        parsed = self.recovery.fields(
            "Package: tfprobability\n"
            "Description: TensorFlow distributions\n"
            " and bijectors.\n"
            "License: Apache-2.0\n"
        )
        self.assertEqual(parsed["Package"], "tfprobability")
        self.assertEqual(parsed["Description"], "TensorFlow distributions and bijectors.")
        self.assertEqual(parsed["License"], "Apache-2.0")

    def test_chunks_cover_every_source_token_without_truncation(self):
        ids = list(range(2, 13))
        chunks = list(self.recovery.load_module("test_raw_chunks", self.recovery.RAW_CHUNKS).chunks(ids, 6))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            [token for chunk in chunks for token, label in zip(chunk["input_ids"], chunk["labels"])
             if label != -100],
            ids + [self.recovery.load_module("test_raw_chunks_eos", self.recovery.RAW_CHUNKS).EOS],
        )
        self.assertEqual(chunks[-1]["is_document_end"], True)
        self.assertTrue(all(len(chunk["input_ids"]) <= 6 for chunk in chunks))
        self.assertEqual(sum(chunk["supervised_tokens"] for chunk in chunks), len(ids) + 1)

    def test_pinned_package_inventory_is_tokenizable_but_metadata_pending(self):
        recovery = self.recovery
        guards = recovery.load_guards()
        self.assertEqual(guards["global_split"]["split"], "train_group")
        self.assertEqual(guards["cpt_partition"]["partition"], "cpt_train")
        source = recovery.package_root()
        metadata = recovery.metadata_evidence(source)
        self.assertEqual(metadata["status"], "missing_description")
        self.assertEqual(metadata["license_status"], "unresolved_no_description_or_license_metadata")
        self.assertTrue(metadata["md5_mentions_description"])
        tokenizer_module = __import__("tokenizers", fromlist=["Tokenizer"])
        tokenizer = tokenizer_module.Tokenizer.from_file(str(recovery.TOKENIZER))
        tokenizer.encode_special_tokens = True
        inventory, repairs, counts = recovery.source_inventory(source, tokenizer)
        self.assertEqual(len(inventory), 21)
        self.assertEqual(counts["regular_R"], 21)
        self.assertEqual(sum(row["bytes"] for row in inventory), 693518)
        self.assertEqual(sum(row["token_count"] for row in inventory), 182645)
        self.assertEqual(sum(row["chunk_count"] for row in inventory), 103)
        self.assertEqual(len(repairs), 0)
        self.assertTrue(all(row["status"] == "tokenizable" for row in inventory))


if __name__ == "__main__":
    unittest.main(verbosity=2)
