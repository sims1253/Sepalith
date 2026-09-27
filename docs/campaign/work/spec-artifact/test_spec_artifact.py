#!/usr/bin/env python3
"""Bounded CPU tests for the RUN-06 released DSpark artifact records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
AUDIT = json.loads((ROOT / "dspark-gguf-audit.json").read_text(encoding="utf-8"))
TRANSFER = json.loads((ROOT / "dspark-download.json").read_text(encoding="utf-8"))
DRAFT = Path(TRANSFER["path"])


class SpecArtifactTests(unittest.TestCase):
    def test_audit_structural_gate_is_complete(self) -> None:
        self.assertTrue(AUDIT["complete_structural_screen"])
        self.assertTrue(all(AUDIT["checks"].values()))
        self.assertEqual(AUDIT["draft_geometry"]["target_layers"], [2, 11, 21, 31, 40])
        self.assertEqual(AUDIT["draft_geometry"]["metadata"]["dflash.block_size"]["actual"], 7)
        self.assertEqual(AUDIT["draft_geometry"]["metadata"]["tokenizer.ggml.mask_token_id"]["actual"], 75982)

    def test_full_tokenizer_arrays_and_special_ids_are_bound(self) -> None:
        comparison = AUDIT["tokenizer_comparison"]
        self.assertTrue(comparison["tokens_types_merges_exact"])
        self.assertTrue(comparison["chat_template_exact"])
        self.assertTrue(comparison["special_ids_exact"])
        self.assertEqual(comparison["non_known_differences"], [])
        self.assertEqual(comparison["token_arrays"]["tokenizer.ggml.tokens"]["draft"]["length"], 130560)
        self.assertEqual(comparison["token_arrays"]["tokenizer.ggml.merges"]["draft"]["length"], 129794)

    def test_download_has_exact_identity_and_no_part_file(self) -> None:
        self.assertTrue(DRAFT.is_file())
        self.assertEqual(DRAFT.stat().st_size, 652730240)
        digest = hashlib.sha256()
        with DRAFT.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), "57df08640f0534a1aac075d1c8bdacdb2b7e5815da6f4e5cfd39ecac3a3f0c26")
        self.assertEqual(list(DRAFT.parent.glob("*.part-*")), [])
        self.assertTrue(TRANSFER["fsync"]["file"])
        self.assertTrue(TRANSFER["fsync"]["directory"])
        self.assertTrue(TRANSFER["no_overwrite"])

    def test_inventory_contains_markov_and_confidence_geometry(self) -> None:
        geometry = AUDIT["draft_geometry"]
        self.assertTrue(geometry["markov_tensors_present"])
        self.assertTrue(geometry["confidence_projection_present"])
        self.assertTrue(geometry["draft_has_no_target_embedding_or_lm_head"])
        self.assertEqual(AUDIT["files"]["draft"]["tensor_count"], 62)
        self.assertEqual(AUDIT["files"]["draft"]["tensor_type_counts"], {"BF16": 39, "F32": 23})


if __name__ == "__main__":
    unittest.main(verbosity=2)
