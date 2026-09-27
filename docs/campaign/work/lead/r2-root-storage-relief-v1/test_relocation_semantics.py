"""Synthetic, payload-free checks for the proposed directory-link strategy."""
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


PACKET = Path(__file__).resolve().parents[5]
WORK_LEAD = PACKET / "docs" / "campaign" / "work" / "lead"
TRAINING_SOURCE = WORK_LEAD / "r2-full-weight-cpt-representative-trainer-v3" / "source" / "experiments" / "training"
NATIVE_EVALUATOR = WORK_LEAD / "r2-task-global500-native-selection" / "packet" / "native_evaluator"
sys.path.insert(0, str(TRAINING_SOURCE))
from model_layout import validate_dense_weights  # noqa: E402


class RelocationSemanticsTests(unittest.TestCase):
    def test_directory_link_keeps_dense_files_regular(self):
        with tempfile.TemporaryDirectory(
            prefix="pre03-symlink-test-", dir="/mnt/e/sepalith/campaign-20260915/data-work"
        ) as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            save_file({"weight": np.zeros((2,), dtype=np.float32)}, str(source / "model.safetensors"))
            linked = root / "linked"
            linked.symlink_to(source, target_is_directory=True)

            result = validate_dense_weights(linked)

            self.assertTrue(linked.is_symlink())
            self.assertFalse((linked / "config.json").is_symlink())
            self.assertFalse((linked / "model.safetensors").is_symlink())
            self.assertEqual(result["weight_files"], ["model.safetensors"])

    def test_individual_weight_link_is_rejected(self):
        with tempfile.TemporaryDirectory(
            prefix="pre03-file-link-test-", dir="/mnt/e/sepalith/campaign-20260915/data-work"
        ) as temporary:
            root = Path(temporary)
            source = root / "source"
            bad = root / "bad"
            source.mkdir()
            bad.mkdir()
            (source / "config.json").write_text("{}\n")
            save_file({"weight": np.zeros((2,), dtype=np.float32)}, str(source / "model.safetensors"))
            (bad / "config.json").write_text("{}\n")
            (bad / "model.safetensors").symlink_to(source / "model.safetensors")

            with self.assertRaisesRegex(ValueError, "symlinked"):
                validate_dense_weights(bad)

    def test_final_row_path_guard_rejects_directory_link_component(self):
        sys.path.insert(0, str(NATIVE_EVALUATOR))
        final_row_gate = importlib.import_module("final_row_gate")
        with tempfile.TemporaryDirectory(
            prefix="pre03-final-guard-test-", dir="/mnt/e/sepalith/campaign-20260915/data-work"
        ) as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            linked = root / "linked"
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaises(Exception):  # exact gate error is implementation-owned
                final_row_gate._safe_metadata_path(linked)


if __name__ == "__main__":
    unittest.main()
