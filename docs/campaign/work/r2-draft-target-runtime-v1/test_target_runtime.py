from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from target_runtime import (
    _tiny_model_manifest,
    run_tiny_integration,
    validate_model_manifest,
)


class TargetRuntimeTest(unittest.TestCase):
    def test_tiny_real_causallm_generation_and_official_cache(self) -> None:
        report = run_tiny_integration()
        result = report["result"]
        self.assertTrue(report["model_manifest_verified"])
        self.assertTrue(report["model_weights_loaded"])
        self.assertEqual(result["written_rows"], 1)
        self.assertEqual(result["loss_masks"], [[0, 0, 0, 1]])
        self.assertTrue(result["tap_outputs_match_hidden_states"])
        self.assertEqual(result["generation_summary"]["eog_terminated"], 1)
        self.assertEqual(result["target_cache_version"], 2)
        self.assertEqual(result["index_record_size"], 56)

    def test_model_hash_gate_fails_before_load_when_config_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-target-manifest-") as temp:
            manifest, _model = _tiny_model_manifest(Path(temp))
            config_path = Path(temp) / "tiny-target" / "config.json"
            config_path.write_text(config_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config hash"):
                validate_model_manifest(manifest)

    def test_model_hash_fields_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "model manifest missing"):
            validate_model_manifest({"model_dir": "/tmp/no-model"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
