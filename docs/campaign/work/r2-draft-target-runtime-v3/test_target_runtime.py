from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from target_runtime import (
    GreedyTargetGenerator,
    MAX_CONTEXT_TOKENS,
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
        self.assertTrue(all(status == "unchecked" for status in result["record_protocol_statuses"]))
        self.assertEqual(result["target_cache_version"], 2)
        self.assertEqual(result["index_record_size"], 56)

    def test_real_generator_reports_unchecked_and_rejects_context_overflow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-target-context-") as temp:
            _manifest, model = _tiny_model_manifest(Path(temp))
            generator = GreedyTargetGenerator(model, vocab_size=32)
            prefix = [0] + [2] * (MAX_CONTEXT_TOKENS - 1)
            result = generator(prefix, 192)
            self.assertEqual(result["generated_ids"], [])
            self.assertEqual(result["input_ids"], prefix)
            self.assertEqual(result["protocol_status"], "unchecked")
            self.assertEqual(
                result["protocol_error"],
                "prompt_plus_generation_exceeds_context_limit",
            )

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

    def test_relocated_train_hash_is_checked_before_runtime(self) -> None:
        with tempfile.NamedTemporaryFile(prefix="sepalith-train-", mode="w") as stream:
            stream.write("wrong source\n")
            stream.flush()
            with self.assertRaisesRegex(ValueError, "relocated TRAIN source SHA-256"):
                # The source gate runs before touching model attributes or
                # opening a cache, so this test needs no model payload.
                from target_runtime import run_target_runtime

                run_target_runtime(
                    rows=[],
                    target_model=object(),
                    target_identity={},
                    output_dir=Path(stream.name).parent / "unused-cache",
                    source_actual_path=stream.name,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
