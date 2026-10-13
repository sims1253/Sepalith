import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET / "source"
sys.path.insert(0, str(SOURCE))
import full_weight_resume_proof as proof


class ResumeProofPreparationTests(unittest.TestCase):
    def fixture_recipe(self, root: Path):
        recipe = json.loads((PACKET / "recipe.json").read_text())
        model = root / "model"
        model.mkdir()
        files = {}
        for index, name in enumerate(recipe["model"]["files"]):
            payload = f"fixture-{index}\n".encode()
            (model / name).write_bytes(payload)
            files[name] = hashlib.sha256(payload).hexdigest()
        recipe["model"]["path"] = str(model)
        recipe["model"]["files"] = files
        recipe["output_root"] = str(root / "output")
        path = root / "recipe.json"
        path.write_text(json.dumps(recipe))
        return path, recipe

    def test_metadata_preflight_binds_eight_ordered_full_rows(self):
        with tempfile.TemporaryDirectory() as value:
            path, recipe = self.fixture_recipe(Path(value))
            loaded = proof.load_recipe(path)
            self.assertEqual(loaded["data"]["rows"], 8)
            self.assertEqual(len(loaded["data"]["row_ids"]), 8)
            self.assertEqual(proof.identity(loaded)["policy"]["checkpoint_kind"], "full_weights")

    def test_model_byte_tamper_rejected_before_framework_imports(self):
        with tempfile.TemporaryDirectory() as value:
            path, recipe = self.fixture_recipe(Path(value))
            model = Path(recipe["model"]["path"])
            (model / "tokenizer.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "model file differs"):
                proof.load_recipe(path)

    def test_source_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            path, recipe = self.fixture_recipe(Path(value))
            recipe["source"]["manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "source manifest binding differs"):
                proof.load_recipe(path)

    def test_bfloat_tensor_fingerprint_is_exact_and_detects_change(self):
        import torch
        one = {"p": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
        two = {"p": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
        three = {"p": torch.tensor([1.0, 3.0], dtype=torch.bfloat16)}
        self.assertEqual(proof.tensor_mapping_digest(one), proof.tensor_mapping_digest(two))
        self.assertNotEqual(proof.tensor_mapping_digest(one), proof.tensor_mapping_digest(three))

    def test_compare_requires_model_optimizer_scheduler_rng_and_cursor_exact(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            path, recipe = self.fixture_recipe(root)
            out = Path(recipe["output_root"])
            common = {
                "global_step": 2, "model_state_sha256": "1" * 64,
                "optimizer_state_sha256": "2" * 64, "scheduler_state_sha256": "3" * 64,
                "cpu_rng_sha256": "4" * 64, "cuda_rng_sha256": "5" * 64,
                "draw_cursor": {"consumed_updates": 2, "next_row_position": 2, "row_ids": ["a", "b"]},
                "optimizer_dispatch_sha256": "6" * 64, "tokenizer_contract": {"status": "verified"},
            }
            for lane in ("uninterrupted", "resumed"):
                (out / lane).mkdir(parents=True)
                (out / lane / "proof-lane.json").write_text(json.dumps(common))
            self.assertEqual(proof.compare(path)["status"], "exact_two_step_resume_proved")
            changed = dict(common); changed["cuda_rng_sha256"] = "7" * 64
            (out / "resumed" / "proof-lane.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "terminal state differs"):
                proof.compare(path)

    def test_live_source_uses_unsloth_first_and_explicit_full_kind_resume(self):
        source = (SOURCE / "full_weight_resume_proof.py").read_text()
        start = source.index('from unsloth import FastLanguageModel')
        torch_import = source.index('import torch', start)
        self.assertLess(start, torch_import)
        self.assertIn('expected_checkpoint_kind=FULL_KIND', source)
        self.assertIn('resume_from_checkpoint=str(resume)', source)
        self.assertIn('train_sampling_strategy="sequential"', source)
        self.assertIn('trainer.updated_positions == expected_lane_positions', source)
        self.assertIn('restore_pinned_tokenizer_contract(', source)
        self.assertLess(source.index('from unsloth import FastLanguageModel'), source.index('import torch', source.index('from unsloth import FastLanguageModel')))


if __name__ == "__main__":
    unittest.main(verbosity=2)
