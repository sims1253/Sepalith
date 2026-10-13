import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace


PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET / "source"
sys.path.insert(0, str(SOURCE))
import full_weight_resume_proof as proof
from trainer_tokenizer_alignment import restore_trainer_eog_alignment


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
        self.assertIn('"train_sampling_strategy": "sequential"', source)
        self.assertIn('trainer.updated_positions == expected_lane_positions', source)
        self.assertIn('restore_pinned_tokenizer_contract(', source)
        self.assertLess(source.index('from unsloth import FastLanguageModel'), source.index('import torch', source.index('from unsloth import FastLanguageModel')))

    def test_actual_transformers_55_accepts_common_cosine_horizon_arguments(self):
        from transformers import TrainingArguments
        with tempfile.TemporaryDirectory() as value:
            args = TrainingArguments(**proof.training_arguments_kwargs(Path(value), 3407))
        self.assertEqual(args.max_steps, 2)
        self.assertEqual(str(args.lr_scheduler_type), "SchedulerType.COSINE")
        self.assertFalse(args.ignore_data_skip)

    def test_runtime_contract_detects_pad_vocab_and_embedding_mutations(self):
        import torch
        vocab = {f"t{index}": index for index in range(130560)}
        vocab["</s>"] = 1
        class Tokenizer:
            bos_token_id = 0; eos_token_id = 1; pad_token_id = 1; pad_token = "</s>"
            def __len__(self): return 130560
            def get_vocab(self): return dict(vocab)
            def convert_ids_to_tokens(self, value): return "</s>" if value == 1 else f"t{value}"
        class Model:
            def __init__(self):
                self.i = SimpleNamespace(weight=torch.ones(2, 2)); self.o = SimpleNamespace(weight=torch.ones(2, 2))
                self.config = SimpleNamespace(bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=1)
                self.generation_config = SimpleNamespace(bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=1)
            def get_input_embeddings(self): return self.i
            def get_output_embeddings(self): return self.o
        model, loaded, reference = Model(), Tokenizer(), Tokenizer()
        embeddings = proof.embedding_identity(model)
        self.assertEqual(proof.assert_runtime_tokenizer(model, loaded, reference, embeddings, "fixture")["pad"], 1)
        loaded.pad_token_id = 2
        with self.assertRaisesRegex(ValueError, "BOS/EOS/PAD"):
            proof.assert_runtime_tokenizer(model, loaded, reference, embeddings, "pad_mutation")
        loaded.pad_token_id = 1; model.i = SimpleNamespace(weight=torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, "embedding"):
            proof.assert_runtime_tokenizer(model, loaded, reference, embeddings, "embedding_mutation")

    def test_trainer_mutation_boundaries_and_serialized_tokenizer_are_checked(self):
        source = (SOURCE / "full_weight_resume_proof.py").read_text()
        self.assertIn('"post_trainer_construction"', source)
        self.assertIn('"train_begin"', source)
        self.assertIn('"first_compute_loss"', source)
        self.assertIn('f"before_checkpoint_{state.global_step}"', source)
        self.assertIn('serialized = assert_serialized_tokenizer(checkpoint', source)
        self.assertIn('"after_training"', source)

    def test_known_trainer_eos_and_pad_alignment_is_narrowly_repaired(self):
        vocab = {f"t{index}": index for index in range(130560)}
        del vocab["t1"]
        del vocab["t130000"]
        vocab["</s>"] = 1
        vocab["<unused_token_477>"] = 130000

        class Tokenizer:
            bos_token_id = 0
            eos_token_id = 1
            eos_token = "</s>"
            pad_token_id = 130000
            _pad_token = "<unused_token_477>"
            padding_side = "right"
            def __len__(self): return 130560
            def get_vocab(self): return dict(vocab)
            def convert_ids_to_tokens(self, value): return "</s>" if value == 1 else f"t{value}"
            @property
            def pad_token(self): return self._pad_token
            @pad_token.setter
            def pad_token(self, value):
                self._pad_token = value
                self.pad_token_id = self.get_vocab()[value]

        class Model:
            def __init__(self):
                self.config = SimpleNamespace(bos_token_id=0, eos_token_id=1, pad_token_id=130000)
                self.generation_config = SimpleNamespace(bos_token_id=0, eos_token_id=[1, 1, 130073], pad_token_id=130000)
            def modules(self): return []

        loaded, reference, model = Tokenizer(), Tokenizer(), Model()
        reference.pad_token = "</s>"
        report = restore_trainer_eog_alignment(model, loaded, reference)
        self.assertEqual(report["before"], {"model_eos": [1], "generation_eos": [1, 1, 130073]})
        self.assertEqual(model.config.eos_token_id, [1, 130073])
        self.assertEqual(model.generation_config.eos_token_id, [1, 130073])
        self.assertEqual(loaded.pad_token_id, 1)
        model.generation_config.eos_token_id = [1, 2, 130073]
        with self.assertRaisesRegex(ValueError, "Unexpected Trainer EOS"):
            restore_trainer_eog_alignment(model, loaded, reference)

    def test_seal_does_not_immediately_reread_full_checkpoint(self):
        source = (SOURCE / "full_weight_resume_proof.py").read_text()
        callback = source[source.index("class SealFullCheckpoint"):source.index("class StopAfterFirstSavedUpdate")]
        self.assertIn("sealed = seal_checkpoint", callback)
        self.assertNotIn("verify_checkpoint(checkpoint", callback)
        self.assertIn("verify_checkpoint(resume", source)
        self.assertIn("manifest = verify_checkpoint(terminal", source)

    def test_source_closure_resolves_protocol_for_tokenizer_contract(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import campaign_tokenizer_contract; import sepalith.campaign_protocol"],
            cwd=SOURCE, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
