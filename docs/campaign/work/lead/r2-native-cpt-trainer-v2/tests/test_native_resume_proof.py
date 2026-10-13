import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "source/experiments/training"
sys.path.insert(0, str(TRAINING))
import full_weight_cpt_trainer as trainer
spec = importlib.util.spec_from_file_location("native_resume_proof_tested", ROOT / "native_resume_proof.py")
proof = importlib.util.module_from_spec(spec); sys.modules[spec.name] = proof; spec.loader.exec_module(proof)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeResumeProofTests(unittest.TestCase):
    def test_actual_transformers_hybrid_resume_and_failure_recovery(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temporary:
            result = proof.run_proof(Path(temporary) / "proof")
            self.assertEqual(result["status"], "pass")
            self.assertTrue(all(result["comparisons"].values()))
            self.assertEqual(result["optimizer_dispatch"]["counts"], {"adamw": 4, "aurora": 2, "muon": 2})
            self.assertTrue(result["failure_recovery"]["prior_native_survives"])
            self.assertTrue(result["failure_recovery"]["prior_durable_survives"])
            self.assertTrue(result["failure_recovery"]["failed_destination_hidden"])

    def test_execution_stop_194_to_322_and_boundary_action(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temporary:
            root = Path(temporary); recipe_path = root / "recipe.json"
            recipe = {"transition": {"global_optimizer_step_offset": 66}, "runtime": {"max_steps": 11443, "mandatory_stop_step": 194, "checkpoint_every": 128}}
            recipe_path.write_text(json.dumps(recipe, sort_keys=True) + "\n")
            admission = root / "execution.json"
            admission.write_text(json.dumps({"schema": "sepalith.sft11.native-cpt-execution-stop.v1", "status": "admitted", "launch_authorized": True, "bound_recipe_sha256": sha(recipe_path), "resume_global_step": 194, "stop_at_global_step": 322}, sort_keys=True) + "\n")
            checked = trainer.validate_execution_stop(recipe_path, recipe, 194, admission)
            self.assertEqual(checked["stop_at_global_step"], 322)
            self.assertFalse(trainer.milestone_action(321, 194, recipe["runtime"], 66, 322)["stop"])
            action = trainer.milestone_action(322, 194, recipe["runtime"], 66, 322)
            self.assertEqual(action, {"save": True, "stop": True, "cursor": 4096})

    def test_execution_stop_wrong_resume_range_and_cadence_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temporary:
            root = Path(temporary); path = root / "recipe"; recipe = {"transition": {"global_optimizer_step_offset": 66}, "runtime": {"max_steps": 1000, "checkpoint_every": 128}}
            path.write_text('{}\n')
            base = {"schema": "sepalith.sft11.native-cpt-execution-stop.v1", "status": "admitted", "launch_authorized": True, "bound_recipe_sha256": sha(path), "resume_global_step": 194}
            for stop, message in ((194, "outside continuation"), (1001, "outside continuation"), (323, "scheduled full checkpoint")):
                admission = root / f"a-{stop}.json"; admission.write_text(json.dumps({**base, "stop_at_global_step": stop}) + "\n")
                with self.assertRaisesRegex(ValueError, message): trainer.validate_execution_stop(path, recipe, 194, admission)

    def test_real_source_frontdoor_and_transient_retention(self):
        record = {"runtime_source": {"manifest_path": str(ROOT / "source-manifest.json"), "manifest_sha256": sha(ROOT / "source-manifest.json")}}
        trainer.verify_source(record)
        recipe = {"runtime": {"micro_batch": 1, "gradient_accumulation": 16, "max_steps": 3, "learning_rate": 3e-6, "scheduler": "constant_with_warmup", "warmup_steps": 2}, "seed": 1}
        self.assertEqual(trainer.training_arguments_kwargs(recipe, "/tmp/unused")["save_total_limit"], 2)

    def test_reusable_native_data_receipt_pin(self):
        path = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json")
        self.assertEqual(sha(path), "a72c1b699fe931e0db0fa2ddfc96a43486e74fc962802e8a541540e16596b88d")
        value = json.loads(path.read_text())
        self.assertEqual(value["status"], "staged_immutable_verified")


if __name__ == "__main__": unittest.main(verbosity=2)
