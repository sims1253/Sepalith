"""LoRA-SFT baseline re-score (docs/training/plan-20260927/lora-baseline/rescore_lora.py).

CPU only: the job file under the night-queue rules, the model manifest, the
static checkpoint checks on a tiny synthetic safetensors file, weight
re-verification against the dry-run receipt and the summary table. The real
models are checked by the script's ``--dry-run`` (receipt in
``docs/training/plan-20260927/receipts/LORA-rescore-dev250-dry-run.json``).
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "docs/training/plan-20260927/lora-baseline"


def load():
    spec = importlib.util.spec_from_file_location("rescore_lora", HERE / "rescore_lora.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 130560, "num_hidden_layers": 42,
          "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "intermediate_size": 6144,
          "tie_word_embeddings": False, "bos_token_id": 0, "pad_token_id": 1, "eos_token_id": [1, 130073]}


class ManifestAndJobTests(unittest.TestCase):
    def test_job_file_runs_the_rescore_under_the_night_queue_rules(self):
        from sepalith.ops import night_queue

        module = load()
        job = json.loads((HERE / "jobs/LORA-rescore-dev250.json").read_text())
        night_queue.validate_job(dict(job))
        self.assertEqual(job["id"], "LORA-rescore-dev250")
        self.assertEqual(job["cwd"], "/home/m0hawk/sandbox/Sepalith")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", job["command"])
        self.assertEqual(job["command"][6], "/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python")
        self.assertFalse(job["resumable"])
        self.assertEqual(job["requires"], [])
        script = job["command"].index("-B") + 1
        self.assertEqual(job["command"][script],
                         "/home/m0hawk/sandbox/Sepalith/docs/training/plan-20260927/lora-baseline/rescore_lora.py")
        args = module.parse_args(job["command"][script + 1:])
        self.assertFalse(args.inner or args.no_guard or args.dry_run or args.only)
        self.assertTrue(str(args.recipe).endswith("c04/bound/sft-v2-lr1e-5.json"))
        self.assertTrue(str(args.output_dir).startswith("/home/m0hawk/.local/state/sepalith/campaign-20260915/"))
        self.assertTrue(str(args.dry_run_output).endswith("receipts/LORA-rescore-dev250-dry-run.json"))
        self.assertTrue(str(args.summary_dir).endswith("plan-20260927/lora-baseline/results"))
        # Every model gets its budget; the guard covers the estimate.
        models = module.load_models(HERE / "models.json")
        self.assertLessEqual(len(models) * module.MODEL_BUDGET_SECONDS, args.guard_seconds + 1800)
        self.assertGreaterEqual(args.guard_seconds / 60, job["estimated_minutes"])

    def test_models_are_local_merged_checkpoints_with_recorded_hashes(self):
        module = load()
        models = module.load_models(HERE / "models.json")
        self.assertEqual([m["label"] for m in models],
                         ["theta0-sft1000", "r2-task-global-b-500", "e750", "r2-task-global-c-1000", "control250"])
        for m in models:
            self.assertTrue(m["path"].startswith("/mnt/e/sepalith/"), m["path"])  # never read over the network
            self.assertNotIn("DAT-07-final", m["path"])
            int(m["weights_sha256"], 16)
            self.assertIn("loss", m["training"])
        self.assertEqual([m["label"] for m in module.load_models(HERE / "models.json", "e750,control250")],
                         ["e750", "control250"])
        with self.assertRaises(SystemExit):
            module.load_models(HERE / "models.json", "nope")

    def test_recipe_settings_are_arm_as_gate_settings(self):
        module = load()
        recipe = module.load_recipe(module.DEFAULT_RECIPE)
        self.assertEqual((recipe["cap"], recipe["batch_size"], recipe["kv_cache"]), (1024, 32, "dynamic"))
        self.assertEqual(recipe["panels"]["dev75"]["sha256"], "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035")
        self.assertEqual(recipe["panels"]["dev250"]["sha256"], "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22")
        self.assertEqual(module.ARM_A["recorded"], {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117})

    def test_cli_rules(self):
        module = load()
        with self.assertRaises(SystemExit):
            module.parse_args([])  # GPU run needs --output-dir
        with self.assertRaises(SystemExit):
            module.parse_args(["--output-dir", "/tmp/x", "--cpu-smoke", "e750"])
        with self.assertRaises(SystemExit):
            module.parse_args(["--dry-run", "--threads", "16"])
        self.assertTrue(module.parse_args(["--dry-run"]).dry_run)


class CheckpointCheckTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "model"
        self.dir.mkdir()
        (self.dir / "config.json").write_text(json.dumps(CONFIG))
        for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.dir / name).write_text("{}")
        save_file({"model.embed_tokens.weight": torch.zeros(4, 3, dtype=torch.bfloat16),
                   "lm_head.weight": torch.zeros(4, 3, dtype=torch.bfloat16)}, str(self.dir / "model.safetensors"))
        self.size = (self.dir / "model.safetensors").stat().st_size
        self.model = {"label": "tiny", "path": str(self.dir), "weights_sha256": "0" * 64}
        self.patch = mock.patch.multiple(self.module, WEIGHTS_BYTES=self.size, TENSORS=2, PARAMETERS=24)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_a_complete_checkpoint_passes(self):
        record = self.module.check_model_dir(self.model)
        self.assertEqual((record["tensors"], record["parameters"], record["dtype"]), (2, 24, "BF16"))
        self.assertIsNone(record["files"]["model.safetensors"]["sha256"])

    def test_truncated_weights_fail(self):
        with mock.patch.object(self.module, "WEIGHTS_BYTES", self.size - 4):
            data = (self.dir / "model.safetensors").read_bytes()
            (self.dir / "model.safetensors").write_bytes(data[:-4])
            with self.assertRaisesRegex(ValueError, "truncated"):
                self.module.check_model_dir(self.model)

    def test_lora_keys_adapters_shards_and_configs_fail(self):
        save_file({"model.embed_tokens.weight": torch.zeros(4, 3, dtype=torch.bfloat16),
                   "q_proj.lora_A.weight": torch.zeros(4, 3, dtype=torch.bfloat16)}, str(self.dir / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "lora 1"):
            self.module.check_model_dir(self.model)
        (self.dir / "adapter_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "adapter files"):
            self.module.check_model_dir(self.model)
        (self.dir / "adapter_config.json").unlink()
        (self.dir / "model-00002.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "one model.safetensors"):
            self.module.check_model_dir(self.model)
        (self.dir / "model-00002.safetensors").unlink()
        (self.dir / "config.json").write_text(json.dumps({**CONFIG, "vocab_size": 32000}))
        with self.assertRaisesRegex(ValueError, "config differs"):
            self.module.check_model_dir(self.model)
        (self.dir / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.module.check_model_dir(self.model)

    def test_weights_are_reverified_against_the_dry_run(self):
        weights = self.dir / "model.safetensors"
        actual = hashlib.sha256(weights.read_bytes()).hexdigest()
        model = {**self.model, "weights_sha256": actual}
        stat = weights.stat()
        dry = {"models": {"tiny": {"status": "ok", "weights_sha256": actual, "weights_bytes": stat.st_size,
                                   "weights_mtime_ns": stat.st_mtime_ns}}}
        self.assertTrue(self.module.verify_weights(model, dry)["method"].startswith("dry-run receipt"))
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        self.assertEqual(self.module.verify_weights(model, dry)["method"], "full sha256 in the job")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.module.verify_weights({**model, "weights_sha256": "f" * 64}, dry)


class SummaryTests(unittest.TestCase):
    def test_markdown_lists_complete_failed_and_skipped_models_with_arm_a_first(self):
        module = load()
        counts75 = {"protocol_valid": 74, "cap_hit": 1, "edit_exact": 27, "strict_noop_correct": 24,
                    "strict_noop_false_suggestions": 8}
        counts250 = {"protocol_valid": 234, "cap_hit": 16, "edit_exact": 79, "strict_noop_correct": 38,
                     "strict_noop_false_suggestions": 12}
        d = module.detail({"counts": counts75, "target_nll": 1.02}, {"counts": counts250, "target_nll": 0.80})
        self.assertEqual(d["dev250"]["score250"], 117)
        training = {"run": "r", "base": "b", "adapter": "a", "loss": "l"}
        receipt = {
            "status": "partial", "started": "s", "finished": "f",
            "settings": {"cap": 1024, "batch_size": 32, "kv_cache": "dynamic", "max_sequence_tokens": 16384},
            "recipe": {"path": "/x/sft-v2-lr1e-5.json", "sha256": "de4bcb0a" * 8},
            "panels": {"dev75": {"sha256": "7c144bcd" * 8}, "dev250": {"sha256": "75f3a5ee" * 8}},
            "arm_a": {"name": "arm A 536", "status": "complete", "training": "full", "detail": d,
                      "metrics": module.ARM_A["recorded"]},
            "models": {"a": {"name": "M1", "step": 500, "status": "complete", "seconds": 600, "training": training,
                             "detail": d, "metrics": {"valid": 70, "cap": 3, "edit": 20, "nofp": 5, "score250": 99}},
                       "b": {"name": "M2", "step": 750, "status": "failed", "error": "OOM", "training": training},
                       "c": {"name": "M3", "step": 1000, "status": "skipped", "reason": "deadline", "training": training}},
        }
        text = module.markdown(receipt)
        table = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]
        self.assertEqual(len(table), 5)
        self.assertTrue(table[1].startswith("| arm A 536 | 536 | 74 | 1 | 27 | 8 | 117 | 79 | 38 | 234 | 16 |"))
        self.assertIn("| M1 | 500 | 70 | 3 | 20 | 5 | 99 |", table[2])
        self.assertIn("failed: OOM", table[3])
        self.assertIn("skipped: deadline", table[4])
        self.assertIn("no adoption decision", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
