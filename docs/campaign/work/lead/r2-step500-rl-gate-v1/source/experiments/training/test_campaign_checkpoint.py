"""CPU checks for archive integrity and real Trainer optimizer/RNG resume."""
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

from campaign_checkpoint import checkpoint_callback, preserve_random_state, verify_checkpoint
from campaign_control import control_callback
from campaign_sft import training_configuration_guard

torch.set_num_threads(1)
IDENTITY = {
    "parent": "cpu-test-random-llama-seed77", "tokenizer": "integer-fixture-v1",
    "renderer": "test-token-sequence-v1", "data": "32-fixed-sequences-v1",
    "source": "checkpoint-regression-test", "policy": "cpu-full-text-lora",
    "schedule": {"max_steps": 8, "seed": 77, "lr": 0.003, "scheduler": "cosine"},
}


def make_model():
    set_seed(77)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        attention_dropout=0.2, use_cache=False,
    ))
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.2,
                                          target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))


def evaluation(model, tokenizer, checkpoint, step):
    # A development evaluator must not perturb future sampler/dropout randomness.
    random.random()
    np.random.random(8)
    torch.rand(8)
    return {"case_ids": ["cpu-fixture"], "denominators": {"fixtures": 1}, "step": step}


class StopAtFour(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 4:
            control.should_training_stop = True
        return control


def trainer(root, *, stop=False, evaluate=True, campaign_control=None):
    root = Path(root)
    data = []
    for i in range(32):
        ids = [2 + i % 20, 3 + i % 20, 4 + i % 20, 1]
        data.append({"input_ids": ids, "attention_mask": [1] * 4, "labels": ids})
    callback = checkpoint_callback(identity=IDENTITY, archive_root=root / "archive", tokenizer=None,
                                   light_every=2, full_every=4,
                                   evaluator=evaluation if evaluate else None,
                                   allow_pending_evaluation=not evaluate,
                                   evaluation_allowed=(campaign_control.evaluation_allowed
                                                       if campaign_control else None))
    callbacks = ([campaign_control] if campaign_control else []) + [callback] + ([StopAtFour()] if stop else [])
    return Trainer(model=make_model(), train_dataset=data, callbacks=callbacks,
                   args=TrainingArguments(
                       output_dir=str(root / "trainer"), use_cpu=True, max_steps=8,
                       per_device_train_batch_size=2, gradient_accumulation_steps=2,
                       learning_rate=0.003, lr_scheduler_type="cosine", seed=77, data_seed=77,
                       save_strategy="steps", save_steps=4, save_total_limit=1,
                       logging_strategy="steps", logging_steps=1, report_to="none", disable_tqdm=True,
                       optim="adamw_torch", dataloader_num_workers=0,
                   ))


class CheckpointTests(unittest.TestCase):
    def test_generation_mode_cleanup_preserves_checkpointing_and_rng_on_failure(self):
        model = make_model()
        layer = next(m for m in model.modules() if hasattr(m, "gradient_checkpointing"))
        layer.gradient_checkpointing = "unsloth"
        model.config.use_cache = False
        model._saved_temp_tokenizer = SimpleNamespace(padding_side="right")
        parameter = next(model.parameters())
        before_rng = torch.get_rng_state().clone()
        calls = []

        def restore_mode(actual, *, use_gradient_checkpointing):
            calls.append(use_gradient_checkpointing)
            del parameter._fast_lora
            if hasattr(actual, "_flag_for_generation"):
                del actual._flag_for_generation
            layer.gradient_checkpointing = True
            actual.config.use_cache = True

        with self.assertRaisesRegex(RuntimeError, "failed readout"):
            with preserve_random_state(model):
                with training_configuration_guard(model, restore_mode):
                    self.assertFalse(model.training)
                    layer.gradient_checkpointing = False
                    model.config.use_cache = True
                    model._saved_temp_tokenizer.padding_side = "left"
                    parameter._fast_lora = "stale inference weights"
                    model._flag_for_generation = True
                    torch.rand(10)
                    raise RuntimeError("failed readout")
        self.assertEqual(calls, ["unsloth"])
        self.assertEqual(layer.gradient_checkpointing, "unsloth")
        self.assertFalse(model.config.use_cache)
        self.assertEqual(model._saved_temp_tokenizer.padding_side, "right")
        self.assertFalse(hasattr(parameter, "_fast_lora"))
        self.assertFalse(hasattr(model, "_flag_for_generation"))
        self.assertTrue(model.training)
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_interruption_restores_optimizer_scheduler_rng_and_data_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            full = trainer(d / "full")
            full.train()
            partial = trainer(d / "resume", stop=True)
            partial.train()
            self.assertEqual(partial.state.global_step, 4)
            checkpoint = d / "resume/archive/full/checkpoint-4"
            manifest = verify_checkpoint(checkpoint, IDENTITY, require_full=True)
            self.assertIn("optimizer.pt", manifest["files"])
            self.assertIn("rng_state.pth", manifest["files"])
            resumed = trainer(d / "resume")
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 8)
            for name, value in full.model.state_dict().items():
                if "lora_" in name:
                    torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            full_losses = [x["loss"] for x in full.state.log_history if "loss" in x]
            resumed_losses = [x["loss"] for x in resumed.state.log_history if "loss" in x]
            self.assertEqual(full_losses, resumed_losses)
            self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            # Trainer prunes locally; both full archives and all light adapters survive.
            self.assertFalse((d / "resume/trainer/checkpoint-4").exists())
            verify_checkpoint(checkpoint, IDENTITY, require_full=True)
            for step in (2, 4, 6, 8):
                verify_checkpoint(d / f"resume/archive/adapters/checkpoint-{step}", IDENTITY)
            self.assertFalse(torch.cuda.is_initialized())

    def test_eval_preserves_training_rng_and_bad_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with_eval = trainer(d / "eval")
            with_eval.train()
            without_eval = trainer(d / "no-eval", evaluate=False)
            without_eval.train()
            for name, value in with_eval.model.state_dict().items():
                if "lora_" in name:
                    torch.testing.assert_close(value, without_eval.model.state_dict()[name], rtol=0, atol=0)
            checkpoint = d / "eval/archive/full/checkpoint-8"
            wrong_identity = dict(IDENTITY, renderer="different-renderer")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                verify_checkpoint(checkpoint, wrong_identity, require_full=True)
            with self.assertRaisesRegex(ValueError, "lightweight"):
                verify_checkpoint(d / "eval/archive/adapters/checkpoint-8", IDENTITY, require_full=True)
            (checkpoint / "optimizer.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                verify_checkpoint(checkpoint, IDENTITY, require_full=True)

    def test_budget_or_decision_stop_saves_between_regular_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            full = trainer(d / "full")
            full.train()
            for reason, stop_step in (("lead_decision", 5), ("deadline", 2)):
                root = d / reason

                class Clock:
                    value = 0

                    def now(self):
                        return self.value

                    def tick(self):
                        self.value += 1
                        return self.value

                clock = Clock()
                budget = control_callback(
                    telemetry_path=root / "attempt-1.jsonl", identity=IDENTITY,
                    deadline="1970-01-01T00:00:10Z" if reason == "deadline" else "2099-01-01T00:00:00Z",
                    reserve_seconds=5, stop_steps=(5,), clock=clock.now, monotonic=clock.tick,
                    resource_probe=lambda: {"measured_value": 123, "scope": "synthetic test"},
                )
                partial = trainer(root, campaign_control=budget)
                partial.train()
                self.assertEqual(partial.state.global_step, stop_step)
                self.assertEqual(budget.stop_reason, reason)
                checkpoint = root / f"archive/full/checkpoint-{stop_step}"
                verify_checkpoint(checkpoint, IDENTITY, require_full=True)
                readout = json.loads((root / f"archive/evaluations/step-{stop_step}.json").read_text())
                self.assertEqual(readout["status"].startswith("deferred"), reason == "deadline")
                events = [json.loads(line) for line in (root / "attempt-1.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["event"], "train_end")
                self.assertEqual(events[-1]["stop_reason"], reason)
                step_events = [event for event in events if event["event"] == "optimizer_step"]
                self.assertEqual(len(step_events), stop_step)
                self.assertTrue(all(event["resources"] == {"measured_value": 123, "scope": "synthetic test"}
                                    for event in step_events))
                resumed = trainer(root)
                resumed.train(resume_from_checkpoint=str(checkpoint))
                for name, value in full.model.state_dict().items():
                    if "lora_" in name:
                        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
