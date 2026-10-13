"""Focused CPU checks for the campaign SFT loss and panel state seam."""
import os
import random
from types import SimpleNamespace
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from campaign_eval import case_evaluation_guard
from campaign_sft import clear_generation_markers, sequential_sft_trainer_class, training_configuration_guard

torch.set_num_threads(1)


class _EmptyLogits:
    def __getattr__(self, name):
        raise AssertionError(f"training path touched EMPTY_LOGITS.{name}")


class _FusedLossModel:
    def __init__(self):
        self.loss = torch.tensor(2.5, requires_grad=True)
        self.seen = None

    def __call__(self, **inputs):
        self.seen = inputs
        return {"loss": self.loss, "logits": _EmptyLogits()}


class _TrainerAccelerator:
    parallelism_config = None


class _HookModel:
    def __init__(self):
        self.training = True
        self._flag_for_generation = True
        self.calls = []
        self.layer = SimpleNamespace(gradient_checkpointing="unsloth")
        self.config = SimpleNamespace(use_cache=True, bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=1)
        self.generation_config = SimpleNamespace(use_cache=True, bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=1)
        self._saved_temp_tokenizer = SimpleNamespace(
            padding_side="right", pad_token="</s>", pad_token_id=1,
            bos_token_id=0, eos_token_id=1,
        )

    def modules(self):
        return [self, self.layer]

    def for_training(self, *, use_gradient_checkpointing):
        self.calls.append(use_gradient_checkpointing)
        self.training = True
        self._flag_for_generation = False

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode
        return self


class CampaignSFTIntegrationTests(unittest.TestCase):
    def test_training_uses_model_fused_loss_without_reading_empty_logits(self):
        trainer_type = sequential_sft_trainer_class()
        trainer = trainer_type.__new__(trainer_type)
        trainer.accelerator = _TrainerAccelerator()
        trainer.label_smoother = None
        trainer.compute_loss_func = None
        trainer.model_accepts_loss_kwargs = False
        trainer.args = SimpleNamespace(average_tokens_across_devices=False, n_gpu=0)
        model = _FusedLossModel()
        labels = torch.tensor([[0, 4, 1]])

        result = trainer.compute_loss(model, {"input_ids": labels.clone(), "labels": labels})

        self.assertIs(result, model.loss)
        self.assertIs(model.seen["labels"], labels)

    def test_case_guard_calls_supported_restore_and_restores_all_rng(self):
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        model = _HookModel()

        with case_evaluation_guard(model):
            random.random()
            np.random.random()
            torch.rand(4)
            model.training = False
            model._flag_for_generation = True
            model.config.use_cache = False
            model.config.bos_token_id = 9
            model.config.eos_token_id = [10]
            model.config.pad_token_id = 11
            model.generation_config.use_cache = False
            model.generation_config.bos_token_id = 12
            model.generation_config.eos_token_id = [13]
            model.generation_config.pad_token_id = 14
            model._saved_temp_tokenizer.padding_side = "left"
            model._saved_temp_tokenizer.pad_token = "<unused_token_0>"
            model._saved_temp_tokenizer.pad_token_id = 99
            model._saved_temp_tokenizer.bos_token_id = 98
            model._saved_temp_tokenizer.eos_token_id = 97

        self.assertEqual(model.calls, ["unsloth"])
        self.assertFalse(model.training)
        self.assertFalse(model._flag_for_generation)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(model.config.bos_token_id, 0)
        self.assertEqual(model.config.eos_token_id, [1, 130073])
        self.assertEqual(model.config.pad_token_id, 1)
        self.assertTrue(model.generation_config.use_cache)
        self.assertEqual(model.generation_config.bos_token_id, 0)
        self.assertEqual(model.generation_config.eos_token_id, [1, 130073])
        self.assertEqual(model.generation_config.pad_token_id, 1)
        self.assertEqual(model._saved_temp_tokenizer.padding_side, "right")
        self.assertEqual(model._saved_temp_tokenizer.pad_token, "</s>")
        self.assertEqual(model._saved_temp_tokenizer.pad_token_id, 1)
        self.assertEqual(model._saved_temp_tokenizer.bos_token_id, 0)
        self.assertEqual(model._saved_temp_tokenizer.eos_token_id, 1)
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before_numpy[0])
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        self.assertEqual(after_numpy[2:], before_numpy[2:])
        torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)

    def test_training_guard_handles_peft_forwarded_generation_marker(self):
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=16,
        ))
        model = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ))
        underlying = model.base_model.model
        underlying._flag_for_generation = True
        self.assertTrue(hasattr(model, "_flag_for_generation"))
        self.assertNotIn("_flag_for_generation", model.__dict__)

        def fragile_unsloth_cleanup(actual, *, use_gradient_checkpointing):
            # Mirrors the pinned hook's hasattr-then-del traversal.  PEFT's
            # forwarded marker would raise if the guard did not remove the
            # direct owner first.
            current = actual
            while hasattr(current, "model"):
                if hasattr(current, "_flag_for_generation"):
                    del current._flag_for_generation
                current = current.model
            if hasattr(current, "_flag_for_generation"):
                del current._flag_for_generation

        self.assertEqual(clear_generation_markers(model), 1)
        underlying._flag_for_generation = True
        with training_configuration_guard(model, fragile_unsloth_cleanup):
            pass
        self.assertFalse(hasattr(model, "_flag_for_generation"))


if __name__ == "__main__":
    unittest.main()
