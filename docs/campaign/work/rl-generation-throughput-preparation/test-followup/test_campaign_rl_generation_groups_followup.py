"""CPU follow-up tests against the patched fixed-ID generation mixin.

The test copies the explicitly supplied patched trainer into a temporary
source-shaped tree, so importing it cannot modify EXEC or the live source.
The fake model returns its received rectangular prompt plus a short terminal
tail; no model weights or CUDA device are used.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest


FROZEN_SOURCE_ROOT = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/"
    "snapshots/f051ebb9c51a5039ec5b3a81852a5f4c32b3284c066afd30c7baf1066e703c97/source"
)
DEFAULT_PATCHED_TRAINER = Path(__file__).resolve().parents[1] / (
    "campaign_rl_train_generation_groups_per_call.py"
)
FALLBACK_PATCHED_TRAINER = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-65574094/docs/campaign/work/"
    "rl-generation-throughput-preparation/campaign_rl_train_generation_groups_per_call.py"
)


def _patched_trainer_path() -> Path:
    value = Path(os.environ.get("RL03_PATCHED_TRAINER", DEFAULT_PATCHED_TRAINER))
    if value.is_file():
        return value
    if FALLBACK_PATCHED_TRAINER.is_file():
        return FALLBACK_PATCHED_TRAINER
    raise unittest.SkipTest(
        "root must copy campaign_rl_train_generation_groups_per_call.py beside this test "
        "or set RL03_PATCHED_TRAINER"
    )


def _load_patched_trainer():
    """Load the patched module with its original package dependencies read-only."""
    if not FROZEN_SOURCE_ROOT.is_dir():
        raise unittest.SkipTest(f"frozen source root is unavailable: {FROZEN_SOURCE_ROOT}")
    temporary = tempfile.TemporaryDirectory(prefix="rl03-followup-trainer-")
    root = Path(temporary.name) / "source"
    training = root / "experiments" / "training"
    training.mkdir(parents=True)
    (root / "packages").symlink_to(FROZEN_SOURCE_ROOT / "packages", target_is_directory=True)
    shutil.copy2(_patched_trainer_path(), training / "campaign_rl_train.py")
    original_training = str(FROZEN_SOURCE_ROOT / "experiments" / "training")
    temporary_training = str(training)
    sys.path.insert(0, temporary_training)
    sys.path.insert(0, original_training)
    spec = importlib.util.spec_from_file_location(
        "campaign_rl_train_rl03_followup", training / "campaign_rl_train.py",
    )
    if spec is None or spec.loader is None:
        temporary.cleanup()
        raise RuntimeError("cannot create patched trainer import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        temporary.cleanup()
        raise
    module._rl03_followup_temporary = temporary
    return module


def _varied_prompt_groups(candidate_count: int = 4) -> list[dict[str, object]]:
    group_ids = (
        [0, 200],
        [0, 201, 202],
        [0, 203, 204, 205],
        [0, 206],
        [0, 207, 208, 209, 210],
        [0, 211, 212],
        [0, 213, 214, 215, 216],
        [0, 217],
    )
    return [
        {"text": "ignored", "ids": list(ids)}
        for ids in group_ids
        for _row in range(candidate_count)
    ]


class PatchedGenerationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from trl.models import unwrap_model_for_generation  # noqa: F401
        except Exception as error:
            raise unittest.SkipTest(f"pinned CPU torch/TRL seam unavailable: {error}")
        cls.torch = torch
        cls.module = _load_patched_trainer()

    @classmethod
    def tearDownClass(cls):
        temporary = getattr(cls.module, "_rl03_followup_temporary", None)
        if temporary is not None:
            temporary.cleanup()

    def _trainer(self, groups_per_call: int):
        torch = self.torch
        module = self.module

        class FakeModel(torch.nn.Module):
            is_gradient_checkpointing = False

            def __init__(self):
                super().__init__()
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append({
                    "input_ids": kwargs["input_ids"].detach().cpu().tolist(),
                    "attention_mask": kwargs["attention_mask"].detach().cpu().tolist(),
                })
                input_ids = kwargs["input_ids"]
                call_index = len(self.calls) - 1
                tail = torch.tensor(
                    [[9000 + call_index * 100 + row, 1] for row in range(input_ids.shape[0])],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                return SimpleNamespace(sequences=torch.cat((input_ids, tail), dim=1))

        class FakeAccelerator:
            device = torch.device("cpu")
            state = SimpleNamespace(deepspeed_plugin=None)

            @staticmethod
            def unwrap_model(model):
                return model

        model = FakeModel()
        trainer = module.FixedIDGRPOTrainerMixin.__new__(module.FixedIDGRPOTrainerMixin)
        trainer.model = model
        trainer.model_wrapped = model
        trainer.accelerator = FakeAccelerator()
        trainer.args = SimpleNamespace(ds3_gather_for_generation=False)
        trainer.processing_class = object()
        trainer.num_generations = 4
        trainer.configure_campaign_runtime(
            generation_guard_factory=lambda _model: nullcontext(),
            prompt_max_tokens=2048,
            completion_max_tokens=192,
            context_max_tokens=2240,
            generation_groups_per_call=groups_per_call,
        )
        return trainer, model

    def test_varied_lengths_p2_and_p4_preserve_prefixes_and_logical_rows(self):
        prompts = _varied_prompt_groups()
        for groups_per_call, expected_calls in ((2, 4), (4, 2)):
            with self.subTest(groups_per_call=groups_per_call):
                trainer, model = self._trainer(groups_per_call)
                prompt_ids, completions, _logprobs, _forward_kwargs = trainer._generate_single_turn(prompts)
                self.assertEqual(len(model.calls), expected_calls)
                self.assertEqual(prompt_ids, [list(row["ids"]) for row in prompts])
                self.assertEqual(len(completions), 32)
                self.assertTrue(all(len(row) == 2 and row[-1] == 1 for row in completions))
                generation = trainer._campaign_last_generation
                self.assertEqual(generation["generation_groups_per_call"], groups_per_call)
                self.assertEqual(generation["generation_call_count"], expected_calls)
                self.assertEqual(
                    [record["group_index"] for record in generation["records"]],
                    [group for group in range(8) for _row in range(4)],
                )
                self.assertEqual(
                    [record["group_row_index"] for record in generation["records"]],
                    list(range(4)) * 8,
                )
                self.assertEqual(
                    [group["call_index"] for group in generation["groups"]],
                    [call for call in range(expected_calls) for _group in range(groups_per_call)]
                    if expected_calls * groups_per_call == 8 else [],
                )
                for call_index, call in enumerate(model.calls):
                    rows = call["input_ids"]
                    masks = call["attention_mask"]
                    self.assertEqual(len(rows), 4 * groups_per_call)
                    self.assertEqual(len(rows), len(masks))
                    width = max(
                        len(prompts[call_index * 4 * groups_per_call + row_index]["ids"])
                        for row_index in range(len(rows))
                    )
                    for row_index, (observed, mask) in enumerate(zip(rows, masks)):
                        expected = prompts[call_index * 4 * groups_per_call + row_index]["ids"]
                        pad = width - len(expected)
                        self.assertEqual(observed, [1] * pad + expected)
                        self.assertEqual(mask, [0] * pad + [1] * len(expected))

    def test_runtime_rejects_missing_and_out_of_set_packing_values(self):
        module = self.module
        trainer = module.FixedIDGRPOTrainerMixin.__new__(module.FixedIDGRPOTrainerMixin)
        for value in (None, 0, 3, 9, True):
            with self.subTest(value=value):
                with self.assertRaises(module.RLTrainError):
                    trainer.configure_campaign_runtime(
                        generation_guard_factory=None,
                        generation_groups_per_call=value,
                    )

    def test_recipe_top_level_and_policy_mismatch_rejects_before_data_reads(self):
        module = self.module
        sampling = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        }
        identity = {
            "parent": {
                "kind": "merged_sft",
                "manifest_sha256": "a" * 64,
                "merged_weights_sha256": "b" * 64,
            },
            "tokenizer": {
                "vocab_size": 130560,
                "bos_id": 0,
                "eos_id": 1,
                "pad_id": 1,
                "native_eog_ids": [1, 130073],
            },
            "renderer": {
                "renderer_id": module.RENDERER_ID,
                "tokenization_policy": module.TOKENIZATION_POLICY,
                "terminal": module.TERMINAL,
                "no_edit": "[NO_EDIT]",
            },
            "source": {"fixture": True},
            "policy": {
                "lora_rank": 16,
                "lora_alpha": 16,
                "target_modules": list(module.TARGET_MODULES),
                "expected_attachments": 294,
                "expected_trainable_parameters": 25116672,
                "prompt_max_tokens": 2048,
                "completion_max_tokens": 192,
                "candidate_count": 4,
                "rollout_rows_per_update": 32,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "generation_groups_per_call": 4,
                "loss_type": "bnpo",
                "scale_rewards": "group",
                "beta": 0,
                "sampling": sampling,
            },
            "schedule": {
                "seed": 3407,
                "sampler_id": module.CampaignRepeatSampler.SAMPLER_ID,
                "generation_batch_size": 32,
                "steps_per_generation": 4,
                "num_iterations": 1,
            },
            "data": {
                "split": "train",
                "admission_status": "admitted",
                "rows_sha256": "c" * 64,
                "context_sha256": "d" * 64,
                "sidecar_artifact_sha256": "d" * 64,
                "selected_ids_sha256": "e" * 64,
                "ordered_ids_sha256": "f" * 64,
                "row_identity_sha256": "0" * 64,
                "selected_ids": ["row-0"],
                "row_identities": [{"id": "row-0"}],
                "row_count": 1,
            },
        }
        recipe = {
            "schema_version": module.TRAIN_SCHEMA_VERSION,
            "identity": identity,
            "generation_groups_per_call": 2,
            "generation_kwargs": sampling,
            "data": {
                "rows_path": "/never/read/rows.jsonl",
                "rows_sha256": "c" * 64,
                "sidecar_path": "/never/read/sidecar.jsonl",
                "sidecar_artifact_sha256": "d" * 64,
                "selected_ids_path": "/never/read/selected.json",
                "selected_ids_sha256": "e" * 64,
            },
        }
        with self.assertRaisesRegex(module.RLTrainError, "differs"):
            module.validate_rl_recipe(copy.deepcopy(recipe))


if __name__ == "__main__":
    unittest.main()

