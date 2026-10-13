"""CPU contract tests for the unlaunched PRM03 task SFT adapter."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve()
TRAINING = HERE.parents[1] / "source" / "experiments" / "training"
sys.path.insert(0, str(TRAINING))

import campaign_launch  # noqa: E402
import campaign_sft_data  # noqa: E402
import campaign_task_sft  # noqa: E402


FIXTURE = HERE.parent / "fixtures" / "task-target-unit-v1.jsonl"


def _recipe(*, initialization="new_lora_on_cpt_merged_parent"):
    params = {
        "max_steps": 1000,
        "per_device_batch": 4,
        "gradient_accumulation": 4,
        "learning_rate": 2e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "max_sequence_tokens": 4096,
    }
    parent_kind = campaign_task_sft.TARGET_INITIALIZATIONS[initialization]
    policy = {
        "stage": campaign_task_sft.STAGE,
        "initialization": initialization,
        "full_text_labels": False,
        "optimizer": "adamw_torch_fused",
        "dtype": "bfloat16",
        "target_learning_rate": 2e-4,
        "weight_decay": 0.0,
        "lr_schedule": "cosine",
        "warmup_ratio": 0.03,
        "seed": 3407,
        "loss_objective": campaign_task_sft.LOSS_OBJECTIVE,
    }
    identity = {
        "parent": {
            "kind": parent_kind,
            "revision": "synthetic-parent-revision",
            "weights_sha256": "a" * 64,
        },
        "tokenizer": {
            "revision": "synthetic-tokenizer",
            "tokenizer_json_sha256": "b" * 64,
            "BOS": 0,
            "EOS_PAD": 1,
            "native_EOG": [1, 130073],
        },
        "renderer": {"id": "zeta2-prm03-v1", "contract_sha256": "c" * 64},
        "data": {"split_id": "synthetic-task-unit-only"},
        "source": "synthetic-source-unit-only",
        "policy": policy,
        "schedule": copy.deepcopy(params),
    }
    return {
        "schema_version": 1,
        "stage": campaign_task_sft.STAGE,
        "identity": identity,
        "parameters": params,
        "renderer_id": "zeta2-prm03-v1",
        "development_max_new_tokens": 192,
        "development_evaluation_route": "native_final_dev_v1",
        "native_acceptance_required": True,
        "hf_evaluation_is_acceptance": False,
        "evaluator_factory": "campaign_eval:development_evaluator",
        "milestones": [250, 500, 1000],
        "target_gate_start_steps": [0, 250, 500],
        "target_resume_steps": [250, 500],
        "mandatory_stop_steps": [],
        "decision_steps": [],
        "checkpoint": {
            "light_every": 250,
            "full_every": 250,
            "evaluation_steps": [250, 500, 1000],
        },
        "resume_from": None,
    }


def _rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


class TaskSFTContractTests(unittest.TestCase):
    def test_policy_accepts_both_fresh_parent_choices(self):
        self.assertTrue(campaign_task_sft.task_target_only_policy(_recipe()))
        self.assertTrue(campaign_task_sft.task_target_only_policy(
            _recipe(initialization="new_lora_on_midtrain_control")
        ))

    def test_policy_rejects_old_pilot_and_unbound_or_mismatched_stage(self):
        old_recipe = json.loads(
            (HERE.parents[2] / "lead" / "target-only25-a" / "recipe.json").read_text()
        )
        self.assertTrue(campaign_sft_data.target_only_pilot_policy(old_recipe))
        with self.assertRaisesRegex(ValueError, "only task_sft_prm03_v1"):
            campaign_task_sft.task_target_only_policy(old_recipe)

        cases = []
        bad = _recipe()
        bad["parameters"]["learning_rate"] = 5e-5
        cases.append(bad)
        bad = _recipe()
        bad["identity"]["parent"]["kind"] = "stale_sft_adapter"
        cases.append(bad)
        bad = _recipe()
        bad["development_max_new_tokens"] = 193
        cases.append(bad)
        bad = _recipe()
        bad["target_gate_start_steps"] = [250, 500]
        cases.append(bad)
        bad = _recipe()
        bad["decision_steps"] = [123]
        cases.append(bad)
        bad = _recipe()
        bad["identity"]["schedule"]["max_steps"] = 999
        cases.append(bad)
        for candidate in cases:
            with self.assertRaises(ValueError):
                campaign_task_sft.task_target_only_policy(candidate)

    def test_complete_target_rows_are_checked_without_truncation(self):
        recipe = _recipe()
        rows = _rows()
        summary = campaign_task_sft.validate_task_rows(recipe, rows)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["target_tokens_including_eos"], 8)
        self.assertEqual(summary["max_target_tokens_including_eos"], 4)
        self.assertTrue(summary["all_complete_targets_within_192"])

        bad = copy.deepcopy(rows[0])
        bad["input_ids"] = bad["input_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "complete target"):
            campaign_task_sft.validate_task_rows(recipe, [bad])

        bad = copy.deepcopy(rows[0])
        bad["target_body_token_count"] += 1
        with self.assertRaisesRegex(ValueError, "incomplete target metadata"):
            campaign_task_sft.validate_task_rows(recipe, [bad])

        bad = copy.deepcopy(rows[0])
        bad["target_body_tokens"] = [10 + (index % 1000) for index in range(191)]
        bad["target_body_token_count"] = 191
        bad["input_ids"] = [0, 10, 11] + bad["target_body_tokens"] + [130073, 1]
        with self.assertRaisesRegex(ValueError, "exceeds 192"):
            campaign_task_sft.validate_task_rows(recipe, [bad])

    def test_target_only_collator_supervises_complete_tail_and_masks_padding(self):
        batch = campaign_sft_data.target_only_collator(_rows())
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, -100, 50, 51, 130073, 1, -100])
        self.assertEqual(batch["labels"][1].tolist(), [-100, -100, -100, -100, 60, 61, 130073, 1])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 1, 1, 0], [1] * 8])
        self.assertEqual(batch["input_ids"][0, -1].item(), 1)
        self.assertEqual(batch["labels"][0, -1].item(), -100)

    def test_stage_launcher_dispatches_without_loading_runtime(self):
        self.assertEqual(campaign_launch.entrypoint_for_stage(campaign_task_sft.STAGE).name,
                         "campaign_task_sft.py")
        self.assertEqual(campaign_launch.entrypoint_for_stage("cpt_raw_r_v1").name,
                         "campaign_cpt.py")
        self.assertEqual(campaign_launch.entrypoint_for_stage("other").name,
                         "campaign_sft.py")
        self.assertEqual(campaign_launch.training_entrypoint({"stage": campaign_task_sft.STAGE}).name,
                         "campaign_task_sft.py")


if __name__ == "__main__":
    unittest.main()
