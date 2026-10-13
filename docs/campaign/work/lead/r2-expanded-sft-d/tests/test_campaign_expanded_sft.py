"""CPU-only contracts for expanded SFT; no model, CUDA, or final data reads."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve()
TRAINING = HERE.parents[1] / "source" / "experiments" / "training"
sys.path.insert(0, str(TRAINING))

import campaign_expanded_sft as expanded  # noqa: E402
import campaign_launch  # noqa: E402
import campaign_sft  # noqa: E402
from test_campaign_task_sft import _recipe as old_recipe, _rows  # noqa: E402


def record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def recipe(root: Path) -> dict:
    prior = old_recipe()
    previous_recipe_path = root / "previous-recipe.json"
    previous_recipe_path.write_text(json.dumps(prior) + "\n")
    checkpoint = {
        "schema_version": 1, "full": True, "step": expanded.SELECTED_PARENT_STEP,
        "identity": prior["identity"], "files": {},
    }
    checkpoint_path = root / "previous-checkpoint-manifest.json"
    checkpoint_path.write_text(json.dumps(checkpoint) + "\n")
    previous_recipe_record = record(previous_recipe_path)
    checkpoint_record = record(checkpoint_path)
    model_path = root / "model"
    model_path.mkdir()
    model_record = {
        "path": str(model_path / "model.safetensors"),
        "sha256": expanded.SELECTED_PARENT_WEIGHTS_SHA256,
    }
    parent_manifest = {
        "schema_version": expanded.PARENT_MANIFEST_SCHEMA,
        "kind": "merged_task_sft",
        "merged_model_path": str(model_path),
        "merged_weights_sha256": expanded.SELECTED_PARENT_WEIGHTS_SHA256,
        "base_model_revision": prior["identity"]["parent"]["revision"],
        "sft_identity": prior["identity"],
        "sft_checkpoint_manifest": checkpoint,
        "recipe_sha256": previous_recipe_record["sha256"],
        "checkpoint_manifest_sha256": checkpoint_record["sha256"],
        "source_cursor": expanded.SELECTED_PARENT_CURSOR,
        "tokenizer_original_bytes_restored": True,
        "tokenizer": {
            "tokenizer_json_sha256": prior["identity"]["tokenizer"]["tokenizer_json_sha256"],
        },
    }
    manifest_path = root / "parent-manifest.json"
    manifest_path.write_text(json.dumps(parent_manifest) + "\n")
    manifest_record = record(manifest_path)

    result = old_recipe()
    result["stage"] = expanded.STAGE
    result["identity"]["policy"].update({
        "stage": expanded.STAGE,
        "initialization": "new_lora_on_sft_merged_parent",
        "loss_objective": expanded.LOSS_OBJECTIVE,
    })
    result["identity"]["parent"] = {
        "kind": "sft_merged",
        "revision": prior["identity"]["parent"]["revision"],
        "weights_sha256": expanded.SELECTED_PARENT_WEIGHTS_SHA256,
        "sft_merged_manifest_sha256": manifest_record["sha256"],
        "previous_checkpoint_step": expanded.SELECTED_PARENT_STEP,
        "previous_source_cursor": expanded.SELECTED_PARENT_CURSOR,
    }
    result["parameters"]["train_max_target_tokens"] = expanded.TRAIN_MAX_TARGET_TOKENS
    result["identity"]["schedule"] = copy.deepcopy(result["parameters"])
    result["model_path"] = str(model_path)
    result["sft_merged_parent"] = {
        "manifest": manifest_record,
        "previous_recipe": previous_recipe_record,
        "previous_checkpoint_manifest": checkpoint_record,
    }
    result["inputs"] = [manifest_record, previous_recipe_record, checkpoint_record, model_record]
    draws = root / "draws.json"
    draws.write_text(json.dumps({"split_id": result["identity"]["data"]["split_id"]}) + "\n")
    result["draw_schedule"]["path"] = str(draws)
    return result


def long_row(body_length: int) -> dict:
    body = [10 + index % 1000 for index in range(body_length)]
    return {
        "id": f"long-{body_length}",
        "input_ids": [0, 10, 11, *body, 130073, 1],
        "target_start": 3,
        "target_body_tokens": body,
        "target_terminal_tokens": [130073],
        "target_body_token_count": len(body),
        "target_terminal_token_count": 1,
    }


class ExpandedSFTTests(unittest.TestCase):
    def test_new_policy_has_separate_train_and_dev_caps(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            candidate = recipe(Path(temp))
            self.assertTrue(expanded.expanded_target_only_policy(candidate))
            self.assertEqual(candidate["parameters"]["train_max_target_tokens"], 1024)
            self.assertEqual(candidate["development_max_new_tokens"], 192)
            candidate["development_max_new_tokens"] = 1024
            with self.assertRaisesRegex(ValueError, "development output cap"):
                expanded.expanded_target_only_policy(candidate)

    def test_complete_target_over_192_is_admitted_without_truncation(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            candidate = recipe(Path(temp))
            row = long_row(250)
            summary = expanded.validate_expanded_rows(candidate, [row])
            self.assertEqual(summary["max_target_tokens_including_eos"], 252)
            self.assertEqual(row["input_ids"][row["target_start"]:],
                             row["target_body_tokens"] + row["target_terminal_tokens"] + [1])

    def test_over_train_cap_and_malformed_or_truncated_targets_reject(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            candidate = recipe(Path(temp))
            with self.assertRaisesRegex(ValueError, "exceeds 1024 TRAIN"):
                expanded.validate_expanded_rows(candidate, [long_row(1023)])
            malformed = long_row(250)
            malformed["target_body_token_count"] -= 1
            with self.assertRaisesRegex(ValueError, "incomplete target metadata"):
                expanded.validate_expanded_rows(candidate, [malformed])
            truncated = long_row(250)
            truncated["input_ids"].pop(-2)
            with self.assertRaisesRegex(ValueError, "complete target"):
                expanded.validate_expanded_rows(candidate, [truncated])

    def test_parent_lineage_crosschecks_and_old_resume_identity_rejects(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            candidate = recipe(Path(temp))
            expanded.validate_sft_merged_parent(candidate)
            wrong = copy.deepcopy(candidate)
            wrong["identity"]["parent"]["previous_checkpoint_step"] = 250
            with self.assertRaisesRegex(ValueError, "selected step 500"):
                expanded.expanded_target_only_policy(wrong)

            resume = Path(temp) / "resume"
            resume.mkdir()
            old_identity = old_recipe()["identity"]
            (resume / "campaign-state.json").write_text(json.dumps({
                "step": 250, "full": True, "identity": old_identity,
                "sampler": {
                    "split_id": candidate["identity"]["data"]["split_id"],
                    "schedule_sha256": candidate["draw_schedule"]["sha256"],
                    "consumed_draws": 4000,
                },
            }))
            (resume / "trainer_state.json").write_text(json.dumps({"global_step": 250}))
            (resume / "campaign-manifest.json").write_text(json.dumps({
                "step": 250, "full": True, "identity": old_identity,
            }))
            candidate["resume_from"] = str(resume)
            with self.assertRaisesRegex(ValueError, "resume step/full identity"):
                expanded.validate_expanded_admission(candidate, _rows())

    def test_entrypoint_and_runtime_use_standard_gpu_checkpointing(self):
        self.assertEqual(campaign_launch.entrypoint_for_stage(expanded.STAGE).name,
                         "campaign_expanded_sft.py")
        self.assertEqual(campaign_sft.runtime_options({"stage": expanded.STAGE}), {
            "loader_kwargs": {"use_gradient_checkpointing": True},
            "gradient_checkpointing": True,
            "logging_steps": 1,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
