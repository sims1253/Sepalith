#!/usr/bin/env python3
"""CPU-only contract checks for the prepared RL-03 live recipes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GENERATED = HERE / "generated-rl03-step1000"
REVIEWED_INPUT = ROOT / "docs/campaign/work/rl-smoke-preparation/prepare_rl_two_update_smoke.py"
REVIEWED_INPUT_SHA256 = "eda26a1027c074c70c73a6755369ea4b88d8fa0944d4b5870d1f8f2b5eec614e"
PARENT_SHA256 = "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12"
ADMISSION_SHA256 = "59efb6dc5361569c247101292c875017913b9cca5d13eb05fe1e52e0f1be3abb"
SCHEDULE_SHA256 = "892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48"
SEQUENCE_SHA256 = "d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2"
ROWS_SHA256 = "e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602"
SIDECAR_SHA256 = "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f"
SELECTED_SHA256 = "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d"
DEADLINE = "2026-09-12T18:20:00Z"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


class RL03LiveRecipePreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = read_json(GENERATED / "preparation-result.json")
        cls.recipes = {
            name: read_json(GENERATED / f"rl-two-update-{name}.recipe.json")
            for name in ("uninterrupted", "split-first", "split-resume")
        }

    def test_reviewed_input_is_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(REVIEWED_INPUT.read_bytes()).hexdigest(),
            REVIEWED_INPUT_SHA256,
        )

    def test_common_identity_and_caps_are_exact(self) -> None:
        identities = [recipe["identity"] for recipe in self.recipes.values()]
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[1], identities[2])
        for recipe in self.recipes.values():
            self.assertEqual(recipe["parent_manifest"]["sha256"], PARENT_SHA256)
            self.assertEqual(recipe["rl02_admission"]["sha256"], ADMISSION_SHA256)
            self.assertEqual(recipe["cuda_memory_fraction"], 0.75)
            self.assertEqual(recipe["identity"]["policy"]["cuda_memory_fraction"], 0.75)
            self.assertEqual(recipe["max_attempt_seconds"], 600)
            self.assertEqual(recipe["termination_grace_seconds"], 60)
            self.assertEqual(recipe["checkpoint_reserve_seconds"], 120)
            self.assertEqual(recipe["deadline"], DEADLINE)
            self.assertEqual(recipe["max_steps"], 2)
            self.assertEqual(recipe["full_save_steps"], 1)
            self.assertEqual(recipe["light_save_steps"], 1)
            self.assertEqual(recipe["evaluation_steps"], [1, 2])

    def test_rl_geometry_and_full_schedule_binding(self) -> None:
        policy = self.recipes["uninterrupted"]["identity"]["policy"]
        schedule = self.recipes["uninterrupted"]["identity"]["schedule"]
        data = self.recipes["uninterrupted"]["data"]
        self.assertEqual(
            {key: policy[key] for key in (
                "candidate_count", "rollout_rows_per_update",
                "per_device_train_batch_size", "gradient_accumulation_steps",
                "prompt_max_tokens", "completion_max_tokens", "context_max_tokens",
            )},
            {
                "candidate_count": 4,
                "rollout_rows_per_update": 32,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "prompt_max_tokens": 2048,
                "completion_max_tokens": 192,
                "context_max_tokens": 2240,
            },
        )
        self.assertEqual(schedule["source_draws"], 24000)
        self.assertEqual(schedule["source_draws_per_update"], 8)
        self.assertEqual(schedule["buffer_reuse"], 4)
        self.assertEqual(schedule["generation_batch_size"], 32)
        self.assertEqual(schedule["steps_per_generation"], 4)
        self.assertEqual(schedule["main_update_ceiling"], 3000)
        self.assertEqual(data["source_draw_schedule_sha256"], SCHEDULE_SHA256)
        self.assertEqual(data["source_draw_sequence_sha256"], SEQUENCE_SHA256)
        self.assertEqual(data["rows_sha256"], ROWS_SHA256)
        self.assertEqual(data["sidecar_artifact_sha256"], SIDECAR_SHA256)
        self.assertEqual(data["selected_ids_sha256"], SELECTED_SHA256)

    def test_three_arm_difference_is_only_control_flow_and_paths(self) -> None:
        control = self.recipes["uninterrupted"]
        split_first = self.recipes["split-first"]
        split_resume = self.recipes["split-resume"]
        self.assertEqual(control["decision_steps"], [])
        self.assertEqual(split_first["decision_steps"], [1])
        self.assertEqual(split_resume["decision_steps"], [])
        self.assertIsNone(control["resume_from"])
        self.assertIsNone(split_first["resume_from"])
        self.assertTrue(split_resume["resume_from"].endswith("split-first/archive/full/checkpoint-1"))
        for key in ("identity", "data", "parent_manifest", "rl02_admission", "deadline"):
            self.assertEqual(control[key], split_first[key])
            self.assertEqual(split_first[key], split_resume[key])

    def test_generation_and_preflight_evidence(self) -> None:
        self.assertEqual(self.result["status"], "recipes_emitted_pending_live_smoke")
        self.assertEqual(self.result["data"]["rows"], 8440)
        self.assertEqual(self.result["source_prefix"]["schedule"]["source_draws"], 24000)
        caps = self.result["live_launch_caps"]
        self.assertEqual(caps["attempts"], 3)
        self.assertEqual(caps["max_attempt_seconds_each"], 600)
        self.assertEqual(caps["absolute_deadline"], DEADLINE)
        self.assertFalse(caps["launch_performed"])

        uninterrupted = read_json(GENERATED / "preflight-uninterrupted.json")
        split_first = read_json(GENERATED / "preflight-split-first.json")
        split_resume = read_json(GENERATED / "preflight-split-resume.json")
        for preflight in (uninterrupted, split_first):
            self.assertEqual(preflight["status"], "preflight_pass")
            self.assertEqual(preflight["records"], 8440)
            self.assertTrue(all(not loaded for loaded in preflight["framework_imports"].values()))
        self.assertEqual(split_resume["status"], "entry_failed")
        self.assertIn("checkpoint-1", split_resume["error"])


if __name__ == "__main__":
    unittest.main()
