#!/usr/bin/env python3
"""Focused CPU-only tests for the RL-05 bounded first-update verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("verify_rl05_first_update.py")
SPEC = importlib.util.spec_from_file_location("verify_rl05_first_update", SCRIPT)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def _source_ids() -> list[str]:
    return [f"source-{index:02d}" for index in range(8)]


def _source_audit(path: Path) -> None:
    payload = {
        "schema_version": "sepalith.rl03.two-update.source-prefix-audit.v1",
        "prefix_row_ids": _source_ids() + [f"later-{index:02d}" for index in range(8)],
        "schedule": {
            "sha256": VERIFY.SOURCE_SCHEDULE_SHA256,
            "sequence_sha256": VERIFY.SOURCE_SEQUENCE_SHA256,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _generation_row(index: int) -> dict[str, object]:
    generated = [100 + index, 1]
    output_sha = hashlib.sha256(json.dumps(generated, separators=(",", ":")).encode("ascii")).hexdigest()
    geometry = {
        "candidate_count": 4,
        "generation_call_count": 4,
        "generation_call_index": index // 8,
        "generation_group_count": 8,
        "generation_groups_per_call": 2,
        "generation_row_count": 32,
        "group_end": (index // 4) + 1,
        "group_index": index // 4,
        "group_row_index": index % 4,
        "group_start": index // 4,
    }
    return {
        "schema_version": VERIFY.GENERATION_SCHEMA,
        "update": {"global_step": 0, "global_step_before_update": 0, "trl_microstep": index // 8},
        "global_step": 0,
        "source_schedule_sha256": VERIFY.SOURCE_SCHEDULE_SHA256,
        "prompt_ids_sha256": hashlib.sha256(f"prompt-{index // 4}".encode()).hexdigest(),
        "generated_ids": generated,
        "generated_ids_sha256": output_sha,
        "generated_token_count": len(generated),
        "terminal_reason": "eos",
        "padded_after_terminal": 0,
        "group_geometry": geometry,
    }


def _reward_row(index: int, generation: dict[str, object]) -> dict[str, object]:
    return {
        "id": _source_ids()[index // 4],
        "output_ids_sha256": generation["generated_ids_sha256"],
        "generated_tokens": generation["generated_token_count"],
        "reward": 0.25,
        "family": "finish_block",
        "operation": "replace",
        "expected_operation": "replace",
        "canonical_eos": True,
        "cap_hit": False,
        "failure": None,
        "protocol_valid": True,
    }


def _write_core(root: Path, *, include_gradient: bool = True, include_telemetry: bool = True) -> None:
    output = root / "output"
    output.mkdir(parents=True)
    generations = [_generation_row(index) for index in range(32)]
    (output / "generation-records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
    )
    (output / "reward-records.jsonl").write_text(
        "".join(json.dumps(_reward_row(index, generations[index])) + "\n" for index in range(32)),
        encoding="utf-8",
    )
    if include_gradient:
        (output / "gradient-records.jsonl").write_text(json.dumps({
            "schema_version": VERIFY.GRADIENT_SCHEMA,
            "step": 0,
            "global_step": 0,
            "finite": True,
            "nonfinite": False,
            "norm": 0.5,
            "grad_present_count": 2,
            "nonzero_tensor_count": 2,
            "trainable_tensor_count": 2,
            "trainable_scope": "requires_grad LoRA adapter tensors only",
        }) + "\n", encoding="utf-8")
    if include_telemetry:
        giant = b'{"event":"train_begin","padding":"' + b"x" * VERIFY.MAX_JSONL_LINE_BYTES + b'"}\n'
        with (root / "telemetry.jsonl").open("wb") as stream:
            stream.write(giant)
            stream.write(json.dumps({"event": "optimizer_step", "step": 1, "remaining_seconds": 10.0}).encode() + b"\n")
            stream.write(json.dumps({
                "event": "trainer_metrics", "step": 1,
                "metrics": {"grad_norm": 0.5, "reward": 0.25, "reward_std": 0.1},
            }).encode() + b"\n")


def _write_bindings(root: Path, temp: Path) -> tuple[Path, Path, Path, str]:
    recipe = temp / "primary-c.recipe.json"
    recipe.write_text("{}\n", encoding="utf-8")
    launch = temp / "primary-c.command.json"
    launch.write_text(json.dumps([
        "/usr/bin/python3",
        "/snapshot/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source/experiments/training/campaign_rl_launch.py",
        str(recipe),
        "--receipt", str(root / "supervision.json"),
    ]), encoding="utf-8")
    preflight = temp / "primary-c-preflight.json"
    preflight.write_text(json.dumps({
        "status": "preflight_pass",
        "framework_imported": False,
        "geometry": {
            "candidate_count": 4, "rollout_rows_per_update": 32,
            "prompt_groups_per_update": 8, "per_device_train_batch_size": 8,
            "generation_batch_size": 32, "gradient_accumulation_steps": 4,
            "steps_per_generation": 4, "num_iterations": 1,
        },
        "development": {
            "factory": "campaign_eval:development_evaluator",
            "renderer_id": VERIFY.EXPECTED_RENDERER,
            "development_panel": {"sha256": VERIFY.EXPECTED_DEV_SHA256, "rows": 75},
            "development_max_new_tokens": 512,
            "parameters": {"max_sequence_tokens": 4096},
            "development_case_ids": [f"dev-{i:02d}" for i in range(75)],
        },
    }), encoding="utf-8")
    return launch, preflight, recipe, hashlib.sha256(recipe.read_bytes()).hexdigest()


class FirstUpdateVerifierTests(unittest.TestCase):
    def test_first_update_passes_and_skips_giant_telemetry_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "RL-primary-p2-c"
            _write_core(root)
            source = temp / "source-prefix-audit.json"
            _source_audit(source)
            launch, preflight, recipe, recipe_sha = _write_bindings(root, temp)
            result = VERIFY.review(
                root,
                source_prefix_path=source,
                host_supervision=temp / "host-supervision",
                launch_path=launch,
                preflight_path=preflight,
                expected_recipe_sha256=recipe_sha,
                admission_path=None,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["first_update"]["status"], "pass")
            self.assertEqual(result["rewards"]["rows"], 32)
            self.assertEqual(result["rewards"]["source_ids"][:4], ["source-00"] * 4)
            self.assertEqual(result["telemetry"]["oversize_lines_skipped"], 1)
            self.assertEqual(result["telemetry"]["event_counts"]["optimizer_step"], 1)

    def test_missing_gradient_stays_pending_until_first_update_is_proven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "RL-primary-p2-c"
            _write_core(root, include_gradient=False, include_telemetry=False)
            source = temp / "source-prefix-audit.json"
            _source_audit(source)
            launch, preflight, recipe, recipe_sha = _write_bindings(root, temp)
            result = VERIFY.review(
                root,
                source_prefix_path=source,
                host_supervision=temp / "host-supervision",
                launch_path=launch,
                preflight_path=preflight,
                expected_recipe_sha256=recipe_sha,
                admission_path=None,
            )
            self.assertEqual(result["status"], "pending")
            self.assertEqual(result["gradient"]["status"], "pending")
            self.assertEqual(result["telemetry"]["status"], "pending")

    def test_compact_admission_can_replace_large_entry_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "RL-primary-p2-memory-a"
            _write_core(root)
            source = temp / "source-prefix-audit.json"
            _source_audit(source)
            launch, _preflight, recipe, recipe_sha = _write_bindings(root, temp)
            admission = temp / "admission.json"
            admission.write_text(json.dumps({
                "status": "accepted_memory_mitigation_and_launch_admission",
                "identity_sha256": "a" * 64,
                "recipe": {"path": str(recipe), "sha256": recipe_sha},
                "source": {"snapshot_id": VERIFY.SOURCE_SHA256},
            }), encoding="utf-8")
            result = VERIFY.review(
                root,
                source_prefix_path=source,
                host_supervision=temp / "host-supervision",
                launch_path=launch,
                preflight_path=None,
                expected_recipe_sha256=None,
                admission_path=admission,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["bindings"]["status"], "pass")
            self.assertIsNone(result["bindings"]["preflight_path"])

    def test_guard_stop_without_core_artifacts_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "RL-primary-p2-c"
            (root / "output").mkdir(parents=True)
            guard = temp / "host-supervision"
            guard.mkdir()
            (guard / "terminal.json").write_text(json.dumps({
                "status": "stopped_or_failed",
                "reason": "host_free_memory_below_floor",
                "child_exit_code": -15,
            }), encoding="utf-8")
            source = temp / "source-prefix-audit.json"
            _source_audit(source)
            launch, preflight, recipe, recipe_sha = _write_bindings(root, temp)
            result = VERIFY.review(
                root,
                source_prefix_path=source,
                host_supervision=guard,
                launch_path=launch,
                preflight_path=preflight,
                expected_recipe_sha256=recipe_sha,
                admission_path=None,
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["host_guard"]["reason"], "host_free_memory_below_floor")

    def test_ordered_output_join_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "RL-primary-p2-c"
            _write_core(root)
            reward_path = root / "output" / "reward-records.jsonl"
            rows = [json.loads(line) for line in reward_path.read_text().splitlines()]
            rows[3]["output_ids_sha256"] = "0" * 64
            reward_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            generation, generation_result = VERIFY.review_generation(root)
            _, rewards = VERIFY.review_rewards(root, generation, [item for item in _source_ids() for _ in range(4)])
            self.assertEqual(generation_result["status"], "pass")
            self.assertEqual(rewards["status"], "fail")
            self.assertTrue(any("output SHA mismatch" in failure for failure in rewards["failures"]))


if __name__ == "__main__":
    unittest.main()
