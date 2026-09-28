from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from profile_orchestrator import (
    EPHEMERAL_CACHE_NAME,
    PROFILE_ROWS,
    PROFILE_STEPS,
    ProfileError,
    StageSpec,
    _safe_manifest_fields,
    build_durable_manifest,
    build_stage_plan,
    clean_environment,
    prepare_run_dir,
    run_stage,
)
from source_inventory import SOURCE_PATHS, _validate_relative


class ProfileOrchestratorTests(unittest.TestCase):
    def test_stage_mapping_is_fixed_and_sequentially_bounded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-profile-plan-") as temp:
            plan = build_stage_plan(
                run=Path(temp) / "run",
                target_manifest_path=Path(temp) / "target.json",
                train_data_path=Path(temp) / "train.jsonl",
                source_root=Path(temp) / "source",
                python_executable=sys.executable,
            )
        self.assertEqual([stage.name for stage in plan], [
            "teacher-cache-profile",
            "native-cuda-smoke",
            "warmstart-trainer-profile",
        ])
        self.assertEqual(plan[0].max_rows, PROFILE_ROWS)
        self.assertNotIn("--allow-bounded-run", plan[0].argv)
        self.assertEqual(plan[-1].max_steps, PROFILE_STEPS)
        self.assertTrue(all(stage.wall_limit_seconds > 0 for stage in plan))

    def test_stage_failure_writes_terminal_receipt_and_propagates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-profile-failure-") as temp:
            run = prepare_run_dir(Path(temp) / "run")
            stage = StageSpec(
                name="failing-stage",
                argv=(sys.executable, "-c", "import sys; sys.exit(7)"),
                cwd=run,
                wall_limit_seconds=10,
            )
            with self.assertRaisesRegex(ProfileError, "failing-stage failed"):
                run_stage(stage, run=run, environment={"PATH": os.environ["PATH"]})
            terminal = json.loads((run / "stages/failing-stage.terminal.json").read_text())
            self.assertEqual(terminal["status"], "failed")
            self.assertEqual(terminal["exit_code"], 7)

    def test_fresh_run_and_durable_cache_exclusion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-profile-artifacts-") as temp:
            run = prepare_run_dir(Path(temp) / "run")
            with self.assertRaisesRegex(ProfileError, "fresh"):
                prepare_run_dir(run)
            cache = run / EPHEMERAL_CACHE_NAME
            cache.mkdir()
            (cache / "shard.bin").write_bytes(b"ephemeral")
            teacher = run / "durable/teacher-profile"
            teacher.mkdir(parents=True)
            (teacher / "teacher_rows.jsonl").write_text('{"id":"tiny"}\n')
            (teacher / "manifest.json").write_text('{}\n')
            result = build_durable_manifest(run)
            paths = {row["path"] for row in result["files"]}
            self.assertTrue(result["teacher_tokens_included"])
            self.assertTrue(result["target_cache_manifest_included"])
            self.assertNotIn(f"{EPHEMERAL_CACHE_NAME}/shard.bin", paths)

    def test_alternate_weights_filename_is_rejected_before_file_load(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-profile-manifest-") as temp:
            model_dir = Path(temp) / "model"
            model_dir.mkdir()
            manifest = {
                "model_dir": str(model_dir),
                "config_file": "config.json",
                "config_sha256": "0" * 64,
                "weights_file": "alternate.safetensors",
                "weights_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(ProfileError, "weights filename"):
                _safe_manifest_fields(manifest, Path(temp) / "manifest.json")

    def test_environment_removes_credentials_and_keeps_profile_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-profile-env-") as temp:
            run = Path(temp) / "run"
            target = {"model_dir": "/explicit/model"}
            inputs = {
                "target": target,
                "public_draft": {"actual_path": "/explicit/public.safetensors"},
            }
            old = os.environ.get("PROFILE_TEST_TOKEN")
            os.environ["PROFILE_TEST_TOKEN"] = "secret-value"
            try:
                env = clean_environment(run=run, inputs=inputs, source_root=Path(temp))
            finally:
                if old is None:
                    os.environ.pop("PROFILE_TEST_TOKEN", None)
                else:
                    os.environ["PROFILE_TEST_TOKEN"] = old
            self.assertNotIn("PROFILE_TEST_TOKEN", env)
            self.assertEqual(env["SEPALITH_DRAFT_CACHE"], str(run / EPHEMERAL_CACHE_NAME))
            self.assertEqual(env["SEPALITH_MINICPM5_TARGET"], "/explicit/model")

    def test_inventory_rejects_nonportable_locations(self) -> None:
        self.assertTrue(all("dependency-overlay" not in path and ".git" not in path for path in SOURCE_PATHS))
        with self.assertRaises(ValueError):
            _validate_relative("dependency-overlay/site-packages/torch.py")
        with self.assertRaises(ValueError):
            _validate_relative(".git/config")


if __name__ == "__main__":
    unittest.main(verbosity=2)
