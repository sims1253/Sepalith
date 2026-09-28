from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from profile_orchestrator import (
    DEEPSPEC_REVISION,
    EPHEMERAL_CACHE_NAME,
    NON_PROFILE_RESERVE_SECONDS,
    PROFILE_LIMIT_SECONDS,
    PROFILE_ROWS,
    PROFILE_STEPS,
    PUBLIC_DRAFT_CONFIG_SHA256,
    STAGE_LIMITS_SECONDS,
    TOTAL_WINDOW_SECONDS,
    ProfileError,
    StageSpec,
    _safe_manifest_fields,
    build_durable_manifest,
    build_stage_plan,
    clean_environment,
    prepare_run_dir,
    run_stage,
    validate_trainer_stage,
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
        self.assertEqual(sum(STAGE_LIMITS_SECONDS.values()), PROFILE_LIMIT_SECONDS)
        self.assertEqual(TOTAL_WINDOW_SECONDS - PROFILE_LIMIT_SECONDS, NON_PROFILE_RESERVE_SECONDS)
        self.assertLessEqual(sum(stage.wall_limit_seconds for stage in plan), PROFILE_LIMIT_SECONDS)
        self.assertIn("r2-draft-target-runtime-v3", plan[0].argv[1])

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
            for transient in ("tmp", "hf-cache", "xdg-cache", "torch-cache", "triton-cache"):
                path = run / transient
                path.mkdir()
                (path / "cache.bin").write_bytes(b"transient")
            teacher = run / "durable/teacher-profile"
            teacher.mkdir(parents=True)
            (teacher / "teacher_rows.jsonl").write_text('{"id":"tiny"}\n')
            (teacher / "manifest.json").write_text('{}\n')
            result = build_durable_manifest(run)
            paths = {row["path"] for row in result["files"]}
            self.assertTrue(result["teacher_tokens_included"])
            self.assertTrue(result["target_cache_manifest_included"])
            self.assertNotIn(f"{EPHEMERAL_CACHE_NAME}/shard.bin", paths)
            for transient in ("tmp", "hf-cache", "xdg-cache", "torch-cache", "triton-cache"):
                self.assertFalse(any(path.startswith(transient + "/") for path in paths))

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
                "deepspec_revision": DEEPSPEC_REVISION,
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
            self.assertEqual(env.get("HOME"), os.environ.get("HOME"))
            for key in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR"):
                self.assertTrue(Path(env[key]).is_relative_to(run))
                self.assertTrue(Path(env[key]).is_dir())

    def test_inventory_rejects_nonportable_locations(self) -> None:
        self.assertTrue(all("dependency-overlay" not in path and ".git" not in path for path in SOURCE_PATHS))
        self.assertTrue(any(path.endswith("r2-draft-target-runtime-v3/target_runtime.py") for path in SOURCE_PATHS))
        with self.assertRaises(ValueError):
            _validate_relative("dependency-overlay/site-packages/torch.py")
        with self.assertRaises(ValueError):
            _validate_relative(".git/config")

    def test_trainer_steps_must_be_contiguous(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sepalith-profile-steps-") as temp:
            log = Path(temp) / "trainer.log"
            log.write_text("step=1/8\nstep=1/8\n", encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "contiguous"):
                validate_trainer_stage(Path(temp), {"log": str(log), "telemetry": str(Path(temp) / "none.jsonl")})

    def test_loader_and_public_config_pins_are_explicit(self) -> None:
        target_source = Path(__file__).parents[1] / "r2-draft-target-runtime-v3" / "target_runtime.py"
        text = target_source.read_text(encoding="utf-8")
        self.assertIn("trust_remote_code=False", text)
        self.assertIn("use_safetensors=True", text)
        warmstart = Path(__file__).parents[1] / "lead" / "r2-draft-cloud-runtime-preparation" / "warmstart_trainer.py"
        # This test is source-only: it does not open a draft or target artifact.
        self.assertEqual(PUBLIC_DRAFT_CONFIG_SHA256, "bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b")
        self.assertIn("PUBLIC_CONFIG_SHA256", warmstart.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
