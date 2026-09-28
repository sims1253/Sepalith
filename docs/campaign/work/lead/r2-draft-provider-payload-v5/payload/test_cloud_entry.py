from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import artifact_upload as uploader
import cloud_entry as entry
import download_staging as downloader
from download_staging import stage_downloads, validate_download_binding
from managed_python import bootstrap_commands, validate_candidate_requirements
from stage_durable import stage


NOW = entry.utc("2026-09-14T00:05:00Z")


def binding() -> dict:
    run_id = "a" * 32
    return {
        "schema": 2,
        "admitted": True,
        "run_id": run_id,
        "image_uri": entry.IMAGE,
        "instance_type": "g5.2xlarge",
        "max_retries": 0,
        "provider_timeout_seconds": 6900,
        "watchdog_timeout_seconds": 7200,
        "watchdog_armed_at_utc": "2026-09-14T00:00:00Z",
        "watchdog_armed_receipt_sha256": "b" * 64,
        "absolute_deadline_utc": "2026-09-14T02:00:00Z",
        "artifact_repo": "scholzmx/sepalith-lora",
        "artifact_prefix": "r2-draft-profile/" + run_id,
        "budget": copy.deepcopy(entry.EXPECTED_BUDGET),
        "install_dependencies": True,
        "run_root": "runs",
        "bootstrap": {
            "uv_version": entry.UV_VERSION,
            "uv_tools_root": "runtime/bootstrap-tools",
            "managed_python_root": "runtime/python",
            "venv_root": "runtime/venv",
            "uv_cache_root": "runtime/uv-cache",
        },
        "downloads": {
            "manifest_destination": "inputs/target-manifest.json",
            "target": {
                "repo_id": entry.PRIVATE_TARGET_REPO,
                "revision": entry.PRIVATE_TARGET_REVISION,
                "prefix": entry.PRIVATE_TARGET_PREFIX,
                "destination": "inputs/target-model",
                "files": dict(entry.TARGET_ARTIFACT_PINS),
            },
            "public": {
                "repo_id": entry.PUBLIC_DRAFT_REPO,
                "revision": entry.PUBLIC_DRAFT_REVISION,
                "destination": "inputs/public-draft",
                "files": {
                    "model.safetensors": entry.PUBLIC_DRAFT_WEIGHTS_SHA256,
                    "config.json": entry.PUBLIC_DRAFT_CONFIG_SHA256,
                },
            },
        },
        "staging": {
            "root": "stage",
            "requirements_candidate": "requirements-cloud-candidate.txt",
            "requirements_resolved": "requirements-cloud-resolved.txt",
            "source_root": "source",
            "target_manifest": "inputs/target-manifest.json",
            "target_model_dir": "inputs/target-model",
            "train_data": "inputs/train.jsonl",
            "public_draft_weights": "inputs/public-draft/model.safetensors",
            "public_draft_config": "inputs/public-draft/config.json",
        },
        "pins": {
            "deepspec_revision": entry.DEEPSPEC_REVISION,
            "train_sha256": entry.TRAIN_SOURCE_SHA256,
            "public_draft_weights_sha256": entry.PUBLIC_DRAFT_WEIGHTS_SHA256,
            "public_draft_config_sha256": entry.PUBLIC_DRAFT_CONFIG_SHA256,
            "target_candidate_merged_weights_sha256": entry.TARGET_CANDIDATE_MERGED_WEIGHTS_SHA256,
            "target_manifest_sha256": "c" * 64,
        },
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeAPI:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.counter = 0

    def repo_info(self, **_kwargs):
        return SimpleNamespace(private=True)

    def file_exists(self, *, filename, **_kwargs):
        return filename in self.files

    def _commit(self):
        self.counter += 1
        return SimpleNamespace(oid=f"commit-{self.counter}")

    def upload_file(self, *, path_or_fileobj, path_in_repo, **_kwargs):
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()
        return self._commit()

    def upload_folder(self, *, folder_path, path_in_repo, **_kwargs):
        root = Path(folder_path)
        for path in root.rglob("*"):
            if path.is_file():
                self.files[path_in_repo + "/" + path.relative_to(root).as_posix()] = path.read_bytes()
        return self._commit()

    def list_repo_tree(self, *, path_in_repo, **_kwargs):
        prefix = path_in_repo.rstrip("/") + "/"
        for name, value in sorted(self.files.items()):
            if name.startswith(prefix):
                yield SimpleNamespace(
                    path=name,
                    size=len(value),
                    lfs=SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()),
                )


def fake_download(api: FakeAPI):
    def download(repo, *, filename, revision, local_dir):
        del repo, revision
        destination = Path(local_dir) / Path(filename).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(api.files[filename])
        return str(destination)

    return download


class CloudEntryTests(unittest.TestCase):
    def test_bootstrap_plan_is_pinned_and_run_owned(self) -> None:
        commands = bootstrap_commands("/usr/bin/python3", "/tmp/run")
        self.assertEqual(commands["uv_install"][-1], entry.UV_SPEC)
        self.assertEqual(commands["python_install"][-3:], ["python", "install", "3.10.19"])
        self.assertIn("--managed-python", commands["python_find"])
        self.assertIn("--no-python-downloads", commands["python_find"])
        self.assertIn("--resolve-links", commands["python_find"])
        self.assertTrue(commands["venv"][-1].endswith("/venv"))

    def test_download_binding_is_immutable_and_canonical(self) -> None:
        b = binding()
        report = validate_download_binding(b)
        self.assertEqual(report["target"]["revision"], entry.PRIVATE_TARGET_REVISION)
        self.assertEqual(report["public"]["revision"], entry.PUBLIC_DRAFT_REVISION)
        bad = copy.deepcopy(b)
        bad["downloads"]["target"]["revision"] = "latest"
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_download_binding(bad)

    def test_download_staging_materializes_and_relocates_manifest_without_network(self) -> None:
        target_bytes = {"model.safetensors": b"target-weights", "config.json": b"target-config"}
        public_bytes = {"model.safetensors": b"public-weights", "config.json": b"public-config"}
        target_pins = {name: hashlib.sha256(value).hexdigest() for name, value in target_bytes.items()}
        public_pins = {name: hashlib.sha256(value).hexdigest() for name, value in public_bytes.items()}
        with tempfile.TemporaryDirectory(prefix="cloud-entry-download-") as temp:
            root = Path(temp)
            payload = root / "payload"
            (payload / "staged/inputs").mkdir(parents=True)
            (payload / "source").mkdir()
            manifest = {
                "model_dir": "/workstation/target",
                "config_file": "config.json",
                "config_sha256": target_pins["config.json"],
                "weights_file": "model.safetensors",
                "weights_sha256": target_pins["model.safetensors"],
                "merged_weights_sha256": target_pins["model.safetensors"],
                "deepspec_revision": entry.DEEPSPEC_REVISION,
            }
            manifest_path = payload / "staged/inputs/target-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            b = binding()
            b["staging"]["root"] = "staged"
            b["pins"]["target_manifest_sha256"] = digest(manifest_path)
            b["downloads"]["target"]["files"] = target_pins
            b["downloads"]["public"]["files"] = public_pins
            run = root / "run"
            run.mkdir()

            def fake_download(repo, *, filename, revision, local_dir, cache_dir, token):
                del revision, cache_dir, token
                name = Path(filename).name
                data = target_bytes[name] if repo == downloader.PRIVATE_TARGET_REPO else public_bytes[name]
                destination = Path(local_dir) / (Path("remote") / name if repo == downloader.PRIVATE_TARGET_REPO else name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                return str(destination)

            with patch.object(downloader, "TARGET_FILES", target_pins), patch.object(downloader, "PUBLIC_FILES", public_pins):
                result = stage_downloads(b, payload_root=payload, run=run, download=fake_download, token="private-token")
            self.assertEqual(result["status"], "model_downloads_staged")
            self.assertTrue((run / "inputs/target-model/model.safetensors").is_file())
            self.assertTrue((run / "inputs/public-draft/config.json").is_file())
            effective = json.loads((run / "inputs/target-manifest.json").read_text())
            self.assertEqual(effective["model_dir"], str((run / "inputs/target-model").resolve()))
            # Match the runtime consumer from an unrelated source CWD.
            import subprocess
            probe = subprocess.run([
                __import__("sys").executable, "-c",
                "import json,pathlib,sys; m=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
                "d=pathlib.Path(m['model_dir']).resolve(); "
                "assert (d/m['weights_file']).is_file(); "
                "assert (d/m.get('config_file','config.json')).is_file()",
                str((run / "inputs/target-manifest.json").resolve()),
            ], cwd=payload, capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(effective["source_manifest_sha256"], digest(manifest_path))

    def test_import_probe_preserves_real_virtualenv_interpreter(self) -> None:
        import venv
        import subprocess
        import managed_python
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
            python = root / "bin/python"
            site = Path(subprocess.check_output([str(python), "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True).strip())
            (site / "sepalith_venv_probe.py").write_text("VALUE = 1\n")
            with patch.object(managed_python, "REQUIRED_IMPORTS", ("sepalith_venv_probe",)):
                report = managed_python.check_imports(python)
            self.assertEqual(report["missing"], [])

    def test_binding_deadline_budget_and_retry_gates(self) -> None:
        b = binding()
        limits = entry.validate_binding(b, now=NOW)
        self.assertEqual(limits["hard_deadline"], entry.utc("2026-09-14T01:55:00Z"))
        self.assertEqual(limits["work_deadline"], entry.utc("2026-09-14T01:40:00Z"))
        self.assertEqual(sum(entry.EXPECTED_BUDGET.values()), entry.TOTAL_WINDOW_SECONDS)
        bad = copy.deepcopy(b)
        bad["max_retries"] = 1
        with self.assertRaisesRegex(entry.EntryError, "retries"):
            entry.validate_binding(bad, now=NOW)
        bad = copy.deepcopy(b)
        bad["budget"]["profile_seconds"] = 3601
        with self.assertRaisesRegex(entry.EntryError, "budgets"):
            entry.validate_binding(bad, now=NOW)
        bad = copy.deepcopy(b)
        del bad["bootstrap"]
        with self.assertRaisesRegex(entry.EntryError, "bootstrap"):
            entry.validate_binding(bad, now=NOW)

    def test_phase_deadline_respects_reserved_refresh_window(self) -> None:
        work = NOW + 1000
        self.assertEqual(entry.phase_deadline(NOW, 3600, work, now=NOW), work)
        with self.assertRaisesRegex(entry.EntryError, "reserved"):
            entry.phase_deadline(work, 10, work, now=work)

    def test_parent_bootstrap_environment_does_not_dirty_fresh_profile(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cloud-entry-env-") as temp:
            root = Path(temp)
            profile = root / "profile"
            profile.mkdir()
            public = root / "model.safetensors"
            public.write_bytes(b"fixture")
            entry.clean_environment(
                run=root / "environment",
                inputs={"public_draft_weights": {"path": str(public)}},
                source_root=root,
                bootstrap_layout={
                    "tools": root / "runtime/bootstrap-tools",
                    "python_root": root / "runtime/python",
                    "venv": root / "runtime/venv",
                    "uv_cache": root / "runtime/uv-cache",
                },
            )
            self.assertEqual(list(profile.iterdir()), [])

    def test_mocked_run_entry_uses_static_inputs_before_bootstrap(self) -> None:
        """Exercise setup wiring far enough to catch an undefined input variable."""
        import time

        with tempfile.TemporaryDirectory(prefix="cloud-entry-run-") as temp:
            root = Path(temp)
            binding_path = root / "binding.json"
            b = binding()
            current = time.time()
            armed = datetime.fromtimestamp(current - 1, timezone.utc)
            absolute = datetime.fromtimestamp(current + 7190, timezone.utc)
            b["watchdog_armed_at_utc"] = armed.isoformat()
            b["absolute_deadline_utc"] = absolute.isoformat()
            b["run_root"] = "runs"
            binding_path.write_text(json.dumps(b), encoding="utf-8")
            source = root / "source"
            candidate = root / "candidate.txt"
            resolved = root / "resolved.txt"
            static_inputs = {
                "source_root": str(source),
                "stage_root": str(root),
                "requirements_candidate": {"path": str(candidate)},
                "requirements_resolved": {"path": str(resolved)},
                "target_manifest": {"path": str(root / "target-manifest.json")},
                "target_model_dir": str(root / "target-model"),
                "train": {"path": str(root / "train.jsonl")},
            }
            final_inputs = dict(static_inputs)
            final_inputs.update(
                {
                    "public_draft_weights": {"path": str(root / "public/model.safetensors")},
                    "public_draft_config": {"path": str(root / "public/config.json")},
                }
            )
            managed_holder: dict[str, Path] = {}

            def command_plan(_system_python, run, *, paths):
                managed = paths["python_root"] / "bin/python3.10"
                managed_holder["path"] = managed
                uv = paths["tools"] / "bin/uv"
                return {
                    "uv_install": ["image-python", "-m", "pip", "install", "uv==0.11.23"],
                    "uv_version": [str(uv), "--version"],
                    "python_install": [str(uv), "python", "install", "3.10.19"],
                    "python_find": [str(uv), "python", "find", "--resolve-links", "3.10.19"],
                    "venv": [str(uv), "venv", "--python", "<discovered-python>", str(paths["venv"])],
                }

            def child(argv, *, env, cwd, deadline, log):
                if Path(log).name in {"sentinel.log", "upload-success.log", "upload-failure.log"}:
                    self.assertEqual(env["HF_HUB_OFFLINE"], "0")
                    self.assertEqual(env["TRANSFORMERS_OFFLINE"], "0")
                del env, cwd, deadline
                log = Path(log)
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("ok\n", encoding="utf-8")
                if log.name == "uv-install.log":
                    uv = log.parent.parent / "runtime/bootstrap-tools/bin/uv"
                    uv.parent.mkdir(parents=True, exist_ok=True)
                    uv.write_text("#!/bin/sh\n", encoding="utf-8")
                    uv.chmod(0o700)
                elif log.name == "uv-version.log":
                    log.write_text("uv 0.11.23\n", encoding="utf-8")
                elif log.name == "python-find.log":
                    log.write_text("/placeholder/python3.10\n", encoding="utf-8")
                elif log.name == "venv-bootstrap.log":
                    managed = managed_holder["path"]
                    managed.parent.mkdir(parents=True, exist_ok=True)
                    managed.write_text("#!/bin/sh\n", encoding="utf-8")
                    managed.chmod(0o700)
                    venv = managed.parents[2] / "venv/bin/python"
                    venv.parent.mkdir(parents=True, exist_ok=True)
                    venv.symlink_to(managed)
                elif log.name == "model-download.log":
                    run = Path(argv[-1])
                    target = run / "inputs/target-model"
                    public = run / "inputs/public"
                    target.mkdir(parents=True, exist_ok=True)
                    public.mkdir(parents=True, exist_ok=True)
                    manifest = run / "inputs/target-manifest.json"
                    manifest.write_text("{}\n", encoding="utf-8")
                    for name in ("model.safetensors", "config.json"):
                        (public / name).write_bytes(b"x")
                    report = {
                        "status": "model_downloads_staged",
                        "effective_manifest": {"path": str(manifest), "model_dir": str(target)},
                        "public_draft": {"files": [{"filename": "model.safetensors", "path": str(public / "model.safetensors")}, {"filename": "config.json", "path": str(public / "config.json")}]},
                    }
                    (run / "staging-download-receipt.json").write_text(json.dumps(report), encoding="utf-8")
                elif log.name == "sentinel.log":
                    run = Path(argv[-1])
                    (run / "sentinel-receipt.json").write_text("{}\n", encoding="utf-8")
                elif log.name == "profile.log":
                    profile = Path(argv[argv.index("--run-dir") + 1])
                    profile.mkdir(parents=True, exist_ok=True)
                    (profile / "profile-terminal.json").write_text(json.dumps({"status": "profile_passed"}), encoding="utf-8")
                return {"status": "succeeded"}

            with patch.object(entry, "validate_download_binding"), patch.object(
                entry,
                "validate_staged_inputs",
                side_effect=[static_inputs, final_inputs],
            ), patch.object(entry, "validate_candidate_requirements", return_value={}), patch.object(
                entry, "clean_environment", return_value={}
            ), patch.object(entry, "bootstrap_commands", side_effect=command_plan), patch.object(
                entry, "discover_result", side_effect=lambda _output, _root: managed_holder["path"]
            ), patch.object(entry, "probe_runtime", return_value={"version": [3, 10, 19]}), patch.object(
                entry, "check_imports", return_value={}
            ), patch.object(entry, "run_child", side_effect=child), patch.object(
                entry, "stage_durable", return_value={"status": "durable_staged"}
            ):
                result = entry.run_entry(binding_path, now=current)
            self.assertEqual(result["status"], "cloud_entry_succeeded")
            self.assertTrue(result["bootstrap"]["managed_python_bootstrapped"])

    def test_profile_command_binds_runtime_v3_and_explicit_inputs(self) -> None:
        b = binding()
        command = entry.profile_command(
            python=Path("/stage/python/bin/python3.10"),
            profile_source=Path("/stage/source/" + entry.PROFILE_RELATIVE),
            profile_run=Path("/run/profile"),
            target_manifest=Path("/stage/inputs/target-manifest.json"),
            train_data=Path("/stage/inputs/train.jsonl"),
            source_root=Path("/stage/source"),
        )
        self.assertIn("--execute", command)
        self.assertEqual(entry.TARGET_RUNTIME_RELATIVE, "docs/campaign/work/r2-draft-target-runtime-v3/target_runtime.py")
        self.assertTrue(command[1].endswith(entry.PROFILE_RELATIVE))
        self.assertEqual(b["max_retries"], 0)

    def test_candidate_and_resolved_dependency_pins_exclude_matplotlib(self) -> None:
        candidate = Path(__file__).with_name("requirements-cloud-candidate.txt")
        resolved = Path(__file__).with_name("requirements-cloud-resolved.txt")
        for path, expected in (
            (candidate, entry.CANDIDATE_REQUIREMENTS_SHA256),
            (resolved, entry.RESOLVED_REQUIREMENTS_SHA256),
        ):
            report = validate_candidate_requirements(path, expected, sha256_file=lambda q: digest(Path(q)))
            self.assertFalse(report["matplotlib_present"])

    def test_success_staging_includes_manifest_checkpoint_and_excludes_transients(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cloud-entry-stage-") as temp:
            run = Path(temp) / "profile"
            run.mkdir()
            (run / "durable").mkdir()
            (run / "checkpoints/step_8").mkdir(parents=True)
            (run / "stages").mkdir()
            (run / "ephemeral-target-cache").mkdir()
            (run / "hf-cache").mkdir()
            (run / "durable/teacher-profile.json").write_text("teacher\n")
            (run / "checkpoints/step_8/model.bin").write_bytes(b"checkpoint")
            (run / "stages/profile.log").write_text("step=8/8\n")
            (run / "ephemeral-target-cache/raw.bin").write_bytes(b"raw")
            (run / "hf-cache/cache.bin").write_bytes(b"cache")
            (run / "profile-terminal.json").write_text(json.dumps({"status": "profile_passed"}))
            entries = []
            for rel in ("durable/teacher-profile.json", "checkpoints/step_8/model.bin", "stages/profile.log"):
                path = run / rel
                entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)})
            (run / "durable/persistence-manifest.json").write_text(json.dumps({"files": entries}))
            result = stage(run, Path(temp) / "durable-upload")
            destination = Path(result["destination"])
            self.assertTrue(result["checkpoints_included"])
            self.assertTrue((destination / "checkpoints/step_8/model.bin").is_file())
            self.assertTrue((destination / "durable/persistence-manifest.json").is_file())
            self.assertFalse((destination / "ephemeral-target-cache/raw.bin").exists())
            self.assertFalse((destination / "hf-cache/cache.bin").exists())

    def test_failed_staging_keeps_partial_checkpoint_and_excludes_raw_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cloud-entry-failure-stage-") as temp:
            run = Path(temp) / "profile"
            (run / "checkpoints/step_8").mkdir(parents=True)
            (run / "stages").mkdir()
            (run / "ephemeral-target-cache").mkdir()
            (run / "checkpoints/step_8/state.pt").write_bytes(b"partial")
            (run / "stages/profile.log").write_text("step=3/8\n")
            (run / "ephemeral-target-cache/shard.bin").write_bytes(b"raw")
            (run / "profile-terminal.json").write_text(json.dumps({"status": "profile_failed"}))
            (run / "entry-failure.json").write_text(json.dumps({"status": "failed"}))
            result = stage(run, Path(temp) / "durable-upload")
            destination = Path(result["destination"])
            self.assertTrue(result["partial_failure_allowed"])
            self.assertTrue((destination / "checkpoints/step_8/state.pt").is_file())
            self.assertTrue((destination / "entry-failure.json").is_file())
            self.assertFalse((destination / "ephemeral-target-cache/shard.bin").exists())

    def test_fake_private_sentinel_and_durable_upload_verify_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cloud-entry-upload-") as temp:
            run = Path(temp) / "run"
            (run / "durable-upload").mkdir(parents=True)
            (run / "durable-upload/profile-terminal.json").write_text("ok\n")
            (run / "durable-upload/checkpoint.bin").write_bytes(b"checkpoint")
            b = binding()
            api = FakeAPI()
            download = fake_download(api)
            sentinel = uploader.upload_sentinel(run, b, api, download)
            self.assertTrue(sentinel["repository_private"])
            self.assertFalse(sentinel["token_persisted"])
            uploaded = uploader.upload_durable(run, b, api, download, mode="success")
            self.assertEqual(uploaded["status"], "durable_upload_verified")
            self.assertIn("upload-manifest.json", uploaded["files"])
            self.assertTrue(uploaded["independent_root_readback_required"])
            (run / "durable-upload/tmp").mkdir()
            (run / "durable-upload/tmp/raw").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "transient"):
                uploader.build_local_manifest(run / "durable-upload")

    def test_remote_hash_mismatch_is_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "hash"):
            uploader.validate_remote({"x": {"bytes": 1, "sha256": "a"}}, {"x": {"bytes": 1, "sha256": "b"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
