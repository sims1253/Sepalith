"""Portable, admitted-only three-stage DSpark profile orchestrator.

The profile is intentionally bounded: eight target rows, one native CUDA
smoke, and eight optimizer steps.  It creates no provider job and has no
credential client.  ``--execute`` additionally requires the root-owned
``SEPALITH_PROFILE_ADMITTED=1`` environment gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from source_inventory import build_inventory


ROOT = Path(__file__).resolve().parent
TRAIN_SOURCE_PATH = "docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl"
TRAIN_SOURCE_SHA256 = "85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e"
DEEPSPEC_REVISION = "005e03b81cec38b7da6399833d609ee89a2587f2"
TARGET_SELECTED_STEP500_MERGED_WEIGHTS_SHA256 = (
    "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c"
)
TARGET_ARTIFACT_PINS = {
    "model.safetensors": TARGET_SELECTED_STEP500_MERGED_WEIGHTS_SHA256,
    "config.json": "f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991",
    "generation_config.json": "7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269",
    "tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config.json": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}
PUBLIC_DRAFT_WEIGHTS_SHA256 = (
    "ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97"
)
PUBLIC_DRAFT_WEIGHTS_BYTES = 647558522
PUBLIC_DRAFT_CONFIG_SHA256 = "bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b"
PROFILE_ROWS = 8
PROFILE_STEPS = 8
PROFILE_CONTEXT = 4096
PROJECT_NAME = "sepalith-r2"
EXP_NAME = "dspark_minicpm5_2b_train_only"
# The first profile is bounded to 1,800 seconds all-in.  Stage allocations
# leave explicit headroom for setup, input binding, durable persistence, and
# terminal receipts inside that outer cap.  Later refresh/integration work is
# reserved separately by the root admission decision.
PROFILE_LIMIT_SECONDS = 1800
TOTAL_WINDOW_SECONDS = 1800
NON_PROFILE_RESERVE_SECONDS = TOTAL_WINDOW_SECONDS - PROFILE_LIMIT_SECONDS
STAGE_EXECUTION_BUDGET_SECONDS = 1500
SETUP_AND_PERSISTENCE_RESERVE_SECONDS = 300
FINAL_REFRESH_ALLOWANCE_SECONDS = 1200
INTEGRATION_ALLOWANCE_SECONDS = 600
STAGE_LIMITS_SECONDS = {
    "teacher-cache-profile": 420,
    "native-cuda-smoke": 240,
    "warmstart-trainer-profile": 720,
}
if sum(STAGE_LIMITS_SECONDS.values()) > STAGE_EXECUTION_BUDGET_SECONDS:
    raise RuntimeError("profile stage limits exceed the stage execution budget")
if STAGE_EXECUTION_BUDGET_SECONDS + SETUP_AND_PERSISTENCE_RESERVE_SECONDS > PROFILE_LIMIT_SECONDS:
    raise RuntimeError("profile stage budget and setup reserve exceed the outer deadline")
EPHEMERAL_CACHE_NAME = "ephemeral-target-cache"
TRANSIENT_RUN_ROOTS = (
    EPHEMERAL_CACHE_NAME,
    "tmp",
    "hf-cache",
    "xdg-cache",
    "torch-cache",
    "triton-cache",
)
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PRIVATE_KEY", "ACCESS_KEY")


class ProfileError(RuntimeError):
    """A bounded profile preflight, stage, or artifact gate failed."""


def _handle_termination(signum, _frame):
    """Turn supervisor termination into an exception so stage cleanup runs."""
    raise ProfileError(f"profile received signal {int(signum)}")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProfileError(reason)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def prepare_run_dir(path: str | Path) -> Path:
    run = Path(path).expanduser().resolve()
    if run.exists():
        require(run.is_dir(), "profile run path is not a directory")
        require(not any(run.iterdir()), "profile run directory must be fresh; implicit resume is forbidden")
    else:
        run.mkdir(parents=True, exist_ok=False)
    (run / "stages").mkdir()
    (run / "durable").mkdir()
    return run


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_manifest_fields(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    required = ("model_dir", "config_sha256", "weights_file", "weights_sha256", "deepspec_revision")
    missing = [name for name in required if not manifest.get(name)]
    require(not missing, "target model manifest missing: " + ", ".join(missing))
    require(
        str(manifest["deepspec_revision"]) == DEEPSPEC_REVISION,
        "target manifest DeepSpec revision differs from the pinned source",
    )
    model_dir = Path(str(manifest["model_dir"])).expanduser()
    if not model_dir.is_absolute():
        model_dir = manifest_path.parent / model_dir
    require(not model_dir.is_symlink(), "target model directory symlink is forbidden")
    model_dir = model_dir.resolve()
    require(model_dir.is_dir(), "target model directory is missing")
    require(manifest.get("config_file", "config.json") == "config.json", "target config filename must be config.json")
    require(manifest["weights_file"] == "model.safetensors", "target weights filename must be model.safetensors")
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    config_path = config_path.resolve()
    weights_path = weights_path.resolve()
    require(_inside(config_path, model_dir), "target config escapes model directory")
    require(_inside(weights_path, model_dir), "target weights escape model directory")
    require(config_path.is_file() and weights_path.is_file(), "target manifest files are missing")
    for filename in TARGET_ARTIFACT_PINS:
        artifact = model_dir / filename
        require(artifact.is_file() and not artifact.is_symlink(), f"target artifact is missing or symlinked: {filename}")
        require(sha256_file(artifact) == TARGET_ARTIFACT_PINS[filename], f"target artifact SHA-256 differs: {filename}")
    for artifact in model_dir.iterdir():
        # HF accepts several alternate/sharded weight conventions.  A profile
        # manifest binds one canonical safetensors file, so reject every other
        # weight-like artifact before a loader can select it implicitly.
        name = artifact.name
        weight_like = name.endswith((
            ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5",
        ))
        if name.endswith(".index.json") or (weight_like and name != "model.safetensors"):
            raise ProfileError(f"sharded or alternative target artifact is forbidden: {name}")
    expected_config = str(manifest["config_sha256"])
    expected_weights = str(manifest["weights_sha256"])
    require(expected_config == TARGET_ARTIFACT_PINS["config.json"], "target config manifest pin differs")
    require(expected_weights == TARGET_ARTIFACT_PINS["model.safetensors"], "target weights manifest pin differs")
    candidate_digest = str(manifest.get("merged_weights_sha256", expected_weights))
    require(candidate_digest == TARGET_SELECTED_STEP500_MERGED_WEIGHTS_SHA256, "target merged-weights candidate pin differs")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "model_dir": str(model_dir),
        "config_file": str(config_path),
        "config_sha256": expected_config,
        "weights_file": str(weights_path),
        "weights_sha256": expected_weights,
        "merged_weights_sha256": candidate_digest,
        "deepspec_revision": str(manifest["deepspec_revision"]),
    }


def verify_inputs(
    *,
    target_manifest_path: str | Path,
    train_data_path: str | Path,
    public_draft_weights_path: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Verify explicit staged inputs without loading a model or parsing rows."""
    manifest_path = Path(target_manifest_path).expanduser().resolve()
    require(manifest_path.is_file(), "explicit target model manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError("explicit target model manifest is not valid JSON") from exc
    require(isinstance(manifest, dict), "target model manifest must be an object")
    target = _safe_manifest_fields(manifest, manifest_path)

    train_path = Path(train_data_path).expanduser().resolve()
    require(train_path.is_file(), "explicit TRAIN source file is missing")
    train_sha256 = sha256_file(train_path)
    require(train_sha256 == TRAIN_SOURCE_SHA256, "TRAIN source SHA-256 differs from admitted source")

    public_path = Path(public_draft_weights_path).expanduser().resolve()
    require(public_path.is_file(), "explicit public draft weights file is missing")
    require(public_path.stat().st_size == PUBLIC_DRAFT_WEIGHTS_BYTES, "public draft weight size differs")
    public_sha256 = sha256_file(public_path)
    require(public_sha256 == PUBLIC_DRAFT_WEIGHTS_SHA256, "public draft weights SHA-256 differs")
    public_config_path = public_path.with_name("config.json")
    require(
        public_config_path.is_file() and not public_config_path.is_symlink(),
        "public draft sibling config.json is missing or symlinked",
    )
    public_config_sha256 = sha256_file(public_config_path)
    require(
        public_config_sha256 == PUBLIC_DRAFT_CONFIG_SHA256,
        "public draft sibling config SHA-256 differs",
    )

    source_root_path = Path(source_root).expanduser().resolve()
    inventory = build_inventory(source_root_path)
    expected_path = ROOT / "source-inventory.json"
    require(expected_path.is_file(), "portable source inventory is missing")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    require(
        inventory.get("files") == expected.get("files")
        and inventory.get("requirements") == expected.get("requirements"),
        "portable source or requirements inventory differs",
    )
    return {
        "target": target,
        "train": {
            "canonical_path": TRAIN_SOURCE_PATH,
            "canonical_sha256": TRAIN_SOURCE_SHA256,
            "actual_path": str(train_path),
            "actual_sha256": train_sha256,
        },
        "public_draft": {
            "actual_path": str(public_path),
            "bytes": public_path.stat().st_size,
            "sha256": public_sha256,
            "config_path": str(public_config_path),
            "config_sha256": public_config_sha256,
        },
        "source_inventory": inventory,
        "deepspec_revision": DEEPSPEC_REVISION,
    }


def clean_environment(*, run: Path, inputs: Mapping[str, Any], source_root: Path) -> dict[str, str]:
    """Create a child environment with credential variables and global caches removed."""
    environment = dict(os.environ)
    for key in list(environment):
        if any(marker in key.upper() for marker in SECRET_MARKERS):
            environment.pop(key, None)
    # Keep HOME untouched.  Every mutable cache and temporary file location is
    # explicit and owned by this fresh run, so a cloud job cannot resume from a
    # workstation cache or consume a provider-wide temporary directory.
    run = run.resolve()
    cache_roots = {
        "tmp": run / "tmp",
        "hf": run / "hf-cache",
        "xdg": run / "xdg-cache",
        "torch": run / "torch-cache",
        "triton": run / "triton-cache",
    }
    for path in cache_roots.values():
        path.mkdir(parents=True, exist_ok=True)
    target = inputs["target"]
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "TMPDIR": str(cache_roots["tmp"]),
            "TEMP": str(cache_roots["tmp"]),
            "TMP": str(cache_roots["tmp"]),
            "HF_HOME": str(cache_roots["hf"]),
            "XDG_CACHE_HOME": str(cache_roots["xdg"]),
            "TORCH_HOME": str(cache_roots["torch"]),
            "TRITON_CACHE_DIR": str(cache_roots["triton"]),
            "CUDA_VISIBLE_DEVICES": "0",
            "SEPALITH_DEEPSPEC_ROOT": str(
                source_root
                / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec"
            ),
            "SEPALITH_HARDENING_ROOT": str(
                source_root / "docs/campaign/work/r2-draft-adapter-hardening-v1"
            ),
            "SEPALITH_UPSTREAM_SMOKE_SOURCE": str(
                source_root / "docs/campaign/work/r2-draft-upstream-smoke-v1/upstream_dspark_smoke.py"
            ),
            "SEPALITH_MINICPM5_TARGET": str(target["model_dir"]),
            "SEPALITH_PUBLIC_DRAFT_WEIGHTS": str(inputs["public_draft"]["actual_path"]),
            "SEPALITH_DRAFT_CACHE": str(run / EPHEMERAL_CACHE_NAME),
            "SEPALITH_DRAFT_OUTPUT_ROOT": str(run / "checkpoints"),
            "SEPALITH_DRAFT_LOG_ROOT": str(run / "tensorboard"),
            "SEPALITH_PROFILE_RUN_DIR": str(run),
            "SEPALITH_PROFILE_TELEMETRY_PATH": str(run / "stages" / "warmstart-trainer-profile.gpu.jsonl"),
        }
    )
    python_paths = [
        source_root / "docs/campaign/work/r2-draft-target-runtime-v3",
        source_root / "docs/campaign/work/r2-draft-adapter-hardening-v1",
        source_root / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation",
        source_root / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec",
    ]
    python_paths.append(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/dependency-overlay'))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    return environment


@dataclass(frozen=True)
class StageSpec:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    wall_limit_seconds: int
    telemetry: bool = False
    max_rows: int | None = None
    max_steps: int | None = None


def build_stage_plan(
    *,
    run: str | Path,
    target_manifest_path: str | Path,
    train_data_path: str | Path,
    source_root: str | Path,
    python_executable: str | Path = sys.executable,
) -> tuple[StageSpec, ...]:
    """Return the fixed sequential profile command map."""
    run = Path(run).resolve()
    source_root = Path(source_root).resolve()
    python = str(Path(python_executable).expanduser())
    target_script = source_root / "docs/campaign/work/r2-draft-target-runtime-v3/target_runtime.py"
    smoke_script = ROOT / "native_cuda_smoke.py"
    smoke_source = source_root / "docs/campaign/work/r2-draft-upstream-smoke-v1/upstream_dspark_smoke.py"
    vendor = source_root / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec"
    trainer_config = source_root / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/profile_config.py"
    trainer_entry = vendor / "train.py"
    return (
        StageSpec(
            name="teacher-cache-profile",
            argv=(
                python,
                str(target_script),
                "--model-manifest",
                str(Path(target_manifest_path).resolve()),
                "--train-data-path",
                str(Path(train_data_path).resolve()),
                "--output-dir",
                str(run / EPHEMERAL_CACHE_NAME),
                "--limit",
                str(PROFILE_ROWS),
                "--device",
                "cuda:0",
                "--dtype",
                "bfloat16",
                "--target-layer-ids",
                "1,10,20,30,39",
            ),
            cwd=source_root,
            wall_limit_seconds=STAGE_LIMITS_SECONDS["teacher-cache-profile"],
            max_rows=PROFILE_ROWS,
        ),
        StageSpec(
            name="native-cuda-smoke",
            argv=(
                python,
                str(smoke_script),
                "--device",
                "cuda:0",
                "--source",
                str(smoke_source),
                "--vendor-root",
                str(vendor),
            ),
            cwd=source_root,
            wall_limit_seconds=STAGE_LIMITS_SECONDS["native-cuda-smoke"],
        ),
        StageSpec(
            name="warmstart-trainer-profile",
            argv=(python, str(trainer_entry), "--config", str(trainer_config)),
            cwd=vendor,
            wall_limit_seconds=STAGE_LIMITS_SECONDS["warmstart-trainer-profile"],
            telemetry=True,
            max_steps=PROFILE_STEPS,
        ),
    )


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            pass


def _gpu_sample() -> dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"at": stamp(), "available": False, "finite": False, "error_type": type(exc).__name__}
    if result.returncode != 0:
        return {"at": stamp(), "available": False, "finite": False, "error_type": "nvidia_smi_failed"}
    devices = []
    try:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise ValueError("unexpected nvidia-smi field count")
            values = [float(field) for field in fields[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite GPU telemetry")
            devices.append(
                {
                    "name": fields[0],
                    "memory_used_mib": values[0],
                    "memory_total_mib": values[1],
                    "utilization_percent": values[2],
                }
            )
    except ValueError as exc:
        return {"at": stamp(), "available": False, "finite": False, "error_type": type(exc).__name__}
    return {"at": stamp(), "available": bool(devices), "finite": bool(devices), "devices": devices}


def run_stage(stage: StageSpec, *, run: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Run one stage with an owned process group and a terminal receipt."""
    log_path = run / "stages" / f"{stage.name}.log"
    telemetry_path = run / "stages" / f"{stage.name}.gpu.jsonl"
    started = time.monotonic()
    started_at = stamp()
    timed_out = False
    process: subprocess.Popen[bytes] | None = None
    telemetry_count = 0
    try:
        with log_path.open("wb") as output:
            process = subprocess.Popen(
                list(stage.argv),
                cwd=str(stage.cwd),
                env=dict(environment),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with telemetry_path.open("w", encoding="utf-8") as telemetry:
                while process.poll() is None:
                    if stage.telemetry:
                        telemetry.write(json.dumps(_gpu_sample(), sort_keys=True) + "\n")
                        telemetry.flush()
                        telemetry_count += 1
                    if time.monotonic() - started >= stage.wall_limit_seconds:
                        timed_out = True
                        _terminate_process_group(process)
                        break
                    time.sleep(1.0 if stage.telemetry else 0.25)
            try:
                exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                exit_code = process.poll()
    except Exception as exc:
        terminal = {
            "at": stamp(),
            "started_at": started_at,
            "status": "failed",
            "stage": stage.name,
            "error_type": type(exc).__name__,
            "argv": list(stage.argv),
            "log": str(log_path),
        }
        write_json(run / "stages" / f"{stage.name}.terminal.json", terminal)
        raise ProfileError(f"{stage.name} could not start or be supervised") from exc
    finally:
        # The outer watchdog can terminate this orchestrator while a stage is
        # running. Always reap the owned child process group before unwinding.
        if process is not None and process.poll() is None:
            _terminate_process_group(process)
    elapsed = time.monotonic() - started
    succeeded = not timed_out and exit_code == 0
    terminal = {
        "at": stamp(),
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "status": "succeeded" if succeeded else "failed",
        "stage": stage.name,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_limit_seconds": stage.wall_limit_seconds,
        "argv": list(stage.argv),
        "cwd": str(stage.cwd),
        "log": str(log_path),
        "telemetry": str(telemetry_path) if stage.telemetry else None,
        "telemetry_samples": telemetry_count,
        "max_rows": stage.max_rows,
        "max_steps": stage.max_steps,
    }
    write_json(run / "stages" / f"{stage.name}.terminal.json", terminal)
    if not succeeded:
        reason = "wall_limit_exceeded" if timed_out else "nonzero_exit"
        raise ProfileError(f"{stage.name} failed: {reason}")
    return terminal


def _last_json_object(log_path: Path) -> dict[str, Any]:
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ProfileError(f"stage log has no terminal JSON object: {log_path.name}")


def _persist_teacher_metadata(run: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    cache = run / EPHEMERAL_CACHE_NAME
    destination = run / "durable" / "teacher-profile"
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = cache / "manifest.json"
    source_rows = cache / "teacher_rows.jsonl"
    require(source_manifest.is_file() and source_rows.is_file(), "teacher cache profile outputs are incomplete")
    shutil.copy2(source_manifest, destination / "manifest.json")
    shutil.copy2(source_rows, destination / "teacher_rows.jsonl")
    try:
        cache_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError("teacher cache manifest is not valid JSON") from exc
    selected_rows = int(report.get("selected_rows", -1))
    cacheable_rows = int(report.get("cacheable_rows", -1))
    written_rows = int(report.get("written_rows", -1))
    written_row_ids = report.get("written_row_ids")
    reconciliation = report.get("reconciliation")
    require(isinstance(written_row_ids, list), "teacher profile omitted written row IDs")
    require(isinstance(reconciliation, Mapping), "teacher profile omitted row reconciliation")
    require(int(cache_manifest.get("num_samples", -1)) == written_rows, "cache manifest count differs from written rows")
    result = {
        "status": "teacher_metadata_persisted",
        "cache_path": str(cache),
        "cache_excluded_from_durable_upload": True,
        "manifest_path": str(destination / "manifest.json"),
        "manifest_sha256": sha256_file(destination / "manifest.json"),
        "teacher_rows_path": str(destination / "teacher_rows.jsonl"),
        "teacher_rows_sha256": sha256_file(destination / "teacher_rows.jsonl"),
        "selected_rows": selected_rows,
        "cacheable_rows": cacheable_rows,
        "written_rows": written_rows,
        "written_row_ids": [str(row_id) for row_id in written_row_ids],
        "reconciliation": dict(reconciliation),
        "target_cache_version": int(report.get("target_cache_version", -1)),
    }
    write_json(run / "durable" / "teacher-profile.json", result)
    return result


def validate_teacher_stage(run: Path, terminal: Mapping[str, Any]) -> dict[str, Any]:
    report = _last_json_object(Path(str(terminal["log"])))
    require(report.get("status") == "target_runtime_passed", "teacher cache profile did not pass target runtime")
    selected_rows = int(report.get("selected_rows", -1))
    cacheable_rows = int(report.get("cacheable_rows", -1))
    written_rows = int(report.get("written_rows", -1))
    written_row_ids = report.get("written_row_ids")
    reconciliation = report.get("reconciliation")
    require(selected_rows == PROFILE_ROWS, "teacher profile did not select exactly eight TRAIN rows")
    require(cacheable_rows > 0, "teacher profile produced no cacheable rows")
    require(written_rows == cacheable_rows, "teacher profile cacheable and written row counts differ")
    require(isinstance(written_row_ids, list) and len(written_row_ids) == written_rows, "teacher profile written row IDs do not match count")
    require(
        isinstance(reconciliation, Mapping)
        and int(reconciliation.get("written_row_count", -1)) == written_rows
        and [str(value) for value in reconciliation.get("written_row_ids", [])] == [str(value) for value in written_row_ids],
        "teacher profile row reconciliation differs from actual writes",
    )
    require(int(report.get("target_cache_version", -1)) == 2, "teacher profile cache version differs")
    return _persist_teacher_metadata(run, report)


def validate_native_stage(run: Path, terminal: Mapping[str, Any]) -> dict[str, Any]:
    report = _last_json_object(Path(str(terminal["log"])))
    require(str(report.get("device", "")).startswith("cuda"), "native smoke did not use CUDA")
    native = report.get("native_flex") or {}
    eager = report.get("eager_dense_cpu") or {}
    require(
        native.get("forward") is True
        and native.get("forward_without_grad") is True
        and native.get("backward") is True,
        "native CUDA FlexAttention forward/backward gate failed",
    )
    # The eager probe is retained as a CPU reference only.  Its gradients do
    # not establish CUDA training, so they are never reported as native proof.
    eager_reference = {
        "loss_finite": eager.get("loss_finite") is True,
        "gradients_finite": eager.get("gradients_finite") is True,
    }
    destination = run / "durable" / "native-cuda-smoke.json"
    write_json(destination, report)
    return {
        "status": "native_cuda_smoke_validated",
        "report_path": str(destination),
        "report_sha256": sha256_file(destination),
        "native_forward": True,
        "native_backward": True,
        "eager_cpu_reference": eager_reference,
    }


def _checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir(), "warm-start checkpoint root is missing")
    rows = []
    for child in sorted(root.iterdir()):
        if not child.name.startswith("step_") or not child.is_dir():
            continue
        bytes_total = 0
        files = 0
        for path in child.rglob("*"):
            if path.is_symlink():
                raise ProfileError("checkpoint symlink is not allowed in profile inventory")
            if path.is_file():
                files += 1
                bytes_total += path.stat().st_size
        rows.append({"name": child.name, "files": files, "bytes": bytes_total})
    return rows


def validate_trainer_stage(run: Path, terminal: Mapping[str, Any]) -> dict[str, Any]:
    log_text = Path(str(terminal["log"])).read_text(encoding="utf-8", errors="replace")
    steps = [(int(value), int(limit)) for value, limit in re.findall(r"\bstep=(\d+)/(\d+)\b", log_text)]
    expected_steps = [(value, PROFILE_STEPS) for value in range(1, PROFILE_STEPS + 1)]
    require(steps == expected_steps, "warm-start profile steps are not contiguous 1..8 without duplicates or regressions")
    telemetry_path = Path(str(terminal["telemetry"]))
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(telemetry_rows and all(row.get("available") and row.get("finite") for row in telemetry_rows), "finite GPU telemetry is missing")
    checkpoint_root = run / "checkpoints" / PROJECT_NAME / EXP_NAME
    checkpoints = _checkpoint_inventory(checkpoint_root)
    require(any(row["name"] == "step_8" for row in checkpoints), "step_8 checkpoint is missing")
    result = {
        "status": "warmstart_profile_validated",
        "steps_observed": [value for value, _ in steps],
        "max_steps": PROFILE_STEPS,
        "gpu_telemetry_path": str(telemetry_path),
        "gpu_telemetry_samples": len(telemetry_rows),
        "gpu_telemetry_finite": True,
        "checkpoint_root": str(checkpoint_root),
        "checkpoints": checkpoints,
        "implicit_resume": False,
    }
    write_json(run / "durable" / "warmstart-profile.json", result)
    return result


def build_durable_manifest(run: Path) -> dict[str, Any]:
    """Inventory durable outputs while excluding raw target cache shards."""
    files = []
    transient_roots = [(run / name).resolve() for name in TRANSIENT_RUN_ROOTS]
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        resolved = path.resolve()
        if any(_inside(resolved, transient) for transient in transient_roots):
            continue
        if path.name == "persistence-manifest.json":
            continue
        relative = path.relative_to(run).as_posix()
        require(".git" not in Path(relative).parts and "dependency-overlay" not in Path(relative).parts, "non-portable durable output path")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    result = {
        "schema": 1,
        "status": "durable_profile_manifest_ready",
        "cache_excluded": EPHEMERAL_CACHE_NAME,
        "transient_roots_excluded": list(TRANSIENT_RUN_ROOTS),
        "durable_upload_root": "durable",
        "teacher_tokens_included": any(row["path"] == "durable/teacher-profile/teacher_rows.jsonl" for row in files),
        "target_cache_manifest_included": any(row["path"] == "durable/teacher-profile/manifest.json" for row in files),
        "checkpoints_included": any(row["path"].startswith("checkpoints/") for row in files),
        "files": files,
    }
    write_json(run / "durable" / "persistence-manifest.json", result)
    return result


def execute_profile(
    *,
    run_dir: str | Path,
    target_manifest_path: str | Path,
    train_data_path: str | Path,
    source_root: str | Path,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    """Run the profile with signal-aware child-process cleanup."""
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    try:
        return _execute_profile_impl(
            run_dir=run_dir,
            target_manifest_path=target_manifest_path,
            train_data_path=train_data_path,
            source_root=source_root,
            python_executable=python_executable,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def _execute_profile_impl(
    *,
    run_dir: str | Path,
    target_manifest_path: str | Path,
    train_data_path: str | Path,
    source_root: str | Path,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    """Run all three bounded stages sequentially after root admission."""
    profile_started = time.monotonic()
    profile_deadline = profile_started + PROFILE_LIMIT_SECONDS
    run = prepare_run_dir(run_dir)
    source_root = Path(source_root).expanduser().resolve()
    public_path = os.environ.get("SEPALITH_PUBLIC_DRAFT_WEIGHTS")
    require(public_path, "SEPALITH_PUBLIC_DRAFT_WEIGHTS is required")
    inputs = verify_inputs(
        target_manifest_path=target_manifest_path,
        train_data_path=train_data_path,
        public_draft_weights_path=public_path,
        source_root=source_root,
    )
    write_json(run / "durable" / "input-binding.json", {
        "target": inputs["target"],
        "train": inputs["train"],
        "public_draft": inputs["public_draft"],
        "deepspec_revision": DEEPSPEC_REVISION,
        "profile_rows": PROFILE_ROWS,
        "profile_steps": PROFILE_STEPS,
        "profile_limit_seconds": PROFILE_LIMIT_SECONDS,
        "stage_execution_budget_seconds": STAGE_EXECUTION_BUDGET_SECONDS,
        "stage_allocated_seconds": sum(STAGE_LIMITS_SECONDS.values()),
        "setup_and_persistence_reserve_seconds": SETUP_AND_PERSISTENCE_RESERVE_SECONDS,
        "final_refresh_allowance_seconds": FINAL_REFRESH_ALLOWANCE_SECONDS,
        "integration_allowance_seconds": INTEGRATION_ALLOWANCE_SECONDS,
        "total_window_seconds": TOTAL_WINDOW_SECONDS,
        "non_profile_reserve_seconds": NON_PROFILE_RESERVE_SECONDS,
    })
    environment = clean_environment(run=run, inputs=inputs, source_root=source_root)
    stages = build_stage_plan(
        run=run,
        target_manifest_path=target_manifest_path,
        train_data_path=train_data_path,
        source_root=source_root,
        python_executable=python_executable,
    )
    stage_reports: list[dict[str, Any]] = []
    try:
        for stage in stages:
            remaining = profile_deadline - time.monotonic()
            require(remaining >= 1.0, "cumulative profile deadline exhausted before stage start")
            bounded_limit = min(stage.wall_limit_seconds, max(1, int(remaining)))
            bounded_stage = replace(stage, wall_limit_seconds=bounded_limit)
            terminal = run_stage(bounded_stage, run=run, environment=environment)
            terminal["profile_limit_seconds"] = PROFILE_LIMIT_SECONDS
            terminal["profile_elapsed_seconds"] = time.monotonic() - profile_started
            terminal["profile_remaining_seconds"] = max(0.0, profile_deadline - time.monotonic())
            write_json(run / "stages" / f"{stage.name}.terminal.json", terminal)
            if stage.name == "teacher-cache-profile":
                validation = validate_teacher_stage(run, terminal)
            elif stage.name == "native-cuda-smoke":
                validation = validate_native_stage(run, terminal)
            else:
                validation = validate_trainer_stage(run, terminal)
            stage_reports.append({"stage": stage.name, "terminal": terminal, "validation": validation})
        require(time.monotonic() < profile_deadline, "cumulative profile deadline exhausted before persistence")
        durable = build_durable_manifest(run)
        profile_elapsed = time.monotonic() - profile_started
        require(profile_elapsed <= PROFILE_LIMIT_SECONDS, "cumulative profile deadline exceeded during persistence")
        result = {
            "status": "profile_passed",
            "run_dir": str(run),
            "profile_rows": PROFILE_ROWS,
            "profile_steps": PROFILE_STEPS,
            "profile_limit_seconds": PROFILE_LIMIT_SECONDS,
            "stage_execution_budget_seconds": STAGE_EXECUTION_BUDGET_SECONDS,
            "stage_allocated_seconds": sum(STAGE_LIMITS_SECONDS.values()),
            "setup_and_persistence_reserve_seconds": SETUP_AND_PERSISTENCE_RESERVE_SECONDS,
            "final_refresh_allowance_seconds": FINAL_REFRESH_ALLOWANCE_SECONDS,
            "integration_allowance_seconds": INTEGRATION_ALLOWANCE_SECONDS,
            "total_window_seconds": TOTAL_WINDOW_SECONDS,
            "non_profile_reserve_seconds": NON_PROFILE_RESERVE_SECONDS,
            "profile_elapsed_seconds": profile_elapsed,
            "profile_remaining_seconds": PROFILE_LIMIT_SECONDS - profile_elapsed,
            "stage_reports": stage_reports,
            "durable_manifest": durable,
            "cloud_launch": False,
        }
        write_json(run / "profile-terminal.json", result)
        return result
    except Exception as exc:
        profile_elapsed = time.monotonic() - profile_started
        failure = {
            "status": "profile_failed",
            "run_dir": str(run),
            "profile_limit_seconds": PROFILE_LIMIT_SECONDS,
            "stage_execution_budget_seconds": STAGE_EXECUTION_BUDGET_SECONDS,
            "stage_allocated_seconds": sum(STAGE_LIMITS_SECONDS.values()),
            "setup_and_persistence_reserve_seconds": SETUP_AND_PERSISTENCE_RESERVE_SECONDS,
            "final_refresh_allowance_seconds": FINAL_REFRESH_ALLOWANCE_SECONDS,
            "integration_allowance_seconds": INTEGRATION_ALLOWANCE_SECONDS,
            "total_window_seconds": TOTAL_WINDOW_SECONDS,
            "non_profile_reserve_seconds": NON_PROFILE_RESERVE_SECONDS,
            "profile_elapsed_seconds": profile_elapsed,
            "profile_remaining_seconds": max(0.0, PROFILE_LIMIT_SECONDS - profile_elapsed),
            "stage_reports": stage_reports,
            "error_type": type(exc).__name__,
            "reason": str(exc)[:240],
            "cloud_launch": False,
        }
        write_json(run / "profile-terminal.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target-model-manifest", required=True, type=Path)
    parser.add_argument("--train-data-path", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=os.environ.get("SEPALITH_PROFILE_SOURCE_ROOT", str(ROOT.parents[3])))
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("SEPALITH_PROFILE_PYTHON", sys.executable))
    parser.add_argument("--execute", action="store_true", help="run only after the root admission environment gate")
    args = parser.parse_args()
    plan = build_stage_plan(
        run=args.run_dir,
        target_manifest_path=args.target_model_manifest,
        train_data_path=args.train_data_path,
        source_root=args.source_root,
        python_executable=args.python_executable,
    )
    if not args.execute:
        print(json.dumps({
            "status": "plan_only",
            "profile_rows": PROFILE_ROWS,
            "profile_steps": PROFILE_STEPS,
            "profile_limit_seconds": PROFILE_LIMIT_SECONDS,
            "total_window_seconds": TOTAL_WINDOW_SECONDS,
            "non_profile_reserve_seconds": NON_PROFILE_RESERVE_SECONDS,
            "stages": [
                {
                    "name": stage.name,
                    "argv": list(stage.argv),
                    "cwd": str(stage.cwd),
                    "wall_limit_seconds": stage.wall_limit_seconds,
                    "max_rows": stage.max_rows,
                    "max_steps": stage.max_steps,
                }
                for stage in plan
            ],
        }, indent=2, sort_keys=True))
        return
    require(os.environ.get("SEPALITH_PROFILE_ADMITTED") == "1", "root profile admission is required for --execute")
    print(json.dumps(execute_profile(
        run_dir=args.run_dir,
        target_manifest_path=args.target_model_manifest,
        train_data_path=args.train_data_path,
        source_root=args.source_root,
        python_executable=args.python_executable,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ProfileError as exc:
        print(json.dumps({"status": "profile_rejected", "error_type": type(exc).__name__, "reason": str(exc)[:240]}), file=sys.stderr)
        raise SystemExit(1)
