"""Root-bound cloud entry for the bounded Sepalith R2 draft profile.

The entry consumes an already staged payload and an independently armed root
watchdog binding.  It verifies source and artifact identities, checks the
managed CPython runtime, runs profile-v2 with target-runtime-v3, and exposes a
private sentinel plus durable-only success/failure upload path.  It creates no
provider job and contains no provider client in the preflight path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from managed_python import (
    MANAGED_PYTHON_VERSION,
    UV_SPEC,
    UV_VERSION,
    bounded_path,
    bootstrap_commands,
    bootstrap_paths,
    check_imports,
    discover_result,
    probe_runtime,
    validate_candidate_requirements,
)
from stage_durable import stage as stage_durable
from download_staging import (
    PUBLIC_DRAFT_REPO,
    PUBLIC_DRAFT_REVISION,
    PRIVATE_TARGET_PREFIX,
    PRIVATE_TARGET_REPO,
    PRIVATE_TARGET_REVISION,
    stage_from_environment,
    validate_download_binding,
)

HERE = Path(__file__).resolve().parent
IMAGE = "docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9"
INSTANCE_TYPE = "g5.2xlarge"
DEEPSPEC_REVISION = "005e03b81cec38b7da6399833d609ee89a2587f2"
PROFILE_RELATIVE = "docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py"
PROFILE_INVENTORY_RELATIVE = "docs/campaign/work/r2-draft-cloud-profile-v2/source-inventory.json"
TARGET_RUNTIME_RELATIVE = "docs/campaign/work/r2-draft-target-runtime-v3/target_runtime.py"
PROFILE_SOURCE_SHA256 = "411d859e8e00411a93a1402ce6aa9943773f67ef038fd166804135c2211315ee"
TARGET_RUNTIME_SHA256 = "6c8fafbd1d197beea370a89e164ab7db642a6fe6324c1033be2960ad73ecbc2f"
PROFILE_INVENTORY_SHA256 = "9b51028caf406821f59be5ccff1e610b4d69a0b3d3a622d9808a66a7a52c97e1"
CANDIDATE_REQUIREMENTS_SHA256 = "0d09b6ba0309870d744a3a9ec99a40fb6aecfeeaf84e8cbc12d83dd5fba8140a"
RESOLVED_REQUIREMENTS_SHA256 = "acd1da83fee378caa992baf4ff751956cbb9cf5d69830745c894e76d2821607e"
TRAIN_SOURCE_SHA256 = "85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e"
PUBLIC_DRAFT_WEIGHTS_SHA256 = "ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97"
PUBLIC_DRAFT_WEIGHTS_BYTES = 647558522
PUBLIC_DRAFT_CONFIG_SHA256 = "bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b"
TARGET_CANDIDATE_MERGED_WEIGHTS_SHA256 = "b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4"
TARGET_ARTIFACT_PINS = {
    "model.safetensors": TARGET_CANDIDATE_MERGED_WEIGHTS_SHA256,
    "config.json": "f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991",
    "generation_config.json": "7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269",
    "tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config.json": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}

TOTAL_WINDOW_SECONDS = 7200
PROVIDER_TIMEOUT_SECONDS = 6900
WATCHDOG_TIMEOUT_SECONDS = 7200
SETUP_SECONDS = 1800
PROFILE_SECONDS = 3600
UPLOAD_SECONDS = 600
FINAL_REFRESH_RESERVE_SECONDS = 900
WATCHDOG_TAIL_SECONDS = 300
PROFILE_ROWS = 8
PROFILE_STEPS = 8
MAX_RETRIES = 0
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PRIVATE_KEY", "ACCESS_KEY")

EXPECTED_BUDGET = {
    "setup_seconds": SETUP_SECONDS,
    "profile_seconds": PROFILE_SECONDS,
    "upload_seconds": UPLOAD_SECONDS,
    "final_refresh_reserve_seconds": FINAL_REFRESH_RESERVE_SECONDS,
    "watchdog_tail_seconds": WATCHDOG_TAIL_SECONDS,
}

STOP_REQUESTED = False


class EntryError(RuntimeError):
    """A root binding, staged-input, phase, or persistence gate failed."""


def stop_requested(*_signals: int) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise EntryError(reason)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EntryError("cloud deadline is not ISO-8601") from exc
    require(parsed.tzinfo is not None, "cloud deadline must include a UTC offset")
    return parsed.timestamp()


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _credential_key(value: Any) -> bool:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")
    return normalized in SECRET_MARKERS or any(normalized.endswith("_" + marker) for marker in SECRET_MARKERS)


def _forbidden_binding_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _credential_key(key)
            or _forbidden_binding_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_forbidden_binding_key(child) for child in value)
    return False


def validate_binding(binding: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    """Validate a root-armed, single-attempt 7,200-second cloud window."""
    now = time.time() if now is None else float(now)
    require(binding.get("schema") == 2 and binding.get("admitted") is True, "root cloud admission is pending")
    require(re.fullmatch(r"[0-9a-f]{32}", str(binding.get("run_id", ""))) is not None, "fresh UUID32 is required")
    require(binding.get("image_uri") == IMAGE and binding.get("instance_type") == INSTANCE_TYPE, "image or instance differs")
    require(binding.get("max_retries") == MAX_RETRIES, "provider retries must remain zero")
    require(binding.get("provider_timeout_seconds") == PROVIDER_TIMEOUT_SECONDS, "provider timeout differs")
    require(binding.get("watchdog_timeout_seconds") == WATCHDOG_TIMEOUT_SECONDS, "watchdog timeout differs")
    require(re.fullmatch(r"[0-9a-f]{64}", str(binding.get("watchdog_armed_receipt_sha256", ""))) is not None, "root watchdog receipt hash is missing")
    armed = utc(str(binding.get("watchdog_armed_at_utc", "")))
    absolute = utc(str(binding.get("absolute_deadline_utc", "")))
    require(armed <= now < absolute, "cloud binding is outside its absolute window")
    require(absolute <= armed + WATCHDOG_TIMEOUT_SECONDS, "absolute deadline exceeds the independent watchdog")
    hard = min(armed + PROVIDER_TIMEOUT_SECONDS, absolute)
    require(now < hard, "provider deadline has expired")
    budget = binding.get("budget")
    require(isinstance(budget, Mapping) and dict(budget) == EXPECTED_BUDGET, "cloud phase budgets differ")
    require(sum(EXPECTED_BUDGET.values()) == TOTAL_WINDOW_SECONDS, "cloud phase budgets do not cover exactly 7,200 seconds")
    work_deadline = min(hard, absolute - FINAL_REFRESH_RESERVE_SECONDS - WATCHDOG_TAIL_SECONDS)
    require(work_deadline > now, "no setup/profile/upload budget remains before final refresh reserve")
    require(binding.get("artifact_repo") == "scholzmx/sepalith-lora", "private artifact repository differs")
    prefix = str(binding.get("artifact_prefix", ""))
    require(prefix.startswith("r2-draft-profile/") and prefix.endswith("/" + str(binding["run_id"])), "private artifact prefix must be fresh and run-bound")
    require(isinstance(binding.get("staging"), Mapping), "explicit staged input binding is missing")
    required_staging = {
        "root", "requirements_candidate", "requirements_resolved", "target_model_dir",
        "source_root", "target_manifest", "train_data", "public_draft_weights", "public_draft_config",
    }
    missing = sorted(required_staging - set(binding["staging"]))
    require(not missing, "staging binding missing: " + ", ".join(missing))
    bootstrap = binding.get("bootstrap")
    require(isinstance(bootstrap, Mapping), "managed Python bootstrap binding is missing")
    require(bootstrap.get("uv_version") == UV_VERSION, "uv bootstrap version differs")
    for key in ("managed_python_root", "venv_root", "uv_tools_root", "uv_cache_root"):
        value = bootstrap.get(key)
        require(isinstance(value, str) and value not in ("", ".") and not Path(value).is_absolute() and ".." not in Path(value).parts, "bootstrap path is not run-relative")
    require(isinstance(binding.get("pins"), Mapping), "explicit source and artifact pins are missing")
    validate_download_binding(binding)
    require(isinstance(binding.get("run_root"), str), "fresh run root is missing")
    require(binding.get("install_dependencies") is True, "resolved candidate dependencies require explicit install admission")
    require(not _forbidden_binding_key(binding), "credential-bearing binding key is forbidden")
    return {
        "armed_at": armed,
        "absolute_deadline": absolute,
        "hard_deadline": hard,
        "work_deadline": work_deadline,
        "budget": dict(EXPECTED_BUDGET),
    }


def phase_deadline(start: float, cap_seconds: int, work_deadline: float, now: float | None = None) -> float:
    now = time.time() if now is None else float(now)
    require(now < work_deadline, "phase starts after the reserved final-refresh window")
    return min(work_deadline, float(start) + int(cap_seconds))


def make_run_dir(root: str | Path, run_id: str) -> Path:
    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run = base / ("run-" + run_id)
    require(not run.exists(), "implicit cloud run resume is forbidden")
    run.mkdir()
    for name in ("logs", "profile"):
        (run / name).mkdir()
    return run


def _check_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), label + " is missing or symlinked")
    digest = sha256_file(path)
    require(digest == expected_sha256, label + " SHA-256 differs")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _validate_source_inventory(source_root: Path) -> dict[str, Any]:
    inventory_path = source_root / PROFILE_INVENTORY_RELATIVE
    _check_file(inventory_path, PROFILE_INVENTORY_SHA256, "profile source inventory")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EntryError("profile source inventory is invalid JSON") from exc
    require(inventory.get("portable") is True, "profile source inventory is not portable")
    rows = list(inventory.get("files", [])) + list(inventory.get("requirements", []))
    require(rows, "profile source inventory is empty")
    checked = []
    for row in rows:
        relative = str(row.get("path", ""))
        require(not Path(relative).is_absolute() and ".git" not in Path(relative).parts and "dependency-overlay" not in Path(relative).parts, "non-portable source inventory path")
        path = source_root / relative
        checked.append(_check_file(path, str(row.get("sha256")), "inventory source " + relative))
    profile = _check_file(source_root / PROFILE_RELATIVE, PROFILE_SOURCE_SHA256, "profile orchestrator")
    runtime = _check_file(source_root / TARGET_RUNTIME_RELATIVE, TARGET_RUNTIME_SHA256, "target runtime")
    return {
        "inventory_path": str(inventory_path),
        "inventory_sha256": PROFILE_INVENTORY_SHA256,
        "files_checked": len(checked),
        "profile": profile,
        "target_runtime": runtime,
        "deepspec_revision": DEEPSPEC_REVISION,
    }


def validate_staged_inputs(
    binding: Mapping[str, Any],
    payload_root: str | Path,
    *,
    target_manifest_override: str | Path | None = None,
    target_model_dir_override: str | Path | None = None,
    public_weights_override: str | Path | None = None,
    public_config_override: str | Path | None = None,
    verify_target_artifacts: bool = True,
    verify_public_artifacts: bool = True,
) -> dict[str, Any]:
    """Verify staged metadata and downloaded artifacts without loading tensors."""
    staging = binding["staging"]
    stage_root = bounded_path(payload_root, str(staging["root"]))
    require(stage_root.is_dir(), "staged payload root is missing")
    source_root = bounded_path(stage_root, str(staging["source_root"]))
    require(source_root.is_dir(), "staged source root is missing")
    pins = binding["pins"]
    require(str(pins.get("deepspec_revision")) == DEEPSPEC_REVISION, "DeepSpec revision pin differs")
    require(str(pins.get("train_sha256")) == TRAIN_SOURCE_SHA256, "TRAIN source pin differs")
    require(str(pins.get("public_draft_weights_sha256")) == PUBLIC_DRAFT_WEIGHTS_SHA256, "public draft weight pin differs")
    require(str(pins.get("public_draft_config_sha256")) == PUBLIC_DRAFT_CONFIG_SHA256, "public draft config pin differs")
    require(str(pins.get("target_candidate_merged_weights_sha256")) == TARGET_CANDIDATE_MERGED_WEIGHTS_SHA256, "target candidate pin differs")
    source_manifest_path = bounded_path(stage_root, str(staging["target_manifest"]))
    target_manifest = Path(target_manifest_override).expanduser().resolve() if target_manifest_override is not None else source_manifest_path
    target_model_dir = (
        Path(target_model_dir_override).expanduser().resolve()
        if target_model_dir_override is not None
        else bounded_path(stage_root, str(staging["target_model_dir"]))
    )
    train_data = bounded_path(stage_root, str(staging["train_data"]))
    public_weights = (
        Path(public_weights_override).expanduser().resolve()
        if public_weights_override is not None
        else bounded_path(stage_root, str(staging["public_draft_weights"]))
    )
    public_config = (
        Path(public_config_override).expanduser().resolve()
        if public_config_override is not None
        else bounded_path(stage_root, str(staging["public_draft_config"]))
    )
    candidate_requirements = bounded_path(stage_root, str(staging["requirements_candidate"]))
    resolved_requirements = bounded_path(stage_root, str(staging["requirements_resolved"]))
    # Binding hashes make each relocation explicit.  Only the public weight and
    # TRAIN bytes are large; they are checked here before any profile loader.
    source_manifest = _check_file(source_manifest_path, str(pins.get("target_manifest_sha256")), "source target model manifest")
    if target_manifest != source_manifest_path:
        target = _check_file(target_manifest, sha256_file(target_manifest), "relocated target model manifest")
    else:
        target = source_manifest
    try:
        target_manifest_record = json.loads(target_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EntryError("target model manifest is invalid JSON") from exc
    require(isinstance(target_manifest_record, Mapping), "target model manifest must be an object")
    require(target_manifest_record.get("deepspec_revision") == DEEPSPEC_REVISION, "target manifest DeepSpec revision differs")
    manifest_model_dir = Path(str(target_manifest_record.get("model_dir", ""))).expanduser()
    if not manifest_model_dir.is_absolute():
        manifest_model_dir = target_manifest.parent / manifest_model_dir
    require(manifest_model_dir.resolve() == target_model_dir.resolve(), "target manifest model directory is not the explicit staged directory")
    require(target_manifest_record.get("weights_file") == "model.safetensors", "target weights filename must be model.safetensors")
    require(target_manifest_record.get("config_file", "config.json") == "config.json", "target config filename must be config.json")
    require(target_manifest_record.get("weights_sha256") == TARGET_ARTIFACT_PINS["model.safetensors"], "target manifest weight pin differs")
    require(target_manifest_record.get("config_sha256") == TARGET_ARTIFACT_PINS["config.json"], "target manifest config pin differs")
    require(target_manifest_record.get("merged_weights_sha256", target_manifest_record.get("weights_sha256")) == TARGET_CANDIDATE_MERGED_WEIGHTS_SHA256, "target manifest merged-weight pin differs")
    target_artifacts = []
    if verify_target_artifacts:
        require(target_model_dir.is_dir() and not target_model_dir.is_symlink(), "target model directory is missing or symlinked")
        manifest_model_dir = Path(str(target_manifest_record.get("model_dir", ""))).expanduser()
        if not manifest_model_dir.is_absolute():
            manifest_model_dir = target_manifest.parent / manifest_model_dir
        require(manifest_model_dir.resolve() == target_model_dir.resolve(), "target manifest model directory is not the explicit staged directory")
        for filename, expected_sha256 in TARGET_ARTIFACT_PINS.items():
            target_artifacts.append(_check_file(target_model_dir / filename, expected_sha256, "target artifact " + filename))
        for artifact in target_model_dir.iterdir():
            name = artifact.name
            weight_like = name.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5"))
            require(not name.endswith(".index.json") and not (weight_like and name != "model.safetensors"), "sharded or alternative target artifact is forbidden: " + name)
    train = _check_file(train_data, TRAIN_SOURCE_SHA256, "TRAIN source")
    public = None
    config = None
    if verify_public_artifacts:
        public = _check_file(public_weights, PUBLIC_DRAFT_WEIGHTS_SHA256, "public draft weights")
        require(public["bytes"] == PUBLIC_DRAFT_WEIGHTS_BYTES, "public draft weight size differs")
        require(public_weights.name == "model.safetensors", "public draft weight filename must be model.safetensors")
        require(public_config.name == "config.json", "public draft config filename must be config.json")
        config = _check_file(public_config, PUBLIC_DRAFT_CONFIG_SHA256, "public draft sibling config")
        require(public_weights.parent == public_config.parent, "public draft weights and config must share a staged directory")
    candidate = _check_file(candidate_requirements, CANDIDATE_REQUIREMENTS_SHA256, "candidate requirements")
    resolved = _check_file(resolved_requirements, RESOLVED_REQUIREMENTS_SHA256, "resolved requirements")
    source = _validate_source_inventory(source_root)
    return {
        "stage_root": str(stage_root),
        "source_root": str(source_root),
        "target_manifest": target,
        "target_model_dir": str(target_model_dir.resolve()),
        "target_artifacts": target_artifacts,
        "train": train,
        "public_draft_weights": public,
        "public_draft_config": config,
        "requirements_candidate": candidate,
        "requirements_resolved": resolved,
        "source_inventory": source,
    }


def clean_environment(
    *,
    run: Path,
    inputs: Mapping[str, Any],
    source_root: Path,
    bootstrap_layout: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    """Make training environment paths explicit while leaving HOME unchanged."""
    environment = dict(os.environ)
    for key in list(environment):
        if any(marker in key.upper() for marker in SECRET_MARKERS):
            environment.pop(key, None)
    cache_roots = {
        "tmp": run / "tmp",
        "hf": run / "hf-cache",
        "xdg": run / "xdg-cache",
        "torch": run / "torch-cache",
        "triton": run / "triton-cache",
    }
    for path in cache_roots.values():
        path.mkdir(parents=True, exist_ok=True)
    bootstrap = dict(bootstrap_layout or bootstrap_paths(run / "runtime"))
    for name, path in bootstrap.items():
        # uv creates the tools, managed-Python, and venv directories.  Keep
        # only their parents ready; an existing venv would look like a resume.
        if name == "uv_cache":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
    profile_root = source_root / "docs/campaign/work/r2-draft-cloud-profile-v2"
    target_root = source_root / "docs/campaign/work/r2-draft-target-runtime-v3"
    vendor_root = source_root / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec"
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
            "SEPALITH_PROFILE_ADMITTED": "1",
            "TMPDIR": str(cache_roots["tmp"]),
            "TEMP": str(cache_roots["tmp"]),
            "TMP": str(cache_roots["tmp"]),
            "HF_HOME": str(cache_roots["hf"]),
            "XDG_CACHE_HOME": str(cache_roots["xdg"]),
            "TORCH_HOME": str(cache_roots["torch"]),
            "TRITON_CACHE_DIR": str(cache_roots["triton"]),
            "PIP_CACHE_DIR": str(run / "pip-cache"),
            "UV_CACHE_DIR": str(bootstrap["uv_cache"]),
            "UV_PYTHON_INSTALL_DIR": str(bootstrap["python_root"]),
            "UV_NO_PROGRESS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "SEPALITH_PROFILE_SOURCE_ROOT": str(source_root),
            "SEPALITH_DEEPSPEC_ROOT": str(vendor_root),
            "SEPALITH_HARDENING_ROOT": str(source_root / "docs/campaign/work/r2-draft-adapter-hardening-v1"),
            "SEPALITH_DRAFT_CACHE": str(run / "ephemeral-target-cache"),
            "SEPALITH_DRAFT_OUTPUT_ROOT": str(run / "checkpoints"),
            "SEPALITH_DRAFT_LOG_ROOT": str(run / "tensorboard"),
            "SEPALITH_PROFILE_RUN_DIR": str(run),
            "SEPALITH_PROFILE_TELEMETRY_PATH": str(run / "stages" / "warmstart-trainer-profile.gpu.jsonl"),
        }
    )
    public = inputs.get("public_draft_weights")
    if isinstance(public, Mapping) and public.get("path"):
        environment["SEPALITH_PUBLIC_DRAFT_WEIGHTS"] = str(public["path"])
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (target_root, source_root / "docs/campaign/work/r2-draft-adapter-hardening-v1", source_root / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation", vendor_root)
    )
    return environment


def _terminate_child(process: subprocess.Popen[bytes]) -> None:
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


def run_child(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path, deadline: float, log: Path) -> dict[str, Any]:
    """Run one owned child process until its phase deadline."""
    require(time.time() < deadline, "child phase has no remaining deadline")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reason: str | None = None
    process: subprocess.Popen[bytes] | None = None
    with log.open("wb") as output:
        process = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            if STOP_REQUESTED:
                reason = "entry termination requested"
            elif time.time() >= deadline:
                reason = "phase deadline"
            if reason:
                _terminate_child(process)
                break
            time.sleep(0.25)
        try:
            exit_code = process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _terminate_child(process)
            exit_code = process.poll()
    elapsed = time.monotonic() - started
    result = {
        "at": stamp(),
        "status": "succeeded" if reason is None and exit_code == 0 else "failed",
        "argv_program": Path(argv[0]).name,
        "exit_code": exit_code,
        "reason": reason,
        "elapsed_seconds": elapsed,
        "deadline": deadline,
        "log": str(log),
    }
    write_json(str(log) + ".terminal.json", result)
    require(result["status"] == "succeeded", f"child phase failed: {result['reason'] or 'nonzero exit'}")
    return result


def profile_command(*, python: Path, profile_source: Path, profile_run: Path, target_manifest: Path, train_data: Path, source_root: Path) -> list[str]:
    return [
        str(python), str(profile_source), "--execute",
        "--run-dir", str(profile_run),
        "--target-model-manifest", str(target_manifest),
        "--train-data-path", str(train_data),
        "--source-root", str(source_root),
    ]


def _upload_command(python: Path, mode: str, run: Path) -> list[str]:
    require(mode in {"sentinel", "success", "failure"}, "invalid artifact upload mode")
    return [str(python), str(HERE / "artifact_upload.py"), mode, str(run)]


def _download_command(python: Path, binding: Path, payload_root: Path, run: Path) -> list[str]:
    return [str(python), str(HERE / "download_staging.py"), str(binding), str(payload_root), str(run)]


def _copy_sentinel_receipt(profile_run: Path, cloud_run: Path) -> None:
    source = cloud_run / "sentinel-receipt.json"
    require(source.is_file(), "private sentinel receipt is missing")
    destination = profile_run / "durable" / "sentinel-receipt.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _failure_record(error: BaseException, phase: str, deadlines: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "at": stamp(),
        "status": "failed; no success admission",
        "phase": phase,
        "error_type": type(error).__name__,
        "reason": str(error)[:240],
        "cloud_launch": False,
        "deadlines": dict(deadlines),
    }


def _bootstrap_layout(run: Path, binding: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve all interpreter/bootstrap paths below this fresh cloud run."""
    spec = binding["bootstrap"]
    layout = {
        key: bounded_path(run, str(spec[name]))
        for key, name in (
            ("tools", "uv_tools_root"),
            ("python_root", "managed_python_root"),
            ("venv", "venv_root"),
            ("uv_cache", "uv_cache_root"),
        )
    }
    values = list(layout.values())
    require(len({str(path) for path in values}) == len(values), "bootstrap paths must be distinct")
    return layout


def run_entry(binding_path: str | Path, *, now: float | None = None) -> dict[str, Any]:
    """Execute the staged profile after root admission; no retries are attempted."""
    binding_path = Path(binding_path).expanduser().resolve(strict=True)
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EntryError("cloud binding is not valid JSON") from exc
    require(isinstance(binding, dict), "cloud binding must be an object")
    deadlines = validate_binding(binding, now=now)
    payload_root = binding_path.parent
    run_root = bounded_path(payload_root, str(binding["run_root"]))
    run = make_run_dir(run_root, str(binding["run_id"]))
    write_json(run / "binding.json", binding)
    profile_run = run / "profile"
    phase = "staging-preflight"
    success = False
    upload_success = False
    failure: dict[str, Any] | None = None
    python: Path | None = None
    runtime: dict[str, Any] | None = None
    environment: dict[str, str] | None = None
    started = time.time() if now is None else float(now)
    setup_deadline = phase_deadline(started, SETUP_SECONDS, deadlines["work_deadline"])
    try:
        validate_download_binding(binding)
        # Validate all small shipped metadata and source identities before
        # spending time or bandwidth on either model download.  Model files
        # are verified in the second call after the isolated downloader exits.
        static_inputs = validate_staged_inputs(
            binding,
            payload_root,
            verify_target_artifacts=False,
            verify_public_artifacts=False,
        )
        source_root = Path(static_inputs["source_root"])
        candidate = Path(static_inputs["requirements_candidate"]["path"])
        resolved = Path(static_inputs["requirements_resolved"]["path"])
        candidate_report = validate_candidate_requirements(candidate, CANDIDATE_REQUIREMENTS_SHA256, sha256_file=sha256_file)
        resolved_report = validate_candidate_requirements(resolved, RESOLVED_REQUIREMENTS_SHA256, sha256_file=sha256_file)
        layout = _bootstrap_layout(run, binding)
        environment = clean_environment(
            # Keep parent bootstrap/cache paths outside the profile run.  The
            # profile's own prepare_run_dir requires an empty directory before
            # it creates its child caches and durable receipts.
            run=run / "environment",
            inputs=static_inputs,
            source_root=source_root,
            bootstrap_layout=layout,
        )
        commands = bootstrap_commands(sys.executable, run, paths=layout)
        phase = "managed-python-bootstrap"
        run_child(
            [str(Path(sys.executable).resolve()), "-m", "pip", "--version"],
            env=environment,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/image-pip-probe.log",
        )
        run_child(
            commands["uv_install"],
            env=environment,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/uv-install.log",
        )
        uv = layout["tools"] / "bin" / "uv"
        require(uv.is_file() and not uv.is_symlink() and os.access(uv, os.X_OK), "pinned uv executable is missing")
        run_child(
            commands["uv_version"],
            env=environment,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/uv-version.log",
        )
        uv_version_output = (run / "logs/uv-version.log").read_text(encoding="utf-8").strip().split()
        require(uv_version_output[:2] == ["uv", UV_VERSION], "uv bootstrap version differs")
        run_child(
            commands["python_install"],
            env=environment,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/python-install.log",
        )
        run_child(
            commands["python_find"],
            env=environment,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/python-find.log",
        )
        managed_python = discover_result(
            (run / "logs/python-find.log").read_text(encoding="utf-8"),
            layout["python_root"],
        )
        runtime = probe_runtime(managed_python, layout["python_root"])
        venv_command = [
            token.replace("<discovered-python>", str(managed_python))
            for token in commands["venv"]
        ]
        run_child(
            venv_command,
            env=environment,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/venv-bootstrap.log",
        )
        python = layout["venv"] / "bin" / "python"
        require(python.is_file() and os.access(python, os.X_OK), "managed venv executable is missing")
        venv_target = python.resolve(strict=True)
        require(
            venv_target == managed_python or venv_target.is_relative_to(layout["python_root"]),
            "managed venv executable resolves outside the verified CPython install",
        )
        if binding.get("install_dependencies"):
            run_child(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--requirement",
                    str(resolved),
                ],
                env=environment,
                cwd=HERE,
                deadline=setup_deadline,
                log=run / "logs/dependency-install.log",
            )
        phase = "model-download-staging"
        download_env = dict(environment)
        # Hugging Face offline mode is correct for profile execution, but the
        # dedicated downloader must be explicitly network-enabled.  The token
        # exists only in this child environment and is scrubbed on child exit.
        download_env["HF_HUB_OFFLINE"] = "0"
        download_env["TRANSFORMERS_OFFLINE"] = "0"
        if "HF_TOKEN" in os.environ:
            download_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
        run_child(
            _download_command(python, run / "binding.json", payload_root, run),
            env=download_env,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/model-download.log",
        )
        try:
            download_report = json.loads((run / "staging-download-receipt.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EntryError("model download receipt is invalid") from exc
        require(download_report.get("status") == "model_downloads_staged", "model download staging did not pass")
        effective_manifest = Path(download_report["effective_manifest"]["path"]).resolve(strict=True)
        target_model_dir = Path(download_report["effective_manifest"]["model_dir"]).resolve(strict=True)
        public_files = {str(row["filename"]): Path(row["path"]).resolve(strict=True) for row in download_report["public_draft"]["files"]}
        inputs = validate_staged_inputs(
            binding,
            payload_root,
            target_manifest_override=effective_manifest,
            target_model_dir_override=target_model_dir,
            public_weights_override=public_files["model.safetensors"],
            public_config_override=public_files["config.json"],
        )
        environment["SEPALITH_PUBLIC_DRAFT_WEIGHTS"] = str(inputs["public_draft_weights"]["path"])
        imports = check_imports(python, environment=environment)
        phase = "private-sentinel"
        upload_env = dict(environment)
        upload_env["HF_HUB_OFFLINE"] = "0"
        upload_env["TRANSFORMERS_OFFLINE"] = "0"
        for key, value in os.environ.items():
            if key == "HF_TOKEN":
                upload_env[key] = value
        run_child(
            _upload_command(python, "sentinel", run),
            env=upload_env,
            cwd=HERE,
            deadline=setup_deadline,
            log=run / "logs/sentinel.log",
        )
        phase = "profile"
        profile_started = time.time()
        profile_deadline = phase_deadline(profile_started, PROFILE_SECONDS, deadlines["work_deadline"])
        run_child(
            profile_command(
                python=python,
                profile_source=source_root / PROFILE_RELATIVE,
                profile_run=profile_run,
                target_manifest=Path(inputs["target_manifest"]["path"]),
                train_data=Path(inputs["train"]["path"]),
                source_root=source_root,
            ),
            env=environment,
            cwd=source_root,
            deadline=profile_deadline,
            log=run / "logs/profile.log",
        )
        profile_terminal = profile_run / "profile-terminal.json"
        require(profile_terminal.is_file(), "profile terminal receipt is missing")
        profile_report = json.loads(profile_terminal.read_text(encoding="utf-8"))
        require(profile_report.get("status") == "profile_passed", "profile did not pass its terminal gates")
        _copy_sentinel_receipt(profile_run, run)
        write_json(profile_run / "entry-terminal.json", {"status": "profile_passed_upload_pending", "at": stamp(), "cloud_launch": False})
        phase = "durable-stage"
        staged = stage_durable(profile_run, run / "durable-upload")
        phase = "durable-upload"
        upload_deadline = phase_deadline(time.time(), UPLOAD_SECONDS, deadlines["work_deadline"])
        run_child(
            _upload_command(python, "success", run),
            env=upload_env,
            cwd=HERE,
            deadline=upload_deadline,
            log=run / "logs/upload-success.log",
        )
        success = True
        upload_success = True
        result = {
            "schema": 1,
            "status": "cloud_entry_succeeded",
            "run_id": binding["run_id"],
            "image_uri": IMAGE,
            "instance_type": INSTANCE_TYPE,
            "max_retries": MAX_RETRIES,
            "managed_python": runtime,
            "bootstrap": {
                "uv_version": UV_VERSION,
                "uv_spec": UV_SPEC,
                "managed_python_bootstrapped": True,
                "venv": str(python),
            },
            "candidate_requirements": candidate_report,
            "resolved_requirements": resolved_report,
            "imports": imports,
            "profile_rows": PROFILE_ROWS,
            "profile_steps": PROFILE_STEPS,
            "durable_stage": staged,
            "profile_limit_seconds": PROFILE_SECONDS,
            "total_window_seconds": TOTAL_WINDOW_SECONDS,
            "final_refresh_reserved_seconds": FINAL_REFRESH_RESERVE_SECONDS,
            "cloud_launch": False,
        }
        write_json(run / "entry-terminal.json", result)
        return result
    except Exception as error:
        failure = _failure_record(error, phase, deadlines)
        write_json(profile_run / "entry-failure.json", failure)
        try:
            if (run / "sentinel-receipt.json").is_file() and profile_run.is_dir():
                _copy_sentinel_receipt(profile_run, run)
            if not (run / "durable-upload").exists():
                stage_durable(profile_run, run / "durable-upload")
            else:
                write_json(run / "durable-upload/entry-failure.json", failure)
        except Exception as stage_error:
            failure["durable_stage_error_type"] = type(stage_error).__name__
            failure["durable_stage_error"] = str(stage_error)[:160]
        python_path = locals().get("python")
        if isinstance(python_path, Path) and python_path.is_file():
            try:
                upload_env = dict(environment or {})
                upload_env["HF_HUB_OFFLINE"] = "0"
                upload_env["TRANSFORMERS_OFFLINE"] = "0"
                for key, value in os.environ.items():
                    if key == "HF_TOKEN":
                        upload_env[key] = value
                failure_deadline = phase_deadline(time.time(), UPLOAD_SECONDS, deadlines["work_deadline"])
                run_child(
                    _upload_command(python_path, "failure", run),
                    env=upload_env,
                    cwd=HERE,
                    deadline=failure_deadline,
                    log=run / "logs/upload-failure.log",
                )
                failure["failure_upload"] = "verified"
            except Exception as upload_error:
                failure["failure_upload"] = "failed"
                failure["failure_upload_error_type"] = type(upload_error).__name__
        write_json(run / "entry-terminal.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binding", type=Path)
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_requested)
    result = run_entry(args.binding)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "cloud_entry_rejected", "error_type": type(exc).__name__, "reason": str(exc)[:240]}))
        raise SystemExit(1)
