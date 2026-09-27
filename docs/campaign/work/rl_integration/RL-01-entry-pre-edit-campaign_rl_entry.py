#!/usr/bin/env python3
"""Admitted, supervised PRM-03 RL entry point.

``campaign_rl_train`` owns the fixed-ID trainer and reward seam.  This module
owns the boundary around that seam: it verifies a future merged-SFT parent,
loads the local BF16 model, attaches the pinned RL adapter, restores the
tokenizer contract, and runs (or resumes) one supervised attempt.  Import and
preflight stay framework-free; model and CUDA imports happen only after every
recipe, data, parent, output, and resource gate has passed.

The parent manifest contract is intentionally explicit.  A recipe contains a
``parent_manifest`` object with ``path`` and ``sha256``.  That file is a
``sepalith.merged-sft.parent-manifest.v1`` JSON object with ``status=accepted``,
``kind=merged_sft``, an absolute ``merged_model_path``, hashes for the exact
``config.json`` and ``generation_config.json`` bytes, a nonempty complete
``weight_inventory`` (relative path, byte count, and SHA256 for every weight),
the aggregate ``merged_weights_sha256`` identity, base revision, SFT identity,
and the pinned tokenizer file hashes/IDs.  The aggregate weight identity is
the SHA256 of the canonical JSON inventory entries, so all listed bytes are
bound without reading weights into memory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


# Keep this entry importable from ``experiments/training``.  None of the
# imports below load torch, Transformers, TRL, Datasets, or Unsloth.
from campaign_checkpoint import verify_checkpoint, write_json
from campaign_control import utc_deadline
from campaign_rl_data import (
    RLDataError,
    load_training_records,
    sha256_file,
)
from campaign_rl_train import (
    COMPLETION_MAX_TOKENS,
    CONTEXT_MAX_TOKENS,
    DEFAULT_CANDIDATE_COUNTS,
    DEFAULT_FULL_SAVE_STEPS,
    DEFAULT_LIGHT_SAVE_STEPS,
    DEFAULT_SEED,
    EXPECTED_ATTACHMENTS,
    EXPECTED_TRAINABLE_PARAMETERS,
    LORA_ALPHA,
    LORA_RANK,
    NATIVE_EOG_IDS,
    PROMPT_MAX_TOKENS,
    TARGET_MODULES,
    TRAIN_SCHEMA_VERSION,
    RLTrainError,
    build_live_trainer,
    preflight_rl_recipe,
)
from sepalith.campaign_protocol import BOS_ID, EOS_ID, VOCAB_SIZE


ENTRY_SCHEMA_VERSION = "sepalith.prm07.rl-entry.v1"
PARENT_MANIFEST_SCHEMA_VERSION = "sepalith.merged-sft.parent-manifest.v1"
CUDA_OCCUPANCY_LIMIT_MIB = 4_096
PARENT_REQUIRED_MODEL_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth"})


class RLEntryError(ValueError):
    """An RL entry recipe or live boundary failed closed validation."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RLEntryError(f"{name} must be an object")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RLEntryError(f"{name} must be a lowercase SHA256")
    return value


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise RLEntryError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RLEntryError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RLEntryError(f"{name} must be finite and positive")
    return number


def _absolute_path(value: object, name: str, *, file: bool | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise RLEntryError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise RLEntryError(f"{name} must be an absolute path")
    if file is True and not path.is_file():
        raise RLEntryError(f"{name} must be an existing file: {path}")
    if file is False and not path.is_dir():
        raise RLEntryError(f"{name} must be an existing directory: {path}")
    return path.resolve()


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RLEntryError(f"cannot read {name}: {path}") from error
    return _mapping(value, name)


def _canonical_inventory(entries: Sequence[Mapping[str, Any]]) -> bytes:
    value = [
        {"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
        for entry in entries
    ]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def weight_inventory_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    """Return the aggregate identity for a complete merged-weight inventory."""
    return hashlib.sha256(_canonical_inventory(entries)).hexdigest()


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def _filesystem_type(path: Path) -> str:
    """Read the filesystem type without importing a framework."""
    existing = _nearest_existing(path)
    try:
        result = subprocess.run(
            ["df", "-T", "-P", str(existing)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RLEntryError(f"cannot inspect filesystem for {path}: {error}") from error
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2 or len(lines[-1]) < 2:
        raise RLEntryError(f"filesystem probe returned no type for {path}")
    # POSIX df -P has Filesystem, 1024-blocks, Used, Available, Capacity,
    # Mounted-on; -T inserts the type after Filesystem.
    if len(lines[-1]) < 7:
        raise RLEntryError(f"filesystem probe returned malformed output for {path}")
    return lines[-1][1]


def assert_fresh_native_ext4_path(path: str | Path, name: str) -> Path:
    """Require a new path whose destination filesystem is local native ext4."""
    value = Path(path)
    if not value.is_absolute():
        raise RLEntryError(f"{name} must be an absolute path")
    if value.exists() or value.is_symlink():
        raise RLEntryError(f"{name} must be fresh and must not already exist: {value}")
    filesystem = _filesystem_type(value)
    if filesystem != "ext4":
        raise RLEntryError(
            f"{name} must be on native ext4; found {filesystem!r} at {_nearest_existing(value)}"
        )
    return value


def _assert_distinct_paths(output: Path, archive: Path) -> None:
    output_resolved, archive_resolved = output.resolve(), archive.resolve()
    if output_resolved == archive_resolved:
        raise RLEntryError("output_dir and archive_root must be distinct")
    try:
        output_resolved.relative_to(archive_resolved)
    except ValueError:
        pass
    else:
        raise RLEntryError("output_dir cannot be inside archive_root")
    try:
        archive_resolved.relative_to(output_resolved)
    except ValueError:
        pass
    else:
        raise RLEntryError("archive_root cannot be inside output_dir")


def _weight_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            result.add(path.relative_to(root).as_posix())
    return result


def verify_merged_parent_manifest(
    wrapper: Mapping[str, Any],
    *,
    verify_weights: bool = True,
) -> dict[str, Any]:
    """Verify the manifest bytes, tokenizer, model path, and every weight file."""
    if set(wrapper) != {"path", "sha256"}:
        raise RLEntryError("parent_manifest must contain exactly path and sha256")
    manifest_path = _absolute_path(wrapper["path"], "parent_manifest.path", file=True)
    expected_manifest_sha = _sha(wrapper["sha256"], "parent_manifest.sha256")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RLEntryError(f"parent manifest SHA256 mismatch: {manifest_path}")
    manifest = _read_json(manifest_path, "merged-SFT parent manifest")
    if manifest.get("schema_version") != PARENT_MANIFEST_SCHEMA_VERSION:
        raise RLEntryError("merged-SFT parent manifest schema mismatch")
    if manifest.get("status") != "accepted" or manifest.get("kind") != "merged_sft":
        raise RLEntryError("merged-SFT parent manifest must be accepted and kind=merged_sft")
    if "manifest_sha256" in manifest and manifest["manifest_sha256"] != actual_manifest_sha:
        raise RLEntryError("merged-SFT parent manifest self-hash differs from wrapper")
    model_path = _absolute_path(manifest.get("merged_model_path"), "manifest.merged_model_path", file=False)
    for filename in PARENT_REQUIRED_MODEL_FILES:
        if not (model_path / filename).is_file():
            raise RLEntryError(f"merged-SFT parent is missing {filename}: {model_path}")
    for filename, field in (
        ("config.json", "config_sha256"),
        ("generation_config.json", "generation_config_sha256"),
    ):
        expected_sha = _sha(manifest.get(field), f"manifest.{field}")
        if sha256_file(model_path / filename) != expected_sha:
            raise RLEntryError(f"merged-SFT {filename} SHA256 differs from parent manifest")
    base_revision = manifest.get("base_model_revision")
    if not isinstance(base_revision, str) or not base_revision:
        raise RLEntryError("manifest.base_model_revision is required")
    merged_weights_sha = _sha(manifest.get("merged_weights_sha256"), "manifest.merged_weights_sha256")
    sft_identity = _mapping(manifest.get("sft_identity"), "manifest.sft_identity")
    if not sft_identity:
        raise RLEntryError("manifest.sft_identity must be nonempty")

    tokenizer = _mapping(manifest.get("tokenizer"), "manifest.tokenizer")
    for name in ("tokenizer_json_sha256", "tokenizer_config_sha256"):
        _sha(tokenizer.get(name), f"manifest.tokenizer.{name}")
    if (
        tokenizer.get("vocab_size") != VOCAB_SIZE
        or tokenizer.get("bos_id") != BOS_ID
        or tokenizer.get("eos_id") != EOS_ID
        or tokenizer.get("pad_id") != EOS_ID
        or list(tokenizer.get("native_eog_ids", ())) != list(NATIVE_EOG_IDS)
    ):
        raise RLEntryError("manifest.tokenizer does not match the pinned PRM03 contract")
    for filename, field in (
        ("tokenizer.json", "tokenizer_json_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        if sha256_file(model_path / filename) != tokenizer[field]:
            raise RLEntryError(f"merged-SFT {filename} SHA256 differs from parent manifest")

    inventory_value = manifest.get("weight_inventory")
    if not isinstance(inventory_value, list) or not inventory_value:
        raise RLEntryError("manifest.weight_inventory must be a nonempty complete list")
    declared_paths = [
        entry.get("path") if isinstance(entry, Mapping) else None
        for entry in inventory_value
    ]
    if declared_paths != sorted(declared_paths, key=lambda value: str(value)):
        raise RLEntryError("manifest.weight_inventory must be sorted by relative path")
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(inventory_value):
        entry = _mapping(raw, f"manifest.weight_inventory[{index}]")
        if set(entry) != {"path", "bytes", "sha256"}:
            raise RLEntryError(
                f"manifest.weight_inventory[{index}] must contain exactly path, bytes, sha256"
            )
        relative = entry["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise RLEntryError(f"manifest.weight_inventory[{index}].path must be relative")
        normalized = Path(relative)
        weight_path = (model_path / normalized).resolve()
        try:
            weight_path.relative_to(model_path)
        except ValueError as error:
            raise RLEntryError(f"manifest weight path escapes merged_model_path: {relative}") from error
        canonical = weight_path.relative_to(model_path).as_posix()
        if canonical != relative or canonical in seen:
            raise RLEntryError(f"manifest weight inventory path is duplicate or noncanonical: {relative}")
        seen.add(canonical)
        byte_count = entry["bytes"]
        if type(byte_count) is not int or byte_count < 1:
            raise RLEntryError(f"manifest.weight_inventory[{index}].bytes is invalid")
        expected_sha = _sha(entry["sha256"], f"manifest.weight_inventory[{index}].sha256")
        if not weight_path.is_file() or weight_path.is_symlink():
            raise RLEntryError(f"manifest weight file is absent or symlinked: {weight_path}")
        observed_bytes = weight_path.stat().st_size
        if observed_bytes != byte_count:
            raise RLEntryError(f"weight byte count mismatch: {weight_path}")
        observed_sha = sha256_file(weight_path) if verify_weights else expected_sha
        if observed_sha != expected_sha:
            raise RLEntryError(f"weight SHA256 mismatch: {weight_path}")
        inventory.append({"path": canonical, "bytes": byte_count, "sha256": expected_sha})
    expected_files = _weight_files(model_path)
    if expected_files != seen:
        missing = sorted(expected_files - seen)
        extra = sorted(seen - expected_files)
        raise RLEntryError(
            f"weight inventory is incomplete (missing={missing[:4]}, extra={extra[:4]})"
        )
    aggregate = weight_inventory_sha256(inventory)
    inventory_sha = _sha(manifest.get("weight_inventory_sha256"), "manifest.weight_inventory_sha256")
    if inventory_sha != aggregate:
        raise RLEntryError("manifest.weight_inventory_sha256 differs from the complete inventory")
    # The parent identity follows the conventional merged-weight artifact
    # hash for a single safetensors file.  A sharded parent has no one file,
    # so its aggregate inventory identity is the merged-weight identity.
    allowed_weight_bindings = {aggregate}
    if len(inventory) == 1:
        allowed_weight_bindings.add(inventory[0]["sha256"])
    if merged_weights_sha not in allowed_weight_bindings:
        raise RLEntryError("manifest.merged_weights_sha256 does not bind its weight inventory")
    return {
        "status": "verified",
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "kind": manifest["kind"],
        "model_path": str(model_path),
        "base_model_revision": base_revision,
        "merged_weights_sha256": merged_weights_sha,
        "config_sha256": _sha(manifest["config_sha256"], "manifest.config_sha256"),
        "generation_config_sha256": _sha(
            manifest["generation_config_sha256"], "manifest.generation_config_sha256",
        ),
        "weight_inventory": inventory,
        "weight_inventory_sha256": aggregate,
        "tokenizer": json.loads(json.dumps(tokenizer, sort_keys=True)),
        "sft_identity": json.loads(json.dumps(sft_identity, sort_keys=True)),
    }


def _validate_parent_identity(recipe: Mapping[str, Any], parent_audit: Mapping[str, Any]) -> None:
    identity = _mapping(recipe.get("identity"), "identity")
    parent = _mapping(identity.get("parent"), "identity.parent")
    if parent.get("kind") != "merged_sft":
        raise RLEntryError("identity.parent.kind must be merged_sft")
    if parent.get("manifest_sha256") != parent_audit["manifest_sha256"]:
        raise RLEntryError("identity.parent.manifest_sha256 differs from the verified parent manifest")
    if parent.get("merged_weights_sha256") != parent_audit["merged_weights_sha256"]:
        raise RLEntryError("identity.parent.merged_weights_sha256 differs from the verified inventory")
    if parent.get("base_model_revision") != parent_audit["base_model_revision"]:
        raise RLEntryError("identity.parent.base_model_revision differs from the verified parent")
    if parent.get("sft_identity") != parent_audit["sft_identity"]:
        raise RLEntryError("identity.parent.sft_identity differs from the verified parent")


def _validate_schedule(recipe: Mapping[str, Any], geometry: object) -> dict[str, Any]:
    max_steps = _positive_int(recipe.get("max_steps"), "max_steps")
    full_every = _positive_int(recipe.get("full_save_steps", DEFAULT_FULL_SAVE_STEPS), "full_save_steps")
    light_every = _positive_int(recipe.get("light_save_steps", DEFAULT_LIGHT_SAVE_STEPS), "light_save_steps")
    if full_every % light_every or full_every > max_steps or max_steps % full_every:
        raise RLEntryError("full checkpoint cadence must divide max_steps and be a multiple of light cadence")
    evaluation = recipe.get("evaluation_steps", [max_steps])
    decisions = recipe.get("decision_steps", [])
    if not isinstance(evaluation, list) or not evaluation:
        raise RLEntryError("evaluation_steps must be a nonempty list")
    if len(set(evaluation)) != len(evaluation) or any(
        type(step) is not int or not 1 <= step <= max_steps or step % full_every for step in evaluation
    ):
        raise RLEntryError("evaluation_steps must be unique positive full-checkpoint boundaries")
    if max_steps not in evaluation:
        raise RLEntryError("the terminal max_steps checkpoint requires a development readout")
    if not isinstance(decisions, list) or len(set(decisions)) != len(decisions) or any(
        type(step) is not int or step not in evaluation for step in decisions
    ):
        raise RLEntryError("decision_steps must be a subset of evaluation_steps")
    _positive_number(recipe.get("checkpoint_reserve_seconds"), "checkpoint_reserve_seconds")
    utc_deadline(recipe.get("deadline"))
    if recipe.get("evaluator_factory") is None and not recipe.get("allow_pending_evaluation", False):
        raise RLEntryError("an evaluator_factory is required unless allow_pending_evaluation is explicit")
    if recipe.get("evaluator_factory") is not None:
        factory = recipe["evaluator_factory"]
        if not isinstance(factory, str) or factory.count(":") != 1:
            raise RLEntryError("evaluator_factory must be module:function")
    for name in ("output_dir", "archive_root", "telemetry_path"):
        _absolute_path(recipe.get(name), f"recipe.{name}")
    output = assert_fresh_native_ext4_path(recipe["output_dir"], "output_dir")
    archive = assert_fresh_native_ext4_path(recipe["archive_root"], "archive_root")
    _assert_distinct_paths(output, archive)
    telemetry = Path(recipe["telemetry_path"])
    if telemetry.exists() or telemetry.is_symlink():
        raise RLEntryError(f"telemetry_path must be fresh: {telemetry}")
    if _filesystem_type(telemetry) != "ext4":
        raise RLEntryError("telemetry_path must be on native ext4")
    _positive_number(recipe.get("max_attempt_seconds"), "max_attempt_seconds")
    _positive_number(recipe.get("termination_grace_seconds"), "termination_grace_seconds")
    if float(recipe["max_attempt_seconds"]) <= (
        float(recipe["termination_grace_seconds"]) + float(recipe["checkpoint_reserve_seconds"])
    ):
        raise RLEntryError("max_attempt_seconds leaves no checkpoint reserve after termination grace")
    if not isinstance(recipe.get("generation_kwargs", {}), Mapping):
        raise RLEntryError("generation_kwargs must be an object")
    if getattr(geometry, "candidate_count", None) not in DEFAULT_CANDIDATE_COUNTS:
        raise RLEntryError("RL candidate count is not an admitted G arm")
    return {
        "max_steps": max_steps,
        "full_save_steps": full_every,
        "light_save_steps": light_every,
        "evaluation_steps": list(evaluation),
        "decision_steps": list(decisions),
        "checkpoint_reserve_seconds": float(recipe["checkpoint_reserve_seconds"]),
    }


def preflight_entry(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all CPU-side entry gates without importing a training framework."""
    if recipe.get("entry_schema_version", ENTRY_SCHEMA_VERSION) != ENTRY_SCHEMA_VERSION:
        raise RLEntryError("RL entry schema mismatch")
    if recipe.get("schema_version") != TRAIN_SCHEMA_VERSION:
        raise RLEntryError("RL recipe schema mismatch")
    identity, geometry = preflight_rl_recipe(recipe)
    parent_wrapper = _mapping(recipe.get("parent_manifest"), "parent_manifest")
    parent_audit = verify_merged_parent_manifest(parent_wrapper)
    _validate_parent_identity(recipe, parent_audit)
    resume_manifest = None
    if recipe.get("resume_from") is not None:
        resume_path = _absolute_path(recipe["resume_from"], "resume_from", file=False)
        try:
            resume_manifest = verify_checkpoint(resume_path, identity, require_full=True)
        except (OSError, TypeError, ValueError) as error:
            raise RLEntryError(f"resume_from is not a matching full checkpoint: {resume_path}") from error
    schedule = _validate_schedule(recipe, geometry)
    paths = _mapping(recipe["data"], "recipe.data")
    sidecar_path = paths.get("sidecar_path", paths.get("context_path"))
    sidecar_sha = paths.get("sidecar_artifact_sha256", paths.get("context_sha256"))
    records, manifest = load_training_records(
        Path(paths["rows_path"]), paths["rows_sha256"],
        Path(sidecar_path), sidecar_sha,
        Path(paths["selected_ids_path"]), paths["selected_ids_sha256"],
        admission_status=paths.get("admission_status", "admitted"),
    )
    if manifest.to_identity() != identity["data"]:
        raise RLEntryError("data identity changed between RL preflight and entry load")
    return {
        "status": "preflight_pass",
        "entry_schema_version": ENTRY_SCHEMA_VERSION,
        "framework_imports": {
            name: name in sys.modules for name in ("torch", "transformers", "trl", "unsloth", "datasets")
        },
        "identity": identity,
        "geometry": geometry.to_dict(),
        "schedule": schedule,
        "parent": parent_audit,
        "resume": resume_manifest,
        "data": manifest.to_identity(),
        "selected_row_ids": list(manifest.selected_ids),
        "records": len(records),
        "limits": {
            "prompt_max_tokens": PROMPT_MAX_TOKENS,
            "completion_max_tokens": COMPLETION_MAX_TOKENS,
            "managed_context_max_tokens": CONTEXT_MAX_TOKENS,
        },
        "CUDA_started": False,
        "training_started": False,
    }


def check_single_cuda_occupancy(
    *,
    max_memory_mib: int = CUDA_OCCUPANCY_LIMIT_MIB,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Require one quiet CUDA device before importing the model stack."""
    _positive_int(max_memory_mib, "max_memory_mib")
    if runner is None:
        runner = subprocess.run
    try:
        result = runner(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RLEntryError(f"cannot inspect the leased CUDA device: {error}") from error
    lines = [line.strip() for line in str(result.stdout).splitlines() if line.strip()]
    if len(lines) != 1:
        raise RLEntryError(f"single-CUDA occupancy guard expected one device, observed {len(lines)}")
    try:
        memory_mib = float(lines[0])
    except ValueError as error:
        raise RLEntryError("nvidia-smi returned a nonnumeric memory usage") from error
    if not math.isfinite(memory_mib) or memory_mib < 0 or memory_mib > max_memory_mib:
        raise RLEntryError(
            f"CUDA occupancy {memory_mib!r} MiB exceeds the quiet-lane limit {max_memory_mib} MiB"
        )
    return {"gpu_count": 1, "memory_used_mib": memory_mib, "limit_mib": max_memory_mib}


def _resource_probe(torch_module: object) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda")  # noqa: B009
    return {
        "cuda_allocated_bytes": int(cuda.memory_allocated()),
        "cuda_reserved_bytes": int(cuda.memory_reserved()),
        "cuda_attempt_peak_allocated_bytes": int(cuda.max_memory_allocated()),
        "cuda_attempt_peak_reserved_bytes": int(cuda.max_memory_reserved()),
        "peak_scope": "process lifetime including model load, optimizer, training and completed evaluations",
    }


def _resolve_stop_reason(trainer: object, terminal_step: int, max_steps: int) -> str:
    """Classify the first explicit control stop, or prove schedule completion."""
    callbacks = list(getattr(trainer, "callbacks", ()))
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks.extend(getattr(callback_handler, "callbacks", ()))
    for callback in callbacks:
        reason = getattr(callback, "stop_reason", None)
        if reason is not None:
            if not isinstance(reason, str) or not reason:
                raise RLEntryError("trainer exposed an invalid explicit stop reason")
            return reason
    if terminal_step != max_steps:
        raise RLEntryError(
            f"trainer exited at step {terminal_step} before max_steps {max_steps} without an explicit stop reason"
        )
    return "schedule_complete"


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _load_live_model(
    parent_audit: Mapping[str, Any], prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any, Any, Any]:
    """Load the verified local parent and restore tokenizer metadata once."""
    # These are the first framework imports in the live path.
    try:
        import torch
        from unsloth import FastLanguageModel
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLEntryError(f"cannot import the pinned CUDA model stack: {error}") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RLEntryError("RL entry requires exactly one leased CUDA device")
    model_path = Path(parent_audit["model_path"])
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=CONTEXT_MAX_TOKENS,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            trust_remote_code=False,
        )
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLEntryError(f"verified merged-SFT parent could not be loaded: {error}") from error
    try:
        from campaign_tokenizer_contract import (
            load_pinned_reference_tokenizer,
            restore_pinned_tokenizer_contract,
        )
        reference_tokenizer = load_pinned_reference_tokenizer(model_path)
        # This is the one post-load repair.  The post-construction audit and
        # post-generation callback only assert; train-begin uses the specific
        # alignment helper required by the pinned Transformers stack.
        tokenizer_audit = restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=prompt_rows,
        )
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLEntryError(f"loaded parent failed the pinned tokenizer contract: {error}") from error
    return model, tokenizer, reference_tokenizer, FastLanguageModel, tokenizer_audit


def _attach_rl_adapter(model: object, fast_model: object) -> tuple[object, dict[str, Any]]:
    try:
        model = fast_model.get_peft_model(
            model,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            target_modules=list(TARGET_MODULES),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=DEFAULT_SEED,
        )
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        attached = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLEntryError(f"cannot attach the pinned RL adapter: {error}") from error
    if trainable != EXPECTED_TRAINABLE_PARAMETERS or len(attached) != EXPECTED_ATTACHMENTS:
        raise RLEntryError(
            f"RL PEFT identity mismatch: {trainable} trainable parameters, {len(attached)} attachments"
        )
    return model, {"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                   "target_modules": list(TARGET_MODULES), "attachments": len(attached),
                   "trainable_parameters": trainable, "module_names": attached}


def run(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Run one admitted CUDA attempt under launcher-provided deadlines."""
    supervised_deadline = os.environ.get("SEPALITH_CAMPAIGN_SOFT_DEADLINE")
    hard_deadline = os.environ.get("SEPALITH_CAMPAIGN_HARD_DEADLINE")
    if not supervised_deadline or not hard_deadline:
        raise RLEntryError("launch through campaign_rl_launch.py under the independent supervisor")
    if not (
        utc_deadline(supervised_deadline) < utc_deadline(hard_deadline)
        <= utc_deadline(recipe["deadline"])
    ):
        raise RLEntryError("supervisor deadlines differ from the admitted recipe")
    admission = preflight_entry(recipe)
    reserve = float(admission["schedule"]["checkpoint_reserve_seconds"])
    if datetime.now(timezone.utc).timestamp() + reserve >= utc_deadline(supervised_deadline):
        raise RLEntryError("the supervised deadline leaves no checkpoint reserve")
    occupancy = check_single_cuda_occupancy()

    output = Path(recipe["output_dir"])
    archive = Path(recipe["archive_root"])
    output.mkdir(parents=True, exist_ok=False)
    archive.mkdir(parents=True, exist_ok=False)
    write_json(output / "admitted-recipe.json", recipe)
    write_json(output / "entry-preflight.json", admission)
    write_json(output / "parent-audit.json", admission["parent"])
    write_json(output / "cuda-occupancy.json", occupancy)

    paths = _mapping(recipe["data"], "recipe.data")
    sidecar_path = paths.get("sidecar_path", paths.get("context_path"))
    sidecar_sha = paths.get("sidecar_artifact_sha256", paths.get("context_sha256"))
    records, manifest = load_training_records(
        Path(paths["rows_path"]), paths["rows_sha256"],
        Path(sidecar_path), sidecar_sha,
        Path(paths["selected_ids_path"]), paths["selected_ids_sha256"],
        admission_status=paths.get("admission_status", "admitted"),
    )

    model, tokenizer, reference_tokenizer, fast_model, tokenizer_audit = _load_live_model(
        admission["parent"], [record.row for record in records],
    )
    model, adapter_audit = _attach_rl_adapter(model, fast_model)
    write_json(output / "load-audit.json", {
        "dtype": str(next(model.parameters()).dtype),
        "parent": admission["parent"],
        "tokenizer_contract": tokenizer_audit,
        "adapter": adapter_audit,
    })

    from campaign_sft import (
        assert_post_trainer_pinned_identity,
        restore_trainer_eog_alignment,
        training_configuration_guard,
    )

    def post_generation_contract(actual_model: object, actual_tokenizer: object) -> None:
        assert_post_trainer_pinned_identity(actual_model, actual_tokenizer, reference_tokenizer, ())

    reward_records_path = output / "reward-records.jsonl"

    def reward_sink(record: Mapping[str, Any]) -> None:
        _append_jsonl(reward_records_path, record)

    # The trainer's control callback uses ``supervised_deadline`` when it is
    # present.  Bind the launcher's soft cutoff into a shallow live recipe so
    # checkpoint persistence and graceful shutdown reserve the same deadline
    # enforced by GNU timeout.  Keep the admitted recipe bytes unchanged in
    # the audit above; this derived value is runtime supervision only.
    live_recipe = dict(recipe)
    live_recipe["supervised_deadline"] = supervised_deadline
    trainer = build_live_trainer(
        live_recipe,
        model=model,
        tokenizer=tokenizer,
        fast_model=fast_model,
        reference_tokenizer=reference_tokenizer,
        post_generation_restore=post_generation_contract,
        post_trainer_contract=assert_post_trainer_pinned_identity,
        resource_probe=lambda: _resource_probe(sys.modules["torch"]),
        loaded_parent_identity={
            "manifest_sha256": admission["parent"]["manifest_sha256"],
            "merged_weights_sha256": admission["parent"]["merged_weights_sha256"],
        },
        evaluation_context=lambda actual_model: training_configuration_guard(actual_model, fast_model.for_training),
    )
    reward = getattr(trainer, "_campaign_reward", None)
    if reward is None or not callable(reward):
        raise RLEntryError("live trainer did not expose its campaign reward")
    reward.event_sink = reward_sink
    trainer_tokenizer = getattr(trainer, "processing_class", tokenizer)
    write_json(output / "post-trainer-contract.json", trainer._campaign_post_trainer_contract)

    from transformers import TrainerCallback

    class PinnedTrainBeginContract(TrainerCallback):
        def on_train_begin(self, args: object, state: object, control: object, **kwargs: object) -> object:
            audit = restore_trainer_eog_alignment(model, trainer_tokenizer, reference_tokenizer)
            write_json(output / "train-begin-tokenizer-contract.json", {
                "step": getattr(state, "global_step"),  # noqa: B009
                **audit,
            })
            return control

    trainer.add_callback(PinnedTrainBeginContract())
    try:
        trainer.train(resume_from_checkpoint=recipe.get("resume_from"))
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "training_failed",
            "error": f"{type(error).__name__}: {error}",
            "identity": recipe["identity"],
        })
        raise
    terminal_step = int(trainer.state.global_step)
    try:
        terminal_contract = assert_post_trainer_pinned_identity(
            model, trainer_tokenizer, reference_tokenizer, (),
        )
        terminal_checkpoint = archive / "full" / f"checkpoint-{terminal_step}"
        checkpoint_manifest = verify_checkpoint(terminal_checkpoint, recipe["identity"], require_full=True)
    except Exception as error:
        raise RLEntryError(f"terminal full checkpoint or tokenizer contract failed: {error}") from error
    write_json(output / "terminal-tokenizer-contract.json", terminal_contract)
    stop_reason = _resolve_stop_reason(
        trainer, terminal_step, int(admission["schedule"]["max_steps"]),
    )
    result = {
        "status": stop_reason,
        "step": terminal_step,
        "checkpoint": str(terminal_checkpoint),
        "checkpoint_manifest": checkpoint_manifest,
        "identity": recipe["identity"],
        "scientific_acceptance": "A trainer exit is not a promotion signal; review development evidence.",
    }
    write_json(output / "terminal.json", result)
    return result


def preflight_file(recipe_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RLEntryError(f"cannot read RL recipe: {recipe_path}") from error
    return preflight_entry(_mapping(value, "RL recipe"))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    result_receipt = args.receipt.resolve()
    if result_receipt.exists() or result_receipt.is_symlink():
        raise RLEntryError(f"each RL entry requires a fresh result receipt: {result_receipt}")
    supervision_receipt = os.environ.get("SEPALITH_CAMPAIGN_LAUNCH_RECEIPT")
    if supervision_receipt and Path(supervision_receipt).resolve() == result_receipt:
        raise RLEntryError("entry result receipt must differ from the immutable supervision receipt")
    try:
        recipe = _mapping(json.loads(args.recipe.read_text(encoding="utf-8")), "RL recipe")
        if args.preflight:
            result = preflight_entry(recipe)
            if supervision_receipt:
                result = {**result, "supervision_receipt": str(Path(supervision_receipt).resolve())}
            write_json(result_receipt, result)
            print(json.dumps(result, sort_keys=True))
            return 0
        result = run(recipe)
        if supervision_receipt:
            result = {**result, "supervision_receipt": str(Path(supervision_receipt).resolve())}
        write_json(result_receipt, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RLEntryError, RLTrainError, RLDataError, ValueError, OSError) as error:
        result = {"status": "entry_failed", "error": f"{type(error).__name__}: {error}"}
        if supervision_receipt:
            result["supervision_receipt"] = str(Path(supervision_receipt).resolve())
        write_json(result_receipt, result)
        print(json.dumps(result, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
