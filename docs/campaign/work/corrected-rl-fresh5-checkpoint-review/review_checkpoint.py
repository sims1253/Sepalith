#!/usr/bin/env python3
"""CPU-only integrity review of the corrected RL fresh5 full checkpoint.

The review starts only after the guarded supervisor terminal marker exists.
It invokes the frozen production ``verify_checkpoint`` implementation, then
uses CPU torch/safetensors reads for optimizer and adapter finiteness.  It
does not resume, load a model, inspect final outputs, or access live state.
"""

from __future__ import annotations

import collections
import datetime as dt
import gc
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import json


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

try:
    _before_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {_before_affinity[0]})
    _after_affinity = sorted(os.sched_getaffinity(0))
except (AttributeError, OSError, IndexError):
    _before_affinity = []
    _after_affinity = []


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
CAMPAIGN = PLAN / "docs/campaign"
LEAD = CAMPAIGN / "work/lead/corrected-rl-fresh5-a"
RUN_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-corrected-theta0-fresh5-v1")
CHECKPOINT = RUN_ROOT / "archive/full/checkpoint-5"
TERMINAL_MARKER = LEAD / "supervisor-terminal.json"
RECIPE_PATH = LEAD / "recipe.json"
ROOT_VERIFICATION = LEAD / "root-verification.json"
SCHEDULE_PATH = CAMPAIGN / "work/corrected-rl-admission-audit-v1/candidate-data/source-row-draw-sequence.json"
SELECTED_IDS_PATH = CAMPAIGN / "work/corrected-rl-admission-audit-v1/candidate-data/selected-train-ids.json"
OUT = CAMPAIGN / "work/corrected-rl-fresh5-checkpoint-review"
RECEIPT = CAMPAIGN / "receipts/RL-08-corrected-rl-fresh5-checkpoint-review.json"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_SOURCE = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/"
    "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source"
    "/experiments/training"
)
CHECKPOINT_SOURCE = FROZEN_SOURCE / "campaign_checkpoint.py"

EXPECTED_RECIPE_SHA = "9042c8e1ad3f5c9580551360678a65fbe66d8f9fc980fa2c68eb94825bd7e281"
EXPECTED_IDENTITY_SHA = "ae851d72f158518f979f61b70f0a292151aed2a017c0cd71b282527ddc604bed"
EXPECTED_CHECKPOINT_SOURCE_SHA = "64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba"
EXPECTED_ROOT_VERIFICATION_SHA = "9cd2e509a30523a1bc97dbbec95615b415982a26038be8a1d82c9b0b61c006cb"

CHECKS: dict[str, dict[str, object]] = {}
FAILURES: list[dict[str, object]] = []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check(name: str, ok: bool, detail: object) -> None:
    CHECKS[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        FAILURES.append({"check": name, "detail": detail})


def finite_torch_tree(value: object, torch: object, path: str = "") -> tuple[bool, int, int, list[str]]:
    """Return finite, tensor count, floating scalar count, and bad paths."""
    if torch.is_tensor(value):
        tensors = 1
        floats = int(value.numel()) if value.is_floating_point() or value.is_complex() else 0
        if value.is_floating_point() or value.is_complex():
            finite = bool(torch.isfinite(value).all())
            return finite, tensors, floats, [] if finite else [path]
        return True, tensors, floats, []
    if isinstance(value, dict):
        finite = True
        tensors = floats = 0
        bad: list[str] = []
        for key, item in value.items():
            child = finite_torch_tree(item, torch, f"{path}/{key}")
            finite = finite and child[0]
            tensors += child[1]
            floats += child[2]
            bad.extend(child[3])
        return finite, tensors, floats, bad
    if isinstance(value, (list, tuple)):
        finite = True
        tensors = floats = 0
        bad = []
        for index, item in enumerate(value):
            child = finite_torch_tree(item, torch, f"{path}/{index}")
            finite = finite and child[0]
            tensors += child[1]
            floats += child[2]
            bad.extend(child[3])
        return finite, tensors, floats, bad
    if isinstance(value, float) and not math.isfinite(value):
        return False, 0, 0, [path]
    return True, 0, 0, []


def scalar_identity(identity: dict[str, object]) -> dict[str, object]:
    """Extract identity fields needed to compare campaign-state safely."""
    data = identity["data"]
    parent = identity["parent"]
    policy = identity["policy"]
    schedule = identity["schedule"]
    source = identity["source"]
    renderer = identity["renderer"]
    tokenizer = identity["tokenizer"]
    return {
        "data": {k: data.get(k) for k in (
            "admission_status", "context_sha256", "ordered_ids_sha256", "row_count",
            "row_identity_sha256", "rows_sha256", "selected_ids_sha256", "sidecar_artifact_sha256", "split",
        )},
        "parent": {k: parent.get(k) for k in ("base_model_revision", "kind", "manifest_sha256", "merged_weights_sha256")},
        "policy": {k: policy.get(k) for k in (
            "beta", "candidate_count", "completion_max_tokens", "context_max_tokens",
            "gradient_accumulation_steps", "loss_type", "model_load_max_seq_length",
            "per_device_train_batch_size", "prompt_max_tokens", "rollout_rows_per_update",
            "scale_rewards", "expected_trainable_parameters",
        )},
        "schedule": {k: schedule.get(k) for k in (
            "buffer_reuse", "generation_batch_size", "gradient_accumulation_steps",
            "sampler_id", "seed", "source_draw_schedule_sha256", "source_draw_sequence_sha256",
            "source_draws", "source_draws_per_update", "steps_per_generation",
        )},
        "renderer": renderer,
        "tokenizer": tokenizer,
        "source": {k: source.get(k) for k in (
            "corrected_train_manifest_sha256", "data_adapter_sha256", "pool_allocator_sha256",
            "protocol_sha256", "source_schedule_sha256", "trainer_sha256", "trl_version",
        )},
    }


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)

    # Do not perform any expensive read until the guarded terminal evidence is
    # present.  The marker is lead-owned evidence, not a process-control call.
    if not TERMINAL_MARKER.is_file():
        check("guarded_terminal_marker", False, "supervisor-terminal.json is absent; no checkpoint bytes read")
        terminal = {}
    else:
        terminal = load_json(TERMINAL_MARKER)
        check(
            "guarded_terminal_marker",
            terminal.get("guard_exit_code") == 0 and float(terminal.get("seconds", 0)) > 0,
            {k: terminal.get(k) for k in ("at", "seconds", "guard_exit_code", "scientific_acceptance")},
        )
    if FAILURES:
        return write_outputs(started, {}, {}, {}, {}, {})

    # Hash and load only the final recipe identity.  verify_checkpoint itself
    # performs the production manifest inventory and full file hashing.
    recipe_sha = sha256_file(RECIPE_PATH)
    recipe = load_json(RECIPE_PATH)
    identity = recipe["identity"]
    root_verification_sha = sha256_file(ROOT_VERIFICATION)
    root_verification = load_json(ROOT_VERIFICATION)
    check(
        "final_recipe_and_root_identity",
        recipe_sha == EXPECTED_RECIPE_SHA
        and root_verification_sha == EXPECTED_ROOT_VERIFICATION_SHA
        and root_verification.get("status") == "pass"
        and root_verification.get("recipe_sha256") == EXPECTED_RECIPE_SHA
        and root_verification.get("identity_sha256") == EXPECTED_IDENTITY_SHA,
        {
            "recipe_sha256": recipe_sha,
            "identity_sha256": root_verification.get("identity_sha256"),
            "root_verification_sha256": root_verification_sha,
            "root_status": root_verification.get("status"),
        },
    )

    verifier_sha = sha256_file(CHECKPOINT_SOURCE)
    check(
        "frozen_production_verifier_source",
        verifier_sha == EXPECTED_CHECKPOINT_SOURCE_SHA,
        {"path": str(CHECKPOINT_SOURCE), "sha256": verifier_sha, "expected": EXPECTED_CHECKPOINT_SOURCE_SHA},
    )

    verifier_error: str | None = None
    manifest: dict[str, object] = {}
    verify_started = time.monotonic()
    try:
        sys.path.insert(0, str(FROZEN_SOURCE))
        from campaign_checkpoint import verify_checkpoint  # frozen production verifier; stdlib at import

        manifest = verify_checkpoint(CHECKPOINT, identity, require_full=True)
    except Exception as exc:  # report exact verifier failure without hiding it
        verifier_error = f"{type(exc).__name__}: {exc}"
    verify_elapsed = time.monotonic() - verify_started
    check(
        "frozen_verify_checkpoint_full",
        verifier_error is None
        and manifest.get("schema_version") == 1
        and manifest.get("full") is True
        and manifest.get("step") == 5
        and manifest.get("identity") == identity,
        {
            "checkpoint": str(CHECKPOINT),
            "elapsed_seconds": verify_elapsed,
            "error": verifier_error,
            "schema_version": manifest.get("schema_version"),
            "full": manifest.get("full"),
            "step": manifest.get("step"),
            "manifest_file_count": len(manifest.get("files", {})),
            "identity_exact_match": manifest.get("identity") == identity,
        },
    )

    manifest_files = manifest.get("files", {})
    checkpoint_file_bytes = sum(int(info["bytes"]) for info in manifest_files.values()) if isinstance(manifest_files, dict) else 0
    checkpoint_manifest_path = CHECKPOINT / "campaign-manifest.json"
    checkpoint_manifest_sha = sha256_file(checkpoint_manifest_path)
    manifest_summary = {
        "path": str(checkpoint_manifest_path),
        "bytes": checkpoint_manifest_path.stat().st_size,
        "sha256": checkpoint_manifest_sha,
        "schema_version": manifest.get("schema_version"),
        "step": manifest.get("step"),
        "full": manifest.get("full"),
        "file_count": len(manifest_files),
        "file_bytes_sum_excluding_manifest": checkpoint_file_bytes,
        "files": manifest_files,
    }

    # Keep a compact identity snapshot, then release the 70MB+ recipe and
    # manifest objects before loading campaign-state and optimizer tensors.
    expected_identity_scalars = scalar_identity(identity)
    del manifest
    del recipe
    del identity
    gc.collect()

    # campaign-state contains the full identity list.  Parse it only after the
    # production verifier has passed, retain sampler/scalar fields, then drop
    # the large identity object before torch reads.
    state_path = CHECKPOINT / "campaign-state.json"
    state = load_json(state_path)
    state_sampler = state.get("sampler", {})
    state_identity = state.get("identity", {})
    state_identity_scalars = scalar_identity(state_identity)
    state_identity_match = state_identity_scalars == expected_identity_scalars
    state_step = state.get("step")
    state_full = state.get("full")
    resume_validation = state.get("resume_validation")
    state_sampler_compact = {k: v for k, v in state_sampler.items() if k != "data_identity"}
    state_data_identity = state_sampler.get("data_identity", {})
    state_sampler_data_scalars = {
        k: state_data_identity.get(k) for k in (
            "admission_status", "context_sha256", "ordered_ids_sha256", "row_count",
            "row_identity_sha256", "rows_sha256", "selected_ids_sha256", "sidecar_artifact_sha256", "split",
        )
    }
    del state
    del state_identity
    del state_sampler
    del state_data_identity
    gc.collect()

    check(
        "campaign_state_identity_and_step",
        state_full is True
        and state_step == 5
        and state_identity_match
        and resume_validation == "Requires matched interruption/resume receipt; file presence alone is insufficient",
        {
            "campaign_state_sha256": sha256_file(state_path),
            "campaign_state_bytes": state_path.stat().st_size,
            "full": state_full,
            "step": state_step,
            "identity_scalar_match": state_identity_match,
            "resume_validation": resume_validation,
        },
    )

    # Bind the source schedule and selected-ID order used to derive the next
    # sampler group.  These are source IDs, not model/final content.
    sequence = load_json(SCHEDULE_PATH)
    selected_doc = load_json(SELECTED_IDS_PATH)
    sequence_ids = sequence["row_ids"]
    selected_ids = selected_doc["row_ids"]
    sequence_file_sha = sha256_file(SCHEDULE_PATH)
    selected_file_sha = sha256_file(SELECTED_IDS_PATH)
    sequence_hash = compact_hash(sequence_ids)
    ordered_ids_hash = compact_hash(selected_ids)
    next_source_draw_ids = sequence_ids[40:48]
    selected_index = {row_id: index for index, row_id in enumerate(selected_ids)}
    next_selected_index = selected_index.get(next_source_draw_ids[0]) if next_source_draw_ids else None
    stream_next_ids = [
        row_id
        for _ in range(int(state_sampler_compact.get("repeat_count", 0)))
        for row_id in next_source_draw_ids
        for _ in range(int(state_sampler_compact.get("candidate_count", 0)))
    ]
    expected_sampler = {
        "sampler_id": "campaign-repeat-manifest-order-v1",
        "seed": 3407,
        "shuffle": False,
        "num_samples": 8246,
        "candidate_count": 4,
        "prompt_groups_per_batch": 8,
        "repeat_count": 8,
        "consumed_rows": 1280,
        "current_index": 1280,
        "consumed_prompt_copies": 320,
        "epoch": 0,
        "source_draws_bound": True,
        "source_draws": 24000,
        "source_draw_cursor": 40,
        "buffer_reuse": 8,
        "generation_rows_per_update": 32,
        "sampler_rows_per_update": 256,
        "selected_id_index": next_selected_index,
        "geometry": {
            "generation_batch_size": 32,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 4,
            "steps_per_generation": 8,
        },
    }
    actual_sampler = {key: state_sampler_compact.get(key) for key in expected_sampler if key != "geometry"}
    actual_sampler["geometry"] = state_sampler_compact.get("geometry")
    sampler_mismatches = {
        key: {"actual": actual_sampler.get(key), "expected": value}
        for key, value in expected_sampler.items()
        if actual_sampler.get(key) != value
    }
    # The source-draw schedule file hash is the hash of the JSONL/JSON file,
    # while sequence_sha256 is the compact row-id-list hash.
    schedule_hash_match = (
        sequence.get("source_draws") == 24000
        and len(sequence_ids) == 24000
        and sequence_hash == sequence.get("sequence_sha256")
        and ordered_ids_hash == sequence.get("ordered_ids_sha256")
        and selected_file_sha == sequence.get("selected_ids_sha256")
        and state_sampler_compact.get("source_draw_sequence_sha256") == sequence.get("sequence_sha256")
        and state_sampler_compact.get("source_draw_schedule_sha256") == sequence_file_sha
        and state_sampler_compact.get("source_draw_cursor") == 40
    )
    next_draws_ok = (
        len(next_source_draw_ids) == 8
        and next_selected_index is not None
        and len(stream_next_ids) == 8 * 8 * 4
        and stream_next_ids[:4] == [next_source_draw_ids[0]] * 4
        and state_sampler_compact.get("consumed_rows", 0) % 256 == 0
    )
    check(
        "sampler_cursor_and_next_draws",
        not sampler_mismatches and schedule_hash_match and next_draws_ok,
        {
            "mismatches": sampler_mismatches,
            "sequence_file_sha256": sequence_file_sha,
            "sequence_sha256": sequence_hash,
            "ordered_ids_sha256": ordered_ids_hash,
            "selected_ids_file_sha256": selected_file_sha,
            "next_source_draw_ids": next_source_draw_ids,
            "next_selected_id_index": next_selected_index,
            "next_stream_rows": len(stream_next_ids),
            "next_stream_first_group": stream_next_ids[:4],
            "source_cursor": state_sampler_compact.get("source_draw_cursor"),
            "consumed_rows": state_sampler_compact.get("consumed_rows"),
            "source_schedule_hash_binding": state_sampler_compact.get("source_draw_schedule_sha256"),
        },
    )
    check(
        "checkpoint_state_data_identity",
        state_sampler_data_scalars == expected_identity_scalars["data"],
        {"actual": state_sampler_data_scalars, "expected": expected_identity_scalars["data"]},
    )

    # Small JSON state files are checked before tensor reads.
    trainer_state_path = CHECKPOINT / "trainer_state.json"
    trainer_state = load_json(trainer_state_path)
    adapter_config_path = CHECKPOINT / "adapter_config.json"
    adapter_config = load_json(adapter_config_path)
    trainer_ok = (
        trainer_state.get("global_step") == 5
        and trainer_state.get("max_steps") == 5
        and trainer_state.get("save_steps") == 5
        and len(trainer_state.get("log_history", [])) == 5
        and trainer_state.get("is_world_process_zero") is True
    )
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    adapter_ok = (
        adapter_config.get("peft_type") == "LORA"
        and adapter_config.get("r") == 16
        and adapter_config.get("lora_alpha") == 16
        and adapter_config.get("lora_dropout") == 0
        and adapter_config.get("bias") == "none"
        and set(adapter_config.get("target_modules", [])) == expected_targets
        and adapter_config.get("task_type") == "CAUSAL_LM"
    )
    check(
        "trainer_and_adapter_config_step5",
        trainer_ok and adapter_ok,
        {
            "trainer": {k: trainer_state.get(k) for k in ("global_step", "max_steps", "save_steps", "num_input_tokens_seen", "is_world_process_zero")},
            "log_history_entries": len(trainer_state.get("log_history", [])),
            "adapter": {k: adapter_config.get(k) for k in ("peft_type", "r", "lora_alpha", "lora_dropout", "bias", "target_modules", "task_type")},
        },
    )

    # CPU-only tensor checks are intentionally after the terminal marker and
    # after the frozen manifest verifier.  No model or CUDA object is loaded.
    tensor_error: str | None = None
    optimizer_summary: dict[str, object] = {}
    adapter_summary: dict[str, object] = {}
    rng_summary: dict[str, object] = {}
    scheduler_summary: dict[str, object] = {}
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        optimizer = torch.load(CHECKPOINT / "optimizer.pt", map_location="cpu", weights_only=False)
        finite, tensor_count, floating_values, bad_paths = finite_torch_tree(optimizer, torch, "optimizer")
        states = optimizer.get("state", {})
        state_steps = [value.get("step") for value in states.values() if isinstance(value, dict) and "step" in value]
        state_steps_ok = len(state_steps) == 588 and all(step == 5 for step in state_steps)
        groups = optimizer.get("param_groups", [])
        groups_ok = (
            len(groups) == 2
            and len(groups[0].get("params", [])) == 588
            and len(groups[1].get("params", [])) == 0
            and groups[0].get("lr") == 2e-5
            and groups[1].get("lr") == 2e-5
        )
        optimizer_summary = {
            "finite": finite,
            "tensor_count": tensor_count,
            "floating_values": floating_values,
            "nonfinite_paths": bad_paths[:10],
            "state_entries": len(states),
            "state_step_values": dict(collections.Counter(str(step) for step in state_steps)),
            "state_steps_all_5": state_steps_ok,
            "param_groups": len(groups),
            "param_group_param_counts": [len(group.get("params", [])) for group in groups],
            "param_group_learning_rates": [group.get("lr") for group in groups],
            "param_groups_geometry_ok": groups_ok,
        }
        check("optimizer_cpu_finite_step5", finite and state_steps_ok and groups_ok, optimizer_summary)
        del optimizer
        gc.collect()

        from safetensors import safe_open

        adapter_keys: list[str] = []
        adapter_bad: list[str] = []
        adapter_values = 0
        adapter_dtypes: collections.Counter[str] = collections.Counter()
        with safe_open(str(CHECKPOINT / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
            adapter_keys = list(handle.keys())
            for key in adapter_keys:
                tensor = handle.get_tensor(key)
                adapter_values += int(tensor.numel())
                adapter_dtypes[str(tensor.dtype)] += 1
                if tensor.is_floating_point() or tensor.is_complex():
                    if not bool(torch.isfinite(tensor).all()):
                        adapter_bad.append(key)
        adapter_summary = {
            "tensor_count": len(adapter_keys),
            "value_count": adapter_values,
            "dtypes": dict(adapter_dtypes),
            "finite": not adapter_bad,
            "nonfinite_keys": adapter_bad[:10],
            "expected_value_count": expected_identity_scalars["policy"]["expected_trainable_parameters"],
        }
        check(
            "adapter_cpu_finite_and_geometry",
            len(adapter_keys) == 588
            and adapter_values == expected_identity_scalars["policy"]["expected_trainable_parameters"]
            and not adapter_bad
            and adapter_dtypes == collections.Counter({"torch.float32": 588}),
            adapter_summary,
        )

        rng = torch.load(CHECKPOINT / "rng_state.pth", map_location="cpu", weights_only=False)
        rng_keys = set(rng) if isinstance(rng, dict) else set()
        rng_summary = {
            "keys": sorted(rng_keys),
            "python_type": type(rng.get("python")).__name__ if isinstance(rng, dict) else None,
            "numpy_type": type(rng.get("numpy")).__name__ if isinstance(rng, dict) else None,
            "cpu": {"type": type(rng.get("cpu")).__name__, "dtype": str(rng["cpu"].dtype), "shape": list(rng["cpu"].shape)} if isinstance(rng.get("cpu"), torch.Tensor) else None,
            "cuda": {"type": type(rng.get("cuda")).__name__, "dtype": str(rng["cuda"].dtype), "shape": list(rng["cuda"].shape)} if isinstance(rng.get("cuda"), torch.Tensor) else None,
        }
        rng_ok = (
            rng_keys == {"python", "numpy", "cpu", "cuda"}
            and isinstance(rng.get("python"), tuple)
            and isinstance(rng.get("numpy"), tuple)
            and isinstance(rng.get("cpu"), torch.Tensor)
            and rng["cpu"].dtype == torch.uint8
            and tuple(rng["cpu"].shape) == (5056,)
            and isinstance(rng.get("cuda"), torch.Tensor)
            and rng["cuda"].dtype == torch.uint8
            and tuple(rng["cuda"].shape) == (16,)
        )
        check("rng_state_cpu_contract", rng_ok, rng_summary)
        del rng
        gc.collect()

        scheduler = torch.load(CHECKPOINT / "scheduler.pt", map_location="cpu", weights_only=False)
        scheduler_summary = {
            "base_lrs": scheduler.get("base_lrs"),
            "last_epoch": scheduler.get("last_epoch"),
            "step_count": scheduler.get("_step_count"),
            "is_initial": scheduler.get("_is_initial"),
            "last_lr": scheduler.get("_last_lr"),
        }
        scheduler_ok = (
            scheduler.get("base_lrs") == [2e-5, 2e-5]
            and scheduler.get("last_epoch") == 5
            and scheduler.get("_step_count") == 6
            and scheduler.get("_is_initial") is False
            and scheduler.get("_last_lr") == [2e-5, 2e-5]
        )
        check("scheduler_cpu_contract_step5", scheduler_ok, scheduler_summary)
        del scheduler
        gc.collect()
        check(
            "cpu_torch_runtime",
            torch.get_num_threads() == 1 and not torch.cuda.is_available(),
            {"torch_version": torch.__version__, "num_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()},
        )
    except Exception as exc:
        tensor_error = f"{type(exc).__name__}: {exc}"
        check("cpu_tensor_checks_completed", False, tensor_error)

    # The review does not load training_args.bin or base model weights.  The
    # adapter checkpoint tensors were read on CPU because their finiteness and
    # parameter geometry are part of this integrity review.  Resume faithfulness
    # remains a separate interruption test.
    checkpoint_identity = {
        "recipe_sha256": recipe_sha,
        "identity_sha256_reported": EXPECTED_IDENTITY_SHA,
        "checkpoint_manifest": manifest_summary,
        "archive_checkpoint": str(CHECKPOINT),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "root_verification_sha256": root_verification_sha,
        "root_verification_full_preflight_sha256": root_verification.get("full_preflight_sha256"),
        "tensor_error": tensor_error,
    }

    review = {
        "schema": "sepalith.rl08.corrected-rl-fresh5-checkpoint-review.v1",
        "task": "RL-08",
        "status": "pass_checkpoint_integrity; no_continuation_judgment" if not FAILURES else "checkpoint_review_failed",
        "at": started.isoformat(),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "cpu_only": True,
            "one_cpu_before": _before_affinity,
            "one_cpu_after": _after_affinity,
            "one_cpu_verified": len(_after_affinity) == 1,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "native_launch": False,
            "resume_or_model_load": False,
            "framework_training_run": False,
            "SSH_cloud_network": False,
            "live_state_lease_process_control": False,
            "final_outputs_or_sealed_evaluation_read": False,
        },
        "terminal_evidence": {
            "path": str(TERMINAL_MARKER),
            "sha256": sha256_file(TERMINAL_MARKER),
            "at": terminal.get("at"),
            "seconds": terminal.get("seconds"),
            "guard_exit_code": terminal.get("guard_exit_code"),
            "scientific_acceptance": terminal.get("scientific_acceptance"),
        },
        "checkpoint_identity": checkpoint_identity,
        "campaign_state": {
            "step": state_step,
            "full": state_full,
            "identity_scalar_match": state_identity_match,
            "data_identity_match": state_sampler_data_scalars == expected_identity_scalars["data"],
            "sampler": state_sampler_compact,
            "resume_validation": resume_validation,
        },
        "sampler_continuation": {
            "source_draw_cursor": state_sampler_compact.get("source_draw_cursor"),
            "consumed_rows": state_sampler_compact.get("consumed_rows"),
            "consumed_prompt_copies": state_sampler_compact.get("consumed_prompt_copies"),
            "current_index": state_sampler_compact.get("current_index"),
            "selected_id_index": next_selected_index,
            "next_source_draw_ids": next_source_draw_ids,
            "next_stream_rows": len(stream_next_ids),
            "next_stream_first_candidate_group": stream_next_ids[:4],
            "sequence_file_sha256": sequence_file_sha,
            "sequence_sha256": sequence_hash,
            "ordered_ids_sha256": ordered_ids_hash,
            "selected_ids_file_sha256": selected_file_sha,
            "candidate_count": state_sampler_compact.get("candidate_count"),
            "repeat_count": state_sampler_compact.get("repeat_count"),
            "buffer_reuse": state_sampler_compact.get("buffer_reuse"),
        },
        "adapter": adapter_summary,
        "optimizer": optimizer_summary,
        "rng": rng_summary,
        "scheduler": scheduler_summary,
        "trainer_state": {
            "global_step": trainer_state.get("global_step"),
            "max_steps": trainer_state.get("max_steps"),
            "save_steps": trainer_state.get("save_steps"),
            "log_history_entries": len(trainer_state.get("log_history", [])),
            "num_input_tokens_seen": trainer_state.get("num_input_tokens_seen"),
        },
        "checks": CHECKS,
        "limits": [
            "This review did not load base model weights, training_args.bin, final outputs, or sealed evaluation; adapter checkpoint tensors were read on CPU for finiteness.",
            "Frozen verify_checkpoint proves sealed byte identity and full-state presence. It does not prove faithful interruption/resume behavior.",
            "Adapter and optimizer finiteness is a checkpoint integrity result, not a quality or continuation recommendation.",
            "DEV/reward/quality analysis and any continuation decision remain with root/editor_acceptance.",
        ],
    }
    return write_outputs(started, review, checkpoint_identity, manifest_summary, adapter_summary, optimizer_summary)


def write_outputs(
    started: dt.datetime,
    review: dict[str, object],
    checkpoint_identity: dict[str, object],
    manifest_summary: dict[str, object],
    adapter_summary: dict[str, object],
    optimizer_summary: dict[str, object],
) -> int:
    review_path = OUT / "review.json"
    input_manifest_path = OUT / "input-manifest.json"
    if not review:
        review = {
            "schema": "sepalith.rl08.corrected-rl-fresh5-checkpoint-review.v1",
            "task": "RL-08",
            "status": "blocked_before_terminal_marker",
            "at": started.isoformat(),
            "checks": CHECKS,
            "limits": ["No checkpoint bytes were read because supervisor-terminal.json was absent."],
        }
    inputs = {
        "schema": "sepalith.rl08.corrected-rl-fresh5-checkpoint-review-inputs.v1",
        "task": "RL-08",
        "terminal_marker": str(TERMINAL_MARKER),
        "recipe": str(RECIPE_PATH),
        "root_verification": str(ROOT_VERIFICATION),
        "checkpoint": str(CHECKPOINT),
        "frozen_verifier": str(CHECKPOINT_SOURCE),
        "schedule": str(SCHEDULE_PATH),
        "selected_ids": str(SELECTED_IDS_PATH),
        "reads_allowed_after_terminal": [
            "full checkpoint manifest and sealed files",
            "optimizer.pt, scheduler.pt, rng_state.pth on CPU",
            "adapter_model.safetensors on CPU",
        ],
        "not_read": [
            "model/base weights",
            "training_args.bin",
            "final outputs or sealed evaluation",
            "live process/state/lease files",
        ],
        "produced_summaries": {
            "checkpoint_identity": checkpoint_identity,
            "manifest": manifest_summary,
            "adapter": adapter_summary,
            "optimizer": optimizer_summary,
        },
    }
    input_manifest_path.write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_sha = sha256_file(review_path)
    input_sha = sha256_file(input_manifest_path)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    receipt = {
        "schema": "sepalith.campaign.receipt.v1",
        "task": "RL-08",
        "status": review.get("status"),
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owner": "stress_fixture",
        "review": str(review_path),
        "review_sha256": review_sha,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": input_sha,
        "review_script": str(script_path),
        "review_script_sha256": script_sha,
        "checkpoint": str(CHECKPOINT),
        "recipe_sha256": EXPECTED_RECIPE_SHA,
        "identity_sha256": EXPECTED_IDENTITY_SHA,
        "checkpoint_manifest_sha256": manifest_summary.get("sha256"),
        "frozen_verifier_sha256": EXPECTED_CHECKPOINT_SOURCE_SHA,
        "full_manifest_file_count": manifest_summary.get("file_count"),
        "full_manifest_files": manifest_summary.get("files"),
        "terminal_marker_sha256": sha256_file(TERMINAL_MARKER) if TERMINAL_MARKER.is_file() else None,
        "cpu_only": True,
        "one_cpu_verified": len(_after_affinity) == 1,
        "native_launch": False,
        "base_model_weights_loaded": False,
        "adapter_checkpoint_tensors_read": bool(adapter_summary),
        "final_outputs_read": False,
        "launch_blocker": "; ".join(str(item["check"]) for item in FAILURES) if FAILURES else None,
        "result": {
            "frozen_full_checkpoint_verification": CHECKS.get("frozen_verify_checkpoint_full", {}).get("ok"),
            "campaign_state_step5": CHECKS.get("campaign_state_identity_and_step", {}).get("ok"),
            "sampler_cursor_next_draws": CHECKS.get("sampler_cursor_and_next_draws", {}).get("ok"),
            "adapter_finite": CHECKS.get("adapter_cpu_finite_and_geometry", {}).get("ok"),
            "optimizer_finite_step5": CHECKS.get("optimizer_cpu_finite_step5", {}).get("ok"),
            "rng_scheduler_contract": CHECKS.get("rng_state_cpu_contract", {}).get("ok") and CHECKS.get("scheduler_cpu_contract_step5", {}).get("ok"),
            "continuation_judgment": "not made from checkpoint alone",
        },
        "limits": review.get("limits", []),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "status": receipt["status"],
        "failures": FAILURES,
        "review": str(review_path),
        "review_sha256": review_sha,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": input_sha,
        "receipt": str(RECEIPT),
        "receipt_sha256": sha256_file(RECEIPT),
        "review_script_sha256": script_sha,
        "manifest_files": manifest_summary.get("file_count"),
        "checkpoint_manifest_sha256": manifest_summary.get("sha256"),
    }
    print(json.dumps(result, indent=2))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
