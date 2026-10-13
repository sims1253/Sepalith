#!/usr/bin/env python3
"""CPU-only independent review for the P2 RL-03 smoke.

The P2 rollout deliberately makes one ``generate`` call cover two logical
groups.  This review therefore checks four calls (8 groups, G=4, 32 rows) at
each optimizer update and joins the generation and reward streams by their
ordered row position plus output-token hash.  No framework, model, CUDA, or
project training module is imported.

An unfinished live process is reported as ``pending``.  The optional split
comparison is only a resume proof when continuous, first-segment, and
resumed-segment reviews are complete.  It compares exact adapter tensor,
optimizer, scheduler, RNG, sampler/source, and ordered output bytes/rows.
Run-specific paths, timing, and trainer metadata are reported under an
explicit allowlist.  Byte hashes keep the verifier CPU-only; a trusted
``torch.load(weights_only=True)`` inspection is not needed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile
from typing import Any, Iterable, Mapping


SOURCE_SHA256 = "15fb2080278022b076223c2a823413fc5cb9c014d50dab42dc9df6d57d3f2f41"
RECIPE_SHA256 = "e4cc42dd72fec1ea9c41d831c896171a3317472416426f0253e4e3a7d5a91ba3"
SPLIT_FIRST_RECIPE_SHA256 = "1a65dd4c3652eb22c526876eb3206a1f942602dd86ec2ff04c0bbe992c874de7"
SPLIT_RESUME_RECIPE_SHA256 = "f6c33a3eac6f0fcd86248d61e8d8b1000c9c8e23cc90e5a41800420c63af5879"
IDENTITY_SHA256 = "9692678934c496caa48bfd662076489698c587023eee82df615329595bb600a3"
SCHEDULE_SHA256 = "892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48"
SEQUENCE_SHA256 = "d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2"
SELECTED_IDS_SHA256 = "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d"
ORDERED_IDS_SHA256 = "7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d"
GROUPS_PER_CALL = 2
CANDIDATES = 4
GROUPS_PER_UPDATE = 8
ROWS_PER_UPDATE = 32
UPDATES = 2
DEV_ROWS = 14
DEV_CAP = 512
LONG_DEV_ID = "dat07-existing-719cd49683667d0fb86fb2fa"
LONG_DEV_PROMPT = 2619
GENERATION_SCHEMA = "sepalith.rl.generation-record.v1"
FULL_STATE_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "adapter_model.safetensors",
    "adapter_config.json",
    "campaign-state.json",
    "campaign-manifest.json",
)
# Exact state/output bytes used for restart equality.  Other checkpoint files
# carry paths, timing, or trainer metadata that are audited under the explicit
# allowlist below rather than treated as restart failures.
EXACT_RESUME_FILES = (
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
METADATA_ALLOWLIST = (
    "README.md",
    "adapter_config.json",
    "campaign-state.json",
    "campaign-manifest.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
)


class ReviewError(Exception):
    """An artifact is malformed or contradicts the P2 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    # This is used only for known run artifacts.  The verifier never searches
    # a drive and never copies the large identity fields into its receipt.
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def short(value: Any, limit: int = 220) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def read_jsonl(path: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return [], {"status": "pending", "path": str(path), "reason": "artifact not present"}
    rows: list[Mapping[str, Any]] = []
    failures: list[str] = []
    trailing_partial = False
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    failures.append(f"empty line {number}")
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    if not line.endswith("\n"):
                        trailing_partial = True
                        break
                    failures.append(f"invalid JSON line {number}: {error}")
                    continue
                if not isinstance(value, Mapping):
                    failures.append(f"line {number} is not an object")
                else:
                    rows.append(value)
    except OSError as error:
        return [], {"status": "fail", "path": str(path), "reason": str(error)}
    if failures:
        return rows, {"status": "fail", "path": str(path), "rows": len(rows), "failures": failures}
    return rows, {
        "status": "pending_partial" if trailing_partial else "present",
        "path": str(path),
        "rows": len(rows),
        **({"reason": "last JSONL line is still being written"} if trailing_partial else {}),
    }


def supervision_path(root: Path) -> Path:
    return root.parent / f"{root.name}-supervision.json"


def result_path_from_supervision(root: Path, supervision: Mapping[str, Any] | None) -> Path | None:
    value = supervision.get("entry_result_receipt") if isinstance(supervision, Mapping) else None
    return Path(value).resolve() if isinstance(value, str) else None


def review_supervision(root: Path, expected_recipe: str = RECIPE_SHA256) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    path = supervision_path(root)
    if not path.is_file():
        return {"status": "pending", "path": str(path), "reason": "immutable supervision receipt absent"}, None
    try:
        value = read_json(path)
    except Exception as error:
        return {"status": "fail", "path": str(path), "reason": f"invalid JSON: {error}"}, None
    if not isinstance(value, Mapping):
        return {"status": "fail", "path": str(path), "reason": "supervision receipt is not an object"}, None
    failures: list[str] = []
    if value.get("recipe_sha256") != expected_recipe:
        failures.append(f"recipe SHA is {short(value.get('recipe_sha256'))!r}")
    argv = value.get("argv")
    if not isinstance(argv, list) or not any(SOURCE_SHA256 in str(item) for item in argv):
        failures.append("argv does not bind frozen P2 source snapshot")
    result = result_path_from_supervision(root, value)
    if result is None or result == path.resolve():
        failures.append("entry result receipt is absent or aliases immutable supervision receipt")
    result_summary: dict[str, Any] | None = None
    if result is not None and result.is_file():
        try:
            result_value = read_json(result)
            if not isinstance(result_value, Mapping):
                failures.append("entry result receipt is not an object")
            else:
                result_summary = {key: result_value.get(key) for key in (
                    "status", "step", "checkpoint", "error", "supervision_receipt"
                ) if key in result_value}
                if result_value.get("supervision_receipt"):
                    bound = Path(str(result_value["supervision_receipt"])).resolve()
                    if bound != path.resolve():
                        failures.append("entry result does not point to immutable supervision receipt")
        except Exception as error:
            failures.append(f"entry result unreadable: {error}")
    status = "pass" if not failures and result_summary is not None else ("pending" if not failures else "fail")
    review = {
        "status": status,
        "path": str(path),
        "immutable": True,
        "supervisor_status": value.get("status"),
        "recipe_sha256": value.get("recipe_sha256"),
        "recipe_sha256_expected": expected_recipe,
        "entry_result_receipt": str(result) if result else None,
        "entry_result": result_summary,
        "failures": failures,
    }
    # The supervisor's preflight embeds all 8,440 row identities.  Callers
    # need only the compact review; retaining that object would needlessly
    # increase peak memory when checkpoint manifests are inspected.
    return review, None


def read_telemetry(path: Path) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return None, [], {"status": "pending", "path": str(path), "reason": "telemetry absent"}
    identity: Mapping[str, Any] | None = None
    events: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    failures: list[str] = []
    lines = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                lines = number
                if not line.strip():
                    failures.append(f"empty line {number}")
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    if not line.endswith("\n"):
                        return identity, events, {
                            "status": "pending_partial", "path": str(path), "lines": lines,
                            "event_counts": dict(counts), "reason": "last telemetry line is incomplete",
                        }
                    failures.append(f"invalid JSON line {number}: {error}")
                    continue
                if not isinstance(value, Mapping):
                    failures.append(f"line {number} is not an object")
                    continue
                event = str(value.get("event"))
                counts[event] += 1
                if identity is None and isinstance(value.get("identity"), Mapping):
                    identity = value["identity"]
                if event in {"train_begin", "optimizer_step", "trainer_metrics", "train_end"}:
                    entry = {key: value.get(key) for key in (
                        "event", "step", "global_step", "stop_reason", "remaining_seconds", "seconds"
                    ) if key in value}
                    if event == "trainer_metrics":
                        metrics = value.get("metrics")
                        if isinstance(metrics, Mapping):
                            entry["metrics"] = {
                                key: metrics.get(key) for key in (
                                    "reward", "reward_std", "frac_reward_zero_std", "grad_norm",
                                    "completion_length", "completions/clipped_ratio",
                                ) if key in metrics
                            }
                    events.append(entry)
    except OSError as error:
        return identity, events, {"status": "fail", "path": str(path), "reason": str(error)}
    return identity, events, {
        "status": "fail" if failures else "present", "path": str(path), "lines": lines,
        "event_counts": dict(counts), **({"failures": failures} if failures else {}),
    }


def identity_review(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if identity is None:
        return {"status": "pending", "reason": "train_begin identity not available"}
    failures: list[str] = []
    try:
        identity_hash = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as error:
        identity_hash = None
        failures.append(f"identity canonicalization failed: {error}")
    if identity_hash != IDENTITY_SHA256:
        failures.append(f"identity SHA is {identity_hash!r}, expected {IDENTITY_SHA256!r}")
    policy = identity.get("policy") if isinstance(identity.get("policy"), Mapping) else {}
    schedule = identity.get("schedule") if isinstance(identity.get("schedule"), Mapping) else {}
    data = identity.get("data") if isinstance(identity.get("data"), Mapping) else {}
    renderer = identity.get("renderer") if isinstance(identity.get("renderer"), Mapping) else {}
    parent = identity.get("parent") if isinstance(identity.get("parent"), Mapping) else {}
    expected = {
        "data.row_count": (data.get("row_count"), 8440),
        "data.ordered_ids_sha256": (data.get("ordered_ids_sha256"), ORDERED_IDS_SHA256),
        "policy.candidate_count": (policy.get("candidate_count"), CANDIDATES),
        "policy.generation_groups_per_call": (policy.get("generation_groups_per_call"), GROUPS_PER_CALL),
        "policy.gradient_accumulation_steps": (policy.get("gradient_accumulation_steps"), 4),
        "policy.per_device_train_batch_size": (policy.get("per_device_train_batch_size"), 8),
        "policy.prompt_max_tokens": (policy.get("prompt_max_tokens"), 2048),
        "policy.completion_max_tokens": (policy.get("completion_max_tokens"), 192),
        "policy.context_max_tokens": (policy.get("context_max_tokens"), 2240),
        "policy.model_load_max_seq_length": (policy.get("model_load_max_seq_length"), 4096),
        "policy.cuda_memory_fraction": (policy.get("cuda_memory_fraction"), 0.75),
        "schedule.source_draws_per_update": (schedule.get("source_draws_per_update"), 8),
        "schedule.generation_batch_size": (schedule.get("generation_batch_size"), ROWS_PER_UPDATE),
        "schedule.buffer_reuse": (schedule.get("buffer_reuse"), 4),
        "schedule.steps_per_generation": (schedule.get("steps_per_generation"), 4),
        "schedule.source_draw_schedule_sha256": (schedule.get("source_draw_schedule_sha256"), SCHEDULE_SHA256),
        "schedule.source_draw_sequence_sha256": (schedule.get("source_draw_sequence_sha256"), SEQUENCE_SHA256),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            failures.append(f"{name}: expected {wanted!r}, got {short(actual)!r}")
    source = identity.get("source")
    frozen_root = source.get("frozen_source_root") if isinstance(source, Mapping) else None
    if SOURCE_SHA256 not in str(frozen_root):
        failures.append("identity.source.frozen_source_root does not bind frozen P2 snapshot")
    if renderer.get("renderer_id") != "zeta2-prm03-v1":
        failures.append("identity renderer is not zeta2-prm03-v1")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "canonical_sha256": identity_hash,
        "row_count": data.get("row_count"),
        "ordered_ids_sha256": data.get("ordered_ids_sha256"),
        "frozen_source_root": frozen_root,
        "parent_manifest_sha256": parent.get("manifest_sha256"),
        "policy": {key: policy.get(key) for key in (
            "candidate_count", "generation_groups_per_call", "gradient_accumulation_steps",
            "per_device_train_batch_size", "prompt_max_tokens", "completion_max_tokens",
            "context_max_tokens", "model_load_max_seq_length", "cuda_memory_fraction",
        )},
        "schedule": {key: schedule.get(key) for key in (
            "source_draws_per_update", "generation_batch_size", "buffer_reuse",
            "steps_per_generation", "source_draw_schedule_sha256", "source_draw_sequence_sha256",
        )},
    }


def load_review(root: Path) -> dict[str, Any]:
    path = root / "output" / "load-audit.json"
    if not path.is_file():
        return {"status": "pending", "path": str(path), "reason": "load audit absent"}
    try:
        value = read_json(path)
    except Exception as error:
        return {"status": "fail", "path": str(path), "reason": f"invalid JSON: {error}"}
    if not isinstance(value, Mapping):
        return {"status": "fail", "path": str(path), "reason": "load audit is not an object"}
    adapter = value.get("adapter") if isinstance(value.get("adapter"), Mapping) else {}
    model_load = value.get("model_load") if isinstance(value.get("model_load"), Mapping) else {}
    allocator = value.get("cuda_allocator") if isinstance(value.get("cuda_allocator"), Mapping) else {}
    failures: list[str] = []
    if value.get("dtype") not in {"torch.bfloat16", "bfloat16"}:
        failures.append(f"dtype is {value.get('dtype')!r}, expected BF16")
    if adapter.get("attachments") != 294:
        failures.append(f"adapter attachments are {adapter.get('attachments')!r}, expected 294")
    if adapter.get("trainable_parameters") != 25116672:
        failures.append(f"trainable parameters are {adapter.get('trainable_parameters')!r}, expected 25116672")
    if model_load.get("requested_max_seq_length") != 4096 or model_load.get("observed_capacity_tokens") != 4096:
        failures.append("loader request/observed capacity is not 4096")
    if allocator.get("cuda_allocator_fraction") != 0.75:
        failures.append("CUDA allocator fraction is not 0.75")
    return {
        "status": "pass" if not failures else "fail", "path": str(path), "failures": failures,
        "dtype": value.get("dtype"),
        "adapter": {key: adapter.get(key) for key in ("attachments", "trainable_parameters", "lora_rank", "lora_alpha")},
        "model_load": {key: model_load.get(key) for key in ("requested_max_seq_length", "observed_capacity_tokens", "status")},
        "cuda_allocator": {key: allocator.get(key) for key in ("cuda_allocator_fraction", "cuda_allocator_total_bytes", "cuda_allocator_cap_bytes")},
    }


def generation_review(root: Path, expected_updates: int = UPDATES, update_start: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, state = read_jsonl(root / "output" / "generation-records.jsonl")
    if state["status"] == "pending":
        return [], state
    failures: list[str] = list(state.get("failures", []))
    compact_rows: list[dict[str, Any]] = []
    by_update: Counter[int] = Counter()
    by_call: Counter[tuple[int, int]] = Counter()
    by_group: Counter[tuple[int, int]] = Counter()
    terminal: Counter[str] = Counter()
    for index, record in enumerate(rows):
        if record.get("schema_version") != GENERATION_SCHEMA:
            failures.append(f"row {index}: schema_version mismatch")
        update = record.get("global_step")
        geometry = record.get("group_geometry")
        if type(update) is not int or not isinstance(geometry, Mapping):
            failures.append(f"row {index}: missing integer global_step/group_geometry")
            continue
        call = geometry.get("generation_call_index")
        group = geometry.get("group_index")
        row_in_group = geometry.get("group_row_index")
        if any(type(item) is not int for item in (call, group, row_in_group)):
            failures.append(f"row {index}: noninteger P2 geometry")
        else:
            by_update[update] += 1
            by_call[(update, call)] += 1
            by_group[(update, group)] += 1
            expected_call = (index % ROWS_PER_UPDATE) // (GROUPS_PER_CALL * CANDIDATES)
            expected_group = (index % ROWS_PER_UPDATE) // CANDIDATES
            expected_row = index % CANDIDATES
            if (call, group, row_in_group) != (expected_call, expected_group, expected_row):
                failures.append(f"row {index}: ordered P2 geometry is {(call, group, row_in_group)!r}, expected {(expected_call, expected_group, expected_row)!r}")
        if geometry.get("generation_call_count") != ROWS_PER_UPDATE // (GROUPS_PER_CALL * CANDIDATES):
            failures.append(f"row {index}: generation_call_count is not 4")
        if geometry.get("generation_groups_per_call") != GROUPS_PER_CALL:
            failures.append(f"row {index}: generation_groups_per_call is not 2")
        if geometry.get("generation_group_count") != GROUPS_PER_UPDATE:
            failures.append(f"row {index}: generation_group_count is not 8")
        if geometry.get("generation_row_count") != ROWS_PER_UPDATE:
            failures.append(f"row {index}: generation_row_count is not 32")
        generated = record.get("generated_ids")
        if not isinstance(generated, list) or not generated or any(type(token) is not int for token in generated):
            failures.append(f"row {index}: generated_ids is empty/noninteger")
        elif record.get("generated_token_count") != len(generated):
            failures.append(f"row {index}: generated_token_count mismatch")
        for key in ("prompt_ids_sha256", "generated_ids_sha256"):
            if not isinstance(record.get(key), str) or len(record[key]) != 64:
                failures.append(f"row {index}: {key} missing")
        if type(record.get("padded_after_terminal")) is not int or record["padded_after_terminal"] < 0:
            failures.append(f"row {index}: invalid padded_after_terminal")
        terminal[str(record.get("terminal_reason"))] += 1
        compact_rows.append({
            "ordinal": index,
            "update": update,
            "call": geometry.get("generation_call_index"),
            "group": geometry.get("group_index"),
            "row": geometry.get("group_row_index"),
            "prompt_sha256": record.get("prompt_ids_sha256"),
            "output_sha256": record.get("generated_ids_sha256"),
            "generated_tokens": record.get("generated_token_count"),
            "terminal_reason": record.get("terminal_reason"),
        })
    expected_updates_set = set(range(update_start, update_start + expected_updates))
    if rows and any(by_update.get(update, 0) != ROWS_PER_UPDATE for update in expected_updates_set):
        failures.append(f"expected each update to contain 32 rows, got {dict(by_update)}")
    if len(rows) == expected_updates * ROWS_PER_UPDATE and sorted(by_update) != sorted(expected_updates_set):
        failures.append(f"expected update indices {sorted(expected_updates_set)}, got {sorted(by_update)}")
    if len(rows) == expected_updates * ROWS_PER_UPDATE:
        for update in expected_updates_set:
            if any(by_call[(update, call)] != GROUPS_PER_CALL * CANDIDATES for call in range(4)):
                    failures.append(f"update {update}: one or more calls do not contain 8 rows")
            if any(by_group[(update, group)] != CANDIDATES for group in range(GROUPS_PER_UPDATE)):
                failures.append(f"update {update}: one or more groups do not contain 4 rows")
    complete = len(rows) == expected_updates * ROWS_PER_UPDATE and state["status"] == "present"
    status = "fail" if failures else ("pass" if complete else "pending")
    return compact_rows, {
        **state, "status": status, "failures": failures, "rows": len(rows),
        "rows_by_update": dict(by_update),
        "calls_by_update": {f"{u}:{c}": by_call[(u, c)] for u in sorted(by_update) for c in range(4)},
        "groups_by_update": {f"{u}:{g}": by_group[(u, g)] for u in sorted(by_update) for g in range(GROUPS_PER_UPDATE)},
        "terminal_reason_counts": dict(terminal),
        "contract": {"calls_per_update": 4, "logical_groups_per_update": 8, "candidates_per_group": 4, "rows_per_update": 32},
    }


def source_prefix(path: Path | None) -> list[str] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = read_json(path)
        ids = value.get("prefix_row_ids") if isinstance(value, Mapping) else None
        if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
            return [item for item in ids for _ in range(CANDIDATES)]
    except Exception:
        return None
    return None


def reward_review(root: Path, generation: list[dict[str, Any]], prefix_path: Path | None, expected_updates: int = UPDATES, offset: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, state = read_jsonl(root / "output" / "reward-records.jsonl")
    if state["status"] == "pending":
        return [], state
    failures: list[str] = list(state.get("failures", []))
    joined: list[dict[str, Any]] = []
    for index, reward in enumerate(rows):
        source_id = reward.get("id")
        output_sha = reward.get("output_ids_sha256")
        if not isinstance(source_id, str) or not source_id:
            failures.append(f"row {index}: missing source ID")
        if not isinstance(output_sha, str) or len(output_sha) != 64:
            failures.append(f"row {index}: missing output hash")
        if not finite(reward.get("reward")):
            failures.append(f"row {index}: nonfinite reward")
        if index < len(generation) and output_sha != generation[index].get("output_sha256"):
            failures.append(f"row {index}: ordered generation/reward output hash mismatch")
        if index < len(generation):
            joined.append({
                "ordinal": index, "update": generation[index].get("update"),
                "call": generation[index].get("call"), "group": generation[index].get("group"),
                "row": generation[index].get("row"), "source_id": source_id,
                "prompt_sha256": generation[index].get("prompt_sha256"),
                "output_sha256": output_sha, "reward": reward.get("reward"),
                "failure": reward.get("failure"),
            })
    if generation and len(rows) != len(generation):
        failures.append(f"reward rows {len(rows)} differ from generation rows {len(generation)}")
    expected = source_prefix(prefix_path)
    if expected is not None:
        observed = [row.get("id") for row in rows]
        wanted = expected[offset:offset + len(observed)]
        if observed != wanted:
            failures.append("reward source IDs do not match admitted ordered source prefix")
    if len(rows) == expected_updates * ROWS_PER_UPDATE and expected is not None:
        wanted = expected[offset:offset + len(rows)]
        if [row.get("id") for row in rows] != wanted:
            failures.append("complete reward source sequence differs from admitted prefix")
    outcomes = Counter(str(row.get("failure")) for row in rows)
    eos = sum(bool(row.get("canonical_eos")) for row in rows)
    caps = sum(bool(row.get("cap_hit")) for row in rows)
    finite_rewards = all(finite(row.get("reward")) for row in rows)
    complete = len(rows) == expected_updates * ROWS_PER_UPDATE and state["status"] == "present"
    status = "fail" if failures else ("pass" if complete else "pending")
    return joined, {
        **state, "status": status, "failures": failures, "rows": len(rows),
        "join_key": "ordered generation/reward row + output_ids_sha256; reward supplies source_id",
        "source_ids_prefix": [row.get("source_id") for row in joined[:16]],
        "failure_counts": dict(outcomes), "canonical_eos": eos, "cap_hits": caps,
        "finite_rewards": finite_rewards,
    }


def gradient_review(root: Path, expected_updates: int = UPDATES) -> dict[str, Any]:
    rows, state = read_jsonl(root / "output" / "gradient-records.jsonl")
    if state["status"] == "pending":
        return state
    failures: list[str] = list(state.get("failures", []))
    for index, row in enumerate(rows):
        if row.get("finite") is not True or row.get("nonfinite") is not False:
            failures.append(f"row {index}: gradient finite flags invalid")
        if type(row.get("grad_present_count")) is not int or row["grad_present_count"] <= 0:
            failures.append(f"row {index}: no present LoRA gradients")
        if type(row.get("nonzero_tensor_count")) is not int or row["nonzero_tensor_count"] <= 0:
            failures.append(f"row {index}: all LoRA gradients are zero")
        if not finite(row.get("norm")) or row["norm"] <= 0:
            failures.append(f"row {index}: gradient norm is not finite/nonzero")
    complete = len(rows) == expected_updates and state["status"] == "present"
    if rows and len(rows) > expected_updates:
        failures.append(f"expected {expected_updates} gradient records, got {len(rows)}")
    return {
        **state, "status": "fail" if failures else ("pass" if complete else "pending"),
        "failures": failures, "steps": [row.get("step") for row in rows],
        "norms": [row.get("norm") for row in rows],
        "grad_present": [row.get("grad_present_count") for row in rows],
        "nonzero_tensors": [row.get("nonzero_tensor_count") for row in rows],
    }


def checkpoint_dir(root: Path, step: int) -> Path | None:
    candidates = (
        root / "archive" / "full" / f"checkpoint-{step}",
        root / "output" / f"checkpoint-{step}",
    )
    return next((path for path in candidates if path.is_dir()), None)


def hash_tree(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in sorted(path.iterdir(), key=lambda target: target.name):
        if item.is_file():
            result[item.name] = {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
    return result


def checkpoint_review(root: Path, step: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
    path = checkpoint_dir(root, step)
    if path is None:
        expected = root / "archive" / "full" / f"checkpoint-{step}"
        return {"status": "pending", "step": step, "path": str(expected), "reason": "full checkpoint absent"}, None
    failures: list[str] = []
    hashes: dict[str, dict[str, Any]] = {}
    for name in FULL_STATE_FILES:
        target = path / name
        if not target.is_file() or target.stat().st_size <= 0:
            failures.append(f"missing/empty {name}")
    try:
        hashes = hash_tree(path)
        state = read_json(path / "campaign-state.json")
        manifest = read_json(path / "campaign-manifest.json")
        trainer_state = read_json(path / "trainer_state.json")
    except Exception as error:
        return {"status": "fail", "step": step, "path": str(path), "failures": [f"checkpoint unreadable: {error}"]}, hashes or None
    if not isinstance(state, Mapping) or state.get("full") is not True or state.get("step") != step:
        failures.append("campaign-state is not a matching full checkpoint")
    if not isinstance(manifest, Mapping) or manifest.get("full") is not True or manifest.get("step") != step:
        failures.append("campaign-manifest is not a matching full checkpoint")
    if not isinstance(trainer_state, Mapping) or trainer_state.get("global_step") != step:
        failures.append("trainer_state.global_step does not match checkpoint step")
    sampler = state.get("sampler") if isinstance(state, Mapping) else None
    if not isinstance(sampler, Mapping):
        failures.append("campaign-state sampler payload missing")
        sampler = {}
    for key, wanted in (("source_draw_schedule_sha256", SCHEDULE_SHA256), ("source_draw_sequence_sha256", SEQUENCE_SHA256)):
        if sampler.get(key) != wanted:
            failures.append(f"sampler.{key} differs from admitted schedule")
    expected_counters = {
        1: {"consumed_rows": 128, "current_index": 128, "source_draw_cursor": 8},
        2: {"consumed_rows": 256, "current_index": 256, "source_draw_cursor": 16},
    }[step]
    for key, wanted in expected_counters.items():
        if sampler.get(key) != wanted:
            failures.append(f"sampler.{key}: expected {wanted}, got {short(sampler.get(key))}")
    # Validate the manifest's own file inventory against the bytes we hashed.
    manifest_files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if isinstance(manifest_files, Mapping):
        for name, expected in manifest_files.items():
            observed = hashes.get(str(name))
            if not isinstance(expected, Mapping) or observed is None or expected.get("bytes") != observed["bytes"] or expected.get("sha256") != observed["sha256"]:
                failures.append(f"manifest file inventory mismatch: {name}")
    compact_state = {
        "full": state.get("full") if isinstance(state, Mapping) else None,
        "step": state.get("step") if isinstance(state, Mapping) else None,
        "path": str(path),
        "sampler": {key: sampler.get(key) for key in (
            "source_draw_schedule_sha256", "source_draw_sequence_sha256",
            "consumed_rows", "current_index", "source_draw_cursor", "consumed_prompt_copies",
        )},
        "file_hashes": hashes,
        "exact_resume_files": list(EXACT_RESUME_FILES),
        "metadata_allowlist": list(METADATA_ALLOWLIST),
    }
    return {
        "status": "pass" if not failures else "fail", "step": step, "path": str(path),
        "failures": failures, "sampler": compact_state["sampler"],
        "file_hashes": hashes,
        "exact_resume_files": list(EXACT_RESUME_FILES),
        "metadata_allowlist": list(METADATA_ALLOWLIST),
        "comparison_scope": "exact adapter_model.safetensors, optimizer.pt, scheduler.pt and rng_state.pth; sampler/source fields and joined output rows; metadata allowlist audited separately",
    }, compact_state


def panel_ids(panel: Path | None) -> list[str]:
    if panel is None or not panel.is_file():
        return []
    rows, state = read_jsonl(panel)
    return [str(row["id"]) for row in rows if isinstance(row.get("id"), str)] if state["status"] in {"present", "pending_partial"} else []


def evaluation_review(root: Path, panel: Path | None, steps: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    expected_ids = panel_ids(panel)
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for step in steps:
        request_path = root / "archive" / "evaluations" / f"step-{step}.json"
        cases_path = root / "archive" / "evaluations" / f"cases-step-{step}.json"
        if not request_path.is_file() or not cases_path.is_file():
            request_status = None
            if request_path.is_file():
                try:
                    request_value = read_json(request_path)
                    request_status = request_value.get("status") if isinstance(request_value, Mapping) else None
                except Exception as error:
                    request_status = f"unreadable:{type(error).__name__}"
            deferred = isinstance(request_status, str) and request_status.startswith("deferred:")
            reports.append({"step": step, "status": "deferred" if deferred else "pending", "request_status": request_status, "request": str(request_path), "cases": str(cases_path)})
            continue
        try:
            request = read_json(request_path)
            cases = read_json(cases_path)
        except Exception as error:
            failures.append(f"step {step}: evaluation unreadable: {error}")
            continue
        results = cases.get("results") if isinstance(cases, Mapping) else None
        summary = cases.get("summary") if isinstance(cases, Mapping) else None
        ids = [row.get("id") for row in results] if isinstance(results, list) else []
        local_failures: list[str] = []
        if cases.get("status") != "complete" or not isinstance(results, list) or len(results) != DEV_ROWS:
            local_failures.append("expected complete 14-case evaluation")
        if expected_ids and sorted(ids) != sorted(expected_ids):
            local_failures.append("case IDs differ from frozen 14-case panel")
        long_rows = [row for row in results if row.get("id") == LONG_DEV_ID] if isinstance(results, list) else []
        if not long_rows or long_rows[0].get("prompt_tokens") != LONG_DEV_PROMPT:
            local_failures.append("2619-token maximal DEV case missing or changed")
        if isinstance(results, list) and any(not isinstance(row.get("generated_tokens"), (int, float)) or row.get("generated_tokens") > DEV_CAP for row in results):
            local_failures.append("DEV generation exceeded cap 512")
        denominators = summary.get("denominators") if isinstance(summary, Mapping) else None
        if not isinstance(denominators, Mapping) or denominators.get("cases") != DEV_ROWS:
            local_failures.append("summary denominator is not 14 cases")
        failures.extend(f"step {step}: {failure}" for failure in local_failures)
        reports.append({
            "step": step, "status": "pass" if not local_failures else "fail",
            "request_status": request.get("status") if isinstance(request, Mapping) else None,
            "cases_status": cases.get("status") if isinstance(cases, Mapping) else None,
            "rows": len(results) if isinstance(results, list) else 0,
            "long_case_prompt_tokens": long_rows[0].get("prompt_tokens") if long_rows else None,
            "denominators": denominators,
        })
    if failures:
        status = "fail"
    elif reports and all(report.get("status") == "pass" for report in reports):
        status = "pass"
    else:
        status = "pending"
    return {"status": status, "failures": failures, "panel": str(panel) if panel else None, "reports": reports}


def first_failure(root: Path) -> str | None:
    for path in (root / "output" / "failure.json", root / "failure.json"):
        if path.is_file():
            try:
                value = read_json(path)
                if isinstance(value, Mapping):
                    return f"{path.name}: {value.get('error', value.get('status', 'failure'))}"
                return f"{path.name}: failure receipt is not an object"
            except Exception as error:
                return f"{path.name}: unreadable failure receipt: {error}"
    host_terminal = root.parent / f"{root.name}-host-supervision" / "terminal.json"
    if host_terminal.is_file():
        try:
            value = read_json(host_terminal)
            if isinstance(value, Mapping) and value.get("status") in {"stopped_or_failed", "failed", "timeout"}:
                return f"host supervision: {value.get('status')}: {value.get('reason')}"
        except Exception as error:
            return f"host supervision terminal unreadable: {error}"
    log = root.parent / f"{root.name}-host-supervision" / "process.log"
    if log.is_file():
        try:
            with log.open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 4096))
                tail = stream.read().decode("utf-8", "replace")
            for line in tail.splitlines():
                if re.search(r"(Traceback|Error|Exception|out of memory|failed|CUBLAS|illegal memory|abort)", line, re.I):
                    return f"process.log tail: {line.strip()[:500]}"
        except OSError as error:
            return f"process.log tail unreadable: {error}"
    return None


def terminal_status(supervision: Mapping[str, Any] | None, supervision_review: Mapping[str, Any], failure: str | None, max_step: int = UPDATES) -> dict[str, Any]:
    result = supervision_review.get("entry_result") if isinstance(supervision_review.get("entry_result"), Mapping) else {}
    status = result.get("status")
    step = result.get("step")
    if failure and failure.startswith("host supervision:"):
        classification = "host_guard_stopped"
    elif status == "deadline" and (not isinstance(step, int) or step < max_step):
        classification = "incomplete_deadline"
    elif failure and status in {"entry_failed", "training_failed"}:
        classification = "failed"
    elif status in {"schedule_complete", "trainer_terminal", "lead_decision"} and step == max_step:
        classification = "complete"
    elif status:
        classification = "terminal_unclassified"
    else:
        classification = "pending"
    return {"status": classification, "entry_status": status, "step": step, "checkpoint": result.get("checkpoint"), "error": failure}


def summarize(root: Path, prefix: Path | None, panel: Path | None, expected_updates: int = UPDATES, checkpoint_steps: tuple[int, ...] = (1, 2), prefix_offset: int = 0, update_start: int = 0, expected_recipe: str = RECIPE_SHA256) -> dict[str, Any]:
    supervision_review, _supervision = review_supervision(root, expected_recipe)
    identity, events, telemetry = read_telemetry(root / "telemetry.jsonl")
    identity_result = identity_review(identity)
    # Do not hold the 50 MiB train_begin identity while loading 148 MiB
    # checkpoint-state JSON documents.
    identity = None
    generation, generations = generation_review(root, expected_updates, update_start)
    joined, rewards = reward_review(root, generation, prefix, expected_updates, prefix_offset)
    gradients = gradient_review(root, expected_updates)
    load = load_review(root)
    checkpoints: list[dict[str, Any]] = []
    checkpoint_states: dict[int, dict[str, Any] | None] = {}
    for step in checkpoint_steps:
        review, state = checkpoint_review(root, step)
        checkpoints.append(review)
        checkpoint_states[step] = state
    evaluations = evaluation_review(root, panel, checkpoint_steps)
    failure = first_failure(root)
    terminal = terminal_status(None, supervision_review, failure, max(checkpoint_steps, default=expected_updates))
    recipe_output = root / "output" / "admitted-recipe.json"
    recipe_output_hash = sha256_file(recipe_output) if recipe_output.is_file() else None
    recipe_output_review = {
        "status": "pass" if recipe_output_hash == expected_recipe else ("pending" if recipe_output_hash is None else "fail"),
        "sha256": recipe_output_hash,
        "expected_sha256": expected_recipe,
    }
    checks = [supervision_review.get("status"), telemetry.get("status"), identity_result.get("status"), generations.get("status"), rewards.get("status"), gradients.get("status"), load.get("status"), *(item.get("status") for item in checkpoints), evaluations.get("status"), recipe_output_review.get("status")]
    if terminal["status"] in {"failed", "incomplete_deadline"}:
        overall = terminal["status"]
    elif any(value == "fail" for value in checks):
        overall = "fail"
    elif any(value in {"pending", "pending_partial", "deferred"} for value in checks) or terminal["status"] == "pending":
        overall = "pending"
    else:
        overall = "pass"
    return {
        "status": overall, "run_root": str(root), "source_snapshot_sha256": SOURCE_SHA256,
        "recipe_sha256_expected": expected_recipe, "identity_sha256_expected": IDENTITY_SHA256,
        "recipe_output_sha256": recipe_output_hash,
        "recipe_output": recipe_output_review,
        "terminal": terminal, "failure_first": failure,
        "supervision": supervision_review, "telemetry": telemetry, "control_events": events,
        "identity": identity_result, "load": load, "generation": generations,
        "rewards": rewards, "gradients": gradients, "checkpoints": checkpoints,
        "evaluations": evaluations, "join_sequence": joined, "checkpoint_states": checkpoint_states,
    }


def cpu_torch_state_comparison(left_path: Path, right_path: Path, filename: str) -> dict[str, Any]:
    """Compare a trusted checkpoint state on CPU after a byte mismatch.

    This function is deliberately lazy: importing this verifier remains
    framework-free.  Only owned optimizer/scheduler/RNG checkpoint files are
    passed here, and ``map_location='cpu'`` prevents device allocation.  The
    optimizer serializer can choose different storage alias layouts while
    representing the same state, so tensor/value equality is reported apart
    from container-byte equality.
    """
    try:
        torch = __import__("torch")
    except Exception as error:
        return {"status": "unavailable", "filename": filename, "reason": f"torch import failed: {error}"}

    def load(path: Path) -> tuple[Any, str]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
        except Exception as error:
            # rng_state.pth from this pinned runtime contains NumPy state that
            # PyTorch's restrictive loader rejects.  It is still trusted owned
            # checkpoint data and is loaded only on CPU for this audit.
            try:
                return torch.load(path, map_location="cpu", weights_only=False), "trusted_cpu_fallback"
            except Exception as fallback_error:
                raise RuntimeError(f"weights_only={error}; trusted fallback={fallback_error}") from fallback_error

    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            return isinstance(a, Mapping) and isinstance(b, Mapping) and set(a) == set(b) and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        try:
            value = a == b
            return bool(value) if not isinstance(value, torch.Tensor) else bool(torch.all(value))
        except Exception:
            return repr(a) == repr(b)

    def tensor_layout(value: Any) -> tuple[int, dict[str, int], dict[str, int]]:
        tensors: list[Any] = []

        def visit(item: Any) -> None:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, Mapping):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        storage_counts: Counter[tuple[int, int]] = Counter()
        for tensor in tensors:
            storage = tensor.untyped_storage()
            storage_counts[(int(storage.data_ptr()), int(storage.nbytes()))] += 1
        histogram = Counter(str(count) for count in storage_counts.values())
        return len(tensors), dict(sorted(histogram.items())), {"storage_groups": len(storage_counts)}

    try:
        left, left_mode = load(left_path)
        right, right_mode = load(right_path)
        left_count, left_alias_hist, left_storage = tensor_layout(left)
        right_count, right_alias_hist, right_storage = tensor_layout(right)
        container: dict[str, Any] = {}
        try:
            with zipfile.ZipFile(left_path) as archive:
                container["left_zip_entries"] = len(archive.namelist())
                container["left_storage_entries"] = sum("/data/" in name for name in archive.namelist())
            with zipfile.ZipFile(right_path) as archive:
                container["right_zip_entries"] = len(archive.namelist())
                container["right_storage_entries"] = sum("/data/" in name for name in archive.namelist())
        except zipfile.BadZipFile:
            container["zip_container"] = "not a zip serialization"
        semantic_equal = equal(left, right)
        return {
            "status": "semantic_equal" if semantic_equal else "semantic_different",
            "filename": filename,
            "load_modes": {"left": left_mode, "right": right_mode},
            "tensor_counts": {"left": left_count, "right": right_count},
            "storage_groups": {"left": left_storage["storage_groups"], "right": right_storage["storage_groups"]},
            "alias_histograms": {"left": left_alias_hist, "right": right_alias_hist},
            "container": container,
            "interpretation": "tensor/value state is equal; differing storage aliasing/zip layout is a serialization-container difference" if semantic_equal else "loaded CPU state values differ",
        }
    except Exception as error:
        return {"status": "error", "filename": filename, "reason": str(error)[:1000]}


def compare_runs(continuous: Mapping[str, Any], first: Mapping[str, Any], resumed: Mapping[str, Any]) -> dict[str, Any]:
    complete_reviews = all(item.get("status") == "pass" for item in (continuous, first, resumed))
    left = continuous.get("join_sequence", [])
    right = list(first.get("join_sequence", [])) + list(resumed.get("join_sequence", []))
    failures: list[dict[str, Any]] = []
    available: dict[str, Any] = {}
    exact_state_audits: dict[str, Any] = {}
    # Ordinal is local to each arm (the resumed segment starts at zero), so
    # remove it before comparing the actual source/prompt/output/reward row.
    left_comparison = [{key: value for key, value in row.items() if key != "ordinal"} for row in left if isinstance(row, Mapping)]
    right_comparison = [{key: value for key, value in row.items() if key != "ordinal"} for row in right if isinstance(row, Mapping)]
    if left_comparison != right_comparison:
        index = next((i for i, (a, b) in enumerate(zip(left_comparison, right_comparison)) if a != b), min(len(left_comparison), len(right_comparison)))
        failures.append({"kind": "ordered_source_generation_reward_output", "first_difference": index, "continuous": left_comparison[index:index + 1], "split": right_comparison[index:index + 1]})
        available["ordered_source_generation_reward_output"] = "fail"
    elif left_comparison and right_comparison:
        available["ordered_source_generation_reward_output"] = "pass"
    else:
        available["ordered_source_generation_reward_output"] = "pending: rows absent"
    left_sources = [row.get("source_id") for row in left if isinstance(row, Mapping)]
    right_sources = [row.get("source_id") for row in right if isinstance(row, Mapping)]
    if left_sources != right_sources:
        failures.append({"kind": "ordered_source_ids", "reason": "source IDs differ after first+resume concatenation"})
        available["ordered_source_ids"] = "fail"
    elif left_sources and right_sources:
        available["ordered_source_ids"] = "pass"
    else:
        available["ordered_source_ids"] = "pending: rows absent"
    boundary_hashes: dict[str, Any] = {}
    for step in (1, 2):
        left_state = continuous.get("checkpoint_states", {}).get(step)
        source = first if step == 1 else resumed
        right_state = source.get("checkpoint_states", {}).get(step)
        boundary_hashes[str(step)] = {
            "continuous_exact_files": {
                name: ((left_state or {}).get("file_hashes") or {}).get(name)
                for name in EXACT_RESUME_FILES
            } if isinstance(left_state, Mapping) else None,
            "split_exact_files": {
                name: ((right_state or {}).get("file_hashes") or {}).get(name)
                for name in EXACT_RESUME_FILES
            } if isinstance(right_state, Mapping) else None,
            "metadata_allowlist": {
                name: {
                    "continuous": ((left_state or {}).get("file_hashes") or {}).get(name)
                    if isinstance(left_state, Mapping) else None,
                    "split": ((right_state or {}).get("file_hashes") or {}).get(name)
                    if isinstance(right_state, Mapping) else None,
                    "comparison": "allowlisted metadata; differences do not invalidate resume equality",
                }
                for name in METADATA_ALLOWLIST
            },
        }
        if not isinstance(left_state, Mapping) or not isinstance(right_state, Mapping):
            available[f"checkpoint_{step}"] = "pending: state absent"
            continue
        available[f"checkpoint_{step}"] = "pass"
        left_files = left_state.get("file_hashes")
        right_files = right_state.get("file_hashes")
        for name in EXACT_RESUME_FILES:
            left_hash = left_files.get(name) if isinstance(left_files, Mapping) else None
            right_hash = right_files.get(name) if isinstance(right_files, Mapping) else None
            if left_hash == right_hash:
                exact_state_audits[f"{step}:{name}"] = {"status": "byte_equal", "filename": name}
                continue
            left_path = Path(str(left_state.get("path", ""))) / name
            right_path = Path(str(right_state.get("path", ""))) / name
            if name in {"optimizer.pt", "scheduler.pt", "rng_state.pth"} and left_path.is_file() and right_path.is_file():
                inspection = cpu_torch_state_comparison(left_path, right_path, name)
                exact_state_audits[f"{step}:{name}"] = {
                    "byte_status": "different", "left": left_hash, "right": right_hash,
                    "cpu_inspection": inspection,
                }
                if inspection.get("status") != "semantic_equal":
                    failures.append({"kind": "exact_resume_state", "step": step, "file": name, "reason": "CPU-loaded optimizer/scheduler/RNG state differs"})
            else:
                exact_state_audits[f"{step}:{name}"] = {"byte_status": "different", "left": left_hash, "right": right_hash}
                failures.append({"kind": "exact_resume_bytes", "step": step, "file": name, "reason": "adapter tensor bytes differ or state inspection unavailable"})
        if left_state.get("sampler") != right_state.get("sampler"):
            failures.append({"kind": "sampler_state", "step": step, "reason": "sampler/source cursor differs"})
    if failures:
        status = "fail"
    elif complete_reviews:
        status = "pass"
    else:
        status = "pending"
    result = {
        "status": status, "failures": failures, "available_checks": available,
        "compared": {"continuous_rows": len(left), "split_rows": len(right), "optimizer_boundaries": [1, 2],
                      "ordered_join_key": "source_id,prompt_sha256,output_sha256,reward,failure",
                      "exact_resume_files": list(EXACT_RESUME_FILES),
                      "metadata_allowlist": list(METADATA_ALLOWLIST),
                      "source_id_sequence": "exact ordered equality"},
        "boundary_hashes": boundary_hashes,
        "exact_state_audits": exact_state_audits,
        "interpretation": "mechanical identity/resume proof only; no quality promotion",
    }
    if not complete_reviews:
        result["reason"] = "one or more arm reviews are incomplete; available equality checks are reported but cannot establish resume proof"
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-prefix-audit", type=Path)
    parser.add_argument("--development-panel", type=Path)
    parser.add_argument("--split-first", type=Path)
    parser.add_argument("--split-resume", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.run_root.resolve()
    result = summarize(root, args.source_prefix_audit, args.development_panel)
    result["observed_at"] = datetime.now(timezone.utc).isoformat()
    if args.split_first and args.split_resume:
        first = summarize(args.split_first.resolve(), args.source_prefix_audit, args.development_panel, 1, (1,), 0, 0, SPLIT_FIRST_RECIPE_SHA256)
        resumed = summarize(args.split_resume.resolve(), args.source_prefix_audit, args.development_panel, 1, (2,), ROWS_PER_UPDATE, 1, SPLIT_RESUME_RECIPE_SHA256)
        result["split_reviews"] = {"first": first, "resume": resumed}
        result["split_comparison"] = compare_runs(result, first, resumed)
    else:
        result["split_comparison"] = {"status": "pending", "reason": "first and resumed split roots were not supplied; continuous run cannot prove resume equality"}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    printable = {key: result[key] for key in ("status", "run_root", "failure_first", "terminal", "generation", "rewards", "gradients", "checkpoints", "evaluations", "supervision") if key in result}
    print(json.dumps(printable, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"pass", "pending", "failed", "incomplete_deadline"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
