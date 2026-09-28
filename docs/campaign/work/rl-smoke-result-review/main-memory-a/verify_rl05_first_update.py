#!/usr/bin/env python3
"""Bounded CPU-only independent review of the first update of the RL main run.

The live entrypoint writes a few very large identity and trainer-begin records.
This verifier never imports a framework, model, CUDA, or project module and
never loads those records.  It reads only compact launch/preflight/guard
metadata and the first 32 generation/reward rows, the first gradient record,
and bounded telemetry JSONL records.  A valid result is evidence for the first
optimizer update only; it says nothing about later updates, checkpoints, or
development quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


SOURCE_SHA256 = "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9"
SOURCE_PREFIX_AUDIT_SHA256 = ""
SOURCE_SCHEDULE_SHA256 = "892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48"
SOURCE_SEQUENCE_SHA256 = "d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2"
EXPECTED_RENDERER = "zeta2-prm03-v1"
EXPECTED_DEV_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
EXPECTED_DEV_ROWS = 75
EXPECTED_DEV_CAP = 512
EXPECTED_DEV_CONTEXT = 4096
EXPECTED_CANDIDATES = 4
EXPECTED_GROUPS_PER_CALL = 2
EXPECTED_CALLS_PER_UPDATE = 4
EXPECTED_GROUPS_PER_UPDATE = 8
EXPECTED_ROWS_PER_UPDATE = 32
EXPECTED_COMPLETION_CAP = 192
GENERATION_SCHEMA = "sepalith.rl.generation-record.v1"
GRADIENT_SCHEMA = "sepalith.rl.gradient-record.v1"
MAX_JSONL_LINE_BYTES = 1 * 1024 * 1024
JSONL_CHUNK_BYTES = 64 * 1024


class ReviewError(Exception):
    """Raised for an invalid compact artifact."""


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def short(value: Any, limit: int = 240) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def read_small_json(path: Path, limit: int = MAX_JSONL_LINE_BYTES) -> Any:
    """Read a known compact JSON artifact without accepting giant receipts."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > limit:
        raise ReviewError(f"refusing JSON artifact larger than {limit} bytes: {path}")
    with path.open("rb") as stream:
        return json.loads(stream.read())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_line(raw: bytes, line_number: int) -> Mapping[str, Any] | None:
    if not raw.strip():
        raise ReviewError(f"empty JSONL line {line_number}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReviewError(f"invalid JSONL line {line_number}: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise ReviewError(f"JSONL line {line_number} is not an object")
    return value


def read_jsonl_prefix(
    path: Path,
    max_records: int,
    *,
    max_line_bytes: int = MAX_JSONL_LINE_BYTES,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Read at most ``max_records`` objects with a bounded line accumulator.

    ``readline()`` is deliberately avoided: telemetry's train-begin line can
    exceed 100 MB.  Oversize lines are counted and discarded one chunk at a
    time.  For generation/reward/gradient streams an oversize line is an
    artifact failure; for telemetry the caller may explicitly allow the
    skipped line.
    """
    state: dict[str, Any] = {
        "path": str(path),
        "status": "pending",
        "lines_seen": 0,
        "records": 0,
        "oversize_lines": 0,
        "oversize_bytes": 0,
        "trailing_partial": False,
        "failures": [],
    }
    if not path.is_file():
        state["reason"] = "artifact not present"
        return [], state

    records: list[Mapping[str, Any]] = []
    pending = bytearray()
    oversize = False

    def finish_line(raw: bytes, line_number: int, newline_bytes: int) -> bool:
        """Parse one completed line. Return true when the prefix is full."""
        state["lines_seen"] = line_number
        state["records"] = len(records)
        try:
            value = _jsonl_line(raw, line_number)
        except ReviewError as error:
            state["failures"].append(str(error))
            return False
        if value is not None:
            records.append(value)
            state["records"] = len(records)
        return len(records) >= max_records

    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                chunk = stream.read(JSONL_CHUNK_BYTES)
                if not chunk:
                    break
                position = 0
                while position < len(chunk):
                    newline = chunk.find(b"\n", position)
                    if newline < 0:
                        fragment = chunk[position:]
                        if oversize:
                            state["oversize_bytes"] += len(fragment)
                        elif len(pending) + len(fragment) > max_line_bytes:
                            state["oversize_lines"] += 1
                            state["oversize_bytes"] += len(pending) + len(fragment)
                            pending.clear()
                            oversize = True
                        else:
                            pending.extend(fragment)
                        break

                    fragment = chunk[position:newline]
                    line_number += 1
                    if oversize:
                        state["oversize_bytes"] += len(fragment) + 1
                        oversize = False
                        pending.clear()
                    else:
                        line_size = len(pending) + len(fragment)
                        if line_size > max_line_bytes:
                            state["oversize_lines"] += 1
                            state["oversize_bytes"] += line_size + 1
                            pending.clear()
                        else:
                            raw = bytes(pending) + fragment
                            pending.clear()
                            if finish_line(raw, line_number, 1):
                                state["status"] = "prefix_complete"
                                return records, state
                    position = newline + 1

            if oversize:
                state["oversize_bytes"] += len(pending)
                state["trailing_partial"] = True
            elif pending:
                line_number += 1
                state["trailing_partial"] = True
                finish_line(bytes(pending), line_number, 0)
    except OSError as error:
        state["status"] = "fail"
        state["failures"].append(str(error))
        return records, state

    state["lines_seen"] = max(state["lines_seen"], line_number)
    state["records"] = len(records)
    if state["failures"] or state["oversize_lines"]:
        state["status"] = "fail"
    elif len(records) >= max_records:
        state["status"] = "prefix_complete"
    else:
        state["status"] = "pending"
        if state["trailing_partial"]:
            state["reason"] = "last JSONL line is incomplete or lacks a newline"
        else:
            state["reason"] = f"only {len(records)} of {max_records} records are present"
    return records, state


def scan_telemetry(path: Path) -> dict[str, Any]:
    """Scan telemetry in chunks and retain only compact first-update events."""
    state: dict[str, Any] = {
        "path": str(path),
        "status": "pending",
        "lines_seen": 0,
        "event_counts": {},
        "oversize_lines_skipped": 0,
        "oversize_bytes_skipped": 0,
        "parse_failures": [],
        "events": [],
    }
    if not path.is_file():
        state["reason"] = "telemetry absent"
        return state

    # Retain no event payload that could carry the full identity.  Only these
    # scalars are useful to this first-update review.
    counts: Counter[str] = Counter()
    compact_events: list[dict[str, Any]] = []

    def visit(raw: bytes, line_number: int) -> None:
        if not raw.strip():
            state["parse_failures"].append(f"empty telemetry line {line_number}")
            return
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            state["parse_failures"].append(f"invalid telemetry line {line_number}: {error.msg}")
            return
        if not isinstance(value, Mapping):
            state["parse_failures"].append(f"telemetry line {line_number} is not an object")
            return
        event = str(value.get("event"))
        counts[event] += 1
        if event not in {"optimizer_step", "trainer_metrics", "train_end"}:
            return
        item: dict[str, Any] = {
            key: value.get(key)
            for key in ("event", "step", "global_step", "stop_reason", "remaining_seconds", "seconds")
            if key in value
        }
        if event == "trainer_metrics" and isinstance(value.get("metrics"), Mapping):
            metrics = value["metrics"]
            item["metrics"] = {
                key: metrics.get(key)
                for key in (
                    "reward", "reward_std", "frac_reward_zero_std", "grad_norm",
                    "completion_length", "completions/clipped_ratio", "num_tokens", "loss",
                )
                if key in metrics
            }
        if len(compact_events) < 12:
            compact_events.append(item)

    try:
        with path.open("rb") as stream:
            buffer = bytearray()
            oversize = False
            line_number = 0
            while True:
                chunk = stream.read(JSONL_CHUNK_BYTES)
                if not chunk:
                    break
                position = 0
                while position < len(chunk):
                    newline = chunk.find(b"\n", position)
                    if newline < 0:
                        fragment = chunk[position:]
                        if oversize:
                            state["oversize_bytes_skipped"] += len(fragment)
                        elif len(buffer) + len(fragment) > MAX_JSONL_LINE_BYTES:
                            state["oversize_lines_skipped"] += 1
                            state["oversize_bytes_skipped"] += len(buffer) + len(fragment)
                            buffer.clear()
                            oversize = True
                        else:
                            buffer.extend(fragment)
                        break
                    fragment = chunk[position:newline]
                    line_number += 1
                    if oversize:
                        state["oversize_bytes_skipped"] += len(fragment) + 1
                        oversize = False
                        buffer.clear()
                    else:
                        line_size = len(buffer) + len(fragment)
                        if line_size > MAX_JSONL_LINE_BYTES:
                            state["oversize_lines_skipped"] += 1
                            state["oversize_bytes_skipped"] += line_size + 1
                            buffer.clear()
                        else:
                            visit(bytes(buffer) + fragment, line_number)
                            buffer.clear()
                    position = newline + 1
            if oversize:
                state["oversize_bytes_skipped"] += len(buffer)
            elif buffer:
                line_number += 1
                visit(bytes(buffer), line_number)
    except OSError as error:
        state["status"] = "fail"
        state["parse_failures"].append(str(error))
        return state

    state["lines_seen"] = line_number
    state["event_counts"] = dict(counts)
    state["events"] = compact_events
    state["status"] = "fail" if state["parse_failures"] else "present"
    return state


def source_prefix(path: Path | None) -> tuple[list[str] | None, dict[str, Any]]:
    if path is None:
        return None, {"status": "pending", "reason": "source-prefix audit was not supplied"}
    try:
        value = read_small_json(path)
    except FileNotFoundError:
        return None, {"status": "pending", "path": str(path), "reason": "source-prefix audit absent"}
    except (OSError, ValueError, ReviewError) as error:
        return None, {"status": "fail", "path": str(path), "reason": str(error)}
    if not isinstance(value, Mapping):
        return None, {"status": "fail", "path": str(path), "reason": "audit is not an object"}
    ids = value.get("prefix_row_ids")
    schedule = value.get("schedule") if isinstance(value.get("schedule"), Mapping) else {}
    failures: list[str] = []
    if value.get("schema_version") != "sepalith.rl03.two-update.source-prefix-audit.v1":
        failures.append("source-prefix schema mismatch")
    if not isinstance(ids, list) or len(ids) < 8 or not all(isinstance(item, str) for item in ids):
        failures.append("source-prefix audit has fewer than eight string IDs")
        ids = []
    if schedule.get("sha256") != SOURCE_SCHEDULE_SHA256:
        failures.append("source schedule SHA mismatch")
    if schedule.get("sequence_sha256") != SOURCE_SEQUENCE_SHA256:
        failures.append("source draw sequence SHA mismatch")
    first_update = list(ids[:8])
    expected = [item for item in first_update for _ in range(EXPECTED_CANDIDATES)]
    return expected, {
        "status": "fail" if failures else "pass",
        "path": str(path),
        "audit_sha256": sha256_file(path),
        "prefix_ids_available": len(ids),
        "first_update_source_ids": first_update,
        "schedule_sha256": schedule.get("sha256"),
        "sequence_sha256": schedule.get("sequence_sha256"),
        "failures": failures,
    }


def review_bindings(
    run_root: Path,
    launch_path: Path | None,
    preflight_path: Path | None,
    expected_recipe_sha256: str | None,
    admission_path: Path | None,
) -> dict[str, Any]:
    failures: list[str] = []
    pending: list[str] = []
    launch: Any = None
    preflight: Any = None
    launch_command: list[str] = []
    if launch_path is None:
        pending.append("launch metadata path was not supplied")
    else:
        try:
            launch = read_small_json(launch_path)
            if isinstance(launch, list):
                launch_command = [str(item) for item in launch]
            elif isinstance(launch, Mapping) and isinstance(launch.get("command"), list):
                launch_command = [str(item) for item in launch["command"]]
            else:
                failures.append("launch metadata does not contain a command list")
        except (OSError, ValueError, ReviewError) as error:
            failures.append(f"launch metadata: {error}")
    recipe_arg = None
    for item in launch_command:
        if item.endswith(".recipe.json"):
            recipe_arg = Path(item)
            break
    if not any(SOURCE_SHA256 in item for item in launch_command):
        failures.append("launch command does not bind the frozen source snapshot")
    if not any("campaign_rl_launch.py" in item for item in launch_command):
        failures.append("launch command is not campaign_rl_launch.py")
    if not any(item == str((run_root / "supervision.json").resolve()) for item in launch_command):
        failures.append("launch command does not bind immutable run supervision.json")

    if preflight_path is None:
        pending.append("compact entry preflight path was not supplied")
    else:
        try:
            preflight = read_small_json(preflight_path)
        except FileNotFoundError:
            pending.append(f"compact entry preflight is not present yet: {preflight_path}")
        except (OSError, ValueError, ReviewError) as error:
            # The live entry also writes a large entry-preflight receipt.  A
            # caller must provide a compact copy; never parse that receipt.
            if "larger than" in str(error):
                pending.append(f"compact entry preflight is unavailable: {error}")
            else:
                failures.append(f"entry preflight: {error}")
    if isinstance(preflight, Mapping):
        if preflight.get("status") != "preflight_pass":
            failures.append(f"entry preflight status is {preflight.get('status')!r}")
        if preflight.get("framework_imported") is not False:
            failures.append("entry preflight imported a framework")
        geometry = preflight.get("geometry") if isinstance(preflight.get("geometry"), Mapping) else {}
        expected_geometry = {
            "candidate_count": EXPECTED_CANDIDATES,
            "rollout_rows_per_update": EXPECTED_ROWS_PER_UPDATE,
            "prompt_groups_per_update": EXPECTED_GROUPS_PER_UPDATE,
            "per_device_train_batch_size": 8,
            "generation_batch_size": EXPECTED_ROWS_PER_UPDATE,
            "gradient_accumulation_steps": 4,
            "steps_per_generation": 4,
            "num_iterations": 1,
        }
        for key, wanted in expected_geometry.items():
            if geometry.get(key) != wanted:
                failures.append(f"preflight geometry {key}={geometry.get(key)!r}, expected {wanted!r}")
        development = preflight.get("development") if isinstance(preflight.get("development"), Mapping) else {}
        parameters = development.get("parameters") if isinstance(development.get("parameters"), Mapping) else {}
        panel = development.get("development_panel") if isinstance(development.get("development_panel"), Mapping) else {}
        for actual, wanted, label in (
            (development.get("factory"), "campaign_eval:development_evaluator", "development factory"),
            (development.get("renderer_id"), EXPECTED_RENDERER, "development renderer"),
            (panel.get("sha256"), EXPECTED_DEV_SHA256, "development panel SHA"),
            (panel.get("rows"), EXPECTED_DEV_ROWS, "development panel rows"),
            (development.get("development_max_new_tokens"), EXPECTED_DEV_CAP, "development cap"),
            (parameters.get("max_sequence_tokens"), EXPECTED_DEV_CONTEXT, "development context"),
        ):
            if actual != wanted:
                failures.append(f"{label}={actual!r}, expected {wanted!r}")
        if not isinstance(development.get("development_case_ids"), list) or len(development["development_case_ids"]) != EXPECTED_DEV_ROWS:
            failures.append("development panel does not bind exactly 75 case IDs")

    admission_value: Mapping[str, Any] | None = None
    actual_recipe_sha256: str | None = None
    if admission_path is not None and admission_path.is_file():
        try:
            candidate = read_small_json(admission_path)
            if isinstance(candidate, Mapping):
                admission_value = candidate
                if expected_recipe_sha256 is None:
                    admitted_recipe = candidate.get("recipe")
                    if isinstance(admitted_recipe, Mapping) and isinstance(admitted_recipe.get("sha256"), str):
                        expected_recipe_sha256 = admitted_recipe["sha256"]
                    elif isinstance(candidate.get("recipe_sha256"), str):
                        expected_recipe_sha256 = candidate["recipe_sha256"]
        except (OSError, ValueError, ReviewError) as error:
            failures.append(f"admission receipt: {error}")
    if admission_value is not None:
        admitted_identity = admission_value.get("identity_sha256")
        if not isinstance(admitted_identity, str) or len(admitted_identity) != 64:
            failures.append("admission receipt does not bind a 64-character identity SHA")
        admitted_source = admission_value.get("source")
        snapshot_id = admitted_source.get("snapshot_id") if isinstance(admitted_source, Mapping) else None
        if not isinstance(snapshot_id, str) or snapshot_id not in " ".join(launch_command):
            failures.append("admission source snapshot is not bound by launch command")
        admitted_recipe = admission_value.get("recipe")
        admitted_recipe_path = admitted_recipe.get("path") if isinstance(admitted_recipe, Mapping) else None
        if recipe_arg is not None and admitted_recipe_path != str(recipe_arg):
            failures.append("admission recipe path differs from launch recipe path")
    if expected_recipe_sha256 is None:
        pending.append("recipe SHA was not supplied by the admission receipt/CLI")
    elif recipe_arg is None:
        failures.append("launch command has no recipe argument")
    elif not recipe_arg.is_file():
        pending.append(f"recipe argument is not present yet: {recipe_arg}")
    else:
        try:
            actual_recipe_sha256 = sha256_file(recipe_arg)
            if actual_recipe_sha256 != expected_recipe_sha256:
                failures.append("recipe SHA differs from admitted recipe")
        except OSError as error:
            failures.append(f"recipe hash: {error}")

    if admission_value is not None:
        admission_status = admission_value.get("status")
    else:
        admission_status = None
    if (
        preflight_path is None
        and admission_status == "accepted_memory_mitigation_and_launch_admission"
    ):
        # The lead admission receipt is intentionally compact; the live
        # entry's 145 MB preflight receipt is outside this verifier's budget.
        # Generation/gradient evidence below proves the first-update geometry;
        # the admission binds the recipe/source/identity for this attempt.
        pending = [
            reason for reason in pending
            if not reason.startswith("compact entry preflight path was not supplied")
        ]
    return {
        "status": "fail" if failures else ("pending" if pending else "pass"),
        "launch_path": str(launch_path) if launch_path else None,
        "preflight_path": str(preflight_path) if preflight_path else None,
        "launch_command": launch_command,
        "recipe_path": str(recipe_arg) if recipe_arg else None,
        "actual_recipe_sha256": actual_recipe_sha256,
        "expected_recipe_sha256": expected_recipe_sha256,
        "admission_path": str(admission_path) if admission_path else None,
        "admission_status": admission_status,
        "admission_identity_sha256": admission_value.get("identity_sha256") if admission_value else None,
        "admission_source_snapshot_id": (
            admission_value.get("source", {}).get("snapshot_id")
            if admission_value and isinstance(admission_value.get("source"), Mapping) else None
        ),
        "framework_imported": preflight.get("framework_imported") if isinstance(preflight, Mapping) else None,
        "failures": failures,
        "pending_reasons": pending,
    }


def review_generation(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "output" / "generation-records.jsonl"
    rows, state = read_jsonl_prefix(path, EXPECTED_ROWS_PER_UPDATE)
    failures = list(state.get("failures", []))
    compact: list[dict[str, Any]] = []
    terminal: Counter[str] = Counter()
    for index, record in enumerate(rows):
        geometry = record.get("group_geometry") if isinstance(record.get("group_geometry"), Mapping) else {}
        actual_geometry = (
            geometry.get("generation_call_index"),
            geometry.get("group_index"),
            geometry.get("group_row_index"),
        )
        expected_geometry = (
            (index // (EXPECTED_GROUPS_PER_CALL * EXPECTED_CANDIDATES)),
            (index // EXPECTED_CANDIDATES),
            (index % EXPECTED_CANDIDATES),
        )
        if record.get("schema_version") != GENERATION_SCHEMA:
            failures.append(f"row {index}: generation schema mismatch")
        update_meta = record.get("update") if isinstance(record.get("update"), Mapping) else {}
        if record.get("global_step") != 0 or update_meta.get("global_step") != 0:
            failures.append(f"row {index}: first-update global step is not zero")
        if actual_geometry != expected_geometry:
            failures.append(f"row {index}: P2 geometry {actual_geometry!r}, expected {expected_geometry!r}")
        for key, wanted in (
            ("candidate_count", EXPECTED_CANDIDATES),
            ("generation_groups_per_call", EXPECTED_GROUPS_PER_CALL),
            ("generation_call_count", EXPECTED_CALLS_PER_UPDATE),
            ("generation_group_count", EXPECTED_GROUPS_PER_UPDATE),
            ("generation_row_count", EXPECTED_ROWS_PER_UPDATE),
        ):
            if geometry.get(key) != wanted:
                failures.append(f"row {index}: geometry {key}={geometry.get(key)!r}, expected {wanted!r}")
        if record.get("source_schedule_sha256") != SOURCE_SCHEDULE_SHA256:
            failures.append(f"row {index}: source schedule SHA mismatch")
        generated = record.get("generated_ids")
        token_count = record.get("generated_token_count")
        if not isinstance(generated, list) or not generated or any(type(token) is not int for token in generated):
            failures.append(f"row {index}: generated_ids is empty or noninteger")
        elif token_count != len(generated):
            failures.append(f"row {index}: generated token count disagrees with generated_ids")
        if type(token_count) is not int or not 1 <= token_count <= EXPECTED_COMPLETION_CAP:
            failures.append(f"row {index}: generated token count {token_count!r} outside 1..{EXPECTED_COMPLETION_CAP}")
        output_sha = record.get("generated_ids_sha256")
        prompt_sha = record.get("prompt_ids_sha256")
        if not isinstance(output_sha, str) or len(output_sha) != 64:
            failures.append(f"row {index}: output token SHA missing")
        if not isinstance(prompt_sha, str) or len(prompt_sha) != 64:
            failures.append(f"row {index}: prompt SHA missing")
        reason = record.get("terminal_reason")
        terminal[str(reason)] += 1
        if reason not in {"eos", "length", "cap", "max_new_tokens"}:
            failures.append(f"row {index}: unsupported terminal reason {reason!r}")
        padded = record.get("padded_after_terminal")
        if type(padded) is not int or padded < 0:
            failures.append(f"row {index}: invalid padded_after_terminal")
        compact.append({
            "ordinal": index,
            "global_step": record.get("global_step"),
            "call": geometry.get("generation_call_index"),
            "group": geometry.get("group_index"),
            "row": geometry.get("group_row_index"),
            "prompt_sha256": prompt_sha,
            "output_sha256": output_sha,
            "generated_tokens": token_count,
            "padded_after_terminal": padded,
            "terminal_reason": reason,
        })
    complete = len(rows) == EXPECTED_ROWS_PER_UPDATE and state.get("status") == "prefix_complete"
    if len(rows) > EXPECTED_ROWS_PER_UPDATE:
        failures.append("generation prefix returned more than 32 rows")
    status = "fail" if failures else ("pass" if complete else "pending")
    return compact, {
        **state,
        "status": status,
        "rows": len(rows),
        "failures": failures,
        "terminal_reason_counts": dict(terminal),
        "contract": {
            "calls_per_update": EXPECTED_CALLS_PER_UPDATE,
            "groups_per_call": EXPECTED_GROUPS_PER_CALL,
            "logical_groups_per_update": EXPECTED_GROUPS_PER_UPDATE,
            "candidates_per_group": EXPECTED_CANDIDATES,
            "rows_per_update": EXPECTED_ROWS_PER_UPDATE,
            "completion_cap": EXPECTED_COMPLETION_CAP,
        },
    }


def review_rewards(
    root: Path,
    generation: list[dict[str, Any]],
    expected_source_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "output" / "reward-records.jsonl"
    rows, state = read_jsonl_prefix(path, EXPECTED_ROWS_PER_UPDATE)
    failures = list(state.get("failures", []))
    joined: list[dict[str, Any]] = []
    families: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    expected_operations: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    observed_ids: list[str] = []
    for index, reward in enumerate(rows):
        source_id = reward.get("id")
        output_sha = reward.get("output_ids_sha256")
        if not isinstance(source_id, str) or not source_id:
            failures.append(f"row {index}: reward source ID missing")
        observed_ids.append(source_id)
        if not isinstance(output_sha, str) or len(output_sha) != 64:
            failures.append(f"row {index}: reward output SHA missing")
        if index >= len(generation):
            continue
        generation_row = generation[index]
        if output_sha != generation_row.get("output_sha256"):
            failures.append(f"row {index}: ordered generation/reward output SHA mismatch")
        if reward.get("generated_tokens") != generation_row.get("generated_tokens"):
            failures.append(f"row {index}: reward/generation token count mismatch")
        if not finite(reward.get("reward")):
            failures.append(f"row {index}: reward is nonfinite")
        if type(reward.get("canonical_eos")) is not bool:
            failures.append(f"row {index}: canonical_eos is not boolean")
        if type(reward.get("cap_hit")) is not bool:
            failures.append(f"row {index}: cap_hit is not boolean")
        # A cap-hit/missing-EOS completion is a valid measured outcome of the
        # rollout and carries protocol_valid=false.  Review the boolean and
        # denominator; do not turn the observed zero-reward outcome into a
        # stream-integrity failure.
        if type(reward.get("protocol_valid")) is not bool:
            failures.append(f"row {index}: protocol_valid is not boolean")
        family = reward.get("family")
        if not isinstance(family, str) or not family:
            failures.append(f"row {index}: reward family missing")
        else:
            families[family] += 1
        operation = reward.get("operation")
        expected_operation = reward.get("expected_operation")
        operations[str(operation)] += 1
        expected_operations[str(expected_operation)] += 1
        failure_counts[str(reward.get("failure"))] += 1
        if reward.get("canonical_eos"):
            terminal_counts["canonical_eos"] += 1
        if reward.get("cap_hit"):
            terminal_counts["cap_hit"] += 1
        joined.append({
            "ordinal": index,
            "source_id": source_id,
            "prompt_sha256": generation_row.get("prompt_sha256"),
            "output_sha256": output_sha,
            "reward": reward.get("reward"),
            "family": family,
            "operation": operation,
            "expected_operation": expected_operation,
            "generated_tokens": reward.get("generated_tokens"),
            "canonical_eos": reward.get("canonical_eos"),
            "cap_hit": reward.get("cap_hit"),
            "failure": reward.get("failure"),
        })

    if len(rows) != len(generation):
        if len(rows) > len(generation):
            failures.append(f"reward rows {len(rows)} exceed generation rows {len(generation)}")
    if len(rows) == EXPECTED_ROWS_PER_UPDATE and expected_source_ids is not None:
        if observed_ids != expected_source_ids:
            first = next((i for i, pair in enumerate(zip(observed_ids, expected_source_ids)) if pair[0] != pair[1]), min(len(observed_ids), len(expected_source_ids)))
            failures.append(f"admitted source sequence differs at row {first}")
    elif expected_source_ids is None:
        failures.append("admitted source prefix is unavailable; exact source join is unproven")

    # Every logical source must occupy four consecutive candidates, and each
    # source's prompt hash must remain constant across its candidate group.
    if len(joined) >= EXPECTED_CANDIDATES:
        for start in range(0, len(joined), EXPECTED_CANDIDATES):
            block = joined[start:start + EXPECTED_CANDIDATES]
            if len(block) < EXPECTED_CANDIDATES:
                break
            if len({row["source_id"] for row in block}) != 1:
                failures.append(f"candidate group at row {start} does not share one source ID")
            if len({row["prompt_sha256"] for row in block}) != 1:
                failures.append(f"candidate group at row {start} does not share one prompt SHA")

    complete = (
        len(rows) == EXPECTED_ROWS_PER_UPDATE
        and len(generation) == EXPECTED_ROWS_PER_UPDATE
        and state.get("status") == "prefix_complete"
    )
    status = "fail" if failures else ("pass" if complete else "pending")
    return joined, {
        **state,
        "status": status,
        "rows": len(rows),
        "failures": failures,
        "join_key": "ordered row + generation prompt/output SHA + reward source ID",
        "source_ids": observed_ids,
        "family_counts": dict(families),
        "operation_counts": dict(operations),
        "expected_operation_counts": dict(expected_operations),
        "failure_counts": dict(failure_counts),
        "termination_counts": dict(terminal_counts),
        "protocol_valid_count": sum(row.get("protocol_valid") is True for row in rows),
        "protocol_invalid_count": sum(row.get("protocol_valid") is False for row in rows),
        "finite_rewards": all(finite(row.get("reward")) for row in rows),
    }


def review_gradient(root: Path) -> dict[str, Any]:
    path = root / "output" / "gradient-records.jsonl"
    rows, state = read_jsonl_prefix(path, 1)
    failures = list(state.get("failures", []))
    first = rows[0] if rows else None
    if first is not None:
        if first.get("schema_version") != GRADIENT_SCHEMA:
            failures.append("gradient schema mismatch")
        if first.get("step") != 0 or first.get("global_step") != 0:
            failures.append("first gradient record is not update zero")
        if first.get("finite") is not True or first.get("nonfinite") is not False:
            failures.append("gradient finite/nonfinite flags are invalid")
        if type(first.get("grad_present_count")) is not int or first["grad_present_count"] <= 0:
            failures.append("no present trainable gradients")
        if type(first.get("nonzero_tensor_count")) is not int or first["nonzero_tensor_count"] <= 0:
            failures.append("all trainable gradients are zero")
        if not finite(first.get("norm")) or first["norm"] <= 0:
            failures.append("gradient norm is not finite and nonzero")
        scope = first.get("trainable_scope")
        if not isinstance(scope, str) or "lora" not in scope.lower():
            failures.append("gradient scope does not identify LoRA adapters")
    complete = first is not None and state.get("status") == "prefix_complete"
    return {
        **state,
        "status": "fail" if failures else ("pass" if complete else "pending"),
        "failures": failures,
        "first": {
            key: first.get(key)
            for key in (
                "step", "global_step", "finite", "nonfinite", "norm",
                "grad_present_count", "nonzero_tensor_count", "trainable_tensor_count",
                "trainable_scope",
            )
        } if first else None,
    }


def review_host_guard(host_root: Path | None) -> dict[str, Any]:
    if host_root is None:
        return {"status": "pending", "reason": "host supervision path was not supplied"}
    terminal_path = host_root / "terminal.json"
    try:
        terminal = read_small_json(terminal_path)
    except FileNotFoundError:
        terminal = None
    except (OSError, ValueError, ReviewError) as error:
        return {"status": "fail", "path": str(terminal_path), "reason": str(error)}
    result: dict[str, Any] = {
        "path": str(host_root),
        "terminal_path": str(terminal_path),
        "terminal": terminal if isinstance(terminal, Mapping) else None,
        "status": "pending" if terminal is None else "present",
    }
    preflight_path = host_root / "preflight.json"
    try:
        preflight = read_small_json(preflight_path)
    except FileNotFoundError:
        preflight = None
    except (OSError, ValueError, ReviewError) as error:
        result["preflight_error"] = str(error)
        preflight = None
    if isinstance(preflight, Mapping):
        initial = preflight.get("initial") if isinstance(preflight.get("initial"), Mapping) else {}
        result["preflight"] = {
            "minimum_free_mib": preflight.get("minimum_free_mib"),
            "initial_available_mib": initial.get("AvailableMBytes"),
            "initial_page_reads_per_sec": initial.get("PageReadsPersec"),
            "initial_pages_input_per_sec": initial.get("PagesInputPersec"),
            "initial_pages_output_per_sec": initial.get("PagesOutputPersec"),
            "initial_driver_events": initial.get("DriverEvents"),
        }
    if isinstance(terminal, Mapping):
        terminal_status = terminal.get("status")
        if terminal_status == "completed":
            result["status"] = "pass"
        elif terminal_status in {"stopped_or_failed", "failed"}:
            result["status"] = "stopped"
            result["reason"] = terminal.get("reason")
            result["child_exit_code"] = terminal.get("child_exit_code")
        else:
            result["status"] = "fail"
            result["reason"] = f"unexpected host terminal status {terminal_status!r}"
    log_path = host_root / "process.log"
    if log_path.is_file():
        try:
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 4096))
                tail = stream.read(4096).decode("utf-8", errors="replace")
            result["process_log_tail"] = tail[-1200:]
        except OSError as error:
            result["process_log_tail_error"] = str(error)
    return result


def review(
    run_root: Path,
    *,
    source_prefix_path: Path | None,
    host_supervision: Path | None,
    launch_path: Path | None,
    preflight_path: Path | None,
    expected_recipe_sha256: str | None,
    admission_path: Path | None,
) -> dict[str, Any]:
    expected_sources, source_review = source_prefix(source_prefix_path)
    bindings = review_bindings(
        run_root, launch_path, preflight_path, expected_recipe_sha256, admission_path
    )
    generation, generation_review = review_generation(run_root)
    rewards, reward_review = review_rewards(run_root, generation, expected_sources)
    gradient_review = review_gradient(run_root)
    telemetry_review = scan_telemetry(run_root / "telemetry.jsonl")
    host_review = review_host_guard(host_supervision)

    core = {
        "generation": generation_review,
        "rewards": reward_review,
        "gradient": gradient_review,
        "telemetry": telemetry_review,
    }
    first_optimizer_events = [
        event for event in telemetry_review.get("events", [])
        if event.get("event") == "optimizer_step" and event.get("step") == 1
    ]
    first_metric_events = [
        event for event in telemetry_review.get("events", [])
        if event.get("event") == "trainer_metrics" and event.get("step") == 1
    ]
    telemetry_failures: list[str] = []
    if not first_optimizer_events:
        telemetry_failures.append("first optimizer_step telemetry event is absent")
    if not first_metric_events:
        telemetry_failures.append("first trainer_metrics telemetry event is absent")
    for event in first_metric_events[:1]:
        metrics = event.get("metrics") if isinstance(event.get("metrics"), Mapping) else {}
        if not finite(metrics.get("grad_norm")) or metrics["grad_norm"] <= 0:
            telemetry_failures.append("first trainer_metrics grad_norm is not finite/nonzero")
        if not finite(metrics.get("reward")):
            telemetry_failures.append("first trainer_metrics reward is nonfinite")
    if telemetry_failures:
        telemetry_review = dict(telemetry_review)
        telemetry_review["first_update_failures"] = telemetry_failures
        telemetry_review["status"] = "fail" if telemetry_review.get("status") == "fail" else "pending"
        core["telemetry"] = telemetry_review
    elif telemetry_review.get("status") == "present":
        telemetry_review = dict(telemetry_review)
        telemetry_review["status"] = "pass"
        core["telemetry"] = telemetry_review

    core_pass = all(core[name].get("status") == "pass" for name in ("generation", "rewards", "gradient", "telemetry"))
    any_core_fail = any(core[name].get("status") == "fail" for name in core)
    host_stopped = host_review.get("status") == "stopped"
    if core_pass and bindings.get("status") == "pass":
        status = "pass"
    elif host_stopped and not core_pass:
        status = "failed"
    elif any_core_fail or bindings.get("status") == "fail":
        status = "fail"
    else:
        status = "pending"

    return {
        "schema_version": "sepalith.rl05.first-update-independent-review.v1",
        "status": status,
        "scope": "first optimizer update only; no later-step, checkpoint, or quality claim",
        "run_root": str(run_root),
        "attempt": run_root.name,
        "source": {
            "frozen_source_sha256": SOURCE_SHA256,
            "prefix_audit": source_review,
            "expected_first_update_rows": EXPECTED_ROWS_PER_UPDATE,
        },
        "bindings": bindings,
        "host_guard": host_review,
        "first_update": {
            "status": "pass" if core_pass else ("fail" if any_core_fail else "pending"),
            "rows": EXPECTED_ROWS_PER_UPDATE,
            "calls": EXPECTED_CALLS_PER_UPDATE,
            "logical_groups": EXPECTED_GROUPS_PER_UPDATE,
            "candidates_per_group": EXPECTED_CANDIDATES,
            "gradient": "one finite nonzero LoRA gradient record required",
            "telemetry_optimizer_step": 1 if first_optimizer_events else None,
            "telemetry_metrics_step": 1 if first_metric_events else None,
        },
        "generation": generation_review,
        "rewards": reward_review,
        "gradient": gradient_review,
        "telemetry": telemetry_review,
        "ordered_join": rewards,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-prefix-audit", type=Path, required=True)
    parser.add_argument("--host-supervision", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--expected-recipe-sha256")
    parser.add_argument("--admission-receipt", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = review(
        args.run_root.resolve(),
        source_prefix_path=args.source_prefix_audit.resolve(),
        host_supervision=args.host_supervision.resolve(),
        launch_path=args.launch.resolve(),
        preflight_path=args.preflight.resolve() if args.preflight else None,
        expected_recipe_sha256=args.expected_recipe_sha256,
        admission_path=args.admission_receipt.resolve() if args.admission_receipt else None,
    )
    result["verifier"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
        "framework_free": True,
        "max_jsonl_line_bytes": MAX_JSONL_LINE_BYTES,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    printable = {
        "status": result["status"],
        "run_root": result["run_root"],
        "generation": result["generation"].get("status"),
        "rewards": result["rewards"].get("status"),
        "gradient": result["gradient"].get("status"),
        "telemetry": result["telemetry"].get("status"),
        "host_guard": result["host_guard"].get("status"),
        "receipt": str(args.json_out.resolve()),
    }
    print(json.dumps(printable, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
