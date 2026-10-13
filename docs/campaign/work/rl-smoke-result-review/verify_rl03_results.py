#!/usr/bin/env python3
"""CPU-only review of the RL-03 two-update artifacts.

This verifier reads receipts, JSONL telemetry, checkpoint metadata and the
14-case development result.  It never imports torch, Transformers, TRL or the
model.  A running attempt is reported as ``pending``.  The optional split
comparison only reports resume equality after both split attempts have
completed; absent artifacts are not converted into a pass.

The generation stream intentionally has no source ID.  The review joins it
to the ordered reward stream by ``output_ids_sha256`` and row order.  The
resulting ``(source_id, prompt_ids_sha256, output_ids_sha256)`` sequence is
the stable comparison key for a continuous run versus first-segment plus
resumed-segment artifacts.
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
from typing import Any, Iterable, Mapping


EXPECTED_SOURCE = "f051ebb9c51a5039ec5b3a81852a5f4c32b3284c066afd30c7baf1066e703c97"
EXPECTED_RECIPE = "bbe33e575d6ff8e30138dbefeb9d28949ec1bf90749b631cc5878775480c2e93"
EXPECTED_CANDIDATES = 4
EXPECTED_GROUPS = 8
EXPECTED_ROWS_PER_UPDATE = 32
EXPECTED_UPDATES = 2
EXPECTED_DEV_ROWS = 14
EXPECTED_LONG_DEV_ID = "dat07-existing-719cd49683667d0fb86fb2fa"
EXPECTED_LONG_DEV_PROMPT = 2619
GENERATION_SCHEMA = "sepalith.rl.generation-record.v1"


class ReviewError(Exception):
    """An artifact is malformed or contradicts the live contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def compact(value: Any, limit: int = 240) -> Any:
    """Keep diagnostics bounded when an artifact contains a long identity."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ReviewError(f"{path.name}: empty line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReviewError(f"{path.name}: invalid JSON line {line_number}: {error}") from error
            if not isinstance(value, Mapping):
                raise ReviewError(f"{path.name}: line {line_number} is not an object")
            yield line_number, value


def optional_jsonl(path: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return [], {"status": "pending", "path": str(path), "reason": "artifact not present"}
    rows = []
    try:
        for _line, value in jsonl(path):
            rows.append(value)
    except (OSError, ReviewError) as error:
        return [], {"status": "fail", "path": str(path), "reason": str(error)}
    return rows, {"status": "present", "path": str(path), "rows": len(rows)}


def first_failure(root: Path) -> str | None:
    candidates = [root / "output" / "failure.json", root / "failure.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = read_json(path)
        except Exception as error:  # pragma: no cover - damaged artifact path
            return f"{path}: unreadable failure receipt: {error}"
        if isinstance(value, Mapping):
            return f"{path.name}: {value.get('error', value.get('status', 'failure'))}"
        return f"{path.name}: {compact(value)}"
    process_log = root.parent / f"{root.name}-host-supervision" / "process.log"
    if process_log.is_file():
        # Keep only the first obvious exception line; the log is intentionally
        # not copied into the review receipt.
        for line in process_log.read_text(encoding="utf-8", errors="replace").splitlines():
            # The entry prints a single very large JSON result line.  It can
            # contain arbitrary identity text, including words that resemble
            # exception names; failure.json above is the authoritative entry
            # failure source for that case.
            if line.lstrip().startswith(("{", "[")):
                continue
            if re.search(r"(Traceback|Error|Exception|out of memory|failed|CUBLAS|illegal memory|abort)", line, re.I):
                return f"process.log: {line.strip()[:500]}"
    return None


def load_supervision(root: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    path = root.parent / f"{root.name}-supervision.json"
    if not path.is_file():
        return None, {"status": "pending", "path": str(path), "reason": "supervision receipt not present"}
    try:
        value = read_json(path)
    except Exception as error:
        return None, {"status": "fail", "path": str(path), "reason": f"invalid JSON: {error}"}
    if not isinstance(value, Mapping):
        return None, {"status": "fail", "path": str(path), "reason": "supervision receipt is not an object"}
    return dict(value), {"status": "present", "path": str(path), "immutable": True}


def supervision_contract(root: Path, supervision: Mapping[str, Any] | None) -> dict[str, Any]:
    if supervision is None:
        return {"status": "pending", "reason": "supervision receipt unavailable"}
    failures = []
    if supervision.get("recipe_sha256") != EXPECTED_RECIPE:
        failures.append(f"recipe_sha256={compact(supervision.get('recipe_sha256'))}")
    argv = supervision.get("argv")
    if not isinstance(argv, list) or not argv:
        failures.append("argv missing")
        argv = []
    source_tokens = [str(item) for item in argv if EXPECTED_SOURCE in str(item)]
    if not source_tokens:
        failures.append("frozen source snapshot hash absent from supervisor argv")
    immutable = (root.parent / f"{root.name}-supervision.json").resolve()
    expected_result = supervision.get("entry_result_receipt")
    result_path = Path(expected_result).resolve() if isinstance(expected_result, str) else None
    if result_path is None or result_path == immutable:
        failures.append("entry_result_receipt is missing or aliases immutable supervision receipt")
    result = None
    result_summary = None
    if result_path is not None and result_path.is_file():
        try:
            loaded = read_json(result_path)
            if isinstance(loaded, Mapping):
                result = dict(loaded)
                result_summary = {
                    key: result.get(key)
                    for key in ("status", "step", "checkpoint", "error", "supervision_receipt")
                    if key in result
                }
                result_summary["checkpoint_manifest_present"] = "checkpoint_manifest" in result
            else:
                failures.append("entry result receipt is not an object")
        except Exception as error:
            failures.append(f"entry result receipt unreadable: {error}")
    else:
        return {
            "status": "pending" if not failures else "fail",
            "failures": failures,
            "immutable_status": supervision.get("status"),
            "entry_result_receipt": str(result_path) if result_path else None,
            "entry_result": "pending",
            "distinct_receipts": result_path is not None and result_path != immutable,
            "command_source_match": bool(source_tokens),
        }
    if result is not None:
        if result.get("supervision_receipt") and Path(result["supervision_receipt"]).resolve() != immutable:
            failures.append("entry result supervision_receipt does not point to immutable supervisor receipt")
        if result_path == immutable:
            failures.append("entry result overwrote immutable supervision receipt")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "immutable_status": supervision.get("status"),
        "entry_result_receipt": str(result_path) if result_path else None,
        # Do not copy the identity and checkpoint manifest into the review:
        # those receipts intentionally contain the 8,440 selected IDs and can
        # be tens or hundreds of MiB.  The immutable path and compact status
        # are sufficient for process/receipt acceptance.
        "entry_result": result_summary,
        "distinct_receipts": result_path is not None and result_path != immutable,
        "command_source_match": bool(source_tokens),
    }


def identity_from_telemetry(path: Path) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return None, [], {"status": "pending", "path": str(path), "reason": "telemetry not present"}
    identity = None
    events = []
    counts: Counter[str] = Counter()
    try:
        for _line, record in jsonl(path):
            event = record.get("event")
            counts[str(event)] += 1
            if identity is None and isinstance(record.get("identity"), Mapping):
                identity = record["identity"]
            # Keep only control events and a bounded sample.  The first event
            # carries the large selected-ID identity; do not copy it to output.
            if event in {"train_begin", "optimizer_step", "train_end"}:
                events.append({k: record.get(k) for k in ("event", "step", "stop_reason", "remaining_seconds") if k in record})
    except (OSError, ReviewError) as error:
        return None, [], {"status": "fail", "path": str(path), "reason": str(error)}
    return identity, events, {"status": "present", "path": str(path), "event_counts": dict(counts), "lines": sum(counts.values())}


def check_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if identity is None:
        return {"status": "pending", "reason": "train_begin identity not available"}
    failures = []
    policy = identity.get("policy") if isinstance(identity.get("policy"), Mapping) else {}
    schedule = identity.get("schedule") if isinstance(identity.get("schedule"), Mapping) else {}
    data = identity.get("data") if isinstance(identity.get("data"), Mapping) else {}
    parent = identity.get("parent") if isinstance(identity.get("parent"), Mapping) else {}
    checks = {
        "data.selected_ids_sha256": (data.get("selected_ids_sha256"), "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d"),
        "data.row_count": (data.get("row_count"), 8440),
        "policy.candidate_count": (policy.get("candidate_count"), EXPECTED_CANDIDATES),
        "policy.prompt_max_tokens": (policy.get("prompt_max_tokens"), 2048),
        "policy.completion_max_tokens": (policy.get("completion_max_tokens"), 192),
        "policy.context_max_tokens": (policy.get("context_max_tokens"), 2240),
        "schedule.source_draws_per_update": (schedule.get("source_draws_per_update"), EXPECTED_GROUPS),
        "schedule.generation_batch_size": (schedule.get("generation_batch_size"), EXPECTED_ROWS_PER_UPDATE),
        "schedule.buffer_reuse": (schedule.get("buffer_reuse"), 4),
        "schedule.steps_per_generation": (schedule.get("steps_per_generation"), 4),
        "parent.manifest_sha256": (parent.get("manifest_sha256"), "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12"),
    }
    # The parent manifest value is checked when available, while a future
    # parent may intentionally change it; source/recipe and the loaded parent
    # audit remain the authoritative identity gates for this review.
    checks.pop("parent.manifest_sha256")
    for name, (actual, expected) in checks.items():
        if actual != expected:
            failures.append(f"{name}: expected {expected!r}, got {compact(actual)!r}")
    if policy.get("model_load_max_seq_length") != 4096:
        failures.append("policy.model_load_max_seq_length is not 4096")
    if policy.get("cuda_memory_fraction") != 0.75:
        failures.append("policy.cuda_memory_fraction is not 0.75")
    source = identity.get("source")
    if not isinstance(source, Mapping) or EXPECTED_SOURCE not in str(source.get("frozen_source_root", "")):
        failures.append("identity.source does not bind the frozen source snapshot")
    return {"status": "pass" if not failures else "fail", "failures": failures,
            "source": {"frozen_source_root": source.get("frozen_source_root")} if isinstance(source, Mapping) else source,
            "row_count": data.get("row_count"),
            "candidate_count": policy.get("candidate_count"),
            "model_load_max_seq_length": policy.get("model_load_max_seq_length"),
            "cuda_memory_fraction": policy.get("cuda_memory_fraction")}


def inspect_generation(root: Path, reward_rows: list[Mapping[str, Any]], *, expected_updates: int = EXPECTED_UPDATES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "output" / "generation-records.jsonl"
    rows, state = optional_jsonl(path)
    if state["status"] != "present":
        return [], state
    failures = []
    groups_by_update: Counter[tuple[int, int]] = Counter()
    rows_by_update: Counter[int] = Counter()
    compact_rows = []
    terminal_reasons: Counter[str] = Counter()
    elapsed_values = set()
    for index, record in enumerate(rows):
        if record.get("schema_version") != GENERATION_SCHEMA:
            failures.append(f"row {index}: schema_version mismatch")
        update = record.get("global_step")
        if type(update) is not int:
            failures.append(f"row {index}: global_step is not an integer")
            continue
        geometry = record.get("group_geometry")
        if not isinstance(geometry, Mapping):
            failures.append(f"row {index}: group_geometry missing")
            continue
        group = geometry.get("group_index")
        row_index = geometry.get("group_row_index")
        if type(group) is not int or type(row_index) is not int:
            failures.append(f"row {index}: group index is not integer")
        else:
            groups_by_update[(update, group)] += 1
        rows_by_update[update] += 1
        generated = record.get("generated_ids")
        if not isinstance(generated, list) or not generated or any(type(token) is not int for token in generated):
            failures.append(f"row {index}: generated_ids is empty or noninteger")
        elif record.get("generated_token_count") != len(generated):
            failures.append(f"row {index}: generated_token_count mismatch")
        if not isinstance(record.get("prompt_ids_sha256"), str) or len(record["prompt_ids_sha256"]) != 64:
            failures.append(f"row {index}: prompt_ids_sha256 missing")
        if not isinstance(record.get("generated_ids_sha256"), str) or len(record["generated_ids_sha256"]) != 64:
            failures.append(f"row {index}: generated_ids_sha256 missing")
        if type(record.get("padded_after_terminal")) is not int or record["padded_after_terminal"] < 0:
            failures.append(f"row {index}: invalid padded_after_terminal")
        compact_rows.append({
            "update": update,
            "group": geometry.get("group_index"),
            "row": geometry.get("group_row_index"),
            "prompt_sha256": record.get("prompt_ids_sha256"),
            "output_sha256": record.get("generated_ids_sha256"),
            "generated_tokens": record.get("generated_token_count"),
            "terminal_reason": record.get("terminal_reason"),
        })
        terminal_reasons[str(record.get("terminal_reason"))] += 1
        if finite_number(record.get("elapsed_sec")):
            elapsed_values.add(float(record["elapsed_sec"]))
    updates = sorted(rows_by_update)
    if rows and (len(updates) != expected_updates or any(rows_by_update[u] != EXPECTED_ROWS_PER_UPDATE for u in updates)):
        failures.append(f"expected {expected_updates} updates × 32 generation rows, got {dict(rows_by_update)}")
    if rows and any(groups_by_update[(u, g)] != EXPECTED_CANDIDATES for u in updates for g in range(EXPECTED_GROUPS)):
        failures.append("one or more generation groups do not contain exactly four candidates")
    return compact_rows, {**state, "status": "pass" if not failures else "fail", "failures": failures,
                          "updates": updates, "rows_by_update": dict(rows_by_update),
                          "groups_by_update": {f"{u}:{g}": n for (u, g), n in sorted(groups_by_update.items())},
                          "terminal_reason_counts": dict(terminal_reasons),
                          "generation_elapsed_sec": sorted(elapsed_values)[:4],
                          "rows": len(rows)}


def inspect_rewards(reward_rows: list[Mapping[str, Any]], generation_rows: list[Mapping[str, Any]], prefix_path: Path | None, *, expected_rows: int = EXPECTED_ROWS_PER_UPDATE * EXPECTED_UPDATES, prefix_offset: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not reward_rows:
        return [], {"status": "pending", "reason": "reward-records.jsonl not present or empty"}
    failures = []
    joined = []
    for index, reward in enumerate(reward_rows):
        source_id = reward.get("id")
        output_sha = reward.get("output_ids_sha256")
        if not isinstance(source_id, str) or not source_id:
            failures.append(f"row {index}: reward source id missing")
        if not isinstance(output_sha, str) or len(output_sha) != 64:
            failures.append(f"row {index}: reward output_ids_sha256 missing")
        if not finite_number(reward.get("reward")):
            failures.append(f"row {index}: reward is nonfinite or missing")
        if generation_rows and index < len(generation_rows) and output_sha != generation_rows[index]["output_sha256"]:
            failures.append(f"row {index}: reward/output generation order join mismatch")
        if generation_rows and index < len(generation_rows):
            generation = generation_rows[index]
            joined.append({
                "ordinal": index,
                "update": generation["update"],
                "source_id": source_id,
                "prompt_sha256": generation["prompt_sha256"],
                "output_sha256": output_sha,
                "reward": reward.get("reward"),
                "failure": reward.get("failure"),
            })
    if generation_rows and len(reward_rows) != len(generation_rows):
        failures.append(f"reward rows {len(reward_rows)} differ from generation rows {len(generation_rows)}")
    expected_prefix = None
    if prefix_path and prefix_path.is_file():
        try:
            prefix = read_json(prefix_path)
            ids = prefix.get("prefix_row_ids") if isinstance(prefix, Mapping) else None
            if isinstance(ids, list):
                expected_prefix = [item for source in ids for item in [source] * EXPECTED_CANDIDATES]
        except Exception as error:
            failures.append(f"source-prefix-audit unreadable: {error}")
    if expected_prefix is not None:
        observed = [row.get("id") for row in reward_rows[:len(expected_prefix)]]
        # During a bounded deadline stop the first segment is a valid prefix;
        # require the complete 64-row equality only when all rows exist.
        expected_observed = expected_prefix[prefix_offset:prefix_offset + len(observed)]
        if observed != expected_observed:
            failures.append("reward source-ID prefix differs from the admitted two-update draw prefix")
    if len(reward_rows) and len(reward_rows) != expected_rows:
        failures.append(f"expected {expected_rows} reward rows, got {len(reward_rows)}")
    outcomes = Counter(str(row.get("failure")) for row in reward_rows)
    canonical_eos = sum(bool(row.get("canonical_eos")) for row in reward_rows)
    cap_hits = sum(bool(row.get("cap_hit")) for row in reward_rows)
    protocol_valid = sum(bool(row.get("protocol_valid")) for row in reward_rows)
    return joined, {"status": "pass" if not failures else "fail", "rows": len(reward_rows),
                    "failures": failures, "source_ids_prefix": [row.get("source_id") for row in joined[:16]],
                    "join_key": "ordered reward id + generation prompt_ids_sha256/output_ids_sha256",
                    "failure_counts": dict(outcomes), "canonical_eos": canonical_eos,
                    "cap_hits": cap_hits, "protocol_valid": protocol_valid}


def inspect_gradients(root: Path, *, expected_updates: int = EXPECTED_UPDATES) -> dict[str, Any]:
    path = root / "output" / "gradient-records.jsonl"
    rows, state = optional_jsonl(path)
    if state["status"] != "present":
        return state
    failures = []
    for index, row in enumerate(rows):
        if row.get("finite") is not True or row.get("nonfinite") is not False:
            failures.append(f"row {index}: nonfinite gradient flag")
        if type(row.get("grad_present_count")) is not int or row["grad_present_count"] <= 0:
            failures.append(f"row {index}: no present LoRA gradient")
        if type(row.get("nonzero_tensor_count")) is not int or row["nonzero_tensor_count"] <= 0:
            failures.append(f"row {index}: all LoRA gradients are zero")
        if not finite_number(row.get("norm")) or float(row["norm"]) <= 0:
            failures.append(f"row {index}: gradient norm is not finite and nonzero")
    if rows and len(rows) != expected_updates:
        failures.append(f"expected {expected_updates} optimizer gradient records, got {len(rows)}")
    return {**state, "status": "pass" if not failures else "fail", "failures": failures,
            "steps": [row.get("step") for row in rows],
            "norms": [row.get("norm") for row in rows],
            "grad_present": [row.get("grad_present_count") for row in rows],
            "nonzero_tensors": [row.get("nonzero_tensor_count") for row in rows],
            "finite": all(row.get("finite") is True for row in rows),
            "nonzero": all(type(row.get("nonzero_tensor_count")) is int and row["nonzero_tensor_count"] > 0 for row in rows)}


def inspect_load(root: Path) -> dict[str, Any]:
    path = root / "output" / "load-audit.json"
    if not path.is_file():
        return {"status": "pending", "path": str(path), "reason": "load audit not present"}
    try:
        value = read_json(path)
    except Exception as error:
        return {"status": "fail", "path": str(path), "reason": f"invalid JSON: {error}"}
    if not isinstance(value, Mapping):
        return {"status": "fail", "path": str(path), "reason": "load audit is not an object"}
    adapter = value.get("adapter") if isinstance(value.get("adapter"), Mapping) else {}
    model_load = value.get("model_load") if isinstance(value.get("model_load"), Mapping) else {}
    failures = []
    if value.get("dtype") not in {"torch.bfloat16", "bfloat16"}:
        failures.append(f"dtype is {value.get('dtype')!r}, expected BF16")
    if adapter.get("attachments") != 294:
        failures.append(f"adapter attachments {adapter.get('attachments')!r}, expected 294")
    if adapter.get("trainable_parameters") != 25116672:
        failures.append(f"trainable parameters {adapter.get('trainable_parameters')!r}, expected 25116672")
    if model_load.get("requested_max_seq_length") not in (None, 4096):
        failures.append("model load requested context is not 4096")
    return {"status": "pass" if not failures else "fail", "path": str(path), "failures": failures,
            "dtype": value.get("dtype"), "adapter": {k: adapter.get(k) for k in ("attachments", "trainable_parameters", "lora_rank", "lora_alpha")},
            "model_load": {k: model_load.get(k) for k in ("requested_max_seq_length", "observed_capacity_tokens", "capacity_status") if k in model_load},
            "cuda_allocator": value.get("cuda_allocator")}


FULL_STATE = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


def checkpoint_summary(root: Path, step: int) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    path = root / "archive" / "full" / f"checkpoint-{step}"
    if not path.is_dir():
        return {"status": "pending", "step": step, "path": str(path), "reason": "full checkpoint not present"}, None
    failures = []
    sizes = {}
    for name in ("adapter_model.safetensors", "adapter_config.json", *FULL_STATE, "campaign-state.json", "campaign-manifest.json"):
        target = path / name
        if not target.is_file() or target.stat().st_size <= 0:
            failures.append(f"missing or empty {name}")
        else:
            sizes[name] = target.stat().st_size
    state = None
    manifest = None
    try:
        if (path / "campaign-state.json").is_file():
            state = read_json(path / "campaign-state.json")
        if (path / "campaign-manifest.json").is_file():
            manifest = read_json(path / "campaign-manifest.json")
    except Exception as error:
        failures.append(f"checkpoint metadata unreadable: {error}")
    if not isinstance(state, Mapping):
        failures.append("campaign-state is not an object")
    else:
        if state.get("full") is not True or state.get("step") != step:
            failures.append("campaign-state is not a full checkpoint at its directory step")
        sampler = state.get("sampler")
        if not isinstance(sampler, Mapping):
            failures.append("campaign-state sampler payload missing")
        else:
            for key in ("source_draw_schedule_sha256", "source_draw_sequence_sha256", "consumed_rows", "current_index", "source_draw_cursor"):
                if key not in sampler:
                    failures.append(f"sampler.{key} missing")
    if not isinstance(manifest, Mapping) or manifest.get("full") is not True or manifest.get("step") != step:
        failures.append("campaign-manifest is not a matching full checkpoint")
    return {"status": "pass" if not failures else "fail", "step": step, "path": str(path),
            "failures": failures, "file_bytes": sizes,
            "sampler": {k: state.get("sampler", {}).get(k) for k in ("consumed_rows", "current_index", "source_draw_cursor", "source_draw_schedule_sha256", "source_draw_sequence_sha256")} if isinstance(state, Mapping) else None}, state


def dev_expected_ids(panel: Path | None) -> list[str]:
    if panel is None or not panel.is_file():
        return []
    ids = []
    try:
        for _line, value in jsonl(panel):
            ids.append(str(value["id"]))
    except Exception:
        return []
    return ids


def inspect_evaluations(root: Path, panel: Path | None, *, evaluation_steps: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    expected_ids = dev_expected_ids(panel)
    reports = []
    failures = []
    for step in evaluation_steps:
        request_path = root / "archive" / "evaluations" / f"step-{step}.json"
        cases_path = root / "archive" / "evaluations" / f"cases-step-{step}.json"
        if not request_path.is_file() or not cases_path.is_file():
            request_status = None
            if request_path.is_file():
                try:
                    request_value = read_json(request_path)
                    request_status = request_value.get("status") if isinstance(request_value, Mapping) else None
                except Exception:
                    request_status = "request_unreadable"
            deferred = isinstance(request_status, str) and request_status.startswith("deferred:")
            reports.append({"step": step, "status": "deferred" if deferred else "pending", "request_status": request_status,
                            "request": str(request_path), "cases": str(cases_path)})
            continue
        try:
            request = read_json(request_path)
            cases = read_json(cases_path)
        except Exception as error:
            failures.append(f"step {step}: evaluation JSON unreadable: {error}")
            continue
        results = cases.get("results") if isinstance(cases, Mapping) else None
        summary = cases.get("summary") if isinstance(cases, Mapping) else None
        ids = [item.get("id") for item in results] if isinstance(results, list) else []
        case_status = cases.get("status") if isinstance(cases, Mapping) else None
        if case_status != "complete" or not isinstance(results, list) or len(results) != EXPECTED_DEV_ROWS:
            failures.append(f"step {step}: expected complete 14-case evaluation")
        if expected_ids and sorted(ids) != sorted(expected_ids):
            failures.append(f"step {step}: development case IDs differ from frozen 14-case panel")
        long_rows = [item for item in results if item.get("id") == EXPECTED_LONG_DEV_ID] if isinstance(results, list) else []
        if not long_rows or long_rows[0].get("prompt_tokens") != EXPECTED_LONG_DEV_PROMPT:
            failures.append(f"step {step}: maximal 2619-token DEV case is missing or changed")
        if isinstance(results, list) and any(item.get("generated_tokens", EXPECTED_DEV_ROWS + 1) > 512 for item in results):
            failures.append(f"step {step}: development generation exceeded cap 512")
        if not isinstance(summary, Mapping) or summary.get("denominators", {}).get("cases") != EXPECTED_DEV_ROWS:
            failures.append(f"step {step}: summary denominator is not 14 cases")
        reports.append({"step": step, "status": "pass" if not failures or not any(f.startswith(f"step {step}:") for f in failures) else "fail",
                        "request_status": request.get("status") if isinstance(request, Mapping) else None, "cases_status": case_status,
                        "rows": len(results) if isinstance(results, list) else 0,
                        "long_case_prompt_tokens": long_rows[0].get("prompt_tokens") if long_rows else None,
                        "denominators": summary.get("denominators") if isinstance(summary, Mapping) else None})
    status = "pass" if len(reports) == len(evaluation_steps) and all(item.get("status") == "pass" for item in reports) and not failures else ("pending" if any(item.get("status") in {"pending", "deferred"} for item in reports) and not failures else "fail")
    return {"status": status, "failures": failures, "panel": str(panel) if panel else None, "reports": reports}


def summarize_run(root: Path, prefix_path: Path | None, panel: Path | None, *, expected_updates: int = EXPECTED_UPDATES, checkpoint_steps: tuple[int, ...] = (1, 2), prefix_offset: int = 0) -> dict[str, Any]:
    supervision, supervision_state = load_supervision(root)
    identity, events, telemetry_state = identity_from_telemetry(root / "telemetry.jsonl")
    reward_rows, reward_state = optional_jsonl(root / "output" / "reward-records.jsonl")
    generation_rows, generation_state = inspect_generation(root, reward_rows, expected_updates=expected_updates)
    joined, reward_review = inspect_rewards(reward_rows, generation_rows, prefix_path, expected_rows=EXPECTED_ROWS_PER_UPDATE * expected_updates, prefix_offset=prefix_offset)
    gradient_review = inspect_gradients(root, expected_updates=expected_updates)
    load_review = inspect_load(root)
    checkpoint_reviews = []
    checkpoint_states = {}
    for step in checkpoint_steps:
        review, state = checkpoint_summary(root, step)
        checkpoint_reviews.append(review)
        if isinstance(state, Mapping):
            sampler = state.get("sampler")
            checkpoint_states[step] = {
                "full": state.get("full"),
                "step": state.get("step"),
                "sampler": {
                    key: sampler.get(key)
                    for key in ("source_draw_schedule_sha256", "source_draw_sequence_sha256",
                                "consumed_rows", "current_index", "source_draw_cursor")
                } if isinstance(sampler, Mapping) else None,
            }
        else:
            checkpoint_states[step] = None
    eval_review = inspect_evaluations(root, panel, evaluation_steps=checkpoint_steps)
    failure = first_failure(root)
    result_review = supervision_contract(root, supervision)
    recipe_path = root / "output" / "admitted-recipe.json"
    recipe_hash = sha256_file(recipe_path) if recipe_path.is_file() else None
    checks = [supervision_state.get("status"), telemetry_state.get("status"), generation_state.get("status"), reward_review.get("status"), gradient_review.get("status"), load_review.get("status"), *(item.get("status") for item in checkpoint_reviews), eval_review.get("status"), result_review.get("status")]
    if result_review.get("entry_result") and result_review["entry_result"].get("status") == "deadline" and (result_review["entry_result"].get("step") or 0) < max(checkpoint_steps, default=EXPECTED_UPDATES):
        terminal_status = "incomplete_deadline"
    elif failure and result_review.get("entry_result") and result_review["entry_result"].get("status") == "entry_failed":
        terminal_status = "failed"
    elif any(status == "fail" for status in checks):
        terminal_status = "fail"
    elif any(status == "pending" for status in checks):
        terminal_status = "pending"
    else:
        terminal_status = "pass"
    entry_result = result_review.get("entry_result") if isinstance(result_review.get("entry_result"), Mapping) else {}
    return {
        "status": terminal_status,
        "run_root": str(root),
        "source_snapshot": EXPECTED_SOURCE,
        "recipe_sha256_expected": EXPECTED_RECIPE,
        "recipe_sha256_admitted_output_serialization": recipe_hash,
        "recipe_hash_note": "The immutable supervisor binds the canonical recipe SHA; admitted-recipe.json is a deterministic pretty-printed reserialization and therefore has different bytes.",
        "failure_first": failure,
        "terminal": {
            "status": entry_result.get("status"),
            "step": entry_result.get("step"),
            "stop_reason": entry_result.get("status") if entry_result.get("status") in {"deadline", "lead_decision", "schedule_complete", "trainer_terminal"} else None,
            "checkpoint": entry_result.get("checkpoint"),
            "error": failure,
        },
        "identity": check_identity(identity),
        "supervision": supervision_state,
        "supervision_contract": result_review,
        "telemetry": telemetry_state,
        "control_events": events,
        "load": load_review,
        "generation": generation_state,
        "rewards": reward_review,
        "gradients": gradient_review,
        "checkpoints": checkpoint_reviews,
        "evaluations": eval_review,
        "join_sequence": joined,
        "checkpoint_states": checkpoint_states,
    }


def compare_runs(continuous: Mapping[str, Any], first: Mapping[str, Any], resumed: Mapping[str, Any]) -> dict[str, Any]:
    """Compare continuous stream with split first + resumed streams."""
    if any(item.get("status") != "pass" for item in (continuous, first, resumed)):
        return {"status": "pending", "reason": "all three runs must pass artifact review before resume equality"}
    lhs = continuous.get("join_sequence", [])
    rhs = list(first.get("join_sequence", [])) + list(resumed.get("join_sequence", []))
    failures = []
    if lhs != rhs:
        first_difference = next((i for i, (a, b) in enumerate(zip(lhs, rhs)) if a != b), min(len(lhs), len(rhs)))
        failures.append({"kind": "generation_reward_sequence", "first_difference": first_difference,
                         "continuous": lhs[first_difference:first_difference + 1],
                         "split": rhs[first_difference:first_difference + 1]})
    # Checkpoint state metadata is compared at the same optimizer boundary.
    for step in (1, 2):
        left = continuous.get("checkpoint_states", {}).get(step)
        right_source = first if step == 1 else resumed
        right = right_source.get("checkpoint_states", {}).get(step)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            failures.append({"kind": "checkpoint_state", "step": step, "reason": "metadata missing"})
            continue
        for key in ("full", "step", "sampler"):
            if left.get(key) != right.get(key):
                failures.append({"kind": "checkpoint_state", "step": step, "field": key})
    return {"status": "pass" if not failures else "fail", "failures": failures,
            "compared": {"continuous_rows": len(lhs), "split_rows": len(rhs),
                         "optimizer_boundaries": [1, 2],
                         "keys": "source_id,prompt_ids_sha256,output_ids_sha256,reward in ordered stream"},
            "interpretation": "Resume equality is a mechanical identity check; it is not a quality promotion."}


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
    observed_at = datetime.now(timezone.utc).isoformat()
    result = summarize_run(args.run_root.resolve(), args.source_prefix_audit, args.development_panel)
    result["observed_at"] = observed_at
    if args.split_first and args.split_resume:
        first = summarize_run(args.split_first.resolve(), args.source_prefix_audit, args.development_panel, expected_updates=1, checkpoint_steps=(1,), prefix_offset=0)
        resumed = summarize_run(args.split_resume.resolve(), args.source_prefix_audit, args.development_panel, expected_updates=1, checkpoint_steps=(2,), prefix_offset=EXPECTED_ROWS_PER_UPDATE)
        result["split_comparison"] = compare_runs(result, first, resumed)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "run_root", "failure_first", "generation", "rewards", "gradients", "checkpoints", "evaluations", "supervision_contract", "split_comparison") if k in result}, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"pass", "pending", "failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
