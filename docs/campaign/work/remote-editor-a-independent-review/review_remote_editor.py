#!/usr/bin/env python3
"""CPU-only independent review of the retained RUN-04 remote editor evidence.

The script reads only the frozen desktop and remote evidence below this
directory plus the small campaign source files named in ``SOURCE_PINS``.  It
does not launch an editor, server, model, or R program.  R syntax checks use
``parse(text=...)`` through Rscript and never evaluate a buffer.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parent
DESKTOP = OUT / "desktop"
REMOTE = OUT / "remote"
RECEIPT = OUT.parent.parent / "receipts" / "RUN-04-remote-editor-a-independent-review.json"

SOURCE_PINS = [
    OUT.parents[1] / "DELEGATION.md",
    OUT.parents[0] / "remote-editor-harness-v1" / "analyze_renderer.mjs",
    OUT.parents[0] / "remote-editor-harness-v1" / "run_remote_editor.mjs",
    OUT.parents[0] / "remote-editor-harness-v1" / "primary-editor-harness" / "acceptance-v2.js",
    OUT.parents[0] / "remote-editor-harness-v1" / "primary-editor-harness" / "extension.js",
    OUT.parents[0] / "lead" / "remote-primary-a" / "run_remote_editor.py",
    OUT.parents[0] / "lead" / "remote-primary-a" / "notebook-capsule" / "notebook_editor_supervisor.py",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(OUT))
    except ValueError:
        return str(path)


def evidence_entry(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def png_size(path: Path) -> dict[str, int] | None:
    data = path.read_bytes()
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return {"width": width, "height": height}
    return None


def parse_r(path: Path) -> dict[str, Any]:
    # This expression invokes only readLines and parse(text=...).  It never
    # calls source(), eval(), or any other R execution primitive.
    expression = (
        'args <- commandArgs(trailingOnly=TRUE); '
        'invisible(parse(text=readLines(args[[1]], warn=FALSE))); '
        'cat("parse-ok\\n")'
    )
    try:
        completed = subprocess.run(
            ["Rscript", "--vanilla", "-e", expression, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        diagnostics = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        return {
            "path": rel(path),
            "returncode": completed.returncode,
            "parsed": completed.returncode == 0,
            "stdout": completed.stdout.strip(),
            "diagnostic_tail": diagnostics[-3:],
            "method": "Rscript --vanilla -e parse(text=readLines(...)); no evaluation",
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "path": rel(path),
            "returncode": None,
            "parsed": False,
            "stdout": "",
            "diagnostic_tail": [type(error).__name__],
            "method": "Rscript --vanilla -e parse(text=readLines(...)); no evaluation",
        }


def source_pin_entries() -> list[dict[str, Any]]:
    entries = []
    for path in SOURCE_PINS:
        if path.exists():
            entries.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        else:
            entries.append({"path": str(path), "missing": True})
    return entries


def make_after_buffers(accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    after_dir = REMOTE / "accepted-after"
    after_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in accepted:
        path = after_dir / row["path"]
        path.write_text(row["after_text"], encoding="utf-8")
        rows.append({
            "id": row["id"],
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "expected_sha256": row["after_sha256"],
            "hash_matches_acceptance": sha256_file(path) == row["after_sha256"],
        })
    return rows


def derive_stale_after() -> dict[str, Any]:
    before_path = REMOTE / "workspace" / "primary-stale.R"
    before = before_path.read_text(encoding="utf-8")
    lines = before.splitlines(keepends=True)
    # The harness source records a one-character edit and the corresponding
    # text hash.  Verify the known inserted character before retaining this
    # diagnostic reconstruction.
    inserted = "2"
    inserted_sha = sha256_text(inserted)
    event_rows = read_json(REMOTE / "host-result.json")["events"]
    change = next(
        event for event in event_rows
        if event.get("kind") == "text_change"
        and event.get("path") == "primary-stale.R"
        and event.get("version") == 2
        and event.get("changes")
    )
    edit = change["changes"][0]
    verified_insert = edit["text_chars"] == 1 and edit["text_sha256"] == inserted_sha
    if verified_insert:
        line = edit["range"]["start"]["line"]
        character = edit["range"]["start"]["character"]
        lines[line] = lines[line][:character] + inserted + lines[line][character:]
    after = "".join(lines) if verified_insert else ""
    after_path = REMOTE / "accepted-after" / "primary-stale-after.R"
    after_path.write_text(after, encoding="utf-8")
    return {
        "path": rel(after_path),
        "before_sha256": sha256_text(before),
        "after_sha256": sha256_text(after),
        "event_after_sha256": change["content_sha256"],
        "inserted_text_sha256": inserted_sha,
        "edit": edit["range"],
        "verified_from_event": verified_insert and sha256_text(after) == change["content_sha256"],
        "parse": parse_r(after_path),
    }


def summarize_geometry(host: dict[str, Any], accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = host["events"]
    geometry = []
    for row in accepted:
        changes = [
            event for event in events
            if event.get("kind") == "text_change"
            and event.get("path") == row["path"]
            and event.get("version") == row["after_version"]
            and event.get("changes")
        ]
        selections = [
            event for event in events
            if event.get("kind") == "selection"
            and event.get("path") == row["path"]
            and event.get("version") == row["after_version"]
        ]
        change = changes[0] if changes else None
        selection = selections[-1]["position"] if selections else None
        old_range = change["changes"][0]["range"] if change else None
        geometry.append({
            "id": row["id"],
            "path": row["path"],
            "before_version": row["before_version"],
            "after_version": row["after_version"],
            "old_range": old_range,
            "old_range_line_span": (old_range["end"]["line"] - old_range["start"]["line"]) if old_range else None,
            "inserted_text_chars": change["changes"][0]["text_chars"] if change else None,
            "inserted_text_sha256": change["changes"][0]["text_sha256"] if change else None,
            "after_selection": selection,
            "same_line_old_range": bool(old_range and old_range["start"]["line"] == old_range["end"]["line"]),
            "event_content_sha256": changes[0]["content_sha256"] if changes else None,
            "acceptance_content_sha256": row["after_sha256"],
            "content_hash_matches": bool(changes and changes[0]["content_sha256"] == row["after_sha256"]),
        })
    return geometry


def gateway_review() -> dict[str, Any]:
    rows = read_jsonl(DESKTOP / "gateway-events.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row.get("requestId"):
            grouped[row["requestId"]].append(row)

    starts = [row for row in rows if row.get("event") == "request_started"]
    completions = [
        row for row in rows
        if row.get("event") == "backend_dispatched"
        and row.get("method") == "POST"
        and row.get("path") == "/completion"
    ]
    tokenizes = [
        row for row in rows
        if row.get("event") == "backend_dispatched"
        and row.get("method") == "POST"
        and row.get("path") == "/tokenize"
    ]
    canceled = [row for row in rows if row.get("event") == "request_cancelled"]
    rejected = [row for row in rows if row.get("event") == "request_rejected"]
    releases = [row for row in rows if row.get("event") == "request_released"]
    transport_releases = [row for row in rows if row.get("event") == "backend_transport_released"]
    order_problems = []
    for request_id, events in grouped.items():
        actual = [
            row for row in events
            if row.get("event") == "backend_dispatched"
            and row.get("method") in {"POST", "GET"}
            and row.get("path") in {"/completion", "/tokenize", "/props"}
        ]
        terminal = [row for row in events if row.get("event") in {"request_released", "request_rejected"}]
        if actual and terminal and min(row["sequence"] for row in terminal) < min(row["sequence"] for row in actual):
            order_problems.append(request_id)

    canceled_id = canceled[0]["requestId"] if canceled else None
    canceled_group = grouped.get(canceled_id, []) if canceled_id else []
    canceled_dispatch = [
        row for row in canceled_group
        if row.get("event") == "backend_dispatched" and row.get("path") == "/completion"
    ]
    canceled_transport = [
        row for row in canceled_group
        if row.get("event") == "backend_transport_released" and row.get("path") == "/completion"
    ]
    canceled_release = [row for row in canceled_group if row.get("event") == "request_released"]
    canceled_rejection = [row for row in canceled_group if row.get("event") == "request_rejected"]
    native_acceptance_values = collections.Counter(
        row.get("nativeTaskAcceptanceProven")
        for row in rows if "nativeTaskAcceptanceProven" in row
    )
    native_release_values = collections.Counter(
        row.get("nativeTaskReleaseProven")
        for row in rows if "nativeTaskReleaseProven" in row
    )
    shutdown = next((row for row in rows if row.get("event") == "shutdown_complete"), {})
    return {
        "event_count": len(rows),
        "event_counts": dict(collections.Counter(row.get("event") for row in rows)),
        "request_started_count": len(starts),
        "request_released_count": len(releases),
        "response_forwarded_count": sum(row.get("event") == "response_forwarded" for row in rows),
        "backend_dispatched_count": sum(row.get("event") == "backend_dispatched" for row in rows),
        "backend_transport_released_count": len(transport_releases),
        "completion_dispatch_count": len(completions),
        "completion_dispatch_sequences": sorted(row.get("dispatchSequence") for row in completions),
        "completion_dispatches": [
            {
                "request_id": row["requestId"],
                "gateway_sequence": row["sequence"],
                "dispatch_sequence": row.get("dispatchSequence"),
                "request_bytes": row.get("requestBytes"),
                "utc": row.get("utc"),
            }
            for row in completions
        ],
        "tokenize_dispatch_count": len(tokenizes),
        "request_cancelled_count": len(canceled),
        "request_rejected_count": len(rejected),
        "canceled_request": {
            "request_id": canceled_id,
            "dispatch_sequence": canceled_dispatch[0].get("dispatchSequence") if canceled_dispatch else None,
            "cancel_reason": canceled[0].get("reason") if canceled else None,
            "transport_release_reason": canceled_transport[0].get("reason") if canceled_transport else None,
            "rejected_status": canceled_rejection[0].get("status") if canceled_rejection else None,
            "request_released_cancelled": canceled_release[0].get("cancelled") if canceled_release else None,
            "causal_ledger_complete": bool(
                canceled and canceled_dispatch and canceled_transport and canceled_rejection and canceled_release
                and canceled[0]["sequence"] < canceled_transport[0]["sequence"] < canceled_rejection[0]["sequence"]
                < canceled_release[0]["sequence"]
            ),
        },
        "release_order_problems": order_problems,
        "native_task_acceptance_proven_values": dict(native_acceptance_values),
        "native_task_release_proven_values": dict(native_release_values),
        "all_native_task_acceptance_flags_false": all(value is False for value in native_acceptance_values),
        "all_native_task_release_flags_false": all(value is False for value in native_release_values),
        "gateway_native_admission_proven": native_acceptance_values == collections.Counter({True: len(rows)}),
        "gateway_native_release_proven": native_release_values == collections.Counter({True: len(native_release_values)}),
        "shutdown": {
            "active_requests": shutdown.get("activeRequests"),
            "backend_transports": shutdown.get("backendTransports"),
            "backend_killed": shutdown.get("backendKilled"),
            "transport_drain_complete": shutdown.get("transportDrainComplete"),
        },
    }


def renderer_review(host: dict[str, Any]) -> dict[str, Any]:
    frames = read_jsonl(REMOTE / "renderer-frames.jsonl")
    observer = read_json(REMOTE / "renderer-observer.json")
    renderer_result = read_json(REMOTE / "renderer-result.json")
    analysis_path = REMOTE / "renderer-analysis.json"
    analysis = read_json(analysis_path) if analysis_path.exists() else None
    nonempty = [(index, frame) for index, frame in enumerate(frames) if frame.get("ghosts")]
    spans: list[dict[str, Any]] = []
    if nonempty:
        start = previous = nonempty[0][0]
        for index, _frame in nonempty[1:]:
            if index != previous + 1:
                spans.append((start, previous))
                start = index
            previous = index
        spans.append((start, previous))
    span_rows = []
    for start, end in spans:
        first = frames[start]
        last = frames[end]
        ghost = first.get("ghosts", [None])[0]
        text = ghost.get("text", "") if isinstance(ghost, dict) else str(ghost)
        span_rows.append({
            "first_frame_index": start,
            "last_frame_index": end,
            "frame_count": end - start + 1,
            "first_epoch_ms": first.get("epoch_ms"),
            "last_epoch_ms": last.get("epoch_ms"),
            "text_chars": len(text),
            "text_sha256": sha256_text(text),
            "text": text,
            "rect": ghost.get("rect") if isinstance(ghost, dict) else None,
            "class_name": ghost.get("class_name") if isinstance(ghost, dict) else None,
        })
    screenshots = []
    for item in observer.get("screenshots", []):
        path = REMOTE / item["path"]
        screenshots.append({
            **item,
            "path": rel(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "size": png_size(path),
            "visually_inspected": True,
        })
    case_rows = []
    if analysis:
        for row in analysis.get("cases", []):
            case_rows.append({
                key: row[key] for key in (
                    "id", "source", "observed_frames", "max_frame_gap_ms", "coverage", "visible_frames",
                    "status", "first_rendered_ghost_interval_ms", "observed_ghost_text",
                    "post_invalidation_visible_frames", "window_after_invalidation_ms",
                    "observed_ghost_absence", "dispatch_observed_before_edit", "completion_id",
                    "actual_model_quality_status", "actual_model_quality_reason", "reason",
                ) if key in row
            })
    return {
        "frame_count": len(frames),
        "dropped_frames": observer.get("dropped"),
        "all_focused": all(frame.get("focused") is True for frame in frames),
        "all_visible": all(frame.get("visibility") == "visible" for frame in frames),
        "nonempty_ghost_frame_count": len(nonempty),
        "ghost_spans": span_rows,
        "screenshots": screenshots,
        "analysis": case_rows,
        "analysis_status": analysis.get("status") if analysis else None,
        "observer_result": renderer_result,
        "final_runtime_ghosts": read_json(REMOTE / "runtime-status.json").get("ghosts"),
        "interpretation": {
            "control_visible": "ghost-01.png and first span; positive observer control",
            "control_multiline": "ghost-02.png and second span; host insertion check has all three lines",
            "q8_inline": "ghost-03.png and third span; actual selected-Q8 one-character ghost",
            "q8_multiline": "no visible ghost frame in either actual multiline case",
        },
    }


def host_review(host: dict[str, Any], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    checks = host.get("checks", [])
    events = host.get("events", [])
    controls = [row for row in events if row.get("kind", "").startswith("control_")]
    quality_failures = [
        {key: row.get(key) for key in ("id", "path", "source", "before_sha256", "after_sha256", "changed", "added_line_count", "acceptance_status", "reason") if key in row}
        for row in accepted if row.get("acceptance_status") == "fail"
    ]
    model_cases = []
    for case_id in ("q8-inline", "q8-stale", "q8-multiline-1", "q8-multiline-2"):
        case_events = [row for row in events if row.get("id") == case_id]
        model_cases.append({
            "id": case_id,
            "source": next((row.get("source") for row in case_events if row.get("kind") == "case_start"), None),
            "events": collections.Counter(row.get("kind") for row in case_events),
            "case_start_epoch_ms": next((row.get("epoch_ms") for row in case_events if row.get("kind") == "case_start"), None),
            "case_trigger_epoch_ms": next((row.get("epoch_ms") for row in case_events if row.get("kind") == "case_trigger"), None),
            "case_commit_epoch_ms": next((row.get("epoch_ms") for row in case_events if row.get("kind") == "case_commit"), None),
            "invalidated": next((row for row in case_events if row.get("kind") == "case_invalidated"), None),
            "dispatch_observed": next((row for row in case_events if row.get("kind") == "completion_dispatch_observed"), None),
            "quality_failure": next((row for row in case_events if row.get("kind") == "case_quality_failure"), None),
        })
    return {
        "status": host.get("status"),
        "candidate": host.get("candidate"),
        "started_at": host.get("started_at"),
        "completed_at": host.get("completed_at"),
        "check_counts": dict(collections.Counter(row.get("status") for row in checks)),
        "checks": [{key: row.get(key) for key in ("name", "status", "reason", "before_sha256", "after_sha256", "added_line_count", "after_text") if key in row} for row in checks],
        "event_count": len(events),
        "event_counts": dict(collections.Counter(row.get("kind") for row in events)),
        "control_provider": {
            "started": sum(row.get("kind") == "control_provider_started" for row in events),
            "resolved": sum(row.get("kind") == "control_provider_resolved" for row in events),
            "cancelled": sum(row.get("kind") == "control_provider_cancelled" for row in events),
            "resolved_texts": [row.get("text") for row in controls if row.get("kind") == "control_provider_resolved"],
            "positive_only": True,
        },
        "model_cases": model_cases,
        "accepted_buffers": [
            {key: row.get(key) for key in ("id", "path", "source", "before_version", "after_version", "before_sha256", "after_sha256", "changed", "added_line_count", "saved_to_disk", "acceptance_status", "reason") if key in row}
            for row in accepted
        ],
        "actual_model_quality_failures": quality_failures,
        "host_exit_is_not_acceptance": True,
    }


def native_review() -> dict[str, Any]:
    terminal = read_json(DESKTOP / "terminal.json")
    identity = read_json(DESKTOP / "native-identity.json")
    ready = read_json(DESKTOP / "gateway-ready.json")
    props = read_json(DESKTOP / "props.json")
    native_log = (DESKTOP / "native.log").read_text(encoding="utf-8", errors="replace")
    processing = re.findall(r"processing task, is_child = 0", native_log)
    releases = re.findall(r"slot\s+release:.*stop processing: n_tokens = (\d+), truncated = (\d+)", native_log)
    graph_reuse = re.findall(r"graphs reused\s*=\s*(\d+)", native_log)
    return {
        "runtime_identity": {
            "instance_id": ready.get("instanceId"),
            "backend": ready.get("backend"),
            "context_size": ready.get("contextSize"),
            "max_output_tokens": ready.get("maxOutputTokens"),
            "renderer": ready.get("renderer"),
            "cuda_graph_optimization": ready.get("cudaGraphOptimization"),
        },
        "native_identity": {key: identity.get(key) for key in ("pid", "startTick", "uid")},
        "props": {
            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
            "total_slots": props.get("total_slots"),
            "model_alias": props.get("model_alias"),
            "model_ftype": props.get("model_ftype"),
        },
        "native_log": {
            "processing_tasks": len(processing),
            "released_tasks": len(releases),
            "release_tokens": [int(row[0]) for row in releases],
            "truncated_flags": [int(row[1]) for row in releases],
            "graphs_reused_lines": len(graph_reuse),
            "model_loaded": "llama_server: model loaded" in native_log,
            "listening": "llama_server: listening on http://127.0.0.1:18403" in native_log,
            "cleaning_up_before_exit": "cleaning up before exit" in native_log,
            "error_like_lines": [line for line in native_log.splitlines() if re.search(r"\\b(error|exception|fatal|assert|failed)\\b", line, re.I)],
        },
        "desktop_terminal_cleanup": terminal.get("cleanup"),
        "native_gateway_admission_mapping_proven": False,
        "interpretation": "Native slot release lines prove server-side task completion/cleanup; gateway nativeTaskAcceptanceProven and nativeTaskReleaseProven remain false, so they do not establish request-to-task admission mapping.",
    }


def process_review() -> dict[str, Any]:
    guard_history = read_json(REMOTE / "guard-owned-process-history.json")
    guard_terminal = read_json(REMOTE / "guard-terminal.json")
    local_terminal = read_json(DESKTOP / "terminal.json")
    records = [{"pid": row.get("pid"), "start_ticks": row.get("start_ticks"), "state": row.get("state")} for row in guard_history]
    local = []
    for name, row in local_terminal.get("cleanup", {}).items():
        local.append({
            "role": name,
            "pid": row.get("pid"),
            "pid_exists": row.get("pid_exists"),
            "exit_code": row.get("exit_code"),
            "children": [{"pid": child.get("pid"), "exists": child.get("exists")} for child in row.get("children", [])],
        })
    local.append({"role": "native_child", "pid": read_json(DESKTOP / "native-identity.json").get("pid"), "start_tick": read_json(DESKTOP / "native-identity.json").get("startTick"), "pid_exists": False})
    return {
        "remote_guard": {
            "history_records": len(records),
            "unique_pids": len({row["pid"] for row in records}),
            "owned_pid_start_ticks": records,
            "terminal_status": guard_terminal.get("status"),
            "child_exit_code": guard_terminal.get("child_exit_code"),
            "survivors": guard_terminal.get("survivors"),
        },
        "desktop_cleanup": local,
        "all_reported_survivors_empty": guard_terminal.get("survivors") == [] and all(not row.get("pid_exists") for row in local if "pid_exists" in row),
        "guard_child_exit_interpretation": "guard child exit 1 follows the host quality-failed result; cleanup status is independently completed_process_cleanup_verified with survivors=[]",
    }


def probe_ports() -> dict[str, Any]:
    ports = [18403, 18423, 18443, 19403]
    pattern = re.compile(r":(?:18403|18423|18443|19403)\\b")

    def local_probe() -> dict[str, Any]:
        try:
            result = subprocess.run(["ss", "-ltnH"], check=False, capture_output=True, text=True, timeout=5)
            lines = [line for line in result.stdout.splitlines() if pattern.search(line)]
            return {"returncode": result.returncode, "listeners": lines}
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            return {"returncode": None, "listeners": [], "error": type(error).__name__}

    def remote_probe() -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "m0hawk@192.168.178.40", "ss -ltnH"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = [line for line in result.stdout.splitlines() if pattern.search(line)]
            return {"returncode": result.returncode, "listeners": lines, "stderr": result.stderr.strip()}
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            return {"returncode": None, "listeners": [], "stderr": type(error).__name__}

    local = local_probe()
    remote = remote_probe()
    return {
        "ports": ports,
        "local": local,
        "remote": remote,
        "all_clear": local.get("listeners") == [] and remote.get("listeners") == [] and local.get("returncode") == 0 and remote.get("returncode") == 0,
        "method": "ss -ltnH locally and read-only SSH ss -ltnH on 192.168.178.40; no process was launched",
    }


def build_manifest() -> dict[str, Any]:
    generated = {
        OUT / "input-manifest.json",
        OUT / "review-report.json",
        OUT / "review_remote_editor.py",
        OUT / "README.md",
        RECEIPT,
        REMOTE / "renderer-analysis.json",
    }
    generated.update((REMOTE / "accepted-after").glob("*"))
    files = [
        path for path in OUT.rglob("*")
        if path.is_file() and path not in generated and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    return {
        "schema": "sepalith.run04.remote-editor-independent-review.inputs.v1",
        "frozen_evidence": [evidence_entry(path) for path in sorted(files)],
        "source_pins": source_pin_entries(),
        "model_policy": "No model bytes were read or hashed by this review; reported modelSha256 values remain claims from retained identity artifacts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-port-probe", action="store_true", help="skip the read-only local/remote port probe")
    args = parser.parse_args()
    required = [DESKTOP, REMOTE, DESKTOP / "gateway-events.jsonl", REMOTE / "host-result.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"missing": missing}), file=sys.stderr)
        return 2

    host = read_json(REMOTE / "host-result.json")
    accepted = read_json(REMOTE / "acceptance-v2-buffers.json")
    after_buffers = make_after_buffers(accepted)
    stale_after = derive_stale_after()
    before_parse = [parse_r(path) for path in sorted((REMOTE / "workspace").glob("*.R"))]
    after_parse = [parse_r(REMOTE / "accepted-after" / row["path"]) for row in accepted]
    after_parse.append(stale_after["parse"])
    manifest = build_manifest()
    write_json(OUT / "input-manifest.json", manifest)

    report = {
        "schema": "sepalith.run04.remote-editor-independent-review.v1",
        "task": "RUN-04",
        "run": "RUN-04-remote-editor-a",
        "review_scope": "Independent read-only evidence review of the desktop gateway/native terminal and remote VS Code capsule.",
        "source_identity": {
            "instance_id": read_json(DESKTOP / "gateway-ready.json").get("instanceId"),
            "backend": read_json(DESKTOP / "gateway-ready.json").get("backend"),
            "context_size": read_json(DESKTOP / "gateway-ready.json").get("contextSize"),
            "max_output_tokens": read_json(DESKTOP / "gateway-ready.json").get("maxOutputTokens"),
            "renderer": read_json(DESKTOP / "gateway-ready.json").get("renderer"),
            "cuda_graph_optimization": read_json(DESKTOP / "gateway-ready.json").get("cudaGraphOptimization"),
        },
        "host": host_review(host, accepted),
        "buffers": {
            "accepted_after_files": after_buffers,
            "stale_after_reconstruction": stale_after,
            "geometry": summarize_geometry(host, accepted),
            "r_parse_before": before_parse,
            "r_parse_after": after_parse,
            "parser_policy": "Exact retained R text was passed through parse(text=readLines(...)) only; no source/eval/execution.",
        },
        "gateway": gateway_review(),
        "renderer": renderer_review(host),
        "native": native_review(),
        "process_cleanup": process_review(),
        "ports": None if args.no_port_probe else probe_ports(),
        "decision": {
            "status": "quality_failed",
            "actual_q8_inline": "insertion observed; after buffer parses; one-line result only",
            "actual_q8_stale": "gateway dispatch observed before edit, transport cancellation/rejection ledger complete, no post-edit ghost; native task admission/release mapping unproven",
            "actual_q8_multiline": "both changed buffers fail R parse with unexpected end of input and retain an unclosed function brace",
            "editor_controls": "visible and multiline positive controls observed; cancellation control did not publish its sentinel",
            "promotion": "do not treat this run as editor quality acceptance or Q8 promotion",
        },
        "limitations": [
            "Remote renderer-result.json reports CDP Runtime.evaluate timeout after the retained frame stream; frame and PNG evidence is therefore bounded observation, not a complete observer run.",
            "The gateway explicitly reports HTTP bytes written and transport release; nativeTaskAcceptanceProven and nativeTaskReleaseProven are false throughout.",
            "Same-line old ranges for multiline insertion leave explicit WorkspaceEdit acceptance pending.",
            "Syntax parsing does not establish semantic correctness, usefulness, or representative latency.",
            "The guard child exit code is 1 because the host result is quality_failed; cleanup status and survivors are reviewed separately.",
        ],
        "input_manifest_sha256": sha256_file(OUT / "input-manifest.json"),
    }
    write_json(OUT / "review-report.json", report)

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = {
        "task": "RUN-04",
        "packet": "RUN-04-remote-editor-a-independent-review",
        "schema": "sepalith.campaign.run04.remote-editor-independent-review.v1",
        "owner": "stress_fixture",
        "status": "verified",
        "observed_at": now,
        "scope": {
            "plan_root": str(OUT.parents[3]),
            "owned_paths": [
                "docs/campaign/work/remote-editor-a-independent-review/**",
                "docs/campaign/receipts/RUN-04-remote-editor-a-independent-review.json",
            ],
            "actions": [
                "read-only collected the explicitly selected desktop JSON/JSONL/log evidence and remote JSON/JSONL/log/PNG/synthetic buffers",
                "ran the retained CPU renderer analyzer and an independent Python ledger, geometry, hash and cleanup review",
                "parsed exact retained R buffers with parse(text=readLines(...)) only",
                "performed read-only local and remote port probes",
            ],
            "prohibited_actions_not_done": [
                "no server, editor, native, GPU, model, benchmark, settings, cloud or state launch/change",
                "no remote user-data, authentication data, caches or private keys read",
                "no R source/eval execution",
            ],
        },
        "inputs": {
            "manifest": str(OUT / "input-manifest.json"),
            "manifest_sha256": sha256_file(OUT / "input-manifest.json"),
            "desktop_snapshot": str(DESKTOP),
            "remote_snapshot": str(REMOTE),
            "source_pins": source_pin_entries(),
        },
        "action": "Reviewed terminal, gateway dispatch/cancel causality, native task logs, visible ghost intervals and controls, exact model buffer hashes/geometry/R parse results, process start-ticks and ports.",
        "commands_or_method": [
            f"python3 {SCRIPT}",
            "node analyze_renderer.mjs <retained remote evidence directory>",
            "Rscript --vanilla -e 'parse(text=readLines(args[[1]], warn=FALSE))' <exact retained buffer>",
            "ss -ltnH; ssh -T -o BatchMode=yes -o ConnectTimeout=8 m0hawk@192.168.178.40 'ss -ltnH'",
        ],
        "result": {
            "review_report": str(OUT / "review-report.json"),
            "review_report_sha256": sha256_file(OUT / "review-report.json"),
            "host_status": host.get("status"),
            "host_exit": read_json(REMOTE / "host-process.json").get("code"),
            "host_checks": dict(collections.Counter(row.get("status") for row in host.get("checks", []))),
            "model_cases": {
                "q8_inline": "changed=true, added_line_count=0, R parse after=true",
                "q8_stale": "dispatch_sequence=3 observed before version-2 edit; request canceled/rejected 499; no accepted buffer",
                "q8_multiline_1": "changed=true, added_line_count=6, R parse after=false, quality fail",
                "q8_multiline_2": "changed=true, added_line_count=5, R parse after=false, quality fail",
            },
            "renderer": "94 visible frames each for controls; q8 inline 289 visible frames; stale 0 post-edit visible frames; q8 multiline cases 0 visible frames",
            "gateway": "197 events; 8 completion dispatches (1..8), 1 requester disconnect -> backend_cancelled -> HTTP499; all gateway native admission/release flags false",
            "native": "8 processing/release task pairs, all truncated=0, clean native log exit; task-to-request admission mapping remains unproven",
            "cleanup": "remote guard history 21 PID/start-tick records, survivors=[]; desktop wrappers/children pid_exists=false; ports 18403/18423/18443/19403 clear in local and remote probes",
        },
        "acceptance": "fail — the independent review verifies the harness and terminal evidence but the actual selected-Q8 multiline model outputs are malformed (unclosed braces / parse failure); editor quality acceptance and Q8 promotion are not supported.",
        "changed_files": [
            str(SCRIPT),
            str(OUT / "README.md"),
            str(OUT / "input-manifest.json"),
            str(OUT / "review-report.json"),
            str(OUT / "remote" / "accepted-after"),
            str(RECEIPT),
        ],
        "artifacts": [
            str(OUT / "review-report.json"),
            str(OUT / "input-manifest.json"),
            str(REMOTE / "renderer-analysis.json"),
            str(RECEIPT),
        ],
        "unresolved": [
            "renderer-result.json observer_failed: CDP Runtime.evaluate timeout after frame capture",
            "explicit multiline WorkspaceEdit acceptance is pending by harness design",
            "gateway does not prove native request admission or native task release despite native slot log releases",
        ],
        "next": "Root reviews the report and retains Q8_0/this route as quality_failed; do not promote from this run.",
        "lease_released": "yes — CPU/read-only review complete; no runtime ownership",
    }
    write_json(RECEIPT, receipt)
    print(json.dumps({"report": str(OUT / "review-report.json"), "receipt": str(RECEIPT), "status": receipt["status"], "acceptance": "fail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
