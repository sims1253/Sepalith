#!/usr/bin/env python3
"""Bounded cancellation/retry probe for the pinned native serving protocol.

This is a client-only probe.  It imports the already tested v2 native probe
after checking its immutable SHA-256, and uses the same TRAIN fixture and
tokenizer contract.  One invocation covers the four selected TRAIN prompts at
both 1 s and 5 s whole-request deadlines (eight sequences maximum).  A cold
stream is cancelled by closing its socket at the deadline, then the exact full
prompt is tokenized and sent again with ``cache_prompt`` enabled.  The retry is
compared with an accepted 60 s baseline when one is supplied.

No model, server, build, or cloud process is started by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import tempfile
import socket
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA_VERSION = "run-06.native-transition-probe.v1"
TRANSITION_MANIFEST_SCHEMA = "run-06.native-transition-manifest.v1"
EXPECTED_NATIVE_PROBE_SHA256 = "f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7"
DEFAULT_CANCEL_DEADLINES_MS = (1000, 5000)
ALLOWED_BATCHES = (64, 128, 256)
MAX_SEQUENCES = 8
MAX_TOTAL_MS = 600000
MAX_RETRY_MS = 60000
MAX_DIAGNOSTIC_MS = 60000
NO_LOG_ACK = "unmeasured_no_log_correlation"


class TransitionProbeError(Exception):
    """A bounded local validation or protocol failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise TransitionProbeError("io_error", f"cannot read {path}: {exc}") from exc


def load_native_probe() -> tuple[Any, Path, str]:
    path = Path(__file__).with_name("runtime_native_probe.py")
    actual = sha256_file(path)
    if actual != EXPECTED_NATIVE_PROBE_SHA256:
        raise TransitionProbeError(
            "native_source_hash_mismatch",
            f"runtime_native_probe.py SHA-256 {actual} != pinned {EXPECTED_NATIVE_PROBE_SHA256}",
        )
    spec = importlib.util.spec_from_file_location("sepalith_pinned_native_probe_v2", path)
    if spec is None or spec.loader is None:
        raise TransitionProbeError("native_import_error", f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path, actual


native, NATIVE_PROBE_PATH, NATIVE_PROBE_SHA256 = load_native_probe()


def _short(value: object, limit: int = 300) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text.replace("\x00", "")[:limit]


def parse_deadlines(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise TransitionProbeError("invalid_plan", "deadlines must be comma-separated integers") from exc
    else:
        parsed = tuple(int(item) for item in value)
    if parsed != DEFAULT_CANCEL_DEADLINES_MS:
        raise TransitionProbeError(
            "invalid_plan",
            "the transition contract requires exactly the 1000 ms and 5000 ms deadlines",
        )
    return parsed


def validate_plan(batch: int, deadlines_ms: tuple[int, ...], retry_deadline_ms: int,
                  cap: int, context: int, diagnostic_ms: int, row_count: int) -> None:
    if batch not in ALLOWED_BATCHES:
        raise TransitionProbeError("invalid_plan", f"batch must be one of {ALLOWED_BATCHES}")
    if len(deadlines_ms) * row_count > MAX_SEQUENCES:
        raise TransitionProbeError("sequence_limit", "planned transition sequences exceed eight")
    if retry_deadline_ms <= 0 or retry_deadline_ms > MAX_RETRY_MS:
        raise TransitionProbeError("invalid_plan", "retry deadline must be in 1..60000 ms")
    if cap <= 0 or cap > native.MAX_CAP:
        raise TransitionProbeError("invalid_plan", "cap must be in 1..192")
    if context <= 0 or context > native.MAX_CONTEXT:
        raise TransitionProbeError("invalid_plan", "context must be in 1..4096")
    if diagnostic_ms <= 0 or diagnostic_ms > MAX_DIAGNOSTIC_MS:
        raise TransitionProbeError("invalid_plan", "diagnostic timeout must be in 1..60000 ms")


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionProbeError(code, f"cannot read JSON {path}: {_short(exc)}") from exc
    if not isinstance(value, dict):
        raise TransitionProbeError(code, f"{path} must contain a JSON object")
    return value


def load_inputs(fixture_path: Path, native_manifest_path: Path,
                transition_manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(transition_manifest_path, "invalid_manifest")
    if manifest.get("schema_version") != TRANSITION_MANIFEST_SCHEMA:
        raise TransitionProbeError("invalid_manifest", "transition manifest schema mismatch")
    actual_native_hash = sha256_file(NATIVE_PROBE_PATH)
    if manifest.get("native_probe_sha256") != actual_native_hash:
        raise TransitionProbeError("native_source_hash_mismatch", "transition manifest native probe hash mismatch")
    if manifest.get("native_probe_sha256") != EXPECTED_NATIVE_PROBE_SHA256:
        raise TransitionProbeError("native_source_hash_mismatch", "transition manifest is not pinned to native v2")
    if manifest.get("fixture_sha256") != sha256_file(fixture_path):
        raise TransitionProbeError("fixture_hash_mismatch", "transition fixture SHA-256 mismatch")
    if manifest.get("native_manifest_sha256") != sha256_file(native_manifest_path):
        raise TransitionProbeError("fixture_hash_mismatch", "native fixture manifest SHA-256 mismatch")
    try:
        rows, native_manifest = native.load_fixture(fixture_path, native_manifest_path)
    except native.ProbeError as exc:
        raise TransitionProbeError(exc.code, exc.message) from exc
    selected = manifest.get("selected_rows")
    if not isinstance(selected, list) or len(selected) != len(rows):
        raise TransitionProbeError("invalid_manifest", "transition selected_rows must cover exactly the native fixture")
    by_id = {row["id"]: row for row in rows}
    ordered: list[dict[str, Any]] = []
    for ordinal, identity in enumerate(selected):
        if not isinstance(identity, dict):
            raise TransitionProbeError("invalid_manifest", f"selected row {ordinal} is not an object")
        row_id = identity.get("row_id")
        row = by_id.get(row_id)
        if row is None:
            raise TransitionProbeError("invalid_manifest", f"selected row {row_id!r} is absent from fixture")
        if identity.get("ordinal") != ordinal:
            raise TransitionProbeError("invalid_manifest", f"selected row {row_id} ordinal mismatch")
        if identity.get("row_sha256") != sha256_bytes(native.canonical_json(row)):
            raise TransitionProbeError("fixture_hash_mismatch", f"selected row {row_id} digest mismatch")
        if identity.get("prompt_sha256") != native.sha256_text(row["prompt_text"]):
            raise TransitionProbeError("fixture_hash_mismatch", f"selected row {row_id} prompt digest mismatch")
        if row.get("split") != "train":
            raise TransitionProbeError("invalid_manifest", f"selected row {row_id} is not TRAIN")
        ordered.append(row)
    if [row["id"] for row in ordered] != [item["row_id"] for item in selected]:
        raise TransitionProbeError("invalid_manifest", "selected row order is not deterministic")
    if len({row["id"] for row in ordered}) != len(rows):
        raise TransitionProbeError("invalid_manifest", "duplicate transition row")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise TransitionProbeError("invalid_manifest", "tokenizer identity is missing")
    for key in ("revision", "json_sha256", "policy"):
        if any(row["tokenizer_revision"] != tokenizer.get("revision") if key == "revision" else
               row["tokenizer_json_sha256"] != tokenizer.get("json_sha256") if key == "json_sha256" else
               row["tokenization_policy"] != tokenizer.get("policy") for row in ordered):
            raise TransitionProbeError("invalid_manifest", f"tokenizer {key} does not match rows")
    if any(row["split"] != "train" for row in ordered):
        raise TransitionProbeError("invalid_manifest", "transition manifest contains a non-TRAIN row")
    return ordered, manifest, native_manifest


def _remaining_ms(deadline_ns: int) -> int:
    return max(0, int((deadline_ns - time.monotonic_ns()) / 1_000_000))


def timed_json(url: str, body: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    result: dict[str, Any] = {
        "started_monotonic_ns": started_ns,
        "timeout_ms": max(0, timeout_ms),
        "status": "pending",
    }
    if timeout_ms <= 0:
        result.update({"status": "timeout", "timed_out": True,
                       "error": "whole deadline elapsed before request"})
        result["ended_monotonic_ns"] = time.monotonic_ns()
        result["wall_ms"] = round((result["ended_monotonic_ns"] - started_ns) / 1_000_000, 3)
        return result
    try:
        status, payload, elapsed = native.request_json(url, "POST", body, timeout_ms / 1000.0)
        result.update({"status": "ok", "http_status": status, "payload": payload,
                       "wall_ms": round(elapsed, 3), "timed_out": False})
    except native.ProbeError as exc:
        result.update({"status": "timeout" if exc.timed_out else exc.code,
                       "http_status": exc.status, "error": _short(exc.message),
                       "timed_out": exc.timed_out})
    ended_ns = time.monotonic_ns()
    result["ended_monotonic_ns"] = ended_ns
    result.setdefault("wall_ms", round((ended_ns - started_ns) / 1_000_000, 3))
    return result


def _set_socket_timeout(response: object, timeout_s: float) -> None:
    native.set_socket_timeout(response, max(0.001, timeout_s))


def stream_with_deadline(url: str, body: dict[str, Any], deadline_ns: int) -> dict[str, Any]:
    """Read source-shaped SSE until completion or close the socket at deadline."""
    request_started_ns = time.monotonic_ns()
    result: dict[str, Any] = {
        "request_started_monotonic_ns": request_started_ns,
        "deadline_monotonic_ns": deadline_ns,
        "response_arrival_monotonic_ns": None,
        "first_event_monotonic_ns": None,
        "first_content_monotonic_ns": None,
        "cancel_initiated_monotonic_ns": None,
        "socket_closed_monotonic_ns": None,
        "timed_out": False,
        "cancelled": False,
        "text": "",
        "returned_token_ids": [],
        "final": {},
        "malformed_sse": False,
        "token_chunk_count": 0,
        "final_token_count": 0,
    }
    response: Any = None
    text_parts: list[str] = []
    token_ids: list[int] = []
    final: dict[str, Any] = {}
    ttft_ns: int | None = None
    complete = False
    try:
        remaining = _remaining_ms(deadline_ns)
        if remaining <= 0:
            result.update({"timed_out": True, "cancelled": True,
                           "cancel_initiated_monotonic_ns": time.monotonic_ns(),
                           "error_code": "timeout", "error": "whole deadline elapsed before /completion"})
        else:
            response = urlopen(Request(
                url, data=native.canonical_json(body), method="POST",
                headers={"Accept": "text/event-stream", "Content-Type": "application/json"}),
                timeout=max(0.001, remaining / 1000.0))
            result["response_arrival_monotonic_ns"] = time.monotonic_ns()
            status = int(response.getcode())
            if status < 200 or status >= 300:
                raise TransitionProbeError("http_error", f"HTTP {status}")
            while True:
                remaining = _remaining_ms(deadline_ns)
                if remaining <= 0:
                    result.update({"timed_out": True, "cancelled": True,
                                   "cancel_initiated_monotonic_ns": time.monotonic_ns(),
                                   "error_code": "timeout", "error": "stream exceeded whole deadline"})
                    break
                _set_socket_timeout(response, remaining / 1000.0)
                try:
                    raw = response.readline()
                except (socket.timeout, TimeoutError) as exc:
                    result.update({"timed_out": True, "cancelled": True,
                                   "cancel_initiated_monotonic_ns": time.monotonic_ns(),
                                   "error_code": "timeout", "error": _short(exc)})
                    break
                if raw == b"":
                    complete = True
                    break
                event_ns = time.monotonic_ns()
                if result["first_event_monotonic_ns"] is None:
                    result["first_event_monotonic_ns"] = event_ns
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    complete = True
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    result["malformed_sse"] = True
                    continue
                if not isinstance(chunk, dict):
                    result["malformed_sse"] = True
                    continue
                content = chunk.get("content", "")
                if not isinstance(content, str):
                    raise TransitionProbeError("invalid_response", "stream content must be a string")
                if content and ttft_ns is None:
                    ttft_ns = event_ns
                    result["first_content_monotonic_ns"] = event_ns
                text_parts.append(content)
                if "tokens" in chunk:
                    try:
                        chunk_tokens = native.id_array(chunk["tokens"], "stream.tokens", allow_empty=True)
                    except native.ProbeError as exc:
                        raise TransitionProbeError(exc.code, exc.message) from exc
                    result["token_chunk_count"] += 1
                    if chunk.get("stop") is True:
                        result["final_token_count"] = len(chunk_tokens)
                    else:
                        token_ids.extend(chunk_tokens)
                for key in ("stop", "stop_type", "stopping_word", "truncated",
                            "tokens_evaluated", "tokens_predicted", "tokens_cached",
                            "n_tokens_cached", "n_prompt_tokens_cache", "timings",
                            "queued_ms", "queued_wait_ms", "queue_ms", "final"):
                    if key in chunk:
                        final[key] = chunk[key]
                if chunk.get("stop") is True:
                    complete = True
                    break
    except TransitionProbeError as exc:
        result.update({"error_code": exc.code, "error": exc.message})
    except HTTPError as exc:
        result.update({"error_code": "http_error", "error": f"HTTP {exc.code}",
                       "http_status": int(exc.code)})
    except (socket.timeout, TimeoutError) as exc:
        result.update({"timed_out": True, "cancelled": True,
                       "cancel_initiated_monotonic_ns": time.monotonic_ns(),
                       "error_code": "timeout", "error": _short(exc)})
    except URLError as exc:
        timed_out = isinstance(exc.reason, (socket.timeout, TimeoutError))
        result.update({"timed_out": timed_out, "cancelled": timed_out,
                       "cancel_initiated_monotonic_ns": time.monotonic_ns() if timed_out else None,
                       "error_code": "timeout" if timed_out else "network_error",
                       "error": _short(exc)})
    except (OSError, http.client.HTTPException) as exc:
        result.update({"error_code": "network_error", "error": _short(exc)})
    finally:
        if result.get("cancelled") and result.get("cancel_initiated_monotonic_ns") is None:
            result["cancel_initiated_monotonic_ns"] = time.monotonic_ns()
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        result["socket_closed_monotonic_ns"] = time.monotonic_ns()
    ended_ns = time.monotonic_ns()
    result.update({
        "ended_monotonic_ns": ended_ns,
        "wall_ms": round((ended_ns - request_started_ns) / 1_000_000, 3),
        "text": "".join(text_parts),
        "returned_token_ids": token_ids,
        "final": final,
        "complete": complete,
        "ttft_ms": None if ttft_ns is None else round((ttft_ns - request_started_ns) / 1_000_000, 3),
    })
    if result.get("cancelled"):
        result["cancel_reason"] = "whole_deadline"
    return result


def _timing_values(streamed: dict[str, Any]) -> tuple[Any, Any, Any]:
    final = streamed.get("final") if isinstance(streamed.get("final"), dict) else {}
    timings = final.get("timings") if isinstance(final.get("timings"), dict) else {}
    cache_n = timings.get("cache_n")
    prompt_n = timings.get("prompt_n")
    queued_wait = next((final.get(key) for key in ("queued_wait_ms", "queued_ms", "queue_ms")
                        if final.get(key) is not None), None)
    if queued_wait is None:
        queued_wait = next((timings.get(key) for key in ("queued_wait_ms", "queued_ms", "queue_ms")
                            if timings.get(key) is not None), None)
    return cache_n, prompt_n, queued_wait


def validate_stream(row: dict[str, Any], streamed: dict[str, Any], cap: int,
                    prompt_ids: list[int]) -> dict[str, Any]:
    returned = streamed.get("returned_token_ids", [])
    final = streamed.get("final") if isinstance(streamed.get("final"), dict) else {}
    checks: dict[str, str] = {}
    if not returned:
        checks["returned_ids"] = "missing"
    elif any(native.is_native_control_token(token) for token in returned[:-1]):
        checks["returned_ids"] = "early_control_token"
    elif returned[-1] != native.EOS_ID:
        checks["returned_ids"] = "noncanonical_eog" if returned[-1] in native.NATIVE_EOG_IDS else "missing_eos1"
    elif len(returned) > cap:
        checks["returned_ids"] = "over_cap"
    else:
        checks["returned_ids"] = "valid_eos1"
    if final.get("tokens_evaluated") is not None:
        checks["prompt_length"] = "exact" if final.get("tokens_evaluated") == len(prompt_ids) else "mismatch"
    else:
        checks["prompt_length"] = "missing"
    checks["stop"] = "eos" if final.get("stop_type") == "eos" and final.get("stop") is True else "unexpected"
    parsed = native.parse_wire_output(streamed.get("text", ""))
    checks["output_parser"] = parsed.get("status", "invalid")
    checks["stream_layout"] = ("partial_chunks_final_empty"
                                if streamed.get("final_token_count") == 0 else "unexpected_final_tokens")
    cache_n, prompt_n, queued_wait = _timing_values(streamed)
    good = (streamed.get("complete") is True
            and checks["returned_ids"] == "valid_eos1"
            and checks["prompt_length"] == "exact"
            and checks["stop"] == "eos"
            and checks["output_parser"] == "accepted"
            and checks["stream_layout"] == "partial_chunks_final_empty"
            and streamed.get("malformed_sse") is False)
    return {
        "protocol_status": "accepted" if good else "protocol_error",
        "protocol_checks": checks,
        "parsed_output": parsed,
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        # A server timing field is retained as a hint, but a queue delay needs
        # correlated server arrival and launch/slot timestamps.  Never turn
        # TTFT or an uncorrelated field into a queue measurement.
        "server_queue_hint_ms": queued_wait,
        "queued_wait_ms": None,
        "queued_wait_status": "unmeasured_no_log_correlation",
        "raw_text": streamed.get("text", ""),
        "raw_text_sha256": native.sha256_text(streamed.get("text", "")),
        "returned_token_ids": returned,
        "returned_token_count": len(returned),
        "ttft_ms": streamed.get("ttft_ms"),
        "end_wall_ms": streamed.get("wall_ms"),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "timings": final.get("timings"),
        "server_ack": {"status": NO_LOG_ACK, "evidence": None},
    }


def run_attempt(base_url: str, row: dict[str, Any], phase: str, cache_prompt: bool,
                deadline_ms: int, cap: int, context: int, diagnostic_ms: int,
                *, combined_started_ns: int | None = None,
                budget_deadline_ns: int | None = None) -> dict[str, Any]:
    expected_prompt = list(row["input_ids"][1:row["target_start"]])
    prompt_ids = [native.BOS_ID, *expected_prompt]
    started_ns = combined_started_ns if combined_started_ns is not None else time.monotonic_ns()
    deadline_ns = started_ns + deadline_ms * 1_000_000
    if budget_deadline_ns is not None:
        deadline_ns = min(deadline_ns, budget_deadline_ns)
    record: dict[str, Any] = {
        "phase": phase,
        "deadline_ms": deadline_ms,
        "deadline_scope": "combined_tokenize_plus_completion" if combined_started_ns is not None else "attempt",
        "started_monotonic_ns": started_ns,
        "deadline_monotonic_ns": deadline_ns,
        "prompt_token_count": row["prompt_token_count"],
        "prompt_ids_count": len(prompt_ids),
        "prompt_ids_sha256": native.sha256_bytes(native.canonical_json(prompt_ids)),
        "prompt_ids_include_manual_bos": True,
        "cache_prompt": cache_prompt,
        "cap": cap,
        "context_size": context,
        "diagnostic_max_ms": min(diagnostic_ms, MAX_DIAGNOSTIC_MS),
        "server_ack": {"status": NO_LOG_ACK, "evidence": None},
    }
    if len(prompt_ids) + cap > context:
        record.update({"protocol_status": "context_budget", "error": "prompt plus cap exceeds context"})
        record["ended_monotonic_ns"] = time.monotonic_ns()
        return record
    tokenize_timeout = _remaining_ms(deadline_ns)
    tokenize_body = {"content": row["prompt_text"], "add_special": False,
                     "parse_special": False, "with_pieces": False}
    tokenized = timed_json(f"{base_url}/tokenize", tokenize_body, tokenize_timeout)
    record["tokenize"] = {key: value for key, value in tokenized.items() if key != "payload"}
    if tokenized.get("status") != "ok":
        record.update({"protocol_status": "timeout" if tokenized.get("timed_out") else "tokenize_error",
                       "error": tokenized.get("error", "tokenize failed"),
                       "timed_out": bool(tokenized.get("timed_out")),
                       "queued_wait_ms": None, "queued_wait_status": "unmeasured_no_log_correlation"})
        record["ended_monotonic_ns"] = time.monotonic_ns()
        return record
    try:
        native_tokens = native.id_array(tokenized["payload"].get("tokens"), "tokenize.tokens", allow_empty=True)
    except (AttributeError, native.ProbeError) as exc:
        message = exc.message if isinstance(exc, native.ProbeError) else _short(exc)
        record.update({"protocol_status": "tokenize_error", "error": message})
        record["ended_monotonic_ns"] = time.monotonic_ns()
        return record
    if any(native.is_native_control_token(token) for token in native_tokens):
        record.update({"protocol_status": "prompt_control_token", "error": "tokenize response contains native control ID"})
        record["ended_monotonic_ns"] = time.monotonic_ns()
        return record
    record["tokenize"].update({
        "tokenized_ids_count": len(native_tokens),
        "tokenized_ids_sha256": native.sha256_bytes(native.canonical_json(native_tokens)),
        "prompt_id_status": "exact_stored_prompt" if native_tokens == expected_prompt else "mismatch",
    })
    if native_tokens != expected_prompt:
        record.update({"protocol_status": "prompt_token_mismatch",
                       "error": "server /tokenize IDs differ from accepted TRAIN row"})
        record["ended_monotonic_ns"] = time.monotonic_ns()
        return record
    completion_body = {
        "prompt": prompt_ids,
        "n_predict": cap,
        "temperature": 0,
        "stream": True,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
    }
    completion_timeout = _remaining_ms(deadline_ns)
    streamed = stream_with_deadline(f"{base_url}/completion", completion_body, deadline_ns)
    record["completion"] = {key: value for key, value in streamed.items()
                             if key not in {"final", "text", "returned_token_ids"}}
    if streamed.get("timed_out"):
        record.update({"protocol_status": "cancelled_after_deadline", "timed_out": True,
                       "cancelled": True, "completion_timeout_ms": completion_timeout,
                       "queued_wait_ms": None, "queued_wait_status": "unmeasured_no_log_correlation"})
        record["ended_monotonic_ns"] = time.monotonic_ns()
        return record
    validation = validate_stream(row, streamed, cap, prompt_ids)
    record.update(validation)
    record["completion_timeout_ms"] = completion_timeout
    record["ended_monotonic_ns"] = time.monotonic_ns()
    return record


def _load_baseline(path: Path | None, phase: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided", "path": None, "records": {}}
    baseline = _read_json(path, "invalid_baseline")
    requests = baseline.get("requests")
    if not isinstance(requests, list):
        raise TransitionProbeError("invalid_baseline", "baseline lacks requests")
    summary = baseline.get("summary") if isinstance(baseline.get("summary"), dict) else {}
    baseline_deadline = summary.get("user_deadline_ms")
    if not isinstance(baseline_deadline, int):
        values = [item.get("user_deadline_ms") for item in requests
                  if isinstance(item, dict) and isinstance(item.get("user_deadline_ms"), int)]
        baseline_deadline = min(values) if values else None
    if baseline_deadline is None or baseline_deadline < 60000:
        return {"status": "rejected_not_60s", "path": str(path),
                "sha256": sha256_file(path), "records": {}}
    index: dict[str, dict[str, Any]] = {}
    for item in requests:
        if not isinstance(item, dict) or item.get("phase") != phase or item.get("rep") != 1:
            continue
        row_id = item.get("row_id")
        if isinstance(row_id, str) and row_id not in index:
            index[row_id] = item
    missing = [row["id"] for row in rows if row["id"] not in index]
    unaccepted = [row_id for row_id, item in index.items() if item.get("protocol_status") != "accepted"]
    status = "ready" if not missing and not unaccepted else "incomplete"
    return {"status": status, "path": str(path), "sha256": sha256_file(path),
            "phase": phase, "deadline_ms": baseline_deadline, "records": index,
            "missing_row_ids": missing, "unaccepted_row_ids": unaccepted}


def compare_retry(retry: dict[str, Any], row_id: str, baseline: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "pending", "row_id": row_id}
    if baseline.get("status") == "not_provided":
        result["status"] = "baseline_not_provided"
        return result
    if baseline.get("status") != "ready":
        result["status"] = "baseline_unavailable"
        result["baseline_status"] = baseline.get("status")
        return result
    expected = baseline["records"].get(row_id)
    if expected is None:
        result["status"] = "baseline_missing_row"
        return result
    result["baseline_row_sha256"] = expected.get("raw_text_sha256")
    same_text = retry.get("raw_text") == expected.get("raw_text")
    same_ids = retry.get("returned_token_ids") == expected.get("returned_token_ids")
    result.update({"status": "pass" if retry.get("protocol_status") == "accepted" and same_text and same_ids else "mismatch",
                   "text_exact": same_text, "token_ids_exact": same_ids})
    if result["status"] != "pass":
        result.update({"retry_raw_text_sha256": retry.get("raw_text_sha256"),
                       "baseline_token_ids": expected.get("returned_token_ids"),
                       "retry_token_ids": retry.get("returned_token_ids")})
    return result


def run_transition(base_url: str, row: dict[str, Any], batch: int, cancel_ms: int,
                   cap: int, context: int, retry_deadline_ms: int,
                   diagnostic_ms: int, baseline: dict[str, Any],
                   budget_deadline_ns: int) -> dict[str, Any]:
    sequence_started_ns = time.monotonic_ns()
    expected_prompt_ids = [native.BOS_ID, *row["input_ids"][1:row["target_start"]]]
    sequence: dict[str, Any] = {
        "sequence_id": f"{row['id']}:cancel-{cancel_ms}ms",
        "row_id": row["id"],
        "batch": batch,
        "cancel_after_ms": cancel_ms,
        "deadline_scope": "cold tokenize plus completion; retry is separately bounded",
        "started_monotonic_ns": sequence_started_ns,
        "expected_prompt_ids_sha256": native.sha256_bytes(native.canonical_json(expected_prompt_ids)),
        "expected_prompt_ids_count": len(expected_prompt_ids),
        "server_ack": {"status": NO_LOG_ACK, "evidence": None},
    }
    cold = run_attempt(base_url, row, "cold", False, cancel_ms, cap, context, diagnostic_ms,
                       combined_started_ns=sequence_started_ns,
                       budget_deadline_ns=budget_deadline_ns)
    sequence["cold"] = cold
    # Do not insert a health request between cancellation and retry.
    sequence["diagnostic"] = {"status": "not_requested", "reason": "preserve immediate cancellation-to-retry timing"}
    if _remaining_ms(budget_deadline_ns) <= 0:
        sequence.update({"protocol_status": "total_budget_exhausted",
                         "retry": None, "ended_monotonic_ns": time.monotonic_ns()})
        return sequence
    retry_started_ns = time.monotonic_ns()
    retry = run_attempt(base_url, row, "retry", True, retry_deadline_ms, cap, context, diagnostic_ms,
                        combined_started_ns=None, budget_deadline_ns=budget_deadline_ns)
    sequence["retry"] = retry
    for phase_record in (cold, retry):
        phase_record["combined_case_wall_ms"] = round((phase_record["ended_monotonic_ns"] - phase_record["started_monotonic_ns"]) / 1_000_000, 3)
    closed = cold.get("completion", {}).get("socket_closed_monotonic_ns")
    sequence["client_close_to_retry_start_ms"] = None if closed is None else (retry["started_monotonic_ns"] - closed) / 1_000_000
    sequence["cold_outcome"] = "cancelled" if cold.get("cancelled") else ("completed_before_cancel_deadline" if cold.get("protocol_status") == "accepted" else "failed_before_cancel")
    comparison = compare_retry(retry, row["id"], baseline)
    sequence["baseline_comparison"] = comparison
    sequence["identity"] = {
        "cold_prompt_ids_sha256": cold.get("prompt_ids_sha256"),
        "retry_prompt_ids_sha256": retry.get("prompt_ids_sha256"),
        "cold_retry_full_prompt_exact": cold.get("prompt_ids_sha256") == retry.get("prompt_ids_sha256") == sequence["expected_prompt_ids_sha256"],
        "tokenized_cold_exact": cold.get("tokenize", {}).get("prompt_id_status") == "exact_stored_prompt",
        "tokenized_retry_exact": retry.get("tokenize", {}).get("prompt_id_status") == "exact_stored_prompt",
    }
    retry["started_monotonic_ns"] = retry.get("started_monotonic_ns", retry_started_ns)
    baseline_ok = comparison["status"] in {"pass", "baseline_not_provided"}
    identity_ok = bool(sequence["identity"]["cold_retry_full_prompt_exact"]
                       and sequence["identity"]["tokenized_cold_exact"]
                       and sequence["identity"]["tokenized_retry_exact"])
    if ((cold.get("cancelled") or cold.get("protocol_status") == "accepted") and retry.get("protocol_status") == "accepted" and identity_ok
            and baseline_ok):
        sequence["protocol_status"] = "completed" if comparison["status"] == "pass" else "completed_pending_baseline"
    else:
        sequence["protocol_status"] = "failed"
    sequence["ended_monotonic_ns"] = time.monotonic_ns()
    sequence["sequence_wall_ms"] = round((sequence["ended_monotonic_ns"] - sequence_started_ns) / 1_000_000, 3)
    return sequence


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    base_url = native.validate_origin(args.url)
    deadlines = parse_deadlines(args.cancel_deadlines_ms)
    fixture_path = Path(args.fixture)
    native_manifest_path = Path(args.native_manifest)
    transition_manifest_path = Path(args.manifest)
    rows, manifest, native_manifest = load_inputs(fixture_path, native_manifest_path, transition_manifest_path)
    validate_plan(args.batch, deadlines, args.retry_deadline_ms, args.cap, args.context,
                  args.diagnostic_timeout_ms, len(rows))
    baseline = _load_baseline(Path(args.baseline_results) if args.baseline_results else None,
                              args.baseline_phase, rows)
    started_ns = time.monotonic_ns()
    budget_deadline_ns = started_ns + MAX_TOTAL_MS * 1_000_000
    sequences: list[dict[str, Any]] = []
    for row in rows:
        for cancel_ms in deadlines:
            if len(sequences) >= MAX_SEQUENCES:
                raise TransitionProbeError("sequence_limit", "more than eight sequences requested")
            if _remaining_ms(budget_deadline_ns) <= 0:
                raise TransitionProbeError("total_budget_exhausted", "600 s total probe budget elapsed")
            sequences.append(run_transition(base_url, row, args.batch, cancel_ms, args.cap, args.context,
                                             args.retry_deadline_ms, args.diagnostic_timeout_ms,
                                             baseline, budget_deadline_ns))
            write_json(Path(args.out).with_suffix(".partial.json"), {"schema_version": SCHEMA_VERSION, "status": "partial", "sequences": sequences, "expected_sequences": 8})
    counts: dict[str, int] = {}
    for sequence in sequences:
        status = sequence.get("protocol_status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    all_completed = all(item.get("protocol_status") == "completed" for item in sequences)
    pending_baseline = all(item.get("protocol_status") == "completed_pending_baseline" for item in sequences)
    status = "completed" if all_completed else ("completed_pending_baseline" if pending_baseline else "failed")
    ended_ns = time.monotonic_ns()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": base_url,
        "server_arm": {"batch": args.batch, "batch_is_external_server_configuration": True},
        "native_probe": {"path": str(NATIVE_PROBE_PATH), "sha256": NATIVE_PROBE_SHA256,
                          "expected_sha256": EXPECTED_NATIVE_PROBE_SHA256,
                          "import_hash_checked": True},
        "fixture": {"path": str(fixture_path), "sha256": sha256_file(fixture_path),
                     "native_manifest_path": str(native_manifest_path),
                     "native_manifest_sha256": sha256_file(native_manifest_path),
                     "transition_manifest_path": str(transition_manifest_path),
                     "transition_manifest_sha256": sha256_file(transition_manifest_path),
                     "schema_version": native_manifest.get("schema_version"),
                     "selected_row_ids": [row["id"] for row in rows],
                     "row_count": len(rows), "split": "train"},
        "tokenizer": manifest.get("tokenizer"),
        "contract": {
            "cancel_deadlines_ms": list(deadlines),
            "retry_deadline_ms": args.retry_deadline_ms,
            "total_budget_ms": MAX_TOTAL_MS,
            "max_sequences": MAX_SEQUENCES,
            "cap": args.cap,
            "context": args.context,
            "cold_cancel": "close client SSE socket at whole deadline; server acknowledgement is not inferred",
            "retry": "same full prompt IDs including manual BOS, cache_prompt=true",
            "diagnostic": "No health request is inserted between cancellation and retry; source queue delay requires external log correlation.",
        },
        "baseline": {key: value for key, value in baseline.items() if key != "records"},
        "summary": {
            "row_count": len(rows), "sequence_count": len(sequences),
            "protocol_status_counts": counts,
            "accepted_retries": sum(item.get("retry", {}).get("protocol_status") == "accepted" for item in sequences if isinstance(item.get("retry"), dict)),
            "cancelled_cold": sum(item.get("cold", {}).get("cancelled") is True for item in sequences),
            "baseline_passes": sum(item.get("baseline_comparison", {}).get("status") == "pass" for item in sequences),
            "prompt_token_counts": {row["id"]: row["prompt_token_count"] for row in rows},
            "total_wall_ms": round((ended_ns - started_ns) / 1_000_000, 3),
            "real_editor_transitions": manifest.get("real_editor_transitions"),
        },
        "sequences": sequences,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name+".", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def parser() -> argparse.ArgumentParser:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18401")
    p.add_argument("--batch", type=int, choices=ALLOWED_BATCHES, required=True,
                   help="batch/ubatch configured on the externally launched server")
    p.add_argument("--fixture", default=str(here / "native-probe-train-fixture.jsonl"))
    p.add_argument("--native-manifest", default=str(here / "native-probe-train-fixture.manifest.json"))
    p.add_argument("--manifest", default=str(here / "runtime-transition-probe.manifest.json"))
    p.add_argument("--baseline-results", help="accepted native v2 result produced with a >=60 s deadline")
    p.add_argument("--baseline-phase", choices=("cold", "warm"), default="cold")
    p.add_argument("--cancel-deadlines-ms", default="1000,5000")
    p.add_argument("--retry-deadline-ms", type=int, default=MAX_RETRY_MS)
    p.add_argument("--diagnostic-timeout-ms", type=int, default=MAX_DIAGNOSTIC_MS)
    p.add_argument("--cap", type=int, default=native.MAX_CAP)
    p.add_argument("--context", type=int, default=native.MAX_CONTEXT)
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run_plan(args)
        write_json(Path(args.out), result)
    except TransitionProbeError as exc:
        result = {"schema_version": SCHEMA_VERSION, "status": "failed",
                  "error_code": exc.code, "error": exc.message}
        write_json(Path(args.out), result)
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result.get("status"), "out": args.out,
                      "summary": result.get("summary")}, sort_keys=True))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
