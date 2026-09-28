#!/usr/bin/env python3
"""Read-only native PRM-03 serving probe.

The probe is a client.  It never starts a server, loads a model, copies data,
or changes a remote host.  It mirrors NativeCampaignClient's important wire
contract: /tokenize uses explicit no-special flags, /completion receives
manual BOS 0 plus integer prompt IDs, and a successful generation ends with
canonical EOS 1.  The same fixture can be run once for baseline and once for
ngram-mod, then compared without contacting the server.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = "run-01.native-prm03-probe.v2"
FIXTURE_SCHEMA = "run-01.native-prm03-train-fixture.v1"
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = (1, 130073)
VOCAB_SIZE = 130560
TERMINAL = ">>>>>>> UPDATED"
NO_EDIT = "[NO_EDIT]"
MAX_CAP = 192
MAX_CONTEXT = 4096
MAX_RL_PROMPT = 2048
DEFAULT_DEADLINE_MS = 5000
DEFAULT_DIAGNOSTIC_MS = 60000
DEADLINE_CANCELLATION = (
    "client socket timeout closes the request; pinned b10453 observes connection "
    "closure through req.should_stop (server-side poll is approximately 1 s); "
    "cancellation latency is source-confirmed but not live-measured"
)
RESERVED_BODY_LINES = {
    "<<<<<<< CURRENT",
    "=======",
    ">>>>>>> UPDATED",
    "<[fim-middle]>",
    "<[fim-prefix]>",
    "<[fim-suffix]>",
    "<|user_cursor|>",
    "<|outline|>",
    NO_EDIT,
}


class ProbeError(Exception):
    """A bounded request or local protocol failure."""

    def __init__(self, code: str, message: str, *, status: int | None = None,
                 timed_out: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.timed_out = timed_out


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def short_text(value: object, limit: int = 300) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\x00", "")
    return text[:limit]


def is_native_control_token(token: int) -> bool:
    return ((0 <= token <= 7) or (10 <= token <= 21) or
            (130072 <= token < VOCAB_SIZE))


def id_array(value: object, name: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ProbeError("invalid_fixture", f"{name} must be a nonempty ID array")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0
           or item >= VOCAB_SIZE for item in value):
        raise ProbeError("invalid_fixture", f"{name} contains an invalid native token ID")
    return list(value)


def validate_fixture_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        raise ProbeError("invalid_fixture", "fixture row must be an object")
    required = {
        "id", "input_ids", "target_start", "target_body_tokens",
        "target_terminal_tokens", "target_body_token_count",
        "target_terminal_token_count", "family", "package_id", "renderer_id",
        "prompt_text", "target_text", "target_body_text", "target_operation",
        "prompt_token_count", "target_token_count", "bos_token_id", "eos_token_id",
        "tokenizer_revision", "tokenizer_json_sha256", "tokenization_policy", "split",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ProbeError("invalid_fixture", f"row {row.get('id')!r} missing {missing}")
    if row["split"] != "train":
        raise ProbeError("invalid_fixture", f"row {row['id']} is not TRAIN")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ProbeError("invalid_fixture", "row id must be nonempty")
    if row["bos_token_id"] != BOS_ID or row["eos_token_id"] != EOS_ID:
        raise ProbeError("invalid_fixture", f"row {row['id']} has non-native BOS/EOS IDs")
    if row["target_operation"] not in {"no_op", "replace", "delete"}:
        raise ProbeError("invalid_fixture", f"row {row['id']} has invalid operation")
    if not isinstance(row["prompt_text"], str) or not row["prompt_text"]:
        raise ProbeError("invalid_fixture", f"row {row['id']} prompt is empty")
    if not isinstance(row["target_text"], str) or not row["target_text"].endswith(TERMINAL):
        raise ProbeError("invalid_fixture", f"row {row['id']} target lacks exact terminal")
    for field in ("target_start", "target_body_token_count", "target_terminal_token_count",
                  "prompt_token_count", "target_token_count"):
        if (not isinstance(row[field], int) or isinstance(row[field], bool)
                or row[field] < 0):
            raise ProbeError("invalid_fixture", f"row {row['id']} has invalid {field}")
    input_ids = id_array(row["input_ids"], "input_ids", allow_empty=False)
    body_ids = id_array(row["target_body_tokens"], "target_body_tokens")
    terminal_ids = id_array(row["target_terminal_tokens"], "target_terminal_tokens",
                            allow_empty=False)
    start = row["target_start"]
    if start != row["prompt_token_count"] + 1:
        raise ProbeError("invalid_fixture", f"row {row['id']} target_start/BOS geometry mismatch")
    if input_ids[0] != BOS_ID or input_ids.count(BOS_ID) != 1:
        raise ProbeError("invalid_fixture", f"row {row['id']} must have exactly one BOS at start")
    if input_ids[-1] != EOS_ID or input_ids.count(EOS_ID) != 1:
        raise ProbeError("invalid_fixture", f"row {row['id']} must have exactly one EOS at end")
    if input_ids[start:start + len(body_ids)] != body_ids:
        raise ProbeError("invalid_fixture", f"row {row['id']} body IDs do not match input_ids")
    terminal_start = start + len(body_ids)
    if input_ids[terminal_start:terminal_start + len(terminal_ids)] != terminal_ids:
        raise ProbeError("invalid_fixture", f"row {row['id']} terminal IDs do not match input_ids")
    if len(input_ids) != terminal_start + len(terminal_ids) + 1:
        raise ProbeError("invalid_fixture", f"row {row['id']} input_ids length is inconsistent")
    if row["target_body_token_count"] != len(body_ids):
        raise ProbeError("invalid_fixture", f"row {row['id']} body count is inconsistent")
    if row["target_terminal_token_count"] != len(terminal_ids):
        raise ProbeError("invalid_fixture", f"row {row['id']} terminal count is inconsistent")
    if row["target_token_count"] != len(body_ids) + len(terminal_ids):
        raise ProbeError("invalid_fixture", f"row {row['id']} target count is inconsistent")
    if row["prompt_token_count"] > MAX_RL_PROMPT:
        raise ProbeError("invalid_fixture", f"row {row['id']} exceeds RL prompt admission 2048")
    if row["target_token_count"] + 1 > MAX_CAP:
        raise ProbeError("invalid_fixture", f"row {row['id']} target plus protocol EOS exceeds probe cap 192")
    prompt_ids = input_ids[1:start]
    if len(prompt_ids) != row["prompt_token_count"]:
        raise ProbeError("invalid_fixture", f"row {row['id']} prompt count is inconsistent")
    if any(token in (BOS_ID, EOS_ID) or is_native_control_token(token)
           for token in prompt_ids):
        raise ProbeError("invalid_fixture", f"row {row['id']} prompt contains native control IDs")
    return row


def validate_origin(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ProbeError("invalid_request", "url must be an HTTP origin without query or fragment")
    if parts.path not in {"", "/"}:
        raise ProbeError("invalid_request", "url must be an HTTP origin without a path prefix")
    return f"{parts.scheme}://{parts.netloc}"


def request_json(url: str, method: str, body: object | None,
                 timeout_s: float) -> tuple[int, object, float]:
    data = None if body is None else canonical_json(body)
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method, headers=headers)
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_s) as response:
            status = int(response.getcode())
            raw = response.read()
    except HTTPError as exc:
        detail = ""
        try:
            detail = short_text(exc.read(1024))
        except OSError:
            pass
        raise ProbeError("http_error", f"HTTP {exc.code}: {detail}",
                         status=int(exc.code)) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProbeError("timeout", f"{method} request exceeded {timeout_s * 1000:.0f} ms",
                         timed_out=True) from exc
    except URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise ProbeError("timeout", f"{method} request exceeded {timeout_s * 1000:.0f} ms",
                             timed_out=True) from exc
        raise ProbeError("network_error", short_text(exc.reason)) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ProbeError("network_error", short_text(exc)) from exc
    elapsed = (time.monotonic() - started) * 1000.0
    if status < 200 or status >= 300:
        raise ProbeError("http_error", f"HTTP {status}", status=status)
    try:
        return status, json.loads(raw.decode("utf-8")), elapsed
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("invalid_response", f"invalid JSON: {short_text(exc)}",
                         status=status) from exc


def set_socket_timeout(response: object, timeout_s: float) -> None:
    """Best-effort per-read timeout so a streaming response has a hard budget."""
    for candidate in (
        getattr(getattr(response, "fp", None), "raw", None),
        getattr(getattr(response, "fp", None), "_sock", None),
    ):
        sock = getattr(candidate, "_sock", candidate)
        if hasattr(sock, "settimeout"):
            try:
                sock.settimeout(max(0.001, timeout_s))
                return
            except OSError:
                continue


def diagnostic(base_url: str, max_ms: int) -> dict[str, object]:
    bounded_ms = min(max_ms, 60000)
    started = time.monotonic()
    result: dict[str, object] = {
        "request": "GET /health",
        "timeout_ms": bounded_ms,
        "status": "unknown",
    }
    try:
        status, body, elapsed = request_json(
            f"{base_url}/health", "GET", None, bounded_ms / 1000.0)
        result.update({"status": "ok", "http_status": status,
                       "elapsed_ms": round(elapsed, 3),
                       "body": short_text(body)})
    except ProbeError as exc:
        result.update({"status": exc.code, "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                       "message": short_text(exc.message),
                       "http_status": exc.status})
    return result


def stream_completion(url: str, body: dict[str, object], deadline_ms: int) -> dict[str, object]:
    request = Request(url, data=canonical_json(body), method="POST",
                      headers={"Accept": "text/event-stream", "Content-Type": "application/json"})
    started = time.monotonic()
    text_parts: list[str] = []
    token_ids: list[int] = []
    token_chunk_count = 0
    final_token_count = 0
    final: dict[str, object] = {}
    malformed_sse = False
    ttft_ms: float | None = None
    result: dict[str, object] = {"text": "", "returned_token_ids": [],
                                 "final": {}, "ttft_ms": None,
                                 "wall_ms": None, "malformed_sse": False,
                                 "token_chunk_count": 0, "final_token_count": 0}
    timeout_s = deadline_ms / 1000.0
    try:
        with urlopen(request, timeout=timeout_s) as response:
            if int(response.getcode()) < 200 or int(response.getcode()) >= 300:
                raise ProbeError("http_error", f"HTTP {response.getcode()}",
                                 status=int(response.getcode()))
            while True:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise ProbeError("timeout", f"stream exceeded {deadline_ms} ms",
                                     timed_out=True)
                set_socket_timeout(response, remaining)
                raw = response.readline()
                if raw == b"":
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    malformed_sse = True
                    continue
                if not isinstance(chunk, dict):
                    malformed_sse = True
                    continue
                content = chunk.get("content", "")
                if not isinstance(content, str):
                    raise ProbeError("invalid_response", "stream content must be a string")
                if content and ttft_ms is None:
                    ttft_ms = (time.monotonic() - started) * 1000.0
                text_parts.append(content)
                if "tokens" in chunk:
                    chunk_tokens = chunk["tokens"]
                    if not isinstance(chunk_tokens, list):
                        raise ProbeError("invalid_response", "stream tokens must be an array")
                    token_chunk_count += 1
                    for token in chunk_tokens:
                        if (not isinstance(token, int) or isinstance(token, bool)
                                or token < 0 or token >= VOCAB_SIZE):
                            raise ProbeError("invalid_token_id", "stream returned an invalid token ID")
                    # b10453 sends one-token arrays on partial SSE chunks, then
                    # an empty tokens array on the final stream response.  Keep
                    # final IDs out of the generated sequence so a future
                    # serializer variant cannot silently duplicate them.
                    if chunk.get("stop") is True:
                        final_token_count = len(chunk_tokens)
                    else:
                        token_ids.extend(chunk_tokens)
                for key in ("stop", "stop_type", "stopping_word", "truncated",
                            "tokens_evaluated", "tokens_predicted", "tokens_cached",
                            "n_tokens_cached", "n_prompt_tokens_cache", "timings", "final"):
                    if key in chunk:
                        final[key] = chunk[key]
                if chunk.get("stop") is True:
                    break
    except ProbeError as exc:
        result.update({"error_code": exc.code, "error": short_text(exc.message),
                       "http_status": exc.status, "timed_out": exc.timed_out})
    except HTTPError as exc:
        result.update({"error_code": "http_error", "error": f"HTTP {exc.code}",
                       "http_status": int(exc.code), "timed_out": False})
    except (socket.timeout, TimeoutError) as exc:
        result.update({"error_code": "timeout", "error": f"stream exceeded {deadline_ms} ms",
                       "timed_out": True})
    except URLError as exc:
        timed_out = isinstance(exc.reason, (socket.timeout, TimeoutError))
        result.update({"error_code": "timeout" if timed_out else "network_error",
                       "error": short_text(exc), "timed_out": timed_out})
    except (OSError, http.client.HTTPException) as exc:
        result.update({"error_code": "network_error", "error": short_text(exc),
                       "timed_out": False})
    result.update({"text": "".join(text_parts), "returned_token_ids": token_ids,
                   "final": final, "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
                   "wall_ms": round((time.monotonic() - started) * 1000.0, 3),
                   "malformed_sse": malformed_sse,
                   "token_chunk_count": token_chunk_count,
                   "final_token_count": final_token_count})
    return result


def parse_wire_output(raw_text: object) -> dict[str, object]:
    """The framing portion of PRM-03 parseOutput, independent of row targets."""
    if not isinstance(raw_text, str):
        return {"status": "invalid", "reason": "output_not_string"}
    if "\r" in raw_text:
        return {"status": "invalid", "reason": "wire_output_contains_cr"}
    lines = raw_text.split("\n")
    terminal_indices = [index for index, line in enumerate(lines) if line == TERMINAL]
    if not terminal_indices:
        return {"status": "invalid", "reason": "missing_exact_terminal"}
    if len(terminal_indices) != 1:
        return {"status": "invalid", "reason": "duplicate_exact_terminal"}
    terminal_index = terminal_indices[0]
    if any(line != "" for line in lines[terminal_index + 1:]):
        return {"status": "invalid", "reason": "content_after_terminal"}
    body = lines[:terminal_index]
    canonical_body = [] if len(body) == 1 and body[0] == "" else body
    if (canonical_body != [NO_EDIT]
            and any(line.strip() in RESERVED_BODY_LINES for line in canonical_body)):
        return {"status": "invalid", "reason": "reserved_full_body_line"}
    operation = "no_op" if canonical_body == [NO_EDIT] else ("delete" if not canonical_body else "replace")
    return {"status": "accepted", "reason": None, "operation": operation,
            "body_text": "\n".join(canonical_body)}


def load_fixture(fixture_path: Path, manifest_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError("invalid_manifest", short_text(exc)) from exc
    if manifest.get("schema_version") != FIXTURE_SCHEMA:
        raise ProbeError("invalid_manifest", "fixture manifest schema mismatch")
    expected_hash = manifest.get("fixture_sha256")
    actual_hash = sha256_bytes(fixture_path.read_bytes())
    if expected_hash != actual_hash:
        raise ProbeError("fixture_hash_mismatch", "fixture SHA-256 does not match manifest")
    rows: list[dict[str, object]] = []
    try:
        with fixture_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProbeError("invalid_fixture", f"line {line_number}: {short_text(exc)}") from exc
                rows.append(validate_fixture_row(row))
    except OSError as exc:
        raise ProbeError("invalid_fixture", short_text(exc)) from exc
    selected = manifest.get("selected_rows")
    if not isinstance(selected, list) or len(selected) != len(rows):
        raise ProbeError("invalid_manifest", "selected_rows count does not match fixture")
    for row, identity in zip(rows, selected):
        if not isinstance(identity, dict) or identity.get("row_id") != row["id"]:
            raise ProbeError("invalid_manifest", "selected row order or ID mismatch")
        if identity.get("row_sha256") != sha256_bytes(canonical_json(row)):
            raise ProbeError("fixture_hash_mismatch", f"row {row['id']} digest mismatch")
        if identity.get("sidecar_split") != "train" or identity.get("context_has_target_or_reward_keys") is not False:
            raise ProbeError("invalid_manifest", f"row {row['id']} sidecar admission is not TRAIN/context-only")
    return rows, manifest


def prompt_band(prompt_count: int) -> str:
    return "short" if prompt_count <= 1024 else "long"


def base_record(row: dict[str, object], arm: str, phase: str, rep: int,
                cap: int, context: int, deadline_ms: int, diagnostic_ms: int) -> dict[str, object]:
    input_ids = row["input_ids"]
    start = row["target_start"]
    prompt_ids = input_ids[:start]
    return {
        "arm": arm,
        "row_id": row["id"],
        "phase": phase,
        "rep": rep,
        "cache_prompt": phase == "warm",
        "ctx_band": prompt_band(row["prompt_token_count"]),
        "prompt_token_count": row["prompt_token_count"],
        "prompt_ids_count": len(prompt_ids),
        "prompt_ids_include_manual_bos": True,
        "prompt_ids_sha256": sha256_bytes(canonical_json(prompt_ids)),
        "prompt_sha256": sha256_text(row["prompt_text"]),
        "target_operation": row["target_operation"],
        "target_token_count": row["target_token_count"],
        "target_token_count_excludes_protocol_eos": True,
        "cap": cap,
        "context_size": context,
        "user_deadline_ms": deadline_ms,
        "diagnostic_max_ms": min(diagnostic_ms, 60000),
        "cap_includes_terminal_eos": True,
        "cap_status": "within_cap" if row["target_token_count"] + 1 <= cap else "fixture_target_over_cap",
        "deadline_scope": "combined tokenize plus completion for this case; diagnostic is outside the user deadline",
        "deadline_cancellation": DEADLINE_CANCELLATION,
        "protocol_status": "pending",
    }


def run_case(base_url: str, row: dict[str, object], arm: str, phase: str, rep: int,
             cap: int, context: int, deadline_ms: int, diagnostic_ms: int) -> dict[str, object]:
    record = base_record(row, arm, phase, rep, cap, context, deadline_ms, diagnostic_ms)
    case_started = time.monotonic()

    def mark_combined_deadline() -> None:
        record.setdefault("combined_case_wall_ms", round((time.monotonic() - case_started) * 1000.0, 3))
        record.setdefault("combined_deadline_remaining_ms", max(
            0, round(deadline_ms - (time.monotonic() - case_started) * 1000.0, 3)))

    def finish() -> dict[str, object]:
        mark_combined_deadline()
        return record

    expected_prompt = row["input_ids"][1:row["target_start"]]
    if len(expected_prompt) + cap > context:
        record.update({"protocol_status": "context_budget", "error": "prompt plus cap exceeds context"})
        return finish()
    tokenize_body = {"content": row["prompt_text"], "add_special": False,
                     "parse_special": False, "with_pieces": False}
    tokenize_timeout_ms = max(0, int(deadline_ms - (time.monotonic() - case_started) * 1000.0))
    record["tokenize_timeout_ms"] = tokenize_timeout_ms
    if tokenize_timeout_ms <= 0:
        record.update({"protocol_status": "timeout", "error": "combined deadline elapsed before /tokenize",
                       "timed_out": True})
        mark_combined_deadline()
        record["diagnostic"] = diagnostic(base_url, diagnostic_ms)
        return finish()
    try:
        status, tokenized, elapsed = request_json(
            f"{base_url}/tokenize", "POST", tokenize_body, tokenize_timeout_ms / 1000.0)
        record["tokenize_http_status"] = status
        record["tokenize_wall_ms"] = round(elapsed, 3)
        if not isinstance(tokenized, dict):
            raise ProbeError("invalid_response", "tokenize response must be an object")
        native_tokens = id_array(tokenized.get("tokens"), "tokenize.tokens", allow_empty=True)
        if any(is_native_control_token(token) for token in native_tokens):
            raise ProbeError("prompt_control_token", "tokenize response contains a native CONTROL token")
        record["prompt_id_status"] = "exact_stored_prompt" if native_tokens == expected_prompt else "mismatch"
        record["tokenized_ids_sha256"] = sha256_bytes(canonical_json(native_tokens))
        record["tokenized_ids_count"] = len(native_tokens)
        if native_tokens != expected_prompt:
            record.update({"protocol_status": "prompt_token_mismatch",
                           "error": "server /tokenize IDs differ from accepted TRAIN row"})
            return finish()
    except ProbeError as exc:
        record.update({"protocol_status": "timeout" if exc.timed_out else "tokenize_error",
                       "error": short_text(exc.message), "http_status": exc.status,
                       "timed_out": exc.timed_out})
        if exc.timed_out:
            mark_combined_deadline()
            record["diagnostic"] = diagnostic(base_url, diagnostic_ms)
        return finish()
    prompt_ids = [BOS_ID, *expected_prompt]
    completion_body = {
        "prompt": prompt_ids,
        "n_predict": cap,
        "temperature": 0,
        "stream": True,
        "cache_prompt": phase == "warm",
        "return_tokens": True,
    }
    completion_timeout_ms = max(0, int(deadline_ms - (time.monotonic() - case_started) * 1000.0))
    record["completion_timeout_ms"] = completion_timeout_ms
    if completion_timeout_ms <= 0:
        record.update({"protocol_status": "timeout", "error": "combined deadline elapsed before /completion",
                       "timed_out": True})
        mark_combined_deadline()
        record["diagnostic"] = diagnostic(base_url, diagnostic_ms)
        return finish()
    streamed = stream_completion(f"{base_url}/completion", completion_body, completion_timeout_ms)
    record.update({"completion_wall_ms": streamed["wall_ms"],
                   "ttft_ms": streamed["ttft_ms"],
                   "returned_token_ids": streamed["returned_token_ids"],
                   "returned_token_count": len(streamed["returned_token_ids"]),
                   "stream_token_chunk_count": streamed["token_chunk_count"],
                   "stream_final_token_count": streamed["final_token_count"],
                   "stream_final_tokens_expected_empty": True,
                   "raw_text": streamed["text"],
                   "raw_text_sha256": sha256_text(streamed["text"]),
                   "sse_malformed": streamed["malformed_sse"]})
    if "error_code" in streamed:
        record.update({"protocol_status": "timeout" if streamed.get("timed_out") else streamed["error_code"],
                       "error": streamed.get("error"), "http_status": streamed.get("http_status"),
                       "timed_out": bool(streamed.get("timed_out"))})
        if streamed.get("timed_out"):
            mark_combined_deadline()
            record["diagnostic"] = diagnostic(base_url, diagnostic_ms)
        return finish()
    final = streamed["final"]
    record["stop"] = final.get("stop")
    record["stop_type"] = final.get("stop_type")
    record["stopping_word"] = final.get("stopping_word")
    record["truncated"] = final.get("truncated")
    record["tokens_evaluated"] = final.get("tokens_evaluated")
    record["tokens_predicted"] = final.get("tokens_predicted")
    record["tokens_cached"] = final.get("tokens_cached")
    record["n_tokens_cached"] = final.get("n_tokens_cached")
    record["n_prompt_tokens_cache"] = final.get("n_prompt_tokens_cache")
    record["timings"] = final.get("timings")
    if isinstance(final.get("timings"), dict):
        record["timing_prompt_processed"] = final["timings"].get("prompt_n")
        record["timing_prompt_cached"] = final["timings"].get("cache_n")
        record["timing_predicted"] = final["timings"].get("predicted_n")
    checks: dict[str, str] = {}
    returned = streamed["returned_token_ids"]
    if not returned:
        checks["returned_ids"] = "missing"
    elif any(is_native_control_token(token) for token in returned[:-1]):
        checks["returned_ids"] = "early_control_token"
    elif returned[-1] != EOS_ID:
        checks["returned_ids"] = "noncanonical_eog" if returned[-1] in NATIVE_EOG_IDS else "missing_eos1"
    elif len(returned) > cap:
        checks["returned_ids"] = "over_cap"
    else:
        checks["returned_ids"] = "valid_eos1"
    if final.get("tokens_evaluated") is not None:
        checks["prompt_length"] = "exact" if final.get("tokens_evaluated") == len(prompt_ids) else "mismatch"
    if final.get("truncated") is True or final.get("stop_type") in {"limit", "length"}:
        checks["stop"] = "truncated"
    elif final.get("stop_type") == "eos":
        checks["stop"] = "eos"
    else:
        checks["stop"] = "unexpected"
    parsed = parse_wire_output(streamed["text"])
    checks["output_parser"] = parsed["status"]
    checks["stream_layout"] = ("partial_chunks_final_empty"
                                if streamed["final_token_count"] == 0
                                else "unexpected_final_tokens")
    record["protocol_checks"] = checks
    record["parsed_output"] = parsed
    record["cap_status"] = "within_cap" if len(returned) <= cap else "exceeded_cap"
    good = (checks.get("returned_ids") == "valid_eos1"
            and checks.get("prompt_length", "exact") == "exact"
            and checks.get("stop") == "eos"
            and checks.get("output_parser") == "accepted"
            and checks.get("stream_layout") == "partial_chunks_final_empty"
            and not streamed["malformed_sse"])
    record["protocol_status"] = "accepted" if good else "protocol_error"
    record["timings_status"] = "present" if isinstance(final.get("timings"), dict) else "missing"
    return finish()


def summarize(rows: list[dict[str, object]], requests: list[dict[str, object]],
             arm: str, cap: int, context: int, deadline_ms: int, diagnostic_ms: int) -> dict[str, object]:
    counts: dict[str, int] = {}
    for record in requests:
        status = str(record.get("protocol_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    prompt_counts = [int(row["prompt_token_count"]) for row in rows]
    return {
        "arm": arm,
        "fixture_rows": len(rows),
        "requests": len(requests),
        "cold_requests": sum(record["phase"] == "cold" for record in requests),
        "warm_requests": sum(record["phase"] == "warm" for record in requests),
        "protocol_status_counts": counts,
        "accepted_requests": counts.get("accepted", 0),
        "actual_prompt_band": {
            "min_tokens": min(prompt_counts),
            "max_tokens": max(prompt_counts),
            "bands": {"short_le_1024": sum(value <= 1024 for value in prompt_counts),
                      "long_1025_to_2048": sum(1025 <= value <= 2048 for value in prompt_counts)},
        },
        "prompt_admission": "RL-02 context-only TRAIN rows, prompt <= 2048",
        "training_context_admission_max": 4096,
        "cap": cap,
        "context_size": context,
        "user_deadline_ms": deadline_ms,
        "deadline_scope": "5000 ms combined tokenize plus completion per cold/warm case",
        "cap_includes_terminal_eos": True,
        "prompt_length_includes_manual_bos": True,
        "warm_cache_semantics": "tokens_evaluated is full task prompt length including BOS; timings.cache_n/prompt_n report cached/processed work",
        "deadline_cancellation": DEADLINE_CANCELLATION,
        "diagnostic_max_ms": min(diagnostic_ms, 60000),
        "eight_k_stress": {"status": "unsupported_pending",
                           "reason": "No synthetic or >2048-token prompt is admitted by this fixture."},
        "no_retry_policy": "one cold and one warm request per row; no automatic retry",
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    base_url = validate_origin(args.url)
    if args.cap > MAX_CAP or args.cap <= 0:
        raise ProbeError("invalid_request", "cap must be in 1..192")
    if args.context <= 0 or args.context > MAX_CONTEXT:
        raise ProbeError("invalid_request", "context must be in 1..4096")
    if args.user_deadline_ms <= 0 or args.diagnostic_timeout_ms <= 0:
        raise ProbeError("invalid_request", "timeouts must be positive")
    rows, manifest = load_fixture(Path(args.fixture), Path(args.manifest))
    requests: list[dict[str, object]] = []
    for row in rows:
        for rep in range(1, args.reps + 1):
            requests.append(run_case(base_url, row, args.arm, "cold", rep,
                                     args.cap, args.context, args.user_deadline_ms,
                                     args.diagnostic_timeout_ms))
            requests.append(run_case(base_url, row, args.arm, "warm", rep,
                                     args.cap, args.context, args.user_deadline_ms,
                                     args.diagnostic_timeout_ms))
    statuses = {record.get("protocol_status") for record in requests}
    overall = "completed" if statuses == {"accepted"} else ("no_server" if statuses and statuses <= {"network_error", "timeout"} else "partial")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": base_url,
        "arm": args.arm,
        "fixture": {"path": str(Path(args.fixture)),
                     "sha256": sha256_bytes(Path(args.fixture).read_bytes()),
                     "manifest_path": str(Path(args.manifest)),
                     "manifest_sha256": sha256_bytes(Path(args.manifest).read_bytes()),
                     "schema_version": manifest.get("schema_version")},
        "native_contract": {"bos_id": BOS_ID, "canonical_eos_id": EOS_ID,
                            "native_eog_ids": list(NATIVE_EOG_IDS), "vocab_size": VOCAB_SIZE,
                            "tokenize_flags": {"add_special": False, "parse_special": False,
                                                "with_pieces": False},
                            "stream_tokens": "b10453 emits one-token partial SSE arrays including EOS; final stream tokens array is empty",
                            "return_tokens": "requested for client parity; it does not populate streamed final tokens",
                            "tokens_evaluated": "full task prompt length including manual BOS, independent of warm cache",
                            "cap": "n_predict is an inclusive generated-ID budget; EOS consumes one returned ID"},
        "server_profile_expected": {"ctx": 4096, "batch": 256, "parallel": 1,
                                     "ngl": 0, "host": "m0pad", "runtime_isolation": "m0pad CPU"},
        "summary": summarize(rows, requests, args.arm, args.cap, args.context,
                              args.user_deadline_ms, args.diagnostic_timeout_ms),
        "requests": requests,
    }


def compare_results(baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError("invalid_compare", short_text(exc)) from exc
    def index(result: object) -> dict[tuple[object, object, object], dict[str, object]]:
        if not isinstance(result, dict) or not isinstance(result.get("requests"), list):
            raise ProbeError("invalid_compare", "result lacks request records")
        out = {}
        for record in result["requests"]:
            if not isinstance(record, dict):
                raise ProbeError("invalid_compare", "request record is not an object")
            key = (record.get("row_id"), record.get("phase"), record.get("rep"))
            out[key] = record
        return out
    left, right = index(baseline), index(candidate)
    keys = sorted(set(left) | set(right), key=repr)
    mismatches: list[dict[str, object]] = []
    comparable = 0
    for key in keys:
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            mismatches.append({"key": list(key), "reason": "missing_pair"})
            continue
        if a.get("protocol_status") != "accepted" or b.get("protocol_status") != "accepted":
            mismatches.append({"key": list(key), "reason": "unaccepted_pair",
                               "baseline_status": a.get("protocol_status"),
                               "candidate_status": b.get("protocol_status")})
            continue
        comparable += 1
        if (a.get("raw_text") != b.get("raw_text") or
                a.get("returned_token_ids") != b.get("returned_token_ids")):
            mismatches.append({"key": list(key), "reason": "exact_output_or_token_id_mismatch",
                               "baseline_text_sha256": a.get("raw_text_sha256"),
                               "candidate_text_sha256": b.get("raw_text_sha256"),
                               "baseline_token_ids": a.get("returned_token_ids"),
                               "candidate_token_ids": b.get("returned_token_ids")})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if comparable == len(keys) and not mismatches else "fail",
        "comparison": "exact baseline/ngram-mod raw text and returned token IDs",
        "baseline": {"path": str(baseline_path), "sha256": sha256_bytes(baseline_path.read_bytes()),
                      "arm": baseline.get("arm")},
        "candidate": {"path": str(candidate_path), "sha256": sha256_bytes(candidate_path.read_bytes()),
                       "arm": candidate.get("arm")},
        "paired_requests": len(keys),
        "comparable_accepted_requests": comparable,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", help="JSONL fixture for a live probe")
    p.add_argument("--manifest", help="fixture manifest for a live probe")
    p.add_argument("--url", default="http://127.0.0.1:18401")
    p.add_argument("--arm", choices=("baseline", "ngram-mod"))
    p.add_argument("--cap", type=int, default=MAX_CAP)
    p.add_argument("--context", type=int, default=MAX_CONTEXT)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--user-deadline-ms", type=int, default=DEFAULT_DEADLINE_MS)
    p.add_argument("--diagnostic-timeout-ms", type=int, default=DEFAULT_DIAGNOSTIC_MS)
    p.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "NGRAM_JSON"))
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.compare:
            result = compare_results(Path(args.compare[0]), Path(args.compare[1]))
        else:
            if not args.fixture or not args.manifest or not args.arm:
                raise ProbeError("invalid_request", "--fixture, --manifest and --arm are required for a live probe")
            if args.reps <= 0:
                raise ProbeError("invalid_request", "reps must be positive")
            result = run_probe(args)
        write_json(Path(args.out), result)
    except ProbeError as exc:
        result = {"schema_version": SCHEMA_VERSION, "status": "failed",
                  "error_code": exc.code, "error": exc.message}
        write_json(Path(args.out), result)
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result.get("status"), "out": args.out,
                      "summary": result.get("summary", result)}, sort_keys=True))
    return 0 if result.get("status") in {"completed", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
