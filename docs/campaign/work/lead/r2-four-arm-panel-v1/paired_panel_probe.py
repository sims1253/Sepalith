#!/usr/bin/env python3
"""Portable 4096-context paired serving client for the TRAIN panel.

This is a client only.  It never starts a model server or loads a model.  It
implements the native wire contract used by the small screen while admitting
the panel's natural 2049..4096 prompt band: explicit no-special tokenization,
manual BOS=0 exactly once, greedy completion with n_predict<=192, native
EOS/EOG validation, and exact streamed token/text checks.  A server response
is accepted only when its generated IDs and protocol framing are valid; no
source target tail is used to repair a response.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import socket
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = (1, 130073)
PAD_ID = 1
VOCAB_SIZE = 130560
MAX_CAP = 192
MAX_CONTEXT = 4096
TERMINAL = ">>>>>>> UPDATED"
NO_EDIT = "[NO_EDIT]"
RESERVED_BODY_LINES = {
    "<<<<<<< CURRENT", "=======", TERMINAL, "<[fim-middle]>",
    "<[fim-prefix]>", "<[fim-suffix]>", "<|user_cursor|>",
    "<|outline|>", NO_EDIT,
}


class ProbeError(Exception):
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
    return (value if isinstance(value, str) else repr(value)).replace("\x00", "")[:limit]


def is_native_control_token(token: int) -> bool:
    return ((0 <= token <= 7) or (10 <= token <= 21) or
            (130072 <= token < VOCAB_SIZE))


def id_array(value: object, name: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ProbeError("invalid_panel", f"{name} must be an ID array")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0
           or item >= VOCAB_SIZE for item in value):
        raise ProbeError("invalid_panel", f"{name} contains an invalid token ID")
    return list(value)


def validate_panel_row(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ProbeError("invalid_panel", "panel row must be an object")
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
        raise ProbeError("invalid_panel", f"row {row.get('id')!r} missing {missing}")
    if row["split"] != "train":
        raise ProbeError("invalid_panel", f"row {row['id']} is not TRAIN")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ProbeError("invalid_panel", "row id must be a nonempty string")
    if row["bos_token_id"] != BOS_ID or row["eos_token_id"] != EOS_ID:
        raise ProbeError("invalid_panel", f"row {row['id']} has non-native BOS/EOS")
    if row["target_operation"] not in {"no_op", "replace", "delete"}:
        raise ProbeError("invalid_panel", f"row {row['id']} has invalid operation")
    if not isinstance(row["prompt_text"], str) or not row["prompt_text"]:
        raise ProbeError("invalid_panel", f"row {row['id']} prompt is empty")
    for field in ("target_start", "target_body_token_count", "target_terminal_token_count",
                  "prompt_token_count", "target_token_count"):
        if (not isinstance(row[field], int) or isinstance(row[field], bool)
                or row[field] < 0):
            raise ProbeError("invalid_panel", f"row {row['id']} has invalid {field}")
    input_ids = id_array(row["input_ids"], "input_ids", allow_empty=False)
    body_ids = id_array(row["target_body_tokens"], "target_body_tokens")
    terminal_ids = id_array(row["target_terminal_tokens"], "target_terminal_tokens", allow_empty=False)
    start = row["target_start"]
    if start != row["prompt_token_count"] + 1:
        raise ProbeError("invalid_panel", f"row {row['id']} target_start/BOS mismatch")
    if input_ids[0] != BOS_ID or input_ids.count(BOS_ID) != 1:
        raise ProbeError("invalid_panel", f"row {row['id']} must have one BOS at start")
    if input_ids[-1] != EOS_ID or input_ids.count(EOS_ID) != 1:
        raise ProbeError("invalid_panel", f"row {row['id']} must have one EOS at end")
    terminal_start = start + len(body_ids)
    if input_ids[start:terminal_start] != body_ids:
        raise ProbeError("invalid_panel", f"row {row['id']} body IDs mismatch")
    if input_ids[terminal_start:terminal_start + len(terminal_ids)] != terminal_ids:
        raise ProbeError("invalid_panel", f"row {row['id']} terminal IDs mismatch")
    if len(input_ids) != terminal_start + len(terminal_ids) + 1:
        raise ProbeError("invalid_panel", f"row {row['id']} input geometry mismatch")
    if row["target_body_token_count"] != len(body_ids) or row["target_terminal_token_count"] != len(terminal_ids):
        raise ProbeError("invalid_panel", f"row {row['id']} target count mismatch")
    if row["target_token_count"] != len(body_ids) + len(terminal_ids):
        raise ProbeError("invalid_panel", f"row {row['id']} target total mismatch")
    if row["target_token_count"] + 1 > MAX_CAP:
        raise ProbeError("invalid_panel", f"row {row['id']} target plus protocol EOS exceeds cap")
    # The completion prompt prepends manual BOS=0, so the context budget is
    # prompt IDs + one BOS + the inclusive generated-ID cap.
    if row["prompt_token_count"] + 1 + MAX_CAP > MAX_CONTEXT:
        raise ProbeError("invalid_panel", f"row {row['id']} prompt plus cap exceeds 4096 context")
    prompt_ids = input_ids[1:start]
    if len(prompt_ids) != row["prompt_token_count"]:
        raise ProbeError("invalid_panel", f"row {row['id']} prompt count mismatch")
    if any(is_native_control_token(token) for token in prompt_ids):
        raise ProbeError("invalid_panel", f"row {row['id']} prompt contains native control ID")
    if not isinstance(row["target_text"], str) or not row["target_text"].endswith(TERMINAL):
        raise ProbeError("invalid_panel", f"row {row['id']} target lacks exact terminal")
    return row


def load_panel(panel_path: Path, manifest_path: Path, row_limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError("invalid_manifest", short_text(exc)) from exc
    if manifest.get("schema_version") != "sepalith.r2.serving.train-panel.v1":
        raise ProbeError("invalid_manifest", "panel manifest schema mismatch")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("split_required") != "train":
        raise ProbeError("invalid_manifest", "manifest does not pin TRAIN source")
    if source.get("sha256") != source.get("expected_sha256"):
        raise ProbeError("invalid_manifest", "manifest source SHA pin is inconsistent")
    panel = manifest.get("panel")
    records = panel.get("records") if isinstance(panel, dict) else None
    if not isinstance(records, list):
        raise ProbeError("invalid_manifest", "manifest panel records are missing")
    raw_lines = panel_path.read_bytes().splitlines(keepends=True)
    if row_limit is not None:
        if row_limit <= 0:
            raise ProbeError("invalid_request", "row limit must be positive")
        raw_lines = raw_lines[:row_limit]
        records = records[:row_limit]
    if len(raw_lines) != len(records):
        raise ProbeError("invalid_manifest", "panel rows and manifest records differ")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw, identity in zip(raw_lines, records):
        try:
            row = validate_panel_row(json.loads(raw))
        except (json.JSONDecodeError, ProbeError) as exc:
            if isinstance(exc, ProbeError):
                raise
            raise ProbeError("invalid_panel", short_text(exc)) from exc
        row_id = row["id"]
        if row_id in seen:
            raise ProbeError("invalid_manifest", f"duplicate panel row {row_id}")
        seen.add(row_id)
        if (not isinstance(identity, dict) or identity.get("row_id") != row_id
                or identity.get("split") != "train"
                or identity.get("row_sha256") != sha256_bytes(raw)):
            raise ProbeError("invalid_manifest", f"panel identity mismatch for {row_id}")
        if identity.get("prompt_token_count") != row["prompt_token_count"]:
            raise ProbeError("invalid_manifest", f"panel prompt count mismatch for {row_id}")
        rows.append(row)
    if not rows:
        raise ProbeError("invalid_panel", "panel is empty")
    return rows, manifest


def validate_origin(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment or parts.path not in {"", "/"}:
        raise ProbeError("invalid_request", "url must be an HTTP origin without a path")
    return f"{parts.scheme}://{parts.netloc}"


def request_json(url: str, method: str, body: object | None, timeout_s: float) -> tuple[int, object, float]:
    data = None if body is None else canonical_json(body)
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    started = time.monotonic()
    try:
        with urlopen(Request(url, data=data, method=method, headers=headers), timeout=timeout_s) as response:
            status = int(response.getcode())
            raw = response.read()
    except HTTPError as exc:
        raise ProbeError("http_error", f"HTTP {exc.code}", status=int(exc.code)) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProbeError("timeout", "JSON request timed out", timed_out=True) from exc
    except URLError as exc:
        timed_out = isinstance(exc.reason, (socket.timeout, TimeoutError))
        raise ProbeError("timeout" if timed_out else "network_error", short_text(exc), timed_out=timed_out) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ProbeError("network_error", short_text(exc)) from exc
    elapsed = (time.monotonic() - started) * 1000.0
    if status < 200 or status >= 300:
        raise ProbeError("http_error", f"HTTP {status}", status=status)
    try:
        return status, json.loads(raw.decode("utf-8")), elapsed
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("invalid_response", short_text(exc), status=status) from exc


def _set_socket_timeout(response: object, timeout_s: float) -> None:
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


def stream_completion(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = Request(url, data=canonical_json(body), method="POST",
                      headers={"Accept": "text/event-stream", "Content-Type": "application/json"})
    started = time.monotonic()
    text_parts: list[str] = []
    token_ids: list[int] = []
    final: dict[str, Any] = {}
    malformed = False
    token_chunks = 0
    final_token_count = 0
    ttft_ms: float | None = None
    result: dict[str, Any] = {"text": "", "returned_token_ids": [], "final": {},
                              "ttft_ms": None, "wall_ms": None, "malformed_sse": False,
                              "token_chunk_count": 0, "final_token_count": 0}
    try:
        with urlopen(request, timeout=timeout_s) as response:
            while True:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise ProbeError("timeout", "completion stream timed out", timed_out=True)
                _set_socket_timeout(response, remaining)
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
                    malformed = True
                    continue
                if not isinstance(chunk, dict):
                    malformed = True
                    continue
                content = chunk.get("content", "")
                if not isinstance(content, str):
                    raise ProbeError("invalid_response", "stream content is not text")
                if content and ttft_ms is None:
                    ttft_ms = (time.monotonic() - started) * 1000.0
                text_parts.append(content)
                if "tokens" in chunk:
                    values = id_array(chunk["tokens"], "stream.tokens")
                    token_chunks += 1
                    if chunk.get("stop") is True:
                        final_token_count = len(values)
                    else:
                        token_ids.extend(values)
                for key in ("stop", "stop_type", "stopping_word", "truncated",
                            "tokens_evaluated", "tokens_predicted", "tokens_cached",
                            "n_tokens_cached", "n_prompt_tokens_cache", "timings"):
                    if key in chunk:
                        final[key] = chunk[key]
                if chunk.get("stop") is True:
                    break
    except ProbeError as exc:
        result.update({"error_code": exc.code, "error": short_text(exc),
                       "http_status": exc.status, "timed_out": exc.timed_out})
    except HTTPError as exc:
        result.update({"error_code": "http_error", "error": f"HTTP {exc.code}",
                       "http_status": int(exc.code), "timed_out": False})
    except (socket.timeout, TimeoutError) as exc:
        result.update({"error_code": "timeout", "error": short_text(exc), "timed_out": True})
    except URLError as exc:
        timed_out = isinstance(exc.reason, (socket.timeout, TimeoutError))
        result.update({"error_code": "timeout" if timed_out else "network_error",
                       "error": short_text(exc), "timed_out": timed_out})
    except (OSError, http.client.HTTPException) as exc:
        result.update({"error_code": "network_error", "error": short_text(exc), "timed_out": False})
    result.update({"text": "".join(text_parts), "returned_token_ids": token_ids,
                   "final": final, "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
                   "wall_ms": round((time.monotonic() - started) * 1000.0, 3),
                   "malformed_sse": malformed, "token_chunk_count": token_chunks,
                   "final_token_count": final_token_count})
    return result


def parse_wire_output(raw_text: object) -> dict[str, Any]:
    if not isinstance(raw_text, str) or "\r" in raw_text:
        return {"status": "invalid", "reason": "output_not_string_or_contains_cr"}
    lines = raw_text.split("\n")
    indices = [i for i, line in enumerate(lines) if line == TERMINAL]
    if len(indices) != 1 or any(line != "" for line in lines[indices[0] + 1:]):
        return {"status": "invalid", "reason": "terminal_missing_duplicate_or_followed_by_content"}
    body = lines[:indices[0]]
    canonical_body = [] if len(body) == 1 and body[0] == "" else body
    if canonical_body != [NO_EDIT] and any(line.strip() in RESERVED_BODY_LINES for line in canonical_body):
        return {"status": "invalid", "reason": "reserved_full_body_line"}
    operation = "no_op" if canonical_body == [NO_EDIT] else ("delete" if not canonical_body else "replace")
    return {"status": "accepted", "operation": operation, "body_text": "\n".join(canonical_body)}


def run_case(base_url: str, row: dict[str, Any], phase: str, rep: int,
             cap: int, context: int, deadline_ms: int) -> dict[str, Any]:
    started = time.monotonic()
    expected_prompt = row["input_ids"][1:row["target_start"]]
    record: dict[str, Any] = {
        "row_id": row["id"], "phase": phase, "rep": rep,
        "cache_prompt": phase == "warm", "prompt_token_count": row["prompt_token_count"],
        "prompt_ids_sha256": sha256_bytes(canonical_json(expected_prompt)),
        "prompt_sha256": sha256_text(row["prompt_text"]), "context_size": context,
        "cap": cap, "target_operation": row["target_operation"],
        "target_token_count": row["target_token_count"],
        "target_token_count_excludes_protocol_eos": True,
        "protocol_status": "pending",
    }

    def finish() -> dict[str, Any]:
        record["combined_case_wall_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        return record

    if len(expected_prompt) + 1 + cap > context:
        record.update({"protocol_status": "context_budget", "error": "prompt plus cap exceeds context"})
        return finish()
    def remaining() -> float:
        value = deadline_ms / 1000.0 - (time.monotonic() - started)
        if value <= 0:
            raise ProbeError("timeout", "combined tokenize/completion deadline elapsed", timed_out=True)
        return value
    try:
        status, tokenized, elapsed = request_json(
            f"{base_url}/tokenize", "POST",
            {"content": row["prompt_text"], "add_special": False,
             "parse_special": False, "with_pieces": False}, remaining())
        record.update({"tokenize_http_status": status, "tokenize_wall_ms": round(elapsed, 3)})
        if not isinstance(tokenized, dict):
            raise ProbeError("invalid_response", "tokenize response is not an object")
        native_tokens = id_array(tokenized.get("tokens"), "tokenize.tokens")
        if any(is_native_control_token(token) for token in native_tokens):
            raise ProbeError("prompt_control_token", "tokenize returned a native control token")
        record["prompt_id_status"] = "exact_stored_prompt" if native_tokens == expected_prompt else "mismatch"
        record["tokenized_ids_count"] = len(native_tokens)
        record["tokenized_ids_sha256"] = sha256_bytes(canonical_json(native_tokens))
        if native_tokens != expected_prompt:
            record.update({"protocol_status": "prompt_token_mismatch", "error": "tokenize IDs differ from TRAIN row"})
            return finish()
        streamed = stream_completion(
            f"{base_url}/completion",
            {"prompt": [BOS_ID, *expected_prompt], "n_predict": cap,
             "temperature": 0, "stream": True, "cache_prompt": phase == "warm",
             "return_tokens": True}, remaining())
    except ProbeError as exc:
        record.update({"protocol_status": "timeout" if exc.timed_out else exc.code,
                       "error": short_text(exc), "timed_out": exc.timed_out, "http_status": exc.status})
        return finish()
    record.update({"completion_wall_ms": streamed["wall_ms"], "ttft_ms": streamed["ttft_ms"],
                   "returned_token_ids": streamed["returned_token_ids"],
                   "returned_token_count": len(streamed["returned_token_ids"]),
                   "raw_text": streamed["text"], "raw_text_sha256": sha256_text(streamed["text"]),
                   "sse_malformed": streamed["malformed_sse"], "stream_token_chunk_count": streamed["token_chunk_count"],
                   "stream_final_token_count": streamed["final_token_count"], "final": streamed["final"]})
    if "error_code" in streamed:
        record.update({"protocol_status": "timeout" if streamed.get("timed_out") else streamed["error_code"],
                       "error": streamed.get("error"), "timed_out": bool(streamed.get("timed_out"))})
        return finish()
    final = streamed["final"]
    returned = streamed["returned_token_ids"]
    checks: dict[str, str] = {}
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
        checks["prompt_length"] = "exact" if final["tokens_evaluated"] == len([BOS_ID, *expected_prompt]) else "mismatch"
    else:
        checks["prompt_length"] = "missing"
    checks["stop"] = "eos" if final.get("stop_type") == "eos" and final.get("stop") is True else "unexpected"
    checks["output_parser"] = parse_wire_output(streamed["text"])["status"]
    checks["stream_layout"] = "partial_chunks_final_empty" if streamed["final_token_count"] == 0 else "unexpected_final_tokens"
    record["protocol_checks"] = checks
    record["parsed_output"] = parse_wire_output(streamed["text"])
    record["protocol_status"] = "accepted" if (
        checks == {
            "returned_ids": "valid_eos1", "prompt_length": "exact", "stop": "eos",
            "output_parser": "accepted", "stream_layout": "partial_chunks_final_empty"
        } and not streamed["malformed_sse"]
    ) else "protocol_error"
    record["timings_status"] = "present" if isinstance(final.get("timings"), dict) else "missing"
    if isinstance(final.get("timings"), dict):
        record["draft_n"] = final["timings"].get("draft_n")
        record["draft_n_accepted"] = final["timings"].get("draft_n_accepted")
    return finish()


def _pctl(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * p
    lo, hi = int(rank), min(len(values) - 1, int(rank) + 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    base_url = validate_origin(args.url)
    if args.cap <= 0 or args.cap > MAX_CAP or args.context <= 0 or args.context > MAX_CONTEXT:
        raise ProbeError("invalid_request", "cap/context exceed native bounds")
    rows, manifest = load_panel(Path(args.panel), Path(args.manifest), args.row_limit)
    requests: list[dict[str, Any]] = []
    for row in rows:
        for rep in range(1, args.reps + 1):
            requests.append(run_case(base_url, row, "cold", rep, args.cap, args.context, args.deadline_ms))
            requests.append(run_case(base_url, row, "warm", rep, args.cap, args.context, args.deadline_ms))
    accepted = sum(record.get("protocol_status") == "accepted" for record in requests)
    walls = [float(record["combined_case_wall_ms"]) for record in requests]
    draft_pairs = [
        (record["draft_n"], record["draft_n_accepted"])
        for record in requests
        if isinstance(record.get("draft_n"), int)
        and isinstance(record.get("draft_n_accepted"), int)
    ]
    complete_draft_counters = len(draft_pairs) == len(requests)
    result: dict[str, Any] = {
        "schema_version": "sepalith.r2.serving.train-panel-probe.v1",
        "status": "completed" if accepted == len(requests) else "partial",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": base_url, "panel": {"path": str(Path(args.panel)), "manifest": str(Path(args.manifest)),
                                     "manifest_sha256": sha256_bytes(Path(args.manifest).read_bytes()),
                                     "source_sha256": manifest["source"]["sha256"], "rows": len(rows),
                                     "row_limit": args.row_limit},
        "native_contract": {"bos_id": BOS_ID, "canonical_eos_id": EOS_ID,
                            "native_eog_ids": list(NATIVE_EOG_IDS), "pad_id": PAD_ID,
                            "context_size": args.context, "cap": args.cap,
                            "greedy": True, "tokenize_flags": {"add_special": False, "parse_special": False,
                                                                    "with_pieces": False},
                            "no_authored_target_tail": True},
        "summary": {"requests": len(requests), "accepted_requests": accepted,
                     "protocol_status_counts": {status: sum(record.get("protocol_status") == status for record in requests)
                                                 for status in sorted({str(record.get("protocol_status")) for record in requests})},
                     "combined_case_wall_ms": {"p50": _pctl(walls, .50), "p95": _pctl(walls, .95)},
                     "draft_counters": {"records_with_draft_fields": len(draft_pairs),
                                         "drafted_tokens": sum(pair[0] for pair in draft_pairs) if complete_draft_counters else None,
                                         "accepted_tokens": sum(pair[1] for pair in draft_pairs) if complete_draft_counters else None,
                                         "status": "present_in_server_timings" if complete_draft_counters else "not_reported"}},
        "requests": requests,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cap", type=int, default=MAX_CAP)
    parser.add_argument("--context", type=int, default=MAX_CONTEXT)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--deadline-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.reps <= 0 or args.deadline_ms <= 0:
        parser.error("reps and deadline-ms must be positive")
    try:
        result = run_probe(args)
    except ProbeError as exc:
        result = {"schema_version": "sepalith.r2.serving.train-panel-probe.v1",
                  "status": "failed", "error_code": exc.code, "error": short_text(exc)}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "requests": result["summary"]["requests"], "accepted": result["summary"]["accepted_requests"], "out": args.out}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
