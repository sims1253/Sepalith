#!/usr/bin/env python3
"""Notebook-local RUN-03 b4 native baseline v5 client.

The client is deliberately small and target-free.  It imports the exact
protected ``render_zeta2`` implementation and ``parse_prediction`` helper
from the already reviewed b4 v2 client after checking their recorded hashes.
It performs one loopback preflight and six sequential ``/tokenize`` plus
``/completion`` cycles: three captured v4 geometries, each once as a first
changed prompt and once as an identical repeat.  Case names are evidence
labels only; no case or operation label enters either request body.

The root operator owns the server.  This program never starts, stops, or
supervises a server, never opens model weights, and never makes network calls
except to the explicitly supplied HTTP loopback URL during a live run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
HERE = Path(__file__).resolve().parent
FIXTURES_PATH = HERE / "fixtures.json"
PREVIOUS_CLIENT = HERE / 'deps/previous_client.py'
PREVIOUS_CLIENT_SHA256 = "2c229cf3c65c15984193fee80e30d515a917ed743c112502e33c53b92b526caa"
RENDERER_PATH = HERE / 'deps/renderer_path.py'
RENDERER_SHA256 = "7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d"
PARSER_PATH = HERE / 'deps/parser_path.ts'
PARSER_SHA256 = "368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285"
SFT_RECEIPT = HERE / 'deps/sft_receipt.json'
CPU_ADMISSION = HERE / 'deps/cpu_admission.json'
PRE08_RECEIPT = HERE / 'deps/pre08_receipt.json'
V4_REVIEW_RECEIPT = HERE / 'deps/v4_review_receipt.json'
RUNTIME_IDENTITY = HERE / 'deps/runtime_identity.json'

MODEL_PATH = "/home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf"
MODEL_BYTES = 2012011904
MODEL_SHA256 = "e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d"
RUNTIME_PATH = "/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server"
RUNTIME_SHA256 = "e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6"
BASE_URL = "http://127.0.0.1:18403"
CONTEXT_SIZE = 8192
MAX_TOKENS = 320
REQUEST_TIMEOUT_S = 5.0
EXTENSION_STOPS = [
    ">>>>>>> UPDATED",
    "<<<<<<< CURRENT",
    "=======",
    "<[fim-middle]>",
    "<[fim-suffix]>",
    "<[fim-prefix]>",
    "<|outline|>",
]
CAPTURED_PROMPT_SHA256 = {
    "replacement": "1e272ca5ae2e9c8a8b8b3bde9dd59d0ee1114c428ab91e1eab9cbe07754ce3b0",
    "no-op-applicability": "0dcaa3de21908f1f316cc2b45d171e538fb1d2e5e1600095ff0e6ab0b41cd35f",
    "post-cancellation-fresh": "6c0e8602214e1bedfdaecddc47d7434d8ce895e945ac58193ee6dd4fae19f9b1",
}
TOKENIZE_REQUEST = {"add_special": True, "parse_special": True, "with_pieces": False}
TASK = "RUN-03-b4-native-baseline-v5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_module(path: Path, expected_sha256: str, name: str):
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{name} hash mismatch: {actual} != {expected_sha256}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_contract() -> dict[str, Any]:
    """Check the exact accepted contract without reading model/runtime bytes."""
    previous_client = load_module(PREVIOUS_CLIENT, PREVIOUS_CLIENT_SHA256, "b4_v2_client")
    sft = read_json(SFT_RECEIPT)
    cpu = read_json(CPU_ADMISSION)
    pre08 = read_json(PRE08_RECEIPT)
    v4_review = read_json(V4_REVIEW_RECEIPT)
    runtime = read_json(RUNTIME_IDENTITY)
    accepted = sft["inputs"]
    source_renderer = accepted["legacy_sources"]["renderer"]
    source_parser = accepted["legacy_sources"]["parser"]
    request = sft["request_contract"]
    cpu_model = cpu["model"]
    cpu_runtime_identity = runtime["files"]
    runtime_binary = next(item for item in cpu_runtime_identity if item["path"] == runtime["binary"])
    bank = pre08["RESULT"]["matched_banked_artifact"]
    renderer = load_module(RENDERER_PATH, RENDERER_SHA256, "render_zeta2")
    if source_renderer["sha256"] != RENDERER_SHA256 or source_parser["sha256"] != PARSER_SHA256:
        raise RuntimeError("accepted renderer/parser receipt identity drifted")
    if sha256_file(PARSER_PATH) != PARSER_SHA256:
        raise RuntimeError("protected TypeScript parser source hash drifted")
    if cpu_model["bytes"] != MODEL_BYTES or cpu_model["sha256"] != MODEL_SHA256:
        raise RuntimeError("CPU admission model identity drifted")
    if bank["bytes"] != MODEL_BYTES or bank["sha256"] != MODEL_SHA256:
        raise RuntimeError("PRE-08 model identity drifted")
    if runtime_binary["sha256"] != RUNTIME_SHA256:
        raise RuntimeError("CPU runtime identity drifted")
    if previous_client.PRODUCTION_MAX_TOKENS != MAX_TOKENS:
        raise RuntimeError("previous client max token contract drifted")
    if previous_client.EXTENSION_STOPS != EXTENSION_STOPS:
        raise RuntimeError("previous client stop list drifted")
    if previous_client.TOKENIZE_REQUEST != TOKENIZE_REQUEST:
        raise RuntimeError("previous client tokenizer policy drifted")
    if request["production_all_rows"]["n_predict"] != MAX_TOKENS:
        raise RuntimeError("accepted request cap drifted")
    if request["production_all_rows"]["stop"] != EXTENSION_STOPS:
        raise RuntimeError("accepted request stop list drifted")
    if request["temperature"] != 0 or request["stream"] is not False or request["cache_prompt"] is not False:
        raise RuntimeError("accepted deterministic request settings drifted")
    if request["tokenizer_request"] != TOKENIZE_REQUEST:
        raise RuntimeError("accepted tokenizer request drifted")
    captured_prompt_hashes = v4_review.get("prompt_parity", {}).get("captured_prompt_sha256", {})
    if captured_prompt_hashes != CAPTURED_PROMPT_SHA256:
        raise RuntimeError("v4 captured prompt geometry hashes drifted")
    return {
        "previous_client": {"path": str(PREVIOUS_CLIENT), "sha256": PREVIOUS_CLIENT_SHA256},
        "renderer": {"path": str(RENDERER_PATH), "symbol": "render_zeta2", "sha256": RENDERER_SHA256},
        "parser": {"path": str(PARSER_PATH), "symbol": "parsePrediction", "sha256": PARSER_SHA256},
        "model": {"path": MODEL_PATH, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256},
        "runtime": {"path": RUNTIME_PATH, "sha256": RUNTIME_SHA256, "version": runtime.get("version")},
        "request": {
            "tokenize_endpoint": "/tokenize",
            "completion_endpoint": "/completion",
            "tokenize": dict(TOKENIZE_REQUEST),
            "n_predict": MAX_TOKENS,
            "temperature": 0,
            "stream": False,
            "cache_prompt": False,
            "stop": list(EXTENSION_STOPS),
            "context_size": CONTEXT_SIZE,
            "timeout_seconds": REQUEST_TIMEOUT_S,
        },
        "renderer_module": renderer,
        "parser_module": previous_client,
        "receipt_sha256": {
            "SFT-08": sha256_file(SFT_RECEIPT),
            "CPU-admission": sha256_file(CPU_ADMISSION),
            "PRE-08": sha256_file(PRE08_RECEIPT),
            "runtime-identity": sha256_file(RUNTIME_IDENTITY),
            "v4-independent-review": sha256_file(V4_REVIEW_RECEIPT),
        },
    }


def load_fixtures(renderer) -> list[dict[str, Any]]:
    raw = read_json(FIXTURES_PATH)
    cases = raw.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise RuntimeError("v5 fixture packet must contain exactly three cases")
    prepared = []
    for case in cases:
        required = ("id", "path", "document", "prefix", "region_old", "cursor_idx", "cursor_character", "suffix", "event_diff", "captured_prompt_sha256")
        if any(key not in case for key in required):
            raise RuntimeError(f"fixture missing required geometry field: {case.get('id')}")
        document_lines = case["document"].split("\n")
        if document_lines[-1] != "":
            raise RuntimeError(f"fixture document must retain trailing LF: {case['id']}")
        if document_lines[: len(case["prefix"])] != case["prefix"]:
            raise RuntimeError(f"fixture prefix does not match document: {case['id']}")
        region_line = len(case["prefix"])
        if document_lines[region_line] != case["region_old"][case["cursor_idx"]]:
            raise RuntimeError(f"fixture region does not match document: {case['id']}")
        if case["cursor_character"] != len(case["region_old"][case["cursor_idx"]]):
            raise RuntimeError(f"fixture cursor is not the captured UTF-16 line-end position: {case['id']}")
        legacy = {
            "suffix": list(case["suffix"]),
            "event_diff": case["event_diff"],
            "path": case["path"],
            "prefix": list(case["prefix"]),
            "region_old": list(case["region_old"]),
            "cursor_idx": case["cursor_idx"],
        }
        prompt = renderer.render_zeta2(legacy)
        prompt_sha = sha256_bytes(prompt.encode("utf-8"))
        if prompt_sha != case["captured_prompt_sha256"]:
            raise RuntimeError(f"captured renderer parity hash mismatch: {case['id']}: {prompt_sha}")
        prepared.append({"case": case, "legacy": legacy, "prompt": prompt, "prompt_sha256": prompt_sha})
    if len({item["prompt_sha256"] for item in prepared}) != 3:
        raise RuntimeError("v5 prompt hashes are not unique")
    return prepared


def base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("client accepts only an HTTP loopback origin")
    return value.rstrip("/")


def http_json(origin: str, endpoint: str, payload: dict[str, Any] | None, timeout_s: float) -> tuple[int, dict[str, Any]]:
    if timeout_s <= 0:
        raise TimeoutError(f"no residual timeout for {endpoint}")
    if payload is None:
        request = urllib.request.Request(origin + endpoint, method="GET")
    else:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(origin + endpoint, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=max(0.01, timeout_s)) as response:
        body = response.read()
        value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError(f"{endpoint} returned non-object JSON")
    return response.status, value


def preflight(origin: str, expected_model_path: str) -> dict[str, Any]:
    started = time.perf_counter_ns()
    health_status, health = http_json(origin, "/health", None, REQUEST_TIMEOUT_S)
    props_status, props = http_json(origin, "/props", None, REQUEST_TIMEOUT_S)
    models_status, models = http_json(origin, "/v1/models", None, REQUEST_TIMEOUT_S)
    observed_model = props.get("model_path")
    observed_context = props.get("default_generation_settings", {}).get("n_ctx")
    result = {
        "health_status": health.get("status"),
        "health_http": health_status,
        "props_http": props_status,
        "models_http": models_status,
        "model_path": observed_model,
        "expected_model_path": expected_model_path,
        "context_size": observed_context,
        "model_count": len(models.get("data", [])) if isinstance(models.get("data"), list) else None,
        "elapsed_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
    }
    if health_status != 200 or health.get("status") != "ok":
        raise RuntimeError(f"native health preflight failed: {result}")
    if props_status != 200 or observed_model != expected_model_path:
        raise RuntimeError(f"native model identity preflight failed: {result}")
    if observed_context != CONTEXT_SIZE:
        raise RuntimeError(f"native context preflight failed: {result}")
    if models_status != 200 or result["model_count"] is None or result["model_count"] < 1:
        raise RuntimeError(f"native model list preflight failed: {result}")
    return result


def parser_case(old_client, raw_text: str) -> list[str]:
    return old_client.parse_prediction(raw_text)


def r_parse(text: str, timeout_s: float) -> dict[str, Any]:
    """Parse a private complete file; stdin() is not a reliable R parse source."""
    started = time.perf_counter_ns()
    try:
        with tempfile.TemporaryDirectory(prefix="sepalith-b4-rparse-") as directory:
            source = Path(directory) / "document.R"
            with source.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text)
            proc = subprocess.run(["Rscript", "--vanilla", "-e", "invisible(parse(file=commandArgs(TRUE)[1]))", str(source)], capture_output=True, text=True, timeout=max(0.01, timeout_s))
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "stderr": proc.stderr[-1500:], "milliseconds": round((time.perf_counter_ns()-started)/1e6, 3), "method": "Rscript private complete file parse"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)[:300], "milliseconds": round((time.perf_counter_ns()-started)/1e6, 3)}

def apply_in_memory(old_client, case: dict[str, Any], parsed_lines: list[str]) -> dict[str, Any]:
    """Apply the old client geometry to an in-memory v4 document only."""
    started = time.perf_counter_ns()
    before_text = case["document"]
    before_lines = before_text.split("\n")
    region_line_index = len(case["prefix"])
    current = case["region_old"][case["cursor_idx"]]
    old_case = {
        "region_old": list(case["region_old"]),
        "cursor_idx": case["cursor_idx"],
        "cursor_character": case["cursor_character"],
    }
    geometry = old_client.inline_geometry(old_case, parsed_lines)
    after_lines = list(before_lines)
    if geometry is not None:
        codepoint_column = old_client._codepoint_column(current, geometry["range_start_character"])
        replacement = (current[:codepoint_column] + "\n".join(parsed_lines)).split("\n")
        after_lines[region_line_index : region_line_index + 1] = replacement
    after_text = "\n".join(after_lines)
    elapsed_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
    return {
        "performed": geometry is not None,
        "action": "no_item" if geometry is None else "apply_matched_inline_range_in_memory",
        "writes": 0,
        "before_sha256": sha256_bytes(before_text.encode("utf-8")),
        "after_sha256": sha256_bytes(after_text.encode("utf-8")),
        "unchanged": before_text == after_text,
        "before_text": before_text,
        "after_text": after_text,
        "inline_geometry": geometry,
        "application_ms": elapsed_ms,
    }


def native_stop(old_client, response: dict[str, Any]) -> dict[str, Any]:
    return old_client.native_stop_metadata(response)


def completion_payload(old_client, prompt: str) -> dict[str, Any]:
    payload = old_client.make_completion_payload(prompt)
    expected = {
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": 0,
        "stop": list(EXTENSION_STOPS),
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
        "timings_per_token": True,
    }
    if payload != expected:
        raise RuntimeError("previous b4 client completion payload is not the pinned contract")
    return payload


def tokenize_payload(old_client, prompt: str) -> dict[str, Any]:
    payload = old_client.make_tokenize_payload(prompt)
    expected = {"content": prompt, **TOKENIZE_REQUEST}
    if payload != expected:
        raise RuntimeError("previous b4 client tokenizer payload is not the pinned contract")
    return payload


def run_one(origin: str, old_client, renderer, prepared: dict[str, Any], phase: str, request_index: int, deadline: float) -> dict[str, Any]:
    started = time.perf_counter_ns()
    case = prepared["case"]
    prompt = prepared["prompt"]
    record: dict[str, Any] = {
        "task": TASK,
        "request_index": request_index,
        "case_id": case["id"],
        "phase": phase,
        "phase_definition": "first changed prompt after preflight (cold-ish observation) or identical repeat; no cache-state claim",
        "prompt_sha256": prepared["prompt_sha256"],
        "prompt_chars": len(prompt),
        "request_contract": {
            "tokenize_endpoint": "/tokenize",
            "completion_endpoint": "/completion",
            "n_predict": MAX_TOKENS,
            "temperature": 0,
            "stream": False,
            "cache_prompt": False,
            "stop": list(EXTENSION_STOPS),
            "tokenizer": dict(TOKENIZE_REQUEST),
            "context_size": CONTEXT_SIZE,
            "request_timeout_s": REQUEST_TIMEOUT_S,
        },
        "label_routing": "none; case/phase labels are not included in request payloads",
        "started_unix_ns": time.time_ns(),
    }
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("cycle budget exhausted before render")
        render_started = time.perf_counter_ns()
        rendered = renderer.render_zeta2(prepared["legacy"])
        render_done = time.perf_counter_ns()
        if rendered != prompt:
            raise RuntimeError("render_zeta2 changed between static preparation and request")
        record["render_ms"] = round((render_done - render_started) / 1_000_000, 3)
        record["request_payload"] = completion_payload(old_client, prompt)
        record["request_payload_sha256"] = canonical_sha256(record["request_payload"])
        record["tokenize_payload"] = tokenize_payload(old_client, prompt)
        record["tokenize_payload_sha256"] = canonical_sha256(record["tokenize_payload"])
        residual = lambda: max(0.01, min(REQUEST_TIMEOUT_S, deadline - time.monotonic()))
        tokenize_started = time.perf_counter_ns()
        token_status, token_response = http_json(origin, "/tokenize", record["tokenize_payload"], residual())
        tokenize_done = time.perf_counter_ns()
        token_ids = token_response.get("tokens")
        if token_status != 200 or not isinstance(token_ids, list) or not all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids):
            raise RuntimeError(f"invalid native tokenize response: status={token_status}")
        record.update({
            "tokenize_ms": round((tokenize_done - tokenize_started) / 1_000_000, 3),
            "tokenize_token_count": len(token_ids),
            "tokenize_token_ids": list(token_ids),
            "native_tokenize_response": token_response,
        })
        completion_started = time.perf_counter_ns()
        completion_status, completion_response = http_json(origin, "/completion", record["request_payload"], residual())
        completion_done = time.perf_counter_ns()
        raw = completion_response.get("content")
        if completion_status != 200 or not isinstance(raw, str):
            raise RuntimeError(f"invalid native completion response: status={completion_status}")
        parse_started = time.perf_counter_ns()
        parsed = parser_case(old_client, raw)
        parse_done = time.perf_counter_ns()
        application = apply_in_memory(old_client, case, parsed)
        r_result = r_parse(application["after_text"], residual())
        if time.monotonic() > deadline:
            raise TimeoutError("cycle exceeded exact five-second foreground budget")
        record.update({
            "completion_ms": round((completion_done - completion_started) / 1_000_000, 3),
            "parse_ms": round((parse_done - parse_started) / 1_000_000, 3),
            "parsed_suggestion_latency_ms": round((parse_done - started) / 1_000_000, 3),
            "raw_completion": raw,
            "parsed_prediction": parsed,
            "parser_valid": True,
            "native_response": completion_response,
            "native_stop": native_stop(old_client, completion_response),
            "application": application,
            "r_parse": r_result,
            "response_ok": True,
            "quality_claim": "none",
            "scope": "native endpoint, protected legacy parser, and in-memory application diagnostic only",
            "cancellation": "not exercised by this non-streaming client; editor v5 owns cancellation evidence",
            "finished_unix_ns": time.time_ns(),
        })
        record["cycle_elapsed_ms"] = round((time.perf_counter_ns() - started) / 1_000_000, 3)
        return record
    except (OSError, RuntimeError, TimeoutError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        record.update({
            "response_ok": False,
            "error": str(exc)[:500],
            "quality_claim": "none",
            "scope": "native endpoint diagnostic only; no partial completion claim",
            "cycle_elapsed_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
        })
        return record


def self_test() -> dict[str, Any]:
    contract = static_contract()
    if r_parse("broken <- function( {", 10.0)["ok"]:
        raise RuntimeError("R parser falsely accepted malformed source")
    prepared = load_fixtures(contract["renderer_module"])
    parser = contract["parser_module"]
    parser_fixture = parser.parse_prediction("<<<<<<< CURRENT\nresult <- 1\n=======\n<[fim-middle]>\n>>>>>>> UPDATED")
    if parser_fixture != ["result <- 1"]:
        raise RuntimeError(f"legacy parser fixture changed: {parser_fixture}")
    app = []
    r_checks = []
    for item in prepared:
        applied = apply_in_memory(parser, item["case"], [item["case"]["region_old"][0]])
        parsed_r = r_parse(applied["after_text"], 10.0)
        expected_valid = item["case"]["id"] != "post-cancellation-fresh"
        if parsed_r["ok"] != expected_valid:
            raise RuntimeError(f"R parse fixture unexpected: {item['case']['id']}: {parsed_r}")
        r_checks.append({"id": item["case"]["id"], "ok": parsed_r["ok"], "expected_valid": expected_valid})
        app.append({"id": item["case"]["id"], "prompt_sha256": item["prompt_sha256"], "application_after_sha256": applied["after_sha256"]})
    return {
        "task": TASK,
        "status": "static_checks_pass",
        "renderer_parity_cases": len(prepared),
        "prompt_sha256": {item["case"]["id"]: item["prompt_sha256"] for item in prepared},
        "parser_fixture": parser_fixture,
        "r_parse_fixtures": r_checks,
        "in_memory_application_checks": app,
        "source_hashes": {key: value for key, value in contract.items() if key in {"previous_client", "renderer", "parser", "model", "runtime"}},
        "request_contract": contract["request"],
        "network_used": False,
        "model_opened": False,
    }


def run_live(args: argparse.Namespace) -> int:
    origin = base_url(args.url)
    contract = static_contract()
    prepared = load_fixtures(contract["renderer_module"])
    output_dir = Path(args.run_root).resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run root: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    output_path = output_dir / "per_request.jsonl"
    summary_path = output_dir / "summary.json"
    run_started = time.monotonic()
    preflight_result = preflight(origin, MODEL_PATH)
    records = []
    request_index = 0
    status = "complete"
    with output_path.open("x", encoding="utf-8") as stream:
        for case_index, item in enumerate(prepared):
            for repetition in range(2):
                if time.monotonic() - run_started >= args.overall_timeout_s:
                    status = "overall_deadline"
                    break
                phase = "cold-ish-changed" if case_index == 0 and repetition == 0 else "changed" if repetition == 0 else "repeat"
                deadline = time.monotonic() + REQUEST_TIMEOUT_S
                record = run_one(origin, contract["parser_module"], contract["renderer_module"], item, phase, request_index, deadline)
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                records.append(record)
                request_index += 1
                if not record.get("response_ok"):
                    status = "failed"
            if status == "overall_deadline":
                break
    successes = [item for item in records if item.get("response_ok")]
    if len(records) != 6:
        status = "incomplete" if status == "complete" else status
    summary = {
        "schema": "RUN-03-b4-native-baseline-v5-summary/v1",
        "task": TASK,
        "status": status,
        "started_unix_ns": time.time_ns(),
        "scope": {
            "server_owner": "root",
            "loopback": origin,
            "model_opened_by_client": False,
            "server_started_by_client": False,
            "quality_claim": "none",
            "label_routing": "none",
            "requests": "three captured v4 prompt geometries, first changed/cold-ish observation and repeat each",
        },
        "identities": {key: value for key, value in contract.items() if key in {"previous_client", "renderer", "parser", "model", "runtime", "receipt_sha256"}},
        "profile": contract["request"],
        "server_preflight": preflight_result,
        "cycles": {
            "requested": 6,
            "written": len(records),
            "successful": len(successes),
            "failed": len(records) - len(successes),
            "by_case": {item["case"]["id"]: {"first_prompt_sha256": item["prompt_sha256"], "repetitions": 2} for item in prepared},
        },
        "timing_definitions": {
            "render_ms": "local protected run_eval.render_zeta2 call",
            "tokenize_ms": "native /tokenize loopback round trip",
            "completion_ms": "native /completion loopback round trip",
            "parse_ms": "protected previous b4 client parse_prediction only",
            "application_ms": "in-memory legacy inline geometry and replacement only",
            "r_parse": "Rscript parse of a private complete file containing resulting in-memory text",
            "parsed_suggestion_latency_ms": "cycle start through protected parser; application/R parse separately recorded in full cycle",
        },
        "limits": [
            "Six target-free synthetic requests are a bounded native diagnostic; no quality, no-op correctness, or promotion claim.",
            "First changed request is called cold-ish only as an observation label; repeated cache_prompt=false calls make no cache-state claim.",
            "This client does not exercise editor ghost publication or request cancellation; v5 editor harness owns those traces.",
            "Model identity is captured from accepted receipts and preflight; this client never opens model bytes.",
        ],
        "artifacts": {
            "per_request_jsonl": {"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": sha256_file(output_path), "rows": len(records)},
            "summary_json": {"path": str(summary_path)},
        },
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    summary["artifacts"]["summary_json"]["bytes"] = summary_path.stat().st_size
    summary["artifacts"]["summary_json"]["sha256"] = sha256_file(summary_path)
    # The summary was already written before its own final hash was known; the
    # recorded artifact hash is the hash of the written summary body.
    print(json.dumps({"task": TASK, "status": status, "cycles": summary["cycles"], "summary": str(summary_path)}, sort_keys=True))
    return 0 if status == "complete" and len(successes) == 6 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run source/renderer/parser checks without network or server")
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--run-root", default=str(HERE / "live-run-v5"))
    parser.add_argument("--overall-timeout-s", type=float, default=45.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.self_test:
            print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
            return 0
        if args.overall_timeout_s <= 0:
            raise ValueError("overall timeout must be positive")
        return run_live(args)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"task": TASK, "status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
