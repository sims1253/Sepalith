#!/usr/bin/env python3
"""Prepare a small b4 notebook endpoint baseline (v2).

This is a client and fixture runner.  It never starts, stops, or supervises a
server and it never loads model weights.  A root-owned b4 llama-server can be
measured later through a loopback URL with the exact legacy zeta2 renderer,
native tokenizer request, extension stop list, and parser used by the accepted
b4 DEV preparation.

The two requests in :data:`SYNTHETIC_CASES` are deliberately synthetic.  They
exercise one replacement and one no-op through the old line-end cursor format;
they are not PRM-03 or DEV rows and make no editor-quality or no-op claim.  The
live command sends each rendered prompt twice.  Only the first request after
server preflight is labelled ``cold``; the first request for the other fixture
is ``different_prompt`` and the second request for each fixture is ``repeat``.
``cache_prompt`` is fixed false, so ``repeat`` makes no cache-hit claim.

Foreground cycles have an absolute five-second budget spanning local render,
native tokenization, completion, and parsing.  A separately requested
diagnostic mode can use a 60-second cycle budget; it is labelled as diagnostic
and is not a foreground latency result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


TASK = "RUN-03-b4-notebook-baseline-v2"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_SOFT_DEADLINE_S = 600.0
DEFAULT_RESERVE_S = 30.0
DEFAULT_CYCLE_DEADLINE_S = 5.0
DEFAULT_DIAGNOSTIC_TIMEOUT_S = 60.0
EXIT_DEADLINE = 124

EXTENSION_STOPS = [
    ">>>>>>> UPDATED",
    "<<<<<<< CURRENT",
    "=======",
    "<[fim-middle]>",
    "<[fim-suffix]>",
    "<[fim-prefix]>",
    "<|outline|>",
]
PRODUCTION_MAX_TOKENS = 320
TOKENIZE_REQUEST = {
    "add_special": True,
    "parse_special": True,
    "with_pieces": False,
}

SCRIPT = Path(__file__).resolve()
PLAN_ROOT = SCRIPT.parents[4]
DEFAULT_REPO_ROOT = PLAN_ROOT
DEFAULT_AUTHORITY_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/SFT-08-b4-dev-preparation.json"
DEFAULT_PRE08_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/PRE-08-fallback-ledger.json"
DEFAULT_MODEL = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf"
)
DEFAULT_RUNTIME = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server"
)
DEFAULT_OUTPUT = Path(
    "/home/m0hawk/.local/state/sepalith-campaign-20260915/"
    "RUN-03-b4-notebook-baseline-v2/per_cycle.jsonl"
)
DEFAULT_SUMMARY = DEFAULT_OUTPUT.with_name("summary.json")


class UnsupportedFixture(ValueError):
    """The legacy request contract cannot represent a synthetic fixture."""


class DeadlineExceeded(RuntimeError):
    """The bounded client reached its soft deadline."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field} must be a non-empty string")
    return value


def authority_contract(
    receipt_path: Path = DEFAULT_AUTHORITY_RECEIPT,
    pre08_receipt_path: Path = DEFAULT_PRE08_RECEIPT,
) -> dict[str, Any]:
    """Read the protected b4 identities from accepted receipts.

    Full hashes are copied from the receipts at runtime.  No notebook model
    fingerprint is fabricated when the protected model is absent on that
    host.  PRE-08 is read as a cross-check because it is the fallback ledger;
    the SFT-08 receipt supplies the accepted request contract and paths.
    """
    accepted = _read_json(receipt_path)
    fallback = _read_json(pre08_receipt_path)
    accepted_inputs = accepted.get("inputs")
    if not isinstance(accepted_inputs, dict):
        raise RuntimeError("accepted b4 receipt has no inputs object")
    model = accepted_inputs.get("model")
    runtime = accepted_inputs.get("runtime")
    sources = accepted_inputs.get("legacy_sources")
    request = accepted.get("request_contract")
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise RuntimeError("accepted b4 receipt lacks model/runtime identity")
    if not isinstance(sources, dict) or not isinstance(request, dict):
        raise RuntimeError("accepted b4 receipt lacks renderer/request contract")
    renderer = sources.get("renderer")
    parser = sources.get("parser")
    if not isinstance(renderer, dict) or not isinstance(parser, dict):
        raise RuntimeError("accepted b4 receipt lacks renderer/parser identity")

    fields = {
        "model_path": _required_string(model.get("path"), "model.path"),
        "model_bytes": model.get("bytes"),
        "model_sha256": _required_string(model.get("sha256"), "model.sha256"),
        "runtime_path": _required_string(runtime.get("path"), "runtime.path"),
        "runtime_sha256": _required_string(runtime.get("sha256"), "runtime.sha256"),
        "runtime_build": runtime.get("build"),
        "runtime_commit": runtime.get("commit"),
        "renderer_path": _required_string(renderer.get("path"), "renderer.path"),
        "renderer_sha256": _required_string(renderer.get("sha256"), "renderer.sha256"),
        "renderer_implementation": _required_string(
            renderer.get("implementation"), "renderer.implementation"
        ),
        "parser_path": _required_string(parser.get("path"), "parser.path"),
        "parser_sha256": _required_string(parser.get("sha256"), "parser.sha256"),
        "parser_implementation": _required_string(
            parser.get("implementation"), "parser.implementation"
        ),
        "context_size": request.get("context_size"),
        "completion_endpoint": request.get("endpoint"),
        "tokenize_endpoint": request.get("tokenizer_endpoint"),
        "max_tokens": request.get("production_all_rows", {}).get("n_predict"),
        "stops": request.get("production_all_rows", {}).get("stop"),
        "temperature": request.get("temperature"),
        "stream": request.get("stream"),
        "cache_prompt": request.get("cache_prompt"),
        "tokenizer_request": request.get("tokenizer_request"),
    }
    if fields["model_bytes"] != 2012011904:
        raise RuntimeError("accepted receipt b4 model byte count drifted")
    if fields["context_size"] != DEFAULT_CONTEXT_SIZE:
        raise RuntimeError("accepted receipt b4 context size drifted")
    if fields["max_tokens"] != PRODUCTION_MAX_TOKENS:
        raise RuntimeError("accepted receipt b4 response cap drifted")
    if fields["stops"] != EXTENSION_STOPS:
        raise RuntimeError("accepted receipt b4 stop list drifted")
    if fields["temperature"] != 0 or fields["stream"] is not False or fields["cache_prompt"] is not False:
        raise RuntimeError("accepted receipt b4 deterministic request contract drifted")
    if fields["tokenizer_request"] != TOKENIZE_REQUEST:
        raise RuntimeError("accepted receipt b4 tokenizer request drifted")

    # The fallback ledger must bind the same protected model, runtime, and
    # renderer.  This catches a copied receipt with a plausible-looking name.
    fallback_result = fallback.get("RESULT")
    if not isinstance(fallback_result, dict):
        raise RuntimeError("PRE-08 fallback ledger has no RESULT object")
    fallback_model = fallback_result.get("matched_banked_artifact")
    fallback_runtime = fallback_result.get("matched_runtime_identities", {}).get(
        "local_cpu_fallback"
    )
    fallback_renderer = fallback_result.get("legacy_contract", {}).get("renderer")
    # PRE-08 is an older ledger schema.  Its identity is nested under RESULT;
    # if that shape changes, fail closed instead of inventing an identity.
    if not isinstance(fallback_model, dict) or not isinstance(fallback_runtime, dict):
        raise RuntimeError("PRE-08 fallback ledger lacks model/runtime identity")
    if fallback_model.get("sha256") != fields["model_sha256"]:
        raise RuntimeError("PRE-08 model hash disagrees with accepted b4 receipt")
    if fallback_runtime.get("sha256") != fields["runtime_sha256"]:
        raise RuntimeError("PRE-08 runtime hash disagrees with accepted b4 receipt")
    fallback_renderer_sha = (
        fallback_renderer.get("run_eval_sha256")
        if isinstance(fallback_renderer, dict)
        else None
    )
    if fallback_renderer_sha not in (None, fields["renderer_sha256"]):
        raise RuntimeError("PRE-08 renderer hash disagrees with accepted b4 receipt")
    # Keep model and runtime identities nested.  The old preparation packet
    # flattened local metadata and accidentally reported runtime bytes as
    # model bytes; this shape makes that collision impossible.
    accepted_artifacts = {
        "model": {
            "path": fields["model_path"],
            "bytes": fields["model_bytes"],
            "sha256": fields["model_sha256"],
        },
        "runtime": {
            "path": fields["runtime_path"],
            "bytes": None,
            "sha256": fields["runtime_sha256"],
            "build": fields["runtime_build"],
            "commit": fields["runtime_commit"],
        },
    }
    return {
        "authority_receipt": str(receipt_path.resolve()),
        "authority_receipt_sha256": sha256_file(receipt_path),
        "fallback_receipt": str(pre08_receipt_path.resolve()),
        "fallback_receipt_sha256": sha256_file(pre08_receipt_path),
        "accepted": fields,
        "accepted_artifacts": accepted_artifacts,
    }


def _load_hashed_module(
    path: Path, expected_sha256: str, name: str
) -> types.ModuleType:
    if not path.is_file():
        raise RuntimeError(f"missing {name} source: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{name} source hash mismatch: {actual}; expected {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legacy_renderer(
    repo_root: Path, contract: dict[str, Any]
) -> types.ModuleType:
    path = repo_root / "experiments/eval/run_eval.py"
    expected = contract["accepted"]["renderer_sha256"]
    module = _load_hashed_module(path, expected, "run_eval")
    if not callable(getattr(module, "render_zeta2", None)):
        raise RuntimeError("pinned renderer has no render_zeta2")
    if not callable(getattr(module, "norm", None)):
        raise RuntimeError("pinned renderer has no norm")
    return module


def verify_static_identities(
    model_path: Path,
    runtime_path: Path,
    repo_root: Path,
    contract: dict[str, Any],
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Verify protected local artifacts before contacting a server.

    The model is hashed in 1 MiB chunks by :func:`sha256_file`; this function
    is only called by the root-owned live command.  Unit tests inject a hash
    function and never open a model.
    """
    accepted = contract["accepted"]
    if not model_path.is_file():
        raise RuntimeError(f"protected b4 model is absent: {model_path}")
    if model_path.stat().st_size != accepted["model_bytes"]:
        raise RuntimeError("protected b4 model byte count mismatch")
    if not runtime_path.is_file() or not os.access(runtime_path, os.X_OK):
        raise RuntimeError(f"b4 runtime is absent or not executable: {runtime_path}")
    renderer_path = repo_root / "experiments/eval/run_eval.py"
    parser_path = repo_root / "extensions/vscode-sepalith/src/extension.ts"
    if hash_file(model_path) != accepted["model_sha256"]:
        raise RuntimeError("protected b4 model hash mismatch")
    if hash_file(runtime_path) != accepted["runtime_sha256"]:
        raise RuntimeError("b4 runtime hash mismatch")
    if hash_file(renderer_path) != accepted["renderer_sha256"]:
        raise RuntimeError("zeta2 renderer hash mismatch")
    if hash_file(parser_path) != accepted["parser_sha256"]:
        raise RuntimeError("extension parser source hash mismatch")
    return {
        "model": {
            "path": str(model_path.resolve()),
            "bytes": model_path.stat().st_size,
            "sha256": accepted["model_sha256"],
        },
        "runtime": {
            "path": str(runtime_path.resolve()),
            "bytes": runtime_path.stat().st_size,
            "sha256": accepted["runtime_sha256"],
            "build": accepted["runtime_build"],
            "commit": accepted["runtime_commit"],
        },
        "renderer": {
            "path": str(renderer_path.resolve()),
            "sha256": accepted["renderer_sha256"],
            "implementation": accepted["renderer_implementation"],
        },
        "parser": {
            "path": str(parser_path.resolve()),
            "sha256": accepted["parser_sha256"],
            "implementation": accepted["parser_implementation"],
        },
    }


def local_artifact_metadata(
    model_path: Path, runtime_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    """Describe local model/runtime files without reading their contents.

    Preparation and notebook discovery can use this helper safely: a model
    hash is reported as ``not_checked`` until the root-owned live verifier has
    streamed the protected model.  Model and runtime byte counts stay in
    separate objects because the runtime is a small executable and is never a
    substitute for the protected GGUF identity.
    """
    accepted = contract["accepted"]
    model_present = model_path.is_file()
    runtime_present = runtime_path.is_file()
    return {
        "model": {
            "path": str(model_path.resolve()),
            "present": model_present,
            "bytes": model_path.stat().st_size if model_present else None,
            "expected_bytes": accepted["model_bytes"],
            "sha256": "not_checked",
            "expected_sha256": accepted["model_sha256"],
        },
        "runtime": {
            "path": str(runtime_path.resolve()),
            "present": runtime_present,
            "bytes": runtime_path.stat().st_size if runtime_present else None,
            "sha256": "not_checked",
            "expected_sha256": accepted["runtime_sha256"],
            "build": accepted["runtime_build"],
            "commit": accepted["runtime_commit"],
        },
    }


def request_phase(item_index: int, repetition_index: int) -> str:
    """Name request state without implying a cache state.

    The server is preflighted once.  Thus only the first request overall is
    cold; the second fixture starts a different prompt and each second call
    is merely a repeat with ``cache_prompt=false``.
    """
    if item_index == 0 and repetition_index == 0:
        return "cold"
    if repetition_index == 0:
        return "different_prompt"
    return "repeat"


def synthetic_cases() -> list[dict[str, Any]]:
    """Return target-free, hand-authored legacy request-cycle fixtures."""
    # Cursor columns are deliberately calculated from the visible source.  The
    # trailing context keeps both synthetic documents above the extension's
    # 30-character provider-eligibility floor; it is fixture context only.
    eligible_suffix = "# b4 endpoint fixture keeps enough source context"
    return [
        {
            "id": "synthetic-b4-replace",
            "family": "synthetic_replace",
            "operation": "replace",
            "path": "R/synthetic_cycle.R",
            "prefix": ["value <- 1"],
            "region_old": ["value <- 1"],
            "cursor_idx": 0,
            "cursor_character": utf16_units("value <- 1"),
            "suffix": ["next <- value + 1", eligible_suffix],
            "event_diff": "",
            "target_lines": ["value <- 2"],
        },
        {
            "id": "synthetic-b4-no-op",
            "family": "synthetic_no_op",
            "operation": "no_op",
            "path": "R/synthetic_cycle.R",
            "prefix": ["done <- TRUE"],
            "region_old": ["done <- TRUE"],
            "cursor_idx": 0,
            "cursor_character": utf16_units("done <- TRUE"),
            "suffix": [eligible_suffix],
            "event_diff": "",
            "target_lines": ["done <- TRUE"],
        },
    ]


def prepare_fixture(
    case: dict[str, Any], renderer: types.ModuleType
) -> dict[str, Any]:
    required = ("id", "family", "operation", "path", "prefix", "region_old", "suffix", "target_lines")
    for field in required:
        if field not in case:
            raise UnsupportedFixture(f"fixture missing {field}")
    if case["operation"] not in {"replace", "no_op"}:
        raise UnsupportedFixture("only replace and no_op are represented")
    if not isinstance(case["family"], str) or not isinstance(case["id"], str):
        raise UnsupportedFixture("fixture id/family must be strings")
    for field in ("prefix", "region_old", "suffix", "target_lines"):
        if not isinstance(case[field], list) or not all(isinstance(v, str) for v in case[field]):
            raise UnsupportedFixture(f"fixture {field} must be a string list")
    if not isinstance(case.get("cursor_idx"), int) or case["cursor_idx"] != 0:
        raise UnsupportedFixture("synthetic fixture uses a single line-end cursor")
    if case["operation"] == "no_op" and case["target_lines"] != case["region_old"]:
        raise UnsupportedFixture("no_op fixture target must preserve the old region")
    if not case["region_old"]:
        raise UnsupportedFixture("fixture requires a visible region line")
    legacy = {
        "suffix": list(case["suffix"]),
        "event_diff": case.get("event_diff", ""),
        "path": case["path"],
        "prefix": list(case["prefix"]),
        "region_old": list(case["region_old"]),
        "cursor_idx": case["cursor_idx"],
    }
    prompt = renderer.render_zeta2(legacy)
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError(f"renderer returned no prompt for {case['id']}")
    if "[NO_EDIT]" in prompt:
        raise RuntimeError("synthetic no-op marker leaked into legacy prompt")
    return {
        "case": case,
        "legacy": legacy,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "target_sha256": sha256_bytes("\n".join(case["target_lines"]).encode("utf-8")),
        "request_payload": make_completion_payload(prompt),
        "tokenize_payload": make_tokenize_payload(prompt),
    }


# Keep this parser in lockstep with the accepted extension.ts parsePrediction
# implementation.  verify_static_identities checks the source hash before a
# live run, so a parser change cannot silently change a measurement.
MARKER_LINE = re.compile(
    r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|"
    r"<\[fim-(middle|prefix|suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$"
)


def parse_prediction(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("completion content must be a string")
    if ">>>>>>>" in text:
        text = text.split(">>>>>>>")[0]
    text = text.replace("<|user_cursor|>", "")
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    lines = [line for line in lines if not MARKER_LINE.match(line)]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    for index in range(2, len(lines)):
        if lines[index] == lines[index - 1] == lines[index - 2]:
            del lines[index:]
            break
    return lines


def make_completion_payload(prompt: str) -> dict[str, Any]:
    """Create the one request body for every fixture and phase."""
    return {
        "prompt": prompt,
        "n_predict": PRODUCTION_MAX_TOKENS,
        "temperature": 0,
        "stop": list(EXTENSION_STOPS),
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
        "timings_per_token": True,
    }


def make_tokenize_payload(prompt: str) -> dict[str, Any]:
    return {"content": prompt, **TOKENIZE_REQUEST}


def _base_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("b4 notebook client accepts only an http loopback URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("server URL must not contain path/query/fragment")
    return url.rstrip("/")


def http_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    url = base_url + endpoint
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    if timeout_s <= 0:
        raise DeadlineExceeded(f"{endpoint} has no residual deadline")
    # The caller supplies the residual absolute-cycle budget.  Do not inflate
    # a short residual to a fixed 200 ms timeout, which would violate the
    # five-second cycle contract near its boundary.
    with urllib.request.urlopen(request, timeout=max(0.01, timeout_s)) as response:
        decoded = json.loads(response.read())
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{endpoint} returned non-object JSON")
    return response.status, decoded


def server_preflight(
    base_url: str,
    expected_model_path: str,
    timeout_s: float,
    request_fn: Callable[..., tuple[int, dict[str, Any]]] = http_json,
) -> dict[str, Any]:
    health_status, health = request_fn(base_url, "/health", None, timeout_s)
    if health_status != 200:
        raise RuntimeError(f"server health status was {health_status}")
    if health.get("status") != "ok":
        raise RuntimeError(f"server health is not ok: {health.get('status')!r}")
    props_status, props = request_fn(base_url, "/props", None, timeout_s)
    if props_status != 200:
        raise RuntimeError(f"server props status was {props_status}")
    models_status, models = request_fn(base_url, "/v1/models", None, timeout_s)
    if models_status != 200:
        raise RuntimeError(f"server model-list status was {models_status}")
    reported_path = props.get("model_path")
    if not isinstance(reported_path, str) or reported_path != expected_model_path:
        raise RuntimeError(
            f"server model identity mismatch: {reported_path!r} != {expected_model_path!r}"
        )
    generation = props.get("default_generation_settings")
    if not isinstance(generation, dict) or generation.get("n_ctx") != DEFAULT_CONTEXT_SIZE:
        raise RuntimeError("server context is not the pinned 8192-token contract")
    return {
        "health": health,
        "props": props,
        "models": models,
        "model_path_reported": reported_path,
        "n_ctx": generation["n_ctx"],
    }


def native_stop_metadata(response: dict[str, Any]) -> dict[str, Any]:
    tokens = response.get("tokens")
    token_ids = tokens if isinstance(tokens, list) else []
    predicted = response.get("tokens_predicted")
    return {
        "stop": response.get("stop"),
        "stop_type": response.get("stop_type"),
        "stopping_word": response.get("stopping_word"),
        "tokens_predicted": predicted,
        "tokens_evaluated": response.get("tokens_evaluated"),
        "tokens_cached": response.get("tokens_cached"),
        "truncated": response.get("truncated"),
        "has_new_line": response.get("has_new_line"),
        "last_generated_token_id": token_ids[-1] if token_ids else None,
        "cap_hit": int(
            response.get("stop_type") == "limit"
            or (isinstance(predicted, int) and predicted >= PRODUCTION_MAX_TOKENS)
        ),
    }


def inline_geometry(case: dict[str, Any], parsed_lines: list[str]) -> dict[str, Any] | None:
    """Mirror the extension's typed-partial range rule for fixture evidence."""
    if not parsed_lines:
        return None
    cursor_idx = case["cursor_idx"]
    current_line = case["region_old"][cursor_idx]
    cursor_character = case.get("cursor_character", utf16_units(current_line))
    typed_partial = current_line[:cursor_character].strip()
    range_start_character = cursor_character
    if len(typed_partial) >= 4 and parsed_lines[0].strip().startswith(typed_partial):
        range_start_character = 0
    return {
        "range_start_line": cursor_idx,
        "range_start_character": range_start_character,
        "range_end_line": cursor_idx,
        "range_end_character": utf16_units(current_line),
        "typed_partial_length": len(typed_partial),
        "typed_partial_rule": "start_zero_only_when_prediction_starts_with_four_plus_char_typed_partial",
        "source": "matched_extension_inline_completion_geometry",
    }


def simulated_application(case: dict[str, Any], parsed_lines: list[str]) -> dict[str, Any]:
    """Apply a synthetic result in memory, without scoring editor quality."""
    before = list(case["prefix"]) + list(case["region_old"]) + list(case["suffix"])
    if case["operation"] == "no_op":
        after = list(before)
        action = "unchanged_no_op"
    else:
        start = len(case["prefix"])
        end = start + len(case["region_old"])
        after = before[:start] + list(parsed_lines) + before[end:]
        action = "replace_region_in_memory"
    return {
        "performed": True,
        "writes": 0,
        "action": action,
        "before_sha256": canonical_sha256(before),
        "after_sha256": canonical_sha256(after),
        "inline_geometry": inline_geometry(case, parsed_lines),
        "scope": "synthetic_in_memory_endpoint_result_only",
        "quality_claim": "none",
        "live_editor_application": "pending_root_editor_check",
    }


def _failure_record(
    prepared: dict[str, Any], phase: str, started_ns: int, error: str
) -> dict[str, Any]:
    return {
        "task": TASK,
        "case_id": prepared["case"]["id"],
        "family_label": prepared["case"]["family"],
        "fixture_operation_label": prepared["case"]["operation"],
        "phase": phase,
        "response_ok": 0,
        "error": error[:500],
        "started_unix_ns": time.time_ns(),
        "cycle_elapsed_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        "cancellation": {
            "status": "not_exercised",
            "root_live_required": True,
            "client_cancelled": False,
        },
        "application": {
            "status": "not_exercised_after_transport_failure",
            "root_live_required": True,
        },
        "cleanup": {
            "client_transport_context_closed": True,
            "server_process_started_by_client": False,
            "server_process_stopped_by_client": False,
            "owned_processes_verified": False,
            "root_supervisor_evidence_required": True,
        },
    }


def run_cycle(
    base_url: str,
    prepared: dict[str, Any],
    phase: str,
    timeout_s: float,
    renderer: types.ModuleType,
    request_fn: Callable[..., tuple[int, dict[str, Any]]] = http_json,
    *,
    cycle_deadline_s: float = DEFAULT_CYCLE_DEADLINE_S,
    diagnostic: bool = False,
    diagnostic_timeout_s: float = DEFAULT_DIAGNOSTIC_TIMEOUT_S,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one bounded request and retain parsed-latency evidence.

    ``timeout_s`` is the remaining overall-run budget.  Each endpoint receives
    the smaller of that budget and the cycle's residual absolute deadline, so
    a slow tokenizer leaves only its actual remainder for completion.  The
    optional diagnostic budget is explicit in the record and must not be
    interpreted as a foreground latency result.
    """
    if timeout_s <= 0:
        raise ValueError("overall remaining timeout must be positive")
    cycle_budget_s = diagnostic_timeout_s if diagnostic else cycle_deadline_s
    if cycle_budget_s <= 0:
        raise ValueError("cycle deadline must be positive")
    cycle_mode = "diagnostic_60s" if diagnostic else "foreground_5s"
    cycle_started = clock()
    cycle_deadline = cycle_started + cycle_budget_s

    def residual_timeout() -> float:
        remaining = min(timeout_s, cycle_deadline - clock())
        if remaining <= 0:
            raise DeadlineExceeded(
                f"{cycle_mode} cycle exceeded its {cycle_budget_s:g}s absolute deadline"
            )
        return remaining

    started_ns = time.perf_counter_ns()
    case = prepared["case"]
    record: dict[str, Any] = {
        "task": TASK,
        "case_id": case["id"],
        "family_label": case["family"],
        "fixture_operation_label": case["operation"],
        "phase": phase,
        "deadline": {
            "mode": cycle_mode,
            "absolute": True,
            "cycle_budget_s": cycle_budget_s,
            "overall_remaining_s": timeout_s,
            "reserve_applies_to_overall_run": True,
        },
        "request_contract": {
            "endpoint": "/completion",
            "tokenize_endpoint": "/tokenize",
            "n_predict": PRODUCTION_MAX_TOKENS,
            "temperature": 0,
            "stop": list(EXTENSION_STOPS),
            "stream": False,
            "cache_prompt": False,
            "tokenizer": dict(TOKENIZE_REQUEST),
        },
        "legacy_renderer": {
            "implementation": "run_eval.render_zeta2",
            "prompt_sha256": prepared["prompt_sha256"],
            "prompt_chars": len(prepared["prompt"]),
        },
        "request_payload": prepared["request_payload"],
        "request_payload_sha256": canonical_sha256(prepared["request_payload"]),
        "tokenize_payload": prepared["tokenize_payload"],
        "tokenize_payload_sha256": canonical_sha256(prepared["tokenize_payload"]),
        "started_unix_ns": time.time_ns(),
    }
    try:
        render_started = time.perf_counter_ns()
        prompt = renderer.render_zeta2(prepared["legacy"])
        render_done = time.perf_counter_ns()
        if prompt != prepared["prompt"]:
            raise RuntimeError("renderer output changed during cycle")
        residual_timeout()
        record["render_ms"] = round((render_done - render_started) / 1_000_000, 3)

        tokenize_started = time.perf_counter_ns()
        tokenize_status, tokenize_response = request_fn(
            base_url,
            "/tokenize",
            prepared["tokenize_payload"],
            residual_timeout(),
        )
        tokenize_done = time.perf_counter_ns()
        if tokenize_status != 200:
            raise RuntimeError(f"tokenize status was {tokenize_status}")
        token_ids = tokenize_response.get("tokens")
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in token_ids
        ):
            raise RuntimeError("native tokenizer did not return integer token IDs")
        record.update(
            {
                "tokenize_ms": round((tokenize_done - tokenize_started) / 1_000_000, 3),
                "tokenize_token_count": len(token_ids),
                "tokenize_token_ids": list(token_ids),
                "native_tokenize_response": tokenize_response,
            }
        )

        completion_started = time.perf_counter_ns()
        completion_status, completion_response = request_fn(
            base_url,
            "/completion",
            prepared["request_payload"],
            residual_timeout(),
        )
        completion_done = time.perf_counter_ns()
        if completion_status != 200:
            raise RuntimeError(f"completion status was {completion_status}")
        content = completion_response.get("content")
        if not isinstance(content, str):
            raise RuntimeError("completion response lacks string content")
        # Parsing is part of the foreground deadline too.  A response that
        # arrives at the boundary is a timeout rather than a successful
        # latency sample.
        parse_started = time.perf_counter_ns()
        parsed = parse_prediction(content)
        parse_done = time.perf_counter_ns()
        residual_timeout()
        record.update(
            {
                "completion_ms": round((completion_done - completion_started) / 1_000_000, 3),
                "parse_ms": round((parse_done - parse_started) / 1_000_000, 3),
                # This is measured from the cycle start through parsed output,
                # rather than TTFT or server-side timings alone.
                "parsed_suggestion_latency_ms": round(
                    (parse_done - started_ns) / 1_000_000, 3
                ),
                "raw_completion": content,
                "parsed_prediction": parsed,
                "parser_valid": 1,
                "native_response": completion_response,
                "native_stop": native_stop_metadata(completion_response),
                "response_ok": 1,
                "application": simulated_application(case, parsed),
                "scope": "endpoint_and_parser_diagnostic_only",
                "quality_claim": "none",
                "cancellation": {
                    "status": "not_exercised",
                    "root_live_required": True,
                    "client_cancelled": False,
                    "reason": "non-streaming full-cycle fixture does not abort a live editor request",
                },
                "cleanup": {
                    "client_transport_context_closed": True,
                    "server_process_started_by_client": False,
                    "server_process_stopped_by_client": False,
                    "owned_processes_verified": False,
                    "root_supervisor_evidence_required": True,
                },
            }
        )
        record["finished_unix_ns"] = time.time_ns()
        record["cycle_elapsed_ms"] = round(
            (time.perf_counter_ns() - started_ns) / 1_000_000, 3
        )
        return record
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        failure = _failure_record(prepared, phase, started_ns, str(exc))
        failure["request_contract"] = record["request_contract"]
        failure["legacy_renderer"] = record["legacy_renderer"]
        failure["request_payload_sha256"] = record["request_payload_sha256"]
        failure["tokenize_payload_sha256"] = record["tokenize_payload_sha256"]
        failure["deadline"] = record["deadline"]
        failure["scope"] = "endpoint_and_parser_diagnostic_only"
        failure["quality_claim"] = "none"
        return failure


def deadline_settings(
    soft: float | None,
    reserve: float | None,
    cycle: float | None = None,
    diagnostic: float | None = None,
) -> dict[str, float]:
    def env_float(name: str, fallback: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return fallback
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be positive")
        return parsed

    chosen_soft = soft if soft is not None else env_float(
        "SEPALITH_B4_NOTEBOOK_SOFT_DEADLINE_S", DEFAULT_SOFT_DEADLINE_S
    )
    chosen_reserve = reserve if reserve is not None else env_float(
        "SEPALITH_B4_NOTEBOOK_RESERVE_S", DEFAULT_RESERVE_S
    )
    chosen_cycle = cycle if cycle is not None else env_float(
        "SEPALITH_B4_NOTEBOOK_CYCLE_DEADLINE_S", DEFAULT_CYCLE_DEADLINE_S
    )
    chosen_diagnostic = diagnostic if diagnostic is not None else env_float(
        "SEPALITH_B4_NOTEBOOK_DIAGNOSTIC_TIMEOUT_S", DEFAULT_DIAGNOSTIC_TIMEOUT_S
    )
    if (
        chosen_soft <= 0
        or chosen_reserve <= 0
        or chosen_cycle <= 0
        or chosen_diagnostic <= 0
    ):
        raise ValueError("deadline and reserve must be positive")
    if chosen_reserve >= chosen_soft:
        raise ValueError("overall reserve must be smaller than the soft deadline")
    if chosen_cycle > chosen_soft - chosen_reserve:
        raise ValueError("cycle deadline cannot exceed the overall budget before reserve")
    return {
        "soft_s": float(chosen_soft),
        "reserve_s": float(chosen_reserve),
        "cycle_s": float(chosen_cycle),
        "diagnostic_s": float(chosen_diagnostic),
    }


def fresh_path(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")


class FsyncJSONL:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        self.file = os.fdopen(fd, "w", encoding="utf-8")
        self.path = path
        self.rows = 0

    def write(self, record: dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.rows += 1

    def close(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        dir_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh_path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_baseline(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    base_url = _base_url(args.url)
    contract = authority_contract(Path(args.authority_receipt), Path(args.pre08_receipt))
    model_path = Path(args.model).resolve()
    runtime_path = Path(args.runtime).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()
    fresh_path(output_path)
    fresh_path(summary_path)
    static = verify_static_identities(model_path, runtime_path, repo_root, contract)
    renderer = load_legacy_renderer(repo_root, contract)
    prepared = [prepare_fixture(case, renderer) for case in synthetic_cases()]
    if len({item["prompt_sha256"] for item in prepared}) != len(prepared):
        raise RuntimeError("synthetic legacy prompt collision")
    deadlines = deadline_settings(
        args.soft_deadline_s,
        args.reserve_s,
        args.cycle_deadline_s,
        args.diagnostic_timeout_s,
    )
    started = time.monotonic()
    overall_deadline = started + deadlines["soft_s"] - deadlines["reserve_s"]

    def bounded_preflight_request(
        request_base_url: str,
        endpoint: str,
        payload: dict[str, Any] | None,
        _timeout_s: float,
    ) -> tuple[int, dict[str, Any]]:
        remaining = overall_deadline - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceeded("overall deadline reserve reached during preflight")
        return http_json(request_base_url, endpoint, payload, min(30.0, remaining))

    server = server_preflight(
        base_url,
        args.server_model_path or str(model_path),
        min(30.0, max(0.01, overall_deadline - time.monotonic())),
        request_fn=bounded_preflight_request,
    )
    records: list[dict[str, Any]] = []
    status = "complete"
    writer = FsyncJSONL(output_path)
    try:
        for item_index, item in enumerate(prepared):
            for repetition_index in range(2):
                phase = request_phase(item_index, repetition_index)
                remaining = overall_deadline - time.monotonic()
                if remaining <= 0:
                    status = "deadline"
                    break
                record = run_cycle(
                    base_url,
                    item,
                    phase,
                    remaining,
                    renderer,
                    cycle_deadline_s=deadlines["cycle_s"],
                    diagnostic=args.diagnostic,
                    diagnostic_timeout_s=deadlines["diagnostic_s"],
                )
                writer.write(record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "case_id": record["case_id"],
                            "phase": phase,
                            "response_ok": record.get("response_ok", 0),
                            "parsed_suggestion_latency_ms": record.get(
                                "parsed_suggestion_latency_ms"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if not record.get("response_ok"):
                    status = (
                        "deadline"
                        if "deadline" in str(record.get("error", "")).lower()
                        else "failed"
                    )
            if status == "deadline":
                break
    finally:
        writer.close()
    completed = [record for record in records if record.get("response_ok") == 1]
    summary: dict[str, Any] = {
        "task": TASK,
        "status": status if len(completed) == len(records) else "failed" if status == "complete" else status,
        "scope": {
            "fixture_kind": "synthetic_target_free_legacy_cycle",
            "cases": [item["case"]["id"] for item in prepared],
            "request_phases": ["cold", "different_prompt", "repeat"],
            "cold_definition": "only the first request after the single server preflight",
            "different_prompt_definition": "first request for the next fixture; no cache-state claim",
            "repeat_definition": "same fixture prompt replay with cache_prompt=false; no cache-hit or warm-state claim",
            "endpoint_and_parser_only": True,
            "quality_claim": "none",
        },
        "identities": {
            "authority_expected": contract["accepted_artifacts"],
            "authority_receipts": {
                "accepted": contract["authority_receipt"],
                "fallback": contract["fallback_receipt"],
            },
            "local_verified": static,
            "server_model_path_expected": args.server_model_path or str(model_path),
        },
        "server_preflight": server,
        "request_contract": {
            "completion_endpoint": "/completion",
            "tokenize_endpoint": "/tokenize",
            "n_predict": PRODUCTION_MAX_TOKENS,
            "temperature": 0,
            "stop": list(EXTENSION_STOPS),
            "stream": False,
            "cache_prompt": False,
            "tokenizer": dict(TOKENIZE_REQUEST),
        },
        "cycles": {
            "requested": len(prepared) * 2,
            "written": len(records),
            "successful": len(completed),
            "failed": len(records) - len(completed),
            "by_phase": {
                phase: [
                    record.get("parsed_suggestion_latency_ms")
                    for record in records
                    if record.get("phase") == phase and record.get("response_ok") == 1
                ]
                for phase in ("cold", "different_prompt", "repeat")
            },
        },
        "timing_definitions": {
            "render_ms": "local pinned render_zeta2 call",
            "tokenize_ms": "native /tokenize HTTP round trip",
            "completion_ms": "native /completion HTTP round trip",
            "parse_ms": "pinned parser only",
            "parsed_suggestion_latency_ms": "cycle start through parsed response, including render/tokenize/completion/parse; bounded by the absolute cycle deadline",
            "foreground_cycle_deadline": "five seconds from cycle start through tokenization, completion, and parsing",
        },
        "evidence_gaps": {
            "cancellation": "not_exercised; root must run an editor-owned abort during a live request",
            "application": "only in-memory synthetic application plan; root must collect actual editor document/application evidence",
            "cleanup": "client closes HTTP contexts; root must provide server supervisor process ownership and survivor evidence",
            "host_metrics": "not collected by this client; root must record notebook CPU/RAM/load and context/output sizes",
        },
        "artifacts": {
            "per_cycle_jsonl": {
                "path": str(output_path),
                "rows": len(records),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            },
            "summary_json": {"path": str(summary_path)},
        },
        "limits": [
            "No server, model, CUDA device, editor, or network process is started by this script.",
            "Fixtures are synthetic and target-free; no PRM-03, DEV, quality, or promotion claim is made.",
            "Repeat is an identical replay with cache_prompt=false; no warm-cache or cache-hit claim is made.",
            "The synthetic application record is in-memory endpoint/parser evidence only; no editor quality or no-op correctness score is computed.",
            "The protected b4 model must be separately present and hash-verified on the host that runs the server.",
        ],
        "deadline": {
            **deadlines,
            "overall_stop_before_reserve_s": deadlines["soft_s"] - deadlines["reserve_s"],
            "mode": "diagnostic" if args.diagnostic else "foreground",
            "diagnostic_is_separate_from_foreground": True,
        },
    }
    atomic_json(summary_path, summary)
    summary["artifacts"]["summary_json"]["sha256"] = sha256_file(summary_path)
    rc = EXIT_DEADLINE if summary["status"] == "deadline" else 1 if summary["status"] == "failed" else 0
    return summary, rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18099")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--server-model-path", default=None)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--authority-receipt", type=Path, default=DEFAULT_AUTHORITY_RECEIPT)
    parser.add_argument("--pre08-receipt", type=Path, default=DEFAULT_PRE08_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--soft-deadline-s", type=float, default=None)
    parser.add_argument("--reserve-s", type=float, default=None)
    parser.add_argument("--cycle-deadline-s", type=float, default=None)
    parser.add_argument("--diagnostic-timeout-s", type=float, default=None)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="use the separately labelled diagnostic cycle budget; not a foreground latency run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, rc = run_baseline(args)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"task": TASK, "status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "task": TASK,
                "status": summary["status"],
                "cycles": summary["cycles"],
                "summary": summary["artifacts"]["summary_json"]["path"],
            },
            sort_keys=True,
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
