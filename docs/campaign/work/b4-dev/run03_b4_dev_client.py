#!/usr/bin/env python3
"""Bounded b4 legacy-renderer replay for the DAT-07 DEV panel.

This client never starts or stops a server.  Root starts the pinned CPU
llama-server and then runs this file against its loopback URL.  The client
loads the exact ``run_eval.render_zeta2`` implementation after checking its
source hash, adapts only the small legacy context contract, asks the server's
own tokenizer for prompt IDs, and sends native ``/completion`` requests so
the full llama.cpp stop metadata is retained.

The input cases are PRM-03 projection records.  Their PRM-03 prompt text is
never used.  Only prefix/region/suffix/history fields that the legacy zeta2
renderer accepts are used; all other context fields remain provenance.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK = "SFT-08/RUN-03"
EXPECTED_CASE_SHA256 = (
    "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
)
EXPECTED_CASE_ROWS = 75
EXPECTED_MODEL_SHA256 = (
    "e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d"
)
EXPECTED_MODEL_BYTES = 2012011904
EXPECTED_RUNTIME_SHA256 = (
    "123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804"
)
EXPECTED_RENDERER_SHA256 = (
    "7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d"
)
EXPECTED_SCENARIOS_SHA256 = (
    "da81802f553e91182ca7dbece30614ad5b12479f3f4e3f5a856a16dcd23fefc3"
)
EXPECTED_NOOP_EVAL_SHA256 = (
    "947d5dc6280bf85b831012116541f067f1104c46a682b3e3dc7ef722893f637d"
)
EXPECTED_EXTENSION_SHA256 = (
    "368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285"
)

EXTENSION_STOPS = [
    ">>>>>>> UPDATED",
    "<<<<<<< CURRENT",
    "=======",
    "<[fim-middle]>",
    "<[fim-suffix]>",
    "<[fim-prefix]>",
    "<|outline|>",
]
# These are the extension's actual production request defaults.  They are
# fixed for every replay row; gold operation labels affect scoring only.
PRODUCTION_MAX_TOKENS = 320
PRODUCTION_STOPS = tuple(EXTENSION_STOPS)
# The old scenario harness used this narrower/longer request.  Keep it as a
# documented comparison point, never as a per-row runtime choice.
HISTORICAL_SCENARIO_MAX_TOKENS = 640
HISTORICAL_SCENARIO_STOPS = (">>>>>>> UPDATED",)
EXPECTED_CONTEXT_SIZE = 8192
LEGACY_CHAR_ADVISORY = 6000
DEFAULT_SOFT_DEADLINE_S = 840.0
DEFAULT_HARD_DEADLINE_S = 900.0
DEFAULT_CHECKPOINT_RESERVE_S = 60.0
EXIT_DEADLINE = 124

SCRIPT = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT.parents[4]
DEFAULT_CASES = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT-07-final-evaluator-cases.jsonl"
)
DEFAULT_MODEL = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf"
)
DEFAULT_SERVER_BINARY = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/"
    "llama-b10453/llama-server"
)
DEFAULT_RUN_DIR = Path(
    "/home/m0hawk/.local/state/sepalith-campaign-20260915/"
    "SFT-08-b4-dev-v1"
)


class UnsupportedCase(ValueError):
    """The legacy renderer cannot represent this semantic case safely."""


class DeadlineExceeded(RuntimeError):
    """The client reached its soft deadline before a safe next step."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise UnsupportedCase(f"{field} must be a list")
    return value


def _require_string_list(value: Any, field: str) -> list[str]:
    values = _require_list(value, field)
    if not all(isinstance(x, str) for x in values):
        raise UnsupportedCase(f"{field} must contain only strings")
    return list(values)


def _load_hashed_module(path: Path, expected_sha256: str, name: str) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing {name} source: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{name} source hash mismatch: {path} has {actual}, "
            f"expected {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legacy_renderer(repo_root: Path) -> Any:
    """Load the pinned renderer without importing a model/framework stack."""
    path = repo_root / "experiments" / "eval" / "run_eval.py"
    module = _load_hashed_module(path, EXPECTED_RENDERER_SHA256, "run_eval")
    if not callable(getattr(module, "render_zeta2", None)):
        raise RuntimeError("pinned run_eval.py has no render_zeta2")
    if not callable(getattr(module, "norm", None)):
        raise RuntimeError("pinned run_eval.py has no norm")
    return module


def verify_static_identities(
    cases_path: Path, model_path: Path, server_binary: Path, repo_root: Path
) -> dict[str, Any]:
    """Hash the exact local inputs before contacting the external server."""
    if not cases_path.is_file():
        raise RuntimeError(f"missing evaluator cases: {cases_path}")
    if not model_path.is_file():
        raise RuntimeError(f"missing b4 model: {model_path}")
    if not server_binary.is_file() or not os.access(server_binary, os.X_OK):
        raise RuntimeError(f"missing or non-executable server: {server_binary}")
    model_bytes = model_path.stat().st_size
    if model_bytes != EXPECTED_MODEL_BYTES:
        raise RuntimeError(
            f"b4 model byte mismatch: {model_path} has {model_bytes}, "
            f"expected {EXPECTED_MODEL_BYTES}"
        )
    cases_sha = sha256_file(cases_path)
    if cases_sha != EXPECTED_CASE_SHA256:
        raise RuntimeError(
            f"evaluator case hash mismatch: {cases_path} has {cases_sha}, "
            f"expected {EXPECTED_CASE_SHA256}"
        )
    model_sha = sha256_file(model_path)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"b4 model hash mismatch: {model_path} has {model_sha}, "
            f"expected {EXPECTED_MODEL_SHA256}"
        )
    runtime_sha = sha256_file(server_binary)
    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError(
            f"llama-server hash mismatch: {server_binary} has {runtime_sha}, "
            f"expected {EXPECTED_RUNTIME_SHA256}"
        )
    renderer_path = repo_root / "experiments" / "eval" / "run_eval.py"
    scenarios_path = repo_root / "experiments" / "eval" / "eval_scenarios.py"
    noop_eval_path = repo_root / "experiments" / "eval" / "eval_noop_fp.py"
    extension_path = repo_root / "extensions" / "vscode-sepalith" / "src" / "extension.ts"
    if not renderer_path.is_file() or sha256_file(renderer_path) != EXPECTED_RENDERER_SHA256:
        raise RuntimeError(f"legacy renderer source identity failed: {renderer_path}")
    if not scenarios_path.is_file() or sha256_file(scenarios_path) != EXPECTED_SCENARIOS_SHA256:
        raise RuntimeError(f"scenario baseline source identity failed: {scenarios_path}")
    if not noop_eval_path.is_file() or sha256_file(noop_eval_path) != EXPECTED_NOOP_EVAL_SHA256:
        raise RuntimeError(f"no-op baseline source identity failed: {noop_eval_path}")
    if not extension_path.is_file() or sha256_file(extension_path) != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError(f"legacy parser source identity failed: {extension_path}")
    return {
        "cases": {
            "path": str(cases_path.resolve()),
            "rows": EXPECTED_CASE_ROWS,
            "sha256": cases_sha,
        },
        "model": {
            "path": str(model_path.resolve()),
            "bytes": model_bytes,
            "sha256": model_sha,
        },
        "runtime": {
            "path": str(server_binary.resolve()),
            "sha256": runtime_sha,
            "build": "10453",
            "commit": "3cb7ffb1a",
            "backend": "CPU, -ngl 0",
        },
        "legacy_renderer": {
            "path": str(renderer_path.resolve()),
            "sha256": EXPECTED_RENDERER_SHA256,
            "implementation": "run_eval.render_zeta2",
        },
        "legacy_baseline": {
            "scenario_path": str(scenarios_path.resolve()),
            "scenario_sha256": EXPECTED_SCENARIOS_SHA256,
            "noop_eval_path": str(noop_eval_path.resolve()),
            "noop_eval_sha256": EXPECTED_NOOP_EVAL_SHA256,
        },
        "legacy_parser": {
            "path": str(extension_path.resolve()),
            "sha256": EXPECTED_EXTENSION_SHA256,
            "implementation": "extension.ts parsePrediction",
        },
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load exactly the frozen 75 rows and retain a source-line identity."""
    rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for line_number, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid case JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"case line {line_number} is not an object")
            row = dict(row)
            row["_source_line_number"] = line_number
            row["_source_line_sha256"] = sha256_bytes(raw.rstrip(b"\r\n"))
            rows.append(row)
    if len(rows) != EXPECTED_CASE_ROWS:
        raise RuntimeError(f"expected {EXPECTED_CASE_ROWS} cases, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise RuntimeError("every case must have a non-empty string id")
    if len(set(ids)) != len(ids):
        raise RuntimeError("evaluator case IDs are not unique")
    if any(row.get("split") != "dev" for row in rows):
        raise RuntimeError("the b4 packet accepts only split=dev evaluator rows")
    return rows


def _history_event(context: dict[str, Any]) -> tuple[str, int]:
    history = _require_list(context.get("history", []), "context.history")
    if len(history) > 1:
        raise UnsupportedCase(
            f"legacy zeta2 accepts one history event, found {len(history)}"
        )
    if not history:
        return "", 0
    event = history[0]
    if not isinstance(event, dict) or not isinstance(event.get("event_diff"), str):
        raise UnsupportedCase("history event lacks a string event_diff")
    return event["event_diff"], 1


def adapt_case(row: dict[str, Any], renderer: Any) -> dict[str, Any]:
    """Project a PRM-03 case to the old line-end-cursor zeta2 contract.

    The new projection stores an actual cursor column.  Legacy zeta2 stores
    only a line index and appends its marker at that line's end.  Splitting
    the selected line at the supplied code-point column moves the marker to a
    line end while moving the untouched remainder into the suffix.  The full
    replacement target stays intact for the historical exact metric.
    """
    context = row.get("context")
    if not isinstance(context, dict):
        raise UnsupportedCase("case has no context object")
    operation = row.get("operation")
    if operation not in ("replace", "no_op"):
        raise UnsupportedCase(f"operation {operation!r} is not in the legacy support set")
    family = row.get("family")
    if not isinstance(family, str) or not family:
        raise UnsupportedCase("case has no semantic family")
    path = context.get("path")
    if not isinstance(path, str) or not path:
        raise UnsupportedCase("context.path must be a non-empty string")
    prefix = _require_string_list(context.get("prefix"), "context.prefix")
    region_old = _require_string_list(context.get("region_old"), "context.region_old")
    suffix_lines = _require_string_list(context.get("suffix_lines"), "context.suffix_lines")
    region_new = _require_string_list(row.get("region_new"), "region_new")
    event_diff, history_count = _history_event(context)

    if operation == "no_op":
        if row.get("target_body_text") != "[NO_EDIT]":
            raise UnsupportedCase("no_op row lacks the frozen [NO_EDIT] sentinel")
        if region_new != region_old:
            raise UnsupportedCase("no_op row does not preserve context.region_old")
    elif not region_new:
        raise UnsupportedCase("empty replacement target is not an edit proposal")

    cursor = context.get("cursor")
    if not isinstance(cursor, dict):
        raise UnsupportedCase("context.cursor must be an object")
    region_line_index = cursor.get("region_line_index")
    code_point_column = cursor.get("code_point_column")
    utf16_column = cursor.get("utf16_column")
    if not isinstance(region_line_index, int) or isinstance(region_line_index, bool):
        raise UnsupportedCase("cursor.region_line_index must be an integer")

    if region_line_index == -1:
        if region_old or code_point_column is not None or utf16_column is not None:
            raise UnsupportedCase(
                "unpositioned legacy cursor is supported only for an empty region"
            )
        legacy_region_old: list[str] = []
        legacy_suffix = list(suffix_lines)
        geometry_mode = "insertion_or_unpositioned_empty_region"
    else:
        if not 0 <= region_line_index < len(region_old):
            raise UnsupportedCase("cursor.region_line_index is outside region_old")
        if not isinstance(code_point_column, int) or isinstance(code_point_column, bool):
            raise UnsupportedCase("cursor.code_point_column must be present")
        if not isinstance(utf16_column, int) or isinstance(utf16_column, bool):
            raise UnsupportedCase("cursor.utf16_column must be present")
        line = region_old[region_line_index]
        if not 0 <= code_point_column <= len(line):
            raise UnsupportedCase("cursor.code_point_column is outside its line")
        if utf16_units(line[:code_point_column]) != utf16_column:
            raise UnsupportedCase(
                "UTF-16/code-point cursor geometry cannot be reconciled safely"
            )
        if operation == "replace":
            # The target must preserve the bytes before the cursor.  This is
            # the lossless condition for the line-end legacy projection.
            if len(region_new) <= region_line_index:
                raise UnsupportedCase("replacement target ends before cursor line")
            if region_new[:region_line_index] != region_old[:region_line_index]:
                raise UnsupportedCase("target prefix before cursor does not match")
            if not region_new[region_line_index].startswith(line[:code_point_column]):
                raise UnsupportedCase("target cursor prefix does not match")
        legacy_region_old = region_old[:region_line_index] + [line[:code_point_column]]
        legacy_suffix = (
            [line[code_point_column:]]
            + region_old[region_line_index + 1 :]
            + suffix_lines
        )
        geometry_mode = (
            "legacy_line_end_cursor_after_codepoint_split"
            if code_point_column < len(line)
            else "legacy_line_end_cursor"
        )

    legacy = {
        "suffix": legacy_suffix,
        "event_diff": event_diff,
        "path": path,
        "prefix": prefix,
        "region_old": legacy_region_old,
        "cursor_idx": len(legacy_region_old) - 1 if legacy_region_old else -1,
    }
    prompt = renderer.render_zeta2(legacy)
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError(f"legacy renderer returned an invalid prompt for {row['id']}")
    if row.get("target_body_text") == "[NO_EDIT]" and "[NO_EDIT]" in prompt:
        raise RuntimeError(f"PRM-03 no-op sentinel leaked into legacy prompt for {row['id']}")
    return {
        "case": row,
        "legacy": legacy,
        "legacy_prompt": prompt,
        "legacy_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "target_lines": region_new,
        "geometry": {
            "mode": geometry_mode,
            "source_region_line_index": region_line_index,
            "source_code_point_column": code_point_column,
            "source_utf16_column": utf16_column,
            "document_eol": context.get("document_eol"),
        },
        "history_count": history_count,
        "legacy_target_sha256": sha256_bytes(
            "\n".join(renderer.norm(region_new)).encode("utf-8")
        ),
    }


# This is the extension's parsePrediction implementation, kept line-for-line
# with the pinned extension source.  The source hash is checked before the
# client starts, so a parser edit cannot silently alter a replay.
MARKER_LINE = re.compile(
    r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|"
    r"<\[fim-(middle|prefix|suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$"
)


def parse_prediction(text: str) -> list[str]:
    """Exact extension.ts parsePrediction, including marker filtering."""
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
    for i in range(2, len(lines)):
        if lines[i] == lines[i - 1] == lines[i - 2]:
            del lines[i:]
            break
    return lines


def make_completion_payload(prompt: str) -> dict[str, Any]:
    """Native /completion request matching the fixed extension defaults."""
    return {
        "prompt": prompt,
        "n_predict": PRODUCTION_MAX_TOKENS,
        "temperature": 0,
        "stop": list(PRODUCTION_STOPS),
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
        "timings_per_token": True,
    }


def make_tokenize_payload(prompt: str) -> dict[str, Any]:
    """Ask llama.cpp to tokenize exactly as its completion route does."""
    return {
        "content": prompt,
        "add_special": True,
        "parse_special": True,
        "with_pieces": False,
    }


def _base_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("b4 client accepts only an http loopback URL")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("server URL must not contain a path, query, or fragment")
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
    with urllib.request.urlopen(request, timeout=max(0.2, timeout_s)) as response:
        raw = response.read()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"{endpoint} returned non-object JSON")
        return response.status, data


def server_preflight(base_url: str, model_path: Path, timeout_s: float) -> dict[str, Any]:
    """Require a responding pinned server and retain its identity responses."""
    health_status, health = http_json(base_url, "/health", None, timeout_s)
    if health_status != 200:
        raise RuntimeError(f"server health status was {health_status}")
    props_status, props = http_json(base_url, "/props", None, timeout_s)
    if props_status != 200:
        raise RuntimeError(f"server props status was {props_status}")
    model_status, models = http_json(base_url, "/v1/models", None, timeout_s)
    if model_status != 200:
        raise RuntimeError(f"server model-list status was {model_status}")
    server_model = props.get("model_path")
    if not isinstance(server_model, str) or Path(server_model).resolve() != model_path.resolve():
        raise RuntimeError(
            f"server model identity mismatch: props.model_path={server_model!r}, "
            f"expected path {str(model_path.resolve())!r}"
        )
    generation = props.get("default_generation_settings")
    if not isinstance(generation, dict):
        raise RuntimeError("server /props lacks default_generation_settings")
    n_ctx = generation.get("n_ctx")
    if n_ctx != EXPECTED_CONTEXT_SIZE:
        raise RuntimeError(
            f"server /props context mismatch: n_ctx={n_ctx!r}, "
            f"expected {EXPECTED_CONTEXT_SIZE}"
        )
    return {
        "health": health,
        "props": props,
        "models": models,
        "model_path_reported": server_model,
        "n_ctx": n_ctx,
    }


def native_tokenize(
    base_url: str, prompt: str, timeout_s: float
) -> dict[str, Any]:
    status, response = http_json(
        base_url, "/tokenize", make_tokenize_payload(prompt), timeout_s
    )
    if status != 200:
        raise RuntimeError(f"tokenize status was {status}")
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        raise RuntimeError("native tokenizer did not return integer token IDs")
    return {
        "request": make_tokenize_payload(prompt),
        "request_sha256": canonical_sha256(make_tokenize_payload(prompt)),
        "response": response,
        "token_count": len(tokens),
        "token_ids": list(tokens),
    }


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("soft deadline reached")
    return remaining


def native_stop_metadata(response: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    tokens = response.get("tokens")
    token_ids = tokens if isinstance(tokens, list) else []
    stop_type = response.get("stop_type")
    predicted = response.get("tokens_predicted")
    cap_hit = int(
        stop_type == "limit"
        or (isinstance(predicted, int) and predicted >= max_tokens)
    )
    return {
        "stop": response.get("stop"),
        "stop_type": stop_type,
        "stopping_word": response.get("stopping_word"),
        "tokens_predicted": predicted,
        "tokens_evaluated": response.get("tokens_evaluated"),
        "tokens_cached": response.get("tokens_cached"),
        "truncated": response.get("truncated"),
        "has_new_line": response.get("has_new_line"),
        "last_generated_token_id": token_ids[-1] if token_ids else None,
        "cap_hit": cap_hit,
    }


def classify_response(
    adapted: dict[str, Any], response: dict[str, Any], renderer: Any
) -> dict[str, Any]:
    """Score a native response without changing the raw completion."""
    content = response.get("content")
    if not isinstance(content, str):
        raise RuntimeError("native completion lacks string content")
    parsed = parse_prediction(content)
    scored = renderer.norm(parsed)
    operation = adapted["case"]["operation"]
    out: dict[str, Any] = {
        "parser_valid": 1,
        "prediction_lines": parsed,
        "prediction_scored_lines": scored,
        "prediction_line_count": len(parsed),
    }
    if operation == "no_op":
        proposal = int(bool(parsed))
        out.update(
            {
                "exact": None,
                "first_line": None,
                "line_f1": None,
                "no_op_proposal": proposal,
                "no_op_correct": int(not proposal),
                "no_op_false_suggestion": proposal,
            }
        )
    else:
        target = renderer.norm(adapted["target_lines"])
        sm = difflib.SequenceMatcher(a=scored, b=target, autojunk=False)
        matched = sum(block.size for block in sm.get_matching_blocks())
        line_f1 = (
            2 * matched / (len(scored) + len(target))
            if (scored or target)
            else 1.0
        )
        out.update(
            {
                "exact": int(scored == target),
                "first_line": int(bool(scored) and bool(target) and scored[0] == target[0]),
                "line_f1": round(line_f1, 4),
                "no_op_proposal": None,
                "no_op_correct": None,
                "no_op_false_suggestion": None,
            }
        )
    return out


def _deadline_settings(args: argparse.Namespace) -> dict[str, float]:
    def env_float(name: str, fallback: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return fallback
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be positive")
        return parsed

    soft = (
        float(args.soft_deadline_s)
        if args.soft_deadline_s is not None
        else env_float("SEPALITH_B4_DEV_SOFT_DEADLINE_S", DEFAULT_SOFT_DEADLINE_S)
    )
    hard = env_float("SEPALITH_B4_DEV_HARD_DEADLINE_S", DEFAULT_HARD_DEADLINE_S)
    reserve = env_float(
        "SEPALITH_B4_DEV_CHECKPOINT_RESERVE_S", DEFAULT_CHECKPOINT_RESERVE_S
    )
    if soft <= 0 or hard <= 0 or reserve <= 0 or soft + reserve > hard:
        raise ValueError(
            f"invalid deadline policy soft={soft} hard={hard} reserve={reserve}"
        )
    return {"soft_s": soft, "hard_s": hard, "checkpoint_reserve_s": reserve}


def _fresh_path(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")


class FsyncJSONL:
    """Append records and make each row durable; fsync failures propagate."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        self.path = path
        self.file = os.fdopen(fd, "w", encoding="utf-8")
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.count += 1

    def close(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _fresh_path(temp)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    with temp.open("xb") as f:
        f.write(data)
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def renderer_contract() -> dict[str, Any]:
    """Return the fixed renderer/request contract used by every replay."""
    return {
        "name": "zeta2",
        "source": "run_eval.render_zeta2",
        "parser": "extension.ts parsePrediction",
        "production_request_max_tokens": PRODUCTION_MAX_TOKENS,
        "production_request_stops": list(PRODUCTION_STOPS),
        "historical_scenario_request": {
            "max_tokens": HISTORICAL_SCENARIO_MAX_TOKENS,
            "stops": list(HISTORICAL_SCENARIO_STOPS),
            "used_for_replay": False,
            "difference_reason": (
                "the production extension uses one fixed request policy; "
                "operation-dependent settings would leak gold labels"
            ),
        },
        "temperature": 0,
        "stream": False,
        "cache_prompt": False,
        "native_tokenization": make_tokenize_payload("<recorded per case>"),
        "unsupported_prm03_fields": [
            "top-level prompt_sha256",
            "context.schema_version",
            "context.replacement_range",
            "context.diagnostics",
            "context.retrieval",
            "context.selected_references",
            "context.scope_lines",
            "context.scope_mode",
            "top-level source_ref",
            "top-level source_provenance",
        ],
        "label_only_fields": [
            "region_new",
            "target_body_text",
            "operation",
            "family",
        ],
        "recorded_only_fields": ["context.document_eol"],
        "geometry_policy": (
            "consume code_point_column and verify utf16_column, then split "
            "the selected line so the legacy marker is at line end"
        ),
    }


def _expected_record_identity(item: dict[str, Any], renderer: Any) -> dict[str, Any]:
    """Build the immutable per-case fields that a resume record must retain."""
    case = item["case"]
    payload = make_completion_payload(item["legacy_prompt"])
    tokenize_request = make_tokenize_payload(item["legacy_prompt"])
    return {
        "task": TASK,
        "case": {
            "id": case["id"],
            "family": case["family"],
            "operation": case["operation"],
            "package_id": case.get("package_id"),
            "group_id": case.get("group_id"),
            "split": case.get("split"),
            "source_line_number": case["_source_line_number"],
            "source_line_sha256": case["_source_line_sha256"],
            "source_prompt_sha256_prm03_ignored": case.get("prompt_sha256"),
            "source_target_sha256": case.get("target_sha256"),
        },
        "legacy_renderer": {
            "name": "zeta2",
            "implementation": "run_eval.render_zeta2",
            "prompt_sha256": item["legacy_prompt_sha256"],
            "prompt_chars": len(item["legacy_prompt"]),
            "geometry": item["geometry"],
            "history_count": item["history_count"],
        },
        "legacy_prompt": item["legacy_prompt"],
        "legacy_target_sha256": item["legacy_target_sha256"],
        "legacy_target_line_count": len(renderer.norm(item["target_lines"])),
        "tokenize_request": tokenize_request,
        "tokenize_request_sha256": canonical_sha256(tokenize_request),
        "completion_request": {
            "endpoint": "/completion",
            "payload": payload,
            "payload_sha256": canonical_sha256(payload),
            "max_tokens": PRODUCTION_MAX_TOKENS,
            "stops": list(PRODUCTION_STOPS),
        },
    }


def _assert_equal_identity(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"resume {label} identity mismatch")


def load_resume_successful_jsonl(
    resume_path: Path,
    resume_summary_path: Path | None,
    adapted: list[dict[str, Any]],
    static: dict[str, Any],
    renderer: Any,
) -> dict[str, Any]:
    """Validate an immutable prior partial run and select successful IDs.

    The JSONL is intentionally not merged into the new output.  Successful
    records identify rows that must not be requested again; failed records stay
    in the prior archive and their IDs remain in the next request segment.
    """
    resume_path = resume_path.resolve()
    if not resume_path.is_file():
        raise RuntimeError(f"missing resume JSONL: {resume_path}")
    summary_path = (
        resume_summary_path.resolve()
        if resume_summary_path is not None
        else resume_path.with_name("summary.json")
    )
    if not summary_path.is_file():
        raise RuntimeError(
            f"resume summary is required to verify model/runtime identity: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read resume summary {summary_path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise RuntimeError("resume summary must be a JSON object")
    _assert_equal_identity(summary.get("task"), TASK, "task")

    previous_static = summary.get("static_identities")
    _assert_equal_identity(previous_static, static, "static source/model/runtime")
    _assert_equal_identity(
        summary.get("renderer_contract"), renderer_contract(), "renderer/request"
    )

    expected_by_id = {item["case"]["id"]: item for item in adapted}
    expected_ids = list(expected_by_id)
    expected_families = dict(sorted(Counter(item["case"]["family"] for item in adapted).items()))
    expected_operations = dict(sorted(Counter(item["case"]["operation"] for item in adapted).items()))
    coverage = summary.get("coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("resume summary lacks coverage identity")
    for key, expected in (
        ("requested_rows", len(adapted)),
        ("adapted_rows", len(adapted)),
        ("excluded_rows", 0),
        ("families", expected_families),
        ("operations", expected_operations),
        ("legacy_prompt_hashes_unique", True),
    ):
        _assert_equal_identity(coverage.get(key), expected, f"coverage.{key}")

    artifact = summary.get("artifacts", {}).get("per_request_jsonl")
    if not isinstance(artifact, dict):
        raise RuntimeError("resume summary lacks per-request artifact identity")
    recorded_path = artifact.get("path")
    if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != resume_path:
        raise RuntimeError("resume per-request artifact path identity mismatch")
    actual_sha = sha256_file(resume_path)
    _assert_equal_identity(artifact.get("sha256"), actual_sha, "per-request archive")
    _assert_equal_identity(artifact.get("bytes"), resume_path.stat().st_size, "per-request bytes")

    records: list[dict[str, Any]] = []
    with resume_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid resume JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"resume line {line_number} is not an object")
            records.append(record)
    _assert_equal_identity(artifact.get("rows"), len(records), "per-request row count")
    denominator = summary.get("denominators", {})
    _assert_equal_identity(denominator.get("completed_rows"), len(records), "completed row count")

    seen: set[str] = set()
    successful_ids: list[str] = []
    failed_ids: list[str] = []
    failed_errors: dict[str, str | None] = {}
    for record in records:
        if record.get("task") != TASK:
            raise RuntimeError("resume task identity mismatch")
        case_record = record.get("case")
        if not isinstance(case_record, dict) or not isinstance(case_record.get("id"), str):
            raise RuntimeError("resume record lacks case.id")
        case_id = case_record["id"]
        if case_id in seen:
            raise RuntimeError(f"duplicate resume case ID: {case_id}")
        seen.add(case_id)
        item = expected_by_id.get(case_id)
        if item is None:
            raise RuntimeError(f"resume case ID is outside frozen 75-row population: {case_id}")
        expected = _expected_record_identity(item, renderer)
        _assert_equal_identity(record.get("case"), expected["case"], f"case {case_id}")
        _assert_equal_identity(
            record.get("legacy_renderer"), expected["legacy_renderer"], f"renderer {case_id}"
        )
        _assert_equal_identity(record.get("legacy_prompt"), expected["legacy_prompt"], f"prompt {case_id}")
        _assert_equal_identity(
            record.get("legacy_target_sha256"), expected["legacy_target_sha256"], f"target {case_id}"
        )
        _assert_equal_identity(
            record.get("legacy_target_line_count"),
            expected["legacy_target_line_count"],
            f"target line count {case_id}",
        )
        token_info = record.get("native_tokenization")
        if not isinstance(token_info, dict):
            raise RuntimeError(f"resume native tokenizer identity missing: {case_id}")
        _assert_equal_identity(token_info.get("request"), expected["tokenize_request"], f"tokenizer {case_id}")
        _assert_equal_identity(
            token_info.get("request_sha256"), expected["tokenize_request_sha256"], f"tokenizer hash {case_id}"
        )
        _assert_equal_identity(record.get("request"), expected["completion_request"], f"request {case_id}")
        response_ok = record.get("response_ok")
        if not isinstance(response_ok, int) or isinstance(response_ok, bool) or response_ok not in (0, 1):
            raise RuntimeError(f"resume response_ok must be integer 0/1: {case_id}")
        if response_ok == 1:
            successful_ids.append(case_id)
        else:
            failed_ids.append(case_id)
            error = record.get("error")
            failed_errors[case_id] = error if isinstance(error, str) else None

    # Return IDs in frozen population order, regardless of prior file order.
    successful_ids = [case_id for case_id in expected_ids if case_id in successful_ids]
    failed_ids = [case_id for case_id in expected_ids if case_id in failed_ids]
    remaining_ids = [case_id for case_id in expected_ids if case_id not in successful_ids]
    return {
        "mode": "resume_successful_jsonl",
        "path": str(resume_path),
        "sha256": actual_sha,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_status": summary.get("status"),
        "rows": len(records),
        "successful_ids": successful_ids,
        "failed_ids": failed_ids,
        "failed_errors": {case_id: failed_errors.get(case_id) for case_id in failed_ids},
        "already_complete": len(successful_ids),
        "remaining_ids": remaining_ids,
        "replayed_failed_ids": [case_id for case_id in remaining_ids if case_id in failed_ids],
    }


def build_resume_state(
    adapted: list[dict[str, Any]],
    previous: dict[str, Any] | None,
    current_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize population, resume, current, and root-review union separately."""
    population_ids = [item["case"]["id"] for item in adapted]
    previous_successful = list(previous.get("successful_ids", [])) if previous else []
    previous_failed = list(previous.get("failed_ids", [])) if previous else []
    previous_successful_set = set(previous_successful)
    requested_ids = [
        case_id for case_id in population_ids if case_id not in previous_successful_set
    ]
    current_successful = [
        record["case"]["id"] for record in current_records if record.get("response_ok") == 1
    ]
    current_failed = [
        record["case"]["id"] for record in current_records if record.get("response_ok") == 0
    ]
    all_result_ids = previous_successful + current_successful
    available_ids = [
        case_id
        for case_id in population_ids
        if case_id in set(previous_successful) or case_id in set(current_successful)
    ]
    missing_ids = [case_id for case_id in population_ids if case_id not in set(available_ids)]
    return {
        "mode": "resume_successful_jsonl" if previous else "fresh",
        "total_population": len(population_ids),
        "already_complete": len(previous_successful),
        "requested": len(population_ids) - len(previous_successful),
        "current_complete": len(current_successful),
        "current_failed": len(current_failed),
        "previous_successful_ids": previous_successful,
        "previous_failed_ids": previous_failed,
        "replayed_failed_ids": [
            case_id for case_id in requested_ids if case_id in set(previous_failed)
        ],
        "current_successful_ids": current_successful,
        "current_failed_ids": current_failed,
        "available_result_ids": available_ids,
        "missing_ids": missing_ids,
        "union": {
            "available_result_count": len(available_ids),
            "missing_count": len(missing_ids),
            "complete": not missing_ids,
            "one_result_per_id": len(all_result_ids) == len(set(all_result_ids))
            and set(all_result_ids).issubset(set(population_ids)),
            "root_review_required": True,
            "accepted_by_root": False,
            "one_accepted_result_per_id": False,
        },
    }


def remaining_adapted(
    adapted: list[dict[str, Any]], previous: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Select the next request segment while preserving frozen population order."""
    completed = set(previous.get("successful_ids", [])) if previous else set()
    population = {item["case"]["id"] for item in adapted}
    unknown = completed - population
    if unknown:
        raise RuntimeError(
            "resume successful IDs are outside frozen population: "
            + ", ".join(sorted(unknown))
        )
    return [item for item in adapted if item["case"]["id"] not in completed]


def _summary_counts(records: list[dict[str, Any]], adapted: list[dict[str, Any]]) -> dict[str, Any]:
    requested = len(adapted)
    rows = len(records)
    response_ok = sum(int(record.get("response_ok", 0)) for record in records)
    parser_valid = sum(int(record.get("parser_valid", 0)) for record in records)
    replacements = sum(1 for item in adapted if item["case"]["operation"] == "replace")
    noops = requested - replacements
    exact_records = [record for record in records if record.get("exact") is not None]
    noop_records = [record for record in records if record.get("no_op_correct") is not None]
    cap_records = [record for record in records if record.get("cap_eligible")]
    by_family: dict[str, dict[str, Any]] = {}
    family_names = sorted({item["case"]["family"] for item in adapted})
    for family in family_names:
        items = [item for item in adapted if item["case"]["family"] == family]
        ids = {item["case"]["id"] for item in items}
        rs = [record for record in records if record["case"]["id"] in ids]
        ers = [record for record in rs if record.get("exact") is not None]
        nrs = [record for record in rs if record.get("no_op_correct") is not None]
        crs = [record for record in rs if record.get("cap_eligible")]
        by_family[family] = {
            "requested": len(items),
            "completed": len(rs),
            "replace_rows": sum(item["case"]["operation"] == "replace" for item in items),
            "no_op_rows": sum(item["case"]["operation"] == "no_op" for item in items),
            "parser_denominator": sum(record.get("response_ok", 0) for record in rs),
            "parser_valid": sum(record.get("parser_valid", 0) for record in rs),
            "exact_n": len(ers),
            "exact_k": sum(record.get("exact", 0) for record in ers),
            "no_op_n": len(nrs),
            "no_op_correct_k": sum(record.get("no_op_correct", 0) for record in nrs),
            "no_op_false_suggestion_k": sum(
                record.get("no_op_false_suggestion", 0) for record in nrs
            ),
            "cap_n": len(crs),
            "cap_hit_k": sum(record.get("cap_hit", 0) for record in crs),
        }
    return {
        "requested_rows": requested,
        "completed_rows": rows,
        "response_denominator": response_ok,
        "parser_denominator": response_ok,
        "parser_valid_k": parser_valid,
        "replace_rows": replacements,
        "exact_denominator": len(exact_records),
        "exact_k": sum(record.get("exact", 0) for record in exact_records),
        "no_op_rows": noops,
        "no_op_denominator": len(noop_records),
        "no_op_correct_k": sum(record.get("no_op_correct", 0) for record in noop_records),
        "no_op_false_suggestion_k": sum(
            record.get("no_op_false_suggestion", 0) for record in noop_records
        ),
        "cap_denominator": len(cap_records),
        "cap_hit_k": sum(record.get("cap_hit", 0) for record in cap_records),
        "by_family": by_family,
    }


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cases_path = Path(args.cases).resolve()
    model_path = Path(args.model).resolve()
    server_binary = Path(args.server_binary).resolve()
    repo_root = Path(args.repo_root).resolve()
    base_url = _base_url(args.url)
    deadlines = _deadline_settings(args)
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()
    _fresh_path(output_path)
    _fresh_path(summary_path)

    static = verify_static_identities(cases_path, model_path, server_binary, repo_root)
    renderer = load_legacy_renderer(repo_root)
    rows = load_cases(cases_path)
    adapted = [adapt_case(row, renderer) for row in rows]
    previous_resume = None
    if args.resume_successful_jsonl is not None:
        previous_resume = load_resume_successful_jsonl(
            Path(args.resume_successful_jsonl),
            Path(args.resume_summary) if args.resume_summary is not None else None,
            adapted,
            static,
            renderer,
        )
    requested_adapted = remaining_adapted(adapted, previous_resume)
    prompt_hashes = [item["legacy_prompt_sha256"] for item in adapted]
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise RuntimeError("legacy-rendered prompt collision in the 75-row panel")
    geometry_counts = Counter(item["geometry"]["mode"] for item in adapted)
    advisory_over = sum(
        len(item["legacy_prompt"]) > LEGACY_CHAR_ADVISORY for item in adapted
    )
    deadline_started = time.monotonic()
    server_info = server_preflight(base_url, model_path, min(30.0, _remaining(deadline_started + deadlines["soft_s"])))
    tokenized: list[dict[str, Any]] = []
    for item in requested_adapted:
        remaining = _remaining(deadline_started + deadlines["soft_s"])
        token_info = native_tokenize(base_url, item["legacy_prompt"], min(30.0, remaining))
        if token_info["token_count"] + PRODUCTION_MAX_TOKENS >= server_info["n_ctx"]:
            raise RuntimeError(
                f"native prompt budget exceeds server context for {item['case']['id']}: "
                f"{token_info['token_count']}+{PRODUCTION_MAX_TOKENS}>={server_info['n_ctx']}"
            )
        tokenized.append(token_info)

    writer = FsyncJSONL(output_path)
    records: list[dict[str, Any]] = []
    status = "complete"
    deadline_at = deadline_started + deadlines["soft_s"]
    try:
        for item, token_info in zip(requested_adapted, tokenized):
            case = item["case"]
            try:
                remaining = _remaining(deadline_at)
            except DeadlineExceeded:
                status = "deadline"
                break
            max_tokens = PRODUCTION_MAX_TOKENS
            stops = list(PRODUCTION_STOPS)
            payload = make_completion_payload(item["legacy_prompt"])
            record: dict[str, Any] = {
                "task": TASK,
                "case": {
                    "id": case["id"],
                    "family": case["family"],
                    "operation": case["operation"],
                    "package_id": case.get("package_id"),
                    "group_id": case.get("group_id"),
                    "split": case.get("split"),
                    "source_line_number": case["_source_line_number"],
                    "source_line_sha256": case["_source_line_sha256"],
                    "source_prompt_sha256_prm03_ignored": case.get("prompt_sha256"),
                    "source_target_sha256": case.get("target_sha256"),
                },
                "legacy_renderer": {
                    "name": "zeta2",
                    "implementation": "run_eval.render_zeta2",
                    "prompt_sha256": item["legacy_prompt_sha256"],
                    "prompt_chars": len(item["legacy_prompt"]),
                    "geometry": item["geometry"],
                    "history_count": item["history_count"],
                },
                "legacy_prompt": item["legacy_prompt"],
                "legacy_target_sha256": item["legacy_target_sha256"],
                "legacy_target_line_count": len(renderer.norm(item["target_lines"])),
                "native_tokenization": token_info,
                "request": {
                    "endpoint": "/completion",
                    "payload": payload,
                    "payload_sha256": canonical_sha256(payload),
                    "max_tokens": max_tokens,
                    "stops": list(stops),
                },
                "started_monotonic_s": round(time.monotonic() - deadline_started, 6),
            }
            try:
                status_code, response = http_json(
                    base_url,
                    "/completion",
                    payload,
                    min(600.0, max(0.2, remaining)),
                )
                if status_code != 200:
                    raise RuntimeError(f"completion status was {status_code}")
                scored = classify_response(item, response, renderer)
                record.update(scored)
                record["response_ok"] = 1
                record["native_stop"] = native_stop_metadata(response, max_tokens)
                record["cap_eligible"] = 1
                record["cap_hit"] = record["native_stop"]["cap_hit"]
                record["raw_completion"] = response.get("content")
                record["native_response"] = response
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                record.update(
                    {
                        "response_ok": 0,
                        "parser_valid": 0,
                        "prediction_lines": None,
                        "prediction_scored_lines": None,
                        "prediction_line_count": None,
                        "exact": None,
                        "first_line": None,
                        "line_f1": None,
                        "no_op_proposal": None,
                        "no_op_correct": None,
                        "no_op_false_suggestion": None,
                        "cap_eligible": 0,
                        "cap_hit": None,
                        "error": str(exc)[:300],
                    }
                )
                if time.monotonic() >= deadline_at:
                    status = "deadline"
                else:
                    # Keep the row and continue, but never label a run with
                    # a transport/model error as a complete benchmark.
                    status = "failed"
            record["finished_monotonic_s"] = round(time.monotonic() - deadline_started, 6)
            writer.write(record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "family": case["family"],
                        "operation": case["operation"],
                        "response_ok": record.get("response_ok", 0),
                        "exact": record.get("exact"),
                        "no_op_correct": record.get("no_op_correct"),
                        "cap_hit": record.get("cap_hit"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        writer.close()

    counts = _summary_counts(records, requested_adapted)
    resume_state = build_resume_state(adapted, previous_resume, records)
    summary: dict[str, Any] = {
        "task": TASK,
        "status": status,
        "started_at_unix_s": time.time() - (time.monotonic() - deadline_started),
        "finished_at_unix_s": time.time(),
        "url": base_url,
        "static_identities": static,
        "server_preflight": server_info,
        "deadline_policy": deadlines,
        "renderer_contract": renderer_contract(),
        "resume": {
            **resume_state,
            "previous_archive": previous_resume,
            "request_order": [item["case"]["id"] for item in requested_adapted],
        },
        "coverage": {
            "requested_rows": len(adapted),
            "current_requested_rows": len(requested_adapted),
            "adapted_rows": len(adapted),
            "excluded_rows": 0,
            "legacy_prompt_hashes_unique": True,
            "geometry_modes": dict(sorted(geometry_counts.items())),
            "legacy_char_advisory": LEGACY_CHAR_ADVISORY,
            "legacy_char_advisory_over_rows": advisory_over,
            "families": dict(
                sorted(
                    Counter(item["case"]["family"] for item in adapted).items()
                )
            ),
            "operations": dict(
                sorted(Counter(item["case"]["operation"] for item in adapted).items())
            ),
        },
        "denominators": counts,
        "artifacts": {
            "per_request_jsonl": {
                "path": str(output_path),
                "rows": len(records),
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
            },
            "summary_json": {"path": str(summary_path)},
        },
        "limits": [
            "No model or server is started by this client; root owns the pinned CPU launch.",
            "The DAT-07 PRM-03 prompt_sha256 is retained for identity only and is never sent.",
            "Deletion is unsupported by the legacy inline-proposal protocol and is rejected by adapt_case.",
            "The b4 fallback is historical evidence; this packet makes no Tuesday quality claim.",
        ],
    }
    atomic_json(summary_path, summary)
    summary["artifacts"]["summary_json"]["sha256"] = sha256_file(summary_path)
    return summary, (
        EXIT_DEADLINE
        if status == "deadline"
        else 1
        if status == "failed"
        else 0
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:18099")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--server-binary", type=Path, default=DEFAULT_SERVER_BINARY)
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_RUN_DIR / "per_request.jsonl")
    ap.add_argument("--summary", type=Path, default=DEFAULT_RUN_DIR / "summary.json")
    ap.add_argument(
        "--resume-successful-jsonl",
        type=Path,
        default=None,
        help=(
            "immutable prior per-request JSONL; successful response_ok=1 rows "
            "are excluded and failed rows are replayed"
        ),
    )
    ap.add_argument(
        "--resume-summary",
        type=Path,
        default=None,
        help="prior summary JSON; defaults to summary.json beside --resume-successful-jsonl",
    )
    ap.add_argument("--soft-deadline-s", type=float, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary, rc = run_benchmark(args)
    except DeadlineExceeded as exc:
        print(json.dumps({"task": TASK, "status": "deadline", "error": str(exc)}), file=sys.stderr)
        return EXIT_DEADLINE
    except Exception as exc:  # fail closed; do not claim a partial benchmark
        print(json.dumps({"task": TASK, "status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "task": TASK,
                "status": summary["status"],
                "completed_rows": summary["denominators"]["completed_rows"],
                "summary": summary["artifacts"]["summary_json"]["path"],
            },
            sort_keys=True,
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
