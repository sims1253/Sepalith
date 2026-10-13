#!/usr/bin/env python3
"""Offline contract checks and root-owned native theta0 notebook probe.

The default ``--self-test`` path has no sockets, subprocesses, models, or
tokenizer files.  The ``--server`` path is intentionally opt-in for root after
the remote server and model have been independently admitted.  It requires a
reference file produced by the pinned tokenizer so a native ``/tokenize``
response cannot be mistaken for tokenizer parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA = "sepalith.run01.theta0-notebook-probe.v1"
BOS_ID = 0
CANONICAL_EOS_ID = 1
NATIVE_EOG_IDS = (1, 130073)
VOCAB_SIZE = 130560
MAX_OUTPUT_TOKENS = 192
CONTEXT_SIZE = 4096
TOKENIZER = {
    "revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
    "tokenizer_json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(raw).hexdigest()


def is_native_control(token: int) -> bool:
    return (0 <= token <= 7) or (10 <= token <= 21) or (130072 <= token < VOCAB_SIZE)


def valid_integer_ids(value: Any, *, allow_controls: bool = True) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < VOCAB_SIZE for token in value)
        and (allow_controls or all(not is_native_control(token) for token in value))
    )


def synthetic_ids(text: str) -> list[int]:
    """A deterministic shape-only tokenizer used by the offline checks.

    These IDs are deliberately not presented as the pinned tokenizer.  They
    only make the three local contract assertions executable without reading a
    tokenizer or model.
    """

    return [100 + (byte % 1000) for byte in text.encode("utf-8")]


def write_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()


def run_self_test() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        checks.append({"name": name, "status": "pass", "detail": detail})

    text = "value <- c(2L, 4L, 8L)\nmean(value)\n"
    tokenize_body = {
        "content": text,
        "add_special": False,
        "parse_special": False,
        "with_pieces": False,
    }
    ids = synthetic_ids(text)
    check(
        "tokenize_flags_and_integer_ids",
        tokenize_body["add_special"] is False
        and tokenize_body["parse_special"] is False
        and tokenize_body["with_pieces"] is False
        and valid_integer_ids(ids, allow_controls=False),
        "synthetic /tokenize shape uses the three pinned false flags and no control IDs",
    )

    prompt = [BOS_ID, *ids]
    canonical_completion = [42, CANONICAL_EOS_ID]
    check(
        "manual_bos_once_and_canonical_eos",
        prompt[0] == BOS_ID
        and prompt.count(BOS_ID) == 1
        and canonical_completion[-1] == CANONICAL_EOS_ID
        and canonical_completion[-1] in NATIVE_EOG_IDS,
        "manual BOS 0 is inserted once; canonical terminal EOS 1 is in native EOG [1, 130073]",
    )

    noncanonical_completion = [42, NATIVE_EOG_IDS[1]]
    check(
        "noncanonical_eog_130073_rejected",
        noncanonical_completion[-1] == 130073
        and noncanonical_completion[-1] in NATIVE_EOG_IDS
        and noncanonical_completion[-1] != CANONICAL_EOS_ID,
        "130073 is detected as native EOG but cannot satisfy the canonical PRM-03 terminal",
    )

    return {
        "schema": SCHEMA,
        "status": "offline_synthetic_checks_passed",
        "synthetic_only": True,
        "tokenizer": TOKENIZER,
        "protocol": {
            "bos_id": BOS_ID,
            "canonical_eos_id": CANONICAL_EOS_ID,
            "native_eog_ids": list(NATIVE_EOG_IDS),
            "vocab_size": VOCAB_SIZE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "context_size": CONTEXT_SIZE,
        },
        "checks": checks,
        "native_claim": "none; live /tokenize and /completion remain root-owned",
    }


def post_json(server: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        f"{server.rstrip('/')}{path}",
        data=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=max(0.1, timeout)) as response:
        raw = response.read()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"HTTP {response.status} from {path}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object JSON from {path}")
    return {"value": value, "response_sha256": hashlib.sha256(raw).hexdigest()}


def load_reference(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "sepalith.run01.theta0-token-reference.v1":
        raise ValueError("reference schema mismatch")
    if value.get("tokenizer") != TOKENIZER:
        raise ValueError("reference tokenizer identity mismatch")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("reference must contain exactly three synthetic cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("reference case shape mismatch")
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate reference case {case_id}")
        seen.add(case_id)
        text = case.get("prompt_text")
        expected = case.get("expected_native_token_ids")
        expected_sha = case.get("expected_native_token_ids_sha256")
        if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != case.get("prompt_sha256"):
            raise ValueError(f"reference prompt hash mismatch for {case_id}")
        if not valid_integer_ids(expected, allow_controls=False) or not isinstance(expected_sha, str):
            raise ValueError(f"reference IDs are not populated for {case_id}")
        if digest(expected) != expected_sha:
            raise ValueError(f"reference ID hash mismatch for {case_id}")
    return value


def classify_completion(response: dict[str, Any], prompt_length: int) -> dict[str, Any]:
    tokens = response.get("tokens")
    evidence: dict[str, Any] = {
        "tokens_integer_ids": valid_integer_ids(tokens),
        "tokens_evaluated": response.get("tokens_evaluated"),
        "tokens_evaluated_matches_prompt": response.get("tokens_evaluated") in (None, prompt_length),
        "stop_type": response.get("stop_type"),
        "content_string": isinstance(response.get("content"), str),
    }
    if not evidence["tokens_integer_ids"] or not tokens:
        evidence["protocol_status"] = "rejected_missing_or_invalid_token_ids"
        return evidence
    terminal = tokens[-1]
    evidence["terminal_id"] = terminal
    evidence["terminal_is_native_eog"] = terminal in NATIVE_EOG_IDS
    evidence["early_control_ids"] = [token for token in tokens[:-1] if is_native_control(token)]
    if terminal == CANONICAL_EOS_ID and not evidence["early_control_ids"] and response.get("stop_type") == "eos":
        evidence["protocol_status"] = "canonical_eos"
    elif terminal == NATIVE_EOG_IDS[1]:
        evidence["protocol_status"] = "noncanonical_eog"
    else:
        evidence["protocol_status"] = "missing_canonical_eos"
    return evidence


def run_live(server: str, reference_path: Path, candidate: str, deadline_ms: int) -> dict[str, Any]:
    reference = load_reference(reference_path)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for case in reference["cases"]:
        row_started = time.monotonic()
        case_id = case["case_id"]
        tokenize_body = {
            "content": case["prompt_text"],
            "add_special": False,
            "parse_special": False,
            "with_pieces": False,
        }
        row: dict[str, Any] = {
            "case_id": case_id,
            "candidate": candidate,
            "prompt_sha256": case["prompt_sha256"],
            "tokenize_request_sha256": digest(tokenize_body),
            "completion_request_sha256": None,
            "applicability": "unavailable",
            "quality_denominator_eligible": False,
            "protocol_evidence": "unavailable",
            "deadline_ms": deadline_ms,
            "started_at_monotonic_ms": round((row_started - started) * 1000, 3),
        }
        try:
            remaining = deadline_ms / 1000 - (time.monotonic() - row_started)
            tokenize_result = post_json(server, "/tokenize", tokenize_body, remaining)
            native = tokenize_result["value"]
            native_ids = native.get("tokens")
            row["tokenize_response_sha256"] = tokenize_result["response_sha256"]
            row["native_token_ids"] = native_ids
            row["native_token_ids_sha256"] = digest(native_ids)
            row["tokenize_integer_ids"] = valid_integer_ids(native_ids, allow_controls=False)
            row["tokenizer_parity"] = native_ids == case["expected_native_token_ids"]
            row["tokenizer_reference_sha256"] = case["expected_native_token_ids_sha256"]
            if not row["tokenize_integer_ids"]:
                row["protocol_evidence"] = "rejected_tokenize_ids"
                continue
            prompt_ids = [BOS_ID, *native_ids]
            row["prompt_ids"] = prompt_ids
            row["manual_bos_count"] = prompt_ids.count(BOS_ID)
            if row["manual_bos_count"] != 1 or prompt_ids[0] != BOS_ID:
                row["protocol_evidence"] = "rejected_bos_contract"
                continue
            completion_body = {
                "prompt": prompt_ids,
                "n_predict": MAX_OUTPUT_TOKENS,
                "temperature": 0,
                "stream": False,
                "cache_prompt": True,
                "return_tokens": True,
            }
            row["completion_request_sha256"] = digest(completion_body)
            remaining = deadline_ms / 1000 - (time.monotonic() - row_started)
            completion_result = post_json(server, "/completion", completion_body, remaining)
            row["completion_response_sha256"] = completion_result["response_sha256"]
            completion = classify_completion(completion_result["value"], len(prompt_ids))
            row["completion"] = completion
            row["protocol_evidence"] = completion["protocol_status"]
            row["applicability"] = "fresh" if completion["protocol_status"] == "canonical_eos" else "rejected"
        except (TimeoutError, HTTPError, URLError, OSError, RuntimeError, ValueError) as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            row["applicability"] = "timeout" if isinstance(error, TimeoutError) else "error"
            row["protocol_evidence"] = "unavailable"
        finally:
            row["elapsed_ms"] = round((time.monotonic() - row_started) * 1000, 3)
            rows.append(row)

    return {
        "schema": SCHEMA,
        "status": "native_probe_complete",
        "synthetic_only": False,
        "candidate": candidate,
        "server": server,
        "tokenizer": TOKENIZER,
        "protocol": {
            "bos_id": BOS_ID,
            "canonical_eos_id": CANONICAL_EOS_ID,
            "native_eog_ids": list(NATIVE_EOG_IDS),
            "vocab_size": VOCAB_SIZE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "context_size": CONTEXT_SIZE,
        },
        "deadline": {
            "normal_or_diagnostic_ms": deadline_ms,
            "timeout_is_excluded_from_quality": True,
            "quality_claims": "none",
        },
        "denominators": {
            "cases": len(rows),
            "tokenizer_parity": sum(row.get("tokenizer_parity") is True for row in rows),
            "canonical_eos": sum(row.get("protocol_evidence") == "canonical_eos" for row in rows),
            "noncanonical_eog": sum(row.get("protocol_evidence") == "noncanonical_eog" for row in rows),
            "timeouts_excluded": sum(row.get("applicability") == "timeout" for row in rows),
            "quality_denominator": 0,
        },
        "reference_path": str(reference_path),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true", help="run offline synthetic checks (default)")
    mode.add_argument("--server", help="already-running root-owned HTTP origin; never launches it")
    parser.add_argument("--reference", type=Path, help="populated pinned-tokenizer reference JSON for --server")
    parser.add_argument("--candidate", choices=("Q8_0", "Q6_K"), help="theta0 quant candidate")
    parser.add_argument("--deadline-ms", type=int, choices=(5000, 60000), default=5000)
    parser.add_argument("--out", type=Path, help="create-only JSON output path")
    args = parser.parse_args()
    if args.server:
        if args.reference is None or args.candidate is None:
            parser.error("--server requires --reference and --candidate")
        result = run_live(args.server, args.reference, args.candidate, args.deadline_ms)
    else:
        result = run_self_test()
    if args.out:
        write_create_only(args.out, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
