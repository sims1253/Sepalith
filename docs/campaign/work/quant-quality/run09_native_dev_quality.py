#!/usr/bin/env python3
"""Bounded native PRM-03 quality replay for the frozen 75-case DEV panel.

This is a client and scorer.  It never starts a server, chooses a model, or
downloads an artifact.  The root wrapper supplies the loopback server and an
opaque model/quantization provenance string.  The client verifies the exact
DEV panel, the frozen HF tokenizer and the pinned PRM-03 renderer before
contacting the server, then renders every case from ``context`` only.

Each case first asks llama.cpp ``/tokenize`` with explicit no-special flags,
prepends the one manual BOS ID 0, and sends that integer prompt to
``/completion``.  The quality route streams so token IDs, EOS and timing are
retained; it uses deterministic native settings with an independent fresh
request per case (temperature zero, ``cache_prompt`` false, no text stop
list).  A stream failure, tokenizer mismatch, noncanonical EOG, cap violation
or parser error is retained as a failed case and is never silently scored as
quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = "sepalith.campaign.run09.native-dev-quant-quality.v2"
TASK = "RUN-09-native-dev-quant-quality"
PANEL_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
PANEL_ROWS = 75
EDIT_CASES = 43
STRICT_NOOP_CASES = 32
TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_CONFIG_SHA256 = "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"
CAMPAIGN_EVAL_SHA256 = "7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f"
CAMPAIGN_CLIENT_SHA256 = "0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333"
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = (1, 130073)
VOCAB_SIZE = 130560
PROMPT_CAP = 4_096
COMPLETION_CAP = 512
MAX_CASE_DEADLINE_SECONDS = 120
MAX_GLOBAL_DEADLINE_SECONDS = 7_200
DEFAULT_CASE_DEADLINE_SECONDS = 120
DEFAULT_GLOBAL_DEADLINE_SECONDS = 7_200
DEFAULT_RESERVE_SECONDS = 60
CONTROL_RANGES = ((0, 7), (10, 21), (130072, VOCAB_SIZE - 1))
PANEL_PATH = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl"
)
TOKENIZER_PATH = Path(
    "/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain"
)
EXECUTION_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")


class QualityError(RuntimeError):
    """A static, transport or quality contract failure."""


class CaseTimeout(QualityError):
    """A case exceeded its combined tokenize plus completion deadline."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(block_size), b""):
                digest.update(block)
    except OSError as error:
        raise QualityError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json(value))


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_fresh_ext4(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise QualityError("output must be an absolute path")
    if path.exists():
        raise QualityError(f"output must be fresh: {path}")
    if not path.parent.is_dir():
        raise QualityError(f"output parent must already exist: {path.parent}")
    try:
        import subprocess
        # GNU stat reports ext4 as the compatibility type ``ext2/ext3`` on
        # some hosts.  findmnt reports the mounted filesystem type directly.
        result = subprocess.run(
            ["/usr/bin/findmnt", "-T", str(path.parent), "-n", "-o", "FSTYPE"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise QualityError(f"cannot identify output filesystem: {path.parent}") from error
    filesystem = result.stdout.strip()
    if filesystem != "ext4":
        raise QualityError(f"output must be native ext4, observed {filesystem!r}")
    return path


def _loopback_origin(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or parsed.path not in {"", "/"} \
            or parsed.query or parsed.fragment or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise QualityError("server URL must be an HTTP loopback origin without a path")
    return f"{parsed.scheme}://{parsed.netloc}"


def _integer_ids(value: object, label: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise QualityError(f"{label} must be a nonempty integer-ID array")
    if any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in value):
        raise QualityError(f"{label} contains an out-of-range token ID")
    return list(value)


def _is_control(token: int) -> bool:
    return any(low <= token <= high for low, high in CONTROL_RANGES)


def _canonical_generation_tokens_valid(protocol: Any, generated: Sequence[int]) -> bool:
    """Use the pinned protocol guard for every quality metric."""
    validator = getattr(protocol, "valid_generation_tokens", None)
    if not callable(validator):
        raise QualityError("pinned protocol does not expose valid_generation_tokens")
    try:
        return bool(validator(generated))
    except Exception as error:
        raise QualityError(f"pinned generation-token guard failed: {error}") from error


def _protocol_modules(execution_root: Path) -> tuple[Any, Path, Path, Path]:
    execution_root = Path(execution_root).resolve()
    package_src = execution_root / "packages" / "sepalith" / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    protocol_path = package_src / "sepalith" / "campaign_protocol.py"
    eval_path = execution_root / "experiments" / "training" / "campaign_eval.py"
    client_path = execution_root / "extensions" / "vscode-sepalith" / "src" / "campaign_client.ts"
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise QualityError(f"PRM-03 protocol hash mismatch: {protocol_path}")
    if sha256_file(eval_path) != CAMPAIGN_EVAL_SHA256:
        raise QualityError(f"development evaluator source hash mismatch: {eval_path}")
    if sha256_file(client_path) != CAMPAIGN_CLIENT_SHA256:
        raise QualityError(f"native client source hash mismatch: {client_path}")
    try:
        protocol = importlib.import_module("sepalith.campaign_protocol")
    except Exception as error:
        raise QualityError(f"cannot import pinned PRM-03 protocol: {error}") from error
    return protocol, protocol_path, eval_path, client_path


def _load_panel(path: Path, protocol: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(path) != PANEL_SHA256:
        raise QualityError(f"DEV panel hash mismatch: {path}")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise QualityError(f"blank DEV panel line {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise QualityError(f"invalid DEV panel JSON line {line_number}") from error
                if not isinstance(row, Mapping):
                    raise QualityError(f"DEV panel line {line_number} is not an object")
                row = dict(row)
                row_id = row.get("id")
                if row.get("split") != "dev" or not isinstance(row_id, str) or not row_id:
                    raise QualityError(f"DEV panel row {row_id!r} has invalid split or ID")
                if row_id in ids:
                    raise QualityError(f"duplicate DEV panel ID {row_id}")
                if not isinstance(row.get("package_id"), str) or not row["package_id"]:
                    raise QualityError(f"DEV panel row {row_id} lacks package identity")
                if not isinstance(row.get("family"), str) or not row["family"]:
                    raise QualityError(f"DEV panel row {row_id} lacks family")
                operation = row.get("operation")
                if operation not in {"replace", "no_op", "delete"}:
                    raise QualityError(f"DEV panel row {row_id} has unsupported operation")
                region_new = row.get("region_new")
                if not isinstance(region_new, list) or any(not isinstance(x, str) for x in region_new):
                    raise QualityError(f"DEV panel row {row_id} lacks region_new lines")
                try:
                    context = protocol.PromptContext.from_mapping(row.get("context"))
                except Exception as error:
                    raise QualityError(f"DEV panel context invalid for {row_id}: {error}") from error
                if operation == "no_op" and list(context.region_old) != region_new:
                    raise QualityError(f"DEV no-op target differs from region_old for {row_id}")
                if operation == "replace" and list(context.region_old) == region_new:
                    raise QualityError(f"DEV replace target is unchanged for {row_id}")
                rows.append({"row": row, "context": context})
                ids.add(row_id)
    except OSError as error:
        raise QualityError(f"cannot read DEV panel: {path}") from error
    if len(rows) != PANEL_ROWS:
        raise QualityError(f"DEV panel count is {len(rows)}, expected {PANEL_ROWS}")
    ids_ordered = [item["row"]["id"] for item in rows]
    return rows, {
        "path": str(Path(path).resolve()),
        "sha256": PANEL_SHA256,
        "rows": len(rows),
        "case_ids_sha256": canonical_sha256(ids_ordered),
        "case_ids": ids_ordered,
        "families": dict(sorted(Counter(item["row"]["family"] for item in rows).items())),
        "operations": dict(sorted(Counter(item["row"]["operation"] for item in rows).items())),
    }


def _load_tokenizer(tokenizer_dir: Path) -> tuple[Any, dict[str, Any]]:
    tokenizer_dir = Path(tokenizer_dir).resolve()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    if sha256_file(tokenizer_json) != TOKENIZER_JSON_SHA256:
        raise QualityError(f"tokenizer.json hash mismatch: {tokenizer_json}")
    if sha256_file(tokenizer_config) != TOKENIZER_CONFIG_SHA256:
        raise QualityError(f"tokenizer_config.json hash mismatch: {tokenizer_config}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), local_files_only=True, use_fast=True, trust_remote_code=False,
        )
    except Exception as error:
        raise QualityError(f"cannot load frozen local HF tokenizer: {error}") from error
    identity = {
        "vocab_size": len(tokenizer),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    _validate_tokenizer_identity(identity)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert) or convert(EOS_ID) != "</s>":
        raise QualityError("frozen tokenizer EOS ID 1 is not </s>")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or len(vocab) != VOCAB_SIZE:
        raise QualityError("frozen tokenizer vocabulary mapping is not complete")
    vocab_digest = canonical_sha256(sorted((str(k), int(v)) for k, v in vocab.items()))
    return tokenizer, {"path": str(tokenizer_dir), "files": {
        "tokenizer.json": TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
    }, "revision": TOKENIZER_REVISION, "identity": identity,
        "vocab_sha256": vocab_digest}


def _validate_tokenizer_identity(identity: Mapping[str, Any]) -> None:
    expected = {"vocab_size": VOCAB_SIZE, "bos_token_id": BOS_ID,
                "eos_token_id": EOS_ID, "pad_token_id": EOS_ID}
    if dict(identity) != expected:
        raise QualityError(f"frozen tokenizer identity mismatch: {dict(identity)}")


def _prepare_cases(rows: Sequence[Mapping[str, Any]], protocol: Any) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for item in rows:
        row = item["row"]
        context = item["context"]
        prompt = protocol.render_prompt(context)
        prepared.append({
            "id": row["id"],
            "family": row["family"],
            "package_id": row["package_id"],
            "operation": row["operation"],
            "expected_region": list(row["region_new"]),
            "expected_noop": row["operation"] == "no_op",
            "context": context,
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "target_sha256": sha256_bytes(canonical_json(row["region_new"])),
        })
    return prepared


def _encode_cases(cases: Sequence[dict[str, Any]], tokenizer: Any, protocol: Any) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    for case in cases:
        try:
            prompt_ids = protocol.encode_prompt(case["context"], tokenizer, include_bos=True)
        except Exception as error:
            raise QualityError(f"HF tokenizer cannot encode {case['id']}: {error}") from error
        if not prompt_ids or prompt_ids[0] != BOS_ID:
            raise QualityError(f"manual BOS identity failed for {case['id']}")
        if len(prompt_ids) + COMPLETION_CAP > PROMPT_CAP:
            raise QualityError(
                f"context budget failed for {case['id']}: {len(prompt_ids)} + {COMPLETION_CAP} > {PROMPT_CAP}"
            )
        case["hf_prompt_ids"] = list(prompt_ids)
        case["hf_prompt_tokens_with_bos"] = len(prompt_ids)
        case["hf_prompt_ids_sha256"] = canonical_sha256(prompt_ids)
        prompt_lengths.append(len(prompt_ids))
    return {"min_tokens_with_bos": min(prompt_lengths), "max_tokens_with_bos": max(prompt_lengths),
            "all_within_context": True}


def make_tokenize_payload(prompt: str) -> dict[str, Any]:
    """Exact b10453 PRM-03 text-tokenization request."""
    return {"content": prompt, "add_special": False,
            "parse_special": False, "with_pieces": False}


def make_completion_payload(prompt_ids: Sequence[int]) -> dict[str, Any]:
    """Exact independent deterministic quality request; the prompt is IDs."""
    return {"prompt": list(prompt_ids), "n_predict": COMPLETION_CAP,
            "temperature": 0, "stream": True, "cache_prompt": False,
            "return_tokens": True}


def _request_json(base_url: str, endpoint: str, body: object | None, timeout_s: float) -> tuple[int, Any, float]:
    if timeout_s <= 0:
        raise CaseTimeout("deadline elapsed before request")
    data = None if body is None else canonical_json(body)
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(base_url + endpoint, data=data, method="GET" if body is None else "POST",
                      headers=headers)
    started = time.monotonic()
    try:
        with urlopen(request, timeout=max(0.001, timeout_s)) as response:
            status = int(response.getcode())
            raw = response.read()
    except HTTPError as error:
        raise QualityError(f"HTTP {error.code} from {endpoint}") from error
    except (socket.timeout, TimeoutError) as error:
        raise CaseTimeout(f"{endpoint} exceeded {timeout_s:.3f}s") from error
    except URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise CaseTimeout(f"{endpoint} exceeded {timeout_s:.3f}s") from error
        raise QualityError(f"network error from {endpoint}: {error.reason}") from error
    except OSError as error:
        raise QualityError(f"network error from {endpoint}: {error}") from error
    elapsed = time.monotonic() - started
    if status < 200 or status >= 300:
        raise QualityError(f"HTTP {status} from {endpoint}")
    try:
        return status, json.loads(raw.decode("utf-8")), elapsed
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityError(f"invalid JSON from {endpoint}") from error


def _socket_timeout(response: Any, timeout_s: float) -> None:
    for candidate in (getattr(getattr(response, "fp", None), "raw", None),
                      getattr(getattr(response, "fp", None), "_sock", None)):
        sock = getattr(candidate, "_sock", candidate)
        if hasattr(sock, "settimeout"):
            try:
                sock.settimeout(max(0.001, timeout_s))
                return
            except OSError:
                pass


def _stream_completion(base_url: str, payload: Mapping[str, Any], deadline: float) -> dict[str, Any]:
    request = Request(base_url + "/completion", data=canonical_json(payload), method="POST",
                      headers={"Accept": "text/event-stream", "Content-Type": "application/json"})
    started = time.monotonic()
    text_parts: list[str] = []
    token_ids: list[int] = []
    final: dict[str, Any] = {}
    malformed_sse = 0
    nonempty_final_tokens = 0
    ttft_ms: float | None = None
    saw_stop = False
    stream_started = False

    def partial_stream() -> dict[str, Any]:
        return {
            "raw_text": "".join(text_parts),
            "returned_token_ids": list(token_ids),
            "final": dict(final),
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
            "wall_ms": round((time.monotonic() - started) * 1000.0, 3),
            "malformed_sse_frames": malformed_sse,
            "final_token_count": nonempty_final_tokens,
            "saw_stop": saw_stop,
            "stream_complete": False,
        }

    def attach_partial(error: Exception) -> None:
        if stream_started:
            setattr(error, "partial_stream", partial_stream())

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaseTimeout("deadline elapsed before /completion")
        with urlopen(request, timeout=max(0.001, remaining)) as response:
            status = int(response.getcode())
            if status < 200 or status >= 300:
                raise QualityError(f"HTTP {status} from /completion")
            stream_started = True
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CaseTimeout("/completion exceeded the per-case deadline")
                _socket_timeout(response, remaining)
                line = response.readline()
                if line == b"":
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded or decoded.startswith(":"):
                    continue
                if not decoded.startswith("data:"):
                    malformed_sse += 1
                    continue
                payload_text = decoded[5:].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_text)
                except json.JSONDecodeError:
                    malformed_sse += 1
                    continue
                if not isinstance(chunk, Mapping):
                    malformed_sse += 1
                    continue
                content = chunk.get("content", "")
                if not isinstance(content, str):
                    raise QualityError("SSE content is not a string")
                if content and ttft_ms is None:
                    ttft_ms = (time.monotonic() - started) * 1000.0
                text_parts.append(content)
                if "tokens" in chunk:
                    chunk_ids = _integer_ids(chunk["tokens"], "SSE tokens")
                    if chunk.get("stop") is True:
                        nonempty_final_tokens += len(chunk_ids)
                    else:
                        token_ids.extend(chunk_ids)
                for key in ("stop", "stop_type", "stopping_word", "truncated",
                            "tokens_evaluated", "tokens_predicted", "tokens_cached",
                            "timings", "has_new_line"):
                    if key in chunk:
                        final[key] = chunk[key]
                if chunk.get("stop") is True:
                    saw_stop = True
                    break
    except CaseTimeout as error:
        attach_partial(error)
        raise
    except HTTPError as error:
        raise QualityError(f"HTTP {error.code} from /completion") from error
    except (socket.timeout, TimeoutError) as error:
        wrapped = CaseTimeout("/completion socket deadline exceeded")
        attach_partial(wrapped)
        raise wrapped from error
    except URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            wrapped = CaseTimeout("/completion socket deadline exceeded")
            attach_partial(wrapped)
            raise wrapped from error
        wrapped = QualityError(f"network error from /completion: {error.reason}")
        attach_partial(wrapped)
        raise wrapped from error
    except OSError as error:
        wrapped = QualityError(f"network error from /completion: {error}")
        attach_partial(wrapped)
        raise wrapped from error
    return {
        "raw_text": "".join(text_parts),
        "returned_token_ids": token_ids,
        "final": final,
        "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
        "wall_ms": round((time.monotonic() - started) * 1000.0, 3),
        "malformed_sse_frames": malformed_sse,
        "final_token_count": nonempty_final_tokens,
        "saw_stop": saw_stop,
        "stream_complete": True,
    }


def _completion_validation(streamed: Mapping[str, Any], case: Mapping[str, Any], protocol: Any) -> dict[str, Any]:
    generated = _integer_ids(streamed.get("returned_token_ids"), "completion token IDs")
    final = streamed.get("final") if isinstance(streamed.get("final"), Mapping) else {}
    terminal = generated[-1] if generated else None
    stop_type = final.get("stop_type")
    cap_hit = terminal != EOS_ID and (
        stop_type in {"limit", "length"} or final.get("truncated") is True
    )
    if (terminal != EOS_ID and isinstance(final.get("tokens_predicted"), int)
            and final["tokens_predicted"] >= COMPLETION_CAP):
        cap_hit = True
    if len(generated) > COMPLETION_CAP:
        cap_status = "overflow"
    elif cap_hit and terminal != EOS_ID:
        cap_status = "hit_without_eos"
    else:
        cap_status = "within_cap"
    eos_status = "missing"
    if terminal == EOS_ID:
        eos_status = "canonical_eos"
    elif terminal in NATIVE_EOG_IDS:
        eos_status = "noncanonical_native_eog"
    elif terminal is not None:
        eos_status = "non_eos_terminal"
    canonical_generation_tokens_valid = _canonical_generation_tokens_valid(protocol, generated)
    wire_raw_text = streamed.get("raw_text")
    if not isinstance(wire_raw_text, str):
        raise QualityError("completion raw text is not a string")
    # b10453 emits the terminal EOS as a token-only partial frame and then an
    # empty final frame.  Decode only the non-EOS body IDs; do not strip or
    # otherwise normalize wire text.  This is the sole EOS accommodation.
    body_ids = generated[:-1] if terminal == EOS_ID else generated
    decoded_body_text: str | None = None
    decode_error: str | None = None
    try:
        tokenizer = case["tokenizer"]
        decoded_body_text = tokenizer.decode(body_ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
        if not isinstance(decoded_body_text, str):
            raise TypeError("tokenizer.decode did not return text")
    except Exception as error:
        decode_error = f"{type(error).__name__}: {error}"
    wire_hf_text_match = decode_error is None and decoded_body_text == wire_raw_text
    try:
        parsed = protocol.parse_output(wire_raw_text, case["context"])
    except Exception as error:
        parsed = None
        parse_error = f"{type(error).__name__}: {error}"
    else:
        parse_error = parsed.reason
    protocol_valid = bool(canonical_generation_tokens_valid and wire_hf_text_match
                          and len(generated) <= COMPLETION_CAP
                          and streamed.get("saw_stop")
                          and streamed.get("final_token_count") == 0
                          and not streamed.get("malformed_sse_frames")
                          and stop_type == "eos"
                          and parsed is not None and parsed.status == "accepted")
    predicted_noop = bool(protocol_valid and parsed.operation == "no_op") if parsed is not None else False
    if parsed is not None and predicted_noop:
        actual_region = list(case["context"].region_old)
    elif parsed is not None:
        actual_region = list(parsed.body)
    else:
        actual_region = []
    exact_region = bool(protocol_valid and actual_region == list(case["expected_region"]))
    if not canonical_generation_tokens_valid:
        protocol_failure = "invalid_generation_tokens"
    elif not wire_hf_text_match:
        protocol_failure = "wire_hf_text_mismatch"
    elif len(generated) > COMPLETION_CAP:
        protocol_failure = "completion_cap_overflow"
    elif not streamed.get("saw_stop"):
        protocol_failure = "missing_sse_stop"
    elif streamed.get("final_token_count") != 0:
        protocol_failure = "final_sse_token_duplication"
    elif streamed.get("malformed_sse_frames"):
        protocol_failure = "malformed_sse"
    elif stop_type != "eos":
        protocol_failure = "noncanonical_stop_type"
    elif parsed is None or parsed.status != "accepted":
        protocol_failure = "invalid_prm03_output"
    else:
        protocol_failure = None
    return {
        "status": "accepted" if protocol_valid else "protocol_error",
        "failure_class": None if protocol_valid else "mechanical",
        "failure": None if protocol_valid else {
            "type": "quality_contract",
            "message": protocol_failure,
        },
        "protocol": {
            "valid": protocol_valid,
            "canonical_generation_tokens_valid": canonical_generation_tokens_valid,
            "parser_status": None if parsed is None else parsed.status,
            "parser_operation": None if parsed is None else parsed.operation,
            "parser_reason": parse_error,
            "early_control_token": bool(generated and any(_is_control(token) for token in generated[:-1])),
            "stop_type": stop_type,
            "saw_stop": bool(streamed.get("saw_stop")),
            "malformed_sse_frames": int(streamed.get("malformed_sse_frames", 0)),
            "final_sse_tokens": int(streamed.get("final_token_count", 0)),
            "wire_hf_text_match": wire_hf_text_match,
            "wire_hf_text_status": "match" if wire_hf_text_match else (
                "decode_error" if decode_error is not None else "mismatch"),
            "wire_hf_decode_error": decode_error,
            "eos_text_normalization": "terminal EOS ID excluded before HF decode; wire text unchanged",
        },
        "eos": {
            "native_eog_ids": list(NATIVE_EOG_IDS),
            "terminal_token": terminal,
            "status": eos_status,
            "canonical": terminal == EOS_ID,
            "inclusive_cap": True,
        },
        "cap": {
            "limit": COMPLETION_CAP,
            "returned_tokens_including_terminal": len(generated),
            "status": cap_status,
            "hit": bool(cap_hit or len(generated) > COMPLETION_CAP
                        or (len(generated) == COMPLETION_CAP and terminal != EOS_ID)),
            "includes_terminal_eos": True,
        },
        "quality": {
            "protocol_valid": protocol_valid,
            "exact_edit": exact_region,
            "predicted_noop": predicted_noop,
            "strict_noop_correct": bool(case["expected_noop"] and predicted_noop),
            "noop_false_positive": bool(case["expected_noop"] and protocol_valid and not predicted_noop),
            "edit_exact": bool(not case["expected_noop"] and exact_region),
            "target_score_scope": "out-of-band-DAT07-label-only",
            "nll": {"status": "unavailable_native_no_logits"},
        },
        "decoded_body_text": decoded_body_text,
        "returned_token_ids": generated,
    }


def _server_preflight(base_url: str, timeout_s: float) -> dict[str, Any]:
    health_status, health, health_seconds = _request_json(base_url, "/health", None, timeout_s)
    if health_status != 200:
        raise QualityError(f"/health returned HTTP {health_status}")
    props_status, props, props_seconds = _request_json(base_url, "/props", None, timeout_s)
    if props_status != 200 or not isinstance(props, Mapping):
        raise QualityError(f"/props returned HTTP {props_status} or a non-object")
    generation = props.get("default_generation_settings")
    if not isinstance(generation, Mapping) or generation.get("n_ctx") != PROMPT_CAP:
        raise QualityError(f"server context must be exactly {PROMPT_CAP}: {generation}")
    return {"health": health, "props": props,
            "model_path_reported": props.get("model_path"),
            "n_ctx": generation.get("n_ctx"),
            "health_wall_seconds": health_seconds,
            "props_wall_seconds": props_seconds}


def _initial_receipt(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "status": "running",
        "started_at": now_iso(),
        "server_url": args.url,
        "model": {"label": args.model_label,
                   "url": getattr(args, "model_url", None),
                   "provenance": args.model_provenance,
                   "artifact_selection": "outer-root-wrapper-only"},
        "panel": {"path": str(Path(args.panel).resolve()), "expected_sha256": PANEL_SHA256,
                  "expected_rows": PANEL_ROWS},
        "contract": {
            "renderer_id": "zeta2-prm03-v1",
            "manual_bos_id": BOS_ID,
            "canonical_eos_id": EOS_ID,
            "native_eog_ids": list(NATIVE_EOG_IDS),
            "tokenization": {"add_special": False, "parse_special": False, "with_pieces": False,
                              "hf_split_special_tokens": True},
            "completion": {**make_completion_payload([]), "prompt": "manual BOS plus native /tokenize IDs",
                            "stop": "omitted"},
            "context_size": PROMPT_CAP,
            "cap_includes_terminal_eos": True,
            "quality_request_scope": "fresh independent request per case; no paired cache study",
        },
        "deadlines": {"case_seconds": args.case_deadline_seconds,
                      "global_seconds": args.deadline_seconds,
                      "reserve_seconds": args.reserve_seconds,
                      "no_five_second_quality_truncation": True},
        "persistence": {"atomic_replace": True, "file_fsync": True, "directory_fsync": True},
        "evaluation_complete": False,
        "all_protocol_accepted": False,
        "denominators": {"panel_rows": PANEL_ROWS, "edit_cases": EDIT_CASES,
                         "strict_noop_cases": STRICT_NOOP_CASES, "attempted_rows": 0,
                         "responses": 0, "partial_responses": 0, "transport_failures": 0,
                         "mechanical_failures": 0, "protocol_error_rows": 0,
                         "protocol_rows": 0, "exact_region_rows": 0,
                         "edit_exact_rows": 0, "strict_noop_correct_rows": 0,
                         "noop_false_positive_rows": 0, "cap_hit_rows": 0},
        "families": {},
        "cases": [],
        "loss_metrics": {"status": "unavailable_native_no_logits",
                         "prompt_nll": None, "target_nll": None},
        "static_preflight": {},
        "unattempted_ids": [],
    }


def _update_metrics(receipt: dict[str, Any], record: Mapping[str, Any]) -> None:
    den = receipt["denominators"]
    den["attempted_rows"] += 1
    family = str(record.get("family"))
    metrics = receipt["families"].setdefault(family, {
        "panel_rows": 0, "attempted_rows": 0, "responses": 0, "partial_responses": 0,
        "transport_failures": 0, "mechanical_failures": 0,
        "protocol_error_rows": 0, "protocol_valid": 0,
        "exact_region": 0, "edit_exact": 0, "strict_noop_correct": 0,
        "noop_false_positive": 0, "cap_hit": 0,
    })
    metrics["attempted_rows"] += 1
    quality = record.get("quality") if isinstance(record.get("quality"), Mapping) else {}
    cap = record.get("cap") if isinstance(record.get("cap"), Mapping) else {}
    if record.get("status") == "accepted" or (
            record.get("response_received") and record.get("response_complete")):
        den["responses"] += 1
        metrics["responses"] += 1
    elif record.get("response_received"):
        den["partial_responses"] += 1
        metrics["partial_responses"] += 1
    if record.get("status") == "protocol_error":
        den["protocol_error_rows"] += 1
        metrics["protocol_error_rows"] += 1
        den["mechanical_failures"] += 1
        metrics["mechanical_failures"] += 1
    elif record.get("status") == "failed":
        if record.get("failure_class") == "transport":
            den["transport_failures"] += 1
            metrics["transport_failures"] += 1
        else:
            den["mechanical_failures"] += 1
            metrics["mechanical_failures"] += 1
    for key, den_key, metric_key in (("protocol_valid", "protocol_rows", "protocol_valid"),
                                     ("exact_edit", "exact_region_rows", "exact_region"),
                                     ("edit_exact", "edit_exact_rows", "edit_exact"),
                                     ("strict_noop_correct", "strict_noop_correct_rows", "strict_noop_correct"),
                                     ("noop_false_positive", "noop_false_positive_rows", "noop_false_positive")):
        if quality.get(key):
            den[den_key] += 1
            metrics[metric_key] += 1
    if cap.get("hit") or cap.get("status") in {"hit_without_eos", "overflow"}:
        den["cap_hit_rows"] += 1
        metrics["cap_hit"] += 1


def _failure_class(error: Exception) -> str:
    """Separate connection/deadline failures from mechanical quality failures."""
    if isinstance(error, (CaseTimeout, HTTPError, URLError, socket.timeout, TimeoutError, OSError)):
        return "transport"
    message = str(error).lower()
    if message.startswith(("http ", "network error", "deadline elapsed", "/completion socket")):
        return "transport"
    return "mechanical"


def _refresh_evaluation_status(receipt: dict[str, Any]) -> None:
    """Expose panel completion independently from quality acceptance."""
    cases = receipt.get("cases")
    attempted = len(cases) if isinstance(cases, list) else 0
    complete = attempted == PANEL_ROWS and not receipt.get("unattempted_ids")
    receipt["evaluation_complete"] = complete
    receipt["all_protocol_accepted"] = bool(
        complete and all(item.get("status") == "accepted" for item in cases)
    )


def _case_failure(case: Mapping[str, Any], error: Exception, *, elapsed_ms: float | None = None) -> dict[str, Any]:
    failure_class = _failure_class(error)
    return {
        "id": case["id"], "family": case["family"], "package_id": case["package_id"],
        "operation_label": case["operation"], "expected_noop": case["expected_noop"],
        "prompt_sha256": case["prompt_sha256"], "target_sha256": case["target_sha256"],
        "hf_prompt_ids": case.get("hf_prompt_ids", []),
        "hf_prompt_tokens_with_bos": case.get("hf_prompt_tokens_with_bos"),
        "native_prompt_ids_without_bos": [],
        "returned_token_ids": [],
        "raw_text": None,
        "server_timing": {},
        "status": "failed", "failure_class": failure_class,
        "response_received": False, "response_complete": False,
        "error": {"type": type(error).__name__, "message": str(error)},
        "protocol": {"valid": False, "status": "not_observed",
                      "canonical_generation_tokens_valid": False},
        "eos": {"status": "not_observed", "canonical": False},
        "cap": {"limit": COMPLETION_CAP, "status": "not_observed",
                "hit": False, "includes_terminal_eos": True},
        "timing": {"wall_ms": elapsed_ms},
        "quality": {"protocol_valid": False, "exact_edit": False,
                    "strict_noop_correct": False, "noop_false_positive": False,
                    "edit_exact": False, "nll": {"status": "unavailable_native_no_logits"}},
    }


def _run_case(base_url: str, case: dict[str, Any], protocol: Any,
              tokenizer: Any, case_deadline_seconds: int) -> dict[str, Any]:
    case["tokenizer"] = tokenizer
    started = time.monotonic()
    deadline = started + case_deadline_seconds
    record: dict[str, Any] = {
        "id": case["id"], "family": case["family"], "package_id": case["package_id"],
        "operation_label": case["operation"], "expected_noop": case["expected_noop"],
        "prompt_sha256": case["prompt_sha256"], "target_sha256": case["target_sha256"],
        "hf_prompt_ids": list(case["hf_prompt_ids"]),
        "hf_prompt_tokens_with_bos": case["hf_prompt_tokens_with_bos"],
        "hf_prompt_ids_sha256": case["hf_prompt_ids_sha256"],
        "request": {}, "response_received": False, "response_complete": False,
    }
    try:
        remaining = deadline - time.monotonic()
        tokenize_payload = make_tokenize_payload(case["prompt"])
        record["request"]["tokenize"] = tokenize_payload
        status, tokenized, tokenize_seconds = _request_json(base_url, "/tokenize", tokenize_payload, remaining)
        record["tokenize_wall_ms"] = round(tokenize_seconds * 1000.0, 3)
        if status != 200 or not isinstance(tokenized, Mapping):
            raise QualityError("/tokenize returned an invalid response")
        native_without_bos = _integer_ids(tokenized.get("tokens"), "native tokenize.tokens")
        record["native_prompt_ids_without_bos"] = native_without_bos
        record["native_prompt_ids_sha256"] = canonical_sha256(native_without_bos)
        record["prompt_token_identity"] = {
            "expected_hf_without_bos_sha256": canonical_sha256(case["hf_prompt_ids"][1:]),
            "native_matches_hf": native_without_bos == case["hf_prompt_ids"][1:],
        }
        if native_without_bos != case["hf_prompt_ids"][1:]:
            raise QualityError("native /tokenize IDs differ from frozen HF prompt IDs")
        prompt_ids = [BOS_ID, *native_without_bos]
        if len(prompt_ids) + COMPLETION_CAP > PROMPT_CAP:
            raise QualityError("manual-BOS prompt plus cap exceeds server context")
        completion_payload = make_completion_payload(prompt_ids)
        record["request"]["completion"] = completion_payload
        streamed = _stream_completion(base_url, completion_payload, deadline)
        record["response_received"] = True
        record["response_complete"] = bool(streamed.get("stream_complete"))
        record["raw_text"] = streamed["raw_text"]
        record["raw_text_sha256"] = sha256_bytes(streamed["raw_text"].encode("utf-8"))
        record["returned_token_ids"] = streamed["returned_token_ids"]
        record["returned_token_ids_sha256"] = canonical_sha256(streamed["returned_token_ids"])
        record["server_timing"] = streamed["final"]
        evaluated = streamed["final"].get("tokens_evaluated")
        record["prompt_evaluation"] = {
            "expected_full_integer_prompt_tokens": len(prompt_ids),
            "tokens_evaluated": evaluated,
            "matches_manual_bos_prompt": evaluated is None or evaluated == len(prompt_ids),
        }
        if evaluated is not None and evaluated != len(prompt_ids):
            raise QualityError("completion.tokens_evaluated differs from the manual-BOS prompt length")
        record["timing"] = {"tokenize_ms": record.get("tokenize_wall_ms"),
                             "ttft_ms": streamed["ttft_ms"], "completion_wall_ms": streamed["wall_ms"],
                             "wall_ms": round((time.monotonic() - started) * 1000.0, 3),
                             "deadline_seconds": case_deadline_seconds,
                             "synchronized": False, "source": "monotonic wall clock"}
        validation = _completion_validation(streamed, case, protocol)
        record.update({key: value for key, value in validation.items() if key not in {"returned_token_ids", "decoded_body_text"}})
        record["decoded_body_text"] = validation["decoded_body_text"]
        record["returned_token_ids"] = validation["returned_token_ids"]
        return record
    except Exception as error:
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        failure = _case_failure(case, error, elapsed_ms=elapsed_ms)
        failure.update({key: value for key, value in record.items()
                        if key not in {"status", "error", "quality", "timing"}})
        failure["error"] = {"type": type(error).__name__, "message": str(error)}
        failure["failure_class"] = _failure_class(error)
        partial = getattr(error, "partial_stream", None)
        if isinstance(partial, Mapping):
            failure["response_received"] = True
            failure["response_complete"] = False
            failure["partial_stream"] = partial
            failure["raw_text"] = partial["raw_text"]
            failure["raw_text_sha256"] = sha256_bytes(partial["raw_text"].encode("utf-8"))
            failure["returned_token_ids"] = list(partial["returned_token_ids"])
            failure["returned_token_ids_sha256"] = canonical_sha256(partial["returned_token_ids"])
            failure["server_timing"] = partial["final"]
        else:
            failure["response_received"] = bool(record.get("response_received"))
            failure["response_complete"] = False
        failure["timing"] = {"wall_ms": elapsed_ms, "deadline_seconds": case_deadline_seconds,
                              "synchronized": False}
        return failure


def run_quality(args: argparse.Namespace) -> dict[str, Any]:
    output = _require_fresh_ext4(Path(args.output))
    receipt = _initial_receipt(output, args)
    receipt["output_preflight"] = {
        "path": str(output),
        "filesystem": "ext4",
        "probe": ["/usr/bin/findmnt", "-T", str(output.parent), "-n", "-o", "FSTYPE"],
        "probe_result": "ext4",
    }
    _write_atomic(output, receipt)
    global_deadline = time.monotonic() + args.deadline_seconds
    try:
        base_url = _loopback_origin(args.url)
        protocol, protocol_path, eval_path, client_path = _protocol_modules(Path(args.execution_root))
        rows, panel_audit = _load_panel(Path(args.panel), protocol)
        tokenizer, tokenizer_audit = _load_tokenizer(Path(args.tokenizer_dir))
        cases = _prepare_cases(rows, protocol)
        prompt_audit = _encode_cases(cases, tokenizer, protocol)
        receipt["server_url"] = base_url
        receipt["panel"] = panel_audit
        receipt["tokenizer"] = tokenizer_audit
        receipt["static_preflight"] = {
            "protocol": {"path": str(protocol_path), "sha256": PROTOCOL_SHA256},
            "campaign_eval": {"path": str(eval_path), "sha256": CAMPAIGN_EVAL_SHA256},
            "campaign_client": {"path": str(client_path), "sha256": CAMPAIGN_CLIENT_SHA256},
            "prompt_geometry": prompt_audit,
            "no_legacy_b4_renderer": True,
        }
        receipt["families"] = {
            family: {"panel_rows": count, "attempted_rows": 0, "responses": 0,
                     "partial_responses": 0,
                     "transport_failures": 0, "mechanical_failures": 0,
                     "protocol_error_rows": 0, "protocol_valid": 0,
                     "exact_region": 0, "edit_exact": 0, "strict_noop_correct": 0,
                     "noop_false_positive": 0, "cap_hit": 0}
            for family, count in sorted(Counter(case["family"] for case in cases).items())
        }
        _write_atomic(output, receipt)
        probe_remaining = global_deadline - time.monotonic() - args.reserve_seconds
        if probe_remaining <= 0:
            raise CaseTimeout("global deadline reserve elapsed before server preflight")
        probe_seconds = min(float(args.case_deadline_seconds), probe_remaining)
        receipt["server"] = _server_preflight(base_url, probe_seconds)
        _write_atomic(output, receipt)
        for index, case in enumerate(cases):
            remaining = global_deadline - time.monotonic()
            if remaining <= args.reserve_seconds + 1.0:
                receipt["status"] = "deadline"
                receipt["unattempted_ids"] = [item["id"] for item in cases[index:]]
                break
            case_budget = max(1, int(min(args.case_deadline_seconds, remaining - args.reserve_seconds)))
            record = _run_case(base_url, case, protocol, tokenizer, int(case_budget))
            receipt["cases"].append(record)
            _update_metrics(receipt, record)
            _refresh_evaluation_status(receipt)
            _write_atomic(output, receipt)
        else:
            # A full 75-case run is terminal even when quality contract rows
            # fail.  Those rows remain visible through protocol_error_rows and
            # all_protocol_accepted, rather than being mislabeled incomplete.
            _refresh_evaluation_status(receipt)
            receipt["status"] = "complete"
        if receipt["status"] == "running":
            receipt["status"] = "deadline"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    receipt["completed_at"] = now_iso()
    _write_atomic(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--url", required=True)
    result.add_argument("--panel", type=Path, default=PANEL_PATH)
    result.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_PATH)
    result.add_argument("--execution-root", type=Path, default=EXECUTION_ROOT)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model-label", required=True)
    result.add_argument("--model-url")
    result.add_argument("--model-provenance", required=True)
    result.add_argument("--case-deadline-seconds", type=int, default=DEFAULT_CASE_DEADLINE_SECONDS)
    result.add_argument("--deadline-seconds", type=int, default=DEFAULT_GLOBAL_DEADLINE_SECONDS)
    result.add_argument("--reserve-seconds", type=int, default=DEFAULT_RESERVE_SECONDS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.case_deadline_seconds <= MAX_CASE_DEADLINE_SECONDS:
        print(json.dumps({"status": "failed", "error": "case deadline must be 1..120"}))
        return 1
    if not 1 <= args.deadline_seconds <= MAX_GLOBAL_DEADLINE_SECONDS:
        print(json.dumps({"status": "failed", "error": "global deadline must be 1..7200"}))
        return 1
    if args.reserve_seconds < 0 or args.reserve_seconds >= args.deadline_seconds:
        print(json.dumps({"status": "failed", "error": "reserve must be nonnegative and below global deadline"}))
        return 1
    if args.model_url is not None and not args.model_url.strip():
        print(json.dumps({"status": "failed", "error": "model-url must be nonempty when supplied"}))
        return 1
    if not args.model_label.strip() or not args.model_provenance.strip():
        print(json.dumps({"status": "failed", "error": "model-label and model-provenance must be nonempty"}))
        return 1
    try:
        result = run_quality(args)
    except Exception as error:
        result = {"schema_version": SCHEMA_VERSION, "task": TASK,
                  "status": "failed", "failure": {"type": type(error).__name__, "message": str(error)}}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
