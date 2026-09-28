"""Pretokenized target-generation bridge for R2 TRAIN rows.

The admitted JSONL rows contain authored answer tails for screening.  A real
TRAIN cache must replace those tails with the exact target model generation.
This module therefore takes only the stored prompt prefix plus a generated
sequence and never copies, repairs, or appends the authored tail.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

from minicpm5_dspark_config import DEFAULT_EOG_TOKEN_IDS


CACHE_WRITER_REQUIRES_EOS = False


def prompt_prefix(row: Mapping[str, Any]) -> list[int]:
    """Extract and validate the prompt-only prefix from an admitted row."""
    input_ids = row.get("input_ids")
    target_start = row.get("target_start")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("row.input_ids must be a non-empty list")
    if not isinstance(target_start, int) or not 0 < target_start <= len(input_ids):
        raise ValueError("row.target_start must point inside input_ids")
    prefix = input_ids[:target_start]
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in prefix):
        raise ValueError("prompt prefix contains a non-integer or negative token")
    return list(prefix)


def _first_stop(generated_ids: list[int], stop_ids: tuple[int, ...]) -> tuple[int | None, int | None]:
    for position, token_id in enumerate(generated_ids):
        if token_id in stop_ids:
            return token_id, position
    return None, None


def classify_generated_tail(
    generated_ids: Iterable[int],
    *,
    generation_cap: int = 192,
    stop_ids: Iterable[int] = DEFAULT_EOG_TOKEN_IDS,
    protocol_status: str = "unchecked",
    protocol_error: str | None = None,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    """Record target-generation outcomes without changing generated tokens.

    A cap hit and an invalid edit protocol are row-level outcomes.  They are
    retained for audit and do not cause the caller to abort the 256-row stage.
    A cache row only becomes uncacheable for structural reasons such as an
    empty generation or an out-of-range token ID.
    """
    if generation_cap <= 0:
        raise ValueError("generation_cap must be positive")
    generated = [int(token_id) for token_id in generated_ids]
    stops = tuple(dict.fromkeys(int(token_id) for token_id in stop_ids))
    eos_token_id, eos_position = _first_stop(generated, stops)
    cap_hit = len(generated) >= generation_cap and eos_token_id is None
    if protocol_status not in {"unchecked", "valid", "invalid"}:
        raise ValueError("protocol_status must be unchecked, valid, or invalid")
    out_of_range = (
        vocab_size is not None
        and any(token_id < 0 or token_id >= int(vocab_size) for token_id in generated)
    )
    status = "eog_terminated" if eos_token_id is not None else "cap_hit" if cap_hit else "length_terminated"
    cacheable = bool(generated) and not out_of_range
    return {
        # This list is the source of truth.  No EOS is synthesized on any path.
        "target_generated_ids": generated,
        "generated_token_count": len(generated),
        "native_stop_ids": list(stops),
        "native_stop_token_id": eos_token_id,
        "native_stop_position": eos_position,
        "generation_status": status,
        "cap_hit": cap_hit,
        "protocol_status": protocol_status,
        "protocol_invalid": protocol_status == "invalid",
        "protocol_error": protocol_error,
        "token_ids_in_vocab": not out_of_range,
        "cacheable": cacheable,
        "cache_rejection_reason": (
            "empty_generation" if not generated else "token_id_out_of_range" if out_of_range else None
        ),
        "cache_writer_requires_eos": CACHE_WRITER_REQUIRES_EOS,
        "eos_was_synthesized": False,
    }


def build_teacher_record(
    row: Mapping[str, Any],
    generated_ids: Iterable[int],
    *,
    generation_cap: int = 192,
    stop_ids: Iterable[int] = DEFAULT_EOG_TOKEN_IDS,
    protocol_status: str = "unchecked",
    protocol_error: str | None = None,
    vocab_size: int | None = None,
    target_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one cache-ready record from a target-generated response."""
    prefix = prompt_prefix(row)
    outcome = classify_generated_tail(
        generated_ids,
        generation_cap=generation_cap,
        stop_ids=stop_ids,
        protocol_status=protocol_status,
        protocol_error=protocol_error,
        vocab_size=vocab_size,
    )
    generated = outcome["target_generated_ids"]
    record = {
        "id": row.get("id"),
        "source_row_id": row.get("id"),
        "prompt_input_ids": prefix,
        # This is exactly prefix + model output.  In particular, cap-hit and
        # invalid-protocol rows remain exact and are never tail-substituted.
        "input_ids": prefix + generated,
        "target_start": len(prefix),
        "loss_mask": [0] * len(prefix) + [1] * len(generated),
        "loss_mask_policy": "ones_on_exact_target_generated_tokens_only",
        "target_source": "target_model_generation",
        "target_identity": dict(target_identity or {}),
    }
    record.update(outcome)
    record["prompt_prefix_sha256"] = sha256(bytes().join(int(x).to_bytes(4, "little", signed=False) for x in prefix)).hexdigest()
    record["generated_ids_sha256"] = sha256(bytes().join(int(x).to_bytes(4, "little", signed=False) for x in generated)).hexdigest()
    return record


def split_cacheable(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep valid-token rows and report rejected rows without aborting a stage."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if item.get("cacheable"):
            accepted.append(item)
        else:
            rejected.append(item)
    return accepted, rejected


def stage_status(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count outcomes for a receipt while preserving cap/protocol failures."""
    counts = {
        "rows": 0,
        "cacheable": 0,
        "empty_generation": 0,
        "cap_hit": 0,
        "protocol_invalid": 0,
        "eog_terminated": 0,
    }
    for record in records:
        counts["rows"] += 1
        counts["cacheable"] += int(bool(record.get("cacheable")))
        counts["empty_generation"] += int(record.get("cache_rejection_reason") == "empty_generation")
        counts["cap_hit"] += int(bool(record.get("cap_hit")))
        counts["protocol_invalid"] += int(bool(record.get("protocol_invalid")))
        counts["eog_terminated"] += int(record.get("generation_status") == "eog_terminated")
    return counts
