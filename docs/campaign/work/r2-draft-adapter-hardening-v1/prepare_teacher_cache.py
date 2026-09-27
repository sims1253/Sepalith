"""Bounded TRAIN-only target regeneration and pretokenized cache staging.

A cloud adapter supplies ``generate(prompt_ids, max_new_tokens)`` using the
actual bound MiniCPM5 target.  This file deliberately contains no model
loading and cannot silently fall back to the authored answer tail.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from teacher_rows import (
    build_teacher_record,
    prompt_prefix,
    stage_status,
)


TargetGenerator = Callable[[list[int], int], Mapping[str, Any]]
TRAIN_SOURCE_PATH = "docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl"
TRAIN_SOURCE_SHA256 = "85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e"


def stable_select_rows(rows: Iterable[Mapping[str, Any]], limit: int = 256) -> list[dict[str, Any]]:
    """Choose a bounded deterministic subset by row ID hash."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = [dict(row) for row in rows]
    row_ids = [row.get("id") for row in selected]
    if any(row_id in (None, "") for row_id in row_ids):
        raise ValueError("every TRAIN row must have a non-empty id")
    normalized_ids = [str(row_id) for row_id in row_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("TRAIN row IDs must be unique before selection")
    selected.sort(key=lambda row: hashlib.sha256(str(row.get("id", "")).encode()).hexdigest())
    return selected[:limit]


def _generation_parts(
    result: Mapping[str, Any],
    prefix: Sequence[int],
) -> tuple[list[int], str, str | None]:
    """Require an explicit generated suffix and validate optional full output."""
    if not isinstance(result, Mapping):
        raise ValueError("generator must return a mapping containing generated_ids")
    generated = result.get("generated_ids")
    if not isinstance(generated, Sequence) or isinstance(
        generated, (str, bytes, bytearray)
    ):
        raise ValueError("generator mapping must provide generated_ids as a token sequence")
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in generated):
        raise ValueError("generator generated_ids must contain integer token IDs")
    generated_ids = [int(token_id) for token_id in generated]
    if "input_ids" in result:
        full = result["input_ids"]
        if not isinstance(full, Sequence) or isinstance(full, (str, bytes, bytearray)):
            raise ValueError("generator input_ids must be a token sequence when provided")
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in full):
            raise ValueError("generator input_ids must contain integer token IDs")
        full_ids = [int(token_id) for token_id in full]
        expected = list(prefix) + generated_ids
        if full_ids != expected:
            raise ValueError(
                "generator input_ids must equal the supplied prompt prefix followed by generated_ids"
            )
    return (
        generated_ids,
        str(result.get("protocol_status", "unchecked")),
        result.get("protocol_error"),
    )


def _generation_error_record(
    row: Mapping[str, Any],
    error: Exception,
    *,
    target_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep malformed-row diagnostics without inventing a cache payload."""
    return {
        "id": row.get("id"),
        "source_row_id": row.get("id"),
        "target_source": "target_model_generation",
        "target_identity": dict(target_identity or {}),
        "target_generated_ids": [],
        "generated_token_count": 0,
        "raw_generated_token_count": 0,
        "generation_status": "generation_error",
        "cacheable": False,
        "cache_rejection_reason": "generation_error",
        "protocol_status": "unchecked",
        "protocol_invalid": False,
        "protocol_error": f"{type(error).__name__}: {error}",
        "eos_was_synthesized": False,
    }


def _validate_train_source(source_path: str | Path | None, source_sha256: str | None) -> dict[str, str]:
    if source_path is None or source_sha256 is None:
        raise ValueError("TRAIN source_path and source_sha256 are required")
    normalized_path = Path(source_path).as_posix()
    if normalized_path != TRAIN_SOURCE_PATH:
        raise ValueError(f"unexpected TRAIN source path: {normalized_path}")
    if str(source_sha256) != TRAIN_SOURCE_SHA256:
        raise ValueError("TRAIN source SHA-256 does not match the admitted file")
    return {"path": normalized_path, "sha256": str(source_sha256)}


def _validate_row_split(row: Mapping[str, Any]) -> None:
    split = row.get("split")
    if split is not None and str(split).lower() not in {"train", "cpt_train"}:
        raise ValueError(f"row {row.get('id')!r} is not in the TRAIN split")


def regenerate_train_rows(
    rows: Iterable[Mapping[str, Any]],
    generate: TargetGenerator,
    *,
    limit: int = 256,
    generation_cap: int = 192,
    vocab_size: int = 130560,
    target_identity: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
    source_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Regenerate a bounded TRAIN subset and continue after row-level issues."""
    source_binding = _validate_train_source(source_path, source_sha256)
    selected = stable_select_rows(rows, limit=limit)
    records: list[dict[str, Any]] = []
    for row in selected:
        try:
            _validate_row_split(row)
            prefix = prompt_prefix(row, vocab_size=vocab_size)
        except Exception as exc:
            records.append(
                _generation_error_record(
                    row,
                    exc,
                    target_identity=target_identity,
                )
            )
            continue
        try:
            result = generate(prefix, generation_cap)
        except Exception as exc:
            # A target-service/model failure is systemic.  Do not turn OOM,
            # device, or adapter failures into an apparently successful stage.
            raise RuntimeError(
                f"target generator failed for TRAIN row {row.get('id')!r}"
            ) from exc
        try:
            generated_ids, protocol_status, protocol_error = _generation_parts(
                result, prefix
            )
            record = build_teacher_record(
                row,
                generated_ids,
                generation_cap=generation_cap,
                protocol_status=protocol_status,
                protocol_error=protocol_error,
                vocab_size=vocab_size,
                target_identity=target_identity,
            )
        except Exception as exc:  # keep the bounded stage auditable and live
            record = _generation_error_record(
                row,
                target_identity=target_identity,
                error=exc,
            )
        records.append(record)
    summary = stage_status(records)
    summary.update(
        {
            "selected_rows": len(selected),
            "generation_cap": generation_cap,
            "train_only": True,
            "train_source": source_binding,
            "abort_on_cap_hit": False,
            "abort_on_protocol_invalid": False,
            "systemic_generator_errors": "raise",
            "authored_tail_substitution": "forbidden",
        }
    )
    return records, summary


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read only the caller-provided JSONL path; no discovery or globbing."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write generated records to an explicit caller-owned output path."""
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")
