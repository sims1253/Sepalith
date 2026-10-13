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

from teacher_rows import build_teacher_record, prompt_prefix, stage_status


TargetGenerator = Callable[[list[int], int], Mapping[str, Any] | Sequence[int]]


def stable_select_rows(rows: Iterable[Mapping[str, Any]], limit: int = 256) -> list[dict[str, Any]]:
    """Choose a bounded deterministic subset by row ID hash."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = [dict(row) for row in rows]
    selected.sort(key=lambda row: hashlib.sha256(str(row.get("id", "")).encode()).hexdigest())
    return selected[:limit]


def _generation_parts(result: Mapping[str, Any] | Sequence[int]) -> tuple[list[int], str, str | None]:
    if isinstance(result, Mapping):
        if "input_ids" in result:
            # An adapter may return the full target sequence.  Only the suffix
            # after its supplied prompt length is accepted.
            generated = result.get("generated_ids")
            if generated is None:
                raise ValueError("generator mapping must provide generated_ids, never authored input_ids")
        else:
            generated = result.get("generated_ids")
        if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes, bytearray)):
            raise ValueError("generator generated_ids must be a token sequence")
        return (
            [int(token_id) for token_id in generated],
            str(result.get("protocol_status", "unchecked")),
            result.get("protocol_error"),
        )
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return [int(token_id) for token_id in result], "unchecked", None
    raise ValueError("generator must return token IDs or a mapping containing generated_ids")


def regenerate_train_rows(
    rows: Iterable[Mapping[str, Any]],
    generate: TargetGenerator,
    *,
    limit: int = 256,
    generation_cap: int = 192,
    vocab_size: int = 130560,
    target_identity: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Regenerate a bounded TRAIN subset and continue after row-level issues."""
    selected = stable_select_rows(rows, limit=limit)
    records: list[dict[str, Any]] = []
    for row in selected:
        prefix = prompt_prefix(row)
        try:
            result = generate(prefix, generation_cap)
            generated_ids, protocol_status, protocol_error = _generation_parts(result)
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
            record = build_teacher_record(
                row,
                [],
                generation_cap=generation_cap,
                protocol_status="unchecked",
                protocol_error=f"generation_error:{type(exc).__name__}:{exc}",
                vocab_size=vocab_size,
                target_identity=target_identity,
            )
            record["generation_status"] = "generation_error"
            record["cache_rejection_reason"] = "generation_error"
            record["cacheable"] = False
        records.append(record)
    summary = stage_status(records)
    summary.update(
        {
            "selected_rows": len(selected),
            "generation_cap": generation_cap,
            "train_only": True,
            "abort_on_cap_hit": False,
            "abort_on_protocol_invalid": False,
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
