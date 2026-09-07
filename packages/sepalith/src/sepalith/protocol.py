"""Versioned edit inputs and a byte-compatible legacy prompt renderer."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Protocol, Sequence

SCHEMA_VERSION = "sepalith.edit-context.v1"
RENDERER_VERSION = "zeta2-v1"


# JSON metadata is intentionally open; this recursive validator rejects non-JSON values.
def _json_copy(value: object) -> Any:  # noqa: ANN401
    """Check JSON types without silently converting keys, tuples or NaN."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("Metadata must contain JSON values with finite numbers and string keys")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _extras(data: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return _json_copy({key: value for key, value in data.items() if key not in known})


def _record(extra: Mapping[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(extra, Mapping):
        raise ValueError("Extra fields must be a mapping")
    if set(extra) & set(fields):
        raise ValueError("Extra fields must not replace schema fields")
    if type(extra) is dict and not extra:
        return {**fields}
    return {**_json_copy(dict(extra)), **fields}


def _metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    return _json_copy(dict(value))


@dataclass(frozen=True)
class SourceRange:
    """Half-open offsets in Unicode code points within the identified content."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end:
            raise ValueError("Source range requires 0 <= start <= end integer offsets")


@dataclass(frozen=True)
class EvidenceRecord:
    """Observed evidence; absent identity/confidence stays absent."""

    source_kind: str
    content: str
    workspace_revision: str | None = None
    content_identity: str | None = None
    path: str | None = None
    source_range: SourceRange | None = None
    symbol: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.source_kind, "source_kind")
        if not self.source_kind:
            raise ValueError("source_kind must not be empty")
        _text(self.content, "content")
        for name in ("workspace_revision", "content_identity", "path", "symbol"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if self.source_range is not None and not isinstance(self.source_range, SourceRange):
            raise ValueError("source_range must be a SourceRange")
        if self.confidence is not None and (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be a finite number in [0, 1]")
        self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return _record(self.extra, {
            "source_kind": self.source_kind, "content": self.content,
            "workspace_revision": self.workspace_revision,
            "content_identity": self.content_identity, "path": self.path,
            "source_range": None if self.source_range is None else {
                "start": self.source_range.start, "end": self.source_range.end},
            "symbol": self.symbol, "confidence": self.confidence,
            "metadata": _metadata(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRecord:
        if not isinstance(data, Mapping):
            raise ValueError("Evidence must be a mapping")
        known = set(cls.__dataclass_fields__) - {"extra"}
        if not {"source_kind", "content"} <= data.keys():
            raise ValueError("Evidence requires source_kind and content")
        values = {key: value for key, value in data.items() if key in known}
        if values.get("source_range") is not None:
            value = values["source_range"]
            if not isinstance(value, dict) or set(value) != {"start", "end"}:
                raise ValueError("source_range requires start and end only")
            values["source_range"] = SourceRange(**value)
        return cls(**values, extra=_extras(data, known))


@dataclass(frozen=True)
class EditContext:
    """Line arrays preserve whitespace; cursor_idx=-1 means no cursor."""

    path: str
    prefix: tuple[str, ...]
    region_old: tuple[str, ...]
    suffix: tuple[str, ...]
    cursor_idx: int
    event_diff: str = ""
    cursor_column: int | None = None
    evidence: tuple[EvidenceRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version!r}")
        _text(self.path, "path")
        _text(self.event_diff, "event_diff")
        for name in ("prefix", "region_old", "suffix"):
            lines = getattr(self, name)
            if not isinstance(lines, (list, tuple)) or any(not isinstance(line, str) for line in lines):
                raise ValueError(f"{name} must be an array of strings")
            object.__setattr__(self, name, tuple(lines))
        if type(self.cursor_idx) is not int:
            raise ValueError("cursor_idx must be an integer")
        # Legacy rendering omits the cursor for any out-of-range index.
        if self.cursor_column is not None and (
            type(self.cursor_column) is not int
            or not 0 <= self.cursor_idx < len(self.region_old)
            or not 0 <= self.cursor_column <= len(self.region_old[self.cursor_idx])
        ):
            raise ValueError("cursor_column must address a code-point boundary on the cursor line")
        if not isinstance(self.evidence, (list, tuple)) or any(not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise ValueError("evidence must contain EvidenceRecord objects")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return _record(self.extra, {
            "schema_version": self.schema_version, "path": self.path,
            "prefix": list(self.prefix), "region_old": list(self.region_old),
            "suffix": list(self.suffix), "cursor_idx": self.cursor_idx,
            "cursor_column": self.cursor_column, "event_diff": self.event_diff,
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": _metadata(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EditContext:
        if not isinstance(data, Mapping):
            raise ValueError("Context must be a mapping")
        known = set(cls.__dataclass_fields__) - {"extra"}
        required = {"schema_version", "path", "prefix", "region_old", "suffix", "cursor_idx"}
        if not required <= data.keys():
            raise ValueError(f"Missing context fields: {sorted(required - data.keys())}")
        values = {key: value for key, value in data.items() if key in known}
        evidence = values.get("evidence", [])
        if not isinstance(evidence, (list, tuple)) or any(not isinstance(item, Mapping) for item in evidence):
            raise ValueError("evidence must be an array of records")
        values["evidence"] = tuple(EvidenceRecord.from_dict(item) for item in evidence)
        return cls(**values, extra=_extras(data, known))

    @classmethod
    def from_legacy(cls, example: Mapping[str, Any]) -> EditContext:
        """Explicit migration of unversioned eval records, including their extras."""
        data = dict(example)
        if "schema_version" in data:
            raise ValueError("Use from_dict for versioned records")
        data["schema_version"] = SCHEMA_VERSION
        data["event_diff"] = data.get("event_diff") or ""
        return cls.from_dict(data)


class Tokenizer(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


def render_context(
    context: EditContext, renderer: str = RENDERER_VERSION, *,
    tokenizer: Tokenizer | None = None, max_tokens: int | None = None,
) -> str:
    """Render only legacy fields; an enforced budget counts supplied tokenizer IDs."""
    if renderer != RENDERER_VERSION:
        raise ValueError(f"Unsupported renderer: {renderer!r}")
    event = context.event_diff.replace("```diff\n", "").replace("```", "")
    event_lines = event.splitlines()
    if event_lines and event_lines[0].startswith("User edited"):
        event_lines = event_lines[1:]
    while event_lines and not event_lines[0].strip():
        event_lines.pop(0)
    parts = ["<[fim-suffix]>", *context.suffix, "<[fim-prefix]><filename>edit_history"]
    if event_lines:
        parts.extend([*event_lines, ""])
    parts.extend([f"<filename>{context.path}", *context.prefix, "<<<<<<< CURRENT"])
    region = list(context.region_old)
    if 0 <= context.cursor_idx < len(region):
        line = region[context.cursor_idx]
        column = len(line) if context.cursor_column is None else context.cursor_column
        region[context.cursor_idx] = line[:column] + "<|user_cursor|>" + line[column:]
    parts.extend([*region, "=======", "<[fim-middle]>"])
    prompt = "\n".join(parts)
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
        if tokenizer is None:
            raise ValueError("An enforced token budget requires a supplied tokenizer")
        count = len(tokenizer.encode(prompt))
        if count > max_tokens:
            raise ValueError(f"Prompt uses {count} tokens; budget is {max_tokens}")
    return prompt


def serialize_snapshot(context: EditContext) -> str:
    """Stable JSON for a captured context; evidence order remains significant."""
    return json.dumps(context.to_dict(), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
