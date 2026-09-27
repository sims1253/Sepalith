"""DAT-04B lossless adapter for structured completion/case rows.

This module is deliberately separate from the worker-data scenario adapter.
It accepts a raw JSONL row together with an audited source reference and
reconstructs a prediction context from prediction-time source bytes only.
Values that describe a future answer (``full_prompt``, ``note`` and
model-generated labels) never enter the rendered context.  A caller must
provide a verifiable pre-edit document image, or explicitly select one of the
bounded source-builder constructors that deterministically assemble a
simulation from observed row fields; the adapter never assembles an image
from a target.

The public entry point is :func:`convert_completion`.  It returns a JSON-like
mapping so that the admission pass can persist a provenance record without
serialising a dataclass or a protocol exception.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from sepalith.campaign_protocol import (
    Cursor,
    Position,
    PromptContext,
    ProtocolError,
    ReplacementRange,
    codepoint_to_utf16_column,
    parse_output,
    serialize_target,
    utf16_length,
)


ADAPTER_VERSION = "dat04b-completion-adapter-v3"
ALLOWED_SPLITS = frozenset(("train_group", "dev_group"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINE_FIELDS = ("prefix", "region_old", "suffix")
_AUTHORITY_FIELDS = frozenset(("region_new", "corpus_target"))
_FUTURE_FIELDS = frozenset(
    (
        "full_prompt",
        "note",
        "instructions",
        "completion",
        "model_target",
        "target",
        "generated_at",
        "seed",
    )
)


class AdapterError(ValueError):
    """A source row cannot be admitted without guessing."""


@dataclass(frozen=True)
class _Document:
    text: str
    lines: tuple[str, ...]
    eol: str
    content_sha256: str
    path: str


@dataclass(frozen=True)
class _Region:
    start_line: int
    start_codepoint: int
    end_line: int
    end_codepoint: int
    old_lines: tuple[str, ...]
    raw_old_lines: tuple[str, ...]


_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class _CachedSourceFile:
    """One immutable, hash-checked source-file view for a batch pass."""

    path: str
    stat_identity: tuple[int, int, int, int, int]
    source_sha256: str
    rows: Mapping[int, bytes]


def _source_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise AdapterError("source_row_file_unreadable") from error
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


class VerifiedSourceCache:
    """Stream each selected JSONL source file once and verify every row.

    The cache is deliberately bound to the file's identity and metadata.  A
    later lookup re-stat's the file, so a mutation cannot turn a cached row
    into evidence for a different source.  Retained row bytes include the
    original JSONL separator (or lack of one on the final line).
    """

    def __init__(self, *, max_rows: int = 5000) -> None:
        if type(max_rows) is not int or max_rows < 1:
            raise AdapterError("source_cache_max_rows_invalid")
        self.max_rows = max_rows
        self._files: dict[str, _CachedSourceFile] = {}
        self.stream_count = 0
        self.rows_loaded = 0

    @staticmethod
    def _ref_identity(source_ref: Mapping[str, Any]) -> tuple[str, int, str, str]:
        file_value = source_ref.get("file")
        line_value = source_ref.get("line", source_ref.get("line_number"))
        raw_hash = source_ref.get("raw_line_sha256")
        source_hash = source_ref.get("source_sha256")
        if not isinstance(file_value, str) or not file_value:
            raise AdapterError("source_file_missing")
        if type(line_value) is not int or line_value < 1:
            raise AdapterError("source_line_missing")
        if not _is_sha256(raw_hash) or not _is_sha256(source_hash):
            raise AdapterError("source_hash_missing_or_invalid")
        return file_value, line_value, raw_hash, source_hash

    def prime(self, refs: Iterable[Mapping[str, Any]]) -> None:
        """Verify and retain all requested rows, with one stream per file."""
        refs_list = list(refs)
        if len(refs_list) > self.max_rows:
            raise AdapterError("source_cache_row_limit_exceeded")
        requested: dict[str, dict[int, tuple[str, str]]] = {}
        for source_ref in refs_list:
            if not isinstance(source_ref, Mapping):
                raise AdapterError("source_ref_must_be_object")
            file_value, line_value, raw_hash, source_hash = self._ref_identity(source_ref)
            by_line = requested.setdefault(file_value, {})
            previous = by_line.get(line_value)
            if previous is not None and previous != (raw_hash, source_hash):
                raise AdapterError("source_row_identity_disagrees_in_batch")
            by_line[line_value] = (raw_hash, source_hash)

        for file_value, by_line in requested.items():
            path = Path(file_value)
            expected_hashes = {item[1] for item in by_line.values()}
            if len(expected_hashes) != 1:
                raise AdapterError("source_file_hash_disagrees_in_batch")
            expected_source_hash = next(iter(expected_hashes))
            existing = self._files.get(file_value)
            if existing is not None:
                current_identity = _source_stat_identity(path)
                if current_identity != existing.stat_identity:
                    raise AdapterError("source_cache_file_mutated")
                if existing.source_sha256 != expected_source_hash:
                    raise AdapterError("source_file_hash_disagrees_in_batch")
                missing = set(by_line) - set(existing.rows)
                if missing:
                    # Batch callers should prime all rows together.  Refuse a
                    # second partial stream rather than silently regressing
                    # to O(N^2) behavior.
                    raise AdapterError("source_cache_row_not_primed")
                continue

            before = _source_stat_identity(path)
            digest = hashlib.sha256()
            retained: dict[int, bytes] = {}
            try:
                with path.open("rb", buffering=4 * 1024 * 1024) as handle:
                    for index, line_bytes in enumerate(handle, 1):
                        digest.update(line_bytes)
                        if index in by_line:
                            retained[index] = line_bytes
            except OSError as error:
                raise AdapterError("source_row_file_unreadable") from error
            after = _source_stat_identity(path)
            if before != after:
                raise AdapterError("source_file_mutated_during_batch")
            observed_source_hash = digest.hexdigest()
            if observed_source_hash != expected_source_hash:
                raise AdapterError("source_file_hash_mismatch")
            missing = set(by_line) - set(retained)
            if missing:
                raise AdapterError("source_line_out_of_range")
            for line_value, (raw_hash, _source_hash) in by_line.items():
                if _sha256_bytes(retained[line_value]) != raw_hash:
                    raise AdapterError("source_raw_line_hash_mismatch")
            self._files[file_value] = _CachedSourceFile(
                path=file_value,
                stat_identity=after,
                source_sha256=observed_source_hash,
                rows=dict(retained),
            )
            self.stream_count += 1
            self.rows_loaded += len(retained)

    def line_bytes(self, source_ref: Mapping[str, Any]) -> bytes:
        """Return a verified row or fail closed if the file changed."""
        file_value, line_value, raw_hash, source_hash = self._ref_identity(source_ref)
        entry = self._files.get(file_value)
        if entry is None:
            raise AdapterError("source_cache_file_not_primed")
        if _source_stat_identity(Path(file_value)) != entry.stat_identity:
            raise AdapterError("source_cache_file_mutated")
        if entry.source_sha256 != source_hash:
            raise AdapterError("source_file_hash_mismatch")
        line_bytes = entry.rows.get(line_value)
        if line_bytes is None:
            raise AdapterError("source_cache_row_not_primed")
        if _sha256_bytes(line_bytes) != raw_hash:
            raise AdapterError("source_raw_line_hash_mismatch")
        return line_bytes


def _excluded(
    reason: str,
    *,
    provenance: Mapping[str, Any] | None = None,
    context: PromptContext | None = None,
    target_body: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the stable exclusion shape used by the admission caller."""
    return {
        "status": "excluded",
        "context": context.to_dict() if context is not None else None,
        "target_body": list(target_body),
        "operation": None,
        "provenance": dict(provenance or {}),
        "reason": reason,
    }


def _converted(
    *,
    context: PromptContext,
    target_body: Sequence[str],
    operation: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "status": "converted",
        "context": context.to_dict(),
        "target_body": list(target_body),
        "operation": operation,
        "provenance": dict(provenance),
    }
    # Keep the prediction-time source window available to admission callers
    # without putting it into the rendered context or target label.
    if isinstance(provenance.get("selection_source"), Mapping):
        result["selection_source"] = dict(provenance["selection_source"])
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_sha256(value: object) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _as_lines(value: object, name: str, *, allow_none: bool = False) -> tuple[str, ...]:
    """Read an existing line array without trimming any code-point."""
    if value is None and allow_none:
        return ()
    if not isinstance(value, (list, tuple)):
        raise AdapterError(f"{name}_must_be_line_array")
    result: list[str] = []
    for line in value:
        if not isinstance(line, str):
            raise AdapterError(f"{name}_contains_non_string")
        if "\n" in line or "\r" in line:
            raise AdapterError(f"{name}_contains_embedded_newline")
        result.append(line)
    return tuple(result)


def _text_lines(value: object, name: str) -> tuple[str, ...]:
    """Convert a complete text label to LF line entries losslessly.

    A terminal LF is retained as a final empty line.  CRLF is transport
    syntax and is normalised to an LF wire line; lone CR is rejected because
    it has no unambiguous wire interpretation.
    """
    if not isinstance(value, str):
        raise AdapterError(f"{name}_must_be_string_or_line_array")
    if "\r" in value:
        if "\r" not in value.replace("\r\n", ""):
            value = value.replace("\r\n", "\n")
        else:
            raise AdapterError(f"{name}_contains_lone_cr")
    return tuple(value.split("\n"))


def _candidate_lines(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return _as_lines(value, name)
    return _text_lines(value, name)


def _detect_eol(text: str) -> str:
    crlf = text.count("\r\n")
    bare_lf = text.count("\n") - crlf
    bare_cr = text.replace("\r\n", "").count("\r")
    if bare_cr or (crlf and bare_lf):
        return "mixed"
    return "crlf" if crlf else "lf"


def _split_document(text: str) -> tuple[str, ...]:
    # Match the TypeScript selector's text.split("\n") geometry, including
    # the real empty line after a terminal LF.  CRLF's carriage return is
    # transport syntax on every non-final split entry; lone CR was rejected
    # by _detect_eol before this function is called.
    parts = text.split("\n")
    return tuple(
        part[:-1] if index < len(parts) - 1 and part.endswith("\r") else part
        for index, part in enumerate(parts)
    )


def _read_document(source_ref: Mapping[str, Any], raw: Mapping[str, Any]) -> _Document:
    role = source_ref.get("document_role")
    allowed_roles = {"pre_edit_observed", "source_derived_simulated_pre_edit"}
    if role not in allowed_roles:
        raise AdapterError("pre_edit_document_role_not_declared")
    document_path = source_ref.get("document_path")
    supplied_text = source_ref.get("document_text")
    if document_path is not None and supplied_text is not None:
        raise AdapterError("document_path_and_document_text_both_supplied")
    if document_path is not None:
        if not isinstance(document_path, str) or not document_path:
            raise AdapterError("document_path_invalid")
        path = Path(document_path)
        try:
            raw_bytes = path.read_bytes()
        except OSError as error:
            raise AdapterError("document_path_unreadable") from error
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AdapterError("document_not_utf8") from error
        observed_hash = _sha256_bytes(raw_bytes)
        display_path = str(path)
    elif supplied_text is not None:
        if not isinstance(supplied_text, str):
            raise AdapterError("document_text_invalid")
        text = supplied_text
        observed_hash = _sha256_text(text)
        display_path = "<provided-pre-edit-document>"
    else:
        constructor = source_ref.get("source_constructor")
        if constructor == "finish_block_v5_prefix":
            if raw.get("family") != "finish_block" or not isinstance(raw.get("prefix"), str):
                raise AdapterError("finish_block_constructor_input_invalid")
            # assemble_sft_v5 builds the prediction context from the authored
            # prefix and places the cursor at its final line.  The target and
            # full_prompt are deliberately absent from this simulated image.
            text = raw["prefix"]
            raw_bytes = text.encode("utf-8")
            observed_hash = _sha256_bytes(raw_bytes)
            display_path = "<source-derived:finish_block_v5_prefix>"
        elif constructor == "scenario_lines_lf":
            try:
                pieces = []
                for field in ("prefix", "region_old", "suffix"):
                    pieces.extend(_as_lines(raw.get(field, []), field))
            except AdapterError:
                raise
            text = "\n".join(pieces)
            observed_hash = _sha256_text(text)
            display_path = "<source-derived:scenario_lines_lf>"
        else:
            raise AdapterError("missing_pre_edit_document")
    declared_hash = source_ref.get("document_sha256")
    if declared_hash is not None:
        if not _is_sha256(declared_hash):
            raise AdapterError("document_sha256_invalid")
        if observed_hash != declared_hash:
            raise AdapterError("document_sha256_mismatch")
    declared_eol = source_ref.get("document_eol")
    eol = _detect_eol(text)
    if declared_eol is not None and declared_eol != eol:
        raise AdapterError("document_eol_mismatch")
    if eol == "mixed":
        # The frozen application contract intentionally requires a caller
        # policy before mapping mixed EOL wire output.
        raise AdapterError("mixed_document_eol_requires_policy")
    return _Document(text, _split_document(text), eol, observed_hash, display_path)


def _observed_parts(
    raw: Mapping[str, Any], source_ref: Mapping[str, Any], document: _Document,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return prefix, selected old lines and suffix from observed row fields.

    The finish-block source builder stores the observed source as a single
    string in ``prefix``.  Its final document line is the current line and
    all preceding lines are prefix context.  This constructor is intentionally
    source-only: target and teacher fields are never consulted.
    """
    constructor = source_ref.get("source_constructor")
    if constructor == "finish_block_v5_prefix":
        if raw.get("family") != "finish_block" or not isinstance(raw.get("prefix"), str):
            raise AdapterError("finish_block_constructor_input_invalid")
        observed_lines = _split_document(raw["prefix"])
        if not observed_lines:
            raise AdapterError("finish_block_observed_prefix_empty")
        prefix = observed_lines[:-1]
        raw_old = (observed_lines[-1],)
        suffix = _as_lines(raw.get("suffix", []), "suffix")
        if tuple(document.lines) != observed_lines:
            raise AdapterError("source_constructor_document_mismatch")
        return prefix, raw_old, suffix
    if constructor == "scenario_lines_lf":
        prefix = _as_lines(raw.get("prefix", []), "prefix")
        raw_old = _as_lines(raw.get("region_old", []), "region_old")
        suffix = _as_lines(raw.get("suffix", []), "suffix")
        expected = tuple(prefix) + tuple(raw_old) + tuple(suffix)
        if tuple(document.lines) != expected:
            raise AdapterError("source_constructor_document_mismatch")
        return prefix, raw_old, suffix
    prefix = _as_lines(raw.get("prefix", []), "prefix")
    raw_old = _as_lines(raw.get("region_old", []), "region_old")
    suffix = _as_lines(raw.get("suffix", []), "suffix")
    return prefix, raw_old, suffix


def _read_line_bytes(path: Path, line_number: int) -> bytes:
    if type(line_number) is not int or line_number < 1:
        raise AdapterError("source_line_must_be_one_based_positive")
    try:
        with path.open("rb") as handle:
            for index, line in enumerate(handle, 1):
                if index == line_number:
                    return line
    except OSError as error:
        raise AdapterError("source_row_file_unreadable") from error
    raise AdapterError("source_line_out_of_range")


def _file_sha256(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError as error:
        raise AdapterError("source_row_file_unreadable") from error
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[key]
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AdapterError("source_row_file_unreadable") from error
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def _verify_source_ref(
    raw: Mapping[str, Any], source_ref: Mapping[str, Any],
    *, verified_source_cache: VerifiedSourceCache | None = None,
) -> dict[str, Any]:
    """Verify DAT-02/DAT-03 row identity before opening target values."""
    split = source_ref.get("split", raw.get("split"))
    if split not in ALLOWED_SPLITS:
        raise AdapterError("split_not_admitted")
    if raw.get("split") is not None and raw.get("split") != split:
        raise AdapterError("raw_and_source_split_disagree")

    file_value = source_ref.get("file")
    line_value = source_ref.get("line", source_ref.get("line_number"))
    raw_hash = source_ref.get("raw_line_sha256")
    source_hash = source_ref.get("source_sha256")
    parsed_line: Mapping[str, Any] | None = None
    if any(value is not None for value in (file_value, line_value, raw_hash, source_hash)):
        if not isinstance(file_value, str) or not file_value:
            raise AdapterError("source_file_missing")
        if type(line_value) is not int or line_value < 1:
            raise AdapterError("source_line_missing")
        if not _is_sha256(raw_hash) or not _is_sha256(source_hash):
            raise AdapterError("source_hash_missing_or_invalid")
        source_path = Path(file_value)
        line_bytes = (
            verified_source_cache.line_bytes(source_ref)
            if verified_source_cache is not None
            else _read_line_bytes(source_path, line_value)
        )
        if _sha256_bytes(line_bytes) != raw_hash:
            raise AdapterError("source_raw_line_hash_mismatch")
        if (
            verified_source_cache is None
            and _file_sha256(source_path) != source_hash
        ):
            raise AdapterError("source_file_hash_mismatch")
        try:
            loaded = json.loads(line_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterError("source_row_not_json") from error
        if not isinstance(loaded, Mapping) or dict(loaded) != dict(raw):
            raise AdapterError("source_row_content_mismatch")
        parsed_line = loaded
    canonical_hash = source_ref.get("canonical_row_sha256")
    if canonical_hash is not None:
        if not _is_sha256(canonical_hash):
            raise AdapterError("canonical_row_hash_invalid")
        if _canonical_json_sha256(raw) != canonical_hash:
            raise AdapterError("canonical_row_hash_mismatch")
    row_id = source_ref.get("row_id")
    if row_id is not None and not isinstance(row_id, str):
        raise AdapterError("source_row_id_invalid")
    return {
        "split": split,
        "file": file_value,
        "line": line_value,
        "raw_line_sha256": raw_hash,
        "source_sha256": source_hash,
        "row_id": row_id,
        "canonical_row_sha256": canonical_hash,
        "row_verified": parsed_line is not None,
    }


def _validate_raw_identity(raw: object) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise AdapterError("raw_row_must_be_object")
    if not isinstance(raw.get("family"), str) or not raw.get("family"):
        raise AdapterError("family_missing")
    if not isinstance(raw.get("path"), str) or not raw.get("path"):
        raise AdapterError("path_missing")
    path = raw["path"]
    if path.startswith("/") or "\\" in path or "\x00" in path:
        raise AdapterError("path_must_be_relative_posix")
    for flag in ("explicit_truncation", "target_truncated", "prompt_truncated",
                 "corpus_target_truncated", "model_target_truncated"):
        if raw.get(flag) is True:
            raise AdapterError("source_or_target_truncated")
    return raw


def _infer_target_convention(raw: Mapping[str, Any], source_ref: Mapping[str, Any]) -> str:
    explicit = source_ref.get("target_convention", raw.get("target_convention"))
    if explicit in ("complete", "suffix"):
        return explicit
    if raw.get("family") in ("mid_roxygen", "finish_block"):
        return "suffix"
    return "complete"


def _cursor_mode(raw: Mapping[str, Any], source_ref: Mapping[str, Any]) -> str:
    explicit = source_ref.get("cursor_mode", raw.get("cursor_mode"))
    if explicit in ("line_end", "line_index"):
        return "line_index"
    if explicit in ("codepoint_offset", "character_offset"):
        return "codepoint_offset"
    if raw.get("family") in ("no_op", "mid_roxygen"):
        return "line_index"
    return "codepoint_offset"


def _cursor_for_region(
    raw: Mapping[str, Any],
    source_ref: Mapping[str, Any],
    old_lines: Sequence[str],
) -> Cursor:
    if not old_lines:
        return Cursor(-1, None, None)
    explicit = source_ref.get("cursor")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise AdapterError("cursor_invalid")
        try:
            cursor = Cursor.from_dict(explicit)
        except ProtocolError as error:
            raise AdapterError("cursor_invalid") from error
        return cursor
    index = raw.get("cursor_idx", source_ref.get("cursor_idx"))
    if type(index) is not int:
        raise AdapterError("cursor_geometry_missing")
    if _cursor_mode(raw, source_ref) == "line_index":
        if not 0 <= index < len(old_lines):
            raise AdapterError("cursor_line_index_out_of_range")
        cp = len(old_lines[index])
        return Cursor(index, cp, codepoint_to_utf16_column(old_lines[index], cp))
    if index < 0 or index > len("\n".join(old_lines)):
        raise AdapterError("cursor_offset_out_of_range")
    before = "\n".join(old_lines)[:index]
    line_index = before.count("\n")
    last_break = before.rfind("\n")
    cp = index if last_break < 0 else index - last_break - 1
    line = old_lines[line_index]
    return Cursor(line_index, cp, codepoint_to_utf16_column(line, cp))


def _line_match_candidates(
    document: _Document,
    prefix: Sequence[str],
    old_lines: Sequence[str],
    suffix: Sequence[str],
    source_ref: Mapping[str, Any],
) -> list[tuple[int, int]]:
    """Find (line, character) occurrences of the observed selected region."""
    explicit_line = source_ref.get("region_start_line")
    explicit_char = source_ref.get("region_start_character", 0)
    if explicit_line is not None:
        if type(explicit_line) is not int or explicit_line < 0:
            raise AdapterError("region_start_line_invalid")
        if type(explicit_char) is not int or explicit_char < 0:
            raise AdapterError("region_start_character_invalid")
        if explicit_line >= len(document.lines):
            raise AdapterError("region_start_line_out_of_range")
        return [(explicit_line, explicit_char)]
    candidates: list[tuple[int, int]] = []
    if old_lines:
        for start in range(len(document.lines)):
            if start + len(old_lines) > len(document.lines):
                continue
            if tuple(document.lines[start:start + len(old_lines)]) != tuple(old_lines):
                # A one-line completion may select a source prefix, not the
                # entire source line.  Find only an unambiguous occurrence.
                if len(old_lines) != 1 or not old_lines[0]:
                    continue
                line = document.lines[start]
                positions = [m.start() for m in re.finditer(
                    re.escape(old_lines[0]), line)]
                if len(positions) != 1:
                    continue
                for pos in positions:
                    candidates.append((start, pos))
                continue
            candidates.append((start, 0))
    else:
        # A canonical empty region is a zero-width insertion at the start of
        # a source line.  Prefix/suffix are used only as observed anchors.
        for start in range(len(document.lines)):
            if prefix and tuple(document.lines[max(0, start - len(prefix)):start]) != tuple(prefix):
                continue
            if suffix and tuple(document.lines[start + 1:start + 1 + len(suffix)]) != tuple(suffix):
                continue
            candidates.append((start, 0))
        if not candidates and prefix and len(document.lines) == len(prefix):
            candidates.append((len(prefix), 0))
    if prefix:
        candidates = [
            item for item in candidates
            if tuple(document.lines[max(0, item[0] - len(prefix)):item[0]]) == tuple(prefix)
        ]
    # Suffix-convention rows deliberately hide the target span from the
    # observed document. Their `suffix` begins after that hidden span, so it
    # is checked against the prediction-time window after the same-line
    # cursor is constructed below. Complete replacement rows have an
    # immediately following suffix and can be checked here.
    suffix_is_immediate = source_ref.get(
        "suffix_is_immediate",
        _infer_target_convention({}, source_ref) != "suffix",
    )
    if suffix and old_lines and suffix_is_immediate:
        candidates = [
            item for item in candidates
            if tuple(document.lines[item[0] + len(old_lines):
                                    item[0] + len(old_lines) + len(suffix)]) == tuple(suffix)
        ]
    return candidates


def _make_region(
    document: _Document,
    raw_old_lines: Sequence[str],
    prefix: Sequence[str],
    suffix: Sequence[str],
    source_ref: Mapping[str, Any],
) -> _Region:
    # PromptContext reserves [''] as a representation of no selected text.
    # Keep the raw value in provenance but use the protocol's canonical [].
    old_lines = () if tuple(raw_old_lines) == ("",) else tuple(raw_old_lines)
    candidates = _line_match_candidates(document, prefix, raw_old_lines, suffix, source_ref)
    if not candidates:
        raise AdapterError("pre_edit_region_not_found")
    if len(candidates) != 1:
        raise AdapterError("pre_edit_region_ambiguous")
    start_line, start_cp = candidates[0]
    if not old_lines:
        return _Region(start_line, start_cp, start_line, start_cp, (), tuple(raw_old_lines))
    end_line = start_line + len(old_lines) - 1
    end_cp = start_cp + len(old_lines[-1]) if len(old_lines) == 1 else len(old_lines[-1])
    if end_line >= len(document.lines):
        raise AdapterError("pre_edit_region_end_out_of_range")
    if start_cp > len(document.lines[start_line]):
        raise AdapterError("region_start_character_out_of_range")
    if end_cp > len(document.lines[end_line]):
        raise AdapterError("region_end_character_out_of_range")
    selected = _selected_text(document.lines, start_line, start_cp, end_line, end_cp)
    if selected != "\n".join(old_lines):
        raise AdapterError("pre_edit_region_geometry_mismatch")
    return _Region(start_line, start_cp, end_line, end_cp, old_lines, tuple(raw_old_lines))


def _selected_text(
    lines: Sequence[str], start_line: int, start_cp: int,
    end_line: int, end_cp: int,
) -> str:
    if start_line == end_line:
        return lines[start_line][start_cp:end_cp]
    parts = [lines[start_line][start_cp:]]
    parts.extend(lines[start_line + 1:end_line])
    parts.append(lines[end_line][:end_cp])
    return "\n".join(parts)


def _replacement_range(
    document: _Document,
    region: _Region,
    source_ref: Mapping[str, Any],
) -> ReplacementRange:
    uri = source_ref.get("uri")
    version = source_ref.get("document_version")
    if not isinstance(uri, str) or not uri:
        raise AdapterError("editor_uri_missing")
    if type(version) is not int or version < 0:
        raise AdapterError("editor_document_version_missing")
    start = Position(region.start_line, utf16_length(document.lines[region.start_line][:region.start_codepoint]))
    end = Position(region.end_line, utf16_length(document.lines[region.end_line][:region.end_codepoint]))
    return ReplacementRange(uri, version, document.content_sha256, start, end)


def _target_candidates(raw: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    authoritative: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    for field in ("region_new", "corpus_target", "model_target", "target"):
        if field not in raw or raw[field] is None:
            continue
        try:
            lines = _candidate_lines(raw[field], field)
        except AdapterError as error:
            item = {"field": field, "error": str(error)}
            if field in _AUTHORITY_FIELDS:
                authoritative.append(item)
            else:
                alternatives.append(item)
            continue
        item = {
            "field": field,
            "lines": list(lines),
            "text_sha256": _sha256_text("\n".join(lines)),
            "line_count": len(lines),
            "char_count": len("\n".join(lines)),
        }
        # Generic target/model labels are alternatives unless the caller
        # explicitly records that the generic field is corpus authoritative.
        if field in _AUTHORITY_FIELDS or raw.get(f"{field}_authority") in ("source", "corpus"):
            authoritative.append(item)
        else:
            alternatives.append(item)
    return authoritative, alternatives


def _normalise_suffix_target(
    lines: Sequence[str], raw: Mapping[str, Any], convention: str,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Preserve finish-block continuation bytes exactly.

    ``assemble_sft_v5`` emits a literal source remainder.  A leading LF is
    meaningful after a signature prefix ending at ``{`` and after a mid-body
    prefix ending at an empty EOF line, so no framing byte is removed here.
    """
    result = tuple(lines)
    framing: dict[str, Any] = {"leading_framing_lf_removed": False}
    if convention == "suffix" and raw.get("family") == "finish_block":
        # finish_block's target is a literal source remainder.  A leading LF
        # is meaningful when the prefix ends at `{`, and is equally
        # meaningful when a mid-body prefix ends at a real empty EOF line.
        framing["finish_target_bytes_preserved"] = True
    return result, framing


def _finish_replacement(
    current_line: str, target_lines: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact finish-block splice body, without adding/removing LF."""
    if not target_lines:
        return (current_line,) if current_line else ()
    target_text = "\n".join(target_lines)
    return tuple((current_line + target_text).split("\n"))


def _verify_finish_literal_splice(
    raw: Mapping[str, Any], document: _Document, current_line: str,
    target_body: Sequence[str],
) -> None:
    """Prove the finish source constructor's byte-contiguous post-edit rule."""
    if raw.get("family") != "finish_block":
        return
    source_prefix = raw.get("prefix")
    if not isinstance(source_prefix, str) or source_prefix != document.text:
        raise AdapterError("finish_source_prefix_document_mismatch")
    source_target = raw.get("corpus_target")
    if source_target is None:
        source_target = raw.get("region_new")
    if isinstance(source_target, str):
        source_target_text = source_target
    elif isinstance(source_target, (list, tuple)):
        source_target_text = "\n".join(_as_lines(source_target, "corpus_target"))
    else:
        raise AdapterError("finish_source_target_text_missing")
    body_text = "\n".join(target_body)
    if current_line:
        if not document.text.endswith(current_line):
            raise AdapterError("finish_current_line_not_at_source_end")
        applied = document.text[:-len(current_line)] + body_text
    else:
        if not document.text.endswith(("\n", "\r")):
            raise AdapterError("finish_empty_current_line_not_at_eof")
        applied = document.text + body_text
    if applied != source_prefix + source_target_text:
        raise AdapterError("finish_literal_prefix_target_splice_mismatch")


def _suffix_replacement(
    current_line: str,
    cursor_cp: int,
    completion_lines: Sequence[str],
    source_ref: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> tuple[str, ...]:
    """Make the full same-line replacement expected by live inline serving."""
    if cursor_cp < 0 or cursor_cp > len(current_line):
        raise AdapterError("suffix_cursor_out_of_current_line")
    join_mode = source_ref.get("completion_join", raw.get("completion_join", "next_line"))
    if join_mode not in ("next_line", "same_line"):
        raise AdapterError("completion_join_invalid")
    continuation = "\n".join(completion_lines)
    if not completion_lines:
        # Empty continuation is a semantic no-op for a suffix row.  Keeping
        # the full current line makes this explicit and avoids a zero-width
        # serving route.
        return (current_line,)
    if join_mode == "same_line":
        text = current_line[:cursor_cp] + continuation + current_line[cursor_cp:]
    else:
        text = current_line[:cursor_cp] + "\n" + continuation + current_line[cursor_cp:]
    return tuple(text.split("\n"))


def _target_operation(
    old_lines: Sequence[str], target_lines: Sequence[str], *, noop_sentinel: bool
) -> tuple[str, tuple[str, ...]]:
    if noop_sentinel:
        return "no_op", tuple(old_lines)
    target = tuple(target_lines)
    old = tuple(old_lines)
    if target == old:
        return "no_op", old
    if not target:
        return ("delete", ()) if old else ("no_op", ())
    if target == ("",):
        raise AdapterError("canonical_empty_target_must_use_empty_array")
    return "replace", target


def _check_target_roundtrip(
    context: PromptContext,
    operation: str,
    target_body: Sequence[str],
) -> None:
    try:
        serialized = serialize_target(operation, target_body)
    except ProtocolError as error:
        raise AdapterError("target_reserved_or_unrepresentable") from error
    parsed = parse_output(serialized, context)
    if parsed.status != "accepted" or parsed.operation != operation:
        raise AdapterError("target_serialization_roundtrip_failed")
    if operation == "replace" and parsed.body != tuple(target_body):
        raise AdapterError("target_serialization_changed_bytes")
    if operation == "delete" and parsed.body:
        raise AdapterError("delete_target_roundtrip_nonempty")


def _target_is_reserved(target_body: Sequence[str]) -> bool:
    # Keep this local to avoid depending on a private protocol constant.
    reserved = {
        "<<<<<<< CURRENT", "=======", ">>>>>>> UPDATED", "<[fim-middle]>",
        "<[fim-prefix]>", "<[fim-suffix]>", "<|user_cursor|>", "<|outline|>",
        "[NO_EDIT]",
    }
    return any(line.strip() in reserved for line in target_body)


def convert_completion(
    raw: object, source_ref: object, *,
    verified_source_cache: VerifiedSourceCache | None = None,
) -> dict[str, Any]:
    """Convert one audited row into a frozen prediction context and label.

    ``source_ref`` must contain the DAT-03 JSONL identity (split, file, line
    and hashes), either a pre-edit observed document (path or text) or an
    explicit source-derived constructor, and editor identity (URI/version).
    Target values are read only for the out-of-band label.  A target
    disagreement between source-authoritative fields is an exclusion; a
    differing model/generic label is retained as a quarantined generative
    alternative in provenance.
    """
    base_provenance: dict[str, Any] = {
        "adapter_id": ADAPTER_VERSION,
        "source_role": "verified_train_or_dev_row",
        "history_provider": "none",
        "diagnostics_provider": "none",
        "retrieval_provider": "none",
        "future_context_fields_ignored": sorted(_FUTURE_FIELDS),
    }
    try:
        row = _validate_raw_identity(raw)
        if not isinstance(source_ref, Mapping):
            raise AdapterError("source_ref_must_be_object")
        source_identity = _verify_source_ref(
            row, source_ref, verified_source_cache=verified_source_cache,
        )
        document = _read_document(source_ref, row)
        convention = _infer_target_convention(row, source_ref)
        prefix, raw_old, suffix = _observed_parts(row, source_ref, document)
        region_ref = dict(source_ref)
        region_ref["target_convention"] = convention
        region = _make_region(document, raw_old, prefix, suffix, region_ref)
        old_lines = region.old_lines
        cursor_row = dict(row)
        if (
            convention == "suffix"
            and row.get("family") == "finish_block"
            and row.get("cursor") is None
            and row.get("cursor_idx") is None
            and source_ref.get("cursor") is None
            and source_ref.get("cursor_idx") is None
        ):
            # The authored finish prefix ends at the cursor.  This is a
            # source-derived observation, not a target-derived cursor.
            cursor_row["cursor_idx"] = len(old_lines[-1]) if old_lines else 0
            cursor_row["cursor_mode"] = "codepoint_offset"
        cursor = _cursor_for_region(cursor_row, source_ref, old_lines)
        authoritative, alternatives = _target_candidates(row)
        if not authoritative:
            raise AdapterError("no_authoritative_complete_target")
        errors = [item for item in authoritative if "error" in item]
        if errors:
            raise AdapterError("authoritative_target_malformed")
        target_values = {tuple(item["lines"]) for item in authoritative}
        if len(target_values) != 1:
            raise AdapterError("conflicting_authoritative_targets")
        selected_source_target = next(iter(target_values))
        selected_source_target, target_framing = _normalise_suffix_target(
            selected_source_target, row, convention
        )
        noop_sentinel = (
            row.get("noop_status") == "authoritative_noop_sentinel"
            or (row.get("family") == "no_op" and not selected_source_target)
        )
        if convention == "suffix":
            # The source row is a completion after the typed head.  Live
            # InlineCompletionItem selects the whole current line, so this
            # adapter emits that same-line replacement shape.  The hidden
            # completion span is never used to build the pre-edit document.
            finish_eof_empty = (
                row.get("family") == "finish_block"
                and not old_lines
                and tuple(region.raw_old_lines) == ("",)
                and cursor.region_line_index == -1
            )
            if not finish_eof_empty and (
                not old_lines or cursor.region_line_index != len(old_lines) - 1
            ):
                raise AdapterError("suffix_completion_requires_cursor_at_region_end")
            current_index = -1 if finish_eof_empty else cursor.region_line_index
            current_line_number = region.start_line if finish_eof_empty else region.start_line + current_index
            current_line = document.lines[current_line_number]
            if not finish_eof_empty and (region.start_codepoint != 0 or current_line != old_lines[-1]):
                raise AdapterError("suffix_current_line_not_fully_observed")
            cursor_cp = 0 if finish_eof_empty else cursor.code_point_column
            if cursor_cp is None:
                raise AdapterError("suffix_cursor_codepoint_missing")
            # Any remaining raw old lines and the declared suffix are
            # prediction-time observations.  A source-derived constructor
            # places them directly after the current line because it cannot
            # fabricate the hidden target span.
            observed_after = (() if finish_eof_empty else tuple(old_lines[current_index + 1:])) + tuple(suffix)
            actual_after = tuple(document.lines[current_line_number + 1:
                                               current_line_number + 1 + len(observed_after)])
            if actual_after != observed_after:
                raise AdapterError("pre_edit_suffix_not_found_after_cursor")
            context_prefix = tuple(prefix) + (() if finish_eof_empty else tuple(old_lines[:current_index]))
            context_suffix = observed_after
            context_old = () if finish_eof_empty else (current_line,)
            context_cursor = (
                Cursor(-1, None, None)
                if finish_eof_empty else Cursor(
                    0, cursor_cp, codepoint_to_utf16_column(current_line, cursor_cp)
                )
            )
            same_line_region = _Region(
                current_line_number, 0, current_line_number, len(current_line),
                () if finish_eof_empty else (current_line,),
                ("",) if finish_eof_empty else (current_line,),
            )
            replacement_range = _replacement_range(document, same_line_region, source_ref)
            selected_target = (
                _finish_replacement(current_line, selected_source_target)
                if row.get("family") == "finish_block" else _suffix_replacement(
                    current_line, cursor_cp, selected_source_target, source_ref, row
                )
            )
            operation, target_body = _target_operation(
                context_old, selected_target, noop_sentinel=noop_sentinel
            )
            _verify_finish_literal_splice(
                row, document, current_line, target_body
            )
            region = same_line_region
        else:
            selected_target = selected_source_target
            operation, target_body = _target_operation(
                old_lines, selected_target, noop_sentinel=noop_sentinel
            )
            context_prefix = prefix
            context_suffix = suffix
            context_old = old_lines
            context_cursor = cursor
            replacement_range = _replacement_range(document, region, source_ref)
        if _target_is_reserved(target_body) and operation == "replace":
            raise AdapterError("reserved_target_line")

        context = PromptContext(
            path=row["path"],
            prefix=context_prefix,
            selected_references=(),
            history=(),
            diagnostics=(),
            retrieval=(),
            scope_mode="off",
            scope_lines=(),
            suffix_lines=context_suffix,
            region_old=context_old,
            cursor=context_cursor,
            replacement_range=replacement_range,
            document_eol=document.eol,
        )
        _check_target_roundtrip(context, operation, target_body)
        alternative_disposition = "none"
        generative_alternatives: list[dict[str, Any]] = []
        for item in alternatives:
            copy = dict(item)
            if "lines" in copy:
                candidate_lines, _ = _normalise_suffix_target(
                    tuple(copy["lines"]), row, convention
                )
                candidate_body = (
                    (
                        _finish_replacement(
                            context_old[0] if context_old else "", candidate_lines
                        )
                        if row.get("family") == "finish_block" else _suffix_replacement(
                            context_old[0], context_cursor.code_point_column or 0,
                            candidate_lines, source_ref, row,
                        )
                    )
                    if convention == "suffix" and (context_old or row.get("family") == "finish_block")
                    else candidate_lines
                )
                copy["semantic_target_matches"] = candidate_body == tuple(target_body)
                if candidate_body != tuple(target_body):
                    alternative_disposition = "authoritative_source_selected_alternative_quarantined"
            generative_alternatives.append(copy)
        if generative_alternatives and alternative_disposition == "none":
            alternative_disposition = "alternatives_consistent_or_unusable"
        provenance = {
            **base_provenance,
            "source_identity": source_identity,
            "source_family": row["family"],
            "source_path": row["path"],
            "target_convention": convention,
            "context_selection": (
                "same_line_primary_replacement_for_suffix_completion"
                if convention == "suffix" else "complete_replacement_region"
            ),
            "unsupported_route_candidate": (
                {
                    "route": "zero_width_suffix_insertion",
                    "status": "unsupported_primary_route",
                    "reason": "live_inline_completion_selects_whole_current_line",
                }
                if convention == "suffix" else None
            ),
            "target_framing": target_framing,
            "finish_splice": (
                {
                    "literal_source_splice_verified": True,
                    "outer_closing_brace_in_label": False,
                }
                if row.get("family") == "finish_block" else None
            ),
            "target_authority": [item["field"] for item in authoritative],
            "target_candidates": authoritative,
            "generative_alternatives": generative_alternatives,
            "conflict_disposition": alternative_disposition,
            "raw_region_old": list(region.raw_old_lines),
            "region_geometry": {
                "start_line": region.start_line,
                "start_codepoint": region.start_codepoint,
                "end_line": region.end_line,
                "end_codepoint": region.end_codepoint,
                "selected_text_sha256": _sha256_text("\n".join(old_lines)),
                "whole_region_verified": True,
            },
            "pre_edit_document": {
                "path": document.path,
                "content_sha256": document.content_sha256,
                "document_eol": document.eol,
                "document_version": source_ref["document_version"],
                "lineage": source_ref["document_role"],
                "source_constructor": source_ref.get("source_constructor"),
            },
            "selection_source": {
                "availability": (
                    "full_snapshot"
                    if source_ref["document_role"] == "pre_edit_observed"
                    else "source_builder_window"
                ),
                "lineage": source_ref["document_role"],
                "path": document.path,
                "content_sha256": document.content_sha256,
                "document_text": (
                    document.text
                    if source_ref["document_role"] == "source_derived_simulated_pre_edit"
                    else None
                ),
                "region_start_line": region.start_line,
                "region_end_line": region.end_line,
                "document_version": source_ref["document_version"],
                "source_jsonl_path": source_identity.get("file"),
                "source_jsonl_line": source_identity.get("line"),
                "source_jsonl_sha256": source_identity.get("source_sha256"),
                "source_jsonl_raw_line_sha256": source_identity.get("raw_line_sha256"),
            },
            "ignored_future_fields": sorted(
                field for field in _FUTURE_FIELDS if field in row
            ),
            "target_body_sha256": _sha256_text("\n".join(target_body)),
            "operation": operation,
        }
        return _converted(
            context=context,
            target_body=target_body,
            operation=operation,
            provenance=provenance,
        )
    except AdapterError as error:
        provenance = dict(base_provenance)
        if isinstance(source_ref, Mapping):
            for key in ("row_id", "split", "file", "line", "raw_line_sha256", "source_sha256"):
                if key in source_ref:
                    provenance.setdefault("source_identity", {})[key] = source_ref[key]
        return _excluded(str(error), provenance=provenance)
    except (KeyError, TypeError, ValueError, ProtocolError) as error:
        return _excluded(f"adapter_error:{type(error).__name__}", provenance=base_provenance)


def source_ref_from_audit(audit_row: Mapping[str, Any], *, document_path: str,
                         uri: str, document_version: int) -> dict[str, Any]:
    """Build a reference from a DAT-03 row-audit entry.

    This helper copies identity only.  It does not copy target fields or
    ``full_prompt`` and therefore cannot turn the audit metadata into a
    future answer by accident.
    """
    required = ("file", "line", "raw_line_sha256", "source_sha256", "split")
    missing = [key for key in required if key not in audit_row]
    if missing:
        raise AdapterError("audit_reference_missing:" + ",".join(missing))
    result = {key: audit_row[key] for key in required}
    for key in ("row_id", "canonical_row_sha256", "group_id"):
        if key in audit_row:
            result[key] = audit_row[key]
    result.update({
        "document_role": "pre_edit_observed",
        "document_path": document_path,
        "uri": uri,
        "document_version": document_version,
    })
    return result


__all__ = [
    "ADAPTER_VERSION", "AdapterError", "VerifiedSourceCache",
    "convert_completion", "source_ref_from_audit",
]
