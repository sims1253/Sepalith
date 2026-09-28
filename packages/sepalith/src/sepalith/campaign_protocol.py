"""PRM-03 shared prompt, output and training-row contract.

This module is deliberately independent from :mod:`sepalith.protocol`.  It
contains pure data validation and text operations so the Python training/eval
side can share fixtures with the TypeScript serving side without importing
VS Code, a model, or a server.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping, Protocol, Sequence


SCHEMA_VERSION = "sepalith.prompt.prm03.v1"
RENDERER_ID = "zeta2-prm03-v1"
# Text is encoded without automatic BOS/EOS insertion and with special-token
# splitting enabled.  The native serving equivalent is
# ``--no-bos --no-parse-special``.  ``build_training_row`` adds BOS 0 once at
# the sequence start and protocol EOS 1 once after the terminal.  The prompt
# renderer's final LF is part of the boundary proof.
TOKENIZATION_POLICY = "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1"
TASK_CONTRACT = (
    "Edit the current region. Output the complete replacement region, "
    "preserving whitespace. For no change, output the ordinary line "
    "[NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an "
    "empty body is reserved for deletion of a nonempty range."
)
TERMINAL = ">>>>>>> UPDATED"
NO_EDIT = "[NO_EDIT]"
CURSOR_MARKER = "<|user_cursor|>"
GENERATION_BOUNDARY = "<[fim-middle]>"
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = (1, 130073)
VOCAB_SIZE = 130560
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_CONFIG_SHA256 = "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCOPES = frozenset(("pin+outline", "outline", "off"))
_EOLS = frozenset(("lf", "crlf", "mixed"))
_RESERVED_BODY_LINES = frozenset(
    (
        "<<<<<<< CURRENT",
        "=======",
        ">>>>>>> UPDATED",
        "<[fim-middle]>",
        "<[fim-prefix]>",
        "<[fim-suffix]>",
        "<|user_cursor|>",
        "<|outline|>",
        NO_EDIT,
    )
)


class TokenEncoder(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        split_special_tokens: bool = True,
    ) -> Sequence[int]: ...


class ProtocolError(ValueError):
    """Input, output or row violated the versioned contract."""


class TokenBoundaryError(ProtocolError):
    """The tokenizer changes a segment at a prompt/response boundary."""


class StaleContextError(ProtocolError):
    """The document changed after the prediction context was captured."""


def is_native_control_token(token: int) -> bool:
    """CONTROL ranges measured from the pinned GGUF, including reserved slots."""
    return 0 <= token <= 7 or 10 <= token <= 21 or 130072 <= token < VOCAB_SIZE


def valid_generation_tokens(tokens: Sequence[int]) -> bool:
    """Require one canonical terminal EOS and no hidden native control tokens."""
    return (bool(tokens) and tokens[-1] == EOS_ID
            and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in tokens)
            and all(not is_native_control_token(token) for token in tokens[:-1]))


def _validate_reserved_slots(text: str) -> None:
    # These added tokens are not marked special in HF, but are CONTROL in
    # native GGUF. Explicit split-special modes therefore encode them
    # differently. Reject these rare literal spellings in both runtimes.
    for match in re.finditer(r"<unused_token_(0|[1-9][0-9]{0,2})>", text):
        if int(match[1]) <= 477:
            raise ProtocolError("reserved_slot_literal_has_inconsistent_tokenization")


def _string(value: object, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ProtocolError(f"{name} must be a {'non-empty ' if nonempty else ''}string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _lines(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ProtocolError(f"{name} must be an array of strings")
    return tuple(value)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing={sorted(missing)}")
        if extra:
            detail.append(f"unexpected={sorted(extra)}")
        raise ProtocolError(f"{name} fields mismatch ({', '.join(detail)})")


def _sha256(value: object, name: str) -> str:
    value = _string(value, name)
    if not _SHA256_RE.fullmatch(value):
        raise ProtocolError(f"{name} must be a lowercase SHA256 hex digest")
    return value


@dataclass(frozen=True)
class Position:
    line: int
    character: int

    def __post_init__(self) -> None:
        _integer(self.line, "position.line")
        _integer(self.character, "position.character")

    @classmethod
    def from_dict(cls, value: object, name: str = "position") -> Position:
        data = _mapping(value, name)
        _exact_keys(data, {"line", "character"}, name)
        return cls(line=_integer(data["line"], f"{name}.line"),
                   character=_integer(data["character"], f"{name}.character"))

    def to_dict(self) -> dict[str, int]:
        return {"line": self.line, "character": self.character}


def _position_key(position: Position) -> tuple[int, int]:
    return position.line, position.character


@dataclass(frozen=True)
class ReplacementRange:
    uri: str
    document_version: int
    content_sha256: str
    start: Position
    end: Position

    def __post_init__(self) -> None:
        _string(self.uri, "replacement_range.uri")
        _integer(self.document_version, "replacement_range.document_version")
        _sha256(self.content_sha256, "replacement_range.content_sha256")
        if not isinstance(self.start, Position) or not isinstance(self.end, Position):
            raise ProtocolError("replacement_range positions must be Position values")
        if _position_key(self.end) < _position_key(self.start):
            raise ProtocolError("replacement_range.end must not precede start")

    @property
    def is_empty(self) -> bool:
        return self.start == self.end

    @property
    def is_same_line(self) -> bool:
        return self.start.line == self.end.line

    @classmethod
    def from_dict(cls, value: object) -> ReplacementRange:
        data = _mapping(value, "replacement_range")
        _exact_keys(data, {"uri", "document_version", "content_sha256", "start", "end"},
                    "replacement_range")
        return cls(
            uri=_string(data["uri"], "replacement_range.uri"),
            document_version=_integer(data["document_version"], "replacement_range.document_version"),
            content_sha256=_sha256(data["content_sha256"], "replacement_range.content_sha256"),
            start=Position.from_dict(data["start"], "replacement_range.start"),
            end=Position.from_dict(data["end"], "replacement_range.end"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "document_version": self.document_version,
            "content_sha256": self.content_sha256,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }


@dataclass(frozen=True)
class Cursor:
    region_line_index: int
    code_point_column: int | None
    utf16_column: int | None

    @classmethod
    def from_dict(cls, value: object) -> Cursor:
        data = _mapping(value, "cursor")
        _exact_keys(data, {"region_line_index", "code_point_column", "utf16_column"}, "cursor")
        index = data["region_line_index"]
        if type(index) is not int or index < -1:
            raise ProtocolError("cursor.region_line_index must be an integer >= -1")
        cp = data["code_point_column"]
        utf16 = data["utf16_column"]
        if cp is not None:
            _integer(cp, "cursor.code_point_column")
        if utf16 is not None:
            _integer(utf16, "cursor.utf16_column")
        return cls(index, cp, utf16)

    def to_dict(self) -> dict[str, int | None]:
        return {
            "region_line_index": self.region_line_index,
            "code_point_column": self.code_point_column,
            "utf16_column": self.utf16_column,
        }


@dataclass(frozen=True)
class EvidenceRecord:
    content: str
    path: str | None = None

    def __post_init__(self) -> None:
        _string(self.content, "evidence.content", nonempty=False)
        if self.path is not None:
            _string(self.path, "evidence.path")

    @classmethod
    def from_dict(cls, value: object, name: str = "evidence") -> EvidenceRecord:
        data = _mapping(value, name)
        _exact_keys(data, {"content", "path"}, name)
        path = data["path"]
        if path is not None:
            path = _string(path, f"{name}.path")
        return cls(content=_string(data["content"], f"{name}.content", nonempty=False), path=path)

    def to_dict(self) -> dict[str, str | None]:
        return {"content": self.content, "path": self.path}


@dataclass(frozen=True)
class HistoryEvent:
    event_id: str
    kind: str
    path: str
    workspace_revision_before: str
    old_text: str
    new_text: str
    range_utf16: ReplacementRange
    event_diff: str

    def __post_init__(self) -> None:
        for name in ("event_id", "kind", "path", "workspace_revision_before", "old_text", "new_text", "event_diff"):
            _string(getattr(self, name), f"history.{name}", nonempty=name not in ("old_text", "new_text", "event_diff"))
        if not isinstance(self.range_utf16, ReplacementRange):
            raise ProtocolError("history.range_utf16 must be a ReplacementRange")

    @classmethod
    def from_dict(cls, value: object) -> HistoryEvent:
        data = _mapping(value, "history event")
        _exact_keys(data, {"event_id", "kind", "path", "workspace_revision_before", "old_text",
                           "new_text", "range_utf16", "event_diff"}, "history event")
        return cls(
            event_id=_string(data["event_id"], "history.event_id"),
            kind=_string(data["kind"], "history.kind"),
            path=_string(data["path"], "history.path"),
            workspace_revision_before=_string(data["workspace_revision_before"], "history.workspace_revision_before"),
            old_text=_string(data["old_text"], "history.old_text", nonempty=False),
            new_text=_string(data["new_text"], "history.new_text", nonempty=False),
            range_utf16=ReplacementRange.from_dict(data["range_utf16"]),
            event_diff=_string(data["event_diff"], "history.event_diff", nonempty=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "path": self.path,
            "workspace_revision_before": self.workspace_revision_before,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "range_utf16": self.range_utf16.to_dict(),
            "event_diff": self.event_diff,
        }


def utf16_length(text: str) -> int:
    """Return the VS Code UTF-16 code-unit length of a Python string."""
    _string(text, "text", nonempty=False)
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def codepoint_to_utf16_column(text: str, column: int) -> int:
    _integer(column, "code-point column")
    if column > len(text):
        raise ProtocolError("code-point column exceeds line length")
    return utf16_length(text[:column])


def utf16_to_codepoint_column(text: str, column: int) -> int:
    _integer(column, "UTF-16 column")
    if column > utf16_length(text):
        raise ProtocolError("UTF-16 column exceeds line length")
    units = 0
    for index, char in enumerate(text):
        if column == units:
            return index
        width = 2 if ord(char) > 0xFFFF else 1
        if units < column < units + width:
            raise ProtocolError("UTF-16 column falls inside a surrogate pair")
        units += width
    return len(text)


def _validate_cursor(cursor: Cursor, region_old: Sequence[str]) -> None:
    if not region_old:
        if cursor != Cursor(-1, None, None):
            raise ProtocolError("empty region requires cursor index -1 and null columns")
        return
    if not 0 <= cursor.region_line_index < len(region_old):
        raise ProtocolError("cursor.region_line_index must address region_old")
    if cursor.code_point_column is None or cursor.utf16_column is None:
        raise ProtocolError("nonempty region requires both cursor columns")
    line = region_old[cursor.region_line_index]
    if cursor.code_point_column > len(line):
        raise ProtocolError("cursor code-point column exceeds line")
    expected_utf16 = codepoint_to_utf16_column(line, cursor.code_point_column)
    if cursor.utf16_column != expected_utf16:
        raise ProtocolError("cursor UTF-16 and code-point columns disagree")
    if utf16_to_codepoint_column(line, cursor.utf16_column) != cursor.code_point_column:
        raise ProtocolError("cursor UTF-16 column is not a code-point boundary")


def _validate_range_geometry(replacement_range: ReplacementRange, region_old: Sequence[str]) -> None:
    """Check that LF-split selected text agrees with the captured VS Code range."""
    if not region_old:
        if not replacement_range.is_empty:
            raise ProtocolError("empty region requires a zero-width replacement range")
        return
    line_span = replacement_range.end.line - replacement_range.start.line
    if line_span != len(region_old) - 1:
        raise ProtocolError("replacement range line span disagrees with region_old line count")
    if line_span == 0:
        expected_width = utf16_length(region_old[0])
        actual_width = replacement_range.end.character - replacement_range.start.character
        if actual_width != expected_width:
            raise ProtocolError("same-line replacement range width disagrees with region_old UTF-16 length")
    elif replacement_range.end.character != utf16_length(region_old[-1]):
        raise ProtocolError("multiline replacement range end disagrees with final region_old UTF-16 length")


@dataclass(frozen=True)
class PromptContext:
    path: str
    prefix: tuple[str, ...]
    selected_references: tuple[EvidenceRecord, ...]
    history: tuple[HistoryEvent, ...]
    diagnostics: tuple[EvidenceRecord, ...]
    retrieval: tuple[EvidenceRecord, ...]
    scope_mode: str
    scope_lines: tuple[str, ...]
    suffix_lines: tuple[str, ...]
    region_old: tuple[str, ...]
    cursor: Cursor
    replacement_range: ReplacementRange
    document_eol: str = "lf"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema_version: {self.schema_version!r}")
        path = _string(self.path, "path")
        if path.startswith("/") or "\\" in path or "\x00" in path:
            raise ProtocolError("path must be a relative POSIX path")
        object.__setattr__(self, "path", path)
        for name in ("prefix", "scope_lines", "suffix_lines", "region_old"):
            object.__setattr__(self, name, _lines(getattr(self, name), name))
        if self.region_old == ("",):
            raise ProtocolError("region_old empty text must use the canonical [] representation")
        if any("\n" in line or "\r" in line
               for name in ("prefix", "scope_lines", "suffix_lines", "region_old")
               for line in getattr(self, name)):
            raise ProtocolError("line arrays must contain one LF-free, CR-free wire line per entry")
        if self.scope_mode not in _SCOPES:
            raise ProtocolError(f"unsupported scope_mode: {self.scope_mode!r}")
        object.__setattr__(self, "selected_references", tuple(self.selected_references))
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "retrieval", tuple(self.retrieval))
        for name in ("selected_references", "diagnostics", "retrieval"):
            if any(not isinstance(item, EvidenceRecord) for item in getattr(self, name)):
                raise ProtocolError(f"{name} must contain EvidenceRecord values")
        if any(not isinstance(item, HistoryEvent) for item in self.history):
            raise ProtocolError("history must contain HistoryEvent values")
        if not isinstance(self.cursor, Cursor):
            raise ProtocolError("cursor must be a Cursor")
        if not isinstance(self.replacement_range, ReplacementRange):
            raise ProtocolError("replacement_range must be a ReplacementRange")
        if self.document_eol not in _EOLS:
            raise ProtocolError(f"unsupported document_eol: {self.document_eol!r}")
        if bool(self.region_old) == self.replacement_range.is_empty:
            raise ProtocolError("region_old emptiness must agree with replacement range geometry")
        _validate_cursor(self.cursor, self.region_old)
        _validate_range_geometry(self.replacement_range, self.region_old)

    @classmethod
    def from_dict(cls, value: object) -> PromptContext:
        data = _mapping(value, "context")
        expected = {
            "schema_version", "path", "prefix", "selected_references", "history", "diagnostics",
            "retrieval", "scope_mode", "scope_lines", "suffix_lines", "region_old", "cursor",
            "replacement_range", "document_eol",
        }
        _exact_keys(data, expected, "context")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            path=_string(data["path"], "path"),
            prefix=_lines(data["prefix"], "prefix"),
            selected_references=tuple(EvidenceRecord.from_dict(x, "selected_reference")
                                      for x in _mapping_list(data["selected_references"], "selected_references")),
            history=tuple(HistoryEvent.from_dict(x)
                          for x in _mapping_list(data["history"], "history")),
            diagnostics=tuple(EvidenceRecord.from_dict(x, "diagnostic")
                              for x in _mapping_list(data["diagnostics"], "diagnostics")),
            retrieval=tuple(EvidenceRecord.from_dict(x, "retrieval")
                            for x in _mapping_list(data["retrieval"], "retrieval")),
            scope_mode=_string(data["scope_mode"], "scope_mode"),
            scope_lines=_lines(data["scope_lines"], "scope_lines"),
            suffix_lines=_lines(data["suffix_lines"], "suffix_lines"),
            region_old=_lines(data["region_old"], "region_old"),
            cursor=Cursor.from_dict(data["cursor"]),
            replacement_range=ReplacementRange.from_dict(data["replacement_range"]),
            document_eol=_string(data["document_eol"], "document_eol"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> PromptContext:
        """Named integration alias for JSON-like mapping inputs."""
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "prefix": list(self.prefix),
            "selected_references": [x.to_dict() for x in self.selected_references],
            "history": [x.to_dict() for x in self.history],
            "diagnostics": [x.to_dict() for x in self.diagnostics],
            "retrieval": [x.to_dict() for x in self.retrieval],
            "scope_mode": self.scope_mode,
            "scope_lines": list(self.scope_lines),
            "suffix_lines": list(self.suffix_lines),
            "region_old": list(self.region_old),
            "cursor": self.cursor.to_dict(),
            "replacement_range": self.replacement_range.to_dict(),
            "document_eol": self.document_eol,
        }


def _mapping_list(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ProtocolError(f"{name} must be an array of objects")
    return value


def _render_records(parts: list[str], records: Sequence[EvidenceRecord]) -> None:
    for record in records:
        if record.path is not None:
            parts.append(f"<filename>{record.path}")
        parts.append(record.content)


def render_prompt(context: PromptContext) -> str:
    """Render model-visible text from pre-edit context only.

    There is intentionally no target/region_new argument.  The target is an
    out-of-band label used by :func:`build_training_row`.
    """
    if not isinstance(context, PromptContext):
        raise ProtocolError("render_prompt requires PromptContext")
    parts: list[str] = [TASK_CONTRACT, f"<filename>{context.path}", *context.prefix]
    parts.append("<filename>selected_references")
    _render_records(parts, context.selected_references)
    parts.append("<filename>edit_history")
    for event in context.history:
        parts.append(event.event_diff)
    parts.append("<filename>diagnostics")
    _render_records(parts, context.diagnostics)
    parts.append("<filename>retrieval")
    _render_records(parts, context.retrieval)
    parts.extend(["<[fim-suffix]>", *context.scope_lines, *context.suffix_lines,
                  "<<<<<<< CURRENT"])
    region = list(context.region_old)
    if context.cursor.region_line_index >= 0:
        index = context.cursor.region_line_index
        line = region[index]
        column = context.cursor.code_point_column
        assert column is not None
        region[index] = line[:column] + CURSOR_MARKER + line[column:]
    parts.extend(region)
    parts.extend(["=======", GENERATION_BOUNDARY])
    # The final LF is part of the v1 prompt boundary.  It prevents the BPE
    # encoder from merging the boundary token with the first response token.
    rendered = "\n".join(parts) + "\n"
    _validate_reserved_slots(rendered)
    return rendered


def split_wire_lines(text: str) -> list[str]:
    """Split LF/CRLF transport without trimming source whitespace."""
    _string(text, "wire text", nonempty=False)
    lines = text.split("\n")
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _reserved_body_line(line: str) -> bool:
    return line.strip() in _RESERVED_BODY_LINES


def _target_parts(operation: str, region_new: Sequence[str]) -> tuple[str, str, str]:
    if operation == "no_op":
        if region_new is None:
            raise ProtocolError("no_op requires the unchanged target region as metadata")
        body = NO_EDIT
        terminal_text = "\n" + TERMINAL
    elif operation == "replace":
        if not region_new or tuple(region_new) == ("",):
            raise ProtocolError("replace requires a nonempty canonical region_new")
        body = "\n".join(region_new)
        terminal_text = "\n" + TERMINAL
    elif operation == "delete":
        if region_new:
            raise ProtocolError("delete requires empty region_new")
        body = ""
        terminal_text = TERMINAL
    else:
        raise ProtocolError(f"unknown operation: {operation!r}")
    _validate_reserved_slots(body)
    return body, body + terminal_text, terminal_text


def serialize_target(operation: str, region_new: Sequence[str]) -> str:
    """Serialize a full response body and exact terminal using LF wire bytes."""
    return _target_parts(operation, region_new)[1]


@dataclass(frozen=True)
class OutputResult:
    status: Literal["accepted", "invalid"]
    operation: Literal["no_op", "replace", "delete"] | None
    body: tuple[str, ...]
    body_text: str
    reason: str | None = None


def _invalid(reason: str) -> OutputResult:
    return OutputResult("invalid", None, (), "", reason)


def parse_output(raw_text: str, context: PromptContext) -> OutputResult:
    """Parse exact-line output without a substring stop or source target oracle."""
    if not isinstance(raw_text, str):
        return _invalid("output_not_string")
    if not isinstance(context, PromptContext):
        raise ProtocolError("parse_output requires PromptContext")
    try:
        _validate_reserved_slots(raw_text)
    except ProtocolError as error:
        return _invalid(str(error))
    lines = split_wire_lines(raw_text)
    terminal_indices = [index for index, line in enumerate(lines) if line == TERMINAL]
    if len(terminal_indices) == 0:
        return _invalid("missing_exact_terminal")
    if len(terminal_indices) != 1:
        return _invalid("duplicate_exact_terminal")
    terminal_index = terminal_indices[0]
    if any(line != "" for line in lines[terminal_index + 1:]):
        return _invalid("content_after_terminal")
    body = lines[:terminal_index]
    # Literal marker-looking text is ordinary source under the selected
    # split-special tokenizer mode.  Only an exact terminal line frames the
    # response; quoted or embedded marker strings remain body bytes.
    # A single empty line before the terminal is the canonical wire spelling
    # of an empty body.  Two empty lines remain a one-LF replacement body.
    if body == [""]:
        body = []
    if body != [NO_EDIT] and any(_reserved_body_line(line) for line in body):
        return _invalid("reserved_full_body_line")
    body_tuple = tuple(body)
    if body_tuple == (NO_EDIT,) or body_tuple == context.region_old:
        return OutputResult("accepted", "no_op", body_tuple, "\n".join(body_tuple))
    if not body_tuple:
        operation: Literal["no_op", "replace", "delete"] = (
            "no_op" if context.replacement_range.is_empty else "delete"
        )
        return OutputResult("accepted", operation, body_tuple, "")
    return OutputResult("accepted", "replace", body_tuple, "\n".join(body_tuple))


def check_stale(
    replacement_range: ReplacementRange,
    *,
    active_uri: str,
    active_document_version: int,
    active_content_sha256: str,
) -> None:
    if (
        active_uri != replacement_range.uri
        or active_document_version != replacement_range.document_version
        or active_content_sha256 != replacement_range.content_sha256
    ):
        raise StaleContextError("document URI, version or pre-edit content identity changed")


def map_wire_to_document_eol(text: str, document_eol: str) -> str:
    """Map LF wire payload to LF/CRLF document separators without trimming."""
    _string(text, "payload", nonempty=False)
    if document_eol == "lf":
        return text
    if document_eol == "crlf":
        if "\r" in text:
            raise ProtocolError("wire payload must use LF separators before CRLF mapping")
        return text.replace("\n", "\r\n")
    if document_eol == "mixed":
        raise ProtocolError("mixed document EOL requires an explicit application policy")
    raise ProtocolError(f"unsupported document_eol: {document_eol!r}")


@dataclass(frozen=True)
class ApplicationPlan:
    operation: Literal["no_op", "replace", "delete"]
    mode: Literal["none", "inline", "workspace_edit_user_accept"]
    replacement_range: ReplacementRange
    replacement_text: str
    user_accept_required: bool


def make_application_plan(
    result: OutputResult,
    context: PromptContext,
    *,
    active_uri: str,
    active_document_version: int,
    active_content_sha256: str,
) -> ApplicationPlan:
    if result.status != "accepted" or result.operation is None:
        raise ProtocolError(result.reason or "invalid output")
    check_stale(
        context.replacement_range,
        active_uri=active_uri,
        active_document_version=active_document_version,
        active_content_sha256=active_content_sha256,
    )
    if result.operation == "no_op":
        return ApplicationPlan("no_op", "none", context.replacement_range, "", False)
    replacement = map_wire_to_document_eol(result.body_text, context.document_eol)
    if context.replacement_range.is_same_line:
        return ApplicationPlan(result.operation, "inline", context.replacement_range, replacement, False)
    return ApplicationPlan(result.operation, "workspace_edit_user_accept", context.replacement_range,
                           replacement, True)


class _EncoderAdapter:
    def __init__(self, tokenizer: TokenEncoder) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        _validate_reserved_slots(text)
        try:
            value = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                split_special_tokens=True,
            )
        except TypeError as error:
            raise ProtocolError(
                "tokenizer must support split_special_tokens=True with "
                "add_special_tokens=False"
            ) from error
        ids = getattr(value, "ids", value)
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise ProtocolError("tokenizer.encode must return a sequence of token IDs")
        result = []
        for token in ids:
            if type(token) is not int or not 0 <= token < VOCAB_SIZE:
                raise ProtocolError("tokenizer IDs must be integers within the pinned vocabulary")
            if is_native_control_token(token):
                raise ProtocolError("text encodes a native CONTROL token before manual BOS/EOS")
            result.append(token)
        return result


def encode_prompt(
    context: PromptContext,
    tokenizer: TokenEncoder,
    *,
    include_bos: bool = True,
) -> list[int]:
    """Return prediction-only prompt IDs under the selected tokenizer mode.

    The default result includes the one manually managed BOS ID used by
    training rows. No target or future edit data is accepted by this helper.
    """
    if not isinstance(context, PromptContext):
        raise ProtocolError("encode_prompt requires PromptContext")
    prompt_ids = _EncoderAdapter(tokenizer).encode(render_prompt(context))
    if any(token in (BOS_ID, EOS_ID) for token in prompt_ids):
        raise ProtocolError("prompt text encodes protocol BOS/EOS IDs before manual BOS")
    return [BOS_ID, *prompt_ids] if include_bos else prompt_ids


def _validate_target(context: PromptContext, operation: str, region_new: Sequence[str]) -> None:
    new = tuple(region_new)
    if any(not isinstance(line, str) for line in new):
        raise ProtocolError("region_new must be an array of strings")
    if any("\n" in line or "\r" in line for line in new):
        raise ProtocolError("region_new must contain one LF-free, CR-free wire line per entry")
    if new == ("",):
        raise ProtocolError("empty region_new text must use the canonical [] representation")
    if operation == "no_op" and new != context.region_old:
        raise ProtocolError("no_op target must equal region_old exactly")
    if operation == "delete" and (not context.region_old or new):
        raise ProtocolError("delete requires nonempty region_old and empty region_new")
    if operation == "replace" and (not new or new == context.region_old):
        raise ProtocolError("replace requires a nonempty changed region_new")


ROW_KEYS = frozenset(
    (
        "id", "input_ids", "target_start", "target_body_tokens", "target_terminal_tokens",
        "target_body_token_count", "target_terminal_token_count",
        "family", "package_id", "renderer_id", "prompt_text", "target_text", "target_body_text",
        "target_operation", "prompt_token_count", "target_token_count", "bos_token_id", "eos_token_id",
        "tokenizer_revision", "tokenizer_json_sha256", "tokenization_policy", "split",
    )
)


def _ids(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise ProtocolError(f"{name} must be an array of nonnegative integer IDs")
    return tuple(value)


def validate_training_row(value: object) -> dict[str, Any]:
    data = _mapping(value, "training row")
    _exact_keys(data, set(ROW_KEYS), "training row")
    _string(data["id"], "row.id")
    input_ids = _ids(data["input_ids"], "row.input_ids")
    body_ids = _ids(data["target_body_tokens"], "row.target_body_tokens")
    terminal_ids = _ids(data["target_terminal_tokens"], "row.target_terminal_tokens")
    start = _integer(data["target_start"], "row.target_start")
    body_count = _integer(data["target_body_token_count"], "row.target_body_token_count")
    terminal_count = _integer(data["target_terminal_token_count"], "row.target_terminal_token_count")
    if start > len(input_ids):
        raise ProtocolError("row.target_start exceeds input_ids")
    if tuple(input_ids[start:start + len(body_ids)]) != body_ids:
        raise ProtocolError("target_body_tokens are not at target_start")
    terminal_start = start + len(body_ids)
    if tuple(input_ids[terminal_start:terminal_start + len(terminal_ids)]) != terminal_ids:
        raise ProtocolError("target_terminal_tokens do not follow target_body_tokens")
    if body_count != len(body_ids) or terminal_count != len(terminal_ids):
        raise ProtocolError("row token-array counts mismatch")
    if len(input_ids) != terminal_start + len(terminal_ids) + 1:
        raise ProtocolError("input_ids must end with exactly one protocol EOS")
    if input_ids[-1] != EOS_ID or input_ids.count(EOS_ID) != 1:
        raise ProtocolError("input_ids must contain EOS ID 1 exactly once at the end")
    if input_ids.count(BOS_ID) != 1 or input_ids[0] != BOS_ID:
        raise ProtocolError("input_ids must contain BOS ID 0 exactly once at the start")
    if any(token >= VOCAB_SIZE or is_native_control_token(token) for token in input_ids[1:-1]):
        raise ProtocolError("row body contains an out-of-vocabulary or native CONTROL token")
    for name in ("family", "package_id", "renderer_id", "prompt_text", "target_text",
                 "target_operation", "tokenization_policy", "tokenizer_revision", "tokenizer_json_sha256", "split"):
        _string(data[name], f"row.{name}")
    _string(data["target_body_text"], "row.target_body_text", nonempty=False)
    if data["renderer_id"] != RENDERER_ID or data["tokenization_policy"] != TOKENIZATION_POLICY:
        raise ProtocolError("row renderer or tokenization policy mismatch")
    operation = data["target_operation"]
    if operation not in ("no_op", "replace", "delete"):
        raise ProtocolError("invalid row target_operation")
    if not data["prompt_text"].endswith("\n"):
        raise ProtocolError("row.prompt_text must include the protocol final LF")
    body_text = data["target_body_text"]
    if operation == "no_op":
        expected_body = NO_EDIT
        expected_target = f"{NO_EDIT}\n{TERMINAL}"
    elif operation == "replace":
        if not body_text:
            raise ProtocolError("replace row must carry a nonempty target body")
        expected_body = body_text
        expected_target = f"{body_text}\n{TERMINAL}"
    else:
        expected_body = ""
        expected_target = TERMINAL
    if body_text != expected_body or data["target_text"] != expected_target:
        raise ProtocolError("row full-text target labels do not match target_operation")
    for name in ("prompt_token_count", "target_token_count", "bos_token_id", "eos_token_id"):
        _integer(data[name], f"row.{name}")
    if data["bos_token_id"] != BOS_ID or data["eos_token_id"] != EOS_ID:
        raise ProtocolError("row BOS/EOS IDs mismatch")
    if data["tokenizer_revision"] != TOKENIZER_REVISION or data["tokenizer_json_sha256"] != TOKENIZER_JSON_SHA256:
        raise ProtocolError("row tokenizer identity mismatch")
    if data["prompt_token_count"] != start - 1 or data["target_token_count"] != len(body_ids) + len(terminal_ids):
        raise ProtocolError("row token counts mismatch")
    return dict(data)


def build_training_row(
    context: PromptContext,
    *,
    operation: Literal["no_op", "replace", "delete"],
    region_new: Sequence[str],
    tokenizer: TokenEncoder,
    row_id: str,
    family: str,
    package_id: str,
    split: str = "train",
) -> dict[str, Any]:
    """Build a full-text labelled row with a proven prompt/continuation boundary."""
    _validate_target(context, operation, region_new)
    prompt_text = render_prompt(context)
    body_text, target_text, _terminal_text = _target_parts(operation, region_new)
    parsed_target = parse_output(target_text, context)
    if parsed_target.status != "accepted" or parsed_target.operation != operation:
        raise ProtocolError("serialized target does not round-trip to its declared operation")
    if operation == "replace" and parsed_target.body != tuple(region_new):
        raise ProtocolError("serialized target does not round-trip to the complete region_new")
    if operation == "delete" and parsed_target.body:
        raise ProtocolError("serialized deletion target does not round-trip to an empty body")
    encoder = _EncoderAdapter(tokenizer)
    prompt_ids = encode_prompt(context, tokenizer, include_bos=False)
    # The response terminal is a separately measured suffix.  The tokenizer
    # may merge the body/newline into its first terminal token, so body tokens
    # are the exact continuation prefix after removing the proven terminal
    # suffix from the full target encoding.
    terminal_ids = encoder.encode(TERMINAL)
    target_ids = encoder.encode(target_text)
    joint_ids = encoder.encode(prompt_text + target_text)
    if joint_ids != prompt_ids + target_ids:
        raise TokenBoundaryError(
            "joint encoding retokenizes the prompt/target boundary; row is not admitted"
        )
    if len(target_ids) < len(terminal_ids) or target_ids[-len(terminal_ids):] != terminal_ids:
        raise TokenBoundaryError(
            "target encoding does not end with the exact terminal token suffix; row is not admitted"
        )
    # `target_body_tokens` is the serialized continuation prefix.  It includes
    # the wire LF before the terminal whenever the serialized target has one;
    # `target_body_text` remains the semantic edit body without that framing LF.
    body_ids = target_ids[:-len(terminal_ids)]
    if any(token == EOS_ID for token in prompt_ids + body_ids + terminal_ids):
        raise ProtocolError("literal/source text encodes EOS ID 1 before protocol EOS")
    if any(token == BOS_ID for token in prompt_ids + body_ids + terminal_ids):
        raise ProtocolError("literal/source text encodes BOS ID 0; exactly-one-BOS policy is violated")
    input_ids = [BOS_ID, *prompt_ids, *target_ids, EOS_ID]
    row = {
        "id": _string(row_id, "row_id"),
        "input_ids": input_ids,
        "target_start": 1 + len(prompt_ids),
        "target_body_tokens": body_ids,
        "target_terminal_tokens": terminal_ids,
        "target_body_token_count": len(body_ids),
        "target_terminal_token_count": len(terminal_ids),
        "family": _string(family, "family"),
        "package_id": _string(package_id, "package_id"),
        "renderer_id": RENDERER_ID,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "target_body_text": body_text,
        "target_operation": operation,
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(target_ids),
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "tokenization_policy": TOKENIZATION_POLICY,
        "split": _string(split, "split"),
    }
    validate_training_row(row)
    return row
