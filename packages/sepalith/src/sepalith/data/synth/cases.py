"""Raw case records in the pipeline's PRM03 context schema.

A case is the pre-edit document (as lines), a replacement region, the target
region and optional history events. ``to_context`` produces exactly the
``context`` object of the DAT10 context sidecar (``PromptContext.to_dict``),
so rows render through ``campaign_protocol.build_training_row`` unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

from sepalith.campaign_protocol import SCHEMA_VERSION, PromptContext, codepoint_to_utf16_column, utf16_length

URI_ROOT = "file:///synth-v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_text(lines: Sequence[str]) -> str:
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Event:
    """One observed edit before the current document state (0-based lines)."""

    path: str
    before_lines: tuple[str, ...]  # whole pre-event document
    start: int  # first replaced line, 0-based, in the pre-event document
    old: tuple[str, ...]
    new: tuple[str, ...]

    def event_diff(self) -> str:
        lineno = self.start + 1
        if len(self.old) == 1 and len(self.new) == 1:
            header = f"@@ -{lineno} +{lineno} @@"
        else:
            header = f"@@ -{lineno},{len(self.old)} +{lineno},{len(self.new)} @@"
        body = "".join(f"-{line}\n" for line in self.old) + "".join(f"+{line}\n" for line in self.new)
        return f'User edited "{self.path}":\n\n```diff\n{header}\n{body}```'

    def to_dict(self, uri: str) -> dict[str, Any]:
        revision = sha256_text(document_text(self.before_lines))
        end_line = self.start + len(self.old) - 1
        return {
            "event_diff": self.event_diff(),
            "event_id": hashlib.md5((revision + self.event_diff()).encode()).hexdigest(),
            "kind": "observed_edit_diff",
            "new_text": "\n".join(self.new),
            "old_text": "\n".join(self.old),
            "path": self.path,
            "range_utf16": {
                "content_sha256": revision,
                "document_version": 0,
                "end": {"character": utf16_length(self.old[-1]), "line": end_line},
                "start": {"character": 0, "line": self.start},
                "uri": uri,
            },
            "workspace_revision_before": revision,
        }


@dataclass
class Case:
    family: str
    package_id: str
    path: str  # model-visible relative path
    doc: tuple[str, ...]  # current (pre-edit) document lines
    region_start: int  # 0-based
    region_end: int  # exclusive; == region_start for an insertion
    target: tuple[str, ...]
    window_start: int = 0
    window_end: int | None = None  # exclusive line bound of prefix+region+suffix
    cursor: tuple[int, int] | None = None  # (region line index, code-point column)
    events: tuple[Event, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def region_old(self) -> tuple[str, ...]:
        return self.doc[self.region_start:self.region_end]

    def uri(self) -> str:
        return f"{URI_ROOT}/{self.package_id.split(':', 1)[-1]}/{self.path}"

    def to_context(self) -> dict[str, Any]:
        end = len(self.doc) if self.window_end is None else self.window_end
        region = list(self.region_old)
        content = sha256_text(document_text(self.doc))
        if region:
            start = {"character": 0, "line": self.region_start}
            stop = {"character": utf16_length(region[-1]), "line": self.region_end - 1}
            index, column = self.cursor if self.cursor is not None else (0, 0)
            cursor = {"code_point_column": column, "region_line_index": index,
                      "utf16_column": codepoint_to_utf16_column(region[index], column)}
        else:
            start = stop = {"character": 0, "line": self.region_start}
            cursor = {"code_point_column": None, "region_line_index": -1, "utf16_column": None}
        context = {
            "cursor": cursor,
            "diagnostics": [],
            "document_eol": "lf",
            "history": [event.to_dict(self.uri()) for event in self.events],
            "path": self.path,
            "prefix": list(self.doc[self.window_start:self.region_start]),
            "region_old": region,
            "replacement_range": {"content_sha256": content, "document_version": 0, "end": stop,
                                  "start": start, "uri": self.uri()},
            "retrieval": [],
            "schema_version": SCHEMA_VERSION,
            "scope_lines": [],
            "scope_mode": "off",
            "selected_references": [],
            "suffix_lines": list(self.doc[self.region_end:end]),
        }
        PromptContext.from_dict(context)  # fail closed on any geometry error
        return context

    def after_doc(self) -> tuple[str, ...]:
        return (*self.doc[:self.region_start], *self.target, *self.doc[self.region_end:])

    def window_after(self) -> tuple[str, ...]:
        end = len(self.doc) if self.window_end is None else self.window_end
        return (*self.doc[self.window_start:self.region_start], *self.target, *self.doc[self.region_end:end])

    def window_before(self) -> tuple[str, ...]:
        end = len(self.doc) if self.window_end is None else self.window_end
        return tuple(self.doc[self.window_start:end])

    def key(self) -> str:
        """Stable identity of the case content (used for row ids)."""
        payload = "\x1e".join([self.family, self.package_id, self.path, document_text(self.doc),
                               str(self.region_start), str(self.region_end), "\n".join(self.target),
                               str(self.window_start), str(self.window_end), str(self.cursor),
                               *[e.event_diff() for e in self.events]])
        return sha256_text(payload)

    def row_id(self) -> str:
        return sha256_text("synth-v1:" + self.key())[:24]


def window_start_for(lines: Sequence[str], region_start: int, budget: int, safe: set[int], *,
                     max_lines: int | None = None) -> int:
    """Earliest 0-based start within ``budget`` UTF-16 units (and ``max_lines``)
    that is a safe line (outside every top-level expression), else region_start."""
    used = 0
    best = region_start
    for index in range(region_start - 1, -1, -1):
        used += utf16_length(lines[index]) + 1
        if used > budget or (max_lines is not None and region_start - index > max_lines):
            break
        if index in safe:
            best = index
    return skip_blank(lines, best, region_start)


def skip_blank(lines: Sequence[str], start: int, limit: int) -> int:
    """Advance a window start past blank lines (air drops leading blank lines)."""
    while start < limit and not lines[start].strip():
        start += 1
    return start
