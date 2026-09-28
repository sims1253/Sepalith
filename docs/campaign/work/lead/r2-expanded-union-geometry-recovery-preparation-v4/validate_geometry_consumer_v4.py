#!/usr/bin/env python3
"""Consumer-side checks for the v4 geometry record.

The consumer checks typed source identity, anchored cursor evidence, and that
the cursor lies in the selected replacement range.  It deliberately does not
require cursor == replacement_range.start: a nonzero original replacement can
contain an independently recorded cursor.  A synthetic builder locator is
preserved as a typed locator and is not converted to an invented filesystem
path.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


GEOMETRY = "sepalith.source-cursor-geometry.v1"
CURSOR_EVIDENCE = "sepalith.source-cursor-evidence.v1"


class GeometryConsumerError(RuntimeError):
    pass


def require(value: Any, message: str) -> None:
    if not value:
        raise GeometryConsumerError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def position(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict) and set(value) == {"line", "character"}, label)
    require(type(value["line"]) is int and value["line"] >= 0, label + ":line")
    require(type(value["character"]) is int and value["character"] >= 0, label + ":character")
    return {"line": value["line"], "character": value["character"]}


def before_or_equal(left: dict[str, int], right: dict[str, int]) -> bool:
    return (left["line"], left["character"]) <= (right["line"], right["character"])


def source_kind(source_path: Any) -> str:
    require(isinstance(source_path, str) and source_path, "source_path")
    if source_path.startswith("/"):
        return "file"
    if source_path.startswith("builder:synthetic/"):
        return "synthetic"
    raise GeometryConsumerError("unsupported_source_locator")


def validate(record: dict[str, Any]) -> dict[str, Any]:
    geometry = record.get("source_cursor_geometry")
    require(isinstance(geometry, dict) and geometry.get("schema") == GEOMETRY, "geometry_schema")
    source_path = geometry.get("source_path")
    source_sha = geometry.get("source_sha256")
    preedit_sha = geometry.get("preedit_sha256")
    window_sha = geometry.get("window_sha256")
    kind = source_kind(source_path)
    require(record.get("source_kind") == kind, "source_kind_join")
    for label, value in (("source_sha256", source_sha), ("preedit_sha256", preedit_sha), ("window_sha256", window_sha)):
        require(isinstance(value, str) and len(value) == 64, label)
    identity = record.get("source_identity")
    require(isinstance(identity, dict), "source_identity")
    if identity.get("source_path") is not None:
        require(identity.get("source_path") == source_path, "source_path_join")
        require(identity.get("source_sha256") == source_sha, "source_sha_join")
        if kind == "synthetic":
            require(identity.get("source_snapshot_is_simulated") is True, "synthetic_identity_marker")
    else:
        nested = identity.get("source_provenance")
        require(isinstance(nested, dict), "source_identity_layout")
        require(nested.get("source_snapshot_path") == source_path, "nested_source_path_join")
        require(nested.get("source_snapshot_sha256") == source_sha, "nested_source_sha_join")
        require(nested.get("after_snapshot_sha256") == preedit_sha, "nested_preedit_join")
        require(nested.get("source_snapshot_is_simulated") is (kind == "synthetic"), "nested_source_kind_marker")
    cursor = position(geometry.get("cursor"), "cursor")
    replacement = geometry.get("replacement_range")
    require(isinstance(replacement, dict) and set(replacement) == {"start", "end"}, "replacement_range")
    start = position(replacement["start"], "replacement_start")
    end = position(replacement["end"], "replacement_end")
    require(before_or_equal(start, end), "replacement_range_reversed")
    require(before_or_equal(start, cursor) and before_or_equal(cursor, end), "cursor_outside_replacement_range")
    supplied = record.get("source_cursor_geometry_sha256")
    require(isinstance(supplied, str) and supplied == digest_text(canonical(geometry)), "geometry_digest")

    evidence = record.get("cursor_evidence")
    require(isinstance(evidence, dict) and evidence.get("schema") == CURSOR_EVIDENCE, "cursor_evidence")
    require(evidence.get("absolute_position") == cursor, "cursor_evidence_absolute_join")
    basis = evidence.get("basis")
    if basis == "explicit_region_cursor":
        index = evidence.get("region_line_index")
        code_point = evidence.get("code_point_column")
        utf16_column = evidence.get("utf16_column")
        prefix_lines = evidence.get("prefix_line_count")
        anchor = position(evidence.get("anchor_position"), "cursor_evidence_anchor")
        region_code_points = evidence.get("region_line_code_points")
        region_utf16_units = evidence.get("region_line_utf16_units")
        require(type(index) is int and index >= 0, "cursor_evidence_region_line_index")
        require(type(code_point) is int and code_point >= 0, "cursor_evidence_code_point_column")
        require(type(utf16_column) is int and utf16_column >= 0, "cursor_evidence_utf16_column")
        require(type(prefix_lines) is int and prefix_lines >= 0, "cursor_evidence_prefix_lines")
        require(type(region_code_points) is int and code_point <= region_code_points, "cursor_evidence_code_point_bounds")
        require(type(region_utf16_units) is int and utf16_column <= region_utf16_units, "cursor_evidence_utf16_bounds")
        require(cursor == {"line": anchor["line"] + index, "character": utf16_column}, "cursor_context_join")
    elif basis == "zero_width_replacement_range_start_sentinel":
        require(start == end and cursor == start, "cursor_sentinel_join")
        require(position(evidence.get("anchor_position"), "cursor_sentinel_anchor") == start, "cursor_sentinel_anchor_join")
        require(evidence.get("region_line_index") == -1, "cursor_sentinel_index")
        require(evidence.get("code_point_column") is None, "cursor_sentinel_code_point")
        require(evidence.get("utf16_column") is None, "cursor_sentinel_utf16")
    else:
        raise GeometryConsumerError("cursor_evidence_basis")
    return geometry
