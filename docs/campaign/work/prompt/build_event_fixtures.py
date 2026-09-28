#!/usr/bin/env python3
"""Build and validate PRM-01 pre-edit event fixtures.

The fixtures are synthetic functional tests.  They exercise the versioned input
contract without claiming that an editor emitted the events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
PLAN_ROOT = HERE.parents[4]
sys.path.insert(0, str(PLAN_ROOT / "packages" / "sepalith" / "src"))

from sepalith.protocol import (  # noqa: E402
    EditContext,
    EvidenceRecord,
    SourceRange,
    render_context,
    serialize_snapshot,
)

SCHEMA_VERSION = "sepalith.prompt.event-fixtures.v1"
CONTEXT_SCHEMA = "sepalith.edit-context.v1"
RENDERER_VERSION = "zeta2-v1"
PACKAGE_ID = "prm-01.synthetic.editor-episodes.v1"
OUTPUT_TERMINATOR = ">>>>>>> UPDATED"
SNAPSHOT_ROOT = "/mnt/e/sepalith/campaign-20260915/source-snapshots/20260912T005156Z"
TOKENIZER_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain/tokenizer.json"
)
TOKENIZER_SHA256 = (
    "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
)
OUT = HERE.with_name("event-fixtures.json")
REQUIRED_EVENTS = (
    "typing",
    "deletion",
    "cursor_move",
    "history_append",
    "history_eviction",
    "diagnostic_refresh",
    "definition_change",
    "anchor_move",
    "file_switch",
    "no_op",
)
REQUIRED_SCOPES = (
    "long_scope",
    "cross_file",
    "local_only",
    "missing_evidence",
    "no_op",
)


def compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def identity(value: str) -> str:
    return "sha256:" + digest_text(value)


def utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def line_position(lines: list[str], index: int) -> dict[str, Any]:
    if not 0 <= index < len(lines):
        return {
            "line": None,
            "code_point_column": None,
            "utf16_column": None,
            "line_code_points": None,
            "line_utf16_units": None,
        }
    line = lines[index]
    return {
        "line": index,
        "code_point_column": len(line),
        "utf16_column": utf16_units(line),
        "line_code_points": len(line),
        "line_utf16_units": utf16_units(line),
    }


def source_range(content: str, needle: str) -> SourceRange:
    start = content.find(needle)
    if start < 0:
        raise ValueError(f"Evidence fragment is absent: {needle!r}")
    return SourceRange(start, start + len(needle))


def file_entry(path: str, content: str, revision: str) -> dict[str, Any]:
    return {
        "path": path,
        "content": content,
        "content_identity": identity(content),
        "workspace_revision": revision,
        "encoding": "utf-8",
        "newline": "lf",
        "observed_state": "pre_edit",
    }


def diff_text(path: str, old: str, new: str, hunk: str = "@@") -> str:
    fence = chr(96) * 3
    return (
        f'User edited "{path}":\n\n{fence}diff\n{hunk}\n'
        f"-{old}\n+{new}\n{fence}"
    )


def event(
    event_id: str,
    kind: str,
    path: str,
    revision: str,
    old: str,
    new: str,
    hunk: str = "@@",
    detail: str | None = None,
) -> dict[str, Any]:
    result = {
        "event_id": event_id,
        "kind": kind,
        "path": path,
        "workspace_revision": revision,
        "old_text": old,
        "new_text": new,
        "event_diff": diff_text(path, old, new, hunk),
        "observed_event": False,
        "state": "pre_edit",
    }
    if detail is not None:
        result["detail"] = detail
    return result


def evidence(
    source_kind: str,
    content: str,
    *,
    revision: str | None = None,
    path: str | None = None,
    needle: str | None = None,
    symbol: str | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        source_kind=source_kind,
        content=content,
        workspace_revision=revision,
        content_identity=None if content is None else identity(content),
        path=path,
        source_range=None if needle is None else source_range(content, needle),
        symbol=symbol,
        confidence=confidence,
        metadata=metadata or {},
    )


def context_for(
    path: str,
    lines: list[str],
    region_start: int,
    region_end: int,
    cursor_idx: int,
    cursor_column: int | None,
    history_events: list[dict[str, Any]],
    evidence_records: list[EvidenceRecord],
    metadata: dict[str, Any],
) -> EditContext:
    if not 0 <= region_start <= region_end <= len(lines):
        raise ValueError("Invalid context region")
    return EditContext(
        schema_version=CONTEXT_SCHEMA,
        path=path,
        prefix=tuple(lines[:region_start]),
        region_old=tuple(lines[region_start:region_end]),
        suffix=tuple(lines[region_end:]),
        cursor_idx=cursor_idx,
        cursor_column=cursor_column,
        event_diff="\n".join(item["event_diff"] for item in history_events),
        evidence=tuple(evidence_records),
        metadata=metadata,
    )


def geometry(context: EditContext) -> dict[str, Any]:
    region = list(context.region_old)
    return {
        "encoding": "utf-8",
        "cursor_column_unit": "unicode_code_point",
        "region_line_geometry": [
            {
                "line": index,
                "code_points": len(line),
                "utf16_units": utf16_units(line),
            }
            for index, line in enumerate(region)
        ],
        "cursor": {
            "cursor_idx": context.cursor_idx,
            "cursor_column_code_points": context.cursor_column,
            "cursor_column_utf16": (
                None
                if context.cursor_idx < 0 or context.cursor_column is None
                else utf16_units(region[context.cursor_idx][: context.cursor_column])
            ),
            "position": line_position(region, context.cursor_idx),
        },
        "conversion_check": (
            context.cursor_column is None
            or context.cursor_idx < 0
            or context.cursor_column
            == len(region[context.cursor_idx][: context.cursor_column])
        ),
    }


def target_geometry(
    region: list[str], cursor_idx: int, cursor_column: int | None
) -> dict[str, Any]:
    return {
        "cursor_column_unit": "unicode_code_point",
        "region_line_geometry": [
            {
                "line": index,
                "code_points": len(line),
                "utf16_units": utf16_units(line),
            }
            for index, line in enumerate(region)
        ],
        "cursor": {
            "cursor_idx": cursor_idx,
            "cursor_column_code_points": cursor_column,
            "cursor_column_utf16": (
                None
                if cursor_idx < 0 or cursor_column is None
                else utf16_units(region[cursor_idx][:cursor_column])
            ),
            "position": line_position(region, cursor_idx),
        },
    }


def target_oracle(
    context: EditContext,
    region_new: list[str],
    cursor_idx: int,
    cursor_column: int | None,
    operation: str,
) -> dict[str, Any]:
    payload = {
        "operation": operation,
        "path": context.path,
        "region_new": region_new,
        "cursor_idx": cursor_idx,
        "cursor_column": cursor_column,
    }
    return {
        "visibility": "out_of_band",
        "operation": operation,
        "region_new": region_new,
        "cursor_idx": cursor_idx,
        "cursor_column": cursor_column,
        "replacement_text": "\n".join(region_new),
        "output_terminator": OUTPUT_TERMINATOR,
        "target_identity": identity(compact(payload)),
        "target_digest": digest_text(compact(payload)),
        "geometry": target_geometry(region_new, cursor_idx, cursor_column),
    }


def history_state(
    capacity: int,
    retained: list[dict[str, Any]],
    *,
    evicted: list[str] | None = None,
    appended: str | None = None,
) -> dict[str, Any]:
    return {
        "capacity": capacity,
        "order": [item["event_id"] for item in retained],
        "retained_events": retained,
        "evicted_event_ids": evicted or [],
        "appended_event_id": appended,
        "order_semantics": "oldest_to_newest",
        "state": "pre_edit",
    }


def input_collision(input_snapshot: dict[str, Any], target: dict[str, Any]) -> bool:
    serialized = compact(input_snapshot)
    old_region = input_snapshot["edit_context"]["region_old"]
    candidates = [
        line
        for line in target["region_new"]
        if line
        and line.strip()
        and len(line.strip()) >= 8
        and line not in old_region
    ]
    return any(candidate in serialized for candidate in candidates)


def make_fixture(
    *,
    event_id: str,
    event_kind: str,
    path: str,
    revision: str,
    files: dict[str, str],
    lines: list[str],
    region_start: int,
    region_end: int,
    cursor_idx: int,
    cursor_column: int | None,
    target_region: list[str],
    target_cursor_idx: int,
    target_cursor_column: int | None,
    operation: str,
    scope: str,
    trigger: str,
    history_events: list[dict[str, Any]],
    history: dict[str, Any],
    evidence_records: list[EvidenceRecord],
    provider_state: dict[str, Any],
    selection: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    context = context_for(
        path,
        lines,
        region_start,
        region_end,
        cursor_idx,
        cursor_column,
        history_events,
        evidence_records,
        {
            "event_id": event_id,
            "source_package": PACKAGE_ID,
            "input_state": "pre_edit",
            "selection_policy": "bounded_pre_edit_context",
        },
    )
    target = target_oracle(
        context,
        target_region,
        target_cursor_idx,
        target_cursor_column,
        operation,
    )
    pre_files = [
        file_entry(file_path, content, revision)
        for file_path, content in sorted(files.items())
    ]
    input_snapshot = {
        "state": "pre_edit",
        "active_file": path,
        "files": pre_files,
        "edit_context": context.to_dict(),
        "history": history,
        "provider_state": provider_state,
        "selection": selection,
        "target_state": "excluded_from_input",
        "future_intent_state": "excluded_from_input",
    }
    collision = input_collision(input_snapshot, target)
    input_serialized = compact(input_snapshot)
    return {
        "fixture_id": event_id,
        "source_package": PACKAGE_ID,
        "schema_version": SCHEMA_VERSION,
        "event": {
            "kind": event_kind,
            "trigger": trigger,
            "operation": operation,
            "path": path,
            "workspace_revision": revision,
            "state": "pre_edit",
        },
        "scope": {
            "class": scope,
            "labels": [scope, event_kind],
            "long_scope": scope == "long_scope",
            "cross_file": scope == "cross_file",
            "local_only": scope == "local_only",
            "missing_evidence": scope == "missing_evidence",
            "no_op": scope == "no_op",
        },
        "provenance": {
            "kind": "synthetic_functional_test",
            "observed_event": False,
            "observed_real_event": False,
            "source_package": PACKAGE_ID,
            "source_snapshot_root": SNAPSHOT_ROOT,
            "source_basis": "contract-shaped synthetic pre-edit state",
            "synthetic_reason": detail,
        },
        "input_snapshot": input_snapshot,
        "input_identity": {
            "pre_edit_snapshot_sha256": digest_text(input_serialized),
            "active_file_content_identity": identity(files[path]),
            "all_pre_edit_file_identities": {
                file_path: identity(content)
                for file_path, content in sorted(files.items())
            },
        },
        "context": context.to_dict(),
        "context_geometry": geometry(context),
        "target_oracle": target,
        "target_geometry": target["geometry"],
        "audit": {
            "input_construction_function": "context_for(lines, history_events, evidence_records)",
            "target_passed_to_input_constructor": False,
            "target_post_edit_content_in_input": False,
            "future_intent_in_input": False,
            "input_state": "pre_edit",
            "target_line_collision": collision,
            "target_collision_allowed": event_kind in ("cursor_move", "no_op"),
            "pre_edit_and_target_separate": True,
            "target_derived_from_future_state": False,
        },
    }


class PinnedTokenizer:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def probe(self, text: str) -> dict[str, Any]:
        return {
            "add_special_tokens_false": self.tokenizer.encode(
                text, add_special_tokens=False
            ).ids,
            "add_special_tokens_true": self.tokenizer.encode(
                text, add_special_tokens=True
            ).ids,
        }


def measure_fixture(fixture: dict[str, Any], tokenizer: PinnedTokenizer) -> None:
    context = EditContext.from_dict(fixture["context"])
    prompt = render_context(context, renderer=RENDERER_VERSION)
    target = fixture["target_oracle"]
    target_text = "\n".join(target["region_new"]) + "\n" + OUTPUT_TERMINATOR
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text)
    fixture["rendered"] = {
        "renderer": RENDERER_VERSION,
        "prompt": prompt,
        "prompt_sha256": digest_text(prompt),
        "serialized_context": serialize_snapshot(context),
        "prompt_token_count": len(prompt_ids),
        "target_text": target_text,
        "target_sha256": digest_text(target_text),
        "target_token_count": len(target_ids),
        "combined_token_count": len(prompt_ids) + len(target_ids),
        "tokenizer": {
            "path": str(tokenizer.path),
            "tokenizer_json_sha256": digest_file(tokenizer.path),
            "special_tokens": "add_special_tokens=False",
            "literal_probes": {
                literal: tokenizer.probe(literal)
                for literal in (
                "<[fim-middle]>",
                "<[fim-suffix]>",
                "<[fim-prefix]>",
                "<|fim_middle|>",
                "</s>",
                "<|endoftext|>",
                )
            },
        },
    }


def fixture_lines(*items: str) -> list[str]:
    return list(items)


def build_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    p = "R/typing.R"
    lines = fixture_lines(
        "# café summaries",
        "summarize <- function(values) {",
        "  café <- values",
        "  result <- mean(café",
        "  invisible(result)",
        "}",
    )
    h1 = event(
        "PRM01-E01-H01",
        "typing",
        p,
        "synthetic-r01",
        "  café <- values",
        "  café <- values[!is.na(values)]",
        "@@ -3 +3 @@",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E01",
            event_kind="typing",
            path=p,
            revision="synthetic-r01",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=2,
            region_end=5,
            cursor_idx=1,
            cursor_column=len("  result <- mean(café"),
            target_region=[
                "  café <- values[is.finite(values)]",
                "  result <- mean(café)",
                "  invisible(result)",
            ],
            target_cursor_idx=1,
            target_cursor_column=len("  result <- mean(café)"),
            operation="replace_current_region",
            scope="local_only",
            trigger="user typed a closing parenthesis in the active call",
            history_events=[h1],
            history=history_state(4, [h1], appended=h1["event_id"]),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r01",
                    path=p,
                    needle="summarize <- function(values)",
                    symbol="summarize",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["summarize"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor",
                "line": 3,
                "code_point_column": len("  result <- mean(café"),
                "utf16_column": utf16_units("  result <- mean(café"),
            },
            detail="Unicode source and an incomplete call exercise exact replacement geometry.",
        )
    )

    p = "R/cleanup.R"
    lines = fixture_lines(
        "clean_values <- function(values) {",
        "  filtered <- values[values > ",
        "  sum(filtered)",
        "}",
    )
    h1 = event(
        "PRM01-E02-H01",
        "deletion",
        p,
        "synthetic-r02",
        "  filtered <- values[values > 0]",
        "  filtered <- values[values > ",
        "@@ -2 +2 @@",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E02",
            event_kind="deletion",
            path=p,
            revision="synthetic-r02",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=1,
            region_end=3,
            cursor_idx=0,
            cursor_column=len("  filtered <- values[values > "),
            target_region=[
                "  filtered <- values[values > 0 & is.finite(values)]",
                "  sum(filtered)",
            ],
            target_cursor_idx=0,
            target_cursor_column=len(
                "  filtered <- values[values > 0 & is.finite(values)]"
            ),
            operation="replace_current_region",
            scope="local_only",
            trigger="user deleted a predicate suffix before prediction",
            history_events=[h1],
            history=history_state(3, [h1], appended=h1["event_id"]),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r02",
                    path=p,
                    needle="clean_values <- function(values)",
                    symbol="clean_values",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": ["incomplete expression"]},
                "definitions": {"status": "available", "symbols": ["clean_values"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor",
                "line": 1,
                "code_point_column": len("  filtered <- values[values > "),
                "utf16_column": len("  filtered <- values[values > "),
            },
            detail="Deletion leaves a syntactically incomplete predicate while preserving whitespace.",
        )
    )

    p = "R/format.R"
    lines = fixture_lines(
        "format_label <- function(name) {",
        '  paste0("«", name, "»")',
        "}",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E03",
            event_kind="cursor_move",
            path=p,
            revision="synthetic-r03",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=1,
            region_end=2,
            cursor_idx=0,
            cursor_column=len('  paste0("«", '),
            target_region=list(lines[1:2]),
            target_cursor_idx=0,
            target_cursor_column=len('  paste0("«", name'),
            operation="move_cursor",
            scope="local_only",
            trigger="cursor moved within the current Unicode line",
            history_events=[],
            history=history_state(4, [], appended=None),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r03",
                    path=p,
                    needle="paste0",
                    symbol="format_label",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["format_label"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor_move",
                "line": 1,
                "from_code_point_column": len('  paste0("«", '),
                "to_code_point_column": len('  paste0("«", name'),
                "from_utf16_column": utf16_units('  paste0("«", '),
                "to_utf16_column": utf16_units('  paste0("«", name'),
            },
            detail="The cursor crosses a non-ASCII delimiter; code-point and UTF-16 columns are both recorded.",
        )
    )

    p = "R/report.R"
    lines = fixture_lines(
        "report <- function(x) {",
        "  total <- sum(x)",
        "  total",
        "}",
    )
    h1 = event(
        "PRM01-E04-H01",
        "typing",
        p,
        "synthetic-r04",
        "  total <- sum(x)",
        "  total <- sum(x, na.rm = FALSE)",
        "@@ -2 +2 @@",
    )
    h2 = event(
        "PRM01-E04-H02",
        "typing",
        p,
        "synthetic-r04",
        "  total",
        "  total",
        "@@ -3 +3 @@",
        detail="append retained event with unchanged text",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E04",
            event_kind="history_append",
            path=p,
            revision="synthetic-r04",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=1,
            region_end=3,
            cursor_idx=1,
            cursor_column=len("  total"),
            target_region=[
                "  total <- sum(x, na.rm = TRUE)",
                "  total",
            ],
            target_cursor_idx=0,
            target_cursor_column=len("  total <- sum(x, na.rm = TRUE)"),
            operation="replace_current_region",
            scope="local_only",
            trigger="a second edit was appended to bounded edit history",
            history_events=[h1, h2],
            history=history_state(3, [h1, h2], appended=h2["event_id"]),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r04",
                    path=p,
                    needle="report <- function(x)",
                    symbol="report",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["report"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor",
                "line": 2,
                "code_point_column": len("  total"),
                "utf16_column": len("  total"),
            },
            detail="The retained history order proves append semantics before the prediction point.",
        )
    )

    p = "R/pipeline.R"
    lines = fixture_lines(
        "pipeline <- function(data) {",
        *[f"  step_{index:02d} <- data" for index in range(1, 31)],
        "  result <- step_30",
        "  result",
        "}",
    )
    h0 = event(
        "PRM01-E05-H00",
        "typing",
        p,
        "synthetic-r05",
        "  step_01 <- data",
        "  step_01 <- normalize(data)",
        "@@ -2 +2 @@",
    )
    h1 = event(
        "PRM01-E05-H01",
        "typing",
        p,
        "synthetic-r05",
        "  step_10 <- data",
        "  step_10 <- transform(data)",
        "@@ -11 +11 @@",
    )
    h2 = event(
        "PRM01-E05-H02",
        "typing",
        p,
        "synthetic-r05",
        "  step_20 <- data",
        "  step_20 <- validate(data)",
        "@@ -21 +21 @@",
    )
    h3 = event(
        "PRM01-E05-H03",
        "typing",
        p,
        "synthetic-r05",
        "  step_25 <- data",
        "  step_25 <- enrich(data)",
        "@@ -26 +26 @@",
    )
    h4 = event(
        "PRM01-E05-H04",
        "typing",
        p,
        "synthetic-r05",
        "  step_30 <- data",
        "  step_30 <- finalize(data)",
        "@@ -31 +31 @@",
    )
    retained = [h2, h3, h4]
    fixtures.append(
        make_fixture(
            event_id="PRM01-E05",
            event_kind="history_eviction",
            path=p,
            revision="synthetic-r05",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=29,
            region_end=33,
            cursor_idx=2,
            cursor_column=len("  result <- step_30"),
            target_region=[
                "  step_29 <- data",
                "  step_30 <- data",
                "  result <- as.data.frame(step_30)",
                "  result",
            ],
            target_cursor_idx=2,
            target_cursor_column=len("  result <- as.data.frame(step_30)"),
            operation="replace_current_region",
            scope="long_scope",
            trigger="history capacity evicted the oldest edit before prediction",
            history_events=retained,
            history=history_state(
                3,
                retained,
                evicted=[h0["event_id"], h1["event_id"]],
                appended=h4["event_id"],
            ),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r05",
                    path=p,
                    needle="pipeline <- function(data)",
                    symbol="pipeline",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["pipeline"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor",
                "line": 29,
                "code_point_column": len("  result <- step_30"),
                "utf16_column": len("  result <- step_30"),
                "truncation_anchor_line": 16,
            },
            detail="A long function keeps the active suffix while bounded history retains only the newest three events.",
        )
    )

    p = "R/diagnostics.R"
    lines = fixture_lines(
        "validate_input <- function(x) {",
        "  if (is.null(x)) {",
        "    return(NULL)",
        "  }",
        "  length(x)",
        "}",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E06",
            event_kind="diagnostic_refresh",
            path=p,
            revision="synthetic-r06",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=1,
            region_end=5,
            cursor_idx=3,
            cursor_column=len("  length(x)"),
            target_region=[
                "  if (is.null(x)) {",
                "    return(NULL)",
                "  }",
                "  if (length(x) == 0L) return(NULL)",
                "  length(x)",
            ],
            target_cursor_idx=3,
            target_cursor_column=len("  if (length(x) == 0L) return(NULL)"),
            operation="insert_diagnostic_guard",
            scope="missing_evidence",
            trigger="language-server diagnostic refresh timed out",
            history_events=[],
            history=history_state(4, [], appended=None),
            evidence_records=[],
            provider_state={
                "diagnostics": {
                    "status": "missing",
                    "reason": "language_server_timeout",
                    "records": [],
                },
                "definitions": {"status": "missing", "reason": "provider_unavailable"},
                "retrieval": {"status": "missing", "reason": "not_configured"},
            },
            selection={
                "kind": "cursor",
                "line": 4,
                "code_point_column": len("  length(x)"),
                "utf16_column": len("  length(x)"),
            },
            detail="All optional providers are explicitly empty; the fixture tests missing-evidence behavior.",
        )
    )

    model = "R/model.R"
    helpers = "R/helpers.R"
    model_lines = fixture_lines(
        "fit_model <- function(data, weights) {",
        "  normalize_model(data)",
        "}",
    )
    helper_lines = fixture_lines(
        "normalize_model <- function(data) {",
        "  scale(data)",
        "}",
    )
    definition_event = event(
        "PRM01-E07-H01",
        "definition_change",
        helpers,
        "synthetic-r07",
        "normalize_model <- function(data)",
        "normalize_model <- function(data, center = TRUE)",
        "@@ -1 +1 @@",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E07",
            event_kind="definition_change",
            path=model,
            revision="synthetic-r07",
            files={
                model: "\n".join(model_lines) + "\n",
                helpers: "\n".join(helper_lines) + "\n",
            },
            lines=model_lines,
            region_start=1,
            region_end=2,
            cursor_idx=0,
            cursor_column=len("  normalize_model(data)"),
            target_region=["  normalize_model(data, center = TRUE)"],
            target_cursor_idx=0,
            target_cursor_column=len("  normalize_model(data, center = TRUE)"),
            operation="replace_call_using_changed_definition",
            scope="cross_file",
            trigger="referenced helper definition changed in another file",
            history_events=[definition_event],
            history=history_state(
                4, [definition_event], appended=definition_event["event_id"]
            ),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(helper_lines) + "\n",
                    revision="synthetic-r07",
                    path=helpers,
                    needle="normalize_model <- function(data)",
                    symbol="normalize_model",
                    confidence=1.0,
                    metadata={"selection": "referenced_definition"},
                ),
                evidence(
                    "file",
                    "\n".join(model_lines) + "\n",
                    revision="synthetic-r07",
                    path=model,
                    needle="fit_model <- function(data, weights)",
                    symbol="fit_model",
                    confidence=1.0,
                ),
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {
                    "status": "available",
                    "symbols": ["normalize_model"],
                    "changed": True,
                },
                "retrieval": {"status": "available", "records": ["helpers.R"]},
            },
            selection={
                "kind": "cursor",
                "line": 1,
                "code_point_column": len("  normalize_model(data)"),
                "utf16_column": len("  normalize_model(data)"),
            },
            detail="Cross-file evidence carries the referenced definition and its pre-edit identity.",
        )
    )

    p = "R/anchor.R"
    lines = fixture_lines(
        "# long anchored scope",
        "build_sections <- function(x) {",
        *[
            f"  section_{index:02d} <- function(x) {{"
            if index % 5 == 0
            else f"  value_{index:02d} <- x + {index}"
            for index in range(1, 46)
        ],
        "  value_45",
        "}",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E08",
            event_kind="anchor_move",
            path=p,
            revision="synthetic-r08",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=30,
            region_end=31,
            cursor_idx=0,
            cursor_column=len(lines[30]),
            target_region=["  value_30 <- x + 30L"],
            target_cursor_idx=0,
            target_cursor_column=len("  value_30 <- x + 30L"),
            operation="move_truncation_anchor_and_replace",
            scope="long_scope",
            trigger="relevance anchor moved forward after a distant edit",
            history_events=[],
            history=history_state(4, [], appended=None),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r08",
                    path=p,
                    needle="build_sections <- function(x)",
                    symbol="build_sections",
                    confidence=1.0,
                    metadata={"anchor_line": 16},
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["build_sections"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor",
                "line": 31,
                "code_point_column": len(lines[30]),
                "utf16_column": len(lines[30]),
                "truncation_anchor": {"before": 0, "after": 16},
            },
            detail="Long scope records a forward anchor transition while preserving the pre-edit source.",
        )
    )

    analysis = "R/analysis.R"
    utils = "R/utils.R"
    analysis_lines = fixture_lines(
        "normalize <- function(x) {",
        "  x",
        "}",
    )
    utils_lines = fixture_lines(
        "scale_vector <- function(x) {",
        "  x / max(abs(x))",
        "}",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E09",
            event_kind="file_switch",
            path=analysis,
            revision="synthetic-r09",
            files={
                analysis: "\n".join(analysis_lines) + "\n",
                utils: "\n".join(utils_lines) + "\n",
            },
            lines=analysis_lines,
            region_start=1,
            region_end=2,
            cursor_idx=0,
            cursor_column=len("  x"),
            target_region=["  scale_vector(x)"],
            target_cursor_idx=0,
            target_cursor_column=len("  scale_vector(x)"),
            operation="replace_after_switching_active_file",
            scope="cross_file",
            trigger="active editor switched from utils.R to analysis.R",
            history_events=[],
            history=history_state(4, [], appended=None),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(analysis_lines) + "\n",
                    revision="synthetic-r09",
                    path=analysis,
                    needle="normalize <- function(x)",
                    symbol="normalize",
                    confidence=1.0,
                ),
                evidence(
                    "file",
                    "\n".join(utils_lines) + "\n",
                    revision="synthetic-r09",
                    path=utils,
                    needle="scale_vector <- function(x)",
                    symbol="scale_vector",
                    confidence=1.0,
                ),
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["scale_vector"]},
                "retrieval": {"status": "available", "records": ["utils.R"]},
            },
            selection={
                "kind": "file_switch",
                "previous_active_file": utils,
                "active_file": analysis,
                "line": 1,
                "code_point_column": len("  x"),
                "utf16_column": len("  x"),
            },
            detail="Cross-file source remains in the pre-edit workspace while the active path changes.",
        )
    )

    p = "R/noop.R"
    lines = fixture_lines(
        "identity <- function(x) {",
        "  marker <- '<[fim-middle]>'  ",
        '  eos <- "</s>"  ',
        '  literal <- "<|endoftext|>"  ',
        "  x  ",
        "}",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E10",
            event_kind="no_op",
            path=p,
            revision="synthetic-r10",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=1,
            region_end=5,
            cursor_idx=3,
            cursor_column=len("  x  "),
            target_region=list(lines[1:5]),
            target_cursor_idx=3,
            target_cursor_column=len("  x  "),
            operation="no_change",
            scope="no_op",
            trigger="prediction requested with unchanged current region",
            history_events=[],
            history=history_state(4, [], appended=None),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r10",
                    path=p,
                    needle="marker <- '<[fim-middle]>'",
                    symbol="identity",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": []},
                "definitions": {"status": "available", "symbols": ["identity"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "cursor",
                "line": 4,
                "code_point_column": len("  x  "),
                "utf16_column": len("  x  "),
            },
            detail="Quoted renderer markers, literal tokenizer EOS text, and trailing spaces test no-op boundary preservation.",
        )
    )
    fixtures[-1]["boundary_case"] = {
        "class": "quoted_marker_like_source",
        "source_literals": ["<[fim-middle]>", "</s>", "<|endoftext|>"],
        "admission": "preserve_as_source_text",
        "full_delimiter_line": "reject_as_ambiguous_until_output_parser_contract_is_frozen",
    }

    p = "R/empty-region.R"
    lines = fixture_lines(
        "make_label <- function() {",
        "  ",
        "}",
    )
    h1 = event(
        "PRM01-E11-H01",
        "typing",
        p,
        "synthetic-r11",
        "make_label <- function() {",
        "make_label <- function() {\n",
        "@@ -1 +1 @@",
    )
    fixtures.append(
        make_fixture(
            event_id="PRM01-E11",
            event_kind="typing",
            path=p,
            revision="synthetic-r11",
            files={p: "\n".join(lines) + "\n"},
            lines=lines,
            region_start=1,
            region_end=2,
            cursor_idx=-1,
            cursor_column=None,
            target_region=['  paste0("label-", Sys.Date())'],
            target_cursor_idx=0,
            target_cursor_column=len('  paste0("label-", Sys.Date())'),
            operation="replace_empty_region",
            scope="local_only",
            trigger="user typed into an empty whitespace-only replacement region",
            history_events=[h1],
            history=history_state(4, [h1], appended=h1["event_id"]),
            evidence_records=[
                evidence(
                    "file",
                    "\n".join(lines) + "\n",
                    revision="synthetic-r11",
                    path=p,
                    needle="make_label <- function()",
                    symbol="make_label",
                    confidence=1.0,
                )
            ],
            provider_state={
                "diagnostics": {"status": "fresh", "records": ["empty body"]},
                "definitions": {"status": "available", "symbols": ["make_label"]},
                "retrieval": {"status": "not_requested", "records": []},
            },
            selection={
                "kind": "empty_region",
                "line": 2,
                "code_point_column": 2,
                "utf16_column": 2,
            },
            detail="Whitespace-only region and cursor_idx=-1 preserve the explicit empty-selection convention.",
        )
    )
    return fixtures


def build_package(tokenizer_path: Path) -> dict[str, Any]:
    tokenizer = PinnedTokenizer(tokenizer_path)
    fixtures = build_fixtures()
    for fixture in fixtures:
        measure_fixture(fixture, tokenizer)
    event_counts = {
        kind: sum(item["event"]["kind"] == kind for item in fixtures)
        for kind in REQUIRED_EVENTS
    }
    scope_counts = {
        scope: sum(item["scope"]["class"] == scope for item in fixtures)
        for scope in REQUIRED_SCOPES
    }
    evidence_counts = {
        "with_evidence": sum(bool(item["context"]["evidence"]) for item in fixtures),
        "without_evidence": sum(not item["context"]["evidence"] for item in fixtures),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_package": PACKAGE_ID,
        "package_identity": {
            "stable_id": PACKAGE_ID,
            "source_script": str(HERE),
            "source_script_sha256": digest_file(HERE),
            "source_kind": "synthetic_functional_test",
            "observed_event": False,
        },
        "source_snapshot": {
            "root": SNAPSHOT_ROOT,
            "snapshot_id": "20260912T005156Z",
            "source_roles": {
                "plan": "plan worktree and contracts",
                "canonical": "canonical assets and environment authority",
                "owner": "active experiment source read-only",
                "execution": "execution conversation source read-only",
            },
            "real_event_trace_status": "not_located_in_bounded_audited_paths",
        },
        "basis": [
            {
                "path": "docs/PROMPT-CONTRACT.md",
                "sha256": "c5017c368143228baff170fff6bfa848d12f2a06a3a5b175257881393c3c9109",
                "role": "input and renderer contract",
            },
            {
                "path": "docs/research/72h-prompt-training-contract.md",
                "sha256": "9e13ce23792e98d638597f93a1368d2b95a2563fc55166b49f5c9ea6472e5241",
                "role": "pre-SFT event and boundary requirements",
            },
            {
                "path": "canonical/extensions/vscode-sepalith/src/extension.ts",
                "sha256": "07855573d974e76582acef99c7642b26dd8b3e4b6cbbd96126ad59107fe1cec8",
                "role": "canonical editor behavior snapshot",
            },
            {
                "path": "canonical/extensions/vscode-sepalith/src/context_build.ts",
                "sha256": "f0dffa8b4ae866c63f25e69cee16e1fa6c82f629b5104c0a18d26c93bcfd837f",
                "role": "canonical context geometry snapshot",
            },
        ],
        "tokenizer": {
            "path": str(tokenizer_path),
            "tokenizer_json_sha256": digest_file(tokenizer_path),
            "expected_sha256": TOKENIZER_SHA256,
            "revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
            "special_tokens": "add_special_tokens=False",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 130560,
        },
        "policy": {
            "pre_edit_only": True,
            "target_visibility": "out_of_band",
            "history_order": "oldest_to_newest",
            "history_input_is_bounded": True,
            "missing_provider_sections_are_empty": True,
            "cursor_columns_are_unicode_code_points": True,
            "utf16_conversion_recorded": True,
            "target_not_used_to_construct_input": True,
            "synthetic_functional_tests_are_not_observed_coverage": True,
        },
        "literal_boundary_policy": {
            "quoted_or_comment_marker_like_text": "admit_and_preserve",
            "tokenizer_eos_or_special_text": "record_probe_and_preserve_source_text",
            "full_delimiter_line": "reject_as_ambiguous_until_output_parser_contract_is_frozen",
            "no_silent_plain_substring_stop": True,
        },
        "coverage": {
            "fixture_count": len(fixtures),
            "observed_real_event_fixtures": 0,
            "synthetic_functional_fixtures": len(fixtures),
            "event_classes": event_counts,
            "scope_classes": scope_counts,
            "evidence_cases": evidence_counts,
            "long_scope_fixtures": [item["fixture_id"] for item in fixtures if item["scope"]["long_scope"]],
            "cross_file_fixtures": [item["fixture_id"] for item in fixtures if item["scope"]["cross_file"]],
            "local_only_fixtures": [item["fixture_id"] for item in fixtures if item["scope"]["local_only"]],
            "missing_evidence_fixtures": [
                item["fixture_id"] for item in fixtures if item["scope"]["missing_evidence"]
            ],
            "no_op_fixtures": [item["fixture_id"] for item in fixtures if item["scope"]["no_op"]],
        },
        "availability_limit": {
            "real_editor_events": "No approved event-log or document-change trace was located in the bounded audited paths for this packet.",
            "excluded_rows": [
                "Git/commit examples include post-edit state and are not observed pre-edit episodes.",
                "Synthetic scenario rows include targets and do not prove editor-event coverage.",
            ],
            "promotion_rule": "Keep all 11 rows labelled synthetic_functional_test until an approved event trace is captured.",
        },
        "fixtures": fixtures,
        "final_set_exclusion": {
            "contains_final_test_rows": False,
            "source_selection": "non-final contract fixtures only",
            "target_set_lock": "No final-test target or post-edit row was read for construction.",
        },
        "isolation": {
            "input_keys": [
                "active_file",
                "files",
                "edit_context",
                "history",
                "provider_state",
                "selection",
                "state",
                "target_state",
                "future_intent_state",
            ],
            "target_keys": [
                "region_new",
                "cursor_idx",
                "cursor_column",
                "replacement_text",
                "target_identity",
                "target_digest",
            ],
            "input_constructor_has_no_target_parameter": True,
            "target_is_out_of_band_oracle": True,
        },
    }


def validate_package(package: dict[str, Any], tokenizer_path: Path) -> None:
    assert package["schema_version"] == SCHEMA_VERSION
    assert package["fixture_package"] == PACKAGE_ID
    assert package["package_identity"]["observed_event"] is False
    assert package["tokenizer"]["tokenizer_json_sha256"] == TOKENIZER_SHA256
    assert digest_file(tokenizer_path) == TOKENIZER_SHA256
    fixtures = package["fixtures"]
    assert len(fixtures) == 11
    assert len({item["fixture_id"] for item in fixtures}) == len(fixtures)
    assert {item["event"]["kind"] for item in fixtures} == set(REQUIRED_EVENTS)
    assert {item["scope"]["class"] for item in fixtures} >= set(REQUIRED_SCOPES)
    for fixture in fixtures:
        assert fixture["source_package"] == PACKAGE_ID
        assert fixture["provenance"]["kind"] == "synthetic_functional_test"
        assert fixture["provenance"]["observed_event"] is False
        assert fixture["audit"]["input_state"] == "pre_edit"
        assert fixture["audit"]["target_passed_to_input_constructor"] is False
        assert fixture["audit"]["pre_edit_and_target_separate"] is True
        assert fixture["audit"]["future_intent_in_input"] is False
        assert fixture["audit"]["target_post_edit_content_in_input"] is False
        assert fixture["context"]["schema_version"] == CONTEXT_SCHEMA
        context = EditContext.from_dict(fixture["context"])
        assert fixture["context"]["path"] == fixture["event"]["path"]
        assert fixture["rendered"]["renderer"] == RENDERER_VERSION
        assert fixture["rendered"]["serialized_context"] == serialize_snapshot(context)
        assert fixture["rendered"]["prompt_sha256"] == digest_text(
            fixture["rendered"]["prompt"]
        )
        assert fixture["rendered"]["tokenizer"]["tokenizer_json_sha256"] == TOKENIZER_SHA256
        assert "<|endoftext|>" in fixture["rendered"]["tokenizer"]["literal_probes"]
        assert fixture["rendered"]["prompt_token_count"] > 0
        assert fixture["rendered"]["target_token_count"] > 0
        assert fixture["input_identity"]["pre_edit_snapshot_sha256"] == digest_text(
            compact(fixture["input_snapshot"])
        )
        files = {
            item["path"]: item
            for item in fixture["input_snapshot"]["files"]
        }
        assert fixture["event"]["path"] in files
        for item in files.values():
            assert item["observed_state"] == "pre_edit"
            assert item["content_identity"] == identity(item["content"])
        history = fixture["input_snapshot"]["history"]
        assert history["state"] == "pre_edit"
        assert history["order"] == [
            item["event_id"] for item in history["retained_events"]
        ]
        assert len(history["order"]) <= history["capacity"]
        assert history["appended_event_id"] in (None, *history["order"])
        assert [item["event_id"] for item in history["retained_events"]] == [
            item["event_id"] for item in fixture["input_snapshot"]["history"]["retained_events"]
        ]
        assert fixture["context"]["event_diff"] == "\n".join(
            item["event_diff"] for item in history["retained_events"]
        )
        if fixture["event"]["kind"] == "history_eviction":
            assert history["evicted_event_ids"]
            assert history["appended_event_id"] == history["order"][-1]
        if fixture["event"]["kind"] == "history_append":
            assert history["appended_event_id"] == history["order"][-1]
        if fixture["event"]["kind"] == "diagnostic_refresh":
            assert not context.evidence
            assert fixture["input_snapshot"]["provider_state"]["diagnostics"]["status"] == "missing"
        for record in context.evidence:
            if record.source_range is not None:
                assert 0 <= record.source_range.start <= record.source_range.end <= len(record.content)
        target = fixture["target_oracle"]
        assert target["visibility"] == "out_of_band"
        assert target["target_digest"] == digest_text(
            compact(
                {
                    "operation": target["operation"],
                    "path": context.path,
                    "region_new": target["region_new"],
                    "cursor_idx": target["cursor_idx"],
                    "cursor_column": target["cursor_column"],
                }
            )
        )
        if fixture["event"]["kind"] not in ("cursor_move", "no_op"):
            assert fixture["audit"]["target_line_collision"] is False
        if fixture["event"]["kind"] == "no_op":
            assert target["region_new"] == list(context.region_old)
            assert fixture["boundary_case"]["class"] == "quoted_marker_like_source"
            assert "<[fim-middle]>" in fixture["boundary_case"]["source_literals"]
            assert "</s>" in fixture["boundary_case"]["source_literals"]
            assert "<|endoftext|>" in fixture["boundary_case"]["source_literals"]
        assert fixture["context_geometry"]["cursor"]["cursor_idx"] == context.cursor_idx
        if context.cursor_column is not None:
            assert context.cursor_idx >= 0
            assert context.cursor_column <= len(context.region_old[context.cursor_idx])
    assert package["coverage"]["observed_real_event_fixtures"] == 0
    assert package["coverage"]["synthetic_functional_fixtures"] == len(fixtures)
    assert package["final_set_exclusion"]["contains_final_test_rows"] is False
    assert package["literal_boundary_policy"]["no_silent_plain_substring_stop"] is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("PRM01_TOKENIZER", str(TOKENIZER_DEFAULT)),
    )
    args = parser.parse_args()
    if not args.write and not args.check:
        parser.error("choose --write or --check")
    tokenizer_path = Path(args.tokenizer)
    if digest_file(tokenizer_path) != TOKENIZER_SHA256:
        raise SystemExit("Pinned tokenizer SHA256 mismatch")
    if args.write:
        package = build_package(tokenizer_path)
        validate_package(package, tokenizer_path)
        OUT.write_text(compact(package) + "\n", encoding="utf-8")
        print(f"wrote {OUT} ({len(package['fixtures'])} fixtures)")
    if args.check:
        package = json.loads(OUT.read_text(encoding="utf-8"))
        validate_package(package, tokenizer_path)
        print(
            f"validated {OUT}: {len(package['fixtures'])} fixtures; "
            f"renderer={RENDERER_VERSION}; observed_real=0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
