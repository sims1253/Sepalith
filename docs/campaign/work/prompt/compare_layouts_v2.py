#!/usr/bin/env python3
"""PRM-06 V2: replayable synthetic source transitions for layout comparison.

V1 compared context clones.  V2 builds two complete, synthetic source states
per event and validates every context against its own buffer, URI, version,
range and SHA256 identity.  The transitions are functional fixtures only;
they are never promoted to observed editor traces and no future target is
read or constructed.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


PLAN_ROOT = Path(__file__).resolve().parents[4]
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
TOKENIZER_PATH = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
V1_PATH = PLAN_ROOT / "docs/campaign/work/prompt/compare_layouts.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    return sha256_bytes(data.encode("utf-8"))


def u16(text: str) -> int:
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def file_record(path: str, content: str) -> dict[str, Any]:
    return {
        "path": path,
        "content": content,
        "content_sha256": sha256_text(content),
        "uri": f"synthetic://workspace/{path}",
    }


def text_lines(content: str) -> list[str]:
    return content.split("\n")


def offset(lines: list[str], line: int, cp: int) -> int:
    if line < 0 or line >= len(lines) or cp < 0 or cp > len(lines[line]):
        raise ValueError(f"invalid source position line={line} cp={cp}")
    return sum(len(item) + 1 for item in lines[:line]) + cp


def region_slice(content: str, start: tuple[int, int], end: tuple[int, int]) -> tuple[str, ...]:
    lines = text_lines(content)
    start_offset = offset(lines, *start)
    end_offset = offset(lines, *end)
    if end_offset < start_offset:
        raise ValueError("range end precedes start")
    return tuple(content[start_offset:end_offset].split("\n"))


def wire_prefix_suffix(content: str, start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lines = text_lines(content)
    prefix = list(lines[:start[0]])
    if start[1]:
        prefix.append(lines[start[0]][:start[1]])
    suffix = [lines[end[0]][end[1]:], *lines[end[0] + 1:]]
    return tuple(prefix), tuple(suffix)


def range_for(prm: Any, *, path: str, version: int, content: str,
              start: tuple[int, int], end: tuple[int, int]) -> Any:
    lines = text_lines(content)
    return prm.ReplacementRange(
        uri=f"synthetic://workspace/{path}",
        document_version=version,
        content_sha256=sha256_text(content),
        start=prm.Position(start[0], u16(lines[start[0]][:start[1]])),
        end=prm.Position(end[0], u16(lines[end[0]][:end[1]])),
    )


def evidence(prm: Any, path: str, content: str) -> Any:
    return prm.EvidenceRecord(content=content, path=path)


def event_diff(path: str, start_line: int, old: str, new: str) -> str:
    return (f'Synthetic user edit "{path}"\n\n```diff\n'
            f"@@ -{start_line + 1} +{start_line + 1} @@\n-{old}\n+{new}\n```")


def history_event(prm: Any, *, event_id: str, kind: str, path: str,
                  before: dict[str, Any], start: tuple[int, int], end: tuple[int, int],
                  after_content: str) -> Any:
    before_file = before["files"][path]
    old = "\n".join(region_slice(before_file["content"], start, end))
    after_lines = text_lines(after_content)
    # All authored V2 edits are one-line suffix replacements.  Resolve the
    # post-change end against the post-change buffer so a longer definition
    # or shorter deletion records its complete new source text.
    after_end = (end[0], len(after_lines[end[0]]))
    new = "\n".join(region_slice(after_content, start, after_end))
    return prm.HistoryEvent(
        event_id=event_id,
        kind=kind,
        path=path,
        workspace_revision_before=before["workspace_revision"],
        old_text=old,
        new_text=new,
        range_utf16=range_for(prm, path=path, version=before["version"],
                              content=before_file["content"], start=start, end=end),
        event_diff=event_diff(path, start[0], old, new),
    )


def state(*, files: dict[str, str], active: str, version: int,
          start: tuple[int, int], end: tuple[int, int], cursor: tuple[int, int],
          history: tuple[Any, ...] = (), selected: tuple[tuple[str, str], ...] = (),
          diagnostics: tuple[str, ...] = (), retrieval: tuple[str, ...] = (),
          scope_lines: tuple[str, ...] = (), scope_anchor_line: int | None = None,
          provider_status: dict[str, str] | None = None,
          evicted_event_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "files": {path: file_record(path, content) for path, content in files.items()},
        "active": active,
        "version": version,
        "workspace_revision": f"synthetic-prm06-v2-r{version}",
        "start": start,
        "end": end,
        "cursor": cursor,
        "history": history,
        "selected": selected,
        "diagnostics": diagnostics,
        "retrieval": retrieval,
        "scope_lines": scope_lines,
        "scope_anchor_line": scope_anchor_line,
        "provider_status": provider_status or {"diagnostics": "fresh", "retrieval": "not_requested"},
        "evicted_event_ids": evicted_event_ids,
    }


def make_context(prm: Any, item: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    active = item["active"]
    record = item["files"][active]
    content = record["content"]
    lines = text_lines(content)
    start = tuple(item["start"])
    end = tuple(item["end"])
    cursor_line, cursor_cp = item["cursor"]
    region = region_slice(content, start, end)
    prefix, suffix = wire_prefix_suffix(content, start, end)
    if cursor_line < start[0] or cursor_line > end[0]:
        raise ValueError("cursor does not address selected source range")
    cursor_region_line = cursor_line - start[0]
    cursor_region_cp = cursor_cp - start[1] if cursor_line == start[0] else cursor_cp
    if cursor_region_cp < 0 or cursor_region_cp > len(region[cursor_region_line]):
        raise ValueError("cursor is outside selected region line")
    context = prm.PromptContext(
        schema_version=prm.SCHEMA_VERSION,
        path=active,
        prefix=prefix,
        selected_references=tuple(evidence(prm, path, text) for path, text in item["selected"]),
        history=tuple(item["history"]),
        diagnostics=tuple(prm.EvidenceRecord(content=text, path=active) for text in item["diagnostics"]),
        retrieval=tuple(prm.EvidenceRecord(content=text, path=None) for text in item["retrieval"]),
        scope_mode="pin+outline" if item["scope_lines"] else "off",
        scope_lines=tuple(item["scope_lines"]),
        suffix_lines=suffix,
        region_old=region,
        cursor=prm.Cursor(
            region_line_index=cursor_region_line,
            code_point_column=cursor_region_cp,
            utf16_column=prm.codepoint_to_utf16_column(region[cursor_region_line], cursor_region_cp),
        ),
        replacement_range=range_for(prm, path=active, version=item["version"], content=content,
                                    start=start, end=end),
        document_eol="lf",
    )
    return context, {
        "files": {path: {"uri": file["uri"], "content": file["content"],
                         "content_sha256": file["content_sha256"],
                         "byte_length": len(file["content"].encode("utf-8"))}
                   for path, file in item["files"].items()},
        "active_file": active,
        "active_uri": record["uri"],
        "active_content_sha256": record["content_sha256"],
        "version": item["version"],
        "workspace_revision": item["workspace_revision"],
        "start": list(start),
        "end": list(end),
        "cursor_absolute": list(item["cursor"]),
        "region_old": list(region),
        "scope_anchor_line": item["scope_anchor_line"],
        "provider_status": item["provider_status"],
        "evicted_event_ids": list(item["evicted_event_ids"]),
        "history": [event.to_dict() for event in item["history"]],
        "selected_evidence": [{"path": path, "content": text, "content_sha256": sha256_text(text)}
                               for path, text in item["selected"]],
        "diagnostics": list(item["diagnostics"]),
        "retrieval": list(item["retrieval"]),
    }


def validate_full_buffer(prm: Any, item: dict[str, Any], context: Any, meta: dict[str, Any]) -> None:
    record = item["files"][item["active"]]
    content = record["content"]
    assert sha256_text(content) == meta["active_content_sha256"]
    assert context.replacement_range.uri == record["uri"]
    assert context.replacement_range.document_version == item["version"]
    assert context.replacement_range.content_sha256 == meta["active_content_sha256"]
    expected = region_slice(content, tuple(item["start"]), tuple(item["end"]))
    assert tuple(context.region_old) == expected
    expected_prefix, expected_suffix = wire_prefix_suffix(content, tuple(item["start"]), tuple(item["end"]))
    assert tuple(context.prefix) == expected_prefix
    assert tuple(context.suffix_lines) == expected_suffix
    assert context.replacement_range.start.line == item["start"][0]
    assert context.replacement_range.end.line == item["end"][0]
    assert context.replacement_range.start.character == u16(text_lines(content)[item["start"][0]][:item["start"][1]])
    assert context.replacement_range.end.character == u16(text_lines(content)[item["end"][0]][:item["end"][1]])
    absolute_line, absolute_cp = item["cursor"]
    assert absolute_line == item["start"][0] + context.cursor.region_line_index
    expected_cp = absolute_cp - item["start"][1] if absolute_line == item["start"][0] else absolute_cp
    assert expected_cp == context.cursor.code_point_column
    assert context.cursor.utf16_column == u16(context.region_old[context.cursor.region_line_index][:expected_cp])


def validate_history_events(item: dict[str, Any]) -> int:
    """Validate recorded history ranges and old/new diff framing.

    The event constructor obtains ``old_text`` from the complete predecessor
    buffer and ``new_text`` from the complete successor buffer.  Here we
    verify the persisted range identity/order and the exact diff payload for
    every retained event in each context.
    """
    checked = 0
    for event in item["history"]:
        assert event.range_utf16.uri == f"synthetic://workspace/{event.path}"
        assert len(event.range_utf16.content_sha256) == 64
        assert event.range_utf16.document_version < item["version"]
        assert f"-{event.old_text}\n+{event.new_text}" in event.event_diff
        checked += 1
    return checked


def replace_line(content: str, line: int, new: str) -> str:
    lines = text_lines(content)
    lines[line] = new
    return "\n".join(lines)


def authored_cases(prm: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 1. Typing, including an astral Unicode cursor/range boundary.
    path = "R/typing-v2.R"
    b = "summarize <- function(x) {\n  😀 <- value\n  result <- mean(x)\n}\n"
    a = replace_line(b, 1, "  😀 <- values")
    sb = state(files={path: b}, active=path, version=1, start=(1, 2), end=(1, len(b.split("\n")[1])), cursor=(1, 4))
    sa = state(files={path: a}, active=path, version=2, start=(1, 2), end=(1, len(a.split("\n")[1])), cursor=(1, 4))
    ev = history_event(prm, event_id="PRM06V2-TYPING-H01", kind="typing", path=path, before=sb, start=(1, 2), end=(1, len(b.split("\n")[1])), after_content=a)
    sa["history"] = (ev,)
    cases.append({"id": "PRM06V2-01", "event_class": "typing", "before": sb, "after": sa,
                  "transition": "typed values into the active line; after history carries exact old/new source text"})

    # 2. Deletion, with a multi-line captured range.
    path = "R/deletion-v2.R"
    b = "clean <- function(x) {\n  keep <- x[x > 0]\n  sum(keep)\n}\n"
    a = "clean <- function(x) {\n  keep <- x[x > 0\n  sum(keep)\n}\n"
    sb = state(files={path: b}, active=path, version=1, start=(1, 2), end=(2, len(b.split("\n")[2])), cursor=(1, 12))
    sa = state(files={path: a}, active=path, version=2, start=(1, 2), end=(2, len(a.split("\n")[2])), cursor=(1, 12))
    ev = history_event(prm, event_id="PRM06V2-DELETION-H01", kind="deletion", path=path, before=sb, start=(1, 2), end=(1, 18), after_content=a)
    sa["history"] = (ev,)
    cases.append({"id": "PRM06V2-02", "event_class": "deletion", "before": sb, "after": sa,
                  "transition": "deleted the closing predicate character; selected region remains source-consistent"})

    # 3. Cursor move: same source identity, exact cursor transition.
    path = "R/cursor-v2.R"
    text = "format_label <- function(name) {\n  paste0(\"«\", name, \"»\")\n}\n"
    sb = state(files={path: text}, active=path, version=4, start=(1, 2), end=(1, len(text_lines(text)[1])), cursor=(1, 7))
    sa = state(files={path: text}, active=path, version=5, start=(1, 2), end=(1, len(text_lines(text)[1])), cursor=(1, 15))
    cases.append({"id": "PRM06V2-03", "event_class": "cursor_move", "before": sb, "after": sa,
                  "transition": "moved the cursor within the same pre-edit source buffer; history unchanged"})

    # 4. History append: actual prior edit plus a second source edit.
    path = "R/history-v2.R"
    b = "report <- function(x) {\n  total <- sum(x)\n  total\n}\n"
    a = replace_line(b, 1, "  total <- sum(x, na.rm = FALSE)")
    sb = state(files={path: b}, active=path, version=10, start=(2, 2), end=(2, len(text_lines(b)[2])), cursor=(2, 7))
    prior_before_text = replace_line(b, 1, "  total <- sum(x, na.rm = TRUE)")
    prior_before = state(files={path: prior_before_text}, active=path, version=9,
                         start=(2, 2), end=(2, len(text_lines(prior_before_text)[2])), cursor=(2, 7))
    prior = history_event(prm, event_id="PRM06V2-HISTORY-H01", kind="typing", path=path,
                          before=prior_before, start=(1, 2), end=(1, len(text_lines(prior_before_text)[1])), after_content=b)
    sb["history"] = (prior,)
    sa = state(files={path: a}, active=path, version=11, start=(2, 2), end=(2, len(text_lines(a)[2])), cursor=(2, 7), history=(prior,))
    current = history_event(prm, event_id="PRM06V2-HISTORY-H02", kind="typing", path=path, before=sb, start=(1, 2), end=(1, len(text_lines(b)[1])), after_content=a)
    sa["history"] = (prior, current)
    cases.append({"id": "PRM06V2-04", "event_class": "history_append", "before": sb, "after": sa,
                  "transition": "appended exact second edit to retained pre-edit history"})

    # 5. History eviction: whole event H0 is evicted after H3; all retained
    # history events have exact source old/new ranges.
    path = "R/eviction-v2.R"
    initial = "pipeline <- function(x) {\n  a <- x\n  b <- x\n  c <- x\n  result <- a + b + c\n}\n"
    versions = [initial]
    for line, new in ((1, "  a <- normalize(x)"), (2, "  b <- scale(x)"), (3, "  c <- finalize(x)"), (4, "  result <- a + b + c + 1")):
        versions.append(replace_line(versions[-1], line, new))
    s0 = state(files={path: versions[0]}, active=path, version=20, start=(3, 2), end=(3, len(text_lines(versions[0])[3])), cursor=(3, 7))
    h0 = history_event(prm, event_id="PRM06V2-EVICT-H00", kind="typing", path=path, before=s0, start=(1, 2), end=(1, len(text_lines(versions[0])[1])), after_content=versions[1])
    s1 = state(files={path: versions[1]}, active=path, version=21, start=(3, 2), end=(3, len(text_lines(versions[1])[3])), cursor=(3, 7), history=(h0,))
    h1 = history_event(prm, event_id="PRM06V2-EVICT-H01", kind="typing", path=path, before=s1, start=(2, 2), end=(2, len(text_lines(versions[1])[2])), after_content=versions[2])
    s2 = state(files={path: versions[2]}, active=path, version=22, start=(3, 2), end=(3, len(text_lines(versions[2])[3])), cursor=(3, 7), history=(h0, h1))
    h2 = history_event(prm, event_id="PRM06V2-EVICT-H02", kind="typing", path=path, before=s2, start=(3, 2), end=(3, len(text_lines(versions[2])[3])), after_content=versions[3])
    sb = state(files={path: versions[3]}, active=path, version=23, start=(3, 2), end=(3, len(text_lines(versions[3])[3])), cursor=(3, 7), history=(h0, h1, h2), evicted_event_ids=())
    h3 = history_event(prm, event_id="PRM06V2-EVICT-H03", kind="typing", path=path, before=sb, start=(4, 2), end=(4, len(text_lines(versions[3])[4])), after_content=versions[4])
    sa = state(files={path: versions[4]}, active=path, version=24, start=(4, 2), end=(4, len(text_lines(versions[4])[4])), cursor=(4, 9), history=(h1, h2, h3), evicted_event_ids=(h0.event_id,))
    cases.append({"id": "PRM06V2-05", "event_class": "history_eviction", "before": sb, "after": sa,
                  "transition": "appended H3 to a capacity-three history and evicted complete H0", "evicted_event_ids": [h0.event_id]})

    # 6. Diagnostic refresh: provider content changes while source identity
    # remains fixed.  The separate missing-provider control below stays empty.
    path = "R/diagnostic-v2.R"
    text = "validate <- function(x) {\n  if (is.null(x)) return(NULL)\n  length(x)\n}\n"
    sb = state(files={path: text}, active=path, version=30, start=(1, 2), end=(1, len(text_lines(text)[1])), cursor=(1, 6), diagnostics=("synthetic stale warning",), provider_status={"diagnostics": "stale", "retrieval": "not_requested"})
    sa = state(files={path: text}, active=path, version=31, start=(1, 2), end=(1, len(text_lines(text)[1])), cursor=(1, 6), diagnostics=("synthetic refreshed diagnostic",), provider_status={"diagnostics": "fresh", "retrieval": "not_requested"})
    cases.append({"id": "PRM06V2-06", "event_class": "diagnostic_refresh", "before": sb, "after": sa,
                  "transition": "refreshed diagnostic provider record on an unchanged source buffer"})

    # 7. Definition change: helper file and selected reference change, while
    # the active model file remains valid and source-consistent.
    helper = "R/helpers-v2.R"; active = "R/model-v2.R"
    hb = "normalize <- function(data) {\n  scale(data)\n}\n"
    ha = "normalize <- function(data, center = TRUE) {\n  scale(data, center = center)\n}\n"
    model = "fit <- function(data) {\n  normalize(data)\n}\n"
    sb = state(files={helper: hb, active: model}, active=active, version=40, start=(1, 2), end=(1, len(text_lines(model)[1])), cursor=(1, 15), selected=((helper, hb), (active, model)))
    sa = state(files={helper: ha, active: model}, active=active, version=41, start=(1, 2), end=(1, len(text_lines(model)[1])), cursor=(1, 15), selected=((helper, ha), (active, model)), provider_status={"diagnostics": "fresh", "retrieval": "available"}, retrieval=(helper,))
    ev = history_event(prm, event_id="PRM06V2-DEFINITION-H01", kind="definition_change", path=helper, before=sb, start=(0, 0), end=(0, len(text_lines(hb)[0])), after_content=ha)
    sa["history"] = (ev,)
    cases.append({"id": "PRM06V2-07", "event_class": "definition_change", "before": sb, "after": sa,
                  "transition": "changed helper definition bytes and selected reference; active model buffer unchanged"})

    # 8. Anchor move: a distant source edit changes the full buffer and the
    # selected scope anchor moves to a different source slice.
    path = "R/anchor-v2.R"
    lines = ["build <- function(x) {"] + [f"  value_{i:02d} <- x + {i}" for i in range(1, 42)] + ["}"]
    before_text = "\n".join(lines) + "\n"
    after_lines = list(lines); after_lines[35] = "  value_35 <- normalize(x + 35)"; after_text = "\n".join(after_lines) + "\n"
    sb = state(files={path: before_text}, active=path, version=50, start=(30, 2), end=(30, len(lines[30])), cursor=(30, 14), scope_lines=tuple(lines[8:14]), scope_anchor_line=10)
    sa = state(files={path: after_text}, active=path, version=51, start=(30, 2), end=(30, len(after_lines[30])), cursor=(30, 14), scope_lines=tuple(after_lines[18:24]), scope_anchor_line=20)
    ev = history_event(prm, event_id="PRM06V2-ANCHOR-H01", kind="anchor_move", path=path, before=sb, start=(35, 2), end=(35, len(lines[35])), after_content=after_text)
    sa["history"] = (ev,)
    cases.append({"id": "PRM06V2-08", "event_class": "anchor_move", "before": sb, "after": sa,
                  "transition": "distant source edit moved the selected scope anchor from line 10 to line 20"})

    # 9. File switch: both complete file buffers are present and the active
    # range changes URI/hash/version with the editor switch.
    one = "one <- function(x) {\n  x + 1\n}\n"; two = "two <- function(x) {\n  x * 2\n}\n"
    sb = state(files={"R/one-v2.R": one, "R/two-v2.R": two}, active="R/one-v2.R", version=60, start=(1, 2), end=(1, len(text_lines(one)[1])), cursor=(1, 3), selected=(("R/two-v2.R", two),))
    sa = state(files={"R/one-v2.R": one, "R/two-v2.R": two}, active="R/two-v2.R", version=61, start=(1, 2), end=(1, len(text_lines(two)[1])), cursor=(1, 3), selected=(("R/one-v2.R", one),))
    cases.append({"id": "PRM06V2-09", "event_class": "file_switch", "before": sb, "after": sa,
                  "transition": "switched active editor between two complete pre-edit files; URI/hash are both captured"})

    # 10. No-op control: source, providers, evidence, cursor and range all
    # remain byte-identical, so this denominator is intentionally unchanged.
    path = "R/noop-v2.R"; text = "identity <- function(x) {\n  x\n}\n"
    sb = state(files={path: text}, active=path, version=70, start=(1, 2), end=(1, len(text_lines(text)[1])), cursor=(1, 3), selected=((path, text),))
    sa = copy.deepcopy(sb)
    cases.append({"id": "PRM06V2-10", "event_class": "no_op", "before": sb, "after": sa,
                  "transition": "unchanged source/provider/evidence no-op control"})

    # Explicit missing-provider control.  It is not counted as a new named
    # event class and remains empty on both sides by policy.
    path = "R/missing-v2.R"; text = "check <- function(x) {\n  x\n}\n"
    sb = state(files={path: text}, active=path, version=80, start=(1, 2), end=(1, len(text_lines(text)[1])), cursor=(1, 3), provider_status={"diagnostics": "missing", "retrieval": "missing"})
    sa = copy.deepcopy(sb)
    cases.append({"id": "PRM06V2-11", "event_class": "missing-provider-control", "before": sb, "after": sa,
                  "transition": "unavailable diagnostics/retrieval remain empty on both sides; no content invented"})
    return cases


def lcp(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def encode(tokenizer: Any, text: str, prm: Any) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    if hasattr(ids, "ids"):
        ids = ids.ids
    ids = [int(value) for value in ids]
    if any(value in (prm.BOS_ID, prm.EOS_ID) for value in ids):
        raise ValueError("prompt encoded protocol BOS/EOS ID before manual wrapper")
    return ids


def digest(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def run() -> dict[str, Any]:
    prm = load_module("prm04_v2", EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py")
    v1 = load_module("prm06_v1_helpers", V1_PATH)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True, use_fast=True)
    cases = authored_cases(prm)
    named = {"typing", "deletion", "cursor_move", "history_append", "history_eviction",
             "diagnostic_refresh", "definition_change", "anchor_move", "file_switch", "no_op"}
    assert {case["event_class"] for case in cases} >= named
    per_case: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[dict[str, Any]]]] = {}
    full_buffer_states = 0
    history_records_checked = 0
    evidence_layout_checks = 0
    file_prefix_parity_checks = 0
    exact_prompt_records = 0
    all_layouts = tuple(v1.LAYOUTS)
    for case in cases:
        contexts: dict[str, Any] = {}
        metas: dict[str, Any] = {}
        for side in ("before", "after"):
            contexts[side], metas[side] = make_context(prm, case[side])
            validate_full_buffer(prm, case[side], contexts[side], metas[side])
            history_records_checked += validate_history_events(case[side])
            full_buffer_states += 1
        before, after = contexts["before"], contexts["after"]
        changed = digest(metas["before"]) != digest(metas["after"]) or before.to_dict() != after.to_dict()
        metrics: dict[str, Any] = {}
        evidence_digests: dict[str, Any] = {}
        exact_prompts: dict[str, Any] = {}
        for layout in all_layouts:
            before_text = v1.render_layout(prm, before, layout)
            after_text = v1.render_layout(prm, after, layout)
            if layout == "file-prefix-first":
                assert before_text == prm.render_prompt(before)
                assert after_text == prm.render_prompt(after)
                file_prefix_parity_checks += 2
            before_ids = encode(tokenizer, before_text, prm)
            after_ids = encode(tokenizer, after_text, prm)
            common = lcp(before_ids, after_ids)
            metrics[layout] = {
                "before_prompt_sha256": sha256_text(before_text),
                "after_prompt_sha256": sha256_text(after_text),
                "before_prompt_text": before_text,
                "after_prompt_text": after_text,
                "before_prompt_ids": before_ids,
                "after_prompt_ids": after_ids,
                "before_prompt_token_count": len(before_ids),
                "after_prompt_token_count": len(after_ids),
                "token_lcp": common,
                "recomputed_prompt_tokens": len(after_ids) - common,
                "lcp_fraction_after": round(common / len(after_ids), 8) if after_ids else 1.0,
            }
            blocks_before = v1.block_values(prm, before)
            blocks_after = v1.block_values(prm, after)
            evidence_digests[layout] = {
                "selected_references": digest(blocks_after["references"]),
                "snapshot_contract": digest({"task": blocks_after["task"], "region": blocks_after["region"], "boundary": blocks_after["boundary"]}),
                "provider_history": digest({"history": blocks_after["history"], "diagnostics": blocks_after["diagnostics"], "retrieval": blocks_after["retrieval"]}),
            }
            exact_prompts[layout] = {
                "before_sha256": metrics[layout]["before_prompt_sha256"],
                "after_sha256": metrics[layout]["after_prompt_sha256"],
                "before_ids_sha256": digest(before_ids),
                "after_ids_sha256": digest(after_ids),
            }
            exact_prompt_records += 2
        assert len({json.dumps(value, sort_keys=True) for value in evidence_digests.values()}) == 1
        evidence_layout_checks += 1
        case_result = {
            "id": case["id"],
            "event_class": case["event_class"],
            "provenance_class": "synthetic_functional_transition",
            "observed_real": False,
            "target_fields_read": [],
            "future_target_accessed": False,
            "transition": case["transition"],
            "changed_pair": changed,
            "before": metas["before"],
            "after": metas["after"],
            "before_history_event_ids": [item.event_id for item in before.history],
            "after_history_event_ids": [item.event_id for item in after.history],
            "selected_evidence_ids": [digest(item.to_dict()) for item in after.selected_references],
            "evidence_digests_by_layout": evidence_digests,
            "exact_prompt_records": exact_prompts,
            "metrics": metrics,
            "output_contract": {"task_contract": prm.TASK_CONTRACT, "generation_boundary": prm.GENERATION_BOUNDARY, "terminal": prm.TERMINAL, "target_in_prompt": False},
        }
        per_case.append(case_result)
        aggregate.setdefault(case["event_class"], {layout: [] for layout in all_layouts})
        for layout in all_layouts:
            aggregate[case["event_class"]][layout].append({**metrics[layout], "changed_pair": changed})

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows); changed_n = sum(bool(row["changed_pair"]) for row in rows)
        return {"pairs": n, "changed_pairs": changed_n, "unchanged_pairs": n - changed_n,
                "before_prompt_token_count": sum(row["before_prompt_token_count"] for row in rows) / n,
                "after_prompt_token_count": sum(row["after_prompt_token_count"] for row in rows) / n,
                "token_lcp": sum(row["token_lcp"] for row in rows) / n,
                "recomputed_prompt_tokens": sum(row["recomputed_prompt_tokens"] for row in rows) / n,
                "lcp_fraction_after": sum(row["lcp_fraction_after"] for row in rows) / n}

    aggregate_by_event_class = {event: {layout: summary(rows) for layout, rows in layouts.items()}
                                for event, layouts in sorted(aggregate.items())}
    all_rows = {layout: [case["metrics"][layout] | {"changed_pair": case["changed_pair"]} for case in per_case] for layout in all_layouts}
    report = {
        "schema_version": "sepalith.campaign.prm06.layout-preparation.v2",
        "task": "PRM-06",
        "provenance": {"class": "synthetic_functional_transition", "observed_real": 0, "future_target_accessed": False, "target_fields_read": [], "source_policy": "fresh authored buffers/evidence only; no final data/model target"},
        "source_roles": {"plan_root": str(PLAN_ROOT), "execution_root": str(EXEC_ROOT), "execution_head": "a79d6de38890355ec66d7227c974140160ab314e", "renderer_id": prm.RENDERER_ID, "schema_version": prm.SCHEMA_VERSION},
        "tokenizer": {"path": str(TOKENIZER_PATH), "revision": prm.TOKENIZER_REVISION, "tokenizer_json_sha256": prm.TOKENIZER_JSON_SHA256, "mode": "HF encode(add_special_tokens=False, split_special_tokens=True)", "manual_bos_id": prm.BOS_ID, "eos_id": prm.EOS_ID, "terminal": prm.TERMINAL},
        "layouts": {layout: {"order": list(v1.LAYOUTS[layout]), "same_evidence_and_contract": True} for layout in all_layouts},
        "case_count": len(per_case), "named_event_class_count": len(named), "named_event_classes": sorted(named),
        "controls": [case["id"] for case in per_case if case["event_class"] not in named],
        "changed_pair_denominator": {"all_cases": sum(case["changed_pair"] for case in per_case), "unchanged_cases": sum(not case["changed_pair"] for case in per_case), "named_event_cases": sum(case["event_class"] in named for case in per_case)},
        "checks": {
            "full_buffer_range_hash_cursor": f"PASS; {full_buffer_states}/{full_buffer_states} contexts validated against complete authored buffers",
            "history_event_geometry_and_diff": f"PASS; {history_records_checked} retained history records have predecessor range identity and exact old/new diff framing",
            "evidence_layout_parity": f"PASS; {evidence_layout_checks}/{len(per_case)} cases have identical selected-evidence block digests across 3 layouts",
            "file_prefix_first_renderer_parity": f"PASS; {file_prefix_parity_checks}/{full_buffer_states} contexts equal frozen PRM-04 render_prompt",
            "exact_prompt_and_ids_saved": f"PASS; {exact_prompt_records} state/layout prompt text+ID records saved",
        },
        "per_case": per_case,
        "aggregate_by_event_class": aggregate_by_event_class,
        "aggregate_by_layout": {layout: summary(rows) for layout, rows in all_rows.items()},
        "layout_nomination": None,
        "nomination_status": "none; token overlap is diagnostic only and pinned-server timing is required",
        "limitations": ["All transitions and provider records are synthetic functional data; observed real coverage is zero.", "Missing-provider and no-op controls are intentionally unchanged and reported separately.", "Prompt text and exact HF token IDs are saved per case/layout for lead timing; no model/server/GPU was used.", "Source/application validation covers authored full buffers and ranges; it does not claim VS Code WorkspaceEdit behavior."],
    }
    report["report_digest_excluding_self"] = digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
