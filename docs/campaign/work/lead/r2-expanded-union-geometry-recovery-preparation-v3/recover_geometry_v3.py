#!/usr/bin/env python3
"""Recover source geometry without replacing an explicit cursor with range start.

The input context stores ``cursor`` relative to ``region_old``.  For a full
snapshot, the producer converts that position to an absolute LSP UTF-16
position and records the evidence used for the conversion.  The only supported
cursor sentinel is ``region_line_index=-1`` with both columns null; it is
accepted only for a zero-width replacement at the prefix boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


SCHEMA = "sepalith.sft11.expanded-union-geometry-recovery.v3"
GEOMETRY = "sepalith.source-cursor-geometry.v1"
CURSOR_EVIDENCE = "sepalith.source-cursor-evidence.v1"


class RecoveryError(RuntimeError):
    pass


def req(value: Any, message: str) -> None:
    if not value:
        raise RecoveryError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pos(value: Any, label: str) -> dict[str, int]:
    req(isinstance(value, dict) and set(value) == {"line", "character"}, label)
    req(type(value["line"]) is int and value["line"] >= 0, label + ":line")
    req(type(value["character"]) is int and value["character"] >= 0, label + ":character")
    return {"line": value["line"], "character": value["character"]}


def utf16_len(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def valid_utf16_position(
    lines: list[str], value: dict[str, int], label: str, allow_eof: bool = False
) -> None:
    if allow_eof and value["line"] == len(lines):
        req(value["character"] == 0, label + ":eof_character")
        return
    req(value["line"] < len(lines), label + ":line_outside_buffer")
    line = lines[value["line"]]
    req(value["character"] <= utf16_len(line), label + ":column_outside_buffer")
    # An LSP UTF-16 position cannot split a surrogate pair.
    units = line.encode("utf-16-le")
    try:
        units[: 2 * value["character"]].decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise RecoveryError(label + ":surrogate_boundary") from exc


def uri_path(uri: str) -> str:
    req(isinstance(uri, str) and uri.startswith("file://"), "replacement_uri")
    return unquote(urlparse(uri).path)


def identity(
    provenance: dict[str, Any], layout: str, replacement_range: dict[str, Any]
) -> tuple[dict[str, Any], str, str, str, dict[str, str]]:
    original = provenance.get("source_identity")
    req(isinstance(original, dict), "source_identity")
    preedit = provenance.get("preedit_sha256", replacement_range.get("content_sha256"))
    req(preedit == replacement_range.get("content_sha256"), "preedit_range_hash_join")
    if layout == "direct":
        source_path = original.get("source_path")
        source_sha = original.get("source_sha256")
        req(isinstance(source_path, str) and source_path.startswith("/"), "direct_source_path")
        req(isinstance(source_sha, str) and len(source_sha) == 64, "direct_source_sha")
        req(
            Path(uri_path(replacement_range["uri"])) == Path(source_path),
            "direct_uri_source_path_join",
        )
        basis = {
            "layout": "direct",
            "source_path_field": "source_identity.source_path",
            "source_sha256_field": "source_identity.source_sha256",
            "preedit_sha256_field": "preedit_sha256",
        }
    else:
        nested = original.get("source_provenance")
        req(isinstance(nested, dict), "nested_source_provenance")
        source_path = nested.get("source_snapshot_path")
        source_sha = nested.get("source_snapshot_sha256")
        req(isinstance(source_path, str) and source_path.startswith("/"), "nested_source_path")
        req(isinstance(source_sha, str) and len(source_sha) == 64, "nested_source_sha")
        req(nested.get("after_snapshot_sha256") == preedit, "nested_after_snapshot_preedit_join")
        logical = uri_path(replacement_range["uri"])
        req(source_path.endswith(logical.removeprefix("/sepalith/")), "nested_uri_source_suffix_join")
        basis = {
            "layout": "nested_original15006",
            "source_path_field": "source_identity.source_provenance.source_snapshot_path",
            "source_sha256_field": "source_identity.source_provenance.source_snapshot_sha256",
            "preedit_sha256_field": "source_identity.source_provenance.after_snapshot_sha256",
        }
    return original, source_path, source_sha, preedit, basis


def _ordered_le(left: dict[str, int], right: dict[str, int]) -> bool:
    return (left["line"], left["character"]) <= (right["line"], right["character"])


def _cursor_from_context(
    context: dict[str, Any],
    parts: list[list[str]],
    start: dict[str, int],
    end: dict[str, int],
    full_claim: bool,
    lines: list[str],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Convert the relative prompt cursor and return auditable evidence.

    ``region_line_index`` and both columns are an explicit relative cursor
    contract.  The null/-1 triple is a separate explicit zero-width sentinel.
    No other missing or partial cursor is guessed from the replacement range.
    """
    cursor = context.get("cursor")
    req(isinstance(cursor, dict), "cursor")
    req(
        set(cursor) == {"code_point_column", "region_line_index", "utf16_column"},
        "cursor_fields",
    )
    region_index = cursor["region_line_index"]
    code_point = cursor["code_point_column"]
    utf16_column = cursor["utf16_column"]
    sentinel = region_index == -1 and code_point is None and utf16_column is None
    if sentinel:
        req(start == end, "cursor_sentinel_requires_zero_width")
        req(start["line"] == len(parts[0]), "cursor_sentinel_prefix_boundary")
        if parts[2]:
            req(start["character"] <= utf16_len(parts[2][0]), "cursor_sentinel_column")
        return dict(start), {
            "schema": CURSOR_EVIDENCE,
            "basis": "zero_width_replacement_range_start_sentinel",
            "region_line_index": -1,
            "code_point_column": None,
            "utf16_column": None,
            "absolute_position": dict(start),
        }

    req(type(region_index) is int and region_index >= 0, "cursor_region_line_index")
    req(type(code_point) is int and code_point >= 0, "cursor_code_point_column")
    req(type(utf16_column) is int and utf16_column >= 0, "cursor_utf16_column")
    req(region_index < len(parts[1]), "cursor_region_line_outside_region")
    line = parts[1][region_index]
    req(code_point <= len(line), "cursor_code_point_outside_region_line")
    expected_utf16 = utf16_len(line[:code_point])
    req(expected_utf16 == utf16_column, "cursor_codepoint_utf16_mismatch")
    absolute = {"line": len(parts[0]) + region_index, "character": utf16_column}
    if full_claim:
        valid_utf16_position(lines, absolute, "cursor")
    else:
        # A partial context is usable only when the captured prefix is anchored
        # at document line zero.  Nonempty ranges remain unresolved below.
        req(start["line"] == len(parts[0]), "partial_cursor_prefix_boundary")
    req(_ordered_le(start, absolute) and _ordered_le(absolute, end), "cursor_outside_replacement_range")
    if not full_claim:
        req(start == end, "partial_nonempty_range_unverified")
    return absolute, {
        "schema": CURSOR_EVIDENCE,
        "basis": "explicit_region_cursor",
        "region_line_index": region_index,
        "code_point_column": code_point,
        "utf16_column": utf16_column,
        "absolute_position": dict(absolute),
        "prefix_line_count": len(parts[0]),
        "region_line_code_points": len(line),
        "region_line_utf16_units": utf16_len(line),
    }


def recover(
    record: dict[str, Any],
    provenance: dict[str, Any],
    context_key: str,
    layout: str,
    expected_row_id: str,
) -> dict[str, Any]:
    req(record.get("row_id") == expected_row_id, "row_id_join")
    req((provenance.get("row_id") or provenance.get("id")) == expected_row_id, "provenance_row_id_join")
    if "selection_target_or_gold_used" in record:
        req(record["selection_target_or_gold_used"] is False, "target_used_for_selection")
    if "context_has_target_or_reward_keys" in record:
        req(record["context_has_target_or_reward_keys"] is False, "context_has_target")

    context = record.get(context_key)
    req(isinstance(context, dict), context_key)
    replacement = context.get("replacement_range")
    req(isinstance(replacement, dict), "replacement_range")
    start = pos(replacement.get("start"), "start")
    end = pos(replacement.get("end"), "end")
    req(_ordered_le(start, end), "range_reversed")
    content = replacement.get("content_sha256")
    req(isinstance(content, str) and len(content) == 64, "content_sha256")
    original, source_path, source_sha, preedit, basis = identity(provenance, layout, replacement)
    eol = {"lf": "\n", "crlf": "\r\n"}.get(context.get("document_eol"))
    req(eol is not None, "document_eol")
    parts = [context.get("prefix"), context.get("region_old"), context.get("suffix_lines")]
    req(all(isinstance(value, list) and all(isinstance(line, str) for line in value) for value in parts), "context_lines")
    lines = sum(parts, [])
    availability = record.get("selection_geometry", {}).get("availability")
    full_claim = availability == "full_snapshot"
    observed = digest_text(eol.join(lines))
    req(not full_claim or observed == preedit, "claimed_full_snapshot_hash_mismatch")
    if full_claim:
        valid_utf16_position(lines, start, "start", True)
        valid_utf16_position(lines, end, "end", True)
    else:
        # A partial context can establish only an anchored zero-width boundary.
        req(start == end, "partial_nonempty_range_unverified")
        req(start["line"] == len(parts[0]), "partial_cursor_prefix_boundary")
        if parts[2]:
            req(start["character"] <= utf16_len(parts[2][0]), "partial_cursor_column")
    absolute_cursor, cursor_evidence = _cursor_from_context(
        context, parts, start, end, full_claim, lines
    )
    window = digest_text(
        canonical(
            {
                "document_eol": context["document_eol"],
                "prefix": parts[0],
                "region_old": parts[1],
                "suffix_lines": parts[2],
            }
        )
    )
    geometry = {
        "schema": GEOMETRY,
        "source_path": source_path,
        "source_sha256": source_sha,
        "preedit_sha256": preedit,
        "cursor": absolute_cursor,
        "replacement_range": {"start": start, "end": end},
        "window_sha256": window,
    }
    return {
        "row_id": expected_row_id,
        "status": "recovered" if full_claim else "recovered_context_geometry_buffer_unavailable",
        "source_identity": original,
        "source_cursor_geometry": geometry,
        "source_cursor_geometry_sha256": digest_text(canonical(geometry)),
        "cursor_evidence": cursor_evidence,
        "geometry_identity_basis": basis,
        "full_buffer_hash_verified": full_claim,
        "target_or_gold_used": False,
    }


def rows(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def stream_map(
    sidecar: Path,
    sidecar_pin: str,
    provenance: Path,
    provenance_pin: str,
    ids_path: Path,
    ids_pin: str,
    context_key: str,
    layout: str,
    output: Path,
) -> dict[str, Any]:
    req(sha(sidecar) == sidecar_pin, "sidecar_sha")
    req(sha(provenance) == provenance_pin, "provenance_sha")
    req(sha(ids_path) == ids_pin, "candidate_ids_sha")
    wanted: list[str] = []
    for item in rows(ids_path):
        wanted.append(item.get("row_id") or item.get("id"))
    req(all(isinstance(item, str) and item for item in wanted) and len(wanted) == len(set(wanted)), "candidate_ids")
    wanted_set = set(wanted)
    prov: dict[str, dict[str, Any]] = {}
    for item in rows(provenance):
        row_id = item.get("row_id") or item.get("id")
        if row_id in wanted_set:
            req(row_id not in prov, "provenance_duplicate_id")
            prov[row_id] = item
    seen: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    for record in rows(sidecar):
        row_id = record.get("row_id")
        if row_id in wanted_set:
            req(row_id not in seen, "sidecar_duplicate_id")
            seen.add(row_id)
            try:
                req(row_id in prov, "provenance_row_missing")
                output_rows.append(recover(record, prov[row_id], context_key, layout, row_id))
            except RecoveryError as exc:
                output_rows.append(
                    {
                        "row_id": row_id,
                        "status": "unresolved",
                        "reason": str(exc),
                        "silent_drop": False,
                        "target_or_gold_used": False,
                    }
                )
    for row_id in sorted(wanted_set - seen):
        output_rows.append(
            {
                "row_id": row_id,
                "status": "unresolved",
                "reason": "sidecar_row_missing",
                "silent_drop": False,
                "target_or_gold_used": False,
            }
        )
    req(not output.exists(), "fresh_output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    ledger = temporary / "geometry.jsonl"
    with ledger.open("x", encoding="utf-8") as stream:
        for item in sorted(output_rows, key=lambda value: value["row_id"]):
            stream.write(canonical(item) + "\n")
    manifest = {
        "schema": SCHEMA,
        "status": "review_only",
        "input_ids": len(wanted),
        "recovered": sum(item["status"].startswith("recovered") for item in output_rows),
        "unresolved": sum(item["status"] == "unresolved" for item in output_rows),
        "outputs": {"geometry.jsonl": {"path": "geometry.jsonl", "rows": len(output_rows), "sha256": sha(ledger)}},
        "training_admission": False,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.rename(temporary, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--candidate-ids-sha256", required=True)
    parser.add_argument("--context-key", choices=("context", "selected_context"), required=True)
    parser.add_argument("--identity-layout", choices=("direct", "nested_original15006"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        canonical(
            stream_map(
                args.sidecar,
                args.sidecar_sha256,
                args.provenance,
                args.provenance_sha256,
                args.candidate_ids,
                args.candidate_ids_sha256,
                args.context_key,
                args.identity_layout,
                args.output,
            )
        )
    )


if __name__ == "__main__":
    main()
