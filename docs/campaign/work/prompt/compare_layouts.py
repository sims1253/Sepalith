#!/usr/bin/env python3
"""CPU-only PRM-06 prompt-layout preparation.

The accepted PRM-01 package contains synthetic pre-edit snapshots.  This
script adapts those snapshots to the shared PRM-04 context type and measures
token-ID prefix reuse for three section orders.  It never reads a target
field.  ``before``/``after`` pairs are explicitly marked synthetic: where an
approved prior provider state is unavailable, the pair is identical and the
report records that no transition was measured.

The output is a JSON report suitable for embedding in the PRM-06 receipt.  It
contains prompt text hashes and token counts, but it does not run a model or a
server and makes no latency claim.
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
from typing import Any, Iterable


PLAN_ROOT = Path(__file__).resolve().parents[4]
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
FIXTURE_PATH = PLAN_ROOT / "docs/campaign/work/prompt/event-fixtures.json"
PRM01_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/PRM-01-event-fixtures.json"
TOKENIZER_PATH = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")


def load_protocol() -> Any:
    path = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
    spec = importlib.util.spec_from_file_location("sepalith_campaign_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen PRM-04 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    return sha256_bytes(data.encode("utf-8"))


def utf16_length(text: str) -> int:
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def source_file(snapshot: dict[str, Any], path: str) -> dict[str, Any]:
    for item in snapshot["files"]:
        if item["path"] == path:
            return item
    raise ValueError(f"active/path file {path!r} missing from bounded snapshot")


def identity_hex(value: Any, *, fallback_text: str | None = None) -> str:
    if isinstance(value, str) and value.startswith("sha256:") and len(value) == 71:
        return value[7:]
    if fallback_text is None:
        raise ValueError(f"missing source content identity: {value!r}")
    return sha256_text(fallback_text)


def evidence_id(record: dict[str, Any]) -> str:
    identity = record.get("content_identity")
    if not isinstance(identity, str):
        identity = "sha256:" + sha256_text(str(record.get("content", "")))
    payload = "\0".join((str(record.get("path", "")), identity, str(record.get("content", ""))))
    return sha256_text(payload)


def make_range(prm: Any, *, path: str, version: int, content_sha: str,
               start_line: int, lines: Iterable[str]) -> Any:
    lines = tuple(lines)
    if not lines:
        start = prm.Position(start_line, 0)
        end = start
    else:
        start = prm.Position(start_line, 0)
        end = prm.Position(start_line + len(lines) - 1, utf16_length(lines[-1]))
    return prm.ReplacementRange(
        uri=f"synthetic://{path}",
        document_version=version,
        content_sha256=content_sha,
        start=start,
        end=end,
    )


def make_history_event(prm: Any, raw: dict[str, Any], *, path: str,
                       content_sha: str, version: int) -> Any:
    old_text = str(raw.get("old_text", ""))
    old_lines = old_text.split("\n") if old_text else ()
    return prm.HistoryEvent(
        event_id=str(raw["event_id"]),
        kind=str(raw["kind"]),
        path=str(raw["path"]),
        workspace_revision_before=str(raw.get("workspace_revision", "synthetic")),
        old_text=old_text,
        new_text=str(raw.get("new_text", "")),
        range_utf16=make_range(
            prm,
            path=str(raw["path"]),
            version=version,
            content_sha=identity_hex(raw.get("content_identity"), fallback_text=""),
            start_line=0,
            lines=old_lines,
        ),
        event_diff=str(raw.get("event_diff", "")),
    )


def evidence_records(prm: Any, records: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(prm.EvidenceRecord(content=str(item.get("content", "")),
                                    path=item.get("path")) for item in records)


def provider_records(prm: Any, values: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(values, list):
        return ()
    # Diagnostics are complete short records in the synthetic package.  The
    # retrieval provider contains identifiers only; preserving those strings
    # makes the absence/identifier boundary visible without inventing a file.
    return tuple(prm.EvidenceRecord(content=str(value), path=path if isinstance(value, dict) else None)
                 for value in values)


def context_from_fixture(prm: Any, fixture: dict[str, Any], ordinal: int) -> tuple[Any, dict[str, Any]]:
    context = fixture["context"]
    snapshot = fixture["input_snapshot"]
    path = str(context["path"])
    file = source_file(snapshot, path)
    source_sha = identity_hex(file.get("content_identity"), fallback_text=str(file["content"]))
    input_identity = fixture.get("input_identity", {})
    snapshot_id = str(input_identity.get("pre_edit_snapshot_sha256", ""))
    if len(snapshot_id) != 64:
        raise ValueError(f"{fixture['fixture_id']} has no pre-edit snapshot ID")

    region = tuple(str(line) for line in context["region_old"])
    selection = snapshot.get("selection", {})
    raw_cursor_index = context.get("cursor_idx")
    raw_cursor_column = context.get("cursor_column")
    cursor_adapter = None
    if raw_cursor_index is None or int(raw_cursor_index) < 0:
        # E11 uses a legacy -1/null sentinel for a whitespace-only *line*.
        # PRM-04 correctly treats ["  "] as a nonempty captured region, so
        # use the bounded selection's real code-point column and line index.
        raw_cursor_index = 0
        raw_cursor_column = selection.get("code_point_column", 0)
        cursor_adapter = "legacy -1/null converted to bounded nonempty-line cursor from pre-edit selection"
    raw_cursor_column = 0 if raw_cursor_column is None else int(raw_cursor_column)
    if not 0 <= raw_cursor_index < len(region):
        raise ValueError(f"{fixture['fixture_id']} cursor index does not address region_old")
    if raw_cursor_column > len(region[raw_cursor_index]):
        # The source fixture can report a document column while the selected
        # region is a bounded line.  The explicit selection is authoritative
        # only when it is geometrically valid; otherwise fail closed.
        raise ValueError(f"{fixture['fixture_id']} cursor column exceeds bounded region line")
    cursor = prm.Cursor(
        region_line_index=int(raw_cursor_index),
        code_point_column=raw_cursor_column,
        utf16_column=prm.codepoint_to_utf16_column(region[raw_cursor_index], raw_cursor_column),
    )

    selection_line = selection.get("line")
    start_line = max(0, int(selection_line) - 1) if isinstance(selection_line, int) else 0
    replacement_range = make_range(
        prm,
        path=path,
        version=ordinal + 1,
        content_sha=source_sha,
        start_line=start_line,
        lines=region,
    )

    provider = snapshot.get("provider_state", {})
    diagnostics_state = provider.get("diagnostics", {})
    retrieval_state = provider.get("retrieval", {})
    history_items = snapshot.get("history", {}).get("retained_events", [])
    history = tuple(
        make_history_event(prm, item, path=path, content_sha=source_sha, version=ordinal + 1)
        for item in history_items
    )
    evidence = context.get("evidence", [])
    scope_mode = "pin+outline" if context.get("prefix") or context.get("suffix") else "off"
    result = prm.PromptContext(
        schema_version=prm.SCHEMA_VERSION,
        path=path,
        prefix=tuple(str(line) for line in context.get("prefix", [])),
        selected_references=evidence_records(prm, evidence),
        history=history,
        diagnostics=provider_records(prm, diagnostics_state.get("records", []), path)
        if diagnostics_state.get("status") != "missing" else (),
        retrieval=provider_records(prm, retrieval_state.get("records", []), path)
        if retrieval_state.get("status") != "missing" else (),
        scope_mode=scope_mode,
        # PRM-01 has no separate scope array.  Keep the bounded prefix/suffix
        # exact and record this empty optional provider field explicitly.
        scope_lines=(),
        suffix_lines=tuple(str(line) for line in context.get("suffix", [])),
        region_old=region,
        cursor=cursor,
        replacement_range=replacement_range,
        document_eol=str(file.get("newline", "lf")),
    )
    metadata = {
        "fixture_id": fixture["fixture_id"],
        "event_class": fixture["event"]["kind"],
        "provenance_class": "synthetic_functional",
        "observed_real": False,
        "target_fields_read": [],
        "future_target_accessed": False,
        "pre_edit_snapshot_id": snapshot_id,
        "active_file_content_sha256": source_sha,
        "source_package": context.get("metadata", {}).get("source_package"),
        "source_workspace_revision": file.get("workspace_revision"),
        "document_eol": file.get("newline"),
        "range_adapter": "bounded synthetic range starts at selection line and spans exact region_old LF lines",
        "cursor_adapter": cursor_adapter,
        "selected_evidence_ids": [evidence_id(item) for item in evidence],
        "selected_evidence_count": len(evidence),
        "provider_status": {
            "definitions": provider.get("definitions", {}).get("status"),
            "diagnostics": diagnostics_state.get("status"),
            "retrieval": retrieval_state.get("status"),
        },
        "unavailable_provider_reasons": {
            key: value.get("reason")
            for key, value in (("definitions", provider.get("definitions", {})),
                               ("diagnostics", diagnostics_state), ("retrieval", retrieval_state))
            if value.get("status") == "missing" and value.get("reason")
        },
        "evicted_event_ids": list(snapshot.get("history", {}).get("evicted_event_ids", [])),
    }
    return result, metadata


def clone_with_history(prm: Any, context: Any, history: tuple[Any, ...]) -> Any:
    return dataclasses.replace(context, history=history)


def synthetic_event(prm: Any, context: Any, event_class: str, fixture_id: str,
                    ordinal: int) -> Any:
    event_id = f"{fixture_id}-PRM06-SYNTHETIC"
    event_diff = f"Synthetic pre-edit {event_class} transition ({event_id}); no target text"
    return prm.HistoryEvent(
        event_id=event_id,
        kind=event_class,
        path=context.path,
        workspace_revision_before=f"synthetic-prm06-r{ordinal + 1:02d}",
        old_text="",
        new_text="",
        range_utf16=dataclasses.replace(context.replacement_range),
        event_diff=event_diff,
    )


def event_pair(prm: Any, fixture: dict[str, Any], base: Any, meta: dict[str, Any], ordinal: int) -> tuple[Any, Any, dict[str, Any]]:
    event_class = str(meta["event_class"])
    fixture_id = str(meta["fixture_id"])
    pair_meta: dict[str, Any] = {
        "pair_id": f"{fixture_id}-PRM06-pair",
        "provenance_class": "synthetic_functional",
        "observed_real": False,
        "target_fields_read": [],
        "future_target_accessed": False,
        "source_snapshot_id": meta["pre_edit_snapshot_id"],
        "event_class": event_class,
    }
    before = base
    after = base
    mutation = "no approved before/after provider state; prompt state held constant"

    if event_class in {"typing", "deletion", "history_append"} and base.history:
        # The retained event list is explicitly pre-edit synthetic history.
        # Removing its newest item gives a bounded before state without using
        # the target or any post-edit provider result.
        before = clone_with_history(prm, base, base.history[:-1])
        mutation = "removed newest retained pre-edit history event; no target text"
    elif event_class == "cursor_move":
        selection = fixture["input_snapshot"].get("selection", {})
        if isinstance(selection.get("from_code_point_column"), int) and isinstance(selection.get("to_code_point_column"), int):
            before = base
            after = dataclasses.replace(
                base,
                cursor=prm.Cursor(
                    region_line_index=base.cursor.region_line_index,
                    code_point_column=int(selection["to_code_point_column"]),
                    utf16_column=prm.codepoint_to_utf16_column(
                        base.region_old[base.cursor.region_line_index],
                        int(selection["to_code_point_column"]),
                    ),
                ),
            )
            mutation = "used bounded synthetic selection from_code_point_column -> to_code_point_column"
    elif event_class == "file_switch":
        # Both file texts and identities are present in this pre-edit
        # snapshot.  Build the prior active-file context from that exact
        # source file, without inventing a target or provider result.
        selection = fixture["input_snapshot"].get("selection", {})
        previous_path = selection.get("previous_active_file")
        if isinstance(previous_path, str):
            previous = source_file(fixture["input_snapshot"], previous_path)
            previous_lines = str(previous["content"]).splitlines()
            if len(previous_lines) >= 3:
                prior_region = (previous_lines[1],)
                prior_prefix = (previous_lines[0],)
                prior_suffix = tuple(previous_lines[2:])
                prior_sha = identity_hex(previous.get("content_identity"), fallback_text=str(previous["content"]))
                prior_selection = int(selection.get("from_code_point_column", 0))
                prior_cursor = prm.Cursor(
                    region_line_index=0,
                    code_point_column=min(prior_selection, len(prior_region[0])),
                    utf16_column=prm.codepoint_to_utf16_column(prior_region[0], min(prior_selection, len(prior_region[0]))),
                )
                prior_range = make_range(prm, path=previous_path, version=ordinal + 1,
                                         content_sha=prior_sha, start_line=1, lines=prior_region)
                before = dataclasses.replace(
                    base, path=previous_path, prefix=prior_prefix, region_old=prior_region,
                    suffix_lines=prior_suffix, cursor=prior_cursor, replacement_range=prior_range,
                )
                mutation = "switched from bounded previous_active_file to active file; both source identities are in pre-edit snapshot"
    elif event_class == "history_eviction":
        mutation = "evicted event IDs have no retained text; unavailable history remains empty"
    elif event_class == "diagnostic_refresh":
        mutation = "diagnostic provider is missing; no synthetic diagnostic content added"
    elif event_class == "definition_change":
        mutation = "prior definition bytes are unavailable; selected references remain exact current pre-edit evidence"
    elif event_class == "anchor_move":
        mutation = "prior scope selection is unavailable; bounded prefix/suffix remain exact"
    elif event_class == "no_op":
        mutation = "no-op event has no source/provider transition"

    pair_meta["mutation"] = mutation
    pair_meta["before"] = {
        "state": "synthetic_pre_edit_before",
        "path": before.path,
        "snapshot_id": meta["pre_edit_snapshot_id"],
        "region_old": list(before.region_old),
        "cursor": before.cursor.to_dict(),
        "replacement_range": before.replacement_range.to_dict(),
        "history_event_ids": [event.event_id for event in before.history],
        "selected_evidence_ids": meta["selected_evidence_ids"],
        "provider_sections": {
            "diagnostics_count": len(before.diagnostics),
            "retrieval_count": len(before.retrieval),
        },
    }
    pair_meta["after"] = {
        "state": "synthetic_pre_edit_after",
        "path": after.path,
        "snapshot_id": meta["pre_edit_snapshot_id"],
        "region_old": list(after.region_old),
        "cursor": after.cursor.to_dict(),
        "replacement_range": after.replacement_range.to_dict(),
        "history_event_ids": [event.event_id for event in after.history],
        "selected_evidence_ids": meta["selected_evidence_ids"],
        "provider_sections": {
            "diagnostics_count": len(after.diagnostics),
            "retrieval_count": len(after.retrieval),
        },
    }
    return before, after, pair_meta


def records_block(prm: Any, records: Iterable[Any]) -> list[str]:
    values = []
    for record in records:
        if record.path is not None:
            values.append(f"<filename>{record.path}")
        values.append(record.content)
    return values


def block_values(prm: Any, context: Any) -> dict[str, list[str]]:
    region = list(context.region_old)
    if context.cursor.region_line_index >= 0:
        index = context.cursor.region_line_index
        column = context.cursor.code_point_column
        assert column is not None
        region[index] = region[index][:column] + prm.CURSOR_MARKER + region[index][column:]
    return {
        "task": [prm.TASK_CONTRACT],
        "file_prefix": [f"<filename>{context.path}", *context.prefix],
        "references": ["<filename>selected_references", *records_block(prm, context.selected_references)],
        "history": ["<filename>edit_history", *(event.event_diff for event in context.history)],
        "diagnostics": ["<filename>diagnostics", *records_block(prm, context.diagnostics)],
        "retrieval": ["<filename>retrieval", *records_block(prm, context.retrieval)],
        "scope_suffix": ["<[fim-suffix]>", *context.scope_lines, *context.suffix_lines],
        "region": ["<<<<<<< CURRENT", *region],
        "boundary": ["=======", prm.GENERATION_BOUNDARY],
    }


LAYOUTS = {
    "suffix-first": ("task", "scope_suffix", "file_prefix", "references", "history", "diagnostics", "retrieval", "region", "boundary"),
    "file-prefix-first": ("task", "file_prefix", "references", "history", "diagnostics", "retrieval", "scope_suffix", "region", "boundary"),
    "stable-reference-first": ("task", "references", "file_prefix", "history", "diagnostics", "retrieval", "scope_suffix", "region", "boundary"),
}


def render_layout(prm: Any, context: Any, layout: str) -> str:
    values = block_values(prm, context)
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout}")
    parts: list[str] = []
    for name in LAYOUTS[layout]:
        parts.extend(values[name])
    return "\n".join(parts) + "\n"


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    if hasattr(ids, "ids"):
        ids = ids.ids
    result = [int(value) for value in ids]
    if any(value in (prm_global.BOS_ID, prm_global.EOS_ID) for value in result):
        raise ValueError("prompt encoded protocol BOS/EOS before manual wrapper")
    return result


def layout_metrics(prm: Any, tokenizer: Any, before: Any, after: Any, layout: str) -> dict[str, Any]:
    before_text = render_layout(prm, before, layout)
    after_text = render_layout(prm, after, layout)
    before_ids = encode(tokenizer, before_text)
    after_ids = encode(tokenizer, after_text)
    common = lcp(before_ids, after_ids)
    return {
        "before_prompt_sha256": sha256_text(before_text),
        "after_prompt_sha256": sha256_text(after_text),
        "before_prompt_token_count": len(before_ids),
        "after_prompt_token_count": len(after_ids),
        "token_lcp": common,
        "recomputed_prompt_tokens": len(after_ids) - common,
        "lcp_fraction_after": round(common / len(after_ids), 8) if after_ids else 1.0,
        "wire_final_lf": before_text.endswith("\n") and after_text.endswith("\n"),
    }


def digest_object(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def run() -> dict[str, Any]:
    global prm_global
    prm = load_protocol()
    prm_global = prm
    package = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixtures = package["fixtures"]
    if len(fixtures) != 11:
        raise ValueError(f"expected 11 accepted PRM-01 fixtures, found {len(fixtures)}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), local_files_only=True, use_fast=True,
    )

    per_case: list[dict[str, Any]] = []
    layout_totals: dict[str, list[dict[str, Any]]] = {name: [] for name in LAYOUTS}
    aggregate_inputs: dict[str, list[dict[str, Any]]] = {}
    scope_members = package.get("coverage", {})

    def scope_class_for(fixture_id: str) -> str:
        # PRM-01's coverage table stores the concrete membership lists.  The
        # count map is retained separately and is not used to infer cases.
        for name in ("long_scope_fixtures", "cross_file_fixtures", "local_only_fixtures",
                     "missing_evidence_fixtures", "no_op_fixtures"):
            if fixture_id in scope_members.get(name, []):
                return {
                    "long_scope_fixtures": "long_scope",
                    "cross_file_fixtures": "cross_file",
                    "local_only_fixtures": "local_only",
                    "missing_evidence_fixtures": "missing_evidence",
                    "no_op_fixtures": "no_op",
                }[name]
        raise ValueError(f"{fixture_id} has no PRM-01 scope class")

    for ordinal, fixture in enumerate(fixtures):
        context, meta = context_from_fixture(prm, fixture, ordinal)
        before, after, pair = event_pair(prm, fixture, context, meta, ordinal)
        scope_class = scope_class_for(str(fixture["fixture_id"]))
        # The baseline order is byte-identical to the frozen PRM-04 renderer.
        baseline = prm.render_prompt(after)
        baseline_layout = render_layout(prm, after, "file-prefix-first")
        if baseline != baseline_layout:
            raise AssertionError(f"{fixture['fixture_id']} file-prefix-first differs from PRM-04 renderer")
        evidence_sets = {}
        metrics: dict[str, Any] = {}
        for layout in LAYOUTS:
            values = block_values(prm, after)
            evidence_sets[layout] = [digest_object(values["references"]), digest_object(values["history"]), digest_object(values["diagnostics"]), digest_object(values["retrieval"])]
            metrics[layout] = layout_metrics(prm, tokenizer, before, after, layout)
            layout_totals[layout].append(metrics[layout])
        if len({json.dumps(value, sort_keys=True) for value in evidence_sets.values()}) != 1:
            raise AssertionError(f"{fixture['fixture_id']} selected evidence blocks differ by layout")
        case = {
            "fixture_id": fixture["fixture_id"],
            "event_class": meta["event_class"],
            "scope_class": scope_class,
            "provenance_class": meta["provenance_class"],
            "observed_real": False,
            "target_fields_read": [],
            "future_target_accessed": False,
            "source_snapshot_id": meta["pre_edit_snapshot_id"],
            "active_file_content_sha256": meta["active_file_content_sha256"],
            "path": context.path,
            "source_package": meta["source_package"],
            "source_workspace_revision": meta["source_workspace_revision"],
            "document_eol": context.document_eol,
            "range_adapter": meta["range_adapter"],
            "cursor_adapter": meta["cursor_adapter"],
            "region_old": list(context.region_old),
            "cursor": context.cursor.to_dict(),
            "replacement_range": context.replacement_range.to_dict(),
            "selected_evidence_ids": meta["selected_evidence_ids"],
            "provider_status": meta["provider_status"],
            "unavailable_provider_reasons": meta["unavailable_provider_reasons"],
            "evicted_event_ids": meta["evicted_event_ids"],
            "pair": pair,
            "layout_order": {name: list(order) for name, order in LAYOUTS.items()},
            "evidence_block_digests_by_layout": evidence_sets,
            "metrics": metrics,
            "output_contract": {
                "task_contract_sha256": sha256_text(prm.TASK_CONTRACT),
                "generation_boundary": prm.GENERATION_BOUNDARY,
                "terminal": prm.TERMINAL,
                "target_in_rendered_input": False,
            },
        }
        per_case.append(case)
        aggregate_inputs.setdefault(meta["event_class"], []).append(case)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(values)
        return {
            "cases": count,
            "before_prompt_token_count": sum(v["before_prompt_token_count"] for v in values) / count,
            "after_prompt_token_count": sum(v["after_prompt_token_count"] for v in values) / count,
            "token_lcp": sum(v["token_lcp"] for v in values) / count,
            "recomputed_prompt_tokens": sum(v["recomputed_prompt_tokens"] for v in values) / count,
            "lcp_fraction_after": sum(v["lcp_fraction_after"] for v in values) / count,
        }

    aggregate_by_layout = {layout: summarize(values) for layout, values in layout_totals.items()}
    aggregate_by_event_class: dict[str, Any] = {}
    for event_class, cases in sorted(aggregate_inputs.items()):
        aggregate_by_event_class[event_class] = {
            layout: summarize([case["metrics"][layout] for case in cases])
            for layout in LAYOUTS
        }
    winner = max(
        LAYOUTS,
        key=lambda name: (aggregate_by_layout[name]["token_lcp"], -aggregate_by_layout[name]["recomputed_prompt_tokens"], name == "file-prefix-first"),
    )
    return {
        "schema_version": "sepalith.campaign.prm06.layout-preparation.v1",
        "task": "PRM-06",
        "provenance": {
            "fixture_package": str(FIXTURE_PATH),
            "fixture_package_sha256": sha256_bytes(FIXTURE_PATH.read_bytes()),
            "fixture_package_observed_real": package.get("coverage", {}).get("observed_real_event_fixtures", 0),
            "fixture_package_synthetic_functional": package.get("coverage", {}).get("synthetic_functional_fixtures", len(fixtures)),
            "target_fields_read": [],
            "future_target_accessed": False,
            "pair_policy": "synthetic before/after prediction-time states; unavailable providers stay empty; no target text is used",
        },
        "source_roles": {
            "plan_root": str(PLAN_ROOT),
            "execution_root": str(EXEC_ROOT),
            "renderer_module": str(EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"),
            "renderer_id": prm.RENDERER_ID,
            "schema_version": prm.SCHEMA_VERSION,
        },
        "tokenizer": {
            "path": str(TOKENIZER_PATH),
            "revision": prm.TOKENIZER_REVISION,
            "tokenizer_json_sha256": prm.TOKENIZER_JSON_SHA256,
            "mode": "HF AutoTokenizer encode(add_special_tokens=False, split_special_tokens=True)",
            "manual_bos_id": prm.BOS_ID,
            "terminal": prm.TERMINAL,
            "eos_id": prm.EOS_ID,
            "wrapper_policy": "metrics use prompt IDs without BOS; production rows add BOS exactly once and EOS after target",
        },
        "layouts": {
            name: {
                "order": list(order),
                "same_evidence_and_contract": True,
                "legacy_control": name == "file-prefix-first",
            }
            for name, order in LAYOUTS.items()
        },
        "case_count": len(per_case),
        "event_class_count": len(aggregate_by_event_class),
        "event_classes": sorted(aggregate_by_event_class),
        "per_case": per_case,
        "aggregate_by_layout": aggregate_by_layout,
        "aggregate_by_event_class": aggregate_by_event_class,
        "provisional_token_overlap_nomination": winner,
        "nomination_status": "CPU token-overlap provisional; final choice requires PRE-07 acceptance and root pinned-server timing",
        "legacy_fallback": "file-prefix-first / zeta2-prm03-v1 frozen order",
        "measurement_limits": [
            "All 11 source fixtures and all 10 event classes are synthetic functional coverage; observed real event coverage is zero in the bounded audited package.",
            "Prompt token LCP and recomputed prompt-token counts are context-budget diagnostics, not wall-clock latency or model-quality evidence.",
            "No model, server, CUDA context, cloud provider, or target label was accessed.",
            "Definition prior bytes, diagnostic refresh output, prior anchor selection, and evicted history payloads are unavailable; those pairs are held constant and reported explicitly.",
        ],
        "report_digest_excluding_self": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    report["report_digest_excluding_self"] = digest_object({k: v for k, v in report.items() if k != "report_digest_excluding_self"})
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
