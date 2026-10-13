#!/usr/bin/env python3
"""Replay frozen DAT-10 roxygen candidates against complete TRAIN sources.

This is a source-support review only.  It never opens DEV/final payloads,
never writes source text, and never changes an admission or training file.
The frozen candidate packet supplies the structured row; the source walk is
re-run on the normalized parent file and the adapter is then run with that
complete file as its pre-edit snapshot.  The output therefore distinguishes
the old bounded synthetic context from a verified full-source reconstruction.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
OUT = PLAN / "docs/campaign/work/lead/r2-source-walk-support-review-v1"
RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-source-walk-support-review.json"
SHARD_ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1")
PRIOR = PLAN / "docs/campaign/work/lead/r2-source-walk-independent-review-v1"
SOURCE_SHA = "0d70ccc40a716a8a27cb508e39a16c9a5119a9bc9d0b5b8ee8362aa4b67f6193"
TOKENIZER_ROWS = "structured-materialization-v1/token-audit/candidate-token-rows.jsonl"
PACKET_ROWS = "structured-materialization-v1/candidate-packets.jsonl"

TRAIN_SOURCE = Path("/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl")

TRAINING_DIR = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training")
SYNTHETIC_DIR = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/synthetic-data")
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import campaign_admission_structured as adapter  # noqa: E402
import campaign_structured_batch as batch  # noqa: E402
import campaign_token_audit as token_audit  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location("dat10_support_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module_spec_missing:" + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


roxygen = load_module("roxygen", SYNTHETIC_DIR / "roxygen_drafting.py")
scenarios = load_module("scenarios", SYNTHETIC_DIR / "scenarios.py")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def boundary(row: dict[str, Any]) -> dict[str, Any]:
    return batch.boundary(row)


def packet_raw(packet: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Reconstruct the structured roxygen row from the frozen packet.

    The packet's result is already the output of the source constructor.  It
    intentionally contains no raw source payload, so this reconstruction
    copies only structured fields and the immutable package/path identity.
    Exact raw line identity remains pinned by the prior independent review.
    """
    result = packet["result"]
    context = result["context"]
    parent = result.get("provenance", {}).get("parent_identity", {})
    package = packet["row_ref"].get("package_id") or parent.get("package")
    raw = {
        "prefix": list(context.get("prefix", [])),
        "region_old": [""],
        "cursor_idx": 0,
        "region_new": list(result.get("target_body", [])),
        "suffix": list(context.get("suffix_lines", [])),
        "event_diff": "",
        "family": packet["family"],
        "package": package,
        "path": context["path"],
    }
    return raw, str(package)


def structural_projection(context: dict[str, Any]) -> dict[str, Any]:
    """Context fields whose equality is independent of absolute file line."""
    return {
        key: context.get(key)
        for key in (
            "schema_version", "path", "prefix", "suffix_lines", "region_old",
            "cursor", "history", "diagnostics", "retrieval", "scope_mode",
            "scope_lines", "document_eol", "selected_references",
        )
    }


def compare_frozen_context(full: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Compare source geometry while allowing full-source context enrichment.

    The old packet was selected from a bounded synthetic envelope.  A
    complete source can validly add lines before and after that envelope, so
    exact prefix/suffix equality would mistake useful context recovery for a
    failure.  The immutable path/region/cursor/history fields must remain
    equal; the captured envelope must remain anchored at the prediction
    region when it is present.
    """
    fixed_keys = (
        "schema_version", "path", "region_old", "cursor", "history",
        "diagnostics", "retrieval", "scope_mode", "scope_lines",
        "selected_references",
    )
    fixed_match = all(full.get(key) == frozen.get(key) for key in fixed_keys)
    frozen_prefix = list(frozen.get("prefix", []))
    full_prefix = list(full.get("prefix", []))
    frozen_suffix = list(frozen.get("suffix_lines", []))
    full_suffix = list(full.get("suffix_lines", []))
    prefix_anchor = not frozen_prefix or (len(full_prefix) >= len(frozen_prefix) and full_prefix[-len(frozen_prefix):] == frozen_prefix)
    suffix_anchor = not frozen_suffix or (len(full_suffix) >= len(frozen_suffix) and full_suffix[:len(frozen_suffix)] == frozen_suffix)
    exact = fixed_match and full_prefix == frozen_prefix and full_suffix == frozen_suffix
    return {
        "fixed_fields_match": fixed_match,
        "prefix_anchor_match": prefix_anchor,
        "suffix_anchor_match": suffix_anchor,
        "structural_anchor_match": fixed_match and prefix_anchor and suffix_anchor,
        "exact_window_match": exact,
        "full_prefix_lines": len(full_prefix),
        "frozen_prefix_lines": len(frozen_prefix),
        "full_suffix_lines": len(full_suffix),
        "frozen_suffix_lines": len(frozen_suffix),
        "document_eol_match": full.get("document_eol") == frozen.get("document_eol"),
        "window_relation": "exact" if exact else "full_source_enriched_or_budget_selected" if fixed_match and prefix_anchor and suffix_anchor else "anchor_mismatch",
    }


def length_bucket(lengths: dict[str, Any]) -> str:
    response = int(lengths.get("response_with_terminal_eos", 0))
    if response <= 192:
        return "response_le_192"
    if response <= 512:
        return "response_193_512"
    if response <= 1024:
        return "response_513_1024"
    return "response_gt_1024"


def locate_inverse_target(source_lines: list[str], target: list[str], suffix: list[str]) -> list[dict[str, int]]:
    """Locate the mined after-state block and its function suffix.

    Roxygen mining reads the normalized file after its documentation block is
    present, while the scenario edit is an insertion before that block.  A
    unique target+suffix occurrence therefore supplies a deterministic,
    complete before-state: replace the target block with one physical blank
    anchor and retain the optional blank already present before the function.
    """
    positions: list[dict[str, int]] = []
    if not target or not suffix:
        return positions
    for start in range(0, len(source_lines) - len(target) + 1):
        if source_lines[start:start + len(target)] != target:
            continue
        after_target = start + len(target)
        gap = 0
        while after_target + gap < len(source_lines) and source_lines[after_target + gap] == "":
            gap += 1
        if source_lines[after_target + gap:after_target + gap + len(suffix)] == suffix:
            # Tree-sitter omits blank lines from its child list, so the
            # constructor permits one or more physical blanks between the
            # roxygen block and the function.  Collapse that gap to the
            # single protocol anchor during inverse reconstruction.
            positions.append({"target_start_line": start, "blank_after_target": gap})
    return positions


def load_candidates() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load only structured packet/token metadata, never source text."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_by_id: dict[str, dict[str, Any]] = {}
    input_files: list[dict[str, Any]] = []
    for shard_no in range(5):
        shard = SHARD_ROOT / f"shard-{shard_no:04d}"
        packet_path = shard / PACKET_ROWS
        token_path = shard / TOKENIZER_ROWS
        packet_count = token_count = 0
        token_stream = token_path.open(encoding="utf-8")
        try:
            for line in token_stream:
                item = json.loads(line)
                token_count += 1
                # Keep no token IDs or rendered source text in memory.
                token_by_id[str(item["row"]["id"])] = {
                    "context": item["context"],
                    "lengths": item["lengths"],
                    "prompt_sha256": item["prompt_sha256"],
                    "target_sha256": item["target_sha256"],
                    "source_provenance": {
                        "source_snapshot_is_simulated": item.get("source_provenance", {}).get("source_snapshot_is_simulated"),
                        "support_status": item.get("source_provenance", {}).get("support_status"),
                        "history_replay": item.get("source_provenance", {}).get("history_replay"),
                    },
                }
        finally:
            token_stream.close()
        packet_stream = packet_path.open(encoding="utf-8")
        try:
            for line in packet_stream:
                packet = json.loads(line)
                packet_count += 1
                validation = packet["validation"]
                source_path = str(validation["source_path"])
                rid = str(packet["row_ref"]["row_id"])
                token = token_by_id.get(rid)
                by_source[source_path].append({
                    "packet": packet,
                    "token": token,
                    "shard": shard_no,
                })
        finally:
            packet_stream.close()
        input_files.extend([
            {"kind": "candidate_packets", "path": str(packet_path), "rows": packet_count, "sha256": sha_file(packet_path)},
            {"kind": "candidate_token_rows", "path": str(token_path), "rows": token_count, "sha256": sha_file(token_path)},
        ])
    return by_source, token_by_id, input_files


def license_check(license_path: str, expected: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if license_path in cache:
        return cache[license_path]
    path = Path(license_path)
    result: dict[str, Any] = {"path": license_path, "expected_sha256": expected.get("sha256")}
    try:
        content = path.read_bytes()
        actual = sha_bytes(content)
        result["actual_sha256"] = actual
        result["exists"] = True
        result["hash_match"] = actual == expected.get("sha256")
        text = content.decode("utf-8")
        fields = [line for line in text.splitlines() if line.startswith(("License:", "License_restricts_use:", "License_is_FOSS:"))]
        result["license_field_present"] = any(line.startswith("License:") for line in fields)
        result["restricted"] = any(line in ("License_restricts_use: yes", "License_is_FOSS: no") for line in fields)
        result["field_count"] = len(fields)
        if not result["hash_match"]:
            result["reason"] = "license_hash_mismatch"
        elif not result["license_field_present"]:
            result["reason"] = "license_field_missing"
        elif result["restricted"]:
            result["reason"] = "source_license_restriction_requires_review"
        else:
            result["reason"] = None
    except FileNotFoundError:
        result.update({"exists": False, "reason": "license_file_missing"})
    except (OSError, UnicodeError) as error:
        result.update({"exists": False, "reason": "license_read_error:" + type(error).__name__})
    cache[license_path] = result
    return result


def replay_source(source_path: str, rows: list[dict[str, Any]], licenses: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay all candidates from one complete normalized source file."""
    source = Path(source_path)
    source_meta: dict[str, Any] = {
        "source_path": source_path,
        "candidate_rows": len(rows),
        "packages": sorted({str(item["packet"]["row_ref"].get("package_id")) for item in rows}),
        "status": "repair_required",
        "constructor_rows": None,
        "parse_error": None,
        "source_exists": source.is_file(),
        "source_sha256": None,
        "expected_source_sha256": sorted({str(item["packet"]["validation"].get("source_sha256")) for item in rows}),
    }
    row_results: list[dict[str, Any]] = []
    source_bytes: bytes | None = None
    source_text: str | None = None
    source_lines: list[str] | None = None
    source_eol = "lf"
    mined: list[dict[str, Any]] = []
    by_boundary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_error: str | None = None
    if not source.is_file():
        source_error = "source_path_missing"
    else:
        try:
            source_bytes = source.read_bytes()
            source_meta["source_bytes"] = len(source_bytes)
            source_meta["source_sha256"] = sha_bytes(source_bytes)
            if source_meta["source_sha256"] not in source_meta["expected_source_sha256"]:
                source_error = "source_hash_mismatch"
            source_text = source_bytes.decode("utf-8")
            if "\r" in source_text.replace("\r\n", ""):
                source_error = source_error or "unsupported_mixed_or_lone_cr"
            source_eol = "crlf" if "\r\n" in source_text else "lf"
            source_lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            tree = scenarios.parser.parse(source_bytes)
            if tree.root_node.has_error:
                source_error = source_error or "normalized_parent_R_parse_error"
            package = str(rows[0]["packet"]["row_ref"].get("package_id"))
            stats = Counter()
            roxygen.extract_file(package, source.name, source_bytes, mined, stats)
            source_meta["constructor_rows"] = len(mined)
            source_meta["constructor_stats"] = dict(stats)
            for actual in mined:
                by_boundary[canonical(boundary(actual))].append(actual)
        except UnicodeDecodeError:
            source_error = source_error or "source_not_utf8"
        except (OSError, ValueError, RuntimeError) as error:
            source_error = source_error or "source_reconstruction_error:" + type(error).__name__
    source_meta["source_error"] = source_error

    for item in rows:
        packet = item["packet"]
        token = item["token"]
        ref0 = packet["row_ref"]
        result0 = packet["result"]
        result_prov = result0.get("provenance", {})
        raw, package = packet_raw(packet)
        expected_boundary = canonical(boundary(raw))
        matches = by_boundary.get(expected_boundary, [])
        checks: dict[str, Any] = {
            "raw_boundary_reconstructed_from_frozen_packet": True,
            "fresh_constructor_boundary_match_count": len(matches),
            "full_snapshot_attempted": source_text is not None and source_error is None,
            "normalized_after_target_match_count": None,
            "inverse_target_reconstruction_verified": False,
            "full_snapshot_derivation": "none",
            "history_expected": False,
            "history_replay_required": False,
            "frozen_snapshot_was_simulated": result_prov.get("source_snapshot_is_simulated") is True,
            "frozen_support_status_pending_source_validator": result_prov.get("support_status") == "structural_conversion_pending_source_validator",
        }
        reasons: list[str] = []
        if token is None:
            reasons.append("token_candidate_missing")
        validation = packet["validation"]
        license_evidence = validation.get("license_evidence", {})
        license_result = license_check(str(license_evidence.get("path", "")), license_evidence, licenses)
        checks["license_revalidated"] = license_result.get("reason") is None
        if license_result.get("reason") is not None:
            reasons.append(str(license_result["reason"]))
        if source_error is not None:
            reasons.append(source_error)
        if not source_error and len(matches) == 0:
            reasons.append("constructor_boundary_not_reproduced")
        elif not source_error and len(matches) > 1:
            reasons.append("constructor_boundary_ambiguous")

        full_result: dict[str, Any] | None = None
        full_context: dict[str, Any] | None = None
        selection: dict[str, Any] | None = None
        inverse_positions: list[dict[str, int]] = []
        derived_text: str | None = None
        derived_hash: str | None = None
        derived_path = f"derived:normalized-after/{source_path}"
        if not reasons and source_text is not None and source_bytes is not None and source_lines is not None:
            inverse_positions = locate_inverse_target(source_lines, raw["region_new"], raw["suffix"])
            checks["normalized_after_target_match_count"] = len(inverse_positions)
            if len(inverse_positions) != 1:
                reasons.append("normalized_after_target_suffix_not_unique")
            else:
                position = inverse_positions[0]
                start = position["target_start_line"]
                removed = len(raw["region_new"]) + position["blank_after_target"]
                pre_lines = source_lines[:start] + [""] + source_lines[start + removed:]
                derived_text = ("\r\n" if source_eol == "crlf" else "\n").join(pre_lines)
                derived_hash = sha_bytes(derived_text.encode("utf-8", "surrogatepass"))
                checks["inverse_target_reconstruction_verified"] = True
                checks["full_snapshot_derivation"] = "normalized_after_target_removed_with_blank_anchor"
        if not reasons and derived_text is not None and derived_hash is not None:
            ref = copy.deepcopy(ref0)
            ref["package_id"] = package
            ref["source_snapshot_text"] = derived_text
            ref["source_snapshot_sha256"] = derived_hash
            ref["source_snapshot_provenance_path"] = derived_path
            ref["cursor_encoding"] = "scenario_char_offset"
            ref["workspace_revision_before"] = derived_hash
            ref["target_line"] = inverse_positions[0]["target_start_line"]
            ref["parent_identity"] = {
                "package": package,
                "path": raw["path"],
                "normalized_after_source_path": source_path,
                "normalized_after_source_sha256": sha_bytes(source_bytes),
                "pre_edit_derivation": "remove_exact_mined_roxygen_target_retain_physical_blank_anchor",
            }
            try:
                full_result = adapter.convert_structured(raw, ref)
                checks["full_snapshot_conversion_status"] = full_result.get("status")
                if full_result.get("status") != "converted":
                    reasons.append("full_snapshot_conversion_failed:" + str(full_result.get("reason")))
                else:
                    full_prov = full_result.get("provenance", {})
                    checks.update({
                        "full_snapshot_is_simulated": full_prov.get("source_snapshot_is_simulated"),
                        "full_snapshot_path_matches": full_prov.get("source_snapshot_path") == derived_path,
                        "full_snapshot_hash_matches": full_prov.get("source_snapshot_sha256") == derived_hash,
                        "post_edit_geometry_verified": full_prov.get("post_edit_geometry_verified") is True,
                        "full_history_replayed": full_prov.get("history_replay", {}).get("replayed") is True,
                    })
                    if full_prov.get("source_snapshot_is_simulated") is not False:
                        reasons.append("full_snapshot_unexpectedly_simulated")
                    if full_prov.get("source_snapshot_path") != derived_path:
                        reasons.append("full_snapshot_path_mismatch")
                    if full_prov.get("source_snapshot_sha256") != derived_hash:
                        reasons.append("full_snapshot_hash_mismatch")
                    if full_prov.get("post_edit_geometry_verified") is not True:
                        reasons.append("post_edit_geometry_unverified")
                    # This exercises the byte/UTF-16 range and post-edit hash
                    # invariant without retaining the applied source text.
                    batch.apply_result(full_result)
                    checks["full_snapshot_application_verified"] = True
                    full_context_obj, selection = token_audit.selected_context(full_result)
                    full_context = full_context_obj.to_dict()
                    checks["selection_support_revalidation_required"] = bool(selection.get("support_revalidation_required"))
                    if checks["selection_support_revalidation_required"]:
                        reasons.append("full_selection_omitted_captured_context")
                    if token is None:
                        pass
                    else:
                        frozen_context = token["context"]
                        context_comparison = compare_frozen_context(full_context, frozen_context)
                        checks["frozen_context_comparison"] = context_comparison
                        if not context_comparison["structural_anchor_match"]:
                            reasons.append("frozen_context_structural_anchor_mismatch")
            except (AssertionError, KeyError, OSError, UnicodeError, ValueError, adapter.ProtocolError) as error:
                checks["full_snapshot_application_verified"] = False
                reasons.append("full_snapshot_reconstruction_error:" + type(error).__name__ + ":" + str(error))
        elif not source_error and len(matches) == 1:
            # This branch is only reachable when a previous row-level issue
            # (license/token) prevents adapter replay; preserve the constructor
            # evidence while keeping the row in the repair queue.
            checks["full_snapshot_skipped_due_to_other_repair"] = True

        status = "supported_source_replay_pending_root_admission" if not reasons else "repair_required"
        projection = structural_projection(full_context) if full_context is not None else None
        frozen_line = result_prov.get("target_start_line")
        full_line = (full_result or {}).get("provenance", {}).get("target_start_line") if full_result else None
        row_results.append({
            "row_id": ref0["row_id"],
            "shard": item["shard"],
            "family": packet["family"],
            "group_id": ref0.get("group_id"),
            "package_id": package,
            "source_file": ref0.get("file"),
            "source_line": ref0.get("line"),
            "source_path": source_path,
            "source_sha256": validation.get("source_sha256"),
            "raw_line_sha256": ref0.get("raw_line_sha256"),
            "status": status,
            "reasons": reasons,
            "repair": bool(reasons),
            "checks": checks,
            "license": {
                "path": license_result.get("path"),
                "expected_sha256": license_result.get("expected_sha256"),
                "actual_sha256": license_result.get("actual_sha256"),
                "reason": license_result.get("reason"),
            },
            "constructor": {
                "match_count": len(matches),
                "target_sha256": result_prov.get("target_sha256"),
                "target_line_frozen_synthetic": frozen_line,
                "target_line_full_snapshot": full_line,
                "absolute_line_relocation_expected": frozen_line != full_line if full_line is not None else None,
            },
            "context": {
                "frozen_snapshot_simulated": result_prov.get("source_snapshot_is_simulated"),
                "frozen_history_replay": result_prov.get("history_replay"),
                "full_snapshot_sha256": (full_result or {}).get("provenance", {}).get("source_snapshot_sha256"),
                "full_snapshot_is_simulated": (full_result or {}).get("provenance", {}).get("source_snapshot_is_simulated"),
                "full_snapshot_derivation": checks.get("full_snapshot_derivation"),
                "full_projection_sha256": sha_bytes(canonical(projection).encode()) if projection is not None else None,
                "history_support": "not_applicable_no_observed_roxygen_event",
            },
            "lengths": token.get("lengths") if token else None,
            "prompt_sha256": token.get("prompt_sha256") if token else None,
            "target_sha256_token_row": token.get("target_sha256") if token else None,
            "admission": "review_only_unadmitted",
        })
    supported_count = sum(1 for row in row_results if row["status"].startswith("supported"))
    if supported_count == len(row_results):
        source_meta["status"] = "supported_source_replay_pending_root_admission"
    elif supported_count:
        source_meta["status"] = "mixed_supported_and_repair_required"
    else:
        source_meta["status"] = "repair_required"
    source_meta["supported_rows"] = supported_count
    source_meta["repair_rows"] = len(row_results) - source_meta["supported_rows"]
    return row_results, source_meta


def build_receipt(summary: dict[str, Any], artifacts: list[dict[str, Any]], input_files: list[dict[str, Any]], source_inventory_path: Path) -> dict[str, Any]:
    return {
        "schema": "DAT-10-source-walk-support-review-v1",
        "status": "complete_review_only",
        "owner": "/root/data_expansion",
        "decision": "source_support_replay_only_root_admission_pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "candidate_rows": summary["candidate_rows"],
            "source_walk_shards": [f"shard-{i:04d}" for i in range(5)],
            "families": ["roxygen_drafting"],
            "train_only": True,
            "opened_dev_or_final_payload": False,
            "admitted_rows": 0,
            "trained_rows": 0,
            "source_content_written": False,
        },
        "inputs": {
            "source_file": {"path": str(TRAIN_SOURCE), "sha256": SOURCE_SHA, "content_written": False},
            "prior_independent_review": {
                "path": str(PRIOR),
                "source_line_replay": str(PRIOR / "source-line-replay.json"),
                "source_line_replay_sha256": sha_file(PRIOR / "source-line-replay.json"),
                "audit_ledger": str(PRIOR / "audit-ledger.jsonl"),
                "audit_ledger_sha256": sha_file(PRIOR / "audit-ledger.jsonl"),
            },
            "frozen_candidate_inputs": input_files,
        },
        "method": {
            "source_constructor": str(SYNTHETIC_DIR / "roxygen_drafting.py"),
            "source_constructor_sha256": sha_file(SYNTHETIC_DIR / "roxygen_drafting.py"),
            "adapter": str(TRAINING_DIR / "campaign_admission_structured.py"),
            "adapter_sha256": sha_file(TRAINING_DIR / "campaign_admission_structured.py"),
            "reconstruction": "fresh tree-sitter roxygen constructor boundary match, unique normalized after-state target+suffix lookup, deterministic target removal with physical blank anchor, then adapter replay on the complete derived before-state",
            "derived_snapshot_policy": "normalized source is the mined after-state; it is never labeled as an observed pre-edit buffer. Derived pre-edit hashes and normalized after-state hashes remain separate in the ledger parent identity.",
            "history": "roxygen rows have empty event_diff; absence of observed edit history is recorded as not_applicable and is not treated as a repair",
            "length_policy": "preserve frozen token lengths; no target truncation; >1024 response rows retained in explicit queue",
        },
        "summary": summary,
        "artifacts": artifacts,
        "source_inventory": {"path": str(source_inventory_path), "sha256": sha_file(source_inventory_path)},
        "constraints": {"cpu_threads_max": 2, "cuda": False, "ssh": False, "cloud": False, "dev_payloads": False, "final_payloads": False},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    by_source, token_by_id, input_files = load_candidates()
    write_json(OUT / "status.json", {
        "schema": "DAT-10-source-walk-support-review-status-v1",
        "owner": "/root/data_expansion",
        "status": "replay_running",
        "candidate_rows": sum(len(rows) for rows in by_source.values()),
        "unique_normalized_sources": len(by_source),
        "source_constructor": str(SYNTHETIC_DIR / "roxygen_drafting.py"),
        "source_sha256": SOURCE_SHA,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "constraints": {"cpu_threads": 2, "cuda": False, "dev_payloads": False, "final_payloads": False, "admission": False, "training": False},
    })
    total_rows = sum(len(rows) for rows in by_source.values())
    all_results: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    licenses: dict[str, dict[str, Any]] = {}
    for index, (source_path, rows) in enumerate(sorted(by_source.items()), 1):
        row_results, source_meta = replay_source(source_path, rows, licenses)
        all_results.extend(row_results)
        inventories.append(source_meta)
        if index == 1 or index % 100 == 0 or index == len(by_source):
            supported = sum(row["status"].startswith("supported") for row in all_results)
            print(json.dumps({"stage": "source_support_replay", "sources_done": index, "sources_total": len(by_source), "rows_done": len(all_results), "rows_total": total_rows, "supported": supported, "repair": len(all_results) - supported, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            write_json(OUT / "progress.json", {
                "schema": "DAT-10-source-walk-support-review-progress-v1",
                "status": "replay_running",
                "sources_done": index,
                "sources_total": len(by_source),
                "rows_done": len(all_results),
                "rows_total": total_rows,
                "supported": supported,
                "repair": len(all_results) - supported,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    all_results.sort(key=lambda row: row["row_id"])
    inventories.sort(key=lambda row: row["source_path"])
    write_jsonl(OUT / "support-ledger.jsonl", all_results)
    supported = [row for row in all_results if row["status"].startswith("supported")]
    repairs = [row for row in all_results if row["repair"]]
    write_jsonl(OUT / "support-verified-candidate-ids.jsonl", [
        {"row_id": row["row_id"], "family": row["family"], "group_id": row["group_id"], "package_id": row["package_id"], "source_path": row["source_path"], "source_line": row["source_line"], "status": row["status"], "admission": "review_only_unadmitted"}
        for row in supported
    ])
    write_jsonl(OUT / "repair-queue.jsonl", repairs)
    write_jsonl(OUT / "source-inventory.jsonl", inventories)

    evidence: dict[str, Any] = {"schema": "DAT-10-source-walk-support-evidence-v1", "representatives": {}, "counts": {}}
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_results:
        reasons = row["reasons"] or ["supported_source_replay"]
        for reason in reasons:
            by_reason[reason].append({
                "row_id": row["row_id"], "shard": row["shard"], "family": row["family"], "package_id": row["package_id"], "source_path": row["source_path"], "source_line": row["source_line"], "reasons": row["reasons"], "checks": row["checks"], "constructor": row["constructor"], "license": row["license"], "lengths": row["lengths"],
            })
    for reason, items in sorted(by_reason.items()):
        evidence["counts"][reason] = len(items)
        evidence["representatives"][reason] = items[:5]
    write_json(OUT / "representative-evidence.json", evidence)

    token_counts = Counter()
    seq_counts = Counter()
    long_rows: list[dict[str, Any]] = []
    for row in all_results:
        lengths = row.get("lengths") or {}
        token_counts[length_bucket(lengths)] += 1
        seq = int(lengths.get("sequence", 0))
        seq_counts["sequence_le_2048" if seq <= 2048 else "sequence_2049_4096" if seq <= 4096 else "sequence_gt_4096"] += 1
        if int(lengths.get("response_with_terminal_eos", 0)) > 1024:
            long_rows.append({"row_id": row["row_id"], "shard": row["shard"], "response_with_terminal_eos": lengths.get("response_with_terminal_eos"), "sequence": lengths.get("sequence"), "status": row["status"]})
    token_paths = [item for item in input_files if item["kind"] == "candidate_token_rows"]
    token_plan = {
        "schema": "DAT-10-source-walk-support-token-export-plan-v1",
        "status": "plan_only_root_admission_required",
        "source_support_gate": "use only support-verified-candidate-ids after root review; repair queue remains out",
        "candidate_rows": len(all_results),
        "source_supported_rows": len(supported),
        "repair_rows": len(repairs),
        "frozen_token_row_inputs": token_paths,
        "source_support_ledger": {"path": str(OUT / "support-ledger.jsonl"), "sha256": sha_file(OUT / "support-ledger.jsonl")},
        "target_policy": {"max_target_tokens": None, "truncate": False, "preserve_complete_targets": True, "long_response_threshold": 1024},
        "sequence_policy": {"max_sequence_tokens": 4096, "over_limit_action": "retain_in_long_context_queue; no truncation"},
        "length_buckets": dict(sorted(token_counts.items())),
        "sequence_buckets": dict(sorted(seq_counts.items())),
        "long_context_queue": {"path": str(OUT / "long-context-queue.jsonl"), "rows": len(long_rows)},
        "reproducible_steps": [
            "verify receipt input hashes and support-ledger row identities",
            "join support-verified IDs to frozen candidate-token rows by row_id",
            "retain complete target token sequences; route response_with_terminal_eos >1024 to long-context profiling",
            "apply root global TRAIN/license/duplicate/admission gates before any training export",
        ],
        "admitted_for_training": False,
    }
    write_jsonl(OUT / "long-context-queue.jsonl", long_rows)
    write_json(OUT / "token-export-plan.json", token_plan)

    reason_counts = Counter(reason for row in all_results for reason in (row["reasons"] or ["supported_source_replay"]))
    family_counts = Counter(row["family"] for row in all_results)
    summary = {
        "candidate_rows": len(all_results),
        "unique_row_ids": len({row["row_id"] for row in all_results}),
        "unique_normalized_sources": len(inventories),
        "supported_source_replay_pending_root_admission": len(supported),
        "repair_required": len(repairs),
        "permanent_exclusions": 0,
        "admitted_rows": 0,
        "trained_rows": 0,
        "families": dict(sorted(family_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "history": {"no_observed_event_rows": len(all_results), "history_replay_required_rows": 0},
        "frozen_synthetic_context_rows": sum(1 for row in all_results if row["checks"].get("frozen_snapshot_was_simulated")),
        "full_snapshot_rows": sum(1 for row in all_results if row["checks"].get("full_snapshot_is_simulated") is False),
        "full_snapshot_reconstruction_failures": sum(1 for row in all_results if any(reason.startswith("full_snapshot_") for reason in row["reasons"])),
        "long_response_gt_1024": len(long_rows),
        "sequence_gt_4096": seq_counts.get("sequence_gt_4096", 0),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    artifacts: list[dict[str, Any]] = []
    for path in [OUT / "support-ledger.jsonl", OUT / "support-verified-candidate-ids.jsonl", OUT / "repair-queue.jsonl", OUT / "source-inventory.jsonl", OUT / "representative-evidence.json", OUT / "long-context-queue.jsonl", OUT / "token-export-plan.json", OUT / "review_support.py"]:
        artifacts.append({"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size, "rows": len(all_results) if path.name == "support-ledger.jsonl" else len(supported) if path.name == "support-verified-candidate-ids.jsonl" else len(repairs) if path.name == "repair-queue.jsonl" else len(inventories) if path.name == "source-inventory.jsonl" else len(long_rows) if path.name == "long-context-queue.jsonl" else None})
    write_json(OUT / "summary.json", summary)
    artifacts.append({"path": str(OUT / "summary.json"), "sha256": sha_file(OUT / "summary.json"), "bytes": (OUT / "summary.json").stat().st_size})
    receipt = build_receipt(summary, artifacts, input_files, OUT / "source-inventory.jsonl")
    write_json(RECEIPT, receipt)
    write_json(OUT / "status.json", {"schema": "DAT-10-source-walk-support-review-status-v1", "owner": "/root/data_expansion", "status": "complete_review_only", "summary": summary, "receipt": str(RECEIPT), "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"stage": "complete", "summary": summary, "receipt": str(RECEIPT)}), flush=True)


if __name__ == "__main__":
    main()
