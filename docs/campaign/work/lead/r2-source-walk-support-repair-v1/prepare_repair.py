#!/usr/bin/env python3
"""Prepare complete before-state/context profiles for DAT-10 repairs.

The input is the immutable support-review ledger and its frozen candidate
packets.  For each context-budget repair, this script replays the normalized
after-state, removes exactly the mined roxygen target, retains one physical
blank anchor, and profiles the resulting complete before-state at increasing
source-selection budgets.  It writes hashes and geometry only; source text,
DEV/final payloads, admissions, and training artifacts are never written.
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
OUT = PLAN / "docs/campaign/work/lead/r2-source-walk-support-repair-v1"
SUPPORT_OUT = PLAN / "docs/campaign/work/lead/r2-source-walk-support-review-v1"
RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-source-walk-support-repair-preparation.json"
SHARD_ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1")
SOURCE_SUPPORT_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-source-walk-support-review.json"
TOKENIZER_ROWS = "structured-materialization-v1/token-audit/candidate-token-rows.jsonl"
PACKET_ROWS = "structured-materialization-v1/candidate-packets.jsonl"
SYNTHETIC_DIR = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/synthetic-data")
TRAINING_DIR = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training")
SOURCE_SHA = "0d70ccc40a716a8a27cb508e39a16c9a5119a9bc9d0b5b8ee8362aa4b67f6193"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
import campaign_admission_structured as adapter  # noqa: E402
import campaign_token_audit as token_audit  # noqa: E402


def load_review_module():
    path = SUPPORT_OUT / "review_support.py"
    spec = importlib.util.spec_from_file_location("dat10_support_repair_review", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("support_review_module_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


review = load_review_module()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    support_ledger = SUPPORT_OUT / "support-ledger.jsonl"
    support_receipt = json.loads(SOURCE_SUPPORT_RECEIPT.read_text())
    receipt_pin = next(item["sha256"] for item in support_receipt["artifacts"] if item["path"] == str(support_ledger))
    if sha_file(support_ledger) != receipt_pin:
        raise ValueError("support_ledger_pin_mismatch")
    repairs: dict[str, dict[str, Any]] = {}
    for line in support_ledger.open(encoding="utf-8"):
        row = json.loads(line)
        if "full_selection_omitted_captured_context" in row.get("reasons", []):
            repairs[row["row_id"]] = row
    packets: dict[str, dict[str, Any]] = {}
    input_files: list[dict[str, Any]] = [{"kind": "support_ledger", "path": str(support_ledger), "rows": len(repairs), "sha256": sha_file(support_ledger)}]
    for shard_no in range(5):
        shard = SHARD_ROOT / f"shard-{shard_no:04d}"
        packet_path = shard / PACKET_ROWS
        token_path = shard / TOKENIZER_ROWS
        for line in packet_path.open(encoding="utf-8"):
            packet = json.loads(line)
            rid = str(packet["row_ref"]["row_id"])
            if rid in repairs:
                packets[rid] = packet
        input_files.append({"kind": "candidate_packets", "path": str(packet_path), "sha256": sha_file(packet_path), "scope_rows": sum(1 for rid in repairs if rid in packets)})
        # Lengths/context are read only for repair IDs; token IDs and rendered
        # text are discarded immediately after the small metadata copy.
        for line in token_path.open(encoding="utf-8"):
            item = json.loads(line)
            rid = str(item["row"]["id"])
            if rid in repairs:
                repairs[rid]["token"] = {
                    "context": item["context"],
                    "lengths": item["lengths"],
                    "prompt_sha256": item["prompt_sha256"],
                    "target_sha256": item["target_sha256"],
                }
        input_files.append({"kind": "candidate_token_rows", "path": str(token_path), "sha256": sha_file(token_path), "scope_rows": sum(1 for rid in repairs if "token" in repairs[rid])})
    if set(repairs) != set(packets):
        raise ValueError("repair_packet_join_incomplete")
    if any("token" not in row for row in repairs.values()):
        raise ValueError("repair_token_join_incomplete")
    return repairs, packets, input_files


def profile_context(result: dict[str, Any], frozen: dict[str, Any], budget: int | str) -> dict[str, Any]:
    if budget == "full_file":
        max_units = 1_000_000_000
    else:
        max_units = int(budget)
    context, selection = token_audit.selected_context(result, max_source_utf16=max_units)
    current = context.to_dict()
    comparison = review.compare_frozen_context(current, frozen)
    omission_count = len(selection.get("omissions", []))
    return {
        "budget": budget,
        "used_utf16_units": selection.get("used_utf16_units"),
        "required_utf16_units": selection.get("required_utf16_units"),
        "overflow": selection.get("overflow"),
        "required_overflow": selection.get("required_overflow"),
        "omission_count": omission_count,
        "support_revalidation_required": bool(selection.get("support_revalidation_required")),
        "captured_context_anchors_match": comparison["structural_anchor_match"],
        "context_window_relation": comparison["window_relation"],
        "full_prefix_lines": len(current.get("prefix", [])),
        "full_suffix_lines": len(current.get("suffix_lines", [])),
        "document_eol": current.get("document_eol"),
        "context_projection_sha256": sha_bytes(json.dumps(review.structural_projection(current), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()),
    }


def process_row(ledger: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    raw, package = review.packet_raw(packet)
    source_path = Path(ledger["source_path"])
    result0 = packet["result"]
    token = ledger["token"]
    output: dict[str, Any] = {
        "row_id": ledger["row_id"],
        "shard": ledger["shard"],
        "family": ledger["family"],
        "group_id": ledger["group_id"],
        "package_id": package,
        "source_file": ledger["source_file"],
        "source_line": ledger["source_line"],
        "source_path": str(source_path),
        "normalized_after_source_sha256": ledger["source_sha256"],
        "raw_line_sha256": ledger["raw_line_sha256"],
        "target_body_line_count": len(raw["region_new"]),
        "target_body_sha256": sha_bytes("\n".join(raw["region_new"]).encode("utf-8")),
        "target_semantics": result0.get("operation"),
        "original_review_reasons": ledger["reasons"],
        "lengths": token["lengths"],
        "prompt_sha256": token["prompt_sha256"],
        "target_sha256_token_row": token["target_sha256"],
        "status": "repair_required",
        "target_rewritten": False,
        "admission": "review_only_unadmitted",
    }
    if not source_path.is_file():
        output["repair_reasons"] = ["source_path_missing"]
        return output
    try:
        source_bytes = source_path.read_bytes()
        source_hash = sha_bytes(source_bytes)
        if source_hash != ledger["source_sha256"]:
            output["repair_reasons"] = ["normalized_after_source_hash_mismatch"]
            return output
        source_text = source_bytes.decode("utf-8")
        source_eol = "crlf" if "\r\n" in source_text else "lf"
        source_lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        positions = review.locate_inverse_target(source_lines, raw["region_new"], raw["suffix"])
        output["normalized_after_target_match_count"] = len(positions)
        if len(positions) != 1:
            output["repair_reasons"] = ["normalized_after_target_suffix_not_unique"]
            return output
        position = positions[0]
        start = position["target_start_line"]
        removed = len(raw["region_new"]) + position["blank_after_target"]
        before_lines = source_lines[:start] + [""] + source_lines[start + removed:]
        before_text = ("\r\n" if source_eol == "crlf" else "\n").join(before_lines)
        before_bytes = before_text.encode("utf-8", "surrogatepass")
        before_hash = sha_bytes(before_bytes)
        expected_before = ledger.get("context", {}).get("full_snapshot_sha256")
        output.update({
            "normalized_after_bytes": len(source_bytes),
            "derived_before_bytes": len(before_bytes),
            "derived_before_sha256": before_hash,
            "derived_before_hash_matches_support_review": before_hash == expected_before,
            "derived_before_target_line": start,
            "normalized_after_removed_line_count": removed,
            "blank_gap_after_target": position["blank_after_target"],
            "inverse_target_reconstruction_verified": True,
            "derived_before_snapshot_label": "derived:normalized-after/" + str(source_path),
        })
        ref = copy.deepcopy(packet["row_ref"])
        ref.update({
            "package_id": package,
            "source_snapshot_text": before_text,
            "source_snapshot_sha256": before_hash,
            "source_snapshot_provenance_path": output["derived_before_snapshot_label"],
            "cursor_encoding": "scenario_char_offset",
            "workspace_revision_before": before_hash,
            "target_line": start,
            "parent_identity": {
                "package": package,
                "path": raw["path"],
                "normalized_after_source_path": str(source_path),
                "normalized_after_source_sha256": source_hash,
                "pre_edit_derivation": "remove_exact_mined_roxygen_target_retain_physical_blank_anchor",
            },
        })
        converted = adapter.convert_structured(raw, ref)
        output["full_snapshot_conversion_status"] = converted.get("status")
        output["full_snapshot_support_status"] = converted.get("provenance", {}).get("support_status")
        output["full_snapshot_is_simulated"] = converted.get("provenance", {}).get("source_snapshot_is_simulated")
        output["post_edit_geometry_verified"] = converted.get("provenance", {}).get("post_edit_geometry_verified") is True
        output["full_snapshot_application_verified"] = False
        if converted.get("status") != "converted":
            output["repair_reasons"] = ["full_snapshot_conversion_failed:" + str(converted.get("reason"))]
            return output
        adapter_result = converted
        # Validate byte/UTF-16 application while discarding the resulting
        # source text immediately.  This does not alter any file.
        import campaign_structured_batch as structured_batch
        structured_batch.apply_result(adapter_result)
        output["full_snapshot_application_verified"] = True
        frozen_context = token["context"]
        profiles = []
        for budget in (6000, 8192, 16384, 32768, "full_file"):
            try:
                profiles.append(profile_context(adapter_result, frozen_context, budget))
            except (AssertionError, KeyError, OSError, UnicodeError, ValueError) as error:
                profiles.append({"budget": budget, "error": type(error).__name__ + ":" + str(error)})
        output["profiles"] = profiles
        output["full_file_context"] = profiles[-1]
        output["context_budget_repair_resolved_at"] = next((item["budget"] for item in profiles if not item.get("error") and item.get("omission_count") == 0 and item.get("captured_context_anchors_match")), None)
        output["context_budget_repair_remaining"] = output["context_budget_repair_resolved_at"] is None
        output["repair_reasons"] = []
        if output["context_budget_repair_remaining"]:
            output["repair_reasons"].append("context_budget_or_anchor_review_remaining_at_full_file")
        else:
            output["status"] = "context_budget_repaired_source_ready_root_review"
    except (OSError, UnicodeDecodeError, UnicodeError, ValueError, adapter.ProtocolError) as error:
        output["repair_reasons"] = ["source_reconstruction_error:" + type(error).__name__ + ":" + str(error)]
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repairs, packets, input_files = load_inputs()
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for rid, ledger in repairs.items():
        grouped[ledger["source_path"]].append((ledger, packets[rid]))
    write_json(OUT / "status.json", {
        "schema": "DAT-10-source-walk-support-repair-status-v1",
        "owner": "/root/data_expansion",
        "status": "repair_replay_running",
        "scope_rows": len(repairs),
        "unique_sources": len(grouped),
        "source_support_receipt": str(SOURCE_SUPPORT_RECEIPT),
        "profiles": [6000, 8192, 16384, 32768, "full_file"],
        "target_rewritten": False,
        "source_payload_written": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    results: list[dict[str, Any]] = []
    for index, source_path in enumerate(sorted(grouped), 1):
        for ledger, packet in grouped[source_path]:
            results.append(process_row(ledger, packet))
        if index == 1 or index % 100 == 0 or index == len(grouped):
            resolved = sum(1 for row in results if row.get("context_budget_repair_resolved_at") is not None)
            print(json.dumps({"stage": "repair_context_replay", "sources_done": index, "sources_total": len(grouped), "rows_done": len(results), "rows_total": len(repairs), "full_reconstruction": sum(1 for row in results if row.get("inverse_target_reconstruction_verified")), "budget_resolved": resolved, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            write_json(OUT / "progress.json", {"schema": "DAT-10-source-walk-support-repair-progress-v1", "status": "repair_replay_running", "sources_done": index, "sources_total": len(grouped), "rows_done": len(results), "rows_total": len(repairs), "full_reconstruction": sum(1 for row in results if row.get("inverse_target_reconstruction_verified")), "budget_resolved": resolved, "elapsed_seconds": round(time.monotonic() - started, 1), "updated_at": datetime.now(timezone.utc).isoformat()})
    results.sort(key=lambda row: row["row_id"])
    write_jsonl(OUT / "repair-source-ledger.jsonl", results)
    resolved = [row for row in results if row.get("context_budget_repair_resolved_at") is not None]
    unresolved = [row for row in results if row not in resolved]
    write_jsonl(OUT / "context-budget-resolved-ids.jsonl", [{"row_id": row["row_id"], "source_path": row["source_path"], "derived_before_sha256": row.get("derived_before_sha256"), "resolved_at_budget": row.get("context_budget_repair_resolved_at"), "target_rewritten": False, "admission": "review_only_unadmitted"} for row in resolved])
    write_jsonl(OUT / "context-budget-unresolved-queue.jsonl", [{"row_id": row["row_id"], "source_path": row["source_path"], "derived_before_sha256": row.get("derived_before_sha256"), "reasons": row.get("repair_reasons"), "profiles": row.get("profiles"), "target_rewritten": False} for row in unresolved])
    profile_counts: dict[str, dict[str, int]] = {}
    for budget in (6000, 8192, 16384, 32768, "full_file"):
        key = str(budget)
        values = [row.get("profiles", []) for row in results]
        selected = [next((item for item in items if item.get("budget") == budget), None) for items in values]
        profile_counts[key] = {
            "rows": len(selected),
            "profile_errors": sum(1 for item in selected if not item or item.get("error")),
            "zero_omission_rows": sum(1 for item in selected if item and not item.get("error") and item.get("omission_count") == 0),
            "captured_anchor_match_rows": sum(1 for item in selected if item and not item.get("error") and item.get("captured_context_anchors_match")),
            "support_revalidation_required_rows": sum(1 for item in selected if item and item.get("support_revalidation_required")),
        }
    reason_counts = Counter(reason for row in results for reason in (row.get("repair_reasons") or ["context_budget_repaired_source_ready_root_review"]))
    summary = {
        "scope_rows": len(results),
        "unique_row_ids": len({row["row_id"] for row in results}),
        "unique_normalized_sources": len(grouped),
        "inverse_target_reconstruction_verified": sum(1 for row in results if row.get("inverse_target_reconstruction_verified")),
        "derived_before_hash_matches_support_review": sum(1 for row in results if row.get("derived_before_hash_matches_support_review")),
        "full_snapshot_conversion_pass": sum(1 for row in results if row.get("full_snapshot_conversion_status") == "converted"),
        "full_snapshot_application_pass": sum(1 for row in results if row.get("full_snapshot_application_verified")),
        "context_budget_resolved_rows": len(resolved),
        "context_budget_unresolved_rows": len(unresolved),
        "repair_reasons": dict(sorted(reason_counts.items())),
        "target_rewritten": 0,
        "admitted_rows": 0,
        "trained_rows": 0,
        "profiles": profile_counts,
        "long_response_gt_1024": sum(1 for row in results if (row.get("lengths") or {}).get("response_with_terminal_eos", 0) > 1024),
        "sequence_gt_4096": sum(1 for row in results if (row.get("lengths") or {}).get("sequence", 0) > 4096),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_json(OUT / "repair-profile-summary.json", summary)
    artifacts = []
    for path in [OUT / "repair-source-ledger.jsonl", OUT / "context-budget-resolved-ids.jsonl", OUT / "context-budget-unresolved-queue.jsonl", OUT / "repair-profile-summary.json", OUT / "prepare_repair.py"]:
        artifacts.append({"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size, "rows": len(results) if path.name == "repair-source-ledger.jsonl" else len(resolved) if path.name == "context-budget-resolved-ids.jsonl" else len(unresolved) if path.name == "context-budget-unresolved-queue.jsonl" else None})
    receipt = {
        "schema": "DAT-10-source-walk-support-repair-preparation-v1",
        "status": "complete_review_only",
        "owner": "/root/data_expansion",
        "decision": "repair_scope_prepared_root_admission_pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"support_review_rows": 7021, "processed_rows": len(results), "anchor_mismatch_overlap": 240, "unique_sources": len(grouped), "target_rewritten": False, "source_payload_written": False, "opened_dev_or_final_payload": False},
        "inputs": {"support_receipt": {"path": str(SOURCE_SUPPORT_RECEIPT), "sha256": sha_file(SOURCE_SUPPORT_RECEIPT)}, "support_ledger": {"path": str(SUPPORT_OUT / "support-ledger.jsonl"), "sha256": sha_file(SUPPORT_OUT / "support-ledger.jsonl")}, "frozen_candidate_inputs": input_files, "source_file": {"path": "/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl", "sha256": SOURCE_SHA, "content_written": False}},
        "method": {"normalized_after_source": "pinned source_path/source_sha256 from support ledger", "inverse": "remove exact mined region_new followed by zero-or-more physical blank lines and insert one blank anchor", "adapter": str(TRAINING_DIR / "campaign_admission_structured.py"), "selection_profiles": [6000, 8192, 16384, 32768, "full_file"], "target_policy": "target body is copied only into in-memory adapter input; no target rewriting or truncation"},
        "summary": summary,
        "artifacts": artifacts,
        "constraints": {"cpu_threads_max": 2, "cuda": False, "ssh": False, "cloud": False, "dev_payloads": False, "final_payloads": False, "admission": False, "training": False},
    }
    write_json(RECEIPT, receipt)
    write_json(OUT / "status.json", {"schema": "DAT-10-source-walk-support-repair-status-v1", "owner": "/root/data_expansion", "status": "complete_review_only", "summary": summary, "receipt": str(RECEIPT), "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"stage": "complete", "summary": summary, "receipt": str(RECEIPT)}), flush=True)


if __name__ == "__main__":
    main()
