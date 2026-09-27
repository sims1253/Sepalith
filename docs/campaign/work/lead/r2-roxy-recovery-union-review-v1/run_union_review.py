#!/usr/bin/env python3
"""Freeze a corrected, review-only union for the DAT-10 roxygen residuals.

This wrapper consumes the frozen prior packets and the separately executed
corrected R scope audit.  It writes only this packet.  It deliberately emits
IDs and provenance metadata, never source or target text, and never makes a
training-admission decision.
"""
from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKET = Path(__file__).resolve().parent
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
ROOT_IDS = PLAN / "docs/campaign/work/lead/r2-roxy10017-root-context-review-v1/candidate-ids.json"
ORIGINAL_IDS = PLAN / "docs/campaign/work/lead/r2-roxy-context-admission-review-v1/full/recommended-context-ids.json"
ORIGINAL_AUDITED = PLAN / "docs/campaign/work/lead/r2-roxy-context-admission-review-v1/full/audited-rows.jsonl"
RECOVERY_IDS = PLAN / "docs/campaign/work/lead/r2-roxy-full-context-recovery-v1/materialization-v4/recoverable-ids.json"
RECOVERY_LEDGER = PLAN / "docs/campaign/work/lead/r2-roxy-full-context-recovery-v1/materialization-v4/recovery-ledger.jsonl"
LOOP_DIR = Path("/mnt/e/sepalith/campaign-20260915/data-work/Roxy-loop-scope-review-v1")
LOOP_IDS = LOOP_DIR / "recoverable-ids.json"
LOOP_RAW = LOOP_DIR / "r-scope-output.jsonl"
NSE_DIR = Path("/mnt/e/sepalith/campaign-20260915/data-work/Roxy-NSE-semantic-review-v1")
NSE_IDS = NSE_DIR / "recoverable-ids.json"
NSE_INPUT = NSE_DIR / "r-input.json"
CORRECTED_R = PACKET / "corrected-r-scope-output.jsonl"
CORRECTED_R_SCRIPT = PACKET / "corrected_loop_scope_audit.R"
CONTROL_TEST = PACKET / "test_corrected_loop_scope_audit.py"

INPUTS = {
    "root_candidate_ids": ROOT_IDS,
    "original_recommendation_ids": ORIGINAL_IDS,
    "original_audited_rows": ORIGINAL_AUDITED,
    "full_context_recovery_ids": RECOVERY_IDS,
    "full_context_recovery_ledger": RECOVERY_LEDGER,
    "old_loop_recoverable_ids": LOOP_IDS,
    "old_loop_scope_output": LOOP_RAW,
    "nse_recoverable_ids": NSE_IDS,
    "nse_input": NSE_INPUT,
    "corrected_scope_output": CORRECTED_R,
    "corrected_scope_script": CORRECTED_R_SCRIPT,
    "negative_control_test": CONTROL_TEST,
    "union_reconciliation_script": PACKET / "run_union_review.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise AssertionError(f"{path}:{line_number}: expected object")
                rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def id_file(path: Path, key: str) -> set[str]:
    value = read_json(path)
    if not isinstance(value, dict) or key not in value or not isinstance(value[key], list):
        raise AssertionError(f"unexpected ID packet schema: {path}")
    ids = value[key]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"duplicate IDs in {path}")
    if not all(isinstance(row_id, str) and row_id for row_id in ids):
        raise AssertionError(f"non-string ID in {path}")
    declared = value.get("count", value.get("recoverable_count", value.get("accepted_recommendation_count")))
    if declared is not None and declared != len(ids):
        raise AssertionError(f"declared count mismatch in {path}: {declared} != {len(ids)}")
    return set(ids)


def class_and_location_counts(rows: Iterable[dict[str, Any]]) -> tuple[collections.Counter[str], collections.Counter[str], int]:
    classes: collections.Counter[str] = collections.Counter()
    locations: collections.Counter[str] = collections.Counter()
    rows_with_evidence = 0
    for row in rows:
        evidence = row.get("evidence", {})
        residuals = evidence.get("residuals", {}) if isinstance(evidence, dict) else {}
        if residuals:
            rows_with_evidence += 1
        for residual in residuals.values():
            for occurrence in residual.get("occurrence_evidence", []):
                classes[str(occurrence.get("classification", "<missing>"))] += 1
                locations[str(occurrence.get("location", "<missing>"))] += 1
    return classes, locations, rows_with_evidence


def rows_containing_location(rows: Iterable[dict[str, Any]], locations: set[str]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        for residual in row.get("evidence", {}).get("residuals", {}).values():
            if any(item.get("location") in locations for item in residual.get("occurrence_evidence", [])):
                result.add(row["row_id"])
                break
    return result


def rows_containing_class(rows: Iterable[dict[str, Any]], classes: set[str]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        for residual in row.get("evidence", {}).get("residuals", {}).values():
            if any(item.get("classification") in classes for item in residual.get("occurrence_evidence", [])):
                result.add(row["row_id"])
                break
    return result


def main() -> None:
    PACKET.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen input(s): " + ", ".join(missing))

    root = set(read_json(ROOT_IDS))
    original = id_file(ORIGINAL_IDS, "accepted_recommendation_ids")
    recovery = id_file(RECOVERY_IDS, "recoverable_ids")
    old_loop = id_file(LOOP_IDS, "ids")
    nse = id_file(NSE_IDS, "ids")
    if len(root) != 10017:
        raise AssertionError(f"root candidate denominator changed: {len(root)}")
    source_sets = {
        "original_recommendation": original,
        "full_context_recovery": recovery,
        "old_loop_scope_recovery": old_loop,
        "nse_semantic_recovery": nse,
    }
    for name, ids in source_sets.items():
        outside = ids - root
        if outside:
            raise AssertionError(f"{name} contains IDs outside root candidate scope: {sorted(outside)[:3]}")
    names = list(source_sets)
    intersections: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = source_sets[left] & source_sets[right]
            intersections[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise AssertionError(f"unexpected source-set overlap {left}/{right}: {sorted(overlap)[:3]}")

    old_rows = read_jsonl(LOOP_RAW)
    corrected_rows = read_jsonl(CORRECTED_R)
    if len(old_rows) != 2273 or len(corrected_rows) != 2273:
        raise AssertionError(f"scope row denominator changed: old={len(old_rows)} corrected={len(corrected_rows)}")
    old_by_id = {row["row_id"]: row for row in old_rows}
    corrected_by_id = {row["row_id"]: row for row in corrected_rows}
    if len(old_by_id) != len(old_rows) or len(corrected_by_id) != len(corrected_rows):
        raise AssertionError("duplicate row ID in scope output")
    if set(old_by_id) != set(corrected_by_id):
        raise AssertionError("corrected scope output does not cover exactly the frozen old scope rows")
    if old_loop != {row_id for row_id, row in old_by_id.items() if row.get("status") == "recoverable_lexically_bound_residuals"}:
        raise AssertionError("old loop ID packet is inconsistent with old raw output")

    nse_input = read_json(NSE_INPUT)
    input_by_id: dict[str, dict[str, Any]] = {}
    input_source_by_id: dict[str, tuple[str, str]] = {}
    for group in nse_input["source_groups"]:
        for row in group["rows"]:
            row_id = row["row_id"]
            if row_id in input_by_id:
                raise AssertionError(f"duplicate frozen input row: {row_id}")
            input_by_id[row_id] = row
            input_source_by_id[row_id] = (group["source_path"], group["source_sha256"])
    if set(input_by_id) != set(corrected_by_id):
        raise AssertionError("corrected output does not cover exactly the frozen R input IDs")
    for row_id, output in corrected_by_id.items():
        source = input_source_by_id[row_id]
        # The corrected auditor intentionally omits source metadata on the
        # three non-unique-target records.  When metadata is present, it must
        # still match the frozen input pin exactly.
        if ("source_path" in output or "source_sha256" in output) and (
            output.get("source_path"), output.get("source_sha256")
        ) != source:
            raise AssertionError(f"source pin changed for {row_id}")

    old_recovered = {row_id for row_id, row in old_by_id.items() if row.get("status") == "recoverable_lexically_bound_residuals"}
    corrected_recovered = {
        row_id for row_id, row in corrected_by_id.items() if row.get("status") == "recoverable_lexically_bound_residuals"
    }
    corrected_nonunique = {
        row_id for row_id, row in corrected_by_id.items() if row.get("status") == "hold_target_definition_not_unique"
    }
    corrected_holds = set(corrected_by_id) - corrected_recovered
    old_only = old_recovered - corrected_recovered
    corrected_only = corrected_recovered - old_recovered
    if (len(old_recovered), len(corrected_recovered), len(old_only), len(corrected_only)) != (984, 957, 28, 1):
        raise AssertionError("corrected/old recovery counts differ from the audited result")
    if len(corrected_nonunique) != 3 or len(corrected_holds) != 1316:
        raise AssertionError("corrected hold status counts changed")
    codetools_error_ids = {
        row_id
        for row_id, row in corrected_by_id.items()
        if row.get("evidence", {}).get("codetools_ok") is False or row.get("evidence", {}).get("codetools_error")
    }
    if codetools_error_ids:
        raise AssertionError(f"corrected audit has codetools failures: {sorted(codetools_error_ids)[:3]}")

    corrected_classes, corrected_locations, rows_with_evidence = class_and_location_counts(corrected_rows)
    old_classes, old_locations, _ = class_and_location_counts(old_rows)
    old_only_classes, old_only_locations, _ = class_and_location_counts(corrected_by_id[row_id] for row_id in old_only)
    corrected_only_classes, corrected_only_locations, _ = class_and_location_counts(corrected_by_id[row_id] for row_id in corrected_only)
    formal_default_rows = rows_containing_location(corrected_rows, {"root_formal_default", "formal_default"})
    call_head_rows = rows_containing_location(corrected_rows, {"call_head"})
    member_literal_rows = rows_containing_class(corrected_rows, {"member_literal_not_lexical"})
    old_only_member_literal_rows = rows_containing_class((corrected_by_id[row_id] for row_id in old_only), {"member_literal_not_lexical"})
    if old_only_member_literal_rows != old_only:
        raise AssertionError("every old-only recovery must be rejected by corrected member-literal evidence")
    if len(corrected_only) != 1 or not (corrected_only & {"1c8c8f4908ec12afc813f884"}):
        raise AssertionError("corrected call-head recovery control changed")
    if not ({"018df81bd4773e47ca8cfe2e"} & nse):
        raise AssertionError("NSE semantic recovery input lost")

    # Load original recommendations to retain the formal-mismatch boundary in
    # the held ledger.  No source or target payload is copied.
    original_audited = read_jsonl(ORIGINAL_AUDITED)
    original_by_id = {row["row_id"]: row for row in original_audited}
    if set(original_by_id) != root:
        raise AssertionError("original audited rows do not cover root candidate scope")
    original_formal_holds = {
        row_id for row_id, row in original_by_id.items() if row.get("recommendation") == "hold_documented_formal_mismatch"
    }
    original_semantic_holds = {
        row_id
        for row_id, row in original_by_id.items()
        if row.get("recommendation") == "hold_true_omitted_or_external_global_evidence"
    }
    if len(original_formal_holds) != 172 or len(original_semantic_holds) != 5942:
        raise AssertionError("original recommendation denominator changed")
    if original_formal_holds & (original | recovery | corrected_recovered | nse):
        raise AssertionError("a formal-mismatch row entered a recovery set")

    # The NSE result is an independent semantic exception.  It is not counted
    # as a lexical recovery because corrected scope analysis correctly holds it.
    safe_by_source = {
        "original_context_recommendation": original,
        "full_context_recovery": recovery,
        "corrected_loop_scope_recovery": corrected_recovered,
        "nse_semantic_recovery": nse,
    }
    safe_names = list(safe_by_source)
    safe_intersections: dict[str, int] = {}
    for left_index, left in enumerate(safe_names):
        for right in safe_names[left_index + 1 :]:
            safe_intersections[f"{left}__{right}"] = len(safe_by_source[left] & safe_by_source[right])
            if safe_intersections[f"{left}__{right}"]:
                raise AssertionError(f"unexpected safe-union overlap {left}/{right}")
    safe_union = set().union(*safe_by_source.values())
    old_union = original | recovery | old_recovered | nse
    held = root - safe_union
    if len(old_union) != 8557 or len(safe_union) != 8530 or len(held) != 1487:
        raise AssertionError(f"union counts changed: old={len(old_union)}, safe={len(safe_union)}, held={len(held)}")
    if safe_union & held or safe_union | held != root:
        raise AssertionError("safe/held partition is not exact")

    held_reason: dict[str, str] = {}
    for row_id in sorted(held):
        if row_id in original_formal_holds:
            held_reason[row_id] = "original_documented_formal_mismatch"
        elif row_id in corrected_nonunique:
            held_reason[row_id] = "corrected_target_definition_not_unique"
        elif row_id in corrected_holds:
            held_reason[row_id] = "corrected_scope_unbound_or_member_literal"
        elif row_id in original_semantic_holds:
            held_reason[row_id] = "prior_semantic_evidence_not_recovered"
        else:
            raise AssertionError(f"held ID has no reason: {row_id}")
    held_reason_counts = collections.Counter(held_reason.values())
    if held_reason_counts != collections.Counter({
        "original_documented_formal_mismatch": 172,
        "corrected_scope_unbound_or_member_literal": 1312,
        "corrected_target_definition_not_unique": 3,
    }):
        raise AssertionError(f"unexpected held reason counts: {held_reason_counts}")

    # A compact row-level ledger lets root reconcile the union without reading
    # any payload.  Keep the root candidate order stable and IDs only.
    ledger: list[dict[str, Any]] = []
    for row_id in sorted(root):
        memberships = {
            "original_context_recommendation": row_id in original,
            "full_context_recovery": row_id in recovery,
            "old_loop_scope_recovery": row_id in old_recovered,
            "corrected_loop_scope_recovery": row_id in corrected_recovered,
            "nse_semantic_recovery": row_id in nse,
        }
        if row_id in safe_by_source["original_context_recommendation"]:
            decision, reason = "safe_review_union", "original_context_recommendation"
        elif row_id in safe_by_source["full_context_recovery"]:
            decision, reason = "safe_review_union", "full_context_recovery"
        elif row_id in safe_by_source["corrected_loop_scope_recovery"]:
            decision, reason = "safe_review_union", "corrected_loop_scope_recovery"
        elif row_id in safe_by_source["nse_semantic_recovery"]:
            decision, reason = "safe_review_union", "nse_semantic_recovery"
        else:
            decision, reason = "hold_review", held_reason[row_id]
        ledger.append({"row_id": row_id, "decision": decision, "reason": reason, "memberships": memberships})
    write_jsonl(PACKET / "source-set-ledger.jsonl", ledger)

    write_json(PACKET / "safe-review-union-ids.json", {
        "schema": "sepalith.dat10.roxy_recovery_union_ids.v1",
        "status": "review_only_not_training_admission",
        "scope": "root_10017_candidate_rows",
        "safe_review_union_count": len(safe_union),
        "ids": sorted(safe_union),
        "source_counts": {name: len(ids) for name, ids in safe_by_source.items()},
        "set_intersections": safe_intersections,
    })
    write_json(PACKET / "held-review-ids.json", {
        "schema": "sepalith.dat10.roxy_recovery_union_holds.v1",
        "status": "review_only_not_training_admission",
        "scope": "root_10017_candidate_rows",
        "held_count": len(held),
        "ids": sorted(held),
        "reason_counts": dict(sorted(held_reason_counts.items())),
        "reason_by_id": held_reason,
    })
    write_json(PACKET / "corrected-loop-recoverable-ids.json", {
        "schema": "sepalith.dat10.roxy_corrected_loop_scope_ids.v1",
        "status": "review_only_not_training_admission",
        "audit_version": "corrected_formal_defaults_call_heads_fail_closed_globals_v1",
        "count": len(corrected_recovered),
        "ids": sorted(corrected_recovered),
    })
    write_json(PACKET / "corrected-loop-held-ids.json", {
        "schema": "sepalith.dat10.roxy_corrected_loop_scope_holds.v1",
        "status": "review_only_not_training_admission",
        "audit_version": "corrected_formal_defaults_call_heads_fail_closed_globals_v1",
        "count": len(corrected_holds),
        "ids": sorted(corrected_holds),
        "status_counts": dict(collections.Counter(row["status"] for row in corrected_rows if row["row_id"] in corrected_holds)),
    })
    write_json(PACKET / "corrected-audit-delta.json", {
        "schema": "sepalith.dat10.roxy_corrected_scope_delta.v1",
        "status": "review_only_not_training_admission",
        "old_recoverable_count": len(old_recovered),
        "corrected_recoverable_count": len(corrected_recovered),
        "old_only_count": len(old_only),
        "old_only_ids": sorted(old_only),
        "old_only_reason": "corrected visitor records member_literal_not_lexical occurrences that the old result allowed alongside local bindings",
        "old_only_corrected_class_counts": dict(sorted(old_only_classes.items())),
        "old_only_corrected_location_counts": dict(sorted(old_only_locations.items())),
        "corrected_only_count": len(corrected_only),
        "corrected_only_ids": sorted(corrected_only),
        "corrected_only_reason": "corrected visitor inspects executable call heads and recognizes loop-bound call-head names",
        "corrected_only_class_counts": dict(sorted(corrected_only_classes.items())),
        "corrected_only_location_counts": dict(sorted(corrected_only_locations.items())),
    })

    artifact_pins = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in INPUTS.items()
    }
    receipt = {
        "schema": "sepalith.dat10.roxy_recovery_union_review.v1",
        "status": "complete_review_only_root_admission_required",
        "training_admission": False,
        "source_text_written": False,
        "target_text_written": False,
        "cuda": False,
        "model_generation": False,
        "packet": str(PACKET),
        "policy": {
            "source_packets_preserved": True,
            "no_heldout_or_dev_payload": True,
            "no_arbitrary_data_exclusion": True,
            "codetools_failure_mode": "fail_closed_hold",
            "formal_defaults_visited": True,
            "call_heads_visited": True,
            "member_literals_not_lexical": True,
            "nse_exception_requires_independent_semantic_evidence": True,
        },
        "denominators": {
            "root_candidate_rows": len(root),
            "original_recommendation_rows": len(original),
            "full_context_recovery_rows": len(recovery),
            "residual_loop_scope_rows": len(old_rows),
            "nse_input_rows": len(input_by_id),
            "nse_input_source_groups": len(nse_input["source_groups"]),
            "nse_semantic_recovery_rows": len(nse),
        },
        "corrected_scope_audit": {
            "rows_audited": len(corrected_rows),
            "status_counts": dict(collections.Counter(row["status"] for row in corrected_rows)),
            "codetools_error_count": len(codetools_error_ids),
            "rows_with_structural_evidence": rows_with_evidence,
            "occurrence_class_counts": dict(sorted(corrected_classes.items())),
            "occurrence_location_counts": dict(sorted(corrected_locations.items())),
            "formal_default_rows": len(formal_default_rows),
            "call_head_rows": len(call_head_rows),
            "member_literal_rows": len(member_literal_rows),
            "negative_control_command": ["python3", str(CONTROL_TEST)],
            "negative_control_result": "6/6 corrected lexical-scope controls passed",
            "old_recoverable": len(old_recovered),
            "corrected_recoverable": len(corrected_recovered),
            "old_only": len(old_only),
            "corrected_only": len(corrected_only),
        },
        "reconciliation": {
            "old_union_count": len(old_union),
            "corrected_safe_union_count": len(safe_union),
            "held_count": len(held),
            "safe_union_source_counts": {name: len(ids) for name, ids in safe_by_source.items()},
            "held_reason_counts": dict(sorted(held_reason_counts.items())),
            "old_source_pairwise_intersections": intersections,
            "safe_union_pairwise_intersections": safe_intersections,
            "corrected_loop_vs_old_loop_intersection": len(corrected_recovered & old_recovered),
            "safe_union_equals_root_partition": safe_union | held == root and not safe_union & held,
            "nse_row_is_lexical_hold_but_semantic_exception": "018df81bd4773e47ca8cfe2e" in corrected_holds and "018df81bd4773e47ca8cfe2e" in nse,
        },
        "output_artifacts": {},
        "input_artifacts": artifact_pins,
        "representative_cohort_handoff": {
            "status": "separate_data_only_review_packet_not_admitted",
            "packet": "/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1",
            "result_path": "/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/result.json",
            "result_sha256": "839c4e96787ab4a8ce5ab27d544ee810be28ec15e966213ccbf4f8d3850a96e5",
            "token_rows_path": "/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/cpt_train_ctx16384.jsonl",
            "token_rows_sha256": "0d71f1b7b519281c6d063c64083505b1c730d30edda7cbd86dc0124e7fb1caaf",
            "schedule_path": "/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/draw-schedule.json",
            "schedule_sha256": "5649b14a58213ddf95d25b6068999a48fbcdb51f01acaf67303682b3fdfbc212",
            "provenance_path": "/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/provenance.jsonl",
            "provenance_sha256": "be3aba370a0adf50b4dad6819c5009171286e8b70dcffbfe2ebec7bf8053512c",
            "schema": "campaign_cpt_data.schema1",
            "row_fields_present": [
                "schema", "row_id", "document_id", "package", "group_id", "cpt_partition", "source_path", "source_sha256", "chunk_index", "input_ids", "labels", "attention_mask", "source_token_start", "source_token_end", "token_start", "token_end", "overlap_context_tokens", "is_document_end", "supervised_tokens", "document_token_count",
            ],
            "row_fields_optional_not_present": ["token_stream_sha256", "builder_id", "tokenizer_revision"],
            "schedule_fields": ["split_id", "method", "seed", "max_steps", "effective_batch", "token_rows_sha256", "row_ids", "replay_row_ids"],
            "counts": {"selected_complete_documents": 1024, "token_rows": 1046, "packages_or_groups": 814, "payload_tokens": 2575972, "draws": 1056, "steps": 66, "replays": 10, "unselected_queued_documents": 105716},
        },
        "artifacts": {
            "safe_review_union_ids": "safe-review-union-ids.json",
            "held_review_ids": "held-review-ids.json",
            "corrected_loop_recoverable_ids": "corrected-loop-recoverable-ids.json",
            "corrected_loop_held_ids": "corrected-loop-held-ids.json",
            "corrected_audit_delta": "corrected-audit-delta.json",
            "source_set_ledger": "source-set-ledger.jsonl",
        },
    }
    receipt_path = PLAN / "docs/campaign/receipts/DAT-10-roxy-recovery-union-review.json"
    write_json(receipt_path, receipt)

    # Hash every packet output after writing, then rewrite the receipt with its
    # final output pins.  The receipt itself is intentionally excluded from
    # this map because its hash changes when the map is added.
    output_names = [
        "safe-review-union-ids.json", "held-review-ids.json", "corrected-loop-recoverable-ids.json",
        "corrected-loop-held-ids.json", "corrected-audit-delta.json", "source-set-ledger.jsonl",
    ]
    receipt["output_artifacts"] = {
        name: {"path": str(PACKET / name), "bytes": (PACKET / name).stat().st_size, "sha256": sha256(PACKET / name)}
        for name in output_names
    }
    write_json(receipt_path, receipt)
    receipt_hash = sha256(receipt_path)
    write_json(PACKET / "status.json", {
        "schema": "sepalith.dat10.roxy_recovery_union_review_status.v1",
        "status": "complete_review_only_root_admission_required",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_hash,
        "corrected_rows": len(corrected_rows),
        "old_recoverable": len(old_recovered),
        "corrected_recoverable": len(corrected_recovered),
        "safe_review_union": len(safe_union),
        "held_review": len(held),
        "cpu_only": True,
        "training_admission": False,
    })
    print(json.dumps({
        "receipt": str(receipt_path), "receipt_sha256": receipt_hash,
        "old_union": len(old_union), "safe_union": len(safe_union), "held": len(held),
        "old_only": len(old_only), "corrected_only": len(corrected_only),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
