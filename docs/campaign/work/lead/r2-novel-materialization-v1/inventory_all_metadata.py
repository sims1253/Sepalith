#!/usr/bin/env python3
"""Inventory every DAT-03 metadata row and plan quota-free materialisation.

This pass reads the complete row-audit JSONL.  It emits metadata only: no
prompt, source document, or target body is copied.  The old 6,690-row roster is
marked as historical selection context, while the source-walk plan covers every
supported global-TRAIN row after explicit CPT-validation reservation.  Shards
are an I/O/resume unit and never a family, row, or time exclusion.
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
try:
    os.sched_setaffinity(0, {0, 1})
except (AttributeError, OSError):
    pass

ROOT = pathlib.Path(__file__).resolve().parents[5]
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
AUDIT = E / "DAT-03-row-audit.jsonl"
GLOBAL = E / "DAT-02-global-split-v2.json"
CPT = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
ROSTER = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/novel-train-audit-roster.jsonl"
EXISTING_PROV = ROOT / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl"
STRUCTURED_REPORT = E / "DAT10-novel-v1/structured/report.json"
STRUCTURED_SELECTED = E / "DAT10-novel-v1/structured/selected-audit-rows.jsonl"
COMPLETION_PACKETS = E / "DAT10-novel-v1/completion/candidate-packets.jsonl"
MATERIALIZED_GATE = E / "DAT10-novel-v1/candidate-gate-v3/candidate-gate-ledger-v3.jsonl"
OWN = ROOT / "docs/campaign/work/lead/r2-novel-materialization-v1"
OUT = E / "DAT10-novel-v1/all-data-inventory-v1"

SUPPORTED_FINISH = "finish_block"
SUPPORTED_STRUCTURED = {
    "roxygen_drafting", "no_op", "rename_propagation", "format_propagation",
    "pipe_rewrite", "na_rm_propagation",
}
SHARD_ROWS = 5000  # resume unit only; every row is assigned to a shard


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                yield json.loads(line)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or "")


def normalize_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v)]
    return [str(value)]


def supported_family(family: str) -> bool:
    return family.startswith(SUPPORTED_FINISH) or family in SUPPORTED_STRUCTURED


def compact_metadata(row: dict[str, Any], *, global_split: str | None, cpt: str,
                     roster_selected: bool, existing_parent: bool,
                     policy_reasons: list[str], final_status: str,
                     route: str, shard: int | None, source_category: str,
                     source_attempt_status: str) -> dict[str, Any]:
    # Explicitly select metadata fields.  Do not serialize source text-bearing
    # fields even if a future audit row grows such fields.
    return {
        "row_id": row.get("row_id"),
        "group_id": row.get("group_id"),
        "family": row.get("family"),
        "source": row.get("source"),
        "source_file": row.get("file"),
        "source_line": row.get("line"),
        "source_sha256": row.get("source_sha256"),
        "raw_line_sha256": row.get("raw_line_sha256"),
        "canonical_row_sha256": row.get("canonical_row_sha256"),
        "split_from_audit": row.get("split"),
        "global_registry_split": global_split,
        "cpt_partition": cpt,
        "license_status": row.get("license_status"),
        "license_evidence_present": row.get("license_evidence_present"),
        "target_kind": row.get("target_kind"),
        "target_sha256": row.get("target_sha256"),
        "target_chars": row.get("target_chars"),
        "target_visible": row.get("target_visible"),
        "explicit_truncation": row.get("explicit_truncation"),
        "boundary_missing": row.get("boundary_missing"),
        "boundary_recoverable": row.get("boundary_recoverable"),
        "boundary_type_errors": row.get("boundary_type_errors"),
        "validator_status": row.get("validator_status"),
        "parent_quality": row.get("parent_quality"),
        "prediction_evidence_status": row.get("prediction_evidence_status"),
        "noop_status": row.get("noop_status"),
        "prompt_chars": row.get("prompt_chars"),
        "normalized_prompt_sha256": row.get("normalized_prompt_sha256"),
        "roster_selected_6690": roster_selected,
        "existing_parent_11505": existing_parent,
        "policy_reasons": policy_reasons,
        "final_status": final_status,
        "materialization_route": route,
        "source_category": source_category,
        "source_attempt_status": source_attempt_status,
        "io_shard": shard,
    }


def main() -> int:
    for path in (AUDIT, GLOBAL, CPT, ROSTER, EXISTING_PROV, STRUCTURED_REPORT,
                 STRUCTURED_SELECTED, COMPLETION_PACKETS, MATERIALIZED_GATE):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    global_doc = json.loads(GLOBAL.read_text(encoding="utf-8"))
    global_groups = {str(g["group_id"]): str(g.get("split")) for g in global_doc.get("groups", [])}
    cpt_doc = json.loads(CPT.read_text(encoding="utf-8"))
    cpt_groups = {str(k): str(v) for k, v in cpt_doc.get("groups", {}).items()}
    roster_ids = {row_id(row) for row in read_jsonl(ROSTER)}
    existing_ids = {str(row.get("id") or row.get("input_row_id")) for row in read_jsonl(EXISTING_PROV)}
    structured_excluded: dict[str, str] = {}
    structured_doc = json.loads(STRUCTURED_REPORT.read_text(encoding="utf-8"))
    for item in structured_doc.get("excluded", []):
        if item.get("id") and item.get("reason"):
            structured_excluded[str(item["id"])] = str(item["reason"])
    attempted_ids: set[str] = set()
    for item in read_jsonl(STRUCTURED_SELECTED):
        if item.get("row_id"):
            attempted_ids.add(str(item["row_id"]))
    for item in read_jsonl(COMPLETION_PACKETS):
        ref = item.get("row_ref") or {}
        if ref.get("row_id"):
            attempted_ids.add(str(ref["row_id"]))
    materialized_ids = {
        str(item.get("row_id")) for item in read_jsonl(MATERIALIZED_GATE)
        if item.get("row_id")
    }
    support_reason_names = {
        "no_op_kind_requires_separate_support_review": "noop_support_review",
        "documentation_external_facts_require_separate_support_review": "documentation_support_review",
    }

    output_path = OUT / "all-metadata-inventory-v1.jsonl"
    queue_paths = {
        "noop_support_review": OUT / "support-review-noop-450-or-more.jsonl",
        "documentation_support_review": OUT / "support-review-documentation-579-or-more.jsonl",
    }
    output = output_path.open("w", encoding="utf-8", newline="\n")
    queues = {key: path.open("w", encoding="utf-8", newline="\n") for key, path in queue_paths.items()}
    statuses = Counter()
    reasons = Counter()
    families = Counter()
    family_status = Counter()
    splits = Counter()
    cpts = Counter()
    routes = Counter()
    source_categories = Counter()
    source_attempts = Counter()
    source_category_by_family = Counter()
    sources = Counter()
    shard_rows: Counter[int] = Counter()
    total = 0
    duplicate_ids = 0
    seen_ids: set[str] = set()
    supported_global_train = 0
    materialization_rows = 0
    unattempted_source_walk_rows = 0
    queue_counts = Counter()
    started = datetime.now(timezone.utc).isoformat()
    progress_path = OWN / "all-data-inventory-progress.json"

    try:
        for total, row in enumerate(read_jsonl(AUDIT), 1):
            rid = row_id(row)
            if not rid:
                raise RuntimeError(f"missing row_id at audit line {total}")
            if rid in seen_ids:
                duplicate_ids += 1
            seen_ids.add(rid)
            family = str(row.get("family") or "")
            families[family] += 1
            group = str(row.get("group_id") or "")
            gs = global_groups.get(group)
            cp = cpt_groups.get(group, "cpt_partition_missing")
            splits[gs or "global_registry_missing"] += 1
            cpts[cp] += 1
            if gs:
                global_split = gs
            else:
                global_split = None
            policy: list[str] = []
            raw_base = normalize_reasons(row.get("base_reasons")) + normalize_reasons(row.get("reasons"))
            raw_pending = normalize_reasons(row.get("pending_reasons"))
            for value in raw_base:
                if value not in policy:
                    policy.append(f"audit:{value}")
            for value in raw_pending:
                if value not in policy:
                    policy.append(f"pending:{value}")
            if gs is None:
                policy.append("global_split_registry_missing")
            elif gs != "train_group":
                policy.append(f"global_split:{gs}")
            if cp == "cpt_validation":
                policy.append("cpt_validation_reserved")
            elif cp == "cpt_partition_missing" and gs == "train_group":
                policy.append("cpt_partition_missing_but_global_train_eligible")
            if not supported_family(family):
                policy.append("family_unsupported_by_reviewed_builder")
            if row.get("license_status") in (None, "missing_unresolved") or row.get("license_evidence_present") is False:
                policy.append("license_provenance_pending")
            if row.get("target_visible") is False:
                policy.append("target_not_visible")
            if row.get("explicit_truncation"):
                policy.append("explicit_truncation")
            boundary_missing = normalize_reasons(row.get("boundary_missing"))
            boundary_errors = normalize_reasons(row.get("boundary_type_errors"))
            if boundary_missing or boundary_errors:
                if row.get("boundary_recoverable"):
                    policy.append("boundary_repairable_queue")
                else:
                    policy.append("boundary_missing_unrecoverable")
            if rid in structured_excluded and structured_excluded[rid] in support_reason_names:
                policy.append(structured_excluded[rid])
            if family.startswith("roxygen") and row.get("parent_quality") in ("scenario_parent_incomplete", "missing", None):
                # This is a metadata support/quality pending signal, not a
                # rejection.  The exact 579 materialized queue is separately
                # pinned from the reviewed structured builder report.
                policy.append("documentation_parent_support_pending")
            if family == "no_op" and row.get("noop_status") != "authoritative_noop_sentinel":
                policy.append("noop_semantics_support_pending")

            hard_heldout = gs is None or gs != "train_group"
            cpt_heldout = cp == "cpt_validation"
            hard_metadata = (
                "target_not_visible" in policy or "explicit_truncation" in policy
                or "boundary_missing_unrecoverable" in policy
                or any(x.startswith("audit:") and x not in ("audit:license_provenance_pending",) for x in policy)
            )
            route = "metadata_only_pending"
            if not supported_family(family):
                route = "unsupported_family_queue"
            elif hard_heldout:
                route = "heldout_metadata_only"
            elif cpt_heldout:
                route = "cpt_validation_metadata_only"
            elif hard_metadata:
                route = "metadata_exclusion_review"
            elif family.startswith("finish_block"):
                route = "completion_source_walk"
            else:
                route = "structured_source_walk"
            if hard_heldout:
                final = "heldout_global_split"
            elif cpt_heldout:
                final = "heldout_cpt_validation"
            elif not supported_family(family):
                final = "unsupported_family_pending_route"
            elif hard_metadata:
                final = "metadata_gate_or_repair_review"
            elif any(x in policy for x in ("noop_support_review", "documentation_support_review")):
                final = "support_review_pending"
            elif any(x in policy for x in ("license_provenance_pending", "documentation_parent_support_pending", "pending:source_parent_evidence_incomplete")):
                final = "source_license_support_pending"
            elif "boundary_repairable_queue" in policy:
                final = "repairable_boundary_pending_source_check"
            else:
                final = "source_materialization_eligible_pending_admission"
            if gs == "train_group" and supported_family(family):
                supported_global_train += 1
                if cp != "cpt_validation" and not hard_metadata:
                    materialization_rows += 1
            if gs is None or gs != "train_group" or cp == "cpt_validation":
                source_category = "heldout_global_or_cpt_validation"
                source_attempt_status = "heldout_metadata_only"
            elif rid in materialized_ids:
                source_category = "verified_eligible_materialized_candidate"
                source_attempt_status = "verified_source_materialized"
            elif rid in existing_ids:
                source_category = "existing_parent_materialized"
                source_attempt_status = "existing_parent"
            elif rid in attempted_ids:
                source_category = "source_checked_excluded_or_repair_queue"
                source_attempt_status = "source_checked"
            elif hard_metadata:
                source_category = "genuine_metadata_exclusion_pending_review"
                source_attempt_status = "metadata_gate_not_source_attempted"
            elif supported_family(family) and gs == "train_group":
                source_category = "unattempted_supported_global_train"
                source_attempt_status = "unattempted_source_walk"
            else:
                source_category = "unsupported_or_unrouted_metadata"
                source_attempt_status = "unattempted"
            source_categories[source_category] += 1
            source_attempts[source_attempt_status] += 1
            source_category_by_family[(family, source_category)] += 1
            if source_category == "unattempted_supported_global_train" and cp != "cpt_validation":
                unattempted_source_walk_rows += 1
            shard = (unattempted_source_walk_rows - 1) // SHARD_ROWS if source_category == "unattempted_supported_global_train" and cp != "cpt_validation" else None
            if shard is not None:
                shard_rows[shard] += 1
            statuses[final] += 1
            routes[route] += 1
            for reason in policy:
                reasons[reason] += 1
            splits[gs or "global_registry_missing"] += 0
            cpts[cp] += 0
            family_status[(family, final)] += 1
            sources[str(row.get("file") or "")] += 1
            rec = compact_metadata(
                row, global_split=gs, cpt=cp, roster_selected=rid in roster_ids,
                existing_parent=rid in existing_ids, policy_reasons=policy,
                final_status=final, route=route, shard=shard,
                source_category=source_category,
                source_attempt_status=source_attempt_status,
            )
            output.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            queue_kind = support_reason_names.get(structured_excluded.get(rid, ""))
            if queue_kind:
                queues[queue_kind].write(json.dumps({
                    **rec,
                    "support_queue_reason": structured_excluded[rid],
                    "support_queue_source": str(STRUCTURED_REPORT),
                    "support_queue_status": "awaiting_explicit_source_or_semantic_support_review",
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                queue_counts[queue_kind] += 1
            if total % 50000 == 0:
                progress_path.write_text(json.dumps({
                    "status": "running", "rows_scanned": total,
                    "input": str(AUDIT), "input_sha256": "deferred_until_complete",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "milestone_policy": "progress_only; no data is dropped at a timebox",
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        output.close()
        for stream in queues.values():
            stream.close()

    if total != 376830:
        raise RuntimeError(f"DAT-03 row count changed or incomplete: {total}")
    if duplicate_ids:
        raise RuntimeError(f"duplicate row ids in complete audit: {duplicate_ids}")

    input_sha = sha_file(AUDIT)
    inv_sha = sha_file(output_path)
    queue_hashes = {key: {"path": str(path), "sha256": sha_file(path), "rows": queue_counts[key]} for key, path in queue_paths.items()}
    # Use deterministic source walk shards as an execution manifest.  Rows are
    # assigned across the complete eligible population; no cap or quota is used.
    shard_manifest_path = OUT / "source-walk-shards-v1.jsonl"
    with shard_manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        for shard in sorted(shard_rows):
            f.write(json.dumps({
                "shard": shard, "rows": shard_rows[shard], "rows_per_shard_io_unit": SHARD_ROWS,
                "status": "planned_unmaterialized", "route": "completion_or_structured_by_inventory_row",
                "resume_semantics": "completed_shards_only; remaining shards remain required",
            }, sort_keys=True, separators=(",", ":")) + "\n")
    source_plan = {
        "schema": "DAT-10-all-supported-source-walk-v1",
        "status": "metadata_inventory_and_resumable_plan",
        "input": str(AUDIT), "input_sha256": input_sha,
        "global_split": str(GLOBAL), "global_split_sha256": sha_file(GLOBAL),
        "cpt_partition": str(CPT), "cpt_partition_sha256": sha_file(CPT),
        "total_metadata_rows": total,
        "complete_audit_required_count": 376830,
        "supported_global_train_rows": supported_global_train,
        "materialization_rows_after_cpt_and_hard_metadata_gates": materialization_rows,
        "unattempted_supported_global_train_rows": unattempted_source_walk_rows,
        "selected_6690_roster_rows": len(roster_ids),
        "existing_parent_rows": len(existing_ids),
        "family_counts": dict(sorted(families.items())),
        "status_counts": dict(sorted(statuses.items())),
        "route_counts": dict(sorted(routes.items())),
        "source_category_counts": dict(sorted(source_categories.items())),
        "source_attempt_status_counts": dict(sorted(source_attempts.items())),
        "source_category_by_family": {
            f"{family}|{category}": count
            for (family, category), count in sorted(source_category_by_family.items())
        },
        "global_split_counts": dict(sorted(splits.items())),
        "cpt_partition_counts": dict(sorted(cpts.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "source_file_row_counts": dict(sorted(sources.items())),
        "io_shards": {"rows_per_shard": SHARD_ROWS, "shard_count": len(shard_rows), "manifest": str(shard_manifest_path), "purpose": "resume/I-O only, no data exclusion"},
        "support_queues": queue_hashes,
        "inventory": {"path": str(output_path), "sha256": inv_sha, "rows": total},
        "policy": {
            "all_eligible_data": True,
            "family_row_time_quotas": False,
            "train_content_only": True,
            "dev_final_content": False,
            "cpt_validation_reserved": True,
            "cpt_partition_missing_global_train": "retain_pending_global_train",
            "long_targets_preserved": True,
            "target_truncation": False,
            "admission": "none; every route remains pending root source/license/duplicate/quality checks",
        },
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cuda_used": False,
    }
    plan_path = OUT / "source-walk-plan-v1.json"
    plan_path.write_text(json.dumps(source_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress_path.write_text(json.dumps({
        "status": "complete", "rows_scanned": total, "input_sha256": input_sha,
        "inventory_sha256": inv_sha, "plan": str(plan_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "milestone_policy": "complete scan; no rows dropped by timebox",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": total, "supported_global_train": supported_global_train,
        "materialization_rows": materialization_rows,
        "statuses": dict(statuses), "support_queues": dict(queue_counts),
        "inventory": str(output_path), "plan": str(plan_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
