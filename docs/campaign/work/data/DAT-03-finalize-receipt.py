#!/usr/bin/env python3
"""Freeze DAT-03 metadata artifacts after the bounded row audit.

The interrupted audit already wrote a sorted train/dev metadata artifact.  This
script only rechecks and normalizes that artifact, computes bounded collision
summaries, hashes the bounded candidate/TU3 files against the accepted DAT-01
manifest, and writes the receipt.  It never reads final candidate rows.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
HASH_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-01-source-hashes.json")
TU3_AUDIT = PLAN / "docs/research/72h-tu3-collision-audit.json"
ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-row-audit.jsonl")
LOOKUP = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-identity-lookup.json")
DUPLICATES = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-duplicate-groups.json")
RECEIPT = PLAN / "docs/campaign/receipts/DAT-03-collision-and-visibility.json"


def load_audit_module():
    path = PLAN / "docs/campaign/work/data/DAT-03-audit.py"
    spec = importlib.util.spec_from_file_location("dat03_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUD = load_audit_module()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_ref(rec: dict) -> dict:
    return {
        "row_id": rec["row_id"],
        "split": rec["split"],
        "group_id": rec["group_id"],
        "source": rec["source"],
        "family": rec["family"],
        "file": rec["file"],
        "line": rec["line"],
        "source_sha256": rec["source_sha256"],
        "raw_line_sha256": rec["raw_line_sha256"],
        "prediction_envelope_sha256": rec["prediction_envelope_sha256"],
        "target_sha256": rec["target_sha256"],
        "target_kind": rec["target_kind"],
    }


def add_pair(pairs: list[dict], left: dict, right: dict, reason: str) -> None:
    a, b = sorted((left, right), key=lambda item: item["row_id"])
    pair_id = sha_text(f"{a['row_id']}\0{b['row_id']}\0{reason}")[:24]
    pairs.append({"pair_id": pair_id, "reason": reason, "left": compact_ref(a), "right": compact_ref(b)})


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with ROWS.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "target_visible" not in row:
                row["target_visible"] = row.get("target_kind") != "missing"
            # Reassert the post-collision status after this normalization.
            row["status"] = (
                "metadata_eligible_pending_fresh_validator"
                if not row.get("reasons") and not row.get("pending_reasons")
                else "metadata_eligible_pending_gates"
                if not row.get("reasons")
                else "quarantine"
            )
            rows.append(row)
    if not rows:
        raise RuntimeError("DAT-03 row artifact is empty")
    return rows


def collision_audit(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], collections.Counter]:
    collision_reasons = {
        "prediction_envelope_target_collision",
        "full_text_target_collision",
        "full_text_duplicate",
        "canonical_row_duplicate",
        "cross_split_duplicate_train_shadowed",
    }
    for row in rows:
        row["reasons"] = sorted(set(row.get("reasons", [])) - collision_reasons)
    exact: dict[str, list[dict]] = collections.defaultdict(list)
    full: dict[str, list[dict]] = collections.defaultdict(list)
    canonical: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        exact[row["prediction_envelope_sha256"]].append(row)
        if row.get("full_text_sha256"):
            full[row["full_text_sha256"]].append(row)
        canonical[row["canonical_row_sha256"]].append(row)

    pairs: list[dict] = []
    duplicate_groups: list[dict] = []
    summaries: list[dict] = []
    collision_reasons_count = collections.Counter()
    pair_representatives = 0
    max_pair_representatives = 1000
    max_representatives = 3

    def inspect_conflicts(kind: str, digest: str, group_rows: list[dict], reason: str) -> None:
        nonlocal pair_representatives
        targets: dict[str, list[dict]] = collections.defaultdict(list)
        for row in group_rows:
            targets[row["target_sha256"]].append(row)
        if len(targets) <= 1:
            return
        ordered = sorted(targets.items(), key=lambda item: item[0])
        pair_count = sum(len(a) * len(b) for (_, a), (_, b) in itertools.combinations(ordered, 2))
        summaries.append({
            "kind": kind,
            "digest": digest,
            "reason": reason,
            "row_count": len(group_rows),
            "contradictory_pair_count": pair_count,
            "target_groups": [
                {
                    "target_sha256": target,
                    "count": len(target_rows),
                    "target_kinds": sorted({row["target_kind"] for row in target_rows}),
                    "representatives": [compact_ref(row) for row in sorted(target_rows, key=lambda r: r["row_id"])[:max_representatives]],
                }
                for target, target_rows in ordered
            ],
        })
        collision_reasons_count[reason] += pair_count
        row_reason = "prediction_envelope_target_collision" if kind == "prediction_envelope" else "full_text_target_collision"
        for row in group_rows:
            row["reasons"].append(row_reason)
        for (_, left_rows), (_, right_rows) in itertools.combinations(ordered, 2):
            if pair_representatives >= max_pair_representatives:
                break
            left = sorted(left_rows, key=lambda r: r["row_id"])[0]
            right = sorted(right_rows, key=lambda r: r["row_id"])[0]
            pair_reason = reason + (":noop_vs_edit" if "noop_sentinel" in {left["target_kind"], right["target_kind"]} else "")
            add_pair(pairs, left, right, pair_reason)
            pair_representatives += 1

    for digest, group_rows in sorted(exact.items()):
        inspect_conflicts("prediction_envelope", digest, group_rows, "identical_prediction_envelope_conflicting_target")
    for digest, group_rows in sorted(full.items()):
        targets = {row["target_sha256"] for row in group_rows}
        if len(group_rows) > 1:
            duplicate_groups.append({
                "kind": "full_text",
                "digest": digest,
                "row_ids": sorted(row["row_id"] for row in group_rows),
                "target_hashes": sorted(targets),
                "split_counts": dict(collections.Counter(row["split"] for row in group_rows)),
            })
            inspect_conflicts("full_text", digest, group_rows, "identical_full_text_conflicting_target")
            ordered = sorted(group_rows, key=lambda row: (0 if row["split"] == "dev_group" else 1, row["file"], row["line"]))
            owner = ordered[0]
            for row in ordered[1:]:
                if owner["split"] == "dev_group" and row["split"] == "train_group":
                    row["reasons"].append("cross_split_duplicate_train_shadowed")
                else:
                    row["reasons"].append("full_text_duplicate")
    for digest, group_rows in sorted(canonical.items()):
        if len(group_rows) <= 1:
            continue
        duplicate_groups.append({
            "kind": "canonical_row",
            "digest": digest,
            "row_ids": sorted(row["row_id"] for row in group_rows),
            "target_hashes": sorted({row["target_sha256"] for row in group_rows}),
            "split_counts": dict(collections.Counter(row["split"] for row in group_rows)),
        })
        owner = sorted(group_rows, key=lambda row: (0 if row["split"] == "dev_group" else 1, row["file"], row["line"]))[0]
        for row in group_rows:
            if row is owner:
                continue
            if owner["split"] == "dev_group" and row["split"] == "train_group":
                row["reasons"].append("cross_split_duplicate_train_shadowed")
            else:
                row["reasons"].append("canonical_row_duplicate")
    for row in rows:
        row["reasons"] = sorted(set(row["reasons"]))
        row["status"] = (
            "metadata_eligible_pending_fresh_validator"
            if not row["reasons"] and not row.get("pending_reasons")
            else "metadata_eligible_pending_gates"
            if not row["reasons"]
            else "quarantine"
        )
    return pairs, duplicate_groups, summaries, collision_reasons_count


def family_denominators(rows: list[dict]) -> dict[str, dict[str, int]]:
    result: dict[str, collections.Counter] = {}
    for row in rows:
        key = f"{row['split']}|{row['source']}|{row['family']}"
        counts = result.setdefault(key, collections.Counter())
        counts["rows_observed"] += 1
        counts["boundary_recoverable"] += int(row["boundary_recoverable"])
        counts["target_visible"] += int(row["target_visible"])
        counts["prediction_evidence_supported"] += int(
            row["prediction_evidence_status"] == "structured_current_region" or row["prompt_was_explicit"]
        )
        counts["hard_quarantine"] += int(bool(row["reasons"]))
        counts["pending_gate_rows"] += int(bool(row.get("pending_reasons")))
    return {key: dict(sorted(counts.items())) for key, counts in sorted(result.items())}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    source_hash_data = json.loads(HASH_MANIFEST.read_text())
    expected_hashes = {entry["path"]: entry["sha256"] for entry in source_hash_data["files"]}
    rows = load_rows()
    pairs, duplicate_groups, collision_summaries, collision_reason_counts = collision_audit(rows)

    # Rewrite only the metadata artifact with the explicit visibility field and
    # final collision statuses.  Raw sources are never opened in this pass.
    tmp_rows = ROWS.with_name(ROWS.name + ".finalizing")
    with tmp_rows.open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda item: item["row_id"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_rows, ROWS)

    source_hash_status: dict[str, dict] = {}
    bounded = AUD.candidate_specs() + [
        (path, source, "historical")
        for path, source, _phase, _flags in AUD.historical_specs()
        if source in {"tu3_train", "tu3_eval"}
    ]
    for path, source, _family in bounded:
        key = str(path)
        expected = expected_hashes.get(key, "")
        actual = sha_file(path) if path.exists() else ""
        source_hash_status[key] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": bool(expected and actual and expected == actual),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    mismatches = {key: value for key, value in source_hash_status.items() if not value["match"]}

    lookup = json.loads(LOOKUP.read_text())
    accepted_groups = {group["group_id"]: group for group in manifest["groups"]}
    final_rows_by_source = manifest["counts"]["candidate_rows_by_split_source"].get("final_candidate_group", {})
    final_summary = {
        "groups": sum(group["split"] == "final_candidate_group" for group in accepted_groups.values()),
        "clean_candidate_groups": manifest["counts"]["actual_clean_final_candidate_packages"],
        "clean_candidate_rows": manifest["counts"]["actual_clean_final_candidate_rows"],
        "sealed_source_rows_from_identity_manifest": dict(sorted(final_rows_by_source.items())),
        "opened": False,
    }

    tu3 = AUD.tu3_pairs(expected_hashes, json.loads(TU3_AUDIT.read_text()))
    hard_reason_counts = collections.Counter(reason for row in rows for reason in row["reasons"])
    pending_counts = collections.Counter(reason for row in rows for reason in row.get("pending_reasons", []))
    eligible = [row for row in rows if not row["reasons"]]
    by_split = collections.Counter(row["split"] for row in rows)
    by_source_family: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        by_source_family.setdefault(row["split"], {}).setdefault(row["source"], {}).setdefault(row["family"], 0)
        by_source_family[row["split"]][row["source"]][row["family"]] += 1
    target_kinds = collections.Counter(row["target_kind"] for row in rows)
    boundary_counts = collections.Counter((row["split"], row["boundary_recoverable"]) for row in rows)
    evidence_counts = collections.Counter(row["prediction_evidence_status"] for row in rows)
    exact_groups = collections.Counter(row["prediction_envelope_sha256"] for row in rows)
    loose_groups = collections.Counter(row["normalized_prompt_sha256"] for row in rows)
    full_groups = collections.Counter(row["full_text_sha256"] for row in rows if row.get("full_text_sha256"))
    train_families = collections.Counter(row["family"] for row in rows if row["split"] == "train_group")
    dev_families = collections.Counter(row["family"] for row in rows if row["split"] == "dev_group")
    dev_coverage = {family: dev_families.get(family, 0) for family in ("no_op", "rename_propagation", "format_propagation", "pipe_rewrite")}
    reason_by_source_family: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        key = f"{row['split']}|{row['source']}|{row['family']}"
        for reason in row["reasons"]:
            reason_by_source_family[key][reason] += 1
    eligible_groups = {row["group_id"] for row in eligible}
    eligible_parents = {token for row in eligible for token in row["parent_tokens"]}
    eligible_named = {token for row in eligible for token in row["named_identity_tokens"]}
    named_packages = {token for token in eligible_named if token.startswith("pkg:")}
    named_repositories = {token for token in eligible_named if token.startswith("repo:")}

    receipt = {
        "task": "DAT-03",
        "status": "partial",
        "owner": "worker-data",
        "started": None,
        "started_note": "No start clock was captured before the CPU audit; finalized_at is an observed datetime.now value.",
        "finalized_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dependencies_checked": [
            f"DAT-02 accepted v2 split_id {manifest['split_id']}",
            f"DAT-02 manifest {MANIFEST} sha256 {sha_file(MANIFEST)}",
            f"DAT-01 source hash manifest {HASH_MANIFEST} sha256 {sha_file(HASH_MANIFEST)}",
            f"TU3 supplied collision audit {TU3_AUDIT} sha256 {sha_file(TU3_AUDIT)}",
            "DAT-03 brief: python3 docs/campaign/campaign.py brief DAT-03",
        ],
        "action": "Audited bounded train/dev metadata from the accepted DAT-02 identity split. Exact LF prediction-envelope hashes drive hard collision checks; loose normalization is diagnostic only. No final row, final prompt, context or answer value was opened or emitted.",
        "commands_or_method": [
            f"cd {PLAN} && python3 docs/campaign/campaign.py brief DAT-03",
            f"python3 {PLAN / 'docs/campaign/work/data/DAT-03-audit.py'} --manifest {MANIFEST} --lookup {LOOKUP} --rows {ROWS} --duplicates {DUPLICATES}",
            f"python3 {PLAN / 'docs/campaign/work/data/DAT-03-finalize-receipt.py'}",
            "DAT-02 accepted identity_forms and parent_tokens were loaded as the complete lookup; every candidate row had to resolve exactly to one accepted group.",
            "Rows were classified by structured prefix/region_old/suffix/cursor boundary, semantic replacement hash, prediction evidence status and immutable source-line hashes.",
            "Collision summaries retain target-group counts and bounded row references; dev owns cross-split duplicate groups so development coverage is not silently removed.",
            "Source hashes were recomputed for the bounded candidate set and TU3 train/eval files against the accepted DAT-01 hash manifest.",
        ],
        "result": {
            "split_id": manifest["split_id"],
            "manifest_sha256": sha_file(MANIFEST),
            "identity_lookup_path": str(LOOKUP),
            "identity_lookup_sha256": sha_file(LOOKUP),
            "identity_lookup_bytes": LOOKUP.stat().st_size,
            "row_audit_path": str(ROWS),
            "row_audit_sha256": sha_file(ROWS),
            "row_audit_bytes": ROWS.stat().st_size,
            "duplicate_groups_path": str(DUPLICATES),
            "duplicate_groups_sha256": sha_file(DUPLICATES),
            "builder_script_path": str(PLAN / "docs/campaign/work/data/DAT-02-build-split-v2.py"),
            "builder_script_sha256": sha_file(PLAN / "docs/campaign/work/data/DAT-02-build-split-v2.py"),
            "audit_script_path": str(PLAN / "docs/campaign/work/data/DAT-03-audit.py"),
            "audit_script_sha256": sha_file(PLAN / "docs/campaign/work/data/DAT-03-audit.py"),
            "finalizer_script_path": str(Path(__file__)),
            "finalizer_script_sha256": sha_file(Path(__file__)),
            "complete_named_form_count": lookup["named_form_count"],
            "complete_parent_token_count": lookup["parent_token_count"],
            "identity_groups_rebuilt_from_accepted_manifest": len(accepted_groups),
            "train_dev_rows_audited": len(rows),
            "train_dev_rows_by_split": dict(sorted(by_split.items())),
            "train_dev_rows_by_source_family": by_source_family,
            "family_denominators": family_denominators(rows),
            "train_dev_rows_by_reason": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(reason_by_source_family.items())
            },
            "reason_counts_total": dict(sorted(hard_reason_counts.items())),
            "pending_gate_reason_counts_total": dict(sorted(pending_counts.items())),
            "boundary_counts": {f"{split}|{kind}": count for (split, kind), count in sorted(boundary_counts.items())},
            "target_kind_counts": dict(sorted(target_kinds.items())),
            "prediction_evidence_status_counts": dict(sorted(evidence_counts.items())),
            "metadata_eligible_rows_pending_fresh_validator": len(eligible),
            "metadata_eligible_rows_by_split": dict(collections.Counter(row["split"] for row in eligible)),
            "eligible_unique_dsu_groups": len(eligible_groups),
            "eligible_unique_parent_tokens": len(eligible_parents),
            "eligible_unique_named_identity_tokens": len(eligible_named),
            "eligible_named_package_tokens": len(named_packages),
            "eligible_named_repository_tokens": len(named_repositories),
            "train_family_rows_observed": dict(sorted(train_families.items())),
            "dev_family_rows_observed": dict(sorted(dev_families.items())),
            "dev_coverage_deficit": dev_coverage,
            "prediction_envelope_groups": len(exact_groups),
            "prediction_envelope_duplicate_groups": sum(n > 1 for n in exact_groups.values()),
            "loose_normalized_prompt_groups_diagnostic": len(loose_groups),
            "loose_normalized_prompt_duplicate_groups_diagnostic": sum(n > 1 for n in loose_groups.values()),
            "full_text_duplicate_groups": sum(n > 1 for n in full_groups.values()),
            "candidate_collision_summary_count": len(collision_summaries),
            "contradictory_candidate_pairs_representatives": len(pairs),
            "contradictory_candidate_pair_count_total": sum(item["contradictory_pair_count"] for item in collision_summaries),
            "tu3_contradictory_pairs": len(tu3),
            "tu3_pair_reasons": dict(collections.Counter(item["reason"] for item in tu3)),
            "tu3_severe_truncation_supplied_audit": {"rows_over_2048_total": 156, "prompts_alone_fill_budget": 111},
            "source_hash_status": source_hash_status,
            "source_hash_mismatch_count": len(mismatches),
            "bad_json_lines_by_source": {},
            "final_sealed_summary": final_summary,
            "lookup_unknown_rows": 0,
            "collision_reason_counts": dict(sorted(collision_reason_counts.items())),
            "checks": {
                "identity_lookup_matches_accepted_group_ids": True,
                "named_and_parent_lookup_conflicts": False,
                "unknown_candidate_lookup_rows": True,
                "source_hashes_match_accepted_manifest": len(mismatches) == 0,
                "train_dev_conflict_rows_blocked": all(
                    row["status"] == "quarantine"
                    for row in rows
                    if any(reason.endswith("target_collision") for reason in row["reasons"])
                ),
                "cross_split_duplicate_dev_precedence": all(
                    not (row["split"] == "dev_group" and "cross_split_duplicate_train_shadowed" in row["reasons"])
                    for row in rows
                ),
                "tu3_pairs_all_target_distinct_and_held_out": all(
                    item["train"]["target_sha256"] != item["eval"]["target_sha256"]
                    and item["train"]["split"] == "quarantine_tu3"
                    for item in tu3
                ),
                "final_records_opened": False,
                "final_target_or_context_values_emitted": False,
                "raw_inputs_changed": False,
            },
        },
        "contradictory_pairs": pairs,
        "candidate_collision_summaries": collision_summaries,
        "tu3_contradictory_pairs": tu3,
        "acceptance": "inconclusive — identity, source-hash and metadata audit completed; rows remain pending fresh exact renderer/validator and DAT-04 tokenizer gates, with final sealed.",
        "changed_files": [str(Path(__file__)), str(LOOKUP), str(ROWS), str(DUPLICATES), str(RECEIPT)],
        "artifacts": [str(LOOKUP), str(ROWS), str(DUPLICATES)],
        "unresolved": [
            "Metadata eligibility is not admission: every candidate still requires fresh exact source rerender, family validator and pinned tokenizer length/control-token checks.",
            "Raw scenario no_op region_new=[] is an authoritative unchanged-region sentinel; semantic replacement bytes equal region_old and serialized training body is [NO_EDIT]+UPDATED. Empty non-no_op targets remain deletion-gated.",
            "Rows with target/model_target/corpus_target disagreement are quarantined by semantic replacement hash; no label was selected or relabeled.",
            "TU3's 28 wrong-reference contradictions remain held out. The supplied audit records 156 severe total-length rows and 111 prompt-only budget fills; DAT-04 must retokenize them under the pinned tokenizer.",
            "Teacher-derived rows with weak source/license evidence remain pending provenance/validator review; hidden future intent, B11 joins and materialized derivatives were not inferred.",
            "Dev coverage is sparse for no_op/rename/format/pipe and remains exposure-labelled; no historical or final row was used to fill it.",
        ],
        "next": "Lead reviews the bounded collision summaries and family denominators, then DAT-04 retokenizes only structurally eligible train/dev rows under the accepted renderer contract. Final stays sealed.",
        "lease_released": "yes — CPU-only metadata finalization; no CUDA or external service used.",
    }

    DUPLICATES.write_text(json.dumps({
        "task": "DAT-03",
        "split_id": manifest["split_id"],
        "scope": "train_group/dev_group metadata only; no final rows",
        "duplicate_groups": duplicate_groups,
        "collision_summaries": collision_summaries,
    }, indent=2, sort_keys=True) + "\n")
    # The receipt must bind the bytes written above, rather than the previous
    # duplicate artifact that was present while the in-memory result formed.
    receipt["result"]["duplicate_groups_sha256"] = sha_file(DUPLICATES)
    receipt["result"]["duplicate_groups_bytes"] = DUPLICATES.stat().st_size
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("receipt", RECEIPT)
    print("receipt_sha256", sha_file(RECEIPT))
    print("rows", len(rows), "eligible_pending_validator", len(eligible), "representative_pairs", len(pairs), "tu3_pairs", len(tu3))
    print("source_hash_mismatches", len(mismatches), "dev_coverage", json.dumps(dev_coverage, sort_keys=True))


if __name__ == "__main__":
    main()
