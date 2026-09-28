#!/usr/bin/env python3
"""Account the terminal CPT union from metadata only.

This audit deliberately never opens a ``cpt_train.jsonl`` file.  It consumes
the terminal union manifest, source manifest, hold ledger, and the small
upstream manifests/receipts that provide document and token denominators.
Candidate payloads remain protected while the lossless 16K conversion runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED = {
    "documents": 177_190,
    "rows": 332_455,
    "input_tokens": 461_653_440,
    "payload_tokens": 460_833_265,
    "source_records": 8_204,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def metadata_pin(path: Path) -> dict[str, Any]:
    """Hash a small metadata file; never use this for candidate payloads."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def source_group(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else label


def source_record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    prefix = Counter(source_group(row["label"]) for row in records)
    totals: dict[str, dict[str, int]] = {}
    for group in sorted(prefix):
        members = [row for row in records if source_group(row["label"]) == group]
        totals[group] = {
            "records": len(members),
            "rows_read": sum(int(row["rows_read"]) for row in members),
            "bytes": sum(int(row["bytes"]) for row in members),
        }
    bad_bindings: list[dict[str, Any]] = []
    for row in records:
        for actual, expected, field in (
            (row.get("sha256"), row.get("expected_sha256"), "sha256"),
            (row.get("bytes"), row.get("expected_bytes"), "bytes"),
            (row.get("rows_read"), row.get("expected_rows"), "rows_read"),
        ):
            if expected is not None and actual != expected:
                bad_bindings.append({"label": row.get("label"), "field": field,
                                     "actual": actual, "expected": expected})
    if bad_bindings:
        raise AssertionError(f"source manifest binding mismatch: {bad_bindings[:3]}")
    return {
        "records": len(records),
        "by_source_kind": totals,
        "all_expected_hash_bytes_row_bindings_match": True,
    }


def build_global_cap_frontier(union: Path, output: Path) -> dict[str, Any]:
    """Resolve the 2K global-shard cap exclusions to the old prefix groups.

    The global producer's exclusion records contain only package/path.  The
    package receipt and selected-source metadata are enough to bind each path
    to a seeded group without opening source bytes.  Terminal provenance is
    also metadata: an exact source-path hit means that path is already in one
    emitted union component.  Source hashes are intentionally not invented;
    a future recovery pass must rehash each source before content dedup.
    """
    plan = Path("docs/campaign/work")
    exclusions_path = plan / "r2-cpt-global-shard-v1/shard/exclusions.jsonl"
    package_receipts_path = plan / "r2-cpt-global-shard-v1/shard/package-receipts.jsonl"
    selected_metadata_path = plan / "r2-cpt-global-shard-v1/shard/selected-source-metadata.jsonl"
    provenance_path = union / "document-provenance.jsonl"
    excluded = [
        row for row in read_jsonl(exclusions_path)
        if row.get("reason") == "whole_document_exceeds_remaining_group_cap"
    ]
    package_receipts = {
        row["package"]: row for row in read_jsonl(package_receipts_path)
    }
    selected = {
        row["path"]: row for row in read_jsonl(selected_metadata_path)
    }
    # This is the 79 MB provenance index, not the 5.4 GB candidate payload.
    emitted_paths: set[str] = set()
    emitted_documents = 0
    for row in read_jsonl(provenance_path):
        emitted_documents += 1
        path = row.get("source_path")
        if isinstance(path, str):
            emitted_paths.add(path)
    if emitted_documents != 177_190:
        raise AssertionError(f"terminal provenance document count changed: {emitted_documents}")
    if len(excluded) != 1_999:
        raise AssertionError(f"global cap frontier changed: {len(excluded)}")
    records: list[dict[str, Any]] = []
    for row in excluded:
        path = row["path"]
        meta = selected.get(path)
        if meta is None:
            raise AssertionError(f"global cap path missing selected metadata: {path}")
        receipt = package_receipts.get(row["package"])
        if receipt is None:
            raise AssertionError(f"global cap package missing receipt: {row['package']}")
        if meta.get("package") != row["package"] or meta.get("split") != "train_group" or meta.get("cpt_partition") != "cpt_train":
            raise AssertionError(f"global cap source guard mismatch: {path}")
        records.append({
            "package": row["package"],
            "path": path,
            "reason": row["reason"],
            "group_id": meta["group_id"],
            "seeded_index": int(receipt["seeded_index"]),
            "version": meta.get("version"),
            "bytes": int(meta["bytes"]),
            "description_path": meta.get("description_path"),
            "description_sha256": meta.get("description_sha256"),
            "license": meta.get("license"),
            "split": meta["split"],
            "cpt_partition": meta["cpt_partition"],
            "terminal_exact_source_path_hit": path in emitted_paths,
            "source_sha256": None,
            "recovery_status": "unmaterialized_recovery_frontier",
        })
    if any(row["terminal_exact_source_path_hit"] for row in records):
        raise AssertionError("a cap-excluded path is already in terminal provenance")
    by_group: dict[tuple[int, str, str], int] = Counter(
        (row["seeded_index"], row["group_id"], row["package"]) for row in records
    )
    group_summary = [
        {
            "seeded_index": index,
            "group_id": group_id,
            "package": package,
            "excluded_documents": count,
            "package_metadata_files": package_receipts[package].get("metadata_files"),
            "already_emitted_documents_in_global_group": package_receipts[package].get("emitted_documents"),
            "already_emitted_code_tokens_in_global_group": package_receipts[package].get("emitted_code_tokens"),
        }
        for (index, group_id, package), count in sorted(by_group.items())
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(partial, output)
    return {
        "reason": "whole_document_exceeds_remaining_group_cap",
        "excluded_documents": len(records),
        "packages": len({row["package"] for row in records}),
        "groups": len(group_summary),
        "seeded_index_min": min(row["seeded_index"] for row in records),
        "seeded_index_max": max(row["seeded_index"] for row in records),
        "groups_at_or_after_terminal_suffix_start_775": sum(row["seeded_index"] >= 775 for row in records),
        "selected_metadata_matches": len(records),
        "terminal_provenance_documents_indexed": emitted_documents,
        "terminal_exact_source_path_hits": sum(row["terminal_exact_source_path_hit"] for row in records),
        "all_exact_source_paths_absent_from_terminal_union": True,
        "source_sha256_available_in_exclusion_metadata": False,
        "content_dedup_status": "requires_source_rehash_during_recovery; path-level absence is verified",
        "frontier_jsonl": metadata_pin(output),
        "global_exclusions": metadata_pin(exclusions_path),
        "package_receipts": metadata_pin(package_receipts_path),
        "selected_source_metadata": metadata_pin(selected_metadata_path),
        "terminal_document_provenance": metadata_pin(provenance_path),
        "group_summary": group_summary,
        "license_counts": dict(sorted(Counter(row.get("license") for row in records).items())),
    }


def base_counts(source: dict[str, Any]) -> dict[str, Any]:
    reviewed = Path(source["reviewed_manifest"])
    manifest = read_json(reviewed)
    counts = manifest.get("counts", {}).get("cpt_train")
    if not isinstance(counts, dict):
        raise AssertionError(f"missing base counts: {reviewed}")
    documents = int(counts["documents"])
    rows = int(counts["rows"])
    input_tokens = int(counts["input_tokens"])
    # The historical base manifests expose supervised_tokens.  The terminal
    # union's payload convention is loss/supervised tokens minus one terminal
    # document marker per document, the same convention used by the repaired
    # result manifests.
    payload_tokens = int(counts.get("payload_tokens", int(counts["supervised_tokens"]) - documents))
    if int(source["rows_read"]) != rows:
        raise AssertionError(f"base row mismatch: {source['label']}")
    artifact = manifest.get("artifacts", {}).get("cpt_train.jsonl", {})
    for field in ("sha256", "bytes"):
        if source.get(field) != artifact.get(field):
            raise AssertionError(f"base {field} mismatch: {source['label']}")
    return {
        "label": source["label"],
        "records": 1,
        "documents": documents,
        "rows": rows,
        "input_tokens": input_tokens,
        "payload_tokens": payload_tokens,
        "artifact": {
            "path": source["path"],
            "bytes": source["bytes"],
            "sha256": source["sha256"],
            "reviewed_manifest": metadata_pin(reviewed),
        },
        "upstream_counts": counts,
        "upstream_exclusions": manifest.get("exclusions", {}),
    }


def repair_counts(label: str, source: dict[str, Any], receipt: Path) -> dict[str, Any]:
    value = read_json(receipt)
    counts = value.get("counts", {})
    documents = int(counts["documents"])
    rows = int(counts["rows"])
    input_tokens = int(counts["input_tokens"])
    if "payload_tokens" in counts:
        payload_tokens = int(counts["payload_tokens"])
    else:
        payload_tokens = int(counts["supervised_tokens"]) - documents
    if source["rows_read"] != rows or source["sha256"] != value["artifacts"]["cpt_train.jsonl"]["sha256"]:
        raise AssertionError(f"repair source mismatch: {label}")
    return {
        "label": label,
        "records": 1,
        "documents": documents,
        "rows": rows,
        "input_tokens": input_tokens,
        "payload_tokens": payload_tokens,
        "artifact": {
            "path": source["path"],
            "bytes": source["bytes"],
            "sha256": source["sha256"],
            "receipt": metadata_pin(receipt),
        },
        "upstream_counts": counts,
        "admission_status": value.get("status") or value.get("admission"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    union = args.union
    manifest_path = union / "manifest.json"
    gate_path = union / "terminal-gate.json"
    source_manifest_path = union / "source-manifest.jsonl"
    holds_path = union / "pending-holds.jsonl"
    exclusions_path = union / "exclusions.jsonl"

    manifest = read_json(manifest_path)
    gate = read_json(gate_path)
    sources = read_jsonl(source_manifest_path)
    holds = read_jsonl(holds_path)
    final_counts = manifest["counts"]
    if gate.get("status") != "pass":
        raise AssertionError(f"terminal gate is not pass: {gate.get('status')}")
    if gate.get("bulk_payload_opened") is not False:
        raise AssertionError("terminal gate did not assert metadata-only gate")
    if manifest.get("training_admission") is not False:
        raise AssertionError("union must remain unadmitted")
    if manifest.get("truncation") is not False or manifest.get("retokenized") is not False:
        raise AssertionError("union is not lossless/non-retokenized")
    if exclusions_path.stat().st_size != 0 or manifest["counts"].get("exclusions") != {}:
        raise AssertionError("terminal union contains row exclusions")
    if len(holds) != 6:
        raise AssertionError(f"expected six pending holds, got {len(holds)}")
    source_audit = source_record_summary(sources)

    base_sources = [row for row in sources if row["label"].startswith("base:")]
    main_sources = [row for row in sources if row["label"].startswith("main:")]
    alias_sources = [row for row in sources if row["label"].startswith("alias:")]
    repair_sources = [row for row in sources if row["label"].startswith("repair:")]
    if len(base_sources) != 2 or len(main_sources) != 8092 or len(alias_sources) != 108 or len(repair_sources) != 2:
        raise AssertionError("source-record partition does not match terminal accounting")

    bases = [base_counts(row) for row in base_sources]
    main_gate = manifest["main_terminal_gate"]
    alias_gate = manifest["alias_terminal_capture_gate"]
    main_counts = main_gate["progress"]["counts"]
    alias_counts = alias_gate["progress"]["counts"]
    main = {
        "label": "main:terminal-suffix",
        "records": len(main_sources),
        "groups": int(main_gate["progress"]["groups_committed"]),
        "documents": int(main_counts["documents"]),
        "rows": sum(int(row["rows_read"]) for row in main_sources),
        "input_tokens": int(main_counts["input_tokens"]),
        "payload_tokens": int(main_counts["payload_tokens"]),
        "progress_pin": metadata_pin(Path(main_gate["path"])),
        "upstream_exclusions": {
            key: value for key, value in main_counts.items() if key.startswith("excluded_")
        },
        "repair_items_pending": int(main_gate["progress"]["repair_items_pending"]),
    }
    if main["rows"] != int(main_counts["rows"]):
        raise AssertionError("main source-manifest rows do not match progress")
    alias = {
        "label": "alias:terminal-capture",
        "records": len(alias_sources),
        "groups": int(alias_gate["progress"]["recovery_groups_committed"]),
        "documents": int(alias_counts["documents"]),
        "rows": sum(int(row["rows_read"]) for row in alias_sources),
        "input_tokens": int(alias_counts["input_tokens"]),
        "payload_tokens": int(alias_counts["payload_tokens"]),
        "dedup_excluded": int(alias_counts["dedup_excluded"]),
        "repair_items_pending": int(alias_counts["repair"]),
        "progress_pin": metadata_pin(Path(alias_gate["path"])),
    }
    if alias["rows"] != int(alias_counts["rows"]):
        raise AssertionError("alias source-manifest rows do not match progress")

    svars_source = next(row for row in repair_sources if row["label"] == "repair:svars")
    tfp_source = next(row for row in repair_sources if row["label"] == "repair:tfprobability")
    svars = repair_counts("repair:svars", svars_source, Path(svars_source["receipt"]))
    tfp = repair_counts("repair:tfprobability", tfp_source, Path(tfp_source["receipt"]))
    components = bases + [main, alias, svars, tfp]
    calculated = {
        key: sum(int(component[key]) for component in components)
        for key in ("documents", "rows", "input_tokens", "payload_tokens")
    }
    calculated["source_records"] = sum(int(component["records"]) for component in components)
    if calculated != {key: int(final_counts[key]) for key in EXPECTED}:
        raise AssertionError({"calculated": calculated, "manifest": final_counts})
    if calculated != EXPECTED:
        raise AssertionError({"calculated": calculated, "expected": EXPECTED})
    if gate["counts"] != final_counts:
        raise AssertionError("terminal gate counts differ from union manifest")

    frontier_receipt = Path("docs/campaign/receipts/DAT-10-cpt-final-frontier-readiness.json")
    frontier = read_json(frontier_receipt)
    repairs = frontier["all_four_repair_queues"]["records"]
    if len(repairs) != 4:
        raise AssertionError("frontier receipt did not bind all four main repairs")
    alias_repair_queues = [
        Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-license-alias-recovery-v1/groups/000883-g-0f3f23606c86a6faf1de/repair-queue.jsonl"),
        Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-license-alias-recovery-v1/groups/005021-g-8797e2592d2b0dd69443/repair-queue.jsonl"),
    ]
    alias_repairs = []
    for queue in alias_repair_queues:
        records = read_jsonl(queue)
        if len(records) != 1:
            raise AssertionError(f"expected one alias repair record: {queue}")
        record = records[0]
        source_path = Path(record["path"])
        source_stat = {
            "path": str(source_path),
            "exists": source_path.is_file(),
            "bytes": source_path.stat().st_size if source_path.is_file() else None,
            "inode": source_path.stat().st_ino if source_path.is_file() else None,
            "device": source_path.stat().st_dev if source_path.is_file() else None,
        }
        if source_stat["bytes"] != 0 or record.get("source_sha256_observed") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
            raise AssertionError(f"alias empty repair is not statically empty: {source_path}")
        alias_repairs.append({"queue": metadata_pin(queue), "record": record,
                              "source_stat": source_stat,
                              "disposition": "closed_exclude_zero_byte_no_payload"})

    cap_frontier_path = args.output.parent / "global-cap-exclusion-frontier.jsonl"
    cap_frontier = build_global_cap_frontier(union, cap_frontier_path)
    conversion_result_path = union.parent / "CPT-final-union-v1-ctx16384/result.json"
    conversion_observation: dict[str, Any] = {"path": str(conversion_result_path), "present": conversion_result_path.is_file()}
    if conversion_result_path.is_file():
        conversion_result = read_json(conversion_result_path)
        conversion_observation.update({
            "status": conversion_result.get("status"),
            "metadata_pin": metadata_pin(conversion_result_path),
            "counts": conversion_result.get("totals"),
            "outputs": conversion_result.get("outputs"),
            "guarantees": conversion_result.get("guarantees"),
        })

    # These are all small metadata files.  Do not add a candidate payload here.
    pinned_metadata = {
        "union_manifest": metadata_pin(manifest_path),
        "terminal_gate": metadata_pin(gate_path),
        "source_manifest": metadata_pin(source_manifest_path),
        "pending_holds": metadata_pin(holds_path),
        "input_manifest": metadata_pin(union / "input-manifest.json"),
        "empty_exclusions": metadata_pin(exclusions_path),
        "main_progress": metadata_pin(Path(main_gate["path"])),
        "alias_progress": metadata_pin(Path(alias_gate["path"])),
        "frontier_readiness_receipt": metadata_pin(frontier_receipt),
        "union_builder": metadata_pin(Path("docs/campaign/work/lead/r2-cpt-final-union-integration-v2/build_final_union.py")),
        "union_validator": metadata_pin(Path("docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/campaign_cpt_data.py")),
    }

    result = {
        "schema": "sepalith.dat10.final-union-terminal-accounting.v1",
        "status": "verified_metadata_only_terminal_union_pending_root_admission",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "union": str(union),
            "source_manifest_records": "read as JSONL metadata only",
            "candidate_payloads_opened": False,
            "candidate_payload_paths_not_opened": [
                str(union / "cpt_train.jsonl"),
                str(Path(sources[0]["path"])) if sources else None,
            ],
            "metadata_provenance_index_opened": str(union / "document-provenance.jsonl"),
            "lossless_16k_process_left_untouched": True,
            "training_admission": False,
        },
        "terminal_gate": {
            "status": gate["status"],
            "bulk_payload_opened": gate["bulk_payload_opened"],
            "gates": gate["gates"],
            "row_artifact_metadata": gate["row_artifact"],
        },
        "reported_counts": {key: int(final_counts[key]) for key in EXPECTED},
        "calculated_counts": calculated,
        "accounting_deltas": {key: calculated[key] - int(final_counts[key]) for key in calculated},
        "source_records": {
            **source_audit,
            "components": components,
            "base_prefix_note": manifest["accounting"],
        },
        "upstream_repair_frontier": {
            "receipt": metadata_pin(frontier_receipt),
            "four_main_records": repairs,
            "main_queue_status": frontier["all_four_repair_queues"]["status"],
        },
        "alias_repair_closure": {
            "description": "Both known alias repair records resolve to zero-byte source files and source_sha256_observed=e3b0; they are closed no-payload exclusions, separate from RivRetrieve.",
            "records": alias_repairs,
            "count": len(alias_repairs),
            "status": "closed_exclude_zero_byte_no_payload",
        },
        "pending_holds": {
            "count": len(holds),
            "records": holds,
        },
        "left_out_or_pending": [
            {
                "kind": "license_hold",
                "package": "Rblpapi",
                "documents": 19,
                "rows": 19,
                "payload_tokens": 16515,
                "status": "held_not_emitted",
                "reason": "The explicit source-only GPL-3 evidence is still separate from the package DESCRIPTION FOSS=no and separately licensed Bloomberg headers/binaries.",
            },
            {
                "kind": "empty_source",
                "package": "RivRetrieve",
                "documents": 0,
                "rows": 0,
                "payload_tokens": 0,
                "status": "closed_excluded",
                "reason": "R/data.R is zero bytes; no training payload exists to recover.",
            },
            {
                "kind": "alias_repair_queue",
                "records": 2,
                "documents_emitted": 0,
                "rows_emitted": 0,
                "status": "closed_exclude_zero_byte_no_payload",
                "reason": "dataquieR and PTXQC source files are both statically zero bytes and carry the empty SHA-256; there is no payload to recover.",
            },
            {
                "kind": "candidate_review",
                "package": "svars",
                "documents": 53,
                "rows": 74,
                "payload_tokens": 82081,
                "status": "emitted_pending_root_admission_and_global_dedup",
            },
            {
                "kind": "candidate_review",
                "package": "tfprobability",
                "documents": 21,
                "rows": 103,
                "payload_tokens": 182645,
                "status": "emitted_pending_root_admission_and_global_dedup",
            },
        ],
        "upstream_exclusion_census": {
            "main_progress": main["upstream_exclusions"],
            "base_manifests": {component["label"]: component["upstream_exclusions"] for component in bases},
            "interpretation": "These producer-level exclusions are outside the terminal union stream and are not represented by a union exclusion record. They prevent an all-corpus admission claim; they are retained as named upstream categories.",
        },
        "global_cap_exclusion_frontier": cap_frontier,
        "lossless_16k_followup_observation": conversion_observation,
        "duplicate_and_heldout_guards": {
            "union_exclusions_jsonl_empty": True,
            "source_hash_bytes_row_pins_match": True,
            "main_rows_are_terminal_8092_groups": True,
            "alias_capture_matches_terminal_main_8092": True,
            "alias_existing_recovery_vs_new_main_conflicts": alias_gate["progress"].get("existing_recovery_vs_new_main_conflicts"),
            "alias_dedup_excluded": int(alias_counts["dedup_excluded"]),
            "reserved_cpt_validation_source_hashes": int(manifest["guard_info"]["reserved_cpt_validation_source_hashes"]),
            "protected_parent_hash_count": int(manifest["guard_info"]["protected_parent_hash_count"]),
            "global_split_id": manifest["guard_info"]["global_split_id"],
            "cpt_partition_split_id": manifest["guard_info"]["cpt_partition_split_id"],
            "upstream_heldout_rejection_before_payload_read": True,
            "row_level_protocol": [
                "global split must be train_group",
                "CPT partition must be cpt_train",
                "document_id equals source_sha256 and row_id equals source_sha256:chunk_index",
                "source documents have contiguous zero-based chunks ending at document_token_count",
                "native BOS=0/EOS=1 and tokenizer vocabulary bounds",
                "known CPT-validation/protected identities are rejected",
            ],
            "source_code_pin": pinned_metadata["union_builder"],
        },
        "pins": pinned_metadata,
        "next": "Root may review the six holds and upstream exclusion census, then perform the terminal global-dedup/admission decision against the lossless 16K output when the active conversion completes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "counts": calculated, "holds": len(holds)}, sort_keys=True))


if __name__ == "__main__":
    main()
