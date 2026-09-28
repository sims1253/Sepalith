#!/usr/bin/env python3
"""Quota-free structured materialization for one prepared DAT10 shard.

The shard input was selected from the complete metadata inventory.  This
runner processes every row in that shard, with a shard-size limit used only for
I/O/resume.  It deliberately does not use the historical per-family or
per-group caps.  Conversion failures become explicit per-row exclusions;
converted candidate packets remain unadmitted until root review.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
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
EXEC = pathlib.Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
TRAINING = EXEC / "experiments" / "training"
PKG = EXEC / "packages" / "sepalith" / "src"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(PKG))

E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
SHARD_ROOT = E / "DAT10-novel-v1/source-walk-shards-v1"
GLOBAL_CHECKPOINT = SHARD_ROOT / "source-walk-progress-v1.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: pathlib.Path, values: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def checkpoint(path: pathlib.Path, value: dict[str, Any]) -> None:
    """Persist a resumable milestone without exposing source content."""
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard < 0:
        raise ValueError("shard_must_be_nonnegative")
    shard = SHARD_ROOT / f"shard-{args.shard:04d}"
    input_path = shard / "structured-audit-rows.jsonl"
    input_manifest = shard / "manifest.json"
    if not input_path.exists() or not input_manifest.exists():
        raise FileNotFoundError(input_path)
    input_meta = json.loads(input_manifest.read_text(encoding="utf-8"))
    if input_meta.get("policy", {}).get("cpt_validation_content") is not False:
        raise RuntimeError("source-walk input does not pin CPT-validation exclusion")
    records = list(jsonl(input_path))
    expected = int(input_meta["route_counts"]["structured"])
    if len(records) != expected:
        raise RuntimeError(f"input record count {len(records)} != manifest {expected}")
    if len({str(r.get("row_id")) for r in records}) != len(records):
        raise RuntimeError("duplicate source-walk input row ids")
    out = shard / "structured-materialization-v1"
    if out.exists():
        raise ValueError("materialization_output_must_be_fresh")
    out.mkdir(parents=True)

    # Use the reviewed converter and constructor modules, but replicate the
    # broad wrapper's per-record checks without its family/group quotas.
    import campaign_admission_structured as adapter  # type: ignore
    import campaign_structured_batch as batch  # type: ignore
    from campaign_token_audit import selected_context  # type: ignore

    scenarios, scenarios_ref = batch.load_source("scenarios")
    roxygen, roxygen_ref = batch.load_source("roxygen_drafting")
    suffix, suffix_ref = batch.load_source("suffix_scenarios")
    constructors = [scenarios_ref, roxygen_ref, suffix_ref]

    # Verify and load every requested raw source row while hashing each source
    # file once.  This may read large NAS files; no target/prompt selection is
    # used to reduce the requested shard.
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_file[str(record["file"])].append(record)
    raw_rows: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    source_started = time.monotonic()
    for filename, requested in sorted(by_file.items()):
        path = pathlib.Path(filename)
        before = path.stat()
        wanted = {int(r["line"]): r for r in requested}
        digest = hashlib.sha256()
        selected = 0
        with path.open("rb", buffering=4 * 1024 * 1024) as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                if line_number not in wanted:
                    continue
                record = wanted[line_number]
                if sha_bytes(raw_line) != record["raw_line_sha256"]:
                    raise RuntimeError(f"raw_line_sha256_mismatch:{record['row_id']}")
                rid = str(record["row_id"])
                raw_rows[rid] = json.loads(raw_line)
                selected += 1
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source_mutated_during_stream:{filename}")
        source_hash = digest.hexdigest()
        if any(str(r["source_sha256"]) != source_hash for r in requested):
            raise RuntimeError(f"source_sha256_mismatch:{filename}")
        source_files.append({"path": filename, "bytes": before.st_size, "sha256": source_hash, "selected_records": selected})

    snapshot_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    mined_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    normalized_source_cache: dict[str, tuple[bytes, str]] = {}
    licenses: dict[str, dict[str, Any]] = {}
    packets: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    repair_queue: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    outcomes = Counter()
    reasons = Counter()
    started = time.monotonic()
    progress_path = out / "progress.json"
    checkpoint_data = {
        "schema": "DAT-10-source-walk-progress-v1",
        "status": "running", "shard": args.shard,
        "input_manifest": str(input_manifest), "input_manifest_sha256": sha_file(input_manifest),
        "attempted": 0, "converted": 0, "excluded": 0,
        "policy": "progress_only; no family/row/time omission",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint(progress_path, checkpoint_data)
    checkpoint(GLOBAL_CHECKPOINT, {"schema": "DAT-10-source-walk-progress-v1", "status": "running", "current_shard": args.shard, "shard": checkpoint_data, "updated_at": checkpoint_data["updated_at"]})

    for index, record in enumerate(records, 1):
        rid = str(record["row_id"])
        family = str(record["family"])
        try:
            raw = raw_rows[rid]
            if raw.get("family") != family:
                raise ValueError("raw_family_mismatch")
            norm = adapter._normalized_path(raw)
            if norm is None:
                raise ValueError("unique_normalized_parent_source_required")
            norm_key = str(norm)
            cached_source = normalized_source_cache.get(norm_key)
            if cached_source is None:
                source_bytes = norm.read_bytes()
                source_hash = sha_bytes(source_bytes)
                if scenarios.parser.parse(source_bytes).root_node.has_error:
                    raise ValueError("normalized_parent_R_parse_error")
                normalized_source_cache[norm_key] = (source_bytes, source_hash)
            else:
                source_bytes, source_hash = cached_source
            key = (family, norm_key, source_hash)
            if family in ("no_op", "roxygen_drafting"):
                if key not in mined_cache:
                    mined: list[dict[str, Any]] = []
                    stats = Counter()
                    if family == "roxygen_drafting":
                        roxygen.extract_file(raw["package"], norm.name, source_bytes, mined, stats)
                    else:
                        suffix.extract_no_op(raw["package"], norm.name, source_bytes, [], Counter(), stats)
                        # The no-op extractor writes into its fourth argument;
                        # invoke it with a retained list to preserve all mined
                        # boundaries rather than a quota-limited subset.
                        mined = []
                        suffix.extract_no_op(raw["package"], norm.name, source_bytes, mined, Counter(), stats)
                    mined_cache[key] = [batch.boundary(r) for r in mined]
                if batch.boundary(raw) not in mined_cache[key]:
                    raise ValueError("fresh_constructor_does_not_reproduce_source_boundary_and_target")
                if family == "no_op" and raw.get("kind") not in ("after_close_brace", "blank_between"):
                    raise ValueError("no_op_kind_requires_separate_support_review")
                if family == "roxygen_drafting" and any(tag in "\n".join(raw.get("region_new", [])) for tag in ("@author", "@references", "@source", "@examples", "@seealso")):
                    raise ValueError("documentation_external_facts_require_separate_support_review")
            else:
                scenarios.validate_example(raw)
            description = norm.parents[1] / "DESCRIPTION"
            description_key = str(description)
            if description_key not in licenses:
                content = description.read_bytes()
                fields = [line for line in content.decode("utf-8").splitlines() if line.startswith(("License:", "License_restricts_use:", "License_is_FOSS:"))]
                if not any(line.startswith("License:") for line in fields):
                    raise ValueError("license_field_missing")
                if any(line in ("License_restricts_use: yes", "License_is_FOSS: no") for line in fields):
                    raise ValueError("source_license_restriction_requires_review")
                licenses[description_key] = {"path": description_key, "sha256": sha_bytes(content), "fields": fields}
            ref = adapter._row_ref_from_audit(record)
            ref["package_id"] = raw["package"]
            snapshot_key = (family, raw.get("package"), raw.get("version"), raw.get("path"), record.get("source"))
            if snapshot_key not in snapshot_cache:
                enriched = adapter._snapshot_ref(raw, dict(ref))
                if enriched is None:
                    raise ValueError("source_snapshot_unresolved")
                snapshot_cache[snapshot_key] = {k: v for k, v in enriched.items() if k not in ref}
            ref.update(copy.deepcopy(snapshot_cache[snapshot_key]))
            result = adapter.convert_structured(raw, ref)
            if result.get("status") != "converted":
                raise ValueError("adapter:" + str(result.get("reason")))
            applied = batch.apply_result(result)
            context, selection = selected_context(result)
            if family == "roxygen_drafting" and list(context.suffix_lines[:len(raw.get("suffix", []))]) != raw.get("suffix", []):
                raise ValueError("required_function_suffix_omitted_by_source_selection")
            if family not in ("no_op", "roxygen_drafting") and len(context.history) != 1:
                raise ValueError("propagation_requires_one_verified_history_event")
            if family in ("rename_propagation", "pipe_rewrite", "na_rm_propagation", "format_propagation") and scenarios.parser.parse(applied.encode()).root_node.has_error:
                raise ValueError("post_edit_full_buffer_R_parse_error")
            packet_ref = adapter._row_ref_from_audit(record)
            packet_ref["package_id"] = raw["package"]
            packet = {
                "row_ref": packet_ref, "family": family, "result": result,
                "validation": {
                    "fresh_source_constructor": True, "full_buffer_application": True,
                    "normalized_parent_R_parse": True, "source_selection_support_check": True,
                    "source_path": str(norm), "source_sha256": sha_bytes(source_bytes),
                    "license_evidence": licenses[description_key],
                    "parent_group_split": "train_group", "admission": False,
                    "quota_free_shard_runner": True,
                },
            }
            packets.append(packet)
            prov_result = result.get("provenance") or {}
            provenance.append({
                "row_id": rid, "group_id": record.get("group_id"), "family": family,
                "source": record.get("source"), "source_file": record.get("file"), "source_line": record.get("line"),
                "package_id": raw.get("package"), "global_split": record.get("split"),
                "cpt_partition": "from_global_cpt_gate_pending_root_review",
                "source_sha256": record.get("source_sha256"), "raw_line_sha256": record.get("raw_line_sha256"),
                "canonical_row_sha256": record.get("canonical_row_sha256"),
                "target_sha256": record.get("target_sha256"), "license_status": record.get("license_status"),
                "source_path": str(norm), "source_file_sha256": sha_bytes(source_bytes),
                "post_edit_snapshot_sha256": prov_result.get("post_edit_snapshot_sha256"),
                "target_truncation": False, "admission": "review_only_unadmitted",
            })
            outcomes[family] += 1
        except (AssertionError, ValueError, KeyError, OSError, UnicodeError) as error:
            reason = str(error) or type(error).__name__
            exclusions.append({
                "row_id": rid, "group_id": record.get("group_id"), "family": family,
                "source": record.get("source"), "source_file": record.get("file"), "source_line": record.get("line"),
                "reason": reason, "status": "source_checked_excluded_or_support_queue",
                "target_sha256": record.get("target_sha256"), "canonical_row_sha256": record.get("canonical_row_sha256"),
                "raw_line_sha256": record.get("raw_line_sha256"), "license_status": record.get("license_status"),
            })
            reasons[reason] += 1
            if any(token in reason for token in ("support_review", "repair", "mixed_document_eol", "boundary", "history_after_prefix")):
                repair_queue.append({
                    "row_id": rid, "group_id": record.get("group_id"), "family": family,
                    "source": record.get("source"), "source_file": record.get("file"), "source_line": record.get("line"),
                    "reason": reason, "repair_status": "queued_for_separate_support_or_repair_review",
                    "target_sha256": record.get("target_sha256"), "raw_line_sha256": record.get("raw_line_sha256"),
                    "canonical_row_sha256": record.get("canonical_row_sha256"),
                })
        if index % 128 == 0:
            checkpoint_data.update({"attempted": index, "converted": len(packets), "excluded": len(exclusions), "updated_at": datetime.now(timezone.utc).isoformat()})
            checkpoint(progress_path, checkpoint_data)
            checkpoint(GLOBAL_CHECKPOINT, {"schema": "DAT-10-source-walk-progress-v1", "status": "running", "current_shard": args.shard, "shard": checkpoint_data, "updated_at": checkpoint_data["updated_at"]})
            print(json.dumps({"stage": "conversion", "shard": args.shard, "attempted": index, "converted": len(packets), "excluded": len(exclusions), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)

    packet_path = out / "candidate-packets.jsonl"
    provenance_path = out / "provenance-ledger.jsonl"
    exclusion_path = out / "exclusions.jsonl"
    repair_path = out / "repair-queue.jsonl"
    summary_path = out / "summary.json"
    manifest_path = out / "manifest.json"
    atomic_jsonl(packet_path, packets)
    atomic_jsonl(provenance_path, provenance)
    atomic_jsonl(exclusion_path, exclusions)
    atomic_jsonl(repair_path, repair_queue)
    summary = {
        "schema": "DAT-10-source-walk-structured-materialization-v1",
        "status": "source_checked_candidates_not_training_registry",
        "shard": args.shard, "input_rows": len(records), "converted": len(packets), "excluded": len(exclusions),
        "converted_families": dict(sorted(outcomes.items())), "exclusion_reasons": dict(sorted(reasons.items())),
        "input": {"path": str(input_path), "sha256": sha_file(input_path), "rows": len(records)},
        "source_files": source_files, "source_seconds": time.monotonic() - source_started,
        "normalized_source_cache_entries": len(normalized_source_cache),
        "constructors": constructors,
        "candidate_packets": {"path": str(packet_path), "sha256": sha_file(packet_path), "rows": len(packets)},
        "provenance": {"path": str(provenance_path), "sha256": sha_file(provenance_path), "rows": len(provenance)},
        "exclusions": {"path": str(exclusion_path), "sha256": sha_file(exclusion_path), "rows": len(exclusions)},
        "repair_queue": {"path": str(repair_path), "sha256": sha_file(repair_path), "rows": len(repair_queue)},
        "target_truncation": False, "heldout_content_included": False, "cuda_started": False,
        "admission": "none; root source/license/duplicate/quality/CPT review remains mandatory",
        "policy": {
            "all_rows_in_prepared_shard_attempted": True, "family_row_time_quotas": False,
            "io_shard_only": True, "long_targets_preserved": True,
            "support_queues_separate": True, "source_content_train_only": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(summary_path, summary)
    manifest = {
        "schema": "DAT-10-source-walk-structured-materialization-manifest-v1",
        "status": "frozen_review_increment_unadmitted",
        "shard": args.shard, "input_manifest": str(input_manifest), "input_manifest_sha256": sha_file(input_manifest),
        "summary": {"path": str(summary_path), "sha256": sha_file(summary_path)},
        "outputs": {
            "candidate_packets": {"path": str(packet_path), "sha256": sha_file(packet_path), "rows": len(packets)},
            "provenance": {"path": str(provenance_path), "sha256": sha_file(provenance_path), "rows": len(provenance)},
            "exclusions": {"path": str(exclusion_path), "sha256": sha_file(exclusion_path), "rows": len(exclusions)},
            "repair_queue": {"path": str(repair_path), "sha256": sha_file(repair_path), "rows": len(repair_queue)},
        },
        "policy": summary["policy"], "admission": summary["admission"],
    }
    atomic_json(manifest_path, manifest)
    checkpoint_data.update({"status": "complete", "attempted": len(records), "converted": len(packets), "excluded": len(exclusions), "manifest": str(manifest_path), "manifest_sha256": sha_file(manifest_path), "updated_at": datetime.now(timezone.utc).isoformat()})
    checkpoint(progress_path, checkpoint_data)
    checkpoint(GLOBAL_CHECKPOINT, {"schema": "DAT-10-source-walk-progress-v1", "status": "complete", "current_shard": args.shard, "shard": checkpoint_data, "updated_at": checkpoint_data["updated_at"]})
    print(json.dumps({"shard": args.shard, "input_rows": len(records), "converted": len(packets), "excluded": len(exclusions), "manifest": str(manifest_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
