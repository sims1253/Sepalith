#!/usr/bin/env python3
"""Prepare a deterministic representative complete-document CPT pilot.

The frozen 16K frontier is already tokenized and provenance-bound.  Selection
is made over complete document provenance records with a stable SHA-256 rank,
so no source length or package quota is imposed.  The materialization pass
then streams the frozen 16K rows once and publishes every row belonging to a
selected document.  All unselected documents remain in an explicit queue.

This packet is diagnostic data only.  It never reads DEV/final payloads and
never changes the frozen frontier or trainer code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import uuid
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
FROZEN = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-16k-frontier-v1")
PROVENANCE = FROZEN / "document-provenance.jsonl"
ROWS_16K = FROZEN / "cpt_train_ctx16384.jsonl"
FROZEN_RESULT = FROZEN / "result.json"
FROZEN_RESULT_SHA = "7e90b54667d3d37032067329a1d4605e56e0e51b09d0e69f9e149f060363b26d"
PROVENANCE_SHA = "be3aba370a0adf50b4dad6819c5009171286e8b70dcffbfe2ebec7bf8053512c"
ROWS_16K_SHA = "16980ad018bcc6a0213e13e1a07849f9df044538e18cfcea8b0d8eaa9de2807e"
SEED = "SFT11-DAT10-representative-v1"
SCHEDULE_SEED = 3407
TARGET_DOCUMENTS = 1024
EFFECTIVE_BATCH = 16
GROUP_ID_REQUIRED_SPLIT = "train_group"
PARTITION_REQUIRED = "cpt_train"
TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
RAW_CHUNKS_SHA = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
MAX_CONTEXT = 16384


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def selection_rank(document_id: str) -> str:
    return hashlib.sha256((SEED + "\0" + document_id).encode("ascii")).hexdigest()


def length_bucket(value: int) -> str:
    if value <= 512:
        return "1-512"
    if value <= 1024:
        return "513-1024"
    if value <= 2048:
        return "1025-2048"
    if value <= 4096:
        return "2049-4096"
    if value <= 8192:
        return "4097-8192"
    if value <= 16384:
        return "8193-16384"
    if value <= 32768:
        return "16385-32768"
    if value <= 65536:
        return "32769-65536"
    if value <= 131072:
        return "65537-131072"
    return "131073+"


def selection_records() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not PROVENANCE.is_file() or not ROWS_16K.is_file() or not FROZEN_RESULT.is_file():
        raise FileNotFoundError("frozen_16k_frontier_missing")
    if sha_file(PROVENANCE) != PROVENANCE_SHA:
        raise ValueError("frozen_provenance_sha256_mismatch")
    frozen_result = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    if sha_file(FROZEN_RESULT) != FROZEN_RESULT_SHA or frozen_result.get("status") != "complete":
        raise ValueError("frozen_result_identity_or_status_mismatch")
    if frozen_result.get("guarantees", {}).get("all_source_tokens_once_per_context") is not True:
        raise ValueError("frozen_frontier_lossless_guarantee_missing")
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(PROVENANCE.open(encoding="utf-8"), 1):
        row = json.loads(line)
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or len(document_id) != 64 or document_id in by_id:
            raise ValueError(f"provenance_document_identity_invalid:{line_number}")
        # The frozen frontier provenance schema carries cpt_partition and
        # group_id, while the global registry's train_group decision is bound
        # by the frozen frontier receipt rather than duplicated per document.
        if row.get("cpt_partition") != PARTITION_REQUIRED:
            raise ValueError(f"frozen_non_train_row:{line_number}")
        if row.get("source_sha256") != document_id or row.get("document_token_count", 0) <= 0:
            raise ValueError(f"provenance_source_identity_invalid:{line_number}")
        row = dict(row)
        row["split"] = GROUP_ID_REQUIRED_SPLIT
        row["selection_rank"] = selection_rank(document_id)
        row["selection_status"] = "unselected_queued_for_production"
        records.append(row)
        by_id[document_id] = row
    if len(records) != frozen_result.get("totals", {}).get("documents"):
        raise ValueError("provenance_document_count_differs_from_frozen_result")
    return records, by_id


def write_selection(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"selection output must be fresh: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    records, by_id = selection_records()
    ordered = sorted(records, key=lambda row: (row["selection_rank"], row["document_id"]))
    selected = ordered[:TARGET_DOCUMENTS]
    selected_ids = {row["document_id"] for row in selected}
    if len(selected_ids) != TARGET_DOCUMENTS:
        raise ValueError("selected_document_ids_not_unique")
    unselected = [row for row in records if row["document_id"] not in selected_ids]
    selected_lengths = [int(row["document_token_count"]) for row in selected]
    package_counts = Counter(row["package"] for row in selected)
    group_counts = Counter(row["group_id"] for row in selected)
    selected_metadata = []
    for row in selected:
        selected_metadata.append({
            "document_id": row["document_id"], "source_sha256": row["source_sha256"],
            "package": row["package"], "group_id": row["group_id"],
            "cpt_partition": row["cpt_partition"], "split": row["split"],
            "source_path": row["source_path"], "document_token_count": row["document_token_count"],
            "original_chunk_count": row["original_chunk_count"], "selection_rank": row["selection_rank"],
            "selection_status": "selected_complete_document",
        })
    queue = []
    for row in unselected:
        queue.append({
            "document_id": row["document_id"], "source_sha256": row["source_sha256"],
            "package": row["package"], "group_id": row["group_id"],
            "cpt_partition": row["cpt_partition"], "split": row["split"],
            "source_path": row["source_path"], "document_token_count": row["document_token_count"],
            "original_chunk_count": row["original_chunk_count"], "selection_rank": row["selection_rank"],
            "selection_status": "unselected_queued_for_production",
        })
    write_jsonl(args.output / "selected-document-provenance.jsonl", selected_metadata)
    write_jsonl(args.output / "unselected-document-queue.jsonl", queue)
    selected_ids_path = args.output / "selected-document-ids.jsonl"
    with selected_ids_path.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(row["document_id"] + "\n")
    bindings = {
        "schema": "sepalith.sft11.representative_pilot_source_binding.v1",
        "frozen_result": {"path": str(FROZEN_RESULT), "sha256": FROZEN_RESULT_SHA},
        "frozen_provenance": {"path": str(PROVENANCE), "sha256": PROVENANCE_SHA},
        # The 3.25 GB frontier row stream is hashed during materialization's
        # single source pass. Selection reads only the 92 MB provenance file.
        "frozen_rows_16k": {"path": str(ROWS_16K), "sha256": ROWS_16K_SHA, "bytes": ROWS_16K.stat().st_size},
        "raw_chunks_sha256": RAW_CHUNKS_SHA, "tokenizer_sha256": TOKENIZER_SHA,
        "global_split": "train_group", "cpt_partition": "cpt_train",
        "heldout_policy": "Input is the frozen CPT TRAIN 16K frontier; DEV/final/CPT-validation payloads were excluded upstream before this packet.",
    }
    write_json(args.output / "source-bindings.json", bindings)
    manifest = {
        "schema": "sepalith.sft11.representative_full_document_pilot_selection.v1",
        "status": "selection_complete_materialization_pending",
        "selection": {
            "seed": SEED, "algorithm": "SHA256(seed\\0document_id) ascending over complete frozen provenance documents",
            "target_documents": TARGET_DOCUMENTS, "selected_documents": len(selected), "unselected_documents_queued": len(unselected),
            "length_filter": None, "package_quota": None, "family_quota": None,
            "natural_length_distribution": True, "selection_order_is_document_level": True,
        },
        "selected": {
            "document_ids_path": str(selected_ids_path), "document_ids_sha256": sha_file(selected_ids_path),
            "provenance_path": str(args.output / "selected-document-provenance.jsonl"),
            "provenance_sha256": sha_file(args.output / "selected-document-provenance.jsonl"),
            "documents": len(selected), "packages": len(package_counts), "groups": len(group_counts),
            "payload_tokens": sum(selected_lengths), "length_buckets": dict(sorted(Counter(length_bucket(x) for x in selected_lengths).items())),
            "length_min": min(selected_lengths), "length_median": statistics.median(selected_lengths),
            "length_p90": sorted(selected_lengths)[math.ceil(0.90 * len(selected_lengths)) - 1],
            "length_p99": sorted(selected_lengths)[math.ceil(0.99 * len(selected_lengths)) - 1],
            "length_max": max(selected_lengths), "package_counts": dict(sorted(package_counts.items())),
        },
        "queued": {
            "path": str(args.output / "unselected-document-queue.jsonl"),
            "sha256": sha_file(args.output / "unselected-document-queue.jsonl"),
            "documents": len(unselected), "all_unselected_frozen_documents_queued": len(unselected) == len(records) - len(selected),
        },
        "source_bindings": bindings,
        "training_admission": False, "diagnostic_cohort_only": True, "production_cap": False,
        "payloads_materialized": False,
    }
    write_json(args.output / "selection-manifest.json", manifest)
    return manifest


def validate_doc_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> int:
    if not rows:
        raise ValueError(f"selected_document_has_no_rows:{metadata['document_id']}")
    document_id = metadata["document_id"]
    # ``original_chunk_count`` belongs to the frozen 2K input artifact. A
    # complete 16K document may have fewer output rows, so output chunk count
    # is proved from contiguous spans and the terminal marker below.
    for index, row in enumerate(rows):
        if row.get("document_id") != document_id or row.get("source_sha256") != document_id:
            raise ValueError(f"selected_document_identity_mismatch:{document_id}")
        if row.get("package") != metadata["package"] or row.get("group_id") != metadata["group_id"] or row.get("cpt_partition") != "cpt_train":
            raise ValueError(f"selected_document_provenance_mismatch:{document_id}")
        if row.get("chunk_index") != index or row.get("row_id") != f"{document_id}:ctx16384:{index}":
            raise ValueError(f"selected_document_chunk_order_mismatch:{document_id}")
        if len(row.get("input_ids", [])) > MAX_CONTEXT or len(row.get("input_ids", [])) < 3:
            raise ValueError(f"selected_document_context_length_invalid:{document_id}")
        if row.get("token_start") != row.get("source_token_start") or row.get("token_end") != row.get("source_token_end"):
            raise ValueError(f"selected_document_span_alias_mismatch:{document_id}")
        if row.get("overlap_context_tokens") != (0 if index == 0 else 1):
            raise ValueError(f"selected_document_overlap_mismatch:{document_id}")
    expected_start = 0
    for row in rows:
        if row["source_token_start"] != expected_start:
            raise ValueError(f"selected_document_source_gap:{document_id}")
        expected_start = row["source_token_end"]
    if expected_start != metadata["document_token_count"] or not rows[-1].get("is_document_end"):
        raise ValueError(f"selected_document_not_complete:{document_id}")
    return sum(row["source_token_end"] - row["source_token_start"] for row in rows)


def load_validator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("representative_pilot_frozen_validator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot_load_validator:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    selection_manifest_path = args.output / "selection-manifest.json"
    if not selection_manifest_path.is_file():
        raise FileNotFoundError("selection_manifest_required")
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection_manifest.get("schema") != "sepalith.sft11.representative_full_document_pilot_selection.v1":
        raise ValueError("selection_manifest_schema_mismatch")
    if selection_manifest.get("status") != "selection_complete_materialization_pending":
        raise ValueError("selection_manifest_not_pending")
    if sha_file(PROVENANCE) != PROVENANCE_SHA or sha_file(FROZEN_RESULT) != FROZEN_RESULT_SHA:
        raise ValueError("frozen_input_changed_before_materialization")
    bindings = json.loads((args.output / "source-bindings.json").read_text(encoding="utf-8"))
    expected_rows_sha = bindings["frozen_rows_16k"]["sha256"]
    if expected_rows_sha != ROWS_16K_SHA:
        raise ValueError("source_binding_rows_sha256_not_pinned")
    if bindings["frozen_rows_16k"]["bytes"] != ROWS_16K.stat().st_size:
        raise ValueError("frozen_row_bytes_changed_before_materialization")
    selected_meta = {row["document_id"]: row for line in (args.output / "selected-document-provenance.jsonl").open(encoding="utf-8") for row in [json.loads(line)]}
    selected_ids = set(selected_meta)
    if len(selected_ids) != TARGET_DOCUMENTS:
        raise ValueError("selected_metadata_count_mismatch")
    validator_path = args.validator
    if not validator_path.is_file():
        raise FileNotFoundError("frozen_validator_missing")
    stage = args.output / (".materialization-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    selected_rows: list[dict[str, Any]] = []
    selected_seen: set[str] = set()
    input_seen: set[str] = set()
    closed_ids: set[str] = set()
    current_id: str | None = None
    current_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    payload_hash = hashlib.sha256()
    start_stat = ROWS_16K.stat()
    try:
        with ROWS_16K.open("rb") as binary, (stage / "cpt_train_ctx16384.jsonl").open("x", encoding="utf-8") as selected_stream:
            for line_number, raw_line in enumerate(binary, 1):
                payload_hash.update(raw_line)
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"frozen_rows_invalid_json:{line_number}") from exc
                document_id = row.get("document_id")
                if not isinstance(document_id, str):
                    raise ValueError(f"frozen_row_missing_document_id:{line_number}")
                if row.get("cpt_partition") != PARTITION_REQUIRED or row.get("group_id") is None:
                    raise ValueError(f"frozen_row_train_identity_invalid:{line_number}")
                input_seen.add(document_id)
                if current_id is None:
                    current_id = document_id
                elif document_id != current_id:
                    if document_id in closed_ids:
                        raise ValueError(f"frozen_document_rows_not_contiguous:{document_id}")
                    if current_id in selected_ids:
                        total = validate_doc_rows(current_rows, selected_meta[current_id])
                        for selected_row in current_rows:
                            selected_stream.write(json.dumps(selected_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                            selected_rows.append(selected_row)
                        selected_seen.add(current_id); counts["selected_documents"] += 1; counts["selected_payload_tokens"] += total; counts["selected_rows"] += len(current_rows)
                    closed_ids.add(current_id)
                    current_id = document_id; current_rows = []
                if document_id in selected_ids:
                    current_rows.append(row)
                counts["input_rows"] += 1
            if current_id in selected_ids:
                total = validate_doc_rows(current_rows, selected_meta[current_id])
                for selected_row in current_rows:
                    selected_stream.write(json.dumps(selected_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    selected_rows.append(selected_row)
                selected_seen.add(current_id); counts["selected_documents"] += 1; counts["selected_payload_tokens"] += total; counts["selected_rows"] += len(current_rows)
                closed_ids.add(current_id)
        end_stat = ROWS_16K.stat()
        if (start_stat.st_dev, start_stat.st_ino, start_stat.st_size, start_stat.st_mtime_ns) != (end_stat.st_dev, end_stat.st_ino, end_stat.st_size, end_stat.st_mtime_ns):
            raise ValueError("frozen_rows_changed_during_stream")
        if payload_hash.hexdigest() != expected_rows_sha:
            raise ValueError("frozen_rows_sha256_mismatch_after_stream")
        if input_seen != set(selected_meta) | {row["document_id"] for line in (args.output / "unselected-document-queue.jsonl").open(encoding="utf-8") for row in [json.loads(line)]}:
            raise ValueError("frozen_input_document_set_differs_from_selection_queue")
        if selected_seen != selected_ids or counts["selected_documents"] != TARGET_DOCUMENTS:
            raise ValueError("not_all_selected_documents_recovered")
        frozen_validator = load_validator(validator_path)
        validation = frozen_validator.validate_materialized_rows(
            selected_rows, max_sequence_tokens=MAX_CONTEXT, require_complete_documents=True
        )
        if validation["documents"] != TARGET_DOCUMENTS or validation["rows"] != counts["selected_rows"]:
            raise ValueError("frozen_validator_selected_totals_mismatch")
        if validation["payload_tokens"] != counts["selected_payload_tokens"]:
            raise ValueError("frozen_validator_selected_payload_mismatch")
        # Write the selected provenance in frozen order and a zero-row repair/exclusion ledger.
        selected_docs = [selected_meta[document_id] for document_id in sorted(selected_seen, key=lambda value: (selection_rank(value), value))]
        write_jsonl(stage / "selected-document-provenance.jsonl", selected_docs)
        write_jsonl(stage / "exclusions.jsonl", [])
        write_jsonl(stage / "repair-queue.jsonl", [])
        row_ids = [row["row_id"] for row in selected_rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("selected_row_ids_not_unique")
        random_order = list(row_ids)
        random.Random(SCHEDULE_SEED).shuffle(random_order)
        replay_count = (-len(random_order)) % EFFECTIVE_BATCH
        replay = random_order[:replay_count]
        draws = random_order + replay
        schedule = {
            "schema": "sepalith.sft11.representative_full_document_pilot_schedule.v1",
            "status": "diagnostic_only_pending_root_admission",
            "split_id": "DAT10-CPT-TRAIN-representative-full-doc-pilot-v1",
            "method": "one_pass_plus_named_replay_v1", "seed": SCHEDULE_SEED,
            "effective_batch": EFFECTIVE_BATCH, "max_steps": len(draws) // EFFECTIVE_BATCH,
            "token_rows_sha256": None, "row_ids": draws, "replay_row_ids": replay,
            "selected_documents": TARGET_DOCUMENTS, "selected_rows": len(row_ids),
            "replay_count": replay_count, "replay_bound": replay_count <= 15,
            "all_unselected_documents_queued": True, "training_admission": False,
        }
        schedule["token_rows_sha256"] = sha_file(stage / "cpt_train_ctx16384.jsonl")
        write_json(stage / "draw-schedule.json", schedule)
        # ``result.json`` contains this artifact map, so including its own
        # digest would create a self-referential value. The receipt records the
        # final result hash separately; this map covers the immutable payload
        # and sidecar files.
        artifacts = {}
        for path in sorted(stage.iterdir()):
            if path.is_file() and path.name != "result.json":
                artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
        result = {
            "schema": "sepalith.sft11.representative_full_document_pilot_result.v1",
            "status": "complete_diagnostic_candidate_pending_final_dedup_root_admission",
            "source": {"frozen_frontier": str(FROZEN), "frozen_rows_sha256": expected_rows_sha, "frozen_rows_bytes": start_stat.st_size,
                       "frozen_document_count": len(input_seen), "frozen_document_provenance_sha256": PROVENANCE_SHA},
            "selection": selection_manifest["selection"],
            "selected": {"documents": counts["selected_documents"], "rows_16k": counts["selected_rows"],
                         "payload_tokens": counts["selected_payload_tokens"], "packages": len({row["package"] for row in selected_docs}),
                         "groups": len({row["group_id"] for row in selected_docs}), "length_buckets": selection_manifest["selected"]["length_buckets"]},
            "input": {"rows_16k": counts["input_rows"], "documents": len(input_seen)},
            "queued": {"documents": len(input_seen) - counts["selected_documents"], "all_unselected_documents_queued": True},
            "draw_schedule": {"path": str(args.output / "draw-schedule.json"), "rows": len(draws), "max_steps": schedule["max_steps"], "replay_count": replay_count},
            "validation": {
                "status": "pass", "validator_path": str(validator_path),
                "validator_sha256": sha_file(validator_path),
                "documents": validation["documents"], "rows": validation["rows"],
                "input_tokens": validation["input_tokens"], "payload_tokens": validation["payload_tokens"],
                "loss_tokens": validation["loss_tokens"],
                "complete_documents_required": True,
            },
            "artifacts": artifacts,
            "coverage": {"all_selected_documents_complete": True, "all_selected_chunks_retained": True, "all_source_tokens_conserved": True,
                         "source_length_filter": None, "package_quota": None, "length_quota": None, "target_truncation": False,
                         "unselected_queued": True, "final_global_dedup_pending": True, "training_admission": False},
        }
        write_json(stage / "result.json", result)
        # Replace only new candidate files. Selection metadata remains the immutable input plan.
        for path in stage.iterdir():
            path.replace(args.output / path.name)
        stage.rmdir()
        final_artifacts = {name: {"bytes": (args.output / name).stat().st_size, "sha256": sha_file(args.output / name)} for name in ["cpt_train_ctx16384.jsonl", "selected-document-provenance.jsonl", "exclusions.jsonl", "repair-queue.jsonl", "draw-schedule.json"]}
        result["artifacts"] = final_artifacts
        write_json(args.output / "result.json", result)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--validator", type=Path, default=PLAN / "docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/campaign_cpt_data.py")
    args = parser.parse_args()
    if args.select_only == args.materialize:
        raise ValueError("choose_exactly_one_of_select_only_or_materialize")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    if args.select_only:
        result = write_selection(args)
    else:
        result = materialize(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
