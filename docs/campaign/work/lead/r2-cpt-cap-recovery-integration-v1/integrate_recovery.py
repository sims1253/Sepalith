#!/usr/bin/env python3
"""Verify and prepare an append-only augmentation from the v3 recovery.

The recovery controller owns source walking.  This tool is deliberately a
separate, read-only verifier until the root terminal marker says that all 81
groups completed successfully.  It reads the recovery candidate payload only
after that gate, checks every frontier state and CPT chunk, and writes a fresh
augmentation directory.  It never rewrites the terminal union, cache, or
existing schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-cpt-cap-recovery-integration-v1"
FRONTIER = PLAN / "docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/combined-frontier.jsonl"
FRONTIER_MANIFEST = PLAN / "docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/combined-frontier-manifest.json"
V3_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-cpt-prefix-cap-recovery-v3.json"
RECOVERY_OUTPUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3")
ROOT_TERMINAL = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3-root.terminal.json")
TERMINAL_PROVENANCE = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1/document-provenance.jsonl")
DEFAULT_ACCOUNTING = PACKET / "integration-accounting.json"
DEFAULT_AUGMENTATION = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3-integration-v1")
GLOBAL_SPLIT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
PROTECTED = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json"
PROFILE_DOCS = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v1/documents.jsonl"
RAW_CHUNK_CONTRACT = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py"
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")

FRONTIER_SHA256 = "4708d69f53e04c48754e48095497eb1c281acac998dd099620b8fae293a1019d"
FRONTIER_MANIFEST_SHA256 = "26f4799d85d72300f52a733c0cd9c42e9d6fb5b220d56a985538b7efb0bf879c"
V3_RECEIPT_SHA256 = "23201d40ead26ee50f9bab919f38159675553f0a47cbf6bbc80ba08ba2dabefd"
V3_BUILDER_SHA256 = "b37ae0fe27015a6596e1e7b8750c0ab77daed4b8375e10603603407258393fc2"
GLOBAL_SPLIT_SHA256 = "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"
PARTITION_SHA256 = "6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06"
PROTECTED_SHA256 = "7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0"
PROFILE_DOCS_SHA256 = "674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68"
TERMINAL_PROVENANCE_SHA256 = "a731974ee8581b9fa571aae9672f6b933743779c24f4253843bff268065c305f"
RAW_CHUNK_CONTRACT_SHA256 = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
TERMINAL_DOCUMENTS = 177190
FRONTIER_ROWS = 2003
FRONTIER_GROUPS = 81
CHUNK_SIZE = 2048
BOS_ID = 0
EOS_ID = 1
VOCAB_SIZE = 130560


class IntegrationError(RuntimeError):
    """A fail-closed integration or geometry violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrationError(f"file_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrationError(f"jsonl_decode:{path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise IntegrationError(f"jsonl_row_not_object:{path}:{line_number}")
            result.append(value)
    return result


def canonical_write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            count += 1
    return count


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_frontier(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if sha256_file(path) != FRONTIER_SHA256:
        raise IntegrationError("frontier_identity_changed")
    rows = read_jsonl(path)
    if len(rows) != FRONTIER_ROWS:
        raise IntegrationError(f"frontier_count:{len(rows)}")
    if Counter(row.get("scope") for row in rows) != Counter({"prefix_global_cap": 1999, "base_broader_package_cap": 4}):
        raise IntegrationError("frontier_scope_counts_changed")
    ordinals = [row.get("frontier_ordinal") for row in rows]
    if sorted(ordinals) != list(range(1, FRONTIER_ROWS + 1)):
        raise IntegrationError("frontier_ordinals_changed")
    by_path: dict[str, dict[str, Any]] = {}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path_value = row.get("path")
        group = row.get("group_id")
        if not isinstance(path_value, str) or not isinstance(group, str) or path_value in by_path:
            raise IntegrationError("frontier_path_identity_invalid")
        if row.get("split") != "train_group" or row.get("cpt_partition") != "cpt_train":
            raise IntegrationError(f"frontier_not_train:{path_value}")
        by_path[path_value] = row
        by_group[group].append(row)
    if len(by_group) != FRONTIER_GROUPS:
        raise IntegrationError(f"frontier_group_count:{len(by_group)}")
    return rows, by_path, by_group


def load_guard_pins(frontier_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str], set[str]]:
    paths = {
        "frontier": (frontier_path, FRONTIER_SHA256),
        "global_split": (GLOBAL_SPLIT, GLOBAL_SPLIT_SHA256),
        "cpt_partition": (PARTITION, PARTITION_SHA256),
        "protected_parent_hashes": (PROTECTED, PROTECTED_SHA256),
        "profile_documents": (PROFILE_DOCS, PROFILE_DOCS_SHA256),
        "terminal_document_provenance": (TERMINAL_PROVENANCE, TERMINAL_PROVENANCE_SHA256),
    }
    pins: dict[str, dict[str, Any]] = {}
    for label, (path, expected) in paths.items():
        actual = pin(path)
        if actual["sha256"] != expected:
            raise IntegrationError(f"guard_identity_changed:{label}")
        pins[label] = actual
    split_doc = json.loads(GLOBAL_SPLIT.read_text(encoding="utf-8"))
    split_rows = split_doc.get("groups")
    if not isinstance(split_rows, list):
        raise IntegrationError("global_split_groups_missing")
    split_by_group = {row["group_id"]: row.get("split") for row in split_rows if isinstance(row, dict) and isinstance(row.get("group_id"), str)}
    partition_doc = json.loads(PARTITION.read_text(encoding="utf-8"))
    partition_by_group = partition_doc.get("groups")
    if not isinstance(partition_by_group, dict):
        raise IntegrationError("cpt_partition_groups_missing")
    protected = set(json.loads(PROTECTED.read_text(encoding="utf-8")))
    if not protected or not all(isinstance(value, str) and len(value) == 40 for value in protected):
        raise IntegrationError("protected_registry_invalid")
    return pins, split_by_group, {str(key): str(value) for key, value in partition_by_group.items()}, protected


def load_source_artifact_pins() -> dict[str, dict[str, Any]]:
    """Pin the immutable producer metadata and contracts used by the groups.

    The six ``input_pins`` in each group manifest are kept separate because
    they are part of the producer's recorded contract.  These additional
    pins bind the manifest/receipt and the byte-level chunk/tokenizer
    contracts that the integration verifier independently relies on.
    """
    paths = {
        "combined_frontier_manifest": (FRONTIER_MANIFEST, FRONTIER_MANIFEST_SHA256),
        "recovery_v3_receipt": (V3_RECEIPT, V3_RECEIPT_SHA256),
        "raw_chunk_contract": (RAW_CHUNK_CONTRACT, RAW_CHUNK_CONTRACT_SHA256),
        "tokenizer": (TOKENIZER, TOKENIZER_SHA256),
    }
    pins: dict[str, dict[str, Any]] = {}
    for label, (path, expected) in paths.items():
        actual = pin(path)
        if actual["sha256"] != expected:
            raise IntegrationError(f"source_artifact_identity_changed:{label}")
        pins[label] = actual

    frontier_manifest = json.loads(FRONTIER_MANIFEST.read_text(encoding="utf-8"))
    if (
        frontier_manifest.get("sha256") != FRONTIER_SHA256
        or frontier_manifest.get("rows") != FRONTIER_ROWS
        or frontier_manifest.get("groups") != FRONTIER_GROUPS
        or frontier_manifest.get("scope_counts") != {
            "prefix_global_cap": 1999,
            "base_broader_package_cap": 4,
        }
    ):
        raise IntegrationError("frontier_manifest_content_changed")

    v3_receipt = json.loads(V3_RECEIPT.read_text(encoding="utf-8"))
    receipt_frontier = v3_receipt.get("frontier")
    receipt_builder = v3_receipt.get("builder")
    receipt_manifest = receipt_frontier.get("combined_frontier_manifest") if isinstance(receipt_frontier, dict) else None
    if (
        not isinstance(receipt_frontier, dict)
        or receipt_frontier.get("sha256") != FRONTIER_SHA256
        or receipt_frontier.get("rows") != FRONTIER_ROWS
        or receipt_frontier.get("groups") != FRONTIER_GROUPS
        or not isinstance(receipt_builder, dict)
        or receipt_builder.get("sha256") != V3_BUILDER_SHA256
        or not isinstance(receipt_manifest, dict)
        or receipt_manifest.get("sha256") != FRONTIER_MANIFEST_SHA256
    ):
        raise IntegrationError("recovery_v3_receipt_content_changed")
    return pins


def load_terminal_index() -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    terminal_paths: set[str] = set()
    terminal_sha: set[str] = set()
    documents: dict[str, dict[str, Any]] = {}
    rows = read_jsonl(TERMINAL_PROVENANCE)
    if len(rows) != TERMINAL_DOCUMENTS:
        raise IntegrationError(f"terminal_document_count:{len(rows)}")
    for row in rows:
        source_path = row.get("source_path")
        source_sha = row.get("source_sha256")
        if not isinstance(source_path, str) or not isinstance(source_sha, str) or len(source_sha) != 64:
            raise IntegrationError("terminal_document_identity_invalid")
        if source_path in terminal_paths or source_sha in terminal_sha:
            raise IntegrationError("terminal_document_duplicate_identity")
        terminal_paths.add(source_path)
        terminal_sha.add(source_sha)
        documents[source_sha] = row
    return terminal_paths, terminal_sha, documents


def terminal_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrationError(f"root_terminal_missing:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("exit_code") != 0:
        raise IntegrationError(f"recovery_controller_failed:{value.get('exit_code')}")
    return {"path": str(path), "sha256": sha256_file(path), "terminal": value}


def live_status(recovery: Path, root_terminal: Path) -> dict[str, Any]:
    progress_path = recovery / "progress.json"
    if not progress_path.is_file():
        return {"status": "pending", "reason": "recovery_progress_missing", "recovery_output": str(recovery)}
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    root_done = root_terminal.is_file()
    return {
        "status": "ready_for_verify" if root_done and progress.get("status") == "complete_candidate" else "running_or_pending",
        "root_terminal_present": root_done,
        "progress": progress,
        "progress_payload_opened_field_ignored": True,
        "interpretation": "progress payload_opened is initialization metadata; group artifacts determine actual source consumption",
    }


def require_str64(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise IntegrationError(f"invalid_sha256:{name}")
    return value


def require_hex(value: Any, length: int, name: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise IntegrationError(f"invalid_hex:{name}")
    return value


def validate_chunk(row: dict[str, Any], document: dict[str, Any], is_final: bool) -> list[int]:
    ident = row.get("row_id")
    source_sha = document["source_sha256"]
    if row.get("schema") != 1 or row.get("cpt_partition") != "cpt_train" or row.get("document_id") != source_sha:
        raise IntegrationError(f"row_identity_invalid:{ident}")
    if row.get("source_path") != document.get("source_path") or row.get("group_id") != document.get("group_id"):
        raise IntegrationError(f"row_source_binding_invalid:{ident}")
    if row.get("row_id") != f"{source_sha}:{row.get('chunk_index')}":
        raise IntegrationError(f"row_id_invalid:{ident}")
    ids = row.get("input_ids")
    labels = row.get("labels")
    attention = row.get("attention_mask")
    if not isinstance(ids, list) or not 3 <= len(ids) <= CHUNK_SIZE:
        raise IntegrationError(f"row_length_invalid:{ident}")
    if any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in ids):
        raise IntegrationError(f"row_token_range_invalid:{ident}")
    if ids[0] != BOS_ID or ids[-1] != EOS_ID or ids.count(BOS_ID) != 1 or ids.count(EOS_ID) != 1:
        raise IntegrationError(f"row_bos_eos_invalid:{ident}")
    if not isinstance(labels, list) or len(labels) != len(ids) or any(type(value) is not int or value < -100 or value >= VOCAB_SIZE for value in labels):
        raise IntegrationError(f"row_labels_invalid:{ident}")
    if attention != [1] * len(ids):
        raise IntegrationError(f"row_attention_invalid:{ident}")
    overlap = row.get("overlap_context_tokens")
    if not isinstance(overlap, int) or overlap < 0 or overlap > len(ids) - 2:
        raise IntegrationError(f"row_overlap_invalid:{ident}")
    start = row.get("source_token_start")
    end = row.get("source_token_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end - start != len(ids) - 2 - overlap:
        raise IntegrationError(f"row_source_span_invalid:{ident}")
    if labels[0] != -100 or any(value != -100 for value in labels[1:1 + overlap]):
        raise IntegrationError(f"row_mask_invalid:{ident}")
    owned_start = 1 + overlap
    if any(labels[index] != ids[index] for index in range(owned_start, len(ids) - 1)):
        raise IntegrationError(f"row_owned_labels_invalid:{ident}")
    if labels[-1] != (EOS_ID if is_final else -100):
        raise IntegrationError(f"row_terminal_eos_invalid:{ident}")
    if row.get("is_document_end") is not is_final:
        raise IntegrationError(f"row_document_end_flag_invalid:{ident}")
    supervised = sum(value != -100 for value in labels)
    if row.get("supervised_tokens") != supervised:
        raise IntegrationError(f"row_supervised_count_invalid:{ident}")
    if row.get("document_token_count") != document.get("document_token_count"):
        raise IntegrationError(f"row_document_token_count_invalid:{ident}")
    if row.get("tokenizer_revision") != TOKENIZER_REVISION:
        raise IntegrationError(f"row_tokenizer_revision_invalid:{ident}")
    if require_str64(row.get("token_stream_sha256"), f"{ident}.token_stream_sha256") != document.get("token_stream_sha256"):
        raise IntegrationError(f"row_token_stream_hash_mismatch:{ident}")
    return ids[owned_start:-1]


def validate_document(document: dict[str, Any], frontier_row: dict[str, Any], group_id: str,
                      protected: set[str]) -> None:
    source_sha = require_str64(document.get("source_sha256"), "document.source_sha256")
    if document.get("document_id") != source_sha or document.get("source_path") != frontier_row["path"]:
        raise IntegrationError(f"document_frontier_identity_invalid:{frontier_row['path']}")
    expected = frontier_row.get("source_sha256")
    if expected not in (None, "") and source_sha != expected:
        raise IntegrationError(f"document_expected_source_hash_mismatch:{frontier_row['path']}")
    if document.get("group_id") != group_id or document.get("split") != "train_group" or document.get("cpt_partition") != "cpt_train":
        raise IntegrationError(f"document_partition_invalid:{frontier_row['path']}")
    if not isinstance(document.get("chunks"), int) or document["chunks"] < 1:
        raise IntegrationError(f"document_chunk_count_invalid:{frontier_row['path']}")
    if not isinstance(document.get("document_token_count"), int) or document["document_token_count"] < 1:
        raise IntegrationError(f"document_token_count_invalid:{frontier_row['path']}")
    source_sha1 = require_hex(document.get("source_sha1"), 40, "document.source_sha1")
    git_blob = require_hex(document.get("git_blob_sha1"), 40, "document.git_blob_sha1")
    if document.get("source_sha1") in protected or git_blob in protected:
        raise IntegrationError(f"document_protected_parent_identity:{frontier_row['path']}")


def verify_group(group_dir: Path, frontier_rows: list[dict[str, Any]], expected_input_pins: dict[str, dict[str, Any]],
                 protected: set[str]) -> dict[str, Any]:
    manifest_path = group_dir / "manifest.json"
    names = {path.name for path in group_dir.iterdir()}
    required = {"manifest.json", "documents.jsonl", "cpt_train.jsonl", "exclusions.jsonl", "repair-queue.jsonl"}
    if names != required:
        raise IntegrationError(f"group_files_invalid:{group_dir.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_id = manifest.get("group_id")
    if group_id != frontier_rows[0]["group_id"] or manifest.get("status") != "complete_candidate":
        raise IntegrationError(f"group_manifest_identity_invalid:{group_dir.name}")
    expected_name = f"{min(row['frontier_ordinal'] for row in frontier_rows):06d}-{group_id}"
    if group_dir.name != expected_name:
        raise IntegrationError(f"group_directory_identity_invalid:{group_dir.name}:{expected_name}")
    expected_manifest_values = {
        "builder_sha256": V3_BUILDER_SHA256,
        "frontier_path": str(FRONTIER),
        "frontier_sha256": FRONTIER_SHA256,
        "raw_chunk_contract_path": str(RAW_CHUNK_CONTRACT),
        "raw_chunk_contract_sha256": RAW_CHUNK_CONTRACT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_revision": TOKENIZER_REVISION,
        "chunk_size": CHUNK_SIZE,
        "max_source_bytes": None,
        "oversized_source_policy": "retain_and_measure_no_4MiB_cap",
        "truncation": False,
        "frontier_ordinals": sorted(row["frontier_ordinal"] for row in frontier_rows),
        "input_pins": expected_input_pins,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise IntegrationError(f"group_manifest_pin_invalid:{group_id}:{key}")
    artifact_names = ("documents.jsonl", "cpt_train.jsonl", "exclusions.jsonl", "repair-queue.jsonl")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_names):
        raise IntegrationError(f"group_artifact_manifest_invalid:{group_id}")
    for name in artifact_names:
        artifact = group_dir / name
        expected = artifacts[name]
        if not isinstance(expected, dict) or expected.get("bytes") != artifact.stat().st_size or expected.get("sha256") != sha256_file(artifact):
            raise IntegrationError(f"group_artifact_hash_invalid:{group_id}:{name}")
    docs = read_jsonl(group_dir / "documents.jsonl")
    rows = read_jsonl(group_dir / "cpt_train.jsonl")
    exclusions = read_jsonl(group_dir / "exclusions.jsonl")
    repairs = read_jsonl(group_dir / "repair-queue.jsonl")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or any(not isinstance(counts.get(key), int) or counts[key] < 0 for key in ("documents", "rows", "input_tokens", "supervised_tokens", "raw_bytes", "code_tokens")):
        raise IntegrationError(f"group_counts_invalid:{group_id}")
    if counts["documents"] != len(docs) or counts["rows"] != len(rows):
        raise IntegrationError(f"group_counts_geometry_invalid:{group_id}")
    if manifest.get("exclusions", {}) != dict(Counter(value.get("reason") for value in exclusions)):
        raise IntegrationError(f"group_exclusion_counts_invalid:{group_id}")
    if manifest.get("repairs", {}) != dict(Counter(value.get("reason") for value in repairs)):
        raise IntegrationError(f"group_repair_counts_invalid:{group_id}")
    frontier_by_path = {row["path"]: row for row in frontier_rows}
    document_by_sha: dict[str, dict[str, Any]] = {}
    state_paths: list[str] = []
    for document in docs:
        path = document.get("source_path")
        if path not in frontier_by_path:
            raise IntegrationError(f"document_not_in_group_frontier:{group_id}:{path}")
        validate_document(document, frontier_by_path[path], group_id, protected)
        source_sha = document["source_sha256"]
        if source_sha in document_by_sha:
            raise IntegrationError(f"group_duplicate_document:{group_id}:{source_sha}")
        document_by_sha[source_sha] = document
        state_paths.append(path)
    rows_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_sha = row.get("source_sha256")
        if source_sha not in document_by_sha:
            raise IntegrationError(f"row_document_missing:{group_id}:{row.get('row_id')}")
        rows_by_sha[source_sha].append(row)
    token_counts = Counter()
    for source_sha, document in document_by_sha.items():
        document_rows = sorted(rows_by_sha[source_sha], key=lambda value: value.get("chunk_index", -1))
        if len(document_rows) != document["chunks"]:
            raise IntegrationError(f"document_chunk_count_mismatch:{group_id}:{source_sha}")
        if [row.get("chunk_index") for row in document_rows] != list(range(document["chunks"])):
            raise IntegrationError(f"document_chunk_sequence_invalid:{group_id}:{source_sha}")
        owned_tokens: list[int] = []
        previous_end = 0
        for index, row in enumerate(document_rows):
            if row.get("overlap_context_tokens") != (0 if index == 0 else 1):
                raise IntegrationError(f"document_overlap_invalid:{group_id}:{source_sha}:{index}")
            if row.get("source_token_start") != previous_end:
                raise IntegrationError(f"document_source_range_gap:{group_id}:{source_sha}:{index}")
            owned = validate_chunk(row, document, index == len(document_rows) - 1)
            owned_tokens.extend(owned)
            previous_end = row["source_token_end"]
            token_counts["rows"] += 1
            token_counts["input_tokens"] += len(row["input_ids"])
            token_counts["supervised_tokens"] += row["supervised_tokens"]
        if previous_end != document["document_token_count"] or len(owned_tokens) != document["document_token_count"]:
            raise IntegrationError(f"document_token_conservation_invalid:{group_id}:{source_sha}")
        if hashlib.sha256(json.dumps(owned_tokens, separators=(",", ":")).encode("ascii")).hexdigest() != document.get("token_stream_sha256"):
            raise IntegrationError(f"document_token_stream_hash_invalid:{group_id}:{source_sha}")
        if token_counts["rows"] and document_rows[-1].get("is_document_end") is not True:
            raise IntegrationError(f"document_terminal_chunk_missing:{group_id}:{source_sha}")
    for value in exclusions + repairs:
        path = value.get("path")
        if path not in frontier_by_path or value.get("group_id") != group_id or not isinstance(value.get("reason"), str) or not value["reason"]:
            raise IntegrationError(f"group_sidecar_identity_invalid:{group_id}:{path}")
        state_paths.append(path)
    expected_paths = set(frontier_by_path)
    if len(state_paths) != len(set(state_paths)) or set(state_paths) != expected_paths:
        raise IntegrationError(f"group_frontier_coverage_invalid:{group_id}")
    if token_counts["rows"] != counts["rows"] or token_counts["input_tokens"] != counts["input_tokens"] or token_counts["supervised_tokens"] != counts["supervised_tokens"]:
        raise IntegrationError(f"group_token_counts_invalid:{group_id}")
    if sum(value.get("source_bytes", 0) for value in docs) != counts["raw_bytes"] or sum(value.get("source_code_tokens", 0) for value in docs) != counts["code_tokens"]:
        raise IntegrationError(f"group_source_counts_invalid:{group_id}")
    return {"group_id": group_id, "frontier_rows": frontier_rows, "documents": docs, "rows": rows,
            "exclusions": exclusions, "repairs": repairs, "manifest": manifest, "document_by_sha": document_by_sha,
            "rows_by_sha": rows_by_sha, "token_counts": dict(token_counts)}


def verify_recovery(recovery: Path, root_terminal: Path) -> dict[str, Any]:
    terminal = terminal_gate(root_terminal)
    progress_path = recovery / "progress.json"
    manifest_path = recovery / "manifest.json"
    if not progress_path.is_file() or not manifest_path.is_file():
        raise IntegrationError("recovery_terminal_artifacts_missing")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") != "complete_candidate" or progress.get("groups_completed") != FRONTIER_GROUPS or progress.get("groups_remaining") != 0:
        raise IntegrationError("recovery_not_terminal")
    top_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    top_builder = top_manifest.get("builder")
    if (
        top_manifest.get("status") != "CPU_materialized_candidate_not_training_admission"
        or top_manifest.get("frontier_expected_sha256") != FRONTIER_SHA256
        or not isinstance(top_builder, dict)
        or top_builder.get("sha256") != V3_BUILDER_SHA256
        or top_manifest.get("frontier_documents") != FRONTIER_ROWS
        or top_manifest.get("frontier_groups") != FRONTIER_GROUPS
        or top_manifest.get("groups_completed") != FRONTIER_GROUPS
    ):
        raise IntegrationError("recovery_manifest_terminal_identity_invalid")
    frontier, frontier_by_path, frontier_by_group = load_frontier(FRONTIER)
    guard_pins, split_by_group, partition_by_group, protected = load_guard_pins(FRONTIER)
    source_artifact_pins = load_source_artifact_pins()
    for group, rows in frontier_by_group.items():
        if split_by_group.get(group) != "train_group" or partition_by_group.get(group) != "cpt_train":
            raise IntegrationError(f"frontier_guard_not_train:{group}")
    terminal_paths, terminal_sha, _ = load_terminal_index()
    group_root = recovery / "groups"
    if not group_root.is_dir():
        raise IntegrationError("recovery_groups_missing")
    group_dirs = [path for path in group_root.iterdir() if not path.name.startswith(".")]
    expected_names = {f"{min(row['frontier_ordinal'] for row in rows):06d}-{group}" for group, rows in frontier_by_group.items()}
    actual_names = {path.name for path in group_dirs}
    if actual_names != expected_names:
        raise IntegrationError(f"recovery_group_set_invalid:{len(actual_names)}:{len(expected_names)}")
    verified_groups = []
    for group, rows in sorted(frontier_by_group.items(), key=lambda item: min(row["frontier_ordinal"] for row in item[1])):
        verified_groups.append(verify_group(group_root / f"{min(row['frontier_ordinal'] for row in rows):06d}-{group}", rows, guard_pins, protected))
    all_documents: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_exclusions: list[dict[str, Any]] = []
    all_repairs: list[dict[str, Any]] = []
    for value in verified_groups:
        all_documents.extend(value["documents"])
        all_rows.extend(value["rows"])
        all_exclusions.extend(value["exclusions"])
        all_repairs.extend(value["repairs"])
    if len(all_documents) + len(all_exclusions) + len(all_repairs) != FRONTIER_ROWS:
        raise IntegrationError("recovery_state_total_invalid")
    candidate_by_sha: dict[str, dict[str, Any]] = {}
    integration_exclusions: list[dict[str, Any]] = []
    for document in sorted(all_documents, key=lambda value: frontier_by_path[value["source_path"]]["frontier_ordinal"]):
        source_sha = document["source_sha256"]
        source_path = document["source_path"]
        reason = None
        duplicate_of = None
        if source_sha in terminal_sha or source_path in terminal_paths:
            reason = "exact_duplicate_terminal_provenance"
            duplicate_of = "terminal_provenance"
        elif source_sha in candidate_by_sha:
            reason = "exact_duplicate_recovery_document"
            duplicate_of = candidate_by_sha[source_sha]["source_path"]
        if reason:
            integration_exclusions.append({"source_path": source_path, "source_sha256": source_sha, "reason": reason, "duplicate_of": duplicate_of})
        else:
            candidate_by_sha[source_sha] = document
    unique_sha = set(candidate_by_sha)
    rows_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["document_id"] in unique_sha:
            rows_by_sha[row["document_id"]].append(row)
    unique_rows = [row for source_sha in sorted(unique_sha, key=lambda value: frontier_by_path[candidate_by_sha[value]["source_path"]]["frontier_ordinal"]) for row in sorted(rows_by_sha[source_sha], key=lambda value: value["chunk_index"])]
    outcome_by_path = {value["path"]: {"source_path": value["path"], "group_id": value["group_id"], "frontier_ordinal": value["frontier_ordinal"], "scope": value["scope"]} for value in frontier}
    for document in all_documents:
        outcome_by_path[document["source_path"]].update({"outcome": "admitted_candidate" if document["source_sha256"] in unique_sha else "integration_dedup_excluded", "source_sha256": document["source_sha256"]})
    for value in all_exclusions:
        outcome_by_path[value["path"]].update({"outcome": "materializer_excluded", "reason": value["reason"], "source_sha256": value.get("source_sha256")})
    for value in all_repairs:
        outcome_by_path[value["path"]].update({"outcome": "materializer_repair", "reason": value["reason"]})
    token_totals = Counter()
    source_totals = Counter()
    for group in verified_groups:
        for key, value in group["token_counts"].items():
            token_totals[key] += value
        source_totals["raw_bytes"] += sum(value.get("source_bytes", 0) for value in group["documents"])
        source_totals["code_tokens"] += sum(value.get("source_code_tokens", 0) for value in group["documents"])
    aggregate_counts = {
        "documents": len(all_documents),
        "rows": token_totals["rows"],
        "input_tokens": token_totals["input_tokens"],
        "supervised_tokens": token_totals["supervised_tokens"],
        "raw_bytes": source_totals["raw_bytes"],
        "code_tokens": source_totals["code_tokens"],
    }
    if top_manifest.get("counts") != aggregate_counts:
        raise IntegrationError("recovery_top_counts_invalid")
    if top_manifest.get("exclusions", {}) != dict(Counter(value.get("reason") for value in all_exclusions)):
        raise IntegrationError("recovery_top_exclusion_counts_invalid")
    if top_manifest.get("repairs", {}) != dict(Counter(value.get("reason") for value in all_repairs)):
        raise IntegrationError("recovery_top_repair_counts_invalid")
    accounting = {
        "schema": "sepalith.dat10.cpt-cap-recovery-integration-accounting.v1",
        "status": "verified_candidate_append_pending_root_admission",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "recovery_output": str(recovery),
        "root_terminal": terminal,
        "progress_payload_opened_field_ignored": True,
        "frontier": {"path": str(FRONTIER), "sha256": FRONTIER_SHA256, "rows": FRONTIER_ROWS, "groups": FRONTIER_GROUPS, "scope_counts": {"prefix_global_cap": 1999, "base_broader_package_cap": 4}},
        "source_guard_pins": guard_pins,
        "source_artifact_pins": source_artifact_pins,
        "terminal_union": {"provenance": pin(TERMINAL_PROVENANCE), "documents": TERMINAL_DOCUMENTS, "unique_source_sha256": len(terminal_sha), "payload_opened": False},
        "outcomes": {"frontier_rows": FRONTIER_ROWS, "materialized_documents": len(all_documents), "materializer_exclusions": len(all_exclusions), "materializer_repairs": len(all_repairs), "integration_dedup_exclusions": len(integration_exclusions), "unique_candidate_documents": len(candidate_by_sha), "unique_candidate_rows": len(unique_rows)},
        "token_conservation": {**dict(token_totals), **dict(source_totals)},
        "integration_exclusions": integration_exclusions,
        "frontier_outcomes": [outcome_by_path[path] for path in sorted(outcome_by_path, key=lambda value: outcome_by_path[value]["frontier_ordinal"])],
        "append_policy": {"append_only": True, "terminal_union_unchanged": True, "cache_unchanged": True, "existing_schedule_unchanged": True, "prompt_target_dedup_remains_root_gate": True},
        "training_admission": False,
    }
    return {"accounting": accounting, "frontier": frontier, "frontier_by_path": frontier_by_path, "groups": verified_groups,
            "documents": all_documents, "rows": all_rows, "unique_documents": list(candidate_by_sha.values()), "unique_rows": unique_rows,
            "integration_exclusions": integration_exclusions, "guard_pins": guard_pins, "terminal": terminal}


def load_checkpoint(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrationError("source_checkpoint_not_object")
    step = value.get("checkpoint_step", value.get("step"))
    cursor = value.get("source_cursor")
    checkpoint_id = value.get("checkpoint_id", value.get("id"))
    if not isinstance(step, int) or step < 0 or not isinstance(cursor, int) or cursor < 0 or not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise IntegrationError("source_checkpoint_requires_explicit_step_cursor_id")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "checkpoint_step": step, "source_cursor": cursor, "checkpoint_id": checkpoint_id, "raw": value}


def build_augmentation(result: dict[str, Any], output: Path, checkpoint_path: Path, effective_batch: int) -> dict[str, Any]:
    if effective_batch < 1:
        raise IntegrationError("effective_batch_must_be_positive")
    checkpoint = load_checkpoint(checkpoint_path)
    if output.exists() and any(output.iterdir()):
        raise IntegrationError(f"augmentation_output_must_be_fresh:{output}")
    output.mkdir(parents=True, exist_ok=True)
    documents = sorted(result["unique_documents"], key=lambda value: result["frontier_by_path"][value["source_path"]]["frontier_ordinal"])
    rows = result["unique_rows"]
    document_path = output / "augmentation-document-provenance.jsonl"
    row_path = output / "augmentation-cpt-train.jsonl"
    outcome_path = output / "frontier-outcomes.jsonl"
    exclusion_path = output / "integration-exclusions.jsonl"
    canonical_write(document_path, documents)
    canonical_write(row_path, rows)
    canonical_write(outcome_path, result["accounting"]["frontier_outcomes"])
    canonical_write(exclusion_path, result["integration_exclusions"])
    row_ids = [row["row_id"] for row in rows]
    alignment_count = (-len(row_ids)) % effective_batch
    alignment_ids = row_ids[:alignment_count]
    schedule = {
        "schema": "sepalith.dat10.cpt-cap-recovery-augmentation-schedule.v1",
        "status": "candidate_pending_root_admission",
        "method": "append_unique_one_pass_plus_named_alignment_replay_v1",
        "source_checkpoint": checkpoint,
        "effective_batch": effective_batch,
        "unique_candidate_documents": len(documents),
        "unique_candidate_rows": len(row_ids),
        "unique_candidate_row_ids_sha256": hashlib.sha256(json.dumps(row_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "unique_first_draw_policy": "each unique recovery row exactly once in frontier/group/chunk order",
        "alignment_replay_policy": "only named new-recovery row IDs needed to align this augmentation to effective_batch",
        "alignment_replay_count": alignment_count,
        "alignment_replay_row_ids": alignment_ids,
        "draw_count": len(row_ids) + alignment_count,
        "existing_schedule_policy": "preserve unconsumed existing schedule from explicit source checkpoint; do not replay consumed initial rows",
        "global_source_cursor": checkpoint["source_cursor"],
        "training_admission": False,
    }
    schedule_path = output / "augmentation-schedule.json"
    atomic_write_json(schedule_path, schedule)
    artifact_names = [document_path, row_path, outcome_path, exclusion_path, schedule_path]
    manifest = {
        "schema": "sepalith.dat10.cpt-cap-recovery-augmentation-manifest.v1",
        "status": "append_only_candidate_pending_root_admission",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_accounting": result["accounting"],
        "source_checkpoint": checkpoint,
        "effective_batch": effective_batch,
        "artifacts": {path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifact_names},
        "rows": len(row_ids), "documents": len(documents), "alignment_replay_count": alignment_count,
        "append_only": True, "terminal_union_unchanged": True, "existing_schedule_unchanged": True,
        "training_admission": False,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {"output": str(output), "manifest": manifest, "schedule": schedule}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("status", "verify", "build"), default="status")
    ap.add_argument("--recovery-output", type=Path, default=RECOVERY_OUTPUT)
    ap.add_argument("--root-terminal", type=Path, default=ROOT_TERMINAL)
    ap.add_argument("--accounting-output", type=Path, default=DEFAULT_ACCOUNTING)
    ap.add_argument("--augmentation-output", type=Path, default=DEFAULT_AUGMENTATION)
    ap.add_argument("--source-checkpoint-json", type=Path)
    ap.add_argument("--effective-batch", type=int)
    args = ap.parse_args()
    if args.mode == "status":
        print(json.dumps(live_status(args.recovery_output, args.root_terminal), indent=2, sort_keys=True))
        return 0
    if not args.root_terminal.is_file():
        print(json.dumps(live_status(args.recovery_output, args.root_terminal), indent=2, sort_keys=True))
        return 2
    result = verify_recovery(args.recovery_output, args.root_terminal)
    args.accounting_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.accounting_output, result["accounting"])
    response: dict[str, Any] = {"mode": args.mode, "accounting": str(args.accounting_output), "accounting_sha256": sha256_file(args.accounting_output), "outcomes": result["accounting"]["outcomes"]}
    if args.mode == "build":
        if args.source_checkpoint_json is None or args.effective_batch is None:
            raise IntegrationError("build_requires_explicit_source_checkpoint_json_and_effective_batch")
        response["augmentation"] = build_augmentation(result, args.augmentation_output, args.source_checkpoint_json, args.effective_batch)
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IntegrationError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        raise SystemExit(3)
