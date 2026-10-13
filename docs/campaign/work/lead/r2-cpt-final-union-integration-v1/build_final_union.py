#!/usr/bin/env python3
"""Build the all-source CPT candidate union behind explicit terminal gates.

This packet closes a provenance gap in the older frontier helpers.  The
``preview`` mode reads only already-frozen alias and repair candidates and
emits a small, review-only candidate stream.  The ``final`` mode refuses to
open the moving main payload until the main 8,092-group materializer and the
alias terminal rescan have both reached their recorded gates.  It then
streams the prior CPT TRAIN shards, the terminal main/alias groups, and the
supported repair candidates through one fail-closed validator and exact
document-source deduplicator.

The output is a 2,048-token schema-1 candidate stream.  Lossless 16K
rechunking is a separately reviewed operation; this script never truncates or
re-tokenizes a row and never marks training admission.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
E = Path("/mnt/e/sepalith/campaign-20260915/data-work")
EXPECTED_GROUPS = 8092
MAIN_START = 775
MAIN_END = MAIN_START + EXPECTED_GROUPS
VOCAB_SIZE = 130560
BOS_ID = 0
EOS_ID = 1

MAIN = E / "CPT-all-eligible-v1"
ALIAS = E / "CPT-license-alias-recovery-v1"
SVARS = E / "CPT-three-repair-recovery-v1/svars/cpt_train.jsonl"
TFP = E / "CPT-tfprobability-recovery-v1/cpt_train.jsonl"
BASE_BROAD = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/broader-shard-v1-2k/cpt_train.jsonl"
BASE_GLOBAL = PLAN / "docs/campaign/work/r2-cpt-global-shard-v1/shard/cpt_train.jsonl"
PROFILE_DOCS = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v1/documents.jsonl"
MAIN_RUN_MANIFEST = MAIN / "run-manifest.json"
ALIAS_BINDING = ALIAS / "binding.json"
MAIN_MATERIALIZER = PLAN / "docs/campaign/work/lead/r2-cpt-all-eligible-v2/materialize_all_eligible.py"
ALIAS_MATERIALIZER = PLAN / "docs/campaign/work/lead/r2-cpt-license-alias-recovery-v1/source/materialize_alias_recovery.py"
GLOBAL_SPLIT = E / "DAT-02-global-split-v2.json"
PARTITION = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
PROTECTED = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json"
RAW_CHUNKS = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py"
VALIDATOR = PLAN / "docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/campaign_cpt_data.py"
RECHUNKER = PLAN / "docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/lossless_rechunk.py"
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")

PINS = {
    "global_split": (GLOBAL_SPLIT, "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"),
    "cpt_partition": (PARTITION, "6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06"),
    "protected_parent_hashes": (PROTECTED, "7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0"),
    "profile_validation_documents": (PROFILE_DOCS, "674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68"),
    "main_run_manifest": (MAIN_RUN_MANIFEST, "a23564dfcfaa15962aedfc467d5463f4cf27567cb958d10d730abb8801d8132e"),
    "alias_binding": (ALIAS_BINDING, "a9369001712e1fd658519ae0de62c86be9e9e1e9ddea4d496abe0cc21ea0e56b"),
    "main_materializer": (MAIN_MATERIALIZER, "03a1a01d55dd86d12d897ec3a47892f432c7a6b1388a4a1dd5f4d5f226fbd404"),
    "alias_materializer": (ALIAS_MATERIALIZER, "f8358a0414d58e1336437035b44b078a5bf1747f61e46a7468ad1c7f54586dbf"),
    "raw_chunks": (RAW_CHUNKS, "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"),
    "validator": (VALIDATOR, "8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa"),
    "tokenizer": (TOKENIZER, "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"),
}


class UnionError(ValueError):
    """A source, split, geometry, or terminal-state contract failed."""


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_validator():
    spec = importlib.util.spec_from_file_location("campaign_cpt_data_union_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise UnionError("validator_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pins() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label, (path, expected) in PINS.items():
        if not path.is_file():
            raise UnionError(f"pinned_file_missing:{label}:{path}")
        actual = sha_file(path)
        if actual != expected:
            raise UnionError(f"pinned_file_changed:{label}:{actual}:{expected}")
        result[label] = {"path": str(path), "sha256": actual}
    return result


def load_guards() -> tuple[dict[str, str], dict[str, str], set[str], dict[str, Any]]:
    split_doc = json.loads(GLOBAL_SPLIT.read_text(encoding="utf-8"))
    groups = split_doc.get("groups")
    if not isinstance(groups, list):
        raise UnionError("global_split_groups_missing")
    split_by_group: dict[str, str] = {}
    for item in groups:
        if not isinstance(item, dict) or not isinstance(item.get("group_id"), str):
            raise UnionError("global_split_group_record_invalid")
        split_by_group[item["group_id"]] = item.get("split")
    partition_doc = json.loads(PARTITION.read_text(encoding="utf-8"))
    partition_by_group = partition_doc.get("groups")
    if not isinstance(partition_by_group, dict):
        raise UnionError("cpt_partition_groups_missing")
    protected = set(json.loads(PROTECTED.read_text(encoding="utf-8")))
    if not all(isinstance(value, str) and len(value) == 40 for value in protected):
        raise UnionError("protected_parent_hash_registry_invalid")

    # The profile shard is metadata only.  Its cpt_validation document
    # identities are reserved before any candidate payload is opened.
    if not PROFILE_DOCS.is_file():
        raise UnionError("profile_validation_metadata_missing")
    reserved: set[str] = set()
    profile_rows = 0
    profile_validation_rows = 0
    for line in PROFILE_DOCS.open(encoding="utf-8"):
        if not line.strip():
            continue
        profile_rows += 1
        row = json.loads(line)
        if row.get("cpt_partition") != "cpt_validation":
            continue
        profile_validation_rows += 1
        for key in ("sha256", "source_sha256", "document_id"):
            value = row.get(key)
            if isinstance(value, str) and len(value) == 64:
                reserved.add(value)
    if profile_validation_rows != 294 or len(reserved) != 294:
        raise UnionError(f"profile_validation_metadata_count_mismatch:{profile_validation_rows}:{len(reserved)}")
    main_manifest = json.loads(MAIN_RUN_MANIFEST.read_text(encoding="utf-8"))
    main_manifest_pins = main_manifest.get("pins", {})
    if main_manifest_pins.get(str(PROTECTED)) != PINS["protected_parent_hashes"][1]:
        raise UnionError("main_upstream_protected_registry_binding_missing")
    if main_manifest.get("heldout", {}).get("non_TRAIN_groups_rejected_before_payload_read") is not True:
        raise UnionError("main_upstream_heldout_gate_missing")
    if main_manifest.get("materializer_sha256") != PINS["main_materializer"][1]:
        raise UnionError("main_materializer_source_binding_missing")
    alias_binding = json.loads(ALIAS_BINDING.read_text(encoding="utf-8"))
    alias_pins = alias_binding.get("pins", {})
    if alias_pins.get(str(PROTECTED)) != PINS["protected_parent_hashes"][1]:
        raise UnionError("alias_upstream_protected_registry_binding_missing")
    return split_by_group, {str(k): str(v) for k, v in partition_by_group.items()}, reserved, {
        "global_split_id": split_doc.get("split_id"),
        "global_split_group_count": len(split_by_group),
        "cpt_partition_split_id": partition_doc.get("split_id"),
        "cpt_partition_counts": partition_doc.get("counts"),
        "protected_parent_hash_count": len(protected),
        "profile_metadata_rows": profile_rows,
        "profile_validation_rows": profile_validation_rows,
        "reserved_cpt_validation_source_hashes": len(reserved),
        "protected_parent_hash_policy": "upstream main/alias materializers are hash-bound and reject raw sha1/git-blob identities before emission; union checks optional row-level identity fields when present and does not infer a match from a SHA-256 value",
        "upstream_protected_binding": {
            "main_run_manifest": str(MAIN_RUN_MANIFEST),
            "main_materializer_sha256": PINS["main_materializer"][1],
            "alias_binding": str(ALIAS_BINDING),
            "alias_materializer_sha256": PINS["alias_materializer"][1],
        },
    }


def iter_jsonl(path: Path, *, stream_stats: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects while hashing the exact bytes read once."""
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise UnionError(f"jsonl_decode_failed:{path}:{rows + 1}:{exc}") from exc
            if not isinstance(value, dict):
                raise UnionError(f"jsonl_row_not_object:{path}:{rows + 1}")
            rows += 1
            yield value
    stream_stats["bytes"] = path.stat().st_size
    stream_stats["sha256"] = digest.hexdigest()
    stream_stats["rows_read"] = rows


def check_artifact(path: Path, pin: Any, label: str) -> None:
    if not path.is_file() or not isinstance(pin, dict):
        raise UnionError(f"artifact_missing:{label}")
    expected_bytes = pin.get("bytes")
    expected_sha = pin.get("sha256")
    if path.stat().st_size != expected_bytes:
        raise UnionError(f"artifact_bytes_changed:{label}")
    # The caller hashes the exact bytes as it streams them.  Avoid a separate
    # full read here: the terminal run can contain more than a gigabyte of
    # committed group payloads.  Size is checked before opening and the
    # streamed digest is compared with this receipt pin before acceptance.


def main_terminal_gate(main_path: Path) -> dict[str, Any]:
    progress_path = main_path / "progress.json"
    if not progress_path.is_file():
        raise UnionError("main_progress_missing")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("groups_committed") != EXPECTED_GROUPS:
        raise UnionError("main_groups_not_terminal")
    if progress.get("groups_remaining") != 0:
        raise UnionError("main_groups_remaining")
    if progress.get("first_uncommitted_seeded_index") not in (None, MAIN_END):
        raise UnionError("main_frontier_not_terminal")
    if progress.get("status") not in {"complete", "all_groups_inventoried_repairs_pending"}:
        raise UnionError("main_status_not_terminal")
    return {
        "path": str(progress_path),
        "sha256": sha_file(progress_path),
        "progress": progress,
    }


def alias_capture_gate(alias_path: Path) -> dict[str, Any]:
    progress_path = alias_path / "progress.json"
    if not progress_path.is_file():
        raise UnionError("alias_progress_missing")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("captured_main_groups") != EXPECTED_GROUPS:
        raise UnionError("alias_rescan_did_not_capture_terminal_main")
    if progress.get("admitted") is True:
        raise UnionError("alias_producer_admitted_before_union_review")
    return {
        "path": str(progress_path),
        "sha256": sha_file(progress_path),
        "progress": progress,
    }


def validate_row(row: dict[str, Any], validator: Any, split_by_group: dict[str, str],
                 partition_by_group: dict[str, str], reserved: set[str], protected: set[str]) -> tuple[str, str]:
    """Validate a materialized row and return (document hash, source path)."""
    try:
        validator.validate_materialized_row(row, max_sequence_tokens=2048)
    except Exception as exc:
        raise UnionError(f"row_geometry_invalid:{exc}") from exc
    if row.get("schema") != 1 or row.get("cpt_partition") != "cpt_train":
        raise UnionError("row_not_cpt_train")
    group_id = row.get("group_id")
    if not isinstance(group_id, str) or split_by_group.get(group_id) != "train_group":
        raise UnionError("row_global_split_not_train")
    if partition_by_group.get(group_id) != "cpt_train":
        raise UnionError("row_cpt_partition_registry_not_train")
    source_sha = row.get("source_sha256")
    document_id = row.get("document_id")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise UnionError("row_source_sha256_missing")
    if document_id != source_sha:
        raise UnionError("row_document_identity_mismatch")
    if source_sha in reserved:
        raise UnionError("row_reserved_cpt_validation_identity")
    # Producer rows normally retain no SHA-1 fields.  When one is present,
    # check it against the protected parent registry; never infer a protected
    # match from an unrelated 64-character SHA-256 value.
    for key in ("sha1", "git_blob_sha1", "parent_sha1", "parent_hash", "source_sha1"):
        value = row.get(key)
        if value in protected:
            raise UnionError("row_known_nontrain_parent_hash")
    if row.get("row_id") != f"{source_sha}:{row.get('chunk_index')}":
        raise UnionError("row_id_identity_mismatch")
    if not isinstance(row.get("document_token_count"), int) or row["document_token_count"] <= 0:
        raise UnionError("row_document_token_count_missing")
    ids = row.get("input_ids")
    if any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in ids):
        raise UnionError("row_token_out_of_range")
    if ids[0] != BOS_ID or ids[-1] != EOS_ID or ids.count(BOS_ID) != 1 or ids.count(EOS_ID) != 1:
        raise UnionError("row_native_bos_eos_protocol_invalid")
    return source_sha, str(row.get("source_path"))


@dataclass
class Source:
    label: str
    path: Path
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    expected_rows: int | None = None
    extra: dict[str, Any] | None = None


class UnionBuilder:
    def __init__(self, output: Path, mode: str, split_by_group: dict[str, str],
                 partition_by_group: dict[str, str], reserved: set[str], protected: set[str],
                 guard_info: dict[str, Any], validator: Any):
        if output.exists() and any(output.iterdir()):
            raise UnionError("output_must_be_fresh")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.mode = mode
        self.split_by_group = split_by_group
        self.partition_by_group = partition_by_group
        self.reserved = reserved
        self.protected = protected
        self.guard_info = guard_info
        self.validator = validator
        self.seen_docs: dict[str, str] = {}
        self.doc_meta: dict[str, tuple[str, str, str]] = {}
        self.doc_states: dict[str, list[tuple[int, int, int, int, bool]]] = {}
        self.seen_rows: set[str] = set()
        self.last_emitted_doc: str | None = None
        self.closed_emitted_docs: set[str] = set()
        self.source_stats: dict[str, Counter[str]] = {}
        self.source_records: list[dict[str, Any]] = []
        self.exclusions: Counter[str] = Counter()
        self.emitted_rows = 0
        self.emitted_payload_tokens = 0
        self.emitted_input_tokens = 0
        self.emitted_documents = 0
        self._cpt = (output / "cpt_train.jsonl.partial").open("x", encoding="utf-8")
        self._docs = (output / "document-provenance.jsonl.partial").open("x", encoding="utf-8")
        self._excluded = (output / "exclusions.jsonl.partial").open("x", encoding="utf-8")
        self._sources = (output / "source-manifest.jsonl.partial").open("x", encoding="utf-8")

    def exclusion(self, source: Source, row: dict[str, Any] | None, reason: str, **detail: Any) -> None:
        self.exclusions[reason] += 1
        value = {
            "source": source.label,
            "path": str(source.path),
            "reason": reason,
        }
        if row is not None:
            for key in ("row_id", "document_id", "source_sha256", "group_id", "package", "source_path", "chunk_index"):
                if key in row:
                    value[key] = row[key]
        value.update(detail)
        self._excluded.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

    def stream(self, source: Source, rows: Iterable[dict[str, Any]]) -> None:
        stats = self.source_stats.setdefault(source.label, Counter())
        for row in rows:
            stats["rows_seen"] += 1
            try:
                source_sha, source_path = validate_row(row, self.validator, self.split_by_group,
                                                       self.partition_by_group, self.reserved, self.protected)
            except UnionError as exc:
                self.exclusion(source, row, str(exc).split(":", 1)[0])
                stats["rows_rejected"] += 1
                continue
            row_id = row["row_id"]
            if row_id in self.seen_rows:
                self.exclusion(source, row, "duplicate_row_id")
                stats["rows_duplicate"] += 1
                continue
            origin = f"{source.label}:{source_path}"
            first_origin = self.seen_docs.get(source_sha)
            if first_origin is not None and first_origin != origin:
                self.exclusion(source, row, "duplicate_source_document", first_origin=first_origin)
                stats["rows_duplicate"] += 1
                continue
            metadata = (str(row.get("package")), str(row.get("group_id")), source_path)
            prior_meta = self.doc_meta.get(source_sha)
            if prior_meta is not None and prior_meta != metadata:
                self.exclusion(source, row, "source_document_metadata_conflict", first_metadata=prior_meta,
                               duplicate_metadata=metadata)
                stats["rows_rejected"] += 1
                continue
            if first_origin is None:
                if source_sha in self.closed_emitted_docs:
                    raise UnionError(f"source_document_rows_noncontiguous:{source_sha}")
                self.seen_docs[source_sha] = origin
                self.doc_meta[source_sha] = metadata
                self.doc_states[source_sha] = []
                self.emitted_documents += 1
                stats["documents_new"] += 1
                self._docs.write(json.dumps({
                    "document_id": source_sha,
                    "source_sha256": source_sha,
                    "package": row["package"],
                    "group_id": row["group_id"],
                    "source_path": source_path,
                    "origin": origin,
                    "document_token_count": row["document_token_count"],
                }, sort_keys=True, separators=(",", ":")) + "\n")
            elif self.last_emitted_doc != source_sha:
                raise UnionError(f"source_document_rows_noncontiguous:{source_sha}")
            state = self.doc_states[source_sha]
            ids = row["input_ids"]
            overlap = row["overlap_context_tokens"]
            start = row["source_token_start"]
            end = row["source_token_end"]
            state.append((row["chunk_index"], start, end, row["document_token_count"], row["is_document_end"]))
            self.seen_rows.add(row_id)
            if self.last_emitted_doc is not None and self.last_emitted_doc != source_sha:
                self.closed_emitted_docs.add(self.last_emitted_doc)
            self.last_emitted_doc = source_sha
            self._cpt.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.emitted_rows += 1
            self.emitted_input_tokens += len(ids)
            self.emitted_payload_tokens += len(ids) - 2 - overlap
            stats["rows_emitted"] += 1
            stats["input_tokens"] += len(ids)
            stats["payload_tokens"] += len(ids) - 2 - overlap

    def record_source(self, source: Source, stats: dict[str, Any]) -> None:
        record = {"label": source.label, "path": str(source.path), **stats}
        if source.expected_sha256 is not None:
            record["expected_sha256"] = source.expected_sha256
        if source.expected_bytes is not None:
            record["expected_bytes"] = source.expected_bytes
        if source.expected_rows is not None:
            record["expected_rows"] = source.expected_rows
        if source.extra:
            record.update(source.extra)
        self.source_records.append(record)
        self._sources.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def validate_documents(self) -> None:
        for source_sha, chunks in self.doc_states.items():
            ordered = sorted(chunks, key=lambda value: value[0])
            indexes = [value[0] for value in ordered]
            if indexes != list(range(len(indexes))):
                raise UnionError(f"document_chunk_indexes_invalid:{source_sha}")
            if not ordered or ordered[0][1] != 0:
                raise UnionError(f"document_source_does_not_start_at_zero:{source_sha}")
            for previous, current in zip(ordered, ordered[1:]):
                if current[1] != previous[2]:
                    raise UnionError(f"document_source_gap_or_overlap:{source_sha}")
                if current[3] != ordered[0][3]:
                    raise UnionError(f"document_token_count_changed:{source_sha}")
                if previous[4]:
                    raise UnionError(f"nonterminal_chunk_marked_terminal:{source_sha}")
            final = ordered[-1]
            if final[2] != final[3] or not final[4]:
                raise UnionError(f"document_target_or_source_truncated:{source_sha}")

    def close(self) -> dict[str, Any]:
        self.validate_documents()
        self._cpt.flush(); self._docs.flush(); self._excluded.flush(); self._sources.flush()
        for stream in (self._cpt, self._docs, self._excluded, self._sources):
            os.fsync(stream.fileno()); stream.close()
        names = {
            "cpt_train.jsonl": "cpt_train.jsonl.partial",
            "document-provenance.jsonl": "document-provenance.jsonl.partial",
            "exclusions.jsonl": "exclusions.jsonl.partial",
            "source-manifest.jsonl": "source-manifest.jsonl.partial",
        }
        artifacts: dict[str, Any] = {}
        for final_name, partial_name in names.items():
            partial = self.output / partial_name
            final = self.output / final_name
            partial.replace(final)
            artifacts[final_name] = {"bytes": final.stat().st_size, "sha256": sha_file(final)}
        return artifacts


def alias_sources(alias_path: Path) -> Iterator[Source]:
    groups = alias_path / "groups"
    if not groups.is_dir():
        raise UnionError("alias_groups_missing")
    for folder in sorted(groups.iterdir()):
        if not folder.is_dir():
            continue
        receipt_path = folder / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        pin = receipt.get("artifacts", {}).get("cpt_train.jsonl")
        payload = folder / "cpt_train.jsonl"
        check_artifact(payload, pin, f"alias:{folder.name}")
        yield Source(f"alias:{folder.name}", payload, pin.get("sha256"), pin.get("bytes"),
                     receipt.get("counts", {}).get("rows"), {"receipt": str(receipt_path), "receipt_sha256": sha_file(receipt_path),
                                                              "receipt_status": receipt.get("status")})


def main_sources(main_path: Path) -> Iterator[Source]:
    groups = main_path / "groups"
    if not groups.is_dir():
        raise UnionError("main_groups_missing")
    for index in range(MAIN_START, MAIN_END):
        matches = list(groups.glob(f"{index:06d}-*"))
        if len(matches) != 1 or not matches[0].is_dir():
            raise UnionError(f"main_group_directory_missing_or_ambiguous:{index}")
        folder = matches[0]
        receipt_path = folder / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("seeded_index") != index:
            raise UnionError(f"main_group_index_mismatch:{index}")
        if receipt.get("status") not in {"complete", "complete_with_repairs_pending"}:
            raise UnionError(f"main_group_not_committed:{index}")
        pin = receipt.get("artifacts", {}).get("cpt_train.jsonl")
        payload = folder / "cpt_train.jsonl"
        check_artifact(payload, pin, f"main:{folder.name}")
        yield Source(f"main:{folder.name}", payload, pin.get("sha256"), pin.get("bytes"),
                     receipt.get("counts", {}).get("rows"), {"seeded_index": index, "receipt": str(receipt_path),
                                          "receipt_sha256": sha_file(receipt_path), "group_id": receipt.get("group_id")})


def repair_sources() -> Iterator[Source]:
    svars_result = json.loads((SVARS.parent / "result.json").read_text(encoding="utf-8"))
    svars_pin = svars_result.get("artifacts", {}).get("cpt_train.jsonl", {})
    check_artifact(SVARS, svars_pin, "repair:svars")
    yield Source("repair:svars", SVARS, svars_pin.get("sha256"), svars_pin.get("bytes"),
                 svars_result.get("counts", {}).get("rows"), {"candidate_status": svars_result.get("admission"),
                 "receipt": str(SVARS.parent / "result.json"), "receipt_sha256": sha_file(SVARS.parent / "result.json")})
    tfp_manifest_path = TFP.parent / "manifest.json"
    tfp_manifest = json.loads(tfp_manifest_path.read_text(encoding="utf-8"))
    tfp_pin = tfp_manifest.get("artifacts", {}).get("cpt_train.jsonl", {})
    check_artifact(TFP, tfp_pin, "repair:tfprobability")
    yield Source("repair:tfprobability", TFP, tfp_pin.get("sha256"), tfp_pin.get("bytes"),
                 tfp_manifest.get("counts", {}).get("rows"), {"candidate_status": tfp_manifest.get("status"),
                 "receipt": str(tfp_manifest_path), "receipt_sha256": sha_file(tfp_manifest_path),
                 "upstream_tar_sha256": tfp_manifest.get("metadata", {}).get("upstream_tar_sha256")})


def base_sources() -> Iterator[Source]:
    yield Source("base:broader-shard-v1-2k", BASE_BROAD)
    yield Source("base:cpt-global-shard-v1", BASE_GLOBAL)


def pending_holds(alias_path: Path, main_path: Path) -> list[dict[str, Any]]:
    holds = [
        {"kind": "license_hold", "package": "Rblpapi", "rows": 19, "documents": 19,
         "payload_tokens": 16515, "status": "held", "reason": "GPL3_source_scope_but_DESCRIPTION_License_is_FOSS_no; bundled Bloomberg headers/binaries remain separate",
         "candidate_path": str(E / "CPT-three-repair-recovery-v1/rblpapi-provisional/cpt_train.jsonl")},
        {"kind": "empty_source", "package": "RivRetrieve", "rows": 0, "documents": 0,
         "payload_tokens": 0, "status": "excluded", "reason": "empty_R_file_degenerate",
         "source_path": "/mnt/h/sepalith/normalized/RivRetrieve/0.1.9/RivRetrieve/R/data.R",
         "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"kind": "repair_queue", "scope": "main", "status": "pending_recovery_review",
         "reason": "terminal main group receipts may retain recoverable repair_items; queue is never silently admitted",
         "main_progress_path": str(main_path / "progress.json")},
        {"kind": "repair_queue", "scope": "license_alias", "status": "pending_recovery_review",
         "reason": "alias group repair queues remain separate from candidate rows",
         "alias_progress_path": str(alias_path / "progress.json")},
        {"kind": "candidate_review", "scope": "svars", "status": "emitted_pending_root_global_dedup_and_admission",
         "reason": "supported repair candidate; final union deduplicates exact source identities"},
        {"kind": "candidate_review", "scope": "tfprobability", "status": "emitted_pending_root_global_dedup_and_admission",
         "reason": "upstream metadata-pinned recovery candidate; final union deduplicates exact source identities"},
    ]
    return holds


def write_holds(path: Path, holds: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in holds:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_rechunk_input_manifest(output: Path, *, status: str, emitted_rows: int,
                                 emitted_documents: int, payload_tokens: int,
                                 cpt_artifact: dict[str, Any]) -> Path:
    """Bind the one-stream union to the existing lossless 16K rechunker."""
    path = output / "input-manifest.json"
    value = {
        "schema": "sepalith.cpt.lossless-rechunk-input.v1",
        "status": status,
        "context_sizes": [16384],
        "inputs": [{
            "label": "final-cpt-train-union",
            "path": str(output / "cpt_train.jsonl"),
            "bytes": cpt_artifact["bytes"],
            "sha256": cpt_artifact["sha256"],
            "rows": emitted_rows,
            "documents": emitted_documents,
            "payload_tokens": payload_tokens,
        }],
        "expected_totals": {
            "rows": emitted_rows,
            "documents": emitted_documents,
            "payload_tokens": payload_tokens,
        },
        "raw_chunks": {
            "path": str(RAW_CHUNKS),
            "sha256": PINS["raw_chunks"][1],
            "bos": BOS_ID,
            "eos": EOS_ID,
            "source_chunk_size": 2048,
        },
        "tokenizer": {"path": str(TOKENIZER), "sha256": PINS["tokenizer"][1], "retokenized": False},
        "truncation": False,
        "training_admission": False,
    }
    write_json(path, value)
    return path


def run_preview(args: argparse.Namespace, pins: dict[str, Any], validator: Any,
                split_by_group: dict[str, str], partition_by_group: dict[str, str],
                reserved: set[str], protected: set[str], guard_info: dict[str, Any]) -> dict[str, Any]:
    builder = UnionBuilder(args.output, "repair_preview", split_by_group, partition_by_group,
                           reserved, protected, guard_info, validator)
    for source in alias_sources(args.alias):
        stats: dict[str, Any] = {}
        builder.stream(source, iter_jsonl(source.path, stream_stats=stats))
        if stats.get("sha256") != source.expected_sha256:
            raise UnionError(f"alias_stream_hash_changed:{source.label}")
        if source.expected_rows is not None and stats.get("rows_read") != source.expected_rows:
            raise UnionError(f"alias_stream_row_count_changed:{source.label}")
        builder.record_source(source, stats)
    for source in repair_sources():
        stats = {}
        builder.stream(source, iter_jsonl(source.path, stream_stats=stats))
        if stats.get("sha256") != source.expected_sha256:
            raise UnionError(f"repair_stream_hash_changed:{source.label}")
        builder.record_source(source, stats)
    artifacts = builder.close()
    input_manifest = write_rechunk_input_manifest(
        args.output, status="repair_preview_candidate_not_training_admission",
        emitted_rows=builder.emitted_rows, emitted_documents=builder.emitted_documents,
        payload_tokens=builder.emitted_payload_tokens, cpt_artifact=artifacts["cpt_train.jsonl"])
    holds = pending_holds(args.alias, args.main)
    write_holds(args.output / "pending-holds.jsonl", holds)
    manifest = {
        "schema": "sepalith.dat10.cpt_final_union_repair_preview.v1",
        "status": "repair_preview_candidate_pending_terminal_main_alias_global_dedup",
        "mode": "repair_preview",
        "source_scope": "frozen alias groups plus svars/tfprobability repair candidates; main/base payloads intentionally not read",
        "source_manifest": str(args.output / "source-manifest.jsonl"),
        "source_manifest_sha256": sha_file(args.output / "source-manifest.jsonl"),
        "artifacts": artifacts,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha_file(input_manifest),
        "pending_holds": {"path": str(args.output / "pending-holds.jsonl"), "sha256": sha_file(args.output / "pending-holds.jsonl"), "records": len(holds)},
        "counts": {"documents": builder.emitted_documents, "rows": builder.emitted_rows,
                   "input_tokens": builder.emitted_input_tokens, "payload_tokens": builder.emitted_payload_tokens,
                   "source_records": len(builder.source_records), "exclusions": dict(builder.exclusions)},
        "source_counts": builder.source_records,
        "guard_info": guard_info,
        "pins": pins,
        "truncation": False,
        "target_text_written": False,
        "heldout_content_read": False,
        "training_admission": False,
        "final_union_ready": False,
        "next": "after main terminal and alias captured_main_groups=8092, run this script with --mode final and a fresh E output",
    }
    write_json(args.output / "manifest.json", manifest)
    return manifest


def run_final(args: argparse.Namespace, pins: dict[str, Any], validator: Any,
              split_by_group: dict[str, str], partition_by_group: dict[str, str],
              reserved: set[str], protected: set[str], guard_info: dict[str, Any]) -> dict[str, Any]:
    # Both gates happen before any main or alias payload is opened.
    main_gate = main_terminal_gate(args.main)
    alias_gate = alias_capture_gate(args.alias)
    builder = UnionBuilder(args.output, "final", split_by_group, partition_by_group,
                           reserved, protected, guard_info, validator)
    for source in base_sources():
        stats: dict[str, Any] = {}
        builder.stream(source, iter_jsonl(source.path, stream_stats=stats))
        builder.record_source(source, stats)
    for source in main_sources(args.main):
        stats = {}
        builder.stream(source, iter_jsonl(source.path, stream_stats=stats))
        if stats.get("sha256") != source.expected_sha256:
            raise UnionError(f"main_stream_hash_changed:{source.label}")
        if source.expected_rows is not None and stats.get("rows_read") != source.expected_rows:
            raise UnionError(f"main_stream_row_count_changed:{source.label}")
        builder.record_source(source, stats)
    for source in alias_sources(args.alias):
        stats = {}
        builder.stream(source, iter_jsonl(source.path, stream_stats=stats))
        if stats.get("sha256") != source.expected_sha256:
            raise UnionError(f"alias_stream_hash_changed:{source.label}")
        if source.expected_rows is not None and stats.get("rows_read") != source.expected_rows:
            raise UnionError(f"alias_stream_row_count_changed:{source.label}")
        builder.record_source(source, stats)
    for source in repair_sources():
        stats = {}
        builder.stream(source, iter_jsonl(source.path, stream_stats=stats))
        if stats.get("sha256") != source.expected_sha256:
            raise UnionError(f"repair_stream_hash_changed:{source.label}")
        builder.record_source(source, stats)
    artifacts = builder.close()
    input_manifest = write_rechunk_input_manifest(
        args.output, status="final_candidate_pending_root_admission",
        emitted_rows=builder.emitted_rows, emitted_documents=builder.emitted_documents,
        payload_tokens=builder.emitted_payload_tokens, cpt_artifact=artifacts["cpt_train.jsonl"])
    holds = pending_holds(args.alias, args.main)
    write_holds(args.output / "pending-holds.jsonl", holds)
    manifest = {
        "schema": "sepalith.dat10.cpt_final_union_candidate.v1",
        "status": "complete_candidate_pending_root_global_dedup_review_and_training_admission",
        "mode": "final",
        "source_scope": "base CPT TRAIN shards, terminal all-eligible main suffix, terminal-captured license aliases, supported svars/tfprobability repairs",
        "main_terminal_gate": main_gate,
        "alias_terminal_capture_gate": alias_gate,
        "source_manifest": str(args.output / "source-manifest.jsonl"),
        "source_manifest_sha256": sha_file(args.output / "source-manifest.jsonl"),
        "artifacts": artifacts,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha_file(input_manifest),
        "pending_holds": {"path": str(args.output / "pending-holds.jsonl"), "sha256": sha_file(args.output / "pending-holds.jsonl"), "records": len(holds)},
        "counts": {"documents": builder.emitted_documents, "rows": builder.emitted_rows,
                   "input_tokens": builder.emitted_input_tokens, "payload_tokens": builder.emitted_payload_tokens,
                   "source_records": len(builder.source_records), "exclusions": dict(builder.exclusions)},
        "source_counts": builder.source_records,
        "guard_info": guard_info,
        "pins": pins,
        "truncation": False,
        "retokenized": False,
        "heldout_content_read": False,
        "training_admission": False,
        "lossless_16k_conversion": "separate reviewed command; not executed by this builder",
        "accounting": {
            "previous_prefix": "775 fully processed package-order units yielded 750 prior global-shard groups; their rows are represented by the two base sources and are not counted as new suffix groups",
            "current_suffix": {"seeded_index_start_inclusive": MAIN_START, "seeded_index_end_exclusive": MAIN_END, "groups": EXPECTED_GROUPS},
        },
    }
    write_json(args.output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main", type=Path, default=MAIN)
    parser.add_argument("--alias", type=Path, default=ALIAS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pins = verify_pins()
        split_by_group, partition_by_group, reserved, guard_info = load_guards()
        protected = set(json.loads(PROTECTED.read_text(encoding="utf-8")))
        validator = load_validator()
        if args.mode == "preview":
            result = run_preview(args, pins, validator, split_by_group, partition_by_group,
                                 reserved, protected, guard_info)
        else:
            result = run_final(args, pins, validator, split_by_group, partition_by_group,
                               reserved, protected, guard_info)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        # A small failure receipt makes terminal-gate failures observable while
        # preserving any partial output for inspection.  It never claims data
        # was admitted.
        if args.output.exists():
            failure = {"schema": "sepalith.dat10.cpt_final_union_failure.v1", "status": "failed",
                       "mode": args.mode, "error": f"{type(exc).__name__}:{exc}",
                       "training_admission": False, "output": str(args.output)}
            try:
                write_json(args.output / "failure.json", failure)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
