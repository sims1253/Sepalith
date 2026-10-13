#!/usr/bin/env python3
"""Inventory and, after metadata repair, materialize tfprobability CPT data.

The current normalized package is missing DESCRIPTION.  The default mode
therefore writes a source-only inventory and an explicit repair ledger.  The
optional ``--materialize`` mode is fail-closed: it requires restored metadata,
the frozen TRAIN/CPT guards, a protected-hash set, and a current main-corpus
seen snapshot before writing token rows.  Neither mode truncates source files
or targets (this package contains source files only).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
from typing import Any

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
NORMALIZED = Path("/mnt/h/sepalith/normalized")
SPLIT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
PROTECTED = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json"
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
RAW_CHUNKS = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py"
RAW_CHUNKS_SHA256 = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
PACKAGE = "tfprobability"
GROUP_ID = "g-1037a9f3b52fac791d3d"
SEEDED_INDEX = 6866
VERSION = "0.15.2"
CHUNK_SIZE = 2048


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(raw: bytes) -> dict[str, str]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sha1": hashlib.sha1(raw).hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module_missing:" + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    key: str | None = None
    for line in text.splitlines():
        if line[:1].isspace() and key:
            result[key] += " " + line.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            result[key] = value.strip()
    return result


def load_guards() -> dict[str, Any]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    matched = []
    for group in split["groups"]:
        forms = group.get("identity_forms", [])
        if "pkg:" + PACKAGE in forms or "pkg:" + PACKAGE.lower() in [str(x).lower() for x in forms]:
            matched.append(group)
    if len(matched) != 1:
        raise ValueError("global_registry_package_identity_mismatch")
    group = matched[0]
    if group.get("group_id") != GROUP_ID or group.get("split") != "train_group":
        raise ValueError("global_registry_train_guard_mismatch")
    partitions = json.loads(PARTITION.read_text(encoding="utf-8"))["groups"]
    if partitions.get(GROUP_ID) != "cpt_train":
        raise ValueError("cpt_partition_guard_mismatch")
    if sha_file(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("tokenizer_identity_mismatch")
    if sha_file(RAW_CHUNKS) != RAW_CHUNKS_SHA256:
        raise ValueError("raw_chunk_identity_mismatch")
    return {
        "global_split": {"path": str(SPLIT), "sha256": sha_file(SPLIT), "group_id": GROUP_ID, "split": "train_group"},
        "cpt_partition": {"path": str(PARTITION), "sha256": sha_file(PARTITION), "partition": "cpt_train"},
        "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA256},
        "raw_chunks": {"path": str(RAW_CHUNKS), "sha256": RAW_CHUNKS_SHA256, "chunk_size": CHUNK_SIZE},
        "protected_parent_hashes": {"path": str(PROTECTED), "sha256": sha_file(PROTECTED)},
    }


def package_root() -> Path:
    root = NORMALIZED / PACKAGE
    versions = sorted(item for item in root.iterdir() if item.is_dir() and not item.is_symlink())
    if [item.name for item in versions] != [VERSION]:
        raise ValueError("normalized_version_scope_mismatch")
    source = versions[0] / PACKAGE
    if source.is_symlink() or not source.is_dir():
        raise ValueError("normalized_package_root_invalid")
    return source


def metadata_evidence(source: Path) -> dict[str, Any]:
    description = source / "DESCRIPTION"
    license_files = [source / name for name in ("LICENSE", "LICENCE", "COPYING", "NOTICE")]
    existing_license_files = [path for path in license_files if path.is_file() and not path.is_symlink()]
    result: dict[str, Any] = {
        "description_path": str(description),
        "description_exists": description.is_file() and not description.is_symlink(),
        "description_sha256": sha_file(description) if description.is_file() else None,
        "license_file_paths": [str(path) for path in existing_license_files],
        "license_file_sha256": {str(path): sha_file(path) for path in existing_license_files},
        "package": None,
        "license": None,
        "status": "missing_description",
    }
    if not result["description_exists"]:
        md5 = source / "MD5"
        result["md5_path"] = str(md5)
        result["md5_sha256"] = sha_file(md5) if md5.is_file() else None
        result["md5_mentions_description"] = bool(md5.is_file() and b"*DESCRIPTION" in md5.read_bytes())
        result["license_status"] = "unresolved_no_description_or_license_metadata"
        return result
    raw = description.read_bytes()
    try:
        dcf = fields(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        result["status"] = "description_non_utf8"
        result["license_status"] = "unresolved_description_decode"
        result["error"] = type(exc).__name__
        return result
    result["package"] = dcf.get("Package")
    result["license"] = dcf.get("License")
    result["description_fields_sha256"] = sha_file(description)
    result["status"] = "available"
    result["license_status"] = "available"
    return result


def source_inventory(source: Path, tokenizer) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    rdir = source / "R"
    if not rdir.is_dir() or rdir.is_symlink():
        raise ValueError("regular_R_directory_missing")
    inventory: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for path in sorted(rdir.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            if path.suffix.lower() == ".r":
                repairs.append({"path": str(path), "reason": "symlink_R_source_unsupported"})
            continue
        if not path.is_file():
            if path.suffix.lower() == ".r":
                repairs.append({"path": str(path), "reason": "nonregular_R_source_unsupported"})
            continue
        if path.suffix.lower() != ".r":
            counts["regular_non_R"] += 1
            continue
        raw = path.read_bytes()
        digest = fingerprints(raw)
        record: dict[str, Any] = {
            "package": PACKAGE,
            "version": VERSION,
            "group_id": GROUP_ID,
            "seeded_index": SEEDED_INDEX,
            "split": "train_group",
            "cpt_partition": "cpt_train",
            "path": str(path),
            "bytes": len(raw),
            **digest,
            "raw_code_hashed": True,
        }
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            record.update({"status": "repair_pending_non_utf8", "token_count": None, "chunk_count": None})
            repairs.append({"path": str(path), "reason": "non_UTF8_R_source_requires_repair", "sha256": digest["sha256"]})
        else:
            if "\0" in text:
                record.update({"status": "repair_pending_NUL", "token_count": None, "chunk_count": None})
                repairs.append({"path": str(path), "reason": "NUL_R_source_requires_repair", "sha256": digest["sha256"]})
            else:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
                    record.update({"status": "repair_pending_tokenizer_roundtrip", "token_count": len(ids), "chunk_count": None})
                    repairs.append({"path": str(path), "reason": "tokenizer_empty_or_roundtrip_failure", "sha256": digest["sha256"]})
                else:
                    record.update({"status": "tokenizable", "token_count": len(ids), "chunk_count": sum(1 for _ in load_module("tfp_chunks", RAW_CHUNKS).chunks(ids, CHUNK_SIZE))})
        inventory.append(record)
        counts["regular_R"] += 1
    inventory.sort(key=lambda row: row["path"])
    return inventory, repairs, counts


def load_seen(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = set(path.read_text(encoding="utf-8").splitlines())
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in values):
        raise ValueError("seen_snapshot_invalid")
    return values


def materialize(source: Path, metadata: dict[str, Any], inventory: list[dict[str, Any]], output: Path,
                tokenizer, seen_snapshot: Path) -> dict[str, Any]:
    if metadata.get("status") != "available" or metadata.get("package") != PACKAGE:
        raise ValueError("materialize_requires_valid_DESCRIPTION")
    raw_chunks = load_module("tfp_raw_chunks_materialize", RAW_CHUNKS)
    if not raw_chunks.allowed_license(str(metadata.get("license", ""))):
        raise ValueError("license_not_in_frozen_recognized_families")
    seen = load_seen(seen_snapshot)
    protected = set(json.loads(PROTECTED.read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    local_seen: set[str] = set()
    counts: Counter[str] = Counter()
    for record in inventory:
        if record.get("status") not in {"tokenizable", "candidate_metadata_verified_pending_dedup"}:
            exclusions.append({**record, "reason": record.get("status", "repair_pending")})
            counts["repair_pending"] += 1
            continue
        path = Path(record["path"])
        raw = path.read_bytes()
        digest = fingerprints(raw)
        reason = None
        if set(digest.values()) & protected:
            reason = "known_nontrain_parent_hash_match"
        elif digest["sha256"] in seen:
            reason = "exact_duplicate_of_main_or_reserved_document"
        elif digest["sha256"] in local_seen:
            reason = "exact_duplicate_within_recovery"
        if reason:
            exclusions.append({**record, "observed_sha256": digest["sha256"], "reason": reason})
            counts["dedup_excluded"] += 1
            continue
        text = raw.decode("utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        produced = []
        for chunk_index, chunk in enumerate(raw_chunks.chunks(ids, CHUNK_SIZE)):
            row = {
                "schema": 1,
                "row_id": digest["sha256"] + ":" + str(chunk_index),
                "document_id": digest["sha256"],
                "package": PACKAGE,
                "group_id": GROUP_ID,
                "cpt_partition": "cpt_train",
                "source_path": str(path),
                "source_sha256": digest["sha256"],
                "chunk_index": chunk_index,
                **chunk,
            }
            rows.append(row)
            produced.append(row["row_id"])
            counts["rows"] += 1
            counts["input_tokens"] += len(row["input_ids"])
            counts["loss_tokens"] += row["supervised_tokens"]
        documents.append({
            **record,
            **digest,
            "document_id": digest["sha256"],
            "source_code_tokens": len(ids),
            "chunks": len(produced),
            "metadata_description_sha256": metadata["description_sha256"],
            "license": metadata["license"],
        })
        local_seen.add(digest["sha256"])
        counts["documents"] += 1
        counts["payload_tokens"] += len(ids)
        counts["raw_bytes"] += len(raw)
    write_jsonl(output / "cpt_train.jsonl", rows)
    write_jsonl(output / "documents.jsonl", documents)
    write_jsonl(output / "exclusions.jsonl", exclusions)
    return {
        "status": "materialized_candidate_not_training_admission",
        "counts": dict(counts),
        "documents": len(documents),
        "rows": len(rows),
        "seen_snapshot_sha256": sha_file(seen_snapshot),
        "protected_hash_count": len(protected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--seen-snapshot", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output_must_be_fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    guards = load_guards()
    source = package_root()
    metadata = metadata_evidence(source)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    tokenizer.encode_special_tokens = True
    inventory, repairs, source_counts = source_inventory(source, tokenizer)
    if metadata["status"] != "available":
        repairs.insert(0, {
            "package": PACKAGE,
            "group_id": GROUP_ID,
            "reason": "description_read_or_decode_failure",
            "detail": metadata["status"],
            "description_path": metadata["description_path"],
            "license_status": metadata["license_status"],
        })
        for record in inventory:
            if record["status"] == "tokenizable":
                record["status"] = "repair_pending_missing_description"
    else:
        for record in inventory:
            if record["status"] == "tokenizable":
                record["status"] = "candidate_metadata_verified_pending_dedup"
    write_jsonl(args.output / "source-inventory.jsonl", inventory)
    write_jsonl(args.output / "repair-ledger.jsonl", repairs)
    materialization = None
    if args.materialize:
        if args.seen_snapshot is None:
            raise ValueError("materialize_requires_seen_snapshot")
        materialization = materialize(source, metadata, inventory, args.output, tokenizer, args.seen_snapshot)
    summary = {
        "schema": "sepalith.dat10.cpt_tfprobability_recovery_summary.v1",
        "status": "metadata_repair_pending" if metadata["status"] != "available" else (materialization or {}).get("status", "candidate_ready_pending_materialization"),
        "package": PACKAGE,
        "version": VERSION,
        "group_id": GROUP_ID,
        "seeded_index": SEEDED_INDEX,
        "global_split": "train_group",
        "cpt_partition": "cpt_train",
        "metadata": metadata,
        "source_counts": dict(source_counts),
        "source_files_examined": len(inventory),
        "source_bytes": sum(int(row["bytes"]) for row in inventory),
        "tokenizable_files": sum(row.get("token_count") is not None and row.get("chunk_count") is not None for row in inventory),
        "repair_records": len(repairs),
        "materialization": materialization,
        "truncation": False,
        "source_text_written": False,
        "target_text_written": False,
        "training_admission": False,
        "guards": guards,
        "artifacts": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            summary["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    write_json(args.output / "manifest.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
