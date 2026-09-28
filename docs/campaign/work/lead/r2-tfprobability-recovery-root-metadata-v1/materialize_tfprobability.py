#!/usr/bin/env python3
"""Prepare a source-pinned tfprobability CPT recovery candidate.

The normalized package is missing DESCRIPTION.  Root supplied a CRAN
metadata supplement and a tarball whose 21 R files were independently shown
to be byte-identical to the normalized tree.  Preflight mode reads only the
small metadata/evidence files.  Materialization is explicit and then reads
the 21 TRAIN source files, retaining complete source documents and recording
every identity check.  It never reads DEV/final payloads, imports a model, or
claims training admission.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKAGE = "tfprobability"
VERSION = "0.15.2"
GROUP_ID = "g-1037a9f3b52fac791d3d"
SEEDED_INDEX = 6866
NORMALIZED_ROOT = Path("/mnt/h/sepalith/normalized/tfprobability/0.15.2/tfprobability")
METADATA_ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/tfprobability-metadata-root-v1")
GLOBAL_SPLIT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
PROTECTED = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json"
RAW_CHUNKS = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py"
RAW_SHA = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")
TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
METADATA_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-tfprobability-upstream-metadata-root.json"
METADATA_RECEIPT_SHA = "9a283d7cc05914ebaded62222b8f46eaa263e4bf983452c2649fbfa8b52fd094"
TAR_SHA = "9268e148396a4231aa11bee32ddd755e879abe243b84c3064e010db1d21bd46c"
DESCRIPTION_SHA = "f8a3ba18f403feb221352417c643ab22bf2c4e65846bed30e974a37778a45163"
MD5_SHA = "41d937583fd88e3191711cdb3f855521966c725e9359d354aa067bd037826073"
CHUNK_SIZE = 2048
CONTEXT_SIZE = 16384
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module_import_failed:" + str(path))
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


def assert_pin(path: Path, expected: str) -> None:
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"sha256_mismatch:{path}:{actual}:{expected}")


def load_and_validate_guards() -> dict[str, Any]:
    assert_pin(METADATA_RECEIPT, METADATA_RECEIPT_SHA)
    evidence = json.loads(METADATA_RECEIPT.read_text(encoding="utf-8"))
    if evidence.get("status") != "exact_upstream_metadata_recovered_source_match":
        raise ValueError("metadata_evidence_status_mismatch")
    if evidence.get("upstream_tar_sha256") != TAR_SHA or evidence.get("description_sha256") != DESCRIPTION_SHA:
        raise ValueError("metadata_evidence_hash_mismatch")
    if evidence.get("description_md5_matches_existing_local_manifest") is not True:
        raise ValueError("metadata_md5_match_not_proven")
    if evidence.get("description_fields") != [
        "Package: tfprobability", "Version: 0.15.2", "License: Apache License (>= 2.0)", "Repository: CRAN"
    ]:
        raise ValueError("metadata_fields_mismatch")
    description = METADATA_ROOT / "DESCRIPTION"
    md5 = METADATA_ROOT / "MD5"
    tar = METADATA_ROOT / "upstream.tar.gz"
    for path, expected in ((description, DESCRIPTION_SHA), (md5, MD5_SHA), (tar, TAR_SHA)):
        assert_pin(path, expected)
    dcf = fields(description.read_text(encoding="utf-8"))
    if dcf.get("Package") != PACKAGE or dcf.get("Version") != VERSION or dcf.get("License") != "Apache License (>= 2.0)":
        raise ValueError("supplement_description_identity_mismatch")
    split = json.loads(GLOBAL_SPLIT.read_text(encoding="utf-8"))
    matches = []
    for group in split.get("groups", []):
        forms = [str(value).lower() for value in group.get("identity_forms", [])]
        if "pkg:" + PACKAGE in forms:
            matches.append(group)
    if len(matches) != 1 or matches[0].get("group_id") != GROUP_ID or matches[0].get("split") != "train_group":
        raise ValueError("global_train_identity_mismatch")
    partitions = json.loads(PARTITION.read_text(encoding="utf-8")).get("groups", {})
    if partitions.get(GROUP_ID) != "cpt_train":
        raise ValueError("cpt_partition_identity_mismatch")
    assert_pin(RAW_CHUNKS, RAW_SHA)
    assert_pin(TOKENIZER, TOKENIZER_SHA)
    return {
        "metadata_receipt": {"path": str(METADATA_RECEIPT), "sha256": METADATA_RECEIPT_SHA},
        "metadata": {"description_path": str(description), "description_sha256": DESCRIPTION_SHA,
                      "md5_path": str(md5), "md5_sha256": MD5_SHA, "license": dcf["License"],
                      "upstream_tar_path": str(tar), "upstream_tar_sha256": TAR_SHA},
        "source": {"root": str(NORMALIZED_ROOT), "group_id": GROUP_ID, "package": PACKAGE, "version": VERSION,
                   "seeded_index": SEEDED_INDEX, "split": "train_group", "cpt_partition": "cpt_train"},
        "guards": {
            "global_split": {"path": str(GLOBAL_SPLIT), "sha256": sha_file(GLOBAL_SPLIT)},
            "cpt_partition": {"path": str(PARTITION), "sha256": sha_file(PARTITION)},
            "protected_parent_hashes": {"path": str(PROTECTED), "sha256": sha_file(PROTECTED)},
            "raw_chunks": {"path": str(RAW_CHUNKS), "sha256": RAW_SHA, "chunk_size": CHUNK_SIZE},
            "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA},
        },
    }


def source_paths_and_tar_names(tar_path: Path) -> list[tuple[Path, str]]:
    if not NORMALIZED_ROOT.is_dir() or NORMALIZED_ROOT.is_symlink():
        raise ValueError("normalized_package_root_invalid")
    rdir = NORMALIZED_ROOT / "R"
    paths = sorted(path for path in rdir.rglob("*") if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".r")
    if len(paths) != 21:
        raise ValueError(f"regular_R_file_count_mismatch:{len(paths)}")
    # CRAN's source archive uses the package name as its top-level directory;
    # the version is carried by DESCRIPTION and the pinned tar identity.
    return [(path, f"{PACKAGE}/R/{path.relative_to(rdir).as_posix()}") for path in paths]


def read_seen_snapshot(path: Path) -> tuple[set[str], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    values = {line for line in raw.decode("utf-8").splitlines() if line}
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in values):
        raise ValueError("seen_snapshot_contains_non_sha256")
    if len(values) != len([line for line in raw.decode("utf-8").splitlines() if line]):
        raise ValueError("seen_snapshot_has_duplicate_hashes")
    return values, digest


def materialize(args: argparse.Namespace, guards: dict[str, Any]) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"fresh output required: {args.output}")
    if args.seen_snapshot is None:
        raise ValueError("materialize_requires_explicit_main_seen_snapshot")
    seen, seen_sha = read_seen_snapshot(args.seen_snapshot)
    protected = set(json.loads(PROTECTED.read_text(encoding="utf-8")))
    raw_chunks = load_module("tfprobability_recovery_raw_chunks", RAW_CHUNKS)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    tokenizer.encode_special_tokens = True
    validator = load_module("tfprobability_recovery_validator", args.validator)
    source_pairs = source_paths_and_tar_names(Path(guards["metadata"]["upstream_tar_path"]))
    stage = args.output.with_name(args.output.name + ".tmp")
    if stage.exists():
        raise FileExistsError(f"stale staging path exists: {stage}")
    stage.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    local_hashes: set[str] = set()
    counts: Counter[str] = Counter()
    tar_path = Path(guards["metadata"]["upstream_tar_path"])
    try:
        with tarfile.open(tar_path, mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            expected_names = {tar_name for _, tar_name in source_pairs}
            if not expected_names <= members.keys():
                raise ValueError("upstream_tar_missing_R_member")
            for source_path, tar_name in source_pairs:
                before = source_path.stat()
                if source_path.is_symlink() or not stat.S_ISREG(before.st_mode):
                    repairs.append({"path": str(source_path), "reason": "source_not_regular"})
                    continue
                raw = source_path.read_bytes()
                after = source_path.stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError(f"source_changed_during_read:{source_path}")
                fp = fingerprints(raw)
                tar_member = archive.extractfile(members[tar_name])
                if tar_member is None:
                    raise ValueError(f"upstream_R_member_unreadable:{tar_name}")
                upstream_raw = tar_member.read()
                if raw != upstream_raw:
                    raise ValueError(f"normalized_upstream_source_mismatch:{source_path}")
                record: dict[str, Any] = {
                    "schema": "sepalith.dat10.tfprobability_source_document.v1",
                    "package": PACKAGE, "version": VERSION, "group_id": GROUP_ID,
                    "seeded_index": SEEDED_INDEX, "split": "train_group", "cpt_partition": "cpt_train",
                    "path": str(source_path), "relative_path": str(source_path.relative_to(NORMALIZED_ROOT)),
                    "upstream_tar_member": tar_name, "bytes": len(raw), **fp,
                    "metadata_description_sha256": DESCRIPTION_SHA,
                    "license": guards["metadata"]["license"], "upstream_tar_sha256": TAR_SHA,
                    "source_category": "regular_R_under_package_R",
                    "source_match": "byte_identical_to_pinned_upstream_tar",
                }
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    record["status"] = "repair_pending_non_utf8"; inventory.append(record)
                    repairs.append({**record, "reason": "non_utf8"}); continue
                if "\0" in text:
                    record["status"] = "repair_pending_NUL"; inventory.append(record)
                    repairs.append({**record, "reason": "NUL_in_source"}); continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
                    record["status"] = "repair_pending_tokenizer_roundtrip"; record["source_code_tokens"] = len(ids); inventory.append(record)
                    repairs.append({**record, "reason": "tokenizer_empty_or_roundtrip_failure"}); continue
                record.update({"status": "candidate_metadata_verified_pending_dedup", "source_code_tokens": len(ids),
                               "chunks_2048": sum(1 for _ in raw_chunks.chunks(ids, CHUNK_SIZE))})
                inventory.append(record)
                reason = None
                if set(fp.values()) & protected:
                    reason = "known_nontrain_parent_hash_match"
                elif fp["sha256"] in seen:
                    reason = "exact_duplicate_of_main_or_reserved_document"
                elif fp["sha256"] in local_hashes:
                    reason = "exact_duplicate_within_recovery"
                if reason:
                    exclusions.append({**record, "reason": reason, "dedup_snapshot_sha256": seen_sha})
                    counts["dedup_excluded"] += 1
                    continue
                produced: list[str] = []
                for chunk_index, chunk in enumerate(raw_chunks.chunks(ids, CHUNK_SIZE)):
                    row = {
                        "schema": 1, "row_id": fp["sha256"] + ":" + str(chunk_index), "document_id": fp["sha256"],
                        "package": PACKAGE, "group_id": GROUP_ID, "cpt_partition": "cpt_train",
                        "source_path": str(source_path), "source_sha256": fp["sha256"], "chunk_index": chunk_index,
                        **chunk,
                    }
                    rows.append(row); produced.append(row["row_id"])
                    counts["rows"] += 1; counts["input_tokens"] += len(row["input_ids"])
                    counts["loss_tokens"] += row["supervised_tokens"]
                document = {
                    **record, "document_id": fp["sha256"], "source_code_tokens": len(ids), "chunks": len(produced),
                    "row_ids": produced, "dedup_snapshot_sha256": seen_sha,
                    "provisional_pending_terminal_global_dedup": True, "training_admission": False,
                }
                documents.append(document); local_hashes.add(fp["sha256"])
                counts["documents"] += 1; counts["payload_tokens"] += len(ids); counts["raw_bytes"] += len(raw)
        # Validate the complete 2K stream before publication.
        checked = validator.validate_materialized_rows(rows, max_sequence_tokens=CHUNK_SIZE, require_complete_documents=True)
        if checked["rows"] != counts["rows"] or checked["documents"] != counts["documents"] or checked["payload_tokens"] != counts["payload_tokens"]:
            raise ValueError("frozen_validator_totals_mismatch")
        if checked["loss_tokens"] != counts["loss_tokens"]:
            raise ValueError("frozen_validator_loss_totals_mismatch")
        write_jsonl(stage / "cpt_train.jsonl", rows)
        write_jsonl(stage / "documents.jsonl", documents)
        write_jsonl(stage / "source-inventory.jsonl", inventory)
        write_jsonl(stage / "exclusions.jsonl", exclusions)
        write_jsonl(stage / "repair-queue.jsonl", repairs)
        input_manifest = {
            "schema": "sepalith.cpt.lossless-rechunk-input.v1", "context_sizes": [CONTEXT_SIZE],
            "status": "candidate_pending_terminal_global_dedup_and_root_admission",
            "expected_totals": {"rows": counts["rows"], "documents": counts["documents"], "payload_tokens": counts["payload_tokens"]},
            "inputs": [{"seeded_index": SEEDED_INDEX, "group_id": GROUP_ID, "package": PACKAGE,
                        "path": str(args.output / "cpt_train.jsonl"), "source_path": str(args.output / "cpt_train.jsonl"),
                        "bytes": None, "sha256": None, "rows": counts["rows"], "documents": counts["documents"],
                        "payload_tokens": counts["payload_tokens"], "cpt_partition": "cpt_train"}],
            "raw_chunks": {"path": str(RAW_CHUNKS), "sha256": RAW_SHA, "bos": 0, "eos": 1, "source_chunk_size": CHUNK_SIZE},
            "original_tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA, "retokenized": False},
            "source_group": {"package": PACKAGE, "version": VERSION, "group_id": GROUP_ID, "seeded_index": SEEDED_INDEX,
                             "global_split": "train_group", "cpt_partition": "cpt_train"},
            "training_admission": False, "truncation": False,
        }
        input_manifest["inputs"][0]["bytes"] = (stage / "cpt_train.jsonl").stat().st_size
        input_manifest["inputs"][0]["sha256"] = sha_file(stage / "cpt_train.jsonl")
        write_json(stage / "input-manifest.json", input_manifest)
        artifacts = {}
        for path in sorted(stage.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
        manifest = {
            "schema": "sepalith.dat10.tfprobability_recovery_manifest.v1",
            "status": "complete_2k_candidate_pending_terminal_global_dedup",
            "package": PACKAGE, "version": VERSION, "group_id": GROUP_ID, "seeded_index": SEEDED_INDEX,
            "global_split": "train_group", "cpt_partition": "cpt_train",
            "metadata": guards["metadata"], "guards": guards["guards"],
            "main_seen_snapshot": {"path": str(args.seen_snapshot), "sha256": seen_sha, "hashes": len(seen)},
            "counts": dict(counts), "source_files_examined": len(inventory),
            "source_bytes": sum(row["bytes"] for row in inventory),
            "source_code_tokens": sum(row.get("source_code_tokens", 0) for row in inventory),
            "source_chunks_2048": sum(row.get("chunks_2048", 0) for row in inventory),
            "source_matches_upstream": sum(row.get("source_match") == "byte_identical_to_pinned_upstream_tar" for row in inventory),
            "validator": {"path": str(args.validator), "sha256": sha_file(args.validator), "result": {k: checked[k] for k in ("rows", "documents", "payload_tokens", "input_tokens", "loss_tokens")}},
            "input_manifest": str(args.output / "input-manifest.json"),
            "artifacts": artifacts,
            "truncation": False, "all_source_payloads_conserved_for_emitted_documents": True,
            "terminal_global_dedup_pending": True, "training_admission": False,
            "source_text_written": False, "target_text_written": False,
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(args.output)
        parent_fd = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def preflight(args: argparse.Namespace, guards: dict[str, Any]) -> dict[str, Any]:
    source_pairs = source_paths_and_tar_names(Path(guards["metadata"]["upstream_tar_path"]))
    tar_path = Path(guards["metadata"]["upstream_tar_path"])
    with tarfile.open(tar_path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
    missing = [tar_name for _, tar_name in source_pairs if tar_name not in members]
    if missing:
        raise ValueError("upstream_tar_missing_members:" + ",".join(missing))
    return {
        "schema": "sepalith.dat10.tfprobability_recovery_preflight.v1",
        "status": "PASS_source_payload_read_deferred",
        "source_file_count_expected": len(source_pairs), "upstream_members_present": len(source_pairs),
        "source_payloads_read": False, "source_text_written": False, "target_text_written": False,
        "training_admission": False, "guards": guards,
        "next": "Run --materialize with the terminal/current main seen snapshot after root I/O-clear; then run the pinned lossless rechunker using input-manifest.json.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--seen-snapshot", type=Path)
    parser.add_argument("--validator", type=Path, default=PLAN / "docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/campaign_cpt_data.py")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    if args.metadata_root != METADATA_ROOT:
        raise ValueError("metadata_root_must_match_pinned_evidence")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    guards = load_and_validate_guards()
    if args.materialize:
        result = materialize(args, guards)
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output_must_be_fresh")
        args.output.mkdir(parents=True, exist_ok=True)
        result = preflight(args, guards)
        write_json(args.output / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
