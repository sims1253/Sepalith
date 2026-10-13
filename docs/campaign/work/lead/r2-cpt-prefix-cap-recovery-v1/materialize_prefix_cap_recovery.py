#!/usr/bin/env python3
"""Materialize the omitted pre-cap CPT source frontier.

This is a source-backed, CPU-only recovery worker.  It deliberately has no
file, package, group, or token budget: the old ``whole_document...cap`` is the
reason the input files were omitted.  It reads only the exact 1,999-record
frontier, validates TRAIN/CPT/licence identity before opening source payload,
then hashes and tokenizes each stable source file.  The terminal CPT payload is
never read; its small document-provenance index is used for exact source
deduplication.

The output is a candidate packet.  It is not a training-admission decision.
Each group is committed atomically, so a root-owned run can be interrupted and
resumed without losing already materialized groups.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
HERE = PLAN / "docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v1"
DEFAULT_FRONTIER = PLAN / "docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/global-cap-exclusion-frontier.jsonl"
DEFAULT_OUTPUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v1")
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
GLOBAL_SPLIT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
PROTECTED = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json"
PROFILE_DOCS = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v1/documents.jsonl"
TERMINAL_PROVENANCE = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1/document-provenance.jsonl")
TERMINAL_PROVENANCE_SHA256 = "a731974ee8581b9fa571aae9672f6b933743779c24f4253843bff268065c305f"
FRONTIER_SHA256 = "acf86ba87bf833b6bd69cf732f2cb4a7d5356bcbd9e38ba2676b385af2b37560"
TERMINAL_DOCUMENT_COUNT = 177190
FRONTIER_COUNT = 1999
CHUNK_SIZE = 2048
BOS_ID = 0
EOS_ID = 1
PAD_ID = 1
VOCAB_SIZE = 130560
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

RAW_CPT = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py"


class RecoveryError(RuntimeError):
    """A fail-closed recovery contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RecoveryError(f"pinned_file_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecoveryError(f"json_decode:{path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RecoveryError(f"json_row_not_object:{path}:{line_number}")
            rows.append(value)
    return rows


def description_fields(text: str) -> dict[str, str]:
    """Parse DESCRIPTION fields while retaining continuation lines."""
    fields: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        if line[:1].isspace() and current is not None:
            fields[current] += " " + line.strip()
        elif ":" in line:
            current, value = line.split(":", 1)
            fields[current] = value.strip()
        else:
            current = None
    return fields


_LICENSE_RE = re.compile(
    r"(^|[ |+,(])(?:A?GPL|LGPL|MIT|BSD|Apache|Artistic|MPL|CC0|Unlimited)(?:[- (]|$)",
    re.IGNORECASE,
)


def allowed_license(value: str) -> bool:
    return bool(_LICENSE_RE.search(value))


def source_fingerprints(raw: bytes) -> dict[str, str]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sha1": hashlib.sha1(raw).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest(),
    }


def token_stream_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def load_raw_contract() -> Any:
    """Load the reviewed chunk contract without importing a model or trainer."""
    spec = importlib.util.spec_from_file_location("raw_cpt_broader_contract", RAW_CPT)
    if spec is None or spec.loader is None:
        raise RecoveryError(f"raw_contract_import_failed:{RAW_CPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_guards(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    if len(frontier) != FRONTIER_COUNT:
        raise RecoveryError(f"frontier_count:{len(frontier)}")
    if sha256_file(DEFAULT_FRONTIER) != FRONTIER_SHA256:
        raise RecoveryError("frontier_identity_changed")

    split_doc = json.loads(GLOBAL_SPLIT.read_text(encoding="utf-8"))
    split_rows = split_doc.get("groups")
    if not isinstance(split_rows, list):
        raise RecoveryError("global_split_groups_missing")
    split_by_group: dict[str, str] = {}
    for row in split_rows:
        if not isinstance(row, dict) or not isinstance(row.get("group_id"), str):
            raise RecoveryError("global_split_row_invalid")
        split_by_group[row["group_id"]] = row.get("split")

    partition_doc = json.loads(PARTITION.read_text(encoding="utf-8"))
    partition_by_group = partition_doc.get("groups")
    if not isinstance(partition_by_group, dict):
        raise RecoveryError("cpt_partition_groups_missing")

    protected = set(json.loads(PROTECTED.read_text(encoding="utf-8")))
    if not protected or not all(isinstance(value, str) and len(value) == 40 for value in protected):
        raise RecoveryError("protected_parent_registry_invalid")

    terminal_paths: set[str] = set()
    terminal_source_sha: set[str] = set()
    terminal_documents = 0
    with TERMINAL_PROVENANCE.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            terminal_documents += 1
            path = row.get("source_path")
            source_sha = row.get("source_sha256")
            if not isinstance(path, str) or not path:
                raise RecoveryError(f"terminal_path_missing:{line_number}")
            if not isinstance(source_sha, str) or len(source_sha) != 64:
                raise RecoveryError(f"terminal_source_sha_missing:{line_number}")
            if path in terminal_paths or source_sha in terminal_source_sha:
                raise RecoveryError(f"terminal_duplicate_identity:{line_number}")
            terminal_paths.add(path)
            terminal_source_sha.add(source_sha)
    if terminal_documents != TERMINAL_DOCUMENT_COUNT:
        raise RecoveryError(f"terminal_document_count:{terminal_documents}")
    if sha256_file(TERMINAL_PROVENANCE) != TERMINAL_PROVENANCE_SHA256:
        raise RecoveryError("terminal_provenance_identity_changed")

    reserved_paths: set[str] = set()
    reserved_sha256: set[str] = set()
    reserved_sha1: set[str] = set()
    reserved_git_blob: set[str] = set()
    for line_number, row in enumerate(read_jsonl(PROFILE_DOCS), 1):
        if row.get("cpt_partition") != "cpt_validation":
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise RecoveryError(f"profile_path_missing:{line_number}")
        reserved_paths.add(path)
        for key, target in (
            ("sha256", reserved_sha256),
            ("sha1", reserved_sha1),
            ("git_blob_sha1", reserved_git_blob),
        ):
            value = row.get(key)
            if isinstance(value, str):
                target.add(value)

    groups = {row["group_id"] for row in frontier}
    if len(groups) != 79:
        raise RecoveryError(f"frontier_group_count:{len(groups)}")
    for row in frontier:
        group = row.get("group_id")
        if row.get("split") != "train_group" or row.get("cpt_partition") != "cpt_train":
            raise RecoveryError(f"frontier_row_not_train:{row.get('path')}")
        if split_by_group.get(group) != "train_group":
            raise RecoveryError(f"global_split_not_train:{group}")
        if partition_by_group.get(group) != "cpt_train":
            raise RecoveryError(f"cpt_partition_not_train:{group}")
        if not isinstance(row.get("path"), str) or not isinstance(row.get("bytes"), int):
            raise RecoveryError("frontier_source_identity_missing")

    return {
        "split_by_group": split_by_group,
        "partition_by_group": {str(k): str(v) for k, v in partition_by_group.items()},
        "protected_parent_hashes": protected,
        "terminal_paths": terminal_paths,
        "terminal_source_sha256": terminal_source_sha,
        "reserved_paths": reserved_paths,
        "reserved_source_sha256": reserved_sha256,
        "reserved_source_sha1": reserved_sha1,
        "reserved_git_blob_sha1": reserved_git_blob,
        "terminal_documents": terminal_documents,
        "profile_documents": len(reserved_paths),
    }


def stable_stat(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return {"path": str(path), "exists": True, "symlink": True}
    try:
        value = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "symlink": False,
        "bytes": value.st_size,
        "inode": value.st_ino,
        "device": value.st_dev,
        "mtime_ns": value.st_mtime_ns,
    }


def stat_identity(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(value.get(key) for key in ("bytes", "mtime_ns", "inode", "device"))


def read_description(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    expected_path = row.get("description_path")
    if not isinstance(expected_path, str) or not expected_path:
        return None, "description_path_missing"
    path = Path(expected_path)
    if path.is_symlink() or not path.is_file():
        return None, "description_unavailable"
    before = stable_stat(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "description_read_error"
    after = stable_stat(path)
    if stat_identity(before) != stat_identity(after):
        return {"path": str(path), "bytes": len(raw)}, "description_changed_during_read"
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row.get("description_sha256"):
        return {"path": str(path), "bytes": len(raw), "sha256": digest}, "description_changed"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"path": str(path), "bytes": len(raw), "sha256": digest}, "description_non_utf8"
    fields = description_fields(text)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": digest,
        "fields": fields,
    }, None


def license_reason(row: dict[str, Any], description: dict[str, Any]) -> str | None:
    fields = description.get("fields", {})
    license_value = str(fields.get("License", "")).strip()
    if not allowed_license(license_value):
        return "license_not_in_recognized_families"
    if fields.get("License_is_FOSS", "").strip().lower() == "no":
        return "license_requires_review_foss_no"
    if fields.get("License_restricts_use", "").strip().lower() == "yes":
        return "license_requires_review_restricts_use"
    expected = str(row.get("license", "")).strip()
    if expected and license_value != expected:
        return "license_metadata_changed"
    return None


def source_record(row: dict[str, Any], description: dict[str, Any], st_before: dict[str, Any],
                  hashes: dict[str, str], code_tokens: int, token_sha: str, chunks: int) -> dict[str, Any]:
    fields = description.get("fields", {})
    return {
        "schema": 1,
        "document_id": hashes["sha256"],
        "package": row["package"],
        "version": row.get("version"),
        "group_id": row["group_id"],
        "split": row["split"],
        "cpt_partition": row["cpt_partition"],
        "source_path": row["path"],
        "source_sha256": hashes["sha256"],
        "source_sha1": hashes["sha1"],
        "git_blob_sha1": hashes["git_blob_sha1"],
        "source_bytes": st_before.get("bytes"),
        "source_utf8_bytes": st_before.get("bytes"),
        "source_code_tokens": code_tokens,
        "token_stream_sha256": token_sha,
        "document_token_count": code_tokens,
        "description_path": description["path"],
        "description_sha256": description["sha256"],
        "license": fields.get("License", ""),
        "license_is_foss": fields.get("License_is_FOSS"),
        "license_restricts_use": fields.get("License_restricts_use"),
        "chunks": chunks,
        "oversized_source": bool(st_before.get("bytes", 0) > 4 * 1024 * 1024),
        "old_omission_reason": row.get("reason"),
        "frontier_seeded_index": row.get("seeded_index"),
        "source_stat": st_before,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            count += 1
    return count


class RecoveryRun:
    def __init__(self, args: argparse.Namespace, frontier: list[dict[str, Any]], guards: dict[str, Any], contract: Any | None):
        self.args = args
        self.frontier = frontier
        self.guards = guards
        self.contract = contract
        self.started = time.monotonic()
        self.builder_sha256 = sha256_file(Path(__file__))
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "groups").mkdir(exist_ok=True)
        self.seen_source_sha: dict[str, str] = {
            value: "terminal_provenance" for value in guards["terminal_source_sha256"]
        }
        self.seen_paths: dict[str, str] = {
            value: "terminal_provenance" for value in guards["terminal_paths"]
        }
        for value in guards["reserved_source_sha256"]:
            self.seen_source_sha.setdefault(value, "cpt_validation_profile")
        for value in guards["reserved_paths"]:
            self.seen_paths.setdefault(value, "cpt_validation_profile")
        self.counts: Counter[str] = Counter()
        self.exclusions: Counter[str] = Counter()
        self.repairs: Counter[str] = Counter()
        self.lengths: Counter[str] = Counter()
        self.completed_groups: set[str] = set()
        self._load_completed_groups()

    def _load_completed_groups(self) -> None:
        """Load only compact completed group metadata for a safe resume."""
        for group_dir in sorted((self.output / "groups").iterdir()):
            if not group_dir.is_dir() or group_dir.name.startswith("."):
                continue
            manifest_path = group_dir / "manifest.json"
            docs_path = group_dir / "documents.jsonl"
            if not manifest_path.is_file() or not docs_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "complete_candidate":
                    continue
                group_id = manifest.get("group_id")
                if not isinstance(group_id, str):
                    continue
                for row in read_jsonl(docs_path):
                    source_sha = row.get("source_sha256")
                    source_path = row.get("source_path")
                    if isinstance(source_sha, str):
                        prior = self.seen_source_sha.get(source_sha)
                        if prior and prior != f"recovery:{group_id}":
                            raise RecoveryError(f"resume_duplicate_source:{source_sha}")
                        self.seen_source_sha[source_sha] = f"recovery:{group_id}"
                    if isinstance(source_path, str):
                        self.seen_paths[source_path] = f"recovery:{group_id}"
                self.completed_groups.add(group_id)
                for key, value in manifest.get("counts", {}).items():
                    if isinstance(value, int):
                        self.counts[key] += value
                for key, value in manifest.get("exclusions", {}).items():
                    if isinstance(value, int):
                        self.exclusions[key] += value
                for key, value in manifest.get("repairs", {}).items():
                    if isinstance(value, int):
                        self.repairs[key] += value
                for key, value in manifest.get("length_histogram_input_ids", {}).items():
                    if isinstance(value, int):
                        self.lengths[key] += value
            except (OSError, json.JSONDecodeError) as exc:
                raise RecoveryError(f"invalid_completed_group:{group_dir}") from exc

    def preflight_stats(self) -> dict[str, Any]:
        stats: Counter[str] = Counter()
        for row in self.frontier:
            path = Path(row["path"])
            actual = stable_stat(path)
            if not actual.get("exists"):
                stats["missing"] += 1
            elif actual.get("symlink"):
                stats["symlink"] += 1
            elif actual.get("bytes") == 0:
                stats["zero_bytes"] += 1
            elif actual.get("bytes", 0) > 4 * 1024 * 1024:
                stats["over_4MiB_retained"] += 1
            else:
                stats["regular"] += 1
            if actual.get("bytes") != row.get("bytes"):
                stats["stat_bytes_changed"] += 1
        return {
            "frontier_documents": len(self.frontier),
            "frontier_groups": len({row["group_id"] for row in self.frontier}),
            "source_stat": dict(stats),
            "payload_opened": False,
            "training_admission": False,
        }

    def _write_exclusion(self, stream: Any, row: dict[str, Any], reason: str, **detail: Any) -> None:
        stream.write(json.dumps({
            "path": row.get("path"), "package": row.get("package"),
            "version": row.get("version"), "group_id": row.get("group_id"),
            "source_sha256": detail.pop("source_sha256", None),
            "reason": reason, **detail,
        }, separators=(",", ":"), sort_keys=True) + "\n")

    def _write_repair(self, stream: Any, row: dict[str, Any], reason: str, **detail: Any) -> None:
        stream.write(json.dumps({
            "path": row.get("path"), "package": row.get("package"),
            "version": row.get("version"), "group_id": row.get("group_id"),
            "reason": reason, **detail,
        }, separators=(",", ":"), sort_keys=True) + "\n")

    def process_group(self, group_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        group_name = f"{min(row['seeded_index'] for row in rows):06d}-{group_id}"
        final_dir = self.output / "groups" / group_name
        if group_id in self.completed_groups:
            return {"group_id": group_id, "status": "already_complete", "documents": 0}
        temp_dir = self.output / "groups" / f".{group_name}.tmp-{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        docs_path = temp_dir / "documents.jsonl"
        rows_path = temp_dir / "cpt_train.jsonl"
        excludes_path = temp_dir / "exclusions.jsonl"
        repairs_path = temp_dir / "repair-queue.jsonl"
        local_counts: Counter[str] = Counter()
        local_exclusions: Counter[str] = Counter()
        local_repairs: Counter[str] = Counter()
        local_lengths: Counter[str] = Counter()
        local_seen: dict[str, str] = {}
        document_rows: list[dict[str, Any]] = []
        try:
            with docs_path.open("w", encoding="utf-8") as docs, rows_path.open("w", encoding="utf-8") as output_rows, \
                    excludes_path.open("w", encoding="utf-8") as excludes, repairs_path.open("w", encoding="utf-8") as repairs:
                for row in sorted(rows, key=lambda value: str(value["path"])):
                    path = Path(row["path"])
                    description, description_error = read_description(row)
                    if description_error:
                        self._write_repair(repairs, row, description_error,
                                           expected_description_sha256=row.get("description_sha256"),
                                           observed_description=description)
                        local_repairs[description_error] += 1
                        continue
                    assert description is not None
                    reason = license_reason(row, description)
                    if reason:
                        detail = {
                            "expected_license": row.get("license"),
                            "observed_license": description.get("fields", {}).get("License"),
                            "description_sha256": description.get("sha256"),
                        }
                        if reason == "license_metadata_changed":
                            self._write_repair(repairs, row, reason, **detail)
                            local_repairs[reason] += 1
                        else:
                            self._write_exclusion(excludes, row, reason, **detail)
                            local_exclusions[reason] += 1
                        continue
                    before = stable_stat(path)
                    if not before.get("exists"):
                        self._write_repair(repairs, row, "source_unavailable", observed_stat=before)
                        local_repairs["source_unavailable"] += 1
                        continue
                    if before.get("symlink"):
                        self._write_repair(repairs, row, "source_symlink_requires_review", observed_stat=before)
                        local_repairs["source_symlink_requires_review"] += 1
                        continue
                    if before.get("bytes") == 0:
                        self._write_exclusion(excludes, row, "empty_source_no_payload", observed_stat=before)
                        local_exclusions["empty_source_no_payload"] += 1
                        continue
                    if before.get("bytes") != row.get("bytes"):
                        self._write_repair(repairs, row, "source_stat_changed", expected_bytes=row.get("bytes"), observed_stat=before)
                        local_repairs["source_stat_changed"] += 1
                        continue
                    try:
                        with path.open("rb") as source:
                            raw = source.read()
                    except OSError as exc:
                        self._write_repair(repairs, row, "source_read_error", error=type(exc).__name__)
                        local_repairs["source_read_error"] += 1
                        continue
                    after = stable_stat(path)
                    if stat_identity(before) != stat_identity(after) or len(raw) != before.get("bytes"):
                        self._write_repair(repairs, row, "source_changed_during_read", before=before, after=after)
                        local_repairs["source_changed_during_read"] += 1
                        continue
                    hashes = source_fingerprints(raw)
                    source_sha = hashes["sha256"]
                    expected_source_sha = row.get("source_sha256")
                    if expected_source_sha not in (None, "") and expected_source_sha != source_sha:
                        self._write_repair(repairs, row, "source_metadata_hash_changed",
                                           expected_source_sha256=expected_source_sha,
                                           observed_source_sha256=source_sha)
                        local_repairs["source_metadata_hash_changed"] += 1
                        continue
                    if source_sha in self.guards["reserved_source_sha256"]:
                        self._write_exclusion(excludes, row, "exact_duplicate_cpt_validation_source", source_sha256=source_sha)
                        local_exclusions["exact_duplicate_cpt_validation_source"] += 1
                        continue
                    if source_sha in self.guards["terminal_source_sha256"] or path.as_posix() in self.guards["terminal_paths"]:
                        self._write_exclusion(excludes, row, "exact_duplicate_terminal_source", source_sha256=source_sha,
                                              terminal_path_match=path.as_posix() in self.guards["terminal_paths"])
                        local_exclusions["exact_duplicate_terminal_source"] += 1
                        continue
                    if source_sha in self.seen_source_sha or source_sha in local_seen:
                        self._write_exclusion(excludes, row, "exact_duplicate_recovery_source", source_sha256=source_sha,
                                              duplicate_of=self.seen_source_sha.get(source_sha) or local_seen.get(source_sha))
                        local_exclusions["exact_duplicate_recovery_source"] += 1
                        continue
                    if any(hashes[key] in self.guards["protected_parent_hashes"] for key in ("sha1", "git_blob_sha1")):
                        self._write_exclusion(excludes, row, "known_nontrain_parent_hash_match", source_sha256=source_sha,
                                              sha1=hashes["sha1"], git_blob_sha1=hashes["git_blob_sha1"])
                        local_exclusions["known_nontrain_parent_hash_match"] += 1
                        continue
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        self._write_repair(repairs, row, "source_non_utf8", error=str(exc))
                        local_repairs["source_non_utf8"] += 1
                        continue
                    if "\0" in text:
                        self._write_repair(repairs, row, "source_contains_nul")
                        local_repairs["source_contains_nul"] += 1
                        continue
                    if self.contract is None:
                        raise RecoveryError("token_contract_not_loaded")
                    try:
                        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
                        decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
                    except Exception as exc:
                        self._write_repair(repairs, row, "tokenizer_error", error=type(exc).__name__)
                        local_repairs["tokenizer_error"] += 1
                        continue
                    if decoded != text:
                        self._write_repair(repairs, row, "tokenizer_roundtrip_mismatch")
                        local_repairs["tokenizer_roundtrip_mismatch"] += 1
                        continue
                    if not ids:
                        self._write_exclusion(excludes, row, "empty_token_stream_degenerate", source_sha256=source_sha)
                        local_exclusions["empty_token_stream_degenerate"] += 1
                        continue
                    if any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in ids):
                        self._write_repair(repairs, row, "token_id_out_of_range", source_sha256=source_sha)
                        local_repairs["token_id_out_of_range"] += 1
                        continue
                    if any(value in (BOS_ID, EOS_ID) for value in ids):
                        self._write_repair(repairs, row, "raw_special_token_id", source_sha256=source_sha)
                        local_repairs["raw_special_token_id"] += 1
                        continue
                    token_sha = token_stream_sha256(ids)
                    chunks = list(self.contract.chunks(ids, CHUNK_SIZE))
                    for chunk_index, chunk in enumerate(chunks):
                        materialized = {
                            "schema": 1,
                            "row_id": f"{source_sha}:{chunk_index}",
                            "document_id": source_sha,
                            "package": row["package"],
                            "group_id": row["group_id"],
                            "cpt_partition": "cpt_train",
                            "source_path": row["path"],
                            "source_sha256": source_sha,
                            "chunk_index": chunk_index,
                            "input_ids": chunk["input_ids"],
                            "labels": chunk["labels"],
                            "attention_mask": chunk["attention_mask"],
                            "source_token_start": chunk["source_token_start"],
                            "source_token_end": chunk["source_token_end"],
                            "overlap_context_tokens": chunk["overlap_context_tokens"],
                            "is_document_end": chunk["is_document_end"],
                            "supervised_tokens": chunk["supervised_tokens"],
                            "document_token_count": len(ids),
                            "token_stream_sha256": token_sha,
                            "builder_id": "r2-cpt-prefix-cap-recovery-v1",
                            "builder_sha256": self.builder_sha256,
                            "tokenizer_revision": TOKENIZER_REVISION,
                        }
                        output_rows.write(json.dumps(materialized, separators=(",", ":"), sort_keys=True) + "\n")
                        local_counts["rows"] += 1
                        local_counts["input_tokens"] += len(chunk["input_ids"])
                        local_counts["supervised_tokens"] += chunk["supervised_tokens"]
                        local_lengths[str(len(chunk["input_ids"]))] += 1
                    doc = source_record(row, description, before, hashes, len(ids), token_sha, len(chunks))
                    docs.write(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
                    document_rows.append(doc)
                    local_counts["documents"] += 1
                    local_counts["raw_bytes"] += len(raw)
                    local_counts["code_tokens"] += len(ids)
                    local_seen[source_sha] = f"recovery:{group_id}"
                docs.flush()
                output_rows.flush()
                excludes.flush()
                repairs.flush()
            for source_sha in local_seen:
                self.seen_source_sha[source_sha] = local_seen[source_sha]
            for doc in document_rows:
                self.seen_paths[doc["source_path"]] = f"recovery:{group_id}"
            local_manifest = {
                "schema": 1,
                "status": "complete_candidate",
                "group_id": group_id,
                "seeded_indices": sorted({row["seeded_index"] for row in rows}),
                "counts": dict(local_counts),
                "exclusions": dict(local_exclusions),
                "repairs": dict(local_repairs),
                "chunk_size": CHUNK_SIZE,
                "max_source_bytes": None,
                "oversized_source_policy": "retain_and_measure_no_4MiB_cap",
                "truncation": False,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "tokenizer_revision": TOKENIZER_REVISION,
                "builder_sha256": self.builder_sha256,
                "artifacts": {
                    name: {"bytes": (temp_dir / name).stat().st_size, "sha256": sha256_file(temp_dir / name)}
                    for name in ("documents.jsonl", "cpt_train.jsonl", "exclusions.jsonl", "repair-queue.jsonl")
                },
                "length_histogram_input_ids": dict(local_lengths),
                "training_admission": False,
            }
            (temp_dir / "manifest.json").write_text(json.dumps(local_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temp_dir, final_dir)
            self.completed_groups.add(group_id)
            self.counts.update(local_counts)
            self.exclusions.update(local_exclusions)
            self.repairs.update(local_repairs)
            self.lengths.update(local_lengths)
            return {"group_id": group_id, "status": "complete_candidate", **dict(local_counts), "exclusions": dict(local_exclusions), "repairs": dict(local_repairs)}
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    @property
    def tokenizer(self) -> Any:
        if not hasattr(self, "_tokenizer"):
            if sha256_file(TOKENIZER) != TOKENIZER_SHA256:
                raise RecoveryError("tokenizer_identity_changed")
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(TOKENIZER))
        return self._tokenizer

    def run(self) -> dict[str, Any]:
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.frontier:
            by_group[row["group_id"]].append(row)
        progress_path = self.output / "progress.json"
        progress = {
            "schema": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "frontier_documents": len(self.frontier),
            "frontier_groups": len(by_group),
            "groups_completed": len(self.completed_groups),
            "groups_remaining": len(by_group) - len(self.completed_groups),
            "payload_opened": False,
            "training_admission": False,
        }
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for index, group_id in enumerate(sorted(by_group, key=lambda value: min(row["seeded_index"] for row in by_group[value])), 1):
            if group_id in self.completed_groups:
                continue
            result = self.process_group(group_id, by_group[group_id])
            progress.update({
                "groups_completed": len(self.completed_groups),
                "groups_remaining": len(by_group) - len(self.completed_groups),
                "last_group": group_id,
                "last_group_result": result,
                "elapsed_seconds": time.monotonic() - self.started,
            })
            progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"group": group_id, "index": index, "groups": len(by_group), "result": result}, sort_keys=True), flush=True)
        progress.update({"status": "complete_candidate", "elapsed_seconds": time.monotonic() - self.started})
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema": 1,
            "status": "CPU_materialized_candidate_not_training_admission",
            "frontier": pin(self.args.frontier),
            "frontier_expected_sha256": FRONTIER_SHA256,
            "frontier_documents": len(self.frontier),
            "frontier_groups": len(by_group),
            "groups_completed": len(self.completed_groups),
            "counts": dict(self.counts),
            "exclusions": dict(self.exclusions),
            "repairs": dict(self.repairs),
            "length_histogram_input_ids": dict(self.lengths),
            "tokenizer": pin(TOKENIZER),
            "tokenizer_revision": TOKENIZER_REVISION,
            "chunk_size": CHUNK_SIZE,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "pad_id": PAD_ID,
            "vocab_size": VOCAB_SIZE,
            "builder": {"path": str(Path(__file__)), "sha256": self.builder_sha256},
            "raw_chunk_contract": pin(RAW_CPT),
            "guards": {
                "global_split": pin(GLOBAL_SPLIT),
                "cpt_partition": pin(PARTITION),
                "protected_parent_hashes": pin(PROTECTED),
                "profile_documents": pin(PROFILE_DOCS),
                "terminal_document_provenance": {
                    **pin(TERMINAL_PROVENANCE),
                    "expected_sha256": TERMINAL_PROVENANCE_SHA256,
                    "documents": TERMINAL_DOCUMENT_COUNT,
                },
            },
            "source_policy": {
                "source_rehashed_after_stable_stat": True,
                "max_source_bytes": None,
                "oversized_source_policy": "retain_and_measure_no_4MiB_cap",
                "target_truncation": False,
                "empty_or_degenerate": "named_exclusion",
                "unreadable_or_changed": "named_repair_queue",
                "dedup": ["terminal_source_sha256", "profile_cpt_validation_sha256", "protected_sha1_or_git_blob", "recovery_source_sha256"],
            },
            "terminal_global_dedup_pending": True,
            "training_admission": False,
            "elapsed_seconds": time.monotonic() - self.started,
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[:2]))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    frontier = read_jsonl(args.frontier)
    guards = load_guards(frontier)
    if args.preflight_only:
        probe = RecoveryRun(args, frontier, guards, None)
        result = probe.preflight_stats()
        result["pins"] = {
            "frontier": pin(args.frontier),
            "tokenizer": pin(TOKENIZER),
            "raw_chunk_contract": pin(RAW_CPT),
            "terminal_provenance": pin(TERMINAL_PROVENANCE),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    contract = load_raw_contract()
    run = RecoveryRun(args, frontier, guards, contract)
    manifest = run.run()
    print(json.dumps({"output": str(args.output), "status": manifest["status"], "counts": manifest["counts"], "exclusions": manifest["exclusions"], "repairs": manifest["repairs"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
