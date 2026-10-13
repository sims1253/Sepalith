#!/usr/bin/env python3
"""Build a bounded, source-bound PRM03 task-SFT candidate mixture.

This is a preparation-only pass.  It reads already audited/tokenized rows and
already source-checked candidate packets, verifies the exact DAT-02 group
binding, tokenizes only the structured/completion packets with the pinned
protocol, and writes a new unadmitted candidate registry.  The completion
packets receive only the accepted strict-v2 EOF outer-brace repair after an
independent exact source-line check.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    except OSError:
        pass
sys.dont_write_bytecode = True

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
OUT = PLAN / "work/r2-task-mixture-v1"
CORRECTED = PLAN / "work/lead/finish-corrected-train-v1/train-token-rows.jsonl"
SHORT = PLAN / "work/r2-short-task-preparation-v1/candidate-packets.jsonl"
STRUCTURED = PLAN / "work/lead/r2-task-materialization-a/structured/candidate-packets.jsonl"
STRUCTURED_REPORT = PLAN / "work/lead/r2-task-materialization-a/structured/report.json"
COMPLETION = PLAN / "work/lead/r2-task-materialization-a/completion/completion-packets.jsonl"
COMPLETION_SUMMARY = PLAN / "work/lead/r2-task-materialization-a/completion/completion-summary.json"
DAT05_PROVENANCE = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/provenance.jsonl")
DAT02 = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")
PROTOCOL = EXEC / "packages/sepalith/src/sepalith/campaign_protocol.py"
SAMPLER = EXEC / "experiments/training/campaign_sampling.py"
STRICT_REPAIR = PLAN / "work/finish-boundary-repair-v2/finish_boundary_repair_v2.py"

EXPECTED_HASHES = {
    CORRECTED: "e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe",
    SHORT: "a301ba4ed8d0355f4a2d3a83404c3bef41ee31b4820b60980af8afe201ebc7d7",
    STRUCTURED: "ac0e726269f2694c60f7b7781355f080f48a8e37b827db4f4b29079dbc1c3069",
    COMPLETION: "c4358f9f37436a5e1e090c4e38379d2219e858b588d9988070a689a01b26834c",
    DAT05_PROVENANCE: "6883f0691301e79180164920414e4752f53edcfe555c73b1ea0b02dd599d2123",
    DAT02: "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09",
    PARTITION: "6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06",
    TOKENIZER: "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    PROTOCOL: "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
    SAMPLER: "60e4e4d60b99a4b9851b3e35194f81fb2d0a2a3c47d0cac931c2091819c03dd0",
}

ROW_OUT = OUT / "candidate-token-rows.jsonl"
PROV_OUT = OUT / "candidate-provenance.jsonl"
EXCLUSIONS_OUT = OUT / "candidate-exclusions.jsonl"
SCHEDULE_OUT = OUT / "proposed-draw-manifest.json"
REPORT_OUT = OUT / "mixture-report.json"
SOURCE_MANIFEST_OUT = OUT / "source-manifest.json"


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return sha256_text(canonical(value))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PreparationError(RuntimeError):
    pass


def require_hashes() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise PreparationError(f"missing pinned input: {path}")
        actual, size = sha256_file(path)
        if actual != expected:
            raise PreparationError(f"hash mismatch for {path}: {actual} != {expected}")
        evidence[str(path)] = {"sha256": actual, "bytes": size}
    # These are read as provenance only; pin them to the exact accepted reports
    # when present, without making a report-only hash a data admission input.
    for path in (STRUCTURED_REPORT, COMPLETION_SUMMARY):
        if path.is_file():
            actual, size = sha256_file(path)
            evidence[str(path)] = {"sha256": actual, "bytes": size, "role": "accepted_input_report"}
    return evidence


def dat02_and_partition() -> tuple[dict[str, dict[str, Any]], dict[str, str], set[str], dict[str, Any]]:
    registry = json.loads(DAT02.read_text(encoding="utf-8"))
    if not isinstance(registry.get("groups"), list):
        raise PreparationError("DAT-02 groups is not a list")
    groups: dict[str, dict[str, Any]] = {}
    for item in registry["groups"]:
        if not isinstance(item, dict) or not isinstance(item.get("group_id"), str):
            raise PreparationError("DAT-02 group lacks group_id")
        gid = item["group_id"]
        if gid in groups:
            raise PreparationError(f"duplicate DAT-02 group: {gid}")
        groups[gid] = item
    partition_doc = json.loads(PARTITION.read_text(encoding="utf-8"))
    pmap = partition_doc.get("groups")
    if not isinstance(pmap, dict):
        raise PreparationError("CPT partition groups is not an object")
    reserved = {gid for gid, value in pmap.items() if value == "cpt_validation"}
    if len(reserved) != 556:
        raise PreparationError(f"expected 556 reserved CPT validation groups, got {len(reserved)}")
    if sum(value == "cpt_train" for value in pmap.values()) != 10163:
        raise PreparationError("unexpected CPT train group count")
    meta = {
        "dat02_split_id": registry.get("split_id"),
        "dat02_group_count": len(groups),
        "partition_schema": partition_doc.get("schema"),
        "partition_split_id": partition_doc.get("split_id"),
        "partition_group_count": len(pmap),
        "cpt_train_groups": sum(value == "cpt_train" for value in pmap.values()),
        "cpt_validation_groups": len(reserved),
        "policy": "require DAT-02 group split train/train_group and explicit partition cpt_train; reject absent mapping",
    }
    return groups, pmap, reserved, meta


def binding_status(gid: object, groups: Mapping[str, Mapping[str, Any]], pmap: Mapping[str, str]) -> str:
    if not isinstance(gid, str) or not gid:
        return "missing_group_id"
    if gid not in groups:
        return "unknown_dat02_group"
    if gid not in pmap:
        return "group_not_in_cpt_partition"
    partition = pmap[gid]
    if partition == "cpt_validation":
        return "cpt_validation_reserved"
    if partition != "cpt_train":
        return "partition_not_cpt_train"
    split = groups[gid].get("split")
    if split not in ("train", "train_group"):
        return f"dat02_split_{split}"
    return "bound_cpt_train"


def iter_jsonl(path: Path):
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PreparationError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise PreparationError(f"non-object JSON at {path}:{line_number}")
            yield line_number, raw, value


def concise_provenance(source: str, row: Mapping[str, Any], *, group_id: str, source_meta: Mapping[str, Any],
                       split_status: str, repair: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep source identity/hash evidence without copying target/prompt text."""
    result: dict[str, Any] = {
        "source_class": source,
        "lineage": source_meta.get("lineage", source),
        "group_id": group_id,
        "split_binding": split_status,
        "package_id": row.get("package_id"),
        "source": source_meta.get("source"),
        "source_path": source_meta.get("source_path"),
        "source_file": source_meta.get("source_file"),
        "source_line": source_meta.get("source_line"),
        "source_sha256": source_meta.get("source_sha256"),
        "raw_line_sha256": source_meta.get("raw_line_sha256"),
        "source_snapshot_path": source_meta.get("source_snapshot_path"),
        "source_snapshot_sha256": source_meta.get("source_snapshot_sha256"),
        "source_snapshot_is_simulated": source_meta.get("source_snapshot_is_simulated"),
        "source_verification": source_meta.get("source_verification"),
        "input_row_id": source_meta.get("input_row_id"),
        "renderer_id": row.get("renderer_id"),
        "tokenization_policy": row.get("tokenization_policy"),
        "tokenizer_revision": row.get("tokenizer_revision"),
        "target_operation": row.get("target_operation"),
        "repair": dict(repair or {}),
    }
    # Avoid serialising null clutter while retaining explicit false values.
    return {key: value for key, value in result.items() if value is not None}


class CandidateWriter:
    def __init__(self, groups: Mapping[str, Mapping[str, Any]], pmap: Mapping[str, str], exclusions: list[dict[str, Any]]):
        self.groups = groups
        self.pmap = pmap
        self.exclusions = exclusions
        self.rows_handle = ROW_OUT.open("xb")
        self.prov_handle = PROV_OUT.open("xb")
        self.rows_hash = hashlib.sha256()
        self.prov_hash = hashlib.sha256()
        self.row_count = 0
        self.provenance_count = 0
        self.row_ids: set[str] = set()
        self.prompt_seen: dict[str, tuple[str, str, str]] = {}
        self.counts = Counter()
        self.family_counts = Counter()
        self.operation_counts = Counter()
        self.source_counts = Counter()
        self.noop_target_tokens = 0
        self.target_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.noop_rows = 0
        self.cap_rows = Counter()
        self.conflicts: list[dict[str, Any]] = []
        self.sampler_rows: list[dict[str, Any]] = []

    def close(self) -> None:
        self.rows_handle.flush()
        self.prov_handle.flush()
        os.fsync(self.rows_handle.fileno())
        os.fsync(self.prov_handle.fileno())
        self.rows_handle.close()
        self.prov_handle.close()

    def reject(self, source: str, ident: object, reason: str, *, family: object = None, group_id: object = None,
               detail: Mapping[str, Any] | None = None) -> None:
        self.exclusions.append({
            "source_class": source,
            "input_id": ident,
            "family": family,
            "group_id": group_id,
            "reason": reason,
            **dict(detail or {}),
        })
        self.counts[f"excluded:{source}:{reason}"] += 1

    def add(self, row: dict[str, Any], *, source: str, group_id: str, source_meta: Mapping[str, Any],
            split_status: str, repair: Mapping[str, Any] | None = None) -> bool:
        ident = row.get("id")
        family = row.get("family")
        if not isinstance(ident, str) or not ident:
            self.reject(source, ident, "row_id_missing", family=family, group_id=group_id)
            return False
        if ident in self.row_ids:
            self.reject(source, ident, "duplicate_row_id", family=family, group_id=group_id)
            return False
        try:
            from sepalith.campaign_protocol import validate_training_row
            validate_training_row(row)
        except Exception as error:  # checked source row or builder result
            self.reject(source, ident, f"row_contract:{type(error).__name__}:{str(error)[:120]}", family=family, group_id=group_id)
            return False
        target_labels = int(row["target_token_count"]) + 1
        total = len(row["input_ids"])
        if target_labels > 192:
            self.cap_rows[f"{source}:target_over_192"] += 1
            self.reject(source, ident, "target_over_192_including_EOS", family=family, group_id=group_id,
                        detail={"target_label_tokens": target_labels})
            return False
        if total > 4096:
            self.cap_rows[f"{source}:total_over_4096"] += 1
            self.reject(source, ident, "total_over_4096", family=family, group_id=group_id,
                        detail={"total_tokens": total})
            return False
        if row.get("split") != "train":
            self.reject(source, ident, "split_not_train", family=family, group_id=group_id)
            return False
        prompt_hash = sha256_text(row["prompt_text"])
        target_hash = sha256_text(row["target_text"])
        prior = self.prompt_seen.get(prompt_hash)
        if prior is not None:
            prior_target_hash, prior_id, prior_source = prior
            reason = "prompt_duplicate_same_target" if target_hash == prior_target_hash else "prompt_conflicting_target"
            self.reject(source, ident, reason, family=family, group_id=group_id,
                        detail={"prompt_sha256": prompt_hash, "prior_id": prior_id, "prior_source_class": prior_source,
                                "target_sha256": target_hash, "prior_target_sha256": prior_target_hash})
            if reason == "prompt_conflicting_target":
                self.conflicts.append({"prompt_sha256": prompt_hash, "retained_id": prior_id,
                                       "excluded_id": ident, "retained_source_class": prior_source,
                                       "excluded_source_class": source})
            return False
        self.prompt_seen[prompt_hash] = (target_hash, ident, source)
        line = (canonical(row) + "\n").encode("utf-8")
        self.rows_handle.write(line)
        self.rows_hash.update(line)
        pmeta = concise_provenance(source, row, group_id=group_id, source_meta=source_meta,
                                   split_status=split_status, repair=repair)
        pmeta.update({
            "id": ident,
            "prompt_sha256": prompt_hash,
            "target_sha256": target_hash,
            "row_canonical_sha256": canonical_hash(row),
            "target_label_tokens_including_protocol_EOS": target_labels,
            "total_tokens": total,
        })
        pline = (canonical(pmeta) + "\n").encode("utf-8")
        self.prov_handle.write(pline)
        self.prov_hash.update(pline)
        self.row_ids.add(ident)
        self.row_count += 1
        self.provenance_count += 1
        self.counts[f"accepted:{source}"] += 1
        self.family_counts[str(family)] += 1
        operation = row["target_operation"]
        self.operation_counts[operation] += 1
        self.target_tokens += target_labels
        self.prompt_tokens += int(row["target_start"])
        self.total_tokens += total
        if operation == "no_op":
            self.noop_rows += 1
            self.noop_target_tokens += target_labels
        self.source_counts[source] += 1
        # Metadata only: no prompt/target/token-ID values enter the sampler.
        self.sampler_rows.append({
            "row_id": ident,
            "family": str(family),
            "source_id": group_id,
            "package_id": str(row["package_id"]),
            "split": "train",
            "semantic_noop": operation == "no_op",
            "operation": operation,
            "prompt_tokens": int(row["target_start"]),
            "target_tokens": target_labels,
            "total_tokens": total,
            "length_bucket": "short" if total <= 2048 else "long",
            "naturally_long": total > 2048,
            "source_kind": "ordinary",
            "provenance": f"{source}:{group_id}",
        })
        return True

    def finalize_hashes(self) -> dict[str, Any]:
        return {
            "candidate_rows_sha256": self.rows_hash.hexdigest(),
            "candidate_rows_bytes": ROW_OUT.stat().st_size,
            "candidate_rows": self.row_count,
            "candidate_provenance_sha256": self.prov_hash.hexdigest(),
            "candidate_provenance_bytes": PROV_OUT.stat().st_size,
            "candidate_provenance_rows": self.provenance_count,
        }


def load_corrected_provenance(corrected_ids: set[str]) -> tuple[dict[str, dict[str, Any]], int]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicate = 0
    for _line_no, _raw, item in iter_jsonl(DAT05_PROVENANCE):
        ident = item.get("id")
        if ident not in corrected_ids:
            continue
        if ident in by_id:
            duplicate += 1
            continue
        by_id[ident] = item
    return by_id, duplicate


def verify_completion_source_refs(
    refs: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Hash each exact completion source JSONL once and retain only requested rows."""
    raw_by_ref: dict[tuple[str, int], dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for source_file, by_line in sorted(refs.items()):
        path = Path(source_file)
        if not path.is_file():
            errors.append({"source_file": source_file, "reason": "source_file_missing"})
            continue
        before = path.stat()
        digest = hashlib.sha256()
        retained = 0
        with path.open("rb", buffering=4 * 1024 * 1024) as stream:
            for line_no, raw in enumerate(stream, 1):
                digest.update(raw)
                expected = by_line.get(line_no)
                if expected is None:
                    continue
                if sha256_bytes(raw) != expected["raw_line_sha256"]:
                    errors.append({"source_file": source_file, "source_line": line_no,
                                   "reason": "source_raw_line_hash_mismatch"})
                    continue
                try:
                    item = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    errors.append({"source_file": source_file, "source_line": line_no,
                                   "reason": "source_row_not_json"})
                    continue
                if not isinstance(item, dict):
                    errors.append({"source_file": source_file, "source_line": line_no,
                                   "reason": "source_row_not_object"})
                    continue
                raw_by_ref[(source_file, line_no)] = item
                retained += 1
        after = path.stat()
        observed = digest.hexdigest()
        expected_file_hashes = {str(v["source_sha256"]) for v in by_line.values()}
        if len(expected_file_hashes) != 1:
            errors.append({"source_file": source_file, "reason": "source_file_expected_hash_disagreement"})
        elif observed != next(iter(expected_file_hashes)):
            errors.append({"source_file": source_file, "reason": "source_file_hash_mismatch",
                           "observed_sha256": observed, "expected_sha256": next(iter(expected_file_hashes))})
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            errors.append({"source_file": source_file, "reason": "source_file_mutated_during_read"})
        evidence.append({"path": source_file, "bytes": after.st_size, "sha256": observed,
                         "requested_lines": len(by_line), "retained_rows": retained,
                         "verification": "full_file_hash_and_exact_requested_raw_line_hash"})
    return raw_by_ref, errors + [{"source_file": "<ref>", **e} for e in []]


def source_meta_corrected(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lineage": "corrected_train_source_bound_registry_row",
        "source": item.get("source"),
        "source_file": item.get("source_file") or item.get("file"),
        "source_line": item.get("source_line") or item.get("line"),
        "source_sha256": item.get("source_sha256"),
        "raw_line_sha256": item.get("raw_line_sha256"),
        "input_row_id": item.get("id"),
        "source_verification": "DAT-05 admitted provenance joined by exact row id; DAT-02 group binding independently checked",
    }


def source_meta_short(packet: Mapping[str, Any]) -> dict[str, Any]:
    p = packet.get("source_provenance", {})
    return {
        "lineage": "short_audited_source_candidate",
        "source": p.get("source_role"),
        "source_path": p.get("source_path"),
        "source_sha256": p.get("source_sha256"),
        "raw_line_sha256": p.get("raw_line_sha256"),
        "source_snapshot_path": p.get("source_snapshot_path"),
        "source_snapshot_sha256": p.get("source_snapshot_sha256"),
        "source_snapshot_is_simulated": p.get("source_snapshot_is_simulated"),
        "input_row_id": p.get("source_document_id") or packet.get("row", {}).get("id"),
        "source_verification": "accepted short-packet source/hash observations and full application flags; no new raw sweep",
    }


def source_meta_structured(packet: Mapping[str, Any]) -> dict[str, Any]:
    rr = packet.get("row_ref", {})
    p = packet.get("result", {}).get("provenance", {})
    v = packet.get("validation", {})
    return {
        "lineage": "structured_source_candidate_packet",
        "source": rr.get("source"),
        "source_file": rr.get("file"),
        "source_line": rr.get("line"),
        "source_sha256": rr.get("source_sha256"),
        "raw_line_sha256": rr.get("raw_line_sha256"),
        "source_snapshot_path": p.get("source_snapshot_path"),
        "source_snapshot_sha256": p.get("source_snapshot_sha256") or v.get("source_sha256"),
        "source_snapshot_is_simulated": p.get("source_snapshot_is_simulated"),
        "input_row_id": rr.get("row_id"),
        "source_verification": {
            "packet_flags": {
                key: v.get(key) for key in ("fresh_source_constructor", "full_buffer_application",
                                             "normalized_parent_R_parse", "source_selection_support_check",
                                             "post_edit_geometry_verified") if key in v
            },
            "basis": "accepted structured report source checks; source snapshots are retained as expected hashes",
        },
    }


def source_meta_completion(packet: Mapping[str, Any], *, raw_matches: Mapping[str, bool]) -> dict[str, Any]:
    rr = packet.get("row_ref", {})
    return {
        "lineage": "completion_source_derived_strict_v2_outer_brace_repair",
        "source": rr.get("source"),
        "source_file": rr.get("source_file"),
        "source_line": rr.get("source_line"),
        "source_sha256": rr.get("source_sha256"),
        "raw_line_sha256": rr.get("raw_line_sha256"),
        "input_row_id": rr.get("row_id"),
        "source_verification": {
            "basis": "exact source JSONL full-file hash plus requested raw-line hash",
            "raw_prefix_matches_selection_document": bool(raw_matches.get("prefix")),
            "raw_constructor_splice_matches_corpus_target": bool(raw_matches.get("target")),
        },
    }


def main() -> int:
    started = time.monotonic()
    for path in (ROW_OUT, PROV_OUT, EXCLUSIONS_OUT, SCHEDULE_OUT, REPORT_OUT, SOURCE_MANIFEST_OUT):
        if path.exists():
            raise PreparationError(f"refusing to overwrite output: {path}")
    input_evidence = require_hashes()
    groups, pmap, reserved, split_meta = dat02_and_partition()
    # Exact corrected-ID join: package names never determine group membership.
    corrected_ids: set[str] = set()
    corrected_lines = 0
    for _line_no, _raw, item in iter_jsonl(CORRECTED):
        corrected_lines += 1
        ident = item.get("id")
        if not isinstance(ident, str) or not ident:
            raise PreparationError("corrected row missing id")
        corrected_ids.add(ident)
    corrected_prov, corrected_prov_duplicates = load_corrected_provenance(corrected_ids)
    exclusions: list[dict[str, Any]] = []
    writer = CandidateWriter(groups, pmap, exclusions)
    writer.counts["input:corrected_rows"] = corrected_lines
    writer.counts["input:corrected_provenance_matches"] = len(corrected_prov)
    writer.counts["input:corrected_provenance_duplicate_records"] = corrected_prov_duplicates

    # Existing corrected rows are already tokenized by the accepted protocol.
    for _line_no, _raw, row in iter_jsonl(CORRECTED):
        ident = row.get("id")
        prov = corrected_prov.get(ident, {})
        gid = prov.get("group_id")
        status = binding_status(gid, groups, pmap)
        if ident not in corrected_prov:
            writer.reject("corrected", ident, "missing_DAT05_exact_id_provenance", family=row.get("family"), group_id=gid)
            continue
        if prov.get("decision") != "admitted" or prov.get("row_split") != "train":
            writer.reject("corrected", ident, "DAT05_provenance_not_admitted_train", family=row.get("family"), group_id=gid)
            continue
        if prov.get("family") not in (None, row.get("family")) and prov.get("semantic_family") not in (None, row.get("family")):
            writer.reject("corrected", ident, "DAT05_family_binding_mismatch", family=row.get("family"), group_id=gid)
            continue
        if prov.get("package_id", prov.get("normalized_package_id")) not in (None, row.get("package_id")) and prov.get("normalized_package_id") != row.get("package_id"):
            writer.reject("corrected", ident, "DAT05_package_binding_mismatch", family=row.get("family"), group_id=gid)
            continue
        if status != "bound_cpt_train":
            writer.reject("corrected", ident, status, family=row.get("family"), group_id=gid)
            continue
        writer.add(row, source="corrected", group_id=gid, source_meta=source_meta_corrected(prov), split_status=status)

    # The 411 short packets are complete token rows.  Preserve their exact
    # source hashes/lineage and use their packet group_id directly.
    for _line_no, _raw, packet in iter_jsonl(SHORT):
        row = packet.get("row", {})
        ident = row.get("id")
        sp = packet.get("source_provenance", {})
        gid = sp.get("group_id")
        status = binding_status(gid, groups, pmap)
        if packet.get("candidate_status") != "source_and_token_checks_passed_not_training_admitted":
            writer.reject("short", ident, "short_candidate_status_not_passed", family=row.get("family"), group_id=gid)
            continue
        if status != "bound_cpt_train":
            writer.reject("short", ident, status, family=row.get("family"), group_id=gid)
            continue
        writer.add(row, source="short", group_id=gid, source_meta=source_meta_short(packet), split_status=status)

    # Structured packets already contain a source/application validator result.
    # We do not perform a broad normalized-corpus scan here; all four source
    # support booleans, exact group binding, and immutable source hashes must be
    # present before this packet can be tokenized.
    structured_source_refs: dict[str, set[str]] = defaultdict(set)
    for _line_no, _raw, packet in iter_jsonl(STRUCTURED):
        rr = packet.get("row_ref", {})
        res = packet.get("result", {})
        val = packet.get("validation", {})
        gid = rr.get("group_id")
        status = binding_status(gid, groups, pmap)
        ident = rr.get("row_id")
        if status != "bound_cpt_train":
            writer.reject("structured", ident, status, family=packet.get("family"), group_id=gid)
            continue
        if res.get("status") != "converted":
            writer.reject("structured", ident, "packet_status_not_converted", family=packet.get("family"), group_id=gid)
            continue
        required_flags = ("fresh_source_constructor", "full_buffer_application", "normalized_parent_R_parse", "source_selection_support_check")
        if any(val.get(flag) is not True for flag in required_flags) or val.get("admission") is not False:
            writer.reject("structured", ident, "structured_source_application_flags_incomplete", family=packet.get("family"), group_id=gid)
            continue
        provenance = res.get("provenance", {})
        snapshot_path = provenance.get("source_snapshot_path")
        snapshot_hash = provenance.get("source_snapshot_sha256") or val.get("source_sha256")
        if not isinstance(snapshot_path, str) or not snapshot_path or not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64:
            writer.reject("structured", ident, "structured_source_snapshot_identity_missing", family=packet.get("family"), group_id=gid)
            continue
        structured_source_refs[snapshot_path].add(snapshot_hash)
        try:
            from sepalith.campaign_protocol import PromptContext, build_training_row
            context = PromptContext.from_mapping(res["context"])
            operation = res.get("operation")
            if operation == "no_op":
                region_new = list(context.region_old)
            elif operation == "delete":
                region_new = []
            elif operation == "replace":
                region_new = list(res.get("target_body", []))
            else:
                raise PreparationError("structured_operation_invalid")
            row = build_training_row(context, operation=operation, region_new=region_new,
                                     tokenizer=ENCODER, row_id=ident, family=rr.get("family", packet.get("family")),
                                     package_id=rr.get("package_id"), split="train")
        except Exception as error:
            writer.reject("structured", ident, f"token_build:{type(error).__name__}:{str(error)[:120]}", family=packet.get("family"), group_id=gid)
            continue
        writer.add(row, source="structured", group_id=gid, source_meta=source_meta_structured(packet), split_status=status)

    # Completion packet source identity is checked against the actual JSONL
    # source rows before strict-v2 repair.  This is two source files here and
    # retains only requested rows, so no target oracle is introduced.
    completion_refs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    completion_ref_status: dict[tuple[str, int], str] = {}
    for _line_no, _raw, packet in iter_jsonl(COMPLETION):
        rr = packet.get("row_ref", {})
        res = packet.get("result", {})
        gid = rr.get("group_id")
        if binding_status(gid, groups, pmap) != "bound_cpt_train" or res.get("status") != "converted":
            continue
        source_file = rr.get("source_file")
        source_line = rr.get("source_line")
        raw_hash = rr.get("raw_line_sha256")
        source_hash = rr.get("source_sha256")
        if not isinstance(source_file, str) or type(source_line) is not int or len(source_file) == 0 or not isinstance(raw_hash, str) or not isinstance(source_hash, str):
            continue
        key = (source_file, source_line)
        previous = completion_refs[source_file].get(source_line)
        value = {"raw_line_sha256": raw_hash, "source_sha256": source_hash}
        if previous is not None and previous != value:
            completion_ref_status[key] = "source_reference_disagreement"
        else:
            completion_refs[source_file][source_line] = value
    raw_by_ref, completion_source_errors = verify_completion_source_refs(completion_refs)
    # The helper's errors are deliberately represented in the report; associate
    # per-file failure with all refs in the second pass below.
    completion_bad_files = {item.get("source_file") for item in completion_source_errors if item.get("reason") in {
        "source_file_missing", "source_file_hash_mismatch", "source_file_mutated_during_read", "source_file_expected_hash_disagreement"
    }}
    source_file_evidence: list[dict[str, Any]] = []
    # Reconstruct concise evidence from the verified refs without a second file read.
    for source_file, by_line in sorted(completion_refs.items()):
        retained = sum((source_file, line) in raw_by_ref for line in by_line)
        if source_file in completion_bad_files:
            observed_hash = next((e.get("observed_sha256") for e in completion_source_errors if e.get("source_file") == source_file), None)
            source_file_evidence.append({"path": source_file, "requested_lines": len(by_line), "retained_rows": retained,
                                         "observed_sha256": observed_hash, "verification": "failed"})
        else:
            # The expected source hash is complete and was checked by the helper.
            source_file_evidence.append({"path": source_file, "requested_lines": len(by_line), "retained_rows": retained,
                                         "expected_sha256": next(iter({v["source_sha256"] for v in by_line.values()})),
                                         "verification": "full_file_hash_and_exact_requested_raw_line_hash"})
    strict_module = load_module("r2_task_mixture_strict_v2", STRICT_REPAIR)
    from sepalith.campaign_protocol import PromptContext, build_training_row, utf16_to_codepoint_column
    for _line_no, _raw, packet in iter_jsonl(COMPLETION):
        rr = packet.get("row_ref", {})
        res = packet.get("result", {})
        ident = rr.get("row_id")
        gid = rr.get("group_id")
        status = binding_status(gid, groups, pmap)
        if status != "bound_cpt_train":
            writer.reject("completion", ident, status, family=packet.get("family"), group_id=gid)
            continue
        if res.get("status") != "converted":
            writer.reject("completion", ident, "packet_status_not_converted", family=packet.get("family"), group_id=gid)
            continue
        key = (rr.get("source_file"), rr.get("source_line"))
        raw = raw_by_ref.get(key)
        if raw is None or rr.get("source_file") in completion_bad_files or key in completion_ref_status:
            writer.reject("completion", ident, "actual_source_ref_not_verified", family=packet.get("family"), group_id=gid)
            continue
        selection = res.get("selection_source", {})
        target_text = "\n".join(res.get("target_body", []))
        raw_prefix = raw.get("prefix")
        raw_target = raw.get("corpus_target")
        # The completion adapter's finish constructor replaces the current
        # final source line.  When corpus_target begins with its framing LF,
        # the packet target starts with that final source line; compare the
        # exact splice rather than falsely requiring those two representations
        # to have identical leading framing.
        current_line = raw_prefix.rsplit("\n", 1)[-1] if isinstance(raw_prefix, str) else ""
        applied_packet = (raw_prefix[:-len(current_line)] + target_text
                          if current_line else raw_prefix + target_text)
        raw_matches = {"prefix": raw_prefix == selection.get("document_text"),
                       "target": isinstance(raw_prefix, str) and isinstance(raw_target, str)
                       and applied_packet == raw_prefix + raw_target}
        if not raw_matches["prefix"] or not raw_matches["target"]:
            writer.reject("completion", ident, "actual_source_prefix_or_target_mismatch", family=packet.get("family"), group_id=gid,
                          detail={"prefix_match": raw_matches["prefix"], "target_match": raw_matches["target"]})
            continue
        try:
            case = {"family": "finish_block", "result": res}
            plan = strict_module.prepare_repair(case, utf16_to_codepoint_column=utf16_to_codepoint_column)
            context = PromptContext.from_mapping(res["context"])
            row = build_training_row(context, operation="replace", region_new=list(plan.repaired_target_lines),
                                     tokenizer=ENCODER, row_id=ident, family="finish_block", package_id=rr.get("package"), split="train")
        except Exception as error:
            writer.reject("completion", ident, f"strict_v2_or_token_build:{type(error).__name__}:{str(error)[:120]}",
                          family=packet.get("family"), group_id=gid)
            continue
        writer.add(row, source="completion", group_id=gid,
                   source_meta=source_meta_completion(packet, raw_matches=raw_matches), split_status=status,
                   repair={"policy": "finish-boundary-repair-v2", "appended_outer_brace": True,
                           "raw_prefix_match": True, "raw_target_match": True,
                           "target_body_sha256_before_brace": sha256_text(target_text),
                           "repaired_target_text_sha256": sha256_text(plan.repaired_target_text)})

    writer.close()
    # Exclusions are written after all source classes, so duplicate/conflict
    # accounting is complete and the file has one stable deterministic order.
    with EXCLUSIONS_OUT.open("xb") as handle:
        for item in exclusions:
            handle.write((canonical(item) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())

    output_hashes = writer.finalize_hashes()
    # Construct a finite, metadata-only sampler manifest.  This is a proposal;
    # the candidate rows remain unadmitted and root must rebind the row hash.
    sampler_module = load_module("r2_task_mixture_sampler", SAMPLER)
    candidate_hash = output_hashes["candidate_rows_sha256"]
    sampler_manifest = sampler_module.build_draw_manifest(
        writer.sampler_rows,
        max_steps=1000,
        effective_batch=16,
        split_id=split_meta.get("dat02_split_id"),
        seed=3407,
        token_rows_sha256=candidate_hash,
        requested_draws=16000,
        noop_fraction=0.10,
        family_ceiling=0.25,
        small_pack_cap=3,
        ordinary_replay_cap=8,
        naturally_long_fraction=0.20,
        short_max_tokens=2048,
        long_max_tokens=4096,
    )
    sampler_module.validate_draw_manifest(sampler_manifest)
    SCHEDULE_OUT.write_text(json.dumps(sampler_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    # Source manifest records exact expected hashes even where the old
    # structured packet's normalized snapshot path is no longer mounted.
    structured_source_manifest: list[dict[str, Any]] = []
    for path, hashes in sorted(structured_source_refs.items()):
        structured_source_manifest.append({"path": path, "expected_sha256": sorted(hashes),
                                           "local_path_present": Path(path).is_file(),
                                           "verification_basis": "structured packet/report source checks; this bounded pass did not sweep raw sources"})
    exclusion_hash, exclusion_bytes = sha256_file(EXCLUSIONS_OUT)
    schedule_hash, schedule_bytes = sha256_file(SCHEDULE_OUT)
    report = {
        "schema": "dat10.r2.task_mixture_preparation.v1",
        "status": "candidate_prepared_not_training_admitted",
        "prepared_at_unix": time.time(),
        "scope": "target-only PRM03 SFT candidate rows; no CUDA/model/framework launch",
        "inputs": input_evidence,
        "split_binding": split_meta,
        "split_binding_policy": "exact group_id joins only; group absent from the explicit CPT partition is excluded",
        "counts": {
            "input_rows": dict(sorted((key, value) for key, value in writer.counts.items() if key.startswith("input:"))),
            "accepted_rows": writer.row_count,
            "accepted_provenance_rows": writer.provenance_count,
            "exclusions": len(exclusions),
            "accepted_by_source": dict(sorted((key.split(":", 1)[1], value) for key, value in writer.counts.items() if key.startswith("accepted:"))),
            "accepted_by_family": dict(sorted(writer.family_counts.items())),
            "accepted_by_operation": dict(sorted(writer.operation_counts.items())),
            "accepted_by_source_class": dict(sorted(writer.source_counts.items())),
            "no_op_rows": writer.noop_rows,
            "no_op_row_fraction": writer.noop_rows / writer.row_count if writer.row_count else 0.0,
            "reserved_cpt_validation_groups": len(reserved),
        },
        "tokens": {
            "candidate_prompt_tokens_target_start_sum": writer.prompt_tokens,
            "candidate_target_label_tokens_including_protocol_EOS": writer.target_tokens,
            "candidate_total_sequence_tokens": writer.total_tokens,
            "no_op_target_label_tokens_including_protocol_EOS": writer.noop_target_tokens,
            "no_op_target_token_fraction": writer.noop_target_tokens / writer.target_tokens if writer.target_tokens else 0.0,
            "target_cap_including_protocol_EOS": 192,
            "sequence_cap": 4096,
        },
        "output": output_hashes | {"exclusions_sha256": exclusion_hash, "exclusions_bytes": exclusion_bytes,
                                   "schedule_sha256": schedule_hash, "schedule_bytes": schedule_bytes},
        "prompt_deduplication": {
            "policy": "first deterministic source-class order corrected, short, structured, completion; exact prompt duplicates dropped; conflicting later target dropped",
            "conflicting_prompt_exclusions": len(writer.conflicts),
            "conflicts": writer.conflicts,
            "output_has_at_most_one_target_per_prompt": True,
        },
        "completion_strict_v2_repair": {
            "policy_source": {"path": str(STRICT_REPAIR), "sha256": EXPECTED_HASHES[STRICT_REPAIR] if STRICT_REPAIR in EXPECTED_HASHES else sha256_file(STRICT_REPAIR)[0]},
            "source_files": source_file_evidence,
            "source_errors": completion_source_errors,
            "repair": "append exactly one ASCII } only after strict EOF geometry/provenance checks; raw source prefix matched exactly and the packet target matched the exact constructor splice to corpus_target",
            "no_observed_editor_trace_claim": True,
        },
        "structured_source_evidence": {
            "unique_snapshot_paths": len(structured_source_manifest),
            "local_snapshot_paths_present": sum(item["local_path_present"] for item in structured_source_manifest),
            "local_snapshot_paths_missing": sum(not item["local_path_present"] for item in structured_source_manifest),
            "expected_hash_conflict_paths": sum(len(item["expected_sha256"]) > 1 for item in structured_source_manifest),
            "manifest": structured_source_manifest,
            "basis": "existing structured report and packet flags; no broad raw-source sweep in this task",
        },
        "sampler_proposal": {
            "path": str(SCHEDULE_OUT),
            "sha256": schedule_hash,
            "status": sampler_manifest.get("status"),
            "max_steps": sampler_manifest.get("max_steps"),
            "effective_batch": sampler_manifest.get("effective_batch"),
            "requested_draws": sampler_manifest.get("policy", {}).get("requested_draws"),
            "draw_count": sampler_manifest.get("draw_count"),
            "semantic_noop": sampler_manifest.get("achieved_mixture", {}).get("semantic_noop"),
            "families": sampler_manifest.get("achieved_mixture", {}).get("families"),
            "length": sampler_manifest.get("achieved_mixture", {}).get("length"),
            "target_tokens": sampler_manifest.get("exposure", {}).get("target_tokens"),
            "no_op_target_token_fraction": (sum(draw["target_tokens"] for draw in sampler_manifest.get("draws", []) if draw.get("semantic_noop")) /
                                             sampler_manifest.get("exposure", {}).get("target_tokens", 1)),
            "policy": "existing finite sampler; no new loss weighting",
        },
        "weighting_assessment": {
            "observation": "short no-op labels are much shorter than edit labels, so their row share overstates their target-token share under target-only per-token loss",
            "candidate_no_op_row_fraction": writer.noop_rows / writer.row_count if writer.row_count else 0.0,
            "candidate_no_op_target_token_fraction": writer.noop_target_tokens / writer.target_tokens if writer.target_tokens else 0.0,
            "recommended_minimal_adjustment": "retain the existing finite sampler's 10% no-op row reservation and 25% family ceiling; if no-op behavior needs a sensitivity run, change only noop_fraction to 0.15 and keep per-token loss unchanged",
            "no_new_loss_framework": True,
        },
        "limitations": [
            "Rows are source-derived/synthetic candidates or inherited corrected rows, not observed editor trajectories.",
            "Completion outer-brace repair is accepted only for strict EOF-bound cases and does not establish user intent or model quality.",
            "Structured source snapshot hashes and application flags are inherited from the accepted packet/report; this bounded pass does not sweep unavailable normalized files.",
            "Rows remain unadmitted until root binds exact candidate hashes, parent/model, source closure, schedule, and DEV/final evaluation.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    SOURCE_MANIFEST_OUT.write_text(json.dumps({
        "schema": "dat10.r2.task_mixture_source_manifest.v1",
        "inputs": input_evidence,
        "dat02": split_meta,
        "completion_actual_source_files": source_file_evidence,
        "structured_expected_source_snapshots": structured_source_manifest,
        "output": output_hashes | {"exclusions_sha256": exclusion_hash, "schedule_sha256": schedule_hash},
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "rows": writer.row_count,
        "families": dict(sorted(writer.family_counts.items())),
        "noops": writer.noop_rows,
        "target_tokens": writer.target_tokens,
        "no_op_target_fraction": writer.noop_target_tokens / writer.target_tokens if writer.target_tokens else 0.0,
        "schedule_status": sampler_manifest.get("status"),
        "schedule_draws": sampler_manifest.get("draw_count"),
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


# Pinned protocol/tokenizer objects are loaded only after input pins and split
# documents are checked.  The module remains importable for CPU unit tests.
sys.path.insert(0, str(EXEC / "packages/sepalith/src"))
from sepalith.campaign_protocol import (  # noqa: E402
    PromptContext,
    build_training_row,
    utf16_to_codepoint_column,
)
from tokenizers import Tokenizer  # noqa: E402

_backend = Tokenizer.from_file(str(TOKENIZER))
_backend.encode_special_tokens = True


class _Encoder:
    def encode(self, text: str, *, add_special_tokens: bool = False, split_special_tokens: bool = True) -> list[int]:
        if add_special_tokens is not False or split_special_tokens is not True:
            raise AssertionError("pinned encoder requires no added specials and split_special_tokens=True")
        return _backend.encode(text, add_special_tokens=False).ids


ENCODER = _Encoder()

if __name__ == "__main__":
    raise SystemExit(main())
