#!/usr/bin/env python3
"""Materialize the v2 roxygen review union as token-only candidate rows.

The reviewed context packets already contain the native PRM03 token rows.  This
pass selects the exact 8,597-ID v2 union, validates token geometry and source
metadata, and writes one E-backed candidate stream.  It never serializes
source/prompt/target text and does not make a training-admission decision.
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-roxy8597-materialization-v1"
OUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-roxy8597-materialization-v1")

V2_DIR = PLAN / "docs/campaign/work/lead/r2-roxy-recovery-union-review-v2"
V2_SAFE = V2_DIR / "safe-review-union-ids.json"
V2_HELD = V2_DIR / "held-review-ids.json"
V2_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-roxy-recovery-union-review-v2.json"
V2_ORIGINAL = PLAN / "docs/campaign/work/lead/r2-roxy-context-admission-review-v1/full/recommended-context-ids.json"
V2_RECOVERY = PLAN / "docs/campaign/work/lead/r2-roxy-full-context-recovery-v1/materialization-v4/recoverable-ids.json"
V2_LOOP = V2_DIR / "corrected-loop-recoverable-ids.json"
V2_NSE = Path("/mnt/e/sepalith/campaign-20260915/data-work/Roxy-NSE-semantic-review-v1/recoverable-ids.json")

SELECTED_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1/selected-context-token-rows.jsonl")
SELECTED_PROFILES = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1/selected-context-profiles.jsonl")
FULL_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-full-file-token-rows.jsonl")
ACCEPTED_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl")

TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
RENDERER = "zeta2-prm03-v1"
TOKENIZATION_POLICY = "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1"
SCENARIO_SOURCE = "/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl"

FORBIDDEN_PAYLOAD_KEYS = {
    "prompt",
    "prompt_text",
    "target",
    "target_text",
    "target_body_text",
    "source_text",
    "source_snapshot_text",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def token_fingerprint(token_row: dict[str, Any]) -> str:
    """Fingerprint exact native prompt/target token payload, including operation."""
    return sha256_bytes(canonical({
        "input_ids": token_row["input_ids"],
        "target_body_tokens": token_row["target_body_tokens"],
        "target_terminal_tokens": token_row["target_terminal_tokens"],
        "target_operation": token_row["target_operation"],
        "target_start": token_row["target_start"],
    }))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


class AtomicJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        self.stream = self.temporary.open("wb")
        self.digest = hashlib.sha256()
        self.rows = 0

    def write(self, value: dict[str, Any]) -> None:
        payload = canonical(value) + b"\n"
        self.stream.write(payload)
        self.digest.update(payload)
        self.rows += 1

    def close(self) -> str:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.temporary.replace(self.path)
        return self.digest.hexdigest()


def set_from_id_packet(path: Path, key: str) -> set[str]:
    packet = read_json(path)
    values = packet.get(key)
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ValueError(f"invalid_id_packet:{path}")
    declared = packet.get("count", packet.get("recoverable_count", packet.get("accepted_recommendation_count")))
    if declared is not None and declared != len(values):
        raise ValueError(f"id_count_mismatch:{path}:{declared}:{len(values)}")
    return set(values)


def forbidden_key_found(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PAYLOAD_KEYS:
                return key
            found = forbidden_key_found(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = forbidden_key_found(child)
            if found:
                return found
    return None


def validate_token_row(token_row: dict[str, Any]) -> dict[str, int]:
    required = {
        "bos_token_id", "eos_token_id", "family", "id", "input_ids", "package_id",
        "prompt_token_count", "renderer_id", "split", "target_body_token_count",
        "target_body_tokens", "target_operation", "target_start", "target_terminal_token_count",
        "target_terminal_tokens", "target_token_count", "tokenization_policy",
        "tokenizer_json_sha256", "tokenizer_revision",
    }
    missing = required - set(token_row)
    if missing:
        raise ValueError("token_row_missing:" + ",".join(sorted(missing)))
    if token_row["family"] != "roxygen_drafting" or token_row["split"] != "train":
        raise ValueError("token_row_not_train_roxygen")
    if token_row["renderer_id"] != RENDERER or token_row["tokenizer_json_sha256"] != TOKENIZER_SHA:
        raise ValueError("tokenizer_or_renderer_mismatch")
    if token_row["tokenizer_revision"] != TOKENIZER_REVISION or token_row["tokenization_policy"] != TOKENIZATION_POLICY:
        raise ValueError("tokenization_policy_mismatch")
    if token_row["target_operation"] != "replace":
        raise ValueError("unexpected_target_operation:" + str(token_row["target_operation"]))
    input_ids = token_row["input_ids"]
    body = token_row["target_body_tokens"]
    terminal = token_row["target_terminal_tokens"]
    if not isinstance(input_ids, list) or not isinstance(body, list) or not isinstance(terminal, list):
        raise ValueError("token_arrays_not_lists")
    prompt_count = int(token_row["prompt_token_count"])
    target_start = int(token_row["target_start"])
    body_count = int(token_row["target_body_token_count"])
    terminal_count = int(token_row["target_terminal_token_count"])
    target_count = int(token_row["target_token_count"])
    if target_start != prompt_count + 1:
        raise ValueError("target_start_geometry_mismatch")
    if body_count != len(body) or terminal_count != len(terminal) or target_count != body_count + terminal_count:
        raise ValueError("target_count_geometry_mismatch")
    if len(input_ids) != target_start + target_count + 1:
        raise ValueError("input_length_geometry_mismatch")
    if not input_ids or input_ids[0] != token_row["bos_token_id"] or input_ids[-1] != token_row["eos_token_id"]:
        raise ValueError("special_token_geometry_mismatch")
    if input_ids[target_start:target_start + body_count] != body:
        raise ValueError("target_body_not_conserved")
    if input_ids[target_start + body_count:target_start + target_count] != terminal:
        raise ValueError("target_terminal_not_conserved")
    return {
        "sequence_tokens": len(input_ids),
        "prompt_tokens": prompt_count,
        "target_body_tokens": body_count,
        "target_tokens": target_count,
    }


def stream_accepted() -> tuple[dict[str, str], dict[str, list[str]], dict[str, Any]]:
    by_id: dict[str, str] = {}
    by_fingerprint: dict[str, list[str]] = collections.defaultdict(list)
    digest = hashlib.sha256()
    rows = 0
    with ACCEPTED_ROWS.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item["id"]
            if row_id in by_id:
                raise ValueError("accepted_duplicate_id:" + row_id)
            fingerprint = token_fingerprint(item)
            by_id[row_id] = fingerprint
            by_fingerprint[fingerprint].append(row_id)
            rows += 1
    if rows != 15006:
        raise ValueError(f"accepted_denominator_changed:{rows}")
    return by_id, dict(by_fingerprint), {"rows": rows, "bytes": ACCEPTED_ROWS.stat().st_size, "sha256": digest.hexdigest()}


def profile_map_for(ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    rows = 0
    with SELECTED_PROFILES.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item["row_id"]
            rows += 1
            if row_id in ids:
                if row_id in profiles:
                    raise ValueError("selected_profile_duplicate:" + row_id)
                profiles[row_id] = item
    if profiles.keys() != ids:
        raise ValueError(f"selected_profile_scope_incomplete:{len(profiles)}:{len(ids)}")
    return profiles, {"rows": rows, "bytes": SELECTED_PROFILES.stat().st_size, "sha256": digest.hexdigest()}


def provenance_from(item: dict[str, Any], source_kind: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    context = item.get("context_evidence") or item.get("source_context") or {}
    target_hash = item.get("target_body_sha256")
    target_semantics = item.get("target_semantics")
    target_rewritten = item.get("target_rewritten")
    if profile is not None:
        target_hash = profile.get("target_body_sha256", target_hash)
        target_semantics = profile.get("target_semantics", target_semantics)
        target_rewritten = profile.get("target_rewritten", target_rewritten)
    provenance = {
        "source_kind": source_kind,
        "admission": item.get("admission"),
        "family": item.get("family"),
        "package_id": item.get("package_id"),
        "group_id": item.get("group_id"),
        "source_file": item.get("source_file"),
        "source_line": item.get("source_line"),
        "source_path": item.get("source_path"),
        "raw_line_sha256": item.get("raw_line_sha256"),
        "normalized_after_source_sha256": item.get("normalized_after_source_sha256"),
        "derived_before_sha256": item.get("derived_before_sha256"),
        "target_body_sha256": target_hash,
        "target_semantics": target_semantics,
        "target_rewritten": target_rewritten,
        "source_text_written": item.get("source_text_written"),
        "target_text_written": item.get("target_text_written"),
        "token_ids_written": item.get("token_ids_written"),
        "license_revalidated_by_frozen_review": item.get("license_revalidated_by_frozen_review"),
        "license_review_reason": item.get("license_review_reason"),
        "context_policy_id": item.get("context_policy_id"),
        "semantic_support_claim": item.get("semantic_support_claim"),
        "original_review_reasons": item.get("original_review_reasons"),
        "context_evidence": context,
    }
    found = forbidden_key_found(provenance)
    if found:
        raise ValueError("payload_text_key_present:" + found)
    if provenance["source_file"] != SCENARIO_SOURCE:
        raise ValueError("non_train_scenario_source")
    if not str(provenance["source_path"] or "").startswith("/mnt/h/sepalith/normalized/"):
        raise ValueError("non_normalized_source_path")
    if provenance["source_text_written"] is not False or provenance["target_text_written"] is not False:
        raise ValueError("source_or_target_text_written")
    return provenance


def update_status(status: str, **fields: Any) -> None:
    value = {
        "schema": "sepalith.dat10.roxy8597_materialization_status.v1",
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "cuda": False,
        "training_admission": False,
        "source_text_written": False,
        "target_text_written": False,
    }
    value.update(fields)
    write_json(PACKET / "status.json", value)


def main() -> None:
    PACKET.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    required = [V2_SAFE, V2_HELD, V2_RECEIPT, V2_ORIGINAL, V2_RECOVERY, V2_LOOP, V2_NSE, SELECTED_ROWS, SELECTED_PROFILES, FULL_ROWS, ACCEPTED_ROWS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing_input:" + ",".join(missing))
    update_status("loading_v2_union_and_accepted_dedup")

    safe_ids = set_from_id_packet(V2_SAFE, "ids")
    held_packet = read_json(V2_HELD)
    held_values = held_packet.get("ids")
    if not isinstance(held_values, list) or len(held_values) != len(set(held_values)):
        raise ValueError("invalid_held_id_packet")
    held_ids = set(held_values)
    if held_packet.get("held_count") != len(held_ids) or len(held_ids) != 1420:
        raise ValueError(f"held_id_count_mismatch:{len(held_ids)}")
    if safe_ids & held_ids:
        raise ValueError("safe_held_overlap")
    original_ids = set_from_id_packet(V2_ORIGINAL, "accepted_recommendation_ids")
    recovery_ids = set_from_id_packet(V2_RECOVERY, "recoverable_ids")
    loop_ids = set_from_id_packet(V2_LOOP, "ids")
    nse_ids = set_from_id_packet(V2_NSE, "ids")
    source_sets = {
        "original_context_recommendation": original_ids,
        "full_context_recovery": recovery_ids,
        "corrected_loop_scope_recovery": loop_ids,
        "nse_semantic_recovery": nse_ids,
    }
    if len(safe_ids) != 8597 or set().union(*source_sets.values()) != safe_ids:
        raise ValueError("v2_union_scope_mismatch")
    if any(len(left & right) for left_name, left in source_sets.items() for right_name, right in source_sets.items() if left_name < right_name):
        raise ValueError("v2_source_set_overlap")
    source_by_id = {row_id: source for source, ids in source_sets.items() for row_id in ids}
    accepted_by_id, accepted_by_fingerprint, accepted_pin = stream_accepted()
    selected_profiles, selected_profiles_pin = profile_map_for(original_ids)

    manifest: dict[str, Any] = {
        "schema": "sepalith.dat10.roxy8597_materialization_manifest.v1",
        "status": "complete_review_only_root_admission_required",
        "training_admission": False,
        "source_text_written": False,
        "target_text_written": False,
        "cuda": False,
        "source_sets": {name: {"count": len(ids), "path": str(path), "sha256": sha256_file(path)} for name, ids, path in [
            ("original_context_recommendation", original_ids, V2_ORIGINAL),
            ("full_context_recovery", recovery_ids, V2_RECOVERY),
            ("corrected_loop_scope_recovery", loop_ids, V2_LOOP),
            ("nse_semantic_recovery", nse_ids, V2_NSE),
        ]},
        "v2_safe_union": {"count": len(safe_ids), "path": str(V2_SAFE), "sha256": sha256_file(V2_SAFE)},
        "v2_held_queue": {
            "count": len(held_ids),
            "path": str(V2_HELD),
            "sha256": sha256_file(V2_HELD),
            "reason_counts": held_packet.get("reason_counts"),
            "materialized": False,
        },
        "v2_review_receipt": {"path": str(V2_RECEIPT), "sha256": sha256_file(V2_RECEIPT)},
        "accepted_dedup_input": accepted_pin,
        "selected_profiles_input": {"path": str(SELECTED_PROFILES), **selected_profiles_pin},
        "policy": {
            "tokenizer_sha256": TOKENIZER_SHA,
            "tokenizer_revision": TOKENIZER_REVISION,
            "renderer_id": RENDERER,
            "tokenization_policy": TOKENIZATION_POLICY,
            "split": "train",
            "family": "roxygen_drafting",
            "target_truncation": False,
            "all_union_rows_retained": True,
            "long_rows_filtered": False,
            "holds_retained_separately": True,
            "payload_text_written": False,
        },
    }
    union_path = OUT / "candidate-union-token-rows.jsonl"
    ledger_path = OUT / "dedup-ledger.jsonl"
    union_writer = AtomicJsonl(union_path)
    ledger_writer = AtomicJsonl(ledger_path)
    seen_ids: set[str] = set()
    seen_fingerprints: dict[str, list[str]] = collections.defaultdict(list)
    source_counts = collections.Counter()
    dedup_counts = collections.Counter()
    dedup_by_id: dict[str, str] = {}
    length_counts = collections.Counter()
    lengths: dict[str, int] = {"sequence_min": 10**18, "sequence_max": 0, "prompt_min": 10**18, "prompt_max": 0, "target_body_min": 10**18, "target_body_max": 0}
    source_artifact_hashes: dict[str, str] = {}
    source_artifact_rows: dict[str, int] = {}
    selected_seen: set[str] = set()
    full_seen: set[str] = set()

    def emit(item: dict[str, Any], source_kind: str, source_artifact: Path, profile: dict[str, Any] | None) -> None:
        row_id = item["row_id"]
        if row_id in seen_ids:
            raise ValueError("candidate_duplicate_id:" + row_id)
        if row_id not in safe_ids or source_by_id[row_id] != source_kind:
            raise ValueError("candidate_source_scope_mismatch:" + row_id)
        token_row = item.get("token_row")
        if not isinstance(token_row, dict) or token_row.get("id") != row_id:
            raise ValueError("token_row_id_mismatch:" + row_id)
        geometry = validate_token_row(token_row)
        provenance = provenance_from(item, source_kind, profile)
        fingerprint = token_fingerprint(token_row)
        accepted_match_ids = accepted_by_fingerprint.get(fingerprint, [])
        if row_id in accepted_by_id:
            if accepted_by_id[row_id] == fingerprint:
                dedup_status = "accepted_id_exact_duplicate"
            else:
                dedup_status = "accepted_id_payload_conflict"
        elif accepted_match_ids:
            dedup_status = "accepted_prompt_target_exact_duplicate"
        elif seen_fingerprints.get(fingerprint):
            dedup_status = "union_internal_exact_duplicate"
        else:
            dedup_status = "new_candidate"
        seen_ids.add(row_id)
        seen_fingerprints[fingerprint].append(row_id)
        dedup_by_id[row_id] = dedup_status
        source_counts[source_kind] += 1
        dedup_counts[dedup_status] += 1
        seq = geometry["sequence_tokens"]
        prompt = geometry["prompt_tokens"]
        target_body = geometry["target_body_tokens"]
        lengths["sequence_min"] = min(lengths["sequence_min"], seq)
        lengths["sequence_max"] = max(lengths["sequence_max"], seq)
        lengths["prompt_min"] = min(lengths["prompt_min"], prompt)
        lengths["prompt_max"] = max(lengths["prompt_max"], prompt)
        lengths["target_body_min"] = min(lengths["target_body_min"], target_body)
        lengths["target_body_max"] = max(lengths["target_body_max"], target_body)
        length_counts["sequence_le_192" if seq <= 192 else "sequence_193_512" if seq <= 512 else "sequence_513_1024" if seq <= 1024 else "sequence_1025_4096" if seq <= 4096 else "sequence_gt_4096"] += 1
        length_counts["target_body_le_192" if target_body <= 192 else "target_body_193_512" if target_body <= 512 else "target_body_513_1024" if target_body <= 1024 else "target_body_gt_1024"] += 1
        length_counts["sequence_gt_8192"] += seq > 8192
        length_counts["sequence_gt_16384"] += seq > 16384
        length_counts["sequence_gt_32768"] += seq > 32768
        length_counts["sequence_gt_65536"] += seq > 65536
        length_counts["sequence_gt_131072"] += seq > 131072
        output = {
            "schema": "sepalith.dat10.roxy8597_candidate_token_row.v1",
            "row_id": row_id,
            "union_source": source_kind,
            "source_artifact": str(source_artifact),
            "dedup_status": dedup_status,
            "accepted_matching_ids": accepted_match_ids,
            "token_payload_sha256": fingerprint,
            "geometry": geometry,
            "provenance": provenance,
            "token_row": token_row,
        }
        if forbidden_key_found(output):
            raise ValueError("output_payload_text_key_present:" + row_id)
        union_writer.write(output)
        ledger_writer.write({
            "schema": "sepalith.dat10.roxy8597_dedup_ledger_row.v1",
            "row_id": row_id,
            "union_source": source_kind,
            "dedup_status": dedup_status,
            "token_payload_sha256": fingerprint,
            "accepted_matching_ids": accepted_match_ids,
            "sequence_tokens": seq,
            "prompt_tokens": prompt,
            "target_body_tokens": target_body,
            "target_tokens": geometry["target_tokens"],
            "source_path": provenance["source_path"],
            "normalized_after_source_sha256": provenance["normalized_after_source_sha256"],
            "target_body_sha256": provenance["target_body_sha256"],
            "source_text_written": False,
            "target_text_written": False,
        })

    def stream_candidates(path: Path, wanted: set[str], source_kind: str, profile_lookup: dict[str, dict[str, Any]] | None) -> None:
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                item = json.loads(line)
                rows += 1
                row_id = item["row_id"]
                if row_id not in wanted:
                    continue
                if source_kind == "original_context_recommendation":
                    selected_seen.add(row_id)
                else:
                    full_seen.add(row_id)
                emit(item, source_kind, path, profile_lookup.get(row_id) if profile_lookup is not None else None)
        source_artifact_hashes[str(path)] = digest.hexdigest()
        source_artifact_rows[str(path)] = rows

    def stream_full_candidates(path: Path, wanted: set[str], source_by_id: dict[str, str]) -> None:
        """Read the shared repaired stream once, retaining each row's exact v2 source kind."""
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                item = json.loads(line)
                rows += 1
                row_id = item["row_id"]
                if row_id not in wanted:
                    continue
                full_seen.add(row_id)
                emit(item, source_by_id[row_id], path, None)
        source_artifact_hashes[str(path)] = digest.hexdigest()
        source_artifact_rows[str(path)] = rows

    update_status("materializing_union_rows", expected_rows=len(safe_ids), accepted_rows=len(accepted_by_id))
    stream_candidates(SELECTED_ROWS, original_ids, "original_context_recommendation", selected_profiles)
    stream_full_candidates(FULL_ROWS, recovery_ids | loop_ids | nse_ids, source_by_id)

    if selected_seen != original_ids:
        raise ValueError(f"selected_scope_incomplete:{len(selected_seen)}:{len(original_ids)}")
    if full_seen != recovery_ids | loop_ids | nse_ids:
        raise ValueError(f"full_scope_incomplete:{len(full_seen)}:{len(recovery_ids | loop_ids | nse_ids)}")
    if seen_ids != safe_ids:
        raise ValueError(f"union_scope_incomplete:{len(seen_ids)}:{len(safe_ids)}")

    union_hash = union_writer.close()
    ledger_hash = ledger_writer.close()
    if lengths["sequence_min"] == 10**18:
        raise ValueError("empty_union")
    lengths["sequence_min"] = int(lengths["sequence_min"])
    lengths["prompt_min"] = int(lengths["prompt_min"])
    lengths["target_body_min"] = int(lengths["target_body_min"])
    manifest.update({
        "counts": {
            "union_rows": len(seen_ids),
            "source_counts": dict(sorted(source_counts.items())),
            "dedup_counts": dict(sorted(dedup_counts.items())),
            "accepted_id_overlap": sum(value for key, value in dedup_counts.items() if key.startswith("accepted_id_")),
            "accepted_exact_prompt_target_overlap": dedup_counts["accepted_prompt_target_exact_duplicate"],
            "new_candidate_rows": dedup_counts["new_candidate"],
            "conflict_rows": dedup_counts["accepted_id_payload_conflict"],
            "internal_exact_duplicate_rows": dedup_counts["union_internal_exact_duplicate"],
            "validated_geometry_rows": len(seen_ids),
            "complete_target_rows": len(seen_ids),
            "target_truncation_detected": 0,
        },
        "lengths": lengths,
        "length_buckets": dict(sorted(length_counts.items())),
        "source_artifacts": {
            path: {"rows": source_artifact_rows[path], "sha256": source_artifact_hashes[path]}
            for path in sorted(source_artifact_hashes)
        },
        "outputs": {
            "candidate_union": {"path": str(union_path), "rows": len(seen_ids), "sha256": union_hash, "bytes": union_path.stat().st_size},
            "dedup_ledger": {"path": str(ledger_path), "rows": len(seen_ids), "sha256": ledger_hash, "bytes": ledger_path.stat().st_size},
        },
    })
    write_json(OUT / "manifest.json", manifest)
    write_json(OUT / "new-candidate-ids.json", {
        "schema": "sepalith.dat10.roxy8597_new_candidate_ids.v1",
        "status": "review_only_not_training_admission",
        "count": dedup_counts["new_candidate"],
        "ids": sorted(row_id for row_id, status in dedup_by_id.items() if status == "new_candidate"),
    })
    update_status("complete_review_only_root_admission_required", **{
        "union_rows": len(seen_ids),
        "new_candidate_rows": dedup_counts["new_candidate"],
        "conflict_rows": dedup_counts["accepted_id_payload_conflict"],
        "candidate_union_path": str(union_path),
        "manifest_path": str(OUT / "manifest.json"),
    })
    print(json.dumps({
        "union_rows": len(seen_ids),
        "source_counts": dict(source_counts),
        "dedup_counts": dict(dedup_counts),
        "sequence": [lengths["sequence_min"], lengths["sequence_max"]],
        "target_body": [lengths["target_body_min"], lengths["target_body_max"]],
        "candidate_union_sha256": union_hash,
        "dedup_ledger_sha256": ledger_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            update_status("failed", error=str(error), traceback=traceback.format_exc(limit=5))
        finally:
            raise
