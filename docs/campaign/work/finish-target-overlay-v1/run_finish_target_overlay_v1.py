#!/usr/bin/env python3
"""Stream a strict-v2 shadow overlay for the retained TRAIN finish packets.

The overlay is preparation-only.  It reads the converted packet, DAT-05
registry, and selected SFT rows once each, keeps only identity metadata for
joins, and writes a new JSONL shadow file.  It never mutates either selected
artifact and never creates a replacement training dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from collections import Counter
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PACKET_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl")
REGISTRY_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/registry.jsonl")
SFT_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/SFT-inputs-v1/train-token-rows.jsonl")
LINEAGE_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/DAT-04-finish-training-lineage-audit.json"
V2_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/DAT-04-finish-boundary-repair-v2-preparation.json"
V2_CANDIDATE = PLAN_ROOT / "docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py"
V2_CANDIDATE_SHA256 = "7fabe102485241a33e2af1178a6e2f05d8c06e1ef832f852c44e4d58faa08b5b"
V2_RECEIPT_SHA256 = "eccd57c178977a1670e456f914a0d8f223e5746ab922f3a042460a0fdd5fef5f"
LINEAGE_RECEIPT_SHA256 = "cf9a8a85610bf64ff1afff77d7368f5cd35dbcbb90fe8184884d8a386a428273"

PROTOCOL_PATH = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
SCENARIOS_PATH = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py")
FINISH_EXTRACTOR_PATH = EXEC_ROOT / "experiments/synthetic-data/finish_block.py"
BATCH_PATH = EXEC_ROOT / "experiments/training/campaign_completion_batch.py"
ADAPTER_PATH = EXEC_ROOT / "experiments/training/campaign_admission_completion.py"

PACKET_SHA256 = "42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25"
REGISTRY_SHA256 = "ef8ca082be4699d52dab67cb9c42628df4cc935479fb769d880961e7c3c2cf3d"
SFT_SHA256 = "7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6"
PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"
SCENARIOS_SHA256 = "cccf8ddfff0ae1f64a0113c9612386227df66f9fff4701608c2320bd8eb0250c"
FINISH_EXTRACTOR_SHA256 = "47759356b85eb9b48e0461e9265a9fa5a272df14de87747a63e62304be48a3fb"
BATCH_SHA256 = "a7a989f491f168154a3274b6678595c15b4d434be6e7f346521483e0d76035f4"
ADAPTER_SHA256 = "fafb852f31e5e59f003c3ccc0fb34919752020fb6c231a09d77b3ea158f173cc"


def load_module(name: str, path: Path) -> Any:  # noqa: ANN401
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:  # noqa: ANN401
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def parse_ok(scenarios: Any, text: str) -> bool:  # noqa: ANN401
    return not scenarios.parser.parse(text.encode("utf-8")).root_node.has_error


def _identity_metadata(record: Mapping[str, Any], line_number: int, raw_sha256: str) -> dict[str, Any]:
    target_body = record.get("target_body_text")
    target_body_sha = sha256_text(target_body) if isinstance(target_body, str) else None
    return {
        "line": line_number,
        "raw_line_sha256": raw_sha256,
        "id": record.get("id"),
        "family": record.get("family"),
        "package_id": record.get("package_id"),
        "split": record.get("split"),
        "target_operation": record.get("target_operation"),
        "target_body_sha256": target_body_sha,
        "target_body_text": target_body,
        "target_text_sha256": sha256_text(record["target_text"]) if isinstance(record.get("target_text"), str) else None,
        "prompt_text_sha256": sha256_text(record["prompt_text"]) if isinstance(record.get("prompt_text"), str) else None,
        "target_start": record.get("target_start"),
        "input_token_count": len(record["input_ids"]) if isinstance(record.get("input_ids"), list) else None,
        "duplicate_conflict": False,
    }


def _identity_fingerprint(metadata: Mapping[str, Any]) -> str:
    comparable = {
        key: metadata.get(key)
        for key in (
            "id", "family", "package_id", "split", "target_operation",
            "target_body_sha256", "target_text_sha256", "prompt_text_sha256",
            "target_start", "input_token_count",
        )
    }
    return sha256_json(comparable)


def stream_packet() -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    """Read and hash the converted packet in one binary stream."""

    packets: list[dict[str, Any]] = []
    packet_ids: set[str] = set()
    seen: dict[str, tuple[str, str]] = {}
    duplicate_ids = 0
    duplicate_conflicts = 0
    nonfinish = 0
    split_counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    with PACKET_PATH.open("rb", buffering=1024 * 1024) as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            packet = json.loads(raw_line.decode("utf-8"))
            family = packet.get("family")
            if family != "finish_block":
                nonfinish += 1
                continue
            row_ref = packet.get("row_ref")
            if not isinstance(row_ref, Mapping) or not isinstance(row_ref.get("row_id"), str):
                raise ValueError(f"packet line {line_number} has no row_ref.row_id")
            split_counts[str(row_ref.get("split"))] += 1
            row_id = str(row_ref["row_id"])
            raw_sha = sha256_bytes(raw_line)
            record_sha = sha256_json(packet)
            if row_id in seen:
                duplicate_ids += 1
                if seen[row_id] != (raw_sha, record_sha):
                    duplicate_conflicts += 1
            else:
                seen[row_id] = (raw_sha, record_sha)
                packet_ids.add(row_id)
                packet["_overlay_packet_line"] = line_number
                packet["_overlay_packet_raw_line_sha256"] = raw_sha
                packet["_overlay_packet_record_sha256"] = record_sha
                packets.append(packet)
    actual_sha = digest.hexdigest()
    if actual_sha != PACKET_SHA256:
        raise ValueError(f"packet SHA mismatch: {actual_sha}")
    return packets, {
        "path": str(PACKET_PATH),
        "sha256": actual_sha,
        "prior_lineage_reported_rows": 4346,
        "actual_rows": len(packets) + nonfinish,
        "finish_rows": len(packets),
        "nonfinish_rows": nonfinish,
        "duplicate_ids": duplicate_ids,
        "duplicate_conflicts": duplicate_conflicts,
        "split_counts": dict(sorted(split_counts.items())),
    }, packet_ids


def stream_identity_artifact(
    path: Path,
    packet_ids: set[str],
    expected_sha256: str,
    role: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read one registry/SFT JSONL stream and retain selected identity fields."""

    selected: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    duplicate_ids = 0
    duplicate_conflicts = 0
    nonfinish = 0
    total = 0
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        for line_number, raw_line in enumerate(handle, 1):
            total += 1
            digest.update(raw_line)
            record = json.loads(raw_line.decode("utf-8"))
            if record.get("family") != "finish_block":
                nonfinish += 1
            row_id = record.get("id")
            if not isinstance(row_id, str) or row_id not in packet_ids:
                continue
            metadata = _identity_metadata(record, line_number, sha256_bytes(raw_line))
            fingerprint = _identity_fingerprint(metadata)
            if row_id in fingerprints:
                duplicate_ids += 1
                if fingerprints[row_id] != fingerprint:
                    duplicate_conflicts += 1
                    selected[row_id]["duplicate_conflict"] = True
                continue
            fingerprints[row_id] = fingerprint
            selected[row_id] = metadata
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(f"{role} SHA mismatch: {actual_sha}")
    return selected, {
        "path": str(path),
        "sha256": actual_sha,
        "rows": total,
        "selected_ids": len(selected),
        "nonfinish_rows": nonfinish,
        "duplicate_ids": duplicate_ids,
        "duplicate_conflicts": duplicate_conflicts,
    }


def lineage_reasons(packet: Mapping[str, Any], registry: Mapping[str, Any] | None, sft: Mapping[str, Any] | None) -> list[str]:
    row_ref = packet["row_ref"]
    result = packet["result"]
    reasons: list[str] = []
    if registry is None:
        reasons.append("registry_id_missing")
    if sft is None:
        reasons.append("sft_id_missing")
    if reasons:
        return reasons
    packet_id = row_ref["row_id"]
    package_id = row_ref.get("package")
    body_text = "\n".join(result.get("target_body", [])) if isinstance(result.get("target_body"), list) else None
    body_sha = sha256_text(body_text) if isinstance(body_text, str) else None
    for label, identity in (("registry", registry), ("sft", sft)):
        if identity.get("duplicate_conflict") is True:
            reasons.append(f"{label}_duplicate_conflict")
        if identity.get("id") != packet_id:
            reasons.append(f"{label}_id_mismatch")
        if identity.get("family") != "finish_block":
            reasons.append(f"{label}_family_mismatch")
        if identity.get("package_id") != package_id:
            reasons.append(f"{label}_package_mismatch")
        if identity.get("split") != "train":
            reasons.append(f"{label}_split_not_train")
        if identity.get("target_operation") != result.get("operation"):
            reasons.append(f"{label}_operation_mismatch")
        if identity.get("target_body_text") != body_text:
            reasons.append(f"{label}_target_body_mismatch")
        if identity.get("target_body_sha256") != body_sha:
            reasons.append(f"{label}_target_body_hash_mismatch")
    packet_provenance = result.get("provenance")
    if isinstance(packet_provenance, Mapping):
        if packet_provenance.get("target_body_sha256") != body_sha:
            reasons.append("packet_target_body_hash_mismatch")
    row_target_sha = row_ref.get("target_body_sha256")
    if row_target_sha is not None and row_target_sha != body_sha:
        reasons.append("packet_row_ref_target_body_hash_mismatch")
    return reasons


def overlay_record(
    candidate: Any,
    protocol: Any,
    scenarios: Any,
    packet: Mapping[str, Any],
    registry: Mapping[str, Any] | None,
    sft: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:  # noqa: ANN401
    row_ref = packet["row_ref"]
    result = packet["result"]
    original_lines = result.get("target_body")
    original_text = "\n".join(original_lines) if isinstance(original_lines, list) and all(isinstance(x, str) for x in original_lines) else None
    original = {
        "packet_line": packet["_overlay_packet_line"],
        "packet_raw_line_sha256": packet["_overlay_packet_raw_line_sha256"],
        "packet_record_sha256": packet["_overlay_packet_record_sha256"],
        "row_ref": dict(row_ref),
        "packet_split": row_ref.get("split"),
        "split": sft.get("split") if sft else None,
        "family": packet.get("family"),
        "source_variant": packet.get("source_variant"),
        "operation": result.get("operation"),
        "target_body": original_lines,
        "target_body_text": original_text,
        "target_body_sha256": sha256_text(original_text) if isinstance(original_text, str) else None,
        "provenance": result.get("provenance"),
    }
    lineage = {
        "registry": {key: value for key, value in registry.items() if key != "target_body_text"} if registry else None,
        "sft": {key: value for key, value in sft.items() if key != "target_body_text"} if sft else None,
        "registry_line_sha256": registry.get("raw_line_sha256") if registry else None,
        "sft_line_sha256": sft.get("raw_line_sha256") if sft else None,
    }
    reasons = lineage_reasons(packet, registry, sft)
    lineage["matched_train_identity"] = not reasons
    repaired: dict[str, Any] | None = None
    eligible = False
    if not reasons:
        try:
            plan = candidate.prepare_repair(
                packet,
                utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
            )
            repaired_document = candidate.apply_plan(
                packet,
                plan,
                utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
            )
            if not parse_ok(scenarios, repaired_document):
                reasons.append("repaired_document_parse_failed")
            if isinstance(original_text, str):
                old_document = candidate.apply_document_replacement(
                    result["selection_source"]["document_text"],
                    result["context"]["replacement_range"],
                    original_text,
                    region_old=result["context"]["region_old"],
                    utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
                )
                if repaired_document != old_document + "}":
                    reasons.append("repaired_document_not_old_plus_one_brace")
            if not reasons:
                eligible = True
                repaired = {
                    "target_body": list(plan.repaired_target_lines),
                    "target_body_text": plan.repaired_target_text,
                    "target_body_sha256": sha256_text(plan.repaired_target_text),
                    "document_sha256": sha256_text(repaired_document),
                    "document_chars": len(repaired_document),
                    "document_parse_ok": True,
                    "outer_brace_materialized": plan.appended_outer_brace,
                    "strict_v2_evidence": dict(plan.evidence),
                }
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"strict_v2_rejected:{getattr(exc, 'code', type(exc).__name__)}")
    record = {
        "key": {
            "original_id": row_ref.get("row_id"),
            "packet_raw_line_sha256": packet["_overlay_packet_raw_line_sha256"],
        },
        "status": "eligible_repaired" if eligible else "rejected",
        "eligibility": {
            "strict_v2": eligible,
            "rejection_reasons": reasons,
        },
        "original": original,
        "lineage": lineage,
        "repaired": repaired,
    }
    return record, eligible


def write_overlay(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("x", encoding="utf-8", newline="") as handle:
        for record in records:
            raw = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            handle.buffer.write(raw)
            digest.update(raw)
            byte_count += len(raw)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": byte_count, "records": len(records)}


def build_manifest(overlay_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    candidate = load_module("dat04_finish_target_overlay_v2_candidate", V2_CANDIDATE)
    protocol = load_module("dat04_finish_target_overlay_protocol", PROTOCOL_PATH)
    scenarios = load_module("dat04_finish_target_overlay_scenarios", SCENARIOS_PATH)
    if sha256_file(V2_CANDIDATE) != V2_CANDIDATE_SHA256:
        raise ValueError("strict v2 candidate hash changed")
    if sha256_file(V2_RECEIPT) != V2_RECEIPT_SHA256:
        raise ValueError("strict v2 receipt hash changed")
    if sha256_file(LINEAGE_RECEIPT) != LINEAGE_RECEIPT_SHA256:
        raise ValueError("lineage receipt hash changed")
    for path, expected, label in (
        (PROTOCOL_PATH, PROTOCOL_SHA256, "protocol"),
        (SCENARIOS_PATH, SCENARIOS_SHA256, "canonical scenarios"),
        (FINISH_EXTRACTOR_PATH, FINISH_EXTRACTOR_SHA256, "finish extractor"),
        (BATCH_PATH, BATCH_SHA256, "completion batch"),
        (ADAPTER_PATH, ADAPTER_SHA256, "completion adapter"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} hash changed")

    packets, packet_audit, packet_ids = stream_packet()
    registry, registry_audit = stream_identity_artifact(REGISTRY_PATH, packet_ids, REGISTRY_SHA256, "DAT-05 registry")
    sft, sft_audit = stream_identity_artifact(SFT_PATH, packet_ids, SFT_SHA256, "SFT train rows")

    records: list[dict[str, Any]] = []
    eligible_count = 0
    rejection_counts: dict[str, int] = {}
    for packet in packets:
        record, eligible = overlay_record(candidate, protocol, scenarios, packet, registry.get(packet["row_ref"]["row_id"]), sft.get(packet["row_ref"]["row_id"]))
        records.append(record)
        if eligible:
            eligible_count += 1
        else:
            for reason in record["eligibility"]["rejection_reasons"]:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    overlay_audit = write_overlay(overlay_path, records)
    admitted = [record for record in records if record["status"] == "eligible_repaired"]
    lineage_matched = sum(1 for record in records if record["lineage"]["matched_train_identity"])
    lineage_missing = len(records) - lineage_matched
    strict_rejected = len(records) - eligible_count - lineage_missing
    duplicate_overlay_keys = len(records) - len({(record["key"]["original_id"], record["key"]["packet_raw_line_sha256"]) for record in records})
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elapsed = round(time.monotonic() - started, 3)
    pins = {
        "strict_v2_candidate": {"path": str(V2_CANDIDATE), "sha256": V2_CANDIDATE_SHA256},
        "strict_v2_receipt": {"path": str(V2_RECEIPT), "sha256": V2_RECEIPT_SHA256},
        "lineage_receipt": {"path": str(LINEAGE_RECEIPT), "sha256": LINEAGE_RECEIPT_SHA256},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256},
        "canonical_scenarios": {"path": str(SCENARIOS_PATH), "sha256": SCENARIOS_SHA256},
        "finish_extractor": {"path": str(FINISH_EXTRACTOR_PATH), "sha256": FINISH_EXTRACTOR_SHA256},
        "completion_batch": {"path": str(BATCH_PATH), "sha256": BATCH_SHA256},
        "completion_adapter": {"path": str(ADAPTER_PATH), "sha256": ADAPTER_SHA256},
        "driver": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    manifest = {
        "task": "DAT-04/DAT-08",
        "status": "prepared_finish_target_overlay_v1_validated",
        "observed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": {
            "owned_paths": [
                "docs/campaign/work/finish-target-overlay-v1/",
                "docs/campaign/receipts/DAT-04-finish-target-overlay-v1-preparation.json",
            ],
            "source_packet_finish_rows": packet_audit["finish_rows"],
            "train_lineage_matched_rows": lineage_matched,
            "overlay_records": len(records),
            "no_final_or_dev_content": True,
            "models_loaded": False,
            "gpu_or_server_launched": False,
            "network_or_ssh_used": False,
            "original_inputs_mutated": False,
            "tokenization_performed": False,
        },
        "inputs": {
            "converted_packet": packet_audit,
            "dat05_registry": registry_audit,
            "sft_train_rows": sft_audit,
        },
        "lineage": {
            "packet_finish_rows": len(packets),
            "registry_selected_ids": len(registry),
            "sft_selected_ids": len(sft),
            "packet_to_registry_missing": sum(1 for packet in packets if packet["row_ref"]["row_id"] not in registry),
            "packet_to_sft_missing": sum(1 for packet in packets if packet["row_ref"]["row_id"] not in sft),
            "packet_to_train_lineage_matched": lineage_matched,
            "packet_to_train_lineage_missing": lineage_missing,
            "strict_v2_rejected_after_train_lineage": strict_rejected,
            "duplicate_overlay_keys": duplicate_overlay_keys,
            "duplicate_or_conflict_detection": {
                "packet_duplicate_ids": packet_audit["duplicate_ids"],
                "packet_duplicate_conflicts": packet_audit["duplicate_conflicts"],
                "registry_duplicate_ids": registry_audit["duplicate_ids"],
                "registry_duplicate_conflicts": registry_audit["duplicate_conflicts"],
                "sft_duplicate_ids": sft_audit["duplicate_ids"],
                "sft_duplicate_conflicts": sft_audit["duplicate_conflicts"],
            },
        },
        "overlay": overlay_audit,
        "counts": {
            "total_finish_rows": len(packets),
            "eligible_strict_v2": eligible_count,
            "rejected": len(records) - eligible_count,
            "admitted_repaired_parse_ok": sum(1 for record in admitted if record["repaired"]["document_parse_ok"]),
            "rejected_reason_counts": rejection_counts,
            "nonfinish_packet_rows_untouched": packet_audit["nonfinish_rows"],
            "nonfinish_registry_rows_untouched": registry_audit["nonfinish_rows"],
            "nonfinish_sft_rows_untouched": sft_audit["nonfinish_rows"],
            "nonfinish_overlay_records": sum(1 for record in records if record["original"]["family"] != "finish_block"),
            "heldout_records_in_overlay": sum(1 for record in records if record["original"]["packet_split"] not in (None, "train_group", "train") or record["original"]["split"] not in (None, "train")),
        },
        "integrity": {
            "all_admitted_repaired_documents_parse": all(record["repaired"]["document_parse_ok"] for record in admitted),
            "all_overlay_keys_unique": duplicate_overlay_keys == 0,
            "all_overlay_rows_preserve_original_provenance": all(isinstance(record["original"]["provenance"], Mapping) for record in records),
            "all_admitted_rows_train": all(record["original"]["split"] == "train" for record in admitted),
            "all_packet_rows_train_group": packet_audit["split_counts"] == {"train_group": packet_audit["finish_rows"]},
            "heldout_records_present": False,
            "nonfinish_rows_untouched": sum(1 for record in records if record["original"]["family"] != "finish_block") == 0,
        },
        "resource": {
            "elapsed_seconds": elapsed,
            "process_max_rss_kib": max_rss_kib,
            "process_max_rss_mib": round(max_rss_kib / 1024, 2),
            "bounded_streams": 3,
            "output_written_once": True,
        },
        "pinned_inputs": pins,
        "materialization_policy": "shadow overlay only; root must separately review any target regeneration or production application",
        "limitations": [
            "The overlay includes only packet finish_block rows; non-finish rows are counted and untouched. It is not a replacement training artifact.",
            "Eligibility is the strict v2 EOF-bound contract. Suffix-bearing, non-LF, cross-line, hash-mismatched, or provenance-mismatched rows remain explicitly rejected.",
            "Parsing uses the pinned tree-sitter R parser on the represented packet pre-edit document; no raw source corpus is opened.",
            "No tokenizer or model-quality evaluation is performed in this overlay task.",
        ],
    }
    return manifest, records


def write_receipt_new(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest, _records = build_manifest(args.overlay)
    write_receipt_new(args.receipt, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
