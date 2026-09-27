#!/usr/bin/env python3
"""Independent, model-free checks for the DAT-04A structured bundle."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


EXECUTION = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PROTOCOL_SRC = EXECUTION / "packages/sepalith/src"
if str(PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_SRC))

from sepalith.campaign_protocol import PromptContext, ProtocolError  # noqa: E402


BUNDLE = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-reference-bundle.json")
OUTPUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-source-checks.json")
AUDIT_SHA256 = "9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd"
MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
MANIFEST_SHA256 = "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"
PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_source(raw: bytes) -> tuple[str, str]:
    text = raw.decode("utf-8")
    lone_cr = text.replace("\r\n", "")
    if "\r" in lone_cr:
        raise ValueError("unsupported_mixed_or_lone_cr")
    eol = "crlf" if "\r\n" in text else "lf"
    return text.replace("\r\n", "\n"), eol


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    bundle_bytes = BUNDLE.read_bytes()
    bundle_sha = sha256_bytes(bundle_bytes)
    bundle = json.loads(bundle_bytes.decode("utf-8"))
    errors: list[str] = []
    if bundle.get("source_audit_sha256") != AUDIT_SHA256:
        errors.append("bundle_source_audit_sha_mismatch")
    if bundle.get("final_rows_opened") is not False:
        errors.append("final_rows_opened_flag_not_false")
    if bundle.get("tu3_rows_opened") is not False:
        errors.append("tu3_rows_opened_flag_not_false")
    if sha256_path(MANIFEST) != MANIFEST_SHA256:
        errors.append("accepted_DAT02_manifest_sha_mismatch")
    if sha256_path(PROTOCOL_SRC / "sepalith/campaign_protocol.py") != PROTOCOL_SHA256:
        errors.append("frozen_PRM04_sha_mismatch")

    converted = 0
    operations: dict[str, int] = {}
    availability: dict[str, int] = {}
    support_status: dict[str, int] = {}
    staged: dict[str, dict[str, Any]] = {}
    row_checks: list[dict[str, Any]] = []
    allowed_ops = {"replace", "no_op", "delete"}
    allowed_splits = {"train_group", "dev_group"}
    for item in bundle.get("rows", []):
        family = str(item.get("family"))
        result = item.get("result")
        if not isinstance(result, dict) or result.get("status") != "converted":
            errors.append(f"{family}:not_converted")
            continue
        converted += 1
        operation = result.get("operation")
        if operation not in allowed_ops:
            errors.append(f"{family}:operation_invalid")
        operations[operation] = operations.get(operation, 0) + 1
        context_data = result.get("context")
        try:
            context = PromptContext.from_dict(context_data)
        except (ProtocolError, TypeError, ValueError):
            errors.append(f"{family}:context_invalid")
            continue
        selection = result.get("selection_source")
        if not isinstance(selection, dict):
            errors.append(f"{family}:selection_source_missing")
            continue
        source_path = Path(str(selection.get("path", "")))
        try:
            source_bytes = source_path.read_bytes()
            source_lf, source_eol = normalize_source(source_bytes)
        except (OSError, UnicodeDecodeError, ValueError):
            errors.append(f"{family}:selection_source_unreadable")
            continue
        source_sha = sha256_bytes(source_bytes)
        source_hash_ok = source_sha == selection.get("content_sha256")
        if not source_hash_ok:
            errors.append(f"{family}:selection_source_hash_mismatch")
        if len(source_bytes) != selection.get("content_bytes"):
            errors.append(f"{family}:selection_source_byte_count_mismatch")
        lines = source_lf.split("\n")
        start = selection.get("region_start_line")
        end = selection.get("region_end_line")
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            errors.append(f"{family}:selection_region_geometry_invalid")
        else:
            expected_region = list(context.region_old)
            actual_region = lines[start:end + 1] if expected_region else []
            if actual_region != expected_region:
                errors.append(f"{family}:selection_region_content_mismatch")
            if bool(expected_region) != (start != end or bool(actual_region)):
                # Empty canonical regions have start == end; nonempty regions
                # have the exact inclusive line span above.
                if expected_region:
                    errors.append(f"{family}:selection_region_empty_mismatch")
        replacement = context.replacement_range.to_dict()
        if replacement["content_sha256"] != selection.get("content_sha256"):
            errors.append(f"{family}:context_selection_hash_mismatch")
        if replacement["uri"] != selection.get("uri") or replacement["document_version"] != selection.get("document_version"):
            errors.append(f"{family}:context_selection_identity_mismatch")
        if selection.get("availability") not in {"full_snapshot", "source_builder_window"}:
            errors.append(f"{family}:selection_availability_invalid")
        availability[str(selection.get("availability"))] = availability.get(str(selection.get("availability")), 0) + 1

        provenance = result.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("split") not in allowed_splits:
            errors.append(f"{family}:split_not_train_or_dev")
            provenance = provenance if isinstance(provenance, dict) else {}
        for name in ("source_sha256", "raw_line_sha256", "before_snapshot_sha256", "after_snapshot_sha256"):
            value = provenance.get(name)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"{family}:{name}_invalid")
        history = provenance.get("history_replay", {})
        if history.get("present"):
            if history.get("replayed") is not True or history.get("pending_target_geometry") is not False:
                errors.append(f"{family}:history_not_fully_replayed")
            if history.get("target_geometry_verified") is not True:
                errors.append(f"{family}:history_geometry_not_verified")
            if history.get("range_content_sha256") != history.get("before_snapshot_sha256"):
                errors.append(f"{family}:history_range_hash_not_before_hash")
            if history.get("after_snapshot_sha256") != provenance.get("after_snapshot_sha256"):
                errors.append(f"{family}:history_after_hash_disagrees")
        status = str(provenance.get("support_status", "missing"))
        support_status[status] = support_status.get(status, 0) + 1
        row_checks.append({
            "family": family,
            "row_id": provenance.get("row_id"),
            "split": provenance.get("split"),
            "operation": operation,
            "availability": selection.get("availability"),
            "source_sha256": source_sha,
            "source_bytes": len(source_bytes),
            "source_eol": source_eol,
            "region_start_line": start,
            "region_end_line": end,
            "history_present": bool(history.get("present")),
            "history_lineage_role": history.get("lineage_role"),
            "support_status": status,
            "admission_eligible": provenance.get("admission_eligible"),
        })
        staged[family] = {
            "path": str(source_path),
            "sha256": source_sha,
            "bytes": len(source_bytes),
        }
    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    output = {
        "task": "DAT-04A",
        "artifact": "independent structured source/provenance checks; non-training",
        "started": started,
        "ended": finished,
        "bundle_path": str(BUNDLE),
        "bundle_sha256": bundle_sha,
        "bundle_bytes": len(bundle_bytes),
        "accepted_inputs": {
            "DAT03_row_audit_sha256": AUDIT_SHA256,
            "DAT02_manifest_sha256": MANIFEST_SHA256,
            "PRM04_protocol_sha256": PROTOCOL_SHA256,
        },
        "converted_count": converted,
        "operation_counts": operations,
        "availability_counts": availability,
        "support_status_counts": support_status,
        "staged_sources": staged,
        "row_checks": row_checks,
        "final_rows_opened": bundle.get("final_rows_opened"),
        "tu3_rows_opened": bundle.get("tu3_rows_opened"),
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("checks", output["result"])
    print("bundle_sha256", bundle_sha)
    print("converted", converted)
    print("errors", len(errors))
    print("output", OUTPUT)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
