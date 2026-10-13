#!/usr/bin/env python3
"""Score selected-R2 F16/Q8 probe outputs without loading model data.

Quality is mechanical: protocol validity, exact target body on edit rows, and
strict ``[NO_EDIT]`` on no-op rows.  Cold records are the one quality record
per row; warm records remain available for latency.  Missing/invalid protocol
records stay in their denominators and are never treated as correct output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def index_requests(result: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    requests = result.get("requests")
    if not isinstance(requests, list):
        raise ValueError("probe output has no requests")
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in requests:
        if not isinstance(record, dict):
            raise ValueError("probe request is not an object")
        key = (str(record.get("row_id")), str(record.get("phase")), int(record.get("rep", 0)))
        if key in indexed:
            raise ValueError(f"duplicate probe request {key}")
        indexed[key] = record
    return indexed


def score_record(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    protocol_valid = record.get("protocol_status") == "accepted"
    parsed = record.get("parsed_output")
    parsed = parsed if isinstance(parsed, dict) else {}
    operation = parsed.get("operation")
    body = parsed.get("body_text")
    expected_operation = row.get("target_operation")
    expected_body = row.get("target_body_text")
    checks = record.get("protocol_checks")
    checks = checks if isinstance(checks, dict) else {}
    final = record.get("final")
    final = final if isinstance(final, dict) else {}
    returned = record.get("returned_token_ids")
    cap_hit = (
        checks.get("returned_ids") == "over_cap"
        or (isinstance(returned, list) and len(returned) > 192)
        or final.get("truncated") is True
        or final.get("stop_type") in {"limit", "length"}
    )
    predicted_noop = protocol_valid and operation == "no_op"
    strict_noop_correct = (
        expected_operation == "no_op" and protocol_valid
        and operation == "no_op" and body == "[NO_EDIT]"
    )
    noop_false_positive = (
        expected_operation == "no_op" and protocol_valid and operation != "no_op"
    )
    exact_edit = (
        expected_operation != "no_op" and protocol_valid
        and operation == expected_operation and body == expected_body
    )
    return {
        "row_id": row["id"],
        "family": row.get("family"),
        "target_operation": expected_operation,
        "protocol_valid": protocol_valid,
        "parser_status": parsed.get("status"),
        "parser_operation": operation,
        "cap_hit": cap_hit,
        "predicted_noop": predicted_noop,
        "strict_noop_correct": strict_noop_correct,
        "noop_false_positive": noop_false_positive,
        "exact_edit": exact_edit,
        "raw_text_sha256": record.get("raw_text_sha256"),
        "returned_ids_sha256": hashlib.sha256(
            json.dumps(record.get("returned_token_ids"), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def summarize(rows: list[dict[str, Any]], result: dict[str, Any], arm: str) -> dict[str, Any]:
    indexed = index_requests(result)
    scored: list[dict[str, Any]] = []
    missing: list[str] = []
    missing_keys: list[list[object]] = []
    expected_keys = {
        (row["id"], phase, 1)
        for row in rows
        for phase in ("cold", "warm")
    }
    for key in sorted(expected_keys - set(indexed), key=repr):
        missing_keys.append(list(key))
    for row in rows:
        key = (row["id"], "cold", 1)
        record = indexed.get(key)
        if record is None:
            missing.append(row["id"])
            scored.append({"row_id": row["id"], "family": row.get("family"),
                           "target_operation": row.get("target_operation"),
                           "protocol_valid": False, "parser_status": None,
                           "parser_operation": None, "cap_hit": False,
                           "predicted_noop": False, "strict_noop_correct": False,
                           "noop_false_positive": False, "exact_edit": False})
        else:
            scored.append(score_record(row, record))
    edits = [item for item in scored if item["target_operation"] != "no_op"]
    noops = [item for item in scored if item["target_operation"] == "no_op"]
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({str(row.get("family")) for row in rows}):
        members = [item for item in scored if str(item.get("family")) == family]
        by_family[family] = {
            "rows": len(members),
            "protocol_valid": sum(item["protocol_valid"] for item in members),
            "cap_hits": sum(item["cap_hit"] for item in members),
            "exact_edits": sum(item["exact_edit"] for item in members),
            "strict_noop_correct": sum(item["strict_noop_correct"] for item in members),
            "noop_false_positives": sum(item["noop_false_positive"] for item in members),
        }
    return {
        "arm": arm,
        "model_identity": result.get("model_identity"),
        "rows": len(rows),
        "missing_cold_rows": missing,
        "missing_request_keys": missing_keys,
        "request_key_complete": not missing_keys,
        "probe_status": result.get("status"),
        "protocol_valid_rows": sum(item["protocol_valid"] for item in scored),
        "protocol_error_rows": sum(not item["protocol_valid"] for item in scored),
        "cap_hit_rows": sum(item["cap_hit"] for item in scored),
        "edit_cases": len(edits),
        "edit_exact_rows": sum(item["exact_edit"] for item in edits),
        "strict_noop_cases": len(noops),
        "strict_noop_correct_rows": sum(item["strict_noop_correct"] for item in noops),
        "noop_false_positive_rows": sum(item["noop_false_positive"] for item in noops),
        "by_family": by_family,
        "rows_detail": scored,
    }


def compare(f16: dict[str, Any], q8: dict[str, Any]) -> dict[str, Any]:
    left, right = index_requests(f16), index_requests(q8)
    keys = sorted(set(left) | set(right), key=repr)
    mismatches: list[dict[str, Any]] = []
    for key in keys:
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            mismatches.append({"key": list(key), "reason": "missing_pair"})
            continue
        if a.get("raw_text") != b.get("raw_text") or a.get("returned_token_ids") != b.get("returned_token_ids"):
            mismatches.append({"key": list(key), "reason": "output_or_token_id_difference",
                               "f16_raw_text_sha256": a.get("raw_text_sha256"),
                               "q8_raw_text_sha256": b.get("raw_text_sha256")})
    return {
        "paired_request_keys": len(keys),
        "exact_raw_text_and_token_id_rows": len(keys) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "parity_is_a_quality_observation": True,
    }


def score_pair(panel_path: Path, manifest_path: Path, f16_path: Path, q8_path: Path) -> dict[str, Any]:
    manifest = load(manifest_path)
    if manifest.get("schema_version") != "sepalith.r2.serving.train-panel.v1":
        raise ValueError("unexpected panel manifest schema")
    records = manifest.get("panel", {}).get("records", [])
    raw_lines = panel_path.read_bytes().splitlines(keepends=True)
    if not isinstance(records, list) or len(records) != len(raw_lines):
        raise ValueError("panel and manifest record counts differ")
    rows = []
    for raw, identity in zip(raw_lines, records):
        row = json.loads(raw)
        if not isinstance(row, dict) or not isinstance(identity, dict):
            raise ValueError("panel and manifest records must be objects")
        if identity.get("row_id") != row.get("id") or identity.get("row_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"panel row byte hash mismatch for {row.get('id')!r}")
        rows.append(row)
    if len(rows) != 8:
        raise ValueError("quant qualification requires exactly eight panel rows")
    f16, q8 = load(f16_path), load(q8_path)
    expected_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for name, result, expected_arm in (("f16", f16, "f16"), ("q8", q8, "q8")):
        if result.get("panel", {}).get("manifest_sha256") != expected_manifest_sha:
            raise ValueError(f"{name} output manifest hash mismatch")
        if result.get("model_identity", {}).get("arm") != expected_arm:
            raise ValueError(f"{name} model identity arm mismatch")
        expected_sha = manifest["selected_r2_identity"][f"{expected_arm}_sha256"]
        if result.get("model_identity", {}).get("sha256") != expected_sha:
            raise ValueError(f"{name} selected R2 model hash mismatch")
    f16_summary = summarize(rows, f16, "f16")
    q8_summary = summarize(rows, q8, "q8")
    measurement_complete = bool(
        f16_summary["request_key_complete"] and q8_summary["request_key_complete"]
    )
    return {
        "schema_version": "sepalith.r2.final-quant.qualification.v1",
        # This packet measures a bounded TRAIN panel.  Never call a pair a
        # quality pass automatically: root must review the denominators and
        # decide whether the evidence is sufficient for the selected R2 gate.
        "status": "measured_requires_root_review" if measurement_complete else "partial",
        "measurement_complete": measurement_complete,
        "panel": {"path": str(panel_path), "manifest": str(manifest_path),
                   "manifest_sha256": expected_manifest_sha, "rows": len(rows),
                   "source_sha256": manifest["source"]["sha256"]},
        "selected_r2_identity": manifest["selected_r2_identity"],
        "f16": f16_summary,
        "q8": q8_summary,
        "paired_comparison": compare(f16, q8),
        "interpretation": {
            "quality_scope": "mechanical TRAIN panel only; no DEV/final claim",
            "exact_edit": "protocol-valid parsed body equals source target_body_text",
            "strict_noop": "protocol-valid parsed operation no_op with exact [NO_EDIT] body",
            "quantization_parity": "F16/Q8 may legitimately differ; mismatches are recorded, not failed",
            "ordinary_decoding": "both arms must use identical server/client greedy settings",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--f16", type=Path, required=True)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_pair(args.panel, args.manifest, args.f16, args.q8)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"schema_version": "sepalith.r2.final-quant.qualification.v1",
                  "status": "failed", "error": str(exc)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "f16_protocol": result["f16"]["protocol_valid_rows"],
                      "q8_protocol": result["q8"]["protocol_valid_rows"],
                      "paired_exact": result["paired_comparison"]["exact_raw_text_and_token_id_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
