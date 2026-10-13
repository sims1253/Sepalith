#!/usr/bin/env python3
"""Reclassify a completed native Q6 quality panel with pinned CPU code.

This reads only the completed panel and the stored quality responses.  It does
not contact a server, load a model, decode prompts, or make a release claim.
The exact campaign_eval.classify and campaign_protocol.PromptContext from the
admitted source tree perform the protocol and region classification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


EXPECTED_PANEL_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
EXPECTED_MODEL_SHA256 = "b11ffcc093b78261af1c5eb450feefcbca6ee35c0cecc22133236506410e143e"
EXPECTED_EVAL_SHA256 = "7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f"
EXPECTED_PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"
EXPECTED_PANEL_ROWS = 75
EXPECTED_EDIT_ROWS = 43
EXPECTED_NOOP_ROWS = 32
COMPLETION_CAP = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval(path: Path, protocol_path: Path) -> Any:
    sys.path.insert(0, str(protocol_path.parents[1]))
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("campaign_eval_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel_cases(path: Path) -> list[Mapping[str, Any]]:
    cases: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"panel has an empty line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"panel line {line_number} is not an object")
            cases.append(value)
    return cases


def compact_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def reclassify(
    *, panel_path: Path, quality_path: Path, eval_path: Path, protocol_path: Path,
) -> dict[str, Any]:
    panel_sha = sha256_file(panel_path)
    if panel_sha != EXPECTED_PANEL_SHA256:
        raise ValueError(f"panel SHA differs: {panel_sha}")
    quality = read_json(quality_path)
    if not isinstance(quality, Mapping):
        raise ValueError("quality artifact must be an object")
    if quality.get("status") != "complete" or quality.get("evaluation_complete") is not True:
        raise ValueError("quality artifact is not terminal complete")
    if quality.get("model", {}).get("provenance") != EXPECTED_MODEL_SHA256:
        raise ValueError("quality model identity differs from pinned Q6 artifact")
    static = quality.get("static_preflight", {})
    if static.get("campaign_eval", {}).get("sha256") != EXPECTED_EVAL_SHA256:
        raise ValueError("quality artifact used a different campaign_eval")
    if static.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("quality artifact used a different protocol")
    eval_sha = sha256_file(eval_path)
    protocol_sha = sha256_file(protocol_path)
    if eval_sha != EXPECTED_EVAL_SHA256 or protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("pinned evaluator/protocol bytes differ from admitted hashes")

    panel = panel_cases(panel_path)
    stored = quality.get("cases")
    if not isinstance(stored, list):
        raise ValueError("quality cases must be a list")
    if len(panel) != EXPECTED_PANEL_ROWS or len(stored) != EXPECTED_PANEL_ROWS:
        raise ValueError("quality denominator is not exactly 75 rows")
    panel_ids = [case.get("id") for case in panel]
    stored_ids = [case.get("id") for case in stored]
    if panel_ids != stored_ids or any(not isinstance(value, str) for value in panel_ids):
        raise ValueError("quality row IDs differ from the pinned DEV panel order")
    panel_ids_sha = hashlib.sha256(
        json.dumps(panel_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if panel_ids_sha != quality.get("panel", {}).get("case_ids_sha256"):
        raise ValueError("quality case-ID order hash differs from the panel")

    classify = load_eval(eval_path, protocol_path).classify
    from sepalith.campaign_protocol import PromptContext
    counts = {
        "panel_rows": 0,
        "edit_cases": 0,
        "strict_noop_cases": 0,
        "responses": 0,
        "protocol_rows": 0,
        "mechanical_failures": 0,
        "protocol_error_rows": 0,
        "edit_exact_rows": 0,
        "strict_noop_correct_rows": 0,
        "noop_false_positive_rows": 0,
        "exact_region_rows": 0,
        "cap_hit_rows": 0,
        "transport_failures": 0,
    }
    by_family: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    failure_ids: list[str] = []
    cap_ids: list[str] = []
    for panel_case, quality_case in zip(panel, stored):
        context = PromptContext.from_mapping(panel_case["context"])
        expected_noop = list(context.region_old) == panel_case["region_new"]
        raw = quality_case.get("raw_text")
        ids = quality_case.get("returned_token_ids")
        if not isinstance(raw, str) or not isinstance(ids, list) or any(type(item) is not int for item in ids):
            raise ValueError(f"stored response is malformed for {panel_case['id']}")
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != quality_case.get("raw_text_sha256"):
            raise ValueError(f"raw output hash differs for {panel_case['id']}")
        if compact_ids_sha256(ids) != quality_case.get("returned_token_ids_sha256"):
            raise ValueError(f"returned ID hash differs for {panel_case['id']}")
        independent = classify(raw, context, panel_case["region_new"], ids)
        cap_hit = len(ids) == COMPLETION_CAP and (not ids or ids[-1] != 1)
        quality_fields = quality_case.get("quality")
        if not isinstance(quality_fields, Mapping):
            raise ValueError(f"quality fields missing for {panel_case['id']}")
        expected_fields = {
            "protocol_valid": bool(independent["protocol_valid"]),
            "predicted_noop": bool(independent["predicted_noop"]),
            "edit_exact": bool(not expected_noop and independent["exact_region"]),
            # The quality artifact's exact_edit field is the all-row exact
            # region result; edit_exact is the edit-only denominator.
            "exact_edit": bool(independent["exact_region"]),
            "strict_noop_correct": bool(expected_noop and independent["predicted_noop"]),
            "noop_false_positive": bool(expected_noop and independent["suggestion"]),
        }
        for field, expected in expected_fields.items():
            if quality_fields.get(field) != expected:
                mismatches.append({"id": panel_case["id"], "field": field,
                                   "stored": quality_fields.get(field), "independent": expected})
        cap = quality_case.get("cap", {})
        if (cap.get("hit") != cap_hit or cap.get("returned_tokens_including_terminal") != len(ids)
                or cap.get("limit") != COMPLETION_CAP):
            mismatches.append({"id": panel_case["id"], "field": "cap", "stored": cap,
                               "independent": {"hit": cap_hit, "length": len(ids), "limit": COMPLETION_CAP}})
        family = str(panel_case.get("family"))
        family_counts = by_family.setdefault(family, {key: 0 for key in (
            "panel_rows", "protocol_rows", "mechanical_failures", "edit_exact_rows",
            "strict_noop_correct_rows", "noop_false_positive_rows", "cap_hit_rows",
        )})
        counts["panel_rows"] += 1
        counts["responses"] += int(bool(quality_case.get("response_received")))
        counts["transport_failures"] += int(not quality_case.get("response_received"))
        counts["protocol_rows"] += int(independent["protocol_valid"])
        counts["mechanical_failures"] += int(not independent["protocol_valid"])
        counts["protocol_error_rows"] += int(not independent["protocol_valid"])
        counts["edit_cases"] += int(not expected_noop)
        counts["strict_noop_cases"] += int(expected_noop)
        counts["edit_exact_rows"] += int(not expected_noop and independent["exact_region"])
        counts["strict_noop_correct_rows"] += int(expected_noop and independent["predicted_noop"])
        counts["noop_false_positive_rows"] += int(expected_noop and independent["suggestion"])
        counts["exact_region_rows"] += int(independent["exact_region"])
        counts["cap_hit_rows"] += int(cap_hit)
        family_counts["panel_rows"] += 1
        family_counts["protocol_rows"] += int(independent["protocol_valid"])
        family_counts["mechanical_failures"] += int(not independent["protocol_valid"])
        family_counts["edit_exact_rows"] += int(not expected_noop and independent["exact_region"])
        family_counts["strict_noop_correct_rows"] += int(expected_noop and independent["predicted_noop"])
        family_counts["noop_false_positive_rows"] += int(expected_noop and independent["suggestion"])
        family_counts["cap_hit_rows"] += int(cap_hit)
        if not independent["protocol_valid"]:
            failure_ids.append(panel_case["id"])
        if cap_hit:
            cap_ids.append(panel_case["id"])
    if mismatches:
        raise ValueError(f"independent classifier mismatches: {mismatches[:3]}")
    if counts["edit_cases"] != EXPECTED_EDIT_ROWS or counts["strict_noop_cases"] != EXPECTED_NOOP_ROWS:
        raise ValueError(f"panel operation denominator differs: {counts}")
    stored_denominators = quality.get("denominators", {})
    compare_fields = (
        "attempted_rows", "cap_hit_rows", "edit_cases", "edit_exact_rows", "exact_region_rows",
        "mechanical_failures", "noop_false_positive_rows", "panel_rows", "partial_responses",
        "protocol_error_rows", "protocol_rows", "responses", "strict_noop_cases",
        "strict_noop_correct_rows", "transport_failures",
    )
    stored_to_local = {
        "attempted_rows": counts["responses"], "partial_responses": 0,
        "cap_hit_rows": counts["cap_hit_rows"], "edit_cases": counts["edit_cases"],
        "edit_exact_rows": counts["edit_exact_rows"], "exact_region_rows": counts["exact_region_rows"],
        "mechanical_failures": counts["mechanical_failures"],
        "noop_false_positive_rows": counts["noop_false_positive_rows"],
        "panel_rows": counts["panel_rows"], "protocol_error_rows": counts["protocol_error_rows"],
        "protocol_rows": counts["protocol_rows"], "responses": counts["responses"],
        "strict_noop_cases": counts["strict_noop_cases"],
        "strict_noop_correct_rows": counts["strict_noop_correct_rows"],
        "transport_failures": counts["transport_failures"],
    }
    denominator_mismatches = {
        field: {"stored": stored_denominators.get(field), "independent": stored_to_local[field]}
        for field in compare_fields if stored_denominators.get(field) != stored_to_local[field]
    }
    if denominator_mismatches:
        raise ValueError(f"denominator mismatch: {denominator_mismatches}")
    return {
        "schema_version": "sepalith.run-09.q6-independent-reclassification.v1",
        "status": "independent_reclassification_pass",
        "quality_artifact_sha256": sha256_file(quality_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "panel": {
            "path": str(panel_path), "sha256": panel_sha, "rows": len(panel),
            "case_ids_sha256": panel_ids_sha, "edit_rows": EXPECTED_EDIT_ROWS,
            "strict_noop_rows": EXPECTED_NOOP_ROWS,
        },
        "pinned_code": {
            "campaign_eval": {"path": str(eval_path), "sha256": eval_sha},
            "campaign_protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        },
        "reclassification": {
            "denominators": counts,
            "by_family": by_family,
            "protocol_failure_ids": failure_ids,
            "cap_hit_ids": cap_ids,
            "stored_quality_fields_match": True,
            "stored_denominators_match": True,
            "raw_and_returned_id_hashes_match": True,
        },
        "interpretation": "CPU protocol/region/no-op diagnostic only; no R semantic-validity or promotion claim.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--campaign-eval", type=Path, required=True)
    parser.add_argument("--campaign-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reclassify(
        panel_path=args.panel, quality_path=args.quality,
        eval_path=args.campaign_eval, protocol_path=args.campaign_protocol,
    )
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    with temporary.open("rb") as stream:
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    with args.out.open("rb") as stream:
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "denominators": result["reclassification"]["denominators"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
