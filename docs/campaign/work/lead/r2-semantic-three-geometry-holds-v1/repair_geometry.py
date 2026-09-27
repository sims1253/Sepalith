#!/usr/bin/env python3
"""Repair three mixed-EOL geometry holds without using target guesses."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
BASE_PATH = PLAN / "docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py"
BASE_SHA256 = "2edfe6c4761271889e2b2f008fcccd00b8bf2adaf712c18cc2d588eb9840136f"
REQUIRED_IDS = (
    "4e2a158ea14eb636107c0a1f",
    "4f19a0591c9b56764705cec3",
    "4fc746562d6b6f36c95df72b",
)
EXPECTED_HOLDS = {
    REQUIRED_IDS[0]: "Hold:full_document_identity_derivation_failed:target_function_not_after_only_blank_gap",
    REQUIRED_IDS[1]: "Hold:full_document_identity_derivation_failed:target_line_block_not_unique",
    REQUIRED_IDS[2]: "Hold:full_document_identity_derivation_failed:target_line_block_not_unique",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_base() -> Any:
    if sha(BASE_PATH) != BASE_SHA256:
        raise RuntimeError("base_materializer_pin_mismatch")
    spec = importlib.util.spec_from_file_location("three_geometry_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("base_materializer_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B = load_base()


def normalized_with_boundaries(raw: bytes) -> tuple[bytes, list[int]]:
    """Normalize CRLF to LF and map every normalized byte boundary to raw."""
    output = bytearray()
    boundaries = [0]
    i = 0
    while i < len(raw):
        if raw[i : i + 2] == b"\r\n":
            output.append(10)
            i += 2
        else:
            if raw[i] == 13:
                raise B.Hold("lone_cr_in_source")
            output.append(raw[i])
            i += 1
        boundaries.append(i)
    return bytes(output), boundaries


def exact_occurrences(haystack: bytes, needle: bytes) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return positions
        positions.append(found)
        start = found + 1


def checked_row(
    semantic: dict[str, Any], provenance: dict[str, Any], packet: dict[str, Any], hold: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rid = semantic.get("row_id")
    if rid not in REQUIRED_IDS or hold != {"reason": EXPECTED_HOLDS[rid], "row_id": rid, "silent_drop": False, "stage": "preparation"}:
        raise B.Hold("held_row_identity_mismatch")
    if semantic.get("status") != "semantic_supported_context_closure_root_review_required" or semantic.get("reasons") != []:
        raise B.Hold("semantic_status_not_supported")
    identity = B.validate_identity(semantic, provenance, packet)
    source = Path(semantic["source_path"])
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    stat_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    stat_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stat_before != stat_after or len(raw) != before.st_size or sha_bytes(raw) != semantic.get("source_sha256"):
        raise B.Hold("source_identity_or_stability_failed")

    normalized, boundaries = normalized_with_boundaries(raw)
    method = semantic.get("target_occurrence_method")
    parse_bytes = raw if method == "exact_bytes" else normalized if method == "uniform_crlf_to_lf" else None
    if parse_bytes is None:
        raise B.Hold("unsupported_occurrence_method")
    parse_sha = sha_bytes(parse_bytes)
    if parse_sha != semantic.get("parse_bytes_sha256") or parse_sha != semantic.get("scope", {}).get("parsed_source_sha256"):
        raise B.Hold("semantic_parse_identity_failed")
    source_lf = normalized.decode("utf-8")
    parse_text = parse_bytes.decode("utf-8")

    target_lines = packet.get("result", {}).get("target_body")
    if not isinstance(target_lines, list) or not target_lines or any(not isinstance(x, str) or "\n" in x or "\r" in x for x in target_lines):
        raise B.Hold("target_body_invalid")
    target = ("\n".join(target_lines) + "\n").encode()
    occurrences = exact_occurrences(parse_bytes, target)
    if len(occurrences) != 1 or semantic.get("target_occurrences") != 1:
        raise B.Hold("target_occurrence_not_exactly_one")
    if sha_bytes(target) != semantic.get("target_sha256") or len(target) != semantic.get("target_bytes"):
        raise B.Hold("target_hash_or_size_mismatch")
    target_start = occurrences[0]
    target_start_line = parse_bytes[:target_start].count(b"\n")
    span = B.checked_span(semantic["context_closure"]["target_definition_span"], len(parse_text.split("\n")), "target_definition")
    if semantic.get("scope", {}).get("target_definition_span") != semantic["context_closure"]["target_definition_span"]:
        raise B.Hold("semantic_span_join_failed")
    # The asserted definition begins immediately after the documentation block.
    # This is stronger than the old document-wide delimiter/blank-gap heuristic.
    if target_start_line + len(target_lines) != span[0]:
        raise B.Hold("target_not_immediately_attached_to_asserted_definition")

    item = {k: semantic[k] for k in ("target_occurrences", "target_definition_name", "documented_params", "imported_symbols", "prior_reviewed_recovery")}
    reasons, resolution, closure = B.SEMANTIC.decide(item, semantic.get("scope"), source_lf)
    if reasons or resolution.get("unresolved") or closure != semantic.get("context_closure"):
        raise B.Hold("semantic_recheck_failed:" + ",".join(reasons))
    if closure.get("required_helper_spans"):
        raise B.Hold("unexpected_helper_spans")

    base = B.PROTOCOL.PromptContext.from_mapping(packet["result"]["context"])
    if packet["result"].get("operation") != "replace" or base.region_old or not base.replacement_range.is_empty:
        raise B.Hold("zero_width_roxygen_geometry_required")
    candidate = B._validate_candidate_window(base, packet, target_lines, source_lf, target_start_line)
    selection_text = packet["result"]["selection_source"]["text"]
    if "\n".join(target_lines) in selection_text:
        raise B.Hold("target_leaked_into_prediction_selection")

    # Map the unique normalized target occurrence back to exact raw bytes. For an
    # exact-bytes row the boundary map is still valid because normalization does
    # not change any byte before or within its LF target occurrence.
    normalized_target_positions = exact_occurrences(normalized, target)
    if len(normalized_target_positions) != 1:
        raise B.Hold("normalized_target_occurrence_not_exactly_one")
    norm_start = normalized_target_positions[0]
    raw_start, raw_end = boundaries[norm_start], boundaries[norm_start + len(target)]
    raw_target = raw[raw_start:raw_end]
    if raw_target.replace(b"\r\n", b"\n") != target:
        raise B.Hold("raw_target_mapping_failed")
    target_eols = [b"\r\n" if raw_target[m.start() - 1 : m.start() + 1] == b"\r\n" else b"\n" for m in __import__("re").finditer(b"\n", raw_target)]
    if len(target_eols) != len(target_lines):
        raise B.Hold("target_block_eol_count_mismatch")
    # Preserve even a transition inside the target block. The empty anchor uses
    # the original final target-line terminator; reapplication restores the
    # captured raw target bytes, including each original internal terminator.
    anchor_eol = target_eols[-1]
    raw_preedit = raw[:raw_start] + anchor_eol + raw[raw_end:]
    if raw_preedit[:raw_start] + raw_target + raw_preedit[raw_start + len(anchor_eol) :] != raw:
        raise B.Hold("raw_full_source_reapplication_failed")
    if raw[:raw_start] != raw_preedit[:raw_start] or raw[raw_end:] != raw_preedit[raw_start + len(anchor_eol) :]:
        raise B.Hold("non_target_source_bytes_changed")
    raw_preedit.decode("utf-8")

    relative = provenance.get("normalized_relative_source_path")
    if not isinstance(relative, str) or not relative or relative.startswith("/") or "\\" in relative:
        raise B.Hold("relative_source_path_invalid")
    context_mapping = base.to_dict()
    prediction = {
        "schema": "sepalith.dat10.semantic_three_geometry.prediction_input.v1",
        "row_id": rid,
        "path": relative,
        "context": context_mapping,
        "preedit_text": selection_text,
        "preedit_sha256": sha_bytes(selection_text.encode()),
        "cursor": base.replacement_range.start.to_dict(),
        "document_eol": base.document_eol,
        "training_admission": False,
    }
    sidecar = {
        "schema": "sepalith.dat10.semantic_three_geometry.training_sidecar.v1",
        "row_id": rid,
        "identity": identity,
        "target_lines": target_lines,
        "target_sha256": semantic["target_sha256"],
        "source": {
            "path": str(source),
            "sha256": semantic["source_sha256"],
            "parse_bytes_sha256": parse_sha,
            "occurrence_method": method,
            "target_normalized_byte_range": [norm_start, norm_start + len(target)],
            "target_raw_byte_range": [raw_start, raw_end],
            "target_start_line_zero_based": target_start_line,
            "target_definition_span_one_based_inclusive": semantic["scope"]["target_definition_span"],
            "target_definition_name": semantic["target_definition_name"],
            "target_physical_eol": "mixed" if len(set(target_eols)) > 1 else "crlf" if anchor_eol == b"\r\n" else "lf",
            "target_physical_eol_sha256": sha_bytes(b"".join(target_eols)),
            "source_lf_count": raw.count(b"\n"),
            "source_crlf_count": raw.count(b"\r\n"),
            "raw_preedit_sha256": sha_bytes(raw_preedit),
            "raw_preedit_bytes": len(raw_preedit),
            "raw_postedit_sha256": sha_bytes(raw),
            "raw_postedit_bytes": len(raw),
            "full_source_reapplication_exact": True,
            "non_target_bytes_preserved": True,
        },
        "candidate_audit": candidate,
        "semantic_recheck": {"reasons": reasons, "unresolved": resolution.get("unresolved", []), "closure_exact": True},
        "training_admission": False,
    }
    return prediction, sidecar


def read_selected(path: Path) -> dict[str, dict[str, Any]]:
    rows, _, _ = B.exact_rows(path)
    missing = set(REQUIRED_IDS) - set(rows)
    if missing:
        raise ValueError("selected_rows_missing:" + ",".join(sorted(missing)))
    return {rid: rows[rid] for rid in REQUIRED_IDS}


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("semantic_manifest", "semantic_ledger", "provenance_ledger", "candidate_packets", "holds"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh_output_required")
    for name in ("semantic_manifest", "semantic_ledger", "provenance_ledger", "candidate_packets", "holds"):
        path = getattr(args, name)
        if sha(path) != getattr(args, "expected_" + name + "_sha256"):
            raise ValueError("input_hash_mismatch:" + name)
    manifest = json.loads(args.semantic_manifest.read_text())
    if manifest.get("status") != "complete_review_only" or manifest.get("exact_id_closure") is not True or manifest.get("output", {}).get("sha256") != sha(args.semantic_ledger):
        raise ValueError("semantic_manifest_binding_invalid")
    semantic = read_selected(args.semantic_ledger)
    provenance = read_selected(args.provenance_ledger)
    packets = read_selected(args.candidate_packets)
    holds = read_selected(args.holds)
    predictions, sidecars = [], []
    for rid in REQUIRED_IDS:
        prediction, sidecar = checked_row(semantic[rid], provenance[rid], packets[rid], holds[rid])
        predictions.append(prediction)
        sidecars.append(sidecar)
    if {x["row_id"] for x in predictions} != set(REQUIRED_IDS) or {x["row_id"] for x in sidecars} != set(REQUIRED_IDS):
        raise ValueError("exact_row_closure_failed")
    args.output.mkdir(parents=True)
    atomic_jsonl(args.output / "prediction-inputs.review-only.jsonl", predictions)
    atomic_jsonl(args.output / "training-sidecar.review-only.jsonl", sidecars)
    outputs = {}
    for name in ("prediction-inputs.review-only.jsonl", "training-sidecar.review-only.jsonl"):
        path = args.output / name
        outputs[name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "rows": 3}
    result = {
        "schema": "sepalith.dat10.semantic_three_geometry_repair.v1",
        "status": "complete_review_only_root_admission_required",
        "rows": 3,
        "row_ids": list(REQUIRED_IDS),
        "exact_row_closure": True,
        "diagnosis": "mixed LF/CRLF source files were interpreted with one document-wide delimiter by the old full-document helper",
        "policy": {"source_backed_span_required": True, "target_occurrence_required": 1, "full_raw_source_reapplication_required": True, "target_free_prediction": True, "training_admission": False},
        "inputs": {name: {"path": str(getattr(args, name)), "sha256": sha(getattr(args, name))} for name in ("semantic_manifest", "semantic_ledger", "provenance_ledger", "candidate_packets", "holds")},
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
