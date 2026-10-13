"""Independent, bounded audit of the frozen DAT-04B completion packet.

This reviewer reads the packet and DAT-02 metadata, but does not admit rows or
reconstruct targets.  It hashes complete source files while decoding only the
selected raw JSONL lines, then checks the source-authoritative prefix and
corpus target for every converted packet.  The projected JSONL is a byte-stable
converted-only view with the source aliases used by the downstream token audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


BATCH_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl")
BATCH_SHA256 = "42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25"
SUMMARY_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch-summary.json")
SUMMARY_SHA256 = "af389a94ba1318c0b1e30327911a63a867255a691b7b574aa86207a4988740b5"
SPLIT_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
SPLIT_SHA256 = "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"
VALIDATOR_PATH = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/cases/validators.py")
VALIDATOR_SHA256 = "5bccc747f6be4fc10131246fdf410ee3d2ee3b35cefd418cbda1db1e7f350626"
REVIEW_VERSION = "dat04b-independent-review.v1"
KNOWN_V3_IDS = (
    "0002bc91539cd90ea3089201",
    "00092523a7f767a566878fd2",
    "0009fceeb83043addb039d56",
)
FUTURE_CONTEXT_KEYS = frozenset(
    ("full_prompt", "note", "instructions", "completion", "model_target",
     "target", "generated_at", "seed")
)


class ReviewError(ValueError):
    """The frozen batch or a source identity failed independent review."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=4 * 1024 * 1024) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReviewError(f"file_unreadable:{path}") from error
    return digest.hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ReviewError(reason)


def _nested_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        found.update(str(key) for key in value)
        for child in value.values():
            found.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_nested_keys(child))
    return found


def _target_text(raw: Mapping[str, Any]) -> str:
    value = raw.get("corpus_target")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    raise ReviewError("raw_corpus_target_missing_or_invalid")


@dataclass
class Packet:
    line: int
    raw_bytes: bytes
    value: dict[str, Any]

    @property
    def row_ref(self) -> Mapping[str, Any]:
        ref = self.value.get("row_ref")
        if not isinstance(ref, Mapping):
            raise ReviewError(f"packet_row_ref_missing:{self.line}")
        return ref

    @property
    def result(self) -> Mapping[str, Any]:
        result = self.value.get("result")
        if not isinstance(result, Mapping):
            raise ReviewError(f"packet_result_missing:{self.line}")
        return result


def _load_split_groups() -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    _require(_sha256_file(SPLIT_PATH) == SPLIT_SHA256, "dat02_split_hash_mismatch")
    try:
        document = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewError("dat02_split_json_invalid") from error
    groups = document.get("groups") if isinstance(document, Mapping) else None
    _require(isinstance(groups, list), "dat02_groups_missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        _require(isinstance(group, Mapping), "dat02_group_not_object")
        group_id = group.get("group_id")
        _require(isinstance(group_id, str) and group_id, "dat02_group_id_missing")
        _require(group_id not in indexed, f"dat02_duplicate_group_id:{group_id}")
        indexed[group_id] = group
    return indexed, document


def _read_batch() -> tuple[list[Packet], list[Packet], dict[str, Any]]:
    _require(_sha256_file(BATCH_PATH) == BATCH_SHA256, "completion_batch_hash_mismatch")
    _require(_sha256_file(SUMMARY_PATH) == SUMMARY_SHA256, "completion_summary_hash_mismatch")
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewError("completion_summary_invalid") from error
    converted: list[Packet] = []
    excluded: list[Packet] = []
    batch_digest = hashlib.sha256()
    try:
        with BATCH_PATH.open("rb", buffering=4 * 1024 * 1024) as stream:
            for line_number, line_bytes in enumerate(stream, 1):
                batch_digest.update(line_bytes)
                try:
                    value = json.loads(line_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ReviewError(f"batch_json_invalid:{line_number}") from error
                _require(isinstance(value, dict), f"batch_packet_not_object:{line_number}")
                packet = Packet(line_number, line_bytes, value)
                status = packet.result.get("status")
                if status == "converted":
                    converted.append(packet)
                elif status == "excluded":
                    excluded.append(packet)
                else:
                    raise ReviewError(f"batch_status_invalid:{line_number}")
    except OSError as error:
        raise ReviewError("completion_batch_unreadable") from error
    _require(batch_digest.hexdigest() == BATCH_SHA256, "completion_batch_stream_hash_mismatch")
    counts = summary.get("result_counts", {})
    _require(counts.get("processed") == len(converted) + len(excluded), "summary_processed_count_mismatch")
    _require(counts.get("converted") == len(converted), "summary_converted_count_mismatch")
    _require(counts.get("excluded") == len(excluded), "summary_excluded_count_mismatch")
    return converted, excluded, summary


def _validate_packet_shape(packet: Packet, groups: Mapping[str, Mapping[str, Any]]) -> None:
    ref = packet.row_ref
    result = packet.result
    _require(packet.value.get("family") == "finish_block", f"packet_family_invalid:{packet.line}")
    _require(isinstance(packet.value.get("source_variant"), str) and packet.value["source_variant"],
             f"packet_source_variant_missing:{packet.line}")
    row_id = ref.get("row_id")
    _require(isinstance(row_id, str) and row_id, f"packet_row_id_missing:{packet.line}")
    _require(ref.get("split") == "train_group", f"packet_not_train_group:{row_id}")
    group_id = ref.get("group_id")
    _require(isinstance(group_id, str) and group_id, f"packet_group_missing:{row_id}")
    group = groups.get(group_id)
    _require(group is not None, f"packet_group_not_in_dat02:{row_id}:{group_id}")
    _require(group.get("split") == "train_group", f"packet_group_not_train:{row_id}:{group_id}")
    source_file = ref.get("source_file")
    source_line = ref.get("source_line")
    _require(isinstance(source_file, str) and Path(source_file).is_absolute(),
             f"packet_source_file_missing:{row_id}")
    _require(type(source_line) is int and source_line >= 1, f"packet_source_line_invalid:{row_id}")
    _require(isinstance(ref.get("package"), str) and ref["package"], f"packet_package_missing:{row_id}")
    _require(isinstance(ref.get("raw_line_sha256"), str) and len(ref["raw_line_sha256"]) == 64,
             f"packet_raw_hash_missing:{row_id}")
    _require(isinstance(ref.get("source_sha256"), str) and len(ref["source_sha256"]) == 64,
             f"packet_source_hash_missing:{row_id}")
    _require(isinstance(result.get("context"), Mapping), f"packet_context_missing:{row_id}")
    _require(not FUTURE_CONTEXT_KEYS.intersection(_nested_keys(result["context"])),
             f"future_field_in_context:{row_id}")
    _require(isinstance(result.get("target_body"), list)
             and all(isinstance(item, str) for item in result["target_body"]),
             f"target_body_invalid:{row_id}")
    _require(result.get("operation") == "replace", f"converted_operation_invalid:{row_id}")
    provenance = result.get("provenance")
    _require(isinstance(provenance, Mapping), f"packet_provenance_missing:{row_id}")
    _require(provenance.get("source_role") == "verified_train_or_dev_row",
             f"source_role_invalid:{row_id}")
    _require(provenance.get("target_authority") == ["corpus_target"],
             f"target_authority_invalid:{row_id}")
    _require(provenance.get("finish_splice", {}).get("literal_source_splice_verified") is True,
             f"finish_splice_not_verified:{row_id}")
    _require(provenance.get("finish_splice", {}).get("outer_closing_brace_in_label") is False,
             f"outer_brace_label_flag_invalid:{row_id}")
    selection = result.get("selection_source")
    _require(isinstance(selection, Mapping), f"selection_source_missing:{row_id}")
    _require(selection.get("availability") == "source_builder_window",
             f"selection_source_availability_invalid:{row_id}")
    _require(isinstance(selection.get("document_text"), str), f"selection_source_text_missing:{row_id}")
    _require(selection.get("source_jsonl_path") == source_file,
             f"selection_source_file_mismatch:{row_id}")
    _require(selection.get("source_jsonl_line") == source_line,
             f"selection_source_line_mismatch:{row_id}")
    _require(selection.get("source_jsonl_raw_line_sha256") == ref.get("raw_line_sha256"),
             f"selection_source_raw_hash_mismatch:{row_id}")
    _require(selection.get("source_jsonl_sha256") == ref.get("source_sha256"),
             f"selection_source_file_hash_mismatch:{row_id}")


def _stream_raw_rows(
    converted: list[Packet],
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], list[dict[str, Any]]]:
    requested: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for packet in converted:
        ref = packet.row_ref
        requested[str(ref["source_file"])][int(ref["source_line"])].add(str(ref["raw_line_sha256"]))
    raw_rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    file_reports: list[dict[str, Any]] = []
    for file_name, line_hashes in sorted(requested.items()):
        path = Path(file_name)
        digest = hashlib.sha256()
        found: set[int] = set()
        try:
            with path.open("rb", buffering=4 * 1024 * 1024) as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    digest.update(raw_line)
                    expected_hashes = line_hashes.get(line_number)
                    if expected_hashes is None:
                        continue
                    observed_line_hash = _sha256_bytes(raw_line)
                    _require(expected_hashes == {observed_line_hash},
                             f"raw_line_hash_mismatch:{file_name}:{line_number}")
                    try:
                        value = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ReviewError(f"raw_source_json_invalid:{file_name}:{line_number}") from error
                    _require(isinstance(value, Mapping), f"raw_source_not_object:{file_name}:{line_number}")
                    raw_rows[(file_name, line_number)] = value
                    found.add(line_number)
        except OSError as error:
            raise ReviewError(f"raw_source_unreadable:{file_name}") from error
        observed_file_hash = digest.hexdigest()
        expected_file_hashes = {
            str(packet.row_ref["source_sha256"])
            for packet in converted if packet.row_ref["source_file"] == file_name
        }
        _require(expected_file_hashes == {observed_file_hash}, f"raw_source_file_hash_mismatch:{file_name}")
        missing = set(line_hashes) - found
        _require(not missing, f"raw_source_lines_missing:{file_name}:{sorted(missing)}")
        file_reports.append({
            "path": file_name,
            "sha256": observed_file_hash,
            "selected_lines": len(line_hashes),
            "streamed_once": True,
            "parsed_selected_lines_only": True,
        })
    return raw_rows, file_reports


def _validate_splices(
    converted: list[Packet], raw_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[Counter[str], Counter[str], list[dict[str, Any]]]:
    license_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    target_mismatches = Counter()
    for packet in converted:
        ref = packet.row_ref
        result = packet.result
        row_id = str(ref["row_id"])
        raw = raw_rows[(str(ref["source_file"]), int(ref["source_line"]))]
        _require(raw.get("family") == "finish_block", f"raw_family_invalid:{row_id}")
        _require(raw.get("package") == ref.get("package"), f"raw_package_mismatch:{row_id}")
        _require(raw.get("path") == ref.get("path"), f"raw_path_mismatch:{row_id}")
        prefix = raw.get("prefix")
        _require(isinstance(prefix, str), f"raw_prefix_invalid:{row_id}")
        target = _target_text(raw)
        body = result["target_body"]
        # The deployed same-line route replaces the complete current line.
        # The source constructor's corpus target begins at the next source
        # byte, so the output body is current region_old followed by that
        # literal target fragment.  An EOF blank line uses canonical [] and
        # therefore contributes no bytes.
        region_old = result["context"].get("region_old")
        _require(isinstance(region_old, list) and all(isinstance(item, str) for item in region_old),
                 f"context_region_old_invalid:{row_id}")
        region_text = "\n".join(region_old)
        expected_body_text = region_text + target
        _require("\n".join(body) == expected_body_text,
                 f"target_body_splice_mismatch:{row_id}")
        selection = result["selection_source"]
        _require(selection["document_text"] == prefix, f"pre_edit_prefix_mismatch:{row_id}")
        _require(selection["content_sha256"] == _sha256_text(prefix), f"pre_edit_hash_mismatch:{row_id}")
        _require(result["context"]["replacement_range"]["content_sha256"] == _sha256_text(prefix),
                 f"context_content_hash_mismatch:{row_id}")
        _require(str(result["context"]["path"]) == str(ref["path"]), f"context_path_mismatch:{row_id}")
        provenance = result["provenance"]
        _require(provenance.get("target_body_sha256") == _sha256_text(expected_body_text),
                 f"provenance_target_hash_mismatch:{row_id}")
        prefix_before_region = prefix[:-len(region_text)] if region_text and prefix.endswith(region_text) else prefix
        _require(not region_text or prefix.endswith(region_text),
                 f"source_prefix_region_geometry_mismatch:{row_id}")
        splice = prefix_before_region + expected_body_text
        _require(selection["document_text"] + target == splice,
                 f"literal_prefix_target_splice_mismatch:{row_id}")
        _require(provenance["r_fragment"].get("passed") is True,
                 f"stored_r_fragment_not_passed:{row_id}")
        license_status = ref.get("license_status")
        license_counts[str(license_status or "missing")] += 1
        if ref.get("license_evidence_present") is not True:
            provenance_counts["license_evidence_missing"] += 1
        if not ref.get("source_url"):
            provenance_counts["source_url_missing"] += 1
        if not ref.get("upstream"):
            provenance_counts["upstream_missing"] += 1
        if not ref.get("named_identity_tokens"):
            provenance_counts["named_identity_tokens_missing"] += 1
        support_status = str(provenance.get("source_support", {}).get("status", "missing"))
        provenance_counts[f"source_support:{support_status}"] += 1
        if "unsupported_route_candidate" in provenance:
            provenance_counts["unsupported_route_candidate_present"] += 1
        if raw.get("target") != target:
            target_mismatches["raw_target_differs_from_corpus_target"] += 1
    return license_counts, provenance_counts, [
        {"kind": kind, "count": count} for kind, count in sorted(target_mismatches.items())
    ]


def _validator_samples(
    converted: list[Packet], raw_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not VALIDATOR_PATH.is_file() or _sha256_file(VALIDATOR_PATH) != VALIDATOR_SHA256:
        raise ReviewError("canonical_validator_hash_mismatch_or_missing")
    root = str(VALIDATOR_PATH.parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from cases.validators import fragment_clean
    except (ImportError, OSError) as error:
        raise ReviewError("canonical_r_fragment_validator_unavailable") from error
    by_variant: dict[str, Packet] = {}
    by_id: dict[str, Packet] = {}
    for packet in converted:
        by_variant.setdefault(str(packet.value["source_variant"]), packet)
        by_id[str(packet.row_ref["row_id"])] = packet
    chosen: list[tuple[str, Packet]] = [("source_variant", by_variant[key]) for key in sorted(by_variant)]
    for row_id in KNOWN_V3_IDS:
        packet = by_id.get(row_id)
        if packet is None:
            raise ReviewError(f"known_v3_row_missing:{row_id}")
        chosen.append(("known_v3", packet))
    samples: list[dict[str, Any]] = []
    for reason, packet in chosen:
        key = (str(packet.row_ref["source_file"]), int(packet.row_ref["source_line"]))
        ref = packet.row_ref
        raw = raw_rows[key]
        target = _target_text(raw)
        passed = bool(fragment_clean(str(raw["prefix"]) + target + "}"))
        if not passed:
            raise ReviewError(f"independent_r_fragment_failed:{ref['row_id']}")
        samples.append({
            "selection": reason,
            "row_id": ref["row_id"],
            "source_variant": packet.value["source_variant"],
            "source_file": ref["source_file"],
            "source_line": ref["source_line"],
            "package": ref.get("package"),
            "license": ref.get("license"),
            "license_status": ref.get("license_status"),
            "source_url": ref.get("source_url"),
            "upstream": ref.get("upstream"),
            "prefix_sha256": _sha256_text(str(raw["prefix"])),
            "corpus_target_sha256": _sha256_text(target),
            "passed": True,
        })
    _require(len(by_variant) == 7, f"source_variant_count_unexpected:{len(by_variant)}")
    return samples


def _project_packet(packet: Packet) -> dict[str, Any]:
    original = packet.value
    ref = packet.row_ref
    projected = json.loads(json.dumps(original, ensure_ascii=False))
    projected_ref = dict(projected["row_ref"])
    projected_ref.update({
        "id": ref.get("row_id"),
        "source_file": ref.get("source_file"),
        "source_line": ref.get("source_line"),
        "package": ref.get("package"),
    })
    projected["row_ref"] = projected_ref
    projected["independent_review"] = {
        "version": REVIEW_VERSION,
        "source_packet_line": packet.line,
        "source_packet_line_sha256": _sha256_bytes(packet.raw_bytes),
        "semantic_fields_identity_projection": True,
        "admission": False,
    }
    _require(projected["result"] == original["result"], f"projection_result_changed:{packet.line}")
    return projected


def run(output_path: Path, report_path: Path) -> dict[str, Any]:
    if output_path.exists() or report_path.exists():
        raise ReviewError("review_output_already_exists_immutable")
    groups, split_document = _load_split_groups()
    converted, excluded, summary = _read_batch()
    for packet in converted:
        _validate_packet_shape(packet, groups)
    raw_rows, source_file_reports = _stream_raw_rows(converted)
    license_counts, provenance_counts, target_mismatches = _validate_splices(converted, raw_rows)
    validator_samples = _validator_samples(converted, raw_rows)
    variants = Counter(str(packet.value["source_variant"]) for packet in converted)
    processed_variants = Counter(
        str(packet.value["source_variant"]) for packet in (*converted, *excluded)
    )
    exclusions = Counter(str(packet.result.get("reason", "missing")) for packet in excluded)
    summary_counts = summary["result_counts"]
    _require(dict(sorted(processed_variants.items())) ==
             dict(sorted(summary_counts["processed_source_variants"].items())),
             "summary_variant_count_mismatch")
    projected = [_project_packet(packet) for packet in converted]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.name + ".tmp")
    try:
        with temp_output.open("wb") as stream:
            for packet in projected:
                stream.write(_json(packet) + b"\n")
        temp_output.replace(output_path)
    except OSError as error:
        try:
            temp_output.unlink()
        except OSError:
            pass
        raise ReviewError("projected_output_write_failed") from error
    output_sha = _sha256_file(output_path)
    report: dict[str, Any] = {
        "review_version": REVIEW_VERSION,
        "inputs": {
            "batch": {"path": str(BATCH_PATH), "sha256": BATCH_SHA256},
            "summary": {"path": str(SUMMARY_PATH), "sha256": SUMMARY_SHA256},
            "dat02_split": {
                "path": str(SPLIT_PATH), "sha256": SPLIT_SHA256,
                "split_id": split_document.get("split_id"),
            },
        },
        "batch": {
            "packets": len(converted) + len(excluded),
            "converted": len(converted),
            "excluded": len(excluded),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "converted_source_variants": dict(sorted(variants.items())),
            "processed_source_variants": dict(sorted(processed_variants.items())),
            "expected_counts_match": True,
        },
        "dat02_train_group_check": {
            "converted_rows_checked": len(converted),
            "groups_indexed": len(groups),
            "all_converted_groups_train_group": True,
            "final_or_tu3_records_opened": False,
        },
        "raw_source_validation": {
            "files_referenced": len(source_file_reports),
            "files_streamed_once": len(source_file_reports),
            "selected_raw_rows_checked": len(raw_rows),
            "all_file_hashes_match": True,
            "all_raw_line_hashes_match": True,
            "line_bytes_include_original_separator": True,
            "files": source_file_reports,
            "literal_prefix_target_splices_checked": len(converted),
            "literal_prefix_target_splices_passed": len(converted),
        },
        "validator": {
            "path": str(VALIDATOR_PATH),
            "sha256": VALIDATOR_SHA256,
            "representative_rows": len(validator_samples),
            "source_variant_minimum": 7,
            "known_v3_rows_requested": list(KNOWN_V3_IDS),
            "samples": validator_samples,
        },
        "license_and_source_provenance": {
            "license_status_counts": dict(sorted(license_counts.items())),
            "provenance_flag_counts": dict(sorted(provenance_counts.items())),
            "raw_target_vs_corpus_target": target_mismatches,
            "representative_samples_are_from_converted_train_rows": True,
        },
        "projection": {
            "path": str(output_path),
            "sha256": output_sha,
            "rows": len(projected),
            "converted_only": True,
            "semantic_result_fields_identity_copied": True,
            "aliases": ["row_ref.id", "row_ref.source_file", "row_ref.source_line", "row_ref.package"],
            "admission": False,
        },
        "limitations": [
            "This is an independent packet/source audit; it does not perform tokenizer admission or model evaluation.",
            "All converted rows carry source_derived_simulated_pre_edit lineage; no observed editor trace is claimed.",
            "DAT-02 metadata was checked for train_group identity; final/TU3 records were not opened.",
            "The projected packet preserves existing semantic result fields and adds only review metadata and source aliases.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(_json(report) + b"\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.output, args.report)
    print(_json(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
