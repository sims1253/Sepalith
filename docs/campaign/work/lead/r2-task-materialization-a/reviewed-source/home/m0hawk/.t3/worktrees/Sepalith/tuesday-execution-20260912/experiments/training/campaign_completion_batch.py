"""Bounded, provenance-labelled materialisation of completion candidates.

This pass is intentionally separate from both the structured scenario adapter
and tokenisation.  It reads the accepted DAT-03 audit, selects only eligible
``train_group`` finish-block rows, streams each referenced source JSONL file
once through :class:`VerifiedSourceCache`, and writes complete conversion
packets.  The source-builder constructor creates a simulated prediction-time
window from ``prefix`` only.  Teacher instructions and target fields remain
out-of-band and never become prompt context.

The real source rows are not observed editor traces.  Every converted packet
therefore carries ``source_derived_simulated_pre_edit`` lineage and a
``source_builder_window`` selection source.  This module does not admit data
or claim model quality; it prepares reviewable candidate packets for the
downstream selector and token audit.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from campaign_admission_completion import (
    ADAPTER_VERSION,
    AdapterError,
    VerifiedSourceCache,
    convert_completion,
)


BATCH_VERSION = "dat04b-completion-batch-v1"
DEFAULT_MAX_ROWS = 5000
DEFAULT_PROFILE_LIMIT = 512
MAX_ROWS = 5000
LINEAGE = "source_derived_simulated_pre_edit"
SOURCE_CONSTRUCTOR = "finish_block_v5_prefix"
_SYNTHETIC_ROOT = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/synthetic-data"
)
_VALIDATOR_PATH = _SYNTHETIC_ROOT / "cases" / "validators.py"
_FRAGMENT_CLEAN: Any = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _canonical_json_sha256(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fragment_clean(text: str) -> bool:
    """Use the deployed synthetic validator without importing it at startup."""
    global _FRAGMENT_CLEAN
    if _FRAGMENT_CLEAN is None:
        if not _VALIDATOR_PATH.is_file():
            raise AdapterError("canonical_fragment_validator_unavailable")
        root = str(_SYNTHETIC_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from cases.validators import fragment_clean
        except (ImportError, OSError) as error:
            raise AdapterError("canonical_fragment_validator_unavailable") from error
        _FRAGMENT_CLEAN = fragment_clean
    return bool(_FRAGMENT_CLEAN(text))


def _validator_identity() -> dict[str, Any]:
    if not _VALIDATOR_PATH.is_file():
        return {
            "path": str(_VALIDATOR_PATH),
            "sha256": None,
            "available": False,
        }
    return {
        "path": str(_VALIDATOR_PATH),
        "sha256": _file_sha256(_VALIDATOR_PATH),
        "available": True,
    }


def _is_eligible_audit_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("split") == "train_group"
        and isinstance(row.get("family"), str)
        and row["family"].startswith("finish_block")
        and row.get("license_evidence_present") is True
        and row.get("license_status") == "direct_row_evidence"
        and isinstance(row.get("named_identity_tokens"), list)
        and bool(row["named_identity_tokens"])
    )


def _scan_audit(
    audit_path: Path, max_rows: int,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """Read one bounded audit file pass and hash all bytes that were read."""
    selected: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    total_lines = 0
    eligible_count = 0
    try:
        with audit_path.open("rb", buffering=4 * 1024 * 1024) as handle:
            for line_number, line_bytes in enumerate(handle, 1):
                total_lines = line_number
                digest.update(line_bytes)
                try:
                    row = json.loads(line_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise AdapterError(f"audit_row_not_json:{line_number}") from error
                if not isinstance(row, Mapping):
                    raise AdapterError(f"audit_row_not_object:{line_number}")
                if _is_eligible_audit_row(row):
                    eligible_count += 1
                    if len(selected) < max_rows:
                        selected.append(dict(row))
    except OSError as error:
        raise AdapterError("audit_file_unreadable") from error
    return selected, digest.hexdigest(), total_lines, eligible_count


def _diverse_order(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin source variants while preserving each variant's audit order."""
    buckets: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        variant = row.get("family")
        if not isinstance(variant, str):
            raise AdapterError("audit_family_missing")
        buckets.setdefault(variant, []).append(dict(row))
    result: list[dict[str, Any]] = []
    while True:
        emitted = False
        for variant in sorted(buckets):
            bucket = buckets[variant]
            if bucket:
                result.append(bucket.pop(0))
                emitted = True
        if not emitted:
            return result


def _source_ref(audit: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    required = ("file", "line", "raw_line_sha256", "source_sha256", "split")
    missing = [key for key in required if key not in audit]
    if missing:
        raise AdapterError("audit_reference_missing:" + ",".join(missing))
    row_id = audit.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise AdapterError("audit_row_id_missing")
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise AdapterError("raw_path_missing")
    return {
        "file": audit["file"],
        "line": audit["line"],
        "raw_line_sha256": audit["raw_line_sha256"],
        "source_sha256": audit["source_sha256"],
        "split": audit["split"],
        "row_id": row_id,
        "group_id": audit.get("group_id"),
        "document_role": LINEAGE,
        "source_constructor": SOURCE_CONSTRUCTOR,
        "target_convention": "suffix",
        "uri": f"file:///dat04b/batch/{row_id}/{path}",
        "document_version": 0,
    }


def _raw_from_cache(
    source_cache: VerifiedSourceCache, source_ref: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        raw_bytes = source_cache.line_bytes(source_ref)
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError("source_row_not_json") from error
    if not isinstance(raw, Mapping):
        raise AdapterError("source_row_not_object")
    return dict(raw)


def _target_text(raw: Mapping[str, Any]) -> str:
    value = raw.get("corpus_target")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    raise AdapterError("finish_source_target_text_missing")


def _row_ref(audit: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy source/package identity while keeping target text out of row_ref."""
    return {
        "row_id": audit.get("row_id"),
        "split": audit.get("split"),
        "family": "finish_block",
        "source_variant": audit.get("family"),
        "source": audit.get("source"),
        "source_file": audit.get("file"),
        "source_line": audit.get("line"),
        "raw_line_sha256": audit.get("raw_line_sha256"),
        "source_sha256": audit.get("source_sha256"),
        "canonical_row_sha256": audit.get("canonical_row_sha256"),
        "group_id": audit.get("group_id"),
        "package": raw.get("package"),
        "path": raw.get("path"),
        "kind": raw.get("kind"),
        "named_identity_tokens": list(audit.get("named_identity_tokens", [])),
        "license_evidence_present": audit.get("license_evidence_present"),
        "license_status": audit.get("license_status"),
        "license": raw.get("license"),
        "source_url": raw.get("source_url"),
        "upstream": raw.get("upstream"),
        "target_sha256": audit.get("target_sha256"),
    }


def _batch_exclusion(
    reason: str, *, row_ref: Mapping[str, Any], source_ref: Mapping[str, Any],
    batch_id: str, r_fragment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "batch_id": batch_id,
        "adapter_id": ADAPTER_VERSION,
        "source_role": "verified_train_or_dev_row",
        "document_role": LINEAGE,
        "source_identity": {
            "file": source_ref.get("file"),
            "line": source_ref.get("line"),
            "raw_line_sha256": source_ref.get("raw_line_sha256"),
            "source_sha256": source_ref.get("source_sha256"),
            "row_id": source_ref.get("row_id"),
        },
        "row_ref_sha256": _canonical_json_sha256(row_ref),
        "selection_source": {
            "availability": "source_builder_window",
            "lineage": LINEAGE,
        },
    }
    if r_fragment is not None:
        provenance["r_fragment"] = dict(r_fragment)
    return {
        "status": "excluded",
        "context": None,
        "target_body": [],
        "operation": None,
        "provenance": provenance,
        "reason": reason,
    }


def _source_support(
    source_cache: VerifiedSourceCache, source_ref: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "verified_stream_cache",
        "file": source_ref.get("file"),
        "line": source_ref.get("line"),
        "file_sha256": source_ref.get("source_sha256"),
        "raw_line_sha256": source_ref.get("raw_line_sha256"),
        "stream_count_at_conversion": source_cache.stream_count,
        "line_bytes_include_separator": True,
    }


def _validate_converted_packet(
    result: Mapping[str, Any], raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Check batch-specific invariants without treating source text as target."""
    context = result.get("context")
    if not isinstance(context, Mapping):
        raise AdapterError("converted_context_missing")
    if "full_prompt" in context or "model_target" in context or "target" in context:
        raise AdapterError("future_context_field_present")
    ignored = result.get("provenance", {}).get("ignored_future_fields", [])
    if "full_prompt" not in ignored:
        raise AdapterError("future_field_ignore_record_missing")
    selection = result.get("selection_source")
    if not isinstance(selection, Mapping):
        raise AdapterError("selection_source_missing")
    document_text = selection.get("document_text")
    if not isinstance(document_text, str):
        raise AdapterError("selection_source_document_text_missing")
    if _sha256_text(document_text) != selection.get("content_sha256"):
        raise AdapterError("selection_source_hash_mismatch")
    if selection.get("lineage") != LINEAGE:
        raise AdapterError("selection_source_lineage_mismatch")
    if raw.get("full_prompt") is not None:
        # Presence in raw is expected.  This marker proves the value was kept
        # out of the context; source code may legitimately share target text.
        return {"future_fields_excluded_from_context": True}
    return {"future_fields_excluded_from_context": True}


def _r_fragment_check(
    raw: Mapping[str, Any], validator: Mapping[str, Any],
) -> dict[str, Any]:
    if not validator.get("available"):
        raise AdapterError("canonical_fragment_validator_unavailable")
    if raw.get("family") != "finish_block":
        return {
            "validator": str(validator["path"]),
            "validator_sha256": validator.get("sha256"),
            "passed": False,
            "reason": "raw_family_not_finish_block",
        }
    prefix = raw.get("prefix")
    if not isinstance(prefix, str):
        return {
            "validator": str(validator["path"]),
            "validator_sha256": validator.get("sha256"),
            "passed": False,
            "reason": "prefix_not_text",
        }
    target_text = _target_text(raw)
    fragment = prefix + target_text + "}"
    passed = _canonical_fragment_clean(fragment)
    return {
        "validator": str(validator["path"]),
        "validator_sha256": validator.get("sha256"),
        "passed": passed,
        "body_fragment_boundary": "raw_prefix_plus_corpus_target_before_outer_brace",
        "outer_closing_brace_in_label": False,
        "prefix_sha256": _sha256_text(prefix),
        "corpus_target_sha256": _sha256_text(target_text),
    }


def _make_batch_id(audit_sha256: str, rows: Sequence[Mapping[str, Any]],
                   max_rows: int, profile_limit: int) -> str:
    material = {
        "version": BATCH_VERSION,
        "audit_sha256": audit_sha256,
        "row_ids": [row.get("row_id") for row in rows],
        "max_rows": max_rows,
        "profile_limit": profile_limit,
    }
    return "dat04b-" + _canonical_json_sha256(material)[:24]


def _write_packet(handle: Any, packet: Mapping[str, Any]) -> None:
    handle.write(_canonical_json(packet).encode("utf-8") + b"\n")


def materialize_completion_batch(
    audit_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    profile_limit: int = DEFAULT_PROFILE_LIMIT,
    expansion_min_converted: int = 1,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialise a deterministic, bounded completion packet JSONL."""
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise AdapterError("max_rows_must_be_between_1_and_5000")
    if type(profile_limit) is not int or not 1 <= profile_limit <= max_rows:
        raise AdapterError("profile_limit_must_be_between_1_and_max_rows")
    if type(expansion_min_converted) is not int or expansion_min_converted < 1:
        raise AdapterError("expansion_min_converted_invalid")
    audit = Path(audit_path)
    output = Path(output_path)
    started = time.perf_counter()
    selected, audit_sha256, audit_lines, eligible_count = _scan_audit(audit, max_rows)
    ordered = _diverse_order(selected)
    if not ordered:
        raise AdapterError("no_eligible_finish_block_rows")
    validator = _validator_identity()

    # Build refs from source identity only.  Reading raw rows occurs only after
    # prime() has verified the exact JSONL bytes and complete file hash.
    refs: list[dict[str, Any]] = []
    preliminary: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for audit_row in ordered:
        # Source path is needed to build the URI, but the raw row is not read
        # until after the source cache has been primed.  Use an identity-only
        # placeholder here; _source_ref does not inspect target values.
        placeholder = {"path": str(audit_row.get("source", "source-row"))}
        ref = _source_ref(audit_row, placeholder)
        refs.append(ref)
        preliminary.append((dict(audit_row), ref))
    cache_started = time.perf_counter()
    source_cache = VerifiedSourceCache(max_rows=max_rows)
    source_cache.prime(refs)
    cache_seconds = time.perf_counter() - cache_started

    batch_id = _make_batch_id(audit_sha256, ordered, max_rows, profile_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    reasons: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    selected_variant_counts: Counter[str] = Counter(
        str(row.get("family")) for row in ordered
    )
    packages: Counter[str] = Counter()
    profile_converted = 0
    profile_excluded = 0
    processed = 0
    converted = 0
    excluded = 0
    selected_limit = min(profile_limit, len(preliminary))
    process_limit = selected_limit
    expansion_decision = "pending_profile"
    profile_started = time.perf_counter()
    profile_elapsed = None
    with output.open("wb") as handle:
        for index, (audit_row, ref) in enumerate(preliminary):
            if index >= process_limit:
                break
            packet_ref: dict[str, Any] = _row_ref(
                audit_row, {"path": str(audit_row.get("source", "source-row"))},
            )
            try:
                raw = _raw_from_cache(source_cache, ref)
                packet_ref = _row_ref(audit_row, raw)
                raw_path = raw.get("path")
                if isinstance(raw_path, str) and raw_path:
                    # The identity cache is already primed; this URI update
                    # only gives callers the actual package-relative path.
                    ref["uri"] = f"file:///dat04b/batch/{ref['row_id']}/{raw_path}"
                variants[str(packet_ref.get("source_variant"))] += 1
                packages[str(packet_ref.get("package"))] += 1
                if raw.get("family") != "finish_block":
                    result = _batch_exclusion(
                        "raw_family_not_finish_block", row_ref=packet_ref,
                        source_ref=ref, batch_id=batch_id,
                    )
                else:
                    fragment = _r_fragment_check(raw, validator)
                    if not fragment.get("passed"):
                        result = _batch_exclusion(
                            "r_fragment_invalid", row_ref=packet_ref,
                            source_ref=ref, batch_id=batch_id,
                            r_fragment=fragment,
                        )
                    else:
                        result = convert_completion(
                            raw, ref, verified_source_cache=source_cache,
                        )
                        if result.get("status") == "converted":
                            _validate_converted_packet(result, raw)
                            result["provenance"]["batch_id"] = batch_id
                            result["provenance"]["source_support"] = _source_support(
                                source_cache, ref,
                            )
                            result["provenance"]["r_fragment"] = fragment
                            result["provenance"]["batch_validation"] = {
                                "future_fields_excluded_from_context": True,
                                "source_constructor": SOURCE_CONSTRUCTOR,
                                "full_text_label": True,
                            }
                        else:
                            result.setdefault("provenance", {})["batch_id"] = batch_id
                            result["provenance"]["source_support"] = _source_support(
                                source_cache, ref,
                            )
                            result["provenance"]["r_fragment"] = fragment
            except AdapterError as error:
                # A row-specific source/constructor failure remains an explicit
                # exclusion.  Source cache failures are raised before output
                # processing and therefore fail the batch as an integrity error.
                result = _batch_exclusion(
                    str(error), row_ref=packet_ref, source_ref=ref,
                    batch_id=batch_id,
                )
            if result.get("status") != "converted":
                result.setdefault("provenance", {})["source_support"] = _source_support(
                    source_cache, ref,
                )
            packet = {
                "row_ref": packet_ref,
                "family": "finish_block",
                "source_variant": audit_row.get("family"),
                "result": result,
            }
            _write_packet(handle, packet)
            processed += 1
            if result.get("status") == "converted":
                converted += 1
            else:
                excluded += 1
                reasons[str(result.get("reason", "unknown"))] += 1
            if processed == selected_limit:
                profile_converted = converted
                profile_excluded = excluded
                profile_elapsed = time.perf_counter() - profile_started
                if profile_converted >= expansion_min_converted:
                    process_limit = len(preliminary)
                    expansion_decision = "expanded_after_profile_converted"
                else:
                    expansion_decision = "stopped_profile_not_useful"
                    break
    profile_seconds = (
        profile_elapsed if profile_elapsed is not None
        else time.perf_counter() - profile_started
    )

    if expansion_decision == "pending_profile":
        # This branch means the selected set itself was shorter than the
        # profile limit.  It is already the complete bounded selection.
        profile_converted = converted
        profile_excluded = excluded
        expansion_decision = "selection_shorter_than_profile"

    total_seconds = time.perf_counter() - started
    output_sha256 = _file_sha256(output)
    source_variants = dict(sorted(variants.items()))
    package_counts = dict(sorted(packages.items()))
    source_file_counts: Counter[str] = Counter(str(ref["file"]) for ref in refs)
    source_file_hashes: dict[str, str] = {}
    for ref in refs:
        path = str(ref["file"])
        expected = str(ref["source_sha256"])
        previous = source_file_hashes.get(path)
        if previous is not None and previous != expected:
            raise AdapterError("source_file_hash_disagrees_in_batch")
        source_file_hashes[path] = expected
    summary: dict[str, Any] = {
        "batch_version": BATCH_VERSION,
        "batch_id": batch_id,
        "adapter_id": ADAPTER_VERSION,
        "audit": {
            "path": str(audit),
            "sha256": audit_sha256,
            "lines_read": audit_lines,
            "eligible_train_finish_rows": eligible_count,
            "eligibility": {
                "split": "train_group",
                "family_prefix": "finish_block",
                "license_evidence_present": True,
                "license_status": "direct_row_evidence",
                "named_identity_tokens_nonempty": True,
            },
        },
        "selection": {
            "candidate_pool_count": len(selected),
            "selected_count": len(ordered),
            "max_rows": max_rows,
            "profile_limit": profile_limit,
            "order": "round_robin_sorted_source_variant_preserving_audit_order",
            "source_variants": dict(sorted(selected_variant_counts.items())),
        },
        "profile_milestone": {
            "attempted": selected_limit,
            "converted": profile_converted,
            "excluded": profile_excluded,
            "conversion_rate": (
                profile_converted / selected_limit if selected_limit else 0.0
            ),
            "elapsed_seconds": profile_seconds,
            "expansion_min_converted": expansion_min_converted,
        },
        "expansion": {
            "decision": expansion_decision,
            "processed_count": processed,
            "remaining_unprocessed_count": len(ordered) - processed,
            "reason": (
                "profile produced at least one converted source-supported row"
                if expansion_decision == "expanded_after_profile_converted"
                else "profile threshold was not met or selection was shorter"
            ),
        },
        "result_counts": {
            "processed": processed,
            "converted": converted,
            "excluded": excluded,
            "exclusion_reasons": dict(sorted(reasons.items())),
            "processed_source_variants": source_variants,
        },
        "source_support": {
            "status": "verified_stream_cache",
            "files_referenced": len({str(ref["file"]) for ref in refs}),
            "files_streamed_once": source_cache.stream_count,
            "rows_loaded": source_cache.rows_loaded,
            "one_stream_per_file": True,
            "mutation_policy": "stat_identity_checked_before_and_after_stream_and_lookup",
            "cache_max_rows": max_rows,
            "cache_seconds": cache_seconds,
            "source_files": [
                {
                    "path": path,
                    "sha256": source_file_hashes[path],
                    "selected_rows": source_file_counts[path],
                }
                for path in sorted(source_file_hashes)
            ],
        },
        "by_package": package_counts,
        "validator": validator,
        "lineage": {
            "document_role": LINEAGE,
            "selection_source_availability": "source_builder_window",
            "observed_editor_trace_count": 0,
            "source_derived_simulated_count": converted,
            "target_is_out_of_band": True,
            "teacher_fields_ignored": ["full_prompt", "note", "instructions", "model_target"],
        },
        "output": {
            "path": str(output),
            "sha256": output_sha256,
            "jsonl_packets": processed,
            "full_text_labels": True,
            "target_truncation": False,
        },
        "timing_seconds": {
            "total": total_seconds,
            "seconds_per_processed_row": total_seconds / processed if processed else None,
            "seconds_per_converted_row": total_seconds / converted if converted else None,
        },
        "limits": [
            "No observed editor event trace is claimed; bounded audit had no such evidence field.",
            "No model generation, tokenisation, serving, or admission decision is performed.",
            "Source-derived simulated pre-edit windows use raw prefix only and are validated by source hash and constructor splice.",
            "Raw target fragments are retained only in converted result provenance/label; the outer closing brace is never added to target_body.",
        ],
    }
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(_canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--profile-limit", type=int, default=DEFAULT_PROFILE_LIMIT)
    args = parser.parse_args(argv)
    summary = materialize_completion_batch(
        args.audit, args.output, max_rows=args.max_rows,
        profile_limit=args.profile_limit, summary_path=args.summary,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
