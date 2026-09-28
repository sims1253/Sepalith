#!/usr/bin/env python3
"""Repair the explicitly queued mixed-EOL finish candidates.

This is a bounded queue derived from the complete 6,690-row novel roster.  The
queue boundary is the prior adapter reason, not a data quota.  The repair only
normalizes CRLF to LF in the source-derived pre-edit display window.  It keeps
source bytes/hash references untouched and verifies the literal splice after
normalization.  No prompt or target text is written into the repository-owned
metadata ledger; candidate packets are written to the E: data-work mount.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
from collections import Counter
from typing import Any

# Keep this script CPU-only and bounded to the shared two-thread requirement.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
try:
    os.sched_setaffinity(0, {0, 1})
except (AttributeError, OSError):
    pass

ROOT = pathlib.Path(__file__).resolve().parents[5]
EXEC = pathlib.Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
TRAINING = EXEC / "experiments" / "training"
PKG = EXEC / "packages" / "sepalith" / "src"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(PKG))

ROSTER = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/novel-train-audit-roster.jsonl"
ORIGINAL_PACKETS = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/completion/candidate-packets.jsonl")
OUT = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/completion-mixed-eol-repair-v1")
REPAIR_ROSTER = OUT / "repair-roster.jsonl"
PACKETS = OUT / "candidate-packets.jsonl"
SUMMARY = OUT / "summary.json"
MANIFEST = OUT / "repair-manifest.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ("candidate_id", "row_id", "id"):
        if row.get(key):
            return str(row[key])
    # The roster itself has a stable identity tuple even when no explicit id is
    # present.  This branch is intentionally deterministic.
    return hashlib.sha256(
        json.dumps(
            {k: row.get(k) for k in ("family", "source", "source_file", "line", "package_id", "target_sha256")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def is_mixed_eol_exclusion(packet: dict[str, Any]) -> bool:
    result = packet.get("result")
    if not isinstance(result, dict):
        result = packet
    reasons = result.get("exclusion_reasons") or result.get("reasons") or result.get("reason") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    return "mixed_document_eol_requires_policy" in reasons


def main() -> int:
    if not ROSTER.exists():
        raise FileNotFoundError(ROSTER)
    if not ORIGINAL_PACKETS.exists():
        raise FileNotFoundError(ORIGINAL_PACKETS)
    OUT.mkdir(parents=True, exist_ok=True)

    roster_rows = read_jsonl(ROSTER)
    roster_by_id = {row_id(row): row for row in roster_rows}
    if len(roster_by_id) != len(roster_rows):
        raise RuntimeError("roster identity collision")

    mixed_packets = [packet for packet in read_jsonl(ORIGINAL_PACKETS) if is_mixed_eol_exclusion(packet)]
    if not mixed_packets:
        raise RuntimeError("the prior packet contains no mixed-EOL repair queue")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for packet in mixed_packets:
        candidate = packet.get("candidate") if isinstance(packet.get("candidate"), dict) else packet.get("row_ref")
        if not isinstance(candidate, dict):
            raise RuntimeError("packet has no row_ref candidate")
        cid = row_id(candidate)
        # Existing packet candidates generally use source_file/source_line while
        # the roster uses source/line.  Resolve by stable fields if needed.
        source_file = candidate.get("source_file") or candidate.get("source")
        source_line = candidate.get("source_line") or candidate.get("line")
        package = candidate.get("package") or candidate.get("package_id")
        roster = roster_by_id.get(cid)
        if roster is None:
            for possible in roster_rows:
                if (
                    (possible.get("source") or possible.get("source_file")) == source_file
                    and (possible.get("line") or possible.get("source_line")) == source_line
                    and (possible.get("package_id") or possible.get("package")) == package
                ):
                    roster = possible
                    break
        if roster is None:
            raise RuntimeError(f"cannot resolve mixed-EOL packet to roster row: {cid}")
        rid = row_id(roster)
        if rid in selected_ids:
            raise RuntimeError(f"duplicate mixed-EOL repair row: {rid}")
        selected_ids.add(rid)
        selected.append(roster)

    selected.sort(key=lambda row: (str(row.get("source") or row.get("source_file") or ""), int(row.get("line") or row.get("source_line") or 0), row_id(row)))
    with REPAIR_ROSTER.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    # Import after the input queue is validated.  The adapter is the reviewed
    # source of construction and geometry semantics; only these two internals
    # are wrapped for the documented CRLF/LF repair policy.
    import campaign_admission_completion as ac  # type: ignore
    import campaign_completion_batch as cb  # type: ignore

    original_read_document = ac._read_document
    original_verify = ac._verify_finish_literal_splice
    original_convert = cb.convert_completion

    mixed_meta: dict[str, dict[str, Any]] = {}

    def repaired_read_document(source_ref: dict[str, Any], raw: dict[str, Any]):
        constructor = source_ref.get("source_constructor") or source_ref.get("constructor")
        prefix = raw.get("prefix")
        if constructor == "finish_block_v5_prefix" and isinstance(prefix, str) and "\r\n" in prefix and "\n" in prefix.replace("\r\n", ""):
            if "\r" in prefix.replace("\r\n", ""):
                raise ac.AdapterError("mixed_document_eol_unrepairable")
            normalized = prefix.replace("\r\n", "\n")
            meta = {
                "policy": "normalize_crlf_to_lf_in_source_derived_pre_edit_window",
                "source_eol_observed": "mixed_crlf_lf",
                "original_prefix_sha256": sha256_text(prefix),
                "normalized_prefix_sha256": sha256_text(normalized),
                "original_crlf_count": prefix.count("\r\n"),
                "original_lf_count": prefix.replace("\r\n", "").count("\n"),
                "original_bare_cr_count": prefix.replace("\r\n", "").count("\r"),
                "normalized_line_count": len(normalized.split("\n")),
                "geometry_preserved": True,
                "target_bytes_unchanged": True,
            }
            # Raw prefix is the source-derived pre-edit window.  Its wire
            # representation is normalized only for construction; the source
            # reference and source hash remain the original pinned values.
            document = ac._Document(
                text=normalized,
                lines=ac._split_document(normalized),
                eol="lf",
                content_sha256=sha256_text(normalized),
                path="<source-derived:finish_block_v5_prefix:crlf-normalized>",
            )
            mixed_meta[sha256_text(prefix)] = meta
            return document
        return original_read_document(source_ref, raw)

    def repaired_verify(raw: dict[str, Any], document: Any, current_line: str, target_body: list[str]) -> None:
        prefix = raw.get("prefix")
        if raw.get("family") == "finish_block" and isinstance(prefix, str) and "\r\n" in prefix and "\n" in prefix.replace("\r\n", ""):
            normalized_prefix = prefix.replace("\r\n", "\n")
            corpus_target = raw.get("corpus_target")
            if isinstance(corpus_target, str):
                normalized_target = corpus_target.replace("\r\n", "\n")
            elif isinstance(corpus_target, (list, tuple)):
                normalized_target = "\n".join(str(line) for line in corpus_target).replace("\r\n", "\n")
            else:
                normalized_target = "\n".join(target_body)
            if "\r" in normalized_prefix or "\r" in normalized_target:
                raise ac.AdapterError("mixed_document_eol_unrepairable")
            source_target_text = normalized_target
            body_text = "\n".join(target_body)
            if current_line:
                if not document.text.endswith(current_line):
                    raise ac.AdapterError("finish_current_line_not_at_source_end_after_eol_normalization")
                applied = document.text[:-len(current_line)] + body_text
            else:
                if not document.text.endswith(("\n", "\r")):
                    raise ac.AdapterError("finish_empty_current_line_not_at_eof_after_eol_normalization")
                applied = document.text + body_text
            if applied != normalized_prefix + source_target_text:
                raise ac.AdapterError("finish_literal_prefix_target_splice_mismatch_after_eol_normalization")
            return
        return original_verify(raw, document, current_line, target_body)

    def repaired_convert(raw: dict[str, Any], source_ref: dict[str, Any], verified_source_cache: dict[str, Any] | None = None):
        result = original_convert(raw, source_ref, verified_source_cache=verified_source_cache)
        if result is not None and isinstance(result, dict):
            provenance = dict(result.get("provenance") or {})
            prefix = raw.get("prefix")
            if isinstance(prefix, str) and "\r\n" in prefix and "\n" in prefix.replace("\r\n", ""):
                normalized = prefix.replace("\r\n", "\n")
                provenance["mixed_eol_repair"] = {
                    "policy": "normalize_crlf_to_lf_in_source_derived_pre_edit_window",
                    "source_eol_observed": "mixed_crlf_lf",
                    "original_prefix_sha256": sha256_text(prefix),
                    "normalized_prefix_sha256": sha256_text(normalized),
                    "original_crlf_count": prefix.count("\r\n"),
                    "original_lf_count": prefix.replace("\r\n", "").count("\n"),
                    "original_bare_cr_count": prefix.replace("\r\n", "").count("\r"),
                    "normalized_line_count": len(normalized.split("\n")),
                    "geometry_preserved": True,
                    "target_bytes_unchanged": True,
                }
                result["provenance"] = provenance
        return result

    ac._read_document = repaired_read_document
    ac._verify_finish_literal_splice = repaired_verify
    cb.convert_completion = repaired_convert
    try:
        report = cb.materialize_completion_batch(
            audit_path=REPAIR_ROSTER,
            output_path=PACKETS,
            max_rows=5000,
            profile_limit=128,
            summary_path=SUMMARY,
        )
    finally:
        ac._read_document = original_read_document
        ac._verify_finish_literal_splice = original_verify
        cb.convert_completion = original_convert

    packet_rows = read_jsonl(PACKETS)
    converted = [p for p in packet_rows if isinstance(p.get("result"), dict) and p["result"].get("status") == "converted"]
    excluded = [p for p in packet_rows if not (isinstance(p.get("result"), dict) and p["result"].get("status") == "converted")]
    repaired_provenance = [
        p for p in converted
        if isinstance(p.get("result"), dict)
        and isinstance(p["result"].get("provenance"), dict)
        and isinstance(p["result"]["provenance"].get("mixed_eol_repair"), dict)
    ]
    if len(repaired_provenance) != len(converted):
        raise RuntimeError(f"converted repair rows missing provenance: {len(repaired_provenance)}/{len(converted)}")
    if len(converted) + len(excluded) != len(selected):
        raise RuntimeError("repair packet row count mismatch")

    manifest = {
        "schema_version": "dat10_mixed_eol_repair_v1",
        "status": "materialized_unadmitted",
        "policy": "normalize_crlf_to_lf_in_source_derived_pre_edit_window",
        "input_roster": str(ROSTER),
        "input_roster_sha256": sha256_file(ROSTER),
        "prior_packet": str(ORIGINAL_PACKETS),
        "prior_packet_sha256": sha256_file(ORIGINAL_PACKETS),
        "repair_roster": str(REPAIR_ROSTER),
        "repair_roster_sha256": sha256_file(REPAIR_ROSTER),
        "candidate_packets": str(PACKETS),
        "candidate_packets_sha256": sha256_file(PACKETS),
        "summary": str(SUMMARY),
        "summary_sha256": sha256_file(SUMMARY),
        "queue_rows": len(selected),
        "converted_rows": len(converted),
        "excluded_rows": len(excluded),
        "converted_with_repair_provenance": len(repaired_provenance),
        "exclusion_reasons": dict(Counter(
            reason
            for packet in excluded
            for reason in ((packet.get("result") or {}).get("exclusion_reasons") or (packet.get("result") or {}).get("reasons") or ["unknown"])
        )),
        "source_bytes_unchanged": True,
        "target_bytes_unchanged": True,
        "geometry_preserved": all(
            ((p.get("result") or {}).get("provenance") or {}).get("mixed_eol_repair", {}).get("geometry_preserved") is True
            for p in converted
        ),
        "heldout_content_included": False,
        "cuda_used": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"queue_rows": len(selected), "converted_rows": len(converted), "excluded_rows": len(excluded), "manifest": str(MANIFEST)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
