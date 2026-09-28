"""Build the finite DAT-06 schedule for the corrected SFT candidate.

This is a CPU-only, metadata-driven wrapper around the reviewed immutable
``campaign_sampling`` implementation.  It requires a complete, already
tokenized corrected row file.  It never edits the original train rows,
sampler metadata, overlay, or DEV panel.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PLAN_ROOT = HERE.parents[1]
DEFAULT_BASE_METADATA = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/SFT-inputs-v1/sampler-metadata.json"
)
DEFAULT_OVERLAY = PLAN_ROOT / "work/finish-target-overlay-v1/finish-target-overlay.jsonl"
DEFAULT_OVERLAY_RECEIPT = PLAN_ROOT / "receipts/DAT-04-finish-target-overlay-v1-preparation.json"
SAMPLER_SOURCE = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/"
    "snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/"
    "source/experiments/training/campaign_sampling.py"
)

EXPECTED_BASE_METADATA_SHA256 = "56ab7f6b8bb4dd01a83292686ca2be6639581dbc49d521b79bbf499e9f037877"
EXPECTED_BASE_TOKEN_ROWS_SHA256 = "7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6"
EXPECTED_OVERLAY_SHA256 = "03cff340de0c2b0e641a0399ddc8de727c39c9aa5e51aeed984c6420117054cf"
EXPECTED_OVERLAY_RECEIPT_SHA256 = "c88e79f55d0841da56a56d115a5e024756666ab9bd171ee73b2897b099d8c45f"
EXPECTED_SAMPLER_SOURCE_SHA256 = "60e4e4d60b99a4b9851b3e35194f81fb2d0a2a3c47d0cac931c2091819c03dd0"
SPLIT_ID = "DAT-02-global-v2-285001f3d93e9f1871df"
SEED = 3407
MAX_STEPS = 200
EFFECTIVE_BATCH = 16
SHORT_MAX_TOKENS = 2048
LONG_MAX_TOKENS = 4096
EXPECTED_BASE_ROWS = 11764
EXPECTED_CORRECTED_ROWS = 11526
EXPECTED_ELIGIBLE = 4051
EXPECTED_REJECTED = 238
EXPECTED_ABSENT = 711
EXPECTED_NONFINISH = 7475
EXPECTED_NOOP_ROWS = 1140


class SchedulePreparationError(ValueError):
    """Raised when a corrected schedule input or output is unsafe to use."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise SchedulePreparationError(f"cannot read JSON {path}: {error}") from error


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SchedulePreparationError(f"{label} is missing: {path}")


def _load_sampler_module():
    _require_file(SAMPLER_SOURCE, "immutable sampler source")
    actual = sha256_file(SAMPLER_SOURCE)
    if actual != EXPECTED_SAMPLER_SOURCE_SHA256:
        raise SchedulePreparationError(
            f"sampler source hash mismatch: expected {EXPECTED_SAMPLER_SOURCE_SHA256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("dat06_campaign_sampling", SAMPLER_SOURCE)
    if spec is None or spec.loader is None:
        raise SchedulePreparationError("cannot load immutable sampler source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _walk_receipt(value: Any, path: str = "") -> Iterable[tuple[str, str, str]]:
    """Yield scalar receipt fields without retaining payload text."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(child, str):
                yield child_path, str(key), child
            else:
                yield from _walk_receipt(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_receipt(child, f"{path}[{index}]")


def _receipt_evidence(path: Path) -> dict[str, Any]:
    _require_file(path, "corrected-row receipt")
    value = _read_json(path)
    fields = list(_walk_receipt(value))
    hashes = sorted({item[2] for item in fields if "sha256" in item[1].lower() and len(item[2]) == 64})
    paths = sorted({item[2] for item in fields if item[1].lower() in {"path", "file", "rows_path", "token_rows_path"}})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "declared_sha256_fields": hashes,
        "declared_paths": paths,
        "status_values": sorted({item[2] for item in fields if item[1].lower() == "status"}),
    }


def _load_overlay_ids(overlay: Path, receipt: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    _require_file(overlay, "finish overlay")
    _require_file(receipt, "overlay receipt")
    receipt_hash = sha256_file(receipt)
    if receipt_hash != EXPECTED_OVERLAY_RECEIPT_SHA256:
        raise SchedulePreparationError(
            f"overlay receipt hash mismatch: expected {EXPECTED_OVERLAY_RECEIPT_SHA256}, got {receipt_hash}"
        )
    digest = hashlib.sha256()
    seen: set[str] = set()
    eligible: set[str] = set()
    rejected: set[str] = set()
    packet_rows = 0
    matched = 0
    status_counts: Counter[str] = Counter()
    with overlay.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except ValueError as error:
                raise SchedulePreparationError(f"overlay line {line_number} is not JSON") from error
            packet_rows += 1
            key = record.get("key", {})
            row_id = key.get("original_id")
            if not isinstance(row_id, str) or not row_id:
                raise SchedulePreparationError(f"overlay line {line_number} has no original_id")
            if row_id in seen:
                raise SchedulePreparationError(f"duplicate overlay ID {row_id}")
            seen.add(row_id)
            status = record.get("status")
            status_counts[str(status)] += 1
            lineage = record.get("lineage", {})
            is_matched = lineage.get("matched_train_identity") is True
            if is_matched:
                matched += 1
            strict = record.get("eligibility", {}).get("strict_v2") is True
            if status == "eligible_repaired" and strict:
                eligible.add(row_id)
            elif is_matched:
                rejected.add(row_id)
    actual = digest.hexdigest()
    if actual != EXPECTED_OVERLAY_SHA256:
        raise SchedulePreparationError(
            f"overlay hash mismatch: expected {EXPECTED_OVERLAY_SHA256}, got {actual}"
        )
    if (packet_rows, matched, len(eligible), len(rejected)) != (
        5000, 4289, EXPECTED_ELIGIBLE, EXPECTED_REJECTED
    ):
        raise SchedulePreparationError(
            "overlay counts do not match the accepted receipt: "
            f"packet={packet_rows} matched={matched} eligible={len(eligible)} rejected={len(rejected)}"
        )
    if eligible & rejected:
        raise SchedulePreparationError("overlay ID is both eligible and rejected")
    return eligible, rejected, {
        "path": str(overlay),
        "sha256": actual,
        "receipt_path": str(receipt),
        "receipt_sha256": receipt_hash,
        "packet_rows": packet_rows,
        "matched_train_rows": matched,
        "eligible_strict_v2": len(eligible),
        "rejected_matched": len(rejected),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _base_metadata(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    _require_file(path, "base sampler metadata")
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_BASE_METADATA_SHA256:
        raise SchedulePreparationError(
            f"base sampler metadata hash mismatch: expected {EXPECTED_BASE_METADATA_SHA256}, got {actual_hash}"
        )
    value = _read_json(path)
    if not isinstance(value, list) or len(value) != EXPECTED_BASE_ROWS:
        raise SchedulePreparationError("base sampler metadata must contain 11,764 records")
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(value, 1):
        if not isinstance(record, dict) or not isinstance(record.get("row_id"), str):
            raise SchedulePreparationError(f"base sampler metadata line {index} has no row_id")
        row_id = record["row_id"]
        if row_id in by_id:
            raise SchedulePreparationError(f"duplicate base sampler row_id {row_id}")
        if record.get("split") != "train" or record.get("admitted") is not True:
            raise SchedulePreparationError(f"base sampler row {row_id} is not admitted train data")
        by_id[row_id] = record
    return value, by_id, actual_hash


def _row_id(record: dict[str, Any]) -> str:
    value = record.get("id", record.get("row_id"))
    if not isinstance(value, str) or not value:
        raise SchedulePreparationError("corrected token row has no id")
    return value


def _updated_sampler_record(
    row: dict[str, Any],
    original: dict[str, Any],
    *,
    correction_status: str,
) -> dict[str, Any]:
    row_id = _row_id(row)
    input_ids = row.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids or any(type(token) is not int for token in input_ids):
        raise SchedulePreparationError(f"{row_id}: input_ids must be a non-empty integer list")
    if input_ids[0] != 0 or input_ids[-1] != 1:
        raise SchedulePreparationError(f"{row_id}: expected manual BOS 0 and terminal EOS 1")
    if len(input_ids) > LONG_MAX_TOKENS:
        raise SchedulePreparationError(f"{row_id}: sequence exceeds {LONG_MAX_TOKENS} tokens")
    target_start = row.get("target_start")
    if type(target_start) is not int or not 1 <= target_start < len(input_ids):
        raise SchedulePreparationError(f"{row_id}: invalid target_start")
    if row.get("split") != "train":
        raise SchedulePreparationError(f"{row_id}: corrected row is not train")
    if row.get("family") != original.get("family") or row.get("package_id") != original.get("package_id"):
        raise SchedulePreparationError(f"{row_id}: family/package identity changed")
    if "source_id" in row and row.get("source_id") != original.get("source_id"):
        raise SchedulePreparationError(f"{row_id}: source identity changed")
    if row.get("prompt_token_count") is not None and row["prompt_token_count"] != target_start - 1:
        raise SchedulePreparationError(f"{row_id}: prompt token count does not match target_start")
    body_tokens = row.get("target_body_tokens")
    terminal_tokens = row.get("target_terminal_tokens")
    if not isinstance(body_tokens, list) or not isinstance(terminal_tokens, list):
        raise SchedulePreparationError(f"{row_id}: tokenized target components are missing")
    if row.get("target_body_token_count") is not None and row["target_body_token_count"] != len(body_tokens):
        raise SchedulePreparationError(f"{row_id}: target body token count mismatch")
    if row.get("target_terminal_token_count") is not None and row["target_terminal_token_count"] != len(terminal_tokens):
        raise SchedulePreparationError(f"{row_id}: target terminal token count mismatch")
    if row.get("target_token_count") is not None and row["target_token_count"] != len(body_tokens) + len(terminal_tokens):
        raise SchedulePreparationError(f"{row_id}: target token count mismatch")
    if row.get("prompt_truncated") is True or row.get("target_truncated") is True:
        raise SchedulePreparationError(f"{row_id}: truncation flag is set")
    if row.get("bos_token_id", 0) != 0 or row.get("eos_token_id", 1) != 1:
        raise SchedulePreparationError(f"{row_id}: tokenizer special-token identity changed")
    if row.get("tokenizer_json_sha256") not in (None, "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"):
        raise SchedulePreparationError(f"{row_id}: tokenizer hash changed")

    prompt_tokens = target_start
    target_tokens = len(input_ids) - target_start
    total_tokens = len(input_ids)
    if prompt_tokens + target_tokens != total_tokens:
        raise SchedulePreparationError(f"{row_id}: token geometry does not close")
    length_bucket = "long" if total_tokens > SHORT_MAX_TOKENS else "short"
    naturally_long = length_bucket == "long"

    if correction_status == "unchanged":
        old_geometry = (
            original.get("prompt_tokens"), original.get("target_tokens"), original.get("total_tokens"),
            original.get("length_bucket"), original.get("naturally_long"),
        )
        new_geometry = (prompt_tokens, target_tokens, total_tokens, length_bucket, naturally_long)
        if old_geometry != new_geometry:
            raise SchedulePreparationError(f"{row_id}: unchanged row token geometry drifted")
    elif prompt_tokens != original.get("prompt_tokens"):
        raise SchedulePreparationError(f"{row_id}: corrected target changed the prompt token prefix")

    record = {
        "admitted": True,
        "family": original["family"],
        "length_bucket": length_bucket,
        "naturally_long": naturally_long,
        "operation": original["operation"],
        "package_id": original["package_id"],
        "prompt_tokens": prompt_tokens,
        "provenance": original.get("provenance", "") + "|finish_target_overlay_v1",
        "row_id": row_id,
        "semantic_noop": original["semantic_noop"],
        "source_id": original["source_id"],
        "source_kind": original["source_kind"],
        "split": "train",
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "correction_status": correction_status,
    }
    return record


def load_corrected_metadata(
    rows_path: Path,
    expected_sha256: str,
    base_by_id: dict[str, dict[str, Any]],
    eligible: set[str],
    rejected: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read corrected rows once, hash them, and retain metadata only."""

    _require_file(rows_path, "corrected token rows")
    if len(expected_sha256) != 64:
        raise SchedulePreparationError("expected corrected row-file hash must be a SHA256 digest")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise SchedulePreparationError("expected corrected row-file hash is not hexadecimal") from error

    digest = hashlib.sha256()
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    correction_counts: Counter[str] = Counter()
    length_changed = 0
    with rows_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
            except ValueError as error:
                raise SchedulePreparationError(f"corrected rows line {line_number} is not JSON") from error
            if not isinstance(row, dict):
                raise SchedulePreparationError(f"corrected rows line {line_number} is not an object")
            row_id = _row_id(row)
            if row_id in seen:
                raise SchedulePreparationError(f"duplicate corrected row_id {row_id}")
            seen.add(row_id)
            original = base_by_id.get(row_id)
            if original is None:
                raise SchedulePreparationError(f"corrected row {row_id} is absent from admitted base train metadata")
            if row_id in rejected:
                raise SchedulePreparationError(f"rejected row {row_id} must be physically excluded")
            status = "eligible_repaired" if row_id in eligible else "unchanged"
            record = _updated_sampler_record(row, original, correction_status=status)
            records.append(record)
            correction_counts[status] += 1
            if status == "eligible_repaired" and (
                record["target_tokens"], record["total_tokens"]
            ) != (original["target_tokens"], original["total_tokens"]):
                length_changed += 1
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SchedulePreparationError(
            f"corrected row-file hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    expected_ids = set(base_by_id) - rejected
    if seen != expected_ids:
        missing = len(expected_ids - seen)
        extra = len(seen - expected_ids)
        raise SchedulePreparationError(
            f"corrected row ID set mismatch: rows={len(seen)} expected={len(expected_ids)} missing={missing} extra={extra}"
        )
    if len(records) != EXPECTED_CORRECTED_ROWS or correction_counts != Counter(
        {"eligible_repaired": EXPECTED_ELIGIBLE, "unchanged": EXPECTED_NONFINISH}
    ):
        raise SchedulePreparationError(
            f"corrected row counts mismatch: {dict(correction_counts)}"
        )
    if sum(record["semantic_noop"] for record in records) != EXPECTED_NOOP_ROWS:
        raise SchedulePreparationError("corrected row file does not retain all 1,140 no-op rows")
    return records, {
        "path": str(rows_path),
        "sha256": actual_sha256,
        "records": len(records),
        "eligible_repaired": correction_counts["eligible_repaired"],
        "unchanged_nonfinish": correction_counts["unchanged"],
        "rejected_physically_excluded": len(rejected),
        "length_changed_eligible_rows": length_changed,
        "semantic_noop_rows": sum(record["semantic_noop"] for record in records),
        "prompt_prefix_policy": "corrected rows must preserve original prompt token count; target geometry is recomputed from token IDs",
    }


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise SchedulePreparationError(f"refusing to overwrite existing output: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _prefix_step_summary(draws: list[dict[str, Any]], step: int) -> dict[str, Any]:
    count = step * EFFECTIVE_BATCH
    prefix = draws[:count]
    families = Counter(item["family"] for item in prefix)
    return {
        "optimizer_step": step,
        "draws": len(prefix),
        "semantic_noop": sum(item["semantic_noop"] for item in prefix),
        "semantic_noop_fraction": sum(item["semantic_noop"] for item in prefix) / count,
        "naturally_long": sum(item["naturally_long"] for item in prefix),
        "families": dict(sorted(families.items())),
        "distinct_rows": len({item["row_id"] for item in prefix}),
    }


def build_schedule(
    *,
    rows_path: Path,
    rows_sha256: str,
    rows_receipt: Path,
    base_metadata_path: Path = DEFAULT_BASE_METADATA,
    overlay_path: Path = DEFAULT_OVERLAY,
    overlay_receipt_path: Path = DEFAULT_OVERLAY_RECEIPT,
    output_dir: Path = HERE,
) -> dict[str, Any]:
    sampler = _load_sampler_module()
    base_values, base_by_id, base_metadata_sha256 = _base_metadata(base_metadata_path)
    eligible, rejected, overlay_evidence = _load_overlay_ids(overlay_path, overlay_receipt_path)
    if not rejected <= set(base_by_id):
        raise SchedulePreparationError("a rejected matched ID is absent from the base train metadata")
    if len(set(base_by_id) - rejected - eligible) != EXPECTED_NONFINISH:
        raise SchedulePreparationError("corrected train population does not resolve to 4,051 + 7,475 rows")
    records, row_evidence = load_corrected_metadata(
        rows_path, rows_sha256, base_by_id, eligible, rejected,
    )
    receipt_evidence = _receipt_evidence(rows_receipt)
    sampler_records = [{key: value for key, value in record.items() if key != "correction_status"} for record in records]
    manifest = sampler.build_draw_manifest(
        sampler_records,
        max_steps=MAX_STEPS,
        effective_batch=EFFECTIVE_BATCH,
        split_id=SPLIT_ID,
        seed=SEED,
        token_rows_sha256=rows_sha256,
        requested_draws=MAX_STEPS * EFFECTIVE_BATCH,
        noop_fraction=0.10,
        family_ceiling=0.25,
        small_pack_cap=3,
        ordinary_replay_cap=8,
        naturally_long_fraction=0.20,
        short_max_tokens=SHORT_MAX_TOKENS,
        long_max_tokens=LONG_MAX_TOKENS,
    )
    sampler.validate_draw_manifest(manifest)
    if manifest.get("status") != "complete" or manifest.get("draw_count") != 3200:
        raise SchedulePreparationError("corrected 200-step schedule is not complete")
    if set(manifest["row_ids"]) & rejected:
        raise SchedulePreparationError("schedule contains a physically excluded rejected row")

    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    metadata_sidecar = {
        "schema_version": "sepalith.dat06.corrected-sampler-metadata.v1",
        "status": "complete_metadata_for_schedule",
        "base_metadata": {
            "path": str(base_metadata_path),
            "sha256": base_metadata_sha256,
            "base_token_rows_sha256": EXPECTED_BASE_TOKEN_ROWS_SHA256,
            "records": len(base_values),
        },
        "corrected_rows": row_evidence,
        "overlay": overlay_evidence,
        "rejected_ids": sorted(rejected),
        "eligible_repaired_ids": sorted(eligible),
        "records": records,
        "policy": {
            "seed": SEED,
            "max_steps": MAX_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "requested_draws": MAX_STEPS * EFFECTIVE_BATCH,
            "split_id": SPLIT_ID,
            "noop_fraction": 0.10,
            "family_ceiling": 0.25,
            "small_pack_cap": 3,
            "ordinary_replay_cap": 8,
            "naturally_long_fraction_ceiling": 0.20,
            "short_max_tokens": SHORT_MAX_TOKENS,
            "long_max_tokens": LONG_MAX_TOKENS,
            "truncation": "forbidden",
        },
        "source": {
            "campaign_sampling_sha256": EXPECTED_SAMPLER_SOURCE_SHA256,
            "sampler_input": "metadata rebuilt from complete corrected token rows; no prompt or target text is retained here",
        },
    }
    sidecar_bytes = json.dumps(metadata_sidecar, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output_dir = output_dir.resolve()
    schedule_path = output_dir / "finish-corrected-draws-200.json"
    sidecar_path = output_dir / "finish-corrected-sampler-metadata.json"
    _write_once(schedule_path, manifest_bytes)
    try:
        _write_once(sidecar_path, sidecar_bytes)
    except Exception:
        schedule_path.unlink(missing_ok=True)
        raise

    family_counts = Counter(item["family"] for item in manifest["draws"])
    summary = {
        "status": "complete",
        "schedule_path": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "corrected_rows": row_evidence,
        "overlay": overlay_evidence,
        "corrected_row_receipt": receipt_evidence,
        "base_metadata_sha256": base_metadata_sha256,
        "token_rows_sha256": rows_sha256,
        "draw_count": manifest["draw_count"],
        "max_steps": manifest["max_steps"],
        "effective_batch": manifest["effective_batch"],
        "family_draws": dict(sorted(family_counts.items())),
        "semantic_noop": manifest["achieved_mixture"]["semantic_noop"],
        "length": manifest["achieved_mixture"]["length"],
        "finite_caps": {
            "family_ceiling": manifest["policy"]["effective_family_ceiling"],
            "ordinary_replay_cap": manifest["policy"]["ordinary_replay_cap"],
            "small_pack_replay_cap": manifest["policy"]["small_pack_replay_cap"],
            "naturally_long_slots": manifest["policy"]["naturally_long_slots"],
        },
        "prefix_steps": [_prefix_step_summary(manifest["draws"], step) for step in (50, 100, 200)],
        "source_policy_preserved": True,
        "excluded_rejected_ids_in_schedule": 0,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="complete corrected token-row JSONL")
    parser.add_argument("--rows-sha256", required=True, help="SHA256 of --rows")
    parser.add_argument("--rows-receipt", type=Path, required=True, help="receipt binding the corrected row artifact")
    parser.add_argument("--base-metadata", type=Path, default=DEFAULT_BASE_METADATA)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--overlay-receipt", type=Path, default=DEFAULT_OVERLAY_RECEIPT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_schedule(
            rows_path=args.rows,
            rows_sha256=args.rows_sha256,
            rows_receipt=args.rows_receipt,
            base_metadata_path=args.base_metadata,
            overlay_path=args.overlay,
            overlay_receipt_path=args.overlay_receipt,
            output_dir=args.output_dir,
        )
    except SchedulePreparationError as error:
        print(json.dumps({"status": "rejected", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
