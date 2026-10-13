#!/usr/bin/env python3
"""Audit the frozen roxy8597 candidate union against the authoritative TRAIN rows.

This is a read-only audit. It never rewrites the frozen candidate union or the
accepted stream.  It emits only a small token/provenance dedup ledger and
manifests to the E-backed output directory; source, prompt, and target text are
never copied to the output.
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-roxy8597-root-fixes-v1"
OUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-roxy8597-root-fixes-v1")
FROZEN_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-roxy8597-materialization-v1/manifest.json")
FROZEN_UNION = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-roxy8597-materialization-v1/candidate-union-token-rows.jsonl")
AUTHORITATIVE_ACCEPTED = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl")
OBSOLETE_ACCEPTED = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl")
PROTOCOL_SOURCE = PLAN / "docs/campaign/work/r2-cpt-global-stage-preparation-v1/source/packages/sepalith/src/sepalith/campaign_protocol.py"

EXPECTED_AUTHORITATIVE_SHA = "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"
EXPECTED_AUTHORITATIVE_ROWS = 15006
EXPECTED_FROZEN_UNION_ROWS = 8597
EXPECTED_VOCAB_SIZE = 130560
BOS_ID = 0
EOS_ID = 1
TERMINAL_TEXT = ">>>>>>> UPDATED"
TERMINAL_IDS = (97666, 24779, 651, 8089, 23877)
FORBIDDEN_TEXT_KEYS = {
    "prompt", "prompt_text", "target", "target_text", "target_body_text",
    "source_text", "source_snapshot_text",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def token_fingerprint(token_row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({
        "input_ids": token_row["input_ids"],
        "target_body_tokens": token_row["target_body_tokens"],
        "target_terminal_tokens": token_row["target_terminal_tokens"],
        "target_operation": token_row["target_operation"],
        "target_start": token_row["target_start"],
    })).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def update_status(status: str, **fields: Any) -> None:
    value = {
        "schema": "sepalith.dat10.roxy8597_root_fixes_status.v1",
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "cuda": False,
        "training_admission": False,
        "frozen_union_untouched": True,
    }
    value.update(fields)
    write_json(PACKET / "status.json", value)


def forbidden_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key in FORBIDDEN_TEXT_KEYS:
                found.append(child_path)
            found.extend(forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, f"{path}/{index}"))
    return found


def is_native_control(token: int) -> bool:
    return 0 <= token <= 7 or 10 <= token <= 21 or 130072 <= token < EXPECTED_VOCAB_SIZE


def validate_token_row(token_row: dict[str, Any], *, full_text: bool) -> list[str]:
    """Return protocol violations, including native BOS/EOS and terminal binding."""
    errors: list[str] = []
    required = {
        "bos_token_id", "eos_token_id", "input_ids", "prompt_token_count",
        "target_body_token_count", "target_body_tokens", "target_operation",
        "target_start", "target_terminal_token_count", "target_terminal_tokens",
        "target_token_count",
    }
    missing = sorted(required - set(token_row))
    if missing:
        return ["missing:" + ",".join(missing)]
    input_ids = token_row.get("input_ids")
    body = token_row.get("target_body_tokens")
    terminal = token_row.get("target_terminal_tokens")
    arrays = (("input_ids", input_ids), ("target_body_tokens", body), ("target_terminal_tokens", terminal))
    for name, values in arrays:
        if not isinstance(values, list):
            errors.append(name + ":not_list")
        elif any(type(token) is not int or not 0 <= token < EXPECTED_VOCAB_SIZE for token in values):
            errors.append(name + ":token_range_or_type")
    if not all(isinstance(values, list) for _, values in arrays):
        return errors
    # Check exact native wrapper independently of metadata fields.
    if token_row.get("bos_token_id") != BOS_ID:
        errors.append("metadata_bos_not_0")
    if token_row.get("eos_token_id") != EOS_ID:
        errors.append("metadata_eos_not_1")
    if not input_ids or input_ids[0] != BOS_ID:
        errors.append("input_bos_not_first_0")
    if not input_ids or input_ids[-1] != EOS_ID:
        errors.append("input_eos_not_last_1")
    if input_ids.count(BOS_ID) != 1:
        errors.append("input_bos_count_not_1")
    if input_ids.count(EOS_ID) != 1:
        errors.append("input_eos_count_not_1")
    if any(is_native_control(token) for token in input_ids[1:-1]):
        errors.append("input_native_control_internal")
    if tuple(terminal) != TERMINAL_IDS:
        errors.append("terminal_tokens_not_canonical")
    if token_row.get("target_terminal_token_count") != len(TERMINAL_IDS):
        errors.append("terminal_count_not_5")
    # Protocol geometry and exact body/terminal placement.
    integer_names = ("prompt_token_count", "target_body_token_count", "target_terminal_token_count", "target_start", "target_token_count")
    for name in integer_names:
        if type(token_row.get(name)) is not int:
            errors.append(name + ":not_integer")
    if not any(name + ":not_integer" in errors for name in integer_names):
        prompt_count = token_row["prompt_token_count"]
        body_count = token_row["target_body_token_count"]
        terminal_count = token_row["target_terminal_token_count"]
        target_count = token_row["target_token_count"]
        target_start = token_row["target_start"]
        if target_start != prompt_count + 1:
            errors.append("target_start_not_prompt_plus_bos")
        if body_count != len(body) or terminal_count != len(terminal):
            errors.append("declared_segment_count_mismatch")
        if target_count != body_count + terminal_count:
            errors.append("declared_target_count_mismatch")
        if len(input_ids) != target_start + target_count + 1:
            errors.append("input_length_not_bos_prompt_target_eos")
        if input_ids[target_start:target_start + body_count] != body:
            errors.append("body_segment_not_conserved")
        if input_ids[target_start + body_count:target_start + target_count] != terminal:
            errors.append("terminal_segment_not_conserved")
    if full_text:
        prompt_text = token_row.get("prompt_text")
        body_text = token_row.get("target_body_text")
        target_text = token_row.get("target_text")
        operation = token_row.get("target_operation")
        if not isinstance(prompt_text, str) or not prompt_text.endswith("\n"):
            errors.append("prompt_text_missing_final_lf")
        if not isinstance(body_text, str) or not isinstance(target_text, str):
            errors.append("full_text_fields_missing")
        elif operation == "replace":
            if not body_text or body_text == "[NO_EDIT]":
                errors.append("replace_body_invalid")
            if target_text != body_text + "\n" + TERMINAL_TEXT:
                errors.append("replace_target_terminal_binding")
        elif operation == "no_op":
            if body_text != "[NO_EDIT]" or target_text != "[NO_EDIT]\n" + TERMINAL_TEXT:
                errors.append("noop_target_terminal_binding")
        elif operation == "delete":
            if body_text != "" or target_text != TERMINAL_TEXT:
                errors.append("delete_target_terminal_binding")
        else:
            errors.append("unknown_target_operation")
    return errors


def run_targeted_controls() -> dict[str, Any]:
    """Small mutation controls prove each newly enforced guard trips."""
    base = {
        "bos_token_id": 0, "eos_token_id": 1, "input_ids": [0, 9, 22, *TERMINAL_IDS, 1],
        "prompt_token_count": 1, "target_body_token_count": 1, "target_body_tokens": [22],
        "target_operation": "replace", "target_start": 2,
        "target_terminal_token_count": 5, "target_terminal_tokens": list(TERMINAL_IDS),
        "target_token_count": 6,
    }
    cases: dict[str, tuple[dict[str, Any], str]] = {}
    cases["valid"] = (dict(base), "ok")
    bad = dict(base); bad["input_ids"] = [2, *base["input_ids"][1:]]; cases["bos_wrong"] = (bad, "input_bos_not_first_0")
    bad = dict(base); bad["input_ids"] = [0, 9, 22, *TERMINAL_IDS, 2]; cases["eos_wrong"] = (bad, "input_eos_not_last_1")
    bad = dict(base); bad["input_ids"] = [0, 0, 9, 22, *TERMINAL_IDS, 1]; bad["target_start"] = 3; cases["duplicate_bos"] = (bad, "input_bos_count_not_1")
    bad = dict(base); bad["input_ids"] = [0, 9, 22, *TERMINAL_IDS, 130560, 1]; bad["target_token_count"] = 7; cases["range_upper_bound"] = (bad, "input_ids:token_range_or_type")
    bad = dict(base); bad["target_terminal_tokens"] = [*TERMINAL_IDS[:-1], 23876]; bad["input_ids"] = [0, 9, 22, *bad["target_terminal_tokens"], 1]; cases["terminal_mismatch"] = (bad, "terminal_tokens_not_canonical")
    bad = dict(base); bad["input_ids"] = [0, 9, 10, *TERMINAL_IDS, 1]; bad["target_body_tokens"] = [10]; cases["native_control"] = (bad, "input_native_control_internal")
    results: dict[str, Any] = {}
    for name, (row, expected) in cases.items():
        errors = validate_token_row(row, full_text=False)
        results[name] = {"errors": errors, "passed": (not errors if expected == "ok" else expected in errors), "expected": expected}
    if not all(item["passed"] for item in results.values()):
        raise AssertionError("targeted_controls_failed:" + json.dumps(results, sort_keys=True))
    return {"count": len(results), "passed": len(results), "cases": results}


def audit_authoritative() -> tuple[dict[str, str], dict[str, list[str]], dict[str, Any]]:
    by_id: dict[str, str] = {}
    by_fp: dict[str, list[str]] = collections.defaultdict(list)
    errors = collections.Counter()
    valid_rows = 0
    digest = hashlib.sha256()
    rows = 0
    with AUTHORITATIVE_ACCEPTED.open("rb", buffering=4 * 1024 * 1024) as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_id = row.get("id")
            if not isinstance(row_id, str):
                errors["missing_id"] += 1
                continue
            if row_id in by_id:
                errors["duplicate_id"] += 1
                continue
            row_errors = validate_token_row(row, full_text=True)
            if not row_errors:
                valid_rows += 1
            for error in row_errors:
                errors[error] += 1
            fingerprint = token_fingerprint(row)
            by_id[row_id] = fingerprint
            by_fp[fingerprint].append(row_id)
    actual_sha = digest.hexdigest()
    if rows != EXPECTED_AUTHORITATIVE_ROWS:
        errors[f"row_count_expected_{EXPECTED_AUTHORITATIVE_ROWS}"] += 1
    if actual_sha != EXPECTED_AUTHORITATIVE_SHA:
        errors["authoritative_hash_mismatch"] += 1
    return by_id, dict(by_fp), {
        "path": str(AUTHORITATIVE_ACCEPTED), "rows": rows,
        "bytes": AUTHORITATIVE_ACCEPTED.stat().st_size, "sha256": actual_sha,
        "expected_sha256": EXPECTED_AUTHORITATIVE_SHA, "protocol_error_counts": dict(sorted(errors.items())),
        "protocol_valid_rows": valid_rows,
    }


def main() -> None:
    PACKET.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    required = [FROZEN_MANIFEST, FROZEN_UNION, AUTHORITATIVE_ACCEPTED, PROTOCOL_SOURCE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing_input:" + ",".join(missing))
    controls = run_targeted_controls()
    update_status("auditing_authoritative_accepted", authoritative_path=str(AUTHORITATIVE_ACCEPTED), targeted_controls=controls)
    authoritative_by_id, authoritative_by_fp, authoritative_pin = audit_authoritative()
    update_status("auditing_frozen_union_against_authoritative", authoritative=authoritative_pin)

    frozen_manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    old_pin = frozen_manifest.get("accepted_dedup_input", {})
    ledger_path = OUT / "authoritative-dedup-ledger.jsonl"
    ledger_tmp = ledger_path.with_name(ledger_path.name + f".{os.getpid()}.tmp")
    ledger_digest = hashlib.sha256()
    ledger_rows = 0
    frozen_digest = hashlib.sha256()
    frozen_rows = 0
    frozen_by_id: set[str] = set()
    candidate_protocol_errors = collections.Counter()
    dedup_counts = collections.Counter()
    source_counts = collections.Counter()
    length_buckets = collections.Counter()
    seen_fingerprints: dict[str, list[str]] = collections.defaultdict(list)
    internal_duplicate_groups: dict[str, list[str]] = collections.defaultdict(list)
    forbidden_count = 0
    provenance_hash_missing = 0
    native_valid_rows = 0
    with FROZEN_UNION.open("rb", buffering=4 * 1024 * 1024) as source, ledger_tmp.open("wb") as ledger:
        for line in source:
            frozen_digest.update(line)
            if not line.strip():
                continue
            frozen_rows += 1
            item = json.loads(line)
            row_id = item.get("row_id")
            frozen_by_id.add(row_id)
            token_row = item.get("token_row")
            errors = validate_token_row(token_row, full_text=False) if isinstance(token_row, dict) else ["token_row_missing"]
            for error in errors:
                candidate_protocol_errors[error] += 1
            if not errors:
                native_valid_rows += 1
            forbidden_count += len(forbidden_keys(item))
            provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
            target_hash = provenance.get("target_body_sha256")
            if not isinstance(target_hash, str) or len(target_hash) != 64:
                provenance_hash_missing += 1
            fingerprint = token_fingerprint(token_row) if isinstance(token_row, dict) and not errors else None
            accepted_matches = authoritative_by_fp.get(fingerprint, []) if fingerprint else []
            if row_id in authoritative_by_id:
                if fingerprint == authoritative_by_id[row_id]:
                    status = "accepted_id_exact_duplicate"
                else:
                    status = "accepted_id_payload_conflict"
            elif accepted_matches:
                status = "accepted_prompt_target_exact_duplicate"
            elif fingerprint and seen_fingerprints.get(fingerprint):
                status = "union_internal_exact_duplicate"
            else:
                status = "new_candidate"
            dedup_counts[status] += 1
            source_counts[item.get("union_source", "missing")] += 1
            if fingerprint:
                prior_ids = seen_fingerprints.get(fingerprint, [])
                if prior_ids:
                    internal_duplicate_groups[fingerprint] = prior_ids + [row_id]
                seen_fingerprints[fingerprint].append(row_id)
            geometry = item.get("geometry") or {}
            seq = geometry.get("sequence_tokens")
            target_body = geometry.get("target_body_tokens")
            if isinstance(seq, int):
                length_buckets["sequence_le_192" if seq <= 192 else "sequence_193_512" if seq <= 512 else "sequence_513_1024" if seq <= 1024 else "sequence_1025_4096" if seq <= 4096 else "sequence_gt_4096"] += 1
                length_buckets["sequence_gt_8192"] += seq > 8192
                length_buckets["sequence_gt_16384"] += seq > 16384
            if isinstance(target_body, int):
                length_buckets["target_body_le_192" if target_body <= 192 else "target_body_193_512" if target_body <= 512 else "target_body_513_1024" if target_body <= 1024 else "target_body_gt_1024"] += 1
            output = {
                "schema": "sepalith.dat10.roxy8597_authoritative_dedup_row.v1",
                "row_id": row_id,
                "union_source": item.get("union_source"),
                "dedup_status": status,
                "token_payload_sha256": fingerprint,
                "accepted_matching_ids": accepted_matches,
                "sequence_tokens": seq,
                "prompt_tokens": geometry.get("prompt_tokens"),
                "target_body_tokens": target_body,
                "target_tokens": geometry.get("target_tokens"),
                "target_body_sha256": target_hash,
                "source_text_written": False,
                "target_text_written": False,
            }
            payload = canonical(output) + b"\n"
            ledger.write(payload)
            ledger_digest.update(payload)
            ledger_rows += 1
    ledger_tmp.flush() if hasattr(ledger_tmp, "flush") else None
    # The with block closed the file before atomic publication.
    os.replace(ledger_tmp, ledger_path)
    frozen_sha = frozen_digest.hexdigest()
    if frozen_rows != EXPECTED_FROZEN_UNION_ROWS:
        raise ValueError(f"frozen_union_row_count:{frozen_rows}")
    if len(frozen_by_id) != frozen_rows:
        raise ValueError(f"frozen_union_duplicate_ids:{frozen_rows - len(frozen_by_id)}")
    if candidate_protocol_errors:
        raise ValueError("candidate_protocol_errors:" + json.dumps(dict(candidate_protocol_errors), sort_keys=True))
    if forbidden_count or provenance_hash_missing:
        raise ValueError(f"candidate_metadata_guard:{forbidden_count}:{provenance_hash_missing}")

    internal_groups = [sorted(ids) for ids in internal_duplicate_groups.values() if ids]
    manifest = {
        "schema": "sepalith.dat10.roxy8597_root_fixes_manifest.v1",
        "status": "complete_review_only_root_admission_required",
        "training_admission": False,
        "cuda": False,
        "frozen_union_untouched": True,
        "inputs": {
            "frozen_manifest": {"path": str(FROZEN_MANIFEST), "sha256": sha256_file(FROZEN_MANIFEST), "accepted_input_recorded_by_frozen_manifest": old_pin},
            "frozen_union": {"path": str(FROZEN_UNION), "rows": frozen_rows, "bytes": FROZEN_UNION.stat().st_size, "sha256": frozen_sha},
            "authoritative_accepted": authoritative_pin,
            "obsolete_accepted_path": {"path": str(OBSOLETE_ACCEPTED), "sha256_recorded_by_frozen_manifest": old_pin.get("sha256"), "status": "superseded_for_dedup"},
            "protocol_source": {"path": str(PROTOCOL_SOURCE), "sha256": sha256_file(PROTOCOL_SOURCE), "terminal_text": TERMINAL_TEXT, "terminal_ids": list(TERMINAL_IDS)},
        },
        "checks": {
            "targeted_controls": controls,
            "authoritative_protocol": authoritative_pin,
            "candidate_protocol": {
                "rows": frozen_rows, "native_valid_rows": native_valid_rows,
                "protocol_error_counts": dict(sorted(candidate_protocol_errors.items())),
                "forbidden_payload_text_key_count": forbidden_count,
                "provenance_target_body_sha_missing": provenance_hash_missing,
            },
            "dedup": {
                "counts": dict(sorted(dedup_counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "internal_exact_duplicate_groups": internal_groups,
                "length_buckets": dict(sorted(length_buckets.items())),
            },
        },
        "outputs": {
            "authoritative_dedup_ledger": {"path": str(ledger_path), "rows": ledger_rows, "bytes": ledger_path.stat().st_size, "sha256": ledger_digest.hexdigest()},
        },
        "policy": {
            "accepted_stream_for_dedup": "authoritative DAT10-finish-source-repair-v3 stream",
            "source_prompt_target_text_written": False,
            "native_bos_id": BOS_ID,
            "native_eos_id": EOS_ID,
            "token_id_range": [0, EXPECTED_VOCAB_SIZE - 1],
            "terminal_protocol": "target_terminal_tokens exactly encode >>>>>>> UPDATED; input_ids wraps one BOS0 and one EOS1",
            "no_training_admission": True,
        },
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(PACKET / "audit-result.json", {
        "schema": "sepalith.dat10.roxy8597_root_fixes_result.v1",
        "manifest_path": str(OUT / "manifest.json"),
        "manifest_sha256": sha256_file(OUT / "manifest.json"),
        "authoritative_sha256": authoritative_pin["sha256"],
        "frozen_union_sha256": frozen_sha,
        "dedup_counts": dict(sorted(dedup_counts.items())),
        "candidate_rows": frozen_rows,
        "accepted_rows": authoritative_pin["rows"],
        "targeted_controls": controls["passed"],
    })
    update_status("complete_review_only_root_admission_required", manifest_path=str(OUT / "manifest.json"), dedup_counts=dict(sorted(dedup_counts.items())), authoritative_sha256=authoritative_pin["sha256"], frozen_union_sha256=frozen_sha)
    print(json.dumps({"accepted": authoritative_pin, "candidate_rows": frozen_rows, "candidate_sha256": frozen_sha, "dedup_counts": dict(dedup_counts), "ledger_sha256": ledger_digest.hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            PACKET.mkdir(parents=True, exist_ok=True)
            update_status("failed", error=str(error), traceback=traceback.format_exc(limit=8))
        finally:
            raise
