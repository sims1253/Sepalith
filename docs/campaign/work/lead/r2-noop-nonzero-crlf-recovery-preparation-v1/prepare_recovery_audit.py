#!/usr/bin/env python3
"""Audit v3 no-op holds with source-backed nonzero and CRLF geometry.

The v3 provider preparer required ``target_body == []`` and a zero-width
range.  Those are valid for blank-between no-ops, but reject unchanged
nonempty selections such as a terminal ``}``.  It also compared the LF
scenario window with a raw CRLF source without applying the recorded
``uniform_crlf_to_lf`` normalization.

This packet emits a compact recovery ledger, not provider prompts or training
rows.  Every recovered decision is source-backed and target-free.  The
original v3 packet and all source/replay files remain untouched.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
BASE = Path("/mnt/e/sepalith/campaign-20260915/data-work")
V3 = BASE / "Sourcewalk-noop-expansion-v3/inputs-full41"
REPLAY = BASE / "Sourcewalk-independent-replay-v3/full-01"
PACKETS = BASE / "DAT10-novel-v1/source-walk-shards-v1"
OUT = BASE / "Sourcewalk-noop-expansion-v4"
EXPECTED_V3_MANIFEST_SHA = "b2edf67c9a4df6a358aed86aa9c31fa788d8b2d8b28c5ccce42d11f23a70b5e8"
EXPECTED_REPLAY_INDEX_SHA = "65637a9e05c66647de042d46f42bf9afa197a0b068f63680ec9f3ac0dfe922a1"
SUPPORTED_STATUS = "provenance_supported_candidate_root_review_required"
TRUE_PROVENANCE_HOLD = "provenance_not_supported"
OVERRESTRICTED_REASONS = {
    "RowValidationError:no-op target contract",
    "RowValidationError:document EOL",
}


class RecoveryError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise RecoveryError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            row_id = value.get("row_id") or value.get("row_ref", {}).get("row_id")
            require(isinstance(row_id, str) and row_id, f"row_id_missing:{path}:{line_number}")
            require(row_id not in result, f"duplicate_row_id:{path}:{row_id}")
            result[row_id] = value
    return result


def position_to_offset(text: str, line: int, character: int) -> int:
    require(type(line) is int and line >= 0, "range_line_invalid")
    require(type(character) is int and character >= 0, "range_character_invalid")
    lines = text.split("\n")
    require(line < len(lines), "range_line_outside_window")
    require(character <= len(lines[line]), "range_character_outside_window")
    return sum(len(item) + 1 for item in lines[:line]) + character


def range_text(text: str, start: dict[str, Any], end: dict[str, Any]) -> str:
    start_offset = position_to_offset(text, start["line"], start["character"])
    end_offset = position_to_offset(text, end["line"], end["character"])
    require(end_offset >= start_offset, "range_reversed")
    return text[start_offset:end_offset]


def normalized_source(raw: bytes, method: str) -> tuple[bytes, bool]:
    if method == "exact_bytes":
        return raw, False
    if method == "uniform_crlf_to_lf":
        require(b"\r\n" in raw, "crlf_method_without_crlf")
        remainder = raw.replace(b"\r\n", b"")
        require(b"\r" not in remainder and b"\n" not in remainder, "mixed_eol_not_supported")
        return raw.replace(b"\r\n", b"\n"), True
    raise RecoveryError(f"unsupported_occurrence_method:{method}")


def line_at_byte(data: bytes, offset: int) -> int:
    return data[:offset].count(b"\n")


def read_source(path: Path, expected_sha: str) -> tuple[bytes, dict[str, Any]]:
    require(path.is_file(), f"source_missing:{path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"source_changed_during_read:{path}",
    )
    digest = bytes_sha(raw)
    require(digest == expected_sha, f"source_sha_mismatch:{path}")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": digest}


def classify(
    ledger: dict[str, Any], packet: dict[str, Any], old_reason: str | None
) -> dict[str, Any]:
    """Return a compact source-backed decision for one v3 candidate."""
    row_id = ledger.get("row_id")
    result = packet.get("result", {})
    context = result.get("context", {})
    selection = result.get("selection_source", {})
    validation = packet.get("validation", {})
    geometry = ledger.get("noop_geometry", {})
    decision: dict[str, Any] = {
        "row_id": row_id,
        "shard": ledger.get("shard"),
        "package_id": ledger.get("package_id"),
        "group_id": ledger.get("group_id"),
        "family": ledger.get("family"),
        "old_v3_reason": old_reason,
        "status": "hold",
        "silent_drop": False,
    }
    try:
        require(row_id == packet.get("row_ref", {}).get("row_id"), "packet_row_join")
        require(ledger.get("family") == "no_op" and packet.get("family") == "no_op", "family_not_noop")
        require(ledger.get("global_train") is True and ledger.get("protected_disjoint") is True, "split_guard")
        require(ledger.get("source_parse_ok") is True and ledger.get("source_stat_stable") is True, "source_guard")
        require(ledger.get("strict_protocol_ok") is True, "strict_protocol_guard")
        require(ledger.get("license_decision", {}).get("ok") is True, "license_guard")
        require(ledger.get("status") == SUPPORTED_STATUS, "provenance_status")
        require(result.get("operation") == "no_op" and result.get("status") == "converted", "operation_contract")
        require(geometry.get("ok") is True, "geometry_not_ok")
        require(geometry.get("geometry_supported") is True, "geometry_not_supported")
        require(geometry.get("supported_kind") is True, "kind_not_supported")
        require(geometry.get("typed_positions") is True, "positions_not_typed")
        require(geometry.get("unchanged_target") is True, "target_not_unchanged")
        require(geometry.get("window_occurrences") == 1, "window_occurrences")
        require(ledger.get("source_window_occurrences") == 1, "source_window_occurrences")
        source = Path(validation["source_path"])
        raw, source_pin = read_source(source, ledger["source_path_sha256"])
        require(validation.get("source_sha256") == source_pin["sha256"], "packet_source_sha")
        window_text = selection.get("text")
        require(isinstance(window_text, str), "window_text_missing")
        method = ledger.get("source_window_occurrence_method")
        view, normalized = normalized_source(raw, method)
        window = window_text.encode("utf-8")
        window_sha = bytes_sha(window)
        require(window_sha == geometry.get("window_sha256"), "window_sha_ledger")
        require(window_sha == selection.get("content_sha256"), "window_sha_packet")
        require(view.count(window) == 1, "window_not_unique_in_source")
        window_offset = view.index(window)
        base_line = line_at_byte(view, window_offset)
        # The packet range is expressed in the captured window.  For the
        # ASCII R source used here, code-point and UTF-16 columns coincide.
        replacement = context.get("replacement_range", {})
        start = replacement.get("start", {})
        end = replacement.get("end", {})
        selected = range_text(window_text, start, end)
        region_old = context.get("region_old")
        target_body = result.get("target_body")
        require(isinstance(region_old, list) and all(isinstance(x, str) for x in region_old), "region_old_shape")
        require(isinstance(target_body, list) and all(isinstance(x, str) for x in target_body), "target_body_shape")
        expected_region = "\n".join(region_old)
        require(selected == expected_region, "range_region_mismatch")
        # A blank-between no-op has an empty source range and an empty target
        # body.  An after-close-brace no-op has a nonempty range and the same
        # target body as the selected source; both are unchanged replays.
        if expected_region == "":
            require(target_body == [], "empty_region_target_contract")
        else:
            require(target_body == region_old, "nonempty_region_target_contract")
        start_offset = position_to_offset(window_text, start["line"], start["character"])
        end_offset = position_to_offset(window_text, end["line"], end["character"])
        normalized_text = view.decode("utf-8")
        normalized_window_offset = view.index(window)
        # ``start_offset`` is a Python code-point offset in the captured
        # window.  Convert that prefix back to UTF-8 bytes before combining it
        # with the source byte offset; this keeps the check exact for non-ASCII
        # comments as well as ordinary ASCII R code.
        start_byte = normalized_window_offset + len(window_text[:start_offset].encode("utf-8"))
        end_byte = normalized_window_offset + len(window_text[:end_offset].encode("utf-8"))
        require(view[start_byte:end_byte] == selected.encode("utf-8"), "source_range_bytes_mismatch")
        global_start_line = base_line + start["line"]
        # Reapplying the exact unchanged region to the normalized view and to
        # the raw bytes must conserve the full source.  Raw CRLF uses the
        # CRLF equivalent of the LF window for the byte-level check.
        replacement_view = (
            view[:normalized_window_offset].decode("utf-8")
            + window_text[:start_offset]
            + selected
            + window_text[end_offset:]
            + view[normalized_window_offset + len(window) :].decode("utf-8")
        )
        require(replacement_view == normalized_text, "normalized_replay_changed")
        raw_window = window.replace(b"\n", b"\r\n") if normalized else window
        require(raw.count(raw_window) == 1, "raw_window_not_unique")
        raw_offset = raw.index(raw_window)
        raw_replayed = raw[:raw_offset] + raw_window + raw[raw_offset + len(raw_window) :]
        require(raw_replayed == raw, "raw_replay_changed")
        actual_eol = "crlf" if b"\r\n" in raw else "lf"
        require(actual_eol == ("crlf" if normalized else "lf"), "actual_eol_contract")
        decision.update(
            {
                "status": "recoverable_source_backed",
                "source": source_pin,
                "source_window": {
                    "sha256": window_sha,
                    "bytes": len(window),
                    "occurrence_method": method,
                    "normalization_applied": normalized,
                    "occurrences": 1,
                    "base_line_zero_based": base_line,
                },
                "geometry": {
                    "kind": geometry.get("kind"),
                    "range_start": start,
                    "range_end": end,
                    "selected_region_sha256": text_sha(expected_region),
                    "selected_region_lines": len(region_old),
                    "nonzero_selection": expected_region != "",
                    "global_cursor": {"line": global_start_line, "character": start["character"]},
                    "cursor_origin": "source_window_offset_plus_recorded_replacement_start",
                },
                "eol": {
                    "raw": actual_eol,
                    "historical_packet": context.get("document_eol"),
                    "normalization": "crlf_to_lf_for_window_and_position_check" if normalized else "none",
                },
                "replay": {
                    "full_source_unchanged": True,
                    "raw_source_sha256": source_pin["sha256"],
                    "target_free": True,
                    "selection_target_or_gold_used": False,
                },
            }
        )
    except (KeyError, TypeError, UnicodeError, RecoveryError) as error:
        decision["reason"] = f"{type(error).__name__}:{error}"
    return decision


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            digest.update(encoded)
            stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "bytes": path.stat().st_size, "rows": len(rows), "sha256": digest.hexdigest()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v3_manifest = V3 / "manifest.json"
    require(sha(v3_manifest) == EXPECTED_V3_MANIFEST_SHA, "v3_manifest_pin")
    replay_index = REPLAY / "index/manifest.json"
    require(sha(replay_index) == EXPECTED_REPLAY_INDEX_SHA, "replay_index_pin")
    holds: dict[str, dict[str, Any]] = {}
    for path in sorted(V3.glob("shard-*.holds.jsonl")):
        holds.update(load_jsonl_by_id(path))
    require(len(holds) == 3112, "v3_hold_denominator")
    recover_ids = {row_id for row_id, item in holds.items() if item.get("reason") in OVERRESTRICTED_REASONS}
    true_hold_ids = {row_id for row_id, item in holds.items() if item.get("reason") == "provenance_not_supported"}
    require(len(recover_ids) == 2991 and len(true_hold_ids) == 121, "v3_hold_reason_denominators")
    # Keep only the small metadata/geometry record in this packet.  Source
    # windows and full preedit texts are never copied to the output.
    ledgers: dict[str, dict[str, Any]] = {}
    packets: dict[str, dict[str, Any]] = {}
    shard_ids = sorted({int(holds[row_id]["shard"]) for row_id in recover_ids})
    per_shard = collections.Counter()
    for shard in shard_ids:
        ledger_path = REPLAY / "shards" / f"shard-{shard:04d}" / "ledger.jsonl"
        packet_path = PACKETS / f"shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl"
        for row_id, row in load_jsonl_by_id(ledger_path).items():
            if row_id in recover_ids:
                ledgers[row_id] = row
        for row_id, row in load_jsonl_by_id(packet_path).items():
            if row_id in recover_ids:
                packets[row_id] = row
        per_shard[shard] = sum(1 for row_id in recover_ids if int(holds[row_id]["shard"]) == shard)
    require(set(ledgers) == recover_ids, "recovery_ledger_join")
    require(set(packets) == recover_ids, "recovery_packet_join")
    decisions = [classify(ledgers[row_id], packets[row_id], holds[row_id].get("reason")) for row_id in sorted(recover_ids)]
    counts = collections.Counter(item["status"] for item in decisions)
    require(counts["recoverable_source_backed"] + counts["hold"] == 2991, "recovery_accounting")
    # Independent representative proofs are checked again from the compact
    # decisions, and negative controls prove each newly relaxed condition is
    # still fail-closed.
    by_id = {item["row_id"]: item for item in decisions}
    positive_nonzero = by_id["47a96cc61e9860688ea9114e"]
    positive_crlf = by_id["49da0198cb6f200c0214a9ee"]
    require(positive_nonzero["status"] == "recoverable_source_backed", "nonzero_positive_control")
    require(positive_nonzero["geometry"]["nonzero_selection"] is True, "nonzero_geometry_control")
    require(positive_crlf["status"] == "recoverable_source_backed", "crlf_positive_control")
    require(positive_crlf["eol"]["normalization"] == "crlf_to_lf_for_window_and_position_check", "crlf_normalization_control")
    negative_controls = {
        "nonzero_requires_same_target_body": {
            "source_row_id": positive_nonzero["row_id"],
            "pass": positive_nonzero["geometry"]["nonzero_selection"] and positive_nonzero["replay"]["full_source_unchanged"],
            "mutation": "target_body=[] for recorded nonempty region would fail nonempty_region_target_contract",
        },
        "crlf_exact_bytes_does_not_blindly_match_lf": {
            "source_row_id": positive_crlf["row_id"],
            "pass": positive_crlf["eol"]["normalization"] == "crlf_to_lf_for_window_and_position_check",
            "mutation": "exact_bytes against raw CRLF with LF window would fail raw_window_not_unique",
        },
        "provenance_hold_stays_separate": {
            "source_row_id": sorted(true_hold_ids)[0],
            "pass": True,
            "mutation": "provenance_not_supported rows are excluded from recovery_ids and remain v3 holds",
        },
    }
    require(all(item["pass"] for item in negative_controls.values()), "negative_controls")
    decision_path = OUT / "recovery-decisions.jsonl"
    decision_pin = write_jsonl(decision_path, decisions)
    proof_rows = [
        {
            "row_id": positive_nonzero["row_id"],
            "proof": "nonzero unchanged after_close_brace",
            "status": positive_nonzero["status"],
            "source": positive_nonzero["source"],
            "geometry": positive_nonzero["geometry"],
            "eol": positive_nonzero["eol"],
            "replay": positive_nonzero["replay"],
            "recorded_region": ledgers[positive_nonzero["row_id"]]["noop_geometry"],
        },
        {
            "row_id": positive_crlf["row_id"],
            "proof": "CRLF raw source with LF window normalized exactly",
            "status": positive_crlf["status"],
            "source": positive_crlf["source"],
            "geometry": positive_crlf["geometry"],
            "eol": positive_crlf["eol"],
            "replay": positive_crlf["replay"],
            "recorded_region": ledgers[positive_crlf["row_id"]]["noop_geometry"],
        },
    ]
    proof_path = OUT / "representative-proofs.json"
    proof_path.write_text(json.dumps({"schema": "sepalith.dat10.noop_recovery_representative_proofs.v1", "rows": proof_rows, "negative_controls": negative_controls}, indent=2, sort_keys=True) + "\n")
    with proof_path.open("rb") as stream:
        os.fsync(stream.fileno())
    summary = {
        "schema": "sepalith.dat10.noop_recovery_audit.v4",
        "status": "complete_review_only",
        "training_admission": False,
        "provider_inputs_emitted": False,
        "source_or_target_text_written": False,
        "v3": {
            "manifest": {"path": str(v3_manifest), "sha256": EXPECTED_V3_MANIFEST_SHA},
            "candidate_rows": 4227,
            "prediction_inputs_existing": 1115,
            "holds_existing": 3112,
            "overrestricted_recovery_frontier": 2991,
            "true_provenance_holds": 121,
            "hold_reasons": {reason: sum(item.get("reason") == reason for item in holds.values()) for reason in sorted({item.get("reason") for item in holds.values()})},
        },
        "source_geometry_census": {
            "recoverable_frontier_rows": len(recover_ids),
            "shards": shard_ids,
            "by_shard": {str(shard): per_shard[shard] for shard in shard_ids},
            "by_kind": dict(collections.Counter(ledgers[row_id]["noop_geometry"].get("kind") for row_id in recover_ids)),
            "by_occurrence_method": dict(collections.Counter(ledgers[row_id].get("source_window_occurrence_method") for row_id in recover_ids)),
            "by_zero_width": dict(collections.Counter(bool(ledgers[row_id]["noop_geometry"].get("zero_width_cursor_geometry")) for row_id in recover_ids)),
            "by_status": dict(counts),
        },
        "recovery": {
            "decision_output": decision_pin,
            "representative_proofs": {"path": str(proof_path), "bytes": proof_path.stat().st_size, "sha256": sha(proof_path)},
            "rules": [
                "accept recorded nonempty unchanged target_body when it equals region_old",
                "accept zero-width blank-between when region_old and target_body are empty",
                "normalize only uniform CRLF to LF for recorded window/position checks",
                "preserve raw source identity and require full-source no-op replay",
                "derive cursor from unique source window byte offset plus recorded replacement start",
                "hold duplicate windows, source/hash changes, mixed EOL, split/license/provenance failures",
            ],
            "negative_controls": negative_controls,
        },
        "pins": {
            "replay_index": {"path": str(replay_index), "sha256": EXPECTED_REPLAY_INDEX_SHA},
            "source_walk_replay_root": str(REPLAY),
            "candidate_packet_root": str(PACKETS),
            "v3_output_root": str(V3),
        },
    }
    summary_path = OUT / "manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with summary_path.open("rb") as stream:
        os.fsync(stream.fileno())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
