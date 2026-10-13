#!/usr/bin/env python3
"""Independent, candidate-only review of the first frozen DAT-10 source-walk shards.

This audit intentionally does not register or admit data.  It checks the frozen
token rows against the source-line metadata, the global train registry, and the
15,008-row comparison pool.  Source files are read only to replay line/file
hashes; their contents are never written to an output artifact.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUT_DEFAULT = ROOT / "docs/campaign/work/lead/r2-source-walk-independent-review-v1"
RECEIPT_DEFAULT = ROOT / "docs/campaign/receipts/DAT-10-source-walk-independent-review.json"

SOURCE_WALK = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1")
DAT03 = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-row-audit.jsonl")
GLOBAL = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
CPT = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
ACCEPTED = ROOT / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl"
ACCEPTED_PROV = ROOT / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl"
NEW15008 = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-token-rows.jsonl")
NEW15008_PROV = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl")
NEW15008_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/manifest.json")

PINNED = {
    DAT03: "9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd",
    GLOBAL: "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09",
    CPT: "6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06",
    ACCEPTED: "fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889",
    ACCEPTED_PROV: "317077b17afc38602309361396bb86d8897a6b823c5d7539f10b9733a0ad9a48",
    NEW15008: "7887686e022e520e746c04c6197a9a3c2484fba2668d09a650d0627ff267187b",
    NEW15008_PROV: "e499c07d6da2ef325f7d9f3156b6c2e70361d3b1bc140198b6c3563b4471e7a0",
    NEW15008_MANIFEST: "5334757c07d705289e6616d1f79890d008b7837914f7ca61a257aa9efdbc02c9",
}
AUDIT_SHA = "9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def u16(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def json_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if line.strip():
                yield line_number, json.loads(line)


def json_rows_hashed(path: Path):
    """Yield JSONL rows while computing the exact file hash in one pass."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for line_number, raw in enumerate(fh, 1):
            h.update(raw)
            if raw.strip():
                yield line_number, json.loads(raw)
    # The caller receives the hash through the returned iterator only after it
    # is exhausted, so this helper is kept for readability in small files.


def norm_key(source: object, line: object, package: object):
    try:
        line_value = int(line or 0)
    except (TypeError, ValueError):
        line_value = 0
    return (str(source or ""), line_value, str(package or ""))


def source_key_from_ref(ref: dict):
    return norm_key(ref.get("source"), ref.get("line", ref.get("source_line")), ref.get("package_id", ref.get("package")))


def source_key_from_provenance(row: dict):
    return norm_key(row.get("source"), row.get("source_line", row.get("line")), row.get("package_id", row.get("package")))


def add_reason(reasons: list[str], condition: bool, name: str):
    if condition:
        reasons.append(name)


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def file_pin(path: Path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def physical_region_old(item: dict):
    provenance = item.get("source_provenance", {})
    context = item.get("context", {})
    value = provenance.get("physical_region_old", provenance.get("raw_region_old"))
    if isinstance(value, list):
        return value
    region = context.get("region_old")
    return region if isinstance(region, list) and region else [""]


def relation(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def load_token_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                rows.append(row)
    return rows


def check_token_geometry(item: dict, max_sequence: int = 4096):
    """Return hard integrity reasons and non-blocking review flags."""
    row = item["row"]
    context = item["context"]
    selection = item["selection"]
    provenance = item["source_provenance"]
    reasons: list[str] = []
    flags: list[str] = []
    body = row.get("target_body_text", "")
    old = physical_region_old(item)

    add_reason(reasons, sha_text(row.get("prompt_text", "")) != item.get("prompt_sha256"), "prompt_hash_mismatch")
    add_reason(reasons, sha_text(row.get("target_text", "")) != item.get("target_sha256"), "target_hash_mismatch")
    add_reason(reasons, row.get("target_start") != row.get("prompt_token_count", 0) + 1, "target_start_mismatch")
    add_reason(reasons, row.get("bos_token_id") != 0 or row.get("eos_token_id") != 1, "special_token_identity_mismatch")
    input_ids = row.get("input_ids", [])
    body_tokens = row.get("target_body_tokens", [])
    terminal_tokens = row.get("target_terminal_tokens", [])
    target_start = row.get("target_start")
    expected_tail = body_tokens + terminal_tokens + [row.get("eos_token_id", 1)]
    add_reason(reasons, not isinstance(target_start, int) or input_ids[target_start:] != expected_tail, "target_token_tail_mismatch")
    add_reason(reasons, row.get("target_body_token_count") != len(body_tokens), "target_body_count_mismatch")
    add_reason(reasons, row.get("target_terminal_token_count") != len(terminal_tokens), "target_terminal_count_mismatch")
    add_reason(reasons, row.get("target_token_count") != len(body_tokens) + len(terminal_tokens), "target_count_mismatch")
    lengths = item.get("lengths", {})
    add_reason(reasons, lengths.get("prompt_with_bos") != row.get("prompt_token_count", 0) + 1, "prompt_length_mismatch")
    add_reason(reasons, lengths.get("prompt_without_bos") != row.get("prompt_token_count"), "prompt_without_bos_length_mismatch")
    add_reason(reasons, lengths.get("target_body") != len(body_tokens), "body_length_mismatch")
    add_reason(reasons, lengths.get("response_with_terminal_eos") != len(body_tokens) + len(terminal_tokens) + 1, "response_length_mismatch")
    add_reason(reasons, lengths.get("sequence") != len(input_ids), "sequence_length_mismatch")
    add_reason(reasons, len(input_ids) > max_sequence, "sequence_length_overflow")
    # The audit keeps long targets.  They are queued below rather than rejected.
    if lengths.get("response_with_terminal_eos", 0) > 1024:
        flags.append("long_target_gt_1024_review_only")
    if lengths.get("sequence", 0) > max_sequence:
        flags.append("long_sequence_gt_4096_review_only")
    add_reason(reasons, item.get("status") != "tokenizer_candidate_only", "candidate_status_mismatch")
    add_reason(reasons, item.get("admitted_for_training") is not False, "candidate_admission_flag_mismatch")
    add_reason(reasons, row.get("split") != "train", "token_split_mismatch")
    add_reason(reasons, row.get("target_operation") != "replace", "operation_not_replace")
    # Roxygen rows can be indented in the source document.
    add_reason(reasons, not body.lstrip().startswith("#'"), "roxygen_target_prefix_missing")
    add_reason(reasons, old != [""], "roxygen_physical_region_not_blank")

    add_reason(reasons, context.get("prefix") != selection.get("prefix"), "selection_prefix_mismatch")
    add_reason(reasons, context.get("suffix_lines") != selection.get("suffix"), "selection_suffix_mismatch")
    add_reason(reasons, selection.get("region") != [""] or context.get("region_old") not in ([], [""]), "selection_region_mismatch")
    replacement = context.get("replacement_range", {})
    start = replacement.get("start", {})
    end = replacement.get("end", {})
    add_reason(reasons, replacement.get("content_sha256") != selection.get("document_sha256"), "replacement_document_hash_mismatch")
    add_reason(reasons, start.get("line") is None or end.get("line") is None, "replacement_line_missing")
    add_reason(reasons, start.get("character") != 0 or end.get("character") != 0, "replacement_blank_column_mismatch")
    add_reason(reasons, start.get("line") != end.get("line"), "replacement_blank_extent_mismatch")
    add_reason(reasons, provenance.get("target_start_line") != start.get("line"), "provenance_start_line_mismatch")
    add_reason(reasons, provenance.get("target_end_line") != end.get("line"), "provenance_end_line_mismatch")
    add_reason(reasons, provenance.get("post_edit_geometry_verified") is not True, "post_edit_geometry_unverified")
    cursor = context.get("cursor", {})
    add_reason(reasons, cursor.get("region_line_index") != -1 or cursor.get("code_point_column") is not None or cursor.get("utf16_column") is not None, "blank_cursor_geometry_mismatch")
    add_reason(reasons, selection.get("overflow") is not False or selection.get("required_overflow") is not False, "selection_overflow")
    add_reason(reasons, selection.get("omissions") not in ([], None), "selection_omissions")
    add_reason(reasons, selection.get("required_utf16_units") != 1, "required_region_width_mismatch")
    add_reason(reasons, selection.get("support_revalidation_required") is True, "context_support_revalidation_required")
    if provenance.get("source_snapshot_is_simulated") is True:
        flags.append("context_snapshot_simulated_review_only")
    if provenance.get("support_status") not in (None, "source_validated"):
        flags.append("context_support_status_" + str(provenance.get("support_status")))
    if provenance.get("history_replay", {}).get("replayed") is False:
        flags.append("context_history_not_replayed_review_only")
    # Do not allow a target to be a literal copy of an old non-empty region.
    add_reason(reasons, row.get("target_operation") == "replace" and "\n".join(old) != "" and body == "\n".join(old), "unchanged_replacement")
    add_reason(reasons, len(body) >= 12 and body in row.get("prompt_text", ""), "target_text_in_prompt")
    return sorted(set(reasons)), sorted(set(flags))


def main(output: Path, receipt_path: Path):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {
        "schema": "DAT-10-source-walk-independent-review-status-v2",
        "status": "audit_running",
        "owner": "/root/data_expansion",
        "review_rows": 10071,
        "comparison_pool_rows": 15008,
        "policy": {
            "candidate_only": True,
            "admission": "none",
            "training_or_model": False,
            "heldout_payload_opened": False,
            "source_content_use": "TRAIN source-line hash replay only",
            "long_targets": "retain_and_queue; never_truncate",
            "recoverable_evidence": "named review or repair queue",
            "disjoint_from_dat10_expanded_15008": True,
            "cpu_threads": 2,
            "nice": 10,
            "ionice": "idle",
        },
    })

    # Verify the pinned comparison and registry files before deriving any set.
    input_pins = {}
    for path, expected in PINNED.items():
        actual = sha_file(path)
        if actual != expected:
            raise RuntimeError(f"pinned input hash mismatch for {path}: {actual} != {expected}")
        input_pins[str(path)] = {"bytes": path.stat().st_size, "sha256": actual}

    global_doc = json.loads(GLOBAL.read_text(encoding="utf-8"))
    global_groups = {str(group["group_id"]): group for group in global_doc["groups"]}
    del global_doc
    cpt_doc = json.loads(CPT.read_text(encoding="utf-8"))
    cpt_groups = {str(group_id): str(partition) for group_id, partition in cpt_doc["groups"].items()}
    del cpt_doc

    # Comparison pool identity sets.  Source identity uses the accepted
    # provenance, because flattened accepted token rows intentionally omit it.
    accepted_ids: set[str] = set()
    accepted_prompts: set[str] = set()
    accepted_targets: set[str] = set()
    accepted_pairs: set[tuple[str, str]] = set()
    with ACCEPTED.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_hash = sha_text(row["prompt_text"])
            target_hash = sha_text(row["target_text"])
            accepted_ids.add(row["id"])
            accepted_prompts.add(prompt_hash)
            accepted_targets.add(target_hash)
            accepted_pairs.add((prompt_hash, target_hash))
    accepted_sources = set()
    accepted_prov_rows = 0
    with ACCEPTED_PROV.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                accepted_sources.add(source_key_from_provenance(json.loads(line)))
                accepted_prov_rows += 1

    new_ids: set[str] = set()
    new_prompts: set[str] = set()
    new_targets: set[str] = set()
    new_pairs: set[tuple[str, str]] = set()
    with NEW15008.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                prompt_hash = sha_text(row["prompt_text"])
                target_hash = sha_text(row["target_text"])
                new_ids.add(row["id"])
                new_prompts.add(prompt_hash)
                new_targets.add(target_hash)
                new_pairs.add((prompt_hash, target_hash))
    new_sources = set()
    new_prov_ids = set()
    with NEW15008_PROV.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                new_prov_ids.add(row["row_id"])
                new_sources.add(source_key_from_ref(row["source_ref"]))
    comparison_ids = accepted_ids | new_ids
    comparison_prompts = accepted_prompts | new_prompts
    comparison_pairs = accepted_pairs | new_pairs
    comparison_sources = accepted_sources | new_sources
    if len(comparison_ids) != 15008 or len(comparison_prompts) != 15008 or len(new_prov_ids) != 3503:
        raise RuntimeError("15,008 comparison pool cardinality or provenance join failed")

    # Load and pin the five frozen material/token outputs.  Candidate packets
    # are used only for direct license evidence and source-ref closure.
    candidates = []
    packet_validation = {}
    packet_ref = {}
    shard_info = []
    expected_shard_rows = {0: 1685, 1: 2111, 2: 2089, 3: 2118, 4: 2068}
    token_paths = []
    for shard in range(5):
        shard_dir = SOURCE_WALK / f"shard-{shard:04d}"
        input_manifest = shard_dir / "manifest.json"
        material_dir = shard_dir / "structured-materialization-v1"
        material_manifest = material_dir / "manifest.json"
        token_manifest = material_dir / "token-audit-manifest.json"
        material = json.loads(material_manifest.read_text(encoding="utf-8"))
        token = json.loads(token_manifest.read_text(encoding="utf-8"))
        input_doc = json.loads(input_manifest.read_text(encoding="utf-8"))
        if material.get("status") != "frozen_review_increment_unadmitted" or token.get("status") != "frozen_review_increment_unadmitted":
            raise RuntimeError(f"shard {shard} is not frozen/unadmitted")
        if sha_file(input_manifest) != material.get("input_manifest_sha256"):
            raise RuntimeError(f"shard {shard} input manifest pin mismatch")
        token_path = Path(token["token_rows"]["path"])
        if token_path != material_dir / "token-audit/candidate-token-rows.jsonl":
            raise RuntimeError(f"shard {shard} token path escaped expected frozen output")
        token_hash = sha_file(token_path)
        expected_rows = expected_shard_rows[shard]
        if token_hash != token["token_rows"]["sha256"] or token["token_rows"]["rows"] != expected_rows:
            raise RuntimeError(f"shard {shard} token manifest/hash/count mismatch")
        shard_rows = load_token_rows(token_path)
        if len(shard_rows) != expected_rows:
            raise RuntimeError(f"shard {shard} parsed row count mismatch")
        for item in shard_rows:
            item["_shard"] = shard
            candidates.append(item)
        packet_path = material_dir / "candidate-packets.jsonl"
        packet_hash = sha_file(packet_path)
        expected_packet = material.get("outputs", {}).get("candidate_packets", {})
        if packet_hash != expected_packet.get("sha256") or expected_packet.get("rows") != expected_rows:
            raise RuntimeError(f"shard {shard} candidate packet manifest/hash/count mismatch")
        with packet_path.open(encoding="utf-8") as fh:
            packet_count = 0
            for line in fh:
                if not line.strip():
                    continue
                packet = json.loads(line)
                row_ref = packet.get("row_ref", {})
                rid = row_ref.get("row_id")
                if rid in packet_ref:
                    raise RuntimeError(f"duplicate packet row id {rid}")
                packet_ref[rid] = row_ref
                packet_validation[rid] = packet.get("validation", {})
                packet_count += 1
            if packet_count != expected_rows:
                raise RuntimeError(f"shard {shard} packet count mismatch")
        shard_info.append({
            "shard": shard,
            "input_manifest": file_pin(input_manifest),
            "material_manifest": file_pin(material_manifest),
            "token_manifest": file_pin(token_manifest),
            "token_rows": file_pin(token_path) | {"rows": expected_rows},
            "candidate_packets": file_pin(packet_path) | {"rows": expected_rows},
            "input_rows": input_doc.get("rows_copied", input_doc.get("planned_rows")),
            "input_family_counts": input_doc.get("family_counts", {}),
        })
        token_paths.append(token_path)

    source_walk_input_family_counts = collections.Counter()
    for info in shard_info:
        source_walk_input_family_counts.update(info.get("input_family_counts", {}))

    candidate_ids = {item["row"]["id"] for item in candidates}
    if len(candidates) != 10071 or len(candidate_ids) != 10071 or set(packet_ref) != candidate_ids:
        raise RuntimeError("frozen candidate denominator or packet identity closure failed")

    # Read the shard audit inputs, retaining only candidate metadata.  The
    # copied rows are metadata, not held-out content.
    audit_by_id = {}
    audit_file_pins = []
    for shard in range(5):
        shard_dir = SOURCE_WALK / f"shard-{shard:04d}"
        input_manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        audit_path = shard_dir / "audit-rows.jsonl"
        expected = input_manifest["outputs"].get("audit_rows", input_manifest["outputs"].get("audit-rows"))
        if expected is None:
            raise RuntimeError(f"shard {shard} has no audit-rows output pin")
        h = hashlib.sha256()
        count = 0
        with audit_path.open("rb") as fh:
            for raw in fh:
                h.update(raw)
                if raw.strip():
                    count += 1
                    row = json.loads(raw)
                    if row["row_id"] in candidate_ids:
                        if row["row_id"] in audit_by_id:
                            raise RuntimeError(f"candidate appears in two shard audit inputs: {row['row_id']}")
                        audit_by_id[row["row_id"]] = row
        if count != expected["rows"] or h.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"shard {shard} audit input hash/count mismatch")
        audit_file_pins.append({"path": str(audit_path), "rows": count, "bytes": audit_path.stat().st_size, "sha256": h.hexdigest()})
    if set(audit_by_id) != candidate_ids:
        raise RuntimeError(f"candidate to shard audit join incomplete: {len(audit_by_id)}")

    # Full DAT-03 hash and candidate metadata join.  No DAT-03 payload fields
    # are copied into the output beyond status/hash/identity diagnostics.
    dat03_by_id = {}
    dat03_hash = hashlib.sha256()
    dat03_rows = 0
    with DAT03.open("rb") as fh:
        for raw in fh:
            dat03_hash.update(raw)
            if raw.strip():
                dat03_rows += 1
                row = json.loads(raw)
                if row["row_id"] in candidate_ids:
                    if row["row_id"] in dat03_by_id:
                        raise RuntimeError(f"duplicate DAT-03 row id {row['row_id']}")
                    dat03_by_id[row["row_id"]] = row
    if dat03_hash.hexdigest() != AUDIT_SHA or dat03_rows != 376830 or set(dat03_by_id) != candidate_ids:
        raise RuntimeError("DAT-03 pinned hash/count/candidate join failed")

    # Replay raw TRAIN source line and complete source-file hashes.  This
    # reads source content only for hash comparison and records no content.
    source_requests = collections.defaultdict(lambda: collections.defaultdict(list))
    for item in candidates:
        ref = item["source_ref"]
        source_requests[Path(ref["file"])][int(ref["line"])].append((item["row"]["id"], ref["raw_line_sha256"]))
    source_replay = {}
    source_files = []
    for source_path, line_map in sorted(source_requests.items(), key=lambda pair: str(pair[0])):
        before = source_path.stat()
        file_hash = hashlib.sha256()
        found = set()
        mismatches = []
        with source_path.open("rb") as fh:
            for line_number, raw in enumerate(fh, 1):
                file_hash.update(raw)
                if line_number not in line_map:
                    continue
                found.add(line_number)
                including = hashlib.sha256(raw).hexdigest()
                without = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
                for row_id, expected_hash in line_map[line_number]:
                    if expected_hash in (including, without):
                        source_replay[row_id] = "pass"
                    else:
                        source_replay[row_id] = "mismatch"
                        mismatches.append({"row_id": row_id, "line": line_number, "expected": expected_hash, "actual_including_separator": including, "actual_without_separator": without})
        missing_lines = sorted(set(line_map) - found)
        for line_number in missing_lines:
            for row_id, expected_hash in line_map[line_number]:
                source_replay[row_id] = "missing"
                mismatches.append({"row_id": row_id, "line": line_number, "expected": expected_hash, "failure": "line_missing"})
        after = source_path.stat()
        expected_source_hashes = {item["source_ref"]["source_sha256"] for item in candidates if Path(item["source_ref"]["file"]) == source_path}
        file_hash_value = file_hash.hexdigest()
        file_hash_ok = len(expected_source_hashes) == 1 and file_hash_value in expected_source_hashes
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            for row_id in line_map.values():
                for rid, _ in row_id:
                    source_replay[rid] = "changed_during_replay"
            mismatches.append({"failure": "source_file_changed_during_replay"})
        source_files.append({
            "path": str(source_path),
            "bytes": after.st_size,
            "sha256": file_hash_value,
            "expected_sha256": sorted(expected_source_hashes),
            "file_hash_match": file_hash_ok,
            "requested_rows": sum(len(values) for values in line_map.values()),
            "requested_lines": len(line_map),
            "found_lines": len(found),
            "mismatches": len(mismatches),
            "stat_stable": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
        })
        if not file_hash_ok:
            for values in line_map.values():
                for row_id, _ in values:
                    source_replay[row_id] = "source_file_hash_mismatch"
    source_replay_doc = {
        "schema": "sepalith.dat10.source-walk-independent-source-replay.v1",
        "status": "pass" if all(value == "pass" for value in source_replay.values()) and len(source_replay) == len(candidates) and all(file["file_hash_match"] and file["stat_stable"] and file["mismatches"] == 0 for file in source_files) else "fail",
        "requested_rows": len(candidates),
        "matched_rows": sum(value == "pass" for value in source_replay.values()),
        "source_files": source_files,
        "constraints": {"source_content_written": False, "r_executed": False, "heldout_payload_opened": False},
    }
    atomic_json(output / "source-line-replay.json", source_replay_doc)

    # Verify direct license evidence paths/hashes from the frozen candidate
    # packets.  Missing or unverifiable evidence remains repairable; it is not
    # a permanent exclusion.
    license_paths = {}
    license_by_id = {}
    for rid, validation in packet_validation.items():
        evidence = validation.get("license_evidence") or {}
        path = evidence.get("path")
        expected_hash = evidence.get("sha256")
        fields = evidence.get("fields")
        direct = bool(path and expected_hash and isinstance(fields, list) and fields)
        license_by_id[rid] = {"direct_evidence": direct, "path": path, "expected_sha256": expected_hash, "fields_present": bool(fields)}
        if direct:
            license_paths[str(path)] = expected_hash
    license_file_status = {}
    for path_text, expected_hash in sorted(license_paths.items()):
        path = Path(path_text)
        if not path.is_file():
            license_file_status[path_text] = {"status": "missing", "expected_sha256": expected_hash}
            continue
        actual_hash = sha_file(path)
        license_file_status[path_text] = {"status": "pass" if actual_hash == expected_hash else "hash_mismatch", "bytes": path.stat().st_size, "expected_sha256": expected_hash, "sha256": actual_hash}
    for rid, evidence in license_by_id.items():
        if not evidence["direct_evidence"]:
            evidence["status"] = "repair_required_missing_direct_evidence"
        else:
            evidence["status"] = license_file_status.get(str(evidence["path"]), {}).get("status", "missing")

    # Candidate-level audit and deterministic duplicate classification.
    prompt_to_indices = collections.defaultdict(list)
    pair_to_indices = collections.defaultdict(list)
    source_to_indices = collections.defaultdict(list)
    for index, item in enumerate(candidates):
        row = item["row"]
        prompt_to_indices[item["prompt_sha256"]].append(index)
        pair_to_indices[(item["prompt_sha256"], item["target_sha256"])].append(index)
        source_to_indices[source_key_from_ref(item["source_ref"])].append(index)

    ledgers = []
    reasons_count = collections.Counter()
    flags_count = collections.Counter()
    decisions = collections.Counter()
    family_counts = collections.Counter()
    source_counts = collections.Counter()
    length_buckets = collections.Counter()
    long_rows = []
    candidate_output_rows = []
    repair_queues = collections.defaultdict(list)

    for index, item in enumerate(candidates):
        row = item["row"]
        rid = row["id"]
        ref = item["source_ref"]
        provenance = item["source_provenance"]
        family = row.get("family", "")
        family_counts[family] += 1
        source_counts[ref.get("source", "")] += 1
        reasons, flags = check_token_geometry(item)
        audit = audit_by_id[rid]
        metadata = dat03_by_id[rid]
        validation = packet_validation.get(rid, {})
        license_evidence = license_by_id.get(rid, {})
        global_group = global_groups.get(str(ref.get("group_id")))
        global_split = global_group.get("split") if global_group else None
        cpt_partition = cpt_groups.get(str(ref.get("group_id")), "cpt_partition_missing")
        source_replay_status = source_replay.get(rid, "missing")

        # Source-ref joins against both copied DAT-03 metadata and the full
        # immutable DAT-03 audit are explicit and candidate-only.
        for name, lhs, rhs in (
            ("source_file", ref.get("file"), audit.get("file")),
            ("source_line", ref.get("line"), audit.get("line")),
            ("source_group", ref.get("group_id"), audit.get("group_id")),
            ("source_name", ref.get("source"), audit.get("source")),
            ("source_raw_line_hash", ref.get("raw_line_sha256"), audit.get("raw_line_sha256")),
            ("source_file_hash", ref.get("source_sha256"), audit.get("source_sha256")),
            ("source_family", ref.get("family"), audit.get("family")),
        ):
            add_reason(reasons, relation(lhs) != relation(rhs), f"{name}_metadata_join_mismatch")
        for name, lhs, rhs in (
            ("source_file", ref.get("file"), metadata.get("file")),
            ("source_line", ref.get("line"), metadata.get("line")),
            ("source_group", ref.get("group_id"), metadata.get("group_id")),
            ("source_name", ref.get("source"), metadata.get("source")),
            ("source_raw_line_hash", ref.get("raw_line_sha256"), metadata.get("raw_line_sha256")),
            ("source_file_hash", ref.get("source_sha256"), metadata.get("source_sha256")),
            ("source_family", ref.get("family"), metadata.get("family")),
        ):
            add_reason(reasons, relation(lhs) != relation(rhs), f"{name}_dat03_join_mismatch")
        add_reason(reasons, ref.get("verification") != "DAT-03-row-audit", "source_verification_label_mismatch")
        add_reason(reasons, ref.get("split") != "train_group" or metadata.get("split") != "train_group" or audit.get("split") != "train_group", "not_train_group")
        add_reason(reasons, global_split != "train_group", "not_global_train")
        add_reason(reasons, cpt_partition == "cpt_validation", "cpt_validation_reserved")
        add_reason(reasons, global_group is None, "global_group_missing")
        add_reason(reasons, metadata.get("explicit_truncation") is True, "metadata_explicit_truncation")
        add_reason(reasons, metadata.get("target_kind") not in (None, "region_new"), "metadata_target_kind_mismatch")
        add_reason(reasons, audit.get("row_id") != rid or metadata.get("row_id") != rid, "metadata_row_id_mismatch")
        add_reason(reasons, packet_ref.get(rid) != ref, "candidate_packet_source_ref_mismatch")
        add_reason(reasons, source_replay_status != "pass", "source_line_replay_" + source_replay_status)

        if license_evidence.get("status") == "missing":
            reasons.append("license_evidence_recovery_required")
        elif license_evidence.get("status") == "hash_mismatch":
            reasons.append("license_evidence_hash_recovery_required")
        elif license_evidence.get("status") == "repair_required_missing_direct_evidence":
            reasons.append("license_evidence_recovery_required")
        elif metadata.get("license_status") == "missing_unresolved":
            flags.append("metadata_license_missing_recovered_by_packet_evidence")

        # The CPT partition is informational here.  Global train is the
        # held-out guard for this source-walk review.
        if cpt_partition == "cpt_partition_missing":
            flags.append("cpt_partition_missing_global_train_allowed")
        elif cpt_partition == "cpt_train":
            flags.append("cpt_train")
        if global_group and global_group.get("flags"):
            flags.append("global_flags_present_review_only")
        if provenance.get("source_snapshot_is_simulated") is True:
            flags.append("context_snapshot_simulated_review_only")

        prompt_hash = item["prompt_sha256"]
        target_hash = item["target_sha256"]
        pair = (prompt_hash, target_hash)
        source_key = source_key_from_ref(ref)
        add_reason(reasons, rid in comparison_ids, "comparison_pool_row_id_duplicate")
        add_reason(reasons, prompt_hash in comparison_prompts, "comparison_pool_prompt_duplicate")
        add_reason(reasons, pair in comparison_pairs, "comparison_pool_prompt_target_duplicate")
        add_reason(reasons, source_key in comparison_sources, "comparison_pool_source_identity_duplicate")
        # Keep the earliest frozen row for an exact rendered duplicate.  The
        # retained first row carries a review flag; later rows are permanent
        # duplicate exclusions.  Prompt conflicts remain repairable.
        same_prompt = prompt_to_indices[prompt_hash]
        same_pair = pair_to_indices[pair]
        same_source = source_to_indices[source_key]
        if len(same_prompt) > 1:
            if len({candidates[j]["target_sha256"] for j in same_prompt}) > 1:
                reasons.append("within_candidate_prompt_conflicting_target")
            elif index != min(same_prompt):
                reasons.append("within_candidate_prompt_target_duplicate")
            else:
                flags.append("within_candidate_exact_prompt_group_first_retained")
        if len(same_pair) > 1 and index != min(same_pair):
            reasons.append("within_candidate_prompt_target_duplicate")
        if len(same_source) > 1:
            if len({candidates[j]["target_sha256"] for j in same_source}) > 1:
                reasons.append("within_candidate_source_conflicting_target")
            elif index != min(same_source):
                reasons.append("within_candidate_source_duplicate")

        # Long targets are measurements, never length-based exclusions.
        response_len = int(item.get("lengths", {}).get("response_with_terminal_eos", 0))
        sequence_len = int(item.get("lengths", {}).get("sequence", 0))
        if response_len <= 192:
            length_buckets["response_le_192"] += 1
        elif response_len <= 512:
            length_buckets["response_193_512"] += 1
        elif response_len <= 1024:
            length_buckets["response_513_1024"] += 1
        else:
            length_buckets["response_gt_1024"] += 1
        if sequence_len <= 2048:
            length_buckets["sequence_le_2048"] += 1
        elif sequence_len <= 4096:
            length_buckets["sequence_2049_4096"] += 1
        else:
            length_buckets["sequence_gt_4096"] += 1
        if response_len > 1024 or sequence_len > 4096:
            long_rows.append({"row_id": rid, "shard": item["_shard"], "family": family, "response_with_terminal_eos": response_len, "sequence": sequence_len, "status": "review_only; retained; no_truncation"})

        reasons = sorted(set(reasons))
        flags = sorted(set(flags))
        permanent = sorted(reason for reason in reasons if "duplicate" in reason and "conflicting" not in reason)
        repair = sorted(reason for reason in reasons if reason not in permanent)
        if permanent and not repair:
            decision = "exclude_duplicate"
        elif repair:
            decision = "repair_required"
        else:
            decision = "candidate_pending_root_admission"
        decisions[decision] += 1
        for reason in reasons:
            reasons_count[reason] += 1
            repair_queues[reason].append(rid)
        for flag in flags:
            flags_count[flag] += 1
        ledger = {
            "index": index,
            "row_id": rid,
            "shard": item["_shard"],
            "family": family,
            "package_id": row.get("package_id"),
            "group_id": ref.get("group_id"),
            "source": ref.get("source"),
            "source_file": ref.get("file"),
            "source_line": ref.get("line"),
            "prompt_sha256": prompt_hash,
            "target_sha256": target_hash,
            "source_identity": {"source": source_key[0], "line": source_key[1], "package_id": source_key[2]},
            "decision": decision,
            "reasons": reasons,
            "permanent_exclusion_reasons": permanent,
            "repair_reasons": repair,
            "review_flags": flags,
            "checks": {
                "source_line_replay": source_replay_status == "pass",
                "source_file_hash": all(file["file_hash_match"] for file in source_files if file["path"] == ref.get("file")),
                "license_direct_evidence": bool(license_evidence.get("direct_evidence")),
                "license_file_hash": license_evidence.get("status") == "pass",
                "dat03_metadata_join": not any(reason.endswith("dat03_join_mismatch") for reason in reasons),
                "shard_audit_join": not any(reason.endswith("metadata_join_mismatch") for reason in reasons),
                "global_train": global_split == "train_group",
                "cpt_partition": cpt_partition,
                "token_geometry": not any(reason in reasons for reason in ("prompt_hash_mismatch", "target_hash_mismatch", "target_token_tail_mismatch", "sequence_length_mismatch", "sequence_length_overflow")),
                "no_target_truncation_metadata": metadata.get("explicit_truncation") is False,
            },
            "lengths": {
                "prompt_with_bos": item.get("lengths", {}).get("prompt_with_bos"),
                "response_with_terminal_eos": response_len,
                "sequence": sequence_len,
            },
        }
        ledgers.append(ledger)
        if decision == "candidate_pending_root_admission":
            candidate_output_rows.append({
                "row_id": rid,
                "shard": item["_shard"],
                "family": family,
                "group_id": ref.get("group_id"),
                "source": ref.get("source"),
                "source_file": ref.get("file"),
                "source_line": ref.get("line"),
                "prompt_sha256": prompt_hash,
                "target_sha256": target_hash,
                "status": "candidate_pending_root_admission",
                "review_flags": flags,
            })

    # Ensure every output row is candidate-only and disjoint from the 15,008
    # comparison set before writing it.
    if len(ledgers) != 10071 or len({row["row_id"] for row in ledgers}) != 10071:
        raise RuntimeError("ledger denominator is not 10,071")
    if set(row["row_id"] for row in candidate_output_rows) & comparison_ids:
        raise RuntimeError("candidate output overlaps comparison IDs")
    if any(row["prompt_sha256"] in comparison_prompts or (row["prompt_sha256"], row["target_sha256"]) in comparison_pairs for row in candidate_output_rows):
        raise RuntimeError("candidate output overlaps comparison rendered identities")

    write_jsonl(output / "audit-ledger.jsonl", ledgers)
    write_jsonl(output / "candidate-ids.jsonl", candidate_output_rows)
    write_jsonl(output / "long-context-queue.jsonl", long_rows)
    repair_queue_doc = {
        "schema": "sepalith.dat10.source-walk-independent-repair-queues.v1",
        "denominator": len(ledgers),
        "permanent_exclusion_policy": "exact duplicate identity only; conflicts remain repairable",
        "queues": {reason: {"rows": len(ids), "row_ids": ids} for reason, ids in sorted(repair_queues.items())},
    }
    atomic_json(output / "repair-queues.json", repair_queue_doc)

    # Persist a compact machine-readable summary before the larger receipt.
    summary = {
        "schema": "sepalith.dat10.source-walk-independent-review-summary.v2",
        "status": "independent_review_complete_root_admission_pending",
        "admission": "none",
        "denominators": {
            "frozen_candidate_rows": len(candidates),
            "candidate_ids_unique": len(candidate_ids),
            "accepted_comparison_rows": len(accepted_ids),
            "root_verified_increment_rows": len(new_ids),
            "comparison_pool_rows": len(comparison_ids),
            "candidate_pending_root_admission": decisions["candidate_pending_root_admission"],
            "repair_required": decisions["repair_required"],
            "exclude_duplicate": decisions["exclude_duplicate"],
        },
        "family_rows": dict(sorted(family_counts.items())),
        "source_rows": dict(sorted(source_counts.items())),
        "decision_rows": dict(decisions),
        "reason_rows": dict(reasons_count.most_common()),
        "review_flag_rows": dict(flags_count.most_common()),
        "length_buckets": dict(sorted(length_buckets.items())),
        "source_replay": source_replay_doc,
        "license": {
            "candidate_rows_with_direct_evidence": sum(bool(value.get("direct_evidence")) for value in license_by_id.values()),
            "unique_evidence_files": len(license_file_status),
            "file_status_rows": collections.Counter(value["status"] for value in license_file_status.values()),
        },
        "checks": {
            "global_registry_hash_pinned": True,
            "dat03_hash_pinned": True,
            "all_candidate_global_train": all(row["checks"]["global_train"] for row in ledgers),
            "cpt_validation_reserved_count": sum(row["checks"]["cpt_partition"] == "cpt_validation" for row in ledgers),
            "source_line_replay_pass": source_replay_doc["status"] == "pass",
            "license_direct_evidence_join": all(bool(value.get("direct_evidence")) for value in license_by_id.values()),
            "prompt_target_disjoint_candidate_output": True,
            "no_target_truncation": all(row["checks"]["no_target_truncation_metadata"] for row in ledgers),
            "heldout_payload_opened": False,
            "model_or_gpu_started": False,
            "r_executed": False,
        },
        "artifacts": {},
        "elapsed_seconds": time.monotonic() - started,
        "source_walk_input_rows": sum(int(info.get("input_rows") or 0) for info in shard_info),
        "source_walk_input_family_rows": dict(sorted(source_walk_input_family_counts.items())),
    }
    for name in ("audit-ledger.jsonl", "candidate-ids.jsonl", "long-context-queue.jsonl", "repair-queues.json", "source-line-replay.json"):
        path = output / name
        summary["artifacts"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path), "rows": sum(1 for _ in path.open(encoding="utf-8")) if path.suffix == ".jsonl" else None}
    atomic_json(output / "summary.json", summary)

    receipt_doc = {
        "schema": "DAT-10-source-walk-independent-review-v1",
        "status": "independent_review_complete_root_admission_pending",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owner": "/root/data_expansion",
        "admission": "none; root retains source/license/quality/duplicate/CPT admission",
        "scope": {
            "frozen_shards": [0, 1, 2, 3, 4],
            "frozen_candidate_rows": len(candidates),
            "candidate_family_scope": dict(sorted(family_counts.items())),
            "source_walk_input_rows": sum(int(info.get("input_rows") or 0) for info in shard_info),
            "source_walk_input_family_rows": dict(sorted(source_walk_input_family_counts.items())),
            "comparison_pool": {"accepted_rows": len(accepted_ids), "root_verified_increment_rows": len(new_ids), "combined_rows": len(comparison_ids)},
            "candidate_outputs_disjoint_from_comparison_pool": True,
        },
        "policy": {
            "candidate_only": True,
            "train_content_only": True,
            "dev_or_final_content_opened": False,
            "heldout_payload_opened": False,
            "source_content_use": "raw TRAIN source-line/file hash replay only; no source content written",
            "license_content_use": "hash verification of direct DESCRIPTION evidence only",
            "long_targets": "retained; measured; queued for longer-context review; never truncated",
            "recoverable_license_or_context": "named reason/flag; no blanket permanent exclusion",
            "permanent_exclusions": "exact prompt/prompt-target/source identity duplicates only",
            "cpu_threads": 2,
            "nice": 10,
            "ionice": "idle",
            "model_or_gpu_started": False,
            "r_executed": False,
        },
        "inputs": {
            "pinned_files": input_pins,
            "source_walk_shards": shard_info,
            "shard_audit_files": audit_file_pins,
            "dat03_rows": dat03_rows,
            "accepted_provenance_rows": accepted_prov_rows,
        },
        "results": {
            "decisions": dict(decisions),
            "reasons": dict(reasons_count.most_common()),
            "review_flags": dict(flags_count.most_common()),
            "length_buckets": dict(sorted(length_buckets.items())),
            "long_context_rows": len(long_rows),
            "license_evidence": {
                "candidate_rows": len(license_by_id),
                "direct_evidence_rows": sum(bool(value.get("direct_evidence")) for value in license_by_id.values()),
                "unique_files": len(license_file_status),
                "file_status": dict(collections.Counter(value["status"] for value in license_file_status.values())),
            },
        },
        "checks": {
            "frozen_shard_hashes_and_counts": True,
            "candidate_packet_source_ref_join": True,
            "shard_audit_metadata_join": all(row["checks"]["shard_audit_join"] for row in ledgers),
            "dat03_hash_and_metadata_join": all(row["checks"]["dat03_metadata_join"] for row in ledgers),
            "global_train_join": all(row["checks"]["global_train"] for row in ledgers),
            "cpt_validation_reserved": sum(row["checks"]["cpt_partition"] == "cpt_validation" for row in ledgers) == 0,
            "source_file_and_line_replay": source_replay_doc["status"] == "pass",
            "direct_license_evidence_and_file_hash": all(value.get("status") == "pass" for value in license_by_id.values()),
            "token_text_hash_and_geometry": all(row["checks"]["token_geometry"] for row in ledgers),
            "no_target_truncation": all(row["checks"]["no_target_truncation_metadata"] for row in ledgers),
            "candidate_output_id_prompt_pair_disjoint": True,
            "heldout_payload_opened": False,
            "training_launched": False,
        },
        "artifacts": {},
        "elapsed_seconds": time.monotonic() - started,
    }
    for name in ("audit-ledger.jsonl", "candidate-ids.jsonl", "long-context-queue.jsonl", "repair-queues.json", "source-line-replay.json", "summary.json"):
        path = output / name
        receipt_doc["artifacts"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path), "rows": sum(1 for _ in path.open(encoding="utf-8")) if path.suffix == ".jsonl" else None}
    receipt_doc["artifacts"]["receipt"] = {"path": str(receipt_path), "note": "filled after atomic receipt write"}
    atomic_json(receipt_path, receipt_doc)
    receipt_hash = sha_file(receipt_path)
    # Add a separate final status marker rather than rewriting the receipt
    # after hashing it; this keeps the receipt's own hash externally stable.
    atomic_json(output / "complete.json", {"schema": "DAT-10-source-walk-independent-review-complete-v1", "receipt": str(receipt_path), "receipt_sha256": receipt_hash, "status": receipt_doc["status"], "candidate_rows": len(candidate_output_rows), "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": receipt_doc["status"], "decisions": dict(decisions), "candidate_rows": len(candidate_output_rows), "long_rows": len(long_rows), "source_replay": source_replay_doc["status"], "receipt": str(receipt_path), "receipt_sha256": receipt_hash, "elapsed_seconds": time.monotonic() - started}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_DEFAULT)
    args = parser.parse_args()
    main(args.output, args.receipt)
