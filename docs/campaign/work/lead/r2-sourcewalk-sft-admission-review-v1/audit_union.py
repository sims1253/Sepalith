#!/usr/bin/env python3
"""Mechanical admission audit for the frozen DAT-10 source-walk rows.

This reads token rows once and writes only hashes, IDs, counts, and reasons. It
does not publish training data or make an admission decision.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def digest_tokens(xs: list[int]) -> str:
    return hashlib.sha256(json.dumps(xs, separators=(",", ":")).encode()).hexdigest()


def sha256_stream_rows(path: Path) -> Iterable[tuple[dict[str, Any], int]]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for line_no, raw in enumerate(f, 1):
            h.update(raw)
            yield json.loads(raw), line_no
    sha256_stream_rows.last = h.hexdigest()  # type: ignore[attr-defined]


def token_contract(r: dict[str, Any]) -> tuple[bool, list[str], str, str]:
    reasons: list[str] = []
    ids = r.get("input_ids")
    body = r.get("target_body_tokens")
    terminal = r.get("target_terminal_tokens")
    start = r.get("target_start")
    if not isinstance(ids, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in ids):
        return False, ["input_ids_not_integer_list"], "", ""
    if not isinstance(body, list) or not isinstance(terminal, list) or not isinstance(start, int):
        return False, ["target_fields_invalid"], "", ""
    if not ids or ids[0] != r.get("bos_token_id") or ids[-1] != r.get("eos_token_id"):
        reasons.append("special_token_boundary")
    expected = body + terminal + [r.get("eos_token_id")]
    if start < 0 or ids[start:] != expected:
        reasons.append("target_suffix_geometry")
    if len(body) != r.get("target_body_token_count"):
        reasons.append("target_body_count")
    if len(terminal) != r.get("target_terminal_token_count"):
        reasons.append("target_terminal_count")
    # Stored counts omit the explicit BOS and terminal EOS tokens.
    if len(ids) != r.get("prompt_token_count", -1) + r.get("target_token_count", -2) + 2:
        reasons.append("sequence_count")
    if r.get("split") != "train":
        reasons.append("split_not_train")
    if not r.get("target_text", "").endswith(">>>>>>> UPDATED"):
        reasons.append("protocol_terminal_text")
    return not reasons, reasons, digest_tokens(ids[:start]), digest_tokens(expected)


def count_dups(items: list[dict[str, Any]], key: str) -> tuple[int, int]:
    c = collections.Counter(x[key] for x in items if x.get(key))
    return sum(1 for n in c.values() if n > 1), sum(n - 1 for n in c.values() if n > 1)


def file_meta(path: Path, sha: str, rows: int) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "rows": rows, "sha256": sha}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    base = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1")
    sw_paths = [Path(x) for x in sorted(glob.glob(str(base / "shard-*" / "structured-materialization-v1" / "token-audit" / "candidate-token-rows.jsonl")))]
    current_path = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl")
    current_ctx = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/context-sidecar.jsonl")
    safe_path = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-roxy8597-materialization-v1/candidate-union-token-rows.jsonl")
    old_ids_path = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-roxy10017-root-context-review-v1/candidate-ids.json")
    sourcewalk_summary_path = Path("/mnt/e/sepalith/campaign-20260915/data-work/Noop-pool-coverage-review-v1/source-walk-all-family-summary.json")

    # Compact cross-pool indexes. Token hashes make renderer identity explicit.
    current: dict[str, dict[str, Any]] = {}
    current_pt: set[tuple[str, str]] = set()
    current_prompt_to_targets: dict[str, set[str]] = collections.defaultdict(set)
    gen = sha256_stream_rows(current_path)
    current_rows = 0
    for obj, line in gen:
        ok, reasons, ph, th = token_contract(obj)
        rid = obj["id"]
        current[rid] = {"prompt_token_sha256": ph, "target_token_sha256": th, "valid": ok, "reasons": reasons, "raw": None}
        current_pt.add((ph, th)); current_prompt_to_targets[ph].add(th); current_rows += 1
    current_sha = sha256_stream_rows.last  # type: ignore[attr-defined]

    ctx_h = hashlib.sha256(); ctx_rows = 0
    with current_ctx.open("rb") as f:
        for raw in f:
            ctx_h.update(raw); obj = json.loads(raw); ctx_rows += 1
            rid = obj["row_id"]
            if rid in current:
                sr = obj["source_identity"].get("source_ref", {})
                current[rid]["raw"] = sr.get("raw_line_sha256")
    current_raw = {v["raw"] for v in current.values() if v.get("raw")}

    safe: dict[str, dict[str, Any]] = {}
    safe_pt: set[tuple[str, str]] = set()
    safe_prompt_to_targets: dict[str, set[str]] = collections.defaultdict(set)
    safe_raw: set[str] = set()
    gen = sha256_stream_rows(safe_path)
    safe_rows = 0
    for obj, line in gen:
        r = obj["token_row"]
        ok, reasons, ph, th = token_contract(r)
        rid = obj["row_id"]; raw_sha = obj["provenance"].get("raw_line_sha256")
        safe[rid] = {"prompt_token_sha256": ph, "target_token_sha256": th, "valid": ok, "reasons": reasons, "raw": raw_sha, "dedup_status": obj.get("dedup_status")}
        safe_pt.add((ph, th)); safe_prompt_to_targets[ph].add(th)
        if raw_sha: safe_raw.add(raw_sha)
        safe_rows += 1
    safe_sha = sha256_stream_rows.last  # type: ignore[attr-defined]
    safe_internal_unique_pairs = len(safe_pt)
    safe_current_id_overlap = len(set(safe) & set(current))
    safe_current_exact_rows = sum(
        1 for v in safe.values()
        if (v["prompt_token_sha256"], v["target_token_sha256"]) in current_pt
    )
    safe_current_prompt_conflict_rows = sum(
        1 for v in safe.values()
        if v["prompt_token_sha256"] in current_prompt_to_targets
        and v["target_token_sha256"] not in current_prompt_to_targets[v["prompt_token_sha256"]]
    )
    safe_new_unique_pairs = len(safe_pt - current_pt)

    old_ids_obj = json.loads(old_ids_path.read_text())
    old_ids = set(old_ids_obj if isinstance(old_ids_obj, list) else old_ids_obj.get("ids", old_ids_obj.get("row_ids", [])))

    all_rows: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    mechanical_holds: dict[str, set[str]] = collections.defaultdict(set)
    family = collections.Counter(); packages = collections.Counter(); shards = collections.Counter()
    sourcewalk_ids: set[str] = set(); sourcewalk_raw: set[str] = set()
    prompt_to_targets: dict[str, set[str]] = collections.defaultdict(set)
    pt_first: dict[tuple[str, str], str] = {}
    source_first: dict[tuple[str, str], str] = {}
    duplicate_rows: dict[str, list[str]] = collections.defaultdict(list)
    source_duplicate_rows: dict[str, list[str]] = collections.defaultdict(list)
    cross = collections.Counter()
    sw_contract_bad: list[dict[str, Any]] = []
    token_report_mismatch: list[dict[str, Any]] = []

    for p in sw_paths:
        shard = p.parts[-4]
        gen = sha256_stream_rows(p); n = 0
        for obj, line in gen:
            n += 1; r = obj["row"]; rid = r["id"]
            ok, reasons, ph, th = token_contract(r)
            declared_ph, declared_th = obj.get("prompt_sha256"), obj.get("target_sha256")
            text_ph = hashlib.sha256(r["prompt_text"].encode()).hexdigest()
            text_th = hashlib.sha256(r["target_text"].encode()).hexdigest()
            if declared_ph != text_ph or declared_th != text_th:
                token_report_mismatch.append({"row_id": rid, "reason": "declared_text_hash_mismatch"})
                mechanical_holds["declared_text_hash_mismatch"].add(rid)
            if not ok:
                mechanical_holds["token_or_protocol_contract"].add(rid)
                sw_contract_bad.append({"row_id": rid, "reasons": reasons})
            fam = r["family"]; family[fam] += 1; packages[r["package_id"]] += 1; shards[shard] += 1
            raw_sha = obj["source_ref"].get("raw_line_sha256")
            if rid in sourcewalk_ids: mechanical_holds["duplicate_row_id"].add(rid)
            sourcewalk_ids.add(rid)
            if raw_sha: sourcewalk_raw.add(raw_sha)
            # Leakage is meaningful for edit targets. No-op intentionally repeats unchanged text.
            if fam != "no_op" and r.get("target_body_text") and r["target_body_text"] in r["prompt_text"]:
                mechanical_holds["target_body_in_prompt"].add(rid)
            if rid in old_ids:
                mechanical_holds["replaced_by_roxy10017_review_path"].add(rid)
            if rid in safe:
                cross["safe_id_overlap"] += 1
                if raw_sha and raw_sha == safe[rid].get("raw"): cross["safe_same_id_raw_parity"] += 1
                if th == safe[rid]["target_token_sha256"]: cross["safe_same_id_target_token_parity"] += 1
            if rid in current: cross["current_id_overlap"] += 1
            if raw_sha in current_raw: cross["current_raw_line_overlap_rows"] += 1
            if raw_sha in safe_raw: cross["safe_raw_line_overlap_rows"] += 1
            if (ph, th) in current_pt:
                cross["current_exact_prompt_target_rows"] += 1; mechanical_holds["exact_duplicate_current15006"].add(rid)
            if ph in current_prompt_to_targets and th not in current_prompt_to_targets[ph]:
                cross["current_prompt_conflict_rows"] += 1; mechanical_holds["prompt_conflict_current15006"].add(rid)
            if (ph, th) in safe_pt:
                cross["safe_exact_prompt_target_rows"] += 1; mechanical_holds["exact_duplicate_safe8597"].add(rid)
            if ph in safe_prompt_to_targets and th not in safe_prompt_to_targets[ph]:
                cross["safe_prompt_conflict_rows"] += 1; mechanical_holds["prompt_conflict_safe8597"].add(rid)
            prompt_to_targets[ph].add(th)
            pt = (ph, th)
            if pt in pt_first:
                mechanical_holds["within_sourcewalk_exact_prompt_target_duplicate"].add(rid)
                duplicate_rows[pt_first[pt]].append(rid)
            else: pt_first[pt] = rid
            source_key = (raw_sha or "", th)
            if raw_sha and source_key in source_first:
                mechanical_holds["within_sourcewalk_source_target_duplicate"].add(rid)
                source_duplicate_rows[source_first[source_key]].append(rid)
            elif raw_sha: source_first[source_key] = rid
            all_rows.append({"row_id": rid, "family": fam, "package_id": r["package_id"], "shard": shard, "prompt_token_sha256": ph, "target_token_sha256": th, "prompt_text_sha256": text_ph, "target_text_sha256": text_th, "raw_line_sha256": raw_sha})
        input_files.append(file_meta(p, sha256_stream_rows.last, n))  # type: ignore[attr-defined]

    contradiction_prompts = {p for p, ts in prompt_to_targets.items() if len(ts) > 1}
    contradiction_ids = {x["row_id"] for x in all_rows if x["prompt_token_sha256"] in contradiction_prompts}
    mechanical_holds["within_sourcewalk_prompt_conflict"].update(contradiction_ids)
    hold_union = set().union(*mechanical_holds.values()) if mechanical_holds else set()
    old_overlap = sourcewalk_ids & old_ids
    # Rows mechanically unique after replacing the entire older 10,017 path. Roxygen
    # semantic support is still a review gate and is not included in ready_now.
    mechanical_frontier = len(sourcewalk_ids - hold_union)
    noop_frontier = sum(1 for x in all_rows if x["family"] == "no_op" and x["row_id"] not in hold_union)
    roxy_frontier = sum(1 for x in all_rows if x["family"] == "roxygen_drafting" and x["row_id"] not in hold_union)

    hold_path = a.out / "mechanical-hold-ledger.jsonl"
    with hold_path.open("w") as f:
        by_id: dict[str, list[str]] = collections.defaultdict(list)
        for reason, ids in mechanical_holds.items():
            for rid in ids: by_id[rid].append(reason)
        for rid in sorted(by_id):
            f.write(json.dumps({"row_id": rid, "reasons": sorted(by_id[rid])}, sort_keys=True) + "\n")
    duplicate_path = a.out / "duplicate-groups.jsonl"
    with duplicate_path.open("w") as f:
        for first, extras in sorted(duplicate_rows.items()):
            f.write(json.dumps({"canonical_row_id": first, "duplicate_row_ids": sorted(extras), "basis": "prompt_and_target_token_hash"}, sort_keys=True) + "\n")
    unresolved_path = a.out / "semantic-review-queues.json"
    unresolved = {
        "converted_roxygen_outside_reviewed_10017": roxy_frontier,
        "gate": "full-context target-support review using R lexical and occurrence-based NSE evidence; missing names alone are not exclusion evidence",
        "preconversion_documentation_external_facts": 109266,
        "preconversion_alternate_noop_kind": 4384,
        "preconversion_missing_description_license_provenance": 144,
        "preconversion_parent_parse_error": 49,
    }
    unresolved_path.write_text(json.dumps(unresolved, indent=2, sort_keys=True) + "\n")

    summary = json.loads(sourcewalk_summary_path.read_text())
    pre = summary["counts"]
    report = {
        "schema": "sepalith.dat10.sourcewalk_sft_admission_review.v1",
        "status": "complete_review_only_no_admission",
        "policy": {"cpu_threads_max": 2, "gpu": False, "model_loaded": False, "final_or_dev_access": False, "training_admission": False},
        "inputs": {
            "sourcewalk_token_rows": input_files,
            "current15006": file_meta(current_path, current_sha, current_rows),
            "current15006_context": file_meta(current_ctx, ctx_h.hexdigest(), ctx_rows),
            "safe_roxy8597": file_meta(safe_path, safe_sha, safe_rows),
            "old_roxy10017_ids": {"path": str(old_ids_path), "rows": len(old_ids), "sha256": hashlib.sha256(old_ids_path.read_bytes()).hexdigest()},
            "sourcewalk_summary": {"path": str(sourcewalk_summary_path), "sha256": hashlib.sha256(sourcewalk_summary_path.read_bytes()).hexdigest()},
        },
        "preconversion": {
            "input_rows": pre["converted_rows"] + pre["excluded_rows"],
            "input_by_family": {"no_op": 8612, "roxygen_drafting": 192141},
            "converted_rows": pre["converted_rows"],
            "converted_by_family": pre["converted_by_family"],
            "excluded_rows": pre["excluded_rows"],
            "exclusion_reasons": pre["exclusion_reasons"],
            "interpretation": {
                "documentation_external_facts_require_separate_support_review": "repair queue; not proof the target is false",
                "no_op_kind_requires_separate_support_review": "semantic-kind queue; not a length cap",
                "missing_DESCRIPTION": "license/provenance repair queue",
                "normalized_parent_R_parse_error": "actual source parse blocker until repaired",
            },
        },
        "converted_audit": {
            "rows": len(all_rows), "unique_ids": len(sourcewalk_ids), "families": dict(family), "packages": len(packages), "shards": len(shards),
            "token_contract_failures": len(sw_contract_bad), "declared_text_hash_mismatches": len(token_report_mismatch),
            "prompt_conflict_groups": len(contradiction_prompts), "prompt_conflict_rows": len(contradiction_ids),
            "within_prompt_target_duplicate_groups": len(duplicate_rows), "within_prompt_target_duplicate_extras": sum(map(len, duplicate_rows.values())),
            "within_source_target_duplicate_groups": len(source_duplicate_rows), "within_source_target_duplicate_extras": sum(map(len, source_duplicate_rows.values())),
            "mechanical_hold_reason_counts": {k: len(v) for k, v in sorted(mechanical_holds.items())},
            "mechanical_hold_union": len(hold_union), "mechanically_unique_frontier": mechanical_frontier,
            "mechanically_unique_frontier_by_family": {"no_op": noop_frontier, "roxygen_drafting": roxy_frontier},
        },
        "cross_pool": {
            **dict(cross),
            "old_roxy10017_id_rows": len(old_ids), "old_roxy10017_sourcewalk_id_overlap": len(old_overlap),
            "safe_roxy8597_ids": len(safe), "safe_roxy_internal_duplicate_rows_retained_by_producer": sum(1 for x in safe.values() if x.get("dedup_status") != "new_candidate"),
            "safe_vs_authoritative_current15006": {
                "id_overlap_rows": safe_current_id_overlap,
                "exact_prompt_target_rows": safe_current_exact_rows,
                "prompt_conflict_rows": safe_current_prompt_conflict_rows,
                "internal_unique_prompt_target_pairs": safe_internal_unique_pairs,
                "new_unique_prompt_target_pairs": safe_new_unique_pairs,
                "note": "recomputed against repaired train-token-rows SHA 3f551c..., superseding the producer's obsolete 65b2... dedup claim",
            },
            "rule": "safe8597 is a replacement view for the reviewed old roxy source identities; never add its older sourcewalk counterpart",
        },
        "union_frontiers": {
            "existing15006": current_rows,
            "reviewed_safe_roxy_rows": safe_rows,
            "reviewed_safe_roxy_new_after_accepted_dedup": safe_new_unique_pairs,
            "mechanically_ready_sourcewalk_noops_pending_root_admission": noop_frontier,
            "strongest_current_review_frontier": current_rows + safe_new_unique_pairs + noop_frontier,
            "additional_sourcewalk_roxygen_mechanically_unique_but_semantic_review_pending": roxy_frontier,
            "maximum_candidate_frontier_if_all_pending_roxygen_pass": current_rows + safe_new_unique_pairs + noop_frontier + roxy_frontier,
            "note": "counts are candidate frontiers, not admission; semantic review and root decision remain mandatory",
        },
        "remaining_checks": [
            "reconstruct full-source context for every remaining roxygen row and verify the target is supported by visible function signatures and source occurrences",
            "apply R lexical-scope and occurrence-based NSE analysis; absence from a global-name list is not an exclusion",
            "hold actual conflicting prompt labels, protocol/token geometry failures, parse failures, and unsupported invented documentation",
            "root-review direct license/provenance bindings and protected split/dedup receipts for the union",
            "materialize a fresh union with explicit replacement/dedup ledger and rerun renderer/token parity without target truncation",
        ],
        "outputs": {
            "mechanical_hold_ledger": {"path": str(hold_path), "rows": len(hold_union)},
            "duplicate_groups": {"path": str(duplicate_path), "rows": len(duplicate_rows)},
            "semantic_review_queues": {"path": str(unresolved_path)},
        },
    }
    report_path = a.out / "review.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for key in ("mechanical_hold_ledger", "duplicate_groups", "semantic_review_queues"):
        p = Path(report["outputs"][key]["path"])
        report["outputs"][key].update({"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "converted": len(all_rows), "holds": len(hold_union), "frontiers": report["union_frontiers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
