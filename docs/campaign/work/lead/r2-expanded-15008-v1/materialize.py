#!/usr/bin/env python3
"""Materialize the independently accepted DAT10 increment without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "packet": "f4ee5e76a4c47f83c99137b64e4e2c95fd1e19a0c619d79cdc583f2918ba633a",
    "accepted_ids": "ac4d6fdfb50cb82667f2cd26444e71b8d204a0c6ac4489fc450571e34f1a0ec0",
    "audit_ledger": "d6f91d2e096265e6854eabdc6ea915f9e1f850cfa80bf2d418d5a7b3a88bbccd",
    "repair_queues": "d8ea8f123aa2674501d22de36bfb8ae493a26813e0169dc186ab425ae2ad44e4",
    "source_replay": "b0cd7c3e811ee908e8f76f3ec5651b747bcc8609da5b7e24cd1f4b9c116b87c4",
    "current_rows": "fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889",
    "current_provenance": "317077b17afc38602309361396bb86d8897a6b823c5d7539f10b9733a0ad9a48",
    "tokenizer_json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            count += 1
    return count


def percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile, with ranks 1..N."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values: list[int], bins: list[int]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    low = 0
    for high in bins:
        label = f"{low}-{high}" if low else f"0-{high}"
        counts[label] = sum(low <= value <= high for value in values)
        low = high + 1
    counts[f">={low}"] = sum(value >= low for value in values)
    return {
        "count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.fmean(values), "p50_nearest_rank": percentile(values, .5),
        "p90_nearest_rank": percentile(values, .9), "p95_nearest_rank": percentile(values, .95),
        "p99_nearest_rank": percentile(values, .99), "bins": counts,
    }


def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-output", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    published_output = args.published_output or args.output
    started = time.monotonic()
    if args.output.exists():
        raise SystemExit(f"refusing non-fresh output: {args.output}")
    args.output.mkdir(parents=True)

    work = args.plan / "docs/campaign/work/lead"
    inputs = {
        "packet": Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/expansion-increment-na-rm-v1/root-review-packet-v4-na-rm.jsonl"),
        "accepted_ids": work / "r2-novel-independent-review-v1/admissible-ids.jsonl",
        "audit_ledger": work / "r2-novel-independent-review-v1/audit-ledger.jsonl",
        "repair_queues": work / "r2-novel-independent-review-v1/repair-queues.json",
        "source_replay": work / "r2-novel-independent-review-v1/source-line-replay.json",
        "current_rows": work / "r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl",
        "current_provenance": work / "r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl",
        "tokenizer_json": args.tokenizer / "tokenizer.json",
        "tokenizer_config": args.tokenizer / "tokenizer_config.json",
    }
    input_meta = {}
    for name, path in inputs.items():
        actual = sha(path)
        if actual != EXPECTED[name]:
            raise AssertionError(f"{name} SHA mismatch: {actual}")
        input_meta[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}

    source_replay = json.loads(inputs["source_replay"].read_text())
    assert source_replay["status"] == "pass" and source_replay["matched_rows"] == 5245
    audit = {x["row_id"]: x for x in read_jsonl(inputs["audit_ledger"])}
    admitted_order = read_jsonl(inputs["accepted_ids"])
    admitted_ids = [x["row_id"] for x in admitted_order]
    assert len(admitted_ids) == len(set(admitted_ids)) == 3503
    assert all(audit[x]["decision"] == "admissible" and not audit[x]["reasons"] for x in admitted_ids)

    packet = read_jsonl(inputs["packet"])
    assert len(packet) == 5245
    packet_by_id = {x["row"]["id"]: x for x in packet}
    assert len(packet_by_id) == 5245 and set(admitted_ids) <= packet_by_id.keys()
    selected = [packet_by_id[row_id] for row_id in admitted_ids]

    # Import only tokenizer/protocol code. No model config or weight file is opened.
    from transformers import AutoTokenizer
    from sepalith.campaign_protocol import PromptContext, build_training_row, render_prompt
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)

    geometry_rows = []
    candidate_receipts = []
    rebuilt_rows = []
    provenance_rows = []
    for position, record in enumerate(selected):
        row = record["row"]
        row_id = row["id"]
        context_raw = record["context"]
        context = PromptContext.from_mapping(context_raw)
        rendered = render_prompt(context)
        if row["target_operation"] == "no_op":
            region_new = context_raw["region_old"]
        elif row["target_operation"] == "delete":
            region_new = []
        else:
            region_new = row["target_body_text"].split("\n")
        rebuilt = build_training_row(
            context, operation=row["target_operation"], region_new=region_new,
            tokenizer=tokenizer, row_id=row_id, family=row["family"],
            package_id=row["package_id"], split=row["split"],
        )
        assert rebuilt == row
        assert hashlib.sha256(rendered.encode()).hexdigest() == record["prompt_sha256"]
        assert hashlib.sha256(row["target_text"].encode()).hexdigest() == record["target_sha256"]

        selection = record["selection"]
        selected_region = selection["region"]
        canonical_selected_region = [] if selected_region == [""] else selected_region
        source_ref = record["source_ref"]
        source_provenance = record["source_provenance"]
        source_identity = source_provenance["source_identity"]
        geom = source_provenance["region_geometry"]
        checks = {
            "rebuilt_row_exact": rebuilt == row,
            "rendered_prompt_exact": rendered == row["prompt_text"],
            "packet_prompt_sha_exact": hashlib.sha256(rendered.encode()).hexdigest() == record["prompt_sha256"],
            "packet_target_sha_exact": hashlib.sha256(row["target_text"].encode()).hexdigest() == record["target_sha256"],
            "selection_prefix_exact": selection["prefix"] == context_raw["prefix"],
            "selection_suffix_exact": selection["suffix"] == context_raw["suffix_lines"],
            "selection_region_canonical_exact": canonical_selected_region == context_raw["region_old"],
            "document_sha_exact": selection["document_sha256"] == context_raw["replacement_range"]["content_sha256"],
            "range_start_exact": selection["spans"][-1]["start_line"] == context_raw["replacement_range"]["start"]["line"],
            "range_end_exact": selection["spans"][-1]["end_line"] == context_raw["replacement_range"]["end"]["line"],
            "provenance_geometry_start_exact": geom["start_line"] == context_raw["replacement_range"]["start"]["line"],
            "provenance_geometry_end_exact": geom["end_line"] == context_raw["replacement_range"]["end"]["line"],
            "source_file_exact": source_ref["file"] == source_identity["file"],
            "source_line_exact": source_ref["line"] == source_identity["line"],
            "source_raw_line_sha_exact": source_ref["raw_line_sha256"] == source_identity["raw_line_sha256"],
            "source_file_sha_exact": source_ref["source_sha256"] == source_identity["source_sha256"],
        }
        if not all(checks.values()):
            raise AssertionError(f"geometry/source join failed for {row_id}: {checks}")
        geometry_rows.append({
            "row_id": row_id, "position": position, "checks": checks,
            "prompt_sha256": record["prompt_sha256"], "target_sha256": record["target_sha256"],
            "rendered_prompt_utf16_units": utf16_units(rendered),
            "prompt_tokens_without_bos": row["prompt_token_count"],
            "prompt_tokens_with_bos": row["target_start"],
            "target_label_tokens_including_protocol_eos": len(row["input_ids"]) - row["target_start"],
            "sequence_tokens": len(row["input_ids"]),
            "context_line_counts": {name: len(context_raw[name]) for name in ("prefix", "scope_lines", "suffix_lines", "region_old")},
            "selection_used_utf16_units": selection["used_utf16_units"],
            "selection_required_utf16_units": selection["required_utf16_units"],
            "selection_overflow": selection["overflow"],
            "selection_omissions": selection["omissions"],
        })
        candidate_receipts.append({
            "schema": "sepalith.dat10.expanded-candidate-receipt.v1", "row_id": row_id,
            "decision": "accepted_candidate_pending_root_admission", "independent_audit_index": audit[row_id]["index"],
            "audit_reasons": audit[row_id]["reasons"], "repair_reasons": audit[row_id]["repair_reasons"],
            "permanent_exclusion_reasons": audit[row_id]["permanent_exclusion_reasons"],
            "license": source_ref["license"], "license_status": source_ref["license_status"],
            "license_evidence_present": source_ref["license_evidence_present"], "source_url": source_ref["source_url"],
            "source_file": source_ref["file"], "source_file_sha256": source_ref["source_sha256"],
            "source_line": source_ref["line"], "source_raw_line_sha256": source_ref["raw_line_sha256"],
            "source_line_replay_receipt_sha256": EXPECTED["source_replay"], "source_line_replay_pass": True,
            "group_id": source_ref["group_id"], "global_split": source_ref["split"],
            "prompt_sha256": record["prompt_sha256"], "target_sha256": record["target_sha256"],
            "row_canonical_sha256": canonical_sha(row), "tokenizer_reencode_exact": True,
            "render_context_geometry_exact": True,
        })
        rebuilt_rows.append(row)
        provenance_rows.append({
            "row_id": row_id, "context": context_raw, "selection": selection,
            "source_provenance": source_provenance, "source_ref": source_ref,
            "prompt_sha256": record["prompt_sha256"], "target_sha256": record["target_sha256"],
        })

    assert all(x["source_ref"]["license_evidence_present"] and x["source_ref"]["license_status"] == "direct_row_evidence" for x in selected)
    current_rows = read_jsonl(inputs["current_rows"])
    current_provenance = read_jsonl(inputs["current_provenance"])
    assert len(current_rows) == len(current_provenance) == 11505
    current_ids = [x["id"] for x in current_rows]
    assert len(set(current_ids)) == 11505
    assert {x["id"] for x in current_provenance} == set(current_ids)
    assert set(current_ids).isdisjoint(admitted_ids)
    combined = current_rows + rebuilt_rows
    assert len(combined) == len({x["id"] for x in combined}) == 15008
    assert len({hashlib.sha256(x["prompt_text"].encode()).hexdigest() for x in combined}) == 15008

    write_jsonl(args.output / "candidate-token-rows.jsonl", rebuilt_rows)
    write_jsonl(args.output / "candidate-context-provenance.jsonl", provenance_rows)
    write_jsonl(args.output / "candidate-per-example-receipts.jsonl", candidate_receipts)
    write_jsonl(args.output / "tokenizer-render-geometry-audit.jsonl", geometry_rows)
    write_jsonl(args.output / "combined-token-rows.jsonl", combined)
    write_jsonl(args.output / "combined-row-origin.jsonl", [
        {"position": i, "row_id": row["id"], "origin": "accepted_11505" if i < 11505 else "dat10_new_3503"}
        for i, row in enumerate(combined)
    ])
    # Keep every review disposition and its reasons available beside the accepted pool.
    dispositions = []
    for record in packet:
        row_id = record["row"]["id"]
        source_ref = record["source_ref"]
        dispositions.append({
            **audit[row_id], "license": source_ref.get("license"),
            "license_status": source_ref.get("license_status"),
            "license_evidence_present": source_ref.get("license_evidence_present"),
            "source_file": source_ref.get("file"), "source_line": source_ref.get("line"),
            "source_raw_line_sha256": source_ref.get("raw_line_sha256"),
        })
    write_jsonl(args.output / "review-disposition-ledger.jsonl", dispositions)

    measures = {
        "target_body_tokens": [x["target_body_token_count"] for x in rebuilt_rows],
        "target_label_tokens_including_protocol_eos": [len(x["input_ids"]) - x["target_start"] for x in rebuilt_rows],
        "prompt_tokens_without_bos": [x["prompt_token_count"] for x in rebuilt_rows],
        "prompt_tokens_with_bos": [x["target_start"] for x in rebuilt_rows],
        "sequence_tokens": [len(x["input_ids"]) for x in rebuilt_rows],
        "rendered_prompt_utf16_units": [x["rendered_prompt_utf16_units"] for x in geometry_rows],
        "selection_used_utf16_units": [x["selection_used_utf16_units"] for x in geometry_rows],
    }
    length_report = {
        "schema": "sepalith.dat10.expanded-15008-lengths.v1", "rows": 3503,
        "family_rows": dict(Counter(x["family"] for x in rebuilt_rows)),
        "operation_rows": dict(Counter(x["target_operation"] for x in rebuilt_rows)),
        "distributions": {
            name: distribution(values, [64, 128, 192, 256, 384, 512, 1024, 2048, 4096])
            for name, values in measures.items()
        },
        "threshold_counts": {
            "target_label_le_192": sum(x <= 192 for x in measures["target_label_tokens_including_protocol_eos"]),
            "target_label_gt_192": sum(x > 192 for x in measures["target_label_tokens_including_protocol_eos"]),
            "sequence_le_2048": sum(x <= 2048 for x in measures["sequence_tokens"]),
            "sequence_2049_to_4096": sum(2048 < x <= 4096 for x in measures["sequence_tokens"]),
            "sequence_gt_4096": sum(x > 4096 for x in measures["sequence_tokens"]),
        },
        "coverage_policy": "reported_without_length_filtering_or_truncation",
    }
    write_json(args.output / "length-distribution.json", length_report)

    family_counts = dict(sorted(Counter(x["family"] for x in combined).items()))
    noop = sum(x["target_operation"] == "no_op" for x in combined)
    edit = len(combined) - noop
    assert noop == 1094 and edit == 13914
    sequence_4096_ids = [x["id"] for x in combined if 2048 < len(x["input_ids"]) <= 4096]
    sequence_over_4096_ids = [x["id"] for x in combined if len(x["input_ids"]) > 4096]
    assert len(sequence_4096_ids) == 1454 and not sequence_over_4096_ids
    sequence_4096_edit_ids = [x["id"] for x in combined if 2048 < len(x["input_ids"]) <= 4096 and x["target_operation"] != "no_op"]
    sequence_4096_noop_ids = [x["id"] for x in combined if 2048 < len(x["input_ids"]) <= 4096 and x["target_operation"] == "no_op"]
    assert len(sequence_4096_edit_ids) == 1451 and len(sequence_4096_noop_ids) == 3
    schedule = {
        "schema": "sepalith.dat10.expanded-15008-schedule-proposal.v1",
        "status": "proposal_only_requires_root_and_training_source_review",
        "pool": {"unique_rows": 15008, "edit_rows": edit, "no_op_rows": noop, "family_rows": family_counts},
        "recommended_preserve_25pct_no_op": {
            "effective_batch": 16, "edit_draws_per_batch": 12, "no_op_draws_per_batch": 4,
            "minimum_updates_covering_all_unique_edits": 1160, "total_draws": 18560,
            "edit_draws": 13920, "no_op_draws": 4640,
            "edit_repeats_after_full_edit_coverage": 6,
            "no_op_repeats_after_full_no_op_coverage": 3546,
            "ordering_requirement": "deterministic seeded queues; exhaust each pool before its own repeats; never omit a row by length",
            "length_bucket_plan": {
                "bucket_4096_batches": 121, "bucket_2048_batches": 1039,
                "bucket_4096_unique_long_rows": 1454,
                "bucket_4096_unique_long_edits": 1451, "bucket_4096_unique_long_no_ops": 3,
                "bucket_4096_long_row_ids_sha256": hashlib.sha256(("\n".join(sequence_4096_ids) + "\n").encode()).hexdigest(),
                "bucket_4096_fill": "1451 unique long edits plus 1 unique short edit fill 1452 edit slots. Include all 3 unique long no-ops plus 481 unique short no-ops in the 484 no-op slots.",
                "bucket_2048_coverage": "cover the remaining 12462 unique short edits and 610 not-yet-drawn unique short no-ops before any respective pool replay; only then use 6 edit repeats and 3546 no-op repeats",
                "global_no_op_ordering": "all 1094 unique no-ops are drawn once before any no-op replay; the 3 long no-ops are assigned to 4096-capable batches and are not preferentially replayed",
            },
        },
        "alternative_exact_one_pass": {
            "effective_batch": 16, "updates": 938, "total_draws": 15008,
            "no_op_fraction": noop / 15008,
            "tradeoff": "covers each row exactly once but does not preserve the established 25 percent no-op draw policy",
        },
        "incompatibility": "1000 updates at 12 edit draws per batch has only 12000 edit slots and cannot cover all 13914 unique edits",
        "length_policy": "all rows scheduled; no cap, truncation, or silent length exclusion",
    }
    write_json(args.output / "schedule-proposal.json", schedule)

    artifacts = {}
    for path in sorted(args.output.iterdir()):
        if path.name == "manifest.json":
            continue
        artifacts[path.name] = {"path": str(published_output / path.name), "bytes": path.stat().st_size, "sha256": sha(path)}
    manifest = {
        "schema": "sepalith.dat10.expanded-15008.v1",
        "status": "candidate_only_root_admission_pending",
        "inputs": input_meta,
        "counts": {"existing_rows": 11505, "new_candidate_rows": 3503, "combined_rows": 15008,
                   "review_rows": 5245, "review_repair_required": 1572, "review_exclude_duplicate": 170},
        "checks": {
            "input_hashes_exact": True, "new_ids_match_independent_admission": True,
            "tokenizer_reencode_exact_3503_of_3503": True, "render_exact_3503_of_3503": True,
            "context_selection_source_geometry_exact_3503_of_3503": True,
            "direct_license_evidence_3503_of_3503": True, "source_line_replay_3503_of_3503": True,
            "combined_ids_unique_15008_of_15008": True, "combined_prompts_unique_15008_of_15008": True,
            "model_instantiated": False, "model_weights_opened": False, "generated_r_executed": False,
            "heldout_payload_opened": False, "length_filtering_or_truncation": False,
        },
        "tokenizer": {"path": str(args.tokenizer), "implementation": "AutoTokenizer only; local_files_only; trust_remote_code false",
                      "tokenizer_json_sha256": EXPECTED["tokenizer_json"], "tokenizer_config_sha256": EXPECTED["tokenizer_config"]},
        "artifacts": artifacts, "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
