#!/usr/bin/env python3
"""Recompute selected CPT exposure from persisted cursors and frozen row schedules.

This script reads JSON metadata and streams the two admitted token-row JSONL files.
It does not load model weights, import a model framework, or use CUDA. It deliberately
does not hash the large JSONL payloads; instead it cross-checks their already-recorded
identity through the recipe, schedule, checkpoint state, and shard manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RELATIVE_SPECS = (
    {
        "name": "broad750",
        "recipe": "docs/campaign/work/lead/r2-cpt-broad-b/recipe.json",
        "rows": "docs/campaign/work/r2-corpus-preparation-v1/broader-shard-v1-2k/cpt_train.jsonl",
        "manifest": "docs/campaign/work/r2-corpus-preparation-v1/broader-shard-v1-2k/manifest.json",
        "schedule": "docs/campaign/work/r2-cpt-long-stage-review-v1/draws-1714.json",
        "checkpoint": "checkpoints/SFT11-CPT-broad-b/full/checkpoint-750",
        "expected_step": 750,
    },
    {
        "name": "global250",
        "recipe": "docs/campaign/work/lead/r2-cpt-global-a/recipe.json",
        "rows": "docs/campaign/work/r2-cpt-global-shard-v1/shard/cpt_train.jsonl",
        "manifest": "docs/campaign/work/r2-cpt-global-shard-v1/shard/manifest.json",
        "schedule": "docs/campaign/work/r2-cpt-global-stage-preparation-v1/draws-1187.json",
        "checkpoint": "checkpoints/SFT11-CPT-global-a/full/checkpoint-250",
        "expected_step": 250,
    },
)


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator


def audit_stage(plan: Path, native: Path, spec: dict[str, Any]) -> dict[str, Any]:
    recipe_path = plan / spec["recipe"]
    rows_path = plan / spec["rows"]
    manifest_path = plan / spec["manifest"]
    schedule_path = plan / spec["schedule"]
    checkpoint_path = native / spec["checkpoint"]

    recipe = load(recipe_path)
    manifest = load(manifest_path)
    schedule = load(schedule_path)
    campaign_state = load(checkpoint_path / "campaign-state.json")
    trainer_state = load(checkpoint_path / "trainer_state.json")

    expected_step = spec["expected_step"]
    effective_batch = schedule["effective_batch"]
    consumed_draws = campaign_state["sampler"]["consumed_draws"]
    recorded_schedule_hash = sha256(schedule_path)
    declared_rows_hash = recipe["train_rows"]["sha256"]

    assert campaign_state["step"] == expected_step
    assert trainer_state["global_step"] == expected_step
    assert consumed_draws == expected_step * effective_batch
    assert campaign_state["sampler"]["schedule_sha256"] == recorded_schedule_hash
    assert recipe["draw_schedule"]["sha256"] == recorded_schedule_hash
    assert recipe["identity"]["data"]["draw_schedule_sha256"] == recorded_schedule_hash
    assert schedule["token_rows_sha256"] == declared_rows_hash
    assert recipe["identity"]["data"]["train_rows_sha256"] == declared_rows_hash
    assert campaign_state["identity"]["data"]["train_rows_sha256"] == declared_rows_hash
    assert campaign_state["identity"]["data"]["draw_schedule_sha256"] == recorded_schedule_hash
    assert Path(recipe["train_rows"]["path"]).resolve() == rows_path.resolve()
    assert Path(recipe["draw_schedule"]["path"]).resolve() == schedule_path.resolve()
    assert manifest["artifacts"]["cpt_train.jsonl"]["sha256"] == declared_rows_hash

    rows: dict[str, dict[str, Any]] = {}
    document_rows: dict[str, set[str]] = defaultdict(set)
    package_rows: dict[str, set[str]] = defaultdict(set)
    prepared = Counter()
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = row["row_id"]
            assert row_id not in rows
            metadata = {
                "document_id": row["document_id"],
                "package": row["package"],
                "group_id": row["group_id"],
                "input_tokens": len(row["input_ids"]),
                "loss_tokens": sum(label != -100 for label in row["labels"]),
                "payload_tokens": row["source_token_end"] - row["source_token_start"],
                "overlap_context_tokens": row["overlap_context_tokens"],
            }
            rows[row_id] = metadata
            document_rows[metadata["document_id"]].add(row_id)
            package_rows[metadata["package"]].add(row_id)
            prepared["rows"] += 1
            for key in ("input_tokens", "loss_tokens", "payload_tokens", "overlap_context_tokens"):
                prepared[key] += metadata[key]

    manifest_counts = manifest["counts"]["cpt_train"]
    assert prepared["rows"] == manifest_counts["rows"]
    assert len(document_rows) == manifest_counts["documents"]
    assert len(package_rows) == manifest_counts["packages"]
    assert prepared["input_tokens"] == manifest_counts["input_tokens"]
    assert prepared["loss_tokens"] == manifest_counts["supervised_tokens"]
    assert prepared["payload_tokens"] == manifest_counts["code_tokens"]

    prefix = schedule["row_ids"][:consumed_draws]
    assert len(prefix) == consumed_draws
    counts = Counter(prefix)
    assert not (set(prefix) - set(rows))
    consumed_rows = set(prefix)
    consumed_documents = {rows[row_id]["document_id"] for row_id in consumed_rows}
    consumed_packages = {rows[row_id]["package"] for row_id in consumed_rows}
    consumed_groups = {rows[row_id]["group_id"] for row_id in consumed_rows}
    totals = Counter()
    for row_id in prefix:
        for key in ("input_tokens", "loss_tokens", "payload_tokens", "overlap_context_tokens"):
            totals[key] += rows[row_id][key]

    complete_documents = {
        document_id for document_id in consumed_documents
        if document_rows[document_id] <= consumed_rows
    }
    partial_documents = consumed_documents - complete_documents
    complete_packages = {
        package for package in consumed_packages
        if package_rows[package] <= consumed_rows
    }
    partial_packages = consumed_packages - complete_packages
    remaining_rows = set(rows) - consumed_rows
    future = schedule["row_ids"][consumed_draws:]

    return {
        "name": spec["name"],
        "paths": {
            "recipe": str(recipe_path),
            "rows": str(rows_path),
            "manifest": str(manifest_path),
            "schedule": str(schedule_path),
            "checkpoint": str(checkpoint_path),
        },
        "identity": {
            "source_sha256": recipe["identity"]["source"],
            "train_rows_declared_sha256": declared_rows_hash,
            "draw_schedule_actual_sha256": recorded_schedule_hash,
            "checkpoint_step": expected_step,
            "effective_batch": effective_batch,
            "sampler_consumed_draws": consumed_draws,
        },
        "prepared": {
            "rows": len(rows),
            "documents": len(document_rows),
            "packages": len(package_rows),
            "input_tokens": prepared["input_tokens"],
            "loss_tokens": prepared["loss_tokens"],
            "payload_tokens": prepared["payload_tokens"],
        },
        "consumed": {
            "draws": len(prefix),
            "distinct_rows": len(consumed_rows),
            "repeated_row_draws": sum(count - 1 for count in counts.values()),
            "documents_touched": len(consumed_documents),
            "documents_complete": len(complete_documents),
            "documents_partial": len(partial_documents),
            "packages_touched": len(consumed_packages),
            "packages_complete": len(complete_packages),
            "packages_partial": len(partial_packages),
            "groups_touched": len(consumed_groups),
            "input_tokens": totals["input_tokens"],
            "loss_tokens": totals["loss_tokens"],
            "payload_tokens": totals["payload_tokens"],
            "masked_overlap_context_tokens": totals["overlap_context_tokens"],
        },
        "remaining": {
            "rows": len(remaining_rows),
            "documents_never_touched": len(set(document_rows) - consumed_documents),
            "documents_with_remaining_chunks": sum(
                not row_ids <= consumed_rows for row_ids in document_rows.values()
            ),
            "packages_never_touched": len(set(package_rows) - consumed_packages),
            "input_tokens": sum(rows[row_id]["input_tokens"] for row_id in remaining_rows),
            "loss_tokens": sum(rows[row_id]["loss_tokens"] for row_id in remaining_rows),
            "payload_tokens": sum(rows[row_id]["payload_tokens"] for row_id in remaining_rows),
            "distinct_rows_named_after_cursor": len(set(future) - consumed_rows),
            "replay_draws_after_cursor": len(future) - len(set(future) - consumed_rows),
            "prepared_rows_never_named": len(set(rows) - set(schedule["row_ids"])),
        },
        "coverage_percent": {
            "rows": pct(len(consumed_rows), len(rows)),
            "input_tokens": pct(totals["input_tokens"], prepared["input_tokens"]),
            "loss_tokens": pct(totals["loss_tokens"], prepared["loss_tokens"]),
        },
        "sets": {
            "consumed_rows": consumed_rows,
            "consumed_documents": consumed_documents,
            "consumed_packages": consumed_packages,
            "all_rows": set(rows),
            "all_documents": set(document_rows),
            "all_packages": set(package_rows),
        },
    }


def serializable(stage: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stage.items() if key != "sets"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb"),
    )
    parser.add_argument(
        "--native",
        type=Path,
        default=Path("/home/m0hawk/.local/state/sepalith/campaign-20260915"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stages = [audit_stage(args.plan, args.native, spec) for spec in RELATIVE_SPECS]
    broad, global_stage = stages
    broad_sets = broad["sets"]
    global_sets = global_stage["sets"]

    lineage_input = broad["consumed"]["input_tokens"] + global_stage["consumed"]["input_tokens"]
    lineage_loss = broad["consumed"]["loss_tokens"] + global_stage["consumed"]["loss_tokens"]
    lineage_payload = broad["consumed"]["payload_tokens"] + global_stage["consumed"]["payload_tokens"]
    prepared_input = broad["prepared"]["input_tokens"] + global_stage["prepared"]["input_tokens"]
    prepared_loss = broad["prepared"]["loss_tokens"] + global_stage["prepared"]["loss_tokens"]
    prepared_payload = broad["prepared"]["payload_tokens"] + global_stage["prepared"]["payload_tokens"]
    lineage_rows = broad_sets["consumed_rows"] | global_sets["consumed_rows"]
    all_rows = broad_sets["all_rows"] | global_sets["all_rows"]

    result = {
        "schema": "sepalith.cpt.actual_exposure_recomputation.v1",
        "stages": [serializable(stage) for stage in stages],
        "cross_shard_overlap": {
            "prepared_rows": len(broad_sets["all_rows"] & global_sets["all_rows"]),
            "prepared_documents": len(broad_sets["all_documents"] & global_sets["all_documents"]),
            "prepared_packages": len(broad_sets["all_packages"] & global_sets["all_packages"]),
            "consumed_rows": len(broad_sets["consumed_rows"] & global_sets["consumed_rows"]),
            "consumed_documents": len(
                broad_sets["consumed_documents"] & global_sets["consumed_documents"]
            ),
            "consumed_packages": len(
                broad_sets["consumed_packages"] & global_sets["consumed_packages"]
            ),
        },
        "selected_lineage": {
            "distinct_rows": len(lineage_rows),
            "distinct_documents_touched": len(
                broad_sets["consumed_documents"] | global_sets["consumed_documents"]
            ),
            "distinct_packages_touched": len(
                broad_sets["consumed_packages"] | global_sets["consumed_packages"]
            ),
            "input_tokens": lineage_input,
            "loss_tokens": lineage_loss,
            "payload_tokens": lineage_payload,
        },
        "combined_materialized_pool": {
            "rows": len(all_rows),
            "documents": len(broad_sets["all_documents"] | global_sets["all_documents"]),
            "packages": len(broad_sets["all_packages"] | global_sets["all_packages"]),
            "input_tokens": prepared_input,
            "loss_tokens": prepared_loss,
            "payload_tokens": prepared_payload,
            "remaining_rows": len(all_rows - lineage_rows),
            "remaining_input_tokens": prepared_input - lineage_input,
            "remaining_loss_tokens": prepared_loss - lineage_loss,
            "remaining_payload_tokens": prepared_payload - lineage_payload,
            "lineage_row_coverage_percent": pct(len(lineage_rows), len(all_rows)),
            "lineage_input_coverage_percent": pct(lineage_input, prepared_input),
            "lineage_loss_coverage_percent": pct(lineage_loss, prepared_loss),
        },
    }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
