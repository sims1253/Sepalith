#!/usr/bin/env python3
"""Select 16K contexts, or verified 32K fallbacks, without target access."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from prepare_fallback_9534 import rows_by_id
from verify_render16 import (
    ContractError,
    EXPECTED_CONTEXT_SIZE,
    EXPECTED_GENERATION_RESERVE,
    EXPECTED_GEOMETRY_HOLDS,
    EXPECTED_PROVIDER_ROWS,
    EXPECTED_SHARDS,
    INPUT_ROOT,
    INPUT_MANIFEST_SHA256,
    load_json,
    sha256_file,
    validate_input_manifest,
)
from verify_render32 import (
    ContractError as Render32ContractError,
    INPUT_ROOT as FALLBACK_INPUT_ROOT,
    INPUT_MANIFEST_SHA256 as FALLBACK_INPUT_MANIFEST_SHA256,
    EXPECTED_CONTEXT_SIZE as RENDER32_CONTEXT_SIZE,
    EXPECTED_SHARDS as RENDER32_SHARDS,
    validate_input_manifest as validate_fallback_input_manifest,
    validate_terminal as validate_render32_terminal,
)


OLD_RENDER16_ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic9535-provider-preparation-v1/render-16k-v1"
)


def _terminal_merge_index(review: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if review.get("schema") != "sepalith.dat10.semantic9534.render16_terminal_merge.v1":
        raise ContractError("unexpected 16K terminal merge schema")
    if review.get("status") != "complete_review_only":
        raise ContractError("16K terminal merge is incomplete")
    if review.get("input_manifest_sha256") != INPUT_MANIFEST_SHA256:
        raise ContractError("16K terminal merge input binding changed")
    if review.get("training_admission") is not False or review.get("target_or_gold_used") is not False:
        raise ContractError("16K terminal merge is not target-free review-only data")
    values = review.get("terminals")
    if not isinstance(values, list) or len(values) != 14:
        raise ContractError("16K terminal merge is missing a shard")
    result: dict[int, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("shard"), int):
            raise ContractError("malformed 16K terminal merge record")
        shard = value["shard"]
        if shard in result or shard not in EXPECTED_SHARDS:
            raise ContractError("duplicate or unexpected 16K terminal merge shard")
        if value.get("status") != "complete" or int(value.get("exit_code", -1)) != 0:
            raise ContractError(f"16K terminal merge contains infrastructure failure: {shard}")
        result[shard] = value
    if set(result) != set(EXPECTED_SHARDS):
        raise ContractError("16K terminal merge shard set is incomplete")
    return result


def _terminal32_merge_index(
    review: dict[str, Any], render32: Path
) -> dict[int, dict[str, Any]]:
    """Validate the verifier's complete 32K merge before policy selection."""

    if review.get("schema") != "sepalith.dat10.semantic9534.render32_terminal_merge.v1":
        raise ContractError("unexpected 32K terminal merge schema")
    if review.get("status") != "complete_review_only":
        raise ContractError("32K terminal merge is incomplete")
    if review.get("input_manifest_sha256") != FALLBACK_INPUT_MANIFEST_SHA256:
        raise ContractError("32K terminal merge input binding changed")
    if review.get("render32_root") != str(render32):
        raise ContractError("32K terminal merge render root changed")
    if review.get("training_admission") is not False or review.get("target_or_gold_used") is not False:
        raise ContractError("32K terminal merge is not target-free review-only data")
    if review.get("provider_rows") != EXPECTED_PROVIDER_ROWS:
        raise ContractError("32K terminal merge provider denominator changed")
    if review.get("geometry_preparation_holds") != EXPECTED_GEOMETRY_HOLDS:
        raise ContractError("32K terminal merge geometry hold denominator changed")
    if review.get("supported_denominator") != EXPECTED_PROVIDER_ROWS + EXPECTED_GEOMETRY_HOLDS:
        raise ContractError("32K terminal merge supported denominator changed")
    if review.get("rerun32_rows") != 466 or review.get("all_output_rows") != 466:
        raise ContractError("32K terminal merge output denominator is not exactly 466")
    values = review.get("terminals")
    if not isinstance(values, list) or len(values) != len(RENDER32_SHARDS):
        raise ContractError("32K terminal merge is missing a shard")
    result: dict[int, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or type(value.get("shard")) is not int:
            raise ContractError("malformed 32K terminal merge record")
        shard = value["shard"]
        if shard in result or shard not in RENDER32_SHARDS:
            raise ContractError("duplicate or unexpected 32K terminal merge shard")
        if value.get("status") != "complete" or value.get("exit_code") != 0:
            raise ContractError(f"32K terminal merge contains infrastructure failure: {shard}")
        if value.get("context_size") != RENDER32_CONTEXT_SIZE or value.get("generation_reserve") != EXPECTED_GENERATION_RESERVE:
            raise ContractError(f"32K terminal merge resource binding changed: {shard}")
        result[shard] = value
    if set(result) != set(RENDER32_SHARDS):
        raise ContractError("32K terminal merge shard set is incomplete")
    return result


def _validate_32_terminal(
    terminal_path: Path,
    terminal: dict[str, Any],
    expected_shard: int,
    input_path: Path,
    output_path: Path,
    expected_rows: int,
    expected_ids: list[str],
) -> dict[str, Any]:
    """Reuse the strict terminal verifier, including ordered row-ID closure."""

    try:
        return validate_render32_terminal(
            terminal,
            expected_shard=expected_shard,
            terminal_path=terminal_path,
            input_path=input_path,
            expected_input_sha256=sha256_file(input_path),
            expected_input_ids=expected_ids,
            output_path=output_path,
            expected_rows=expected_rows,
        )
    except Render32ContractError as exc:
        raise ContractError(f"32K terminal verification failed: {terminal_path}: {exc}") from exc


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "rows": len(values), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finalize(
    inputs: Path,
    render16: Path,
    terminal_review: Path,
    terminal32_review: Path,
    fallback_inputs: Path,
    render32: Path,
    output: Path,
) -> dict[str, Any]:
    if render16.resolve() == OLD_RENDER16_ROOT.resolve():
        raise ContractError("preserved failed 16K output root cannot be selected")
    if output.exists():
        raise ContractError(f"fresh context-policy output required: {output}")
    input_manifest_path = inputs / "manifest.json"
    input_manifest = load_json(input_manifest_path)
    if inputs.resolve() != INPUT_ROOT.resolve():
        raise ContractError("unexpected provider input root")
    validate_input_manifest(input_manifest, input_manifest_path)
    review = load_json(terminal_review)
    terminal_index = _terminal_merge_index(review)
    if review.get("render16_root") != str(render16):
        raise ContractError("16K terminal merge render root changed")
    terminal32 = load_json(terminal32_review)
    terminal32_index = _terminal32_merge_index(terminal32, render32)
    fallback_manifest = load_json(fallback_inputs / "manifest.json")
    if fallback_inputs.resolve() != FALLBACK_INPUT_ROOT.resolve():
        raise ContractError("unexpected fallback input root")
    validate_fallback_input_manifest(fallback_manifest, fallback_inputs / "manifest.json")
    if int(fallback_manifest.get("provider_rows", -1)) != EXPECTED_PROVIDER_ROWS:
        raise ContractError("fallback provider denominator changed")
    if int(fallback_manifest.get("geometry_preparation_holds", -1)) != EXPECTED_GEOMETRY_HOLDS:
        raise ContractError("fallback geometry hold denominator changed")
    fallback_shards = {int(value["shard"]): value for value in fallback_manifest.get("shards", [])}
    if set(fallback_shards) != set(EXPECTED_SHARDS):
        raise ContractError("fallback shard set is incomplete")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    selected: list[dict[str, Any]] = []
    followup: dict[int, list[dict[str, Any]]] = {shard: [] for shard in EXPECTED_SHARDS}
    counts: collections.Counter[str] = collections.Counter()
    stage32_terminals: list[dict[str, Any]] = []
    try:
        for bound in input_manifest["shards"]:
            shard = int(bound["shard"])
            source_path = inputs / str(bound["provider"]["path"])
            render16_path = render16 / f"shard-{shard:04d}.jsonl"
            review_record = terminal_index[shard]
            if review_record.get("output_sha256") != sha256_file(render16_path):
                raise ContractError(f"16K output hash changed after terminal review: {shard}")
            source = rows_by_id(source_path, stage="provider input")
            rendered16 = rows_by_id(render16_path, stage="16K render", target_free=True)
            expected_rows = int(bound["provider_rows"])
            if len(source) != expected_rows or len(rendered16) != expected_rows:
                raise ContractError(f"partial 16K shard {shard}")
            if set(source) != set(rendered16):
                raise ContractError(f"16K source/result row-ID closure mismatch: {shard}")
            fallback_entry = fallback_shards[shard]
            fallback_path = fallback_inputs / str(fallback_entry["path"])
            if sha256_file(fallback_path) != fallback_entry.get("sha256"):
                raise ContractError(f"fallback input hash changed: {shard}")
            fallback = rows_by_id(fallback_path, stage="32K fallback input")
            expected_fallback_rows = int(fallback_entry.get("provider_rows", fallback_entry.get("rows", -1)))
            if len(fallback) != expected_fallback_rows:
                raise ContractError(f"fallback input row count changed: {shard}")
            expected_ids = {row_id for row_id, value in rendered16.items() if value.get("status") != "supported"}
            if set(fallback) != expected_ids:
                raise ContractError(f"fallback set does not equal 16K policy holds: {shard}")

            rendered32: dict[str, dict[str, Any]] = {}
            if expected_fallback_rows:
                terminal32_path = render32 / f"shard-{shard:04d}.terminal.json"
                output32_path = render32 / f"shard-{shard:04d}.jsonl"
                if not terminal32_path.is_file():
                    raise ContractError(f"missing 32K terminal for nonempty fallback shard {shard}")
                terminal32_record = terminal32_index[shard]
                if terminal32_record.get("terminal_path") != str(terminal32_path):
                    raise ContractError(f"32K merge terminal path changed: {shard}")
                stage32_terminals.append(
                    _validate_32_terminal(
                        terminal32_path,
                        load_json(terminal32_path),
                        shard,
                        fallback_path,
                        output32_path,
                        expected_fallback_rows,
                        list(fallback),
                    )
                )
                rendered32 = rows_by_id(output32_path, stage="32K render", target_free=True)
                if set(rendered32) != set(fallback):
                    raise ContractError(f"32K source/result row-ID closure mismatch: {shard}")
            elif (render32 / f"shard-{shard:04d}.terminal.json").exists() or (render32 / f"shard-{shard:04d}.jsonl").exists():
                raise ContractError(f"unexpected 32K artifact for empty fallback shard {shard}")

            for row_id in sorted(source):
                result16 = rendered16[row_id]
                if result16.get("status") == "supported":
                    selected.append({**result16, "policy_resolution": "selected_16k"})
                    counts["selected_16k"] += 1
                    continue
                if result16.get("status") != "hold":
                    raise ContractError(f"unknown 16K result status {result16.get('status')}: {row_id}")
                result32 = rendered32.get(row_id)
                if result32 is None:
                    raise ContractError(f"missing 32K result for 16K hold: {row_id}")
                if result32.get("status") == "supported":
                    selected.append({**result32, "policy_resolution": "fallback_supported_32k"})
                    counts["fallback_supported_32k"] += 1
                elif result32.get("status") == "hold":
                    hold = {**result32, "policy_resolution": "hold_after_32k"}
                    selected.append(hold)
                    counts["hold_after_32k"] += 1
                    if "complete_span_context_budget" in result32.get("reasons", []):
                        followup[shard].append(source[row_id])
                        counts["context_only_followup_64k_128k"] += 1
                else:
                    raise ContractError(f"unknown 32K result status {result32.get('status')}: {row_id}")

        if len(selected) != EXPECTED_PROVIDER_ROWS or len({x["row_id"] for x in selected}) != EXPECTED_PROVIDER_ROWS:
            raise ContractError("selected context denominator is not exactly 9534")
        selected_file = temporary / "selected-contexts.jsonl"
        selected_record = _write_jsonl(selected_file, sorted(selected, key=lambda value: value["row_id"]))
        followup_dir = temporary / "followup-context-inputs"
        followup_dir.mkdir()
        followup_records: list[dict[str, Any]] = []
        for shard, values in sorted(followup.items()):
            path = followup_dir / f"shard-{shard:04d}.jsonl"
            followup_records.append({"shard": shard, **_write_jsonl(path, values)})
        manifest = {
            "schema": "sepalith.dat10.semantic9534.context_policy.v1",
            "status": "complete_review_only",
            "provider_rows": EXPECTED_PROVIDER_ROWS,
            "denominator": EXPECTED_PROVIDER_ROWS,
            "geometry_preparation_holds": EXPECTED_GEOMETRY_HOLDS,
            "supported_denominator": EXPECTED_PROVIDER_ROWS + EXPECTED_GEOMETRY_HOLDS,
            "accounted_source_review_rows": EXPECTED_PROVIDER_ROWS + EXPECTED_GEOMETRY_HOLDS,
            "counts": dict(counts),
            "selected": selected_record,
            "followup_context_inputs": {
                "profiles": [65536, 131072],
                "reason": "32K context budget only; no exclusion ceiling",
                "shards": followup_records,
            },
            "render16_terminal_merge_path": str(terminal_review),
            "render16_terminal_merge_sha256": sha256_file(terminal_review),
            "render32_terminal_count": len(stage32_terminals),
            "render32_terminals": stage32_terminals,
            "target_or_gold_used": False,
            "training_admission": False,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        output.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory)
        os.close(directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--render16", type=Path, required=True)
    parser.add_argument("--terminal-review", type=Path, required=True)
    parser.add_argument("--terminal32-review", type=Path, required=True)
    parser.add_argument("--fallback-inputs", type=Path, required=True)
    parser.add_argument("--render32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                args.inputs,
                args.render16,
                args.terminal_review,
                args.terminal32_review,
                args.fallback_inputs,
                args.render32,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
