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


def _validate_32_terminal(
    terminal_path: Path, terminal: dict[str, Any], input_path: Path, output_path: Path, expected_rows: int
) -> dict[str, Any]:
    schema = terminal.get("schema")
    if schema not in {
        "sepalith.dat10.semantic9535.render_shard_terminal.v1",
        "sepalith.dat10.semantic9534.render_shard_rich_terminal.v1",
    }:
        raise ContractError(f"unexpected 32K terminal schema: {schema}")
    if terminal.get("status") != "complete" or int(terminal.get("exit_code", -1)) != 0:
        raise ContractError(f"32K infrastructure failure: {terminal_path}")
    if int(terminal.get("context_size", -1)) != 32768:
        raise ContractError(f"wrong 32K context size: {terminal_path}")
    if int(terminal.get("generation_reserve", -1)) != EXPECTED_GENERATION_RESERVE:
        raise ContractError(f"wrong 32K generation reserve: {terminal_path}")
    input_record = terminal.get("input")
    if not isinstance(input_record, dict) or input_record.get("sha256") != sha256_file(input_path):
        raise ContractError(f"32K input binding mismatch: {terminal_path}")
    output_record = terminal.get("output")
    if not isinstance(output_record, dict) or not output_path.is_file():
        raise ContractError(f"32K output missing: {terminal_path}")
    output_sha = sha256_file(output_path)
    if output_record.get("sha256") != output_sha or Path(str(output_record.get("path", ""))).resolve() != output_path.resolve():
        raise ContractError(f"32K output binding mismatch: {terminal_path}")
    if int(output_record.get("bytes", -1)) != output_path.stat().st_size:
        raise ContractError(f"32K output byte count mismatch: {terminal_path}")
    rendered = rows_by_id(output_path, stage="32K render", target_free=True)
    if len(rendered) != expected_rows or int(output_record.get("rows", -1)) != expected_rows:
        raise ContractError(f"partial 32K render: {terminal_path}")
    return {
        "terminal_path": str(terminal_path),
        "terminal_sha256": sha256_file(terminal_path),
        "output_sha256": output_sha,
        "output_rows": expected_rows,
    }


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
    fallback_manifest = load_json(fallback_inputs / "manifest.json")
    if fallback_manifest.get("schema") != "sepalith.dat10.semantic9534.fallback32_inputs.v1":
        raise ContractError("unexpected fallback manifest schema")
    if fallback_manifest.get("status") != "complete_target_free_review_only":
        raise ContractError("fallback input manifest is incomplete")
    if fallback_manifest.get("training_admission") is not False or fallback_manifest.get("target_or_gold_used") is not False:
        raise ContractError("fallback inputs are not target-free review-only data")
    if fallback_manifest.get("source_input_manifest_sha256") != INPUT_MANIFEST_SHA256:
        raise ContractError("fallback input binding changed")
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
                stage32_terminals.append(
                    _validate_32_terminal(
                        terminal32_path,
                        load_json(terminal32_path),
                        fallback_path,
                        output32_path,
                        expected_fallback_rows,
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
                args.fallback_inputs,
                args.render32,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
