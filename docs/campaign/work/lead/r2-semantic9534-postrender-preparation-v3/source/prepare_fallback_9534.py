#!/usr/bin/env python3
"""Build the exact 32K fallback input set from a verified 16K retry.

Only rows whose verified 16K policy result is unsupported are copied.  The
source rows are still target-free provider inputs.  A failed or partial
terminal is an infrastructure error and stops the command before a fallback
manifest is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from verify_render16 import (
    ContractError,
    EXPECTED_GEOMETRY_HOLDS,
    EXPECTED_PROVIDER_ROWS,
    EXPECTED_SHARDS,
    INPUT_MANIFEST_SHA256,
    load_json,
    sha256_file,
    validate_input_manifest,
)


OLD_RENDER16_ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic9535-provider-preparation-v1/render-16k-v1"
)


def rows_by_id(
    path: Path, *, stage: str, target_free: bool = False
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot open {stage}: {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"malformed {stage} at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("row_id"), str):
                raise ContractError(f"invalid {stage} row at {path}:{line_number}")
            row_id = row["row_id"]
            if row_id in result:
                raise ContractError(f"duplicate {stage} row_id: {row_id}")
            # Rich hold records omit this optional marker; omission remains
            # target-free because target data is never passed to the provider.
            marker = row.get("selection_target_or_gold_used")
            if target_free and marker not in (None, False):
                raise ContractError(f"{stage} is not target-free: {row_id}")
            result[row_id] = row
    return result


def _terminal_index(review: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if review.get("schema") != "sepalith.dat10.semantic9534.render16_terminal_merge.v1":
        raise ContractError("unexpected terminal merge schema")
    if review.get("status") != "complete_review_only":
        raise ContractError(f"terminal merge is not complete: {review.get('status')}")
    if review.get("training_admission") is not False or review.get("target_or_gold_used") is not False:
        raise ContractError("terminal merge is not review-only target-free data")
    if review.get("input_manifest_sha256") != INPUT_MANIFEST_SHA256:
        raise ContractError("terminal merge is bound to a different input manifest")
    terminals = review.get("terminals")
    if not isinstance(terminals, list) or len(terminals) != len(EXPECTED_SHARDS):
        raise ContractError("terminal merge does not contain all 14 terminals")
    index: dict[int, dict[str, Any]] = {}
    for record in terminals:
        if not isinstance(record, dict):
            raise ContractError("malformed terminal merge record")
        shard = record.get("shard")
        if not isinstance(shard, int) or shard in index:
            raise ContractError("duplicate or invalid terminal merge shard")
        if record.get("status") != "complete" or int(record.get("exit_code", -1)) != 0:
            raise ContractError(f"terminal merge contains failed shard {shard}")
        index[shard] = record
    if set(index) != set(EXPECTED_SHARDS):
        raise ContractError("terminal merge shard set is not exactly 0027..0040")
    return index


def prepare(inputs: Path, render16: Path, terminal_review: Path, output: Path) -> dict[str, Any]:
    if render16.resolve() == OLD_RENDER16_ROOT.resolve():
        raise ContractError("preserved failed 16K output root cannot be used")
    if output.exists():
        raise ContractError(f"fresh fallback output required: {output}")
    input_manifest_path = inputs / "manifest.json"
    input_manifest = load_json(input_manifest_path)
    denominators = validate_input_manifest(input_manifest, input_manifest_path)
    review = load_json(terminal_review)
    terminal_index = _terminal_index(review)
    if review.get("render16_root") != str(render16):
        raise ContractError("terminal merge render root does not match fallback input")
    if int(review.get("provider_rows", -1)) != EXPECTED_PROVIDER_ROWS:
        raise ContractError("terminal merge provider denominator changed")
    if int(review.get("geometry_preparation_holds", -1)) != EXPECTED_GEOMETRY_HOLDS:
        raise ContractError("terminal merge geometry hold denominator changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    entries: list[dict[str, Any]] = []
    held_ids: set[str] = set()
    try:
        for bound in input_manifest["shards"]:
            shard = int(bound["shard"])
            input_path = inputs / str(bound["provider"]["path"])
            render_path = render16 / f"shard-{shard:04d}.jsonl"
            record = terminal_index[shard]
            if record.get("output_sha256") != sha256_file(render_path):
                raise ContractError(f"terminal merge output hash mismatch on shard {shard}")
            source = rows_by_id(input_path, stage="provider input")
            rendered = rows_by_id(render_path, stage="16K render", target_free=True)
            expected_rows = int(bound["provider_rows"])
            if len(source) != expected_rows or len(rendered) != expected_rows:
                raise ContractError(f"partial fallback source/render shard {shard}")
            if set(source) != set(rendered):
                raise ContractError(f"source/render row-ID closure mismatch on shard {shard}")
            held = [source[row_id] for row_id in sorted(source) if rendered[row_id].get("status") != "supported"]
            held_ids.update(row["row_id"] for row in held)
            shard_path = temporary / f"shard-{shard:04d}.jsonl"
            with shard_path.open("x", encoding="utf-8") as stream:
                for row in held:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            entries.append(
                {
                    "shard": shard,
                    "rows": len(held),
                    "provider_rows": len(held),
                    "path": shard_path.name,
                    "sha256": sha256_file(shard_path),
                    "bytes": shard_path.stat().st_size,
                    "render16_terminal_sha256": record["terminal_sha256"],
                    "render16_output_sha256": record["output_sha256"],
                }
            )
        if len(held_ids) > EXPECTED_PROVIDER_ROWS:
            raise ContractError("fallback rows exceed provider denominator")
        manifest = {
            "schema": "sepalith.dat10.semantic9534.fallback32_inputs.v1",
            "status": "complete_target_free_review_only",
            "provider_rows": denominators["provider_rows"],
            "denominator": denominators["provider_rows"],
            "geometry_preparation_holds": denominators["geometry_preparation_holds"],
            "supported_denominator": denominators["supported_denominator"],
            "accounted_source_review_rows": denominators["supported_denominator"],
            "rerun32_rows": len(held_ids),
            "held_id_count": len(held_ids),
            "hold_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(held_ids)) + ("\n" if held_ids else "")).encode("utf-8")
            ).hexdigest(),
            "shards": sorted(entries, key=lambda item: item["shard"]),
            "source_input_manifest_sha256": sha256_file(input_manifest_path),
            "render16_terminal_merge_path": str(terminal_review),
            "render16_terminal_merge_sha256": sha256_file(terminal_review),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inputs, args.render16, args.terminal_review, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
