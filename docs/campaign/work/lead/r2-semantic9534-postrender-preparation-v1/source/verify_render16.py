#!/usr/bin/env python3
"""Verify the fresh Semantic9534 16K retry before any fallback is launched.

This command is deliberately a terminal verifier, rather than a renderer.  It
binds every shard to the immutable provider input manifest and records the
terminal, output, row-count, and row-ID hashes.  A missing, partial, or failed
shard is an error; it cannot become a policy hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


INPUT_MANIFEST_SHA256 = (
    "db1dd374d1d029695c9b8daef59151272d0db0a912276f8073b24b0df4f8e9a0"
)
INPUT_ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic9535-provider-preparation-v1/render-inputs-v1"
)
EXPECTED_SHARDS = tuple(range(27, 41))
EXPECTED_PROVIDER_ROWS = 9534
EXPECTED_GEOMETRY_HOLDS = 1
EXPECTED_CONTEXT_SIZE = 16384
EXPECTED_GENERATION_RESERVE = 2048
RICH_TERMINAL_SCHEMA = "sepalith.dat10.semantic9534.render_shard_rich_terminal.v1"


class ContractError(ValueError):
    """Raised when a review artifact cannot be safely bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON metadata: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"metadata must be an object: {path}")
    return value


def accounting(provider_rows: int, geometry_preparation_holds: int) -> dict[str, int]:
    """Return the two explicit denominators used by downstream review."""

    if provider_rows != EXPECTED_PROVIDER_ROWS:
        raise ContractError(f"provider denominator changed: {provider_rows}")
    if geometry_preparation_holds != EXPECTED_GEOMETRY_HOLDS:
        raise ContractError(
            f"geometry hold denominator changed: {geometry_preparation_holds}"
        )
    return {
        "provider_rows": provider_rows,
        "geometry_preparation_holds": geometry_preparation_holds,
        "supported_denominator": provider_rows + geometry_preparation_holds,
    }


def validate_input_manifest(manifest: dict[str, Any], input_manifest_path: Path) -> dict[str, Any]:
    if manifest.get("schema") != "sepalith.dat10.semantic9535.provider_inputs.v1":
        raise ContractError(f"unexpected provider-input schema: {manifest.get('schema')}")
    if manifest.get("status") != "complete_target_free_review_only":
        raise ContractError(f"provider inputs are not complete review-only data: {manifest.get('status')}")
    if manifest.get("training_admission") is not False:
        raise ContractError("provider inputs are training-admitted")
    if manifest.get("target_or_gold_copied_to_provider_inputs") is not False:
        raise ContractError("provider inputs contain target/gold material")
    if manifest.get("exact_supported_denominator_closure") is not True:
        raise ContractError("provider denominator closure is not exact")
    bound_shards = manifest.get("shards")
    if not isinstance(bound_shards, list) or [int(x.get("shard", -1)) for x in bound_shards] != list(EXPECTED_SHARDS):
        raise ContractError("provider shard set/order is not 0027..0040")
    provider_rows = int(manifest.get("provider_rows", -1))
    geometry_holds = int(manifest.get("geometry_preparation_holds", -1))
    result = accounting(provider_rows, geometry_holds)
    if int(manifest.get("supported_denominator", -1)) != result["supported_denominator"]:
        raise ContractError("supported denominator does not equal provider rows plus geometry hold")
    if sha256_file(input_manifest_path) != INPUT_MANIFEST_SHA256:
        raise ContractError("input manifest hash changed")
    return result


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"missing JSONL stage output: {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"JSONL row is not an object at {path}:{line_number}")
            yield value


def row_ids(
    path: Path, expected_rows: int, *, stage: str, target_free: bool = False
) -> tuple[list[str], int]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in _iter_rows(path):
        row_id = value.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ContractError(f"{stage} row has no nonempty row_id: {path}")
        if row_id in seen:
            raise ContractError(f"duplicate {stage} row_id {row_id}: {path}")
        seen.add(row_id)
        ids.append(row_id)
        # Rich hold records omit this optional marker; omission is still
        # target-free because the renderer has no target input at this stage.
        marker = value.get("selection_target_or_gold_used")
        if target_free and marker not in (None, False):
            raise ContractError(f"{stage} is not target-free: {row_id}")
    if len(ids) != expected_rows:
        raise ContractError(
            f"partial {stage} shard {path.name}: {len(ids)} rows, expected {expected_rows}"
        )
    return ids, len(ids)


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def validate_terminal(
    terminal: dict[str, Any],
    *,
    terminal_path: Path,
    input_path: Path,
    output_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate one rich terminal and its stage output, including row IDs."""

    if terminal.get("schema") != RICH_TERMINAL_SCHEMA:
        raise ContractError(f"unexpected terminal schema: {terminal.get('schema')}")
    shard = terminal.get("shard")
    if not isinstance(shard, int) or shard not in EXPECTED_SHARDS:
        raise ContractError(f"unexpected terminal shard: {shard}")
    if terminal.get("status") != "complete" or int(terminal.get("exit_code", -1)) != 0:
        raise ContractError(
            f"infrastructure failure on shard {shard}: "
            f"status={terminal.get('status')} exit_code={terminal.get('exit_code')}"
        )
    if int(terminal.get("context_size", -1)) != EXPECTED_CONTEXT_SIZE:
        raise ContractError(f"wrong context size on shard {shard}")
    if int(terminal.get("generation_reserve", -1)) != EXPECTED_GENERATION_RESERVE:
        raise ContractError(f"wrong generation reserve on shard {shard}")
    input_record = terminal.get("input")
    if not isinstance(input_record, dict) or input_record.get("sha256") != sha256_file(input_path):
        raise ContractError(f"terminal input binding mismatch on shard {shard}")
    output_record = terminal.get("output")
    if not isinstance(output_record, dict):
        raise ContractError(f"missing output record on shard {shard}")
    if Path(str(output_record.get("path", ""))).resolve() != output_path.resolve():
        raise ContractError(f"terminal output path mismatch on shard {shard}")
    if not output_path.is_file():
        raise ContractError(f"missing output on shard {shard}")
    output_sha = sha256_file(output_path)
    if output_record.get("sha256") != output_sha:
        raise ContractError(f"terminal output hash mismatch on shard {shard}")
    output_bytes = output_path.stat().st_size
    if int(output_record.get("bytes", -1)) != output_bytes:
        raise ContractError(f"terminal output byte count mismatch on shard {shard}")
    output_ids, output_rows = row_ids(
        output_path, expected_rows, stage="render16", target_free=True
    )
    if int(output_record.get("rows", -1)) != output_rows:
        raise ContractError(f"terminal output row count mismatch on shard {shard}")
    log_record = terminal.get("log")
    if not isinstance(log_record, dict):
        raise ContractError(f"missing terminal log record on shard {shard}")
    log_path = Path(str(log_record.get("path", "")))
    if not log_path.is_file() or log_record.get("sha256") != sha256_file(log_path):
        raise ContractError(f"terminal log binding mismatch on shard {shard}")
    return {
        "shard": shard,
        "terminal_path": str(terminal_path),
        "terminal_sha256": sha256_file(terminal_path),
        "status": terminal["status"],
        "exit_code": terminal["exit_code"],
        "input_path": str(input_path),
        "input_sha256": input_record["sha256"],
        "input_rows": expected_rows,
        "input_ids_sha256": ids_sha256(
            row_ids(input_path, expected_rows, stage="input")[0]
        ),
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "output_bytes": output_bytes,
        "output_rows": output_rows,
        "output_ids_sha256": ids_sha256(output_ids),
        "log_path": str(log_path),
        "log_sha256": log_record["sha256"],
    }


def verify(input_manifest_path: Path, render16: Path, output_path: Path) -> dict[str, Any]:
    input_manifest = load_json(input_manifest_path)
    accounting_result = validate_input_manifest(input_manifest, input_manifest_path)
    input_root = input_manifest_path.parent
    if input_root.resolve() != INPUT_ROOT.resolve():
        raise ContractError(f"unexpected input root: {input_root}")
    terminals: list[dict[str, Any]] = []
    for bound in input_manifest["shards"]:
        shard = int(bound["shard"])
        provider = bound.get("provider")
        if not isinstance(provider, dict):
            raise ContractError(f"missing provider binding on shard {shard}")
        input_path = input_root / str(provider["path"])
        output_path_for_shard = render16 / f"shard-{shard:04d}.jsonl"
        terminal_path = render16 / f"shard-{shard:04d}.terminal.json"
        if not terminal_path.is_file():
            raise ContractError(f"missing terminal record on shard {shard}")
        terminal = load_json(terminal_path)
        if int(provider.get("rows", provider.get("provider_rows", -1))) != int(bound["provider_rows"]):
            raise ContractError(f"input manifest row binding mismatch on shard {shard}")
        terminals.append(
            validate_terminal(
                terminal,
                terminal_path=terminal_path,
                input_path=input_path,
                output_path=output_path_for_shard,
                expected_rows=int(bound["provider_rows"]),
            )
        )
    if len(terminals) != len(EXPECTED_SHARDS) or {x["shard"] for x in terminals} != set(EXPECTED_SHARDS):
        raise ContractError("terminal set is not exactly 14 shards")
    result = {
        "schema": "sepalith.dat10.semantic9534.render16_terminal_merge.v1",
        "status": "complete_review_only",
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "render16_root": str(render16),
        "terminal_count": len(terminals),
        "provider_rows": accounting_result["provider_rows"],
        "geometry_preparation_holds": accounting_result["geometry_preparation_holds"],
        "supported_denominator": accounting_result["supported_denominator"],
        "terminals": sorted(terminals, key=lambda item: item["shard"]),
        "terminal_hashes_preserved": True,
        "row_counts_and_ids_preserved": True,
        "target_or_gold_used": False,
        "training_admission": False,
    }
    if output_path.exists():
        raise ContractError(f"fresh verifier output required: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--render16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input_manifest, args.render16, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
