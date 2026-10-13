#!/usr/bin/env python3
"""Verify the target-free Semantic9534 32K fallback render.

The fallback input manifest is the only source of the 466-row rerender set.
Every terminal is bound to its manifest shard, input file, output file, and log.
The input and output row-ID sequences are compared in order; a same-count
substitution or permutation cannot pass as a complete render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


INPUT_MANIFEST_SHA256 = (
    "ecd3c405e9ae2f3c567dd1a7441364e37d466a03fb0428260aad8926aa6012e4"
)
INPUT_ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic9535-provider-preparation-v1/render-32k-semantic9534-postrender-v1/inputs"
)
RENDER32_ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic9535-provider-preparation-v1/render-32k-semantic9534-postrender-root-v1"
)
RENDER16_MANIFEST_SHA256 = (
    "58855a14639474f6dae74aba8a081a8752034fe789b464bb494a1567c17db5c4"
)
EXPECTED_SHARDS = tuple(range(27, 41))
EXPECTED_PROVIDER_ROWS = 9534
EXPECTED_GEOMETRY_HOLDS = 1
EXPECTED_SUPPORTED_DENOMINATOR = 9535
EXPECTED_RERUN_ROWS = 466
EXPECTED_CONTEXT_SIZE = 32768
EXPECTED_GENERATION_RESERVE = 2048
RICH_TERMINAL_SCHEMA = "sepalith.dat10.semantic9534.render_shard_rich_terminal.v1"


class ContractError(ValueError):
    """Raised when a review artifact cannot be safely bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot hash file: {path}: {exc}") from exc
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON metadata: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"metadata must be an object: {path}")
    return value


def _iter_rows(path: Path, stage: str) -> Iterable[dict[str, Any]]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot open {stage}: {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"malformed {stage} at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ContractError(f"non-object {stage} at {path}:{line_number}")
            yield value


def ordered_ids(
    path: Path,
    expected_rows: int,
    *,
    stage: str,
    target_free: bool,
    check_status: bool = False,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in _iter_rows(path, stage):
        row_id = value.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ContractError(f"{stage} row has no nonempty row_id: {path}")
        if row_id in seen:
            raise ContractError(f"duplicate {stage} row_id {row_id}: {path}")
        seen.add(row_id)
        marker = value.get("selection_target_or_gold_used")
        if target_free and marker not in (None, False):
            raise ContractError(f"{stage} is not target-free: {row_id}")
        if check_status and value.get("status") not in {"supported", "hold"}:
            raise ContractError(
                f"{stage} has unexpected policy status {value.get('status')}: {row_id}"
            )
        ids.append(row_id)
    if len(ids) != expected_rows:
        raise ContractError(
            f"partial {stage} shard {path.name}: {len(ids)} rows, expected {expected_rows}"
        )
    return ids


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def _require_int(value: Any, expected: int, message: str) -> None:
    if type(value) is not int or value != expected:
        raise ContractError(f"{message}: expected {expected}, got {value}")


def validate_input_manifest(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    if manifest.get("schema") != "sepalith.dat10.semantic9534.fallback32_inputs.v1":
        raise ContractError(f"unexpected fallback input schema: {manifest.get('schema')}")
    if manifest.get("status") != "complete_target_free_review_only":
        raise ContractError(f"fallback inputs are incomplete: {manifest.get('status')}")
    if manifest.get("training_admission") is not False:
        raise ContractError("fallback inputs are training-admitted")
    if manifest.get("target_or_gold_used") is not False:
        raise ContractError("fallback inputs contain target or gold material")
    if manifest.get("source_input_manifest_sha256") != (
        "db1dd374d1d029695c9b8daef59151272d0db0a912276f8073b24b0df4f8e9a0"
    ):
        raise ContractError("fallback source input manifest binding changed")
    if sha256_file(path) != INPUT_MANIFEST_SHA256:
        raise ContractError("fallback input manifest hash changed")
    _require_int(manifest.get("provider_rows"), EXPECTED_PROVIDER_ROWS, "provider denominator")
    _require_int(manifest.get("denominator"), EXPECTED_PROVIDER_ROWS, "fallback denominator")
    _require_int(
        manifest.get("geometry_preparation_holds"),
        EXPECTED_GEOMETRY_HOLDS,
        "geometry hold denominator",
    )
    _require_int(
        manifest.get("supported_denominator"),
        EXPECTED_SUPPORTED_DENOMINATOR,
        "supported denominator",
    )
    _require_int(
        manifest.get("accounted_source_review_rows"),
        EXPECTED_SUPPORTED_DENOMINATOR,
        "accounted source review rows",
    )
    _require_int(manifest.get("rerun32_rows"), EXPECTED_RERUN_ROWS, "32K rerun rows")
    _require_int(manifest.get("held_id_count"), EXPECTED_RERUN_ROWS, "held ID count")
    merge_path = Path(str(manifest.get("render16_terminal_merge_path", "")))
    if not merge_path.is_file():
        raise ContractError(f"missing pinned 16K terminal merge: {merge_path}")
    if manifest.get("render16_terminal_merge_sha256") != RENDER16_MANIFEST_SHA256:
        raise ContractError("16K terminal merge hash pin changed")
    if sha256_file(merge_path) != RENDER16_MANIFEST_SHA256:
        raise ContractError("16K terminal merge hash changed")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(EXPECTED_SHARDS):
        raise ContractError("fallback manifest must contain exactly 14 shards")
    for expected, item in zip(EXPECTED_SHARDS, shards):
        if not isinstance(item, dict):
            raise ContractError("malformed fallback shard binding")
        _require_int(item.get("shard"), expected, "fallback shard identity")
        expected_name = f"shard-{expected:04d}.jsonl"
        if item.get("path") != expected_name:
            raise ContractError(f"fallback path/shard binding mismatch: {expected}")
        rows = item.get("rows")
        provider_rows = item.get("provider_rows")
        if type(rows) is not int or rows < 0 or provider_rows != rows:
            raise ContractError(f"fallback row denominator mismatch: {expected}")
        if type(item.get("bytes")) is not int or item["bytes"] < 0:
            raise ContractError(f"fallback byte pin missing: {expected}")
        if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
            raise ContractError(f"fallback input hash pin missing: {expected}")
    return manifest


def validate_terminal(
    terminal: dict[str, Any],
    *,
    expected_shard: int,
    terminal_path: Path,
    input_path: Path,
    expected_input_sha256: str,
    expected_input_ids: list[str],
    output_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    if terminal.get("schema") != RICH_TERMINAL_SCHEMA:
        raise ContractError(f"unexpected 32K terminal schema: {terminal.get('schema')}")
    _require_int(terminal.get("shard"), expected_shard, "32K terminal shard")
    if terminal.get("status") != "complete":
        raise ContractError(
            f"32K infrastructure failure on shard {expected_shard}: "
            f"status={terminal.get('status')} exit_code={terminal.get('exit_code')}"
        )
    _require_int(terminal.get("exit_code"), 0, "32K terminal exit code")
    _require_int(terminal.get("context_size"), EXPECTED_CONTEXT_SIZE, "32K context size")
    _require_int(
        terminal.get("generation_reserve"),
        EXPECTED_GENERATION_RESERVE,
        "32K generation reserve",
    )
    input_record = terminal.get("input")
    actual_input_sha = sha256_file(input_path)
    if (
        not isinstance(input_record, dict)
        or Path(str(input_record.get("path", ""))).resolve() != input_path.resolve()
        or input_record.get("sha256") != actual_input_sha
        or input_record.get("sha256") != expected_input_sha256
    ):
        raise ContractError(f"32K terminal input binding mismatch on shard {expected_shard}")
    input_ids = ordered_ids(
        input_path,
        expected_rows,
        stage="32K fallback input",
        target_free=True,
    )
    if input_ids != expected_input_ids:
        raise ContractError(f"32K fallback input order changed on shard {expected_shard}")
    output_record = terminal.get("output")
    if not isinstance(output_record, dict):
        raise ContractError(f"missing 32K output record on shard {expected_shard}")
    if Path(str(output_record.get("path", ""))).resolve() != output_path.resolve():
        raise ContractError(f"32K terminal output path mismatch on shard {expected_shard}")
    if not output_path.is_file():
        raise ContractError(f"missing 32K output on shard {expected_shard}")
    actual_output_sha = sha256_file(output_path)
    output_ids = ordered_ids(
        output_path,
        expected_rows,
        stage="32K render",
        target_free=True,
        check_status=True,
    )
    if output_ids != input_ids:
        if set(output_ids) == set(input_ids):
            raise ContractError(f"32K row-ID order mismatch on shard {expected_shard}")
        raise ContractError(f"32K row-ID set mismatch on shard {expected_shard}")
    if output_record.get("sha256") != actual_output_sha:
        raise ContractError(f"32K output hash mismatch on shard {expected_shard}")
    if output_record.get("bytes") != output_path.stat().st_size:
        raise ContractError(f"32K output byte count mismatch on shard {expected_shard}")
    if output_record.get("rows") != len(output_ids):
        raise ContractError(f"32K output row count mismatch on shard {expected_shard}")
    log_record = terminal.get("log")
    if not isinstance(log_record, dict):
        raise ContractError(f"missing 32K terminal log record on shard {expected_shard}")
    log_path = Path(str(log_record.get("path", "")))
    if not log_path.is_file() or log_record.get("sha256") != sha256_file(log_path):
        raise ContractError(f"32K terminal log binding mismatch on shard {expected_shard}")
    return {
        "shard": expected_shard,
        "terminal_path": str(terminal_path),
        "terminal_sha256": sha256_file(terminal_path),
        "status": terminal["status"],
        "exit_code": terminal["exit_code"],
        "context_size": terminal["context_size"],
        "generation_reserve": terminal["generation_reserve"],
        "input_path": str(input_path),
        "input_sha256": actual_input_sha,
        "input_rows": len(input_ids),
        "input_ids_sha256": ids_sha256(input_ids),
        "output_path": str(output_path),
        "output_sha256": actual_output_sha,
        "output_bytes": output_path.stat().st_size,
        "output_rows": len(output_ids),
        "output_ids_sha256": ids_sha256(output_ids),
        "log_path": str(log_path),
        "log_sha256": log_record["sha256"],
    }


def verify(input_manifest_path: Path, render32: Path, output: Path) -> dict[str, Any]:
    if input_manifest_path.resolve() != (INPUT_ROOT / "manifest.json").resolve():
        raise ContractError(f"unexpected fallback input manifest path: {input_manifest_path}")
    if render32.resolve() != RENDER32_ROOT.resolve():
        raise ContractError(f"unexpected 32K output root: {render32}")
    if output.exists():
        raise ContractError(f"fresh 32K terminal merge required: {output}")
    manifest = validate_input_manifest(load_json(input_manifest_path), input_manifest_path)
    input_ids_by_shard: dict[int, list[str]] = {}
    all_ids: list[str] = []
    for bound in manifest["shards"]:
        shard = int(bound["shard"])
        input_path = INPUT_ROOT / str(bound["path"])
        expected_path = INPUT_ROOT / f"shard-{shard:04d}.jsonl"
        if input_path.resolve() != expected_path.resolve():
            raise ContractError(f"fallback input path escaped shard binding: {shard}")
        if input_path.stat().st_size != int(bound["bytes"]):
            raise ContractError(f"fallback input byte count changed: {shard}")
        if sha256_file(input_path) != bound["sha256"]:
            raise ContractError(f"fallback input hash changed: {shard}")
        ids = ordered_ids(
            input_path,
            int(bound["provider_rows"]),
            stage="32K fallback input",
            target_free=True,
        )
        input_ids_by_shard[shard] = ids
        all_ids.extend(ids)
    if len(all_ids) != EXPECTED_RERUN_ROWS or len(set(all_ids)) != EXPECTED_RERUN_ROWS:
        raise ContractError("fallback input IDs do not account for exactly 466 unique rows")
    hold_digest = hashlib.sha256(
        ("\n".join(sorted(all_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    if manifest.get("hold_ids_sha256") != hold_digest:
        raise ContractError("fallback hold ID digest changed")
    terminals: list[dict[str, Any]] = []
    for bound in manifest["shards"]:
        shard = int(bound["shard"])
        expected_rows = int(bound["provider_rows"])
        terminal_path = render32 / f"shard-{shard:04d}.terminal.json"
        output_path = render32 / f"shard-{shard:04d}.jsonl"
        if not terminal_path.is_file():
            raise ContractError(f"missing 32K terminal on shard {shard}")
        terminals.append(
            validate_terminal(
                load_json(terminal_path),
                expected_shard=shard,
                terminal_path=terminal_path,
                input_path=INPUT_ROOT / str(bound["path"]),
                expected_input_sha256=bound["sha256"],
                expected_input_ids=input_ids_by_shard[shard],
                output_path=output_path,
                expected_rows=expected_rows,
            )
        )
    if sum(item["output_rows"] for item in terminals) != EXPECTED_RERUN_ROWS:
        raise ContractError("32K output denominator is not exactly 466")
    result = {
        "schema": "sepalith.dat10.semantic9534.render32_terminal_merge.v1",
        "status": "complete_review_only",
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "render32_root": str(render32),
        "provider_rows": EXPECTED_PROVIDER_ROWS,
        "geometry_preparation_holds": EXPECTED_GEOMETRY_HOLDS,
        "supported_denominator": EXPECTED_SUPPORTED_DENOMINATOR,
        "rerun32_rows": EXPECTED_RERUN_ROWS,
        "terminal_count": len(terminals),
        "terminals": terminals,
        "all_output_rows": sum(item["output_rows"] for item in terminals),
        "target_or_gold_used": False,
        "training_admission": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with (temporary / output.name).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary / output.name, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory)
        os.close(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    try:
        temporary.rmdir()
    except OSError:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--render32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input_manifest, args.render32, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
