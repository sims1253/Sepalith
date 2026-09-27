#!/usr/bin/env python3
"""Select a deterministic, TRAIN-only paired serving panel.

Rows are copied byte-for-byte from an accepted TRAIN JSONL source after its
expected SHA-256 has been checked.  The selector does not tokenize, pad,
append EOS, synthesize 4K/8K prompts, or read any model.  It only makes a
reviewable panel for a separately admitted serving client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_SHA256 = "e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe"


def _digest_key(row_id: str) -> str:
    return hashlib.sha256(row_id.encode("utf-8")).hexdigest()


def _keep_smallest(selected: list[tuple[str, int, str, dict[str, Any]]], item: tuple[str, int, str, dict[str, Any]], limit: int) -> None:
    """Keep the lexicographically smallest stable keys without retaining all rows."""
    if limit <= 0:
        return
    selected.append(item)
    selected.sort(key=lambda value: (value[0], value[1]))
    if len(selected) > limit:
        selected.pop()


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("id", row.get("row_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("row has no nonempty string id")
    return value


def _prompt_tokens(row: dict[str, Any]) -> int:
    value = row.get("prompt_token_count")
    if isinstance(value, int):
        return value
    value = row.get("target_start")
    if isinstance(value, int):
        # target_start includes the native BOS in this source.  The row's
        # prompt_token_count is the value sent by the serving probe, which
        # excludes that manual BOS.
        return value - 1
    raise ValueError("row has no integer prompt_token_count/target_start")


def _target_tokens(row: dict[str, Any]) -> int:
    value = row.get("target_token_count")
    if isinstance(value, int):
        return value
    input_ids = row.get("input_ids")
    start = row.get("target_start")
    if isinstance(input_ids, list) and isinstance(start, int):
        return len(input_ids) - start
    raise ValueError("row has no integer target token count")


def _family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        family = row.get("family")
        result[str(family)] = result.get(str(family), 0) + 1
    return dict(sorted(result.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rows-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--short-count", type=int, default=20)
    parser.add_argument("--long-count", type=int, default=20)
    parser.add_argument("--short-min", type=int, default=1536)
    parser.add_argument("--short-max", type=int, default=2048)
    parser.add_argument("--long-min", type=int, default=2049)
    parser.add_argument("--long-max", type=int, default=4096)
    parser.add_argument("--target-max", type=int, default=191)
    args = parser.parse_args()
    if args.short_count < 0 or args.long_count < 0:
        raise SystemExit("panel counts must be nonnegative")
    if args.short_min > args.short_max or args.long_min > args.long_max:
        raise SystemExit("invalid prompt bands")

    source = args.rows.resolve()
    hasher = hashlib.sha256()
    ordinal = 0
    total_rows = 0
    split_counts: dict[str, int] = {}
    eligible_counts = {"2k_band": 0, "4k_cap_long_band": 0}
    rejected_target_cap = 0
    selected: dict[str, list[tuple[str, int, str, dict[str, Any]]]] = {
        "2k_band": [],
        "4k_cap_long_band": [],
    }
    with source.open("rb") as handle:
        for raw in handle:
            ordinal += 1
            hasher.update(raw)
            if not raw.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at source line {ordinal}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"source line {ordinal} is not an object")
            split = row.get("split")
            split_counts[str(split)] = split_counts.get(str(split), 0) + 1
            if split != "train":
                continue
            try:
                row_id = _row_id(row)
                prompt_tokens = _prompt_tokens(row)
                target_tokens = _target_tokens(row)
            except ValueError as exc:
                raise SystemExit(f"source line {ordinal}: {exc}") from exc
            if target_tokens > args.target_max:
                rejected_target_cap += 1
                continue
            if args.short_min <= prompt_tokens <= args.short_max:
                band = "2k_band"
                eligible_counts[band] += 1
                _keep_smallest(selected[band], (_digest_key(row_id), ordinal, raw.decode("utf-8"), row), args.short_count)
            elif args.long_min <= prompt_tokens <= args.long_max:
                band = "4k_cap_long_band"
                eligible_counts[band] += 1
                _keep_smallest(selected[band], (_digest_key(row_id), ordinal, raw.decode("utf-8"), row), args.long_count)

    actual_sha256 = hasher.hexdigest()
    if actual_sha256 != args.rows_sha256:
        raise SystemExit(f"source SHA-256 mismatch: expected {args.rows_sha256}, got {actual_sha256}")

    selected_rows: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    selected_raw_lines: list[str] = []
    for band in ("2k_band", "4k_cap_long_band"):
        values = sorted(selected[band], key=lambda value: (value[0], value[1]))
        for panel_ordinal, (digest, source_line, raw_text, row) in enumerate(values):
            prompt_tokens = _prompt_tokens(row)
            target_tokens = _target_tokens(row)
            selected_rows.append(row)
            selected_raw_lines.append(raw_text)
            selected_records.append(
                {
                    "panel_ordinal": len(selected_records),
                    "band": band,
                    "row_id": _row_id(row),
                    "family": row.get("family"),
                    "package_id": row.get("package_id"),
                    "source_line": source_line,
                    "row_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    "stable_selection_key": digest,
                    "prompt_token_count": prompt_tokens,
                    "target_token_count": target_tokens,
                    "target_operation": row.get("target_operation"),
                    "split": row.get("split"),
                }
            )

    requested = {"2k_band": args.short_count, "4k_cap_long_band": args.long_count}
    actual = {band: len(selected[band]) for band in requested}
    sufficient = all(actual[band] >= requested[band] for band in requested)
    if selected_rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as output:
            # Use the exact source line captured above, including all source
            # fields.  A trailing newline is normalized only at file boundary.
            for raw_line in selected_raw_lines:
                output.write(raw_line)
                if not raw_line.endswith("\n"):
                    output.write("\n")

    prompt_values = [_prompt_tokens(row) for row in selected_rows]
    manifest = {
        "schema_version": "sepalith.r2.serving.train-panel.v1",
        "status": "pass" if sufficient else "insufficient_candidates",
        "source": {
            "path": str(source),
            "sha256": actual_sha256,
            "expected_sha256": args.rows_sha256,
            "split_required": "train",
            "rows_seen": total_rows,
            "split_counts": split_counts,
        },
        "selection": {
            "method": "stable SHA-256(row id) order, bounded heap per band",
            "requested_rows": requested,
            "selected_rows": actual,
            "eligible_rows_after_target_cap": eligible_counts,
            "rejected_target_cap_rows": rejected_target_cap,
            "prompt_token_definition": "source prompt_token_count; manual native BOS is out-of-band for serving prompt IDs",
            "target_token_cap": args.target_max,
            "bands": {
                "2k_band": {"min_prompt_tokens": args.short_min, "max_prompt_tokens": args.short_max},
                "4k_cap_long_band": {"min_prompt_tokens": args.long_min, "max_prompt_tokens": args.long_max},
            },
        },
        "panel": {
            "path": str(args.out),
            "row_count": len(selected_rows),
            "prompt_min": min(prompt_values) if prompt_values else None,
            "prompt_max": max(prompt_values) if prompt_values else None,
            "family_counts": _family_counts(selected_rows),
            "records": selected_records,
            "contains_final_or_dev": any(row.get("split") != "train" for row in selected_rows),
            "contains_authored_padding_or_eos": False,
        },
        "serving_contract": {
            "context_size": 4096,
            "completion_cap": 192,
            "native_bos_id": 0,
            "native_eos_ids": [1, 130073],
            "pad_id": 1,
            "prompt_ids": "for tokenize use source input_ids[1:target_start] (prompt_token_count IDs); for completion send [0, *prompt_ids] exactly once",
            "target_tail": "retain source target fields for audit only; do not append authored tokens to a generated response",
            "existing_probe_prompt_limit": 2048,
            "2k_band_existing_probe_compatible": True,
            "4k_cap_long_band_existing_probe_compatible": False,
            "4k_runner_requirement": "a separately reviewed client must admit 2049..4096 prompts under the same 4096 context contract",
        },
        "eight_k_stress": {
            "status": "unsupported_daily_context4096",
            "prompt_tokens": 8192,
            "action": "reject_before_launch_or_run_in_separate_explicit_profile",
            "reason": "daily serving context is pinned to 4096; this packet contains no 8K rows and performs no truncation or padding",
        },
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "source_rows": total_rows, "selected": actual, "source_sha256": actual_sha256}))
    return 0 if sufficient else 2


if __name__ == "__main__":
    raise SystemExit(main())
