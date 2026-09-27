#!/usr/bin/env python3
"""Independent CPU check for the frozen v5 source-draw candidate.

Only row ``id``, ``family``, and ``split`` metadata are retained while the
approved JSONL is streamed.  The production loader and sampler are imported
from the execution tree; no framework, model, prompt, or sidecar is loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


FAMILIES = (
    "finish_block",
    "format_propagation",
    "na_rm_propagation",
    "no_op",
    "pipe_rewrite",
    "rename_propagation",
    "roxygen_drafting",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_metadata(rows_path: Path) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    with rows_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            raw = json.loads(line)
            row_id = raw.get("id")
            family = raw.get("family")
            split = raw.get("split")
            if not isinstance(row_id, str) or not isinstance(family, str):
                raise AssertionError(f"invalid metadata at rows line {line_number}")
            if row_id in metadata:
                raise AssertionError(f"duplicate row ID at rows line {line_number}")
            metadata[row_id] = (family, split)
    return metadata


def family_counts(row_ids: list[str], metadata: dict[str, tuple[str, str]]) -> dict[str, int]:
    observed = Counter(metadata[row_id][0] for row_id in row_ids)
    return {family: observed[family] for family in FAMILIES}


def max_prefix_discrepancy(row_ids: list[str], metadata: dict[str, tuple[str, str]], targets: dict[str, int]) -> tuple[float, int]:
    total = len(row_ids)
    seen: Counter[str] = Counter()
    maximum = 0.0
    at = 0
    for position, row_id in enumerate(row_ids, 1):
        seen[metadata[row_id][0]] += 1
        discrepancy = max(
            abs(seen[family] - position * targets[family] / total)
            for family in FAMILIES
        )
        if discrepancy > maximum:
            maximum, at = discrepancy, position
    return maximum, at


def coverage(row_ids: list[str], metadata: dict[str, tuple[str, str]], width: int) -> dict[str, object]:
    blocks = [row_ids[index:index + width] for index in range(0, len(row_ids), width)]
    sizes = Counter(len({metadata[row_id][0] for row_id in block}) for block in blocks)
    return {
        "width": width,
        "blocks": len(blocks),
        "family_coverage_histogram": {str(key): sizes[key] for key in sorted(sizes)},
        "all_seven_family_blocks": sum(
            len({metadata[row_id][0] for row_id in block}) == len(FAMILIES)
            for block in blocks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-training", type=Path, required=True)
    parser.add_argument("--schedule-v5", type=Path, required=True)
    parser.add_argument("--schedule-v5-sha256", required=True)
    parser.add_argument("--schedule-v4", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.schedule_v5) != args.schedule_v5_sha256:
        raise AssertionError("v5 artifact hash mismatch")
    v5 = json.loads(args.schedule_v5.read_text(encoding="utf-8"))
    v4 = json.loads(args.schedule_v4.read_text(encoding="utf-8"))
    if v5.get("source_draws") != 24_000 or len(v5.get("row_ids", ())) != 24_000:
        raise AssertionError("v5 source-draw count is not 24,000")
    if v5.get("source_draws_per_update") != 8 or v5.get("candidate_count") != 4:
        raise AssertionError("v5 geometry is not the admitted G4/8-group arm")
    if v5.get("buffer_reuse") != 4 or v5.get("gradient_accumulation_steps") != 4:
        raise AssertionError("v5 buffer geometry is not the admitted four-step arm")

    sys.path.insert(0, str(args.execution_training))
    import campaign_rl_data as data  # noqa: PLC0415
    import campaign_rl_train as train  # noqa: PLC0415

    selected = data.load_selected_ids(args.selected_ids, v5["selected_ids_sha256"])
    loaded = train.load_source_draw_schedule(
        args.schedule_v5,
        args.schedule_v5_sha256,
        selected_ids=selected,
        selected_ids_sha256=v5["selected_ids_sha256"],
        ordered_ids_sha256=v5["ordered_ids_sha256"],
        row_identity_sha256=v5["row_identity_sha256"],
        candidate_count=4,
        source_draws_per_update=8,
        buffer_reuse=4,
    )
    metadata = family_metadata(args.rows)
    v5_ids = list(loaded["row_ids"])
    v4_ids = list(v4["row_ids"])
    if not all(row_id in metadata and metadata[row_id][1] == "train" for row_id in v5_ids):
        raise AssertionError("v5 contains an absent or non-train row")
    if not all(row_id in set(selected) for row_id in v5_ids):
        raise AssertionError("v5 contains a row outside the unique selected-ID set")
    if Counter(v5_ids) != Counter(v4_ids):
        raise AssertionError("v5 changed the row-draw multiset")
    within_family_order = {
        family: [row_id for row_id in v5_ids if metadata[row_id][0] == family]
        == [row_id for row_id in v4_ids if metadata[row_id][0] == family]
        for family in FAMILIES
    }
    if not all(within_family_order.values()):
        raise AssertionError("v5 changed within-family row order")
    if train.source_draw_sequence_sha256(v5_ids) != v5["sequence_sha256"]:
        raise AssertionError("production sequence hash does not match v5")

    targets = family_counts(v5_ids, metadata)
    discrepancy, discrepancy_at = max_prefix_discrepancy(v5_ids, metadata, targets)

    class DataSource:
        def __len__(self) -> int:
            return len(selected)

    selected_index = {row_id: index for index, row_id in enumerate(selected)}
    sampler = train.CampaignRepeatSampler(
        DataSource(), candidate_count=4, prompt_groups_per_batch=8,
        repeat_count=4, generation_batch_size=32,
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        steps_per_generation=4,
        source_draw_sequence=[selected_index[row_id] for row_id in v5_ids],
        source_draw_sequence_sha256=v5["sequence_sha256"],
        source_draw_schedule_sha256=args.schedule_v5_sha256,
    )
    stream = list(sampler)
    expected_first_update: list[int] = []
    for _ in range(4):
        for row_id in v5_ids[:8]:
            expected_first_update.extend([selected_index[row_id]] * 4)
    resume_checks = {}
    for updates in (1, 25, 100):
        cursor = updates * 128
        resume_checks[str(updates)] = (
            sampler.indices_from(cursor, 128) == stream[cursor:cursor + 128]
            and sampler.state(consumed_rows=cursor)["source_draw_cursor"] == updates * 8
        )
    if len(stream) != 384_000 or stream[:128] != expected_first_update or not all(resume_checks.values()):
        raise AssertionError("production sampler geometry or resume reconstruction failed")

    result = {
        "schema_version": "sepalith.rl02.v5-independent-verification.v1",
        "inputs": {
            "schedule_v5": {"path": str(args.schedule_v5), "sha256": args.schedule_v5_sha256},
            "schedule_v4": {"path": str(args.schedule_v4), "sha256": sha256(args.schedule_v4)},
            "selected_ids": {"path": str(args.selected_ids), "sha256": sha256(args.selected_ids)},
            "rows_metadata": {"path": str(args.rows), "sha256": sha256(args.rows), "rows": len(metadata)},
            "execution_training": str(args.execution_training),
        },
        "loader": {
            "import": "campaign_rl_data + campaign_rl_train CPU-safe surfaces",
            "source_draws": loaded["source_draws"],
            "source_draws_per_update": loaded["source_draws_per_update"],
            "buffer_reuse": loaded["buffer_reuse"],
            "hash_stable_before_after": loaded["hash_stable_before_after"],
            "selected_ids_contract": loaded["unique_selected_ids_contract"],
        },
        "v5_identity": {
            "sequence_sha256": v5["sequence_sha256"],
            "selected_ids_sha256": v5["selected_ids_sha256"],
            "ordered_ids_sha256": v5["ordered_ids_sha256"],
            "row_identity_sha256": v5["row_identity_sha256"],
            "family_counts": targets,
            "multiset_equal_v4": Counter(v5_ids) == Counter(v4_ids),
            "within_family_order": within_family_order,
            "unique_selected_ids_contract": v5["unique_selected_ids_contract"],
        },
        "prefixes": {
            str(size): family_counts(v5_ids[:size], metadata)
            for size in (25, 100, 250)
        },
        "max_prefix_discrepancy": {
            "draws": 24_000,
            "absolute_count_error": discrepancy,
            "at_prefix": discrepancy_at,
        },
        "block_coverage": {
            str(width): coverage(v5_ids, metadata, width)
            for width in (8, 25, 100, 250)
        },
        "sampler": {
            "rows": len(stream),
            "rows_per_update": 128,
            "first_update_exact": stream[:128] == expected_first_update,
            "resume_exact": resume_checks,
            "source_cursor_at_updates": {str(updates): updates * 8 for updates in (1, 25, 100)},
        },
        "status": "verified_for_root_review; v5 is a candidate and no launch was performed",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
