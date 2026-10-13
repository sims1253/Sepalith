#!/usr/bin/env python3
"""Bounded source-only regression loop for the RL family-prefix defect.

The loop operates on family totals from the recorded audit.  It deliberately
does not read rows, prompts, models, or regenerate a source-draw artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SEED = 3407
FAMILY_ORDER = (
    "finish_block",
    "format_propagation",
    "na_rm_propagation",
    "no_op",
    "pipe_rewrite",
    "rename_propagation",
    "roxygen_drafting",
)
TOTAL_DRAWS = 24_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(family: str) -> str:
    return hashlib.sha256(f"{SEED}|family|{family}".encode("utf-8")).hexdigest()


def quota_balanced_families(targets: dict[str, int]) -> list[str]:
    """Use integer weighted deficit; ties use the stable key ascending."""

    total = sum(targets.values())
    assert total == TOTAL_DRAWS
    used: Counter[str] = Counter()
    sequence: list[str] = []
    for position in range(1, total + 1):
        eligible = [family for family in FAMILY_ORDER if used[family] < targets[family]]
        assert eligible
        family = min(
            eligible,
            key=lambda item: (
                -(position * targets[item] - used[item] * total),
                stable_key(item),
                item,
            ),
        )
        used[family] += 1
        sequence.append(family)
    assert dict(used) == targets
    return sequence


def counts(values: list[str]) -> dict[str, int]:
    observed = Counter(values)
    return {family: observed[family] for family in FAMILY_ORDER}


def max_same_family_run(values: list[str]) -> int:
    longest = 0
    current = None
    length = 0
    for value in values:
        if value == current:
            length += 1
        else:
            current, length = value, 1
        longest = max(longest, length)
    return longest


def prefix_summary(sequence: list[str], targets: dict[str, int], size: int) -> dict[str, object]:
    observed = counts(sequence[:size])
    expected = {family: size * targets[family] / TOTAL_DRAWS for family in FAMILY_ORDER}
    error = {family: observed[family] - expected[family] for family in FAMILY_ORDER}
    return {
        "draws": size,
        "family_counts": observed,
        "max_absolute_quota_error": max(abs(value) for value in error.values()),
        "quota_error": error,
        "families_present": [family for family in FAMILY_ORDER if observed[family]],
        "max_same_family_run": max_same_family_run(sequence[:size]),
    }


def block_summary(sequence: list[str], size: int) -> dict[str, object]:
    blocks = [sequence[index:index + size] for index in range(0, len(sequence), size)]
    family_ranges = {}
    for family in FAMILY_ORDER:
        values = [block.count(family) for block in blocks]
        family_ranges[family] = {"min": min(values), "max": max(values)}
    return {
        "block_size": size,
        "block_count": len(blocks),
        "family_count_ranges": family_ranges,
        "max_same_family_run": max_same_family_run(sequence),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    final = audit["prefixes"]["3000"]
    targets = {family: int(final[family]) for family in FAMILY_ORDER}
    assert sum(targets.values()) == TOTAL_DRAWS
    assert audit["prefixes"]["25"]["finish_block"] == 200
    assert audit["prefixes"]["100"]["finish_block"] == 800
    assert audit["prefixes"]["250"]["finish_block"] == 2000

    sequence = quota_balanced_families(targets)
    prefix_sizes = (25, 100, 250)
    block_sizes = (8, 25, 100, 250)
    result = {
        "schema_version": "sepalith.rl02.prefix-mixture-review.v1",
        "seed": SEED,
        "total_draws": TOTAL_DRAWS,
        "input_audit": {"path": str(args.audit), "sha256": sha256(args.audit)},
        "allocator_source": {"path": str(args.source), "sha256": sha256(args.source)},
        "current_allocator_reproduction": {
            "prefix_25_finish_block": audit["prefixes"]["25"]["finish_block"],
            "prefix_100_finish_block": audit["prefixes"]["100"]["finish_block"],
            "prefix_250_finish_block": audit["prefixes"]["250"]["finish_block"],
            "first_transition_draws": audit["first_transitions"],
            "failure": "family targets are emitted family-by-family, so finish_block occupies the first 3,825 draws",
        },
        "proposed_allocator": {
            "formula": "priority=(position*family_target)-(family_used*total_draws), position is 1-based",
            "tie_break": "stable SHA256(seed|family|name) ascending, then family name ascending",
            "family_targets": targets,
            "sequence_sha256": hashlib.sha256("\n".join(sequence).encode("utf-8")).hexdigest(),
            "final_family_counts": counts(sequence),
            "prefixes": {str(size): prefix_summary(sequence, targets, size) for size in prefix_sizes},
            "blocks": {str(size): block_summary(sequence, size) for size in block_sizes},
            "exact_final_multiset": counts(sequence) == targets,
            "max_same_family_run": max_same_family_run(sequence),
        },
        "row_queue_contract": {
            "status": "source-review-only",
            "queue_order": "reuse existing _rotation_order per family",
            "replay_caps": "apply existing _row_capacity and capacities before enqueueing",
            "selected_ids": "unchanged; no row IDs were read or regenerated by this review",
            "required_apply_test": "assert every emitted row exposure is <= its existing capacity and queue exhaustion fails closed",
        },
        "launch_status": "not a launch artifact; root must generate and hash a new row-ID draw sequence after applying the scheduler",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
