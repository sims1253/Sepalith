"""CPU-only contract model for adjacent fixed-ID generation-group packing.

This module deliberately contains no model, tokenizer, CUDA, TRL, or Unsloth
imports.  It mirrors the ordering and identity checks in the source patch so
the packing and resume properties can be tested without launching a trainer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ALLOWED_GENERATION_GROUPS_PER_CALL = (1, 2, 4, 8)
SUPPORTED_CANDIDATE_COUNTS = (2, 4)


class GenerationGroupingError(ValueError):
    """A call-level packing contract failed closed."""


def resolve_generation_groups_per_call(
    value: object,
    name: str = "generation_groups_per_call",
    *,
    max_groups: int | None = None,
) -> int:
    if type(value) is not int or value < 1:
        raise GenerationGroupingError(f"{name} must be an integer >= 1")
    if value not in ALLOWED_GENERATION_GROUPS_PER_CALL:
        raise GenerationGroupingError(
            f"{name} must be one of {ALLOWED_GENERATION_GROUPS_PER_CALL}"
        )
    if max_groups is not None and value > max_groups:
        raise GenerationGroupingError(
            f"{name} cannot exceed the generation buffer's {max_groups} groups"
        )
    return value


def plan_generation_calls(
    prompt_ids: Sequence[Sequence[int]],
    *,
    candidate_count: int,
    groups_per_call: int,
) -> dict[str, Any]:
    """Partition rows into complete logical groups and adjacent model calls."""
    if candidate_count not in SUPPORTED_CANDIDATE_COUNTS:
        raise GenerationGroupingError("candidate_count is outside the admitted G set")
    groups_per_call = resolve_generation_groups_per_call(groups_per_call)
    if not prompt_ids:
        raise GenerationGroupingError("generation batch is empty")
    if len(prompt_ids) % candidate_count:
        raise GenerationGroupingError("generation batch ends inside a logical group")
    logical_groups: list[tuple[tuple[int, ...], ...]] = []
    for start in range(0, len(prompt_ids), candidate_count):
        rows = tuple(tuple(row) for row in prompt_ids[start:start + candidate_count])
        if any(row != rows[0] for row in rows[1:]):
            raise GenerationGroupingError(
                f"logical group {start // candidate_count} contains different stored prompts"
            )
        logical_groups.append(rows)
    calls = tuple(
        tuple(logical_groups[start:start + groups_per_call])
        for start in range(0, len(logical_groups), groups_per_call)
    )
    return {
        "candidate_count": candidate_count,
        "groups_per_call": groups_per_call,
        "logical_groups": tuple(logical_groups),
        "calls": calls,
        "row_count": len(prompt_ids),
        "call_count": len(calls),
    }


def flatten_call(call: Sequence[Sequence[Sequence[int]]]) -> tuple[tuple[int, ...], ...]:
    """Return the exact model-call row order for one planned call."""
    return tuple(tuple(row) for group in call for row in group)


def materialize_batched_outputs(
    plan: Mapping[str, Any],
    generated_rows: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Attach output rows to logical indices while preserving call order."""
    calls = plan.get("calls")
    if not isinstance(calls, tuple) or not calls:
        raise GenerationGroupingError("plan has no calls")
    expected_rows = sum(len(flatten_call(call)) for call in calls)
    if len(generated_rows) != expected_rows:
        raise GenerationGroupingError(
            f"model returned {len(generated_rows)} rows; expected {expected_rows}"
        )
    records: list[dict[str, Any]] = []
    row_cursor = 0
    for call_index, call in enumerate(calls):
        rows = flatten_call(call)
        for local_index, prompt in enumerate(rows):
            group_offset, row_index = divmod(local_index, int(plan["candidate_count"]))
            records.append({
                "call_index": call_index,
                "group_index": sum(len(item) for item in calls[:call_index]) + group_offset,
                "group_row_index": row_index,
                "prompt_ids": prompt,
                "generated_ids": tuple(generated_rows[row_cursor]),
            })
            row_cursor += 1
    return records


def validate_recipe_binding(recipe: Mapping[str, Any]) -> int:
    """Require top-level and identity.policy values to be equal and finite."""
    identity = recipe.get("identity")
    if not isinstance(identity, Mapping):
        raise GenerationGroupingError("identity is required")
    policy = identity.get("policy")
    if not isinstance(policy, Mapping):
        raise GenerationGroupingError("identity.policy is required")
    candidate_count = policy.get("candidate_count")
    rollout_rows = policy.get("rollout_rows_per_update")
    if type(candidate_count) is not int or type(rollout_rows) is not int:
        raise GenerationGroupingError("policy geometry is required")
    if candidate_count <= 0 or rollout_rows <= 0 or rollout_rows % candidate_count:
        raise GenerationGroupingError("policy geometry is invalid")
    max_groups = rollout_rows // candidate_count
    top = resolve_generation_groups_per_call(recipe.get("generation_groups_per_call"), "recipe.generation_groups_per_call", max_groups=max_groups)
    policy_value = resolve_generation_groups_per_call(policy.get("generation_groups_per_call"), "identity.policy.generation_groups_per_call", max_groups=max_groups)
    if top != policy_value:
        raise GenerationGroupingError("recipe and identity.policy generation_groups_per_call differ")
    return top
