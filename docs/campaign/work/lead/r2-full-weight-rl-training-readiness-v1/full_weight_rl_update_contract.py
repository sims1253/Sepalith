#!/usr/bin/env python3
"""Framework-free contract at the rollout-to-full-weight-update boundary.

This does not implement an RL optimizer.  It converts one already committed,
identity-bound rollout buffer into the exact prompt/completion masks and raw
rewards that a live GRPO trainer may consume.  Log probabilities remain a
live-policy computation inside the pinned TRL trainer.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

BOS_ID = 0
PAD_ID = 1
NATIVE_EOG_IDS = {1, 130073}


class UpdateContractError(ValueError):
    pass


def require(value: bool, reason: str) -> None:
    if not value:
        raise UpdateContractError(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("ascii")).hexdigest()


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{label} is not a SHA-256")
    return value


def validate_update_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    require(binding.get("schema") == "sepalith.rl11.full-weight-update-binding.v1",
            "update binding schema mismatch")
    require(binding.get("optimizer_updates") is True, "binding is diagnostic-only")
    require(binding.get("objective") == "trl-0.24-grpo-bnpo-group-beta0",
            "unreviewed RL objective")
    require(binding.get("candidate_count") in {4, 8}, "candidate count is not admitted")
    require(binding.get("policy_step") == binding.get("optimizer_global_step"),
            "rollout policy and optimizer step differ")
    require(binding.get("rollout_cursor") == binding.get("optimizer_global_step"),
            "rollout cursor and optimizer step differ")
    require(binding.get("rollout_complete") is True, "partial rollout buffer cannot update")
    require(binding.get("same_process_live_policy_logprobs") is True,
            "offline or stale policy log probabilities are not admitted")
    require(binding.get("full_weight_checkpoint_kind") == "full_weights",
            "checkpoint kind is not dense full weights")
    require(binding.get("sampler_boundary") == "one_generation_buffer_per_optimizer_update",
            "update is not at a sampler/checkpoint boundary")
    for name in (
        "source_manifest_sha256", "model_manifest_sha256", "model_weights_sha256",
        "tokenizer_json_sha256", "reward_buffer_manifest_sha256",
        "prompt_context_manifest_sha256", "generation_policy_sha256",
        "optimizer_dispatch_sha256",
    ):
        _digest(binding.get(name), name)
    return dict(binding)


def _completion_mask(ids: Sequence[int], cap_hit: bool) -> list[int]:
    require(isinstance(ids, list) and ids, "generated IDs are empty")
    require(all(type(token) is int and 0 <= token < 130560 for token in ids),
            "generated ID is invalid")
    eog = [index for index, token in enumerate(ids) if token in NATIVE_EOG_IDS]
    if cap_hit:
        require(not eog, "cap-hit completion contains terminal EOG")
    else:
        require(len(eog) == 1 and eog[0] == len(ids) - 1,
                "completed generation must have exactly one terminal EOG")
    # The pilot normalizer removes post-terminal padding.  Every retained
    # completion token, including its terminal EOG, participates in GRPO loss.
    return [1] * len(ids)


def prepare_update(
    *, binding: Mapping[str, Any], groups: Sequence[Mapping[str, Any]],
    prompt_ids_by_row: Mapping[str, Sequence[int]], expected_first_group: int,
) -> dict[str, Any]:
    """Validate complete adjacent groups and expose loss-ready record metadata."""
    bound = validate_update_binding(binding)
    g = int(bound["candidate_count"])
    require(groups, "update has no rollout groups")
    expected_indexes = list(range(expected_first_group, expected_first_group + len(groups)))
    require([item.get("group_index") for item in groups] == expected_indexes,
            "rollout group cursor is not contiguous")
    rows: list[dict[str, Any]] = []
    varying_groups = 0
    for group in groups:
        row_id = group.get("row_id")
        require(isinstance(row_id, str) and row_id in prompt_ids_by_row,
                "rollout row has no bound prompt")
        prompt = list(prompt_ids_by_row[row_id])
        require(prompt and prompt[0] == BOS_ID and PAD_ID not in prompt[1:],
                "prompt violates one-BOS/no-EOS geometry")
        prompt_hash = ids_sha256(prompt)
        generations = group.get("generations")
        rewards = group.get("rewards")
        require(isinstance(generations, list) and isinstance(rewards, list)
                and len(generations) == len(rewards) == g,
                "rollout group is incomplete")
        require([r.get("candidate_index") for r in generations] == list(range(g)),
                "generation candidate order mismatch")
        require([r.get("candidate_index") for r in rewards] == list(range(g)),
                "reward candidate order mismatch")
        values: list[float] = []
        for generation, reward in zip(generations, rewards):
            require(generation.get("row_id") == row_id == reward.get("row_id"),
                    "row identity differs across rollout and reward")
            require(generation.get("group_index") == group.get("group_index")
                    == reward.get("group_index"), "group identity differs")
            require(generation.get("prompt_ids_sha256") == prompt_hash,
                    "generation prompt differs from the bound prompt")
            ids = generation.get("generated_ids")
            require(generation.get("generated_ids_sha256") == ids_sha256(ids),
                    "generated ID hash mismatch")
            require(reward.get("output_ids_sha256") == generation.get("generated_ids_sha256"),
                    "reward IDs differ from generated IDs")
            require(reward.get("parser_infrastructure_failure") is False,
                    "infrastructure failure cannot become a reward")
            value = reward.get("reward")
            require(not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(float(value)), "reward is not finite")
            mask = _completion_mask(ids, generation.get("cap_hit") is True)
            rows.append({
                "group_index": group["group_index"], "row_id": row_id,
                "candidate_index": generation["candidate_index"],
                "prompt_ids": prompt, "completion_ids": list(ids),
                "completion_loss_mask": mask, "reward": float(value),
                "cap_hit": bool(generation.get("cap_hit")),
            })
            values.append(float(value))
        varying_groups += int(any(value != values[0] for value in values[1:]))
    require(varying_groups > 0, "rollout update has no within-group reward signal")
    payload = {
        "schema": "sepalith.rl11.full-weight-update-batch.v1",
        "binding": bound,
        "first_group": expected_first_group,
        "next_group": expected_first_group + len(groups),
        "group_count": len(groups), "candidate_count": g,
        "completion_rows": len(rows), "varying_reward_groups": varying_groups,
        "rows": rows,
    }
    payload["batch_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload
