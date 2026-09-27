#!/usr/bin/env python3
"""Explicit PRM03 target-only SFT stage built on the reviewed SFT seams.

This module is a small policy/data adapter.  The reviewed SFT module remains the
owner of the collator, fused Trainer loss, checkpoint callback, tokenizer
contract, and process supervision.  No data or model is loaded on import.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import campaign_sft as reviewed_sft


STAGE = "task_sft_prm03_v1"
LOSS_OBJECTIVE = "prm03_target_only_next_token_v1"
MAX_SEQUENCE_TOKENS = 4096
DEV_COMPLETION_CAP = 192
MILESTONES = (250, 500, 1000)
TARGET_INITIALIZATIONS = {
    "new_lora_on_cpt_merged_parent": "cpt_merged",
    "new_lora_on_midtrain_control": "midtrain_control",
}
TARGET_MODULES = tuple(reviewed_sft.TARGET_MODULES)
_HEX = frozenset("0123456789abcdef")


def _sha256(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _HEX for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int_list(value: Any, name: str, *, minimum: int = 1) -> list[int]:
    if (not isinstance(value, list)
            or any(type(item) is not int or item < minimum for item in value)):
        raise ValueError(f"{name} must be a list of integers >= {minimum}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return list(value)


def task_target_only_policy(recipe: Mapping[str, Any]) -> bool:
    """Validate the new stage identity and return the target-only selection."""
    if recipe.get("schema_version") != 1 or recipe.get("stage") != STAGE:
        raise ValueError("This adapter accepts only task_sft_prm03_v1")
    identity = recipe.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Task SFT requires an explicit training identity")
    policy = identity.get("policy")
    parent = identity.get("parent")
    if not isinstance(policy, Mapping) or not isinstance(parent, Mapping):
        raise ValueError("Task SFT identity requires policy and parent objects")
    data = identity.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Task SFT requires an immutable data identity")
    _nonempty(data.get("split_id"), "identity.data.split_id")
    for record_name, identity_name in (
        ("token_rows", "train_rows_sha256"),
        ("draw_schedule", "draw_schedule_sha256"),
        ("development_panel", "development_panel_sha256"),
    ):
        record = recipe.get(record_name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Task SFT requires a {record_name} record")
        actual = _sha256(record.get("sha256"), record_name + ".sha256")
        if data.get(identity_name) != actual:
            raise ValueError(f"Task SFT {record_name} hash differs from identity.data")
    required_policy = {
        "stage": STAGE,
        "loss_objective": LOSS_OBJECTIVE,
        "full_text_labels": False,
        "optimizer": "adamw_torch_fused",
        "dtype": "bfloat16",
        "target_learning_rate": 2e-4,
        "weight_decay": 0.0,
        "lr_schedule": "cosine",
        "warmup_ratio": 0.03,
        "seed": 3407,
        "gradient_checkpointing": "gpu_standard_v1",
        "logging_steps": 1,
    }
    for name, expected in required_policy.items():
        if policy.get(name) != expected:
            raise ValueError(f"Task SFT policy {name} differs from the explicit stage")
    initialization = policy.get("initialization")
    if initialization not in TARGET_INITIALIZATIONS:
        raise ValueError("Task SFT must start from a fresh CPT merge or Midtrain control")
    if parent.get("kind") != TARGET_INITIALIZATIONS[initialization]:
        raise ValueError("Task SFT parent kind does not match its initialization policy")
    _nonempty(parent.get("revision"), "identity.parent.revision")
    _sha256(parent.get("weights_sha256"), "identity.parent.weights_sha256")
    if parent.get("weights_sha256") in {
        "0" * 64, "f" * 64,
    }:
        raise ValueError("Task SFT parent weight identity is an unusable placeholder")
    params = recipe.get("parameters")
    if not isinstance(params, Mapping):
        raise ValueError("Task SFT requires explicit parameters")
    if identity.get("schedule") != params:
        raise ValueError("Task SFT schedule identity differs from parameters")
    expected = {
        "max_steps": 1000,
        "learning_rate": 2e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
    }
    for name, value in expected.items():
        if params.get(name) != value:
            raise ValueError(f"Task SFT parameter {name} differs from the new stage")
    if (type(params.get("per_device_batch")) is not int
            or params["per_device_batch"] not in (1, 2, 4)
            or type(params.get("gradient_accumulation")) is not int
            or params["per_device_batch"] * params["gradient_accumulation"] != 16):
        raise ValueError("Task SFT requires effective batch 16 with a supported microbatch")
    renderer = identity.get("renderer")
    renderer_id = _nonempty(recipe.get("renderer_id"), "renderer_id")
    if not isinstance(renderer, Mapping) or renderer.get("id") != renderer_id:
        raise ValueError("Task SFT renderer identity is not bound to the recipe")
    if recipe.get("development_max_new_tokens") != DEV_COMPLETION_CAP:
        raise ValueError("Task SFT development output cap must be exactly 192")
    route = recipe.get("development_evaluation_route")
    if route not in ("native_final_dev_v1", "hf_diagnostic_only"):
        raise ValueError("Task SFT must name native final DEV or an explicitly diagnostic HF route")
    if recipe.get("native_acceptance_required") is not True:
        raise ValueError("Task SFT acceptance must require the native final DEV route")
    if recipe.get("hf_evaluation_is_acceptance") is not False:
        raise ValueError("HF DEV evaluation cannot be an acceptance authority")
    _nonempty(recipe.get("evaluator_factory"), "evaluator_factory")
    milestones = _int_list(recipe.get("milestones"), "milestones")
    if tuple(milestones) != MILESTONES:
        raise ValueError("Task SFT milestones must be exactly 250, 500 and 1000")
    gate_starts = _int_list(recipe.get("target_gate_start_steps"), "target_gate_start_steps", minimum=0)
    if 0 not in gate_starts or any(step not in (0, 250, 500) for step in gate_starts):
        raise ValueError("Task SFT target gate must allow fresh step 0 and only 250/500 resume milestones")
    resume_steps = _int_list(recipe.get("target_resume_steps"), "target_resume_steps")
    if tuple(resume_steps) != (250, 500):
        raise ValueError("Task SFT resume checkpoints must be the matched full 250/500 milestones")
    mandatory_stops = _int_list(recipe.get("mandatory_stop_steps"), "mandatory_stop_steps")
    if any(step not in MILESTONES[:-1] for step in mandatory_stops):
        raise ValueError("Mandatory task-stage stops may only use pre-terminal milestones")
    decision_steps = recipe.get("decision_steps")
    if (not isinstance(decision_steps, list)
            or any(type(step) is not int or step not in MILESTONES for step in decision_steps)):
        raise ValueError("Task SFT decision steps must be declared milestones")
    if recipe.get("resume_from") is not None and not mandatory_stops:
        # A caller can still resume after an externally interrupted run.  The
        # full checkpoint validator remains authoritative; this message only
        # prevents a recipe from implying an automatic milestone continuation.
        if recipe.get("resume_milestone") not in MILESTONES[:-1]:
            raise ValueError("A resumed task stage must name its source milestone")
    checkpoint = recipe.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Task SFT requires checkpoint cadence")
    evaluation_steps = checkpoint.get("evaluation_steps")
    if not isinstance(evaluation_steps, list) or not set(MILESTONES) <= set(evaluation_steps):
        raise ValueError("Task SFT must schedule DEV milestones 250/500/1000")
    if any(type(step) is not int or step < 1 for step in evaluation_steps):
        raise ValueError("Task SFT checkpoint evaluation steps must be positive integers")
    return True


def validate_task_rows(recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require complete target tails within the production 192-token cap."""
    cap = recipe.get("development_max_new_tokens")
    if cap != DEV_COMPLETION_CAP:
        raise ValueError("Task target validation requires the 192-token cap")
    target_lengths = []
    for row in rows:
        ident = row.get("id", "<missing>")
        ids = row.get("input_ids")
        start = row.get("target_start")
        body = row.get("target_body_tokens")
        terminal = row.get("target_terminal_tokens")
        body_count = row.get("target_body_token_count")
        terminal_count = row.get("target_terminal_token_count")
        if (not isinstance(ids, list) or type(start) is not int
                or not isinstance(body, list) or not isinstance(terminal, list)
                or not terminal or not 1 < start < len(ids) - 1
                or type(body_count) is not int or body_count != len(body)
                or type(terminal_count) is not int or terminal_count != len(terminal)
                or len(ids) > MAX_SEQUENCE_TOKENS or len(ids) < 4
                or ids[0] != 0 or ids[-1] != 1 or ids.count(0) != 1 or ids.count(1) != 1):
            raise ValueError(f"{ident}: incomplete target metadata")
        tail = ids[start:]
        expected = list(body) + list(terminal) + [1]
        if tail != expected:
            raise ValueError(f"{ident}: target metadata does not cover the complete target and EOS")
        if len(tail) > cap:
            raise ValueError(f"{ident}: complete target including EOS exceeds 192 tokens")
        if any(type(token) is not int or not 0 <= token < 130560 for token in ids):
            raise ValueError(f"{ident}: target metadata contains an invalid token")
        target_lengths.append(len(tail))
    if not target_lengths:
        raise ValueError("Task SFT target data is empty")
    return {
        "rows": len(target_lengths),
        "target_tokens_including_eos": sum(target_lengths),
        "max_target_tokens_including_eos": max(target_lengths),
        "all_complete_targets_within_192": max(target_lengths) <= cap,
    }


def validate_task_admission(recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind the already-verified files and full resume state to the task identity.

    The reviewed preflight verifies input/checkpoint bytes before invoking this
    hook. This task-only hook checks their cross-file meaning without touching
    the legacy recipe path or loading serialized optimizer/model state.
    """
    summary = validate_task_rows(recipe, rows)
    schedule = json.loads(Path(recipe["draw_schedule"]["path"]).read_text())
    if schedule.get("split_id") != recipe["identity"]["data"]["split_id"]:
        raise ValueError("Task SFT draw split differs from identity.data")
    if recipe.get("resume_from"):
        root = Path(recipe["resume_from"])
        state = json.loads((root / "campaign-state.json").read_text())
        trainer = json.loads((root / "trainer_state.json").read_text())
        manifest = json.loads((root / "campaign-manifest.json").read_text())
        step = state.get("step")
        if (type(step) is not int or step not in (250, 500)
                or trainer.get("global_step") != step or manifest.get("step") != step
                or state.get("full") is not True or manifest.get("full") is not True
                or state.get("identity") != recipe["identity"]
                or manifest.get("identity") != recipe["identity"]):
            raise ValueError("Task SFT resume step/full identity is inconsistent")
        sampler = state.get("sampler", {})
        if (sampler.get("split_id") != schedule["split_id"]
                or sampler.get("schedule_sha256") != recipe["draw_schedule"]["sha256"]
                or sampler.get("consumed_draws") != step * 16):
            raise ValueError("Task SFT resume sampler split/schedule/cursor is inconsistent")
        if recipe.get("resume_milestone", step) != step:
            raise ValueError("Task SFT named resume milestone differs from checkpoint")
    return summary


def preflight(recipe: dict[str, Any]):
    task_target_only_policy(recipe)
    return reviewed_sft.preflight(
        recipe, policy_fn=task_target_only_policy, row_validator=validate_task_admission,
    )


def run(recipe: dict[str, Any]) -> None:
    task_target_only_policy(recipe)
    reviewed_sft.run(
        recipe, policy_fn=task_target_only_policy, row_validator=validate_task_admission,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.preflight_only:
        rows, draws, schedule, exposure = preflight(recipe)
        print(json.dumps({
            "status": "preflight_pass", "stage": STAGE, "rows": len(rows),
            "draws": len(draws), "updates": len(exposure),
            "split_id": schedule["split_id"], "target_cap": DEV_COMPLETION_CAP,
            "evaluation_route": recipe["development_evaluation_route"],
            "CUDA_started": False,
        }))
    else:
        run(recipe)


if __name__ == "__main__":
    main()
