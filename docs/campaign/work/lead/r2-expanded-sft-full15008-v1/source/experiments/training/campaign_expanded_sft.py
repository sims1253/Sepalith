#!/usr/bin/env python3
"""Expanded target-only SFT stage built on the reviewed SFT seams.

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


STAGE = "expanded_task_sft_full15006_v1"
LOSS_OBJECTIVE = "expanded_target_only_next_token_v1"
MAX_SEQUENCE_TOKENS = 4096
TRAIN_MAX_TARGET_TOKENS = 1024
DEV_COMPLETION_CAP = 192
MILESTONES = (240, 480, 720, 1160)
TARGET_INITIALIZATIONS = {
    "new_lora_on_sft_merged_parent": "sft_merged",
}
SELECTED_PARENT_WEIGHTS_SHA256 = "51232c1c491a4857c55c61669e383e72770b9d1ca49c021b4c7d69bd11ac3c21"
SELECTED_PARENT_STEP = 750
SELECTED_PARENT_CURSOR = 12000
PARENT_MANIFEST_SCHEMA = "sepalith.r2-task-sft.parent-manifest.v1"
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


def _record(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an immutable file record")
    _nonempty(value.get("path"), name + ".path")
    _sha256(value.get("sha256"), name + ".sha256")
    return value


def expanded_target_only_policy(recipe: Mapping[str, Any]) -> bool:
    """Validate the new stage identity and return the target-only selection."""
    if recipe.get("schema_version") != 1 or recipe.get("stage") != STAGE:
        raise ValueError(f"This adapter accepts only {STAGE}")
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
        raise ValueError("Expanded SFT must start with a new LoRA on the selected merged SFT parent")
    if parent.get("kind") != TARGET_INITIALIZATIONS[initialization]:
        raise ValueError("Task SFT parent kind does not match its initialization policy")
    _nonempty(parent.get("revision"), "identity.parent.revision")
    _sha256(parent.get("weights_sha256"), "identity.parent.weights_sha256")
    if parent.get("weights_sha256") in {
        "0" * 64, "f" * 64,
    }:
        raise ValueError("Task SFT parent weight identity is an unusable placeholder")
    if parent.get("weights_sha256") != SELECTED_PARENT_WEIGHTS_SHA256:
        raise ValueError("Expanded SFT parent is not the selected merged E750 weights")
    _sha256(parent.get("sft_merged_manifest_sha256"),
            "identity.parent.sft_merged_manifest_sha256")
    if (parent.get("previous_checkpoint_step") != SELECTED_PARENT_STEP
            or parent.get("previous_source_cursor") != SELECTED_PARENT_CURSOR):
        raise ValueError("Expanded SFT parent must bind selected step 750 and source cursor 12000")
    params = recipe.get("parameters")
    if not isinstance(params, Mapping):
        raise ValueError("Task SFT requires explicit parameters")
    if identity.get("schedule") != params:
        raise ValueError("Task SFT schedule identity differs from parameters")
    expected = {
        "max_steps": 1160,
        "learning_rate": 2e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "train_max_target_tokens": TRAIN_MAX_TARGET_TOKENS,
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
        raise ValueError("Task SFT milestones must be exactly 240, 480, 720 and 1160")
    gate_starts = _int_list(recipe.get("target_gate_start_steps"), "target_gate_start_steps", minimum=0)
    compatibility = recipe.get("resume_identity_compatibility")
    allowed_gate_starts = (0, 240, 480, 720)
    if compatibility is not None:
        interruption_step = compatibility.get("checkpoint_step")
        full_every = (recipe.get("checkpoint") or {}).get("full_every")
        future_stops = [step for step in recipe.get("mandatory_stop_steps", [])
                        if type(step) is int and type(interruption_step) is int and step > interruption_step]
        if (type(interruption_step) is not int or type(full_every) is not int
                or interruption_step < 1 or interruption_step % full_every
                or not future_stops):
            raise ValueError("Expanded recovery requires a full-cadence interruption below a mandatory stop")
        allowed_gate_starts = tuple(sorted({0, interruption_step, 240, 480, 720}))
    if tuple(gate_starts) != allowed_gate_starts:
        raise ValueError("Task SFT target gate starts differ from the admitted recovery lifecycle")
    resume_steps = _int_list(recipe.get("target_resume_steps"), "target_resume_steps")
    if tuple(resume_steps) != (240, 480, 720):
        raise ValueError("Task SFT resume checkpoints must be the matched full 240/480/720 milestones")
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
        allowed_resume_milestones = (*MILESTONES[:-1],)
        if compatibility is not None:
            allowed_resume_milestones += (compatibility["checkpoint_step"],)
        if recipe.get("resume_milestone") not in allowed_resume_milestones:
            raise ValueError("A resumed task stage must name its source milestone")
    checkpoint = recipe.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Task SFT requires checkpoint cadence")
    evaluation_steps = checkpoint.get("evaluation_steps")
    if not isinstance(evaluation_steps, list) or not set(MILESTONES) <= set(evaluation_steps):
        raise ValueError("Task SFT must schedule DEV milestones 240/480/720/1160")
    if any(type(step) is not int or step < 1 for step in evaluation_steps):
        raise ValueError("Task SFT checkpoint evaluation steps must be positive integers")
    return True


def validate_sft_merged_parent(recipe: Mapping[str, Any]) -> None:
    """Cross-check the selected merged parent against its immutable lineage files."""
    binding = recipe.get("sft_merged_parent")
    if not isinstance(binding, Mapping):
        raise ValueError("Expanded SFT requires an sft_merged_parent binding")
    manifest_record = _record(binding.get("manifest"), "sft_merged_parent.manifest")
    recipe_record = _record(binding.get("previous_recipe"), "sft_merged_parent.previous_recipe")
    checkpoint_record = _record(
        binding.get("previous_checkpoint_manifest"),
        "sft_merged_parent.previous_checkpoint_manifest",
    )
    declared = {
        (item.get("path"), item.get("sha256"))
        for item in recipe.get("inputs", []) if isinstance(item, Mapping)
    }
    if any((record["path"], record["sha256"]) not in declared for record in (
            manifest_record, recipe_record, checkpoint_record)):
        raise ValueError("Expanded SFT parent lineage files must be declared immutable inputs")
    model_record = next((item for item in recipe.get("inputs", [])
                         if isinstance(item, Mapping)
                         and item.get("path") == str(Path(recipe["model_path"]) / "model.safetensors")), None)
    if not model_record or model_record.get("sha256") != SELECTED_PARENT_WEIGHTS_SHA256:
        raise ValueError("Expanded SFT model input does not match selected merged E750 weights")
    manifest = json.loads(Path(manifest_record["path"]).read_text())
    previous_recipe = json.loads(Path(recipe_record["path"]).read_text())
    checkpoint = json.loads(Path(checkpoint_record["path"]).read_text())
    parent = recipe["identity"]["parent"]
    checks = (
        manifest.get("schema_version") == PARENT_MANIFEST_SCHEMA,
        manifest.get("kind") == "merged_task_sft",
        manifest.get("merged_model_path") == recipe.get("model_path"),
        manifest.get("merged_weights_sha256") == parent.get("weights_sha256"),
        manifest.get("base_model_revision") == parent.get("revision"),
        manifest_record["sha256"] == parent.get("sft_merged_manifest_sha256"),
        manifest.get("recipe_sha256") == recipe_record["sha256"],
        manifest.get("checkpoint_manifest_sha256") == checkpoint_record["sha256"],
        manifest.get("sft_identity") == previous_recipe.get("identity"),
        manifest.get("sft_checkpoint_manifest") == checkpoint,
        checkpoint.get("identity") == previous_recipe.get("identity"),
        checkpoint.get("full") is True,
        checkpoint.get("step") == SELECTED_PARENT_STEP,
        manifest.get("source_cursor") == SELECTED_PARENT_CURSOR,
        manifest.get("tokenizer_original_bytes_restored") is True,
        manifest.get("tokenizer", {}).get("tokenizer_json_sha256")
            == recipe["identity"]["tokenizer"].get("tokenizer_json_sha256"),
    )
    if not all(checks):
        raise ValueError("Expanded SFT merged-parent identity or lineage is inconsistent")


def validate_expanded_rows(recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require complete target tails within the independent TRAIN target cap."""
    cap = recipe.get("parameters", {}).get("train_max_target_tokens")
    if cap != TRAIN_MAX_TARGET_TOKENS:
        raise ValueError("Expanded target validation requires the 1024-token TRAIN cap")
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
            raise ValueError(f"{ident}: complete target including EOS exceeds 1024 TRAIN tokens")
        if any(type(token) is not int or not 0 <= token < 130560 for token in ids):
            raise ValueError(f"{ident}: target metadata contains an invalid token")
        target_lengths.append(len(tail))
    if not target_lengths:
        raise ValueError("Task SFT target data is empty")
    return {
        "rows": len(target_lengths),
        "target_tokens_including_eos": sum(target_lengths),
        "max_target_tokens_including_eos": max(target_lengths),
        "train_max_target_tokens": cap,
        "all_complete_targets_within_train_cap": max(target_lengths) <= cap,
    }


def validate_expanded_admission(recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind the already-verified files and full resume state to the task identity.

    The reviewed preflight verifies input/checkpoint bytes before invoking this
    hook. This task-only hook checks their cross-file meaning without touching
    the legacy recipe path or loading serialized optimizer/model state.
    """
    summary = validate_expanded_rows(recipe, rows)
    validate_sft_merged_parent(recipe)
    schedule = json.loads(Path(recipe["draw_schedule"]["path"]).read_text())
    data_identity = recipe["identity"]["data"]
    coverage = schedule.get("coverage", {})
    checks = schedule.get("checks", {})
    if (summary["rows"] != 15006 or data_identity.get("candidate_rows") != 15006
            or data_identity.get("candidate_edit_rows") != 13912
            or data_identity.get("candidate_noop_rows") != 1094
            or data_identity.get("new_finish_rows") != 3503
            or coverage.get("distinct_rows_drawn") != 15006
            or coverage.get("first_all_rows_step") != 1160
            or schedule.get("draw_count") != 18560
            or schedule.get("max_steps") != 1160
            or schedule.get("excluded") != []
            or checks.get("all_eligible_ids_drawn") is not True
            or checks.get("no_target_truncation") is not True
            or checks.get("every_unique_noop_before_noop_replay") is not True
            or checks.get("every_unique_edit_before_edit_replay") is not True):
        raise ValueError("Full15006 schedule/data coverage identity is inconsistent")
    if recipe.get("resume_from"):
        root = Path(recipe["resume_from"])
        state = json.loads((root / "campaign-state.json").read_text())
        trainer = json.loads((root / "trainer_state.json").read_text())
        manifest = json.loads((root / "campaign-manifest.json").read_text())
        step = state.get("step")
        checkpoint_identity = reviewed_sft.resume_checkpoint_identity(recipe, recipe["identity"])
        compatible_step = (recipe.get("resume_identity_compatibility") or {}).get("checkpoint_step")
        if (type(step) is not int or step not in (250, 500, compatible_step)
                or trainer.get("global_step") != step or manifest.get("step") != step
                or state.get("full") is not True or manifest.get("full") is not True
                or state.get("identity") != checkpoint_identity
                or manifest.get("identity") != checkpoint_identity):
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
    expanded_target_only_policy(recipe)
    return reviewed_sft.preflight(
        recipe, policy_fn=expanded_target_only_policy, row_validator=validate_expanded_admission,
    )


def run(recipe: dict[str, Any]) -> None:
    expanded_target_only_policy(recipe)
    reviewed_sft.run(
        recipe, policy_fn=expanded_target_only_policy, row_validator=validate_expanded_admission,
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
            "split_id": schedule["split_id"],
            "train_target_cap": TRAIN_MAX_TARGET_TOKENS,
            "development_generation_cap": DEV_COMPLETION_CAP,
            "evaluation_route": recipe["development_evaluation_route"],
            "CUDA_started": False,
        }))
    else:
        run(recipe)


if __name__ == "__main__":
    main()
