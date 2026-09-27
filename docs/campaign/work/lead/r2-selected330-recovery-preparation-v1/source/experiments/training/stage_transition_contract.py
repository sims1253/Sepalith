"""Fail-closed metadata contract for a full-weight CPT corpus transition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(record, name):
    path = Path(record.get("path", ""))
    require(path.is_file() and sha256(path) == record.get("sha256"), f"{name} differs")
    return path


def source_identity(recipe):
    """Reproduce the frozen source-stage checkpoint identity, without substitution."""
    schema = recipe.get("schema")
    common = {
        "parent": {"candidate_id": recipe["parent"]["candidate_id"], "weights_sha256": recipe["parent"]["files"]["model.safetensors"], "saved_precision": recipe["parent"]["saved_precision"]},
        "tokenizer": {"sha256": recipe["parent"]["files"]["tokenizer.json"], "bos": 0, "eos_pad": 1},
        "renderer": {"kind": "pretokenized_raw_r_cpt_v1", "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"]},
        "source": {"manifest_sha256": recipe["source"]["manifest_sha256"]},
    }
    if schema == "sepalith.sft11.full-weight-cpt-representative-bound.v3":
        common.update({
            "data": {"cohort_id": recipe["cohort"]["id"], "rows_sha256": recipe["cohort"]["rows"]["sha256"]},
            "policy": {"stage": "full_weight_cpt_representative_v3", "optimizer": recipe["runtime"]["optimizer"]},
            "schedule": {k: recipe["runtime"][k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_ratio", "checkpoint_every", "mandatory_stop_step")},
        })
        return common
    if schema == "sepalith.sft11.full-weight-cpt-full-corpus-bound.v1":
        common.update({
            "data": {"cohort_id": recipe["cohort"]["id"], "rows_sha256": recipe["cohort"]["rows"]["sha256"], "streaming_cache_manifest_sha256": recipe["cohort"]["streaming_cache"]["manifest_sha256"]},
            "policy": {"stage": "full_weight_cpt_full_corpus_v1", "optimizer": recipe["runtime"]["optimizer"]},
            "schedule": {k: recipe["runtime"][k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_steps", "checkpoint_every", "mandatory_stop_step")},
        })
        return common
    if schema == "sepalith.sft11.full-weight-cpt-stage-transition-bound.v1":
        common["parent"]["source_checkpoint_manifest_sha256"] = recipe["transition"]["source_checkpoint"]["manifest_sha256"]
        common.update({
            "data": {"cohort_id": recipe["cohort"]["id"], "rows_sha256": recipe["cohort"]["rows"]["sha256"], "streaming_cache_manifest_sha256": recipe["cohort"]["streaming_cache"]["manifest_sha256"]},
            "policy": {"stage": "full_weight_cpt_stage_transition_v1", "optimizer": recipe["runtime"]["optimizer"]},
            "schedule": {**{k: recipe["runtime"][k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_steps", "checkpoint_every", "mandatory_stop_step")}, "global_optimizer_step_offset": recipe["transition"]["global_optimizer_step_offset"], "destination_updates": recipe["cohort"]["updates"]},
        })
        return common
    raise ValueError("unsupported source recipe schema")


def warmup_steps(recipe):
    runtime = recipe["runtime"]
    if "warmup_steps" in runtime:
        return runtime["warmup_steps"]
    return int(runtime["max_steps"] * runtime["warmup_ratio"])


def validate_source_closure(source_recipe, transition):
    record = transition["source_manifest"]
    manifest_path = pin(record, "source-stage source manifest")
    require(source_recipe["source"]["manifest_path"] == str(manifest_path.resolve()), "source recipe names another source manifest")
    require(source_recipe["source"]["manifest_sha256"] == record["sha256"], "source recipe source identity differs")
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest.get("files"), list) and manifest["files"], "source-stage source manifest is empty")
    for item in manifest["files"]:
        target = manifest_path.parent / item["path"]
        require(target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"], f"source-stage source differs:{item['path']}")


def inspect_transition(recipe, *, verify_payload=False):
    transition = recipe["transition"]
    decision_path = pin(transition["source_decision"], "source checkpoint decision")
    decision = json.loads(decision_path.read_text())
    source_recipe_path = pin(transition["source_recipe"], "source recipe")
    source_recipe = json.loads(source_recipe_path.read_text())
    validate_source_closure(source_recipe, transition)
    checkpoint = Path(transition["source_checkpoint"]["path"])
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), "source checkpoint missing")
    manifest_path = checkpoint / "campaign-manifest.json"
    require(manifest_path.is_file() and sha256(manifest_path) == transition["source_checkpoint"]["manifest_sha256"], "source checkpoint manifest differs")
    manifest = json.loads(manifest_path.read_text())
    state = json.loads((checkpoint / "campaign-state.json").read_text())
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    for name in ("campaign-state.json", "trainer_state.json", "scheduler.pt", "rng_state.pth"):
        target = checkpoint / name
        recorded = manifest.get("files", {}).get(name, {})
        require(target.is_file() and target.stat().st_size == recorded.get("bytes") and sha256(target) == recorded.get("sha256"), f"source checkpoint metadata differs:{name}")
    expected_identity = source_identity(source_recipe)
    offset = transition["global_optimizer_step_offset"]
    preserve_prefix = transition.get("destination_sampler") == "preserve_verified_prefix"
    step = transition["source_global_step"] if preserve_prefix else offset
    from native_runtime_contract import canonical_source_checkpoint
    decision_checkpoint = canonical_source_checkpoint(recipe, checkpoint)
    require(decision.get("checkpoint") == decision_checkpoint and decision.get("checkpoint_manifest_sha256") == transition["source_checkpoint"]["manifest_sha256"] and decision.get("weights_sha256") == manifest.get("files", {}).get("model.safetensors", {}).get("sha256"), "source decision/checkpoint identity differs")
    require(manifest.get("checkpoint_kind") == "full_weights" and manifest.get("full") is True, "source checkpoint is not full-weight resumable state")
    require(manifest.get("identity") == expected_identity and state.get("identity") == expected_identity, "source checkpoint identity differs from source recipe")
    require(manifest.get("step") == state.get("step") == trainer_state.get("global_step") == step, "source checkpoint/global optimizer step differs")
    admitted_steps = {source_recipe["runtime"]["mandatory_stop_step"], source_recipe["runtime"]["max_steps"]}
    admitted_steps.update(source_recipe["runtime"].get("selected_milestones", ()))
    admitted_steps.update(source_recipe["runtime"].get("evaluation_steps", ()))
    root_selected_checkpoint = (
        decision.get("schema") == "sepalith.campaign-receipt.v1"
        and decision.get("owner") == "lead"
        and decision.get("global_step") == step
        and str(decision.get("status", "")).endswith("_continuation_candidate")
    )
    require(type(step) is int and step >= 0 and (step in admitted_steps or root_selected_checkpoint), "source checkpoint is not an admitted source milestone")
    sampler = state.get("sampler", {})
    if source_recipe.get("schema") == "sepalith.sft11.full-weight-cpt-stage-transition-bound.v1":
        prior_offset = source_recipe["transition"]["global_optimizer_step_offset"]
        require(type(prior_offset) is int and 0 <= prior_offset < step, "prior transition offset differs")
        require(sampler.get("method") == "sequential_frozen_draw_schedule_stage_local", "source stage-local sampler method differs")
        require(sampler.get("global_optimizer_step_offset") == prior_offset, "source stage-local offset differs")
        require(sampler.get("draw_schedule_sha256") == source_recipe["cohort"]["draw_schedule"]["sha256"], "source stage-local schedule differs")
        require(sampler.get("stage_cursor") == sampler.get("cursor") == (step - prior_offset) * 16, "source stage-local cursor differs")
        require(sampler.get("global_step") == step and sampler.get("ignore_data_skip") is True, "source stage-local global state differs")
    else:
        require(sampler.get("cursor") == step * 16 and sampler.get("global_step") == step and sampler.get("ignore_data_skip") is False, "source sampler state differs")
    require(recipe["runtime"]["optimizer"] == source_recipe["runtime"]["optimizer"], "optimizer configuration cannot change across transition")
    require(recipe["runtime"]["learning_rate"] == source_recipe["runtime"]["learning_rate"], "learning rate cannot change across transition")
    require(recipe["runtime"]["scheduler"] == source_recipe["runtime"]["scheduler"] == "constant_with_warmup", "only exact constant-with-warmup continuation is prepared")
    require(recipe["runtime"]["warmup_steps"] == warmup_steps(source_recipe), "warmup schedule cannot restart or change")
    require(step >= recipe["runtime"]["warmup_steps"], "source checkpoint precedes completed warmup")
    require(recipe["runtime"]["max_steps"] == offset + recipe["cohort"]["updates"], "destination global horizon does not include source-step offset")
    destination_cursor = 0
    if preserve_prefix:
        from prefix_extension_contract import verify_consumed_prefix
        destination_cursor = verify_consumed_prefix(source_recipe, recipe, state)
        require(offset + destination_cursor // 16 == step, "preserved prefix/global optimizer step differs")
    require(recipe["cohort"]["rows"]["sha256"] != source_recipe["cohort"]["rows"]["sha256"], "stage transition must bind a distinct destination corpus")
    files = manifest.get("files", {})
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "tokenizer.json"):
        require(name in files and files[name].get("bytes", 0) > 0 and len(files[name].get("sha256", "")) == 64, f"source checkpoint lacks {name}")
    parent = recipe["parent"]
    require(parent.get("kind") == "full_weight_stage_checkpoint" and Path(parent.get("path", "")).resolve() == checkpoint.resolve(), "destination parent is not the source checkpoint")
    require(parent["files"]["model.safetensors"] == files["model.safetensors"]["sha256"] and parent["files"]["tokenizer.json"] == files["tokenizer.json"]["sha256"], "destination parent bytes differ from source checkpoint")
    if verify_payload:
        from campaign_checkpoint import verify_checkpoint
        verify_checkpoint(checkpoint, expected_identity, require_full=True, expected_checkpoint_kind="full_weights", native_bundle_id=recipe.get('storage_relocation',{}).get('bundle_id'))
    return {"source_step": step, "source_recipe_schema": source_recipe["schema"], "source_identity_sha256": hashlib.sha256(canonical(expected_identity)).hexdigest(), "source_recipe_sha256": sha256(source_recipe_path), "source_checkpoint_manifest_sha256": sha256(manifest_path), "source_sampler_provenance": sampler, "destination_initial_cursor": destination_cursor}


def stage_cursor_from_checkpoint(campaign_state, source_step, destination_schedule_sha256):
    sampler = campaign_state.get("sampler", {})
    require(sampler.get("method") == "sequential_frozen_draw_schedule_stage_local", "destination sampler method differs")
    require(sampler.get("global_optimizer_step_offset") == source_step, "destination checkpoint source-step offset differs")
    require(sampler.get("draw_schedule_sha256") == destination_schedule_sha256, "destination sampler schedule differs")
    cursor = sampler.get("stage_cursor")
    require(type(cursor) is int and cursor >= 0 and cursor % 16 == 0, "destination stage cursor differs")
    require(sampler.get("cursor") == cursor, "destination cursor differs from stage cursor")
    require(sampler.get("global_step") == source_step + cursor // 16, "destination cursor/global step differs")
    require(sampler.get("ignore_data_skip") is True, "destination checkpoint lacks explicit dataset-managed cursor")
    return cursor
