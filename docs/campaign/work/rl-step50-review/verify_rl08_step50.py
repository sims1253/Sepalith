#!/usr/bin/env python3
"""Independent, bounded review of the RL continuation through DEV50.

This verifier reads the named completed checkpoint and the first 25 update
windows of the still-running continuation.  It uses the pinned campaign
parser for the 75 DEV cases, streams known checkpoint hashes while excluding
the adapter model weight, and never imports torch, CUDA, Transformers, TRL,
Unsloth, or a model.  It does not read the large live telemetry file.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


RUN_DEFAULT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c")
RECIPE_DEFAULT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/main-rl-preparation/primary-mb4-full5-c.recipe.json")
SEQUENCE_DEFAULT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/rl-pool/source-row-draw-sequence-v6-interleaved-mb4.json")
PANEL_DEFAULT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl")
ROWS_DEFAULT = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/eligible-train-rows.jsonl")
PROTOCOL_DEFAULT = Path("/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source/packages/sepalith/src")

EXPECTED = {
    "source": "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9",
    "recipe": "fc1de4c4b4e9b1da9617edbc99162d1db1df31080de95eed2bc75c695d5725c3",
    "identity": "48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2",
    "schedule": "2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132",
    "sequence": "dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6",
    "selected": "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d",
    "ordered": "7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d",
    "row_identity": "d3187195f0ebe401b69df244bbbff0f2a28fb0500185428701f933e31e8eab60",
    "rows": "e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602",
    "context": "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f",
    "panel": "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21",
    "parent_manifest": "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12",
    "parent_weights": "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d",
    "parent_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
}

EXPECTED_DATA = {
    "row_count": 8440, "split": "train", "admission_status": "admitted",
    "selected_ids_sha256": EXPECTED["selected"], "ordered_ids_sha256": EXPECTED["ordered"],
    "row_identity_sha256": EXPECTED["row_identity"], "rows_sha256": EXPECTED["rows"],
    "context_sha256": EXPECTED["context"], "sidecar_artifact_sha256": EXPECTED["context"],
}
EXPECTED_POLICY = {
    "beta": 0, "candidate_count": 4, "completion_max_tokens": 192,
    "context_max_tokens": 2240, "cuda_memory_fraction": 0.75,
    "expected_attachments": 294, "expected_trainable_parameters": 25116672,
    "generation_groups_per_call": 2, "gradient_accumulation_steps": 8,
    "lora_alpha": 16, "lora_rank": 16, "loss_type": "bnpo",
    "model_load_max_seq_length": 4096, "per_device_train_batch_size": 4,
    "prompt_max_tokens": 2048, "rollout_rows_per_update": 32,
    "scale_rewards": "group",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
EXPECTED_RENDERER = {
    "renderer_id": "zeta2-prm03-v1", "schema_version": "sepalith.prompt.prm03.v1",
    "no_edit": "[NO_EDIT]", "terminal": ">>>>>>> UPDATED",
    "tokenization_policy": "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1",
}
EXPECTED_SCHEDULE = {
    "buffer_reuse": 8, "development_cases": 75, "development_every": 25,
    "development_panel_sha256": EXPECTED["panel"], "first_development_update": 5,
    "full_save_steps": 5, "generation_batch_size": 32, "light_save_steps": 5,
    "main_update_ceiling": 3000, "num_iterations": 1,
    "sampler_id": "campaign-repeat-manifest-order-v1", "seed": 3407,
    "source_draw_schedule_sha256": EXPECTED["schedule"],
    "source_draw_sequence_sha256": EXPECTED["sequence"], "source_draws": 24000,
    "source_draws_per_update": 8, "stage_id": "primary-grpo-p2-2048x192-v5-full5",
    "steps_per_generation": 8,
}
EXPECTED_SOURCE = {
    "protocol_sha256": "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
    "rl02_admission_status": "rl_data_admitted", "source_schedule_sha256": EXPECTED["schedule"],
    "trainer_sha256": "78d27aa98cebc80292d1871a39821eee5a1a705b5a8f412ce270d1707d724443",
    "trl_version": "0.24.0",
}
EXPECTED_TOKENIZER = {"bos_id": 0, "eos_id": 1, "pad_id": 1, "vocab_size": 130560, "revision": EXPECTED["parent_revision"]}
EXPECTED_PARENT = {"kind": "merged_sft", "manifest_sha256": EXPECTED["parent_manifest"], "merged_weights_sha256": EXPECTED["parent_weights"], "base_model_revision": EXPECTED["parent_revision"]}
LONG_DEV_ID = "dat07-existing-719cd49683667d0fb86fb2fa"
GENERATION_SCHEMA = "sepalith.rl.generation-record.v1"
GRADIENT_SCHEMA = "sepalith.rl.gradient-record.v1"
HEX64 = set("0123456789abcdef")


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def add_check(checks: list[dict[str, Any]], name: str, status: str, details: Any = None) -> None:
    record: dict[str, Any] = {"name": name, "status": status}
    if details is not None:
        record["details"] = details
    checks.append(record)


def safe(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items() if not isinstance(v, (dict, list))}
    if isinstance(value, list):
        return [safe(v) for v in value[:12]] + (["..."] if len(value) > 12 else [])
    return value


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: non-object JSONL row")
            rows.append(value)
    return rows


def jq_projection(path: Path, expression: str) -> dict[str, Any]:
    process = subprocess.run(["jq", "-c", expression, str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        raise RuntimeError(f"jq failed for {path}: {process.stderr.strip()[:300]}")
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"projection for {path} is not an object")
    return value


MANIFEST_PROJECTION = r'''{
  step, full, schema_version,
  identity: {
    data: {row_count: .identity.data.row_count, split: .identity.data.split, admission_status: .identity.data.admission_status,
      selected_ids_sha256: .identity.data.selected_ids_sha256, ordered_ids_sha256: .identity.data.ordered_ids_sha256,
      row_identity_sha256: .identity.data.row_identity_sha256, rows_sha256: .identity.data.rows_sha256,
      context_sha256: .identity.data.context_sha256, sidecar_artifact_sha256: .identity.data.sidecar_artifact_sha256},
    parent: {kind: .identity.parent.kind, manifest_sha256: .identity.parent.manifest_sha256,
      merged_weights_sha256: .identity.parent.merged_weights_sha256, base_model_revision: .identity.parent.base_model_revision},
    policy: {beta: .identity.policy.beta, candidate_count: .identity.policy.candidate_count,
      completion_max_tokens: .identity.policy.completion_max_tokens, context_max_tokens: .identity.policy.context_max_tokens,
      cuda_memory_fraction: .identity.policy.cuda_memory_fraction, expected_attachments: .identity.policy.expected_attachments,
      expected_trainable_parameters: .identity.policy.expected_trainable_parameters,
      generation_groups_per_call: .identity.policy.generation_groups_per_call,
      gradient_accumulation_steps: .identity.policy.gradient_accumulation_steps, lora_alpha: .identity.policy.lora_alpha,
      lora_rank: .identity.policy.lora_rank, loss_type: .identity.policy.loss_type,
      model_load_max_seq_length: .identity.policy.model_load_max_seq_length,
      per_device_train_batch_size: .identity.policy.per_device_train_batch_size,
      prompt_max_tokens: .identity.policy.prompt_max_tokens, rollout_rows_per_update: .identity.policy.rollout_rows_per_update,
      scale_rewards: .identity.policy.scale_rewards, target_modules: .identity.policy.target_modules},
    renderer: {renderer_id: .identity.renderer.renderer_id, schema_version: .identity.renderer.schema_version,
      no_edit: .identity.renderer.no_edit, terminal: .identity.renderer.terminal,
      tokenization_policy: .identity.renderer.tokenization_policy},
    schedule: {buffer_reuse: .identity.schedule.buffer_reuse, development_cases: .identity.schedule.development_cases,
      development_every: .identity.schedule.development_every, development_panel_sha256: .identity.schedule.development_panel_sha256,
      first_development_update: .identity.schedule.first_development_update, full_save_steps: .identity.schedule.full_save_steps,
      generation_batch_size: .identity.schedule.generation_batch_size, light_save_steps: .identity.schedule.light_save_steps,
      main_update_ceiling: .identity.schedule.main_update_ceiling, num_iterations: .identity.schedule.num_iterations,
      sampler_id: .identity.schedule.sampler_id, seed: .identity.schedule.seed,
      source_draw_schedule_sha256: .identity.schedule.source_draw_schedule_sha256,
      source_draw_sequence_sha256: .identity.schedule.source_draw_sequence_sha256, source_draws: .identity.schedule.source_draws,
      source_draws_per_update: .identity.schedule.source_draws_per_update, stage_id: .identity.schedule.stage_id,
      steps_per_generation: .identity.schedule.steps_per_generation},
    source: {protocol_sha256: .identity.source.protocol_sha256, rl02_admission_status: .identity.source.rl02_admission_status,
      source_schedule_sha256: .identity.source.source_schedule_sha256, trainer_sha256: .identity.source.trainer_sha256,
      trl_version: .identity.source.trl_version},
    tokenizer: {bos_id: .identity.tokenizer.bos_id, eos_id: .identity.tokenizer.eos_id,
      pad_id: .identity.tokenizer.pad_id, vocab_size: .identity.tokenizer.vocab_size, revision: .identity.tokenizer.revision}
  },
  files: (.files | to_entries | map({name: .key, bytes: .value.bytes, sha256: .value.sha256}))
}'''

STATE_PROJECTION = r'''{
  step, full, resume_validation,
  identity: {
    data: {row_count: .identity.data.row_count, split: .identity.data.split, admission_status: .identity.data.admission_status,
      selected_ids_sha256: .identity.data.selected_ids_sha256, ordered_ids_sha256: .identity.data.ordered_ids_sha256,
      row_identity_sha256: .identity.data.row_identity_sha256, rows_sha256: .identity.data.rows_sha256,
      context_sha256: .identity.data.context_sha256, sidecar_artifact_sha256: .identity.data.sidecar_artifact_sha256},
    parent: {kind: .identity.parent.kind, manifest_sha256: .identity.parent.manifest_sha256,
      merged_weights_sha256: .identity.parent.merged_weights_sha256, base_model_revision: .identity.parent.base_model_revision},
    policy: {candidate_count: .identity.policy.candidate_count, model_load_max_seq_length: .identity.policy.model_load_max_seq_length,
      per_device_train_batch_size: .identity.policy.per_device_train_batch_size,
      gradient_accumulation_steps: .identity.policy.gradient_accumulation_steps,
      rollout_rows_per_update: .identity.policy.rollout_rows_per_update},
    renderer: {renderer_id: .identity.renderer.renderer_id, schema_version: .identity.renderer.schema_version},
    schedule: {source_draw_schedule_sha256: .identity.schedule.source_draw_schedule_sha256,
      source_draw_sequence_sha256: .identity.schedule.source_draw_sequence_sha256,
      steps_per_generation: .identity.schedule.steps_per_generation, stage_id: .identity.schedule.stage_id},
    source: {protocol_sha256: .identity.source.protocol_sha256, rl02_admission_status: .identity.source.rl02_admission_status,
      source_schedule_sha256: .identity.source.source_schedule_sha256, trainer_sha256: .identity.source.trainer_sha256,
      trl_version: .identity.source.trl_version},
    tokenizer: {bos_id: .identity.tokenizer.bos_id, eos_id: .identity.tokenizer.eos_id,
      pad_id: .identity.tokenizer.pad_id, vocab_size: .identity.tokenizer.vocab_size}
  },
  sampler: {buffer_reuse: .sampler.buffer_reuse, candidate_count: .sampler.candidate_count,
    consumed_prompt_copies: .sampler.consumed_prompt_copies, consumed_rows: .sampler.consumed_rows,
    current_index: .sampler.current_index, epoch: .sampler.epoch, generation_rows_per_update: .sampler.generation_rows_per_update,
    num_samples: .sampler.num_samples, prompt_groups_per_batch: .sampler.prompt_groups_per_batch,
    repeat_count: .sampler.repeat_count, sampler_id: .sampler.sampler_id, sampler_rows_per_update: .sampler.sampler_rows_per_update,
    seed: .sampler.seed, selected_id_index: .sampler.selected_id_index, shuffle: .sampler.shuffle,
    source_draw_cursor: .sampler.source_draw_cursor, source_draw_schedule_sha256: .sampler.source_draw_schedule_sha256,
    source_draw_sequence_sha256: .sampler.source_draw_sequence_sha256, source_draws: .sampler.source_draws,
    source_draws_bound: .sampler.source_draws_bound,
    geometry: {generation_batch_size: .sampler.geometry.generation_batch_size,
      gradient_accumulation_steps: .sampler.geometry.gradient_accumulation_steps,
      per_device_train_batch_size: .sampler.geometry.per_device_train_batch_size,
      steps_per_generation: .sampler.geometry.steps_per_generation},
    data_identity: {row_count: .sampler.data_identity.row_count, split: .sampler.data_identity.split,
      admission_status: .sampler.data_identity.admission_status, selected_ids_sha256: .sampler.data_identity.selected_ids_sha256,
      ordered_ids_sha256: .sampler.data_identity.ordered_ids_sha256, row_identity_sha256: .sampler.data_identity.row_identity_sha256,
      rows_sha256: .sampler.data_identity.rows_sha256, context_sha256: .sampler.data_identity.context_sha256,
      sidecar_artifact_sha256: .sampler.data_identity.sidecar_artifact_sha256}}
}'''

RECIPE_PROJECTION = r'''{
  id, schema_version, model_load_max_seq_length, max_steps, full_save_steps, light_save_steps,
  development_max_new_tokens, renderer_id, evaluator_factory, evaluation_steps,
  parameters: {max_sequence_tokens: .parameters.max_sequence_tokens},
  data: {admission_status: .data.admission_status, context_sha256: .data.context_sha256,
    rows_sha256: .data.rows_sha256,
    selected_ids_sha256: .data.selected_ids_sha256, sidecar_artifact_sha256: .data.sidecar_artifact_sha256,
    source_draw_schedule_sha256: .data.source_draw_schedule_sha256, source_draws: .data.source_draws},
  development_panel: {sha256: .development_panel.sha256},
  main_development_panel: {rows: .main_development_panel.rows, sha256: .main_development_panel.sha256},
  identity: {data: {row_count: .identity.data.row_count, split: .identity.data.split, admission_status: .identity.data.admission_status,
      selected_ids_sha256: .identity.data.selected_ids_sha256, ordered_ids_sha256: .identity.data.ordered_ids_sha256,
      row_identity_sha256: .identity.data.row_identity_sha256, rows_sha256: .identity.data.rows_sha256,
      context_sha256: .identity.data.context_sha256, sidecar_artifact_sha256: .identity.data.sidecar_artifact_sha256},
    parent: {kind: .identity.parent.kind, manifest_sha256: .identity.parent.manifest_sha256,
      merged_weights_sha256: .identity.parent.merged_weights_sha256, base_model_revision: .identity.parent.base_model_revision},
    policy: {beta: .identity.policy.beta, candidate_count: .identity.policy.candidate_count,
      completion_max_tokens: .identity.policy.completion_max_tokens, context_max_tokens: .identity.policy.context_max_tokens,
      cuda_memory_fraction: .identity.policy.cuda_memory_fraction, expected_attachments: .identity.policy.expected_attachments,
      expected_trainable_parameters: .identity.policy.expected_trainable_parameters,
      generation_groups_per_call: .identity.policy.generation_groups_per_call,
      gradient_accumulation_steps: .identity.policy.gradient_accumulation_steps, lora_alpha: .identity.policy.lora_alpha,
      lora_rank: .identity.policy.lora_rank, loss_type: .identity.policy.loss_type,
      model_load_max_seq_length: .identity.policy.model_load_max_seq_length, per_device_train_batch_size: .identity.policy.per_device_train_batch_size,
      prompt_max_tokens: .identity.policy.prompt_max_tokens, rollout_rows_per_update: .identity.policy.rollout_rows_per_update,
      scale_rewards: .identity.policy.scale_rewards, target_modules: .identity.policy.target_modules},
    renderer: {renderer_id: .identity.renderer.renderer_id, schema_version: .identity.renderer.schema_version,
      no_edit: .identity.renderer.no_edit, terminal: .identity.renderer.terminal, tokenization_policy: .identity.renderer.tokenization_policy},
    schedule: {buffer_reuse: .identity.schedule.buffer_reuse, development_cases: .identity.schedule.development_cases,
      development_every: .identity.schedule.development_every, development_panel_sha256: .identity.schedule.development_panel_sha256,
      first_development_update: .identity.schedule.first_development_update, full_save_steps: .identity.schedule.full_save_steps,
      generation_batch_size: .identity.schedule.generation_batch_size, light_save_steps: .identity.schedule.light_save_steps,
      main_update_ceiling: .identity.schedule.main_update_ceiling, num_iterations: .identity.schedule.num_iterations,
      sampler_id: .identity.schedule.sampler_id, seed: .identity.schedule.seed,
      source_draw_schedule_sha256: .identity.schedule.source_draw_schedule_sha256, source_draw_sequence_sha256: .identity.schedule.source_draw_sequence_sha256,
      source_draws: .identity.schedule.source_draws, source_draws_per_update: .identity.schedule.source_draws_per_update,
      stage_id: .identity.schedule.stage_id, steps_per_generation: .identity.schedule.steps_per_generation},
    source: {protocol_sha256: .identity.source.protocol_sha256, rl02_admission_status: .identity.source.rl02_admission_status,
      source_schedule_sha256: .identity.source.source_schedule_sha256, trainer_sha256: .identity.source.trainer_sha256, trl_version: .identity.source.trl_version},
    tokenizer: {bos_id: .identity.tokenizer.bos_id, eos_id: .identity.tokenizer.eos_id, pad_id: .identity.tokenizer.pad_id,
      vocab_size: .identity.tokenizer.vocab_size, revision: .identity.tokenizer.revision}}
}'''


def compare_mapping(checks: list[dict[str, Any]], prefix: str, actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        add_check(checks, prefix, "fail", {"actual_type": type(actual).__name__})
        return False
    errors = {key: {"expected": wanted, "actual": safe(actual.get(key))} for key, wanted in expected.items() if actual.get(key) != wanted}
    add_check(checks, prefix, "pass" if not errors else "fail", {"mismatches": errors} if errors else None)
    return not errors


def verify_checkpoint(checks: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    archive = run / "archive/full/checkpoint-50"
    output = run / "output/checkpoint-50"
    result: dict[str, Any] = {"archive": str(archive), "output": str(output), "model_weight_hash": "skipped_by_scope"}
    if not archive.is_dir() or not output.is_dir():
        add_check(checks, "checkpoint.directories", "fail", {"archive": archive.is_dir(), "output": output.is_dir()})
        return {**result, "status": "fail"}
    try:
        am = jq_projection(archive / "campaign-manifest.json", MANIFEST_PROJECTION)
        om = jq_projection(output / "campaign-manifest.json", MANIFEST_PROJECTION)
        ast = jq_projection(archive / "campaign-state.json", STATE_PROJECTION)
        ost = jq_projection(output / "campaign-state.json", STATE_PROJECTION)
    except Exception as error:
        add_check(checks, "checkpoint.projection", "fail", repr(error)[:300])
        return {**result, "status": "fail"}

    add_check(checks, "checkpoint.manifest.header", "pass" if {k: am.get(k) for k in ("step", "full", "schema_version")} == {"step": 50, "full": True, "schema_version": 1} else "fail", {k: am.get(k) for k in ("step", "full", "schema_version")})
    add_check(checks, "checkpoint.manifest.archive_output_identity", "pass" if am.get("identity") == om.get("identity") else "fail")
    add_check(checks, "checkpoint.manifest.archive_output_header", "pass" if {k: am.get(k) for k in ("step", "full", "schema_version")} == {k: om.get(k) for k in ("step", "full", "schema_version")} else "fail")
    identity_ok = (
        compare_mapping(checks, "checkpoint.identity.data", am["identity"].get("data"), EXPECTED_DATA)
        and compare_mapping(checks, "checkpoint.identity.parent", am["identity"].get("parent"), EXPECTED_PARENT)
        and compare_mapping(checks, "checkpoint.identity.policy", am["identity"].get("policy"), EXPECTED_POLICY)
        and compare_mapping(checks, "checkpoint.identity.renderer", am["identity"].get("renderer"), EXPECTED_RENDERER)
        and compare_mapping(checks, "checkpoint.identity.schedule", am["identity"].get("schedule"), EXPECTED_SCHEDULE)
        and compare_mapping(checks, "checkpoint.identity.source", am["identity"].get("source"), EXPECTED_SOURCE)
        and compare_mapping(checks, "checkpoint.identity.tokenizer", am["identity"].get("tokenizer"), EXPECTED_TOKENIZER)
    )
    add_check(checks, "checkpoint.identity.supplied_sha256", "pending", {"claimed": EXPECTED["identity"], "detail": "canonical identity hash is an admission value; projected identity fields match independently"})

    listed = {str(item["name"]): item for item in am.get("files", []) if isinstance(item, dict) and item.get("name")}
    add_check(checks, "checkpoint.file_inventory", "pass" if len(listed) == 12 else "fail", {"count": len(listed), "names": sorted(listed)})
    file_results: dict[str, Any] = {}
    hash_failures = 0
    for name, entry in sorted(listed.items()):
        if name == "adapter_model.safetensors":
            file_results[name] = {"status": "skipped_model_weight", "bytes_manifest": entry.get("bytes"), "sha256_manifest": entry.get("sha256")}
            continue
        file_results[name] = {}
        for label, directory in (("archive", archive), ("output", output)):
            path = directory / name
            if not path.is_file():
                file_results[name][label] = {"status": "fail", "error": "missing"}
                hash_failures += 1
                continue
            size, digest = sha256_file(path)
            same = size == entry.get("bytes") and digest == entry.get("sha256")
            file_results[name][label] = {"status": "pass" if same else "fail", "bytes": size, "sha256": digest}
            if not same:
                hash_failures += 1
    add_check(checks, "checkpoint.non_model_file_hashes", "pass" if hash_failures == 0 else "fail", {"checked_files": len(listed) - 1, "skipped": "adapter_model.safetensors"})
    add_check(checks, "checkpoint.archive_output_file_sets", "pass" if sorted(p.name for p in archive.iterdir() if p.is_file()) == sorted(p.name for p in output.iterdir() if p.is_file()) else "fail")

    state_expected = {"step": 50, "full": True}
    state_ok = compare_mapping(checks, "checkpoint.state.header", {k: ast.get(k) for k in state_expected}, state_expected)
    sampler_expected = {
        "buffer_reuse": 8, "candidate_count": 4, "consumed_prompt_copies": 3200,
        "consumed_rows": 12800, "current_index": 12800, "epoch": 0,
        "generation_rows_per_update": 32, "num_samples": 8440, "prompt_groups_per_batch": 8,
        "repeat_count": 8, "sampler_id": "campaign-repeat-manifest-order-v1", "sampler_rows_per_update": 256,
        "seed": 3407, "selected_id_index": 8243, "shuffle": False, "source_draw_cursor": 400,
        "source_draw_schedule_sha256": EXPECTED["schedule"], "source_draw_sequence_sha256": EXPECTED["sequence"],
        "source_draws": 24000, "source_draws_bound": True,
    }
    sampler_ok = compare_mapping(checks, "checkpoint.state.sampler", ast.get("sampler"), sampler_expected)
    geometry_ok = compare_mapping(checks, "checkpoint.state.geometry", ast.get("sampler", {}).get("geometry"), {"generation_batch_size": 32, "gradient_accumulation_steps": 8, "per_device_train_batch_size": 4, "steps_per_generation": 8})
    data_ok = compare_mapping(checks, "checkpoint.state.data_identity", ast.get("sampler", {}).get("data_identity"), EXPECTED_DATA)
    add_check(checks, "checkpoint.state.archive_output_equal", "pass" if ast == ost else "fail")
    add_check(checks, "checkpoint.state.resume_validation_text", "pending", ast.get("resume_validation"))
    return {
        **result,
        "status": "pass" if identity_ok and hash_failures == 0 and state_ok and sampler_ok and geometry_ok and data_ok else "fail",
        "manifest_sha256": sha256_file(archive / "campaign-manifest.json")[1],
        "state_sha256": sha256_file(archive / "campaign-state.json")[1],
        "files": file_results,
        "identity_fields_match": identity_ok,
        "state": {"step": ast.get("step"), "full": ast.get("full"), "source_draw_cursor": ast.get("sampler", {}).get("source_draw_cursor"), "consumed_rows": ast.get("sampler", {}).get("consumed_rows"), "selected_id_index": ast.get("sampler", {}).get("selected_id_index")},
    }


def verify_runtime(checks: list[dict[str, Any]], run: Path, recipe_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        recipe = jq_projection(recipe_path, RECIPE_PROJECTION)
        recipe_size, recipe_sha = sha256_file(recipe_path)
    except Exception as error:
        add_check(checks, "runtime.recipe", "fail", repr(error)[:300])
        return {"status": "fail"}
    add_check(checks, "runtime.recipe_sha256", "pass" if recipe_sha == EXPECTED["recipe"] else "fail", {"bytes": recipe_size, "sha256": recipe_sha})
    recipe_fields = {"id": "primary-grpo-p2-2048x192-v5-full5-c", "schema_version": "sepalith.prm07.rl-train.v1", "model_load_max_seq_length": 4096, "max_steps": 3000, "full_save_steps": 5, "light_save_steps": 5, "development_max_new_tokens": 512, "renderer_id": "zeta2-prm03-v1", "evaluator_factory": "campaign_eval:development_evaluator"}
    recipe_ok = compare_mapping(checks, "runtime.recipe.fields", recipe, recipe_fields)
    add_check(checks, "runtime.recipe.parameters.max_sequence_tokens", "pass" if recipe.get("parameters", {}).get("max_sequence_tokens") == 4096 else "fail", recipe.get("parameters"))
    add_check(checks, "runtime.recipe.evaluation_steps_explicit", "pass" if isinstance(recipe.get("evaluation_steps"), list) and recipe.get("evaluation_steps") else "fail", recipe.get("evaluation_steps"))
    recipe_data_ok = compare_mapping(checks, "runtime.recipe.data", recipe.get("data"), {"admission_status": "admitted", "context_sha256": EXPECTED["context"], "rows_sha256": EXPECTED["rows"], "selected_ids_sha256": EXPECTED["selected"], "sidecar_artifact_sha256": EXPECTED["context"], "source_draw_schedule_sha256": EXPECTED["schedule"], "source_draws": 24000})
    add_check(checks, "runtime.recipe.development_panel", "pass" if recipe.get("development_panel") == {"sha256": EXPECTED["panel"]} else "fail", recipe.get("development_panel"))

    entry_path = run / "output/entry-preflight.json"
    entry_expr = r'''{status, CUDA_started, training_started, framework_imports,
      geometry: {candidate_count: .geometry.candidate_count, generation_batch_size: .geometry.generation_batch_size,
        gradient_accumulation_steps: .geometry.gradient_accumulation_steps, per_device_train_batch_size: .geometry.per_device_train_batch_size,
        prompt_groups_per_update: .geometry.prompt_groups_per_update, rollout_rows_per_update: .geometry.rollout_rows_per_update, steps_per_generation: .geometry.steps_per_generation},
      limits, runtime, resume_audit,
      development_evaluator: {factory: .development_evaluator.factory, renderer_id: .development_evaluator.renderer_id, rows: .development_evaluator.rows,
        development_max_new_tokens: .development_evaluator.development_max_new_tokens, parameters: {max_sequence_tokens: .development_evaluator.parameters.max_sequence_tokens}},
      parent: {kind: .parent.kind, manifest_sha256: .parent.manifest_sha256, merged_weights_sha256: .parent.merged_weights_sha256, base_model_revision: .parent.base_model_revision}}
'''
    try:
        entry = jq_projection(entry_path, entry_expr)
    except Exception as error:
        add_check(checks, "runtime.entry_preflight", "fail", repr(error)[:300])
        return {"status": "fail", "recipe_sha256": recipe_sha}
    entry_expected = {"status": "preflight_pass", "CUDA_started": False, "training_started": False, "framework_imports": {"datasets": False, "torch": False, "transformers": False, "trl": False, "unsloth": False}}
    entry_ok = compare_mapping(checks, "runtime.entry_preflight.header", entry, entry_expected)
    add_check(checks, "runtime.entry_preflight.geometry", "pass" if entry.get("geometry") == {"candidate_count": 4, "generation_batch_size": 32, "gradient_accumulation_steps": 8, "per_device_train_batch_size": 4, "prompt_groups_per_update": 8, "rollout_rows_per_update": 32, "steps_per_generation": 8} else "fail", entry.get("geometry"))
    add_check(checks, "runtime.entry_preflight.parent", "pass" if entry.get("parent") == {"kind": "merged_sft", "manifest_sha256": EXPECTED["parent_manifest"], "merged_weights_sha256": EXPECTED["parent_weights"], "base_model_revision": EXPECTED["parent_revision"]} else "fail", entry.get("parent"))
    add_check(checks, "runtime.entry_preflight.development_evaluator", "pass" if entry.get("development_evaluator") == {"factory": "campaign_eval:development_evaluator", "renderer_id": "zeta2-prm03-v1", "rows": 75, "development_max_new_tokens": 512, "parameters": {"max_sequence_tokens": 4096}} else "fail", entry.get("development_evaluator"))
    resume_audit = entry.get("resume_audit", {})
    add_check(checks, "runtime.entry_preflight.resume_audit", "pass" if resume_audit == {"complete_optimizer_boundary": True, "consumed_rows": 6400, "source_draw_cursor": 200, "source_schedule_bound": True, "status": "verified", "step": 25} else "fail", resume_audit)

    try:
        load = json.loads((run / "output/load-audit.json").read_text(encoding="utf-8"))
        fields = {"dtype": load.get("dtype"), "attachments": load.get("adapter", {}).get("attachments"), "lora_rank": load.get("adapter", {}).get("lora_rank"), "lora_alpha": load.get("adapter", {}).get("lora_alpha"), "trainable_parameters": load.get("adapter", {}).get("trainable_parameters"), "allocator_fraction": load.get("cuda_allocator", {}).get("cuda_allocator_fraction"), "requested_max_seq_length": load.get("model_load", {}).get("requested_max_seq_length"), "observed_capacity_tokens": load.get("model_load", {}).get("observed_capacity_tokens"), "tokenizer_status": load.get("tokenizer_contract", {}).get("status")}
        wanted = {"dtype": "torch.bfloat16", "attachments": 294, "lora_rank": 16, "lora_alpha": 16, "trainable_parameters": 25116672, "allocator_fraction": 0.75, "requested_max_seq_length": 4096, "observed_capacity_tokens": 4096, "tokenizer_status": "verified"}
        load_ok = compare_mapping(checks, "runtime.load_audit", fields, wanted)
    except Exception as error:
        add_check(checks, "runtime.load_audit", "fail", repr(error)[:300])
        load_ok = False
    return {"status": "pass" if recipe_ok and recipe_data_ok and entry_ok and load_ok else "fail", "recipe_sha256": recipe_sha, "recipe_bytes": recipe_size, "entry_status": entry.get("status"), "load_fields": fields if "fields" in locals() else {}}


def generation_hash(ids: Any) -> str | None:
    if not isinstance(ids, list):
        return None
    try:
        return hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    except (TypeError, ValueError, UnicodeEncodeError):
        return None


def verify_records(checks: list[dict[str, Any]], run: Path, sequence_path: Path, rows_path: Path) -> dict[str, Any]:
    out = run / "output"
    try:
        all_generations = parse_jsonl(out / "generation-records.jsonl")
        all_rewards = parse_jsonl(out / "reward-records.jsonl")
        all_gradients = parse_jsonl(out / "gradient-records.jsonl")
    except Exception as error:
        add_check(checks, "records.parse", "fail", repr(error)[:300])
        return {"status": "fail"}
    generations = [row for row in all_generations if isinstance(row.get("global_step"), int) and 25 <= row["global_step"] <= 49]
    rewards = all_rewards[: len(generations)]
    step_counts = collections.Counter(row.get("global_step") for row in generations)
    expected_steps = {step: 32 for step in range(25, 50)}
    add_check(checks, "records.generation.window", "pass" if len(generations) == 800 and dict(step_counts) == expected_steps else "fail", {"records": len(generations), "steps": dict(step_counts), "all_records": len(all_generations)})
    generation_errors: list[str] = []
    terminal_counts: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(generations):
        local = index % 32
        geo = row.get("group_geometry") if isinstance(row.get("group_geometry"), dict) else {}
        expected_geo = {"candidate_count": 4, "generation_call_count": 4, "generation_group_count": 8, "generation_groups_per_call": 2, "generation_row_count": 32, "group_index": local // 4, "group_row_index": local % 4, "group_start": (local // 4) * 4, "group_end": (local // 4) * 4 + 4, "generation_call_index": (local // 4) // 2}
        for key, wanted in expected_geo.items():
            if geo.get(key) != wanted: generation_errors.append(f"row{index}:geometry.{key}")
        step = row.get("global_step")
        if row.get("schema_version") != GENERATION_SCHEMA or row.get("global_step_before_update") != step or row.get("update", {}).get("global_step") != step: generation_errors.append(f"row{index}:step_schema")
        # The c continuation was restored at global step 25.  Its first
        # post-restore generation therefore has microstep 0, then advances by
        # eight for each subsequent update.
        if row.get("trl_microstep") != (step - 25) * 8: generation_errors.append(f"row{index}:microstep")
        if row.get("source_schedule_sha256") != EXPECTED["schedule"]: generation_errors.append(f"row{index}:schedule")
        ids = row.get("generated_ids")
        if not isinstance(ids, list) or any(type(token) is not int or not 0 <= token < 130560 for token in ids): generation_errors.append(f"row{index}:ids")
        if not isinstance(ids, list) or row.get("generated_token_count") != len(ids) or not isinstance(row.get("generated_token_count"), int) or not 0 <= row["generated_token_count"] <= 192: generation_errors.append(f"row{index}:token_count")
        if row.get("generated_ids_sha256") != generation_hash(ids) or not isinstance(row.get("generated_ids_sha256"), str) or len(row["generated_ids_sha256"]) != 64 or any(c not in HEX64 for c in row["generated_ids_sha256"]): generation_errors.append(f"row{index}:ids_hash")
        if not isinstance(row.get("prompt_ids_sha256"), str) or len(row["prompt_ids_sha256"]) != 64: generation_errors.append(f"row{index}:prompt_hash")
        if not isinstance(row.get("padded_after_terminal"), int) or row["padded_after_terminal"] < 0: generation_errors.append(f"row{index}:padding")
        if not isinstance(row.get("elapsed_sec"), (int, float)) or not math.isfinite(row["elapsed_sec"]) or row["elapsed_sec"] < 0: generation_errors.append(f"row{index}:elapsed")
        terminal_counts[str(row.get("terminal_reason"))] += 1
    add_check(checks, "records.generation.schema_geometry_join", "pass" if not generation_errors else "fail", generation_errors[:12] if generation_errors else {"calls_per_update": 4, "groups_per_update": 8})
    add_check(checks, "records.generation.termination", "pass", dict(terminal_counts))

    reward_errors: list[str] = []
    reward_ids = [str(row.get("id")) for row in rewards]
    candidate_counts = collections.Counter(reward_ids)
    reward_families: collections.Counter[str] = collections.Counter()
    reward_operations: collections.Counter[str] = collections.Counter()
    expected_operations: collections.Counter[str] = collections.Counter()
    reward_failures: collections.Counter[str] = collections.Counter()
    reward_caps: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(rewards):
        if row.get("output_ids_sha256") != generations[index].get("generated_ids_sha256"): reward_errors.append(f"row{index}:output_hash_join")
        if row.get("generated_tokens") != generations[index].get("generated_token_count"): reward_errors.append(f"row{index}:token_count_join")
        if not isinstance(row.get("reward"), (int, float)) or not math.isfinite(row["reward"]) or not isinstance(row.get("line_f1"), (int, float)) or not math.isfinite(row["line_f1"]): reward_errors.append(f"row{index}:nonfinite")
        reward_families[str(row.get("family"))] += 1
        reward_operations[str(row.get("operation"))] += 1
        expected_operations[str(row.get("expected_operation"))] += 1
        if row.get("failure") is not None: reward_failures[str(row.get("failure"))] += 1
        reward_caps["cap" if row.get("cap_hit") else "noncap"] += 1
        if index and row.get("id") != rewards[index - 1].get("id") and index % 4 != 0: reward_errors.append(f"row{index}:candidate_contiguity")
    add_check(checks, "records.reward.window", "pass" if len(rewards) == 800 else "fail", {"records": len(rewards), "all_records": len(all_rewards)})
    add_check(checks, "records.reward.candidate_groups", "pass" if len(candidate_counts) == 200 and set(candidate_counts.values()) == {4} else "fail", {"unique_sources": len(candidate_counts), "histogram": dict(collections.Counter(candidate_counts.values()))})
    add_check(checks, "records.reward.generation_join", "pass" if not reward_errors else "fail", reward_errors[:12] if reward_errors else {"rows": len(rewards)})

    try:
        sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
        sequence_ids = [str(value) for value in sequence["row_ids"]]
        sequence_sha = sha256_file(sequence_path)[1]
    except Exception as error:
        add_check(checks, "records.source_sequence", "fail", repr(error)[:300])
        sequence_ids, sequence_sha = [], None
    source_ids = [str(row.get("id")) for row in rewards[::4]]
    expected_source_ids = sequence_ids[200:400]
    source_join = source_ids == expected_source_ids and len(source_ids) == 200 and sequence_sha == "2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132"
    add_check(checks, "records.source_cursor_200_400", "pass" if source_join else "fail", {"source_rows": len(source_ids), "cursor_start": 200, "cursor_end": 400, "sequence_sha256": sequence_sha})

    source_meta: dict[str, tuple[Any, Any]] = {}
    target = set(source_ids)
    try:
        with rows_path.open("rb") as stream:
            for raw in stream:
                value = json.loads(raw)
                row_id = str(value.get("id"))
                if row_id in target: source_meta[row_id] = (value.get("family"), value.get("target_operation"))
    except Exception as error:
        add_check(checks, "records.source_lookup", "fail", repr(error)[:300])
    family_errors = []
    for source_index, source_id in enumerate(source_ids):
        expected = source_meta.get(source_id)
        for row in rewards[source_index * 4 : source_index * 4 + 4]:
            if expected is None or (row.get("family"), row.get("expected_operation")) != expected:
                family_errors.append(f"source{source_index}:{source_id}")
    add_check(checks, "records.source_family_operation", "pass" if len(source_meta) == 200 and not family_errors else "fail", {"mapped": len(source_meta), "errors": family_errors[:8]})

    gradients = [row for row in all_gradients if isinstance(row.get("step"), int) and 25 <= row["step"] <= 49]
    gradient_errors: list[str] = []
    zero_updates: list[int] = []
    for index, row in enumerate(gradients):
        step = row.get("step")
        if row.get("schema_version") != GRADIENT_SCHEMA or row.get("global_step") != step or not (25 <= step <= 49): gradient_errors.append(f"gradient{index}:step_schema")
        if row.get("finite") is not True or row.get("nonfinite") is not False: gradient_errors.append(f"gradient{index}:finite")
        if row.get("trainable_tensor_count") != 588 or row.get("grad_present_count") != 588 or row.get("nonzero_tensor_count") not in (0, 588): gradient_errors.append(f"gradient{index}:tensor_presence")
        if row.get("nonzero_tensor_count") == 0: zero_updates.append(step + 1)
        if not isinstance(row.get("norm"), (int, float)) or not math.isfinite(row["norm"]) or row["norm"] < 0: gradient_errors.append(f"gradient{index}:norm")
    finite_updates = [row.get("step") + 1 for row in gradients if row.get("finite") is True]
    gradient_ok = len(gradients) == 25 and finite_updates == list(range(26, 51)) and not gradient_errors
    add_check(checks, "records.gradients.finite_updates_26_50", "pass" if gradient_ok else "fail", {"records": len(gradients), "updates": finite_updates, "zero_gradient_updates": zero_updates, "errors": gradient_errors[:8]})

    trainer = json.loads((run / "output/checkpoint-50/trainer_state.json").read_text(encoding="utf-8"))
    history = {row.get("step"): row for row in trainer.get("log_history", []) if isinstance(row, dict) and isinstance(row.get("step"), int)}
    zero_variance_updates = [step for step in range(26, 51) if history.get(step, {}).get("reward_std") == 0.0 and history.get(step, {}).get("frac_reward_zero_std") == 1.0 and history.get(step, {}).get("grad_norm") == 0.0]
    zero_alignment = zero_variance_updates == zero_updates
    add_check(checks, "records.zero_variance_alignment", "pass" if zero_alignment else "fail", {"zero_gradient_updates": zero_updates, "zero_reward_variance_updates": zero_variance_updates})
    add_check(checks, "records.trainer_state_step", "pass" if trainer.get("global_step") == 50 else "fail", {"global_step": trainer.get("global_step"), "max_steps": trainer.get("max_steps")})
    return {
        "status": "pass" if len(generations) == 800 and len(rewards) == 800 and not generation_errors and not reward_errors and source_join and len(source_meta) == 200 and not family_errors and gradient_ok and zero_alignment else "fail",
        "generation_records_window": len(generations), "reward_records_window": len(rewards), "all_generation_records": len(all_generations), "all_reward_records": len(all_rewards),
        "source_cursor": {"start": 200, "end": 400, "rows": len(source_ids), "exact": source_join},
        "terminal_reasons": dict(terminal_counts), "reward_family_counts": dict(reward_families), "reward_operation_counts": dict(reward_operations), "expected_operation_counts": dict(expected_operations), "reward_failure_counts": dict(reward_failures), "reward_cap_counts": dict(reward_caps),
        "generation_reward_join": not generation_errors and not reward_errors, "source_family_errors": family_errors[:8], "gradient_updates": finite_updates, "zero_gradient_updates": zero_updates, "zero_variance_updates": zero_variance_updates,
    }


def verify_development(checks: list[dict[str, Any]], run: Path, panel_path: Path) -> dict[str, Any]:
    try:
        panel = parse_jsonl(panel_path)
        panel_sha = sha256_file(panel_path)[1]
        cases = json.loads((run / "archive/evaluations/cases-step-50.json").read_text(encoding="utf-8"))
        prior = json.loads((run.parent / "RL-primary-p2-mb4-full5-b/archive/evaluations/cases-step-25.json").read_text(encoding="utf-8"))
    except Exception as error:
        add_check(checks, "development.parse", "fail", repr(error)[:300])
        return {"status": "fail"}
    panel_ids = [str(row.get("id")) for row in panel]
    results = cases.get("results", []) if isinstance(cases, dict) else []
    result_ids = [str(row.get("id")) for row in results if isinstance(row, dict)]
    panel_ok = len(panel) == 75 and len(set(panel_ids)) == 75 and panel_sha == EXPECTED["panel"]
    identity_ok = cases.get("status") == "complete" and cases.get("step") == 50 and len(results) == 75 and len(set(result_ids)) == 75 and result_ids == panel_ids and cases.get("summary", {}).get("panel_sha256") == EXPECTED["panel"]
    add_check(checks, "development.panel_identity", "pass" if panel_ok else "fail", {"rows": len(panel), "sha256": panel_sha})
    add_check(checks, "development.cases_identity", "pass" if identity_ok else "fail", {"status": cases.get("status"), "step": cases.get("step"), "rows": len(results), "ids_exact_panel": result_ids == panel_ids})
    summary = cases.get("summary", {})
    denom = summary.get("denominators", {})
    denom_ok = all(denom.get(k) == v for k, v in {"cases": 75, "edits": 43, "strict_noop": 32}.items())
    add_check(checks, "development.denominators", "pass" if denom_ok else "fail", {k: denom.get(k) for k in ("cases", "edits", "strict_noop")})
    try:
        sys.path.insert(0, str(PROTOCOL_DEFAULT))
        from sepalith.campaign_protocol import PromptContext, parse_output, valid_generation_tokens  # type: ignore
    except Exception as error:
        add_check(checks, "development.pinned_parser_import", "fail", repr(error)[:300])
        return {"status": "fail"}
    panel_by_id = {str(row.get("id")): row for row in panel}
    parser_errors: list[str] = []
    counts = collections.Counter()
    family_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    long_case: dict[str, Any] | None = None
    for row in results:
        row_id = str(row.get("id"))
        case = panel_by_id.get(row_id)
        if case is None:
            parser_errors.append(f"{row_id}:missing_panel")
            continue
        try:
            context = PromptContext.from_mapping(case["context"])
            parsed = parse_output(row.get("raw_output", ""), context)
            valid = valid_generation_tokens(row.get("generated_ids", [])) and parsed.status == "accepted"
            predicted_noop = valid and parsed.operation == "no_op"
            actual = list(context.region_old) if predicted_noop else list(parsed.body)
            exact = valid and actual == list(case.get("region_new", []))
            failure = None if valid else (parsed.reason or "noncanonical_or_missing_EOS")
        except Exception:
            parser_errors.append(f"{row_id}:exception")
            continue
        for key, value in {"protocol_valid": valid, "predicted_noop": predicted_noop, "exact_region": exact, "failure": failure}.items():
            if row.get(key) != value: parser_errors.append(f"{row_id}:{key}")
        expected_noop = list(context.region_old) == list(case.get("region_new", []))
        if row.get("expected_noop") != expected_noop: parser_errors.append(f"{row_id}:expected_noop")
        family = str(case.get("family"))
        family_counts[family].update({"cases": 1, "protocol_valid": int(valid), "exact_region": int(exact), "predicted_noop": int(predicted_noop), "cap_hit": int(bool(row.get("cap_hit")))})
        counts.update({"protocol_valid": int(valid), "exact_region": int(exact), "predicted_noop": int(predicted_noop), "suggestion": int(valid and not predicted_noop), "cap_hit": int(bool(row.get("cap_hit"))), "strict_noop_correct": int(expected_noop and predicted_noop), "strict_noop_false_suggestions": int(expected_noop and valid and not predicted_noop), "edit_exact": int(not expected_noop and exact)})
        if row_id == LONG_DEV_ID:
            long_case = {key: row.get(key) for key in ("id", "family", "prompt_tokens", "generated_tokens", "cap_hit", "protocol_valid", "exact_region", "predicted_noop", "failure")}
    expected_counts = {"protocol_valid": 70, "exact_region": 50, "predicted_noop": 27, "suggestion": 43, "cap_hit": 5, "strict_noop_correct": 25, "strict_noop_false_suggestions": 5, "edit_exact": 25}
    stored_counts = summary.get("counts", {})
    count_ok = all(counts.get(k) == v and stored_counts.get(k) == v for k, v in expected_counts.items())
    add_check(checks, "development.pinned_parser_reclassification", "pass" if not parser_errors else "fail", {"count": len(parser_errors), "errors": parser_errors[:12]})
    add_check(checks, "development.counts", "pass" if count_ok else "fail", {"independent": dict(counts), "stored": {k: stored_counts.get(k) for k in expected_counts}})
    long_ok = long_case is not None and long_case.get("prompt_tokens") == 2619
    add_check(checks, "development.maximal_prompt_retained", "pass" if long_ok else "fail", long_case)

    prior_by_id = {str(row.get("id")): row for row in prior.get("results", []) if isinstance(row, dict)}
    class_fields = ("protocol_valid", "exact_region", "predicted_noop", "cap_hit", "failure")
    changed: list[dict[str, Any]] = []
    for row in results:
        row_id = str(row.get("id"))
        old = prior_by_id.get(row_id)
        if old is None: changed.append({"id": row_id, "kind": "missing_prior"})
        else:
            diff = {key: [old.get(key), row.get(key)] for key in class_fields if old.get(key) != row.get(key)}
            if diff: changed.append({"id": row_id, "family": row.get("family"), "changes": diff})
    expected_change = [{"id": "f43de3e77f2d92ed7b223464", "family": "finish_block", "changes": {"protocol_valid": [False, True], "cap_hit": [True, False], "failure": ["missing_exact_terminal", None]}}]
    add_check(checks, "development.step25_classification_delta", "pass" if changed == expected_change else "fail", {"changed": changed})
    interpretation_ok = "no R semantic-validity" in str(summary.get("interpretation", "")) and "claim" in str(summary.get("interpretation", "")).lower()
    add_check(checks, "development.interpretation", "pass" if interpretation_ok else "fail", summary.get("interpretation"))
    return {"status": "pass" if panel_ok and identity_ok and denom_ok and not parser_errors and count_ok and long_ok and changed == expected_change and interpretation_ok else "fail", "artifact": str(run / "archive/evaluations/cases-step-50.json"), "artifact_sha256": sha256_file(run / "archive/evaluations/cases-step-50.json")[1], "panel_sha256": panel_sha, "case_count": len(results), "step": cases.get("step"), "denominators": {k: denom.get(k) for k in ("cases", "edits", "strict_noop")}, "counts": dict(counts), "family_counts": {key: dict(value) for key, value in sorted(family_counts.items())}, "changed_classifications_from_step25": changed, "long_case": long_case, "result_ids_exact_panel": result_ids == panel_ids}


def run_verifier(run: Path, recipe: Path, sequence: Path, panel: Path, rows: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checkpoint = verify_checkpoint(checks, run)
    runtime = verify_runtime(checks, run, recipe)
    records = verify_records(checks, run, sequence, rows)
    development = verify_development(checks, run, panel)
    hard_fail = any(item["status"] == "fail" for item in checks)
    return {
        "schema_version": "sepalith.rl-08.step50-independent-review.v1",
        "status": "fail_mechanical" if hard_fail else "pass_mechanical_quality_pending",
        "run_root": str(run), "recipe": str(recipe), "sequence": str(sequence), "panel": str(panel), "rows": str(rows),
        "expected": EXPECTED,
        "checkpoint": checkpoint, "runtime": runtime, "records": records, "development": development,
        "checks": checks,
        "unresolved": [
            "This is mechanical artifact evidence through DEV50; no quality promotion or semantic R validity claim is made.",
            "The c continuation remained active beyond the reviewed window; records after global step 49 were excluded from update-26..50 acceptance.",
            "The supplied identity SHA256 is recorded and all projected identity fields match, but the canonical identity serialization is not present as a standalone artifact for independent re-derivation.",
            "The adapter_model.safetensors file was deliberately not hashed; all other checkpoint files listed by the full manifest were streamed and matched in archive and output.",
            "The live 52 MiB telemetry file was not read; no terminal claim is made for the continuing c run.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--recipe", type=Path, default=RECIPE_DEFAULT)
    parser.add_argument("--sequence", type=Path, default=SEQUENCE_DEFAULT)
    parser.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    parser.add_argument("--rows", type=Path, default=ROWS_DEFAULT)
    parser.add_argument("--json-out", type=Path, default=Path(__file__).with_name("RL-08-step50-independent-review.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_verifier(args.run_root, args.recipe, args.sequence, args.panel, args.rows)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checkpoint": result["checkpoint"].get("status"), "runtime": result["runtime"].get("status"), "records": result["records"].get("status"), "development": result["development"].get("status"), "failed_checks": sum(item["status"] == "fail" for item in result["checks"])}, sort_keys=True))
    return 1 if result["status"] == "fail_mechanical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
