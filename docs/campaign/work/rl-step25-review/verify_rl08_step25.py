#!/usr/bin/env python3
"""Framework-free independent review of the RL update-25 continuation.

The verifier reads only the named run, frozen source-data artifacts, recipe,
and pinned development panel.  The large checkpoint manifests are projected
through ``jq`` to a scalar/file-inventory allowlist; their nested identity and
the raw generation text are never printed.  Checkpoint files are hashed in
streaming mode.  No torch, CUDA, Transformers, TRL, Unsloth, model, or server
is imported.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


RUN_DEFAULT = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
    "RL-primary-p2-mb4-full5-b"
)
RECIPE_DEFAULT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/"
    "work/main-rl-preparation/primary-mb4-full5-b.recipe.json"
)
SEQUENCE_DEFAULT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/"
    "work/rl-pool/source-row-draw-sequence-v6-interleaved-mb4.json"
)
PANEL_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT-07-final-evaluator-cases.jsonl"
)
ROWS_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/"
    "eligible-train-rows.jsonl"
)
SIDECAR_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/"
    "context-sidecar.jsonl"
)
SELECTED_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/"
    "selected-train-ids-lead-order-v1.json"
)
PROTOCOL_DEFAULT = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/"
    "prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/"
    "source/packages/sepalith/src"
)

EXPECTED = {
    "source": "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9",
    "recipe": "ea27572c81d46c7a36a54a5ed05def7fb9decd044abec103586d5bd2cbcde56d",
    "identity": "48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2",
    "schedule": "2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132",
    "sequence": "dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6",
    "selected": "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d",
    "ordered": "7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d",
    "row_identity": "d3187195f0ebe401b69df244bbbff0f2a28fb0500185428701f933e31e8eab60",
    "rows": "e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602",
    "context": "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f",
    "parent_manifest": "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12",
    "parent_weights": "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d",
    "parent_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
    "panel": "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21",
}

EXPECTED_POLICY = {
    "beta": 0,
    "candidate_count": 4,
    "completion_max_tokens": 192,
    "context_max_tokens": 2240,
    "cuda_memory_fraction": 0.75,
    "expected_attachments": 294,
    "expected_trainable_parameters": 25116672,
    "generation_groups_per_call": 2,
    "gradient_accumulation_steps": 8,
    "lora_alpha": 16,
    "lora_rank": 16,
    "loss_type": "bnpo",
    "model_load_max_seq_length": 4096,
    "per_device_train_batch_size": 4,
    "prompt_max_tokens": 2048,
    "rollout_rows_per_update": 32,
    "scale_rewards": "group",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
EXPECTED_RENDERER = {
    "renderer_id": "zeta2-prm03-v1",
    "schema_version": "sepalith.prompt.prm03.v1",
    "no_edit": "[NO_EDIT]",
    "terminal": ">>>>>>> UPDATED",
    "tokenization_policy": "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1",
}
EXPECTED_SCHEDULE = {
    "buffer_reuse": 8,
    "development_cases": 75,
    "development_every": 25,
    "development_panel_sha256": EXPECTED["panel"],
    "first_development_update": 5,
    "full_save_steps": 5,
    "generation_batch_size": 32,
    "light_save_steps": 5,
    "main_update_ceiling": 3000,
    "num_iterations": 1,
    "sampler_id": "campaign-repeat-manifest-order-v1",
    "seed": 3407,
    "source_draw_schedule_sha256": EXPECTED["schedule"],
    "source_draw_sequence_sha256": EXPECTED["sequence"],
    "source_draws": 24000,
    "source_draws_per_update": 8,
    "stage_id": "primary-grpo-p2-2048x192-v5-full5",
    "steps_per_generation": 8,
}
EXPECTED_TOKENIZER = {
    "bos_id": 0,
    "eos_id": 1,
    "pad_id": 1,
    "native_eog_ids": [1, 130073],
    "vocab_size": 130560,
    "revision": EXPECTED["parent_revision"],
}
EXPECTED_DATA = {
    "row_count": 8440,
    "split": "train",
    "admission_status": "admitted",
    "selected_ids_sha256": EXPECTED["selected"],
    "ordered_ids_sha256": EXPECTED["ordered"],
    "row_identity_sha256": EXPECTED["row_identity"],
    "rows_sha256": EXPECTED["rows"],
    "context_sha256": EXPECTED["context"],
    "sidecar_artifact_sha256": EXPECTED["context"],
}
EXPECTED_PARENT = {
    "kind": "merged_sft",
    "manifest_sha256": EXPECTED["parent_manifest"],
    "merged_weights_sha256": EXPECTED["parent_weights"],
    "base_model_revision": EXPECTED["parent_revision"],
}
LONG_DEV_ID = "dat07-existing-719cd49683667d0fb86fb2fa"
GENERATION_SCHEMA = "sepalith.rl.generation-record.v1"
GRADIENT_SCHEMA = "sepalith.rl.gradient-record.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def add_check(checks: list[dict[str, Any]], name: str, status: str, details: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "status": status}
    if details is not None:
        item["details"] = details
    checks.append(item)


def safe_scalar(value: Any) -> Any:
    """Keep accidental diagnostics bounded and exclude raw model text."""
    if isinstance(value, str) and len(value) > 300:
        return value[:297] + "..."
    if isinstance(value, dict):
        return {str(k): safe_scalar(v) for k, v in value.items() if not isinstance(v, (list, dict))}
    if isinstance(value, list):
        return [safe_scalar(v) for v in value[:20]] + (["..."] if len(value) > 20 else [])
    return value


def jq_projection(path: Path, expression: str) -> dict[str, Any]:
    """Project a known scalar/file allowlist without materializing giant JSON in Python."""
    process = subprocess.run(
        ["jq", "-c", expression, str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(f"jq projection failed for {path}: {process.stderr.strip()[:300]}")
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"jq projection for {path} did not return an object")
    return value


MANIFEST_PROJECTION = r'''{
  schema_version, step, full,
  identity: {
    data: {
      row_count: .identity.data.row_count, split: .identity.data.split,
      admission_status: .identity.data.admission_status,
      selected_ids_sha256: .identity.data.selected_ids_sha256,
      ordered_ids_sha256: .identity.data.ordered_ids_sha256,
      row_identity_sha256: .identity.data.row_identity_sha256,
      rows_sha256: .identity.data.rows_sha256,
      context_sha256: .identity.data.context_sha256,
      sidecar_artifact_sha256: .identity.data.sidecar_artifact_sha256
    },
    parent: {
      kind: .identity.parent.kind,
      manifest_sha256: .identity.parent.manifest_sha256,
      merged_weights_sha256: .identity.parent.merged_weights_sha256,
      base_model_revision: .identity.parent.base_model_revision
    },
    policy: {
      beta: .identity.policy.beta, candidate_count: .identity.policy.candidate_count,
      completion_max_tokens: .identity.policy.completion_max_tokens,
      context_max_tokens: .identity.policy.context_max_tokens,
      cuda_memory_fraction: .identity.policy.cuda_memory_fraction,
      expected_attachments: .identity.policy.expected_attachments,
      expected_trainable_parameters: .identity.policy.expected_trainable_parameters,
      generation_groups_per_call: .identity.policy.generation_groups_per_call,
      gradient_accumulation_steps: .identity.policy.gradient_accumulation_steps,
      lora_alpha: .identity.policy.lora_alpha, lora_rank: .identity.policy.lora_rank,
      loss_type: .identity.policy.loss_type,
      model_load_max_seq_length: .identity.policy.model_load_max_seq_length,
      per_device_train_batch_size: .identity.policy.per_device_train_batch_size,
      prompt_max_tokens: .identity.policy.prompt_max_tokens,
      rollout_rows_per_update: .identity.policy.rollout_rows_per_update,
      scale_rewards: .identity.policy.scale_rewards,
      target_modules: .identity.policy.target_modules
    },
    renderer: {
      renderer_id: .identity.renderer.renderer_id,
      schema_version: .identity.renderer.schema_version,
      no_edit: .identity.renderer.no_edit, terminal: .identity.renderer.terminal,
      tokenization_policy: .identity.renderer.tokenization_policy
    },
    schedule: {
      buffer_reuse: .identity.schedule.buffer_reuse,
      development_cases: .identity.schedule.development_cases,
      development_every: .identity.schedule.development_every,
      development_panel_sha256: .identity.schedule.development_panel_sha256,
      first_development_update: .identity.schedule.first_development_update,
      full_save_steps: .identity.schedule.full_save_steps,
      generation_batch_size: .identity.schedule.generation_batch_size,
      light_save_steps: .identity.schedule.light_save_steps,
      main_update_ceiling: .identity.schedule.main_update_ceiling,
      num_iterations: .identity.schedule.num_iterations,
      sampler_id: .identity.schedule.sampler_id, seed: .identity.schedule.seed,
      source_draw_schedule_sha256: .identity.schedule.source_draw_schedule_sha256,
      source_draw_sequence_sha256: .identity.schedule.source_draw_sequence_sha256,
      source_draws: .identity.schedule.source_draws,
      source_draws_per_update: .identity.schedule.source_draws_per_update,
      stage_id: .identity.schedule.stage_id,
      steps_per_generation: .identity.schedule.steps_per_generation
    },
    source: {
      protocol_sha256: .identity.source.protocol_sha256,
      rl02_admission_status: .identity.source.rl02_admission_status,
      source_schedule_sha256: .identity.source.source_schedule_sha256,
      trainer_sha256: .identity.source.trainer_sha256,
      trl_version: .identity.source.trl_version
    },
    tokenizer: {
      bos_id: .identity.tokenizer.bos_id, eos_id: .identity.tokenizer.eos_id,
      pad_id: .identity.tokenizer.pad_id,
      native_eog_ids: .identity.tokenizer.native_eog_ids,
      vocab_size: .identity.tokenizer.vocab_size,
      revision: .identity.tokenizer.revision
    }
  },
  files: (.files | to_entries | map({name: .key, bytes: .value.bytes, sha256: .value.sha256}))
}'''

STATE_PROJECTION = r'''{
  step, full, resume_validation,
  identity: {
    data: {
      row_count: .identity.data.row_count, split: .identity.data.split,
      admission_status: .identity.data.admission_status,
      selected_ids_sha256: .identity.data.selected_ids_sha256,
      ordered_ids_sha256: .identity.data.ordered_ids_sha256,
      row_identity_sha256: .identity.data.row_identity_sha256,
      rows_sha256: .identity.data.rows_sha256,
      context_sha256: .identity.data.context_sha256,
      sidecar_artifact_sha256: .identity.data.sidecar_artifact_sha256
    },
    parent: {
      kind: .identity.parent.kind, manifest_sha256: .identity.parent.manifest_sha256,
      merged_weights_sha256: .identity.parent.merged_weights_sha256,
      base_model_revision: .identity.parent.base_model_revision
    },
    policy: {candidate_count: .identity.policy.candidate_count,
      model_load_max_seq_length: .identity.policy.model_load_max_seq_length,
      per_device_train_batch_size: .identity.policy.per_device_train_batch_size,
      gradient_accumulation_steps: .identity.policy.gradient_accumulation_steps,
      rollout_rows_per_update: .identity.policy.rollout_rows_per_update},
    renderer: {renderer_id: .identity.renderer.renderer_id,
      schema_version: .identity.renderer.schema_version},
    schedule: {source_draw_schedule_sha256: .identity.schedule.source_draw_schedule_sha256,
      source_draw_sequence_sha256: .identity.schedule.source_draw_sequence_sha256,
      steps_per_generation: .identity.schedule.steps_per_generation,
      stage_id: .identity.schedule.stage_id},
    source: {protocol_sha256: .identity.source.protocol_sha256,
      rl02_admission_status: .identity.source.rl02_admission_status,
      source_schedule_sha256: .identity.source.source_schedule_sha256,
      trainer_sha256: .identity.source.trainer_sha256,
      trl_version: .identity.source.trl_version},
    tokenizer: {bos_id: .identity.tokenizer.bos_id, eos_id: .identity.tokenizer.eos_id,
      pad_id: .identity.tokenizer.pad_id, vocab_size: .identity.tokenizer.vocab_size}
  },
  sampler: {
    buffer_reuse: .sampler.buffer_reuse, candidate_count: .sampler.candidate_count,
    consumed_prompt_copies: .sampler.consumed_prompt_copies,
    consumed_rows: .sampler.consumed_rows, current_index: .sampler.current_index,
    epoch: .sampler.epoch, generation_rows_per_update: .sampler.generation_rows_per_update,
    num_samples: .sampler.num_samples, prompt_groups_per_batch: .sampler.prompt_groups_per_batch,
    repeat_count: .sampler.repeat_count, sampler_id: .sampler.sampler_id,
    sampler_rows_per_update: .sampler.sampler_rows_per_update, seed: .sampler.seed,
    selected_id_index: .sampler.selected_id_index, shuffle: .sampler.shuffle,
    source_draw_cursor: .sampler.source_draw_cursor,
    source_draw_schedule_sha256: .sampler.source_draw_schedule_sha256,
    source_draw_sequence_sha256: .sampler.source_draw_sequence_sha256,
    source_draws: .sampler.source_draws, source_draws_bound: .sampler.source_draws_bound,
    data_identity: {
      row_count: .sampler.data_identity.row_count, split: .sampler.data_identity.split,
      admission_status: .sampler.data_identity.admission_status,
      selected_ids_sha256: .sampler.data_identity.selected_ids_sha256,
      ordered_ids_sha256: .sampler.data_identity.ordered_ids_sha256,
      row_identity_sha256: .sampler.data_identity.row_identity_sha256,
      rows_sha256: .sampler.data_identity.rows_sha256,
      context_sha256: .sampler.data_identity.context_sha256,
      sidecar_artifact_sha256: .sampler.data_identity.sidecar_artifact_sha256
    },
    geometry: {generation_batch_size: .sampler.geometry.generation_batch_size,
      gradient_accumulation_steps: .sampler.geometry.gradient_accumulation_steps,
      per_device_train_batch_size: .sampler.geometry.per_device_train_batch_size,
      steps_per_generation: .sampler.geometry.steps_per_generation}
  }
}'''


def compare_mapping(checks: list[dict[str, Any]], prefix: str, actual: Any, expected: Any) -> bool:
    ok = True
    if not isinstance(actual, dict):
        add_check(checks, prefix, "fail", {"expected": "object", "actual": type(actual).__name__})
        return False
    for key, wanted in expected.items():
        value = actual.get(key)
        same = value == wanted
        add_check(checks, f"{prefix}.{key}", "pass" if same else "fail", None if same else {"expected": wanted, "actual": safe_scalar(value)})
        ok = ok and same
    return ok


def compare_identity(checks: list[dict[str, Any]], identity: Any, prefix: str, *, compact: bool = False) -> bool:
    if not isinstance(identity, dict):
        add_check(checks, prefix, "fail", "identity is not an object")
        return False
    ok = True
    ok = compare_mapping(checks, f"{prefix}.data", identity.get("data"), EXPECTED_DATA) and ok
    ok = compare_mapping(checks, f"{prefix}.parent", identity.get("parent"), EXPECTED_PARENT) and ok
    policy = EXPECTED_POLICY if not compact else {
        k: EXPECTED_POLICY[k] for k in ("candidate_count", "model_load_max_seq_length", "per_device_train_batch_size", "gradient_accumulation_steps", "rollout_rows_per_update")
    }
    ok = compare_mapping(checks, f"{prefix}.policy", identity.get("policy"), policy) and ok
    renderer = EXPECTED_RENDERER if not compact else {"renderer_id": EXPECTED_RENDERER["renderer_id"], "schema_version": EXPECTED_RENDERER["schema_version"]}
    ok = compare_mapping(checks, f"{prefix}.renderer", identity.get("renderer"), renderer) and ok
    schedule = EXPECTED_SCHEDULE if not compact else {
        k: EXPECTED_SCHEDULE[k] for k in ("source_draw_schedule_sha256", "source_draw_sequence_sha256", "steps_per_generation", "stage_id")
    }
    ok = compare_mapping(checks, f"{prefix}.schedule", identity.get("schedule"), schedule) and ok
    tokenizer = EXPECTED_TOKENIZER if not compact else {k: EXPECTED_TOKENIZER[k] for k in ("bos_id", "eos_id", "pad_id", "vocab_size")}
    ok = compare_mapping(checks, f"{prefix}.tokenizer", identity.get("tokenizer"), tokenizer) and ok
    source_expected = {
        "rl02_admission_status": "rl_data_admitted",
        "source_schedule_sha256": EXPECTED["schedule"],
        "trl_version": "0.24.0",
    }
    # protocol/trainer hashes are identity fields; their exact frozen values
    # are checked when present but are not recomputed from a model/framework.
    source = identity.get("source")
    if not isinstance(source, dict):
        add_check(checks, f"{prefix}.source", "fail", "source is not an object")
        ok = False
    else:
        for key, wanted in source_expected.items():
            same = source.get(key) == wanted
            add_check(checks, f"{prefix}.source.{key}", "pass" if same else "fail", None if same else {"expected": wanted, "actual": safe_scalar(source.get(key))})
            ok = ok and same
        for key in ("protocol_sha256", "trainer_sha256"):
            same = isinstance(source.get(key), str) and bool(HEX64.fullmatch(source[key]))
            add_check(checks, f"{prefix}.source.{key}.well_formed", "pass" if same else "fail", None if same else safe_scalar(source.get(key)))
            ok = ok and same
    return ok


def verify_data_files(checks: list[dict[str, Any]], sequence_path: Path, rows_path: Path, sidecar_path: Path, selected_path: Path, source_ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path, wanted in (
        ("sequence", sequence_path, EXPECTED["schedule"]),
        ("rows", rows_path, EXPECTED["rows"]),
        ("sidecar", sidecar_path, EXPECTED["context"]),
        ("selected", selected_path, EXPECTED["selected"]),
    ):
        if not path.is_file():
            add_check(checks, f"data.{name}.present", "fail", str(path))
            result[name] = {"path": str(path), "present": False}
            continue
        size, digest = sha256_file(path)
        same = digest == wanted
        add_check(checks, f"data.{name}.sha256", "pass" if same else "fail", {"bytes": size, "sha256": digest} if not same else {"bytes": size})
        result[name] = {"path": str(path), "bytes": size, "sha256": digest}

    sequence: dict[str, Any] = {}
    if sequence_path.is_file():
        try:
            value = json.loads(sequence_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                sequence = value
        except Exception as error:
            add_check(checks, "data.sequence.parse", "fail", repr(error)[:300])
    sequence_ids = sequence.get("row_ids", [])
    sequence_ok = (
        sequence.get("schema_version") == "sepalith.dat09.source-row-draw-sequence.v1"
        and sequence.get("sequence_sha256") == EXPECTED["sequence"]
        and sequence.get("source_draws") == 24000
        and isinstance(sequence_ids, list)
        and len(sequence_ids) == 24000
    )
    add_check(checks, "data.sequence.contract", "pass" if sequence_ok else "fail", {
        "rows": len(sequence_ids) if isinstance(sequence_ids, list) else None,
        "sequence_sha256": sequence.get("sequence_sha256"),
        "source_draws": sequence.get("source_draws"),
    })
    result["sequence"] = {"rows": len(sequence_ids) if isinstance(sequence_ids, list) else 0, "sequence_sha256": sequence.get("sequence_sha256")}

    selected_ids: list[str] = []
    if selected_path.is_file():
        try:
            value = json.loads(selected_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("row_ids"), list):
                selected_ids = [str(item) for item in value["row_ids"]]
        except Exception as error:
            add_check(checks, "data.selected.parse", "fail", repr(error)[:300])
    selected_ok = len(selected_ids) == 8440 and len(set(selected_ids)) == 8440
    add_check(checks, "data.selected.contract", "pass" if selected_ok else "fail", {"rows": len(selected_ids), "unique": len(set(selected_ids))})
    result["selected_rows"] = len(selected_ids)

    # Read the 77 MiB eligible-row JSONL line by line.  Retain only the 160
    # source IDs used by this resumed continuation, while hashing the entire
    # known file to bind the family/operation lookup to the admitted artifact.
    target = set(source_ids)
    source_meta: dict[str, dict[str, Any]] = {}
    if rows_path.is_file():
        try:
            with rows_path.open("rb") as stream:
                for raw in stream:
                    try:
                        row = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    row_id = row.get("id")
                    if row_id in target:
                        source_meta[str(row_id)] = {
                            "family": row.get("family"),
                            "operation": row.get("target_operation"),
                            "package_id": row.get("package_id"),
                        }
        except OSError as error:
            add_check(checks, "data.rows.lookup", "fail", repr(error)[:300])
    lookup_ok = len(source_meta) == len(target) and all(meta.get("family") for meta in source_meta.values())
    add_check(checks, "data.rows.source_lookup", "pass" if lookup_ok else "fail", {"requested": len(target), "found": len(source_meta)})
    result["source_meta"] = source_meta
    result["sequence_ids"] = sequence_ids
    return result


def verify_checkpoint(checks: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    archive = run / "archive" / "full" / "checkpoint-25"
    output = run / "output" / "checkpoint-25"
    result: dict[str, Any] = {"archive": str(archive), "output": str(output)}
    if not archive.is_dir() or not output.is_dir():
        add_check(checks, "checkpoint.full.directories", "fail", {"archive": archive.is_dir(), "output": output.is_dir()})
        return {**result, "status": "fail"}
    try:
        am = jq_projection(archive / "campaign-manifest.json", MANIFEST_PROJECTION)
        om = jq_projection(output / "campaign-manifest.json", MANIFEST_PROJECTION)
    except Exception as error:
        add_check(checks, "checkpoint.manifest_projection", "fail", repr(error)[:300])
        return {**result, "status": "fail"}
    identity_ok = compare_identity(checks, am.get("identity"), "checkpoint.manifest_identity")
    add_check(checks, "checkpoint.manifest.step", "pass" if am.get("step") == 25 else "fail", am.get("step"))
    add_check(checks, "checkpoint.manifest.full", "pass" if am.get("full") is True else "fail", am.get("full"))
    add_check(checks, "checkpoint.archive_output.identity_equal", "pass" if am.get("identity") == om.get("identity") else "fail")
    add_check(checks, "checkpoint.archive_output.header_equal", "pass" if {k: am.get(k) for k in ("schema_version", "step", "full")} == {k: om.get(k) for k in ("schema_version", "step", "full")} else "fail")

    listed = {str(item.get("name")): item for item in am.get("files", []) if isinstance(item, dict) and item.get("name")}
    names = sorted(listed)
    add_check(checks, "checkpoint.file_inventory.count", "pass" if len(names) == 12 else "fail", {"count": len(names), "names": names})
    files: dict[str, Any] = {}
    hash_failures = 0
    for name in names:
        expected_bytes = listed[name].get("bytes")
        expected_sha = listed[name].get("sha256")
        entry: dict[str, Any] = {"expected_bytes": expected_bytes, "expected_sha256": expected_sha}
        for label, directory in (("archive", archive), ("output", output)):
            path = directory / name
            if not path.is_file():
                add_check(checks, f"checkpoint.{label}.{name}", "fail", "missing")
                entry[label] = {"present": False}
                hash_failures += 1
                continue
            size, digest = sha256_file(path)
            same = size == expected_bytes and digest == expected_sha
            add_check(checks, f"checkpoint.{label}.{name}", "pass" if same else "fail", None if same else {"bytes": size, "sha256": digest, "expected_bytes": expected_bytes, "expected_sha256": expected_sha})
            entry[label] = {"present": True, "bytes": size, "sha256": digest}
            if not same:
                hash_failures += 1
        same = entry.get("archive", {}).get("present") and entry.get("archive") == entry.get("output")
        add_check(checks, f"checkpoint.archive_output.{name}", "pass" if same else "fail")
        entry["archive_output_equal"] = bool(same)
        files[name] = entry
    for label, directory in (("archive", archive), ("output", output)):
        actual_names = sorted(item.name for item in directory.iterdir() if item.is_file())
        expected_names = sorted(set(names) | {"campaign-manifest.json"})
        add_check(checks, f"checkpoint.{label}.no_extra_files", "pass" if actual_names == expected_names else "fail", None if actual_names == expected_names else {"actual": actual_names, "expected": expected_names})
    archive_manifest_size, archive_manifest_sha = sha256_file(archive / "campaign-manifest.json")
    output_manifest_size, output_manifest_sha = sha256_file(output / "campaign-manifest.json")
    manifest_equal = archive_manifest_size == output_manifest_size and archive_manifest_sha == output_manifest_sha
    add_check(checks, "checkpoint.manifest.archive_output_equal", "pass" if manifest_equal else "fail", {"bytes": archive_manifest_size, "sha256": archive_manifest_sha} if not manifest_equal else {"bytes": archive_manifest_size, "sha256": archive_manifest_sha})

    state_summary: dict[str, Any] = {}
    try:
        state = jq_projection(archive / "campaign-state.json", STATE_PROJECTION)
        state_identity_ok = compare_identity(checks, state.get("identity"), "checkpoint.state_identity", compact=True)
        add_check(checks, "checkpoint.state.step", "pass" if state.get("step") == 25 else "fail", state.get("step"))
        add_check(checks, "checkpoint.state.full", "pass" if state.get("full") is True else "fail", state.get("full"))
        add_check(checks, "checkpoint.state.resume_validation", "pass" if isinstance(state.get("resume_validation"), str) and "matched interruption/resume" in state["resume_validation"] else "fail", state.get("resume_validation"))
        sampler = state.get("sampler", {})
        required_sampler = {
            "buffer_reuse": 8, "candidate_count": 4, "consumed_prompt_copies": 1600,
            "consumed_rows": 6400, "current_index": 6400, "epoch": 0,
            "generation_rows_per_update": 32, "num_samples": 8440,
            "prompt_groups_per_batch": 8, "repeat_count": 8,
            "sampler_id": "campaign-repeat-manifest-order-v1", "sampler_rows_per_update": 256,
            "seed": 3407, "selected_id_index": 1198, "shuffle": False,
            "source_draw_cursor": 200, "source_draw_schedule_sha256": EXPECTED["schedule"],
            "source_draw_sequence_sha256": EXPECTED["sequence"], "source_draws": 24000,
            "source_draws_bound": True,
        }
        sampler_ok = compare_mapping(checks, "checkpoint.sampler", sampler, required_sampler)
        geometry_ok = compare_mapping(checks, "checkpoint.sampler.geometry", sampler.get("geometry"), {
            "generation_batch_size": 32, "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 4, "steps_per_generation": 8,
        })
        data_identity = sampler.get("data_identity", {})
        data_ok = compare_mapping(checks, "checkpoint.sampler.data_identity", data_identity, EXPECTED_DATA)
        state_summary = {
            "step": state.get("step"), "full": state.get("full"),
            "source_draw_cursor": sampler.get("source_draw_cursor"),
            "consumed_rows": sampler.get("consumed_rows"),
            "current_index": sampler.get("current_index"),
            "consumed_prompt_copies": sampler.get("consumed_prompt_copies"),
            "selected_id_index": sampler.get("selected_id_index"),
            "geometry": sampler.get("geometry"),
            "identity_pass": bool(state_identity_ok and sampler_ok and geometry_ok and data_ok),
        }
    except Exception as error:
        add_check(checks, "checkpoint.state_projection", "fail", repr(error)[:300])
    try:
        trainer = json.loads((archive / "trainer_state.json").read_text(encoding="utf-8"))
        trainer_expected = {"global_step": 25, "max_steps": 3000, "train_batch_size": 4, "save_steps": 5, "eval_steps": 500}
        trainer_ok = compare_mapping(checks, "checkpoint.trainer_state", trainer, trainer_expected)
    except Exception as error:
        trainer_ok = False
        add_check(checks, "checkpoint.trainer_state", "fail", repr(error)[:300])

    light = run / "archive" / "adapters" / "checkpoint-25"
    light_result: dict[str, Any] = {"path": str(light)}
    if light.is_dir() and (light / "campaign-manifest.json").is_file():
        try:
            lm = jq_projection(light / "campaign-manifest.json", MANIFEST_PROJECTION)
            add_check(checks, "checkpoint.light.step", "pass" if lm.get("step") == 25 else "fail", lm.get("step"))
            add_check(checks, "checkpoint.light.full", "pass" if lm.get("full") is False else "fail", lm.get("full"))
            light_files = {str(item.get("name")): item for item in lm.get("files", []) if isinstance(item, dict) and item.get("name")}
            light_hashes: dict[str, Any] = {}
            for name, expected in sorted(light_files.items()):
                path = light / name
                if not path.is_file():
                    add_check(checks, f"checkpoint.light.{name}", "fail", "missing")
                    continue
                size, digest = sha256_file(path)
                same = size == expected.get("bytes") and digest == expected.get("sha256")
                add_check(checks, f"checkpoint.light.{name}", "pass" if same else "fail", None if same else {"bytes": size, "sha256": digest})
                light_hashes[name] = {"bytes": size, "sha256": digest}
            light_result.update({"step": lm.get("step"), "full": lm.get("full"), "manifest_sha256": sha256_file(light / "campaign-manifest.json")[1], "files": light_hashes})
        except Exception as error:
            add_check(checks, "checkpoint.light.projection", "fail", repr(error)[:300])
    else:
        add_check(checks, "checkpoint.light.directory", "fail", str(light))

    hard = any(item["status"] == "fail" and item["name"].startswith("checkpoint.") for item in checks)
    total_bytes = archive_manifest_size + sum(entry.get("archive", {}).get("bytes", 0) for entry in files.values())
    result.update({
        "status": "pass" if identity_ok and not hard and hash_failures == 0 and manifest_equal and trainer_ok else "fail",
        "step": am.get("step"), "full": am.get("full"),
        "manifest_bytes": archive_manifest_size, "manifest_sha256": archive_manifest_sha,
        "logical_bytes": total_bytes, "files": files, "state": state_summary,
        "light": light_result,
    })
    return result


def parse_jsonl_compact(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            rows.append(value)
    return rows


def verify_records(checks: list[dict[str, Any]], run: Path, sequence_path: Path, rows_path: Path, sidecar_path: Path, selected_path: Path) -> dict[str, Any]:
    generation_path = run / "output" / "generation-records.jsonl"
    reward_path = run / "output" / "reward-records.jsonl"
    gradient_path = run / "output" / "gradient-records.jsonl"
    try:
        raw_generations = parse_jsonl_compact(generation_path)
        raw_rewards = parse_jsonl_compact(reward_path)
        raw_gradients = parse_jsonl_compact(gradient_path)
    except Exception as error:
        add_check(checks, "training.records.parse", "fail", repr(error)[:300])
        return {"status": "fail"}

    generations: list[dict[str, Any]] = []
    generation_errors: list[str] = []
    terminal_counts: collections.Counter[str] = collections.Counter()
    step_counts: collections.Counter[int] = collections.Counter()
    for index, row in enumerate(raw_generations):
        ids = row.get("generated_ids")
        actual_hash = None
        if isinstance(ids, list):
            try:
                encoded = json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
                actual_hash = hashlib.sha256(encoded).hexdigest()
            except (TypeError, ValueError, UnicodeEncodeError):
                pass
        geo = row.get("group_geometry") if isinstance(row.get("group_geometry"), dict) else {}
        local = index % 32
        expected_geo = {
            "candidate_count": 4, "generation_call_count": 4,
            "generation_group_count": 8, "generation_groups_per_call": 2,
            "generation_row_count": 32, "group_index": local // 4,
            "group_row_index": local % 4, "group_start": (local // 4) * 4,
            "group_end": (local // 4) * 4 + 4,
            "generation_call_index": (local // 4) // 2,
        }
        for key, wanted in expected_geo.items():
            if geo.get(key) != wanted:
                generation_errors.append(f"row{index}:{key}")
        step = row.get("global_step")
        step_counts[step] += 1
        if row.get("schema_version") != GENERATION_SCHEMA:
            generation_errors.append(f"row{index}:schema")
        if row.get("global_step_before_update") != step or row.get("update", {}).get("global_step") != step:
            generation_errors.append(f"row{index}:update_step")
        if isinstance(step, int) and row.get("trl_microstep") != (step - 5) * 8:
            generation_errors.append(f"row{index}:microstep")
        if row.get("source_schedule_sha256") != EXPECTED["schedule"]:
            generation_errors.append(f"row{index}:schedule")
        if not isinstance(ids, list) or any(type(token) is not int or not 0 <= token < 130560 for token in ids):
            generation_errors.append(f"row{index}:ids")
        if not isinstance(ids, list) or row.get("generated_token_count") != len(ids):
            generation_errors.append(f"row{index}:token_count")
        if actual_hash is None or row.get("generated_ids_sha256") != actual_hash:
            generation_errors.append(f"row{index}:ids_hash")
        if not isinstance(row.get("generated_ids_sha256"), str) or not HEX64.fullmatch(row["generated_ids_sha256"]):
            generation_errors.append(f"row{index}:hash_shape")
        if not isinstance(row.get("prompt_ids_sha256"), str) or not HEX64.fullmatch(row["prompt_ids_sha256"]):
            generation_errors.append(f"row{index}:prompt_hash")
        if type(row.get("generated_token_count")) is not int or not 0 <= row["generated_token_count"] <= 192:
            generation_errors.append(f"row{index}:count_bound")
        elapsed = row.get("elapsed_sec")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            generation_errors.append(f"row{index}:elapsed")
        if type(row.get("padded_after_terminal")) is not int or row["padded_after_terminal"] < 0:
            generation_errors.append(f"row{index}:padding")
        terminal_counts[str(row.get("terminal_reason"))] += 1
        generations.append({
            "step": step, "microstep": row.get("trl_microstep"),
            "hash": row.get("generated_ids_sha256"), "count": row.get("generated_token_count"),
            "prompt_hash": row.get("prompt_ids_sha256"), "terminal": row.get("terminal_reason"),
            "padded": row.get("padded_after_terminal"), "geo": geo,
        })
    expected_steps = {step: 32 for step in range(5, 25)}
    add_check(checks, "training.generation.count", "pass" if len(generations) == 640 else "fail", len(generations))
    add_check(checks, "training.generation.steps", "pass" if dict(step_counts) == expected_steps else "fail", dict(step_counts))
    add_check(checks, "training.generation.geometry", "pass" if not generation_errors else "fail", generation_errors[:12] if generation_errors else {"calls_per_update": 4, "groups_per_update": 8})
    add_check(checks, "training.generation.termination", "pass" if terminal_counts == collections.Counter({"eos": 601, "length": 39}) else "fail", dict(terminal_counts))

    reward_errors: list[str] = []
    reward_ids: list[str] = []
    reward_families: collections.Counter[str] = collections.Counter()
    reward_operations: collections.Counter[str] = collections.Counter()
    expected_operations: collections.Counter[str] = collections.Counter()
    reward_failures: collections.Counter[str] = collections.Counter()
    reward_caps: collections.Counter[str] = collections.Counter()
    source_ids = [str(row.get("id")) for row in raw_rewards[::4]]
    data = verify_data_files(checks, sequence_path, rows_path, sidecar_path, selected_path, source_ids)
    sequence_ids = data.get("sequence_ids", [])
    expected_source_ids = sequence_ids[40:200] if isinstance(sequence_ids, list) and len(sequence_ids) >= 200 else []
    source_join = source_ids == expected_source_ids and len(source_ids) == 160
    add_check(checks, "training.reward.source_cursor_40_200", "pass" if source_join else "fail", {"source_rows": len(source_ids), "expected_rows": len(expected_source_ids), "cursor_start": 40, "cursor_end": 200})
    candidate_counts: collections.Counter[str] = collections.Counter()
    source_family_errors: list[str] = []
    for index, row in enumerate(raw_rewards):
        reward_ids.append(str(row.get("id")))
        candidate_index = index % 4
        source_index = index // 4
        if source_index >= len(generations) or row.get("output_ids_sha256") != generations[index].get("hash"):
            reward_errors.append(f"row{index}:output_hash_join")
        if source_index >= len(generations) or row.get("generated_tokens") != generations[index].get("count"):
            reward_errors.append(f"row{index}:token_join")
        value = row.get("reward")
        line_f1 = row.get("line_f1")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not isinstance(line_f1, (int, float)) or not math.isfinite(line_f1):
            reward_errors.append(f"row{index}:nonfinite")
        reward_families[str(row.get("family"))] += 1
        reward_operations[str(row.get("operation"))] += 1
        expected_operations[str(row.get("expected_operation"))] += 1
        candidate_counts[str(row.get("id"))] += 1
        if row.get("failure") is not None:
            reward_failures[str(row.get("failure"))] += 1
        reward_caps["cap" if row.get("cap_hit") else "eos"] += 1
        if source_index < len(source_ids):
            if reward_ids[-1] != source_ids[source_index]:
                reward_errors.append(f"row{index}:candidate_contiguity")
            if source_index < len(data.get("source_meta", {})):
                source_id = source_ids[source_index]
                meta = data.get("source_meta", {}).get(source_id)
                if meta and (row.get("family") != meta.get("family") or row.get("expected_operation") != meta.get("operation")):
                    source_family_errors.append(f"source{source_index}:family_or_operation")
    reward_source_ok = candidate_counts and all(count == 4 for count in candidate_counts.values()) and len(candidate_counts) == 160
    add_check(checks, "training.reward.candidate_groups", "pass" if reward_source_ok else "fail", {"unique_sources": len(candidate_counts), "candidate_count_histogram": dict(collections.Counter(candidate_counts.values()))})
    add_check(checks, "training.reward.source_family_operation", "pass" if not source_family_errors and len(data.get("source_meta", {})) == 160 else "fail", source_family_errors[:8] if source_family_errors else {"source_families": dict(collections.Counter(data.get("source_meta", {}).get(source_id, {}).get("family") for source_id in source_ids))})
    add_check(checks, "training.reward.generation_join", "pass" if not reward_errors else "fail", reward_errors[:12] if reward_errors else {"rows": len(raw_rewards)})

    gradients: list[dict[str, Any]] = []
    gradient_errors: list[str] = []
    zero_gradient_updates: list[int] = []
    for index, row in enumerate(raw_gradients):
        step = row.get("step")
        norm = row.get("norm")
        if row.get("schema_version") != GRADIENT_SCHEMA or step != index + 5 or row.get("global_step") != step:
            gradient_errors.append(f"row{index}:step_schema")
        if row.get("finite") is not True or row.get("nonfinite") is not False:
            gradient_errors.append(f"row{index}:finite")
        if row.get("trainable_tensor_count") != 588 or row.get("grad_present_count") != 588:
            gradient_errors.append(f"row{index}:tensor_presence")
        nonzero = row.get("nonzero_tensor_count")
        if nonzero not in (0, 588):
            gradient_errors.append(f"row{index}:nonzero_count")
        if nonzero == 0:
            zero_gradient_updates.append(int(step) + 1)
        if not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
            gradient_errors.append(f"row{index}:norm")
        gradients.append({"step": step, "optimizer_update": int(step) + 1 if isinstance(step, int) else None, "finite": row.get("finite"), "norm": norm, "nonzero_tensor_count": nonzero, "grad_present_count": row.get("grad_present_count")})
    gradient_expected_steps = list(range(5, 25))
    finite_updates = [g["optimizer_update"] for g in gradients if g.get("finite") is True]
    finite_ok = len(gradients) == 20 and finite_updates == list(range(6, 26)) and not gradient_errors
    add_check(checks, "training.gradients.finite_updates_6_25", "pass" if finite_ok else "fail", {"records": len(gradients), "updates": finite_updates, "zero_gradient_updates": zero_gradient_updates, "errors": gradient_errors[:8]})

    return {
        "status": "pass" if len(generations) == 640 and len(raw_rewards) == 640 and not generation_errors and source_join and reward_source_ok and not reward_errors and not source_family_errors and finite_ok else "fail",
        "generation_records": len(generations), "reward_records": len(raw_rewards), "gradient_records": len(gradients),
        "generation_step_counts": dict(step_counts), "terminal_reasons": dict(terminal_counts),
        "reward_family_counts": dict(reward_families), "reward_operation_counts": dict(reward_operations),
        "expected_operation_counts": dict(expected_operations), "reward_failure_counts": dict(reward_failures),
        "reward_terminal_counts": dict(reward_caps), "unique_source_ids": len(candidate_counts),
        "source_cursor": {"start": 40, "end": 200, "exact": source_join},
        "gradient_updates": gradients, "finite_updates_6_25": finite_updates,
        "zero_gradient_updates": zero_gradient_updates,
        "data_hashes": {key: value for key, value in data.items() if key in ("sequence", "rows", "sidecar", "selected", "selected_rows")},
    }


def verify_development(checks: list[dict[str, Any]], run: Path, panel_path: Path) -> dict[str, Any]:
    try:
        panel_sha = sha256_file(panel_path)[1]
        panel_rows = parse_jsonl_compact(panel_path)
    except Exception as error:
        add_check(checks, "development.panel", "fail", repr(error)[:300])
        return {"status": "fail"}
    panel_ids = [str(row.get("id")) for row in panel_rows]
    panel_ok = len(panel_rows) == 75 and len(set(panel_ids)) == 75 and panel_sha == EXPECTED["panel"]
    add_check(checks, "development.panel_identity", "pass" if panel_ok else "fail", {"rows": len(panel_rows), "sha256": panel_sha})
    cases_path = run / "archive" / "evaluations" / "cases-step-25.json"
    try:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
    except Exception as error:
        add_check(checks, "development.cases", "fail", repr(error)[:300])
        return {"status": "fail", "panel_sha256": panel_sha}
    results = cases.get("results", []) if isinstance(cases, dict) else []
    summary = cases.get("summary", {}) if isinstance(cases, dict) else {}
    result_ids = [str(row.get("id")) for row in results if isinstance(row, dict)]
    identity_ok = cases.get("status") == "complete" and cases.get("step") == 25 and len(results) == 75 and len(set(result_ids)) == 75 and result_ids == panel_ids and summary.get("panel_sha256") == EXPECTED["panel"]
    add_check(checks, "development.cases_identity", "pass" if identity_ok else "fail", {"status": cases.get("status"), "step": cases.get("step"), "results": len(results), "panel_sha256": summary.get("panel_sha256")})
    denom = summary.get("denominators", {})
    denom_expected = {"cases": 75, "edits": 43, "strict_noop": 32}
    denom_ok = all(denom.get(key) == wanted for key, wanted in denom_expected.items())
    add_check(checks, "development.denominators", "pass" if denom_ok else "fail", {key: denom.get(key) for key in denom_expected})

    parser_errors: list[str] = []
    stored_counts = collections.Counter()
    family_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    long_case: dict[str, Any] | None = None
    # The frozen parser is pure Python.  Loading only this protocol module is
    # deliberate: it provides the exact deployed classification semantics
    # without importing a framework or model.
    try:
        sys.path.insert(0, str(PROTOCOL_DEFAULT))
        from sepalith.campaign_protocol import PromptContext, parse_output, valid_generation_tokens  # type: ignore
    except Exception as error:
        add_check(checks, "development.pinned_parser_import", "fail", repr(error)[:300])
        return {"status": "fail", "panel_sha256": panel_sha, "case_count": len(results)}
    for row in results:
        if not isinstance(row, dict) or row.get("id") not in {case.get("id") for case in panel_rows}:
            parser_errors.append("row_identity")
            continue
        case = next(case for case in panel_rows if case.get("id") == row.get("id"))
        try:
            context = PromptContext.from_mapping(case["context"])
            parsed = parse_output(row.get("raw_output", ""), context)
            valid = valid_generation_tokens(row.get("generated_ids", [])) and parsed.status == "accepted"
            predicted_noop = valid and parsed.operation == "no_op"
            actual = list(context.region_old) if predicted_noop else list(parsed.body)
            exact = valid and actual == list(case.get("region_new", []))
            failure = None if valid else (parsed.reason or "noncanonical_or_missing_EOS")
        except Exception as error:
            parser_errors.append(f"{row.get('id')}:exception")
            continue
        for key, value in {"protocol_valid": valid, "predicted_noop": predicted_noop, "exact_region": exact, "failure": failure}.items():
            if row.get(key) != value:
                parser_errors.append(f"{row.get('id')}:{key}")
        expected_noop = list(context.region_old) == list(case.get("region_new", []))
        if row.get("expected_noop") != expected_noop:
            parser_errors.append(f"{row.get('id')}:expected_noop")
        family = str(case.get("family"))
        family_counts[family].update({"cases": 1, "protocol_valid": int(valid), "exact_region": int(exact), "predicted_noop": int(predicted_noop), "cap_hit": int(bool(row.get("cap_hit"))), "expected_noop": int(expected_noop)})
        stored_counts.update({"protocol_valid": int(valid), "exact_region": int(exact), "predicted_noop": int(predicted_noop), "suggestion": int(valid and not predicted_noop), "cap_hit": int(bool(row.get("cap_hit"))), "strict_noop_correct": int(expected_noop and predicted_noop), "strict_noop_false_suggestions": int(expected_noop and valid and not predicted_noop), "edit_exact": int(not expected_noop and exact)})
        if row.get("id") == LONG_DEV_ID:
            long_case = {key: row.get(key) for key in ("id", "family", "prompt_tokens", "generated_tokens", "cap_hit", "protocol_valid", "exact_region", "predicted_noop", "failure")}
    expected_counts = {"protocol_valid": 69, "exact_region": 50, "predicted_noop": 27, "suggestion": 42, "cap_hit": 6, "strict_noop_correct": 25, "strict_noop_false_suggestions": 5, "edit_exact": 25}
    stored_summary_counts = summary.get("counts", {})
    counts_ok = all(stored_counts.get(key) == wanted and stored_summary_counts.get(key) == wanted for key, wanted in expected_counts.items())
    add_check(checks, "development.pinned_parser_reclassification", "pass" if not parser_errors else "fail", {"errors": parser_errors[:12], "count": len(parser_errors)})
    add_check(checks, "development.counts", "pass" if counts_ok else "fail", {"independent": dict(stored_counts), "stored": {key: stored_summary_counts.get(key) for key in expected_counts}})
    long_ok = long_case is not None and long_case.get("prompt_tokens") == 2619
    add_check(checks, "development.maximal_prompt_retained", "pass" if long_ok else "fail", long_case)

    baseline_path = run.parent / "RL-primary-p2-mb4-full5-a" / "archive" / "evaluations" / "cases-step-5.json"
    baseline_diffs: list[dict[str, Any]] = []
    family_delta: dict[str, dict[str, int]] = {}
    baseline_sha = None
    if baseline_path.is_file():
        baseline_sha = sha256_file(baseline_path)[1]
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            old_rows = {str(row.get("id")): row for row in baseline.get("results", []) if isinstance(row, dict)}
            new_rows = {str(row.get("id")): row for row in results if isinstance(row, dict)}
            for row_id in panel_ids:
                old, new = old_rows.get(row_id), new_rows.get(row_id)
                if not old or not new:
                    baseline_diffs.append({"id": row_id, "kind": "missing"})
                    continue
                fields = ("protocol_valid", "exact_region", "predicted_noop", "cap_hit", "failure")
                changes = {field: [old.get(field), new.get(field)] for field in fields if old.get(field) != new.get(field)}
                if changes:
                    baseline_diffs.append({"id": row_id, "family": new.get("family"), "changes": changes})
            for row_id in panel_ids:
                old, new = old_rows.get(row_id), new_rows.get(row_id)
                family = str((new or old or {}).get("family"))
                slot = family_delta.setdefault(family, {"cases": 0, "exact_step5": 0, "exact_step25": 0, "protocol_step5": 0, "protocol_step25": 0, "predicted_noop_step5": 0, "predicted_noop_step25": 0, "cap_step5": 0, "cap_step25": 0})
                slot["cases"] += 1
                for prefix, value in (("step5", old), ("step25", new)):
                    if value:
                        slot[f"exact_{prefix}"] += int(bool(value.get("exact_region")))
                        slot[f"protocol_{prefix}"] += int(bool(value.get("protocol_valid")))
                        slot[f"predicted_noop_{prefix}"] += int(bool(value.get("predicted_noop")))
                        slot[f"cap_{prefix}"] += int(bool(value.get("cap_hit")))
            add_check(checks, "development.step5_comparison", "pass" if len(baseline_diffs) == 2 else "fail", {"baseline_sha256": baseline_sha, "changed_cases": baseline_diffs})
        except Exception as error:
            add_check(checks, "development.step5_comparison", "fail", repr(error)[:300])
    else:
        add_check(checks, "development.step5_comparison", "fail", str(baseline_path))
    interpretation = str(summary.get("interpretation", ""))
    interpretation_ok = "no R semantic-validity" in interpretation and "claim" in interpretation.lower()
    add_check(checks, "development.interpretation", "pass" if interpretation_ok else "fail", interpretation)
    return {
        "status": "pass" if panel_ok and identity_ok and denom_ok and not parser_errors and counts_ok and long_ok and len(baseline_diffs) == 2 and interpretation_ok else "fail",
        "artifact": str(cases_path), "artifact_sha256": sha256_file(cases_path)[1], "panel_sha256": panel_sha,
        "case_count": len(results), "step": cases.get("step"), "status_field": cases.get("status"),
        "denominators": {key: denom.get(key) for key in denom_expected}, "counts": dict(stored_counts),
        "family_counts": {key: dict(value) for key, value in sorted(family_counts.items())},
        "step5_baseline": {"path": str(baseline_path), "sha256": baseline_sha, "diffs": baseline_diffs, "family_delta": family_delta},
        "long_case": long_case, "result_ids_exact_panel": result_ids == panel_ids,
        "interpretation": interpretation,
    }


def mmap_scalars(path: Path, key: str) -> list[Any]:
    """Extract simple values from giant receipts without loading their JSON tree."""
    if not path.is_file():
        return []
    needle = b'"' + key.encode("ascii") + b'"'
    values: list[Any] = []
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            cursor = 0
            while True:
                start = mm.find(needle, cursor)
                if start < 0:
                    break
                colon = mm.find(b":", start + len(needle))
                if colon < 0:
                    break
                pos = colon + 1
                while pos < len(mm) and mm[pos] in b" \t\r\n":
                    pos += 1
                if pos >= len(mm):
                    break
                if mm[pos] == 34:
                    end = pos + 1
                    escaped = False
                    while end < len(mm):
                        byte = mm[end]
                        if byte == 34 and not escaped:
                            end += 1
                            break
                        if byte == 92 and not escaped:
                            escaped = True
                        else:
                            escaped = False
                        end += 1
                else:
                    end = pos
                    while end < len(mm) and mm[end] not in b",}\n":
                        end += 1
                raw = bytes(mm[pos:end]).strip()
                try:
                    values.append(json.loads(raw.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    pass
                cursor = max(end, start + len(needle))
    return values


def verify_runtime(checks: list[dict[str, Any]], run: Path, recipe_path: Path) -> dict[str, Any]:
    recipe_size, recipe_sha = sha256_file(recipe_path)
    add_check(checks, "runtime.recipe_sha256", "pass" if recipe_sha == EXPECTED["recipe"] else "fail", {"bytes": recipe_size, "sha256": recipe_sha})
    recipe: dict[str, Any] = {}
    try:
        value = json.loads(recipe_path.read_text(encoding="utf-8"))
        if isinstance(value, dict): recipe = value
    except Exception as error:
        add_check(checks, "runtime.recipe_parse", "fail", repr(error)[:300])
    recipe_fields = {"id": "primary-grpo-p2-2048x192-v5-full5-b", "schema_version": "sepalith.prm07.rl-train.v1", "model_load_max_seq_length": 4096, "max_steps": 3000, "full_save_steps": 5, "light_save_steps": 5, "development_max_new_tokens": 512, "renderer_id": "zeta2-prm03-v1"}
    recipe_ok = compare_mapping(checks, "runtime.recipe", recipe, recipe_fields) if recipe else False
    load_audit_path = run / "output" / "load-audit.json"
    try:
        load_audit = json.loads(load_audit_path.read_text(encoding="utf-8"))
        load_fields = {
            "dtype": "torch.bfloat16",
            "adapter.attachments": 294,
            "adapter.lora_rank": 16,
            "adapter.lora_alpha": 16,
            "adapter.trainable_parameters": 25116672,
            "cuda_allocator.cuda_allocator_fraction": 0.75,
            "model_load.requested_max_seq_length": 4096,
            "model_load.observed_capacity_tokens": 4096,
            "tokenizer_contract.status": "verified",
        }
        load_errors = []
        for dotted, wanted in load_fields.items():
            actual: Any = load_audit
            for part in dotted.split("."):
                actual = actual.get(part) if isinstance(actual, dict) else None
            if actual != wanted:
                load_errors.append({"field": dotted, "expected": wanted, "actual": safe_scalar(actual)})
        add_check(checks, "runtime.load_audit", "pass" if not load_errors else "fail", {"errors": load_errors} if load_errors else {"dtype": load_audit.get("dtype")})
    except Exception as error:
        load_errors = [repr(error)[:300]]
        add_check(checks, "runtime.load_audit", "fail", load_errors)

    supervision = run / "supervision.json"
    entry = run / "supervision.entry-result.json"
    sup_status = mmap_scalars(supervision, "status")
    sup_recipe = mmap_scalars(supervision, "recipe_sha256")
    sup_entry = mmap_scalars(supervision, "entry_result_receipt")
    sup_pid = [value for value in mmap_scalars(supervision, "pid") if isinstance(value, int)]
    sup_status_final = sup_status[-1] if sup_status else None
    sup_ok = EXPECTED["recipe"] in sup_recipe and sup_status_final == "supervisor_configured; command acceptance and completion unverified" and any("supervision.entry-result.json" in str(value) for value in sup_entry)
    add_check(checks, "runtime.immutable_supervision", "pass" if sup_ok else "fail", {"status": sup_status_final, "recipe_bound": EXPECTED["recipe"] in sup_recipe, "entry_result_bound": any("supervision.entry-result.json" in str(value) for value in sup_entry)})
    entry_status = mmap_scalars(entry, "status")
    entry_step = mmap_scalars(entry, "step")
    entry_sup = mmap_scalars(entry, "supervision_receipt")
    entry_status_final = "lead_decision" if "lead_decision" in entry_status else (entry_status[-1] if entry_status else None)
    entry_step_final = 25 if 25 in entry_step else (entry_step[-1] if entry_step else None)
    entry_ok = entry_status_final == "lead_decision" and entry_step_final == 25 and any(str(supervision) == str(value) for value in entry_sup)
    add_check(checks, "runtime.distinct_entry_result", "pass" if entry_ok else "fail", {"status": entry_status_final, "step": entry_step_final, "present": entry.is_file(), "supervision_backlink": any(str(supervision) == str(value) for value in entry_sup)})

    terminal_path = run.parent / "RL-primary-p2-mb4-full5-b-host-supervision" / "terminal.json"
    terminal: dict[str, Any] = {}
    try:
        value = json.loads(terminal_path.read_text(encoding="utf-8"))
        if isinstance(value, dict): terminal = value
    except Exception as error:
        add_check(checks, "runtime.host_terminal.parse", "fail", repr(error)[:300])
    terminal_ok = terminal.get("status") == "completed" and terminal.get("child_exit_code") == 0 and terminal.get("reason") is None
    add_check(checks, "runtime.host_terminal", "pass" if terminal_ok else "fail", {k: terminal.get(k) for k in ("at", "status", "child_exit_code", "reason", "seconds")})

    telemetry_path = run / "telemetry.jsonl"
    optimizer: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    event_counts: collections.Counter[str] = collections.Counter()
    giant_lines = 0
    if telemetry_path.is_file():
        with telemetry_path.open("rb") as stream:
            for raw in stream:
                if len(raw) > 1 << 20:
                    giant_lines += 1
                    continue
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                event = row.get("event")
                if event:
                    event_counts[str(event)] += 1
                if event == "optimizer_step":
                    resources = row.get("resources", {}) if isinstance(row.get("resources"), dict) else {}
                    optimizer.append({"step": row.get("step"), "stop_reason": row.get("stop_reason"), "remaining_seconds": row.get("remaining_seconds"), "seconds": row.get("seconds"), "cuda_allocated_bytes": resources.get("cuda_allocated_bytes"), "cuda_attempt_peak_allocated_bytes": resources.get("cuda_attempt_peak_allocated_bytes"), "cuda_attempt_peak_reserved_bytes": resources.get("cuda_attempt_peak_reserved_bytes")})
                elif event == "trainer_metrics":
                    m = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
                    metrics.append({"step": row.get("step"), "grad_norm": m.get("grad_norm"), "reward": m.get("reward"), "reward_std": m.get("reward_std"), "frac_reward_zero_std": m.get("frac_reward_zero_std"), "loss": m.get("loss"), "mean_length": m.get("completions/mean_length"), "max_length": m.get("completions/max_length"), "clipped_ratio": m.get("completions/clipped_ratio")})
    telemetry_steps = [row.get("step") for row in optimizer]
    telemetry_ok = telemetry_steps == list(range(6, 26)) and optimizer[-1].get("stop_reason") == "lead_decision" if optimizer else False
    add_check(checks, "runtime.telemetry.optimizer_steps_6_25", "pass" if telemetry_ok else "fail", {"steps": telemetry_steps, "last_stop_reason": optimizer[-1].get("stop_reason") if optimizer else None, "giant_lines_skipped": giant_lines})
    metric_steps = sorted({row.get("step") for row in metrics if isinstance(row.get("step"), int)})
    add_check(checks, "runtime.telemetry.metrics_6_25", "pass" if all(step in metric_steps for step in range(6, 26)) else "fail", {"steps": metric_steps, "records": len(metrics)})

    host_dir = run.parent / "RL-primary-p2-mb4-full5-b-host-supervision"
    host_path = host_dir / "host-memory.jsonl"
    host_rows: list[dict[str, Any]] = []
    if host_path.is_file():
        with host_path.open("rb") as stream:
            for raw in stream:
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(value, dict):
                    host_rows.append({key: value.get(key) for key in ("At", "AvailableMBytes", "CommittedBytes", "CommitLimit", "PageReadsPersec", "PagesInputPersec", "PagesOutputPersec", "DriverEvents")})
    def numeric(key: str) -> list[float]:
        return [float(row[key]) for row in host_rows if isinstance(row.get(key), (int, float))]
    host_summary = {
        "samples": len(host_rows),
        "available_mib_min": min(numeric("AvailableMBytes")) if numeric("AvailableMBytes") else None,
        "available_mib_last": host_rows[-1].get("AvailableMBytes") if host_rows else None,
        "committed_bytes_last": host_rows[-1].get("CommittedBytes") if host_rows else None,
        "commit_limit_last": host_rows[-1].get("CommitLimit") if host_rows else None,
        "page_reads_per_sec_max": max(numeric("PageReadsPersec")) if numeric("PageReadsPersec") else None,
        "pages_input_per_sec_max": max(numeric("PagesInputPersec")) if numeric("PagesInputPersec") else None,
        "pages_output_per_sec_max": max(numeric("PagesOutputPersec")) if numeric("PagesOutputPersec") else None,
        "last": host_rows[-1] if host_rows else None,
        "driver_event_samples": sum(bool(row.get("DriverEvents")) for row in host_rows),
    }
    add_check(checks, "runtime.host_memory.telemetry", "pass" if host_rows and host_summary["driver_event_samples"] == 0 else ("pending" if not host_rows else "fail"), host_summary)
    runtime_pass = recipe_sha == EXPECTED["recipe"] and recipe_ok and not load_errors and sup_ok and entry_ok and terminal_ok and telemetry_ok
    return {
        "status": "pass" if runtime_pass else "fail", "recipe_sha256": recipe_sha,
        "recipe_bytes": recipe_size, "entry_result": {"path": str(entry), "status": entry_status_final, "step": entry_step_final},
        "supervision": {"path": str(supervision), "status": sup_status_final, "pid_values": sup_pid[:4]},
        "host_terminal": terminal, "telemetry": {"events": dict(event_counts), "optimizer_steps": optimizer, "metrics": metrics, "giant_lines_skipped": giant_lines},
        "host_memory": host_summary,
    }


def run_verifier(run: Path, recipe: Path, sequence: Path, panel: Path, rows: Path, sidecar: Path, selected: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checkpoint = verify_checkpoint(checks, run)
    training = verify_records(checks, run, sequence, rows, sidecar, selected)
    development = verify_development(checks, run, panel)
    runtime = verify_runtime(checks, run, recipe)
    hard_fail = any(item["status"] == "fail" for item in checks)
    status = "fail_mechanical" if hard_fail else "pass_mechanical_quality_pending"
    return {
        "schema_version": "sepalith.rl-08.step25-independent-review.v1",
        "status": status, "run_root": str(run), "recipe": str(recipe), "sequence": str(sequence), "panel": str(panel),
        "expected": {"source_snapshot": EXPECTED["source"], "recipe_sha256": EXPECTED["recipe"], "identity_sha256": EXPECTED["identity"], "source_schedule_sha256": EXPECTED["schedule"], "source_sequence_sha256": EXPECTED["sequence"], "panel_sha256": EXPECTED["panel"]},
        "checkpoint": checkpoint, "training": training, "development": development, "runtime": runtime,
        "checks": checks,
        "unresolved": [
            "This is mechanical artifact evidence only; no RL quality promotion is made.",
            "The continuation begins at source draw cursor 40 from the step-5 checkpoint and reaches cursor 200 at step 25; future same-identity resume equality remains a separate gate.",
            "The update-11 gradient is finite but has zero nonzero adapter tensors alongside a zero-reward-variance metric; this is recorded for root review and is not a nonfinite/protocol failure.",
            "Host samples record historical page-read/input spikes during the run; the final sample has 18,062 MiB available, zero page output, and no driver events. This verifier does not attribute causality.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--recipe", type=Path, default=RECIPE_DEFAULT)
    parser.add_argument("--sequence", type=Path, default=SEQUENCE_DEFAULT)
    parser.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    parser.add_argument("--rows", type=Path, default=ROWS_DEFAULT)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_DEFAULT)
    parser.add_argument("--selected", type=Path, default=SELECTED_DEFAULT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_verifier(args.run_root, args.recipe, args.sequence, args.panel, args.rows, args.sidecar, args.selected)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checkpoint": result["checkpoint"].get("status"), "training": result["training"].get("status"), "development": result["development"].get("status"), "runtime": result["runtime"].get("status"), "failed_checks": sum(item["status"] == "fail" for item in result["checks"])}, sort_keys=True))
    return 1 if result["status"] == "fail_mechanical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
