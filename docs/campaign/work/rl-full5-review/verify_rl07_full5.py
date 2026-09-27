#!/usr/bin/env python3
"""Bounded, framework-free verifier for the RL-07 full5 artifact.

The verifier deliberately reads only the known run paths.  It hashes checkpoint
files in streaming mode, skips the giant identity line in telemetry, and never
imports torch, CUDA, Transformers, TRL, or Unsloth.  A complete checkpoint is
reported independently from development evaluation and process completion.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import mmap
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "source": "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9",
    "recipe": "c85b7e3e9fd7927d429a6d45085e98d3f5bb8b06047e468f5bd7dc9190939969",
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
    "previous_adapter": "ab2caa03ed9ab682d64af220f648fe9be6f1ac8f9c1da28f3e980d29da36be3c",
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

EXPECTED_TOKENIZER = {
    "bos_id": 0,
    "eos_id": 1,
    "pad_id": 1,
    "native_eog_ids": [1, 130073],
    "vocab_size": 130560,
    "revision": EXPECTED["parent_revision"],
}

EXPECTED_SCHEDULE = {
    "buffer_reuse": 8,
    "development_cases": 75,
    "development_every": 25,
    "first_development_update": 5,
    "full_save_steps": 5,
    "light_save_steps": 5,
    "generation_batch_size": 32,
    "main_update_ceiling": 3000,
    "num_iterations": 1,
    "sampler_id": "campaign-repeat-manifest-order-v1",
    "seed": 3407,
    "source_draw_schedule_sha256": EXPECTED["schedule"],
    "source_draw_sequence_sha256": EXPECTED["sequence"],
    "source_draws": 24000,
    "source_draws_per_update": 8,
    "steps_per_generation": 8,
}

EXPECTED_FILES = {
    "README.md": (5351, "a755e35444ec785e93a6433c57f4ae24c95129d87611e432397fad3fecdba952"),
    "adapter_config.json": (1354, "5b01cb0c6ed9bb1bdf3d41b33d1daa1da7cdc6e17b3ed922f96d2fc0f8dfa9b0"),
    "adapter_model.safetensors": (100544848, EXPECTED["previous_adapter"]),
    "campaign-state.json": (148011281, "9fd1f04a498bff3d8dfdfb0482205d9c1a607ee07f8da16663f08d2b9a36554e"),
    "chat_template.jinja": (9060, "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c"),
    "optimizer.pt": (51712341, "7112c8d9c6c33c244b01f9c4fbc7d6729b917ad4b34712a6139a468307dd8011"),
    "rng_state.pth": (14645, "d23761a4230c0d39a27d4cc1f7f9beed1375063ec8c47e577a2a6f391a7b367b"),
    "scheduler.pt": (1465, "fab3d2b257c78ef5e23915b2c7640c8c42d2084b27d193cd0df20826eb70ada5"),
    "tokenizer.json": (9894271, "d5ede0bcd21e0676a58b176937262fb80a06a32b5cb4ed8bfed8b7d11e45b0e1"),
    "tokenizer_config.json": (94399, "71e726da77f3f0bbfde07d916acae8bd5a316d1857a00979cdc2e9ed3061034c"),
    "trainer_state.json": (5692, "8e79b93d1e562e45215a076512ae36fe37bce3565730b5fce8192b718dfe783e"),
    "training_args.bin": (6737, "df973af67a3714219fabb249a9c3884a94864f839fe57175f7207325882c2c7b"),
}

LONG_DEV_ID = "dat07-existing-719cd49683667d0fb86fb2fa"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def path_get(obj: Any, path: str, missing: Any = None) -> Any:
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return missing
        obj = obj[part]
    return obj


def is_pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def mmap_scalar_values(path: Path, key: str) -> list[Any]:
    """Extract simple JSON scalar values without materializing giant receipts."""
    if not path.exists():
        return []
    pattern = re.compile(rb'"' + re.escape(key.encode()) + rb'"\s*:\s*([^,}\n]+)')
    values: list[Any] = []
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for match in pattern.finditer(mm):
                raw = match.group(1).strip()
                # A quoted string may contain commas, so use a bounded scan from
                # the match start for strings.  All target receipt values are
                # simple strings, booleans, numbers, or null.
                if raw.startswith(b'"'):
                    start = match.start(1)
                    end = start + 1
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
                    raw = mm[start:end]
                try:
                    values.append(json.loads(raw.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    continue
    return values


def compact_file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def add_check(checks: list[dict[str, Any]], name: str, status: str, details: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "status": status}
    if details is not None:
        item["details"] = details
    checks.append(item)


def verify_identity(checks: list[dict[str, Any]], identity: dict[str, Any], where: str) -> None:
    data = identity.get("data", {})
    parent = identity.get("parent", {})
    policy = identity.get("policy", {})
    renderer = identity.get("renderer", {})
    schedule = identity.get("schedule", {})
    source = identity.get("source", {})
    tokenizer = identity.get("tokenizer", {})

    for key, expected in {
        "selected_ids_sha256": EXPECTED["selected"],
        "ordered_ids_sha256": EXPECTED["ordered"],
        "row_identity_sha256": EXPECTED["row_identity"],
        "rows_sha256": EXPECTED["rows"],
        "context_sha256": EXPECTED["context"],
        "sidecar_artifact_sha256": EXPECTED["context"],
        "row_count": 8440,
        "split": "train",
        "admission_status": "admitted",
    }.items():
        actual = data.get(key)
        add_check(checks, f"{where}.data.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})

    for key, expected in {
        "kind": "merged_sft",
        "manifest_sha256": EXPECTED["parent_manifest"],
        "merged_weights_sha256": EXPECTED["parent_weights"],
        "base_model_revision": EXPECTED["parent_revision"],
    }.items():
        actual = parent.get(key)
        add_check(checks, f"{where}.parent.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})

    for key, expected in EXPECTED_POLICY.items():
        actual = policy.get(key)
        add_check(checks, f"{where}.policy.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})

    for key, expected in EXPECTED_RENDERER.items():
        actual = renderer.get(key)
        add_check(checks, f"{where}.renderer.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})

    for key, expected in EXPECTED_SCHEDULE.items():
        actual = schedule.get(key)
        add_check(checks, f"{where}.schedule.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})

    for key, expected in EXPECTED_TOKENIZER.items():
        actual = tokenizer.get(key)
        add_check(checks, f"{where}.tokenizer.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})


def verify_checkpoint(checks: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    full = run / "archive" / "full" / "checkpoint-5"
    output = run / "output" / "checkpoint-5"
    if not full.is_dir():
        add_check(checks, "full_checkpoint.directory", "fail", str(full))
        return {"status": "fail", "step": None, "files": {}}

    manifest_path = full / "campaign-manifest.json"
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:  # pragma: no cover - live artifact error path
        add_check(checks, "full_checkpoint.manifest_parse", "fail", repr(exc))
        return {"status": "fail", "step": None, "files": {}}

    add_check(checks, "full_checkpoint.manifest_step", "pass" if manifest.get("step") == 5 else "fail", manifest.get("step"))
    add_check(checks, "full_checkpoint.manifest_full", "pass" if manifest.get("full") is True else "fail", manifest.get("full"))
    verify_identity(checks, manifest.get("identity", {}), "full_checkpoint.manifest_identity")

    listed = manifest.get("files", {})
    file_results: dict[str, Any] = {}
    for rel, expected in listed.items():
        if not isinstance(expected, dict):
            add_check(checks, f"full_checkpoint.file.{rel}", "fail", "manifest entry is not an object")
            continue
        exp_bytes, exp_sha = expected.get("bytes"), expected.get("sha256")
        archive_file = full / rel
        output_file = output / rel
        result: dict[str, Any] = {"expected_bytes": exp_bytes, "expected_sha256": exp_sha}
        for label, path in (("archive", archive_file), ("output", output_file)):
            if not path.is_file():
                add_check(checks, f"full_checkpoint.file.{label}.{rel}", "fail", "missing")
                result[label] = {"present": False}
                continue
            size, digest = sha256_file(path)
            ok = size == exp_bytes and digest == exp_sha
            add_check(checks, f"full_checkpoint.file.{label}.{rel}", "pass" if ok else "fail", {"bytes": size, "sha256": digest})
            result[label] = {"present": True, "bytes": size, "sha256": digest}
        a = result.get("archive", {})
        o = result.get("output", {})
        same = bool(a.get("present") and o.get("present") and a.get("bytes") == o.get("bytes") and a.get("sha256") == o.get("sha256"))
        add_check(checks, f"full_checkpoint.file.archive_output_equal.{rel}", "pass" if same else "fail")
        result["archive_output_equal"] = same
        file_results[rel] = result

    archive_manifest_size, archive_manifest_sha = sha256_file(manifest_path)
    output_manifest = output / "campaign-manifest.json"
    output_manifest_sha = None
    output_manifest_size = None
    if output_manifest.is_file():
        output_manifest_size, output_manifest_sha = sha256_file(output_manifest)
    manifest_equal = output_manifest_sha == archive_manifest_sha and output_manifest_size == archive_manifest_size
    add_check(checks, "full_checkpoint.manifest.archive_output_equal", "pass" if manifest_equal else "fail", {"archive_sha256": archive_manifest_sha, "output_sha256": output_manifest_sha})

    # The adapter archive is a separate light artifact.  Confirm its adapter
    # bytes and the checkpoint-5 metadata without treating it as full state.
    adapter = run / "archive" / "adapters" / "checkpoint-5"
    adapter_model = adapter / "adapter_model.safetensors"
    if adapter_model.is_file():
        adapter_size, adapter_sha = sha256_file(adapter_model)
        adapter_ok = adapter_size == EXPECTED_FILES["adapter_model.safetensors"][0] and adapter_sha == EXPECTED["previous_adapter"]
        add_check(checks, "adapter_archive.adapter_model", "pass" if adapter_ok else "fail", {"bytes": adapter_size, "sha256": adapter_sha})
    else:
        add_check(checks, "adapter_archive.adapter_model", "fail", "missing")

    state_path = full / "campaign-state.json"
    state_summary: dict[str, Any] = {}
    try:
        state = read_json(state_path)
        add_check(checks, "full_checkpoint.state_parse", "pass")
        add_check(checks, "full_checkpoint.state_full", "pass" if state.get("full") is True else "fail", state.get("full"))
        add_check(checks, "full_checkpoint.state_step", "pass" if state.get("step") == 5 else "fail", state.get("step"))
        if isinstance(state.get("identity"), dict):
            verify_identity(checks, state["identity"], "full_checkpoint.state_identity")
        sampler = state.get("sampler", {})
        required_sampler = {
            "buffer_reuse": 8,
            "candidate_count": 4,
            "consumed_prompt_copies": 320,
            "consumed_rows": 1280,
            "current_index": 1280,
            "epoch": 0,
            "generation_rows_per_update": 32,
            "num_samples": 8440,
            "prompt_groups_per_batch": 8,
            "repeat_count": 8,
            "sampler_id": "campaign-repeat-manifest-order-v1",
            "sampler_rows_per_update": 256,
            "seed": 3407,
            "selected_id_index": 5460,
            "shuffle": False,
            "source_draw_cursor": 40,
            "source_draw_schedule_sha256": EXPECTED["schedule"],
            "source_draw_sequence_sha256": EXPECTED["sequence"],
            "source_draws": 24000,
            "source_draws_bound": True,
        }
        for key, expected in required_sampler.items():
            actual = sampler.get(key)
            add_check(checks, f"full_checkpoint.sampler.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})
        sampler_geometry = sampler.get("geometry", {})
        for key, expected in {
            "generation_batch_size": 32,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 4,
            "steps_per_generation": 8,
        }.items():
            actual = sampler_geometry.get(key)
            add_check(checks, f"full_checkpoint.sampler.geometry.{key}", "pass" if actual == expected else "fail", {"expected": expected, "actual": actual})
        state_summary = {key: sampler.get(key) for key in ("source_draw_cursor", "consumed_rows", "current_index", "consumed_prompt_copies", "selected_id_index")}
        state_summary["geometry"] = {key: sampler_geometry.get(key) for key in ("generation_batch_size", "gradient_accumulation_steps", "per_device_train_batch_size", "steps_per_generation")}
        state_summary["full"] = state.get("full")
        state_summary["step"] = state.get("step")
        del sampler, state
        gc.collect()
    except Exception as exc:  # pragma: no cover - live artifact error path
        add_check(checks, "full_checkpoint.state_parse", "fail", repr(exc))

    trainer_state_path = full / "trainer_state.json"
    try:
        trainer = read_json(trainer_state_path)
        trainer_ok = trainer.get("global_step") == 5 and trainer.get("max_steps") == 3000 and trainer.get("train_batch_size") == 4
        add_check(checks, "full_checkpoint.trainer_state", "pass" if trainer_ok else "fail", {k: trainer.get(k) for k in ("global_step", "max_steps", "train_batch_size", "save_steps", "eval_steps")})
        del trainer
    except Exception as exc:  # pragma: no cover
        add_check(checks, "full_checkpoint.trainer_state", "fail", repr(exc))

    hard_fail = any(c["status"] == "fail" for c in checks if c["name"].startswith(("full_checkpoint.", "adapter_archive.")))
    return {
        "status": "fail" if hard_fail else "pass",
        "step": manifest.get("step"),
        "full": manifest.get("full"),
        "manifest_sha256": archive_manifest_sha,
        "manifest_bytes": archive_manifest_size,
        "files": file_results,
        "state": state_summary,
        "adapter_model_sha256": EXPECTED["previous_adapter"],
    }


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: record is not an object")
            rows.append(value)
    return rows


def verify_generation_and_rewards(checks: list[dict[str, Any]], run: Path, sequence_path: Path) -> dict[str, Any]:
    generation_path = run / "output" / "generation-records.jsonl"
    reward_path = run / "output" / "reward-records.jsonl"
    try:
        generations = parse_jsonl(generation_path)
        rewards = parse_jsonl(reward_path)
    except Exception as exc:
        add_check(checks, "training.records_parse", "fail", repr(exc))
        return {"status": "fail", "generations": 0, "rewards": 0}

    step_counts = collections.Counter(row.get("global_step") for row in generations)
    expected_counts = {step: 32 for step in range(5)}
    add_check(checks, "training.generation_count", "pass" if len(generations) == 160 else "fail", len(generations))
    add_check(checks, "training.generation_step_counts", "pass" if dict(step_counts) == expected_counts else "fail", dict(step_counts))
    add_check(checks, "training.reward_count", "pass" if len(rewards) == 160 else "fail", len(rewards))

    generation_errors: list[str] = []
    call_indices: collections.Counter[int] = collections.Counter()
    group_indices: collections.Counter[int] = collections.Counter()
    terminal_counts: collections.Counter[str] = collections.Counter()
    generated_hashes: list[str] = []
    for index, row in enumerate(generations):
        step = row.get("global_step")
        geo = row.get("group_geometry") or {}
        expected_geo = {
            "candidate_count": 4,
            "generation_call_count": 4,
            "generation_group_count": 8,
            "generation_groups_per_call": 2,
            "generation_row_count": 32,
            "group_index": index % 32 // 4,
            "group_row_index": index % 4,
            "group_start": (index % 32 // 4) * 4,
            "group_end": (index % 32 // 4) * 4 + 4,
            "generation_call_index": (index % 32 // 4) // 2,
        }
        for key, expected in expected_geo.items():
            if geo.get(key) != expected:
                generation_errors.append(f"row{index}:{key}={geo.get(key)!r}, want {expected!r}")
        if row.get("schema_version") != "sepalith.rl.generation-record.v1":
            generation_errors.append(f"row{index}:schema_version")
        if row.get("source_schedule_sha256") != EXPECTED["schedule"]:
            generation_errors.append(f"row{index}:source_schedule_sha256")
        if row.get("trl_microstep") != (step or 0) * 8:
            generation_errors.append(f"row{index}:trl_microstep")
        count = row.get("generated_token_count")
        elapsed = row.get("elapsed_sec")
        generated_ids = row.get("generated_ids")
        if not isinstance(generated_ids, list) or not generated_ids or any(type(token) is not int for token in generated_ids):
            generation_errors.append(f"row{index}:generated_ids")
        elif len(generated_ids) != count:
            generation_errors.append(f"row{index}:generated_token_count")
        if not isinstance(row.get("generated_ids_sha256"), str) or len(row["generated_ids_sha256"]) != 64:
            generation_errors.append(f"row{index}:generated_ids_sha256")
        if not isinstance(row.get("prompt_ids_sha256"), str) or len(row["prompt_ids_sha256"]) != 64:
            generation_errors.append(f"row{index}:prompt_ids_sha256")
        if not isinstance(count, int) or count < 0 or count > 192:
            generation_errors.append(f"row{index}:generated_token_count")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            generation_errors.append(f"row{index}:elapsed_sec")
        if type(row.get("padded_after_terminal")) is not int or row["padded_after_terminal"] < 0:
            generation_errors.append(f"row{index}:padded_after_terminal")
        generated_hashes.append(row.get("generated_ids_sha256"))
        call_indices[geo.get("generation_call_index")] += 1
        group_indices[geo.get("group_index")] += 1
        terminal_counts[str(row.get("terminal_reason"))] += 1
    add_check(checks, "training.generation_geometry", "pass" if not generation_errors else "fail", generation_errors[:8] if generation_errors else {"calls": dict(call_indices), "groups": dict(group_indices)})

    sequence_error = None
    try:
        sequence = read_json(sequence_path)
        source_ids = sequence.get("row_ids", [])
        add_check(checks, "training.source_sequence.identity", "pass" if sequence.get("sequence_sha256") == EXPECTED["sequence"] and sequence.get("source_draws") == 24000 else "fail", {"sequence_sha256": sequence.get("sequence_sha256"), "source_draws": sequence.get("source_draws")})
    except Exception as exc:
        source_ids = []
        sequence_error = repr(exc)
        add_check(checks, "training.source_sequence.identity", "fail", sequence_error)

    reward_errors: list[str] = []
    reward_ids: list[str] = []
    reward_families: collections.Counter[str] = collections.Counter()
    reward_operations: collections.Counter[str] = collections.Counter()
    reward_expected_operations: collections.Counter[str] = collections.Counter()
    reward_failures: collections.Counter[str] = collections.Counter()
    reward_terminal = collections.Counter()
    unique_ids: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rewards):
        rid = row.get("id")
        reward_ids.append(rid)
        if index >= len(generated_hashes) or row.get("output_ids_sha256") != generated_hashes[index]:
            reward_errors.append(f"row{index}:output_ids_sha256 join")
        if index >= len(generations) or row.get("generated_tokens") != generations[index].get("generated_token_count"):
            reward_errors.append(f"row{index}:generated_tokens join")
        if row.get("schema_version") is not None and row.get("schema_version") != "sepalith.rl.reward-record.v1":
            reward_errors.append(f"row{index}:schema_version")
        for key in ("reward", "line_f1"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                reward_errors.append(f"row{index}:{key} nonfinite")
        reward_families[str(row.get("family"))] += 1
        reward_operations[str(row.get("operation"))] += 1
        reward_expected_operations[str(row.get("expected_operation"))] += 1
        if row.get("failure") is not None:
            reward_failures[str(row.get("failure"))] += 1
        reward_terminal["cap_hit" if row.get("cap_hit") else "eos"] += 1
        entry = unique_ids.setdefault(str(rid), {"rows": 0, "family": row.get("family"), "expected_operation": row.get("expected_operation"), "operations": set()})
        entry["rows"] += 1
        entry["operations"].add(row.get("operation"))
        if entry["family"] != row.get("family") or entry["expected_operation"] != row.get("expected_operation"):
            reward_errors.append(f"row{index}:candidate identity drift")

    contiguous = all(reward_ids[i] == reward_ids[i // 4 * 4] for i in range(len(reward_ids)))
    collapsed = reward_ids[::4]
    source_join = len(source_ids) >= len(collapsed) and collapsed == source_ids[: len(collapsed)]
    add_check(checks, "training.reward_candidate_contiguity", "pass" if contiguous and all(v["rows"] == 4 for v in unique_ids.values()) else "fail", {"unique_ids": len(unique_ids), "candidate_counts": collections.Counter(v["rows"] for v in unique_ids.values())})
    add_check(checks, "training.reward_source_order_join", "pass" if source_join else "fail", {"rows_joined": len(collapsed), "sequence_prefix": source_ids[:2], "reward_prefix": collapsed[:2]})
    add_check(checks, "training.generation_reward_join", "pass" if not reward_errors else "fail", reward_errors[:8] if reward_errors else {"rows": len(rewards)})

    gradient_path = run / "output" / "gradient-records.jsonl"
    try:
        gradients = parse_jsonl(gradient_path)
    except Exception as exc:
        gradients = []
        add_check(checks, "training.gradient_parse", "fail", repr(exc))
    gradient_errors: list[str] = []
    norms: list[float] = []
    for index, row in enumerate(gradients):
        norm = row.get("norm")
        if row.get("global_step") != index or row.get("step") != index or row.get("finite") is not True or row.get("nonfinite") is not False:
            gradient_errors.append(f"row{index}:step/finite")
        if row.get("trainable_tensor_count") != 588 or row.get("grad_present_count") != 588 or not isinstance(row.get("nonzero_tensor_count"), int) or row.get("nonzero_tensor_count") <= 0:
            gradient_errors.append(f"row{index}:tensor counts")
        if not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm <= 0:
            gradient_errors.append(f"row{index}:norm")
        else:
            norms.append(float(norm))
    add_check(checks, "training.gradient_records", "pass" if len(gradients) == 5 and not gradient_errors else "fail", {"records": len(gradients), "norms": norms, "errors": gradient_errors[:8]})

    return {
        "status": "pass" if len(generations) == 160 and len(rewards) == 160 and not generation_errors and not reward_errors and source_join and len(gradients) == 5 and not gradient_errors else "fail",
        "generations": len(generations),
        "rewards": len(rewards),
        "generation_step_counts": dict(step_counts),
        "terminal_reasons": dict(terminal_counts),
        "reward_terminal": dict(reward_terminal),
        "reward_families": dict(reward_families),
        "reward_operations": dict(reward_operations),
        "reward_expected_operations": dict(reward_expected_operations),
        "reward_failures": dict(reward_failures),
        "unique_source_ids": len(unique_ids),
        "gradient_records": len(gradients),
        "gradient_norms": norms,
    }


def verify_development(checks: list[dict[str, Any]], run: Path, panel_path: Path) -> dict[str, Any]:
    panel_sha = None
    panel_rows: list[dict[str, Any]] = []
    if panel_path.is_file():
        _, panel_sha = sha256_file(panel_path)
        try:
            panel_rows = parse_jsonl(panel_path)
        except Exception as exc:
            add_check(checks, "development.panel_parse", "fail", repr(exc))
    else:
        add_check(checks, "development.panel", "fail", "missing")
    panel_ids = [str(row.get("id")) for row in panel_rows]
    panel_ok = len(panel_rows) == 75 and len(set(panel_ids)) == 75 and panel_sha == EXPECTED["panel"]
    add_check(checks, "development.panel_identity", "pass" if panel_ok else "fail", {"rows": len(panel_rows), "sha256": panel_sha})
    panel_expected = collections.Counter(row.get("operation") for row in panel_rows)

    cases_path = run / "archive" / "evaluations" / "cases-step-5.json"
    if not cases_path.is_file():
        add_check(checks, "development.cases", "pending", "cases-step-5.json not present")
        return {"status": "pending", "cases": 0, "panel_sha256": panel_sha}
    try:
        cases = read_json(cases_path)
    except Exception as exc:
        add_check(checks, "development.cases_parse", "fail", repr(exc))
        return {"status": "fail", "cases": 0, "panel_sha256": panel_sha}
    results = cases.get("results", [])
    summary = cases.get("summary", {})
    status = cases.get("status")
    result_ids = [str(row.get("id")) for row in results if isinstance(row, dict)]
    complete = status == "complete" and len(results) == 75 and len(set(result_ids)) == 75 and result_ids == panel_ids
    if summary.get("panel_sha256") != EXPECTED["panel"]:
        complete = False
    if summary.get("denominators", {}).get("cases") != 75:
        complete = False
    add_check(checks, "development.cases_identity", "pass" if complete else ("pending" if status != "complete" or len(results) < 75 else "fail"), {"status": status, "step": cases.get("step"), "results": len(results), "panel_sha256": summary.get("panel_sha256")})

    counts = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    failure_counts: collections.Counter[str] = collections.Counter()
    max_prompt = 0
    long_case: dict[str, Any] | None = None
    for row in results:
        family_counts[str(row.get("family"))] += 1
        if row.get("cap_hit"):
            counts["cap_hit"] += 1
        if row.get("protocol_valid"):
            counts["protocol_valid"] += 1
        if row.get("predicted_noop"):
            counts["predicted_noop"] += 1
        if row.get("exact_region"):
            counts["exact_region"] += 1
        if row.get("failure") is not None:
            failure_counts[str(row.get("failure"))] += 1
        prompt_tokens = row.get("prompt_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens > max_prompt:
            max_prompt = prompt_tokens
        if row.get("id") == LONG_DEV_ID:
            long_case = {key: row.get(key) for key in ("id", "family", "prompt_tokens", "generated_tokens", "cap_hit", "protocol_valid", "exact_region", "failure", "predicted_noop")}

    expected_denominators = {"cases": 75, "edits": 43, "strict_noop": 32}
    denom = summary.get("denominators", {})
    denom_ok = all(denom.get(key) == value for key, value in expected_denominators.items())
    add_check(checks, "development.denominators", "pass" if denom_ok else ("pending" if not complete else "fail"), {key: denom.get(key) for key in expected_denominators})
    long_ok = long_case is not None and long_case.get("prompt_tokens") == 2619
    add_check(checks, "development.maximal_prompt_retained", "pass" if long_ok else ("pending" if not complete else "fail"), long_case)
    interpretation = str(summary.get("interpretation", ""))
    interpretation_ok = "no R semantic-validity" in interpretation and "claim" in interpretation.lower()
    add_check(checks, "development.interpretation", "pass" if interpretation_ok else "fail", summary.get("interpretation"))
    return {
        "status": "pass" if complete and denom_ok and long_ok else ("pending" if not complete else "fail"),
        "cases": len(results),
        "case_status": status,
        "panel_sha256": panel_sha,
        "counts": dict(counts),
        "family_counts": dict(family_counts),
        "failure_counts": dict(failure_counts),
        "summary_counts": summary.get("counts", {}),
        "denominators": {key: denom.get(key) for key in expected_denominators},
        "max_prompt_tokens": max_prompt,
        "long_case": long_case,
        "result_ids_exact_panel": result_ids == panel_ids,
        "expected_operation_counts": dict(panel_expected),
        "interpretation": summary.get("interpretation"),
    }


def verify_runtime(checks: list[dict[str, Any]], run: Path, recipe_path: Path) -> dict[str, Any]:
    recipe_sha = None
    if recipe_path.is_file():
        _, recipe_sha = sha256_file(recipe_path)
    recipe_ok = recipe_sha == EXPECTED["recipe"]
    add_check(checks, "runtime.recipe_sha256", "pass" if recipe_ok else "fail", {"expected": EXPECTED["recipe"], "actual": recipe_sha})

    supervision = run / "supervision.json"
    supervision_values = {key: mmap_scalar_values(supervision, key) for key in ("status", "recipe_sha256", "entry_result_receipt", "pid", "soft_deadline", "hard_deadline")}
    sup_recipe_ok = EXPECTED["recipe"] in supervision_values["recipe_sha256"]
    add_check(checks, "runtime.supervision_recipe_binding", "pass" if sup_recipe_ok else "fail", supervision_values["recipe_sha256"])
    # The preflight object contains many row-level status fields.  The
    # immutable supervisor status is the final top-level status in this
    # receipt's stable serialization.
    supervision_status = supervision_values["status"][-1] if supervision_values["status"] else None
    add_check(checks, "runtime.supervision_immutable_status", "pass" if supervision_status == "supervisor_configured; command acceptance and completion unverified" else "fail", supervision_status)
    pid_values = [value for value in supervision_values["pid"] if isinstance(value, int)]
    supervisor_pid = pid_values[0] if pid_values else None
    alive = is_pid_alive(supervisor_pid)

    entry_path = run / "supervision.entry-result.json"
    entry_values = {key: mmap_scalar_values(entry_path, key) for key in ("status", "step", "checkpoint", "supervision_receipt")}
    entry_status = "lead_decision" if "lead_decision" in entry_values["status"] else (entry_values["status"][-1] if entry_values["status"] else None)
    entry_step = 5 if 5 in entry_values["step"] else (entry_values["step"][-1] if entry_values["step"] else None)
    entry_ok = entry_status == "lead_decision" and entry_step == 5
    add_check(checks, "runtime.entry_result_distinct_receipt", "pass" if entry_ok else "pending", {"present": entry_path.is_file(), "status": entry_status, "step": entry_step, "entry_path": str(entry_path)})
    add_check(checks, "runtime.supervisor_pid_terminal", "pass" if entry_ok and not alive else "pending", {"supervisor_pid": supervisor_pid, "alive": alive})

    telemetry = run / "telemetry.jsonl"
    telemetry_events: collections.Counter[str] = collections.Counter()
    decision_steps: list[int] = []
    stop_reasons: list[dict[str, Any]] = []
    giant_identity_lines = 0
    if telemetry.is_file():
        with telemetry.open("rb") as fh:
            for raw in fh:
                if len(raw) > 2 * 1024 * 1024:
                    giant_identity_lines += 1
                    if b'"event": "train_begin"' in raw[:2048]:
                        telemetry_events["train_begin"] += 1
                    continue
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                event = row.get("event")
                if event:
                    telemetry_events[str(event)] += 1
                if event == "optimizer_step" and row.get("stop_reason") is not None:
                    stop_reasons.append({"step": row.get("step"), "stop_reason": row.get("stop_reason")})
                    if row.get("stop_reason") == "lead_decision":
                        decision_steps.append(row.get("step"))
    decision_ok = 5 in decision_steps
    add_check(checks, "runtime.telemetry.lead_decision_step5", "pass" if decision_ok else "pending", {"decision_steps": decision_steps, "stop_reasons": stop_reasons})
    return {
        "recipe_sha256": recipe_sha,
        "supervision": {"status": supervision_status, "recipe_sha256": supervision_values["recipe_sha256"], "pid": supervisor_pid, "alive": alive},
        "entry_result": {"path": str(entry_path), "status": entry_status, "step": entry_step, "present": entry_path.is_file()},
        "telemetry": {"events": dict(telemetry_events), "decision_steps": decision_steps, "stop_reasons": stop_reasons, "giant_identity_lines_skipped": giant_identity_lines},
        "terminal": bool(entry_ok and not alive and decision_ok),
    }


def verify_load_audit(checks: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    path = run / "output" / "load-audit.json"
    try:
        audit = read_json(path)
    except Exception as exc:
        add_check(checks, "load_audit.parse", "fail", repr(exc))
        return {"status": "fail"}
    expected = {
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
    failures = []
    for key, wanted in expected.items():
        actual = path_get(audit, key)
        if actual != wanted:
            failures.append({"key": key, "expected": wanted, "actual": actual})
    add_check(checks, "load_audit.identity", "pass" if not failures else "fail", {"failures": failures, "dtype": audit.get("dtype")})
    return {"status": "pass" if not failures else "fail", "dtype": audit.get("dtype"), "observed_capacity_tokens": path_get(audit, "model_load.observed_capacity_tokens"), "cuda_memory_fraction": path_get(audit, "cuda_allocator.cuda_allocator_fraction"), "attachments": path_get(audit, "adapter.attachments"), "trainable_parameters": path_get(audit, "adapter.trainable_parameters")}


def run_verifier(run: Path, recipe_path: Path, sequence_path: Path, panel_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checkpoint = verify_checkpoint(checks, run)
    training = verify_generation_and_rewards(checks, run, sequence_path)
    development = verify_development(checks, run, panel_path)
    runtime = verify_runtime(checks, run, recipe_path)
    load_audit = verify_load_audit(checks, run)
    full_checkpoint = checkpoint.get("status") == "pass"
    dev_complete = development.get("status") == "pass"
    training_records = training.get("status") == "pass"
    runtime_terminal = runtime.get("terminal") is True
    load_ok = load_audit.get("status") == "pass"
    overall = "pass_mechanical_quality_pending" if full_checkpoint and training_records and dev_complete and runtime_terminal and load_ok else "pending_or_failed"
    if any(item["status"] == "fail" for item in checks):
        # A pending development/terminal gate is expected while a live attempt
        # is in flight.  Hard artifact failures remain visible in checks and
        # make the overall status explicit.
        hard = any(item["status"] == "fail" and not item["name"].startswith("development.") for item in checks)
        if hard:
            overall = "fail_mechanical"
    result = {
        "schema_version": "sepalith.rl-07.full5-independent-review.v1",
        "status": overall,
        "run_root": str(run),
        "recipe_path": str(recipe_path),
        "sequence_path": str(sequence_path),
        "panel_path": str(panel_path),
        "expected": {"source_snapshot": EXPECTED["source"], "recipe_sha256": EXPECTED["recipe"], "identity_sha256": EXPECTED["identity"], "previous_light5_adapter_sha256": EXPECTED["previous_adapter"]},
        "checkpoint": checkpoint,
        "training": training,
        "development": development,
        "runtime": runtime,
        "load_audit": load_audit,
        "checks": checks,
        "unresolved": [
            "A lead_decision at step 5 and a full checkpoint are mechanical evidence only; no quality promotion is made by this verifier.",
            "The full5 run has no resume equality evidence. A same-identity continuation must compare adapter, optimizer, scheduler, RNG, sampler, source and output state independently.",
        ],
    }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_verifier(args.run_root, args.recipe, args.sequence, args.panel)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checkpoint": result["checkpoint"].get("status"), "training": result["training"].get("status"), "development": result["development"].get("status"), "terminal": result["runtime"].get("terminal"), "failed_checks": sum(item["status"] == "fail" for item in result["checks"])}, sort_keys=True))
    return 1 if result["status"] == "fail_mechanical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
