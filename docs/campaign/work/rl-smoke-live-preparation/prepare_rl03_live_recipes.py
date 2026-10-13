#!/usr/bin/env python3
"""Prepare, and only prepare, the two-update PRM-03 RL smoke recipes.

This tool has two deliberately separate phases.  The inspect phase reads the
already frozen DEV panels, loads a *tokenizer only*, and makes the small
evaluation panel plus a source-prefix audit.  The recipe phase additionally
requires a root-supplied admitted RL-02 receipt and an accepted merged-SFT
parent manifest.  It verifies those inputs and emits three recipes which all
carry the same seven-field training identity:

* uninterrupted control, two updates;
* a lead decision after update one; and
* a fresh attempt resumed from that update-one full checkpoint.

The source schedule remains the complete 24,000-draw v4 schedule.  The
two-update prefix is an audit aid, not a replacement schedule and not a new
quota.  No model, CUDA device, server, cloud service, launcher, or campaign
state is touched here.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ENTRY_SCHEMA_VERSION = "sepalith.prm07.rl-entry.v1"
TRAIN_SCHEMA_VERSION = "sepalith.prm07.rl-train.v1"
EVALUATOR_FACTORY = "campaign_eval:development_evaluator"
RENDERER_ID = "zeta2-prm03-v1"
TOKENIZATION_POLICY = (
    "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_"
    "terminal_eos1_final_lf_v1"
)
TERMINAL = ">>>>>>> UPDATED"
NO_EDIT = "[NO_EDIT]"
VOCAB_SIZE = 130560
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = [1, 130073]
MODEL_LOAD_MAX_SEQ_LENGTH = 4096
DEV_MAX_SEQUENCE_TOKENS = 4096
DEV_MAX_NEW_TOKENS = 512
RL_PROMPT_MAX_TOKENS = 2048
RL_COMPLETION_MAX_TOKENS = 192
RL_CONTEXT_MAX_TOKENS = 2240
SMOKE_UPDATES = 2
SMOKE_SOURCE_GROUPS_PER_UPDATE = 8
SMOKE_SOURCE_PREFIX_DRAWS = SMOKE_UPDATES * SMOKE_SOURCE_GROUPS_PER_UPDATE
SMOKE_MAX_ATTEMPT_SECONDS = 600
SMOKE_TERMINATION_GRACE_SECONDS = 60
SMOKE_CHECKPOINT_RESERVE_SECONDS = 120
LIVE_SMOKE_ATTEMPTS = 3
CUDA_MEMORY_FRACTION = 0.75
LIVE_DEADLINE = "2026-09-12T18:20:00Z"
EXPECTED_MAX_PROMPT_TOKENS = 2619
DEFAULT_DEV14 = "/mnt/e/sepalith/campaign-20260915/data-work/SFT-smoke-dev-14-v1.jsonl"
DEFAULT_DEV75 = "/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl"
DEFAULT_ROWS = "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/eligible-train-rows.jsonl"
DEFAULT_SIDECAR = "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/context-sidecar.jsonl"
DEFAULT_SELECTED = "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/selected-train-ids-lead-order-v1.json"
DEFAULT_SCHEDULE = str(
    Path(__file__).resolve().parents[3] / "campaign/work/rl-pool/source-row-draw-sequence-v4.json"
)
DEFAULT_TOKENIZER = "/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain"


class PreparationError(ValueError):
    """A preparation input or derived contract failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreparationError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def _verified_file(path: str | Path, expected: str | None, name: str) -> tuple[Path, str]:
    value = Path(path)
    if not value.is_absolute() or not value.is_file():
        raise PreparationError(f"{name} must be an absolute existing file: {value}")
    actual = _sha256_file(value)
    if expected is not None and actual != expected:
        raise PreparationError(f"{name} SHA256 mismatch: expected {expected}, observed {actual}")
    return value.resolve(), actual


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fresh_output(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise PreparationError(f"output-dir must be absolute: {value}")
    if value.exists() or value.is_symlink():
        raise PreparationError(f"output-dir must be fresh and must not already exist: {value}")
    value.mkdir(parents=True)
    return value


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read {name}: {path}") from error
    if not isinstance(value, Mapping):
        raise PreparationError(f"{name} must be a JSON object: {path}")
    return value


def _jsonl_bytes(path: Path, name: str) -> tuple[bytes, list[dict[str, Any]], dict[str, bytes]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PreparationError(f"cannot read {name}: {path}") from error
    records: list[dict[str, Any]] = []
    raw_by_id: dict[str, bytes] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise PreparationError(f"{name} has an empty line at {line_number}")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PreparationError(f"{name} has invalid JSON at line {line_number}") from error
        if not isinstance(value, dict):
            raise PreparationError(f"{name} line {line_number} is not an object")
        case_id = value.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise PreparationError(f"{name} line {line_number} has no nonempty id")
        if case_id in raw_by_id:
            raise PreparationError(f"{name} has duplicate id {case_id}")
        records.append(value)
        raw_by_id[case_id] = line.rstrip(b"\r\n") + b"\n"
    if not records:
        raise PreparationError(f"{name} is empty")
    return raw, records, raw_by_id


def _install_execution_paths(execution_root: Path) -> None:
    for relative in ("experiments/training", "packages/sepalith/src"):
        value = str((execution_root / relative).resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _inspect_panels(
    *,
    execution_root: Path,
    dev14_path: Path,
    dev14_sha256: str,
    dev75_path: Path,
    dev75_sha256: str,
    tokenizer_path: Path,
    output: Path,
    expected_max_prompt_tokens: int = EXPECTED_MAX_PROMPT_TOKENS,
) -> dict[str, Any]:
    """Inspect the complete DEV panel with a tokenizer, never a model."""
    actual14 = _sha256_file(dev14_path)
    if actual14 != dev14_sha256:
        raise PreparationError(f"development 14-case SHA256 mismatch: {actual14}")
    actual75 = _sha256_file(dev75_path)
    if actual75 != dev75_sha256:
        raise PreparationError(f"development 75-case SHA256 mismatch: {actual75}")
    dev14_raw, dev14_rows, dev14_raw_by_id = _jsonl_bytes(dev14_path, "development 14-case panel")
    _, dev75_rows, dev75_raw_by_id = _jsonl_bytes(dev75_path, "development 75-case panel")
    dev14_ids = [row["id"] for row in dev14_rows]
    dev75_ids = [row["id"] for row in dev75_rows]
    if len(dev14_rows) != 14 or len(dev75_rows) != 75:
        raise PreparationError(
            f"unexpected DEV panel sizes: 14-case={len(dev14_rows)}, 75-case={len(dev75_rows)}"
        )
    if any(row.get("split") != "dev" or not row.get("package_id") or not row.get("family")
           for row in dev14_rows + dev75_rows):
        raise PreparationError("every DEV panel row must retain split/package_id/family provenance")
    if not set(dev14_ids).issubset(dev75_ids):
        raise PreparationError("the 14-case panel contains an ID absent from the 75-case panel")

    _install_execution_paths(execution_root)
    try:
        from sepalith.campaign_protocol import PromptContext, encode_prompt
        from transformers import AutoTokenizer
    except Exception as error:  # pragma: no cover - environment-specific import
        raise PreparationError(f"tokenizer-only DEV inspection requires the canonical CPU venv: {error}") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, use_fast=True, trust_remote_code=False,
        )
    except Exception as error:  # pragma: no cover - environment-specific load
        raise PreparationError(f"cannot load tokenizer-only inspection input: {tokenizer_path}") from error

    prompt_lengths: dict[str, int] = {}
    for row in dev75_rows:
        try:
            context = PromptContext.from_mapping(row["context"])
            prompt_lengths[row["id"]] = len(encode_prompt(context, tokenizer))
        except Exception as error:
            raise PreparationError(f"cannot encode DEV context for {row['id']}: {error}") from error
    longest_id = max(prompt_lengths, key=prompt_lengths.get)
    longest_tokens = prompt_lengths[longest_id]
    if longest_tokens != expected_max_prompt_tokens:
        raise PreparationError(
            f"DEV maximum prompt changed: expected {expected_max_prompt_tokens}, observed {longest_tokens} ({longest_id})"
        )
    if longest_id not in dev14_ids:
        small_raw = dev14_raw
        if not small_raw.endswith((b"\n", b"\r")):
            small_raw += b"\n"
        small_raw += dev75_raw_by_id[longest_id]
        small_case_ids = dev14_ids + [longest_id]
        small_panel_size = 15
    else:
        # Preserve the frozen 14-case bytes when the maximal case is already
        # present; this currently is the no_op case ending in ...fa2a9be.
        small_raw = dev14_raw
        small_case_ids = dev14_ids
        small_panel_size = 14
    longest_row = next(row for row in dev75_rows if row["id"] == longest_id)
    target_body_tokens = longest_row.get("target_body_token_count")
    target_terminal_tokens = longest_row.get("target_terminal_token_count")
    if type(target_body_tokens) is not int or type(target_terminal_tokens) is not int:
        raise PreparationError("the longest DEV row lacks integer target token audit fields")
    full_sequence_tokens = longest_tokens + target_body_tokens + target_terminal_tokens + 1
    small_path = output / "development-panel-small.jsonl"
    small_path.write_bytes(small_raw)
    small_sha256 = _sha256_file(small_path)
    panel_audit = {
        "schema_version": "sepalith.rl03.two-update.dev-panel-audit.v1",
        "source_panel_14": {"path": str(dev14_path), "sha256": dev14_sha256, "rows": 14},
        "source_panel_75": {"path": str(dev75_path), "sha256": dev75_sha256, "rows": 75},
        "small_panel": {"path": str(small_path), "sha256": small_sha256, "rows": small_panel_size},
        "small_case_ids": small_case_ids,
        "max_prompt": {
            "case_id": longest_id,
            "tokens": longest_tokens,
            "full_sequence_tokens": full_sequence_tokens,
            "full_sequence_formula": "prompt + target_body + target_terminal + one separator",
        },
        "prompt_lengths": prompt_lengths,
        "tokenizer": {
            "path": str(tokenizer_path),
            "tokenizer_json_sha256": _sha256_file(tokenizer_path / "tokenizer.json"),
            "tokenizer_config_sha256": _sha256_file(tokenizer_path / "tokenizer_config.json"),
            "inspection": "AutoTokenizer only; no model or CUDA load",
        },
        "target_is_not_used": True,
    }
    _write_json(output / "development-panel-audit.json", panel_audit)
    return panel_audit


def _source_prefix(
    *,
    schedule_path: Path,
    schedule_sha256: str,
    selected_ids_path: Path | None,
    selected_ids_sha256: str | None,
    output: Path,
) -> dict[str, Any]:
    """Audit the first two update groups without changing the full schedule."""
    schedule = _json_object(schedule_path, "source draw schedule")
    if schedule.get("schema_version") != "sepalith.dat09.source-row-draw-sequence.v1":
        raise PreparationError("source draw schedule schema is not the admitted v1 contract")
    row_ids = schedule.get("row_ids")
    if not isinstance(row_ids, list) or any(not isinstance(value, str) or not value for value in row_ids):
        raise PreparationError("source draw schedule row_ids must be a nonempty string list")
    required = {
        "buffer_reuse": 4,
        "candidate_count": 4,
        "completions_per_update": 32,
        "gradient_accumulation_steps": 4,
        "prompt_groups_per_update": 8,
        "source_draws": len(row_ids),
        "steps_per_generation": 4,
    }
    for name, expected in required.items():
        if schedule.get(name) != expected:
            raise PreparationError(f"source schedule {name} must be {expected}, observed {schedule.get(name)!r}")
    if len(row_ids) % SMOKE_SOURCE_GROUPS_PER_UPDATE or len(row_ids) != 24000:
        raise PreparationError("the v4 source schedule must contain complete 24,000 draws")
    sequence_payload = json.dumps(row_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sequence_sha256 = hashlib.sha256(sequence_payload).hexdigest()
    if schedule.get("sequence_sha256") != sequence_sha256:
        raise PreparationError("source schedule sequence_sha256 does not match row_ids")
    if _sha256_file(schedule_path) != schedule_sha256:
        raise PreparationError("source schedule bytes changed during prefix inspection")
    selected_sha256 = None
    selected_ids: list[str] | None = None
    if selected_ids_path is not None:
        selected_value = _json_object(selected_ids_path, "selected train IDs")
        if (selected_value.get("schema_version") != "sepalith.prm07.selected-train-ids.v1"
                or selected_value.get("split") != "train"):
            raise PreparationError("selected IDs schema/split is not the admitted train contract")
        selected_ids = selected_value.get("row_ids")
        if (not isinstance(selected_ids, list) or len(selected_ids) != 8440
                or len(set(selected_ids)) != len(selected_ids)
                or any(not isinstance(value, str) or not value for value in selected_ids)):
            raise PreparationError("selected IDs must be the unique 8,440-row RL-02 order")
        selected_sha256 = _sha256_file(selected_ids_path)
        if selected_ids_sha256 is not None and selected_sha256 != selected_ids_sha256:
            raise PreparationError("selected IDs SHA256 mismatch")
        selected_set = set(selected_ids)
        if any(value not in selected_set for value in row_ids):
            raise PreparationError("source schedule contains a row outside selected IDs")
    prefix_ids = row_ids[:SMOKE_SOURCE_PREFIX_DRAWS]
    if len(prefix_ids) != SMOKE_SOURCE_PREFIX_DRAWS:
        raise PreparationError("source schedule is shorter than the two-update prefix")
    prefix = {
        "schema_version": "sepalith.rl03.two-update.source-prefix-audit.v1",
        "schedule": {
            "path": str(schedule_path),
            "sha256": schedule_sha256,
            "sequence_sha256": sequence_sha256,
            "source_draws": len(row_ids),
            "source_draws_per_update": SMOKE_SOURCE_GROUPS_PER_UPDATE,
            "updates_audited": SMOKE_UPDATES,
        },
        "selected_ids": ({"path": str(selected_ids_path), "sha256": selected_sha256, "rows": len(selected_ids)}
                          if selected_ids is not None else None),
        "prefix_draws": len(prefix_ids),
        "prefix_row_ids": prefix_ids,
        "groups": [prefix_ids[index:index + SMOKE_SOURCE_GROUPS_PER_UPDATE]
                    for index in range(0, len(prefix_ids), SMOKE_SOURCE_GROUPS_PER_UPDATE)],
        "accounting": {
            "candidate_count": 4,
            "completions_per_update": 32,
            "buffer_reuse": 4,
            "sampler_rows_per_update": 128,
            "sampler_rows_two_updates": 256,
            "source_draws_two_updates": 16,
        },
        "quota_contract": "The prefix is an audit view; recipes bind the complete 24,000-draw schedule and its hashes.",
    }
    _write_json(output / "source-prefix-audit.json", prefix)
    return prefix


def _contains(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return value == needle
    if isinstance(value, Mapping):
        return any(_contains(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, needle) for item in value)
    return False


def _verify_rl02_admission(path: Path, expected_hashes: Sequence[str]) -> Mapping[str, Any]:
    receipt = _json_object(path, "RL-02 admission receipt")
    status = str(receipt.get("status", "")).lower()
    positive = receipt.get("launch_admitted") is True or status in {
        "admitted", "accepted", "rl02_admitted", "source_admitted", "launch_admitted",
    }
    if not positive or any(word in status for word in ("candidate", "pending", "blocked", "false")):
        raise PreparationError(
            "RL-02 receipt is not an explicit admitted receipt; candidate/review receipts cannot be promoted"
        )
    missing = [digest for digest in expected_hashes if not _contains(receipt, digest)]
    if missing:
        raise PreparationError(
            "RL-02 admission receipt does not bind all supplied input hashes: " + ", ".join(missing)
        )
    return receipt


def _identity_and_recipes(
    *,
    args: argparse.Namespace,
    output: Path,
    parent_audit: Mapping[str, Any],
    admission_receipt_sha256: str,
    admission_receipt: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    source_prefix: Mapping[str, Any],
    panel_audit: Mapping[str, Any],
    schedule_sha256: str,
    protocol_sha256: str,
    trainer_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    parent_identity = {
        "kind": "merged_sft",
        "manifest_sha256": parent_audit["manifest_sha256"],
        "merged_weights_sha256": parent_audit["merged_weights_sha256"],
        "base_model_revision": parent_audit["base_model_revision"],
        "sft_identity": parent_audit["sft_identity"],
    }
    parent_tokenizer = parent_audit["tokenizer"]
    tokenizer_identity = {
        "revision": args.tokenizer_revision,
        "json_sha256": parent_tokenizer["tokenizer_json_sha256"],
        "config_sha256": parent_tokenizer["tokenizer_config_sha256"],
        "tokenizer_json_sha256": parent_tokenizer["tokenizer_json_sha256"],
        "tokenizer_config_sha256": parent_tokenizer["tokenizer_config_sha256"],
        "vocab_size": VOCAB_SIZE,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": EOS_ID,
        "native_eog_ids": NATIVE_EOG_IDS,
    }
    renderer_identity = {
        "schema_version": "sepalith.prompt.prm03.v1",
        "renderer_id": RENDERER_ID,
        "tokenization_policy": TOKENIZATION_POLICY,
        "terminal": TERMINAL,
        "no_edit": NO_EDIT,
    }
    source_identity = {
        "frozen_source_root": str(args.execution_root.resolve()),
        "protocol_sha256": protocol_sha256,
        "trainer_sha256": trainer_sha256,
        "trl_version": "0.24.0",
        "source_schedule_sha256": schedule_sha256,
        "rl02_admission_receipt_sha256": admission_receipt_sha256,
        "rl02_admission_status": admission_receipt.get("status", "admitted"),
    }
    sampling = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }
    policy_identity = {
        "cuda_memory_fraction": CUDA_MEMORY_FRACTION,
        "lora_rank": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "expected_attachments": 294,
        "expected_trainable_parameters": 25116672,
        "model_load_max_seq_length": MODEL_LOAD_MAX_SEQ_LENGTH,
        "prompt_max_tokens": RL_PROMPT_MAX_TOKENS,
        "completion_max_tokens": RL_COMPLETION_MAX_TOKENS,
        "context_max_tokens": RL_CONTEXT_MAX_TOKENS,
        "candidate_count": 4,
        "rollout_rows_per_update": 32,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "loss_type": "bnpo",
        "scale_rewards": "group",
        "beta": 0,
        "sampling": sampling,
    }
    schedule_identity = {
        "seed": 3407,
        "sampler_id": "campaign-repeat-manifest-order-v1",
        "generation_batch_size": 32,
        "steps_per_generation": 4,
        "num_iterations": 1,
        "source_draw_schedule_sha256": schedule_sha256,
        "source_draw_sequence_sha256": source_prefix["schedule"]["sequence_sha256"],
        "source_draws": 24000,
        "source_draws_per_update": 8,
        "buffer_reuse": 4,
        "smoke_updates": SMOKE_UPDATES,
        "main_update_ceiling": 3000,
    }
    identity = {
        "parent": parent_identity,
        "tokenizer": tokenizer_identity,
        "renderer": renderer_identity,
        "data": dict(data_identity),
        "source": source_identity,
        "policy": policy_identity,
        "schedule": schedule_identity,
    }
    small_panel = panel_audit["small_panel"]
    common = {
        "entry_schema_version": ENTRY_SCHEMA_VERSION,
        "schema_version": TRAIN_SCHEMA_VERSION,
        "identity": identity,
        "cuda_memory_fraction": CUDA_MEMORY_FRACTION,
        "parent_manifest": {"path": args.parent_manifest, "sha256": args.parent_manifest_sha256},
        "rl02_admission": {"path": args.rl02_admission_receipt, "sha256": admission_receipt_sha256},
        "model_load_max_seq_length": MODEL_LOAD_MAX_SEQ_LENGTH,
        "data": {
            "rows_path": args.rows_path,
            "rows_sha256": args.rows_sha256,
            "sidecar_path": args.sidecar_path,
            "sidecar_artifact_sha256": args.sidecar_sha256,
            "context_sha256": args.sidecar_sha256,
            "selected_ids_path": args.selected_ids_path,
            "selected_ids_sha256": args.selected_ids_sha256,
            "admission_status": "admitted",
            "source_draw_schedule_path": args.source_schedule_path,
            "source_draw_schedule_sha256": schedule_sha256,
            "source_draw_sequence_sha256": source_prefix["schedule"]["sequence_sha256"],
            "source_draws": 24000,
            "buffer_reuse": 4,
        },
        "evaluator_factory": EVALUATOR_FACTORY,
        "renderer_id": RENDERER_ID,
        "parameters": {"max_sequence_tokens": DEV_MAX_SEQUENCE_TOKENS},
        "development_panel": {"path": small_panel["path"], "sha256": small_panel["sha256"]},
        "development_case_ids": panel_audit["small_case_ids"],
        "development_max_new_tokens": DEV_MAX_NEW_TOKENS,
        "main_development_panel": {
            "path": args.dev75_path,
            "sha256": args.dev75_sha256,
            "rows": 75,
            "required_for_main": True,
        },
        "generation_kwargs": sampling,
        "max_steps": SMOKE_UPDATES,
        "full_save_steps": 1,
        "light_save_steps": 1,
        "evaluation_steps": [1, 2],
        "checkpoint_reserve_seconds": SMOKE_CHECKPOINT_RESERVE_SECONDS,
        "deadline": args.deadline,
        "max_attempt_seconds": SMOKE_MAX_ATTEMPT_SECONDS,
        "termination_grace_seconds": SMOKE_TERMINATION_GRACE_SECONDS,
        "learning_rate": 2e-5,
        "warmup_steps": 0,
        "allow_pending_evaluation": False,
    }
    run_root = Path(args.run_root).resolve()
    if not run_root.is_absolute():
        raise PreparationError("run-root must be absolute")
    specs = {
        "uninterrupted": {"decision_steps": [], "resume_from": None},
        "split-first": {"decision_steps": [1], "resume_from": None},
        "split-resume": {
            "decision_steps": [],
            "resume_from": str(run_root / "split-first" / "archive" / "full" / "checkpoint-1"),
        },
    }
    recipes: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        recipe = dict(common)
        recipe.update({
            "id": f"rl03-two-update-{name}-v1",
            "decision_steps": spec["decision_steps"],
            "resume_from": spec["resume_from"],
            "output_dir": str(run_root / name / "output"),
            "archive_root": str(run_root / name / "archive"),
            "telemetry_path": str(run_root / name / "telemetry.jsonl"),
        })
        recipes[name] = recipe
    return identity, recipes


def _require_normal_args(args: argparse.Namespace) -> None:
    required = (
        "parent_manifest", "parent_manifest_sha256", "rl02_admission_receipt",
        "rl02_admission_receipt_sha256",
        "rows_path", "rows_sha256", "sidecar_path", "sidecar_sha256",
        "selected_ids_path", "selected_ids_sha256", "source_schedule_path",
        "run_root", "deadline", "protocol_sha256", "trainer_sha256", "tokenizer_revision",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise PreparationError("recipe generation requires explicit root inputs: " + ", ".join(missing))
    try:
        deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise PreparationError("deadline must be an ISO-8601 UTC timestamp") from error
    if deadline.tzinfo is None:
        raise PreparationError("deadline must include an explicit timezone")
    if args.deadline != LIVE_DEADLINE:
        raise PreparationError(
            f"RL-03 live recipe deadline must be the authorized fixed deadline {LIVE_DEADLINE}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dev14-path", default=DEFAULT_DEV14)
    parser.add_argument("--dev14-sha256", default="9192bd278c0342ce94d029ee5b1b9964b3fce0048ed6b99e4bcc0d97aaf06b59")
    parser.add_argument("--dev75-path", default=DEFAULT_DEV75)
    parser.add_argument("--dev75-sha256", default="b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21")
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--expected-max-prompt-tokens", type=int, default=EXPECTED_MAX_PROMPT_TOKENS)
    parser.add_argument("--selected-ids-path", default=DEFAULT_SELECTED)
    parser.add_argument("--selected-ids-sha256", default="24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d")
    parser.add_argument("--source-schedule-path", default=DEFAULT_SCHEDULE)
    parser.add_argument("--source-schedule-sha256", default="892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48")
    parser.add_argument("--rows-path", default=DEFAULT_ROWS)
    parser.add_argument("--rows-sha256", default="e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602")
    parser.add_argument("--sidecar-path", default=DEFAULT_SIDECAR)
    parser.add_argument("--sidecar-sha256", default="6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f")
    parser.add_argument("--parent-manifest")
    parser.add_argument("--parent-manifest-sha256")
    parser.add_argument("--rl02-admission-receipt")
    parser.add_argument("--rl02-admission-receipt-sha256")
    parser.add_argument("--run-root")
    parser.add_argument("--deadline")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--trainer-sha256")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--inspect-only", action="store_true", help="inspect DEV panels and source prefix; emit no recipes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        execution_root = args.execution_root.resolve()
        if not execution_root.is_dir():
            raise PreparationError(f"execution-root is not a directory: {execution_root}")
        output = _fresh_output(args.output_dir)
        dev14_path, dev14_sha = _verified_file(args.dev14_path, args.dev14_sha256, "dev14-path")
        dev75_path, dev75_sha = _verified_file(args.dev75_path, args.dev75_sha256, "dev75-path")
        tokenizer_path = Path(args.tokenizer_path).resolve()
        if not tokenizer_path.is_dir():
            raise PreparationError(f"tokenizer-path must be an existing local directory: {tokenizer_path}")
        panel_audit = _inspect_panels(
            execution_root=execution_root,
            dev14_path=dev14_path,
            dev14_sha256=dev14_sha,
            dev75_path=dev75_path,
            dev75_sha256=dev75_sha,
            tokenizer_path=tokenizer_path,
            output=output,
            expected_max_prompt_tokens=args.expected_max_prompt_tokens,
        )
        schedule_path, schedule_sha = _verified_file(
            args.source_schedule_path, args.source_schedule_sha256, "source-schedule-path",
        )
        selected_path, selected_sha = _verified_file(
            args.selected_ids_path, args.selected_ids_sha256, "selected-ids-path",
        )
        source_prefix = _source_prefix(
            schedule_path=schedule_path,
            schedule_sha256=schedule_sha,
            selected_ids_path=selected_path,
            selected_ids_sha256=selected_sha,
            output=output,
        )
        if args.inspect_only:
            _write_json(output / "preparation-result.json", {
                "status": "inspection_complete_recipes_not_emitted",
                "panel_audit": panel_audit,
                "source_prefix": source_prefix,
                "framework": "tokenizer-only inspection; no model/CUDA/launch/state",
            })
            print(json.dumps({"status": "inspection_complete", "output_dir": str(output)}, sort_keys=True))
            return 0

        _require_normal_args(args)
        parent_path, parent_sha = _verified_file(args.parent_manifest, args.parent_manifest_sha256, "parent-manifest")
        admission_path, admission_sha = _verified_file(
            args.rl02_admission_receipt,
            args.rl02_admission_receipt_sha256,
            "rl02-admission-receipt",
        )
        admission_hashes = [
            args.rows_sha256, args.sidecar_sha256, args.selected_ids_sha256,
            schedule_sha, source_prefix["schedule"]["sequence_sha256"],
        ]
        admission = _verify_rl02_admission(admission_path, admission_hashes)
        _install_execution_paths(execution_root)
        try:
            from campaign_checkpoint import check_identity
            from campaign_rl_data import load_training_records
            from campaign_rl_entry import verify_merged_parent_manifest
            from campaign_rl_train import load_source_draw_schedule
        except Exception as error:  # pragma: no cover - import environment-specific
            raise PreparationError(f"framework-free RL preparation imports failed: {error}") from error
        parent_audit = verify_merged_parent_manifest({"path": str(parent_path), "sha256": parent_sha})
        inspected_tokenizer = panel_audit["tokenizer"]
        parent_tokenizer = parent_audit["tokenizer"]
        for field in ("tokenizer_json_sha256", "tokenizer_config_sha256"):
            if parent_tokenizer.get(field) != inspected_tokenizer.get(field):
                raise PreparationError(
                    f"accepted parent tokenizer {field} differs from the tokenizer used for DEV inspection"
                )
        rows_path, rows_sha = _verified_file(args.rows_path, args.rows_sha256, "rows-path")
        sidecar_path, sidecar_sha = _verified_file(args.sidecar_path, args.sidecar_sha256, "sidecar-path")
        selected_path, selected_sha = _verified_file(args.selected_ids_path, args.selected_ids_sha256, "selected-ids-path")
        records, manifest = load_training_records(
            rows_path, rows_sha, sidecar_path, sidecar_sha,
            selected_path, selected_sha, admission_status="admitted",
        )
        data_identity = manifest.to_identity()
        if data_identity.get("row_count") != 8440 or len(records) != 8440:
            raise PreparationError(f"RL-02 admitted data count changed: records={len(records)}")
        loaded_schedule = load_source_draw_schedule(
            schedule_path,
            schedule_sha,
            selected_ids=manifest.selected_ids,
            selected_ids_sha256=manifest.selected_ids_sha256,
            ordered_ids_sha256=manifest.ordered_ids_sha256,
            row_identity_sha256=manifest.row_identity_sha256,
            candidate_count=4,
            source_draws_per_update=8,
            buffer_reuse=4,
        )
        if loaded_schedule["sequence_sha256"] != source_prefix["schedule"]["sequence_sha256"]:
            raise PreparationError("loaded source schedule differs from the inspected source prefix")
        protocol_path = execution_root / "packages/sepalith/src/sepalith/campaign_protocol.py"
        trainer_path = execution_root / "experiments/training/campaign_rl_train.py"
        if _sha256_file(protocol_path) != args.protocol_sha256:
            raise PreparationError("protocol source SHA256 differs from supplied identity")
        if _sha256_file(trainer_path) != args.trainer_sha256:
            raise PreparationError("campaign_rl_train source SHA256 differs from supplied identity")
        identity, recipes = _identity_and_recipes(
            args=args,
            output=output,
            parent_audit=parent_audit,
            admission_receipt_sha256=admission_sha,
            admission_receipt=admission,
            data_identity=data_identity,
            source_prefix=source_prefix,
            panel_audit=panel_audit,
            schedule_sha256=schedule_sha,
            protocol_sha256=args.protocol_sha256,
            trainer_sha256=args.trainer_sha256,
        )
        # This checks the constructed identity's JSON shape without importing
        # TRL and makes malformed future recipe contracts fail in preparation.
        check_identity(identity)
        recipe_paths: dict[str, str] = {}
        for name, recipe in recipes.items():
            path = output / f"rl-two-update-{name}.recipe.json"
            _write_json(path, recipe)
            recipe_paths[name] = str(path)
        result = {
            "status": "recipes_emitted_pending_live_smoke",
            "live_launch_caps": {
                "attempts": LIVE_SMOKE_ATTEMPTS,
                "max_attempt_seconds_each": SMOKE_MAX_ATTEMPT_SECONDS,
                "absolute_deadline": args.deadline,
                "cuda_memory_fraction": CUDA_MEMORY_FRACTION,
                "launch_performed": False,
            },
            "recipe_paths": recipe_paths,
            "identity_sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "data": {"rows": len(records), "identity": data_identity},
            "parent": parent_audit,
            "admission_receipt_sha256": admission_sha,
            "panel_audit": panel_audit,
            "source_prefix": source_prefix,
            "live_launch": "not performed by this tool; root must run the launcher after theta0 admission",
        }
        _write_json(output / "preparation-result.json", result)
        print(json.dumps({"status": result["status"], "output_dir": str(output), "recipes": recipe_paths}, sort_keys=True))
        return 0
    except PreparationError as error:
        print(f"prepare_rl_two_update_smoke: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
