#!/usr/bin/env python3
"""Bounded PRM-07 SFT length/memory profile preparation.

The command has two explicit modes.  ``--preflight`` verifies an immutable
candidate-token JSONL file, a separate immutable train-row ID manifest, and
the pinned MiniCPM artifacts without importing a model framework.  ``--profile``
repeats those checks, performs a live process/VRAM guard, and only then imports
Unsloth and torch for the disposable measurement.

The candidate file is deliberately not discovered from a directory.  It must
be supplied with its SHA256, and the selected train IDs must be supplied in a
second hashed JSON file.  Every selected row is a complete PRM-03 training row;
its input_ids are measured as stored.  Dynamic padding uses EOS ID 1, but
padding positions have attention 0 and label -100, so padding is never counted
as useful sequence or loss tokens.

This module does not duplicate the SFT trainer.  The live path reuses the
current LoRA attachment policy and only runs forward/backward/optimizer steps
on the selected rows.  Real RL generation and recomputed policy-logprob
training remain a separate GPU gate; the pure accounting validator below keeps
that handoff explicit and fail-closed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence


EXECUTION_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SRC = EXECUTION_ROOT / "packages" / "sepalith" / "src"
if str(PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_SRC))

from sepalith.campaign_protocol import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    NATIVE_EOG_IDS,
    TOKENIZER_CONFIG_SHA256 as PROTOCOL_TOKENIZER_CONFIG_SHA256,
    TOKENIZER_JSON_SHA256 as PROTOCOL_TOKENIZER_JSON_SHA256,
    TOKENIZER_REVISION as PROTOCOL_TOKENIZER_REVISION,
    VOCAB_SIZE,
    is_native_control_token,
    validate_training_row,
)
from campaign_tokenizer_contract import (  # noqa: E402
    TokenizerContractError,
    load_pinned_reference_tokenizer,
    restore_pinned_tokenizer_contract,
)


PROFILE_SCHEMA_VERSION = "sepalith.prm07.profile.v1"
CANDIDATE_SCHEMA_VERSION = "sepalith.prm07.candidate-token-rows.v1"
SELECTED_IDS_SCHEMA_VERSION = "sepalith.prm07.selected-train-ids.v1"

EXPECTED_MODEL_PATH = Path(
    "/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain"
)
EXPECTED_MODEL_REVISION = PROTOCOL_TOKENIZER_REVISION
EXPECTED_WEIGHT_SHA256 = "38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad"
EXPECTED_WEIGHT_BYTES = 5_033_557_128
EXPECTED_TOKENIZER_JSON_SHA256 = PROTOCOL_TOKENIZER_JSON_SHA256
EXPECTED_TOKENIZER_CONFIG_SHA256 = PROTOCOL_TOKENIZER_CONFIG_SHA256
EXPECTED_CONFIG_SHA256 = "59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180"
EXPECTED_GENERATION_CONFIG_SHA256 = "9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1"

SHORT_MAX_TOKENS = 2_048
LONG_MAX_TOKENS = 4_096
RL_PROMPT_MAX_TOKENS = 2_048
RL_COMPLETION_MAX_TOKENS = 192
DEFAULT_WALL_TIMEOUT_SECONDS = 1_800.0
DEFAULT_MAX_EXISTING_VRAM_MIB = 1_024.0
DEFAULT_WARMUP_STEPS = 1
DEFAULT_TIMED_STEPS = 3
DEFAULT_BATCH_SIZE = 1

# Keep this policy textually aligned with campaign_sft.py.  The live loader
# imports TARGET_MODULES after the pre-import guard and checks equality again.
SFT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
SFT_LORA_RANK = 32
SFT_LORA_ALPHA = 64
SFT_EXPECTED_ATTACHMENTS = 294
SFT_EXPECTED_TRAINABLE_PARAMETERS = 50_233_344
SFT_LEARNING_RATE = 2e-4
SFT_PAD_TOKEN_ID = EOS_ID

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProfileInputError(ValueError):
    """An immutable profile input failed closed validation."""


class ProfileGuardError(RuntimeError):
    """A live profile cannot safely claim the device/process lease."""


class ProfileTimeout(RuntimeError):
    """The profile exceeded its outer wall-clock bound."""


@dataclass(frozen=True)
class ProfileInput:
    candidate_file: Path
    candidate_sha256: str
    selected_ids_file: Path
    selected_ids_sha256: str
    model_path: Path = EXPECTED_MODEL_PATH


@dataclass(frozen=True)
class Deadline:
    started: float
    seconds: float

    def check(self) -> None:
        if time.monotonic() - self.started >= self.seconds:
            raise ProfileTimeout(f"profile exceeded {self.seconds:g} second wall-clock bound")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProfileInputError(f"{name} must be a lowercase SHA256")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ProfileInputError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected: str, label: str) -> None:
    expected = _validate_sha256(expected, label)
    if not path.is_absolute() or not path.is_file():
        raise ProfileInputError(f"{label} path must be an absolute file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ProfileInputError(f"{label} SHA256 mismatch: {path}")


def load_selected_train_ids(path: Path, expected_sha256: str) -> tuple[str, ...]:
    """Read the explicit train-only ID manifest; never infer a split."""
    verify_file_sha256(path, expected_sha256, "selected_train_ids")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileInputError(f"selected_train_ids is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "split", "row_ids"}:
        raise ProfileInputError("selected_train_ids must contain exactly schema_version, split, row_ids")
    if value["schema_version"] != SELECTED_IDS_SCHEMA_VERSION or value["split"] != "train":
        raise ProfileInputError("selected_train_ids schema or split is not the PRM-07 train contract")
    row_ids = value["row_ids"]
    if not isinstance(row_ids, list) or not row_ids:
        raise ProfileInputError("selected_train_ids.row_ids must be a nonempty array")
    if any(type(row_id) is not str or not row_id for row_id in row_ids):
        raise ProfileInputError("selected_train_ids.row_ids must contain nonempty strings")
    if len(set(row_ids)) != len(row_ids):
        raise ProfileInputError("selected_train_ids.row_ids must be unique")
    return tuple(row_ids)


def _validate_candidate_row(value: object, line_number: int) -> tuple[dict[str, Any], str]:
    entry_kind = "protocol_training_row"
    audit_lengths: Mapping[str, Any] | None = None
    if isinstance(value, dict) and "row" in value:
        # DAT-04 pilot output is intentionally a candidate-only envelope.  It
        # must not be mistaken for a training-registry admission just because
        # its nested row is structurally valid.
        if value.get("status") != "tokenizer_candidate_only" or value.get("admitted_for_training") is not False:
            raise ProfileInputError(
                f"candidate row {line_number} is not an unadmitted tokenizer-candidate envelope"
            )
        source_ref = value.get("source_ref")
        if not isinstance(source_ref, dict) or source_ref.get("split") != "train_group":
            raise ProfileInputError(
                f"candidate row {line_number} does not carry a train-group source reference"
            )
        lengths = value.get("lengths")
        if not isinstance(lengths, dict):
            raise ProfileInputError(f"candidate row {line_number} is missing its length audit")
        audit_lengths = lengths
        entry_kind = "tokenizer_candidate_only"
        value = value["row"]
    try:
        row = validate_training_row(value)
    except (TypeError, ValueError, KeyError) as error:
        raise ProfileInputError(f"candidate row {line_number} failed PRM-03 validation: {error}") from error
    if row["split"] != "train":
        raise ProfileInputError(
            f"candidate row {row['id']} has split {row['split']!r}; dev/final rows are forbidden"
        )
    ids = row["input_ids"]
    if len(ids) < 3:
        raise ProfileInputError(f"candidate row {row['id']} is too short to profile")
    if ids[0] != BOS_ID or ids[-1] != EOS_ID:
        raise ProfileInputError(f"candidate row {row['id']} has invalid BOS/EOS boundaries")
    if entry_kind == "tokenizer_candidate_only":
        # Length metadata is evidence, not the length source.  Reject stale
        # self-consistent metadata rather than profiling a mislabeled bucket.
        assert audit_lengths is not None
        expected = {
            "prompt_with_bos": int(row["prompt_token_count"]) + 1,
            "prompt_without_bos": int(row["prompt_token_count"]),
            "target_body": len(row["target_body_tokens"]),
            "response_with_terminal_eos": int(row["target_token_count"]) + 1,
            "sequence": len(row["input_ids"]),
        }
        for name, actual in expected.items():
            if audit_lengths.get(name) != actual:
                raise ProfileInputError(
                    f"candidate row {row['id']} length audit mismatch for {name}"
                )
    return row, entry_kind


def load_profile_rows(profile_input: ProfileInput) -> tuple[tuple[str, ...], list[dict[str, Any]], dict[str, Any]]:
    """Verify the candidate registry and return rows in selected-ID order."""
    selected_ids = load_selected_train_ids(
        profile_input.selected_ids_file, profile_input.selected_ids_sha256,
    )
    verify_file_sha256(profile_input.candidate_file, profile_input.candidate_sha256, "candidate_file")
    rows_by_id: dict[str, dict[str, Any]] = {}
    entry_kinds: dict[str, str] = {}
    total_rows = 0
    try:
        stream = profile_input.candidate_file.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ProfileInputError(f"cannot open candidate_file: {profile_input.candidate_file}") from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ProfileInputError(f"candidate_file contains a blank line at {line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProfileInputError(f"candidate_file line {line_number} is not JSON") from error
            row, entry_kind = _validate_candidate_row(raw, line_number)
            row_id = row["id"]
            if row_id in rows_by_id:
                raise ProfileInputError(f"candidate_file repeats row id {row_id}")
            rows_by_id[row_id] = row
            entry_kinds[row_id] = entry_kind
            total_rows += 1
    missing = [row_id for row_id in selected_ids if row_id not in rows_by_id]
    if missing:
        raise ProfileInputError(f"selected train IDs are absent from candidate_file: {missing[:4]}")
    selected = [rows_by_id[row_id] for row_id in selected_ids]
    buckets = {"short": 0, "long": 0, "out_of_profile": 0}
    for row in selected:
        length = len(row["input_ids"])
        if length <= SHORT_MAX_TOKENS:
            buckets["short"] += 1
        elif length <= LONG_MAX_TOKENS:
            buckets["long"] += 1
        else:
            buckets["out_of_profile"] += 1
    if buckets["out_of_profile"]:
        raise ProfileInputError(
            f"{buckets['out_of_profile']} selected rows exceed the natural {LONG_MAX_TOKENS}-token profile ceiling"
        )
    summary = {
        "candidate_file": str(profile_input.candidate_file),
        "candidate_sha256": profile_input.candidate_sha256,
        "selected_ids_file": str(profile_input.selected_ids_file),
        "selected_ids_sha256": profile_input.selected_ids_sha256,
        "candidate_row_count": total_rows,
        "selected_row_count": len(selected),
        "selected_row_ids": list(selected_ids),
        "natural_buckets": buckets,
        "entry_kinds": {
            kind: sum(value == kind for value in entry_kinds.values())
            for kind in sorted(set(entry_kinds.values()))
        },
        "selection": "explicit hashed train IDs; no directory or split discovery",
    }
    return selected_ids, selected, summary


def model_identity(model_path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Verify the exact staged base identity without importing a model."""
    expected = EXPECTED_MODEL_PATH.resolve()
    try:
        actual = model_path.resolve()
    except OSError as error:
        raise ProfileInputError(f"cannot resolve model path: {model_path}") from error
    if actual != expected:
        raise ProfileInputError(f"PRM-07 requires the pinned model path {expected}, got {actual}")
    result: dict[str, Any] = {
        "path": str(expected),
        "revision": EXPECTED_MODEL_REVISION,
        "weights_sha256": EXPECTED_WEIGHT_SHA256,
        "weights_bytes": EXPECTED_WEIGHT_BYTES,
        "tokenizer_json_sha256": EXPECTED_TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "generation_config_sha256": EXPECTED_GENERATION_CONFIG_SHA256,
        "verification": "PRE-05-minicpm-artifacts.json pinned identity",
    }
    if not verify_files:
        return result
    artifacts = {
        "model.safetensors": (EXPECTED_WEIGHT_SHA256, EXPECTED_WEIGHT_BYTES),
        "tokenizer.json": (EXPECTED_TOKENIZER_JSON_SHA256, None),
        "tokenizer_config.json": (EXPECTED_TOKENIZER_CONFIG_SHA256, None),
        "config.json": (EXPECTED_CONFIG_SHA256, None),
        "generation_config.json": (EXPECTED_GENERATION_CONFIG_SHA256, None),
    }
    observed: dict[str, Any] = {}
    for name, (expected_hash, expected_bytes) in artifacts.items():
        path = expected / name
        if not path.is_file():
            raise ProfileInputError(f"pinned model artifact is missing: {path}")
        size = path.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            raise ProfileInputError(f"pinned model artifact size mismatch: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ProfileInputError(f"pinned model artifact SHA256 mismatch: {path}")
        observed[name] = {"bytes": size, "sha256": actual_hash}
    result["files"] = observed
    return result


def classify_length(token_count: int) -> str:
    if type(token_count) is not int or token_count < 1:
        raise ProfileInputError("token_count must be a positive integer")
    if token_count <= SHORT_MAX_TOKENS:
        return "short"
    if token_count <= LONG_MAX_TOKENS:
        return "long"
    raise ProfileInputError(f"token_count exceeds {LONG_MAX_TOKENS}: {token_count}")


def _bucket_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {"short": [], "long": []}
    for row in rows:
        bucket = classify_length(len(row["input_ids"]))
        buckets[bucket].append(row)
    return buckets


def collate_profile_rows(rows: Sequence[Mapping[str, Any]], torch_module: Any) -> dict[str, Any]:
    """Dynamic-pad complete rows while masking padding by position."""
    if not rows:
        raise ProfileInputError("cannot collate an empty profile batch")
    lengths = [len(row["input_ids"]) for row in rows]
    width = max(lengths)
    input_ids = torch_module.full(
        (len(rows), width), SFT_PAD_TOKEN_ID, dtype=torch_module.long,
    )
    attention = torch_module.zeros_like(input_ids)
    labels = torch_module.full_like(input_ids, -100)
    prompt_tokens = 0
    target_tokens = 0
    for index, row in enumerate(rows):
        values = torch_module.tensor(row["input_ids"], dtype=torch_module.long)
        length = lengths[index]
        input_ids[index, :length] = values
        attention[index, :length] = 1
        labels[index, :length] = values
        prompt_tokens += int(row["prompt_token_count"])
        # The row count excludes the final protocol EOS, which is a real
        # full-text label position and is included in this denominator.
        target_tokens += int(row["target_token_count"]) + 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "lengths": lengths,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
    }


def batch_rows(rows: Sequence[Mapping[str, Any]], batch_size: int) -> list[list[Mapping[str, Any]]]:
    if type(batch_size) is not int or batch_size < 1:
        raise ProfileInputError("batch_size must be a positive integer")
    return [list(rows[index:index + batch_size]) for index in range(0, len(rows), batch_size)]


def _device_of(model: Any, torch_module: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch_module.device("cpu")


def _synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)


def _memory_snapshot(torch_module: Any, device: Any) -> tuple[int | None, int | None]:
    if getattr(device, "type", None) != "cuda":
        return None, None
    return (
        int(torch_module.cuda.memory_allocated(device)),
        int(torch_module.cuda.memory_reserved(device)),
    )


def _reset_peak_memory(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.reset_peak_memory_stats(device)


def _peak_memory(torch_module: Any, device: Any) -> tuple[int | None, int | None]:
    if getattr(device, "type", None) != "cuda":
        return None, None
    return (
        int(torch_module.cuda.max_memory_allocated(device)),
        int(torch_module.cuda.max_memory_reserved(device)),
    )


def _loss_from_output(output: Any) -> Any:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, Mapping):
        loss = output.get("loss")
    if loss is None:
        raise ProfileInputError("profile model output has no loss")
    return loss


def _optimizer_step(model: Any, optimizer: Any, batch: Mapping[str, Any], device: Any) -> float:
    model_batch = {
        name: value.to(device)
        for name, value in batch.items()
        if name in {"input_ids", "attention_mask", "labels"}
    }
    output = model(**model_batch)
    loss = _loss_from_output(output)
    observed_loss = float(loss.detach().item())
    if not math.isfinite(observed_loss):
        raise RuntimeError('non_finite_profile_loss')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return observed_loss


def _bucket_denominators(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    actual = sum(len(row["input_ids"]) for row in rows)
    prompt = sum(int(row["prompt_token_count"]) for row in rows)
    target = sum(int(row["target_token_count"]) + 1 for row in rows)
    causal = sum(max(len(row["input_ids"]) - 1, 0) for row in rows)
    return {
        "actual_sequence_tokens": actual,
        "full_text_label_tokens": actual,
        "causal_objective_tokens": causal,
        "prompt_tokens_without_manual_bos": prompt,
        "target_tokens_including_protocol_eos": target,
    }


def measure_sft_profile(
    model: Any,
    optimizer: Any,
    torch_module: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    timed_steps: int = DEFAULT_TIMED_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    deadline: Deadline | None = None,
) -> dict[str, Any]:
    """Run finite natural short/long forward/backward/optimizer measurements."""
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ProfileInputError("warmup_steps must be a nonnegative integer")
    if type(timed_steps) is not int or timed_steps < 1:
        raise ProfileInputError("timed_steps must be a positive integer")
    buckets = _bucket_rows(rows)
    device = _device_of(model, torch_module)
    model.train()
    result: dict[str, Any] = {}
    for name in ("short", "long"):
        bucket = buckets[name]
        denominators = _bucket_denominators(bucket)
        if not bucket:
            result[name] = {
                "status": "no_selected_rows",
                "rows": 0,
                "requested_timed_steps": timed_steps,
                **denominators,
            }
            continue
        batches = batch_rows(bucket, batch_size)
        warmup_batch = batches[0]
        timed_batches = batches[:timed_steps]
        _reset_peak_memory(torch_module, device)
        for _ in range(warmup_steps):
            if deadline is not None:
                deadline.check()
            _optimizer_step(model, optimizer, collate_profile_rows(warmup_batch, torch_module), device)
        losses: list[float] = []
        timed_tokens = 0
        padded_slots = 0
        _synchronize(torch_module, device)
        started = time.perf_counter_ns()
        for current in timed_batches:
            if deadline is not None:
                deadline.check()
            batch = collate_profile_rows(current, torch_module)
            timed_tokens += sum(batch["lengths"])
            padded_slots += len(current) * max(batch["lengths"]) - sum(batch["lengths"])
            losses.append(_optimizer_step(model, optimizer, batch, device))
        _synchronize(torch_module, device)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
        peak_allocated, peak_reserved = _peak_memory(torch_module, device)
        result[name] = {
            "status": "measured",
            "rows": len(bucket),
            "warmup_steps": warmup_steps,
            "requested_timed_steps": timed_steps,
            "timed_steps": len(timed_batches),
            "batch_size": batch_size,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "seconds_per_timed_step": elapsed / len(timed_batches),
            "actual_timed_tokens": timed_tokens,
            "padded_slots_excluded_from_denominator": padded_slots,
            "tokens_per_second": timed_tokens / elapsed if elapsed > 0 else None,
            "mean_loss": sum(losses) / len(losses),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "memory_scope": "bucket warmup plus timed steps; null on CPU",
            **denominators,
        }
    return result


def validate_rl_candidate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    prompt_max_tokens: int = RL_PROMPT_MAX_TOKENS,
    completion_max_tokens: int = RL_COMPLETION_MAX_TOKENS,
) -> dict[str, Any]:
    """Validate/count an explicit RL rollout handoff without generating one.

    Records are expected to contain integer ``prompt_tokens`` and
    ``generated_tokens`` arrays plus ``terminal_reason`` (``eos`` or
    ``length``).  This is intentionally useful for a later real-model probe:
    it makes cap hits, canonical/noncanonical EOG and native CONTROL tokens
    visible instead of treating decoded text as evidence.
    """
    if type(prompt_max_tokens) is not int or prompt_max_tokens < 1:
        raise ProfileInputError("prompt_max_tokens must be a positive integer")
    if type(completion_max_tokens) is not int or completion_max_tokens < 1:
        raise ProfileInputError("completion_max_tokens must be a positive integer")
    counts = {
        "records": 0, "prompt_tokens": 0, "generated_tokens": 0,
        "canonical_eos": 0, "noncanonical_eog": 0, "cap_hits": 0,
        "control_before_terminal": 0, "invalid_records": 0,
    }
    for index, record in enumerate(records):
        prompt = record.get("prompt_tokens")
        generated = record.get("generated_tokens")
        reason = record.get("terminal_reason")
        if (not isinstance(prompt, list) or not isinstance(generated, list)
                or any(type(token) is not int for token in prompt + generated)
                or len(prompt) > prompt_max_tokens or len(generated) > completion_max_tokens
                or reason not in {"eos", "length"}):
            raise ProfileInputError(f"RL record {index} violates prompt/completion caps or schema")
        if not prompt or prompt[0] != BOS_ID:
            raise ProfileInputError(f"RL record {index} prompt must begin with one manual BOS ID")
        if any(token < 0 or token >= VOCAB_SIZE for token in prompt + generated):
            raise ProfileInputError(f"RL record {index} contains an out-of-range token")
        counts["records"] += 1
        counts["prompt_tokens"] += len(prompt)
        counts["generated_tokens"] += len(generated)
        if reason == "length":
            if len(generated) != completion_max_tokens:
                raise ProfileInputError(f"RL record {index} length stop is not exactly the cap")
            counts["cap_hits"] += 1
        elif generated and generated[-1] == EOS_ID:
            counts["canonical_eos"] += 1
        elif generated and generated[-1] in NATIVE_EOG_IDS:
            counts["noncanonical_eog"] += 1
        else:
            raise ProfileInputError(f"RL record {index} eos stop lacks an explicit terminal EOG token")
        if any(is_native_control_token(token) for token in generated[:-1]):
            counts["control_before_terminal"] += 1
    return {
        "status": "accounted",
        "prompt_max_tokens": prompt_max_tokens,
        "completion_max_tokens": completion_max_tokens,
        **counts,
    }


def _frameworks_imported() -> list[str]:
    prefixes = ("torch", "unsloth", "transformers", "trl")
    return sorted(
        name for name in sys.modules
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes)
    )


def live_resource_guard(*, max_existing_vram_mib: float = DEFAULT_MAX_EXISTING_VRAM_MIB) -> dict[str, Any]:
    """Fail before torch/CUDA import if another live process owns the device."""
    if _frameworks_imported():
        raise ProfileGuardError("framework imported before the live process/VRAM guard")
    if not isinstance(max_existing_vram_mib, (int, float)) or not math.isfinite(max_existing_vram_mib) or max_existing_vram_mib < 0:
        raise ProfileGuardError("max_existing_vram_mib must be finite and nonnegative")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ProfileGuardError("nvidia-smi is unavailable; refusing a profile without a device guard")
    try:
        processes = subprocess.run(
            [executable, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        devices = subprocess.run(
            [executable, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProfileGuardError(f"nvidia-smi guard failed: {error}") from error
    active: list[dict[str, int]] = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not all(field.isdigit() for field in fields):
            raise ProfileGuardError("nvidia-smi returned an unparseable compute-process row")
        active.append({"pid": int(fields[0]), "used_memory_mib": int(fields[1])})
    gpu_rows: list[dict[str, int]] = []
    for line in devices.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise ProfileGuardError("nvidia-smi returned an unparseable GPU row")
        gpu_rows.append({"index": int(fields[0]), "memory_used_mib": int(fields[1]), "memory_total_mib": int(fields[2])})
    if len(gpu_rows) != 1:
        raise ProfileGuardError(f"PRM-07 requires one visible GPU; observed {len(gpu_rows)}")
    if active:
        raise ProfileGuardError(f"GPU has active compute processes: {len(active)}")
    if gpu_rows[0]["memory_used_mib"] > max_existing_vram_mib:
        raise ProfileGuardError(
            f"GPU memory already used {gpu_rows[0]['memory_used_mib']} MiB > guard {max_existing_vram_mib:g} MiB"
        )
    return {
        "status": "clear",
        "nvidia_smi": executable,
        "gpu": gpu_rows[0],
        "active_compute_processes": active,
        "max_existing_vram_mib": max_existing_vram_mib,
    }


@contextmanager
def wall_clock_limit(seconds: float) -> Iterable[Deadline]:
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 7_200:
        raise ProfileInputError("wall timeout must be finite and in (0, 7200] seconds")
    deadline = Deadline(time.monotonic(), float(seconds))
    previous_handler = None
    timer_installed = False
    if hasattr(signal, "SIGALRM") and threading_main():
        previous_handler = signal.getsignal(signal.SIGALRM)
        def alarm_handler(_signum: int, _frame: Any) -> None:
            raise ProfileTimeout(f"profile exceeded {seconds:g} second wall-clock bound")
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        timer_installed = True
    try:
        yield deadline
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def threading_main() -> bool:
    # SIGALRM is only safe to install from the interpreter's main thread.
    return threading.current_thread() is threading.main_thread()


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, MemoryError) or "out of memory" in text or "cuda error" in text and "memory" in text


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_sft_components(
    model_path: Path,
    *,
    max_sequence_tokens: int,
    guard: Mapping[str, Any],
    parity_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load the disposable model only after ``live_resource_guard`` passed."""
    if _frameworks_imported():
        raise ProfileGuardError("framework was imported before _load_sft_components")
    if guard.get("status") != "clear":
        raise ProfileGuardError("missing clear live-resource guard result")
    from campaign_sft import TARGET_MODULES
    if tuple(TARGET_MODULES) != SFT_TARGET_MODULES:
        raise ProfileInputError("campaign_sft TARGET_MODULES differ from the admitted seven-module policy")
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path), max_seq_length=max_sequence_tokens,
        dtype=torch.bfloat16, load_in_4bit=False, trust_remote_code=False,
    )
    try:
        reference_tokenizer = load_pinned_reference_tokenizer(model_path)
        tokenizer_audit = restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=parity_rows,
        )
    except TokenizerContractError as error:
        raise ProfileInputError(f"loaded tokenizer failed pinned contract: {error}") from error
    model = FastLanguageModel.get_peft_model(
        model, r=SFT_LORA_RANK, lora_alpha=SFT_LORA_ALPHA, lora_dropout=0,
        target_modules=list(SFT_TARGET_MODULES), bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    attached = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if trainable != SFT_EXPECTED_TRAINABLE_PARAMETERS or len(attached) != SFT_EXPECTED_ATTACHMENTS:
        raise ProfileInputError(
            f"LoRA attachment mismatch: trainable={trainable}, attachments={len(attached)}"
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=SFT_LEARNING_RATE, weight_decay=0.0, fused=True,
    )
    return model, tokenizer, optimizer, torch, tokenizer_audit


def run_live_profile(
    profile_input: ProfileInput,
    *,
    warmup_steps: int,
    timed_steps: int,
    batch_size: int,
    max_existing_vram_mib: float,
    deadline: Deadline,
) -> dict[str, Any]:
    _ids, rows, input_summary = load_profile_rows(profile_input)
    deadline.check()
    base_identity = model_identity(profile_input.model_path, verify_files=True)
    deadline.check()
    guard = live_resource_guard(max_existing_vram_mib=max_existing_vram_mib)
    deadline.check()
    model, tokenizer, optimizer, torch_module, tokenizer_audit = _load_sft_components(
        profile_input.model_path, max_sequence_tokens=LONG_MAX_TOKENS, guard=guard,
        parity_rows=rows,
    )
    del tokenizer
    metrics = measure_sft_profile(
        model, optimizer, torch_module, rows,
        warmup_steps=warmup_steps, timed_steps=timed_steps,
        batch_size=batch_size, deadline=deadline,
    )
    return {
        "model_identity": base_identity,
        "live_guard": guard,
        "tokenizer_contract": tokenizer_audit,
        "input": input_summary,
        "sft_policy": {
            "target_modules": list(SFT_TARGET_MODULES), "lora_rank": SFT_LORA_RANK,
            "lora_alpha": SFT_LORA_ALPHA, "expected_attachments": SFT_EXPECTED_ATTACHMENTS,
            "expected_trainable_parameters": SFT_EXPECTED_TRAINABLE_PARAMETERS,
            "objective": "full_text_loss", "pad_token_id": SFT_PAD_TOKEN_ID,
            "optimizer": "torch.optim.AdamW(fused=True)",
        },
        "sft_metrics": metrics,
        "rl_probe": {
            "status": "pending_real_model_generation_and_recomputed_policy_logprob",
            "prompt_cap": RL_PROMPT_MAX_TOKENS,
            "completion_cap": RL_COMPLETION_MAX_TOKENS,
            "reason": "This command profiles SFT only; RL generation and recomputed policy log probabilities require a separate probe.",
        },
    }


def preflight_profile(profile_input: ProfileInput) -> dict[str, Any]:
    """Run all non-framework checks; this function never imports torch."""
    _ids, rows, input_summary = load_profile_rows(profile_input)
    identity = model_identity(profile_input.model_path, verify_files=True)
    buckets = _bucket_rows(rows)
    denominators = {
        name: _bucket_denominators(bucket) for name, bucket in buckets.items()
    }
    return {
        "status": "preflight_pass",
        "CUDA_started": False,
        "framework_imports": _frameworks_imported(),
        "input": input_summary,
        "model_identity": identity,
        "natural_buckets": {
            name: {"rows": len(bucket), **denominators[name]}
            for name, bucket in buckets.items()
        },
        "profile_policy": {
            "short_max_tokens": SHORT_MAX_TOKENS,
            "long_max_tokens": LONG_MAX_TOKENS,
            "padding": "pad ID 1; attention 0 and labels -100 by position",
            "truncation": "forbidden; selected full input_ids are measured as supplied",
        },
    }


def _base_receipt(args: argparse.Namespace, profile_input: ProfileInput) -> dict[str, Any]:
    return {
        "task": "PRM-07",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "owner": "worker-runtime" if args.preflight else "lead",
        "started_at": _now(),
        "mode": "preflight" if args.preflight else "profile",
        "execution_root": str(EXECUTION_ROOT),
        "input": {
            "candidate_file": str(profile_input.candidate_file),
            "candidate_sha256": profile_input.candidate_sha256,
            "selected_ids_file": str(profile_input.selected_ids_file),
            "selected_ids_sha256": profile_input.selected_ids_sha256,
            "model_path": str(profile_input.model_path),
            "selection_rule": "explicit selected train IDs only; no dev/final/call discovery",
        },
        "limits": {
            "short_max_tokens": SHORT_MAX_TOKENS,
            "long_max_tokens": LONG_MAX_TOKENS,
            "rl_prompt_max_tokens": RL_PROMPT_MAX_TOKENS,
            "rl_completion_max_tokens": RL_COMPLETION_MAX_TOKENS,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "batch_size": args.batch_size,
            "wall_timeout_seconds": args.wall_timeout_seconds,
            "max_existing_vram_mib": args.max_existing_vram_mib,
        },
        "resource_lease": "CPU-only preflight" if args.preflight else "Lead-owned guarded disposable CUDA profile",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--profile", action="store_true")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--selected-train-ids", type=Path, required=True)
    parser.add_argument("--selected-ids-sha256", required=True)
    parser.add_argument("--model-path", type=Path, default=EXPECTED_MODEL_PATH)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--timed-steps", type=int, default=DEFAULT_TIMED_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--wall-timeout-seconds", type=float, default=DEFAULT_WALL_TIMEOUT_SECONDS)
    parser.add_argument("--max-existing-vram-mib", type=float, default=DEFAULT_MAX_EXISTING_VRAM_MIB)
    args = parser.parse_args(argv)
    profile_input = ProfileInput(
        candidate_file=args.candidate_file,
        candidate_sha256=args.candidate_sha256,
        selected_ids_file=args.selected_train_ids,
        selected_ids_sha256=args.selected_ids_sha256,
        model_path=args.model_path,
    )
    receipt = _base_receipt(args, profile_input)
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        print(json.dumps({"status": "refused", "reason": f"receipt already exists: {receipt_path}"}))
        return 2
    exit_code = 0
    try:
        with wall_clock_limit(args.wall_timeout_seconds) as deadline:
            if args.preflight:
                receipt["result"] = preflight_profile(profile_input)
            else:
                receipt["result"] = run_live_profile(
                    profile_input,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    batch_size=args.batch_size,
                    max_existing_vram_mib=args.max_existing_vram_mib,
                    deadline=deadline,
                )
            receipt["status"] = "preflight_pass" if args.preflight else "complete"
    except ProfileTimeout as error:
        receipt["status"] = "timeout"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    except ProfileGuardError as error:
        receipt["status"] = "guard_failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    except ProfileInputError as error:
        receipt["status"] = "input_failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    except BaseException as error:
        receipt["status"] = "oom" if _is_oom(error) else "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    finally:
        receipt["ended_at"] = _now()
        try:
            _atomic_write_json(receipt_path, receipt)
        except BaseException as error:
            print(json.dumps({"status": "receipt_write_failed", "error": str(error)}))
            return 1
    print(json.dumps(receipt, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
