#!/usr/bin/env python3
"""Pinned PRM-03 GRPO/BNPO runner preparation.

This module keeps framework imports behind explicit live-builder calls.  The
CPU-safe surface validates an admitted row/context manifest, resolves the
actual TRL 0.24.0 generation/update geometry, computes protocol rewards, and
provides a deterministic repeated-group sampler.  The live builder accepts an
already-loaded, lead-owned model and tokenizer; it never downloads or chooses a
parent and it never starts training itself.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import difflib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


_EXECUTION_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_SRC = _EXECUTION_ROOT / "packages" / "sepalith" / "src"
if str(_PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(_PROTOCOL_SRC))

from sepalith.campaign_protocol import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    NATIVE_EOG_IDS,
    PromptContext,
    ProtocolError,
    TERMINAL,
    TOKENIZATION_POLICY,
    RENDERER_ID,
    VOCAB_SIZE,
    is_native_control_token,
    parse_output,
    valid_generation_tokens,
)

from campaign_checkpoint import check_identity  # noqa: E402
from campaign_r_parse_probe import RParseOnlyProbe  # noqa: E402
from campaign_reward_v2 import BufferEvidence, score_candidate  # noqa: E402
from campaign_rl_buffer import RewardBufferIndex  # noqa: E402
from campaign_rl_data import (  # noqa: E402
    COMPLETION_MAX_TOKENS,
    CONTEXT_MAX_TOKENS,
    PROMPT_MAX_TOKENS,
    RLDataError,
    RLDataManifest,
    load_training_records,
    records_to_dataset_rows,
    sha256_file,
)
from campaign_rl_profile import (  # noqa: E402
    account_generation_records,
    trim_generated_sequence,
)


TRAIN_SCHEMA_VERSION = "sepalith.prm07.rl-train.v1"
TRL_VERSION = "0.24.0"
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
EXPECTED_ATTACHMENTS = 294
EXPECTED_TRAINABLE_PARAMETERS = 25_116_672
DEFAULT_SEED = 3407
DEFAULT_CANDIDATE_COUNTS = (2, 4)
# This is a call-level packing knob only.  G, the source prompt-group order,
# reward grouping, and the 32 completion rows per update remain unchanged.
ALLOWED_GENERATION_GROUPS_PER_CALL = (1, 2, 4, 8)
# The legacy smoke's G=4, per-device completion batch 8, and accumulation 4
# exposed 32 completion rows per optimizer update.  The geometry helper keeps
# that exposure when G=4 and gives the caller an explicit alternative when a
# live memory gate requires a smaller arm.
LEGACY_ROLLOUT_ROWS_PER_UPDATE = 32
# Keep the 32 completion rows/update exposure while using a profiled-size
# optimizer microbatch.  The alternate 4x8 arm is accepted explicitly below;
# the default follows the lead's measured G=4 policy arm (8x4).
DEFAULT_POLICY_MICROBATCH_SIZE = 8
DEFAULT_POLICY_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_STEPS = 0
DEFAULT_LIGHT_SAVE_STEPS = 10
DEFAULT_FULL_SAVE_STEPS = 50
DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
}
_GENERATION_KWARG_KEYS = frozenset(DEFAULT_GENERATION_KWARGS)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_DRAW_SCHEDULE_SCHEMA = "sepalith.dat09.source-row-draw-sequence.v1"
GENERATION_RECORD_SCHEMA = "sepalith.rl.generation-record.v1"
GRADIENT_RECORD_SCHEMA = "sepalith.rl.gradient-record.v1"


class RLTrainError(ValueError):
    """A recipe, reward, sampler, or TRL boundary failed closed validation."""


class RLGenerationError(RLTrainError):
    """The fixed-ID generation adapter returned unusable geometry."""


class DurableTelemetrySink:
    """Append one JSON record at a time and fsync it before returning."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise RLTrainError("durable telemetry record must be an object")
        try:
            encoded = json.dumps(
                dict(record), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        except (TypeError, ValueError) as error:
            raise RLTrainError(f"telemetry record is not finite JSON: {error}") from error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                raise RLTrainError(f"durable telemetry write failed at {self.path}: {error}") from error

    __call__ = write


def _generation_telemetry_record(
    record: Mapping[str, Any],
    *,
    accounting: Mapping[str, Any],
    accounting_row: Mapping[str, Any] | None,
    global_step: int,
    trl_microstep: int,
    source_schedule_sha256: str | None,
    group_geometry: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Make a compact per-row record without persisting prompt text."""
    prompt_ids = record.get("prompt_tokens")
    generated_ids = record.get("generated_tokens")
    if (not isinstance(prompt_ids, list) or not isinstance(generated_ids, list)
            or any(type(token) is not int for token in prompt_ids + generated_ids)):
        raise RLGenerationError("generation telemetry received non-integer IDs")
    accounting_summary = {
        key: value for key, value in accounting.items() if key != "records_detail"
    }
    return {
        "schema_version": GENERATION_RECORD_SCHEMA,
        "update": {
            "global_step": global_step,
            "global_step_before_update": global_step,
            "trl_microstep": trl_microstep,
        },
        "global_step": global_step,
        "global_step_before_update": global_step,
        "trl_microstep": trl_microstep,
        "source_schedule_sha256": source_schedule_sha256,
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "generated_ids": list(generated_ids),
        "generated_ids_sha256": hashlib.sha256(
            json.dumps(generated_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "generated_token_count": len(generated_ids),
        "terminal_reason": record.get("terminal_reason"),
        "padded_after_terminal": int(record.get("padded_after_terminal", 0)),
        "accounting": accounting_summary,
        "accounting_row": dict(accounting_row) if accounting_row is not None else None,
        "group_geometry": dict(group_geometry),
        "elapsed_sec": float(elapsed_seconds),
    }


def _emit_generation_telemetry(
    trainer: Any,
    records: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> None:
    """Durably emit generation rows, propagating sink failures."""
    sink = getattr(trainer, "_campaign_generation_record_sink", None)
    if sink is None:
        return
    state = getattr(trainer, "state", None)
    global_step = int(getattr(state, "global_step", 0))
    # TRL's GRPOTrainer increments _step for each dataloader microstep and
    # calls generation at the start of a generation buffer.  Keep this value
    # as the actual microstep; fixture objects without TRL use zero.
    trl_microstep = int(getattr(trainer, "_step", getattr(trainer, "_campaign_trl_microstep", 0)))
    details = accounting.get("records_detail") if isinstance(accounting, Mapping) else None
    source_schedule_sha256 = getattr(trainer, "_campaign_source_draw_schedule_sha256", None)
    group_by_index = {
        int(group.get("group_index", index)): group
        for index, group in enumerate(groups)
    }
    writer = getattr(sink, "write", sink)
    try:
        for index, record in enumerate(records):
            group_index = int(record.get("group_index", 0))
            group = group_by_index.get(group_index, {})
            geometry = {
                "group_index": group_index,
                "group_row_index": int(record.get("group_row_index", index)),
                "candidate_count": int(group.get("candidate_count", getattr(trainer, "num_generations", 0))),
                "group_start": group.get("start"),
                "group_end": group.get("end"),
                "generation_call_index": group.get("call_index"),
                "generation_group_count": len(groups),
                "generation_row_count": len(records),
                "generation_groups_per_call": int(
                    getattr(trainer, "_campaign_generation_groups_per_call", 1)
                ),
                "generation_call_count": (
                    (len(groups) + int(getattr(trainer, "_campaign_generation_groups_per_call", 1)) - 1)
                    // int(getattr(trainer, "_campaign_generation_groups_per_call", 1))
                    if groups else 0
                ),
            }
            accounting_row = None
            if isinstance(details, list) and index < len(details) and isinstance(details[index], Mapping):
                accounting_row = details[index]
            writer(_generation_telemetry_record(
                record, accounting=accounting, accounting_row=accounting_row,
                global_step=global_step, trl_microstep=trl_microstep,
                source_schedule_sha256=source_schedule_sha256,
                group_geometry=geometry, elapsed_seconds=elapsed_seconds,
            ))
    except Exception as error:
        raise RLGenerationError(f"generation telemetry sink failed: {error}") from error


def collect_gradient_record(model: Any, *, step: int, torch_module: Any | None = None) -> dict[str, Any]:
    """Aggregate finite LoRA gradients with one scalar device-to-CPU copy.

    Only ``requires_grad`` parameters whose names identify LoRA adapters are
    inspected.  A zero gradient is recorded and accepted; a nonfinite one is
    recorded and rejected by the callback.
    """
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception as error:  # pragma: no cover - live environment-specific
            raise RLTrainError(f"cannot import torch for gradient telemetry: {error}") from error
    try:
        named_parameters = list(model.named_parameters())
    except (AttributeError, TypeError) as error:
        raise RLTrainError(f"gradient telemetry model lacks named_parameters: {error}") from error
    trainable: list[tuple[str, Any]] = []
    for name, parameter in named_parameters:
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        if "lora" not in str(name).lower():
            raise RLTrainError(f"non-LoRA trainable tensor is outside gradient telemetry: {name}")
        trainable.append((str(name), parameter))
    if not trainable:
        raise RLTrainError("gradient telemetry found no trainable LoRA tensors")
    first_device = None
    for _name, parameter in trainable:
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            first_device = getattr(gradient, "device", None)
            break
    if first_device is None:
        first_device = getattr(trainable[0][1], "device", None)
    if first_device is None:
        first_device = torch_module.device("cpu")
    finite = torch_module.ones((), dtype=torch_module.float32, device=first_device)
    present = torch_module.zeros((), dtype=torch_module.float32, device=first_device)
    nonzero = torch_module.zeros((), dtype=torch_module.float32, device=first_device)
    sumsq = torch_module.zeros((), dtype=torch_module.float32, device=first_device)
    for name, parameter in trainable:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        if getattr(gradient, "device", first_device) != first_device:
            raise RLTrainError(f"gradient telemetry requires one device; {name} differs")
        present = present + torch_module.ones((), dtype=torch_module.float32, device=first_device)
        finite = finite * torch_module.isfinite(gradient).all().to(dtype=torch_module.float32)
        nonzero = nonzero + (torch_module.count_nonzero(gradient) > 0).to(dtype=torch_module.float32)
        # Keep the reduction on the gradient device.  ``dtype`` controls the
        # accumulator without materializing a second full FP32 gradient tensor.
        gradient_detached = gradient.detach()
        sumsq = sumsq + torch_module.sum(
            gradient_detached * gradient_detached,
            dtype=torch_module.float32,
        )
    # This is the only aggregate transfer to CPU.  No per-parameter .item()
    # or full FP32 model copy is performed.
    aggregate = torch_module.stack((finite, present, nonzero, sumsq)).detach().cpu().tolist()
    finite_value = bool(float(aggregate[0]) >= 0.5)
    norm_value = math.sqrt(float(aggregate[3])) if finite_value else None
    return {
        "schema_version": GRADIENT_RECORD_SCHEMA,
        "step": int(step),
        "global_step": int(step),
        "finite": finite_value,
        "nonfinite": not finite_value,
        "trainable_tensor_count": len(trainable),
        "grad_present_count": int(round(float(aggregate[1]))),
        "nonzero_tensor_count": int(round(float(aggregate[2]))),
        "norm": norm_value,
        "aggregation": "on_device_scalar_sum_one_cpu_transfer",
        "trainable_scope": "requires_grad LoRA adapter tensors only",
    }


def gradient_telemetry_callback(sink: Any) -> Any:
    """Create the live-only Transformers callback for durable grad records."""
    from transformers import TrainerCallback

    class CampaignGradientTelemetryCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args
            model = kwargs.get("model")
            if model is None:
                raise RLTrainError("gradient telemetry callback did not receive the model")
            record = collect_gradient_record(model, step=int(state.global_step))
            # Persist the nonfinite observation before rejecting the update.
            sink.write(record)
            if not record["finite"]:
                raise RLTrainError(
                    f"nonfinite LoRA gradient at optimizer step {record['step']}"
                )
            return control

    return CampaignGradientTelemetryCallback()


@dataclass(frozen=True)
class TRLGeometry:
    """A complete generation/update geometry after TRL post-init."""

    candidate_count: int
    rollout_rows_per_update: int
    prompt_groups_per_update: int
    per_device_train_batch_size: int
    generation_batch_size: int
    gradient_accumulation_steps: int
    steps_per_generation: int
    num_iterations: int

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_count": self.candidate_count,
            "rollout_rows_per_update": self.rollout_rows_per_update,
            "prompt_groups_per_update": self.prompt_groups_per_update,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "generation_batch_size": self.generation_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_generation": self.steps_per_generation,
            "num_iterations": self.num_iterations,
        }


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RLTrainError(f"{name} must be a lowercase SHA256")
    return value


def _int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RLTrainError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_generation_groups_per_call(
    value: object,
    name: str = "generation_groups_per_call",
    *,
    max_groups: int | None = None,
) -> int:
    """Resolve the finite adjacent-group packing arm used by live generation."""
    groups = _int(value, name, 1)
    if groups not in ALLOWED_GENERATION_GROUPS_PER_CALL:
        raise RLTrainError(
            f"{name} must be one of {ALLOWED_GENERATION_GROUPS_PER_CALL}"
        )
    if max_groups is not None and groups > max_groups:
        raise RLTrainError(f"{name} cannot exceed the generation buffer's {max_groups} groups")
    return groups


def source_draw_sequence_sha256(row_ids: Sequence[str]) -> str:
    """Hash the ordered source-row DRAW sequence, including repetitions."""
    if not isinstance(row_ids, (list, tuple)):
        raise RLTrainError("source draw sequence must be a list or tuple")
    if any(type(row_id) is not str or not row_id for row_id in row_ids):
        raise RLTrainError("source draw sequence IDs must be nonempty strings")
    return hashlib.sha256(
        json.dumps(list(row_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_source_draw_schedule(
    path: Path,
    expected_sha256: str,
    *,
    selected_ids: Sequence[str],
    selected_ids_sha256: str,
    ordered_ids_sha256: str,
    row_identity_sha256: str,
    candidate_count: int,
    source_draws_per_update: int,
    buffer_reuse: int,
) -> dict[str, Any]:
    """Read and verify the frozen source-presentation schedule.

    The selected-ID contract remains a unique ordered dataset identity.  The
    schedule is a separate ordered draw stream and may repeat those IDs only
    within the finite replay/exposure policy sealed by the pool preparation.
    """
    path = Path(path)
    expected_sha256 = _sha256(expected_sha256, "source draw schedule SHA256")
    if not path.is_absolute() or not path.is_file():
        raise RLTrainError(f"source draw schedule must be an absolute file: {path}")
    before = sha256_file(path)
    if before != expected_sha256:
        raise RLTrainError(f"source draw schedule SHA256 mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RLTrainError(f"source draw schedule is not readable JSON: {path}") from error
    after = sha256_file(path)
    if before != after:
        raise RLTrainError("source draw schedule changed while being read")
    if not isinstance(value, Mapping) or value.get("schema_version") != SOURCE_DRAW_SCHEDULE_SCHEMA:
        raise RLTrainError("source draw schedule schema mismatch")
    selected_ids = list(selected_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise RLTrainError("recipe selected IDs must remain unique before schedule binding")
    if value.get("selected_ids_sha256") != selected_ids_sha256:
        raise RLTrainError("source draw schedule selected-ID hash differs from recipe")
    if value.get("ordered_ids_sha256") != ordered_ids_sha256:
        raise RLTrainError("source draw schedule ordered-ID hash differs from recipe")
    if value.get("row_identity_sha256") != row_identity_sha256:
        raise RLTrainError("source draw schedule row identity differs from recipe")
    if type(value.get("candidate_count")) is not int or value.get("candidate_count") != candidate_count:
        raise RLTrainError("source draw schedule candidate count differs from recipe")
    if (type(value.get("source_draws_per_update")) is not int
            or value.get("source_draws_per_update") != source_draws_per_update):
        raise RLTrainError("source draw schedule groups per update differ from recipe")
    if type(value.get("buffer_reuse")) is not int or value.get("buffer_reuse") != buffer_reuse:
        raise RLTrainError("source draw schedule buffer reuse differs from recipe")
    if ("prompt_groups_per_update" in value
            and (type(value["prompt_groups_per_update"]) is not int
                 or value["prompt_groups_per_update"] != source_draws_per_update)):
        raise RLTrainError("source draw schedule prompt groups differ from source draws")
    if ("completions_per_update" in value
            and (type(value["completions_per_update"]) is not int
                 or value["completions_per_update"] != candidate_count * source_draws_per_update)):
        raise RLTrainError("source draw schedule completion geometry differs from recipe")
    # These fields are redundant by design: they make the frozen schedule
    # self-describing and catch a schedule copied from a different TRL arm.
    # Keep the loader compatible with pre-binding fixtures that omit them,
    # while rejecting an explicitly recorded mismatch.
    for name in ("gradient_accumulation_steps", "steps_per_generation"):
        if name in value and value[name] != buffer_reuse:
            raise RLTrainError(f"source draw schedule {name} differs from buffer reuse")
    row_ids = value.get("row_ids")
    if not isinstance(row_ids, list):
        raise RLTrainError("source draw schedule row_ids must be a list")
    source_draws = value.get("source_draws")
    if type(source_draws) is not int or source_draws != len(row_ids) or source_draws < 1:
        raise RLTrainError("source draw schedule source_draws is inconsistent")
    if source_draws % source_draws_per_update:
        raise RLTrainError("source draw schedule ends inside an optimizer update")
    if source_draw_sequence_sha256(row_ids) != value.get("sequence_sha256"):
        raise RLTrainError("source draw schedule sequence hash mismatch")
    selected_set = set(selected_ids)
    if any(type(row_id) is not str or row_id not in selected_set for row_id in row_ids):
        raise RLTrainError("source draw schedule contains an ID outside unique selected train rows")
    return {
        "path": str(path),
        "sha256": before,
        "sequence_sha256": value["sequence_sha256"],
        "source_draws": source_draws,
        "row_ids": row_ids,
        "selected_ids_sha256": value["selected_ids_sha256"],
        "ordered_ids_sha256": value["ordered_ids_sha256"],
        "row_identity_sha256": value["row_identity_sha256"],
        "candidate_count": candidate_count,
        "source_draws_per_update": source_draws_per_update,
        "buffer_reuse": buffer_reuse,
        "gradient_accumulation_steps": value.get("gradient_accumulation_steps", buffer_reuse),
        "steps_per_generation": value.get("steps_per_generation", buffer_reuse),
        "hash_stable_before_after": True,
        "unique_selected_ids_contract": True,
    }


def resolve_generation_kwargs(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the small, identity-bearing sampling policy for the first arm."""
    if value is not None and not isinstance(value, Mapping):
        raise RLTrainError("generation_kwargs must be an object")
    values = dict(DEFAULT_GENERATION_KWARGS)
    if value:
        unknown = set(value) - _GENERATION_KWARG_KEYS
        if unknown:
            raise RLTrainError(f"generation_kwargs contains unsupported keys: {sorted(unknown)}")
        values.update(value)
    if type(values["do_sample"]) is not bool or not values["do_sample"]:
        raise RLTrainError("generation_kwargs.do_sample must remain true for the sampled RL arm")
    for name, minimum in (("temperature", 0.0), ("top_p", 0.0), ("repetition_penalty", 0.0)):
        number = values[name]
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(float(number)):
            raise RLTrainError(f"generation_kwargs.{name} must be finite")
        if float(number) <= minimum:
            raise RLTrainError(f"generation_kwargs.{name} must be > {minimum}")
    if float(values["top_p"]) > 1.0:
        raise RLTrainError("generation_kwargs.top_p must be <= 1")
    return values


def resolve_trl_geometry(
    candidate_count: int,
    *,
    rollout_rows_per_update: int = LEGACY_ROLLOUT_ROWS_PER_UPDATE,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    world_size: int = 1,
) -> TRLGeometry:
    """Resolve a valid TRL batch shape with one generation per update.

    TRL 0.24.0 forbids configuring generation_batch_size and
    steps_per_generation together.  We therefore configure the optimizer
    microbatch and ``steps_per_generation`` and let TRL derive the generation
    batch.  With ``num_iterations=1`` and steps_per_generation equal to
    gradient accumulation, one generation buffer is consumed by exactly one
    optimizer update.  Generation itself is chunked into calls of
    generation_groups_per_call * G rows in the fixed-ID adapter, while each
    logical G-sized group remains intact; a 32-row buffer therefore never
    becomes a 32-prompt model.generate call unless that arm is explicit.
    """
    candidate_count = _int(candidate_count, "candidate_count", 2)
    if candidate_count not in DEFAULT_CANDIDATE_COUNTS:
        raise RLTrainError(f"candidate_count must be one of {DEFAULT_CANDIDATE_COUNTS}")
    rollout_rows_per_update = _int(rollout_rows_per_update, "rollout_rows_per_update", candidate_count)
    if rollout_rows_per_update % candidate_count:
        raise RLTrainError("rollout_rows_per_update must be divisible by candidate_count")
    world_size = _int(world_size, "world_size", 1)
    if world_size != 1:
        raise RLTrainError("the first fixed-ID runner requires one process until cross-rank group tests exist")

    # A one-group memory arm remains useful for profiling.  For the default
    # 32-row arm use the measured 8x4 shape; callers can select 4x8 with both
    # arguments explicit.  A non-default row count without an explicit shape
    # means one optimizer microbatch, preserving the old one-group helper.
    if per_device_train_batch_size is None and gradient_accumulation_steps is None:
        if rollout_rows_per_update == LEGACY_ROLLOUT_ROWS_PER_UPDATE:
            per_device_train_batch_size = DEFAULT_POLICY_MICROBATCH_SIZE
            gradient_accumulation_steps = DEFAULT_POLICY_GRADIENT_ACCUMULATION_STEPS
        else:
            per_device_train_batch_size = rollout_rows_per_update
            gradient_accumulation_steps = 1
    elif per_device_train_batch_size is None:
        gradient_accumulation_steps = _int(
            gradient_accumulation_steps, "gradient_accumulation_steps", 1,
        )
        if rollout_rows_per_update % (gradient_accumulation_steps * world_size):
            raise RLTrainError("rollout_rows_per_update must equal microbatch times accumulation")
        per_device_train_batch_size = rollout_rows_per_update // (gradient_accumulation_steps * world_size)
    elif gradient_accumulation_steps is None:
        per_device_train_batch_size = _int(
            per_device_train_batch_size, "per_device_train_batch_size", candidate_count,
        )
        if rollout_rows_per_update % (per_device_train_batch_size * world_size):
            raise RLTrainError("rollout_rows_per_update must equal microbatch times accumulation")
        gradient_accumulation_steps = rollout_rows_per_update // (per_device_train_batch_size * world_size)
    else:
        per_device_train_batch_size = _int(
            per_device_train_batch_size, "per_device_train_batch_size", candidate_count,
        )
        gradient_accumulation_steps = _int(
            gradient_accumulation_steps, "gradient_accumulation_steps", 1,
        )
    assert per_device_train_batch_size is not None
    assert gradient_accumulation_steps is not None
    if per_device_train_batch_size * gradient_accumulation_steps * world_size != rollout_rows_per_update:
        raise RLTrainError("rollout_rows_per_update must equal microbatch times accumulation")
    if rollout_rows_per_update == LEGACY_ROLLOUT_ROWS_PER_UPDATE and per_device_train_batch_size not in (4, 8):
        raise RLTrainError("32-row RL updates require a 4x8 or 8x4 microbatch/accumulation shape")
    groups = rollout_rows_per_update // candidate_count
    return TRLGeometry(
        candidate_count=candidate_count,
        rollout_rows_per_update=rollout_rows_per_update,
        prompt_groups_per_update=groups,
        per_device_train_batch_size=per_device_train_batch_size,
        generation_batch_size=rollout_rows_per_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_generation=gradient_accumulation_steps,
        num_iterations=1,
    )


def make_grpo_config(
    *,
    candidate_count: int,
    rollout_rows_per_update: int = LEGACY_ROLLOUT_ROWS_PER_UPDATE,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    output_dir: str | Path = "/tmp/sepalith-rl-preflight",
    max_steps: int = 1,
    save_steps: int = DEFAULT_FULL_SAVE_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    use_cpu: bool = False,
    generation_kwargs: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[Any, TRLGeometry]:
    """Construct actual TRL 0.24.0 config without conflicting batch fields."""
    geometry = resolve_trl_geometry(
        candidate_count,
        rollout_rows_per_update=rollout_rows_per_update,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    sampling = resolve_generation_kwargs(generation_kwargs)
    try:
        from trl import GRPOConfig
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLTrainError(f"cannot import the pinned TRL GRPOConfig: {error}") from error
    _int(max_steps, "max_steps", 1)
    _int(save_steps, "save_steps", 1)
    _int(warmup_steps, "warmup_steps", 0)
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise RLTrainError("learning_rate must be positive")
    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": geometry.per_device_train_batch_size,
        "gradient_accumulation_steps": geometry.gradient_accumulation_steps,
        # Pass only steps_per_generation.  TRL derives generation_batch_size
        # from the microbatch and process count, avoiding its mutually
        # exclusive generation_batch_size/steps_per_generation fields.
        "steps_per_generation": geometry.steps_per_generation,
        "num_generations": geometry.candidate_count,
        "max_prompt_length": PROMPT_MAX_TOKENS,
        "max_completion_length": COMPLETION_MAX_TOKENS,
        "max_steps": max_steps,
        "learning_rate": float(learning_rate),
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "constant_with_warmup",
        "loss_type": "bnpo",
        "scale_rewards": "group",
        "beta": 0.0,
        # Keep the policy-logprob temperature identical to the direct
        # generation adapter below.  A mismatch would make the sampled
        # distribution and the on-policy loss silently use different scales.
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "repetition_penalty": sampling["repetition_penalty"],
        "remove_unused_columns": False,
        "shuffle_dataset": False,
        "dataloader_num_workers": 0,
        "dataloader_drop_last": True,
        "use_vllm": False,
        "vllm_importance_sampling_correction": False,
        "num_iterations": 1,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_only_model": False,
        "ignore_data_skip": False,
        "logging_steps": 1,
        "log_completions": False,
        "report_to": "none",
        "eval_strategy": "no",
        "seed": DEFAULT_SEED,
        "data_seed": DEFAULT_SEED,
        "bf16": not use_cpu,
        "fp16": False,
        "use_cpu": use_cpu,
    }
    if extra:
        forbidden = {
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "generation_batch_size", "steps_per_generation",
            "num_generations", "max_prompt_length", "max_completion_length",
            "num_iterations", "loss_type", "beta", "remove_unused_columns", "shuffle_dataset",
            "generation_kwargs",
        }
        overlap = forbidden.intersection(extra)
        if overlap:
            raise RLTrainError(f"extra config cannot override pinned fields: {sorted(overlap)}")
        values.update(extra)
    try:
        config = GRPOConfig(**values)
    except Exception as error:
        raise RLTrainError(f"pinned TRL config rejected resolved geometry: {error}") from error
    observed = {
        "generation_batch_size": int(config.generation_batch_size),
        "steps_per_generation": int(config.steps_per_generation),
        "per_device_train_batch_size": int(config.per_device_train_batch_size),
        "num_generations": int(config.num_generations),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "num_iterations": int(config.num_iterations),
    }
    expected = {
        "generation_batch_size": geometry.generation_batch_size,
        "steps_per_generation": geometry.steps_per_generation,
        "per_device_train_batch_size": geometry.per_device_train_batch_size,
        "num_generations": geometry.candidate_count,
        "gradient_accumulation_steps": geometry.gradient_accumulation_steps,
        "num_iterations": 1,
    }
    if observed != expected:
        raise RLTrainError(f"TRL changed the requested one-generation geometry: {observed} != {expected}")
    return config, geometry


class CampaignRepeatSampler:
    """Stateless ordered repeated-group sampler for full-resume reconstruction."""

    SAMPLER_ID = "campaign-repeat-manifest-order-v1"

    def __init__(
        self,
        data_source: SizedLike,
        *,
        candidate_count: int,
        prompt_groups_per_batch: int,
        repeat_count: int = 1,
        seed: int = DEFAULT_SEED,
        shuffle: bool = False,
        generation_batch_size: int | None = None,
        per_device_train_batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        steps_per_generation: int | None = None,
        source_draw_sequence: Sequence[int] | None = None,
        source_draw_sequence_sha256: str | None = None,
        source_draw_schedule_sha256: str | None = None,
    ) -> None:
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.candidate_count = _int(candidate_count, "candidate_count", 2)
        if self.candidate_count not in DEFAULT_CANDIDATE_COUNTS:
            raise RLTrainError(f"candidate_count must be one of {DEFAULT_CANDIDATE_COUNTS}")
        self.prompt_groups_per_batch = _int(prompt_groups_per_batch, "prompt_groups_per_batch", 1)
        self.repeat_count = _int(repeat_count, "repeat_count", 1)
        self.seed = _int(seed, "seed", 0)
        if shuffle:
            raise RLTrainError("shuffle is not admitted for the first stateless resume sampler")
        self.shuffle = False
        self.generation_batch_size = generation_batch_size
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.steps_per_generation = steps_per_generation
        if source_draw_sequence is None:
            # Preserve the original ordered-dataset fallback for small unit
            # fixtures.  A live campaign recipe must provide a frozen draw
            # schedule; build_live_trainer fails closed before constructing
            # this fallback.
            self.source_draw_sequence = tuple(range(self.num_samples))
            self.source_draws_bound = False
        else:
            if not isinstance(source_draw_sequence, (list, tuple)):
                raise RLTrainError("source_draw_sequence must be a list or tuple")
            self.source_draw_sequence = tuple(source_draw_sequence)
            self.source_draws_bound = True
            if len(self.source_draw_sequence) % self.prompt_groups_per_batch:
                raise RLTrainError(
                    "source draw sequence must end at a complete prompt-group update"
                )
            if source_draw_sequence_sha256 is None:
                raise RLTrainError("source draw sequence hash is required when schedule is bound")
        if any(
            type(index) is not int or not 0 <= index < self.num_samples
            for index in self.source_draw_sequence
        ):
            raise RLTrainError("source draw sequence contains an out-of-range dataset index")
        if source_draw_sequence_sha256 is not None:
            self.source_draw_sequence_sha256 = _sha256(
                source_draw_sequence_sha256, "source_draw_sequence_sha256",
            )
        else:
            self.source_draw_sequence_sha256 = None
        if source_draw_schedule_sha256 is not None:
            self.source_draw_schedule_sha256 = _sha256(
                source_draw_schedule_sha256, "source_draw_schedule_sha256",
            )
        else:
            self.source_draw_schedule_sha256 = None
        for name, value in (
            ("generation_batch_size", generation_batch_size),
            ("per_device_train_batch_size", per_device_train_batch_size),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
            ("steps_per_generation", steps_per_generation),
        ):
            if value is not None:
                _int(value, name, 1)
        if (generation_batch_size is not None
                and per_device_train_batch_size is not None
                and steps_per_generation is not None
                and generation_batch_size != per_device_train_batch_size * steps_per_generation):
            raise RLTrainError("sampler generation batch does not match microbatch times steps")
        if (gradient_accumulation_steps is not None and steps_per_generation is not None
                and gradient_accumulation_steps != steps_per_generation):
            raise RLTrainError(
                "sampler gradient accumulation must equal steps_per_generation for one buffer per update"
            )
        if steps_per_generation is not None and self.repeat_count != steps_per_generation:
            raise RLTrainError(
                "sampler buffer reuse must equal steps_per_generation for the actual TRL geometry"
            )

    def __iter__(self) -> Iterator[int]:
        indexes = self.source_draw_sequence
        width = self.prompt_groups_per_batch
        for first in range(0, len(indexes), width):
            chunk = indexes[first:first + width]
            if len(chunk) != width:
                continue
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.candidate_count):
                        yield index

    def __len__(self) -> int:
        complete_groups = len(self.source_draw_sequence) // self.prompt_groups_per_batch
        return complete_groups * self.prompt_groups_per_batch * self.candidate_count * self.repeat_count

    def state(self, *, consumed_rows: int = 0, epoch: int = 0) -> dict[str, Any]:
        consumed_rows = _int(consumed_rows, "consumed_rows", 0)
        epoch = _int(epoch, "epoch", 0)
        if consumed_rows % self.candidate_count:
            raise RLTrainError("sampler consumed_rows must end at a complete candidate group")
        if (self.generation_batch_size is not None
                and consumed_rows % self.generation_batch_size):
            raise RLTrainError("sampler consumed_rows must end at a generation/update boundary")
        prompt_copies = consumed_rows // self.candidate_count
        repeat_block = self.prompt_groups_per_batch * self.repeat_count
        cycle, within = divmod(prompt_copies, repeat_block)
        selected_id_index = (
            self.source_draw_sequence[
                (cycle * self.prompt_groups_per_batch + within % self.prompt_groups_per_batch)
                % len(self.source_draw_sequence)
            ]
            if self.source_draw_sequence
            else 0
        )
        source_update_rows = self.prompt_groups_per_batch * self.candidate_count * self.repeat_count
        source_draw_cursor = (
            (consumed_rows // source_update_rows) * self.prompt_groups_per_batch
            if source_update_rows
            else 0
        )
        state = {
            "sampler_id": self.SAMPLER_ID,
            "seed": self.seed,
            "shuffle": False,
            "num_samples": self.num_samples,
            "candidate_count": self.candidate_count,
            "prompt_groups_per_batch": self.prompt_groups_per_batch,
            "repeat_count": self.repeat_count,
            "consumed_rows": consumed_rows,
            "current_index": consumed_rows,
            "consumed_prompt_copies": prompt_copies,
            "selected_id_index": selected_id_index,
            "epoch": epoch,
        }
        if self.source_draws_bound:
            state.update({
                "source_draws_bound": True,
                "source_draws": len(self.source_draw_sequence),
                "source_draw_cursor": source_draw_cursor,
                "buffer_reuse": self.repeat_count,
                "source_draw_sequence_sha256": self.source_draw_sequence_sha256,
                "source_draw_schedule_sha256": self.source_draw_schedule_sha256,
            })
        geometry = {
            "generation_batch_size": self.generation_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_generation": self.steps_per_generation,
        }
        if any(value is not None for value in geometry.values()):
            state["geometry"] = {
                name: value for name, value in geometry.items() if value is not None
            }
            if self.generation_batch_size is not None:
                state["generation_rows_per_update"] = self.generation_batch_size
                if self.steps_per_generation is not None:
                    state["sampler_rows_per_update"] = (
                        self.generation_batch_size * self.steps_per_generation
                    )
        return state

    def indices_from(self, consumed_rows: int, count: int) -> list[int]:
        """Reconstruct a stream slice from a checkpoint boundary.

        The sampler has no mutable cursor, so Trainer resume can reproduce the
        exact next rows from the saved completion-row count.  Requiring a full
        batch boundary catches accidental use of a per-microbatch count at the
        optimizer checkpoint seam.
        """
        consumed_rows = _int(consumed_rows, "consumed_rows", 0)
        count = _int(count, "count", 1)
        if consumed_rows % (self.prompt_groups_per_batch * self.candidate_count):
            raise RLTrainError("sampler resume cursor is inside a prompt group batch")
        if (self.generation_batch_size is not None
                and consumed_rows % self.generation_batch_size):
            raise RLTrainError("sampler resume cursor is inside a generation buffer")
        stream_length = len(self)
        if consumed_rows > stream_length or count > stream_length - consumed_rows:
            raise RLTrainError("sampler resume slice exceeds available stream")
        from itertools import islice
        return list(islice(iter(self), consumed_rows, consumed_rows + count))


# A tiny structural alias avoids importing torch's Sampler on CPU preflight.
class SizedLike:
    def __len__(self) -> int:  # pragma: no cover - protocol-like helper
        raise NotImplementedError


def line_f1(pred_lines: Sequence[str] | None, expected_lines: Sequence[str] | None) -> float:
    """The legacy difflib line-F1, retained as reward shaping only."""
    pred = [line.rstrip() for line in (pred_lines or [])]
    expected = [line.rstrip() for line in (expected_lines or [])]
    while pred and pred[-1] == "":
        pred.pop()
    while expected and expected[-1] == "":
        expected.pop()
    if pred == expected:
        return 1.0
    if not pred or not expected:
        return 0.0
    matches = sum(block.size for block in difflib.SequenceMatcher(
        a=pred, b=expected, autojunk=False,
    ).get_matching_blocks())
    precision = matches / len(pred)
    recall = matches / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class CampaignPRM03Reward:
    """Protocol, restraint, repetition, and per-row syntax reward for TRAIN."""

    __name__ = "campaign_prm03_reward_coverage_v2"

    def __init__(
        self,
        tokenizer: Any | None = None,
        *,
        decoder: Callable[..., str] | None = None,
        max_completion_tokens: int = COMPLETION_MAX_TOKENS,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
        reward_buffer_index: RewardBufferIndex | None = None,
        parse_probe: Callable[[str], bool] | None = None,
    ) -> None:
        if type(max_completion_tokens) is not int or max_completion_tokens < 1:
            raise RLTrainError("max_completion_tokens must be positive")
        if tokenizer is None and decoder is None:
            raise RLTrainError("a pinned tokenizer or explicit decoder is required")
        if reward_buffer_index is None:
            raise RLTrainError("reward coverage v2 requires a hash-checked reward-buffer index")
        if parse_probe is None:
            raise RLTrainError("reward coverage v2 requires an explicit parse-only probe")
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.max_completion_tokens = max_completion_tokens
        self.event_sink = event_sink
        self.reward_buffer_index = reward_buffer_index
        self.parse_probe = parse_probe
        self.last_records: list[dict[str, Any]] = []

    @staticmethod
    def _ids(value: object, index: int) -> list[int]:
        if hasattr(value, "detach"):
            value = value.detach().tolist()
        elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise RLTrainError(f"completion {index} IDs are not a sequence")
        values = list(value)
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in values):
            raise RLTrainError(f"completion {index} has a noninteger or out-of-range token")
        return values

    def _decode(self, ids: Sequence[int]) -> str:
        kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        if self.decoder is not None:
            return self.decoder(list(ids), **kwargs)
        assert self.tokenizer is not None
        return self.tokenizer.decode(list(ids), **kwargs)

    def _buffer_evidence(
        self, row_id: str, envelope: Mapping[str, Any], context: PromptContext,
    ) -> tuple[BufferEvidence, Callable[[str], bool] | None]:
        if envelope.get("row_id") != row_id:
            raise RLTrainError(f"reward buffer row identity differs for {row_id}")
        context_sha = str(context.replacement_range.content_sha256)
        if envelope.get("context_sha256") != context_sha:
            raise RLTrainError(f"reward buffer context identity differs for {row_id}")
        mode = envelope.get("syntax_evidence_mode")
        if mode == "unverified":
            if envelope.get("syntax_available") is not False or not envelope.get("repair_reason"):
                raise RLTrainError(f"unverified syntax evidence differs for {row_id}")
            return BufferEvidence(
                "unverified", None, context_sha, None, None, None, "", None,
            ), None
        if mode not in {"complete_document", "completion_prefix", "framed_fragment"}:
            raise RLTrainError(f"syntax evidence mode differs for {row_id}")
        baseline = self.reward_buffer_index.baseline_text(envelope)
        parser_identity = json.dumps(envelope.get("parser_identity"), sort_keys=True, separators=(",", ":"))
        evidence = BufferEvidence(
            str(mode), baseline, str(envelope["baseline_sha256"]),
            envelope.get("baseline_parse_ok"), envelope.get("gold_applied_parse_ok"),
            envelope.get("framed_projection_parse_ok"), str(envelope.get("diagnostic_suffix", "")),
            parser_identity,
        )
        return evidence, self.parse_probe

    def score_one(
        self,
        context_value: Mapping[str, Any] | PromptContext,
        target_operation: str,
        target_body_text: str,
        generated_ids: Sequence[int],
        reward_buffer: Mapping[str, Any],
        *,
        row_id: str,
        family: str = "",
        package_id: str = "",
    ) -> tuple[float, dict[str, Any]]:
        if not row_id:
            raise RLTrainError("reward coverage v2 requires a row ID")
        try:
            context = context_value if isinstance(context_value, PromptContext) else PromptContext.from_mapping(context_value)
        except (ProtocolError, TypeError, ValueError) as error:
            raise RLTrainError(f"reward context {row_id} is invalid: {error}") from error
        buffer, candidate_probe = self._buffer_evidence(row_id, reward_buffer, context)
        try:
            ids = self._ids(generated_ids, 0)
            failure = None
        except RLTrainError:
            ids = []
            failure = "invalid_token_ids"
        controls = 0
        canonical_eos = False
        noncanonical_eog = False
        if failure is None:
            if not ids:
                failure = "empty_completion"
            elif len(ids) > self.max_completion_tokens:
                failure = "completion_cap_exceeded"
            elif EOS_ID in ids[:-1]:
                failure = "premature_canonical_eos"
            else:
                controls = sum(is_native_control_token(token) for token in ids[:-1])
                if controls:
                    failure = "control_before_terminal"
                elif ids[-1] == 130073:
                    noncanonical_eog = True
                    failure = "noncanonical_eog"
                elif ids[-1] != EOS_ID:
                    failure = "missing_canonical_eos"
                elif not valid_generation_tokens(ids):
                    failure = "invalid_generation_tokens"
                else:
                    canonical_eos = True
        parsed = None
        if failure is None:
            try:
                parsed = parse_output(self._decode(ids[:-1]), context)
            except (ProtocolError, TypeError, ValueError) as error:
                failure = f"parse_error:{error}"
            if failure is None and parsed is not None and (parsed.status != "accepted" or parsed.operation is None):
                failure = parsed.reason or "invalid_protocol_output"
        protocol_valid = failure is None and parsed is not None
        operation = parsed.operation if protocol_valid else None
        body_lines = list(parsed.body) if protocol_valid and operation != "no_op" else []
        value, detail = score_candidate(
            target_operation=target_operation,
            target_body_text=target_body_text,
            region_old=list(context.region_old),
            protocol_valid=protocol_valid,
            protocol_failure=failure,
            operation=operation,
            body_lines=body_lines,
            replacement_range=context.replacement_range.to_dict(),
            document_eol=context.document_eol,
            buffer=buffer,
            parse_probe=candidate_probe,
        )
        actual = list(context.region_old) if protocol_valid and operation == "no_op" else body_lines
        expected = (list(context.region_old) if target_operation == "no_op" else
                    [] if target_operation == "delete" else target_body_text.split("\n"))
        record: dict[str, Any] = {
            "id": row_id, "family": family, "package_id": package_id,
            "generated_tokens": len(ids), "canonical_eos": canonical_eos,
            "noncanonical_eog": noncanonical_eog, "control_before_terminal": controls,
            "cap_hit": len(ids) == self.max_completion_tokens,
            "protocol_valid": protocol_valid, "operation": operation,
            "expected_operation": target_operation,
            "line_f1": line_f1(actual, expected) if protocol_valid else 0.0,
            "failure": failure, "reward": value,
            "reward_policy": "protocol_restraint_syntax_coverage_v2",
            "syntax_evidence_mode": buffer.mode,
            **detail,
        }
        record["reward_policy"] = "protocol_restraint_syntax_coverage_v2"
        record["output_ids_sha256"] = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        return value, record

    def __call__(
        self,
        *,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Sequence[Sequence[int]],
        context: Sequence[Mapping[str, Any] | PromptContext],
        target_operation: Sequence[str],
        target_body_text: Sequence[str],
        reward_buffer: Sequence[Mapping[str, Any]],
        id: Sequence[str],
        family: Sequence[str] | None = None,
        package_id: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        del prompts, completions
        size = len(completion_ids)
        fields = [context, target_operation, target_body_text, reward_buffer, id]
        if any(len(value) != size for value in fields):
            raise RLTrainError("reward metadata and completion IDs are not aligned")
        family_field = family or [""] * size
        package_field = package_id or [""] * size
        if any(len(value) != size for value in (family_field, package_field)):
            raise RLTrainError("reward provenance metadata is not aligned")
        values: list[float] = []
        self.last_records = []
        for index in range(size):
            value, record = self.score_one(
                context[index], target_operation[index], target_body_text[index], completion_ids[index],
                reward_buffer[index], row_id=id[index], family=family_field[index],
                package_id=package_field[index],
            )
            values.append(value)
            self.last_records.append(record)
            if self.event_sink is not None:
                self.event_sink(record)
        return values


class FixedIDGRPOTrainerMixin:
    """Private-API adapter shared by the dynamically loaded TRL subclass."""

    def configure_campaign_runtime(
        self,
        *,
        generation_guard_factory: Callable[[Any], Any] | None,
        post_generation_restore: Callable[[Any, Any], None] | None = None,
        generation_record_sink: Any | None = None,
        prompt_max_tokens: int = PROMPT_MAX_TOKENS,
        completion_max_tokens: int = COMPLETION_MAX_TOKENS,
        context_max_tokens: int = CONTEXT_MAX_TOKENS,
        generation_kwargs: Mapping[str, Any] | None = None,
        generation_groups_per_call: int = 1,
    ) -> None:
        if type(prompt_max_tokens) is not int or prompt_max_tokens < 1:
            raise RLTrainError("prompt_max_tokens must be positive")
        if type(completion_max_tokens) is not int or completion_max_tokens < 1:
            raise RLTrainError("completion_max_tokens must be positive")
        if type(context_max_tokens) is not int or context_max_tokens < prompt_max_tokens + completion_max_tokens:
            raise RLTrainError("context_max_tokens is smaller than prompt plus completion caps")
        self._campaign_generation_guard_factory = generation_guard_factory
        self._campaign_post_generation_restore = post_generation_restore
        self._campaign_prompt_max_tokens = prompt_max_tokens
        self._campaign_completion_max_tokens = completion_max_tokens
        self._campaign_context_max_tokens = context_max_tokens
        self._campaign_generation_kwargs = dict(generation_kwargs or {})
        self._campaign_generation_groups_per_call = resolve_generation_groups_per_call(
            generation_groups_per_call, "runtime generation_groups_per_call",
        )
        # Fixture/unit callers leave this unset.  build_live_trainer installs
        # the fsynced output-dir sink explicitly for a real trainer.
        self._campaign_generation_record_sink = generation_record_sink
        self._campaign_runtime_configured = True

    def _prompt_ids(self, prompt: Any, index: int) -> list[int]:
        if not isinstance(prompt, Mapping):
            raise RLGenerationError(f"prompt {index} is not an ID envelope")
        values = prompt.get("ids")
        if not isinstance(values, (list, tuple)) or any(type(token) is not int for token in values):
            raise RLGenerationError(f"prompt {index} envelope IDs are not integer IDs")
        values = list(values)
        if not values or values[0] != BOS_ID:
            raise RLGenerationError(f"prompt {index} does not begin with manual BOS {BOS_ID}")
        if len(values) > getattr(self, "_campaign_prompt_max_tokens", PROMPT_MAX_TOKENS):
            raise RLGenerationError(f"prompt {index} exceeds the PRM03 prompt cap")
        if any(token < 0 or token >= VOCAB_SIZE for token in values):
            raise RLGenerationError(f"prompt {index} contains an out-of-range ID")
        if values.count(BOS_ID) != 1 or EOS_ID in values[1:]:
            raise RLGenerationError(f"prompt {index} violates one-BOS/no-EOS geometry")
        if any(is_native_control_token(token) for token in values[1:]):
            raise RLGenerationError(f"prompt {index} contains a native CONTROL ID")
        return values

    @staticmethod
    def _padded_prompt_tensors(prompt_ids: Sequence[Sequence[int]], torch_module: Any, device: Any, pad_id: int = EOS_ID):
        if not prompt_ids:
            raise RLGenerationError("generation batch is empty")
        width = max(len(ids) for ids in prompt_ids)
        input_ids = torch_module.full((len(prompt_ids), width), pad_id, dtype=torch_module.long, device=device)
        attention = torch_module.zeros((len(prompt_ids), width), dtype=torch_module.long, device=device)
        for index, ids in enumerate(prompt_ids):
            input_ids[index, width - len(ids):] = torch_module.tensor(ids, dtype=torch_module.long, device=device)
            attention[index, width - len(ids):] = 1
        return input_ids, attention, width

    def _checked_generation_kwargs(self) -> dict[str, Any]:
        protected = {
            "do_sample": True,
            "max_new_tokens": self._campaign_completion_max_tokens,
            "eos_token_id": list(NATIVE_EOG_IDS),
            "pad_token_id": EOS_ID,
            "bos_token_id": BOS_ID,
            "num_return_sequences": 1,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": True,
        }
        raw = dict(self._campaign_generation_kwargs)
        unknown = set(raw) - _GENERATION_KWARG_KEYS - set(protected)
        if unknown:
            raise RLGenerationError(f"generation kwargs contain unsupported keys: {sorted(unknown)}")
        sampling = {key: raw[key] for key in raw if key in _GENERATION_KWARG_KEYS}
        values = resolve_generation_kwargs(sampling)
        for key, expected in protected.items():
            if key in raw:
                actual = raw[key]
                if key == "eos_token_id":
                    actual = list(actual) if isinstance(actual, (list, tuple)) else actual
                if actual != expected:
                    raise RLGenerationError(f"generation kwarg {key} attempts to change pinned value")
            values[key] = expected
        return values

    def _generate_prompt_batch(
        self, prompt_groups: Sequence[Sequence[list[int]]], torch_module: Any,
    ) -> tuple[list[list[int]], list[list[int]], list[dict[str, Any]]]:
        """Generate adjacent complete G groups in one rectangular model call.

        The logical group boundary remains explicit: every inner sequence has
        exactly G identical stored prompts, and rows are flattened only for
        the model call.  The caller restores logical group and row indices
        from this unchanged order before reward/accounting.
        """
        if not prompt_groups:
            raise RLGenerationError("generation prompt batch is empty")
        group_size = int(self.num_generations)
        configured_groups = resolve_generation_groups_per_call(
            getattr(self, "_campaign_generation_groups_per_call", 1),
            "runtime generation_groups_per_call",
        )
        if len(prompt_groups) > configured_groups:
            raise RLGenerationError(
                f"generation call has {len(prompt_groups)} groups; "
                f"configured generation_groups_per_call={configured_groups}"
            )
        flattened_prompt_ids: list[list[int]] = []
        for group_index, prompt_ids in enumerate(prompt_groups):
            if not isinstance(prompt_ids, (list, tuple)) or len(prompt_ids) != group_size:
                raise RLGenerationError(
                    f"generation group {group_index} has "
                    f"{len(prompt_ids) if isinstance(prompt_ids, (list, tuple)) else 'non-sequence'} rows; "
                    f"expected G={group_size}"
                )
            first = tuple(prompt_ids[0])
            if any(tuple(ids) != first for ids in prompt_ids[1:]):
                raise RLGenerationError(
                    f"generation group {group_index} contains different stored prompts"
                )
            flattened_prompt_ids.extend(list(ids) for ids in prompt_ids)
        prompt_ids = flattened_prompt_ids
        device = self.accelerator.device
        generate_inputs, attention, padded_width = self._padded_prompt_tensors(
            prompt_ids, torch_module, device,
        )
        wrapped = getattr(self, "model_wrapped", self.model)
        generation_kwargs = self._checked_generation_kwargs()
        # Match the pinned TRL generation unwrap seam.  No tokenizer call
        # occurs here; generate receives the stored integer IDs directly.
        from trl.models import unwrap_model_for_generation
        with unwrap_model_for_generation(
            wrapped,
            self.accelerator,
            gather_deepspeed3_params=getattr(self.args, "ds3_gather_for_generation", False),
        ) as unwrapped:
            with torch_module.no_grad(), torch_module.inference_mode():
                output = unwrapped.generate(
                    input_ids=generate_inputs,
                    attention_mask=attention,
                    **generation_kwargs,
                )
        sequences = getattr(output, "sequences", output)
        if (getattr(sequences, "ndim", None) != 2
                or int(sequences.shape[0]) != len(prompt_ids)):
            raise RLGenerationError("generation returned an unexpected candidate batch")
        if int(sequences.shape[1]) < padded_width:
            raise RLGenerationError("generation output is shorter than its padded prompt")
        completions: list[list[int]] = []
        records: list[dict[str, Any]] = []
        for index, ids in enumerate(prompt_ids):
            expected_prefix = [EOS_ID] * (padded_width - len(ids)) + list(ids)
            observed_prefix = sequences[index, :padded_width].detach().tolist()
            if observed_prefix != expected_prefix:
                raise RLGenerationError(f"generation altered stored prompt IDs for prompt {index}")
            generated_tail = sequences[index, padded_width:].detach().tolist()
            if any(type(token) is not int for token in generated_tail):
                raise RLGenerationError("generation returned noninteger token IDs")
            trimmed = trim_generated_sequence(
                ids + generated_tail,
                len(ids),
                completion_max_tokens=self._campaign_completion_max_tokens,
            )
            generated = list(trimmed["generated_tokens"])
            if not generated:
                raise RLGenerationError(f"generation returned no completion tokens for prompt {index}")
            completions.append(generated)
            records.append({
                "prompt_tokens": list(ids),
                "generated_tokens": generated,
                "terminal_reason": trimmed["terminal_reason"],
                "padded_after_terminal": trimmed["padded_after_terminal"],
            })
        return [list(ids) for ids in prompt_ids], completions, records

    def _generate_prompt_group(
        self, prompt_ids: list[list[int]], torch_module: Any,
    ) -> tuple[list[list[int]], list[list[int]], list[dict[str, Any]]]:
        """Compatibility wrapper for one logical G-sized prompt group."""
        return self._generate_prompt_batch([prompt_ids], torch_module)

    def _generate_single_turn(self, prompts: list[Any], images: Any = None):
        generation_started = time.perf_counter()
        if not getattr(self, "_campaign_runtime_configured", False):
            raise RLGenerationError("CampaignGRPOTrainer runtime was not configured")
        if images is not None:
            raise RLGenerationError("PRM03 fixed-ID trainer does not admit image inputs")
        group_size = int(self.num_generations)
        if group_size not in DEFAULT_CANDIDATE_COUNTS:
            raise RLGenerationError(f"unsupported generation group size G={group_size}")
        groups_per_call = resolve_generation_groups_per_call(
            getattr(self, "_campaign_generation_groups_per_call", 1),
            "runtime generation_groups_per_call",
        )
        if not prompts:
            raise RLGenerationError("generation batch is empty")
        if len(prompts) % group_size:
            raise RLGenerationError("generation batch does not contain complete G-sized groups")
        prompt_ids = [self._prompt_ids(prompt, index) for index, prompt in enumerate(prompts)]
        if any(len(ids) + self._campaign_completion_max_tokens > self._campaign_context_max_tokens for ids in prompt_ids):
            raise RLGenerationError("prompt plus completion exceeds the managed context cap")
        try:
            import torch
        except Exception as error:  # pragma: no cover - live environment-specific
            raise RLGenerationError(f"cannot import torch for live generation: {error}") from error

        all_prompt_ids: list[list[int]] = []
        all_completions: list[list[int]] = []
        all_records: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        guard_factory = self._campaign_generation_guard_factory
        guard = guard_factory(self.model) if guard_factory is not None else nullcontext()
        with guard:
            self.model.eval()
            call_width = group_size * groups_per_call
            for call_index, start in enumerate(range(0, len(prompt_ids), call_width)):
                end = min(start + call_width, len(prompt_ids))
                if (end - start) % group_size:
                    raise RLGenerationError("generation call ended inside a logical G group")
                call_groups = [
                    prompt_ids[group_start:group_start + group_size]
                    for group_start in range(start, end, group_size)
                ]
                group_prompt_ids, group_completions, group_records = self._generate_prompt_batch(
                    call_groups, torch,
                )
                for local_index, record in enumerate(group_records):
                    record["group_index"] = start // group_size + local_index // group_size
                    record["group_row_index"] = local_index % group_size
                    record["generation_call_index"] = call_index
                all_prompt_ids.extend(group_prompt_ids)
                all_completions.extend(group_completions)
                all_records.extend(group_records)
                for local_group_index in range(len(call_groups)):
                    group_start = start + local_group_index * group_size
                    group_end = group_start + group_size
                    group_prompt_ids_slice = group_prompt_ids[
                        local_group_index * group_size:(local_group_index + 1) * group_size
                    ]
                    groups.append({
                        "group_index": group_start // group_size,
                        "start": group_start,
                        "end": group_end,
                        "candidate_count": group_size,
                        "call_index": call_index,
                        "prompt_ids": [list(ids) for ids in group_prompt_ids_slice],
                    })
        if self._campaign_post_generation_restore is not None:
            self._campaign_post_generation_restore(self.model, self.processing_class)
        try:
            accounting = account_generation_records(
                all_records,
                prompt_max_tokens=self._campaign_prompt_max_tokens,
                completion_max_tokens=self._campaign_completion_max_tokens,
            )
        except (RLDataError, ValueError, RuntimeError) as error:
            raise RLGenerationError(f"generation accounting failed: {error}") from error
        self._campaign_last_generation = {
            "records": all_records,
            "accounting": accounting,
            "prompt_count": len(all_prompt_ids),
            "candidate_count": group_size,
            "groups": groups,
            "generation_batch_size": len(all_prompt_ids),
            "generation_chunk_size": group_size * groups_per_call,
            "generation_groups_per_call": groups_per_call,
            "generation_call_count": len({group.get("call_index") for group in groups}),
        }
        _emit_generation_telemetry(
            self, all_records, accounting, groups,
            time.perf_counter() - generation_started,
        )
        return all_prompt_ids, all_completions, None, {}


def _import_pinned_grpo_trainer() -> type[Any]:
    """Import TRL's trainer while honoring its optional-dependency flags.

    TRL 0.24.0 calls Transformers' ``_is_package_available`` with its
    ``return_version`` result for every optional package.  The result is a
    ``(bool, version)`` tuple, and the public TRL predicates therefore treat a
    missing optional package such as mergekit/llm_blender as truthy.  That
    makes importing ``GRPOTrainer`` fail before the trainer class is usable.
    Normalize those flags at this boundary; no optional package is installed
    or imported, and the correction is scoped to the current process.
    """
    try:
        import trl.import_utils as trl_import_utils
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLTrainError(f"cannot import the pinned TRL import boundary: {error}") from error
    for name in dir(trl_import_utils):
        if not name.startswith("_") or not name.endswith("_available"):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple) and len(value) == 2 and type(value[0]) is bool:
            setattr(trl_import_utils, name, value[0])
    try:
        from trl import GRPOTrainer
    except Exception as error:  # pragma: no cover - environment-specific
        raise RLTrainError(f"cannot import the pinned TRL GRPOTrainer: {error}") from error
    return GRPOTrainer


def campaign_grpo_trainer_class() -> type[Any]:
    """Return the pinned dynamic subclass without importing TRL at module load."""
    GRPOTrainer = _import_pinned_grpo_trainer()

    class CampaignGRPOTrainer(FixedIDGRPOTrainerMixin, GRPOTrainer):
        def _get_train_sampler(self, dataset=None):
            dataset = self.train_dataset if dataset is None else dataset
            if self.shuffle_dataset:
                raise RLTrainError("shuffle_dataset must remain false for the first campaign sampler")
            generation_batch_size = int(self.args.generation_batch_size)
            if generation_batch_size % int(self.num_generations):
                raise RLTrainError("TRL generation batch is not divisible by num_generations")
            expected_dataloader_batch = int(self._train_batch_size) * int(self.args.steps_per_generation)
            if expected_dataloader_batch != generation_batch_size:
                raise RLTrainError(
                    "one-generation geometry mismatch: dataloader batch does not equal generation batch"
                )
            if int(self.num_iterations) != 1:
                raise RLTrainError("num_iterations must remain 1; stale rollout reuse is not admitted")
            if int(self.args.steps_per_generation) != int(self.args.gradient_accumulation_steps):
                raise RLTrainError(
                    "steps_per_generation must equal gradient_accumulation_steps for one buffer per update"
                )
            source_draw_sequence = getattr(self, "_campaign_source_draw_sequence", None)
            if getattr(self, "_campaign_source_draw_schedule_required", False) and source_draw_sequence is None:
                raise RLTrainError(
                    "the admitted RL recipe requires a frozen source draw schedule"
                )
            return CampaignRepeatSampler(
                dataset,
                candidate_count=int(self.num_generations),
                prompt_groups_per_batch=generation_batch_size // int(self.num_generations),
                repeat_count=int(self.num_iterations) * int(self.args.steps_per_generation),
                seed=int(self.args.seed),
                shuffle=False,
                generation_batch_size=generation_batch_size,
                per_device_train_batch_size=int(self._train_batch_size),
                gradient_accumulation_steps=int(self.args.gradient_accumulation_steps),
                steps_per_generation=int(self.args.steps_per_generation),
                source_draw_sequence=source_draw_sequence,
                source_draw_sequence_sha256=getattr(self, "_campaign_source_draw_sequence_sha256", None),
                source_draw_schedule_sha256=getattr(self, "_campaign_source_draw_schedule_sha256", None),
            )

        def campaign_sampler_state(self) -> dict[str, Any]:
            sampler = self._get_train_sampler(self.train_dataset)
            state = sampler.state(
                # TRL's dataloader consumes one generation-sized batch for
                # each accumulation step.  The first buffer is reused for
                # those batches, so the sampler stream advances by the
                # generation batch multiplied by steps_per_generation at an
                # optimizer checkpoint boundary.
                consumed_rows=(
                    int(self.state.global_step)
                    * int(self.args.generation_batch_size)
                    * int(self.args.steps_per_generation)
                ),
                epoch=int(self.state.epoch or 0),
            )
            # Keep the ordered selected-ID list/hash in the checkpoint's
            # sampler payload.  Trainer's data-skip mechanism reconstructs
            # the stream from consumed_rows, while this identity prevents a
            # later dataset with the same length from being mistaken for it.
            data_identity = getattr(self, "_campaign_data_manifest", None)
            if data_identity is not None:
                state["data_identity"] = data_identity
            return state

    CampaignGRPOTrainer.__name__ = "CampaignGRPOTrainer"
    return CampaignGRPOTrainer


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RLTrainError(f"{name} must be an object")
    return value


def validate_rl_recipe(recipe: Mapping[str, Any]) -> tuple[dict[str, Any], TRLGeometry]:
    """Validate identity, admission, policy and geometry without model imports."""
    if recipe.get("schema_version") != TRAIN_SCHEMA_VERSION:
        raise RLTrainError("RL recipe schema mismatch")
    identity = dict(_require_mapping(recipe.get("identity"), "identity"))
    try:
        checked_identity = check_identity(identity)
    except (TypeError, ValueError) as error:
        raise RLTrainError(f"incomplete RL identity: {error}") from error
    training_mode = recipe.get("training_mode", "adapter")
    if training_mode not in {"adapter", "full_weights"}:
        raise RLTrainError("training_mode must be adapter or full_weights")
    parent = _require_mapping(checked_identity["parent"], "identity.parent")
    expected_parent_kind = "merged_sft" if training_mode == "adapter" else "full_weight_edit_sft_checkpoint"
    if parent.get("kind") != expected_parent_kind:
        raise RLTrainError(f"identity.parent.kind must be {expected_parent_kind}")
    _sha256(parent.get("manifest_sha256"), "identity.parent.manifest_sha256")
    parent_weight_key = "merged_weights_sha256" if training_mode == "adapter" else "model_weights_sha256"
    _sha256(parent.get(parent_weight_key), f"identity.parent.{parent_weight_key}")
    tokenizer = _require_mapping(checked_identity["tokenizer"], "identity.tokenizer")
    if (tokenizer.get("vocab_size") != VOCAB_SIZE or tokenizer.get("bos_id") != BOS_ID
            or tokenizer.get("eos_id") != EOS_ID or tokenizer.get("pad_id") != EOS_ID
            or list(tokenizer.get("native_eog_ids", ())) != list(NATIVE_EOG_IDS)):
        raise RLTrainError("identity.tokenizer does not match the pinned PRM03 IDs")
    renderer = _require_mapping(checked_identity["renderer"], "identity.renderer")
    if renderer.get("renderer_id") != RENDERER_ID or renderer.get("tokenization_policy") != TOKENIZATION_POLICY:
        raise RLTrainError("identity.renderer does not match PRM03")
    if renderer.get("terminal") != TERMINAL or renderer.get("no_edit") != "[NO_EDIT]":
        raise RLTrainError("identity.renderer terminal/no-edit markers mismatch")
    policy = _require_mapping(checked_identity["policy"], "identity.policy")
    if training_mode == "adapter":
        if (policy.get("lora_rank") != LORA_RANK or policy.get("lora_alpha") != LORA_ALPHA
                or tuple(policy.get("target_modules", ())) != TARGET_MODULES
                or policy.get("expected_attachments") != EXPECTED_ATTACHMENTS
                or policy.get("expected_trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS):
            raise RLTrainError("identity.policy does not match the adapter RL profile contract")
    elif (policy.get("checkpoint_kind") != "full_weights"
          or policy.get("expected_parameter_tensors") != 381
          or policy.get("expected_trainable_parameters") != 2_516_756_480
          or policy.get("peft_tensors") != 0):
        raise RLTrainError("identity.policy does not match the dense RL profile contract")
    if (policy.get("prompt_max_tokens") != PROMPT_MAX_TOKENS
            or policy.get("completion_max_tokens") != COMPLETION_MAX_TOKENS):
        raise RLTrainError("identity.policy does not match the RL token geometry")
    policy_groups_per_call = resolve_generation_groups_per_call(
        policy.get("generation_groups_per_call"),
        "identity.policy.generation_groups_per_call",
    )
    recipe_groups_per_call = resolve_generation_groups_per_call(
        recipe.get("generation_groups_per_call"),
        "recipe.generation_groups_per_call",
    )
    if recipe_groups_per_call != policy_groups_per_call:
        raise RLTrainError(
            "recipe.generation_groups_per_call differs from identity.policy"
        )
    for name in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if name not in policy:
            raise RLTrainError(f"identity.policy.{name} is required to bind the profiled update shape")
    candidate_count = _int(policy.get("candidate_count"), "identity.policy.candidate_count", 2)
    geometry = resolve_trl_geometry(
        candidate_count,
        rollout_rows_per_update=_int(policy.get("rollout_rows_per_update"), "identity.policy.rollout_rows_per_update", candidate_count),
        per_device_train_batch_size=_int(
            policy.get("per_device_train_batch_size"),
            "identity.policy.per_device_train_batch_size", candidate_count,
        ),
        gradient_accumulation_steps=_int(
            policy.get("gradient_accumulation_steps"),
            "identity.policy.gradient_accumulation_steps", 1,
        ),
    )
    resolve_generation_groups_per_call(
        recipe_groups_per_call,
        "generation_groups_per_call",
        max_groups=geometry.prompt_groups_per_update,
    )
    if policy.get("loss_type") != "bnpo" or policy.get("scale_rewards") != "group" or policy.get("beta") != 0:
        raise RLTrainError("first RL identity must explicitly use BNPO/group/beta=0")
    if policy.get("reward_policy_id") != "protocol_restraint_syntax_coverage_v2":
        raise RLTrainError("identity.policy.reward_policy_id differs")
    for name in ("reward_source_sha256", "reward_parse_harness_sha256"):
        _sha256(policy.get(name), f"identity.policy.{name}")
    sampling = policy.get("sampling")
    if not isinstance(sampling, Mapping):
        raise RLTrainError("identity.policy.sampling must record the generation policy")
    if resolve_generation_kwargs(sampling) != dict(sampling):
        raise RLTrainError("identity.policy.sampling must contain exactly the validated generation policy")
    recipe_sampling = resolve_generation_kwargs(recipe.get("generation_kwargs"))
    if recipe.get("generation_kwargs") is not None and recipe_sampling != dict(sampling):
        raise RLTrainError("recipe.generation_kwargs differs from identity.policy.sampling")
    schedule = _require_mapping(checked_identity["schedule"], "identity.schedule")
    if schedule.get("seed") != DEFAULT_SEED or schedule.get("sampler_id") != CampaignRepeatSampler.SAMPLER_ID:
        raise RLTrainError("identity.schedule sampler/seed mismatch")
    if (schedule.get("generation_batch_size") != geometry.generation_batch_size
            or schedule.get("steps_per_generation") != geometry.steps_per_generation
            or schedule.get("num_iterations") != 1):
        raise RLTrainError("first RL identity must bind one generation buffer per optimizer update")
    source_schedule_fields = (
        "source_draw_schedule_sha256", "source_draw_sequence_sha256", "source_draws",
        "source_draws_per_update", "buffer_reuse",
    )
    source_schedule_present = any(field in schedule for field in source_schedule_fields)
    if source_schedule_present:
        if any(field not in schedule for field in source_schedule_fields):
            raise RLTrainError("identity.schedule source draw binding is incomplete")
        _sha256(schedule["source_draw_schedule_sha256"], "identity.schedule.source_draw_schedule_sha256")
        _sha256(schedule["source_draw_sequence_sha256"], "identity.schedule.source_draw_sequence_sha256")
        source_draws = _int(schedule["source_draws"], "identity.schedule.source_draws", 1)
        source_draws_per_update = _int(
            schedule["source_draws_per_update"],
            "identity.schedule.source_draws_per_update",
            1,
        )
        if source_draws_per_update != geometry.prompt_groups_per_update:
            raise RLTrainError("identity source draw geometry differs from generation geometry")
        if schedule["buffer_reuse"] != geometry.steps_per_generation:
            raise RLTrainError("identity source draw buffer reuse differs from TRL geometry")
        for name in ("gradient_accumulation_steps", "steps_per_generation"):
            if name in schedule and schedule[name] != geometry.steps_per_generation:
                raise RLTrainError(f"identity source draw {name} differs from TRL geometry")
        if source_draws % source_draws_per_update:
            raise RLTrainError("identity source draw schedule ends inside an optimizer update")
    data = _require_mapping(checked_identity["data"], "identity.data")
    if data.get("split") != "train" or data.get("admission_status") != "admitted":
        raise RLTrainError("identity.data must explicitly name admitted train data")
    for name in (
        "rows_sha256", "context_sha256", "selected_ids_sha256", "ordered_ids_sha256",
        "sidecar_artifact_sha256", "row_identity_sha256", "row_identities",
        "selected_ids", "row_count", "reward_buffer_manifest_sha256",
        "reward_buffer_sidecar_sha256",
    ):
        if name not in data:
            raise RLTrainError(f"identity.data.{name} is required")
    for name in (
        "rows_sha256", "context_sha256", "sidecar_artifact_sha256",
        "selected_ids_sha256", "ordered_ids_sha256", "row_identity_sha256",
        "reward_buffer_manifest_sha256", "reward_buffer_sidecar_sha256",
    ):
        _sha256(data[name], f"identity.data.{name}")
    if data["sidecar_artifact_sha256"] != data["context_sha256"]:
        raise RLTrainError("identity.data.sidecar_artifact_sha256 must equal context_sha256")
    row_count = _int(data["row_count"], "identity.data.row_count", 1)
    selected_ids = data["selected_ids"]
    if (not isinstance(selected_ids, list) or len(selected_ids) != row_count
            or any(type(row_id) is not str or not row_id for row_id in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise RLTrainError("identity.data.selected_ids must be a unique ordered list matching row_count")
    row_identities = data["row_identities"]
    if (not isinstance(row_identities, list) or len(row_identities) != row_count
            or any(not isinstance(item, Mapping) for item in row_identities)
            or [item.get("id") for item in row_identities] != selected_ids):
        raise RLTrainError("identity.data.row_identities must preserve selected ID order")
    if "context_snapshot_sha256" in data:
        _sha256(data["context_snapshot_sha256"], "identity.data.context_snapshot_sha256")
    paths = _require_mapping(recipe.get("data"), "recipe.data")
    for name in ("rows_path", "rows_sha256", "selected_ids_path", "selected_ids_sha256",
                 "reward_buffer_manifest_path", "reward_buffer_manifest_sha256"):
        if name not in paths:
            raise RLTrainError(f"recipe.data.{name} is required")
    if not (paths.get("sidecar_path", paths.get("context_path"))
            and paths.get("sidecar_artifact_sha256", paths.get("context_sha256"))):
        raise RLTrainError("recipe.data.sidecar_path/sidecar_artifact_sha256 (context aliases accepted) are required")
    if paths["reward_buffer_manifest_sha256"] != data["reward_buffer_manifest_sha256"]:
        raise RLTrainError("recipe reward-buffer manifest differs from identity.data")
    parse_spec = _require_mapping(recipe.get("reward_parse"), "recipe.reward_parse")
    if (parse_spec.get("operation") != "base::parse_only"
            or parse_spec.get("generated_r_executed") is not False
            or parse_spec.get("harness_sha256") != policy["reward_parse_harness_sha256"]):
        raise RLTrainError("recipe reward parse binding differs from identity.policy")
    if source_schedule_present:
        for name in (
            "source_draw_schedule_path", "source_draw_schedule_sha256",
            "source_draw_sequence_sha256", "source_draws", "buffer_reuse",
        ):
            if name not in paths:
                raise RLTrainError(f"recipe.data.{name} is required with source draw binding")
        _sha256(paths["source_draw_schedule_sha256"], "recipe.data.source_draw_schedule_sha256")
        _sha256(paths["source_draw_sequence_sha256"], "recipe.data.source_draw_sequence_sha256")
        if paths["source_draws"] != schedule["source_draws"]:
            raise RLTrainError("recipe.data.source_draws differs from identity.schedule")
        if paths["buffer_reuse"] != schedule["buffer_reuse"]:
            raise RLTrainError("recipe.data.buffer_reuse differs from identity.schedule")
    if (("sidecar_path" in paths or "sidecar_artifact_sha256" in paths)
            and paths.get("context_snapshot_sha256") is not None):
        raise RLTrainError("RL-02 sidecar recipes cannot bind one common context snapshot")
    return dict(checked_identity), geometry


def _assert_manifest_identity(manifest: RLDataManifest, identity_data: Mapping[str, Any]) -> None:
    actual = manifest.to_identity()
    for name in (
        "rows_sha256", "context_sha256", "sidecar_artifact_sha256",
        "selected_ids_sha256", "ordered_ids_sha256", "row_identity_sha256",
        "row_identities", "row_count", "split", "admission_status", "selected_ids",
        "reward_buffer_manifest_sha256", "reward_buffer_sidecar_sha256",
    ):
        if actual.get(name) != identity_data.get(name):
            raise RLTrainError(
                f"loaded data identity differs at {name}: "
                f"{actual.get(name)!r} != {identity_data.get(name)!r}"
            )
    if "context_snapshot_sha256" in identity_data:
        if actual.get("context_snapshot_sha256") != identity_data.get("context_snapshot_sha256"):
            raise RLTrainError(
                "loaded data identity differs at context_snapshot_sha256: "
                f"{actual.get('context_snapshot_sha256')!r} != {identity_data.get('context_snapshot_sha256')!r}"
            )


def _load_recipe_source_draw_binding(
    recipe: Mapping[str, Any],
    identity: Mapping[str, Any],
    geometry: TRLGeometry,
    manifest: RLDataManifest,
) -> tuple[dict[str, Any], list[int]]:
    """Load the frozen source draw stream and map its IDs to dataset indices."""
    identity_schedule = _require_mapping(identity.get("schedule"), "identity.schedule")
    paths = _require_mapping(recipe.get("data"), "recipe.data")
    schedule_path = paths.get("source_draw_schedule_path")
    schedule_sha = paths.get("source_draw_schedule_sha256")
    if not isinstance(schedule_path, str) or not isinstance(schedule_sha, str):
        raise RLTrainError(
            "recipe.data.source_draw_schedule_path/source_draw_schedule_sha256 are required"
        )
    if schedule_sha != identity_schedule.get("source_draw_schedule_sha256"):
        raise RLTrainError("recipe source draw schedule hash differs from identity")
    if paths.get("source_draw_sequence_sha256") != identity_schedule.get("source_draw_sequence_sha256"):
        raise RLTrainError("recipe source draw sequence hash differs from identity")
    if paths.get("source_draws") != identity_schedule.get("source_draws"):
        raise RLTrainError("recipe source draw count differs from identity")
    if paths.get("buffer_reuse") != identity_schedule.get("buffer_reuse"):
        raise RLTrainError("recipe source draw buffer reuse differs from identity")
    schedule = load_source_draw_schedule(
        Path(schedule_path), schedule_sha,
        selected_ids=manifest.selected_ids,
        selected_ids_sha256=manifest.selected_ids_sha256,
        ordered_ids_sha256=manifest.ordered_ids_sha256,
        row_identity_sha256=manifest.row_identity_sha256,
        candidate_count=geometry.candidate_count,
        source_draws_per_update=geometry.prompt_groups_per_update,
        buffer_reuse=geometry.steps_per_generation,
    )
    if schedule["source_draws"] != identity_schedule["source_draws"]:
        raise RLTrainError("loaded source draw count differs from identity")
    selected_index = {row_id: index for index, row_id in enumerate(manifest.selected_ids)}
    try:
        indices = [selected_index[row_id] for row_id in schedule["row_ids"]]
    except KeyError as error:  # pragma: no cover - loader normally catches this
        raise RLTrainError(f"source draw schedule row is outside selected IDs: {error}") from error
    return schedule, indices


def preflight_rl_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact files and return a framework-free launch packet."""
    identity, geometry = validate_rl_recipe(recipe)
    paths = _require_mapping(recipe["data"], "recipe.data")
    sidecar_path = paths.get("sidecar_path", paths.get("context_path"))
    sidecar_sha = paths.get("sidecar_artifact_sha256", paths.get("context_sha256"))
    records, manifest = load_training_records(
        Path(paths["rows_path"]), paths["rows_sha256"],
        Path(sidecar_path), sidecar_sha,
        Path(paths["selected_ids_path"]), paths["selected_ids_sha256"],
        admission_status=paths.get("admission_status", "admitted"),
        reward_buffer_manifest_path=Path(paths["reward_buffer_manifest_path"]),
        reward_buffer_manifest_sha256=paths["reward_buffer_manifest_sha256"],
    )
    _assert_manifest_identity(manifest, identity["data"])
    source_draw_schedule = None
    identity_schedule = _require_mapping(identity.get("schedule"), "identity.schedule")
    if "source_draw_schedule_sha256" in identity_schedule:
        source_draw_schedule, _ = _load_recipe_source_draw_binding(
            recipe, identity, geometry, manifest,
        )
    if len(records) < geometry.prompt_groups_per_update:
        raise RLTrainError(
            f"only {len(records)} selected rows; {geometry.prompt_groups_per_update} prompt groups per update required"
        )
    return {
        "status": "preflight_pass",
        "framework_imports": {name: name in sys.modules for name in ("torch", "transformers", "trl", "unsloth")},
        "identity": identity,
        "data": manifest.to_identity(),
        "selected_row_ids": list(manifest.selected_ids),
        "geometry": geometry.to_dict(),
        "limits": {
            "prompt_max_tokens": PROMPT_MAX_TOKENS,
            "completion_max_tokens": COMPLETION_MAX_TOKENS,
            "managed_context_max_tokens": CONTEXT_MAX_TOKENS,
        },
        "records": len(records),
        "source_draw_schedule": source_draw_schedule,
        "live_launch": "blocked until lead supplies the verified merged-SFT parent, runtime transition helper, and resource lease",
    }


def _factory(value: object, name: str) -> Any:
    if not isinstance(value, str) or ":" not in value:
        raise RLTrainError(f"{name} must be module:function")
    module_name, function_name = value.split(":", 1)
    try:
        return getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise RLTrainError(f"cannot load {name}: {value}") from error


def build_live_trainer(
    recipe: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    fast_model: Any | None = None,
    reference_tokenizer: Any | None = None,
    generation_guard_factory: Callable[[Any], Any] | None = None,
    post_generation_restore: Callable[[Any, Any], None] | None = None,
    post_trainer_contract: Callable[[Any, Any, Any, Sequence[Mapping[str, Any]]], Mapping[str, Any]] | None = None,
    resource_probe: Callable[[], Mapping[str, Any]] | None = None,
    loaded_parent_identity: Mapping[str, Any] | None = None,
    evaluation_context: Callable[[Any], Any] | None = None,
    reward_parse_probe: Callable[[str], bool] | None = None,
    trainer_class_factory: Callable[[], type[Any]] | None = None,
    checkpoint_kind: str = "adapter",
    production_binding: Mapping[str, Any] | None = None,
    full_weight_optimizer_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build, but do not train, an admitted single-process GRPO trainer.

    Model loading, resource ownership, and the parent merge happen outside this
    function.  The caller must pass the lead-verified model/tokenizer pair.
    """
    identity, geometry = validate_rl_recipe(recipe)
    paths = _require_mapping(recipe["data"], "recipe.data")
    sidecar_path = paths.get("sidecar_path", paths.get("context_path"))
    sidecar_sha = paths.get("sidecar_artifact_sha256", paths.get("context_sha256"))
    records, manifest = load_training_records(
        Path(paths["rows_path"]), paths["rows_sha256"],
        Path(sidecar_path), sidecar_sha,
        Path(paths["selected_ids_path"]), paths["selected_ids_sha256"],
        admission_status=paths.get("admission_status", "admitted"),
        reward_buffer_manifest_path=Path(paths["reward_buffer_manifest_path"]),
        reward_buffer_manifest_sha256=paths["reward_buffer_manifest_sha256"],
    )
    _assert_manifest_identity(manifest, identity["data"])
    source_draw_schedule, source_draw_indices = _load_recipe_source_draw_binding(
        recipe, identity, geometry, manifest,
    )
    if loaded_parent_identity is None:
        raise RLTrainError("lead must provide the identity of the already-loaded merged-SFT parent")
    loaded_parent_identity = _require_mapping(loaded_parent_identity, "loaded_parent_identity")
    expected_parent = identity["parent"]
    weight_key = "model_weights_sha256" if recipe.get("training_mode") == "full_weights" else "merged_weights_sha256"
    if (loaded_parent_identity.get("manifest_sha256") != expected_parent.get("manifest_sha256")
            or loaded_parent_identity.get(weight_key) != expected_parent.get(weight_key)):
        raise RLTrainError("loaded model parent identity differs from the admitted SFT parent")
    sampling = resolve_generation_kwargs(recipe.get("generation_kwargs", identity["policy"]["sampling"]))
    if sampling != dict(identity["policy"]["sampling"]):
        raise RLTrainError("live generation policy differs from the admitted identity")
    generation_groups_per_call = resolve_generation_groups_per_call(
        recipe.get("generation_groups_per_call"),
        "recipe.generation_groups_per_call",
        max_groups=geometry.prompt_groups_per_update,
    )
    if generation_groups_per_call != identity["policy"].get("generation_groups_per_call"):
        raise RLTrainError(
            "live generation_groups_per_call differs from identity.policy"
        )
    config, _ = make_grpo_config(
        candidate_count=geometry.candidate_count,
        rollout_rows_per_update=geometry.rollout_rows_per_update,
        per_device_train_batch_size=geometry.per_device_train_batch_size,
        gradient_accumulation_steps=geometry.gradient_accumulation_steps,
        output_dir=recipe["output_dir"],
        max_steps=_int(recipe.get("max_steps", 1), "max_steps", 1),
        save_steps=_int(recipe.get("full_save_steps", DEFAULT_FULL_SAVE_STEPS), "full_save_steps", 1),
        learning_rate=recipe.get("learning_rate", DEFAULT_LEARNING_RATE),
        warmup_steps=recipe.get("warmup_steps", DEFAULT_WARMUP_STEPS),
        use_cpu=False,
        generation_kwargs=sampling,
    )
    if checkpoint_kind not in {"adapter", "full_weights"}:
        raise RLTrainError("checkpoint_kind must be adapter or full_weights")
    if checkpoint_kind == "full_weights":
        if trainer_class_factory is None or production_binding is None or full_weight_optimizer_config is None:
            raise RLTrainError("full_weights trainer requires production class, binding and optimizer config")
    elif any(value is not None for value in (
        trainer_class_factory, production_binding, full_weight_optimizer_config,
    )):
        raise RLTrainError("production trainer inputs cannot enter adapter training")
    trainer_class = (
        trainer_class_factory() if trainer_class_factory is not None
        else campaign_grpo_trainer_class()
    )
    from datasets import Dataset
    reward_buffer_index = RewardBufferIndex.load(
        Path(paths["reward_buffer_manifest_path"]),
        paths["reward_buffer_manifest_sha256"],
        manifest.selected_ids,
    )
    if reward_parse_probe is None:
        parse_spec = _require_mapping(recipe.get("reward_parse"), "recipe.reward_parse")
        if (parse_spec.get("operation") != "base::parse_only"
                or parse_spec.get("generated_r_executed") is not False):
            raise RLTrainError("reward parse service must be fixed parse-only")
        reward_parse_probe = RParseOnlyProbe(
            Path(parse_spec["harness_path"]),
            str(parse_spec["harness_sha256"]),
            timeout_seconds=float(parse_spec.get("timeout_seconds", 5.0)),
        )
    reward = CampaignPRM03Reward(
        tokenizer=tokenizer,
        reward_buffer_index=reward_buffer_index,
        parse_probe=reward_parse_probe,
    )
    dataset = Dataset.from_list(records_to_dataset_rows(records))
    output_dir = Path(recipe["output_dir"])
    generation_record_sink = DurableTelemetrySink(output_dir / "generation-records.jsonl")
    gradient_record_sink = DurableTelemetrySink(output_dir / "gradient-records.jsonl")
    gradient_callback = gradient_telemetry_callback(gradient_record_sink)

    if generation_guard_factory is None:
        if fast_model is None or not callable(getattr(fast_model, "for_training", None)):
            raise RLTrainError("a verified generation guard or FastLanguageModel.for_training hook is required")
        from campaign_sft import training_configuration_guard
        def generation_guard(actual_model: Any) -> Any:
            return training_configuration_guard(actual_model, fast_model.for_training)
        generation_guard_factory = generation_guard
    if reference_tokenizer is None and post_trainer_contract is None:
        raise RLTrainError("a pinned reference tokenizer or post-trainer contract checker is required")

    archive_root = Path(recipe["archive_root"])
    deadline = recipe.get("supervised_deadline") or recipe.get("deadline")
    if not isinstance(deadline, str) or not deadline:
        raise RLTrainError("a supervised UTC deadline is required")
    from campaign_control import control_callback
    from campaign_checkpoint import checkpoint_callback
    control = control_callback(
        telemetry_path=Path(recipe["telemetry_path"]),
        identity=identity,
        deadline=deadline,
        reserve_seconds=float(recipe["checkpoint_reserve_seconds"]),
        stop_steps=tuple(recipe.get("decision_steps", ())),
        resource_probe=resource_probe,
    )
    evaluator = None
    if recipe.get("evaluator_factory"):
        evaluator_factory = _factory(recipe["evaluator_factory"], "evaluator_factory")
        evaluator = evaluator_factory(recipe)
    trainer_box: dict[str, Any] = {}
    def sampler_state() -> dict[str, Any]:
        trainer = trainer_box.get("trainer")
        if trainer is None:
            return {"status": "pre-trainer", "data": manifest.to_identity()}
        return trainer.campaign_sampler_state()

    # The callback captures the state factory before the trainer exists; the
    # closure resolves it only at on_save, after construction.
    checkpoint = checkpoint_callback(
        identity=identity,
        archive_root=archive_root,
        tokenizer=tokenizer,
        light_every=_int(recipe.get("light_save_steps", DEFAULT_LIGHT_SAVE_STEPS), "light_save_steps", 1),
        full_every=_int(recipe.get("full_save_steps", DEFAULT_FULL_SAVE_STEPS), "full_save_steps", 1),
        evaluator=evaluator,
        allow_pending_evaluation=bool(recipe.get("allow_pending_evaluation", False)),
        evaluation_allowed=control.evaluation_allowed,
        evaluation_steps=tuple(recipe.get("evaluation_steps", ())),
        evaluation_context=evaluation_context,
        sampler_state=sampler_state,
        checkpoint_kind=checkpoint_kind,
    )
    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=dataset,
        callbacks=[control, checkpoint, gradient_callback],
        args=config,
    )
    if checkpoint_kind == "full_weights":
        from full_weight_rl_production import install_production_runtime
        install_production_runtime(
            trainer, production_binding, full_weight_optimizer_config,
        )
    trainer_box["trainer"] = trainer
    # The sampler receives the verified source-presentation sequence before
    # TRL constructs its dataloader.  It then expands each source draw by G
    # and repeats the generation buffer for accumulation; selected IDs remain
    # the unique data identity stored in ``manifest``.
    trainer._campaign_source_draw_sequence = tuple(source_draw_indices)
    trainer._campaign_source_draw_sequence_sha256 = source_draw_schedule["sequence_sha256"]
    trainer._campaign_source_draw_schedule_sha256 = source_draw_schedule["sha256"]
    trainer._campaign_source_draw_schedule_required = True
    trainer.configure_campaign_runtime(
        generation_guard_factory=generation_guard_factory,
        post_generation_restore=post_generation_restore,
        generation_record_sink=generation_record_sink,
        prompt_max_tokens=PROMPT_MAX_TOKENS,
        completion_max_tokens=COMPLETION_MAX_TOKENS,
        context_max_tokens=CONTEXT_MAX_TOKENS,
        generation_kwargs=sampling,
        generation_groups_per_call=generation_groups_per_call,
    )
    if post_trainer_contract is None:
        from campaign_sft import assert_post_trainer_pinned_identity
        post_trainer_contract = assert_post_trainer_pinned_identity
    trainer_tokenizer = getattr(trainer, "processing_class", tokenizer)
    trainer._campaign_post_trainer_contract = post_trainer_contract(
        model, trainer_tokenizer, reference_tokenizer, [record.row for record in records],
    )
    trainer._campaign_data_manifest = manifest.to_identity()
    trainer._campaign_identity = identity
    trainer._campaign_geometry = geometry.to_dict()
    trainer._campaign_reward = reward
    trainer._campaign_generation_records_path = str(generation_record_sink.path)
    trainer._campaign_gradient_records_path = str(gradient_record_sink.path)
    return trainer


def preflight_file(recipe_path: Path) -> dict[str, Any]:
    """Read a recipe JSON and return the framework-free preflight result."""
    try:
        recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RLTrainError(f"cannot read RL recipe: {recipe_path}") from error
    if not isinstance(recipe, Mapping):
        raise RLTrainError("RL recipe must be a JSON object")
    return preflight_rl_recipe(recipe)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    from campaign_checkpoint import write_json
    try:
        result = preflight_file(args.recipe)
    except (RLTrainError, RLDataError) as error:
        result = {"status": "preflight_failed", "error": str(error)}
        write_json(args.receipt, result)
        print(json.dumps(result, sort_keys=True))
        return 2
    write_json(args.receipt, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
