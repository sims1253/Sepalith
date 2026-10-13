#!/usr/bin/env python3
"""Bounded PRM-07/RL-01 policy-logprob and memory probe.

``--preflight`` validates a hashed, explicit train-candidate JSONL file and a
hashed train-ID manifest, then checks the pinned MiniCPM artifacts without
importing torch, Unsloth, Transformers, or TRL.  ``--profile`` repeats those
checks, takes the live process/VRAM guard, and only then loads a disposable
base plus a fresh r16/a16 adapter.

The live probe generates two bounded candidate batches (2 and 4 candidates
per explicit prompt by default), then teacher-forces those exact returned
integer IDs with an attention mask and backpropagates their mean negative
policy log probability.  This measures the generation-to-policy-gradient
mechanism and its memory cost.  It is not GRPO training, reward evidence, or
a quality claim.  Candidate prompts contain the stored manual BOS ID; no
prompt text is re-rendered or re-tokenized.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


EXECUTION_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SRC = EXECUTION_ROOT / "packages" / "sepalith" / "src"
if str(PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_SRC))

# campaign_profile is intentionally framework-free at import time.  Reusing
# its input/model identity and live guard keeps the two PRM-07 lanes on one
# immutable candidate and MiniCPM contract.
from campaign_profile import (  # noqa: E402
    EXPECTED_MODEL_PATH,
    EXPECTED_MODEL_REVISION,
    EXPECTED_WEIGHT_SHA256,
    EXPECTED_WEIGHT_BYTES,
    EXPECTED_TOKENIZER_JSON_SHA256,
    EXPECTED_TOKENIZER_CONFIG_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_GENERATION_CONFIG_SHA256,
    ProfileInput,
    ProfileInputError,
    ProfileGuardError,
    ProfileTimeout,
    _frameworks_imported,
    _is_oom,
    _atomic_write_json,
    live_resource_guard,
    load_profile_rows,
    model_identity,
    wall_clock_limit,
)
from campaign_tokenizer_contract import (  # noqa: E402
    TokenizerContractError,
    load_pinned_reference_tokenizer,
    restore_pinned_tokenizer_contract,
)
from sepalith.campaign_protocol import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    NATIVE_EOG_IDS,
    VOCAB_SIZE,
    is_native_control_token,
)


RL_PROFILE_SCHEMA_VERSION = "sepalith.prm07.rl-profile.v1"
RL_LORA_RANK = 16
RL_LORA_ALPHA = 16
RL_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
RL_EXPECTED_ATTACHMENTS = 294
RL_EXPECTED_TRAINABLE_PARAMETERS = 25_116_672
RL_PROMPT_MAX_TOKENS = 2_048
RL_COMPLETION_MAX_TOKENS = 192
RL_CONTEXT_MAX_TOKENS = RL_PROMPT_MAX_TOKENS + RL_COMPLETION_MAX_TOKENS
DEFAULT_CANDIDATE_COUNTS = (2, 4)
DEFAULT_PROMPT_COUNT = 1
DEFAULT_WALL_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_EXISTING_VRAM_MIB = 1_024.0


class RLInputError(ValueError):
    """An explicit RL probe input failed closed validation."""


class RLGuardError(RuntimeError):
    """The disposable live RL probe cannot safely take the device lease."""


class RLProbeError(RuntimeError):
    """The model mechanism probe produced an invalid or unusable result."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_of(model: Any, torch_module: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch_module.device("cpu")


def _synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)


def _reset_peak_memory(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(torch_module: Any, device: Any) -> dict[str, int | None]:
    if getattr(device, "type", None) != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
    }


def _peak_memory(torch_module: Any, device: Any) -> dict[str, int | None]:
    if getattr(device, "type", None) != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
    }


def _validate_candidate_counts(value: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(value)
    if not counts or any(type(count) is not int or not 1 <= count <= 4 for count in counts):
        raise RLInputError("generation candidate counts must be nonempty integers in 1..4")
    if len(set(counts)) != len(counts):
        raise RLInputError("generation candidate counts must be unique")
    return counts


def parse_candidate_counts(value: str) -> tuple[int, ...]:
    """Parse a bounded ordered list such as ``2,4``."""
    if not isinstance(value, str) or not value.strip():
        raise RLInputError("--generation-candidates must be a comma-separated list")
    try:
        counts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise RLInputError("--generation-candidates contains a non-integer") from error
    return _validate_candidate_counts(counts)


def _as_prompt_ids(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not ids or len(set(ids)) != len(ids) or len(ids) > 4:
        raise RLInputError("--prompt-ids must contain 1..4 unique nonempty IDs")
    return ids


def prepare_prompt_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_count: int = DEFAULT_PROMPT_COUNT,
    prompt_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select eligible prompts in explicit row order using stored integer IDs."""
    if type(prompt_count) is not int or not 1 <= prompt_count <= 4:
        raise RLInputError("prompt_count must be an integer in 1..4")
    requested = set(prompt_ids) if prompt_ids is not None else None
    if requested is not None and (
        len(requested) != len(tuple(prompt_ids or ())) or len(requested) > 4
    ):
        raise RLInputError("prompt_ids must contain 1..4 unique IDs")
    eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in seen_ids:
            raise RLInputError("selected rows must have unique string IDs")
        seen_ids.add(row_id)
        if requested is not None and row_id not in requested:
            continue
        stored = row.get("input_ids")
        target_start = row.get("target_start")
        if (not isinstance(stored, list) or any(type(token) is not int for token in stored)
                or type(target_start) is not int or not 1 <= target_start < len(stored)):
            raise RLInputError(f"row {row_id} has invalid stored prompt geometry")
        prompt = list(stored[:target_start])
        if not prompt or prompt[0] != BOS_ID:
            raise RLInputError(f"row {row_id} does not begin its stored prompt with manual BOS {BOS_ID}")
        if len(prompt) > RL_PROMPT_MAX_TOKENS:
            continue
        if any(token < 0 or token >= VOCAB_SIZE for token in prompt):
            raise RLInputError(f"row {row_id} prompt contains an out-of-range token")
        eligible.append({
            "id": row_id,
            "family": row.get("family"),
            "package_id": row.get("package_id"),
            "prompt_tokens": prompt,
            "prompt_length": len(prompt),
            "target_start": target_start,
            "source": "stored_input_ids_prefix_before_target_start",
        })
    if requested is not None:
        missing = sorted(requested - {record["id"] for record in eligible})
        if missing:
            raise RLInputError(f"requested prompt IDs are absent or exceed the {RL_PROMPT_MAX_TOKENS}-token cap: {missing[:4]}")
    if len(eligible) < prompt_count:
        raise RLInputError(
            f"only {len(eligible)} selected prompts fit the {RL_PROMPT_MAX_TOKENS}-token cap; {prompt_count} required"
        )
    chosen = eligible if requested is not None else eligible[:prompt_count]
    if requested is not None and len(chosen) != len(requested):
        raise RLInputError("requested prompt selection did not resolve exactly")
    return chosen


def trim_generated_sequence(
    sequence: Sequence[int],
    prompt_length: int,
    *,
    completion_max_tokens: int = RL_COMPLETION_MAX_TOKENS,
) -> dict[str, Any]:
    """Strip the exact prompt and framework padding, retaining terminal IDs."""
    if type(prompt_length) is not int or prompt_length < 1:
        raise RLInputError("prompt_length must be positive")
    if type(completion_max_tokens) is not int or completion_max_tokens < 1:
        raise RLInputError("completion_max_tokens must be positive")
    values = [int(token) for token in sequence]
    if len(values) < prompt_length:
        raise RLProbeError("generation sequence is shorter than its exact prompt")
    tail = values[prompt_length:]
    if len(tail) > completion_max_tokens:
        raise RLProbeError("generation sequence exceeded the declared completion cap")
    terminal_index = next((index for index, token in enumerate(tail) if token in NATIVE_EOG_IDS), None)
    if terminal_index is None:
        generated = tail
        reason = "length" if len(generated) == completion_max_tokens else "unknown"
        padded_after_terminal = 0
    else:
        generated = tail[:terminal_index + 1]
        padded_after_terminal = len(tail) - len(generated)
        reason = "eos"
    if any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in generated):
        raise RLProbeError("generation contains an invalid token ID")
    return {
        "generated_tokens": generated,
        "terminal_reason": reason,
        "padded_after_terminal": padded_after_terminal,
        "raw_generation_width": len(tail),
    }


def account_generation_records(
    records: Sequence[Mapping[str, Any]],
    *,
    prompt_max_tokens: int = RL_PROMPT_MAX_TOKENS,
    completion_max_tokens: int = RL_COMPLETION_MAX_TOKENS,
) -> dict[str, Any]:
    """Expose canonical/noncanonical EOG, cap, and CONTROL accounting."""
    if type(prompt_max_tokens) is not int or prompt_max_tokens < 1:
        raise RLInputError("prompt_max_tokens must be positive")
    if type(completion_max_tokens) is not int or completion_max_tokens < 1:
        raise RLInputError("completion_max_tokens must be positive")
    counts: dict[str, int] = {
        "records": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "canonical_eos": 0,
        "noncanonical_eog": 0,
        "cap_hits": 0,
        "control_before_terminal": 0,
        "invalid_terminal": 0,
        "padded_after_terminal": 0,
        "valid_canonical_records": 0,
    }
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = record.get("prompt_tokens")
        generated = record.get("generated_tokens")
        reason = record.get("terminal_reason")
        if (not isinstance(prompt, list) or not isinstance(generated, list)
                or any(type(token) is not int for token in prompt + generated)
                or len(prompt) > prompt_max_tokens or len(generated) > completion_max_tokens
                or not prompt or prompt[0] != BOS_ID):
            raise RLInputError(f"generation record {index} violates prompt/BOS/cap shape")
        if any(token < 0 or token >= VOCAB_SIZE for token in prompt + generated):
            raise RLInputError(f"generation record {index} contains an out-of-range token")
        if not generated:
            raise RLProbeError(f"generation record {index} has no generated token")
        terminal = generated[-1]
        controls = sum(is_native_control_token(token) for token in generated[:-1])
        if controls:
            counts["control_before_terminal"] += 1
        if reason == "length":
            if len(generated) != completion_max_tokens:
                raise RLInputError(f"generation record {index} length stop is not at the cap")
            counts["cap_hits"] += 1
        elif reason == "eos" and terminal == EOS_ID:
            counts["canonical_eos"] += 1
        elif reason == "eos" and terminal in NATIVE_EOG_IDS:
            counts["noncanonical_eog"] += 1
        else:
            counts["invalid_terminal"] += 1
        if terminal == EOS_ID and reason == "eos" and not controls:
            counts["valid_canonical_records"] += 1
        counts["records"] += 1
        counts["prompt_tokens"] += len(prompt)
        counts["generated_tokens"] += len(generated)
        counts["padded_after_terminal"] += int(record.get("padded_after_terminal", 0))
        normalized.append({
            "prompt_tokens": list(prompt),
            "generated_tokens": list(generated),
            "terminal_reason": reason,
            "terminal_token": terminal,
            "control_before_terminal": controls,
            "padded_after_terminal": int(record.get("padded_after_terminal", 0)),
        })
    return {"status": "accounted", **counts, "records_detail": normalized}


def build_policy_tensors(
    prompts: Sequence[Sequence[int]],
    generated: Sequence[Sequence[int]],
    torch_module: Any,
    device: Any,
    *,
    pad_token_id: int = EOS_ID,
    context_max_tokens: int = RL_CONTEXT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build right-padded exact prompt+response tensors and a response mask.

    ``response_mask`` is indexed like ``input_ids[:, 1:]``.  Its first true
    position is ``len(prompt)-1``, so the logit immediately before the first
    generated token is used.  Padding is excluded by both attention and this
    response mask.
    """
    if len(prompts) != len(generated) or not prompts:
        raise RLInputError("policy batch needs equally sized nonempty prompt and generation lists")
    full_sequences: list[list[int]] = []
    for index, (prompt, response) in enumerate(zip(prompts, generated)):
        if not prompt or prompt[0] != BOS_ID:
            raise RLInputError(f"policy row {index} prompt lacks manual BOS")
        if not response:
            raise RLInputError(f"policy row {index} has no generated tokens")
        if len(prompt) > RL_PROMPT_MAX_TOKENS or len(response) > RL_COMPLETION_MAX_TOKENS:
            raise RLInputError(f"policy row {index} exceeds prompt or completion cap")
        full = [int(token) for token in prompt] + [int(token) for token in response]
        if len(full) > context_max_tokens:
            raise RLInputError(f"policy row {index} exceeds managed context size {context_max_tokens}")
        if any(token < 0 or token >= VOCAB_SIZE for token in full):
            raise RLInputError(f"policy row {index} contains an out-of-range token")
        full_sequences.append(full)
    width = max(len(values) for values in full_sequences)
    input_ids = torch_module.full(
        (len(full_sequences), width), pad_token_id,
        dtype=torch_module.long, device=device,
    )
    attention_mask = torch_module.zeros(
        (len(full_sequences), width), dtype=torch_module.long, device=device,
    )
    response_mask = torch_module.zeros(
        (len(full_sequences), max(width - 1, 1)), dtype=torch_module.bool, device=device,
    )
    for index, (full, prompt, response) in enumerate(zip(full_sequences, prompts, generated)):
        size = len(full)
        input_ids[index, :size] = torch_module.tensor(full, dtype=torch_module.long, device=device)
        attention_mask[index, :size] = 1
        start = len(prompt) - 1
        response_mask[index, start:start + len(response)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask[:, :max(width - 1, 1)],
        "full_lengths": [len(values) for values in full_sequences],
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "generated_lengths": [len(response) for response in generated],
    }


def recompute_policy_logprob(
    model: Any,
    torch_module: Any,
    prompts: Sequence[Sequence[int]],
    generated: Sequence[Sequence[int]],
    device: Any,
    *,
    context_max_tokens: int = RL_CONTEXT_MAX_TOKENS,
) -> tuple[Any, dict[str, Any]]:
    """Return a differentiable mean negative log probability for exact IDs."""
    batch = build_policy_tensors(
        prompts, generated, torch_module, device,
        context_max_tokens=context_max_tokens,
    )
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        labels=None,
        return_dict=True,
    )
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None or logits.ndim != 3 or logits.shape[:2] != batch["input_ids"].shape:
        raise RLProbeError("policy model returned logits with unexpected geometry")
    if logits.shape[-1] < VOCAB_SIZE:
        raise RLProbeError("policy model vocabulary is smaller than the pinned native vocabulary")
    mask = batch["response_mask"]
    # Select response positions before the FP32/log-softmax expansion.  A
    # complete [batch, context, vocab] conversion needlessly materializes the
    # prompt distribution and can dominate the RL probe's memory footprint.
    causal_logits = logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]
    response_logits = causal_logits[mask]
    response_targets = targets[mask].unsqueeze(-1)
    if response_logits.ndim != 2 or response_logits.shape[0] == 0:
        raise RLProbeError("response mask selected no policy logits")
    # Float32 log-softmax avoids BF16 reduction error while retaining the
    # gradient path to the disposable adapter parameters.
    log_probs = torch_module.log_softmax(response_logits.float(), dim=-1)
    values = log_probs.gather(-1, response_targets).squeeze(-1)
    if values.numel() != sum(len(response) for response in generated):
        raise RLProbeError("response mask does not cover exactly the generated IDs")
    loss = -values.mean()
    if not bool(torch_module.isfinite(loss).item()):
        raise RLProbeError("policy log-probability loss is non-finite")
    return loss, {
        "generated_token_denominator": int(values.numel()),
        "mean_negative_logprob": float(loss.detach().item()),
        "full_lengths": batch["full_lengths"],
        "prompt_lengths": batch["prompt_lengths"],
        "generated_lengths": batch["generated_lengths"],
        "full_logits_shape": [int(value) for value in logits.shape],
        "response_logits_shape": [int(value) for value in response_logits.shape],
        "logits_selected_before_fp32": True,
    }


def check_finite_nonzero_gradients(model: Any, torch_module: Any) -> dict[str, Any]:
    """Require finite, nonzero gradients on the trainable adapter parameters."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RLProbeError("model has no trainable adapter parameters")
    finite = True
    nonzero = False
    total = 0
    nonzero_parameters = 0
    for parameter in trainable:
        total += int(parameter.numel())
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch_module.isfinite(gradient).all().item()):
            finite = False
        magnitude = float(gradient.detach().abs().sum().item())
        if magnitude != 0.0:
            nonzero = True
            nonzero_parameters += 1
    if not finite or not nonzero:
        raise RLProbeError(
            f"adapter gradients failed finite/nonzero check: finite={finite}, nonzero={nonzero}"
        )
    return {
        "trainable_parameters": total,
        "parameters_with_nonzero_gradient": nonzero_parameters,
        "finite": finite,
        "nonzero": nonzero,
    }


@contextmanager
def generation_mode(model: Any, configuration_guard: Any, for_training: Any) -> Iterable[None]:
    """Use the campaign SFT guard around generation and restore training mode."""
    with configuration_guard(model, for_training):
        model.eval()
        yield


def _load_rl_components(
    model_path: Path,
    *,
    guard: Mapping[str, Any],
    parity_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    """Load MiniCPM and a disposable r16/a16 adapter after the guard."""
    if _frameworks_imported():
        raise RLGuardError("framework imported before the RL live resource guard")
    if guard.get("status") != "clear":
        raise RLGuardError("a clear live resource guard result is required")
    from campaign_sft import TARGET_MODULES, training_configuration_guard
    if tuple(TARGET_MODULES) != RL_TARGET_MODULES:
        raise RLInputError("campaign_sft target modules differ from the RL r16/a16 contract")
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=RL_CONTEXT_MAX_TOKENS,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=False,
    )
    try:
        reference_tokenizer = load_pinned_reference_tokenizer(model_path)
        tokenizer_audit = restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=parity_rows,
        )
    except TokenizerContractError as error:
        raise RLInputError(f"loaded tokenizer failed pinned MiniCPM contract: {error}") from error
    model = FastLanguageModel.get_peft_model(
        model,
        r=RL_LORA_RANK,
        lora_alpha=RL_LORA_ALPHA,
        lora_dropout=0,
        target_modules=list(RL_TARGET_MODULES),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    attached = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if len(attached) != RL_EXPECTED_ATTACHMENTS or trainable != RL_EXPECTED_TRAINABLE_PARAMETERS:
        raise RLInputError(
            f"RL LoRA attachment mismatch: attachments={len(attached)}, trainable={trainable}"
        )
    return model, tokenizer, torch, FastLanguageModel, training_configuration_guard, {
        "target_modules": list(RL_TARGET_MODULES),
        "lora_rank": RL_LORA_RANK,
        "lora_alpha": RL_LORA_ALPHA,
        "attachments": len(attached),
        "trainable_parameters": trainable,
    }, tokenizer_audit


def _generate_for_prompts(
    model: Any,
    torch_module: Any,
    prompts: Sequence[Mapping[str, Any]],
    candidate_count: int,
    *,
    deadline: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one concurrent batch per prompt using exact stored IDs."""
    if type(candidate_count) is not int or not 1 <= candidate_count <= 4:
        raise RLInputError("candidate_count must be in 1..4")
    device = _device_of(model, torch_module)
    _reset_peak_memory(torch_module, device)
    _synchronize(torch_module, device)
    started = time.perf_counter_ns()
    records: list[dict[str, Any]] = []
    for prompt_record in prompts:
        deadline.check()
        prompt = list(prompt_record["prompt_tokens"])
        prompt_tensor = torch_module.tensor(
            [prompt] * candidate_count, dtype=torch_module.long, device=device,
        )
        attention = torch_module.ones_like(prompt_tensor)
        output = model.generate(
            input_ids=prompt_tensor,
            attention_mask=attention,
            max_new_tokens=RL_COMPLETION_MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=list(NATIVE_EOG_IDS),
            pad_token_id=EOS_ID,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        sequences = getattr(output, "sequences", output)
        if sequences.ndim != 2 or sequences.shape[0] != candidate_count:
            raise RLProbeError("generation returned an unexpected candidate batch")
        for candidate_index in range(candidate_count):
            trimmed = trim_generated_sequence(
                sequences[candidate_index].detach().tolist(), len(prompt),
            )
            records.append({
                "prompt_id": prompt_record["id"],
                "candidate_index": candidate_index,
                "prompt_tokens": prompt,
                **trimmed,
            })
    _synchronize(torch_module, device)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return records, {
        "candidate_count": candidate_count,
        "prompt_count": len(prompts),
        "elapsed_seconds": elapsed,
        "synchronized": getattr(device, "type", None) == "cuda",
        "memory_after": _memory_snapshot(torch_module, device),
        "memory_peak": _peak_memory(torch_module, device),
    }


def _run_policy_phase(
    model: Any,
    torch_module: Any,
    prompts: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    deadline: Any,
) -> dict[str, Any]:
    prompt_by_id = {record["id"]: record["prompt_tokens"] for record in prompts}
    prompt_ids = [prompt_by_id[record["prompt_id"]] for record in records]
    generated = [list(record["generated_tokens"]) for record in records]
    device = _device_of(model, torch_module)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = None
    _reset_peak_memory(torch_module, device)
    _synchronize(torch_module, device)
    recompute_started = time.perf_counter_ns()
    deadline.check()
    loss, stats = recompute_policy_logprob(model, torch_module, prompt_ids, generated, device)
    _synchronize(torch_module, device)
    recompute_seconds = (time.perf_counter_ns() - recompute_started) / 1_000_000_000
    deadline.check()
    backward_started = time.perf_counter_ns()
    loss.backward()
    _synchronize(torch_module, device)
    backward_seconds = (time.perf_counter_ns() - backward_started) / 1_000_000_000
    gradients = check_finite_nonzero_gradients(model, torch_module)
    return {
        "elapsed_seconds": recompute_seconds + backward_seconds,
        "recompute_seconds": recompute_seconds,
        "backward_seconds": backward_seconds,
        "synchronized": getattr(device, "type", None) == "cuda",
        "loss_finite": bool(torch_module.isfinite(loss.detach()).item()),
        "loss": stats,
        "gradients": gradients,
        "memory_after": _memory_snapshot(torch_module, device),
        "memory_peak": _peak_memory(torch_module, device),
        "policy_batch_rows": len(records),
        "policy_prompt_count": len(prompts),
    }


def preflight_rl(
    profile_input: ProfileInput,
    *,
    prompt_count: int = DEFAULT_PROMPT_COUNT,
    prompt_ids: Sequence[str] | None = None,
    candidate_counts: Sequence[int] = DEFAULT_CANDIDATE_COUNTS,
) -> dict[str, Any]:
    """Validate the immutable inputs without importing model frameworks."""
    if _frameworks_imported():
        raise RLGuardError("frameworks are already imported; preflight must start clean")
    candidate_counts = _validate_candidate_counts(candidate_counts)
    _selected_ids, rows, input_summary = load_profile_rows(profile_input)
    identity = model_identity(profile_input.model_path, verify_files=True)
    prompts = prepare_prompt_records(rows, prompt_count=prompt_count, prompt_ids=prompt_ids)
    bounded_input = dict(input_summary)
    bounded_input.pop("selected_row_ids", None)
    return {
        "status": "preflight_pass",
        "CUDA_started": False,
        "framework_imports": _frameworks_imported(),
        "input": bounded_input,
        "model_identity": identity,
        "prompt_policy": {
            "prompt_count": len(prompts),
            "prompt_ids": [record["id"] for record in prompts],
            "eligible_prompt_rows": sum(
                len(row["input_ids"][:row["target_start"]]) <= RL_PROMPT_MAX_TOKENS for row in rows
            ),
            "prompt_max_tokens": RL_PROMPT_MAX_TOKENS,
            "manual_bos_id": BOS_ID,
            "source": "stored complete input_ids prefix through target_start; no text re-tokenization",
        },
        "generation_policy": {
            "candidate_counts": list(candidate_counts),
            "completion_max_tokens": RL_COMPLETION_MAX_TOKENS,
            "eos_ids": list(NATIVE_EOG_IDS),
            "canonical_eos_id": EOS_ID,
            "control_accounting": "CONTROL IDs before terminal are counted and invalidate canonical acceptance",
        },
        "framework_free": True,
    }


def run_live_rl_profile(
    profile_input: ProfileInput,
    *,
    prompt_count: int,
    prompt_ids: Sequence[str] | None,
    candidate_counts: Sequence[int],
    max_existing_vram_mib: float,
    deadline: Any,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    candidate_counts = _validate_candidate_counts(candidate_counts)
    _selected_ids, rows, input_summary = load_profile_rows(profile_input)
    deadline.check()
    identity = model_identity(profile_input.model_path, verify_files=True)
    prompts = prepare_prompt_records(rows, prompt_count=prompt_count, prompt_ids=prompt_ids)
    guard = live_resource_guard(max_existing_vram_mib=max_existing_vram_mib)
    deadline.check()
    model, tokenizer, torch_module, fast_model, configuration_guard, adapter, tokenizer_audit = _load_rl_components(
        profile_input.model_path, guard=guard, parity_rows=rows,
    )
    reference_tokenizer = load_pinned_reference_tokenizer(profile_input.model_path)
    device = _device_of(model, torch_module)
    # Generation is the only inference-mode section.  The campaign guard
    # restores Unsloth caches/checkpointing and invokes for_training in its
    # finally block before the differentiable policy pass.
    generation_results: list[dict[str, Any]] = []
    policy_results: list[dict[str, Any]] = []
    cleanup_contracts: list[dict[str, Any]] = []
    def save_progress(stage: str) -> None:
        if progress_path is not None:
            _atomic_write_json(progress_path, {
                "status": "partial", "stage": stage,
                "observed_at": _now(), "model_identity": identity,
                "adapter": adapter, "tokenizer_contract": tokenizer_audit,
                "prompt_records": prompts, "generation_results": generation_results,
                "policy_results": policy_results,
                "cleanup_contracts": cleanup_contracts,
                "quality_claim": False,
            })
    save_progress("model_loaded")
    # Finish and preserve the smaller arm before testing the larger batch.
    # An OOM in the larger arm must not discard an already measured arm.
    for candidate_count in candidate_counts:
        with generation_mode(model, configuration_guard, fast_model.for_training):
            records, generation_metrics = _generate_for_prompts(
                model, torch_module, prompts, candidate_count, deadline=deadline,
            )
            accounting = account_generation_records(records)
            generation_results.append({
                "candidate_count": candidate_count,
                "rollout_batch_rows": len(records),
                "records": records,
                "accounting": accounting,
                "generation": generation_metrics,
            })
            save_progress(f"generated_candidates_{candidate_count}_before_cleanup")
        cleanup_contracts.append(restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=rows,
        ))
        save_progress(f"generated_candidates_{candidate_count}")
        deadline.check()
        # generation_mode already performed the supported training transition.
        result = generation_results[-1]
        policy = _run_policy_phase(
            model, torch_module, prompts, result["records"], deadline=deadline,
        )
        policy_results.append({
            "candidate_count": result["candidate_count"],
            "policy_batch_rows": len(result["records"]),
            "policy": policy,
        })
        save_progress(f"policy_candidates_{candidate_count}")
    bounded_input = dict(input_summary)
    bounded_input.pop("selected_row_ids", None)
    return {
        "status": "complete",
        "model_identity": identity,
        "live_guard": guard,
        "tokenizer_contract": tokenizer_audit,
        "adapter": adapter,
        "input": bounded_input,
        "prompt_records": prompts,
        "generation_results": generation_results,
        "policy_results": policy_results,
        "cleanup_contracts": cleanup_contracts,
        "batch_accounting": {
            "prompt_count": len(prompts),
            "candidate_counts": list(candidate_counts),
            "policy_rows_by_candidate_count": {
                str(result["candidate_count"]): len(result["records"])
                for result in generation_results
            },
            "definition": "policy batch rows = prompt_count * candidate_count; generation is issued per prompt, then exact rollouts are recomputed together",
        },
        "interpretation": "RL policy-logprob/memory mechanism probe only; no reward, GRPO update, or quality claim",
        "synchronized_timings": True,
        "limits": {
            "prompt_max_tokens": RL_PROMPT_MAX_TOKENS,
            "completion_max_tokens": RL_COMPLETION_MAX_TOKENS,
            "managed_context_max_tokens": RL_CONTEXT_MAX_TOKENS,
        },
    }


def _base_receipt(args: argparse.Namespace, profile_input: ProfileInput, candidate_counts: Sequence[int]) -> dict[str, Any]:
    live = not args.preflight
    return {
        "task": "PRM-07/RL-01",
        "schema_version": RL_PROFILE_SCHEMA_VERSION,
        "owner": "lead" if live else "worker-runtime",
        "started_at": _now(),
        "mode": "preflight" if args.preflight else "profile",
        "execution_root": str(EXECUTION_ROOT),
        "resource_lease": (
            "lead-owned guarded disposable single-GPU RL probe; worker did not launch"
            if live else "CPU-only preparation; no model/server/CUDA launch"
        ),
        "input": {
            "candidate_file": str(profile_input.candidate_file),
            "candidate_sha256": profile_input.candidate_sha256,
            "selected_ids_file": str(profile_input.selected_ids_file),
            "selected_ids_sha256": profile_input.selected_ids_sha256,
            "model_path": str(profile_input.model_path),
            "selection_rule": "explicit hashed train IDs only; no dev/final/call discovery",
        },
        "limits": {
            "prompt_max_tokens": RL_PROMPT_MAX_TOKENS,
            "completion_max_tokens": RL_COMPLETION_MAX_TOKENS,
            "managed_context_max_tokens": RL_CONTEXT_MAX_TOKENS,
            "prompt_count": args.prompt_count,
            "candidate_counts": list(candidate_counts),
            "wall_timeout_seconds": args.wall_timeout_seconds,
            "max_existing_vram_mib": args.max_existing_vram_mib,
        },
        "policy": {
            "target_modules": list(RL_TARGET_MODULES),
            "lora_rank": RL_LORA_RANK,
            "lora_alpha": RL_LORA_ALPHA,
            "expected_attachments": RL_EXPECTED_ATTACHMENTS,
            "expected_trainable_parameters": RL_EXPECTED_TRAINABLE_PARAMETERS,
            "objective": "mean negative log probability of actual generated integer IDs",
        },
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
    parser.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--prompt-ids", default=None)
    parser.add_argument("--generation-candidates", default="2,4")
    parser.add_argument("--wall-timeout-seconds", type=float, default=DEFAULT_WALL_TIMEOUT_SECONDS)
    parser.add_argument("--max-existing-vram-mib", type=float, default=DEFAULT_MAX_EXISTING_VRAM_MIB)
    args = parser.parse_args(argv)
    try:
        candidate_counts = parse_candidate_counts(args.generation_candidates)
        prompt_ids = _as_prompt_ids(args.prompt_ids)
    except RLInputError as error:
        print(json.dumps({"status": "input_failed", "error": str(error)}))
        return 2
    profile_input = ProfileInput(
        candidate_file=args.candidate_file,
        candidate_sha256=args.candidate_sha256,
        selected_ids_file=args.selected_train_ids,
        selected_ids_sha256=args.selected_ids_sha256,
        model_path=args.model_path,
    )
    receipt = _base_receipt(args, profile_input, candidate_counts)
    receipt["prompt_ids"] = list(prompt_ids) if prompt_ids is not None else None
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        print(json.dumps({"status": "refused", "reason": f"receipt already exists: {receipt_path}"}))
        return 2
    exit_code = 0
    try:
        with wall_clock_limit(args.wall_timeout_seconds) as deadline:
            if args.preflight:
                receipt["result"] = preflight_rl(
                    profile_input,
                    prompt_count=args.prompt_count,
                    prompt_ids=prompt_ids,
                    candidate_counts=candidate_counts,
                )
            else:
                receipt["result"] = run_live_rl_profile(
                    profile_input,
                    prompt_count=args.prompt_count,
                    prompt_ids=prompt_ids,
                    candidate_counts=candidate_counts,
                    max_existing_vram_mib=args.max_existing_vram_mib,
                    deadline=deadline,
                    progress_path=receipt_path.with_suffix(".partial.json"),
                )
            receipt["status"] = "preflight_pass" if args.preflight else "complete"
    except ProfileTimeout as error:
        receipt["status"] = "timeout"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    except (RLGuardError, ProfileGuardError) as error:
        receipt["status"] = "guard_failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    except (RLInputError, ProfileInputError) as error:
        receipt["status"] = "input_failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
    except BaseException as error:
        receipt["status"] = "oom" if _is_oom(error) else "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        receipt["ended_at"] = _now()
        try:
            _atomic_write_json(receipt_path, receipt)
        except OSError as error:
            print(json.dumps({"status": "receipt_write_failed", "error": str(error)}))
            return 1
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
