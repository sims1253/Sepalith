"""Card C14 step 2: full-weight GRPO memory probe on the PC GPU (measurement only).

The probe answers one question for card C14: does full-weight GRPO on arm A
(checkpoint 536, about 2.5B parameters) fit on the 32 GB card next to the
desktop, and how long does one update take? It adopts nothing and makes no
gate decision.

One GRPO update of the campaign driver (``campaign_rl_train``) has
``LEGACY_ROLLOUT_ROWS_PER_UPDATE`` = 32 completion rows (G = 8: 4 prompts,
G = 4: 8 prompts). Each configuration runs in its own subprocess under
``cuda_host_guard`` and measures, after loading the model and materialising
the SFT optimizer state (aurora_mix, FP32 state, from the bound SFT recipe):

* ``rollout``: in-process sampled generation with the KV cache, the driver's
  fixed-ID left-padded prompts and sampling settings, every row forced to the
  1,024-token completion cap (worst case), in calls of ``groups_per_call`` x G
  rows; optionally with the optimizer state moved to host memory meanwhile;
* ``scoring``: no-grad completion log-probabilities, chunked over sequence and
  vocabulary from the final hidden states (full FP32 logits would not fit);
* ``backward``: micro-batch 1, full weights, gradient checkpointing, a
  GRPO-shaped loss over every row of the update through the same chunked
  log-probability (a custom autograd function that never materialises a full
  logit row block);
* ``step``: the optimizer step;
* ``natural_rollout``: the same rollout without forcing the cap, for a
  realistic generation time.

Each phase records peak allocated and reserved memory and its wall time. An
out-of-memory error ends that configuration and is a recorded result. The
caching allocator is capped at the physically free memory so that the Windows
driver's system-memory fallback cannot hide an overflow. A configuration
*fits* when its process peak (reserved plus CUDA context) leaves at least
1.5 GiB free with the desktop at the runner's start limit of 6,144 MiB.

The ladder follows the card: start with G = 8 and one generation call per
update; if it does not fit, try smaller rollout micro-batches, then moving the
optimizer state to host memory during generation, then G = 6, and stop at the
first configuration that fits. G = 4 runs the same ladder. Every attempt is
recorded, including the ones skipped because an earlier result already
decides them.

    python -m sepalith.training.rl.grpo_memory_probe run --docs-dir docs/training/plan-20260927/c14

Prompts are the longest TRAIN prompts of the DAT10 v4 rows (no DEV row and no
sealed data). Outputs: ``receipt.json`` and ``summary.md`` in a fresh run
directory under ``~/.local/state/sepalith/campaign-20260915/C14-grpo-memory-probe/``,
copied to ``--docs-dir``.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import cast

import torch

from sepalith.campaign_protocol import BOS_ID, EOS_ID, NATIVE_EOG_IDS
from sepalith.training.optim.full_weight_optimizer import OptimizerConfig, build_full_weight_optimizer
from sepalith.training.paths import REPO_ROOT
from sepalith.training.rl.campaign_rl_profile import trim_generated_sequence
from sepalith.training.rl.campaign_rl_train import (
    DEFAULT_GENERATION_KWARGS,
    LEGACY_ROLLOUT_ROWS_PER_UPDATE,
    FixedIDGRPOTrainerMixin,
    resolve_generation_kwargs,
)
from sepalith.training.rl.length_policy import REVIEWED_POLICY_V4, TRAIN_ROWS_V4_BRACEINDENT_SHA256, require_row

Json = dict[str, object]

SCHEMA = "sepalith.c14.grpo-memory-probe.v1"
RESULT_SCHEMA = "sepalith.c14.grpo-memory-probe-config.v1"
INPUTS_SCHEMA = "sepalith.c14.grpo-memory-probe-inputs.v1"
MODEL_PATH = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536")
MODEL_SHA256 = "befe587a0d4969e33b0daaa8fe66247371ec32f331616db74724616878ad6030"
TRAIN_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/"
                  "train-token-rows.jsonl")
SFT_RECIPE = REPO_ROOT / "docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json"
STATE_ROOT = Path.home() / ".local/state/sepalith/campaign-20260915/C14-grpo-memory-probe"
CUDA_LOCK = Path.home() / ".local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock"

MIB = 1024 * 1024
ROWS_PER_UPDATE = LEGACY_ROLLOUT_ROWS_PER_UPDATE
G6_ROWS_PER_UPDATE = 30  # five groups of six: the nearest whole-group count to 32 rows
COMPLETION_TOKENS = REVIEWED_POLICY_V4.completion_max_tokens
DESKTOP_BUDGET_MIB = 6144  # the night runner's start limit for GPU memory already in use
MARGIN_MIB = 1536  # card C14: fit with 1.5 GB to spare
ALLOCATOR_SAFETY_MIB = 256
DEFAULT_SEQ_CHUNK = 512
DEFAULT_VOCAB_CHUNK = 16384
DEFAULT_SEED = 3407
WARM_STEP_TOKENS = 256
SYNTHETIC_TOKEN_RANGE = (22, 130072)  # outside every native CONTROL range
DEFAULT_PHASES = ("rollout", "scoring", "backward", "step", "natural_rollout")
ROLLOUT_PHASES = frozenset({"rollout", "natural_rollout"})
KNOWN_PHASES = frozenset({"load", "warm_step", *DEFAULT_PHASES})
EXIT_ERROR = 3


class ProbeError(RuntimeError):
    """The probe cannot run or its inputs are wrong."""


class _PhaseAbort(Exception):
    """A phase ran out of memory or failed; the configuration stops here."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _mib(value: object) -> float | None:
    return None if not isinstance(value, (int, float)) else round(float(value) / MIB, 1)


# Configurations and the ladder ----------------------------------------------------------------

@dataclass(frozen=True)
class ProbeConfig:
    name: str
    candidate_count: int
    rows_per_update: int
    groups_per_call: int
    offload_optimizer: bool = False
    trick: str = "baseline"
    role: str = "ladder"
    logprob_impl: str = "chunked"
    phases: tuple[str, ...] = DEFAULT_PHASES
    backward_rows: int | None = None
    score_rows: int | None = None
    seq_chunk: int = DEFAULT_SEQ_CHUNK
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK
    inject_oom_phase: str | None = None  # tests only

    def __post_init__(self) -> None:
        if self.candidate_count < 2 or self.rows_per_update % self.candidate_count:
            raise ProbeError(f"{self.name}: rows_per_update must hold whole groups of G")
        if self.groups_per_call < 1 or self.groups_per_update % self.groups_per_call:
            raise ProbeError(f"{self.name}: groups_per_call must divide the groups per update")
        if self.logprob_impl not in ("chunked", "full_logits"):
            raise ProbeError(f"{self.name}: unknown logprob_impl {self.logprob_impl}")
        unknown = set(self.phases) - KNOWN_PHASES
        if unknown or len(set(self.phases)) != len(self.phases):
            raise ProbeError(f"{self.name}: invalid phases {self.phases}")
        if self.seq_chunk < 1 or self.vocab_chunk < 1:
            raise ProbeError(f"{self.name}: chunk sizes must be positive")

    @property
    def groups_per_update(self) -> int:
        return self.rows_per_update // self.candidate_count

    @property
    def rows_per_call(self) -> int:
        return self.groups_per_call * self.candidate_count

    @property
    def effective_backward_rows(self) -> int:
        return self.rows_per_update if self.backward_rows is None else min(self.backward_rows, self.rows_per_update)

    @property
    def effective_score_rows(self) -> int:
        return self.score_rows if self.score_rows is not None else min(self.candidate_count, 8)

    def to_mapping(self) -> Json:
        value = cast(Json, asdict(self))
        value["phases"] = list(self.phases)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ProbeConfig:
        data = json.loads(json.dumps(dict(value)))  # plain JSON values for the dataclass fields
        phases = data.get("phases", list(DEFAULT_PHASES))
        if not isinstance(phases, (list, tuple)):
            raise ProbeError("phases must be a list")
        data["phases"] = tuple(str(phase) for phase in phases)
        return cls(**data)


def _descending_divisors(value: int) -> list[int]:
    return [d for d in range(value, 0, -1) if value % d == 0]


def ladder(candidate_count: int, rows_per_update: int = ROWS_PER_UPDATE) -> list[ProbeConfig]:
    """The card's order: one call per update, smaller rollout micro-batches, then optimizer offload."""
    groups = rows_per_update // candidate_count
    sizes = _descending_divisors(groups)
    configs = []
    for offload in (False, True):
        for index, size in enumerate(sizes):
            trick = "optimizer_offload" if offload else ("baseline" if index == 0 else "smaller_rollout_microbatch")
            tag = "offload" if offload else "resident"
            configs.append(ProbeConfig(name=f"g{candidate_count}-r{size * candidate_count}-{tag}",
                                       candidate_count=candidate_count, rows_per_update=rows_per_update,
                                       groups_per_call=size, offload_optimizer=offload, trick=trick))
    return configs


def default_plan() -> list[tuple[str, list[ProbeConfig]]]:
    """G = 8 ladder plus the G = 6 fallback, the G = 4 ladder, then one informational run."""
    g8 = ladder(8) + [ProbeConfig(name="g6-r6-offload", candidate_count=6, rows_per_update=G6_ROWS_PER_UPDATE,
                                  groups_per_call=1, offload_optimizer=True, trick="g6")]
    informational = [ProbeConfig(
        name="info-full-logits-mb1", candidate_count=8, rows_per_update=ROWS_PER_UPDATE, groups_per_call=1,
        role="informational", trick="stock_trl_logprob_path", logprob_impl="full_logits",
        phases=("backward", "step"), backward_rows=8)]
    return [("g8", g8), ("g4", ladder(4)), ("informational", informational)]


def assess(result: Mapping[str, object], *, desktop_budget_mib: float = DESKTOP_BUDGET_MIB,
           margin_mib: float = MARGIN_MIB) -> Json:
    """Did a configuration fit? Uses the process peak (reserved + CUDA context) against the budget."""
    config = ProbeConfig.from_mapping(cast(Mapping[str, object], result["config"]))
    phases = cast(Mapping[str, Mapping[str, object]], result.get("phases") or {})
    device = cast(Mapping[str, object], result.get("device") or {})
    planned = ["load", "warm_step", *config.phases]
    failed = [name for name in planned if name in phases and phases[name].get("status") not in ("ok", None)]
    completed = result.get("status") == "completed" and all(
        phases.get(name, {}).get("status") == "ok" for name in planned)
    total = device.get("total_mib")
    context = device.get("context_mib")
    peaks = {name: phases[name].get("peak_reserved_bytes") for name in planned if name in phases}
    measured = {name: float(value) / MIB for name, value in peaks.items() if isinstance(value, (int, float))}
    if not isinstance(total, (int, float)) or not measured:
        return {"measured": False, "completed": completed, "fits": None, "failed_phases": failed,
                "binding_phase": failed[0] if failed else None, "reason": "no GPU memory measurement"}
    context_mib = float(context) if isinstance(context, (int, float)) else 0.0
    process_peak = max(measured.values()) + context_mib
    limit = float(total) - desktop_budget_mib - margin_mib
    fits = completed and process_peak <= limit
    over = [name for name, value in measured.items() if value + context_mib > limit]
    binding = failed[0] if failed else (max(over, key=lambda name: measured[name]) if over else None)
    other = device.get("other_used_mib_at_start")
    return {
        "measured": True, "completed": completed, "fits": fits, "failed_phases": failed,
        "physical_overflow": (process_peak + float(other) > float(total)) if isinstance(other, (int, float)) else None,
        "binding_phase": None if fits else binding,
        "process_peak_mib": round(process_peak, 1),
        "phase_process_peak_mib": {name: round(value + context_mib, 1) for name, value in measured.items()},
        "limit_mib": round(limit, 1),
        "headroom_at_desktop_budget_mib": round(float(total) - desktop_budget_mib - process_peak, 1),
        "headroom_at_observed_desktop_mib": (round(float(total) - float(other) - process_peak, 1)
                                             if isinstance(other, (int, float)) else None),
        "desktop_budget_mib": desktop_budget_mib, "margin_mib": margin_mib,
    }


def _skip_reason(config: ProbeConfig, attempts: Sequence[Mapping[str, object]],
                 ladder_attempts: Sequence[Mapping[str, object]]) -> str | None:
    """An earlier attempt already decides this configuration."""
    for attempt in ladder_attempts:
        verdict = cast(Mapping[str, object], attempt.get("assessment") or {})
        binding = verdict.get("binding_phase")
        if verdict.get("measured") and binding is not None and binding not in ROLLOUT_PHASES:
            return (f"{attempt['name']} did not fit in phase {binding}; the remaining tricks only change "
                    "the rollout, so they cannot make it fit")
    for attempt in attempts:
        verdict = cast(Mapping[str, object], attempt.get("assessment") or {})
        other = ProbeConfig.from_mapping(cast(Mapping[str, object], attempt["config"]))
        if (verdict.get("measured") and verdict.get("binding_phase") == "rollout"
                and other.offload_optimizer == config.offload_optimizer
                and other.rows_per_call <= config.rows_per_call and "rollout" in config.phases):
            return (f"dominated by {attempt['name']}: the rollout did not fit at {other.rows_per_call} rows per "
                    f"call with the same optimizer placement, and this configuration generates "
                    f"{config.rows_per_call} rows per call from prompts at least as long")
    return None


def run_ladder(configs: Sequence[ProbeConfig], execute: Callable[[ProbeConfig], Json],
               attempts: list[Json], *, can_start: Callable[[], str | None]) -> Json:
    """Run configurations in order until one fits; append every attempt (run or skipped) to ``attempts``."""
    ladder_attempts: list[Json] = []
    fitted: str | None = None
    stopped: str | None = None
    for config in configs:
        if fitted is not None:
            break
        reason = _skip_reason(config, attempts, ladder_attempts) if config.role == "ladder" else None
        blocked = stopped or can_start()
        record: Json = {"name": config.name, "config": config.to_mapping()}
        if blocked is not None:
            stopped = blocked
            record.update({"status": "not_run", "reason": blocked})
        elif reason is not None:
            record.update({"status": "skipped", "reason": reason})
        else:
            record = execute(config)
            run_ladder_assessment = cast(Mapping[str, object], record.get("assessment") or {})
            if run_ladder_assessment.get("fits") is True and config.role == "ladder":
                fitted = config.name
        attempts.append(record)
        ladder_attempts.append(record)
    return {"first_fit": fitted, "attempts": [record["name"] for record in ladder_attempts], "stopped": stopped}


# Prompts ---------------------------------------------------------------------------------------

class _PromptCaps:
    _campaign_prompt_max_tokens = REVIEWED_POLICY_V4.prompt_max_tokens


def select_longest_prompts(rows_path: Path, rows_sha256: str | None, count: int) -> tuple[list[Json], Json]:
    """The ``count`` longest distinct TRAIN prompts, each admitted by the length policy and the driver."""
    digest = hashlib.sha256()
    best: dict[tuple[int, ...], Json] = {}
    rows = 0
    with rows_path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            rows += 1
            if row.get("split") != "train":
                raise ProbeError(f"row {row.get('id')} is not TRAIN")
            require_row(row, REVIEWED_POLICY_V4)
            prompt = tuple(row["input_ids"][:row["target_start"]])
            if prompt in best and str(best[prompt]["id"]) <= row["id"]:
                continue
            best[prompt] = {"id": row["id"], "family": row.get("family"), "prompt_length": len(prompt)}
    observed = digest.hexdigest()
    if rows_sha256 is not None and observed != rows_sha256:
        raise ProbeError(f"TRAIN rows hash {observed} differs from the expected {rows_sha256}")
    ranked = sorted(best.items(), key=lambda item: (-len(item[0]), str(item[1]["id"])))[:count]
    if len(ranked) < count:
        raise ProbeError(f"only {len(ranked)} distinct TRAIN prompts; {count} required")
    prompts = []
    for index, (tokens, meta) in enumerate(ranked):
        checked = FixedIDGRPOTrainerMixin._prompt_ids(  # the driver's fixed-ID prompt contract
            cast(FixedIDGRPOTrainerMixin, _PromptCaps()), {"ids": list(tokens)}, index)
        prompts.append({**meta, "prompt_tokens": checked})
    source = {"path": str(rows_path), "sha256": observed, "rows": rows, "distinct_prompts": len(best),
              "length_policy": REVIEWED_POLICY_V4.policy_id, "length_policy_sha256": REVIEWED_POLICY_V4.sha256,
              "selection": f"{count} longest distinct prompts, ties by row id", "split": "train"}
    return prompts, source


# Chunked log-probabilities ------------------------------------------------------------------------

class ChunkedTokenLogProb(torch.autograd.Function):
    """Per-token log p(target) = h.W[t]/T - logsumexp(h.W/T), never holding more than one logit tile.

    Forward and backward walk (sequence chunk x vocabulary chunk) tiles; the
    backward recomputes each tile's logits from the saved log-sum-exp. Targets
    below zero are ignored (log-probability 0, no gradient).
    """

    @staticmethod
    def forward(ctx: object, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                temperature: float, seq_chunk: int, vocab_chunk: int) -> torch.Tensor:
        logsumexp, target_logit = _chunked_forward(hidden, weight, targets, temperature, seq_chunk, vocab_chunk)
        valid = targets >= 0
        context = cast(_FunctionContext, ctx)
        context.save_for_backward(hidden, weight, targets, logsumexp)
        context.settings = (temperature, seq_chunk, vocab_chunk)
        return torch.where(valid, target_logit - logsumexp, torch.zeros_like(logsumexp))

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        context = cast(_FunctionContext, ctx)
        hidden, weight, targets, logsumexp = context.saved_tensors
        temperature, seq_chunk, vocab_chunk = context.settings
        need_hidden, need_weight = context.needs_input_grad[0], context.needs_input_grad[1]
        accumulate = logsumexp.dtype
        upstream = torch.where(targets >= 0, grad_output.to(accumulate), torch.zeros_like(logsumexp))
        safe = targets.clamp_min(0)
        rows, vocab = hidden.shape[0], weight.shape[0]
        grad_hidden = torch.zeros(hidden.shape, dtype=accumulate, device=hidden.device) if need_hidden else None
        grad_weight = torch.empty_like(weight) if need_weight else None
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            block = weight[v0:v1]
            block_wide = block.to(accumulate)
            tile_grad = torch.zeros((v1 - v0, hidden.shape[1]), dtype=accumulate, device=hidden.device) \
                if need_weight else None
            for s0 in range(0, rows, seq_chunk):
                s1 = min(s0 + seq_chunk, rows)
                h = hidden[s0:s1]
                logits = (h @ block.T).to(accumulate) / temperature
                dlogits = torch.exp(logits - logsumexp[s0:s1, None]).mul_(-upstream[s0:s1, None])
                inside = (safe[s0:s1] >= v0) & (safe[s0:s1] < v1)
                index = (safe[s0:s1] - v0).clamp(0, v1 - v0 - 1)
                dlogits.scatter_add_(1, index[:, None], torch.where(inside, upstream[s0:s1], 0.0)[:, None])
                dlogits.div_(temperature)
                if grad_hidden is not None:
                    grad_hidden[s0:s1] += dlogits @ block_wide
                if tile_grad is not None:
                    tile_grad += dlogits.T @ h.to(accumulate)
            if grad_weight is not None and tile_grad is not None:
                grad_weight[v0:v1] = tile_grad.to(weight.dtype)
        return (grad_hidden.to(hidden.dtype) if grad_hidden is not None else None, grad_weight,
                None, None, None, None)


class _FunctionContext:
    saved_tensors: tuple[torch.Tensor, ...]
    settings: tuple[float, int, int]
    needs_input_grad: tuple[bool, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _chunked_forward(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, temperature: float,
                     seq_chunk: int, vocab_chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab = hidden.shape[0], weight.shape[0]
    accumulate = torch.promote_types(hidden.dtype, torch.float32)  # FP32 for bf16 models, FP64 in tests
    logsumexp = torch.empty(rows, dtype=accumulate, device=hidden.device)
    target_logit = torch.zeros(rows, dtype=accumulate, device=hidden.device)
    safe = targets.clamp_min(0)
    for s0 in range(0, rows, seq_chunk):
        s1 = min(s0 + seq_chunk, rows)
        h = hidden[s0:s1]
        running = torch.full((s1 - s0,), -math.inf, dtype=accumulate, device=hidden.device)
        picked = torch.zeros(s1 - s0, dtype=accumulate, device=hidden.device)
        for v0 in range(0, vocab, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab)
            logits = (h @ weight[v0:v1].T).to(accumulate) / temperature
            running = torch.logaddexp(running, torch.logsumexp(logits, dim=-1))
            inside = (safe[s0:s1] >= v0) & (safe[s0:s1] < v1)
            index = (safe[s0:s1] - v0).clamp(0, v1 - v0 - 1)
            picked = torch.where(inside, logits.gather(1, index[:, None]).squeeze(1), picked)
        logsumexp[s0:s1] = running
        target_logit[s0:s1] = picked
    return logsumexp, target_logit


def chunked_token_logprobs(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, *,
                           temperature: float = 1.0, seq_chunk: int = DEFAULT_SEQ_CHUNK,
                           vocab_chunk: int = DEFAULT_VOCAB_CHUNK) -> torch.Tensor:
    """Log-probabilities of ``targets`` (shape [N]) from hidden states [N, H] and the LM head [V, H]."""
    if hidden.ndim != 2 or weight.ndim != 2 or targets.ndim != 1 or hidden.shape[0] != targets.shape[0]:
        raise ProbeError("chunked_token_logprobs takes hidden [N, H], weight [V, H] and targets [N]")
    if hidden.shape[1] != weight.shape[1]:
        raise ProbeError("hidden size differs from the LM head")
    return cast(torch.Tensor, ChunkedTokenLogProb.apply(hidden, weight, targets, float(temperature),
                                                        int(seq_chunk), int(vocab_chunk)))


def completion_batch(rows: Sequence[tuple[list[int], list[int]]], pad_id: int, device: torch.device
                     ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Right-padded prompt+completion ids, attention mask and each row's (prompt, completion) lengths."""
    width = max(len(prompt) + len(completion) for prompt, completion in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    spans = []
    for index, (prompt, completion) in enumerate(rows):
        sequence = list(prompt) + list(completion)
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
        spans.append((len(prompt), len(completion)))
    return ids, mask, spans


def gather_completion_states(hidden: torch.Tensor, ids: torch.Tensor, spans: Sequence[tuple[int, int]]
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened states at positions P-1 .. P+C-2 (each predicts a completion token) and their targets."""
    states, targets = [], []
    for row, (prompt, completion) in enumerate(spans):
        states.append(hidden[row, prompt - 1:prompt - 1 + completion])
        targets.append(ids[row, prompt:prompt + completion])
    return torch.cat(states, dim=0), torch.cat(targets, dim=0)


def model_completion_logprobs(model: torch.nn.Module, rows: Sequence[tuple[list[int], list[int]]], *,
                              temperature: float, seq_chunk: int, vocab_chunk: int, impl: str = "chunked",
                              pad_id: int = EOS_ID) -> list[torch.Tensor]:
    """Per-row completion token log-probabilities (gradients flow if enabled)."""
    device = next(model.parameters()).device
    ids, mask, spans = completion_batch(rows, pad_id, device)
    lengths = [completion for _, completion in spans]
    if impl == "chunked":
        backbone = cast(Callable[..., object], model.get_submodule("model"))
        output = backbone(input_ids=ids, attention_mask=mask, use_cache=False)
        hidden = cast(torch.Tensor, getattr(output, "last_hidden_state"))  # noqa: B009
        states, targets = gather_completion_states(hidden, ids, spans)
        head = cast(torch.nn.Linear, model.get_submodule("lm_head"))
        values = chunked_token_logprobs(states, head.weight, targets, temperature=temperature,
                                        seq_chunk=seq_chunk, vocab_chunk=vocab_chunk)
        return list(values.split(lengths))
    if impl != "full_logits":
        raise ProbeError(f"unknown logprob implementation {impl}")
    # The stock TRL path: logits for the completion window, FP32 / temperature, log-softmax, gather.
    out: list[torch.Tensor] = []
    caller = cast(Callable[..., object], model)
    for row, (prompt, completion) in enumerate(spans):
        length = prompt + completion
        output = caller(input_ids=ids[row:row + 1, :length], attention_mask=mask[row:row + 1, :length],
                        use_cache=False, logits_to_keep=completion + 1)
        logits = cast(torch.Tensor, getattr(output, "logits"))[0, :-1].float() / temperature  # noqa: B009
        out.append(torch.log_softmax(logits, dim=-1).gather(1, ids[row, prompt:length, None]).squeeze(1))
    return out


def group_advantages(rewards: Sequence[float], candidate_count: int) -> list[float]:
    """Group-normalised advantages (``scale_rewards="group"``); a flat group gets zeros."""
    out = []
    for start in range(0, len(rewards), candidate_count):
        group = list(rewards[start:start + candidate_count])
        mean = sum(group) / len(group)
        std = math.sqrt(sum((value - mean) ** 2 for value in group) / max(1, len(group) - 1))
        out.extend((value - mean) / (std + 1e-4) for value in group)
    return out


# The child: one configuration on one device ------------------------------------------------------

def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _nvidia_smi_used_mib() -> float | None:
    executable = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        result = subprocess.run([executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return float(result.stdout.strip().splitlines()[0].split(",")[0])
    except ValueError:
        return None


class DeviceSampler:
    """Polls device-wide used memory (desktop included) in a thread; a cross-check of the allocator peak."""

    def __init__(self, device: torch.device, interval: float = 0.25) -> None:
        self.device = device
        self.interval = interval
        self.peak_used_mib: float | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while not self._stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info(self.device)
            except RuntimeError:
                return
            used = (total - free) / MIB
            with self._lock:
                if self.peak_used_mib is None or used > self.peak_used_mib:
                    self.peak_used_mib = used
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self.peak_used_mib = None

    def read(self) -> float | None:
        with self._lock:
            return None if self.peak_used_mib is None else round(self.peak_used_mib, 1)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


@dataclass
class ChildState:
    config: ProbeConfig
    inputs: Json
    result_path: Path
    device: torch.device
    result: Json = field(default_factory=dict)
    sampler: DeviceSampler | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    rows: list[tuple[list[int], list[int]]] = field(default_factory=list)

    @property
    def cuda(self) -> bool:
        return self.device.type == "cuda"

    def save(self) -> None:
        write_json_atomic(self.result_path, self.result)

    def phases(self) -> dict[str, Json]:
        return cast(dict[str, Json], self.result.setdefault("phases", {}))

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def memory(self) -> Json:
        if not self.cuda:
            return {"allocated_bytes": None, "reserved_bytes": None}
        return {"allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(self.device))}

    def run_phase(self, name: str, body: Callable[[Json], None]) -> None:
        record: Json = {"status": "running", "started": utc_now()}
        self.phases()[name] = record
        cast(list[str], self.result.setdefault("phase_order", [])).append(name)
        self.save()
        if self.cuda:
            self.sync()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.sampler is not None:
            self.sampler.reset()
        record["before"] = self.memory()
        started = time.perf_counter()
        try:
            if self.config.inject_oom_phase == name:
                raise torch.OutOfMemoryError(f"CUDA out of memory (injected in phase {name} for a test)")
            body(record)
            self.sync()
            record["status"] = "ok"
        except Exception as error:  # recorded in the result (re-raised as _PhaseAbort); the configuration stops
            record["status"] = "oom" if _is_oom(error) else "error"
            record["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
            if record["status"] == "error":
                record["traceback"] = traceback.format_exc()[-6000:]
            raise _PhaseAbort(name) from error
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            if self.cuda:
                record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
                record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(self.device))
                record["peak_reserved_mib"] = _mib(record["peak_reserved_bytes"])
                record["after"] = self.memory()
            if self.sampler is not None:
                record["device_used_peak_mib"] = self.sampler.read()
            record["finished"] = utc_now()
            self.save()


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(value.numel() * value.element_size() for state in optimizer.state.values()
               for value in state.values() if torch.is_tensor(value))


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    """Move every optimizer state tensor to ``device``; return the state bytes now held there."""
    held = 0
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                if value.device != device:
                    state[key] = value.to(device)
                held += value.numel() * value.element_size()
    return held


def _generation_kwargs(inputs: Mapping[str, object], max_new: int, forced: bool) -> Json:
    sampling = resolve_generation_kwargs(cast(Mapping[str, object], inputs.get("generation_kwargs")))
    values: Json = {**sampling, "max_new_tokens": max_new, "eos_token_id": list(cast(list[int], inputs["eog_ids"])),
                    "pad_token_id": EOS_ID, "bos_token_id": BOS_ID, "num_return_sequences": 1, "use_cache": True,
                    "return_dict_in_generate": False}
    if forced:
        values["min_new_tokens"] = max_new
    return values


def _rollout(state: ChildState, record: Json, *, forced: bool) -> list[tuple[list[int], list[int]]]:
    config, model = state.config, cast(torch.nn.Module, state.model)
    max_new = int(cast(int, state.inputs["completion_tokens"]))
    prompts = cast(list[Json], state.inputs["prompts"])[:config.groups_per_update]
    groups = [cast(list[int], prompt["prompt_tokens"]) for prompt in prompts]
    kwargs = _generation_kwargs(state.inputs, max_new, forced)
    if config.offload_optimizer and state.optimizer is not None:
        started = time.perf_counter()
        record["offloaded_bytes"] = _move_optimizer_state(state.optimizer, torch.device("cpu"))
        state.sync()
        if state.cuda:
            torch.cuda.empty_cache()
        record["offload_seconds"] = round(time.perf_counter() - started, 3)
        record["after_offload"] = state.memory()
    model.eval()
    rows: list[tuple[list[int], list[int]]] = []
    calls: list[Json] = []
    generate = cast(Callable[..., torch.Tensor], getattr(model, "generate"))  # noqa: B009
    try:
        for call, start in enumerate(range(0, len(groups), config.groups_per_call)):
            chunk = groups[start:start + config.groups_per_call]
            prompt_ids = [list(prompt) for prompt in chunk for _ in range(config.candidate_count)]
            input_ids, attention, width = FixedIDGRPOTrainerMixin._padded_prompt_tensors(prompt_ids, torch, state.device)
            torch.manual_seed(DEFAULT_SEED + call + (1000 if not forced else 0))
            began = time.perf_counter()
            with torch.inference_mode():
                sequences = generate(input_ids=input_ids, attention_mask=attention, **kwargs)
            state.sync()
            seconds = time.perf_counter() - began
            if not torch.equal(sequences[:, :width], input_ids):
                raise ProbeError("generation changed the stored prompt ids")
            lengths, reasons = [], {}
            for row, ids in enumerate(prompt_ids):
                tail = sequences[row, width:].tolist()
                trimmed = trim_generated_sequence(ids + tail, len(ids), completion_max_tokens=max_new)
                completion = list(cast(list[int], trimmed["generated_tokens"]))
                rows.append((ids, completion))
                lengths.append(len(completion))
                reason = str(trimmed["terminal_reason"])
                reasons[reason] = reasons.get(reason, 0) + 1
            steps = int(sequences.shape[1]) - width
            calls.append({"rows": len(prompt_ids), "padded_prompt_width": width, "decode_steps": steps,
                          "seconds": round(seconds, 3), "generated_tokens": sum(lengths),
                          "max_completion": max(lengths), "terminal_reasons": reasons,
                          "row_steps_per_second": round(len(prompt_ids) * steps / seconds, 1) if seconds else None})
            record["calls"] = calls
            del sequences, input_ids, attention
    finally:
        model.train()
    generation_seconds = sum(float(cast(float, call["seconds"])) for call in calls)
    tokens = sum(int(cast(int, call["generated_tokens"])) for call in calls)
    record.update({"forced_to_cap": forced, "rows": len(rows), "generated_tokens": tokens,
                   "generation_seconds": round(generation_seconds, 3),
                   "tokens_per_second": round(tokens / generation_seconds, 1) if generation_seconds else None,
                   "mean_completion_tokens": round(tokens / len(rows), 1) if rows else None})
    if config.offload_optimizer and state.optimizer is not None:
        started = time.perf_counter()
        _move_optimizer_state(state.optimizer, state.device)
        state.sync()
        record["reload_seconds"] = round(time.perf_counter() - started, 3)
    return rows


def _synthetic_rows(state: ChildState) -> list[tuple[list[int], list[int]]]:
    config = state.config
    vocab = int(getattr(cast(torch.nn.Module, state.model).config, "vocab_size"))  # noqa: B009
    low, high = SYNTHETIC_TOKEN_RANGE[0], min(SYNTHETIC_TOKEN_RANGE[1], vocab)
    generator = torch.Generator().manual_seed(DEFAULT_SEED)
    prompts = cast(list[Json], state.inputs["prompts"])[:config.groups_per_update]
    length = int(cast(int, state.inputs["completion_tokens"]))
    rows = []
    for prompt in prompts:
        for _ in range(config.candidate_count):
            completion = torch.randint(low, high, (length,), generator=generator).tolist()
            rows.append((list(cast(list[int], prompt["prompt_tokens"])), completion))
    return rows


def child_main(config_path: Path, inputs_path: Path, result_path: Path) -> int:
    config = ProbeConfig.from_mapping(json.loads(config_path.read_text()))
    inputs = cast(Json, json.loads(inputs_path.read_text()))
    if inputs.get("schema") != INPUTS_SCHEMA:
        raise ProbeError("inputs schema mismatch")
    device = torch.device(str(inputs["device"]))
    state = ChildState(config=config, inputs=inputs, result_path=result_path, device=device)
    state.result = {"schema": RESULT_SCHEMA, "config": config.to_mapping(), "status": "running", "started": utc_now(),
                    "pid": os.getpid(), "phases": {}, "phase_order": [],
                    "environment": {"torch": torch.__version__,
                                    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    torch.manual_seed(DEFAULT_SEED)
    if state.cuda:
        other = _nvidia_smi_used_mib()
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        torch.ones((8, 8), device=device) @ torch.ones((8, 8), device=device)  # cuBLAS handle and workspace
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        free_mib, total_mib = free / MIB, total / MIB
        used_mib = total_mib - free_mib
        context = used_mib - other if other is not None else None
        cap_mib = free_mib - ALLOCATOR_SAFETY_MIB
        torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, cap_mib / total_mib)), device)
        state.result["device"] = {
            "name": torch.cuda.get_device_name(device), "total_mib": round(total_mib, 1),
            "other_used_mib_at_start": other, "used_mib_after_context": round(used_mib, 1),
            "context_mib": round(context, 1) if context is not None else 512.0,
            "context_mib_source": "nvidia-smi before CUDA init" if context is not None else "assumed 512 MiB",
            "allocator_cap_mib": round(cap_mib, 1),
            "allocator_cap_note": "set_per_process_memory_fraction at the free memory after CUDA init minus "
                                  f"{ALLOCATOR_SAFETY_MIB} MiB, so an overflow raises OOM instead of spilling "
                                  "into Windows system memory",
        }
        state.sampler = DeviceSampler(device)
        state.sampler.start()
    state.save()
    status, code = "completed", 0
    try:
        _run_child_phases(state)
    except _PhaseAbort as abort:
        phase = state.phases()[str(abort)]
        status = "oom" if phase.get("status") == "oom" else "error"
        code = 0 if status == "oom" else EXIT_ERROR
        state.result["failed_phase"] = str(abort)
    finally:
        if state.sampler is not None:
            state.sampler.stop()
        state.result["status"] = status
        state.result["finished"] = utc_now()
        if state.cuda:
            state.result["max_reserved_bytes_over_run"] = max(
                (int(cast(int, phase.get("peak_reserved_bytes") or 0)) for phase in state.phases().values()), default=0)
        state.save()
    return code


def _run_child_phases(state: ChildState) -> None:
    from transformers import AutoModelForCausalLM

    config, inputs, device = state.config, state.inputs, state.device
    temperature = float(cast(float, resolve_generation_kwargs(
        cast(Mapping[str, object], inputs.get("generation_kwargs")))["temperature"]))
    dtype = getattr(torch, str(inputs["dtype"]))

    def load(record: Json) -> None:
        kwargs: Json = {"dtype": dtype, "attn_implementation": str(inputs.get("attn_implementation", "sdpa")),
                        "local_files_only": True, "trust_remote_code": False}
        if state.cuda:
            kwargs["device_map"] = {"": str(device)}
        model = AutoModelForCausalLM.from_pretrained(str(inputs["model_path"]), **kwargs)
        model.to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        named = list(model.named_parameters())
        if not all(parameter.requires_grad for _, parameter in named):
            raise ProbeError("a parameter is frozen; the probe needs every weight trainable")
        record.update({"parameter_tensors": len(named), "parameters": sum(p.numel() for _, p in named),
                       "dtype": str(dtype), "attn_implementation": str(getattr(model.config, "_attn_implementation", "")),
                       "gradient_checkpointing": True, "weights_bytes": sum(p.numel() * p.element_size() for _, p in named)})
        state.model = model

    def warm_step(record: Json) -> None:
        model = cast(torch.nn.Module, state.model)
        optimizer_config = json.loads(json.dumps(inputs["optimizer"]))
        optimizer_config["adam_betas"] = tuple(optimizer_config["adam_betas"])
        optimizer, manifest = build_full_weight_optimizer(model, OptimizerConfig(**optimizer_config))
        state.optimizer = optimizer
        prompt = cast(list[int], cast(list[Json], inputs["prompts"])[0]["prompt_tokens"])[:WARM_STEP_TOKENS]
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        output = cast(Callable[..., object], model)(input_ids=ids, labels=ids, use_cache=False)
        cast(torch.Tensor, getattr(output, "loss")).backward()  # noqa: B009
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record.update({"optimizer_arm": manifest["arm"], "optimizer_counts": manifest["counts"],
                       "optimizer_state_bytes": _optimizer_state_bytes(optimizer), "tokens": len(prompt),
                       "note": "one real step materialises the FP32 optimizer state; grads are freed (set_to_none)"})

    def rollout(record: Json) -> None:
        state.rows = _rollout(state, record, forced=True)

    def scoring(record: Json) -> None:
        model = cast(torch.nn.Module, state.model)
        rows = state.rows or _synthetic_rows(state)
        model.eval()
        try:
            values = []
            with torch.no_grad():
                for start in range(0, len(rows), config.effective_score_rows):
                    batch = rows[start:start + config.effective_score_rows]
                    values.extend(model_completion_logprobs(model, batch, temperature=temperature,
                                                            seq_chunk=config.seq_chunk, vocab_chunk=config.vocab_chunk))
        finally:
            model.train()
        tokens = sum(int(value.numel()) for value in values)
        total = float(sum(float(value.sum()) for value in values))
        if not math.isfinite(total):
            raise ProbeError("non-finite scoring log-probabilities")
        record.update({"rows": len(rows), "batch_rows": config.effective_score_rows, "tokens": tokens,
                       "mean_token_logprob": round(total / tokens, 4) if tokens else None,
                       "seq_chunk": config.seq_chunk, "vocab_chunk": config.vocab_chunk,
                       "full_fp32_logits_bytes_unchunked": tokens * int(getattr(model.config, "vocab_size")) * 4})  # noqa: B009

    def backward(record: Json) -> None:
        model = cast(torch.nn.Module, state.model)
        rows = state.rows or _synthetic_rows(state)
        state.rows = rows
        selected = rows[:config.effective_backward_rows]
        generator = torch.Generator().manual_seed(DEFAULT_SEED)
        rewards = torch.rand(len(rows), generator=generator).tolist()
        advantages = group_advantages(rewards, config.candidate_count)
        normaliser = sum(len(completion) for _, completion in rows)  # BNPO: tokens of the whole update
        model.train()
        times, losses = [], []
        for index, row in enumerate(selected):
            began = time.perf_counter()
            (values,) = model_completion_logprobs(model, [row], temperature=temperature, seq_chunk=config.seq_chunk,
                                                  vocab_chunk=config.vocab_chunk, impl=config.logprob_impl)
            loss = -(advantages[index] * values).sum() / normaliser
            loss.backward()
            state.sync()
            times.append(time.perf_counter() - began)
            losses.append(float(loss.detach()))
            record["micro_batches_done"] = index + 1
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        # Per-tensor norms in the gradient dtype (reductions accumulate in FP32); no FP32 copy of a 1 GiB matrix.
        norm = float(torch.stack([g.norm().float() for g in grads]).norm())
        if not math.isfinite(norm) or not all(math.isfinite(value) for value in losses):
            raise ProbeError("non-finite loss or gradient")
        record.update({"micro_batch": 1, "micro_batches": len(selected), "logprob_impl": config.logprob_impl,
                       "sequence_tokens_max": max(len(p) + len(c) for p, c in selected),
                       "seconds_per_micro_batch": [round(value, 3) for value in times],
                       "seconds_per_micro_batch_median": round(sorted(times)[len(times) // 2], 3),
                       "seconds_total": round(sum(times), 3), "grad_norm": norm,
                       "grad_tensors": len(grads), "loss": "-(advantage * token logprob).sum() / update tokens (BNPO, beta 0)"})

    def step(record: Json) -> None:
        optimizer = cast(torch.optim.Optimizer, state.optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record["optimizer_state_bytes"] = _optimizer_state_bytes(optimizer)

    def natural_rollout(record: Json) -> None:
        _rollout(state, record, forced=False)

    bodies: dict[str, Callable[[Json], None]] = {"rollout": rollout, "scoring": scoring, "backward": backward,
                                                  "step": step, "natural_rollout": natural_rollout}
    state.run_phase("load", load)
    state.run_phase("warm_step", warm_step)
    state.result["resident_after_warm_step"] = state.memory()
    for name in config.phases:
        state.run_phase(name, bodies[name])


# The parent: ladder, guard, receipt ---------------------------------------------------------------

@dataclass
class ProbeSettings:
    run_dir: Path
    model_path: Path = MODEL_PATH
    model_sha256: str | None = MODEL_SHA256
    rows_path: Path = TRAIN_ROWS
    rows_sha256: str | None = TRAIN_ROWS_V4_BRACEINDENT_SHA256
    sft_recipe_path: Path = SFT_RECIPE
    device: str = "cuda"
    dtype: str = "bfloat16"
    completion_tokens: int = COMPLETION_TOKENS
    eog_ids: tuple[int, ...] = tuple(NATIVE_EOG_IDS)
    guard: bool = True
    python: str = sys.executable
    budget_seconds: float = 10200.0
    config_seconds: int = 1800
    min_config_seconds: float = 420.0
    desktop_budget_mib: float = DESKTOP_BUDGET_MIB
    margin_mib: float = MARGIN_MIB
    docs_dir: Path | None = None
    plan: list[tuple[str, list[ProbeConfig]]] = field(default_factory=default_plan)
    poll_seconds: float = 5.0
    deadline_epoch: float | None = None
    stop_file: Path | None = None
    job_result: Path | None = None


class Probe:
    def __init__(self, settings: ProbeSettings) -> None:
        self.settings = settings
        self.started = time.time()
        self.deadline = self.started + settings.budget_seconds
        if settings.deadline_epoch is not None:
            self.deadline = min(self.deadline, settings.deadline_epoch)
        self.stop_reason: str | None = None
        self.current: subprocess.Popen[bytes] | None = None
        self.index = 0

    def request_stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def check_stop(self) -> str | None:
        stop_file = self.settings.stop_file
        if stop_file is not None and stop_file.exists():
            self.request_stop("stop request file")
        if time.time() >= self.deadline:
            self.request_stop("deadline")
        return self.stop_reason

    def can_start(self) -> str | None:
        if self.check_stop() is not None:
            return f"not started: {self.stop_reason}"
        if self.deadline - time.time() < self.settings.min_config_seconds:
            return f"not started: less than {self.settings.min_config_seconds:.0f} s left before the probe deadline"
        return None

    def inputs(self) -> Json:
        settings = self.settings
        largest = max(config.groups_per_update for _, configs in settings.plan for config in configs)
        prompts, source = select_longest_prompts(settings.rows_path, settings.rows_sha256, largest)
        recipe = cast(Json, json.loads(settings.sft_recipe_path.read_text()))
        optimizer = cast(Json, cast(Json, recipe["runtime"])["optimizer"])
        model_sha = sha256_file(settings.model_path / "model.safetensors")
        if settings.model_sha256 is not None and model_sha != settings.model_sha256:
            raise ProbeError(f"model.safetensors hash {model_sha} differs from {settings.model_sha256}")
        return {
            "schema": INPUTS_SCHEMA, "model_path": str(settings.model_path), "model_sha256": model_sha,
            "device": settings.device, "dtype": settings.dtype, "attn_implementation": "sdpa",
            "completion_tokens": settings.completion_tokens, "eog_ids": list(settings.eog_ids),
            "generation_kwargs": dict(DEFAULT_GENERATION_KWARGS), "optimizer": optimizer,
            "sft_recipe": {"path": str(settings.sft_recipe_path), "sha256": sha256_file(settings.sft_recipe_path)},
            "prompts": prompts, "prompt_source": source,
        }

    def execute(self, config: ProbeConfig) -> Json:
        settings = self.settings
        self.index += 1
        directory = settings.run_dir / "configs" / f"{self.index:02d}-{config.name}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json_atomic(directory / "config.json", config.to_mapping())
        result_path = directory / "result.json"
        argv = [settings.python, "-B", "-m", "sepalith.training.rl.grpo_memory_probe", "child",
                "--config", str(directory / "config.json"), "--inputs", str(settings.run_dir / "inputs.json"),
                "--result", str(result_path)]
        budget = int(max(60, min(settings.config_seconds, self.deadline - time.time())))
        record: Json = {"name": config.name, "config": config.to_mapping(), "directory": str(directory),
                        "started": utc_now()}
        for attempt in range(2):
            guard_dir = directory / ("guard" if attempt == 0 else f"guard-retry{attempt}")
            if settings.guard:
                CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
                CUDA_LOCK.touch(exist_ok=True)
                write_json_atomic(directory / "command.json", argv)
                command = [settings.python, "-B", "-m", "sepalith.training.guards.cuda_host_guard",
                           "--command-json", str(directory / "command.json"), "--output", str(guard_dir),
                           "--seconds", str(budget)]
            else:
                command = argv
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
            began = time.monotonic()
            with (directory / "driver-child.log").open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                           start_new_session=not settings.guard)
                self.current = process
                code = self._wait(process, budget + 120)
                self.current = None
            record["exit_code"] = code
            record["seconds"] = round(time.monotonic() - began, 1)
            terminal = guard_dir / "terminal.json"
            if settings.guard and terminal.exists():
                guard_terminal = cast(Json, json.loads(terminal.read_text()))
                record["guard"] = guard_terminal
                if guard_terminal.get("status") == "not_launched" and attempt == 0 and self.check_stop() is None:
                    time.sleep(120)
                    continue
            break
        record["finished"] = utc_now()
        result = cast(Json, json.loads(result_path.read_text())) if result_path.exists() else None
        if result is None:
            record["status"] = "not_launched" if cast(Json, record.get("guard") or {}).get("status") == "not_launched" \
                else "crashed"
            record["assessment"] = {"measured": False, "fits": None, "binding_phase": None,
                                    "reason": "no result file"}
            return record
        if result.get("status") == "running":
            result["status"] = "interrupted" if self.stop_reason else "crashed"
        record["status"] = result["status"]
        record["result"] = str(result_path)
        record["device"] = result.get("device")
        record["phases"] = summarise_phases(result)
        record["failed_phase"] = result.get("failed_phase")
        record["assessment"] = assess(result, desktop_budget_mib=settings.desktop_budget_mib,
                                      margin_mib=settings.margin_mib)
        record["timing"] = update_timing(result)
        return record

    def _wait(self, process: subprocess.Popen[bytes], limit: float) -> int:
        began = time.monotonic()
        signalled = False
        while True:
            try:
                return process.wait(timeout=self.settings.poll_seconds)
            except subprocess.TimeoutExpired:
                pass
            over = time.monotonic() - began > limit
            if (self.check_stop() is not None or over) and not signalled:
                signalled = True
                self._signal(process, signal.SIGTERM)
                try:
                    return process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    self._signal(process, signal.SIGKILL)
                    return process.wait(timeout=30)

    def _signal(self, process: subprocess.Popen[bytes], number: signal.Signals) -> None:
        try:
            if self.settings.guard:
                process.send_signal(number)  # the guard stops its own process group
            else:
                os.killpg(process.pid, number)
        except ProcessLookupError:
            pass

    def run(self) -> Json:
        settings = self.settings
        settings.run_dir.mkdir(parents=True, exist_ok=False)
        receipt: Json = {"schema": SCHEMA, "card": "C14", "step": 2, "started": utc_now(), "status": "running",
                         "purpose": "measurement only: no adoption, no M-gate decision; the R4 M >= 15% gate is unchanged",
                         "run_dir": str(settings.run_dir)}
        write_json_atomic(settings.run_dir / "receipt.json", receipt)
        inputs = self.inputs()
        write_json_atomic(settings.run_dir / "inputs.json", inputs)
        summary_inputs: Json = {key: value for key, value in inputs.items() if key != "prompts"}
        summary_inputs["prompts"] = [{key: value for key, value in prompt.items() if key != "prompt_tokens"}
                                     for prompt in cast(list[Json], inputs["prompts"])]
        receipt["inputs"] = summary_inputs
        receipt["driver_reuse"] = {
            "module": "sepalith.training.rl.campaign_rl_train",
            "rows_per_update": ROWS_PER_UPDATE, "generation_kwargs": dict(DEFAULT_GENERATION_KWARGS),
            "prompt_contract": "FixedIDGRPOTrainerMixin._prompt_ids and _padded_prompt_tensors (left padding, pad id 1)",
            "trim": "campaign_rl_profile.trim_generated_sequence at the 1,024-token cap",
            "length_policy": REVIEWED_POLICY_V4.policy_id,
            "loss": "TRL 0.24 GRPO with loss_type bnpo, beta 0, num_iterations 1 has the gradient "
                    "-advantage * grad log p / update tokens; the probe uses that loss directly",
        }
        receipt["fit_rule"] = {"desktop_budget_mib": settings.desktop_budget_mib, "margin_mib": settings.margin_mib,
                               "rule": "process peak (max peak reserved over phases + CUDA context) <= "
                                       "total - desktop budget - margin"}
        attempts: list[Json] = []
        ladders: Json = {}
        receipt["attempts"] = attempts
        receipt["ladders"] = ladders
        for name, configs in settings.plan:
            ladders[name] = run_ladder(configs, self._execute_and_save(receipt), attempts, can_start=self.can_start)
            write_json_atomic(settings.run_dir / "receipt.json", receipt)
        receipt["finished"] = utc_now()
        receipt["seconds"] = round(time.time() - self.started, 1)
        receipt["status"] = "partial" if self.stop_reason else "complete"
        receipt["stop_reason"] = self.stop_reason
        receipt["projection"] = projections(attempts)
        write_json_atomic(settings.run_dir / "receipt.json", receipt)
        summary = render_summary(receipt)
        (settings.run_dir / "summary.md").write_text(summary)
        if settings.docs_dir is not None:
            receipt["docs_copies"] = self._copy_to_docs(receipt, summary)
            write_json_atomic(settings.run_dir / "receipt.json", receipt)
        if settings.job_result is not None:
            write_json_atomic(settings.job_result, {"summary": one_line(receipt), "needs_user": None,
                                                    "evaluations": [str(settings.run_dir / "receipt.json")]})
        return receipt

    def _execute_and_save(self, receipt: Json) -> Callable[[ProbeConfig], Json]:
        def execute(config: ProbeConfig) -> Json:
            record = self.execute(config)
            write_json_atomic(self.settings.run_dir / "receipt.json", receipt)
            return record
        return execute

    def _copy_to_docs(self, receipt: Json, summary: str) -> Json:
        docs = cast(Path, self.settings.docs_dir)
        docs.mkdir(parents=True, exist_ok=True)
        stem = "C14-memory-probe"
        if (docs / f"{stem}.json").exists() or (docs / f"{stem}.md").exists():
            stem = f"{stem}-{self.settings.run_dir.name}"
        json_path, md_path = docs / f"{stem}.json", docs / f"{stem}.md"
        write_json_atomic(json_path, receipt)
        md_path.write_text(summary)
        return {"receipt": str(json_path), "summary": str(md_path)}


def summarise_phases(result: Mapping[str, object]) -> Json:
    out: Json = {}
    for name, phase in cast(Mapping[str, Mapping[str, object]], result.get("phases") or {}).items():
        keep = {key: phase.get(key) for key in ("status", "seconds", "peak_reserved_mib", "device_used_peak_mib",
                                                 "tokens_per_second", "generation_seconds", "generated_tokens",
                                                 "mean_completion_tokens", "offload_seconds", "reload_seconds",
                                                 "seconds_total", "seconds_per_micro_batch_median",
                                                 "micro_batches", "sequence_tokens_max", "error",
                                                 "optimizer_state_bytes") if phase.get(key) is not None}
        peak = phase.get("peak_allocated_bytes")
        if isinstance(peak, int):
            keep["peak_allocated_mib"] = _mib(peak)
        out[name] = keep
    return out


def update_timing(result: Mapping[str, object]) -> Json:
    """Seconds per GRPO update: worst case (every row at the cap) and with natural completion lengths."""
    phases = cast(Mapping[str, Mapping[str, object]], result.get("phases") or {})
    config = ProbeConfig.from_mapping(cast(Mapping[str, object], result["config"]))

    def ok(name: str) -> Mapping[str, object] | None:
        phase = phases.get(name)
        return phase if phase is not None and phase.get("status") == "ok" else None

    rollout, scoring, backward, step, natural = (ok(name) for name in
                                                 ("rollout", "scoring", "backward", "step", "natural_rollout"))
    timing: Json = {}
    if backward is not None:
        done = int(cast(int, backward.get("micro_batches") or 0))
        total = float(cast(float, backward.get("seconds_total") or 0.0))
        timing["backward_seconds_per_update"] = round(total * config.rows_per_update / done, 1) if done else None
    if rollout is None or backward is None or step is None:
        return timing
    rollout_s = float(cast(float, rollout["seconds"]))
    backward_s = float(cast(float, timing["backward_seconds_per_update"] or 0.0))
    step_s = float(cast(float, step["seconds"]))
    scoring_s = float(cast(float, scoring["seconds"])) if scoring is not None else 0.0
    timing.update({"rollout_seconds_worst_case": round(rollout_s, 1), "rollout_tokens_per_second": rollout.get("tokens_per_second"),
                   "scoring_seconds": round(scoring_s, 1), "step_seconds": round(step_s, 1),
                   "update_seconds_worst_case": round(rollout_s + backward_s + step_s, 1),
                   "update_seconds_worst_case_with_scoring": round(rollout_s + scoring_s + backward_s + step_s, 1)})
    if natural is not None:
        natural_s = float(cast(float, natural["seconds"]))
        timing.update({"rollout_seconds_natural": round(natural_s, 1),
                       "natural_mean_completion_tokens": natural.get("mean_completion_tokens"),
                       "update_seconds_natural_rollout": round(natural_s + backward_s + step_s, 1),
                       "update_seconds_note": "natural = natural-length rollout + backward at cap-length rows "
                                              "(an upper bound for the backward) + step; scoring is excluded because "
                                              "the driver at beta 0 and num_iterations 1 does not compute old log-probs"})
    return timing


def projections(attempts: Sequence[Mapping[str, object]]) -> list[Json]:
    out = []
    for attempt in attempts:
        verdict = cast(Mapping[str, object], attempt.get("assessment") or {})
        timing = cast(Mapping[str, object], attempt.get("timing") or {})
        if verdict.get("fits") is not True:
            continue
        row: Json = {"name": attempt["name"]}
        for label in ("update_seconds_worst_case", "update_seconds_natural_rollout"):
            seconds = timing.get(label)
            if isinstance(seconds, (int, float)) and seconds > 0:
                row[label] = seconds
                row[f"{label}_updates_per_8h"] = int(8 * 3600 // seconds)
                row[f"{label}_updates_per_16h"] = int(16 * 3600 // seconds)
        out.append(row)
    return out


def one_line(receipt: Mapping[str, object]) -> str:
    ladders = cast(Mapping[str, Mapping[str, object]], receipt.get("ladders") or {})
    fits = {name: ladder_.get("first_fit") for name, ladder_ in ladders.items() if name != "informational"}
    return (f"C14 GRPO memory probe {receipt.get('status')}: first fit {fits}; "
            f"receipt {receipt.get('run_dir')}/receipt.json (measurement only, no gate decision)")


def _fmt(value: object, digits: int = 0) -> str:
    if isinstance(value, bool) or value is None:
        return "-" if value is None else ("yes" if value else "no")
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def render_summary(receipt: Mapping[str, object]) -> str:
    attempts = cast(list[Mapping[str, object]], receipt.get("attempts") or [])
    rule = cast(Mapping[str, object], receipt.get("fit_rule") or {})
    lines = [
        "# C14 step 2: full-weight GRPO memory probe",
        "",
        f"Status: **{receipt.get('status')}**" + (f" (stopped: {receipt.get('stop_reason')})" if receipt.get("stop_reason") else ""),
        f"Run: `{receipt.get('run_dir')}`, {receipt.get('started')} to {receipt.get('finished')}.",
        "",
        "Measurement only: nothing is adopted and no gate is decided here. The R4 `M >= 15%` gate is unchanged.",
        "",
        "Model: arm A checkpoint 536 (bf16, full weights, gradient checkpointing, SDPA). Optimizer: the SFT recipe's "
        "aurora_mix with FP32 state. Prompts: the longest TRAIN prompts (DAT10 v4 rows). Update: 32 rows "
        "(G = 8: 4 prompts, G = 4: 8 prompts; G = 6: 30 rows). Rollout rows are forced to the 1,024-token cap.",
        "",
        f"Fit rule: process peak (allocator reserved + CUDA context) <= total - {_fmt(rule.get('desktop_budget_mib'))} MiB "
        f"desktop budget - {_fmt(rule.get('margin_mib'))} MiB margin. The desktop column is the GPU memory other "
        "processes held when each configuration started (nvidia-smi); headroom is shown at the budget and at that value.",
        "",
        "## Attempts",
        "",
        "| # | config | trick | status | rollout | scoring | backward | step | natural | process peak MiB | "
        "headroom @budget | headroom @observed | desktop MiB | fits | binding |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for number, attempt in enumerate(attempts, start=1):
        config = cast(Mapping[str, object], attempt.get("config") or {})
        phases = cast(Mapping[str, Mapping[str, object]], attempt.get("phases") or {})
        verdict = cast(Mapping[str, object], attempt.get("assessment") or {})
        device = cast(Mapping[str, object], attempt.get("device") or {})

        def cell(name: str, phases: Mapping[str, Mapping[str, object]] = phases) -> str:
            phase = phases.get(name)
            if phase is None:
                return "-"
            if phase.get("status") != "ok":
                return str(phase.get("status"))
            peak = phase.get("peak_reserved_mib")
            return f"{_fmt(peak / 1024, 1)} GiB" if isinstance(peak, (int, float)) else "ok"

        status = str(attempt.get("status"))
        if status in ("skipped", "not_run"):
            status = f"{status}: {attempt.get('reason')}"
        lines.append("| " + " | ".join([
            str(number), str(attempt.get("name")), str(config.get("trick")), status,
            cell("rollout"), cell("scoring"), cell("backward"), cell("step"), cell("natural_rollout"),
            _fmt(verdict.get("process_peak_mib")), _fmt(verdict.get("headroom_at_desktop_budget_mib")),
            _fmt(verdict.get("headroom_at_observed_desktop_mib")), _fmt(device.get("other_used_mib_at_start")),
            _fmt(verdict.get("fits")), _fmt(verdict.get("binding_phase")),
        ]) + " |")
    lines += ["", "Phase cells are peak reserved GiB per phase (allocator only; add the CUDA context for the process).", "",
              "## Time per update", "",
              "| config | rollout s (cap) | rollout tok/s | natural rollout s | natural mean tokens | scoring s | "
              "backward s/update | step s | update s (cap) | update s (natural) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for attempt in attempts:
        timing = cast(Mapping[str, object], attempt.get("timing") or {})
        if not timing:
            continue
        lines.append("| " + " | ".join([str(attempt.get("name"))] + [_fmt(timing.get(key), 1) for key in (
            "rollout_seconds_worst_case", "rollout_tokens_per_second", "rollout_seconds_natural",
            "natural_mean_completion_tokens", "scoring_seconds", "backward_seconds_per_update", "step_seconds",
            "update_seconds_worst_case", "update_seconds_natural_rollout")]) + " |")
    lines += ["", "## Ladders", ""]
    for name, ladder_ in cast(Mapping[str, Mapping[str, object]], receipt.get("ladders") or {}).items():
        stopped = f" (stopped: {ladder_.get('stopped')})" if ladder_.get("stopped") else ""
        if name == "informational":
            lines.append(f"- {name} (not part of the ladder): {', '.join(map(str, cast(list[object], ladder_.get('attempts') or [])))}{stopped}")
            continue
        lines.append(f"- {name}: first fit **{ladder_.get('first_fit') or 'none'}**{stopped}")
    lines += ["", "## Projection (fitting configurations)", ""]
    projection = cast(list[Mapping[str, object]], receipt.get("projection") or [])
    if not projection:
        lines.append("No configuration fitted, so there is no throughput projection.")
    for row in projection:
        lines.append(f"- {row.get('name')}: {_fmt(row.get('update_seconds_worst_case'), 1)} s per update at the cap "
                     f"({_fmt(row.get('update_seconds_worst_case_updates_per_16h'))} per 16 h, "
                     f"{_fmt(row.get('update_seconds_worst_case_updates_per_8h'))} per 8 h); "
                     f"{_fmt(row.get('update_seconds_natural_rollout'), 1)} s with natural completion lengths "
                     f"({_fmt(row.get('update_seconds_natural_rollout_updates_per_16h'))} per 16 h, "
                     f"{_fmt(row.get('update_seconds_natural_rollout_updates_per_8h'))} per 8 h).")
    lines += ["", "Card step 3 compares one update with 4 minutes; that decision is not taken here.",
              "Reward v2.1 is CPU-only and not run here (C11: about 2.7 s per 1,000 samples on 8 workers, "
              "well under a second per 32-row update).", ""]
    return "\n".join(lines)


def default_run_dir() -> Path:
    return STATE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the whole probe (the night job)")
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--docs-dir", type=Path)
    run.add_argument("--budget-seconds", type=float, default=10200.0)
    run.add_argument("--config-seconds", type=int, default=1800)
    run.add_argument("--no-guard", action="store_true", help="run children without cuda_host_guard (tests only)")
    child = sub.add_parser("child", help="one configuration (started by run)")
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--inputs", type=Path, required=True)
    child.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "child":
        return child_main(args.config, args.inputs, args.result)
    deadline = os.environ.get("SEPALITH_DEADLINE_EPOCH")
    stop = os.environ.get("SEPALITH_STOP_REQUEST_FILE")
    result = os.environ.get("SEPALITH_JOB_RESULT")
    settings = ProbeSettings(run_dir=args.run_dir or default_run_dir(), docs_dir=args.docs_dir,
                             guard=not args.no_guard, budget_seconds=args.budget_seconds,
                             config_seconds=args.config_seconds,
                             deadline_epoch=float(deadline) if deadline else None,
                             stop_file=Path(stop) if stop else None, job_result=Path(result) if result else None)
    probe = Probe(settings)
    signal.signal(signal.SIGTERM, lambda *_: probe.request_stop("SIGTERM"))
    receipt = probe.run()
    print(json.dumps({"status": receipt["status"], "run_dir": str(settings.run_dir),
                      "ladders": receipt["ladders"]}, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
