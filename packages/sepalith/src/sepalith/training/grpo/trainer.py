#!/usr/bin/env python3
"""C15 GRPO trainer: one process per night segment on one CUDA device (``cuda:0``).

The trainer runs rollout buffers until the next R6 evaluation step, the update
limit, ``--stop-after-update``, the recipe-bound save-and-stop file or the
night deadline, then saves full state and exits so the night driver can run
the evaluator on a free GPU. It uses the configuration the C14 memory probe
fitted (``g8-r16-resident``): G = 8, 4 prompts (32 rows) per update, rollout in
16-row ``generate`` calls with the KV cache, the optimizer resident, micro-batch
1 forward and backward through the probe's chunked log-probabilities.

One rollout buffer covers ``updates_per_buffer`` (2) optimizer updates:

1. Dynamic sampling (``algorithm.dynamic_sample``): draw 1.5 x the 8 groups the
   buffer needs from the pool sampler, generate G = 8 completions per prompt at
   temperature 1.0, top_p 1.0, top_k off, score them with reward v2.1, drop
   zero-variance groups, refill.
2. Rollout-time log-probabilities (no grad, micro-batch 1) for the rows of the
   buffer's later updates; the first update is on-policy (``old = logp.detach()``).
3. Per update: linear warmup over 10 updates then a constant LR (a tenth of the
   SFT LR, aurora_mix), the DAPO token-normalised clipped surrogate
   (``loss.row_surrogate``), truncated completions masked, beta = 0, one
   optimizer step, one ``grpo_update`` telemetry event with the card's fields.

Saves happen only at buffer boundaries; a deadline stop picks the last
boundary whose next buffer and save would not finish before
``SEPALITH_DEADLINE_EPOCH``. Every ``torch.cuda`` call goes through
``DeviceOps`` with an explicit device index, after the CUDA context exists.

Exit codes: 0 progress saved; 4 nothing done (no update, nothing saved);
5 R7 (checkpoint disk below 150 GB, nothing written); anything else a failure.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import cast

import torch

from sepalith.campaign_protocol import BOS_ID, EOS_ID, NATIVE_EOG_IDS
from sepalith.training.checkpoint.campaign_checkpoint import flush_directory, flush_files
from sepalith.training.grpo import algorithm, recipe as recipe_module, rules
from sepalith.training.grpo.state import CHECKPOINT_SCHEMA, STATE_SCHEMA, check_continue, file_entry, verify_checkpoint
from sepalith.training.grpo.loss import ClipStats, row_surrogate

Json = dict[str, object]

EXIT_OK = 0
EXIT_NO_PROGRESS = 4
EXIT_DISK = 5
RUN_SCHEMA = "sepalith.c15.grpo-run.v1"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MIB = 1024 * 1024


class TrainerError(RuntimeError):
    """The trainer cannot continue safely."""


class DiskLow(TrainerError):
    """R7: the checkpoint disk would drop below its floor."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise TrainerError(message)


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


# CUDA ---------------------------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    """An explicit device: a bare "cuda" becomes ``cuda:0`` (``set_device`` rejects a device without an index)."""
    device = torch.device(name)
    return torch.device("cuda", 0) if device.type == "cuda" and device.index is None else device


def _touch_context(device: torch.device) -> None:
    torch.empty(1, device=device)


class DeviceOps:
    """Every ``torch.cuda`` call of the trainer, with an explicit index and after the context exists.

    ``cuda`` is ``torch.cuda`` in production and a fake in tests; on a CPU
    device it is unused and every call is a no-op.
    """

    def __init__(self, device: torch.device, cuda: object | None = None,
                 touch: Callable[[torch.device], None] = _touch_context) -> None:
        self.device = device
        self.cuda = cuda
        self.touch = touch
        self.initialized = device.type != "cuda"
        if device.type == "cuda":
            require(device.index is not None, "a CUDA device needs an explicit index")
            require(cuda is not None, "a CUDA device needs torch.cuda")

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _cuda(self) -> object:
        require(self.initialized, "CUDA context not initialised before a memory or RNG call")
        return self.cuda

    def init(self) -> Json:
        """Select the device and create its context; memory statistics are valid only after this."""
        if not self.is_cuda:
            return {"device": str(self.device)}
        cuda = self.cuda
        getattr(cuda, "set_device")(self.device)  # noqa: B009
        self.touch(self.device)
        getattr(cuda, "synchronize")(self.device)  # noqa: B009
        self.initialized = True
        free, total = getattr(cuda, "mem_get_info")(self.device)  # noqa: B009
        return {"device": str(self.device), "name": getattr(cuda, "get_device_name")(self.device),  # noqa: B009
                "free_mib": round(free / MIB, 1), "total_mib": round(total / MIB, 1)}

    def cap_allocator(self, safety_mib: float) -> Json:
        """Cap the caching allocator at the free memory so an overflow is an OOM, not the sysmem fallback."""
        if not self.is_cuda:
            return {}
        cuda = self._cuda()
        free, total = getattr(cuda, "mem_get_info")(self.device)  # noqa: B009
        fraction = max(0.05, min(1.0, (free - safety_mib * MIB) / total))
        getattr(cuda, "set_per_process_memory_fraction")(fraction, self.device)  # noqa: B009
        return {"allocator_fraction": round(fraction, 4), "allocator_cap_mib": round(fraction * total / MIB, 1)}

    def reset_peak(self) -> None:
        if self.is_cuda:
            getattr(self._cuda(), "reset_peak_memory_stats")(self.device)  # noqa: B009

    def synchronize(self) -> None:
        if self.is_cuda:
            getattr(self._cuda(), "synchronize")(self.device)  # noqa: B009

    def memory(self) -> Json:
        if not self.is_cuda:
            return {}
        cuda = self._cuda()
        return {name: int(getattr(cuda, name)(self.device)) for name in
                ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")}

    def rng_state(self) -> torch.Tensor | None:
        return cast(torch.Tensor, getattr(self._cuda(), "get_rng_state")(self.device)) if self.is_cuda else None  # noqa: B009

    def set_rng_state(self, state: torch.Tensor | None) -> None:
        if self.is_cuda:
            require(state is not None, "the checkpoint lacks the CUDA RNG state")
            getattr(self._cuda(), "set_rng_state")(state, self.device)  # noqa: B009


# Rollout and log-probabilities ------------------------------------------------------------------------

@dataclass
class Row:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    completion: tuple[int, ...]
    terminal_reason: str
    reward: float = 0.0
    exact: bool = False
    outcome: str = ""

    @property
    def truncated(self) -> bool:
        return self.terminal_reason == "length"


def generation_kwargs(runtime: Mapping[str, object], pre: Mapping[str, object]) -> Json:
    """Sampling at the card's T and top_p with top-k off; every field set, so the checkpoint's
    ``generation_config`` (top_p 0.95) cannot leak in."""
    return {"do_sample": True, "temperature": float(cast(float, pre["temperature"])), "top_p": float(cast(float, pre["top_p"])),
            "top_k": int(cast(int, pre["top_k"])), "repetition_penalty": 1.0,
            "max_new_tokens": int(cast(int, runtime["completion_max_tokens"])),
            "eos_token_id": list(NATIVE_EOG_IDS), "pad_token_id": EOS_ID, "bos_token_id": BOS_ID,
            "num_return_sequences": 1, "use_cache": True, "return_dict_in_generate": False}


def rollout(model: torch.nn.Module, prompts: Sequence[tuple[str, tuple[int, ...]]], *, group_size: int, rows_per_call: int,
            kwargs: Mapping[str, object], device: torch.device,
            generate: Callable[..., torch.Tensor] | None = None) -> list[list[Row]]:
    """G sampled completions per prompt, in calls of ``rows_per_call`` rows (left-padded, fixed prompt IDs)."""
    from sepalith.training.rl.campaign_rl_profile import trim_generated_sequence
    from sepalith.training.rl.campaign_rl_train import FixedIDGRPOTrainerMixin

    require(rows_per_call % group_size == 0, "a generate call holds whole groups")
    per_call = rows_per_call // group_size
    cap = int(cast(int, kwargs["max_new_tokens"]))
    call = generate if generate is not None else cast(Callable[..., torch.Tensor], getattr(model, "generate"))  # noqa: B009
    groups: list[list[Row]] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(prompts), per_call):
            chunk = prompts[start:start + per_call]
            ids = [list(prompt) for _, prompt in chunk for _ in range(group_size)]
            input_ids, attention, width = FixedIDGRPOTrainerMixin._padded_prompt_tensors(ids, torch, device)
            with torch.inference_mode():
                sequences = call(input_ids=input_ids, attention_mask=attention, **kwargs)
            require(torch.equal(sequences[:, :width], input_ids), "generation changed the prompt IDs")
            tails = sequences[:, width:].tolist()
            for index, (prompt_id, prompt) in enumerate(chunk):
                rows = []
                for offset in range(group_size):
                    row_ids = ids[index * group_size + offset]
                    trimmed = trim_generated_sequence(row_ids + tails[index * group_size + offset], len(row_ids),
                                                      completion_max_tokens=cap)
                    reason = str(trimmed["terminal_reason"])
                    require(reason in ("eos", "length"), f"generation ended without a terminal reason ({reason})")
                    rows.append(Row(prompt_id, tuple(prompt), tuple(int(t) for t in cast(list[int], trimmed["generated_tokens"])),
                                    reason))
                groups.append(rows)
    finally:
        model.train(was_training)
    return groups


def row_logprobs(model: torch.nn.Module, row: Row, *, seq_chunk: int, vocab_chunk: int) -> torch.Tensor:
    """Per-token log p(completion | prompt) at temperature 1.0, micro-batch 1, chunked over sequence and vocabulary."""
    from sepalith.training.rl.grpo_memory_probe import model_completion_logprobs

    (values,) = model_completion_logprobs(model, [(list(row.prompt_ids), list(row.completion))], temperature=1.0,
                                          seq_chunk=seq_chunk, vocab_chunk=vocab_chunk, impl="chunked", pad_id=EOS_ID)
    return values


# Reward ---------------------------------------------------------------------------------------------

RewardFn = Callable[[Sequence[str], Sequence[tuple[int, ...]]], list[Json]]


class PoolReward:
    """Reward v2.1 over the pool bundle; references are built once per prompt."""

    def __init__(self, bundle: Mapping[str, object], scorer: object, decoder: Callable[[Sequence[int]], str]) -> None:
        self.bundle = bundle
        self.scorer = scorer
        self.decoder = decoder
        self.references: dict[str, object] = {}

    def __call__(self, ids: Sequence[str], completions: Sequence[tuple[int, ...]]) -> list[Json]:
        from sepalith.training.grpo.pool import BundleEntry, reference_spec
        from sepalith.training.rl.reward_v21 import RewardV21Scorer, Sample

        scorer = cast(RewardV21Scorer, self.scorer)
        missing = sorted({identifier for identifier in ids if identifier not in self.references})
        if missing:
            specs = [reference_spec(cast(BundleEntry, self.bundle[identifier])) for identifier in missing]
            self.references.update(scorer.build_references(specs))  # ty: ignore[invalid-argument-type]
        samples = [Sample(identifier, completion_ids=tuple(completion)) for identifier, completion in zip(ids, completions, strict=True)]
        return scorer.score(self.references, samples, self.decoder)  # ty: ignore[invalid-argument-type]


def make_pool_reward(recipe: Mapping[str, object], bundle: Mapping[str, object]) -> PoolReward:
    from sepalith.training.rl.air_canonical import AirFormatter
    from sepalith.training.rl.campaign_r_parse_probe import DEFAULT_HARNESS, RParseOnlyProbe, sha_file
    from sepalith.training.rl.reward_v21 import RewardV21Config, RewardV21Scorer, tokenizer_decoder

    runtime = cast(Json, recipe["runtime"])
    settings = cast(Json, runtime["reward"])
    reward = cast(Json, cast(Json, recipe["pre_registered"])["reward"])
    probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS), command=(str(settings["rscript"]), "--vanilla"))
    formatter = AirFormatter((str(settings["air"]),))
    config = RewardV21Config(partial_credit=str(reward["partial_credit"]), partial_weight=float(cast(float, reward["partial_weight"])),
                             max_completion_tokens=int(cast(int, runtime["completion_max_tokens"])))
    scorer = RewardV21Scorer(probe, formatter, config, workers=int(cast(int, settings["workers"])))
    parent = Path(str(cast(Json, recipe["parent"])["path"]))
    return PoolReward(bundle, scorer, tokenizer_decoder(parent / "tokenizer.json"))


# Trainer --------------------------------------------------------------------------------------------

@dataclass
class Settings:
    deadline: float | None = None
    stop_file: Path | None = None
    stop_after: int | None = None
    clock: Callable[[], float] = time.time


class GRPOTrainer:
    def __init__(self, recipe: Mapping[str, object], recipe_sha: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 ops: DeviceOps, reward: RewardFn, pool_ids: Sequence[str], prompts: Mapping[str, tuple[int, ...]],
                 settings: Settings, update: int = 0, cursor: int = 0, resumed_from: int | None = None,
                 generate: Callable[..., torch.Tensor] | None = None) -> None:
        self.recipe, self.recipe_sha = recipe, recipe_sha
        self.runtime = cast(Json, recipe["runtime"])
        self.pre = cast(Json, recipe["pre_registered"])
        self.model, self.optimizer, self.ops, self.reward = model, optimizer, ops, reward
        self.prompts = prompts
        self.sampler = algorithm.PoolSampler(pool_ids, int(cast(int, self.runtime["seed"])), cursor)
        self.settings = settings
        self.update = update
        self.resumed_from = resumed_from
        self.generate = generate
        outputs = cast(Json, recipe["outputs"])
        self.archive = Path(str(outputs["archive"]))
        self.telemetry = self.archive / "telemetry.jsonl"
        self.stop_path = Path(str(outputs["graceful_stop"]))
        self.parent = Path(str(cast(Json, recipe["parent"])["path"]))
        self.group_size = int(cast(int, self.runtime["group_size"]))
        self.groups_per_update = int(cast(int, self.runtime["groups_per_update"]))
        self.updates_per_buffer = int(cast(int, self.runtime["updates_per_buffer"]))
        sampling = cast(Json, self.pre["dynamic_sampling"])
        self.oversample = float(cast(float, sampling["oversample"]))
        self.max_rounds = int(cast(int, sampling["max_rounds"]))
        self.max_draw_factor = float(cast(float, sampling["max_draw_factor"]))
        self.kwargs = generation_kwargs(self.runtime, self.pre)
        self.buffer_seconds: list[float] = []
        self.save_seconds: list[float] = []
        self.saved: list[int] = []
        self.stop_reasons: list[str] = []

    # schedule and stops -----------------------------------------------------------------------
    def lr_factor(self, update: int) -> float:
        return min(1.0, update / int(cast(int, self.pre["warmup_updates"])))

    def set_lr(self, update: int) -> float:
        factor = self.lr_factor(update)
        for group in self.optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * factor
        return factor

    def target_stop(self) -> int:
        limits = [step for step in rules.evaluation_steps(int(cast(int, self.pre["max_updates"]))) if step > self.update]
        require(limits, "no update remains below the pilot's limit")
        target = limits[0]
        if self.settings.stop_after is not None:
            target = min(target, self.settings.stop_after)
        require(target > self.update and (target - self.update) % self.updates_per_buffer == 0,
                f"the stop at update {target} is not a rollout-buffer boundary")
        return target

    def graceful_stop_requested(self) -> bool:
        if self.settings.stop_file is not None and self.settings.stop_file.exists():
            return True  # the night runner's own stop request (16:50), in case the driver's relay is late
        if not self.stop_path.exists():
            return False
        request = json.loads(self.stop_path.read_text())
        require(request == {"action": "save_and_stop", "bound_recipe_sha256": self.recipe_sha}, "graceful stop request identity differs")
        return True

    def deadline_too_close(self) -> bool:
        if self.settings.deadline is None:
            return False
        buffer = max(self.buffer_seconds[-3:]) if self.buffer_seconds else float(cast(float, self.runtime["buffer_seconds_prior"]))
        save = self.save_seconds[-1] if self.save_seconds else float(cast(float, self.runtime["save_seconds_prior"]))
        margin = float(cast(float, self.runtime["deadline_margin_seconds"]))
        return self.settings.clock() + buffer + save + margin >= self.settings.deadline

    def stop_reason(self, target: int) -> str | None:
        if self.update >= target:
            return "stop_after_update" if self.settings.stop_after == target else "evaluation_or_limit"
        if self.graceful_stop_requested():
            return "graceful_stop"
        if self.deadline_too_close():
            return "deadline"
        return None

    # one buffer -------------------------------------------------------------------------------
    def emit(self, record: Json) -> None:
        append_jsonl(self.telemetry, {**record, "at": time.time()})

    def draw(self, count: int) -> list[algorithm.Group]:
        drawn = self.sampler.draw(count)
        prompts = [(identifier, self.prompts[identifier]) for _, identifier in drawn]
        groups = rollout(self.model, prompts, group_size=self.group_size,
                         rows_per_call=int(cast(int, self.runtime["rollout_rows_per_call"])), kwargs=self.kwargs,
                         device=self.ops.device, generate=self.generate)
        rows = [row for group in groups for row in group]
        records = self.reward([row.prompt_id for row in rows], [row.completion for row in rows])
        require(len(records) == len(rows), "reward returned a different number of records")
        for row, record in zip(rows, records, strict=True):
            row.reward = float(cast(float, record["reward"]))
            row.exact = bool(record["exact"])
            row.outcome = str(record.get("outcome"))
            require(math.isfinite(row.reward), "non-finite reward")
        return [algorithm.Group(identifier, position, tuple(row.reward for row in group), group)
                for (position, identifier), group in zip(drawn, groups, strict=True)]

    def logprobs(self, row: Row) -> torch.Tensor:
        return row_logprobs(self.model, row, seq_chunk=int(cast(int, self.runtime["seq_chunk"])),
                            vocab_chunk=int(cast(int, self.runtime["vocab_chunk"])))

    def run_buffer(self) -> list[Json]:
        began = self.settings.clock()
        self.ops.reset_peak()
        needed = self.updates_per_buffer * self.groups_per_update
        plan = algorithm.dynamic_sample(needed, self.draw, oversample=self.oversample, max_draw_factor=self.max_draw_factor,
                                        max_rounds=self.max_rounds)
        rollout_seconds = self.settings.clock() - began
        assignment = algorithm.assign_updates(plan.selected, self.updates_per_buffer, self.groups_per_update)
        drawn_rows = [row for group in plan.drawn for row in cast(list[Row], group.rows)]
        buffer_stats: Json = {
            "groups_drawn": len(plan.drawn), "rows_drawn": len(drawn_rows), "refill_rounds": plan.rounds - 1,
            "round_sizes": plan.round_sizes, "surplus_discarded": plan.surplus_discarded, "refill_exhausted": plan.exhausted,
            "zero_variance_share_before": plan.zero_variance_share_before,
            "reward_mean_drawn": math.fsum(row.reward for row in drawn_rows) / len(drawn_rows),
            "exact_rate_drawn": sum(row.exact for row in drawn_rows) / len(drawn_rows),
            "completion_tokens_mean_drawn": sum(len(row.completion) for row in drawn_rows) / len(drawn_rows),
            "truncated_drawn": sum(row.truncated for row in drawn_rows), "rollout_seconds": rollout_seconds,
        }
        started_old = self.settings.clock()
        old: dict[tuple[int, int, int], torch.Tensor] = {}
        with torch.no_grad():
            for index, groups in enumerate(assignment[1:], start=1):
                for g, group in enumerate(groups):
                    advantages = group.advantages
                    for r, row in enumerate(cast(list[Row], group.rows)):
                        if not row.truncated and advantages[r] != 0.0:
                            old[(index, g, r)] = self.logprobs(row).detach()
        old_seconds = self.settings.clock() - started_old
        events = []
        for index, groups in enumerate(assignment):
            started = self.settings.clock()
            self.update += 1
            factor = self.set_lr(self.update)
            rows = [row for group in groups for row in cast(list[Row], group.rows)]
            tokens = sum(len(row.completion) for row in rows if not row.truncated)
            stats = ClipStats()
            loss_value = 0.0
            backward_rows = 0
            for g, group in enumerate(groups):
                advantages = group.advantages
                for r, row in enumerate(cast(list[Row], group.rows)):
                    if row.truncated:
                        continue
                    if advantages[r] == 0.0:
                        stats.tokens += len(row.completion)  # zero advantage: no gradient, still in N
                        continue
                    logp = self.logprobs(row)
                    reference = old[(index, g, r)] if index else logp.detach()
                    loss_sum, row_stats = row_surrogate(logp, reference, advantages[r],
                                                        eps_low=float(cast(float, self.pre["epsilon_low"])),
                                                        eps_high=float(cast(float, self.pre["epsilon_high"])))
                    scaled = loss_sum / tokens
                    require(bool(torch.isfinite(scaled.detach())), "non-finite loss")
                    scaled.backward()
                    loss_value += float(scaled.detach())
                    stats.add(row_stats)
                    backward_rows += 1
            grad_norm = None
            if backward_rows:
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                grad_norm = float(torch.stack([g.detach().norm().float() for g in grads]).norm())
                require(math.isfinite(grad_norm), "non-finite gradient")
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            self.ops.synchronize()
            all_advantages = [a for group in groups for a in group.advantages]
            event: Json = {
                "event": "grpo_update", "update": self.update, "buffer_position": index, "resumed_from": self.resumed_from,
                "lr_factor": factor, "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
                "groups_trained": len(groups), "rows_trained": len(rows),
                "rows_masked_truncated": sum(row.truncated for row in rows), "rows_backward": backward_rows,
                "zero_variance_share_after": algorithm.zero_variance_share_after(groups, self.groups_per_update),
                "mean_abs_advantage": (math.fsum(abs(a) for a in all_advantages) / len(all_advantages)) if all_advantages else None,
                "reward_mean_trained": (math.fsum(row.reward for row in rows) / len(rows)) if rows else None,
                "exact_rate_trained": (sum(row.exact for row in rows) / len(rows)) if rows else None,
                "completion_tokens_mean_trained": (sum(len(row.completion) for row in rows) / len(rows)) if rows else None,
                "loss": loss_value if backward_rows else None, "grad_norm": grad_norm, "optimizer_step": bool(backward_rows),
                **stats.summary(), **buffer_stats, "old_logprob_seconds": old_seconds,
                "update_seconds": self.settings.clock() - started, **self.ops.memory(),
            }
            self.emit(event)
            events.append(event)
        self.buffer_seconds.append(self.settings.clock() - began)
        return events

    # saving ---------------------------------------------------------------------------------
    def check_disk(self) -> None:
        free = shutil.disk_usage(self.archive).free
        floor = int(cast(int, self.runtime["min_free_disk_bytes"]))
        need = int(cast(int, self.runtime["checkpoint_bytes_estimate"]))
        if free - need < floor:
            raise DiskLow(f"R7: {free / 1e9:.0f} GB free on the checkpoint disk; a {need / 1e9:.1f} GB save would leave "
                          f"less than {floor / 1e9:.0f} GB")

    def save(self) -> Path:
        started = self.settings.clock()
        self.check_disk()
        step = self.update
        final = self.archive / "full" / f"checkpoint-{step}"
        require(not final.exists(), f"checkpoint {final} already exists")
        temporary = self.archive / "full" / f".partial-checkpoint-{step}-{os.getpid()}"
        temporary.mkdir(parents=True)
        save_pretrained = cast(Callable[..., None], getattr(self.model, "save_pretrained"))  # noqa: B009
        save_pretrained(str(temporary), max_shard_size="50GB")
        for name in TOKENIZER_FILES:
            shutil.copyfile(self.parent / name, temporary / name)
        torch.save(self.optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save({"cpu": torch.get_rng_state(), "cuda": self.ops.rng_state()}, temporary / "rng_state.pt")
        state: Json = {"schema": STATE_SCHEMA, "update": step, "sampler": self.sampler.state(), "recipe_sha256": self.recipe_sha,
                       "resumed_from": self.resumed_from, "updates_per_buffer": self.updates_per_buffer,
                       "parent_model_sha256": cast(Json, cast(Json, self.recipe["parent"])["files"])["model.safetensors"]}
        write_json(temporary / "trainer-state.json", state)
        flush_files(temporary)
        files = {path.name: file_entry(path) for path in sorted(temporary.iterdir())}
        manifest: Json = {"schema": CHECKPOINT_SCHEMA, "checkpoint_kind": "grpo_full_state", "step": step,
                          "recipe_sha256": self.recipe_sha, "arm": cast(Json, self.recipe["arm"])["name"], "files": files}
        write_json(temporary / "campaign-manifest.json", manifest)
        flush_directory(temporary)
        os.rename(temporary, final)
        flush_directory(final.parent)
        seconds = self.settings.clock() - started
        self.save_seconds.append(seconds)
        self.saved.append(step)
        self.emit({"event": "checkpoint_durable", "step": step, "path": str(final), "seconds": seconds})
        return final

    # the segment ------------------------------------------------------------------------------
    def run(self) -> Json:
        target = self.target_stop()
        start = self.update
        self.emit({"event": "segment_start", "update": self.update, "target": target, "resumed_from": self.resumed_from,
                   "sampler": self.sampler.state()})
        if self.graceful_stop_requested() or self.deadline_too_close():
            reason = "graceful_stop" if self.graceful_stop_requested() else "deadline"
            self.emit({"event": "segment_end", "update": self.update, "reason": f"{reason}_before_first_buffer"})
            return {"status": "no_progress", "reason": reason, "update": self.update}
        reason = None
        while reason is None:
            self.run_buffer()
            reason = self.stop_reason(target)
        self.stop_reasons.append(reason)
        path = self.save()
        report: Json = {"schema": RUN_SCHEMA, "status": "saved", "reason": reason, "from_update": start, "update": self.update,
                        "checkpoint": str(path), "buffer_seconds": self.buffer_seconds, "save_seconds": self.save_seconds}
        self.emit({"event": "segment_end", **report})
        return report


# Process entry ----------------------------------------------------------------------------------------

def start_cuda(ops: DeviceOps, runtime: Mapping[str, object]) -> tuple[Json, Json]:
    """Context first, then the allocator cap and the peak-statistics reset (both need the context)."""
    info = ops.init()
    allocator = ops.cap_allocator(float(cast(float, runtime["allocator_safety_mib"])))
    ops.reset_peak()
    return info, allocator


def load_policy(path: Path, device: torch.device, runtime: Mapping[str, object]) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=getattr(torch, str(runtime["dtype"])),
                                                 attn_implementation=str(runtime["attn_implementation"]),
                                                 device_map={"": str(device)}, local_files_only=True, trust_remote_code=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def check_model(model: torch.nn.Module, expected: Mapping[str, object]) -> None:
    named = list(model.named_parameters())
    require(len(named) == expected["parameter_tensors"] and sum(p.numel() for _, p in named) == expected["parameters"]
            and all(p.requires_grad for _, p in named), "full-weight trainable inventory differs")
    require(not any("lora_" in name.lower() for name, _ in named), "a PEFT parameter entered the full-weight run")


def build_optimizer(model: torch.nn.Module, runtime: Mapping[str, object], *, check_dispatch: bool = True) -> torch.optim.Optimizer:
    from sepalith.training.optim.full_weight_optimizer import OptimizerConfig, build_full_weight_optimizer

    config = dict(cast(Json, runtime["optimizer"]))
    config["adam_betas"] = tuple(cast(list[float], config["adam_betas"]))
    optimizer, manifest = build_full_weight_optimizer(model, OptimizerConfig(**config))  # ty: ignore[invalid-argument-type]
    if check_dispatch:
        require(manifest["ordered_rows_sha256"] == runtime["optimizer_dispatch_sha256"], "optimizer dispatch differs from the recipe")
    return optimizer


def load_optimizer_state(optimizer: torch.optim.Optimizer, path: Path) -> None:
    """Memory-mapped from disk so the 12 GB state never sits twice in host memory."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, OSError):
        state = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)
    del state


def run(recipe_path: Path, resume: Path | None = None, stop_after: int | None = None) -> int:
    from sepalith.training.grpo.pool import load_bundle, read_pool
    from sepalith.training.sft.night_controls import deadline_from_env

    recipe = recipe_module.load(recipe_path)
    recipe_sha = recipe_module.sha256(recipe_path)
    runtime = cast(Json, recipe["runtime"])
    root = recipe_module.repository_root(recipe)
    archive = Path(str(cast(Json, recipe["outputs"])["archive"]))
    if resume is None:
        require(not (archive / "full").exists() or not any((archive / "full").glob("checkpoint-*")),
                "a fresh run needs an archive without checkpoints")
    archive.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(runtime["device"]))
    require(device.type == "cuda" and device.index == 0, "the trainer runs on cuda:0")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA device required")
    ops = DeviceOps(device, torch.cuda)
    device_info, allocator = start_cuda(ops, runtime)
    pool_record = cast(Json, recipe["pool"])
    pool_rows = read_pool(root / str(cast(Json, pool_record["pool"])["path"]), str(cast(Json, pool_record["pool"])["sha256"]))
    pool_ids = [str(row["id"]) for row in pool_rows]
    bundle_record = cast(Json, pool_record["bundle"])
    bundle = load_bundle(Path(str(bundle_record["path"])), str(bundle_record["sha256"]), pool_ids)
    prompts = {identifier: entry.prompt_ids for identifier, entry in bundle.items()}
    prompt_cap = int(cast(int, runtime["prompt_max_tokens"]))
    require(all(len(prompt) <= prompt_cap for prompt in prompts.values()), "a pool prompt exceeds the prompt cap")
    update, cursor, resumed_from = 0, 0, None
    source = Path(str(cast(Json, recipe["parent"])["path"]))
    if resume is not None:
        state = verify_checkpoint(resume, recipe_sha)
        update, resumed_from = int(cast(int, state["update"])), int(cast(int, state["update"]))
        cursor = int(cast(int, cast(Json, state["sampler"])["cursor"]))
        if update in rules.evaluation_steps(int(cast(int, cast(Json, recipe["pre_registered"])["max_updates"]))):
            check_continue(recipe, recipe_sha, resume, update)
        source = resume
    else:
        require(recipe_module.sha256(source / "model.safetensors")
                == cast(Json, cast(Json, recipe["parent"])["files"])["model.safetensors"], "starting model weights differ")
        torch.manual_seed(int(cast(int, runtime["seed"])))
    model = load_policy(source, device, runtime)
    check_model(model, cast(Json, recipe["expected_model"]))
    optimizer = build_optimizer(model, runtime)
    if resume is not None:
        load_optimizer_state(optimizer, resume / "optimizer.pt")
        rng = torch.load(resume / "rng_state.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        ops.set_rng_state(rng["cuda"])
    reward = make_pool_reward(recipe, bundle)
    deadline = deadline_from_env()
    stop = os.environ.get("SEPALITH_STOP_REQUEST_FILE")
    trainer = GRPOTrainer(recipe, recipe_sha, model=model, optimizer=optimizer, ops=ops, reward=reward, pool_ids=pool_ids,
                          prompts=prompts, settings=Settings(deadline=deadline, stop_file=Path(stop) if stop else None,
                                                             stop_after=stop_after),
                          update=update, cursor=cursor, resumed_from=resumed_from)
    trainer.emit({"event": "run_start", "recipe_sha256": recipe_sha, "resume": str(resume) if resume else None,
                  "device": device_info, "allocator": allocator, "memory_after_load": ops.memory(),
                  "generation_kwargs": trainer.kwargs, "pool_prompts": len(pool_ids)})
    try:
        report = trainer.run()
    except DiskLow as error:
        trainer.emit({"event": "r7_disk_stop", "update": trainer.update, "error": str(error)})
        print(json.dumps({"status": "r7_disk", "error": str(error)}), flush=True)
        return EXIT_DISK
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] == "no_progress":
        return EXIT_NO_PROGRESS
    write_json(archive / "run-result.json", report)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args(argv)
    return run(args.recipe, args.resume, args.stop_after_update)


if __name__ == "__main__":
    raise SystemExit(main())
