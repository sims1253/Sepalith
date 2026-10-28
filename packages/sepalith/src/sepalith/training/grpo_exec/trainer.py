#!/usr/bin/env python3
"""C18 GRPO trainer on the execution environment: one process per night segment on ``cuda:0``.

It is the C15 trainer (``training.grpo.trainer.GRPOTrainer``: the C14 memory fit ``g8-r16-resident``,
dynamic sampling, the DAPO token-normalised loss with 0.2/0.28 clipping and truncation masking,
beta = 0, saves at rollout-buffer boundaries, deadline and graceful stops, ``DeviceOps`` with an
explicit device index) with three changes:

1. Prompts come from ``mix.MixedSampler``: the exec training split, with every fifth drawn prompt from
   the legacy C14 TRAIN pool. Rewards are routed (``exec_env.reward.RoutedReward``): exec tasks get the
   execution reward (``sepalith.reward.exec.v1`` at gate c631141; ``exact`` is all-pass), legacy prompts
   reward v2.1. Group advantages stay within a prompt, so the two reward scales never mix.
2. Anti-forgetting by SFT cross-entropy replay: each optimizer update adds
   ``coefficient x (token-mean NLL of 8 TRAIN rows' target tokens)`` to the DAPO loss before the step
   (``mix.replay_ids``: a stateless schedule over the 8,000 replay rows). A KL penalty to a frozen 803
   was ruled out by the memory measurements bound in the recipe (``memory_fit``).
3. Evaluation steps follow R6x (every 50 updates, ``grpo_exec.rules``) and the telemetry splits drawn
   groups, rewards and zero-variance shares by source and logs the replay NLL and both gradient norms.

The validation split and the frozen hold-out are never loaded here: the exec reward holds only the
training tasks.

Exit codes as C15: 0 progress saved; 4 nothing done; 5 R7 disk stop; anything else a failure.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import time
from typing import Any, cast

import torch

from sepalith.training.grpo import algorithm
from sepalith.training.grpo.loss import ClipStats, row_surrogate
from sepalith.training.grpo.state import check_continue, verify_checkpoint
from sepalith.training.grpo.trainer import (
    EXIT_DISK,
    EXIT_NO_PROGRESS,
    EXIT_OK,
    DeviceOps,
    DiskLow,
    GRPOTrainer,
    PoolReward,
    Row,
    Settings,
    TrainerError,
    build_optimizer,
    check_model,
    load_optimizer_state,
    load_policy,
    require,
    resolve_device,
    start_cuda,
    write_json,
)
from sepalith.training.grpo_exec import mix, recipe as recipe_module, rules

Json = dict[str, object]


class ExecGRPOTrainer(GRPOTrainer):
    """``GRPOTrainer`` with mixed prompts, routed rewards and the SFT replay term."""

    def __init__(self, recipe: Mapping[str, object], recipe_sha: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 ops: DeviceOps, reward: Callable[[Sequence[str], Sequence[tuple[int, ...]]], list[Json]],
                 exec_ids: Sequence[str], legacy_ids: Sequence[str], prompts: Mapping[str, tuple[int, ...]],
                 replay: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]], settings: Settings, update: int = 0,
                 cursor: int = 0, resumed_from: int | None = None,
                 generate: Callable[..., torch.Tensor] | None = None) -> None:
        super().__init__(recipe, recipe_sha, model=model, optimizer=optimizer, ops=ops, reward=reward, pool_ids=exec_ids,
                         prompts=prompts, settings=settings, update=update, cursor=0, resumed_from=resumed_from, generate=generate)
        self.sampler = mix.MixedSampler(exec_ids, legacy_ids, int(cast(int, self.runtime["seed"])),
                                        int(cast(int, self.runtime["legacy_every"])), cursor)
        require(set(prompts) == set(exec_ids) | set(legacy_ids), "prompts must cover exactly the exec and legacy pools")
        self.legacy = frozenset(legacy_ids)
        self.replay = replay
        self.replay_ids = sorted(replay)
        self.replay_rows = int(cast(int, self.runtime["replay_rows_per_update"]))
        self.replay_coefficient = float(cast(float, self.runtime["replay_coefficient"]))
        require(self.replay_rows >= 1 and self.replay_coefficient >= 0 and self.replay_ids, "replay settings are invalid")

    def target_stop(self) -> int:
        limits = [step for step in rules.evaluation_steps(int(cast(int, self.pre["max_updates"]))) if step > self.update]
        require(limits, "no update remains below the pilot's limit")
        target = limits[0]
        if self.settings.stop_after is not None:
            target = min(target, self.settings.stop_after)
        require(target > self.update and (target - self.update) % self.updates_per_buffer == 0,
                f"the stop at update {target} is not a rollout-buffer boundary")
        return target

    # helpers ------------------------------------------------------------------------------------
    def source(self, prompt_id: str) -> str:
        return "legacy" if prompt_id in self.legacy else "exec"

    def source_stats(self, drawn: Sequence[algorithm.Group]) -> Json:
        out: Json = {}
        for name in ("exec", "legacy"):
            groups = [g for g in drawn if self.source(g.prompt_id) == name]
            rows = [row for g in groups for row in cast(list[Row], g.rows)]
            out[f"{name}_groups_drawn"] = len(groups)
            out[f"{name}_zero_variance_share_before"] = (sum(g.zero_variance for g in groups) / len(groups)) if groups else None
            out[f"{name}_reward_mean_drawn"] = (math.fsum(r.reward for r in rows) / len(rows)) if rows else None
            out[f"{name}_{'all_pass' if name == 'exec' else 'exact'}_drawn"] = (sum(r.exact for r in rows) / len(rows)) if rows else None
            out[f"{name}_completion_tokens_mean_drawn"] = (sum(len(r.completion) for r in rows) / len(rows)) if rows else None
        return out

    def replay_term(self, update: int) -> Json:
        """Backward ``coefficient x token-mean NLL`` of the update's replay rows; returns its statistics."""
        ids = mix.replay_ids(self.replay_ids, int(cast(int, self.runtime["seed"])), update, self.replay_rows)
        rows = [Row(identifier, *self.replay[identifier], "eos") for identifier in ids]
        tokens = sum(len(row.completion) for row in rows)
        require(tokens > 0, "replay rows have no target tokens")
        nll = 0.0
        for row in rows:
            logp = self.logprobs(row)
            loss = -logp.sum() * (self.replay_coefficient / tokens)
            require(bool(torch.isfinite(loss.detach())), "non-finite replay loss")
            loss.backward()
            nll += float(-logp.detach().sum())
        return {"replay_rows": len(rows), "replay_tokens": tokens, "replay_nll": nll / tokens,
                "replay_loss": self.replay_coefficient * nll / tokens}

    def grad_norm(self) -> float:
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        value = float(torch.stack([g.detach().norm().float() for g in grads]).norm()) if grads else 0.0
        require(math.isfinite(value), "non-finite gradient")
        return value

    # one buffer -------------------------------------------------------------------------------
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
            **self.source_stats(plan.drawn),
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
            grad_norm = grad_norm_grpo = None
            replay: Json = {"replay_rows": 0, "replay_tokens": 0, "replay_nll": None, "replay_loss": None}
            if backward_rows:
                grad_norm_grpo = self.grad_norm()
                replay = self.replay_term(self.update)
                grad_norm = self.grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            self.ops.synchronize()
            all_advantages = [a for group in groups for a in group.advantages]
            event: Json = {
                "event": "grpo_update", "update": self.update, "buffer_position": index, "resumed_from": self.resumed_from,
                "lr_factor": factor, "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
                "groups_trained": len(groups), "rows_trained": len(rows),
                "exec_groups_trained": sum(1 for group in groups if self.source(group.prompt_id) == "exec"),
                "legacy_groups_trained": sum(1 for group in groups if self.source(group.prompt_id) == "legacy"),
                "rows_masked_truncated": sum(row.truncated for row in rows), "rows_backward": backward_rows,
                "zero_variance_share_after": algorithm.zero_variance_share_after(groups, self.groups_per_update),
                "mean_abs_advantage": (math.fsum(abs(a) for a in all_advantages) / len(all_advantages)) if all_advantages else None,
                "reward_mean_trained": (math.fsum(row.reward for row in rows) / len(rows)) if rows else None,
                "exact_rate_trained": (sum(row.exact for row in rows) / len(rows)) if rows else None,
                "completion_tokens_mean_trained": (sum(len(row.completion) for row in rows) / len(rows)) if rows else None,
                "loss": loss_value if backward_rows else None, "grad_norm": grad_norm, "grad_norm_grpo": grad_norm_grpo,
                "optimizer_step": bool(backward_rows), **replay,
                **stats.summary(), **buffer_stats, "old_logprob_seconds": old_seconds,
                "update_seconds": self.settings.clock() - started, **self.ops.memory(),
            }
            self.emit(event)
            events.append(event)
        self.buffer_seconds.append(self.settings.clock() - began)
        return events


# Process entry ----------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[Json]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_geometry(prompt: Sequence[int], cap: int) -> bool:
    return bool(prompt) and prompt[0] == 0 and list(prompt).count(0) == 1 and 1 not in prompt and len(prompt) <= cap


def load_inputs(recipe: Mapping[str, object], root: Path) -> tuple[dict[str, tuple[int, ...]], list[str], list[str],
                                                                    dict[str, tuple[tuple[int, ...], tuple[int, ...]]], Path, Mapping]:
    """Exec training prompts, legacy prompts and bundle, replay rows (all hash-checked)."""
    from sepalith.training.grpo.pool import load_bundle, read_pool

    pool = cast(Json, recipe["pool"])
    cap = int(cast(int, cast(Json, recipe["runtime"])["prompt_max_tokens"]))
    train = recipe_module.check_file(root, pool["train"], "exec training tasks")
    tasks = read_jsonl(train)
    prompts = {str(t["id"]): tuple(int(x) for x in cast(list[int], t["prompt_ids"])) for t in tasks}
    require(len(prompts) == len(tasks) == int(cast(int, cast(Json, pool["train"])["tasks"])), "exec task ids repeat or differ in count")
    legacy = cast(Json, pool["legacy"])
    legacy_rows = read_pool(root / str(cast(Json, legacy["pool"])["path"]), str(cast(Json, legacy["pool"])["sha256"]))
    legacy_ids = [str(row["id"]) for row in legacy_rows]
    bundle_record = cast(Json, legacy["bundle"])
    bundle = load_bundle(Path(str(bundle_record["path"])), str(bundle_record["sha256"]), legacy_ids)
    require(not set(prompts) & set(legacy_ids), "exec task ids collide with legacy prompt ids")
    prompts.update({identifier: entry.prompt_ids for identifier, entry in bundle.items()})
    require(all(prompt_geometry(prompt, cap) for prompt in prompts.values()), "a prompt breaks the PRM03 geometry or the cap")
    replay_path = recipe_module.check_file(root, pool["replay"], "replay rows")
    replay = {str(r["id"]): (tuple(cast(list[int], r["prompt_ids"])), tuple(cast(list[int], r["target_ids"])))
              for r in read_jsonl(replay_path)}
    require(len(replay) == int(cast(int, cast(Json, pool["replay"])["rows"])), "replay row count differs")
    return prompts, sorted(str(t["id"]) for t in tasks), legacy_ids, replay, train, bundle


RewardFn = Callable[[Sequence[str], Sequence[tuple[int, ...]]], list[Json]]


def make_reward(recipe: Mapping[str, object], train_tasks: Path, bundle: Mapping) -> tuple[RewardFn, Any]:
    """``RoutedReward`` (exec reward for exec tasks, reward v2.1 for legacy prompts) and the harness to close."""
    from sepalith.exec_env.gate import gate_version
    from sepalith.exec_env.harness import ExecHarness
    from sepalith.exec_env.reward import ExecReward, ExecRewardConfig, RoutedReward, load_tasks
    from sepalith.training.grpo.trainer import make_pool_reward
    from sepalith.training.rl.reward_v21 import tokenizer_decoder

    version = gate_version()
    gate = cast(Json, cast(Json, cast(Json, recipe["pre_registered"])["reward"])["exec"])
    require(version["fingerprint"] == cast(Json, gate["gate"])["fingerprint"], f"exec_env fingerprint {version['fingerprint']} differs")
    runtime = cast(Json, recipe["runtime"])
    settings = cast(Json, runtime["reward"])
    config = ExecRewardConfig(w_behaviour=float(cast(float, gate["w_behaviour"])), w_all=float(cast(float, gate["w_all"])),
                              exact_bonus=float(cast(float, gate["exact_bonus"])),
                              max_completion_tokens=int(cast(int, runtime["completion_max_tokens"])),
                              overlong_factor=float(cast(float, gate["overlong_factor"])),
                              overlong_slack=int(cast(int, gate["overlong_slack"])),
                              overlong_scale=float(cast(float, gate["overlong_scale"])))
    harness = ExecHarness(workers=int(cast(int, settings["exec_workers"])), nice=int(cast(int, settings["exec_nice"])))
    parent = Path(str(cast(Json, recipe["parent"])["path"]))
    exec_reward = ExecReward(load_tasks(train_tasks), harness, tokenizer_decoder(parent / "tokenizer.json"), config)
    legacy_recipe = {**recipe, "pre_registered": {"reward": cast(Json, cast(Json, recipe["pre_registered"])["reward"])["legacy"]}}
    legacy: PoolReward = make_pool_reward(legacy_recipe, bundle)
    return cast(RewardFn, RoutedReward(exec_reward, legacy)), harness


def run(recipe_path: Path, resume: Path | None = None, stop_after: int | None = None) -> int:
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
    prompts, exec_ids, legacy_ids, replay, train_tasks, bundle = load_inputs(recipe, root)
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
    reward, harness = make_reward(recipe, train_tasks, bundle)
    stop = os.environ.get("SEPALITH_STOP_REQUEST_FILE")
    trainer = ExecGRPOTrainer(recipe, recipe_sha, model=model, optimizer=optimizer, ops=ops, reward=reward,
                              exec_ids=exec_ids, legacy_ids=legacy_ids, prompts=prompts, replay=replay,
                              settings=Settings(deadline=deadline_from_env(), stop_file=Path(stop) if stop else None,
                                                stop_after=stop_after),
                              update=update, cursor=cursor, resumed_from=resumed_from)
    trainer.emit({"event": "run_start", "recipe_sha256": recipe_sha, "resume": str(resume) if resume else None,
                  "device": device_info, "allocator": allocator, "memory_after_load": ops.memory(),
                  "generation_kwargs": trainer.kwargs, "exec_prompts": len(exec_ids), "legacy_prompts": len(legacy_ids),
                  "replay_rows": len(replay), "sampler": trainer.sampler.state(), "at_wall": time.time()})
    try:
        report = trainer.run()
    except DiskLow as error:
        trainer.emit({"event": "r7_disk_stop", "update": trainer.update, "error": str(error)})
        print(json.dumps({"status": "r7_disk", "error": str(error)}), flush=True)
        return EXIT_DISK
    finally:
        harness.close()
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


__all__ = ["ExecGRPOTrainer", "TrainerError", "main", "run"]
