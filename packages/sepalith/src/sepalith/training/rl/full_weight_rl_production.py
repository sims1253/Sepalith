#!/usr/bin/env python3
"""Dense GRPO production seam over the accepted fixed-ID/PRM03 trainer.

This module does not load a model or start training.  It composes the reviewed
full-weight optimizer with the pinned campaign GRPO class and validates the
actual generated/rewarded completion IDs before inherited TRL computes loss.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from sepalith.training.optim.full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
from sepalith.training.contracts.runtime_update_contract import ids_sha256, prepare_update

SHA_FIELDS = (
    "source_manifest_sha256", "model_manifest_sha256", "model_weights_sha256",
    "tokenizer_json_sha256", "reward_buffer_manifest_sha256",
    "prompt_context_manifest_sha256", "generation_policy_sha256",
    "optimizer_dispatch_sha256",
    "optimizer_config_sha256", "saved_precision_manifest_sha256",
)


class ProductionRLError(ValueError):
    pass


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ProductionRLError(reason)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()


def validate_production_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable production inputs without importing torch or TRL."""
    require(binding.get("schema") == "sepalith.rl11.full-weight-production-binding.v1",
            "production binding schema mismatch")
    require(binding.get("status") == "root_admitted", "production binding is not root admitted")
    require(binding.get("optimizer_updates") is True, "production binding is diagnostic-only")
    require(binding.get("reward_implementation") == "CampaignPRM03Reward",
            "production reward is not CampaignPRM03Reward")
    require(binding.get("synthetic_reward") is False, "synthetic reward is forbidden")
    require(binding.get("train_split_only") is True, "reward/data binding is not TRAIN-only")
    require(binding.get("sealed_final_access") is False, "sealed final access is forbidden")
    require(binding.get("checkpoint_kind") == "full_weights", "checkpoint kind must be full_weights")
    require(binding.get("expected_parameter_tensors") == 381, "dense tensor count differs")
    require(binding.get("expected_trainable_parameters") == 2516756480,
            "dense trainable parameter count differs")
    require(binding.get("peft_tensors") == 0, "PEFT tensors are forbidden")
    for name in SHA_FIELDS:
        value = binding.get(name)
        require(isinstance(value, str) and len(value) == 64 and
                all(ch in "0123456789abcdef" for ch in value), f"{name} is not a SHA-256")
    geometry = binding.get("geometry")
    require(isinstance(geometry, Mapping), "geometry is missing")
    for name in ("candidate_count", "prompt_groups_per_update", "steps_per_generation"):
        require(type(geometry.get(name)) is int and geometry[name] > 0, f"geometry.{name} must be positive integer")
    require(geometry["candidate_count"] in (4, 8), "candidate count is not admitted")
    require(geometry["steps_per_generation"] == binding.get("gradient_accumulation_steps"),
            "one-buffer-per-update geometry differs")
    for name in ("initial_policy_step", "initial_rollout_group_cursor"):
        require(type(binding.get(name)) is int and binding[name] >= 0, f"{name} must be nonnegative integer")
    require(binding["initial_rollout_group_cursor"] ==
            binding["initial_policy_step"] * geometry["prompt_groups_per_update"],
            "initial group cursor differs from policy step")
    sampling = binding.get("sampling")
    require(isinstance(sampling, Mapping) and sampling.get("do_sample") is True,
            "production sampling must be explicit")
    require(type(sampling.get("seed")) is int and not isinstance(sampling.get("seed"), bool),
            "sampling seed must be an integer")
    signal = binding.get("signal_evidence")
    require(isinstance(signal, Mapping) and signal.get("status") == "root_admitted",
            "reward signal evidence is not root admitted")
    required_families = binding.get("required_reward_families")
    require(isinstance(required_families, list) and required_families and
            all(isinstance(name, str) and name for name in required_families) and
            len(required_families) == len(set(required_families)),
            "required reward families are missing or duplicated")
    family_groups = signal.get("groups_by_family")
    varying = signal.get("varying_groups_by_family")
    require(isinstance(family_groups, Mapping) and isinstance(varying, Mapping),
            "per-family reward signal denominators are missing")
    require(set(family_groups) == set(required_families) == set(varying),
            "reward signal families differ from binding")
    for family in required_families:
        require(type(family_groups[family]) is int and family_groups[family] > 0,
                f"reward family {family} has no groups")
        require(type(varying[family]) is int and 0 < varying[family] <= family_groups[family],
                f"reward family {family} has no admitted within-group signal")
    require(type(signal.get("parser_infrastructure_failures")) is int and
            signal["parser_infrastructure_failures"] >= 0,
            "parser infrastructure denominator is missing")
    return json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))


def _row_id(item: Mapping[str, Any]) -> str:
    value = item.get("id")
    require(isinstance(value, str) and value, "live input row ID is missing")
    return value


def build_live_groups(
    *, inputs: Sequence[Mapping[str, Any]], generation: Mapping[str, Any],
    reward_records: Sequence[Mapping[str, Any]], completion_ids_list: Sequence[Any],
    candidate_count: int, first_group: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Join live generation and PRM03 reward records in exact TRL order."""
    records = generation.get("records")
    require(isinstance(records, list), "live generation records are missing")
    require(len(inputs) == len(records) == len(reward_records) == len(completion_ids_list), "live rollout/reward/completion lengths differ")
    require(len(records) > 0 and len(records) % candidate_count == 0,
            "live rollout does not contain complete groups")
    groups: list[dict[str, Any]] = []
    prompts: dict[str, list[int]] = {}
    for local, start in enumerate(range(0, len(records), candidate_count)):
        input_group = inputs[start:start + candidate_count]
        generation_group = records[start:start + candidate_count]
        reward_group = reward_records[start:start + candidate_count]
        completion_group = completion_ids_list[start:start + candidate_count]
        row_id = _row_id(input_group[0])
        require(all(_row_id(item) == row_id for item in input_group),
                "TRL group mixes source rows")
        prompt = input_group[0].get("prompt")
        require(isinstance(prompt, Mapping) and isinstance(prompt.get("ids"), list),
                "live input prompt IDs are missing")
        prompt_ids = list(prompt["ids"])
        require(all(item.get("prompt") == prompt for item in input_group),
                "TRL group prompt envelopes differ")
        if row_id in prompts:
            require(prompts[row_id] == prompt_ids, "one row ID maps to different prompts")
        prompts[row_id] = prompt_ids
        group_index = first_group + local
        generations: list[dict[str, Any]] = []
        rewards: list[dict[str, Any]] = []
        for candidate, (gen, rew, trl_ids) in enumerate(zip(generation_group, reward_group, completion_group, strict=False)):
            ids = gen.get("generated_tokens")
            require(isinstance(ids, list) and all(type(token) is int for token in ids), "generated token IDs are missing")
            if hasattr(trl_ids, "detach"):
                trl_ids = trl_ids.detach().tolist()
            elif hasattr(trl_ids, "tolist") and not isinstance(trl_ids, (list, tuple)):
                trl_ids = trl_ids.tolist()
            require(isinstance(trl_ids, (list, tuple)) and list(trl_ids) == ids, "TRL completion IDs differ from fixed-ID generation")
            require(gen.get("prompt_tokens") == prompt_ids, "fixed-ID generation prompt differs from live input prompt")
            digest = ids_sha256(ids)
            terminal = gen.get("terminal_reason")
            require(terminal in {"eos", "length"}, "fixed-ID generation terminal reason is invalid")
            padded = gen.get("padded_after_terminal", 0)
            require(type(padded) is int and padded >= 0, "fixed-ID padded-after-terminal count is invalid")
            cap_hit = terminal == "length"
            require(type(rew.get("reward")) in (int, float) and not isinstance(rew.get("reward"), bool),
                    "PRM03 reward value is missing")
            require(rew.get("id") == row_id, "PRM03 reward row differs from generation")
            generations.append({
                "group_index": group_index, "row_id": row_id,
                "candidate_index": candidate, "prompt_ids_sha256": ids_sha256(prompt_ids),
                "generated_ids": list(ids), "generated_ids_sha256": digest,
                "cap_hit": cap_hit,
            })
            rewards.append({
                "group_index": group_index, "row_id": row_id,
                "candidate_index": candidate, "output_ids_sha256": rew.get("output_ids_sha256"),
                "reward": rew.get("reward"),
                "parser_infrastructure_failure": rew.get("parser_infrastructure_failure", False),
            })
        groups.append({"group_index": group_index, "row_id": row_id,
                       "generations": generations, "rewards": rewards})
    return groups, prompts


class ProductionRuntimeContractMixin:
    """Validate live fixed-ID generation plus PRM03 rewards before TRL loss."""

    def configure_production_contract(self, binding: Mapping[str, Any]) -> None:
        self._campaign_production_binding = validate_production_binding(binding)
        self._campaign_verified_update = None

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        binding = validate_production_binding(self._campaign_production_binding)
        manifest = getattr(self, "full_weight_optimizer_manifest", None)
        require(isinstance(manifest, Mapping), "full-weight optimizer manifest is missing")
        require(canonical_sha256(manifest) == binding["optimizer_dispatch_sha256"],
                "live optimizer dispatch differs from binding")
        return optimizer

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        binding = validate_production_binding(self._campaign_production_binding)
        reward = getattr(self, "_campaign_reward", None)
        require(reward is not None and reward.__class__.__name__ == "CampaignPRM03Reward",
                "live reward callable is not CampaignPRM03Reward")
        reward_records = getattr(reward, "last_records", None)
        require(isinstance(reward_records, list), "PRM03 reward records are missing")
        step = int(self.state.global_step)
        require(step >= binding["initial_policy_step"], "live policy step precedes admitted parent")
        geometry = binding["geometry"]
        first_group = binding["initial_rollout_group_cursor"] + (
            step - binding["initial_policy_step"]
        ) * geometry["prompt_groups_per_update"]
        groups, prompt_ids = build_live_groups(
            inputs=inputs, generation=self._campaign_last_generation,
            reward_records=reward_records, completion_ids_list=completion_ids_list,
            candidate_count=geometry["candidate_count"],
            first_group=first_group,
        )
        require(len(groups) == geometry["prompt_groups_per_update"],
                "live prompt-group count differs from binding")
        update_binding = {
            "schema": "sepalith.rl11.full-weight-update-binding.v1",
            "optimizer_updates": True,
            "objective": "trl-0.24-grpo-bnpo-group-beta0",
            "candidate_count": geometry["candidate_count"],
            "policy_step": step, "optimizer_global_step": step,
            "rollout_cursor": step, "rollout_complete": True,
            "same_process_live_policy_logprobs": True,
            "full_weight_checkpoint_kind": "full_weights",
            "sampler_boundary": "one_generation_buffer_per_optimizer_update",
            **{name: binding[name] for name in SHA_FIELDS},
        }
        validated = prepare_update(binding=update_binding, groups=groups,
                                   prompt_ids_by_row=prompt_ids, expected_first_group=first_group)
        # Flat groups remain valid; aggregate signal admission is external and
        # is bound by signal_evidence.  TRL naturally gives them zero advantage.
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        record = json.dumps(validated, sort_keys=True, separators=(",", ":"),
                            allow_nan=False) + "\n"
        with (output_dir / "verified-update-batches.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(record)
            stream.flush()
            os.fsync(stream.fileno())
        self._campaign_verified_update = validated
        return rewards

    def campaign_sampler_state(self) -> dict[str, Any]:
        state = dict(super().campaign_sampler_state())
        binding = validate_production_binding(self._campaign_production_binding)
        step = int(self.state.global_step)
        verified = self._campaign_verified_update
        require(step == binding["initial_policy_step"] or isinstance(verified, Mapping),
                "checkpoint follows no verified rollout update")
        if isinstance(verified, Mapping):
            require(verified["binding"]["policy_step"] == step - 1,
                    "checkpoint step does not follow verified policy update")
        next_group = binding["initial_rollout_group_cursor"] + (
            step - binding["initial_policy_step"]
        ) * binding["geometry"]["prompt_groups_per_update"]
        state.update({
            "rollout_buffer_state": "empty",
            "rollout_group_cursor": next_group,
            "optimizer_global_step": step,
            "policy_step": step,
            "sampled_generation_rng": "trainer_rng_state_pth_cpu_cuda_restored_before_next_rollout",
            "generation_policy_sha256": binding["generation_policy_sha256"],
            "last_update_batch_sha256": verified.get("batch_sha256") if isinstance(verified, Mapping) else None,
        })
        return state


def production_trainer_class(base_campaign_grpo: type[Any]) -> type[Any]:
    """Compose dense optimizer before the pinned campaign/TRL base class."""
    class FullWeightCampaignGRPO(
        ProductionRuntimeContractMixin, FullWeightOptimizerTrainerMixin, base_campaign_grpo
    ):
        pass
    FullWeightCampaignGRPO.__name__ = "FullWeightCampaignGRPO"
    return FullWeightCampaignGRPO


def install_production_runtime(trainer: Any, binding: Mapping[str, Any], optimizer: Mapping[str, Any]) -> dict[str, Any]:
    """Install binding/config before Trainer creates optimizer or scheduler."""
    checked = validate_production_binding(binding)
    require(getattr(trainer, "optimizer", None) is None and getattr(trainer, "lr_scheduler", None) is None,
            "optimizer/scheduler already created")
    require(isinstance(optimizer, Mapping), "optimizer config is missing")
    require(canonical_sha256(optimizer) == checked["optimizer_config_sha256"],
            "optimizer config differs from production binding")
    trainer.full_weight_optimizer_config = OptimizerConfig(**dict(optimizer))
    trainer.configure_production_contract(checked)
    return checked


def build_production_trainer(recipe: Mapping[str, Any], *, model: Any, tokenizer: Any,
                             production_binding: Mapping[str, Any], optimizer: Mapping[str, Any],
                             **runtime: Any) -> Any:
    """Invoke the accepted live PRM03 builder with the dense production class."""
    checked = validate_production_binding(production_binding)
    require(recipe.get("training_mode") == "full_weights", "recipe is not full-weight RL")
    identity = recipe.get("identity")
    require(isinstance(identity, Mapping), "recipe identity is missing")
    parent = identity.get("parent")
    data = identity.get("data")
    policy = identity.get("policy")
    require(isinstance(parent, Mapping) and isinstance(data, Mapping) and isinstance(policy, Mapping),
            "recipe parent/data/policy identity is incomplete")
    exact = {
        "source_manifest_sha256": identity.get("source", {}).get("manifest_sha256"),
        "model_manifest_sha256": parent.get("manifest_sha256"),
        "model_weights_sha256": parent.get("model_weights_sha256"),
        "tokenizer_json_sha256": parent.get("tokenizer_json_sha256"),
        "reward_buffer_manifest_sha256": data.get("reward_buffer_manifest_sha256"),
        "prompt_context_manifest_sha256": data.get("context_sha256"),
        "generation_policy_sha256": canonical_sha256(policy.get("sampling")),
        "optimizer_config_sha256": canonical_sha256(optimizer),
        "saved_precision_manifest_sha256": parent.get("saved_precision_manifest_sha256"),
    }
    for name, value in exact.items():
        require(checked[name] == value, f"production binding differs from recipe at {name}")
    named = list(model.named_parameters())
    require(len(named) == checked["expected_parameter_tensors"],
            "loaded dense tensor count differs from binding")
    require(sum(parameter.numel() for _, parameter in named if parameter.requires_grad) ==
            checked["expected_trainable_parameters"],
            "loaded trainable parameter count differs from binding")
    require(not any("lora_" in name.lower() for name, _ in named),
            "loaded model contains PEFT tensors")
    resume_from = recipe.get("resume_from")
    if resume_from is not None:
        from sepalith.training.checkpoint.campaign_checkpoint import verify_checkpoint
        resume_path = Path(resume_from)
        manifest = verify_checkpoint(
            resume_path, identity, require_full=True,
            expected_checkpoint_kind="full_weights",
        )
        validate_resume_boundary(resume_path, checked, manifest)
    from sepalith.training.rl import campaign_rl_train
    base = campaign_rl_train.campaign_grpo_trainer_class()
    trainer = campaign_rl_train.build_live_trainer(
        recipe, model=model, tokenizer=tokenizer,
        trainer_class_factory=lambda: production_trainer_class(base),
        checkpoint_kind="full_weights", production_binding=checked,
        full_weight_optimizer_config=optimizer, **runtime,
    )
    trainer._campaign_verified_resume_from = str(resume_from) if resume_from is not None else None
    return trainer


def train_production(trainer: Any) -> Any:
    """Start only the already-built trainer with its verified resume path."""
    require(hasattr(trainer, "_campaign_production_binding"),
            "trainer has no production binding")
    return trainer.train(resume_from_checkpoint=trainer._campaign_verified_resume_from)


def validate_resume_boundary(checkpoint: Path, binding: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Small-metadata guard before the dense checkpoint verifier reads payloads."""
    checked = validate_production_binding(binding)
    require(manifest.get("checkpoint_kind") == "full_weights" and manifest.get("full") is True,
            "resume checkpoint is not full_weights/full")
    files = manifest.get("files")
    require(isinstance(files, Mapping), "resume file inventory is missing")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "campaign-state.json"):
        require(name in files and isinstance(files[name], Mapping) and files[name].get("bytes", 0) > 0,
                f"resume checkpoint lacks {name}")
    state_path = Path(checkpoint) / "campaign-state.json"
    state = json.loads(state_path.read_text())
    sampler = state.get("sampler")
    require(isinstance(sampler, Mapping) and sampler.get("rollout_buffer_state") == "empty",
            "resume checkpoint is inside a rollout buffer")
    step = state.get("step")
    require(type(step) is int and not isinstance(step, bool) and step >= checked["initial_policy_step"],
            "resume step is invalid")
    expected = checked["initial_rollout_group_cursor"] + (
        step - checked["initial_policy_step"]
    ) * checked["geometry"]["prompt_groups_per_update"]
    require(sampler.get("rollout_group_cursor") == expected, "resume rollout group cursor differs")
    require(sampler.get("optimizer_global_step") == step == sampler.get("policy_step"),
            "resume optimizer/policy steps differ")
    require(sampler.get("generation_policy_sha256") == checked["generation_policy_sha256"],
            "resume generation policy differs")
    return {"step": step, "rollout_group_cursor": expected, "rng_file": "rng_state.pth",
            "rollout_buffer_state": "empty"}
