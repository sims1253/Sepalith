#!/usr/bin/env python3
"""Bound recipes for the C15 GRPO pilot and its smoke run, and the fail-closed loader.

A bound recipe pins everything the night jobs use: the starting model (arm A
checkpoint 536, since no offline step was adopted) and its bound gate result,
the C14 pool and bundle, the SFT recipe whose learning rate and optimizer the
pilot derives from, the C14 memory-probe fit, the DEV panels and generation
settings of the gates, the outputs, and a source manifest of the code. The
pre-registered C15 values live in ``PRE_REGISTERED``; the loader refuses a
recipe that differs from them.

Paths inside the repository are stored relative to ``source.repository_root``
(the deploy checkout); everything else is absolute.

    python -m sepalith.training.grpo.recipe source-manifest --output docs/training/plan-20260927/c15/bound/source-manifest.json
    python -m sepalith.training.grpo.recipe bind --kind pilot --output docs/training/plan-20260927/c15/bound/grpo-pilot-1-arm-a-536.json
    python -m sepalith.training.grpo.recipe verify --recipe <bound recipe> --repository-root .
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import cast

from sepalith.training.grpo import rules

Json = dict[str, object]

BOUND_SCHEMA = "sepalith.c15.grpo-bound.v1"
SOURCE_SCHEMA = "sepalith.c15.grpo-source.v1"
DEPLOY_ROOT = Path("/home/m0hawk/sandbox/Sepalith")
STATE = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915")
CHECKPOINTS = Path("/mnt/e/sepalith/campaign-20260915/checkpoints")
SFT_RECIPE = "docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json"
SFT_RECIPE_SHA256 = "de4bcb0a6da1f3481874aeb82a19d80eae0ec6389b87ba18d1e4a9ca25bf2a4e"
# The starting model's identity and its gate-536 result, as bound for C13 (status/C13.json).
START_RECIPE = "docs/training/plan-20260927/c13/bound/dpo-2-arm-a-536-t12.json"
POOL_MANIFEST = "docs/training/plan-20260927/c14/C14-grpo-pool-arm-a-536.json"
PROBE_RECEIPT = "docs/training/plan-20260927/c14/C14-memory-probe-20261022T082309Z.json"
PROBE_FIT = "g8-r16-resident"
SOURCE_MANIFEST = "docs/training/plan-20260927/c15/bound/source-manifest.json"
# aurora_mix dispatch of the plain-transformers bf16 load (the C14 probe's path): every tensor bf16, layer
# norms included. The SFT and C13 trainers load with unsloth, which keeps the 85 norm weights in fp32
# (expected-aurora-mix-manifest.json, 4b2f51cc...); the names, shapes and dispatch are the same.
OPTIMIZER_DISPATCH_SHA256 = "162a62e5e3c07030ed401dcb6e0f28b37c931d6af45150386989cb739e1afbf4"
AIR = "/home/m0hawk/.local/bin/air"
RSCRIPT = "/usr/bin/Rscript"
SOURCE_FILES = (
    "packages/sepalith/src/sepalith/campaign_protocol.py",
    "packages/sepalith/src/sepalith/training/grpo/__init__.py",
    "packages/sepalith/src/sepalith/training/grpo/algorithm.py",
    "packages/sepalith/src/sepalith/training/grpo/dev_generation.py",
    "packages/sepalith/src/sepalith/training/grpo/loss.py",
    "packages/sepalith/src/sepalith/training/grpo/night.py",
    "packages/sepalith/src/sepalith/training/grpo/pool.py",
    "packages/sepalith/src/sepalith/training/grpo/recipe.py",
    "packages/sepalith/src/sepalith/training/grpo/rules.py",
    "packages/sepalith/src/sepalith/training/grpo/state.py",
    "packages/sepalith/src/sepalith/training/grpo/trainer.py",
    "packages/sepalith/src/sepalith/training/rl/grpo_memory_probe.py",
    "packages/sepalith/src/sepalith/training/rl/reward_v21.py",
    "packages/sepalith/src/sepalith/training/rl/air_canonical.py",
    "packages/sepalith/src/sepalith/training/rl/campaign_r_parse_probe.py",
    "packages/sepalith/src/sepalith/training/rl/reward_parse_only.R",
    "packages/sepalith/src/sepalith/training/rl/campaign_reward_v2.py",
    "packages/sepalith/src/sepalith/training/rl/campaign_rl_profile.py",
    "packages/sepalith/src/sepalith/training/rl/campaign_rl_train.py",
    "packages/sepalith/src/sepalith/training/rl/length_policy.py",
    "packages/sepalith/src/sepalith/training/optim/full_weight_optimizer.py",
    "packages/sepalith/src/sepalith/training/checkpoint/campaign_checkpoint.py",
    "packages/sepalith/src/sepalith/training/eval/campaign_eval.py",
    "packages/sepalith/src/sepalith/training/eval/milestone_gate.py",
    "packages/sepalith/src/sepalith/training/contracts/campaign_tokenizer_contract.py",
    "packages/sepalith/src/sepalith/training/contracts/trainer_tokenizer_alignment.py",
    "packages/sepalith/src/sepalith/training/guards/cuda_host_guard.py",
    "packages/sepalith/src/sepalith/training/sft/night_arm.py",
)
EVALUATOR_FILES = (
    "packages/sepalith/src/sepalith/training/grpo/dev_generation.py",
    "packages/sepalith/src/sepalith/training/eval/campaign_eval.py",
    "packages/sepalith/src/sepalith/training/eval/milestone_gate.py",
)

# Card C15, pre-registered. The loader compares a bound recipe with this exactly.
PRE_REGISTERED: Json = {
    "card": "C15",
    "start": "latest adopted offline model, or the SFT model if none was adopted: arm A checkpoint 536",
    "prompts": "the C14 pool",
    "group_size": 8,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "dynamic_sampling": {"oversample": 1.5, "drop": "groups whose rewards are all equal", "refill": "before each rollout buffer",
                         "max_rounds": 4, "max_draw_factor": 4.0, "surplus": "discarded"},
    "loss_type": "dapo",
    "scale_rewards": "none",
    "epsilon_low": 0.2,
    "epsilon_high": 0.28,
    "mask_truncated_completions": True,
    "beta": 0.0,
    "reference_model": None,
    "learning_rate_ratio_to_sft": 0.1,
    "optimizer": "the SFT recipe's aurora_mix (same family), fresh state",
    "warmup_updates": 10,
    "scheduler": "linear warmup over 10 updates (update u uses min(1, u / 10) x LR), then constant",
    "max_grad_norm": 0.0,
    "save": "full state at every update boundary that ends a rollout buffer; the deadline stop happens on such a boundary",
    "max_updates": rules.MAX_UPDATES,
    "evaluation": {"dev75_every": rules.DEV75_EVERY, "dev250_every": rules.DEV250_EVERY, "rule": "R6"},
    "reward": {"policy": "sepalith.reward.v2.1", "partial_credit": "gain", "partial_weight": 0.5,
               "advantage_signal": "reward", "pass_signal": "exact"},
}


class RecipeError(ValueError):
    """A bound recipe is malformed or differs from its pins."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise RecipeError(message)


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: object) -> None:
    require(not path.exists(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def obj(value: object, label: str) -> Json:
    require(isinstance(value, dict), f"{label} must be an object")
    return cast(Json, value)


def file_record(root: Path, relative: str) -> Json:
    path = root / relative
    require(path.is_file(), f"missing {path}")
    return {"path": relative, "sha256": sha256(path)}


def check_file(root: Path, record: object, label: str) -> Path:
    entry = obj(record, label)
    raw = Path(str(entry.get("path")))
    path = raw if raw.is_absolute() else root / raw
    require(path.is_file() and sha256(path) == entry.get("sha256"), f"{label} {path} differs from its pinned hash")
    return path


# Source manifest -----------------------------------------------------------------------------------

def build_source_manifest(root: Path) -> Json:
    files = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256(root / name)} for name in SOURCE_FILES]
    return {"schema": SOURCE_SCHEMA, "root": "repository", "files": files, "evaluator_files": list(EVALUATOR_FILES)}


def verify_source(root: Path, record: Mapping[str, object]) -> Json:
    path = check_file(root, {"path": record.get("manifest_path"), "sha256": record.get("manifest_sha256")}, "source manifest")
    manifest = obj(json.loads(path.read_text()), "source manifest")
    require(manifest.get("schema") == SOURCE_SCHEMA, "source manifest schema differs")
    files = cast(list[Json], manifest.get("files"))
    require([item.get("path") for item in files] == list(SOURCE_FILES), "source manifest file list differs from the code's")
    for item in files:
        target = root / str(item["path"])
        require(target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"],
                f"source differs from the bound manifest: {item['path']}")
    return manifest


def evaluator_hashes(root: Path) -> dict[str, str]:
    return {name: sha256(root / name) for name in EVALUATOR_FILES}


# Binding --------------------------------------------------------------------------------------------

def outputs(name: str) -> Json:
    return {"archive": str(CHECKPOINTS / name), "graceful_stop": str(STATE / "control" / f"{name}-save-stop.json"),
            "evidence_directory": str(STATE / "evaluation-gates" / name)}


def bind(kind: str, local_root: Path, *, deploy_root: Path = DEPLOY_ROOT, decided_by: str) -> Json:
    """A pilot or smoke recipe from the pinned inputs under ``local_root`` (paths recorded for ``deploy_root``)."""
    require(kind in ("pilot", "smoke"), "kind is pilot or smoke")
    root = local_root.resolve()
    sft = obj(json.loads((root / SFT_RECIPE).read_text()), "SFT recipe")
    require(sha256(root / SFT_RECIPE) == SFT_RECIPE_SHA256, "SFT recipe differs from its pin")
    start = obj(json.loads((root / START_RECIPE).read_text()), "start recipe")
    parent = copy.deepcopy(obj(start["parent"], "parent"))
    parent["tag"] = "arm-a-536"
    reference = copy.deepcopy(obj(start["parent_reference"], "parent_reference"))
    pool_manifest = obj(json.loads((root / POOL_MANIFEST).read_text()), "pool manifest")
    probe = obj(json.loads((root / PROBE_RECEIPT).read_text()), "probe receipt")
    fits = obj(probe["ladders"], "ladders")
    require(obj(fits["g8"], "g8")["first_fit"] == PROBE_FIT, "the probe's G = 8 fit differs")
    sft_runtime = obj(sft["runtime"], "SFT runtime")
    optimizer = copy.deepcopy(obj(sft_runtime["optimizer"], "SFT optimizer"))
    ratio = cast(float, PRE_REGISTERED["learning_rate_ratio_to_sft"])
    optimizer["hidden_lr"] = round(cast(float, optimizer["hidden_lr"]) * ratio, 12)
    optimizer["side_lr"] = round(cast(float, optimizer["side_lr"]) * ratio, 12)
    gate = obj(start["dev_gate"], "start dev_gate")
    name = "C15-grpo-pilot-1" if kind == "pilot" else "C15-grpo-smoke"
    out = outputs(name)
    steps = rules.evaluation_steps()
    recipe: Json = {
        "schema": BOUND_SCHEMA, "card": "C15", "kind": kind, "status": "bound_not_launched", "decided_by": decided_by,
        "arm": {"name": name, "purpose": ("C15 GRPO pilot from arm A 536 on the C14 pool under R6" if kind == "pilot" else
                                          "C15 smoke: real CUDA path, a few updates, one save and resume, one DEV75 evaluation")},
        "r4_override": {"rule": "R4 step 3 (M >= 15%)", "latest_census_M": {"t1.0": 0.125083, "t1.2": 0.147474},
                        "decision": "user decision 2026-10-23 ~11:45: start online GRPO anyway after both offline "
                                    "iterations (RFT and DPO at T = 1.0 and T = 1.2) were discarded at every gate",
                        "unchanged": ["C15 recipe", "R6 stop and continue rules", "R8 (at most one model upload per night)"]},
        "parent": parent, "parent_reference": reference,
        "sft_recipe": {"path": SFT_RECIPE, "sha256": SFT_RECIPE_SHA256, "learning_rate": sft_runtime["learning_rate"],
                       "optimizer": sft_runtime["optimizer"]},
        "pool": {"manifest": file_record(root, POOL_MANIFEST), "pool": pool_manifest["pool"], "bundle": pool_manifest["bundle"],
                 "counts": {key: cast(Json, pool_manifest["counts"])[key] for key in ("band", "all_correct_sampled", "pool", "M")}},
        "memory_fit": {"receipt": file_record(root, PROBE_RECEIPT), "config": PROBE_FIT,
                       "peak_mib": 23110, "headroom_at_desktop_budget_mib": 3325, "update_seconds_cap": 155.2,
                       "update_seconds_natural": 32.2},
        "pre_registered": copy.deepcopy(PRE_REGISTERED),
        "runtime": {
            "group_size": 8, "groups_per_update": 4, "rows_per_update": 32, "updates_per_buffer": 2,
            "rollout_rows_per_call": 16, "optimizer_placement": "resident", "micro_batch": 1,
            "seq_chunk": 512, "vocab_chunk": 16384, "completion_max_tokens": 1024, "prompt_max_tokens": 3072,
            "eog_ids": [1, 130073], "pad_id": 1, "bos_id": 0, "dtype": "bfloat16", "attn_implementation": "sdpa",
            "gradient_checkpointing": True, "device": "cuda:0", "allocator_safety_mib": 256,
            "learning_rate": optimizer["hidden_lr"], "optimizer": optimizer, "optimizer_dispatch_sha256": OPTIMIZER_DISPATCH_SHA256,
            "seed": 3407,
            "reward": {"workers": 8, "air": AIR, "rscript": RSCRIPT},
            "buffer_seconds_prior": 180.0, "save_seconds_prior": 360.0, "deadline_margin_seconds": 90.0,
            "min_free_disk_bytes": 150 * 10**9, "checkpoint_bytes_estimate": 17_700_000_000,
        },
        "dev_gate": {
            "rule": "R6", "evaluation_steps": steps, "mandatory_steps": steps,
            "panels_by_step": "dev75 every 50 updates, dev75 + dev250 every 150 updates",
            "evidence_directory": out["evidence_directory"], "generation_batching": gate["generation_batching"],
            "generation_max_new_tokens": gate["generation_max_new_tokens"], "panel": gate["panel"], "panels": gate["panels"],
            "override_file_pattern": f"{out['evidence_directory']}/override-step-<step>.json",
        },
        "outputs": {"archive": out["archive"], "graceful_stop": out["graceful_stop"], "trainer": f"{out['archive']}/runtime"},
        "smoke": {"stops": [4, 6], "eval_step": 4, "eval_panels": ["dev75"]} if kind == "smoke" else None,
        "source": {"repository_root": str(deploy_root), "manifest_path": SOURCE_MANIFEST,
                   "manifest_sha256": sha256(root / SOURCE_MANIFEST)},
        "expected_model": {"dtype": "bfloat16", "parameter_tensors": 381, "parameters": 2516756480, "peft": False},
        "seed": 3407, "final_set_access": False,
    }
    validate(recipe)
    return recipe


# Loading ---------------------------------------------------------------------------------------------

def validate(recipe: Mapping[str, object]) -> None:
    """Structural checks that need no files."""
    require(recipe.get("schema") == BOUND_SCHEMA and recipe.get("card") == "C15", "not a C15 bound recipe")
    require(recipe.get("pre_registered") == PRE_REGISTERED, "pre-registered C15 values differ")
    require(recipe.get("final_set_access") is False, "final-set access must be false")
    runtime = obj(recipe.get("runtime"), "runtime")
    group = cast(int, runtime["group_size"])
    require(group == PRE_REGISTERED["group_size"], "group size differs")
    require(runtime["rows_per_update"] == group * cast(int, runtime["groups_per_update"]), "rows per update differ")
    per_buffer = cast(int, runtime["updates_per_buffer"])
    require(per_buffer >= 1 and all(step % per_buffer == 0 for step in rules.evaluation_steps()),
            "every evaluation step must end a rollout buffer")
    per_call = cast(int, runtime["rollout_rows_per_call"])
    require(per_call % group == 0 and per_call <= 16, "rollout calls hold whole groups and at most the fitted 16 rows")
    require(runtime["micro_batch"] == 1 and runtime["optimizer_placement"] == "resident", "the C14 fit is micro-batch 1, resident")
    require(runtime["device"] == "cuda:0", "the device must carry an explicit index")
    sft = obj(recipe.get("sft_recipe"), "sft_recipe")
    optimizer = obj(runtime["optimizer"], "optimizer")
    base = obj(sft["optimizer"], "SFT optimizer")
    ratio = cast(float, PRE_REGISTERED["learning_rate_ratio_to_sft"])
    for key in ("hidden_lr", "side_lr"):
        require(abs(cast(float, optimizer[key]) - cast(float, base[key]) * ratio) < 1e-15, f"{key} is not a tenth of the SFT's")
    require({k: v for k, v in optimizer.items() if k not in ("hidden_lr", "side_lr")}
            == {k: v for k, v in base.items() if k not in ("hidden_lr", "side_lr")}, "optimizer family settings differ from the SFT's")
    gate = obj(recipe.get("dev_gate"), "dev_gate")
    require(gate.get("evaluation_steps") == rules.evaluation_steps() == gate.get("mandatory_steps"), "evaluation steps differ from R6")
    smoke = recipe.get("smoke")
    if recipe.get("kind") == "smoke":
        stops = cast(list[int], obj(smoke, "smoke")["stops"])
        require(len(stops) == 2 and 0 < stops[0] < stops[1] < rules.DEV75_EVERY and all(s % per_buffer == 0 for s in stops),
                "smoke stops must be buffer boundaries before the first evaluation")
    else:
        require(recipe.get("kind") == "pilot" and smoke is None, "kind must be pilot (without smoke) or smoke")


def repository_root(recipe: Mapping[str, object], override: Path | None = None) -> Path:
    return (override or Path(str(obj(recipe["source"], "source")["repository_root"]))).resolve()


def load(path: Path, *, root_override: Path | None = None, check_source: bool = True, check_data: bool = True) -> Json:
    """Load and verify a bound recipe: structure, small pinned files, the source manifest and the data hashes."""
    recipe = obj(json.loads(Path(path).read_text()), "recipe")
    validate(recipe)
    root = repository_root(recipe, root_override)
    sft = obj(recipe["sft_recipe"], "sft_recipe")
    check_file(root, sft, "SFT recipe")
    if check_source:
        verify_source(root, obj(recipe["source"], "source"))
    if check_data:
        pool = obj(recipe["pool"], "pool")
        check_file(root, pool["manifest"], "pool manifest")
        check_file(root, pool["pool"], "pool")
        check_file(root, pool["bundle"], "bundle")
        check_file(root, obj(recipe["memory_fit"], "memory_fit")["receipt"], "memory probe receipt")
        reward = obj(obj(recipe["runtime"], "runtime")["reward"], "reward")
        for key in ("air", "rscript"):
            require(os.access(str(reward[key]), os.X_OK), f"{key} is not executable at {reward[key]}")
    return recipe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source-manifest")
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[6])
    binder = sub.add_parser("bind")
    binder.add_argument("--kind", choices=("pilot", "smoke"), required=True)
    binder.add_argument("--output", type=Path, required=True)
    binder.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[6])
    binder.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT)
    binder.add_argument("--decided-by", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--recipe", type=Path, required=True)
    verify.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "source-manifest":
        write_new(args.output, build_source_manifest(args.local_root.resolve()))
    elif args.command == "bind":
        write_new(args.output, bind(args.kind, args.local_root, deploy_root=args.deploy_root, decided_by=args.decided_by))
    else:
        recipe = load(args.recipe, root_override=args.repository_root)
        print(json.dumps({"status": "verified", "arm": obj(recipe["arm"], "arm")["name"], "sha256": sha256(args.recipe)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
