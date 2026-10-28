#!/usr/bin/env python3
"""Bound recipes for the C18 GRPO pilot on the execution environment and its smoke run (fail-closed loader).

A bound recipe pins the starting model (arm A checkpoint 803) and its bound DEV gate results, the exec
pool (training and validation task files, the pool manifest), the legacy replay pool (the C14 pool
and bundle, reward v2.1), the SFT replay rows, the SFT recipe the learning rate derives from, the
memory measurements behind the anti-forgetting choice, the in-run gate panels, the final evaluation
inputs, the outputs and a source manifest of the code. ``PRE_REGISTERED`` is the card's recipe; the
loader refuses a recipe that differs from it.

    python -m sepalith.training.grpo_exec.recipe source-manifest --output docs/training/plan-20260927/c18/bound/source-manifest.json
    python -m sepalith.training.grpo_exec.recipe bind --kind pilot --output docs/training/plan-20260927/c18/bound/grpo-exec-pilot-1-arm-a-803.json --decided-by ...
    python -m sepalith.training.grpo_exec.recipe verify --recipe <bound recipe> --repository-root .
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import os
from pathlib import Path
from typing import cast

from sepalith.training.grpo import recipe as c15
from sepalith.training.grpo_exec import rules

Json = dict[str, object]

BOUND_SCHEMA = "sepalith.c18.grpo-exec-bound.v1"
SOURCE_SCHEMA = "sepalith.c18.grpo-exec-source.v1"
DEPLOY_ROOT = c15.DEPLOY_ROOT
STATE = c15.STATE
CHECKPOINTS = c15.CHECKPOINTS
sha256 = c15.sha256
write_new = c15.write_new
check_file = c15.check_file
file_record = c15.file_record
RecipeError = c15.RecipeError
require = c15.require
obj = c15.obj

START_PATH = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-803")
START_FILES = {
    "campaign-manifest.json": "b4690e4d13d125ff1cb3c20ee197888c89b964c8afcf4d4bf4038bf4c72d8971",
    "chat_template.jinja": "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c",
    "config.json": "49ffaed87528c7fc3c1795f41a1b50a4621e577d4719e7a62ff54833d6452563",
    "generation_config.json": "0f9e07ed38da2371971f322d8ad8d2f77da32514b66798d089e55faba704844b",
    "model.safetensors": "6c1505b842a0c9ca0d8d8502c3b541dbaec4b9aa5a54de1905097a24d4c21b37",
    "tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config.json": "0888551b46656b68bdb4402b79598f49ee6bb7b7e8be5f60a89278c439fe6896",
}
SFT_GATES = STATE / "evaluation-gates" / "SFT-v2-r4-lr1e-5"
START_RESULTS = {
    "dev75": (SFT_GATES / "results/dev-dev75-step-803.20261014T031828-959466265.json",
              "762d9b5ee92dab4028a791546c05be3b5c20f26f89f551f2da5b77e75b12ecea"),
    "dev250": (SFT_GATES / "results/dev-dev250-step-803.20261014T031828-959466265.json",
               "608371cc128409812c923bb53ca9bfa2837f7ef69538b61f9c0ca20913bb654a"),
}
START_METRICS = {"cap": 0, "edit": 28, "edit_total": 104, "noedit": 1, "nofp": 9, "score250": 109, "valid": 75}
EXEC_EVAL = STATE / "exec-eval"
HOLDOUT_TASKS = (EXEC_EVAL / "EXEC-holdout-baseline/inputs/tasks.jsonl",
                 "4d9a1a36bc266e4c193257f5f22a1d4c5dc23b2305e63d4ff1ab7203d7d139be")
START_HOLDOUT_COMPLETIONS = (EXEC_EVAL / "EXEC-holdout-checkpoints/completions-arm-a-803.jsonl",
                             "a876037106cea9b84e878e458f238ea9b0aaca506996fca2fccbb0793cdef02e")
START_HOLDOUT_RECORDS = (EXEC_EVAL / "EXEC-holdout-checkpoints/records-arm-a-803.jsonl",
                         "95fa2f4e0f2899f40ba84f5f088e173f0e378de7542a4ad4327a620af32e643a")
FREEZE_SHA256 = "9e74c1644c96576693b48a5fb37cfdd5af93c7501c505943a5fc31852e0eca19"
GATE = {"commit": "c631141b0b58cbd9135a424c2cd59f1305fe634e", "fingerprint": "7f8e073fbedb4b9d"}
CPT_PANEL = ("/mnt/e/sepalith/campaign-20260915/data-work/C09-cpt-validation-2k/campaign-20260915/cpt/validation/cpt_validation-2k.jsonl",
             "efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8")
CPT_RECORDED_START_NLL = 1.003412254762526  # c09 harness, saved FP32 norms restored (exec-eval/cpt-forgetting-803-1071.json)
POOL_MANIFEST = "docs/training/plan-20260927/c18/C18-exec-pool.json"
REPLAY = Path("/mnt/e/sepalith/campaign-20260915/data-work/C18-grpo-exec-v1/sft-replay.jsonl")
SOURCE_MANIFEST = "docs/training/plan-20260927/c18/bound/source-manifest.json"
SMOKE_VAL_LIMIT = 96
SOURCE_FILES = (
    *c15.SOURCE_FILES,
    "packages/sepalith/src/sepalith/training/grpo_exec/__init__.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/cpt_probe.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/dev_generation.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/jobs.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/mix.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/night.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/recipe.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/rules.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/trainer.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/val_eval.py",
    "packages/sepalith/src/sepalith/exec_env/harness.py",
    "packages/sepalith/src/sepalith/exec_env/reward.py",
    "packages/sepalith/src/sepalith/exec_env/r_worker.R",
    "packages/sepalith/src/sepalith/exec_env/testlib.R",
    "packages/sepalith/src/sepalith/exec_env/sandbox.sh",
    "packages/sepalith/src/sepalith/exec_eval/jobs.py",
    "packages/sepalith/src/sepalith/training/eval/campaign_cpt_eval.py",
)
EVALUATOR_FILES = (
    *c15.EVALUATOR_FILES,
    "packages/sepalith/src/sepalith/training/grpo_exec/dev_generation.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/val_eval.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/cpt_probe.py",
    "packages/sepalith/src/sepalith/exec_env/reward.py",
)

# Card C18, pre-registered. The loader compares a bound recipe with this exactly.
PRE_REGISTERED: Json = {
    "card": "C18",
    "start": "arm A checkpoint 803 (user go-ahead 2026-10-28)",
    "prompts": "the C18 exec training split; 1 in 5 drawn prompts from the C14 TRAIN pool (legacy replay, reward v2.1)",
    "legacy_every": 5,
    "group_size": 8, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "dynamic_sampling": {"oversample": 1.5, "drop": "groups whose rewards are all equal", "refill": "before each rollout buffer",
                         "max_rounds": 4, "max_draw_factor": 4.0, "surplus": "discarded"},
    "loss_type": "dapo", "scale_rewards": "none", "epsilon_low": 0.2, "epsilon_high": 0.28, "mask_truncated_completions": True,
    "beta": 0.0, "reference_model": None,
    "anti_forgetting": {"kind": "sft_cross_entropy_replay", "rows_per_update": 8, "coefficient": 0.1,
                        "loss": "coefficient x token-mean NLL of the update's replay rows' target tokens, added to the DAPO "
                                "loss before the optimizer step (updates without a signal group take no step)",
                        "rows": "seeded 8,000-row sample of the v4 TRAIN rows arm A was trained on",
                        "why_not_kl": "a frozen bf16 803 reference (4,800 MiB) does not fit next to the measured GRPO peak"},
    "learning_rate_ratio_to_sft": 0.1,
    "optimizer": "the SFT recipe's aurora_mix (same family), fresh state",
    "warmup_updates": 10,
    "scheduler": "linear warmup over 10 updates (update u uses min(1, u / 10) x LR), then constant",
    "max_grad_norm": 0.0,
    "save": "full state at every update boundary that ends a rollout buffer; the deadline stop happens on such a boundary",
    "max_updates": rules.MAX_UPDATES,
    "evaluation": {"every": rules.EVAL_EVERY, "panels": "DEV75 + validation exec (greedy) + CPT 2K", "dev250_every": rules.DEV250_EVERY,
                   "rule": "R6x", "plateau": {"evaluations": rules.PLATEAU_EVALUATIONS, "gain": rules.PLATEAU_GAIN}},
    "reward": {"exec": {"policy": "sepalith.reward.exec.v1", "w_behaviour": 0.5, "w_all": 0.5, "exact_bonus": 0.0,
                        "overlong_factor": 2.0, "overlong_slack": 6, "overlong_scale": 0.5, "gate": GATE},
               "legacy": {"policy": "sepalith.reward.v2.1", "partial_credit": "gain", "partial_weight": 0.5},
               "advantage_signal": "reward", "pass_signal": "exec all-pass / legacy exact"},
    "selection": "the accepted evaluation checkpoint with the highest validation all-pass (then validation mean reward, "
                 "then the earliest)",
    "final_evaluation": {
        "set": "frozen exec hold-out (holdout-freeze-v1, 3,803 tasks from 630 functions), greedy, exec reward at c631141",
        "comparison": "selected checkpoint vs arm A 803, paired by task, function-clustered bootstrap (2,000 resamples)",
        "primary": "all-pass difference with 95% CI lower bound > 0",
        "secondary": ["mean exec reward difference", "exact-match difference", "protocol validity",
                      f"DEV250 score250 >= 803's 109 - {rules.SCORE250_MARGIN}", f"CPT 2K NLL rise <= {rules.CPT_FINAL_LIMIT} nats"],
        "adopt": "primary met, DEV250 non-inferior and CPT within the limit"},
}


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


def repository_root(recipe: Mapping[str, object], override: Path | None = None) -> Path:
    return (override or Path(str(obj(recipe["source"], "source")["repository_root"]))).resolve()


# Binding --------------------------------------------------------------------------------------------

def outputs(name: str) -> Json:
    return c15.outputs(name)


def external(path: Path, digest: str, label: str) -> Json:
    require(path.is_file() and sha256(path) == digest, f"{label} {path} differs from its pin")
    return {"path": str(path), "sha256": digest}


def bind(kind: str, local_root: Path, *, deploy_root: Path = DEPLOY_ROOT, decided_by: str, name: str | None = None) -> Json:
    """A pilot or smoke recipe from the pinned inputs under ``local_root`` (paths recorded for ``deploy_root``).

    ``name`` overrides the arm name, and with it the archive, stop file and evidence directory (a smoke run
    from a pre-merge worktree must not share outputs with the main-rooted smoke)."""
    from sepalith.training.offline import rules as offline_rules

    require(kind in ("pilot", "smoke"), "kind is pilot or smoke")
    root = local_root.resolve()
    sft = obj(json.loads((root / c15.SFT_RECIPE).read_text()), "SFT recipe")
    require(sha256(root / c15.SFT_RECIPE) == c15.SFT_RECIPE_SHA256, "SFT recipe differs from its pin")
    require(all(sha256(START_PATH / name) == digest for name, digest in START_FILES.items() if name != "model.safetensors"),
            "starting checkpoint files differ from their pins")
    results = {name: external(path, digest, f"start {name} result") for name, (path, digest) in START_RESULTS.items()}
    summaries = {name: obj(json.loads(Path(str(results[name]["path"])).read_text())["summary"], name) for name in results}
    metrics = offline_rules.offline_metrics(summaries["dev75"], summaries["dev250"])
    require(metrics == START_METRICS, f"starting model metrics differ: {metrics}")
    pool = obj(json.loads((root / POOL_MANIFEST).read_text()), "pool manifest")
    legacy = obj(json.loads((root / c15.POOL_MANIFEST).read_text()), "C14 pool manifest")
    replay_receipt = obj(json.loads(REPLAY.with_suffix(".receipt.json").read_text()), "replay receipt")
    sft_runtime = obj(sft["runtime"], "SFT runtime")
    optimizer = copy.deepcopy(obj(sft_runtime["optimizer"], "SFT optimizer"))
    ratio = cast(float, PRE_REGISTERED["learning_rate_ratio_to_sft"])
    optimizer["hidden_lr"] = round(cast(float, optimizer["hidden_lr"]) * ratio, 12)
    optimizer["side_lr"] = round(cast(float, optimizer["side_lr"]) * ratio, 12)
    sft_gate = obj(sft["dev_gate"], "SFT dev_gate")
    panels = copy.deepcopy(obj(sft["dev_panels"], "SFT dev_panels"))
    name = name or ("C18-grpo-exec-pilot-1" if kind == "pilot" else "C18-grpo-exec-smoke")
    out = outputs(name)
    steps = rules.evaluation_steps()
    train, val = obj(pool["train"], "train"), obj(pool["validation"], "validation")
    anti = cast(Json, PRE_REGISTERED["anti_forgetting"])
    recipe: Json = {
        "schema": BOUND_SCHEMA, "card": "C18", "kind": kind, "status": "bound_not_launched", "decided_by": decided_by,
        "arm": {"name": name, "purpose": ("C18 GRPO on the exec environment from arm A 803 under R6x" if kind == "pilot" else
                                          "C18 smoke: real CUDA path, a few updates, one save and resume, one gate evaluation "
                                          "(DEV75, a validation subset, CPT 2K with the 803 reference)")},
        "parent": {"candidate_id": "SFT-v2-r4-lr1e-5-803", "kind": "gated_full_weight_checkpoint", "path": str(START_PATH),
                   "step": 803, "tag": "arm-a-803", "files": dict(START_FILES)},
        "parent_reference": {"results": results, "metrics": metrics, "generation_runtime": {
            "batch_size": 32, "dtype": "bf16", "kv_cache": "dynamic", "max_new_tokens": 1024,
            "ordering": "prompt_tokens_descending_then_case_id_ascending", "pad_token_id": 1, "padding_side": "left"},
            "cpt_recorded_nll_fp32_norms": CPT_RECORDED_START_NLL,
            "exec_holdout": {"completions": external(*START_HOLDOUT_COMPLETIONS, "803 hold-out completions"),
                             "records": external(*START_HOLDOUT_RECORDS, "803 hold-out records"),
                             "all_pass": 0.3098, "mean_reward": 0.3793, "exact_match": 0.2275, "protocol_valid": 0.9727}},
        "sft_recipe": {"path": c15.SFT_RECIPE, "sha256": c15.SFT_RECIPE_SHA256, "learning_rate": sft_runtime["learning_rate"],
                       "optimizer": sft_runtime["optimizer"]},
        "pool": {"manifest": file_record(root, POOL_MANIFEST),
                 "train": {"path": train["path"], "sha256": train["sha256"], "tasks": train["tasks"], "functions": train["functions"]},
                 "val": {"path": val["path"], "sha256": val["sha256"], "tasks": val["tasks"], "functions": val["functions"]},
                 "legacy": {"manifest": file_record(root, c15.POOL_MANIFEST), "pool": legacy["pool"], "bundle": legacy["bundle"]},
                 "replay": {"path": replay_receipt["path"], "sha256": replay_receipt["sha256"], "rows": replay_receipt["rows"]}},
        "memory_fit": {"receipt": file_record(root, c15.PROBE_RECEIPT), "config": c15.PROBE_FIT,
                       "probe_process_peak_mib": 23110.4, "fit_limit_mib": 24899.4, "card_total_mib": 32579.4,
                       "c15_pilot_peak_reserved_mib": 28360.0, "c15_pilot_peak_allocated_mib": 27451.8,
                       "c15_pilot_allocator_cap_mib": 30669.0, "bf16_reference_weights_mib": 4800.0,
                       "kl_reference_fits": False,
                       "note": "probe peak + reference = 27,910 MiB > the 24,899 MiB limit (6,144 desktop + 1,536 margin); C15's "
                               "measured peak allocated + reference = 32,252 MiB > the 30,669 MiB allocator cap"},
        "pre_registered": copy.deepcopy(PRE_REGISTERED),
        "runtime": {
            "group_size": 8, "groups_per_update": 4, "rows_per_update": 32, "updates_per_buffer": 2,
            "rollout_rows_per_call": 16, "optimizer_placement": "resident", "micro_batch": 1,
            "seq_chunk": 512, "vocab_chunk": 16384, "completion_max_tokens": 1024, "prompt_max_tokens": 3072,
            "eog_ids": [1, 130073], "pad_id": 1, "bos_id": 0, "dtype": "bfloat16", "attn_implementation": "sdpa",
            "gradient_checkpointing": True, "device": "cuda:0", "allocator_safety_mib": 256,
            "learning_rate": optimizer["hidden_lr"], "optimizer": optimizer, "optimizer_dispatch_sha256": c15.OPTIMIZER_DISPATCH_SHA256,
            "seed": 3407, "legacy_every": PRE_REGISTERED["legacy_every"],
            "replay_rows_per_update": anti["rows_per_update"], "replay_coefficient": anti["coefficient"],
            "reward": {"workers": 8, "air": c15.AIR, "rscript": c15.RSCRIPT, "exec_workers": 8, "exec_nice": 10},
            "buffer_seconds_prior": 240.0, "save_seconds_prior": 300.0, "deadline_margin_seconds": 90.0,
            "min_free_disk_bytes": 150 * 10**9, "checkpoint_bytes_estimate": 17_700_000_000,
        },
        "dev_gate": {
            "rule": "R6x", "evaluation_steps": steps, "mandatory_steps": steps,
            "panels_by_step": "DEV75 + validation exec + CPT 2K every 50 updates, DEV250 too every 150",
            "evidence_directory": out["evidence_directory"], "generation_batching": sft_gate["generation_batching"],
            "generation_max_new_tokens": sft_gate["generation_max_new_tokens"],
            "panel": {"path": obj(panels["dev75"], "dev75")["path"], "sha256": obj(panels["dev75"], "dev75")["sha256"]},
            "panels": panels,
            "val": {"tasks": {"path": val["path"], "sha256": val["sha256"]}, "n_tasks": val["tasks"], "batch_size": 32, "cap": 1024,
                    "decoding": "greedy", "limit": SMOKE_VAL_LIMIT if kind == "smoke" else None},
            "cpt": {"panel": {"path": CPT_PANEL[0], "sha256": CPT_PANEL[1]}, "rows": 499, "max_sequence_tokens": 2048,
                    "load": "plain bf16 as the trainer and the DEV evaluator (no FP32 norm restore), bf16 autocast"},
            "override_file_pattern": f"{out['evidence_directory']}/override-step-<step>.json",
        },
        "final_evaluation": {"holdout_tasks": external(*HOLDOUT_TASKS, "hold-out tasks"), "freeze_manifest_sha256": FREEZE_SHA256,
                             "start_completions": external(*START_HOLDOUT_COMPLETIONS, "803 hold-out completions"),
                             "job": "sepalith.exec_eval.jobs run --kind holdout, models [arm-a-803, c18-selected]"},
        "outputs": {"archive": out["archive"], "graceful_stop": out["graceful_stop"], "trainer": f"{out['archive']}/runtime"},
        "smoke": {"stops": [4, 6], "eval_step": 4, "eval_panels": ["dev75", "val", "cpt"]} if kind == "smoke" else None,
        "source": {"repository_root": str(deploy_root), "manifest_path": SOURCE_MANIFEST,
                   "manifest_sha256": sha256(root / SOURCE_MANIFEST)},
        "expected_model": {"dtype": "bfloat16", "parameter_tensors": 381, "parameters": 2516756480, "peft": False},
        "seed": 3407, "final_set_access": False, "holdout_in_training": False,
    }
    validate(recipe)
    return recipe


# Loading ---------------------------------------------------------------------------------------------

def validate(recipe: Mapping[str, object]) -> None:
    """Structural checks that need no files."""
    require(recipe.get("schema") == BOUND_SCHEMA and recipe.get("card") == "C18", "not a C18 bound recipe")
    require(recipe.get("pre_registered") == PRE_REGISTERED, "pre-registered C18 values differ")
    require(recipe.get("final_set_access") is False and recipe.get("holdout_in_training") is False,
            "final-set access and hold-out training must be false")
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
    anti = cast(Json, PRE_REGISTERED["anti_forgetting"])
    require(runtime["legacy_every"] == PRE_REGISTERED["legacy_every"] and runtime["replay_rows_per_update"] == anti["rows_per_update"]
            and runtime["replay_coefficient"] == anti["coefficient"], "runtime mixing or replay differs from the pre-registration")
    sft = obj(recipe.get("sft_recipe"), "sft_recipe")
    optimizer = obj(runtime["optimizer"], "optimizer")
    base = obj(sft["optimizer"], "SFT optimizer")
    ratio = cast(float, PRE_REGISTERED["learning_rate_ratio_to_sft"])
    for key in ("hidden_lr", "side_lr"):
        require(abs(cast(float, optimizer[key]) - cast(float, base[key]) * ratio) < 1e-15, f"{key} is not a tenth of the SFT's")
    require({k: v for k, v in optimizer.items() if k not in ("hidden_lr", "side_lr")}
            == {k: v for k, v in base.items() if k not in ("hidden_lr", "side_lr")}, "optimizer family settings differ from the SFT's")
    gate = obj(recipe.get("dev_gate"), "dev_gate")
    require(gate.get("evaluation_steps") == rules.evaluation_steps() == gate.get("mandatory_steps"), "evaluation steps differ from R6x")
    pool = obj(recipe.get("pool"), "pool")
    require(obj(gate.get("val"), "val")["tasks"] == {"path": obj(pool["val"], "val")["path"], "sha256": obj(pool["val"], "val")["sha256"]},
            "the validation gate must read the pool's validation split")
    require(obj(pool["train"], "train")["sha256"] != obj(pool["val"], "val")["sha256"], "train and validation files are the same")
    smoke = recipe.get("smoke")
    if recipe.get("kind") == "smoke":
        stops = cast(list[int], obj(smoke, "smoke")["stops"])
        require(len(stops) == 2 and 0 < stops[0] < stops[1] < rules.EVAL_EVERY and all(s % per_buffer == 0 for s in stops),
                "smoke stops must be buffer boundaries before the first evaluation")
    else:
        require(recipe.get("kind") == "pilot" and smoke is None and obj(gate["val"], "val").get("limit") is None,
                "kind must be pilot (without smoke or a validation limit) or smoke")


def load(path: Path, *, root_override: Path | None = None, check_source: bool = True, check_data: bool = True) -> Json:
    """Load and verify a bound recipe: structure, pinned files, the source manifest and the data hashes."""
    recipe = obj(json.loads(Path(path).read_text()), "recipe")
    validate(recipe)
    root = repository_root(recipe, root_override)
    check_file(root, obj(recipe["sft_recipe"], "sft_recipe"), "SFT recipe")
    if check_source:
        verify_source(root, obj(recipe["source"], "source"))
    if check_data:
        pool = obj(recipe["pool"], "pool")
        check_file(root, pool["manifest"], "pool manifest")
        for key in ("train", "val", "replay"):
            check_file(root, pool[key], f"{key} file")
        legacy = obj(pool["legacy"], "legacy")
        check_file(root, legacy["manifest"], "C14 pool manifest")
        check_file(root, legacy["pool"], "C14 pool")
        check_file(root, legacy["bundle"], "C14 bundle")
        check_file(root, obj(recipe["memory_fit"], "memory_fit")["receipt"], "memory probe receipt")
        check_file(root, obj(obj(recipe["dev_gate"], "dev_gate")["cpt"], "cpt")["panel"], "CPT 2K panel")
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
    binder.add_argument("--name", help="arm name (default C18-grpo-exec-pilot-1 / C18-grpo-exec-smoke)")
    verify = sub.add_parser("verify")
    verify.add_argument("--recipe", type=Path, required=True)
    verify.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "source-manifest":
        write_new(args.output, build_source_manifest(args.local_root.resolve()))
    elif args.command == "bind":
        write_new(args.output, bind(args.kind, args.local_root, deploy_root=args.deploy_root, decided_by=args.decided_by,
                                       name=args.name))
    else:
        recipe = load(args.recipe, root_override=args.repository_root)
        print(json.dumps({"status": "verified", "arm": obj(recipe["arm"], "arm")["name"], "sha256": sha256(args.recipe)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
