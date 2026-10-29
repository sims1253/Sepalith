#!/usr/bin/env python3
"""Card C18b: C18 with CPT-text replay against the forgetting C18 measured (recipe module, data, entry hooks).

C18 stopped for the user at update 100 on the CPT 2K rise: +0.0117 at update 50 and +0.0238 at 100,
spread over 485 of 499 rows and all 50 packages. Over the same updates, the SFT replay NLL stayed flat or
fell (0.215 at updates 1-10, 0.176 at 76-100). The SFT term held its own distribution, but it did not
protect raw R code. SFT itself had moved the CPT NLL +0.052 from the CPT parent to arm A 536. C18b keeps
every C18 setting and adds one term:

    loss += 0.2 x token-mean NLL of 4 CPT training chunks (<= 2,048 tokens) per optimizer update

The chunks come from a seeded 8,000-row sample of the CPT training union (``cpt_train`` partition), with
hold-out-slice packages and the CPT 2K validation packages excluded. The probe stays on held-out
packages, so the gate measures generalisation, not memorisation.

This module is a drop-in recipe module for the C18 code, which is not edited, so the C18 recipes keep
verifying. ``install()`` points the C18 driver, trainer and evaluators at this module and at the C18b
entry modules. Each C18b entry module (``c18b_night``, ``c18b_trainer``, ``c18b_val_eval``,
``c18b_cpt_probe``, ``c18b_dev_generation``) calls it before the C18 ``main``.

    python -m sepalith.training.grpo_exec.c18b cpt-replay --out <DATA>/cpt-replay.jsonl
    python -m sepalith.training.grpo_exec.c18b source-manifest --output docs/training/plan-20260927/c18b/bound/source-manifest.json
    python -m sepalith.training.grpo_exec.c18b bind --kind pilot --output ... --decided-by ...
    python -m sepalith.training.grpo_exec.c18b verify --recipe <bound recipe> [--repository-root .]
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import json
import os
from pathlib import Path
import random
import sys
from typing import cast

from sepalith.training.grpo_exec import recipe as c18
from sepalith.training.grpo_exec.recipe import (  # noqa: F401 - the recipe-module interface the C18 code uses
    CHECKPOINTS,
    DEPLOY_ROOT,
    STATE,
    RecipeError,
    check_file,
    file_record,
    obj,
    repository_root,
    require,
    sha256,
    write_new,
)

Json = dict[str, object]

BOUND_SCHEMA = "sepalith.c18b.grpo-exec-bound.v1"
SOURCE_SCHEMA = "sepalith.c18b.grpo-exec-source.v1"
SOURCE_MANIFEST = "docs/training/plan-20260927/c18b/bound/source-manifest.json"
CPT_TRAIN = (Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1/cpt_train.jsonl"),
             "3fef51d311a492a62bea8dbe0694725a9ee7869e31f998d55a64e3b35cf89227", 332455)
CPT_REPLAY = Path("/mnt/e/sepalith/campaign-20260915/data-work/C18-grpo-exec-v1/cpt-replay.jsonl")
CPT_REPLAY_SCHEMA = "sepalith.c18b.cpt-replay-row.v1"
CPT_MAX_TOKENS = 2048
CPT_SEED_OFFSET = 101
SOURCE_FILES = (
    *c18.SOURCE_FILES,
    "packages/sepalith/src/sepalith/training/grpo_exec/c18b.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/c18b_trainer.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/c18b_night.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/c18b_val_eval.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/c18b_cpt_probe.py",
    "packages/sepalith/src/sepalith/training/grpo_exec/c18b_dev_generation.py",
)
EVALUATOR_FILES = (*c18.EVALUATOR_FILES, "packages/sepalith/src/sepalith/training/grpo_exec/c18b.py")
ENTRY = {"trainer": "sepalith.training.grpo_exec.c18b_trainer", "dev": "sepalith.training.grpo_exec.c18b_dev_generation",
         "val": "sepalith.training.grpo_exec.c18b_val_eval", "cpt": "sepalith.training.grpo_exec.c18b_cpt_probe"}

CPT_REPLAY_SETTINGS: Json = {
    "rows_per_update": 4, "coefficient": 0.2, "max_tokens": CPT_MAX_TOKENS,
    "loss": "coefficient x token-mean NLL of the update's CPT chunks' supervised tokens, added with the SFT replay term "
            "before the optimizer step",
    "rows": "seeded 8,000-row sample of the CPT training union (cpt_train), hold-out-slice and CPT 2K validation packages "
            "excluded"}
PRE_REGISTERED: Json = copy.deepcopy(c18.PRE_REGISTERED)
PRE_REGISTERED.update({
    "card": "C18b",
    "start": "arm A checkpoint 803 (fresh start; C18 checkpoint 100 is the C18 result under its own final evaluation)",
    "anti_forgetting": {**cast(Json, c18.PRE_REGISTERED["anti_forgetting"]), "kind": "sft_and_cpt_cross_entropy_replay",
                        "cpt": CPT_REPLAY_SETTINGS,
                        "why_cpt": "C18 forgot CPT text uniformly (485/499 rows, 50/50 packages, +0.012 per 50 updates) while "
                                   "its SFT replay NLL did not rise; only CPT-text replay targets that distribution"},
    "comparison_with_c18": "descriptive: the CPT rise per 50 updates and the validation all-pass at matched updates; "
                           "the expected CPT rise at update 50 is <= 0.006 (C18: 0.0117)",
})


# Source manifest -------------------------------------------------------------------------------------

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


# CPT replay rows ---------------------------------------------------------------------------------------

def cpt_row(row: Mapping[str, object]) -> Json | None:
    """``{prompt_ids, target_ids}``: the unsupervised head (BOS, overlap) as prompt, the supervised span as target."""
    ids, labels = cast(list[int], row["input_ids"]), cast(list[int], row["labels"])
    if not ids or ids[0] != 0 or len(ids) != len(labels) or len(ids) > CPT_MAX_TOKENS:
        return None
    start = next((i for i, label in enumerate(labels) if label != -100), None)
    if start is None:
        return None
    end = len(labels)
    while end > start and labels[end - 1] == -100:
        end -= 1
    if any(labels[i] != ids[i] for i in range(start, end)) or end - start < 16:
        return None
    return {"prompt_ids": ids[:start], "target_ids": ids[start:end]}


def cpt_replay_rows(path: Path, expected_sha: str, total: int, count: int, excluded_packages: set[str],
                    seed: int = 3407) -> tuple[list[Json], Json]:
    """A seeded sample: candidate line numbers first (1.5x), then the first ``count`` eligible ones in line order."""
    require(sha256(path) == expected_sha, f"{path} differs from its pin")
    candidates = set(random.Random(f"c18b-cpt-replay:{seed}").sample(range(total), min(total, int(count * 1.5))))
    rows: list[Json] = []
    reasons: Counter = Counter()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream):
            if number not in candidates:
                continue
            row = json.loads(line)
            if row.get("cpt_partition") != "cpt_train":
                reasons["not_cpt_train"] += 1
                continue
            if str(row.get("package")) in excluded_packages:
                reasons["excluded_package"] += 1
                continue
            value = cpt_row(row)
            if value is None:
                reasons["shape"] += 1
                continue
            rows.append({"schema": CPT_REPLAY_SCHEMA, "id": str(row["row_id"]), "package": str(row["package"]), **value})
    require(len(rows) >= count, f"only {len(rows)} eligible CPT rows for {count}")
    rows = rows[:count]
    tokens = [len(cast(list, r["target_ids"])) for r in rows]
    return rows, {"source": str(path), "source_sha256": expected_sha, "lines": total, "candidates": len(candidates),
                  "sampled": len(rows), "seed": seed, "skipped": dict(reasons), "packages": len({r["package"] for r in rows}),
                  "target_tokens_mean": round(sum(tokens) / len(tokens), 1), "excluded_packages": len(excluded_packages)}


def build_cpt_replay(out: Path, count: int) -> Json:
    from sepalith.training.grpo_exec.pool import holdout_identity, no_sealed, write_jsonl_new

    no_sealed(out, CPT_TRAIN[0])
    _, held_packages = holdout_identity()
    panel = Path(c18.CPT_PANEL[0])
    require(sha256(panel) == c18.CPT_PANEL[1], "CPT 2K panel differs from its pin")
    validation_packages = {str(json.loads(line)["package"]) for line in panel.read_text().splitlines() if line.strip()}
    rows, info = cpt_replay_rows(CPT_TRAIN[0], CPT_TRAIN[1], CPT_TRAIN[2], count, held_packages | validation_packages)
    require(not {r["package"] for r in rows} & validation_packages, "a CPT validation package entered the replay")
    receipt = {"path": str(out), "sha256": write_jsonl_new(out, rows), "rows": len(rows), **info,
               "excluded": "hold-out-slice packages (206) and CPT 2K validation packages"}
    c18.write_new(out.with_suffix(".receipt.json"), receipt)
    return receipt


# Binding and loading -----------------------------------------------------------------------------------

def bind(kind: str, local_root: Path, *, deploy_root: Path = DEPLOY_ROOT, decided_by: str, name: str | None = None) -> Json:
    name = name or ("C18b-grpo-exec-pilot-1" if kind == "pilot" else "C18b-grpo-exec-smoke")
    recipe = c18.bind(kind, local_root, deploy_root=deploy_root, decided_by=decided_by, name=name)
    receipt = obj(json.loads(CPT_REPLAY.with_suffix(".receipt.json").read_text()), "CPT replay receipt")
    root = local_root.resolve()
    recipe.update(schema=BOUND_SCHEMA, card="C18b", pre_registered=copy.deepcopy(PRE_REGISTERED))
    cast(Json, recipe["arm"])["purpose"] = ("C18b: C18 plus CPT-text replay, from arm A 803 under R6x" if kind == "pilot" else
                                           "C18b smoke: real CUDA path with both replay terms, one save and resume, one gate")
    cast(Json, recipe["pool"])["cpt_replay"] = {"path": receipt["path"], "sha256": receipt["sha256"], "rows": receipt["rows"]}
    runtime = cast(Json, recipe["runtime"])
    runtime.update(cpt_replay_rows_per_update=CPT_REPLAY_SETTINGS["rows_per_update"],
                   cpt_replay_coefficient=CPT_REPLAY_SETTINGS["coefficient"], buffer_seconds_prior=200.0)
    recipe["c18_reference"] = {"card": "C18", "result_checkpoint": str(CHECKPOINTS / "C18-grpo-exec-pilot-1/full/checkpoint-100"),
                               "cpt_rise": {"50": 0.011716, "100": 0.023766}, "val_all_pass": {"0": 0.282878, "50": 0.297767,
                                                                                               "100": 0.305211}}
    recipe["source"] = {"repository_root": str(deploy_root), "manifest_path": SOURCE_MANIFEST,
                        "manifest_sha256": sha256(root / SOURCE_MANIFEST)}
    validate(recipe)
    return recipe


def validate(recipe: Mapping[str, object]) -> None:
    """C18's structural checks on a C18 view of the recipe, plus the C18b additions."""
    require(recipe.get("schema") == BOUND_SCHEMA and recipe.get("card") == "C18b", "not a C18b bound recipe")
    require(recipe.get("pre_registered") == PRE_REGISTERED, "pre-registered C18b values differ")
    view = copy.deepcopy(dict(recipe))
    view.update(schema=c18.BOUND_SCHEMA, card="C18", pre_registered=copy.deepcopy(c18.PRE_REGISTERED))
    c18.validate(view)
    runtime = obj(recipe.get("runtime"), "runtime")
    require(runtime.get("cpt_replay_rows_per_update") == CPT_REPLAY_SETTINGS["rows_per_update"]
            and runtime.get("cpt_replay_coefficient") == CPT_REPLAY_SETTINGS["coefficient"], "CPT replay settings differ")
    pool = obj(recipe.get("pool"), "pool")
    require(isinstance(pool.get("cpt_replay"), dict), "the CPT replay rows are not bound")


def load(path: Path, *, root_override: Path | None = None, check_source: bool = True, check_data: bool = True) -> Json:
    recipe = obj(json.loads(Path(path).read_text()), "recipe")
    validate(recipe)
    root = repository_root(recipe, root_override)
    check_file(root, obj(recipe["sft_recipe"], "sft_recipe"), "SFT recipe")
    if check_source:
        verify_source(root, obj(recipe["source"], "source"))
    if check_data:
        pool = obj(recipe["pool"], "pool")
        check_file(root, pool["manifest"], "pool manifest")
        for key in ("train", "val", "replay", "cpt_replay"):
            check_file(root, pool[key], f"{key} file")
        legacy = obj(pool["legacy"], "legacy")
        for key in ("manifest", "pool", "bundle"):
            check_file(root, legacy[key], f"C14 {key}")
        check_file(root, obj(recipe["memory_fit"], "memory_fit")["receipt"], "memory probe receipt")
        check_file(root, obj(obj(recipe["dev_gate"], "dev_gate")["cpt"], "cpt")["panel"], "CPT 2K panel")
        reward = obj(obj(recipe["runtime"], "runtime")["reward"], "reward")
        for key in ("air", "rscript"):
            require(os.access(str(reward[key]), os.X_OK), f"{key} is not executable at {reward[key]}")
    return recipe


# Entry hooks -------------------------------------------------------------------------------------------

def install() -> None:
    """Point the C18 driver, trainer and evaluators at this recipe module and the C18b entry modules."""
    from sepalith.training.grpo_exec import c18b_trainer, cpt_probe, dev_generation, night, trainer, val_eval

    this = sys.modules[__name__]
    for module in (night, trainer, val_eval, cpt_probe, dev_generation):
        module.recipe_module = this  # ty: ignore[invalid-assignment]
    hooks = {"TRAINER": ENTRY["trainer"], "DEV_EVALUATOR": ENTRY["dev"], "VAL_EVALUATOR": ENTRY["val"], "CPT_PROBE": ENTRY["cpt"],
             "UPDATE_FIELDS": (*night.UPDATE_FIELDS, "cpt_replay_nll", "grad_norm_after_sft")}
    for name, value in hooks.items():  # module constants the C18 driver reads at call time
        setattr(night, name, value)
    trainer.ExecGRPOTrainer = c18b_trainer.CptReplayTrainer  # ty: ignore[invalid-assignment]


def jobs(root: Path = DEPLOY_ROOT) -> dict[str, Json]:
    """Night-queue job files for the C18b smoke and pilot (written, never enqueued here)."""
    from sepalith.training.grpo_exec.jobs import command

    bound = "docs/training/plan-20260927/c18b/bound"
    smoke = command(root, "sepalith.training.grpo_exec.c18b_night", "smoke", "--recipe", str(root / f"{bound}/grpo-exec-smoke-arm-a-803.json"))
    pilot = command(root, "sepalith.training.grpo_exec.c18b_night", "run", "--recipe", str(root / f"{bound}/grpo-exec-pilot-1-arm-a-803.json"))
    return {
        "C18b-grpo-exec-smoke": {
            "id": "C18b-grpo-exec-smoke", "cwd": str(root), "env": [], "priority": 40, "resumable": False, "estimated_minutes": 45,
            "requires": [], "command": smoke,
            "note": ("C18b smoke (cuda:0): C18 plus CPT-text replay (0.2 x 4 CPT chunks per update) from arm A 803; updates 1-4, save, "
                     "DEV75 + 96 validation tasks + CPT 2K on checkpoint 4 and the 803 CPT reference, resume to 6. Exit 0 pass, 1 fail. "
                     "Receipt: /mnt/e/sepalith/campaign-20260915/checkpoints/C18b-grpo-exec-smoke/smoke-receipt.json")},
        "C18b-grpo-exec-pilot-1": {
            "id": "C18b-grpo-exec-pilot-1", "cwd": str(root), "env": [], "priority": 45, "resumable": True, "estimated_minutes": 960,
            "requires": ["C18b-grpo-exec-smoke"], "command": pilot,
            "note": ("C18b GRPO pilot on the exec environment from arm A 803 with SFT and CPT-text replay (card C18 addendum C18b): "
                     "the 803 references, then DEV75 + validation exec + CPT 2K every 50 updates, DEV250 every 150, R6x as C18 "
                     "(stop-for-user exits 3; the standing nofp/cap rule applies); complete at 1,000 updates or a plateau; writes "
                     "final-holdout-job.json. Resumable (exit 75).")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rows = sub.add_parser("cpt-replay")
    rows.add_argument("--out", type=Path, default=CPT_REPLAY)
    rows.add_argument("--rows", type=int, default=8000)
    source = sub.add_parser("source-manifest")
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[6])
    binder = sub.add_parser("bind")
    binder.add_argument("--kind", choices=("pilot", "smoke"), required=True)
    binder.add_argument("--output", type=Path, required=True)
    binder.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[6])
    binder.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT)
    binder.add_argument("--decided-by", required=True)
    binder.add_argument("--name")
    writer = sub.add_parser("write-jobs")
    writer.add_argument("--output", type=Path, required=True)
    writer.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT)
    verify = sub.add_parser("verify")
    verify.add_argument("--recipe", type=Path, required=True)
    verify.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "cpt-replay":
        print(json.dumps(build_cpt_replay(args.out, args.rows), indent=1))
    elif args.command == "source-manifest":
        write_new(args.output, build_source_manifest(args.local_root.resolve()))
    elif args.command == "bind":
        write_new(args.output, bind(args.kind, args.local_root, deploy_root=args.deploy_root, decided_by=args.decided_by,
                                   name=args.name))
    elif args.command == "write-jobs":
        for identifier, job in jobs(args.deploy_root).items():
            write_new(args.output / f"{identifier}.json", job)
        print(json.dumps(sorted(jobs(args.deploy_root))))
    else:
        recipe = load(args.recipe, root_override=args.repository_root)
        print(json.dumps({"status": "verified", "arm": obj(recipe["arm"], "arm")["name"], "sha256": sha256(args.recipe)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
