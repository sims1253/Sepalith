#!/usr/bin/env python3
"""C13 offline recipes: parent-free templates, parent binding and the fail-closed loader.

The C04 bound-recipe style, for the two offline steps:

- ``template``: everything pre-registered except the starting model: the data
  (RFT cohort and schedule, or DPO pairs and pair schedule), the DEV panels,
  the optimizer, the learning-rate ratio to the SFT LR, the gates at halves.
- ``bind``: adds the starting model (any gated full-weight checkpoint: arm A
  536, arm B 684 if R3 adopts it, or an adopted RFT model for DPO), the bound
  gate decision whose DEV75 and DEV250 results are the step's reference, the
  learning rate (SFT LR / 3 for RFT, / 10 for DPO), outputs and source.
- ``load_offline``: what the trainer, the reference pass, DEV generation and
  the night driver call; every hash is checked again.

Pre-registered (card C13): aurora_mix with the CPT mechanics, side LR = LR / 10,
32 warmup updates then cosine decay to 10% at the step's horizon, weight decay
0, effective batch 16 (rows for RFT, pairs for DPO), micro-batch 1, seed 3407,
DEV75 and DEV250 gates at half and at the end of the horizon under the R1-style
offline rule (``offline.rules``), R5 at the end, every checkpoint retained.
DPO: sigmoid loss, beta 0.1, one pass over the pairs, reference
log-probabilities of the starting model precomputed and stored.

Switching the parent is cheap: bind the same template again with
``--parent``, ``--parent-decision`` and ``--parent-recipe`` of the new start model.
A different census (for example on arm B after R3) needs a new data build and
template; nothing else changes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import cast

from sepalith.training.sft import recipe_v2
from sepalith.training.sft.recipe_v2 import (
    CHECKPOINT_BASE,
    CPT_OPTIMIZER_MECHANICS,
    EFFECTIVE_BATCH,
    GATE_ROOT,
    GENERATION_MAX_NEW_TOKENS,
    MIN_LR_RATE,
    PARENT_FILES,
    RETENTION,
    SEED,
    SIDE_LR_RATIO,
    STATE_ROOT,
    TOKENIZER_SHA256,
    WARMUP_STEPS,
    Json,
    PathMap,
    file_record,
    generation_batching_record,
    require,
    sha256,
    write_new,
)

TEMPLATE_SCHEMA = "sepalith.c13.offline-template.v1"
BOUND_SCHEMA = "sepalith.c13.offline-bound.v1"
SOURCE_SCHEMA = "sepalith.c13.offline-source.v1"
REFERENCE_SCHEMA = "sepalith.c13.reference-logps-manifest.v1"
DECISION_SCHEMA = "sepalith.sft11.full-weight-edit-dev-decision.v1"
KINDS = ("rft", "dpo")
LR_RATIO = {"rft": 1 / 3, "dpo": 1 / 10}
OBJECTIVE = {"rft": "exact_target_body_terminal_eos_only", "dpo": "dpo_sigmoid_completion_logp_sum"}
BETA = 0.1
SFT_LEARNING_RATES = (3e-6, 1e-5, 3e-5)
MAX_SEQUENCE_TOKENS = 4096
SOURCE_GLOBS = (*recipe_v2.SOURCE_GLOBS, "packages/sepalith/src/sepalith/training/offline/*.py")
DEV75 = Path("/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl")
DEV250 = Path("/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/dev250-v1.jsonl")
DEV250_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/manifest.json")
OPTIMIZER_DISPATCH = "docs/campaign/work/lead/r2-full-weight-hybrid-smoke-root-v3/expected-aurora-mix-manifest.json"
_TAG = re.compile(r"[a-z0-9][a-z0-9.-]{0,40}")


def _obj(value: object, label: str) -> Json:
    require(isinstance(value, dict), f"{label} must be an object")
    return cast(Json, value)


def _str(value: object, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a non-empty string")
    return cast(str, value)


def _int(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer")
    return cast(int, value)


def halves(updates: int) -> list[int]:
    return [round(updates / 2), updates]


def stem(step_name: str, parent_tag: str) -> str:
    return f"C13-{step_name}-{parent_tag}"


# Source manifest ---------------------------------------------------------------------------------

def source_files(root: Path) -> list[str]:
    files = sorted({str(path.relative_to(root)) for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()})
    require(files, "source globs found no files")
    return files


def build_source_manifest(root: Path) -> Json:
    root = Path(root).resolve()
    files = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256(root / name)} for name in source_files(root)]
    return {"schema": SOURCE_SCHEMA, "root": "repository", "globs": list(SOURCE_GLOBS), "files": files,
            "evaluator_files_hashed_per_gate": list(recipe_v2.EVALUATOR_FILES)}


def verify_source(record: Mapping[str, object]) -> Json:
    path = Path(_str(record.get("manifest_path"), "source manifest path"))
    root = Path(_str(record.get("repository_root"), "source repository root"))
    require(path.is_file() and sha256(path) == record.get("manifest_sha256"), "source manifest differs")
    manifest = _obj(json.loads(path.read_text()), "source manifest")
    require(manifest.get("schema") == SOURCE_SCHEMA, "source manifest schema differs")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "source manifest lists no files")
    listed = set()
    for item in cast(list[object], files):
        entry = _obj(item, "source item")
        name = _str(entry.get("path"), "source path")
        target = root / name
        require(target.is_file() and target.stat().st_size == entry.get("bytes") and sha256(target) == entry.get("sha256"),
                f"source differs:{name}")
        listed.add(name)
    require(set(source_files(root)) == listed, "source file set differs from the bound manifest")
    return manifest


# Parent reference --------------------------------------------------------------------------------

def checkpoint_step(path: Path) -> int:
    suffix = Path(path).name.removeprefix("checkpoint-")
    require(suffix.isdigit(), f"not a checkpoint directory: {path}")
    return int(suffix)


def parent_files(parent: Path) -> dict[str, str]:
    return {name: sha256(parent / name) for name in PARENT_FILES}


def reference_from_decision(decision_path: Path, parent: Path, panels: Mapping[str, object]) -> Json:
    """The starting model's own gate: decision, both results and their metrics, all bound by hash."""
    from sepalith.training.eval.milestone_gate import generation_settings, requested_generation_settings
    from sepalith.training.offline.rules import offline_metrics

    decision = _obj(json.loads(decision_path.read_text()), "parent decision")
    manifest_sha = sha256(parent / "campaign-manifest.json")
    require(decision.get("schema") == DECISION_SCHEMA and decision.get("evaluation_complete") is True
            and decision.get("outcome") == "continue" and decision.get("continue_training") is True,
            "the starting model's gate decision is not a completed continue decision")
    require(decision.get("checkpoint_manifest_sha256") == manifest_sha and decision.get("step") == checkpoint_step(parent),
            "the gate decision belongs to another checkpoint")
    expected = requested_generation_settings(GENERATION_MAX_NEW_TOKENS, *_batching())
    require(decision.get("generation_runtime") == expected, "the starting model's gate decoded with other generation settings")
    results = _obj(decision.get("results"), "decision results")
    summaries, records = {}, {}
    for name in ("dev75", "dev250"):
        entry = _obj(results.get(name), name)
        path = Path(_str(entry.get("path"), f"{name} result"))
        require(path.is_file() and sha256(path) == entry.get("sha256"), f"{name} result differs from its decision")
        value = _obj(json.loads(path.read_text()), f"{name} result")
        panel = _obj(panels[name], name)
        require(value.get("status") == "complete" and _obj(value.get("panel"), "panel").get("sha256") == panel["sha256"]
                and _obj(value.get("checkpoint"), "checkpoint").get("campaign_manifest_sha256") == manifest_sha,
                f"{name} result binding differs")
        require(generation_settings(value) == expected, f"{name} result decoded with other generation settings")
        summaries[name] = _obj(value.get("summary"), "summary")
        records[name] = {"path": str(path), "sha256": sha256(path)}
    metrics = offline_metrics(summaries["dev75"], summaries["dev250"])
    recorded = _obj(decision.get("metrics"), "decision metrics")
    require(all(recorded.get(k) == metrics[k] for k in ("valid", "cap", "edit", "nofp", "score250")),
            "decision metrics do not match its results")
    return {"decision": {"path": str(decision_path), "sha256": sha256(decision_path)}, "results": records,
            "checkpoint_manifest_sha256": manifest_sha, "bound_recipe_sha256": decision.get("bound_recipe_sha256"),
            "step": decision.get("step"), "metrics": metrics, "generation_runtime": expected}


def _batching() -> tuple[int, str]:
    record = generation_batching_record()
    return cast(int, record["batch_size"]), cast(str, record["kv_cache"])


def sft_learning_rate_of(recipe_path: Path, decision_recipe_sha: object) -> float:
    """The SFT LR behind a starting model: an SFT v2 arm's LR, or the one an offline recipe was bound with."""
    require(sha256(recipe_path) == decision_recipe_sha, "the parent recipe is not the one its gate decision binds")
    recipe = _obj(json.loads(recipe_path.read_text()), "parent recipe")
    if recipe.get("schema") == recipe_v2.BOUND_SCHEMA:
        value = _obj(recipe["runtime"], "runtime")["learning_rate"]
    else:
        require(recipe.get("schema") == BOUND_SCHEMA, "the parent recipe is neither an SFT v2 nor a C13 recipe")
        value = _obj(recipe["pre_registered"], "pre_registered")["sft_learning_rate"]
    require(isinstance(value, float) and value in SFT_LEARNING_RATES, "the SFT LR is not one of the pilot LRs")
    return cast(float, value)


# Template ----------------------------------------------------------------------------------------

def _panels() -> Json:
    def count(path: Path) -> int:
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    dev75 = {"path": str(DEV75), "sha256": sha256(DEV75)}
    return {"dev75": {**dev75, "cases": count(DEV75)},
            "dev250": {"path": str(DEV250), "sha256": sha256(DEV250), "cases": count(DEV250),
                       "manifest": {"path": str(DEV250_MANIFEST), "sha256": sha256(DEV250_MANIFEST)}}}


def build_template(kind: str, data_manifest: Path, *, step_name: str, paths: PathMap, panels: Json | None = None) -> Json:
    require(kind in KINDS and _TAG.fullmatch(step_name), "kind or step name differs")
    manifest = _obj(json.loads(data_manifest.read_text()), "data manifest")
    require(manifest.get("schema") == "sepalith.c13.offline-data-manifest.v1" and manifest.get("split") == "train"
            and manifest.get("final_set_access") is False, "data manifest differs")
    outputs = _obj(manifest["outputs"], "data outputs")

    def checked(name: str) -> tuple[Path, Json]:
        record = _obj(outputs[name], name)
        path = Path(_str(record["path"], name))
        require(path.is_file() and sha256(path) == record["sha256"], f"data file differs: {name}")
        return path, file_record(path)

    panels = panels or _panels()
    dispatch_path = paths.local / OPTIMIZER_DISPATCH
    dispatch = _obj(json.loads(dispatch_path.read_text()), "optimizer dispatch")
    if kind == "rft":
        rows_path, rows = checked("rft_cohort_rows")
        schedule_path, schedule = checked("rft_draw_schedule")
        _, profile = checked("rft_length_profile")
        plan = _obj(json.loads(schedule_path.read_text()), "schedule")
        coverage = _obj(plan["coverage"], "coverage")
        policy = _obj(plan["policy"], "policy")
        require(policy.get("schedule_version") == recipe_v2.SCHEDULE_VERSION and policy.get("seed") == SEED
                and _obj(plan["input"], "input").get("sha256") == rows["sha256"], "RFT schedule is not schedule v2 on the cohort")
        tokens = _obj(_obj(plan["token_denominators"], "tokens")["unique"], "unique")
        updates = _int(plan["max_steps"], "updates")
        profile_value = _obj(json.loads(Path(_str(profile["path"], "profile")).read_text()), "profile")
        data = {"id": f"C13-rft-cohort-{manifest['model_id']}", "rows": rows, "draw_schedule": schedule,
                "length_profile": profile, "manifest": file_record(data_manifest), "max_sequence_tokens": 16384,
                "unique_rows": coverage["eligible_rows"], "updates": updates, "input_tokens": tokens["input_tokens"],
                "loss_tokens": tokens["supervised_target_tokens_including_eos"], "body_tokens": tokens["target_body_tokens"],
                "observed_max_sequence_tokens": _obj(profile_value["sequence_tokens"], "profile")["max"],
                "target_truncation": False,
                "schedule_semantics": {"version": recipe_v2.SCHEDULE_VERSION, "edit_per_batch": policy["edit_per_batch"],
                                       "noop_per_batch": policy["noop_per_batch"], "noop_share": coverage["noop_share"],
                                       "exposure_histogram_by_pool": coverage["exposure_histogram_by_pool"]},
                "composition": manifest["cohort"]}
    else:
        pairs_path, pairs = checked("dpo_pairs")
        schedule_path, schedule = checked("dpo_pair_schedule")
        plan = _obj(json.loads(schedule_path.read_text()), "pair schedule")
        require(plan.get("pairs_sha256") == pairs["sha256"] and plan.get("seed") == SEED
                and plan.get("effective_batch") == EFFECTIVE_BATCH, "pair schedule is not for these pairs")
        updates = _int(plan["max_steps"], "updates")
        dpo = _obj(manifest["dpo"], "dpo")
        data = {"id": f"C13-dpo-pairs-{manifest['model_id']}", "rows": pairs, "draw_schedule": schedule,
                "manifest": file_record(data_manifest), "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                "unique_rows": plan["pairs"], "updates": updates, "draws": plan["draw_count"],
                "observed_max_sequence_tokens": dpo["max_sequence_tokens"], "tokens": dpo["tokens"],
                "target_truncation": False, "composition": {k: dpo[k] for k in ("chosen_sources", "rejected_outcomes", "by_family")}}
    require(halves(updates)[0] > WARMUP_STEPS, "the first gate must come after warmup")
    return {
        "schema": TEMPLATE_SCHEMA, "status": "template_parent_unbound", "launch_authorized": False, "parent": None,
        "kind": kind, "step_name": step_name, "seed": SEED, "cohort": data,
        "development": {"cases": cast(Json, panels["dev75"])["cases"], "final_set_access": False,
                        "generation_budget": {"default_candidate": GENERATION_MAX_NEW_TOKENS, "prepared_max_new_tokens_bound": 8192,
                                              "root_selected": True,
                                              "policy": "no silent truncation; report cap hit, response completeness and stop token"},
                        "generation_evaluation": "required_for_edit_and_noop_quality",
                        "teacher_forced_diagnostic": "separate_metric_not_edit_accuracy",
                        "panel": {k: cast(Json, panels["dev75"])[k] for k in ("path", "sha256")}},
        "dev_panels": panels,
        "expected_model": {"dtype": "bfloat16", "parameter_tensors": 381, "parameters": 2516756480, "peft": False},
        "optimizer_dispatch": {**file_record(dispatch_path, paths.to_deploy(dispatch_path)),
                               "ordered_rows_sha256": dispatch["ordered_rows_sha256"]},
        "pre_registered": {
            "card": "C13", "kind": kind, "learning_rate_ratio_to_sft": LR_RATIO[kind],
            "optimizer_mechanics": CPT_OPTIMIZER_MECHANICS, "side_lr_ratio": SIDE_LR_RATIO, "scheduler": "cosine_with_min_lr",
            "warmup_steps": WARMUP_STEPS, "min_lr_rate": MIN_LR_RATE, "weight_decay": recipe_v2.WEIGHT_DECAY,
            "effective_batch": EFFECTIVE_BATCH, "unit": "rows" if kind == "rft" else "pairs", "gate_updates": halves(updates),
            "gate_rule": "R1-offline (sepalith.training.offline.rules: R1 later-gate table vs the starting model, then vs the "
                         "previous gate; noedit_dodge and, for DPO, dpo_chosen_collapse are stop-for-user)",
            "adoption_rule": "R5 (docs/training/plan-20260927/RULES.md) at the step's result checkpoint",
            **({"beta": BETA, "loss": "sigmoid", "reference": "starting model, bf16, precomputed and hashed"} if kind == "dpo" else {}),
        },
    }


# Binding -----------------------------------------------------------------------------------------

def runtime_for(learning_rate: float, evidence_directory: str, updates: int) -> Json:
    gates = halves(updates)
    optimizer = dict(CPT_OPTIMIZER_MECHANICS)
    optimizer.update(hidden_lr=learning_rate, side_lr=float(f"{learning_rate * SIDE_LR_RATIO:.6g}"))
    return {"micro_batch": 1, "gradient_accumulation": EFFECTIVE_BATCH, "effective_batch": EFFECTIVE_BATCH, "max_steps": updates,
            "learning_rate": learning_rate, "optimizer": optimizer, "scheduler": "cosine_with_min_lr",
            "lr_scheduler_kwargs": {"min_lr_rate": MIN_LR_RATE}, "warmup_steps": WARMUP_STEPS,
            "weight_decay": recipe_v2.WEIGHT_DECAY, "checkpoint_every": updates, "evaluation_steps": gates,
            "selected_milestones": gates, "mandatory_dev_gate_steps": gates, "telemetry_every": 1,
            "gate_evidence_directory": evidence_directory, "development_max_new_tokens": GENERATION_MAX_NEW_TOKENS}


def bind(template_path: Path, output: Path, *, parent: Path, parent_decision: Path, parent_recipe: Path, parent_tag: str,
         decided_by: str, source_manifest: Path, paths: PathMap) -> Json:
    template = _obj(json.loads(template_path.read_text()), "template")
    require(template.get("schema") == TEMPLATE_SCHEMA and template.get("parent") is None
            and template.get("launch_authorized") is False, "template differs")
    require(_TAG.fullmatch(parent_tag) is not None and decided_by.strip(), "parent tag or decision record missing")
    kind, step_name = _str(template["kind"], "kind"), _str(template["step_name"], "step name")
    parent = parent.resolve()
    files = parent_files(parent)
    require(files["tokenizer.json"] == TOKENIZER_SHA256, "parent tokenizer differs")
    panels = _obj(template["dev_panels"], "panels")
    reference = reference_from_decision(parent_decision.resolve(), parent, panels)
    sft_lr = sft_learning_rate_of(parent_recipe.resolve(), reference["bound_recipe_sha256"])
    learning_rate = float(f"{sft_lr * LR_RATIO[kind]:.6g}")
    updates = _int(_obj(template["cohort"], "cohort")["updates"], "updates")
    name = stem(step_name, parent_tag)
    evidence = f"{GATE_ROOT}/{name}"
    outputs = {"trainer": f"{CHECKPOINT_BASE}/training/{name}/runtime", "archive": f"{CHECKPOINT_BASE}/checkpoints/{name}",
               "graceful_stop": f"{STATE_ROOT}/control/{name}-save-stop.json"}
    bound = copy.deepcopy(template)
    pre = _obj(bound["pre_registered"], "pre_registered")
    pre.update(sft_learning_rate=sft_lr, learning_rate=learning_rate)
    bound.update({
        "schema": BOUND_SCHEMA, "status": "bound_not_launched", "launch_authorized": True,
        "parent": {"kind": "gated_full_weight_checkpoint", "candidate_id": f"{parent.parent.parent.name}-{checkpoint_step(parent)}",
                   "tag": parent_tag, "path": str(parent), "step": checkpoint_step(parent), "files": files},
        "parent_reference": reference,
        "arm": {"name": name, "learning_rate": learning_rate,
                "purpose": f"C13 {step_name}: {kind.upper()} from {parent_tag}, LR {learning_rate:g} (SFT LR {sft_lr:g} x "
                           f"{LR_RATIO[kind]:.4g}), gates {halves(updates)} under R1-offline, R5 at the end"},
        "runtime": runtime_for(learning_rate, evidence, updates),
        "dev_gate": {"mandatory_steps": halves(updates), "evidence_directory": evidence,
                     "panel": _obj(template["development"], "development")["panel"], "panels": panels, "rule": "R1-offline",
                     "continuation_policy": "automatic_rule_R1_offline_with_human_override_file",
                     "override_file_pattern": f"{evidence}/override-step-<step>.json",
                     "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
                     "generation_batching": generation_batching_record()},
        "outputs": outputs,
        "checkpoint_storage": {"mode": "e_same_filesystem_atomic", "c_hot_stage": "not_admitted", "required_same_filesystem": True,
                               "trainer_root": outputs["trainer"], "archive_root": outputs["archive"]},
        "source": {"manifest_path": paths.to_deploy(source_manifest), "manifest_sha256": sha256(source_manifest),
                   "repository_root": str(paths.deploy)},
        "template": {"path": paths.to_deploy(template_path), "sha256": sha256(template_path)},
        "parent_recipe": {"path": paths.to_deploy(parent_recipe), "sha256": sha256(parent_recipe)},
        "retention": dict(RETENTION),
        "decided_by": decided_by,
    })
    if kind == "dpo":
        root = f"{CHECKPOINT_BASE}/training/{name}-reference"
        bound["reference"] = {"logps": f"{root}/reference-logps.jsonl", "manifest": f"{root}/reference-manifest.json",
                              "dtype": "bfloat16", "policy": "computed once under the starting model by offline.reference"}
    validate(bound)
    write_new(output, bound)
    return bound


def validate(recipe: Mapping[str, object]) -> None:
    """Refuse a recipe whose optimizer, schedule, gates or retention differ from the pre-registration."""
    kind = recipe.get("kind")
    require(kind in KINDS, "kind differs")
    runtime = _obj(recipe.get("runtime"), "runtime")
    pre = _obj(recipe.get("pre_registered"), "pre_registered")
    lr = runtime.get("learning_rate")
    sft_lr = pre.get("sft_learning_rate")
    require(isinstance(lr, float) and isinstance(sft_lr, float) and sft_lr in SFT_LEARNING_RATES
            and math.isclose(lr, sft_lr * LR_RATIO[cast(str, kind)], rel_tol=1e-5) and lr == pre.get("learning_rate"),
            "LR differs from the pre-registered ratio to the SFT LR")
    optimizer = _obj(runtime.get("optimizer"), "optimizer")
    require({k: optimizer.get(k) for k in CPT_OPTIMIZER_MECHANICS} == CPT_OPTIMIZER_MECHANICS
            and set(optimizer) == set(CPT_OPTIMIZER_MECHANICS) | {"hidden_lr", "side_lr"}
            and optimizer.get("hidden_lr") == lr and math.isclose(cast(float, optimizer.get("side_lr")), cast(float, lr) * SIDE_LR_RATIO,
                                                                  rel_tol=1e-5), "optimizer differs from the CPT mechanics")
    require(runtime.get("scheduler") == "cosine_with_min_lr" and runtime.get("lr_scheduler_kwargs") == {"min_lr_rate": MIN_LR_RATE}
            and runtime.get("warmup_steps") == WARMUP_STEPS and runtime.get("weight_decay") == recipe_v2.WEIGHT_DECAY,
            "LR schedule differs")
    require(runtime.get("micro_batch") == 1 and runtime.get("gradient_accumulation") == EFFECTIVE_BATCH
            and runtime.get("effective_batch") == EFFECTIVE_BATCH, "batch geometry differs")
    updates = _int(_obj(recipe.get("cohort"), "cohort").get("updates"), "updates")
    gates = halves(updates)
    require(runtime.get("max_steps") == updates == runtime.get("checkpoint_every"), "horizon differs from the schedule")
    require(runtime.get("evaluation_steps") == gates == runtime.get("mandatory_dev_gate_steps") == runtime.get("selected_milestones")
            == _obj(recipe.get("dev_gate"), "dev_gate").get("mandatory_steps") == pre.get("gate_updates"), "gates are not the halves")
    require(recipe.get("seed") == SEED and recipe.get("retention") == RETENTION, "seed or retention differs")
    if kind == "dpo":
        require(pre.get("beta") == BETA and pre.get("loss") == "sigmoid", "DPO loss differs")
    gate = _obj(recipe.get("dev_gate"), "dev_gate")
    require(gate.get("generation_batching") == generation_batching_record(), "DEV generation batching differs")


# Loading -----------------------------------------------------------------------------------------

def reference_manifest(recipe: Mapping[str, object]) -> Json:
    """The verified manifest of the stored reference log-probabilities (DPO)."""
    record = _obj(recipe.get("reference"), "reference")
    path = Path(_str(record.get("manifest"), "reference manifest"))
    require(path.is_file(), f"reference log-probabilities missing: run the reference job first ({path})")
    manifest = _obj(json.loads(path.read_text()), "reference manifest")
    parent = _obj(recipe["parent"], "parent")
    rows = _obj(_obj(recipe["cohort"], "cohort")["rows"], "pairs")
    require(manifest.get("schema") == REFERENCE_SCHEMA and manifest.get("status") == "complete"
            and manifest.get("parent_weights_sha256") == _obj(parent["files"], "files")["model.safetensors"]
            and manifest.get("pairs_sha256") == rows["sha256"], "reference manifest is for another model or pair file")
    logps = Path(_str(record.get("logps"), "reference logps"))
    require(logps.is_file() and sha256(logps) == manifest.get("logps_sha256"), "reference log-probabilities differ from their manifest")
    return manifest


def load_offline(path: str | os.PathLike[str], *, require_reference: bool = True) -> Json:
    """Fail-closed load of a C13 bound recipe."""
    from sepalith.training.eval.milestone_gate import mandatory_steps
    from sepalith.training.sft.full_weight_edit_sft import validate_context_and_storage

    recipe = _obj(json.loads(Path(path).read_text()), "bound recipe")
    require(recipe.get("schema") == BOUND_SCHEMA and recipe.get("status") == "bound_not_launched"
            and recipe.get("launch_authorized") is True, "not a C13 bound recipe")
    for key in ("template", "parent_recipe"):
        record = _obj(recipe.get(key), key)
        target = Path(_str(record.get("path"), key))
        require(target.is_file() and sha256(target) == record.get("sha256"), f"{key} differs")
    verify_source(_obj(recipe.get("source"), "source"))
    validate(recipe)
    mandatory_steps(recipe)
    validate_context_and_storage(recipe)
    cohort = _obj(recipe["cohort"], "cohort")
    panels = _obj(_obj(recipe["dev_gate"], "dev_gate")["panels"], "panels")
    dev250 = _obj(panels["dev250"], "dev250")
    records = [cohort["rows"], cohort["draw_schedule"], cohort["manifest"], panels["dev75"], dev250, dev250["manifest"],
               _obj(recipe["development"], "development")["panel"], recipe["optimizer_dispatch"]]
    if recipe["kind"] == "rft":
        records.append(cohort["length_profile"])
    for item in records:
        entry = _obj(item, "input")
        target = Path(_str(entry.get("path"), "input path"))
        require(target.is_file() and sha256(target) == entry.get("sha256"), f"input differs:{target}")
    parent = _obj(recipe["parent"], "parent")
    root = Path(_str(parent.get("path"), "parent path"))
    require(parent.get("kind") == "gated_full_weight_checkpoint" and root.is_dir(), "parent missing")
    for name, expected in _obj(parent["files"], "parent files").items():
        require((root / name).is_file() and sha256(root / name) == expected, f"parent differs:{name}")
    reference = _obj(recipe["parent_reference"], "parent_reference")
    again = reference_from_decision(Path(_str(_obj(reference["decision"], "decision")["path"], "decision")), root, panels)
    require(again == reference, "the starting model's gate reference differs")
    if recipe["kind"] == "dpo" and require_reference:
        reference_manifest(recipe)
    return recipe


def identity(recipe: Mapping[str, object]) -> Json:
    """The checkpoint identity every offline checkpoint is sealed and verified with."""
    kind = cast(str, recipe["kind"])
    parent, cohort = _obj(recipe["parent"], "parent"), _obj(recipe["cohort"], "cohort")
    runtime, gate = _obj(recipe["runtime"], "runtime"), _obj(recipe["dev_gate"], "dev_gate")
    data = {"cohort_id": cohort["id"], "rows_sha256": _obj(cohort["rows"], "rows")["sha256"],
            "draw_schedule_sha256": _obj(cohort["draw_schedule"], "schedule")["sha256"]}
    policy = {"stage": f"c13_offline_{kind}", "arm": _obj(recipe["arm"], "arm")["name"], "optimizer": runtime["optimizer"],
              "full_text_labels": False, "mandatory_dev_gate_steps": gate["mandatory_steps"],
              "development_panel_sha256": _obj(gate["panel"], "panel")["sha256"],
              "dev250_panel_sha256": _obj(_obj(gate["panels"], "panels")["dev250"], "dev250")["sha256"]}
    if kind == "dpo":
        data["reference_logps_sha256"] = reference_manifest(recipe)["logps_sha256"]
        policy["beta"] = _obj(recipe["pre_registered"], "pre_registered")["beta"]
    return {"parent": {"candidate_id": parent["candidate_id"], "weights_sha256": _obj(parent["files"], "files")["model.safetensors"]},
            "tokenizer": {"sha256": _obj(parent["files"], "files")["tokenizer.json"], "bos": 0, "eos_pad": 1, "native_eog": [1, 130073]},
            "renderer": {"id": "zeta2-prm03-v1", "objective": OBJECTIVE[kind], "max_sequence_tokens": cohort["max_sequence_tokens"]},
            "data": data, "source": {"manifest_sha256": _obj(recipe["source"], "source")["manifest_sha256"]}, "policy": policy,
            "schedule": {k: runtime[k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate",
                                                 "scheduler", "lr_scheduler_kwargs", "warmup_steps", "weight_decay", "checkpoint_every")}}


# CLI ---------------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy-root", default="/home/m0hawk/sandbox/Sepalith")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[6]))
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--kind", choices=KINDS, required=True)
    template.add_argument("--step-name", required=True, help="rft-1 or dpo-1")
    template.add_argument("--data-manifest", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    source = sub.add_parser("source-manifest")
    source.add_argument("--output", type=Path, required=True)
    binder = sub.add_parser("bind")
    binder.add_argument("--template", type=Path, required=True)
    binder.add_argument("--parent", type=Path, required=True, help="the starting checkpoint directory")
    binder.add_argument("--parent-decision", type=Path, required=True, help="the starting checkpoint's gate decision record")
    binder.add_argument("--parent-recipe", type=Path, required=True, help="the recipe that decision binds")
    binder.add_argument("--parent-tag", required=True, help="for example arm-a-536")
    binder.add_argument("--decided-by", required=True)
    binder.add_argument("--source-manifest", type=Path, required=True)
    binder.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="load a bound recipe with deploy paths mapped to the local tree")
    verify.add_argument("--recipe", type=Path, required=True)
    verify.add_argument("--no-reference", action="store_true", help="skip the DPO reference check (before the reference job)")
    args = parser.parse_args(argv)
    paths = PathMap(Path(args.deploy_root), Path(args.local_root).resolve())
    if args.command == "template":
        write_new(args.output, build_template(args.kind, args.data_manifest, step_name=args.step_name, paths=paths))
    elif args.command == "source-manifest":
        write_new(args.output, build_source_manifest(paths.local))
    elif args.command == "bind":
        bind(args.template, args.output, parent=args.parent, parent_decision=args.parent_decision, parent_recipe=args.parent_recipe,
             parent_tag=args.parent_tag, decided_by=args.decided_by, source_manifest=args.source_manifest, paths=paths)
    else:
        recipe = paths.localize(json.loads(args.recipe.read_text()))
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "recipe.json"
            local.write_text(json.dumps(recipe))
            load_offline(local, require_reference=not args.no_reference)
        print(json.dumps({"status": "verified", "recipe": str(args.recipe), "sha256": sha256(args.recipe)}))
        return 0
    print(json.dumps({"status": "written", "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
