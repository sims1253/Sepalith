#!/usr/bin/env python3
"""Editing-SFT recipe v2: pre-registered template, arm admission and binder.

Card C04 pre-registers everything except the learning rate, which is the only
thing that differs between arms:

- schedule v2 (13 edits plus 3 no-ops per update, 1,071 updates, seed 3407);
- aurora_mix with the CPT mechanics; hidden LR is the arm's LR, side LR is LR / 10;
- 32 warmup updates, then cosine decay to 10% of peak at update 1,071;
- weight decay 0;
- DEV gates at updates 268, 536, 803 and 1,071, decided by rule R1.

Learning-rate ceiling. The binder refuses any LR above 3e-5. That is ten
times the CPT hidden LR (3e-6, side 3e-7) the parent was trained with.
Aurora and Muon updates are orthogonalized and rescaled to a fixed RMS
(``rms_scale`` 0.2 times LR) whatever the gradient size, so the LR directly
sets how far each update moves the weights. A pilot of 268 updates (one
quarter of the horizon) cannot show slow forgetting of CPT knowledge, so the
search stays within one decade above the CPT rate: 3e-6, 1e-5 and 3e-5.

Paths. Bound recipes are written for the checkout the night jobs run from
(``--deploy-root``, normally /home/m0hawk/sandbox/Sepalith) while the hashes
are computed from ``--local-root``. Both trees must hold identical files;
``verify`` checks that.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import cast

from sepalith.training.sft.schedule_v2 import SCHEDULE_VERSION, gate_updates

TEMPLATE_SCHEMA = "sepalith.sft-v2.full-weight-edit-sft-template.v1"
ADMISSION_SCHEMA = "sepalith.sft-v2.full-weight-edit-sft-arm-admission.v1"
BOUND_SCHEMA = "sepalith.sft-v2.full-weight-edit-sft-bound.v1"
SOURCE_SCHEMA = "sepalith.training-package-source.v1"
EXAMPLE_MARKERS = ("EXAMPLE_ONLY", "FILL")

# Values for the current 15,006-row cohort. The template derives the horizon from
# the schedule and the gates from the horizon (quarters), so a corrected row file
# only needs a new schedule, length profile and data queue, not code changes.
UPDATES = 1071
GATES = (268, 536, 803, 1071)
EFFECTIVE_BATCH = 16
WARMUP_STEPS = 32
MIN_LR_RATE = 0.1
SIDE_LR_RATIO = 0.1
WEIGHT_DECAY = 0.0
LR_UPPER_BOUND = 3e-5
CPT_HIDDEN_LR = 3e-6
PILOT_LEARNING_RATES = (3e-6, 1e-5, 3e-5)
SEED = 3407
GENERATION_MAX_NEW_TOKENS = 1024
RETENTION = {"policy": "retain_all", "preserve_selected_milestones": True,
             "reason": "R7: deleting any checkpoint is stop-for-user; every gate checkpoint and every deadline save is kept"}
LR_UPPER_BOUND_RATIONALE = (
    "3e-5 is ten times the CPT hidden LR of 3e-6. Aurora/Muon updates have a fixed RMS of rms_scale x LR, so the LR "
    "directly sets how far each update moves the weights, and a 268-update pilot cannot reveal slow forgetting of CPT "
    "knowledge; the search therefore stays within one decade above the CPT rate."
)
# aurora_mix mechanics as in the CPT recipe (r2-cpt450-cadence64-continuation-preparation-v2).
CPT_OPTIMIZER_MECHANICS = {
    "arm": "aurora_mix", "momentum": 0.95, "ns_steps": 5, "rms_scale": 0.2, "aurora_beta": 0.5, "aurora_K": 2,
    "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "weight_decay": WEIGHT_DECAY, "state_dtype": "float32",
    "bf16_update_policy": "stochastic_round", "stochastic_round_chunk_elements": 1048576,
}
PARENT_FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "campaign-manifest.json")
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
STATE_ROOT = "/home/m0hawk/.local/state/sepalith/campaign-20260915"
GATE_ROOT = f"{STATE_ROOT}/evaluation-gates"
BASELINE_PATH = f"{GATE_ROOT}/SFT-v2-parent-11586-baseline/baseline.json"
CHECKPOINT_BASE = "/mnt/e/sepalith/campaign-20260915"
# Package modules on the trainer's path; recipes bind their hashes. The
# training/eval modules (DEV generation and the gate-evidence files) are hashed
# into every gate decision instead, so the evaluator can change (card C05,
# PR #41) without rebinding the training recipes.
SOURCE_GLOBS = (
    "packages/sepalith/src/sepalith/campaign_protocol.py",
    "packages/sepalith/src/sepalith/training/paths.py",
    "packages/sepalith/src/sepalith/training/sft/*.py",
    "packages/sepalith/src/sepalith/training/optim/*.py",
    "packages/sepalith/src/sepalith/training/checkpoint/*.py",
    "packages/sepalith/src/sepalith/training/contracts/*.py",
    "packages/sepalith/src/sepalith/training/guards/*.py",
    "packages/sepalith/src/sepalith/training/supervisors/request_graceful_stop.py",
)
EVALUATOR_FILES = (
    "packages/sepalith/src/sepalith/training/eval/run_dev_generation.py",
    "packages/sepalith/src/sepalith/training/eval/campaign_eval.py",
    "packages/sepalith/src/sepalith/training/eval/dev_model_state.py",
    "packages/sepalith/src/sepalith/training/eval/milestone_gate.py",
    "packages/sepalith/src/sepalith/training/eval/prepare_dev_gate.py",
)
_ARM = re.compile(r"[a-z0-9][a-z0-9.-]{0,40}")

Json = dict[str, object]


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, recorded: str | None = None) -> Json:
    return {"path": recorded or str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_new(path: Path, value: object) -> None:
    require(not path.exists(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _object(value: object, label: str) -> Json:
    require(isinstance(value, dict), f"{label} must be an object")
    return cast(Json, value)


def _int(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer")
    return cast(int, value)


def _str(value: object, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a non-empty string")
    assert isinstance(value, str)
    return value


def arm_tag(learning_rate: float) -> str:
    mantissa, exponent = f"{learning_rate:.0e}".split("e")
    return f"lr{mantissa}e{int(exponent)}"


class PathMap:
    """Rewrite paths recorded under ``deploy`` to the tree under ``local``."""

    def __init__(self, deploy: Path, local: Path) -> None:
        self.deploy, self.local = Path(deploy), Path(local)

    def to_local(self, value: str) -> Path:
        path = Path(value)
        return self.local / path.relative_to(self.deploy) if path.is_relative_to(self.deploy) else path

    def to_deploy(self, path: Path) -> str:
        path = Path(path).resolve()
        return str(self.deploy / path.relative_to(self.local.resolve())) if path.is_relative_to(self.local.resolve()) else str(path)

    def localize(self, value: object) -> object:
        if isinstance(value, str) and (value == str(self.deploy) or value.startswith(str(self.deploy) + "/")):
            return str(self.to_local(value))
        if isinstance(value, list):
            return [self.localize(item) for item in value]
        if isinstance(value, dict):
            return {key: self.localize(item) for key, item in value.items()}
        return value


# Source manifest ---------------------------------------------------------------------------

def source_files(root: Path) -> list[str]:
    files = sorted({str(path.relative_to(root)) for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()})
    require(files, "source globs found no files")
    return files


def build_source_manifest(root: Path) -> Json:
    root = Path(root).resolve()
    files = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256(root / name)} for name in source_files(root)]
    return {"schema": SOURCE_SCHEMA, "root": "repository", "globs": list(SOURCE_GLOBS), "files": files,
            "evaluator_files_hashed_per_gate": list(EVALUATOR_FILES)}


def verify_source(record: Mapping[str, object]) -> Json:
    path = Path(_str(record.get("manifest_path"), "source manifest path"))
    root = Path(_str(record.get("repository_root"), "source repository root"))
    require(path.is_file() and sha256(path) == record.get("manifest_sha256"), "source manifest differs")
    manifest = _object(json.loads(path.read_text()), "source manifest")
    require(manifest.get("schema") == SOURCE_SCHEMA, "source manifest schema differs")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "source manifest lists no files")
    assert isinstance(files, list)
    for item in files:
        item = _object(item, "source manifest item")
        target = root / _str(item.get("path"), "source path")
        require(target.is_file() and target.stat().st_size == item.get("bytes") and sha256(target) == item.get("sha256"),
                f"source differs:{item.get('path')}")
    current = set(source_files(root))
    require(current == {_object(item, "source manifest item")["path"] for item in files},
            "source file set differs from the bound manifest")
    return manifest


def evaluator_hashes(root: Path) -> dict[str, str]:
    return {name: sha256(Path(root) / name) for name in EVALUATOR_FILES}


# Template --------------------------------------------------------------------------------

def build_length_profile(rows: Path, rows_sha256: str) -> Json:
    """The ``sepalith.full-weight-edit-sft-length-profile.v1`` record the trainer checks for a row file."""
    require(sha256(rows) == rows_sha256, "TRAIN rows differ from the expected hash")
    lengths, targets = [], []
    with rows.open() as stream:
        for line in stream:
            row = json.loads(line)
            lengths.append(len(row["input_ids"]))
            targets.append(len(row["input_ids"]) - row["target_start"])
    thresholds = (2048, 4096, 8192, 16384)
    return {"schema": "sepalith.full-weight-edit-sft-length-profile.v1", "rows_sha256": rows_sha256, "rows": len(lengths),
            "sequence_tokens": {"sum": sum(lengths), "max": max(lengths),
                                "over_threshold_counts": {str(t): sum(1 for n in lengths if n > t) for t in thresholds}},
            "supervised_target_tokens_including_eos": {"sum": sum(targets), "max": max(targets)},
            "observed_fit_without_truncation": max(lengths) <= thresholds[-1]}


def verify_data_queue_v2(record: Mapping[str, object], cohort: Mapping[str, object]) -> Json:
    """The data queue must name this recipe's row file as its accepted active cohort."""
    path = Path(_str(record.get("path"), "data queue path"))
    require(path.is_file() and sha256(path) == record.get("sha256"), "editing data queue binding differs")
    queue = _object(json.loads(path.read_text()), "data queue")
    require(queue.get("schema") == "sepalith.sft11.full-weight-edit-data-queue.v2", "data queue schema differs")
    active = _object(queue.get("active_cohort"), "active cohort")
    rows = _object(cohort.get("rows"), "cohort rows")
    require(active.get("status") == "accepted_root_reviewed" and active.get("rows") == cohort.get("unique_rows")
            and active.get("rows_sha256") == rows.get("sha256"), "data queue does not accept this row file")
    require(queue.get("additional_roxy_rows") == record.get("additional_roxy_rows")
            and queue.get("additional_roxy_status") == record.get("additional_roxy_status"), "recipe data queue summary differs")
    return queue


def build_template(*, rows: Path, rows_sha256: str, schedule: Path, materialization: Path, length_profile: Path,
                   data_queue: Path, optimizer_dispatch: Path, dev75: Path, dev250: Path, dev250_manifest: Path,
                   paths: PathMap, cohort_id: str = "DAT10-finish-source-repair-v3-full15006") -> Json:
    """Assemble the pre-registered template from the cohort, schedule and panels."""
    require(sha256(rows) == rows_sha256, "TRAIN rows differ from the expected hash")
    plan = _object(json.loads(schedule.read_text()), "schedule")
    coverage = _object(plan["coverage"], "schedule coverage")
    policy = _object(plan["policy"], "schedule policy")
    tokens = _object(_object(plan["token_denominators"], "denominators")["unique"], "unique denominators")
    updates = plan["max_steps"]
    require(type(updates) is int and policy.get("schedule_version") == SCHEDULE_VERSION
            and policy.get("gate_updates") == gate_updates(updates) and policy.get("seed") == SEED, "schedule is not schedule v2")
    require(_object(plan["input"], "schedule input").get("sha256") == rows_sha256, "schedule is for another row file")
    assert isinstance(updates, int)
    gates = gate_updates(updates)
    profile = _object(json.loads(length_profile.read_text()), "length profile")
    require(profile.get("rows_sha256") == rows_sha256, "length profile is for another row file")
    dispatch = _object(json.loads(optimizer_dispatch.read_text()), "optimizer dispatch")
    dev250_cases = sum(1 for line in dev250.read_text().splitlines() if line.strip())
    dev75_cases = sum(1 for line in dev75.read_text().splitlines() if line.strip())
    record = paths.to_deploy
    dev75_record = {"path": str(dev75), "sha256": sha256(dev75)}
    return {
        "schema": TEMPLATE_SCHEMA,
        "status": "template_parent_and_learning_rate_unbound",
        "launch_authorized": False,
        "parent": None,
        "seed": SEED,
        "cohort": {
            "id": cohort_id,
            "rows": {"path": str(rows), "sha256": rows_sha256, "bytes": rows.stat().st_size},
            "draw_schedule": file_record(schedule, record(schedule)),
            "manifest": file_record(materialization, record(materialization)),
            "length_profile": file_record(length_profile, record(length_profile)),
            "max_sequence_tokens": 16384,
            "unique_rows": coverage["eligible_rows"],
            "updates": updates,
            "input_tokens": tokens["input_tokens"],
            "loss_tokens": tokens["supervised_target_tokens_including_eos"],
            "body_tokens": tokens["target_body_tokens"],
            "observed_max_sequence_tokens": _object(profile["sequence_tokens"], "profile")["max"],
            "observed_max_target_tokens_including_eos": _object(profile["supervised_target_tokens_including_eos"], "profile")["max"],
            "target_truncation": False,
            "scope": "accepted_repaired_15006_only_additional_roxy_pending",
            "schedule_semantics": {
                "version": SCHEDULE_VERSION, "edit_per_batch": policy["edit_per_batch"],
                "noop_per_batch": policy["noop_per_batch"], "noop_share": coverage["noop_share"],
                "exposure_histogram_by_pool": coverage["exposure_histogram_by_pool"],
                "replay_draws_by_reason": coverage["replay_draws_by_reason"],
                "gate_updates": gates,
            },
        },
        "data_queue": {**file_record(data_queue, record(data_queue)), "additional_roxy_rows": 10017,
                       "additional_roxy_status": "pending_root_admission"},
        "development": {
            "cases": dev75_cases,
            "final_set_access": False,
            "generation_budget": {"default_candidate": GENERATION_MAX_NEW_TOKENS,
                                  "policy": "no silent truncation; report cap hit, response completeness, stop token, all-case metrics, and complete-case metrics",
                                  "prepared_max_new_tokens_bound": 8192, "root_selected": True},
            "generation_evaluation": "required_for_edit_and_noop_quality",
            "teacher_forced_diagnostic": "separate_metric_not_edit_accuracy",
            "in_callback_policy": "mandatory optimizer-boundary stop; DEV75 and DEV250 generation run in a separate process after the trainer exits",
            "panel": dev75_record,
        },
        "dev_panels": {
            "dev75": {**dev75_record, "cases": dev75_cases},
            "dev250": {"path": str(dev250), "sha256": sha256(dev250), "cases": dev250_cases,
                       "manifest": {"path": str(dev250_manifest), "sha256": sha256(dev250_manifest)}},
        },
        "expected_model": {"dtype": "bfloat16", "parameter_tensors": 381, "parameters": 2516756480, "peft": False},
        "optimizer_dispatch": {**file_record(optimizer_dispatch, record(optimizer_dispatch)),
                               "ordered_rows_sha256": dispatch["ordered_rows_sha256"]},
        "pre_registered": {
            "card": "C04",
            "optimizer_mechanics": CPT_OPTIMIZER_MECHANICS,
            "side_lr_ratio": SIDE_LR_RATIO,
            "scheduler": "cosine_with_min_lr",
            "warmup_steps": WARMUP_STEPS,
            "min_lr_rate": MIN_LR_RATE,
            "weight_decay": WEIGHT_DECAY,
            "learning_rate_upper_bound": LR_UPPER_BOUND,
            "learning_rate_upper_bound_rationale": LR_UPPER_BOUND_RATIONALE,
            "cpt_hidden_learning_rate": CPT_HIDDEN_LR,
            "pilot_learning_rates": list(PILOT_LEARNING_RATES),
            "gate_updates": gates,
            "gate_rule": "R1 (docs/training/plan-20260927/RULES.md)",
            "parent_baseline": BASELINE_PATH,
            "arms_differ_only_in": "learning_rate (hidden LR) and side LR = LR / 10",
        },
        "retention": {"durable_archive_latest": 2, "preserve_selected_milestones": True},
    }


def admission_template(template_sha256: str) -> Json:
    """An admission skeleton whose example values are unusable on purpose."""
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "EXAMPLE_ONLY_TEMPLATE_NOT_A_DECISION",
        "launch_authorized": False,
        "template_sha256": template_sha256,
        "arm": {"name": "EXAMPLE_ONLY_lr1e-5", "learning_rate": "EXAMPLE_ONLY_1e-05",
                "purpose": "EXAMPLE_ONLY_describe_the_arm"},
        "parent": {"kind": "cpt_full_weight_checkpoint", "candidate_id": "EXAMPLE_ONLY_CPT-11586", "step": "EXAMPLE_ONLY_11586",
                   "path": "EXAMPLE_ONLY_/ABSOLUTE/CHECKPOINT", "selection_status": "EXAMPLE_ONLY_user_selected_for_edit_sft_parent",
                   "files": {name: "FILL_SHA256" for name in PARENT_FILES}},
        "decided_by": "EXAMPLE_ONLY_who_decided_and_where_it_is_recorded",
        "note": "Every EXAMPLE_ONLY or FILL value must be replaced; the binder refuses any that remain.",
    }


def parent_admission(parent: Path, *, step: int, learning_rate: float, arm: str, purpose: str,
                     template_sha256: str, decided_by: str) -> Json:
    files = {name: sha256(parent / name) for name in PARENT_FILES}
    return {"schema": ADMISSION_SCHEMA, "status": "admitted", "launch_authorized": True, "template_sha256": template_sha256,
            "arm": {"name": arm, "learning_rate": learning_rate, "purpose": purpose},
            "parent": {"kind": "cpt_full_weight_checkpoint", "candidate_id": f"CPT-{step}", "step": step, "path": str(parent),
                       "selection_status": "user_selected_for_edit_sft_parent", "files": files},
            "decided_by": decided_by}


def _markers(value: object, where: str = "admission") -> list[str]:
    if isinstance(value, str):
        return [where] if any(marker in value for marker in EXAMPLE_MARKERS) else []
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _markers(item, f"{where}.{key}")]
    if isinstance(value, list):
        return [hit for index, item in enumerate(value) for hit in _markers(item, f"{where}[{index}]")]
    return []


# Binding ---------------------------------------------------------------------------------

def arm_outputs(arm: str) -> Json:
    return {
        "trainer": f"{CHECKPOINT_BASE}/training/SFT-v2-{arm}/runtime",
        "archive": f"{CHECKPOINT_BASE}/checkpoints/SFT-v2-{arm}",
        "graceful_stop": f"{STATE_ROOT}/control/SFT-v2-{arm}-save-stop.json",
        "evidence_directory": f"{GATE_ROOT}/SFT-v2-{arm}",
    }


def runtime_for(learning_rate: float, evidence_directory: str, updates: int = UPDATES) -> Json:
    gates = gate_updates(updates)
    optimizer = dict(CPT_OPTIMIZER_MECHANICS)
    optimizer.update(hidden_lr=learning_rate, side_lr=float(f"{learning_rate * SIDE_LR_RATIO:.6g}"))
    return {
        "micro_batch": 1, "gradient_accumulation": 16, "effective_batch": EFFECTIVE_BATCH, "max_steps": updates,
        "learning_rate": learning_rate, "optimizer": optimizer, "scheduler": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": MIN_LR_RATE}, "warmup_steps": WARMUP_STEPS, "weight_decay": WEIGHT_DECAY,
        # Gate stops are forced by the mandatory-gate control; the save cadence only covers the terminal update.
        "checkpoint_every": updates, "evaluation_steps": gates, "selected_milestones": gates,
        "mandatory_dev_gate_steps": gates, "telemetry_every": 1, "gate_evidence_directory": evidence_directory,
        "development_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    }


def bind(template_path: Path, admission_path: Path, output: Path, *, source_manifest: Path, paths: PathMap) -> Json:
    template = _object(json.loads(template_path.read_text()), "template")
    admission = _object(json.loads(admission_path.read_text()), "admission")
    require(template.get("schema") == TEMPLATE_SCHEMA and template.get("parent") is None
            and template.get("launch_authorized") is False, "template differs")
    hits = _markers(admission)
    require(not hits, f"admission still holds EXAMPLE_ONLY or FILL values: {hits[:5]}")
    require(admission.get("schema") == ADMISSION_SCHEMA and admission.get("status") == "admitted"
            and admission.get("launch_authorized") is True, "admission differs")
    require(admission.get("template_sha256") == sha256(template_path), "admission template hash differs")
    arm = _object(admission.get("arm"), "arm")
    name = _str(arm.get("name"), "arm name")
    require(_ARM.fullmatch(name), "arm name differs")
    learning_rate = arm.get("learning_rate")
    require(type(learning_rate) is float and 0 < learning_rate <= LR_UPPER_BOUND,
            f"editing LR must be a float in (0, {LR_UPPER_BOUND}]")
    assert isinstance(learning_rate, float)
    parent = _object(admission.get("parent"), "parent")
    require(parent.get("kind") == "cpt_full_weight_checkpoint"
            and parent.get("selection_status") == "user_selected_for_edit_sft_parent", "selected parent differs")
    files = _object(parent.get("files"), "parent files")
    require(set(files) == set(PARENT_FILES), "parent file inventory differs")
    require(files["tokenizer.json"] == TOKENIZER_SHA256, "parent tokenizer differs")
    updates = _object(template["cohort"], "cohort")["updates"]
    require(type(updates) is int, "template horizon missing")
    assert isinstance(updates, int)
    outputs = arm_outputs(name)
    evidence = str(outputs.pop("evidence_directory"))
    bound = copy.deepcopy(template)
    bound.update({
        "schema": BOUND_SCHEMA,
        "status": "bound_not_launched",
        "launch_authorized": True,
        "parent": parent,
        "arm": arm,
        "runtime": runtime_for(learning_rate, evidence, updates),
        "dev_gate": {
            "mandatory_steps": gate_updates(updates),
            "evidence_directory": evidence,
            "panel": _object(template["development"], "development")["panel"],
            "panels": template["dev_panels"],
            "rule": "R1",
            "parent_baseline": {"path": BASELINE_PATH},
            "continuation_policy": "automatic_rule_R1_with_human_override_file",
            "override_file_pattern": f"{evidence}/override-step-<step>.json",
            "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        },
        "outputs": outputs,
        "checkpoint_storage": {"mode": "e_same_filesystem_atomic", "c_hot_stage": "not_admitted",
                               "required_same_filesystem": True, "trainer_root": outputs["trainer"],
                               "archive_root": outputs["archive"]},
        "source": {"manifest_path": paths.to_deploy(source_manifest), "manifest_sha256": sha256(source_manifest),
                   "repository_root": str(paths.deploy)},
        "root_admission": {"path": paths.to_deploy(admission_path), "sha256": sha256(admission_path)},
        "template": {"path": paths.to_deploy(template_path), "sha256": sha256(template_path)},
        # Replaces the template's v1-style "latest 2" entry: R7 makes any checkpoint deletion stop-for-user.
        "retention": dict(RETENTION),
    })
    validate_pre_registered(bound)
    write_new(output, bound)
    return bound


def validate_pre_registered(recipe: Mapping[str, object]) -> None:
    """Refuse a v2 recipe whose schedule, optimizer or gates differ from card C04."""
    runtime = _object(recipe.get("runtime"), "runtime")
    learning_rate = runtime.get("learning_rate")
    require(type(learning_rate) is float and 0 < learning_rate <= LR_UPPER_BOUND, "editing LR exceeds the v2 bound")
    assert isinstance(learning_rate, float)
    optimizer = _object(runtime.get("optimizer"), "optimizer")
    require({key: optimizer.get(key) for key in CPT_OPTIMIZER_MECHANICS} == CPT_OPTIMIZER_MECHANICS
            and set(optimizer) == set(CPT_OPTIMIZER_MECHANICS) | {"hidden_lr", "side_lr"}, "optimizer mechanics differ from CPT")
    side = optimizer.get("side_lr")
    require(optimizer.get("hidden_lr") == learning_rate and isinstance(side, float)
            and math.isclose(side, learning_rate * SIDE_LR_RATIO, rel_tol=1e-9), "hidden/side LR differ")
    require(runtime.get("scheduler") == "cosine_with_min_lr" and runtime.get("lr_scheduler_kwargs") == {"min_lr_rate": MIN_LR_RATE}
            and runtime.get("warmup_steps") == WARMUP_STEPS and runtime.get("weight_decay") == WEIGHT_DECAY, "LR schedule differs")
    require(runtime.get("micro_batch") == 1 and runtime.get("gradient_accumulation") == 16
            and runtime.get("effective_batch") == EFFECTIVE_BATCH, "batch geometry differs")
    cohort = _object(recipe.get("cohort"), "cohort")
    updates = cohort.get("updates")
    require(type(updates) is int and runtime.get("max_steps") == updates and runtime.get("checkpoint_every") == updates,
            "horizon differs from the cohort schedule")
    assert isinstance(updates, int)
    require(runtime.get("evaluation_steps") == gate_updates(updates) == runtime.get("mandatory_dev_gate_steps")
            == runtime.get("selected_milestones"), "gate updates are not the quarters of the horizon")
    require(recipe.get("seed") == SEED, "seed differs")
    require(recipe.get("retention") == RETENTION, "checkpoint retention must keep every checkpoint")
    require(_object(cohort.get("schedule_semantics"), "semantics").get("version") == SCHEDULE_VERSION,
            "cohort is not bound to schedule v2")
    gate = _object(recipe.get("dev_gate"), "dev_gate")
    panels = _object(gate.get("panels"), "panels")
    require(set(panels) == {"dev75", "dev250"} and _object(panels["dev75"], "dev75").get("sha256") == _object(gate.get("panel"), "panel").get("sha256"),
            "DEV panels differ")
    for name, panel in panels.items():
        panel = _object(panel, name)
        cases = panel.get("cases")
        require(isinstance(cases, int) and cases > 0, f"{name} case count missing")
    require(_object(panels["dev250"], "dev250").get("manifest") is not None, "DEV250 manifest binding missing")


def load_bound_v2(path: str | os.PathLike[str]) -> Json:
    """Fail-closed load of a v2 bound recipe; the trainer and the evaluator call this."""
    from sepalith.training.eval.milestone_gate import mandatory_steps
    from sepalith.training.sft.full_weight_edit_sft import validate_context_and_storage

    recipe = _object(json.loads(Path(path).read_text()), "bound recipe")
    require(recipe.get("schema") == BOUND_SCHEMA and recipe.get("status") == "bound_not_launched"
            and recipe.get("launch_authorized") is True, "bound recipe is not a v2 bound recipe")
    for key in ("root_admission", "template"):
        record = _object(recipe.get(key), key)
        target = Path(_str(record.get("path"), f"{key} path"))
        require(target.is_file() and sha256(target) == record.get("sha256"), f"{key} differs")
    verify_source(_object(recipe.get("source"), "source"))
    verify_data_queue_v2(_object(recipe.get("data_queue"), "data_queue"), _object(recipe.get("cohort"), "cohort"))
    validate_pre_registered(recipe)
    runtime = _object(recipe["runtime"], "runtime")
    require(_int(runtime["max_steps"], "max_steps") % _int(runtime["checkpoint_every"], "checkpoint_every") == 0,
            "checkpoint cadence differs")
    mandatory_steps(recipe)
    validate_context_and_storage(recipe)
    cohort = _object(recipe["cohort"], "cohort")
    gate = _object(recipe["dev_gate"], "dev_gate")
    panels = _object(gate["panels"], "panels")
    dev250 = _object(panels["dev250"], "dev250")
    records = [cohort["rows"], cohort["draw_schedule"], cohort["manifest"], cohort["length_profile"],
               _object(recipe["development"], "development")["panel"], panels["dev75"], dev250, dev250["manifest"]]
    for record in records:
        record = _object(record, "input")
        target = Path(_str(record.get("path"), "input path"))
        require(target.is_file() and sha256(target) == record.get("sha256"), f"input differs:{target}")
    parent = _object(recipe["parent"], "parent")
    require(parent.get("kind") == "cpt_full_weight_checkpoint", "v2 parent must be a full-weight CPT checkpoint")
    root = Path(_str(parent.get("path"), "parent path"))
    require(root.is_dir(), "parent path missing")
    for name, expected in _object(parent["files"], "parent files").items():
        require((root / name).is_file() and sha256(root / name) == expected, f"parent differs:{name}")
    dispatch = _object(recipe["optimizer_dispatch"], "optimizer dispatch")
    target = Path(_str(dispatch.get("path"), "dispatch path"))
    require(target.is_file() and sha256(target) == dispatch.get("sha256"), "optimizer dispatch differs")
    return recipe


# CLI -------------------------------------------------------------------------------------

def _paths(args: argparse.Namespace) -> PathMap:
    return PathMap(Path(args.deploy_root), Path(args.local_root).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy-root", default="/home/m0hawk/sandbox/Sepalith")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[6]))
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    for name in ("rows", "schedule", "materialization", "length-profile", "data-queue", "optimizer-dispatch",
                 "dev75", "dev250", "dev250-manifest", "output"):
        template.add_argument(f"--{name}", type=Path, required=True)
    template.add_argument("--rows-sha256", required=True)
    template.add_argument("--cohort-id", default="DAT10-finish-source-repair-v3-full15006")
    profile = sub.add_parser("length-profile")
    profile.add_argument("--rows", type=Path, required=True)
    profile.add_argument("--rows-sha256", required=True)
    profile.add_argument("--output", type=Path, required=True)
    skeleton = sub.add_parser("admission-template")
    skeleton.add_argument("--template", type=Path, required=True)
    skeleton.add_argument("--output", type=Path, required=True)
    admit = sub.add_parser("admission")
    admit.add_argument("--template", type=Path, required=True)
    admit.add_argument("--parent", type=Path, required=True)
    admit.add_argument("--parent-step", type=int, required=True)
    admit.add_argument("--learning-rate", type=float, required=True)
    admit.add_argument("--arm")
    admit.add_argument("--purpose", required=True)
    admit.add_argument("--decided-by", required=True)
    admit.add_argument("--output", type=Path, required=True)
    source = sub.add_parser("source-manifest")
    source.add_argument("--output", type=Path, required=True)
    binder = sub.add_parser("bind")
    binder.add_argument("--template", type=Path, required=True)
    binder.add_argument("--admission", type=Path, required=True)
    binder.add_argument("--source-manifest", type=Path, required=True)
    binder.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="load a bound recipe with deploy paths mapped to the local tree")
    verify.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.command == "template":
        value = build_template(rows=args.rows, rows_sha256=args.rows_sha256, schedule=args.schedule,
                               materialization=args.materialization, length_profile=args.length_profile,
                               data_queue=args.data_queue, optimizer_dispatch=args.optimizer_dispatch, dev75=args.dev75,
                               dev250=args.dev250, dev250_manifest=args.dev250_manifest, paths=paths,
                               cohort_id=args.cohort_id)
        write_new(args.output, value)
    elif args.command == "length-profile":
        write_new(args.output, build_length_profile(args.rows, args.rows_sha256))
    elif args.command == "admission-template":
        write_new(args.output, admission_template(sha256(args.template)))
    elif args.command == "admission":
        name = args.arm or arm_tag(args.learning_rate)
        write_new(args.output, parent_admission(args.parent, step=args.parent_step, learning_rate=args.learning_rate, arm=name,
                                                purpose=args.purpose, template_sha256=sha256(args.template),
                                                decided_by=args.decided_by))
    elif args.command == "source-manifest":
        write_new(args.output, build_source_manifest(paths.local))
    elif args.command == "bind":
        bind(args.template, args.admission, args.output, source_manifest=args.source_manifest, paths=paths)
    else:
        recipe = paths.localize(json.loads(args.recipe.read_text()))
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "recipe.json"
            local.write_text(json.dumps(recipe))
            load_bound_v2(local)
        print(json.dumps({"status": "verified", "recipe": str(args.recipe), "sha256": sha256(args.recipe)}))
        return 0
    print(json.dumps({"status": "written", "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
