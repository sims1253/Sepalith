"""Bind the C20 protocol to the frozen CPU artifacts and main checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import cast

from sepalith.training.grpo_exec import pool
from sepalith.training.sft import c20_data as data, recipe_v2

SCHEMA = "sepalith.c20.full-weight-sft.v1"
MAIN = Path("/home/m0hawk/sandbox/Sepalith")
PYTHON = MAIN / ".venv-sft/bin/python"
PACKET = data.ROOT / "c20"
STATE = Path(recipe_v2.GATE_ROOT)
CAMPAIGN = Path(recipe_v2.CHECKPOINT_BASE)
PARENT = CAMPAIGN / "checkpoints/C19-grpo-exec-pilot-1/full/checkpoint-100"
C19_EVIDENCE = STATE / "C19-grpo-exec-pilot-1"
MEMORY_MIB = 14_500
DISK_FLOOR = 150_000_000_000
CHECKPOINT_BYTES = 19_000_000_000
LR = 2.5e-6
PILOT_STEPS = [100, 200, 300]
START = "c19-100"
SELECTED = "c20-selected"
COMPLETIONS = {
    "path": str(C19_EVIDENCE / "final-step-100/completions-c19-selected.jsonl"),
    "sha256": "3e73e1d130ce0320aa937ee22786eb0adbd7a6706c28f749c89baa56cf1055d0",
}
JOB_IDS = {"smoke": "C20-sft-finish-smoke", "pilot": "C20-sft-finish-pilot",
           "final": "C20-final-holdout"}


def require_disk(reserve: int = 0) -> None:
    pool.require(shutil.disk_usage(CAMPAIGN).free >= DISK_FLOOR + reserve,
                 "R7 disk floor: insufficient free space for the next full checkpoint")


def limit_cuda(memory_mib: int) -> None:
    import torch

    pool.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA device required")
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    budget = memory_mib * 1024**2
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), device=0)


def checkpoint_files() -> dict:
    manifest = PARENT / "campaign-manifest.json"
    digest = pool.sha256(manifest)
    pool.require(digest == "c6123c01a348ed1c2389603574545cf6b545986c57e2054c6879beff569b4a57",
                 "adopted C19 manifest differs")
    inventory = data.read(manifest)["files"]
    return {name: digest if name == manifest.name else inventory[name]["sha256"]
            for name in recipe_v2.PARENT_FILES}


def stages(mode: str) -> list[int]:
    return [2] if mode == "smoke" else PILOT_STEPS.copy()


def runtime(mode: str) -> dict:
    return {**recipe_v2.runtime_for(LR, str(STATE / JOB_IDS[mode]), data.UPDATES),
            "optimizer": {**recipe_v2.CPT_OPTIMIZER_MECHANICS, "hidden_lr": LR,
                          "side_lr": 2.5e-7}, "optimizer_state_offload": True,
            "checkpoint_every": 100, "evaluation_steps": stages(mode), "mandatory_dev_gate_steps": stages(mode),
            "selected_milestones": stages(mode), "cuda_memory_limit_mib": MEMORY_MIB}


def bind(repo: Path) -> None:
    packet = repo / PACKET
    template = data.read(repo / data.ROOT / "c04/bound/sft-v2-lr1e-5.json")
    c19 = data.read(repo / data.C19_RECIPE)
    validation_path = packet / "validation-manifest.json"
    mix_path = packet / "mix-manifest.json"
    validation = data.read(data.DATA / validation_path.name)
    mix = data.read(data.DATA / mix_path.name)
    pool.write_json_new(validation_path, validation)
    pool.write_json_new(mix_path, mix)
    tracked = recipe_v2.build_source_manifest(repo)
    extra_files = {}
    for pattern in (
        "packages/sepalith/src/sepalith/exec_env/*",
        "packages/sepalith/src/sepalith/exec_eval/*.py",
        "packages/sepalith/src/sepalith/training/eval/*.py",
        "packages/sepalith/src/sepalith/training/grpo_exec/*.py",
    ):
        for file in sorted(repo.glob(pattern)):
            if file.is_file():
                extra_files[str(file.relative_to(repo))] = pool.sha256(file)
    tracked["c20_evaluation_files"] = extra_files
    source_manifest = packet / "source-manifest.json"
    pool.write_json_new(source_manifest, tracked)
    source = {"manifest_path": str(MAIN / PACKET / source_manifest.name),
              "manifest_sha256": pool.sha256(source_manifest), "repository_root": str(MAIN)}
    parent = {"path": str(PARENT), "candidate_id": START, "files": checkpoint_files()}
    dev_gate = c19["dev_gate"]
    for mode in ("smoke", "pilot"):
        evidence = STATE / JOB_IDS[mode]
        archive = CAMPAIGN / "checkpoints" / JOB_IDS[mode]
        trainer = CAMPAIGN / "training" / JOB_IDS[mode] / "runtime"
        bound = {
            "schema": SCHEMA, "mode": mode, "source": source, "seed": data.SEED,
            "parent": parent, "arm": {"name": JOB_IDS[mode]},
            "cohort": {"id": "C20-sft-finish-v1", "allow_cpt_replay": True,
                       "rows": mix["rows"], "draw_schedule": mix["schedule"],
                       "length_profile": mix["length_profile"], "unique_rows": mix["distinct_rows"],
                       "updates": data.UPDATES, "max_sequence_tokens": data.CONTEXT,
                       **{k: mix[k] for k in ("input_tokens", "loss_tokens", "body_tokens")}},
            "validation_manifest": recipe_v2.file_record(validation_path, str(MAIN / PACKET / validation_path.name)),
            "mix_manifest": recipe_v2.file_record(mix_path, str(MAIN / PACKET / mix_path.name)),
            "validation": validation["tasks"], "development": template["development"],
            "data_queue": {"additional_roxy_rows": 0},
            "dev_gate": {k: dev_gate[k] for k in ("panel", "panels", "cpt", "generation_max_new_tokens")},
            "optimizer_dispatch": template["optimizer_dispatch"],
            "runtime": runtime(mode), "retention": recipe_v2.RETENTION,
            "outputs": {"archive": str(archive), "trainer": str(trainer), "evidence": str(evidence),
                        "graceful_stop": str(evidence / "save-stop.json")},
            "checkpoint_storage": {"mode": "e_same_filesystem_atomic", "c_hot_stage": "not_admitted",
                                   "required_same_filesystem": True, "trainer_root": str(trainer),
                                   "archive_root": str(archive)},
        }
        bound["dev_gate"]["mandatory_steps"] = stages(mode)
        bound["dev_gate"]["evidence_directory"] = str(evidence)
        pool.write_json_new(packet / f"{mode}.json", bound)
    final = {"tasks": c19["final_evaluation"]["holdout_tasks"], "completions": COMPLETIONS, "n_tasks": 3803}
    pool.write_json_new(packet / "final.json", final)
    previous: list[str] = []
    for mode, priority, minutes in (("smoke", 40, 45), ("pilot", 45, 720), ("final", 30, 240)):
        identifier = JOB_IDS[mode]
        recipe_path = MAIN / PACKET / f"{mode}.json"
        job = {"id": identifier, "cwd": str(MAIN), "env": [], "priority": priority,
               "estimated_minutes": minutes, "resumable": mode != "smoke", "requires": previous.copy(),
               "command": ["/usr/bin/env", f"PYTHONPATH={MAIN}/packages/sepalith/src", "OMP_NUM_THREADS=8",
                           "MKL_NUM_THREADS=8", "OPENBLAS_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=0",
                           "TOKENIZERS_PARALLELISM=false", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                           str(PYTHON), "-B", "-m", "sepalith.training.sft.c20_run", mode,
                           "--recipe", str(recipe_path), "--sha256",
                           pool.sha256(packet / recipe_path.name)]}
        pool.write_json_new(packet / "jobs" / f"{identifier}.json", job)
        previous = [identifier]


def verify(recipe: dict) -> dict:
    from sepalith.training.eval.milestone_gate import DEV250_PANEL_SHA, PANEL_SHA
    from sepalith.training.sft.full_weight_edit_sft import validate_context_and_storage

    mode = recipe["mode"]
    pool.require(mode in ("smoke", "pilot") and recipe["schema"] == SCHEMA, "C20 recipe differs")
    pool.require(recipe["runtime"] == runtime(mode) and recipe["seed"] == data.SEED
                 and recipe["dev_gate"]["mandatory_steps"] == stages(mode)
                 and recipe["retention"] == recipe_v2.RETENTION, "C20 runtime differs")
    source = recipe_v2.verify_source(recipe["source"])
    root = Path(recipe["source"]["repository_root"])
    for name, digest in cast(dict[str, str], source["c20_evaluation_files"]).items():
        data.checked({"path": str(root / name), "sha256": digest})
    validate_context_and_storage(recipe)
    validation = data.read(data.checked(recipe["validation_manifest"]))
    mix = data.read(data.checked(recipe["mix_manifest"]))
    pool.require(validation["tasks"] == recipe["validation"] and
                 mix["sources"]["validation_manifest_sha256"] == recipe["validation_manifest"]["sha256"],
                 "validation and mix linkage differs")
    cohort = recipe["cohort"]
    pool.require(cohort["rows"] == mix["rows"] and cohort["draw_schedule"] == mix["schedule"]
                 and cohort["length_profile"] == mix["length_profile"] and cohort["allow_cpt_replay"] is True,
                 "frozen cohort differs")
    panels = recipe["dev_gate"]["panels"]
    pool.require(panels["dev75"]["sha256"] == PANEL_SHA and panels["dev250"]["sha256"] == DEV250_PANEL_SHA
                 and recipe["development"]["panel"]["sha256"] == PANEL_SHA, "DEV panel differs")
    for record in (recipe["validation"], cohort["rows"], cohort["draw_schedule"], cohort["length_profile"],
                   recipe["optimizer_dispatch"], panels["dev75"], panels["dev250"], panels["dev250"]["manifest"],
                   recipe["dev_gate"]["cpt"]["panel"]):
        data.checked(record)
    pool.require(recipe["parent"] == {"path": str(PARENT), "candidate_id": START, "files": checkpoint_files()},
                 "C19 parent differs")
    for name, digest in recipe["parent"]["files"].items():
        data.checked({"path": str(PARENT / name), "sha256": digest})
    return recipe


def dry_run(repo: Path) -> None:
    from sepalith.ops.night_queue import validate_job
    from sepalith.training.grpo_exec.c19_night import guard
    from sepalith.training.sft.full_weight_edit_sft import _validated_cohort

    guard()
    pool.require(os.access(PYTHON, os.X_OK), "main checkout interpreter missing")
    paths = recipe_v2.PathMap(MAIN, repo)
    for mode, identifier in JOB_IDS.items():
        packet = repo / PACKET
        job = data.read(packet / "jobs" / f"{identifier}.json")
        validate_job(job)
        recipe_path = packet / f"{mode}.json"
        pool.require(job["cwd"] == str(MAIN) and job["command"][1] == f"PYTHONPATH={MAIN}/packages/sepalith/src"
                     and job["command"][-3] == str(MAIN / PACKET / recipe_path.name)
                     and job["command"][-1] == pool.sha256(recipe_path) and str(PYTHON) in job["command"],
                     "queue job path or binding differs")
        recipe = data.read(recipe_path)
        if mode == "final":
            # The frozen tasks are opened only after the pilot passes in the queued final job.
            pool.require(Path(recipe["tasks"]["path"]).is_file() and recipe["completions"] == COMPLETIONS
                         and Path(COMPLETIONS["path"]).is_file(), "final inputs missing")
        else:
            local = paths.localize(recipe)
            assert isinstance(local, dict)
            _, result = _validated_cohort(verify(local))
            print(json.dumps({"mode": mode, **result}, sort_keys=True))
    require_disk(7 * CHECKPOINT_BYTES)
    print("Main-bound jobs, R, data, parent files and disk verified.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "bind", "verify"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(data.build(args.repo), indent=2))
    elif args.command == "bind":
        bind(args.repo)
    else:
        dry_run(args.repo)


if __name__ == "__main__":
    main()
