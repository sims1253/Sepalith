#!/usr/bin/env python3
"""C18 end of pilot by user decision: DEV250 of the chosen checkpoint and the pilot summary the verdict reads.

The driver selects a checkpoint and runs its DEV250 only when the pilot ends by rule (rollback,
plateau, update 1,000). When the user ends the pilot instead (``user-stop-step-<n>.json`` in the
evidence directory), this module does the same two steps for the user's checkpoint:

- ``dev250`` (night job, GPU through a child process): DEV250 on the checkpoint with the driver's own
  path (``night.ExecGRPOArm.selected_dev250``: the C18 DEV evaluator under ``cuda_host_guard``, bound to
  the pilot recipe, written to ``<evidence>/selection/``), then ``summary``;
- ``summary`` (CPU): ``<archive>/pilot-summary.json`` with the user-stop selection, the start metrics,
  the checkpoint's validation, CPT and DEV250 results, so that
  ``python -m sepalith.training.grpo_exec.jobs verdict`` can produce the pre-registered verdict once
  the frozen hold-out job has finished.

The selection is the user's (recorded in the user-stop file), not the pre-registered best-validation or
plateau rule; the summary says so.

    python -m sepalith.training.grpo_exec.final dev250 --recipe <pilot recipe> --step 100
    python -m sepalith.training.grpo_exec.final summary --recipe <pilot recipe> --step 100
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import signal
import time
from typing import cast

from sepalith.training.grpo import state
from sepalith.training.grpo.night import counts_from
from sepalith.training.grpo_exec.night import ExecGRPOArm
from sepalith.training.sft.night_arm import (
    EXIT_DEFERRED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_NEEDS_USER,
    DriverError,
    EvaluationFailed,
    Settings,
    read_json,
    sha256,
    write_json,
)

Json = dict[str, object]
SUMMARY_SCHEMA = "sepalith.c18.grpo-exec-pilot-summary.v1"
USER_STOP_SCHEMA = "sepalith.c18.user-stop.v1"


def user_stop(arm: ExecGRPOArm, step: int) -> tuple[Path, Json]:
    path = arm.evidence_dir / f"user-stop-step-{step}.json"
    if not path.is_file():
        raise DriverError(f"no user decision for step {step}: {path}")
    value = read_json(path)
    if value.get("schema") != USER_STOP_SCHEMA or value.get("step") != step \
            or Path(str(value.get("selected_checkpoint"))) != arm.checkpoint(step):
        raise DriverError(f"{path} does not select checkpoint {step}")
    return path, value


def dev250_result(arm: ExecGRPOArm, step: int) -> Path | None:
    decision = arm.latest_decision(step)
    entries = cast(Json, (decision or {}).get("results") or {})
    if "dev250" in entries:
        return Path(str(cast(Json, entries["dev250"])["path"]))
    done = [p for p in sorted((arm.evidence_dir / "selection").glob(f"dev250-step-{step}.*.json"))
            if not p.name.endswith(".admission.json")]
    return done[-1] if done else None


def summary(arm: ExecGRPOArm, step: int) -> Json:
    """``pilot-summary.json`` for a user-stop selection; needs the checkpoint's evaluation decision and DEV250."""
    stop_path, stop = user_stop(arm, step)
    state.verify_checkpoint(arm.checkpoint(step), arm.recipe_sha)
    decision = arm.latest_decision(step)
    if decision is None or decision.get("evaluation_complete") is not True:
        raise DriverError(f"evaluation {step} has no complete decision")
    metrics = cast(Json, decision["metrics"])
    dev250 = dev250_result(arm, step)
    if dev250 is None:
        raise DriverError(f"checkpoint {step} has no DEV250 result yet")
    manifest = sha256(arm.checkpoint(step) / "campaign-manifest.json")
    problems = arm.result_problems("dev250", dev250, step, manifest)
    if problems:
        raise DriverError(f"DEV250 result does not bind: {problems}")
    start = arm.start_reference()
    score = counts_from(cast(Json, read_json(dev250)["summary"]), "dev250")["score250"]
    accepted = arm.accepted(step)
    value: Json = {
        "schema": SUMMARY_SCHEMA, "arm": cast(Json, arm.recipe["arm"])["name"], "status": "user_stop", "last_evaluation": step,
        "selection_rule": ("user decision (user-stop file), not the pre-registered best-validation or plateau rule: "
                           f"{stop.get('reason')}"),
        "user_stop": {"path": str(stop_path), "sha256": sha256(stop_path)},
        "start": {"checkpoint": cast(Json, arm.recipe["parent"])["path"], "metrics": start},
        "accepted": accepted, "evaluation_decision": {"outcome": decision.get("outcome"), "path": str(arm.decisions(step)[-1])},
        "updates_completed": (arm.updates() or [{"update": 0}])[-1]["update"],
        "selected": {"step": step, "checkpoint": str(arm.checkpoint(step)), "val_all_pass": metrics["val_all_pass"],
                     "val_mean_reward": metrics["val_mean_reward"], "cpt_nll": metrics["cpt_nll"],
                     "cpt_rise_vs_start": float(cast(float, metrics["cpt_nll"])) - float(cast(float, start["cpt_nll"])),
                     "dev250": {"path": str(dev250), "sha256": sha256(dev250), "score250": score}},
        "result_checkpoint": str(arm.checkpoint(step)),
        "final_evaluation": {"job": str(arm.evidence_dir / "final-holdout-job.json"),
                             "verdict": "python -m sepalith.training.grpo_exec.jobs verdict --summary <this file> --metrics "
                                        f"{arm.evidence_dir / 'final' / 'metrics.json'}"},
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(arm.archive / "pilot-summary.json", value)
    return value


def run_dev250(arm: ExecGRPOArm, step: int) -> int:
    user_stop(arm, step)
    result = arm.selected_dev250(step)
    if result == "defer":
        return arm.finish(EXIT_DEFERRED, f"DEV250 of checkpoint {step} deferred (too close to the deadline or stopped)")
    value = summary(arm, step)
    selected = cast(Json, value["selected"])
    return arm.finish(EXIT_DONE, f"C18 checkpoint {step}: DEV250 score250 {cast(Json, selected['dev250'])['score250']} "
                                 f"(803: {cast(Json, cast(Json, value['start'])['metrics'])['score250']}); pilot-summary.json written",
                      arm_result={"status": "user_stop", "gate": step, "result_checkpoint": selected["checkpoint"]})


def dev250_job(code_root: Path, recipe: Path, step: int, python: Path) -> Json:
    """Night job: ``code_root`` supplies cwd and PYTHONPATH (a worktree until merged), ``recipe`` stays the bound pilot recipe."""
    env = ("PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=0", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "OMP_NUM_THREADS=8",
           "MKL_NUM_THREADS=8", "TOKENIZERS_PARALLELISM=false", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    return {"id": f"C18-final-dev250-step-{step}", "cwd": str(code_root), "env": [], "priority": 30, "resumable": False,
            "estimated_minutes": 25, "requires": [],
            "command": ["/usr/bin/env", f"PYTHONPATH={code_root}/packages/sepalith/src", *env, str(python), "-B", "-m",
                        "sepalith.training.grpo_exec.final", "dev250", "--recipe", str(recipe), "--step", str(step)],
            "note": (f"C18 user-stop result (checkpoint {step}): DEV250 with the driver's evaluator (bound to the pilot recipe, "
                     "cuda:0 under cuda_host_guard), then pilot-summary.json for the pre-registered verdict (secondary endpoint "
                     "score250 >= 105). Exit 0 done, 75 deferred, 3 an input check failed, 1 evaluator failure.")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("dev250", "summary", "write-job"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--eval-budget-seconds", type=float, default=2400.0)
    parser.add_argument("--no-guard", action="store_true", help="run children without cuda_host_guard (tests only)")
    parser.add_argument("--code-root", type=Path, help="write-job: checkout for cwd and PYTHONPATH")
    parser.add_argument("--python", type=Path, default=Path("/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python"))
    parser.add_argument("--output", type=Path, help="write-job: job file")
    args = parser.parse_args(argv)
    if args.command == "write-job":
        job = dev250_job(args.code_root.resolve(), args.recipe.resolve(), args.step, args.python)
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"job": str(args.output)}))
        return EXIT_DONE
    settings = Settings.from_env(args.recipe, guard=not args.no_guard, eval_budget_seconds=args.eval_budget_seconds)
    arm = ExecGRPOArm(settings)
    if arm.kind != "pilot":
        print(json.dumps({"error": "a pilot recipe is required"}))
        return EXIT_FAILED
    signal.signal(signal.SIGTERM, arm.on_signal)
    try:
        if args.command == "summary":
            value = summary(arm, args.step)
            print(json.dumps({"summary": str(arm.archive / "pilot-summary.json"), "selected": value["selected"]}, indent=1))
            return EXIT_DONE
        return run_dev250(arm, args.step)
    except EvaluationFailed as error:
        return arm.finish(EXIT_FAILED, f"DEV250 failed: {error}", needs_user=str(error))
    except (DriverError, ValueError, state.StateError) as error:
        return arm.finish(EXIT_NEEDS_USER, f"input check failed: {error}", needs_user=str(error))


if __name__ == "__main__":
    raise SystemExit(main())
