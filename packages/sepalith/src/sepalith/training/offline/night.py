#!/usr/bin/env python3
"""Night-queue driver for the C13 offline steps.

``night run --recipe <rft or dpo recipe>`` is the C04 driver (``sft.night_arm.Arm``)
with three changes:

1. the trainer is ``offline.trainer`` and DEV generation is ``offline.dev_generation``;
2. gates follow ``offline.rules.offline_gate``: the R1 later-gate table against
   the starting model's bound gate result at the first gate and against the
   previous gate after that, plus the ``noedit_dodge`` row and, for DPO, the
   ``dpo_chosen_collapse`` row fed from the trainer's ``dpo_update`` telemetry;
3. after the last gate (or a rollback), R5 compares the step's result with the
   starting model and writes ``r5-decision.json`` to the evidence directory.

``night reference --recipe <dpo recipe>`` runs ``offline.reference`` under the
CUDA host guard until the reference manifest exists.

Exit codes for the night runner: 0 finished (R5 adopt or discard, or the
reference complete); 75 saved and deferred to the next night; 3 stopped for the
user (a gate's stop-for-user, or R5's stop-for-user band; write an override
file for a gate and ``night_queue retry``); anything else is a failure.
Publishing an adopted model (R8) and binding the next step are morning work.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import signal
import subprocess
import time
from typing import cast

from sepalith.training.eval.milestone_gate import requested_generation_settings
from sepalith.training.offline import rules
from sepalith.training.offline.recipe import BOUND_SCHEMA
from sepalith.training.sft import gate_rule
from sepalith.training.sft.night_arm import (
    DECISION_SCHEMA,
    EXIT_DEFERRED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_NEEDS_USER,
    PANELS,
    STATUS_BY_OUTCOME,
    Arm,
    DriverError,
    EvaluationFailed,
    Launcher,
    Settings,
    Stopped,
    read_json,
    sha256,
    stamp,
    write_json,
)
from sepalith.training.sft.recipe_v2 import evaluator_hashes

R5_SCHEMA = "sepalith.c13.r5-decision.v1"
Json = dict[str, object]


def _obj(value: object, label: str) -> Json:
    if not isinstance(value, dict):
        raise DriverError(f"{label} must be an object")
    return cast(Json, value)


class OfflineArm(Arm):
    """``Arm`` for a C13 bound recipe."""

    def __init__(self, settings: Settings,
                 launcher_factory: Callable[[Settings, Path, Callable[[], bool]], Launcher] = Launcher) -> None:
        self.settings = settings
        self.recipe_path = settings.recipe_path.resolve()
        self.recipe = read_json(self.recipe_path)
        if self.recipe.get("schema") != BOUND_SCHEMA:
            raise DriverError("the offline driver runs only C13 bound recipes")
        self.kind = str(self.recipe["kind"])
        self.recipe_sha = sha256(self.recipe_path)
        outputs = _obj(self.recipe["outputs"], "outputs")
        self.archive = Path(str(outputs["archive"]))
        self.trainer_root = Path(str(outputs["trainer"]))
        self.graceful_stop = Path(str(outputs["graceful_stop"]))
        self.gate = _obj(self.recipe["dev_gate"], "dev_gate")
        self.gates = [int(x) for x in cast(list[int], self.gate["mandatory_steps"])]
        self.evidence_dir = Path(str(self.gate["evidence_directory"]))
        self.runs_dir = self.evidence_dir / "runs"
        self.repository_root = Path(str(_obj(self.recipe["source"], "source")["repository_root"]))
        batching = _obj(self.gate["generation_batching"], "generation_batching")
        self.generation_batch_size = settings.generation_batch_size or cast(int, batching["batch_size"])
        self.generation_kv_cache = settings.generation_kv_cache or cast(str, batching["kv_cache"])
        try:
            self.generation = requested_generation_settings(self.gate.get("generation_max_new_tokens"), self.generation_batch_size,
                                                            self.generation_kv_cache)
        except ValueError as error:
            raise DriverError(f"DEV generation settings: {error}") from None
        self.signalled = False
        self.launcher = launcher_factory(settings, self.runs_dir, self.stop_requested)
        self.evaluations: list[Json] = []
        self.rule_decisions: list[Json] = []

    # starting model --------------------------------------------------------------------------
    def start_metrics(self) -> Json:
        """The starting model's metrics from its bound gate results, re-read and re-hashed."""
        reference = _obj(self.recipe["parent_reference"], "parent_reference")
        summaries = {}
        for name in PANELS:
            entry = _obj(_obj(reference["results"], "results")[name], name)
            path = Path(str(entry["path"]))
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise DriverError(f"starting model {name} result differs")
            value = read_json(path)
            problems = self.generation_problems(name, value)
            if problems:
                raise DriverError(f"starting model: {problems}")
            summaries[name] = _obj(value["summary"], "summary")
        metrics = rules.offline_metrics(summaries["dev75"], summaries["dev250"])
        if metrics != reference.get("metrics"):
            raise DriverError("starting model metrics do not match its bound results")
        return dict(metrics)

    def chosen_log_ratios(self, step: int) -> list[float]:
        telemetry = self.archive / "telemetry.jsonl"
        by_step: dict[int, float] = {}
        if telemetry.is_file():
            for line in telemetry.read_text().splitlines():
                event = json.loads(line)
                if event.get("event") == "dpo_update" and int(event["step"]) <= step:
                    by_step[int(event["step"])] = float(event["chosen_log_ratio"])
        if sorted(by_step) != list(range(1, step + 1)):
            raise DriverError(f"DPO telemetry does not cover updates 1..{step}")
        return [by_step[s] for s in range(1, step + 1)]

    # evaluation ------------------------------------------------------------------------------
    def evaluate_panels(self, step: int, checkpoint: Path, *, parent_baseline: bool, out_dir: Path) -> dict[str, Path]:
        if parent_baseline:
            raise DriverError("offline steps compare with the starting model's bound gate; there is no step-0 baseline")
        attempt = stamp()
        panels = _obj(self.gate["panels"], "panels")
        results = {}
        for name in PANELS:
            panel = _obj(panels[name], name)
            output = out_dir / f"dev-{name}-step-{step}.{attempt}.json"
            admission = out_dir / f"dev-{name}-step-{step}.{attempt}.admission.json"
            write_json(admission, {
                "schema": "sepalith.sft11.full-weight-edit-dev-generation-admission.v1", "status": "admitted",
                "purpose": "development_generation", "cuda_authorized": True, "final_set_access": False,
                "bound_recipe_sha256": self.recipe_sha, "checkpoint_manifest_sha256": sha256(checkpoint / "campaign-manifest.json"),
                "panel_sha256": panel["sha256"], "output": str(output),
                "deadline_seconds": int(min(3600, self.settings.eval_budget_seconds)),
            }, fresh=True)
            args = ["sepalith.training.offline.dev_generation", "--bound-recipe", str(self.recipe_path), "--checkpoint",
                    str(checkpoint), "--dev-admission", str(admission), "--output", str(output), "--panel", name,
                    "--generation-batch-size", str(self.generation_batch_size), "--generation-kv-cache", str(self.generation_kv_cache)]
            code = self.launcher.run(f"dev-{name}-step-{step}", args, budget=self.settings.eval_budget_seconds, on_stop=self.terminate)
            if self.stop_requested():
                raise Stopped(f"stop requested during {name} generation")
            if code != 0 or not output.is_file():
                raise EvaluationFailed(f"{name} generation for step {step} exited {code} without a result")
            results[name] = output
            self.evaluations.append({"step": step, "panel": name, "result": str(output), "sha256": sha256(output)})
        return results

    def decide(self, step: int, results: Mapping[str, Path] | None, failure: str | None,
               override: tuple[Path, Json] | None) -> tuple[str, Path]:
        checkpoint = self.checkpoint(step)
        manifest_sha = sha256(checkpoint / "campaign-manifest.json")
        first = step == self.gates[0]
        kind = "offline_first_gate" if first else "offline_later_gate"
        metrics: Json | None = None
        rule: Json | None = None
        problems = [failure] if failure else []
        if results is not None:
            for name in PANELS:
                problems += self.result_problems(name, results[name], step, manifest_sha)
        if not problems and results is not None:
            summaries = {name: _obj(read_json(results[name])["summary"], "summary") for name in PANELS}
            metrics = {**rules.offline_metrics(summaries["dev75"], summaries["dev250"]), "loss": self.gate_loss(step)}
            try:
                start = self.start_metrics()
                reference = start if first else self.previous_metrics(step)
                ratios = self.chosen_log_ratios(step) if self.kind == "dpo" else None
            except (DriverError, ValueError, KeyError) as error:
                problems.append(f"reference metrics unavailable: {error}")
            else:
                rule = rules.offline_gate(metrics, reference, first=first, chosen_log_ratios=ratios,
                                          start_score250=cast(int, start["score250"]))
        if problems or rule is None:
            rule = gate_rule.failure(kind, "; ".join(problems) or "no rule inputs")
        outcome = str(rule["outcome"])
        decided_by = "rule_R1_offline"
        override_record = None
        if override is not None:
            override_record = {"path": str(override[0]), "sha256": sha256(override[0]), "outcome": override[1]["outcome"],
                               "reason": override[1]["reason"], "author": override[1].get("author")}
            outcome, decided_by = str(override[1]["outcome"]), "human_override"
            if outcome == gate_rule.CONTINUE and (results is None or any("binding" in p for p in problems)):
                outcome, decided_by = gate_rule.STOP_FOR_USER, "human_override_refused_without_bound_results"
        decision: Json = {
            "schema": DECISION_SCHEMA, "status": STATUS_BY_OUTCOME[outcome], "continue_training": outcome == gate_rule.CONTINUE,
            "step": step, "outcome": outcome, "decided_by": decided_by, "rule": rule, "rule_outcome": rule["outcome"],
            "override": override_record, "metrics": metrics, "evaluation_complete": results is not None,
            "bound_recipe_sha256": self.recipe_sha, "checkpoint_manifest_sha256": manifest_sha,
            "panel_sha256": _obj(self.gate["panel"], "panel")["sha256"],
            "generation_result_sha256": sha256(results["dev75"]) if results else None,
            "results": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (results or {}).items()},
            "dev250": _obj(self.gate["panels"], "panels")["dev250"], "generation_runtime": self.generation,
            "evaluator_source_sha256": evaluator_hashes(self.repository_root),
            "last_good_checkpoint": self.last_good(step) if outcome == gate_rule.ROLLBACK else None,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        attempt = len(self.decisions(step)) + 1
        path = self.evidence_dir / "decisions" / f"decision-step-{step}-{attempt}.json"
        write_json(path, decision, fresh=True)
        if outcome == gate_rule.CONTINUE and results is not None:
            from sepalith.training.eval.prepare_dev_gate import prepare

            prepare(self.recipe_path, checkpoint, results["dev75"], path, self.evidence_path(step))
        self.rule_decisions.append({"rule": "R1-offline", "step": step, "kind": kind, "inputs": rule.get("inputs", {}),
                                    "rule_outcome": rule["outcome"], "outcome": outcome, "decided_by": decided_by,
                                    "decision": str(path)})
        return outcome, path

    # R5 --------------------------------------------------------------------------------------
    def gate_metrics(self, step: int) -> Json:
        decisions = self.decisions(step)
        if not decisions:
            raise DriverError(f"no decision for gate {step}")
        decision = read_json(decisions[-1])
        return _obj(decision.get("metrics"), f"gate {step} metrics")

    def apply_r5(self, result_step: int | None) -> int:
        """R5 against the starting model; ``result_step`` None means the step's result is the starting model."""
        path = self.evidence_dir / "r5-decision.json"
        arm = str(_obj(self.recipe["arm"], "arm")["name"])
        if path.is_file():
            record = read_json(path)
        else:
            start = self.start_metrics()
            if result_step is None:
                rule: Json = {"rule": "R5", "outcome": rules.DISCARD, "rows": [],
                              "problems": ["rolled back to the starting model: no offline result to adopt"], "inputs": {"start": start}}
                checkpoint = None
            else:
                rule = rules.apply_r5(self.gate_metrics(result_step), start)
                checkpoint = str(self.checkpoint(result_step))
            record = {"schema": R5_SCHEMA, "arm": arm, "kind": self.kind, "bound_recipe_sha256": self.recipe_sha,
                      "starting_model": _obj(self.recipe["parent"], "parent")["path"], "result_checkpoint": checkpoint,
                      "result_step": result_step, "rule": rule, "outcome": rule["outcome"], "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            write_json(path, record, fresh=True)
        outcome = str(record["outcome"])
        self.rule_decisions.append({"rule": "R5", "inputs": _obj(record["rule"], "rule").get("inputs", {}), "outcome": outcome,
                                    "decision": str(path)})
        result = {"status": f"r5_{outcome}", "result_checkpoint": record["result_checkpoint"], "r5_decision": str(path)}
        if outcome == rules.ADOPT:
            return self.finish(EXIT_DONE, f"{arm}: R5 adopt {record['result_checkpoint']}; publish it under R8 and bind the next step",
                               arm_result=result)
        if outcome == rules.DISCARD:
            return self.finish(EXIT_DONE, f"{arm}: R5 discard; the next step starts from the starting model", arm_result=result)
        return self.finish(EXIT_NEEDS_USER, f"{arm}: R5 stop-for-user", arm_result=result,
                           needs_user=f"{arm}: R5 is in its stop-for-user band; see {path}")

    # main loop -------------------------------------------------------------------------------
    def train(self, latest: int, stop_after_step: int | None = None) -> int:
        if latest == 0 and (self.archive.exists() or self.trainer_root.exists()):
            moment = int(time.time())
            for root in (self.archive, self.trainer_root):
                if root.exists():
                    root.rename(root.with_name(f"{root.name}.stale-{moment}"))
        self.clear_stale_stop_file()
        args = ["sepalith.training.offline.trainer", "run", "--recipe", str(self.recipe_path)]
        if latest:
            args += ["--resume", str(self.checkpoint(latest))]
        if stop_after_step is not None:
            args += ["--stop-after-step", str(stop_after_step)]
        return self.launcher.run(f"train-from-{latest}", args, budget=None, on_stop=self.request_trainer_stop)

    def run(self) -> int:
        arm = str(_obj(self.recipe["arm"], "arm")["name"])
        if self.kind == "dpo":
            from sepalith.training.offline.recipe import reference_manifest

            try:
                reference_manifest(self.recipe)
            except ValueError as error:
                raise DriverError(f"{error}") from None
        final = self.gates[-1]
        previous_latest = None
        while True:
            if self.stop_requested():
                return self.finish(EXIT_DEFERRED, f"{arm}: stop requested; saved state resumes next night")
            steps = self.durable_steps()
            latest = steps[-1] if steps else 0
            if latest in self.gates:
                outcome = self.gate_step(latest)
                if outcome == "defer":
                    return self.finish(EXIT_DEFERRED, f"{arm}: gate {latest} evaluation deferred to the next night")
                if outcome == gate_rule.ROLLBACK:
                    good = [gate for gate in self.gates if gate < latest and self.evidence_path(gate).is_file()]
                    return self.apply_r5(good[-1] if good else None)
                if outcome == gate_rule.STOP_FOR_USER:
                    return self.finish(EXIT_NEEDS_USER, f"{arm}: gate stop-for-user at {latest}",
                                       needs_user=(f"{arm} gate {latest}: R1-offline says stop-for-user; see {self.decisions(latest)[-1]}. "
                                                   f"To decide, write {self.evidence_dir}/override-step-{latest}.json and retry the job."))
                if latest == final:
                    return self.apply_r5(final)
            if latest == previous_latest:
                raise DriverError(f"the trainer exited without progress past update {latest}")
            if self.settings.seconds_left() < self.settings.min_train_seconds:
                return self.finish(EXIT_DEFERRED, f"{arm}: too close to the deadline to train; resumes at {latest}")
            previous_latest = latest
            code = self.train(latest)
            if code != 0:
                return self.finish(EXIT_FAILED, f"{arm}: trainer exited {code} from update {latest}",
                                   needs_user=f"{arm}: trainer failed from update {latest}; see {self.runs_dir}")
            if self.stop_requested() or self.settings.seconds_left() <= 0:
                steps = self.durable_steps()
                now_latest = steps[-1] if steps else 0
                if now_latest not in self.gates:
                    return self.finish(EXIT_DEFERRED, f"{arm}: saved at update {now_latest} at the deadline")


def run_reference(arm: OfflineArm) -> int:
    """Compute the DPO reference log-probabilities; resumable across nights."""
    from sepalith.training.offline.recipe import reference_manifest

    if arm.kind != "dpo":
        raise DriverError("the reference pass belongs to a DPO recipe")
    manifest = Path(str(_obj(arm.recipe["reference"], "reference")["manifest"]))

    def complete() -> bool:
        if not manifest.is_file():
            return False
        reference_manifest(arm.recipe)
        return True

    if complete():
        return arm.finish(EXIT_DONE, f"reference log-probabilities already complete: {manifest}")
    if arm.settings.seconds_left() < arm.settings.min_train_seconds:
        return arm.finish(EXIT_DEFERRED, "too close to the deadline for the reference pass")

    def leave_running(_process: subprocess.Popen[bytes]) -> None:
        """The child sees the stop-request file and the deadline itself and flushes before exiting."""

    code = arm.launcher.run("reference", ["sepalith.training.offline.reference", "--recipe", str(arm.recipe_path)], budget=None,
                            on_stop=leave_running)
    if code != 0:
        return arm.finish(EXIT_FAILED, f"reference pass exited {code}", needs_user=f"reference pass failed; see {arm.runs_dir}")
    if complete():
        arm.evaluations.append({"reference_manifest": str(manifest), "sha256": sha256(manifest)})
        return arm.finish(EXIT_DONE, f"reference log-probabilities complete: {manifest}")
    return arm.finish(EXIT_DEFERRED, "reference pass saved its prefix at the deadline; resumes next night")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "reference"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--eval-budget-seconds", type=float, default=2400.0)
    parser.add_argument("--min-train-seconds", type=float, default=900.0)
    parser.add_argument("--no-guard", action="store_true", help="run children without cuda_host_guard (tests only)")
    args = parser.parse_args(argv)
    settings = Settings.from_env(args.recipe, guard=not args.no_guard, eval_budget_seconds=args.eval_budget_seconds,
                                 min_train_seconds=args.min_train_seconds)
    arm = OfflineArm(settings)
    signal.signal(signal.SIGTERM, arm.on_signal)
    try:
        if args.command == "reference":
            return run_reference(arm)
        return arm.run()
    except (DriverError, EvaluationFailed, ValueError) as error:
        return arm.finish(EXIT_FAILED, f"driver error: {error}", needs_user=str(error))


if __name__ == "__main__":
    raise SystemExit(main())
