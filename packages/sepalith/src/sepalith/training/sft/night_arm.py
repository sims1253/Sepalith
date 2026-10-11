#!/usr/bin/env python3
"""Night-queue driver for editing-SFT v2 arms (card C04).

One invocation is one night job. The driver never touches CUDA itself; it
starts the trainer and the DEV evaluator as separate processes, one at a time,
each under ``cuda_host_guard``:

1. Train until the next mandatory gate, the night deadline or a stop request.
   The trainer exits, which frees its GPU memory, after saving full state.
2. At a gate, run DEV75 and DEV250 generation (``run_dev_generation``), one
   process per panel, apply rule R1 and write the decision record that
   ``milestone_gate`` expects. A human override file takes precedence.
3. On ``continue``, write the gate evidence and resume from the gate checkpoint.

Exit codes for the night runner: 0 the job is finished (arm complete, pilot
gate reached, or rolled back); 75 saved and deferred to the next night; 3
stopped for the user (R1 stop-for-user; write an override file, then
``night_queue retry``); anything else is a failure.

Environment from the night runner: ``SEPALITH_DEADLINE_EPOCH``,
``SEPALITH_HARD_STOP_EPOCH``, ``SEPALITH_STOP_REQUEST_FILE`` and
``SEPALITH_JOB_RESULT``. SIGTERM is translated into the trainer's
recipe-bound save-and-stop file; it is never forwarded to the trainer.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from types import FrameType
from typing import cast

from sepalith.training.sft import gate_rule
from sepalith.training.sft.recipe_v2 import BOUND_SCHEMA, evaluator_hashes

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_NEEDS_USER = 3
EXIT_DEFERRED = 75
DECISION_SCHEMA = "sepalith.sft11.full-weight-edit-dev-decision.v1"
RESULT_SCHEMA = "sepalith.sft11.full-weight-edit-dev-generation.v1"
OVERRIDE_SCHEMA = "sepalith.sft-v2.gate-override.v1"
BASELINE_SCHEMA = "sepalith.sft-v2.parent-baseline.v1"
ARM_RESULT_SCHEMA = "sepalith.sft-v2.arm-result.v1"
CUDA_LOCK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock")
PANELS = ("dev75", "dev250")
STATUS_BY_OUTCOME = {gate_rule.CONTINUE: "admitted_for_continuation", gate_rule.STOP_FOR_USER: "stopped_for_user",
                     gate_rule.ROLLBACK: "rolled_back"}

Json = dict[str, object]


class DriverError(RuntimeError):
    """The arm cannot proceed without someone looking at it."""


class EvaluationFailed(RuntimeError):
    """A DEV generation process failed or produced no result."""


class Stopped(RuntimeError):
    """The night runner asked the job to stop."""


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Json:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise DriverError(f"expected a JSON object: {path}")
    return cast(Json, value)


def write_json(path: Path, value: object, *, fresh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and path.exists():
        raise DriverError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _object(value: object, label: str) -> Json:
    if not isinstance(value, dict):
        raise DriverError(f"{label} must be an object")
    return cast(Json, value)


def _ints(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise DriverError(f"{label} must be a list of integers")
    return cast(list[int], value)


def _str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DriverError(f"{label} must be a non-empty string")
    return value


def stamp() -> str:
    """A sortable, unique attempt label."""
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"


def _epoch(environ: Mapping[str, str], name: str) -> float | None:
    value = environ.get(name)
    return float(value) if value not in (None, "") else None


# Launching child processes ------------------------------------------------------------------

@dataclass
class Settings:
    recipe_path: Path
    python: str = sys.executable
    guard: bool = True
    eval_budget_seconds: float = 2400.0
    min_train_seconds: float = 900.0
    stop_after_gate: int | None = None
    deadline: float | None = None
    hard_stop: float | None = None
    stop_request_file: Path | None = None
    job_result: Path | None = None
    poll_seconds: float = 5.0
    clock: Callable[[], float] = time.time
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, recipe_path: Path, environ: Mapping[str, str] | None = None, *, guard: bool = True,
                 eval_budget_seconds: float = 2400.0, min_train_seconds: float = 900.0,
                 stop_after_gate: int | None = None) -> Settings:
        env = os.environ if environ is None else environ
        stop = env.get("SEPALITH_STOP_REQUEST_FILE")
        result = env.get("SEPALITH_JOB_RESULT")
        return cls(recipe_path=recipe_path, deadline=_epoch(env, "SEPALITH_DEADLINE_EPOCH"),
                   hard_stop=_epoch(env, "SEPALITH_HARD_STOP_EPOCH"), stop_request_file=Path(stop) if stop else None,
                   job_result=Path(result) if result else None, guard=guard, eval_budget_seconds=eval_budget_seconds,
                   min_train_seconds=min_train_seconds, stop_after_gate=stop_after_gate)

    def seconds_left(self) -> float:
        return math.inf if self.deadline is None else self.deadline - self.clock()


class Launcher:
    """Run ``python -m <module> ...`` once, optionally under the CUDA host guard."""

    def __init__(self, settings: Settings, work_dir: Path, stop_requested: Callable[[], bool]) -> None:
        self.settings, self.work_dir, self.stop_requested = settings, work_dir, stop_requested

    def guard_seconds(self, budget: float | None) -> int:
        now = self.settings.clock()
        limit = 93600.0 if self.settings.hard_stop is None else self.settings.hard_stop - now - 30
        if budget is not None:
            limit = min(limit, budget)
        return int(max(60.0, min(93600.0, limit)))

    def run(self, label: str, module_args: Sequence[str], *, budget: float | None,
            on_stop: Callable[[subprocess.Popen[bytes]], None]) -> int:
        directory = self.work_dir / f"{stamp()}-{label}"
        directory.mkdir(parents=True, exist_ok=False)
        argv = [self.settings.python, "-B", "-m", *module_args]
        if self.settings.guard:
            CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
            CUDA_LOCK.touch(exist_ok=True)
            write_json(directory / "command.json", argv)
            command = [self.settings.python, "-B", "-m", "sepalith.training.guards.cuda_host_guard",
                       "--command-json", str(directory / "command.json"), "--output", str(directory / "guard"),
                       "--seconds", str(self.guard_seconds(budget))]
        else:
            command = argv
        env = dict(os.environ)
        env.update(self.settings.extra_env)
        with (directory / "driver-child.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            signalled = False
            while True:
                try:
                    return process.wait(timeout=self.settings.poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
                if not signalled and self.stop_requested():
                    signalled = True
                    on_stop(process)


# Arm state ------------------------------------------------------------------------------------

class Arm:
    def __init__(self, settings: Settings, launcher_factory: Callable[[Settings, Path, Callable[[], bool]], Launcher] = Launcher) -> None:
        self.settings = settings
        self.recipe_path = settings.recipe_path.resolve()
        self.recipe = read_json(self.recipe_path)
        if self.recipe.get("schema") != BOUND_SCHEMA:
            raise DriverError("the night driver runs only v2 bound recipes")
        self.recipe_sha = sha256(self.recipe_path)
        outputs = _object(self.recipe["outputs"], "outputs")
        self.archive = Path(_str(outputs["archive"], "archive"))
        self.trainer_root = Path(_str(outputs["trainer"], "trainer"))
        self.graceful_stop = Path(_str(outputs["graceful_stop"], "graceful_stop"))
        gate = _object(self.recipe["dev_gate"], "dev_gate")
        self.gate = gate
        self.gates = _ints(gate["mandatory_steps"], "mandatory_steps")
        self.evidence_dir = Path(_str(gate["evidence_directory"], "evidence_directory"))
        self.runs_dir = self.evidence_dir / "runs"
        self.repository_root = Path(_str(_object(self.recipe["source"], "source")["repository_root"], "repository_root"))
        self.signalled = False
        self.launcher = launcher_factory(settings, self.runs_dir, self.stop_requested)
        self.evaluations: list[Json] = []
        self.rule_decisions: list[Json] = []

    # stop requests ---------------------------------------------------------------------------
    def stop_requested(self) -> bool:
        stop_file = self.settings.stop_request_file
        return self.signalled or (stop_file is not None and stop_file.exists())

    def on_signal(self, _number: int, _frame: FrameType | None) -> None:
        self.signalled = True

    def request_trainer_stop(self, _process: subprocess.Popen[bytes]) -> None:
        """Write the recipe-bound save-and-stop file the trainer checks at each optimizer boundary."""
        payload = {"action": "save_and_stop", "bound_recipe_sha256": self.recipe_sha}
        self.graceful_stop.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.graceful_stop, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")

    @staticmethod
    def terminate(process: subprocess.Popen[bytes]) -> None:
        process.terminate()

    def clear_stale_stop_file(self) -> None:
        if self.graceful_stop.exists():
            os.replace(self.graceful_stop, self.graceful_stop.with_name(f"{self.graceful_stop.name}.consumed-{int(time.time())}"))

    # checkpoints -----------------------------------------------------------------------------
    def durable_steps(self) -> list[int]:
        steps = []
        for path in (self.archive / "full").glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if suffix.isdigit() and (path / "campaign-manifest.json").is_file():
                steps.append(int(suffix))
        return sorted(steps)

    def checkpoint(self, step: int) -> Path:
        return self.archive / "full" / f"checkpoint-{step}"

    def evidence_path(self, step: int) -> Path:
        return self.evidence_dir / f"dev-gate-step-{step}.json"

    def decisions(self, step: int) -> list[Path]:
        found = []
        for path in (self.evidence_dir / "decisions").glob(f"decision-step-{step}-*.json"):
            suffix = path.stem.removeprefix(f"decision-step-{step}-")
            if suffix.isdigit():
                found.append((int(suffix), path))
        return [path for _, path in sorted(found)]

    def override(self, step: int) -> tuple[Path, Json] | None:
        path = self.evidence_dir / f"override-step-{step}.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value.get("schema") != OVERRIDE_SCHEMA or value.get("step") != step:
            raise DriverError(f"override file {path} has the wrong schema or step")
        if value.get("bound_recipe_sha256") != self.recipe_sha:
            raise DriverError(f"override file {path} is bound to another recipe")
        if value.get("outcome") not in gate_rule.OUTCOMES:
            raise DriverError(f"override outcome must be one of {gate_rule.OUTCOMES}")
        manifest = value.get("checkpoint_manifest_sha256")
        if manifest is not None and manifest != sha256(self.checkpoint(step) / "campaign-manifest.json"):
            raise DriverError(f"override file {path} names another checkpoint")
        if not isinstance(value.get("reason"), str) or not value["reason"]:
            raise DriverError("an override needs a reason")
        return path, value

    # evaluation ------------------------------------------------------------------------------
    def evaluate_panels(self, step: int, checkpoint: Path, *, parent_baseline: bool, out_dir: Path) -> dict[str, Path]:
        """The single call site for DEV generation: one run_dev_generation process per panel.

        Card C05 adds a one-load option for both panels; switch here when it lands.
        """
        attempt = stamp()
        panels = _object(self.gate["panels"], "panels")
        results = {}
        for name in PANELS:
            panel = _object(panels[name], name)
            output = out_dir / f"dev-{name}-step-{step}.{attempt}.json"
            admission = out_dir / f"dev-{name}-step-{step}.{attempt}.admission.json"
            manifest = checkpoint / "campaign-manifest.json"
            write_json(admission, {
                "schema": "sepalith.sft11.full-weight-edit-dev-generation-admission.v1", "status": "admitted",
                "purpose": "development_generation", "cuda_authorized": True, "final_set_access": False,
                "bound_recipe_sha256": self.recipe_sha, "checkpoint_manifest_sha256": sha256(manifest),
                "panel_sha256": panel["sha256"], "output": str(output),
                "deadline_seconds": int(min(3600, self.settings.eval_budget_seconds)),
            }, fresh=True)
            # run_dev_generation evaluates SFT checkpoints; DEV75 is its default panel and
            # `--panel dev250` needs the C05 evaluator (PR #41). Step 0 uses parent_baseline.
            module = "sepalith.training.sft.parent_baseline" if parent_baseline else "sepalith.training.eval.run_dev_generation"
            args = [module, "--bound-recipe", str(self.recipe_path), "--checkpoint", str(checkpoint),
                    "--dev-admission", str(admission), "--output", str(output)]
            if parent_baseline or name != "dev75":
                args += ["--panel", name]
            code = self.launcher.run(f"dev-{name}-step-{step}", args, budget=self.settings.eval_budget_seconds,
                                     on_stop=self.terminate)
            if self.stop_requested():
                raise Stopped(f"stop requested during {name} generation")
            if code != 0 or not output.is_file():
                raise EvaluationFailed(f"{name} generation for step {step} exited {code} without a result")
            results[name] = output
            self.evaluations.append({"step": step, "panel": name, "result": str(output), "sha256": sha256(output)})
        return results

    def result_problems(self, name: str, path: Path, step: int, manifest_sha: str) -> list[str]:
        value = read_json(path)
        panel = _object(_object(self.gate["panels"], "panels")[name], name)
        summary = _object(value.get("summary"), "summary")
        denominators = _object(summary.get("denominators"), "denominators")
        checks = {
            "schema": value.get("schema") == RESULT_SCHEMA and value.get("status") == "complete",
            "step": value.get("step") == step,
            "recipe": _object(value.get("bound_recipe"), "bound_recipe").get("sha256") == self.recipe_sha,
            "checkpoint": _object(value.get("checkpoint"), "checkpoint").get("campaign_manifest_sha256") == manifest_sha,
            "panel": _object(value.get("panel"), "panel").get("sha256") == panel["sha256"] == summary.get("panel_sha256"),
            "cases": denominators.get("cases") == panel["cases"],
            "final_set": value.get("final_set_access") is False,
        }
        return [f"{name} result {key} binding differs" for key, ok in checks.items() if not ok]

    def gate_loss(self, step: int) -> float | None:
        telemetry = self.archive / "telemetry.jsonl"
        loss = None
        if telemetry.is_file():
            for line in telemetry.read_text().splitlines():
                event = json.loads(line)
                logs = event.get("logs") or {}
                if event.get("event") == "log" and event.get("step") == step and "loss" in logs:
                    loss = float(logs["loss"])
        return loss

    def baseline_metrics(self) -> Json:
        path = Path(_str(_object(self.gate["parent_baseline"], "parent_baseline")["path"], "baseline path"))
        if not path.is_file():
            raise DriverError(f"parent baseline missing: {path} (run the C07 baseline job first)")
        record = read_json(path)
        parent = _object(self.recipe["parent"], "parent")
        files = _object(parent["files"], "parent files")
        if record.get("schema") != BASELINE_SCHEMA or record.get("parent_weights_sha256") != files["model.safetensors"]:
            raise DriverError("parent baseline is for another parent")
        summaries = {}
        for name in PANELS:
            entry = _object(_object(record["panels"], "baseline panels")[name], name)
            result = Path(_str(entry["result"], "baseline result"))
            if not result.is_file() or sha256(result) != entry["sha256"]:
                raise DriverError(f"baseline {name} result differs")
            problems = self.result_problems(name, result, 0, _str(record["parent_manifest_sha256"], "manifest"))
            # The baseline was produced under one arm's recipe; only that binding may differ.
            problems = [p for p in problems if "recipe binding" not in p]
            if problems:
                raise DriverError(f"baseline {name}: {problems}")
            summaries[name] = _object(read_json(result)["summary"], "summary")
        metrics = gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"])
        if metrics != record.get("metrics"):
            raise DriverError("baseline metrics do not match its results")
        return dict(metrics)

    def previous_metrics(self, step: int) -> Json:
        from sepalith.training.eval.milestone_gate import verify_gate_evidence

        previous = self.gates[self.gates.index(step) - 1]
        evidence = verify_gate_evidence(self.recipe, self.recipe_path, previous, self.checkpoint(previous))
        decision = read_json(Path(evidence["root_decision"]["path"]))
        return _object(decision.get("metrics"), "previous metrics")

    def last_good(self, step: int) -> str:
        earlier = [gate for gate in self.gates if gate < step and self.evidence_path(gate).is_file()]
        return str(self.checkpoint(earlier[-1])) if earlier else _str(_object(self.recipe["parent"], "parent")["path"], "parent")

    def decide(self, step: int, results: Mapping[str, Path] | None, failure: str | None,
               override: tuple[Path, Json] | None) -> tuple[str, Path]:
        checkpoint = self.checkpoint(step)
        manifest_sha = sha256(checkpoint / "campaign-manifest.json")
        first = step == self.gates[0]
        kind = "first_gate" if first else "later_gate"
        metrics: Json | None = None
        rule: Json | None = None
        problems = [failure] if failure else []
        if results is not None:
            for name in PANELS:
                problems += self.result_problems(name, results[name], step, manifest_sha)
        if not problems and results is not None:
            summaries = {name: _object(read_json(results[name])["summary"], "summary") for name in PANELS}
            metrics = {**gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"]), "loss": self.gate_loss(step)}
            try:
                reference = self.baseline_metrics() if first else self.previous_metrics(step)
            except (DriverError, ValueError, KeyError) as error:
                problems.append(f"reference metrics unavailable: {error}")
            else:
                rule = gate_rule.apply_r1(metrics, first=first, reference=reference)
        if problems or rule is None:
            rule = gate_rule.failure(kind, "; ".join(problems) or "no rule inputs")
        outcome = str(rule["outcome"])
        decided_by = "rule_R1"
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
            "panel_sha256": _object(self.gate["panel"], "panel")["sha256"],
            "generation_result_sha256": sha256(results["dev75"]) if results else None,
            "results": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (results or {}).items()},
            "dev250": _object(self.gate["panels"], "panels")["dev250"],
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
        self.rule_decisions.append({"rule": "R1", "step": step, "kind": kind, "inputs": rule.get("inputs", {}),
                                    "rule_outcome": rule["outcome"], "outcome": outcome, "decided_by": decided_by,
                                    "decision": str(path)})
        return outcome, path

    def gate_step(self, step: int) -> str:
        """Bring the gate at ``step`` to a decision; returns an R1 outcome or ``defer``."""
        if self.evidence_path(step).is_file():
            return gate_rule.CONTINUE
        decisions = self.decisions(step)
        override = self.override(step)
        prior = read_json(decisions[-1]) if decisions else None
        results = None
        if prior is not None and prior.get("evaluation_complete") is True:
            prior_override = _object(prior["override"], "override")["sha256"] if prior.get("override") else None
            if override is None or sha256(override[0]) == prior_override:
                return str(prior["outcome"])
            entries = _object(prior["results"], "results")
            results = {name: Path(_str(_object(entries[name], name)["path"], "result path")) for name in PANELS}
        if results is None:
            if self.settings.seconds_left() < self.settings.eval_budget_seconds:
                return "defer"
            try:
                results = self.evaluate_panels(step, self.checkpoint(step), parent_baseline=False, out_dir=self.evidence_dir / "results")
            except Stopped:
                return "defer"
            except EvaluationFailed as error:
                outcome, _ = self.decide(step, None, f"evaluator crash: {error}", override)
                return outcome
        outcome, _ = self.decide(step, results, None, override)
        return outcome

    # training --------------------------------------------------------------------------------
    def train(self, latest: int, stop_after_step: int | None = None) -> int:
        if latest == 0 and (self.archive.exists() or self.trainer_root.exists()):
            # A fresh start that never reached a durable checkpoint; keep it for inspection.
            stamp = int(time.time())
            for root in (self.archive, self.trainer_root):
                if root.exists():
                    os.replace(root, root.with_name(f"{root.name}.stale-{stamp}"))
        self.clear_stale_stop_file()
        args = ["sepalith.training.sft.full_weight_edit_sft", "run", "--recipe", str(self.recipe_path)]
        if latest:
            args += ["--resume", str(self.checkpoint(latest))]
        if stop_after_step is not None:
            args += ["--stop-after-step", str(stop_after_step)]
        return self.launcher.run(f"train-from-{latest}", args, budget=None, on_stop=self.request_trainer_stop)

    def finish(self, code: int, summary: str, *, needs_user: str | None = None, arm_result: Json | None = None) -> int:
        record = {"summary": summary, "needs_user": needs_user, "evaluations": self.evaluations,
                  "rule_decisions": self.rule_decisions,
                  "checkpoints": [str(self.checkpoint(step)) for step in self.durable_steps()]}
        if arm_result is not None:
            write_json(self.archive / "arm-result.json", {"schema": ARM_RESULT_SCHEMA, "recipe": str(self.recipe_path),
                                                          "recipe_sha256": self.recipe_sha, **arm_result})
        if self.settings.job_result is not None:
            write_json(self.settings.job_result, record)
        print(json.dumps({"exit": code, "summary": summary, "needs_user": needs_user}), flush=True)
        return code

    def run(self) -> int:
        arm_name = _str(_object(self.recipe["arm"], "arm")["name"], "arm")
        final = self.gates[-1]
        if self.settings.stop_after_gate is not None and self.settings.stop_after_gate not in self.gates:
            raise DriverError("stop-after-gate must be one of the mandatory gates")
        previous_latest = None
        while True:
            if self.stop_requested():
                return self.finish(EXIT_DEFERRED, f"{arm_name}: stop requested; saved state resumes next night")
            steps = self.durable_steps()
            latest = steps[-1] if steps else 0
            if latest in self.gates:
                outcome = self.gate_step(latest)
                if outcome == "defer":
                    return self.finish(EXIT_DEFERRED, f"{arm_name}: gate {latest} evaluation deferred to the next night")
                pilot_stop = self.settings.stop_after_gate == latest
                if outcome == gate_rule.ROLLBACK:
                    good = self.last_good(latest)
                    return self.finish(EXIT_DONE, f"{arm_name}: R1 rollback at gate {latest}; result is {good}",
                                       arm_result={"status": "rolled_back", "gate": latest, "result_checkpoint": good})
                if outcome == gate_rule.STOP_FOR_USER:
                    # A pilot that misses the R1 floor is an R2 discard; a crash or a binding problem still needs the user.
                    last = read_json(self.decisions(latest)[-1]) if self.decisions(latest) else {}
                    floor_only = last.get("evaluation_complete") is True and not _object(last.get("rule", {}), "rule").get("problems")
                    if pilot_stop and floor_only:
                        return self.finish(EXIT_DONE, f"{arm_name}: pilot gate {latest} did not pass the R1 first-gate floor (R2 discards it)",
                                           arm_result={"status": "pilot_gate_failed_r1", "gate": latest})
                    return self.finish(EXIT_NEEDS_USER, f"{arm_name}: R1 stop-for-user at gate {latest}",
                                       needs_user=(f"{arm_name} gate {latest}: R1 says stop-for-user; see {self.decisions(latest)[-1]}. "
                                                   f"To decide, write {self.evidence_dir}/override-step-{latest}.json and retry the job."))
                if latest == final:
                    return self.finish(EXIT_DONE, f"{arm_name}: complete at update {final}",
                                       arm_result={"status": "complete", "gate": final, "result_checkpoint": str(self.checkpoint(final))})
                if pilot_stop:
                    return self.finish(EXIT_DONE, f"{arm_name}: pilot stopped at gate {latest} with R1 continue",
                                       arm_result={"status": "pilot_gate_passed", "gate": latest,
                                                   "result_checkpoint": str(self.checkpoint(latest))})
            if latest == previous_latest:
                raise DriverError(f"the trainer exited without progress past update {latest}")
            if self.settings.seconds_left() < self.settings.min_train_seconds:
                return self.finish(EXIT_DEFERRED, f"{arm_name}: too close to the deadline to train; resumes at {latest}")
            previous_latest = latest
            code = self.train(latest)
            if code != 0:
                return self.finish(EXIT_FAILED, f"{arm_name}: trainer exited {code} from update {latest}",
                                   needs_user=f"{arm_name}: trainer failed from update {latest}; see {self.runs_dir}")
            if self.stop_requested() or self.settings.seconds_left() <= 0:
                steps = self.durable_steps()
                now_latest = steps[-1] if steps else 0
                if now_latest not in self.gates:
                    return self.finish(EXIT_DEFERRED, f"{arm_name}: saved at update {now_latest} at the deadline")


# Baseline and smoke -------------------------------------------------------------------------

def run_baseline(arm: Arm) -> int:
    parent = _object(arm.recipe["parent"], "parent")
    path = Path(_str(_object(arm.gate["parent_baseline"], "parent_baseline")["path"], "baseline path"))
    if path.is_file():
        arm.baseline_metrics()
        return arm.finish(EXIT_DONE, f"parent baseline already exists: {path}")
    checkpoint = Path(_str(parent["path"], "parent path"))
    try:
        results = arm.evaluate_panels(0, checkpoint, parent_baseline=True, out_dir=path.parent)
    except Stopped:
        return arm.finish(EXIT_DEFERRED, "baseline stopped by the night runner")
    summaries = {name: _object(read_json(result)["summary"], "summary") for name, result in results.items()}
    metrics = gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"])
    record = {"schema": BASELINE_SCHEMA, "parent_path": str(checkpoint),
              "parent_weights_sha256": _object(parent["files"], "files")["model.safetensors"],
              "parent_manifest_sha256": sha256(checkpoint / "campaign-manifest.json"),
              "recipe": {"path": str(arm.recipe_path), "sha256": arm.recipe_sha},
              "panels": {name: {"result": str(result), "sha256": sha256(result),
                                "panel_sha256": _object(_object(arm.gate["panels"], "panels")[name], name)["sha256"]}
                         for name, result in results.items()},
              "metrics": metrics, "target_nll": {name: summaries[name].get("target_nll") for name in PANELS},
              "evaluator_source_sha256": evaluator_hashes(arm.repository_root),
              "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(path, record, fresh=True)
    arm.baseline_metrics()
    arm.rule_decisions.append({"rule": "baseline", "step": 0, "inputs": dict(metrics), "outcome": "recorded"})
    return arm.finish(EXIT_DONE, f"parent baseline: {json.dumps(metrics, sort_keys=True)}")


def smoke_receipt(arm: Arm, phases: list[Json]) -> Json:
    events = [json.loads(line) for line in (arm.archive / "telemetry.jsonl").read_text().splitlines()]
    updates = [e for e in events if e.get("event") == "pre_optimizer"]
    times = [float(e["at"]) for e in updates]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False) if b - a > 0]
    saves = [e for e in events if e.get("event") == "checkpoint_durable"]
    stops = [e for e in events if str(e.get("event", "")).endswith("_save_at_optimizer_boundary")]
    save_seconds = [float(save["at"]) - float(stop["at"]) for stop, save in zip(stops, saves, strict=False)]
    resume_seconds = None
    if len(phases) > 1:
        later = [t for t in times if t >= float(_str(str(phases[1]["started"]), "started"))]
        resume_seconds = later[0] - float(str(phases[1]["started"])) if later else None
    return {"schema": "sepalith.sft-v2.smoke-receipt.v1", "recipe_sha256": arm.recipe_sha, "phases": phases,
            "updates_observed": len(updates), "seconds_per_update_median": statistics.median(gaps) if gaps else None,
            "peak_allocated_bytes": max((int(e.get("max_allocated_bytes", e.get("allocated_bytes", 0))) for e in updates), default=None),
            "peak_reserved_bytes": max((int(e.get("max_reserved_bytes", e.get("reserved_bytes", 0))) for e in updates), default=None),
            "save_seconds": save_seconds, "resume_to_first_update_seconds": resume_seconds,
            "masking_gate": str(arm.archive / "masking-gate.json"), "guard_logs": str(arm.runs_dir),
            "host_memory": "see guard/host-memory.jsonl in each run directory"}


def run_smoke(arm: Arm, first_stop: int, second_stop: int) -> int:
    if not 0 < first_stop < second_stop < arm.gates[0]:
        raise DriverError("smoke stops must lie before the first gate")
    phases: list[Json] = []
    for target in (first_stop, second_stop):
        steps = arm.durable_steps()
        latest = steps[-1] if steps else 0
        if latest >= target:
            continue
        if arm.settings.seconds_left() < arm.settings.min_train_seconds:
            return arm.finish(EXIT_DEFERRED, f"smoke: too close to the deadline before update {target}")
        started = time.time()
        code = arm.train(latest, stop_after_step=target)
        phases.append({"from": latest, "to": target, "exit": code, "started": started, "seconds": time.time() - started})
        if code != 0:
            return arm.finish(EXIT_FAILED, f"smoke: trainer exited {code}", needs_user="smoke trainer failed; see the run logs")
        steps = arm.durable_steps()
        if not steps or steps[-1] != target:
            return arm.finish(EXIT_DEFERRED, f"smoke: stopped at {steps[-1] if steps else 0} before {target}")
    receipt = smoke_receipt(arm, phases)
    write_json(arm.archive / "smoke-receipt.json", receipt)
    arm.evaluations.append({"smoke_receipt": str(arm.archive / "smoke-receipt.json")})
    return arm.finish(EXIT_DONE, f"smoke complete: {json.dumps({k: receipt[k] for k in ('seconds_per_update_median', 'peak_allocated_bytes', 'save_seconds', 'resume_to_first_update_seconds')})}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("arm", "baseline", "smoke"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--stop-after-gate", type=int, help="pilot arms: stop once this gate has a decision")
    parser.add_argument("--eval-budget-seconds", type=float, default=2400.0,
                        help="start a gate evaluation only if this many seconds remain before the deadline")
    parser.add_argument("--min-train-seconds", type=float, default=900.0)
    parser.add_argument("--first-stop", type=int, default=10)
    parser.add_argument("--second-stop", type=int, default=20)
    parser.add_argument("--no-guard", action="store_true", help="run children without cuda_host_guard (tests only)")
    args = parser.parse_args(argv)
    settings = Settings.from_env(args.recipe, guard=not args.no_guard, eval_budget_seconds=args.eval_budget_seconds,
                                 min_train_seconds=args.min_train_seconds, stop_after_gate=args.stop_after_gate)
    arm = Arm(settings)
    signal.signal(signal.SIGTERM, arm.on_signal)
    try:
        if args.command == "baseline":
            return run_baseline(arm)
        if args.command == "smoke":
            return run_smoke(arm, args.first_stop, args.second_stop)
        return arm.run()
    except (DriverError, EvaluationFailed) as error:
        return arm.finish(EXIT_FAILED, f"driver error: {error}", needs_user=str(error))


if __name__ == "__main__":
    raise SystemExit(main())
