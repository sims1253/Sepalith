#!/usr/bin/env python3
"""Night-queue driver for the C15 GRPO pilot and its smoke run.

``night run --recipe <pilot recipe>`` is the C04/C13 driver (``sft.night_arm.Arm``)
for GRPO. It never touches CUDA itself; it starts ``grpo.trainer`` and
``grpo.dev_generation`` one at a time, each under ``cuda_host_guard``:

1. Train until the next R6 evaluation step (every 50 updates), the night
   deadline or a stop request. The trainer saves full state at a rollout-buffer
   boundary and exits, which frees the GPU.
2. At an evaluation step run DEV75 (and DEV250 every 150 updates), apply R6
   (``grpo.rules.apply_r6``) against the previous evaluation, the run's start
   window and the starting model, and write ``decisions/decision-step-<n>-<k>.json``.
   An override file ``override-step-<n>.json`` takes precedence, as in C04/C13.
3. On continue, write ``continue-step-<n>.json`` (the trainer refuses to pass
   an evaluation step without it) and resume.

The pilot ends at R6 rollback (the result is the last evaluation checkpoint R6
continued from, or the starting model), at an R6 stop-for-user, or after the
evaluation at update 1,500; then ``pilot-summary.json`` compares the
evaluations with the starting model on DEV75 and DEV250. Publishing (R8) and
adoption are morning work.

``night smoke --recipe <smoke recipe>``: train to update 4 and save, evaluate
DEV75 on that checkpoint, resume to update 6 and save, then check the
telemetry, the resume and the evaluator result and write ``smoke-receipt.json``.

Exit codes for the night runner: 0 finished (pilot complete or rolled back,
smoke passed); 75 saved and deferred to the next night; 3 stopped for the user
(R6 stop-for-user, R7 disk; write an override file and ``night_queue retry``);
anything else is a failure, including a trainer that made no progress.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import signal
import statistics
import time
from typing import cast

from sepalith.training.eval.milestone_gate import requested_generation_settings
from sepalith.training.grpo import recipe as recipe_module, rules, state
from sepalith.training.sft.night_arm import (
    EXIT_DEFERRED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_NEEDS_USER,
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

Json = dict[str, object]
DECISION_SCHEMA = "sepalith.c15.grpo-r6-decision.v1"
SUMMARY_SCHEMA = "sepalith.c15.grpo-pilot-summary.v1"
SMOKE_SCHEMA = "sepalith.c15.grpo-smoke-receipt.v1"
TRAINER = "sepalith.training.grpo.trainer"
EVALUATOR = "sepalith.training.grpo.dev_generation"
TRAINER_NO_PROGRESS = 4
TRAINER_DISK = 5
PANEL_SECONDS = {"dev75": 900.0, "dev250": 1800.0}
STATUS_BY_OUTCOME = {rules.CONTINUE: "admitted_for_continuation", rules.STOP_FOR_USER: "stopped_for_user",
                     rules.ROLLBACK: "rolled_back"}
UPDATE_FIELDS = ("zero_variance_share_before", "zero_variance_share_after", "mean_abs_advantage", "reward_mean_drawn",
                 "exact_rate_drawn", "completion_tokens_mean_drawn", "clip_fraction")


def _obj(value: object, label: str) -> Json:
    if not isinstance(value, dict):
        raise DriverError(f"{label} must be an object")
    return cast(Json, value)


def counts_from(summary: Mapping[str, object], panel: str) -> Json:
    """R1 counts from one evaluator summary: DEV75 gives valid/cap/edit/nofp, DEV250 gives score250."""
    counts = _obj(summary.get("counts"), "counts")
    if panel == "dev75":
        return {"valid": counts["protocol_valid"], "cap": counts["cap_hit"], "edit": counts["edit_exact"],
                "nofp": counts["strict_noop_false_suggestions"], "dev75_predicted_noop": counts.get("predicted_noop")}
    return {"score250": cast(int, counts["edit_exact"]) + cast(int, counts["strict_noop_correct"]),
            "dev250_edit_exact": counts["edit_exact"], "dev250_noop_correct": counts["strict_noop_correct"]}


class GRPOArm(Arm):
    """``Arm`` for a C15 bound recipe."""

    def __init__(self, settings: Settings,
                 launcher_factory: Callable[[Settings, Path, Callable[[], bool]], Launcher] = Launcher, *,
                 root_override: Path | None = None, verify_files: bool = True) -> None:
        self.settings = settings
        self.recipe_path = settings.recipe_path.resolve()
        self.recipe = recipe_module.load(self.recipe_path, root_override=root_override, check_source=verify_files,
                                         check_data=verify_files)
        self.recipe_sha = sha256(self.recipe_path)
        self.kind = str(self.recipe["kind"])
        outputs = _obj(self.recipe["outputs"], "outputs")
        self.archive = Path(str(outputs["archive"]))
        self.trainer_root = Path(str(outputs["trainer"]))
        self.graceful_stop = Path(str(outputs["graceful_stop"]))
        self.gate = _obj(self.recipe["dev_gate"], "dev_gate")
        self.max_updates = int(cast(int, _obj(self.recipe["pre_registered"], "pre_registered")["max_updates"]))
        self.gates = rules.evaluation_steps(self.max_updates)
        self.evidence_dir = Path(str(self.gate["evidence_directory"]))
        self.runs_dir = self.evidence_dir / "runs"
        self.repository_root = recipe_module.repository_root(self.recipe, root_override)
        batching = _obj(self.gate["generation_batching"], "generation_batching")
        self.generation_batch_size = cast(int, batching["batch_size"])
        self.generation_kv_cache = cast(str, batching["kv_cache"])
        try:
            self.generation = requested_generation_settings(self.gate.get("generation_max_new_tokens"), self.generation_batch_size,
                                                            self.generation_kv_cache)
        except ValueError as error:
            raise DriverError(f"DEV generation settings: {error}") from None
        self.signalled = False
        self.launcher = launcher_factory(settings, self.runs_dir, self.stop_requested)
        self.evaluations: list[Json] = []
        self.rule_decisions: list[Json] = []

    # starting model and telemetry ---------------------------------------------------------------
    def start_metrics(self) -> Json:
        """The starting model's DEV75 and DEV250 counts from its bound gate results, re-read and re-hashed."""
        from sepalith.training.offline import rules as offline_rules

        reference = _obj(self.recipe["parent_reference"], "parent_reference")
        summaries = {}
        for name in ("dev75", "dev250"):
            entry = _obj(_obj(reference["results"], "results")[name], name)
            path = Path(str(entry["path"]))
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise DriverError(f"starting model {name} result differs")
            value = read_json(path)
            problems = self.generation_problems(name, value)
            if problems:
                raise DriverError(f"starting model: {problems}")
            summaries[name] = _obj(value["summary"], "summary")
        metrics = offline_rules.offline_metrics(summaries["dev75"], summaries["dev250"])
        if metrics != reference.get("metrics"):
            raise DriverError("starting model metrics do not match its bound results")
        return dict(metrics)

    def updates(self) -> list[Json]:
        """``grpo_update`` events, the last record of each update winning (a resumed segment redoes lost updates)."""
        path = self.archive / "telemetry.jsonl"
        latest: dict[int, Json] = {}
        if path.is_file():
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event.get("event") == "grpo_update" and type(event.get("update")) is int:
                    latest[int(event["update"])] = event
        return [latest[update] for update in sorted(latest)]

    def window(self, first: int, last: int) -> Json:
        try:
            return rules.window_stats(self.updates(), first, last)
        except ValueError as error:
            raise DriverError(f"telemetry window {first}-{last}: {error}") from None

    # decisions ----------------------------------------------------------------------------------
    def evidence_path(self, step: int) -> Path:
        return state.continue_record_path(self.recipe, step)

    def latest_decision(self, step: int) -> Json | None:
        paths = self.decisions(step)
        return read_json(paths[-1]) if paths else None

    def evaluation_metrics(self, step: int) -> Json:
        decision = self.latest_decision(step)
        if decision is None or decision.get("outcome") != rules.CONTINUE or not self.evidence_path(step).is_file():
            raise DriverError(f"evaluation {step} has no continue decision")
        return _obj(decision.get("metrics"), f"evaluation {step} metrics")

    def last_good(self, step: int) -> str:
        earlier = [gate for gate in self.gates if gate < step and self.evidence_path(gate).is_file()]
        return str(self.checkpoint(earlier[-1])) if earlier else str(_obj(self.recipe["parent"], "parent")["path"])

    def evaluate_panels(self, step: int, checkpoint: Path, *, parent_baseline: bool, out_dir: Path,
                        panels: Sequence[str] = ()) -> dict[str, Path]:
        if parent_baseline:
            raise DriverError("GRPO compares with the starting model's bound gate result; there is no step-0 baseline")
        attempt = stamp()
        bound = _obj(self.gate["panels"], "panels")
        results = {}
        for name in panels or rules.evaluation_panels(step):
            panel = _obj(bound[name], name)
            output = out_dir / f"dev-{name}-step-{step}.{attempt}.json"
            admission = out_dir / f"dev-{name}-step-{step}.{attempt}.admission.json"
            write_json(admission, {
                "schema": "sepalith.sft11.full-weight-edit-dev-generation-admission.v1", "status": "admitted",
                "purpose": "development_generation", "cuda_authorized": True, "final_set_access": False,
                "bound_recipe_sha256": self.recipe_sha, "checkpoint_manifest_sha256": sha256(checkpoint / "campaign-manifest.json"),
                "panel_sha256": panel["sha256"], "output": str(output),
                "deadline_seconds": int(min(3600, self.settings.eval_budget_seconds)),
            }, fresh=True)
            args = [EVALUATOR, "--bound-recipe", str(self.recipe_path), "--checkpoint", str(checkpoint),
                    "--dev-admission", str(admission), "--output", str(output), "--panel", name]
            code = self.launcher.run(f"dev-{name}-step-{step}", args, budget=self.settings.eval_budget_seconds, on_stop=self.terminate)
            if self.stop_requested():
                raise Stopped(f"stop requested during {name} generation")
            if code != 0 or not output.is_file():
                raise EvaluationFailed(f"{name} generation for step {step} exited {code} without a result")
            results[name] = output
            self.evaluations.append({"step": step, "panel": name, "result": str(output), "sha256": sha256(output)})
        return results

    def r6_inputs(self, step: int, counts: Json) -> dict[str, object]:
        start = self.start_metrics()
        previous = step - rules.DEV75_EVERY
        previous75 = start if previous <= 0 else self.evaluation_metrics(previous)
        dev250 = "score250" in counts
        previous_250_step = step - rules.DEV250_EVERY
        previous250 = (start if previous_250_step <= 0 else self.evaluation_metrics(previous_250_step)) if dev250 else None
        window = self.window(max(previous, 0) + 1, step)
        prior = []
        for earlier in (step - 2 * rules.DEV75_EVERY, step - rules.DEV75_EVERY):
            if earlier > 0:
                decision = self.latest_decision(earlier)
                if decision is None:
                    raise DriverError(f"evaluation {earlier} has no decision")
                prior.append(_obj(decision.get("window"), f"evaluation {earlier} window"))
        return {"start": start, "previous75": previous75, "previous250": previous250, "window": window,
                "prior_windows": prior, "start_window": self.window(1, rules.START_WINDOW),
                "reward_window": self.window(max(previous_250_step, 0) + 1, step) if dev250 else None}

    def decide(self, step: int, results: Mapping[str, Path] | None, failure: str | None,
               override: tuple[Path, Json] | None) -> tuple[str, Path]:
        checkpoint = self.checkpoint(step)
        manifest_sha = sha256(checkpoint / "campaign-manifest.json")
        problems = [failure] if failure else []
        metrics: Json | None = None
        inputs: dict[str, object] = {}
        rule: Json | None = None
        versus_start: Json | None = None
        if results is not None:
            for name, path in results.items():
                problems += self.result_problems(name, path, step, manifest_sha)
        if not problems and results is not None:
            counts: Json = {}
            for name, path in results.items():
                counts.update(counts_from(_obj(read_json(path)["summary"], "summary"), name))
            try:
                inputs = self.r6_inputs(step, counts)
            except (DriverError, ValueError, KeyError) as error:
                problems.append(f"R6 inputs unavailable: {error}")
            else:
                window = _obj(inputs["window"], "window")
                metrics = {**counts, "loss": window.get("loss")}
                rule = rules.apply_r6(metrics, previous75=_obj(inputs["previous75"], "previous75"),
                                      previous250=cast("Json | None", inputs["previous250"]), start=_obj(inputs["start"], "start"),
                                      window=window, prior_windows=cast(list[Json], inputs["prior_windows"]),
                                      start_window=_obj(inputs["start_window"], "start_window"),
                                      reward_window=cast("Json | None", inputs["reward_window"]))
                versus_start = rules.r1_versus_start(metrics, _obj(inputs["start"], "start"))
        if problems or rule is None:
            rule = rules.failure("; ".join(problems) or "no rule inputs")
        outcome = str(rule["outcome"])
        decided_by = "rule_R6"
        override_record = None
        if override is not None:
            override_record = {"path": str(override[0]), "sha256": sha256(override[0]), "outcome": override[1]["outcome"],
                               "reason": override[1]["reason"], "author": override[1].get("author")}
            outcome, decided_by = str(override[1]["outcome"]), "human_override"
            if outcome == rules.CONTINUE and (results is None or any("binding" in p for p in problems)):
                outcome, decided_by = rules.STOP_FOR_USER, "human_override_refused_without_bound_results"
        decision: Json = {
            "schema": DECISION_SCHEMA, "status": STATUS_BY_OUTCOME[outcome], "continue_training": outcome == rules.CONTINUE,
            "step": step, "panels": sorted(results) if results else list(rules.evaluation_panels(step)), "outcome": outcome,
            "decided_by": decided_by, "rule": rule, "rule_outcome": rule["outcome"], "override": override_record,
            "metrics": metrics, "window": inputs.get("window"), "r1_versus_start": versus_start,
            "evaluation_complete": results is not None, "bound_recipe_sha256": self.recipe_sha,
            "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": manifest_sha,
            "results": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (results or {}).items()},
            "generation_runtime": self.generation,
            "evaluator_source_sha256": recipe_module.evaluator_hashes(self.repository_root),
            "last_good_checkpoint": self.last_good(step) if outcome == rules.ROLLBACK else None,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        attempt = len(self.decisions(step)) + 1
        path = self.evidence_dir / "decisions" / f"decision-step-{step}-{attempt}.json"
        write_json(path, decision, fresh=True)
        if outcome == rules.CONTINUE and results is not None:
            write_json(self.evidence_path(step), state.continue_record(self.recipe_sha, step, checkpoint, path), fresh=True)
        self.rule_decisions.append({"rule": "R6", "step": step, "kind": rule.get("kind"), "inputs": {"metrics": metrics,
                                    "window": inputs.get("window")}, "rows": rule.get("rows"), "rule_outcome": rule["outcome"],
                                    "outcome": outcome, "decided_by": decided_by, "decision": str(path)})
        return outcome, path

    def gate_step(self, step: int) -> str:
        """Bring the evaluation at ``step`` to an R6 outcome, or ``defer``."""
        if self.evidence_path(step).is_file():
            return rules.CONTINUE
        decisions = self.decisions(step)
        override = self.override(step)
        prior = read_json(decisions[-1]) if decisions else None
        results = None
        if prior is not None and prior.get("evaluation_complete") is True:
            prior_override = _obj(prior["override"], "override")["sha256"] if prior.get("override") else None
            if override is None or sha256(override[0]) == prior_override:
                return str(prior["outcome"])
            entries = _obj(prior["results"], "results")
            results = {name: Path(str(_obj(entries[name], name)["path"])) for name in entries}
        if results is None:
            needed = sum(PANEL_SECONDS[name] for name in rules.evaluation_panels(step))
            if self.settings.seconds_left() < needed:
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

    # summary --------------------------------------------------------------------------------------
    def nightly_candidate(self) -> str | None:
        """R8: the latest evaluation checkpoint with DEV250 that passes R1 against the starting model."""
        for step in reversed(self.gates):
            decision = self.latest_decision(step)
            versus = _obj(decision.get("r1_versus_start") or {}, "r1_versus_start") if decision else {}
            if decision and decision.get("outcome") == rules.CONTINUE and versus.get("with_score250") \
                    and versus.get("outcome") == rules.CONTINUE:
                return str(self.checkpoint(step))
        return None

    def pilot_summary(self, status: str, result_checkpoint: str) -> Json:
        start = self.start_metrics()
        evaluations = []
        for step in self.gates:
            decision = self.latest_decision(step)
            if decision is not None:
                evaluations.append({"step": step, "outcome": decision.get("outcome"), "metrics": decision.get("metrics"),
                                    "r1_versus_start": _obj(decision.get("r1_versus_start") or {}, "r1").get("outcome"),
                                    "decision": str(self.decisions(step)[-1])})
        updates = self.updates()
        summary: Json = {"schema": SUMMARY_SCHEMA, "arm": _obj(self.recipe["arm"], "arm")["name"], "status": status,
                         "start": {"checkpoint": _obj(self.recipe["parent"], "parent")["path"], "metrics": start},
                         "result_checkpoint": result_checkpoint, "evaluations": evaluations,
                         "updates_completed": updates[-1]["update"] if updates else 0,
                         "nightly_candidate_r8": self.nightly_candidate(),
                         "adoption": "not decided by the driver: the morning summary compares these evaluations with the "
                                     "start on DEV75 and DEV250; only an adopted checkpoint gets a stable grpo-<n>/ folder (R8)",
                         "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(self.archive / "pilot-summary.json", summary)
        return summary

    def finish(self, code: int, summary: str, *, needs_user: str | None = None, arm_result: Json | None = None) -> int:
        candidate = self.nightly_candidate() if self.kind == "pilot" else None
        if candidate is not None:
            summary = f"{summary}; R8 nightly candidate {candidate}"
        return super().finish(code, summary, needs_user=needs_user, arm_result=arm_result)

    # training -------------------------------------------------------------------------------------
    def train(self, latest: int, stop_after_step: int | None = None) -> int:
        if latest == 0 and self.archive.exists():
            moment = int(time.time())
            self.archive.rename(self.archive.with_name(f"{self.archive.name}.stale-{moment}"))
        self.clear_stale_stop_file()
        args = [TRAINER, "run", "--recipe", str(self.recipe_path)]
        if latest:
            args += ["--resume", str(self.checkpoint(latest))]
        if stop_after_step is not None:
            args += ["--stop-after-update", str(stop_after_step)]
        return self.launcher.run(f"train-from-{latest}", args, budget=None, on_stop=self.request_trainer_stop)

    def trainer_exit(self, code: int, latest: int) -> int | None:
        """The job's exit for a trainer exit code, or None to keep going."""
        arm = str(_obj(self.recipe["arm"], "arm")["name"])
        if code == 0:
            return None
        if code == TRAINER_DISK:
            return self.finish(EXIT_NEEDS_USER, f"{arm}: R7 checkpoint disk below 150 GB; nothing saved after update {latest}",
                               needs_user=f"{arm}: R7 says stop-for-user: free space on the checkpoint disk before the next save")
        if code == TRAINER_NO_PROGRESS and (self.stop_requested() or self.settings.seconds_left() < self.settings.min_train_seconds):
            return self.finish(EXIT_DEFERRED, f"{arm}: no buffer fitted before the deadline; resumes at {latest}")
        return self.finish(EXIT_FAILED, f"{arm}: trainer exited {code} from update {latest}",
                           needs_user=f"{arm}: trainer failed from update {latest} (exit {code}); see {self.runs_dir}")

    def run(self) -> int:
        arm = str(_obj(self.recipe["arm"], "arm")["name"])
        if self.kind != "pilot":
            raise DriverError("night run takes the pilot recipe; use night smoke for the smoke recipe")
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
                    return self.finish(EXIT_DEFERRED, f"{arm}: evaluation {latest} deferred to the next night")
                if outcome == rules.ROLLBACK:
                    good = self.last_good(latest)
                    self.pilot_summary("rolled_back", good)
                    return self.finish(EXIT_DONE, f"{arm}: R6 rollback at update {latest}; result is {good}",
                                       arm_result={"status": "rolled_back", "gate": latest, "result_checkpoint": good})
                if outcome == rules.STOP_FOR_USER:
                    return self.finish(EXIT_NEEDS_USER, f"{arm}: R6 stop-for-user at update {latest}",
                                       needs_user=(f"{arm} update {latest}: R6 says stop-for-user; see {self.decisions(latest)[-1]}. "
                                                   f"To decide, write {self.evidence_dir}/override-step-{latest}.json and retry the job."))
                if latest == final:
                    self.pilot_summary("complete", str(self.checkpoint(final)))
                    return self.finish(EXIT_DONE, f"{arm}: complete at update {final}",
                                       arm_result={"status": "complete", "gate": final, "result_checkpoint": str(self.checkpoint(final))})
            if latest == previous_latest:
                raise DriverError(f"the trainer exited without progress past update {latest}")
            if self.settings.seconds_left() < self.settings.min_train_seconds:
                return self.finish(EXIT_DEFERRED, f"{arm}: too close to the deadline to train; resumes at {latest}")
            previous_latest = latest
            exit_code = self.trainer_exit(self.train(latest), latest)
            if exit_code is not None:
                return exit_code
            if self.stop_requested() or self.settings.seconds_left() <= 0:
                steps = self.durable_steps()
                now_latest = steps[-1] if steps else 0
                if now_latest not in self.gates:
                    return self.finish(EXIT_DEFERRED, f"{arm}: saved at update {now_latest} at the deadline")


# Smoke ----------------------------------------------------------------------------------------------

def smoke_receipt(arm: GRPOArm, stops: Sequence[int], phases: Sequence[Json], result: Path) -> tuple[Json, list[str]]:
    problems: list[str] = []
    updates = arm.updates()
    numbers = [int(cast(int, event["update"])) for event in updates]
    if numbers != list(range(1, stops[-1] + 1)):
        problems.append(f"telemetry updates {numbers} differ from 1..{stops[-1]}")
    for event in updates:
        missing = [name for name in UPDATE_FIELDS if not isinstance(event.get(name), (int, float))
                   or not math.isfinite(float(cast(float, event[name])))]
        if missing and event.get("groups_trained"):
            problems.append(f"update {event['update']} lacks finite {missing}")
    resumed = [event.get("resumed_from") for event in updates if cast(int, event["update"]) > stops[0]]
    if any(value != stops[0] for value in resumed):
        problems.append(f"updates after {stops[0]} were not resumed from checkpoint {stops[0]}: {resumed}")
    if not any(event.get("optimizer_step") for event in updates):
        problems.append("no optimizer step ran (no update had a signal group)")
    for step in stops:
        try:
            state.verify_checkpoint(arm.checkpoint(step), arm.recipe_sha)
        except state.StateError as error:
            problems.append(f"checkpoint {step}: {error}")
    value = read_json(result)
    problems += arm.result_problems("dev75", result, stops[0], sha256(arm.checkpoint(stops[0]) / "campaign-manifest.json"))
    counts = counts_from(_obj(value["summary"], "summary"), "dev75")
    seconds = [float(cast(float, e["update_seconds"])) for e in updates]
    rollout = [float(cast(float, e["rollout_seconds"])) for e in updates if e.get("buffer_position") == 0]
    events = [json.loads(line) for line in (arm.archive / "telemetry.jsonl").read_text().splitlines()]
    saves = [float(e["seconds"]) for e in events if e.get("event") == "checkpoint_durable"]
    peaks = [int(cast(int, e["max_memory_reserved"])) for e in updates if isinstance(e.get("max_memory_reserved"), int)]
    receipt: Json = {
        "schema": SMOKE_SCHEMA, "recipe_sha256": arm.recipe_sha, "stops": list(stops), "phases": list(phases),
        "updates": numbers, "dev75": {"result": str(result), "sha256": sha256(result), "counts": counts},
        "update_seconds_median": statistics.median(seconds) if seconds else None,
        "rollout_seconds_per_buffer": rollout, "save_seconds": saves,
        "seconds_per_update_including_rollout": ((sum(seconds) + sum(rollout)) / len(seconds)) if seconds else None,
        "peak_reserved_mib": round(max(peaks) / 2**20, 1) if peaks else None,
        "zero_variance_share_before": [e.get("zero_variance_share_before") for e in updates],
        "clip_fraction": [e.get("clip_fraction") for e in updates],
        "problems": problems, "status": "pass" if not problems else "fail",
    }
    return receipt, problems


def run_smoke(arm: GRPOArm) -> int:
    smoke = _obj(arm.recipe.get("smoke"), "smoke")
    stops = [int(x) for x in cast(list[int], smoke["stops"])]
    eval_step = int(cast(int, smoke["eval_step"]))
    if eval_step != stops[0]:
        raise DriverError("the smoke evaluates the first save")
    phases: list[Json] = []
    result: Path | None = None
    for index, target in enumerate(stops):
        steps = arm.durable_steps()
        latest = steps[-1] if steps else 0
        if latest < target:
            if latest != (stops[index - 1] if index else 0):
                raise DriverError(f"smoke archive is at update {latest}, expected {stops[index - 1] if index else 0}")
            if arm.settings.seconds_left() < arm.settings.min_train_seconds:
                return arm.finish(EXIT_DEFERRED, f"smoke: too close to the deadline before update {target}")
            started = time.time()
            code = arm.train(latest, stop_after_step=target)
            phases.append({"from": latest, "to": target, "exit": code, "started": started, "seconds": time.time() - started})
            exit_code = arm.trainer_exit(code, latest)
            if exit_code is not None:
                return exit_code
            steps = arm.durable_steps()
            if not steps or steps[-1] != target:
                return arm.finish(EXIT_FAILED, f"smoke: trainer stopped at {steps[-1] if steps else 0} before {target}",
                                  needs_user="smoke trainer saved no checkpoint at its target; see the run logs")
        if index == 0:
            found = sorted((arm.evidence_dir / "results").glob(f"dev-dev75-step-{eval_step}.*.json"))
            found = [path for path in found if not path.name.endswith(".admission.json")]
            if found:
                result = found[-1]
            else:
                try:
                    result = arm.evaluate_panels(eval_step, arm.checkpoint(eval_step), parent_baseline=False,
                                                 out_dir=arm.evidence_dir / "results", panels=("dev75",))["dev75"]
                except Stopped:
                    return arm.finish(EXIT_DEFERRED, "smoke: stopped during the DEV75 evaluation")
                except EvaluationFailed as error:
                    return arm.finish(EXIT_FAILED, f"smoke: {error}", needs_user=f"smoke DEV75 evaluation failed: {error}")
    receipt, problems = smoke_receipt(arm, stops, phases, cast(Path, result))
    write_json(arm.archive / "smoke-receipt.json", receipt)
    arm.evaluations.append({"smoke_receipt": str(arm.archive / "smoke-receipt.json")})
    if problems:
        return arm.finish(EXIT_FAILED, f"smoke failed: {problems}", needs_user=f"C15 smoke failed: {problems[0]}")
    keys = ("update_seconds_median", "seconds_per_update_including_rollout", "save_seconds", "peak_reserved_mib")
    return arm.finish(EXIT_DONE, f"smoke passed: {json.dumps({key: receipt[key] for key in keys})}; DEV75 {json.dumps(receipt['dev75'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "smoke"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--eval-budget-seconds", type=float, default=2400.0)
    parser.add_argument("--min-train-seconds", type=float, default=1200.0)
    parser.add_argument("--no-guard", action="store_true", help="run children without cuda_host_guard (tests only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(args.recipe, guard=not args.no_guard, eval_budget_seconds=args.eval_budget_seconds,
                                 min_train_seconds=args.min_train_seconds)
    arm = GRPOArm(settings)
    signal.signal(signal.SIGTERM, arm.on_signal)
    try:
        if args.command == "smoke":
            if arm.kind != "smoke":
                raise DriverError("night smoke takes the smoke recipe")
            return run_smoke(arm)
        return arm.run()
    except (DriverError, EvaluationFailed, ValueError, state.StateError) as error:
        return arm.finish(EXIT_FAILED, f"driver error: {error}", needs_user=str(error))


if __name__ == "__main__":
    raise SystemExit(main())
