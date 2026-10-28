#!/usr/bin/env python3
"""Night-queue driver for the C18 GRPO pilot on the exec environment and its smoke run.

``night run --recipe <pilot recipe>`` is the C15 driver (``training.grpo.night.GRPOArm``) with the C18
recipe, trainer and gate. It never touches CUDA itself; every GPU step is a child process under
``cuda_host_guard``, one at a time:

0. Once per pilot: the starting model's validation exec result and CPT 2K NLL (``start/``), measured
   with the same evaluators as every later checkpoint. Its DEV75/DEV250 counts are its bound gate-803
   results.
1. Train until the next evaluation step (every 50 updates), the deadline or a stop request
   (``grpo_exec.trainer``; full state saved at a rollout-buffer boundary).
2. At an evaluation step: DEV75, the validation exec gate and the CPT 2K probe (and DEV250 every 150
   updates); R6x (``grpo_exec.rules.apply``) writes ``decisions/decision-step-<n>-<k>.json``; an
   ``override-step-<n>.json`` takes precedence; on continue, ``continue-step-<n>.json`` lets the
   trainer pass the step.
3. The pilot ends at an R6x rollback, after the update-1,000 evaluation, or at a plateau (no accepted
   evaluation in the last 6 beat the earlier best validation all-pass by 1 point). It then selects the
   accepted checkpoint with the best validation all-pass, runs DEV250 on it if it has none, and writes
   ``pilot-summary.json`` plus the final hold-out evaluation job (``final-holdout-job.json``) for the
   orchestrator to enqueue. The frozen hold-out is never read here.

``night smoke --recipe <smoke recipe>``: train to update 4 and save; DEV75, the first 96 validation
tasks and the CPT probe on checkpoint 4 plus the CPT probe of arm A 803; resume to update 6 and save;
check telemetry, resume and results; write ``smoke-receipt.json``.

Exit codes for the night runner: 0 finished; 75 saved and deferred; 3 stopped for the user;
anything else a failure.
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
from sepalith.training.grpo import rules as r6, state
from sepalith.training.grpo.night import TRAINER_DISK, TRAINER_NO_PROGRESS, GRPOArm, counts_from
from sepalith.training.grpo_exec import recipe as recipe_module, rules
from sepalith.training.sft.night_arm import (
    EXIT_DEFERRED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_NEEDS_USER,
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
DECISION_SCHEMA = "sepalith.c18.grpo-exec-r6x-decision.v1"
SUMMARY_SCHEMA = "sepalith.c18.grpo-exec-pilot-summary.v1"
SMOKE_SCHEMA = "sepalith.c18.grpo-exec-smoke-receipt.v1"
TRAINER = "sepalith.training.grpo_exec.trainer"
DEV_EVALUATOR = "sepalith.training.grpo_exec.dev_generation"
VAL_EVALUATOR = "sepalith.training.grpo_exec.val_eval"
CPT_PROBE = "sepalith.training.grpo_exec.cpt_probe"
VAL_SCHEMA = "sepalith.c18.val-exec-result.v1"
CPT_SCHEMA = "sepalith.c18.cpt-probe-result.v1"
PANEL_SECONDS = {"dev75": 900.0, "dev250": 1800.0, "val": 1500.0, "cpt": 360.0}
STATUS_BY_OUTCOME = {rules.CONTINUE: "admitted_for_continuation", rules.STOP_FOR_USER: "stopped_for_user",
                     rules.ROLLBACK: "rolled_back"}
UPDATE_FIELDS = ("zero_variance_share_before", "zero_variance_share_after", "mean_abs_advantage", "reward_mean_drawn",
                 "exact_rate_drawn", "completion_tokens_mean_drawn", "clip_fraction", "replay_nll", "grad_norm",
                 "grad_norm_grpo")
WINDOW_SECONDS = 60_000.0  # 00:00 to 16:40


def _obj(value: object, label: str) -> Json:
    if not isinstance(value, dict):
        raise DriverError(f"{label} must be an object")
    return cast(Json, value)


class ExecGRPOArm(GRPOArm):
    """``GRPOArm`` for a C18 bound recipe."""

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
        self.start_dir = self.evidence_dir / "start"
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

    # telemetry ----------------------------------------------------------------------------------
    def window(self, first: int, last: int) -> Json:
        try:
            return rules.window_stats(self.updates(), first, last)
        except ValueError as error:
            raise DriverError(f"telemetry window {first}-{last}: {error}") from None

    # result bindings -----------------------------------------------------------------------------
    def val_problems(self, path: Path, step: int, manifest_sha: str) -> list[str]:
        value = read_json(path)
        val = _obj(self.gate["val"], "val")
        tasks = _obj(val["tasks"], "tasks")
        expected = int(cast(int, val["limit"])) if val.get("limit") is not None else int(cast(int, val["n_tasks"]))
        reward = _obj(_obj(_obj(self.recipe["pre_registered"], "pre_registered")["reward"], "reward")["exec"], "exec")
        result_tasks = _obj(value.get("tasks"), "tasks")
        checks = {"schema": value.get("schema") == VAL_SCHEMA and value.get("status") == "complete",
                  "step": value.get("step") == step,
                  "recipe": _obj(value.get("bound_recipe"), "bound_recipe").get("sha256") == self.recipe_sha,
                  "checkpoint": _obj(value.get("checkpoint"), "checkpoint").get("campaign_manifest_sha256") == manifest_sha,
                  "tasks": result_tasks.get("sha256") == tasks["sha256"] and result_tasks.get("n") == expected,
                  "gate": _obj(value.get("gate_version"), "gate_version").get("fingerprint") == _obj(reward["gate"], "gate")["fingerprint"],
                  "holdout": value.get("holdout_access") is False and value.get("final_set_access") is False}
        return [f"val result {key} binding differs" for key, ok in checks.items() if not ok]

    def cpt_problems(self, path: Path, step: int, manifest_sha: str) -> list[str]:
        value = read_json(path)
        cpt = _obj(self.gate["cpt"], "cpt")
        checks = {"schema": value.get("schema") == CPT_SCHEMA and value.get("status") == "complete",
                  "step": value.get("step") == step,
                  "recipe": _obj(value.get("bound_recipe"), "bound_recipe").get("sha256") == self.recipe_sha,
                  "checkpoint": _obj(value.get("checkpoint"), "checkpoint").get("campaign_manifest_sha256") == manifest_sha,
                  "panel": _obj(value.get("panel"), "panel").get("sha256") == _obj(cpt["panel"], "panel")["sha256"]
                  and _obj(value.get("panel"), "panel").get("rows") == cpt["rows"],
                  "finite": isinstance(value.get("mean_causal_nll"), float) and math.isfinite(cast(float, value["mean_causal_nll"]))}
        return [f"cpt result {key} binding differs" for key, ok in checks.items() if not ok]

    def problems_for(self, name: str, path: Path, step: int, manifest_sha: str) -> list[str]:
        if name == "val":
            return self.val_problems(path, step, manifest_sha)
        if name == "cpt":
            return self.cpt_problems(path, step, manifest_sha)
        return self.result_problems(name, path, step, manifest_sha)

    # evaluation processes ------------------------------------------------------------------------
    def run_child(self, label: str, args: list[str], output: Path, budget: float) -> Path:
        code = self.launcher.run(label, args, budget=budget, on_stop=self.terminate)
        if self.stop_requested():
            raise Stopped(f"stop requested during {label}")
        if code != 0 or not output.is_file():
            raise EvaluationFailed(f"{label} exited {code} without a result")
        self.evaluations.append({"label": label, "result": str(output), "sha256": sha256(output)})
        return output

    def evaluate_panels(self, step: int, checkpoint: Path, *, parent_baseline: bool, out_dir: Path,
                        panels: Sequence[str] = ()) -> dict[str, Path]:
        if parent_baseline:
            raise DriverError("the starting model's DEV counts are its bound gate-803 results; use start_measurements")
        attempt = stamp()
        out_dir.mkdir(parents=True, exist_ok=True)
        bound = _obj(self.gate["panels"], "panels")
        budget = self.settings.eval_budget_seconds
        results: dict[str, Path] = {}
        for name in panels or rules.evaluation_panels(step):
            output = out_dir / f"{name}-step-{step}.{attempt}.json"
            if name in ("dev75", "dev250"):
                panel = _obj(bound[name], name)
                admission = out_dir / f"{name}-step-{step}.{attempt}.admission.json"
                write_json(admission, {
                    "schema": "sepalith.sft11.full-weight-edit-dev-generation-admission.v1", "status": "admitted",
                    "purpose": "development_generation", "cuda_authorized": True, "final_set_access": False,
                    "bound_recipe_sha256": self.recipe_sha, "checkpoint_manifest_sha256": sha256(checkpoint / "campaign-manifest.json"),
                    "panel_sha256": panel["sha256"], "output": str(output), "deadline_seconds": int(min(3600, budget)),
                }, fresh=True)
                args = [DEV_EVALUATOR, "--bound-recipe", str(self.recipe_path), "--checkpoint", str(checkpoint),
                        "--dev-admission", str(admission), "--output", str(output), "--panel", name]
            else:
                module = VAL_EVALUATOR if name == "val" else CPT_PROBE
                args = [module, "--bound-recipe", str(self.recipe_path), "--checkpoint", str(checkpoint), "--output", str(output)]
            results[name] = self.run_child(f"{name}-step-{step}", args, output, budget)
        return results

    def start_measurements(self) -> Json | str:
        """The starting model's validation all-pass, mean reward and CPT NLL; runs them once; 'defer' if out of time."""
        found: dict[str, Path] = {}
        self.start_dir.mkdir(parents=True, exist_ok=True)
        for name in ("val", "cpt"):
            done = sorted(p for p in self.start_dir.glob(f"{name}-start.*.json") if ".records." not in p.name and ".rows." not in p.name)
            if done:
                found[name] = done[-1]
                continue
            if self.settings.seconds_left() < PANEL_SECONDS[name]:
                return "defer"
            output = self.start_dir / f"{name}-start.{stamp()}.json"
            module = VAL_EVALUATOR if name == "val" else CPT_PROBE
            found[name] = self.run_child(f"{name}-start", [module, "--bound-recipe", str(self.recipe_path), "--start",
                                                           "--output", str(output)], output, self.settings.eval_budget_seconds)
        manifest = _obj(self.recipe["parent"], "parent")["files"]
        problems = self.val_problems(found["val"], 0, str(_obj(manifest, "files")["campaign-manifest.json"])) + \
            self.cpt_problems(found["cpt"], 0, str(_obj(manifest, "files")["campaign-manifest.json"]))
        if problems:
            raise DriverError(f"starting model measurements: {problems}")
        overall = _obj(_obj(read_json(found["val"])["metrics"], "metrics")["overall"], "overall")
        return {"val_all_pass": overall["all_pass"], "val_mean_reward": overall["mean_reward"],
                "cpt_nll": read_json(found["cpt"])["mean_causal_nll"],
                "val_result": str(found["val"]), "cpt_result": str(found["cpt"])}

    def start_reference(self) -> Json:
        measured = self.start_measurements()
        if measured == "defer":
            raise DriverError("starting model measurements are missing")
        return {**self.start_metrics(), **_obj(measured, "start measurements")}

    # decisions -------------------------------------------------------------------------------------
    def current_metrics(self, results: Mapping[str, Path]) -> Json:
        counts: Json = {}
        for name in ("dev75", "dev250"):
            if name in results:
                counts.update(counts_from(_obj(read_json(results[name])["summary"], "summary"), name))
        overall = _obj(_obj(read_json(results["val"])["metrics"], "metrics")["overall"], "overall")
        counts.update(val_all_pass=overall["all_pass"], val_mean_reward=overall["mean_reward"],
                      val_exact_match=overall.get("exact_match"), val_protocol_valid=overall.get("protocol_valid"),
                      cpt_nll=read_json(results["cpt"])["mean_causal_nll"])
        return counts

    def rule_inputs(self, step: int) -> dict[str, object]:
        start = self.start_reference()
        previous = step - rules.EVAL_EVERY
        previous75 = start if previous <= 0 else self.evaluation_metrics(previous)
        prior = []
        for earlier in (step - 2 * rules.EVAL_EVERY, step - rules.EVAL_EVERY):
            if earlier > 0:
                decision = self.latest_decision(earlier)
                if decision is None:
                    raise DriverError(f"evaluation {earlier} has no decision")
                prior.append(_obj(decision.get("window"), f"evaluation {earlier} window"))
        return {"start": start, "previous75": previous75, "window": self.window(max(previous, 0) + 1, step),
                "prior_windows": prior, "start_window": self.window(1, rules.START_WINDOW)}

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
                problems += self.problems_for(name, path, step, manifest_sha)
            missing = set(rules.evaluation_panels(step)) - set(results)
            if missing:
                problems.append(f"panels missing: {sorted(missing)}")
        if not problems and results is not None:
            try:
                counts = self.current_metrics(results)
                inputs = self.rule_inputs(step)
            except (DriverError, ValueError, KeyError) as error:
                problems.append(f"R6x inputs unavailable: {error}")
            else:
                window = _obj(inputs["window"], "window")
                metrics = {**counts, "loss": window.get("loss")}
                rule = rules.apply(metrics, previous75=_obj(inputs["previous75"], "previous75"), start=_obj(inputs["start"], "start"),
                                   window=window, prior_windows=cast(list[Json], inputs["prior_windows"]),
                                   start_window=_obj(inputs["start_window"], "start_window"))
                versus_start = r6.r1_versus_start(metrics, _obj(inputs["start"], "start"))
        if problems or rule is None:
            rule = rules.failure("; ".join(problems) or "no rule inputs")
        outcome = str(rule["outcome"])
        decided_by = "rule_R6x"
        override_record = None
        if override is not None:
            override_record = {"path": str(override[0]), "sha256": sha256(override[0]), "outcome": override[1]["outcome"],
                               "reason": override[1]["reason"], "author": override[1].get("author")}
            outcome, decided_by = str(override[1]["outcome"]), "human_override"
            if outcome == rules.CONTINUE and (results is None or metrics is None or any("binding" in p for p in problems)):
                outcome, decided_by = rules.STOP_FOR_USER, "human_override_refused_without_bound_results"
        decision: Json = {
            "schema": DECISION_SCHEMA, "status": STATUS_BY_OUTCOME[outcome], "continue_training": outcome == rules.CONTINUE,
            "step": step, "panels": sorted(results) if results else list(rules.evaluation_panels(step)), "outcome": outcome,
            "decided_by": decided_by, "rule": rule, "rule_outcome": rule["outcome"], "override": override_record,
            "metrics": metrics, "window": inputs.get("window"), "start": inputs.get("start"), "r1_versus_start": versus_start,
            "evaluation_complete": results is not None and metrics is not None, "bound_recipe_sha256": self.recipe_sha,
            "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": manifest_sha,
            "results": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (results or {}).items()},
            "generation_runtime": self.generation,
            "evaluator_source_sha256": recipe_module.evaluator_hashes(self.repository_root),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        attempt = len(self.decisions(step)) + 1
        path = self.evidence_dir / "decisions" / f"decision-step-{step}-{attempt}.json"
        write_json(path, decision, fresh=True)
        if outcome == rules.CONTINUE and metrics is not None:
            write_json(self.evidence_path(step), state.continue_record(self.recipe_sha, step, checkpoint, path), fresh=True)
        self.rule_decisions.append({"rule": "R6x", "step": step, "kind": rule.get("kind"), "inputs": {"metrics": metrics},
                                    "rows": rule.get("rows"), "rule_outcome": rule["outcome"], "outcome": outcome,
                                    "decided_by": decided_by, "decision": str(path)})
        return outcome, path

    def gate_step(self, step: int) -> str:
        """Bring the evaluation at ``step`` to an R6x outcome, or ``defer``."""
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

    # selection and summary ------------------------------------------------------------------------
    def accepted(self, up_to: int) -> list[Json]:
        out = []
        for step in self.gates:
            if step > up_to or not self.evidence_path(step).is_file():
                continue
            decision = self.latest_decision(step)
            metrics = _obj(decision.get("metrics") if decision else None, f"evaluation {step} metrics")
            out.append({"step": step, "val_all_pass": metrics["val_all_pass"], "val_mean_reward": metrics["val_mean_reward"],
                        "cpt_nll": metrics["cpt_nll"], "score250": metrics.get("score250")})
        return out

    def plateaued(self, latest: int) -> bool:
        start = self.start_reference()
        return rules.plateau(float(cast(float, start["val_all_pass"])), self.accepted(latest))

    def selected_dev250(self, step: int) -> Path | str:
        """The selected checkpoint's DEV250 result: from its evaluation, a post-pilot run, or 'defer'."""
        decision = self.latest_decision(step)
        entries = _obj(decision.get("results") if decision else {}, "results")
        if "dev250" in entries:
            return Path(str(_obj(entries["dev250"], "dev250")["path"]))
        done = sorted((self.evidence_dir / "selection").glob(f"dev250-step-{step}.*.json"))
        done = [p for p in done if not p.name.endswith(".admission.json")]
        if done:
            return done[-1]
        if self.settings.seconds_left() < PANEL_SECONDS["dev250"]:
            return "defer"
        try:
            return self.evaluate_panels(step, self.checkpoint(step), parent_baseline=False,
                                        out_dir=self.evidence_dir / "selection", panels=("dev250",))["dev250"]
        except Stopped:
            return "defer"

    def finish_pilot(self, status: str, latest: int) -> int:
        arm = str(_obj(self.recipe["arm"], "arm")["name"])
        start = self.start_reference()
        accepted = self.accepted(latest)
        chosen = rules.select(accepted)
        summary: Json = {"schema": SUMMARY_SCHEMA, "arm": arm, "status": status, "last_evaluation": latest,
                         "start": {"checkpoint": _obj(self.recipe["parent"], "parent")["path"], "metrics": start},
                         "accepted": accepted, "updates_completed": (self.updates() or [{"update": 0}])[-1]["update"],
                         "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if chosen is None:
            summary.update(selected=None, result_checkpoint=str(_obj(self.recipe["parent"], "parent")["path"]),
                           final_evaluation="none: no accepted evaluation, the result is arm A 803")
            write_json(self.archive / "pilot-summary.json", summary)
            return self.finish(EXIT_DONE, f"{arm}: {status} at update {latest} without an accepted evaluation; result is 803",
                               arm_result={"status": status, "gate": latest, "result_checkpoint": summary["result_checkpoint"]})
        step = int(cast(int, chosen["step"]))
        dev250 = self.selected_dev250(step)
        if dev250 == "defer":
            return self.finish(EXIT_DEFERRED, f"{arm}: {status}; DEV250 of the selected checkpoint {step} deferred")
        dev250_path = cast(Path, dev250)
        problems = self.result_problems("dev250", dev250_path, step, sha256(self.checkpoint(step) / "campaign-manifest.json"))
        if problems:
            return self.finish(EXIT_NEEDS_USER, f"{arm}: selected DEV250 does not bind: {problems}", needs_user=str(problems))
        score = counts_from(_obj(read_json(dev250_path)["summary"], "summary"), "dev250")["score250"]
        from sepalith.training.grpo_exec.jobs import final_holdout_job

        selected = str(self.checkpoint(step))
        job = final_holdout_job(self.recipe, selected, self.evidence_dir / "final", self.repository_root)
        job_path = self.evidence_dir / "final-holdout-job.json"
        write_json(job_path, job)
        summary.update(selected={**chosen, "checkpoint": selected, "dev250": {"path": str(dev250_path), "score250": score},
                                 "cpt_rise_vs_start": float(cast(float, chosen["cpt_nll"])) - float(cast(float, start["cpt_nll"]))},
                       result_checkpoint=selected,
                       final_evaluation={"job": str(job_path), "status": "to enqueue (orchestrator)",
                                         "verdict": "rules.final_verdict on the job's metrics.json once it finishes"})
        write_json(self.archive / "pilot-summary.json", summary)
        return self.finish(EXIT_DONE, f"{arm}: {status} at update {latest}; selected checkpoint {step} (val all-pass "
                                      f"{chosen['val_all_pass']}); enqueue {job_path}",
                           arm_result={"status": status, "gate": latest, "result_checkpoint": selected,
                                       "final_holdout_job": str(job_path)})

    # training -------------------------------------------------------------------------------------
    def train(self, latest: int, stop_after_step: int | None = None) -> int:
        if latest == 0 and self.archive.exists() and not any((self.archive / "full").glob("checkpoint-*")):
            moment = int(time.time())
            self.archive.rename(self.archive.with_name(f"{self.archive.name}.stale-{moment}"))
        self.clear_stale_stop_file()
        args = [TRAINER, "run", "--recipe", str(self.recipe_path)]
        if latest:
            args += ["--resume", str(self.checkpoint(latest))]
        if stop_after_step is not None:
            args += ["--stop-after-update", str(stop_after_step)]
        return self.launcher.run(f"train-from-{latest}", args, budget=None, on_stop=self.request_trainer_stop)

    def run(self) -> int:
        arm = str(_obj(self.recipe["arm"], "arm")["name"])
        if self.kind != "pilot":
            raise DriverError("night run takes the pilot recipe; use night smoke for the smoke recipe")
        final = self.gates[-1]
        previous_latest = None
        while True:
            if self.stop_requested():
                return self.finish(EXIT_DEFERRED, f"{arm}: stop requested; saved state resumes next night")
            try:
                measured = self.start_measurements()
            except Stopped:
                return self.finish(EXIT_DEFERRED, f"{arm}: stopped during the starting model measurements")
            except EvaluationFailed as error:
                return self.finish(EXIT_NEEDS_USER, f"{arm}: starting model measurement failed: {error}", needs_user=str(error))
            if measured == "defer":
                return self.finish(EXIT_DEFERRED, f"{arm}: starting model measurements deferred to the next night")
            steps = self.durable_steps()
            latest = steps[-1] if steps else 0
            if latest in self.gates:
                outcome = self.gate_step(latest)
                if outcome == "defer":
                    return self.finish(EXIT_DEFERRED, f"{arm}: evaluation {latest} deferred to the next night")
                if outcome == rules.ROLLBACK:
                    return self.finish_pilot("rolled_back", latest)
                if outcome == rules.STOP_FOR_USER:
                    return self.finish(EXIT_NEEDS_USER, f"{arm}: R6x stop-for-user at update {latest}",
                                       needs_user=(f"{arm} update {latest}: R6x says stop-for-user; see {self.decisions(latest)[-1]}. "
                                                   f"To decide, write {self.evidence_dir}/override-step-{latest}.json and retry the job."))
                if latest == final:
                    return self.finish_pilot("complete", latest)
                if self.plateaued(latest):
                    return self.finish_pilot("plateau", latest)
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

def smoke_receipt(arm: ExecGRPOArm, stops: Sequence[int], phases: Sequence[Json], results: Mapping[str, Path],
                  start_cpt: Path) -> tuple[Json, list[str]]:
    problems: list[str] = []
    updates = arm.updates()
    numbers = [int(cast(int, event["update"])) for event in updates]
    if numbers != list(range(1, stops[-1] + 1)):
        problems.append(f"telemetry updates {numbers} differ from 1..{stops[-1]}")
    for event in updates:
        if event.get("groups_trained"):
            missing = [name for name in UPDATE_FIELDS if not isinstance(event.get(name), (int, float))
                       or not math.isfinite(float(cast(float, event[name])))]
            if missing:
                problems.append(f"update {event['update']} lacks finite {missing}")
    resumed = [event.get("resumed_from") for event in updates if cast(int, event["update"]) > stops[0]]
    if any(value != stops[0] for value in resumed):
        problems.append(f"updates after {stops[0]} were not resumed from checkpoint {stops[0]}: {resumed}")
    if not any(event.get("optimizer_step") for event in updates):
        problems.append("no optimizer step ran (no update had a signal group)")
    if not sum(int(cast(int, e.get("exec_groups_drawn") or 0)) for e in updates if e.get("buffer_position") == 0):
        problems.append("no exec group was drawn")
    if not sum(int(cast(int, e.get("legacy_groups_drawn") or 0)) for e in updates if e.get("buffer_position") == 0):
        problems.append("no legacy group was drawn")
    for step in stops:
        try:
            state.verify_checkpoint(arm.checkpoint(step), arm.recipe_sha)
        except state.StateError as error:
            problems.append(f"checkpoint {step}: {error}")
    manifest = sha256(arm.checkpoint(stops[0]) / "campaign-manifest.json")
    for name, path in results.items():
        problems += arm.problems_for(name, path, stops[0], manifest)
    start_manifest = str(_obj(_obj(arm.recipe["parent"], "parent")["files"], "files")["campaign-manifest.json"])
    problems += arm.cpt_problems(start_cpt, 0, start_manifest)
    counts = counts_from(_obj(read_json(results["dev75"])["summary"], "summary"), "dev75")
    val = read_json(results["val"])
    cpt_now, cpt_start = float(cast(float, read_json(results["cpt"])["mean_causal_nll"])), float(cast(float, read_json(start_cpt)["mean_causal_nll"]))
    seconds = [float(cast(float, e["update_seconds"])) for e in updates]
    rollout = [float(cast(float, e["rollout_seconds"])) + float(cast(float, e["old_logprob_seconds"]))
               for e in updates if e.get("buffer_position") == 0]
    events = [json.loads(line) for line in (arm.archive / "telemetry.jsonl").read_text().splitlines()]
    saves = [float(e["seconds"]) for e in events if e.get("event") == "checkpoint_durable"]
    caps = [float(cast(float, _obj(e.get("allocator"), "allocator").get("allocator_cap_mib", 0))) for e in events
            if e.get("event") == "run_start" and e.get("allocator")]
    peaks = [int(cast(int, e["max_memory_reserved"])) for e in updates if isinstance(e.get("max_memory_reserved"), int)]
    per_update = ((sum(seconds) + sum(rollout)) / len(seconds)) if seconds else None
    val_scale = int(cast(int, _obj(arm.gate["val"], "val")["n_tasks"])) / max(1, int(cast(int, _obj(val["tasks"], "tasks")["n"])))
    gate_seconds = (float(cast(float, read_json(results["dev75"]).get("elapsed_seconds") or 0))
                    + val_scale * float(cast(float, val.get("elapsed_seconds") or 0))
                    + float(cast(float, read_json(results["cpt"]).get("load_seconds") or 0))
                    + float(cast(float, read_json(results["cpt"]).get("eval_seconds") or 0)) + 180.0)
    block = (rules.EVAL_EVERY * per_update + gate_seconds + (statistics.mean(saves) if saves else 300.0)) if per_update else None
    peak_mib = round(max(peaks) / 2**20, 1) if peaks else None
    receipt: Json = {
        "schema": SMOKE_SCHEMA, "recipe_sha256": arm.recipe_sha, "stops": list(stops), "phases": list(phases), "updates": numbers,
        "results": {name: {"path": str(path), "sha256": sha256(path)} for name, path in results.items()},
        "dev75": counts, "val_subset": _obj(val["metrics"], "metrics")["overall"],
        "cpt": {"checkpoint_4": cpt_now, "start": cpt_start, "rise": cpt_now - cpt_start, "start_result": str(start_cpt)},
        "update_seconds_median": statistics.median(seconds) if seconds else None,
        "rollout_seconds_per_buffer": rollout, "save_seconds": saves, "seconds_per_update_including_rollout": per_update,
        "gate_seconds_projected_full_val": round(gate_seconds, 1),
        "projected_updates_per_night": int(WINDOW_SECONDS / block * rules.EVAL_EVERY) if block else None,
        "peak_reserved_mib": peak_mib, "allocator_cap_mib": max(caps) if caps else None,
        "memory_headroom_mib": round(max(caps) - peak_mib, 1) if caps and peak_mib is not None else None,
        "replay_nll": [e.get("replay_nll") for e in updates], "grad_norm_grpo": [e.get("grad_norm_grpo") for e in updates],
        "grad_norm": [e.get("grad_norm") for e in updates],
        "zero_variance_share_before": [e.get("zero_variance_share_before") for e in updates],
        "exec_zero_variance_share_before": [e.get("exec_zero_variance_share_before") for e in updates],
        "clip_fraction": [e.get("clip_fraction") for e in updates],
        "problems": problems, "status": "pass" if not problems else "fail",
    }
    return receipt, problems


def smoke_result(arm: ExecGRPOArm, name: str, step: int) -> Path | None:
    found = sorted((arm.evidence_dir / "results").glob(f"{name}-step-{step}.*.json"))
    found = [p for p in found if not p.name.endswith((".admission.json",)) and ".records." not in p.name and ".rows." not in p.name]
    return found[-1] if found else None


def run_smoke(arm: ExecGRPOArm) -> int:
    smoke = _obj(arm.recipe.get("smoke"), "smoke")
    stops = [int(x) for x in cast(list[int], smoke["stops"])]
    eval_step = int(cast(int, smoke["eval_step"]))
    if eval_step != stops[0]:
        raise DriverError("the smoke evaluates the first save")
    phases: list[Json] = []
    results: dict[str, Path] = {}
    start_cpt: Path | None = None
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
            try:
                for name in cast(list[str], smoke["eval_panels"]):
                    existing = smoke_result(arm, name, eval_step)
                    results[name] = existing or arm.evaluate_panels(eval_step, arm.checkpoint(eval_step), parent_baseline=False,
                                                                    out_dir=arm.evidence_dir / "results", panels=(name,))[name]
                done = sorted(p for p in arm.start_dir.glob("cpt-start.*.json") if ".rows." not in p.name)
                if done:
                    start_cpt = done[-1]
                else:
                    arm.start_dir.mkdir(parents=True, exist_ok=True)
                    output = arm.start_dir / f"cpt-start.{stamp()}.json"
                    start_cpt = arm.run_child("cpt-start", [CPT_PROBE, "--bound-recipe", str(arm.recipe_path), "--start",
                                                            "--output", str(output)], output, arm.settings.eval_budget_seconds)
            except Stopped:
                return arm.finish(EXIT_DEFERRED, "smoke: stopped during the gate evaluation")
            except EvaluationFailed as error:
                return arm.finish(EXIT_FAILED, f"smoke: {error}", needs_user=f"C18 smoke evaluation failed: {error}")
    receipt, problems = smoke_receipt(arm, stops, phases, results, cast(Path, start_cpt))
    write_json(arm.archive / "smoke-receipt.json", receipt)
    arm.evaluations.append({"smoke_receipt": str(arm.archive / "smoke-receipt.json")})
    if problems:
        return arm.finish(EXIT_FAILED, f"smoke failed: {problems}", needs_user=f"C18 smoke failed: {problems[0]}")
    keys = ("update_seconds_median", "seconds_per_update_including_rollout", "projected_updates_per_night", "save_seconds",
            "peak_reserved_mib", "memory_headroom_mib", "cpt")
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
    arm = ExecGRPOArm(settings)
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


__all__ = ["TRAINER_DISK", "TRAINER_NO_PROGRESS", "ExecGRPOArm", "main"]
