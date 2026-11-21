"""Run C20 smoke, pilot and final evaluation as three dependent night jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from types import FrameType

from sepalith.exec_eval import jobs as exec_jobs
from sepalith.ops.night_queue import EXIT_DEFERRED
from sepalith.training.checkpoint.campaign_checkpoint import verify_checkpoint
from sepalith.training.grpo_exec.c19_night import guard
from sepalith.training.grpo_exec.pool import read_jsonl, require, sha256, write_jsonl_new
from sepalith.training.sft.c20 import data, recipe as config
from sepalith.training.sft.c20 import trainer as sft
from sepalith.training.sft.c20.evaluator import BATCH_SIZE, CAP, limit_cuda
from sepalith.training.sft.c20.night_controls import stop_payload
from sepalith.training.sft.night_controls import deadline_from_env

NEEDS_USER = 3
VERIFY_MINUTES = 10
PANEL_MINUTES = {"dev75": 20, "cpt": 20, "dev250": 45}
VAL_LIMIT_SMOKE = 24
MODULE = "sepalith.training.sft.c20.run"
TRAINER = "sepalith.training.sft.c20.trainer"
EVALUATOR = "sepalith.training.sft.c20.evaluator"


class QueueStop:
    """Translate the queue's SIGTERM and stop file into a safe boundary."""

    signalled = False

    def on_signal(self, _number: int, _frame: FrameType | None) -> None:
        self.signalled = True

    def requested(self) -> bool:
        return self.signalled or exec_jobs.stop_requested()


queue_stop = QueueStop()


def checkpoint(recipe: dict, step: int) -> Path:
    return config.PARENT if step == 0 else Path(recipe["outputs"]["archive"]) / "full" / f"checkpoint-{step}"


def evidence(recipe: dict, step: int) -> Path:
    return Path(recipe["outputs"]["evidence"]) / f"step-{step}"


def binding(recipe: dict, recipe_path: Path, step: int) -> dict:
    return {"step": step, "recipe_sha256": sha256(recipe_path),
            "checkpoint_manifest_sha256": sha256(checkpoint(recipe, step) / "campaign-manifest.json")}


def continuation(recipe: dict, recipe_path: Path, resume: Path | None) -> dict:
    step = int(resume.name.removeprefix("checkpoint-")) if resume is not None else 0
    stages = config.stages(recipe["mode"])
    prior = [s for s in stages if s <= step]
    for gate_step in prior:
        record = data.read(evidence(recipe, gate_step) / "gate.json")
        require(record["binding"] == binding(recipe, recipe_path, gate_step), "prior gate binding differs")
        for pin in record["inputs"].values():
            data.checked(pin)
    upcoming = [s for s in stages if s > step]
    require(upcoming, "trainer has passed the last mandatory gate")
    return {"initial_step": step, "next_mandatory_stop": upcoming[0], "prior_gates_verified": prior}


def promote(start: list[dict], candidate: list[dict], nofp_rise: int, cpt_rise: float) -> dict:
    """Apply the in-run criteria to complete, identically indexed validation records."""
    def identity(row: dict) -> tuple:
        return row["task_id"], row["function_id"], row["family"], row["target_lines"]

    expected = {identity(r) for r in start}
    require(len(expected) == len(start) == len(candidate) and expected == {identity(r) for r in candidate},
            "validation pairs differ")
    groups = []
    for rows in (start, candidate):
        medium_long = [r for r in rows if r["family"] == "finish_block" and 3 <= r["target_lines"] <= 16]
        rename = [r for r in rows if r["family"] == "rename_propagation"]
        require(medium_long and rename, "validation subgroups must be populated")
        groups.append((medium_long, rename))
    (old, old_rename), (new, new_rename) = groups
    n_target = len(new)
    target_passes = sum(r["all_pass"] for r in new)
    target_gain = target_passes - sum(r["all_pass"] for r in old)
    rename_drop = sum(r["all_pass"] for r in old_rename) - sum(r["all_pass"] for r in new_rename)
    diff = exec_jobs.paired_difference(old, new, "all_pass", "c20:finish-block-3-16", reps=2000)
    return {"accepted": (target_gain * 100 >= 2 * n_target and diff["ci95"][0] > 0
                         and rename_drop * 100 <= len(new_rename) and nofp_rise <= 2
                         and math.isfinite(cpt_rise) and cpt_rise <= 0.03),
            "finish_difference": diff, "finish_all_pass": target_passes / n_target,
            "rename_drop": rename_drop / len(new_rename), "dev75_nofp_rise": nofp_rise, "cpt_nll_rise": cpt_rise}


def remaining(minutes: int) -> bool:
    deadline = deadline_from_env()
    assert deadline is not None
    return not queue_stop.requested() and deadline - time.time() >= 60 * minutes


def guarded_launch(module: str, key: str, *args: str, save_and_stop: Callable[[], None] | None = None) -> int:
    attempt = Path(os.environ["SEPALITH_NIGHT_ATTEMPT_DIR"])
    out = attempt / f"{key}-{time.time_ns()}"
    command = out.with_suffix(".json")
    exec_jobs.CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
    exec_jobs.CUDA_LOCK.touch(exist_ok=True)
    exec_jobs.write_json(command, [sys.executable, "-B", "-m", module, *args])
    seconds = max(1, int(float(os.environ["SEPALITH_HARD_STOP_EPOCH"]) - time.time() - 30))
    argv = [sys.executable, "-B", "-m", "sepalith.training.guards.cuda_host_guard",
            "--command-json", str(command), "--output", str(out), "--seconds", str(seconds)]
    env = dict(os.environ)
    env.pop("SEPALITH_JOB_RESULT", None)
    stopping = False
    with subprocess.Popen(argv, env=env) as process:
        while True:
            try:
                process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if not stopping and queue_stop.requested():
                    stopping = True
                    if save_and_stop is not None:
                        save_and_stop()
                    else:
                        process.terminate()
    if (save_and_stop is None and stopping and process.returncode == -signal.SIGTERM
            and not (out / "terminal.json").exists()):
        return EXIT_DEFERRED  # stopped during guard preflight, before it could write a receipt
    terminal = data.read(out / "terminal.json")
    code = terminal.get("child_exit_code")
    if save_and_stop is None and (
        (stopping and terminal["reason"] == "guard_terminated")
        or (code == EXIT_DEFERRED and terminal["reason"] is None)
    ):
        return EXIT_DEFERRED
    require(terminal["status"] == "completed" and code == 0,
            f"{key} stopped in host guard: {terminal}; see {out}")
    return 0


def evaluate_worker(recipe_path: Path, step: int, panel_name: str) -> None:
    import torch

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer, embedding_identity, restore_trainer_eog_alignment,
    )
    from sepalith.training.eval.campaign_cpt_eval import package_holdout_evaluator
    from sepalith.training.eval.campaign_eval import development_evaluator, load_development_model
    from sepalith.training.grpo.night import counts_from
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recipe = sft.load_bound(recipe_path)
    cp = checkpoint(recipe, step)
    out = evidence(recipe, step)
    out.mkdir(parents=True, exist_ok=True)
    limit_cuda(config.MEMORY_MIB)
    if panel_name == "cpt":
        model = AutoModelForCausalLM.from_pretrained(
            str(cp), dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True, trust_remote_code=False)
        spec = recipe["dev_gate"]["cpt"]
        model.config.use_cache = False
        evaluate = package_holdout_evaluator({"validation_rows": spec["panel"], "validation_batch_size": 1,
                                               "parameters": {"max_sequence_tokens": spec["max_sequence_tokens"]},
                                               "materialized_rows": True})
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            summary = evaluate(model, None, cp, step)
        nll = summary["metrics"]["mean_causal_nll"]
        require(summary["denominators"]["validation_rows"] == spec["rows"] and math.isfinite(nll),
                "CPT 2K denominator or loss differs")
        metrics = {"nll": nll}
    else:
        panel = recipe["dev_gate"]["panels"][panel_name]
        cases_dir = out / f"{panel_name}-cases"
        cases_dir.mkdir(exist_ok=True)
        model = load_development_model(cp, dtype="bf16", device="cuda:0")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(cp), local_files_only=True, trust_remote_code=False)
        reference = load_pinned_reference_tokenizer(config.PARENT)
        restore_trainer_eog_alignment(model, tokenizer, reference)
        assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "c20_dev_generation")
        evaluate = development_evaluator({"development_panel": panel, "renderer_id": "zeta2-prm03-v1",
            "development_case_ids": [r["id"] for r in read_jsonl(Path(panel["path"]))],
            "development_max_new_tokens": CAP, "parameters": {"max_sequence_tokens": 16384},
            "evaluation_output_directory": str(cases_dir), "generation_batch_size": BATCH_SIZE,
            "generation_kv_cache": "dynamic"})
        summary = evaluate(model, tokenizer, cp, step)
        require(summary["denominators"]["cases"] == panel["cases"] and summary["panel_sha256"] == panel["sha256"],
                "DEV denominator or panel differs")
        metrics = counts_from(summary, panel_name)
    exec_jobs.write_json(out / f"{panel_name}.json", {
        "binding": binding(recipe, recipe_path, step), "metrics": metrics, "summary": summary,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    })


def panel(recipe: dict, recipe_path: Path, step: int, name: str) -> dict | None:
    result = evidence(recipe, step) / f"{name}.json"
    if not result.exists():
        if not remaining(PANEL_MINUTES[name]):
            return None
        code = guarded_launch(MODULE, f"{name}-{step}", "_evaluate", "--recipe", str(recipe_path),
                              "--step", str(step), "--panel", name)
        if code == EXIT_DEFERRED:
            return None
        require(code == 0, f"{name} evaluator exited {code}")
    value = data.read(result)
    require(value["binding"] == binding(recipe, recipe_path, step), f"{name} result binding differs")
    return value["metrics"]


def exec_run(kind: str, out: Path, tasks: dict, models: list[dict], job_id: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    models_path = out / "models.json"
    model_record = {"models": models}
    if models_path.exists():
        require(data.read(models_path) == model_record, "stale exec model inputs")
    else:
        exec_jobs.write_json(models_path, model_record)
    return guarded_launch(EVALUATOR, job_id, "--kind", kind, "--job-id", job_id,
                  "--work-dir", str(out), "--tasks", tasks["path"], "--tasks-sha256", tasks["sha256"],
                  "--models", str(models_path), "--model-budget-minutes", str(VERIFY_MINUTES))


def validation(recipe: dict, recipe_path: Path, step: int) -> tuple[Path, dict] | None:
    out = evidence(recipe, step) / "validation"
    cp = checkpoint(recipe, step)
    label = config.START if step == 0 else config.SELECTED
    tasks = recipe["validation"]
    count = data.read(Path(recipe["validation_manifest"]["path"]))["stats"]["tasks"]
    if recipe["mode"] == "smoke":
        path = evidence(recipe, step) / "smoke-tasks.jsonl"
        if not path.exists():
            rows = read_jsonl(Path(tasks["path"]))
            write_jsonl_new(path, rows[:VAL_LIMIT_SMOKE])
        tasks = {"path": str(path), "sha256": sha256(path)}
        count = VAL_LIMIT_SMOKE
    identity = binding(recipe, recipe_path, step)
    receipt_path = out / "receipt.json"
    pin = out / "binding.json"
    if pin.exists():
        require(data.read(pin) == identity, "stale validation checkpoint")
    else:
        exec_jobs.write_json(pin, identity)
    if not receipt_path.exists() or data.read(receipt_path)["status"] != "complete":
        if not remaining(VERIFY_MINUTES):
            return None
        code = exec_run("validation", out, tasks, [{"label": label, "name": label, "path": str(cp)}],
                        f'{recipe["arm"]["name"]}-val-{step}')
        if code == EXIT_DEFERRED:
            return None
        require(code == 0, f"exec validation exited {code}")
    receipt = data.read(receipt_path)
    require(receipt["status"] == "complete" and receipt["tasks"]["sha256"] == tasks["sha256"],
            "incomplete validation receipt or wrong panel")
    metrics = data.read(out / "metrics.json")[label]
    require(metrics["coverage"] == {"tasks_scored": count, "tasks_total": count}, "validation coverage differs")
    return out / f"records-{label}.jsonl", metrics


def record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def gate(recipe: dict, recipe_path: Path, step: int) -> dict | None:
    directory = evidence(recipe, step)
    report_path = directory / "gate.json"
    expected = binding(recipe, recipe_path, step)
    if report_path.exists():
        report = data.read(report_path)
        require(report["binding"] == expected, "stale gate result")
        for pin in report["inputs"].values():
            data.checked(pin)
        return report
    val = validation(recipe, recipe_path, step)
    if val is None:
        return None
    dev75 = panel(recipe, recipe_path, step, "dev75")
    cpt = panel(recipe, recipe_path, step, "cpt")
    if dev75 is None or cpt is None:
        return None
    rows, metrics = val
    report = {"binding": expected, "inputs": {"validation": record(rows),
              "validation_metrics": record(directory / "validation/metrics.json"),
              "validation_receipt": record(directory / "validation/receipt.json"),
              "dev75": record(directory / "dev75.json"), "cpt": record(directory / "cpt.json")},
              "dev75": dev75, "cpt": cpt, "validation": metrics}
    if step and recipe["mode"] == "pilot":
        baseline = gate(recipe, recipe_path, 0)
        require(baseline is not None, "missing baseline gate")
        assert baseline is not None
        report["criteria"] = promote(read_jsonl(data.checked(baseline["inputs"]["validation"])), read_jsonl(rows),
                                      dev75["nofp"] - baseline["dev75"]["nofp"], cpt["nll"] - baseline["cpt"]["nll"])
    exec_jobs.write_json(report_path, report)
    return report


def emit(value: dict, code: int = 0) -> int:
    print(json.dumps(value, sort_keys=True), flush=True)
    if path := os.environ.get("SEPALITH_JOB_RESULT"):
        exec_jobs.write_json(Path(path), {"summary": json.dumps(value, sort_keys=True),
                                         "needs_user": value.get("needs_user")})
    return code


def verify_saved_checkpoint(recipe: dict, step: int) -> None:
    manifest = verify_checkpoint(checkpoint(recipe, step), sft.identity(recipe), require_full=True,
                                 expected_checkpoint_kind="full_weights")
    require(manifest["step"] == step, "durable checkpoint step differs")


def train_segment(recipe: dict, recipe_path: Path, current: int, stop: int | None = None) -> int:
    config.require_disk(config.CHECKPOINT_BYTES)
    stop_path = Path(recipe["outputs"]["graceful_stop"])
    payload = stop_payload(sha256(recipe_path))
    if stop_path.exists():
        require(data.read(stop_path) == payload, "graceful stop request identity differs")
        stop_path.rename(stop_path.with_name(f"{stop_path.name}.consumed-{time.time_ns()}"))

    def request_save() -> None:
        if stop_path.exists():
            require(data.read(stop_path) == payload, "graceful stop request identity differs")
        else:
            sft.write_json(stop_path, payload)

    args = ["--recipe", str(recipe_path)]
    if current:
        args += ["--resume", str(checkpoint(recipe, current))]
    if stop:
        args += ["--stop-after-step", str(stop)]
    code = guarded_launch(TRAINER, f"train-from-{current}", *args, save_and_stop=request_save)
    require(code == 0, f"SFT trainer exited {code}")
    report = data.read(Path(recipe["outputs"]["archive"]) / "run-result.json")
    step = report["global_step"]
    require(report["initial_cursor"] == current * 16 and step > current
            and report["observed_draws"] == (step - current) * 16
            and report["terminal_checkpoint"] == str(checkpoint(recipe, step))
            and report["peak_cuda_reserved_bytes"] <= config.MEMORY_MIB * 1024**2,
            "trainer resume cursor, checkpoint or GPU peak differs")
    verify_saved_checkpoint(recipe, step)
    report["binding"] = binding(recipe, recipe_path, step)
    exec_jobs.write_json(evidence(recipe, step) / "train-segment.json", report)
    return step


def smoke_receipt(recipe: dict, recipe_path: Path) -> dict:
    segments = {}
    peak = 0
    for step in (1, 2):
        verify_saved_checkpoint(recipe, step)
        path = evidence(recipe, step) / "train-segment.json"
        segment = data.read(path)
        require(segment["binding"] == binding(recipe, recipe_path, step)
                and segment["initial_cursor"] == (step - 1) * 16 and segment["observed_draws"] == 16
                and segment["terminal_checkpoint"] == str(checkpoint(recipe, step)),
                "smoke did not resume for a full update")
        peak = max(peak, segment["peak_cuda_reserved_bytes"])
        segments[str(step)] = record(path)
    require(0 < peak <= config.MEMORY_MIB * 1024**2, "smoke GPU peak differs")
    directory = evidence(recipe, 2)
    path = directory / "gate.json"
    report = data.read(path)
    expected = binding(recipe, recipe_path, 2)
    require(report["binding"] == expected, "stale smoke gate")
    expected_inputs = {
        "validation": directory / f"validation/records-{config.SELECTED}.jsonl",
        "validation_receipt": directory / "validation/receipt.json",
        "validation_metrics": directory / "validation/metrics.json",
        "dev75": directory / "dev75.json",
        "cpt": directory / "cpt.json",
    }
    require(report["inputs"].keys() == expected_inputs.keys(), "smoke gate inputs differ")
    for key, path_expected in expected_inputs.items():
        require(data.checked(report["inputs"][key]) == path_expected, f"smoke {key} path differs")
    for key in ("dev75", "cpt"):
        result = data.read(expected_inputs[key])
        require(result["binding"] == expected and result["metrics"] == report[key],
                f"smoke {key} evidence differs")
    validation_receipt = data.read(expected_inputs["validation_receipt"])
    tasks = directory / "smoke-tasks.jsonl"
    label = config.SELECTED
    require(validation_receipt["status"] == "complete"
            and validation_receipt["tasks"]["sha256"] == sha256(tasks)
            and validation_receipt["models"][label]["path"] == str(checkpoint(recipe, 2))
            and data.read(expected_inputs["validation_metrics"])[label] == report["validation"]
            and report["validation"]["coverage"] == {"tasks_scored": VAL_LIMIT_SMOKE, "tasks_total": VAL_LIMIT_SMOKE},
            "smoke validation evidence differs")
    return {"status": "pass", "binding": expected, "segments": segments, "gate": record(path),
            "peak_cuda_reserved_bytes": peak}


def verify_smoke(pilot: dict) -> None:
    expected = pilot["smoke_recipe"]
    recipe_path = data.checked(expected)
    path = config.STATE / config.JOB_IDS["smoke"] / "smoke-pass.json"
    require(path.is_file(), "C20 smoke has not passed")
    receipt = data.read(path)
    require(receipt.get("status") == "pass"
            and receipt.get("binding", {}).get("recipe_sha256") == expected["sha256"],
            "C20 smoke receipt belongs to a different recipe")
    smoke = data.read(recipe_path)
    require(smoke["mode"] == "smoke" and smoke["source"] == pilot["source"], "C20 smoke recipe differs")
    require(receipt == smoke_receipt(smoke, recipe_path), "C20 smoke receipt evidence differs")


def close_pilot(recipe: dict, recipe_path: Path, reports: list[dict]) -> int:
    accepted = [r for r in reports if r["criteria"]["accepted"]]
    summary = {"recipe": record(recipe_path), "result_checkpoint": str(config.PARENT),
               "status": "keep_c19", "gates": reports}
    if accepted:
        winner = max(accepted, key=lambda r: (r["criteria"]["finish_all_pass"], -r["binding"]["step"]))
        step = winner["binding"]["step"]
        dev250 = panel(recipe, recipe_path, step, "dev250")
        if dev250 is None:
            return EXIT_DEFERRED
        summary["dev250"] = dev250
        summary["dev250_result"] = record(evidence(recipe, step) / "dev250.json")
        if dev250["score250"] >= 108:
            summary.update(status="pass", selected=winner, result_checkpoint=str(checkpoint(recipe, step)))
    path = Path(recipe["outputs"]["archive"]) / "pilot-summary.json"
    exec_jobs.write_json(path, summary)
    return emit({"status": summary["status"], "summary": str(path),
                 "checkpoint": summary["result_checkpoint"]})


def train_job(recipe_path: Path) -> int:
    recipe = sft.load_bound(recipe_path)
    mode = recipe["mode"]
    archive = Path(recipe["outputs"]["archive"])
    steps = config.stages(mode)
    smoke = mode == "smoke"
    if not smoke:
        verify_smoke(recipe)
        if gate(recipe, recipe_path, 0) is None:
            return EXIT_DEFERRED
    current = max((int(p.name.removeprefix("checkpoint-")) for p in (archive / "full").glob("checkpoint-*")), default=0)
    reports = []
    for stage in steps:
        if queue_stop.requested():
            if current:
                verify_saved_checkpoint(recipe, current)
            return EXIT_DEFERRED
        while current < stage:
            if not remaining(VERIFY_MINUTES):
                return EXIT_DEFERRED
            current = train_segment(recipe, recipe_path, current, 1 if smoke and not current else None)
            if queue_stop.requested() or (not smoke and current < stage):
                return EXIT_DEFERRED
        verify_saved_checkpoint(recipe, stage)
        report = gate(recipe, recipe_path, stage)
        if report is None:
            return EXIT_DEFERRED
        reports.append(report)
    if smoke:
        path = config.STATE / config.JOB_IDS["smoke"] / "smoke-pass.json"
        exec_jobs.write_json(path, smoke_receipt(recipe, recipe_path))
        return emit({"status": "pass", "summary": str(path)})
    return close_pilot(recipe, recipe_path, reports)


def final_job(recipe_path: Path) -> int:
    archive = config.CAMPAIGN / "checkpoints" / config.JOB_IDS["pilot"]
    summary = data.read(archive / "pilot-summary.json")
    if summary["status"] == "keep_c19":
        return emit({"status": "skipped", "checkpoint": str(config.PARENT), "reason": "pilot failed"})
    require(summary["status"] == "pass", "pilot did not pass")
    pilot_path = data.checked(summary["recipe"])
    pilot = sft.load_bound(pilot_path)
    winner = summary["selected"]
    step = winner["binding"]["step"]
    cp = checkpoint(pilot, step)
    require(step in config.PILOT_STEPS and winner["binding"] == binding(pilot, pilot_path, step)
            and winner["criteria"]["accepted"], "pilot selection differs")
    dev250 = data.read(data.checked(summary["dev250_result"]))
    for gate_step in (0, step):
        require((evidence(pilot, gate_step) / "gate.json").is_file(), "pilot gate receipt missing")
        gate_report = gate(pilot, pilot_path, gate_step)
        require(gate_report is not None and (gate_step == 0 or gate_report == winner),
                "pilot gate evidence differs")
    cpt = winner["criteria"]["cpt_nll_rise"]
    require(dev250["binding"] == winner["binding"] and dev250["metrics"]["score250"] >= 108 and cpt <= 0.03,
            "pilot non-inferiority evidence differs")
    verify_checkpoint(cp, sft.identity(pilot), require_full=True, expected_checkpoint_kind="full_weights")
    if not remaining(VERIFY_MINUTES):
        return EXIT_DEFERRED
    spec = data.read(recipe_path)
    out = config.STATE / config.JOB_IDS["final"] / f"final-step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    pin = out / "binding.json"
    if pin.exists():
        require(data.read(pin) == winner["binding"], "stale final checkpoint")
    else:
        exec_jobs.write_json(pin, winner["binding"])
    baseline = data.checked(spec["completions"])
    seeded = out / f"completions-{config.START}.jsonl"
    if not seeded.exists():
        temporary = seeded.with_suffix(".tmp")
        shutil.copyfile(baseline, temporary)
        temporary.replace(seeded)
    require(sha256(seeded) == spec["completions"]["sha256"], "seeded completions differ")
    models = [{"label": label, "name": label, "path": str(path)}
              for label, path in ((config.START, config.PARENT), (config.SELECTED, cp))]
    receipt_path = out / "receipt.json"
    if not receipt_path.exists():
        exec_jobs.write_json(receipt_path, {"schema": exec_jobs.RECEIPT_SCHEMA, "job": config.JOB_IDS["final"],
            "kind": "holdout", "started": exec_jobs.now(), "attempts": [], "status": "pending",
            "models": {config.START: {**models[0], "status": "generated", "pinned_completions": spec["completions"]}}})
    receipt = data.read(receipt_path)
    require(receipt["models"][config.START]["status"] == "generated", "pinned baseline cannot be regenerated")
    if receipt["status"] != "complete":
        code = exec_run("holdout", out, spec["tasks"], models, config.JOB_IDS["final"])
        if code:
            return code
    metrics = data.read(out / "metrics.json")
    start, selected = metrics[config.START], metrics[config.SELECTED]
    diff = selected[f"vs_{config.START}"]["all_pass"]
    n = spec["n_tasks"]
    require(all(m["coverage"] == {"tasks_scored": n, "tasks_total": n} for m in (start, selected))
            and diff["n"] == n, "hold-out pairs incomplete")
    adopt = diff["ci95"][0] > 0
    result = {"adopt": adopt, "result_checkpoint": str(cp if adopt else config.PARENT),
              "all_pass_difference": diff, "dev250": dev250["metrics"]["score250"], "cpt_nll_rise": cpt,
              "finish_block": {label: {bucket: metrics[label]["by_bucket"][bucket] for bucket in ("medium", "long")}
                               for label in (config.START, config.SELECTED)}}
    exec_jobs.write_json(out / "verdict.json", result)
    return emit(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*config.JOB_IDS, "_evaluate"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--step", type=int)
    parser.add_argument("--panel", choices=("dev75", "dev250", "cpt"))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, queue_stop.on_signal)
    try:
        require(deadline_from_env() is not None, "night queue deadline required")
        guard()
        if args.mode == "_evaluate":
            require(args.step is not None and args.panel is not None, "evaluation step and panel required")
            evaluate_worker(args.recipe, args.step, args.panel)
            return 0
        require(os.environ.get("SEPALITH_NIGHT_JOB_ID") == config.JOB_IDS[args.mode], "night queue job identity differs")
        require(args.sha256 == sha256(args.recipe), "job recipe hash differs")
        if args.mode == "final":
            config.verify_source(data.read(args.recipe)["source"])
            return final_job(args.recipe)
        require(data.read(args.recipe)["mode"] == args.mode, "training recipe mode differs")
        code = train_job(args.recipe)
        if code == EXIT_DEFERRED and args.mode == "smoke":
            return emit({"needs_user": "C20 smoke did not finish before the deadline."}, NEEDS_USER)
        return code
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        return emit({"needs_user": str(error)}, NEEDS_USER)


if __name__ == "__main__":
    raise SystemExit(main())
