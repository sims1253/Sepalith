"""Run the C20 step-300 final evaluation authorized by the DEV250 exception."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess

from sepalith.training.sft.c20 import run

JOB_ID = "C20-final-holdout-exception"
STEP = 300
RECEIPT_SCHEMA = "sepalith.c20.dev250-exception.v1"
SOURCE_SCHEMA = "sepalith.c20.exception-source.v1"
WAIVED_CRITERION = "dev250 score250 >= 108"
CPT_MAX_RISE = 0.03
ADOPT_CRITERIA = ["holdout all-pass difference vs C19-100 CI95 lower bound > 0", "CPT NLL rise <= 0.03"]
PACKET = run.config.PACKET
SOURCE_FILES = (
    "packages/sepalith/src/sepalith/training/c20_exception/__init__.py",
    "packages/sepalith/src/sepalith/training/c20_exception/final.py",
    str(PACKET / "dev250-exception.json"),
    str(PACKET / "final.json"),
)


def verify_source(repo: Path, manifest_path: Path, digest: str) -> None:
    path = run.data.checked({"path": str(manifest_path), "sha256": digest})
    manifest = run.data.read(path)
    files = manifest["files"]
    run.require(manifest["schema"] == SOURCE_SCHEMA and files.keys() == set(SOURCE_FILES),
                "exception source manifest differs")
    for name, sha256 in files.items():
        run.data.checked({"path": str(repo / name), "sha256": sha256})


def verify_pilot(receipt: dict, summary_path: Path, dev250_path: Path) -> tuple[Path, dict]:
    run.require(receipt["schema"] == RECEIPT_SCHEMA and receipt["waived_criterion"] == WAIVED_CRITERION
                and receipt["adopt_criteria"] == ADOPT_CRITERIA, "exception scope differs")
    summary = run.data.read(run.data.checked({
        "path": str(summary_path), "sha256": receipt["pilot_summary_sha256"],
    }))
    dev250_pin = {"path": str(dev250_path), "sha256": receipt["dev250_sha256"]}
    dev250 = run.data.read(run.data.checked(dev250_pin))
    run.require(summary["status"] == "keep_c19" and summary["result_checkpoint"] == str(run.config.PARENT)
                and summary["dev250_result"] == dev250_pin, "pilot did not stop at the bound DEV250 result")
    run.require(dev250["metrics"] == summary["dev250"] == receipt["observed_dev250"],
                "pilot DEV250 metrics differ")

    # close_pilot omits "selected" when DEV250 misses. Replay its selection from the retained gates.
    reports = summary["gates"]
    run.require([report["binding"]["step"] for report in reports] == run.config.PILOT_STEPS,
                "pilot gate steps differ")
    accepted = [report for report in reports if report["criteria"]["accepted"] is True]
    run.require(accepted, "no passing pilot gate")
    winner = max(accepted, key=lambda report: (report["criteria"]["finish_all_pass"], -report["binding"]["step"]))
    identity = winner["binding"]
    run.require(identity["step"] == STEP and identity == receipt["binding"] == dev250["binding"],
                "exception step-300 binding differs")

    pilot_path = run.data.checked(summary["recipe"])
    pilot = run.sft.load_bound(pilot_path)
    run.require(pilot["mode"] == "pilot" and run.binding(pilot, pilot_path, STEP) == identity,
                "pilot checkpoint binding differs")

    for step in (0, STEP):
        run.require((run.evidence(pilot, step) / "gate.json").is_file(), "pilot gate receipt missing")
    baseline = run.gate(pilot, pilot_path, 0)
    run.require(baseline is not None and run.gate(pilot, pilot_path, STEP) == winner,
                "pilot gate evidence differs")
    assert baseline is not None
    # Verify every promotion criterion against the hash-checked validation and gate evidence.
    cpt_rise = winner["cpt"]["nll"] - baseline["cpt"]["nll"]
    criteria = run.promote(
        run.read_jsonl(run.data.checked(baseline["inputs"]["validation"])),
        run.read_jsonl(run.data.checked(winner["inputs"]["validation"])),
        winner["dev75"]["nofp"] - baseline["dev75"]["nofp"], cpt_rise,
    )
    run.require(criteria == winner["criteria"] and criteria["accepted"], "pilot promotion criteria differ")
    cp = run.checkpoint(pilot, STEP)
    run.verify_saved_checkpoint(pilot, STEP)
    return cp, winner


def evaluate(out: Path, spec: dict, cp: Path, winner: dict, exception: dict) -> int:
    out.mkdir(parents=True, exist_ok=True)
    pin = out / "binding.json"
    identity = winner["binding"]
    if pin.exists():
        run.require(run.data.read(pin) == identity, "stale final checkpoint")
    else:
        run.exec_jobs.write_json(pin, identity)

    baseline = run.data.checked(spec["completions"])
    seeded = out / f"completions-{run.config.START}.jsonl"
    if not seeded.exists():
        temporary = seeded.with_suffix(".tmp")
        shutil.copyfile(baseline, temporary)
        temporary.replace(seeded)
    run.require(run.sha256(seeded) == spec["completions"]["sha256"], "seeded completions differ")

    models = [{"label": label, "name": label, "path": str(path)}
              for label, path in ((run.config.START, run.config.PARENT), (run.config.SELECTED, cp))]
    receipt_path = out / "receipt.json"
    if not receipt_path.exists():
        run.exec_jobs.write_json(receipt_path, {
            "schema": run.exec_jobs.RECEIPT_SCHEMA, "job": JOB_ID, "kind": "holdout",
            "started": run.exec_jobs.now(), "attempts": [], "status": "pending",
            "models": {run.config.START: {
                **models[0], "status": "generated", "pinned_completions": spec["completions"],
            }},
        })
    receipt = run.data.read(receipt_path)
    run.require(receipt["models"][run.config.START]["status"] == "generated",
                "pinned baseline cannot be regenerated")
    if receipt["status"] != "complete":
        code = run.exec_run("holdout", out, spec["tasks"], models, JOB_ID)
        if code:
            return code

    metrics = run.data.read(out / "metrics.json")
    start, selected = metrics[run.config.START], metrics[run.config.SELECTED]
    diff = selected[f"vs_{run.config.START}"]["all_pass"]
    n = spec["n_tasks"]
    run.require(all(m["coverage"] == {"tasks_scored": n, "tasks_total": n} for m in (start, selected))
                and diff["n"] == n, "hold-out pairs incomplete")
    cpt_rise = winner["criteria"]["cpt_nll_rise"]
    lower = diff["ci95"][0]
    adopt = math.isfinite(lower) and lower > 0 and math.isfinite(cpt_rise) and cpt_rise <= CPT_MAX_RISE
    result = {
        "adopt": adopt, "result_checkpoint": str(cp if adopt else run.config.PARENT),
        "exception": exception, "all_pass_difference": diff,
        "dev250": exception["observed_dev250"]["score250"], "cpt_nll_rise": cpt_rise,
        "finish_block": {label: {bucket: metrics[label]["by_bucket"][bucket] for bucket in ("medium", "long")}
                         for label in (run.config.START, run.config.SELECTED)},
    }
    run.exec_jobs.write_json(out / "verdict.json", result)
    return run.emit(result)


def final_job(repo: Path) -> int:
    spec = run.data.read(repo / PACKET / "final.json")
    run.config.verify_source(spec["source"])
    run.require(spec["completions"] == run.config.COMPLETIONS, "baseline completions pin differs")
    exception = run.data.read(repo / PACKET / "dev250-exception.json")
    pilot_id = run.config.JOB_IDS["pilot"]
    summary = run.config.CAMPAIGN / "checkpoints" / pilot_id / "pilot-summary.json"
    dev250 = run.config.STATE / pilot_id / f"step-{STEP}" / "dev250.json"
    cp, winner = verify_pilot(exception, summary, dev250)
    if not run.remaining(run.VERIFY_MINUTES):
        return run.EXIT_DEFERRED
    out = run.config.STATE / JOB_ID / f"final-step-{STEP}-exception"
    return evaluate(out, spec, cp, winner, exception)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, run.queue_stop.on_signal)
    try:
        run.require(run.deadline_from_env() is not None, "night queue deadline required")
        run.guard()
        run.require(os.environ.get("SEPALITH_NIGHT_JOB_ID") == JOB_ID, "night queue job identity differs")
        verify_source(run.config.MAIN, args.source_manifest, args.sha256)
        return final_job(run.config.MAIN)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        return run.emit({"needs_user": str(error)}, run.NEEDS_USER)


if __name__ == "__main__":
    raise SystemExit(main())
