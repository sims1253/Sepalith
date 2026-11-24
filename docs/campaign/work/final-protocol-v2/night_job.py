#!/usr/bin/env python3
"""Run the sealed-final v2 launcher, or its dev dress rehearsal, as one night-queue GPU job.

    night_job.py final     --root STATE/final              # the A5 scoring run (user-gated)
    night_job.py rehearsal --root STATE/rehearsals/<name>  # the dev dress rehearsal on the same path

ROOT holds `plan.json`, `freeze.json` and `freeze-approved.sha256`. The digest file is written apart from
the receipt: by the user for the final run, by `dress_rehearsal.py assemble` for a rehearsal.

Every check below runs before the launcher starts, so a refusal leaves no run-state marker:

* the job runs from the main checkout, under the night runner, with the pinned environment;
* the queue's running record equals the reviewed job file, and the plan (pinned by the freeze receipt)
  pins that job file's bytes under `night_job`;
* no `sepalith.training` process is running, and at least MIN_WINDOW_MINUTES remain before the deadline.

The launcher cannot resume from a crash without retained completions, so the job is one-shot and starts only
with ample margin (an estimated 1.5 h against a 16 h 40 min window). It runs under `cuda_host_guard`, which
holds the CUDA lock and stops the job if host memory runs low. The runner's preflight has already checked the
GPU and host memory. Only the status and the reason reach the night report; the result stays in the state
directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MAIN_CHECKOUT = Path("/home/m0hawk/sandbox/Sepalith")
STATE = Path.home() / ".local/state/sepalith/sealed-final-v2"
QUEUE_RUNNING = Path.home() / ".local/state/sepalith/night-queue/running"
CUDA_LOCK = Path.home() / ".local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock"
JOB_DIR = ROOT / "docs/training/plan-20260927/c17/jobs"
JOBS = {"final": "C17-sealed-final-v2-score", "rehearsal": "C17-sealed-final-v2-gpu-rehearsal"}
MIN_WINDOW_MINUTES = 360
REQUIRED_ENV = {
    "PYTHONPATH": str(ROOT / "packages/sepalith/src"),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
RUNNER_ENV = ("SEPALITH_DEADLINE_EPOCH", "SEPALITH_HARD_STOP_EPOCH", "SEPALITH_NIGHT_ATTEMPT_DIR")
JOB_FIELDS = ("id", "command", "cwd", "env", "resumable", "estimated_minutes", "priority", "requires")
Json = dict[str, Any]


class JobRefused(RuntimeError):
    """A pre-launch check failed; the launcher was not started."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(path: Path, digest: str) -> Json:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise JobRefused(f"{path.name} differs from its pinned digest")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise JobRefused(f"{path.name} is not a JSON object")
    return value


def training_processes(proc: Path = Path("/proc")) -> list[int]:
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        # Python processes started as `-m sepalith.training...`, the way every training job runs.
        if any(flag == b"-m" and module.startswith(b"sepalith.training") for flag, module in zip(argv, argv[1:], strict=False)):
            found.append(int(entry.name))
    return found


def check_environment(environ: Mapping[str, str], cwd: Path, now: float) -> float:
    """Main checkout, runner variables, pinned environment and time margin; return the hard-stop epoch."""
    if ROOT != MAIN_CHECKOUT or cwd != ROOT:
        raise JobRefused("the job must run from the main checkout")
    if missing := [name for name in RUNNER_ENV if not environ.get(name)]:
        raise JobRefused(f"not started by the night runner: {missing}")
    if wrong := [name for name, value in REQUIRED_ENV.items() if environ.get(name) != value]:
        raise JobRefused(f"environment differs from the job pin: {wrong}")
    if float(environ["SEPALITH_DEADLINE_EPOCH"]) - now < MIN_WINDOW_MINUTES * 60:
        raise JobRefused("too little of the window is left for a one-shot sealed run")
    return float(environ["SEPALITH_HARD_STOP_EPOCH"])


def check_job(mode: str, plan: Json, running: Path = QUEUE_RUNNING) -> None:
    """The queued record is the reviewed job file, and the frozen plan pins that file's bytes."""
    job_file = JOB_DIR / f"{JOBS[mode]}.json"
    reviewed = json.loads(job_file.read_bytes())
    record = json.loads((running / job_file.name).read_bytes())
    if any(record.get(key) != reviewed.get(key) for key in JOB_FIELDS):
        raise JobRefused("the queued job differs from the reviewed job file")
    pin = plan.get("night_job")
    if not isinstance(pin, dict) or pin.get("path") != str(job_file) or pin.get("sha256") != sha256_file(job_file):
        raise JobRefused("the plan does not pin this job file")


def load_frozen(root: Path) -> tuple[Json, str]:
    digest = (root / "freeze-approved.sha256").read_text().strip()
    freeze = pinned(root / "freeze.json", digest)
    return pinned(root / "plan.json", freeze["plan_sha256"]), digest


def launcher_command(mode: str, root: Path, digest: str) -> list[str]:
    if mode == "final":
        return [sys.executable, "-B", str(HERE / "protocol.py"), "--plan", str(root / "plan.json"),
                "--freeze-receipt", str(root / "freeze.json"), "--freeze-sha256", digest]
    return [sys.executable, "-B", str(HERE.parent / "final-panel-v1/dress_rehearsal.py"), "score",
            "--root", str(root), "--freeze-sha256", digest]


def guarded_run(command: list[str], attempt: Path, hard_stop: float) -> Json:
    """Run the launcher under cuda_host_guard; forward a stop request to the guard."""
    out = attempt / f"sealed-final-{time.time_ns()}"
    command_json = out.with_suffix(".json")
    command_json.write_text(json.dumps(command) + "\n")
    CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
    CUDA_LOCK.touch(exist_ok=True)
    seconds = max(1, int(hard_stop - time.time() - 60))
    argv = [sys.executable, "-B", "-m", "sepalith.training.guards.cuda_host_guard",
            "--command-json", str(command_json), "--output", str(out), "--seconds", str(seconds)]
    with subprocess.Popen(argv) as guard:
        # The run cannot save part-way: a stop request ends it, and the run stays stopped for the user.
        signal.signal(signal.SIGTERM, lambda *_: guard.terminate())
        guard.wait()
    terminal = out / "terminal.json"
    return json.loads(terminal.read_bytes()) if terminal.is_file() else {"status": "guard_failed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=tuple(JOBS))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    result_path = os.environ.get("SEPALITH_JOB_RESULT")
    expected = STATE / "final" if args.mode == "final" else STATE / "rehearsals"
    try:
        if root.resolve() != root or not (root == expected if args.mode == "final" else root.parent == expected):
            raise JobRefused("unexpected run root")
        hard_stop = check_environment(os.environ, Path.cwd(), time.time())
        plan, digest = load_frozen(root)
        check_job(args.mode, plan)
        if busy := training_processes():
            raise JobRefused(f"sepalith.training processes are running: {busy}")
    except (JobRefused, OSError, KeyError, ValueError) as error:
        summary = {"summary": f"sealed-final v2 {args.mode} not started: {error}",
                   "needs_user": [f"sealed-final v2 {args.mode} refused before launch: {error}"]}
        if result_path:
            Path(result_path).write_text(json.dumps(summary) + "\n")
        print(summary["summary"], file=sys.stderr)
        return 2

    terminal = guarded_run(launcher_command(args.mode, root, digest), Path(os.environ["SEPALITH_NIGHT_ATTEMPT_DIR"]),
                           hard_stop)
    result_file = Path(plan["state_dir"]) / "result.json"
    result = json.loads(result_file.read_bytes()) if result_file.is_file() else {}
    status, reason = result.get("status", "no_result"), result.get("reason")
    report: Json = {"summary": f"sealed-final v2 {args.mode}: {status}" + (f" ({reason})" if reason else ""),
                    "guard": {key: terminal.get(key) for key in ("status", "reason", "child_exit_code", "seconds")}}
    if args.mode == "rehearsal" and (root / "timing.json").is_file():
        report["timing"] = json.loads((root / "timing.json").read_bytes())
    report["needs_user"] = [f"sealed-final v2 {args.mode} {status}: review {result_file}"]
    if result_path:
        Path(result_path).write_text(json.dumps(report) + "\n")
    print(report["summary"])
    return 0 if status == "complete" and terminal.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
