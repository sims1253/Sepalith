"""Run queued GPU jobs inside the nightly window (01:00 to 09:00 Europe/Berlin).

The runner starts jobs one at a time and only while the window is open and the
machine is idle: no GPU compute processes, little GPU memory in use, enough
host memory and at least 150 GB free on the checkpoint disk. Every job gets:

- ``SEPALITH_DEADLINE_EPOCH`` (08:40): save and stop at the next safe boundary
  once this passes, then exit 75 if work remains or 0 if the job is finished.
- a graceful-stop request at 08:50: ``SEPALITH_STOP_REQUEST_FILE`` is written
  and SIGTERM goes to the job's leader process only.
- a hard stop at 09:00: every process the job started is killed with SIGKILL,
  including processes that moved to their own session.

Jobs may write JSON to ``SEPALITH_JOB_RESULT`` with ``checkpoints``,
``evaluations``, ``rule_decisions``, ``summary`` and ``needs_user``. The
runner copies it into the morning report under
``~/.local/state/sepalith/night-reports/<YYYY-MM-DD>.{json,md}``.

Usage::

    python -m sepalith.ops.night_runner run [--test-window DEADLINE GRACEFUL HARD]
    python -m sepalith.ops.night_runner check
    python -m sepalith.ops.night_runner hold
    python -m sepalith.ops.night_runner window
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as clock_time, timedelta
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import FrameType
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

from sepalith.ops.night_queue import (DEFAULT_STATE_ROOT, EXIT_DEFERRED, Job, Queue, QueueError, _write_json,
                                      attempts_of)

BERLIN = ZoneInfo("Europe/Berlin")
START = clock_time(1, 0)
DEADLINE = clock_time(8, 40)
GRACEFUL = clock_time(8, 50)
HARD = clock_time(9, 0)
REPORT_SCHEMA = "sepalith.night-report.v1"
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
CUDA_LOCK = Path.home() / ".local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock"
SERVICE = "sepalith-night-runner.service"
# `systemctl is-active` states that mean the runner is not running. "failed" is
# normal: the runner exits 1 whenever the report needs the user.
STOPPED_SERVICE_STATES = ("inactive", "failed")
# Passed to every job in addition to its allow-list. None of these are secrets.
BASE_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
            "XDG_RUNTIME_DIR", "WSL_DISTRO_NAME", "WSL_INTEROP")
_MEMORY_QUERY = ("$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction Stop;"
                 "[pscustomobject]@{AvailableMBytes=$m.AvailableMBytes;CommittedBytes=$m.CommittedBytes;"
                 "CommitLimit=$m.CommitLimit} | ConvertTo-Json")

Record = dict[str, Any]
T = TypeVar("T")


@dataclass(frozen=True)
class Window:
    label: str
    night: str
    start: float
    deadline: float
    graceful: float
    hard: float
    test: bool = False

    def describe(self) -> Record:
        return {"label": self.label, "night": self.night, "test": self.test,
                **{name: local_iso(getattr(self, name)) for name in ("start", "deadline", "graceful", "hard")}}


def local_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, BERLIN).isoformat(timespec="seconds")


def _at(day: date, moment: clock_time) -> float:
    return datetime.combine(day, moment, BERLIN).timestamp()


def night_window(now: float) -> Window:
    """The window that is open now, or the next one if today's has ended."""
    today = datetime.fromtimestamp(now, BERLIN).date()
    day = today if now < _at(today, HARD) else today + timedelta(days=1)
    return Window(day.isoformat(), day.isoformat(), _at(day, START), _at(day, DEADLINE), _at(day, GRACEFUL),
                  _at(day, HARD))


def test_window(now: float, deadline_minutes: float, graceful_minutes: float, hard_minutes: float) -> Window:
    """A compressed window that opens now, for dry runs."""
    if not 0 < deadline_minutes <= graceful_minutes <= hard_minutes:
        raise ValueError("test window needs 0 < deadline <= graceful <= hard minutes")
    local = datetime.fromtimestamp(now, BERLIN)
    return Window(f"{local:%Y-%m-%d}-test-{local:%H%M%S}", local.date().isoformat(), now,
                  now + deadline_minutes * 60, now + graceful_minutes * 60, now + hard_minutes * 60, test=True)


class Clock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


@dataclass(frozen=True)
class Limits:
    max_gpu_used_mib: int = 2048
    min_free_disk_bytes: int = 150 * 10**9
    disk_path: Path = Path("/mnt/e")
    # The campaign supervisors admit work at 16 GiB free Windows memory, fail
    # below 4 GiB or above 90% commit, and require 6 GiB available in Linux.
    host_admission_mib: int = 16384
    host_commit_ratio: float = 0.9
    linux_min_available_mib: int = 6144
    cuda_lock: Path = CUDA_LOCK


class Probes:
    """Machine observations. Tests replace these with fakes."""

    def gpu(self) -> Record:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30, check=True).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30, check=True).stdout
        return parse_gpu(used, apps)

    def host_memory(self) -> Record:
        result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", _MEMORY_QUERY],
                                capture_output=True, text=True, timeout=30, check=True)
        value = json.loads(result.stdout)
        linux = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:")}
        return {"windows_available_mib": value["AvailableMBytes"], "committed_bytes": value["CommittedBytes"],
                "commit_limit_bytes": value["CommitLimit"], "linux_available_mib": linux["MemAvailable"] // 1024}

    def runner_service(self) -> str:
        return subprocess.run(["systemctl", "--user", "is-active", SERVICE], capture_output=True, text=True,
                              timeout=30, check=False).stdout.strip()

    def disk_free(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def cuda_lock_free(self, path: Path) -> bool:
        if not path.exists():
            return True
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            os.close(fd)
        return True


def parse_gpu(used: str, apps: str) -> Record:
    values = [int(line.strip()) for line in used.splitlines() if line.strip()]
    if not values:
        raise ValueError("nvidia-smi reported no GPU")
    pids = [line.strip() for line in apps.splitlines() if line.strip() and "No running" not in line]
    return {"used_mib": max(values), "compute_pids": pids}


def preflight(probes: Probes, limits: Limits) -> list[Record]:
    """Admission checks; a probe that fails counts as a failed check."""
    checks: list[Record] = []

    def check(name: str, probe: Callable[[], Record]) -> None:
        try:
            checks.append({"name": name, **probe()})
        except Exception as error:  # noqa: BLE001 - any probe failure blocks admission
            checks.append({"name": name, "ok": False, "error": f"{type(error).__name__}: {error}"})

    def gpu() -> Record:
        value = probes.gpu()
        return {"ok": not value["compute_pids"] and int(str(value["used_mib"])) < limits.max_gpu_used_mib,
                "limit_mib": limits.max_gpu_used_mib, **value}

    def memory() -> Record:
        value = probes.host_memory()
        available = float(str(value["windows_available_mib"]))
        ratio = float(str(value["committed_bytes"])) / float(str(value["commit_limit_bytes"]))
        ok = (available >= limits.host_admission_mib and ratio <= limits.host_commit_ratio
              and float(str(value["linux_available_mib"])) >= limits.linux_min_available_mib)
        return {"ok": ok, "commit_ratio": round(ratio, 4), "admission_mib": limits.host_admission_mib,
                "linux_min_available_mib": limits.linux_min_available_mib, **value}

    def disk() -> Record:
        free = probes.disk_free(limits.disk_path)
        return {"ok": free >= limits.min_free_disk_bytes, "path": str(limits.disk_path), "free_bytes": free,
                "limit_bytes": limits.min_free_disk_bytes}

    def lock() -> Record:
        return {"ok": probes.cuda_lock_free(limits.cuda_lock), "path": str(limits.cuda_lock)}

    check("gpu_idle", gpu)
    check("host_memory", memory)
    check("disk_free", disk)
    check("cuda_lock_free", lock)
    return checks


def _process_table() -> dict[int, tuple[int, int]]:
    """pid -> (parent pid, process group) for every live process."""
    table: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if fields[0] != "Z":  # exited, waiting to be reaped
            table[int(entry.name)] = (int(fields[1]), int(fields[2]))
    return table


def descendants(root: int) -> list[int]:
    table = _process_table()
    children: dict[int, list[int]] = {}
    for pid, (parent, _group) in table.items():
        children.setdefault(parent, []).append(pid)
    found: list[int] = []
    stack = list(children.get(root, []))
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(children.get(pid, []))
    return found


def become_subreaper() -> bool:
    """Orphans of a job reparent to the runner, so the hard stop can find them."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        return False


def _signal(pid: int, number: int) -> None:
    try:
        os.kill(pid, number)
    except ProcessLookupError:
        pass


@dataclass
class JobRun:
    id: str
    attempt: int
    started: str = ""
    ended: str = ""
    exit_code: int | None = None
    outcome: str = ""
    reason: str = ""
    log: str = ""
    graceful_stop_at: str | None = None
    hard_kill_at: str | None = None
    killed_pids: list[int] = field(default_factory=list)
    leftover_pids: list[int] = field(default_factory=list)
    result: Record = field(default_factory=dict)


class NightRunner:
    def __init__(self, state_root: Path, window: Window, *, clock: Clock | None = None, probes: Probes | None = None,
                 limits: Limits | None = None, environ: Mapping[str, str] | None = None, poll_seconds: float = 2.0,
                 preflight_retry_seconds: float = 300.0, min_start_seconds: float = 600.0,
                 operator_grace_seconds: float = 45.0, subreaper: bool = True) -> None:
        self.state_root = Path(state_root)
        self.queue = Queue(self.state_root / "night-queue")
        self.reports = self.state_root / "night-reports"
        self.window = window
        self.clock = clock or Clock()
        self.probes = probes or Probes()
        self.limits = limits or Limits()
        self.environ = dict(os.environ if environ is None else environ)
        self.poll_seconds = poll_seconds
        self.preflight_retry_seconds = preflight_retry_seconds
        self.min_start_seconds = min_start_seconds
        self.operator_grace_seconds = operator_grace_seconds
        self.subreaper = become_subreaper() if subreaper else False
        self.stop_requested = False
        self.run_record: Record = {}

    # Signals -------------------------------------------------------------
    def request_stop(self, _number: int = 0, _frame: FrameType | None = None) -> None:
        self.stop_requested = True

    # Report --------------------------------------------------------------
    def report_path(self, suffix: str) -> Path:
        return self.reports / f"{self.window.label}.{suffix}"

    def _write_report(self) -> None:
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.report_path("json")
        report: Record = json.loads(path.read_text()) if path.exists() else {
            "schema": REPORT_SCHEMA, "label": self.window.label, "window": self.window.describe(), "runs": []}
        runs = [run for run in report["runs"] if run.get("run_id") != self.run_record["run_id"]]
        report["runs"] = [*runs, self.run_record]
        _write_json(path, report)
        self.report_path("md").write_text(render_markdown(report))

    # Queue ---------------------------------------------------------------
    def _recover_stale(self) -> list[str]:
        """running/ jobs from a runner that died. Returns needs-user messages."""
        messages = []
        for job in self.queue.jobs("running"):
            identifier = str(job["id"])
            attempts = attempts_of(job)
            last = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
            group = last.get("pgid")
            alive = False
            if isinstance(group, int):
                try:
                    os.killpg(group, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
            if alive:
                messages.append(f"{identifier} is still in running/ with live process group {group}; "
                                "the runner will not start new jobs until it is resolved")
                continue
            job["outcome"] = {"state": "failed", "reason": "runner_lost", "at": local_iso(self.clock.now())}
            self.queue.move(identifier, "running", "failed", job)
            messages.append(f"{identifier} was left in running/ by a lost runner and was moved to failed/")
        return messages

    def select(self, tried: set[str]) -> tuple[Job | None, list[Record]]:
        now = self.clock.now()
        done = {str(job["id"]) for job in self.queue.jobs("done")}
        failed = {str(job["id"]) for job in self.queue.jobs("failed")}
        candidates = [(job, state) for state in ("pending", "deferred") for job in self.queue.jobs(state)]
        candidates.sort(key=lambda item: (item[0]["priority"], item[1] != "deferred", item[0].get("created", ""),
                                          item[0]["id"]))
        skipped: list[Record] = []
        for job, state in candidates:
            identifier = str(job["id"])
            requires = [str(other) for other in job["requires"]]
            if identifier in tried:
                continue
            if any(other in failed for other in requires):
                skipped.append({"id": identifier, "reason": "requires_failed_job"})
            elif not all(other in done for other in requires):
                skipped.append({"id": identifier, "reason": "waiting_for_requirements"})
            elif not job["resumable"] and now + float(str(job["estimated_minutes"])) * 60 > self.window.deadline:
                skipped.append({"id": identifier, "reason": "does_not_fit_before_deadline"})
            else:
                job["_state"] = state
                return job, skipped
        return None, skipped

    # Execution -----------------------------------------------------------
    def _reap(self, leader: int) -> int | None:
        status = None
        while True:
            try:
                pid, raw = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return status
            if pid == 0:
                return status
            if pid == leader:
                status = os.waitstatus_to_exitcode(raw)

    def _tree(self, leader: int) -> list[int]:
        pids = set(descendants(os.getpid() if self.subreaper else leader))
        if not self.subreaper:
            pids.add(leader)
            pids.update(pid for pid, (_parent, group) in _process_table().items() if group == leader)
        return sorted(pids)

    def _kill_tree(self, leader: int) -> list[int]:
        pids = self._tree(leader)
        for pid in pids:
            _signal(pid, signal.SIGKILL)
        return pids

    def _cleanup_leftovers(self, leader: int) -> list[int]:
        """Processes a job left behind after its leader exited."""
        self._reap(leader)
        leftovers = self._tree(leader)
        if not leftovers:
            return []
        for pid in leftovers:
            _signal(pid, signal.SIGTERM)
        self._wait_gone(leader, 10)
        self._kill_tree(leader)
        self._wait_gone(leader, 10)
        return leftovers

    def _wait_gone(self, leader: int, seconds: float) -> None:
        limit = time.monotonic() + seconds
        while time.monotonic() < limit and self._tree(leader):
            self._reap(leader)
            time.sleep(0.1)

    def execute(self, job: Job) -> JobRun | None:
        """Run one job. None if it left its queue state (cancelled) after selection."""
        identifier = str(job["id"])
        started = local_iso(self.clock.now())
        attempt: Record = {"window": self.window.label, "started": started}

        def prepare(record: Job) -> None:
            number = len(attempts_of(record)) + 1
            attempt.update(attempt=number, directory=str(
                self.queue.root / "attempts" / self.window.label / f"{identifier}-{number}"))
            record["attempts"] = [*attempts_of(record), attempt]

        claimed = self.queue.claim(identifier, str(job.pop("_state")), prepare)
        if claimed is None:
            return None
        job, state = claimed, "running"
        run = JobRun(identifier, int(attempt["attempt"]), started=started)
        directory = Path(str(attempt["directory"]))
        run.log = str(directory / "output.log")
        result_path = directory / "result.json"
        stop_path = directory / "stop-request.json"
        env = {name: self.environ[name] for name in BASE_ENV if name in self.environ}
        names = [str(name) for name in job["env"]]
        missing = [name for name in names if name not in self.environ]
        env.update({name: self.environ[name] for name in names if name in self.environ})
        env.update({
            "SEPALITH_NIGHT_JOB_ID": identifier,
            "SEPALITH_NIGHT_ATTEMPT_DIR": str(directory),
            "SEPALITH_DEADLINE_EPOCH": f"{self.window.deadline:.0f}",
            "SEPALITH_GRACEFUL_STOP_EPOCH": f"{self.window.graceful:.0f}",
            "SEPALITH_HARD_STOP_EPOCH": f"{self.window.hard:.0f}",
            "SEPALITH_STOP_REQUEST_FILE": str(stop_path),
            "SEPALITH_JOB_RESULT": str(result_path),
        })
        if missing or not Path(str(job["cwd"])).is_dir():
            run.outcome, run.reason = "failed", ("missing_env:" + ",".join(missing)) if missing else "missing_cwd"
            run.ended = run.started
            return self._finish(job, state, run, attempt)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with open(run.log, "ab") as log:
                process = subprocess.Popen([str(part) for part in job["command"]],
                                           cwd=str(job["cwd"]), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            run.outcome, run.reason = "failed", f"launch_failed:{type(error).__name__}: {error}"
            run.ended = local_iso(self.clock.now())
            return self._finish(job, state, run, attempt)
        attempt.update(pid=process.pid, pgid=process.pid)
        _write_json(self.queue.path("running", identifier), job)
        print(f"started {identifier} attempt {run.attempt} pid {process.pid}", flush=True)
        status = None
        graceful_deadline = self.window.graceful
        hard_deadline = self.window.hard
        killed_at: float | None = None
        while status is None:
            status = self._reap(process.pid)
            if status is not None:
                break
            now = self.clock.now()
            if self.stop_requested and run.graceful_stop_at is None:
                graceful_deadline = now
                hard_deadline = min(hard_deadline, now + self.operator_grace_seconds)
            if now >= hard_deadline and (killed_at is None or now - killed_at >= 5):
                if killed_at is None:
                    killed_at = now
                    run.hard_kill_at = local_iso(now)
                run.killed_pids = sorted(set(run.killed_pids) | set(self._kill_tree(process.pid)))
                if now - killed_at > 60:
                    run.reason = "unkillable_after_hard_stop"
                    break
            elif now >= graceful_deadline and run.graceful_stop_at is None:
                run.graceful_stop_at = local_iso(now)
                reason = "operator_stop" if self.stop_requested else "night_window_graceful_stop"
                stop_path.write_text(json.dumps({"action": "save_and_stop", "reason": reason,
                                                 "at": run.graceful_stop_at}) + "\n")
                _signal(process.pid, signal.SIGTERM)
            self.clock.sleep(self.poll_seconds)
        process.returncode = status
        run.exit_code = status
        run.leftover_pids = self._cleanup_leftovers(process.pid)
        run.ended = local_iso(self.clock.now())
        if result_path.exists():
            try:
                value = json.loads(result_path.read_text())
                run.result = value if isinstance(value, dict) else {"error": "result is not a JSON object"}
            except (OSError, json.JSONDecodeError) as error:
                run.result = {"error": f"unreadable result: {error}"}
        if run.reason:
            run.outcome = "failed"
        elif killed_at is not None:
            run.outcome, run.reason = "failed", "hard_stop_killed"
        elif status == 0:
            run.outcome = "done"
        elif status == EXIT_DEFERRED and job["resumable"]:
            run.outcome, run.reason = "deferred", "stopped_at_safe_boundary"
        else:
            run.outcome, run.reason = "failed", f"exit_code_{status}"
        return self._finish(job, state, run, attempt)

    def _finish(self, job: Job, state: str, run: JobRun, attempt: Record) -> JobRun:
        attempt.update(ended=run.ended, exit_code=run.exit_code, outcome=run.outcome, reason=run.reason)
        job["outcome"] = {"state": run.outcome, "reason": run.reason, "at": run.ended}
        self.queue.move(run.id, state, run.outcome, job)
        print(f"{run.id}: {run.outcome} ({run.reason or 'ok'}), exit code {run.exit_code}", flush=True)
        return run

    # Main loop -----------------------------------------------------------
    def run(self) -> Record:
        self.queue.ensure()
        lock = open(self.queue.root / "runner.lock", "a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise QueueError("Another night runner holds the queue lock") from None
        try:
            return self._run()
        finally:
            lock.close()

    def _run(self) -> Record:
        started = self.clock.now()
        self.run_record = {"run_id": f"{os.getpid()}-{started:.0f}", "host": socket.gethostname(),
                           "runner_pid": os.getpid(), "started": local_iso(started), "ended": None,
                           "stop_reason": None, "subreaper": self.subreaper, "limits": _limits(self.limits),
                           "preflight": [], "jobs": [], "skipped": [], "withdrawn": [], "needs_user": [],
                           "queue": {}}
        needs: list[str] = self.run_record["needs_user"]
        needs.extend(self._recover_stale())
        blocked = bool(self.queue.jobs("running"))
        tried: set[str] = set()
        stop_reason = "stale_running_job" if blocked else ""
        while not stop_reason:
            now = self.clock.now()
            if self.stop_requested:
                stop_reason = "operator_stop"
            elif now < self.window.start:
                if self.window.start - now > 600:
                    stop_reason = "before_window"
                else:
                    self.clock.sleep(self.window.start - now)
                continue
            elif now >= self.window.deadline - self.min_start_seconds:
                stop_reason = "window_closed" if now >= self.window.deadline else "too_close_to_deadline"
            if stop_reason:
                break
            job, skipped = self.select(tried)
            self.run_record["skipped"] = skipped
            if job is None:
                stop_reason = "no_eligible_job"
                break
            checks = preflight(self.probes, self.limits)
            passed = all(check["ok"] for check in checks)
            self.run_record["preflight"].append(
                {"at": local_iso(now), "for_job": job["id"], "passed": passed, "checks": checks})
            self._write_report()
            if not passed:
                if self.clock.now() + self.preflight_retry_seconds >= self.window.deadline - self.min_start_seconds:
                    stop_reason = "preflight_failed"
                    failing = [str(check["name"]) for check in checks if not check["ok"]]
                    needs.append(f"Preflight never passed tonight ({', '.join(failing)}); {job['id']} did not start")
                else:
                    self.clock.sleep(self.preflight_retry_seconds)
                continue
            tried.add(str(job["id"]))
            run = self.execute(job)
            if run is None:
                self.run_record["withdrawn"].append(str(job["id"]))
                continue
            self.run_record["jobs"].append(asdict(run))
            if run.outcome == "failed":
                needs.append(f"{run.id} failed ({run.reason}); see {run.log}")
            if run.leftover_pids:
                needs.append(f"{run.id} left processes behind after exiting: {run.leftover_pids}")
            if run.result.get("needs_user"):
                needs.append(f"{run.id}: {run.result['needs_user']}")
            self._write_report()
        self.run_record.update(ended=local_iso(self.clock.now()), stop_reason=stop_reason,
                               queue=self.queue.summary())
        self._write_report()
        return self.run_record


def _limits(limits: Limits) -> Record:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(limits).items()}


def render_markdown(report: Mapping[str, Any]) -> str:
    window = report["window"]
    lines = [f"# Night report {report['label']}", "",
             f"Window: {window['start']} to {window['hard']} (deadline {window['deadline']}, "
             f"graceful stop {window['graceful']}){' - compressed test window' if window.get('test') else ''}", ""]
    for run in report.get("runs", []):
        lines += [f"## Runner {run['run_id']}", "",
                  f"Started {run['started']}, ended {run['ended']}, stop reason `{run['stop_reason']}`.", ""]
        if run["needs_user"]:
            lines += ["**Needs the user:**", "", *[f"- {item}" for item in run["needs_user"]], ""]
        if run["jobs"]:
            lines += ["| Job | Attempt | Outcome | Exit | Started | Ended | Graceful stop | Hard kill |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for job in run["jobs"]:
                outcome = job["outcome"] + (f" ({job['reason']})" if job["reason"] else "")
                lines.append(f"| {job['id']} | {job['attempt']} | {outcome} | {job['exit_code']} | {job['started']} "
                             f"| {job['ended']} | {job['graceful_stop_at'] or '-'} | {job['hard_kill_at'] or '-'} |")
            lines.append("")
            for job in run["jobs"]:
                result = job["result"]
                lines.append(f"### {job['id']}")
                lines.append("")
                lines.append(f"- Log: `{job['log']}`")
                if result.get("summary"):
                    lines.append(f"- Summary: {result['summary']}")
                for key in ("checkpoints", "evaluations", "rule_decisions"):
                    if result.get(key):
                        lines.append(f"- {key.replace('_', ' ').capitalize()}: `{json.dumps(result[key])}`")
                if result.get("error"):
                    lines.append(f"- Result file: {result['error']}")
                lines.append("")
        else:
            lines += ["No job ran.", ""]
        failed = [check for entry in run["preflight"] if not entry["passed"] for check in entry["checks"]
                  if not check["ok"]]
        if failed:
            lines += ["Failed preflight checks:", "", *[f"- `{json.dumps(check)}`" for check in failed[-4:]], ""]
        if run.get("withdrawn"):
            lines += ["Withdrawn before start: " + ", ".join(run["withdrawn"]), ""]
        if run["skipped"]:
            lines += ["Not started:", "", *[f"- {item['id']}: {item['reason']}" for item in run["skipped"]], ""]
        queue = run.get("queue") or {}
        if queue:
            lines += ["Queue afterwards: " + ", ".join(f"{state} {len(ids)}" for state, ids in queue.items()), ""]
    check = report.get("post_window_check")
    if isinstance(check, dict):
        lines += ["## Post-window check", "", f"At {check['at']}: {'clean' if check['ok'] else 'NOT CLEAN'}.",
                  f"`{json.dumps(check)}`", ""]
    return "\n".join(lines)


def post_window_check(state_root: Path, now: float, probes: Probes | None = None) -> Record:
    """Log whether anything is still running after the window closed.

    Clean means every observation succeeded and showed nothing running; an
    observation that fails makes the check not clean.
    """
    probes = probes or Probes()
    queue = Queue(state_root / "night-queue")
    problems: list[str] = []

    def observe(name: str, probe: Callable[[], T]) -> T | None:
        try:
            return probe()
        except Exception as error:  # noqa: BLE001 - recorded, and the check is not clean
            problems.append(f"{name}: {type(error).__name__}: {error}")
            return None

    running = observe("queue", lambda: [str(job["id"]) for job in queue.jobs("running")])
    service = observe("runner_service", probes.runner_service)
    gpu = observe("gpu", probes.gpu)
    if service is not None and service not in STOPPED_SERVICE_STATES:
        problems.append(f"runner_service: {service or 'no state'}")
    if gpu is not None and not isinstance(gpu.get("compute_pids"), list):
        problems.append("gpu: no compute process list")
    today = datetime.fromtimestamp(now, BERLIN).date().isoformat()
    ok = not problems and not running and gpu is not None and not gpu["compute_pids"]
    check: Record = {"at": local_iso(now), "night": today, "running_jobs": running, "runner_service": service,
                     "gpu": gpu, "problems": problems, "ok": ok}
    reports = state_root / "night-reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "checks.jsonl").open("a") as handle:
        handle.write(json.dumps(check) + "\n")
    path = reports / f"{today}.json"
    if path.exists():
        report = json.loads(path.read_text())
        report["post_window_check"] = check
        _write_json(path, report)
        (reports / f"{today}.md").write_text(render_markdown(report))
    return check


def hold(state_root: Path, clock: Clock | None = None) -> str:
    """Keep the WSL session alive through the window; used by the Windows wake task.

    Returns early once the runner has finished and nothing is running, so the
    PC may sleep again.
    """
    clock = clock or Clock()
    window = night_window(clock.now())
    if clock.now() < window.start - 3600:
        return "outside_night"
    end = window.hard + 300
    queue = Queue(state_root / "night-queue")
    while clock.now() < end:
        if clock.now() > window.start + 600:
            active = subprocess.run(["systemctl", "--user", "is-active", SERVICE], capture_output=True, text=True,
                                    timeout=30, check=False).stdout.strip()
            if active not in ("active", "activating") and not queue.jobs("running"):
                return "runner_finished"
        clock.sleep(min(60.0, end - clock.now()))
    return "window_over"


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sepalith.ops.night_runner", description=__doc__.split("\n\n")[0])
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT,
                        help="holds night-queue/ and night-reports/ (default: %(default)s)")
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="run queued jobs while the window is open")
    run.add_argument("--test-window", nargs=3, type=float, metavar=("DEADLINE", "GRACEFUL", "HARD"),
                     help="compressed window opening now; minutes until deadline, graceful stop and hard stop")
    run.add_argument("--max-gpu-used-mib", type=int, default=Limits.max_gpu_used_mib)
    run.add_argument("--min-free-disk-gb", type=float, default=Limits.min_free_disk_bytes / 10**9)
    run.add_argument("--disk-path", type=Path, default=Limits.disk_path)
    run.add_argument("--host-admission-mib", type=int, default=Limits.host_admission_mib)
    run.add_argument("--linux-min-available-mib", type=int, default=Limits.linux_min_available_mib)
    run.add_argument("--cuda-lock", type=Path, default=Limits.cuda_lock)
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.add_argument("--preflight-retry-seconds", type=float, default=300.0)
    run.add_argument("--min-start-minutes", type=float, default=10.0,
                     help="start no job later than this many minutes before the deadline")
    commands.add_parser("check", help="log whether anything still runs after the window")
    commands.add_parser("hold", help="keep WSL alive until the runner is finished or 09:05")
    commands.add_parser("window", help="print the current or next window")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    now = time.time()
    if args.action == "window":
        print(json.dumps(night_window(now).describe(), indent=2))
        return 0
    if args.action == "check":
        check = post_window_check(args.state_root, now)
        print(json.dumps(check))
        return 0 if check["ok"] else 1
    if args.action == "hold":
        print(hold(args.state_root), flush=True)
        return 0
    window = test_window(now, *args.test_window) if args.test_window else night_window(now)
    limits = Limits(max_gpu_used_mib=args.max_gpu_used_mib, min_free_disk_bytes=int(args.min_free_disk_gb * 10**9),
                    disk_path=args.disk_path, host_admission_mib=args.host_admission_mib,
                    linux_min_available_mib=args.linux_min_available_mib, cuda_lock=args.cuda_lock)
    runner = NightRunner(args.state_root, window, limits=limits, poll_seconds=args.poll_seconds,
                         preflight_retry_seconds=args.preflight_retry_seconds,
                         min_start_seconds=args.min_start_minutes * 60)
    signal.signal(signal.SIGTERM, runner.request_stop)
    signal.signal(signal.SIGINT, runner.request_stop)
    print(json.dumps({"window": window.describe(), "state_root": str(args.state_root)}), flush=True)
    try:
        record = runner.run()
    except QueueError as error:
        print(f"night_runner: {error}", file=sys.stderr)
        return 3
    print(json.dumps({"stop_reason": record["stop_reason"], "jobs": len(record["jobs"]),
                      "report": str(runner.report_path("md"))}), flush=True)
    return 1 if record["needs_user"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
