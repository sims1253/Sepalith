"""File-based queue for the nightly GPU window.

Each job is one JSON file named ``<id>.json`` in exactly one state directory:
``pending/``, ``running/``, ``done/``, ``failed/`` or ``deferred/``. The runner
moves files between them with atomic renames. Jobs carry an environment
allow-list of variable names only; values are copied from the runner's
environment at launch and are never written to disk.

Usage::

    python -m sepalith.ops.night_queue enqueue --id ID --cwd DIR [--env NAME]... \
        [--priority N] [--resumable] [--estimated-minutes M] [--requires ID]... -- CMD ARG...
    python -m sepalith.ops.night_queue enqueue --file job.json
    python -m sepalith.ops.night_queue list [--state STATE] [--json]
    python -m sepalith.ops.night_queue show ID
    python -m sepalith.ops.night_queue cancel ID
    python -m sepalith.ops.night_queue retry ID
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

SCHEMA = "sepalith.night-job.v1"
STATES = ("pending", "running", "done", "failed", "deferred")
DEFAULT_STATE_ROOT = Path.home() / ".local/state/sepalith"
DEFAULT_ROOT = DEFAULT_STATE_ROOT / "night-queue"
# Exit code a resumable job uses to say "saved at a safe boundary, not finished".
EXIT_DEFERRED = 75

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
_REQUIRED = {"id", "command", "cwd"}
_OPTIONAL = {"env", "priority", "resumable", "estimated_minutes", "requires", "note"}
_RUNTIME = {"schema", "created", "attempts", "outcome"}

Job = dict[str, Any]


class QueueError(ValueError):
    """An invalid job or queue operation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_id(value: object) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise QueueError(f"Invalid job id: {value!r}")
    return value


def validate_job(value: object) -> Job:
    """Check a job definition and fill defaults. Runtime keys are preserved."""
    if not isinstance(value, dict):
        raise QueueError("A job must be a JSON object")
    keys = {str(key) for key in value}
    if missing := _REQUIRED - keys:
        raise QueueError(f"Job is missing {sorted(missing)}")
    if unknown := keys - _REQUIRED - _OPTIONAL - _RUNTIME:
        raise QueueError(f"Unknown job keys {sorted(unknown)}; env takes names only, never values")
    job: Job = {str(key): item for key, item in value.items()}
    job_id(job["id"])
    command = job["command"]
    if not isinstance(command, list) or not command or not all(isinstance(v, str) and v for v in command):
        raise QueueError("command must be a non-empty list of non-empty strings")
    cwd = job["cwd"]
    if not isinstance(cwd, str) or not Path(cwd).is_absolute():
        raise QueueError("cwd must be an absolute path")
    env = job.setdefault("env", [])
    if not isinstance(env, list) or not all(isinstance(v, str) and _ENV_NAME.fullmatch(v) for v in env):
        raise QueueError("env must be a list of environment variable names")
    if len(set(env)) != len(env):
        raise QueueError("env names must be unique")
    priority = job.setdefault("priority", 50)
    if not isinstance(priority, int) or isinstance(priority, bool) or not 0 <= priority <= 1000:
        raise QueueError("priority must be an integer from 0 (first) to 1000 (last)")
    if not isinstance(job.setdefault("resumable", False), bool):
        raise QueueError("resumable must be true or false")
    minutes = job.setdefault("estimated_minutes", 60)
    if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or not 0 < minutes <= 24 * 60:
        raise QueueError("estimated_minutes must be a positive number of minutes")
    requires = job.setdefault("requires", [])
    if not isinstance(requires, list):
        raise QueueError("requires must be a list of job ids")
    for other in requires:
        if job_id(other) == job["id"]:
            raise QueueError("A job cannot require itself")
    if not isinstance(job.setdefault("note", ""), str):
        raise QueueError("note must be a string")
    job.setdefault("schema", SCHEMA)
    if job["schema"] != SCHEMA:
        raise QueueError(f"Unsupported job schema {job['schema']!r}")
    if not isinstance(job.setdefault("attempts", []), list):
        raise QueueError("attempts must be a list")
    return job


def attempts_of(job: Mapping[str, Any]) -> list[Any]:
    value = job.get("attempts", [])
    return value if isinstance(value, list) else []


def secret_leaks(job: Mapping[str, object], environ: Mapping[str, str]) -> list[str]:
    """Names of secret-looking variables whose value appears in the job text."""
    text = json.dumps(job)
    return sorted(name for name, value in environ.items()
                  if _SECRET_NAME.search(name) and len(value) >= 8 and value in text)


def _write_json(path: Path, value: object) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Queue:
    """State directories under one root. Only the runner moves jobs to running/."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def ensure(self) -> None:
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)

    def path(self, state: str, identifier: str) -> Path:
        return self.root / state / f"{job_id(identifier)}.json"

    def locate(self, identifier: str) -> str | None:
        found = [state for state in STATES if self.path(state, identifier).exists()]
        if len(found) > 1:
            raise QueueError(f"Job {identifier} exists in several states: {found}")
        return found[0] if found else None

    def load(self, state: str, identifier: str) -> Job:
        return validate_job(json.loads(self.path(state, identifier).read_text()))

    def jobs(self, state: str) -> list[Job]:
        directory = self.root / state
        if not directory.is_dir():
            return []
        return [self.load(state, p.stem) for p in sorted(directory.glob("*.json"))]

    def enqueue(self, job: Mapping[str, object], environ: Mapping[str, str] | None = None) -> Job:
        record = validate_job(dict(job))
        if record["attempts"] or "outcome" in record:
            raise QueueError("New jobs cannot carry attempts or an outcome")
        if leaks := secret_leaks(record, os.environ if environ is None else environ):
            raise QueueError(f"The job contains the value of {leaks}; list the names in env instead")
        self.ensure()
        identifier = str(record["id"])
        if (state := self.locate(identifier)) is not None:
            raise QueueError(f"Job {identifier} already exists in {state}/")
        record["created"] = record.get("created") or utc_now()
        target = self.path("pending", identifier)
        _write_json(target, record)
        return record

    def move(self, identifier: str, source: str, target: str, record: Job | None = None) -> Job:
        """Rewrite the record in place, then rename it into the target state."""
        if source not in STATES or target not in STATES:
            raise QueueError(f"Unknown state {source!r} or {target!r}")
        path = self.path(source, identifier)
        if record is None:
            record = self.load(source, identifier)
        _write_json(path, validate_job(record))
        self.ensure()
        destination = self.path(target, identifier)
        if destination.exists():
            raise QueueError(f"Job {identifier} already exists in {target}/")
        os.replace(path, destination)
        return record

    def cancel(self, identifier: str) -> Job:
        state = self.locate(identifier)
        if state not in ("pending", "deferred"):
            raise QueueError(f"Only pending or deferred jobs can be cancelled; {identifier} is {state}")
        record = self.load(state, identifier)
        record["outcome"] = {"state": "failed", "reason": "cancelled", "at": utc_now()}
        return self.move(identifier, state, "failed", record)

    def retry(self, identifier: str) -> Job:
        state = self.locate(identifier)
        if state not in ("failed", "deferred"):
            raise QueueError(f"Only failed or deferred jobs can be retried; {identifier} is {state}")
        record = self.load(state, identifier)
        record.pop("outcome", None)
        return self.move(identifier, state, "pending", record)

    def summary(self) -> dict[str, list[str]]:
        return {state: [str(job["id"]) for job in self.jobs(state)] for state in STATES}


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sepalith.ops.night_queue", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="queue root (default: %(default)s)")
    commands = parser.add_subparsers(dest="action", required=True)
    add = commands.add_parser("enqueue", help="add a job to pending/")
    add.add_argument("--file", type=Path, help="read the job definition from a JSON file")
    add.add_argument("--id")
    add.add_argument("--cwd", type=Path)
    add.add_argument("--env", action="append", default=[], metavar="NAME", help="allow-listed variable name")
    add.add_argument("--priority", type=int, default=50, help="0 runs first, 1000 last")
    add.add_argument("--resumable", action="store_true", help="the job saves and exits 75 at the deadline")
    add.add_argument("--estimated-minutes", type=float, default=60)
    add.add_argument("--requires", action="append", default=[], metavar="ID", help="run only after ID is done")
    add.add_argument("--note", default="")
    add.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND ARG...")
    show = commands.add_parser("list", help="list jobs by state")
    show.add_argument("--state", choices=STATES)
    show.add_argument("--json", action="store_true")
    for name in ("show", "cancel", "retry"):
        commands.add_parser(name).add_argument("id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    queue = Queue(args.root)
    try:
        if args.action == "enqueue":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if args.file is not None:
                if args.id or args.cwd or command:
                    raise QueueError("--file cannot be combined with --id, --cwd or a command")
                job = json.loads(args.file.read_text())
            else:
                if not args.id or args.cwd is None or not command:
                    raise QueueError("enqueue needs --id, --cwd and -- COMMAND (or --file)")
                job = {"id": args.id, "command": command, "cwd": str(args.cwd.resolve()), "env": args.env,
                       "priority": args.priority, "resumable": args.resumable,
                       "estimated_minutes": args.estimated_minutes, "requires": args.requires, "note": args.note}
            record = queue.enqueue(job)
            print(f"queued {record['id']} in {queue.path('pending', str(record['id']))}")
        elif args.action == "list":
            states = [args.state] if args.state else list(STATES)
            listing = {state: queue.jobs(state) for state in states}
            if args.json:
                print(json.dumps(listing, indent=2))
            for state in states if not args.json else []:
                for job in listing[state]:
                    flags = "resumable" if job["resumable"] else "one-shot"
                    print(f"{state:9} {job['id']:40} p={job['priority']:<4} ~{job['estimated_minutes']}m {flags}"
                          f" attempts={len(attempts_of(job))}")
        elif args.action == "show":
            state = queue.locate(args.id)
            if state is None:
                raise QueueError(f"No job {args.id}")
            print(json.dumps({"state": state, **queue.load(state, args.id)}, indent=2))
        elif args.action == "cancel":
            queue.cancel(args.id)
            print(f"cancelled {args.id}")
        else:
            queue.retry(args.id)
            print(f"moved {args.id} back to pending/")
    except (QueueError, OSError, json.JSONDecodeError) as error:
        print(f"night_queue: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
