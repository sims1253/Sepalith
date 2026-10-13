#!/usr/bin/env python3
"""Small, local campaign board with a dependency-aware state machine.

The board is deliberately a standard-library program.  It is a control-plane
view of the campaign: reading it or serving it never launches training,
CUDA, cloud, or other jobs.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import fcntl
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


SCHEMA_VERSION = 1
STATUSES = {"todo", "in_progress", "blocked", "done", "deferred"}
KINDS = {"required", "conditional", "optional"}
MANIFEST_KEYS = {"schemaVersion", "campaignId", "title", "deadline", "phases", "tasks"}
PHASE_KEYS = {"id", "title", "window", "goal"}
TASK_KEYS = {
    "id",
    "phase",
    "title",
    "summary",
    "kind",
    "resource",
    "role",
    "effort",
    "dependsOn",
    "steps",
    "acceptance",
    "outputs",
    "sources",
    "delegate",
    "stopRule",
}
OPTIONAL_TASK_KEYS = {"gate"}
STATE_KEYS = {"schemaVersion", "campaignId", "revision", "tasks", "updatedAt"}
TASK_STATE_KEYS = {"status", "owner", "note", "receipt", "updatedAt"}
STATIC_EXTENSIONS = {
    ".html",
    ".css",
    ".js",
    ".mjs",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".webp",
}
_ID_RE = re.compile(r"^[^/\\\x00]+$")


class CampaignError(Exception):
    """A user-correctable campaign or input error."""


class ConflictError(CampaignError):
    """An optimistic-concurrency revision did not match the stored state."""


class NotFoundError(CampaignError):
    """A requested task or file does not exist."""


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be an object")
    return value


def _exact_keys(value: dict, required: set[str], optional: set[str], label: str) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise CampaignError(f"{label} missing required field(s): {', '.join(missing)}")
    if unknown:
        raise CampaignError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"{label} must be a non-empty string")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CampaignError(f"{label} must be a string")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CampaignError(f"{label} must be a string or list of strings")
    return value


def _task_id(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    if not _ID_RE.match(text) or text in {".", ".."}:
        raise CampaignError(f"{label} contains an invalid path-like identifier")
    return text


def validate_dag(manifest: dict) -> dict:
    """Validate manifest shape, references, and dependency acyclicity."""

    manifest = _require_object(manifest, "manifest")
    _exact_keys(manifest, MANIFEST_KEYS, set(), "manifest")
    if manifest["schemaVersion"] != SCHEMA_VERSION:
        raise CampaignError(f"manifest schemaVersion must be {SCHEMA_VERSION}")
    for field in ("campaignId", "title", "deadline"):
        _nonempty_string(manifest[field], f"manifest.{field}")
    phases = manifest["phases"]
    if not isinstance(phases, list) or not phases:
        raise CampaignError("manifest.phases must be a non-empty list")
    phase_ids: set[str] = set()
    for index, phase in enumerate(phases):
        phase = _require_object(phase, f"phase[{index}]")
        _exact_keys(phase, PHASE_KEYS, set(), f"phase[{index}]")
        pid = _task_id(phase["id"], f"phase[{index}].id")
        if pid in phase_ids:
            raise CampaignError(f"duplicate phase id: {pid}")
        phase_ids.add(pid)
        for field in ("title", "window", "goal"):
            _nonempty_string(phase[field], f"phase[{index}].{field}")

    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise CampaignError("manifest.tasks must be a non-empty list")
    task_ids: set[str] = set()
    task_map: dict[str, dict] = {}
    for index, task in enumerate(tasks):
        task = _require_object(task, f"task[{index}]")
        _exact_keys(task, TASK_KEYS, OPTIONAL_TASK_KEYS, f"task[{index}]")
        tid = _task_id(task["id"], f"task[{index}].id")
        if tid in task_ids:
            raise CampaignError(f"duplicate task id: {tid}")
        task_ids.add(tid)
        task_map[tid] = task
        phase = _task_id(task["phase"], f"task[{tid}].phase")
        if phase not in phase_ids:
            raise CampaignError(f"task {tid} references unknown phase {phase}")
        for field in ("title", "summary", "resource", "role", "effort", "stopRule"):
            _nonempty_string(task[field], f"task[{tid}].{field}")
        delegate = task["delegate"]
        if isinstance(delegate, str):
            if not delegate.strip():
                raise CampaignError(f"task[{tid}].delegate must not be empty")
        elif not isinstance(delegate, bool):
            raise CampaignError(f"task[{tid}].delegate must be a boolean or string")
        kind = _nonempty_string(task["kind"], f"task[{tid}].kind")
        if kind not in KINDS:
            raise CampaignError(f"task {tid} kind must be one of {sorted(KINDS)}")
        deps = task["dependsOn"]
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            raise CampaignError(f"task {tid}.dependsOn must be a list of strings")
        if tid in deps:
            raise CampaignError(f"task {tid} cannot depend on itself")
        for field in ("steps", "acceptance", "outputs", "sources"):
            _string_list(task[field], f"task[{tid}].{field}")
        if "gate" in task and not isinstance(task["gate"], (dict, list, str, int, float, bool, type(None))):
            raise CampaignError(f"task {tid}.gate must be JSON-compatible")

    for tid, task in task_map.items():
        for dep in task["dependsOn"]:
            if dep not in task_ids:
                raise CampaignError(f"task {tid} references unknown dependency {dep}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(tid: str, trail: list[str]) -> None:
        if tid in visiting:
            cycle = " -> ".join(trail + [tid])
            raise CampaignError(f"dependency cycle: {cycle}")
        if tid in visited:
            return
        visiting.add(tid)
        for dep in task_map[tid]["dependsOn"]:
            visit(dep, trail + [tid])
        visiting.remove(tid)
        visited.add(tid)

    for tid in task_map:
        visit(tid, [])
    return manifest


def validate_manifest(manifest: dict) -> dict:
    """Public manifest validator used by the CLI and the board builder."""

    return validate_dag(manifest)


# CamelCase aliases make the small validation API easy for the build script to
# import without imposing a naming convention on it.
validateDAG = validate_dag
validateManifest = validate_manifest


def _manifest_task_map(manifest: dict) -> dict[str, dict]:
    return {task["id"]: task for task in manifest["tasks"]}


def empty_state(manifest: dict) -> dict:
    manifest = validate_manifest(manifest)
    timestamp = _now()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "campaignId": manifest["campaignId"],
        "revision": 0,
        "tasks": {
            task["id"]: {
                "status": "todo",
                "owner": "",
                "note": "",
                "receipt": "",
                "updatedAt": timestamp,
            }
            for task in manifest["tasks"]
        },
        "updatedAt": timestamp,
    }


def _descendants(manifest: dict, task_id: str) -> set[str]:
    task_map = _manifest_task_map(manifest)
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for tid, task in task_map.items():
            if tid in result:
                continue
            if task_id in task["dependsOn"] or any(dep in result for dep in task["dependsOn"]):
                result.add(tid)
                changed = True
    return result


def _check_rollback(manifest: dict, current: dict, candidate: dict) -> None:
    """Reject silently invalidating work that has already started or finished."""

    for task in manifest["tasks"]:
        tid = task["id"]
        before = current["tasks"][tid]["status"]
        after = candidate["tasks"][tid]["status"]
        if before == "done" and after != "done":
            active_descendants = [
                child
                for child in _descendants(manifest, tid)
                if current["tasks"][child]["status"] in {"in_progress", "done"}
            ]
            if active_descendants:
                raise CampaignError(
                    f"cannot revert prerequisite {tid}: descendant work already started or finished "
                    f"({', '.join(sorted(active_descendants))})"
                )


def _task_entry_is_empty(entry: dict) -> bool:
    """Whether an entry still contains only the initial, unassigned state."""

    return (
        entry["status"] == "todo"
        and not entry["owner"].strip()
        and not entry["note"].strip()
        and not entry["receipt"].strip()
    )


def _check_import_conflicts(manifest: dict, current: dict, candidate: dict) -> None:
    """Prevent an offline import from replacing existing task progress.

    An import carries one optimistic revision, but a copied state file cannot
    prove that its task-level changes descended from the current file.  Permit
    progress to be added only to entries that are still empty, and require
    populated entries to be byte-for-byte equal.  A caller that needs to
    revise an existing task must use the task update API with the current
    revision (or reconcile the file explicitly before importing).
    """

    for task in manifest["tasks"]:
        tid = task["id"]
        before = current["tasks"][tid]
        after = candidate["tasks"][tid]
        if not _task_entry_is_empty(before) and before != after:
            raise CampaignError(
                f"import would overwrite existing progress for {tid}; reconcile it with a revision-checked update"
            )


def validate_state(manifest: dict, state: dict) -> dict:
    """Validate state shape, task IDs, status requirements, and dependency gates."""

    manifest = validate_manifest(manifest)
    state = _require_object(state, "state")
    _exact_keys(state, STATE_KEYS, set(), "state")
    if state["schemaVersion"] != SCHEMA_VERSION:
        raise CampaignError(f"state schemaVersion must be {SCHEMA_VERSION}")
    if state["campaignId"] != manifest["campaignId"]:
        raise CampaignError("state campaignId does not match manifest")
    if not _is_int(state["revision"]) or state["revision"] < 0:
        raise CampaignError("state.revision must be a non-negative integer")
    _string(state["updatedAt"], "state.updatedAt")
    states = state["tasks"]
    if not isinstance(states, dict):
        raise CampaignError("state.tasks must be an object")
    task_map = _manifest_task_map(manifest)
    expected_ids = set(task_map)
    actual_ids = set(states)
    unknown = sorted(actual_ids - expected_ids)
    missing = sorted(expected_ids - actual_ids)
    if unknown:
        raise CampaignError(f"state has unknown task id(s): {', '.join(unknown)}")
    if missing:
        raise CampaignError(f"state is missing task id(s): {', '.join(missing)}")

    for tid, task in task_map.items():
        entry = _require_object(states[tid], f"state.tasks.{tid}")
        _exact_keys(entry, TASK_STATE_KEYS, set(), f"state.tasks.{tid}")
        status = _nonempty_string(entry["status"], f"state.tasks.{tid}.status")
        if status not in STATUSES:
            raise CampaignError(f"state.tasks.{tid}.status must be one of {sorted(STATUSES)}")
        for field in ("owner", "note", "receipt", "updatedAt"):
            _string(entry[field], f"state.tasks.{tid}.{field}")
        if status == "done" and (not entry["note"].strip() or not entry["receipt"].strip()):
            raise CampaignError(f"task {tid} marked done requires a non-empty note and receipt")
        if status == "deferred":
            if task["kind"] == "required":
                raise CampaignError(f"required task {tid} cannot be deferred; block it or edit the main plan")
            if not entry["note"].strip():
                raise CampaignError(f"deferred task {tid} requires a reason in note")
        if status == "blocked" and not entry["note"].strip():
            raise CampaignError(f"blocked task {tid} requires a reason in note")
        if status in {"in_progress", "done"}:
            unresolved = [
                dep for dep in task["dependsOn"] if states[dep]["status"] != "done"
            ]
            if unresolved:
                raise CampaignError(
                    f"task {tid} cannot be {status} until dependencies are done: {', '.join(unresolved)}"
                )
    return state


validateState = validate_state


def _read_json(path: Path, label: str) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise NotFoundError(f"{label} not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise CampaignError(f"{label} is not valid JSON: {exc}") from None
    except OSError as exc:
        raise CampaignError(f"cannot read {label}: {exc}") from None


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _file_lock(path: Path, exclusive: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CampaignStore:
    """Manifest/state access with one lock and atomic state replacement."""

    def __init__(self, directory: str | os.PathLike[str] | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.tasks_path = self.directory / "tasks.json"
        self.state_path = self.directory / "state.json"
        self.lock_path = self.directory / ".state.json.lock"

    def load_manifest(self) -> dict:
        return validate_manifest(_read_json(self.tasks_path, "tasks.json"))

    def _load_state_unlocked(self, manifest: dict, initialize: bool) -> dict:
        if not self.state_path.exists():
            if not initialize:
                raise NotFoundError(f"state.json not found: {self.state_path}")
            state = empty_state(manifest)
            _write_atomic(self.state_path, state)
            return state
        state = _read_json(self.state_path, "state.json")
        return validate_state(manifest, state)

    def read_state(self, initialize: bool = True) -> dict:
        manifest = self.load_manifest()
        if initialize and not self.state_path.exists():
            with _file_lock(self.lock_path, True):
                self._load_state_unlocked(manifest, True)
        with _file_lock(self.lock_path, False):
            return copy.deepcopy(self._load_state_unlocked(manifest, initialize))

    def campaign_payload(self) -> dict:
        manifest = self.load_manifest()
        state = self.read_state(True)
        return {"manifest": manifest, "state": state}

    def _mutate(self, expected_revision: int | None, operation):
        manifest = self.load_manifest()
        with _file_lock(self.lock_path, True):
            current = self._load_state_unlocked(manifest, True)
            if expected_revision is not None:
                if not _is_int(expected_revision) or expected_revision < 0:
                    raise CampaignError("revision must be a non-negative integer")
                if current["revision"] != expected_revision:
                    raise ConflictError(
                        f"revision conflict: expected {expected_revision}, current {current['revision']}"
                    )
            candidate = operation(copy.deepcopy(current), manifest)
            candidate = validate_state(manifest, candidate)
            _check_rollback(manifest, current, candidate)
            candidate["revision"] = current["revision"] + 1
            candidate["updatedAt"] = _now()
            _write_atomic(self.state_path, candidate)
            return copy.deepcopy(candidate)

    def update(self, task_id: str, changes: dict, expected_revision: int | None = None) -> dict:
        task_id = _task_id(task_id, "taskId")
        changes = _require_object(changes, "changes")
        if not changes:
            raise CampaignError("changes must contain at least one field")
        unknown = sorted(set(changes) - {"status", "owner", "note", "receipt"})
        if unknown:
            raise CampaignError(f"changes has unknown field(s): {', '.join(unknown)}")
        for field, value in changes.items():
            if field == "status":
                if not isinstance(value, str) or value not in STATUSES:
                    raise CampaignError(f"changes.status must be one of {sorted(STATUSES)}")
            elif not isinstance(value, str):
                raise CampaignError(f"changes.{field} must be a string")

        def operation(current: dict, manifest: dict) -> dict:
            if task_id not in current["tasks"]:
                raise NotFoundError(f"unknown task id: {task_id}")
            entry = current["tasks"][task_id]
            for field, value in changes.items():
                entry[field] = value
            entry["updatedAt"] = _now()
            return current

        return self._mutate(expected_revision, operation)

    def import_state(self, candidate: dict, expected_revision: int) -> dict:
        candidate = copy.deepcopy(_require_object(candidate, "state"))
        if not _is_int(expected_revision) or expected_revision < 0:
            raise CampaignError("revision must be a non-negative integer")

        def operation(current: dict, manifest: dict) -> dict:
            # Validate candidate against the current manifest before changing
            # its revision.  The request revision, rather than a possibly
            # stale embedded candidate revision, protects the write.
            validate_state(manifest, candidate)
            _check_import_conflicts(manifest, current, candidate)
            return candidate

        return self._mutate(expected_revision, operation)

    def import_file(self, path: str | os.PathLike[str], expected_revision: int | None = None) -> dict:
        payload = _read_json(Path(path), "import state")
        payload = _require_object(payload, "import state")
        if "state" in payload:
            _exact_keys(payload, {"revision", "state"}, set(), "import envelope")
            if expected_revision is not None and payload["revision"] != expected_revision:
                raise ConflictError(
                    f"revision conflict: file expects {payload['revision']}, CLI expects {expected_revision}"
                )
            expected_revision = payload["revision"]
            candidate = payload["state"]
        else:
            # An offline export carries its own branch revision, not this
            # store's base revision. Read the live revision, then protect the
            # mutation with that value and task-level conflict checks.
            if expected_revision is None:
                expected_revision = self.read_state()["revision"]
            candidate = payload
        return self.import_state(candidate, expected_revision)

    def ready_tasks(self) -> list[dict]:
        manifest = self.load_manifest()
        state = self.read_state(True)
        return [
            task
            for task in manifest["tasks"]
            if state["tasks"][task["id"]]["status"] == "todo"
            and all(state["tasks"][dep]["status"] == "done" for dep in task["dependsOn"])
        ]

    def brief(self, task_id: str) -> str:
        manifest = self.load_manifest()
        state = self.read_state(True)
        task_map = _manifest_task_map(manifest)
        if task_id not in task_map:
            raise NotFoundError(f"unknown task id: {task_id}")
        task = task_map[task_id]
        entry = state["tasks"][task_id]
        phases = {phase["id"]: phase for phase in manifest["phases"]}
        phase = phases[task["phase"]]

        def bullets(value: object) -> str:
            values = _string_list(value, "brief field")
            return "\n".join(f"- {item}" for item in values) if values else "- (none listed)"

        lines = [
            f"# {task['id']}: {task['title']}",
            "",
            f"- **Campaign:** {manifest['title']} ({manifest['campaignId']})",
            f"- **Deadline:** {manifest['deadline']}",
            f"- **Phase:** {phase['title']} ({phase['window']})",
            f"- **Kind:** {task['kind']}",
            f"- **Status:** {entry['status']}",
            f"- **Owner:** {entry['owner'] or '(unassigned)'}",
            f"- **Resource:** {task['resource']}",
            f"- **Role:** {task['role']}",
            f"- **Effort:** {task['effort']}",
            "",
            "## Summary",
            "",
            task["summary"],
            "",
            "## Dependencies",
            "",
            ", ".join(task["dependsOn"]) if task["dependsOn"] else "None",
            "",
            "## Steps",
            "",
            bullets(task["steps"]),
            "",
            "## Acceptance",
            "",
            bullets(task["acceptance"]),
            "",
            "## Outputs",
            "",
            bullets(task["outputs"]),
            "",
            "## Sources",
            "",
            bullets(task["sources"]),
            "",
            "## Delegation and stop rule",
            "",
            f"Delegate: {task['delegate']}",
            "",
            task["stopRule"],
            "",
            "## Control constraints",
            "",
            "CUDA launch is lead-only. This brief and the board are control-plane text; reading, validating, importing, or serving them never launches a job.",
        ]
        if "gate" in task:
            lines.extend(["", "## Gate", "", "```json", json.dumps(task["gate"], indent=2, ensure_ascii=False), "```"])
        if entry["note"]:
            lines.extend(["", "## Current note", "", entry["note"]])
        if entry["receipt"]:
            lines.extend(["", f"Receipt: `{entry['receipt']}`"])
        return "\n".join(lines) + "\n"


def _default_directory() -> Path:
    return Path(__file__).resolve().parent


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and operate the Sepalith campaign board")
    parser.add_argument("--directory", default=str(_default_directory()), help="campaign folder containing tasks.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate tasks.json and state.json")
    list_parser = sub.add_parser("list", help="list campaign tasks")
    list_parser.add_argument("--ready", action="store_true", help="show only todo tasks whose dependencies are done")
    brief_parser = sub.add_parser("brief", help="print a self-contained task brief")
    brief_parser.add_argument("id")
    update_parser = sub.add_parser("update", help="update one task state")
    update_parser.add_argument("id")
    update_parser.add_argument("--status", required=True, choices=sorted(STATUSES))
    update_parser.add_argument("--owner")
    update_parser.add_argument("--note")
    update_parser.add_argument("--receipt")
    update_parser.add_argument("--revision", type=int, help="expected state revision")
    import_parser = sub.add_parser("import-state", help="optimistically import a state JSON file")
    import_parser.add_argument("file")
    import_parser.add_argument("--revision", type=int, help="expected current revision")
    serve_parser = sub.add_parser("serve", help="serve the local board and API")
    serve_parser.add_argument("--port", type=int, default=8766)
    return parser


def _command_main(args: argparse.Namespace) -> int:
    store = CampaignStore(args.directory)
    if args.command == "validate":
        manifest = store.load_manifest()
        state = store.read_state(True)
        validate_state(manifest, state)
        print(f"valid: {len(manifest['tasks'])} tasks, {len(manifest['phases'])} phases, revision {state['revision']}")
        return 0
    if args.command == "list":
        manifest = store.load_manifest()
        state = store.read_state(True)
        for task in manifest["tasks"]:
            entry = state["tasks"][task["id"]]
            ready = entry["status"] == "todo" and all(
                state["tasks"][dep]["status"] == "done" for dep in task["dependsOn"]
            )
            if args.ready and not ready:
                continue
            deps = ",".join(task["dependsOn"]) or "-"
            print(f"{task['id']}\t{entry['status']}\t{'ready' if ready else 'waiting'}\t{deps}\t{task['title']}")
        return 0
    if args.command == "brief":
        print(store.brief(args.id), end="")
        return 0
    if args.command == "update":
        changes = {field: getattr(args, field) for field in ("status", "owner", "note", "receipt") if getattr(args, field) is not None}
        _print_json(store.update(args.id, changes, args.revision))
        return 0
    if args.command == "import-state":
        _print_json(store.import_file(args.file, args.revision))
        return 0
    if args.command == "serve":
        serve_campaign(store, args.port)
        return 0
    raise CampaignError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _cli_parser()
    args = parser.parse_args(argv)
    try:
        return _command_main(args)
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


class CampaignHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, directory: str | os.PathLike[str]):
        self.store = CampaignStore(directory)
        super().__init__(server_address, CampaignRequestHandler)


class CampaignRequestHandler(BaseHTTPRequestHandler):
    server: CampaignHTTPServer

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - BaseHTTPRequestHandler API
        # Keep the board useful in a terminal without dumping request bodies.
        sys.stderr.write("campaign: " + (format % args) + "\n")

    def _local_host(self) -> bool:
        host = self.headers.get("Host", "")
        if not host:
            self._error(HTTPStatus.BAD_REQUEST, "Host header is required")
            return False
        try:
            hostname = urlsplit("//" + host).hostname
        except ValueError:
            hostname = None
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            self._error(HTTPStatus.FORBIDDEN, "only localhost requests are allowed")
            return False
        return True

    def _local_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = None
        try:
            parsed = urlsplit(origin)
            hostname = parsed.hostname
        except ValueError:
            hostname = None
        if parsed is None or parsed.scheme not in {"http", "https"}:
            self._error(HTTPStatus.FORBIDDEN, "only localhost origins are allowed")
            return False
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            self._error(HTTPStatus.FORBIDDEN, "only localhost origins are allowed")
            return False
        return True

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _json(self, status: HTTPStatus, value: object) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _text(self, status: HTTPStatus, value: str, content_type: str = "text/plain; charset=utf-8") -> None:
        payload = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            raise CampaignError("Content-Length must be an integer") from None
        if length < 0 or length > 1_048_576:
            raise CampaignError("request body is missing or too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"request body is not valid JSON: {exc}") from None

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._local_host():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path == "/api/campaign":
                self._json(HTTPStatus.OK, self.server.store.campaign_payload())
                return
            if path.startswith("/api/brief/"):
                task_id = unquote(path[len("/api/brief/"):])
                if not task_id or "/" in task_id:
                    raise NotFoundError("invalid task id")
                self._text(HTTPStatus.OK, self.server.store.brief(task_id))
                return
            if path == "/":
                path = "/board.html"
            self._serve_static(path)
        except NotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except CampaignError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def _serve_static(self, path: str) -> None:
        relative = unquote(path).lstrip("/")
        if not relative or any(part in {"", ".", ".."} or part.startswith(".") for part in relative.split("/")):
            raise NotFoundError("static file not found")
        candidate = (self.server.store.directory / relative).resolve()
        try:
            candidate.relative_to(self.server.store.directory)
        except ValueError:
            raise NotFoundError("static file not found") from None
        if candidate.name in {"state.json", "tasks.json"} or candidate.suffix.lower() not in STATIC_EXTENSIONS:
            raise NotFoundError("static file not found")
        if not candidate.is_file():
            raise NotFoundError("static file not found")
        try:
            payload = candidate.read_bytes()
        except OSError as exc:
            raise CampaignError(f"cannot serve static file: {exc}") from None
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._local_host() or not self._local_origin():
            return
        path = urlsplit(self.path).path
        try:
            payload = _require_object(self._body(), "request body")
            if path == "/api/state":
                _exact_keys(payload, {"revision", "taskId", "changes"}, set(), "state request")
                task_id = _task_id(payload["taskId"], "taskId")
                changes = _require_object(payload["changes"], "changes")
                result = self.server.store.update(task_id, changes, payload["revision"])
                self._json(HTTPStatus.OK, result)
                return
            if path == "/api/import":
                _exact_keys(payload, {"revision", "state"}, set(), "import request")
                result = self.server.store.import_state(payload["state"], payload["revision"])
                self._json(HTTPStatus.OK, result)
                return
            self._error(HTTPStatus.NOT_FOUND, "API route not found")
        except ConflictError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except NotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except CampaignError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))


def serve_campaign(store: CampaignStore, port: int = 8766) -> None:
    if not _is_int(port) or not 1 <= port <= 65535:
        raise CampaignError("port must be between 1 and 65535")
    # Validate before opening the socket so a malformed board fails clearly.
    store.load_manifest()
    store.read_state(True)
    try:
        server = CampaignHTTPServer(("127.0.0.1", port), store.directory)
    except OSError as exc:
        raise CampaignError(f"cannot bind 127.0.0.1:{port}: {exc}") from None
    print(f"campaign board: http://127.0.0.1:{port}/board.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
