"""A serial, durable local runner. No training libraries are imported.

Only this runner's jobs are coordinated. Existing supervisors must be drained
before activating it on the shared machine. Steps must remain in the foreground.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from typing import TypedDict


class SnapshotEntry(TypedDict):
    path: str
    sha256: str
    mode: int


class RunnerError(ValueError):
    """An invalid recipe or a state requiring operator attention."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value):
    """Publish a complete record before a terminal database transition."""
    # Each publication owns its temporary file. A stale or step-created
    # <record>.tmp symlink must never redirect a write outside the attempt.
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise RunnerError(f"Invalid identifier: {value!r}")
    return value


def _relative(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise RunnerError(f"Expected a relative path without '..': {value!r}")
    return Path(value)


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _under(root, relative):
    path = root / _relative(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RunnerError(f"Path escapes its root: {relative}")
    return path


def validate_recipe(recipe):
    """Validate executable structure; scientific acceptance is a recipe step."""
    if not isinstance(recipe, dict) or type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise RunnerError("Recipe requires schema_version=1")
    allowed = {"schema_version", "id", "description", "snapshot", "depends_on", "resource",
               "python", "env", "inputs", "steps", "provenance"}
    if set(recipe) - allowed:
        raise RunnerError(f"Unknown recipe fields: {sorted(map(str, set(recipe) - allowed))}")
    _name(recipe.get("id"))
    if not re.fullmatch(r"[0-9a-f]{64}", str(recipe.get("snapshot", ""))):
        raise RunnerError("snapshot must be a SHA-256 identifier from the snapshot command")
    if recipe.get("resource", "cpu") not in ("cpu", "gpu", "quiet"):
        raise RunnerError("resource must be cpu, gpu, or quiet")
    deps = recipe.get("depends_on", [])
    if not isinstance(deps, list) or any(_name(d) == recipe["id"] for d in deps):
        raise RunnerError("depends_on must list other experiment identifiers")
    if len(set(deps)) != len(deps):
        raise RunnerError("Duplicate dependency")
    env = recipe.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str)
                                         or "=" in k or "\0" in k + v for k, v in env.items()):
        raise RunnerError("env must map environment names to strings")
    if "python" in recipe and (not isinstance(recipe["python"], str)
                               or not Path(recipe["python"]).is_absolute()):
        raise RunnerError("python must be an absolute interpreter path")
    inputs = recipe.get("inputs", [])
    if not isinstance(inputs, list):
        raise RunnerError("inputs must be a list")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise RunnerError("Each input requires exactly path and sha256")
        if not isinstance(item["path"], str) or not Path(item["path"]).is_absolute():
            raise RunnerError("Input paths must be absolute")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])):
            raise RunnerError("Input sha256 must contain 64 lowercase hex characters")
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RunnerError("At least one step is required")
    names = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) - {"id", "argv", "artifacts"}:
            raise RunnerError("Step fields are id, argv and artifacts")
        if _name(step.get("id")) in names:
            raise RunnerError("Duplicate step identifier")
        names.add(step["id"])
        argv = step.get("argv")
        if not isinstance(argv, list) or not argv or not argv[0] or any(not isinstance(a, str) or "\0" in a for a in argv):
            raise RunnerError("argv must be a nonempty list of strings; shell strings are not accepted")
        artifacts = step.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise RunnerError("artifacts must be a list of relative file paths")
        for artifact in artifacts:
            _relative(artifact)
    try:
        _json(recipe)
    except (TypeError, ValueError) as exc:
        raise RunnerError("Recipe must contain finite JSON values") from exc
    return recipe


class Runner:
    """Queue, snapshot and execute experiments through a small persistent interface."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("snapshots", "attempts"):
            (self.root / folder).mkdir(exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES ('paused', '1');
                CREATE TABLE IF NOT EXISTS jobs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    recipe TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued');
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, job TEXT NOT NULL, status TEXT NOT NULL,
                    phase TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0,
                    worker_pid INTEGER, child_pid INTEGER, child_group INTEGER,
                    started REAL NOT NULL, finished REAL, detail TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS operator_resolutions (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt TEXT UNIQUE NOT NULL, record TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS operator_resolutions_no_update
                    BEFORE UPDATE ON operator_resolutions BEGIN
                        SELECT RAISE(ABORT, 'Operator resolution records are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS operator_resolutions_no_delete
                    BEFORE DELETE ON operator_resolutions BEGIN
                        SELECT RAISE(ABORT, 'Operator resolution records are append-only');
                    END;
            """)

    @contextlib.contextmanager
    def _db(self):
        db = sqlite3.connect(self.root / "state.sqlite3", timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def _dispatch_lock(self):
        with (self.root / "dispatch.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunnerError("Another dispatcher owns this queue") from exc
            yield

    def snapshot(self, repo, includes):
        """Copy explicitly selected Git-visible files, including uncommitted bytes."""
        repo = Path(repo).resolve()
        if not includes:
            raise RunnerError("Select source files/directories explicitly with --include")
        prefixes = [_relative(p) for p in includes]
        raw = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z", "--cached",
                                       "--others", "--exclude-standard"])
        visible = sorted(set(os.fsdecode(p) for p in raw.split(b"\0") if p))
        selected = [p for p in visible if any(Path(p) == prefix or prefix in Path(p).parents
                                             or str(prefix) == "." for prefix in prefixes)]
        if not selected:
            raise RunnerError("No Git-visible files match the requested source paths")
        with tempfile.TemporaryDirectory(prefix="snapshot-", dir=self.root) as tmp:
            stage = Path(tmp)
            tree = stage / "source"
            tree.mkdir()
            entries: list[SnapshotEntry] = []
            for rel in selected:
                src = _under(repo, rel)
                if src.is_symlink() or not src.is_file():
                    raise RunnerError(f"Snapshot requires regular files: {rel}")
                if src.resolve().is_relative_to(self.root):
                    raise RunnerError("Runner state cannot be included in a source snapshot")
                dst = tree / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                mode = 0o555 if src.stat().st_mode & 0o111 else 0o444
                dst.chmod(mode)
                with dst.open("rb") as handle:
                    os.fsync(handle.fileno())
                entries.append({"path": rel, "sha256": _hash(dst), "mode": mode})
            identity = hashlib.sha256(_json(entries).encode()).hexdigest()
            manifest = {"schema_version": 1, "id": identity, "files": entries,
                        "origin": str(repo), "captured_at": time.time(),
                        "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                                             text=True).strip()}
            # Verify the working files still match their copied bytes before
            # publishing. Snapshot capture is not a substitute for coordinating
            # simultaneous edits spanning multiple files.
            for entry in entries:
                if _hash(repo / entry["path"]) != entry["sha256"]:
                    raise RunnerError("Source changed during snapshot; capture again after edits finish")
            _write_json(stage / "manifest.json", manifest)
            for directory in sorted((p for p in tree.rglob("*") if p.is_dir()), reverse=True):
                _sync_dir(directory)
            _sync_dir(tree)
            target = self.root / "snapshots" / identity
            try:
                stage.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            _sync_dir(target.parent)
        self.verify_snapshot(identity)
        return identity

    def verify_snapshot(self, identity):
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity)):
            raise RunnerError("Invalid snapshot identifier")
        folder = self.root / "snapshots" / identity
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            entries = manifest["files"]
            if hashlib.sha256(_json(entries).encode()).hexdigest() != identity:
                raise RunnerError("Snapshot manifest hash mismatch")
            tree = folder / "source"
            actual = {str(p.relative_to(tree)) for p in tree.rglob("*") if p.is_file() or p.is_symlink()}
            if actual != {entry["path"] for entry in entries}:
                raise RunnerError("Snapshot file inventory changed")
            for entry in entries:
                path = _under(tree, entry["path"])
                if path.is_symlink() or _hash(path) != entry["sha256"] or (path.stat().st_mode & 0o777) != entry["mode"]:
                    raise RunnerError(f"Snapshot content changed: {entry['path']}")
            return tree
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RunnerError(f"Snapshot is missing or invalid: {identity}") from exc

    def enqueue(self, recipe):
        recipe = validate_recipe(recipe)
        self.verify_snapshot(recipe["snapshot"])
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            graph = {row["id"]: json.loads(row["recipe"]).get("depends_on", [])
                     for row in db.execute("SELECT id, recipe FROM jobs")}
            if recipe["id"] in graph:
                raise RunnerError("Experiment already exists; use retry for a failed attempt")
            graph[recipe["id"]] = recipe.get("depends_on", [])
            def visit(node, path):
                if node in path:
                    raise RunnerError("Dependency cycle: " + " -> ".join(path + [node]))
                for dep in graph.get(node, []):
                    visit(dep, path + [node])
            for node in graph:
                visit(node, [])
            db.execute("INSERT INTO jobs(id, recipe) VALUES (?, ?)", (recipe["id"], _json(recipe)))
        return recipe["id"]

    def pause(self):
        with self._db() as db:
            db.execute("UPDATE settings SET value='1' WHERE key='paused'")

    def resume(self):
        with self._db() as db:
            db.execute("UPDATE settings SET value='0' WHERE key='paused'")

    def plan(self):
        """Inspect eligibility without launching anything or hashing large inputs."""
        with self._db() as db:
            paused = db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0] == "1"
            jobs = [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY seq")]
            attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY started")]
            resolutions = [json.loads(row[0]) for row in db.execute(
                "SELECT record FROM operator_resolutions ORDER BY seq")]
        states = {j["id"]: j["status"] for j in jobs}
        for job in jobs:
            job["recipe"] = json.loads(job["recipe"])
            job["blocked_by"] = [d for d in job["recipe"].get("depends_on", []) if states.get(d) != "succeeded"]
        return {"paused": paused, "jobs": jobs, "attempts": attempts, "operator_resolutions": resolutions,
                "policy": "serial; external workloads are not coordinated"}

    def _update_attempt(self, attempt, **values):
        with self._db() as db:
            keys = list(values)
            db.execute("UPDATE attempts SET " + ", ".join(k + "=?" for k in keys) + " WHERE id=?",
                       [values[k] for k in keys] + [attempt])

    def _finish(self, attempt, status, detail=""):
        with self._db() as db:
            db.execute("UPDATE attempts SET status=?, phase='finished', detail=?, finished=? WHERE id=?",
                       (status, detail, time.time(), attempt))
            db.execute("UPDATE jobs SET status=? WHERE id=(SELECT job FROM attempts WHERE id=?)",
                       (status, attempt))

    def run_next(self):
        """Run one eligible experiment synchronously under the dispatcher lock."""
        with self._dispatch_lock():
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT 1 FROM attempts WHERE status='running'").fetchone():
                    raise RunnerError("An unfinished attempt needs reconciliation before dispatch")
                if db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0] == "1":
                    return None
                jobs = [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY seq")]
                succeeded = {j["id"] for j in jobs if j["status"] == "succeeded"}
                job = next((j for j in jobs if j["status"] == "queued" and
                            set(json.loads(j["recipe"]).get("depends_on", [])) <= succeeded), None)
                if job is None:
                    return None
                recipe = json.loads(job["recipe"])
                attempt = uuid.uuid4().hex
                db.execute("INSERT INTO attempts(id, job, status, phase, worker_pid, started) "
                           "VALUES (?, ?, 'running', 'preflight', ?, ?)",
                           (attempt, job["id"], os.getpid(), time.time()))
                db.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
            work = self.root / "attempts" / attempt
            try:
                work.mkdir()
                _sync_dir(work.parent)
                _write_json(work / "recipe.json", recipe)
                source = self.verify_snapshot(recipe["snapshot"])
                interpreter = recipe.get("python", sys.executable)
                if not Path(interpreter).is_file() or not os.access(interpreter, os.X_OK):
                    raise RunnerError(f"Interpreter is unavailable: {interpreter}")
                for item in recipe.get("inputs", []):
                    if not Path(item["path"]).is_file():
                        raise RunnerError(f"Input is missing or not a regular file: {item['path']}")
                    if _hash(Path(item["path"])) != item["sha256"]:
                        raise RunnerError(f"Input hash mismatch: {item['path']}")
                # An attempt-local source copy supports old scripts that write beside
                # themselves while keeping the content-addressed snapshot untouched.
                shutil.copytree(source, work / "source")
                source_entries = json.loads((source.parent / "manifest.json").read_text())["files"]
                def verify_attempt_source():
                    for entry in source_entries:
                        path = _under(work / "source", entry["path"])
                        if path.is_symlink() or not path.is_file() or _hash(path) != entry["sha256"] or (path.stat().st_mode & 0o777) != entry["mode"]:
                            raise RunnerError(f"Attempt source was modified: {entry['path']}")
                substitutions = {"{source}": str(work / "source"), "{run}": str(work), "{python}": interpreter}
                def resolve(value):
                    for token, replacement in substitutions.items():
                        value = value.replace(token, replacement)
                    return value
                env = dict(os.environ)
                env.update({k: resolve(v) for k, v in recipe.get("env", {}).items()})
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                if recipe.get("resource", "cpu") != "gpu":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                _write_json(work / "runtime.json", {"runner_python": sys.version,
                            "runner_source_sha256": _hash(Path(__file__)), "step_python": interpreter,
                            "platform": sys.platform, "resource": recipe.get("resource", "cpu"),
                            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                            "inherited_environment_captured": False})
                executed = []
                for index, step in enumerate(recipe["steps"]):
                    verify_attempt_source()
                    argv = [resolve(a) for a in step["argv"]]
                    with (work / f"{index:02d}-{step['id']}.log").open("ab") as log:
                        self._update_attempt(attempt, phase="launching", step=index, child_pid=None, child_group=None)
                        try:
                            child = subprocess.Popen(argv, cwd=work / "source", env=env,
                                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                        except (OSError, ValueError):
                            # Popen failed before returning a child: unlike a crash
                            # in this interval, its absence is known.
                            self._update_attempt(attempt, phase="checking")
                            raise
                        self._update_attempt(attempt, phase="executing", child_pid=child.pid, child_group=child.pid)
                        code = child.wait()
                        log.flush()
                        os.fsync(log.fileno())
                    # Check this process group. Children that escape with setsid
                    # cannot be contained by this runner and are forbidden by its
                    # foreground-step contract.
                    if _group_alive(child.pid):
                        self._update_attempt(attempt, phase="needs_review", detail="Step left live child processes")
                        raise RunnerError("Step left child processes; queue remains blocked for reconciliation")
                    self._update_attempt(attempt, phase="checking", child_pid=None, child_group=None)
                    executed.append({"step": step["id"], "argv": argv, "returncode": code})
                    _write_json(work / "execution.json", executed)
                    if code != 0:
                        raise RunnerError(f"Step {step['id']} exited with code {code}")
                    verify_attempt_source()
                    artifacts = []
                    for relative in step.get("artifacts", []):
                        path = _under(work, relative)
                        if not path.is_file() or path.stat().st_size == 0 or path.is_symlink():
                            raise RunnerError(f"Missing or empty artifact after {step['id']}: {relative}")
                        artifacts.append({"path": relative, "sha256": _hash(path), "bytes": path.stat().st_size})
                        with path.open("rb") as handle:
                            os.fsync(handle.fileno())
                        # Artifact directories may have been created by the step.
                        for directory in (path.parent, *path.parent.parents):
                            if directory == work.parent:
                                break
                            _sync_dir(directory)
                    _write_json(work / f"{index:02d}-{step['id']}-artifacts.json", artifacts)
                self._finish(attempt, "succeeded")
            except (Exception, KeyboardInterrupt) as exc:
                with self._db() as db:
                    state = dict(db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone())
                # In launch/execute phases the workload may still exist. Do not
                # free its resource claim or retry it merely because the parent failed.
                if state["phase"] in ("launching", "executing", "needs_review"):
                    self._update_attempt(attempt, detail=str(exc))
                    raise RunnerError(f"Attempt {attempt} needs reconciliation: {exc}") from exc
                self._finish(attempt, "failed", str(exc))
            return attempt

    def recover(self, attempt):
        """Close an interrupted attempt only when no recorded process group lives."""
        with self._dispatch_lock(), self._db() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row is None or row["status"] != "running":
                raise RunnerError("Expected an unfinished attempt")
            if row["phase"] == "launching":
                raise RunnerError("Interrupted launch has unknown child identity; manual process audit required")
            if row["child_group"] and _group_alive(row["child_group"]):
                raise RunnerError("The recorded process group is still alive; leaving it untouched")
            db.execute("UPDATE attempts SET status='interrupted', phase='finished', finished=?, "
                       "detail='Dispatcher stopped; explicit retry required' WHERE id=?", (time.time(), attempt))
            db.execute("UPDATE jobs SET status='interrupted' WHERE id=?", (row["job"],))

    def retry(self, job):
        with self._db() as db:
            cursor = db.execute("UPDATE jobs SET status='queued' WHERE id=? AND status IN ('failed','interrupted')", (job,))
            if cursor.rowcount != 1:
                raise RunnerError("Only failed or interrupted experiments can be retried")

    def resolve_unknown(self, attempt, audit):
        """Record an operator's process audit; never infer absence or stop a process."""
        fields = {"schema_version", "attempt", "operator", "decision", "checked_at",
                  "all_processes_stopped", "findings", "evidence"}
        if not isinstance(audit, dict) or set(audit) != fields:
            raise RunnerError("Audit requires exactly: " + ", ".join(sorted(fields)))
        if type(audit["schema_version"]) is not int or audit["schema_version"] != 1:
            raise RunnerError("Audit requires schema_version=1")
        if audit["attempt"] != attempt or audit["decision"] not in ("failed", "interrupted"):
            raise RunnerError("Audit must name this attempt and decide failed or interrupted")
        if audit["all_processes_stopped"] is not True:
            raise RunnerError("Operator must affirm that all processes for this attempt have stopped")
        for field in ("operator", "checked_at", "findings", "evidence"):
            if not isinstance(audit[field], str) or not audit[field].strip():
                raise RunnerError(f"Audit {field} must be nonempty text")
        try:
            checked = datetime.fromisoformat(audit["checked_at"].replace("Z", "+00:00"))
            if checked.tzinfo is None:
                raise ValueError("timezone required")
            checked_at = checked.timestamp()
        except (ValueError, OverflowError) as exc:
            raise RunnerError("Audit checked_at must be an ISO timestamp with a timezone") from exc
        with self._dispatch_lock(), self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0] != "1":
                raise RunnerError("Pause the queue before resolving an unknown launch")
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row is None or row["status"] != "running" or row["phase"] != "launching":
                raise RunnerError("Expected an unfinished attempt with an unknown launch")
            now = time.time()
            if not row["started"] <= checked_at <= now:
                raise RunnerError("Process audit must occur after the attempt started and not in the future")
            if row["child_group"] and _group_alive(row["child_group"]):
                raise RunnerError("The recorded process group is still alive; leaving it untouched")
            recipe = db.execute("SELECT recipe FROM jobs WHERE id=?", (row["job"],)).fetchone()[0]
            record = {"schema_version": 1, "recorded_at": now, "attempt_before": dict(row),
                      "recipe": json.loads(recipe), "audit": audit}
            # One transaction preserves the original receipt and the operator's
            # evidence before changing status. A failed commit releases no claim.
            db.execute("INSERT INTO operator_resolutions(attempt, record) VALUES (?, ?)",
                       (attempt, _json(record)))
            detail = "Operator resolved unknown launch; explicit retry required"
            db.execute("UPDATE attempts SET status=?, phase='finished', finished=?, detail=? WHERE id=?",
                       (audit["decision"], now, detail, attempt))
            db.execute("UPDATE jobs SET status=? WHERE id=?", (audit["decision"], row["job"]))
            return record


def _group_alive(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Orphaned zombies can persist under a container's PID 1. They cannot
    # execute or write artifacts and should not reserve a resource forever.
    # Any unreadable process state keeps the conservative live answer.
    try:
        with os.scandir("/proc") as processes:
            for item in processes:
                if not item.name.isdigit():
                    continue
                try:
                    state = (Path(item.path) / "stat").read_text().rsplit(")", 1)[1].split()
                    if int(state[2]) == group and state[0] not in ("Z", "X"):
                        return True
                except FileNotFoundError:
                    continue
        return False
    except (OSError, ValueError, IndexError):
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="Persistent local state directory")
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--repo", required=True)
    snap.add_argument("--include", action="append", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("recipe", type=Path)
    for name in ("plan", "pause", "resume", "run-next", "run"):
        sub.add_parser(name)
    recover = sub.add_parser("recover")
    recover.add_argument("attempt")
    retry = sub.add_parser("retry")
    retry.add_argument("job")
    resolve = sub.add_parser("resolve-unknown")
    resolve.add_argument("attempt")
    resolve.add_argument("--audit", required=True, type=Path,
                         help="JSON with operator decision and captured process-audit evidence")
    args = parser.parse_args(argv)
    try:
        runner = Runner(args.state)
        if args.command == "snapshot":
            result = runner.snapshot(args.repo, args.include)
        elif args.command == "enqueue":
            result = runner.enqueue(json.loads(args.recipe.read_text()))
        elif args.command == "run-next":
            result = runner.run_next()
        elif args.command == "run":
            result = []
            while True:
                attempt = runner.run_next()
                if attempt is None:
                    break
                result.append(attempt)
                status = next(a["status"] for a in runner.plan()["attempts"] if a["id"] == attempt)
                if status != "succeeded":
                    print(json.dumps(result, indent=2))
                    return 1
        elif args.command == "recover":
            result = runner.recover(args.attempt)
        elif args.command == "retry":
            result = runner.retry(args.job)
        elif args.command == "resolve-unknown":
            result = runner.resolve_unknown(args.attempt, json.loads(args.audit.read_text()))
        else:
            result = getattr(runner, args.command)()
        print(json.dumps(result, indent=2))
        if args.command == "run-next" and result:
            status = next(a["status"] for a in runner.plan()["attempts"] if a["id"] == result)
            return 0 if status == "succeeded" else 1
        return 0
    except (RunnerError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"sepalith-run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
