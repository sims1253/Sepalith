"""Sandboxed, persistent R execution workers for execution-grounded rewards.

``ExecHarness(workers=8)`` starts that many long-lived ``Rscript`` processes
(``r_worker.R``), each inside ``sandbox.sh``:

- ``unshare -rnm``: a new user, network and mount namespace. The network
  namespace has only a downed loopback, so nothing is reachable.
- every mount is remounted read-only; a private tmpfs (``tmpfs_mb``) is the
  working directory, ``HOME`` and ``TMPDIR``, and the only writable path.
- ``prlimit --as``: an address-space limit per process (forked children inherit it).
- per job: a fresh forked child (``parallel::mcparallel``), an R elapsed-time
  limit and a hard kill after ``timeout + 0.25`` s; output is discarded.

If ``unshare -rnm`` is unavailable the harness falls back to plain processes
with the same limits and ``http_proxy`` pointing nowhere, and says so in
``sandbox_mode``.

A job is (candidate source, test script, expected top-level names, timeout).
The result is a structured ``ExecResult``: parse, structure and source status,
per-assertion outcomes by category, an error class and timing. Worker faults
(the worker process itself dies or speaks garbage) are infrastructure errors:
the job is retried on a fresh worker and then raised, never scored.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
WORKER = HERE / "r_worker.R"
TESTLIB = HERE / "testlib.R"
SANDBOX = HERE / "sandbox.sh"


class HarnessError(RuntimeError):
    """An infrastructure fault (not attributable to the candidate)."""


@dataclass(frozen=True)
class ExecJob:
    source: str
    test: str
    expect: tuple[str, ...] = ()
    timeout: float = 3.0
    id: str = ""


@dataclass
class ExecResult:
    id: str
    status: str  # ok | timeout | candidate_exit | child_error
    parse_ok: bool | None
    structure_ok: bool | None
    sourced_ok: bool | None
    checks: list[tuple[str, bool]] = field(default_factory=list)
    error_class: str | None = None
    error: str | None = None
    elapsed: float = 0.0
    wall: float = 0.0

    @property
    def passed(self) -> int:
        return sum(ok for _, ok in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for category, ok in self.checks:
            cell = out.setdefault(category, [0, 0])
            cell[0] += int(ok)
            cell[1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "parse_ok": self.parse_ok, "structure_ok": self.structure_ok,
                "sourced_ok": self.sourced_ok, "passed": self.passed, "total": self.total,
                "by_category": self.by_category(), "error_class": self.error_class, "error": self.error,
                "elapsed": round(self.elapsed, 4), "wall": round(self.wall, 4)}


def namespaces_available() -> bool:
    if shutil.which("unshare") is None:
        return False
    try:
        return subprocess.run(["unshare", "-rnm", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class _Worker:
    def __init__(self, index: int, rscript: str, mem_mb: int, tmpfs_mb: int, sandbox: bool, nice: int) -> None:
        self.index = index
        self.workdir = tempfile.mkdtemp(prefix=f"sepalith-exec-{index}-")
        cmd = [rscript, "--vanilla", str(WORKER), str(TESTLIB)]
        if sandbox:
            cmd = ["unshare", "-rnm", "--propagation", "private", "sh", str(SANDBOX), self.workdir, str(tmpfs_mb), "--",
                   *cmd]
        if shutil.which("prlimit"):
            cmd = ["prlimit", f"--as={mem_mb * 1024 * 1024}", f"--fsize={tmpfs_mb * 1024 * 1024}", "--core=0", *cmd]
        env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        env.update(http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9", R_LIBS_USER=self.workdir,
                   TMPDIR=self.workdir, HOME=self.workdir, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=self.workdir, env=env, preexec_fn=(lambda: os.nice(max(0, nice - os.nice(0)))) if nice else None)

    def run(self, job: ExecJob) -> ExecResult:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        src = job.source.encode("utf-8")
        test = job.test.encode("utf-8")
        for name in job.expect:
            if "," in name or '"' in name or "\\" in name:
                raise HarnessError(f"unsupported top-level name {name!r}")
        header = json.dumps({"id": job.id, "timeout": float(job.timeout), "n_src": len(src), "n_test": len(test),
                             "expect": ",".join(job.expect)}, ensure_ascii=True)
        started = time.perf_counter()
        try:
            self.proc.stdin.write(header.encode() + b"\n" + src + test)
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise HarnessError(f"worker {self.index} pipe failed: {error}") from error
        if not line:
            raise HarnessError(f"worker {self.index} exited (code {self.proc.poll()})")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise HarnessError(f"worker {self.index} sent garbage: {line[:200]!r}") from error
        checks = []
        for row in str(raw.get("checks") or "").split("\n"):
            if row:
                category, _, ok = row.rpartition("\t")
                checks.append((category, ok == "1"))
        return ExecResult(str(raw["id"]), str(raw["status"]), raw.get("parse_ok"), raw.get("structure_ok"),
                          raw.get("sourced_ok"), checks, raw.get("error_class"), raw.get("error"),
                          float(raw.get("elapsed") or 0.0), time.perf_counter() - started)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        shutil.rmtree(self.workdir, ignore_errors=True)


class ExecHarness:
    """A pool of sandboxed R workers. Thread-safe; ``run_many`` fans out over all workers."""

    def __init__(self, workers: int = 8, *, rscript: str = "Rscript", mem_mb: int = 2048, tmpfs_mb: int = 64,
                 sandbox: bool | None = None, nice: int = 10, retries: int = 1) -> None:
        self.sandbox = namespaces_available() if sandbox is None else sandbox
        self.sandbox_mode = "unshare-rnm+ro-remount+tmpfs" if self.sandbox else "process-only"
        self.settings = (rscript, mem_mb, tmpfs_mb, self.sandbox, nice)
        self.retries = retries
        self._idle: queue.Queue[_Worker] = queue.Queue()
        self._all: list[_Worker] = []
        self._lock = threading.Lock()
        for index in range(workers):
            self._spawn(index)
        self.workers = workers

    def _spawn(self, index: int) -> None:
        worker = _Worker(index, *self.settings)
        with self._lock:
            self._all.append(worker)
        self._idle.put(worker)

    def run(self, job: ExecJob) -> ExecResult:
        last: Exception | None = None
        for _attempt in range(self.retries + 1):
            worker = self._idle.get()
            try:
                result = worker.run(job)
            except HarnessError as error:
                last = error
                worker.close()
                with self._lock:
                    self._all.remove(worker)
                self._spawn(worker.index)
                continue
            self._idle.put(worker)
            return result
        raise HarnessError(f"job {job.id} failed on {self.retries + 1} workers: {last}")

    def run_many(self, jobs: Sequence[ExecJob]) -> list[ExecResult]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.run, jobs))

    def close(self) -> None:
        with self._lock:
            workers, self._all = self._all, []
        for worker in workers:
            worker.close()

    def __enter__(self) -> ExecHarness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
