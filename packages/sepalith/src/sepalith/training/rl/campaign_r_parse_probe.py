#!/usr/bin/env python3
"""Fixed-harness R parse probe. Candidate text is parsed and never executed.

Reward v2.1 (card C11) fixes the reward-v2 parser bug: only the harness's
designated syntax-error exit code counts as invalid R. Any other nonzero exit,
a signal, a timeout or unusable output is an infrastructure failure. It is
logged and retried, and if it persists it raises
``ParseInfrastructureError``. It is never scored as a model failure.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import logging
from pathlib import Path
import subprocess
import tempfile
import threading
import time

PARSE_OK_EXIT = 0
USAGE_ERROR_EXIT = 2
SYNTAX_ERROR_EXIT = 3
PARSE_OPERATION = "base::parse(file=...,keep.source=FALSE)"
DEFAULT_HARNESS = Path(__file__).with_name("reward_parse_only.R")

LOGGER = logging.getLogger("sepalith.training.rl.parse_probe")


class ParseProbeError(RuntimeError):
    """The probe is misconfigured or could not produce a syntax verdict."""


class ParseInfrastructureError(ParseProbeError):
    """R did not finish normally. The candidate was not judged."""


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RParseOnlyProbe:
    """Run the pinned harness with ``Rscript --vanilla``.

    ``probe(text)`` returns True when the text parses and False on the
    designated syntax-error exit. ``parse_many(texts)`` parses a batch in one R
    process. Both retry infrastructure failures up to ``max_attempts`` times.
    """

    def __init__(
        self,
        harness: Path,
        expected_sha256: str,
        *,
        timeout_seconds: float = 5.0,
        command: Sequence[str] = ("Rscript", "--vanilla"),
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        batch_seconds_per_input: float = 0.25,
    ) -> None:
        if sha_file(harness) != expected_sha256:
            raise ParseProbeError("parse_harness_hash_mismatch")
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ParseProbeError("parse_timeout_out_of_bounds")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ParseProbeError("parse_max_attempts_out_of_bounds")
        self.harness = harness.resolve()
        self.harness_sha256 = expected_sha256
        self.timeout_seconds = float(timeout_seconds)
        self.command = tuple(command)
        self.max_attempts = max_attempts
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.batch_seconds_per_input = float(batch_seconds_per_input)
        self.infrastructure_failures = 0
        self._lock = threading.Lock()
        self._r_version: str | None = None

    # -- infrastructure ---------------------------------------------------
    def _run(self, arguments: Sequence[str], timeout: float, label: str) -> subprocess.CompletedProcess[bytes]:
        """Run the harness once per attempt; return only a normal exit.

        A normal exit is one of the designated codes. Everything else is
        logged, counted and retried, then raised.
        """
        last = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = subprocess.run(
                    [*self.command, str(self.harness), *arguments],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                last = f"timeout_after_{timeout:g}s"
            except OSError as error:
                last = f"os_error:{type(error).__name__}"
            else:
                if result.returncode in (PARSE_OK_EXIT, SYNTAX_ERROR_EXIT):
                    return result
                if result.returncode < 0:
                    last = f"signal_{-result.returncode}"
                else:
                    last = f"exit_{result.returncode}"
            with self._lock:
                self.infrastructure_failures += 1
            LOGGER.warning("R parse probe %s infrastructure failure (%s), attempt %d of %d",
                           label, last, attempt, self.max_attempts)
            if attempt < self.max_attempts:
                time.sleep(self.retry_delay_seconds)
        raise ParseInfrastructureError(f"parse_probe_infrastructure_failure:{label}:{last}")

    def r_version(self) -> str:
        """Return the parser identity, such as ``R 4.6.1 (2026-06-24)``."""
        if self._r_version is None:
            result = self._run(["--version"], self.timeout_seconds, "version")
            text = result.stdout.decode("utf-8", "replace").strip()
            if result.returncode != PARSE_OK_EXIT or not text.startswith("R "):
                raise ParseInfrastructureError(f"parse_probe_version_unusable:{text[:80]!r}")
            self._r_version = text
        return self._r_version

    def identity(self) -> dict[str, object]:
        return {
            "operation": PARSE_OPERATION,
            "generated_r_executed": False,
            "command": " ".join(self.command),
            "harness_sha256": self.harness_sha256,
            "r_version": self.r_version(),
            "syntax_error_exit_code": SYNTAX_ERROR_EXIT,
        }

    # -- parsing ----------------------------------------------------------
    def __call__(self, text: str) -> bool:
        if not isinstance(text, str):
            raise ParseProbeError("parse_input_not_text")
        with tempfile.TemporaryDirectory(prefix="sepalith-parse-") as directory:
            path = Path(directory) / "input.R"
            path.write_text(text, encoding="utf-8", newline="")
            result = self._run([str(path)], self.timeout_seconds, "single")
        return result.returncode == PARSE_OK_EXIT

    def parse_many(self, texts: Sequence[str]) -> list[bool]:
        """Parse many texts in one R process; same verdicts as ``__call__``."""
        if any(not isinstance(text, str) for text in texts):
            raise ParseProbeError("parse_input_not_text")
        if not texts:
            return []
        timeout = self.timeout_seconds + self.batch_seconds_per_input * len(texts)
        with tempfile.TemporaryDirectory(prefix="sepalith-parse-") as directory:
            root = Path(directory)
            paths = []
            for index, text in enumerate(texts):
                path = root / f"input-{index}.R"
                path.write_text(text, encoding="utf-8", newline="")
                paths.append(str(path))
            listing = root / "inputs.txt"
            listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
            output = root / "verdicts.tsv"
            for attempt in range(1, self.max_attempts + 1):
                result = self._run(["--batch", str(listing), str(output)], timeout, "batch")
                verdicts = self._read_batch(output, len(texts)) if result.returncode == PARSE_OK_EXIT else None
                if verdicts is not None:
                    return verdicts
                with self._lock:
                    self.infrastructure_failures += 1
                LOGGER.warning("R parse probe batch output unusable, attempt %d of %d",
                               attempt, self.max_attempts)
                output.unlink(missing_ok=True)
        raise ParseInfrastructureError("parse_probe_infrastructure_failure:batch:unusable_output")

    @staticmethod
    def _read_batch(path: Path, count: int) -> list[bool] | None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        if len(lines) != count + 1 or lines[-1] != f"DONE\t{count}":
            return None
        verdicts = []
        for index, line in enumerate(lines[:-1]):
            if line == f"{index}\t{PARSE_OK_EXIT}":
                verdicts.append(True)
            elif line == f"{index}\t{SYNTAX_ERROR_EXIT}":
                verdicts.append(False)
            else:
                return None
        return verdicts
