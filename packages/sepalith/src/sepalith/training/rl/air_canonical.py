"""Canonical R text through the `air` formatter (reward v2.1, card C11).

``AirFormatter.format`` returns the formatted text, or ``None`` when air
rejects the input as unparsable. Any other failure (a missing binary, a signal,
a timeout, an unexpected exit) is an infrastructure failure: it is logged and
retried, then raised. It is never scored as a model failure.
"""
from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import subprocess
import threading
import time

AIR_SYNTAX_EXIT = 255
AIR_SYNTAX_MESSAGE = "Failed to parse"
# air reads stdin and needs a nominal path. --no-configuration skips any
# project or user air.toml, so formatting does not depend on the host.
STDIN_PATH = "/nonexistent/sepalith-reward-v21/canonical.R"

LOGGER = logging.getLogger("sepalith.training.rl.air_canonical")


class AirError(RuntimeError):
    """air is misconfigured or produced no usable verdict."""


class AirInfrastructureError(AirError):
    """air did not finish normally. The text was not judged."""


def normalize_text(text: str) -> str:
    """Normalize line endings, strip trailing whitespace and final blank lines."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stripped = [line.rstrip() for line in lines]
    while stripped and stripped[-1] == "":
        stripped.pop()
    return "\n".join(stripped)


def normalize_lines(lines: Sequence[str]) -> tuple[str, ...]:
    """Line-level form of ``normalize_text`` for region bodies."""
    return tuple(normalize_text("\n".join(lines)).split("\n")) if lines else ()


class AirFormatter:
    def __init__(
        self,
        command: Sequence[str] = ("air",),
        *,
        timeout_seconds: float = 20.0,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise AirError("air_timeout_out_of_bounds")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise AirError("air_max_attempts_out_of_bounds")
        self.command = tuple(command)
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max_attempts
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.infrastructure_failures = 0
        self._lock = threading.Lock()
        self._version: str | None = None

    def _run(self, arguments: Sequence[str], stdin: str | None, label: str) -> subprocess.CompletedProcess[str]:
        last = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = subprocess.run(
                    [*self.command, *arguments],
                    input=stdin if stdin is not None else "",
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                    env={**os.environ, "NO_COLOR": "1"},
                )
            except subprocess.TimeoutExpired:
                last = f"timeout_after_{self.timeout_seconds:g}s"
            except (OSError, UnicodeError) as error:
                last = f"error:{type(error).__name__}"
            else:
                if result.returncode == 0:
                    return result
                if result.returncode == AIR_SYNTAX_EXIT and AIR_SYNTAX_MESSAGE in result.stderr:
                    return result
                last = (f"signal_{-result.returncode}" if result.returncode < 0
                        else f"exit_{result.returncode}")
            with self._lock:
                self.infrastructure_failures += 1
            LOGGER.warning("air %s infrastructure failure (%s), attempt %d of %d",
                           label, last, attempt, self.max_attempts)
            if attempt < self.max_attempts:
                time.sleep(self.retry_delay_seconds)
        raise AirInfrastructureError(f"air_infrastructure_failure:{label}:{last}")

    def version(self) -> str:
        if self._version is None:
            result = self._run(["--version"], None, "version")
            text = result.stdout.strip()
            if result.returncode != 0 or not text.startswith("air "):
                raise AirInfrastructureError(f"air_version_unusable:{text[:80]!r}")
            self._version = text
        return self._version

    def format(self, text: str) -> str | None:
        """Return air's formatting of ``text``, or None if air cannot parse it."""
        if not isinstance(text, str):
            raise AirError("air_input_not_text")
        result = self._run(
            ["format", "--no-configuration", "--stdin-file-path", STDIN_PATH], text, "format",
        )
        return result.stdout if result.returncode == 0 else None
