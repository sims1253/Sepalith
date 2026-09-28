#!/usr/bin/env python3
"""Pinned R parse-only probe with separate syntax and infrastructure outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence


EXPECTED_BUFFER_PARSER_IDENTITY = {
    "command": "Rscript --vanilla parse_only.R",
    "generated_r_executed": False,
    "operation": "base::parse(file=...,keep.source=FALSE)",
    "r_version": "R 4.6.1 (2026-06-24)",
}


class ParseProbeError(RuntimeError):
    """The parser service failed; callers must not convert this to a reward."""


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RParseOnlyProbe:
    def __init__(
        self,
        harness: Path,
        expected_sha256: str,
        *,
        timeout_seconds: float = 5.0,
        command: Sequence[str] = ("Rscript", "--vanilla"),
        expected_identity: Mapping[str, Any] = EXPECTED_BUFFER_PARSER_IDENTITY,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if sha_file(harness) != expected_sha256:
            raise ParseProbeError("parse_harness_hash_mismatch")
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ParseProbeError("parse_timeout_out_of_bounds")
        if tuple(command) != ("Rscript", "--vanilla"):
            raise ParseProbeError("parse_command_identity_mismatch")
        if dict(expected_identity) != EXPECTED_BUFFER_PARSER_IDENTITY:
            raise ParseProbeError("buffer_parser_identity_mismatch")
        executable = shutil.which(command[0])
        if executable is None:
            raise ParseProbeError("parse_executable_missing")
        self.harness = harness.resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.command = (executable, "--vanilla")
        self._runner = runner
        try:
            version = runner(
                [*self.command, "-e", (
                    'cat(paste("R", getRversion(), '
                    'sprintf("(%s-%s-%s)", R.version$year, R.version$month, R.version$day)))'
                )],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ParseProbeError(f"parse_version_probe_failed:{type(error).__name__}") from error
        if version.returncode != 0:
            raise ParseProbeError(f"parse_version_probe_exit:{version.returncode}")
        if version.stdout != EXPECTED_BUFFER_PARSER_IDENTITY["r_version"]:
            raise ParseProbeError("parse_r_version_mismatch")
        self.identity = dict(EXPECTED_BUFFER_PARSER_IDENTITY)
        self.identity_sha256 = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def assert_buffer_identity(self, value: object) -> None:
        if not isinstance(value, Mapping) or dict(value) != self.identity:
            raise ParseProbeError("row_parser_identity_mismatch")

    def __call__(self, text: str) -> bool:
        if not isinstance(text, str):
            raise ParseProbeError("parse_input_not_text")
        with tempfile.NamedTemporaryFile("w", suffix=".R", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            try:
                result = self._runner(
                    [*self.command, str(self.harness), stream.name],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ParseProbeError(f"parse_probe_failed:{type(error).__name__}") from error
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        if result.returncode < 0:
            raise ParseProbeError(f"parse_probe_signal:{-result.returncode}")
        raise ParseProbeError(f"parse_probe_infrastructure_exit:{result.returncode}")
