#!/usr/bin/env python3
"""Fixed-harness R parse probe. Candidate text is parsed and never executed."""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence


class ParseProbeError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RParseOnlyProbe:
    def __init__(self, harness: Path, expected_sha256: str, *, timeout_seconds: float = 5.0,
                 command: Sequence[str] = ("Rscript", "--vanilla")) -> None:
        if sha_file(harness) != expected_sha256:
            raise ParseProbeError("parse_harness_hash_mismatch")
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ParseProbeError("parse_timeout_out_of_bounds")
        self.harness = harness.resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.command = tuple(command)

    def __call__(self, text: str) -> bool:
        if not isinstance(text, str):
            raise ParseProbeError("parse_input_not_text")
        with tempfile.NamedTemporaryFile("w", suffix=".R", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            try:
                result = subprocess.run(
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
        return result.returncode == 0
