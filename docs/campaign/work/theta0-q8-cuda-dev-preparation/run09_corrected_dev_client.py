#!/usr/bin/env python3
"""Run the pinned RUN-09 scorer against the versioned corrected DEV panel.

The accepted scorer predates the six-row finish-target correction and keeps the
original panel digest as a module constant.  This adapter makes the correction
explicit: it verifies the new panel digest and the unchanged scorer digest,
then changes only the scorer's expected panel digest before calling its normal
CLI.  It does not change scoring, rendering, tokenization, or transport.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Sequence


REFERENCE_SCORER = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/quant-quality/run09_native_dev_quality.py"
)
REFERENCE_SCORER_SHA256 = (
    "e574a9453066b5fa5a8e33c02777e8bdd875e208d4c3c51c00e5ede13bba5362"
)
CORRECTED_PANEL = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
)
CORRECTED_PANEL_SHA256 = (
    "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
)
CORRECTED_PANEL_ROWS = 75


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _load_reference():
    if sha256_file(REFERENCE_SCORER) != REFERENCE_SCORER_SHA256:
        raise RuntimeError("pinned RUN-09 scorer source hash mismatch")
    if sha256_file(CORRECTED_PANEL) != CORRECTED_PANEL_SHA256:
        raise RuntimeError("corrected DEV panel hash mismatch")
    if _panel_rows(CORRECTED_PANEL) != CORRECTED_PANEL_ROWS:
        raise RuntimeError("corrected DEV panel row count mismatch")
    spec = importlib.util.spec_from_file_location(
        "sepalith_run09_native_dev_quality_reference", REFERENCE_SCORER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned RUN-09 scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This is the sole intentional adaptation.  The reference scorer still
    # performs its complete panel/schema/context/tokenization/quality checks.
    module.PANEL_SHA256 = CORRECTED_PANEL_SHA256
    module.PANEL_PATH = CORRECTED_PANEL
    module.PANEL_ROWS = CORRECTED_PANEL_ROWS
    return module


def _panel_arg(argv: Sequence[str]) -> Path | None:
    for index, value in enumerate(argv):
        if value == "--panel" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
    return None


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    supplied = _panel_arg(forwarded)
    if supplied is not None and supplied != CORRECTED_PANEL.resolve():
        print(json.dumps({
            "status": "failed",
            "error": "this adapter accepts only the pinned corrected DEV panel",
        }, sort_keys=True))
        return 1
    if supplied is None:
        forwarded.extend(["--panel", str(CORRECTED_PANEL)])
    try:
        module = _load_reference()
        return int(module.main(forwarded))
    except Exception as error:
        print(json.dumps({
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
