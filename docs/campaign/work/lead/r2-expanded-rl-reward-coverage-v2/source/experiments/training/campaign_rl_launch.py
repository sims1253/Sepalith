#!/usr/bin/env python3
"""Supervise one admitted RL entry attempt with GNU timeout.

The wrapper performs the complete CPU preflight before creating the supervisor
receipt.  It then runs the actual entry point with a soft deadline for graceful
checkpointing and a hard deadline enforced by GNU timeout's process-group kill.
Each attempt must provide fresh output/archive paths and a fresh receipt.
The supervisor receipt is immutable after configuration; the entry writes its
result or failure to a distinct sibling receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from campaign_checkpoint import digest, write_json
from campaign_control import utc_deadline
from campaign_rl_entry import RLEntryError, preflight_file


TIMEOUT = Path("/usr/bin/timeout")


def entry_result_receipt_path(supervision_receipt: Path) -> Path:
    """Return the distinct entry-result path paired with a supervisor receipt."""
    receipt = Path(supervision_receipt).resolve()
    suffix = receipt.suffix or ".json"
    return receipt.with_name(f"{receipt.stem}.entry-result{suffix}")


def supervised_command(recipe: dict, command: list[str], *, now: float | None = None) -> tuple[list[str], dict[str, Any]]:
    """Build a soft GNU timeout whose kill-after grace reaches the hard cutoff."""
    import math

    now = time.time() if now is None else now
    try:
        duration = float(recipe["max_attempt_seconds"])
        grace = float(recipe["termination_grace_seconds"])
        reserve = float(recipe["checkpoint_reserve_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise RLEntryError("attempt duration, termination grace, and checkpoint reserve are required") from error
    if any(not math.isfinite(value) or value <= 0 for value in (duration, grace, reserve)):
        raise RLEntryError("attempt duration, termination grace and checkpoint reserve must be positive and finite")
    hard = min(utc_deadline(recipe["deadline"]), now + duration)
    soft = hard - grace
    if soft - now <= reserve:
        raise RLEntryError("no attempt budget remains after termination grace and checkpoint reserve")
    if not TIMEOUT.is_file():
        raise RLEntryError("the pinned GNU timeout executable is absent")
    argv = [
        str(TIMEOUT), "--signal=TERM", f"--kill-after={grace:.6f}s",
        f"{soft - now:.6f}s", *command,
    ]
    return argv, {
        "soft_deadline": datetime.fromtimestamp(soft, timezone.utc).isoformat(),
        "hard_deadline": datetime.fromtimestamp(hard, timezone.utc).isoformat(),
        "timeout_binary_sha256": digest(TIMEOUT),
        "termination_grace_seconds": grace,
        "checkpoint_reserve_seconds": reserve,
        "limitation": "OS scheduling and uninterruptible kernel I/O can delay process exit; inspect live ownership before release.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    supervision_receipt = args.receipt.resolve()
    entry_receipt = entry_result_receipt_path(supervision_receipt)
    if supervision_receipt.exists() or supervision_receipt.is_symlink():
        raise RLEntryError(f"each RL launch requires a fresh supervision receipt: {supervision_receipt}")
    if entry_receipt.exists() or entry_receipt.is_symlink():
        raise RLEntryError(f"each RL launch requires a fresh entry-result receipt: {entry_receipt}")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise RLEntryError("RL recipe must be a JSON object")
    # This repeats at most a bounded file/hash read and proves the launcher
    # cannot dispatch a recipe that the entry would reject before model load.
    preflight = preflight_file(args.recipe)
    command = [
        sys.executable,
        str(Path(__file__).with_name("campaign_rl_entry.py").resolve()),
        str(args.recipe.resolve()),
        "--receipt", str(entry_receipt),
    ]
    supervisor_argv, supervision = supervised_command(recipe, command)
    write_json(supervision_receipt, {
        "status": "supervisor_configured; command acceptance and completion unverified",
        "pid": os.getpid(),
        "recipe_sha256": digest(args.recipe),
        "entry_result_receipt": str(entry_receipt),
        "preflight": {
            "status": preflight["status"],
            "parent_manifest_sha256": preflight["parent"]["manifest_sha256"],
            "data_identity": preflight["data"],
            "geometry": preflight["geometry"],
        },
        "argv": supervisor_argv,
        **supervision,
    })
    env = dict(
        os.environ,
        SEPALITH_CAMPAIGN_SOFT_DEADLINE=supervision["soft_deadline"],
        SEPALITH_CAMPAIGN_HARD_DEADLINE=supervision["hard_deadline"],
        SEPALITH_CAMPAIGN_CHECKPOINT_RESERVE_SECONDS=str(supervision["checkpoint_reserve_seconds"]),
        SEPALITH_CAMPAIGN_LAUNCH_RECEIPT=str(supervision_receipt),
    )
    os.execve(TIMEOUT, supervisor_argv, env)
    return 0  # pragma: no cover - os.execve replaces the process


if __name__ == "__main__":
    raise SystemExit(main())
