#!/usr/bin/env python3
"""Run one admitted foreground training command under GNU timeout.

The existing experiment runner owns dispatch and the process group. This wrapper
adds a deadline monitor that remains alive if the lead agent disconnects. The
training callback saves before the soft deadline; timeout terminates a stalled
process group, with a final kill scheduled no later than the hard deadline.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from campaign_checkpoint import digest, write_json
from campaign_control import utc_deadline

TIMEOUT = Path("/usr/bin/timeout")


def supervised_command(recipe, command, *, now=None):
    now = time.time() if now is None else now
    duration, grace = recipe["max_attempt_seconds"], recipe["termination_grace_seconds"]
    reserve = recipe["checkpoint_reserve_seconds"]
    if any(not math.isfinite(value) or value <= 0 for value in (duration, grace, reserve)):
        raise ValueError("Attempt duration, termination grace and checkpoint reserve must be positive and finite")
    hard = min(utc_deadline(recipe["deadline"]), now + duration)
    soft = hard - grace
    if soft - now <= reserve:
        raise ValueError("No attempt budget remains after termination grace and checkpoint reserve")
    if not TIMEOUT.is_file():
        raise ValueError("The pinned GNU timeout executable is absent")
    argv = [str(TIMEOUT), "--signal=TERM", f"--kill-after={grace:.6f}s", f"{soft - now:.6f}s", *command]
    return argv, {
        "soft_deadline": datetime.fromtimestamp(soft, timezone.utc).isoformat(),
        "hard_deadline": datetime.fromtimestamp(hard, timezone.utc).isoformat(),
        "timeout_binary_sha256": digest(TIMEOUT),
        "termination_grace_seconds": grace,
        "limitation": "OS scheduling and uninterruptible kernel I/O can delay process exit; inspect live ownership before release.",
    }


def entrypoint_for_stage(stage):
    """Select the explicit stage adapter without loading a model or CUDA."""
    if stage == "task_sft_prm03_v1":
        return Path(__file__).with_name("campaign_task_sft.py")
    if stage == "cpt_raw_r_v1":
        return Path(__file__).with_name("campaign_cpt.py")
    return Path(__file__).with_name("campaign_sft.py")


def training_entrypoint(recipe):
    """Select the admitted stage entrypoint before constructing argv."""
    stage = recipe.get("stage") or recipe.get("identity", {}).get("policy", {}).get("stage")
    return entrypoint_for_stage(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("Each launch requires a new supervision receipt")
    recipe = json.loads(args.recipe.read_text())
    entry = training_entrypoint(recipe)
    command = [sys.executable, str(entry), str(args.recipe.resolve())]
    argv, supervision = supervised_command(recipe, command)
    write_json(args.receipt, {
        "status": "supervisor_configured; command acceptance and completion unverified",
        "pid": os.getpid(), "recipe_sha256": digest(args.recipe), "argv": argv, **supervision,
    })
    env = dict(os.environ, SEPALITH_CAMPAIGN_SOFT_DEADLINE=supervision["soft_deadline"],
               SEPALITH_CAMPAIGN_HARD_DEADLINE=supervision["hard_deadline"])
    os.execve(TIMEOUT, argv, env)


if __name__ == "__main__":
    main()
