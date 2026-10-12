#!/usr/bin/env python3
"""Night-queue job files for cards C07 and C08 (written by card C04, not enqueued).

Enqueue one with ``python3 -m sepalith.ops.night_queue enqueue --file <job>.json``.
The commands run the editing-SFT v2 driver (``sepalith.training.sft.night_arm``)
from the deploy checkout with the training interpreter.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

PYTHON = ".venv-sft/bin/python"
PILOTS = (("lr1e-5", 20), ("lr3e-6", 21), ("lr3e-5", 22))  # middle LR first (card C08)


def command(root: Path, *args: str) -> list[str]:
    return ["/usr/bin/env", f"PYTHONPATH={root}/packages/sepalith/src", "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=0",
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", str(root / PYTHON), "-B", "-m", "sepalith.training.sft.night_arm", *args]


def recipe(root: Path, arm: str) -> str:
    return str(root / "docs/training/plan-20260927/c04/bound" / f"sft-v2-{arm}.json")


def jobs(root: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {
        "C07-baseline": {
            "id": "C07-baseline", "cwd": str(root), "env": [], "priority": 10, "resumable": False, "estimated_minutes": 75,
            "requires": [], "command": command(root, "baseline", "--recipe", recipe(root, "lr1e-5")),
            "note": "C07 job 1: DEV75 and DEV250 on parent 11,586 (bf16), the step-0 baseline R1 compares against",
        },
        "C07-sft-smoke": {
            "id": "C07-sft-smoke", "cwd": str(root), "env": [], "priority": 11, "resumable": True, "estimated_minutes": 30,
            "requires": [], "command": command(root, "smoke", "--recipe", recipe(root, "smoke-lr1e-5"),
                                               "--first-stop", "10", "--second-stop", "20"),
            "note": "C07 job 3: LR 1e-5 smoke, save at update 10, resume, stop at 20; writes smoke-receipt.json",
        },
    }
    for arm, priority in PILOTS:
        identifier = f"C08-arm-{arm}"
        out[identifier] = {
            "id": identifier, "cwd": str(root), "env": [], "priority": priority, "resumable": True, "estimated_minutes": 120,
            "requires": ["C07-baseline"],
            "command": command(root, "arm", "--recipe", recipe(root, arm), "--stop-after-gate", "268"),
            "note": f"C08 learning-rate pilot {arm}: train to gate 268, evaluate DEV75 and DEV250, apply R1, stop",
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root", type=Path, default=Path("/home/m0hawk/sandbox/Sepalith"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for identifier, job in jobs(args.deploy_root).items():
        (args.output / f"{identifier}.json").write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(sorted(jobs(args.deploy_root))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
