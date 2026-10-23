#!/usr/bin/env python3
"""Night-queue job files for the C15 smoke and pilot (written, never enqueued here).

``C15-grpo-smoke`` (not resumable, about 30 minutes) proves the real CUDA path:
a few updates, one save and resume, one DEV75 evaluation. ``C15-grpo-pilot-1``
(resumable) is the nightly pilot from arm A 536 and requires the smoke.
Enqueue with ``python3 -m sepalith.ops.night_queue enqueue --file <job>.json``.

    python -m sepalith.training.grpo.jobs --output docs/training/plan-20260927/c15/jobs
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from sepalith.training.grpo.recipe import DEPLOY_ROOT, write_new

Json = dict[str, object]
PYTHON = ".venv-sft/bin/python"
BOUND = "docs/training/plan-20260927/c15/bound"
SMOKE_RECIPE = f"{BOUND}/grpo-smoke-arm-a-536.json"
PILOT_RECIPE = f"{BOUND}/grpo-pilot-1-arm-a-536.json"


def command(root: Path, *args: str) -> list[str]:
    return ["/usr/bin/env", f"PYTHONPATH={root}/packages/sepalith/src", "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=0",
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8",
            "TOKENIZERS_PARALLELISM=false", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            str(root / PYTHON), "-B", "-m", "sepalith.training.grpo.night", *args]


def jobs(root: Path = DEPLOY_ROOT) -> dict[str, Json]:
    return {
        "C15-grpo-smoke": {
            "id": "C15-grpo-smoke", "cwd": str(root), "env": [], "priority": 40, "resumable": False,
            "estimated_minutes": 40, "requires": [],
            "command": command(root, "smoke", "--recipe", str(root / SMOKE_RECIPE)),
            "note": ("C15 smoke on the real CUDA path (cuda:0): GRPO from arm A 536 on the C14 pool, train to update 4 and save, "
                     "DEV75 on checkpoint 4, resume to update 6 and save; checks telemetry, resume and the evaluator result. "
                     "Exit 0 pass, 1 fail (nothing done is a failure). Outputs: /mnt/e/.../checkpoints/C15-grpo-smoke/smoke-receipt.json"),
        },
        "C15-grpo-pilot-1": {
            "id": "C15-grpo-pilot-1", "cwd": str(root), "env": [], "priority": 45, "resumable": True,
            "estimated_minutes": 960, "requires": ["C15-grpo-smoke"],
            "command": command(root, "run", "--recipe", str(root / PILOT_RECIPE)),
            "note": ("C15 GRPO pilot from arm A 536 (R4 M >= 15% gate overridden by the user on 2026-10-23): DEV75 every 50 "
                     "updates, DEV250 every 150, R6 decides (stop-for-user exits 3: write override-step-<n>.json in the evidence "
                     "directory and retry; rollback ends the pilot), 1,500 updates at most; resumable at rollout-buffer "
                     "boundaries (exit 75 at the deadline). Re-enqueue is automatic while it exits 75; R8 publishing is morning work."),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT)
    args = parser.parse_args(argv)
    written = []
    for identifier, job in jobs(args.deploy_root).items():
        write_new(args.output / f"{identifier}.json", job)
        written.append(identifier)
    print(json.dumps(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
