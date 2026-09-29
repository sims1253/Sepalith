#!/usr/bin/env python3
"""Run the full-weight training runtime tests on CPU with the training interpreter.

These tests need torch, transformers, trl and unsloth plus the campaign data on
/mnt/e, so they are separate from the stdlib-only core suite.
"""
from pathlib import Path
import os
import subprocess
import sys

DEFAULT_PYTHON = "/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = os.environ.get("SEPALITH_TRAINING_PYTHON", DEFAULT_PYTHON)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(root / "packages/sepalith/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
        "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "8"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    print(f"Checking the training runtime on CPU with {python}.", flush=True)
    return subprocess.run(
        ["nice", "-n", "10", python, "-B", "-m", "unittest", "discover", "-s",
         str(root / "packages/sepalith/tests/training"), "-p", "test_*.py", "-v", *sys.argv[1:]],
        cwd=root, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
