#!/usr/bin/env python3
"""Run the standard-library core suite without models, NAS data or installations."""
from pathlib import Path
import os
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(root / "packages/sepalith/src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    print("Checking sepalith core and legacy prompt parity (stdlib, local fixtures).", flush=True)
    return subprocess.run(
        [sys.executable, "-B", "-s", "-m", "unittest", "discover", "-s",
         str(root / "packages/sepalith/tests"), "-p", "test_*.py", "-v"],
        cwd=root, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
