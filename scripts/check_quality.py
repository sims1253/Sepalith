#!/usr/bin/env python3
"""Run pinned Python checks without installing training dependencies."""
import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", action="store_true", help="Audit experiments and scripts instead of the package")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = ["uv", "run", "--project", str(root / "tools/quality"), "--locked"]
    commands = [["ruff", "check"], ["ty", "check", "--error-on-warning"]]
    if args.research:
        # Override default discovery; keep the same lint and type rules.
        commands = [
            ["ruff", "check", "--config", 'include = ["*.py"]', "experiments", "scripts"],
            ["ty", "check", "--error-on-warning", "--config",
             'src.include = ["experiments", "scripts"]', "experiments", "scripts"],
        ]
    failed = False
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(prefix + command, cwd=root, check=False)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
