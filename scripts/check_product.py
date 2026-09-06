#!/usr/bin/env python3
"""Check product code with installed npm dependencies, fake assets and tiny child processes."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    extension = root / 'extensions/vscode-sepalith'
    if not (extension / 'node_modules').is_dir():
        print('Install editor dependencies first: npm --prefix extensions/vscode-sepalith ci', file=sys.stderr)
        return 2
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    npm = 'npm.cmd' if os.name == 'nt' else 'npm'
    commands = [[npm, '--prefix', str(extension), 'run', name]
                for name in ('compile', 'bundle', 'check-context', 'check-runtime', 'check-process')]
    commands.extend([
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'scripts/packaging', '-p', 'test_*.py', '-v'],
        [sys.executable, 'experiments/training/test_quant_export.py'],
    ])
    for command in commands:
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
