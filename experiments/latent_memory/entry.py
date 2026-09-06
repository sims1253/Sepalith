"""Absolute-path entry point for immutable runner source snapshots."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / 'packages/sepalith/src')]
from experiments.latent_memory.cli import main

if __name__ == '__main__':
    main()
