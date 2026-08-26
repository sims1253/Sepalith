"""pytest bootstrap: repo root on sys.path so the namespace package
`experiments.training.poc_diff` (and its poc_twin imports) resolves when
pytest is invoked from anywhere."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
