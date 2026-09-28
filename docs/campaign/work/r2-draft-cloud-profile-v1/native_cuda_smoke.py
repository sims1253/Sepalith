"""Run the pinned real DeepSpec tiny smoke from a relocatable vendor root.

The wrapper keeps the upstream smoke implementation unchanged while replacing
its development-only vendor path with ``SEPALITH_DEEPSPEC_ROOT``.  It does not
load a checkpoint or read campaign rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    ROOT.parent / "r2-draft-upstream-smoke-v1" / "upstream_dspark_smoke.py"
)


def load_upstream_smoke(source_path: str | Path, vendor_root: str | Path):
    source = Path(source_path).resolve()
    vendor = Path(vendor_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"upstream smoke source is missing: {source}")
    if not vendor.is_dir():
        raise FileNotFoundError(f"DeepSpec vendor root is missing: {vendor}")
    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("sepalith_pinned_upstream_smoke", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream smoke source: {source}")
    module = importlib.util.module_from_spec(spec)
    # The source inventory and its helper use this module-level root.  The
    # upstream model, loss, and cache classes remain the pinned real classes.
    module.VENDOR = vendor
    spec.loader.exec_module(module)
    module.VENDOR = vendor
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source", default=os.environ.get("SEPALITH_UPSTREAM_SMOKE_SOURCE", str(DEFAULT_SOURCE)))
    parser.add_argument("--vendor-root", default=os.environ.get("SEPALITH_DEEPSPEC_ROOT"))
    args = parser.parse_args()
    if not str(args.device).startswith("cuda"):
        raise SystemExit("profile smoke requires an explicit CUDA device")
    if not args.vendor_root:
        raise SystemExit("SEPALITH_DEEPSPEC_ROOT is required")
    module = load_upstream_smoke(args.source, args.vendor_root)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("profile smoke requested CUDA but torch.cuda.is_available() is false")
    print(json.dumps(module.run_smoke(device=args.device), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
