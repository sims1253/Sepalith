#!/usr/bin/env python3
"""Replay all numeric analyses from frozen inputs; no model or network calls."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory",
        "snapshot",
        "cache-inventory",
        "cache-snapshot",
        "annotations",
        "out",
        "private-out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.private_out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    env = os.environ | {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    def run(script, *args):
        subprocess.run(
            [sys.executable, str(here / script), *map(str, args)], env=env, check=True
        )

    common = ["--inventory", a.inventory, "--snapshot", a.snapshot]
    doc = a.private_out / "doc"
    run("doc_sync.py", *common, "--out", doc)
    run(
        "doc_score.py",
        "--packets",
        doc / "packets.json",
        "--mapping",
        doc / "unblind.json",
        "--annotations",
        a.annotations,
        "--out",
        a.out / "doc-sync-results.json",
    )
    run(
        "b8b.py",
        *common,
        "--out",
        a.out / "b8b-results.json",
        "--private-out",
        a.private_out / "b8b-discordant.json",
    )
    run(
        "loc1.py",
        *common,
        "--cache-inventory",
        a.cache_inventory,
        "--cache-snapshot",
        a.cache_snapshot,
        "--out",
        a.out / "loc1-results.json",
        "--private-out",
        a.private_out / "loc1-rankings.json",
    )
    run(
        "uncertainty.py",
        "--b8b",
        a.out / "b8b-results.json",
        "--loc1",
        a.out / "loc1-results.json",
        "--out",
        a.out / "uncertainty.json",
    )


if __name__ == "__main__":
    main()
