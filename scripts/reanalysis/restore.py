#!/usr/bin/env python3
"""Restore private snapshots only when live input bytes match recorded hashes."""

import argparse
import hashlib
import json
from pathlib import Path


def restore(inventory, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for r in json.loads(Path(inventory).read_text())["inputs"]:
        if not r["exists"]:
            continue
        target = destination / r["sha256"]
        data = target.read_bytes() if target.exists() else Path(r["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise ValueError(
                "Evidence changed; obtain the recorded version: " + r["path"]
            )
        if not target.exists():
            target.write_bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True)
    p.add_argument("--snapshot", required=True)
    a = p.parse_args()
    restore(a.inventory, a.snapshot)


if __name__ == "__main__":
    main()
