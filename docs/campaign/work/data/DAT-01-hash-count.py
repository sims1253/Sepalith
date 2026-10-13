#!/usr/bin/env python3
"""Recompute the DAT-01 source hashes and JSONL line counts.

The input manifest contains an explicit, shallow list of audited paths.  This
script reads those paths only, uses at most two CPU workers, and writes a new
manifest outside the repository when an output path is supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path


def one(entry: dict[str, object]) -> dict[str, object]:
    path = Path(str(entry["path"]))
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
            rows += block.count(b"\n")
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/mnt/e/sepalith/campaign-20260915/data-work/DAT-01-source-hashes.json",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    source = json.loads(manifest_path.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(one, source["files"]))
    result = {
        "algorithm": "sha256",
        "thread_cap": 2,
        "raw_inputs_read_only": True,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": sorted(files, key=lambda item: str(item["path"])),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
