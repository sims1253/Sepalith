#!/usr/bin/env python3
"""Freeze explicit local inputs without changing or publishing source evidence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    p = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else None


def freeze(source, paths, destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    head = git(source, "rev-parse", "HEAD")
    for path in sorted(set(paths)):
        path = Path(path).absolute()
        record = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"Input changed while reading: {path}")
            digest = sha(data)
            target = destination / digest
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Snapshot collision")
            target.write_bytes(data)
            record.update(sha256=digest, bytes=len(data), mtime_ns=after.st_mtime_ns)
            try:
                relative = str(path.relative_to(source))
                record.update(
                    source_head=head,
                    head_blob=git(source, "rev-parse", f"{head}:{relative}"),
                    status=git(
                        source, "status", "--short", "--ignored", "--", relative
                    ),
                )
            except ValueError:
                record["source_head"] = None
        records.append(record)
    return {
        "schema": 1,
        "source_root": str(source),
        "source_head": head,
        "source_status": git(source, "status", "--porcelain"),
        "inputs": records,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--paths", type=Path, required=True, help="JSON array of absolute paths"
    )
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = freeze(a.source.resolve(), json.loads(a.paths.read_text()), a.snapshot)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
