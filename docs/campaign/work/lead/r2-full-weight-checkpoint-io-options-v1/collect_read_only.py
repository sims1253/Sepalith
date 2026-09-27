#!/usr/bin/env python3
"""Collect checkpoint metadata and topology without opening large checkpoint files."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def small_sha256(path: Path) -> str:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"refusing large read: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disk(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path), "st_dev": os.stat(path).st_dev,
        "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("fresh output required")
    proc = Path("/proc") / str(args.pid)
    proc_io = {}
    if (proc / "io").is_file():
        for line in (proc / "io").read_text().splitlines():
            key, value = line.split(":", 1); proc_io[key] = int(value)
    checkpoints = []
    if args.checkpoint_root.is_dir():
        for directory in sorted(args.checkpoint_root.glob("**/checkpoint-*")):
            if not directory.is_dir(): continue
            files = []
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    stat = path.stat()
                    files.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            checkpoints.append({"path": str(directory), "files": files, "total_bytes": sum(x["bytes"] for x in files)})
    plan = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
    sources = []
    for relative in (
        "docs/campaign/work/lead/r2-full-weight-resume-proof-v3/source/campaign_checkpoint.py",
        "docs/campaign/work/lead/r2-full-weight-resume-proof-v3/source/full_weight_resume_proof.py",
        "docs/campaign/work/lead/r2-full-weight-resume-proof-v4/source/campaign_checkpoint.py",
        "docs/campaign/work/lead/r2-full-weight-resume-proof-v4/source/full_weight_resume_proof.py",
        "docs/campaign/work/lead/r2-full-weight-resume-proof-v4/source-manifest.json",
        "docs/campaign/work/lead/r2-full-weight-resume-proof-v4/recipe.json",
    ):
        path = plan / relative
        sources.append({"path": relative, "bytes": path.stat().st_size, "sha256": small_sha256(path)})
    value = {
        "schema": "sepalith.sft11.checkpoint-io-read-only-evidence.v1",
        "observed_at": now(), "pid": args.pid, "pid_exists": proc.is_dir(),
        "proc_io": proc_io,
        "disks": [disk(Path("/")), disk(Path("/mnt/c")), disk(Path("/mnt/e")), disk(Path("/mnt/f"))],
        "mounts": {
            "/": {"filesystem": "ext4", "backing": "/dev/sdc WSL VHD"},
            "/mnt/c": {"filesystem": "9p/DrvFS over NTFS", "physical": "Samsung SSD 970 EVO Plus 2TB NVMe"},
            "/mnt/e": {"filesystem": "9p/DrvFS over NTFS", "physical": "WDC WD40EZRZ-00GXCB0 4TB SATA"},
            "/mnt/f": {"filesystem": "9p/DrvFS over NTFS", "physical": "SuperSpeedDelight 2TB Storage Spaces; media type not reported"}
        },
        "checkpoints": checkpoints,
        "small_source_bindings": sources,
        "large_files_opened_by_probe": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
