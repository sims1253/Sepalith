#!/usr/bin/env python3
"""Bounded 256 MiB C-drive sequential I/O probe with owned cleanup."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

SIZE = 256 * 1024 * 1024
CHUNK_SIZE = 4 * 1024 * 1024
CHUNKS = SIZE // CHUNK_SIZE
SEED = b"sepalith-sft11-nvme-stage-probe-v1"


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def memory_snapshot():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
            key, rest = line.split(":", 1); values[key] = int(rest.split()[0]) * 1024
    return values


def process_snapshot(pid: int):
    root = Path("/proc") / str(pid)
    value = {"pid": pid, "exists": root.is_dir()}
    if not root.is_dir(): return value
    status = {}
    for line in (root / "status").read_text().splitlines():
        if line.startswith(("Name:", "State:", "VmRSS:")):
            key, rest = line.split(":", 1); status[key] = rest.strip()
    value["status"] = status
    if (root / "io").is_file():
        io = {}
        for line in (root / "io").read_text().splitlines():
            key, rest = line.split(":", 1); io[key] = int(rest)
        value["io"] = io
    return value


def expected_hash(chunk: bytes) -> str:
    digest = hashlib.sha256()
    for _ in range(CHUNKS): digest.update(chunk)
    return digest.hexdigest()


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0: raise OSError("short probe write")
        view = view[count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contended-pid", type=int, default=1374952)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit("fresh output required")
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / f"nvme-stage-{uuid.uuid4().hex}.probe"
    chunk = hashlib.shake_256(SEED).digest(CHUNK_SIZE)
    expected = expected_hash(chunk)
    before = shutil.disk_usage(args.directory)
    value = {
        "schema": "sepalith.sft11.nvme-stage-256mib-probe.v1",
        "started_at": now(), "directory": str(args.directory), "path": str(path),
        "bytes": SIZE, "chunk_bytes": CHUNK_SIZE, "chunks": CHUNKS,
        "expected_sha256": expected, "free_bytes_before": before.free,
        "memory_before": memory_snapshot(), "contended_process_before": process_snapshot(args.contended_pid),
        "targeted_cache_release": {"attempted": False, "supported": False},
    }
    fd = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        started = time.monotonic_ns()
        for _ in range(CHUNKS): write_all(fd, chunk)
        write_done = time.monotonic_ns()
        os.fsync(fd)
        fsync_done = time.monotonic_ns()
        value["write_seconds"] = (write_done - started) / 1e9
        value["fsync_seconds"] = (fsync_done - write_done) / 1e9
        value["write_plus_fsync_seconds"] = (fsync_done - started) / 1e9
        value["write_plus_fsync_mib_per_second"] = 256 / value["write_plus_fsync_seconds"]
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            value["targeted_cache_release"]["attempted"] = True
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                value["targeted_cache_release"]["supported"] = True
            except OSError as error:
                value["targeted_cache_release"]["error"] = f"{error.errno}:{error.strerror}"
        os.close(fd); fd = None
        digest = hashlib.sha256(); count = 0
        read_started = time.monotonic_ns()
        with path.open("rb", buffering=0) as stream:
            while block := stream.read(CHUNK_SIZE):
                digest.update(block); count += len(block)
        read_done = time.monotonic_ns()
        value["read_hash_seconds"] = (read_done - read_started) / 1e9
        value["read_hash_mib_per_second"] = 256 / value["read_hash_seconds"]
        value["actual_bytes"] = count
        value["actual_sha256"] = digest.hexdigest()
        value["verification"] = "pass" if count == SIZE and digest.hexdigest() == expected else "fail"
        if value["verification"] != "pass": raise ValueError("probe byte/hash verification failed")
    finally:
        if fd is not None: os.close(fd)
        if path.exists(): path.unlink()
        value["cleanup"] = {"path_absent": not path.exists(), "removed_only_owned_unique_file": True}
        value["free_bytes_after"] = shutil.disk_usage(args.directory).free
        value["memory_after"] = memory_snapshot()
        value["contended_process_after"] = process_snapshot(args.contended_pid)
        value["ended_at"] = now()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
