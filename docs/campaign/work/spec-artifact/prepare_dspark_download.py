#!/usr/bin/env python3
"""Download and atomically admit the pinned public DSpark GGUF without clobbering."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

URL = (
    "https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF/resolve/"
    """a261d2b4abc9c9ebfbad2af8a817a09802fc4ca3/"""
    "MiniCPM5-2.6B-DSpark.gguf"
)
OUT_DIR = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-bf16")
FINAL = OUT_DIR / "MiniCPM5-2.6B-DSpark.gguf"
EXPECTED_BYTES = 652_730_240
EXPECTED_SHA256 = "57df08640f0534a1aac075d1c8bdacdb2b7e5815da6f4e5cfd39ecac3a3f0c26"
WORK = Path(__file__).resolve().parent
LOG = WORK / "dspark-download-curl.log"
REPORT = WORK / "dspark-download.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    existing = FINAL.exists()
    if existing:
        size = FINAL.stat().st_size
        digest = sha256_file(FINAL) if size == EXPECTED_BYTES else None
        if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
            raise SystemExit(f"refusing to overwrite existing mismatched {FINAL}: {size} {digest}")
        result = {
            "status": "already_present_verified",
            "url": URL,
            "path": str(FINAL),
            "bytes": size,
            "sha256": digest,
            "expected_bytes": EXPECTED_BYTES,
            "expected_sha256": EXPECTED_SHA256,
            "no_overwrite": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        REPORT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return 0

    part = OUT_DIR / f".{FINAL.name}.part-{os.getpid()}-{time.time_ns()}"
    command = [
        "curl", "--fail", "--location", "--retry", "2", "--retry-all-errors",
        "--connect-timeout", "15", "--max-time", "900", "--output", str(part), URL,
    ]
    with LOG.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "url": URL, "part": str(part)}, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        completed = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", "900s", *command],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        log.write(f"\nexit_code={completed.returncode}\n")
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise SystemExit(f"bounded curl failed with exit code {completed.returncode}; part preserved at {part}")
    size = part.stat().st_size
    digest = sha256_file(part)
    fsync_file(part)
    if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise SystemExit(f"download identity mismatch; part preserved at {part}: {size} {digest}")
    # A hard link is an exclusive promotion: it fails rather than replacing a concurrent file.
    try:
        os.link(part, FINAL)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite concurrently-created {FINAL}; part preserved at {part}")
    part.unlink()
    dir_fd = os.open(OUT_DIR, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    result = {
        "status": "downloaded_and_atomically_promoted",
        "url": URL,
        "path": str(FINAL),
        "bytes": size,
        "sha256": digest,
        "expected_bytes": EXPECTED_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "curl_exit_code": completed.returncode,
        "curl_log": str(LOG),
        "no_overwrite": True,
        "temporary_path": str(part),
        "fsync": {"file": True, "directory": True},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    REPORT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
