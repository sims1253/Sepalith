#!/usr/bin/env python3
"""Bounded, credential-free staging of the pinned Midtrain base artifact."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

URL = "https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain/resolve/8dc5f6055b90fe4b9422340810b270b9569f37f3/model.safetensors"
EXPECTED_BYTES = 5_033_557_128
EXPECTED_SHA256 = "38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad"
NATIVE_DIR = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native")
FINAL = NATIVE_DIR / "model.safetensors"
WORK = Path(__file__).resolve().parent
LAUNCH = WORK / "midtrain-download.launch.json"
TERMINAL = WORK / "midtrain-download.terminal.json"
LOG = WORK / "midtrain-download.log"
REPORT = WORK / "midtrain-download.json"
SOURCE_SMALL_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
SMALL_FILES = {
    "config.json": (704, "59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180"),
    "generation_config.json": (213, "9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1"),
    "tokenizer.json": (9_894_271, "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"),
    "tokenizer_config.json": (94_391, "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"),
    "special_tokens_map.json": (551, "82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db"),
    "chat_template.jinja": (9_060, "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    fsync_file(path)


def verify_small_source_files() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha) in SMALL_FILES.items():
        source = SOURCE_SMALL_DIR / name
        if not source.is_file():
            raise SystemExit(f"missing small source file: {source}")
        actual_bytes = source.stat().st_size
        actual_sha = sha256_file(source)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise SystemExit(f"PRE-05 small-file mismatch for {source}: {actual_bytes} {actual_sha}")
        result[name] = {
            "source": str(source),
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "pre05_match": True,
        }
    return result


def copy_small_files(source_records: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    copied: dict[str, dict[str, object]] = {}
    for name, record in source_records.items():
        source = Path(str(record["source"]))
        destination = NATIVE_DIR / name
        if destination.exists():
            actual = (destination.stat().st_size, sha256_file(destination))
            expected = (int(record["bytes"]), str(record["sha256"]))
            if actual != expected:
                raise SystemExit(f"refusing to overwrite mismatched existing small file {destination}")
            copied[name] = {**record, "destination": str(destination), "status": "already_present_verified"}
            continue
        temporary = NATIVE_DIR / f".{name}.part-{os.getpid()}-{time.time_ns()}"
        shutil.copyfile(source, temporary)
        fsync_file(temporary)
        actual = (temporary.stat().st_size, sha256_file(temporary))
        expected = (int(record["bytes"]), str(record["sha256"]))
        if actual != expected:
            raise SystemExit(f"small-file copy mismatch for {name}: {actual}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise SystemExit(f"refusing to overwrite concurrent small file {destination}")
        temporary.unlink()
        copied[name] = {**record, "destination": str(destination), "status": "copied_and_verified"}
    return copied


def download_weight() -> dict[str, object]:
    if FINAL.exists():
        size = FINAL.stat().st_size
        digest = sha256_file(FINAL) if size == EXPECTED_BYTES else None
        if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
            raise SystemExit(f"refusing to overwrite existing mismatched weight {FINAL}: {size} {digest}")
        return {
            "status": "already_present_verified",
            "path": str(FINAL),
            "bytes": size,
            "sha256": digest,
            "elapsed_seconds": 0.0,
            "stream_limit_bytes": EXPECTED_BYTES + 1,
            "no_overwrite": True,
        }

    temporary = NATIVE_DIR / f".{FINAL.name}.part-{os.getpid()}-{time.time_ns()}"
    command = [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=30s", "420s",
        "curl", "--fail", "--location", "--retry", "2", "--retry-all-errors",
        "--connect-timeout", "15", "--max-time", "420", "--max-filesize", str(EXPECTED_BYTES + 1),
        "--silent", "--show-error", URL,
    ]
    start = time.monotonic()
    launch_record = {
        "url": URL,
        "command": command,
        "temporary_path": str(temporary),
        "expected_bytes": EXPECTED_BYTES,
        "stream_limit_bytes": EXPECTED_BYTES + 1,
        "started_monotonic": start,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "credential_policy": "No Authorization, cookie, or custom request headers supplied; stderr log captures curl errors only.",
    }
    write_json(LAUNCH, launch_record)
    byte_count = 0
    next_report = start + 10.0
    status = "failed"
    return_code: int | None = None
    with LOG.open("w", encoding="utf-8") as log:
        log.write(json.dumps(launch_record, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, env=env)
        assert process.stdout is not None
        with temporary.open("wb") as stream:
            while True:
                chunk = process.stdout.read(8 * 1024 * 1024)
                now = time.monotonic()
                if chunk:
                    byte_count += len(chunk)
                    if byte_count > EXPECTED_BYTES + 1:
                        process.terminate()
                        raise SystemExit(f"stream exceeded limit at {byte_count} bytes; part preserved at {temporary}")
                    stream.write(chunk)
                if now >= next_report:
                    elapsed = max(now - start, 1e-9)
                    print(f"progress bytes={byte_count} elapsed_seconds={elapsed:.1f} rate_mib_s={byte_count / elapsed / 1048576:.2f}", flush=True)
                    next_report = now + 10.0
                if not chunk:
                    break
            stream.flush()
            os.fsync(stream.fileno())
        return_code = process.wait()
        elapsed = time.monotonic() - start
        log.write(json.dumps({"exit_code": return_code, "bytes_received": byte_count, "elapsed_seconds": elapsed}, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
    terminal = {
        "exit_code": return_code,
        "bytes_received": byte_count,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "temporary_path": str(temporary),
        "stream_limit_bytes": EXPECTED_BYTES + 1,
    }
    write_json(TERMINAL, terminal)
    if return_code != 0:
        raise SystemExit(f"bounded curl failed with exit code {return_code}; part preserved at {temporary}")
    size = temporary.stat().st_size
    digest = sha256_file(temporary)
    if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise SystemExit(f"weight identity mismatch; part preserved at {temporary}: {size} {digest}")
    fsync_file(temporary)
    try:
        os.link(temporary, FINAL)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite concurrently-created {FINAL}; part preserved at {temporary}")
    temporary.unlink()
    fsync_dir(NATIVE_DIR)
    status = "downloaded_and_atomically_promoted"
    return {
        "status": status,
        "path": str(FINAL),
        "bytes": size,
        "sha256": digest,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "transfer_rate_mib_s": round(size / max(time.monotonic() - start, 1e-9) / 1048576, 3),
        "curl_exit_code": return_code,
        "stream_limit_bytes": EXPECTED_BYTES + 1,
        "no_overwrite": True,
        "fsync_file": True,
        "fsync_directory": True,
        "temporary_part_removed": True,
    }


def main() -> int:
    started = time.time()
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    source_records = verify_small_source_files()
    weight = download_weight()
    copied = copy_small_files(source_records)
    fsync_dir(NATIVE_DIR)
    report = {
        "schema_version": "sepalith.pre03.native-midtrain-staging.v1",
        "status": "complete" if weight["status"] in {"downloaded_and_atomically_promoted", "already_present_verified"} else "partial",
        "url": URL,
        "revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
        "native_directory": str(NATIVE_DIR),
        "weight": weight,
        "small_files": copied,
        "pre05_source_small_files": source_records,
        "elapsed_seconds_total": round(time.time() - started, 3),
        "cpu_policy": {"threads": 1, "cuda_visible_devices": "", "ssh": False, "cloud": False, "model_load": False},
    }
    write_json(REPORT, report)
    print(json.dumps({"status": report["status"], "weight": weight, "report": str(REPORT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
