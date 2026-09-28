#!/usr/bin/env python3
"""Resume the preserved Midtrain part with an exact HTTP Range, then admit it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

URL = "https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain/resolve/8dc5f6055b90fe4b9422340810b270b9569f37f3/model.safetensors"
REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
EXPECTED_BYTES = 5_033_557_128
EXPECTED_SHA256 = "38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad"
NATIVE_DIR = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native")
PART = NATIVE_DIR / ".model.safetensors.part-3316664-1789231351948352402"
FINAL = NATIVE_DIR / "model.safetensors"
WORK = Path(__file__).resolve().parent
LAUNCH = WORK / "midtrain-resume.launch.json"
HEADERS = WORK / "midtrain-resume.headers"
LOG = WORK / "midtrain-resume.log"
TERMINAL = WORK / "midtrain-resume.terminal.json"
REPORT = WORK / "midtrain-resume.json"
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
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    fsync_file(path)


def parse_range_headers(path: Path, offset: int, end: int) -> dict[str, object]:
    text = path.read_text(encoding="latin-1")
    blocks = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("HTTP/") and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    parsed = []
    for block in blocks:
        status = block[0] if block else ""
        headers: dict[str, str] = {}
        for line in block[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        parsed.append({"status": status, "headers": headers})
    range_blocks = [item for item in parsed if item["headers"].get("content-range")]
    if not range_blocks:
        raise SystemExit(f"HTTP response did not contain Content-Range; headers preserved at {path}")
    selected = range_blocks[-1]
    expected_range = f"bytes {offset}-{end}/{EXPECTED_BYTES}"
    content_range = selected["headers"].get("content-range")
    content_length = selected["headers"].get("content-length")
    remaining = end - offset + 1
    if selected["status"].split()[1:2] != ["206"]:
        raise SystemExit(f"range response status is not 206: {selected['status']}")
    if content_range != expected_range:
        raise SystemExit(f"wrong Content-Range {content_range!r}, expected {expected_range!r}")
    if content_length != str(remaining):
        raise SystemExit(f"wrong Content-Length {content_length!r}, expected {remaining}")
    return {
        "selected_status": selected["status"],
        "content_range": content_range,
        "content_length": int(content_length),
        "expected_content_range": expected_range,
        "redirect_or_prior_blocks": len(parsed) - 1,
        "raw_headers_sha256": sha256_file(path),
    }


def verify_small_sources() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha) in SMALL_FILES.items():
        source = SOURCE_SMALL_DIR / name
        actual_bytes = source.stat().st_size
        actual_sha = sha256_file(source)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise SystemExit(f"PRE-05 small-file mismatch for {source}: {actual_bytes} {actual_sha}")
        records[name] = {"source": str(source), "bytes": actual_bytes, "sha256": actual_sha, "pre05_match": True}
    return records


def copy_small_files(records: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for name, record in records.items():
        destination = NATIVE_DIR / name
        expected = (int(record["bytes"]), str(record["sha256"]))
        if destination.exists():
            actual = (destination.stat().st_size, sha256_file(destination))
            if actual != expected:
                raise SystemExit(f"refusing to overwrite mismatched companion {destination}: {actual}")
            output[name] = {**record, "destination": str(destination), "status": "already_present_verified"}
            continue
        temporary = NATIVE_DIR / f".{name}.part-{os.getpid()}-{time.time_ns()}"
        shutil.copyfile(Path(str(record["source"])), temporary)
        fsync_file(temporary)
        actual = (temporary.stat().st_size, sha256_file(temporary))
        if actual != expected:
            raise SystemExit(f"companion identity mismatch for {name}: {actual}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise SystemExit(f"refusing to overwrite concurrent companion {destination}")
        temporary.unlink()
        output[name] = {**record, "destination": str(destination), "status": "copied_and_verified"}
    return output


def main() -> int:
    started = time.monotonic()
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    if FINAL.exists():
        size = FINAL.stat().st_size
        digest = sha256_file(FINAL) if size == EXPECTED_BYTES else None
        if (size, digest) != (EXPECTED_BYTES, EXPECTED_SHA256):
            raise SystemExit(f"refusing to overwrite mismatched existing final {FINAL}: {size} {digest}")
        small_sources = verify_small_sources()
        copied = copy_small_files(small_sources)
        fsync_dir(NATIVE_DIR)
        report = {"status": "already_present_verified", "final": {"path": str(FINAL), "bytes": size, "sha256": digest}, "small_files": copied}
        write_json(REPORT, report)
        return 0
    if not PART.exists() or PART.stat().st_size != 4_194_304_000:
        raise SystemExit(f"preserved paused part missing or changed: {PART}")
    offset = PART.stat().st_size
    end = EXPECTED_BYTES - 1
    remaining = EXPECTED_BYTES - offset
    range_part = NATIVE_DIR / f".model.safetensors.range-{os.getpid()}-{time.time_ns()}"
    command = [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=20s", "180s",
        "curl", "--fail", "--location", "--retry", "2", "--retry-all-errors",
        "--connect-timeout", "15", "--max-time", "180", "--max-filesize", str(remaining + 1),
        "--range", f"{offset}-{end}", "--dump-header", str(HEADERS), "--silent", "--show-error", URL,
    ]
    launch = {
        "url": URL,
        "revision": REVISION,
        "command": command,
        "range": {"start": offset, "end": end, "expected_total": EXPECTED_BYTES, "expected_suffix_bytes": remaining},
        "preserved_part": str(PART),
        "range_part": str(range_part),
        "environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "credential_policy": "No Authorization, cookie, or custom request headers supplied.",
        "started_monotonic": started,
    }
    write_json(LAUNCH, launch)
    byte_count = 0
    next_report = started + 10.0
    process = None
    try:
        with LOG.open("w", encoding="utf-8") as log:
            log.write(json.dumps(launch, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, env=env)
            assert process.stdout is not None
            with range_part.open("wb") as stream:
                while True:
                    chunk = process.stdout.read(8 * 1024 * 1024)
                    now = time.monotonic()
                    if chunk:
                        byte_count += len(chunk)
                        if byte_count > remaining + 1:
                            process.terminate()
                            raise SystemExit(f"range stream exceeded ceiling at {byte_count}; suffix preserved at {range_part}")
                        stream.write(chunk)
                    if now >= next_report:
                        elapsed = max(now - started, 1e-9)
                        print(f"progress suffix_bytes={byte_count} elapsed_seconds={elapsed:.1f} rate_mib_s={byte_count / elapsed / 1048576:.2f}", flush=True)
                        next_report = now + 10.0
                    if not chunk:
                        break
                stream.flush()
                os.fsync(stream.fileno())
            return_code = process.wait()
            log.write(json.dumps({"exit_code": return_code, "suffix_bytes": byte_count, "elapsed_seconds": time.monotonic() - started}, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        raise
    fsync_file(HEADERS)
    header_record = parse_range_headers(HEADERS, offset, end)
    if return_code != 0:
        raise SystemExit(f"bounded range curl failed with exit code {return_code}; suffix preserved at {range_part}")
    suffix_bytes = range_part.stat().st_size
    suffix_sha = sha256_file(range_part)
    if suffix_bytes != remaining:
        raise SystemExit(f"range suffix size mismatch: {suffix_bytes} != {remaining}; suffix preserved at {range_part}")
    with PART.open("ab") as destination, range_part.open("rb") as source:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    combined_bytes = PART.stat().st_size
    combined_sha = sha256_file(PART)
    if combined_bytes != EXPECTED_BYTES or combined_sha != EXPECTED_SHA256:
        raise SystemExit(f"combined weight identity mismatch; preserved at {PART}: {combined_bytes} {combined_sha}")
    fsync_file(PART)
    try:
        os.link(PART, FINAL)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite concurrently-created final {FINAL}; full part preserved")
    PART.unlink()
    range_part.unlink()
    fsync_dir(NATIVE_DIR)
    small_sources = verify_small_sources()
    copied = copy_small_files(small_sources)
    fsync_dir(NATIVE_DIR)
    elapsed = time.monotonic() - started
    terminal = {
        "status": "complete",
        "curl_exit_code": return_code,
        "range": {"start": offset, "end": end, "suffix_bytes": suffix_bytes, "suffix_sha256": suffix_sha, "headers": header_record},
        "combined": {"path": str(FINAL), "bytes": combined_bytes, "sha256": combined_sha},
        "elapsed_seconds": round(elapsed, 3),
        "rate_mib_s": round(suffix_bytes / max(elapsed, 1e-9) / 1048576, 3),
        "file_fsync": True,
        "directory_fsync": True,
        "exclusive_promotion": True,
        "preserved_part_removed": True,
        "range_part_removed": True,
        "small_files_copied": True,
    }
    write_json(TERMINAL, terminal)
    report = {
        "schema_version": "sepalith.pre03.native-midtrain-staging.resume.v1",
        "status": "complete",
        "source": {"repository": "openbmb/MiniCPM5-2B-Midtrain", "revision": REVISION, "url": URL, "expected_bytes": EXPECTED_BYTES, "expected_sha256": EXPECTED_SHA256},
        "native_directory": str(NATIVE_DIR),
        "weight": terminal["combined"],
        "resume": {"preserved_part_bytes": offset, "range": terminal["range"], "terminal": str(TERMINAL), "log": str(LOG), "launch": str(LAUNCH)},
        "small_files": copied,
        "resource_policy": {"cpu_threads": 1, "cuda_visible_devices": "", "timeout_seconds": 180, "no_ssh": True, "no_cloud": True, "no_model_load": True},
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(REPORT, report)
    print(json.dumps({"status": report["status"], "weight": report["weight"], "small_files": sorted(copied), "report": str(REPORT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
