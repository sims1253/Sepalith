#!/usr/bin/env python3
"""Remote, framework-free CPU server helper for the v2 notebook capsule.

This file is staged by the root-run local orchestrator. It never imports the
planning host's scorer or tokenizer; those stay on the other side of SSH.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

SCHEMA = "run-09.theta0-notebook-cpu-dev.v2"
PROFILE = {"backend": "cpu", "context": 4096, "quality_cap": 512,
           "threads": 6, "threads_batch": 6, "threads_http": 2, "parallel": 1,
           "batch": 256, "ubatch": 256, "ngl": 0}


class ServerError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ServerError("MemAvailable unavailable")


def children(pid: int) -> list[int]:
    try:
        return [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []


def audit_cpu(pid: int) -> dict[str, Any]:
    maps = []
    fds = []
    try:
        maps = [line for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
                if "libggml-vulkan" in line or "libvulkan" in line]
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("/dev/dri/"):
                fds.append(target)
    except OSError:
        pass
    if maps or fds:
        raise ServerError("CPU server acquired Vulkan mappings or DRI FDs")
    return {"pid": pid, "vulkan_mappings": maps, "dri_fds": fds,
            "backend": "cpu", "ngl": 0, "vulkan_dri_required": False}


def stop_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def server_argv(binary: str, model: str, port: int) -> list[str]:
    return [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port),
            "-t", "6", "-tb", "6", "--threads-http", "2", "--parallel", "1",
            "-c", "4096", "-b", "256", "-ub", "256", "-ngl", "0", "-lv", "4"]


def verify_identity(path: Path, expected_sha: str, binary: Path, expected_binary_sha: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise ServerError("runtime identity hash mismatch")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(identity, dict) or identity.get("binary") != str(binary):
        raise ServerError("runtime identity binary mismatch")
    entries = identity.get("files")
    if not isinstance(entries, list) or not entries:
        raise ServerError("runtime identity lacks dependency entries")
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ServerError("malformed dependency entry")
        dep = Path(item["path"])
        if dep.is_symlink() or not dep.is_file():
            raise ServerError(f"dependency is not a regular file: {dep}")
        if dep.stat().st_size != item.get("bytes") or sha256_file(dep) != item.get("sha256"):
            raise ServerError(f"dependency hash mismatch: {dep}")
    if sha256_file(binary) != expected_binary_sha:
        raise ServerError("CPU binary hash mismatch")
    return {"version": identity.get("version"), "files": len(entries),
            "sha256": expected_sha, "binary_sha256": expected_binary_sha}


def wait_health(process: subprocess.Popen[Any], port: int, deadline: float, floor: int) -> int:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ServerError(f"timeout wrapper exited {process.returncode}")
        if available_mib() < floor:
            raise ServerError("remote memory floor reached during CPU load")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    kids = children(process.pid)
                    if not kids:
                        raise ServerError("healthy wrapper has no server child")
                    audit_cpu(kids[0])
                    return kids[0]
        except (OSError, URLError):
            time.sleep(0.5)
    raise ServerError("remote CPU health timeout")


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-identity", required=True, type=Path)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--server-seconds", required=True, type=int)
    parser.add_argument("--memory-floor", required=True, type=int)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    if args.run_root.exists():
        raise ServerError("remote run root already exists")
    if args.server_seconds != 1150 or args.memory_floor != 2048:
        raise ServerError("v2 server deadline/floor is fixed at 1150s/2048MiB")
    binary = Path(args.binary)
    model = Path(args.model)
    audit_identity = verify_identity(args.runtime_identity, args.runtime_identity_sha256,
                                     binary, "e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6")
    if model.is_symlink() or not model.is_file() or sha256_file(model) != args.model_sha256:
        raise ServerError("Q8 model hash mismatch")
    args.run_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("GGML_") or key == "SEPALITH_VK_TRACE":
            del env[key]
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                "OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "LD_LIBRARY_PATH": str(binary.parent)})
    command = ["/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=15s",
               "1150s", *server_argv(str(binary), str(model), args.port)]
    log = (args.run_root / "server.log").open("xb")
    process: subprocess.Popen[Any] | None = None
    result: dict[str, Any] = {"schema_version": SCHEMA, "profile": PROFILE,
                              "argv": command, "identity": audit_identity,
                              "model_sha256": args.model_sha256, "memory_floor_mib": args.memory_floor}
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=str(args.run_root), env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        write_json(args.run_root / "launch.json", {**result, "at": now(), "pid": process.pid})
        child = wait_health(process, args.port, started + 90, args.memory_floor)
        result["server_child_pid"] = child
        result["device_audit"] = audit_cpu(child)
        write_json(args.run_root / "live-device-audit.json", result["device_audit"])
        print(json.dumps({"kind": "ready", "schema_version": SCHEMA, "profile": PROFILE,
                          "server_pid": process.pid, "server_child_pid": child}, sort_keys=True), flush=True)
        end = started + args.server_seconds
        while time.monotonic() < end:
            ready, _, _ = select.select([sys.stdin], [], [], 1)
            if ready:
                sys.stdin.readline()
                break
            if process.poll() is not None:
                raise ServerError(f"server exited {process.returncode}")
            if available_mib() < args.memory_floor:
                raise ServerError("remote memory floor reached during evaluation")
        result["status"] = "stopped_by_operator" if time.monotonic() < end else "deadline"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if process is not None:
            stop_group(process)
            result["server_exit_code"] = process.returncode
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            write_json(args.run_root / "terminal.json", result)
        except FileExistsError:
            pass
        log.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse()))
    except Exception as exc:
        print(f"remote CPU helper failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
