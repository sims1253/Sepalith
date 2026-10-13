#!/usr/bin/env python3
"""Root-owned local CPU replay for the selected theta0 Q8 DEV panel.

Importing this module is framework-free and launch-free. The root operator
executes it on the notebook host after checking the pinned runtime identity.
It never enables Vulkan or requires a Vulkan DRI device.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


SCHEMA = "run-09.theta0-notebook-cpu-dev.v1"
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
REMOTE = Path("/home/m0hawk/.local/share/sepalith-campaign-20260915")
DEFAULT_BINARY = REMOTE / "build-b10453-avx2/bin/llama-server"
DEFAULT_MODEL = REMOTE / "models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf"
DEFAULT_PANEL = PLAN / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
DEFAULT_CLIENT = PLAN / "docs/campaign/work/theta0-q8-cuda-dev-preparation/run09_corrected_dev_client.py"
DEFAULT_RUNTIME_IDENTITY = PLAN / "docs/campaign/work/lead/notebook-b4-cpu-runtime-identity.json"
DEFAULT_B4_RECEIPT = PLAN / "docs/campaign/receipts/RUN-03-b4-host-baseline.json"
PANEL_SHA256 = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
PANEL_ROWS = 75
CLIENT_SHA256 = "4c9caecaacfdf88f2124f9c8be9f6c8727d150ccee0886f29f544c6e01c12cb0"
MODEL_SHA256 = "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559"
RUNTIME_IDENTITY_SHA256 = "0569e1707d79bb3f8f5f39a9a9e63d1fcb1f6f5cf613e8a5f07bbae8ddb1968e"
B4_RECEIPT_SHA256 = "f4d85faf91bbd99efcb12ef1b6283b8db54b3392416b283eb9a955fedcb035ba"
BINARY_SHA256 = "e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6"
PROFILE = {
    "backend": "cpu",
    "context": 4096,
    "cap": 192,
    "threads": 6,
    "threads_batch": 6,
    "threads_http": 2,
    "parallel": 1,
    "batch": 256,
    "ubatch": 256,
    "ngl": 0,
}
MEMORY_FLOOR_MIB = 2048


class PreparationError(RuntimeError):
    """A fail-closed identity, resource, or lifecycle error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_regular(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PreparationError(f"{label} is absent: {path}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise PreparationError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def verify_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected:
        raise PreparationError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise PreparationError("MemAvailable is missing")


def require_memory_floor() -> int:
    value = available_memory_mib()
    if value < MEMORY_FLOOR_MIB:
        raise PreparationError(
            f"available memory {value} MiB is below {MEMORY_FLOOR_MIB} MiB floor"
        )
    return value


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def load_runtime_identity(path: Path) -> dict[str, Any]:
    identity = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise PreparationError("runtime identity is not an object")
    if identity.get("binary") != str(DEFAULT_BINARY):
        raise PreparationError("runtime identity binary differs from pinned CPU binary")
    files = identity.get("files")
    if not isinstance(files, list) or not files:
        raise PreparationError("runtime identity lacks dependency files")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise PreparationError("runtime identity has malformed dependency entry")
    return identity


def verify_runtime_files(identity: dict[str, Any], binary: Path) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for item in identity["files"]:
        path = require_regular(Path(item["path"]), "runtime dependency")
        if path != Path(item["path"]).resolve():
            raise PreparationError("runtime dependency path changed during resolution")
        expected_size = item.get("bytes")
        expected_hash = item.get("sha256")
        if not isinstance(expected_size, int) or not isinstance(expected_hash, str):
            raise PreparationError("runtime dependency entry lacks size/hash")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise PreparationError(f"runtime dependency identity mismatch: {path}")
        verified.append({"path": str(path), "bytes": expected_size, "sha256": expected_hash})
    binary_entry = next((item for item in verified if item["path"] == str(binary)), None)
    if binary_entry is None or binary_entry["sha256"] != BINARY_SHA256:
        raise PreparationError("CPU binary is not the pinned b10453 artifact")
    return verified


def server_argv(binary: Path, model: Path, port: int) -> list[str]:
    return [
        str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-t", "6", "-tb", "6", "--threads-http", "2", "--parallel", "1",
        "-c", "4096", "-b", "256", "-ub", "256", "-ngl", "0", "-lv", "4",
    ]


def cpu_environment() -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    removed: list[str] = []
    for key in list(env):
        if key.startswith("GGML_") or key in {
            "SEPALITH_VK_TRACE", "GGML_BACKEND_PATH", "GGML_CUDA_GRAPH_OPT",
        }:
            removed.append(key)
            del env[key]
    env["LD_LIBRARY_PATH"] = str(DEFAULT_BINARY.parent)
    return env, sorted(set(removed))


def child_pids(pid: int) -> list[int]:
    try:
        value = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return []
    return [int(item) for item in value if item.isdigit()]


def device_audit(pid: int) -> dict[str, Any]:
    maps: list[str] = []
    dri_fds: list[str] = []
    try:
        maps = [
            line for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
            if "libggml-vulkan" in line or "libvulkan" in line
        ]
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if "/dev/dri/" in target:
                dri_fds.append(target)
    except OSError:
        pass
    if maps or dri_fds:
        raise PreparationError("CPU child acquired Vulkan library mappings or DRI FDs")
    return {"pid": pid, "vulkan_mappings": maps, "dri_fds": dri_fds}


def stop_group(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
    return {"pid": proc.pid, "exit_code": proc.returncode, "gone": proc.poll() is not None}


def wait_health(proc: subprocess.Popen[bytes], port: int, deadline: float) -> int:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise PreparationError(f"CPU server exited during load: {proc.returncode}")
        require_memory_floor()
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if int(response.status) == 200:
                    descendants = child_pids(proc.pid)
                    if not descendants:
                        raise PreparationError("healthy timeout wrapper has no server child")
                    for child in descendants:
                        device_audit(child)
                    return descendants[0]
        except (OSError, URLError):
            time.sleep(0.5)
    raise PreparationError("CPU server health deadline exceeded")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise PreparationError(f"refusing to overwrite existing run root: {args.run_root}")
    if args.port < 1024 or args.port > 65500 or not port_is_free(args.port):
        raise PreparationError(f"port is unavailable: {args.port}")
    if args.overall_deadline_seconds <= 0 or args.overall_deadline_seconds > 1200:
        raise PreparationError("overall deadline must be in 1..1200 seconds")
    if args.client_deadline_seconds <= 0 or args.client_deadline_seconds > 1000:
        raise PreparationError("client deadline must be in 1..1000 seconds")
    if args.runtime_identity.resolve() != DEFAULT_RUNTIME_IDENTITY.resolve():
        raise PreparationError("only the pinned notebook CPU runtime identity is accepted")
    identity_file = verify_hash(
        require_regular(args.runtime_identity, "runtime identity"),
        RUNTIME_IDENTITY_SHA256,
        "runtime identity",
    )
    identity = load_runtime_identity(args.runtime_identity)
    binary = require_regular(args.binary, "CPU binary")
    deps = verify_runtime_files(identity, binary)
    if binary != DEFAULT_BINARY.resolve():
        raise PreparationError("only the pinned CPU binary path is accepted")
    model = verify_hash(require_regular(args.model, "theta0 Q8 model"), MODEL_SHA256, "theta0 Q8 model")
    panel = verify_hash(require_regular(args.panel, "corrected DEV panel"), PANEL_SHA256, "corrected DEV panel")
    with args.panel.open(encoding="utf-8") as stream:
        rows = sum(1 for line in stream if line.strip())
    if rows != PANEL_ROWS:
        raise PreparationError(f"corrected panel has {rows} rows, expected {PANEL_ROWS}")
    client = verify_hash(require_regular(args.client, "corrected DEV client"), CLIENT_SHA256, "corrected DEV client")
    receipt = verify_hash(require_regular(args.b4_receipt, "b4 host receipt"), B4_RECEIPT_SHA256, "b4 host receipt")
    env, removed = cpu_environment()
    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime_now(),
        "profile": PROFILE,
        "memory_floor_mib": MEMORY_FLOOR_MIB,
        "deadline": {
            "overall_seconds": args.overall_deadline_seconds,
            "client_seconds": args.client_deadline_seconds,
            "server_hard_seconds": 1150,
            "reserve_seconds": 60,
        },
        "binary": {"identity": identity_file, "verified_files": deps},
        "model": model,
        "panel": {**panel, "rows": rows},
        "client": client,
        "b4_runtime_receipt": receipt,
        "environment": {
            "cuda_visible_devices": "",
            "vulkan_dri_required": False,
            "removed_backend_variables": removed,
            "ld_library_path": str(DEFAULT_BINARY.parent),
        },
        "source_adapters": {
            "quality_client": str(args.client),
            "notebook_vulkan_server_reference": str(
                PLAN / "docs/campaign/work/lead/theta0-notebook-dev-a/notebook_theta0_quality_server.py"
            ),
            "notebook_vulkan_launcher_reference": str(
                PLAN / "docs/campaign/work/lead/theta0-notebook-dev-a/run_notebook_theta0_quality.py"
            ),
        },
        "quality_scope": "same corrected DEV75 panel; diagnostic CPU comparator only; no release promotion",
    }


def datetime_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args: argparse.Namespace) -> int:
    plan = preflight(args)
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json(args.run_root / "preflight.json", plan)
    started = time.monotonic()
    outer_deadline = started + args.overall_deadline_seconds
    env, removed = cpu_environment()
    server_command = [
        "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=15s",
        "1150s", *server_argv(args.binary.resolve(), args.model.resolve(), args.port),
    ]
    server_log = args.run_root / "server.log"
    client_log = args.run_root / "client.log"
    server: subprocess.Popen[bytes] | None = None
    outcome: dict[str, Any] = {
        "schema_version": SCHEMA,
        "profile": PROFILE,
        "server_command": server_command,
        "client_command": [],
        "removed_backend_variables": removed,
    }
    try:
        with server_log.open("xb") as log:
            server = subprocess.Popen(
                server_command, cwd=str(args.run_root), env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        launch = {
            "schema_version": SCHEMA, "at": datetime_now(),
            "supervisor_pid": os.getpid(), "server_timeout_pid": server.pid,
            "argv": server_command, "profile": PROFILE,
            "runtime_identity_sha256": RUNTIME_IDENTITY_SHA256,
            "model_sha256": MODEL_SHA256, "panel_sha256": PANEL_SHA256,
            "cpu_only": True, "vulkan_dri_required": False,
            "memory_floor_mib": MEMORY_FLOOR_MIB,
        }
        write_json(args.run_root / "launch.json", launch)
        server_child = wait_health(server, args.port, min(outer_deadline, started + 60))
        outcome["server_child_pid"] = server_child
        outcome["device_audit"] = device_audit(server_child)
        if time.monotonic() >= outer_deadline:
            raise PreparationError("overall deadline reached before client")
        client_command = [
            "/usr/bin/timeout", "--foreground", "--signal=TERM",
            "--kill-after=15s", f"{args.client_deadline_seconds}s",
            sys.executable, "-B", str(args.client),
            "--url", f"http://127.0.0.1:{args.port}",
            "--panel", str(args.panel),
            "--output", str(args.run_root / "quality.json"),
            "--model-label", "theta0-step1000-Q8_0-CPU-ngl0",
            "--model-provenance", MODEL_SHA256,
            "--case-deadline-seconds", "120",
            "--deadline-seconds", str(args.client_deadline_seconds),
            "--reserve-seconds", "60",
        ]
        outcome["client_command"] = client_command
        with client_log.open("xb") as log:
            client = subprocess.Popen(
                client_command, cwd=str(args.run_root), env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while client.poll() is None:
                if time.monotonic() >= outer_deadline:
                    stop_group(client)
                    raise PreparationError("overall deadline reached during client")
                require_memory_floor()
                if server.poll() is not None:
                    stop_group(client)
                    raise PreparationError(f"CPU server exited during evaluation: {server.returncode}")
                time.sleep(1)
        outcome["client_exit_code"] = client.returncode
        if client.returncode != 0:
            raise PreparationError(f"quality client failed with exit {client.returncode}")
        if not (args.run_root / "quality.json").is_file():
            raise PreparationError("quality client exited without quality.json")
        outcome["quality_json"] = {
            "path": str(args.run_root / "quality.json"),
            "bytes": (args.run_root / "quality.json").stat().st_size,
            "sha256": sha256_file(args.run_root / "quality.json"),
        }
        outcome["status"] = "completed"
        return 0
    except Exception as exc:
        outcome["status"] = "failed"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if server is not None:
            outcome["server"] = stop_group(server)
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            write_json(args.run_root / "terminal.json", outcome)
        except FileExistsError:
            pass


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--client", type=Path, default=DEFAULT_CLIENT)
    p.add_argument("--runtime-identity", type=Path, default=DEFAULT_RUNTIME_IDENTITY)
    p.add_argument("--b4-receipt", type=Path, default=DEFAULT_B4_RECEIPT)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--port", type=int, default=18413)
    p.add_argument("--overall-deadline-seconds", type=int, default=1200)
    p.add_argument("--client-deadline-seconds", type=int, default=1000)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        print(f"theta0 notebook CPU preparation failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
