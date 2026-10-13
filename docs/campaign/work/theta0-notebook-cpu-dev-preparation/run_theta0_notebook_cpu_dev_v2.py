#!/usr/bin/env python3
"""Root-run CPU DEV75 comparator with a local scorer and remote CPU server.

The planning host owns the pinned panel, HF tokenizer and corrected scorer.  The
notebook host receives only this framework-free server helper and the runtime
identity through an SSH loopback tunnel.  Importing this module performs no
framework import, model load, network access, or process launch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable

SCHEMA = "run-09.theta0-notebook-cpu-dev.v2"
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXECUTION_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
LOCAL_PYTHON = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python")
REMOTE_HOST = "m0hawk@192.168.178.40"
REMOTE_ROOT = "/home/m0hawk/.local/share/sepalith-campaign-20260915"
DEFAULT_BINARY = f"{REMOTE_ROOT}/build-b10453-avx2/bin/llama-server"
DEFAULT_MODEL = f"{REMOTE_ROOT}/models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf"
PANEL = PLAN / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
CLIENT = PLAN / "docs/campaign/work/theta0-q8-cuda-dev-preparation/run09_corrected_dev_client.py"
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
RUNTIME_IDENTITY = PLAN / "docs/campaign/work/lead/notebook-b4-cpu-runtime-identity.json"
B4_RECEIPT = PLAN / "docs/campaign/receipts/RUN-03-b4-host-baseline.json"
REMOTE_HELPER = Path(__file__).with_name("remote_notebook_theta0_cpu_server_v2.py")
PANEL_SHA256 = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
CLIENT_SHA256 = "4c9caecaacfdf88f2124f9c8be9f6c8727d150ccee0886f29f544c6e01c12cb0"
RUNTIME_IDENTITY_SHA256 = "0569e1707d79bb3f8f5f39a9a9e63d1fcb1f6f5cf613e8a5f07bbae8ddb1968e"
B4_RECEIPT_SHA256 = "f4d85faf91bbd99efcb12ef1b6283b8db54b3392416b283eb9a955fedcb035ba"
TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_CONFIG_SHA256 = "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"
MODEL_SHA256 = "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559"
BINARY_SHA256 = "e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6"
PANEL_ROWS = 75
MEMORY_FLOOR_MIB = 2048
PROFILE = {
    "backend": "cpu", "context": 4096, "quality_cap": 512,
    "threads": 6, "threads_batch": 6, "threads_http": 2, "parallel": 1,
    "batch": 256, "ubatch": 256, "ngl": 0,
}


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} must be a regular file: {path}")
    return path.resolve()


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    resolved = require_file(path, label)
    actual = sha256_file(resolved)
    if actual != expected:
        raise PreparationError(f"{label} hash mismatch: {actual} != {expected}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def datetime_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise PreparationError("/proc/meminfo has no MemAvailable")


def cpu_environment() -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    removed: list[str] = []
    for key in list(env):
        if key.startswith("GGML_") or key == "SEPALITH_VK_TRACE":
            removed.append(key)
            del env[key]
    return env, sorted(removed)


def server_argv(binary: str, model: str, port: int) -> list[str]:
    return [
        binary, "-m", model, "--host", "127.0.0.1", "--port", str(port),
        "-t", "6", "-tb", "6", "--threads-http", "2", "--parallel", "1",
        "-c", "4096", "-b", "256", "-ub", "256", "-ngl", "0", "-lv", "4",
    ]


def client_argv(run_root: Path, port: int, panel: Path = PANEL,
                client: Path = CLIENT, tokenizer: Path = TOKENIZER,
                execution_root: Path = EXECUTION_ROOT) -> list[str]:
    # run09_native_dev_quality.py owns the fixed COMPLETION_CAP=512.  Do not
    # add a category-dependent or second cap argument here.
    return [
        "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=15s", "1000s",
        str(LOCAL_PYTHON), "-B", str(client), "--url", f"http://127.0.0.1:{port}",
        "--panel", str(panel), "--tokenizer-dir", str(tokenizer),
        "--execution-root", str(execution_root), "--output", str(run_root / "quality.json"),
        "--model-label", "theta0-step1000-Q8_0-CPU-ngl0", "--model-provenance", MODEL_SHA256,
        "--case-deadline-seconds", "120", "--deadline-seconds", "1000",
        "--reserve-seconds", "60",
    ]


def _safe_remote_path(value: str, label: str) -> str:
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", value):
        raise PreparationError(f"{label} contains unsafe remote path characters")
    return value


def ssh_server_argv(stage: str, run_root: str, port: int) -> list[str]:
    stage = _safe_remote_path(stage, "remote staging path")
    run_root = _safe_remote_path(run_root, "remote run root")
    return [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3", "-L", f"127.0.0.1:{port}:127.0.0.1:{port}",
        REMOTE_HOST, "python3", f"{stage}/{REMOTE_HELPER.name}",
        "--binary", DEFAULT_BINARY, "--model", DEFAULT_MODEL,
        "--runtime-identity", f"{stage}/runtime_identity.json",
        "--runtime-identity-sha256", RUNTIME_IDENTITY_SHA256,
        "--model-sha256", MODEL_SHA256, "--run-root", run_root,
        "--port", str(port), "--server-seconds", "1150", "--memory-floor", str(MEMORY_FLOOR_MIB),
    ]


def load_identity(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreparationError(f"cannot read runtime identity: {path}") from exc
    if not isinstance(value, dict) or value.get("binary") != DEFAULT_BINARY:
        raise PreparationError("runtime identity is not the pinned b10453 CPU identity")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise PreparationError("runtime identity dependency list is empty")
    binary_entries = [x for x in files if isinstance(x, dict) and x.get("path") == DEFAULT_BINARY]
    if len(binary_entries) != 1 or binary_entries[0].get("sha256") != BINARY_SHA256:
        raise PreparationError("runtime identity does not bind the pinned CPU binary")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise PreparationError("runtime identity has malformed dependency")
        if not isinstance(item.get("bytes"), int) or not isinstance(item.get("sha256"), str):
            raise PreparationError("runtime identity dependency lacks size/hash")
    return value


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise PreparationError(f"refusing to overwrite local run root: {args.run_root}")
    if not 1024 <= args.port <= 65500 or not port_is_free(args.port):
        raise PreparationError(f"local tunnel port unavailable: {args.port}")
    if not 1 <= args.overall_deadline_seconds <= 1200:
        raise PreparationError("overall deadline must be 1..1200 seconds")
    if not 1 <= args.client_deadline_seconds <= 1000:
        raise PreparationError("client deadline must be 1..1000 seconds")
    if args.remote_run_root != f"{REMOTE_ROOT}/runs/{args.run_root.name}":
        raise PreparationError("remote run root must correspond to the fresh local run name")
    identity_path = require_hash(args.runtime_identity, RUNTIME_IDENTITY_SHA256, "runtime identity")
    identity = load_identity(args.runtime_identity)
    client_info = require_hash(args.client, CLIENT_SHA256, "corrected local DEV client")
    panel_info = require_hash(args.panel, PANEL_SHA256, "corrected DEV75 panel")
    panel_rows = sum(1 for line in args.panel.open(encoding="utf-8") if line.strip())
    if panel_rows != PANEL_ROWS:
        raise PreparationError(f"panel rows {panel_rows} != {PANEL_ROWS}")
    tokenizer_json = require_hash(args.tokenizer / "tokenizer.json", TOKENIZER_JSON_SHA256, "local HF tokenizer.json")
    tokenizer_config = require_hash(args.tokenizer / "tokenizer_config.json", TOKENIZER_CONFIG_SHA256, "local HF tokenizer_config.json")
    execution_root = require_file(args.execution_root / "experiments/training/campaign_eval.py", "local EXEC campaign_eval.py")
    b4_info = require_hash(args.b4_receipt, B4_RECEIPT_SHA256, "b4 runtime receipt")
    env, removed = cpu_environment()
    return {
        "schema_version": SCHEMA, "generated_utc": datetime_now(), "profile": PROFILE,
        "quality_contract": {"context_tokens": 4096, "completion_cap": 512,
                             "cap_source": "pinned run09_native_dev_quality.py fixed COMPLETION_CAP"},
        "placement": {"scorer": "local planning host", "hf_tokenizer": "local planning host",
                       "panel": "local planning host", "server": "remote notebook CPU over SSH loopback"},
        "paths": {"local_client": client_info, "local_panel": panel_info,
                  "local_tokenizer": {"path": str(args.tokenizer), "tokenizer.json": tokenizer_json,
                                      "tokenizer_config.json": tokenizer_config},
                  "local_execution_root": str(args.execution_root), "remote_binary": DEFAULT_BINARY,
                  "remote_model": DEFAULT_MODEL, "runtime_identity": identity_path, "b4_receipt": b4_info},
        "runtime_identity_entries": len(identity["files"]), "panel_rows": panel_rows,
        "environment": {"local_cuda_visible_devices": "", "local_threads": 1,
                         "remote_threads": 6, "vulkan_dri_required": False,
                         "removed_backend_variables": removed},
        "deadlines": {"overall_seconds": args.overall_deadline_seconds,
                      "client_seconds": args.client_deadline_seconds,
                      "remote_server_hard_seconds": 1150, "reserve_seconds": 60,
                      "memory_floor_mib": MEMORY_FLOOR_MIB},
        "remote_run_root": args.remote_run_root,
        "quality_scope": "matched corrected DEV75 diagnostic comparator; no promotion",
    }


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


def run_command(command: list[str], *, input_text: str | None = None) -> None:
    result = subprocess.run(command, input=input_text, text=True, check=False)
    if result.returncode:
        raise PreparationError(f"command failed ({result.returncode}): {shlex.join(command)}")


def wait_ready(ssh: subprocess.Popen[str], deadline: float) -> dict[str, Any]:
    if ssh.stdout is None:
        raise PreparationError("SSH stdout is unavailable")
    while time.monotonic() < deadline:
        ready, _, _ = select.select([ssh.stdout], [], [], 1)
        if not ready:
            if ssh.poll() is not None:
                raise PreparationError(f"SSH server helper exited: {ssh.returncode}")
            continue
        line = ssh.stdout.readline()
        if not line:
            raise PreparationError("SSH server helper closed before ready")
        value = json.loads(line)
        if value.get("kind") != "ready" or value.get("profile") != PROFILE:
            raise PreparationError("remote ready record does not bind the CPU profile")
        return value
    raise PreparationError("remote CPU server readiness deadline exceeded")


def run(args: argparse.Namespace) -> int:
    plan = preflight(args)
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json(args.run_root / "preflight.json", plan)
    env, removed = cpu_environment()
    stage = f"{args.remote_run_root}.staging"
    started = time.monotonic()
    outer_deadline = started + args.overall_deadline_seconds
    ssh: subprocess.Popen[str] | None = None
    client: subprocess.Popen[Any] | None = None
    outcome: dict[str, Any] = {"schema_version": SCHEMA, "profile": PROFILE,
                               "client_command": client_argv(args.run_root, args.port,
                                                             args.panel, args.client,
                                                             args.tokenizer, args.execution_root),
                               "ssh_command": ssh_server_argv(stage, args.remote_run_root, args.port),
                               "removed_backend_variables": removed}
    try:
        remote_guard = (
            f"test ! -e {shlex.quote(stage)} && test ! -e {shlex.quote(args.remote_run_root)} "
            f"&& mkdir -p {shlex.quote(stage)}"
        )
        run_command(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                     REMOTE_HOST, remote_guard])
        run_command(["scp", "-q", "-o", "BatchMode=yes", str(REMOTE_HELPER),
                     f"{REMOTE_HOST}:{stage}/{REMOTE_HELPER.name}"])
        run_command(["scp", "-q", "-o", "BatchMode=yes", str(args.runtime_identity),
                     f"{REMOTE_HOST}:{stage}/runtime_identity.json"])
        ssh = subprocess.Popen(ssh_server_argv(stage, args.remote_run_root, args.port),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=(args.run_root / "ssh.stderr").open("xb"),
                               text=True, start_new_session=True)
        ready = wait_ready(ssh, min(outer_deadline, started + 90))
        write_json(args.run_root / "remote-ready.json", ready)
        if time.monotonic() >= outer_deadline:
            raise PreparationError("overall deadline reached before local scorer")
        client_log = (args.run_root / "client.log").open("xb")
        client = subprocess.Popen(outcome["client_command"], cwd=str(args.run_root), env=env,
                                  stdout=client_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        while client.poll() is None:
            if time.monotonic() >= outer_deadline:
                raise PreparationError("overall deadline reached during local scorer")
            if ssh.poll() is not None:
                raise PreparationError(f"remote SSH exited during scorer: {ssh.returncode}")
            if available_memory_mib() < MEMORY_FLOOR_MIB:
                raise PreparationError("planning host memory floor reached")
            time.sleep(1)
        outcome["client_exit_code"] = client.returncode
        if client.returncode != 0:
            raise PreparationError(f"local scorer failed: {client.returncode}")
        quality = args.run_root / "quality.json"
        if not quality.is_file():
            raise PreparationError("local scorer produced no quality.json")
        outcome["quality"] = {"path": str(quality), "bytes": quality.stat().st_size,
                               "sha256": sha256_file(quality)}
        outcome["status"] = "completed"
        return 0
    except Exception as exc:
        outcome["status"] = "failed"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if client is not None:
            stop_group(client)
        if ssh is not None:
            if ssh.stdin is not None and ssh.poll() is None:
                try:
                    ssh.stdin.write("STOP\n")
                    ssh.stdin.flush()
                except OSError:
                    pass
            try:
                ssh.wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_group(ssh)
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            write_json(args.run_root / "terminal.json", outcome)
        except FileExistsError:
            pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--remote-run-root")
    result.add_argument("--port", type=int, default=18413)
    result.add_argument("--panel", type=Path, default=PANEL)
    result.add_argument("--client", type=Path, default=CLIENT)
    result.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    result.add_argument("--execution-root", type=Path, default=EXECUTION_ROOT)
    result.add_argument("--runtime-identity", type=Path, default=RUNTIME_IDENTITY)
    result.add_argument("--b4-receipt", type=Path, default=B4_RECEIPT)
    result.add_argument("--overall-deadline-seconds", type=int, default=1200)
    result.add_argument("--client-deadline-seconds", type=int, default=1000)
    result.add_argument("--remote-binary", default=DEFAULT_BINARY)
    result.add_argument("--remote-model", default=DEFAULT_MODEL)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.remote_run_root is None:
        parsed.remote_run_root = f"{REMOTE_ROOT}/runs/{parsed.run_root.name}"
    try:
        raise SystemExit(run(parsed))
    except Exception as error:
        print(f"theta0 notebook CPU v2 failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
