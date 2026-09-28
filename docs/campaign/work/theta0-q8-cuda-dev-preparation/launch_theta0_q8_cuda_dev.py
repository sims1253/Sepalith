#!/usr/bin/env python3
"""Guarded local-CUDA RUN-09 theta0-Q8 DEV evaluator.

This is a root-launch helper.  Importing it or running its static tests never
starts a server and never opens the model.  A root launch verifies the selected
GGUF by a streaming SHA-256 pass, starts only a fresh process group, waits for
health and n_ctx=4096, and invokes one of the two explicitly separate clients:
the corrected 75-case quality scorer (512 completion cap), or the existing
four-row TRAIN diagnostic (192 completion cap).  No process-wide kill or
existing run-directory cleanup is performed.
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
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
CUDA_ROOT = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453")
SERVER = CUDA_ROOT / "llama-server"
MODEL = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/"
    "SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf"
)
PANEL = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
PANEL_MANIFEST = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/manifest.json"
ADAPTER = PLAN_ROOT / "docs/campaign/work/theta0-q8-cuda-dev-preparation/run09_corrected_dev_client.py"
REFERENCE_SCORER = PLAN_ROOT / "docs/campaign/work/quant-quality/run09_native_dev_quality.py"
DIAGNOSTIC = PLAN_ROOT / "docs/campaign/work/serving-readiness/runtime_native_probe.py"
TRAIN_FIXTURE = PLAN_ROOT / "docs/campaign/work/serving-readiness/native-probe-train-fixture.jsonl"
TRAIN_MANIFEST = PLAN_ROOT / "docs/campaign/work/serving-readiness/native-probe-train-fixture.manifest.json"
TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
PYTHON = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python")

SERVER_SHA256 = "e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee"
MODEL_SHA256 = "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559"
PANEL_SHA256 = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
PANEL_MANIFEST_SHA256 = "4528ff6dbd8cfe70580901d9254a1a6931dc8c334c5dd2c55e49cf92d1a5377a"
REFERENCE_SCORER_SHA256 = "e574a9453066b5fa5a8e33c02777e8bdd875e208d4c3c51c00e5ede13bba5362"
DIAGNOSTIC_SHA256 = "f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7"
TRAIN_FIXTURE_SHA256 = "4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008"
TRAIN_MANIFEST_SHA256 = "0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3"
TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_CONFIG_SHA256 = "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"
PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"
CAMPAIGN_EVAL_SHA256 = "7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f"
CAMPAIGN_CLIENT_SHA256 = "0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333"

SERVER_GUARD_SECONDS = 1_200
QUALITY_GLOBAL_SECONDS = 1_100
QUALITY_RESERVE_SECONDS = 60
QUALITY_CASE_SECONDS = 120
QUALITY_COMPLETION_CAP = 512
DIAGNOSTIC_USER_DEADLINE_MS = 5_000
DIAGNOSTIC_TIMEOUT_MS = 60_000
DIAGNOSTIC_COMPLETION_CAP = 192
CONTEXT_SIZE = 4_096
SERVER_PORT_FORBIDDEN = 18_099
HEALTH_TIMEOUT_SECONDS = 90
HEALTH_POLL_SECONDS = 1.0


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise PreparationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def fsync_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _require_ext4_parent(path: Path) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise PreparationError("run-root must be absolute and its parent must exist")
    if path.exists():
        raise PreparationError(f"run-root must be fresh: {path}")
    try:
        result = subprocess.run(
            ["/usr/bin/findmnt", "-T", str(path.parent), "-n", "-o", "FSTYPE"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreparationError(f"cannot inspect run-root filesystem: {error}") from error
    if result.stdout.strip() != "ext4":
        raise PreparationError(f"run-root parent must be ext4, got {result.stdout.strip()!r}")


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise PreparationError(f"{label} hash mismatch: {observed} != {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def _validate_port(port: int) -> None:
    if not 1_024 <= port <= 65_535 or port == SERVER_PORT_FORBIDDEN:
        raise PreparationError("port must be a fresh non-privileged port other than CPU server 18099")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise PreparationError(f"port {port} is not free: {error}") from error


def server_argv(port: int) -> list[str]:
    return [
        str(SERVER), "-m", str(MODEL),
        "--host", "127.0.0.1", "--port", str(port),
        "-t", "6", "-tb", "6", "--threads-http", "2",
        "--parallel", "1", "-c", str(CONTEXT_SIZE), "-b", "256", "-ub", "256",
        "-lv", "4", "-ngl", "99",
    ]


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("GGML_CUDA_"):
            del env[key]
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "GGML_CUDA_GRAPH_OPT": "0",
        "OMP_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": f"{CUDA_ROOT}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
    })
    return env


def quality_argv(port: int, output: Path) -> list[str]:
    return [
        "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=15s",
        f"{QUALITY_GLOBAL_SECONDS}s", str(PYTHON), "-B", str(ADAPTER),
        "--url", f"http://127.0.0.1:{port}", "--panel", str(PANEL),
        "--tokenizer-dir", str(TOKENIZER_DIR), "--execution-root", str(EXEC_ROOT),
        "--output", str(output), "--model-label", "theta0-step1000-Q8_0-CUDA-graph0",
        "--model-provenance", MODEL_SHA256,
        "--case-deadline-seconds", str(QUALITY_CASE_SECONDS),
        "--deadline-seconds", str(QUALITY_GLOBAL_SECONDS),
        "--reserve-seconds", str(QUALITY_RESERVE_SECONDS),
    ]


def diagnostic_argv(port: int, output: Path) -> list[str]:
    return [
        "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=15s",
        f"{QUALITY_GLOBAL_SECONDS}s", str(PYTHON), "-B", str(DIAGNOSTIC),
        "--fixture", str(TRAIN_FIXTURE), "--manifest", str(TRAIN_MANIFEST),
        "--url", f"http://127.0.0.1:{port}", "--arm", "baseline",
        "--cap", str(DIAGNOSTIC_COMPLETION_CAP), "--context", str(CONTEXT_SIZE),
        "--reps", "1", "--user-deadline-ms", str(DIAGNOSTIC_USER_DEADLINE_MS),
        "--diagnostic-timeout-ms", str(DIAGNOSTIC_TIMEOUT_MS), "--out", str(output),
    ]


def _request_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise PreparationError(f"server preflight failed for {url}: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"server preflight did not return an object: {url}")
    return value


def _wait_for_server(base_url: str, process: subprocess.Popen[Any]) -> dict[str, Any]:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise PreparationError(f"server exited before health: {process.returncode}")
        try:
            health = _request_json(f"{base_url}/health", 2.0)
        except PreparationError as error:  # connection refused/503 during load
            last_error = str(error)
            time.sleep(HEALTH_POLL_SECONDS)
            continue
        if health.get("status") != "ok":
            last_error = f"health status {health.get('status')!r}"
            time.sleep(HEALTH_POLL_SECONDS)
            continue
        try:
            props = _request_json(f"{base_url}/props", 5.0)
        except PreparationError as error:  # transient props endpoint startup
            last_error = str(error)
            time.sleep(HEALTH_POLL_SECONDS)
            continue
        generation = props.get("default_generation_settings")
        n_ctx = generation.get("n_ctx") if isinstance(generation, Mapping) else None
        if n_ctx != CONTEXT_SIZE:
            raise PreparationError(f"server n_ctx is {n_ctx!r}, expected {CONTEXT_SIZE}")
        return {"health": health, "props": props}
    raise PreparationError(f"health timeout after {HEALTH_TIMEOUT_SECONDS}s: {last_error}")


def _terminate_group(process: subprocess.Popen[Any] | None) -> dict[str, Any]:
    if process is None:
        return {"started": False, "returncode": None}
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
            process.wait(timeout=5)
    return {"started": True, "returncode": process.returncode}


def _input_inventory(mode: str) -> dict[str, Any]:
    inventory = {
        "server": _require_hash(SERVER, SERVER_SHA256, "CUDA llama-server"),
        "panel": _require_hash(PANEL, PANEL_SHA256, "corrected DEV panel"),
        "panel_manifest": _require_hash(PANEL_MANIFEST, PANEL_MANIFEST_SHA256, "corrected DEV manifest"),
        "adapter": _require_hash(ADAPTER, sha256_file(ADAPTER), "corrected-panel adapter"),
        "reference_scorer": _require_hash(REFERENCE_SCORER, REFERENCE_SCORER_SHA256, "RUN-09 scorer"),
    }
    if mode == "quality":
        inventory.update({
            "protocol": {"path": str(EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"),
                         "sha256": PROTOCOL_SHA256},
            "campaign_eval": {"path": str(EXEC_ROOT / "experiments/training/campaign_eval.py"),
                              "sha256": CAMPAIGN_EVAL_SHA256},
            "campaign_client": {"path": str(EXEC_ROOT / "extensions/vscode-sepalith/src/campaign_client.ts"),
                                 "sha256": CAMPAIGN_CLIENT_SHA256},
            "tokenizer": {"path": str(TOKENIZER_DIR),
                          "json_sha256": TOKENIZER_JSON_SHA256,
                          "config_sha256": TOKENIZER_CONFIG_SHA256},
        })
    else:
        inventory.update({
            "diagnostic": _require_hash(DIAGNOSTIC, DIAGNOSTIC_SHA256, "TRAIN diagnostic"),
            "train_fixture": _require_hash(TRAIN_FIXTURE, TRAIN_FIXTURE_SHA256, "TRAIN fixture"),
            "train_manifest": _require_hash(TRAIN_MANIFEST, TRAIN_MANIFEST_SHA256, "TRAIN manifest"),
        })
    return inventory


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    _require_ext4_parent(run_root)
    _validate_port(args.port)
    mode = args.mode
    run_root.mkdir()
    started_at = datetime.now(timezone.utc).isoformat()
    output = run_root / ("quality.json" if mode == "quality" else "diagnostic.json")
    launch: dict[str, Any] = {
        "schema_version": "sepalith.run09.theta0-q8-cuda-dev-launch.v1",
        "task": "RUN-09/SFT-08",
        "status": "preflight",
        "mode": mode,
        "started_at": started_at,
        "run_root": str(run_root),
        "server_guard_seconds": SERVER_GUARD_SECONDS,
        "port": args.port,
        "server_argv": server_argv(args.port),
        "env_contract": {"CUDA_VISIBLE_DEVICES": "0", "GGML_CUDA_GRAPH_OPT": "0",
                          "threads": {"server": 6, "http": 2}},
        "quality_contract": {"panel_rows": 75, "edit_rows": 43, "strict_noop_rows": 32,
                             "completion_cap": QUALITY_COMPLETION_CAP,
                             "case_deadline_seconds": QUALITY_CASE_SECONDS,
                             "global_deadline_seconds": QUALITY_GLOBAL_SECONDS,
                             "reserve_seconds": QUALITY_RESERVE_SECONDS},
        "diagnostic_contract": {"fixture_rows": 4, "completion_cap": DIAGNOSTIC_COMPLETION_CAP,
                                "user_deadline_ms": DIAGNOSTIC_USER_DEADLINE_MS,
                                "diagnostic_timeout_ms": DIAGNOSTIC_TIMEOUT_MS},
        "output": str(output),
    }
    fsync_json(run_root / "launch.json", launch)
    server_process: subprocess.Popen[Any] | None = None
    client_process: subprocess.CompletedProcess[Any] | None = None
    terminal: dict[str, Any] = {}
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def _interrupt(signum: int, _frame: Any) -> None:
        raise PreparationError(f"launcher interrupted by signal {signum}")

    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        launch["inputs"] = _input_inventory(mode)
        # This is the only model-byte read in an admitted root run, and it is
        # deliberately before server start so the selected identity is bound.
        launch["inputs"]["model"] = _require_hash(MODEL, MODEL_SHA256, "selected theta0 Q8 GGUF")
        launch["binary_version"] = subprocess.run(
            [str(SERVER), "--version"], check=True, capture_output=True,
            text=True, timeout=10, env=_base_env(),
        ).stdout.strip()
        launch["status"] = "server_starting"
        fsync_json(run_root / "launch.json", launch)
        server_log = (run_root / "server.log").open("w", encoding="utf-8")
        try:
            server_process = subprocess.Popen(
                ["/usr/bin/timeout", "--foreground", "--signal=TERM",
                 "--kill-after=15s", f"{SERVER_GUARD_SECONDS}s", *server_argv(args.port)],
                stdin=subprocess.DEVNULL, stdout=server_log, stderr=subprocess.STDOUT,
                env=_base_env(), start_new_session=True,
            )
            launch["server_pid"] = server_process.pid
            fsync_json(run_root / "launch.json", launch)
            base_url = f"http://127.0.0.1:{args.port}"
            launch["server_preflight"] = _wait_for_server(base_url, server_process)
            launch["status"] = "client_starting"
            fsync_json(run_root / "launch.json", launch)
            command = quality_argv(args.port, output) if mode == "quality" else diagnostic_argv(args.port, output)
            launch["client_argv"] = command
            fsync_json(run_root / "launch.json", launch)
            client_log_path = run_root / "client.log"
            with client_log_path.open("w", encoding="utf-8") as client_log:
                client_process = subprocess.run(
                    command, env=_base_env(), check=False,
                    stdout=client_log, stderr=subprocess.STDOUT,
                )
            terminal["client_returncode"] = client_process.returncode
            terminal["client_log_bytes"] = client_log_path.stat().st_size
            terminal["client_log_sha256"] = sha256_file(client_log_path)
            if output.exists():
                terminal["client_output_bytes"] = output.stat().st_size
                terminal["client_output_sha256"] = sha256_file(output)
            launch["status"] = "client_terminal"
            fsync_json(run_root / "launch.json", launch)
        finally:
            server_log.flush()
            server_log.close()
    except Exception as error:
        terminal["failure"] = {"type": type(error).__name__, "message": str(error)}
        launch["status"] = "failed"
    finally:
        terminal["server"] = _terminate_group(server_process)
        terminal["completed_at"] = datetime.now(timezone.utc).isoformat()
        terminal["mode"] = mode
        fsync_json(run_root / "terminal.json", terminal)
        launch["terminal"] = terminal
        fsync_json(run_root / "launch.json", launch)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
    return 0 if not terminal.get("failure") and terminal.get("client_returncode") == 0 else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--port", type=int, default=18_403)
    result.add_argument("--mode", choices=("quality", "diagnostic"), default="quality")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": {"type": type(error).__name__,
                                                               "message": str(error)}}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
