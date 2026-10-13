#!/usr/bin/env python3
"""Run one admitted CPU quant arm with owned process cleanup."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ARMS = ("q8", "q4_calibrated", "iq3", "iq2")
OWNED_STOP_SIGNALS = (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual}")


def load_probe(path: Path):
    spec = importlib.util.spec_from_file_location("quant_panel_probe", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_port_free(host: str, port: int) -> None:
    with socket.socket() as sock:
        if sock.connect_ex((host, port)) == 0:
            raise ValueError(f"port already in use: {host}:{port}")


def port_is_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) != 0


def cleanup_owned_group(process: subprocess.Popen) -> None:
    """Stop only the process group created for this runner's server."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def read_health(url: str, timeout: float = 1.0):
    try:
        with urlopen(url, timeout=timeout) as response:
            raw = response.read()
            return int(response.status), json.loads(raw.decode("utf-8"))
    except (OSError, URLError, ValueError):
        return None


def snapshot_preserved(packet: dict) -> dict:
    result = {}
    for name, entry in packet["preserve"].items():
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        path = Path(entry["path"])
        actual = sha256(path) if path.is_file() else None
        result[name] = {"path": str(path), "expected_sha256": entry["sha256"], "actual_sha256": actual,
                        "status": "match" if actual == entry["sha256"] else "mismatch"}
        if actual != entry["sha256"]:
            raise ValueError(f"preserved profile mismatch before run: {name}")
    return result


def telemetry_loop(pid: int, output: Path, stop: threading.Event) -> None:
    clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    page = os.sysconf("SC_PAGE_SIZE")
    while not stop.wait(1.0):
        record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": pid}
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            record.update({"utime_s": int(fields[13]) / clk, "stime_s": int(fields[14]) / clk,
                           "rss_bytes": int(fields[23]) * page})
        except (OSError, ValueError, IndexError):
            record["process"] = "absent"
        try:
            record["loadavg"] = Path("/proc/loadavg").read_text().strip()
            record["meminfo"] = {line.split(":", 1)[0]: line.split()[1]
                                 for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith(("MemAvailable:", "SwapFree:"))}
        except OSError:
            pass
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def quality(rows: list[dict], requests: list[dict]) -> dict:
    expected = {row["id"]: row for row in rows}
    cold = [record for record in requests if record["phase"] == "cold"]
    for record in requests:
        row = expected[record["row_id"]]
        record["quality"] = {
            "strict_exact_target_text": record.get("raw_text") == row["target_text"],
            "operation_match": record.get("parsed_output", {}).get("operation") == row["target_operation"],
            "protocol_accepted": record.get("protocol_status") == "accepted",
            "context_aware_semantic_noop_assessed": False,
        }
    return {
        "denominator": len(cold),
        "strict_exact_target_text": sum(r["quality"]["strict_exact_target_text"] for r in cold),
        "operation_match": sum(r["quality"]["operation_match"] for r in cold),
        "protocol_accepted": sum(r["quality"]["protocol_accepted"] for r in cold),
        "strict_no_edit_target_denominator": sum(expected[r["row_id"]]["target_operation"] == "no_op" for r in cold),
        "strict_no_edit_target_exact": sum(r["quality"]["strict_exact_target_text"] and expected[r["row_id"]]["target_operation"] == "no_op" for r in cold),
        "context_aware_false_suggestion_rate": None,
        "limitation": "Strict TRAIN target metrics only; unchanged replacement can be a semantic no-op and is not classified here.",
    }


def run(args: argparse.Namespace) -> dict:
    packet_path = args.packet.resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_sha = sha256(packet_path)
    authorized_arms = packet.get("authorized_arms", list(packet.get("models", {})))
    if (packet.get("status") != "root_hashes_bound_no_model_load" or args.arm not in ARMS
            or args.arm not in authorized_arms or args.arm not in packet.get("models", {})):
        raise ValueError("unbound packet or invalid arm")
    arm = packet["models"][args.arm]
    binary = Path(packet["remote"]["binary"])
    assert_hash(binary, packet["remote"]["binary_sha256"], "server binary")
    assert_hash(Path(arm["path"]), arm["sha256"], f"{args.arm} model")
    if Path(arm["path"]).stat().st_size != arm["bytes"]:
        raise ValueError("model size mismatch")
    base = packet_path.parent
    panel_path, manifest_path, probe_path = (base / packet["panel"][key] for key in ("path", "manifest", "probe"))
    for path, field in ((panel_path, "panel_sha256"), (manifest_path, "manifest_sha256"), (probe_path, "probe_sha256")):
        assert_hash(path, packet["panel"][field], field)
    preserved_before = snapshot_preserved(packet)
    host, port = packet["remote"]["host_address"], packet["remote"]["port"]
    check_port_free(host, port)
    run_root = args.run_root.resolve()
    prefix = Path(packet["remote"]["run_root_prefix"]).resolve()
    if run_root.parent != prefix or run_root.exists():
        raise ValueError("run root must be one fresh direct child of the approved prefix")
    argv = [item.format(binary=str(binary), model_path=arm["path"]) for item in packet["launch"]["argv_template"]]
    if args.preflight_only:
        return {"status": "preflight_pass", "arm": args.arm, "packet_sha256": packet_sha,
                "argv": argv, "preserved_before": preserved_before}
    if args.admission is None:
        raise ValueError("root admission is required for a model load")
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    required = {"schema": "sepalith.r2.notebook-quant-admission.v1", "status": "admitted",
                "task": packet["task"], "arm": args.arm, "packet_sha256": packet_sha,
                "model_sha256": arm["sha256"], "resource": "cpu",
                "maximum_threads": packet["resource"]["maximum_threads"]}
    if any(admission.get(key) != value for key, value in required.items()):
        raise ValueError("root admission does not bind this packet/arm/model/resource")
    if admission.get("maximum_seconds") != packet["resource"]["per_arm_max_seconds"]:
        raise ValueError("root admission timebox mismatch")
    run_root.mkdir(parents=True)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(packet["launch"]["environment"])
    process = None
    log_handle = None
    telemetry = None
    stop = threading.Event()
    terminal = {"status": "failed", "server_pid": None}
    def stop_requested(signum, _frame):
        name = signal.Signals(signum).name
        raise RuntimeError(f"supervisor received {name}; owned cleanup required")
    prior_handlers = {sig: signal.getsignal(sig) for sig in OWNED_STOP_SIGNALS}
    for sig in OWNED_STOP_SIGNALS:
        signal.signal(sig, stop_requested)
    signal.alarm(packet["resource"]["per_arm_max_seconds"])
    try:
        log_handle = (run_root / "server.log").open("wb")
        process = subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        terminal["server_pid"] = process.pid
        try:
            server_start_ticks = Path(f"/proc/{process.pid}/stat").read_text().split()[21]
        except (OSError, IndexError):
            server_start_ticks = None
        launch = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arm": args.arm,
                  "supervisor_pid": os.getpid(), "server_pid": process.pid, "server_process_group": process.pid,
                  "server_start_ticks": server_start_ticks, "argv": argv, "packet_sha256": packet_sha,
                  "model": arm, "environment": packet["launch"]["environment"]}
        write_json(run_root / "launch.json", launch)
        telemetry = threading.Thread(target=telemetry_loop, args=(process.pid, run_root / "telemetry.jsonl", stop), daemon=True)
        telemetry.start()
        health = None
        ready_deadline = time.monotonic() + packet["launch"]["ready_timeout_seconds"]
        while time.monotonic() < ready_deadline and process.poll() is None:
            health = read_health(f"http://{host}:{port}{packet['launch']['health_path']}")
            if health and health[0] == 200:
                break
            time.sleep(.25)
        if not health or health[0] != 200:
            raise RuntimeError(f"server did not become healthy; exit={process.poll()}")
        write_json(run_root / "health.json", {"status": health[0], "body": health[1]})
        probe = load_probe(probe_path)
        rows, manifest = probe.load_panel(panel_path, manifest_path, packet["v1"]["rows"])
        requests = []
        for row in rows:
            for phase in packet["v1"]["phases"]:
                requests.append(probe.run_case(f"http://{host}:{port}", row, phase, 1,
                                               packet["v1"]["cap"], packet["v1"]["context"], packet["v1"]["deadline_ms"]))
        result = {"schema": "sepalith.r2.notebook-quant-arm.v1", "arm": args.arm, "model": arm,
                  "deadline_ms": packet["v1"]["deadline_ms"], "rows": len(rows), "requests": requests}
        result["quality"] = quality(rows, requests)
        write_json(run_root / packet["result_contract"]["five_second_results"], result)
        failed_ids = []
        for record in requests:
            if record.get("protocol_status") != "accepted" and record["row_id"] not in failed_ids:
                failed_ids.append(record["row_id"])
        diagnostic = []
        for row_id in failed_ids[:packet["v1"]["diagnostic_max_failed_rows"]]:
            row = next(row for row in rows if row["id"] == row_id)
            for phase in packet["v1"]["phases"]:
                diagnostic.append(probe.run_case(f"http://{host}:{port}", row, phase, 1,
                                                 packet["v1"]["cap"], packet["v1"]["context"],
                                                 packet["v1"]["diagnostic_deadline_ms"]))
        diagnostic_result = {"schema": "sepalith.r2.notebook-quant-diagnostic.v1", "arm": args.arm,
                             "deadline_ms": packet["v1"]["diagnostic_deadline_ms"],
                             "excluded_from_five_second_metrics": True, "requests": diagnostic}
        if diagnostic:
            diagnostic_result["quality"] = quality(rows, diagnostic)
        write_json(run_root / packet["result_contract"]["diagnostic_results"], diagnostic_result)
        terminal = {"status": "completed", "server_pid": process.pid, "five_second_requests": len(requests),
                    "diagnostic_requests": len(diagnostic), "elapsed_seconds_before_cleanup": time.monotonic() - started}
        return result
    except BaseException as exc:
        terminal["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            cleanup_owned_group(process)
        stop.set()
        if telemetry is not None:
            telemetry.join(timeout=2)
        if log_handle is not None:
            log_handle.close()
        server_pid = process.pid if process is not None else None
        terminal.update({"server_exit_code": process.returncode if process is not None else None,
                         "server_pid_exists": bool(server_pid and Path(f"/proc/{server_pid}").exists()),
                         "port_free_after": port_is_free(host, port),
                         "elapsed_seconds": time.monotonic() - started,
                         "preserved_after": snapshot_preserved(packet)})
        write_json(run_root / "terminal.json", terminal)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    print(json.dumps({"status": result["status"] if "status" in result else "completed", "arm": args.arm}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
