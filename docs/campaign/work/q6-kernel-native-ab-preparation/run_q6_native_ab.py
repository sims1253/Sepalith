#!/usr/bin/env python3
"""Root-owned, bounded baseline/candidate Q6 native probe capsule.

Importing this file does not load a model, start a server, import a framework,
or contact a remote host. The worker only runs the CPU policy tests; the root
operator runs this capsule after reviewing the two compiled binaries and
their build receipts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
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


SCHEMA = "run-05.q6-native-ab-preparation.v1"
SOURCE_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
SOURCE_MANIFEST_SHA256 = "b0e6fbb71b2528924480c6da34f9bd90c87f7b847c504b7e3ee8132fb446bc7e"
PATCH_SHA256 = "f9f57597ae119e433095c52e3de63b83a0cdbf472f982a123389256a01763310"
MODEL_SHA256 = "7ea2ddfd45dce35d5016af8f1d1ba14408d29764df5a90f3a0dfe6f886ecec3578"
PROBE_SHA256 = "f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7"
FIXTURE_SHA256 = "4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008"
FIXTURE_MANIFEST_SHA256 = "0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3"
EXPECTED_FIXTURE_SCHEMA = "run-01.native-prm03-train-fixture.v1"
PROFILE = {
    "context": 4096,
    "batch": 256,
    "ubatch": 256,
    "threads": 6,
    "threads_http": 2,
    "parallel": 1,
    "ngl": 99,
    "cap": 192,
}
TRACE_ENV = "SEPALITH_VK_TRACE"
REMOVED_ENV = (
    "GGML_BACKEND_PATH",
    "GGML_CUDA_GRAPH_OPT",
    "GGML_VK_Q6K_EXACT_DIV_POISON",
    "GGML_VK_FORCE_MMVQ",
    "GGML_VK_DISABLE_MMVQ",
)
MEMORY_FLOOR_MIB = 2048
ARM_ALIASES = {"baseline": "baseline", "candidate": "ngram-mod"}


class CapsuleError(RuntimeError):
    """A fail-closed preflight, supervision, or parity error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapsuleError(f"invalid JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CapsuleError(f"receipt is not an object: {path}")
    return value


def require_regular(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CapsuleError(f"{label} is absent: {path}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise CapsuleError(f"{label} must be a regular, non-symlink file: {path}")
    return resolved


def verify_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected:
        raise CapsuleError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def available_memory_mib(meminfo: str | None = None) -> int:
    text = meminfo
    if text is None:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) // 1024
    raise CapsuleError("MemAvailable is missing from /proc/meminfo")


def require_memory_floor() -> int:
    available = available_memory_mib()
    if available < MEMORY_FLOOR_MIB:
        raise CapsuleError(
            f"host memory floor is not met: {available} MiB available, "
            f"{MEMORY_FLOOR_MIB} MiB required"
        )
    return available


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def profile_argv(binary: Path, model: Path, port: int) -> list[str]:
    return [
        str(binary), "-m", str(model), "--host", "127.0.0.1",
        "--port", str(port), "-t", "6", "-tb", "6", "--threads-http", "2",
        "--parallel", "1", "-c", "4096", "-b", "256", "-ub", "256",
        "-ngl", "99", "-lv", "4",
    ]


def child_environment(*, trace: bool, base: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "6"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    removed: list[str] = []
    for name in REMOVED_ENV:
        if name in env:
            removed.append(name)
            env.pop(name, None)
    if trace:
        env[TRACE_ENV] = "1"
    else:
        env.pop(TRACE_ENV, None)
        removed.append(TRACE_ENV)
    return env, removed


def load_probe(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("q6_native_probe_v1", path)
    if spec is None or spec.loader is None:
        raise CapsuleError(f"cannot load probe module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_artifact(binary: Path, artifacts_path: Path, arm: str) -> dict[str, Any]:
    receipt = read_json(artifacts_path)
    if receipt.get("status") != "compiled_only" or receipt.get("arm") != arm:
        raise CapsuleError(f"{arm} artifact receipt is not compiled_only for {arm}")
    entries = receipt.get("artifacts")
    if not isinstance(entries, list):
        raise CapsuleError(f"{arm} artifact receipt lacks artifacts")
    binary_resolved = require_regular(binary, f"{arm} binary")
    matching = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            candidate = Path(str(entry["path"])).resolve(strict=True)
        except (KeyError, OSError):
            continue
        if candidate == binary_resolved:
            matching.append(entry)
    if len(matching) != 1:
        raise CapsuleError(f"{arm} binary is not uniquely bound by {artifacts_path}")
    entry = matching[0]
    expected_bytes = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_bytes, int) or not isinstance(expected_sha, str):
        raise CapsuleError(f"{arm} binary artifact entry is incomplete")
    if binary_resolved.stat().st_size != expected_bytes:
        raise CapsuleError(f"{arm} binary byte count differs from artifact receipt")
    actual_sha = sha256_file(binary_resolved)
    if actual_sha != expected_sha:
        raise CapsuleError(f"{arm} binary SHA differs from artifact receipt")
    return {
        "arm": arm,
        "artifact_receipt": str(artifacts_path),
        "artifact_receipt_sha256": sha256_file(artifacts_path),
        "binary": str(binary_resolved),
        "bytes": expected_bytes,
        "sha256": actual_sha,
        "cache_sha256": receipt.get("cache_sha256"),
    }


def verify_build_evidence(evidence_root: Path) -> dict[str, Any]:
    terminal_path = require_regular(evidence_root / "terminal.json", "build terminal")
    terminal = read_json(terminal_path)
    if terminal.get("status") != "compiled_only" or set(terminal.get("completed_arms", [])) != {"baseline", "candidate"}:
        raise CapsuleError("build evidence is not a completed two-arm compiled_only result")
    identities: dict[str, Any] = {}
    for arm in ("baseline", "candidate"):
        identity_path = require_regular(evidence_root / f"{arm}-source-identity.json", f"{arm} source identity")
        identity = read_json(identity_path)
        if identity.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
            raise CapsuleError(f"{arm} source manifest identity differs from pinned closure")
        if arm == "baseline" and identity.get("q6_shader_sha256") != "bc785a2457aa5b04d416e18f5ae2304da267e35cea7b54ac67b5d20987e450af":
            raise CapsuleError("baseline Q6 shader identity differs from the pinned baseline")
        if arm == "candidate":
            changes = identity.get("changes", [])
            if not isinstance(changes, list) or not any("reviewed Q6 patch" in str(item) for item in changes):
                raise CapsuleError("candidate source receipt does not identify the reviewed Q6 patch")
        identities[arm] = {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
            "source_manifest_sha256": identity["source_manifest_sha256"],
            "q6_shader_sha256": identity.get("q6_shader_sha256"),
            "changes": identity.get("changes"),
        }
    return {
        "root": str(evidence_root),
        "terminal": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "identities": identities,
        "artifacts": {
            arm: {
                "path": str(evidence_root / f"{arm}-artifacts.json"),
                "sha256": sha256_file(evidence_root / f"{arm}-artifacts.json"),
            }
            for arm in ("baseline", "candidate")
        },
    }


def wait_health(proc: subprocess.Popen[bytes], url: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise CapsuleError(f"server exited before health check (exit {proc.returncode})")
        require_memory_floor()
        try:
            with urlopen(url + "/health", timeout=1.0) as response:
                if 200 <= int(response.status) < 300:
                    return
        except (OSError, URLError):
            time.sleep(0.5)
    raise CapsuleError("server health deadline exceeded")


def stop_group(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
    return {"pid": proc.pid, "exit_code": proc.returncode, "gone": proc.poll() is not None}


def normalize_probe(raw_path: Path, normalized_path: Path, arm: str) -> dict[str, Any]:
    result = read_json(raw_path)
    expected_alias = ARM_ALIASES[arm]
    if result.get("arm") != expected_alias:
        raise CapsuleError(f"probe arm alias mismatch for {arm}: {result.get('arm')!r}")
    normalized = dict(result)
    normalized["arm"] = arm
    normalized["capsule_arm"] = arm
    normalized["probe_cli_arm_alias"] = expected_alias
    write_json(normalized_path, normalized)
    return normalized


def run_probe_process(
    probe_path: Path,
    fixture: Path,
    manifest: Path,
    url: str,
    arm: str,
    phase: str,
    phase_dir: Path,
    deadline: float,
) -> tuple[int, dict[str, Any]]:
    raw_path = phase_dir / "probe-cli.json"
    stdout_path = phase_dir / "probe-client.log"
    deadline_ms = 60000 if phase == "trace" else 5000
    command = [
        sys.executable, "-B", str(probe_path),
        "--fixture", str(fixture), "--manifest", str(manifest), "--url", url,
        "--arm", ARM_ALIASES[arm], "--cap", str(PROFILE["cap"]),
        "--context", str(PROFILE["context"]), "--reps", "1",
        "--user-deadline-ms", str(deadline_ms),
        "--diagnostic-timeout-ms", "60000", "--out", str(raw_path),
    ]
    with stdout_path.open("x", encoding="utf-8") as stdout:
        proc = subprocess.Popen(
            command, stdout=stdout, stderr=subprocess.STDOUT,
            cwd=str(phase_dir), start_new_session=True,
        )
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    stop_group(proc)
                    return 124, {"status": "deadline", "command": command, "raw_path": str(raw_path)}
                require_memory_floor()
                time.sleep(0.5)
        except Exception:
            stop_group(proc)
            raise
    if proc.returncode != 0:
        return int(proc.returncode), {
            "status": "probe_process_failed",
            "command": command,
            "raw_path": str(raw_path),
        }
    if not raw_path.is_file():
        return 1, {"status": "probe_output_missing", "command": command}
    normalized_path = phase_dir / "probe.json"
    result = normalize_probe(raw_path, normalized_path, arm)
    return 0, {
        "status": result.get("status"),
        "command": command,
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "normalized_path": str(normalized_path),
        "normalized_sha256": sha256_file(normalized_path),
        "probe_status": result.get("status"),
        "requests": result.get("summary", {}).get("requests"),
        "accepted_requests": result.get("summary", {}).get("accepted_requests"),
    }


def run_phase(
    *,
    arm: str,
    phase: str,
    binary: Path,
    model: Path,
    fixture: Path,
    manifest: Path,
    probe_path: Path,
    run_root: Path,
    port: int,
    phase_deadline: float,
) -> dict[str, Any]:
    phase_dir = run_root / arm / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    if not port_is_free(port):
        raise CapsuleError(f"port {port} is not free for {arm}/{phase}")
    env, removed = child_environment(trace=phase == "trace")
    command = profile_argv(binary, model, port)
    server_log = phase_dir / "server.log"
    with server_log.open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env,
            cwd=str(phase_dir), start_new_session=True,
        )
    launch = {
        "schema_version": SCHEMA,
        "arm": arm,
        "phase": phase,
        "pid": proc.pid,
        "port": port,
        "argv": command,
        "environment_policy": {
            "trace_enabled": phase == "trace",
            "removed_graph_or_poison_variables": removed,
            "cuda_visible_devices": "",
            "omp_num_threads": "6",
            "openblas_num_threads": "1",
            "mkl_num_threads": "1",
        },
        "server_profile": PROFILE,
        "timing_claim": phase == "timing",
    }
    write_json(phase_dir / "launch.json", launch)
    url = f"http://127.0.0.1:{port}"
    result: dict[str, Any] = {
        "arm": arm, "phase": phase, "port": port,
        "server_log": str(server_log), "launch": str(phase_dir / "launch.json"),
        "timing_claim": phase == "timing",
    }
    try:
        wait_health(proc, url, phase_deadline)
        code, probe_result = run_probe_process(
            probe_path, fixture, manifest, url, arm, phase, phase_dir, phase_deadline,
        )
        result["probe"] = probe_result
        if code != 0 or probe_result.get("probe_status") != "completed":
            raise CapsuleError(f"{arm}/{phase} probe did not complete: {probe_result}")
        result["status"] = "completed"
        return result
    finally:
        result["server"] = stop_group(proc)
        if "status" not in result:
            result["status"] = "failed"
        write_json(phase_dir / "terminal.json", result)


STABLE_FIELDS = (
    "protocol_status", "returned_token_ids", "raw_text", "raw_text_sha256",
    "cap", "context_size", "cap_status", "stop", "stop_type",
    "stopping_word", "truncated", "tokens_evaluated", "tokens_predicted",
    "tokens_cached", "n_tokens_cached", "n_prompt_tokens_cache",
    "cache_prompt", "prompt_token_count", "prompt_ids_sha256",
    "prompt_sha256", "prompt_id_status", "parsed_output", "protocol_checks",
)


def compare_stable_fields(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = read_json(left_path)
    right = read_json(right_path)

    def index(value: dict[str, Any]) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
        records = value.get("requests")
        if not isinstance(records, list):
            raise CapsuleError(f"probe result has no requests: {value}")
        output: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise CapsuleError("probe request is not an object")
            key = (record.get("row_id"), record.get("phase"), record.get("rep"))
            if key in output:
                raise CapsuleError(f"duplicate probe key: {key!r}")
            output[key] = record
        return output

    a, b = index(left), index(right)
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(a) | set(b), key=repr):
        if key not in a or key not in b:
            mismatches.append({"key": list(key), "field": "record", "reason": "missing_pair"})
            continue
        for field in STABLE_FIELDS:
            if a[key].get(field) != b[key].get(field):
                mismatches.append({
                    "key": list(key), "field": field,
                    "left": a[key].get(field), "right": b[key].get(field),
                })
    return {
        "status": "pass" if not mismatches and len(a) == len(b) else "fail",
        "left": str(left_path), "right": str(right_path),
        "paired_requests": min(len(a), len(b)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "excluded_timing_fields": [
            "combined_case_wall_ms", "tokenize_wall_ms",
            "completion_wall_ms", "ttft_ms", "timings",
        ],
    }


def compare_phases(probe_path: Path, run_root: Path, phase: str) -> dict[str, Any]:
    baseline = run_root / "baseline" / phase / "probe.json"
    candidate = run_root / "candidate" / phase / "probe.json"
    helper_comparison = probe_path.compare_results(baseline, candidate)
    stable = compare_stable_fields(baseline, candidate)
    return {
        "phase": phase,
        "status": "pass" if helper_comparison.get("status") == "pass" and stable["status"] == "pass" else "fail",
        "helper_exact_output_comparison": helper_comparison,
        "stable_contract_comparison": stable,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--baseline-artifacts", type=Path, required=True)
    parser.add_argument("--candidate-artifacts", type=Path, required=True)
    parser.add_argument("--build-evidence-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-probe", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=18401)
    parser.add_argument("--overall-deadline-seconds", type=int, default=1200)
    parser.add_argument("--phase-deadline-seconds", type=int, default=240)
    parser.add_argument("--trace-only", action="store_true",
                        help="run one trace phase per arm and omit timing phases")
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace, probe: Any) -> dict[str, Any]:
    if args.model_sha256 != MODEL_SHA256:
        raise CapsuleError("model SHA argument must equal the pinned Q6_K model identity")
    if args.overall_deadline_seconds <= 0 or args.overall_deadline_seconds > 1200:
        raise CapsuleError("overall deadline must be in 1..1200 seconds")
    if args.phase_deadline_seconds <= 0:
        raise CapsuleError("phase deadline must be positive")
    if args.run_root.exists():
        raise CapsuleError(f"run root already exists; refusing overwrite: {args.run_root}")
    if args.base_port < 1024 or args.base_port > 64000 or args.base_port + 3 > 65535:
        raise CapsuleError("base port does not leave four valid ports")
    ports = list(range(args.base_port, args.base_port + 4))
    if any(not port_is_free(port) for port in ports):
        raise CapsuleError(f"one or more required ports are already occupied: {ports}")
    evidence = verify_build_evidence(args.build_evidence_root)
    binary_receipts = {
        "baseline": verify_artifact(args.baseline_binary, args.baseline_artifacts, "baseline"),
        "candidate": verify_artifact(args.candidate_binary, args.candidate_artifacts, "candidate"),
    }
    probe_identity = verify_hash(require_regular(args.runtime_probe, "runtime native probe"), PROBE_SHA256, "runtime native probe")
    fixture_identity = verify_hash(require_regular(args.fixture, "TRAIN fixture"), FIXTURE_SHA256, "TRAIN fixture")
    manifest_identity = verify_hash(require_regular(args.manifest, "fixture manifest"), FIXTURE_MANIFEST_SHA256, "fixture manifest")
    try:
        rows, manifest = probe.load_fixture(args.fixture, args.manifest)
    except Exception as exc:
        raise CapsuleError(f"native probe fixture admission failed: {exc}") from exc
    if manifest.get("schema_version") != EXPECTED_FIXTURE_SCHEMA:
        raise CapsuleError("fixture schema differs from the pinned TRAIN schema")
    if len(rows) != 4:
        raise CapsuleError(f"Q6 capsule expects four selected TRAIN rows, got {len(rows)}")
    model_identity = verify_hash(args.model, MODEL_SHA256, "Q6_K model")
    return {
        "schema_version": SCHEMA,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(args.run_root),
        "build_evidence": evidence,
        "source": {
            "commit": SOURCE_COMMIT,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "candidate_patch_sha256": PATCH_SHA256,
        },
        "binaries": binary_receipts,
        "model": model_identity,
        "fixture": {
            **fixture_identity, "manifest": manifest_identity,
            "schema_version": manifest.get("schema_version"),
            "row_ids": [row["id"] for row in rows],
            "prompt_token_counts": [row["prompt_token_count"] for row in rows],
            "selected_rows": len(rows),
        },
        "profile": PROFILE,
        "ports": ports,
        "memory_floor_mib": MEMORY_FLOOR_MIB,
        "overall_deadline_seconds": args.overall_deadline_seconds,
        "phase_deadline_seconds": args.phase_deadline_seconds,
        "trace_only": bool(args.trace_only),
        "probe_import": "framework-free runtime_native_probe.py; no server/model load at import",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        probe = load_probe(args.runtime_probe)
        plan = preflight(args, probe)
        args.run_root.mkdir(parents=True, exist_ok=False)
        write_json(args.run_root / "preflight.json", plan)
        started = time.monotonic()
        overall_deadline = started + args.overall_deadline_seconds
        phases = ["trace"] if args.trace_only else ["trace", "timing"]
        outputs: dict[str, Any] = {
            "schema_version": SCHEMA,
            "arms": {"baseline": {}, "candidate": {}},
            "comparisons": [],
        }
        for arm, binary in (
            ("baseline", args.baseline_binary.resolve()),
            ("candidate", args.candidate_binary.resolve()),
        ):
            for phase in phases:
                if time.monotonic() >= overall_deadline:
                    raise CapsuleError("overall deadline reached before all phases")
                phases_left = (
                    (2 * len(phases))
                    - len(outputs["arms"]["baseline"])
                    - len(outputs["arms"]["candidate"])
                )
                reserve = max(0, phases_left - 1) * args.phase_deadline_seconds
                phase_deadline = min(
                    overall_deadline - reserve,
                    time.monotonic() + args.phase_deadline_seconds,
                )
                outputs["arms"][arm][phase] = run_phase(
                    arm=arm, phase=phase, binary=binary, model=args.model.resolve(),
                    fixture=args.fixture.resolve(), manifest=args.manifest.resolve(),
                    probe_path=args.runtime_probe.resolve(), run_root=args.run_root,
                    port=args.base_port + (0 if arm == "baseline" else 2) + (0 if phase == "trace" else 1),
                    phase_deadline=phase_deadline,
                )
        for phase in phases:
            comparison = compare_phases(probe, args.run_root, phase)
            outputs["comparisons"].append(comparison)
            write_json(args.run_root / f"comparison-{phase}.json", comparison)
            if comparison["status"] != "pass":
                raise CapsuleError(f"{phase} baseline/candidate parity failed")
        outputs["status"] = "completed"
        outputs["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(args.run_root / "summary.json", outputs)
        write_json(args.run_root / "terminal.json", {
            "schema_version": SCHEMA, "status": "completed",
            "elapsed_seconds": outputs["elapsed_seconds"],
            "trace_only": bool(args.trace_only),
            "acceptance": "mechanical native wire/output parity only; no quality or latency promotion",
        })
        return 0
    except Exception as exc:
        if args.run_root.exists():
            try:
                write_json(args.run_root / "terminal.json", {
                    "schema_version": SCHEMA, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "acceptance": "no Q6 promotion or numerical claim",
                })
            except FileExistsError:
                pass
        print(f"q6 native A/B capsule failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
