#!/usr/bin/env python3
"""Run the reviewed target-free semantic preparer over shards 12--26."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
SOURCE = PLAN / "docs/campaign/work/lead/r2-semantic763-context-admission-v1"
PREPARER = SOURCE / "prepare_inputs.py"
PREPARER_ARTIFACT_MANIFEST = SOURCE / "artifact-manifest.json"
INPUT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-semantic-streaming-queue-shards12plus-v1"
)
OUTPUT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic10952-preparation-v1"
)
PACKET = PLAN / "docs/campaign/work/lead/r2-semantic10952-preparation-v1"
RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-semantic10952-preparation.json"
PYTHON = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python")
SHARDS = list(range(12, 27))
SUPPORTED = "semantic_supported_context_closure_root_review_required"
QUEUE_MANIFEST_SHA = "bf37575b2338fe4c5cfa1ff5ed022ce00f3e280bd91c93316ba94a8e2b6ad6a4"
SOURCE_MANIFEST = SOURCE / "source-manifest.json"
SOURCE_MANIFEST_SHA = "9558a835154f518302e75e3308da8b68659eb8a94009c3020bcb2d0f0fd1ce5c"
PREPARER_ARTIFACT_MANIFEST_SHA = "681d354775fa95aa88830855548e74f51af94c3dc60171828f6c1d09ac07bb60"
TIME_LIMIT = 2700


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_pin_check() -> dict:
    assert SOURCE_MANIFEST.is_file()
    assert sha(SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    assert manifest["status"] == "frozen_review_only"
    for item in manifest["files"]:
        path = SOURCE / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha(path) == item["sha256"], path
    assert PREPARER_ARTIFACT_MANIFEST.is_file()
    assert sha(PREPARER_ARTIFACT_MANIFEST) == PREPARER_ARTIFACT_MANIFEST_SHA
    preparer_manifest = json.loads(PREPARER_ARTIFACT_MANIFEST.read_text())
    assert preparer_manifest["status"] == "frozen_preparation"
    preparer_item = next(item for item in preparer_manifest["files"] if item["path"] == "prepare_inputs.py")
    assert PREPARER.stat().st_size == preparer_item["bytes"]
    assert sha(PREPARER) == preparer_item["sha256"]
    return {
        "path": str(SOURCE_MANIFEST),
        "bytes": SOURCE_MANIFEST.stat().st_size,
        "sha256": SOURCE_MANIFEST_SHA,
        "preparer_artifact_manifest": {
            "path": str(PREPARER_ARTIFACT_MANIFEST),
            "bytes": PREPARER_ARTIFACT_MANIFEST.stat().st_size,
            "sha256": PREPARER_ARTIFACT_MANIFEST_SHA,
        },
        "preparer": pin(PREPARER),
    }


def input_spec(shard: int) -> tuple[Path, dict, dict]:
    folder = INPUT / f"shard-{shard:04d}"
    manifest_path = folder / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "sepalith.dat10.sourcewalk_roxy_semantic.v6"
    assert manifest["status"] == "complete_review_only"
    assert manifest["training_admission"] is False
    assert manifest["exact_id_closure"] is True
    binding = manifest["streaming_binding"]
    assert binding["shard"] == shard
    assert binding["queued_rows"] == manifest["rows"]
    assert binding["receipt_rows"] == binding["provenance_ledger"]["rows"]
    expected = manifest["status_counts"].get(SUPPORTED, 0)
    records = {
        "semantic-manifest": {"path": manifest_path, "sha256": sha(manifest_path)},
        "semantic-ledger": manifest["output_binding"],
        "provenance-ledger": binding["provenance_ledger"],
        "candidate-packets": binding["candidate_packet"],
    }
    for record in records.values():
        path = Path(record["path"])
        assert path.is_file(), path
        if "bytes" in record:
            assert path.stat().st_size == record["bytes"], path
    return manifest_path, manifest, records


def output_pin(path: Path, rows: int) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "rows": rows}


def hold_counts(path: Path) -> dict:
    counts = collections.Counter()
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            assert item["silent_drop"] is False
            counts[item.get("reason", "missing_reason")] += 1
    return dict(counts)


def prepare_shard(shard: int, deadline: float) -> dict:
    manifest_path, manifest, records = input_spec(shard)
    destination = OUTPUT / f"shard-{shard:04d}"
    assert not destination.exists(), destination
    command = [str(PYTHON), "-B", str(PREPARER)]
    for name, record in records.items():
        command.extend([f"--{name}", str(record["path"]), f"--expected-{name}-sha256", record["sha256"]])
    command.extend(["--output", str(destination)])
    log = OUTPUT / f"shard-{shard:04d}.log"
    remaining = max(1, int(deadline - time.monotonic()))
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=remaining)
    assert result.returncode == 0, f"preparer_failed:{shard}:{result.returncode}; inspect {log}"
    report_path = destination / "preparation-manifest.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    expected = manifest["status_counts"].get(SUPPORTED, 0)
    assert report["semantic_supported"] == expected
    assert report["exact_supported_id_accounting"] is True
    outputs = {}
    for name in ("prediction-inputs.jsonl", "training-sidecar.jsonl", "preparation-holds.jsonl"):
        output_path = destination / name
        outputs[name] = output_pin(output_path, report["outputs"][name]["rows"])
        assert outputs[name]["sha256"] == report["outputs"][name]["sha256"]
        assert outputs[name]["bytes"] == report["outputs"][name]["bytes"]
    return {
        "shard": shard,
        "input_manifest": pin(manifest_path),
        "input_semantic_rows": manifest["rows"],
        "supported_denominator": expected,
        "prepared_rows": report["prediction_inputs"],
        "hold_rows": report["preparation_holds"],
        "hold_reason_counts": hold_counts(destination / "preparation-holds.jsonl"),
        "preparation_manifest": pin(report_path),
        "outputs": outputs,
        "elapsed_log": str(log),
    }


def main() -> int:
    assert not OUTPUT.exists(), "fresh_output_required"
    assert PYTHON.is_file(), PYTHON
    started = time.monotonic()
    source_manifest_pin = source_pin_check()
    queue_manifest = INPUT / "streaming-manifest.json"
    assert sha(queue_manifest) == QUEUE_MANIFEST_SHA
    queue = json.loads(queue_manifest.read_text())
    assert queue["status"] == "partial_review_only"
    assert queue["training_admission"] is False
    assert queue["requested_shards"] == SHARDS
    assert queue["semantic_rows"] == 27700
    OUTPUT.mkdir(parents=True)
    launch = {
        "schema": "sepalith.dat10.semantic10952_launch.v1",
        "status": "running",
        "training_admission": False,
        "started_at_utc": now(),
        "launch_pid": os.getpid(),
        "argv": sys.argv,
        "python": str(PYTHON),
        "cpu_policy": {"cores": [8, 10], "cuda": False, "cloud": False, "nice": 10, "ionice_class": 3},
        "time_limit_seconds": TIME_LIMIT,
        "source": source_manifest_pin,
        "queue_manifest": pin(queue_manifest),
        "queue_manifest_status": queue["status"],
        "input_shards": SHARDS,
        "output": str(OUTPUT),
        "reviewed_preparer": pin(PREPARER),
    }
    atomic_json(PACKET / "launch.json", launch)
    atomic_json(OUTPUT / "status.json", launch)
    results = []
    terminal = None
    try:
        deadline = started + TIME_LIMIT
        for shard in SHARDS:
            result = prepare_shard(shard, deadline)
            results.append(result)
            atomic_json(
                OUTPUT / "status.json",
                {**launch, "status": "running", "updated_at_utc": now(), "completed": results},
            )
        assert sum(x["supported_denominator"] for x in results) == 10952
        assert sum(x["prepared_rows"] + x["hold_rows"] for x in results) == 10952
        verify_log = OUTPUT / "independent-geometry-verify.log"
        remaining = max(1, int(deadline - time.monotonic()))
        command = [str(PYTHON), "-B", str(PACKET / "verify.py"), "--root", str(OUTPUT)]
        with verify_log.open("x") as stream:
            verify_result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
        assert verify_result.returncode == 0, f"independent_geometry_verify_failed:{verify_log}"
        verify = json.loads(verify_log.read_text().strip())
        assert verify["prepared_rows"] + verify["hold_rows"] == 10952
        terminal = {
            "schema": "sepalith.dat10.semantic10952_terminal.v1",
            "status": "prepared_not_training_admitted",
            "training_admission": False,
            "finished_at_utc": now(),
            "elapsed_seconds": time.monotonic() - started,
            "launch_pid": os.getpid(),
            "shards": results,
            "independent_geometry_verification": verify,
            "verify_log": pin(verify_log),
            "source_manifest": source_manifest_pin,
            "queue_manifest": pin(queue_manifest),
            "accepted_scope_rows": 10952,
            "prepared_rows": sum(x["prepared_rows"] for x in results),
            "hold_rows": sum(x["hold_rows"] for x in results),
            "exact_denominator_closure": True,
            "cuda_used": False,
        }
    except BaseException as exc:
        terminal = {
            "schema": "sepalith.dat10.semantic10952_terminal.v1",
            "status": "failed_or_interrupted_partial",
            "training_admission": False,
            "finished_at_utc": now(),
            "elapsed_seconds": time.monotonic() - started,
            "launch_pid": os.getpid(),
            "shards": results,
            "error": type(exc).__name__ + ":" + str(exc),
            "accepted_scope_rows": 10952,
            "prepared_rows": sum(x["prepared_rows"] for x in results),
            "hold_rows": sum(x["hold_rows"] for x in results),
            "exact_denominator_closure": False,
            "cuda_used": False,
        }
        raise
    finally:
        if terminal is not None:
            atomic_json(OUTPUT / "terminal.json", terminal)
            atomic_json(PACKET / "terminal.json", terminal)
            atomic_json(OUTPUT / "status.json", {**launch, **terminal, "status": terminal["status"], "updated_at_utc": now()})
    assert terminal["status"] == "prepared_not_training_admitted"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
