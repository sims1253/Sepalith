#!/usr/bin/env python3
"""Finalize a completed shard preparation after independent verification."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-semantic10952-preparation-v1"
VERIFY = PACKET / "verify.py"
OUTPUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/Semantic10952-preparation-v1")
INPUT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-semantic-streaming-queue-shards12plus-v1"
)
PYTHON = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python")
SHARDS = list(range(12, 27))
SUPPORTED = "semantic_supported_context_closure_root_review_required"
QUEUE_MANIFEST_SHA = "bf37575b2338fe4c5cfa1ff5ed022ce00f3e280bd91c93316ba94a8e2b6ad6a4"
EXPECTED_ROWS = 10952


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rows(path: Path) -> int:
    with path.open() as stream:
        return sum(1 for _ in stream)


def hold_counts(path: Path) -> dict:
    counts = collections.Counter()
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            assert item["silent_drop"] is False
            reason = item.get("reason")
            assert isinstance(reason, str) and reason
            counts[reason] += 1
    return dict(counts)


def inspect_shard(shard: int) -> dict:
    input_manifest_path = INPUT / f"shard-{shard:04d}/manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text())
    assert input_manifest["status"] == "complete_review_only"
    assert input_manifest["training_admission"] is False
    assert input_manifest["exact_id_closure"] is True
    binding = input_manifest["streaming_binding"]
    assert binding["shard"] == shard
    assert binding["queued_rows"] == input_manifest["rows"]
    assert binding["receipt_rows"] == binding["provenance_ledger"]["rows"]
    expected = input_manifest["status_counts"].get(SUPPORTED, 0)

    folder = OUTPUT / f"shard-{shard:04d}"
    report_path = folder / "preparation-manifest.json"
    report = json.loads(report_path.read_text())
    assert report["semantic_supported"] == expected
    assert report["exact_supported_id_accounting"] is True
    assert report["prediction_inputs"] == report["training_sidecar_rows"]
    assert report["prediction_inputs"] + report["preparation_holds"] == expected
    output_pins = {}
    for name in ("prediction-inputs.jsonl", "training-sidecar.jsonl", "preparation-holds.jsonl"):
        path = folder / name
        expected_output = report["outputs"][name]
        output_pins[name] = {
            **pin(path),
            "rows": rows(path),
        }
        assert output_pins[name]["sha256"] == expected_output["sha256"]
        assert output_pins[name]["bytes"] == expected_output["bytes"]
        assert output_pins[name]["rows"] == expected_output["rows"]
    return {
        "shard": shard,
        "input_manifest": pin(input_manifest_path),
        "input_semantic_rows": input_manifest["rows"],
        "supported_denominator": expected,
        "prepared_rows": report["prediction_inputs"],
        "hold_rows": report["preparation_holds"],
        "hold_reason_counts": hold_counts(folder / "preparation-holds.jsonl"),
        "preparation_manifest": pin(report_path),
        "outputs": output_pins,
    }


def main() -> int:
    assert OUTPUT.is_dir()
    assert PYTHON.is_file()
    queue_manifest = INPUT / "streaming-manifest.json"
    assert sha(queue_manifest) == QUEUE_MANIFEST_SHA
    queue = json.loads(queue_manifest.read_text())
    assert queue["status"] == "partial_review_only"
    assert queue["training_admission"] is False
    assert queue["requested_shards"] == SHARDS
    assert queue["semantic_rows"] == 27700

    failed_path = OUTPUT / "terminal.json"
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "failed_or_interrupted_partial"
    assert failed.get("error", "").startswith("AssertionError:independent_geometry_verify_failed:")
    failed_snapshot = PACKET / "attempts" / f"attempt-{failed['launch_pid']}" / "terminal.json"
    failed_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not failed_snapshot.exists():
        shutil.copy2(failed_path, failed_snapshot)
    failed_pin = pin(failed_snapshot)

    shard_results = [inspect_shard(shard) for shard in SHARDS]
    assert sum(item["supported_denominator"] for item in shard_results) == EXPECTED_ROWS
    assert sum(item["prepared_rows"] + item["hold_rows"] for item in shard_results) == EXPECTED_ROWS

    verify_log = OUTPUT / "independent-geometry-verify-final.log"
    command = [str(PYTHON), "-B", str(VERIFY), "--root", str(OUTPUT)]
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
    with verify_log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    assert result.returncode == 0, f"independent_geometry_verify_failed:{verify_log}"
    verify = json.loads(verify_log.read_text().strip())
    assert verify["status"] == "pass"
    assert verify["prepared_rows"] + verify["hold_rows"] == EXPECTED_ROWS
    assert verify["prediction_target_free"] is True
    assert verify["full_target_sidecar"] is True
    assert verify["exact_source_reapplication"] is True
    assert verify["training_admission"] is False

    launch_path = PACKET / "launch.json"
    launch = json.loads(launch_path.read_text())
    assert launch["training_admission"] is False
    terminal = {
        "schema": "sepalith.dat10.semantic10952_terminal.v2",
        "status": "prepared_not_training_admitted",
        "training_admission": False,
        "finished_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "launch_pid": failed["launch_pid"],
        "launch": pin(launch_path),
        "supersedes_failed_terminal": failed_pin,
        "shards": shard_results,
        "independent_geometry_verification": verify,
        "verify_log": pin(verify_log),
        "source": launch["source"],
        "queue_manifest": pin(queue_manifest),
        "accepted_scope_rows": EXPECTED_ROWS,
        "prepared_rows": sum(item["prepared_rows"] for item in shard_results),
        "hold_rows": sum(item["hold_rows"] for item in shard_results),
        "hold_reason_counts": verify["hold_reason_counts"],
        "exact_denominator_closure": True,
        "full_preedit_preserved": True,
        "complete_target_sidecar": True,
        "target_truncation": "none",
        "cuda_used": False,
        "review_only": True,
    }
    atomic_json(OUTPUT / "terminal.json", terminal)
    atomic_json(PACKET / "terminal.json", terminal)
    atomic_json(
        OUTPUT / "status.json",
        {**launch, **terminal, "status": terminal["status"]},
    )
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
