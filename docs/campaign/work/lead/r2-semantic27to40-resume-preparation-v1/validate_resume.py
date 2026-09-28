#!/usr/bin/env python3
"""Fail-closed preflight for the post-timeout semantic queue resume.

This helper is intentionally read-only.  It validates the immutable replay
and frozen semantic-worker pins, checks every existing child output, and
reports the exact missing shard set.  It never removes or rewrites an
incomplete temporary child; the operator must quarantine such a directory
before running the printed command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
REPLAY = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-independent-replay-v3/full-01"
)
OUTPUT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-semantic-streaming-queue-shards27to40-v1"
)
PREP = PLAN / "docs/campaign/work/lead/r2-semantic27to40-queue-preparation-v1"
V3 = PLAN / "docs/campaign/work/lead/r2-semantic-queue-root-launch-v3"
INDEX_SHA = "65637a9e05c66647de042d46f42bf9afa197a0b068f63680ec9f3ac0dfe922a1"
SOURCE_MANIFEST_SHA = "4321412035ca36dc9b5e0f95176142df040cebf3d59ef20a9c007a4f80fb400b"
WORKER_SHA = "69d999cd0b7309775435126bf8fdd5f54e24bf793b9ad200fb1b3efbb2379e77"
LAUNCHER_SHA = "fcce86cda22ca91a28efbb09332ce436f663b9fc535011a3bd1064b13e245971"
EXPECTED = list(range(27, 41))


class ResumeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResumeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256(canonical(sorted(ids)).encode("utf-8")).hexdigest()


def line_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"empty semantic row:{path}:{line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"semantic row is not object:{path}:{line_number}")
            rows.append(value)
    return rows


def pin(path: Path, expected: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing:{path}")
    digest = sha(path)
    if expected is not None:
        require(digest == expected, f"hash_mismatch:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def validate_replay() -> dict[str, Any]:
    index_path = REPLAY / "index/manifest.json"
    index_pin = pin(index_path, INDEX_SHA)
    index = json.loads(index_path.read_text())
    require(index.get("schema") == "sepalith.dat10.sourcewalk_raw_index.v3", "replay_index_schema")
    require(index.get("status") == "complete", "replay_index_not_complete")
    require(index.get("requested_shards") == list(range(41)), "replay_index_scope")
    entries = {item.get("shard"): item for item in index.get("index_files", [])}
    require(set(entries) == set(range(41)) and len(entries) == 41, "replay_index_missing_shards")
    exit_path = REPLAY / "controller-exit-code.txt"
    require(exit_path.read_text().strip() == "0", "replay_controller_not_successful")
    receipt_shards: list[int] = []
    for shard in range(41):
        receipt_path = REPLAY / "shards" / f"shard-{shard:04d}" / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("status") == "complete" and receipt.get("shard") == shard, f"replay_receipt:{shard}")
        require(receipt.get("rows") == entries[shard].get("rows"), f"replay_rows:{shard}")
        receipt_shards.append(shard)
    require(receipt_shards == list(range(41)), "replay_receipt_scope")
    return {
        "index": index_pin,
        "controller_exit_code": 0,
        "shards": 41,
        "rows": sum(entries[shard].get("rows", 0) for shard in range(41)),
        "all_receipts_complete": True,
    }


def validate_child(shard: int) -> dict[str, Any]:
    child = OUTPUT / f"shard-{shard:04d}"
    manifest_path = child / "manifest.json"
    output_path = child / "semantic-ledger.jsonl"
    manifest_pin = pin(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "sepalith.dat10.sourcewalk_roxy_semantic.v6", f"child_schema:{shard}")
    require(manifest.get("status") == "complete_review_only", f"child_status:{shard}")
    require(manifest.get("training_admission") is False, f"child_admission:{shard}")
    require(manifest.get("exact_id_closure") is True, f"child_id_closure:{shard}")
    output_decl = manifest.get("output")
    require(isinstance(output_decl, dict), f"child_output_decl:{shard}")
    require(output_decl.get("path") == str(output_path), f"child_output_path:{shard}")
    output_pin = pin(output_path, output_decl.get("sha256"))
    require(output_pin["bytes"] == output_decl.get("bytes"), f"child_output_bytes:{shard}")
    rows = line_rows(output_path)
    ids = [row.get("row_id") for row in rows]
    require(all(isinstance(value, str) and value for value in ids), f"child_row_ids:{shard}")
    require(len(ids) == len(set(ids)), f"child_duplicate_ids:{shard}")
    require(len(rows) == output_decl.get("rows") == manifest.get("rows"), f"child_row_count:{shard}")
    require(ids_sha(ids) == output_decl.get("row_ids_sha256"), f"child_id_digest:{shard}")
    require(manifest.get("output_binding") == output_decl, f"child_output_binding:{shard}")
    require(isinstance(manifest.get("source_provenance_binding_sha256"), str), f"child_source_binding:{shard}")
    require(isinstance(manifest.get("queued_ids_sha256"), str), f"child_queued_binding:{shard}")
    return {
        "shard": shard,
        "manifest": manifest_pin,
        "semantic_output": {**output_pin, "rows": len(rows), "row_ids_sha256": ids_sha(ids)},
        "source_provenance_binding_sha256": manifest["source_provenance_binding_sha256"],
        "queued_ids_sha256": manifest["queued_ids_sha256"],
    }


def active_queue_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "run_streaming_semantic_queue.py" in line or "r2-semantic27to40-queue-preparation-v1/run.py" in line
    ]


def command(missing: list[int]) -> list[str]:
    return [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "5400",
        "ionice",
        "-c3",
        "nice",
        "-n",
        "10",
        "taskset",
        "-c",
        "8,10",
        "/usr/bin/python3",
        "-B",
        str(V3 / "source/run_streaming_semantic_queue.py"),
        "--replay-root",
        str(REPLAY),
        "--output",
        str(OUTPUT),
        "--shards",
        ",".join(str(shard) for shard in missing),
        "--max-workers",
        "2",
    ]


def validate() -> dict[str, Any]:
    require(sha(V3 / "source-manifest.json") == SOURCE_MANIFEST_SHA, "semantic_source_manifest_pin")
    require(sha(V3 / "source/run_streaming_semantic_queue.py") == WORKER_SHA, "semantic_worker_pin")
    require(sha(PREP / "run.py") == LAUNCHER_SHA, "semantic_launcher_pin")
    replay = validate_replay()
    children = []
    for child_path in sorted(OUTPUT.glob("shard-*/manifest.json")):
        try:
            shard = int(child_path.parent.name.rsplit("-", 1)[1])
        except ValueError as error:
            raise ResumeError(f"unexpected_child_name:{child_path}") from error
        require(shard in EXPECTED, f"unexpected_child_shard:{shard}")
        children.append(validate_child(shard))
    completed = [item["shard"] for item in children]
    require(completed == sorted(set(completed)), "duplicate_or_unsorted_children")
    missing = [shard for shard in EXPECTED if shard not in completed]
    require(missing, "resume_has_no_missing_shards")
    temp_dirs = sorted(str(path) for path in OUTPUT.glob(".shard-*"))
    require(not temp_dirs, "unquarantined_incomplete_temp_dirs")
    processes = active_queue_processes()
    require(not processes, "semantic_queue_still_active")
    return {
        "schema": "sepalith.dat10.semantic27to40_resume_preflight.v1",
        "status": "ready_for_root_review",
        "training_admission": False,
        "replay": replay,
        "semantic": {
            "requested_shards": EXPECTED,
            "completed_shards": completed,
            "completed_rows": sum(item["semantic_output"]["rows"] for item in children),
            "missing_shards": missing,
            "children": children,
            "output": str(OUTPUT),
            "quarantined_temp_root": str(OUTPUT / "quarantine"),
        },
        "pins": {
            "source_manifest": pin(V3 / "source-manifest.json", SOURCE_MANIFEST_SHA),
            "worker": pin(V3 / "source/run_streaming_semantic_queue.py", WORKER_SHA),
            "launcher": pin(PREP / "run.py", LAUNCHER_SHA),
            "prior_terminal": pin(PREP / "terminal.json"),
        },
        "resume": {
            "timeout_seconds": 5400,
            "cpu_cores": [8, 10],
            "max_workers": 2,
            "nice": 10,
            "ionice_class": 3,
            "cuda": False,
            "cloud": False,
            "reuse_verified_children": True,
            "retry_only_missing": True,
            "command": command(missing),
        },
        "active_processes": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    value = validate()
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
