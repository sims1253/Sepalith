#!/usr/bin/env python3
"""Fail-closed semantic queue launcher for replay shards 27--40.

The default mode only performs intake planning.  ``--execute`` is required for
the lead-owned classifier launch after every replay shard has a verified
terminal receipt.  Existing semantic children are skipped only when their
source binding, output bytes/hash, and complete row-ID digest all match the
current replay inputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-semantic27to40-queue-preparation-v1"
V3 = PLAN / "docs/campaign/work/lead/r2-semantic-queue-root-launch-v3"
REPLAY_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-independent-replay-v3/full-01"
)
OUTPUT_DEFAULT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-semantic-streaming-queue-shards27to40-v1"
)
BASE = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT10-novel-v1/source-walk-shards-v1"
)
GLOBAL = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
CPT = PLAN / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
DRIVER = PLAN / "docs/campaign/work/lead/r2-sourcewalk-independent-replay-v3/full_replay.py"
ANALYZER = V3 / "source/analyze_semantics.py"
SCOPE = V3 / "source/semantic_scope.R"
NAMESPACE = V3 / "source/namespace_scope.R"
QUEUE_RUNNER = V3 / "source/run_streaming_semantic_queue.py"
SOURCE_MANIFEST = V3 / "source-manifest.json"
PYTHON = Path("/usr/bin/python3")
SELECTED = list(range(27, 41))
PRIOR = list(range(27))
QUEUED = "provenance_pass_semantic_analyzer_queued"
DRIVER_SHA = "63d165ae1488c1f38174f7e24f68a51e4c80c2d48e5aec44a0446fa062a61f7f"
INDEX_SHA = "65637a9e05c66647de042d46f42bf9afa197a0b068f63680ec9f3ac0dfe922a1"
GLOBAL_SHA = "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"
CPT_SHA = "6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06"
HOLD_SHA = "6a626df3c55b6bc1907eb3b64f8e8fec50095f07db47db0d796f17e658dda751"
STRICT_SHA = "7f042596a67ec9983f6f409a8913f4bc8df3361548a1e91fb77c9f914dafcf37"
LICENSE_SHA = "e02d588ac77d3a8bf8710217c8d1c678e730681de998932219afe75a2709dd80"
SOURCE_MANIFEST_SHA = "4321412035ca36dc9b5e0f95176142df040cebf3d59ef20a9c007a4f80fb400b"
TIME_LIMIT = 1500


class IntakeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntakeError(message)


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


def pin(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def stable_pin(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict:
    require(path.is_file(), f"missing:{path}")
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"changed_during_hash:{path}",
    )
    require(digest == expected_sha, f"hash_mismatch:{path}")
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, f"bytes_mismatch:{path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_code_pin() -> dict:
    require(SOURCE_MANIFEST.is_file(), "semantic_source_manifest_missing")
    stable_pin(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA)
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    require(manifest.get("schema") == "sepalith.root-semantic-queue-source.v3", "semantic_source_schema")
    for item in manifest["files"]:
        stable_pin(V3 / item["path"], item["sha256"], item["bytes"])
    require(DRIVER.is_file() and sha(DRIVER) == DRIVER_SHA, "replay_driver_pin")
    require(sha(GLOBAL) == GLOBAL_SHA, "global_split_pin")
    require(sha(CPT) == CPT_SHA, "cpt_partition_pin")
    code = {
        "analyzer_sha256": sha(ANALYZER),
        "scope_helper_sha256": sha(SCOPE),
        "namespace_helper_sha256": sha(NAMESPACE),
    }
    return {
        "source_manifest": pin(SOURCE_MANIFEST),
        "source_files": manifest["files"],
        "driver": pin(DRIVER),
        "global_split": pin(GLOBAL),
        "cpt_partition": pin(CPT),
        "fixed_review_bindings": {
            "hold_ledger_sha256": HOLD_SHA,
            "strict_validator_sha256": STRICT_SHA,
            "license_parser_sha256": LICENSE_SHA,
        },
        "semantic_code": code,
        "queue_runner": pin(QUEUE_RUNNER),
    }


def load_index(replay: Path) -> tuple[Path, dict, dict[int, dict], str]:
    path = replay / "index/manifest.json"
    stable_pin(path, INDEX_SHA)
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == "sepalith.dat10.sourcewalk_raw_index.v3", "replay_index_schema")
    require(manifest.get("status") == "complete", "replay_index_not_complete")
    require(manifest.get("requested_shards") == list(range(41)), "replay_index_scope")
    require(manifest.get("driver_sha256") == DRIVER_SHA, "replay_index_driver")
    entries = {item.get("shard"): item for item in manifest.get("index_files", [])}
    require(len(entries) == len(manifest.get("index_files", [])), "replay_index_duplicate_shards")
    require(set(entries) == set(range(41)), "replay_index_missing_shards")
    return path, manifest, entries, INDEX_SHA


def validate_replay_shard(
    replay: Path, shard: int, index: dict, entries: dict[int, dict], index_sha: str
) -> dict:
    require(shard in entries, f"replay_index_shard_missing:{shard}")
    entry = entries[shard]
    index_path = Path(entry["path"])
    index_pin = stable_pin(index_path, entry["sha256"], entry["bytes"])
    receipt_path = replay / "shards" / f"shard-{shard:04d}" / "receipt.json"
    require(receipt_path.is_file(), f"replay_terminal_receipt_missing:{shard}")
    receipt_pin = pin(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    require(
        receipt.get("schema") == "sepalith.dat10.sourcewalk_provenance_shard.v3"
        and receipt.get("status") == "complete"
        and receipt.get("shard") == shard,
        f"replay_terminal_receipt_incomplete:{shard}",
    )
    require(receipt.get("rows") == entry.get("rows"), f"replay_row_count:{shard}")
    token_pin = next(
        (item for item in index["input_inventory"]["shard_pins"] if item.get("shard") == shard),
        None,
    )
    require(token_pin is not None, f"token_pin_missing:{shard}")
    packet_manifest_path = BASE / f"shard-{shard:04d}/structured-materialization-v1/manifest.json"
    require(packet_manifest_path.is_file(), f"candidate_manifest_missing:{shard}")
    packet_manifest = json.loads(packet_manifest_path.read_text())
    require(packet_manifest.get("policy", {}).get("source_content_train_only") is True, f"non_train_packet:{shard}")
    packet_decl = packet_manifest["outputs"]["candidate_packets"]
    packet_path = Path(packet_decl["path"])
    packet_pin = stable_pin(packet_path, packet_decl["sha256"], packet_decl["bytes"] if "bytes" in packet_decl else None)
    required_binding = {
        "candidate_packet_manifest_sha256": sha(packet_manifest_path),
        "candidate_packets_sha256": packet_decl["sha256"],
        "cpt_sha256": CPT_SHA,
        "driver_sha256": DRIVER_SHA,
        "global_sha256": GLOBAL_SHA,
        "hold_ledger_sha256": HOLD_SHA,
        "index_manifest_sha256": index_sha,
        "index_shard_sha256": entry["sha256"],
        "license_parser_sha256": LICENSE_SHA,
        "strict_validator_sha256": STRICT_SHA,
        "token_manifest_sha256": token_pin["manifest_sha256"],
        "token_rows_sha256": token_pin["sha256"],
    }
    require(receipt.get("binding") == required_binding, f"replay_binding_changed:{shard}")
    outputs = receipt.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == 1, f"replay_output_contract:{shard}")
    ledger_decl = outputs[0]
    ledger_path = Path(ledger_decl["path"])
    ledger_pin = stable_pin(ledger_path, ledger_decl["sha256"], ledger_decl["bytes"])
    rows = jsonl(ledger_path)
    require(len(rows) == ledger_decl["rows"] == receipt["rows"], f"replay_ledger_rows:{shard}")
    row_ids = [row.get("row_id") for row in rows]
    require(all(isinstance(row_id, str) and row_id for row_id in row_ids), f"replay_row_ids:{shard}")
    require(len(row_ids) == len(set(row_ids)), f"replay_duplicate_row_ids:{shard}")
    require(all(row.get("shard") == shard for row in rows), f"replay_row_shard:{shard}")
    queued_ids = sorted(
        row["row_id"] for row in rows if row.get("family") == "roxygen_drafting" and row.get("status") == QUEUED
    )
    source = {
        "shard": shard,
        "provenance_receipt": receipt_pin,
        "provenance_ledger": {
            **ledger_pin,
            "rows": len(rows),
            "row_ids_sha256": ids_sha(row_ids),
        },
        "queued_ids": queued_ids,
        "queued_rows": len(queued_ids),
        "queued_ids_sha256": ids_sha(queued_ids),
        "candidate_packet": {**packet_pin, "rows": packet_decl["rows"]},
        "index_shard": {**index_pin, "rows": entry["rows"]},
        "binding": required_binding,
    }
    source["binding_sha256"] = hashlib.sha256(canonical(source).encode("utf-8")).hexdigest()
    source.update(
        {
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_pin["sha256"],
            "receipt_rows": len(rows),
            "ledger_path": str(ledger_path),
            "ledger_sha256": ledger_decl["sha256"],
            "packet_path": str(packet_path),
            "packet_sha256": packet_decl["sha256"],
        }
    )
    return source


def validate_sources(replay: Path, require_all: bool) -> dict:
    source_pin = source_code_pin()
    index_path, index, entries, index_sha = load_index(replay)
    sources = []
    missing = []
    for shard in SELECTED:
        receipt_path = replay / "shards" / f"shard-{shard:04d}" / "receipt.json"
        if not receipt_path.is_file():
            missing.append(shard)
            continue
        sources.append(validate_replay_shard(replay, shard, index, entries, index_sha))
    if require_all:
        require(not missing, "replay_terminal_receipts_missing:" + ",".join(map(str, missing)))
    queued = [row_id for source in sources for row_id in source["queued_ids"]]
    require(len(queued) == len(set(queued)), "selected_queued_ids_overlap")
    return {
        "source_pin": source_pin,
        "replay_index": {**pin(index_path), "status": index["status"], "requested_shards": index["requested_shards"]},
        "index_sha256": index_sha,
        "selected_shards": SELECTED,
        "shard_count": len(SELECTED),
        "completed_shards": [source["shard"] for source in sources],
        "missing_terminal_shards": missing,
        "source_rows_completed": sum(source["receipt_rows"] for source in sources),
        "queued_rows_completed": len(queued),
        "queued_ids_sha256_completed": ids_sha(queued),
        "sources": sources,
        "all_terminal_commits_verified": not missing,
        "no_replay_prior_shards": not set(SELECTED) & set(PRIOR),
    }


def semantic_code(source_pin: dict) -> dict:
    return source_pin["semantic_code"]


def reusable_child(target: Path, source: dict, code: dict) -> dict | None:
    manifest_path = target / "manifest.json"
    ledger_path = target / "semantic-ledger.jsonl"
    if not manifest_path.is_file() and not target.exists():
        return None
    require(manifest_path.is_file() and ledger_path.is_file(), f"nonreusable_semantic_child:{target}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "sepalith.dat10.sourcewalk_roxy_semantic.v6", f"semantic_child_schema:{target}")
    require(manifest.get("status") == "complete_review_only", f"semantic_child_not_complete:{target}")
    require(manifest.get("training_admission") is False, f"semantic_child_admission:{target}")
    require(manifest.get("code") == code, f"semantic_child_code:{target}")
    require(manifest.get("streaming_binding") == source, f"semantic_child_source_binding:{target}")
    require(manifest.get("source_provenance_binding_sha256") == source["binding_sha256"], f"semantic_child_source_digest:{target}")
    require(manifest.get("queued_ids_sha256") == source["queued_ids_sha256"], f"semantic_child_queued_digest:{target}")
    output = manifest.get("output")
    require(isinstance(output, dict) and output.get("path") == str(ledger_path), f"semantic_child_output_path:{target}")
    ledger_pin = stable_pin(ledger_path, output["sha256"], output["bytes"])
    rows = jsonl(ledger_path)
    ids = [row.get("row_id") for row in rows]
    require(len(rows) == output["rows"] == source["queued_rows"], f"semantic_child_output_rows:{target}")
    require(sorted(ids) == source["queued_ids"] and ids_sha(ids) == source["queued_ids_sha256"], f"semantic_child_output_ids:{target}")
    require(output.get("row_ids_sha256") == ids_sha(ids), f"semantic_child_output_row_digest:{target}")
    require(manifest.get("output_binding") == output, f"semantic_child_output_binding:{target}")
    return {
        "shard": source["shard"],
        "status": "reused_independently_verified",
        "manifest": pin(manifest_path),
        "semantic_output": {**ledger_pin, "rows": len(rows), "row_ids_sha256": ids_sha(ids)},
        "source_binding_sha256": source["binding_sha256"],
        "queued_rows": source["queued_rows"],
    }


def existing_output_plan(output: Path, sources: list[dict], code: dict) -> dict:
    if not output.exists():
        return {"exists": False, "reusable": [], "to_compute": SELECTED}
    stream = output / "streaming-manifest.json"
    if stream.is_file():
        value = json.loads(stream.read_text())
        require(value.get("requested_shards") == SELECTED, "existing_stream_scope")
        require(value.get("replay_index_sha256") == INDEX_SHA, "existing_stream_index")
        require(value.get("training_admission") is False, "existing_stream_admission")
    source_by_shard = {source["shard"]: source for source in sources}
    reusable = []
    to_compute = []
    for shard in SELECTED:
        target = output / f"shard-{shard:04d}"
        if not target.exists():
            to_compute.append(shard)
        else:
            require(shard in source_by_shard, f"existing_child_source_not_terminal:{shard}")
            reusable.append(reusable_child(target, source_by_shard[shard], code))
    extras = [p.name for p in output.iterdir() if p.is_dir() and p.name.startswith("shard-") and p.name not in {f"shard-{s:04d}" for s in SELECTED}]
    require(not extras, "existing_output_extra_shards:" + ",".join(sorted(extras)))
    return {"exists": True, "reusable": reusable, "to_compute": to_compute}


def make_command(replay: Path, output: Path) -> list[str]:
    require(SELECTED == list(range(27, 41)) and len(SELECTED) == 14, "selected_range_contract")
    return [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        str(TIME_LIMIT),
        "ionice",
        "-c3",
        "nice",
        "-n",
        "10",
        "taskset",
        "-c",
        "8,10",
        str(PYTHON),
        "-B",
        str(QUEUE_RUNNER),
        "--replay-root",
        str(replay),
        "--output",
        str(output),
        "--shards",
        ",".join(map(str, SELECTED)),
        "--max-workers",
        "2",
    ]


def plan(replay: Path, output: Path, require_all: bool) -> dict:
    intake = validate_sources(replay, require_all=require_all)
    code = semantic_code(intake["source_pin"])
    output_plan = existing_output_plan(output, intake["sources"], code) if intake["all_terminal_commits_verified"] else {
        "exists": output.exists(),
        "reusable": [],
        "to_compute": SELECTED,
        "deferred_until_all_replay_receipts": True,
    }
    return {
        "schema": "sepalith.dat10.semantic27to40_launch_plan.v1",
        "status": "ready_to_execute" if intake["all_terminal_commits_verified"] else "waiting_for_replay_terminal_commits",
        "training_admission": False,
        "replay_root": str(replay),
        "output": str(output),
        "selected_shards": SELECTED,
        "shard_count": 14,
        "no_replay_prior_shards": True,
        "intake": intake,
        "output_plan": output_plan,
        "command": make_command(replay, output),
        "time_limit_seconds": TIME_LIMIT,
        "cpu_policy": {"cores": [8, 10], "max_workers": 2, "nice": 10, "ionice_class": 3, "cuda": False, "cloud": False},
        "review_only": True,
    }


def execute(replay: Path, output: Path) -> int:
    prepared = plan(replay, output, require_all=True)
    output.mkdir(parents=True, exist_ok=True)
    log = output.parent / "Sourcewalk-semantic-streaming-queue-shards27to40-v1.log"
    require(not log.exists(), f"launch_log_exists:{log}")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": "/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1",
        }
    )
    command = make_command(replay, output)
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        launch = {
            **prepared,
            "schema": "sepalith.dat10.semantic27to40_launch.v1",
            "status": "running",
            "started_at_utc": started,
            "controller_pid": os.getpid(),
            "classifier_pid": child.pid,
            "command": command,
            "environment": {key: env[key] for key in ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")},
            "log": str(log),
        }
        atomic_json(PACKET / "launch.json", launch)
        atomic_json(PACKET / "status.json", launch)
        code = child.wait()
    stream_manifest = output / "streaming-manifest.json"
    terminal = {
        "schema": "sepalith.dat10.semantic27to40_terminal.v1",
        "status": "complete_review_only" if code == 0 else "failed_or_partial",
        "training_admission": False,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exit_code": code,
        "launch": pin(PACKET / "launch.json"),
        "output": str(output),
        "log": pin(log),
    }
    if stream_manifest.is_file():
        terminal["streaming_manifest"] = pin(stream_manifest)
    atomic_json(PACKET / "terminal.json", terminal)
    atomic_json(PACKET / "status.json", {**launch, **terminal, "status": terminal["status"]})
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, default=REPLAY_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--execute", action="store_true", help="launch only after all replay receipts pass")
    args = parser.parse_args()
    if args.execute:
        return execute(args.replay_root, args.output)
    print(json.dumps(plan(args.replay_root, args.output, require_all=False), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
