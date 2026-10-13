#!/usr/bin/env python3
"""Run the prepared DAT10 source-walk shards with resumable receipts.

The shard size is an I/O unit.  This driver never uses a row, family, or
elapsed-time cutoff: a prepared row is either converted, or receives an
explicit source/support exclusion in that shard's receipt.  A failed shard
stops the walk and records the failure so an operator can repair or rerun it
without treating the remaining rows as reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
try:
    os.sched_setaffinity(0, {0, 1})
except (AttributeError, OSError):
    pass

E = Path("/mnt/e/sepalith/campaign-20260915/data-work")
ROOT = E / "DAT10-novel-v1/source-walk-shards-v1"
WORK = Path(__file__).resolve().parent
MATERIALIZER = WORK / "materialize_source_walk_shard.py"
FREEZER = WORK / "freeze_source_walk_shard.py"
BASE_REVIEW = E / "DAT10-novel-v1/expansion-increment-na-rm-v1/root-review-packet-v4-na-rm.jsonl"
PROGRESS = ROOT / "source-walk-all-progress-v1.json"
LOG_ROOT = ROOT / "logs"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_shards() -> list[int]:
    result: list[int] = []
    for path in sorted(ROOT.glob("shard-*/manifest.json")):
        meta = load(path)
        if int(meta.get("shard", -1)) == 0:
            continue
        if meta.get("status") != "source_walk_input_prepared_unmaterialized":
            raise RuntimeError(f"source_walk_input_not_frozen:{path}")
        if meta.get("policy", {}).get("cpt_validation_content") is not False:
            raise RuntimeError(f"cpt_validation_policy_missing:{path}")
        if meta.get("policy", {}).get("dev_final_content") is not False:
            raise RuntimeError(f"dev_final_policy_missing:{path}")
        if int(meta.get("route_counts", {}).get("completion", 0)):
            raise RuntimeError(f"completion_route_requires_explicit_runner:{path}")
        result.append(int(meta["shard"]))
    return result


def initial_state(shards: list[int], base_review: Path) -> dict[str, Any]:
    return {
        "schema": "DAT-10-source-walk-all-progress-v1",
        "status": "running",
        "policy": {
            "prepared_rows_are_all_attempted": True,
            "family_row_time_quotas": False,
            "no_arbitrary_omission": True,
            "target_truncation": False,
            "heldout_content_included": False,
            "long_context_queue_is_review_only": True,
        },
        "input": {
            "shards": shards,
            "prepared_rows": sum(
                int(load(ROOT / f"shard-{i:04d}/manifest.json")["planned_rows"])
                for i in shards
            ),
            "shard_plan": str(E / "DAT10-novel-v1/all-data-inventory-v1/source-walk-shards-v1.jsonl"),
            "shard_plan_sha256": sha(E / "DAT10-novel-v1/all-data-inventory-v1/source-walk-shards-v1.jsonl"),
        },
        "base_review_packet": {
            "path": str(base_review),
            "sha256": sha(base_review),
        },
        "completed_shards": [],
        "shards": {},
        "updated_at": now(),
    }


def run_one(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write((f"\n[{now()}] command: " + " ".join(command) + "\n").encode())
        log.flush()
        # The driver itself is launched under nice/ionice by the owner.  Keep
        # one process active so the explicit two-thread CPU lease is honored.
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command_failed:{result.returncode}:{log_path}")


def shard_result(shard_id: int, base_review: Path) -> dict[str, Any]:
    shard = ROOT / f"shard-{shard_id:04d}"
    input_meta = load(shard / "manifest.json")
    expected = int(input_meta["planned_rows"])
    material = shard / "structured-materialization-v1"
    material_manifest = material / "manifest.json"
    if not material_manifest.exists():
        run_one(
            [sys.executable, "-u", str(MATERIALIZER), "--shard", str(shard_id)],
            LOG_ROOT / f"shard-{shard_id:04d}-structured.log",
        )
    if not material_manifest.exists():
        raise RuntimeError("materialization_manifest_missing_after_runner")
    material_meta = load(material_manifest)
    if material_meta.get("status") != "frozen_review_increment_unadmitted":
        raise RuntimeError("materialization_not_frozen_review_increment")
    attempted = int(load(material / "summary.json")["input_rows"])
    if attempted != expected:
        raise RuntimeError(f"materialization_attempt_count_mismatch:{attempted}:{expected}")

    token_manifest = material / "token-audit-manifest.json"
    if not token_manifest.exists():
        run_one(
            [sys.executable, "-u", str(FREEZER), "--shard", str(shard_id), "--base-review", str(base_review)],
            LOG_ROOT / f"shard-{shard_id:04d}-freeze.log",
        )
    if not token_manifest.exists():
        raise RuntimeError("token_audit_manifest_missing_after_runner")
    token_meta = load(token_manifest)
    if token_meta.get("status") != "frozen_review_increment_unadmitted":
        raise RuntimeError("token_audit_not_frozen_review_increment")
    return {
        "status": "complete_token_audited",
        "input_rows": expected,
        "converted_rows": int(token_meta["token_rows"]["rows"]),
        "long_context_rows": int(token_meta["long_context_queue"]["rows"]),
        "material_manifest": {"path": str(material_manifest), "sha256": sha(material_manifest)},
        "token_manifest": {"path": str(token_manifest), "sha256": sha(token_manifest)},
        "review_packet": token_meta["root_review_packet"],
        "updated_at": now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--base-review", type=Path, default=BASE_REVIEW)
    args = parser.parse_args()
    if args.start < 1:
        raise ValueError("source_walk_does_not_rerun_shard_zero")
    if not args.base_review.exists():
        raise FileNotFoundError(args.base_review)
    shards = expected_shards()
    selected = [i for i in shards if i >= args.start and (args.end is None or i <= args.end)]
    if not selected:
        raise RuntimeError("no_prepared_shards_in_requested_range")
    state = initial_state(shards, args.base_review)
    if PROGRESS.exists():
        old = load(PROGRESS)
        if old.get("input", {}).get("prepared_rows") != state["input"]["prepared_rows"]:
            raise RuntimeError("prepared_row_denominator_changed")
        if old.get("base_review_packet", {}).get("sha256") != state["base_review_packet"]["sha256"]:
            raise RuntimeError("base_review_packet_changed")
        state = old
    for shard_id in selected:
        if str(shard_id) in state.get("shards", {}):
            prior = state["shards"][str(shard_id)]
            if prior.get("status") == "complete_token_audited":
                continue
        state.update({"status": "running", "current_shard": shard_id, "updated_at": now()})
        atomic_json(PROGRESS, state)
        try:
            result = shard_result(shard_id, args.base_review)
        except Exception as error:
            state.update({"status": "blocked_on_shard", "current_shard": shard_id, "error": str(error), "updated_at": now()})
            atomic_json(PROGRESS, state)
            raise
        state.setdefault("shards", {})[str(shard_id)] = result
        state["completed_shards"] = sorted(int(x) for x, value in state["shards"].items() if value.get("status") == "complete_token_audited")
        state.update({"status": "running", "current_shard": shard_id, "updated_at": now()})
        atomic_json(PROGRESS, state)
        print(json.dumps({"shard": shard_id, **result}, sort_keys=True), flush=True)
    if set(state.get("completed_shards", [])) >= set(shards):
        state.update({"status": "complete", "current_shard": None, "updated_at": now()})
        atomic_json(PROGRESS, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
