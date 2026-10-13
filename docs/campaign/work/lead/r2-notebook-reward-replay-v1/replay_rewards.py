#!/usr/bin/env python3
"""Replay frozen rewards over existing outputs; never generates or executes R."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        path = args.input / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"input artifact differs: {name}")
    trainer_path = args.source / "experiments/training/campaign_rl_train.py"
    protocol_path = args.source / "packages/sepalith/src/sepalith/campaign_protocol.py"
    pins = manifest["source_pins"]
    if digest(trainer_path) != pins["campaign_rl_train.py"]["sha256"]:
        raise ValueError("frozen trainer source differs")
    if digest(protocol_path) != pins["campaign_protocol.py"]["sha256"]:
        raise ValueError("frozen protocol source differs")
    sys.path.insert(0, str(args.source / "experiments/training"))
    sys.path.insert(0, str(args.source / "packages/sepalith/src"))
    tokenizers = importlib.import_module("tokenizers")
    trainer = importlib.import_module("campaign_rl_train")
    tokenizer = tokenizers.Tokenizer.from_file(str(args.input / "tokenizer.json"))

    contexts_list = rows(args.input / "contexts.jsonl")
    outputs = rows(args.input / "outputs.jsonl")
    recorded = rows(args.input / "recorded-rewards.jsonl")
    contexts = {row["row_id"]: row for row in contexts_list}
    if len(contexts) != 80 or len(outputs) != 320 or len(recorded) != 320:
        raise ValueError("replay row cardinality differs")
    if any(output["row_id"] != reference["id"] for output, reference in zip(outputs, recorded)):
        raise ValueError("output/reward row ID order differs")

    reward = trainer.CampaignPRM03Reward(
        decoder=lambda ids, **kwargs: tokenizer.decode(ids, skip_special_tokens=kwargs["skip_special_tokens"])
    )
    mismatches = []
    batch_timings = []
    replay_rows = []
    for offset in range(0, len(outputs), args.batch_size):
        batch_started = time.perf_counter()
        for output, reference in zip(outputs[offset:offset + args.batch_size], recorded[offset:offset + args.batch_size]):
            context = contexts[output["row_id"]]
            score, actual = reward.score_one(
                context["context"], context["target_operation"], context["target_body_text"], output["generated_ids"],
                row_id=context["row_id"], family=context["family"], package_id=context["package_id"],
            )
            exact = actual == reference and score == reference["reward"]
            replay_rows.append({"row_index": output["row_index"], "id": output["row_id"], "score": score, "exact_record": exact})
            if not exact:
                differing = sorted(set(actual) | set(reference))
                differing = [key for key in differing if actual.get(key) != reference.get(key)]
                mismatches.append({"row_index": output["row_index"], "id": output["row_id"], "fields": differing,
                                   "actual": actual, "recorded": reference})
        elapsed = time.perf_counter() - batch_started
        batch_timings.append({"start": offset, "rows": min(args.batch_size, len(outputs) - offset), "seconds": elapsed})

    args.output.mkdir(parents=True, exist_ok=False)
    replay_path = args.output / "replay-rows.jsonl"
    replay_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in replay_rows), encoding="utf-8")
    result = {
        "schema": "sepalith.rl08.reward-replay-result.v1",
        "status": "exact_parity" if not mismatches else "mismatch",
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "threads_requested": int(os.environ.get("OMP_NUM_THREADS", "0")),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "rows": 320,
        "unique_contexts": 80,
        "exact_score_rows": sum(row["score"] == reference["reward"] for row, reference in zip(replay_rows, recorded)),
        "exact_full_record_rows": sum(row["exact_record"] for row in replay_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "batch_size": args.batch_size,
        "batch_timings": batch_timings,
        "cpu_work_seconds": time.perf_counter() - started,
        "input_manifest_sha256": digest(args.input / "manifest.json"),
        "trainer_sha256": digest(trainer_path),
        "protocol_sha256": digest(protocol_path),
        "replay_rows_sha256": digest(replay_path),
        "constraints": {"model_generation": False, "generated_r_execution": False},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "host", "rows", "exact_score_rows", "exact_full_record_rows", "mismatch_count", "cpu_work_seconds")}, sort_keys=True))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
