#!/usr/bin/env python3
"""Materialize the bounded RL-08 reward replay inputs from pinned artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PINS = {
    "generation-records.jsonl": "dc69a6cdf30881f488b9e774f1799dc6815a6107d51ca5c9569fbd2fb484b742",
    "reward-records.jsonl": "9e4d1c3146ae9a9b7b14b5c02ab190a3cb47911a0a425cabcc326a401d98f857",
    "recipe.json": "13ae035d539e974b4fc2dfd4d17b2069d3d90f2b57666d9ad71e2f410d336dd0",
    "eligible-train-rows.jsonl": "7e9cf35e8ecbf1af151df78bfc42c07d7b1d24f6ab82ec47770463d967fe9465",
    "context-sidecar.jsonl": "265b80762efc9544230f1aba09e492906760e98a34431593fdeb4813d6e169ac",
    "source-row-draw-sequence.json": "7e82f42eafa97bbb07a410ff2e4078cc96744cc45933f1167f7ebc3fbd558e98",
    "tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "campaign_rl_train.py": "95e88ee22f5b8636cd6ffb8e2c21fadcb74a4f58e9e5c267217a8a47bda35c83",
    "campaign_protocol.py": "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def require_pin(path: Path, name: str) -> None:
    actual = digest(path)
    if actual != PINS[name]:
        raise ValueError(f"{name} hash differs: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--rewards", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "generation-records.jsonl": args.generation,
        "reward-records.jsonl": args.rewards,
        "recipe.json": args.recipe,
        "eligible-train-rows.jsonl": args.rows,
        "context-sidecar.jsonl": args.sidecar,
        "source-row-draw-sequence.json": args.schedule,
        "tokenizer.json": args.tokenizer,
        "campaign_rl_train.py": args.trainer,
        "campaign_protocol.py": args.protocol,
    }
    for name, path in paths.items():
        require_pin(path, name)

    generations = list(read_jsonl(args.generation))
    rewards = list(read_jsonl(args.rewards))
    if len(generations) != 320 or len(rewards) != 320:
        raise ValueError("expected exactly 320 generation and reward rows")
    if any(g["generated_ids_sha256"] != r["output_ids_sha256"] for g, r in zip(generations, rewards)):
        raise ValueError("generation/reward output hash alignment differs")

    group_ids = []
    for offset in range(0, 320, 4):
        ids = {row["id"] for row in rewards[offset:offset + 4]}
        if len(ids) != 1:
            raise ValueError(f"reward group {offset // 4} does not contain one row ID")
        group_ids.append(next(iter(ids)))
    if len(set(group_ids)) != 80:
        raise ValueError("expected 80 unique TRAIN source IDs")
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if schedule["row_ids"][:80] != group_ids:
        raise ValueError("reward group IDs differ from the first 80 frozen source draws")

    wanted = set(group_ids)
    training = {}
    for row in read_jsonl(args.rows):
        if row.get("id") in wanted:
            training[row["id"]] = row
    contexts = {}
    for row in read_jsonl(args.sidecar):
        if row.get("row_id") in wanted:
            contexts[row["row_id"]] = row
    if set(training) != wanted or set(contexts) != wanted:
        raise ValueError("minimal context join is incomplete")

    args.output.mkdir(parents=True, exist_ok=False)
    context_rows = []
    for row_id in group_ids:
        source, sidecar = training[row_id], contexts[row_id]
        if sidecar["context_has_target_or_reward_keys"] is not False:
            raise ValueError(f"context sidecar leaks target fields for {row_id}")
        if source["family"] != sidecar["family"] or source["package_id"] != sidecar["package_id"]:
            raise ValueError(f"provenance join differs for {row_id}")
        context_rows.append({
            "row_id": row_id,
            "family": source["family"],
            "package_id": source["package_id"],
            "target_operation": source["target_operation"],
            "target_body_text": source["target_body_text"],
            "context": sidecar["context"],
            "prompt_sha256": sidecar["prompt_sha256"],
        })
    output_rows = []
    for index, (generation, reward) in enumerate(zip(generations, rewards)):
        ids = generation["generated_ids"]
        canonical = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()
        if canonical != generation["generated_ids_sha256"]:
            raise ValueError(f"generated ID canonical hash differs at row {index}")
        output_rows.append({
            "row_index": index,
            "row_id": reward["id"],
            "candidate_index": index % 4,
            "generated_ids": ids,
            "generated_ids_sha256": canonical,
        })
    write_jsonl(args.output / "contexts.jsonl", context_rows)
    write_jsonl(args.output / "outputs.jsonl", output_rows)
    write_jsonl(args.output / "recorded-rewards.jsonl", rewards)
    (args.output / "tokenizer.json").write_bytes(args.tokenizer.read_bytes())

    manifest = {
        "schema": "sepalith.rl08.reward-replay-inputs.v1",
        "status": "materialized_from_pinned_existing_rollouts",
        "rows": {"contexts": 80, "outputs": 320, "recorded_rewards": 320, "candidates_per_context": 4},
        "source_pins": {name: {"sha256": PINS[name], "bytes": path.stat().st_size} for name, path in paths.items()},
        "artifacts": {},
        "constraints": {"model_generation": False, "generated_r_execution": False, "full_context_dump_staged": False},
    }
    for name in ("contexts.jsonl", "outputs.jsonl", "recorded-rewards.jsonl", "tokenizer.json"):
        path = args.output / name
        manifest["artifacts"][name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
