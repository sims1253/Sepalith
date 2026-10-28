#!/usr/bin/env python3
"""C18 in-run validation gate: greedy completions on the validation split, scored with the exec reward (night only, ``cuda:0``).

One process per evaluation: verify the bound recipe, the checkpoint (or ``--start`` for arm A 803 itself,
whose files are checked against the recipe's pins), the validation task file and the exec_env
fingerprint; load the model like the exec-eval jobs (bf16, pinned parent tokenizer contract); generate
greedily with the DEV gate's settings (batch 32 rows, longest prompt first, dynamic KV cache, cap
1,024, stop ids 1 and 130073); score every completion on the CPU (8 R workers, nice 10); write
``records`` and a result with all-pass, mean reward, exact match, protocol validity and cap hits,
overall and by family and cut length. The frozen hold-out is never read.

    python -m sepalith.training.grpo_exec.val_eval --bound-recipe R --checkpoint C --output OUT.json
    python -m sepalith.training.grpo_exec.val_eval --bound-recipe R --start --output OUT.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import time
from typing import cast

from sepalith.training.grpo_exec import recipe as recipe_module

Json = dict[str, object]
RESULT_SCHEMA = "sepalith.c18.val-exec-result.v1"
DEVICE = "cuda:0"
STOP_IDS = (1, 130073)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[Json]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def open_cuda(torch: object) -> object:
    """Select ``cuda:0``, create its context, then reset the peak statistics (memory calls need the context)."""
    cuda = getattr(torch, "cuda")  # noqa: B009
    require(cuda.is_available() and cuda.device_count() == 1, "exactly one visible CUDA device required")
    device = getattr(torch, "device")("cuda", 0)  # noqa: B009
    cuda.set_device(device)
    getattr(torch, "zeros")(1, device=device)  # noqa: B009
    cuda.synchronize(device)
    cuda.reset_peak_memory_stats(device)
    return device


def metrics(records: Sequence[Mapping[str, object]]) -> Json:
    def block(rows: Sequence[Mapping[str, object]]) -> Json:
        n = len(rows)
        if not n:
            return {"n": 0}
        return {"n": n, "functions": len({r["function_id"] for r in rows}),
                "all_pass": round(sum(bool(r["all_pass"]) for r in rows) / n, 6),
                "mean_reward": round(sum(float(cast(float, r["reward"])) for r in rows) / n, 6),
                "exact_match": round(sum(bool(r["exact_match"]) for r in rows) / n, 6),
                "protocol_valid": round(sum(bool(r["protocol_valid"]) for r in rows) / n, 6),
                "cap_hit": round(sum(bool(r["cap_hit"]) for r in rows) / n, 6),
                "outcomes": dict(sorted(Counter(str(r["outcome"]) for r in rows).items()))}
    return {"overall": block(records),
            "by_family": {f: block([r for r in records if r["family"] == f]) for f in sorted({str(r["family"]) for r in records})},
            "by_bucket": {b: block([r for r in records if r["family"] == "finish_block" and r["bucket"] == b])
                          for b in ("short", "medium", "long")}}


def generate_all(tasks: Sequence[Json], generate: Callable[[list[list[int]]], list[list[int]]], batch: int) -> dict[str, list[int]]:
    from sepalith.training.eval.campaign_eval import generation_batches

    out: dict[str, list[int]] = {}
    for indexes in generation_batches([(str(t["id"]), len(cast(list[int], t["prompt_ids"]))) for t in tasks], batch):
        chosen = [tasks[i] for i in indexes]
        rows = generate([list(cast(list[int], t["prompt_ids"])) for t in chosen])
        require(len(rows) == len(chosen), "generation returned the wrong number of rows")
        for task, ids in zip(chosen, rows, strict=True):
            out[str(task["id"])] = [int(x) for x in ids]
    return out


def score(tasks_path: Path, tasks: Sequence[Json], completions: Mapping[str, list[int]], parent: Path,
          reward_settings: Mapping[str, object], workers: int = 8) -> list[Json]:
    from sepalith.exec_env.harness import ExecHarness
    from sepalith.exec_env.reward import ExecReward, ExecRewardConfig, load_tasks
    from sepalith.exec_eval.jobs import bucket_of
    from sepalith.training.rl.reward_v21 import tokenizer_decoder

    exec_tasks = {k: v for k, v in load_tasks(tasks_path).items() if k in completions}
    config = ExecRewardConfig(w_behaviour=float(cast(float, reward_settings["w_behaviour"])),
                              w_all=float(cast(float, reward_settings["w_all"])),
                              exact_bonus=float(cast(float, reward_settings["exact_bonus"])),
                              overlong_factor=float(cast(float, reward_settings["overlong_factor"])),
                              overlong_slack=int(cast(int, reward_settings["overlong_slack"])),
                              overlong_scale=float(cast(float, reward_settings["overlong_scale"])))
    ids = [str(t["id"]) for t in tasks]
    with ExecHarness(workers=min(workers, 8), nice=10) as harness:
        reward = ExecReward(exec_tasks, harness, tokenizer_decoder(parent / "tokenizer.json"), config)
        scored = reward(ids, [tuple(completions[i]) for i in ids])
    records = []
    for task, record in zip(tasks, scored, strict=True):
        records.append({"task_id": task["id"], "function_id": task["function_id"], "family": task["family"],
                        "bucket": bucket_of(task), "reward": record["reward"], "all_pass": bool(record["exact"]),
                        "exact_match": bool(record.get("exact_match")), "protocol_valid": bool(record.get("protocol_valid")),
                        "cap_hit": bool(record.get("cap_hit")), "outcome": record.get("outcome"),
                        "generated_tokens": record.get("generated_tokens")})
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bound-recipe", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", type=Path)
    target.add_argument("--start", action="store_true", help="evaluate the starting model (arm A 803)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe_path, output = args.bound_recipe.resolve(), args.output.resolve()
    recipe = recipe_module.load(recipe_path, check_data=False)
    recipe_sha = recipe_module.sha256(recipe_path)
    require(output.is_absolute() and not output.exists(), "fresh result required")
    output.parent.mkdir(parents=True, exist_ok=True)
    gate = cast(Json, recipe["dev_gate"])
    val = cast(Json, gate["val"])
    parent = cast(Json, recipe["parent"])
    parent_path = Path(str(parent["path"]))
    if args.start:
        checkpoint, step = parent_path, 0
        for name, digest in cast(Json, parent["files"]).items():
            if name != "model.safetensors":
                require(recipe_module.sha256(checkpoint / name) == digest, f"starting model {name} differs")
        manifest_sha = recipe_module.sha256(checkpoint / "campaign-manifest.json")
    else:
        from sepalith.training.grpo.state import verify_checkpoint

        checkpoint = args.checkpoint.resolve()
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        verify_checkpoint(checkpoint, recipe_sha)
        manifest_sha = recipe_module.sha256(checkpoint / "campaign-manifest.json")
    tasks_record = cast(Json, val["tasks"])
    tasks_path = Path(str(tasks_record["path"]))
    require(recipe_module.sha256(tasks_path) == tasks_record["sha256"], "validation tasks differ from their pin")
    from sepalith.exec_env.gate import gate_version

    reward_settings = cast(Json, cast(Json, cast(Json, recipe["pre_registered"])["reward"])["exec"])
    version = gate_version()
    require(version["fingerprint"] == cast(Json, reward_settings["gate"])["fingerprint"], "exec_env fingerprint differs")
    tasks = read_jsonl(tasks_path)
    limit = val.get("limit")
    if limit is not None:
        tasks = tasks[:int(cast(int, limit))]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""), "root CUDA lease must expose one device")
    started = time.monotonic()
    import torch

    from sepalith.exec_eval.jobs import load_model
    from sepalith.training.eval.campaign_eval import batched_greedy_generate

    device = open_cuda(torch)
    model, contract = load_model(checkpoint, DEVICE)
    cap, batch = int(cast(int, val["cap"])), int(cast(int, val["batch_size"]))

    def generate(prompts: list[list[int]], model: object = model) -> list[list[int]]:
        return batched_greedy_generate(model, prompts, cap=cap, stop_ids=STOP_IDS, kv_cache="dynamic", device=DEVICE)

    completions = generate_all(tasks, generate, batch)
    generation_seconds = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    del model
    torch.cuda.empty_cache()
    records = score(tasks_path, tasks, completions, parent_path, reward_settings)
    records_path = output.with_suffix(".records.jsonl")
    with records_path.open("x", encoding="utf-8") as stream:
        for record, task in zip(records, tasks, strict=True):
            stream.write(json.dumps({**record, "completion_ids": completions[str(task["id"])]}) + "\n")
    summary = metrics(records)
    result: Json = {"schema": RESULT_SCHEMA, "status": "complete", "step": step, "start": bool(args.start),
                    "bound_recipe": {"path": str(recipe_path), "sha256": recipe_sha},
                    "checkpoint": {"path": str(checkpoint), "campaign_manifest_sha256": manifest_sha},
                    "tasks": {"path": str(tasks_path), "sha256": tasks_record["sha256"], "n": len(tasks), "limit": limit},
                    "gate_version": version, "metrics": summary,
                    "records": {"path": str(records_path), "sha256": recipe_module.sha256(records_path)},
                    "settings": {"decoding": "greedy", "cap": cap, "rows_per_batch": batch, "kv_cache": "dynamic",
                                 "stop_ids": list(STOP_IDS), "dtype": "bf16"},
                    "load": contract, "generation_seconds": round(generation_seconds, 1),
                    "elapsed_seconds": round(time.monotonic() - started, 1), "peak_cuda_allocated_bytes": peak,
                    "holdout_access": False, "final_set_access": False}
    recipe_module.write_new(output, result)
    print(json.dumps({"status": "complete", "step": step, "output": str(output),
                      "all_pass": cast(Json, summary["overall"]).get("all_pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
