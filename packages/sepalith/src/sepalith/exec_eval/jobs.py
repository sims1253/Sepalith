#!/usr/bin/env python3
"""Execution-reward measurement jobs for the night queue (measurement only: no training, no decisions).

Two jobs share this module:

- ``EXEC-holdout-baseline``: greedy completions (the DEV gate evaluator's settings: bf16, cap 1,024,
  batch 32, dynamic KV cache, left padding, longest prompt first, stop ids 1 and 130073) on tasks built
  from the **frozen hold-out** (``holdout-freeze-v1.json``, sha256-checked, read from its read-only
  copy), scored with the execution reward at gate c631141. Models in order: arm A 536, then (if time
  allows) GRPO checkpoint 600 and LoRA E750. Per model: mean reward, all-pass, exact-match and protocol
  validity rates, by family and cut length, with function-clustered bootstrap CIs and paired
  differences against arm A.
- ``EXEC-train-census``: k = 8 samples at temperature 1.0 from arm A 536 on tasks from the gated
  training slices, scored the same way: the share of tasks with 0 < passes < k (the GRPO signal share),
  all-wrong and all-right shares by family and cut length, and the mean reward.

Stages:

1. ``prepare-holdout`` / ``prepare-census`` (CPU, before the job): build the tasks with the existing
   builder (``exec_env.tasks.build_for_function``: synth finish_block cuts, PRM03 rendering, per-cut
   observability and inferability gates; rename_propagation too), drop cuts whose hidden remainder
   touches an untested branch from the freeze manifest (SCOPED/BUGFIX declarations), keep at most
   ``--per-function`` tasks per function (stratified by family and cut length), and write
   ``tasks.jsonl`` plus a receipt with its sha256.
2. ``run`` (the night entry point, GPU): takes the CUDA lock through ``cuda_host_guard`` and runs
   ``--inner``, which verifies the inputs (tasks sha256, freeze manifest sha256, exec_env fingerprint
   ``7f8e073fbedb4b9d``, model files), generates model by model with completions appended per batch to
   ``<work-dir>/completions-<label>.jsonl`` (crash-safe; a rerun resumes after the last complete batch),
   then scores every completion on the CPU (8 R workers, nice 10) and writes ``records-<label>.jsonl``,
   ``metrics.json`` and ``receipt.json`` / ``receipt.md``. Before every batch it checks
   ``SEPALITH_DEADLINE_EPOCH`` and the stop-request file; SIGTERM ends the inner process at once (the
   current batch is redone on resume).

Exit codes: 0 finished; 75 saved with work left (resumable: the runner defers and reruns it); 3 an input
check failed (needs the user); anything else is a failure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
from typing import Any

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1")
MANIFEST = DATA / "holdout-freeze-v1.json"
MANIFEST_SHA256 = "9e74c1644c96576693b48a5fb37cfdd5af93c7501c505943a5fc31852e0eca19"
GATE_COMMIT = "c631141b0b58cbd9135a424c2cd59f1305fe634e"
GATE_FINGERPRINT = "7f8e073fbedb4b9d"
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586")
CUDA_LOCK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock")
RECEIPT_SCHEMA = "sepalith.exec-eval.receipt.v1"
TASKS_SCHEMA = "sepalith.exec-eval.tasks-receipt.v1"
CAP = 1024
BATCH = 32
CENSUS_K = 8
STOP_IDS = (1, 130073)
DEADLINE_MARGIN_SECONDS = 240
BOOTSTRAP = 2000
EXIT_OK, EXIT_NEEDS_USER, EXIT_DEFERRED = 0, 3, 75


class InputError(RuntimeError):
    """An input check failed; the job exits 3."""


# helpers ------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a torn last line (a crash mid-write) is ignored."""
    out = []
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def deadline_left() -> float:
    value = os.environ.get("SEPALITH_DEADLINE_EPOCH")
    return math.inf if value in (None, "") else float(value) - time.time()


def stop_requested() -> bool:
    path = os.environ.get("SEPALITH_STOP_REQUEST_FILE")
    return bool(path) and Path(path).exists()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def expand_slices(spec: str) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        match = re.fullmatch(r"([A-Za-z]+)(\d+)\.\.([A-Za-z]+)?(\d+)", part.strip())
        if match:
            width = len(match.group(2))
            out += [f"{match.group(1)}{n:0{width}d}" for n in range(int(match.group(2)), int(match.group(4)) + 1)]
        elif part.strip():
            out.append(part.strip())
    return out


def bucket_of(task: dict) -> str:
    return str(task.get("curriculum", {}).get("bucket") or "short")


# task preparation (CPU) ------------------------------------------------------------------------------
def remainder_lines(task: dict, unit_lines: int) -> tuple[int, int]:
    """1-based first and last line of the unit that the task's hidden remainder (or region) covers."""
    before = task["exec"]["before"].split("\n") if task["exec"]["before"] else []
    after = task["exec"]["after"].split("\n") if task["exec"]["after"] else []
    return len(before) + 1, unit_lines - len(after)


def untested_spans(unit: str, mutant_ids: set[str]) -> list[tuple[int, int]]:
    from sepalith.exec_env import mutants

    if not mutant_ids:
        return []
    tokens = mutants.parse_tokens([unit])[0]
    return [(m.line1, m.line2) for m in mutants.generate(unit, tokens or []) if m.id in mutant_ids]


def select_tasks(tasks: list[dict], per_function: int, seed: str) -> list[dict]:
    """At most ``per_function`` tasks, round-robin over (family, cut length) strata, deterministic."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task in sorted(tasks, key=lambda t: t["id"]):
        strata[(task["family"], bucket_of(task))].append(task)
    rng = random.Random(f"exec-eval:{seed}")
    for group in strata.values():
        rng.shuffle(group)
    out: list[dict] = []
    keys = sorted(strata)
    while len(out) < per_function and any(strata[k] for k in keys):
        for key in keys:
            if strata[key] and len(out) < per_function:
                out.append(strata[key].pop())
    return out


def build_tasks(directories: list[tuple[str, Path, set[str]]], out_dir: Path, per_function: int, workers: int,
                families: set[str], source: dict, max_tasks: int | None = None) -> dict:
    from sepalith.exec_env import tasks as task_builder
    from sepalith.exec_env.gate import gate_version
    from sepalith.exec_env.harness import ExecHarness

    version = gate_version()
    if version["fingerprint"] != GATE_FINGERPRINT:
        raise InputError(f"exec_env fingerprint {version['fingerprint']} is not the pinned {GATE_FINGERPRINT}")
    tokenizer = task_builder.load_tokenizer()
    reasons: Counter = Counter()
    kept: list[dict] = []
    per_function_built: dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    with ExecHarness(workers=min(workers, 8), nice=10) as harness:
        for index, (slice_id, directory, untested) in enumerate(directories):
            if max_tasks is not None and len(kept) >= max_tasks:
                reasons["functions_not_built_task_cap_reached"] += 1
                continue
            built, why = task_builder.build_for_function(directory, harness, tokenizer, families)
            reasons.update(why)
            unit = (directory / "source.R").read_text(encoding="utf-8")
            spans = untested_spans(unit, untested)
            n_lines = len(unit.rstrip("\n").split("\n"))
            usable = []
            for task in built:
                first, last = remainder_lines(task, n_lines)
                if any(not (l2 < first or l1 > last) for l1, l2 in spans):
                    reasons["cut_touches_untested_branch"] += 1
                    continue
                task["slice"] = slice_id
                usable.append(task)
            chosen = select_tasks(usable, per_function, directory.name)
            reasons["dropped_by_per_function_cap"] += len(usable) - len(chosen)
            per_function_built[directory.name] = len(usable)
            kept += chosen
            if (index + 1) % 50 == 0:
                print(f"[{index + 1}/{len(directories)}] tasks {len(kept)}", flush=True)
    kept.sort(key=lambda t: (t["function_id"], t["id"]))
    tasks_path = out_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as stream:
        for task in kept:
            stream.write(json.dumps(task) + "\n")
    receipt = {"schema": TASKS_SCHEMA, "created": now(), "source": source, "tasks": str(tasks_path),
               "tasks_sha256": sha256(tasks_path), "n_tasks": len(kept),
               "n_functions": len({t["function_id"] for t in kept}), "functions_considered": len(directories),
               "functions_without_tasks": sum(1 for n in per_function_built.values() if n == 0),
               "per_function_cap": per_function, "families": dict(Counter(t["family"] for t in kept)),
               "buckets": dict(Counter(f"{t['family']}:{bucket_of(t)}" for t in kept)),
               "builder_reasons": dict(sorted(reasons.items())), "gate_version": version,
               "builder": "sepalith.exec_env.tasks.build_for_function (finish_block + rename_propagation)"}
    write_json(out_dir / "tasks-receipt.json", receipt)
    return receipt


def holdout_directories() -> tuple[list[tuple[str, Path, set[str]]], dict]:
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise InputError("hold-out freeze manifest sha256 differs from the pinned value")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["gate"]["commit"] != GATE_COMMIT or manifest["gate"]["fingerprint"] != GATE_FINGERPRINT:
        raise InputError("freeze manifest was not made at the pinned gate")
    copy = Path(manifest["read_only_copy"]["path"])
    untested: dict[tuple[str, str], set[str]] = defaultdict(set)
    for branch in manifest["untested_branches"]:
        untested[(branch["slice"], branch["function"])].add(branch["mutant"])
    out = []
    for record in manifest["functions"]:
        directory = copy / record["slice"] / record["function"]
        for name, digest in record["sha256"].items():
            if sha256(directory / name) != digest:
                raise InputError(f"{record['slice']}/{record['function']}/{name} differs from the freeze manifest")
        out.append((record["slice"], directory, untested[(record["slice"], record["function"])]))
    source = {"kind": "holdout", "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA256,
              "read_only_copy": str(copy), "functions": len(out),
              "untested_branch_declarations": sum(len(v) for v in untested.values())}
    return out, source


def census_directories(root: Path, slices: Sequence[str]) -> tuple[list[tuple[str, Path, set[str]]], dict]:
    held = set()
    if MANIFEST.is_file():
        held = {r["function"] for r in json.loads(MANIFEST.read_text())["functions"]}
    out, counts = [], Counter()
    for slice_id in slices:
        if not slice_id.startswith("S"):
            raise InputError(f"{slice_id} is not a training slice")
        for directory in sorted(p for p in (root / slice_id).iterdir() if (p / "source.R").is_file()):
            if (directory / "SKIP").exists():
                counts["skipped"] += 1
                continue
            gate_path = directory / "gate.json"
            if directory.name in held:
                counts["in_holdout"] += 1
                continue
            if not gate_path.is_file() or not (directory / "test.R").is_file():
                counts["not_gated"] += 1
                continue
            gate = json.loads(gate_path.read_text())
            version = gate.get("gate_version") or {}
            if version.get("commit") != GATE_COMMIT or version.get("fingerprint") != GATE_FINGERPRINT \
                    or version.get("exec_env_modified"):
                counts["gated_at_other_version"] += 1
                continue
            if gate.get("verdict") != "pass":
                counts["gate_not_pass"] += 1
                continue
            counts["used"] += 1
            out.append((slice_id, directory, set()))
    # hash order, so a task cap keeps a slice-independent random subset of functions
    out.sort(key=lambda item: hashlib.sha256(f"census:{item[1].name}".encode()).hexdigest())
    return out, {"kind": "census", "root": str(root), "slices": list(slices), "function_counts": dict(counts),
                 "function_order": "sha256('census:' + function id)"}


# generation (GPU) ------------------------------------------------------------------------------------
GenerateFn = Callable[[list[list[int]], int], list[list[list[int]]]]


def pending_work(tasks: list[dict], done: set[tuple[str, int]], samples: int) -> list[dict]:
    return [t for t in tasks if any((t["id"], s) not in done for s in range(samples))]


def generate_model(label: str, tasks: list[dict], completions: Path, generate: GenerateFn, *, samples: int,
                   prompts_per_batch: int, log: Callable[[str], None] = print) -> dict:
    """Generate for every task not yet in ``completions``; append each finished batch atomically.

    ``generate(prompts, batch_index)`` returns, per prompt, ``samples`` lists of generated ids.
    Returns ``{"status": "complete" | "deferred", ...}``; deferred means the deadline or a stop
    request came before the work was done.
    """
    from sepalith.training.eval.campaign_eval import generation_batches

    if completions.is_file():  # drop a torn last line (a crash mid-write) before appending
        data = completions.read_bytes()
        if data and not data.endswith(b"\n"):
            completions.write_bytes(data[:data.rfind(b"\n") + 1])
    done = {(row["task_id"], int(row["sample"])) for row in read_jsonl(completions)}
    todo = pending_work(tasks, done, samples)
    batches = generation_batches([(t["id"], len(t["prompt_ids"])) for t in todo], prompts_per_batch)
    started = time.monotonic()
    generated = 0
    completions.parent.mkdir(parents=True, exist_ok=True)
    for number, batch in enumerate(batches):
        if stop_requested() or deadline_left() < DEADLINE_MARGIN_SECONDS:
            return {"status": "deferred", "generated": generated, "remaining_tasks": len(todo) - number * prompts_per_batch,
                    "seconds": round(time.monotonic() - started, 1)}
        chosen = [todo[i] for i in batch]
        outputs = generate([t["prompt_ids"] for t in chosen], number)
        if len(outputs) != len(chosen) or any(len(o) != samples for o in outputs):
            raise RuntimeError("generation returned the wrong number of rows")
        lines = []
        for task, rows in zip(chosen, outputs, strict=True):
            for sample, ids in enumerate(rows):
                if (task["id"], sample) not in done:
                    lines.append(json.dumps({"model": label, "task_id": task["id"], "sample": sample,
                                             "completion_ids": [int(x) for x in ids]}))
        with completions.open("a", encoding="utf-8") as stream:
            stream.write("".join(line + "\n" for line in lines))
            stream.flush()
            os.fsync(stream.fileno())
        generated += len(lines)
        if number % 10 == 0:
            log(f"{label}: batch {number + 1}/{len(batches)}, {generated} completions, "
                f"{time.monotonic() - started:.0f}s")
    return {"status": "complete", "generated": generated, "seconds": round(time.monotonic() - started, 1)}


def batched_sample_generate(model: object, prompts: list[list[int]], *, k: int, cap: int, seed: int,
                            device: str) -> list[list[list[int]]]:
    """k samples per prompt at temperature 1.0 (top_p 1, top_k off), left padding, stop ids 1 and 130073."""
    import torch

    from sepalith.training.eval.campaign_eval import (
        GENERATION_BOS_TOKEN_ID,
        GENERATION_PAD_TOKEN_ID,
        truncate_at_stop,
    )

    expanded = [prompt for prompt in prompts for _ in range(k)]
    width = max(len(p) for p in expanded)
    input_ids = torch.full((len(expanded), width), GENERATION_PAD_TOKEN_ID, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for row, prompt in enumerate(expanded):
        input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention[row, width - len(prompt):] = 1
    torch.manual_seed(seed)
    sequence = getattr(model, "generate")(input_ids=input_ids, attention_mask=attention, do_sample=True, temperature=1.0,  # noqa: B009
                              top_p=1.0, top_k=0, max_new_tokens=cap, eos_token_id=list(STOP_IDS),
                              pad_token_id=GENERATION_PAD_TOKEN_ID, bos_token_id=GENERATION_BOS_TOKEN_ID,
                              repetition_penalty=1.0, use_cache=True)
    rows = [truncate_at_stop(r, STOP_IDS, cap) for r in sequence[:, width:].tolist()]
    return [rows[i * k:(i + 1) * k] for i in range(len(prompts))]


def load_model(path: Path, device: str) -> tuple[object, dict]:
    """bf16 model on ``device`` with the DEV evaluator's tokenizer contract (pinned parent tokenizer)."""
    from transformers import AutoTokenizer

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval import campaign_eval

    tokenizer = AutoTokenizer.from_pretrained(str(PARENT), local_files_only=True, trust_remote_code=False)
    reference = load_pinned_reference_tokenizer(PARENT)
    model = campaign_eval.load_development_model(path, dtype="bf16", device=device)
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "exec_eval_load")
    model.eval()
    return model, {"tokenizer_repair": repair, "tokenizer_contract": audit}


# scoring (CPU) ------------------------------------------------------------------------------------------
def score_model(label: str, tasks_path: Path, work: Path, reward: Callable[..., list[dict]], by_id: dict[str, dict]) -> list[dict]:
    rows = read_jsonl(work / f"completions-{label}.jsonl")
    seen: set[tuple[str, int]] = set()
    unique = []
    for row in rows:
        key = (row["task_id"], int(row["sample"]))
        if key not in seen and row["task_id"] in by_id:
            seen.add(key)
            unique.append(row)
    records: list[dict] = []
    for start in range(0, len(unique), 512):
        chunk = unique[start:start + 512]
        scored = reward([r["task_id"] for r in chunk], [tuple(r["completion_ids"]) for r in chunk])
        for row, record in zip(chunk, scored, strict=True):
            task = by_id[row["task_id"]]
            records.append({"model": label, "task_id": row["task_id"], "sample": row["sample"],
                            "function_id": task["function_id"], "family": task["family"], "bucket": bucket_of(task),
                            "target_lines": task.get("curriculum", {}).get("target_lines"),
                            "reward": record["reward"], "all_pass": bool(record["exact"]),
                            "exact_match": bool(record.get("exact_match")),
                            "protocol_valid": bool(record.get("protocol_valid")), "cap_hit": bool(record.get("cap_hit")),
                            "outcome": record.get("outcome"), "generated_tokens": record.get("generated_tokens"),
                            "behaviour": (record.get("execution") or {}).get("behaviour")})
    with (work / f"records-{label}.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return records


def cluster_bootstrap(records: list[dict], value: Callable[[dict], float], seed: str, reps: int = BOOTSTRAP) -> dict:
    """Mean of ``value`` and a 95% percentile CI resampling functions (clusters) with replacement."""
    by_function: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_function[r["function_id"]].append(float(value(r)))
    clusters = [(sum(v), len(v)) for _, v in sorted(by_function.items())]
    if not clusters:
        return {"mean": None, "ci95": None, "n": 0, "functions": 0}
    total = sum(s for s, _ in clusters) / sum(n for _, n in clusters)
    rng = random.Random(f"exec-eval-bootstrap:{seed}")
    stats = []
    for _ in range(reps):
        pick = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        stats.append(sum(s for s, _ in pick) / sum(n for _, n in pick))
    stats.sort()
    return {"mean": round(total, 4), "ci95": [round(stats[int(0.025 * reps)], 4), round(stats[int(0.975 * reps) - 1], 4)],
            "n": sum(n for _, n in clusters), "functions": len(clusters)}


def paired_difference(a: list[dict], b: list[dict], key: str, seed: str, reps: int = BOOTSTRAP) -> dict:
    """Mean of b - a on common tasks, function-clustered bootstrap CI."""
    va = {r["task_id"]: (r["function_id"], float(r[key])) for r in a}
    pairs = defaultdict(list)
    for r in b:
        if r["task_id"] in va:
            pairs[r["function_id"]].append(float(r[key]) - va[r["task_id"]][1])
    rows = [{"function_id": f, "d": d} for f, ds in pairs.items() for d in ds]
    return cluster_bootstrap(rows, lambda r: r["d"], seed, reps)


def holdout_metrics(records: list[dict], label: str, n_tasks: int) -> dict:
    def block(rows: list[dict], tag: str) -> dict:
        return {"n": len(rows), "functions": len({r["function_id"] for r in rows}),
                "mean_reward": cluster_bootstrap(rows, lambda r: r["reward"], f"{label}:{tag}:reward"),
                "all_pass": cluster_bootstrap(rows, lambda r: r["all_pass"], f"{label}:{tag}:pass"),
                "exact_match": cluster_bootstrap(rows, lambda r: r["exact_match"], f"{label}:{tag}:exact"),
                "protocol_valid": round(sum(r["protocol_valid"] for r in rows) / len(rows), 4) if rows else None,
                "cap_hit": round(sum(r["cap_hit"] for r in rows) / len(rows), 4) if rows else None,
                "outcomes": dict(Counter(str(r["outcome"]) for r in rows))}

    out = {"coverage": {"tasks_scored": len(records), "tasks_total": n_tasks}, "overall": block(records, "all")}
    out["by_family"] = {f: block([r for r in records if r["family"] == f], f"family:{f}")
                        for f in sorted({r["family"] for r in records})}
    out["by_bucket"] = {b: block([r for r in records if r["bucket"] == b and r["family"] == "finish_block"],
                                 f"bucket:{b}")
                        for b in ("short", "medium", "long")}
    return out


def census_metrics(records: list[dict], k: int) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    complete = {t: rows for t, rows in by_task.items() if len(rows) >= k}

    def block(items: list[list[dict]]) -> dict:
        if not items:
            return {"tasks": 0}
        passes = [sum(r["all_pass"] for r in rows[:k]) for rows in items]
        rewards = [[r["reward"] for r in rows[:k]] for rows in items]
        return {"tasks": len(items),
                "signal_share_pass": round(sum(1 for p in passes if 0 < p < k) / len(items), 4),
                "all_wrong_share": round(sum(1 for p in passes if p == 0) / len(items), 4),
                "all_right_share": round(sum(1 for p in passes if p == k) / len(items), 4),
                "reward_variance_share": round(sum(1 for rs in rewards if max(rs) - min(rs) > 1e-9) / len(items), 4),
                "mean_reward": round(sum(sum(rs) for rs in rewards) / (k * len(items)), 4),
                "mean_pass_rate": round(sum(passes) / (k * len(items)), 4)}

    items = list(complete.values())
    out = {"k": k, "tasks_complete": len(items), "tasks_partial": len(by_task) - len(items), "overall": block(items)}
    out["by_family"] = {f: block([rows for rows in items if rows[0]["family"] == f]) for f in sorted({r[0]["family"] for r in items})}
    out["by_bucket"] = {b: block([rows for rows in items if rows[0]["bucket"] == b and rows[0]["family"] == "finish_block"])
                        for b in ("short", "medium", "long")}
    return out


def markdown(receipt: dict) -> str:
    lines = [f"# {receipt['job']} ({receipt['status']})", "", f"Started {receipt['started']}; "
             f"tasks {receipt['tasks']['n_tasks']} from {receipt['tasks']['n_functions']} functions; gate {GATE_COMMIT[:7]}.", ""]
    metrics = receipt.get("metrics", {})
    if receipt["kind"] == "holdout":
        lines += ["| Model | Tasks | Mean reward (95% CI) | All-pass | Exact | Protocol valid |", "| --- | --- | --- | --- | --- | --- |"]
        for label, m in metrics.items():
            o = m.get("overall") or {}
            if not o.get("n"):
                continue
            def fmt(cell: dict) -> str:
                return f"{cell['mean']:.3f} [{cell['ci95'][0]:.3f}, {cell['ci95'][1]:.3f}]"
            lines.append(f"| {label} | {o['n']} | {fmt(o['mean_reward'])} | {fmt(o['all_pass'])} | {fmt(o['exact_match'])} "
                         f"| {o['protocol_valid']:.3f} |")
    else:
        for label, m in metrics.items():
            o = m.get("overall") or {}
            lines.append(f"- {label}: {json.dumps(o)}")
    return "\n".join(lines) + "\n"


# job --------------------------------------------------------------------------------------------------
def load_models(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    models = data["models"]
    if not models or len({m["label"] for m in models}) != len(models):
        raise InputError("models file needs distinct labels")
    return models


def check_inputs(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    from sepalith.exec_env.gate import gate_version

    if sha256(args.tasks) != args.tasks_sha256:
        raise InputError("tasks file sha256 differs from the job's pinned value")
    if args.kind == "holdout" and sha256(MANIFEST) != MANIFEST_SHA256:
        raise InputError("hold-out freeze manifest sha256 differs")
    version = gate_version()
    if version["fingerprint"] != GATE_FINGERPRINT:
        raise InputError(f"exec_env fingerprint {version['fingerprint']} is not {GATE_FINGERPRINT}")
    models = load_models(args.models)
    for model in models:
        for name in ("config.json", "model.safetensors"):
            if not (Path(model["path"]) / name).is_file():
                raise InputError(f"{model['label']}: {name} missing in {model['path']}")
    tasks = read_jsonl(args.tasks)
    return tasks, models, version


def inner(args: argparse.Namespace) -> int:
    work: Path = args.work_dir
    status_path = args.attempt_dir / "inner-status.json"

    def finish(code: int, note: str = "") -> int:
        write_json(status_path, {"exit": code, "note": note, "at": now()})
        return code

    def on_term(*_: object) -> None:
        write_json(status_path, {"exit": EXIT_DEFERRED, "note": "SIGTERM: current batch is redone on resume", "at": now()})
        os._exit(EXIT_DEFERRED)

    signal.signal(signal.SIGTERM, on_term)
    try:
        tasks, models, version = check_inputs(args)
    except InputError as error:
        return finish(EXIT_NEEDS_USER, str(error))
    samples = args.k if args.kind == "census" else 1
    if args.kind == "census":
        models = models[:1]
    receipt_path = work / "receipt.json"
    receipt: dict[str, Any] = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {
        "schema": RECEIPT_SCHEMA, "job": args.job_id, "kind": args.kind, "started": now(),
        "purpose": "measurement only: no training, no adoption decision", "attempts": [], "models": {}}
    receipt.update(status="running", tasks={"path": str(args.tasks), "sha256": args.tasks_sha256, "n_tasks": len(tasks),
                                            "n_functions": len({t["function_id"] for t in tasks})},
                   gate={"commit_pinned": GATE_COMMIT, "fingerprint": version["fingerprint"],
                         "checkout_commit": version["commit"]},
                   settings={"dtype": "bf16", "cap": args.cap, "stop_ids": list(STOP_IDS), "renderer": "zeta2-prm03-v1",
                             "decoding": "greedy" if samples == 1 else f"sampling k={samples} T=1.0 top_p=1 top_k=0",
                             "prompts_per_batch": args.batch if samples == 1 else max(1, args.batch // samples),
                             "rows_per_batch": args.batch, "kv_cache": "dynamic", "padding": "left",
                             "ordering": "prompt tokens descending, then task id",
                             "tokenizer": f"pinned parent tokenizer ({PARENT})"})
    receipt["attempts"].append({"attempt": str(args.attempt_dir), "started": now()})
    write_json(receipt_path, receipt)
    if args.kind == "holdout":
        receipt["freeze_manifest"] = {"path": str(MANIFEST), "sha256": MANIFEST_SHA256}

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return finish(EXIT_NEEDS_USER, "exactly one visible CUDA device is required")
    device = "cuda:0"
    torch.cuda.init()
    torch.zeros(1, device=device)  # create the context before any memory query
    receipt["device"] = {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    deferred = False
    for spec in models:
        label = spec["label"]
        record = receipt["models"].setdefault(label, {"name": spec["name"], "path": spec["path"], "status": "pending"})
        if record.get("status") == "generated":
            continue
        if deadline_left() < args.model_budget_minutes * 60 and record.get("status") == "pending":
            record["status"] = "pending"
            record["note"] = "not started: too little time before the deadline"
            deferred = True
            continue
        started = time.monotonic()
        model = None
        try:
            model, contract = load_model(Path(spec["path"]), device)
            record["load"] = {"seconds": round(time.monotonic() - started, 1), **contract,
                              "weights_bytes": (Path(spec["path"]) / "model.safetensors").stat().st_size}
            if samples == 1:
                from sepalith.training.eval.campaign_eval import batched_greedy_generate

                def generate(prompts: list[list[int]], _number: int, model: object = model) -> list[list[list[int]]]:
                    return [[ids] for ids in batched_greedy_generate(model, prompts, cap=args.cap, stop_ids=STOP_IDS,
                                                                     kv_cache="dynamic", device=device)]
                per_batch = args.batch
            else:
                def generate(prompts: list[list[int]], number: int, model: object = model,
                             label: str = label) -> list[list[list[int]]]:
                    seed = int(hashlib.sha256(f"{label}:{number}:{len(prompts)}:{prompts[0][-8:]}".encode()).hexdigest()[:8], 16)
                    return batched_sample_generate(model, prompts, k=samples, cap=args.cap, seed=seed, device=device)
                per_batch = max(1, args.batch // samples)
            outcome = generate_model(label, tasks, work / f"completions-{label}.jsonl", generate, samples=samples,
                                     prompts_per_batch=per_batch)
            record.setdefault("generation", []).append(outcome)
            record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            record["status"] = "generated" if outcome["status"] == "complete" else "partial"
            deferred = deferred or outcome["status"] != "complete"
        except Exception as error:  # noqa: BLE001 - record, keep the other models
            record.update(status="failed", error=f"{type(error).__name__}: {str(error).splitlines()[0][:300]}")
        finally:
            del model
            torch.cuda.empty_cache()
        write_json(receipt_path, receipt)
        if deferred:
            break
    # CPU scoring of everything generated so far
    score(args, receipt, tasks)
    complete = all(receipt["models"].get(m["label"], {}).get("status") == "generated" for m in models)
    failed = any(receipt["models"].get(m["label"], {}).get("status") == "failed" for m in models)
    receipt["status"] = "complete" if complete else ("failed_models" if failed and not deferred else "partial")
    receipt["finished"] = now()
    write_json(receipt_path, receipt)
    (work / "receipt.md").write_text(markdown(receipt))
    if complete:
        return finish(EXIT_OK)
    if deferred:
        return finish(EXIT_DEFERRED, "deadline or stop request; resumes from the saved completions")
    return finish(1, "a model failed")


def score(args: argparse.Namespace, receipt: dict, tasks: list[dict]) -> None:
    from sepalith.exec_env.harness import ExecHarness
    from sepalith.exec_env.reward import ExecReward, load_tasks
    from sepalith.training.rl.reward_v21 import tokenizer_decoder

    by_id = {t["id"]: t for t in tasks}
    exec_tasks = load_tasks(args.tasks)
    started = time.monotonic()
    all_records: dict[str, list[dict]] = {}
    with ExecHarness(workers=min(args.score_workers, 8), nice=10) as harness:
        reward = ExecReward(exec_tasks, harness, tokenizer_decoder(PARENT / "tokenizer.json"))
        for label in list(receipt["models"]):
            if (args.work_dir / f"completions-{label}.jsonl").is_file():
                all_records[label] = score_model(label, args.tasks, args.work_dir, reward, by_id)
    metrics: dict = {}
    if args.kind == "holdout":
        for label, records in all_records.items():
            metrics[label] = holdout_metrics(records, label, len(tasks))
        first = next(iter(all_records), None)
        if first:
            for label, records in all_records.items():
                if label != first and records:
                    metrics[label]["vs_" + first] = {
                        key: paired_difference(all_records[first], records, key, f"{label}:{key}")
                        for key in ("reward", "all_pass", "exact_match")}
    else:
        for label, records in all_records.items():
            metrics[label] = census_metrics(records, args.k)
    receipt["metrics"] = metrics
    receipt["scoring"] = {"seconds": round(time.monotonic() - started, 1), "workers": min(args.score_workers, 8),
                          "reward": "sepalith.reward.exec.v1 (protocol, region, parse, then 0.5 x behaviour + 0.5 x all-pass)",
                          "records": {label: str(args.work_dir / f"records-{label}.jsonl") for label in all_records}}
    write_json(args.work_dir / "metrics.json", metrics)


def outer(args: argparse.Namespace) -> int:
    work: Path = args.work_dir
    attempt = work / "attempts" / time.strftime("%Y%m%dT%H%M%S")
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "sepalith.exec_eval.jobs", "run", "--inner", "--attempt-dir", str(attempt),
               "--kind", args.kind, "--job-id", args.job_id, "--work-dir", str(work), "--tasks", str(args.tasks),
               "--tasks-sha256", args.tasks_sha256, "--models", str(args.models), "--k", str(args.k),
               "--batch", str(args.batch), "--cap", str(args.cap), "--score-workers", str(args.score_workers),
               "--model-budget-minutes", str(args.model_budget_minutes)]
    if args.no_guard:
        launch = command
    else:
        CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
        CUDA_LOCK.touch(exist_ok=True)
        write_json(attempt / "command.json", command)
        hard = os.environ.get("SEPALITH_HARD_STOP_EPOCH")
        seconds = args.guard_seconds if hard in (None, "") else min(args.guard_seconds, float(hard) - time.time() - 120)
        launch = [sys.executable, "-B", "-m", "sepalith.training.guards.cuda_host_guard", "--command-json",
                  str(attempt / "command.json"), "--output", str(attempt / "guard"), "--seconds", str(int(max(60, seconds)))]
    with (attempt / "outer.log").open("wb") as log:
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
        signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        code = process.wait()
    status = json.loads((attempt / "inner-status.json").read_text()) if (attempt / "inner-status.json").is_file() else {}
    inner_code = status.get("exit")
    receipt = json.loads((work / "receipt.json").read_text()) if (work / "receipt.json").is_file() else {}
    if inner_code in (EXIT_OK, EXIT_DEFERRED, EXIT_NEEDS_USER):
        final = inner_code
    elif code != 0 and receipt.get("models"):
        final = EXIT_DEFERRED if any(m.get("status") in ("pending", "partial") for m in receipt["models"].values()) else 1
    else:
        final = 1
    summary = {"exit": final, "inner": status, "status": receipt.get("status"), "receipt": str(work / "receipt.json"),
               "headline": {label: (m.get("overall") or {}).get("mean_reward", {}).get("mean")
                            if args.kind == "holdout" else (m.get("overall") or {}).get("signal_share_pass")
                            for label, m in (receipt.get("metrics") or {}).items()}}
    result = os.environ.get("SEPALITH_JOB_RESULT")
    if result:
        needs = None if final in (EXIT_OK, EXIT_DEFERRED) else f"{args.job_id}: {status.get('note') or 'failed'}; see {attempt}"
        write_json(Path(result), {"summary": json.dumps(summary, sort_keys=True), "needs_user": needs,
                                  "evaluations": [str(work / "receipt.json"), str(work / "metrics.json")]})
    print(json.dumps(summary, sort_keys=True), flush=True)
    return final


# CLI --------------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ph = sub.add_parser("prepare-holdout")
    ph.add_argument("--out-dir", type=Path, required=True)
    ph.add_argument("--per-function", type=int, default=8)
    ph.add_argument("--workers", type=int, default=8)
    ph.add_argument("--families", default="finish_block,rename_propagation")
    pc = sub.add_parser("prepare-census")
    pc.add_argument("--out-dir", type=Path, required=True)
    pc.add_argument("--root", type=Path, default=DATA / "authors")
    pc.add_argument("--slices", default="S000..S017")
    pc.add_argument("--per-function", type=int, default=2)
    pc.add_argument("--max-tasks", type=int, default=1200)
    pc.add_argument("--workers", type=int, default=8)
    pc.add_argument("--families", default="finish_block,rename_propagation")
    run = sub.add_parser("run")
    run.add_argument("--kind", choices=("holdout", "census"), required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--tasks-sha256", required=True)
    run.add_argument("--models", type=Path, required=True)
    run.add_argument("--k", type=int, default=CENSUS_K)
    run.add_argument("--batch", type=int, default=BATCH, help="generation rows per batch (DEV gate: 32)")
    run.add_argument("--cap", type=int, default=CAP)
    run.add_argument("--score-workers", type=int, default=8)
    run.add_argument("--model-budget-minutes", type=int, default=20, help="start a model only with this much time left")
    run.add_argument("--guard-seconds", type=int, default=27000)
    run.add_argument("--no-guard", action="store_true")
    run.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--attempt-dir", type=Path, help=argparse.SUPPRESS)
    sc = sub.add_parser("score-only", help="CPU: re-score saved completions and rewrite metrics")
    sc.add_argument("--kind", choices=("holdout", "census"), required=True)
    sc.add_argument("--work-dir", type=Path, required=True)
    sc.add_argument("--tasks", type=Path, required=True)
    sc.add_argument("--k", type=int, default=CENSUS_K)
    sc.add_argument("--score-workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-holdout":
            directories, source = holdout_directories()
            receipt = build_tasks(directories, args.out_dir, args.per_function, args.workers, set(args.families.split(",")), source)
            print(json.dumps({k: v for k, v in receipt.items() if k != "builder_reasons"}, indent=1))
            return EXIT_OK
        if args.command == "prepare-census":
            directories, source = census_directories(args.root, expand_slices(args.slices))
            out_dir: Path = args.out_dir
            receipt = build_tasks(directories, out_dir, args.per_function, args.workers, set(args.families.split(",")),
                                  source, max_tasks=args.max_tasks)
            receipt["max_tasks"] = args.max_tasks
            write_json(out_dir / "tasks-receipt.json", receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k != "builder_reasons"}, indent=1))
            return EXIT_OK
        if args.command == "score-only":
            receipt_path = args.work_dir / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            score(args, receipt, read_jsonl(args.tasks))
            write_json(receipt_path, receipt)
            (args.work_dir / "receipt.md").write_text(markdown(receipt))
            return EXIT_OK
        if args.inner:
            return inner(args)
        return outer(args)
    except InputError as error:
        print(f"input check failed: {error}", file=sys.stderr)
        return EXIT_NEEDS_USER


if __name__ == "__main__":
    raise SystemExit(main())
