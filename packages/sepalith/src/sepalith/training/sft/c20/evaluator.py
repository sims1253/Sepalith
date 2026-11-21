"""Run C20 greedy exec evaluation with the historical generation and scoring helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from sepalith.exec_eval import jobs
from sepalith.training.grpo_exec.pool import require
from sepalith.training.sft.c20 import data, recipe as config

BATCH_SIZE = 32
CAP = 1024
DEVICE = "cuda:0"


def limit_cuda(memory_mib: int) -> None:
    import torch

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA device required")
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    budget = memory_mib * 1024**2
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), device=0)


def run(args: argparse.Namespace) -> int:
    from sepalith.training.eval.campaign_eval import batched_greedy_generate

    # The historical input check verifies the freeze only for the final hold-out.
    tasks, models, version = jobs.check_inputs(args)
    out = args.work_dir
    path = out / "receipt.json"
    receipt = data.read(path) if path.is_file() else {
        "schema": jobs.RECEIPT_SCHEMA, "job": args.job_id, "kind": args.kind,
        "started": jobs.now(), "attempts": [], "models": {},
    }
    receipt.update(
        status="running",
        tasks={"path": str(args.tasks), "sha256": args.tasks_sha256, "n_tasks": len(tasks),
               "n_functions": len({task["function_id"] for task in tasks})},
        gate={"commit_pinned": jobs.GATE_COMMIT, "fingerprint": version["fingerprint"],
              "checkout_commit": version["commit"]},
        settings={"dtype": "bf16", "cap": CAP, "stop_ids": list(jobs.STOP_IDS),
                  "renderer": "zeta2-prm03-v1", "decoding": "greedy", "prompts_per_batch": BATCH_SIZE,
                  "rows_per_batch": BATCH_SIZE, "kv_cache": "dynamic", "padding": "left",
                  "ordering": "prompt tokens descending, then task id",
                  "tokenizer": f"pinned parent tokenizer ({jobs.PARENT})"},
    )
    receipt["attempts"].append({"started": jobs.now()})
    if args.kind == "holdout":
        receipt["freeze_manifest"] = {"path": str(jobs.MANIFEST), "sha256": jobs.MANIFEST_SHA256}
    jobs.write_json(path, receipt)
    for spec in models:
        label = spec["label"]
        record = receipt["models"].setdefault(label, {**spec, "status": "pending"})
        if record["status"] == "generated":
            continue
        if jobs.stop_requested() or jobs.deadline_left() < args.model_budget_minutes * 60:
            receipt["status"] = "partial"
            jobs.write_json(path, receipt)
            return jobs.EXIT_DEFERRED
        import torch

        limit_cuda(config.MEMORY_MIB)
        torch.cuda.init()
        torch.zeros(1, device=DEVICE)
        receipt["device"] = {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        started = time.monotonic()
        model, contract = jobs.load_model(Path(spec["path"]), DEVICE)
        record["load"] = {"seconds": round(time.monotonic() - started, 1), **contract,
                          "weights_bytes": (Path(spec["path"]) / "model.safetensors").stat().st_size}

        def generate(prompts: list[list[int]], _number: int, model: object = model) -> list[list[list[int]]]:
            return [[ids] for ids in batched_greedy_generate(
                model, prompts, cap=CAP, stop_ids=jobs.STOP_IDS, kv_cache="dynamic", device=DEVICE,
            )]

        outcome = jobs.generate_model(label, tasks, out / f"completions-{label}.jsonl", generate,
                                      samples=1, prompts_per_batch=BATCH_SIZE)
        record.setdefault("generation", []).append(outcome)
        record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(DEVICE))
        del generate, model
        torch.cuda.empty_cache()
        if outcome["status"] != "complete":
            record["status"] = receipt["status"] = "partial"
            jobs.write_json(path, receipt)
            return jobs.EXIT_DEFERRED
        record["status"] = "generated"
        jobs.write_json(path, receipt)

    # Validation and final use the same single-sample metrics and paired bootstrap.
    jobs.score(argparse.Namespace(**{**vars(args), "kind": "holdout"}), receipt, tasks)
    receipt.update(status="complete", finished=jobs.now())
    jobs.write_json(path, receipt)
    (out / "receipt.md").write_text(jobs.markdown({**receipt, "kind": "holdout"}))
    return jobs.EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("validation", "holdout"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-sha256", required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model-budget-minutes", type=int, required=True)
    parser.add_argument("--score-workers", type=int, default=8)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
