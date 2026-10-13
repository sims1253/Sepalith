#!/usr/bin/env python3
"""PC bf16 greedy references for the C03 fp16 parity check (night job).

GPU work: enqueue ``c03/jobs/C03-parity-references.json``; never run it outside
the nightly window. Everything is generated with the settings the R1 gates are
bound to (the recipe's ``dev_gate.generation_batching``: batch 32, dynamic KV
cache, prompts sorted by prompt tokens descending then case id, left padding
with pad id 1, cap 1,024, bf16), so a Kaggle run with the same settings in
fp16 is a like-for-like comparison.

For each model (the bound parent 11,586, then ``--checkpoint``, by default the
lr1e-5 gate-268 checkpoint, which is closer to what the census will run):

1. ``dev75``: DEV75 through ``campaign_eval.development_evaluator``; writes
   ``<label>/dev75.cases/cases.json`` (evaluator schema, R1 counts) and
   ``<label>/dev75-generations.jsonl``.
2. ``train200``: the 200 TRAIN prompts from ``train_prompts.py`` (seed 3407,
   stratified by family; the file's sha256 is pinned with
   ``--prompts-sha256``) through ``campaign_eval.batched_greedy_generate``;
   writes ``<label>/train200-generations.jsonl``.

The generation records match ``scripts/kaggle/infer_entry.py``'s
``generations.jsonl``, so ``compare_parity.py`` reads both. When the night's
own DEV75 results for a model exist (``--determinism-reference
label=cases.json``), the receipt also reports run-to-run identity against
them. Nothing is uploaded; the gate checkpoint stays local.

Without ``--inner`` the script creates a fresh attempt directory, takes the
CUDA lock through ``cuda_host_guard`` and runs itself with ``--inner`` under
it. The receipt is ``<attempt>/references.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "sepalith.c03.parity-references.v1"
HERE = Path(__file__).resolve().parent
DEFAULT_RECIPE = "/home/m0hawk/sandbox/Sepalith/docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json"
DEFAULT_CHECKPOINT = "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-268"
TRAIN_PROMPTS_SHA256 = "3a99e2ad6a6b04995cd52a2614d67e13e31ba8e30eae2d2fc4a6d502ff250dc2"
CUDA_LOCK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock")
PHASE_SECONDS = {"dev75": 900, "train200": 1500}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", type=Path, default=Path(DEFAULT_RECIPE))
    p.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT), help="second model (default: lr1e-5 gate 268)")
    p.add_argument("--checkpoint-label", default="lr1e-5-gate-268")
    p.add_argument("--skip-checkpoint", action="store_true", help="parent only")
    p.add_argument("--prompts-sha256", default=TRAIN_PROMPTS_SHA256)
    p.add_argument("--determinism-reference", action="append", default=[],
                   help="label=/path/cases.json: the night's own DEV75 cases for that model")
    p.add_argument("--output-dir", type=Path, required=True, help="a fresh attempt directory is created inside")
    p.add_argument("--guard-seconds", type=int, default=5400)
    p.add_argument("--no-guard", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--attempt-dir", type=Path, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def deadline_left():
    value = os.environ.get("SEPALITH_DEADLINE_EPOCH")
    return math.inf if value in (None, "") else float(value) - time.time()


def record(case_id, generated, tokenizer, cap, stop_ids, **extra):
    """Same fields as infer_entry.generation_record."""
    stop = generated[-1] if generated and generated[-1] in stop_ids else None
    text_ids = generated[:-1] if stop is not None else generated
    return {"id": case_id, "sample": 0, "generated_ids": generated, "generated_tokens": len(generated),
            "text": tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "finish_reason": "stop" if stop is not None else "length", "stop_token_id": stop,
            "cap_hit": stop is None and len(generated) == cap, **extra}


def identity(reference, other):
    """Exact generated-id identity of ``other`` against ``reference`` (id -> ids)."""
    divergent = []
    for case_id, ids in reference.items():
        theirs = other.get(case_id)
        if theirs != ids:
            position = next((i for i, (a, b) in enumerate(zip(ids, theirs or [])) if a != b), min(len(ids), len(theirs or [])))
            divergent.append({"id": case_id, "first_divergence": position})
    return {"cases": len(reference), "identical": len(reference) - len(divergent),
            "identity_rate": (len(reference) - len(divergent)) / len(reference) if reference else None,
            "divergent": divergent}


class Run:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, fn):
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        value = fn()
        torch.cuda.synchronize()
        return value, {"seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def inner(args):
    import torch
    from transformers import AutoTokenizer

    from sepalith.campaign_protocol import NATIVE_EOG_IDS, PromptContext, build_training_row
    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval import campaign_eval
    from sepalith.training.sft.recipe_v2 import load_bound_v2

    sys.path.insert(0, str(HERE))
    import train_prompts

    out = args.attempt_dir
    receipt_path = out / "references.json"
    receipt = {"schema": SCHEMA, "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": {}}

    def save():
        write_json(receipt_path, receipt)

    recipe_path = args.recipe.resolve()
    recipe = load_bound_v2(recipe_path)
    gate = recipe["dev_gate"]
    cap = gate["generation_max_new_tokens"]
    bound = gate["generation_batching"]
    batch, kv_cache = bound["batch_size"], bound["kv_cache"]
    panel = gate["panels"]["dev75"]
    max_sequence = recipe["cohort"]["max_sequence_tokens"]
    parent = Path(recipe["parent"]["path"])
    tokenizer_sha = recipe["parent"]["files"]["tokenizer.json"]
    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        raise SystemExit("exactly one visible CUDA device required")

    prompts_path = out / "c03-train200-seed3407.jsonl"
    with Path(train_prompts.DEFAULT_ROWS).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if sha256_file(train_prompts.DEFAULT_ROWS) != train_prompts.ROWS_SHA256:
        raise SystemExit("TRAIN rows sha256 differs")
    chosen, quota = train_prompts.select(rows)
    del rows
    prompts_path.write_text("".join(json.dumps(train_prompts.prompt_record(row), sort_keys=True) + "\n" for row in chosen))
    if sha256_file(prompts_path) != args.prompts_sha256:
        raise SystemExit("TRAIN prompts sha256 differs from the pinned selection")
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines()]

    receipt.update(
        recipe={"path": str(recipe_path), "sha256": sha256_file(recipe_path)},
        settings={"dtype": "bf16", "attention": "sdpa (transformers default)", "cap": cap, "batch_size": batch,
                  "kv_cache": kv_cache, "bound_generation_batching": bound, "stop_ids": list(NATIVE_EOG_IDS)},
        inputs={"dev75": {"path": panel["path"], "sha256": panel["sha256"]},
                "train200": {"path": str(prompts_path), "sha256": args.prompts_sha256, "quota": quota,
                             "source_sha256": train_prompts.ROWS_SHA256}},
        device={"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )
    save()
    determinism = dict(item.split("=", 1) for item in args.determinism_reference)
    models = [("parent-11586", parent)]
    if not args.skip_checkpoint:
        models.append((args.checkpoint_label, args.checkpoint.resolve()))
    measure = Run(torch)

    for label, path in models:
        entry = {"path": str(path), "status": "running", "phases": {}}
        receipt["models"][label] = entry
        if deadline_left() < sum(PHASE_SECONDS.values()) + 120:
            entry["status"] = "skipped_deadline"
            save()
            continue
        if sha256_file(path / "tokenizer.json") != tokenizer_sha:
            raise SystemExit(f"{label}: tokenizer.json differs from the parent's")
        entry["model_safetensors_sha256"] = sha256_file(path / "model.safetensors")
        load_started = time.monotonic()
        reference = load_pinned_reference_tokenizer(path)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        model = campaign_eval.load_development_model(path, dtype="bf16", device="cuda")
        restore_trainer_eog_alignment(model, tokenizer, reference)
        assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "c03_parity_references")
        named = list(model.named_parameters())
        if len(named) != 381 or sum(x.numel() for _, x in named) != 2516756480:
            raise SystemExit(f"{label} is not the exact dense model")
        model.eval()
        entry["load_seconds"] = time.monotonic() - load_started
        model_dir = out / label
        model_dir.mkdir()
        save()

        # 1. DEV75 through the evaluator.
        if deadline_left() >= PHASE_SECONDS["dev75"]:
            case_dir = model_dir / "dev75.cases"
            case_dir.mkdir()
            ids = [json.loads(line)["id"] for line in Path(panel["path"]).read_text().splitlines() if line.strip()]
            evaluate = campaign_eval.development_evaluator({
                "development_panel": {"path": panel["path"], "sha256": panel["sha256"]}, "renderer_id": "zeta2-prm03-v1",
                "development_case_ids": sorted(ids), "development_max_new_tokens": cap,
                "parameters": {"max_sequence_tokens": max_sequence}, "evaluation_output_directory": str(case_dir),
                "generation_batch_size": batch, "generation_kv_cache": kv_cache})
            summary, stats = measure(lambda evaluate=evaluate: evaluate(model, tokenizer, path, 0))
            cases = json.loads((case_dir / "cases.json").read_text())["results"]
            prompt_sha = {}
            for line in Path(panel["path"]).read_text().splitlines():
                if line.strip():
                    case = json.loads(line)
                    row = build_training_row(PromptContext.from_mapping(case["context"]), operation=case["operation"],
                                             region_new=case["region_new"], tokenizer=tokenizer, row_id=case["id"],
                                             family=case["family"], package_id=case["package_id"], split="dev")
                    prompt_sha[case["id"]] = sha256_ids(row["input_ids"][:row["target_start"]])
            write_jsonl(model_dir / "dev75-generations.jsonl", [
                record(c["id"], c["generated_ids"], tokenizer, cap, NATIVE_EOG_IDS, prompt_tokens=c["prompt_tokens"],
                       prompt_sha256=prompt_sha[c["id"]], family=c["family"])
                for c in cases])
            counts = summary["counts"]
            phase = {**stats, "status": "complete", "counts": counts,
                     "r1_dev75": {"valid": counts["protocol_valid"], "cap": counts["cap_hit"],
                                  "edit": counts["edit_exact"], "nofp": counts["strict_noop_false_suggestions"]},
                     "prompt_nll": summary["prompt_nll"], "target_nll": summary["target_nll"],
                     "generated_tokens": sum(c["generated_tokens"] for c in cases),
                     "generation_runtime": summary.get("generation_runtime"),
                     "cases_path": str(case_dir / "cases.json"), "cases_sha256": sha256_file(case_dir / "cases.json")}
            night = determinism.get(label)
            if night and Path(night).is_file():
                theirs = {c["id"]: c["generated_ids"] for c in json.loads(Path(night).read_text())["results"]}
                phase["vs_night_results"] = {"path": night, **identity(theirs, {c["id"]: c["generated_ids"] for c in cases})}
            entry["phases"]["dev75"] = phase
        else:
            entry["phases"]["dev75"] = {"status": "skipped_deadline"}
        save()

        # 2. TRAIN-200 through the batched greedy path.
        if deadline_left() >= PHASE_SECONDS["train200"]:
            def train_run():
                out_rows = {}
                with torch.no_grad():
                    for index, indexes in enumerate(campaign_eval.generation_batches(
                            [(p["id"], len(p["prompt_token_ids"])) for p in prompts], batch)):
                        generated = campaign_eval.batched_greedy_generate(
                            model, [prompts[i]["prompt_token_ids"] for i in indexes], cap=cap,
                            stop_ids=NATIVE_EOG_IDS, kv_cache=kv_cache, device="cuda")
                        for i, ids in zip(indexes, generated, strict=True):
                            out_rows[i] = record(prompts[i]["id"], ids, tokenizer, cap, NATIVE_EOG_IDS,
                                                 prompt_tokens=len(prompts[i]["prompt_token_ids"]),
                                                 prompt_sha256=sha256_ids(prompts[i]["prompt_token_ids"]),
                                                 family=prompts[i]["family"], batch_index=index)
                return [out_rows[i] for i in range(len(prompts))]
            generated, stats = measure(train_run)
            write_jsonl(model_dir / "train200-generations.jsonl", generated)
            tokens = sum(r["generated_tokens"] for r in generated)
            entry["phases"]["train200"] = {**stats, "status": "complete", "generated_tokens": tokens,
                                           "tokens_per_second": tokens / stats["seconds"],
                                           "cap_hits": sum(r["cap_hit"] for r in generated),
                                           "generations_sha256": sha256_file(model_dir / "train200-generations.jsonl")}
        else:
            entry["phases"]["train200"] = {"status": "skipped_deadline"}
        entry["status"] = "complete" if all(p.get("status") == "complete" for p in entry["phases"].values()) else "partial"
        save()
        del model
        torch.cuda.empty_cache()

    receipt["status"] = "complete" if all(m["status"] == "complete" for m in receipt["models"].values()) else "partial"
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}), flush=True)
    return 0


def outer(args):
    attempt = args.output_dir / time.strftime("%Y%m%dT%H%M%S")
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--inner", "--attempt-dir", str(attempt),
               "--recipe", str(args.recipe), "--output-dir", str(args.output_dir), "--checkpoint", str(args.checkpoint),
               "--checkpoint-label", args.checkpoint_label, "--prompts-sha256", args.prompts_sha256]
    for item in args.determinism_reference:
        command += ["--determinism-reference", item]
    if args.skip_checkpoint:
        command.append("--skip-checkpoint")
    if args.no_guard:
        launch = command
    else:
        CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
        CUDA_LOCK.touch(exist_ok=True)
        write_json(attempt / "command.json", command)
        hard = os.environ.get("SEPALITH_HARD_STOP_EPOCH")
        seconds = args.guard_seconds if hard in (None, "") else min(args.guard_seconds, float(hard) - time.time() - 60)
        launch = [sys.executable, "-B", "-m", "sepalith.training.guards.cuda_host_guard", "--command-json",
                  str(attempt / "command.json"), "--output", str(attempt / "guard"), "--seconds", str(int(max(60, seconds)))]
    with (attempt / "outer.log").open("wb") as log:
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
        signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        code = process.wait()
    receipt_path = attempt / "references.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    status = (receipt or {}).get("status")
    summary = {"exit": code, "receipt": str(receipt_path), "status": status,
               "models": {label: {name: {k: phase.get(k) for k in ("status", "seconds", "r1_dev75", "cap_hits")}
                                  for name, phase in model.get("phases", {}).items()}
                          for label, model in ((receipt or {}).get("models") or {}).items()}}
    needs_user = None if code == 0 and status == "complete" else f"C03 parity references did not complete ({status}); see {attempt}"
    result = os.environ.get("SEPALITH_JOB_RESULT")
    if result:
        write_json(Path(result), {"summary": json.dumps(summary, sort_keys=True), "evaluations": [str(receipt_path)],
                                  "needs_user": needs_user})
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if code == 0 and status == "complete" else 1


def main(argv=None):
    args = parse_args(argv)
    return inner(args) if args.inner else outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
