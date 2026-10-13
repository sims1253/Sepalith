#!/usr/bin/env python3
"""GPU validation of batched greedy DEV generation on parent 11,586 (C07 follow-up).

GPU work: enqueue it as a night job (``c07/jobs/C07-batched-eval-validation.json``);
never run it outside the nightly window. It reads the bound recipe (fail-closed
``load_bound_v2``: parent weights, panels, source manifest) and writes only into
a fresh attempt directory under ``--output-dir``.

Phases, in order; each starts only if enough time is left before
``SEPALITH_DEADLINE_EPOCH`` and is otherwise recorded as skipped:

1. ``subset``: 24 DEV75 cases drawn with seed 3407, ``max_new_tokens`` 256.
   Per-case (batch 1, the original evaluator call), then batched at the
   recipe's batch size twice (run-to-run repeatability), then batched with the
   static KV cache. Reports the exact output identity rate against batch 1, the
   first-divergence token positions, tokens/s and peak memory of each run.
2. ``memory_probe``: for each candidate batch size, the B longest DEV75+DEV250
   prompts at the real cap. The parent never emits a stop token, so every row
   runs to the cap: the worst case for memory and time. Includes one
   teacher-forced forward of the longest case (the evaluator runs both). Stops
   at the first out-of-memory.
3. ``decision``: the device budget is the card's total memory less a margin;
   a probe fits if this process's non-PyTorch use (CUDA context, plus what the
   desktop holds, measured live) plus its peak reserved memory stays within it.
   The recommended batch size is the fitting candidate with the lowest projected
   time for both panels at the cap (ties go to the smaller batch).
4. ``full``: DEV75 then DEV250 through the real evaluator
   (``campaign_eval.development_evaluator``) at the recipe's batch size and
   the real cap: wall time per panel, peak memory and the R1 counts. The counts
   are informative; the step-0 baseline is the ``night_arm baseline`` job.

Without ``--inner`` the script creates the attempt directory, takes the CUDA
lock through ``cuda_host_guard`` (host-memory watch, hard time limit) and runs
itself with ``--inner`` under it. The receipt is ``<attempt>/validation.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

SCHEMA = "sepalith.c07.batched-eval-validation.v1"
SUBSET_CASES = 24
SUBSET_SEED = 3407
SUBSET_MAX_NEW_TOKENS = 256
DEFAULT_CANDIDATES = (16, 32, 48, 64)
GIB = 1024 ** 3
CUDA_LOCK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock")
DEFAULT_RECIPE = "/home/m0hawk/sandbox/Sepalith/docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", type=Path, default=Path(DEFAULT_RECIPE), help="bound v2 recipe (parent, panels, cap, batching)")
    p.add_argument("--output-dir", type=Path, required=True, help="a fresh attempt directory is created inside")
    p.add_argument("--batch-size", type=int, help="batch size for the subset and full phases (default: the recipe's)")
    p.add_argument("--candidates", default=",".join(map(str, DEFAULT_CANDIDATES)), help="memory-probe batch sizes")
    p.add_argument("--margin-gib", type=float, default=1.5, help="device memory kept free beyond the measured use")
    p.add_argument("--skip", default="", help="comma-separated phases to skip (subset, memory_probe, full)")
    p.add_argument("--guard-seconds", type=int, default=5400, help="cuda_host_guard hard limit")
    p.add_argument("--no-guard", action="store_true", help="run the inner process directly (tests only)")
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--attempt-dir", type=Path, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def deadline_left():
    value = os.environ.get("SEPALITH_DEADLINE_EPOCH")
    return math.inf if value in (None, "") else float(value) - time.time()


def first_divergence(reference, other):
    """Index of the first differing token, or None if the outputs are identical."""
    if reference == other:
        return None
    for index, (a, b) in enumerate(zip(reference, other, strict=False)):
        if a != b:
            return index
    return min(len(reference), len(other))


def compare(reference, other):
    """Identity of ``other`` against ``reference`` (both id -> generated ids)."""
    rows = []
    for case_id, ids in reference.items():
        position = first_divergence(ids, other[case_id])
        if position is not None:
            rows.append({"id": case_id, "first_divergence": position, "reference_tokens": len(ids),
                         "other_tokens": len(other[case_id])})
    same = len(reference) - len(rows)
    return {"cases": len(reference), "identical": same, "identity_rate": same / len(reference),
            "first_divergence_positions": sorted(row["first_divergence"] for row in rows), "divergent": rows}


# Inner (CUDA) process -----------------------------------------------------------------------------

class Run:
    """Timing and peak memory of one measured block."""

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
        seconds = time.monotonic() - started
        return value, {"seconds": seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
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
    from sepalith.training.sft import gate_rule
    from sepalith.training.sft.recipe_v2 import load_bound_v2, sha256

    out = args.attempt_dir
    receipt_path = out / "validation.json"
    skip = {name for name in args.skip.split(",") if name}
    receipt = {"schema": SCHEMA, "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phases": {}}

    def save():
        write_json(receipt_path, receipt)

    def phase_allowed(name, seconds_needed):
        if name in skip:
            receipt["phases"][name] = {"status": "skipped", "reason": "--skip"}
        elif deadline_left() < seconds_needed:
            receipt["phases"][name] = {"status": "skipped", "reason": f"less than {int(seconds_needed)} s before the night deadline"}
        else:
            return True
        save()
        return False

    recipe_path = args.recipe.resolve()
    recipe = load_bound_v2(recipe_path)
    gate = recipe["dev_gate"]
    cap = gate["generation_max_new_tokens"]
    bound = gate["generation_batching"]
    batch = args.batch_size or bound["batch_size"]
    candidates = sorted({int(x) for x in args.candidates.split(",") if x} | {batch})
    for size in candidates:
        campaign_eval.generation_batching(size)
    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        raise SystemExit("exactly one visible CUDA device required")
    free_start, total = torch.cuda.mem_get_info()
    parent = Path(recipe["parent"]["path"])
    receipt.update(
        recipe={"path": str(recipe_path), "sha256": sha256(recipe_path)},
        parent={"path": str(parent), "model_safetensors_sha256": recipe["parent"]["files"]["model.safetensors"],
                "campaign_manifest_sha256": sha256(parent / "campaign-manifest.json")},
        panels={name: {"path": gate["panels"][name]["path"], "sha256": gate["panels"][name]["sha256"]} for name in ("dev75", "dev250")},
        settings={"dtype": "bf16", "attention": "sdpa (transformers default)", "cap": cap, "batch_size": batch,
                  "bound_generation_batching": bound, "candidates": candidates, "margin_gib": args.margin_gib,
                  "subset": {"cases": SUBSET_CASES, "seed": SUBSET_SEED, "max_new_tokens": SUBSET_MAX_NEW_TOKENS,
                             "panel": "dev75"}},
        device={"name": torch.cuda.get_device_name(0), "total_bytes": total, "free_bytes_at_start": free_start,
                "used_by_others_at_start_bytes": total - free_start, "torch": torch.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )
    save()

    load_started = time.monotonic()
    reference = load_pinned_reference_tokenizer(parent)
    tokenizer = AutoTokenizer.from_pretrained(str(parent), local_files_only=True, trust_remote_code=False)
    model = campaign_eval.load_development_model(parent, dtype="bf16", device="cuda")
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "c07_batched_eval_validation")
    named = list(model.named_parameters())
    if len(named) != 381 or sum(x.numel() for _, x in named) != 2516756480:
        raise SystemExit("parent is not the exact dense model")
    model.eval()
    torch.cuda.synchronize()
    free_loaded, _ = torch.cuda.mem_get_info()
    # Everything on the device that is not this process's PyTorch allocator: desktop, other processes, our CUDA context.
    outside_torch = (total - free_loaded) - torch.cuda.memory_reserved()
    receipt["load"] = {"seconds": time.monotonic() - load_started, "tokenizer_repair": repair,
                       "reserved_after_load_bytes": torch.cuda.memory_reserved(), "outside_torch_bytes": outside_torch}
    save()

    rows = {}
    for name in ("dev75", "dev250"):
        prepared = []
        for line in Path(gate["panels"][name]["path"]).read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if case["split"] != "dev":
                raise SystemExit("only DEV rows may enter this validation")
            context = PromptContext.from_mapping(case["context"])
            row = build_training_row(context, operation=case["operation"], region_new=case["region_new"], tokenizer=tokenizer,
                                     row_id=case["id"], family=case["family"], package_id=case["package_id"], split="dev")
            campaign_eval.validate_development_capacity(row, recipe["cohort"]["max_sequence_tokens"], cap, case["id"])
            prepared.append((case["id"], row["input_ids"][:row["target_start"]], row))
        rows[name] = prepared
    measure = Run(torch)

    def per_case(cases, max_new):
        out_ids = {}
        with torch.no_grad():
            for case_id, prompt, _ in cases:
                ids = torch.tensor([prompt], device="cuda", dtype=torch.long)
                sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                          **campaign_eval.greedy_generate_kwargs(max_new, NATIVE_EOG_IDS))
                out_ids[case_id] = sequence[0, len(prompt):].tolist()
        return out_ids

    def batched(cases, max_new, size, kv_cache="dynamic"):
        out_ids = {}
        with torch.no_grad():
            for indexes in campaign_eval.generation_batches([(c[0], len(c[1])) for c in cases], size):
                generated = campaign_eval.batched_greedy_generate(model, [cases[i][1] for i in indexes], cap=max_new,
                                                                  stop_ids=NATIVE_EOG_IDS, kv_cache=kv_cache, device="cuda")
                out_ids.update({cases[i][0]: ids for i, ids in zip(indexes, generated, strict=True)})
        return out_ids

    def run_record(ids, stats):
        tokens = sum(len(v) for v in ids.values())
        return {**stats, "generated_tokens": tokens, "tokens_per_second": tokens / stats["seconds"],
                "cap_hits": sum(len(v) == SUBSET_MAX_NEW_TOKENS for v in ids.values())}

    # 1. subset ----------------------------------------------------------------------------------
    if phase_allowed("subset", 1500):
        ids75 = sorted(case_id for case_id, _, _ in rows["dev75"])
        chosen = set(random.Random(SUBSET_SEED).sample(ids75, SUBSET_CASES))
        subset = [c for c in rows["dev75"] if c[0] in chosen]
        phase = {"status": "running", "case_ids": [c[0] for c in subset],
                 "prompt_tokens": {"min": min(len(c[1]) for c in subset), "max": max(len(c[1]) for c in subset)}}
        receipt["phases"]["subset"] = phase
        single, stats = measure(lambda: per_case(subset, SUBSET_MAX_NEW_TOKENS))
        phase["batch_1"] = run_record(single, stats)
        save()
        first, stats = measure(lambda: batched(subset, SUBSET_MAX_NEW_TOKENS, batch))
        phase[f"batch_{batch}"] = {**run_record(first, stats), "vs_batch_1": compare(single, first)}
        save()
        second, stats = measure(lambda: batched(subset, SUBSET_MAX_NEW_TOKENS, batch))
        phase[f"batch_{batch}_repeat"] = {**run_record(second, stats), "vs_first_batched_run": compare(first, second)}
        save()
        static, stats = measure(lambda: batched(subset, SUBSET_MAX_NEW_TOKENS, batch, "static"))
        phase[f"batch_{batch}_static_cache"] = {**run_record(static, stats), "vs_batch_1": compare(single, static),
                                                "vs_dynamic_batched": compare(first, static)}
        phase["speedup_batched_over_batch_1"] = (phase[f"batch_{batch}"]["tokens_per_second"]
                                                 / phase["batch_1"]["tokens_per_second"])
        phase["status"] = "complete"
        write_json(out / "subset-generations.json", {"batch_1": single, f"batch_{batch}": first,
                                                     f"batch_{batch}_repeat": second, f"batch_{batch}_static_cache": static})
        save()

    # 2. memory probe ----------------------------------------------------------------------------
    probes = []
    if phase_allowed("memory_probe", 900):
        everything = rows["dev75"] + rows["dev250"]
        longest = sorted(everything, key=lambda c: (-len(c[1]), c[0]))
        receipt["phases"]["memory_probe"] = {"status": "running", "probes": probes}
        step_seconds = None
        for size in candidates:
            needed = 120 + (cap * step_seconds * 1.5 if step_seconds else 300)
            if deadline_left() < needed:
                probes.append({"batch_size": size, "status": "skipped_deadline"})
                break
            cases = longest[:size]
            record = {"batch_size": size, "prompt_tokens_max": len(cases[0][1]), "prompt_tokens_min": len(cases[-1][1])}

            def probe(cases=cases, size=size):
                with torch.no_grad():
                    full = torch.tensor([cases[0][2]["input_ids"]], device="cuda", dtype=torch.long)
                    logits = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits
                    campaign_eval.loss_sums(logits, full, cases[0][2]["target_start"])
                    del logits
                    return campaign_eval.batched_greedy_generate(model, [c[1] for c in cases], cap=cap,
                                                                 stop_ids=NATIVE_EOG_IDS, device="cuda")
            try:
                generated, stats = measure(probe)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                record.update(status="out_of_memory", error=str(error).splitlines()[0])
                probes.append(record)
                save()
                break
            steps = max(len(ids) for ids in generated)
            step_seconds = stats["seconds"] / steps
            projected = outside_torch + stats["peak_reserved_bytes"]
            record.update(status="ok", **stats, decode_steps=steps, seconds_per_step=step_seconds,
                          cap_hits=sum(len(ids) == cap for ids in generated),
                          projected_device_bytes=projected, projected_device_gib=projected / GIB,
                          fits=projected <= total - args.margin_gib * GIB,
                          projected_full_panels_seconds=sum(math.ceil(len(rows[n]) / size) for n in rows) * cap * step_seconds)
            probes.append(record)
            save()
            if not record["fits"]:
                break
        receipt["phases"]["memory_probe"]["status"] = "complete"
        save()

    # 3. decision --------------------------------------------------------------------------------
    fitting = [p for p in probes if p.get("status") == "ok" and p["fits"]]
    recommended = min(fitting, key=lambda p: (round(p["projected_full_panels_seconds"]), p["batch_size"]))["batch_size"] if fitting else None
    bound_probe = next((p for p in probes if p["batch_size"] == bound["batch_size"]), None)
    receipt["decision"] = {
        "budget_bytes": total - args.margin_gib * GIB, "budget_gib": (total - args.margin_gib * GIB) / GIB,
        "rule": "largest benefit within memory: fitting candidate with the lowest projected both-panel time at the cap; ties to the smaller batch",
        "recommended_batch_size": recommended, "bound_batch_size": bound["batch_size"],
        "bound_fits_memory": None if bound_probe is None or bound_probe.get("status") != "ok" else bound_probe["fits"],
        "rebind_needed": recommended is not None and recommended != bound["batch_size"],
    }
    save()

    # 4. full panels -----------------------------------------------------------------------------
    if phase_allowed("full", 3000):
        phase = {"status": "running", "batch_size": batch, "kv_cache": "dynamic", "cap": cap, "panels": {}}
        receipt["phases"]["full"] = phase
        summaries = {}
        for name in ("dev75", "dev250"):
            case_dir = out / f"full-{name}.cases"
            case_dir.mkdir()
            panel = gate["panels"][name]
            evaluate = campaign_eval.development_evaluator({
                "development_panel": {"path": panel["path"], "sha256": panel["sha256"]}, "renderer_id": "zeta2-prm03-v1",
                "development_case_ids": sorted(c[0] for c in rows[name]), "development_max_new_tokens": cap,
                "parameters": {"max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"]},
                "evaluation_output_directory": str(case_dir), "generation_batch_size": batch, "generation_kv_cache": "dynamic"})
            summary, stats = measure(lambda evaluate=evaluate: evaluate(model, tokenizer, parent, 0))
            summaries[name] = summary
            cases = json.loads((case_dir / "cases.json").read_text())["results"]
            phase["panels"][name] = {**stats, "cases": len(cases), "generated_tokens": sum(c["generated_tokens"] for c in cases),
                                     "cap_hits": summary["counts"]["cap_hit"], "counts": summary["counts"],
                                     "generation_runtime": summary.get("generation_runtime"),
                                     "cases_path": str(case_dir / "cases.json"), "cases_sha256": sha256(case_dir / "cases.json")}
            save()
        phase["r1_counts_informative_only"] = gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"])
        phase["total_seconds"] = sum(p["seconds"] for p in phase["panels"].values())
        phase["fits_25_minutes"] = phase["total_seconds"] <= 25 * 60
        phase["each_panel_fits_eval_budget_2400s"] = all(p["seconds"] <= 2400 for p in phase["panels"].values())
        phase["status"] = "complete"
        save()

    receipt["status"] = "complete"
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(json.dumps({"status": "complete", "receipt": str(receipt_path)}), flush=True)
    return 0


# Outer process: fresh attempt directory, CUDA lock and host-memory guard -------------------------

def outer(args):
    attempt = args.output_dir / time.strftime("%Y%m%dT%H%M%S")
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--inner", "--attempt-dir", str(attempt),
               "--recipe", str(args.recipe), "--output-dir", str(args.output_dir), "--candidates", args.candidates,
               "--margin-gib", str(args.margin_gib), "--skip", args.skip]
    if args.batch_size is not None:
        command += ["--batch-size", str(args.batch_size)]
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
    receipt_path = attempt / "validation.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    decision = (receipt or {}).get("decision") or {}
    full = ((receipt or {}).get("phases") or {}).get("full") or {}
    summary = {"exit": code, "receipt": str(receipt_path), "status": (receipt or {}).get("status"),
               "recommended_batch_size": decision.get("recommended_batch_size"),
               "bound_batch_size": decision.get("bound_batch_size"), "full_panels_seconds": full.get("total_seconds")}
    needs_user = None
    if code != 0 or receipt is None or receipt.get("status") != "complete":
        needs_user = f"C07 batched-eval validation did not complete; see {attempt}"
    elif decision.get("rebind_needed") or decision.get("bound_fits_memory") is False:
        needs_user = (f"C07 batched-eval validation recommends batch {decision.get('recommended_batch_size')} "
                      f"(bound: {decision.get('bound_batch_size')}); see {receipt_path}")
    result = os.environ.get("SEPALITH_JOB_RESULT")
    if result:
        write_json(Path(result), {"summary": json.dumps(summary, sort_keys=True), "evaluations": [str(receipt_path)],
                                  "needs_user": needs_user})
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if code == 0 and receipt is not None and receipt.get("status") == "complete" else 1


def main(argv=None):
    args = parse_args(argv)
    if args.inner:
        return inner(args)
    return outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
