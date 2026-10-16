#!/usr/bin/env python3
"""C09 day-after: CPT 2K held-out causal loss (forgetting check) for SFT checkpoints.

    .venv-sft/bin/python -B docs/training/plan-20260927/c09/cpt_forgetting.py \\
        --model parent=/mnt/e/.../checkpoint-11586 --model arm-a-536=/mnt/e/.../checkpoint-536 \\
        --panel /mnt/e/sepalith/campaign-20260915/data-work/C09-cpt-validation-2k/.../cpt_validation-2k.jsonl \\
        --output /mnt/e/sepalith/campaign-20260915/quant-work/arm-a/cpt-forgetting.json --device cuda --gpu-window

The panel is the CPT 2K validation file (499 rows, 50 packages, sha256
``efb43495...``; public copy ``campaign-20260915/cpt/validation/cpt_validation-2k.jsonl``
in ``scholzmx/sepalith``).  Each model goes through the CPT trainer's own
evaluator, ``campaign_cpt_eval.package_holdout_evaluator`` (batch size 1,
token-weighted mean causal NLL), so the number is the one the CPT run
reported.  Loading follows the trainer as closely as plain transformers
allows: bf16 weights, the 85 FP32 norm tensors restored with
``restore_saved_fp32``, and the forward pass under bf16 autocast.

The first ``--model`` is the reference.  Pass the CPT parent first: it
reproduces the trainer's 0.94197 and so measures this harness's offset, and
every later model gets a paired delta against it (per-row wins and a
package-cluster bootstrap 95% interval) as well as a delta against the
recorded 0.94197.  The result is a diagnostic, not a gate: no rule in
RULES.md thresholds it.

GPU use needs ``--device cuda --gpu-window`` (night window only).  CPU smoke
runs (``--device cpu``) use at most 8 threads; run them under ``nice -n 10``.
``--limit`` takes the first N rows and marks the result as a smoke.
If ``SEPALITH_JOB_RESULT`` is set (night runner), a short summary goes there.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages" / "sepalith" / "src"))

PANEL_SHA256 = "efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8"
PANEL_ROWS = 499
MAX_SEQUENCE_TOKENS = 2048
PARENT_RECORDED_NLL = 0.9419698406022244  # trainer value for CPT step 11,586 (production-20260927/comparison.json)
PARENT_WEIGHTS_SHA256 = "4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51"
EXPECTED_TENSORS, EXPECTED_PARAMETERS = 381, 2_516_756_480
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10_000, 3407
SEALED_MARKERS = ("DAT-07-final", "sealed-final")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=DIR",
                    help="HF checkpoint dir; the first one is the paired reference (pass the CPT parent)")
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new result JSON; per-row losses go to <output>.rows.json")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--gpu-window", action="store_true", help="required for --device cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N rows only (smoke)")
    return ap.parse_args(argv)


def package_bootstrap(reference: list[dict], candidate: list[dict]) -> list[float]:
    """95% interval of the token-weighted NLL delta (candidate - reference), resampling packages."""
    by_package: dict[str, list[float]] = {}
    for ref, cand in zip(reference, candidate, strict=True):
        if ref["row_id"] != cand["row_id"] or ref["loss_tokens"] != cand["loss_tokens"]:
            raise SystemExit("row order or token counts differ between models")
        totals = by_package.setdefault(ref["package_id"], [0.0, 0.0, 0.0])
        totals[0] += ref["loss_sum"]
        totals[1] += cand["loss_sum"]
        totals[2] += ref["loss_tokens"]
    packages = list(by_package.values())
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [packages[rng.randrange(len(packages))] for _ in packages]
        tokens = sum(p[2] for p in draw)
        deltas.append((sum(p[1] for p in draw) - sum(p[0] for p in draw)) / tokens)
    deltas.sort()
    return [deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]]


def evaluate_model(label: str, directory: Path, recipe: dict, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    from sepalith.training.checkpoint.saved_precision import restore_saved_fp32
    from sepalith.training.eval.campaign_cpt_eval import package_holdout_evaluator

    weights = directory / "model.safetensors"
    started = time.monotonic()
    weights_sha = sha256(weights)
    model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=torch.bfloat16, device_map={"": device},
                                                 local_files_only=True, trust_remote_code=False)
    named = list(model.named_parameters())
    if len(named) != EXPECTED_TENSORS or sum(p.numel() for _, p in named) != EXPECTED_PARAMETERS:
        raise SystemExit(f"{label}: not the dense 2.52B checkpoint")
    precision = restore_saved_fp32(model, weights)
    model.config.use_cache = False
    loaded = time.monotonic()
    evaluate = package_holdout_evaluator(recipe)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        result = evaluate(model, None, directory, None)
    finished = time.monotonic()
    del model
    if device == "cuda":
        peak = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        peak = None
    return {"label": label, "checkpoint": str(directory), "model_safetensors_sha256": weights_sha,
            "fp32_tensors_restored": precision["fp32_tensors_restored"],
            "mean_causal_nll": result["metrics"]["mean_causal_nll"], "denominators": result["denominators"],
            "row_metrics": result["row_metrics"], "load_seconds": round(loaded - started, 1),
            "eval_seconds": round(finished - loaded, 1), "peak_cuda_allocated_bytes": peak}


def main(argv=None) -> int:
    a = parse_args(argv)
    models = []
    for item in a.model:
        label, _, path = item.partition("=")
        directory = Path(path).resolve()
        if not label or not (directory / "model.safetensors").is_file() or not (directory / "config.json").is_file():
            raise SystemExit(f"bad --model {item!r}")
        models.append((label, directory))
    if len({label for label, _ in models}) != len(models):
        raise SystemExit("duplicate --model label")
    for path in (a.panel, a.output, *(d for _, d in models)):
        if any(marker in str(path) for marker in SEALED_MARKERS):
            raise SystemExit(f"refusing a sealed final-set path: {path}")
    if a.device == "cuda" and not a.gpu_window:
        raise SystemExit("CUDA runs only inside the night window (--gpu-window)")
    if a.device == "cpu":
        if not 1 <= a.threads <= 8:
            raise SystemExit("daytime CPU rule: 1 to 8 threads")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = a.output.resolve()
    rows_path = output.with_suffix(".rows.json")
    if output.exists() or rows_path.exists():
        raise SystemExit(f"result already exists: {output}")
    panel = a.panel.resolve()
    if sha256(panel) != PANEL_SHA256:
        raise SystemExit(f"{panel} is not the CPT 2K validation panel")

    import torch
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    elif not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cpt-forgetting-") as scratch:
        rows_file = panel
        if a.limit:
            rows_file = Path(scratch) / "panel-limit.jsonl"
            lines = panel.read_text().splitlines(keepends=True)[:a.limit]
            rows_file.write_text("".join(lines))
        recipe = {"validation_rows": {"path": str(rows_file)}, "validation_batch_size": 1,
                  "parameters": {"max_sequence_tokens": MAX_SEQUENCE_TOKENS}, "materialized_rows": True}
        runs = [evaluate_model(label, directory, recipe, a.device) for label, directory in models]

    expected_rows = a.limit or PANEL_ROWS
    for run in runs:
        if run["denominators"]["validation_rows"] != expected_rows:
            raise SystemExit(f"{run['label']}: evaluated {run['denominators']['validation_rows']} rows")
    reference = runs[0]
    parent_reproduction = None
    if reference["model_safetensors_sha256"] == PARENT_WEIGHTS_SHA256 and not a.limit:
        parent_reproduction = {"recorded": PARENT_RECORDED_NLL, "this_harness": reference["mean_causal_nll"],
                               "offset": reference["mean_causal_nll"] - PARENT_RECORDED_NLL}
    comparisons = []
    for run in runs[1:]:
        wins = sum(c["mean_causal_nll"] < r["mean_causal_nll"] for r, c in zip(reference["row_metrics"], run["row_metrics"], strict=True))
        delta = run["mean_causal_nll"] - reference["mean_causal_nll"]
        comparisons.append({
            "label": run["label"], "reference": reference["label"],
            "delta_vs_reference": delta, "relative_change_percent_vs_reference": 100 * delta / reference["mean_causal_nll"],
            "package_bootstrap95_delta": package_bootstrap(reference["row_metrics"], run["row_metrics"]),
            "rows_lower_than_reference": wins, "rows": len(run["row_metrics"]),
            "delta_vs_recorded_parent": run["mean_causal_nll"] - PARENT_RECORDED_NLL,
            "relative_change_percent_vs_recorded_parent": 100 * (run["mean_causal_nll"] - PARENT_RECORDED_NLL) / PARENT_RECORDED_NLL,
        })

    import transformers
    result = {
        "schema": "sepalith.c09.cpt-forgetting.v1", "status": "smoke" if a.limit else "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": {"path": str(panel), "sha256": PANEL_SHA256, "rows_evaluated": expected_rows,
                  "hf_copy": "scholzmx/sepalith:campaign-20260915/cpt/validation/cpt_validation-2k.jsonl"},
        "evaluator": "sepalith.training.eval.campaign_cpt_eval.package_holdout_evaluator (batch 1, max 2,048 tokens)",
        "precision": "bf16 weights, saved FP32 norm tensors restored, bf16 autocast",
        "runtime": {"device": a.device, "torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "threads": a.threads if a.device == "cpu" else None},
        "models": [{k: v for k, v in run.items() if k != "row_metrics"} for run in runs],
        "parent_recorded_nll": PARENT_RECORDED_NLL, "parent_reproduction": parent_reproduction,
        "comparisons": comparisons, "rows_file": str(rows_path),
        "gate": "diagnostic only; no rule thresholds the CPT 2K loss after SFT",
        "final_set_access": False, "wall_seconds": round(time.monotonic() - started, 1),
    }
    write_json(rows_path, {"panel_sha256": PANEL_SHA256,
                           "models": {run["label"]: run["row_metrics"] for run in runs}})
    result["rows_sha256"] = sha256(rows_path)
    write_json(output, result)
    summary = {run["label"]: round(run["mean_causal_nll"], 5) for run in runs}
    print(json.dumps({"status": result["status"], "output": str(output), "mean_causal_nll": summary,
                      "parent_reproduction": parent_reproduction, "comparisons": comparisons}, indent=2))
    job_result = os.environ.get("SEPALITH_JOB_RESULT")
    if job_result:
        write_json(Path(job_result), {"evaluations": [{"name": "cpt-2k-forgetting", "output": str(output),
                                                       "mean_causal_nll": summary,
                                                       "parent_reproduction": parent_reproduction}],
                                      "summary": "C09 CPT 2K forgetting check: " + ", ".join(
                                          f"{k} {v}" for k, v in summary.items())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
