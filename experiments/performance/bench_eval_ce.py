"""Compare bounded evaluation logits against the historical full projection.

Default: tiny CPU correctness smoke. --measure explicitly enables timing.
Coordinate an idle resource window before timing; this script is not a scheduler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/training/poc_twin"))


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--resource-window", help="Recorded window identifier; obtain resource ownership first")
    ap.add_argument("--batch", type=positive, default=2)
    ap.add_argument("--seq", type=positive, default=65)
    ap.add_argument("--hidden", type=positive, default=32)
    ap.add_argument("--vocab", type=positive, default=257)
    ap.add_argument("--chunk", type=positive, default=32)
    ap.add_argument("--repeats", type=positive, default=5)
    ap.add_argument("--threads", type=positive, default=1)
    ap.add_argument("--matmul-precision", choices=("highest", "high"), default="high")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.threads > 8:
        ap.error("shared-machine policy limits CPU threads to 8")
    if args.measure and (not args.resource_window or not 3 <= args.repeats <= 30):
        ap.error("measurement requires --resource-window and 3..30 repeats")
    if args.device == "cuda" and not args.measure:
        ap.error("CUDA requires an explicitly coordinated --measure run")

    if args.output and args.output.exists():
        ap.error("output already exists; choose a new receipt path")

    import torch
    import torch.nn.functional as F
    from model import chunked_eval_ce

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(1273)
    h = torch.randn(args.batch, args.seq, args.hidden).to(args.device)
    w = (torch.randn(args.vocab, args.hidden) * 0.02).to(args.device)
    targets = torch.randint(args.vocab, (args.batch, args.seq)).to(args.device)

    @torch.no_grad()
    def reference():
        logits = F.linear(h, w)
        lg = logits.view(-1, logits.size(-1))
        tg = targets.reshape(-1)
        total = 0.0
        for start in range(0, tg.numel(), args.chunk):
            total += F.cross_entropy(lg[start:start + args.chunk].float(),
                                     tg[start:start + args.chunk], reduction="sum").item()
        return total

    def candidate():
        total, count = chunked_eval_ce(h, w, targets, chunk=args.chunk)
        if count != targets.numel():
            raise RuntimeError("candidate changed the token count")
        return total

    def check(actual, expected):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-4):
            raise RuntimeError(f"CE parity failed: candidate={actual}, reference={expected}")

    expected, actual = reference(), candidate()
    check(actual, expected)
    report = {
        "schema_version": 1,
        "kind": "paired_measurement" if args.measure else "correctness_smoke",
        "parity_pass": True, "reference_ce_sum": expected, "candidate_ce_sum": actual,
        "tokens": targets.numel(),
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "runtime": {"torch": torch.__version__, "python": platform.python_version(),
                    "platform": platform.platform(), "threads": torch.get_num_threads(),
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu"},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "experiments/training/poc_twin/model.py")},
        "limits": "Synthetic evaluation projection only; no training-step or serving speed claim.",
    }
    if args.measure:
        sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
        for _ in range(2):
            reference()
            candidate()
        sync()
        timings = {"reference": [], "candidate": []}
        peaks = {"reference": [], "candidate": []}
        for rep in range(args.repeats):
            order = (("reference", reference), ("candidate", candidate))
            if rep % 2:
                order = order[::-1]
            for name, fn in order:
                sync()
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    resident = torch.cuda.memory_allocated()
                start = time.perf_counter_ns()
                value = fn()
                sync()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                check(value, expected)
                timings[name].append(elapsed_ms)
                if args.device == "cuda":
                    peaks[name].append(torch.cuda.max_memory_allocated() - resident)
        medians = {name: statistics.median(values) for name, values in timings.items()}
        report.update(samples_ms=timings, median_ms=medians,
                      reference_over_candidate=medians["reference"] / medians["candidate"],
                      incremental_peak_allocated_bytes=peaks)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Preserve prior runs rather than silently replacing evidence.
        with args.output.open("x") as f:
            f.write(rendered + "\n")
    print(rendered)
    if args.measure:
        print(f"METRIC candidate_ms={report['median_ms']['candidate']}")
        print(f"METRIC speedup={report['reference_over_candidate']}")


if __name__ == "__main__":
    main()
