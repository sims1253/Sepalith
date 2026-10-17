#!/usr/bin/env python3
"""Receipt for the C12 vLLM census-throughput trial (PC side, CPU only).

Reads the fetched Kaggle run of ``scripts/kaggle/vllm_census_entry.py`` and the
PC's bf16 references for arm A (checkpoint 536: the night gate's DEV75 and
DEV250 evaluator ``cases.json``, bound batching: batch 32, length-descending,
left pad 1, dynamic cache, cap 1,024) and writes one JSON receipt:

- throughput: session generated tokens/s (both T4s) for census sampling, the
  speed-up over the bound-batching path's 106 tokens/s per two-T4 session
  (C03), and the projected Kaggle GPU-hours of the full census (15,006 TRAIN
  prompts x 8 samples, cap 1,024) and of reduced variants (k = 4, cap 512),
  with per-session start-up overhead;
- sampling statistics of the trial subset: completion length, cap hits, stop
  ids, and a token-exact match to the TRAIN target per sample (a lower bound of
  reward v2.1's ``exact``, which canonicalizes with air), the share of prompts
  that are all-correct / all-wrong / mixed under that proxy;
- greedy parity (fp16): vLLM vs the PC bf16 references on DEV75 and DEV250,
  vLLM vs the bound-batching transformers fp16 run on DEV75 (same session),
  transformers fp16 vs bf16 on DEV75, and vLLM against itself (DEV75 alone vs
  inside DEV75+DEV250): token identity, first divergence, the R1 counts
  (``valid``, ``cap``, ``edit``, ``nofp``) and per-case flag agreement; target
  NLL (teacher-forced) per panel on each side.

The C03 parity criteria (DEV75 counts within 1, >= 90% token identity, no NaN)
are reported for reference; for the census the decisive checks are the counts
and the NLL, since token identity under different batching is known to be low
even between two bf16 runs on the PC (54% batch 1 vs 32 on the parent).

Example::

    python3 docs/training/plan-20260927/c12/vllm_trial_receipt.py \\
        --run ~/.local/state/sepalith/kaggle/runs/c12-vllm-trial/c12-vllm-trial \\
        --output docs/training/plan-20260927/receipts/C12-vllm-trial.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c03"))
from compare_parity import identity, r1_comparison  # noqa: E402

GATES = Path.home() / ".local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5/results"
REFERENCES = {
    "dev75": GATES / "dev-dev75-step-536.20261014T021133-673743425.cases/cases.json",
    "dev250": GATES / "dev-dev250-step-536.20261014T021133-673743425.cases/cases.json",
}
BOUND_SESSION_TPS = 106.0  # C03: bound batching, batch 32, two T4s, parent rows to the cap
TRAIN_PROMPTS = 15006
SESSION_HOURS = 11.5  # usable hours of a 12-hour Kaggle session


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def cases(path):
    return {row["id"]: row for row in json.loads(Path(path).read_text())["results"]}


def by_panel(records, panel):
    return {r["id"]: r for r in records if r["panel"] == panel}


def nll(rows):
    tokens = sum(r["target_tokens"] for r in rows)
    return sum(r["target_nll_sum"] for r in rows) / tokens if tokens else None


def reference_nll(results):
    tokens = sum(r["loss"]["target_tokens"] for r in results.values())
    return sum(r["loss"]["target_nll_sum"] for r in results.values()) / tokens


def compare(reference, candidate):
    entry = {"identity": identity(reference, candidate)}
    entry["identity"].pop("divergent")
    entry["r1"] = r1_comparison(reference, candidate)
    return entry


def quantiles(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"mean": round(statistics.mean(values), 1), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99),
            "max": values[-1]}


def sampling_stats(samples, prompts):
    target = {p["id"]: p["target_ids"] for p in prompts}
    lengths = [s["generated_tokens"] for s in samples]
    per_prompt = collections.defaultdict(list)
    for s in samples:
        per_prompt[s["id"]].append(s["generated_ids"] == target[s["id"]])
    shares = collections.Counter("all_correct" if all(v) else "all_wrong" if not any(v) else "mixed"
                                 for v in per_prompt.values())
    families = collections.defaultdict(list)
    for s in samples:
        families[s["family"]].append(s)
    return {
        "samples": len(samples), "prompts": len(per_prompt),
        "generated_tokens": quantiles(lengths),
        "cap_hits": sum(s["cap_hit"] for s in samples),
        "cap_hit_rate": round(sum(s["cap_hit"] for s in samples) / len(samples), 4),
        "stop_ids": dict(collections.Counter(str(s["stop_token_id"]) for s in samples)),
        "token_exact_proxy": {
            "note": "generated ids == TRAIN target ids (incl. EOS); a lower bound of reward v2.1 exact",
            "sample_rate": round(sum(v for vs in per_prompt.values() for v in vs) / len(samples), 4),
            "prompt_shares": {k: round(shares[k] / len(per_prompt), 4) for k in ("all_correct", "all_wrong", "mixed")},
        },
        "per_family": {f: {"samples": len(v), "mean_tokens": round(statistics.mean(s["generated_tokens"] for s in v), 1),
                           "cap_hit_rate": round(sum(s["cap_hit"] for s in v) / len(v), 4),
                           "token_exact_rate": round(sum(s["generated_ids"] == target[s["id"]] for s in v) / len(v), 4)}
                       for f, v in sorted(families.items())},
    }


def projection(tps, lengths, overhead_hours):
    """Kaggle GPU-hours (session hours; two T4s per session) for census variants at the measured throughput."""
    out = {}
    for k, cap in ((8, 1024), (8, 512), (4, 1024), (4, 512)):
        mean_tokens = statistics.mean(min(n, cap) for n in lengths)
        tokens = TRAIN_PROMPTS * k * mean_tokens
        hours = tokens / tps / 3600
        sessions = max(1, math.ceil(hours / SESSION_HOURS))
        out[f"k{k}_cap{cap}"] = {"mean_tokens_per_sample": round(mean_tokens, 1), "generated_tokens": round(tokens),
                                 "generation_hours": round(hours, 2), "sessions": sessions,
                                 "gpu_hours_with_overhead": round(hours + sessions * overhead_hours, 2)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="fetched run folder (summary.json, vllm-*.jsonl, hf-dev75/)")
    p.add_argument("--driver-receipt", type=Path, help="default: <run>/../driver-receipt.json")
    p.add_argument("--hf-dev75-run", type=Path,
                   help="C03's bound-batching fp16 DEV75 run of arm A (cases.json, summary.json); default <run>/hf-dev75")
    p.add_argument("--hf-train200-run", type=Path,
                   help="C03's bound-batching fp16 TRAIN-200 run of arm A (generations.jsonl, summary.json)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    run = a.run
    summary = json.loads((run / "summary.json").read_text())
    driver_path = a.driver_receipt or run.parent / "driver-receipt.json"
    driver = json.loads(driver_path.read_text()) if driver_path.is_file() else {}
    receipt = {"schema": "sepalith.c12.vllm-trial-receipt.v1", "run": str(run), "status": summary.get("status"),
               "kernel": driver.get("kernel"), "commit": driver.get("commit"),
               "gpu_hours_used": driver.get("gpu_hours_used"), "model": summary.get("model"),
               "inputs": summary.get("inputs"), "timeline": summary.get("timeline"),
               "vllm_install": summary.get("vllm_install"), "error": summary.get("error")}
    receipt["vllm_attempts"] = [{"spec": attempt.get("spec"), "skipped": attempt.get("skipped"),
                                 "workers": {w: [{k: v for k, v in x.items() if k != "log_tail"} for x in runs]
                                             for w, runs in (attempt.get("workers") or {}).items()}}
                                for attempt in summary.get("vllm_attempts", [])]
    workers = summary.get("vllm_workers") or []
    receipt["engine"] = [{k: w.get(k) for k in ("worker", "vllm", "torch", "device_name", "config", "engine_seconds",
                                                 "engine_kwargs", "engine", "dev_greedy", "dev75_rerun", "dev_nll",
                                                 "sampling_stopped_early")} for w in workers]
    tp = summary.get("throughput")
    receipt["throughput"] = tp
    samples = read_jsonl(run / "vllm-train-samples.jsonl")
    prompts = read_jsonl(run / "train-prompts.jsonl")
    if tp and samples:
        receipt["throughput"]["speedup_vs_bound_batching"] = round(tp["session_tokens_per_second"] / BOUND_SESSION_TPS, 2)
        receipt["sampling"] = sampling_stats(samples, prompts)
        timeline = dict(summary.get("timeline") or [])
        # Per-session overhead of a census session: bootstrap + model + vLLM install + engine start (no HF reference).
        install = max((x.get("seconds") or 0) for x in summary.get("vllm_install") or [{}])
        engine = max((w.get("engine_seconds") or 0) for w in workers) if workers else 0
        overhead = (timeline.get("model", 0) + install + engine + 60) / 3600
        receipt["projection"] = {"session_tokens_per_second": tp["session_tokens_per_second"],
                                 "overhead_hours_per_session": round(overhead, 3),
                                 "variants": projection(tp["session_tokens_per_second"],
                                                        [s["generated_tokens"] for s in samples], overhead),
                                 "bound_batching_reference": "C03: 67 to 133 session-hours for k8 cap 1,024"}
    # Greedy parity.
    dev = read_jsonl(run / "vllm-dev-greedy.jsonl")
    rerun = {r["id"]: r for r in read_jsonl(run / "vllm-dev75-rerun.jsonl")}
    nll_rows = read_jsonl(run / "vllm-dev-nll.jsonl")
    parity = {}
    refs = {name: cases(path) for name, path in REFERENCES.items()}
    hf_dir = a.hf_dev75_run or run / "hf-dev75"
    hf = cases(hf_dir / "cases.json") if (hf_dir / "cases.json").is_file() else None
    hf_train = None
    if a.hf_train200_run and (a.hf_train200_run / "generations.jsonl").is_file():
        hf_train = {r["id"]: r for r in read_jsonl(a.hf_train200_run / "generations.jsonl") if r.get("sample", 0) == 0}
    if dev:
        for panel in ("dev75", "dev250"):
            parity[f"vllm_fp16_vs_bf16_{panel}"] = compare(refs[panel], by_panel(dev, panel))
        if hf:
            parity["vllm_fp16_vs_hf_fp16_dev75"] = compare(hf, by_panel(dev, "dev75"))
        if hf_train:
            entry = {"identity": identity(hf_train, by_panel(dev, "train200"))}
            entry["identity"].pop("divergent")
            cand = by_panel(dev, "train200")
            ids = sorted(set(hf_train) & set(cand))
            entry["cap_hits"] = {"hf_fp16": sum(bool(hf_train[i]["cap_hit"]) for i in ids),
                                 "vllm_fp16": sum(bool(cand[i]["cap_hit"]) for i in ids)}
            entry["mean_tokens"] = {"hf_fp16": round(statistics.mean(len(hf_train[i]["generated_ids"]) for i in ids), 1),
                                    "vllm_fp16": round(statistics.mean(len(cand[i]["generated_ids"]) for i in ids), 1)}
            parity["vllm_fp16_vs_hf_fp16_train200"] = entry
        if rerun:
            parity["vllm_dev75_alone_vs_mixed"] = compare(by_panel(dev, "dev75"), rerun)
    if hf:
        parity["hf_fp16_vs_bf16_dev75"] = compare(refs["dev75"], hf)
    parity["target_nll"] = {
        "bf16_pc": {panel: round(reference_nll(refs[panel]), 4) for panel in refs},
        "hf_fp16_kaggle": {"dev75": round(reference_nll(hf), 4)} if hf else None,
        "vllm_fp16": {panel: (round(nll([r for r in nll_rows if r["panel"] == panel]), 4)
                              if any(r["panel"] == panel for r in nll_rows) else None) for panel in refs},
    }
    hf_summary = (summary.get("hf_dev75") or {}).get("result") or {}
    if (hf_dir / "summary.json").is_file():
        hf_summary = json.loads((hf_dir / "summary.json").read_text()).get("result") or hf_summary
    parity["nonfinite"] = {"hf_fp16_calls_with_nonfinite_logits": hf_summary.get("calls_with_nonfinite_logits"),
                           "hf_fp16_nll_finite": hf_summary.get("nll_finite"),
                           "vllm_nll_finite": all(math.isfinite(r["target_nll_sum"]) for r in nll_rows) if nll_rows else None}
    receipt["parity"] = parity
    vd = parity.get("vllm_fp16_vs_bf16_dev75")
    if vd:
        receipt["c03_criteria_vllm_vs_bf16_dev75"] = {
            "counts_within_1": vd["r1"]["within_tolerance"],
            "identity_at_least_90pct": (vd["identity"]["identity_rate"] or 0) >= 0.9,
            "identity_rate": vd["identity"]["identity_rate"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    brief = {"throughput": receipt.get("throughput"), "projection": (receipt.get("projection") or {}).get("variants"),
             "r1": {k: v["r1"]["difference"] for k, v in parity.items() if isinstance(v, dict) and "r1" in v},
             "identity": {k: v["identity"]["identity_rate"] for k, v in parity.items() if isinstance(v, dict) and "identity" in v},
             "nll": parity["target_nll"]}
    print(json.dumps(brief, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
