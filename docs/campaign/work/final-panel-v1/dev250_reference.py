#!/usr/bin/env python3
"""DEV250 reference rates of the plan roster, reweighted to the final panel's family mix.

The final panel has 122 code-cut finish cases and no roxygen, against DEV250's
55 authored and 56 code-cut finish cases and 11 roxygen cases. Its primary
rate is not comparable with score250 unless both use one mix. This script
reads only dev material: the DEV250 panel and the existing per-case DEV250
results of each roster model. It writes one aggregate receipt.

Each family's DEV250 rate is weighted by that family's share of the final mix.
finish_block uses the code-cut DEV250 cases only, because every final finish
case is code-cut. The 95% intervals come from a stratified case bootstrap
(10,000 replicates, seed 3407): within each family, the cases are resampled
with replacement, and the same draws serve every model, so differences are
paired.

    python docs/campaign/work/final-panel-v1/dev250_reference.py OUT.json [--mix build-receipt.json]

`--mix` adds the rates at a build's actual selected family counts (a build
receipt holds aggregate counts only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

Json = dict[str, Any]
SEED = 3407
REPS = 10_000
PANEL = {"path": "/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/dev250-v1.jsonl",
         "sha256": "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22"}
GATES = "/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates"
RESULTS = {
    "arm-a-803": {
        "path": f"{GATES}/SFT-v2-r4-lr1e-5/results/dev-dev250-step-803.20261014T031828-959466265.cases/cases.json",
        "runtime": "campaign_eval, Transformers bf16, CUDA, batch 32 (SFT-v2-r4 gate, 2026-10-14)"},
    "c19-100-bf16": {
        "path": f"{GATES}/C19-grpo-exec-pilot-1/selection/dev250-step-100.20261110T041705-881845839.cases/cases.json",
        "runtime": "campaign_eval, Transformers bf16, CUDA, batch 32 (C19 selection, 2026-11-10)"},
    "c19-100-q6_k": {
        "path": "/mnt/e/sepalith/campaign-20260915/quant-work/c19-100/eval/Q6_K-dev250.cases/cases.json",
        "runtime": "C09 GGUF dev generation, llama.cpp (C17 quant gate, 2026-11-11)"},
}
COMPARISONS = (("c19-100-bf16", "arm-a-803"), ("c19-100-q6_k", "c19-100-bf16"))
# The recipe's final quotas: finish_block takes roxygen's 11 slots, all code-cut.
FINAL_MIX = {"finish_block": 122, "no_op": 50, "pipe_rewrite": 28, "format_propagation": 24,
             "rename_propagation": 23, "na_rm_propagation": 3}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratum(row: Json) -> str:
    """DEV250 stratum: finish_block splits into its authored and code-cut halves by source file."""
    if row["family"] != "finish_block":
        return row["family"]
    return "finish_authored" if "authored" in row["source_ref"]["file"] else "finish_codecut"


def correct(result: Json) -> int:
    return int(bool(result["exact_region"]) if not result["expected_noop"] else bool(result["predicted_noop"]))


def weighted(rates: dict[str, float], mix: dict[str, int]) -> float:
    total = sum(mix.values())
    return sum(n * rates["finish_codecut" if f == "finish_block" else f] for f, n in mix.items() if n) / total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--mix", type=Path, help="a final-panel build receipt; adds rates at its selected mix")
    args = parser.parse_args(argv)
    if sha256_file(Path(PANEL["path"])) != PANEL["sha256"]:
        raise SystemExit("DEV250 panel differs from its pin")
    rows = {}
    for line in Path(PANEL["path"]).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = stratum(row)
    strata = sorted(set(rows.values()))
    ids = {s: sorted(i for i, v in rows.items() if v == s) for s in strata}

    outcome: dict[str, dict[str, int]] = {}
    sources = {}
    for label, source in RESULTS.items():
        results = json.loads(Path(source["path"]).read_bytes())["results"]
        if sorted(r["id"] for r in results) != sorted(rows):
            raise SystemExit(f"{label}: results do not cover DEV250 exactly")
        outcome[label] = {r["id"]: correct(r) for r in results}
        sources[label] = {**source, "sha256": sha256_file(Path(source["path"]))}

    mixes = {"final_quota_mix": FINAL_MIX}
    if args.mix:
        build = json.loads(args.mix.read_bytes())
        mixes["build_selected_mix"] = {f: n for f, n in build["allocation"]["selected_by_family"].items() if n}

    per_model: Json = {}
    for label, by_id in outcome.items():
        counts = {s: [sum(by_id[i] for i in ids[s]), len(ids[s])] for s in strata}
        rates = {s: c / n for s, (c, n) in counts.items()}
        per_model[label] = {"score250": sum(by_id.values()), "strata": counts,
                            "rates": {name: weighted(rates, mix) for name, mix in mixes.items()}}

    rng = random.Random(SEED)
    samples: dict[str, dict[str, list[float]]] = {name: {label: [] for label in outcome} for name in mixes}
    for _ in range(REPS):
        draw = {s: [ids[s][rng.randrange(len(ids[s]))] for _ in ids[s]] for s in strata}
        for label, by_id in outcome.items():
            rates = {s: sum(by_id[i] for i in draw[s]) / len(draw[s]) for s in strata}
            for name, mix in mixes.items():
                samples[name][label].append(weighted(rates, mix))

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [ordered[len(ordered) // 40 - 1], ordered[39 * len(ordered) // 40 - 1]]

    for label in outcome:
        per_model[label]["ci95"] = {name: interval(samples[name][label]) for name in mixes}
    comparisons = {}
    for a, b in COMPARISONS:
        comparisons[f"{a}_minus_{b}"] = {
            name: {"delta": per_model[a]["rates"][name] - per_model[b]["rates"][name],
                   "ci95": interval([x - y for x, y in zip(samples[name][a], samples[name][b], strict=True)])}
            for name in mixes}
    receipt = {
        "schema": "sepalith.final-panel.v1.dev250-reference",
        "panel": PANEL, "results": sources, "mixes": mixes,
        "method": "per-family DEV250 rate weighted by the mix; finish_block uses the 56 code-cut DEV250 cases; "
                  "stratified case bootstrap, 10,000 replicates, seed 3407, paired across models",
        "final_content_opened": False,
        "models": per_model, "comparisons": comparisons,
    }
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({label: {"score250": m["score250"], **m["rates"]} for label, m in per_model.items()},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
