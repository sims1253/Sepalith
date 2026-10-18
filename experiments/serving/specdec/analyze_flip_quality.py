#!/usr/bin/env python3
"""Flip quality-neutrality analysis (TRAIN ground truth, existing artifacts).

For the two full-set gate runs (8T rep2, 4T), compare baseline and spec
greedy outputs against the dataset targets: exact-match rates overall, on
the flipped subset, well-formedness (terminal line), and divergence sizes.

Emits METRIC lines for the autoresearch loop.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(
    "/mnt/e/sepalith/campaign-20260915/serving-work/specdec-20261017/runs"
)
EOS_IDS = {1, 130073}
TERMINAL = [97666, 24779, 651, 8089, 23877]


def norm(tokens: list[int] | None) -> list[int]:
    if not tokens:
        return []
    t = list(tokens)
    while t and t[-1] in EOS_IDS:
        t.pop()
    return t


def load_run(dir_name: str) -> dict[str, dict[str, list[int]]]:
    rows = [json.loads(line) for line in (RUNS / dir_name / "per_request.jsonl").read_text().splitlines()]
    out: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        arm = "base" if r["arm"] == "baseline" else "spec"
        out.setdefault(arm, {})[r["trace_id"]] = r.get("tokens") or []
    return out


def main() -> None:
    traces = [json.loads(line) for line in (HERE / "traces.jsonl").read_text().splitlines()]
    target = {t["trace_id"]: norm(t["target_ids"]) for t in traces}

    for label, dir_name in (("8T", "20261018T104958-ngsimple-n2m4"),
                            ("4T", "20261018T182701-ngsimple-n2m4")):
        run = load_run(dir_name)
        common = [t for t in target if t in run.get("base", {}) and t in run.get("spec", {})]
        flips = [t for t in common if run["base"][t] != run["spec"][t]]

        base_match = [t for t in common if norm(run["base"][t]) == target[t]]
        spec_match = [t for t in common if norm(run["spec"][t]) == target[t]]
        # on the flipped subset: who matches the ground truth?
        fb = [t for t in flips if norm(run["base"][t]) == target[t]]
        fs = [t for t in flips if norm(run["spec"][t]) == target[t]]
        def wellformed(toks: list[int], term: list[int] = TERMINAL) -> bool:
            n = norm(toks)
            return n[-5:] == term or n[:2] == [41448, 369]  # terminal or [NO_EDIT]-ish
        fw_b = [t for t in flips if wellformed(run["base"][t])]
        fw_s = [t for t in flips if wellformed(run["spec"][t])]
        # divergence magnitude on flips
        def prefix(a: list[int], b: list[int]) -> int:
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            return i
        div_frac = [
            prefix(run["base"][t], run["spec"][t]) / max(1, len(run["base"][t]))
            for t in flips
        ]

        stats = {
            "n": len(common), "flips": len(flips),
            "base_exact": len(base_match), "spec_exact": len(spec_match),
            "flip_base_exact": len(fb), "flip_spec_exact": len(fs),
            "flip_both_mismatch": len([t for t in flips
                                       if t not in fb and t not in fs]),
            "flip_base_wellformed": len(fw_b), "flip_spec_wellformed": len(fw_s),
            "flip_median_agreement_frac": round(statistics.median(div_frac), 3)
            if div_frac else None,
        }
        print(f"[{label}] {json.dumps(stats)}", flush=True)
        tag = "" if label == "8T" else "_4t"
        print(f"METRIC spec_exact_match{tag}={len(spec_match) / len(common):.4f}")
        print(f"METRIC base_exact_match{tag}={len(base_match) / len(common):.4f}")
        print(f"METRIC flip_spec_exact_frac{tag}={len(fs) / max(1, len(flips)):.4f}")
        print(f"METRIC flip_base_exact_frac{tag}={len(fb) / max(1, len(flips)):.4f}")
        print(f"METRIC flip_spec_wellformed_frac{tag}={len(fw_s) / max(1, len(flips)):.4f}")
        out = RUNS / dir_name / "flip_quality.json"
        out.write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
