#!/usr/bin/env python3
"""B8 verdict helper: paired McNemar vs the banked b4 control (exact binomial).

Pairs per-example rows from results_scenarios_b8_midtrain_qwen35_2b.jsonl
against results_scenarios_b4_qwen35_2b.jsonl on the eval row key (i, id).
Reports discordant counts + two-sided exact p for each binary metric
(valid_pass, exact). Same statistic the gate B-β audit used (audit
vocabulary: p<0.05 call, else TIE/TIE-UNDERPOWERED depending on n_discord).
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).parent


def load(stem):
    rows = [json.loads(l) for l in open(HERE / f"results_scenarios_{stem}.jsonl")]
    return {(r["i"], r["id"]): r for r in rows}


def mcnemar(a, b, key):
    """a, b: dicts key->row. Returns (b01, b10, p_two_sided_exact_binomial)."""
    common = sorted(set(a) & set(b))
    b01 = b10 = 0  # b01: control+ / arm-, b10: control- / arm+
    for k in common:
        x, y = bool(a[k][key]), bool(b[k][key])
        if x and not y:
            b01 += 1
        elif y and not x:
            b10 += 1
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # exact two-sided binomial test at p=0.5 (sum of tails <= observed tail mass)
    k = min(b01, b10)
    def pmf(i):
        return math.comb(n, i) / (2.0 ** n)
    p_obs = pmf(k)
    p = sum(pmf(i) for i in range(n + 1) if pmf(i) <= p_obs + 1e-15)
    return b01, b10, min(1.0, p)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "b8_midtrain_qwen35_2b"
    ctl = sys.argv[2] if len(sys.argv) > 2 else "b4_qwen35_2b"
    A, C = load(arm), load(ctl)
    print(f"paired rows: {len(set(A) & set(C))} (arm {len(A)}, control {len(C)})")
    for key in ("valid_pass", "exact"):
        ar = 100.0 * sum(bool(r[key]) for r in A.values()) / max(len(A), 1)
        cr = 100.0 * sum(bool(r[key]) for r in C.values()) / max(len(C), 1)
        b01, b10, p = mcnemar(C, A, key)
        print(f"{key}: arm {ar:.1f}% vs control {cr:.1f}% | "
              f"discord ctl+/arm- {b01}, ctl-/arm+ {b10} | exact p={p:.4f}")
    # per-family valid deltas (context for the verdict table)
    fams = sorted({r["family"] for r in C.values()})
    print("per-family valid% (arm vs ctl):")
    for f in fams:
        ai = [bool(A[k]["valid_pass"]) for k in A if A[k]["family"] == f]
        ci = [bool(C[k]["valid_pass"]) for k in C if C[k]["family"] == f]
        if ai and ci:
            print(f"  {f:24s} {100*sum(ai)/len(ai):5.1f} (n={len(ai)})  vs  "
                  f"{100*sum(ci)/len(ci):5.1f} (n={len(ci)})")


if __name__ == "__main__":
    main()
