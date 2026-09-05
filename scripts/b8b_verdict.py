#!/usr/bin/env python3
"""B8b verdict helper: paired stats vs the banked b4 control.

Assembles from persisted per-example rows (repo experiments/eval/):
  - scenarios: valid/exact McNemar (exact binomial, b8_mcnemar.mcnemar)
    + per-family valid/exact table (format_propagation = the format family)
  - noopFP: all-row proposal rate + scored FPR (expectation==no_proposal),
    plus paired McNemar on proposal by row id
  - midtyping: (i,sha) join check vs banked b4 rows + mean line_f1
Usage: python3 scripts/b8b_verdict.py [arm_stem]   (default b8b_stacked_qwen35_2b)
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/m0hawk/Documents/Sepalith")
EV = REPO / "experiments" / "eval"
sys.path.insert(0, str(EV))
from b8_mcnemar import mcnemar, load  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "b8b_stacked_qwen35_2b"
CTL = "b4_qwen35_2b"


def jload(p, row_keys=None):
    out = []
    for l in open(p):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if row_keys is None or all(k in r for k in row_keys):
            out.append(r)
    return out


A, C = load(ARM), load(CTL)
common = sorted(set(A) & set(C))
print(f"scenarios: paired rows {len(common)} (arm {len(A)}, ctl {len(C)})")
for key in ("valid_pass", "exact"):
    ar = 100.0 * sum(bool(A[k][key]) for k in A) / max(len(A), 1)
    cr = 100.0 * sum(bool(C[k][key]) for k in C) / max(len(C), 1)
    b01, b10, p = mcnemar(C, A, key)
    print(f"  {key:10s} arm {ar:5.1f}% vs b4 {cr:5.1f}% | "
          f"discord ctl+/arm- {b01}, ctl-/arm+ {b10} | exact p={p:.6g}")

print("per-family (arm vs b4): valid% / exact% (n)")
fams = sorted({r["family"] for r in C.values()})
for f in fams:
    ka = [k for k in A if A[k]["family"] == f]
    kc = [k for k in C if C[k]["family"] == f]
    av = 100 * sum(bool(A[k]["valid_pass"]) for k in ka) / max(len(ka), 1)
    cv = 100 * sum(bool(C[k]["valid_pass"]) for k in kc) / max(len(kc), 1)
    ax = 100 * sum(bool(A[k]["exact"]) for k in ka) / max(len(ka), 1)
    cx = 100 * sum(bool(C[k]["exact"]) for k in kc) / max(len(kc), 1)
    # family-scoped discord (small n -> read as context, exact binomial underpowered)
    d01 = sum(1 for k in ka if bool(C[k]["valid_pass"]) and not bool(A[k]["valid_pass"]))
    d10 = sum(1 for k in ka if not bool(C[k]["valid_pass"]) and bool(A[k]["valid_pass"]))
    print(f"  {f:22s} {av:5.1f}/{ax:5.1f}  vs  {cv:5.1f}/{cx:5.1f}  (n={len(ka)}; "
          f"discord {d01}/{d10})")

# ---- noopFP ----
for stem, label in ((ARM, "arm"), (CTL, "b4 ")):
    try:
        rows = jload(EV / f"results_noop_fp_{stem}.jsonl")
    except FileNotFoundError:
        print(f"noopFP: {label} rows missing"); continue
    allr = 100.0 * sum(bool(r.get("proposal")) for r in rows) / len(rows)
    sc = [r for r in rows if r["expectation"] == "no_proposal"]
    fpr = 100.0 * sum(bool(r.get("proposal")) for r in sc) / max(len(sc), 1)
    print(f"noopFP {label}: all {allr:.1f}% (n={len(rows)}) | scored {fpr:.1f}% (n={len(sc)})")
ra = {(r["id"]): r for r in jload(EV / f"results_noop_fp_{ARM}.jsonl")}
rc = {(r["id"]): r for r in jload(EV / f"results_noop_fp_{CTL}.jsonl")}
b01, b10, p = mcnemar(rc, ra, "proposal")
print(f"noopFP paired McNemar (proposal): ctl+/arm- {b01}, ctl-/arm+ {b10}, p={p:.6g}")

# ---- midtyping join check + line_f1 ----
for suf in ("", "_suffix"):
    try:
        ma = jload(EV / f"results_{ARM}_midtyping{suf}.jsonl", row_keys=("i", "sha", "line_f1"))
        mc = jload(EV / f"results_{CTL}_midtyping{suf}.jsonl", row_keys=("i", "sha", "line_f1"))
    except FileNotFoundError:
        print(f"midtyping{suf or '.raw'}: rows missing"); continue
    ka = [(r["i"], r["sha"]) for r in ma]
    kc = [(r["i"], r["sha"]) for r in mc]
    join = "PASS" if ka == kc and len(ka) == 18 else "FAIL"
    fa = sum(r["line_f1"] for r in ma) / max(len(ma), 1)
    fc = sum(r["line_f1"] for r in mc) / max(len(mc), 1)
    ex_a = sum(bool(r["exact"]) for r in ma)
    ex_c = sum(bool(r["exact"]) for r in mc)
    print(f"midtyping{' raw' if not suf else ' suffix'}: join {join} "
          f"({len(ka)}/18 keys, same order={ka == kc}) | line_f1 arm "
          f"{fa:.3f} vs b4 {fc:.3f} | exact arm {ex_a}/18 vs b4 {ex_c}/18")
