#!/usr/bin/env python3
"""tu2_readout.py — TU2 verdict math (queue §3 TU2, pre-registered).

Joins the per-arm eval_scenarios rows by row id (sha1(prompt)[:12], identical
across arms) and computes: pooled + per-family exact rates, deltas, exact
two-sided McNemar (sufficiency_judge.mcnemar_exact convention, re-implemented
stdlib-only so this runs anywhere), the noopFP guardrail table, and the
llama-bench t/s rows. Persists the paired per-example file (eval-v2
con-scorable: exact + valid_pass + pred per arm per row).

Pre-registered comparisons (TU2_RESULTS.md amendment):
  b624 vs a624  — primary: solve-gate filter effect (624/624 matched)
  c113 vs a624  — teacher-target vs raw (volume-asymmetric: 113 vs 624)
  b113 vs a624  — solve-gated@113 vs raw (same asymmetry)
  c113 vs b113  — the consistent-teacher mechanism, PROMPT-IDENTICAL pair
Verdict rule: WINNER-RESOLVE iff b624 or c113 beats a624 pooled on exact
(McNemar) with noopFP not worse than a624's.

Usage: python3 tu2_readout.py   (from anywhere; paths absolute)
Writes results/tu2_paired_scenarios.jsonl + results/tu2_verdict.json next to
this file; prints the readout block LAST.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
RESULTS = HERE / "results"
ARMS = ["a624", "b624", "c113", "b113"]  # a624 = (a') control; b113 optional
FAMILIES = ("rename_propagation", "pipe_rewrite", "format_propagation",
            "doc_sync", "na_rm_propagation")
# (arm, control) pairs; b/c discordant counts are (control-fail&arm-win,
# control-win&arm-fail) i.e. positive delta -> b > c
COMPARISONS = [("b624", "a624"), ("c113", "a624"), ("b113", "a624"),
               ("c113", "b113")]


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p (sufficiency_judge.py convention)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_scen(stem: str):
    p = EVAL / f"results_scenarios_tu2_{stem}.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def load_noop(stem: str):
    p = EVAL / f"results_noop_fp_tu2_{stem}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    fp_rows = [r for r in rows if r["expectation"] == "no_proposal"]
    fp = sum(1 for r in fp_rows if r.get("proposal"))
    per_class = {}
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        per_class[cls] = dict(
            n=len(rs),
            proposal_rate=round(sum(1 for r in rs if r.get("proposal"))
                                / len(rs), 4))
    return dict(model=f"tu2_{stem}", cases_total=len(rows),
                no_op_cases=len(fp_rows),
                FALSE_POSITIVE_RATE=round(fp / max(1, len(fp_rows)), 4),
                by_class=per_class)


def paired_mcnemar(arm_rows, ctl_rows, fam=None):
    """(n, arm_exact, ctl_exact, delta_pp, b, c, p) over joined ids."""
    ids = [i for i in arm_rows if i in ctl_rows
           and (fam is None or arm_rows[i]["family"] == fam)]
    a = sum(arm_rows[i]["exact"] for i in ids)
    c_ = sum(ctl_rows[i]["exact"] for i in ids)
    b = sum(1 for i in ids if not ctl_rows[i]["exact"] and arm_rows[i]["exact"])
    c = sum(1 for i in ids if ctl_rows[i]["exact"] and not arm_rows[i]["exact"])
    n = len(ids)
    return dict(n=n, arm_exact=a, ctl_exact=c_,
                arm_rate=round(a / n, 4) if n else None,
                ctl_rate=round(c_ / n, 4) if n else None,
                delta_pp=round(100 * (a - c_) / n, 2) if n else None,
                discordant_arm_only_win=b, discordant_ctl_only_win=c,
                mcnemar_p=round(mcnemar_exact(b, c), 6) if n else None)


def bench_row(stem: str):
    """(pp512, tg128) tokens/s from the chain's per-arm bench.log (one run
    per file; llama-bench prints the model's INTERNAL name, identical across
    arms, so any table row in this arm's own log is this arm's)."""
    p = Path(f"/mnt/h/sepalith/runs/tu2_{stem}/bench.log")
    if not p.exists():
        return None
    out = {}
    for line in p.read_text(errors="replace").splitlines():
        if "pp512" not in line and "tg128" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        # columns: |model|size|params|backend|threads|test|tps ± err|
        try:
            val = parts[7].split("±")[0].strip()
            out[parts[6]] = float(val)
            out["backend"] = parts[4]
        except (IndexError, ValueError):
            continue
    return out or None


def main():
    scen = {s: load_scen(s) for s in ARMS}
    have = [s for s in ARMS if scen[s]]
    noop = {s: load_noop(s) for s in ARMS}
    bench = {s: bench_row(s) for s in ARMS}

    # ---- pooled + per-family exact rates per arm --------------------------
    rates = {}
    for s in have:
        rows = list(scen[s].values())
        per_fam = {}
        for f in FAMILIES:
            rs = [r for r in rows if r["family"] == f]
            per_fam[f] = dict(n=len(rs),
                              exact=round(sum(r["exact"] for r in rs) / len(rs), 4)
                              if rs else None,
                              valid=round(sum(r["valid_pass"] for r in rs) / len(rs), 4)
                              if rs else None)
        rates[s] = dict(
            n=len(rows),
            exact=round(sum(r["exact"] for r in rows) / len(rows), 4),
            valid=round(sum(r["valid_pass"] for r in rows) / len(rows), 4),
            families=per_fam,
            mean_latency_s=round(sum(r["latency_s"] for r in rows) / len(rows), 2))

    # ---- pre-registered McNemar comparisons -------------------------------
    comps = {}
    for arm, ctl in COMPARISONS:
        if arm not in have or ctl not in have:
            continue
        pooled = paired_mcnemar(scen[arm], scen[ctl])
        fams = {f: paired_mcnemar(scen[arm], scen[ctl], f) for f in FAMILIES}
        comps[f"{arm}_vs_{ctl}"] = dict(pooled=pooled, families=fams)

    # ---- paired per-example persistence (eval-v2 re-scorable) -------------
    all_ids = sorted({i for s in have for i in scen[s]})
    RESULTS.mkdir(exist_ok=True)
    paired_path = RESULTS / "tu2_paired_scenarios.jsonl"
    with open(paired_path, "w") as fh:
        for i in all_ids:
            base = next(scen[s][i] for s in have if i in scen[s])
            rec = dict(id=i, family=base["family"], package=base["package"],
                       path=base["path"], note=base.get("note", ""))
            for s in have:
                r = scen[s].get(i)
                rec[s] = None if r is None else dict(
                    exact=r["exact"], valid_pass=r["valid_pass"],
                    fail_kind=r.get("fail_kind"), reward=r.get("reward"),
                    pred=r.get("pred"), latency_s=r.get("latency_s"))
            fh.write(json.dumps(rec) + "\n")

    out = dict(arms=rates, comparisons=comps, noop_fp=noop, bench_tps=bench,
               paired_rows=str(paired_path), n_paired=len(all_ids))
    (RESULTS / "tu2_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
