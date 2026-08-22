#!/usr/bin/env python3
"""Aggregate the occupancy-sweep JSONLs into the gate-5 verdict tables.

Per (model, ctx, nlen, pos) cell: n, exact rate (+ Wilson 95% CI), mean
edit similarity, near-miss/no-answer rates, reasoning-fallback rate.
Per (model, ctx): exact rate by nlen (the degradation curves) and the
hybrid-vs-dense delta. Reads whatever rows exist (partial batteries OK;
n per cell reported).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("meta") or "error" in r:
            continue
        rows.append(r)
    return rows


def agg(rows):
    """-> {(ctx,nlen,pos): stats}, {(ctx,nlen): stats}"""
    cell = defaultdict(list)
    for r in rows:
        cell[(r["ctx_target"], r["nlen_target"], r["pos"])].append(r)
    out = {}
    for k, rs in sorted(cell.items()):
        n = len(rs)
        ex = sum(r["exact"] for r in rs)
        sim = sum(r["edit_sim"] for r in rs) / n
        nm = sum(1 for r in rs if r["tax"] == "near_miss") / n
        na = sum(1 for r in rs if r["tax"] == "no_answer") / n
        fr = sum(r.get("from_reasoning", 0) for r in rs) / n
        lo, hi = wilson(ex / n, n)
        out[k] = dict(n=n, exact=ex / n, ci=(round(lo, 3), round(hi, 3)),
                      sim=round(sim, 3), near_miss=round(nm, 3),
                      no_answer=round(na, 3), from_reasoning=round(fr, 3),
                      ntok=sum(r["needle_tokens"] for r in rs) / n,
                      ptok=sum(r["prompt_tokens"] for r in rs) / n)
    return out


def collapse(cells):
    """(ctx,nlen) collapsed over positions, n-weighted."""
    out = {}
    for ctx in (1024, 2048, 4096, 8192):
        for nlen in (40, 100, 300):
            rs = [v for (c, l, _), v in cells.items() if c == ctx and l == nlen]
            if not rs:
                continue
            n = sum(v["n"] for v in rs)
            ex = sum(v["exact"] * v["n"] for v in rs) / n
            sim = sum(v["sim"] * v["n"] for v in rs) / n
            lo, hi = wilson(ex, n)
            out[(ctx, nlen)] = dict(n=n, exact=round(ex, 3),
                                    ci=(round(lo, 3), round(hi, 3)),
                                    sim=round(sim, 3))
    return out


def main():
    files = {"lfm25_hybrid": HERE / "results_occupancy_lfm25.jsonl",
             "dense_ref": HERE / "results_occupancy_dense_ref.jsonl"}
    all_cells, all_coll = {}, {}
    for name, p in files.items():
        if not p.exists():
            print(f"## {name}: MISSING {p}", file=sys.stderr)
            continue
        rows = load(p)
        label = rows[0]["model"] if rows else name
        mode = rows[0].get("mode", "?") if rows else "?"
        cells = agg(rows)
        coll = collapse(cells)
        all_cells[name], all_coll[name] = cells, coll
        print(f"===== {label} (mode={mode}, {len(rows)} rows) =====")
        print(f"{'ctx':>5} {'nlen':>5} {'n':>4} {'exact':>6} {'95% CI':>15} "
              f"{'sim':>6} {'near':>5} {'noans':>5} {'frmR':>5} "
              f"{'needle_tok':>10}")
        for (ctx, nlen), v in sorted(coll.items()):
            ci = f"[{v['ci'][0]:.2f},{v['ci'][1]:.2f}]"
            sub = [c for (cc, ll, _), c in cells.items()
                   if cc == ctx and ll == nlen]
            ntok = sum(c["ntok"] for c in sub) / max(1, len(sub))
            na = sum(c["no_answer"] * c["n"] for c in sub) / v["n"]
            fr = sum(c["from_reasoning"] * c["n"] for c in sub) / v["n"]
            nm = sum(c["near_miss"] * c["n"] for c in sub) / v["n"]
            print(f"{ctx:>5} {nlen:>5} {v['n']:>4} {v['exact']:>6.3f} "
                  f"{ci:>15} {v['sim']:>6.3f} {nm:>5.2f} {na:>5.2f} "
                  f"{fr:>5.2f} {ntok:>10.0f}")
        print()
        # position breakdown (start/middle/end) per ctx
        print("by position (exact rate):")
        hdr = "      " + "".join(f"{p:>22}" for p in
                                 ("start", "middle", "end"))
        print(hdr)
        for ctx in (1024, 2048, 4096, 8192):
            line = f"{ctx:>5} "
            for pos in ("start", "middle", "end"):
                v = cells.get((ctx, 40, pos))
                if v:
                    line += f"{v['exact']:>8.2f}(n={v['n']:<2}){'':>8}"
                else:
                    line += f"{'--':>22}"
            print(line)
        print()

    if "lfm25_hybrid" in all_coll and "dense_ref" in all_coll:
        print("===== delta (hybrid - dense), exact rate, collapsed =====")
        print(f"{'ctx':>5} {'nlen':>5} {'lfm':>6} {'dense':>6} {'delta':>7}")
        for ctx in (1024, 2048, 4096, 8192):
            for nlen in (40, 100, 300):
                a = all_coll["lfm25_hybrid"].get((ctx, nlen))
                b = all_coll["dense_ref"].get((ctx, nlen))
                if a and b:
                    print(f"{ctx:>5} {nlen:>5} {a['exact']:>6.3f} "
                          f"{b['exact']:>6.3f} {a['exact']-b['exact']:>+7.3f}")


if __name__ == "__main__":
    main()
