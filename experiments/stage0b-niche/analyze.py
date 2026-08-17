#!/usr/bin/env python3
"""Aggregate niche-gate results: per-language metrics + bootstrap CI on the R-Python gap."""
import json, random, re, sys, difflib

def load(path):
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if isinstance(r, dict) and "lang" in r and "exact" in r:
            out.append(r)
    return out

def copy_baseline(path):
    """Baseline (0) from DESIGN.md §7: predict the region unchanged (copy-from-context)."""
    exs = [json.loads(l) for l in open(path)]
    out = {}
    for lang in ("python", "r"):
        sub = [e for e in exs if e["lang"] == lang]
        ex = sum(int(norm(e["region_old"]) == norm(e["region_new"])) for e in sub)
        fl = sum(int(e["region_old"] and e["region_new"] and
                     norm(e["region_old"])[0] == norm(e["region_new"])[0]) for e in sub)
        f1s = []
        for e in sub:
            a, b = norm(e["region_old"]), norm(e["region_new"])
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            m = sum(x.size for x in sm.get_matching_blocks())
            f1s.append(2*m/(len(a)+len(b)) if (a or b) else 1.0)
        out[lang] = dict(n=len(sub), exact=ex/len(sub), first_line=fl/len(sub),
                         line_f1=sum(f1s)/len(f1s))
    return out

def norm(lines):
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1]:
        lines.pop()
    return lines

def rate(rs, key):
    return sum(r[key] for r in rs) / len(rs) if rs else 0.0

def boot_gap(a, b, key, n=10000, seed=1):
    """Bootstrap CI for mean(a[key]) - mean(b[key])."""
    rnd = random.Random(seed)
    va = [r[key] for r in a]; vb = [r[key] for r in b]
    if not va or not vb:
        return None
    obs = sum(va)/len(va) - sum(vb)/len(vb)
    ds = []
    for _ in range(n):
        sa = [rnd.choice(va) for _ in va]; sb = [rnd.choice(vb) for _ in vb]
        ds.append(sum(sa)/len(sa) - sum(sb)/len(sb))
    ds.sort()
    return obs, ds[int(0.025*n)], ds[int(0.975*n)]

for path in sys.argv[1:]:
    rs = [r for r in load(path) if "lang" in r]
    if not rs:
        continue
    model = path.split("results_")[1].split(".")[0]
    base = copy_baseline(re.sub(r"results_.*\.jsonl", "examples.jsonl", path))
    print(f"\n=== {model} (n={len(rs)}) ===  [copy-from-context baseline: "
          f"py exact={base['python']['exact']:.3f} f1={base['python']['line_f1']:.3f} | "
          f"r exact={base['r']['exact']:.3f} f1={base['r']['line_f1']:.3f}]")
    for lang in ("python", "r"):
        sub = [r for r in rs if r["lang"] == lang]
        print(f"  {lang:7s} n={len(sub):3d}  exact={rate(sub,'exact'):.3f}  "
              f"first_line={rate(sub,'first_line'):.3f}  line_f1={rate(sub,'line_f1'):.3f}  "
              f"empty/format_fail={rate(sub,'empty'):.3f}  p50_lat={sorted(x['latency_s'] for x in sub)[len(sub)//2]:.1f}s")
    py = [r for r in rs if r["lang"] == "python"]
    rr = [r for r in rs if r["lang"] == "r"]
    for key in ("exact", "first_line", "line_f1"):
        g = boot_gap(py, rr, key)
        if g:
            print(f"  gap python-r [{key}]: {g[0]:+.3f}  95%CI [{g[1]:+.3f}, {g[2]:+.3f}]")
