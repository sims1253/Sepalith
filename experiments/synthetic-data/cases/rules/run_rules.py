#!/usr/bin/env python3
"""run_rules.py — run the transform-rule registry: selftests + corpus matrix.

Modes (system python3 from experiments/synthetic-data, never .venv-sft):

  python3 cases/rules/run_rules.py --selftest
      Execute every rule's SELFTEST snippets through REAL tree-sitter parses
      (detector -> rewrite -> splice re-parse -> assembled row -> registry
      gate). The contributor loop: add a rule, run this, zero corpus access.

  python3 cases/rules/run_rules.py --base-samples 20
      Collect N corpus base samples (/mnt/h/sepalith/normalized, highest
      version per package — the compound.py funnel WITHOUT the comment
      requirement, so rules are exercised on generic functions), run every
      registered rule, gate every row through the existing cases validators
      + the registry's own check_rewrite_row, and write
      results/rules_proto/{scenarios.jsonl,stats.json}. Zero LLM calls.

Prevalence of catalog signals: probe_prevalence.py (separate, cheaper scan).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

import scenarios as S                       # noqa: E402
import cases.corpus as C                    # noqa: E402
import cases.validators as V                # noqa: E402
from cases.compound import BaseSample       # noqa: E402
from cases.rules import (REGISTRY, base_sample_id, check_rewrite_row,  # noqa
                         derivation_key, load_rules, make_row,
                         splice_reparse)

OUT_DEFAULT = HERE.parents[1] / "results" / "rules_proto" / "scenarios.jsonl"
STATS_DEFAULT = HERE.parents[1] / "results" / "rules_proto" / "stats.json"


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def run_selftest() -> int:
    rules = load_rules()
    print(f"[selftest] {len(rules)} rules registered: {sorted(rules)}")
    n_case = n_fail = 0
    for rid, r in sorted(rules.items()):
        for case in (r.selftest or []):
            code, opts = case[0], (case[1] if len(case) > 1 else {})
            n_case += 1
            why = opts.get("why", "")
            try:
                b = S.Bundle("selftest", "R/selftest.R", code)
                fns = [n for n in V._walk(b.tree.root_node)
                       if n.type == "function_definition"]
                assert len(fns) == 1, f"{len(fns)} functions in snippet"
                bs = BaseSample(b, fns[0], 0)
                if r.kind == "metadata":
                    ann = r.annotate(bs)
                    for k, v in (opts.get("expect_annotations") or {}).items():
                        assert ann.get(k) == v, \
                            f"{k} = {ann.get(k)!r}, expected {v!r}"
                else:
                    sites = r.detector(bs)
                    want = opts.get("expect_sites")
                    if want is not None:
                        assert len(sites) == want, \
                            f"{len(sites)} sites, expected {want}"
                    if sites:
                        rw = r.rewrite(bs, sites[0])
                        assert rw is not None, "rewrite() returned None"
                        assert splice_reparse(
                            bs, sites[0].sb, sites[0].eb,
                            rw.span_text or "\n".join(rw.lines)), \
                            "spliced function does not re-parse"
                        fn = opts.get("first_new")
                        if fn is not None:
                            assert rw.lines[0] == fn, \
                                f"first line {rw.lines[0]!r} != {fn!r}"
                        row = make_row(bs, r, sites[0], rw)
                        ok, reason = check_rewrite_row(row, r)
                        assert ok, f"registry gate: {reason}"
                    else:
                        assert want == 0, "expected sites but got none"
            except AssertionError as e:
                n_fail += 1
                print(f"  FAIL {rid}: {e}" + (f"  [{why}]" if why else ""))
            except Exception as e:                     # noqa: BLE001
                n_fail += 1
                print(f"  ERROR {rid}: {type(e).__name__}: {e}")
    print(f"[selftest] {n_case - n_fail}/{n_case} cases pass"
          + ("" if n_fail else " — ALL GREEN"))
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# corpus base samples (compound.collect_base_samples without the comment
# requirement — rules fire on generic functions)
# ---------------------------------------------------------------------------

def collect_samples(rng: random.Random, want: int, params: dict):
    min_body = int(params.get("min_body_lines", 6))
    max_body = int(params.get("max_body_lines", 40))
    n_tidy = int(params.get("tidy_packages", 30))
    n_rand = int(params.get("random_packages", 60))
    time_budget = float(params.get("time_budget_s", 420))

    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in S.list_packages() if p not in set(tidy)]
    rng.shuffle(rest)
    pool = tidy[:n_tidy] + rest[:n_rand]
    rng.shuffle(pool)

    funnel = dict(functions_seen=0, size_ok=0, files=0, files_prescreen=0,
                  prescreen_hits={}, packages=set())
    rules = load_rules()
    prescreens = [(rid, [re.compile(p) for p in r.prescreen])
                  for rid, r in sorted(rules.items()) if r.prescreen]
    samples: list[BaseSample] = []
    pkg_counts: dict[str, int] = {}
    per_pkg_cap = int(params.get("per_package_samples", 2))
    t0 = time.time()
    site_id = 0
    for b in C.iter_bundles_highest(pool, rng):
        funnel["files"] += 1
        funnel["packages"].add(b.package)
        if time.time() - t0 > time_budget or len(samples) >= want:
            break
        if pkg_counts.get(b.package, 0) >= per_pkg_cap:
            continue
        # signal-directed pre-selection: only files carrying >= 1 rule
        # signal contribute base samples (the corpus is air-normalized, so
        # lint signals are sparse in a uniform sample — this is what makes
        # per-rule yield measurable at N=20; hit rates land in the funnel)
        hits = [rid for rid, pats in prescreens
                if all(p.search(b.src) for p in pats)]
        for rid in hits:
            funnel["prescreen_hits"][rid] = \
                funnel["prescreen_hits"].get(rid, 0) + 1
        if not hits:
            continue
        funnel["files_prescreen"] += 1
        n_file = 0
        per_file_cap = int(params.get("per_file_samples", 1))
        fns = [n for n in V._walk(b.tree.root_node)
               if n.type == "function_definition"]
        # prefer the function that actually carries the signal (prescreen
        # matched the FILE; the site may sit in a sibling function)
        fns.sort(key=lambda fn: not any(
            p.search(b.src[fn.start_byte:fn.end_byte])
            for _rid, pats in prescreens for p in pats))
        for fn in fns:
            if len(samples) >= want or n_file >= per_file_cap:
                break
            if pkg_counts.get(b.package, 0) >= per_pkg_cap:
                break
            funnel["functions_seen"] += 1
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            _body, _head, _r0, _r1, nb = geom
            if not min_body <= len(nb) <= max_body:
                continue
            funnel["size_ok"] += 1
            samples.append(BaseSample(b, fn, site_id))
            site_id += 1
            n_file += 1
            pkg_counts[b.package] = pkg_counts.get(b.package, 0) + 1
    funnel["packages"] = len(funnel["packages"])
    for i, bs in enumerate(samples):
        bs.site_id = i
    return samples, funnel


# ---------------------------------------------------------------------------
# matrix run
# ---------------------------------------------------------------------------

def run_matrix(samples, params):
    rules = load_rules()
    cap = int(params.get("cap_per_rule", 2))
    dparams = dict(cap_per_rule=cap)            # derivation params on rows
    rows_out: list[dict] = []
    seen_keys: set[str] = set()                 # derivation dedup
    stats = dict(per_rule={}, per_sample=[], per_base_sample={}, failures=[],
                 annotations=[], dups_dropped=0)
    for bs in samples:
        bsid = base_sample_id(bs)
        book = stats["per_base_sample"].setdefault(
            bsid, dict(package=bs.b.package, path=bs.b.rel,
                       body_lines=bs.nbody, rows=0, families=set(), rules=[]))
        smells = {}
        for rid, r in sorted(rules.items()):
            t0 = time.time()
            if r.kind == "metadata":
                try:
                    ann = r.annotate(bs)
                except Exception as e:                  # noqa: BLE001
                    ann = dict(error=f"{type(e).__name__}: {e}")
                ann = dict(ann, base_sample_id=bsid,
                           package=bs.b.package, path=bs.b.rel)
                stats["annotations"].append(ann)
                smells.update({k: v for k, v in ann.items()
                               if isinstance(v, (int, str))})
                continue
            slot = stats["per_rule"].setdefault(
                rid, dict(family=r.family, determinism=r.determinism,
                          status=r.status, rl_ready=r.is_rl_ready,
                          attempted=0, sites_seen=0, rows=0,
                          rejects=dict(), rejects_total=0))
            slot["attempted"] += 1
            try:
                sites = r.detector(bs)
            except Exception as e:                      # noqa: BLE001
                sites = []
                slot["rejects"][f"EXC {type(e).__name__}"] = \
                    slot["rejects"].get(f"EXC {type(e).__name__}", 0) + 1
                stats["failures"].append(
                    dict(rule=rid, base_sample_id=bsid,
                         package=bs.b.package, reason=f"detector EXC: {e}"[:140],
                         seconds=round(time.time() - t0, 3)))
            slot["sites_seen"] += len(sites)
            if not sites:
                slot["rejects"]["no site (signal absent)"] = \
                    slot["rejects"].get("no site (signal absent)", 0) + 1
                slot["rejects_total"] += 1
                continue
            n = 0
            for site in sites:
                if n >= cap:
                    break
                try:
                    rw = r.rewrite(bs, site)
                except Exception as e:                  # noqa: BLE001
                    rw = None
                    stats["failures"].append(
                        dict(rule=rid, base_sample_id=bsid,
                             package=bs.b.package,
                             reason=f"rewrite EXC: {e}"[:140], seconds=0))
                if rw is None:
                    slot["rejects"]["rewrite None"] = \
                        slot["rejects"].get("rewrite None", 0) + 1
                    continue
                if not splice_reparse(bs, site.sb, site.eb,
                                      rw.span_text or "\n".join(rw.lines)):
                    slot["rejects"]["splice re-parse"] = \
                        slot["rejects"].get("splice re-parse", 0) + 1
                    stats["failures"].append(
                        dict(rule=rid, base_sample_id=bsid,
                             package=bs.b.package,
                             reason="spliced function does not re-parse",
                             seconds=0))
                    continue
                dkey = derivation_key(
                    bsid, r, dict(dparams, site_row=site.row,
                                  site_col=site.sb))
                if dkey in seen_keys:
                    stats["dups_dropped"] += 1     # same derivation twice
                    continue
                row = make_row(bs, r, site, rw, params=dparams)
                if smells:
                    row["smells"] = smells
                ok, reason = check_rewrite_row(row, r)
                if not ok:
                    slot["rejects"][reason.split(":")[0][:48]] = \
                        slot["rejects"].get(reason.split(":")[0][:48], 0) + 1
                    stats["failures"].append(
                        dict(rule=rid, base_sample_id=bsid,
                             package=bs.b.package, reason=reason[:140],
                             seconds=0))
                    continue
                seen_keys.add(dkey)
                rows_out.append(row)
                n += 1
                book["rows"] += 1
                book["families"].add(r.family)
                book["rules"].append(f"{rid}@{r.version}")
            slot["rows"] += n
            slot["rejects_total"] += 0 if n else 1
        stats["per_sample"].append(
            dict(site=bs.site_id, base_sample_id=bsid,
                 package=bs.b.package, path=bs.b.rel,
                 body_lines=bs.nbody, families=sorted(book["families"]),
                 n_families=len(book["families"]), n_rows=book["rows"]))
    for book in stats["per_base_sample"].values():
        book["families"] = sorted(book["families"])
    return rows_out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python cases/rules/run_rules.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stats", type=Path, default=STATS_DEFAULT)
    args = ap.parse_args(argv)
    if args.selftest:
        return run_selftest()

    params = dict(seed=args.seed)
    t0 = time.time()
    rng = random.Random(args.seed)
    samples, funnel = collect_samples(rng, args.base_samples, params)
    print(f"[base] {len(samples)} base samples (funnel: {json.dumps(funnel)}) "
          f"in {time.time() - t0:.0f}s")
    if not samples:
        sys.exit("no base samples collected")

    rows, stats = run_matrix(samples, params)
    per_sample = stats["per_sample"]
    fam_counts = [p["n_families"] for p in per_sample]
    ann = [a for a in stats["annotations"]]
    cyclo = sorted(a.get("cyclo", 0) for a in ann)
    print(f"[matrix] {len(rows)} validated rows from {len(samples)} base "
          f"samples; mean families/sample "
          f"{sum(fam_counts) / len(fam_counts):.2f}; "
          f"failures logged: {len(stats['failures'])}")
    for rid, slot in sorted(stats["per_rule"].items()):
        rej = ", ".join(f"{k}={v}" for k, v in sorted(slot["rejects"].items())
                        if v) or "-"
        print(f"  {rid:20s} rows={slot['rows']:3d} "
              f"sites={slot['sites_seen']:3d} over "
              f"{slot['attempted']} samples | rejects: {rej}")
    books = stats["per_base_sample"]
    if books:
        ys = sorted(v["rows"] for v in books.values())
        print(f"[bookkeeping] {len(books)} base samples (stable ids); "
              f"rows/sample min={ys[0]} median={ys[len(ys) // 2]} max={ys[-1]}; "
              f"duplicate derivations dropped: {stats['dups_dropped']}")
    if cyclo:
        import statistics
        print(f"[smells] cyclo median={statistics.median(cyclo)} "
              f"p90={cyclo[int(0.9 * (len(cyclo) - 1))]} max={max(cyclo)}; "
              f"flags={sum(1 for a in ann if a.get('flags'))}/{len(ann)} "
              f"samples carry >= 1 smell flag")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(base_samples=len(samples), funnel=funnel, rows=len(rows),
                  rules=len(load_rules()), per_rule=stats["per_rule"],
                  per_sample=per_sample, failures=stats["failures"][:400],
                  failure_count=len(stats["failures"]),
                  annotations=stats["annotations"], llm_calls=0,
                  seed=args.seed, elapsed_s=round(time.time() - t0, 1))
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(report, indent=1))
    print(f"[out] {args.out} + {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
