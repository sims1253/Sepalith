#!/usr/bin/env python3
"""finish_block_compound.py — the compound finish_block WAVE RUNNER.

Deterministic corpus wave for the finish_block family via the registry
rules in cases/rules/rules_finish_block.py: scan the normalized CRAN
corpus (highest version per package), take ONE base sample per eligible
function, derive the cut-point x packaging matrix (D1, zero LLM calls,
zero quota, no GPU), gate every row through the family gate
(splice-exact + re-parse + render compatibility), and write the family's
own row schema (drop-in for assemble_sft_v5.load_finish_block) extended
with the cases conventions + the registry parent-link contract.

Domain diversity is a first-class stratum: the 20 odd-domain seed
packages from results/ideation_tournament/domain_seeds.json (tunnel
engineering, dairy breeding, audio synthesis, psychometrics, ...) are
scanned FIRST and tagged on their rows (seed_domain) — the register CRAN
under-covers and glm's general knowledge covers.

Usage (system python3 from experiments/synthetic-data, never .venv-sft):
  python3 finish_block_compound.py --selftest
  python3 finish_block_compound.py --wave --base-samples 3000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scenarios as S                             # noqa: E402
import cases.corpus as C                          # noqa: E402
import cases.validators as V                      # noqa: E402
from cases.compound import BaseSample             # noqa: E402
import cases.rules.rules_finish_block as FB       # noqa: E402
from cases.rules import load_rules                # noqa: E402
import rewrite_author_zai as ZA                   # noqa: E402

DATASETS = Path("/mnt/h/sepalith/datasets/cases_v1")
OUT_ROWS = DATASETS / "finish_block_compound.jsonl"
OUT_STATS = DATASETS / "finish_block_compound.stats.json"
LOCAL_OUT = HERE / "results" / "finish_block_compound"
SEEDS = HERE / "results" / "ideation_tournament" / "domain_seeds.json"

MIN_BODY_NB_WAVE = 2      # wave-level floor: 1-statement bodies only feed
                          # the signature cut; not worth a base sample slot
MAX_BODY_NB_WAVE = 40


# ---------------------------------------------------------------------------
# selftest — the FULL positive suite under the family gate
# ---------------------------------------------------------------------------

def _bs_from_code(code: bytes) -> BaseSample:
    b = S.Bundle("selftest", "R/selftest.R", code)
    fns = [n for n in V._walk(b.tree.root_node)
           if n.type == "function_definition"]
    assert len(fns) == 1, f"{len(fns)} functions in snippet"
    return BaseSample(b, fns[0], 0)


def run_family_selftest() -> int:
    load_rules()
    n_case = n_fail = 0
    for code, opts in FB.FAMILY_SELFTEST:
        n_case += 1
        name = opts.get("name", "?")
        try:
            bs = _bs_from_code(code)
            rows, st = FB.derive_all(bs)
            assert not st["gate_failures"], \
                f"gate failures (bugs): {st['gate_failures']}"
            assert st["cuts"] == opts["expect_cuts"], \
                f"cuts {st['cuts']} != {opts['expect_cuts']}"
            # every row re-passes the family gate independently
            for row in rows:
                ok, reason = FB.family_gate(bs, row)
                assert ok, f"family_gate: {reason}"
                # render essentials (assemble_sft_v5.render_finish_block)
                if row["kind"] == "mid_body":
                    assert "{\n" in row["prefix"], "no '{\\n' split point"
                    head = row["prefix"].split("{\n", 1)[-1]
                    assert head.splitlines() or row["kind"] == "signature"
                assert len(row["target"].strip()) >= 30
            stripped = [r for r in rows
                        if r["derivation"]["params"]["docstring"] == "strip"]
            if opts.get("expect_stripped"):
                assert stripped, "expected docstring-stripped twins"
                assert all("#'" not in r["prefix"] for r in stripped), \
                    "stripped twin still carries roxygen"
            else:
                assert not stripped, "strip emitted without a docstring"
            if opts.get("pairs_complete"):
                by_pair: dict[str, set] = {}
                for r in rows:
                    by_pair.setdefault(
                        r["derivation"]["pair_key"], set()).add(
                        r["derivation"]["params"]["docstring"])
                for pk, kinds in by_pair.items():
                    if any(r["derivation"]["rule_id"].startswith("fb_cut_")
                           for r in rows
                           if r["derivation"]["pair_key"] == pk):
                        assert kinds == {"keep", "strip"}, \
                            f"pair {pk} incomplete: {kinds}"
            if opts.get("mid_nested_leads_with_brace"):
                mn = [r for r in rows if r["cut"] == "mid_nested"]
                assert mn, "expected a mid_nested row"
                first = next(l for l in mn[0]["target"].split("\n")
                             if l.strip())
                assert first.lstrip().startswith("}"), \
                    f"mid_nested target must lead with the close: {first!r}"
                # its stripped/keep pair + the keep row's prefix ends INSIDE
                # the nested block (the difficulty)
                keep = next(r for r in mn
                            if r["derivation"]["params"]["docstring"] == "keep")
                assert keep["prefix"].splitlines()[-1].strip() and \
                    "for (" in keep["prefix"], "prefix should end inside the loop"
            # determinism: derive again, byte-identical rows
            rows2, _ = FB.derive_all(bs)
            assert json.dumps(rows, sort_keys=True) == \
                json.dumps(rows2, sort_keys=True), "derivation not deterministic"
            print(f"  ok {name}: cuts={st['cuts']} rows={len(rows)} "
                  f"(strip twins: {len(stripped)})")
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL {name}: {e}")
    print(f"[family-selftest] {n_case - n_fail}/{n_case} cases pass"
          + ("" if n_fail else " — ALL GREEN"))
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# provenance per package (version dir + DESCRIPTION)
# ---------------------------------------------------------------------------

_DESC_CACHE: dict[str, dict] = {}


def package_provenance(pkg: str, ver_dir: Path) -> dict:
    if pkg in _DESC_CACHE:
        return _DESC_CACHE[pkg]
    ver = ver_dir.name
    lic, url = "", ""
    desc = None
    for child in (ver_dir / pkg,):
        d = child / "DESCRIPTION"
        if d.exists():
            desc = d
            break
    if desc is None:                    # case-mismatched dir names
        try:
            for child in ver_dir.iterdir():
                if child.name.lower() == pkg.lower() \
                        and (child / "DESCRIPTION").exists():
                    desc = child / "DESCRIPTION"
                    break
        except OSError:
            pass
    if desc is not None:
        try:
            text = desc.read_text(errors="replace")
            for line in text.splitlines():
                s = line.strip()
                if line.startswith("License:") and not lic:
                    lic = s[len("License:"):].strip()
                elif line.startswith("URL:") and not url:
                    url = s[len("URL:"):].strip()
        except OSError:
            pass
    prov = dict(version=ver, license=lic, upstream=url.split(",")[0].strip(),
                source_url=(f"https://cran.r-project.org/src/contrib/"
                            f"{pkg}_{ver}.tar.gz"))
    _DESC_CACHE[pkg] = prov
    return prov


# ---------------------------------------------------------------------------
# the wave
# ---------------------------------------------------------------------------

def build_pool(rng: random.Random, params: dict) -> tuple[list[str], dict]:
    seeds = {}
    if SEEDS.exists():
        try:
            seeds = json.loads(SEEDS.read_text()).get("index") or {}
        except (ValueError, OSError):
            seeds = {}
    seed_pkgs = [p for ps in seeds.values() for p in ps]
    known = set(S.list_packages())
    seed_pkgs = [p for p in dict.fromkeys(seed_pkgs) if p in known]
    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in known if p not in set(tidy) | set(seed_pkgs)]
    rng.shuffle(rest)
    n_rand = int(params.get("random_packages", 400))
    pool = seed_pkgs + tidy[:int(params.get("tidy_packages", 30))] \
        + rest[:n_rand]
    rng.shuffle(rest)
    pool += rest[n_rand:n_rand + int(params.get("extra_packages", 0))]
    return pool, seeds


def run_wave(args) -> int:
    load_rules()
    rng = random.Random(args.seed)
    params = dict(seed=args.seed, tidy_packages=args.tidy_packages,
                  random_packages=args.random_packages)
    pool, seeds = build_pool(rng, params)
    pkg_domain = {p: d for d, ps in seeds.items() for p in ps}

    versions = C._resolve_pkg_versions(pool)
    funnel = dict(files=0, functions=0, braced=0, named=0, size_ok=0,
                  samples=0, packages=set(),
                  seed_packages_hit=0, no_version_dir=0)
    restraints: Counter = Counter()
    per_rule: Counter = Counter()
    per_cut: Counter = Counter()
    per_base: dict[str, dict] = {}
    rows_out: list[dict] = []
    seen_keys: set[str] = set()
    dups = 0
    t0 = time.time()
    pkg_counts: dict[str, int] = {}
    per_pkg_cap = int(params.get("per_package_samples", 4))
    per_file_cap = 1

    for pkg in pool:
        if time.time() - t0 > args.time_budget or funnel["samples"] >= args.base_samples:
            break
        vd = versions.get(pkg)
        if vd is None:
            funnel["no_version_dir"] += 1
            continue
        rdir = C.ASTFIM.src_root_for(Path(vd), pkg)
        if rdir is None:
            continue
        prov = package_provenance(pkg, Path(vd))
        try:
            files = sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r")))
        except OSError:
            continue
        n_pkg_samples = 0
        for f in files:
            if time.time() - t0 > args.time_budget \
                    or funnel["samples"] >= args.base_samples \
                    or n_pkg_samples >= per_pkg_cap:
                break
            try:
                src = f.read_bytes()
            except OSError:
                continue
            if not src or len(src) > S.MAX_FILE_BYTES:
                continue
            funnel["files"] += 1
            b = S.Bundle(pkg, f"R/{f.name}", src)
            n_file = 0
            for fn in (n for n in V._walk(b.tree.root_node)
                       if n.type == "function_definition"):
                if funnel["samples"] >= args.base_samples \
                        or n_file >= per_file_cap:
                    break
                funnel["functions"] += 1
                geom = C._fn_body(b, fn)
                if geom is None:
                    continue
                funnel["braced"] += 1
                try:
                    bs = BaseSample(b, fn, funnel["samples"])
                except ValueError:
                    continue
                if FB._lhs_name(bs) is None:
                    continue
                funnel["named"] += 1
                if not MIN_BODY_NB_WAVE <= bs.nbody <= MAX_BODY_NB_WAVE:
                    continue
                funnel["size_ok"] += 1
                prov2 = dict(prov, package=pkg, path=b.rel)
                if pkg in pkg_domain:
                    prov2["seed_domain"] = pkg_domain[pkg]
                try:
                    rows, st = FB.derive_all(bs, prov=prov2)
                except Exception as e:                 # noqa: BLE001 — logged
                    restraints[f"EXC {type(e).__name__}"] += 1
                    continue
                for k, v in st["restraints"].items():
                    restraints[k] += v
                if not rows:
                    continue
                kept = []
                for row in rows:
                    if row["derivation_key"] in seen_keys:
                        dups += 1
                        continue
                    seen_keys.add(row["derivation_key"])
                    kept.append(row)
                    per_rule[row["transform"]] += 1
                    per_cut[row["cut"]] += 1
                if not kept:
                    continue
                rows_out.extend(kept)
                funnel["samples"] += 1
                n_pkg_samples += 1
                n_file += 1
                if pkg in pkg_domain:
                    funnel["seed_packages_hit"] += 1
                bsid = kept[0]["derivation"]["base_sample_id"]
                per_base[bsid] = dict(
                    package=pkg, path=b.rel, fn=kept[0]["fn"],
                    body_lines=bs.nbody, rows=len(kept),
                    cuts=st["cuts"], seed_domain=pkg_domain.get(pkg, ""),
                    has_docstring=st["has_docstring"])
        funnel["packages"].add(pkg)

    # ---- write -----------------------------------------------------------
    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        # deterministic single-pass generator: a fresh run REWRITES the file
        # (bit-for-bit regenerable from corpus + rules; no cross-run appends)
        try:
            OUT_ROWS.write_text("")
        except OSError:
            pass
        for row in rows_out:
            ZA._append_line(OUT_ROWS, row)
    seed_rows = sum(1 for r in rows_out if r.get("seed_domain"))
    per_domain = Counter(r["seed_domain"] for r in rows_out
                         if r.get("seed_domain"))
    ys = sorted(v["rows"] for v in per_base.values())
    stats = dict(
        case="finish_block_compound", wave="deterministic-corpus",
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        base_samples=funnel["samples"], rows=len(rows_out),
        rows_per_base=dict(min=ys[0] if ys else 0,
                           median=ys[len(ys) // 2] if ys else 0,
                           max=ys[-1] if ys else 0),
        seed_domain_rows=seed_rows, seed_domain_packages=funnel[
            "seed_packages_hit"],
        per_domain=dict(per_domain), per_rule=dict(per_rule),
        per_cut=dict(per_cut), dups_dropped=dups,
        funnel=dict(funnel, packages=len(funnel["packages"])),
        restraints=dict(restraints),
        llm_calls=0, seed=args.seed,
        elapsed_s=round(time.time() - t0, 1),
        # registry 3.2.5 bookkeeping + the environment stamp convention
        per_base_sample=per_base,
        environment=FB.environment_stamp(),
        validator_manifest=FB.gates_manifest(),
    )
    if not args.dry_run:
        ZA._write_json(OUT_STATS, stats)
        ZA._write_json(LOCAL_OUT / "stats.json", stats)
    print(f"[wave] {len(rows_out)} rows from {funnel['samples']} base "
          f"samples in {time.time()-t0:.0f}s; per_cut={dict(per_cut)}; "
          f"per_rule={dict(per_rule)}")
    print(f"[wave] seed-domain rows: {seed_rows} over "
          f"{funnel['seed_packages_hit']} base samples "
          f"({dict(per_domain)})")
    print(f"[wave] restraints: {dict(restraints)}; dups dropped: {dups}")
    print(f"[wave] out: {OUT_ROWS} + {OUT_STATS}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 finish_block_compound.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wave", action="store_true")
    ap.add_argument("--base-samples", type=int, default=3000)
    ap.add_argument("--tidy-packages", type=int, default=30)
    ap.add_argument("--random-packages", type=int, default=400)
    ap.add_argument("--time-budget", type=float, default=5400)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return run_family_selftest()
    if args.wave:
        return run_wave(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
