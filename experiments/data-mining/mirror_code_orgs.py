#!/usr/bin/env python3
"""Curated permissive-org code mirror for the A2 transfer strata.

Shallow-clones a curated list of permissively-licensed repos per stratum
(the acquisition plan), license-checks each (LICENSE file scan), and
extracts the stratum's source files to a staging tree for tokenization.
Resume: repos with a done marker are skipped.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from pathlib import Path

MIRROR = Path("/mnt/h/sepalith/a2_code_mirror")
PERMISSIVE = ("mit", "apache", "bsd", "isc", "postgresql", "public domain",
              "mozilla", "gpl", "lgpl")   # gpl-family per house rule

ORGS = {
    "python": [
        "numpy/numpy", "scipy/scipy", "pandas-dev/pandas",
        "scikit-learn/scikit-learn", "matplotlib/matplotlib",
        "statsmodels/statsmodels", "pydata/xarray", "dask/dask",
        "sympy/sympy", "networkx/networkx", "astropy/astropy",
        "scikit-image/scikit-image", "pvlib/pvlib-python",
    ],
    "julia": ["JuliaLang/julia", "JuliaArrays/Arrays.jl",
              "JuliaData/DataFrames.jl", "JuliaStats/StatsBase.jl",
              "JuliaPlots/Plots.jl"],
    "js_ts": ["microsoft/TypeScript", "d3/d3", "plotly/plotly.js",
              "parcel-bundler/parcel"],
    "c_cpp": ["sqlite/sqlite", "madler/zlib", "glennrp/libpng",
              "curl/curl"],
}
EXTS = {"python": [".py"], "julia": [".jl"], "js_ts": [".ts", ".js", ".tsx"],
        "c_cpp": [".c", ".h", ".cc", ".cpp", ".hpp"]}


def clone(slug: str, dest: Path) -> bool:
    try:
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{slug}.git", str(dest)],
                       capture_output=True, timeout=600)
        return (dest / ".git").exists() or dest.exists()
    except subprocess.TimeoutExpired:
        return False


def license_ok(dest: Path) -> str | None:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING",
                 "COPYING.txt", "LICENSE.MIT"):
        p = dest / name
        if p.exists():
            try:
                head = p.read_text(errors="replace")[:800].lower()
            except OSError:
                continue
            for lic in PERMISSIVE:
                if lic in head:
                    return f"{name}:{lic}"
    return None


def extract(repo_dir: Path, stratum: str, out: Path):
    n = 0
    for f in repo_dir.rglob("*"):
        if f.is_file() and f.suffix in EXTS[stratum] \
                and f.stat().st_size < 400_000 \
                and "test" not in str(f.relative_to(repo_dir)).lower()[:60]:
            rel = f.relative_to(repo_dir)
            (out / rel.parent).mkdir(parents=True, exist_ok=True)
            try:
                (out / rel).write_bytes(f.read_bytes())
                n += 1
            except OSError:
                pass
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", default="python,julia,js_ts,c_cpp")
    args = ap.parse_args()
    ledger = []
    ledger_p = MIRROR / "license_ledger.jsonl"
    for stratum in args.strata.split(","):
        for slug in ORGS.get(stratum, []):
            dest = MIRROR / "repos" / slug.replace("/", "__")
            done = MIRROR / "staging" / stratum / (slug.replace("/", "__") + ".done")
            if done.exists():
                continue
            print(f"[{stratum}] {slug} ...", flush=True)
            if not clone(slug, dest):
                ledger.append(dict(slug=slug, stratum=stratum, ok=False,
                                   why="clone_failed"))
                continue
            lic = license_ok(dest)
            if not lic:
                ledger.append(dict(slug=slug, stratum=stratum, ok=False,
                                   why="no_permissive_license"))
                subprocess.run(["rm", "-rf", str(dest)])
                continue
            n = extract(dest, stratum, MIRROR / "staging" / stratum)
            done.write_text(f"{n} files")
            ledger.append(dict(slug=slug, stratum=stratum, ok=True,
                               license=lic, files=n))
            with open(ledger_p, "a") as fh:
                fh.write(json.dumps(ledger[-1]) + "\n")
            print(f"  ok {lic} {n} files", flush=True)
    ok = sum(1 for r in ledger if r.get("ok"))
    print(f"DONE ok={ok}/{len(ledger)}", flush=True)


if __name__ == "__main__":
    main()
