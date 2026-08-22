#!/usr/bin/env python3
"""finish_block_author.py — DOMAIN-DIVERSE authored base samples for the
compound finish_block family (the glm-corner lever).

The corpus side of finish_block is CRAN-shaped; the landscape benchmark
showed glm-5.3 winning the intent layer on general body-filling — the
register CRAN under-covers (odd domains) is exactly where authored base
samples help. This driver:

  1. draws a cell from the 20-domain seed index
     (results/ideation_tournament/domain_seeds.json — tunnel engineering,
     dairy breeding, audio synthesis, psychometrics, ...) with a concrete
     SUBTOPIC per draw (deterministic pool, authored below);
  2. asks a FREE backend (default xpreview-free zen; --backend oxalpha-nous
     as the third-provider lane) for ONE realistic domain function under a
     property checklist — docstring presence drawn 50/50 (the user axis:
     real users type bare signatures; every authored function still derives
     BOTH variants via fb_docstring_strip);
  3. gates the function (tree-sitter + Rscript parse, one named top-level
     function, >=2 top-level statements, checklist truth);
  4. derives ALL cut-point + packaging variants through the SAME registry
     rules (cases/rules/rules_finish_block.derive_all) so authored rows are
     gate-identical to corpus rows.

Outputs (cases conventions, resume-safe, <=N in flight, 429 breaker):
  /mnt/h/sepalith/datasets/cases_v1/finish_block_authored.jsonl        rows
  /mnt/h/sepalith/datasets/cases_v1/finish_block_authored_bases.jsonl  samples
  (+ .done.jsonl / .stats.json sidecars). Bounded n=400 FIRST — this file
  is the quality gate before any volume decision.

Usage (system python3 from experiments/synthetic-data):
  python3 finish_block_author.py author --n 400 --workers 2
  python3 finish_block_author.py spotcheck --n 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scenarios as S                             # noqa: E402
import cases.validators as V                      # noqa: E402
import cases.rules.rules_finish_block as FB       # noqa: E402
from cases.rules import load_rules                # noqa: E402
from cases.backends import BackendError           # noqa: E402
import rewrite_author_spark as RAS                # noqa: E402
import rewrite_author_zai as ZA                   # noqa: E402

DATASETS = Path("/mnt/h/sepalith/datasets/cases_v1")
OUT_ROWS = DATASETS / "finish_block_authored.jsonl"
OUT_BASES = DATASETS / "finish_block_authored_bases.jsonl"


def out_paths(backend_name: str) -> tuple[Path, Path]:
    """Per-backend output files (the rewrite-wave convention): the default
    lane owns the canonical names; every other backend gets its own slug so
    parallel lanes never interleave and each stays purgeable/resumable."""
    if backend_name == "xpreview-free":
        return OUT_ROWS, OUT_BASES
    slug = backend_name.replace(":", "-").replace("/", "-")
    return (DATASETS / f"finish_block_authored_{slug}.jsonl",
            DATASETS / f"finish_block_authored_{slug}_bases.jsonl")
SEEDS = HERE / "results" / "ideation_tournament" / "domain_seeds.json"

STYLES = ("tidyverse", "base R", "mixed tidyverse and base R")
LENGTHS = ("S", "M", "L", "L")      # bias to M/L: cut diversity needs body
ROXY = ("full", "none")             # the docstring axis, ~50/50
COMMENTS = ("none", "sparse", "rich")
CONSTRUCTS = {
    "has_loop": ("a for or while loop", ("for", "while")),
    "has_trycatch": ("a tryCatch guard around the risky step",
                     ("tryCatch",)),
    "has_stats_calls": ("stats calls such as mean/sd/median/quantile",
                        ("mean(", "sd(", "median(", "quantile(")),
    "has_pipe_chain": ("a pipe chain", ("%>%", "|>")),
}

# concrete subtopics per seed-index domain (the deterministic topic pool;
# authored here — the seed packages anchor the style, the subtopic forces
# domain-SPECIFIC logic instead of generic data munging)
SUBTOPICS = {
    "web/api": ("paginated GET loop with retry and backoff; query builder "
                "with URL encoding; response flattening to a data.frame; "
                "rate-limit header parsing"),
    "finance": ("loan amortization schedule builder; IRR via bisection; "
                "rolling volatility window; coupon date generation"),
    "weather/climate": ("growing-degree-day accumulation; frost return "
                        "periods; PET from temperature; rainfall "
                        "intensity-duration curves"),
    "gis/spatial": ("great-circle bearing and distance; bbox prefilter "
                    "join; raster row extraction at points; grid snapping "
                    "of survey coordinates"),
    "chemometrics/spectroscopy": ("rubberband baseline correction; "
                                  "Savitzky-Golay smoothing; peak picking "
                                  "on spectra; TMS normalization"),
    "pharma/clinical": ("3+3 dose escalation logic; ADaM PARAMCD recode; "
                        "last-observation-carry-forward; treatment-emergent "
                        "flagging"),
    "sports/analytics": ("running Elo ratings; rolling xG means; shot "
                         "distance buckets; fixture congestion index"),
    "genomics": ("k-mer counting; GC-content sliding window; read-length "
                 "histogram; allele-fraction filtering"),
    "audio/synthesis/music": ("ADSR envelope table; MIDI note-to-frequency "
                              "table with detune; wavetable oscillator with "
                              "interpolation; beat quantizer to a 1/16 grid"),
    "ecology": ("detection-history collapse for occupancy; rarefaction "
                "curve; transect density estimate; dispersal kernel sampler"),
    "psychometrics": ("Cronbach alpha per scale; item-total correlations; "
                      "2PL initial difficulties; DIF flagging"),
    "geotechnics/tunnel": ("tunnel-wall convergence from survey rings; RMR "
                           "rating aggregation; shotcrete thickness QA "
                           "pass/fail; face advance-rate smoothing"),
    "epidemiology": ("R0 from a serial interval; 7-day rolling incidence; "
                     "household attack rates; weekly case-fatality"),
    "ocean/marine": ("tide harmonic residual; CTD profile binning by depth; "
                     "SST anomaly vs climatology; bent-cable trend setup"),
    "econometrics/forecasting": ("seasonal-naive residual check; outlier "
                                 "flagging; AR-order grid by AICc; rolling-"
                                 "origin MAPE"),
    "agriculture": ("dairy lactation curve from survey records; GxE "
                    "stability index; silage dry-matter adjustment; breeding "
                    "value prep"),
    "energy/grid": ("load duration curve; seasonal decomposition of EIA "
                    "series; peak-shaving dispatch; sensor register table "
                    "with CRC sanity checks"),
    "transport/traffic": ("EV charging session profiling; GTFS headway "
                          "calculator; link congestion percentiles; stop "
                          "isochrone buckets"),
    "insurance/actuarial": ("loss development triangle; territory risk "
                            "relativities; pure-premium smoothing; prior "
                            "assembly for a spawner model"),
    "imaging/microscopy": ("ROI intensity quantiles; z-stack max "
                           "projection; PSF FWHM estimate from beads; "
                           "threshold segmentation counts"),
}

PROMPT = """You are writing ONE realistic R function for a small production package in an UNUSUAL domain — the kind of focused utility a domain scientist would actually commit. Follow this checklist EXACTLY:

- domain: {domain} — pick ONE of: {subtopic}
- style: {style}
- body length: {nlines} non-blank lines (aim within ±2)
- roxygen block above the function: {roxy}
- interior plain comments (lines starting with #, not #'): {ncom}
- the body must contain {constructs}
- allowed dependencies: {context}

Rules: the function must be a self-contained top-level definition `name <- function(...) {{ ... }}`; realistic argument names and defaults; use domain-real constants and units (Hz, mm, hectares, bps, degrees, ...); NO placeholders, no TODO, no sessionInfo, no printing; the body must have at least 2 top-level statements and end by returning a value; keep every line under 100 characters.{extra}

Return ONLY a JSON object: {{"function": "<the complete roxygen block (if requested) plus the function, exactly as it would appear in the .R file>", "domain_echo": "{domain}", "subtopic_used": "<the subtopic you picked>"}}"""

PROMPT = PROMPT.replace(
    "- domain: {domain} — pick ONE of: {subtopic}",
    "- domain: {domain} — pick ONE of: {subtopic} (in the spirit of the "
    "CRAN packages: {seeds})")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _slug(domain: str) -> str:
    return domain.replace("/", "-").replace(" ", "-")


def load_domains() -> dict[str, list[str]]:
    idx = json.loads(SEEDS.read_text()).get("index") or {}
    return {d: ps for d, ps in idx.items() if d in SUBTOPICS}


def draw_cell(rng: random.Random, domains: dict[str, list[str]]) -> dict:
    cons = rng.sample(sorted(CONSTRUCTS), rng.choice((0, 1, 1, 2)))
    domain = rng.choice(sorted(domains))
    return dict(domain=domain, seeds=domains[domain],
                style=rng.choice(STYLES), length=rng.choice(LENGTHS),
                roxygen=rng.choice(ROXY), constructs=cons,
                comments=rng.choice(COMMENTS))


def build_prompt(cell: dict, feedback: str | None = None) -> str:
    nlines = {"S": "4-8", "M": "9-20", "L": "21-40"}[cell["length"]]
    roxy = {"full": "a FULL roxygen block (@title, @description, every "
                    "@param, @return)",
            "none": "NO roxygen block at all — the bare function only"}[
        cell["roxygen"]]
    ncom = {"none": "zero", "sparse": "1 to 2 short ones",
            "rich": "3 or more"}[cell["comments"]]
    cons = "; AND ".join(CONSTRUCTS[c][0] for c in cell["constructs"]) \
        if cell["constructs"] else "no special construct requirements"
    context = ("base R only (stats/utils recommended)" if cell["style"] ==
               "base R" else "CRAN dependencies you name and use correctly")
    extra = ""
    if feedback:
        extra = (f" NOTE: a previous attempt was REJECTED because: "
                 f"{feedback} Follow the checklist exactly.")
    subtopics = SUBTOPICS[cell["domain"]]
    return PROMPT.format(domain=cell["domain"], subtopic=subtopics,
                         seeds=", ".join(cell["seeds"][:4]),
                         style=cell["style"], nlines=nlines, roxy=roxy,
                         ncom=ncom, constructs=cons, context=context,
                         extra=extra)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gate_sample(cell: dict, fn_text: str) -> tuple[dict, str]:
    """(gates, fail-summary) — the checklist must be TRUTHFULLY met."""
    g: dict = {}
    lines = [l.rstrip() for l in fn_text.replace("\r\n", "\n").split("\n")]
    tb = S.parser.parse(fn_text.encode())
    g["parses"] = not tb.root_node.has_error
    if not g["parses"]:
        return g, "parses"
    fns = [n for n in S.traverse(tb.root_node)
           if n.type == "function_definition"]
    top = [c for c in tb.root_node.children
           if c.type == "binary_operator" and any(
               k.type == "function_definition" for k in c.children)]
    g["one_function"] = len(fns) == 1 and len(top) == 1
    if not g["one_function"]:
        return g, "one_function"
    body = next(c for c in fns[0].children if c.type == "braced_expression")
    stmts = [c for c in body.children if c.is_named]
    g["statements"] = len(stmts) >= 2
    body_lines = [l for l in lines if l.strip()
                  and not l.strip().startswith("#")]
    lo, hi = {"S": (3, 9), "M": (8, 22), "L": (20, 42)}[cell["length"]]
    g["length"] = lo <= len(body_lines) <= hi
    g["roxygen"] = (cell["roxygen"] == "full"
                    and "#' @param" in fn_text and "#' @return" in fn_text) \
        or (cell["roxygen"] == "none" and "#'" not in fn_text)
    g["no_placeholder"] = not re.search(
        r"TODO|FIXME|sessionInfo|placeholder", fn_text)
    g["line_len"] = max((len(l) for l in lines), default=0) <= 120
    g["return_shaped"] = bool(stmts) and FB._return_shaped(
        stmts[-1], fn_text.encode())
    # Rscript parse (belt and braces vs tree-sitter)
    fh = tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, dir="/tmp")
    fh.write(fn_text)
    fh.close()
    try:
        r = subprocess.run(["Rscript", "-e",
                            f"invisible(parse('{fh.name}'))"],
                           capture_output=True, timeout=20)
        g["rscript_parse"] = r.returncode == 0
    except subprocess.TimeoutExpired:
        g["rscript_parse"] = False
    finally:
        Path(fh.name).unlink(missing_ok=True)
    fails = [k for k in ("parses", "one_function", "rscript_parse",
                         "statements", "length", "roxygen", "no_placeholder",
                         "line_len", "return_shaped") if not g.get(k, True)]
    return g, ",".join(fails)


def derive_rows(fn_text: str, cell: dict, prompt: str,
                backend_name: str, model: str):
    """Synthetic Bundle -> BaseSample -> the SAME deterministic matrix.
    The base id is the registry's _author_base_id (hash of the function
    node) so bases and rows join on one canonical id."""
    b = S.Bundle(f"author:{_slug(cell['domain'])}",
                 "author/anonymous.R", (fn_text + "\n").encode())
    fn = next((n for n in V._walk(b.tree.root_node)
               if n.type == "function_definition"), None)
    if fn is None:
        return [], dict(error="no function node")
    bs = FB.BaseSample(b, fn, 0)
    base_id = FB._author_base_id(bs)
    b.rel = f"author/{base_id}.R"
    prov = dict(package=b.package, path=b.rel, seed_domain=cell["domain"],
                version="authored-1", license="synthetic-authored",
                source_url="", upstream="")
    rows, st = FB.derive_all(bs, prov=prov, origin="author")
    out = []
    for r in rows:
        r = dict(r)
        r["case"] = "finish_block_author"
        r["backend"] = backend_name
        r["model"] = model
        r["full_prompt"] = prompt
        r["generated_at"] = _now()
        r["content_hash"] = hashlib.sha1(
            (f"{r['transform']}\x00{r['derivation_key']}").encode()
        ).hexdigest()
        out.append(r)
    st["base_id"] = base_id
    return out, st


# ---------------------------------------------------------------------------
# author loop
# ---------------------------------------------------------------------------

def cmd_author(args) -> int:
    load_rules()
    backend = RAS.resolve_backend(args.backend)
    domains = load_domains()
    if not domains:
        sys.exit(f"no seed domains at {SEEDS}")
    rng = random.Random(args.seed)
    rows_path, bases_path = out_paths(backend.name)
    done_path = Path(str(rows_path) + ".done.jsonl")
    done = ZA.load_done(done_path)
    hashes = ZA.load_hashes(OUT_ROWS)
    breaker = ZA.Breaker()
    stats = dict(started=_now(), backend=backend.name, model=backend.model,
                 workers=args.workers, target=args.n, attempted=0,
                 accepted=0, rows=0, backend_error=0, retried=0, dups=0,
                 gate_rejects=Counter(), domains=Counter(), per_rule=Counter(),
                 per_cut=Counter(), quality_gate=args.n <= 400)
    lock = threading.Lock()
    stop_hard = threading.Event()
    t0 = time.time()
    last_log = 0.0

    def process(i: int) -> dict:
        cell = draw_cell(rng, domains)
        with lock:
            stats["attempted"] += 1
        feedback = None
        first_prompt = ""
        for attempt in (1, 2):
            if not breaker.wait_turn() or stop_hard.is_set():
                return dict(kind="aborted", key=cell["domain"])
            prompt = build_prompt(cell, feedback)
            if attempt == 1:
                first_prompt = prompt
            try:
                raw = backend.complete(prompt)
                breaker.report(ok=True, rate_error=False)
            except BackendError as e:
                breaker.report(ok=False, rate_error=(e.kind == "rate"))
                with lock:
                    stats["backend_error"] += 1
                    # free-lane flakiness is expected (measured: transient
                    # provider 5xx at ~1/2 calls, retried through by the
                    # backend); the 429 Breaker still guards sustained storms
                    if stats["backend_error"] >= 60:
                        stop_hard.set()
                return dict(kind="backend_error", key=cell["domain"])
            obj = ZA.extract_json_object(ZA.strip_fences(raw))
            fn_text = obj.get("function") if isinstance(obj, dict) else None
            if not isinstance(fn_text, str) or "function" not in fn_text:
                feedback = ("the response was not the JSON object with the "
                            "function")
                continue
            if str(obj.get("domain_echo", "")).strip().lower() \
                    != cell["domain"].strip().lower():
                feedback = f"domain_echo must be {cell['domain']!r}"
                continue
            fn_text = "\n".join(ZA._norm_trim(fn_text))
            g, summary = gate_sample(cell, fn_text)
            if not summary:
                rows, dstats = derive_rows(fn_text, cell, first_prompt,
                                           backend.name, backend.model)
                base_id = dstats.get("base_id", "")
                return dict(kind="accepted", key=f"{cell['domain']}:{base_id}",
                            cell=cell, base_id=base_id, fn_text=fn_text,
                            rows=rows, gates=g, dstats=dstats, attempts=attempt)
            feedback = f"checklist mismatch: {summary}"
            if attempt == 2:
                with lock:
                    for k in summary.split(","):
                        stats["gate_rejects"][k] += 1
                return dict(kind="rejected", key=cell["domain"],
                            reason=summary)
            with lock:
                stats["retried"] += 1
        return dict(kind="rejected", key=cell["domain"], reason="unparseable")

    ex = ThreadPoolExecutor(max_workers=args.workers)
    outstanding: set[Future] = set()
    it = iter(range(args.n))
    n_done = 0
    try:
        while True:
            if time.time() - t0 > args.time_budget or breaker.stop.is_set() \
                    or stop_hard.is_set():
                break
            while len(outstanding) < args.workers * 2:
                try:
                    i = next(it)
                except StopIteration:
                    break
                outstanding.add(ex.submit(process, i))
            if not outstanding:
                break
            done_set, _ = wait(outstanding, timeout=30,
                               return_when=FIRST_EXCEPTION)
            for fut in done_set:
                outstanding.discard(fut)
                try:
                    res = fut.result()
                except Exception as e:                   # noqa: BLE001
                    print(f"  [worker-exception] {e!r}", flush=True)
                    continue
                n_done += 1
                key = res["key"]
                if key in done:
                    continue
                if res["kind"] == "accepted":
                    cell = res["cell"]
                    base_rec = dict(
                        id=res["base_id"],
                        origin=dict(kind="author",
                                    package=f"author:{_slug(cell['domain'])}",
                                    path=f"author/{res['base_id']}.R"),
                        domain=cell["domain"],
                        properties=dict(requested=cell, gates=res["gates"],
                                        subtopic_axes=len(SUBTOPICS)),
                        code=dict(full_text=res["fn_text"]),
                        derivations=[dict(transform=r.get("transform"),
                                          cut=r.get("cut"),
                                          family=r.get("family"))
                                     for r in res["rows"]],
                        case="finish_block_author", backend=backend.name,
                        model=backend.model, generated_at=_now())
                    ZA._append_line(bases_path, base_rec)
                    new_rows = []
                    for row in res["rows"]:
                        if row["content_hash"] in hashes:
                            with lock:
                                stats["dups"] += 1
                            continue
                        hashes.add(row["content_hash"])
                        ZA._append_line(rows_path, row)
                        new_rows.append(row)
                    with lock:
                        stats["accepted"] += 1
                        stats["rows"] += len(new_rows)
                        stats["domains"][cell["domain"]] += 1
                        for r in new_rows:
                            stats["per_rule"][r.get("transform", "?")] += 1
                            stats["per_cut"][r.get("cut", "?")] += 1
                    rec = dict(key=key, ok=True, rows=len(new_rows),
                               domain=cell["domain"],
                               cuts=res["dstats"].get("cuts"),
                               attempts=res.get("attempts"), ts=_now())
                elif res["kind"] == "rejected":
                    rec = dict(key=key, ok=False, reason=res["reason"][:160],
                               ts=_now())
                else:
                    rec = None
                if rec is not None:
                    ZA._append_line(done_path, rec)
                    done[key] = rec
            if time.time() - last_log > 60:
                last_log = time.time()
                bs = backend.stats_summary()
                print(f"  [progress] samples={n_done}/{args.n} "
                      f"accepted={stats['accepted']} rows={stats['rows']} "
                      f"domains={len(stats['domains'])} "
                      f"rejects={dict(stats['gate_rejects'])} "
                      f"backend_ok={bs['ok']} 429={bs['err_429']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                ZA._write_json(Path(str(rows_path) + ".stats.json"),
                               dict(stats, ts=_now(),
                                    elapsed_s=round(time.time() - t0, 1),
                                    backend_stats=backend.stats_summary(),
                                    environment=FB.environment_stamp(),
                                    validator_manifest=FB.gates_manifest()))
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        ZA._write_json(Path(str(rows_path) + ".stats.json"),
                       dict(stats, ts=_now(), final=True,
                            elapsed_s=round(time.time() - t0, 1),
                            backend_stats=backend.stats_summary(),
                            environment=FB.environment_stamp(),
                            validator_manifest=FB.gates_manifest()))
    print(f"[author] FINISHED samples={n_done} accepted={stats['accepted']} "
          f"rows={stats['rows']} domains={len(stats['domains'])} "
          f"in {time.time()-t0:.0f}s", flush=True)
    return 0


def cmd_spotcheck(args) -> int:
    if not OUT_BASES.exists():
        sys.exit(f"no authored bases at {OUT_BASES} yet")
    bases = [json.loads(l) for l in OUT_BASES.read_text().splitlines() if l]
    rows = [json.loads(l) for l in OUT_ROWS.read_text().splitlines() if l] \
        if OUT_ROWS.exists() else []
    rng = random.Random(args.seed)
    for base in rng.sample(bases, min(args.n, len(bases))):
        print("=" * 70)
        print(f"[{base['id']}] domain={base['domain']} "
              f"backend={base['backend']}")
        print("gates:", json.dumps(base["properties"]["gates"]))
        print("-" * 70)
        print(base["code"]["full_text"])
        mine = [r for r in rows
                if r["derivation"]["base_sample_id"] == base["id"]]
        print("-" * 70)
        print(f"derived rows: {len(mine)} "
              f"cuts={sorted({r['cut'] for r in mine})}")
        if mine:
            r = max(mine, key=lambda r: len(r["target"]))
            print(f"hardest ({r['cut']}/{r['kind']}) target "
                  f"({len(r['target'].splitlines())} lines):")
            print(r["target"])
    print(f"[spotcheck] {min(args.n, len(bases))} of {len(bases)} bases, "
          f"{len(rows)} rows total")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 finish_block_author.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("author")
    p.add_argument("--backend", default="xpreview-free")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=9173)
    p.add_argument("--time-budget", type=float, default=14400)
    p.set_defaults(fn=cmd_author)
    p = sub.add_parser("spotcheck")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=5)
    p.set_defaults(fn=cmd_spotcheck)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
