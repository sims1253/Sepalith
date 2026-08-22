#!/usr/bin/env python3
"""compound_author_spark.py — compounding BASE SAMPLES authored by
muse-spark (docs/research/compounding-samples-design.md section 2.1/b).

The corpus under-covers property cells; this driver AUTHORS new R base
samples from a modular property-grid prompt (domain x style x length x
roxygen x constructs x comment_density x context), gates the authored
function HARD against the requested properties, then feeds it through the
PROVEN deterministic derivation matrix (cases.compound.run_matrix — the
prototype's ~4.7 families / ~10 rows per sample), so one LLM call yields
a multi-family scenario bundle:

  authored fn --gates--> BaseSample(synthetic Bundle) --run_matrix-->
  compound rows (deterministic transforms, their own validators)

Outputs (cases conventions; rows carry case/backend/model + author
origin + content-hash base_sample_id so authored-derived rows are
attributable and purgeable):
  /mnt/h/sepalith/datasets/cases_v1/compound_spark.jsonl      (rows)
  /mnt/h/sepalith/datasets/cases_v1/base_samples_spark.jsonl  (samples)
  (+ .done.jsonl / .stats.json sidecars). Resume-safe; <=N in flight.

Usage (system python3 from experiments/synthetic-data):
  python3 compound_author_spark.py author [--n 500] [--workers 3]
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
sys.path.insert(0, str(HERE))

import scenarios as S                                   # noqa: E402
import cases.compound as CP                             # noqa: E402
import rewrite_author_spark as RAS                      # noqa: E402
import rewrite_author_zai as ZA                         # noqa: E402
from cases.backends import BackendError                 # noqa: E402

OUT_DIR = HERE / "results" / "compound_author_spark"
DATASETS = Path("/mnt/h/sepalith/datasets/cases_v1")
OUT_ROWS = DATASETS / "compound_spark.jsonl"
OUT_BASES = DATASETS / "base_samples_spark.jsonl"

DOMAINS = ("pharma/clinical (ADaM-style derived datasets)", "bioinformatics",
           "finance/risk", "geo/spatial", "text/NLP", "data import/export",
           "file/string utilities", "statistical methods")
STYLES = ("tidyverse", "base R", "mixed tidyverse and base R")
LENGTHS = ("S", "M", "L")                    # <=8 / 9-20 / 21-40 body lines
ROXY = ("none", "title-only", "full")
CONSTRUCTS = {
    "has_loop": ("a for or while loop", ("for", "while")),
    "has_trycatch": ("a tryCatch (or try) guard around the risky step",
                     ("tryCatch", "try(")),
    "has_pipe_chain": ("a pipe chain", ("%>%", "|>")),
    "has_ggplot": ("a ggplot build", ("ggplot(", "ggplot2::")),
    "has_dplyr_verbs": ("dplyr verbs (filter/mutate/summarise/group_by)",
                        ("filter(", "mutate(", "summarise(", "summarize(",
                         "group_by(")),
    "has_stats_calls": ("stats calls such as mean/sd/median/lm/t.test",
                        ("mean(", "sd(", "median(", "lm(", "t.test(")),
}
COMMENTS = ("none", "sparse", "rich")
CONTEXTS = ("base packages only", "CRAN dependencies you name in @importFrom/"
             "library calls")

PROMPT = """You are writing ONE realistic R function for a production package — the kind of code a senior R developer would commit. Follow this checklist EXACTLY:

- domain: {domain}
- style: {style}
- body length: {nlines} non-blank lines (aim within ±2)
- roxygen block above the function: {roxygen}
- the body must contain {constructs}
- interior plain comments (lines starting with #, not #'): {ncomments}
- allowed dependencies: {context}

Rules: the function must be self-contained at the top level as `name <- function(...) {{ ... }}`; use realistic argument names and defaults; no placeholders, no TODO, no printing of sessionInfo; {extra}

Return ONLY a JSON object: {{"function": "<the complete roxygen block (if requested) plus the function, exactly as it would appear in the .R file>"}}"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def draw_cell(rng: random.Random) -> dict:
    cons = rng.sample(sorted(CONSTRUCTS), rng.choice((1, 1, 2)))
    return dict(domain=rng.choice(DOMAINS), style=rng.choice(STYLES),
                length=rng.choice(LENGTHS), roxygen=rng.choice(ROXY),
                constructs=cons, comments=rng.choice(COMMENTS),
                context=rng.choice(CONTEXTS))


def cell_id(cell: dict) -> str:
    return hashlib.sha1(json.dumps(cell, sort_keys=True)
                        .encode()).hexdigest()[:12]


def build_prompt(cell: dict, feedback: str | None = None) -> str:
    nlines = {"S": "4-8", "M": "9-20", "L": "21-40"}[cell["length"]]
    roxy = {"none": "no roxygen block at all",
            "title-only": "a one-line title roxygen block (#' @title ...)",
            "full": "a full roxygen block (@title, @description, every "
                    "@param, @return)"}[cell["roxygen"]]
    ncom = {"none": "zero", "sparse": "1 to 2 short ones",
            "rich": "3 or more"}[cell["comments"]]
    cons = "; AND ".join(CONSTRUCTS[c][0] for c in cell["constructs"])
    if cell["style"] == "base R" and "has_pipe_chain" in cell["constructs"]:
        cons += " (use the native |> pipe for the base-R style)"
    extra = "keep every line under 100 characters."
    if feedback:
        extra += (f" NOTE: a previous attempt was REJECTED because: "
                  f"{feedback} Follow the checklist exactly.")
    return PROMPT.format(domain=cell["domain"], style=cell["style"],
                         nlines=nlines, roxygen=roxy, constructs=cons,
                         ncomments=ncom, context=cell["context"], extra=extra)


# ---------------------------------------------------------------------------
# gates: the authored function must TRUTHFULLY match the cell
# ---------------------------------------------------------------------------

def _norm_lines(text: str) -> list[str]:
    return [l.rstrip() for l in text.replace("\r\n", "\n").split("\n")]


def gate_sample(cell: dict, fn_text: str) -> tuple[dict, list[str]]:
    g: dict = {}
    lines = _norm_lines(fn_text)
    body = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    g["parses"] = True
    tb = S.parser.parse(fn_text.encode())
    g["parses"] = not tb.root_node.has_error
    if not g["parses"]:
        return g, []
    fns = [n for n in S.traverse(tb.root_node)
           if n.type == "function_definition"]
    g["one_function"] = len(fns) == 1
    if not g["one_function"]:
        return g, []
    # interior plain comments (between the braces)
    inside = False
    ncom = 0
    for l in lines:
        t = l.strip()
        if not inside and re.search(r"<-\s*function\s*\(", l):
            inside = True
            continue
        if inside:
            if t.startswith("#") and not t.startswith("#'") and len(t) > 2:
                ncom += 1
            if t == "}":
                break
    want_com = {"none": 0, "sparse": 2, "rich": 3}[cell["comments"]]
    g["comments"] = (ncom == 0 if cell["comments"] == "none"
                     else (1 <= ncom <= want_com if cell["comments"] == "sparse"
                           else ncom >= want_com))
    lo, hi = {"S": (3, 9), "M": (8, 22), "L": (20, 42)}[cell["length"]]
    g["length"] = lo <= len(body) <= hi
    g["constructs"] = all(any(tok in fn_text for tok in CONSTRUCTS[c][1])
                          for c in cell["constructs"])
    g["roxygen"] = (cell["roxygen"] == "none"
                    and "#'" not in fn_text) or \
                   (cell["roxygen"] == "title-only"
                    and fn_text.count("#' @param") == 0 and "#" in fn_text) or \
                   (cell["roxygen"] == "full"
                    and "#' @param" in fn_text and "#' @return" in fn_text)
    g["no_placeholder"] = not re.search(r"TODO|FIXME|sessionInfo|placeholder",
                                        fn_text)
    # Rscript parse (belt and braces vs tree-sitter)
    fh = tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     dir="/tmp")
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
    return g, []


GATE_ORDER = ("parses", "one_function", "rscript_parse", "comments", "length",
              "constructs", "roxygen", "no_placeholder")


def gate_summary(gates: dict) -> str:
    fails = [k for k in GATE_ORDER if gates.get(k) is False]
    return ",".join(fails)


def verified_props(cell: dict, fn_text: str, g: dict) -> dict:
    lines = _norm_lines(fn_text)
    return dict(requested=cell, body_lines=len(
        [l for l in lines if l.strip() and not l.strip().startswith("#")]),
        comments=cell["comments"], gates={k: v for k, v in g.items()})


def derive_rows(base_id: str, fn_text: str, cell: dict, g: dict,
                prompt: str) -> tuple[list[dict], dict]:
    """Synthetic Bundle -> BaseSample -> the deterministic matrix."""
    b = S.Bundle(f"author:{cell['domain'].split()[0]}", f"author/{base_id}.R",
                 (fn_text + "\n").encode())
    tb = S.parser.parse(b.src)
    fn = next((n for n in S.traverse(tb.root_node)
               if n.type == "function_definition"), None)
    if fn is None:
        return [], dict(error="no function node")
    bs = CP.BaseSample(b, fn, 0)
    rows, stats = CP.run_matrix([bs], dict(seed=13))
    out = []
    for r in rows:
        r = dict(r)
        r["base_sample"] = base_id
        r["case"] = "compound_author_spark"
        r["backend"] = "opencode-spark"
        r["model"] = "muse-spark-1.2-contributor"
        r["origin_kind"] = "author"
        r["origin_cell"] = cell_id(cell)
        r["full_prompt"] = prompt
        r["generated_at"] = _now()
        r["content_hash"] = hashlib.sha1(
            (f"{r.get('family')}\x00{r.get('transform')}\x00{base_id}\x00"
             f"{json.dumps(r.get('region_old', []))}").encode()).hexdigest()
        out.append(r)
    return out, stats


# ---------------------------------------------------------------------------
# author loop
# ---------------------------------------------------------------------------

def cmd_author(args) -> int:
    backend = RAS.resolve_backend(args.backend)
    rng = random.Random(args.seed)
    done_path = Path(str(OUT_ROWS) + ".done.jsonl")
    done = ZA.load_done(done_path)
    hashes = ZA.load_hashes(OUT_ROWS)
    breaker = ZA.Breaker()
    stats = dict(started=_now(), backend=backend.name, model=backend.model,
                 workers=args.workers, target=args.n, attempted=0,
                 accepted=0, rows=0, backend_error=0, retried=0,
                 gate_rejects=Counter(), dups=0, cells=Counter(),
                 per_family=Counter())
    lock = threading.Lock()
    stop_hard = threading.Event()
    t0 = time.time()
    last_log = 0.0
    n_ok = 0

    def process(i: int) -> dict:
        cell = draw_cell(rng)
        cid = cell_id(cell)
        with lock:
            stats["attempted"] += 1
        feedback = None
        first_prompt = ""
        for attempt in (1, 2):
            if not breaker.wait_turn() or stop_hard.is_set():
                return dict(kind="aborted", key=cid)
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
                    if stats["backend_error"] >= 15:
                        stop_hard.set()
                return dict(kind="backend_error", key=cid)
            obj = ZA.extract_json_object(ZA.strip_fences(raw))
            fn_text = obj.get("function") if isinstance(obj, dict) else None
            if not isinstance(fn_text, str) or "function" not in fn_text:
                feedback = "the response was not the JSON object with the function"
                continue
            fn_text = "\n".join(ZA._norm_trim(fn_text))
            g, _ = gate_sample(cell, fn_text)
            summary = gate_summary(g)
            if not summary:
                base_id = "bs:" + hashlib.sha1(
                    fn_text.encode("utf-8", "surrogateescape")).hexdigest()[:16]
                rows, dstats = derive_rows(base_id, fn_text, cell, g,
                                           first_prompt)
                return dict(kind="accepted", key=f"{cid}:{base_id}",
                            cell=cell, base_id=base_id, fn_text=fn_text,
                            rows=rows, gates=g, dstats=dstats,
                            attempts=attempt)
            feedback = f"checklist mismatch: {summary}"
            if attempt == 2:
                with lock:
                    for k in summary.split(","):
                        stats["gate_rejects"][k] += 1
                return dict(kind="rejected", key=cid, reason=summary)
            with lock:
                stats["retried"] += 1
        return dict(kind="rejected", key=cid, reason="unparseable")

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
                except Exception as e:
                    print(f"  [worker-exception] {e!r}", flush=True)
                    continue
                n_done += 1
                key = res["key"]
                if key in done:
                    continue
                if res["kind"] == "accepted":
                    base_rec = dict(
                        id=res["base_id"], origin=dict(kind="author",
                                                       package=f"author",
                                                       path=f"author/{res['base_id']}.R"),
                        code=dict(full_text=res["fn_text"]),
                        properties=verified_props(res["cell"], res["fn_text"],
                                                  res["gates"]),
                        derivations=[dict(transform=r.get("transform"),
                                          family=r.get("family"))
                                     for r in res["rows"]],
                        case="compound_author_spark", backend=backend.name,
                        model=backend.model, generated_at=_now())
                    ZA._append_line(OUT_BASES, base_rec)
                    new_rows = []
                    for row in res["rows"]:
                        if row["content_hash"] in hashes:
                            with lock:
                                stats["dups"] += 1
                            continue
                        hashes.add(row["content_hash"])
                        ZA._append_line(OUT_ROWS, row)
                        new_rows.append(row)
                    with lock:
                        stats["accepted"] += 1
                        stats["rows"] += len(new_rows)
                        stats["cells"][cell_id(res["cell"])] += 1
                        for r in new_rows:
                            stats["per_family"][r.get("family", "?")] += 1
                    rec = dict(key=key, ok=True, rows=len(new_rows),
                               families=sorted({r.get("family")
                                                for r in res["rows"]}),
                               ts=_now())
                elif res["kind"] == "rejected":
                    rec = dict(key=key, ok=False, reason=res["reason"][:160],
                               ts=_now())
                else:
                    rec = None
                if rec is not None:
                    ZA._append_line(done_path, rec)
                    done[key] = rec
                    if rec.get("ok"):
                        n_ok += 1
            if time.time() - last_log > 60:
                last_log = time.time()
                bs = backend.stats_summary()
                print(f"  [progress] samples={n_done}/{args.n} "
                      f"accepted={stats['accepted']} rows={stats['rows']} "
                      f"families={dict(stats['per_family'])} "
                      f"rejects={dict(stats['gate_rejects'])} "
                      f"backend_ok={bs['ok']} 429={bs['err_429']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                ZA._write_json(Path(str(OUT_ROWS) + ".stats.json"),
                               dict(stats, ts=_now(),
                                    elapsed_s=round(time.time() - t0, 1),
                                    backend_stats=backend.stats_summary(),
                                    counts=dict(stats)))
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        ZA._write_json(Path(str(OUT_ROWS) + ".stats.json"),
                       dict(stats, ts=_now(), final=True,
                            elapsed_s=round(time.time() - t0, 1),
                            backend_stats=backend.stats_summary(),
                            counts=dict(stats)))
    print(f"[author] FINISHED samples={n_done} accepted={stats['accepted']} "
          f"rows={stats['rows']} in {time.time()-t0:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 compound_author_spark.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("author")
    p.add_argument("--backend", default="opencode-spark")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--time-budget", type=float, default=43200)
    p.set_defaults(fn=cmd_author)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
