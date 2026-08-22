#!/usr/bin/env python3
"""doc_sync_spark.py — doc_sync reinforcement with muse-spark as author.

doc_sync is the weakest SFT family (v6 regressed to 0.0) because every
existing row's @param descriptions are the DETERMINISTIC name-grammar
strings (scenarios.doc_desc_for_name) — validate_example's missing_param
branch hard-rejects anything else, so only grammar-covered param names
ever produced rows and every description is formulaic.

This driver keeps the PROVEN v2 missing_param mining geometry (scenarios.
extract_doc_sync_missing: real undocumented signature params, the
event = real signature with those args removed, region = the block's
@param tag window, cursor at end of last tag) but:

  * mines ANY valid identifier param (not just grammar-covered names);
  * the description is AUTHORED by muse-spark, grounded in the function
    body (how the code actually uses each argument);
  * a NEW variant "llm_param" replaces the canned string-equality gate
    with content gates: exact name coverage in signature order, style
    normalisation copied from the block's own @param tags (capitalise /
    trailing period via scenarios._doc_style + _styled_desc), 3-25 word
    single-line descriptions with no tag injection, region_new =
    region_old + appended lines only, _sig_args_added still verifying the
    event, and a tree-sitter splice re-parse of the whole file.

Outputs (cases conventions, family stays "doc_sync" so this is direct
reinforcement of the existing family; per-row case/backend/model fields
attribute the author):
  /mnt/h/sepalith/datasets/cases_v1/doc_sync_spark.jsonl (+ .done.jsonl,
  + .stats.json). Resume-safe done-keys; <=N in flight; 429 breaker.

Usage (system python3 from experiments/synthetic-data):
  python3 doc_sync_spark.py mine [--specs 4000]
  python3 doc_sync_spark.py author [--workers 6]
  python3 doc_sync_spark.py stats
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scenarios as S                                   # noqa: E402
import rewrite_author_spark as RAS                      # noqa: E402
import rewrite_author_zai as ZA                         # noqa: E402
from cases.backends import BackendError                 # noqa: E402

OUT_DIR = HERE / "results" / "doc_sync_spark"
POOL_PATH = OUT_DIR / "spec_pool.jsonl"
DATASETS = Path("/mnt/h/sepalith/datasets/cases_v1")
OUT_FILE = DATASETS / "doc_sync_spark.jsonl"

FN_LINES_MAX = 60          # prompt grounding cap
DESC_MAX_WORDS = 25
DESC_MIN_WORDS = 3
DESC_MAX_CHARS = 200


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# mining: v2 missing_param geometry, grammar restriction lifted
# ---------------------------------------------------------------------------

def extract_spec(b: S.Bundle, cap: int = 2) -> list[dict]:
    """Specs for functions whose roxygen block documents SOME signature
    params but not all (scenarios.extract_doc_sync_missing criteria, minus
    the doc_desc_for_name grammar filter)."""
    src = b.src
    out: list[dict] = []
    for fn in S.traverse(b.tree.root_node):
        if len(out) >= cap:
            break
        if fn.type != "function_definition":
            continue
        params = next((c for c in fn.children if c.type == "parameters"),
                      None)
        if params is None:
            continue
        parent = fn.parent
        if (parent is None or parent.type != "binary_operator"
                or not parent.children
                or parent.children[0].type != "identifier"):
            continue
        ordered = S._ordered_params(src, params)
        if not ordered:
            continue
        fn_name = S.node_text(src, parent.children[0]) \
            .decode("utf-8", "replace")
        top_row = min(fn.start_point[0], parent.children[0].start_point[0])
        if top_row <= 0:
            continue
        r, block = top_row - 1, []
        while r >= 0 and S.ROXY_LINE_RE.match(b.line_str(r)):
            block.append(r)
            r -= 1
        if not block:
            continue
        block.reverse()
        block_lines = [b.line_str(rr) for rr in block]
        if any(re.match(r"^\s*#'\s*@inherit", l) for l in block_lines):
            continue
        documented = set(S._roxy_param_names(block_lines))
        pnames = [nm for nm, _ in ordered]
        if not (documented & set(pnames)):
            continue
        prow = [rr for rr in block if S.ROXY_PARAM_TAG_RE.match(b.line_str(rr))]
        if not prow:
            continue
        cand = [(nm, node) for nm, node in ordered
                if nm not in documented and nm != "..."
                and S.IDENT_RE.match(nm) and nm not in S.RESERVED]
        if not cand:
            continue
        rr = prow[-1] + 1
        while rr <= block[-1]:
            t = b.line_str(rr)
            m = S.ROXY_TAG_RE.match(t) if hasattr(S, "ROXY_TAG_RE") else None
            if (not S.ROXY_LINE_RE.match(t) or m or t.strip() == "#'"):
                break
            rr += 1
        win = list(range(prow[0], rr))
        if not (win[0] >= 1 and 0 < len(win) <= S.MISSING_REGION_MAX_LINES):
            continue
        region_old = [b.line_str(x) for x in win]
        if any(len(l) > S.FORMAT_LINE_MAX for l in region_old):
            continue
        end = params.end_byte
        if src[end - 1:end] != b")":
            continue
        sig_first = top_row
        sig_last = b.rowcol(end - 1)[0]
        if not (1 <= sig_last - sig_first + 1 <= S.MISSING_EVENT_MAX_LINES):
            continue
        event = None
        missing = None
        for k in range(min(S.MISSING_MAX_PARAMS, len(cand)), 0, -1):
            event = S._missing_event_lines(b, ordered, cand[:k],
                                           sig_first, sig_last)
            if event is not None:
                missing = cand[:k]
                break
        if missing is None or event is None:
            continue
        ev_old_lines, ev_new_lines = event
        # grounding text: the function's defining statement, capped
        fstart = parent.start_byte \
            if parent.type == "binary_operator" else fn.start_byte
        fn_text = src[fstart:fn.end_byte].decode("utf-8", "replace")
        if fn_text.count("\n") > FN_LINES_MAX:
            fn_text = "\n".join(fn_text.split("\n")[:FN_LINES_MAX]) \
                + "\n# ... (truncated)"
        defaults = {nm: S.node_text(src, node).decode("utf-8", "replace")
                    for nm, node in missing}
        style = S._doc_style(S._roxy_param_descs(region_old))
        origin = f"{b.package}|{b.rel}"
        base_id = "bs:" + hashlib.sha1(
            (origin + "\x00" + fn_name + "\x00"
             + ",".join(nm for nm, _ in ordered)).encode(
                 "utf-8", "surrogateescape")).hexdigest()[:16]
        names = [nm for nm, _ in missing]
        spec = dict(
            id=f"ds:{base_id}:+{'+'.join(names)}", base_sample_id=base_id,
            package=b.package, path=b.rel, fn_name=fn_name,
            region_rows=[win[0], win[-1]], region_old=region_old,
            prefix=[b.line_str(x) for x in range(max(0, win[0] - 10),
                                                 win[0])],
            missing=names, defaults=defaults,
            ev_old_lines=ev_old_lines, ev_new_lines=ev_new_lines,
            sig_first=sig_first, style=dict(cap=style[0], period=style[1]),
            fn_text=fn_text,
            src_b64=base64.b64encode(src).decode("ascii"),
            generated_at=_now())
        out.append(spec)
    return out


def cmd_mine(args) -> int:
    rng = random.Random(args.seed)
    existing: set[str] = set()
    if POOL_PATH.exists() and not args.fresh:
        for line in POOL_PATH.read_text().splitlines():
            try:
                existing.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
    pkgs = RAS.RW.sample_packages(rng, args.tidy_packages,
                                  args.random_packages)
    print(f"[mine] {len(pkgs)} packages (seed={args.seed}), "
          f"{len(existing)} specs already pooled", flush=True)
    quota = args.specs
    stats = dict(files=0, functions=0, specs=0, packages=set(),
                 per_nmissing=Counter())
    buf: list[dict] = []
    t0 = time.time()
    for b in S.iter_bundles(pkgs, rng):
        if quota <= 0 or time.time() - t0 > args.time_budget:
            break
        stats["files"] += 1
        stats["packages"].add(b.package)
        for fn in (n for n in S.traverse(b.tree.root_node)
                   if n.type == "function_definition"):
            stats["functions"] += 1
        for spec in extract_spec(b):
            if quota <= 0:
                break
            buf.append(spec)
            quota -= 1
            stats["specs"] += 1
            stats["per_nmissing"][len(spec["missing"])] += 1
    new = [s for s in buf if s["id"] not in existing]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL_PATH, "a") as fh:
        for s in new:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    report = dict(ts=_now(), seed=args.seed,
                  elapsed_s=round(time.time() - t0, 1),
                  files=stats["files"], functions=stats["functions"],
                  packages=len(stats["packages"]), new_specs=len(new),
                  pool_total=len(existing) + len(new),
                  per_nmissing=dict(stats["per_nmissing"]),
                  quota_left=quota)
    ZA._write_json(OUT_DIR / "mine_report.json", report)
    print(f"[mine] +{len(new)} specs in {report['elapsed_s']}s; "
          f"pool now {report['pool_total']}; "
          f"per_nmissing={report['per_nmissing']}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# authoring
# ---------------------------------------------------------------------------

PROMPT_HEADER = """You are completing the roxygen documentation of an R function. Its documentation block already documents some parameters, but {n} parameter(s) of the signature are missing a @param tag: {names}.

Write the missing "#' @param <name> <description>" line for EACH missing parameter. The description must be grounded in how the function below actually USES the argument: say what it is (type/shape when clear from the code) and what it does here. Match the wording style of the existing @param lines (do not repeat the parameter name as the whole description). One line each, {lo}-{hi} words, no markdown.

Existing @param lines in the block (match their style):
{existing}

Function (may be truncated):
```r
{fn}
```

Return ONLY a JSON object mapping each missing parameter name to its description text (no "#' @param" prefix, no newlines):
{{"param_name": "description", ...}} with EXACTLY these keys: {names}"""


def build_prompt(spec: dict, feedback: str | None = None) -> str:
    p = PROMPT_HEADER.format(
        n=len(spec["missing"]), names=", ".join(spec["missing"]),
        existing="\n".join([l for l in spec["region_old"]
                            if S.ROXY_PARAM_TAG_RE.match(l)][:8]),
        fn=spec["fn_text"][:3000], lo=DESC_MIN_WORDS, hi=DESC_MAX_WORDS)
    if feedback:
        p += (f"\n\nNOTE: a previous attempt was REJECTED because: "
              f"{feedback} Return ONLY the JSON object with exactly the "
              f"listed keys.")
    return p


def extract_params(raw: str) -> dict | None:
    obj = ZA.extract_json_object(S.strip_fences(raw)
                                 if hasattr(S, "strip_fences")
                                 else ZA.strip_fences(raw))
    if isinstance(obj, dict) and all(isinstance(v, str) for v in obj.values()):
        return obj
    return None


def desc_ok(name: str, desc: str) -> bool:
    d = desc.strip()
    if not d or "\n" in d or "@" in d or len(d) > DESC_MAX_CHARS:
        return False
    words = d.split()
    if not (DESC_MIN_WORDS <= len(words) <= DESC_MAX_WORDS):
        return False
    if d.lower().strip(". ") == name.replace("_", " ").lower():
        return False                       # bare name echo
    if d.lower().startswith(f"the {name.lower()} "):
        return len(words) >= DESC_MIN_WORDS   # fine if it says something
    return True


def run_gates(spec: dict, descs: dict) -> tuple[dict, list[str]]:
    """Validate authored descriptions; returns (gates, ins_lines)."""
    g: dict = {}
    names = spec["missing"]
    g["keys_exact"] = set(descs) == set(names)
    if not g["keys_exact"]:
        return g, []
    ok = {nm: desc_ok(nm, descs[nm]) for nm in names}
    g["desc_ok"] = all(ok.values())
    if not g["desc_ok"]:
        g["bad_descs"] = [nm for nm, v in ok.items() if not v]
        return g, []
    cap, per = spec["style"]["cap"], spec["style"]["period"]
    pm = re.match(r"\s*#'", spec["region_old"][-1])
    ins = []
    for nm in names:                      # signature order, styled by me
        d = descs[nm].strip().rstrip(".?!").strip()
        ins.append(f"{pm.group(0)} @param {nm} "
                   f"{S._styled_desc(d, cap, per)}")
    region_new = spec["region_old"] + ins
    g["region_append"] = region_new[:len(spec["region_old"])] == \
        spec["region_old"] and \
        1 <= len(ins) <= S.MISSING_MAX_PARAMS
    # splice re-parse: whole file with region_new in place of region_old
    src = base64.b64decode(spec["src_b64"])
    lines = src.decode("utf-8", "replace").split("\n")
    r0, r1 = spec["region_rows"]
    spliced = "\n".join(lines[:r0] + region_new + lines[r1 + 1:]) \
        .encode("utf-8", "surrogateescape")
    tb = S.parser.parse(spliced)
    g["splice"] = not tb.root_node.has_error
    g["event_sig"] = S._sig_args_added(spec["ev_old_lines"],
                                       spec["ev_new_lines"], names)
    g["names_order"] = [re.match(r"(\s*#') @param ([.\w]+|\.\.\.) (.+)$", l)
                        .group(2) for l in ins] == names
    return g, ins


def gate_summary(gates: dict) -> str:
    fails = [k for k in ("keys_exact", "desc_ok", "region_append", "splice",
                         "event_sig", "names_order")
             if gates.get(k) is False]
    if not fails:
        return ""
    d = ""
    if "desc_ok" in fails and gates.get("bad_descs"):
        d = f" (bad: {gates['bad_descs']})"
    return ",".join(fails) + d


def build_row(spec: dict, ins_lines: list[str], prompt: str) -> dict:
    region_new = spec["region_old"] + ins_lines
    names = spec["missing"]
    note = (f"document missing argument(s) {', '.join(names)} of "
            f"{spec['fn_name']} (event added them to the signature)")
    row = S.make_multiline_example(
        "doc_sync", spec["package"], spec["path"], spec["prefix"],
        spec["region_old"], region_new,
        len("\n".join(spec["region_old"])), spec["ev_old_lines"],
        spec["ev_new_lines"], spec["sig_first"] + 1, note)
    chash = hashlib.sha1(
        f"doc_sync\x00{spec['base_sample_id']}\x00{'+'.join(names)}"
        .encode()).hexdigest()
    row.update(dict(
        variant="llm_param", case="doc_sync_spark",
        base_sample_id=spec["base_sample_id"],
        derivation=dict(rule_id="doc_sync/llm_param", rule_version=1,
                        arm="missing_param", spec_id=spec["id"]),
        backend="opencode-spark", model="muse-spark-1.2-contributor",
        full_prompt=prompt, generated_at=_now(), content_hash=chash,
        constraint_spec=spec["id"], determinism="D3 author-LLM (gated)"))
    return row


def cmd_author(args) -> int:
    backend = RAS.resolve_backend(args.backend)
    if not POOL_PATH.exists():
        sys.exit(f"no spec pool at {POOL_PATH}; run `mine` first")
    specs = []
    for line in POOL_PATH.read_text().splitlines():
        try:
            specs.append(json.loads(line))
        except ValueError:
            continue
    done_path = Path(str(OUT_FILE) + ".done.jsonl")
    done = ZA.load_done(done_path)
    hashes = ZA.load_hashes(OUT_FILE)
    pending = [s for s in specs if s["id"] not in done]
    print(f"[author] backend={backend.name} model={backend.model} "
          f"pool={len(specs)} pending={len(pending)} workers={args.workers}",
          flush=True)
    if args.max > 0:
        pending = pending[:args.max]

    breaker = ZA.Breaker()
    stats = dict(started=_now(), backend=backend.name, model=backend.model,
                 workers=args.workers, pool=len(specs), pending=len(pending),
                 attempted=0, accepted=0, retried=0, backend_error=0,
                 layer1_json=0, gate_rejects=Counter(), dups=0, rows=0)
    lock = threading.Lock()
    stop_hard = threading.Event()
    t0 = time.time()
    last_log = 0.0

    def process(spec: dict) -> dict:
        with lock:
            stats["attempted"] += 1
        feedback = None
        first_prompt = ""
        for attempt in (1, 2):
            if not breaker.wait_turn() or stop_hard.is_set():
                return dict(kind="aborted", spec=spec)
            prompt = build_prompt(spec, feedback)
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
                return dict(kind="backend_error", spec=spec)
            descs = extract_params(raw)
            if descs is None:
                with lock:
                    stats["layer1_json"] += 1
                feedback = "the response was not a JSON object mapping " \
                           "names to descriptions."
                continue
            gates, ins_lines = run_gates(spec, descs)
            summary = gate_summary(gates)
            if not summary:
                row = build_row(spec, ins_lines, first_prompt)
                row["backend"] = backend.name
                row["model"] = backend.model
                row["gates"] = gates
                return dict(kind="accepted", spec=spec, rows=[row],
                            attempts=attempt)
            feedback = summary
            if attempt == 2:
                with lock:
                    for k in summary.split(" (")[0].split(","):
                        stats["gate_rejects"][k] += 1
                return dict(kind="rejected", spec=spec, reason=summary)
            with lock:
                stats["retried"] += 1
        return dict(kind="rejected", spec=spec, reason="unparseable")

    def flush_stats(final: bool = False):
        rep = dict(stats)
        rep.update(dict(ts=_now(), out=str(OUT_FILE), final=final,
                        rows_total=stats["rows"],
                        elapsed_s=round(time.time() - t0, 1),
                        backend_stats=backend.stats_summary(),
                        breaker=dict(rate_errors=breaker.rate_errors,
                                     rounds=breaker.rounds,
                                     stopped=breaker.stop.is_set()),
                        counts={k: (dict(v) if isinstance(v, Counter) else v)
                                for k, v in stats.items()},
                        done_keys=len(done)))
        ZA._write_json(Path(str(OUT_FILE) + ".stats.json"), rep)

    ex = ThreadPoolExecutor(max_workers=args.workers)
    outstanding: set[Future] = set()
    it = iter(pending)
    n_done = 0
    try:
        while True:
            if time.time() - t0 > args.time_budget or breaker.stop.is_set() \
                    or stop_hard.is_set():
                break
            while len(outstanding) < args.workers * 2:
                try:
                    s = next(it)
                except StopIteration:
                    break
                outstanding.add(ex.submit(process, s))
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
                spec = res["spec"]
                if res["kind"] == "accepted":
                    row = res["rows"][0]
                    if row["content_hash"] in hashes:
                        with lock:
                            stats["dups"] += 1
                    else:
                        hashes.add(row["content_hash"])
                        ZA._append_line(OUT_FILE, row)
                        with lock:
                            stats["rows"] += 1
                            stats["accepted"] += 1
                    rec = dict(key=spec["id"], ok=True, attempts=res["attempts"],
                               ts=_now())
                elif res["kind"] == "rejected":
                    rec = dict(key=spec["id"], ok=False,
                               reason=res["reason"][:200], ts=_now())
                else:
                    rec = None
                if rec is not None:
                    ZA._append_line(done_path, rec)
                    done[spec["id"]] = rec
            if time.time() - last_log > 60:
                last_log = time.time()
                bs = backend.stats_summary()
                print(f"  [progress] done={n_done}/{len(pending)} "
                      f"rows={stats['rows']} "
                      f"rejects={dict(stats['gate_rejects'])} "
                      f"backend_ok={bs['ok']} 429={bs['err_429']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                flush_stats()
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        flush_stats(final=True)
    why = ("breaker" if breaker.stop.is_set() else
           "backend-hard-fail" if stop_hard.is_set() else
           "drained" if n_done >= len(pending) else "time-budget")
    print(f"[author] FINISHED done={n_done}/{len(pending)} "
          f"rows={stats['rows']} in {time.time()-t0:.0f}s; stopped={why}",
          flush=True)
    return 0


def cmd_stats(_args) -> int:
    sp = Path(str(OUT_FILE) + ".stats.json")
    if sp.exists():
        print(sp.read_text())
    else:
        print("(no stats yet)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 doc_sync_spark.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mine")
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--tidy-packages", type=int, default=438)
    p.add_argument("--random-packages", type=int, default=6000)
    p.add_argument("--specs", type=int, default=4000)
    p.add_argument("--time-budget", type=float, default=3600)
    p.add_argument("--fresh", action="store_true")
    p.set_defaults(fn=cmd_mine)
    p = sub.add_parser("author")
    p.add_argument("--backend", default="opencode-spark")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--time-budget", type=float, default=43200)
    p.set_defaults(fn=cmd_author)
    p = sub.add_parser("stats")
    p.set_defaults(fn=cmd_stats)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
