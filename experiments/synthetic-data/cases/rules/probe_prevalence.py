#!/usr/bin/env python3
"""probe_prevalence.py — measured prevalence of the rule-registry catalog
signals on a sample of /mnt/h/sepalith/normalized.

Two detector tiers over one seeded package sample (highest version per
package, <= 40 files/pkg, the iter_bundles_highest convention):

  regex tier  per FILE, on string-stripped source (crude proxies — upper
              bounds; the doc labels them as such)
  ast tier    per FUNCTION via tree-sitter: the six REAL registry detectors
              (the number the matrix actually yields), the smell metrics
              (cyclo/nesting/body-length histograms), and structural
              signals that regex cannot see (library-in-function, unused
              locals, dead code after return, [[i]] growth in loops, ifelse
              in loops, switch without default, partial-arg matches)

Output: results/rules_proto/prevalence.json + a printed per-signal table
with corpus-wide extrapolation (x 14202/packages_scanned on the per-package
file and function rates; the 40-file cap undercounts big packages — the
extrapolations are conservative FLOORS).
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
from cases.rules import load_rules          # noqa: E402
from cases.rules.rules_metadata import _cyclo, _max_nesting  # noqa: E402

OUT_DEFAULT = HERE.parents[1] / "results" / "rules_proto" / "prevalence.json"

# ---------------------------------------------------------------------------
# regex tier (per file; src = string-stripped bytes)
# ---------------------------------------------------------------------------

REGEX_SIGNALS = {
    "tf_any": [re.compile(rb"(?<![\w.$@])[TF](?![\w.$@])")],
    "one_length_any": [re.compile(rb"(?<![\w.\d])\d+\s*:\s*(?:length|nrow|ncol|NROW|NCOL)\s*\(")],
    "one_length_lit1": [re.compile(rb"(?<![\w.\d])1\s*:\s*(?:length|nrow|ncol|NROW|NCOL)\s*\(")],
    "paste_any": [re.compile(rb"(?<![\w.])paste\s*\(")],
    "paste_sep_empty": [re.compile(rb'(?<![\w.])paste0?\s*\('), re.compile(rb'sep\s*=\s*""')],
    "class_eq": [re.compile(rb"class\s*\([^)\n]{0,80}\)\s*(?:==|!=)")],
    "sapply_any": [re.compile(rb"(?<![\w.])sapply\s*\(")],
    "ifelse_any": [re.compile(rb"(?<![\w.])ifelse\s*\(")],
    "superassign": [re.compile(rb"<<-")],
    "strings_as_factors": [re.compile(rb"stringsAsFactors")],
    "drop_hazard_proxy": [re.compile(rb"\[\s*[\w.\"']+\s*,\s*\]")],
    "seq_empty_call": [re.compile(rb"(?<![\w.])seq\s*\(\s*\)")],
    "stopifnot_existing": [re.compile(rb"(?<![\w.])stopifnot\s*\(")],
    "single_quote_any": [re.compile(rb"(?<![\w)])'[^'\n]{1,80}'")],
    "semicolon_eol": [re.compile(rb";[ \t]*(?:\n|$)")],
    "equals_top_assign": [re.compile(rb"(?m)^\s*[\w.]+\s*=\s*[^=]")],
    "todo_comment": [re.compile(rb"#.*(TODO|FIXME)")],
    "unnecessary_lambda": [re.compile(rb"(?<![\w.])(?:sapply|lapply|vapply)\s*\([^,]{1,40},\s*function\s*\(\s*(\w+)\s*\)\s*(\w+)\s*\(\s*\1\s*\)")],
    "in_chain_equals": [re.compile(rb"==[^\n&|]{1,60}\|\s*[\w.\"]+\s*==")],
    "vector_logic_proxy": [re.compile(rb"if\s*\([^)\n]*==[^)\n]*&&")],
    "library_call": [re.compile(rb"(?<![\w.])library\s*\(")],
    "partial_arg_na": [re.compile(rb"(?<![\w.])(?:mean|sd|var|median|sum|min|max|quantile|range|sort|order|round|signif)\s*\([^)\n]{0,120}?\bna\s*=")],
}

# ---------------------------------------------------------------------------
# ast tier helpers (per function)
# ---------------------------------------------------------------------------

_STAT_FNS = {"mean", "sd", "var", "median", "sum", "min", "max", "quantile"}
_PARTIAL_OK = {"na"}                     # unique prefix of na.rm


def _fn_signals(bs, rules):
    """Per-function AST signals for the probe."""
    src = bs.b.src
    sig = {}
    for rid, r in rules.items():
        if r.kind == "rewrite":
            try:
                sig[f"det:{rid}"] = len(r.detector(bs))
            except Exception:                       # noqa: BLE001
                sig[f"det:{rid}"] = 0
    sig["cyclo"] = _cyclo(src, bs.body)
    sig["nesting"] = _max_nesting(bs.body)
    sig["body_lines"] = bs.nbody
    callees = {S.callee_name(src, n) for n in V._walk(bs.body) if n.type == "call"}
    sig["library_in_fn"] = int("library" in callees or "require" in callees)
    # unused locals: LHS identifiers of top-level assignments never read as
    # a bare identifier elsewhere in the body (crude but tree-sitter-true)
    assigned = {}
    for st in C._body_statements(bs.body):
        if st.type == "binary_operator" and st.children and \
                st.children[0].type == "identifier":
            tok = src[st.children[0].start_byte:st.children[0].end_byte] \
                .decode("utf-8", "replace")
            assigned[tok] = st.children[0]
    reads = {}
    for n in V._walk(bs.body):
        if n.type != "identifier":
            continue
        if any(_same(a, n) for a in assigned.values()):
            continue
        tok = src[n.start_byte:n.end_byte].decode("utf-8", "replace")
        reads[tok] = reads.get(tok, 0) + 1
    sig["unused_locals"] = sum(1 for t in assigned if reads.get(t, 0) == 0)
    # dead code: a non-last top-level statement that is return/stop
    stmts = C._body_statements(bs.body)
    dead = 0
    for i, st in enumerate(stmts[:-1]):
        nxt = stmts[i + 1]
        if st.type == "call" and S.callee_name(src, st) in ("return", "stop") \
                and nxt.type != "comment":
            dead += 1
    sig["dead_after_terminator"] = dead
    # [[i]] growth inside for-loops (preallocation smell proxy)
    growth = 0
    for n in V._walk(bs.body):
        if n.type != "for_statement":
            continue
        for d in V._walk(n):
            if d.type == "binary_operator" and d.children and \
                    d.children[0].type == "subset2" and \
                    src[d.children[1].start_byte:d.children[1].end_byte] == b"<-":
                growth += 1
    sig["growth_in_loop"] = min(growth, 1)
    # ifelse inside for-loop body
    iif = 0
    for n in V._walk(bs.body):
        if n.type != "for_statement":
            continue
        for d in V._walk(n):
            if d.type == "call" and S.callee_name(src, d) == "ifelse":
                iif += 1
    sig["ifelse_in_loop"] = min(iif, 1)
    # switch without a default arm: last argument not named & not a call —
    # crude: switch(...) whose final argument is an identifier (fallthrough
    # value) counts as HAVING a default; a trailing comma means none
    sw_nodes = 0
    for n in V._walk(bs.body):
        if n.type != "call" or S.callee_name(src, n) != "switch":
            continue
        args = next((c for c in n.children if c.type == "arguments"), None)
        if args is None:
            continue
        argv = [a for a in args.children if a.type == "argument"]
        has_default = False
        if argv:
            last = argv[-1]
            named = any(c.type == "=" for c in last.children)
            val = V._argument_value(last)
            has_default = (not named) and val is not None and \
                val.type == "identifier"
        if not has_default:
            sw_nodes += 1
    sig["switch_no_default"] = sw_nodes
    # partial argument matching on stat-call formals (na = for na.rm)
    pa = 0
    for n in V._walk(bs.body):
        if n.type != "call" or S.callee_name(src, n) not in _STAT_FNS:
            continue
        args = next((c for c in n.children if c.type == "arguments"), None)
        for a in (args.children if args is not None else []):
            if a.type != "argument":
                continue
            kids = list(a.children)
            if not any(c.type == "=" for c in kids) or not kids:
                continue
            nm = src[kids[0].start_byte:kids[0].end_byte].decode(
                "utf-8", "replace")
            if nm in _PARTIAL_OK and nm not in ("na.rm",):
                pa += 1
    sig["partial_arg_match"] = pa
    return sig


def _same(a, b) -> bool:
    return a.type == b.type and a.start_byte == b.start_byte \
        and a.end_byte == b.end_byte


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python cases/rules/probe_prevalence.py")
    ap.add_argument("--packages", type=int, default=120)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--time-budget", type=float, default=560)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    rules = load_rules()
    rng = random.Random(args.seed)
    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in S.list_packages() if p not in set(tidy)]
    rng.shuffle(rest)
    pool = tidy[:args.packages // 4] + rest[:3 * args.packages // 4]

    st = dict(packages_scanned=0, files=0, functions=0, functions_sized=0,
              regex_hits={}, ast_fn={}, ast_any={}, cyclo_hist={},
              elapsed_s=0, detectors={})
    t0 = time.time()
    for b in C.iter_bundles_highest(pool, rng):
        if time.time() - t0 > args.time_budget:
            break
        st["files"] += 1
        if b.package not in st.setdefault("_pkgs", set()):
            st["_pkgs"].add(b.package)
            st["packages_scanned"] += 1
        stripped = S.strip_strings(b.src)
        for name, pats in REGEX_SIGNALS.items():
            if all(p.search(stripped) for p in pats):
                st["regex_hits"][name] = st["regex_hits"].get(name, 0) + 1
        # roxygen hygiene: @export without @return (doc_sync tie-in)
        text = b.src.decode("utf-8", "replace")
        n_exp = len(re.findall(r"#'\s*@export", text))
        n_ret = len(re.findall(r"#'\s*@return", text))
        if n_exp and not n_ret:
            st["regex_hits"]["roxygen_export_no_return"] = \
                st["regex_hits"].get("roxygen_export_no_return", 0) + 1
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            st["functions"] += 1
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            _body, _head, _r0, _r1, nb = geom
            if not 3 <= len(nb) <= 60:
                continue
            st["functions_sized"] += 1
            try:
                bs = BaseSample(b, fn, -1)
                sig = _fn_signals(bs, rules)
            except Exception:                           # noqa: BLE001
                continue
            for k, v in sig.items():
                if k in ("cyclo", "nesting", "body_lines"):
                    continue                      # distributions, not hits
                if v:
                    st["ast_fn"][k] = st["ast_fn"].get(k, 0) + v
                    st["ast_any"][k] = st["ast_any"].get(k, 0) + 1
            c = min(sig["cyclo"], 30)
            st["cyclo_hist"][str(c)] = st["cyclo_hist"].get(str(c), 0) + 1
            for key, thr in (("fn:cyclo>=8", 8), ("fn:cyclo>=15", 15),
                             ("fn:nesting>=5", 5), ("fn:body>40", 40)):
                if (sig["cyclo"] >= thr if key.startswith("fn:cyclo")
                        else sig["nesting"] >= thr if
                        key.startswith("fn:nesting") else sig["body_lines"] > thr):
                    st["ast_any"][key] = st["ast_any"].get(key, 0) + 1
    st.pop("_pkgs", None)
    st["elapsed_s"] = round(time.time() - t0, 1)

    files, fns, pkgs = st["files"], st["functions_sized"], st["packages_scanned"]
    scale_files = 14202 / max(1, pkgs)
    scale_fns = 14202 / max(1, pkgs)
    print(f"[probe] packages={pkgs} files={files} functions={st['functions']} "
          f"(sized {fns}) in {st['elapsed_s']}s; per-pkg: "
          f"{files / max(1, pkgs):.1f} files, {fns / max(1, pkgs):.1f} sized fns")
    print(f"{'signal':34s} {'files%':>7s} {'~corpus files':>14s}   "
          f"{'fn%':>7s} {'~corpus fns':>12s}")
    order = (sorted(REGEX_SIGNALS) + ["roxygen_export_no_return"]
             + sorted(st["ast_any"]))
    for name in order:
        fr = st["regex_hits"].get(name)
        ar = st["ast_any"].get(name)
        tot = st["ast_fn"].get(name)
        fpc = f"{100 * fr / files:.2f}%" if fr is not None else ""
        fxc = f"{int(fr * scale_files):,}" if fr is not None else ""
        apc = f"{100 * ar / fns:.3f}%" if ar is not None else ""
        axc = f"{int(ar * scale_fns):,}" if ar is not None else ""
        extra = f" (sites {tot})" if tot else ""
        if fr is None and ar is None:
            continue
        print(f"  {name:32s} {fpc:>7s} {fxc:>14s}   {apc:>7s} {axc:>12s}{extra}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(st, indent=1))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
