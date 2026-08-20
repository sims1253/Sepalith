#!/usr/bin/env python3
"""rewrite_maint_proto.py — "rewrite this block" maintainability-track prototype.

Maintainability-first rewrite exploration (static-analysis entry point): lint
findings and code smells as the QUANTIFIED definition of "better", a mocked
author-under-constraint loop with verifiable gates, a lint census of the real
corpus, code-smell metric distributions, and a modernization-yield census of
the existing git-mined edit_pairs corpus. Zero LLM calls, zero GPU, judge-free
ground truth. Companion design doc: docs/research/rewrite-maintainability-track.md.

Subcommands (all write under results/rewrite_maint_proto/):
  lint-census  Run jarl (+ry) on a sample of corpus FILES (copied to local tmp;
               jarl on drvfs is ~500x slower than on tmpfs) -> rule hit counts.
  smells       Tree-sitter code-smell metrics over corpus FUNCTIONS
               (cyclomatic, nesting, length, duplication, magic numbers, dead
               assigns) -> distributions + rewrite-yield estimates.
  edit-pairs   Classify a random sample of edit_pairs_v1 examples by
               modernization direction (purrr/dplyr/pipe/seq_along/... symbol
               deltas, styler-shaped whitespace diffs) -> yield estimate.
  constraint   The author-under-constraint loop: mine corpus functions WITH
               mechanical lint findings, emit the deterministic "fix exactly
               these findings, touch nothing else" spec per function, MOCK the
               author (5 hand-written mechanical fixers), then run the gates:
               parse, splice, diff-minimality, LOC, lint-delta (jarl before vs
               after), jarl-agreement (my detector vs jarl's own finding/fix),
               behavior preservation (Rscript before/after on simple inputs
               where the function is callable). Emits modernization_rewrite
               rows + honest pass rates.

Usage (system python3 from experiments/synthetic-data — NOT .venv-sft):
  python3 rewrite_maint_proto.py lint-census
  python3 rewrite_maint_proto.py smells
  python3 rewrite_maint_proto.py edit-pairs --sample 500
  python3 rewrite_maint_proto.py constraint --functions 50
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # scenarios/, cases/

import scenarios as S                             # noqa: E402
from scenarios import Bundle, node_text          # noqa: E402
import cases.corpus as C                          # noqa: E402
import cases.validators as V                      # noqa: E402
from cases.compound import BaseSample, _same_node  # noqa: E402

OUT_DIR = HERE / "results" / "rewrite_maint_proto"
GENERATED_AT = "2026-08-20T00:00:00"              # fixed: deterministic output
JARL_TMP = Path(tempfile.gettempdir()) / "rw_jarl_tmp"

EDIT_PAIRS = Path("/mnt/h/sepalith/datasets/edit_pairs_v1/examples.jsonl")

# jarl rule -> our finding class. Classes:
#   MECHANICAL — fix expressible as a rule; exact target derivable statically
#   JUDGMENT   — flagged, a rewrite exists, but the target is open-ended
#   RESTRAINT  — flagged (or info-level); the RIGHT move is usually no_op
#   HYGIENE    — suppression-comment plumbing, not a rewrite at all
# (Full 68-rule table with rationale lives in the design doc; this map carries
#  the rules the prototype measures.)
RULE_CLASS = {
    "true_false_symbol": "MECHANICAL", "seq": "MECHANICAL",
    "class_equals": "MECHANICAL", "paste0_sep": "MECHANICAL",
    "sapply_vapply": "MECHANICAL",
    "equals_na": "MECHANICAL", "equals_nan": "MECHANICAL", "equals_null": "MECHANICAL",
    "nzchar": "MECHANICAL", "length_levels": "MECHANICAL", "lengths": "MECHANICAL",
    "string_boundary": "MECHANICAL", "sprintf": "JUDGMENT",
    "any_is_na": "JUDGMENT", "outer_negation": "JUDGMENT", "redundant_ifelse": "JUDGMENT",
    "redundant_equals": "JUDGMENT", "unnecessary_nesting": "JUDGMENT",
    "comparison_negation": "MECHANICAL", "which_grepl": "MECHANICAL",
    "unreachable_code": "JUDGMENT", "empty_assignment": "MECHANICAL",
    "unused_function": "JUDGMENT", "browser": "MECHANICAL",
    "duplicated_arguments": "MECHANICAL", "duplicated_function_definition": "JUDGMENT",
    "for_loop_index": "JUDGMENT", "for_loop_dup_index": "JUDGMENT", "repeat": "JUDGMENT",
    "matrix_apply": "JUDGMENT", "list2df": "JUDGMENT", "sample_int": "MECHANICAL",
    "is_numeric": "JUDGMENT", "length_test": "JUDGMENT", "coalesce": "JUDGMENT",
    "all_equal": "JUDGMENT", "if_always_true": "JUDGMENT", "implicit_assignment": "JUDGMENT",
    "internal_function": "RESTRAINT", "undesirable_function": "RESTRAINT",
    "assignment": "RESTRAINT", "quotes": "RESTRAINT", "numeric_leading_zero": "RESTRAINT",
    "download_file": "RESTRAINT", "system_file": "RESTRAINT", "fixed_regex": "JUDGMENT",
    "grepv": "JUDGMENT", "sort": "JUDGMENT", "seq2": "MECHANICAL",
    "vector_logic": "JUDGMENT", "dplyr_filter_out": "JUDGMENT",
    "dplyr_group_by_ungroup": "JUDGMENT", "any_duplicated": "MECHANICAL",
    # suppression-comment rules (HYGIENE) and testthat rules (JUDGMENT in tests)
}
TESTTHAT_RULES = {"expect_length", "expect_match", "expect_named", "expect_no_match",
                  "expect_not", "expect_null", "expect_s3_class",
                  "expect_true_false", "expect_type"}
HYGIENE_RULES = {"blanket_suppression", "misnamed_suppression", "misplaced_suppression",
                 "misplaced_file_suppression", "outdated_suppression",
                 "unexplained_suppression", "unmatched_range_suppression",
                 "invalid_chunk_suppression"}


def jarl_json(paths: list[Path], timeout: int = 60) -> list[dict]:
    """jarl diagnostics for local file paths (JSON). One process, many files."""
    r = subprocess.run(
        ["jarl", "check", "--allow-no-vcs", "--output-format", "json",
         *(str(p) for p in paths)],
        capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(r.stdout).get("diagnostics", [])
    except (ValueError, AttributeError):
        return []


def ry_codes(path: Path, timeout: int = 30) -> Counter:
    r = subprocess.run(["ry", "check", str(path)], capture_output=True,
                       text=True, timeout=timeout)
    return Counter(re.findall(r"\b(RY\d+)\b", r.stdout + r.stderr))


# ---------------------------------------------------------------------------
# corpus sampling shared by lint-census / smells / constraint
# ---------------------------------------------------------------------------

def sample_packages(rng: random.Random, n_tidy: int, n_rest: int) -> list[str]:
    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in S.list_packages() if p not in set(tidy)]
    rng.shuffle(rest)
    pool = tidy[:n_tidy] + rest[:n_rest]
    rng.shuffle(pool)
    return pool


def fn_full_text(bs: BaseSample) -> str:
    """Defining statement text (assignment LHS + function) of the base sample."""
    src = bs.b.src
    start = bs.fn.parent.start_byte if (bs.fn.parent is not None
                                        and bs.fn.parent.type == "binary_operator") \
        else bs.fn.start_byte
    return src[start:bs.fn.end_byte].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# MODE 1: lint census (file-level, jarl + ry)
# ---------------------------------------------------------------------------

def cmd_lint_census(args) -> int:
    rng = random.Random(args.seed)
    pool = sample_packages(rng, args.tidy_packages, args.random_packages)
    JARL_TMP.mkdir(parents=True, exist_ok=True)
    local_files, meta = [], []
    per_pkg: Counter = Counter()
    per_pkg_cap = 2                      # breadth over depth: more packages
    t0 = time.time()
    for b in C.iter_bundles_highest(pool, rng):
        if len(local_files) >= args.files:
            break
        if per_pkg[b.package] >= per_pkg_cap:
            continue
        dst = JARL_TMP / f"census_{len(local_files):04d}.R"
        try:
            dst.write_bytes(b.src)
        except OSError:
            continue
        per_pkg[b.package] += 1
        local_files.append(dst)
        meta.append(dict(package=b.package, path=b.rel, bytes=len(b.src),
                         lines=b.nlines()))
    print(f"[census] {len(local_files)} files staged locally "
          f"({time.time()-t0:.0f}s)")

    # jarl: one process per batch of files (fast on tmpfs)
    t0 = time.time()
    all_diags = []
    for i in range(0, len(local_files), 25):
        all_diags += jarl_json(local_files[i:i + 25])
    jarl_secs = time.time() - t0
    per_rule = Counter(d["message"]["name"] for d in all_diags)
    fixable = Counter(d["message"]["name"] for d in all_diags
                      if d.get("fix") and not d["fix"].get("to_skip", False))
    files_hit = Counter()
    for d in all_diags:
        files_hit[Path(d["filename"]).name] += 1
    files_with = sum(1 for v in files_hit.values() if v)

    # ry: type/static diagnostics per file
    t0 = time.time()
    ry_per_rule = Counter()
    ry_files = 0
    for f in local_files:
        got = ry_codes(f)
        if got:
            ry_files += 1
        ry_per_rule += got
    ry_secs = time.time() - t0

    def klass(rule: str) -> str:
        if rule in HYGIENE_RULES:
            return "HYGIENE"
        if rule in TESTTHAT_RULES:
            return "JUDGMENT(test)"
        return RULE_CLASS.get(rule, "UNCLASSIFIED")

    by_class = Counter()
    for rule, n in per_rule.items():
        by_class[klass(rule)] += n
    report = dict(
        seed=args.seed, files=len(local_files),
        packages=sorted({m["package"] for m in meta}),
        jarl=dict(diagnostics=sum(per_rule.values()),
                  files_with_findings=files_with,
                  pct_files_with_findings=round(100 * files_with
                                                / max(1, len(local_files)), 1),
                  per_rule=dict(per_rule.most_common()),
                  fixable_per_rule=dict(fixable.most_common()),
                  by_class=dict(by_class.most_common()),
                  unclassified_rules=sorted(r for r in per_rule
                                            if klass(r) == "UNCLASSIFIED"),
                  seconds=round(jarl_secs, 1)),
        ry=dict(files_with_diag=ry_files, per_rule=dict(ry_per_rule.most_common()),
                seconds=round(ry_secs, 1)),
        file_meta=meta[:200],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lint_census.json").write_text(json.dumps(report, indent=1))
    print(f"[census] jarl: {sum(per_rule.values())} diagnostics in "
          f"{files_with}/{len(local_files)} files "
          f"({report['jarl']['pct_files_with_findings']}%), "
          f"ry: {sum(ry_per_rule.values())} diags in {ry_files} files")
    for rule, n in per_rule.most_common(15):
        print(f"  {rule:28s} {n:5d}  fixable={fixable.get(rule, 0):5d} "
              f"[{klass(rule)}]")
    return 0


# ---------------------------------------------------------------------------
# MODE 2: code-smell metrics (pure tree-sitter)
# ---------------------------------------------------------------------------

COMPARISON_OPS = {b"==", b"!=", b"<", b">", b"<=", b">="}
ARITH_OPS = {b"+", b"-", b"*", b"/", b"^", b"%%", b"%/%"}
BRANCH_CALLS = {"ifelse", "switch", "case_when", "fcase", "if_else"}
MAGIC_OK = {"0", "1", "2", "0.5", "1L", "2L", "0L", "3", "2L", "-1", "100"}


def fn_metrics(bs: BaseSample) -> dict:
    src, body = bs.b.src, bs.body
    cyclomatic = 1
    max_nesting = 0
    for n in V._walk(body):
        if n.type in ("if_statement", "for_statement", "while_statement",
                      "repeat_statement"):
            cyclomatic += 1
        elif n.type == "call" and S.callee_name(src, n) in BRANCH_CALLS:
            cyclomatic += 1
        elif n.type == "binary_operator" and len(n.children) > 1 \
                and node_text(src, n.children[1]) in (b"&&", b"||"):
            cyclomatic += 1
        if n.type in ("if_statement", "for_statement", "while_statement",
                      "repeat_statement", "braced_expression"):
            d, a = 0, n.parent
            while a is not None and not _same_node(a, body):
                if a.type in ("if_statement", "for_statement",
                              "while_statement", "repeat_statement"):
                    d += 1
                a = a.parent
            max_nesting = max(max_nesting, d)
    nb = [r for r in range(bs.r0 + 1, bs.r1) if bs.b.line_str(r).strip()]
    fn_lines = [bs.b.line_str(r) for r in range(bs.top_row, bs.r1 + 1)]
    # duplication: repeated normalized non-trivial lines within the function
    norm = [l.strip() for l in fn_lines
            if len(l.strip()) >= 20 and not l.strip().startswith("#")]
    dup_lines = sum(c - 1 for c in Counter(norm).values() if c > 1)
    # magic numbers: numeric literals outside ALLCAPS-named assignments
    magic = 0
    for n in V._walk(body):
        if n.type not in ("float", "integer"):
            continue
        txt = node_text(src, n).decode("utf-8", "replace")
        if txt.lstrip("-") in MAGIC_OK:
            continue
        p = n.parent
        if p is not None and p.type == "binary_operator" \
                and p.children and p.children[0].type == "identifier":
            lhs = node_text(src, p.children[0]).decode("utf-8", "replace")
            if lhs.isupper() or lhs.startswith("k"):
                continue
        magic += 1
    # dead assigns: local assigned once, never READ afterwards in the body
    body_txt = S.strip_strings(node_text(src, body)).decode("utf-8", "replace")
    dead = 0
    for n in V._walk(body):
        if n.type != "binary_operator" or len(n.children) < 3 \
                or node_text(src, n.children[1]) != b"<-":
            continue
        lhs = n.children[0]
        if lhs.type != "identifier":
            continue
        name = node_text(src, lhs).decode("utf-8", "replace")
        after = body_txt[n.end_byte - body.start_byte:]
        reads = len(re.findall(rf"(?<![\w.]){re.escape(name)}(?![\w.])", after))
        assigns = len(re.findall(rf"(?<![\w.]){re.escape(name)}(?![\w.])\s*(?:<-|=)",
                                 after))
        if reads == assigns == 0:
            dead += 1
    return dict(
        cyclomatic=cyclomatic, nesting=max_nesting, body_lines=len(nb),
        fn_lines=len(fn_lines), dup_lines=dup_lines, magic_numbers=magic,
        dead_assigns=dead)


def pct(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def cmd_smells(args) -> int:
    rng = random.Random(args.seed)
    pool = sample_packages(rng, args.tidy_packages, args.random_packages)
    t0, fns = time.time(), 0
    agg = dict(cyclomatic=[], nesting=[], body_lines=[], dup_lines=[],
               magic_numbers=[], dead_assigns=[])
    hot = dict(cyc_ge_10=0, nest_ge_4=0, body_ge_40=0, dup_ge_3=0,
               magic_ge_3=0, dead_ge_1=0, any_smell=0, all_clean=0)
    examples = []
    for b in C.iter_bundles_highest(pool, rng):
        if fns >= args.functions or time.time() - t0 > args.time_budget:
            break
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            if fns >= args.functions:
                break
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            _body, _head, r0, r1, nb = geom
            if not 4 <= len(nb) <= 120:
                continue
            try:
                bs = BaseSample(b, fn, -1)
            except ValueError:
                continue
            m = fn_metrics(bs)
            if m["body_lines"] < 4:
                continue
            fns += 1
            for k in agg:
                agg[k].append(m[k])
            hit = (m["cyclomatic"] >= 10) + (m["nesting"] >= 4) \
                + (m["body_lines"] >= 40) + (m["dup_lines"] >= 3) \
                + (m["magic_numbers"] >= 3) + (m["dead_assigns"] >= 1)
            if m["cyclomatic"] >= 10:
                hot["cyc_ge_10"] += 1
            if m["nesting"] >= 4:
                hot["nest_ge_4"] += 1
            if m["body_lines"] >= 40:
                hot["body_ge_40"] += 1
            if m["dup_lines"] >= 3:
                hot["dup_ge_3"] += 1
            if m["magic_numbers"] >= 3:
                hot["magic_ge_3"] += 1
            if m["dead_assigns"] >= 1:
                hot["dead_ge_1"] += 1
            if hit:
                hot["any_smell"] += 1
                if len(examples) < 12 and hit >= 2:
                    examples.append(dict(package=b.package, path=b.rel,
                                         row=r0, **m))
            else:
                hot["all_clean"] += 1
    out = dict(seed=args.seed, functions=fns, elapsed_s=round(time.time() - t0, 1),
               hot=hot, hot_pct={k: round(100 * v / max(1, fns), 1)
                                 for k, v in hot.items()},
               distributions={k: dict(mean=round(sum(v) / max(1, len(v)), 2),
                                      p50=pct(sorted(v), 0.50),
                                      p75=pct(sorted(v), 0.75),
                                      p90=pct(sorted(v), 0.90),
                                      p95=pct(sorted(v), 0.95),
                                      p99=pct(sorted(v), 0.99),
                                      max=max(v) if v else 0)
                              for k, v in agg.items()},
               examples=examples)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "smells.json").write_text(json.dumps(out, indent=1))
    print(f"[smells] {fns} functions in {out['elapsed_s']}s; "
          f"any_smell {out['hot_pct']['any_smell']}%")
    for k, v in out["distributions"].items():
        print(f"  {k:14s} mean={v['mean']:6.2f} p90={v['p90']} "
              f"p99={v['p99']} max={v['max']}")
    return 0


# ---------------------------------------------------------------------------
# MODE 3: edit_pairs modernization census
# ---------------------------------------------------------------------------

TIDY_VERBS = r"filter|mutate|select|arrange|group_by|summarise|summarize|slice|rename|left_join|inner_join|full_join|anti_join|across"
PURRR_FNS = r"map|map_chr|map_dbl|map_lgl|map_int|walk|walk2|imap|pmap|map2|modify|reduce|keep|discard|compact|flatten|possibly|safely|quietly"
STR_FNS = r"str_detect|str_replace|str_subset|str_extract|str_c|str_glue"


def _sym_delta(old: str, new: str, pat: str) -> int:
    o = len(re.findall(pat, old))
    n = len(re.findall(pat, new))
    return n - o


def classify_edit_pair(row: dict) -> tuple[str, list[str]]:
    old = "\n".join(row.get("region_old") or [])
    new = "\n".join(row.get("region_new") or [])
    path = row.get("path", "")
    labels = []
    o_nb = [l.strip() for l in old.splitlines() if l.strip()]
    n_nb = [l.strip() for l in new.splitlines() if l.strip()]
    ws_only = old != new and o_nb == n_nb
    if ws_only:
        labels.append("styler_whitespace")
    gains = {
        "purrr_gain": _sym_delta(old, new, rf"\b(?:purrr::)?(?:{PURRR_FNS})\("),
        "dplyr_gain": _sym_delta(old, new, rf"\b(?:dplyr::)?(?:{TIDY_VERBS})\("),
        "tidyr_gain": _sym_delta(old, new, r"\b(?:tidyr::)?(?:pivot_longer|pivot_wider|unite|separate|nest|unnest)\("),
        "ggplot_gain": _sym_delta(old, new, r"\b(?:ggplot\(|geom_|scale_|facet_|labs\()"),
        "stringr_gain": _sym_delta(old, new, rf"\b(?:stringr::)?(?:{STR_FNS})\("),
        "data_table_gain": _sym_delta(old, new, r"\b(?:data\.table\(|DT\[|:=)"),
        "vapply_gain": _sym_delta(old, new, r"\bvapply\("),
        "sapply_delta": _sym_delta(old, new, r"\bsapply\("),
        "lapply_gain": _sym_delta(old, new, r"\blapply\("),
        "for_loop_delta": _sym_delta(old, new, r"\bfor\s*\("),
        "seq_along_gain": _sym_delta(old, new, r"\bseq_along\(|\bseq_len\(") -
        _sym_delta(old, new, r"\b1:\s*(?:length|nrow|ncol)\("),
        "inherits_gain": _sym_delta(old, new, r"\binherits\(") -
        _sym_delta(old, new, r"\bclass\([^)]*\)\s*=="),
        "TF_fix": -(_sym_delta(old, new, r"(?<![\w.])[TF](?![\w.(])")),
        "pipe_gain": _sym_delta(old, new, r"%>%|\|>"),
    }
    if gains["purrr_gain"] > 0:
        labels.append("purrr_gain")
    if gains["vapply_gain"] > 0 and gains["sapply_delta"] < 0:
        labels.append("sapply_to_vapply")
    if gains["for_loop_delta"] < 0 and (gains["purrr_gain"] > 0
                                        or gains["lapply_gain"] > 0):
        labels.append("loop_to_apply")
    if gains["purrr_gain"] < 0:
        labels.append("purrr_loss")
    if gains["dplyr_gain"] > 0:
        labels.append("dplyr_gain")
    if gains["dplyr_gain"] < 0:
        labels.append("dplyr_loss")
    if gains["for_loop_delta"] > 0:
        labels.append("apply_to_loop")
    if gains["seq_along_gain"] > 0:
        labels.append("seq_safety_gain")
    if gains["inherits_gain"] > 0:
        labels.append("class_to_inherits")
    if gains["TF_fix"] > 0:
        labels.append("TF_symbol_fix")
    if gains["pipe_gain"] > 0:
        labels.append("pipe_gain")
    if gains["pipe_gain"] < 0:
        labels.append("pipe_loss")
    for k in ("tidyr_gain", "ggplot_gain", "stringr_gain", "data_table_gain"):
        if gains[k] > 0:
            labels.append(k)
    if path.endswith((".Rd", ".Rmd", ".md", ".Qmd", ".yaml", ".yml", ".csv")) \
            or "/man/" in path or "NEWS" in path or "CHANGELOG" in path.upper():
        labels.append("docs_file")
    elif "test" in path.lower() or "/tests/" in path:
        labels.append("tests_file")
    # primary label: strongest modernization signal, else doc/test/other
    prio = ["loop_to_apply", "sapply_to_vapply", "purrr_gain", "dplyr_gain",
            "tidyr_gain", "ggplot_gain", "stringr_gain", "data_table_gain",
            "seq_safety_gain", "class_to_inherits", "TF_symbol_fix",
            "pipe_gain", "purrr_loss", "dplyr_loss", "pipe_loss",
            "apply_to_loop", "styler_whitespace", "docs_file", "tests_file"]
    primary = next((p for p in prio if p in labels), None)
    if primary is None:
        primary = "other_code_edit"
    return primary, labels


def cmd_edit_pairs(args) -> int:
    rng = random.Random(args.seed)
    rows = []
    with open(EDIT_PAIRS) as fh:
        for i, line in enumerate(fh):
            rows.append((i, json.loads(line)))
    sample = rng.sample(rows, min(args.sample, len(rows)))
    prim, multi = Counter(), Counter()
    for _i, row in sample:
        p, labels = classify_edit_pair(row)
        prim[p] += 1
        for l in labels:
            multi[l] += 1
    n = len(sample)
    total = 15983
    modern = sum(v for k, v in prim.items()
                 if k not in ("other_code_edit", "docs_file", "tests_file",
                              "purrr_loss", "dplyr_loss", "pipe_loss",
                              "apply_to_loop", "styler_whitespace"))
    report = dict(seed=args.seed, sampled=n, corpus_total=total,
                  primary=dict(prim.most_common()),
                  primary_pct={k: round(100 * v / n, 2) for k, v in prim.items()},
                  primary_extrapolated={k: int(round(v * total / n))
                                        for k, v in prim.items()},
                  multilabel=dict(multi.most_common()),
                  directional_modernization_count=modern,
                  directional_modernization_pct=round(100 * modern / n, 2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "edit_pairs_modernization.json").write_text(
        json.dumps(report, indent=1))
    print(f"[edit-pairs] {n} rows classified; directional modernization "
          f"{modern} ({report['directional_modernization_pct']}%) "
          f"-> ~{int(round(modern * total / n))} of {total}")
    for k, v in prim.most_common(12):
        print(f"  {k:22s} {v:4d}  ({100*v/n:5.2f}%)  ~{int(round(v*total/n))}")
    return 0


# ---------------------------------------------------------------------------
# MODE 4: author-under-constraint loop (detectors -> spec -> mock author ->
# gates)
# ---------------------------------------------------------------------------

def _call_args(call_node) -> list:
    args = next((c for c in call_node.children if c.type == "arguments"), None)
    return [a for a in (args.children if args is not None else [])
            if a.type == "argument"]


def _arg_name(src, arg) -> str | None:
    if not any(c.type == "=" for c in arg.children):
        return None
    for c in arg.children:
        if c.type == "identifier":
            return node_text(src, c).decode("utf-8", "replace")
    return None


MECHANICAL_RULES = ("true_false_symbol", "seq_safety", "paste0_sep",
                    "sapply_vapply", "class_equals")


def detect_findings(bs: BaseSample) -> list[dict]:
    """Conservative static detectors for the 5 mechanical fix families,
    mirroring jarl/lintr rules. Every finding is a single-line byte splice."""
    src, b = bs.b.src, bs.b
    out = []

    def param_names() -> set[str]:
        params = next((c for c in bs.fn.children if c.type == "parameters"),
                      None)
        if params is None:
            return set()
        return {node_text(src, p).decode("utf-8", "replace")
                for p in V._walk(params) if p.type == "identifier"}

    params = param_names()

    def tf_literal_position(n) -> bool:
        """T/F used as a logical LITERAL (not param/LHS/callee/subset) AND the
        whole function is T/F-clean otherwise (scope matched to jarl's
        function-local analysis — a T anywhere else disables all T/F fixes)."""
        txt = node_text(src, n).decode("utf-8", "replace")
        if txt not in ("T", "F") or txt in params:
            return False
        p = n.parent
        if p is None or p.type == "parameters":
            return False
        if p.type == "call" and p.children and _same_node(p.children[0], n):
            return False                     # callee position T(...)
        if p.type == "binary_operator" and p.children \
                and _same_node(p.children[0], n) \
                and node_text(src, p.children[1]) in (b"<-", b"="):
            return False                     # assignment LHS
        if p.type in ("subset", "subset2"):
            return False                     # T[i] / T[[i]]
        nxt = src[n.end_byte:n.end_byte + 2]
        if nxt[:1] in (b"$", b"@", b"(") or nxt[:2] == b"::":
            return False
        return True

    tf_nodes = [n for n in V._walk(bs.fn)
                if n.type == "identifier"
                and node_text(src, n).decode("utf-8", "replace") in ("T", "F")]
    tf_all_literal = bool(tf_nodes) and all(tf_literal_position(n)
                                            for n in tf_nodes)

    for n in V._walk(bs.body):
        # (1) T/F logical symbols
        if n.type == "identifier" and tf_all_literal \
                and tf_literal_position(n):
            txt = node_text(src, n).decode("utf-8", "replace")
            out.append(dict(
                rule="true_false_symbol", row=n.start_point[0],
                erow=n.end_point[0],
                sb=n.start_byte, eb=n.end_byte, old=txt,
                new="TRUE" if txt == "T" else "FALSE",
                rationale="T/F can be shadowed; spell the logical"))
        # (2) 1:length(x) -> seq_along(x); 1:nrow(x) -> seq_len(nrow(x))
        if n.type == "binary_operator" and len(n.children) >= 3 \
                and node_text(src, n.children[1]) == b":":
            lhs, rhs = n.children[0], n.children[2]
            if node_text(src, lhs).decode("utf-8", "replace") != "1":
                continue
            if rhs.type != "call":
                continue
            callee = S.callee_name(src, rhs)
            args = _call_args(rhs)
            if callee == "length" and len(args) == 1:
                arg_txt = node_text(src, V._argument_value(args[0])
                                    or args[0]).decode("utf-8", "replace")
                new = f"seq_along({arg_txt})"
            elif callee in ("nrow", "ncol", "NROW", "NCOL") and len(args) == 1:
                rhs_txt = node_text(src, rhs).decode("utf-8", "replace")
                new = f"seq_len({rhs_txt})"
            else:
                continue
            out.append(dict(rule="seq_safety", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte,
                            old=node_text(src, n).decode("utf-8", "replace"),
                            new=new,
                            rationale="1:x yields c(1,0) when x is empty"))
        # (3) paste(..., sep = "") -> paste0(...)
        if n.type == "call" and S.callee_name(src, n) == "paste":
            args = _call_args(n)
            names = [_arg_name(src, a) for a in args]
            if "sep" not in names or "collapse" in names:
                continue
            sep_arg = next(a for a, nm in zip(args, names) if nm == "sep")
            val = V._argument_value(sep_arg)
            if val is None or node_text(src, val).decode(
                    "utf-8", "replace").strip() not in ('""', "''"):
                continue
            arg_start, arg_end = args[0].start_byte, args[-1].end_byte
            txt = src[arg_start:arg_end].decode("utf-8", "replace")
            new_args = re.sub(r",\s*sep\s*=\s*(?:\"\"|'')", "", txt)
            new_args = re.sub(r"sep\s*=\s*(?:\"\"|'')\s*,", "", new_args)
            if not new_args.strip() or re.search(r"\bsep\b", new_args):
                continue
            out.append(dict(rule="paste0_sep", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte,
                            old=node_text(src, n).decode("utf-8", "replace"),
                            new=f"paste0({new_args.strip()})",
                            rationale='paste(x, y, sep="") is paste0(x, y)'))
        # (4) sapply(X, \\(i) <inferable single expr>) -> vapply(..., FUN.VALUE)
        if n.type == "call" and S.callee_name(src, n) == "sapply":
            args = _call_args(n)
            if len(args) != 2:
                continue
            lam = V._argument_value(args[1])
            if lam is None or lam.type != "function_definition":
                continue
            pars = next((c for c in lam.children if c.type == "parameters"),
                        None)
            n_formals = len([c for c in (pars.children if pars else [])
                             if c.is_named])
            if n_formals != 1:
                continue
            br = next((c for c in lam.children if c.type == "braced_expression"),
                      None)
            stmts = ([c for c in br.children if c.is_named] if br is not None
                     else [c for c in lam.children if c.is_named
                           and c.type not in ("parameters",)])
            if len(stmts) != 1:
                continue          # exactly one statement/expression in the body
            expr = stmts[0]
            if expr is None:
                continue
            fun_value = None
            et = node_text(src, expr)
            if expr.type == "binary_operator" and len(expr.children) > 1:
                op = node_text(src, expr.children[1])
                if op in COMPARISON_OPS or op in (b"%in%", b"&&", b"||"):
                    fun_value = "logical(1)"
                elif op in ARITH_OPS:
                    fun_value = "numeric(1)"
            elif expr.type == "call":
                cn = S.callee_name(src, expr)
                if cn and re.fullmatch(r"is\.\w+", cn):
                    fun_value = "logical(1)"
                elif cn in ("nchar",):
                    fun_value = "integer(1)"
                elif cn in ("sqrt", "log", "log2", "log10", "exp", "abs",
                            "floor", "ceiling", "round", "signif", "sign"):
                    fun_value = "numeric(1)"
            if fun_value is None:
                continue
            x_txt = node_text(src, V._argument_value(args[0])
                              or args[0]).decode("utf-8", "replace")
            lam_txt = node_text(src, lam).decode("utf-8", "replace")
            out.append(dict(rule="sapply_vapply", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte,
                            old=node_text(src, n).decode("utf-8", "replace"),
                            new=f"vapply({x_txt}, {lam_txt}, {fun_value})",
                            rationale="vapply fails loudly on type drift"))
        # (5) class(x) == "foo" -> inherits(x, "foo")
        if n.type == "binary_operator" and len(n.children) >= 3 \
                and node_text(src, n.children[1]) == b"==":
            lhs, rhs = n.children[0], n.children[2]
            cls_node = lit_node = None
            if lhs.type == "call" and S.callee_name(src, lhs) == "class" \
                    and len(_call_args(lhs)) == 1 and rhs.type == "string":
                cls_node, lit_node = lhs, rhs
            elif rhs.type == "call" and S.callee_name(src, rhs) == "class" \
                    and len(_call_args(rhs)) == 1 and lhs.type == "string":
                cls_node, lit_node = rhs, lhs
            if cls_node is None:
                continue
            obj = V._argument_value(_call_args(cls_node)[0])
            obj_txt = node_text(src, obj or cls_node).decode("utf-8", "replace")
            lit = node_text(src, lit_node).decode("utf-8", "replace")
            out.append(dict(rule="class_equals", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte,
                            old=node_text(src, n).decode("utf-8", "replace"),
                            new=f"inherits({obj_txt}, {lit})",
                            rationale="class(x)=='foo' breaks on S3 subclasses"))
    return out


def apply_fixes(src: bytes, findings: list[dict]) -> bytes:
    """Mock author: apply every finding's splice (descending byte order)."""
    out = src
    for f in sorted(findings, key=lambda f: -f["sb"]):
        out = out[:f["sb"]] + f["new"].encode() + out[f["eb"]:]
    return out


def detect_injectable(bs: BaseSample) -> list[dict]:
    """INVERSE of detect_findings (the reverse-strip arm): corpus-clean sites
    whose 'dirty' twin is a mechanical finding. The corpus original becomes
    the tier-1 exact GT; the injected dirty variant is the prompt state."""
    src, b = bs.b.src, bs.b
    out = []
    for n in V._walk(bs.body):
        if n.type in ("true", "false"):
            txt = "T" if n.type == "true" else "F"
            out.append(dict(rule="true_false_symbol", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte,
                            old=node_text(src, n).decode("utf-8", "replace"),
                            new=txt, rationale="injected T/F symbol",
                            inject=True,
                            fix=node_text(src, n).decode("utf-8", "replace")))
        elif n.type == "call" and S.callee_name(src, n) == "paste0":
            args = _call_args(n)
            names = [_arg_name(src, a) for a in args]
            if "collapse" in names or not args:
                continue
            if any(nm == "sep" for nm in names):
                continue
            txt = node_text(src, n).decode("utf-8", "replace")
            m = re.fullmatch(r"paste0\((.*)\)", txt, re.S)
            if not m:
                continue
            out.append(dict(rule="paste0_sep", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte, old=txt,
                            new=f'paste({m.group(1)}, sep = "")',
                            rationale="injected paste(..., sep = \"\")",
                            inject=True, fix=txt))
        elif n.type == "call" and S.callee_name(src, n) == "vapply":
            args = _call_args(n)
            if len(args) != 3:
                continue
            fv = node_text(src, V._argument_value(args[2]) or args[2]) \
                .decode("utf-8", "replace").strip()
            if fv not in ("logical(1)", "numeric(1)", "integer(1)",
                          "character(1)"):
                continue
            x_txt = node_text(src, V._argument_value(args[0]) or args[0]) \
                .decode("utf-8", "replace")
            lam_txt = node_text(src, V._argument_value(args[1]) or args[1]) \
                .decode("utf-8", "replace")
            txt = node_text(src, n).decode("utf-8", "replace")
            out.append(dict(rule="sapply_vapply", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte, old=txt,
                            new=f"sapply({x_txt}, {lam_txt})",
                            rationale="injected sapply (type-unsafe)",
                            inject=True, fix=txt))
        elif n.type == "call" and S.callee_name(src, n) == "seq_along":
            args = _call_args(n)
            if len(args) != 1:
                continue
            arg_txt = node_text(src, V._argument_value(args[0]) or args[0]) \
                .decode("utf-8", "replace")
            txt = node_text(src, n).decode("utf-8", "replace")
            out.append(dict(rule="seq_safety", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.start_byte, eb=n.end_byte, old=txt,
                            new=f"1:length({arg_txt})",
                            rationale="injected 1:length(x)",
                            inject=True, fix=txt))
    return out




# --- fix-issue direction: deterministic BUG injection (corpus-exact GT) ----

def detect_bug_injectable(bs: BaseSample) -> list[dict]:
    """INVERSE rule direction for the FIX-ISSUE mode: inject a small
    hard-to-spot defect into corpus-clean code; the target is the ORIGINAL
    (tier-1 exact). Rules (wave-1 set):
      boundary_operator  <= -> < (and >= -> >) in for/if/while conditions
      char_swap          one occurrence of a declared local becomes a
                         transposed-char typo that collides with nothing
      wrong_variable     an occurrence of a local is swapped for a
                         similar-named sibling local (shared >=3-char prefix)
    All same-line single splices; every finding carries fix=original text."""
    src, b = bs.b.src, bs.b
    out = []

    def is_id(txt: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z.][\w.]*", txt))

    for n in V._walk(bs.body):
        # (b1) boundary-operator mutation inside loop/if conditions
        if n.type == "binary_operator" and len(n.children) >= 3 \
                and node_text(src, n.children[1]) in (b"<=", b">="):
            # in-condition iff the ancestor chain from the operator to the
            # statement passes through NO braced body (tree-sitter-r exposes
            # the condition as a bare child, parens are punctuation)
            anc, in_cond, saw_body = n.parent, False, False
            while anc is not None and anc is not bs.body:
                if anc.type == "braced_expression":
                    saw_body = True
                if anc.type in ("if_statement", "for_statement",
                                "while_statement"):
                    in_cond = not saw_body
                    break
                anc = anc.parent
            if not in_cond:
                continue
            op = node_text(src, n.children[1]).decode()
            new = "<" if op == "<=" else ">"
            out.append(dict(rule="boundary_operator", row=n.start_point[0],
                            erow=n.end_point[0],
                            sb=n.children[1].start_byte,
                            eb=n.children[1].end_byte, old=op, new=new,
                            rationale="injected off-by-one boundary mutation",
                            inject=True, fix=op))
        # (b2/b3) identifier-level defects on declared locals
        if n.type != "binary_operator" or len(n.children) < 3 \
                or node_text(src, n.children[1]) != b"<-":
            continue
        lhs = n.children[0]
        if lhs.type != "identifier":
            continue
        name = node_text(src, lhs).decode("utf-8", "replace")
        if not is_id(name) or len(name) < 3:
            continue
        # candidate sibling locals: other declared names sharing a 3-char+
        # prefix (wrong_variable) and non-colliding typo twins (char_swap)
        siblings = set()
        for m in V._walk(bs.body):
            if m.type == "binary_operator" and len(m.children) >= 3 \
                    and node_text(src, m.children[1]) == b"<-" \
                    and m.children[0].type == "identifier":
                other = node_text(src, m.children[0]).decode("utf-8", "replace")
                if other != name and is_id(other) and len(other) >= 2:
                    pre = next((k for k in range(min(len(name), len(other)), 2, -1)
                                if name[:k] == other[:k]), 0)
                    if pre >= 3:
                        siblings.add(other)
        body_txt = S.strip_strings(node_text(src, bs.body)).decode(
            "utf-8", "replace")
        occ = [m for m in re.finditer(
            rf"(?<![\w.]){re.escape(name)}(?![\w.])", body_txt)]
        # ONE read occurrence after the assignment becomes the defect site
        for m in occ:
            byte0 = bs.body.start_byte + m.start()
            row = b.rowcol(byte0)[0]
            if row == n.start_point[0]:
                continue                      # the declaration line itself
            if len(out) >= 6:
                break
            if siblings and len(name) >= 4:
                sib = sorted(siblings)[0]
                out.append(dict(rule="wrong_variable", row=row, erow=row,
                                sb=byte0, eb=byte0 + len(name), old=name,
                                new=sib,
                                rationale=f"injected wrong variable ({sib})",
                                inject=True, fix=name))
                break
            sw = name[0] + name[2] + name[1] + name[3:] if len(name) > 3 \
                else name[::-1]
            if sw != name and sw not in b.id_names and is_id(sw):
                out.append(dict(rule="char_swap", row=row, erow=row,
                                sb=byte0, eb=byte0 + len(name), old=name,
                                new=sw,
                                rationale="injected transposed-char typo",
                                inject=True, fix=name))
                break
    return out


# --- Rscript behavior gate -------------------------------------------------

HEURISTIC_ARGS = {
    "x": "c(3, 1, 2)", "xs": "c(3, 1, 2)", "vec": "c(3, 1, 2)",
    "v": "c(3, 1, 2)", "values": "c(3, 1, 2)", "input": "c(3, 1, 2)",
    "i": "2L", "j": "1L", "k": "1L", "n": "3L", "idx": "2L", "index": "2L",
    "s": 'c("a", "b")', "str": 'c("a", "b")', "string": 'c("a", "b")',
    "txt": 'c("a", "b")', "labels": 'c("a", "b")',
    "df": "data.frame(a = 1:3, b = c(1.5, 2.5, 3.5))",
    "data": "data.frame(a = 1:3, b = c(1.5, 2.5, 3.5))",
    "dat": "data.frame(a = 1:3, b = c(1.5, 2.5, 3.5))",
    "na_rm": "FALSE", "na.rm": "FALSE", "verbose": "FALSE",
    "sep": '"_"', "collapse": '""', "prefix": '"p_"',
    "y": "c(1, 2, 3)", "text": 'c("a", "b")', "name": '"n1"',
    "w": "c(1, 1, 1)", "col": '"a"', "mu": "1.5", "sigma": "1.5",
}


def fn_signature_callable(bs: BaseSample) -> str | None:
    """'name(arg, ...)' when every required formal has a heuristic arg or a
    default; None when not callable with simple inputs."""
    src = bs.b.src
    parent = bs.fn.parent
    name = ""
    if parent is not None and parent.type == "binary_operator" and parent.children:
        name = node_text(src, parent.children[0]).decode("utf-8", "replace")
    params = next((c for c in bs.fn.children if c.type == "parameters"), None)
    call_args = []
    if params is not None:
        for p in (c for c in params.children if c.is_named):
            if p.type != "parameter":
                return None
            kids = list(p.children)
            if any(c.type == "=" for c in kids):
                continue                          # formal with a default
            txt = node_text(src, p).decode("utf-8", "replace").strip()
            if txt == "...":
                continue                          # absorbs zero extra args
            if txt not in HEURISTIC_ARGS:
                return None
            call_args.append(HEURISTIC_ARGS[txt])
    if not re.fullmatch(r"[A-Za-z.][\w.]*", name or ""):
        return None
    return f"{name}({', '.join(call_args)})"


def behavior_probe(fn_text: str, call_txt: str, timeout: int = 10) -> str | None:
    """Stable stdout signature of calling the function; None on infra failure."""
    script = (fn_text + "\nres <- tryCatch(" + call_txt +
              ", error = function(e) paste0('ERROR: ', class(e)[1], ': ', "
              "conditionMessage(e)))\n"
              "cat(paste(capture.output(str(res)), collapse = '\\n'))\n"
              "cat('\\n;;;')\ncat(is.null(res))\n")
    try:
        r = subprocess.run(["Rscript", "-e", script], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if r.returncode == 0 else None


def check_rewrite_row(row: dict) -> tuple[bool, str]:
    """The intended modernization_rewrite row_check (a registered version
    would live in validators.py): region_old/region_new are the flagged
    block's original/fixed line lists (1..8 lines), both non-blank, they
    differ, the target is not longer, and the geometry pins the function
    (closing brace visible after the site). Structural parse of the target
    happens at the function level (gates parse/splice), not per-line."""
    for f in ("prefix", "region_old", "region_new", "suffix"):
        if not isinstance(row.get(f), list) or not row[f]:
            return False, f"{f} missing/empty"
    if not 1 <= len(row["region_old"]) <= 8:
        return False, "region_old must be 1..8 lines"
    if not 1 <= len(row["region_new"]) <= 8:
        return False, "region_new must be 1..8 lines"
    if row["region_old"] == row["region_new"]:
        return False, "no-op rewrite row"
    if not any(l.strip() for l in row["region_new"]):
        return False, "region_new is blank"
    if row["cursor_idx"] != 0:
        return False, "cursor must sit at the changed line start"
    if len([l for l in row["region_new"] if l.strip()]) > \
            len([l for l in row["region_old"] if l.strip()]):
        return False, "rewrite target is longer than the flagged block"
    if not any("}" in l for l in row["suffix"]) and \
            "}" not in row["region_new"][-1]:
        return False, "function closing brace not visible after the site"
    return True, ""



def drop_overlaps(findings: list[dict]) -> list[dict]:
    """Keep the outermost of any overlapping spans (nested paste-in-paste
    etc.): descending byte-splice with stale offsets corrupts the inner
    site, so a constraint spec must carry only non-overlapping findings."""
    keep, last_end = [], -1
    for f in sorted(findings, key=lambda f: (f["sb"], -(f["eb"] - f["sb"]))):
        if f["sb"] >= last_end:
            keep.append(f)
            last_end = f["eb"]
    return keep


def cmd_constraint(args) -> int:
    rng = random.Random(args.seed)
    pool = sample_packages(rng, args.tidy_packages, args.random_packages)
    JARL_TMP.mkdir(parents=True, exist_ok=True)
    specs, rows_out = [], []
    stats = dict(functions_scanned=0, specs_emitted=0, findings_total=0,
                 per_rule=Counter(), per_arm=Counter(),
                 gate=Counter(), spec_outcomes=[], failures=[])
    quota = dict(mine=args.functions, inject=args.inject_functions,
                 buinject=args.bug_functions)
    t0 = time.time()
    for b in C.iter_bundles_highest(pool, rng):
        if all(v <= 0 for v in quota.values()) or \
                time.time() - t0 > args.time_budget:
            break
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            if all(v <= 0 for v in quota.values()):
                break
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            _body, _head, _r0, _r1, nb = geom
            if not 6 <= len(nb) <= 60:
                continue
            stats["functions_scanned"] += 1
            try:
                bs = BaseSample(b, fn, stats["specs_emitted"])
            except ValueError:
                continue
            # arm 1: MINE real lint findings in the corpus text
            # (span cap 8 lines = the family's region convention)
            if quota["mine"] > 0:
                findings = drop_overlaps([
                    f for f in detect_findings(bs)
                    if b.line_str(f["row"]).strip()
                    and f.get("erow", f["row"]) - f["row"] < 8])
                if findings and run_case(bs, findings, "mine", specs,
                                         rows_out, stats, quota):
                    quota["mine"] -= 1
                    continue
            # arm 2: INJECT a dirty twin at a corpus-clean site (reverse
            # strip: the corpus original is the tier-1 exact GT)
            if quota["inject"] > 0:
                findings = drop_overlaps([
                    f for f in detect_injectable(bs)
                    if b.line_str(f["row"]).strip()
                    and f.get("erow", f["row"]) - f["row"] < 8])
                if findings and run_case(bs, findings, "inject", specs,
                                         rows_out, stats, quota):
                    quota["inject"] -= 1
            # arm 3: BUG injection (fix-issue direction; corpus-exact GT;
            # behavior-difference is the POINT, so the behavior gate is n/a)
            if quota["buinject"] > 0:
                one = drop_overlaps([
                    f for f in detect_bug_injectable(bs)
                    if b.line_str(f["row"]).strip()
                    and f.get("erow", f["row"]) - f["row"] < 8])
                if one and run_case(bs, one[:1], "buinject", specs,
                                    rows_out, stats, quota):
                    quota["buinject"] -= 1

    n_mine = stats["per_arm"]["mine"]
    n_inj = stats["per_arm"]["inject"]
    stats["gate"].setdefault("buinject.gt_corpus_exact", 0)
    report = dict(seed=args.seed, elapsed_s=round(time.time() - t0, 1),
                  functions_scanned=stats["functions_scanned"],
                  specs_emitted=stats["specs_emitted"],
                  findings_total=stats["findings_total"],
                  per_rule=dict(stats["per_rule"].most_common()),
                  per_arm=dict(stats["per_arm"]),
                  gates=dict(stats["gate"]),
                  gate_pass_rates={},
                  spec_outcomes=stats["spec_outcomes"],
                  failure_examples=stats["failures"][:40])
    n_bu = stats["per_arm"]["buinject"]
    for arm, n in (("mine", n_mine), ("inject", n_inj), ("buinject", n_bu)):
        for k in ("parse", "splice", "diff_minimal", "loc", "lint_delta",
                  "jarl_agreement", "row_check", "behavior_passed",
                  "behavior_failed", "behavior_skipped", "gt_corpus_exact"):
            report["gate_pass_rates"][f"{arm}.{k}"] = round(
                stats["gate"][f"{arm}.{k}"] / max(1, n), 3)
        att = stats["gate"][f"{arm}.behavior_passed"] \
            + stats["gate"][f"{arm}.behavior_failed"]
        report["gate_pass_rates"][f"{arm}.behavior_attempted"] = att
        if att:
            report["gate_pass_rates"][f"{arm}.behavior_pass_given_attempted"] \
                = round(stats["gate"][f"{arm}.behavior_passed"] / att, 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "constraint_specs.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in specs) + "\n")
    (OUT_DIR / "rewrite_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n")
    (OUT_DIR / "gates.json").write_text(json.dumps(report, indent=1))
    print(f"[constraint] scanned {stats['functions_scanned']} fns; "
          f"specs: mine={n_mine} inject={n_inj} "
          f"({stats['findings_total']} findings) in {report['elapsed_s']}s")
    print(f"  per_rule: {report['per_rule']}")
    print(f"  gates: {json.dumps(report['gate_pass_rates'])}")
    return 0


def _span_delta(findings: list[dict], upto: int) -> int:
    """Byte-length change caused by splices starting before `upto`."""
    return sum(len(f["new"].encode()) - (f["eb"] - f["sb"])
               for f in findings if f["sb"] < upto)


def _cover_rows(findings: list[dict], top: int) -> set[int]:
    """Function-local line indexes a finding may touch (over-approximation
    covering both the old span and the replacement's line count)."""
    rows: set[int] = set()
    for f in findings:
        n_old = f["old"].count("\n") + 1
        n_new = f["new"].count("\n") + 1
        r0 = f["row"] - top
        rows.update(range(r0, r0 + max(n_old, n_new)))
    return rows


def base_sample_id(bs: BaseSample) -> str:
    """Stable content-hash id of the normalized base (the compounding
    design's bs:<sha1 of code+origin> convention) — NOT a counter, so
    derived rows link to the same parent across runs and the eval holdout
    splits at the BASE-SAMPLE level (all rows of one base in one split)."""
    import hashlib
    origin = f"{bs.b.package}|{bs.b.rel}"
    code = fn_full_text(bs)
    return "bs:" + hashlib.sha1(
        (origin + "\x00" + code).encode("utf-8", "surrogateescape")
    ).hexdigest()[:16]


def run_case(bs: BaseSample, findings: list[dict], arm: str, specs: list,
             rows_out: list, stats: dict, quota: dict) -> bool:
    b = bs.b
    src = b.src
    bs_base_id = base_sample_id(bs)
    start = bs.fn.parent.start_byte if (bs.fn.parent is not None
                                        and bs.fn.parent.type == "binary_operator") \
        else bs.fn.start_byte
    end = bs.fn.end_byte
    if arm == "mine":
        dirty_bytes = src                                   # corpus as-is
        fixed_bytes = apply_fixes(src, findings)            # mock author
    else:
        dirty_bytes = apply_fixes(src, findings)            # new = dirty text
        fixed_bytes = _restore(dirty_bytes, findings)       # mock author
    if arm == "mine":
        # corpus as-is vs mock-authored fix (fix changes length -> adjust end)
        before_full = src[start:end].decode("utf-8", "replace")
        after_full = fixed_bytes[start:end + _span_delta(findings, end)] \
            .decode("utf-8", "replace")
    else:
        # injected dirty variant vs its corpus-exact restoration
        before_full = dirty_bytes[start:end + _span_delta(findings, end)] \
            .decode("utf-8", "replace")
        after_full = fixed_bytes[start:end].decode("utf-8", "replace")
    corpus_full = src[start:end].decode("utf-8", "replace")

    stats["specs_emitted"] += 1
    stats["per_arm"][arm] += 1
    stats["findings_total"] += len(findings)
    for f in findings:
        stats["per_rule"][f["rule"]] += 1
    spec = dict(
        id=f"rw:{arm}:{bs_base_id}", arm=arm, base_sample_id=bs_base_id,
        package=b.package, path=b.rel,
        fn_head=b.line_str(bs.head_row), rows=[bs.top_row, bs.r1],
        constraint=("rewrite the block below fixing exactly the listed "
                    "findings; touch nothing else; do not reformat; the "
                    "result must not be longer than the input"),
        findings=[dict(rule=f["rule"], row=f["row"], col=b.rowcol(f["sb"])[1],
                       snippet=f["old"], rationale=f["rationale"])
                  for f in findings],
        generated_at=GENERATED_AT)
    specs.append(spec)
    out = dict(id=spec["id"], package=b.package, arm=arm,
               rules=[f["rule"] for f in findings], gates={})
    G = stats["gate"]

    # G1 parse: tree-sitter fragment + Rscript parse of the fixed function
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     dir=str(JARL_TMP)) as fh:
        fh.write(after_full)
        after_path = Path(fh.name)
    try:
        r = subprocess.run(["Rscript", "-e",
                            f"invisible(parse('{after_path}'))"],
                           capture_output=True, text=True, timeout=30)
        ok = V.fragment_clean(after_full) and r.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    G[f"{arm}.parse"] += int(ok)
    out["gates"]["parse"] = bool(ok)

    # G2 splice: the whole FILE re-parses with the fixed function in place
    tb = S.parser.parse(fixed_bytes)
    ok = not tb.root_node.has_error and \
        not any(n.type == "ERROR" or n.is_missing
                for n in V._walk(tb.root_node))
    G[f"{arm}.splice"] += int(ok)
    out["gates"]["splice"] = bool(ok)

    # G3 diff minimality: changed lines are within the finding covers
    before_lines = before_full.split("\n")
    after_lines = after_full.split("\n")
    sm = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    changed = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "equal":
            changed.update(range(i1, max(i1 + 1, i2)))
    cover = _cover_rows(findings, bs.top_row)
    ok = changed <= cover
    G[f"{arm}.diff_minimal"] += int(ok)
    out["gates"]["diff_minimal"] = bool(ok)
    if not ok:
        out["diff_changed_outside"] = sorted(changed - cover)[:6]

    # G4 LOC delta: never longer after the rewrite
    nb_b = sum(1 for l in before_lines if l.strip())
    nb_a = sum(1 for l in after_lines if l.strip())
    ok = nb_a <= nb_b
    G[f"{arm}.loc"] += int(ok)
    out["gates"]["loc"] = bool(ok)
    out["loc_delta"] = nb_a - nb_b

    # G5/G6 jarl lint-delta + detector agreement (local tmp copies)
    bf = JARL_TMP / f"before_{arm}_{bs.site_id}.R"
    af = JARL_TMP / f"after_{arm}_{bs.site_id}.R"
    bf.write_text(before_full)
    af.write_text(after_full)
    diags_b = jarl_json([bf])
    diags_a = jarl_json([af])
    keys_b = {(d["message"]["name"], d["location"]["row"]) for d in diags_b}
    keys_a = {(d["message"]["name"], d["location"]["row"]) for d in diags_a}
    ALIAS = {"seq_safety": "seq", "class_equals": "class_equals",
             "true_false_symbol": "true_false_symbol",
             "paste0_sep": None, "sapply_vapply": None,
             "boundary_operator": None, "char_swap": None,
             "wrong_variable": None}
    agree = 0
    for f in findings:
        rule = ALIAS.get(f["rule"], f["rule"])
        if rule is None:
            continue                       # no jarl counterpart rule
        local_row = f["row"] - bs.top_row + 1
        if (rule, local_row) in keys_b:
            agree += 1
    checkable = [f for f in findings if ALIAS.get(f["rule"], f["rule"])]
    ok = (agree == len(checkable)) if checkable else True
    G[f"{arm}.jarl_agreement"] += int(ok)
    out["gates"]["jarl_agreement"] = bool(ok)
    out["jarl_agreement_n"] = f"{agree}/{len(checkable)}"
    targeted = {(ALIAS.get(f["rule"], f["rule"]),
                 f["row"] - bs.top_row + 1) for f in checkable}
    ok = targeted.isdisjoint(keys_a) and (keys_a - keys_b == set())
    G[f"{arm}.lint_delta"] += int(ok)
    out["gates"]["lint_delta"] = bool(ok)
    out["lint_new"] = sorted(f"{r}@{row}" for r, row in (keys_a - keys_b))[:5]

    # G7 (inject arms only): GT is the corpus original, byte-exact
    if arm in ("inject", "buinject"):
        ok = after_full == corpus_full
        G[f"{arm}.gt_corpus_exact"] += int(ok)
        out["gates"]["gt_corpus_exact"] = bool(ok)

    # G8 behavior preservation (Rscript, simple inputs where callable).
    # n/a for the bug arm: behavior-difference is the point of an injected
    # defect (the GT is the corpus original, verified byte-exact by G7).
    if arm == "buinject":
        G[f"{arm}.behavior_skipped"] += 1
        out["gates"]["behavior"] = "skipped(n/a: injected bug)"
    elif (call_txt := fn_signature_callable(bs)) is None:
        G[f"{arm}.behavior_skipped"] += 1
        out["gates"]["behavior"] = "skipped(not-callable)"
    else:
        out_b = behavior_probe(before_full, call_txt)
        out_a = behavior_probe(after_full, call_txt)
        if out_b is None or out_a is None:
            G[f"{arm}.behavior_skipped"] += 1
            out["gates"]["behavior"] = "skipped(timeout/infra)"
        elif out_b == out_a:
            G[f"{arm}.behavior_passed"] += 1
            out["gates"]["behavior"] = "passed"
        else:
            out_b2 = behavior_probe(before_full, call_txt)
            if out_b2 != out_b:
                G[f"{arm}.behavior_skipped"] += 1
                out["gates"]["behavior"] = "skipped(nondeterministic)"
            else:
                G[f"{arm}.behavior_failed"] += 1
                out["gates"]["behavior"] = "FAILED"
                out["behavior_diff"] = [out_b[:160], out_a[:160]]
                stats["failures"].append(
                    dict(id=spec["id"], kind="behavior",
                         rules=out["rules"], package=b.package,
                         detail=[out_b[:160], out_a[:160]]))

    # emit rows: ONE modernization_rewrite row per finding (edit-pair schema;
    # region_old/region_new may span the finding's full line range)
    row_ok_all = True
    for f in findings:
        f_row0, f_erow = f["row"], f.get("erow", f["row"])
        line_delta = f["new"].count("\n") - f["old"].count("\n")
        corpus_lines = [b.line_str(r) for r in range(f_row0, f_erow + 1)]
        if arm == "mine":
            one = apply_fixes(src, [f])
            fixed_lines = [one.decode("utf-8", "replace").split("\n")[r]
                           for r in range(f_row0, f_erow + 1 + line_delta)]
            old_lines, new_lines = corpus_lines, fixed_lines
        else:
            dirty_src = apply_fixes(src, [f])
            dirty_lines = [dirty_src.decode("utf-8", "replace").split("\n")[r]
                           for r in range(f_row0, f_erow + 1 + line_delta)]
            old_lines, new_lines = dirty_lines, corpus_lines
        prefix = [b.line_str(r) for r in
                  range(max(0, bs.head_row - 8), f_row0)]
        suffix = [b.line_str(r) for r in
                  range(f_erow + 1 + max(0, line_delta),
                        min(b.nlines(), bs.r1 + 1 + 8))]
        rowd = dict(family="modernization_rewrite",
                    transform=f"fix_{f['rule']}", arm=arm,
                    transform_id=f"modernization_rewrite/{f['rule']}@1",
                    base_sample_id=bs_base_id, package=b.package, path=b.rel,
                    row=f_row0,
                    prefix=prefix or [""], region_old=old_lines,
                    region_new=new_lines, suffix=suffix, cursor_idx=0,
                    event_diff="",
                    note=(f"rewrite the flagged line(s): {f['rationale']} "
                          f"(fix exactly this finding, touch nothing else)"),
                    case="rewrite_maint_proto", backend="deterministic",
                    model="mock-author", full_prompt="",
                    generated_at=GENERATED_AT, corpus_target="\n".join(new_lines),
                    constraint_spec=spec["id"],
                    base_sample=bs_base_id,
                    determinism=("static+validator" if arm == "mine"
                                 else "pure-static (corpus-exact GT)"),
                    gates={k: out["gates"].get(k)
                           for k in ("parse", "splice", "diff_minimal",
                                     "lint_delta")})
        ok, _r = check_rewrite_row(rowd)
        row_ok_all = row_ok_all and ok
        rows_out.append(rowd)
    G[f"{arm}.row_check"] += int(row_ok_all)
    out["gates"]["row_check"] = bool(row_ok_all)
    stats["spec_outcomes"].append(out)
    return True


def _restore(dirty: bytes, findings: list[dict]) -> bytes:
    """Inject-arm mock author: restore every injected span to the corpus
    original. Processed in ASCENDING byte order with a cumulative delta:
    restoring a lower span shifts every higher span, so each restore writes
    at (original sb + accumulated growth)."""
    # each finding's injected text sits in dirty at (orig sb + net shift of
    # all LOWER injections); splice descending so no processed span moves
    def dirty_pos(f):
        shift = sum(len(g["new"].encode()) - (g["eb"] - g["sb"])
                    for g in findings if g["sb"] < f["sb"])
        return f["sb"] + shift
    order = sorted(findings, key=lambda f: -dirty_pos(f))
    out = dirty
    for f in order:
        sb = dirty_pos(f)
        out = out[:sb] + f["fix"].encode() + \
            out[sb + len(f["new"].encode()):]
    return out



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 rewrite_maint_proto.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lint-census")
    p.add_argument("--files", type=int, default=150)
    p.add_argument("--tidy-packages", type=int, default=45)
    p.add_argument("--random-packages", type=int, default=60)
    p.add_argument("--seed", type=int, default=29)
    p.set_defaults(fn=cmd_lint_census)

    p = sub.add_parser("smells")
    p.add_argument("--functions", type=int, default=15000)
    p.add_argument("--tidy-packages", type=int, default=50)
    p.add_argument("--random-packages", type=int, default=120)
    p.add_argument("--time-budget", type=float, default=600)
    p.add_argument("--seed", type=int, default=31)
    p.set_defaults(fn=cmd_smells)

    p = sub.add_parser("edit-pairs")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=37)
    p.set_defaults(fn=cmd_edit_pairs)

    p = sub.add_parser("constraint")
    p.add_argument("--functions", type=int, default=50)
    p.add_argument("--inject-functions", type=int, default=40)
    p.add_argument("--bug-functions", type=int, default=25)
    p.add_argument("--tidy-packages", type=int, default=60)
    p.add_argument("--random-packages", type=int, default=90)
    p.add_argument("--time-budget", type=float, default=900)
    p.add_argument("--seed", type=int, default=41)
    p.set_defaults(fn=cmd_constraint)

    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
