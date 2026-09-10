"""Tests for experiments/data-mining/build_tu3.py (TU3 cross-file pack).

Pure-CPU, no network, no NAS: the corpus is a fixtures-only normalized-
mirror shape built under tmp_path (the build_smoke_corpus packages plus
holdout/no-derivatives extras). Covers the cross-file miner, every
family validator (positive + corruption cases), the corrupted-twin
generator and its gate-rejection rule, the S5 budget math (ref <=120
lines / 4000 chars, prefix <=80 / suffix <=40, 16,000-char pack budget,
2-5 files), license handling (field propagation + no-derivatives
exclusion), and the --smoke end-to-end run with the mock backend.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / \
    "experiments" / "data-mining" / "build_tu3.py"
_spec = importlib.util.spec_from_file_location("build_tu3", _SCRIPT)
tu3 = importlib.util.module_from_spec(_spec)
sys.modules["build_tu3"] = tu3
_spec.loader.exec_module(tu3)

_HOLDOUT_JSON = "/nonexistent/holdout.json"


def _holdout_name(prefix: str) -> str:
    n = prefix
    while not tu3.is_holdout(n):
        n += "x"
    return n


def _free_name(prefix: str = "pkg") -> str:
    n = prefix
    while tu3.is_holdout(n):
        n += "y"
    return n


def _mine_args(corpus: Path, work: Path, **kw) -> SimpleNamespace:
    base = dict(corpus=str(corpus), work=str(work), holdout_json=_HOLDOUT_JSON,
                seeds=12, pool_mult=1.0, seed=42, per_package_cap=3,
                max_packages=0, exclude_no_derivatives=True)
    base.update(kw)
    return SimpleNamespace(**base)


def _fixture_corpus(tmp_path: Path) -> Path:
    """Smoke packages + a holdout package + an extra no-derivatives one."""
    corpus = tmp_path / "normalized"
    tu3.build_smoke_corpus(corpus)
    hold = _holdout_name("holdpkg")
    hp = corpus / hold / "1.0" / hold / "R"
    hp.mkdir(parents=True)
    (corpus / hold / "1.0" / hold / "DESCRIPTION").write_text(
        f"Package: {hold}\nVersion: 1.0\nLicense: GPL-3\n")
    (hp / "a.R").write_text("h_fn <- function(x) { x + 1 }\n")
    (hp / "b.R").write_text("h_res <- h_fn(2)\n")
    nd = _free_name("ndpkg")
    np_ = corpus / nd / "1.0" / nd / "R"
    np_.mkdir(parents=True)
    (corpus / nd / "1.0" / nd / "DESCRIPTION").write_text(
        f"Package: {nd}\nVersion: 1.0\nLicense: CC BY-NC-ND 4.0\n")
    (np_ / "d.R").write_text("nd_fn2 <- function(x) { x * 2 }\n")
    (np_ / "u.R").write_text("nd_res2 <- nd_fn2(3)\n")
    return corpus


def _mined_seeds(tmp_path: Path) -> list[dict]:
    corpus = _fixture_corpus(tmp_path)
    work = tmp_path / "work"
    tu3.cmd_mine(_mine_args(corpus, work))
    return [json.loads(l) for l in
            (work / "seeds.jsonl").read_text().splitlines() if l.strip()]


def _seed_of(seeds: list[dict], family: str) -> dict:
    return next(s for s in seeds if s["family"] == family)


# ------------------------------------------------------------- budget math

def test_family_targets_proportional_to_caps():
    t = tu3.family_targets(500)
    assert t == {"rename_function_xfile": 175, "arg_rename_xfile": 125,
                 "doc_sync_to_reference": 125, "port_with_reference": 75}
    assert sum(t.values()) == 500
    for fam, cap in tu3.FAMILY_CAPS.items():
        assert t[fam] <= cap
    assert tu3.family_targets(500) == tu3.family_targets(500)  # stable


def test_trim_reference_whole_file_when_it_fits():
    lines = [f"line{i}" for i in range(50)]
    shown, anchor = tu3.trim_reference(lines, 10)
    assert shown == lines and anchor == 10


def test_trim_reference_window_and_caps():
    lines = [f"line{i}" for i in range(400)]
    shown, anchor = tu3.trim_reference(lines, 200)
    assert len(shown) <= tu3.REF_MAX_LINES
    assert len("\n".join(shown)) <= tu3.REF_MAX_CHARS
    assert 0 <= anchor < len(shown)      # anchor row survives the trim
    # char cap forces trimming even under the line cap
    fat = ["x <- " + "a" * 900 for _ in range(10)]
    shown2, _ = tu3.trim_reference(fat, 5)
    assert len("\n".join(shown2)) <= tu3.REF_MAX_CHARS


def test_trim_target_statement_aligned_and_parseable():
    body = ["f1 <- function(x) {", "  x + 1", "}", ""]
    big = [f"v{i} <- {i}" for i in range(200)]
    lines = body + ["tgt <- fn(1)"] + big
    tree, _ = tu3.parse_src("\n".join(lines))
    prefix, suffix = tu3.trim_target(lines, tree, 4, 4)
    assert len(prefix) <= tu3.TARGET_PREFIX_LINES
    assert len(suffix) <= tu3.TARGET_SUFFIX_LINES
    window = "\n".join(prefix + ["tgt <- fn(1)"] + suffix)
    assert tu3.fragment_clean(window)    # never cuts mid-statement


def test_edit_row_cross_budget_and_markers():
    ex = dict(path="R/b.R", prefix=["p <- 1"], suffix=["s <- 2"],
              region_old=["r <- fn(x)"], region_new=["r <- fn2(x)"],
              cursor_idx=0, event_diff="@@ e @@\n-a\n+b",
              references=[dict(path="R/a.R",
                               render_lines=["fn2 <- function(x) x"])])
    row, why = tu3.edit_row_cross(ex, "rename_function_xfile", "pkg")
    assert row is not None, why
    assert row["text"] == row["prompt"] + row["target"]
    assert row["target"].endswith(tu3.UPDATED.lstrip("\n"))
    for marker in ("<[fim-suffix]>", "<[fim-prefix]>",
                   "<filename>edit_history", "<filename>R/a.R",
                   "<filename>R/b.R", "<<<<<<< CURRENT", "=======",
                   "<[fim-middle]>", tu3.CURSOR2, "-a", "+b"):
        assert marker in row["prompt"], marker
    big = ["z <- " + "a" * 200] * 90
    row2, why2 = tu3.edit_row_cross(dict(ex, region_new=big), "f", "p")
    assert row2 is None and why2 == f"over_{tu3.MAX_CHARS_CROSS}"


# ------------------------------------------------------------------ miner

def test_mine_fixture_yields_all_families_with_records(tmp_path):
    seeds = _mined_seeds(tmp_path)
    fams = {s["family"] for s in seeds}
    assert fams == set(tu3.FAMILIES)          # all four mined from fixtures
    for s in seeds:
        assert s["parent_link"].endswith(f"@{tu3.RULE}")
        sha_a, sha_b = s["parent_link"].split("@", 1)[0].split("|")
        assert len(sha_a) == 64 and len(sha_b) == 64
        assert {sha_a, sha_b} == {s["references"][0]["file_sha"],
                                  s["target"]["file_sha"]}
        assert 2 <= 1 + len(s["references"]) <= 5      # S5 file budget
        assert len(s["target"]["region_old"]) <= tu3.MAX_BLOCK_LINES
        ref = s["references"][0]
        assert len(ref["render_lines"]) <= tu3.REF_MAX_LINES
        assert len("\n".join(ref["render_lines"])) <= tu3.REF_MAX_CHARS
        assert len(s["target"]["prefix"]) <= tu3.TARGET_PREFIX_LINES
        assert len(s["target"]["suffix"]) <= tu3.TARGET_SUFFIX_LINES
        roles = {e["metadata"]["role"] for e in s["evidence"]}
        assert roles == {"reference", "target"}
        for e in s["evidence"]:
            assert e["source_kind"] == "file" and e["content_identity"]
            assert set(e) >= {"source_kind", "content", "path",
                              "content_identity", "source_range"}
        assert s["difficulty"]["label"] in ("simple", "medium", "hard")
        assert tu3.fragment_clean("\n".join(
            s["target"]["prefix"] + s["target"]["region_old"]
            + s["target"]["suffix"]))


def test_mine_excludes_holdout_and_no_derivatives(tmp_path):
    seeds = _mined_seeds(tmp_path)
    pkgs = {s["package"] for s in seeds}
    assert not any(p.startswith("holdpkg") for p in pkgs)
    assert not any(p.startswith("ndpkg") for p in pkgs)
    assert all(not tu3.is_no_derivatives(s["license"]) for s in seeds)
    work = tmp_path / "work"
    lic = json.loads((work / "licenses.json").read_text())
    nd = [p for p, v in lic["packages"].items()
          if v["excluded_no_derivatives"]]
    assert any(p.startswith("ndpkg") for p in nd)   # recorded, excluded


def test_mine_include_no_derivatives_flags_rows(tmp_path):
    corpus = _fixture_corpus(tmp_path)
    work = tmp_path / "work2"
    tu3.cmd_mine(_mine_args(corpus, work, exclude_no_derivatives=False,
                            seeds=40))   # pool big enough to keep all cands
    seeds = [json.loads(l) for l in
             (work / "seeds.jsonl").read_text().splitlines() if l.strip()]
    nd = [s for s in seeds if s["package"].startswith("ndpkg")]
    assert nd and all(s["public_ok"] is False for s in nd)


def test_f1_seed_shape_and_event(tmp_path):
    seeds = _mined_seeds(tmp_path)
    s = _seed_of(seeds, "rename_function_xfile")
    m = s["meta"]
    assert m["new"] == m["old"] + "2" or m["new"].startswith(m["old"])
    assert f"-{m['old']}" in s["event_diff"]
    assert f"+{m['new']}" in s["event_diff"]
    ref = s["references"][0]
    assert any(m["new"] in l for l in ref["render_lines"])
    assert not any(tu3.word_re(m["old"]).search(l)
                   for l in ref["render_lines"])
    assert any(tu3.word_re(m["old"]).search(l)
               for l in s["target"]["region_old"])


def test_reference_excerpt_shows_event_only_in_render(tmp_path):
    """Reference immutability base: corpus_lines keep the original def, the
    render carries the event, and both are recorded per row."""
    seeds = _mined_seeds(tmp_path)
    s = _seed_of(seeds, "arg_rename_xfile")
    ref = s["references"][0]
    assert ref["corpus_lines"] != ref["render_lines"]    # event applied
    assert any(s["meta"]["old"] in l for l in ref["corpus_lines"])


def test_f4_reference_presence_gate(tmp_path):
    """No tidyverse reference file in the package -> no tidyverse port
    seed (the reference-presence check is a mining gate)."""
    seeds = _mined_seeds(tmp_path)
    f4 = [s for s in seeds if s["family"] == "port_with_reference"]
    assert f4
    for s in f4:
        d = s["meta"]["direction"]
        assert tu3.REF_IDIOM_RE[d](
            "\n".join(s["references"][0]["render_lines"]))


def test_difficulty_and_dedupe_stats(tmp_path):
    corpus = _fixture_corpus(tmp_path)
    work = tmp_path / "work"
    tu3.cmd_mine(_mine_args(corpus, work))
    st = json.loads((work / "seeds.jsonl.stats.json").read_text())
    assert st["stats"]["seeds_total"] > 0
    assert st["stats"]["difficulty"]
    assert set(st["family_targets"]) == set(tu3.FAMILIES)


# ---------------------------------------------------- validators (per family)

def _f1_seed():
    """Hand-built F1 seed (validator unit level, no corpus needed)."""
    return dict(
        family="rename_function_xfile",
        target=dict(prefix=["other <- 1"],
                    region_old=["res <- prep_data(df, weight = w)",
                                "lab <- paste0(\"a\", res)"],
                    suffix=["tail <- 2"], cursor_idx=0),
        references=[dict(path="R/a.R", render_lines=[])],
        meta=dict(old="prep_data", new="prep_data2", fn="prep_data",
                  n_args=2))


def test_f1_validator_positive():
    seed = _f1_seed()
    good = ["res <- prep_data2(df, weight = w)",
            "lab <- paste0(\"a\", res)"]
    ok, why = tu3.validate_region_new(good, seed)
    assert ok, why


def test_f1_validator_corruptions():
    seed = _f1_seed()
    cases = {
        "no_op_text": ["res <- prep_data(df, weight = w)",
                       'lab <- paste0("a", res)'],
        "old_symbol_still_present":
            ["res <- prep_data2(df, weight = w, z = prep_data(1))",
             'lab <- paste0("a", res)'],
        "new_symbol_missing": ["res <- other_fn(df, weight = w)",
                               'lab <- paste0("a", res)'],
        "other_identifiers_changed":
            ["res <- prep_data2(df2, weight = w)",
             'lab <- paste0("a", res)'],
        "structure_changed":
            ["res <- prep_data2(df, weight = w, extra = 1)",
             'lab <- paste0("a", res)'],
        "untouched_statement_modified":       # string-only edit, no ids
            ["res <- prep_data2(df, weight = w)",
             'lab <- paste0("b", res)'],
    }
    # adding a statement changes structure too: the structure gate fires
    # first (gate ordering is part of the contract)
    ok, why = tu3.validate_region_new(
        ["res <- prep_data2(df, weight = w)", 'lab <- paste0("a", res)',
         "added <- 9"], _f1_seed())
    assert not ok and why == "structure_changed"
    for want, rn in cases.items():
        ok, why = tu3.validate_region_new(rn, seed)
        assert not ok and why == want, (want, why)


def _f2_seed():
    return dict(
        family="arg_rename_xfile",
        target=dict(prefix=[],
                    region_old=["out <- fit(d, weight = k, scale = s)"],
                    suffix=[], cursor_idx=0),
        references=[dict(path="R/a.R", render_lines=[])],
        meta=dict(old="weight", new="wts", fn="fit", n_args=3))


def test_f2_validator_positive():
    ok, why = tu3.validate_region_new(
        ["out <- fit(d, wts = k, scale = s)"], _f2_seed())
    assert ok, why


def test_f2_validator_corruptions():
    seed = _f2_seed()
    cases = {
        "no_op_text": ["out <- fit(d, weight = k, scale = s)"],
        "old_arg_still_present":
            ["out <- fit(d, wts = k, scale = s, weight = q)"],
        "new_arg_missing": ["out <- fit(d, mass = k, scale = s)"],
        "callee_missing":                     # callee edited = F1 row
            ["out <- fit2(d, wts = k, scale = s)"],
        "other_identifiers_changed":          # unrelated identifier edited
            ["out2 <- fit(d, wts = k, scale = s)"],
        "named_args_reordered_or_positional": # same structure, wrong order
            ["out <- fit(d, scale = s, wts = k)"],
        "positional_count_changed":
            ["out <- fit(d, wts = k, scale = s, 2)"],
        "untouched_statement_modified": None,   # filled below
    }
    seed2 = dict(
        seed,
        target=dict(seed["target"],
                    region_old=["out <- fit(d, weight = k, scale = s)",
                                'side <- paste0("a", 1)'],
                    suffix=[]))
    cases["untouched_statement_modified"] = \
        ["out <- fit(d, wts = k, scale = s)",
         'side <- paste0("b", 1)']
    for want, rn in cases.items():
        s = seed2 if want == "untouched_statement_modified" else seed
        ok, why = tu3.validate_region_new(rn, s)
        assert not ok and why == want, (want, why)


def _f3_seed():
    region = ["#' Title", "#' @param x input", "#' @return numeric",
              "#' @export"]
    return dict(
        family="doc_sync_to_reference",
        target=dict(prefix=[], region_old=region, suffix=[], cursor_idx=1),
        references=[dict(path="R/u.R", render_lines=[])],
        meta=dict(fn="foo", p="scale", desc="supplies `scale`; see usage.",
                  anchor_idx=2, terminal_idx=2, old="scale", new="scale",
                  n_args=2))


def test_f3_validator_positive_at_anchor():
    seed = _f3_seed()
    good = seed["target"]["region_old"][:2] \
        + ["#' @param scale supplies `scale`; see usage."] \
        + seed["target"]["region_old"][2:]
    ok, why = tu3.validate_region_new(good, seed)
    assert ok, why


def test_f3_validator_corruptions():
    seed = _f3_seed()
    m = seed["meta"]
    good_line = f"#' @param {m['p']} {m['desc']}"
    cases = {
        "f3_line_mismatch": ["#' Title", "#' @param x input", good_line
                             + " altered", "#' @return numeric",
                             "#' @export"],
        "f3_anchor_violation": ["#' Title", good_line, "#' @param x input",
                                "#' @return numeric", "#' @export"],
        "f3_not_single_line_insertion": ["#' Title", "#' @param x input",
                                         good_line, good_line,
                                         "#' @return numeric", "#' @export"],
        "f3_other_lines_touched": ["#' Title2", "#' @param x input",
                                   good_line, "#' @return numeric",
                                   "#' @export"],
        "no_op_text": seed["target"]["region_old"],
    }
    for want, rn in cases.items():
        ok, why = tu3.validate_region_new(rn, seed)
        assert not ok and why == want, (want, why)


def _f4_seed(direction="vectorize"):
    region = ["loop_total <- function(xs) {", "  tot <- 0",
              "  for (i in seq_along(xs)) {", "    tot <- tot + xs[i]",
              "  }", "  tot", "}"]
    return dict(
        family="port_with_reference",
        target=dict(prefix=[], region_old=region, suffix=[], cursor_idx=2),
        references=[dict(path="R/ref.R",
                         render_lines=["m <- vapply(xs, identity, 1)"])],
        meta=dict(direction=direction, exempt_all=direction
                  in tu3.PORT_EXEMPT_ALL, idiom_lines=[2], n_args=0,
                  old=direction, new=direction))


def test_f4_validator_positive_vectorize():
    seed = _f4_seed("vectorize")
    good = ["loop_total <- function(xs) {", "  tot <- 0",
            "  tot <- vapply(xs, function(z) z, numeric(1))", "  tot", "}"]
    ok, why = tu3.validate_region_new(good, seed)
    assert ok, why


def test_f4_validator_corruptions():
    cases = {
        "idiom_vectorize_still_looping": [
            "loop_total <- function(xs) {", "  tot <- 0",
            "  for (i in xs) {", "    tot <- tot + xs[i]", "  }",
            "  tot", "}"],
        "idiom_tidyverse_missing": ["x <- 1"],
        "untouched_statement_modified": None,   # filled below
    }
    seed_u = dict(
        _f4_seed("vectorize"),
        target=dict(
            _f4_seed("vectorize")["target"],
            region_old=["side <- sum(xs)"] +
            _f4_seed("vectorize")["target"]["region_old"]))
    cases["untouched_statement_modified"] = \
        ["side <- mean(xs)"] + \
        ["loop_total <- function(xs) {", "  tot <- 0",
         "  tot <- vapply(xs, function(z) z, numeric(1))",
         "  tot", "}"]
    for want, rn in cases.items():
        seed = seed_u if want == "untouched_statement_modified" else (
            _f4_seed("tidyverse")
            if want == "idiom_tidyverse_missing" else _f4_seed("vectorize"))
        ok, why = tu3.validate_region_new(rn, seed)
        assert not ok and why == want, (want, why)
    # base_r contamination: pipes survive the port
    seed_b = _f4_seed("base_r")
    ok, why = tu3.validate_region_new(
        ["res <- dplyr::mutate(d, v = 1)"], seed_b)
    assert not ok and why == "idiom_base_r_contaminated"


def test_universal_gates():
    seed = _f1_seed()
    ok, why = tu3.validate_region_new(["this is << not R"], seed)
    assert not ok and why == "not_clean_r"
    ok, why = tu3.validate_region_new(["   "], seed)
    assert not ok and why == "empty_region_new"
    ok, why = tu3.validate_region_new(["x <- 1"] * 26, seed)
    assert not ok and why == f"over_{tu3.MAX_BLOCK_LINES}_lines"
    ok, why = tu3.validate_region_new(
        ["res <- prep_data2(df, weight = w) <filename>R/a.R"], seed)
    assert not ok and why == "edits_reference_file"
    # splice: parses alone, breaks the target file after splice
    seed_sp = dict(seed, target=dict(seed["target"], suffix=["}"]))
    ok, why = tu3.validate_region_new(
        ["res <- prep_data2(df, weight = w)",
         "lab <- paste0(\"a\", res)"], seed_sp)
    assert not ok and why == "splice_not_clean"
    # whitespace-only change: identifier-inclusive ast signature is
    # whitespace-blind -> the ast arm of the B11 triple fires
    ok, why = tu3.validate_region_new(
        ["  res <- prep_data(df, weight = w)",
         '  lab <- paste0("a", res)'], seed)
    assert not ok and why == "no_op_ast"
    # the normalized-text arm of the triple (operator/whitespace aware)
    assert tu3._norm_text("a  <-  1\nb") == tu3._norm_text("a <- 1\n b")
    assert tu3._norm_text("x <= y") != tu3._norm_text("x < y")


def test_no_op_ast_sees_renames():
    """B11's identifier-blind signature would call every rename a no-op;
    TU3's named signature must not (rename = real edit), while a pure
    reformat still no-ops."""
    assert tu3.struct_key_named("x <- foo(1)") != \
        tu3.struct_key_named("x <- foo2(1)")
    assert tu3.struct_key_named("x <- foo(1)") == \
        tu3.struct_key_named("x   <-  foo( 1 )")


# ------------------------------------------------------------ twin generator

def test_twins_two_kinds_gate_rejects_noop_answer(tmp_path):
    seeds = _mined_seeds(tmp_path)
    for fam in tu3.FAMILIES:
        s = _seed_of(seeds, fam)
        twins = tu3.build_twin_exs(s)
        assert len(twins) == 2, fam
        assert {t["kind"] for t in twins} == set(tu3.TWIN_KINDS)
        for t in twins:
            assert tu3.twin_gate_check(s, t), \
                f"{fam}/{t['kind']}: no-edit answer must FAIL the gate"
            tgt = t["target"]
            ex = dict(path=s["target"]["path"], prefix=tgt["prefix"],
                      suffix=tgt["suffix"], region_old=tgt["region_old"],
                      region_new=tgt["region_old"],
                      cursor_idx=tgt["cursor_idx"],
                      event_diff=t["event_diff"],
                      references=t["references"])
            row, why = tu3.edit_row_cross(
                ex, fam, s["package"], expected_noop=True,
                twin_kind=t["kind"])
            assert row is not None, why
            assert row["expected_noop"] is True
            assert row["target"].startswith("\n".join(tgt["region_old"])
                                            .rstrip()[:20])


def test_twin_renders_differ_from_original(tmp_path):
    seeds = _mined_seeds(tmp_path)
    s = _seed_of(seeds, "rename_function_xfile")
    base_prompt = tu3.render_zeta2_cross(dict(
        s["target"], path=s["target"]["path"],
        event_diff=s["event_diff"], references=s["references"]))
    for t in tu3.build_twin_exs(s):
        tgt = t["target"]
        p = tu3.render_zeta2_cross(dict(
            tgt, path=s["target"]["path"], event_diff=t["event_diff"],
            references=t["references"]))
        assert p != base_prompt
    # no-op twin: reference shows the ORIGINAL corpus lines
    noop = next(t for t in tu3.build_twin_exs(s)
                if t["kind"] == "no_op_reference")
    assert noop["references"][0]["render_lines"] == \
        s["references"][0]["corpus_lines"]
    # wrong twin: a DIFFERENT new name in the event + reference
    wrong = next(t for t in tu3.build_twin_exs(s)
                 if t["kind"] == "wrong_reference")
    assert s["meta"]["new"] not in wrong["event_diff"]


def test_f3_noop_twin_already_documented(tmp_path):
    seeds = _mined_seeds(tmp_path)
    s = _seed_of(seeds, "doc_sync_to_reference")
    noop = next(t for t in tu3.build_twin_exs(s)
                if t["kind"] == "no_op_reference")
    joined = "\n".join(noop["target"]["region_old"])
    assert f"#' @param {s['meta']['p']} " in joined


# ------------------------------------------------------ licenses + DESCRIPTION

def test_is_no_derivatives():
    for ok in ("GPL-3", "MIT + file LICENSE", "Apache License (== 2.0)",
               "GPL-2 | GPL-3", "CC BY-4.0", "LGPL-3", "?"):
        assert not tu3.is_no_derivatives(ok), ok
    for bad in ("CC BY-NC-ND 4.0", "CC-BY-ND 3.0", "Artistic-2.0 + ND",
                "no_derivatives", "CC BY-SA + NoDerivs"):
        assert tu3.is_no_derivatives(bad), bad


def test_parse_description_continuations(tmp_path):
    d = tmp_path / "DESCRIPTION"
    d.write_text("Package: foo\nTitle: A very long\n    title here\n"
                 "License: GPL-3\n")
    parsed = tu3.parse_description(d)
    assert parsed["Title"] == "A very long title here"
    assert tu3.license_field(parsed) == "GPL-3"


def test_remove_param_safe_cases():
    h = "f <- function(x, scale = 1) {"
    assert tu3.remove_param(h, "scale") == "f <- function(x) {"
    assert tu3.remove_param("f <- function(x, k) {", "k") == \
        "f <- function(x) {"
    assert tu3.remove_param("f <- function(x, k, j = 2) {", "k") == \
        "f <- function(x, j = 2) {"
    # unsafe defaults are rejected, never corrupted
    assert tu3.remove_param('f <- function(x, k = c("a","b")) {', "k") \
        is None
    assert tu3.remove_param("f <- function(x, k = rep(1, n)) {", "k") \
        is None


# ------------------------------------------------------- backend machinery

def test_preflight_fail_closed_without_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ZAI_API_KEY"):
        tu3.preflight_backend("zai")


def test_backends_registered_and_assign_backend_stable():
    # via tu3.b11 (its own import): the sibling test file re-executes
    # build_nextcoder_r under the same name, so a fresh import here can be
    # a different module instance with equal-looking classes
    assert tu3.AUTHOR_BACKENDS["zai"] is tu3.b11.ZaiAuthorBackend
    assert tu3.AUTHOR_BACKENDS["opencode-spark-free"] \
        is tu3.b11.SparkFreeAuthorBackend
    assert tu3.AUTHOR_BACKENDS["mock"].name == "mock"
    keys = [f"tu3_seed:{i}:rename_function_xfile:p0" for i in range(12)]
    bes = ["zai", "opencode-spark-free"]
    a1 = [tu3.assign_backend(k, bes) for k in keys]
    assert a1 == [tu3.assign_backend(k, bes) for k in keys]


# --------------------------------------------- evidence schema compatibility

def test_evidence_records_load_through_protocol():
    try:
        sys.path.insert(0, str(_SCRIPT.parents[2] / "packages" / "sepalith"
                               / "src"))
        from sepalith.protocol import EvidenceRecord
    except Exception:
        pytest.skip("sepalith.protocol not importable in this env")
    e = tu3.evidence_record("R/a.R", "deadbeef" * 8, ["x <- 1"], 3,
                            "foo", "reference")
    rec = EvidenceRecord.from_dict(e)
    assert rec.source_kind == "file" and rec.symbol == "foo"
    assert rec.source_range.start == 3


# --------------------------------------------- end-to-end smoke (fixtures only)

def test_smoke_end_to_end_and_resume(tmp_path, capsys, monkeypatch):
    for k in ("ZAI_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    work = tmp_path / "smoke"
    rc = tu3.main(["all", "--smoke", "--work", str(work),
                   "--out", str(work / "out")])
    assert rc == 0
    out = work / "out"
    for f in ("train.jsonl", "eval.jsonl", "manifest.json",
              "gates_report.json", "licenses.json", "probe_sample.jsonl"):
        assert (out / f).exists(), f
    train = [json.loads(l) for l in
             (out / "train.jsonl").read_text().splitlines() if l.strip()]
    evals = [json.loads(l) for l in
             (out / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert train and evals
    schema = {"text", "prompt", "target", "family", "package_or_repo",
              "has_types", "expected_noop", "twin_kind"}
    twins = [r for r in evals if r["expected_noop"]]
    assert twins                            # corrupted twins are eval rows
    assert all(r["twin_kind"] in tu3.TWIN_KINDS for r in twins)
    for r in train + evals:
        assert set(r) == schema
        assert r["text"] == r["prompt"] + r["target"]
        assert r["target"].endswith(">>>>>>> UPDATED")
        assert not r["expected_noop"] or r["twin_kind"]
    fams = {r["family"] for r in train + evals}
    assert fams == set(tu3.FAMILIES)
    gates = json.loads((out / "gates_report.json").read_text())
    assert gates["twins"]["gate_rejects_noop_answer"] is True
    assert gates["glm_probe"]["pass_at1_band"] == [0.30, 0.85]
    assert gates["glm_probe"]["status"].startswith("ready")
    assert gates["validator_acceptance"]["per_family"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["eval_twins"] == len(twins)
    assert manifest["targets"]["family_targets"] == \
        tu3.family_targets(12)
    # resume: every task key terminal -> nothing new authored
    n_before = len((work / "authored.jsonl").read_text().splitlines())
    assert tu3.main(["author", "--smoke", "--work", str(work)]) == 0
    assert "pending=0" in capsys.readouterr().out
    assert len((work / "authored.jsonl").read_text().splitlines()) \
        == n_before


def test_mock_backend_gate_failure_env(monkeypatch):
    monkeypatch.setenv("TU3_MOCK_INVALID", "1")
    b = tu3.MockAuthorBackend()
    hint = "#MOCK-HINT " + json.dumps(dict(
        task_key="k", family="rename_function_xfile",
        meta=dict(old="foo", new="foo2"),
        region_old=["r <- foo(x)"], expected_line=None))
    raw = b._complete_once("prompt\n" + hint)
    obj = json.loads(raw)
    assert any("<<" in l for l in obj["region_new"])
    ok, why = tu3.validate_region_new(obj["region_new"], _f1_seed())
    assert not ok
