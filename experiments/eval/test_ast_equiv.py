"""Tests for ast_equiv.py (V1b).

Run:  cd /home/m0hawk/Documents/Sepalith && \
      uv run --with pytest python -m pytest experiments/eval/test_ast_equiv.py -q

Each test asserts what the implementation ACTUALLY guarantees; the blind
spots that are kept by design (documented in the module docstring) are
asserted as such, not hidden.
"""

import json
import os
import sys
from argparse import Namespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ast_equiv import (  # noqa: E402
    HAVE_TS,
    RSCRIPT,
    align_midtyping,
    canonical,
    detect_pred_cap,
    load_rows,
    py_canonical,
    r_canonical,
    rscript_parses,
    score_file,
    score_pair,
    score_row,
)

needs_ts = pytest.mark.skipif(not HAVE_TS, reason="tree-sitter-r not importable")
needs_rscript = pytest.mark.skipif(not RSCRIPT, reason="Rscript not on PATH")


def eq(pred, gt, lang="r"):
    return score_pair(pred, gt, lang)


# ---------------------------------------------------------- (a) reformat ---
@needs_ts
def test_reformat_only_diff():
    sc = eq("f( a=1, b=2 )", "f(a=1,b=2)")
    assert sc["ast_equiv"] == 1 and sc["tree_equal"]
    assert "f( a=1, b=2 )" != "f(a=1,b=2)"  # exact string match is 0: the gap


@needs_ts
def test_whitespace_and_statements():
    assert eq("x <- 1\ny <- 2", "x<-1;\ny<-2")["ast_equiv"] == 1


@needs_ts
def test_assignment_operator_normalized():  # N6: <- vs =
    assert eq("x <- 1", "x = 1")["ast_equiv"] == 1
    assert eq("x <<- 1", "x <- 1")["ast_equiv"] == 0  # super-assign differs


@needs_ts
def test_comment_dropped():  # N1
    assert eq("# a comment\nx <- 1", "x <- 1")["ast_equiv"] == 1


# ------------------------------------------------------- (b) named args ---
@needs_ts
def test_named_arg_reorder():  # N5
    assert eq("g(alpha = 1, beta = \"s\")", "g(beta = \"s\", alpha = 1)")[
        "ast_equiv"] == 1


@needs_ts
def test_named_before_positional_is_equivalent():
    assert eq("g(a = 1, x)", "g(x, a = 1)")["ast_equiv"] == 1


@needs_ts
def test_named_values_not_swapped_when_distinguishable():
    assert eq("g(a = x, b = y)", "g(a = y, b = x)")["ast_equiv"] == 0


@needs_ts
def test_literal_value_swap_is_invisible():  # documented N2 blind spot
    assert eq("g(a = 1, b = 2)", "g(a = 2, b = 1)")["ast_equiv"] == 1


# -------------------------------------------- (c)/(f) positional & twins ---
@needs_ts
def test_positional_args_swapped_is_zero():  # case (c)
    assert eq("h(a, b)", "h(b, a)")["ast_equiv"] == 0


@needs_ts
def test_positional_literal_swap_is_invisible():  # N2: values placeholdered
    assert eq("h(1, 2)", "h(2, 1)")["ast_equiv"] == 1


@needs_ts
def test_corrupted_twin_swap_args():  # case (f): RL README corrupted-twin rule
    gt = "file.copy(from_path, to_path)"
    assert eq(gt, gt)["ast_equiv"] == 1
    assert eq("file.copy(to_path, from_path)", gt)["ast_equiv"] == 0


@needs_ts
def test_corrupted_twin_subset_bracket_class():  # x[i] vs x[[i]] differ (N7)
    assert eq("out[recent, ]", "out[[recent]]")["ast_equiv"] == 0


@needs_ts
def test_corrupted_twin_battery_line():
    gt = "on.exit(file_delete(built_path2), add = TRUE)"
    assert eq("on.exit(file_delete(built_path2), add=TRUE)", gt)["ast_equiv"] == 1
    assert eq("on.exit(add=TRUE, file_delete(built_path2))", gt)["ast_equiv"] == 1  # R-legal reorder
    assert eq("on.exit(file_delete(TRUE), add = built_path2)", gt)["ast_equiv"] == 0


# --------------------------------------------- (d) identifier guarantees ---
@needs_ts
def test_wrong_variable_caught_by_vars_check():  # N4 mitigation
    sc = eq("x + y", "x + z")
    assert sc["tree_equal"] and sc["ast_equiv"] == 0  # blind spot is closed
    assert sc["notes"].count("vars_differ") == 1


@needs_ts
def test_identifier_reuse_structure_matters():  # N3 keeps which-name-where
    assert eq("x + x", "x + y")["ast_equiv"] == 0
    assert eq("foo(a, b)", "foo(b, a)")["ast_equiv"] == 0


@needs_ts
def test_wrong_side_rename_from_real_gap_row():
    # actual v8_2 gap-row pattern (rename direction swapped)
    assert eq("    manual2 = manual,", "    manual = manual2,")["ast_equiv"] == 0


@needs_ts
def test_consistent_global_rename_is_zero():  # documented N3/N4 semantics
    # an order-preserving bijection exists ({a,b,foo}->{c,d,foo}), so the
    # trees ARE equal; the vars multiset check rejects the rename: the GT
    # names its targets (v2 TODO: bijection from scenario metadata)
    sc = eq("foo(c, d)", "foo(a, b)")
    assert sc["tree_equal"] is True
    assert sc["ast_equiv"] == 0 and "vars_differ" in sc["notes"]


@needs_ts
def test_consistent_rename_within_expression():
    # x renamed to z at BOTH occurrences: still 0 (set changed -> map shift)
    sc = eq("z + z * z", "x + x * x")
    assert sc["ast_equiv"] == 0


# ------------------------------------------------------- (e) broken preds ---
@needs_ts
def test_syntax_broken_pred():
    c = r_canonical("x <- <bad")
    assert c["parses"] is False
    sc = eq("x <- <bad", "x <- 1")
    assert sc["ast_equiv"] == 0


@needs_ts
def test_empty_pred():
    sc = eq("", "x <- 1")
    assert sc["ast_equiv"] == 0 and "empty_pred" in sc["notes"]


@needs_ts
def test_no_gt_is_weak():
    sc = eq("x <- 1", None)
    assert sc["weak"] is True and sc["ast_equiv"] is None


@needs_ts
def test_mid_expression_fragment_text_fallback():
    # GT region itself is mid-expression (pipe continuation): structural
    # tiers do not apply; the battery's text normalization decides
    sc = eq("  ) |>", "  ) |>")
    assert sc["ast_equiv"] == 1
    assert "mid_expression_fragment" in sc["notes"]
    assert eq("  ) |> f()", "  ) |>")["ast_equiv"] == 0
    assert eq("  testthat::with_reporter(reporter2, {",
              "  testthat::with_reporter(reporter2, {")["ast_equiv"] == 1


@needs_ts
def test_fragment_row_scored_as_fragment():
    o = score_row(dict(pred="  ) |>", lang="r"), gt_text="  ) |>")
    assert o["fragment"] is True and o["ast_equiv"] == 1
    o2 = score_row(dict(pred="x <- 1", lang="r"), gt_text="x <- 1")
    assert o2["fragment"] is False


# ------------------------------------------------ fragment tolerance (R) ---
@needs_ts
def test_trailing_comma_fragment_parses():
    c = r_canonical("      vignettes2 = vignettes,")
    assert c["parses"] == 1 and "trailing_comma" in c["notes"]
    assert eq("      vignettes2 = vignettes,", "vignettes2 = vignettes")[
        "ast_equiv"] == 1  # comma is dropped symmetrically
    assert eq("      vignettes2 = vignettes,", "vignettes = vignettes2")[
        "ast_equiv"] == 0  # wrong-side rename still caught on a fragment


@needs_ts
def test_unclosed_delimiter_fragment():
    c = r_canonical("  caps$rstudio_console <- function(data) {")
    assert c["parses"] == 1 and "unclosed_delim" in c["notes"]
    assert eq("f(a = 1, b = 2", "f(a=1,b=2)")["ast_equiv"] == 1  # lower bound


# --------------------------------------------------------- literals (N2) ---
@needs_ts
def test_string_values_placeholdered():
    assert eq("s <- \"aaa\"", "s <- 'bbb'")["ast_equiv"] == 1  # documented


@needs_ts
def test_literal_class_kept():
    assert eq("flag <- TRUE", "flag <- 1")["ast_equiv"] == 0
    assert eq("v <- NULL", "v <- NA")["ast_equiv"] == 0


# ------------------------------------------------------ comment-only (N1) ---
@needs_ts
def test_comment_only_region_text_fallback():
    gt = "#' @param quiet Suppress output\n#' @export"
    same = "#'   @param quiet  Suppress output\n#'  @export"
    diff = "#' @param loud Extra output\n#' @export"
    assert eq(same, gt)["ast_equiv"] == 1
    assert "comments_text_only" in eq(same, gt)["notes"]
    assert eq(diff, gt)["ast_equiv"] == 0
    assert eq("x <- 1", gt)["ast_equiv"] == 0  # code vs comment-only region


# ------------------------------------------------------------- Python tier --
def test_py_named_arg_reorder():
    assert eq("f(a=1, b=\"s\")", "f(b=\"s\", a=1)", "python")["ast_equiv"] == 1


def test_py_positional_swap():
    assert eq("f(a, b)", "f(b, a)", "python")["ast_equiv"] == 0


def test_py_broken_pred():
    c = py_canonical("def f(:")
    assert c["parses"] is False
    assert eq("def f(:", "x = 1", "python")["ast_equiv"] == 0


def test_py_consistent_rename_zero_by_set_shift():
    sc = eq("foo(z) + bar(z)", "foo(a) + bar(a)", "python")
    assert sc["tree_equal"] is False and sc["ast_equiv"] == 0


def test_py_whitespace_only():
    assert eq("x = [1, 2]", "x=[1,2]", "python")["ast_equiv"] == 1


# ------------------------------------------------------- strict tier (R) ---
@needs_rscript
def test_rscript_strict_tier():
    assert rscript_parses("x <- 1") is True
    assert rscript_parses("f(a=,") is False


@needs_rscript
@needs_ts
def test_strict_tier_rejects_valid_fragment_divergence():
    # documented divergence: top-level trailing comma is a valid mid-call
    # fragment (tree tier: parses=1) but R parse() rejects it (strict: False)
    assert rscript_parses("quiet2 = quiet,") is False
    assert r_canonical("quiet2 = quiet,")["parses"] == 1


# -------------------------------------------------------------- row level ---
@needs_ts
def test_score_row_truncation_flag():
    row = dict(family="pipe_rewrite", id="x", i=0, exact=0,
               pred="f(" + "a" * 500, lang="r")
    o = score_row(row, gt_text="f(aaaa)")
    assert o["truncated"] is True
    assert "truncated_lower_bound" in o["norm_notes"]
    assert o["parses"] in (0, 1)  # still scored
    short = score_row(dict(row, pred="f(1)"), gt_text="f(1)")
    assert short["truncated"] is False


@needs_ts
def test_score_row_pass_through_ids_and_gap_fields():
    row = dict(id="abc", family="rename_propagation", i=3, package="devtools",
               path="R/check.R", model="m", exact=0, valid_pass=1,
               pred="document(pkg, quiet2=quiet)")
    o = score_row(row, gt_text="document(pkg, quiet2 = quiet)")
    assert o["id"] == "abc" and o["family"] == "rename_propagation"
    assert o["exact"] == 0 and o["ast_equiv"] == 1  # the gap row
    assert o["parses"] == 1 and o["parse_tool"] == "treesitter"


@needs_ts
def test_score_row_truncated_prefix_is_uncomputable():
    gt = "x <- 1\n" + "y <- 2\n" * 60
    pred = gt[:400]  # stored pred cut at the cap, strict prefix of GT
    o = score_row(dict(pred=pred, lang="r"), gt_text=gt)
    assert o["truncated"] is True
    assert o["ast_equiv"] is None and "truncated_prefix" in o["norm_notes"]
    # a truncated pred that is NOT a GT prefix is still scored (lower bound)
    o2 = score_row(dict(pred="f(" + "a" * 500, lang="r"), gt_text="f(bbb)")
    assert o2["truncated"] is True and o2["ast_equiv"] == 0


@needs_ts
def test_score_row_vars_diff_reported():
    o = score_row(dict(pred="x + z", lang="r"), gt_text="x + y")
    assert o["ast_equiv"] == 0
    assert o["vars_pred_only"] == ["z"] and o["vars_gt_only"] == ["y"]


def test_midtyping_alignment_full_line_echo():
    gt_lines = ["document(pkg, quiet = FALSE)", "  x <- 1"]
    pred_lines = ["document(pkg, quiet = FALSE)", "x <- 1"]  # whole-line echo
    p, g, note = align_midtyping(pred_lines, gt_lines)
    assert note == "midtyping_full_line"
    assert eq("\n".join(p), "\n".join(g))["ast_equiv"] == 1


def test_midtyping_alignment_suffix():
    gt_lines = ["document(pkg, quiet = FALSE)", "x <- 1"]
    # completion-only pred line 0 (align=suffix convention): reconstruct
    p, g, note = align_midtyping(["et = FALSE)", "x <- 1"], gt_lines)
    assert note.startswith("midtyping_cut=")  # 29-char line, 11-char completion
    assert eq("\n".join(p), "\n".join(g))["ast_equiv"] == 1


def test_midtyping_alignment_unaligned():
    p, g, note = align_midtyping(["zzz"], ["document(pkg, quiet = FALSE)"])
    assert note == "midtyping_unaligned"


# ------------------------------------------------------------ file level ---
def _tmp_results(tmp_path, rows, agg_block=True):
    p = tmp_path / "results_testmid_midtyping.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if agg_block:  # multi-line pretty-printed aggregate must be skipped
            f.write('{\n "model": "zeta2",\n "agg": {"r": {"n": 1}}\n}\n')
    return p


def test_load_rows_skips_aggregate_block(tmp_path):
    p = _tmp_results(tmp_path, [dict(i=0, lang="r", repo="cli", path="R/a.R",
                                     sha="s1", exact=0, pred="x <- 1")])
    rows = load_rows(p)
    assert len(rows) == 1 and rows[0]["sha"] == "s1"


def test_detect_pred_cap():
    assert detect_pred_cap([{"family": "f", "pred": ""}]) == 400
    assert detect_pred_cap([{"lang": "r", "pred": ""}]) == 600


@needs_ts
def test_score_file_end_to_end_gap(tmp_path, capsys):
    rows = [dict(i=0, lang="r", repo="cli", path="R/a.R", sha="s1", exact=0,
                 pred="f( a=1, b=2 )")]
    p = _tmp_results(tmp_path, rows, agg_block=False)
    examples = tmp_path / "examples.jsonl"
    ex = dict(lang="r", repo="cli", path="R/a.R", sha="s1",
              region_old=["old line"], region_new=["f(a=1,b=2)"])
    examples.write_text(json.dumps(ex) + "\n")
    out = tmp_path / "out.jsonl"
    args = Namespace(results=p, lang="r", out=out, strict=False,
                     scen_dir=tmp_path, examples=examples)
    agg = score_file(args)
    a = agg["lang:r"]
    assert a["n"] == 1 and a["exact"] == 0 and a["ast_equiv"] == 1
    assert a["gap"] == 1.0 and a["gap_rows"] == 1  # the exact->ast_equiv gap
    lines = [json.loads(l) for l in open(out)]
    assert lines[-1]["aggregate"]["total"] == 1
    assert any("midtyping_cut" in n for n in lines[0]["norm_notes"]) or \
        "midtyping_unaligned" in lines[0]["norm_notes"]
    printed = capsys.readouterr().out
    assert "lang:r" in printed  # the gap table


@needs_ts
def test_canonical_shapes_stable():
    # sanity: canonical form is a deterministic function of the tree
    a = canonical("f( a=1, b=2 )", "r")["form"]
    b = canonical("f(a=1,b=2)", "r")["form"]
    assert a == b and a.startswith("program")


def test_full_raw_recovers_capped_exact_prediction():
    from ast_equiv import score_row
    gt = "value <- " + " + ".join(["some_identifier"] * 40)
    result = score_row(dict(pred=gt[:400], raw=gt, exact=1), gt)
    assert result["ast_equiv"] == 1
    assert result["truncated"] is False
    assert result["prediction_source"] == "raw_recovered"


def test_unrelated_raw_does_not_remove_truncation():
    from ast_equiv import score_row
    gt = "value <- " + " + ".join(["some_identifier"] * 40)
    result = score_row(dict(pred=gt[:400], raw="unrelated <- value", exact=1), gt)
    assert result["ast_equiv"] is None
    assert result["truncated"] is True
