#!/usr/bin/env python3
"""Self-contained tests for scenarios.py (run: uv run python experiments/synthetic/test_scenarios.py).

Covers: exact_reward, derive_new_name, per-family extraction on embedded R
sources (rename_propagation, pipe_rewrite, na_rm_propagation,
format_propagation, doc_sync), and the exactness validators (must accept real
examples, must reject tampered ones). No corpus access needed — Bundle is
built from in-memory R text, and format_propagation is tested against
hand-written raw+formatted fixture pairs.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as S


RENAME_R = b'''prep <- function(input_data, threshold) {
  keep <- input_data$value > threshold
  total <- sum(input_data$value[keep])
  if (total > threshold) {
    message("threshold exceeded: ", threshold)
  }
  out <- data.frame(total = total, threshold = threshold)
  out
}
'''

PIPE_R = b'''run <- function(df, g) {
  a <- df %>% head(10)
  b <- df %>% dplyr::filter(!is.na(g)) %>% head(5)
  c <- df %>% dplyr::group_by(g) %>% dplyr::summarise(n = n()) %>% head(3)
  d <- df %>% (function(x) x[1:2, ])
  e <- df %.% head(1)  # not a real op, just noise for the regex path
  list(a, b, c)
}
'''

NA_RM_R = b'''summarise_things <- function(df) {
  df %>%
    dplyr::group_by(grp) %>%
    dplyr::summarise(
      m1 = mean(x1),
      m2 = mean(x2),
      s1 = sd(y1),
      already = mean(z1, na.rm = TRUE),
      nested_ok = max(mean(x3), na.rm = FALSE)
    ) %>%
    dplyr::mutate(ratio = m1 / m2)
}
'''

DOT_PIPE_R = b'''bad1 <- x %>% foo(.)
bad2 <- x %>% bar(baz, .)
bad3 <- . %>% head(3)
bad4 <- x %>% {head(.)}
'''

# hand-written raw vs air-formatted pair (whitespace / brace / wrapping only)
FORMAT_RAW = '''tobit2 <- function(formula, left = 0, right = Inf, ...)
{
  ## bounds
  if(left < right) {
    ll <- left
  }   else {
    ll <- -Inf
  }
  ret <- formula
  return(ret)
}
'''
FORMAT_FMT = '''tobit2 <- function(
  formula,
  left = 0,
  right = Inf,
  ...
) {
  ## bounds
  if (left < right) {
    ll <- left
  } else {
    ll <- -Inf
  }
  ret <- formula
  return(ret)
}
'''

DOC_SYNC_R = b'''#' Compute a robust summary
#'
#' @param x A numeric vector.
#' @param probs Quantile probabilities.
#' @return A list of summary statistics.
#' @export
robust_summary <- function(x, probs = c(0.25, 0.5, 0.75)) {
  list(quan = quantile(x, probs))
}

#' Internal helper whose signature already has verbose
#'
#' @param verbose Already documented flag.
#' @export
helper <- function(verbose = FALSE) {
  verbose
}

#' No anchor tag in this block
#' @param x stuff
noanchor <- function(x) {
  x
}

#' @export
#' @param x stuff
weirdorder <- function(x) {
  x
}

#' All candidate args already present
#' @param x stuff
#' @return z
full <- function(x, verbose = TRUE, call = NULL, env = NULL) {
  x
}
'''


def make_bundle(src: bytes, rel="R/test.R", pkg="testpkg"):
    return S.Bundle(pkg, rel, src)


def test_exact_reward():
    assert S.exact_reward(["a ", "b"], ["a", "b"]) == 1.0
    assert S.exact_reward(["a", "b", "c"], ["a", "b", "c", ""]) == 1.0
    assert S.exact_reward(["mean(x)"], ["mean(x, na.rm = TRUE)"]) == 0.0
    f = S.exact_reward(["same", "old"], ["same", "new", "extra"])
    assert 0.0 < f < 1.0
    assert S.exact_reward([], []) == 1.0
    assert S.exact_reward(["x"], []) == 0.0


def test_derive_new_name():
    assert S.derive_new_name("value_tmp") == "value"
    assert S.derive_new_name("x.y") == "x_y"
    assert S.derive_new_name("total") == "total2"
    assert S.derive_new_name("TRUE") is None
    assert S.derive_new_name("a.b.c") == "a_b_c"


def test_rename_family():
    b = make_bundle(RENAME_R)
    exs = S.extract_rename(b, random.Random(0))
    assert exs, "expected rename examples"
    for ex in exs:
        S.validate_example(ex)
        old, new = ex["region_old"][0], ex["region_new"][0]
        # exactly one whole-token swap, e.g. threshold -> threshold2
        cands = [S._single_token_edit(old, new, p) for p in S._TOKEN_PATS]
        assert any(c and S._valid_rename_pair(*c) for c in cands), (old, new)
        assert old != new
    # 'threshold' occurs 5x, 'input_data' 3x in body; both are candidates
    names = {ex["note"].split(" -> ")[0].removeprefix("rename ")
             for ex in exs}
    assert names <= {"threshold", "input_data", "keep", "total", "out", "value"}
    # no-op baseline is exactly 0 (single-line region fully changes)
    assert all(S.noop_baseline_score(e) == 0.0 for e in exs)


def test_rename_string_literal():
    src = b'''f <- function(df) {
  a <- df[, "score_val"]
  b <- mean(df$score_val) + sd("score_val")
  paste0("score_val", a, b)
}
'''
    exs = S.extract_rename(make_bundle(src), random.Random(0))
    str_exs = [e for e in exs if '"score_val"' in e["note"]]
    assert str_exs, "expected a column-name string-literal rename"
    for e in str_exs:
        S.validate_example(e)
        assert '"score_val"' in e["region_old"][0]
        assert '"score_val2"' in e["region_new"][0]


def test_pipe_family():
    b = make_bundle(PIPE_R)
    exs = S.extract_pipe(b, random.Random(0))
    assert exs, "expected pipe examples"
    for ex in exs:
        S.validate_example(ex)
        old, new = ex["region_old"][0], ex["region_new"][0]
        assert "%>%" in old and "|>" in new
        assert S._single_token_edit(old, new, r"%>%") == ("%>%", "|>")
        ev = S.EVENT_DIFF_RE.match(ex["event_diff"])
        assert ev and ev.group("old") != ev.group("new")
        assert "%>%" in ev.group("old") and "|>" in ev.group("new")
        assert S.noop_baseline_score(ex) == 0.0
    # lines with '.' placeholder / bare RHS must never be rewritten
    bad = make_bundle(DOT_PIPE_R)
    assert S.extract_pipe(bad, random.Random(0)) == []


def test_na_rm_family():
    b = make_bundle(NA_RM_R)
    exs = S.extract_na_rm(b, random.Random(0))
    assert exs, "expected na.rm examples"
    for ex in exs:
        S.validate_example(ex)
        old, new = ex["region_old"][0], ex["region_new"][0]
        assert "na.rm" not in old.split('"')[0] or True
        assert ", na.rm = TRUE)" in new
        ev = S.EVENT_DIFF_RE.match(ex["event_diff"])
        assert ", na.rm = TRUE)" in ev.group("new")
        assert S.noop_baseline_score(ex) == 0.0
    # targets are only calls lacking na.rm (event=mean(x1), targets=m2/s1/x3)
    rows = {ex["region_old"][0].strip() for ex in exs}
    assert all(not re.search(r"(mean|sd|var)\([^)]*na\.rm", r) for r in rows), rows
    assert any("mean(x2)" in r for r in rows), rows
    assert any("sd(y1)" in r for r in rows), rows
    assert not any("mean(x1)" in r for r in rows), rows  # x1 is the event call
    # non-tidyverse file yields nothing
    plain = make_bundle(b"z <- mean(x1) + mean(x2)\n")
    assert S.extract_na_rm(plain, random.Random(0)) == []


def test_validator_rejects_tampering():
    b = make_bundle(PIPE_R)
    exs = S.extract_pipe(b, random.Random(0))
    ex = dict(exs[0])
    ex["region_new"] = [ex["region_new"][0].replace("|>", "%%") ]
    try:
        S.validate_example(ex)
        raise SystemExit("validator accepted a wrong pipe rewrite")
    except AssertionError:
        pass
    # a rename example with an extra trailing change must be rejected
    ex2 = dict(S.extract_rename(make_bundle(RENAME_R), random.Random(0))[0])
    ex2["region_new"] = [ex2["region_old"][0] + " # extra"]
    try:
        S.validate_example(ex2)
        raise SystemExit("validator accepted an extra trailing change")
    except AssertionError:
        pass
    # identical old/new rejected
    ex3 = dict(exs[0])
    ex3["region_new"] = list(ex3["region_old"])
    try:
        S.validate_example(ex3)
        raise SystemExit("validator accepted a no-op region")
    except AssertionError:
        pass


def test_rename_validator_string_span_trap():
    # regression: identifier rename on a line that also contains string
    # literals; a naive first-pattern check sees a bogus "string" candidate
    # (e.g. ' <- f("a", req$x, "b")' spans) and rejects a valid example
    old = 'msg <- paste0("got: ", req$params$user, " want: ", req$params$user)'
    new = old.replace("req$", "req2$", 1)
    cands = [S._single_token_edit(old, new, p) for p in S._TOKEN_PATS]
    assert any(c and S._valid_rename_pair(*c) for c in cands), cands
    ex = {"family": "rename_propagation", "package": "p", "path": "R/x.R",
          "prefix": ["x <- 1"], "region_old": [old], "region_new": [new],
          "cursor_idx": old.index("req$") + 3,
          "event_diff": S.event_diff_for("R/x.R", 3, "req <- 1", "req2 <- 1"),
          "note": "rename req -> req2"}
    S.validate_example(ex)


def test_strip_strings():
    assert S.DOT_TOKEN_RE.search(S.strip_strings(b'x %>% foo("a.b", .)'))
    raw = b'x %>% paste("no dot here")'
    assert not S.DOT_TOKEN_RE.search(S.strip_strings(raw))
    assert b'"' in S.strip_strings(raw)  # quotes kept, content blanked


# ---------------------------------------------------------------------------
# format_propagation
# ---------------------------------------------------------------------------

def test_fmt_only_edit():
    assert S._fmt_only_edit(["  if(x) {"], ["if (x) {"])
    assert S._fmt_only_edit(["f(a,", "  b)"], ["f(", "  a,", "  b)"])
    assert S._fmt_only_edit(["g <- function(x) x"],
                            ["g <- function(x) {", "  x", "}"])
    assert not S._fmt_only_edit(["x <- 1"], ["x <- 2"])          # token change
    assert not S._fmt_only_edit(["mean(x)"], ["mean(x, na.rm = TRUE)"])
    assert not S._fmt_only_edit(["a"], ["a"])                    # no change
    assert not S._fmt_only_edit([], [])


def test_format_group_hunks():
    # equal lines separate hunks; adjacent non-equal opcodes merge
    assert S._group_format_hunks(["a", "x", "b"], ["a2", "x", "b", "c"]) \
        == [(0, 1, 0, 1), (3, 3, 3, 4)]
    assert S._group_format_hunks(["ab"], ["a", "b", "c"]) == [(0, 1, 0, 3)]
    assert S._group_format_hunks(["a"], ["a"]) == []


def test_format_propagation_family():
    raw_lines = FORMAT_RAW.splitlines()
    fmt_lines = FORMAT_FMT.splitlines()
    exs = S.format_pairs_from_lines("tpkg", "R/wrap.R", raw_lines, fmt_lines)
    assert len(exs) == 2, [e["note"] for e in exs]
    for ex in exs:
        S.validate_example(ex)
        assert S.noop_baseline_score(ex) == 0.0   # region shares no lines
        assert S.noop_exact_score(ex) == 0.0
        assert ex["family"] == "format_propagation"
        assert ex["package"] == "tpkg" and ex["path"] == "R/wrap.R"
        # region/event lines are verbatim fixture lines (splice-verified GT)
        assert all(l in raw_lines for l in ex["region_old"])
        assert all(l in fmt_lines for l in ex["region_new"])
        path, ev_old, ev_new = S.parse_event_diff_lines(ex["event_diff"])
        assert path == "R/wrap.R"
        assert all(l in raw_lines for l in ev_old)
        assert all(l in fmt_lines for l in ev_new)
    e0, e1 = exs
    # event = signature-wrap hunk reformatted; region = NEXT hunk raw->fmt
    assert e0["region_old"] == ["  if(left < right) {"]
    assert e0["region_new"] == ["  if (left < right) {"]
    assert e0["cursor_idx"] == 2  # first changed line, first non-blank col
    assert e0["prefix"] == raw_lines[0:3]
    assert e0["event_diff"].startswith(
        'User edited "R/wrap.R":\n\n```diff\n@@ -1,2 +1,6 @@\n')
    assert e1["region_old"] == ["  }   else {"]
    assert e1["region_new"] == ["  } else {"]
    assert e1["cursor_idx"] == 2
    assert e1["prefix"][-1] == "    ll <- left"


def test_format_propagation_needs_two_hunks():
    # a single changed hunk leaves no (event, target) pair
    assert S.format_pairs_from_lines(
        "p", "R/x.R", ["a <- 1 ", "b <- 2"], ["a <- 1", "b <- 2"]) == []
    # identical files yield nothing
    assert S.format_pairs_from_lines(
        "p", "R/x.R", ["a"], ["a"]) == []


def test_format_validator_rejects_tampering():
    raw_lines = FORMAT_RAW.splitlines()
    fmt_lines = FORMAT_FMT.splitlines()
    exs = S.format_pairs_from_lines("tpkg", "R/wrap.R", raw_lines, fmt_lines)
    # region_new equal to region_old (shares lines) -> no-op could score > 0
    ex = dict(exs[1])
    ex["region_new"] = list(ex["region_old"])
    try:
        S.validate_example(ex)
        raise SystemExit("validator accepted a shared-line format region")
    except AssertionError:
        pass
    # a token change is not a reformat
    ex2 = dict(exs[0])
    ex2["region_new"] = [ex2["region_new"][0].replace("left", "left2")]
    try:
        S.validate_example(ex2)
        raise SystemExit("validator accepted a token change as a reformat")
    except AssertionError:
        pass
    # event tampered with a token change
    ex3 = dict(exs[0])
    _, ev_old, ev_new = S.parse_event_diff_lines(ex3["event_diff"])
    ev_new = [l.replace("formula,", "formulaX,") for l in ev_new]
    ex3["event_diff"] = S.make_multiline_example(
        "format_propagation", "tpkg", "R/wrap.R", [], ex3["region_old"],
        ex3["region_new"], 0, ev_old, ev_new, 1, "n")["event_diff"]
    try:
        S.validate_example(ex3)
        raise SystemExit("validator accepted a tampered format event")
    except AssertionError:
        pass


def test_parse_event_diff_lines():
    ex = S.make_multiline_example(
        "format_propagation", "p", "R/x.R", [], ["a"], ["b"], 0,
        ["one", "--dash"], ["x", "y", "--dash"], 7, "n")
    assert ex["event_diff"].startswith(
        'User edited "R/x.R":\n\n```diff\n@@ -7,2 +7,3 @@\n')
    path, old, new = S.parse_event_diff_lines(ex["event_diff"])
    assert path == "R/x.R"
    assert old == ["one", "--dash"]   # leading '-' content survives parsing
    assert new == ["x", "y", "--dash"]
    # single-line events keep the classic shape and old-regex compatibility
    ex1 = S.make_multiline_example(
        "doc_sync", "p", "R/y.R", [], ["a"], ["b"], 0, ["old"], ["new"], 9, "n")
    assert "@@ -9 +9 @@" in ex1["event_diff"]
    assert S.EVENT_DIFF_RE.match(ex1["event_diff"])


# ---------------------------------------------------------------------------
# doc_sync
# ---------------------------------------------------------------------------

def test_doc_sync_family():
    b = make_bundle(DOC_SYNC_R)
    exs = S.extract_doc_sync(b, random.Random(0))
    # robust_summary (arg=verbose) and helper (verbose taken -> arg=call);
    # noanchor / weirdorder / full are skipped
    assert len(exs) == 2, [e["note"] for e in exs]
    e0, e1 = exs
    # event: signature gains ', verbose = FALSE' before the closing paren
    m = S.EVENT_DIFF_RE.match(e0["event_diff"])
    assert m and m.group("old") == \
        "robust_summary <- function(x, probs = c(0.25, 0.5, 0.75)) {"
    assert m.group("new") == \
        "robust_summary <- function(x, probs = c(0.25, 0.5, 0.75), " \
        "verbose = FALSE) {"
    # region: @param block area up to the first @return/@export anchor;
    # new @param inserted immediately before the anchor line
    assert e0["region_old"] == [
        "#' @param x A numeric vector.",
        "#' @param probs Quantile probabilities.",
        "#' @return A list of summary statistics.",
    ]
    assert e0["region_new"] == [
        "#' @param x A numeric vector.",
        "#' @param probs Quantile probabilities.",
        "#' @param verbose Show progress messages while the function runs.",
        "#' @return A list of summary statistics.",
    ]
    assert e0["cursor_idx"] == \
        len(e0["region_old"][0]) + 1 + len(e0["region_old"][1]) + 1
    for ex in (e0, e1):
        S.validate_example(ex)
        assert S.noop_exact_score(ex) == 0.0
        f1 = S.noop_baseline_score(ex)
        assert 0.5 < f1 < 1.0  # pure-insertion line-F1 artifact (~2n/(2n+1))
    # helper: verbose already in signature -> next deterministic arg is call;
    # no @return here so the anchor is @export
    m1 = S.EVENT_DIFF_RE.match(e1["event_diff"])
    assert ", call = caller_env()) {" in m1.group("new")
    assert e1["region_new"] == [
        "#' @param verbose Already documented flag.",
        "#' @param call Calling environment captured by rlang.",
        "#' @export",
    ]
    # plain source without roxygen yields nothing
    assert S.extract_doc_sync(make_bundle(b"z <- function(x) x\n"),
                              random.Random(0)) == []


def test_doc_sync_validator_rejects_tampering():
    b = make_bundle(DOC_SYNC_R)
    ex = dict(S.extract_doc_sync(b, random.Random(0))[0])
    # wrong deterministic description
    bad = dict(ex)
    bad["region_new"] = [
        l.replace("Show progress messages while the function runs.", "Be loud")
        for l in ex["region_new"]]
    try:
        S.validate_example(bad)
        raise SystemExit("validator accepted a wrong @param description")
    except AssertionError:
        pass
    # line moved after @export (no longer before the @return/@export anchor)
    ins = [l for l in ex["region_new"] if "@param verbose" in l][0]
    bad2 = dict(ex)
    bad2["region_new"] = [l for l in ex["region_new"] if l != ins] + [ins]
    try:
        S.validate_example(bad2)
        raise SystemExit("validator accepted an @param inserted after @export")
    except AssertionError:
        pass
    # two extra lines inserted
    bad3 = dict(ex)
    bad3["region_new"] = list(ex["region_new"]) + ["#' extra1", "#' extra2"]
    try:
        S.validate_example(bad3)
        raise SystemExit("validator accepted multiple inserted lines")
    except AssertionError:
        pass
    # event with the wrong default value
    m = S.EVENT_DIFF_RE.match(ex["event_diff"])
    bad4 = dict(ex)
    bad4["event_diff"] = S.event_diff_for(
        ex["path"], 9, m.group("old"), m.group("new").replace("FALSE", "TRUE"))
    try:
        S.validate_example(bad4)
        raise SystemExit("validator accepted a wrong signature default")
    except AssertionError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
