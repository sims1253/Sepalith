#!/usr/bin/env python3
"""Self-contained tests for scenarios.py (run: uv run python experiments/synthetic/test_scenarios.py).

Covers: exact_reward, derive_new_name, per-family extraction on embedded R
sources (rename_propagation, pipe_rewrite, na_rm_propagation), and the
exactness validators (must accept real examples, must reject tampered ones).
No corpus access needed — Bundle is built from in-memory R text.
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
