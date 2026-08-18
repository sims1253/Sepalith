#!/usr/bin/env python3
"""Self-contained tests for comment_to_code.py (run:
uv run python experiments/synthetic/test_comment_to_code.py).

Covers: variant-A extraction on embedded R sources (roxygen excluded,
blank-line/other-comment/inline-comment stops, statement-count gate,
undefined-name gate), variant-B candidate windows (comment-free, 2-8 lines,
>= 2 calls or pipe), the LLM comment gate + normalization, record
construction/validator exactness (must accept real examples, must reject
tampered ones), the mock-LLM synthetic pipeline, and comment-density stats.
No corpus access and no network — Bundle is built from in-memory R text and
the LLM is mocked.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as S
import comment_to_code as C


REAL_R = b'''#' Roxygen title for prep
prep <- function(input_data, threshold) {
  # keep only rows above the threshold
  keep <- input_data$value > threshold
  out <- input_data[keep, ]

  # scale and total the result
  scaled <- out * 2
  total <- sum(scaled)
  total
}
'''

SINGLE_STMT_R = b'''single <- function(x) {
  # only one statement follows
  y <- x + 1

  y
}
'''

BLANK_FIRST_R = b'''blank_first <- function(x) {
  # a blank line comes before any code

  x + 1
  x + 2
}
'''

TWO_COMMENTS_R = b'''two_comments <- function(x) {
  # first comment
  # second comment right below
  a <- x + 1
  b <- x + 2
  a + b
}
'''

INLINE_R = b'''inline_cmt <- function(x) {
  # inline comment below truncates the block
  a <- x + 1
  b <- mean(a)  # inline trailing comment
  c <- sum(b)
  c
}
'''

UNDEF_R = b'''undef <- function(x) {
  # uses nothing bound anywhere
  zz1 <- mystery_a(x) + mystery_b(x)
  zz2 <- mystery_c(zz1) * mystery_d(zz1)
  zz3 <- mystery_e(zz2)
  zz3
}
'''

LOCAL_HELPER_R = b'''bound_ok <- function(x) {
  helper <- function(v) sum(v)
  # call the locally defined helper
  r1 <- helper(x)
  r2 <- max(r1, 1)
  r1 + r2
}
'''

PIPE_R = b'''run <- function(df, g) {
  a <- df %>% head(10)
  b <- df %>% dplyr::filter(!is.na(g)) %>% head(5)
  c <- df %>% dplyr::group_by(g) %>% dplyr::summarise(n = n()) %>% head(3)
  out <- list(a, b, c)
  print(out)
  out
}
'''

MERGE_R = b'''tidy <- function(df, g) {
  res1 <- df %>% dplyr::filter(!is.na(g))
  res2 <- df %>% dplyr::group_by(g) %>% dplyr::summarise(m = mean(g, na.rm = TRUE))
  res3 <- base::merge(res1, res2, by = "g")
  res3
}
'''


def make_bundle(src: bytes, rel="R/test.R", pkg="testpkg"):
    return S.Bundle(pkg, rel, src)


def test_real_pairs_extraction():
    b = make_bundle(REAL_R)
    exs = C.extract_comment_pairs(b)
    assert len(exs) == 2, f"expected 2 pairs, got {len(exs)}"
    first = exs[0]
    assert first["family"] == "comment_to_code_real"
    assert first["region_old"] == [""]
    assert first["region_new"] == ["  keep <- input_data$value > threshold",
                                   "  out <- input_data[keep, ]"]
    assert first["event_diff"] == ""
    assert first["prefix"][-1] == "  # keep only rows above the threshold"
    assert "#' " not in "\n".join(first["prefix"])  # roxygen never leaks in
    for ex in exs:
        C.validate_example(ex)
        assert C.noop_baseline_score(ex) == 0.0
        assert S.exact_reward(ex["region_new"], ex["region_new"]) == 1.0


def test_real_pairs_negative_cases():
    # exactly one qualifying comment (>= 2 statements, nothing between)
    assert C.extract_comment_pairs(make_bundle(SINGLE_STMT_R)) == []
    # blank line directly after the comment -> no block
    assert C.extract_comment_pairs(make_bundle(BLANK_FIRST_R)) == []
    # only the LAST comment of a run can be the cursor line
    exs = C.extract_comment_pairs(make_bundle(TWO_COMMENTS_R))
    assert len(exs) == 1
    assert exs[0]["prefix"][-1].strip() == "# second comment right below"
    # an inline trailing comment stops the block -> 1 statement -> rejected
    assert C.extract_comment_pairs(make_bundle(INLINE_R)) == []
    # clearly undefined-heavy block is skipped
    assert C.extract_comment_pairs(make_bundle(UNDEF_R)) == []


def test_real_pairs_local_helper_bound():
    exs = C.extract_comment_pairs(make_bundle(LOCAL_HELPER_R))
    assert len(exs) == 1, "'helper' is bound by an earlier prefix assignment"
    C.validate_example(exs[0])


def test_candidate_blocks():
    cands = C.candidate_blocks(make_bundle(PIPE_R))
    assert cands, "expected pipe candidates"
    for c in cands:
        lines = c["block"]
        assert 2 <= len(lines) <= 8
        assert all(l.strip() and not C.is_comment_line(l) for l in lines)
        assert not any(C.line_has_inline_comment(l.encode()) for l in lines)
        calls, has_pipe = C.fragment_calls_and_pipes("\n".join(lines))
        assert calls >= 2 or has_pipe
        # prefix reaches back to the enclosing signature
        assert c["prefix_lines"][0].startswith("run <- function")
    # multi-statement runs are cut at complete statements only
    for c in C.candidate_blocks(make_bundle(MERGE_R + PIPE_R)):
        stmts = C.fragment_statements("\n".join(c["block"]))
        assert stmts is not None, "windows must never split a statement"
    # a run that does not parse cleanly (cut mid-statement) yields nothing
    mid = b"broken <- function(x) {\n  x %>%, head(2)\n  tail(1)\n}\n"
    assert C.fragment_statements("x %>%, head(2)\ntail(1)") is None
    assert C.candidate_blocks(make_bundle(mid)) == []


def test_make_synthetic_example_and_validator():
    cands = C.candidate_blocks(make_bundle(PIPE_R))
    ex = C.make_synthetic_example(
        dict(package="p", path="R/x.R", prefix_lines=cands[0]["prefix_lines"],
             block=cands[0]["block"], calls=9, pipe=True, fn="run"),
        "Filter, group and preview the frame", "mock/model")
    C.validate_example(ex)
    assert ex["family"] == "comment_to_code_synthetic"
    assert ex["generator"] == "mock/model"
    assert ex["prefix"][-1].strip() == "# Filter, group and preview the frame"
    assert ex["region_new"] == cands[0]["block"]
    assert C.noop_baseline_score(ex) == 0.0


def test_comment_gate_and_normalization():
    assert C._gate_ok("Compute the group means")
    assert not C._gate_ok("")
    assert not C._gate_ok("x" * 91)
    assert not C._gate_ok("a; b")
    assert not C._gate_ok("a <- b")
    assert not C._gate_ok("call(x)")
    assert not C._gate_ok("two\nlines")
    assert C.normalize_comment("#  Compute means") == "Compute means"
    assert C.normalize_comment('"quoted comment"') == "quoted comment"
    assert C.normalize_comment("   ") is None
    assert C.normalize_comment(None) is None


def test_validator_rejects_tampering():
    exs = C.extract_comment_pairs(make_bundle(REAL_R))
    ex = dict(exs[0])

    t1 = dict(ex)
    t1["region_new"] = list(t1["region_old"])
    t2 = dict(ex)
    t2["region_new"] = ["  a <- 1", "  ", "  b <- 2"]  # blank line inside
    t3 = dict(ex)
    t3["region_new"] = ["  a <- 1", "  # sneaky comment", "  b <- 2"]
    t4 = dict(ex)
    t4["cursor_idx"] = 3
    t5 = dict(ex)
    t5["event_diff"] = S.event_diff_for("R/test.R", 1, "a", "b")
    t6 = dict(ex)
    t6["prefix"] = list(ex["prefix"][:-1]) + ["  #' roxygen instead"]
    t7 = dict(ex)
    t7["region_new"] = ["  a <- 1", "  b <- 2} dropped"]  # does not re-parse
    for t in (t1, t2, t3, t4, t5, t6, t7):
        try:
            C.validate_example(t)
            raise SystemExit(f"validator accepted tampered example: {t}")
        except AssertionError:
            pass

    # synthetic variant must carry a generator and a gated comment
    cands = C.candidate_blocks(make_bundle(PIPE_R))
    base = dict(package="p", path="R/x.R", prefix_lines=cands[0]["prefix_lines"],
                block=cands[0]["block"], calls=9, pipe=True, fn="run")
    bad_gen = C.make_synthetic_example(base, "Fine comment text", "mock/model")
    del bad_gen["generator"]
    bad_cmt = C.make_synthetic_example(base, "mean(x) of the thing", "mock/model")
    for t in (bad_gen, bad_cmt):
        try:
            C.validate_example(t)
            raise SystemExit("validator accepted malformed synthetic example")
        except AssertionError:
            pass


def test_build_synthetic_with_mock_llm():
    orig = C.generate_comment
    calls = {"n": 0}

    def fake(code, k1, k2, zai_key=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return "bad ; comment", "mock/model"      # gate-rejected once
        if calls["n"] == 2:
            return "good descriptive comment", "mock/model"  # regeneration
        return None, ""                                # later blocks dropped

    C.generate_comment = fake
    try:
        cands = C.candidate_blocks(make_bundle(PIPE_R + MERGE_R))
        assert len(cands) >= 2
        for c in cands:
            c["package"], c["path"] = "testpkg", "R/test.R"
        out, api = C.build_synthetic(cands[:3], 2, "", "", verbose=False)
    finally:
        C.generate_comment = orig
    assert len(out) == 1
    ex = out[0]
    assert ex["prefix"][-1].strip() == "# good descriptive comment"
    assert ex["generator"] == "mock/model"
    C.validate_example(ex)
    assert api["gate_first_rejects"] == 1
    assert api["dropped"]["api"] == len(cands[:3]) - 1  # remaining got None
    assert api["dropped"]["gate"] == 0


def test_prompt_embeds_code_as_plain_text():
    # regression guard: the comment-generation prompt must embed the code
    # block as PLAIN newline-joined text; the Python list repr of the block
    # lines (as passed by build_synthetic -> generate_comment(cand["block"]))
    # must never leak into a prompt.
    block = ["res <- df %>% dplyr::filter(!is.na(g))",
             "out <- sum(res$g, na.rm = TRUE)"]
    sent = []
    orig_post = C._post

    def capture(url, api_key, payload, timeout, source):
        sent.append(payload["messages"][0]["content"])
        return '{"comment": "Filter missing groups and total them"}'

    C._post = capture
    try:
        C.call_opencode(block, "k")           # list input at the API boundary
        C.call_openrouter(block, "k")
        cmt, gen = C.generate_comment(block, "k1", "k2")  # full pipeline path
    finally:
        C._post = orig_post
    assert C._plain_code(block) == "\n".join(block)
    assert C._plain_code("\n".join(block)) == "\n".join(block)
    assert len(sent) == 3
    for prompt in sent:
        assert "\n".join(block) in prompt, "code must be embedded newline-joined"
        assert "['" not in prompt and "', '" not in prompt, \
            f"Python list-repr leaked into prompt: {prompt!r}"
        assert "dplyr::filter(!is.na(g))\nout <- sum(res$g" in prompt
    assert cmt == "Filter missing groups and total them"
    assert gen.startswith("opencode/")


def test_prefix_bindings_and_dedup():
    bound = C.prefix_bindings(["prep <- function(x, y = 1, ...) {",
                               "  keep <- 1", "  # now"])
    assert {"prep", "x", "y", "keep"} <= bound
    assert C.normalize_block(["  a <- 1 ", "  b <- 2"]) == \
        C.normalize_block(["a <- 1", "    b <- 2"])
    # density: 2 bodies, 1 with an intra-body comment
    dens = {"p1": dict(bodies=10, with_comment=3), "p2": dict(bodies=30, with_comment=0)}
    summ = C.density_summary(dens)
    assert summ["bodies_total"] == 40
    assert summ["bodies_with_intra_comment"] == 3
    assert summ["pct_with_intra_comment"] == 7.5
    assert summ["per_package_pct"]["n_packages"] == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
