"""Unit tests for the H1 rig. CPU-only, no network, no server.

  uv run --with pytest python -m pytest experiments/harness_search/test_harness_search.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "post-processing"))
sys.path.insert(0, str(HERE.parent / "synthetic-data"))

import pytest

import harness_config as HC
import rscan
import render as R
import parsep as P
import scorer as SC
import methods as ME
import server as SRV


# ---------------------------------------------------------------- carve ----
def test_carve_disjoint_and_deterministic():
    rows, manifest = ME.__dict__.get("_carved", (None, None)) or (None, None)
    import carve_d_harness as C
    rows, manifest = C.carve()
    assert len(rows) == 256
    eval_pkgs, eval_prompts = C.eval_split_sets()
    for r in rows:
        assert r["package"] not in eval_pkgs[r["family"]]
        assert r["_prompt"] not in eval_prompts[r["family"]]
        C.scenarios.validate_example(r)  # every carved row must validate
    rows2, _ = C.carve()
    ids1 = [C.row_id(r["_prompt"]) for r in rows]
    ids2 = [C.row_id(r["_prompt"]) for r in rows2]
    assert ids1 == ids2  # seed-locked
    # manifest row coverage
    mf_rows = [m for f in manifest["per_family"].values() for m in f["rows"]]
    assert sorted(m["id"] for m in mf_rows) == sorted(ids1)


def test_carve_on_disk_matches():
    rows = ME.load_d_harness()
    assert len(rows) == 256
    assert all("id" in r and r["_prompt"] for r in rows)


# ------------------------------------------------------------- parity ------
def test_default_render_is_edit_row_byte_identical():
    from assemble_sft_v2 import edit_row
    import carve_d_harness as C
    for fam in C.FAMILIES:
        n = 0
        for line in open(C.SCEN_DIR / f"{fam}.jsonl"):
            row = json.loads(line)
            rr = edit_row(dict(row), fam, row["package"])
            if rr is None:
                continue
            mine, _meta = R.render_scenario(row, HC.DEFAULT_CONFIG, HC.DEFAULT_TEXTS)
            assert mine == rr["prompt"]
            n += 1
            if n >= 40:
                break


def test_default_noop_render_is_extension_port():
    import eval_noop_fp as EN
    cases = EN.build_cases(corpus_n=4, seed=7)
    for c in cases[:24]:
        mine, _ = R.build_scoped_prompt(c.lines, c.cursor_line, c.cursor_char,
                                        c.rel_path, HC.DEFAULT_CONFIG,
                                        HC.DEFAULT_TEXTS)
        ref, _trunc = EN.build_prompt(c.lines, c.cursor_line, c.cursor_char,
                                      c.rel_path)
        assert mine == ref


def test_knobs_change_render():
    # a prefix window WITH top-level named functions (outline exercisable)
    row = dict(family="rename_propagation", package="p", path="R/a.R",
               prefix=["#' doc", "alpha <- function(a) {", "  a", "}",
                       "beta <- function(b) {", "  b", "}"],
               suffix=[], region_old=["  foo <- 1"],
               region_new=["  foo2 <- 1"], cursor_idx=0,
               event_diff='User edited "R/a.R":\n\n```diff\n@@ -1 +1 @@\n-a\n+b\n```',
               note="")
    base, _ = R.render_scenario(row, HC.DEFAULT_CONFIG, HC.DEFAULT_TEXTS)
    outline_cfg = dict(HC.DEFAULT_CONFIG, outline=[20, 800])
    with_outline, meta = R.render_scenario(row, outline_cfg, HC.DEFAULT_TEXTS)
    assert with_outline != base and meta["outline_lines"] > 0
    assert "<|outline|>" in with_outline
    assert "2 alpha" in with_outline and "5 beta" in with_outline
    gepa = dict(HC.DEFAULT_TEXTS, instruction_line="Predict the full region.")
    with_instr, _ = R.render_scenario(row, HC.DEFAULT_CONFIG, gepa)
    assert with_instr.count("Predict the full region.") == 1
    assert with_instr.split("<<<<<<< CURRENT")[-2].strip().endswith(
        "Predict the full region.")


def test_cap_truncates_long_prefix_keep_tail():
    long_prefix = [f"  x{i} <- {i}  # {'y' * 90}" for i in range(200)]
    row = dict(family="rename_propagation", package="p", path="R/a.R",
               prefix=long_prefix, suffix=[], region_old=["  foo <- 1"],
               region_new=["  foo2 <- 1"], cursor_idx=0,
               event_diff='User edited "R/a.R":\n\n```diff\n@@ -1 +1 @@\n-a\n+b\n```',
               note="")
    small = dict(HC.DEFAULT_CONFIG, prefix_suffix_cap=2000)
    p, meta = R.render_scenario(row, small, HC.DEFAULT_TEXTS)
    assert meta["truncated_prefix"] > 0
    kept = p.split("<filename>R/a.R\n", 1)[1].split("\n<<<<<<< CURRENT")[0]
    assert kept.endswith("  foo <- 1"[:0] + long_prefix[-1]) or \
        long_prefix[-1] in kept  # tail kept
    assert long_prefix[0] not in p  # head dropped


# ---------------------------------------------------------------- rscan ----
R_SNIPPET = """#' Title
f_outer <- function(x) {
  a <- 1
  inner <- function(y) {
    y + a
  }
  inner(x)
}

g_top <- function(z) {
  z * 2
}
""".splitlines()


def test_rscan_enclosing_and_outline():
    lines = list(R_SNIPPET)
    # cursor on "    y + a" (line 4 0-based): innermost enclosing = inner
    pin = rscan.find_enclosing_function_by_scan(lines, 4)
    assert pin is not None and pin[0] == 3 and pin[2] == "inner"
    # cursor inside f_outer but not inner
    pin2 = rscan.find_enclosing_function_by_scan(lines, 2)
    assert pin2 is not None and pin2[2] == "f_outer"
    # cursor ON inner's closing brace (line 5): inner is closed at the
    # cursor -> enclosing is f_outer (inside set BEFORE the breaks)
    pin3 = rscan.find_enclosing_function_by_scan(lines, 5)
    assert pin3[2] == "f_outer"
    # cursor at top level (the blank line between functions)
    assert rscan.find_enclosing_function_by_scan(lines, 8) is None
    ol = rscan.outline_from_scan(lines)
    assert [e[1] for e in ol] == ["f_outer", "g_top"]
    fmt = rscan.format_outline(ol, 60, 1500)
    assert fmt[0].startswith("2 f_outer")


def test_rscan_comment_and_string_blinding():
    line = '  s <- "func { fake"  # function( {'
    assert rscan.clean_r_line(line).count("{") == 0
    assert "function" not in rscan.clean_r_line(line).split("#")[0].replace('""', "")


def test_scope_pin_cap_fallback():
    lines = list(R_SNIPPET)
    scope = rscan.build_scope(lines, 2, 50, "off")  # tiny pin cap
    assert scope["pin"] is None  # cannot fit -> outline-only fallback
    scope2 = rscan.build_scope(lines, 2, 4000, [60, 1500])
    assert scope2["pin"] is not None
    # doc rule 1: pinned function's entry dropped from outline
    names = [e.split(" ", 1)[1] for e in scope2["outline"]]
    assert "f_outer" not in names


# ---------------------------------------------------------------- parse ----
def test_parse_prediction_toggles():
    cfg_on = dict(HC.DEFAULT_CONFIG)               # both toggles on
    cfg_off = dict(HC.DEFAULT_CONFIG, parse_marker_drop=False,
                   parse_rep_cut=False)
    raw = "<<<<<<< CURRENT\n=======\nline A\nline A\nline A\n>>>>>>> UPDATED junk"
    assert P.parse_prediction(raw, cfg_on) == ["line A", "line A"]
    assert P.parse_prediction(raw, cfg_off) == [
        "<<<<<<< CURRENT", "=======", "line A", "line A", "line A"]
    assert P.parse_prediction("  x <- 1\r\n\r\n", cfg_on) == ["  x <- 1"]


def test_post_heuristics():
    row = dict(region_old=["#' some comment"], cursor_idx=13)
    cfg = dict(HC.DEFAULT_CONFIG)
    pred = ["mean(x, na.rm = TRUE)"]
    out = P.apply_post_heuristics(pred, row, cfg)
    assert out == ["", "mean(x, na.rm = TRUE)"]  # comment heuristic
    out2 = P.apply_post_heuristics(pred, row, dict(cfg, comment_heuristic=False))
    assert out2 == pred
    # full_region_replace OFF glues the typed partial
    row2 = dict(region_old=["  sum(x) / len"], cursor_idx=15)
    pred2 = ["  sum(x) / length(x)"]
    glued = P.apply_post_heuristics(
        pred2, row2, dict(cfg, comment_heuristic=False, full_region_replace=False))
    assert glued == ["  sum(x) / len" + "  sum(x) / length(x)".lstrip()]


# --------------------------------------------------------------- scorer ----
def _fake_row(fam="rename_propagation", rid="r1"):
    return dict(id=rid, family=fam, package="p", path="R/a.R",
                prefix=["a <- 1"], suffix=[], region_old=["  foo <- 1"],
                region_new=["  foo2 <- 1"], cursor_idx=0,
                event_diff='User edited "R/a.R":\n\n```diff\n@@ -1 +1 @@\n-a\n+b\n```',
                note="")


def test_validator_integration_both_rollouts_rule():
    row = _fake_row()
    good = "  foo2 <- 1" + "\n>>>>>>> UPDATED"
    bad = "  bar <- 2" + "\n>>>>>>> UPDATED"
    sc_on = SC.row_outcome(row, [dict(text=good, latency=1.0),
                                 dict(text=good, latency=1.2)],
                           HC.DEFAULT_CONFIG)
    assert sc_on["passed"] == 1 and sc_on["unstable"] == 0
    sc_unst = SC.row_outcome(row, [dict(text=good, latency=1.0),
                                   dict(text=bad, latency=1.0)],
                             HC.DEFAULT_CONFIG)
    assert sc_unst["passed"] == 0 and sc_unst["unstable"] == 1
    sc_off = SC.row_outcome(row, [dict(text=bad, latency=1.0),
                                  dict(text=bad, latency=1.0)],
                            HC.DEFAULT_CONFIG)
    assert sc_off["passed"] == 0 and sc_off["unstable"] == 0


def test_score_candidate_and_guardrails():
    rows = [_fake_row(rid=f"r{i}") for i in range(10)]
    # 6 rows pass both, 2 fail both, 2 unstable -> exact .6, unstable .2
    rolls = {}
    for i, r in enumerate(rows):
        good = "  foo2 <- 1"
        bad = "  nope <- 9"
        if i < 6:
            pair = [dict(text=good, latency=1.0), dict(text=good, latency=1.0)]
        elif i < 8:
            pair = [dict(text=bad, latency=1.0), dict(text=bad, latency=1.0)]
        else:
            pair = [dict(text=good, latency=1.0), dict(text=bad, latency=1.0)]
        rolls[r["id"]] = pair
    sc = SC.score_candidate(rows, rolls, HC.DEFAULT_CONFIG)
    assert sc["exact_pass"] == pytest.approx(0.6)
    assert sc["unstable_frac"] == pytest.approx(0.2)
    g = SC.guardrails(sc, dict(proposal_rate=0.10),
                      dict(noop_rate=0.10, p95_latency_s=2.0))
    assert g["noop_ok"] and g["latency_ok"]
    g2 = SC.guardrails(dict(p95_latency_s=2.61), dict(proposal_rate=0.135),
                       dict(noop_rate=0.10, p95_latency_s=2.0))
    assert not g2["noop_ok"] and not g2["latency_ok"]  # +2pp and 1.3x edges
    g3 = SC.guardrails(dict(p95_latency_s=2.6), dict(proposal_rate=0.12),
                       dict(noop_rate=0.10, p95_latency_s=2.0))
    assert g3["noop_ok"] and g3["latency_ok"]  # exactly at the lines passes


def test_selection_key_ordering():
    a = dict(exact_pass=0.5, unstable_frac=0.1, p95_latency_s=2.0)
    b = dict(exact_pass=0.5, unstable_frac=0.05, p95_latency_s=9.0)
    c = dict(exact_pass=0.6, unstable_frac=0.9, p95_latency_s=9.0)
    assert SC.selection_key(b) > SC.selection_key(a)  # fewer unstable wins
    assert SC.selection_key(c) > SC.selection_key(b)  # exact dominates


# -------------------------------------------------------------- config -----
def test_config_validation_and_fingerprint():
    with pytest.raises(ValueError):
        HC.validate_config(dict(HC.DEFAULT_CONFIG, max_tokens=999))
    with pytest.raises(ValueError):
        HC.validate_config(dict(HC.DEFAULT_CONFIG, bogus=1))
    cfg = HC.validate_config(HC.DEFAULT_CONFIG)
    assert cfg["pin"] == "off"
    assert HC.fingerprint(cfg) == HC.fingerprint(dict(HC.DEFAULT_CONFIG))
    assert HC.fingerprint(cfg) != HC.fingerprint(
        dict(HC.DEFAULT_CONFIG, max_tokens=640))
    t = HC.validate_texts(dict(instruction_line="ok text"))
    assert t["checklist_line"] == ""
    with pytest.raises(ValueError):
        HC.validate_texts(dict(instruction_line="has <<<<<<< CURRENT inside"))
    with pytest.raises(ValueError):
        HC.validate_texts(dict(instruction_line="x" * 300))


# -------------------------------------------------------------- budget -----
class _FakeClient(SRV.PairClient):
    """Offline PairClient: _do_complete overridden, same ledger math."""

    def __init__(self):
        super().__init__(port=0)
        self.n = 0

    def _do_complete(self, prompt, max_tokens, stops):
        self.n += 1
        return f"pred-{self.n}", 0.5 + 0.01 * self.n


def test_pair_client_ledger_and_cache():
    c = _FakeClient()
    a1 = c.pair("p1", ["s"], 320)
    assert c.ledger["new_completions"] == 2 and len(a1) == 2
    a2 = c.pair("p1", ["s"], 320)
    assert a2 == a1 and c.ledger["new_completions"] == 2
    assert c.ledger["pair_reuses"] == 1
    c.pair("p1", ["s"], 128)  # different max_tokens -> new key
    assert c.ledger["new_completions"] == 4
    c.single("p1", ["s"], 320)
    assert c.ledger["pair_reuses"] == 2  # single reuses the pair
    c.single("pz", ["s"], 320)
    assert c.ledger["new_completions"] == 5


def test_proposer_usage_normalization():
    from proposer import _norm_usage
    u = _norm_usage(dict(prompt_tokens=100, completion_tokens=50,
                         total_tokens=150,
                         completion_tokens_details=dict(reasoning_tokens=30)))
    assert u == dict(prompt_tokens=100, completion_tokens=50,
                     total_tokens=150, reasoning_tokens=30)
    u2 = _norm_usage(dict(input_tokens=7, output_tokens=3))
    assert u2["prompt_tokens"] == 7 and u2["completion_tokens"] == 3
    assert _norm_usage(None) == {}


def test_fallback_deterministic_and_valid():
    import random
    rng = random.Random("hill-1")
    parents = [dict(HC.DEFAULT_CONFIG)]
    c1 = ME.fallback_config(rng, parents)
    rng2 = random.Random("hill-1")
    c2 = ME.fallback_config(rng2, [dict(HC.DEFAULT_CONFIG)])
    assert c1 == c2
    HC.validate_config(c1)
    g = ME.fallback_gepa(random.Random("gepa-1"), [dict(HC.DEFAULT_TEXTS)])
    HC.validate_texts(g)


def test_evaluator_budget_deltas():
    """Evaluator must attribute exactly the completions its client spent."""
    rows = [dict(_fake_row(rid=f"r{i}"), prefix=[f"a{i} <- {i}"])
            for i in range(3)]
    cases = []

    class Case:
        id = "c0"
        cls = "a_after_close_brace"
        lines = ["f <- function(x) {", "  x", "}"]
        cursor_line = 2
        cursor_char = 1
        rel_path = "R/a.R"

    cases.append(Case())
    client = _FakeClient()
    ev = ME.Evaluator(rows, cases, client,
                      baseline=dict(p95_latency_s=99.0, noop_rate=0.0))
    out = ev.evaluate(HC.DEFAULT_CONFIG, HC.DEFAULT_TEXTS, "t")
    # 3 scenario pairs + 1 noop pair = 8 completions
    assert out["completions"] == 8
    assert out["pair_reuses"] == 0
    ev.evaluate(HC.DEFAULT_CONFIG, HC.DEFAULT_TEXTS, "t2")  # all cache hits
    out2 = ev.evaluate(dict(HC.DEFAULT_CONFIG, parse_rep_cut=False), None, "t3")
    assert out2["completions"] == 0  # parse-only knob: full reuse
    assert out2["pair_reuses"] == 4  # 3 scenario rows + 1 noop case
