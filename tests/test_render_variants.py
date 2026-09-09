"""Tests for experiments/eval/render_variants.py (H3-S0 pack).

Run:  cd <worktree root> && \
      uv run --with pytest python -m pytest tests/test_render_variants.py -q

Pure-CPU, deterministic, no network, no serving, no NAS. Covers the four
load-bearing contracts of the H3-S0 task:
  1. render/parse round-trips (prompt-level exact inverse + completion
     identity) for every variant on fixture examples;
  2. format_fail on garbage vs compliant completions (the zero-shot
     compliance check), including the must-FAIL zeta1 control;
  3. byte-level prefix stability across two keystroke-states of the same
     edit: prefix-dominant variants keep an identical pre-cursor-zone and a
     >512-token stable head (the K0 granularity floor), while zeta2's head
     collapses on the suffix-head-shift event (control);
  4. the --smoke entry point (mock: rendering only).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "experiments" / "eval" / "render_variants.py"
_spec = importlib.util.spec_from_file_location("render_variants", _SCRIPT)
rv = importlib.util.module_from_spec(_spec)
sys.modules["render_variants"] = rv
_spec.loader.exec_module(rv)

sys.path.insert(0, str(_SCRIPT.parent))
import run_eval  # noqa: E402  (byte-exactness comparisons)

CANDIDATES = ["v10_psmtail_merge", "v11_psmtail_merge_empty",
              "v05_psm_merge", "v06_psm_merge_empty", "v14_psm_merge_nohist"]
ALL = CANDIDATES + ["v13_zeta1_alpaca", "zeta2"]

FIXTURES = [rv.fixture_2k(), rv.fixture_small(), rv.fixture_scenario_row(),
            rv.midtyping_state(rv.fixture_small())]


def _bad_cursor_fixture():
    ex = rv.fixture_small()
    ex["cursor_idx"] = 99  # out of range: render omits the marker entirely
    return ex


# --------------------------------------------------------------- registry

def test_registry_shape():
    assert set(rv.VARIANTS) == set(ALL)
    for name, val in rv.VARIANTS.items():
        render, parse = val
        assert callable(render) and callable(parse)
        assert name in rv.STOP_STRINGS and name in rv.REQUIRED_MARKERS
        assert name in rv.VARIANT_INFO and name in rv.SECTION_ORDER
    # K4: deterministic stop-string terminators per variant
    assert rv.STOP_STRINGS["v13_zeta1_alpaca"] == [rv.END1]
    for name in CANDIDATES + ["zeta2"]:
        assert rv.STOP_STRINGS[name] == [rv.TERMINATOR]
    # axes pinned to the matrix §1.1 rows
    info = rv.VARIANT_INFO
    assert (info["v10_psmtail_merge"]["ordering"], info["v10_psmtail_merge"]["vocab"],
            info["v10_psmtail_merge"]["history"], info["v10_psmtail_merge"]["cursor"]) == \
        ("PSM-T", "MV1", "HP3", "CE2")
    assert (info["v05_psm_merge"]["ordering"], info["v05_psm_merge"]["history"]) == ("PSM", "HP2")
    assert (info["v06_psm_merge_empty"]["cursor"], info["v11_psmtail_merge_empty"]["cursor"]) == ("CE1", "CE1")
    assert info["v14_psm_merge_nohist"]["history"] == "HP0"
    assert info["zeta2"]["ordering"] == "SPM" and info["zeta2"]["history"] == "HP1"


def test_k2_paper_rule():
    for name in CANDIDATES:
        ok, fails = rv.k2_verdict(name)
        assert ok, f"{name} must pass K2, failed: {fails}"
    # the incumbent fails (iii) suffix-at-head; the control fails (i) too
    ok, fails = rv.k2_verdict("zeta2")
    assert not ok and any("suffix_at_head" in f for f in fails)
    ok, fails = rv.k2_verdict("v13_zeta1_alpaca")
    assert not ok and any("history_precedes_prefix" in f for f in fails)


# --------------------------------------------------------------- rendering

@pytest.mark.parametrize("name", ALL)
def test_markers_appear_exactly_once(name):
    ex = rv.fixture_2k()
    prompt = rv.VARIANTS[name][0](ex)
    if name == "v13_zeta1_alpaca":
        markers = [rv.START1, rv.END1, rv.CURSOR1]
    else:
        markers = [rv.FIM_SUFFIX, rv.REGION_OPEN, rv.REGION_CLOSE,
                   rv.FIM_MIDDLE, rv.CURSOR2]
        if name != "v14_psm_merge_nohist":
            markers.append(rv.HIST_HEADER)
    for m in markers:
        assert prompt.count(m) == 1, f"{name}: {m!r} x{prompt.count(m)}"
    assert rv.VARIANTS[name][0](ex) == prompt  # deterministic


def test_v10_generation_tail_byte_identical_to_zeta2():
    """Matrix §1.2a: V10 keeps the incumbent's exact generation tail."""
    ex = rv.fixture_2k()
    pz, p10 = rv.render_zeta2(ex), rv.render_v10(ex)
    tail = lambda p: p[p.index(rv.REGION_OPEN):]  # noqa: E731
    assert tail(p10) == tail(pz)


def test_v13_is_run_eval_zeta1():
    """The control must be the registry ancestor, byte-exact."""
    for ex in FIXTURES:
        full = dict(ex, suffix=ex.get("suffix", []), prefix=ex.get("prefix", []),
                    region_old=ex.get("region_old", []))
        assert rv.VARIANTS["v13_zeta1_alpaca"][0](ex) == run_eval.render_zeta1(full)


def test_merge_variants_reorder_the_same_marker_multiset():
    """Matrix §1.2c: MV1 candidates carry the zeta2 marker multiset, so their
    structural byte count equals the incumbent's (delta 0)."""
    ex = rv.fixture_2k()
    ref = rv.structural_bytes(rv.render_zeta2(ex), ex, "zeta2")
    for name in ["v05_psm_merge", "v10_psmtail_merge"]:
        assert rv.structural_bytes(rv.VARIANTS[name][0](ex), ex, name) == ref
    # CE1 twins: the cursor marker becomes its own line instead of riding
    # the last region line -> exactly +1 structural byte (the newline)
    for name in ["v06_psm_merge_empty", "v11_psmtail_merge_empty"]:
        assert rv.structural_bytes(rv.VARIANTS[name][0](ex), ex, name) == ref + 1
    # HP0 drops the 36-byte history header, its blank separator line and
    # the join newline that glued the block in (38 structural bytes total)
    assert (rv.structural_bytes(rv.render_v14(ex), ex, "v14_psm_merge_nohist")
            == ref - len(rv.HIST_HEADER) - 2)


def test_structural_overhead_k3():
    ex = rv.fixture_2k()
    merge = rv.structural_bytes(rv.render_v05(ex), ex, "v05_psm_merge")
    v13 = rv.structural_bytes(rv.VARIANTS["v13_zeta1_alpaca"][0](ex), ex,
                              "v13_zeta1_alpaca")
    assert v13 >= 3 * merge                      # ~384 B vs ~128 B
    # at the matrix's ~3.5 B/tok estimate the control crosses the 5% K3 line
    # (110 tok / 2048); at the conservative 4.0 B/tok house mean it sits just
    # under — recorded honestly instead of hard-asserting one estimate
    assert v13 / 3.5 / rv.CTX_BUDGET_TOK > 0.05
    assert 0.03 < v13 / rv.BYTES_PER_TOK / rv.CTX_BUDGET_TOK < 0.06


# --------------------------------------------------------------- round-trips

@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("ex", FIXTURES + [_bad_cursor_fixture()],
                         ids=["2k", "small", "scenario-row", "midtyping",
                              "bad-cursor"])
def test_prompt_level_inverse(name, ex):
    """parse_prompt(render(ex)) recovers the canonical example exactly."""
    render, _ = rv.VARIANTS[name]
    assert rv.parse_prompt(render(ex), name) == rv.canonical_example(ex, name)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("ex", FIXTURES, ids=["2k", "small", "scenario-row",
                                              "midtyping"])
def test_completion_identity(name, ex):
    """The parser is the exact inverse of the render's completion contract."""
    _, parse = rv.VARIANTS[name]
    assert parse(rv.completion(ex, name)) == rv.norm(ex["region_new"])


def test_parse_merge_mirrors_run_eval_zeta2():
    """K1 comparability: the shared merge parse IS parse_pred's zeta2 rule."""
    texts = ["a\n>>>>>>> UPDATED", "```\nb\n>>>>>>> UPDATED",
             "c<|user_cursor|>\n>>>>>>> UPDATED", "no terminator at all",
             "  x  \n\n>>>>>>> UPDATED\n"]
    for t in texts:
        assert rv.parse_merge(t) == run_eval.parse_pred("zeta2", t)


def test_ce1_collapses_region_and_is_keystroke_invariant():
    ex = rv.fixture_2k()
    for name in ["v06_psm_merge_empty", "v11_psmtail_merge_empty"]:
        prompt = rv.VARIANTS[name][0](ex)
        assert rv.REGION_OPEN + "\n" + rv.CURSOR2 + "\n" + rv.REGION_CLOSE in prompt
        canon = rv.canonical_example(ex, name)
        assert canon["region_old"] == [] and canon["cursor_idx"] == 0
        assert canon["cursor_encoding"] == "CE1"
    a, b = rv.keystroke_pair(ex, "typing")
    assert rv.render_v06(a) == rv.render_v06(b)   # no partial is encoded


# --------------------------------------------------------------- format_fail

def test_format_fail_on_garbage():
    f = rv.format_fail
    assert f("") == (True, "empty")
    assert f(None) == (True, "empty")
    assert f("   \n  \n") == (True, "empty")
    assert f("The quick brown fox jumps over the lazy dog") == (True, "no_markers")
    assert f("The quick brown fox jumps", "v05_psm_merge") == (True, "no_markers")
    # terminator alone delimits an empty region
    assert f(">>>>>>> UPDATED", "v10_psmtail_merge") == (True, "empty_region")
    # a stop-served clean region carries no markers by construction
    assert f("mean(x, na.rm = TRUE)", "v10_psmtail_merge", stopped=True) == (False, "ok")
    assert f("mean(x, na.rm = TRUE)", "v10_psmtail_merge", stopped=False) == \
        (True, "no_markers")
    # prose under a consumed stop string parses as a "region" -> NOT a format
    # fail (scores exact=0 instead; the pre-registered run_eval agg metric)
    assert f("random prose", "v05_psm_merge", stopped=True) == (False, "ok")


def test_format_fail_zeta1_control():
    f = rv.format_fail
    assert f("no tags at all", "v13_zeta1_alpaca") == (True, "no_markers")
    # END1 present but START1 missing: marker evidence yes, parse None
    assert f("x <- 1" + rv.END1, "v13_zeta1_alpaca") == (True, "unparseable")
    assert f(rv.START1 + "y <- 1" + rv.END1, "v13_zeta1_alpaca") == (False, "ok")
    with pytest.raises(KeyError):
        f("x", "no_such_variant")


@pytest.mark.parametrize("name", ALL)
def test_format_fail_accepts_canonical_completions(name):
    ex = rv.fixture_small()
    assert rv.format_fail(rv.completion(ex, name), name) == (False, "ok")


# ------------------------------------------------------- prefix stability

def test_prefix_stability_typing_event():
    """Insert-before-cursor (matrix arm a): every prefix-dominant variant
    keeps an identical pre-cursor-zone AND a >512-token stable head."""
    ex = rv.fixture_2k()
    for name in CANDIDATES:
        m = rv.prefix_stability(ex, name, "typing")
        assert m["prezone_identical"], name
        assert m["over_k0_floor"], (name, m)
    # honest positive control: the incumbent also localizes pure typing
    # (§0.2 reading 1, arm a measured 80% cached)
    m = rv.prefix_stability(ex, "zeta2", "typing")
    assert m["prezone_identical"] and m["over_k0_floor"]


def test_prefix_stability_cursor_advance_control():
    """Suffix-head shift (the m2/backspace class): candidates keep a
    >512-token stable head; zeta2's head collapses (the DIFFERS control)."""
    ex = rv.fixture_2k()
    for name in ["v05_psm_merge", "v06_psm_merge_empty", "v14_psm_merge_nohist"]:
        m = rv.prefix_stability(ex, name, "cursor_advance")
        assert m["prezone_identical"], name
        assert m["over_k0_floor"], (name, m)
    for name in ["v10_psmtail_merge", "v11_psmtail_merge_empty"]:
        m = rv.prefix_stability(ex, name, "cursor_advance")
        # the suffix sits mid-prompt, so the pre-zone is not byte-identical,
        # but the pinned file-prefix head alone clears the K0 floor
        assert not m["prezone_identical"], name
        assert m["over_k0_floor"], (name, m)
    m = rv.prefix_stability(ex, "zeta2", "cursor_advance")
    assert not m["over_k0_floor"] and not m["prezone_identical"]
    assert m["stable_bytes"] < 32  # divergence right after <[fim-suffix]>


def test_prefix_stability_edit_event_k2():
    """Debounced edit near the cursor (history ring rotates + prefix line
    changes): candidates keep the file-prefix head; the incumbent keeps only
    the suffix block (~474 tok: below one 512-tok granule = zero reuse); the
    alpaca control's history-first head dies (K2 sub-rule i)."""
    ex = rv.fixture_2k()
    for name in CANDIDATES:
        m = rv.prefix_stability(ex, name, "edit_event")
        assert m["over_k0_floor"], (name, m)
    assert not rv.prefix_stability(ex, "zeta2", "edit_event")["over_k0_floor"]
    assert not rv.prefix_stability(ex, "v13_zeta1_alpaca", "edit_event")["over_k0_floor"]


def test_keystroke_pair_geometry():
    ex = rv.fixture_2k()
    a, b = rv.keystroke_pair(ex, "typing")
    assert b["region_old"][-1] == a["region_old"][-1] + "e"
    assert b["suffix"] == a["suffix"]              # post-cursor text unchanged
    a, b = rv.keystroke_pair(ex, "cursor_advance")
    assert b["suffix"][0] == a["suffix"][0][1:]    # suffix head shifts 1 byte
    assert b["suffix"][1:] == a["suffix"][1:]
    a, b = rv.keystroke_pair(ex, "edit_event")
    assert "touched" in b["prefix"][-31]
    assert b["event_diff"] != a["event_diff"]
    assert "@@ -41,3 +41,4 @@" in b["event_diff"]  # rotated: new hunk appended
    with pytest.raises(ValueError):
        rv.keystroke_pair(rv.fixture_small(), "bogus")


def test_est_tokens_is_conservative():
    assert rv.est_tokens("") == 0.0
    # 512-token floor must NOT be cleared by a 2K-byte marker-only head
    assert rv.est_tokens("x" * 2048) <= 512
    assert rv.est_tokens("x" * 4096) > 512


# --------------------------------------------------------------- smoke (mock)

def test_smoke_cli(capsys=None):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--smoke"],
        capture_output=True, text=True, timeout=120, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ALL:
        assert name in proc.stdout
    last = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")][-1]
    payload = json.loads(last)
    assert set(payload) == {"smoke", "rows", "prefix_stability_2k", "k0_floor_tok"}
    assert len(payload["rows"]) == len(ALL) * len(rv.SMOKE_FIXTURES)
    assert payload["k0_floor_tok"] == rv.K0_FLOOR_TOK
    st = {r["variant"]: r for r in payload["prefix_stability_2k"]}
    for name in CANDIDATES:
        for kind in ("typing", "cursor_advance", "edit_event"):
            assert st[name][kind]["over_k0_floor"], (name, kind)
    assert not st["zeta2"]["cursor_advance"]["over_k0_floor"]
    # --json mode emits only machine-readable lines
    proc2 = subprocess.run([sys.executable, str(_SCRIPT), "--smoke", "--json"],
                           capture_output=True, text=True, timeout=120, cwd=str(REPO))
    assert proc2.returncode == 0
    json.loads(proc2.stdout.strip().splitlines()[-1])
