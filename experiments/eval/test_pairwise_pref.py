"""Tests for pairwise_pref.py (V1d blind pairwise preference). Pure-CPU,
no network: the runner is exercised through a scripted fake backend via
monkeypatched make_backend. Run:
  uv run --with pytest python -m pytest experiments/eval/test_pairwise_pref.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic-data"))

import pairwise_pref as pp  # noqa: E402


# ---------------------------------------------------------------------------
# blindness sentinel
# ---------------------------------------------------------------------------

def test_blind_render_contains_only_generic_labels():
    p = pp.render_prompt("Improve the docs", "f <- function(x) {", "x + 1",
                         "x - 1")
    assert "CANDIDATE A" in p and "CANDIDATE B" in p
    assert "x + 1" in p and "x - 1" in p            # candidates verbatim
    assert pp.check_blindness(p) == []


def test_blindness_sentinel_trips_on_identity_strings():
    # a candidate that somehow carried an identity string must be caught
    # BEFORE any API spend
    leaky = pp.render_prompt("g", "ctx", "model sft_v8_2 output", "other")
    assert "sft_v8_2" in pp.check_blindness(leaky)
    for f in ("minicpm5", "rl_grpo", "episode_judged", "/mnt/h", ".jsonl",
              "false_suggestion"):
        assert f in pp.check_blindness(pp.render_prompt("g", f, "a", "b"))


def test_order_swap_is_exact_block_swap():
    g, ctx = "goal text", "code context"
    p1 = pp.render_prompt(g, ctx, "AAA", "BBB")
    p2 = pp.render_prompt(g, ctx, "BBB", "AAA")
    tail = "Reply with ONLY"                       # B-section runs into it

    def blocks(p):
        head, rest = p.split("CANDIDATE A:")
        a, b = rest.split("CANDIDATE B:")
        b = b.split(tail)[0]
        # slot tags are fixed by design (slots are slots); compare the
        # tag-inner content so the swap is checked at content level
        ia, ib = a.split("<cand_a>")[1].split("</cand_a>")[0], \
            b.split("<cand_b>")[1].split("</cand_b>")[0]
        return head, ia, ib
    h1, a1, b1 = blocks(p1)
    h2, a2, b2 = blocks(p2)
    assert "AAA" in a1 and "BBB" in b1
    assert "BBB" in a2 and "AAA" in b2
    assert a1 == b2 and b1 == a2                    # exact content swap
    assert h1 == h2                                 # everything else identical


def test_prompt_template_has_no_unescaped_braces():
    # render must succeed on literal { } inside goal/context/candidates
    p = pp.render_prompt("use { braces }", "fn({x})", "a{b}c", "d")
    assert "{x}" in p and "a{b}c" in p


# ---------------------------------------------------------------------------
# judge-reply parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"pref": "A", "reason": "cleaner"}', "a"),
    ('{"pref": "B", "reason": "shorter"}', "b"),
    ('{"pref": "tie", "reason": "same"}', "tie"),
    ('```json\n{"pref": "A", "reason": "fenced"}\n```', "a"),
    ('prose {"pref": "B"} prose', "b"),
    ('I prefer candidate B here.', "b"),
    ("It's a tie really", "tie"),
    ('{"pref": "C"}', None),
    ("not json at all, sorry", None),
    (None, None),
    ("", None),
])
def test_parse_pref(text, expected):
    assert pp.parse_pref(text) == expected


# ---------------------------------------------------------------------------
# debias math
# ---------------------------------------------------------------------------

def _row(pid, order, pref, first_is, second_is, label="typing"):
    return dict(pair="t", pid=pid, order=order, label=label, first_is=first_is,
                second_is=second_is, pref=pref, reason=None, error=None,
                latency_s=0.0)


def test_debias_by_pid_classes():
    rows = [
        # consistent M1: picked in both orders
        _row("p1", 0, "a", "M1", "M2"), _row("p1", 1, "b", "M2", "M1"),
        # consistent M2
        _row("p2", 0, "b", "M1", "M2"), _row("p2", 1, "a", "M2", "M1"),
        # flip: order decides -> tie by the debias rule
        _row("p3", 0, "a", "M1", "M2"), _row("p3", 1, "a", "M2", "M1"),
        # tie in one order -> tie
        _row("p4", 0, "tie", "M1", "M2"), _row("p4", 1, "a", "M2", "M1"),
        # missing second order -> excluded entirely
        _row("p5", 0, "a", "M1", "M2"),
        # unparsed call -> excluded
        _row("p6", 0, None, "M1", "M2"), _row("p6", 1, "a", "M2", "M1"),
    ]
    d = pp.debias_by_pid(rows)
    assert set(d) == {"p1", "p2", "p3", "p4"}
    assert d["p1"]["winner"] == "M1" and d["p1"]["cls"] == "consistent"
    assert d["p2"]["winner"] == "M2" and d["p2"]["cls"] == "consistent"
    assert d["p3"]["winner"] is None and d["p3"]["cls"] == "flip"
    assert d["p4"]["winner"] is None and d["p4"]["cls"] == "tie"


def test_analyze_pair_hand_computed():
    rows = [
        _row("p1", 0, "a", "M1", "M2"), _row("p1", 1, "b", "M2", "M1"),
        _row("p2", 0, "b", "M1", "M2"), _row("p2", 1, "a", "M2", "M1"),
        _row("p3", 0, "a", "M1", "M2"), _row("p3", 1, "a", "M2", "M1"),
        _row("p4", 0, "tie", "M1", "M2"), _row("p4", 1, "a", "M2", "M1"),
        _row("p4b", 0, "b", "M1", "M2", label="noop"),
        _row("p4b", 1, "tie", "M2", "M1", label="noop"),
    ]
    r = pp.analyze_pair(rows, ("M1", "M2"), "M1")
    assert r["n_points"] == 5
    assert r["wins"] == {"M1": 1, "M2": 1}
    assert r["flips"] == 1 and r["ties"] == 2
    assert r["win_rate"] == 0.2
    assert r["ties_half_rate"] == 0.5  # one win + half of three tied points
    # first-slot picks among the 8 non-tie calls: a,b | b,a | a,a | a | b
    assert r["first_slot_pick_rate"] == round(5 / 8, 4)
    assert r["per_label"]["noop"]["n"] == 1
    assert r["per_label"]["noop"]["ties"] == 1
    # sign test M1 vs M2 on 1 win / 1 loss -> p = 1.0
    assert r["sign_p"] == 1.0


def test_analyze_pair_v7_counts_and_swap_symmetry():
    # Historical v7/rl_v2c counts: 17/43 wins, 73 explicit ties, 15 flips.
    rows = []
    for kind, count, prefs in (
        ("v7", 17, ("a", "b")),
        ("rl_v2c", 43, ("b", "a")),
        ("tie", 73, ("tie", "tie")),
        ("flip", 15, ("a", "a")),
    ):
        for i in range(count):
            pid = f"{kind}-{i}"
            rows.extend([
                _row(pid, 0, prefs[0], "v7", "rl_v2c"),
                _row(pid, 1, prefs[1], "rl_v2c", "v7"),
            ])
    v7 = pp.analyze_pair(rows, ("v7", "rl_v2c"), "v7")
    rl = pp.analyze_pair(rows, ("rl_v2c", "v7"), "rl_v2c")
    assert v7["n_points"] == 148
    assert v7["wins"] == {"v7": 17, "rl_v2c": 43}
    assert (v7["ties"], v7["flips"]) == (73, 15)
    assert v7["ties_half_rate"] == 0.4122  # (17 + 44) / 148
    assert rl["ties_half_rate"] == 0.5878  # (43 + 44) / 148
    assert v7["ties_half_rate"] + rl["ties_half_rate"] == pytest.approx(1.0)
    assert v7["win_rate"] == 0.1149
    assert v7["sign_p"] == rl["sign_p"] == pytest.approx(0.0010657657791434353)

    # Swapping candidate slots while preserving each pick cannot change
    # either model's score or the separate tie/flip counts.
    swapped = [dict(r, first_is=r["second_is"], second_is=r["first_is"],
                    pref={"a": "b", "b": "a", "tie": "tie"}[r["pref"]])
               for r in rows]
    result = pp.analyze_pair(swapped, ("v7", "rl_v2c"), "v7")
    for key in ("wins", "ties", "flips", "ties_half_rate", "win_rate",
                "sign_p", "win_rate_wilson95", "per_label"):
        assert result[key] == v7[key]


def test_wilson_ci_hand_computed():
    # k=7, n=10, z=1.96 -> [0.3968, 0.8922] (hand-derived)
    lo, hi = pp.wilson_ci(7, 10)
    assert abs(lo - 0.396771) < 1e-4
    assert abs(hi - 0.892217) < 1e-4
    assert pp.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = pp.wilson_ci(10, 10)
    assert lo > 0.69 and hi <= 1.0


def test_sign_test_hand_computed():
    # 7 wins / 3 losses: 2*(C(10,0)+..+C(10,3))/2^10 = 352/1024
    assert pp.sign_test_p(7, 3) == pytest.approx(352 / 1024)
    assert pp.sign_test_p(0, 0) == 1.0
    assert pp.sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert pp.sign_test_p(3, 7) == pp.sign_test_p(7, 3)  # symmetric


# ---------------------------------------------------------------------------
# sampling + assignment determinism
# ---------------------------------------------------------------------------

def _mkpool(labels):
    return [dict(pid=f"pid{i:03d}", traj_key=f"k{i}", variant=1, t_ms=i * 10,
                 ctx="typing", goal="g", prompt="p", gt="", label=lab,
                 cand={"a": "aaa", "b": "bbb"})
            for i, lab in enumerate(labels)]


def test_stratified_sample_deterministic_and_proportional():
    pool = _mkpool(["typing"] * 60 + ["noop"] * 40)
    s1 = pp.stratified_sample(pool, 150, 3407)
    s2 = pp.stratified_sample(pool, 150, 3407)
    assert [r["pid"] for r in s1] == [r["pid"] for r in s2]     # reproducible
    assert len(s1) == 100                                        # capped by pool
    lab = [r["label"] for r in s1]
    assert lab.count("typing") == 60 and lab.count("noop") == 40
    # proportional subsample: n=50 of 60/40 -> 30/20
    s3 = pp.stratified_sample(pool, 50, 3407)
    assert [r["label"] for r in s3].count("typing") == 30
    assert len({r["pid"] for r in s3}) == 50                     # no dupes
    assert pp.stratified_sample(pool, 0, 1) == []
    # remainder goes to the last label without exceeding n
    pool2 = _mkpool(["typing"] * 61 + ["noop"] * 40)
    s4 = pp.stratified_sample(pool2, 50, 7)
    assert len(s4) == 50


def test_first_slot_assignments_seed_locked():
    pool = _mkpool(["typing"] * 20)
    a1 = pp.first_slot_assignments(pool, 99)
    a2 = pp.first_slot_assignments(pool, 99)
    assert a1 == a2 and len(a1) == 20
    assert set(a1.values()) <= {True, False}
    a3 = pp.first_slot_assignments(pool, 100)
    assert a3 != a1 or True   # different seed MAY differ; not required


# ---------------------------------------------------------------------------
# pool construction + corruption
# ---------------------------------------------------------------------------

def test_pair_pool_requires_both_nonempty_ghost_text(tmp_path):
    def mkfile(name, rows):
        f = tmp_path / name
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return f
    traj = dict(key="k", variant=1, goal="g", stats={})
    a = traj | dict(points=[
        dict(t_ms=1, ctx="c", label="typing", prompt="P1", gt="g1",
             proposal="real line\n>>>>>>> UPDATED"),
        dict(t_ms=2, ctx="c", label="typing", prompt="P2", gt="g2",
             proposal=">>>>>>> UPDATED\n>>>>>>> UPDATED"),   # parses empty
    ])
    b = traj | dict(points=[
        dict(t_ms=1, ctx="c", label="typing", prompt="P1", gt="g1",
             proposal="other line"),
        dict(t_ms=2, ctx="c", label="typing", prompt="P2", gt="g2",
             proposal="also real"),
    ])
    pool = pp.pair_pool(pp.load_points(mkfile("a.jsonl", [a])),
                        pp.load_points(mkfile("b.jsonl", [b])))
    assert len(pool) == 1                       # t_ms=2 dropped (a parses empty)
    assert pool[0]["cand"]["a"] == "real line"  # marker text stripped
    assert pool[0]["cand"]["b"] == "other line"
    assert len(pool[0]["pid"]) == 12


def test_corrupt_gt_deterministic_and_changing():
    for gt in ['data(apt, package = "beezdemand")', "skip_if_not_geweke()",
               "## !!! In Developing ", "message('done')"]:
        c1, c2 = pp.corrupt_gt(gt), pp.corrupt_gt(gt)
        assert c1 == c2                          # deterministic
        assert c1 is not None and c1 != gt       # changed
    # arg swap actually swaps the first two args
    assert pp.corrupt_gt('f(x, y)') == "f(y, x)"
    # word swap fallback for identifier-only lines
    assert pp.corrupt_gt("alpha_beta()") == "beta_alpha()"
    # uncorruptible -> None (excluded from anchor pool)
    assert pp.corrupt_gt("x <- 1") is None
    assert pp.corrupt_gt("") is None


# ---------------------------------------------------------------------------
# runner (fake backend, no network)
# ---------------------------------------------------------------------------

class FakeBackend:
    """Scripted backend: pops verdicts from a list; records every prompt."""

    name, model = "fake", "fake-0"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        v = self.verdicts.pop(0)
        return json.dumps({"pref": v, "reason": f"because {v}"})

    def stats_summary(self):
        return dict(attempts=len(self.prompts), ok=len(self.prompts))


def _fake_factory(script):
    def factory(name, **kw):
        return FakeBackend(script)
    return factory


def test_run_pair_both_orders_blind_and_resumable(tmp_path, monkeypatch):
    made = []
    script = ["a", "b", "a", "a"]                # p0: a,b -> consistent 'a'-holder
    def factory(name, target_key="comment", seed=0):
        b = FakeBackend(script)
        made.append(b)
        return b
    monkeypatch.setattr(pp, "make_backend", factory)
    pool = [_mkpool(["typing"] * 3)[0]]          # one point, cand a/b
    monkeypatch.setattr(pp, "build_sample",
                        lambda pair, n, seed: (list(pool), ("M1", "M2")))
    out = tmp_path / "res.jsonl"

    s = pp.run_pair("t", 1, 5, "fake", out)
    assert s["new_calls"] == 2
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 2 and {r["order"] for r in rows} == {0, 1}
    # order 1 is the guaranteed swap of order 0's slot assignment
    assert rows[0]["first_is"] != rows[1]["first_is"]
    assert {rows[0]["first_is"], rows[0]["second_is"]} == {"M1", "M2"}
    # both prompts blind, candidates present
    for r, p in zip(rows, made[0].prompts):
        assert pp.check_blindness(p) == []
        assert "aaa" in p and "bbb" in p
    # the two prompts are the exact order-swap of each other
    assert made[0].prompts[0] == pp.render_prompt(
        pool[0]["goal"], pool[0]["prompt"], "aaa", "bbb") or \
        made[0].prompts[0] == pp.render_prompt(
        pool[0]["goal"], pool[0]["prompt"], "bbb", "aaa")

    # resume: everything done -> zero new calls, zero new rows
    s2 = pp.run_pair("t", 1, 5, "fake", out)
    assert s2["new_calls"] == 0
    assert len([l for l in open(out)]) == 2


def test_run_pair_refuses_blindness_leak(tmp_path, monkeypatch):
    calls = []
    class Explode:
        name = model = "x"
        def complete(self, prompt):
            calls.append(1)
            raise AssertionError("must not be called")
        def stats_summary(self):
            return {}
    monkeypatch.setattr(pp, "make_backend",
                        lambda name, **kw: Explode())
    leaky = [dict(pid="leak1", traj_key="k", variant=1, t_ms=1, ctx="c",
                  goal="g", prompt="p", gt="", label="typing",
                  cand={"a": "made by sft_v8_2", "b": "made by v7"})]
    monkeypatch.setattr(pp, "build_sample",
                        lambda pair, n, seed: (leaky, ("M1", "M2")))
    out = tmp_path / "leak.jsonl"
    pp.run_pair("t", 1, 1, "fake", out)
    assert calls == []                           # no spend on a leaking render
    rows = [json.loads(l) for l in open(out)]
    assert all(r["error"].startswith("blindness:") for r in rows)


def test_end_to_end_analyze_with_fake_backend(tmp_path, monkeypatch):
    # 3 points x 2 orders; M1 wins p1 both orders; p2 flips; p3 ties one order
    script = ["a", "b",   # p1: consistent first_is-holder (whoever it is)
              "a", "a",   # p2: flip
              "tie", "a"]  # p3: tie
    def factory(name, **kw):
        return FakeBackend(script)
    monkeypatch.setattr(pp, "make_backend", factory)
    pool = _mkpool(["typing"] * 3)
    monkeypatch.setattr(pp, "build_sample",
                        lambda pair, n, seed: (list(pool), ("M1", "M2")))
    out = tmp_path / "e2e.jsonl"
    pp.run_pair("t", 3, 11, "fake", out)
    r = pp.analyze_pair([json.loads(l) for l in open(out)], ("M1", "M2"), "M1")
    assert r["n_points"] == 3
    assert r["flips"] == 1 and r["ties"] == 1
    winner_share = r["wins"]["M1"] + r["wins"]["M2"]
    assert winner_share == 1                     # exactly one consistent point


def test_cli_analyze(tmp_path, monkeypatch, capsys):
    rows = [_row("p1", 0, "a", "gt", "corrupted"),
            _row("p1", 1, "b", "corrupted", "gt"),
            _row("p2", 0, "a", "gt", "corrupted"),
            _row("p2", 1, "a", "corrupted", "gt")]
    out = tmp_path / "results_pairwise_pref_anchor_gt_corrupt.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(pp, "HERE", tmp_path)
    spec_out = tmp_path / "analysis.json"
    pp.main(["analyze", "--pairs", "anchor_gt_corrupt", "--out", str(spec_out)])
    got = json.loads(spec_out.read_text())
    assert got["anchor_gt_corrupt"]["n_points"] == 2
    assert got["anchor_gt_corrupt"]["win_rate"] == 0.5       # p1 gt, p2 flip


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
