import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag1_gate1 as g


def feat(mean_lp, **over):
    f = {name: 0.5 for name in g.FEATURES}
    f.update(mean_lp=mean_lp, min_lp=mean_lp, first_lp=mean_lp, last_lp=mean_lp,
             pred_len=2, stop_is_word=1.0, stop_is_eos=0.0, stop_is_limit=0.0)
    f.update(over)
    return f


def row(kind, cid, mean_lp, label, feature_ms=0.01, predicted_ms=50.0, drift=False):
    return {"arm": g.ARM, "kind": kind, "id": cid, "features": feat(mean_lp),
            "feature_ms": feature_ms, "predicted_ms": predicted_ms, "n_tokens": 2,
            "label": label, "regen": {}, "drift": drift}


def fp_label(cls="c_stmt_line_end"):
    return {"proposal": True, "expectation": "no_proposal", "cls": cls}


def ok_label():
    return {"proposal": False, "expectation": "no_proposal", "cls": "a_after_close_brace"}


def exact_label(family="rename_propagation"):
    return {"exact": True, "valid_pass": True, "fail_kind": None,
            "valid_reason": None, "family": family}


def write_rows(tmp_path, rows):
    (tmp_path / "logprob-features.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))


# --- label join -----------------------------------------------------------

def test_join_labels_by_case_id_not_position(tmp_path):
    archive = tmp_path / "archive.jsonl"
    rows_in = [
        {"arm": g.ARM, "kind": "noop", "id": "b", **fp_label()},
        {"arm": "Q6_K_imatrix", "kind": "noop", "id": "x", **fp_label()},
        {"arm": g.ARM, "kind": "noop", "id": "a", **ok_label()},
        {"arm": g.ARM, "kind": "scenario", "id": "s", **exact_label()},
    ]
    archive.write_text("".join(json.dumps(r) + "\n" for r in rows_in))
    labels = g.load_labels(archive)
    assert set(labels) == {("noop", "b"), ("noop", "a"), ("scenario", "s")}
    cases = [{"kind": "noop", "id": "a", "prompt": ""},
             {"kind": "noop", "id": "b", "prompt": ""},
             {"kind": "scenario", "id": "s", "prompt": "",
              "scenario": {"region_new": ["z"]}}]
    responses = [
        {"arm": g.ARM, "kind": "noop", "id": "b", "at": 0.0,
         "data": g.mock_response([-2.0, -2.0], "word", 50.0, "x")},
        {"arm": g.ARM, "kind": "noop", "id": "a", "at": 0.0,
         "data": g.mock_response([-0.2, -0.2], "word", 50.0, "")},
        {"arm": g.ARM, "kind": "scenario", "id": "s", "at": 0.0,
         "data": g.mock_response([-0.1, -0.1], "word", 50.0, "z")},
    ]
    joined = g.build_feature_rows(responses, cases, labels)
    by_id = {r["id"]: r for r in joined}
    assert by_id["a"]["label"]["proposal"] is False
    assert by_id["b"]["label"]["cls"] == "c_stmt_line_end"
    assert by_id["s"]["label"]["exact"] is True
    assert not any(r["drift"] for r in joined)   # order differed; join by id


def test_join_raises_on_missing_banked_label(tmp_path):
    labels = {("noop", "a"): fp_label()}
    cases = [{"kind": "noop", "id": "a", "prompt": ""},
             {"kind": "noop", "id": "ghost", "prompt": ""}]
    with pytest.raises(ValueError, match="missing banked label"):
        g.build_feature_rows(
            [{"kind": "noop", "id": "ghost", "data": {}}], cases, labels)


# --- feature extraction ---------------------------------------------------

def test_features_from_response_math():
    data = {
        "tokens": [5, 7],
        "stop_type": "limit",
        "tokens_predicted": 2,
        "completion_probabilities": [
            [{"id": 999, "token": "a", "bytes": [], "logprob": -0.5},
             {"id": 5, "token": "b", "bytes": [], "logprob": -1.0}],
            [{"id": 7, "token": "c", "bytes": [], "logprob": -2.0},
             {"id": 8, "token": "d", "bytes": [], "logprob": -3.0}],
        ],
        "timings": {"predicted_ms": 10.0},
    }
    f, ms = g.features_from_response(data)
    assert f["mean_lp"] == pytest.approx(-1.5)      # sampled ids 5 and 7
    assert f["min_lp"] == pytest.approx(-2.0)
    assert f["first_lp"] == pytest.approx(-1.0)     # id match, not entry 0
    assert f["last_lp"] == pytest.approx(-2.0)
    assert f["stop_is_limit"] == 1.0 and f["stop_is_word"] == 0.0
    assert f["pred_len"] == 2
    # per-token entropy over the renormalized top-N candidates, then averaged
    import math
    def ent(lps):
        z = sum(math.exp(v) for v in lps)
        return -sum(math.exp(v) / z * math.log(math.exp(v) / z) for v in lps)
    h = (ent([-0.5, -1.0]) + ent([-2.0, -3.0])) / 2
    assert f["mean_entropy"] == pytest.approx(h)
    assert 0.0 <= f["mean_entropy_norm"] <= 1.0
    assert ms > 0.0


def test_features_reject_missing_probs():
    with pytest.raises(ValueError, match="completion_probabilities"):
        g.features_from_response({"tokens": [1], "stop_type": "word"})


# --- sweep math -----------------------------------------------------------

def test_gate_metrics_hand_math():
    vec = {"sup": [1.0, 1.0, 0.0, 0.0], "keep": [0.0, 0.0, 1.0, 1.0]}
    m = g.gate_metrics([1.0, 0.0, 1.0, 0.0], vec)
    assert m["noopfp_reduction"] == pytest.approx(0.5)
    assert m["correct_retention"] == pytest.approx(0.5)
    assert m["abstained"] == 2.0


def test_feature_curve_monotone_synthetic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    vec = {"sup": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
           "keep": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}
    curve = g.feature_curve(values, "lt", vec)
    by_thr = {p["threshold"]: p for p in curve}
    assert by_thr[4.0]["noopfp_reduction"] == pytest.approx(1.0)   # abstain v<4
    assert by_thr[4.0]["correct_retention"] == pytest.approx(1.0)
    assert curve[0]["noopfp_reduction"] == 0.0                     # abstain none
    assert curve[-1]["correct_retention"] == 0.0                   # abstain all
    reds = [p["noopfp_reduction"] for p in curve]
    assert reds == sorted(reds)                                    # coverage grows
    gt = g.feature_curve(values, "gt", vec)
    assert gt[0]["noopfp_reduction"] == 0.0 and gt[-1]["correct_retention"] == 0.0


def test_pareto_front_drops_dominated_points():
    pts = [
        {"correct_retention": 1.0, "noopfp_reduction": 0.3},
        {"correct_retention": 1.0, "noopfp_reduction": 0.5},
        {"correct_retention": 0.9, "noopfp_reduction": 0.5},   # dominated
        {"correct_retention": 0.8, "noopfp_reduction": 0.9},
    ]
    front = g.pareto_front(pts)
    keys = {(p["correct_retention"], p["noopfp_reduction"]) for p in front}
    assert keys == {(1.0, 0.5), (0.8, 0.9)}


# --- verdict paths --------------------------------------------------------

def passing_cohort():
    rows = ([row("noop", f"fp{i}", -2.0 + 0.1 * i, fp_label()) for i in range(4)]
            + [row("noop", "ok", -0.1, ok_label())]
            + [row("scenario", f"e{i}", -0.2 + 0.05 * i, exact_label()) for i in range(3)])
    return rows


def test_sweep_pass_verdict_and_targets(tmp_path):
    write_rows(tmp_path, passing_cohort())
    g.sweep(tmp_path)
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    ev = json.loads((tmp_path / "evaluation.json").read_text())
    assert verdict["verdict"] == "GATE1-PASS"
    assert verdict["adoption"] == "NOT-ASSESSED"
    assert verdict["targets"]["noopfp_relative_reduction"]["value"] == pytest.approx(1.0)
    assert verdict["targets"]["correct_retention"]["value"] == pytest.approx(1.0)
    assert verdict["targets"]["latency_overhead_p95"]["pass"] is True
    assert ev["baseline"]["false_suggestions"] == 4
    assert ev["baseline"]["correct_proposals"] == 3
    assert ev["best_gate"]["feature"] in g.FEATURES
    assert ev["bootstrap"]["noopfp_reduction_ci95"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert ev["gate_disabled"] == {"noopfp_reduction": 0.0, "correct_retention": 1.0}


def test_sweep_fail_verdict_when_no_gate_meets_targets(tmp_path):
    # correct rows sit at BOTH extremes: every tail holding >=2 of the 10
    # false suggestions also holds a correct row, so no locked threshold
    # reaches 20% reduction at >=99% retention (2 corrects: any loss < 0.99).
    rows = [row("scenario", "c-lo", -3.0, exact_label()),
            row("scenario", "c-hi", 1.0, exact_label())]
    rows += [row("noop", f"fp{i}", -2.0 + 0.1 * i, fp_label()) for i in range(10)]
    write_rows(tmp_path, rows)
    g.sweep(tmp_path)
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    ev = json.loads((tmp_path / "evaluation.json").read_text())
    assert verdict["verdict"] == "GATE1-FAIL"
    assert verdict["targets"]["noopfp_relative_reduction"]["pass"] is False
    assert ev["best_gate"] is None


def test_label_drift_guard_blocks_stale_labels(tmp_path):
    rows = passing_cohort()
    for r in rows[:2]:
        r["drift"] = True                       # 2/8 = 25% > 10% ceiling
    write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="label drift"):
        g.sweep(tmp_path)


def test_bootstrap_is_deterministic():
    rows = passing_cohort()
    vec = g.cohort_vectors(rows)
    gate = {"feature": "mean_lp", "direction": "lt", "threshold": -1.0}
    a = g.paired_bootstrap(rows, gate, vec, n=200, seed=7)
    b = g.paired_bootstrap(rows, gate, vec, n=200, seed=7)
    assert a == b


def test_latency_overhead_target():
    rows = [row("noop", f"fp{i}", -2.0, fp_label(), feature_ms=0.05, predicted_ms=1.0)
            for i in range(20)]
    assert g.latency_overhead(rows)["pass"] is True
    rows[0]["feature_ms"] = rows[1]["feature_ms"] = 0.5   # p95 crosses 10%
    lat = g.latency_overhead(rows)
    assert lat["pass"] is False and lat["p95"] == pytest.approx(0.5)


# --- extract guards + smoke ----------------------------------------------

def test_extract_validates_frozen_cohort_before_serving(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "cases.jsonl").write_text("".join(
        json.dumps({"kind": "noop", "id": f"n{i}", "prompt": "p"}) + "\n"
        for i in range(3)))
    with pytest.raises(ValueError, match="frozen 255"):
        g.extract(tmp_path, assets, tmp_path / "m.gguf", tmp_path / "a.jsonl")


def test_smoke_end_to_end(tmp_path, capsys):
    g.smoke(tmp_path)
    out = capsys.readouterr().out
    assert "SMOKE OK" in out
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "GATE1-PASS"
    assert verdict["adoption"] == "NOT-ASSESSED"
