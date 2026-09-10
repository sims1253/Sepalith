#!/usr/bin/env python3
"""AG1 gate 1 — logprob-confidence abstention gates on the banked b4 cohort.

The noopFP structural floor (120/204 = 58.8% scored false-suggestion rate on
no-op cases, b4-other-quant-quality 2026-09-08) is the top unresolved product
problem. Gate 1 asks the cheap question first (backlog AG1 row, 2026-09-10):
does a serving-time-available confidence score — built ONLY from what
llama.cpp already returns per request — suppress >=20% of scored false
suggestions while retaining >=99% of correct proposals, at <=10% added CPU
cost?  This is selective prediction on EXISTING labels; no new judging, no
training, no queue edits.

Feature availability (verified in the pinned b10453 server source,
experiments/bin/src/llamacpp-b10453/tools/server/):
  - raw POST /completion with "n_probs": N (alias "logprobs") returns
    "completion_probabilities": one array per generated token of top-N
    candidates {id, token, bytes, logprob} (server-task.cpp
    to_json_non_oaicompat; server-schema.cpp n_probs alias logprobs), plus
    "tokens" (sampled ids), "stop_type" (eos|word|limit), "stopping_word",
    "tokens_predicted", "timings".
  - /v1/completions with logprobs:true top_logprobs:N returns the same
    candidates under choices[0].logprobs.content (server-common.cpp:1352).
  extract therefore uses the raw /completion endpoint (stop_type is only
  there) at temperature 0 with the b4 stops.

Subcommands:
  extract  serve b4 Q8 (GpuServer pattern, port 18478), re-run the frozen
           513-case cohort collecting logprobs, join the BANKED labels from
           the b4-other archive requests.jsonl by (kind, case id), write
           ag1-responses.jsonl (audit) + logprob-features.jsonl.
  sweep    CPU-only risk-coverage sweep over feature x direction x threshold
           grids on logprob-features.jsonl; Pareto front, best gate meeting
           the AG1 targets, paired bootstrap CIs, latency overhead bound;
           writes evaluation.json + verdict.json (GATE1-PASS/FAIL, adoption
           NOT-ASSESSED).
  smoke    10 mock cases through the same join + features + sweep path.

Latency basis: feature_ms (measured feature computation per row) divided by
the banked per-row decode time (timings.predicted_ms). This bounds the
gate's added CPU cost against decode only — full shadow-serving cycles are
gate 2 (backlog AG1 step 5), not assessed here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_noop_fp as noop                                  # noqa: E402
import eval_scenarios as scenario                            # noqa: E402
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write  # noqa: E402

ARM = "Q8_0"
PORT = 18478
CTX = 8192
N_PROBS = 10                 # top-N candidates per token for entropy
SERVER_FLAGS = ["-lv", "4", "--seed", "20260905"]
MAX_TOKENS = {"scenario": 640, "noop": 320}   # b4-other cohort limits
LOW_LP = -3.0                # frac_lp_below cut (nats)

RED_TARGET = 0.20            # >=20% relative noopFP reduction
RET_TARGET = 0.99            # >=99% correct-proposal retention
LAT_TARGET = 0.10            # <=10% added CPU-cycle overhead
BOOT_N = 2000
BOOT_SEED = 20260910
CLUSTER_BOOT_N = 1000
MAX_DRIFT_RATE = 0.10        # regenerated-vs-banked label disagreement ceiling

# serving-time-available features (all from one /completion response)
FEATURES = ["mean_lp", "min_lp", "first_lp", "last_lp", "mean_entropy",
            "max_entropy", "mean_entropy_norm", "frac_lp_below", "pred_len",
            "stop_is_word", "stop_is_eos", "stop_is_limit"]


# ---------------------------------------------------------------------------
# labels: the banked b4-other archive rows, joined by (kind, case id)
# ---------------------------------------------------------------------------

def load_labels(archive_requests: Path, arm: str = ARM) -> dict:
    """Banked per-case outcomes from the frozen b4-other archive rows."""
    labels = {}
    for line in Path(archive_requests).read_text().splitlines():
        r = json.loads(line)
        if r.get("arm") != arm:
            continue
        key = (r["kind"], r["id"])
        if key in labels:
            raise ValueError(f"duplicate banked label for {key}")
        if r["kind"] == "scenario":
            labels[key] = {f: r.get(f) for f in
                           ("exact", "valid_pass", "fail_kind", "valid_reason", "family")}
        else:
            labels[key] = {f: r.get(f) for f in ("proposal", "expectation", "cls")}
    if not labels:
        raise ValueError(f"no {arm} rows in {archive_requests}")
    return labels


def check_join(cases: list, labels: dict) -> None:
    missing = [(r["kind"], r["id"]) for r in cases if (r["kind"], r["id"]) not in labels]
    if missing:
        shown = ", ".join(str(k) for k in missing[:5])
        raise ValueError(f"banked labels missing for {len(missing)} cases: {shown}")


# ---------------------------------------------------------------------------
# features from one raw /completion response
# ---------------------------------------------------------------------------

def sampled_logprob(cands: list, tok_id) -> dict:
    """The candidate entry for the sampled token id (greedy: entry 0)."""
    for c in cands:
        if c.get("id") == tok_id:
            return c
    return cands[0]


def token_entropy_nats(cands: list) -> float:
    """Entropy over the renormalized top-N candidate distribution."""
    ps = [math.exp(c["logprob"]) for c in cands]
    z = sum(ps)
    if z <= 0.0:
        return 0.0
    h = -sum((p / z) * math.log(p / z) for p in ps if p > 0.0)
    return h


def features_from_response(data: dict) -> tuple:
    """(features dict, feature_ms) — CPU cost is measured, not estimated."""
    t0 = time.perf_counter()
    tokens = data.get("tokens") or []
    probs = data.get("completion_probabilities") or []
    if not probs:
        raise ValueError("no completion_probabilities: set n_probs > 0")
    n = len(probs)   # one entry per generated token; data["tokens"] is empty unless return_tokens
    if n == 0:
        return None, 0.0   # no proposal generated: nothing to gate (correct no-op)
    lps, ents = [], []
    for i in range(n):
        entry = probs[i]
        if not isinstance(entry, dict) or "logprob" not in entry:
            raise ValueError(f"unexpected completion_probabilities shape at token {i}")
        lps.append(entry["logprob"])
        cands = entry.get("top_logprobs") or [entry]
        ents.append(token_entropy_nats(cands))
    k = min(len(cands) for cands in probs[:n]) if n else 1
    norm = math.log(max(2, k))
    stop = data.get("stop_type") or "none"
    feats = {
        "mean_lp": sum(lps) / n,
        "min_lp": min(lps),
        "first_lp": lps[0],
        "last_lp": lps[-1],
        "mean_entropy": sum(ents) / n,
        "max_entropy": max(ents),
        "mean_entropy_norm": (sum(ents) / n) / norm if norm > 0 else 0.0,
        "frac_lp_below": sum(lp < LOW_LP for lp in lps) / n,
        "pred_len": data.get("tokens_predicted", n),
        "stop_is_word": float(stop == "word"),
        "stop_is_eos": float(stop == "eos"),
        "stop_is_limit": float(stop == "limit"),
    }
    return feats, (time.perf_counter() - t0) * 1000.0


def regen_outcomes(case: dict, data: dict) -> dict:
    """Re-derive the proposal/exact outcome from the regenerated text."""
    text = data.get("content") or ""
    if case["kind"] == "scenario":
        pred = scenario.parse_pred("zeta2", text)
        gt = [l.rstrip() for l in case["scenario"]["region_new"]]
        while gt and not gt[-1]:
            gt.pop()
        return {"exact": pred == gt}
    return {"proposal": bool(noop.parse_prediction(text))}


def drift(case: dict, label: dict, regen: dict) -> bool:
    if case["kind"] == "scenario":
        return bool(regen["exact"]) != bool(label.get("exact"))
    return bool(regen["proposal"]) != bool(label.get("proposal"))


def build_feature_rows(responses: list, cases: list, labels: dict) -> list:
    """Join banked labels by (kind, id) and compute features per response."""
    skipped_empty = 0
    by_key = {(c["kind"], c["id"]): c for c in cases}
    rows = []
    for resp in responses:
        key = (resp["kind"], resp["id"])
        case = by_key.get(key)
        if case is None:
            raise ValueError(f"response without frozen case: {key}")
        if key not in labels:
            raise ValueError(f"missing banked label for {key}")
        feats, fms = features_from_response(resp["data"])
        if feats is None:
            skipped_empty += 1
            continue
        regen = regen_outcomes(case, resp["data"])
        row = {"arm": ARM, "kind": key[0], "id": key[1],
               "features": feats, "feature_ms": fms,
               "predicted_ms": ((resp["data"].get("timings") or {}).get("predicted_ms")),
               "n_tokens": len(resp["data"].get("tokens") or []),
               "label": labels[key], "regen": regen,
               "drift": drift(case, labels[key], regen)}
        rows.append(row)
    if len({(r["kind"], r["id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate feature rows")
    return rows


# ---------------------------------------------------------------------------
# extract: serve Q8, re-run the frozen cohort with logprobs
# ---------------------------------------------------------------------------

def post_completion(port: int, prompt: str, n_predict: int, stop: list) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0, "stop": stop, "n_probs": N_PROBS,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def extract(run: Path, assets: Path, model: Path, archive: Path,
            port: int = PORT) -> None:
    run.mkdir(parents=True, exist_ok=True)
    skipped_empty = 0
    cases = [json.loads(l) for l in (assets / "cases.jsonl").read_text().splitlines()]
    if len(cases) != 513 or len({(r["kind"], r["id"]) for r in cases}) != 513:
        raise ValueError("Require the frozen 255 scenario + 258 noop cohort")
    if sum(r["kind"] == "scenario" for r in cases) != 255:
        raise ValueError("Scenario cohort changed")
    labels = load_labels(archive)
    check_join(cases, labels)          # fail before any GPU time is spent
    server_bin = Path(os.environ["S1_RUNTIME"]) / "llama-server"
    if not server_bin.is_file():
        raise ValueError("Missing frozen runtime (S1_RUNTIME)")
    server = None

    def expire(*_):
        raise DeadlineExceeded("AG1 extract exceeded the 2h bound")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(7200)
    try:
        server = GpuServer(model, port, SERVER_FLAGS, ctx=CTX, server=server_bin,
                           foreground=True, log_path=run / f"server-{ARM}.log")
        server.start(ready_timeout=1800)
        check_offload((run / f"server-{ARM}.log").read_text())
        with (run / "ag1-responses.jsonl").open("x") as out:
            for c in cases:
                limit = MAX_TOKENS[c["kind"]]
                if tokenize(server.port, c["prompt"]) + limit + 16 > CTX:
                    raise ValueError(f"Prompt exceeds context: {(c['kind'], c['id'])}")
                stops = [scenario.STOP] if c["kind"] == "scenario" else noop.EXT_STOPS
                data = post_completion(server.port, c["prompt"], limit, stops)
                if data.get("stop_type") not in ("word", "eos", "limit") \
                        or not data.get("completion_probabilities"):
                    raise ValueError(f"Invalid response for {(c['kind'], c['id'])}")
                out.write(json.dumps({"arm": ARM, "kind": c["kind"], "id": c["id"],
                                      "at": time.time(), "data": data},
                                     allow_nan=False) + "\n")
                out.flush()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if server is not None:
            server.stop()
    responses = [json.loads(l) for l in (run / "ag1-responses.jsonl").read_text().splitlines()]
    rows = build_feature_rows(responses, cases, labels)
    with (run / "logprob-features.jsonl").open("x") as out:
        for r in rows:
            out.write(json.dumps(r, allow_nan=False) + "\n")
    n_drift = sum(r["drift"] for r in rows)
    print(json.dumps({"rows": len(rows), "label_drift": n_drift,
                      "skipped_empty_generations": skipped_empty}))


# ---------------------------------------------------------------------------
# sweep: risk-coverage over feature x direction x threshold (CPU only)
# ---------------------------------------------------------------------------

def cohort_vectors(rows: list) -> dict:
    """sup = scored false suggestion; keep = correct proposal (banked)."""
    sup = [1.0 if (r["kind"] == "noop" and r["label"].get("expectation") == "no_proposal"
                   and r["label"].get("proposal")) else 0.0 for r in rows]
    keep = [1.0 if (r["kind"] == "scenario" and r["label"].get("exact")) else 0.0 for r in rows]
    bad_edit = [1.0 if (r["kind"] == "scenario" and not r["label"].get("exact")) else 0.0
                for r in rows]
    judgment = [1.0 if (r["kind"] == "noop" and r["label"].get("expectation") == "judgment")
                else 0.0 for r in rows]
    return {"sup": sup, "keep": keep, "bad_edit": bad_edit, "judgment": judgment}


def gate_metrics(abstain: list, vec: dict) -> dict:
    """Risk-coverage point for one abstain indicator vector."""
    def rate(num, den):
        return num / den if den else None
    def col(name):
        return vec.get(name) or []
    fp_total, fp_supp = sum(vec["sup"]), sum(a * s for a, s in zip(abstain, vec["sup"]))
    kp_total, kp_lost = sum(vec["keep"]), sum(a * k for a, k in zip(abstain, vec["keep"]))
    return {"noopfp_reduction": rate(fp_supp, fp_total),
            "correct_retention": rate(kp_total - kp_lost, kp_total),
            "incorrect_edit_suppressed": rate(
                sum(a * b for a, b in zip(abstain, col("bad_edit"))), sum(col("bad_edit"))),
            "judgment_suppressed": rate(
                sum(a * j for a, j in zip(abstain, col("judgment"))), sum(col("judgment"))),
            "abstained": sum(abstain)}


def feature_curve(values: list, direction: str, vec: dict) -> list:
    """Sweep thresholds over one feature. direction 'lt': abstain if v < t;
    'gt': abstain if v > t.  Tie groups keep the sweep finite and exact;
    sentinels min-1 / max+1 keep every emitted threshold finite and JSON-safe
    while still covering the abstain-none and abstain-all endpoints."""
    lo, hi = min(values) - 1.0, max(values) + 1.0
    order = sorted(range(len(values)), key=lambda i: values[i],
                   reverse=(direction == "gt"))
    points = []
    prefix = [0.0] * len(values)

    def emit(threshold):
        points.append({"threshold": threshold, **gate_metrics(prefix, vec)})

    emit(lo if direction == "lt" else hi)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            prefix[order[k]] = 1.0
        emit(values[order[j]] if j < len(order) else
             (hi if direction == "lt" else lo))
        i = j
    return points


def sweep_gates(rows: list) -> dict:
    vec = cohort_vectors(rows)
    if sum(vec["sup"]) < 1 or sum(vec["keep"]) < 1:
        raise ValueError("Cohort needs >=1 scored false suggestion and >=1 correct proposal")
    curves = {}
    for name in FEATURES:
        values = [r["features"][name] for r in rows]
        for direction in ("lt", "gt"):
            pts = feature_curve(values, direction, vec)
            for p in pts:
                p["feature"], p["direction"] = name, direction
            curves[f"{name}|{direction}"] = pts
    allpts = [p for pts in curves.values() for p in pts]
    return {"curves": curves, "points": allpts, "vec": vec}


def pareto_front(points: list) -> list:
    """Non-dominated (correct_retention, noopfp_reduction) maxima."""
    front = []
    for p in sorted(points, key=lambda q: (-(q["correct_retention"] or -1),
                                           -(q["noopfp_reduction"] or -1))):
        if p["correct_retention"] is None or p["noopfp_reduction"] is None:
            continue
        if not any(f["correct_retention"] >= p["correct_retention"]
                   and f["noopfp_reduction"] >= p["noopfp_reduction"]
                   and (f["correct_retention"], f["noopfp_reduction"]) !=
                   (p["correct_retention"], p["noopfp_reduction"]) for f in front):
            front.append(p)
    return front


def best_gate(points: list) -> dict:
    """Highest reduction among gates meeting both statistical targets."""
    meet = [p for p in points
            if p["noopfp_reduction"] is not None and p["correct_retention"] is not None
            and p["noopfp_reduction"] >= RED_TARGET
            and p["correct_retention"] >= RET_TARGET]
    if not meet:
        return {}
    return max(meet, key=lambda p: (p["noopfp_reduction"], p["correct_retention"],
                                    -p["abstained"], p["feature"], p["direction"],
                                    p["threshold"]))


def abstain_flags(rows: list, gate: dict) -> list:
    vals = [r["features"][gate["feature"]] for r in rows]
    t = gate["threshold"]
    return [1.0 if (v < t if gate["direction"] == "lt" else v > t) else 0.0 for v in vals]


def paired_bootstrap(rows: list, gate: dict, vec: dict, n: int = BOOT_N,
                     seed: int = BOOT_SEED) -> dict:
    """Case-level paired CI: one resample drives BOTH metrics (locked gate)."""
    flags = abstain_flags(rows, gate)
    rng = random.Random(seed)
    reds, rets = [], []
    for _ in range(n):
        idx = [rng.randrange(len(rows)) for _ in range(len(rows))]
        sub = {"sup": [vec["sup"][i] for i in idx], "keep": [vec["keep"][i] for i in idx]}
        m = gate_metrics([flags[i] for i in idx], sub)
        if m["noopfp_reduction"] is not None:
            reds.append(m["noopfp_reduction"])
        if m["correct_retention"] is not None:
            rets.append(m["correct_retention"])
    def ci(xs):
        if not xs:
            return None
        xs.sort()
        q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1)))]
        return [q(0.025), q(0.975)]
    return {"n_resamples": n, "seed": seed, "used_red": len(reds), "used_ret": len(rets),
            "noopfp_reduction_ci95": ci(reds), "correct_retention_ci95": ci(rets)}


def cluster_bootstrap(rows: list, gate: dict, vec: dict, n: int = CLUSTER_BOOT_N,
                      seed: int = BOOT_SEED + 1) -> dict:
    """Sensitivity: resample cls (noop) / family (scenario) clusters."""
    clusters = {}
    for i, r in enumerate(rows):
        clusters.setdefault(r["label"].get("cls") or r["label"].get("family") or "?", []).append(i)
    names = sorted(clusters)
    rng = random.Random(seed)
    flags = abstain_flags(rows, gate)
    reds, rets = [], []
    for _ in range(n):
        idx = [i for name in (rng.choice(names) for _ in range(len(names)))
               for i in clusters[name]]
        sub = {"sup": [vec["sup"][i] for i in idx], "keep": [vec["keep"][i] for i in idx]}
        m = gate_metrics([flags[i] for i in idx], sub)
        if m["noopfp_reduction"] is not None:
            reds.append(m["noopfp_reduction"])
        if m["correct_retention"] is not None:
            rets.append(m["correct_retention"])
    def ci(xs):
        if not xs:
            return None
        xs.sort()
        q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1)))]
        return [q(0.025), q(0.975)]
    return {"n_resamples": n, "seed": seed,
            "noopfp_reduction_ci95": ci(reds), "correct_retention_ci95": ci(rets)}


def latency_overhead(rows: list) -> dict:
    """p50/p95 of feature_ms / banked decode ms (the gate's CPU cost bound)."""
    ratios = [r["feature_ms"] / r["predicted_ms"] for r in rows
              if r.get("feature_ms") is not None
              and (r.get("predicted_ms") or 0) > 0]
    if not ratios:
        return {"assessed": False, "reason": "no per-row decode timings"}
    ratios.sort()
    q = lambda f: ratios[min(len(ratios) - 1, int(f * (len(ratios) - 1)))]
    p50, p95 = q(0.5), q(0.95)
    return {"assessed": True, "n": len(ratios), "p50": p50, "p95": p95,
            "basis": "measured feature computation vs banked timings.predicted_ms "
                     "(decode only; full shadow-serving cycles are gate 2)",
            "target": LAT_TARGET, "pass": p95 <= LAT_TARGET}


def breakdown(rows: list, flags: list) -> dict:
    out = {}
    for i, r in enumerate(rows):
        g = r["label"].get("cls") or r["label"].get("family") or "?"
        d = out.setdefault(g, {"n": 0, "abstained": 0, "sup": 0, "sup_abstained": 0,
                               "keep": 0, "keep_lost": 0})
        d["n"] += 1
        d["abstained"] += int(flags[i] > 0)
        if r["kind"] == "noop" and r["label"].get("expectation") == "no_proposal":
            d["sup"] += int(bool(r["label"].get("proposal")))
            d["sup_abstained"] += int(bool(r["label"].get("proposal")) and flags[i] > 0)
        if r["kind"] == "scenario" and r["label"].get("exact"):
            d["keep"] += 1
            d["keep_lost"] += int(flags[i] > 0)
    return out


def sweep(run: Path) -> None:
    rows = [json.loads(l) for l in (run / "logprob-features.jsonl").read_text().splitlines()]
    if len({(r["kind"], r["id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate rows in logprob-features.jsonl")
    n_drift = sum(r["drift"] for r in rows)
    if n_drift > MAX_DRIFT_RATE * len(rows):
        raise ValueError(f"label drift {n_drift}/{len(rows)} exceeds {MAX_DRIFT_RATE:.0%}: "
                         "banked labels do not reproduce; re-extract or re-bank")
    s = sweep_gates(rows)
    vec = s["vec"]
    scored = sum(1 for r in rows if r["kind"] == "noop"
                 and r["label"].get("expectation") == "no_proposal")
    baseline = {"scored_noop": scored, "false_suggestions": int(sum(vec["sup"])),
                "noopfp_rate": sum(vec["sup"]) / scored if scored else None,
                "correct_proposals": int(sum(vec["keep"])),
                "scenarios": sum(1 for r in rows if r["kind"] == "scenario")}
    gate = best_gate(s["points"])
    if not gate:
        write(run / "evaluation.json", dict(
            baseline=baseline, label_drift=n_drift, curves=s["curves"],
            pareto_front=pareto_front(s["points"]), best_gate=None,
            bootstrap=None, latency=latency_overhead(rows), breakdown=None,
            boundary="No feature/direction/threshold meets both statistical targets"))
        write(run / "verdict.json", dict(
            verdict="GATE1-FAIL", adoption="NOT-ASSESSED",
            targets={"noopfp_relative_reduction": {"target": RED_TARGET, "pass": False},
                     "correct_retention": {"target": RET_TARGET, "pass": False}},
            boundary="No gate on the swept features meets >=20% noopFP reduction at "
                     ">=99% correct retention; per the AG1 kill/park rule write the "
                     "negative and stop integration."))
        return
    flags = abstain_flags(rows, gate)
    point = gate_metrics(flags, vec)
    boot = paired_bootstrap(rows, gate, vec)
    clus = cluster_bootstrap(rows, gate, vec)
    lat = latency_overhead(rows)
    retained_exact = sum(1 for i, r in enumerate(rows)
                         if r["kind"] == "scenario" and r["label"].get("exact")
                         and not flags[i])
    n_scen = baseline["scenarios"]
    length_only = best_gate([p for p in s["points"] if p["feature"] == "pred_len"])
    write(run / "evaluation.json", dict(
        baseline=baseline, label_drift=n_drift, curves=s["curves"],
        pareto_front=pareto_front(s["points"]),
        best_gate=dict(gate, **point, scenario_exact_after_gate=retained_exact / n_scen if n_scen else None),
        gate_disabled=dict(noopfp_reduction=0.0, correct_retention=1.0),
        length_only_baseline=length_only,
        bootstrap=boot, cluster_bootstrap=clus, latency=lat,
        breakdown_at_gate=breakdown(rows, flags),
        boundary="Single-arm Q8 replay of the frozen 513-case cohort; banked labels "
                 "from the b4-other archive; no calibration/confirmation split; "
                 "retention powered by n_correct rows only (at zero observed loss the "
                 f"binomial 95% upper bound on the loss rate is ~3/n_correct = "
                 f"3/{int(sum(vec['keep']))}); X5-S1 precedent 44.5-80.3% "
                 "incorrect-suppressed @0-lost is the mechanism bar, not a transfer."))
    targets = {
        "noopfp_relative_reduction": {"value": point["noopfp_reduction"],
                                      "target": RED_TARGET, "pass":
                                      point["noopfp_reduction"] >= RED_TARGET},
        "correct_retention": {"value": point["correct_retention"], "target": RET_TARGET,
                              "pass": point["correct_retention"] >= RET_TARGET,
                              "n_correct": int(sum(vec["keep"]))},
        "latency_overhead_p95": {"value": lat.get("p95"), "target": LAT_TARGET,
                                 "pass": bool(lat.get("assessed")) and lat["p95"] <= LAT_TARGET,
                                 "basis": lat.get("basis", "")},
    }
    write(run / "verdict.json", dict(
        verdict="GATE1-PASS" if all(t["pass"] for t in targets.values()) else "GATE1-FAIL",
        adoption="NOT-ASSESSED", targets=targets,
        boundary="Gate-1 only: locked threshold selected on the SAME cohort it is "
                 "reported on (confirmation split is the next step); latency is the "
                 "feature-compute vs decode bound, not shadow-serving cycles; adoption "
                 "needs gate-2 CPU shadow run + threshold/rollback plumbing."))
    print(json.dumps({"verdict": json.loads((run / "verdict.json").read_text())["verdict"],
                      "gate": {k: gate[k] for k in ("feature", "direction", "threshold")},
                      "reduction": point["noopfp_reduction"],
                      "retention": point["correct_retention"]}))


# ---------------------------------------------------------------------------
# smoke: 10 mock cases through join + features + sweep
# ---------------------------------------------------------------------------

def mock_response(lps: list, stop_type: str, predicted_ms: float, content: str,
                  tok_base: int = 1000) -> dict:
    toks, probs = [], []
    for i, lp in enumerate(lps):
        tid = tok_base + i
        toks.append(tid)
        cands = [{"id": tid, "token": "x", "bytes": [1], "logprob": lp},
                 {"id": tid + 500000, "token": "y", "bytes": [2], "logprob": lp - 0.7}]
        probs.append(cands)
    return {"content": content, "tokens": toks, "stop_type": stop_type,
            "stopping_word": "", "tokens_predicted": len(lps),
            "completion_probabilities": probs,
            "timings": {"predicted_ms": predicted_ms}}


def smoke(run: Path) -> None:
    """10 cases; mean_lp perfectly separates false suggestions from correct."""
    run.mkdir(parents=True, exist_ok=True)
    mock_cases, mock_archive, mock_responses = [], [], []
    spec = [
        ("noop", "n-fp1", "no_proposal", {"proposal": True, "expectation": "no_proposal",
                                          "cls": "c_stmt_line_end"}, -2.5),
        ("noop", "n-fp2", "no_proposal", {"proposal": True, "expectation": "no_proposal",
                                          "cls": "c_blank_in_fn"}, -2.1),
        ("noop", "n-fp3", "no_proposal", {"proposal": True, "expectation": "no_proposal",
                                          "cls": "c_mid_identifier"}, -1.9),
        ("noop", "n-fp4", "no_proposal", {"proposal": True, "expectation": "no_proposal",
                                          "cls": "a_file_end"}, -1.7),
        ("noop", "n-ok1", "no_proposal", {"proposal": False, "expectation": "no_proposal",
                                          "cls": "a_after_close_brace"}, -0.1),
        ("noop", "n-ok2", "judgment", {"proposal": True, "expectation": "judgment",
                                       "cls": "b_mid_typing"}, -0.3),
        ("scenario", "s-e1", None, {"exact": True, "valid_pass": True, "fail_kind": None,
                                    "valid_reason": None, "family": "rename_propagation"}, -0.2),
        ("scenario", "s-e2", None, {"exact": True, "valid_pass": True, "fail_kind": None,
                                    "valid_reason": None, "family": "pipe_rewrite"}, -0.1),
        ("scenario", "s-e3", None, {"exact": True, "valid_pass": True, "fail_kind": None,
                                    "valid_reason": None, "family": "format_propagation"}, -0.05),
        ("scenario", "s-bad", None, {"exact": False, "valid_pass": False, "fail_kind": "x",
                                     "valid_reason": "y", "family": "doc_sync"}, -1.0),
    ]
    for kind, cid, _exp, label, lp in spec:
        mock_cases.append({"kind": kind, "id": cid, "prompt": "p",
                           **({"scenario": {"region_new": ["z"]}} if kind == "scenario" else {})})
        mock_archive.append({"arm": ARM, "kind": kind, "id": cid, **label})
        # content reproduces the banked outcome, so drift stays 0
        if kind == "scenario":
            content = "z" if label["exact"] else "q"
        else:
            content = "x" if label["proposal"] else ""
        mock_responses.append({"arm": ARM, "kind": kind, "id": cid, "at": 0.0,
                               "data": mock_response([lp, lp + 0.01], "word", 50.0, content)})
    # shuffle the archive order: the join must be by id, not by position
    mock_archive.reverse()
    archive_path = run / "mock-archive-requests.jsonl"
    archive_path.write_text("".join(json.dumps(r) + "\n" for r in mock_archive))
    labels = load_labels(archive_path)
    rows = build_feature_rows(mock_responses, mock_cases, labels)
    with (run / "logprob-features.jsonl").open("w") as out:
        for r in rows:
            out.write(json.dumps(r, allow_nan=False) + "\n")
    sweep(run)
    verdict = json.loads((run / "verdict.json").read_text())
    assert verdict["verdict"] == "GATE1-PASS", verdict
    assert verdict["adoption"] == "NOT-ASSESSED", verdict
    ev = json.loads((run / "evaluation.json").read_text())
    assert ev["best_gate"]["noopfp_reduction"] == 1.0, ev["best_gate"]
    assert ev["best_gate"]["correct_retention"] == 1.0, ev["best_gate"]
    assert ev["baseline"]["false_suggestions"] == 4
    assert ev["baseline"]["correct_proposals"] == 3
    print("SMOKE OK: join by id + sweep math + verdict "
          f"{verdict['verdict']} (reduction=1.0, retention=1.0, 4 FP suppressed, "
          "3/3 correct kept)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["extract", "sweep", "smoke"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, help="frozen cohort assets dir (cases.jsonl)")
    p.add_argument("--model", type=Path, help="Q8 gguf to serve")
    p.add_argument("--archive", type=Path,
                   help="b4-other archive requests.jsonl holding the banked labels")
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()
    if a.action == "smoke":
        smoke(a.run)
    elif a.action == "sweep":
        sweep(a.run)
    else:
        if not (a.assets and a.model and a.archive):
            p.error("extract requires --assets, --model and --archive")
        extract(a.run, a.assets, a.model, a.archive, port=a.port)


if __name__ == "__main__":
    main()
