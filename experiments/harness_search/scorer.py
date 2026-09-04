"""scorer.py — the pre-registered lexicographic scorer (plan §1.2).

Objective (MAXIMIZE): `exact` = validator-verdict rate on D_harness via the
EXACT scenarios.validate_example (eval_scenarios.validator_verdict code
path), under the plan's noise control: 2 rollouts per candidate-example
pair at temp 0; a row passes only when BOTH rollouts pass the validator;
verdict disagreement marks the row unstable (counts as not-pass, reported).

Guardrails (eligibility):
  noopFP: proposal rate on the scored noop classes (a/c/d) must not exceed
  baseline + 2pp. p95 latency: p95 over the candidate's measured request
  wall-times must stay <= 1.3x the default-config p95 on the same rows and
  server.

Selection key (deterministic): (exact_pass, -unstable_frac, -p95) among
guardrail-passing candidates; ties broken by config fingerprint (asc) so
runs are reproducible. Pure functions throughout — unit-tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "synthetic-data"))

from eval_scenarios import validator_verdict            # noqa: E402  (exact code path)
from parsep import parse_prediction, apply_post_heuristics  # noqa: E402


def predict(text: str, row: dict, cfg: dict) -> list[str]:
    """Full harness path for one rollout: parse + post-heuristics."""
    return apply_post_heuristics(parse_prediction(text, cfg), row, cfg)


def row_outcome(row: dict, rollouts: list[dict], cfg: dict) -> dict:
    """One candidate-example pair. rollouts = [{"text","latency"}, ...]x2."""
    verdicts, exacts = [], []
    for r in rollouts:
        try:
            pred = predict(r["text"], row, cfg)
            ok, kind, reason = validator_verdict(row, pred)
            verdicts.append(bool(ok))
            exacts.append(None)  # filled by caller if needed
        except Exception as e:  # transport/parse error scores 0
            verdicts.append(False)
    agree = verdicts[0] == verdicts[1]
    return dict(rid=row.get("id"), family=row["family"],
                passed=int(verdicts[0] and verdicts[1]),
                unstable=int(not agree),
                v0=verdicts[0], v1=verdicts[1],
                latencies=[r["latency"] for r in rollouts])


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def score_candidate(rows: list[dict], rollouts_by_row: dict, cfg: dict) -> dict:
    """rows: D_harness rows; rollouts_by_row: rid -> [{"text","latency"}x2]."""
    outcomes = [row_outcome(r, rollouts_by_row[r["id"]], cfg) for r in rows]
    n = len(outcomes)
    fam = {}
    for f in sorted({o["family"] for o in outcomes}):
        fo = [o for o in outcomes if o["family"] == f]
        fam[f] = dict(n=len(fo), exact=round(sum(o["passed"] for o in fo) / len(fo), 4))
    lats = [l for o in outcomes for l in o["latencies"]]
    return dict(
        n_rows=n,
        exact_pass=round(sum(o["passed"] for o in outcomes) / max(1, n), 4),
        unstable_frac=round(sum(o["unstable"] for o in outcomes) / max(1, n), 4),
        families=fam,
        p50_latency_s=round(sorted(lats)[len(lats) // 2], 3) if lats else None,
        p95_latency_s=round(_p95(lats), 3),
        mean_latency_s=round(sum(lats) / max(1, len(lats)), 3),
        _outcomes=outcomes,
    )


NOOP_SCORED_CLASSES = ("a", "c", "d")  # b_* is judgment/informational


def noop_proposal(text: str, cfg: dict) -> bool:
    """A proposal = parse non-empty (what the extension ghost-renders)."""
    return len(parse_prediction(text, cfg)) > 0


def score_noop(cases: list, texts_by_case: dict, cfg: dict) -> dict:
    """cases: eval_noop_fp Case objects; texts_by_case: case.id -> rollouts."""
    scored, props = 0, 0
    per_cls = {}
    for c in cases:
        rolls = texts_by_case.get(c.id)
        if rolls is None:
            continue
        proposed = any(noop_proposal(r["text"], cfg) for r in rolls)
        if c.cls[0] in "acd":
            scored += 1
            props += int(proposed)
        per_cls.setdefault(c.cls, [0, 0])
        per_cls[c.cls][0] += int(proposed)
        per_cls[c.cls][1] += 1
    return dict(scored_n=scored, proposal_rate=round(props / max(1, scored), 4),
                proposals=props, per_class={k: [v[0], v[1]] for k, v in sorted(per_cls.items())})


def guardrails(score: dict, noop_score: dict, baseline: dict) -> dict:
    """(ok, details) vs the default-config baselines (plan §1.2)."""
    noop_ok = noop_score["proposal_rate"] <= baseline["noop_rate"] + 0.02
    lat_ok = score["p95_latency_s"] <= 1.3 * baseline["p95_latency_s"]
    return dict(noop_ok=bool(noop_ok), latency_ok=bool(lat_ok),
                noop_gap=round(noop_score["proposal_rate"] - baseline["noop_rate"], 4),
                latency_ratio=round(score["p95_latency_s"] / max(1e-9, baseline["p95_latency_s"]), 3))


def selection_key(score: dict) -> tuple:
    """Lexicographic selection key (higher = better). Ties resolve to the
    FIRST-evaluated candidate (Python max is stable) — deterministic."""
    return (score["exact_pass"], -score["unstable_frac"],
            -score["p95_latency_s"])
