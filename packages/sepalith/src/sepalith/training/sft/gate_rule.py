"""Rule R1 (docs/training/plan-20260927/RULES.md) for automatic SFT gates.

The rule is a pure function of five DEV counts and the training loss, so it
can be tested on synthetic inputs and replayed from a decision record.

Metric names follow RULES.md:

- ``valid``: DEV75 cases whose output parses under PRM03 (of 75)
- ``cap``: DEV75 cases that hit the generation cap (of 75)
- ``edit``: DEV75 edit cases with an exact region (of 43)
- ``nofp``: DEV75 no-op cases where the model suggested an edit (of 32)
- ``score250``: DEV250 exact edits plus correct no-ops (of 250)
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import cast

CONTINUE = "continue"
STOP_FOR_USER = "stop_for_user"
ROLLBACK = "rollback"
OUTCOMES = (CONTINUE, STOP_FOR_USER, ROLLBACK)
_SEVERITY = {CONTINUE: 0, STOP_FOR_USER: 1, ROLLBACK: 2}

METRICS = ("valid", "cap", "edit", "nofp", "score250")
# First-gate floor and the later-gate bands, as written in RULES.md R1.
FIRST_GATE_MIN_VALID = 60
FIRST_GATE_MAX_CAP = 2
LATER_GATE_MIN_VALID = 68

Metrics = Mapping[str, object]
Record = dict[str, object]


def worst(outcomes: list[str]) -> str:
    return max(outcomes, key=_SEVERITY.__getitem__) if outcomes else CONTINUE


def metrics_from_summaries(dev75: Mapping[str, object], dev250: Mapping[str, object]) -> dict[str, int]:
    """Read the R1 counts from two ``campaign_eval.development_evaluator`` summaries."""
    if not isinstance(dev75["counts"], Mapping) or not isinstance(dev250["counts"], Mapping):
        raise TypeError("evaluator summaries need a counts mapping")
    counts75 = cast(Mapping[str, object], dev75["counts"])
    counts250 = cast(Mapping[str, object], dev250["counts"])
    values = {
        "valid": counts75["protocol_valid"],
        "cap": counts75["cap_hit"],
        "edit": counts75["edit_exact"],
        "nofp": counts75["strict_noop_false_suggestions"],
        "score250": _count(counts250["edit_exact"]) + _count(counts250["strict_noop_correct"]),
    }
    return {name: _count(value) for name, value in values.items()}


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"metric count must be a nonnegative integer: {value!r}")
    return value


def _finite_problems(metrics: Metrics, label: str, *, loss: bool) -> list[str]:
    problems = []
    for name in METRICS:
        value = metrics.get(name)
        if type(value) is not int:
            problems.append(f"{label}.{name} is missing or not a count ({value!r})")
    if loss:
        value = metrics.get("loss")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            problems.append(f"{label}.loss is missing or not finite ({value!r})")
    return problems


def _int(metrics: Metrics, name: str) -> int:
    value = metrics[name]
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def first_gate(current: Metrics, parent: Metrics) -> Record:
    """First gate of an arm against the step-0 parent baseline."""
    problems = _finite_problems(current, "current", loss=True) + _finite_problems(parent, "parent", loss=False)
    if problems:
        return {"rule": "R1", "kind": "first_gate", "outcome": STOP_FOR_USER, "rows": [], "problems": problems}
    rows = [
        _row("valid", _int(current, "valid"), f">= {FIRST_GATE_MIN_VALID}",
             CONTINUE if _int(current, "valid") >= FIRST_GATE_MIN_VALID else STOP_FOR_USER),
        _row("cap", _int(current, "cap"), f"<= {FIRST_GATE_MAX_CAP}",
             CONTINUE if _int(current, "cap") <= FIRST_GATE_MAX_CAP else STOP_FOR_USER),
        _row("loss", current["loss"], "finite", CONTINUE),
        _row("edit", _int(current, "edit"), f"> parent edit {_int(parent, 'edit')}",
             CONTINUE if _int(current, "edit") > _int(parent, "edit") else STOP_FOR_USER),
    ]
    return {"rule": "R1", "kind": "first_gate", "outcome": worst([str(r["outcome"]) for r in rows]),
            "rows": rows, "problems": []}


def _band(change: int, cont: int, stop: int) -> str:
    """continue for change <= cont, stop-for-user up to stop, rollback beyond."""
    if change <= cont:
        return CONTINUE
    return STOP_FOR_USER if change <= stop else ROLLBACK


def later_gate(current: Metrics, previous: Metrics) -> Record:
    """Later gate against the previous gate of the same arm (R1 table)."""
    problems = _finite_problems(current, "current", loss=True) + _finite_problems(previous, "previous", loss=False)
    if problems:
        return {"rule": "R1", "kind": "later_gate", "outcome": STOP_FOR_USER, "rows": [], "problems": problems}
    edit_drop = _int(previous, "edit") - _int(current, "edit")
    nofp_rise = _int(current, "nofp") - _int(previous, "nofp")
    valid_drop = _int(previous, "valid") - _int(current, "valid")
    cap = _int(current, "cap")
    score_drop = _int(previous, "score250") - _int(current, "score250")
    valid_outcome = _band(valid_drop, 1, 4)
    if _int(current, "valid") < LATER_GATE_MIN_VALID:
        valid_outcome = ROLLBACK
    rows = [
        _row("edit_drop", edit_drop, "continue 0-2, stop 3-7, rollback >= 8", _band(edit_drop, 2, 7)),
        _row("nofp_rise", nofp_rise, "continue 0-1, stop 2-3, rollback >= 4", _band(nofp_rise, 1, 3)),
        _row("valid_drop", valid_drop, f"continue 0-1, stop 2-4, rollback >= 5 or valid < {LATER_GATE_MIN_VALID}"
             f" (valid {_int(current, 'valid')})", valid_outcome),
        _row("cap", cap, "continue 0-1, stop 2-4, rollback >= 5", _band(cap, 1, 4)),
        _row("score250_drop", score_drop, "continue 0-4, stop 5-9, rollback >= 10", _band(score_drop, 4, 9)),
    ]
    return {"rule": "R1", "kind": "later_gate", "outcome": worst([str(r["outcome"]) for r in rows]),
            "rows": rows, "problems": []}


def _row(name: str, value: object, band: str, outcome: str) -> Record:
    return {"name": name, "value": value, "band": band, "outcome": outcome}


def apply_r1(current: Metrics, *, first: bool, reference: Metrics) -> Record:
    """Apply R1; ``reference`` is the parent baseline for the first gate, else the previous gate."""
    result = first_gate(current, reference) if first else later_gate(current, reference)
    result["inputs"] = {"current": dict(current), "parent" if first else "previous": dict(reference)}
    return result


def failure(kind: str, reason: str) -> Record:
    """An evaluator crash or a wrong panel or checkpoint hash is always stop-for-user."""
    return {"rule": "R1", "kind": kind, "outcome": STOP_FOR_USER, "rows": [], "problems": [reason], "inputs": {}}
