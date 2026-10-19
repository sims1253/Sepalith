"""Pre-registered decision rules for the C13 offline steps (RFT and DPO).

Gates (R1-style, docs/training/plan-20260927/status/C13.json ``gate_plan``):

- Every gate of an offline step, including the first, applies the R1
  later-gate table (``gate_rule.later_gate``). The reference of the first gate
  is the starting model's own bound gate result (for example arm A's gate 536);
  later gates compare with the previous gate of the same step. R1's first-gate
  floor ("edit beats the parent's edit") is written for SFT from a CPT parent
  that cannot edit; an offline step starts from an already gated SFT model.
- Two extra rows, both stop-for-user, guard against the failure modes the card
  and the wiki review name:

  - ``noedit_dodge``: edit cases answered ``[NO_EDIT]`` (DEV75 plus DEV250)
    rise by 3 or more over the reference while exact edits (DEV75 ``edit`` plus
    DEV250 exact edits) do not rise. The policy is dodging errors by abstaining.
  - ``dpo_chosen_collapse`` (DPO only): the chosen log-ratio
    log pi(chosen) - log pi_ref(chosen), averaged over blocks of 50 updates,
    fell in each of the last 3 blocks and the last block is negative, while
    ``score250`` is not above the starting model's.

- The worst row wins; a NaN, an evaluator crash or a binding problem is
  stop-for-user, as in R1.

Adoption (R5, RULES.md), compared with the model the step started from: adopt
if ``score250`` rises by 5 or more, ``nofp`` rises by at most 1 and ``valid``
drops by at most 1; stop-for-user if ``score250`` rises by 1 to 4 with the
other two conditions met; discard otherwise.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

from sepalith.training.sft import gate_rule

ADOPT = "adopt"
DISCARD = "discard"
R5_OUTCOMES = (ADOPT, gate_rule.STOP_FOR_USER, DISCARD)
R5_MIN_RISE = 5
NOEDIT_DODGE_RISE = 3
COLLAPSE_BLOCK = 50
COLLAPSE_BLOCKS = 3
EXTRA_METRICS = ("noedit", "edit_total")

Record = dict[str, object]


def _counts(summary: Mapping[str, object]) -> Mapping[str, object]:
    counts = summary["counts"]
    if not isinstance(counts, Mapping):
        raise TypeError("evaluator summary needs a counts mapping")
    return cast(Mapping[str, object], counts)


def _count(counts: Mapping[str, object], key: str) -> int:
    value = counts[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"metric count must be a nonnegative integer: {key}={value!r}")
    return value


def offline_metrics(dev75: Mapping[str, object], dev250: Mapping[str, object]) -> dict[str, int]:
    """The five R1 counts plus ``noedit`` and ``edit_total`` from two evaluator summaries."""
    metrics = gate_rule.metrics_from_summaries(dev75, dev250)
    c75, c250 = _counts(dev75), _counts(dev250)
    noedit = sum(_count(c, "predicted_noop") - _count(c, "strict_noop_correct") for c in (c75, c250))
    if noedit < 0:
        raise ValueError("predicted no-ops are fewer than correct no-ops")
    metrics.update(noedit=noedit, edit_total=_count(c75, "edit_exact") + _count(c250, "edit_exact"))
    return metrics


def _int(metrics: Mapping[str, object], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _row(name: str, value: object, band: str, outcome: str) -> Record:
    return {"name": name, "value": value, "band": band, "outcome": outcome}


def noedit_dodge_row(current: Mapping[str, object], reference: Mapping[str, object]) -> Record:
    rise = _int(current, "noedit") - _int(reference, "noedit")
    edit_rise = _int(current, "edit_total") - _int(reference, "edit_total")
    dodge = rise >= NOEDIT_DODGE_RISE and edit_rise <= 0
    return _row("noedit_dodge", {"noedit_rise": rise, "edit_total_rise": edit_rise},
                f"stop if [NO_EDIT] on edit cases rises by >= {NOEDIT_DODGE_RISE} while exact edits do not rise",
                gate_rule.STOP_FOR_USER if dodge else gate_rule.CONTINUE)


def block_means(values: Sequence[float], block: int = COLLAPSE_BLOCK) -> list[float]:
    """Means of consecutive full blocks; a trailing partial block is ignored."""
    return [math.fsum(values[i:i + block]) / block for i in range(0, len(values) - block + 1, block)]


def dpo_chosen_collapse_row(chosen_log_ratios: Sequence[float], score250: int, start_score250: int) -> Record:
    """The card's stop condition: chosen log-probability falling steadily while score250 does not improve."""
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in chosen_log_ratios):
        return _row("dpo_chosen_collapse", "non-finite chosen log-ratio", "finite", gate_rule.STOP_FOR_USER)
    means = block_means(chosen_log_ratios)
    tail = means[-(COLLAPSE_BLOCKS + 1):]
    falling = len(tail) == COLLAPSE_BLOCKS + 1 and all(b < a for a, b in zip(tail, tail[1:], strict=False)) and tail[-1] < 0
    collapse = falling and score250 <= start_score250
    return _row("dpo_chosen_collapse", {"block_means": [round(m, 4) for m in tail], "score250": score250,
                                        "start_score250": start_score250},
                f"stop if the {COLLAPSE_BLOCK}-update chosen log-ratio means fell {COLLAPSE_BLOCKS} times in a row to "
                "below 0 while score250 <= the starting model's", gate_rule.STOP_FOR_USER if collapse else gate_rule.CONTINUE)


def offline_gate(current: Mapping[str, object], reference: Mapping[str, object], *, first: bool,
                 chosen_log_ratios: Sequence[float] | None = None, start_score250: int | None = None) -> Record:
    """R1 later-gate table against ``reference`` plus the offline guard rows."""
    rule = gate_rule.later_gate(current, reference)
    rule["rule"] = "R1-offline"
    rule["kind"] = "offline_first_gate" if first else "offline_later_gate"
    problems = cast(list[str], rule["problems"])
    for label, metrics in (("current", current), ("reference", reference)):
        for name in EXTRA_METRICS:
            if type(metrics.get(name)) is not int:
                problems.append(f"{label}.{name} is missing or not a count")
    if not problems:
        rows = cast(list[Record], rule["rows"])
        rows.append(noedit_dodge_row(current, reference))
        if chosen_log_ratios is not None:
            if start_score250 is None:
                raise ValueError("the DPO guard needs the starting model's score250")
            rows.append(dpo_chosen_collapse_row(chosen_log_ratios, _int(current, "score250"), start_score250))
        rule["outcome"] = gate_rule.worst([str(row["outcome"]) for row in rows])
    else:
        rule["outcome"] = gate_rule.STOP_FOR_USER
    rule["inputs"] = {"current": dict(current), "start_model" if first else "previous": dict(reference)}
    return rule


def apply_r5(result: Mapping[str, object], start: Mapping[str, object]) -> Record:
    """R5: offline step adoption against the model the step started from."""
    problems = [f"{label}.{name} is missing or not a count" for label, metrics in (("result", result), ("start", start))
                for name in ("score250", "nofp", "valid") if type(metrics.get(name)) is not int]
    if problems:
        return {"rule": "R5", "outcome": gate_rule.STOP_FOR_USER, "rows": [], "problems": problems,
                "inputs": {"result": dict(result), "start": dict(start)}}
    rise = _int(result, "score250") - _int(start, "score250")
    nofp_rise = _int(result, "nofp") - _int(start, "nofp")
    valid_drop = _int(start, "valid") - _int(result, "valid")
    guards = nofp_rise <= 1 and valid_drop <= 1
    if guards and rise >= R5_MIN_RISE:
        outcome = ADOPT
    elif guards and 1 <= rise < R5_MIN_RISE:
        outcome = gate_rule.STOP_FOR_USER
    else:
        outcome = DISCARD
    rows = [_row("score250_rise", rise, "adopt >= 5, stop-for-user 1-4, discard <= 0", outcome),
            _row("nofp_rise", nofp_rise, "<= 1", "pass" if nofp_rise <= 1 else "fail"),
            _row("valid_drop", valid_drop, "<= 1", "pass" if valid_drop <= 1 else "fail")]
    return {"rule": "R5", "outcome": outcome, "rows": rows, "problems": [],
            "inputs": {"result": dict(result), "start": dict(start)}}
