"""Rule R6 (docs/training/plan-20260927/RULES.md) for the C15 GRPO pilot, as a pure function.

R6: evaluate DEV75 every 50 updates and DEV250 every 150 updates.

- The R1 later-gate table against the previous evaluation. An evaluation
  without DEV250 (updates 50, 100, 200, ...) applies the DEV75 rows (edit,
  nofp, valid, cap); ``score250`` is compared at the DEV250 evaluations, against
  the previous DEV250 evaluation. The first evaluation's reference is the
  starting model's bound gate result (arm A 536).
- ``zero_variance_streak``: stop-for-user if the share of zero-variance groups
  after filtering, averaged over each evaluation's window of updates, is above
  50% at three evaluations in a row.
- ``length_factor``: stop-for-user if the window's mean completion length
  (all generated rows) is more than twice or less than half the start, where
  the start is the mean over the run's first ``START_WINDOW`` updates.
- ``reward_hacking`` (DEV250 evaluations): rollback if the mean training
  reward of the window since the previous DEV250 evaluation is above the start
  window's while ``score250`` is 10 or more below the starting model's.

The worst row wins. Missing or non-finite inputs, an evaluator crash or a
binding problem are stop-for-user, as in R1.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

CONTINUE = "continue"
STOP_FOR_USER = "stop_for_user"
ROLLBACK = "rollback"
OUTCOMES = (CONTINUE, STOP_FOR_USER, ROLLBACK)
_SEVERITY = {CONTINUE: 0, STOP_FOR_USER: 1, ROLLBACK: 2}

DEV75_EVERY = 50
DEV250_EVERY = 150
MAX_UPDATES = 1500
START_WINDOW = 10
ZERO_VARIANCE_LIMIT = 0.5
ZERO_VARIANCE_STREAK = 3
LENGTH_FACTOR = 2.0
REWARD_HACK_SCORE_DROP = 10
LATER_GATE_MIN_VALID = 68  # R1
DEV75_METRICS = ("valid", "cap", "edit", "nofp")
WINDOW_FIELDS = ("zero_variance_share_before", "zero_variance_share_after", "completion_tokens_mean_drawn",
                 "reward_mean_drawn", "exact_rate_drawn", "mean_abs_advantage", "clip_fraction", "loss")

Record = dict[str, object]


def worst(outcomes: Sequence[str]) -> str:
    return max(outcomes, key=_SEVERITY.__getitem__) if outcomes else CONTINUE


def evaluation_panels(update: int) -> tuple[str, ...]:
    """The panels R6 evaluates at ``update`` (none between evaluations)."""
    if update <= 0 or update % DEV75_EVERY:
        return ()
    return ("dev75", "dev250") if update % DEV250_EVERY == 0 else ("dev75",)


def evaluation_steps(max_updates: int = MAX_UPDATES) -> list[int]:
    return list(range(DEV75_EVERY, max_updates + 1, DEV75_EVERY))


def _band(change: int, cont: int, stop: int) -> str:
    if change <= cont:
        return CONTINUE
    return STOP_FOR_USER if change <= stop else ROLLBACK


def _row(name: str, value: object, band: str, outcome: str) -> Record:
    return {"name": name, "value": value, "band": band, "outcome": outcome}


def _count(metrics: Mapping[str, object], name: str, label: str, problems: list[str]) -> int | None:
    value = metrics.get(name)
    if type(value) is not int or value < 0:
        problems.append(f"{label}.{name} is missing or not a count ({value!r})")
        return None
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def r1_rows(current: Mapping[str, object], previous75: Mapping[str, object],
            previous250: Mapping[str, object] | None, problems: list[str], *, label: str = "previous") -> list[Record]:
    """The R1 later-gate rows; ``score250`` only when ``previous250`` is given."""
    now = {name: _count(current, name, "current", problems) for name in DEV75_METRICS}
    before = {name: _count(previous75, name, label, problems) for name in DEV75_METRICS}
    score_now = score_before = None
    if previous250 is not None:
        score_now = _count(current, "score250", "current", problems)
        score_before = _count(previous250, "score250", f"{label}_dev250", problems)
    if problems:
        return []
    values = cast(dict[str, int], now)
    reference = cast(dict[str, int], before)
    edit_drop = reference["edit"] - values["edit"]
    nofp_rise = values["nofp"] - reference["nofp"]
    valid_drop = reference["valid"] - values["valid"]
    valid_outcome = ROLLBACK if values["valid"] < LATER_GATE_MIN_VALID else _band(valid_drop, 1, 4)
    rows = [
        _row("edit_drop", edit_drop, "continue 0-2, stop 3-7, rollback >= 8", _band(edit_drop, 2, 7)),
        _row("nofp_rise", nofp_rise, "continue 0-1, stop 2-3, rollback >= 4", _band(nofp_rise, 1, 3)),
        _row("valid_drop", valid_drop, f"continue 0-1, stop 2-4, rollback >= 5 or valid < {LATER_GATE_MIN_VALID} "
             f"(valid {values['valid']})", valid_outcome),
        _row("cap", values["cap"], "continue 0-1, stop 2-4, rollback >= 5", _band(values["cap"], 1, 4)),
    ]
    if score_now is not None and score_before is not None:
        drop = score_before - score_now
        rows.append(_row("score250_drop", drop, "continue 0-4, stop 5-9, rollback >= 10", _band(drop, 4, 9)))
    return rows


def window_stats(events: Sequence[Mapping[str, object]], first: int, last: int) -> Record:
    """Means of the per-update telemetry over updates ``first..last``; every update must appear exactly once."""
    by_update: dict[int, Mapping[str, object]] = {}
    for event in events:
        if event.get("event") != "grpo_update":
            continue
        update = event.get("update")
        if type(update) is int and first <= update <= last:
            if update in by_update:
                raise ValueError(f"update {update} appears twice in the telemetry")
            by_update[update] = event
    missing = [update for update in range(first, last + 1) if update not in by_update]
    if missing:
        raise ValueError(f"telemetry lacks updates {missing[:5]}{'...' if len(missing) > 5 else ''}")
    out: Record = {"first": first, "last": last, "updates": last - first + 1}
    for name in WINDOW_FIELDS:
        values = [by_update[update].get(name) for update in range(first, last + 1)]
        finite = [float(cast(float, value)) for value in values if _finite(value)]
        out[name] = math.fsum(finite) / len(finite) if finite else None
    out["updates_without_signal"] = sum(1 for update in range(first, last + 1)
                                        if by_update[update].get("groups_trained") == 0)
    return out


def apply_r6(current: Mapping[str, object], *, previous75: Mapping[str, object],
             previous250: Mapping[str, object] | None, start: Mapping[str, object],
             window: Mapping[str, object], prior_windows: Sequence[Mapping[str, object]],
             start_window: Mapping[str, object], reward_window: Mapping[str, object] | None) -> Record:
    """R6 at one evaluation.

    ``current`` holds the DEV75 counts, ``score250`` at DEV250 evaluations and
    the window's mean ``loss``. ``previous75``/``previous250`` are the previous
    DEV75 and DEV250 evaluations (the starting model for the first ones).
    ``window`` and ``prior_windows`` (oldest first) are ``window_stats`` of the
    evaluation windows; ``reward_window`` covers the updates since the previous
    DEV250 evaluation and is given exactly at DEV250 evaluations.
    """
    problems: list[str] = []
    dev250 = previous250 is not None
    if dev250 != (reward_window is not None):
        problems.append("the reward window must accompany exactly the DEV250 evaluations")
    rows = r1_rows(current, previous75, previous250, problems)
    if not _finite(current.get("loss")):
        problems.append(f"current.loss is missing or not finite ({current.get('loss')!r})")
    shares = [w.get("zero_variance_share_after") for w in [*prior_windows, window]][-ZERO_VARIANCE_STREAK:]
    length_now, length_start = window.get("completion_tokens_mean_drawn"), start_window.get("completion_tokens_mean_drawn")
    if not all(_finite(value) for value in shares):
        problems.append("a zero-variance share is missing or not finite")
    if not (_finite(length_now) and _finite(length_start) and float(cast(float, length_start)) > 0):
        problems.append("the completion-length window or start is missing")
    reward_now = reward_window.get("reward_mean_drawn") if reward_window is not None else None
    reward_start = start_window.get("reward_mean_drawn")
    if dev250 and not (_finite(reward_now) and _finite(reward_start)):
        problems.append("the reward window or start is missing")
    start_score = _count(start, "score250", "start", problems) if dev250 else None
    if not problems:
        streak = len(shares) == ZERO_VARIANCE_STREAK and all(float(cast(float, s)) > ZERO_VARIANCE_LIMIT for s in shares)
        rows.append(_row("zero_variance_streak", [round(float(cast(float, s)), 4) for s in shares],
                         f"stop if > {ZERO_VARIANCE_LIMIT:.0%} after filtering at {ZERO_VARIANCE_STREAK} evaluations in a row",
                         STOP_FOR_USER if streak else CONTINUE))
        factor = float(cast(float, length_now)) / float(cast(float, length_start))
        rows.append(_row("length_factor", round(factor, 4),
                         f"stop if the mean completion length moves beyond x{LENGTH_FACTOR:g} or /{LENGTH_FACTOR:g} of the start",
                         STOP_FOR_USER if factor > LENGTH_FACTOR or factor < 1 / LENGTH_FACTOR else CONTINUE))
        if dev250:
            reward_rise = float(cast(float, reward_now)) - float(cast(float, reward_start))
            score_drop = cast(int, start_score) - cast(int, current["score250"])
            hacking = reward_rise > 0 and score_drop >= REWARD_HACK_SCORE_DROP
            rows.append(_row("reward_hacking", {"reward_rise": round(reward_rise, 6), "score250_drop_from_start": score_drop},
                             f"rollback if the reward rises while score250 falls by >= {REWARD_HACK_SCORE_DROP} from the start",
                             ROLLBACK if hacking else CONTINUE))
    outcome = STOP_FOR_USER if problems else worst([str(row["outcome"]) for row in rows])
    return {"rule": "R6", "kind": "grpo_dev75_dev250" if dev250 else "grpo_dev75", "outcome": outcome, "rows": rows,
            "problems": problems,
            "inputs": {"current": dict(current), "previous_dev75": dict(previous75),
                       "previous_dev250": dict(previous250) if previous250 is not None else None,
                       "window": dict(window), "prior_windows": [dict(w) for w in prior_windows],
                       "start_window": dict(start_window),
                       "reward_window": dict(reward_window) if reward_window is not None else None}}


def r1_versus_start(current: Mapping[str, object], start: Mapping[str, object]) -> Record:
    """The R1 later-gate table against the starting model (R8: which checkpoint may go to ``nightly/``)."""
    problems: list[str] = []
    rows = r1_rows(current, start, start if "score250" in current else None, problems, label="start")
    return {"rule": "R1-vs-start", "outcome": STOP_FOR_USER if problems else worst([str(r["outcome"]) for r in rows]),
            "rows": rows, "problems": problems, "with_score250": "score250" in current}


def failure(reason: str) -> Record:
    """An evaluator crash or a binding problem: stop-for-user."""
    return {"rule": "R6", "kind": "failure", "outcome": STOP_FOR_USER, "rows": [], "problems": [reason], "inputs": {}}
