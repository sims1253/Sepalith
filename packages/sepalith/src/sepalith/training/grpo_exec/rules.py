"""Card C18 in-run gate (R6x: R6 adapted to the exec environment), checkpoint selection and the
pre-registered final endpoints, as pure functions (core suite).

Every ``EVAL_EVERY`` (50) updates the driver evaluates the checkpoint on DEV75, the validation split
(greedy, exec reward) and the CPT 2K forgetting probe, plus DEV250 every ``DEV250_EVERY`` (150). The
starting model (arm A 803) is the reference for everything except the R1 DEV75 rows, which compare
with the previous accepted evaluation as in R6. Rows (the worst wins; missing or non-finite inputs,
an evaluator crash or a binding problem are stop-for-user):

| Row | Continue | Stop for user | Rollback |
| --- | --- | --- | --- |
| R1 DEV75 vs previous: edit drop / nofp rise / valid drop / cap | as R6 | as R6 | as R6 (valid < 68 too) |
| validation all-pass drop vs start (points) | < 2.0 | 2.0 to < 4.0 | >= 4.0 |
| validation mean exec reward drop vs start | < 0.02 | 0.02 to < 0.04 | >= 0.04 |
| CPT 2K NLL rise vs start (nats, same harness) | <= 0.015 | 0.015 to 0.03 | > 0.03 |
| DEV250 score250 drop vs start (DEV250 steps) | <= 4 | 5 to 9 | >= 10 |
| zero-variance share after filtering > 50% at 3 evaluations in a row | no | yes | |
| mean completion length beyond x2 or /2 of the first 10 updates | no | yes | |

An evaluation whose outcome is continue (by rule or human override) is *accepted*. The pilot is
complete at ``MAX_UPDATES`` or when the best validation all-pass has not risen by at least
``PLATEAU_GAIN`` over the best before the last ``PLATEAU_EVALUATIONS`` accepted evaluations (the start
counts). The selected checkpoint is the accepted one with the highest validation all-pass (then the
highest validation mean reward, then the earliest). The frozen hold-out is never used in-run.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

from sepalith.training.grpo import rules as r6

CONTINUE, STOP_FOR_USER, ROLLBACK = r6.CONTINUE, r6.STOP_FOR_USER, r6.ROLLBACK
OUTCOMES = r6.OUTCOMES

EVAL_EVERY = 50
DEV250_EVERY = 150
MAX_UPDATES = 1000
START_WINDOW = 10
VAL_PASS_STOP, VAL_PASS_ROLLBACK = 0.02, 0.04
VAL_REWARD_STOP, VAL_REWARD_ROLLBACK = 0.02, 0.04
CPT_STOP, CPT_ROLLBACK = 0.015, 0.03
SCORE250_CONTINUE, SCORE250_STOP = 4, 9
PLATEAU_EVALUATIONS = 6
PLATEAU_GAIN = 0.01
WINDOW_FIELDS = (*r6.WINDOW_FIELDS, "exec_reward_mean_drawn", "legacy_reward_mean_drawn", "exec_all_pass_drawn",
                 "exec_zero_variance_share_before", "legacy_zero_variance_share_before", "replay_nll", "grad_norm",
                 "grad_norm_grpo")
# Final evaluation (pre-registered)
SCORE250_MARGIN = 4
CPT_FINAL_LIMIT = 0.03

Record = dict[str, object]


def evaluation_steps(max_updates: int = MAX_UPDATES) -> list[int]:
    return list(range(EVAL_EVERY, max_updates + 1, EVAL_EVERY))


def evaluation_panels(update: int) -> tuple[str, ...]:
    if update <= 0 or update % EVAL_EVERY:
        return ()
    base = ("dev75", "val", "cpt")
    return (*base, "dev250") if update % DEV250_EVERY == 0 else base


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _row(name: str, value: object, band: str, outcome: str) -> Record:
    return {"name": name, "value": value, "band": band, "outcome": outcome}


def _drop_band(drop: float, stop: float, rollback: float) -> str:
    if drop >= rollback:
        return ROLLBACK
    return STOP_FOR_USER if drop >= stop else CONTINUE


def window_stats(events: Sequence[Mapping[str, object]], first: int, last: int) -> Record:
    """R6 window statistics plus the C18 fields (means over the updates ``first..last``)."""
    out = r6.window_stats(events, first, last)
    by_update = {cast(int, e["update"]): e for e in events if e.get("event") == "grpo_update" and type(e.get("update")) is int}
    for name in WINDOW_FIELDS:
        if name in out:
            continue
        values = [by_update[u].get(name) for u in range(first, last + 1)]
        finite = [float(cast(float, v)) for v in values if _finite(v)]
        out[name] = math.fsum(finite) / len(finite) if finite else None
    return out


def apply(current: Mapping[str, object], *, previous75: Mapping[str, object], start: Mapping[str, object],
          window: Mapping[str, object], prior_windows: Sequence[Mapping[str, object]],
          start_window: Mapping[str, object]) -> Record:
    """R6x at one evaluation.

    ``current``: DEV75 counts (valid, cap, edit, nofp), ``val_all_pass``, ``val_mean_reward``, ``cpt_nll``,
    ``loss`` (window mean) and ``score250`` at DEV250 steps. ``start``: the starting model's DEV75
    counts, ``score250``, ``val_all_pass``, ``val_mean_reward`` and ``cpt_nll`` (same harnesses).
    """
    problems: list[str] = []
    rows = r6.r1_rows(current, previous75, None, problems)
    for name in ("val_all_pass", "val_mean_reward", "cpt_nll", "loss"):
        if not _finite(current.get(name)):
            problems.append(f"current.{name} is missing or not finite ({current.get(name)!r})")
    for name in ("val_all_pass", "val_mean_reward", "cpt_nll"):
        if not _finite(start.get(name)):
            problems.append(f"start.{name} is missing or not finite ({start.get(name)!r})")
    dev250 = "score250" in current
    if dev250 and (type(current.get("score250")) is not int or type(start.get("score250")) is not int):
        problems.append("score250 of the current or the start evaluation is missing")
    shares = [w.get("zero_variance_share_after") for w in [*prior_windows, window]][-r6.ZERO_VARIANCE_STREAK:]
    length_now, length_start = window.get("completion_tokens_mean_drawn"), start_window.get("completion_tokens_mean_drawn")
    if not all(_finite(value) for value in shares):
        problems.append("a zero-variance share is missing or not finite")
    if not (_finite(length_now) and _finite(length_start) and float(cast(float, length_start)) > 0):
        problems.append("the completion-length window or start is missing")
    if not problems:
        # differences rounded to 1e-9 so a band edge (0.38 - 0.34 = 0.0399999...) falls where it reads
        pass_drop = round(float(cast(float, start["val_all_pass"])) - float(cast(float, current["val_all_pass"])), 9)
        rows.append(_row("val_all_pass_drop", round(pass_drop, 6),
                         f"continue < {VAL_PASS_STOP}, stop {VAL_PASS_STOP} to < {VAL_PASS_ROLLBACK}, rollback >= {VAL_PASS_ROLLBACK}",
                         _drop_band(pass_drop, VAL_PASS_STOP, VAL_PASS_ROLLBACK)))
        reward_drop = round(float(cast(float, start["val_mean_reward"])) - float(cast(float, current["val_mean_reward"])), 9)
        rows.append(_row("val_mean_reward_drop", round(reward_drop, 6),
                         f"continue < {VAL_REWARD_STOP}, stop {VAL_REWARD_STOP} to < {VAL_REWARD_ROLLBACK}, "
                         f"rollback >= {VAL_REWARD_ROLLBACK}", _drop_band(reward_drop, VAL_REWARD_STOP, VAL_REWARD_ROLLBACK)))
        rise = round(float(cast(float, current["cpt_nll"])) - float(cast(float, start["cpt_nll"])), 9)
        cpt = ROLLBACK if rise > CPT_ROLLBACK else (STOP_FOR_USER if rise > CPT_STOP else CONTINUE)
        rows.append(_row("cpt_nll_rise", round(rise, 6),
                         f"continue <= {CPT_STOP}, stop <= {CPT_ROLLBACK}, rollback > {CPT_ROLLBACK} nats", cpt))
        if dev250:
            drop = cast(int, start["score250"]) - cast(int, current["score250"])
            outcome = CONTINUE if drop <= SCORE250_CONTINUE else (STOP_FOR_USER if drop <= SCORE250_STOP else ROLLBACK)
            rows.append(_row("score250_drop_from_start", drop, "continue 0-4, stop 5-9, rollback >= 10", outcome))
        streak = len(shares) == r6.ZERO_VARIANCE_STREAK and all(float(cast(float, s)) > r6.ZERO_VARIANCE_LIMIT for s in shares)
        rows.append(_row("zero_variance_streak", [round(float(cast(float, s)), 4) for s in shares],
                         "stop if > 50% after filtering at 3 evaluations in a row", STOP_FOR_USER if streak else CONTINUE))
        factor = float(cast(float, length_now)) / float(cast(float, length_start))
        rows.append(_row("length_factor", round(factor, 4), "stop beyond x2 or /2 of the first 10 updates",
                         STOP_FOR_USER if factor > r6.LENGTH_FACTOR or factor < 1 / r6.LENGTH_FACTOR else CONTINUE))
    outcome = STOP_FOR_USER if problems else r6.worst([str(row["outcome"]) for row in rows])
    return {"rule": "R6x", "kind": "c18_dev75_val_cpt" + ("_dev250" if dev250 else ""), "outcome": outcome, "rows": rows,
            "problems": problems,
            "inputs": {"current": dict(current), "previous_dev75": dict(previous75), "start": dict(start),
                       "window": dict(window), "prior_windows": [dict(w) for w in prior_windows],
                       "start_window": dict(start_window)}}


def failure(reason: str) -> Record:
    return {"rule": "R6x", "kind": "failure", "outcome": STOP_FOR_USER, "rows": [], "problems": [reason], "inputs": {}}


def select(accepted: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    """The accepted evaluation with the highest val all-pass, then val mean reward, then the earliest step."""
    if not accepted:
        return None
    return sorted(accepted, key=lambda e: (-float(cast(float, e["val_all_pass"])), -float(cast(float, e["val_mean_reward"])),
                                           int(cast(int, e["step"]))))[0]


def plateau(start_val_all_pass: float, accepted: Sequence[Mapping[str, object]]) -> bool:
    """True when the last ``PLATEAU_EVALUATIONS`` accepted evaluations never beat the earlier best by ``PLATEAU_GAIN``."""
    if len(accepted) < PLATEAU_EVALUATIONS:
        return False
    ordered = sorted(accepted, key=lambda e: int(cast(int, e["step"])))
    earlier = [start_val_all_pass, *(float(cast(float, e["val_all_pass"])) for e in ordered[:-PLATEAU_EVALUATIONS])]
    recent = [float(cast(float, e["val_all_pass"])) for e in ordered[-PLATEAU_EVALUATIONS:]]
    return max(recent) < max(earlier) + PLATEAU_GAIN


def final_verdict(holdout: Mapping[str, object], *, dev250_candidate: int, dev250_start: int, cpt_rise: float) -> Record:
    """Pre-registered endpoints from the hold-out paired comparison (``metrics[candidate]['vs_<start>']``).

    Primary: the all-pass difference (candidate - 803, function-clustered bootstrap 95% CI) has a lower
    bound > 0. Secondary (reported; each must hold for adoption): mean exec reward and exact-match
    differences, protocol validity, DEV250 score250 >= start - 4, CPT 2K rise <= 0.03 nats.
    """
    def cell(name: str) -> Mapping[str, object]:
        return cast(Mapping[str, object], holdout[name])
    all_pass = cell("all_pass")
    ci = cast(Sequence[float], all_pass["ci95"])
    primary = float(ci[0]) > 0
    secondary = {
        "mean_reward_difference": dict(cell("reward")), "exact_match_difference": dict(cell("exact_match")),
        "dev250_score250": {"candidate": dev250_candidate, "start": dev250_start,
                            "non_inferior": dev250_candidate >= dev250_start - SCORE250_MARGIN},
        "cpt_nll_rise": {"value": cpt_rise, "within_limit": cpt_rise <= CPT_FINAL_LIMIT},
    }
    adopt = primary and bool(cast(Mapping[str, object], secondary["dev250_score250"])["non_inferior"]) \
        and bool(cast(Mapping[str, object], secondary["cpt_nll_rise"])["within_limit"])
    return {"primary": {"endpoint": "hold-out all-pass difference vs arm A 803 (paired, function-clustered 95% CI)",
                        "estimate": all_pass["mean"], "ci95": list(ci), "met": primary},
            "secondary": secondary, "adopt": adopt}
