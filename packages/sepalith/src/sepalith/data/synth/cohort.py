"""No-op exposure rule for SYNTH cohort schedules (user decision: each no-op
row is seen at most 3 times). Counted from the draws themselves; schedule_v2's
``noop_exposures_at_most_cap`` compares against its own pass count and is true
by construction, so it cannot enforce this rule."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json

NOOP_EXPOSURE_MAX = 3


def noop_exposures(schedule: Mapping) -> Counter:
    return Counter(d["row_id"] for d in schedule["draws"] if d.get("semantic_noop"))


def assert_noop_cap(schedule: Mapping, cap: int = NOOP_EXPOSURE_MAX) -> dict:
    counts = noop_exposures(schedule)
    worst = max(counts.values(), default=0)
    if worst > cap:
        raise AssertionError(f"a no-op row is drawn {worst} times (> {cap})")
    return {"max_noop_exposures": worst, "histogram": dict(sorted(Counter(counts.values()).items())), "cap": cap}


# ---------------------------------------------------------------------------
# Dose: SYNTH exposures and token shares (v3.1)
# ---------------------------------------------------------------------------
REPEAT_SEPARATOR = "~x"


def repeat_rows(lines: list[str], exposures: int) -> list[str]:
    """Each SYNTH row ``exposures`` times; copies 2..n get the id ``<id>~x<k>`` so the
    schedule builder (unique ids, every edit row once) draws each copy once."""
    out = list(lines)
    for k in range(2, exposures + 1):
        for line in lines:
            row = json.loads(line)
            row["id"] = f"{row['id']}{REPEAT_SEPARATOR}{k}"
            out.append(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def base_id(row_id: str) -> str:
    return row_id.split(REPEAT_SEPARATOR, 1)[0]


def dose(draws: Sequence[Mapping], synth_ids: set[str], length_class: Mapping[str, str]) -> dict:
    """Draw and token shares of SYNTH. ``length_class`` maps finish_block row ids
    (TRAIN and SYNTH, base ids) to short (<= 8 target lines), medium (9-20), long (> 20)."""
    def share(part: float, whole: float) -> float:
        return round(part / whole, 4) if whole else 0.0

    is_synth = [base_id(d["row_id"]) in synth_ids for d in draws]
    finish = [(d, s) for d, s in zip(draws, is_synth, strict=True) if d.get("family") == "finish_block"]
    ml = [(d, s) for d, s in finish if length_class.get(base_id(d["row_id"])) in ("medium", "long")]
    target = "supervised_target_tokens"
    finish_target = sum(d[target] for d, _ in finish)
    synth_finish_target = sum(d[target] for d, s in finish if s)
    synth_ml_target = sum(d[target] for d, s in ml if s)
    synth_finish_rows = [(d, s) for d, s in finish if s]
    return {
        "draws": {"all": len(draws), "finish": len(finish), "synth_all": sum(is_synth),
                  "synth_finish": len(synth_finish_rows),
                  "synth_share_of_finish_draws": share(len(synth_finish_rows), len(finish)),
                  "synth_share_of_all_draws": share(sum(is_synth), len(draws))},
        "tokens": {
            "meaning": "supervised target tokens include the terminal line and EOS; prompt tokens exclude BOS",
            "synth_share_of_finish_target_tokens": share(synth_finish_target, finish_target),
            "synth_share_of_all_supervised_tokens": share(sum(d[target] for d, s in zip(draws, is_synth, strict=True)
                                                              if s), sum(d[target] for d in draws)),
            "synth_share_of_finish_prompt_tokens": share(sum(d["prompt_tokens"] for d, s in finish if s),
                                                         sum(d["prompt_tokens"] for d, _ in finish)),
            "medium_long_share_of_finish_target_tokens": {
                "cohort": share(sum(d[target] for d, _ in ml), finish_target),
                "train": share(sum(d[target] for d, s in ml if not s), finish_target - synth_finish_target),
                "synth": share(synth_ml_target, synth_finish_target)},
            "synth_share_of_medium_long_finish_target_tokens": share(synth_ml_target, sum(d[target] for d, _ in ml)),
            "finish_target_tokens": finish_target, "synth_finish_target_tokens": synth_finish_target},
    }
