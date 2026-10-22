"""No-op exposure rule for SYNTH cohort schedules (user decision: each no-op
row is seen at most 3 times). Counted from the draws themselves; schedule_v2's
``noop_exposures_at_most_cap`` compares against its own pass count and is true
by construction, so it cannot enforce this rule."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

NOOP_EXPOSURE_MAX = 3


def noop_exposures(schedule: Mapping) -> Counter:
    return Counter(d["row_id"] for d in schedule["draws"] if d.get("semantic_noop"))


def assert_noop_cap(schedule: Mapping, cap: int = NOOP_EXPOSURE_MAX) -> dict:
    counts = noop_exposures(schedule)
    worst = max(counts.values(), default=0)
    if worst > cap:
        raise AssertionError(f"a no-op row is drawn {worst} times (> {cap})")
    return {"max_noop_exposures": worst, "histogram": dict(sorted(Counter(counts.values()).items())), "cap": cap}
