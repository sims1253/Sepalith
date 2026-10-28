"""C18 prompt mixing and the SFT replay schedule, without torch (core suite).

- ``MixedSampler`` interleaves the exec-task pool with the legacy TRAIN replay pool (the C14 pool,
  scored with reward v2.1): global draw position ``p`` is a legacy prompt iff ``p % every == every - 1``
  (``every = 5``: 20% of drawn prompts). Each pool walks its own seeded per-epoch permutation
  (``algorithm.PoolSampler``). The state is one cursor; the two pool cursors follow from it, so a
  checkpoint's ``sampler.cursor`` resumes both exactly.
- ``replay_ids(update)``: the SFT replay rows of an optimizer update are positions
  ``(update - 1) * rows .. update * rows - 1`` of a seeded per-epoch permutation of the replay rows,
  so the schedule needs no state beyond the update count.
"""
from __future__ import annotations

from collections.abc import Sequence

from sepalith.training.grpo.algorithm import GRPOError, PoolSampler

LEGACY_SEED_OFFSET = 1
REPLAY_SEED_OFFSET = 2


def legacy_before(cursor: int, every: int) -> int:
    """Legacy positions in ``[0, cursor)``."""
    return cursor // every


class MixedSampler:
    """Exec and legacy prompts at a fixed 1-in-``every`` legacy pattern; ``draw`` returns ``(position, id)``."""

    def __init__(self, exec_ids: Sequence[str], legacy_ids: Sequence[str], seed: int, every: int, cursor: int = 0) -> None:
        if every < 2:
            raise GRPOError("the legacy pattern needs every >= 2")
        if set(exec_ids) & set(legacy_ids):
            raise GRPOError("exec and legacy prompt ids overlap")
        if cursor < 0:
            raise GRPOError("the sampler cursor is negative")
        self.every = every
        self.seed = seed
        self.cursor = cursor
        legacy = legacy_before(cursor, every)
        self.exec = PoolSampler(exec_ids, seed, cursor - legacy)
        self.legacy = PoolSampler(legacy_ids, seed + LEGACY_SEED_OFFSET, legacy)
        self.legacy_set = frozenset(legacy_ids)

    def is_legacy_position(self, position: int) -> bool:
        return position % self.every == self.every - 1

    def draw(self, count: int) -> list[tuple[int, str]]:
        out = []
        for position in range(self.cursor, self.cursor + count):
            source = self.legacy if self.is_legacy_position(position) else self.exec
            ((_, identifier),) = source.draw(1)
            out.append((position, identifier))
        self.cursor += count
        return out

    def state(self) -> dict[str, int]:
        return {"cursor": self.cursor, "seed": self.seed, "pool_size": len(self.exec.ids), "legacy_pool_size": len(self.legacy.ids),
                "legacy_every": self.every, "exec_cursor": self.exec.cursor, "legacy_cursor": self.legacy.cursor}


def replay_ids(ids: Sequence[str], seed: int, update: int, rows: int) -> list[str]:
    """The SFT replay rows of optimizer update ``update`` (1-based)."""
    if update < 1 or rows < 1:
        raise GRPOError("replay needs update >= 1 and rows >= 1")
    sampler = PoolSampler(ids, seed + REPLAY_SEED_OFFSET, (update - 1) * rows)
    return [identifier for _, identifier in sampler.draw(rows)]
