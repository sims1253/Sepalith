"""GRPO bookkeeping for the C15 recipe, without torch (testable in the core suite).

- Advantages are group-relative and not divided by the group standard
  deviation (card C15): ``A_i = r_i - mean(r_group)``. A group whose rewards are
  all equal is a zero-variance group and gets exactly zero advantages.
- Dynamic sampling (card C15): draw 1.5 times the groups a rollout buffer needs,
  drop zero-variance groups and refill with fresh prompts until the buffer is
  full. Refill stops after ``max_rounds`` rounds or ``max_draw_factor`` times the
  needed prompts; an update short of signal groups then trains on the ones it
  has, and its empty slots count as zero-variance groups after filtering (the
  quantity R6 watches). Surplus signal groups are discarded, never carried into
  the next buffer: a save at a buffer boundary leaves no partial rollout state.
- The pool sampler walks seeded per-epoch permutations of the pool; its state is
  one cursor, saved with every checkpoint.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
import random

ZERO_VARIANCE_TOLERANCE = 1e-9


class GRPOError(ValueError):
    """A GRPO invariant does not hold."""


def zero_variance(rewards: Sequence[float]) -> bool:
    if not rewards:
        raise GRPOError("a group needs rewards")
    if not all(math.isfinite(value) for value in rewards):
        raise GRPOError("non-finite reward")
    return max(rewards) - min(rewards) <= ZERO_VARIANCE_TOLERANCE


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """``r - mean(r)`` without std scaling; exactly zero for a zero-variance group."""
    if zero_variance(rewards):
        return [0.0] * len(rewards)
    mean = math.fsum(rewards) / len(rewards)
    return [value - mean for value in rewards]


@dataclass
class Group:
    """One prompt's G completions and their rewards; ``rows`` is opaque to the planner."""

    prompt_id: str
    position: int
    rewards: tuple[float, ...]
    rows: object = None

    @property
    def zero_variance(self) -> bool:
        return zero_variance(self.rewards)

    @property
    def advantages(self) -> list[float]:
        return group_advantages(self.rewards)


@dataclass
class BufferPlan:
    selected: list[Group]
    drawn: list[Group]
    rounds: int
    needed: int
    exhausted: bool
    surplus_discarded: int = 0
    round_sizes: list[int] = field(default_factory=list)

    @property
    def zero_variance_drawn(self) -> int:
        return sum(1 for group in self.drawn if group.zero_variance)

    @property
    def zero_variance_share_before(self) -> float:
        return self.zero_variance_drawn / len(self.drawn) if self.drawn else 1.0


def dynamic_sample(needed: int, draw: Callable[[int], Sequence[Group]], *, oversample: float = 1.5,
                   max_draw_factor: float = 4.0, max_rounds: int = 4) -> BufferPlan:
    """Fill a buffer with ``needed`` signal groups; ``draw(n)`` generates and scores n fresh prompts."""
    if needed < 1 or oversample < 1 or max_rounds < 1 or max_draw_factor < oversample:
        raise GRPOError("dynamic sampling settings are invalid")
    limit = math.ceil(max_draw_factor * needed)
    selected: list[Group] = []
    drawn: list[Group] = []
    sizes: list[int] = []
    surplus = 0
    rounds = 0
    while len(selected) < needed and rounds < max_rounds and len(drawn) < limit:
        size = min(math.ceil(oversample * (needed - len(selected))), limit - len(drawn))
        groups = list(draw(size))
        if len(groups) != size:
            raise GRPOError(f"draw returned {len(groups)} groups for {size} prompts")
        rounds += 1
        sizes.append(size)
        drawn.extend(groups)
        for group in groups:
            if group.zero_variance:
                continue
            if len(selected) < needed:
                selected.append(group)
            else:
                surplus += 1
    return BufferPlan(selected, drawn, rounds, needed, len(selected) < needed, surplus, sizes)


def assign_updates(selected: Sequence[Group], updates: int, groups_per_update: int) -> list[list[Group]]:
    """Consecutive slices of the selected groups, one per update; short buffers leave the last updates short."""
    if updates < 1 or groups_per_update < 1 or len(selected) > updates * groups_per_update:
        raise GRPOError("cannot assign groups to updates")
    return [list(selected[index * groups_per_update:(index + 1) * groups_per_update]) for index in range(updates)]


def zero_variance_share_after(groups: Sequence[Group], groups_per_update: int) -> float:
    """Share of an update's group slots without a signal group after refill."""
    if any(group.zero_variance for group in groups):
        raise GRPOError("zero-variance groups never enter an update")
    return (groups_per_update - len(groups)) / groups_per_update


class PoolSampler:
    """Seeded per-epoch permutations of the pool; ``cursor`` counts prompts drawn so far."""

    def __init__(self, ids: Sequence[str], seed: int, cursor: int = 0) -> None:
        if not ids or len(set(ids)) != len(ids):
            raise GRPOError("the pool needs distinct ids")
        if cursor < 0:
            raise GRPOError("the sampler cursor is negative")
        self.ids = list(ids)
        self.seed = seed
        self.cursor = cursor
        self._epoch: int | None = None
        self._order: list[int] = []

    def order(self, epoch: int) -> list[int]:
        if epoch != self._epoch:
            order = list(range(len(self.ids)))
            random.Random(f"sepalith-grpo-pool:{self.seed}:{epoch}").shuffle(order)
            self._epoch, self._order = epoch, order
        return self._order

    def draw(self, count: int) -> list[tuple[int, str]]:
        out = []
        for position in range(self.cursor, self.cursor + count):
            epoch, offset = divmod(position, len(self.ids))
            out.append((position, self.ids[self.order(epoch)[offset]]))
        self.cursor += count
        return out

    def state(self) -> dict[str, int]:
        return {"cursor": self.cursor, "seed": self.seed, "pool_size": len(self.ids)}
