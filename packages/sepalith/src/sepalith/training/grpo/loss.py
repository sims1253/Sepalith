"""The C15 policy loss: token-normalised DAPO surrogate with asymmetric clipping, beta = 0.

For one update with rows i (completion tokens t, advantage A_i, current and
rollout-time log-probabilities):

    ratio = exp(logp - logp_old)
    loss  = sum_i sum_t -min(ratio * A_i, clip(ratio, 1 - 0.2, 1 + 0.28) * A_i) / N

``N`` is the number of completion tokens of all rows that enter the loss in
this update (TRL 0.24 ``loss_type="dapo"``: the per-token loss summed over the
batch and divided by the batch's completion tokens). Truncated completions
(no end-of-generation token before the 1,024-token cap) never enter the loss
and do not count in ``N`` (TRL ``mask_truncated_completions``); their reward
still enters the group mean. There is no KL term and no reference model.

The rows are processed one at a time (micro-batch 1): each row's loss is
divided by the update's ``N`` before ``backward``, so the accumulated gradient
equals the gradient of the update loss. The clip fraction follows TRL: tokens
whose ratio lies below ``1 - eps_low`` with a negative advantage, or above
``1 + eps_high`` with a positive one, over all loss tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28


@dataclass
class ClipStats:
    tokens: int = 0
    clipped_low: int = 0
    clipped_high: int = 0
    ratio_sum: float = 0.0
    ratio_max: float = 0.0
    ratio_min: float = float("inf")

    def add(self, other: ClipStats) -> None:
        self.tokens += other.tokens
        self.clipped_low += other.clipped_low
        self.clipped_high += other.clipped_high
        self.ratio_sum += other.ratio_sum
        self.ratio_max = max(self.ratio_max, other.ratio_max)
        self.ratio_min = min(self.ratio_min, other.ratio_min)

    def summary(self) -> dict[str, float | int | None]:
        tokens = self.tokens
        return {"loss_tokens": tokens,
                "clip_fraction": (self.clipped_low + self.clipped_high) / tokens if tokens else None,
                "clip_fraction_low": self.clipped_low / tokens if tokens else None,
                "clip_fraction_high": self.clipped_high / tokens if tokens else None,
                "ratio_mean": self.ratio_sum / tokens if tokens else None,
                "ratio_max": self.ratio_max if tokens else None,
                "ratio_min": self.ratio_min if tokens else None}


def row_surrogate(logp: torch.Tensor, old_logp: torch.Tensor, advantage: float, *, eps_low: float = EPSILON_LOW,
                  eps_high: float = EPSILON_HIGH) -> tuple[torch.Tensor, ClipStats]:
    """Summed clipped surrogate loss of one row's tokens (divide by the update's ``N``) and its clip counts."""
    if logp.shape != old_logp.shape or logp.ndim != 1:
        raise ValueError("logp and old_logp must be matching 1-D token vectors")
    old = old_logp.detach().to(logp.dtype)
    ratio = torch.exp(logp - old)
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    per_token = -torch.minimum(ratio * advantage, clipped * advantage)
    with torch.no_grad():
        detached = ratio.detach()
        low = int(((detached < 1.0 - eps_low) & (advantage < 0)).sum())
        high = int(((detached > 1.0 + eps_high) & (advantage > 0)).sum())
        stats = ClipStats(int(detached.numel()), low, high, float(detached.sum()),
                          float(detached.max()) if detached.numel() else 0.0,
                          float(detached.min()) if detached.numel() else float("inf"))
    return cast(torch.Tensor, per_token.sum()), stats


def update_token_count(lengths: list[int], truncated: list[bool]) -> int:
    """``N``: completion tokens of the rows that enter the loss (truncated rows excluded)."""
    if len(lengths) != len(truncated):
        raise ValueError("lengths and truncation flags differ in length")
    return sum(length for length, cut in zip(lengths, truncated, strict=True) if not cut)
