"""Sigmoid DPO on the full-weight runtime (card C13 step 3).

- ``completion_logp``: the summed log-probability of a sequence's completion
  tokens (body, protocol terminal and EOS: the SFT target positions), from one
  unpadded forward pass with ``logits_to_keep`` so only completion logits are
  materialized. Prompt tokens never enter the sum.
- ``dpo_loss``: standard sigmoid DPO, -log sigmoid(beta * ((pi_c - ref_c) -
  (pi_r - ref_r))), beta 0.1. Reference log-probabilities are precomputed once
  by ``offline.reference`` and stored, so no reference model is resident.
- ``PairDataset`` and ``collate_pair``: one pair per micro-batch, in the frozen
  pair schedule's order, with the stored reference values attached.
- ``DPOLossMixin``: ``compute_loss`` for ``transformers.Trainer`` (with the
  full-weight optimizer mixin: aurora_mix, bf16 stochastic rounding). The batch
  carries no ``labels`` key, so the Trainer divides each pair's loss by the
  gradient accumulation and an update optimizes the mean over its 16 pairs.
- ``pair_batch_gate`` and ``reference_alignment``: before update 1, the live
  first update must hold the scheduled pairs with the right prompt/completion
  split, and the policy's log-probabilities at step 0 must match the stored
  reference values (a misaligned reference differs by many nats).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import hashlib
import json
import math
import time
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

BETA = 0.1
EOS = 1
ALIGNMENT_MEAN_ABS = 0.5
ALIGNMENT_MAX_ABS = 2.0
BATCH_KEYS = {"chosen_input_ids", "rejected_input_ids", "target_start", "ref_logps", "pair_index", "_draw_position"}

Json = dict[str, object]


class _ModelOutput(Protocol):
    logits: torch.Tensor


class DPOGateError(ValueError):
    """The first DPO update is not the scheduled, correctly split pairs, or the reference is misaligned."""


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise DPOGateError(reason)


def completion_logp(model: Callable[..., _ModelOutput], input_ids: torch.Tensor, target_start: int) -> torch.Tensor:
    """Summed log-probability of ``input_ids[0, target_start:]`` given its prefix (one row, no padding)."""
    import torch
    import torch.nn.functional as functional

    _require(input_ids.ndim == 2 and input_ids.shape[0] == 1, "completion_logp takes one unpadded row")
    length = int(input_ids.shape[1])
    _require(1 < target_start < length, "prompt/completion boundary outside the row")
    keep = length - target_start + 1
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), logits_to_keep=keep, use_cache=False)
    logits = output.logits
    _require(logits.shape[1] == keep, "the model ignored logits_to_keep")
    # Position p predicts token p + 1: positions target_start-1 .. length-2 predict the completion.
    predicted = logits[0, :-1, :].float()
    targets = input_ids[0, target_start:]
    return -functional.cross_entropy(predicted, targets, reduction="sum")


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float = BETA) -> tuple[torch.Tensor, Json]:
    """Sigmoid DPO loss and its per-pair statistics."""
    import torch.nn.functional as functional

    chosen_ratio = policy_chosen - ref_chosen
    rejected_ratio = policy_rejected - ref_rejected
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -functional.logsigmoid(logits)
    stats: Json = {"loss": float(loss.detach().mean()), "chosen_log_ratio": float(chosen_ratio.detach().mean()),
                   "rejected_log_ratio": float(rejected_ratio.detach().mean()),
                   "margin": float(logits.detach().mean() / beta), "accuracy": float((logits.detach() > 0).float().mean()),
                   "policy_chosen_logp": float(policy_chosen.detach().mean()),
                   "policy_rejected_logp": float(policy_rejected.detach().mean())}
    return loss.mean(), stats


# Data ------------------------------------------------------------------------------------------

class PairRows:
    """Pairs, their stored reference log-probabilities and the frozen pair schedule."""

    def __init__(self, pairs: Sequence[Mapping[str, object]], draws: Sequence[int],
                 references: Sequence[tuple[float, float]] | None) -> None:
        _require(references is None or len(references) == len(pairs), "one reference pair per DPO pair")
        self.pairs = [validate_pair(pair) for pair in pairs]
        self.draws = list(draws)
        _require(all(type(i) is int and 0 <= i < len(self.pairs) for i in self.draws), "pair schedule index outside the pairs")
        self.references = list(references) if references is not None else None

    def item(self, position: int) -> Json:
        index = self.draws[position]
        pair = self.pairs[index]
        prompt = cast(list[int], pair["prompt_ids"])
        reference = self.references[index] if self.references is not None else (math.nan, math.nan)
        return {"chosen_ids": prompt + cast(list[int], pair["chosen_ids"]),
                "rejected_ids": prompt + cast(list[int], pair["rejected_ids"]), "target_start": len(prompt),
                "ref": [reference[0], reference[1]], "pair_index": index, "_draw_position": position}


def validate_pair(pair: Mapping[str, object]) -> Json:
    prompt, chosen, rejected = pair.get("prompt_ids"), pair.get("chosen_ids"), pair.get("rejected_ids")
    for name, ids in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
        _require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < 130560 for x in ids), f"{name} ids invalid")
    prompt_ids, chosen_ids, rejected_ids = cast(list[int], prompt), cast(list[int], chosen), cast(list[int], rejected)
    _require(prompt_ids[0] == 0 and prompt_ids.count(0) == 1 and EOS not in prompt_ids, "prompt must start with BOS only")
    for name, ids in (("chosen", chosen_ids), ("rejected", rejected_ids)):
        _require(ids[-1] == EOS and ids.count(EOS) == 1 and 0 not in ids, f"{name} completion must end with its only EOS")
    _require(chosen_ids != rejected_ids, "chosen equals rejected")
    _require(pair.get("split") == "train", "a DPO pair must be TRAIN-derived")
    return dict(pair)


class PairDataset:
    """``torch`` dataset view of ``PairRows`` in schedule order."""

    def __init__(self, rows: PairRows) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows.draws)

    def __getitem__(self, position: int) -> Json:
        return self.rows.item(position)


def collate_pair(items: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
    import torch

    _require(len(items) == 1, "DPO micro-batches hold exactly one pair")
    item = items[0]
    ref = cast(list[float], item["ref"])
    return {"chosen_input_ids": torch.tensor([cast(list[int], item["chosen_ids"])], dtype=torch.long),
            "rejected_input_ids": torch.tensor([cast(list[int], item["rejected_ids"])], dtype=torch.long),
            "target_start": torch.tensor([cast(int, item["target_start"])], dtype=torch.long),
            "ref_logps": torch.tensor([ref], dtype=torch.float64),
            "pair_index": torch.tensor([cast(int, item["pair_index"])], dtype=torch.long),
            "_draw_position": torch.tensor([cast(int, item["_draw_position"])], dtype=torch.long)}


# Trainer ---------------------------------------------------------------------------------------

class _TrainerState(Protocol):
    global_step: int


class DPOLossMixin:
    """``compute_loss`` for pair batches. Set ``dpo_beta``; read ``dpo_records`` after each update."""

    dpo_beta: float = BETA
    state: _TrainerState

    def dpo_positions(self, positions: list[int]) -> None:
        """Hook for the frozen-schedule cursor check."""

    def compute_loss(self, model: Callable[..., _ModelOutput], inputs: dict[str, torch.Tensor], return_outputs: bool = False,
                     num_items_in_batch: object = None) -> torch.Tensor:
        _require(set(inputs) == BATCH_KEYS, f"DPO batch keys differ: {sorted(inputs)}")
        self.dpo_positions([int(x) for x in inputs["_draw_position"].detach().cpu().tolist()])
        start = int(inputs["target_start"][0])
        policy_chosen = completion_logp(model, inputs["chosen_input_ids"], start)
        policy_rejected = completion_logp(model, inputs["rejected_input_ids"], start)
        ref = inputs["ref_logps"][0].to(policy_chosen.device)
        _require(bool(ref.isfinite().all()), "reference log-probabilities missing")
        loss, stats = dpo_loss(policy_chosen, policy_rejected, ref[0].float(), ref[1].float(), self.dpo_beta)
        stats.update(pair_index=int(inputs["pair_index"][0]), step=int(self.state.global_step),
                     ref_chosen_logp=float(ref[0]), ref_rejected_logp=float(ref[1]))
        records = getattr(self, "dpo_records", None)
        if records is None:
            records = []
            self.dpo_records = records
        records.append(stats)
        return loss


def update_summary(records: Sequence[Mapping[str, object]]) -> Json:
    """Mean statistics of one optimizer update's pairs."""
    keys = ("loss", "chosen_log_ratio", "rejected_log_ratio", "margin", "accuracy")
    count = len(records)
    _require(count > 0, "no pair statistics for the update")
    return {"pairs": count, **{key: math.fsum(float(cast(float, r[key])) for r in records) / count for key in keys}}


def reference_alignment(records: Sequence[Mapping[str, object]], *, mean_abs: float = ALIGNMENT_MEAN_ABS,
                        max_abs: float = ALIGNMENT_MAX_ABS) -> Json:
    """Before update 1 the policy is the reference model: stored and live log-probabilities must agree."""
    diffs = [abs(float(cast(float, r[key]))) for r in records for key in ("chosen_log_ratio", "rejected_log_ratio")]
    _require(diffs, "no step-0 pair statistics")
    result = {"schema": "sepalith.c13.reference-alignment.v1", "pairs": len(records),
              "mean_abs_nats": math.fsum(diffs) / len(diffs), "max_abs_nats": max(diffs),
              "limits": {"mean_abs_nats": mean_abs, "max_abs_nats": max_abs}}
    _require(result["mean_abs_nats"] <= mean_abs and result["max_abs_nats"] <= max_abs,
             f"stored reference log-probabilities do not match the starting policy: {json.dumps(result)}")
    return {**result, "status": "pass"}


class _GateTrainer(Protocol):
    def get_batch_samples(self, epoch_iterator: Iterator[dict[str, object]], num_batches: int,
                          device: object) -> tuple[list[dict[str, object]], object]: ...


def pair_batch_gate(trainer: _GateTrainer, dataloader: object, expected: Sequence[Mapping[str, object]], *,
                    accumulation: int, global_step: int) -> Json:
    """Verify the first update's live micro-batches against the scheduled pairs before update 1."""
    import torch
    from torch.utils.data import DataLoader

    started = time.monotonic()
    _require(global_step == 0, "the pair gate runs only before update 1")
    _require(isinstance(dataloader, DataLoader), "live dataloader missing")
    assert isinstance(dataloader, DataLoader)
    _require(len(expected) == accumulation, "expected first-update pairs missing")
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generators = [g for g in (getattr(dataloader, "generator", None), getattr(dataloader.sampler, "generator", None))
                  if isinstance(g, torch.Generator)]
    states = [g.get_state() for g in generators]
    try:
        batches, _ = trainer.get_batch_samples(DataLoader.__iter__(dataloader), accumulation, torch.device("cpu"))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for generator, state in zip(generators, states, strict=True):
            generator.set_state(state)
    _require(len(batches) == accumulation, "missing accumulation micro-batches")
    completion_tokens = 0
    for position, (batch, item) in enumerate(zip(batches, expected, strict=True)):
        _require(set(batch) == BATCH_KEYS, f"unexpected model inputs {sorted(batch)}")
        start = int(cast("torch.Tensor", batch["target_start"])[0])
        chosen = cast("torch.Tensor", batch["chosen_input_ids"])[0].tolist()
        rejected = cast("torch.Tensor", batch["rejected_input_ids"])[0].tolist()
        _require(int(cast("torch.Tensor", batch["_draw_position"])[0]) == position, "draw position differs from the schedule")
        _require(int(cast("torch.Tensor", batch["pair_index"])[0]) == item["pair_index"], "pair differs from the schedule")
        _require(chosen == item["chosen_ids"] and rejected == item["rejected_ids"], "dataloader changed tokens")
        _require(start == item["target_start"] and chosen[:start] == rejected[:start], "chosen and rejected prompts differ")
        _require(chosen[start:] != rejected[start:] and chosen[-1] == EOS and rejected[-1] == EOS,
                 "completions must differ and end with EOS")
        ref = cast("torch.Tensor", batch["ref_logps"])[0].tolist()
        _require(ref == list(cast(list[float], item["ref"])) and all(math.isfinite(v) and v <= 0 for v in ref),
                 "reference log-probabilities differ or are not log-probabilities")
        completion_tokens += len(chosen) - start + len(rejected) - start
    identity = hashlib.sha256(json.dumps([[item["chosen_ids"], item["rejected_ids"], item["target_start"]] for item in expected],
                                         separators=(",", ":")).encode()).hexdigest()
    return {"schema": "sepalith.c13.dpo-pair-batch-gate.v1", "status": "pass", "before_optimizer_step": 1,
            "micro_batches_checked": len(batches), "completion_tokens": completion_tokens, "pairs_sha256": identity,
            "elapsed_seconds": time.monotonic() - started}
