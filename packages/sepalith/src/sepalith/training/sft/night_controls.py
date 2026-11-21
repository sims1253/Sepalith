"""Night-window controls for the editing-SFT trainer.

``DeadlineStop`` honours ``SEPALITH_DEADLINE_EPOCH`` from the night runner
(card C02): once the deadline has passed, the trainer finishes the optimizer
update in flight, saves full state and stops. ``stop_after_step`` is the same
mechanism at a fixed update, used by the C07 smoke run.

``pre_update_masking_gate`` ports the check from
``docs/campaign/work/corrective-target-only-gates-v1/target_only_gate.py``.
Before update 1 it pulls the first optimizer update's micro-batches from the
live trainer's dataloader and verifies that prompt and padding tokens are
masked and that the target body, the protocol terminal and EOS are
supervised. Any failure raises, which aborts training before the first update.
The fused-loss witness of the original gate is not ported: it was written for
the LoRA batch-4 geometry and needs CUDA kernels.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from typing import Protocol, SupportsInt

DEADLINE_ENV = "SEPALITH_DEADLINE_EPOCH"


class MaskingGateError(ValueError):
    """The live trainer's first batch is not target-only."""


class TrainerControl(Protocol):
    should_save: bool
    should_training_stop: bool


class GateArgs(Protocol):
    remove_unused_columns: bool
    dataloader_num_workers: int
    gradient_accumulation_steps: int
    per_device_train_batch_size: int


class GateParameter(Protocol):
    grad: object


class GateModel(Protocol):
    def parameters(self) -> Iterator[GateParameter]: ...


class GateTrainer(Protocol):
    """The parts of ``transformers.Trainer`` the masking gate reads."""

    args: GateArgs
    model: GateModel

    def get_batch_samples(self, epoch_iterator: Iterator[dict[str, object]], num_batches: int,
                          device: object) -> tuple[list[dict[str, object]], SupportsInt | None]: ...


def deadline_from_env(environ: Mapping[str, str] | None = None) -> float | None:
    value = (os.environ if environ is None else environ).get(DEADLINE_ENV)
    if value in (None, ""):
        return None
    deadline = float(value)
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError(f"{DEADLINE_ENV} must be a positive epoch")
    return deadline


@dataclass(frozen=True)
class DeadlineStop:
    """Save-and-stop decision at an optimizer boundary."""

    deadline_epoch: float | None = None
    stop_after_step: int | None = None

    def reason(self, step: int, now: float | None = None) -> str | None:
        if self.stop_after_step is not None and step >= self.stop_after_step:
            return "stop_after_step"
        if self.deadline_epoch is not None and (time.time() if now is None else now) >= self.deadline_epoch:
            return "deadline"
        return None

    def apply(self, step: int, control: TrainerControl, now: float | None = None) -> str | None:
        """Set the Trainer control flags to save and stop when the deadline applies."""
        reason = self.reason(step, now)
        if reason is not None:
            control.should_save = True
            control.should_training_stop = True
        return reason


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise MaskingGateError(f"pre_update_masking_gate: {reason}")


def verify_microbatch(batch: Mapping[str, object], rows: Sequence[Mapping[str, object]],
                      positions: Sequence[int] | None = None) -> dict[str, int]:
    """Check one collated micro-batch against the admitted rows it must contain."""
    import torch

    keys = set(batch)
    _require(keys in ({"input_ids", "attention_mask", "labels"}, {"input_ids", "attention_mask", "labels", "_draw_position"}),
             f"unexpected model inputs {sorted(keys)}")
    ids, attention, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    _require(isinstance(ids, torch.Tensor) and isinstance(attention, torch.Tensor) and isinstance(labels, torch.Tensor),
             "batch tensors missing")
    assert isinstance(ids, torch.Tensor) and isinstance(attention, torch.Tensor) and isinstance(labels, torch.Tensor)
    _require(ids.ndim == 2 and ids.shape == labels.shape == attention.shape and ids.shape[0] == len(rows), "batch shape")
    _require(ids.dtype == labels.dtype == torch.long, "token dtype")
    if "_draw_position" in batch:
        observed = batch["_draw_position"]
        _require(isinstance(observed, torch.Tensor) and positions is not None
                 and observed.tolist() == list(positions), "draw positions differ from the frozen schedule")
    counts = {"supervised_tokens": 0, "prompt_supervised_tokens": 0, "padding_supervised_tokens": 0,
              "eos_supervised_rows": 0, "rows": len(rows)}
    for index, row in enumerate(rows):
        row_ids = row["input_ids"]
        start = row["target_start"]
        body, terminal = row["target_body_tokens"], row["target_terminal_tokens"]
        _require(isinstance(row_ids, list) and type(start) is int and isinstance(body, list) and isinstance(terminal, list),
                 "row target metadata missing")
        assert isinstance(row_ids, list) and isinstance(start, int) and isinstance(body, list) and isinstance(terminal, list)
        end = len(row_ids)
        _require(end <= ids.shape[1] and 1 < start < end - 1, "boundary or truncation")
        _require(row_ids[start:] == body + terminal + [1], "target body, terminal and EOS metadata lost")
        _require(ids[index, :end].tolist() == row_ids, "dataloader changed tokens or order")
        _require(bool(attention[index, :end].eq(1).all()) and bool(attention[index, end:].eq(0).all()), "attention or padding")
        _require(bool(ids[index, end:].eq(1).all()), "padding token changed")
        prompt_supervised = int(labels[index, :start].ne(-100).sum())
        padding_supervised = int(labels[index, end:].ne(-100).sum())
        counts["prompt_supervised_tokens"] += prompt_supervised
        counts["padding_supervised_tokens"] += padding_supervised
        _require(prompt_supervised == 0, "prompt tokens are supervised")
        _require(padding_supervised == 0, "padding tokens are supervised")
        supervised = labels[index, start:end].tolist()
        _require(supervised[: len(body)] == body, "target body is not supervised")
        _require(supervised[len(body):len(body) + len(terminal)] == terminal, "protocol terminal is not supervised")
        _require(supervised[-1] == 1, "EOS is not supervised")
        counts["eos_supervised_rows"] += 1
        counts["supervised_tokens"] += end - start
    _require(int(labels[:, 1:].ne(-100).sum()) == counts["supervised_tokens"], "causal target denominator")
    return counts


def pre_update_masking_gate(trainer: GateTrainer, dataloader: object, expected_rows: Sequence[Mapping[str, object]],
                            *, accumulation: int, global_step: int) -> dict[str, object]:
    """Verify the first optimizer update's micro-batches before any update runs."""
    import torch
    from torch.utils.data import DataLoader

    started = time.monotonic()
    _require(global_step == 0, "the masking gate runs only before update 1")
    args = trainer.args
    _require(args.remove_unused_columns is False, "metadata columns may be removed")
    _require(args.dataloader_num_workers == 0, "worker prefetch is not reviewed")
    _require(args.gradient_accumulation_steps == accumulation, "accumulation geometry differs")
    _require(getattr(trainer, "compute_loss_func", None) is None, "alternate loss function")
    _require(getattr(trainer, "label_smoother", None) is None, "label smoothing")
    optimizer = getattr(trainer, "optimizer", None)
    _require(optimizer is None or not optimizer.state, "a fresh run has restored optimizer state")
    _require(all(p.grad is None for p in trainer.model.parameters()), "gradients exist before update 1")
    _require(isinstance(dataloader, DataLoader), "live dataloader missing")
    assert isinstance(dataloader, DataLoader)
    micro = args.per_device_train_batch_size
    _require(len(expected_rows) == micro * accumulation, "expected first-update rows missing")
    # Iterate the live loader's own sampler and collator without Accelerate's
    # begin/end bookkeeping. Creating an iterator draws a base seed from the
    # torch RNG, and stochastic rounding makes RNG state part of the run, so
    # every generator is restored afterwards and training is unaffected.
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generators = [g for g in (getattr(dataloader, "generator", None), getattr(dataloader.sampler, "generator", None))
                  if isinstance(g, torch.Generator)]
    generator_states = [g.get_state() for g in generators]
    try:
        iterator = DataLoader.__iter__(dataloader)
        batches, denominator = trainer.get_batch_samples(iterator, accumulation, torch.device("cpu"))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for generator, state in zip(generators, generator_states, strict=True):
            generator.set_state(state)
    _require(len(batches) == accumulation, "missing accumulation micro-batches")
    totals: dict[str, int] = {}
    per_batch = []
    for index, batch in enumerate(batches):
        rows = expected_rows[index * micro:(index + 1) * micro]
        counts = verify_microbatch(batch, rows, list(range(index * micro, (index + 1) * micro)))
        per_batch.append(counts["supervised_tokens"])
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    _require(denominator is not None and int(denominator) == totals["supervised_tokens"],
             "trainer loss denominator differs from the supervised target tokens")
    identity = hashlib.sha256(json.dumps([[row["input_ids"], row["target_start"]] for row in expected_rows],
                                         separators=(",", ":")).encode()).hexdigest()
    return {"schema": "sepalith.sft-v2.pre-update-masking-gate.v1", "status": "pass", "before_optimizer_step": 1,
            "micro_batches_checked": len(batches), "rows_checked": totals["rows"],
            "target_denominator": totals["supervised_tokens"], "micro_batch_denominators": per_batch,
            "prompt_supervised_tokens": totals["prompt_supervised_tokens"],
            "padding_supervised_tokens": totals["padding_supervised_tokens"],
            "eos_supervised_rows": totals["eos_supervised_rows"], "rows_sha256": identity,
            "ported_from": "docs/campaign/work/corrective-target-only-gates-v1/target_only_gate.py (verify_batch and trainer checks)",
            "not_ported": "fused-loss witness and reference loss (LoRA batch-4 geometry, CUDA only)",
            "elapsed_seconds": time.monotonic() - started}
