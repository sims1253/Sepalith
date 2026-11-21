"""C20 stop requests and first-update masking checks for mixed edit and CPT rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from sepalith.training.sft.night_controls import (
    DeadlineStop as EditDeadlineStop,
    GateTrainer,
    _require,
    verify_microbatch as verify_edit_microbatch,
)


def stop_payload(recipe_hash: str) -> dict[str, str]:
    return {"action": "save_and_stop", "bound_recipe_sha256": recipe_hash}


@dataclass(frozen=True)
class DeadlineStop(EditDeadlineStop):
    stop_path: Path | None = None
    recipe_hash: str = ""

    def reason(self, step: int, now: float | None = None) -> str | None:
        if self.stop_path is not None and self.stop_path.exists():
            if json.loads(self.stop_path.read_text()) != stop_payload(self.recipe_hash):
                raise ValueError("graceful stop request identity differs")
            return "graceful_stop"
        return super().reason(step, now)


def verify_microbatch(
    batch: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    positions: Sequence[int] | None = None,
) -> dict[str, int]:
    """Check one collated micro-batch against the admitted rows it must contain."""
    if all(row.get("family") != "cpt_replay" for row in rows):
        return verify_edit_microbatch(batch, rows, positions)

    import torch

    from sepalith.training.sft.campaign_edit_data import verify_batch

    model_inputs = {key: value for key, value in batch.items() if key != "_draw_position"}
    _require(
        model_inputs.keys() == {"input_ids", "attention_mask", "labels"},
        "unexpected model inputs",
    )
    ids, attention, labels = (model_inputs[key] for key in ("input_ids", "attention_mask", "labels"))
    _require(
        isinstance(ids, torch.Tensor)
        and isinstance(attention, torch.Tensor)
        and isinstance(labels, torch.Tensor),
        "batch tensors missing",
    )
    assert (
        isinstance(ids, torch.Tensor)
        and isinstance(attention, torch.Tensor)
        and isinstance(labels, torch.Tensor)
    )
    _require(
        ids.ndim == 2 and ids.shape == labels.shape == attention.shape and ids.shape[0] == len(rows),
        "batch shape",
    )
    _require(ids.dtype == labels.dtype == torch.long, "token dtype")
    if "_draw_position" in batch:
        observed = batch["_draw_position"]
        _require(
            isinstance(observed, torch.Tensor)
            and positions is not None
            and observed.tolist() == list(positions),
            "draw positions differ from the frozen schedule",
        )
    eos = 0
    for index, row in enumerate(rows):
        tokens, start = row["input_ids"], row["target_start"]
        _require(
            isinstance(tokens, list) and type(start) is int,
            "row target metadata missing",
        )
        assert isinstance(tokens, list) and isinstance(start, int)
        end = len(tokens)
        _require(0 < start < end <= ids.shape[1], "boundary or truncation")
        _require(bool(ids[index, end:].eq(1).all()), "padding token changed")
        eos += int(tokens[-1] == 1)
    supervised = verify_batch(model_inputs, rows)
    return {
        "supervised_tokens": supervised,
        "prompt_supervised_tokens": 0,
        "padding_supervised_tokens": 0,
        "eos_supervised_rows": eos,
        "rows": len(rows),
    }


def pre_update_masking_gate(
    trainer: GateTrainer,
    dataloader: object,
    expected_rows: Sequence[Mapping[str, object]],
    *,
    accumulation: int,
    global_step: int,
) -> dict[str, object]:
    """Verify the first optimizer update's micro-batches before any update runs."""
    import torch
    from torch.utils.data import DataLoader

    started = time.monotonic()
    _require(global_step == 0, "the masking gate runs only before update 1")
    args = trainer.args
    _require(args.remove_unused_columns is False, "metadata columns may be removed")
    _require(args.dataloader_num_workers == 0, "worker prefetch is not reviewed")
    _require(
        args.gradient_accumulation_steps == accumulation,
        "accumulation geometry differs",
    )
    _require(getattr(trainer, "compute_loss_func", None) is None, "alternate loss function")
    _require(getattr(trainer, "label_smoother", None) is None, "label smoothing")
    optimizer = getattr(trainer, "optimizer", None)
    _require(
        optimizer is None or not optimizer.state,
        "a fresh run has restored optimizer state",
    )
    _require(
        all(p.grad is None for p in trainer.model.parameters()),
        "gradients exist before update 1",
    )
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
    generators = [
        g
        for g in (
            getattr(dataloader, "generator", None),
            getattr(dataloader.sampler, "generator", None),
        )
        if isinstance(g, torch.Generator)
    ]
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
        rows = expected_rows[index * micro : (index + 1) * micro]
        counts = verify_microbatch(batch, rows, list(range(index * micro, (index + 1) * micro)))
        per_batch.append(counts["supervised_tokens"])
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    _require(
        denominator is not None and int(denominator) == totals["supervised_tokens"],
        "trainer loss denominator differs from the supervised target tokens",
    )
    identity = hashlib.sha256(
        json.dumps(
            [[row["input_ids"], row["target_start"]] for row in expected_rows],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema": "sepalith.sft-v2.pre-update-masking-gate.v1",
        "status": "pass",
        "before_optimizer_step": 1,
        "micro_batches_checked": len(batches),
        "rows_checked": totals["rows"],
        "target_denominator": totals["supervised_tokens"],
        "micro_batch_denominators": per_batch,
        "prompt_supervised_tokens": totals["prompt_supervised_tokens"],
        "padding_supervised_tokens": totals["padding_supervised_tokens"],
        "eos_supervised_rows": totals["eos_supervised_rows"],
        "rows_sha256": identity,
        "ported_from": "docs/campaign/work/corrective-target-only-gates-v1/target_only_gate.py (verify_batch and trainer checks)",
        "not_ported": "fused-loss witness and reference loss (LoRA batch-4 geometry, CUDA only)",
        "elapsed_seconds": time.monotonic() - started,
    }
