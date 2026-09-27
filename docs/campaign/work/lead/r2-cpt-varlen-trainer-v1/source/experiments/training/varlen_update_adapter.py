"""One logical 16-draw CPT update executed as sequential variable-length packs.

The outer Transformers Trainer sees one dataset item and one optimizer step.
This mixin performs each physical pack's forward/backward immediately with the
same original supervised-token denominator, so no pack graph is retained.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class VarlenContractError(ValueError):
    pass


def require(value, message):
    if not value:
        raise VarlenContractError(message)


def supervised_tokens(row: Mapping[str, Any]) -> int:
    labels = row["labels"]
    return sum(value != -100 for value in labels[1:])


def pack_optimizer_window(rows: Sequence[Mapping[str, Any]], *, first_position: int,
                          token_cap: int, effective_batch: int = 16) -> dict:
    require(type(first_position) is int and first_position >= 0, "first draw position differs")
    require(type(token_cap) is int and token_cap > 0, "physical token cap differs")
    require(len(rows) == effective_batch, "optimizer update membership differs")
    packs, current, current_tokens = [], [], 0
    positions = []
    for offset, row in enumerate(rows):
        expected_position = first_position + offset
        require(row.get("_draw_position") == expected_position, "draw order differs")
        ids, labels, mask = row.get("input_ids"), row.get("labels"), row.get("attention_mask")
        require(isinstance(ids, list) and ids and isinstance(labels, list) and len(ids) == len(labels),
                "token/label arrays differ")
        require(mask == [1] * len(ids), "source row attention mask differs")
        require(labels[0] == -100, "member boundary label is exposed")
        require(len(ids) <= token_cap, "complete row exceeds physical token cap")
        if current and current_tokens + len(ids) > token_cap:
            packs.append(_collate_pack(current)); current, current_tokens = [], 0
        current.append(row); current_tokens += len(ids); positions.append(expected_position)
    if current:
        packs.append(_collate_pack(current))
    denominator = sum(supervised_tokens(row) for row in rows)
    require(denominator > 0, "optimizer update has no supervised tokens")
    require(sum(pack["local_supervised_tokens"] for pack in packs) == denominator,
            "packed denominator differs")
    return {"physical_packs": packs, "loss_denominator": denominator,
            "draw_positions": positions, "logical_rows": effective_batch}


def _collate_pack(rows: Sequence[Mapping[str, Any]]) -> dict:
    input_ids, labels, positions, lengths = [], [], [], []
    for row in rows:
        length = len(row["input_ids"])
        input_ids.extend(row["input_ids"]); labels.extend(row["labels"])
        positions.extend(range(length)); lengths.append(length)
    return {"input_ids": [input_ids], "labels": [labels], "position_ids": [positions],
            "packed_seq_lengths": lengths,
            "local_supervised_tokens": sum(supervised_tokens(row) for row in rows)}


class PackedOptimizerWindowDataset:
    """View an already cursor-sliced sequential dataset as logical updates."""
    def __init__(self, dataset, *, first_position: int, token_cap: int, effective_batch: int = 16):
        require(type(first_position) is int and first_position >= 0 and first_position % effective_batch == 0,
                "resume cursor is not an optimizer boundary")
        require(len(dataset) % effective_batch == 0, "remaining draws are not whole optimizer updates")
        self.dataset = dataset; self.first_position = first_position
        self.token_cap = token_cap; self.effective_batch = effective_batch

    def __len__(self):
        return len(self.dataset) // self.effective_batch

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        local = index * self.effective_batch
        rows = [self.dataset[local + offset] for offset in range(self.effective_batch)]
        return pack_optimizer_window(rows, first_position=self.first_position + local,
                                     token_cap=self.token_cap, effective_batch=self.effective_batch)

    def close(self):
        close = getattr(self.dataset, "close", None)
        if close is not None:
            close()


def logical_update_collator(batch):
    require(len(batch) == 1, "Trainer physical batch must contain one logical update")
    return batch[0]


class VarlenUpdateTrainerMixin:
    """Trainer seam: many physical backwards followed by one outer optimizer step."""
    def before_varlen_update(self, draw_positions, denominator, physical_packs):
        pass

    @staticmethod
    def _tensorize_pack(pack):
        import torch
        allowed = {"input_ids", "labels", "position_ids", "packed_seq_lengths",
                   "local_supervised_tokens"}
        require(set(pack) == allowed, "physical pack fields differ")
        return {
            "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "position_ids": torch.tensor(pack["position_ids"], dtype=torch.long),
            "packed_seq_lengths": torch.tensor(pack["packed_seq_lengths"], dtype=torch.int32),
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        import torch
        require(num_items_in_batch is None, "outer Trainer supplied an unexpected denominator")
        require(self.args.gradient_accumulation_steps == 1,
                "physical packs require Trainer gradient_accumulation_steps=1")
        require(self.args.world_size == 1, "prepared varlen seam is single-device only")
        require(self.model_accepts_loss_kwargs is True,
                "model must accept explicit num_items_in_batch")
        require(set(inputs) == {"physical_packs", "loss_denominator", "draw_positions", "logical_rows"},
                "logical update fields differ")
        packs = inputs["physical_packs"]; denominator = inputs["loss_denominator"]
        positions = inputs["draw_positions"]
        require(inputs["logical_rows"] == len(positions) and len(positions) > 0,
                "logical update membership differs")
        require(type(denominator) is int and denominator > 0, "loss denominator differs")
        require(sum(pack["local_supervised_tokens"] for pack in packs) == denominator,
                "physical pack denominators differ")
        self.before_varlen_update(positions, denominator, len(packs))
        model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        total = None
        for pack in packs:
            prepared = self._prepare_inputs(self._tensorize_pack(pack))
            with self.compute_loss_context_manager():
                # Skip any subclass compute_loss cursor hook: this method owns
                # the logical cursor and passes only model inputs downstream.
                loss = super(VarlenUpdateTrainerMixin, self).compute_loss(
                    model, prepared, num_items_in_batch=denominator)
            require(torch.is_tensor(loss) and loss.ndim == 0 and bool(torch.isfinite(loss).item()),
                    "physical pack loss is invalid")
            self.accelerator.backward(loss)
            detached = loss.detach()
            total = detached if total is None else total + detached
            del loss, prepared
        require(total is not None, "logical update contains no physical pack")
        return total
