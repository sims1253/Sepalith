#!/usr/bin/env python3
"""CPU contract for isolation-preserving varlen packing within one update.

This is a preparation seam, not an admitted trainer. A CUDA probe must prove
that the patched Llama path consumes reset position_ids through FlashAttention
varlen and produces matching loss/gradients before production use.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence


class PackError(ValueError): pass


def _validate(row: Mapping[str, Any], expected_position: int, cap: int) -> None:
    ids, labels = row.get("input_ids"), row.get("labels")
    if not isinstance(ids, list) or not isinstance(labels, list) or len(ids) != len(labels) or not ids:
        raise PackError("row token/label arrays differ")
    if len(ids) > cap: raise PackError("row exceeds physical token cap")
    if row.get("_draw_position") != expected_position: raise PackError("draw cursor differs")
    if labels[0] != -100: raise PackError("member BOS label must remain masked")


def pack_update(rows: Sequence[Mapping[str, Any]], *, first_position: int,
                effective_batch: int = 16, physical_token_cap: int = 16_384) -> list[list[Mapping[str, Any]]]:
    """First-fit rows, in original order, without moving a row across updates."""
    if len(rows) != effective_batch: raise PackError("one update must contain exact effective batch")
    bins: list[list[Mapping[str, Any]]] = []
    totals: list[int] = []
    for index, row in enumerate(rows):
        _validate(row, first_position + index, physical_token_cap)
        length = len(row["input_ids"])
        for slot, total in enumerate(totals):
            if total + length <= physical_token_cap:
                bins[slot].append(row); totals[slot] += length; break
        else:
            bins.append([row]); totals.append(length)
    return bins


def collate_varlen(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten members with reset positions and no attention_mask.

    Transformers 5.5 FlashAttention recognizes reset position_ids as packed
    sequence boundaries. The absence of attention_mask is intentional. The
    first label of every member remains -100, preventing a cross-boundary
    next-token loss. Actual model compatibility remains a CUDA gate.
    """
    if not members: raise PackError("physical batch is empty")
    ids: list[int] = []; labels: list[int] = []; positions: list[int] = []
    slices = []; cursor = 0
    for row in members:
        rid, values, targets = row.get("row_id", row.get("id")), row["input_ids"], row["labels"]
        if not isinstance(rid, str) or not rid: raise PackError("member row id is invalid")
        if targets[0] != -100: raise PackError("member boundary label is not masked")
        start = cursor; stop = start + len(values)
        ids.extend(values); labels.extend(targets); positions.extend(range(len(values)))
        slices.append({"row_id": rid, "draw_position": row["_draw_position"], "start": start, "stop": stop})
        cursor = stop
    return {"input_ids": [ids], "labels": [labels], "position_ids": [positions],
            "member_slices": slices, "loss_denominator": sum(v != -100 for v in labels[1:])}


def verify_conservation(rows: Sequence[Mapping[str, Any]], packed: Mapping[str, Any]) -> None:
    flat_ids, flat_labels, pos = packed["input_ids"][0], packed["labels"][0], packed["position_ids"][0]
    if len(flat_ids) != len(flat_labels) or len(flat_ids) != len(pos): raise PackError("flat arrays differ")
    by_position = {row["_draw_position"]: row for row in rows}
    if len(by_position) != len(rows): raise PackError("duplicate draw position")
    for record in packed["member_slices"]:
        row = by_position.get(record["draw_position"]); start, stop = record["start"], record["stop"]
        if row is None or flat_ids[start:stop] != row["input_ids"] or flat_labels[start:stop] != row["labels"]:
            raise PackError("member token or label slice changed")
        if pos[start:stop] != list(range(stop-start)): raise PackError("member positions do not reset")
        if flat_labels[start] != -100: raise PackError("cross-member next-token label is exposed")
