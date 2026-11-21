"""Validate C20 CPT replay rows and collate their exact causal targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from sepalith.training.sft import campaign_edit_data as edits


def target_end(row: dict) -> list[int]:
    return [] if row.get("family") == "cpt_replay" and row["input_ids"][-1] != 1 else [1]


def validate_cpt_row(row: dict, max_sequence_tokens: int) -> dict:
    ids = row["input_ids"]
    start = row["target_start"]
    body = row["target_body_tokens"]
    if (
        row.get("split") != "train"
        or row.get("renderer_id") != "c18b-cpt-text"
        or row.get("tokenizer_json_sha256")
        != "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
        or not 1 <= start <= 2
        or not start < len(ids) <= max_sequence_tokens
        or any(type(x) is not int or not 0 <= x < 130560 for x in ids)
        or ids[0] != 0
        or 0 in ids[1:]
        or 1 in ids[:-1]
        or not body
        or row["target_terminal_tokens"] != []
        or ids[start:] != body + target_end(row)
    ):
        raise ValueError(f"{row['id']}: invalid CPT replay row")
    return row


def validate_row(row: dict, max_sequence_tokens: int) -> dict:
    if row.get("family") == "cpt_replay":
        return validate_cpt_row(row, max_sequence_tokens)
    return edits.validate_row(row, max_sequence_tokens)


def inspect(
    rows_record: dict,
    schedule_record: dict,
    max_sequence_tokens: int,
    effective_batch: int = 16,
) -> tuple[list[dict], list[int], dict, dict[str, int]]:
    rows_path = Path(rows_record["path"])
    schedule_path = Path(schedule_record["path"])
    if edits.sha(rows_path) != rows_record["sha256"] or edits.sha(schedule_path) != schedule_record["sha256"]:
        raise ValueError("data/schedule hash differs")
    rows = []
    index = {}
    tot = {"input_tokens": 0, "loss_tokens": 0, "body_tokens": 0}
    with rows_path.open() as f:
        for line in f:
            x = validate_row(json.loads(line), max_sequence_tokens)
            rid = x["id"]
            if rid in index:
                raise ValueError("duplicate row ID")
            index[rid] = len(rows)
            rows.append(x)
            tot["input_tokens"] += len(x["input_ids"])
            tot["loss_tokens"] += len(x["input_ids"]) - x["target_start"]
            tot["body_tokens"] += len(x["target_body_tokens"])
    schedule = json.loads(schedule_path.read_text())
    records = schedule.get("draws")
    if (
        schedule.get("schema") != "sepalith.sft.full-coverage-batchmix-draws.v2"
        or schedule.get("input", {}).get("sha256") != rows_record["sha256"]
        or schedule.get("excluded") != []
    ):
        raise ValueError("schedule source/exclusions differ")
    if schedule.get("row_ids") != sorted(index) or schedule.get("coverage", {}).get(
        "distinct_rows_drawn"
    ) != len(rows):
        raise ValueError("schedule eligible identity differs")
    if (
        not isinstance(records, list)
        or len(records) != schedule.get("draw_count")
        or len(records) != schedule.get("max_steps") * effective_batch
    ):
        raise ValueError("schedule draw count differs")
    draws = []
    for pos, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValueError(f"draw is not an object: {pos}")
        rid = r.get("row_id")
        row = rows[index[rid]] if rid in index else None
        if (
            row is None
            or r.get("draw_index") != pos
            or r.get("update") != pos // effective_batch + 1
            or r.get("total_tokens") != len(row["input_ids"])
            or r.get("supervised_target_tokens") != len(row["input_ids"]) - row["target_start"]
            or r.get("semantic_noop") != (row["family"] == "no_op")
        ):
            raise ValueError(f"draw differs:{pos}")
        draws.append(index[rid])
    if (
        set(draws) != set(range(len(rows)))
        or schedule.get("checks", {}).get("no_target_truncation") is not True
    ):
        raise ValueError("coverage/truncation differs")
    return rows, draws, schedule, tot


def target_only_collator(batch: list[dict], max_sequence_tokens: int) -> dict[str, Tensor]:
    import torch

    if all(row["family"] != "cpt_replay" for row in batch):
        return edits.target_only_collator(batch, max_sequence_tokens)
    checked = []
    for x in batch:
        if len(x["input_ids"]) > max_sequence_tokens or x["input_ids"][x["target_start"] :] != x[
            "target_body_tokens"
        ] + x["target_terminal_tokens"] + target_end(x):
            raise ValueError("target row changed")
        checked.append(x)
    width = max(len(x["input_ids"]) for x in checked)
    ids = torch.full((len(checked), width), 1, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, x in enumerate(checked):
        n = len(x["input_ids"])
        v = torch.tensor(x["input_ids"], dtype=torch.long)
        ids[i, :n] = v
        mask[i, :n] = 1
        labels[i, x["target_start"] : n] = v[x["target_start"] :]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
