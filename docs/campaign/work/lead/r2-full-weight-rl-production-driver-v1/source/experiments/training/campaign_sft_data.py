"""Read admitted, already-tokenized rows and a frozen SFT draw schedule.

This layer never renders, truncates, pads with supervision, or chooses a split.
DAT-04/06 and the shared protocol own those decisions before a training launch.
"""
import json
from pathlib import Path

from campaign_checkpoint import digest


def verified_file(record):
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Expected an absolute input file: {path}")
    if digest(path) != record["sha256"]:
        raise ValueError(f"Input hash differs: {path}")
    return path


def inspect_training_data(token_file, draw_file, *, renderer_id, max_sequence_tokens,
                          max_steps, effective_batch, vocab_size=130560):
    token_file, draw_file = verified_file(token_file), verified_file(draw_file)
    rows, index = [], {}
    with token_file.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            ident = row["id"]
            if not isinstance(ident, str) or not ident or ident in index:
                raise ValueError(f"Missing or duplicate row identity at line {line_number}")
            if row["split"] != "train" or row["renderer_id"] != renderer_id:
                raise ValueError(f"Wrong split or renderer for row {ident}")
            if not row.get("family") or not row.get("package_id"):
                raise ValueError(f"Missing provenance for row {ident}")
            ids = row["input_ids"]
            if not isinstance(ids, list) or not 3 <= len(ids) <= max_sequence_tokens:
                raise ValueError(f"Invalid full-sequence length for row {ident}")
            if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
                raise ValueError(f"Invalid token ID for row {ident}")
            if ids[0] != 0 or ids[-1] != 1 or ids.count(0) != 1 or ids.count(1) != 1:
                raise ValueError(f"BOS/EOS boundary mismatch for row {ident}")
            start, body_tokens = row["target_start"], row["target_body_tokens"]
            if type(start) is not int or not 1 < start < len(ids) - 1:
                raise ValueError(f"Invalid prompt/target boundary for row {ident}")
            if (not isinstance(body_tokens, list) or len(body_tokens) >= len(ids) - start
                    or any(type(token) is not int for token in body_tokens)
                    or ids[start:start + len(body_tokens)] != body_tokens):
                raise ValueError(f"Invalid target-body token prefix for row {ident}")
            index[ident] = len(rows)
            rows.append(row)
    schedule = json.loads(draw_file.read_text())
    if schedule["max_steps"] != max_steps or schedule["effective_batch"] != effective_batch:
        raise ValueError("Draw schedule differs from the predeclared optimizer schedule")
    if schedule["token_rows_sha256"] != digest(token_file):
        raise ValueError("Draw schedule refers to different token rows")
    draw_ids = schedule["row_ids"]
    if len(draw_ids) != max_steps * effective_batch or not rows:
        raise ValueError("Draw count must cover exactly the declared optimizer schedule")
    if not schedule.get("split_id"):
        raise ValueError("Draw schedule requires the global split identity")
    missing = set(draw_ids) - index.keys()
    if missing:
        raise ValueError(f"Draws refer to {len(missing)} absent training rows")
    draw_indices = [index[ident] for ident in draw_ids]
    exposure = []
    for offset in range(0, len(draw_indices), effective_batch):
        batch = [rows[i] for i in draw_indices[offset:offset + effective_batch]]
        exposure.append({
            "step": offset // effective_batch + 1, "draws": len(batch),
            "prompt_loss_tokens": sum(r["target_start"] - 1 for r in batch),
            "target_loss_tokens": sum(len(r["input_ids"]) - r["target_start"] for r in batch),
            "target_body_tokens": sum(len(r["target_body_tokens"]) for r in batch),
            "total_tokens": sum(len(r["input_ids"]) for r in batch),
            "max_sequence_tokens": max(len(r["input_ids"]) for r in batch),
            "families": {family: sum(r["family"] == family for r in batch)
                         for family in sorted({r["family"] for r in batch})},
        })
    return rows, draw_indices, schedule, exposure


def full_text_collator(batch):
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    ids = torch.full((len(batch), width), 1, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(batch):
        length = len(row["input_ids"])
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[i, :length] = values
        attention[i, :length] = 1
        labels[i, :length] = values
    # EOS and padding share ID 1. Mask by position, never by token value.
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}
