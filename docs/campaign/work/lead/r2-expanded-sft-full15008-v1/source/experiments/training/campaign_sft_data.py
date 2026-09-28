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
                          max_steps, effective_batch, split_id=None, vocab_size=130560):
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
    if schedule.get("schema") == "sepalith.sft.full-coverage-batchmix-draws.v2":
        # V2 keeps the eligible-set inventory in row_ids and the actual
        # deterministic training order in structured draws.  Validate those
        # records before adapting them to the reviewed sequential loader.
        records = schedule.get("draws")
        policy = schedule.get("policy", {})
        coverage = schedule.get("coverage", {})
        checks = schedule.get("checks", {})
        source = schedule.get("input", {})
        if (not isinstance(records, list)
                or schedule.get("draw_count") != len(records)
                or policy.get("effective_batch") != effective_batch
                or source.get("sha256") != digest(token_file)
                or source.get("rows_seen") != len(rows)
                or schedule.get("row_ids") != sorted(index)
                or schedule.get("excluded") != []
                or coverage.get("distinct_rows_drawn") != len(rows)
                or checks.get("all_eligible_ids_drawn") is not True
                or checks.get("no_target_truncation") is not True):
            raise ValueError("Full-coverage v2 schedule metadata is inconsistent")
        draw_ids = []
        for position, record in enumerate(records):
            row_id = record.get("row_id") if isinstance(record, dict) else None
            row = rows[index[row_id]] if row_id in index else None
            if (row is None or record.get("draw_index") != position
                    or record.get("update") != position // effective_batch + 1
                    or record.get("total_tokens") != len(row["input_ids"])
                    or record.get("supervised_target_tokens")
                        != len(row["input_ids"]) - row["target_start"]
                    or record.get("semantic_noop") != (row["family"] == "no_op")):
                raise ValueError(f"Invalid full-coverage v2 draw at index {position}")
            draw_ids.append(row_id)
        schedule = dict(schedule)
        schedule.update({
            "effective_batch": effective_batch,
            "token_rows_sha256": digest(token_file),
            "split_id": split_id,
            "row_ids": draw_ids,
        })
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


def target_only_collator(batch):
    """Keep context visible; supervise exact target through EOS, without shifting.

    The model's causal shift makes the last prompt position predict the first
    target token. EOS and PAD share ID1, so masks must use positions.
    """
    if not batch:
        raise ValueError("Target-only batches must contain at least one row")
    for row in batch:
        ids, start = row["input_ids"], row["target_start"]
        if (not isinstance(ids, list) or not 4 <= len(ids) <= 4096
                or any(type(token) is not int or not 0 <= token < 130560 for token in ids)
                or ids[0] != 0 or ids[-1] != 1 or ids.count(0) != 1 or ids.count(1) != 1
                or type(start) is not int or not 1 < start < len(ids) - 1):
            raise ValueError("Invalid full target-only row; truncation is forbidden")
        body, terminal = row["target_body_tokens"], row["target_terminal_tokens"]
        if (not isinstance(body, list) or not isinstance(terminal, list) or not terminal
                or any(type(token) is not int for token in body + terminal)
                or ids[start:] != body + terminal + [1]):
            raise ValueError("Complete target body, terminal and EOS must match input_ids")
    output = full_text_collator(batch)
    for index, row in enumerate(batch):
        output["labels"][index, :row["target_start"]] = -100
    return output


def target_only_pilot_policy(recipe):
    """Explicit isolated variant; old full-text recipes retain their behavior."""
    policy, params = recipe["identity"]["policy"], recipe["parameters"]
    selected = policy.get("stage") == "finish_correction_target_only_pilot50_v1"
    if not selected:
        if policy.get("full_text_labels") is not True:
            raise ValueError("Unknown loss objective requires a new admitted stage")
        return False
    if (policy.get("full_text_labels") is not False or policy.get("loss_objective") != "exact_target_and_eos_v1"
            or policy.get("initialization") != "new_lora_on_merged_sft1000_theta0"
            or params["learning_rate"] != 5e-5 or params["max_steps"] != 200
            or params["max_sequence_tokens"] != 4096
            or recipe["identity"]["parent"].get("revision") != "SFT-primary-step1000-theta0"
            or recipe["identity"]["parent"].get("weights_sha256") != "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d"
            or recipe["decision_steps"] not in ([25], [50])
            or recipe["checkpoint"] != {"light_every": 25, "full_every": 25, "evaluation_steps": [25, 50, 200]}):
        raise ValueError("Target-only pilot must retain theta0, exact objective and bounded 25/50-step gates")
    if (recipe["decision_steps"] == [25]) != (recipe.get("resume_from") is None):
        raise ValueError("Start fresh to25; only an accepted pilot full25 may continue to50")
    return True
