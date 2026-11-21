"""Freeze C20 validation, then build one deterministic mixed SFT epoch."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import cast

from sepalith.training.grpo_exec import pool
from sepalith.training.sft import recipe_v2

SEED = 3407
UPDATES = 300
CONTEXT = 2048
CPT_TARGET_RANGE = (64, 160)
DOSE = {"target": 8, "short": 2, "rename": 2, "no_op": 3, "cpt": 1}
ROOT = Path("docs/training/plan-20260927")
DATA = pool.OUT_DATA.parent / "C20-sft-finish-v1" / "frozen"
C19_RECIPE = ROOT / "c19/bound/grpo-exec-pilot-1-c18b-300.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    return cast(list[dict], pool.read_jsonl(path))


def checked(record: dict) -> Path:
    path = Path(record["path"])
    pool.no_sealed(path.resolve())
    pool.require(path.is_file() and pool.sha256(path) == record["sha256"], f"input differs: {path}")
    return path


def identities(tasks: list[dict]) -> tuple[set[str], set[str]]:
    return {t["function_id"] for t in tasks}, {t["package"].casefold() for t in tasks}


def belongs(task: dict, split: tuple[set[str], set[str]]) -> bool:
    return task["function_id"] in split[0] and task["package"].casefold() in split[1]


def target(task: dict) -> bool:
    return task["family"] == "finish_block" and 3 <= task["curriculum"]["target_lines"] <= 16


def partition(manifest: dict) -> tuple[list[dict], list[dict], set[str]]:
    original_train = read_rows(checked(manifest["train"]))
    original_val = read_rows(checked(manifest["validation"]))
    train, val = identities(original_train), identities(original_val)
    held_functions, held_packages = pool.holdout_identity()
    excluded = {p.casefold() for p in held_packages}
    pool.require(not train[0] & val[0] and not train[1] & val[1], "C18 splits overlap")
    pool.require(not (train[0] | val[0]) & held_functions and not (train[1] | val[1]) & excluded,
                 "C18 splits overlap hold-out identities")

    built = manifest["built"]
    receipt = read(checked({"path": built["receipt"], "sha256": built["receipt_sha256"]}))
    pool.require(receipt["tasks_sha256"] == built["sha256"] and
                 receipt["source"]["kind"] == "c18-train" and
                 receipt["gate_version"]["fingerprint"] == pool.GATE_FINGERPRINT,
                 "uncapped source receipt differs")
    tasks = read_rows(checked(built))
    by_id = {t["id"]: t for t in tasks}
    pool.require(len(by_id) == len(tasks) and all(by_id.get(t["id"]) == t for t in original_train + original_val),
                 "capped tasks differ from uncapped source")
    pool.require(all(t["function_id"] not in held_functions and t["package"].casefold() not in excluded
                     for t in tasks), "hold-out identity reached uncapped source")
    validation = sorted((t for t in tasks if belongs(t, val)), key=lambda t: t["id"])
    training = sorted((t for t in tasks if belongs(t, train)), key=lambda t: t["id"])
    pool.require(len(validation) > len(original_val) and sum(map(target, validation)) > sum(map(target, original_val)),
                 "validation did not expand")
    return validation, training, excluded | val[1]


def token_row(rid: str, family: str, package: str, prompt: list[int],
              body: list[int], terminal: list[int], eos: list[int]) -> dict:
    return {
        "id": rid, "family": family, "package_id": package, "split": "train",
        "renderer_id": "zeta2-prm03-v1" if family != "cpt_replay" else "c18b-cpt-text",
        "tokenizer_json_sha256": recipe_v2.TOKENIZER_SHA256,
        "input_ids": prompt + body + terminal + eos, "target_start": len(prompt),
        "prompt_token_count": len(prompt) - 1, "target_body_tokens": body,
        "target_terminal_tokens": terminal, "target_body_token_count": len(body),
        "target_terminal_token_count": len(terminal), "target_token_count": len(body) + len(terminal),
    }


def gold(task: dict, terminal: list[int]) -> dict:
    completion = task["gold_completion_ids"]
    boundary = len(completion) - len(terminal) - 1
    pool.require(boundary > 0 and completion[boundary:] == terminal + [1], "exec gold terminal differs")
    return token_row("exec:" + task["id"], task["family"], task["package"],
                     task["prompt_ids"], completion[:boundary], terminal, [1])


def draw_batches(groups: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    decks: dict[str, list[dict]] = {}
    for name, dose in DOSE.items():
        rows = sorted(groups[name], key=lambda r: r["id"])
        n = UPDATES * dose
        pool.require(len(rows) * (3 if name == "no_op" else 1) >= n, f"insufficient {name}: {len(rows)} for {n}")
        draws: list[dict] = []
        while len(draws) < n:
            draws.extend(rng.sample(rows, min(n - len(draws), len(rows))))
        decks[name] = draws
    unique = {row["id"]: row for deck in decks.values() for row in deck}
    ordered = []
    for step in range(UPDATES):
        batch = [row for name, dose in DOSE.items() for row in decks[name][step * dose:(step + 1) * dose]]
        rng.shuffle(batch)
        ordered.extend(batch)
    draws = [{"draw_index": i, "update": i // 16 + 1, "row_id": row["id"],
              "semantic_noop": row["family"] == "no_op", "total_tokens": len(row["input_ids"]),
              "supervised_target_tokens": len(row["input_ids"]) - row["target_start"]}
             for i, row in enumerate(ordered)]
    return [unique[k] for k in sorted(unique)], draws


def build(repo: Path, out: Path = DATA) -> dict:
    from tokenizers import Tokenizer

    from sepalith.campaign_protocol import TERMINAL
    from sepalith.training.sft.campaign_edit_data import inspect, validate_row

    source = repo / ROOT / "c18/C18-exec-pool.json"
    manifest = read(source)
    validation, training, excluded = partition(manifest)
    val_path = out / "validation.jsonl"
    val_sha = pool.write_jsonl_new(val_path, validation)
    val_manifest_path = out / "validation-manifest.json"
    val_manifest = {
        "source_manifest_sha256": pool.sha256(source),
        "source_records": {key: manifest[key]["sha256"] for key in ("built", "train", "validation")},
        "tasks": {"path": str(val_path), "sha256": val_sha},
        "stats": pool.stats(validation),
        "finish_block_3_to_16_tasks": sum(map(target, validation)),
        "finish_block_3_to_16_functions": len({t["function_id"] for t in validation if target(t)}),
    }
    pool.write_json_new(val_manifest_path, val_manifest)

    c19 = read(repo / C19_RECIPE)
    parent_tokenizer = Path(c19["parent"]["path"]) / "tokenizer.json"
    pool.require(pool.sha256(parent_tokenizer) == recipe_v2.TOKENIZER_SHA256, "tokenizer differs")
    terminal = Tokenizer.from_file(str(parent_tokenizer)).encode(TERMINAL, add_special_tokens=False).ids
    groups: dict[str, list[dict]] = {name: [] for name in DOSE}
    for task in training:
        if len(task["prompt_ids"]) + len(task["gold_completion_ids"]) > CONTEXT:
            continue
        if target(task):
            groups["target"].append(gold(task, terminal))
        elif task["family"] == "finish_block" and task["curriculum"]["target_lines"] < 3:
            groups["short"].append(gold(task, terminal))
        elif task["family"] == "rename_propagation":
            groups["rename"].append(gold(task, terminal))

    from sepalith.training.rl.reward_v21 import DATA_ROOT, TRAIN_INPUTS

    no_op_name, no_op_sha = TRAIN_INPUTS["rows"]
    no_op_source = {"path": str(DATA_ROOT / no_op_name), "sha256": no_op_sha}
    with checked(no_op_source).open() as stream:
        for line in stream:
            row = json.loads(line)
            pool.require(row["split"] == "train", "no-op source contains non-TRAIN row")
            if (row["family"] == "no_op" and row["package_id"].casefold() not in excluded
                    and len(row["input_ids"]) <= CONTEXT):
                groups["no_op"].append(row)

    cpt_source = c19["pool"]["cpt_replay"]
    for row in read_rows(checked(cpt_source)):
        completion = row["target_ids"]
        if row["package"].casefold() not in excluded and CPT_TARGET_RANGE[0] <= len(completion) <= CPT_TARGET_RANGE[1]:
            eos = [1] if completion[-1] == 1 else []
            body = completion[:-1] if eos else completion
            groups["cpt"].append(token_row("cpt:" + row["id"], "cpt_replay", row["package"],
                                            row["prompt_ids"], body, [], eos))

    rows, draws = draw_batches(groups)
    for row in rows:
        validate_row(row, CONTEXT, allow_cpt=True)
    rows_path = out / "rows.jsonl"
    rows_record = {"path": str(rows_path), "sha256": pool.write_jsonl_new(rows_path, rows)}
    schedule_path = out / "schedule.json"
    schedule = {"schema": "sepalith.sft.full-coverage-batchmix-draws.v2", "input": rows_record,
                "excluded": [], "row_ids": [r["id"] for r in rows], "draws": draws,
                "draw_count": len(draws), "max_steps": UPDATES,
                "coverage": {"distinct_rows_drawn": len(rows)}, "checks": {"no_target_truncation": True}}
    pool.write_json_new(schedule_path, schedule)
    schedule_record = recipe_v2.file_record(schedule_path)
    profile_path = out / "length-profile.json"
    pool.write_json_new(profile_path, recipe_v2.build_length_profile(rows_path, str(rows_record["sha256"])))
    _, _, _, totals = inspect(rows_record, schedule_record, CONTEXT, allow_cpt=True)
    exposure = Counter(r["row_id"] for r in draws)
    capped_ids = {r["id"] for r in read_rows(checked(manifest["train"]))}
    mix = {"seed": SEED, "updates": UPDATES, "dose_per_update": DOSE,
           "eligible": {k: len(v) for k, v in groups.items()}, "distinct_rows": len(rows), "cpt_target_length_in_tokens": list(CPT_TARGET_RANGE),
           "cpt_loss_tokens": sum(len(r["input_ids"]) - r["target_start"] for r in rows if r["family"] == "cpt_replay"),
           "exposures": dict(Counter(exposure.values())),
           "uncapped_exec_rows": sum(r["id"].startswith("exec:") and r["id"][5:] not in capped_ids for r in rows),
           "sources": {"no_op": no_op_source, "cpt": cpt_source,
                       "validation_manifest_sha256": pool.sha256(val_manifest_path)},
           "rows": rows_record, "schedule": schedule_record, "length_profile": recipe_v2.file_record(profile_path),
           **totals}
    pool.write_json_new(out / "mix-manifest.json", mix)
    return {"validation": val_manifest, "mix": mix}
