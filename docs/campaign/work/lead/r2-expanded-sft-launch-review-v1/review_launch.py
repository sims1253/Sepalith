#!/usr/bin/env python3
"""Independent CPU review of the immutable expanded SFT-B launch contract."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
B = PLAN / "docs/campaign/work/lead/r2-expanded-sft-b"
RECIPE_PATH = B / "recipe.json"
RUNNER_RECIPE_PATH = B / "runner-recipe.json"
SCHEDULE_PATH = B / "expanded-full-noop25-16000-draw-manifest.json"
SNAPSHOT_ID = "56132b2fd3b21cef88043cff5ea2b1f4db7495a481f3b0d0a7485a5ee3584676"
SNAPSHOT_MANIFEST = (Path("/home/m0hawk/.local/state/sepalith/campaign-20260915") /
                     "runner-expanded-sft-v1/snapshots" / SNAPSHOT_ID / "manifest.json")
REVIEWED_SOURCE = PLAN / "docs/campaign/work/lead/r2-expanded-sft-a/reviewed-source"
EXPECTED_RECIPE_SHA = "ba64a66e4f78d9a13b5db95ab3f7dd68a0b09382159c1b533ed8e6f23665eb07"
EXPECTED_RUNNER_RECIPE_SHA = "5e922d8c221fe8d8b3f20555f09d917ead5318739d05dacad8433f2dc65837ac"
EXPECTED_SCHEDULE_SHA = "2ad2074d194b99bdef6826eb74543669fd8a7cf3490aedd692844b6402bed498"
EXPECTED_ROWS_SHA = "fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889"
EXPECTED_PARENT_SHA = "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    recipe = json.loads(RECIPE_PATH.read_text())
    runner = json.loads(RUNNER_RECIPE_PATH.read_text())
    schedule = json.loads(SCHEDULE_PATH.read_text())
    snapshot = json.loads(SNAPSHOT_MANIFEST.read_text())
    require(digest(RECIPE_PATH) == EXPECTED_RECIPE_SHA, "B recipe hash changed")
    require(digest(RUNNER_RECIPE_PATH) == EXPECTED_RUNNER_RECIPE_SHA, "B runner recipe hash changed")
    require(digest(SCHEDULE_PATH) == EXPECTED_SCHEDULE_SHA, "B schedule hash changed")

    params = recipe["parameters"]
    require(recipe["id"] == runner["id"] == "sft11-expanded-postsft500-b-20260914",
            "runner/recipe ID mismatch")
    require(recipe["stage"] == recipe["identity"]["policy"]["stage"] == "expanded_task_sft_v1",
            "stage mismatch")
    require(recipe["launch_authorized"] is True and recipe["resume_from"] is None,
            "launch must be authorized with a fresh optimizer")
    require(recipe["parent_binding"]["optimizer_resume"] == "forbidden; fresh LoRA and optimizer",
            "parent binding permits inherited optimizer")
    require(recipe["identity"]["schedule"] == params, "schedule identity mismatch")
    require(params == {"gradient_accumulation": 8, "learning_rate": 0.0002,
                       "lora_alpha": 64, "lora_rank": 32, "max_sequence_tokens": 4096,
                       "max_steps": 1000, "per_device_batch": 2,
                       "train_max_target_tokens": 1024}, "training parameters changed")
    require(params["per_device_batch"] * params["gradient_accumulation"] == 16,
            "effective batch is not 16")
    require(recipe["development_max_new_tokens"] == 192, "DEV cap changed")
    require(recipe["milestones"] == [250, 500, 1000], "milestones changed")
    require(recipe["target_gate_start_steps"] == [0, 250, 500], "startup gate steps changed")
    require(recipe["target_resume_steps"] == [250, 500], "resume steps changed")
    require(recipe["decision_steps"] == recipe["mandatory_stop_steps"] == [250],
            "first attempt must stop at 250")
    require(recipe["checkpoint"] == {"evaluation_steps": [250, 500, 1000],
                                     "full_every": 50, "light_every": 50},
            "checkpoint lifecycle changed")

    parent = recipe["identity"]["parent"]
    require(parent["kind"] == "sft_merged" and
            parent["weights_sha256"] == EXPECTED_PARENT_SHA and
            parent["previous_checkpoint_step"] == 500 and
            parent["previous_source_cursor"] == 8000 and
            recipe["identity"]["policy"]["initialization"] == "new_lora_on_sft_merged_parent",
            "selected post-SFT500 parent mismatch")
    manifest_record = recipe["sft_merged_parent"]["manifest"]
    prior_record = recipe["sft_merged_parent"]["previous_recipe"]
    checkpoint_record = recipe["sft_merged_parent"]["previous_checkpoint_manifest"]
    for record in (manifest_record, prior_record, checkpoint_record):
        require(digest(Path(record["path"])) == record["sha256"], "parent lineage hash mismatch")
    manifest = json.loads(Path(manifest_record["path"]).read_text())
    prior = json.loads(Path(prior_record["path"]).read_text())
    checkpoint = json.loads(Path(checkpoint_record["path"]).read_text())
    require(manifest["kind"] == "merged_task_sft" and
            manifest["merged_weights_sha256"] == EXPECTED_PARENT_SHA and
            manifest["merged_model_path"] == recipe["model_path"] and
            manifest["recipe_sha256"] == prior_record["sha256"] and
            manifest["checkpoint_manifest_sha256"] == checkpoint_record["sha256"] and
            manifest["sft_identity"] == prior["identity"] and
            manifest["sft_checkpoint_manifest"] == checkpoint and
            checkpoint["identity"] == prior["identity"] and checkpoint["full"] is True and
            checkpoint["step"] == 500 and manifest["source_cursor"] == 8000,
            "parent manifest/checkpoint/recipe cross-check failed")

    require(runner["snapshot"] == snapshot["id"] == recipe["identity"]["source"] == SNAPSHOT_ID,
            "runner source snapshot identity mismatch")
    input_records = {(item["path"], item["sha256"]) for item in runner["inputs"]}
    require((str(RECIPE_PATH), EXPECTED_RECIPE_SHA) in input_records,
            "runner does not pin the B recipe")
    snapshot_hashes = {item["path"]: item["sha256"] for item in snapshot["files"]}
    require(len(snapshot_hashes) == 12 and snapshot["git_head"] == "36cccc91bcd7d5bd19772df76a93f335ef62d155",
            "source snapshot closure changed")
    for relative, expected in snapshot_hashes.items():
        path = REVIEWED_SOURCE / relative
        require(digest(path) == expected, f"reviewed source differs: {relative}")
        original = PLAN / "docs/campaign/work/lead/r2-expanded-sft-adapter-v1/source" / relative
        require((str(original), expected) in input_records, f"runner missing source input: {relative}")

    rows_path = Path(recipe["token_rows"]["path"])
    require(digest(rows_path) == recipe["token_rows"]["sha256"] == EXPECTED_ROWS_SHA,
            "TRAIN row hash mismatch")
    rows = {}
    max_sequence = max_target = targets_over_192 = 0
    family_rows = Counter()
    with rows_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            ident = row["id"]
            require(ident not in rows, f"duplicate TRAIN row {ident}")
            ids, start = row["input_ids"], row["target_start"]
            body, terminal = row["target_body_tokens"], row["target_terminal_tokens"]
            require(row["split"] == "train" and row["renderer_id"] == recipe["renderer_id"],
                    f"split/renderer mismatch line {line_number}")
            require(ids[0] == 0 and ids[-1] == 1 and ids.count(0) == ids.count(1) == 1,
                    f"BOS/EOS mismatch {ident}")
            require(1 < start < len(ids) - 1 and terminal and ids[start:] == body + terminal + [1],
                    f"truncated/incomplete target {ident}")
            require(row["target_body_token_count"] == len(body) and
                    row["target_terminal_token_count"] == len(terminal),
                    f"target count mismatch {ident}")
            target_length = len(ids) - start
            require(len(ids) <= 4096 and target_length <= 1024, f"length cap exceeded {ident}")
            require(all(type(token) is int and 0 <= token < 130560 for token in ids),
                    f"invalid token {ident}")
            rows[ident] = row
            max_sequence = max(max_sequence, len(ids))
            max_target = max(max_target, target_length)
            targets_over_192 += target_length > 192
            family_rows[row["family"]] += 1
    require(len(rows) == 11505 and targets_over_192 == 1963 and
            max_sequence == 3064 and max_target == 933, "TRAIN geometry mismatch")

    draws, row_ids = schedule["draws"], schedule["row_ids"]
    require(schedule["draw_count"] == len(draws) == len(row_ids) == 16000 and
            schedule["token_rows_sha256"] == EXPECTED_ROWS_SHA and
            schedule["effective_batch"] == 16 and schedule["max_steps"] == 1000,
            "schedule horizon/data mismatch")
    require(schedule["policy"]["batch_noop_slots"] == 4 and
            schedule["policy"]["batch_edit_slots"] == 12 and
            schedule["policy"]["truncation"] == "forbidden",
            "schedule mixture/truncation policy mismatch")
    exposures = Counter(row_ids)
    require(set(exposures) == set(rows) and min(exposures.values()) == 1 and
            max(exposures.values()) == 4, "schedule does not cover every TRAIN row")
    for index, (draw, ident) in enumerate(zip(draws, row_ids)):
        row = rows[ident]
        require(draw["draw_index"] == index and draw["row_id"] == ident and
                draw["family"] == row["family"] and
                draw["semantic_noop"] == (row["family"] == "no_op") and
                draw["target_tokens"] == len(row["input_ids"]) - row["target_start"] and
                draw["total_tokens"] == len(row["input_ids"]),
                f"draw metadata mismatch {index}")
    for offset in range(0, 16000, 16):
        noops = sum(rows[ident]["family"] == "no_op" for ident in row_ids[offset:offset + 16])
        require(noops == 4, f"batch {offset // 16 + 1} does not contain 4/16 no-ops")
    first250 = row_ids[:4000]
    require(len(set(first250)) == 4000 and
            sum(rows[ident]["family"] == "no_op" for ident in first250) == 1000,
            "first 250 updates are not distinct 25%-no-op draws")

    sft_source = (REVIEWED_SOURCE / "experiments/training/campaign_sft.py").read_text()
    gate_source = (REVIEWED_SOURCE / "experiments/training/target_only_gate.py").read_text()
    require("lora_dropout=0" in sft_source and
            "SequentialSampler" in sft_source and
            'save_only_model=False, ignore_data_skip=False' in sft_source and
            '"consumed_draws": trainer.state.global_step * 16' in sft_source and
            'not any(isinstance(m, torch.nn.Dropout) and m.p' in gate_source,
            "dropout/sampler/checkpoint/startup guard source mismatch")
    require(recipe["identity"]["policy"]["gradient_checkpointing"] == "gpu_standard_v1" and
            "expanded_task_sft_v1" in sft_source and '"gradient_checkpointing": True' in sft_source,
            "standard GPU checkpointing path mismatch")

    print(json.dumps({
        "status": "pass",
        "recipe_sha256": EXPECTED_RECIPE_SHA,
        "runner_recipe_sha256": EXPECTED_RUNNER_RECIPE_SHA,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA,
        "snapshot": SNAPSHOT_ID,
        "rows": len(rows),
        "draws": len(row_ids),
        "batches_checked": len(row_ids) // 16,
        "distinct_first250": len(set(first250)),
        "first250_noop_draws": 1000,
        "all_rows_first_covered_step": schedule["coverage"]["first_all_rows_step"],
        "targets_over_192": targets_over_192,
        "max_target_including_eos": max_target,
        "max_sequence": max_sequence,
        "parent_step": checkpoint["step"],
        "fresh_optimizer": recipe["resume_from"] is None,
        "CUDA_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
