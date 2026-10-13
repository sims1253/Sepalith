#!/usr/bin/env python3
"""Assemble a stage-transition admission from terminal, root-admitted metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(record, label):
    require(set(record) == {"path", "sha256"}, f"{label} pin fields differ")
    path = Path(record["path"])
    require(path.is_file() and sha256(path) == record["sha256"], f"{label} differs")
    return path


def load_pinned(record, label):
    return json.loads(pinned(record, label).read_text())


def exact_subset(actual, expected, label):
    for key, value in expected.items():
        require(actual.get(key) == value, f"{label} differs:{key}")


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "admission output must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assemble(config_path, data_admission_path, output):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    require(config.get("schema") == "sepalith.sft11.cpt-expanded-stage-launch-inputs.v1", "launch input schema differs")
    require(config.get("launch_authorized") is False, "preparation config unexpectedly authorizes launch")

    template_path = pinned(config["stage_transition_template"], "stage-transition template")
    template = json.loads(template_path.read_text())
    require(template.get("schema") == "sepalith.sft11.full-weight-cpt-stage-transition-template.v1", "stage template schema differs")
    require(template.get("launch_authorized") is False, "stage template unexpectedly authorizes launch")

    result = load_pinned(config["lossless_result"], "lossless result")
    require(result.get("schema") == "sepalith.cpt.lossless-rechunk-result.v1" and result.get("status") == "complete", "lossless result is not terminal")
    rows_record = result["artifacts"]["cpt_train_ctx16384.jsonl"]
    result_counts = result["outputs"]["16384"]
    exact_subset(rows_record, {k: config["expected"]["rows"][k] for k in ("bytes", "sha256")}, "lossless row identity")
    exact_subset(result_counts, config["expected"]["lossless_counts"], "lossless counts")
    exact_subset(result["totals"], {"documents": config["expected"]["documents"], "payload_tokens": config["expected"]["payload_tokens"]}, "lossless totals")
    require(result.get("guarantees") == config["required_lossless_guarantees"], "lossless guarantees differ")

    schedule = load_pinned(config["draw_schedule"], "draw schedule")
    core = schedule.get("schedule", schedule)
    exact_subset(core, config["expected"]["schedule"], "draw schedule")
    exact_subset(core.get("coverage", {}), config["expected"]["coverage"], "draw coverage")
    require(core.get("replay_row_ids") and len(core["replay_row_ids"]) == config["expected"]["named_replays"], "named replay list differs")
    require(len(set(core["replay_row_ids"])) == len(core["replay_row_ids"]), "named replay IDs are not distinct")
    require(core.get("token_rows_sha256") == rows_record["sha256"], "schedule row identity differs")
    require(core.get("stage_transition") == config["expected"]["schedule_transition"], "schedule transition differs")

    cache_root = Path(config["streaming_cache_path"])
    cache_manifest_path = cache_root / "manifest.json"
    require(cache_manifest_path.is_file(), "terminal streaming cache manifest is absent")
    cache_manifest_sha = sha256(cache_manifest_path)
    cache = json.loads(cache_manifest_path.read_text())
    require(cache.get("schema") == "sepalith.sft11.cpt-streaming-cache.v1" and cache.get("status") == "complete", "streaming cache is not terminal")
    require(cache.get("max_sequence_tokens") == 16384, "streaming cache context differs")
    expected_cache_rows = {k: config["expected"]["rows"][k] for k in ("path", "sha256")}
    require(cache.get("source", {}).get("rows") == expected_cache_rows, "cache row source differs")
    require(cache.get("source", {}).get("draw_schedule") == config["draw_schedule"], "cache schedule source differs")
    exact_subset(cache.get("counts", {}), config["expected"]["cache_counts_without_packages"], "cache counts")
    packages = cache.get("counts", {}).get("packages")
    require(type(packages) is int and packages > 0, "terminal cache package count missing")
    require(config["expected"]["stage_updates"] * 16 == config["expected"]["unique_rows"] + config["expected"]["named_replays"], "coverage arithmetic differs")
    require(cache.get("contract") == config["required_cache_contract"], "streaming cache contract differs")
    required_cache_files = {"input_ids.i32le", "labels.i32le", "draw_ordinals.i64le", "index.sqlite3"}
    require(set(cache.get("files", {})) == required_cache_files, "terminal cache file inventory differs")
    for name, record in cache["files"].items():
        require(type(record.get("bytes")) is int and record["bytes"] > 0 and len(record.get("sha256", "")) == 64, f"cache file record differs:{name}")
        target = cache_root / name
        require(target.is_file() and target.stat().st_size == record["bytes"], f"cache file size differs:{name}")

    data_admission_path = Path(data_admission_path)
    require(data_admission_path.is_file(), "root DATA_ADMISSION is required")
    data_admission = json.loads(data_admission_path.read_text())
    require(data_admission.get("schema") == "sepalith.sft11.cpt-expanded-snapshot-data-admission.v1", "DATA_ADMISSION schema differs")
    require(data_admission.get("status") == "admitted" and data_admission.get("training_admission") is True, "root did not admit the snapshot for training")
    require(data_admission.get("stage_transition_binding_authorized") is True, "root did not authorize stage-transition binding")
    require(data_admission.get("all_source_pool_closed") is False, "DATA_ADMISSION obscures the open source frontier")
    expected_snapshot = {
        "scope": config["scope"],
        "rows_sha256": rows_record["sha256"],
        "cache_manifest_sha256": cache_manifest_sha,
        "draw_schedule_sha256": config["draw_schedule"]["sha256"],
        "unique_rows": config["expected"]["unique_rows"],
        "documents": config["expected"]["documents"],
        "packages": packages,
        "input_tokens": config["expected"]["input_tokens"],
        "payload_tokens": config["expected"]["payload_tokens"],
        "loss_tokens": config["expected"]["loss_tokens"],
        "named_replays": config["expected"]["named_replays"],
        "stage_updates": config["expected"]["stage_updates"],
        "max_sequence_tokens": 16384,
    }
    require(data_admission.get("snapshot") == expected_snapshot, "DATA_ADMISSION snapshot differs")
    require(data_admission.get("pending_frontier") == config["pending_frontier"], "DATA_ADMISSION frontier differs")

    source = config["source_checkpoint66"]
    source_recipe = load_pinned(source["source_recipe"], "source recipe")
    require(source_recipe.get("runtime", {}).get("learning_rate") == 3e-6, "source hidden learning rate differs")
    optimizer = source_recipe["runtime"]["optimizer"]
    require(optimizer.get("hidden_lr") == 3e-6 and optimizer.get("side_lr") == 3e-7, "source optimizer rates differ")
    require(source_recipe["runtime"].get("scheduler") == "constant_with_warmup", "source scheduler differs")
    source_runtime = source_recipe["runtime"]
    if "warmup_steps" in source_runtime:
        source_warmup = source_runtime["warmup_steps"]
    else:
        ratio = source_runtime.get("warmup_ratio")
        require(type(ratio) is float and 0 <= ratio <= 1, "source warmup ratio differs")
        source_warmup = int(source_runtime["max_steps"] * ratio)
    require(source_warmup == 2, "source warmup differs")
    pinned(source["source_decision"], "source decision")
    pinned(source["source_manifest"], "source manifest")
    pinned(config["dtype_audit"], "dtype audit")

    selected = config["selected"]
    require(selected["optimizer"] == optimizer, "selected optimizer is not byte-semantic source optimizer")
    require(selected["learning_rate"] == 3e-6 and selected["warmup_steps"] == 2, "selected LR/warmup differs")
    require(selected["micro_batch"] == 1 and selected["gradient_accumulation"] == 16, "selected effective batch differs")
    require(selected["checkpoint_every"] == 128 and selected["mandatory_stop_stage_step"] == 128, "first checkpoint gate differs")
    require(selected["evaluation_stage_steps"] == [128, 11443] and selected["selected_stage_milestones"] == [128, 11443], "evaluation/preservation milestones differ")
    require(selected["evaluation_stage_steps"][-1] == config["expected"]["stage_updates"], "terminal evaluation differs from destination horizon")
    require(selected["telemetry_every"] == 1, "telemetry cadence differs")

    admission = {
        "schema": "sepalith.sft11.full-weight-cpt-stage-transition-root-admission.v1",
        "status": "admitted",
        "launch_authorized": True,
        "template_sha256": config["stage_transition_template"]["sha256"],
        "cohort": {
            "id": config["cohort_id"],
            "scope": "complete_admitted_train_corpus",
            "rows": {"path": str(Path(config["expected"]["rows"]["path"]).resolve()), "sha256": rows_record["sha256"]},
            "draw_schedule": config["draw_schedule"],
            "manifest": config["lossless_result"],
            "streaming_cache": {"path": str(cache_root.resolve()), "manifest_sha256": cache_manifest_sha},
            "unique_rows": config["expected"]["unique_rows"], "documents": config["expected"]["documents"], "packages": packages,
            "input_tokens": config["expected"]["input_tokens"], "payload_tokens": config["expected"]["payload_tokens"], "loss_tokens": config["expected"]["loss_tokens"],
            "max_sequence_tokens": 16384, "named_replays": config["expected"]["named_replays"], "updates": config["expected"]["stage_updates"],
        },
        "data_admission": {"path": str(data_admission_path.resolve()), "sha256": sha256(data_admission_path)},
        "corpus_manifest": config["lossless_result"],
        "dtype_audit": config["dtype_audit"],
        "saved_precision": source["saved_precision"],
        "selected": selected,
        "transition": {
            "destination_sampler": "fresh_cursor_zero",
            "global_optimizer_step_offset": 66,
            "scheduler_state": "load_source_without_rewarm",
            "source_checkpoint": source["source_checkpoint"],
            "source_decision": source["source_decision"],
            "source_manifest": source["source_manifest"],
            "source_recipe": source["source_recipe"],
        },
    }
    write_new(output, admission)
    return {"status": "root_admitted_not_launched", "output": str(Path(output).resolve()), "sha256": sha256(output), "cache_manifest_sha256": cache_manifest_sha, "packages": packages, "stage_terminal_step": 11443, "global_first_gate_step": 194, "global_terminal_step": 11509}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--data-admission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.inputs, args.data_admission, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
