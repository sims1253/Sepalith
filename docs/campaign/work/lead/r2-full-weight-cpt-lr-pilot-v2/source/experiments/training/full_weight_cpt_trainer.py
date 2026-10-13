#!/usr/bin/env python3
"""Full-weight CPT Trainer over an immutable token-row cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
PROTOCOL = SOURCE / "packages" / "sepalith" / "src"
if str(PROTOCOL) not in sys.path: sys.path.insert(0, str(PROTOCOL))

EXPECTED_PARAMETERS = 2_516_756_480
EXPECTED_PARAMETER_TENSORS = 381


def require(value, message):
    if not value: raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())


def verify_source(recipe):
    record = recipe["source"]
    path = Path(record["manifest_path"])
    require(path.is_file() and sha256(path) == record["manifest_sha256"], "source manifest differs")
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == "sepalith.full-weight-cpt-lr-pilot-source.v3", "source schema differs")
    for item in manifest["files"]:
        target = path.parent / item["path"]
        require(target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"], f"source differs:{item['path']}")


def load_bound(path):
    recipe = json.loads(Path(path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-cpt-lr-pilot-bound.v3" and recipe.get("status") == "root_admitted_not_launched", "bound recipe is not root-admitted")
    admission = recipe.get("root_admission", {})
    require(Path(admission.get("path", "")).is_file() and sha256(admission["path"]) == admission.get("sha256"), "root admission differs")
    verify_source(recipe)
    registry = recipe["cohort_registry"]
    require(Path(registry["path"]).is_file() and sha256(registry["path"]) == registry["sha256"] and registry["all_corpus_complete"] is False, "cohort queue binding differs")
    runtime = recipe["runtime"]
    require(runtime["micro_batch"] in (1, 2) and runtime["micro_batch"] * runtime["gradient_accumulation"] == runtime["effective_batch"] == 16, "effective batch differs")
    require(runtime["max_steps"] == recipe["cohort"]["updates"], "cohort horizon differs")
    require(runtime["checkpoint_every"] > 0 and runtime["max_steps"] % runtime["checkpoint_every"] == 0, "checkpoint cadence differs")
    require(runtime["max_steps"] == 24 and runtime["micro_batch"] == 1 and runtime["gradient_accumulation"] == 16, "LR pilot horizon/batching differs")
    require(runtime["scheduler"] == "constant_with_warmup" and int(runtime["max_steps"] * runtime["warmup_ratio"]) == 2, "LR pilot warmup differs")
    require(runtime["checkpoint_every"] == 24 and runtime["evaluation_steps"] == [24] and runtime["selected_milestones"] == [24], "LR pilot must save/evaluate terminal only")
    pair = (runtime["optimizer"].get("hidden_lr"), runtime["optimizer"].get("side_lr"))
    require(runtime["optimizer"].get("arm") == "aurora_mix" and pair in ((3e-5, 3e-6), (1e-5, 1e-6)) and runtime["learning_rate"] == pair[0], "LR pilot arm differs")
    validate_context_and_storage(recipe)
    for record in (recipe["cohort"]["rows"], recipe["cohort"]["draw_schedule"], recipe["cohort"]["manifest"], recipe["validation"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    parent = recipe["parent"]
    require(parent.get("kind") == "merged_cpt_parent", "full-weight CPT parent kind differs")
    root = Path(parent["path"]); require(root.is_dir(), "parent path missing")
    for name, expected in parent["files"].items():
        target = root / name; require(target.is_file() and sha256(target) == expected, f"parent differs:{name}")
    dispatch = recipe["optimizer_dispatch"]
    require(Path(dispatch["path"]).is_file() and sha256(dispatch["path"]) == dispatch["sha256"], "optimizer dispatch differs")
    return recipe


def validate_context_and_storage(recipe):
    train_context = recipe["cohort"].get("max_sequence_tokens")
    require(type(train_context) is int and train_context in (2048, 4096, 8192, 16384, 32768), "TRAIN context is not a prepared power-of-two bound")
    require(recipe["validation"].get("max_sequence_tokens") == 2048, "heldout comparator must remain fixed at 2048 tokens")
    storage = recipe.get("checkpoint_storage", {})
    require(storage.get("mode") == "e_same_filesystem_atomic" and storage.get("c_hot_stage") == "not_admitted", "checkpoint storage policy differs")
    require(storage.get("required_same_filesystem") is True, "checkpoint publication must require one filesystem")
    trainer_path, archive_path = Path(recipe["outputs"]["trainer"]), Path(recipe["outputs"]["archive"])
    require(str(trainer_path).startswith("/mnt/e/") and str(archive_path).startswith("/mnt/e/"), "v2 permits only E trainer/archive paths")
    require(Path(storage.get("trainer_root", "")) == trainer_path and Path(storage.get("archive_root", "")) == archive_path, "checkpoint storage paths differ from outputs")


def training_arguments_kwargs(recipe, output):
    runtime = recipe["runtime"]
    warmup_steps = int(runtime["max_steps"] * runtime["warmup_ratio"])
    return {"output_dir": str(output), "per_device_train_batch_size": runtime["micro_batch"], "gradient_accumulation_steps": runtime["gradient_accumulation"], "max_steps": runtime["max_steps"], "learning_rate": runtime["learning_rate"], "lr_scheduler_type": runtime["scheduler"], "warmup_steps": warmup_steps, "max_grad_norm": 0.0, "bf16": True, "use_cache": False, "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}, "save_strategy": "steps", "save_steps": runtime["checkpoint_every"], "save_only_model": False, "save_total_limit": 2, "logging_strategy": "steps", "logging_steps": 1, "logging_first_step": True, "eval_strategy": "no", "report_to": "none", "disable_tqdm": True, "remove_unused_columns": False, "dataloader_num_workers": 0, "dataloader_pin_memory": False, "train_sampling_strategy": "sequential", "ignore_data_skip": False, "seed": recipe["seed"], "data_seed": recipe["seed"]}


class FrozenTokenRowDataset:
    """Offset-indexed schedule view; every draw is fixed by row ID."""
    def __init__(self, recipe):
        from campaign_cpt_data import validate_materialized_row
        rows_path = Path(recipe["cohort"]["rows"]["path"])
        offsets, row_order, totals = {}, [], {"input_tokens": 0, "loss_tokens": 0}
        packages, documents = set(), set()
        with rows_path.open("rb") as stream:
            while True:
                offset = stream.tell(); line = stream.readline()
                if not line: break
                raw = json.loads(line); normalized = validate_materialized_row(raw, max_sequence_tokens=recipe["cohort"]["max_sequence_tokens"])
                ident = normalized["id"]; require(ident not in offsets, "duplicate cohort row ID")
                offsets[ident] = offset; row_order.append(ident)
                totals["input_tokens"] += len(raw["input_ids"]); totals["loss_tokens"] += sum(x != -100 for x in raw["labels"][1:])
                packages.add(raw["package"]); documents.add(raw["document_id"])
        cohort = recipe["cohort"]
        require(len(row_order) == cohort["unique_rows"] and totals == {"input_tokens": cohort["input_tokens"], "loss_tokens": cohort["loss_tokens"]}, "cohort denominators differ")
        schedule = json.loads(Path(cohort["draw_schedule"]["path"]).read_text())
        require(schedule.get("token_rows_sha256") == cohort["rows"]["sha256"], "schedule row binding differs")
        draws = schedule.get("row_ids")
        require(isinstance(draws, list) and len(draws) == cohort["updates"] * 16, "schedule draw denominator differs")
        require(set(draws) == set(row_order) and draws[:len(row_order)] == row_order, "cohort must expose every unique row before named replay")
        require(len(draws) - len(row_order) == cohort["named_replays"], "named replay denominator differs")
        self.path, self.offsets, self.draws = rows_path, offsets, draws
        self.packages, self.documents = packages, documents

    def __len__(self): return len(self.draws)

    def __getitem__(self, position):
        ident = self.draws[position]
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[ident]); row = json.loads(stream.readline())
        return {"input_ids": row["input_ids"], "labels": row["labels"], "attention_mask": row["attention_mask"], "_draw_position": position}


def _validated_cohort(recipe):
    dataset = FrozenTokenRowDataset(recipe)
    from campaign_cpt_data import read_jsonl
    validation_context = recipe["validation"]["max_sequence_tokens"]
    valid = read_jsonl(recipe["validation"]["path"], max_sequence_tokens=validation_context, materialized=True)
    require(len(valid) == recipe["validation"]["rows"] == 499, "validation denominator differs")
    valid_packages = {row["package"] for row in valid}; valid_documents = {row["document_id"] for row in valid}
    require(not dataset.packages.intersection(valid_packages) and not dataset.documents.intersection(valid_documents), "CPT holdout leakage")
    return dataset, {"status": "pass", "cohort_rows": len(dataset.offsets), "draws": len(dataset), "validation_rows": len(valid), "train_context_tokens": recipe["cohort"]["max_sequence_tokens"], "validation_context_tokens": validation_context, "cohort_scope": recipe["cohort"]["scope"], "cuda_started": False}


def _cohort_preflight(recipe):
    return _validated_cohort(recipe)[1]


def preflight(recipe_path):
    return _cohort_preflight(load_bound(recipe_path))


def preflight_template(template_path):
    recipe = json.loads(Path(template_path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-cpt-lr-pilot-template.v3" and recipe.get("launch_authorized") is False, "preparation template differs")
    validate_context_and_storage(recipe)
    verify_source(recipe)
    registry = recipe["cohort_registry"]
    require(Path(registry["path"]).is_file() and sha256(registry["path"]) == registry["sha256"] and registry["all_corpus_complete"] is False, "cohort queue binding differs")
    for record in (recipe["cohort"]["rows"], recipe["cohort"]["draw_schedule"], recipe["cohort"]["manifest"], recipe["validation"], recipe["optimizer_dispatch"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    for parent in recipe["parent_candidates"].values():
        root = Path(parent["path"])
        for name, expected in parent["files"].items():
            target = root / name; require(target.is_file() and sha256(target) == expected, f"parent candidate differs:{name}")
    return _cohort_preflight(recipe)


def identity(recipe):
    return {
        "parent": {"candidate_id": recipe["parent"]["candidate_id"], "weights_sha256": recipe["parent"]["files"]["model.safetensors"]},
        "tokenizer": {"sha256": recipe["parent"]["files"]["tokenizer.json"], "bos": 0, "eos_pad": 1},
        "renderer": {"kind": "pretokenized_raw_r_cpt_v1", "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"]},
        "data": {"cohort_id": recipe["cohort"]["id"], "rows_sha256": recipe["cohort"]["rows"]["sha256"]},
        "source": {"manifest_sha256": recipe["source"]["manifest_sha256"]},
        "policy": {"stage": "full_weight_cpt_lr_pilot_v1", "optimizer": recipe["runtime"]["optimizer"]},
        "schedule": {k: recipe["runtime"][k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_ratio", "checkpoint_every")},
    }


def retain_archives(archive_root, milestones, latest=2):
    full = Path(archive_root) / "full"; checkpoints = []
    for path in full.glob("checkpoint-*"):
        try: step = int(path.name.split("-")[-1])
        except ValueError: continue
        if path.is_dir(): checkpoints.append((step, path))
    preserve = set(milestones) | {step for step, _ in sorted(checkpoints)[-latest:]}
    retired = []
    for step, path in checkpoints:
        if step in preserve: continue
        temporary = path.with_name(f".retiring-{path.name}-{os.getpid()}")
        os.rename(path, temporary); shutil.rmtree(temporary); retired.append(step)
    return {"preserved": sorted(preserve & {s for s, _ in checkpoints}), "retired": sorted(retired)}


def publish_sealed_checkpoint(source, destination, sealed_manifest):
    """Atomically publish a sealed checkpoint on the same filesystem.

    ``seal_checkpoint`` has already flushed and SHA-256 inventoried every
    payload byte.  A same-filesystem rename preserves those exact inodes and
    avoids copy plus three redundant 17+ GiB reads.  Resume and root acceptance
    independently rehash the durable destination.
    """
    from campaign_checkpoint import flush_directory

    source, destination = Path(source), Path(destination)
    require(source.is_dir(), "sealed trainer checkpoint missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "durable checkpoint destination already exists")
    require(source.parent.stat().st_dev == destination.parent.stat().st_dev, "atomic checkpoint publication requires one filesystem")
    stored = json.loads((source / "campaign-manifest.json").read_text())
    require(stored == sealed_manifest, "sealed checkpoint manifest changed before publication")
    observed = {}
    for path in sorted(source.rglob("*")):
        require(not path.is_symlink(), "sealed checkpoint contains a symlink")
        if path.is_file() and path.name != "campaign-manifest.json":
            observed[str(path.relative_to(source))] = path.stat().st_size
    require(observed == {name: value["bytes"] for name, value in sealed_manifest["files"].items()}, "sealed checkpoint file set/size changed before publication")
    flush_directory(source)
    os.rename(source, destination)
    flush_directory(source.parent); flush_directory(destination.parent)
    require(json.loads((destination / "campaign-manifest.json").read_text()) == sealed_manifest, "published checkpoint manifest differs")
    return destination


def run(recipe_path, resume=None):
    recipe = load_bound(recipe_path); recipe_hash = sha256(recipe_path)
    dataset, run_preflight = _validated_cohort(recipe); runtime = recipe["runtime"]
    require(run_preflight["status"] == "pass", "in-process cohort/holdout preflight failed")
    train_context = recipe["cohort"]["max_sequence_tokens"]
    output = Path(recipe["outputs"]["trainer"]); archive = Path(recipe["outputs"]["archive"])
    if resume is None: require(not output.exists() and not archive.exists(), "fresh run output/archive must not exist")
    else:
        resume = Path(resume); require(resume.is_dir(), "resume checkpoint missing")
        from campaign_checkpoint import verify_checkpoint
        prior = verify_checkpoint(resume, identity(recipe), require_full=True, expected_checkpoint_kind="full_weights")
        initial_cursor = prior["step"] * 16
    if resume is None: initial_cursor = 0
    output.mkdir(parents=True, exist_ok=True); archive.mkdir(parents=True, exist_ok=True)
    telemetry = archive / "telemetry.jsonl"; stop_path = Path(recipe["outputs"]["graceful_stop"])

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
    from campaign_checkpoint import seal_checkpoint, verify_checkpoint
    from campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from trainer_tokenizer_alignment import (assert_runtime_tokenizer, assert_serialized_tokenizer,
        embedding_identity, restore_trainer_eog_alignment)
    from campaign_cpt_data import causal_lm_collator
    from campaign_cpt_eval import package_holdout_evaluator

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one root-owned CUDA device required")
    set_seed(recipe["seed"])
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=recipe["parent"]["path"], max_seq_length=train_context, dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=True, float32_mixed_precision=False, fast_inference=False, trust_remote_code=False, use_gradient_checkpointing=True)
    reference = load_pinned_reference_tokenizer(Path(recipe["parent"]["path"]))
    post_load_repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    require(post_load_repair.get("repair", {}).get("vocab_mapping_unchanged") is True, "post-load token repair changed vocabulary")
    embeddings_before_trainer = embedding_identity(model)
    token_stage_audits = [{"stage": "post_load_repair", **post_load_repair}, assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "post_load")]
    model.config.use_cache = False
    try: FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    except TypeError: FastLanguageModel.for_training(model)
    model.train(); named = list(model.named_parameters())
    require(len(named) == EXPECTED_PARAMETER_TENSORS and sum(p.numel() for _, p in named) == EXPECTED_PARAMETERS and all(p.requires_grad for _, p in named), "full-weight trainable inventory differs")
    require(not any("lora_" in n.lower() or "modules_to_save" in n.lower() for n, _ in named), "PEFT parameter entered full-weight run")

    expected_dispatch = json.loads(Path(recipe["optimizer_dispatch"]["path"]).read_text())
    selected_milestones = set(runtime["selected_milestones"])
    evaluator_recipe = {"validation_rows": recipe["validation"], "validation_batch_size": 1, "parameters": {"max_sequence_tokens": recipe["validation"]["max_sequence_tokens"]}, "materialized_rows": True}
    evaluator = package_holdout_evaluator(evaluator_recipe)
    token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "before_parent_baseline"))
    baseline = evaluator(model, tokenizer, Path(recipe["parent"]["path"]), 0)
    token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "after_parent_baseline"))
    baseline_path = archive / "evaluations" / "baseline-step-0.json"
    write_json(baseline_path, baseline)
    observed_positions = []

    def collate(items):
        positions = [item.pop("_draw_position") for item in items]
        batch = causal_lm_collator(items, max_sequence_tokens=train_context)
        batch["_draw_position"] = torch.tensor(positions, dtype=torch.long)
        return batch

    class CampaignCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            repair = restore_trainer_eog_alignment(model, tokenizer, reference)
            require(repair.get("repair", {}).get("vocab_mapping_unchanged") is True, "train-begin token repair changed vocabulary")
            token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "train_begin"))
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            finite = nonzero = 0
            for _, parameter in named:
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()), "missing or nonfinite full-weight gradient")
                finite += 1; nonzero += int(bool(parameter.grad.detach().abs().max().item() > 0))
            if (int(state.global_step) + 1) % runtime["telemetry_every"] == 0:
                append_jsonl(telemetry, {"event": "pre_optimizer", "next_step": int(state.global_step) + 1, "finite_gradient_tensors": finite, "nonzero_gradient_tensors": nonzero, "allocated_bytes": int(torch.cuda.memory_allocated()), "reserved_bytes": int(torch.cuda.memory_reserved()), "at": time.time()})
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs: require(math.isfinite(float(logs["loss"])), "nonfinite logged loss")
            append_jsonl(telemetry, {"event": "log", "step": int(state.global_step), "logs": logs or {}, "at": time.time()}); return control

        def on_step_end(self, args, state, control, **kwargs):
            if stop_path.exists():
                request = json.loads(stop_path.read_text())
                require(request == {"action": "save_and_stop", "bound_recipe_sha256": recipe_hash}, "graceful stop request identity differs")
                control.should_save = True; control.should_training_stop = True
                append_jsonl(telemetry, {"event": "graceful_stop_observed_at_optimizer_boundary", "step": int(state.global_step), "at": time.time()})
            return control

        def on_save(self, args, state, control, **kwargs):
            step = int(state.global_step); source = Path(args.output_dir) / f"checkpoint-{step}"
            token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, f"before_checkpoint_{step}"))
            serialized = assert_serialized_tokenizer(source, reference, recipe["parent"]["files"]["tokenizer.json"], sha256)
            sampler = {"method": "sequential_frozen_draw_schedule", "draw_schedule_sha256": recipe["cohort"]["draw_schedule"]["sha256"], "cursor": step * 16, "global_step": step, "effective_batch": 16, "ignore_data_skip": False}
            sealed = seal_checkpoint(source, identity(recipe), step, full=True, sampler=sampler, checkpoint_kind="full_weights")
            destination = publish_sealed_checkpoint(source, archive / "full" / source.name, sealed)
            token_stage_audits.append({"stage": f"serialized_checkpoint_{step}", **serialized})
            if step in set(runtime["evaluation_steps"]):
                token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, f"before_evaluation_{step}"))
                result = evaluator(kwargs["model"], tokenizer, destination, step)
                token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, f"after_evaluation_{step}"))
                write_json(archive / "evaluations" / f"step-{step}.json", result)
            retention = retain_archives(archive, selected_milestones, latest=2)
            append_jsonl(telemetry, {"event": "checkpoint_durable", "step": step, "path": str(destination), "retention": retention, "at": time.time()})
            return control

    class FullWeightTrainer(FullWeightOptimizerTrainerMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None): return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
        def compute_loss(self, model, inputs, *args, **kwargs):
            if not observed_positions:
                token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "first_compute_loss"))
            positions = [int(x) for x in inputs.pop("_draw_position").detach().cpu().tolist()]
            expected = list(range(initial_cursor + len(observed_positions), initial_cursor + len(observed_positions) + len(positions)))
            require(positions == expected, "actual draw cursor differs from frozen schedule")
            observed_positions.extend(positions)
            return super().compute_loss(model, inputs, *args, **kwargs)
        def create_optimizer(self):
            result = super().create_optimizer()
            for key in ("arm", "ordered_rows", "ordered_rows_sha256", "parameter_objects", "parameters", "counts"):
                require(self.full_weight_optimizer_manifest.get(key) == expected_dispatch.get(key), f"optimizer dispatch differs:{key}")
            return result

    optimizer = dict(runtime["optimizer"]); optimizer.pop("arm", None)
    optimizer_config = OptimizerConfig(arm=runtime["optimizer"]["arm"], **optimizer)
    args = TrainingArguments(**training_arguments_kwargs(recipe, output))
    trainer = FullWeightTrainer(model=model, args=args, train_dataset=dataset, data_collator=collate, processing_class=tokenizer, callbacks=[CampaignCallback()]); trainer.full_weight_optimizer_config = optimizer_config
    post_trainer_repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    require(post_trainer_repair.get("repair", {}).get("vocab_mapping_unchanged") is True, "post-Trainer token repair changed vocabulary")
    token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "post_trainer_construction"))
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    terminal_tokenizer = assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "after_training")
    terminal_step = int(trainer.state.global_step); expected_draws = (terminal_step * 16) - initial_cursor
    require(len(observed_positions) == expected_draws, "terminal observed draw count differs")
    terminal = archive / "full" / f"checkpoint-{terminal_step}"
    require(terminal.is_dir(), "terminal durable full checkpoint missing")
    verify_checkpoint(terminal, identity(recipe), require_full=True, expected_checkpoint_kind="full_weights")
    report = {"schema": "sepalith.sft11.full-weight-cpt-lr-pilot-run.v2", "status": "schedule_complete" if terminal_step == runtime["max_steps"] else "gracefully_stopped", "global_step": terminal_step, "initial_cursor": initial_cursor, "observed_draws": len(observed_positions), "last_draw_position": observed_positions[-1] if observed_positions else None, "train_loss": float(result.training_loss), "terminal_checkpoint": str(terminal), "baseline_evaluation": {"path": str(baseline_path), "sha256": sha256(baseline_path)}, "tokenizer_contract": terminal_tokenizer, "tokenizer_stage_audits": token_stage_audits, "in_process_preflight": run_preflight, "cohort_scope": "short_lr_signal_experiment_not_production_dataset_cap"}
    write_json(archive / "run-result.json", report); print(json.dumps(report, sort_keys=True)); return report


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("preflight-template", "preflight", "run")); p.add_argument("--recipe", type=Path, required=True); p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if a.command == "preflight-template": print(json.dumps(preflight_template(a.recipe), sort_keys=True)); return 0
    if a.command == "preflight": print(json.dumps(preflight(a.recipe), sort_keys=True)); return 0
    run(a.recipe, a.resume); return 0


if __name__ == "__main__": raise SystemExit(main())
