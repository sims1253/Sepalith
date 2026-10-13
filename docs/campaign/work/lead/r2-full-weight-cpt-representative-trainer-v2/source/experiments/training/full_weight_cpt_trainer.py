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
    require(manifest.get("schema") == "sepalith.full-weight-cpt-representative-trainer-source.v2", "source schema differs")
    for item in manifest["files"]:
        target = path.parent / item["path"]
        require(target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"], f"source differs:{item['path']}")


def load_bound(path):
    recipe = json.loads(Path(path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-cpt-representative-bound.v2" and recipe.get("status") == "root_admitted_not_launched", "bound recipe is not root-admitted")
    admission = recipe.get("root_admission", {})
    require(Path(admission.get("path", "")).is_file() and sha256(admission["path"]) == admission.get("sha256"), "root admission differs")
    verify_source(recipe)
    dtype_audit = recipe.get("dtype_audit", {})
    require(Path(dtype_audit.get("path", "")).is_file() and sha256(dtype_audit["path"]) == dtype_audit.get("sha256"), "saved precision audit differs")
    data_admission = recipe.get("data_admission", {})
    require(Path(data_admission.get("path", "")).is_file() and sha256(data_admission["path"]) == data_admission.get("sha256"), "data admission differs")
    runtime = recipe["runtime"]
    require(runtime["micro_batch"] in (1, 2) and runtime["micro_batch"] * runtime["gradient_accumulation"] == runtime["effective_batch"] == 16, "effective batch differs")
    require(runtime["max_steps"] == recipe["cohort"]["updates"], "cohort horizon differs")
    require(type(runtime["checkpoint_every"]) is int and runtime["checkpoint_every"] > 0, "checkpoint cadence differs")
    require(type(runtime["mandatory_stop_step"]) is int and 0 < runtime["mandatory_stop_step"] < runtime["max_steps"], "mandatory intermediate stop differs")
    require(runtime["mandatory_stop_step"] % runtime["checkpoint_every"] == 0, "mandatory stop lacks scheduled full checkpoint")
    require(runtime["mandatory_stop_step"] in runtime["evaluation_steps"] and runtime["mandatory_stop_step"] in runtime["selected_milestones"], "mandatory stop must be evaluated and preserved")
    require(runtime["max_steps"] in runtime["selected_milestones"], "terminal milestone must be preserved")
    validate_context_and_storage(recipe)
    for record in (recipe["cohort"]["rows"], recipe["cohort"]["draw_schedule"], recipe["cohort"]["manifest"], recipe["validation"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    parent = recipe["parent"]
    require(parent.get("kind") == "merged_cpt_parent", "full-weight CPT parent kind differs")
    precision = parent.get("saved_precision", {})
    require(type(precision.get("fp32_tensors")) is int and precision["fp32_tensors"] >= 0, "parent saved FP32 tensor count differs")
    require(type(precision.get("fp32_elements")) is int and precision["fp32_elements"] >= 0, "parent saved FP32 element count differs")
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
        from campaign_cpt_data import validate_materialized_rows, validate_draw_schedule
        rows_path = Path(recipe["cohort"]["rows"]["path"])
        offsets, raw_rows = {}, []
        with rows_path.open("rb") as stream:
            while True:
                offset = stream.tell(); line = stream.readline()
                if not line: break
                raw = json.loads(line); ident = raw.get("row_id")
                require(isinstance(ident, str) and ident not in offsets, "duplicate or invalid cohort row ID")
                offsets[ident] = offset; raw_rows.append(raw)
        cohort = recipe["cohort"]
        summary = validate_materialized_rows(raw_rows, max_sequence_tokens=cohort["max_sequence_tokens"], require_complete_documents=True)
        expected = {"rows": cohort["unique_rows"], "documents": cohort["documents"], "packages": cohort["packages"], "input_tokens": cohort["input_tokens"], "payload_tokens": cohort["payload_tokens"], "loss_tokens": cohort["loss_tokens"]}
        observed = {"rows": summary["rows"], "documents": summary["documents"], "packages": len(summary["packages"]), "input_tokens": summary["input_tokens"], "payload_tokens": summary["payload_tokens"], "loss_tokens": summary["loss_tokens"]}
        require(observed == expected, "cohort row/document/token denominators differ")
        schedule = json.loads(Path(cohort["draw_schedule"]["path"]).read_text())
        schedule_audit = validate_draw_schedule(schedule, summary["rows_checked"], token_rows_sha256=cohort["rows"]["sha256"], max_steps=cohort["updates"], effective_batch=16)
        draws = schedule_audit["row_ids"]
        require(len(draws) - len(offsets) == cohort["named_replays"], "named replay denominator differs")
        first = draws[:len(offsets)]
        require(len(set(first)) == len(offsets) and set(first) == set(offsets), "every unique row must appear before any replay")
        self.path, self.offsets, self.draws = rows_path, offsets, draws
        self.packages = set(summary["packages"])
        self.documents = {item["document_sha256"] for item in summary["documents_detail"]}

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
    valid_packages = {row["package"] for row in valid}; valid_documents = {row["source_sha256"] for row in valid}
    require(not dataset.packages.intersection(valid_packages) and not dataset.documents.intersection(valid_documents), "CPT holdout leakage")
    return dataset, {"status": "pass", "cohort_rows": len(dataset.offsets), "draws": len(dataset), "validation_rows": len(valid), "train_context_tokens": recipe["cohort"]["max_sequence_tokens"], "validation_context_tokens": validation_context, "cohort_scope": "representative_complete_documents_diagnostic_not_all_eligible_corpus", "cuda_started": False}


def _cohort_preflight(recipe):
    return _validated_cohort(recipe)[1]


def preflight(recipe_path):
    return _cohort_preflight(load_bound(recipe_path))


def preflight_template(template_path):
    recipe = json.loads(Path(template_path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-cpt-representative-template.v2" and recipe.get("launch_authorized") is False, "preparation template differs")
    verify_source(recipe)
    dtype_audit = recipe.get("dtype_audit", {})
    require(Path(dtype_audit.get("path", "")).is_file() and sha256(dtype_audit["path"]) == dtype_audit.get("sha256"), "saved precision audit differs")
    require(recipe.get("cohort") is None, "preparation template must not prebind producer data")
    for record in (recipe["validation"], recipe["optimizer_dispatch"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    for parent in (item for item in recipe["parent_candidates"].values() if item.get("available") is True):
        root = Path(parent["path"])
        require(root.is_dir() and parent.get("kind") == "merged_cpt_parent", "prepared parent metadata differs")
        # The bound preflight rehashes the multi-GB weights. Template review
        # checks only small identity/config files and does not read model payloads.
        for name, expected in parent["files"].items():
            if name == "model.safetensors": continue
            target = root / name; require(target.is_file() and sha256(target) == expected, f"parent candidate differs:{name}")
    return {"status": "prepared_inputs_pending", "cohort": None, "validation_rows": 499, "cuda_started": False}


def identity(recipe):
    return {
        "parent": {"candidate_id": recipe["parent"]["candidate_id"], "weights_sha256": recipe["parent"]["files"]["model.safetensors"], "saved_precision": recipe["parent"]["saved_precision"]},
        "tokenizer": {"sha256": recipe["parent"]["files"]["tokenizer.json"], "bos": 0, "eos_pad": 1},
        "renderer": {"kind": "pretokenized_raw_r_cpt_v1", "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"]},
        "data": {"cohort_id": recipe["cohort"]["id"], "rows_sha256": recipe["cohort"]["rows"]["sha256"]},
        "source": {"manifest_sha256": recipe["source"]["manifest_sha256"]},
        "policy": {"stage": "full_weight_cpt_representative_v2", "optimizer": recipe["runtime"]["optimizer"]},
        "schedule": {k: recipe["runtime"][k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_ratio", "checkpoint_every", "mandatory_stop_step")},
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


def validate_continuation(recipe_path, recipe, resume, continuation_admission):
    require(continuation_admission is not None, "resume requires a root continuation admission")
    path = Path(continuation_admission); require(path.is_file(), "continuation admission missing")
    value = json.loads(path.read_text())
    require(value.get("schema") == "sepalith.sft11.cpt-representative-continuation-admission.v2", "continuation admission schema differs")
    require(value.get("status") == "admitted" and value.get("decision") == "continue", "continuation is not admitted")
    require(value.get("bound_recipe_sha256") == sha256(recipe_path), "continuation refers to another recipe")
    require(Path(value.get("checkpoint", "")).resolve() == Path(resume).resolve(), "continuation refers to another checkpoint")
    manifest = Path(resume) / "campaign-manifest.json"
    require(manifest.is_file() and value.get("checkpoint_manifest_sha256") == sha256(manifest), "continuation checkpoint manifest differs")
    require(value.get("step") == recipe["runtime"]["mandatory_stop_step"], "continuation step is not the mandatory gate")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def milestone_action(step, initial_cursor, runtime):
    """Return save/stop decisions at an optimizer boundary."""
    require(type(step) is int and step >= 0 and initial_cursor % 16 == 0, "milestone cursor input differs")
    mandatory = runtime["mandatory_stop_step"]
    return {"save": step in {mandatory, runtime["max_steps"]},
            "stop": step == mandatory and initial_cursor < step,
            "cursor": step * 16}


def run(recipe_path, resume=None, continuation_admission=None):
    recipe = load_bound(recipe_path); recipe_hash = sha256(recipe_path)
    dataset, run_preflight = _validated_cohort(recipe); runtime = recipe["runtime"]
    require(run_preflight["status"] == "pass", "in-process cohort/holdout preflight failed")
    train_context = recipe["cohort"]["max_sequence_tokens"]
    output = Path(recipe["outputs"]["trainer"]); archive = Path(recipe["outputs"]["archive"])
    if resume is None: require(not output.exists() and not archive.exists(), "fresh run output/archive must not exist")
    else:
        resume = Path(resume); require(resume.is_dir(), "resume checkpoint missing")
        continuation = validate_continuation(recipe_path, recipe, resume, continuation_admission)
        from campaign_checkpoint import verify_checkpoint
        prior = verify_checkpoint(resume, identity(recipe), require_full=True, expected_checkpoint_kind="full_weights")
        initial_cursor = prior["step"] * 16
    if resume is None: initial_cursor = 0; continuation = None
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
    from saved_precision import restore_saved_fp32

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
    model.train()
    saved_precision_audit = restore_saved_fp32(model, Path(recipe["parent"]["path"]) / "model.safetensors")
    expected_precision = recipe["parent"]["saved_precision"]
    require(saved_precision_audit["fp32_tensors_restored"] == expected_precision["fp32_tensors"], "restored FP32 tensor count differs from admitted parent")
    require(sum(item["elements"] for item in saved_precision_audit["tensors"]) == expected_precision["fp32_elements"], "restored FP32 element count differs from admitted parent")
    named = list(model.named_parameters())
    require(len(named) == EXPECTED_PARAMETER_TENSORS and sum(p.numel() for _, p in named) == EXPECTED_PARAMETERS and all(p.requires_grad for _, p in named), "full-weight trainable inventory differs")
    require(not any("lora_" in n.lower() or "modules_to_save" in n.lower() for n, _ in named), "PEFT parameter entered full-weight run")

    expected_dispatch = json.loads(Path(recipe["optimizer_dispatch"]["path"]).read_text())
    selected_milestones = set(runtime["selected_milestones"])
    evaluator_recipe = {"validation_rows": recipe["validation"], "validation_batch_size": 1, "parameters": {"max_sequence_tokens": recipe["validation"]["max_sequence_tokens"]}, "materialized_rows": True}
    evaluator = package_holdout_evaluator(evaluator_recipe)
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
            step = int(state.global_step)
            milestone = milestone_action(step, initial_cursor, runtime)
            if milestone["save"]:
                control.should_save = True
            if milestone["stop"]:
                control.should_save = True; control.should_training_stop = True
                append_jsonl(telemetry, {"event": "mandatory_milestone_stop_at_optimizer_boundary", "step": step, "cursor": milestone["cursor"], "at": time.time()})
            if stop_path.exists():
                request = json.loads(stop_path.read_text())
                require(request == {"action": "save_and_stop", "bound_recipe_sha256": recipe_hash}, "graceful stop request identity differs")
                control.should_save = True; control.should_training_stop = True
                append_jsonl(telemetry, {"event": "graceful_stop_observed_at_optimizer_boundary", "step": step, "at": time.time()})
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
    report = {"schema": "sepalith.sft11.full-weight-cpt-representative-run.v2", "status": "schedule_complete" if terminal_step == runtime["max_steps"] else ("mandatory_milestone_stopped" if terminal_step == runtime["mandatory_stop_step"] else "gracefully_stopped"), "global_step": terminal_step, "initial_cursor": initial_cursor, "observed_draws": len(observed_positions), "last_draw_position": observed_positions[-1] if observed_positions else None, "train_loss": float(result.training_loss), "terminal_checkpoint": str(terminal), "tokenizer_contract": terminal_tokenizer, "tokenizer_stage_audits": token_stage_audits, "saved_precision_audit": saved_precision_audit, "in_process_preflight": run_preflight, "cohort_scope": "representative_complete_documents_diagnostic_not_all_eligible_corpus", "continuation_admission": continuation}
    write_json(archive / "run-result.json", report); print(json.dumps(report, sort_keys=True)); return report


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("preflight-template", "preflight", "run")); p.add_argument("--recipe", type=Path, required=True); p.add_argument("--resume", type=Path); p.add_argument("--continuation-admission", type=Path)
    a = p.parse_args()
    if a.command == "preflight-template": print(json.dumps(preflight_template(a.recipe), sort_keys=True)); return 0
    if a.command == "preflight": print(json.dumps(preflight(a.recipe), sort_keys=True)); return 0
    run(a.recipe, a.resume, a.continuation_admission); return 0


if __name__ == "__main__": raise SystemExit(main())
