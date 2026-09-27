#!/usr/bin/env python3
"""Full-weight target-only editing SFT over an immutable repaired TRAIN cohort."""
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
    require(manifest.get("schema") == "sepalith.full-weight-edit-sft-source.v1", "source schema differs")
    for item in manifest["files"]:
        target = path.parent / item["path"]
        require(target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"], f"source differs:{item['path']}")


def verify_data_queue(record):
    path = Path(record["path"])
    require(path.is_file() and sha256(path) == record["sha256"], "editing data queue binding differs")
    queue = json.loads(path.read_text())
    require(queue.get("schema") == "sepalith.sft11.full-weight-edit-data-queue.v2", "data queue schema differs")
    active = queue.get("active_cohort", {})
    require(active.get("status") == "accepted_root_reviewed" and active.get("rows") == 15006 and active.get("rows_sha256") == "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e", "accepted cohort queue identity differs")
    pending = queue.get("pending_cohorts")
    require(isinstance(pending, list) and len(pending) == 1, "pending cohort registry differs")
    roxy = pending[0]
    require(roxy.get("id") == "roxygen-supported-context-10017-v1" and roxy.get("rows") == 10017 and roxy.get("admitted_rows") == 0 and roxy.get("status") == "review_complete_training_admission_pending", "roxygen queue status differs")
    require(queue.get("additional_roxy_rows") == 10017 and queue.get("additional_roxy_status") == "pending_root_admission", "roxygen queue admission differs")
    return queue


def load_bound(path):
    recipe = json.loads(Path(path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-edit-sft-bound.v1" and recipe.get("status") == "root_admitted_not_launched", "bound recipe is not root-admitted")
    admission = recipe.get("root_admission", {})
    require(Path(admission.get("path", "")).is_file() and sha256(admission["path"]) == admission.get("sha256"), "root admission differs")
    verify_source(recipe)
    registry = recipe["data_queue"]
    queue = verify_data_queue(registry)
    require(registry["additional_roxy_rows"] == queue["additional_roxy_rows"] and registry["additional_roxy_status"] == queue["additional_roxy_status"], "recipe data queue summary differs")
    runtime = recipe["runtime"]
    require(runtime["micro_batch"] in (1, 2) and runtime["micro_batch"] * runtime["gradient_accumulation"] == runtime["effective_batch"] == 16, "effective batch differs")
    require(runtime["max_steps"] == recipe["cohort"]["updates"], "cohort horizon differs")
    require(runtime["checkpoint_every"] > 0 and runtime["max_steps"] % runtime["checkpoint_every"] == 0, "checkpoint cadence differs")
    validate_context_and_storage(recipe)
    for record in (recipe["cohort"]["rows"], recipe["cohort"]["draw_schedule"], recipe["cohort"]["manifest"], recipe["cohort"]["length_profile"], recipe["development"]["panel"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    parent = recipe["parent"]
    require(parent.get("kind") == "merged_cpt_parent", "full-weight editing parent must be a selected merged CPT parent")
    root = Path(parent["path"]); require(root.is_dir(), "parent path missing")
    for name, expected in parent["files"].items():
        target = root / name; require(target.is_file() and sha256(target) == expected, f"parent differs:{name}")
    dispatch = recipe["optimizer_dispatch"]
    require(Path(dispatch["path"]).is_file() and sha256(dispatch["path"]) == dispatch["sha256"], "optimizer dispatch differs")
    return recipe


def validate_context_and_storage(recipe):
    train_context = recipe["cohort"].get("max_sequence_tokens")
    require(type(train_context) is int and train_context in (2048, 4096, 8192, 16384, 32768), "TRAIN context is not a prepared power-of-two bound")
    require(recipe["development"].get("max_new_tokens") == 192 and recipe["development"].get("final_set_access") is False, "development/final boundary differs")
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
    def __init__(self, recipe):
        from campaign_edit_data import inspect
        rows, draws, schedule, totals = inspect(recipe["cohort"]["rows"], recipe["cohort"]["draw_schedule"], recipe["cohort"]["max_sequence_tokens"], 16)
        require(len(rows)==recipe["cohort"]["unique_rows"] and len(draws)==recipe["cohort"]["updates"]*16, "editing cohort denominator differs")
        require(totals=={k:recipe["cohort"][k] for k in ("input_tokens","loss_tokens","body_tokens")}, "editing token denominators differ")
        self.rows,self.draws,self.schedule=rows,draws,schedule
    def __len__(self): return len(self.draws)
    def __getitem__(self, position):
        row=dict(self.rows[self.draws[position]]);row["_draw_position"]=position;return row

def _validated_cohort(recipe):
    dataset=FrozenTokenRowDataset(recipe);panel=recipe["development"]["panel"]
    profile=json.loads(Path(recipe["cohort"]["length_profile"]["path"]).read_text())
    lengths=[len(row["input_ids"]) for row in dataset.rows]; targets=[len(row["input_ids"])-row["target_start"] for row in dataset.rows]
    require(profile.get("schema")=="sepalith.full-weight-edit-sft-length-profile.v1" and profile.get("rows_sha256")==recipe["cohort"]["rows"]["sha256"],"length profile identity differs")
    require(profile.get("rows")==len(dataset.rows) and profile.get("sequence_tokens",{}).get("sum")==sum(lengths) and profile.get("sequence_tokens",{}).get("max")==max(lengths),"sequence length profile differs")
    require(profile.get("supervised_target_tokens_including_eos",{}).get("sum")==sum(targets) and profile.get("supervised_target_tokens_including_eos",{}).get("max")==max(targets),"target length profile differs")
    require(profile.get("observed_fit_without_truncation") is True and max(lengths)<=recipe["cohort"]["max_sequence_tokens"],"observed sequence exceeds context")
    require(Path(panel["path"]).is_file() and sha256(panel["path"])==panel["sha256"],"development panel differs")
    require(recipe["development"]["teacher_forced_diagnostic"]=="separate_metric_not_edit_accuracy" and recipe["development"]["generation_evaluation"]=="required_for_edit_and_noop_quality","development metric roles differ")
    return dataset,{"status":"pass","unique_rows":len(dataset.rows),"draws":len(dataset),"updates":recipe["cohort"]["updates"],"max_sequence_tokens":recipe["cohort"]["max_sequence_tokens"],"development_cases":recipe["development"]["cases"],"additional_roxy_rows_pending":recipe["data_queue"]["additional_roxy_rows"],"cuda_started":False}

def preflight(recipe_path): return _validated_cohort(load_bound(recipe_path))[1]
def preflight_template(template_path):
    recipe=json.loads(Path(template_path).read_text())
    require(recipe.get("schema")=="sepalith.sft11.full-weight-edit-sft-template.v1" and recipe.get("launch_authorized") is False and recipe.get("parent") is None,"template must remain parent-null and launch-refusing")
    validate_context_and_storage(recipe);verify_source(recipe)
    for record in (recipe["cohort"]["rows"],recipe["cohort"]["draw_schedule"],recipe["cohort"]["manifest"],recipe["cohort"]["length_profile"],recipe["development"]["panel"],recipe["optimizer_dispatch"]):
        target=Path(record["path"]);require(target.is_file() and sha256(target)==record["sha256"],f"input differs:{target}")
    registry=recipe["data_queue"];queue=verify_data_queue(registry);require(registry["additional_roxy_rows"]==queue["additional_roxy_rows"] and registry["additional_roxy_status"]==queue["additional_roxy_status"],"data queue summary differs")
    return _validated_cohort(recipe)[1]

def identity(recipe):
    return {"parent":{"candidate_id":recipe["parent"]["candidate_id"],"weights_sha256":recipe["parent"]["files"]["model.safetensors"]},"tokenizer":{"sha256":recipe["parent"]["files"]["tokenizer.json"],"bos":0,"eos_pad":1,"native_eog":[1,130073]},"renderer":{"id":"zeta2-prm03-v1","objective":"exact_target_body_terminal_eos_only","max_sequence_tokens":recipe["cohort"]["max_sequence_tokens"]},"data":{"cohort_id":recipe["cohort"]["id"],"rows_sha256":recipe["cohort"]["rows"]["sha256"],"draw_schedule_sha256":recipe["cohort"]["draw_schedule"]["sha256"]},"source":{"manifest_sha256":recipe["source"]["manifest_sha256"]},"policy":{"stage":"full_weight_edit_sft_v1","optimizer":recipe["runtime"]["optimizer"],"full_text_labels":False},"schedule":{k:recipe["runtime"][k] for k in ("max_steps","effective_batch","micro_batch","gradient_accumulation","learning_rate","scheduler","warmup_ratio","checkpoint_every")}}

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


def build_evaluation_request(recipe, step, checkpoint):
    """Record a checkpoint-bound evaluation request without changing model mode."""
    return {"schema":"sepalith.full-weight-edit-evaluation-request.v1","status":"pending_resource_test_or_post_training_evaluator","step":step,"checkpoint":str(checkpoint),"panel":recipe["development"]["panel"],"teacher_forced":{"role":"diagnostic_only_not_edit_accuracy"},"generation":{"role":"required_edit_noop_quality","max_new_tokens":192},"final_set_access":False,"same_model_in_callback_executed":False}


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
    from campaign_edit_data import target_only_collator, verify_batch

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
    observed_positions = []

    def collate(items):
        positions = [item.pop("_draw_position") for item in items]
        rows=[dict(item) for item in items]
        batch=target_only_collator(rows,max_sequence_tokens=train_context)
        verify_batch(batch,rows)
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
                request=build_evaluation_request(recipe,step,destination)
                write_json(archive / "evaluations" / f"request-step-{step}.json",request)
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
    report = {"schema": "sepalith.sft11.full-weight-edit-sft-run.v1", "status": "schedule_complete" if terminal_step == runtime["max_steps"] else "gracefully_stopped", "global_step": terminal_step, "initial_cursor": initial_cursor, "observed_draws": len(observed_positions), "last_draw_position": observed_positions[-1] if observed_positions else None, "train_loss": float(result.training_loss), "terminal_checkpoint": str(terminal), "tokenizer_contract": terminal_tokenizer, "tokenizer_stage_audits": token_stage_audits, "in_process_preflight": run_preflight, "cohort_scope": "accepted_repaired15006_additional_roxy10017_pending"}
    write_json(archive / "run-result.json", report); print(json.dumps(report, sort_keys=True)); return report


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("preflight-template", "preflight", "run")); p.add_argument("--recipe", type=Path, required=True); p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if a.command == "preflight-template": print(json.dumps(preflight_template(a.recipe), sort_keys=True)); return 0
    if a.command == "preflight": print(json.dumps(preflight(a.recipe), sort_keys=True)); return 0
    run(a.recipe, a.resume); return 0


if __name__ == "__main__": raise SystemExit(main())
