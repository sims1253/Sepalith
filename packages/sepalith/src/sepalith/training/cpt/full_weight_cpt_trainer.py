#!/usr/bin/env python3
"""Full-weight CPT Trainer for an explicit full-state corpus transition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import fcntl
from typing import Sequence

from sepalith.training.paths import CHECKPOINT_ROOT


EXPECTED_PARAMETERS = 2_516_756_480
EXPECTED_PARAMETER_TENSORS = 381
_SOURCE_LOCKS = []


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

def phase_event(path,event,started,**fields):
    value={'event':event,'elapsed_seconds':time.monotonic()-started,'at':time.time(),**fields};append_jsonl(path,value);print(json.dumps(value,sort_keys=True),flush=True);return value


def verify_source(recipe):
    record = recipe.get("runtime_source")
    require(isinstance(record,dict), "runtime source remains unbound")
    path = Path(record["manifest_path"])
    require(path.is_file(),"runtime source manifest missing")
    manifest_fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0));fcntl.flock(manifest_fd,fcntl.LOCK_SH);before=os.fstat(manifest_fd);raw=b''
    with os.fdopen(os.dup(manifest_fd),'rb') as stream:raw=stream.read()
    after=os.fstat(manifest_fd);require(before.st_dev==after.st_dev and before.st_ino==after.st_ino and before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns and before.st_ctime_ns==after.st_ctime_ns,'runtime source manifest changed during verification')
    require(hashlib.sha256(raw).hexdigest()==record['manifest_sha256'],'runtime source manifest differs');_SOURCE_LOCKS.append(manifest_fd)
    manifest = json.loads(raw)
    require(manifest.get("schema") == "sepalith.sft11.native-cpt-trainer-source.v2", "runtime source schema differs")
    for item in manifest["files"]:
        target = path.parent / item["path"]
        fd=os.open(target,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0));fcntl.flock(fd,fcntl.LOCK_SH);before=os.fstat(fd);h=hashlib.sha256()
        with os.fdopen(os.dup(fd),'rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        after=os.fstat(fd);require(before.st_dev==after.st_dev and before.st_ino==after.st_ino and before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns and before.st_ctime_ns==after.st_ctime_ns,'runtime source changed during verification')
        require(before.st_size==item['bytes'] and h.hexdigest()==item['sha256'],f"runtime source differs:{item['path']}");_SOURCE_LOCKS.append(fd)


def validate_stage_schedule(recipe):
    """Validate global runtime steps against the destination stage-local cadence."""
    runtime = recipe["runtime"]
    offset = recipe["transition"]["global_optimizer_step_offset"]
    require(type(offset) is int and offset >= 0, "global optimizer step offset differs")
    require(runtime["max_steps"] == offset + recipe["cohort"]["updates"], "global horizon/source offset differs")
    require(type(runtime["checkpoint_every"]) is int and runtime["checkpoint_every"] > 0, "checkpoint cadence differs")
    mandatory = runtime["mandatory_stop_step"]
    require(type(mandatory) is int and offset < mandatory < runtime["max_steps"], "mandatory intermediate stop differs")
    require((mandatory - offset) % runtime["checkpoint_every"] == 0, "mandatory stop lacks scheduled stage-local full checkpoint")
    for name in ("evaluation_steps", "selected_milestones"):
        steps = runtime[name]
        require(isinstance(steps, list) and steps == sorted(set(steps)), f"{name} differ")
        require(all(type(step) is int and offset < step <= runtime["max_steps"] for step in steps), f"{name} leave destination stage")
    require(mandatory in runtime["evaluation_steps"] and mandatory in runtime["selected_milestones"], "mandatory stop must be evaluated and preserved")
    require(runtime["max_steps"] in runtime["evaluation_steps"] and runtime["max_steps"] in runtime["selected_milestones"], "terminal milestone must be evaluated and preserved")
    return {"offset": offset, "mandatory_stage_step": mandatory - offset, "terminal_stage_step": runtime["max_steps"] - offset}


def load_bound(path):
    recipe = json.loads(Path(path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-cpt-stage-transition-bound.v1" and recipe.get("status") == "root_admitted_not_launched", "bound recipe is not root-admitted")
    admission = recipe.get("root_admission", {})
    require(Path(admission.get("path", "")).is_file() and sha256(admission["path"]) == admission.get("sha256"), "root admission differs")
    verify_source(recipe)
    from sepalith.training.contracts.native_runtime_contract import validate as validate_native
    native = validate_native(recipe, identity)
    dtype_audit = recipe.get("dtype_audit", {})
    require(Path(dtype_audit.get("path", "")).is_file() and sha256(dtype_audit["path"]) == dtype_audit.get("sha256"), "saved precision audit differs")
    data_admission = recipe.get("data_admission", {})
    require(Path(data_admission.get("path", "")).is_file() and sha256(data_admission["path"]) == data_admission.get("sha256"), "data admission differs")
    corpus_manifest = recipe.get("corpus_manifest", {})
    require(Path(corpus_manifest.get("path", "")).is_file() and sha256(corpus_manifest["path"]) == corpus_manifest.get("sha256"), "corpus manifest differs")
    runtime = recipe["runtime"]
    require(runtime["micro_batch"] in (1, 2) and runtime["micro_batch"] * runtime["gradient_accumulation"] == runtime["effective_batch"] == 16, "effective batch differs")
    validate_stage_schedule(recipe)
    validate_context_and_storage(recipe)
    for record in (recipe["cohort"]["manifest"], recipe["validation"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    parent = recipe["parent"]
    require(parent.get("kind") == "full_weight_stage_checkpoint", "full-weight CPT transition parent kind differs")
    precision = parent.get("saved_precision", {})
    require(type(precision.get("fp32_tensors")) is int and precision["fp32_tensors"] >= 0, "parent saved FP32 tensor count differs")
    require(type(precision.get("fp32_elements")) is int and precision["fp32_elements"] >= 0, "parent saved FP32 element count differs")
    root = Path(parent["path"]); require(root.is_dir(), "parent path missing")
    dispatch = recipe["optimizer_dispatch"]
    require(Path(dispatch["path"]).is_file() and sha256(dispatch["path"]) == dispatch["sha256"], "optimizer dispatch differs")
    recipe['_native_validation']=native
    return recipe


def validate_context_and_storage(recipe):
    train_context = recipe["cohort"].get("max_sequence_tokens")
    require(type(train_context) is int and train_context in (2048, 4096, 8192, 16384, 32768), "TRAIN context is not a prepared power-of-two bound")
    require(recipe["validation"].get("max_sequence_tokens") == 2048, "heldout comparator must remain fixed at 2048 tokens")
    storage = recipe.get("checkpoint_storage", {})
    require(storage.get("mode") == "native_hot_to_e_durable_atomic" and storage.get("c_hot_stage") == "root_admitted", "checkpoint storage policy differs")
    require(storage.get("required_same_filesystem") is False, "native-to-E publication must be cross-filesystem")
    trainer_path, archive_path = Path(recipe["outputs"]["trainer"]), Path(recipe["outputs"]["archive"])
    require(str(trainer_path).startswith("/home/m0hawk/.local/state/sepalith/campaign-20260915/") and str(archive_path).startswith(CHECKPOINT_ROOT), "native trainer/E archive paths differ")
    require(Path(storage.get("trainer_root", "")) == trainer_path and Path(storage.get("archive_root", "")) == archive_path, "checkpoint storage paths differ from outputs")
    require(recipe.get('retention',{}).get('trainer_save_total_limit')==1 and recipe['retention'].get('durable_archive_latest')==2,'native hot/durable retention differs')
    require(storage.get('trainer_transient_save_total_limit')==2 and storage.get('native_retained_after_durable_publication')==1,'native transient/durable retention metadata differs')


def training_arguments_kwargs(recipe, output):
    runtime = recipe["runtime"]
    warmup_steps = runtime["warmup_steps"]
    # Keep the prior native checkpoint until the new checkpoint has reached E.
    # The callback prunes to one native checkpoint only after durable publish.
    return {"output_dir": str(output), "per_device_train_batch_size": runtime["micro_batch"], "gradient_accumulation_steps": runtime["gradient_accumulation"], "max_steps": runtime["max_steps"], "learning_rate": runtime["learning_rate"], "lr_scheduler_type": runtime["scheduler"], "warmup_steps": warmup_steps, "max_grad_norm": 0.0, "bf16": True, "use_cache": False, "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}, "save_strategy": "no", "save_only_model": False, "save_total_limit": 2, "logging_strategy": "steps", "logging_steps": 1, "logging_first_step": True, "eval_strategy": "no", "report_to": "none", "disable_tqdm": True, "remove_unused_columns": False, "dataloader_num_workers": 0, "dataloader_pin_memory": False, "train_sampling_strategy": "sequential", "ignore_data_skip": True, "seed": recipe["seed"], "data_seed": recipe["seed"]}


def deterministic_sequential_dataloader(trainer):
    """Build the production loader without consuming checkpointed model RNG.

    PyTorch's DataLoader iterator otherwise draws its base seed from global CPU
    RNG after Transformers restores rng_state.pth. A dedicated generator keeps
    stage-cursor resume from perturbing model/optimizer random state.
    """
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    require(trainer.args.dataloader_num_workers == 0, "reviewed deterministic loader requires zero workers")
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(trainer.args.data_seed))
    loader = DataLoader(trainer.train_dataset, batch_size=trainer._train_batch_size,
        sampler=SequentialSampler(trainer.train_dataset), collate_fn=trainer.data_collator,
        num_workers=0, pin_memory=trainer.args.dataloader_pin_memory,
        drop_last=trainer.args.dataloader_drop_last, generator=generator)
    return trainer.accelerator.prepare(loader)


def _validated_cohort(recipe, initial_cursor=0):
    from sepalith.training.cpt.streaming_trainer_adapter import dataset_from_bound_recipe
    return dataset_from_bound_recipe(recipe, initial_cursor)

def _cohort_preflight(recipe):
    return _validated_cohort(recipe)[1]


def preflight(recipe_path):
    return _cohort_preflight(load_bound(recipe_path))

def preflight_resume(recipe_path,resume,continuation_admission,execution_stop_admission=None,ordinary_canary_transition_admission=None):
    recipe=load_bound(recipe_path);resume=Path(resume);continuation=validate_continuation(recipe_path,recipe,resume,continuation_admission)
    from sepalith.training.cpt.ordinary_canary_resume import resolve_resume_identity
    resume_identity,resume_lineage=resolve_resume_identity(recipe_path,identity(recipe),resume,ordinary_canary_transition_admission)
    from sepalith.training.checkpoint.campaign_checkpoint import verify_checkpoint
    manifest=verify_checkpoint(resume,resume_identity,require_full=True,expected_checkpoint_kind='full_weights',native_bundle_id=recipe['_native_validation']['bundle_id'])
    from sepalith.training.contracts.stage_transition_contract import stage_cursor_from_checkpoint
    state=json.loads((resume/'campaign-state.json').read_text());cursor=stage_cursor_from_checkpoint(state,recipe['transition']['global_optimizer_step_offset'],recipe['cohort']['draw_schedule']['sha256']);dataset,cohort=_validated_cohort(recipe,cursor);dataset.close()
    execution_stop=validate_execution_stop(recipe_path,recipe,manifest['step'],execution_stop_admission)
    return {'status':'pass','resume_step':manifest['step'],'initial_cursor':cursor,'continuation_admission':continuation,'ordinary_canary_resume_lineage':resume_lineage,'execution_stop_admission':execution_stop,'cohort':cohort,'cuda_started':False}


def preflight_template(template_path):
    recipe = json.loads(Path(template_path).read_text())
    require(recipe.get("schema") == "sepalith.sft11.full-weight-cpt-stage-transition-template.v1" and recipe.get("launch_authorized") is False, "preparation template differs")
    verify_source(recipe)
    dtype_audit = recipe.get("dtype_audit", {})
    require(Path(dtype_audit.get("path", "")).is_file() and sha256(dtype_audit["path"]) == dtype_audit.get("sha256"), "saved precision audit differs")
    for record in (recipe["validation"], recipe["optimizer_dispatch"]):
        target = Path(record["path"]); require(target.is_file() and sha256(target) == record["sha256"], f"input differs:{target}")
    require(recipe.get("cohort") is None and recipe.get("data_admission") is None and recipe.get("parent") is None and recipe.get("transition") is None, "stage-transition template must remain unbound")
    return {"status": "source_checkpoint_destination_corpus_lr_and_gates_pending", "cohort": None, "validation_rows": 499, "cuda_started": False}


def identity(recipe):
    return {
        "parent": {"candidate_id": recipe["parent"]["candidate_id"], "weights_sha256": recipe["parent"]["files"]["model.safetensors"], "saved_precision": recipe["parent"]["saved_precision"], "source_checkpoint_manifest_sha256": recipe["transition"]["source_checkpoint"]["manifest_sha256"]},
        "tokenizer": {"sha256": recipe["parent"]["files"]["tokenizer.json"], "bos": 0, "eos_pad": 1},
        "renderer": {"kind": "pretokenized_raw_r_cpt_v1", "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"]},
        "data": {"cohort_id": recipe["cohort"]["id"], "rows_sha256": recipe["cohort"]["rows"]["sha256"], "streaming_cache_manifest_sha256": recipe["cohort"]["streaming_cache"]["manifest_sha256"]},
        "source": {"manifest_sha256": recipe["source"]["manifest_sha256"]},
        "policy": {"stage": "full_weight_cpt_stage_transition_v1", "optimizer": recipe["runtime"]["optimizer"]},
        "schedule": {**{k: recipe["runtime"][k] for k in ("max_steps", "effective_batch", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_steps", "checkpoint_every", "mandatory_stop_step")}, "global_optimizer_step_offset": recipe["transition"]["global_optimizer_step_offset"], "destination_updates": recipe["cohort"]["updates"]},
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
    from sepalith.training.checkpoint.campaign_checkpoint import flush_directory

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
    require(value.get("schema") == "sepalith.sft11.cpt-stage-transition-continuation-admission.v1", "continuation admission schema differs")
    require(value.get("status") == "admitted" and value.get("decision") == "continue" and value.get("launch_authorized") is True, "continuation is not admitted")
    require(value.get("bound_recipe_sha256") == sha256(recipe_path), "continuation refers to another recipe")
    require(Path(value.get("checkpoint", "")).resolve() == Path(resume).resolve(), "continuation refers to another checkpoint")
    manifest = Path(resume) / "campaign-manifest.json"
    require(manifest.is_file() and value.get("checkpoint_manifest_sha256") == sha256(manifest), "continuation checkpoint manifest differs")
    require(value.get("step") == json.loads(manifest.read_text()).get("step"), "continuation step differs from checkpoint")
    require(recipe["transition"]["global_optimizer_step_offset"] < value["step"] <= recipe["runtime"]["max_steps"], "continuation step is outside destination stage")
    from sepalith.training.contracts.stage_transition_contract import stage_cursor_from_checkpoint
    state = json.loads((Path(resume) / "campaign-state.json").read_text())
    stage_cursor_from_checkpoint(state, recipe["transition"]["global_optimizer_step_offset"], recipe["cohort"]["draw_schedule"]["sha256"])
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def validate_initial_transition(recipe_path, recipe, resume, transition_admission):
    require(transition_admission is not None, "initial corpus transition requires explicit root admission")
    admission = Path(transition_admission)
    expected = recipe["root_admission"]
    require(admission.is_file() and admission.resolve() == Path(expected["path"]).resolve() and sha256(admission) == expected["sha256"], "transition admission differs from bound root admission")
    require(Path(resume).resolve() == Path(recipe["transition"]["source_checkpoint"]["path"]).resolve(), "initial transition must load the admitted source checkpoint")
    from sepalith.training.contracts.stage_transition_contract import inspect_transition
    return {"root_admission": str(admission.resolve()), **inspect_transition(recipe, verify_payload=True)}


def validate_execution_stop(recipe_path, recipe, resume_step, admission_path):
    require(admission_path is not None, "native continuation requires an execution-stop admission")
    path = Path(admission_path); require(path.is_file(), "execution-stop admission missing")
    value = json.loads(path.read_text())
    require(value.get("schema") == "sepalith.sft11.native-cpt-execution-stop.v1", "execution-stop schema differs")
    require(value.get("status") == "admitted" and value.get("launch_authorized") is True, "execution stop is not admitted")
    require(value.get("bound_recipe_sha256") == sha256(recipe_path), "execution stop refers to another recipe")
    require(value.get("resume_global_step") == resume_step, "execution stop refers to another resume step")
    stop = value.get("stop_at_global_step"); offset = recipe["transition"]["global_optimizer_step_offset"]
    require(type(stop) is int and resume_step < stop <= recipe["runtime"]["max_steps"], "execution stop is outside continuation")
    require((stop - offset) % recipe["runtime"]["checkpoint_every"] == 0, "execution stop lacks a scheduled full checkpoint")
    return {"path": str(path.resolve()), "sha256": sha256(path), "stop_at_global_step": stop}


def milestone_action(step, initial_global_step, runtime, source_step=0, execution_stop=None):
    """Return save/stop decisions at an optimizer boundary."""
    require(type(step) is int and step >= source_step and initial_global_step >= source_step, "milestone input differs")
    mandatory = runtime["mandatory_stop_step"]
    stage_step = step - source_step
    stop = execution_stop if execution_stop is not None else mandatory
    return {"save": step in {mandatory, runtime["max_steps"], stop} or (stage_step > 0 and stage_step % runtime["checkpoint_every"] == 0),
            "stop": step == stop and initial_global_step < step,
            "cursor": (step - source_step) * 16}


def retain_native_after_durable(output, current_step):
    """Prune to one hot checkpoint only after the current one is durable."""
    output = Path(output); removed = []
    for path in sorted(output.glob("checkpoint-*")):
        try: step = int(path.name.rsplit("-", 1)[1])
        except ValueError: continue
        if step == current_step: continue
        retiring = path.with_name(f".retiring-{path.name}-{os.getpid()}")
        os.rename(path, retiring); shutil.rmtree(retiring); removed.append(step)
    return removed


def run(recipe_path, resume=None, continuation_admission=None, transition_admission=None, execution_stop_admission=None, ordinary_canary_transition_admission=None):
    started=time.monotonic();print(json.dumps({'event':'load_bound_start','at':time.time()},sort_keys=True),flush=True)
    recipe = load_bound(recipe_path); recipe_hash = sha256(recipe_path); runtime = recipe["runtime"];early=Path(recipe['outputs']['archive'])/'startup-telemetry.jsonl';phase_event(early,'load_bound_end',started)
    output = Path(recipe["outputs"]["trainer"]); archive = Path(recipe["outputs"]["archive"])
    require(resume is not None, "stage transition requires a full source or destination checkpoint")
    resume = Path(resume); require(resume.is_dir(), "resume checkpoint missing")
    source_step = recipe["transition"]["global_optimizer_step_offset"]
    require(transition_admission is None,'native trainer is admitted only for a verified same-stage checkpoint after global194')
    continuation = validate_continuation(recipe_path, recipe, resume, continuation_admission)
    from sepalith.training.cpt.ordinary_canary_resume import resolve_resume_identity
    resume_identity,resume_lineage=resolve_resume_identity(recipe_path,identity(recipe),resume,ordinary_canary_transition_admission)
    from sepalith.training.checkpoint.campaign_checkpoint import verify_checkpoint
    prior = verify_checkpoint(resume, resume_identity, require_full=True, expected_checkpoint_kind="full_weights", native_bundle_id=recipe['_native_validation']['bundle_id'])
    from sepalith.training.contracts.stage_transition_contract import stage_cursor_from_checkpoint
    state = json.loads((resume / "campaign-state.json").read_text())
    initial_cursor = stage_cursor_from_checkpoint(state, source_step, recipe["cohort"]["draw_schedule"]["sha256"])
    initial_global_step = prior["step"]
    execution_stop = validate_execution_stop(recipe_path, recipe, initial_global_step, execution_stop_admission)
    phase_event(early,'resume_validation_end',started,initial_global_step=initial_global_step,initial_cursor=initial_cursor)
    dataset, run_preflight = _validated_cohort(recipe, initial_cursor)
    require(run_preflight["status"] == "pass", "in-process cohort/holdout preflight failed")
    train_context = recipe["cohort"]["max_sequence_tokens"]
    phase_event(early,'dataset_preflight_end',started,initial_cursor=initial_cursor)
    output.mkdir(parents=True, exist_ok=True); archive.mkdir(parents=True, exist_ok=True)
    telemetry = archive / "telemetry.jsonl"; stop_path = Path(recipe["outputs"]["graceful_stop"])

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from sepalith.training.optim.full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
    from sepalith.training.checkpoint.campaign_checkpoint import seal_checkpoint, verify_checkpoint
    from sepalith.training.checkpoint.native_checkpoint_publish import publish as publish_native
    from sepalith.training.cpt.native_capacity import capacity as native_capacity
    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (assert_runtime_tokenizer, assert_serialized_tokenizer,
        embedding_identity, restore_trainer_eog_alignment)
    from sepalith.training.cpt.campaign_cpt_data import causal_lm_collator
    from sepalith.training.eval.campaign_cpt_eval import package_holdout_evaluator
    from sepalith.training.checkpoint.saved_precision import restore_saved_fp32

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one root-owned CUDA device required")
    set_seed(recipe["seed"])
    phase_event(early,'model_load_start',started)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(resume), max_seq_length=train_context, dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=True, float32_mixed_precision=False, fast_inference=False, trust_remote_code=False, use_gradient_checkpointing=True)
    phase_event(early,'model_load_end',started)
    reference = load_pinned_reference_tokenizer(resume)
    post_load_repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    require(post_load_repair.get("repair", {}).get("vocab_mapping_unchanged") is True, "post-load token repair changed vocabulary")
    embeddings_before_trainer = embedding_identity(model)
    token_stage_audits = [{"stage": "post_load_repair", **post_load_repair}, assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "post_load")]
    model.config.use_cache = False
    try: FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    except TypeError: FastLanguageModel.for_training(model)
    model.train()
    saved_precision_audit = restore_saved_fp32(model, resume / "model.safetensors")
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
            milestone = milestone_action(step, initial_global_step, runtime, source_step, execution_stop["stop_at_global_step"])
            if milestone["save"]:
                control.should_save = True
            if milestone["stop"]:
                control.should_save = True; control.should_training_stop = True
                append_jsonl(telemetry, {"event": "root_admitted_execution_stop_at_optimizer_boundary", "step": step, "cursor": milestone["cursor"], "at": time.time()})
            if stop_path.exists():
                request = json.loads(stop_path.read_text())
                require(request == {"action": "save_and_stop", "bound_recipe_sha256": recipe_hash}, "graceful stop request identity differs")
                control.should_save = True; control.should_training_stop = True
                append_jsonl(telemetry, {"event": "graceful_stop_observed_at_optimizer_boundary", "step": step, "at": time.time()})
            if control.should_save:
                native_root=Path(recipe['checkpoint_storage']['native_capacity_root']);existing=sum(p.stat().st_size for cp in Path(args.output_dir).glob('checkpoint-*') for p in cp.rglob('*') if p.is_file())
                native_capacity(native_root,recipe['_native_validation']['total_bytes'],existing,recipe['checkpoint_storage']['expected_full_checkpoint_bytes'])
                phase_event(telemetry,'checkpoint_save_requested',started,step=step,existing_native_hot_bytes=existing)
            return control

        def on_save(self, args, state, control, **kwargs):
            step = int(state.global_step); source = Path(args.output_dir) / f"checkpoint-{step}"
            phase_event(telemetry,'trainer_serialization_end_seal_start',started,step=step)
            token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, f"before_checkpoint_{step}"))
            serialized = assert_serialized_tokenizer(source, reference, recipe["parent"]["files"]["tokenizer.json"], sha256)
            stage_cursor = (step - source_step) * 16
            sampler = {"method": "sequential_frozen_draw_schedule_stage_local", "draw_schedule_sha256": recipe["cohort"]["draw_schedule"]["sha256"], "cursor": stage_cursor, "stage_cursor": stage_cursor, "global_step": step, "global_optimizer_step_offset": source_step, "effective_batch": 16, "ignore_data_skip": True}
            if resume_lineage is not None:sampler["resume_lineage"]=resume_lineage
            sealed = seal_checkpoint(source, identity(recipe), step, full=True, sampler=sampler, checkpoint_kind="full_weights")
            phase_event(telemetry,'native_seal_end_publish_start',started,step=step,checkpoint_bytes=sum(v['bytes']for v in sealed['files'].values()))
            destination=archive/'full'/source.name
            published=publish_native(source,destination,sha256(source/'campaign-manifest.json'))
            write_json(archive/'publish-receipts'/f'{source.name}.json',published)
            phase_event(telemetry,'durable_E_publish_end',started,step=step,destination=str(destination))
            removed_native = retain_native_after_durable(args.output_dir, step)
            token_stage_audits.append({"stage": f"serialized_checkpoint_{step}", **serialized})
            if step in set(runtime["evaluation_steps"]):
                token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, f"before_evaluation_{step}"))
                result = evaluator(kwargs["model"], tokenizer, destination, step)
                token_stage_audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, f"after_evaluation_{step}"))
                write_json(archive / "evaluations" / f"step-{step}.json", result)
            retention = retain_archives(archive, selected_milestones, latest=2)
            append_jsonl(telemetry, {"event": "checkpoint_durable", "step": step, "path": str(destination), "retention": retention, "native_retired_after_durable": removed_native, "at": time.time()})
            return control

    class FullWeightTrainer(FullWeightOptimizerTrainerMixin, Trainer):
        def get_train_dataloader(self): return deterministic_sequential_dataloader(self)
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
    phase_event(early,'trainer_resume_start',started,resume=str(resume))
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    phase_event(telemetry,'trainer_train_end',started,global_step=int(trainer.state.global_step))
    terminal_tokenizer = assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "after_training")
    terminal_step = int(trainer.state.global_step); expected_draws = ((terminal_step - source_step) * 16) - initial_cursor
    require(len(observed_positions) == expected_draws, "terminal observed draw count differs")
    terminal = archive / "full" / f"checkpoint-{terminal_step}"
    require(terminal.is_dir(), "terminal durable full checkpoint missing")
    phase_event(telemetry,'terminal_durable_verify_start',started,checkpoint=str(terminal));verify_checkpoint(terminal, identity(recipe), require_full=True, expected_checkpoint_kind="full_weights");phase_event(telemetry,'terminal_durable_verify_end',started,checkpoint=str(terminal))
    report = {"schema": "sepalith.sft11.full-weight-cpt-stage-transition-run.v2", "status": "schedule_complete" if terminal_step == runtime["max_steps"] else ("root_admitted_execution_stopped" if terminal_step == execution_stop["stop_at_global_step"] else "gracefully_stopped"), "global_step": terminal_step, "global_optimizer_step_offset": source_step, "stage_step": terminal_step-source_step, "initial_cursor": initial_cursor, "observed_draws": len(observed_positions), "last_draw_position": observed_positions[-1] if observed_positions else None, "train_loss": float(result.training_loss), "terminal_checkpoint": str(terminal), "tokenizer_contract": terminal_tokenizer, "tokenizer_stage_audits": token_stage_audits, "saved_precision_audit": saved_precision_audit, "in_process_preflight": run_preflight, "cohort_scope": recipe["cohort"]["scope"], "continuation_admission": continuation, "ordinary_canary_resume_lineage":resume_lineage, "execution_stop_admission": execution_stop}
    write_json(archive / "run-result.json", report); print(json.dumps(report, sort_keys=True)); return report


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("preflight-template", "preflight", "preflight-resume", "run")); p.add_argument("--recipe", type=Path, required=True); p.add_argument("--resume", type=Path); p.add_argument("--continuation-admission", type=Path); p.add_argument("--stage-transition-admission", type=Path); p.add_argument("--execution-stop-admission", type=Path); p.add_argument("--ordinary-canary-transition-admission", type=Path)
    a = p.parse_args()
    if a.command == "preflight-template": print(json.dumps(preflight_template(a.recipe), sort_keys=True)); return 0
    if a.command == "preflight": print(json.dumps(preflight(a.recipe), sort_keys=True)); return 0
    if a.command == "preflight-resume": require(a.resume is not None,'resume preflight requires checkpoint');print(json.dumps(preflight_resume(a.recipe,a.resume,a.continuation_admission,a.execution_stop_admission,a.ordinary_canary_transition_admission),sort_keys=True));return 0
    run(a.recipe, a.resume, a.continuation_admission, a.stage_transition_admission, a.execution_stop_admission,a.ordinary_canary_transition_admission); return 0


if __name__ == "__main__": raise SystemExit(main())
