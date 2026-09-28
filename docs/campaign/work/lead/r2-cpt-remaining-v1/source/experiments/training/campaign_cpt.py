#!/usr/bin/env python3
"""Minimal raw-R CPT runner built from the reviewed SFT training seams.

Import and ``--preflight-only`` are CPU/data checks. ``run`` is intentionally a
root-admitted CUDA path: the root wrapper must own the global CUDA lock, host
memory guard, deadline, and process group before this module is called.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import resource

from campaign_checkpoint import (check_identity, digest, checkpoint_callback,
                                 preserve_random_state, verify_checkpoint, write_json)
from campaign_control import control_callback, utc_deadline
from campaign_launch import supervised_command
from campaign_cpt_parent import check_merged_cpt_parent
from campaign_cpt_data import (CptDataError, causal_lm_collator, read_jsonl,
                               validate_draw_schedule, validate_materialized_rows,
                               validate_package_holdout, verify_causal_batch)
from campaign_sft import (TARGET_MODULES, assert_post_trainer_pinned_identity,
                          restore_trainer_eog_alignment, sequential_sft_trainer_class,
                          training_configuration_guard)
from campaign_tokenizer_contract import (EXPECTED_TOKENIZER_IDENTITY,
                                          NATIVE_EOG_IDS, TokenizerContractError,
                                          load_pinned_reference_tokenizer,
                                          restore_pinned_tokenizer_contract)

STAGE = "cpt_raw_r_v1"
LOSS_OBJECTIVE = "raw_causal_next_token_all_v1"
DEFAULT_GPU_USED_CAP_MIB = 6144
EXPECTED_TRAINABLE_PARAMETERS = 50_233_344
EXPECTED_ATTACHED_MODULES = 294


def _record(path_record: dict) -> Path:
    if not isinstance(path_record, dict) or set(path_record) != {"path", "sha256"}:
        raise ValueError("Input records must contain only absolute path and SHA256")
    return_path = Path(path_record["path"])
    if not return_path.is_absolute() or not return_path.is_file():
        raise ValueError(f"Expected an existing absolute input: {return_path}")
    if digest(return_path) != path_record["sha256"]:
        raise ValueError(f"Input hash differs: {return_path}")
    return return_path


def _load_rows(record: dict, *, max_sequence_tokens: int, materialized: bool,
               require_complete_documents: bool = True):
    path = _record(record)
    rows = read_jsonl(path, max_sequence_tokens=max_sequence_tokens, materialized=materialized,
                      require_complete_documents=require_complete_documents)
    summary = (validate_materialized_rows(rows, max_sequence_tokens=max_sequence_tokens,
                                          require_complete_documents=require_complete_documents)
               if materialized else None)
    # Keep the on-disk rows separate from their normalized training view.  The
    # former is needed for a second, materialized holdout validation; the latter
    # is the only shape admitted to the Dataset/collator boundary.
    return path, rows, summary


def _check_identity_contract(recipe: dict, params: dict) -> dict:
    if recipe.get("schema_version") != 1 or recipe.get("stage") != STAGE:
        raise ValueError("This entrypoint accepts only the new raw-R CPT stage")
    identity = check_identity(recipe["identity"])
    if identity["policy"].get("stage") != STAGE or identity["policy"].get("loss_objective") != LOSS_OBJECTIVE:
        raise ValueError("CPT identity does not declare the raw causal objective")
    initialization = identity["policy"].get("initialization")
    if initialization not in ("new_lora_on_midtrain", "new_lora_on_merged_cpt_parent"):
        raise ValueError("CPT initialization policy is not admitted")
    if initialization == "new_lora_on_merged_cpt_parent":
        check_merged_cpt_parent(recipe, verify_model_files=False)
    if identity["policy"].get("optimizer") != "adamw_torch_fused":
        raise ValueError("CPT optimizer must remain the reviewed fused AdamW path")
    if identity["policy"].get("dtype") != "bfloat16":
        raise ValueError("CPT dtype must be BF16")
    if identity["renderer"].get("id") != "raw-r-cpt-v1":
        raise ValueError("Raw CPT rows must carry the raw-r renderer identity")
    if identity["tokenizer"].get("BOS") != 0 or identity["tokenizer"].get("EOS_PAD") != 1:
        raise ValueError("CPT tokenizer boundary identity differs")
    if identity["tokenizer"].get("native_EOG") != list(NATIVE_EOG_IDS):
        raise ValueError("CPT native EOG identity differs")
    if identity["schedule"] != params:
        raise ValueError("CPT schedule identity differs from parameters")
    if params.get("max_sequence_tokens") != recipe.get("initial_context_tokens", 2048):
        raise ValueError("CPT context length differs from its explicit profile")
    if params["max_sequence_tokens"] != 2048:
        raise ValueError("This initial CPT profile is 2K; a 4K profile needs a new recipe identity")
    if params["learning_rate"] != 1e-4 or params["lora_rank"] != 32 or params["lora_alpha"] != 64:
        raise ValueError("CPT candidate changed the explicit LR/LoRA stage")
    if params["per_device_batch"] not in (1, 2) or params["per_device_batch"] * params["gradient_accumulation"] != 16:
        raise ValueError("CPT effective batch must be 16 with microbatch 1 or 2")
    if params["max_steps"] < 1:
        raise ValueError("CPT max_steps must be positive")
    return identity


def _check_checkpoint_contract(recipe: dict, params: dict) -> None:
    checkpoint = recipe["checkpoint"]
    if set(checkpoint) != {"light_every", "full_every", "evaluation_steps"}:
        raise ValueError("CPT checkpoint cadence fields are incomplete")
    light, full = checkpoint["light_every"], checkpoint["full_every"]
    if type(light) is not int or type(full) is not int or light < 1 or full < 1 or full % light:
        raise ValueError("CPT full checkpoint cadence must be a positive multiple")
    steps = checkpoint["evaluation_steps"]
    if not isinstance(steps, list) or not steps or len(set(steps)) != len(steps):
        raise ValueError("CPT evaluation steps must be a nonempty unique list")
    if any(type(step) is not int or step < 1 or step > params["max_steps"] for step in steps):
        raise ValueError("CPT evaluation step is outside max_steps")
    decisions = recipe.get("decision_steps", [])
    if any(type(step) is not int or step < 1 or step > params["max_steps"] for step in decisions):
        raise ValueError("CPT decision step is outside max_steps")
    if not set(decisions) <= set(steps):
        raise ValueError("Every CPT decision step must have a nominated holdout evaluation")
    if not math.isfinite(recipe["checkpoint_reserve_seconds"]) or recipe["checkpoint_reserve_seconds"] <= 0:
        raise ValueError("CPT checkpoint reserve must be finite and positive")


def preflight(recipe: dict, *, verify_model_files: bool = False) -> dict:
    """Validate a candidate recipe without importing CUDA/model/framework code."""
    params = recipe["parameters"]
    identity = _check_identity_contract(recipe, params)
    _check_checkpoint_contract(recipe, params)
    if not recipe.get("materialized_rows", True):
        raise ValueError("The admitted CPT profile consumes raw_cpt materialized rows")
    if not Path(recipe["model_path"]).is_absolute():
        raise ValueError("CPT model_path must be an absolute staged Midtrain path")
    utc_deadline(recipe["deadline"])
    supervised_command(recipe, ["cpt-preflight-does-not-execute"])
    if recipe.get("resource", {}).get("host_free_gib_min", 0) < 8:
        raise ValueError("CPT host admission must reserve at least 8 GiB free")
    if recipe.get("resource", {}).get("gpu_memory_used_cap_mib", 0) < DEFAULT_GPU_USED_CAP_MIB:
        raise ValueError("CPT GPU occupancy cap must allow the measured 6144 MiB baseline")
    for key in ("output_dir", "archive_dir"):
        destination = Path(recipe[key])
        if not destination.is_absolute() or destination.exists():
            raise ValueError(f"Use a fresh absolute {key} for every CPT attempt: {destination}")
    partial_train = recipe.get("exposure_remainder_rows") is True
    train_path, train_raw_rows, train_summary = _load_rows(
        recipe["train_rows"], max_sequence_tokens=params["max_sequence_tokens"], materialized=True,
        require_complete_documents=not partial_train)
    validation_path, validation_raw_rows, validation_summary = _load_rows(
        recipe["validation_rows"], max_sequence_tokens=params["max_sequence_tokens"], materialized=True)
    train_rows = train_summary["rows_checked"]
    validation_rows = validation_summary["rows_checked"]
    if any(row["cpt_partition"] != "cpt_train" for row in train_raw_rows):
        raise ValueError("CPT training input contains a validation partition")
    if any(row["cpt_partition"] != "cpt_validation" for row in validation_raw_rows):
        raise ValueError("CPT holdout input contains a training partition")
    if partial_train:
        train_packages = sorted({row["package_id"] for row in train_rows})
        validation_packages = sorted({row["package_id"] for row in validation_rows})
        if set(train_packages) & set(validation_packages):
            raise ValueError("package leakage between CPT train and holdout")
        train_docs = {row["source"]["document_sha256"] for row in train_rows}
        validation_docs = {row["source"]["document_sha256"] for row in validation_rows}
        if train_docs & validation_docs:
            raise ValueError("document leakage between CPT train and holdout")
        holdout = {"train_packages": train_packages,
                   "validation_packages": validation_packages}
    else:
        holdout = validate_package_holdout(train_raw_rows, validation_raw_rows,
                                           max_sequence_tokens=params["max_sequence_tokens"], materialized=True)
    data_identity = identity["data"]
    expected_hashes = {
        "train_rows_sha256": digest(train_path),
        "validation_rows_sha256": digest(validation_path),
        "draw_schedule_sha256": digest(_record(recipe["draw_schedule"])),
    }
    for key, value in expected_hashes.items():
        if data_identity.get(key) != value:
            raise ValueError(f"CPT data identity differs for {key}")
    if sorted(data_identity.get("train_package_ids", [])) != holdout["train_packages"]:
        raise ValueError("CPT train package identity differs")
    if sorted(data_identity.get("validation_package_ids", [])) != holdout["validation_packages"]:
        raise ValueError("CPT validation package identity differs")
    if data_identity.get("split_id") != recipe.get("split_id"):
        raise ValueError("CPT split identity differs")
    schedule_path = _record(recipe["draw_schedule"])
    schedule = json.loads(schedule_path.read_text())
    schedule_check = validate_draw_schedule(
        schedule, train_rows, token_rows_sha256=digest(train_path),
        max_steps=params["max_steps"], effective_batch=16,
    )
    if schedule_check["split_id"] != recipe["split_id"]:
        raise ValueError("CPT draw split differs from recipe")
    input_records = recipe.get("inputs", [])
    input_paths = {record.get("path") for record in input_records if isinstance(record, dict)}
    required_paths = {str(train_path), str(validation_path), str(schedule_path)}
    if not required_paths <= input_paths:
        raise ValueError("CPT data files must also be declared immutable recipe inputs")
    if verify_model_files:
        model_path = Path(recipe["model_path"])
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            candidate = model_path / name
            match = next((record for record in input_records if record.get("path") == str(candidate)), None)
            if match is None or _record(match) != candidate:
                raise ValueError(f"CPT model input is not pinned: {candidate}")
    if identity["policy"].get("initialization") == "new_lora_on_merged_cpt_parent":
        check_merged_cpt_parent(recipe, verify_model_files=verify_model_files)
    resume_manifest = None
    if recipe.get("resume_from"):
        # Verify the full state and exact identity before any model/framework
        # import.  A path or file-presence check alone cannot establish this.
        resume_manifest = verify_checkpoint(recipe["resume_from"], identity, require_full=True)
        if resume_manifest["step"] >= params["max_steps"]:
            raise ValueError("CPT resume checkpoint must precede the requested max_steps")
        binding = recipe.get("resume_binding")
        required = {"schema", "reason", "checkpoint_step", "checkpoint_manifest_sha256",
                    "schedule_sha256", "consumed_draws"}
        if not isinstance(binding, dict) or set(binding) != required:
            raise ValueError("CPT resume requires an exact interruption binding")
        if (binding["schema"] != "sepalith.cpt.full-cadence-resume.v1"
                or binding["reason"] != "external_interruption"):
            raise ValueError("CPT resume binding mode is not admitted")
        manifest_path = Path(recipe["resume_from"]) / "campaign-manifest.json"
        if digest(manifest_path) != binding["checkpoint_manifest_sha256"]:
            raise ValueError("CPT resume checkpoint manifest hash differs")
        step = resume_manifest["step"]
        if (binding["checkpoint_step"] != step
                or step % recipe["checkpoint"]["full_every"]):
            raise ValueError("CPT resume is not the pinned full-cadence checkpoint")
        state = json.loads((Path(recipe["resume_from"]) / "campaign-state.json").read_text())
        sampler = state.get("sampler", {})
        if (state.get("step") != step or state.get("identity") != identity
                or sampler.get("schedule_sha256") != binding["schedule_sha256"]
                or binding["schedule_sha256"] != recipe["draw_schedule"]["sha256"]
                or sampler.get("consumed_draws") != binding["consumed_draws"]
                or binding["consumed_draws"] != step * 16):
            raise ValueError("CPT resume identity/schedule/cursor differs")
    elif recipe.get("resume_binding") is not None:
        raise ValueError("Fresh CPT recipe cannot carry a resume binding")
    return {
        "identity": identity, "train_rows": train_rows, "validation_rows": validation_rows,
        "train_summary": {key: value for key, value in train_summary.items() if key != "rows_checked"},
        "validation_summary": {key: value for key, value in validation_summary.items() if key != "rows_checked"},
        "holdout": {key: value for key, value in holdout.items()
                    if key not in ("train_rows_checked", "validation_rows_checked")},
        "schedule": schedule_check, "schedule_raw": schedule,
        "resume_manifest": resume_manifest,
        "resume_step": None if resume_manifest is None else resume_manifest["step"],
    }


def run(recipe: dict) -> None:
    """Run only after root's CUDA lock/host guard and admitted deadlines."""
    soft = os.environ.get("SEPALITH_CAMPAIGN_SOFT_DEADLINE")
    hard = os.environ.get("SEPALITH_CAMPAIGN_HARD_DEADLINE")
    if not soft or not hard:
        raise ValueError("Launch through the root campaign supervisor")
    if not (utc_deadline(soft) < utc_deadline(hard) <= utc_deadline(recipe["deadline"])):
        raise ValueError("Supervisor deadlines differ from the CPT recipe")
    checked = preflight(recipe, verify_model_files=True)
    params = recipe["parameters"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    values = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(values) != 1 or float(values[0]) > recipe["resource"]["gpu_memory_used_cap_mib"]:
        raise RuntimeError("GPU occupancy exceeds the root-admitted CPT ceiling")
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTConfig
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CPT requires the root-leased single CUDA device")
    output, archive = Path(recipe["output_dir"]), Path(recipe["archive_dir"])
    output.mkdir(parents=True, exist_ok=False); archive.mkdir(parents=True, exist_ok=False)
    write_json(output / "admitted-recipe.json", recipe)
    write_json(output / "declared-token-exposure.json", {
        "schema": "sepalith.cpt.exposure.v1", "objective": LOSS_OBJECTIVE,
        "train": {key: value for key, value in checked["train_summary"].items() if key != "documents_detail"},
        "validation": {key: value for key, value in checked["validation_summary"].items() if key != "documents_detail"},
        "labels": "supplied explicit labels; BOS and overlap are masked; internal EOS is masked; terminal EOS is supervised",
    })
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=recipe["model_path"], max_seq_length=params["max_sequence_tokens"],
        dtype=torch.bfloat16, load_in_4bit=False, trust_remote_code=False,
        use_gradient_checkpointing=True,
    )
    try:
        reference_tokenizer = load_pinned_reference_tokenizer(Path(recipe["model_path"]))
        tokenizer_audit = restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=[],
        )
    except TokenizerContractError as error:
        raise ValueError(f"Loaded tokenizer failed the pinned MiniCPM contract: {error}") from error
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0, target_modules=TARGET_MODULES,
        bias="none", use_gradient_checkpointing=True, random_state=3407,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    attached = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if trainable != recipe.get("expected_trainable_parameters", EXPECTED_TRAINABLE_PARAMETERS):
        raise ValueError(f"MiniCPM attachment parameter count differs: {trainable}")
    if len(attached) != recipe.get("expected_attached_modules", EXPECTED_ATTACHED_MODULES):
        raise ValueError(f"MiniCPM attachment module count differs: {len(attached)}")
    write_json(output / "load-audit.json", {
        "trainable_parameters": trainable, "modules": attached,
        "dtype": str(next(model.parameters()).dtype), "tokenizer_contract": tokenizer_audit,
        "objective": LOSS_OBJECTIVE,
        "gradient_checkpointing_requested": True,
        "checkpoint_function": torch.utils.checkpoint.checkpoint.__name__,
    })
    rows = checked["train_rows"]
    row_index = {row["id"]: index for index, row in enumerate(rows)}
    draw_ids = checked["schedule"]["row_ids"]
    draw_indices = [row_index[ident] for ident in draw_ids]
    dataset_rows = [{key: row[key] for key in ("input_ids", "labels", "attention_mask")}
                    for row in rows]
    dataset = Dataset.from_list(dataset_rows).select(draw_indices)
    validation_rows = checked["validation_rows"]
    module_name, factory_name = recipe.get("evaluator_factory", "campaign_cpt_eval:package_holdout_evaluator").split(":")
    evaluator = getattr(importlib.import_module(module_name), factory_name)(recipe)
    if not callable(evaluator):
        raise ValueError("CPT evaluator factory did not return a callable")

    # Establish a step-zero causal-loss diagnostic on the exact frozen
    # package-held-out rows before Trainer can update the adapter.  Preserve
    # Python/NumPy/CPU-CUDA RNG and the Unsloth training configuration around
    # this read-only witness.  A resumed attempt records the verified source
    # step instead; it must not relabel a post-update model as step zero.
    baseline_path = output / "step-0-cpt-validation-baseline.json"
    if checked["resume_step"] is None:
        with preserve_random_state(model):
            with training_configuration_guard(model, FastLanguageModel.for_training):
                baseline = evaluator(model, tokenizer, "pre-update", 0)
        write_json(baseline_path, {
            "schema": "sepalith.cpt.step-zero-baseline.v1", "status": "verified",
            "step": 0, "objective": LOSS_OBJECTIVE,
            "evaluation": baseline,
            "random_state": "preserved_before_first_optimizer_update",
            "training_configuration": "guarded_and_restored",
            "quality_gate": "diagnostic_only_until_a_separate_SFT_stage",
        })
    else:
        write_json(baseline_path, {
            "schema": "sepalith.cpt.step-zero-baseline.v1", "status": "skipped_on_resume",
            "step": checked["resume_step"],
            "resume_from": recipe["resume_from"],
            "resume_checkpoint_step": checked["resume_step"],
            "reason": "A resumed adapter cannot be relabeled as a fresh step-zero baseline",
        })
    budget = control_callback(
        telemetry_path=output / "telemetry.jsonl", identity=recipe["identity"],
        deadline=soft, reserve_seconds=recipe["checkpoint_reserve_seconds"],
        stop_steps=recipe.get("decision_steps", []),
        resource_probe=lambda: {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_attempt_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_attempt_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_floor_gib": recipe["resource"]["host_free_gib_min"],
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    )
    trainer = None
    schedule = checked["schedule"]
    checkpoints = recipe["checkpoint"]
    hook = checkpoint_callback(
        identity=recipe["identity"], archive_root=archive, tokenizer=tokenizer,
        light_every=checkpoints["light_every"], full_every=checkpoints["full_every"],
        evaluator=evaluator, evaluation_allowed=budget.evaluation_allowed,
        evaluation_steps=checkpoints["evaluation_steps"],
        evaluation_context=lambda current_model: training_configuration_guard(
            current_model, FastLanguageModel.for_training,
        ),
        sampler_state=lambda: {
            "method": "declared raw CPT sequential draw schedule", "split_id": schedule["split_id"],
            "schedule_sha256": recipe["draw_schedule"]["sha256"],
            "consumed_draws": trainer.state.global_step * 16,
        },
    )
    trainer_class = sequential_sft_trainer_class()
    trainer = trainer_class(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        data_collator=lambda batch: causal_lm_collator(
            batch, max_sequence_tokens=params["max_sequence_tokens"],
        ), callbacks=[budget, hook],
        args=SFTConfig(
            output_dir=str(output / "trainer"), per_device_train_batch_size=params["per_device_batch"],
            gradient_accumulation_steps=params["gradient_accumulation"], max_steps=params["max_steps"],
            num_train_epochs=1, learning_rate=params["learning_rate"], warmup_ratio=0.03,
            lr_scheduler_type="cosine", seed=3407, data_seed=3407, bf16=True, fp16=False,
            optim="adamw_torch_fused", weight_decay=0.0, max_length=params["max_sequence_tokens"],
            packing=False, completion_only_loss=False, dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_num_workers=0, remove_unused_columns=False,
            save_strategy="steps", save_steps=checkpoints["full_every"], save_total_limit=2,
            save_only_model=False, ignore_data_skip=False, logging_steps=1, eval_strategy="no", report_to="none",
        ),
    )
    # Fresh-run witness: verify the actual Dataset/collator tensors before the
    # first optimizer step.  The expected rows deliberately use the minimal
    # Dataset shape; normalized provenance remains outside this boundary.
    if checked["resume_step"] is None:
        expected = [dataset_rows[index] for index in draw_indices[:params["per_device_batch"]]]
        with preserve_random_state(model):
            first_batch = next(iter(trainer.get_train_dataloader()))
        denominator = verify_causal_batch(first_batch, expected)
        write_json(output / "first-step-supervision-check.json", {
            "status": "verified", "global_step": 0,
            "rows": list(draw_ids[:params["per_device_batch"]]),
            "causal_loss_denominator": denominator,
            "policy": LOSS_OBJECTIVE,
        })
    else:
        write_json(output / "first-step-supervision-check.json", {
            "status": "skipped_on_resume", "global_step": checked["resume_step"],
            "resume_from": recipe["resume_from"],
            "reason": "Trainer state is loaded by train(); the verified full checkpoint manifest supplies the pre-load resume step.",
        })
    trainer_tokenizer = getattr(trainer, "processing_class", tokenizer)
    write_json(output / "post-trainer-contract.json", assert_post_trainer_pinned_identity(
        model, trainer_tokenizer, reference_tokenizer, [],
    ))
    from transformers import TrainerCallback

    class CptTrainBeginContract(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            audit = restore_trainer_eog_alignment(model, trainer_tokenizer, reference_tokenizer)
            write_json(output / "train-begin-tokenizer-contract.json", {"step": state.global_step, **audit})
            return control

    trainer.add_callback(CptTrainBeginContract())
    trainer.train(resume_from_checkpoint=recipe.get("resume_from"))
    write_json(output / "terminal-tokenizer-contract.json", assert_post_trainer_pinned_identity(
        model, trainer_tokenizer, reference_tokenizer, [],
    ))
    terminal = archive / "full" / f"checkpoint-{trainer.state.global_step}"
    verify_checkpoint(terminal, recipe["identity"], require_full=True)
    write_json(output / "terminal.json", {
        "step": trainer.state.global_step, "status": budget.stop_reason or "schedule_complete",
        "checkpoint": str(terminal), "identity": recipe["identity"],
        "resumed_from_step": checked["resume_step"],
        "scientific_acceptance": "CPT causal loss is diagnostic; SFT continuation and edit quality require a fresh later stage.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-model-files", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.preflight_only:
        checked = preflight(recipe, verify_model_files=args.verify_model_files)
        print(json.dumps({
            "status": "preflight_pass", "train_rows": checked["train_summary"]["rows"],
            "validation_rows": checked["validation_summary"]["rows"],
            "train_documents": checked["train_summary"]["documents"],
            "validation_documents": checked["validation_summary"]["documents"],
            "train_loss_tokens": checked["train_summary"]["loss_tokens"],
            "validation_loss_tokens": checked["validation_summary"]["loss_tokens"],
            "draws": checked["schedule"]["draws"], "CUDA_started": False,
        }))
    else:
        run(recipe)


if __name__ == "__main__":
    main()
