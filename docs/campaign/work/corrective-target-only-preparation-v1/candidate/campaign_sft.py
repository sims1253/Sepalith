#!/usr/bin/env python3
"""Explicit Tuesday SFT recipe; no launch or CUDA imports on import/preflight.

The campaign lead admits a reviewed recipe through the experiment runner. This
entry point uses the existing Unsloth/TRL stack and pre-tokenized full-text rows.
Each resumed attempt has new output/archive paths and the original schedule.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess

from campaign_checkpoint import check_identity, checkpoint_callback, verify_checkpoint, write_json
from campaign_control import control_callback, utc_deadline
from campaign_launch import supervised_command
from campaign_sft_data import (full_text_collator, target_only_collator, target_only_pilot_policy,
                               inspect_training_data, verified_file)
from campaign_tokenizer_contract import (  # noqa: E402
    EXPECTED_TOKENIZER_IDENTITY,
    NATIVE_EOG_IDS,
    TokenizerContractError,
    load_pinned_reference_tokenizer,
    restore_pinned_tokenizer_contract,
    verify_selected_prompt_parity,
)


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RUNTIME_CONFIG_FIELDS = ("use_cache", "bos_token_id", "eos_token_id", "pad_token_id")
RUNTIME_TOKENIZER_FIELDS = ("padding_side", "pad_token", "pad_token_id", "bos_token_id", "eos_token_id")


def _config_id_list(value):
    if type(value) is int:
        return (value,)
    if isinstance(value, (list, tuple)) and all(type(item) is int for item in value):
        return tuple(value)
    return None


def assert_post_trainer_pinned_identity(model, tokenizer, reference_tokenizer, prompt_rows):
    """Fail before training if SFTTrainer changed the pinned token contract."""
    identity = (
        len(tokenizer), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
    )
    if identity != EXPECTED_TOKENIZER_IDENTITY:
        raise ValueError(
            f"SFTTrainer changed tokenizer identity: {identity}; expected {EXPECTED_TOKENIZER_IDENTITY}"
        )
    loaded_vocab = tokenizer.get_vocab()
    reference_vocab = reference_tokenizer.get_vocab()
    if len(loaded_vocab) != len(reference_vocab) or loaded_vocab != reference_vocab:
        raise ValueError("SFTTrainer changed the pinned tokenizer vocabulary mapping")
    configs = {
        "model": getattr(model, "config", None),
        "generation": getattr(model, "generation_config", None),
    }
    config_audit = {}
    for name, config in configs.items():
        if config is None:
            raise ValueError(f"SFTTrainer removed the {name} configuration")
        bos = getattr(config, "bos_token_id", None)
        eos = _config_id_list(getattr(config, "eos_token_id", None))
        pad = getattr(config, "pad_token_id", None)
        if bos != 0 or eos != tuple(NATIVE_EOG_IDS) or pad != 1:
            raise ValueError(
                f"SFTTrainer changed {name} token identity: "
                f"bos={bos}, eos={eos}, pad={pad}"
            )
        config_audit[name] = {"bos_token_id": bos, "eos_token_id": list(eos), "pad_token_id": pad}
    parity = verify_selected_prompt_parity(tokenizer, reference_tokenizer, prompt_rows)
    return {
        "status": "verified",
        "tokenizer_identity": {
            "vocab_size": identity[0], "bos_token_id": identity[1],
            "eos_token_id": identity[2], "pad_token_id": identity[3],
        },
        "vocab_mapping_equal_reference": True,
        "configs": config_audit,
        "prompt_parity": parity,
    }


def restore_trainer_eog_alignment(model, tokenizer, reference_tokenizer):
    """Undo only the pinned HF Trainer's canonical-EOS alignment.

    Transformers 5.5 aligns tokens inside train(), after construction. It
    turns model EOS [1,130073] into 1 and prepends 1 to generation EOS.
    Restore the original native stop set before the first optimizer update.
    Any other EOS change remains a hard failure.
    """
    model_ids = _config_id_list(model.config.eos_token_id)
    generation_ids = _config_id_list(model.generation_config.eos_token_id)
    native = tuple(NATIVE_EOG_IDS)
    if model_ids not in (native, (1,)) or generation_ids not in (native, (1, *native)):
        raise ValueError("Unexpected Trainer EOS alignment; refusing contract repair")
    before = {"model_eos": list(model_ids), "generation_eos": list(generation_ids)}
    model.config.eos_token_id = list(native)
    model.generation_config.eos_token_id = list(native)
    repair = restore_pinned_tokenizer_contract(
        model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=[],
    )
    verified = assert_post_trainer_pinned_identity(model, tokenizer, reference_tokenizer, [])
    return {"before": before, "repair": repair, "verified": verified}


def clear_generation_markers(model):
    """Remove Unsloth generation flags from their direct owning modules.

    PEFT forwards missing attributes through its wrapped model.  The pinned
    Unsloth ``for_training`` cleanup checks ``hasattr`` and then deletes the
    marker on each wrapper; a forwarded marker can therefore make deletion
    target the wrong object and raise ``AttributeError``.  Removing only
    direct dictionary entries on every module leaves the supported cleanup
    free to restore modes without touching delegated attributes.
    """
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0
    cleared = 0
    for module in modules():
        namespace = getattr(module, "__dict__", None)
        if namespace is not None and namespace.pop("_flag_for_generation", None) is not None:
            cleared += 1
    return cleared


def capture_runtime_contract(model):
    """Capture mutable model/config state touched by inference-mode helpers."""
    configs, tokenizers = {}, {}
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return configs, tokenizers
    for module in modules():
        for attribute in ("config", "generation_config"):
            config = getattr(module, attribute, None)
            if config is not None:
                values = {
                    field: deepcopy(getattr(config, field))
                    for field in RUNTIME_CONFIG_FIELDS if hasattr(config, field)
                }
                if values:
                    configs[id(config)] = (config, values)
        tokenizer = getattr(module, "_saved_temp_tokenizer", None)
        if tokenizer is not None:
            values = {
                field: deepcopy(getattr(tokenizer, field))
                for field in RUNTIME_TOKENIZER_FIELDS if hasattr(tokenizer, field)
            }
            if values:
                tokenizers[id(tokenizer)] = (tokenizer, values)
    return configs, tokenizers


def restore_runtime_contract(configs, tokenizers):
    """Restore captured config/tokenizer fields without hiding cleanup errors."""
    for config, values in configs.values():
        for field, value in values.items():
            setattr(config, field, value)
    for tokenizer, values in tokenizers.values():
        for field, value in values.items():
            setattr(tokenizer, field, value)


@contextmanager
def training_configuration_guard(model, restore_mode):
    """Undo generation's Unsloth caches/mode changes, including on failure.

    generate() entered while model.eval() is active does not automatically
    call for_training(). Merely calling model.train() leaves checkpointing
    disabled and cached LoRA weights alive in the pinned Unsloth version.
    """
    checkpointing = []
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            checkpointing.append((module, module.gradient_checkpointing))
    configs, tokenizers = capture_runtime_contract(model)
    mode = next((value for _, value in checkpointing if value), False)
    try:
        yield
    finally:
        # This also removes generation's _fast_lora parameter caches and
        # _flag_for_generation markers. Use the library's supported cleanup.
        try:
            clear_generation_markers(model)
            restore_mode(model, use_gradient_checkpointing=mode)
        finally:
            for module, value in checkpointing:
                module.gradient_checkpointing = value
            restore_runtime_contract(configs, tokenizers)


def sequential_sft_trainer_class():
    """Use the identical draw order in production and the CPU resume check."""
    import torch
    from transformers import Trainer
    from trl import SFTTrainer

    class SequentialSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Use the model's fused loss without requiring training logits.

            The pinned Unsloth causal-LM forward deliberately returns its
            ``EMPTY_LOGITS`` sentinel for a labelled fused-CE call unless
            ``UNSLOTH_RETURN_LOGITS=1``.  TRL's SFTTrainer adds entropy and
            token-accuracy metrics by reading that sentinel.  The campaign
            objective is the model loss over the explicit collator labels;
            the development evaluator obtains its own full logits with a
            separate labels=None forward.  Calling the HF Trainer loss path
            preserves the model-returned fused loss and avoids the sentinel.
            """
            return Trainer.compute_loss(
                self, model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    return SequentialSFTTrainer


def preflight(recipe):
    if recipe["schema_version"] != 1:
        raise ValueError("Unsupported campaign SFT recipe version")
    identity = check_identity(recipe["identity"])
    params = recipe["parameters"]
    target_only = target_only_pilot_policy(recipe)
    if identity["schedule"] != params:
        raise ValueError("Checkpoint schedule identity differs from recipe parameters")
    if params["per_device_batch"] * params["gradient_accumulation"] != 16:
        raise ValueError("Primary SFT requires effective batch 16")
    if (params["learning_rate"] != (5e-5 if target_only else 2e-4)
            or params["lora_rank"] != 32 or params["lora_alpha"] != 64):
        raise ValueError("Primary recipe changed; record and implement a new admitted stage")
    if params["max_steps"] < 1 or params["max_sequence_tokens"] < 3:
        raise ValueError("Training schedule and context capacity must be positive")
    checkpoints = recipe["checkpoint"]
    if (checkpoints["light_every"] < 1 or checkpoints["full_every"] < 1
            or checkpoints["full_every"] % checkpoints["light_every"]):
        raise ValueError("Full checkpoint cadence must be a multiple of adapter cadence")
    evaluation_steps = checkpoints["evaluation_steps"]
    if (not 1 <= len(evaluation_steps) <= 4 or len(set(evaluation_steps)) != len(evaluation_steps)
            or any(type(step) is not int or not 1 <= step <= params["max_steps"] for step in evaluation_steps)):
        raise ValueError("Nominate one to four unique development checkpoints before launch")
    if not set(recipe["decision_steps"]) <= set(evaluation_steps) or params["max_steps"] not in evaluation_steps:
        raise ValueError("Every decision boundary and the final scheduled step require a development readout")
    if any(step % checkpoints["full_every"] and step not in recipe["decision_steps"]
           and step != params["max_steps"] for step in evaluation_steps):
        raise ValueError("Nominated readouts require scheduled or decision-boundary full checkpoints")
    if not math.isfinite(recipe["checkpoint_reserve_seconds"]) or recipe["checkpoint_reserve_seconds"] <= 0:
        raise ValueError("A finite positive checkpoint reserve is required")
    utc_deadline(recipe["deadline"])
    supervised_command(recipe, ["preflight-does-not-execute"])
    for record in recipe["inputs"]:
        verified_file(record)
    required_inputs = {str(verified_file(recipe[key])) for key in ("token_rows", "draw_schedule", "development_panel")}
    if not required_inputs <= {record["path"] for record in recipe["inputs"]}:
        raise ValueError("Data files must also be declared as immutable recipe inputs")
    if not Path(recipe["model_path"]).is_absolute():
        raise ValueError("Use the staged, pinned local model path")
    model_files = {str(Path(recipe["model_path"]) / name) for name in
                   ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")}
    if not model_files <= {record["path"] for record in recipe["inputs"]}:
        raise ValueError("Pinned model/config/tokenizer files must be declared recipe inputs")
    if not recipe.get("evaluator_factory"):
        raise ValueError("A development evaluator is required, including during the resume smoke")
    for key in ("output_dir", "archive_dir"):
        path = Path(recipe[key])
        if not path.is_absolute() or path.exists():
            raise ValueError(f"Use a fresh absolute {key} for every attempt: {path}")
    if recipe.get("resume_from"):
        restored = verify_checkpoint(recipe["resume_from"], identity, require_full=True)
        if target_only and restored["step"] != 25:
            raise ValueError("Target-only pilot resumes only its matched full25 state")
    return inspect_training_data(
        recipe["token_rows"], recipe["draw_schedule"], renderer_id=recipe["renderer_id"],
        max_sequence_tokens=params["max_sequence_tokens"], max_steps=params["max_steps"],
        effective_batch=16,
    )


def run(recipe):
    supervised_deadline = os.environ.get("SEPALITH_CAMPAIGN_SOFT_DEADLINE")
    hard_deadline = os.environ.get("SEPALITH_CAMPAIGN_HARD_DEADLINE")
    if not supervised_deadline or not hard_deadline:
        raise ValueError("Launch through campaign_launch.py under the existing experiment runner")
    if not (utc_deadline(supervised_deadline) < utc_deadline(hard_deadline) <= utc_deadline(recipe["deadline"])):
        raise ValueError("Supervisor deadlines differ from the admitted recipe")
    rows, draws, schedule, exposure = preflight(recipe)
    if datetime.now(timezone.utc).timestamp() + recipe["checkpoint_reserve_seconds"] >= utc_deadline(supervised_deadline):
        raise ValueError("The training deadline does not leave time for a checkpoint")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    if len(gpu.stdout.splitlines()) != 1 or float(gpu.stdout.strip()) > 4096:
        raise RuntimeError("GPU occupancy does not match the lead-admitted single-device lane")
    # Preserve the tested import order. No changes to the active owner's source.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTConfig

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Primary SFT requires the single leased CUDA device")
    params, checkpoints = recipe["parameters"], recipe["checkpoint"]
    target_only = target_only_pilot_policy(recipe)
    output, archive = Path(recipe["output_dir"]), Path(recipe["archive_dir"])
    output.mkdir(parents=True)
    archive.mkdir(parents=True)
    write_json(output / "admitted-recipe.json", recipe)
    if target_only:
        exposure = [{**item, "masked_prompt_prediction_tokens": item["prompt_loss_tokens"],
                     "prompt_loss_tokens": 0, "loss_objective": "exact_target_and_eos_v1"}
                    for item in exposure]
    write_json(output / "declared-token-exposure.json", exposure)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=recipe["model_path"], max_seq_length=params["max_sequence_tokens"],
        dtype=torch.bfloat16, load_in_4bit=False, trust_remote_code=False,
    )
    try:
        reference_tokenizer = load_pinned_reference_tokenizer(Path(recipe["model_path"]))
        tokenizer_audit = restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=rows,
        )
    except TokenizerContractError as error:
        raise ValueError(f"Loaded tokenizer failed the pinned MiniCPM contract: {error}") from error
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0, target_modules=TARGET_MODULES,
        bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attached = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if trainable != 50_233_344 or len(attached) != 294:
        raise ValueError(f"MiniCPM attachment mismatch: {trainable} parameters, {len(attached)} modules")
    write_json(output / "load-audit.json", {"trainable_parameters": trainable, "modules": attached,
                                          "dtype": str(next(model.parameters()).dtype),
                                          "tokenizer_contract": tokenizer_audit})
    dataset_fields = (["input_ids", "target_start", "target_body_tokens", "target_terminal_tokens"]
                      if target_only else ["input_ids"])
    dataset = Dataset.from_list([{key: row[key] for key in dataset_fields} for row in rows]).select(draws)
    del draws
    module_name, factory_name = recipe["evaluator_factory"].split(":")
    evaluator = getattr(importlib.import_module(module_name), factory_name)(recipe)
    if not callable(evaluator):
        raise ValueError("Development evaluator factory did not return a callable")
    budget = control_callback(
        telemetry_path=output / "telemetry.jsonl", identity=recipe["identity"],
        deadline=supervised_deadline, reserve_seconds=recipe["checkpoint_reserve_seconds"],
        stop_steps=sorted(set(recipe["decision_steps"]) | ({50} if target_only else set())),
        resource_probe=lambda: {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_attempt_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_attempt_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_scope": "process lifetime including model load, optimizer, training and completed evaluations",
        },
    )
    hook = checkpoint_callback(
        identity=recipe["identity"], archive_root=archive, tokenizer=tokenizer,
        light_every=checkpoints["light_every"], full_every=checkpoints["full_every"],
        evaluator=evaluator, evaluation_allowed=budget.evaluation_allowed,
        evaluation_steps=checkpoints["evaluation_steps"],
        evaluation_context=lambda model: training_configuration_guard(model, FastLanguageModel.for_training),
        sampler_state=lambda: {"method": "frozen sequential draw schedule", "split_id": schedule["split_id"],
                               "schedule_sha256": recipe["draw_schedule"]["sha256"],
                               "consumed_draws": trainer.state.global_step * 16},
    )

    trainer = sequential_sft_trainer_class()(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        data_collator=target_only_collator if target_only else full_text_collator, callbacks=[budget, hook],
        args=SFTConfig(
            output_dir=str(output / "trainer"), per_device_train_batch_size=params["per_device_batch"],
            gradient_accumulation_steps=params["gradient_accumulation"], max_steps=params["max_steps"],
            num_train_epochs=1, learning_rate=params["learning_rate"], warmup_ratio=0.03, lr_scheduler_type="cosine",
            seed=3407, data_seed=3407, bf16=True, fp16=False, optim="adamw_torch_fused", weight_decay=0.0,
            max_length=params["max_sequence_tokens"], packing=False, completion_only_loss=False,
            dataset_kwargs={"skip_prepare_dataset": True}, dataloader_num_workers=0,
            remove_unused_columns=not target_only,
            save_strategy="steps", save_steps=checkpoints["full_every"], save_total_limit=2,
            save_only_model=False, ignore_data_skip=False, logging_steps=20, eval_strategy="no",
            report_to="none",
        ),
    )
    trainer_tokenizer = getattr(trainer, "processing_class", tokenizer)
    post_trainer_contract = assert_post_trainer_pinned_identity(
        model, trainer_tokenizer, reference_tokenizer, rows,
    )
    write_json(output / "post-trainer-contract.json", post_trainer_contract)
    from transformers import TrainerCallback

    class PinnedTrainBeginContract(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            audit = restore_trainer_eog_alignment(model, trainer_tokenizer, reference_tokenizer)
            write_json(output / "train-begin-tokenizer-contract.json", {
                "step": state.global_step, **audit,
            })
            return control

    trainer.add_callback(PinnedTrainBeginContract())
    del rows
    trainer.train(resume_from_checkpoint=recipe.get("resume_from"))
    write_json(output / "terminal-tokenizer-contract.json", assert_post_trainer_pinned_identity(
        model, trainer_tokenizer, reference_tokenizer, [],
    ))
    terminal = archive / "full" / f"checkpoint-{trainer.state.global_step}"
    verify_checkpoint(terminal, recipe["identity"], require_full=True)
    write_json(output / "terminal.json", {
        "step": trainer.state.global_step, "status": budget.stop_reason or "schedule_complete",
        "checkpoint": str(terminal), "identity": recipe["identity"],
        "scientific_acceptance": "Lead must review development evidence; command exit is not promotion",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.preflight_only:
        rows, draws, schedule, exposure = preflight(recipe)
        print(json.dumps({"status": "preflight_pass", "rows": len(rows), "draws": len(draws),
                          "updates": len(exposure), "split_id": schedule["split_id"], "CUDA_started": False}))
    else:
        run(recipe)


if __name__ == "__main__":
    main()
