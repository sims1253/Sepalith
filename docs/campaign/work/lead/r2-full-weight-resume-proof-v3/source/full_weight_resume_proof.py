#!/usr/bin/env python3
"""Two-step real Trainer interruption/resume proof for full-weight CPT.

Preparation and preflight are CPU metadata-only.  Root owns each live lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "packages/sepalith/src"))
EXPECTED_PARAMETERS = 2_516_756_480
EXPECTED_PARAMETER_TENSORS = 381
FULL_KIND = "full_weights"


def fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    fail(recipe.get("schema") == "sepalith.sft11.full-weight-resume-proof.v2", "recipe schema differs")
    fail(recipe.get("steps") == {"interrupted": 1, "terminal": 2}, "proof step geometry differs")
    fail(recipe.get("batch") == {"per_device": 1, "gradient_accumulation": 1, "sampler": "sequential"}, "proof batch geometry differs")
    fail(recipe.get("checkpoint_kind") == FULL_KIND, "proof checkpoint kind differs")
    fail(recipe.get("expected_parameters") == EXPECTED_PARAMETERS, "parameter count binding differs")
    fail(recipe.get("expected_parameter_tensors") == EXPECTED_PARAMETER_TENSORS, "parameter tensor binding differs")
    fail(recipe.get("optimizer", {}).get("arm") == "aurora_mix", "proof optimizer arm differs")
    fail(recipe.get("scheduler") == {"type": "cosine", "horizon_steps": 2, "warmup_steps": 0}, "proof scheduler binding differs")
    for section in ("source", "model", "data", "optimizer"):
        fail(isinstance(recipe.get(section), Mapping), f"recipe {section} binding missing")
    source = recipe["source"]
    source_manifest = Path(source["manifest_path"])
    fail(source_manifest.is_file() and sha256(source_manifest) == source["manifest_sha256"], "source manifest binding differs")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    fail(manifest.get("schema") == "sepalith.full-weight-resume-proof-source.v2", "source manifest schema differs")
    for item in manifest.get("files", []):
        candidate = source_manifest.parent / item["path"]
        fail(candidate.is_file() and candidate.stat().st_size == item["bytes"] and sha256(candidate) == item["sha256"], f"source differs:{item['path']}")
    for binding_name in ("model", "data"):
        binding = recipe[binding_name]
        target = Path(binding["path"])
        fail(target.exists(), f"{binding_name} path missing")
    data = recipe["data"]
    fail(sha256(Path(data["path"])) == data["sha256"], "data bytes differ")
    rows = [json.loads(line) for line in Path(data["path"]).read_text(encoding="utf-8").splitlines() if line]
    fail(len(rows) == 8 and [row["row_id"] for row in rows] == data["row_ids"], "proof row identity/order differs")
    for row in rows:
        fail(len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"]) == 2048, "proof row geometry differs")
        fail(any(value != -100 for value in row["labels"]), "proof row has no labels")
    model = recipe["model"]
    model_root = Path(model["path"])
    for name, expected in model["files"].items():
        candidate = model_root / name
        fail(candidate.is_file() and sha256(candidate) == expected, f"model file differs:{name}")
    optimizer = recipe["optimizer"]
    optimizer_source = Path(optimizer["source_path"])
    expected_dispatch = Path(optimizer["expected_dispatch_path"])
    fail(sha256(optimizer_source) == optimizer["source_sha256"], "optimizer source differs")
    fail(sha256(expected_dispatch) == optimizer["expected_dispatch_sha256"], "optimizer dispatch differs")
    return recipe


def identity(recipe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent": recipe["model"],
        "tokenizer": {"tokenizer_json_sha256": recipe["model"]["files"]["tokenizer.json"]},
        "renderer": {"kind": "pretokenized_cpt_rows", "version": 1},
        "data": {"sha256": recipe["data"]["sha256"], "rows": 8},
        "source": {"manifest_sha256": recipe["source"]["manifest_sha256"]},
        "policy": {"objective": "causal_lm", "checkpoint_kind": FULL_KIND, "optimizer": "aurora_mix"},
        "schedule": {"max_steps": 2, "batch": 1, "gradient_accumulation": 1, "sampler": "sequential"},
    }


def tensor_mapping_digest(mapping: Mapping[Any, Any]) -> str:
    """Digest nested state without retaining additional full-sized copies."""
    hasher = hashlib.sha256()

    def visit(label: str, value: Any) -> None:
        if hasattr(value, "detach") and hasattr(value, "shape"):
            detached = value.detach().contiguous()
            hasher.update(b"tensor\0" + label.encode("utf-8") + b"\0")
            hasher.update(str(detached.dtype).encode("ascii") + b"\0")
            hasher.update(canonical(list(detached.shape)) + b"\0")
            # Byte view supports BF16, unlike direct NumPy conversion.
            hasher.update(detached.view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes())
        elif isinstance(value, Mapping):
            hasher.update(b"mapping\0" + label.encode("utf-8") + b"\0")
            for key in sorted(value, key=lambda item: str(item)):
                visit(f"{label}/{key}", value[key])
        elif isinstance(value, (list, tuple)):
            hasher.update(b"sequence\0" + label.encode("utf-8") + b"\0")
            for index, item in enumerate(value):
                visit(f"{label}/{index}", item)
        else:
            hasher.update(b"scalar\0" + label.encode("utf-8") + b"\0" + canonical(value) + b"\0")

    visit("root", mapping)
    return hasher.hexdigest()


def training_arguments_kwargs(output_dir: Path, seed: int) -> dict[str, Any]:
    """One shared two-step horizon for uninterrupted, stopped, and resumed lanes."""
    return {
        "output_dir": str(output_dir), "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1, "max_steps": 2,
        "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_steps": 0,
        "max_grad_norm": 0.0, "bf16": True, "use_cache": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "steps", "save_steps": 1, "save_only_model": False,
        "logging_strategy": "steps", "logging_steps": 1, "logging_first_step": True,
        "eval_strategy": "no", "report_to": "none", "disable_tqdm": True,
        "remove_unused_columns": False, "dataloader_num_workers": 0,
        "dataloader_pin_memory": False, "train_sampling_strategy": "sequential",
        "ignore_data_skip": False, "seed": seed, "data_seed": seed,
    }


def run_lane(recipe_path: Path, lane: str) -> dict[str, Any]:
    recipe = load_recipe(recipe_path)
    output_root = Path(recipe["output_root"])
    lane_root = output_root / lane
    fail(not lane_root.exists(), f"lane output must be fresh:{lane_root}")
    terminal_step = 1 if lane == "interrupted" else 2
    resume = None
    if lane == "resumed":
        resume = output_root / "interrupted" / "checkpoint-1"
        fail(resume.is_dir(), "interrupted checkpoint-1 missing")
        sys.path.insert(0, str(HERE))
        from campaign_checkpoint import verify_checkpoint
        verify_checkpoint(resume, identity(recipe), require_full=True, expected_checkpoint_kind=FULL_KIND)

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
    from campaign_checkpoint import seal_checkpoint, verify_checkpoint
    from campaign_tokenizer_contract import load_pinned_reference_tokenizer, restore_pinned_tokenizer_contract

    fail(torch.cuda.is_available() and torch.cuda.device_count() == 1, "proof requires exactly one root-owned CUDA device")
    seed = int(recipe["seed"])
    set_seed(seed)
    rows = [json.loads(line) for line in Path(recipe["data"]["path"]).read_text(encoding="utf-8").splitlines() if line]

    class ProofDataset(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            item = {name: row[name] for name in ("input_ids", "attention_mask", "labels")}
            item["_proof_position"] = index
            return item

    class SealFullCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            sampler = {
                "method": "torch.utils.data.SequentialSampler",
                "dataset_sha256": recipe["data"]["sha256"],
                "dataset_rows": 8,
                "effective_batch": 1,
                "cursor": int(state.global_step),
                "next_row_position": int(state.global_step),
                "ignore_data_skip": False,
            }
            seal_checkpoint(checkpoint, identity(recipe), int(state.global_step), full=True, sampler=sampler, checkpoint_kind=FULL_KIND)
            verify_checkpoint(checkpoint, identity(recipe), require_full=True, expected_checkpoint_kind=FULL_KIND)
            return control

    class StopAfterFirstSavedUpdate(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if lane == "interrupted" and int(state.global_step) == 1:
                control.should_save = True
                control.should_training_stop = True
            return control

    expected_dispatch = json.loads(Path(recipe["optimizer"]["expected_dispatch_path"]).read_text(encoding="utf-8"))

    class ProofTrainer(FullWeightOptimizerTrainerMixin, Trainer):
        updated_positions: list[int]

        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        def compute_loss(self, model, inputs, *args, **kwargs):
            positions = inputs.pop("_proof_position")
            self.updated_positions.extend(int(value) for value in positions.detach().cpu().tolist())
            return super().compute_loss(model, inputs, *args, **kwargs)

        def create_optimizer(self):
            result = super().create_optimizer()
            for key in ("arm", "ordered_rows", "ordered_rows_sha256", "parameter_objects", "parameters", "counts"):
                fail(self.full_weight_optimizer_manifest.get(key) == expected_dispatch.get(key), f"actual 381-tensor optimizer dispatch differs:{key}")
            return result

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(recipe["model"]["path"]), max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=True,
        float32_mixed_precision=False, fast_inference=False,
        trust_remote_code=False, use_gradient_checkpointing=True,
    )
    reference_tokenizer = load_pinned_reference_tokenizer(Path(recipe["model"]["path"]))
    tokenizer_audit = restore_pinned_tokenizer_contract(
        model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=(),
    )
    fail(tokenizer_audit.get("status") == "verified", "post-load tokenizer repair was not verified")
    fail(tokenizer_audit.get("vocab_mapping_unchanged") is True, "post-load tokenizer vocabulary changed")
    fail(tokenizer_audit.get("after", {}).get("bos_token_id") == 0, "post-load BOS differs")
    fail(tokenizer_audit.get("after", {}).get("eos_token_id") == 1, "post-load EOS differs")
    fail(tokenizer_audit.get("after", {}).get("pad_token_id") == 1, "post-load PAD differs")
    model.config.use_cache = False
    try:
        FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    except TypeError:
        FastLanguageModel.for_training(model)
    model.train()
    named = list(model.named_parameters())
    fail(len(named) == EXPECTED_PARAMETER_TENSORS, "trainable tensor count differs")
    fail(sum(parameter.numel() for _, parameter in named) == EXPECTED_PARAMETERS, "parameter count differs")
    fail(all(parameter.requires_grad for _, parameter in named), "full-weight proof contains frozen parameters")
    fail(not any("lora_" in name.lower() or "modules_to_save" in name.lower() for name, _ in named), "PEFT parameter entered proof")

    optimizer_config = OptimizerConfig(**recipe["optimizer"]["config"])
    argument_values = training_arguments_kwargs(lane_root, seed)
    # The optimizer owns different hidden/side rates. TrainingArguments must
    # expose the hidden rate for scheduler construction without changing the
    # optimizer's per-group rates.
    argument_values["learning_rate"] = optimizer_config.hidden_lr
    args = TrainingArguments(**argument_values)
    trainer = ProofTrainer(
        model=model, args=args, train_dataset=ProofDataset(), processing_class=tokenizer,
        callbacks=[StopAfterFirstSavedUpdate(), SealFullCheckpoint()],
    )
    trainer.updated_positions = []
    trainer.full_weight_optimizer_config = optimizer_config
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    fail(int(trainer.state.global_step) == terminal_step, "Trainer terminal step differs")
    expected_lane_positions = [1] if lane == "resumed" else list(range(terminal_step))
    fail(trainer.updated_positions == expected_lane_positions, "actual sequential TRAIN draw positions differ")
    terminal = lane_root / f"checkpoint-{terminal_step}"
    manifest = verify_checkpoint(terminal, identity(recipe), require_full=True, expected_checkpoint_kind=FULL_KIND)
    report = {
        "schema": "sepalith.sft11.full-weight-resume-lane.v2",
        "lane": lane, "global_step": int(trainer.state.global_step),
        "resume_from": str(resume) if resume else None,
        "checkpoint": str(terminal), "checkpoint_manifest": manifest,
        "model_state_sha256": tensor_mapping_digest(model.state_dict()),
        "optimizer_state_sha256": tensor_mapping_digest(trainer.optimizer.state_dict()),
        "scheduler_state_sha256": hashlib.sha256(canonical(trainer.lr_scheduler.state_dict())).hexdigest(),
        "cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "cuda_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "draw_cursor": {"consumed_updates": terminal_step, "next_row_position": terminal_step, "row_ids": recipe["data"]["row_ids"][:terminal_step]},
        "lane_observed_positions": trainer.updated_positions,
        "train_loss": float(result.training_loss),
        "optimizer_dispatch_sha256": hashlib.sha256(canonical(trainer.full_weight_optimizer_manifest)).hexdigest(),
        "tokenizer_contract": tokenizer_audit,
    }
    write_json(lane_root / "proof-lane.json", report)
    return report


def compare(recipe_path: Path) -> dict[str, Any]:
    recipe = load_recipe(recipe_path)
    root = Path(recipe["output_root"])
    direct = json.loads((root / "uninterrupted" / "proof-lane.json").read_text(encoding="utf-8"))
    resumed = json.loads((root / "resumed" / "proof-lane.json").read_text(encoding="utf-8"))
    fields = (
        "global_step", "model_state_sha256", "optimizer_state_sha256",
        "scheduler_state_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
        "draw_cursor", "optimizer_dispatch_sha256", "tokenizer_contract",
    )
    comparisons = {field: direct[field] == resumed[field] for field in fields}
    fail(all(comparisons.values()), "interrupted/resumed terminal state differs from uninterrupted state")
    result = {
        "schema": "sepalith.sft11.full-weight-resume-proof.v2",
        "status": "exact_two_step_resume_proved", "comparisons": comparisons,
        "uninterrupted_report_sha256": sha256(root / "uninterrupted" / "proof-lane.json"),
        "resumed_report_sha256": sha256(root / "resumed" / "proof-lane.json"),
        "limitations": "Two updates over fixed TRAIN rows prove mechanics only; this is not a long-training recipe or quality result.",
    }
    write_json(root / "proof-result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run-lane", "compare"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--lane", choices=("uninterrupted", "interrupted", "resumed"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        recipe = load_recipe(args.recipe)
        print(json.dumps({"status": "pass", "rows": 8, "steps": recipe["steps"], "cuda_started": False}, sort_keys=True))
        return 0
    if args.command == "run-lane":
        fail(args.lane is not None, "--lane is required")
        print(json.dumps(run_lane(args.recipe, args.lane), sort_keys=True))
        return 0
    print(json.dumps(compare(args.recipe), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
