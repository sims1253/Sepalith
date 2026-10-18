#!/usr/bin/env python3
"""Full-weight trainer for the C13 offline steps (night only, one CUDA device).

It follows ``sft.full_weight_edit_sft.run`` step for step and reuses its parts:
the unsloth bf16 full-weight load, the tokenizer contracts, the aurora_mix
optimizer mixin with its dispatch check and bf16 stochastic rounding, the
frozen sequential schedule with a cursor check, the mandatory gate stop,
``DeadlineStop`` and the recipe-bound save-and-stop file, sealed checkpoints
published atomically on /mnt/e, and retain-all retention. What differs:

- the recipe is a C13 bound recipe (``offline.recipe.load_offline``) with gates
  at the halves of the step's horizon;
- ``rft``: the RFT cohort through the SFT target-only collator and the SFT
  pre-update masking gate (it is SFT on verified samples);
- ``dpo``: one pair per micro-batch, ``offline.dpo.DPOLossMixin`` with the
  stored reference log-probabilities, the pair-batch gate before update 1, the
  reference-alignment gate at the first optimizer boundary, and one
  ``dpo_update`` telemetry event per update (loss, accuracy, margins, chosen and
  rejected log-ratios) that the night driver's DPO guard reads.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import time
from typing import Protocol, cast

from sepalith.training.offline.recipe import identity, load_offline
from sepalith.training.sft.full_weight_edit_sft import (
    EXPECTED_PARAMETER_TENSORS,
    EXPECTED_PARAMETERS,
    FrozenTokenRowDataset,
    append_jsonl,
    apply_retention,
    build_evaluation_request,
    enforce_mandatory_gate_boundary,
    publish_sealed_checkpoint,
    training_arguments_kwargs,
    write_json,
)
from sepalith.training.sft.recipe_v2 import Json, require, sha256

RUN_SCHEMA = "sepalith.c13.offline-run.v1"


class _State(Protocol):
    global_step: int


class _Control(Protocol):
    should_save: bool
    should_training_stop: bool


class _Args(Protocol):
    output_dir: str


def read_jsonl(path: Path) -> list[Json]:
    with path.open() as stream:
        return [cast(Json, json.loads(line)) for line in stream if line.strip()]


def load_pairs(recipe: Json, *, with_reference: bool = True) -> object:
    """The DPO pairs in schedule order with their stored reference log-probabilities."""
    from sepalith.training.offline.dpo import PairDataset, PairRows

    cohort = cast(Json, recipe["cohort"])
    rows_record, schedule_record = cast(Json, cohort["rows"]), cast(Json, cohort["draw_schedule"])
    pairs_path, schedule_path = Path(cast(str, rows_record["path"])), Path(cast(str, schedule_record["path"]))
    require(sha256(pairs_path) == rows_record["sha256"] and sha256(schedule_path) == schedule_record["sha256"], "DPO inputs differ")
    pairs = read_jsonl(pairs_path)
    plan = cast(Json, json.loads(schedule_path.read_text()))
    draws = cast(list[int], plan["draws"])
    require(len(pairs) == cohort["unique_rows"] and len(draws) == cast(int, cohort["updates"]) * 16, "DPO pair denominators differ")
    references = None
    if with_reference:
        reference = cast(Json, recipe["reference"])
        values = read_jsonl(Path(cast(str, reference["logps"])))
        require([v["pair_id"] for v in values] == [p["pair_id"] for p in pairs], "reference rows are not in pair order")
        references = [(float(cast(float, v["chosen_logp"])), float(cast(float, v["rejected_logp"]))) for v in values]
        require(all(math.isfinite(c) and math.isfinite(r) for c, r in references), "non-finite reference log-probability")
    return PairDataset(PairRows(pairs, draws, references))


def rft_dataset(recipe: Json) -> tuple[FrozenTokenRowDataset, Json]:
    """The RFT cohort through the SFT frozen-schedule dataset, with the length profile checked."""
    dataset = FrozenTokenRowDataset(recipe)
    cohort = cast(Json, recipe["cohort"])
    profile = cast(Json, json.loads(Path(cast(str, cast(Json, cohort["length_profile"])["path"])).read_text()))
    lengths = [len(row["input_ids"]) for row in dataset.rows]
    targets = [len(row["input_ids"]) - row["target_start"] for row in dataset.rows]
    sequence = cast(Json, profile["sequence_tokens"])
    supervised = cast(Json, profile["supervised_target_tokens_including_eos"])
    require(profile.get("rows_sha256") == cast(Json, cohort["rows"])["sha256"] and profile.get("rows") == len(dataset.rows)
            and sequence.get("sum") == sum(lengths) and sequence.get("max") == max(lengths)
            and supervised.get("sum") == sum(targets) and profile.get("observed_fit_without_truncation") is True,
            "RFT length profile differs")
    return dataset, {"status": "pass", "unique_rows": len(dataset.rows), "draws": len(dataset), "cuda_started": False}


def preflight(recipe_path: Path) -> Json:
    recipe = load_offline(recipe_path)
    if recipe["kind"] == "rft":
        return rft_dataset(recipe)[1]
    dataset = cast(Sequence[object], load_pairs(recipe))
    return {"status": "pass", "pairs": cast(Json, recipe["cohort"])["unique_rows"], "draws": len(dataset), "cuda_started": False}


def run(recipe_path: Path, resume: Path | None = None, stop_after_step: int | None = None) -> Json:
    from sepalith.training.eval.milestone_gate import continuation_plan
    from sepalith.training.sft.night_controls import DeadlineStop, deadline_from_env, pre_update_masking_gate

    recipe = load_offline(recipe_path)
    recipe_hash = sha256(recipe_path)
    kind = cast(str, recipe["kind"])
    ident = identity(recipe)
    night = DeadlineStop(deadline_from_env(), stop_after_step)
    gate_plan = continuation_plan(recipe, recipe_path, resume)
    runtime = cast(Json, recipe["runtime"])
    accumulation = cast(int, runtime["gradient_accumulation"])
    cohort = cast(Json, recipe["cohort"])
    train_context = cast(int, cohort["max_sequence_tokens"])
    if kind == "rft":
        rows_dataset, run_preflight = rft_dataset(recipe)
        dataset: object = rows_dataset
    else:
        dataset = load_pairs(recipe)
        run_preflight = {"status": "pass", "pairs": cohort["unique_rows"], "draws": len(cast(Sequence[object], dataset))}
    outputs = cast(Json, recipe["outputs"])
    output, archive = Path(cast(str, outputs["trainer"])), Path(cast(str, outputs["archive"]))
    if resume is None:
        require(not output.exists() and not archive.exists(), "fresh run output/archive must not exist")
        initial_cursor = 0
    else:
        from sepalith.training.checkpoint.campaign_checkpoint import verify_checkpoint

        require(resume.is_dir(), "resume checkpoint missing")
        prior = verify_checkpoint(resume, ident, require_full=True, expected_checkpoint_kind="full_weights")
        initial_cursor = prior["step"] * accumulation
    require(gate_plan["initial_step"] * accumulation == initial_cursor, "gate plan/resume cursor differs")
    next_stop = cast(int, gate_plan["next_mandatory_stop"])
    require(stop_after_step is None or gate_plan["initial_step"] < stop_after_step <= next_stop, "stop-after step lies outside this segment")
    stops: list[str] = []
    holder: dict[str, object] = {}
    output.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    telemetry = archive / "telemetry.jsonl"
    stop_path = Path(cast(str, outputs["graceful_stop"]))

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    from sepalith.training.checkpoint.campaign_checkpoint import seal_checkpoint, verify_checkpoint
    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        assert_serialized_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.offline.dpo import DPOLossMixin, collate_pair, pair_batch_gate, reference_alignment, update_summary
    from sepalith.training.optim.full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
    from sepalith.training.sft.campaign_edit_data import target_only_collator, verify_batch

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one root-owned CUDA device required")
    set_seed(cast(int, recipe["seed"]))
    parent = cast(Json, recipe["parent"])
    parent_path = cast(str, parent["path"])
    parent_files = cast(Json, parent["files"])
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=parent_path, max_seq_length=train_context, dtype=torch.bfloat16,
                                                         load_in_4bit=False, full_finetuning=True, float32_mixed_precision=False,
                                                         fast_inference=False, trust_remote_code=False, use_gradient_checkpointing=True)
    reference_tokenizer = load_pinned_reference_tokenizer(Path(parent_path))
    repair = restore_trainer_eog_alignment(model, tokenizer, reference_tokenizer)
    require(repair.get("repair", {}).get("vocab_mapping_unchanged") is True, "post-load token repair changed vocabulary")
    embeddings = embedding_identity(model)
    audits: list[object] = [{"stage": "post_load_repair", **repair},
                            assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embeddings, "post_load")]
    model.config.use_cache = False
    try:
        FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    except TypeError:
        FastLanguageModel.for_training(model)
    model.train()
    named = list(model.named_parameters())
    require(len(named) == EXPECTED_PARAMETER_TENSORS and sum(p.numel() for _, p in named) == EXPECTED_PARAMETERS
            and all(p.requires_grad for _, p in named), "full-weight trainable inventory differs")
    require(not any("lora_" in n.lower() or "modules_to_save" in n.lower() for n, _ in named), "PEFT parameter entered full-weight run")
    expected_dispatch = cast(Json, json.loads(Path(cast(str, cast(Json, recipe["optimizer_dispatch"])["path"])).read_text()))
    observed: list[int] = []

    def check_positions(positions: list[int]) -> None:
        if not observed:
            audits.append(assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embeddings, "first_compute_loss"))
        start = initial_cursor + len(observed)
        require(positions == list(range(start, start + len(positions))), "actual draw cursor differs from the frozen schedule")
        observed.extend(positions)

    def collate(items: list[Json]) -> dict[str, object]:
        if kind == "dpo":
            return cast(dict[str, object], collate_pair(items))
        positions = [cast(int, item.pop("_draw_position")) for item in items]
        rows = [dict(item) for item in items]
        batch = target_only_collator(rows, max_sequence_tokens=train_context)
        verify_batch(batch, rows)
        batch["_draw_position"] = torch.tensor(positions, dtype=torch.long)
        return batch

    def first_items() -> list[Json]:
        if kind == "dpo":
            return [cast(Json, cast(Sequence[object], dataset)[i]) for i in range(accumulation)]
        rows = cast(FrozenTokenRowDataset, dataset)
        return [rows.rows[rows.draws[i]] for i in range(accumulation)]

    class Callback(TrainerCallback):
        def on_train_begin(self, args: _Args, state: _State, control: _Control, **kwargs: object) -> object:
            again = restore_trainer_eog_alignment(model, tokenizer, reference_tokenizer)
            require(again.get("repair", {}).get("vocab_mapping_unchanged") is True, "train-begin token repair changed vocabulary")
            audits.append(assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embeddings, "train_begin"))
            if int(state.global_step) == 0:
                loader = kwargs.get("train_dataloader")
                if kind == "dpo":
                    gate = pair_batch_gate(holder["trainer"], loader, first_items(), accumulation=accumulation,  # ty: ignore[invalid-argument-type]
                                           global_step=0)
                    name = "pair-batch-gate.json"
                else:
                    gate = pre_update_masking_gate(holder["trainer"], loader, first_items(),  # ty: ignore[invalid-argument-type]
                                                   accumulation=accumulation, global_step=0)
                    name = "masking-gate.json"
                write_json(archive / name, gate)
                append_jsonl(telemetry, {"event": f"{name.removesuffix('.json')}_pass", "at": time.time()})
            return control

        def on_pre_optimizer_step(self, args: _Args, state: _State, control: _Control, **kwargs: object) -> object:
            step = int(state.global_step)
            finite = nonzero = 0
            for _, parameter in named:
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()), "missing or nonfinite full-weight gradient")
                finite += 1
                nonzero += int(bool(parameter.grad.detach().abs().max().item() > 0))
            if kind == "dpo" and step == 0 and gate_plan["initial_step"] == 0:
                trainer = cast(DPOLossMixin, holder["trainer"])
                alignment = reference_alignment(getattr(trainer, "dpo_records", []))
                write_json(archive / "reference-alignment.json", alignment)
                append_jsonl(telemetry, {"event": "reference_alignment_pass", "mean_abs_nats": alignment["mean_abs_nats"], "at": time.time()})
            append_jsonl(telemetry, {"event": "pre_optimizer", "next_step": step + 1, "finite_gradient_tensors": finite,
                                     "nonzero_gradient_tensors": nonzero, "allocated_bytes": int(torch.cuda.memory_allocated()),
                                     "reserved_bytes": int(torch.cuda.memory_reserved()),
                                     "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                     "max_reserved_bytes": int(torch.cuda.max_memory_reserved()), "at": time.time()})
            return control

        def on_log(self, args: _Args, state: _State, control: _Control, logs: dict[str, object] | None = None,
                   **kwargs: object) -> object:
            if logs and "loss" in logs:
                require(math.isfinite(float(cast(float, logs["loss"]))), "nonfinite logged loss")
            append_jsonl(telemetry, {"event": "log", "step": int(state.global_step), "logs": logs or {}, "at": time.time()})
            return control

        def on_step_end(self, args: _Args, state: _State, control: _Control, **kwargs: object) -> object:
            step = int(state.global_step)
            if kind == "dpo":
                trainer = holder["trainer"]
                records = getattr(trainer, "dpo_records", [])
                summary = update_summary(records)
                require(all(math.isfinite(float(cast(float, v))) for v in summary.values()), "nonfinite DPO statistics")
                append_jsonl(telemetry, {"event": "dpo_update", "step": step, **summary, "at": time.time()})
                records.clear()
            if enforce_mandatory_gate_boundary(step, next_stop, control):
                append_jsonl(telemetry, {"event": "mandatory_dev_gate_stop_at_optimizer_boundary", "step": step, "at": time.time()})
            reason = night.apply(step, control)
            if reason is not None:
                stops.append(reason)
                append_jsonl(telemetry, {"event": f"{reason}_save_at_optimizer_boundary", "step": step,
                                         "deadline_epoch": night.deadline_epoch, "at": time.time()})
            if stop_path.exists():
                request = json.loads(stop_path.read_text())
                require(request == {"action": "save_and_stop", "bound_recipe_sha256": recipe_hash}, "graceful stop request identity differs")
                control.should_save = True
                control.should_training_stop = True
                append_jsonl(telemetry, {"event": "graceful_stop_observed_at_optimizer_boundary", "step": step, "at": time.time()})
            return control

        def on_save(self, args: _Args, state: _State, control: _Control, **kwargs: object) -> object:
            step = int(state.global_step)
            source = Path(args.output_dir) / f"checkpoint-{step}"
            audits.append(assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embeddings, f"before_checkpoint_{step}"))
            serialized = assert_serialized_tokenizer(source, reference_tokenizer, parent_files["tokenizer.json"], sha256)
            sampler = {"method": "sequential_frozen_draw_schedule", "draw_schedule_sha256": cast(Json, cohort["draw_schedule"])["sha256"],
                       "cursor": step * accumulation, "global_step": step, "effective_batch": accumulation, "ignore_data_skip": False}
            sealed = seal_checkpoint(source, ident, step, full=True, sampler=sampler, checkpoint_kind="full_weights")
            destination = publish_sealed_checkpoint(source, archive / "full" / source.name, sealed)
            audits.append({"stage": f"serialized_checkpoint_{step}", **serialized})
            if step in set(cast(list[int], runtime["evaluation_steps"])):
                write_json(archive / "evaluations" / f"request-step-{step}.json", build_evaluation_request(recipe, step, destination))
            retention = apply_retention(recipe, archive)
            append_jsonl(telemetry, {"event": "checkpoint_durable", "step": step, "path": str(destination), "retention": retention,
                                     "at": time.time()})
            return control

    beta = float(cast(float, cast(Json, recipe["pre_registered"]).get("beta", 0.0)))

    def check_dispatch(manifest: dict[str, object] | None) -> None:
        require(manifest is not None, "optimizer dispatch manifest missing")
        assert manifest is not None
        for key in ("arm", "ordered_rows", "ordered_rows_sha256", "parameter_objects", "parameters", "counts"):
            require(manifest.get(key) == expected_dispatch.get(key), f"optimizer dispatch differs:{key}")

    class DPOTrainer(FullWeightOptimizerTrainerMixin, DPOLossMixin, Trainer):
        dpo_beta = beta

        def _get_train_sampler(self, train_dataset: object = None) -> object:
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        def dpo_positions(self, positions: list[int]) -> None:
            check_positions(positions)

        def create_optimizer(self) -> object:
            result = super().create_optimizer()
            check_dispatch(self.full_weight_optimizer_manifest)
            return result

    class RFTTrainer(FullWeightOptimizerTrainerMixin, Trainer):
        def _get_train_sampler(self, train_dataset: object = None) -> object:
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        def compute_loss(self, model: object, inputs: dict[str, object], *args: object, **kwargs: object) -> object:
            positions = [int(x) for x in cast("torch.Tensor", inputs.pop("_draw_position")).detach().cpu().tolist()]
            check_positions(positions)
            return super().compute_loss(model, inputs, *args, **kwargs)

        def create_optimizer(self) -> object:
            result = super().create_optimizer()
            check_dispatch(self.full_weight_optimizer_manifest)
            return result

    trainer_class = DPOTrainer if kind == "dpo" else RFTTrainer
    optimizer = json.loads(json.dumps(runtime["optimizer"]))
    config = OptimizerConfig(arm=optimizer.pop("arm"), **optimizer)
    args = TrainingArguments(**training_arguments_kwargs(recipe, output))
    trainer = trainer_class(model=model, args=args, train_dataset=dataset, data_collator=collate, processing_class=tokenizer,
                             callbacks=[Callback()])
    trainer.full_weight_optimizer_config = config
    holder["trainer"] = trainer
    again = restore_trainer_eog_alignment(model, tokenizer, reference_tokenizer)
    require(again.get("repair", {}).get("vocab_mapping_unchanged") is True, "post-Trainer token repair changed vocabulary")
    audits.append(assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embeddings, "post_trainer_construction"))
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    terminal_tokenizer = assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embeddings, "after_training")
    terminal_step = int(trainer.state.global_step)
    require(terminal_step <= next_stop, "training continued beyond the mandatory DEV gate")
    require(len(observed) == terminal_step * accumulation - initial_cursor, "terminal observed draw count differs")
    terminal = archive / "full" / f"checkpoint-{terminal_step}"
    require(terminal.is_dir(), "terminal durable full checkpoint missing")
    verify_checkpoint(terminal, ident, require_full=True, expected_checkpoint_kind="full_weights")
    status = "awaiting_mandatory_dev_gate" if terminal_step == next_stop else ("deadline_stopped" if stops else "gracefully_stopped")
    report: Json = {"schema": RUN_SCHEMA, "kind": kind, "status": status, "stop_reasons": stops, "global_step": terminal_step,
                    "initial_cursor": initial_cursor, "observed_draws": len(observed), "train_loss": float(result.training_loss),
                    "terminal_checkpoint": str(terminal), "tokenizer_contract": terminal_tokenizer, "tokenizer_stage_audits": audits,
                    "in_process_preflight": run_preflight, "gate_plan": gate_plan}
    write_json(archive / "run-result.json", report)
    print(json.dumps({k: report[k] for k in ("kind", "status", "global_step", "train_loss")}, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(args.recipe), sort_keys=True))
        return 0
    run(args.recipe, args.resume, args.stop_after_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
