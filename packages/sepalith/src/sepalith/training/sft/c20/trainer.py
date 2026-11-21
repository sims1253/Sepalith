"""Train the C20 mixed cohort while reusing the C04 trainer contracts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from torch import Tensor

from sepalith.training.sft.full_weight_edit_sft import (
    EXPECTED_PARAMETERS,
    EXPECTED_PARAMETER_TENSORS,
    FrozenTokenRowDataset,
    append_jsonl,
    apply_retention,
    build_evaluation_request,
    enforce_mandatory_gate_boundary,
    identity_v2,
    publish_sealed_checkpoint,
    require,
    sha256,
    training_arguments_kwargs,
    write_json,
)
from sepalith.training.sft.c20 import data, recipe as config
from sepalith.training.sft.c20.evaluator import limit_cuda
from sepalith.training.sft.c20.night_controls import (
    DeadlineStop,
    pre_update_masking_gate,
)
from sepalith.training.sft.c20.rows import inspect, target_only_collator
from sepalith.training.sft.campaign_edit_data import verify_batch
from sepalith.training.sft.night_controls import deadline_from_env


def load_bound(path: Path) -> dict:
    return config.verify(data.read(path))


def identity(recipe: dict) -> dict:
    value = identity_v2(recipe)
    value["policy"]["stage"] = "c20_sft_finish"
    value["policy"]["validation_manifest_sha256"] = recipe["validation_manifest"]["sha256"]
    value["policy"]["cpt_text_replay"] = True
    value["policy"]["optimizer_state_offload"] = recipe["runtime"]["optimizer_state_offload"]
    value["policy"]["cuda_memory_limit_mib"] = recipe["runtime"]["cuda_memory_limit_mib"]
    return value


class C20Dataset(FrozenTokenRowDataset):
    def __init__(self, recipe: dict) -> None:
        rows, draws, schedule, totals = inspect(
            recipe["cohort"]["rows"],
            recipe["cohort"]["draw_schedule"],
            recipe["cohort"]["max_sequence_tokens"],
            16,
        )
        require(
            len(rows) == recipe["cohort"]["unique_rows"] and len(draws) == recipe["cohort"]["updates"] * 16,
            "editing cohort denominator differs",
        )
        require(
            totals == {k: recipe["cohort"][k] for k in ("input_tokens", "loss_tokens", "body_tokens")},
            "editing token denominators differ",
        )
        self.rows, self.draws, self.schedule = rows, draws, schedule


def validated_cohort(recipe: dict) -> tuple[C20Dataset, dict]:
    dataset = C20Dataset(recipe)
    panel = recipe["development"]["panel"]
    profile = json.loads(Path(recipe["cohort"]["length_profile"]["path"]).read_text())
    lengths = [len(row["input_ids"]) for row in dataset.rows]
    targets = [len(row["input_ids"]) - row["target_start"] for row in dataset.rows]
    require(
        profile.get("schema") == "sepalith.full-weight-edit-sft-length-profile.v1"
        and profile.get("rows_sha256") == recipe["cohort"]["rows"]["sha256"],
        "length profile identity differs",
    )
    require(
        profile.get("rows") == len(dataset.rows)
        and profile.get("sequence_tokens", {}).get("sum") == sum(lengths)
        and profile.get("sequence_tokens", {}).get("max") == max(lengths),
        "sequence length profile differs",
    )
    require(
        profile.get("supervised_target_tokens_including_eos", {}).get("sum") == sum(targets)
        and profile.get("supervised_target_tokens_including_eos", {}).get("max") == max(targets),
        "target length profile differs",
    )
    require(
        profile.get("observed_fit_without_truncation") is True
        and max(lengths) <= recipe["cohort"]["max_sequence_tokens"],
        "observed sequence exceeds context",
    )
    require(
        Path(panel["path"]).is_file() and sha256(panel["path"]) == panel["sha256"],
        "development panel differs",
    )
    require(
        recipe["development"]["teacher_forced_diagnostic"] == "separate_metric_not_edit_accuracy"
        and recipe["development"]["generation_evaluation"] == "required_for_edit_and_noop_quality",
        "development metric roles differ",
    )
    return dataset, {
        "status": "pass",
        "unique_rows": len(dataset.rows),
        "draws": len(dataset),
        "updates": recipe["cohort"]["updates"],
        "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"],
        "development_cases": recipe["development"]["cases"],
        "additional_roxy_rows_pending": recipe["data_queue"]["additional_roxy_rows"],
        "cuda_started": False,
    }


Loss = TypeVar("Loss", bound="Tensor | tuple[Tensor, object]")


def checked_loss(result: Loss) -> Loss:
    """Reject a nonfinite loss before Trainer can backpropagate it."""
    import torch

    loss = result[0] if isinstance(result, tuple) else result
    require(bool(torch.isfinite(loss.detach()).all()), "nonfinite raw C20 loss")
    return result


def run(recipe_path: Path, resume: Path | None = None, stop_after_step: int | None = None) -> dict:
    from sepalith.training.sft.c20.run import continuation

    recipe = load_bound(recipe_path)
    recipe_hash = sha256(recipe_path)
    stop_path = Path(recipe["outputs"]["graceful_stop"])
    night = DeadlineStop(deadline_from_env(), stop_after_step, stop_path, recipe_hash)
    gate_plan = continuation(recipe, recipe_path, resume)
    dataset, run_preflight = validated_cohort(recipe)
    runtime = recipe["runtime"]
    require(run_preflight["status"] == "pass", "in-process cohort/holdout preflight failed")
    train_context = recipe["cohort"]["max_sequence_tokens"]
    output = Path(recipe["outputs"]["trainer"])
    archive = Path(recipe["outputs"]["archive"])
    from sepalith.training.checkpoint.campaign_checkpoint import (
        seal_checkpoint,
        verify_checkpoint,
    )

    initial_cursor = 0
    if resume is None:
        require(
            not any((archive / "full").glob("checkpoint-*")),
            "fresh run must not replace a durable checkpoint",
        )
    else:
        prior = verify_checkpoint(
            resume,
            identity(recipe),
            require_full=True,
            expected_checkpoint_kind="full_weights",
        )
        initial_cursor = prior["step"] * 16
    require(
        gate_plan["initial_step"] * 16 == initial_cursor,
        "gate plan/resume cursor differs",
    )
    next_mandatory_stop = gate_plan["next_mandatory_stop"]
    require(
        stop_after_step is None or gate_plan["initial_step"] < stop_after_step <= next_mandatory_stop,
        "stop-after step lies outside this segment",
    )
    stops = []
    holder: dict[str, FullWeightTrainer] = {}
    output.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    telemetry = archive / "telemetry.jsonl"

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        set_seed,
    )
    from sepalith.training.optim.full_weight_optimizer import (
        FullWeightOptimizerTrainerMixin,
        OptimizerConfig,
    )
    from sepalith.training.contracts.campaign_tokenizer_contract import (
        load_pinned_reference_tokenizer,
    )
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        assert_serialized_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )

    limit_cuda(runtime["cuda_memory_limit_mib"])
    set_seed(recipe["seed"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=recipe["parent"]["path"],
        max_seq_length=train_context,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,
        float32_mixed_precision=False,
        fast_inference=False,
        trust_remote_code=False,
        use_gradient_checkpointing=True,
    )
    reference = load_pinned_reference_tokenizer(Path(recipe["parent"]["path"]))
    post_load_repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    require(
        post_load_repair.get("repair", {}).get("vocab_mapping_unchanged") is True,
        "post-load token repair changed vocabulary",
    )
    embeddings_before_trainer = embedding_identity(model)
    token_stage_audits = [
        {"stage": "post_load_repair", **post_load_repair},
        assert_runtime_tokenizer(model, tokenizer, reference, embeddings_before_trainer, "post_load"),
    ]
    model.config.use_cache = False
    try:
        FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    except TypeError:
        FastLanguageModel.for_training(model)
    model.train()
    named = list(model.named_parameters())
    require(
        len(named) == EXPECTED_PARAMETER_TENSORS
        and sum(p.numel() for _, p in named) == EXPECTED_PARAMETERS
        and all(p.requires_grad for _, p in named),
        "full-weight trainable inventory differs",
    )
    require(
        not any("lora_" in n.lower() or "modules_to_save" in n.lower() for n, _ in named),
        "PEFT parameter entered full-weight run",
    )

    expected_dispatch = json.loads(Path(recipe["optimizer_dispatch"]["path"]).read_text())
    observed_positions = []

    def collate(items: list[dict]) -> dict[str, Tensor]:
        positions = [item.pop("_draw_position") for item in items]
        rows = [dict(item) for item in items]
        batch = target_only_collator(rows, max_sequence_tokens=train_context)
        verify_batch(batch, rows)
        batch["_draw_position"] = torch.tensor(positions, dtype=torch.long)
        return batch

    class CampaignCallback(TrainerCallback):
        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> TrainerControl:
            repair = restore_trainer_eog_alignment(model, tokenizer, reference)
            require(
                repair.get("repair", {}).get("vocab_mapping_unchanged") is True,
                "train-begin token repair changed vocabulary",
            )
            token_stage_audits.append(
                assert_runtime_tokenizer(
                    model,
                    tokenizer,
                    reference,
                    embeddings_before_trainer,
                    "train_begin",
                )
            )
            if int(state.global_step) == 0:
                # Pre-update masking gate (card C04): abort before update 1 unless the live first batch is target-only.
                expected = [
                    dataset.rows[dataset.draws[i]]
                    for i in range(runtime["micro_batch"] * runtime["gradient_accumulation"])
                ]
                gate = pre_update_masking_gate(
                    holder["trainer"],
                    kwargs.get("train_dataloader"),
                    expected,
                    accumulation=runtime["gradient_accumulation"],
                    global_step=int(state.global_step),
                )
                write_json(archive / "masking-gate.json", gate)
                append_jsonl(
                    telemetry,
                    {
                        "event": "pre_update_masking_gate_pass",
                        "rows_sha256": gate["rows_sha256"],
                        "at": time.time(),
                    },
                )
            return control

        def on_pre_optimizer_step(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> TrainerControl:
            finite = nonzero = 0
            for _, parameter in named:
                require(
                    parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()),
                    "missing or nonfinite full-weight gradient",
                )
                finite += 1
                nonzero += int(bool(parameter.grad.detach().abs().max().item() > 0))
            if (int(state.global_step) + 1) % runtime["telemetry_every"] == 0:
                append_jsonl(
                    telemetry,
                    {
                        "event": "pre_optimizer",
                        "next_step": int(state.global_step) + 1,
                        "finite_gradient_tensors": finite,
                        "nonzero_gradient_tensors": nonzero,
                        "allocated_bytes": int(torch.cuda.memory_allocated()),
                        "reserved_bytes": int(torch.cuda.memory_reserved()),
                        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "at": time.time(),
                    },
                )
            return control

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs: object,
        ) -> TrainerControl:
            if logs and "loss" in logs:
                require(math.isfinite(float(logs["loss"])), "nonfinite logged loss")
            append_jsonl(
                telemetry,
                {
                    "event": "log",
                    "step": int(state.global_step),
                    "logs": logs or {},
                    "at": time.time(),
                },
            )
            return control

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> TrainerControl:
            if enforce_mandatory_gate_boundary(int(state.global_step), next_mandatory_stop, control):
                append_jsonl(
                    telemetry,
                    {
                        "event": "mandatory_dev_gate_stop_at_optimizer_boundary",
                        "step": int(state.global_step),
                        "at": time.time(),
                    },
                )
            reason = night.apply(int(state.global_step), control)
            if reason is not None:
                stops.append(reason)
                append_jsonl(
                    telemetry,
                    {
                        "event": f"{reason}_save_at_optimizer_boundary",
                        "step": int(state.global_step),
                        "deadline_epoch": night.deadline_epoch,
                        "at": time.time(),
                    },
                )
            return control

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> TrainerControl:
            step = int(state.global_step)
            source = Path(args.output_dir) / f"checkpoint-{step}"
            token_stage_audits.append(
                assert_runtime_tokenizer(
                    model,
                    tokenizer,
                    reference,
                    embeddings_before_trainer,
                    f"before_checkpoint_{step}",
                )
            )
            serialized = assert_serialized_tokenizer(
                source, reference, recipe["parent"]["files"]["tokenizer.json"], sha256
            )
            sampler = {
                "method": "sequential_frozen_draw_schedule",
                "draw_schedule_sha256": recipe["cohort"]["draw_schedule"]["sha256"],
                "cursor": step * 16,
                "global_step": step,
                "effective_batch": 16,
                "ignore_data_skip": False,
            }
            config.require_disk()
            sealed = seal_checkpoint(
                source,
                identity(recipe),
                step,
                full=True,
                sampler=sampler,
                checkpoint_kind="full_weights",
            )
            destination = publish_sealed_checkpoint(source, archive / "full" / source.name, sealed)
            token_stage_audits.append({"stage": f"serialized_checkpoint_{step}", **serialized})
            if step in set(runtime["evaluation_steps"]):
                request = build_evaluation_request(recipe, step, destination)
                write_json(archive / "evaluations" / f"request-step-{step}.json", request)
            retention = apply_retention(recipe, archive)
            append_jsonl(
                telemetry,
                {
                    "event": "checkpoint_durable",
                    "step": step,
                    "path": str(destination),
                    "retention": retention,
                    "at": time.time(),
                },
            )
            return control

    class FullWeightTrainer(FullWeightOptimizerTrainerMixin, Trainer):
        def _get_train_sampler(self, train_dataset: C20Dataset | None = None) -> SequentialSampler:
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        def compute_loss(
            self, model: torch.nn.Module, inputs: dict, *args: object, **kwargs: object
        ) -> Tensor | tuple[Tensor, object]:
            if not observed_positions:
                token_stage_audits.append(
                    assert_runtime_tokenizer(
                        model,
                        tokenizer,
                        reference,
                        embeddings_before_trainer,
                        "first_compute_loss",
                    )
                )
            positions = [int(x) for x in inputs.pop("_draw_position").detach().cpu().tolist()]
            expected = list(
                range(
                    initial_cursor + len(observed_positions),
                    initial_cursor + len(observed_positions) + len(positions),
                )
            )
            require(positions == expected, "actual draw cursor differs from frozen schedule")
            observed_positions.extend(positions)
            return checked_loss(super().compute_loss(model, inputs, *args, **kwargs))

        def create_optimizer(self) -> torch.optim.Optimizer:
            if self.optimizer is None:
                from sepalith.training.sft.c20.optimizer import (
                    build_offloaded_optimizer,
                )

                self.optimizer, self.full_weight_optimizer_manifest = build_offloaded_optimizer(
                    self.model, optimizer_config
                )
            result = super().create_optimizer()
            manifest = self.full_weight_optimizer_manifest
            assert manifest is not None
            for key in (
                "arm",
                "ordered_rows",
                "ordered_rows_sha256",
                "parameter_objects",
                "parameters",
                "counts",
            ):
                require(
                    manifest.get(key) == expected_dispatch.get(key),
                    f"optimizer dispatch differs:{key}",
                )
            return result

    optimizer = dict(runtime["optimizer"])
    optimizer.pop("arm", None)
    optimizer_config = OptimizerConfig(arm=runtime["optimizer"]["arm"], **optimizer)
    args = TrainingArguments(**{**training_arguments_kwargs(recipe, output), "save_total_limit": None})
    trainer = FullWeightTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[CampaignCallback()],
    )
    trainer.full_weight_optimizer_config = optimizer_config
    holder["trainer"] = trainer
    post_trainer_repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    require(
        post_trainer_repair.get("repair", {}).get("vocab_mapping_unchanged") is True,
        "post-Trainer token repair changed vocabulary",
    )
    token_stage_audits.append(
        assert_runtime_tokenizer(
            model,
            tokenizer,
            reference,
            embeddings_before_trainer,
            "post_trainer_construction",
        )
    )
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    terminal_tokenizer = assert_runtime_tokenizer(
        model, tokenizer, reference, embeddings_before_trainer, "after_training"
    )
    terminal_step = int(trainer.state.global_step)
    require(
        terminal_step <= next_mandatory_stop,
        "training continued beyond mandatory DEV gate",
    )
    expected_draws = (terminal_step * 16) - initial_cursor
    require(
        len(observed_positions) == expected_draws,
        "terminal observed draw count differs",
    )
    terminal = archive / "full" / f"checkpoint-{terminal_step}"
    require(terminal.is_dir(), "terminal durable full checkpoint missing")
    report = {
        "schema": "sepalith.sft11.full-weight-edit-sft-run.v1",
        "status": "awaiting_mandatory_dev_gate"
        if terminal_step == next_mandatory_stop
        else ("deadline_stopped" if "deadline" in stops else "gracefully_stopped"),
        "stop_reasons": stops,
        "global_step": terminal_step,
        "initial_cursor": initial_cursor,
        "observed_draws": len(observed_positions),
        "last_draw_position": observed_positions[-1] if observed_positions else None,
        "train_loss": float(result.training_loss),
        "terminal_checkpoint": str(terminal),
        "tokenizer_contract": terminal_tokenizer,
        "tokenizer_stage_audits": token_stage_audits,
        "in_process_preflight": run_preflight,
        "cohort_scope": recipe["cohort"]["id"],
        "gate_plan": gate_plan,
    }
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_json(archive / "run-result.json", report)
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    run(args.recipe, args.resume, args.stop_after_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
