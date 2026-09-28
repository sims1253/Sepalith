#!/usr/bin/env python3
"""Lead-owned CUDA resource smoke for full-weight MiniCPM.

The worker only prepares this script.  The lead may run it on the leased GPU
after reviewing the receipt.  It performs a fresh, unsaved forward/backward
and optimizer step from the merged SFT-500 weights.  It does not generate,
rank, or expose training data and it never resumes an old optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


EXPECTED_PARAMETERS = 2_516_756_480


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument(
        "--optimizer",
        choices=("paged_adamw_8bit", "torch_adamw_bf16", "none", "aurora_mix"),
        default="paged_adamw_8bit",
    )
    p.add_argument("--max-sequence-tokens", type=int, default=4096)
    return p.parse_args()


def read_rows(path: Path, max_sequence_tokens: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("smoke input is empty")
    for row in rows:
        input_ids = row["input_ids"]
        labels = row["labels"]
        attention = row["attention_mask"]
        if not (len(input_ids) == len(labels) == len(attention)):
            raise ValueError(f"row {row.get('row_id')} has inconsistent token arrays")
        if len(input_ids) > max_sequence_tokens:
            raise ValueError(
                f"row {row.get('row_id')} has {len(input_ids)} tokens > bound "
                f"{max_sequence_tokens}; smoke must never truncate"
            )
        if not any(label != -100 for label in labels):
            raise ValueError(f"row {row.get('row_id')} has no supervised labels")
    return rows


def optimizer_state_bytes(optimizer) -> tuple[int, dict[str, int]]:
    total = 0
    dtypes: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if not hasattr(value, "numel"):
                continue
            size = int(value.numel()) * int(value.element_size())
            total += size
            key = str(value.dtype)
            dtypes[key] = dtypes.get(key, 0) + size
    return total, dtypes


def main() -> int:
    args = parse_args()
    if args.report.exists():
        raise ValueError("Smoke report path must be fresh")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    # This must happen before importing Unsloth or torch: the Unsloth loss path
    # otherwise may retain a full logits return mode from the host environment.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": "sepalith.sft11.full_weight_cuda_resource_smoke.v1",
        "status": "starting",
        "model": str(args.model),
        "rows": str(args.rows),
        "optimizer": args.optimizer,
        "full_finetuning": True,
        "load_in_4bit": False,
        "dtype": "bfloat16",
        "max_sequence_tokens": args.max_sequence_tokens,
        "steps": args.steps,
        "resume_from": None,
        "checkpoint_saved": False,
        "quality_or_generation_evaluation": False,
        "phase": "initializing",
        "breadcrumbs": [],
    }
    torch = None

    def persist() -> None:
        if torch is not None and torch.cuda.is_available():
            report["memory_live"] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    def phase(name: str) -> None:
        report["phase"] = name
        report["breadcrumbs"].append({"phase": name, "at": time.time()})
        persist()

    try:
        # Import Unsloth before the explicit torch import.  Unsloth imports its
        # pinned torch stack and applies the model/loss patches during import.
        phase("import_unsloth")
        from unsloth import FastLanguageModel
        import torch as _torch

        torch = _torch
        torch.manual_seed(20260914)
        torch.cuda.manual_seed_all(20260914)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this lead-owned resource smoke")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected one leased CUDA device, found {torch.cuda.device_count()}")
        torch.cuda.reset_peak_memory_stats()
        phase("read_and_validate_rows")
        rows = read_rows(args.rows, args.max_sequence_tokens)

        phase("load_full_weight_model")
        started = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model),
            max_seq_length=args.max_sequence_tokens,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=True,
            float32_mixed_precision=False,
            fast_inference=False,
            trust_remote_code=False,
            use_gradient_checkpointing=True,
        )
        load_seconds = time.time() - started
        phase("configure_training")
        model.config.use_cache = False
        try:
            FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
        except TypeError:
            FastLanguageModel.for_training(model)
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except (AttributeError, TypeError):
            # The loader already requested checkpointing on supported Unsloth builds.
            pass
        model.train()

        phase("parameter_and_peft_guards")
        all_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        trainable_parameter_tensors = sum(
            1 for parameter in model.parameters() if parameter.requires_grad
        )
        peft_names = [
            name for name, parameter in model.named_parameters()
            if "lora_" in name.lower() or "modules_to_save" in name.lower()
        ]
        if peft_names:
            raise RuntimeError(f"full-weight smoke rejected PEFT parameters: {peft_names[:5]}")
        if all_parameters != EXPECTED_PARAMETERS or trainable_parameters != all_parameters:
            raise RuntimeError(
                f"full-weight parameter guard failed: trainable={trainable_parameters}, "
                f"all={all_parameters}, expected={EXPECTED_PARAMETERS}"
            )

        phase("construct_fresh_optimizer")
        if args.optimizer == "aurora_mix":
            from full_weight_optimizer import OptimizerConfig, build_full_weight_optimizer
            local = Path(__file__).resolve().parent
            configs = json.loads((local / "smoke-configs.json").read_text())
            values = dict(configs["common"], **configs["smoke_arms"]["aurora_mix"])
            values["adam_betas"] = tuple(values["adam_betas"])
            optimizer, actual_manifest = build_full_weight_optimizer(model, OptimizerConfig(**values))
            expected_manifest = json.loads((local / "expected-aurora-mix-manifest.json").read_text())
            report["optimizer_dispatch"] = actual_manifest
            persist()
            semantic_keys = ("arm", "ordered_rows", "ordered_rows_sha256", "parameter_objects", "parameters", "counts")
            if any(actual_manifest[key] != expected_manifest[key] for key in semantic_keys):
                raise ValueError("Actual full-weight optimizer dispatch differs from expected manifest")
        elif args.optimizer == "paged_adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW8bit(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=1e-5,
                betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        elif args.optimizer == "torch_adamw_bf16":
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=1e-5,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                foreach=False,
                fused=False,
            )
        else:
            optimizer = None

        step_records = []
        device = next(model.parameters()).device
        for step in range(args.steps):
            row = rows[step % len(rows)]
            phase(f"step_{step + 1}_forward_backward")
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
            labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite or missing loss at smoke step {step}")
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.time() - t0
            finite_grad_count = 0
            nonzero_grad_count = 0
            for parameter in model.parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise RuntimeError(f"non-finite gradient at smoke step {step}")
                finite_grad_count += 1
                if bool(parameter.grad.detach().abs().max().item() > 0):
                    nonzero_grad_count += 1
            if finite_grad_count != trainable_parameter_tensors:
                raise RuntimeError(
                    f"gradient coverage failed at smoke step {step}: "
                    f"{finite_grad_count}/{trainable_parameter_tensors} parameter tensors"
                )
            if optimizer is not None:
                phase(f"step_{step + 1}_optimizer")
                samples = {}
                for name, parameter in model.named_parameters():
                    sample_count = min(256, parameter.numel())
                    indices = (torch.arange(sample_count, dtype=torch.int64, device=parameter.device) * (parameter.numel()-1)) // max(1, sample_count-1)
                    samples[name] = (indices, parameter.detach().view(-1)[indices].clone())
                torch.cuda.synchronize()
                optimizer_started = time.time()
                optimizer.step()
                torch.cuda.synchronize()
                optimizer_seconds = time.time() - optimizer_started
                delta_rows = []
                for name, parameter in model.named_parameters():
                    indices, before = samples[name]
                    after = parameter.detach().view(-1)[indices]
                    delta_rows.append({"name": name, "sampled_elements": int(indices.numel()), "changed_sampled_elements": int((before != after).sum().item())})
                for state in optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value) and value.is_floating_point():
                            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                                raise ValueError("Optimizer state is not finite FP32")
                report.setdefault("optimizer_readouts", []).append({"step": step+1, "seconds": optimizer_seconds, "parameter_delta_samples": delta_rows, "all_floating_state_finite_fp32": True, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())})
                del samples, delta_rows
                persist()
            state_bytes, state_dtypes = optimizer_state_bytes(optimizer) if optimizer is not None else (0, {})
            step_records.append(
                {
                    "step": step + 1,
                    "row_id": row.get("row_id"),
                    "sequence_tokens": len(row["input_ids"]),
                    "supervised_tokens": sum(label != -100 for label in row["labels"]),
                    "loss": float(loss.detach().cpu()),
                    "finite_loss": True,
                    "finite_gradient_parameter_tensors": finite_grad_count,
                    "nonzero_gradient_parameter_tensors": nonzero_grad_count,
                    "trainable_parameter_tensors": trainable_parameter_tensors,
                    "backward_seconds": backward_seconds,
                    "optimizer_state_bytes_observed": state_bytes,
                    "optimizer_state_dtypes": state_dtypes,
                }
            )
            report["step_records"] = step_records
            persist()
            del output, loss, input_ids, attention_mask, labels

        phase("complete")
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report.update(
            {
                "status": "completed_resource_smoke",
                "trainable_parameters": trainable_parameters,
                "all_parameters": all_parameters,
                "trainable_parameter_tensors": trainable_parameter_tensors,
                "load_seconds": load_seconds,
                "step_records": step_records,
                "memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "free_bytes_after": int(free_bytes),
                    "total_bytes": int(total_bytes),
                },
                "guards": {
                    "no_target_truncation": True,
                    "use_cache": bool(model.config.use_cache),
                    "labels_path_exercises_fused_or_chunked_loss": True,
                    "peft_parameter_names": peft_names,
                },
            }
        )
        persist()
        print(json.dumps(report, indent=2))
        return 0
    except BaseException as error:
        report.update(
            {
                "status": "failed_resource_smoke",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        persist()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
