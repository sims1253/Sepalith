#!/usr/bin/env python3
"""Read-only audit of the full-weight CPT LR=1e-4 pilot.

This script reads JSON/JSONL inputs and bounded safetensors samples. It never
imports or constructs the model and never reads optimizer tensor payloads.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

import numpy as np


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-full-weight-cpt-lr-pilot-v1"
ARCHIVE = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-LR-pilot-lr1e-4-v1")
PARENT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-CPT-global-a-250-merged/model.safetensors")
CHILD = ARCHIVE / "full/checkpoint-24/model.safetensors"
OUT = Path(__file__).resolve().parent / "independent-audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def safetensors_header(path: Path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return size, json.loads(stream.read(size))


def sample_tensor(path: Path, header_bytes: int, info: dict, indexes: list[int]) -> np.ndarray:
    item_bytes = {"BF16": 2, "F32": 4}[info["dtype"]]
    chunks = []
    with path.open("rb") as stream:
        for index in indexes:
            stream.seek(8 + header_bytes + info["data_offsets"][0] + item_bytes * index)
            chunks.append(stream.read(item_bytes))
    raw = b"".join(chunks)
    if info["dtype"] == "BF16":
        values = np.frombuffer(raw, dtype="<u2")
        return (values.astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, dtype="<f4")


def bounded_parameter_samples():
    parent_header_bytes, parent_header = safetensors_header(PARENT)
    child_header_bytes, child_header = safetensors_header(CHILD)
    names = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.20.mlp.up_proj.weight",
        "model.layers.41.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.20.self_attn.k_proj.weight",
        "model.layers.41.self_attn.o_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.20.input_layernorm.weight",
        "model.norm.weight",
    ]
    rows = []
    total_bytes_per_file = 0
    for name in names:
        before_info, after_info = parent_header[name], child_header[name]
        assert before_info["shape"] == after_info["shape"]
        count = math.prod(before_info["shape"])
        sample_count = min(4096, count)
        block = sample_count // 4
        indexes = []
        for segment in range(4):
            start = (count - block) * segment // 3
            indexes.extend(range(start, start + block))
        before = sample_tensor(PARENT, parent_header_bytes, before_info, indexes)
        after = sample_tensor(CHILD, child_header_bytes, after_info, indexes)
        delta = after - before
        weight_rms = float(np.sqrt(np.mean(before * before)))
        delta_rms = float(np.sqrt(np.mean(delta * delta)))
        rows.append({
            "name": name,
            "shape": before_info["shape"],
            "parent_dtype": before_info["dtype"],
            "checkpoint_dtype": after_info["dtype"],
            "sample_elements": len(indexes),
            "changed_fraction": float(np.count_nonzero(delta) / len(delta)),
            "weight_rms": weight_rms,
            "delta_rms": delta_rms,
            "relative_delta_rms_pct": 100.0 * delta_rms / weight_rms,
            "delta_abs_max": float(np.max(np.abs(delta))),
            "finite": bool(np.isfinite(after).all()),
        })
        total_bytes_per_file += len(indexes) * max(
            {"BF16": 2, "F32": 4}[before_info["dtype"]],
            {"BF16": 2, "F32": 4}[after_info["dtype"]],
        )
    return {
        "method": "four deterministic contiguous segments per named tensor; header plus at most 4096 values per tensor",
        "payload_upper_bound_bytes_per_file": total_bytes_per_file,
        "not_a_full_parameter_norm": True,
        "rows": rows,
    }


def main():
    recipe_path = PACKET / "bound-recipe.lr1e-4.root.json"
    recipe = load(recipe_path)
    baseline_path = ARCHIVE / "evaluations/baseline-step-0.json"
    terminal_path = ARCHIVE / "evaluations/step-24.json"
    baseline, terminal = load(baseline_path), load(terminal_path)
    run_result = load(ARCHIVE / "run-result.json")
    telemetry = jsonl(ARCHIVE / "telemetry.jsonl")
    train = jsonl(PACKET / "panel/train384-ctx8192.jsonl")
    provenance = jsonl(PACKET / "panel/provenance.jsonl")
    validation = jsonl(Path(recipe["validation"]["path"]))
    logs = [row["logs"] for row in telemetry if row.get("event") == "log" and "loss" in row.get("logs", {})]
    gradients = [row for row in telemetry if row.get("event") == "pre_optimizer"]
    train_doc_lengths = sorted(row["complete_document_tokens"] for row in provenance)
    validation_lengths = sorted(len(row["input_ids"]) for row in validation)
    nll0 = baseline["metrics"]["mean_causal_nll"]
    nll24 = terminal["metrics"]["mean_causal_nll"]
    source = (PACKET / "source/experiments/training/full_weight_cpt_trainer.py").read_text()
    panel_source = (PACKET / "source/prepare_panel.py").read_text()
    evidence = {
        "schema": "sepalith.sft11.full-weight-cpt-regression-independent-audit.v1",
        "status": "complete_cpu_read_only",
        "scope": "TRAIN-only causal-loss diagnosis; no DEV/final/model load/CUDA/training",
        "input_hashes": {
            "bound_recipe": sha256(recipe_path),
            "baseline": sha256(baseline_path),
            "terminal": sha256(terminal_path),
            "run_result": sha256(ARCHIVE / "run-result.json"),
            "telemetry": sha256(ARCHIVE / "telemetry.jsonl"),
            "panel_rows": sha256(PACKET / "panel/train384-ctx8192.jsonl"),
            "panel_provenance": sha256(PACKET / "panel/provenance.jsonl"),
        },
        "evaluation": {
            "baseline_nll": nll0,
            "step24_nll": nll24,
            "absolute_delta": nll24 - nll0,
            "relative_nll_pct": 100.0 * (nll24 / nll0 - 1.0),
            "baseline_perplexity": math.exp(nll0),
            "step24_perplexity": math.exp(nll24),
            "relative_perplexity_pct": 100.0 * (math.exp(nll24 - nll0) - 1.0),
            "same_denominators": baseline["denominators"] == terminal["denominators"],
            "denominators": baseline["denominators"],
            "same_case_ids": baseline["case_ids"] == terminal["case_ids"],
            "unique_documents": len(baseline["case_ids"]),
            "aggregate_only_no_per_row_losses": "rows" not in baseline and "rows" not in terminal,
        },
        "execution": {
            "status": run_result["status"],
            "global_step": run_result["global_step"],
            "observed_draws": run_result["observed_draws"],
            "last_draw_position": run_result["last_draw_position"],
            "all_gradient_tensors_finite_each_step": all(row["finite_gradient_tensors"] == 381 for row in gradients),
            "all_gradient_tensors_nonzero_each_step": all(row["nonzero_gradient_tensors"] == 381 for row in gradients),
            "gradient_steps": len(gradients),
            "gradient_norm_min": min(row["grad_norm"] for row in logs),
            "gradient_norm_max": max(row["grad_norm"] for row in logs),
            "gradient_norm_mean": statistics.mean(row["grad_norm"] for row in logs),
            "train_loss_reported": run_result["train_loss"],
            "train_loss_first_8_mean": statistics.mean(row["loss"] for row in logs[:8]),
            "train_loss_last_8_mean": statistics.mean(row["loss"] for row in logs[-8:]),
            "logged_lr_sequence": [row["learning_rate"] for row in logs],
            "warmup_interpretation": "optimizer step1 uses 0; step2 uses 5e-5; steps3-24 use 1e-4",
        },
        "tokenizer_and_eval_mode": {
            "all_runtime_audits_bos0_eos1_pad1": all(
                row.get("bos") == 0 and row.get("eos") == 1 and row.get("pad") == 1
                for row in run_result["tokenizer_stage_audits"] if "bos" in row
            ),
            "serialized_tokenizer_sha256": next(
                row["tokenizer_json_sha256"] for row in run_result["tokenizer_stage_audits"]
                if row["stage"] == "serialized_checkpoint_24"
            ),
            "expected_tokenizer_sha256": recipe["parent"]["files"]["tokenizer.json"],
            "baseline_before_trainer_construction": source.index("baseline = evaluator(") < source.index("trainer = FullWeightTrainer("),
            "evaluator_model_eval_no_grad_and_restores_mode": all(token in (PACKET / "source/experiments/training/campaign_cpt_eval.py").read_text() for token in ("model.eval()", "torch.no_grad()", "model.train(was_training)")),
            "terminal_evaluation_uses_live_model": 'evaluator(kwargs["model"]' in source,
            "serialized_checkpoint_not_reloaded_for_terminal_eval": True,
        },
        "cohort_geometry": {
            "train_rows": len(train),
            "train_packages": len({row["package"] for row in train}),
            "train_documents": len({row["document_id"] for row in train}),
            "all_train_rows_exact_8192": all(len(row["input_ids"]) == 8192 for row in train),
            "all_train_rows_first_nonterminal_chunk": all(row["chunk_index"] == 0 and not row["is_document_end"] for row in train),
            "selected_payload_tokens": sum(row["supervised_tokens"] for row in train),
            "complete_document_tokens": sum(train_doc_lengths),
            "fraction_of_selected_documents_payload_used": sum(row["supervised_tokens"] for row in train) / sum(train_doc_lengths),
            "train_document_token_length": {"min": min(train_doc_lengths), "median": statistics.median(train_doc_lengths), "max": max(train_doc_lengths)},
            "validation_rows": len(validation),
            "validation_packages": len({row["package"] for row in validation}),
            "validation_documents": len({row["document_id"] for row in validation}),
            "validation_terminal_rows": sum(bool(row["is_document_end"]) for row in validation),
            "validation_nonterminal_rows": sum(not row["is_document_end"] for row in validation),
            "validation_input_length": {"min": min(validation_lengths), "median": statistics.median(validation_lengths), "max": max(validation_lengths), "mean": statistics.mean(validation_lengths)},
            "package_overlap": len({row["package"] for row in train} & {row["package"] for row in validation}),
            "document_overlap": len({row["document_id"] for row in train} & {row["document_id"] for row in validation}),
            "selection_is_sorted_not_seeded_random": 'for root in sorted(BASE.iterdir())' in panel_source and "random" not in panel_source,
            "selection_takes_smallest_document_id": "chosen=min(eligible,key=lambda x:x['document_id'])" in panel_source,
            "metadata_files_scanned_before_384_packages": load(PACKET / "panel/manifest.json")["selection"]["metadata_files_scanned"],
        },
        "bounded_parameter_sampling": bounded_parameter_samples(),
    }
    OUT.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": evidence["status"], "output": str(OUT), "sha256": sha256(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
