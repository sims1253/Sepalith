"""Optional package-held-out TRAIN causal-loss evaluator.

This evaluator is deliberately separate from the DEV edit panel. It reports
causal next-token loss on package-disjoint TRAIN rows and cannot establish an
SFT continuation or edit-quality gate.
"""
from __future__ import annotations

from typing import Any

from sepalith.training.cpt.campaign_cpt_data import causal_lm_collator, causal_loss_denominator, read_jsonl


def _case_id(row: dict[str, Any]) -> str:
    """Return the materializer's document identity without changing row shape."""
    if isinstance(row.get("document_id"), str) and row["document_id"]:
        return row["document_id"]
    source = row.get("source")
    if isinstance(source, dict) and isinstance(source.get("source_id"), str) and source["source_id"]:
        return source["source_id"]
    raise ValueError("CPT holdout row has no document identity")


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("row_id") or row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("CPT holdout row has no unique row identity")
    return value


def _package_id(row: dict[str, Any]) -> str:
    value = row.get("package_id") or row.get("package")
    if not isinstance(value, str) or not value:
        raise ValueError("CPT holdout row has no package identity")
    return value


def package_holdout_evaluator(recipe: dict[str, Any]):
    record = recipe["validation_rows"]
    rows = read_jsonl(record["path"], max_sequence_tokens=recipe["parameters"]["max_sequence_tokens"],
                      materialized=recipe.get("materialized_rows", True))
    batch_size = int(recipe.get("validation_batch_size", 1))
    if batch_size < 1:
        raise ValueError("validation_batch_size must be positive")

    def evaluate(model, tokenizer, checkpoint, step):
        import torch

        was_training = model.training
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        row_metrics = []
        device = next(model.parameters()).device
        model.eval()
        try:
            with torch.no_grad():
                for offset in range(0, len(rows), batch_size):
                    batch_rows = rows[offset:offset + batch_size]
                    batch = causal_lm_collator(
                        batch_rows, max_sequence_tokens=recipe["parameters"]["max_sequence_tokens"]
                    )
                    batch = {name: value.to(device) for name, value in batch.items()}
                    denominator = causal_loss_denominator(batch)
                    if denominator <= 0:
                        raise ValueError("package holdout batch has no causal loss targets")
                    output = model(**batch)
                    loss = getattr(output, "loss", None)
                    if loss is None or not torch.isfinite(loss):
                        raise ValueError("model did not return a finite fused causal loss")
                    value = float(loss.detach().cpu())
                    numerator = value * denominator
                    total_loss += numerator
                    total_tokens += denominator
                    for row in batch_rows:
                        # Pilot fixes batch_size=1, preserving exact per-row attribution.
                        if len(batch_rows) != 1:
                            raise ValueError("per-row pilot diagnostics require validation_batch_size=1")
                        row_metrics.append({"row_id": _row_id(row), "document_id": _case_id(row), "package_id": _package_id(row), "loss_tokens": denominator, "loss_sum": numerator, "mean_causal_nll": value})
                    batches += 1
        finally:
            model.train(was_training)
        if total_tokens <= 0:
            raise ValueError("package holdout has no causal loss tokens")
        return {
            "schema": "sepalith.cpt.package-holdout-eval.v1",
            "checkpoint": str(checkpoint), "step": step,
            "validation_role": "package_heldout_train_causal_lm",
            "metrics": {"mean_causal_nll": total_loss / total_tokens},
            "denominators": {
                "validation_rows": len(rows), "validation_loss_tokens": total_tokens,
                "validation_batches": batches,
            },
            "case_ids": sorted({_case_id(row) for row in rows}),
            "row_metrics": row_metrics,
            "row_metrics_contract": "ordered validation rows; aggregate numerator and denominator remain authoritative",
            "quality_gate": "diagnostic_only_until_a_separate_SFT_stage",
            "edit_accuracy_gate": "not_applicable",
        }

    return evaluate
