#!/usr/bin/env python3
"""Pure decision gate for source-walk semantic evidence records.

The streaming producer must construct the evidence from exact frozen source
bytes. This module does not infer support from identifier names alone.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

CANONICAL_BOS = 0
CANONICAL_EOS = 1
VOCAB_SIZE = 130560

OCCURRENCE_SUPPORTED = {
    "formal_binding_reference",
    "local_binding_reference",
    "member_label_reference",
    "package_namespace_reference",
    "r_builtin_or_operator",
    "call_argument_label",
    "same_file_definition_selected",
    "exact_namespace_import",
}


def strict_token_contract(row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    ids = row.get("input_ids")
    body = row.get("target_body_tokens")
    terminal = row.get("target_terminal_tokens")
    start = row.get("target_start")
    if row.get("bos_token_id") != CANONICAL_BOS or row.get("eos_token_id") != CANONICAL_EOS:
        failures.append("noncanonical_special_token_metadata")
    if not isinstance(ids, list) or any(type(x) is not int or not 0 <= x < VOCAB_SIZE for x in ids):
        failures.append("invalid_token_id")
        return failures
    if not isinstance(body, list) or not isinstance(terminal, list) or type(start) is not int:
        failures.append("invalid_target_token_fields")
        return failures
    if not ids or ids[0] != CANONICAL_BOS or ids[-1] != CANONICAL_EOS:
        failures.append("noncanonical_sequence_boundary")
    if terminal != row.get("canonical_terminal_tokens"):
        failures.append("noncanonical_terminal_tokens")
    if ids[start:] != body + terminal + [CANONICAL_EOS]:
        failures.append("target_suffix_geometry")
    if len(ids) != row.get("prompt_token_count", -1) + row.get("target_token_count", -2) + 2:
        failures.append("sequence_count")
    if row.get("split") != "train":
        failures.append("split_not_train")
    return failures


def classify_evidence(e: dict[str, Any]) -> dict[str, Any]:
    """Classify one exact evidence record without silently dropping it."""
    blockers: list[str] = []
    review: list[str] = []

    required_true = {
        "source_sha256_matches": "source_hash_mismatch",
        "raw_line_sha256_matches": "raw_line_hash_mismatch",
        "global_train_split_verified": "split_or_registry_unverified",
        "protected_split_disjoint": "protected_split_overlap_or_unverified",
        "direct_license_verified": "license_unverified",
        "before_after_inverse_roundtrip": "source_reconstruction_failure",
        "unique_target_definition": "target_definition_not_unique",
        "target_range_matches": "target_range_mismatch",
        "source_parses": "source_parse_failure",
        "complete_target": "target_incomplete_or_truncated",
        "prompt_excludes_target": "hidden_target_leakage",
        "required_spans_retained": "necessary_context_span_missing",
    }
    for field, reason in required_true.items():
        if e.get(field) is not True:
            blockers.append(reason)

    blockers.extend(strict_token_contract(e.get("token_row", {})))
    if e.get("documented_params_duplicate"):
        blockers.append("duplicate_param_tags")
    if e.get("documented_params_missing_from_formals"):
        blockers.append("invented_or_wrong_param")

    for occurrence in e.get("reference_occurrences", []):
        category = occurrence.get("category")
        if category in OCCURRENCE_SUPPORTED:
            continue
        if category == "nse_data_column":
            if not all(occurrence.get(k) for k in ("call_head", "call_span_sha256", "data_expression_sha256")):
                review.append("nse_occurrence_evidence_incomplete")
        elif category == "unqualified_external_callable":
            review.append("external_callable_requires_package_api_evidence")
        else:
            review.append("unbound_reference_requires_evidence")

    # Prose or behavior claims need a cited same-package source occurrence;
    # formals alone cannot prove natural-language correctness.
    unsupported_claims = e.get("unsupported_documentation_claims", [])
    if unsupported_claims:
        review.append("documentation_claim_source_support_required")

    if blockers:
        status = "hold_mechanical_or_identity_blocker"
    elif review:
        status = "hold_semantic_evidence_required"
    else:
        status = "supported_candidate_root_review_required"
    return {
        "row_id": e.get("row_id"),
        "status": status,
        "blockers": sorted(set(blockers)),
        "review_reasons": sorted(set(review)),
        "training_admission": False,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(x["status"] for x in results)
    if sum(statuses.values()) != len(results):
        raise AssertionError("denominator_loss")
    return {"rows": len(results), "status_counts": dict(sorted(statuses.items())), "training_admission": False}
