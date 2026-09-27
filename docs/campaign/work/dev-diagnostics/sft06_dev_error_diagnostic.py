#!/usr/bin/env python3
"""Reproducible, read-only SFT-06 diagnostic for the 14 finish/roxygen rows.

This diagnostic deliberately does not recompute training scores or alter labels.
It joins the locked DAT-07 development panel to the completed step-500
per-case archive, strips the protocol terminator, and records a small semantic
review of the six finish_block and eight roxygen_drafting rows.  The review
categories are kept separate from evaluator fields such as ``exact_region``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


CASES_PATH = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT-07-final-evaluator-cases.jsonl"
)
STEP500_PATH = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/"
    "SFT-primary-3000-a/evaluations/cases-step-500.json"
)
STEP1000_CASES_PATH = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/"
    "SFT-primary-3000-a/evaluations/cases-step-1000.json"
)
STEP1000_SUMMARY_PATH = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/"
    "SFT-primary-3000-a/evaluations/step-1000.json"
)

CASES_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
STEP500_SHA256 = "3e06b8990ab7808dc0d5683ba108425d450391114e76f35399cbf1377a2c012a"

TARGET_FAMILIES = ("finish_block", "roxygen_drafting")

# These are bounded review decisions, made from the visible context and target
# in the locked rows.  They are diagnostic labels and never feed the evaluator.
PRIMARY_REVIEW = {
    # Protocol-valid roxygen that describes the visible function sufficiently
    # for a useful alternative, even when it is much shorter than the target.
    "dat07-existing-c54ed09efae0ca38cff22981": "valid_alternative_docs",
    "dat07-existing-06847c759fd32bdf16f2e598": "valid_alternative_docs",
    "dat07-existing-5b5e844d3a8795c52149f848": "valid_alternative_docs",
    "dat07-existing-06097c12b0328d475be857e1": "valid_alternative_docs",
    "dat07-existing-d38a792699b4cccded445dc0": "valid_alternative_docs",
    # Protocol-valid prose whose central claim conflicts with the visible
    # function's target (vector vs matrix; restricted cubic vs B-spline).
    "dat07-existing-152d62f7a57472f8abb30ca9": "doc_semantic_mismatch",
    "dat07-existing-81ddc4c6ddd226b04dcf0910": "doc_semantic_mismatch",
    # Protocol-valid code that parses after the fixture's documented synthetic
    # close, but implements a different operation from the target.
    "e623a61b5a4c066358a477f2": "unsupported_behavior",
    "157517ba47dbab157f7c361a": "unsupported_behavior",
    "f43de3e77f2d92ed7b223464": "unsupported_behavior",
    # Four capped/missing-terminal generations.  Three also have a repeated
    # block; regmed is capped without a repeated identical block.
    "dat07-existing-2838b904f1e44b7440e6da9f": "malformed_or_truncated",
    "4f08633513b5c525240d2540": "malformed_or_truncated",
    "d11581e9cfa4e3971aa1466e": "malformed_or_truncated",
    "04834fef4fe59742f13677a9": "malformed_or_truncated",
}

SECONDARY_REVIEW = {
    "dat07-existing-c54ed09efae0ca38cff22981": ["shorter_than_target"],
    "dat07-existing-152d62f7a57472f8abb30ca9": ["shorter_than_target"],
    "dat07-existing-2838b904f1e44b7440e6da9f": ["repetition"],
    "dat07-existing-06847c759fd32bdf16f2e598": [
        "under_specified_docs",
        "generated_wrapper_context",
    ],
    "dat07-existing-81ddc4c6ddd226b04dcf0910": ["algorithm_name_mismatch"],
    "dat07-existing-5b5e844d3a8795c52149f848": ["shorter_than_target"],
    "dat07-existing-06097c12b0328d475be857e1": ["shorter_than_target"],
    "dat07-existing-d38a792699b4cccded445dc0": ["shorter_than_target"],
    "e623a61b5a4c066358a477f2": ["algorithm_and_shape_mismatch"],
    "4f08633513b5c525240d2540": ["repetition"],
    "d11581e9cfa4e3971aa1466e": ["repetition"],
    "157517ba47dbab157f7c361a": ["algorithm_and_shape_mismatch"],
    "f43de3e77f2d92ed7b223464": ["algorithm_and_input_contract_mismatch"],
    "04834fef4fe59742f13677a9": ["truncated_comment_growth"],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decoded_output(raw_output: str) -> str:
    """Return only the generated replacement, before the exact terminal."""

    return raw_output.split(">>>>>>>", 1)[0].rstrip("\r\n")


def projected_code(case: dict[str, Any], replacement: str, *, synthetic_close: bool) -> str:
    """Project a replacement into the case's visible context.

    The finish_block fixture intentionally ends at the function body and has
    no suffix containing the outer ``}``.  One synthetic close lets tree-sitter
    inspect expression structure without calling the fixture incomplete output
    a parser error.  It is recorded in every finish parse result.
    """

    context = case["context"]
    lines = (
        list(context.get("prefix", []))
        + replacement.splitlines()
        + list(context.get("suffix_lines", []))
    )
    text = "\n".join(lines) + "\n"
    if synthetic_close:
        text += "}\n"
    return text


def repeated_block(lines: list[str], *, minimum_repeats: int = 3) -> dict[str, Any] | None:
    """Find a short, contiguous repeated block while ignoring blank blocks."""

    limit = min(8, max(1, len(lines) // minimum_repeats))
    for width in range(1, limit + 1):
        for start in range(0, len(lines) - width * minimum_repeats + 1):
            block = lines[start : start + width]
            if not any(line.strip() for line in block):
                continue
            repeats = 1
            while (
                start + (repeats + 1) * width <= len(lines)
                and lines[start + repeats * width : start + (repeats + 1) * width]
                == block
            ):
                repeats += 1
            if repeats >= minimum_repeats:
                return {"block_lines": block, "width": width, "repeats": repeats}
    return None


def tree_sitter_parse(text: str) -> dict[str, Any]:
    """Parse with the canonical R tree-sitter installation when available."""

    try:
        from tree_sitter import Language, Parser
        import tree_sitter_r
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        return {
            "available": False,
            "has_error": None,
            "error_nodes": [],
            "unavailable_reason": f"{type(exc).__name__}: {exc}",
        }

    parser = Parser(Language(tree_sitter_r.language()))
    tree = parser.parse(text.encode("utf-8"))
    errors: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if node.is_error or node.is_missing:
            errors.append(
                {
                    "type": node.type,
                    "start": list(node.start_point),
                    "end": list(node.end_point),
                    "is_error": bool(node.is_error),
                    "is_missing": bool(node.is_missing),
                }
            )
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return {
        "available": True,
        "has_error": bool(tree.root_node.has_error),
        "error_nodes": errors[:8],
    }


def bounded_excerpt(text: str, limit: int = 240) -> str:
    one_line = " | ".join(text.strip().splitlines())
    one_line = " ".join(one_line.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def inspect_step1000() -> dict[str, Any]:
    """Report step-1000 state without consuming it for the step-500 review."""

    result: dict[str, Any] = {
        "cases_path": str(STEP1000_CASES_PATH),
        "summary_path": str(STEP1000_SUMMARY_PATH),
        "cases_present": STEP1000_CASES_PATH.exists(),
        "summary_present": STEP1000_SUMMARY_PATH.exists(),
        "completed_manifest_present": False,
        "admitted": False,
        "reason": "no completed step-1000 manifest/summary",
    }
    if STEP1000_CASES_PATH.exists():
        result["cases_sha256"] = sha256_file(STEP1000_CASES_PATH)
        try:
            partial = json.loads(STEP1000_CASES_PATH.read_text())
            result["status"] = partial.get("status")
            result["step"] = partial.get("step")
            result["result_count"] = len(partial.get("results", []))
            result["expected_case_count"] = len(partial.get("expected_case_ids", [])) or 75
            result["completed_manifest_present"] = bool(
                STEP1000_SUMMARY_PATH.exists()
                and partial.get("status") == "complete"
                and len(partial.get("results", []))
                == result["expected_case_count"]
                == 75
            )
        except Exception as exc:  # pragma: no cover - defensive evidence record
            result["read_error"] = f"{type(exc).__name__}: {exc}"
    if STEP1000_SUMMARY_PATH.exists():
        result["summary_sha256"] = sha256_file(STEP1000_SUMMARY_PATH)
    if result["completed_manifest_present"]:
        result["reason"] = (
            "step-1000 completed after the bounded step-500 snapshot; it is "
            "reported for handoff and not used in this review"
        )
    return result


def analyse() -> dict[str, Any]:
    cases_bytes = CASES_PATH.read_bytes()
    step500_bytes = STEP500_PATH.read_bytes()
    cases = load_jsonl(CASES_PATH)
    step500 = json.loads(step500_bytes)
    rows = {row["id"]: row for row in cases}
    results = {row["id"]: row for row in step500["results"]}

    if sha256_bytes(cases_bytes) != CASES_SHA256:
        raise RuntimeError("locked DAT-07 panel hash changed")
    if sha256_bytes(step500_bytes) != STEP500_SHA256:
        raise RuntimeError("completed step-500 archive hash changed")
    if len(cases) != 75 or len(results) != 75:
        raise RuntimeError("expected complete 75-case panel and step-500 archive")

    selected_ids = [
        row["id"] for row in cases if row.get("family") in TARGET_FAMILIES
    ]
    if len(selected_ids) != 14 or set(selected_ids) != set(PRIMARY_REVIEW):
        raise RuntimeError("target family rows do not match the bounded 14-row review")

    reviewed: list[dict[str, Any]] = []
    for case_id in selected_ids:
        case = rows[case_id]
        result = results.get(case_id)
        if result is None:
            raise RuntimeError(f"missing step-500 result for {case_id}")
        for key in ("family", "package_id"):
            if result.get(key) != case.get(key):
                raise RuntimeError(f"step-500 {key} mismatch for {case_id}")

        output = decoded_output(result["raw_output"])
        target = case["target_body_text"]
        family = case["family"]
        synthetic_close = family == "finish_block"
        target_parse = tree_sitter_parse(
            projected_code(case, target, synthetic_close=synthetic_close)
        )
        output_parse = tree_sitter_parse(
            projected_code(case, output, synthetic_close=synthetic_close)
        )
        output_lines = output.splitlines()
        repetition = repeated_block(output_lines)
        review = PRIMARY_REVIEW[case_id]
        secondary = list(SECONDARY_REVIEW[case_id])
        if repetition is not None and "repetition" not in secondary:
            secondary.append("repetition")
        if not result["protocol_valid"] and "protocol_invalid" not in secondary:
            secondary.append("protocol_invalid")
        reviewed.append(
            {
                "id": case_id,
                "family": family,
                "package_id": case["package_id"],
                "path": case["context"]["path"],
                "group_id": case["group_id"],
                "strict_textual_mismatch": not bool(result["exact_region"]),
                "evaluator": {
                    "exact_region": bool(result["exact_region"]),
                    "protocol_valid": bool(result["protocol_valid"]),
                    "cap_hit": bool(result["cap_hit"]),
                    "failure": result.get("failure"),
                    "generated_tokens": result.get("generated_tokens"),
                    "prompt_tokens": result.get("prompt_tokens"),
                },
                "diagnostic_category": review,
                "secondary_flags": secondary,
                "target_lines": len(target.rstrip("\r\n").splitlines()),
                "generated_lines": len(output_lines),
                "target_sha256": case["target_sha256"],
                "generated_code_sha256": sha256_bytes(output.encode("utf-8")),
                "repetition": repetition,
                "projected_parse": {
                    "backend": "tree-sitter-r@canonical-.venv",
                    "synthetic_close_added": synthetic_close,
                    "target": target_parse,
                    "generated": output_parse,
                },
                "bounded_evidence": {
                    "target": bounded_excerpt(target),
                    "generated": bounded_excerpt(output),
                },
            }
        )

    category_counts = collections.Counter(row["diagnostic_category"] for row in reviewed)
    family_counts: dict[str, Any] = {}
    for family in TARGET_FAMILIES:
        family_rows = [row for row in reviewed if row["family"] == family]
        family_counts[family] = {
            "cases": len(family_rows),
            "exact_region": sum(not row["strict_textual_mismatch"] for row in family_rows),
            "strict_textual_mismatch": sum(
                row["strict_textual_mismatch"] for row in family_rows
            ),
            "protocol_valid": sum(row["evaluator"]["protocol_valid"] for row in family_rows),
            "cap_hit": sum(row["evaluator"]["cap_hit"] for row in family_rows),
            "categories": dict(collections.Counter(row["diagnostic_category"] for row in family_rows)),
        }

    return {
        "task": "SFT-06",
        "diagnostic": "bounded_dev_error_diagnostic",
        "status": "complete",
        "evidence_policy": {
            "panel": "DAT-07 final evaluator cases used as the locked 75-case DEV panel",
            "checkpoint": "step-500 only",
            "final_data_used": False,
            "cross_tokenizer_raw_loss_comparison": False,
            "diagnostic_is_second_sft_arm": False,
        },
        "inputs": {
            "cases_path": str(CASES_PATH),
            "cases_sha256": sha256_bytes(cases_bytes),
            "cases": len(cases),
            "step500_path": str(STEP500_PATH),
            "step500_sha256": sha256_bytes(step500_bytes),
            "step500_status": step500.get("status"),
            "step500": len(results),
            "step1000": inspect_step1000(),
        },
        "scope": {
            "families": list(TARGET_FAMILIES),
            "rows": len(reviewed),
            "row_ids": selected_ids,
            "finish_block": 6,
            "roxygen_drafting": 8,
        },
        "counts": {
            "rows": len(reviewed),
            "exact_region": sum(not row["strict_textual_mismatch"] for row in reviewed),
            "strict_textual_mismatch": sum(row["strict_textual_mismatch"] for row in reviewed),
            "protocol_valid": sum(row["evaluator"]["protocol_valid"] for row in reviewed),
            "protocol_invalid_or_capped": sum(
                not row["evaluator"]["protocol_valid"] for row in reviewed
            ),
            "repeated_block": sum(row["repetition"] is not None for row in reviewed),
            "categories": dict(category_counts),
        },
        "family_counts": family_counts,
        "rows": reviewed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="write JSON evidence to this path")
    args = parser.parse_args()
    result = analyse()
    rendered = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
