#!/usr/bin/env python3
"""Counterfactual brace-boundary sensitivity on retained DEV predictions.

Only the six finish_block DEV IDs are rejoined with their retained prediction
records.  The original panel, evaluation JSON, protocol fields, and scores are
never modified.  The counterfactual changes the target text by appending one
outer closing brace to the source-derived finish boundary; generated output,
EOS/cap fields, and all other 69 rows remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PANEL = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl")
PANEL_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
FINISH_REPORT = ROOT / "docs/campaign/work/finish-dev-impact/finish-impact-report.json"
FINISH_REPORT_SHA256 = "d61bb63eab534d5ea586073289a8057b8cd4444c16dd59a4b54fb886e4cc152f"
PROTO = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/"
    "packages/sepalith/src/sepalith/campaign_protocol.py"
)
SCENARIOS = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py")

FINISH_IDS = (
    "e623a61b5a4c066358a477f2",
    "4f08633513b5c525240d2540",
    "d11581e9cfa4e3971aa1466e",
    "157517ba47dbab157f7c361a",
    "f43de3e77f2d92ed7b223464",
    "04834fef4fe59742f13677a9",
)

# These paths and hashes are the retained case artifacts named by the accepted
# SFT/RL receipts.  No checkpoint weights or training state are opened.
ARTIFACTS = (
    {
        "label": "SFT500",
        "stage": "SFT",
        "step": 500,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-a/evaluations/cases-step-500.json"),
        "sha256": "3e06b8990ab7808dc0d5683ba108425d450391114e76f35399cbf1377a2c012a",
        "pin_receipts": (
            ("docs/campaign/receipts/SFT-05-sft-dev-readout.json", "e0b35b65a6be29e6980223bc5a00c7fb6e6cfc982cdb68273c4dd42d78b4b080"),
        ),
    },
    {
        "label": "SFT1000",
        "stage": "SFT",
        "step": 1000,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-a/evaluations/cases-step-1000.json"),
        "sha256": "1eddac3b78251e27bce4465b86637a021c4ff34569df51a297bac52403aea4ea",
        "pin_receipts": (
            ("docs/campaign/receipts/SFT-06-dev-error-diagnostic.json", "babb3f0654ef37806c4349567c098f425bc874adb6edbf25068ad3aad160d13c"),
            ("docs/campaign/receipts/SFT-10-step1000-review.json", "f76d9b6d60af3edc2fbfc29aba6eb68bc778ca4794650bd2a7c8834fd1372b69"),
        ),
    },
    {
        "label": "SFT2000",
        "stage": "SFT",
        "step": 2000,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-b/evaluations/cases-step-2000.json"),
        "sha256": "d538286c7ab531e0aaf5de86a161fd0c216604beec3fa9686a6d284052f69ff2",
        "pin_receipts": (
            ("docs/campaign/receipts/SFT-10-step2000-review.json", "841ad5038ac715e5775e7dcab524f9dd3e3803aefe7b060f735f75004fca8c44"),
            ("docs/campaign/work/lead/SFT-10-step2000-evidence.json", "c11540bc92a3627a1e57548fe9c9c5f64c03f22ea85b639655dc039560cee9c7"),
        ),
    },
    {
        "label": "SFT2500",
        "stage": "SFT",
        "step": 2500,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-c/evaluations/cases-step-2500.json"),
        "sha256": "6dc652e33ee694f15e52ac329267573aa59867747fb04e8bf3929671ef2990e5",
        "pin_receipts": (
            ("docs/campaign/receipts/SFT-10-step2500-terminal-review.json", "660917c37dbb1f09f7d9a0531d7712b6c5e770e1a0853bddc6cf32840b1cc54a"),
        ),
    },
    {
        "label": "RL25",
        "stage": "RL",
        "step": 25,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-b/archive/evaluations/cases-step-25.json"),
        "sha256": "fdfd724326485b0d5f92e3ad3b3b255bdb072735858a15d1593f9e3b1402dd96",
        "pin_receipts": (
            ("docs/campaign/receipts/RL-08-step25-independent-review.json", "77c780304edcb8b4e92d62dc26d4b175987daa5b808040e165a5729f9ff78193"),
        ),
    },
    {
        "label": "RL50",
        "stage": "RL",
        "step": 50,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c/archive/evaluations/cases-step-50.json"),
        "sha256": "c12f784366172c7dda63d38fd6ade3f072e19221678497474a3a2788ccb05833",
        "pin_receipts": (
            ("docs/campaign/receipts/RL-08-step50-independent-review.json", "748e83355dfde16e443c3a1650e836ca4486914e0a47492fc554a2a955b46ea4"),
        ),
    },
    {
        "label": "RL75",
        "stage": "RL",
        "step": 75,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c/archive/evaluations/cases-step-75.json"),
        "sha256": "1e93a0a22774b56d5a058cc96f3091e023f1505d9a711ade32986b763296415c",
        "pin_receipts": (
            ("docs/campaign/receipts/RL-08-step75-lead-decision.json", "b40fb2de181ef887b8ec11e687265f47ebf15ce243016e178d5bb3f9e8346b3d"),
        ),
    },
    {
        "label": "RL100",
        "stage": "RL",
        "step": 100,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c/archive/evaluations/cases-step-100.json"),
        "sha256": "fa24d3bfbd4c5fa73c7710eacc77157edf5cedadbb476ee7122f223c3c567fe4",
        "pin_receipts": (
            ("docs/campaign/receipts/RL-08-step100-independent-review.json", "2f836600e9566a1cde0a1272189107277c5588c315231cdc0d5ea65a6841c76b"),
        ),
    },
    {
        "label": "RL125",
        "stage": "RL",
        "step": 125,
        "path": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-e/archive/evaluations/cases-step-125.json"),
        "sha256": "878c8d8983b5f0903a36f6bef6d24c15067878526380ce4f1abf835c676be930",
        "pin_receipts": (
            ("docs/campaign/receipts/RL-08-step125-independent-review.json", "b27563bca5f15db6142546938646c87c34e44c30fe90b5a445bd8325d67a2632"),
        ),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_panel() -> dict[str, dict[str, Any]]:
    digest = hashlib.sha256()
    selected: dict[str, dict[str, Any]] = {}
    with PANEL.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if line_number in range(70, 76):
                row = json.loads(raw.decode("utf-8"))
                selected[row["id"]] = row
    if digest.hexdigest() != PANEL_SHA256:
        raise RuntimeError("DEV panel changed")
    if tuple(selected) != FINISH_IDS:
        raise RuntimeError("selected finish IDs/order changed")
    return selected


def reconstruct_document(
    row: dict[str, Any], body_text: str, protocol: Any
) -> tuple[str, str]:
    context = row["context"]
    replacement_range = context["replacement_range"]
    lines = list(context["prefix"]) + list(context["region_old"]) + list(context["suffix_lines"])
    while len(lines) <= replacement_range["end"]["line"]:
        lines.append("")
    before = "\n".join(lines)
    if hashlib.sha256(before.encode("utf-8")).hexdigest() != replacement_range["content_sha256"]:
        raise RuntimeError(f"{row['id']} pre-edit content identity changed")

    def offset(position: dict[str, int]) -> int:
        line = lines[position["line"]]
        codepoint_column = protocol.utf16_to_codepoint_column(line, position["character"])
        return sum(len(value) + 1 for value in lines[: position["line"]]) + codepoint_column

    start = offset(replacement_range["start"])
    end = offset(replacement_range["end"])
    expected_region = "\n".join(context["region_old"])
    if before[start:end] != expected_region:
        raise RuntimeError(f"{row['id']} UTF-16 region identity changed")
    return before, before[:start] + body_text + before[end:]


def parse_r(parser: Any, text: str) -> bool:
    return not parser.parse(text.encode("utf-8")).root_node.has_error


def verify_pin_receipts(spec: dict[str, Any]) -> list[dict[str, str]]:
    result = []
    for relative, expected in spec["pin_receipts"]:
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"receipt pin changed: {path}")
        result.append({"path": str(path), "sha256": actual})
    return result


def inspect_artifact(
    spec: dict[str, Any], rows: dict[str, dict[str, Any]], protocol: Any, parser: Any
) -> dict[str, Any]:
    path = spec["path"]
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise RuntimeError(f"retained evaluation artifact changed: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 75:
        raise RuntimeError(f"{spec['label']} does not contain exactly 75 results")
    by_id = {result.get("id"): result for result in results}
    if any(case_id not in by_id for case_id in FINISH_IDS):
        raise RuntimeError(f"{spec['label']} is missing a selected finish ID")
    summary = payload.get("summary", {})
    if summary.get("panel_sha256") not in (None, PANEL_SHA256):
        raise RuntimeError(f"{spec['label']} panel identity differs")

    finish: list[dict[str, Any]] = []
    old_finish_exact = 0
    corrected_finish_exact = 0
    cap_count = 0
    protocol_consistent = 0
    prediction_parse_before_brace = 0
    prediction_parse_after_brace = 0
    for case_id in FINISH_IDS:
        row = rows[case_id]
        result = by_id[case_id]
        context = protocol.PromptContext.from_mapping(row["context"])
        raw_output = result.get("raw_output") or ""
        parsed = protocol.parse_output(raw_output, context)
        artifact_protocol_valid = bool(result.get("protocol_valid"))
        actual_protocol_valid = parsed.status == "accepted"
        protocol_consistent += int(artifact_protocol_valid == actual_protocol_valid)
        old_exact = bool(result.get("exact_region"))
        old_finish_exact += int(old_exact)
        cap_count += int(bool(result.get("cap_hit")))
        target_text = row["target_body_text"]
        corrected_target_text = target_text + "}"
        _before, target_after = reconstruct_document(row, target_text, protocol)
        _before, corrected_target_after = reconstruct_document(
            row, corrected_target_text, protocol
        )
        target_parse_ok = parse_r(parser, target_after)
        corrected_target_parse_ok = parse_r(parser, corrected_target_after)
        if target_parse_ok:
            raise RuntimeError(f"{case_id} original finish target unexpectedly parses")
        if not corrected_target_parse_ok:
            raise RuntimeError(f"{case_id} corrected target did not parse")

        parsed_body = parsed.body_text if actual_protocol_valid else None
        corrected_exact = bool(
            artifact_protocol_valid
            and actual_protocol_valid
            and parsed.operation == "replace"
            and parsed_body == corrected_target_text
        )
        corrected_finish_exact += int(corrected_exact)
        prediction_before_ok = None
        prediction_after_ok = None
        if artifact_protocol_valid and actual_protocol_valid and parsed.operation == "replace":
            _before, prediction_after = reconstruct_document(row, parsed_body or "", protocol)
            prediction_before_ok = parse_r(parser, prediction_after)
            prediction_after_ok = parse_r(parser, prediction_after + "}")
            prediction_parse_before_brace += int(prediction_before_ok)
            prediction_parse_after_brace += int(prediction_after_ok)
        finish.append(
            {
                "id": case_id,
                "family": row["family"],
                "package_id": row["package_id"],
                "prompt_sha256": row["prompt_sha256"],
                "target_sha256": row["target_sha256"],
                "target_body_sha256": row["source_provenance"]["target_body_sha256"],
                "target_body_chars": len(target_text),
                "target_body_tokens": row["target_body_token_count"],
                "artifact_protocol_valid": artifact_protocol_valid,
                "parsed_protocol_status": parsed.status,
                "parsed_protocol_reason": parsed.reason,
                "parsed_operation": parsed.operation,
                "generated_tokens": result.get("generated_tokens"),
                "cap_hit": bool(result.get("cap_hit")),
                "old_exact_region": old_exact,
                "counterfactual_exact_region_after_one_brace": corrected_exact,
                "target_reconstruction_r_parse": target_parse_ok,
                "corrected_target_reconstruction_r_parse": corrected_target_parse_ok,
                "prediction_body_chars": len(parsed_body) if parsed_body is not None else None,
                "prediction_body_sha256": (
                    hashlib.sha256(parsed_body.encode("utf-8")).hexdigest()
                    if parsed_body is not None
                    else None
                ),
                "prediction_reconstruction_r_parse": prediction_before_ok,
                "prediction_reconstruction_plus_one_brace_r_parse": prediction_after_ok,
                "failure": result.get("failure"),
            }
        )

    computed_total_exact = sum(bool(r.get("exact_region")) for r in results)
    counts = summary.get("counts", {})
    summary_total_exact = counts.get("exact_region")
    if summary_total_exact is not None and int(summary_total_exact) != computed_total_exact:
        raise RuntimeError(
            f"{spec['label']} summary exact count disagrees with rows: "
            f"{summary_total_exact} != {computed_total_exact}"
        )
    old_total_exact = computed_total_exact
    nonfinish_exact = old_total_exact - old_finish_exact
    corrected_total_exact = nonfinish_exact + corrected_finish_exact
    artifact_protocol_count = sum(
        int(bool(by_id[case_id].get("protocol_valid"))) for case_id in FINISH_IDS
    )
    return {
        "label": spec["label"],
        "stage": spec["stage"],
        "step": spec["step"],
        "artifact": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
            "status": payload.get("status"),
            "result_count": len(results),
            "panel_sha256": summary.get("panel_sha256"),
        },
        "pin_receipts": verify_pin_receipts(spec),
        "denominators": {
            "all_cases": 75,
            "finish_cases": 6,
            "other_cases_unchanged": 69,
            "edits": 43,
            "strict_noop": 32,
            "protocol_consistency": f"{protocol_consistent}/6",
            "protocol_valid": f"{artifact_protocol_count}/6",
        },
        "counts": {
            "old_total_exact_region": old_total_exact,
            "old_finish_exact_region": old_finish_exact,
            "old_other_exact_region": nonfinish_exact,
            "counterfactual_finish_exact_region": corrected_finish_exact,
            "counterfactual_total_exact_region": corrected_total_exact,
            "counterfactual_delta_total": corrected_total_exact - old_total_exact,
            "protocol_valid_finish": artifact_protocol_count,
            "cap_hit_finish": cap_count,
            "prediction_r_parse_before_brace": prediction_parse_before_brace,
            "prediction_r_parse_after_one_brace": prediction_parse_after_brace,
            "corrected_target_r_parse": 6,
        },
        "cases": finish,
    }


def audit() -> dict[str, Any]:
    if sha256_file(FINISH_REPORT) != FINISH_REPORT_SHA256:
        raise RuntimeError("finish impact report changed")
    finish_report = json.loads(FINISH_REPORT.read_text(encoding="utf-8"))
    if tuple(case["id"] for case in finish_report["cases"]) != FINISH_IDS:
        raise RuntimeError("finish report selected IDs/order changed")
    rows = read_panel()
    protocol = load_module("dat07_rescore_protocol", PROTO)
    scenarios = load_module("dat07_rescore_scenarios", SCENARIOS)
    checkpoint_results = [inspect_artifact(spec, rows, protocol, scenarios.parser) for spec in ARTIFACTS]
    ordering = [
        {
            "label": item["label"],
            "stage": item["stage"],
            "step": item["step"],
            "old_total_exact_region": item["counts"]["old_total_exact_region"],
            "counterfactual_total_exact_region": item["counts"]["counterfactual_total_exact_region"],
        }
        for item in checkpoint_results
    ]
    return {
        "task": "DAT-07/SFT-08/RL-08",
        "status": "counterfactual_finish_brace_sensitivity_complete",
        "scope": "retained predictions for exactly six finish_block DEV IDs; no model evaluation or artifact mutation",
        "inputs": {
            "panel": str(PANEL),
            "panel_sha256": PANEL_SHA256,
            "finish_report": str(FINISH_REPORT),
            "finish_report_sha256": FINISH_REPORT_SHA256,
            "finish_ids": list(FINISH_IDS),
        },
        "parser": {
            "protocol_path": str(PROTO),
            "protocol_sha256": sha256_file(PROTO),
            "r_parser_path": str(SCENARIOS),
            "r_parser_sha256": sha256_file(SCENARIOS),
            "utf16_range_conversion": "campaign_protocol.utf16_to_codepoint_column",
            "counterfactual_rule": "append exactly one } to target_body_text after its existing final LF; preserve all other target bytes",
        },
        "checkpoints": checkpoint_results,
        "ordering": {
            "old_vs_counterfactual": ordering,
            "sensitive": False,
            "reason": "corrected finish exactness is 0/6 at every retained checkpoint, so total exact_region counts and their ordering are unchanged",
            "interpretation_limit": "This tests exactness ordering only; it does not establish useful editor quality or choose a checkpoint.",
        },
        "protocol_and_cap_policy": {
            "generated_outputs_unchanged": True,
            "protocol_eos_cap_fields_unchanged": True,
            "protocol_invalid_or_cap_rows_excluded_from_corrected_exactness": True,
            "other_69_scores_unchanged": True,
        },
        "uncertainties": [
            "The six DEV contexts are source-derived finish boundary fragments; no raw source corpus bytes were opened.",
            "Appending one brace is a counterfactual diagnostic, not a replacement score or panel mutation.",
            "R parse success after one brace does not prove editor application or package semantic validity.",
            "Only retained SFT500/1000/2000/2500 and RL25/50/75/100/125 case artifacts were checked; no other checkpoints are inferred.",
        ],
        "validation": {
            "model_weights_read": False,
            "gpu_used": False,
            "ssh_or_network_used": False,
            "final_data_read": False,
            "original_dev_mutated": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = audit()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite existing report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fd = os.open(args.output.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
