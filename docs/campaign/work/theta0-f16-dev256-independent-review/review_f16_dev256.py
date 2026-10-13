#!/usr/bin/env python3
"""Independent CPU-only review of the final theta0 F16 CUDA DEV run.

The reviewed DEV run is read after native execution has completed.  This
wrapper reuses the previously reviewed RUN-05 scorer for panel rendering,
HF/native prompt token checks, wire-token decoding, PRM-03 parsing, cap
classification and saved-decision checks.  It independently reviews all 75
F16 cases and the accepted Q8 CUDA b256/192 baseline, then compares the two
profiles by case.

No model bytes are opened or hashed.  The F16 model identity is checked only
against the immutable provenance string in quality.json and the expected
digest in the lead manifest.  The wrapper does not launch native, CUDA, SSH,
network or state-changing operations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
STATE_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training")
TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")

WORK_DIR = PLAN_ROOT / "docs/campaign/work/theta0-f16-dev256-independent-review"
RECEIPT_PATH = PLAN_ROOT / "docs/campaign/receipts/RUN-09-theta0-f16-dev256-independent-review.json"
BASE_REVIEW_SCRIPT = PLAN_ROOT / "docs/campaign/work/theta0-cuda-dev192-review-v1/review_dev192.py"
BASE_REVIEW_REPORT = PLAN_ROOT / "docs/campaign/work/theta0-cuda-dev192-review-v1/review-report.json"

PANEL = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
PANEL_MANIFEST = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/manifest.json"
PROTOCOL = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
CAMPAIGN_CLIENT = EXEC_ROOT / "extensions/vscode-sepalith/src/campaign_client.ts"
CAMPAIGN_EVAL = EXEC_ROOT / "experiments/training/campaign_eval.py"
REFERENCE_SCORER = PLAN_ROOT / "docs/campaign/work/quant-quality/run09_native_dev_quality.py"

Q8_CONTROLLER = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-dev192-a/run_dev192.py"
Q8_CAP_ADAPTER = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-dev192-a/run_dev192_client.py"
Q8_CONTROLLER_MANIFEST = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-dev192-a/manifest.json"
Q8_RUN = STATE_ROOT / "RUN-05-theta0-cuda-dev192-a/Q8_0-b256"

F16_LEAD_DIR = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-f16-dev256-a"
F16_CONTROLLER = F16_LEAD_DIR / "run_dev192.py"
F16_CAP_ADAPTER = F16_LEAD_DIR / "run_dev192_client.py"
F16_LEAD_MANIFEST = F16_LEAD_DIR / "manifest.json"
F16_RUN = STATE_ROOT / "RUN-05-theta0-cuda-f16-dev256-a/F16-b256"
F16_ROOT_TERMINAL = STATE_ROOT / "RUN-05-theta0-cuda-f16-dev256-a/terminal.json"

F16_MODEL_PATH = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/"
    "SFT-primary-step1000-runtime-gguf/model-F16.gguf"
)
F16_MODEL_SHA256 = "ee7ba2fd7e7b6446d74224e1e1f9e4724fbcfa54e56e60c4b623981d1e4f45ed"
SERVER_PATH = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/"
    "llama-cuda-b10453/llama-server"
)
SERVER_SHA256 = "e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee"

PANEL_ROWS = 75
EDIT_ROWS = 43
NOOP_ROWS = 32
CONTEXT_SIZE = 4096
CAP = 192
BATCH = 256
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = [1, 130073]
VOCAB_SIZE = 130560
BUILD_INFO = "b10453-3cb7ffb1a"


class ReviewError(RuntimeError):
    """An immutable input or independently checked result violated the contract."""


def load_base_review() -> Any:
    spec = importlib.util.spec_from_file_location("reviewed_dev192", BASE_REVIEW_SCRIPT)
    if spec is None or spec.loader is None:
        raise ReviewError(f"cannot load reviewed scorer {BASE_REVIEW_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(base: Any, path: Path) -> Any:
    return base.read_json(path)


def scalar_props(props: Mapping[str, Any]) -> dict[str, Any]:
    default = props.get("default_generation_settings", {})
    return {
        "build_info": props.get("build_info"),
        "model_ftype": props.get("model_ftype"),
        "model_path": props.get("model_path"),
        "n_ctx": default.get("n_ctx") if isinstance(default, Mapping) else None,
        "endpoint_props": props.get("endpoint_props"),
        "endpoint_metrics": props.get("endpoint_metrics"),
        "endpoint_slots": props.get("endpoint_slots"),
        "bos_token": props.get("bos_token"),
        "eos_token": props.get("eos_token"),
    }


def argv_value(argv: Sequence[Any], option: str) -> str | None:
    values = [str(value) for value in argv]
    try:
        index = values.index(option)
    except ValueError:
        return None
    return values[index + 1] if index + 1 < len(values) else None


def f16_server_diagnostics(base: Any, run: Path) -> dict[str, Any]:
    """Check the completed F16 server evidence without touching its model."""
    props_path = run / "props.json"
    launch_path = run / "server-launch.json"
    env_paths = sorted(run.glob("server-env-*.json"))
    children_path = run / "server-children.json"
    log_path = run / "server.log"
    props = read_json(base, props_path)
    launch = read_json(base, launch_path)
    terminal = read_json(base, run / "terminal.json")
    env = read_json(base, env_paths[0]) if len(env_paths) == 1 else None
    children = read_json(base, children_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    argv = launch.get("argv", [])
    if not isinstance(argv, list):
        argv = []
    batch = argv_value(argv, "-b")
    ubatch = argv_value(argv, "-ub")
    context_arg = argv_value(argv, "-c")
    graph_use = re.findall(r"USE_GRAPHS\s*=\s*(\d+)", log_text)
    graph_nodes = re.findall(r"graph nodes\s*=\s*(\d+)", log_text)
    graph_splits = re.findall(r"graph splits\s*=\s*(\d+)", log_text)
    offloads = re.findall(r"offloaded\s+(\d+/\d+)\s+layers", log_text)
    reused = re.findall(r"graphs reused\s*=\s*(\d+)", log_text)
    cleanups = re.findall(r"cleaning up before exit", log_text)
    compact = scalar_props(props)
    terminal_children = terminal.get("server", {}).get("children", [])
    checks = {
        "props_n_ctx_4096": compact["n_ctx"] == CONTEXT_SIZE,
        "props_model_ftype_f16": compact["model_ftype"] == "F16",
        "props_build_pinned": compact["build_info"] == BUILD_INFO,
        "props_endpoint_props_false": compact["endpoint_props"] is False,
        "props_model_path_pinned": compact["model_path"] == str(F16_MODEL_PATH),
        "launch_batch_equals_ubatch": batch == str(BATCH) and ubatch == str(BATCH),
        "launch_context_4096": context_arg == str(CONTEXT_SIZE),
        "launch_server_pinned": str(SERVER_PATH) in [str(value) for value in argv],
        "env_graph_opt_zero": env == ["GGML_CUDA_GRAPH_OPT=0"],
        "log_uses_cuda_graphs": graph_use == ["1"],
        "log_graph_geometry": graph_nodes == ["1308"] and graph_splits == ["2"],
        "log_graph_reuse": bool(reused),
        "log_offloaded_43_of_43": "43/43" in offloads,
        "log_clean_exit": bool(cleanups),
        "children_recorded": isinstance(children, list) and bool(children),
        "children_clean_after_exit": (
            isinstance(terminal_children, list)
            and all(
                child.get("exists") is False
                for child in terminal_children
                if isinstance(child, Mapping)
            )
        ),
    }
    return {
        "props_sha256": base.sha256_file(props_path),
        "server_launch_sha256": base.sha256_file(launch_path),
        "server_env_sha256": base.sha256_file(env_paths[0]) if len(env_paths) == 1 else None,
        "server_children_sha256": base.sha256_file(children_path),
        "server_log_sha256": base.sha256_file(log_path),
        "props": compact,
        "env": env,
        "children": children,
        "terminal_children": terminal_children,
        "argv": argv,
        "batch": batch,
        "ubatch": ubatch,
        "context_arg": context_arg,
        "graph_use_values": [int(value) for value in graph_use],
        "graph_nodes_values": [int(value) for value in graph_nodes],
        "graph_splits_values": [int(value) for value in graph_splits],
        "offloaded_values": offloads,
        "graph_reuse_samples": [int(value) for value in reused[-3:]],
        "graph_reuse_observations": len(reused),
        "cleanup_observations": len(cleanups),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def f16_source_checks(
    base: Any, quality: Mapping[str, Any], panel_audit: Mapping[str, Any],
    tokenizer_audit: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = quality.get("contract", {})
    completion = contract.get("completion", {}) if isinstance(contract, Mapping) else {}
    static = quality.get("static_preflight", {})
    panel = quality.get("panel", {})
    model = quality.get("model", {})
    server = quality.get("server", {})
    props = server.get("props", {}) if isinstance(server, Mapping) else {}
    tokenizer = quality.get("tokenizer", {})
    quality_tokenization = contract.get("tokenization", {}) if isinstance(contract, Mapping) else {}
    manifest_controller_sha = manifest.get(str(F16_CONTROLLER))
    manifest_adapter_sha = manifest.get(str(F16_CAP_ADAPTER))
    controller_sha = base.sha256_file(F16_CONTROLLER)
    adapter_sha = base.sha256_file(F16_CAP_ADAPTER)
    checks = {
        "quality_status_complete": quality.get("status") == "complete" and quality.get("evaluation_complete") is True,
        "quality_cap_matches": completion.get("n_predict") == CAP,
        "quality_context_4096": contract.get("context_size") == CONTEXT_SIZE,
        "quality_renderer_pinned": contract.get("renderer_id") == "zeta2-prm03-v1",
        "quality_manual_bos_zero": contract.get("manual_bos_id") == BOS_ID,
        "quality_native_eog_pinned": contract.get("native_eog_ids") == NATIVE_EOG_IDS,
        "quality_split_special_true": quality_tokenization.get("hf_split_special_tokens") is True,
        "quality_add_special_false": quality_tokenization.get("add_special") is False,
        "quality_parse_special_false": quality_tokenization.get("parse_special") is False,
        "quality_protocol_hash": static.get("protocol", {}).get("sha256") == base.EXPECTED["protocol_sha256"],
        "quality_campaign_client_hash": static.get("campaign_client", {}).get("sha256") == base.EXPECTED["campaign_client_sha256"],
        "quality_campaign_eval_hash": static.get("campaign_eval", {}).get("sha256") == base.EXPECTED["campaign_eval_sha256"],
        "quality_prompt_geometry_within_context": static.get("prompt_geometry", {}).get("all_within_context") is True,
        "quality_panel_hash": (
            panel.get("sha256") == base.EXPECTED["panel_sha256"]
            and panel.get("rows") == PANEL_ROWS
            and panel.get("case_ids_sha256") == panel_audit.get("ordered_ids_sha256")
        ),
        "quality_model_provenance": model.get("provenance") == F16_MODEL_SHA256,
        "quality_model_path": server.get("model_path_reported") == str(F16_MODEL_PATH),
        "quality_server_context": server.get("n_ctx") == CONTEXT_SIZE,
        "quality_server_props_ftype": props.get("model_ftype") == "F16",
        "quality_server_props_build": props.get("build_info") == BUILD_INFO,
        "quality_server_props_endpoint_false": props.get("endpoint_props") is False,
        "quality_tokenizer_hashes": tokenizer.get("files") == {
            "tokenizer.json": base.EXPECTED["tokenizer_json_sha256"],
            "tokenizer_config.json": base.EXPECTED["tokenizer_config_sha256"],
        },
        "quality_tokenizer_identity": tokenizer.get("identity") == tokenizer_audit.get("identity"),
        "quality_no_transport_failures": quality.get("denominators", {}).get("transport_failures") == 0,
        "quality_no_partial_responses": quality.get("denominators", {}).get("partial_responses") == 0,
        "lead_manifest_model_identity": manifest.get(str(F16_MODEL_PATH)) == F16_MODEL_SHA256,
        "lead_manifest_controller_identity": manifest_controller_sha == controller_sha,
        "lead_manifest_adapter_identity": manifest_adapter_sha == adapter_sha,
        "lead_controller_pins_output_192": "out192" in F16_CONTROLLER.read_text(encoding="utf-8"),
        "lead_adapter_pins_cap_192": "COMPLETION_CAP = 192" in F16_CAP_ADAPTER.read_text(encoding="utf-8"),
        "lead_controller_uses_f16_model": "model-F16.gguf" in F16_CONTROLLER.read_text(encoding="utf-8"),
    }
    return {
        "model": {
            "path": str(F16_MODEL_PATH),
            "expected_sha256": F16_MODEL_SHA256,
            "provenance_saved": model.get("provenance"),
            "lead_manifest_expected": manifest.get(str(F16_MODEL_PATH)),
            "bytes_read_or_hashed": False,
        },
        "server_binary": {
            "path": str(SERVER_PATH),
            "expected_sha256": SERVER_SHA256,
            "bytes_read_or_hashed": False,
            "identity_source": "lead manifest and saved server argv",
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def review_f16(
    base: Any, protocol: Any, tokenizer: Any,
    panel_items: Sequence[Mapping[str, Any]], panel_audit: Mapping[str, Any],
    tokenizer_audit: Mapping[str, Any],
) -> dict[str, Any]:
    quality_path = F16_RUN / "quality.json"
    quality = read_json(base, quality_path)
    cases = quality.get("cases")
    ordered_ids = [item["row"]["id"] for item in panel_items]
    if not isinstance(cases, list) or len(cases) != PANEL_ROWS:
        raise ReviewError(f"F16: quality case count is not {PANEL_ROWS}")
    if [case.get("id") for case in cases] != ordered_ids:
        raise ReviewError("F16: quality case order differs from corrected panel")
    records = [
        base.independent_case(protocol, tokenizer, item, case, CAP)
        for item, case in zip(panel_items, cases)
    ]
    independent = base._denominators(records)
    saved = quality.get("denominators", {})
    denominator_fields = [
        "attempted_rows", "responses", "panel_rows", "edit_cases", "strict_noop_cases",
        "protocol_rows", "protocol_error_rows", "cap_hit_rows", "exact_region_rows",
        "edit_exact_rows", "strict_noop_correct_rows", "noop_false_positive_rows",
        "mechanical_failures", "transport_failures", "partial_responses",
    ]
    denominator_comparison = {
        field: {
            "independent": independent.get(field),
            "saved": saved.get(field),
            "match": independent.get(field) == saved.get(field),
        }
        for field in denominator_fields
    }
    if not all(item["match"] for item in denominator_comparison.values()):
        raise ReviewError(f"F16: independent denominators differ from saved quality")
    saved_decisions_match = all(
        all(
            record["decisions"][key] == record["decisions"]["saved"][key]
            for key in (
                "protocol_valid", "exact_region", "edit_exact", "predicted_noop",
                "strict_noop_correct", "noop_false_positive",
            )
        )
        and record["cap"]["decision_match"]
        for record in records
    )
    if not saved_decisions_match:
        raise ReviewError("F16: one or more saved per-case decisions differ")
    manifest = read_json(base, F16_LEAD_MANIFEST)
    source_checks = f16_source_checks(
        base, quality, panel_audit, tokenizer_audit, manifest,
    )
    if not source_checks["all_checks_pass"]:
        raise ReviewError(f"F16 source checks failed: {source_checks['checks']}")
    terminal_checks = base._terminal_checks(F16_RUN, quality_path)
    diagnostics = f16_server_diagnostics(base, F16_RUN)
    return {
        "label": "theta0-F16-CUDA-b256-out192",
        "run_path": str(F16_RUN),
        "run_files": {
            "quality_sha256": base.sha256_file(quality_path),
            "quality_bytes": quality_path.stat().st_size,
            "client_launch_sha256": base.sha256_file(F16_RUN / "client-launch.json"),
            "server_launch_sha256": base.sha256_file(F16_RUN / "server-launch.json"),
            "terminal_sha256": base.sha256_file(F16_RUN / "terminal.json"),
            "server_log_sha256": base.sha256_file(F16_RUN / "server.log"),
            "client_log_sha256": base.sha256_file(F16_RUN / "client.log"),
        },
        "cap": CAP,
        "source_checks": source_checks,
        "terminal_checks": terminal_checks,
        "server_diagnostics": diagnostics,
        "independent_denominators": independent,
        "saved_denominators": {field: saved.get(field) for field in denominator_fields},
        "denominator_comparison": denominator_comparison,
        "records": records,
        "all_rows_independently_checked": len(records) == PANEL_ROWS,
        "all_saved_decisions_match": saved_decisions_match,
    }


def compact_side(record: Mapping[str, Any]) -> dict[str, Any]:
    generation = record["generation"]
    decisions = record["decisions"]
    return {
        "raw_text_sha256": generation["raw_text_sha256"],
        "returned_token_ids_sha256": generation["returned_token_ids_sha256"],
        "returned_tokens": generation["returned_tokens_including_terminal"],
        "terminal_id": generation["terminal_id"],
        "canonical_generation_tokens": generation["canonical_generation_tokens"],
        "stop_type": generation["stop_type"],
        "saw_stop": generation["saw_stop"],
        "parser_status": generation["parser_status"],
        "parser_operation": generation["parser_operation"],
        "parser_reason": generation["parser_reason"],
        "cap_hit": record["cap"]["hit"],
        "protocol_valid": decisions["protocol_valid"],
        "exact_region": decisions["exact_region"],
        "edit_exact": decisions["edit_exact"],
        "strict_noop_correct": decisions["strict_noop_correct"],
        "noop_false_positive": decisions["noop_false_positive"],
    }


def compare_profiles(
    q8_review: Mapping[str, Any], f16_review: Mapping[str, Any],
) -> dict[str, Any]:
    q8_by_id = {record["id"]: record for record in q8_review["records"]}
    f16_by_id = {record["id"]: record for record in f16_review["records"]}
    if set(q8_by_id) != set(f16_by_id):
        raise ReviewError("Q8 and F16 profiles do not have the same 75 case IDs")
    fields = [
        ("generation", "raw_text_sha256"),
        ("generation", "returned_token_ids_sha256"),
        ("generation", "returned_tokens_including_terminal"),
        ("generation", "terminal_id"),
        ("generation", "decoded_body_sha256"),
        ("generation", "wire_hf_text_match"),
        ("generation", "canonical_generation_tokens"),
        ("generation", "stop_type"),
        ("generation", "saw_stop"),
        ("generation", "truncated"),
        ("generation", "parser_status"),
        ("generation", "parser_operation"),
        ("generation", "parser_reason"),
        ("cap", "hit"),
        ("decisions", "protocol_valid"),
        ("decisions", "exact_region"),
        ("decisions", "edit_exact"),
        ("decisions", "predicted_noop"),
        ("decisions", "strict_noop_correct"),
        ("decisions", "noop_false_positive"),
    ]
    output_signature = {
        "generation.raw_text_sha256", "generation.returned_token_ids_sha256",
        "generation.returned_tokens_including_terminal", "generation.terminal_id",
        "generation.decoded_body_sha256",
    }
    decision_signature = {
        "cap.hit", "decisions.protocol_valid", "decisions.exact_region",
        "decisions.edit_exact", "decisions.predicted_noop",
        "decisions.strict_noop_correct", "decisions.noop_false_positive",
    }
    row_comparisons: list[dict[str, Any]] = []
    output_changed_ids: list[str] = []
    decision_changed_ids: list[str] = []
    field_counts: Counter[str] = Counter()
    for row_id in q8_by_id:
        q8 = q8_by_id[row_id]
        f16 = f16_by_id[row_id]
        changed: dict[str, dict[str, Any]] = {}
        for section, field in fields:
            q8_value = q8[section].get(field)
            f16_value = f16[section].get(field)
            if q8_value != f16_value:
                key = f"{section}.{field}"
                changed[key] = {"q8_b256": q8_value, "f16_b256": f16_value}
                field_counts[key] += 1
        output_changed = any(key in changed for key in output_signature)
        decision_changed = any(key in changed for key in decision_signature)
        if output_changed:
            output_changed_ids.append(row_id)
        if decision_changed:
            decision_changed_ids.append(row_id)
        row_comparisons.append({
            "id": row_id,
            "family": q8["family"],
            "operation": q8["operation"],
            "output_changed": output_changed,
            "decision_changed": decision_changed,
            "changed_fields": changed,
            "q8_b256": compact_side(q8),
            "f16_b256": compact_side(f16),
        })
    q8_denoms = q8_review["independent_denominators"]
    f16_denoms = f16_review["independent_denominators"]
    delta_fields = [
        "protocol_rows", "protocol_error_rows", "cap_hit_rows", "exact_region_rows",
        "edit_exact_rows", "strict_noop_correct_rows", "noop_false_positive_rows",
        "mechanical_failures",
    ]
    delta = {field: f16_denoms[field] - q8_denoms[field] for field in delta_fields}
    by_operation: dict[str, Any] = {}
    for operation in ("replace", "no_op"):
        ids = [row_id for row_id in q8_by_id if q8_by_id[row_id]["operation"] == operation]
        by_operation[operation] = {
            "rows": len(ids),
            "output_changed_rows": sum(row_id in output_changed_ids for row_id in ids),
            "decision_changed_rows": sum(row_id in decision_changed_ids for row_id in ids),
            "q8_b256": {
                "protocol_rows": sum(q8_by_id[row_id]["decisions"]["protocol_valid"] for row_id in ids),
                "exact_region_rows": sum(q8_by_id[row_id]["decisions"]["exact_region"] for row_id in ids),
                "edit_exact_rows": sum(q8_by_id[row_id]["decisions"]["edit_exact"] for row_id in ids),
                "strict_noop_correct_rows": sum(q8_by_id[row_id]["decisions"]["strict_noop_correct"] for row_id in ids),
                "noop_false_positive_rows": sum(q8_by_id[row_id]["decisions"]["noop_false_positive"] for row_id in ids),
            },
            "f16_b256": {
                "protocol_rows": sum(f16_by_id[row_id]["decisions"]["protocol_valid"] for row_id in ids),
                "exact_region_rows": sum(f16_by_id[row_id]["decisions"]["exact_region"] for row_id in ids),
                "edit_exact_rows": sum(f16_by_id[row_id]["decisions"]["edit_exact"] for row_id in ids),
                "strict_noop_correct_rows": sum(f16_by_id[row_id]["decisions"]["strict_noop_correct"] for row_id in ids),
                "noop_false_positive_rows": sum(f16_by_id[row_id]["decisions"]["noop_false_positive"] for row_id in ids),
            },
        }
    quality_buckets = {
        "edit_exact": {
            "denominator": EDIT_ROWS,
            "q8_b256": q8_denoms["edit_exact_rows"],
            "f16_b256": f16_denoms["edit_exact_rows"],
            "delta_f16_minus_q8": f16_denoms["edit_exact_rows"] - q8_denoms["edit_exact_rows"],
        },
        "strict_noop_correct": {
            "denominator": NOOP_ROWS,
            "q8_b256": q8_denoms["strict_noop_correct_rows"],
            "f16_b256": f16_denoms["strict_noop_correct_rows"],
            "delta_f16_minus_q8": f16_denoms["strict_noop_correct_rows"] - q8_denoms["strict_noop_correct_rows"],
        },
        "noop_false_positive": {
            "denominator": NOOP_ROWS,
            "q8_b256": q8_denoms["noop_false_positive_rows"],
            "f16_b256": f16_denoms["noop_false_positive_rows"],
            "delta_f16_minus_q8": f16_denoms["noop_false_positive_rows"] - q8_denoms["noop_false_positive_rows"],
        },
    }
    quality_regression = (
        f16_denoms["edit_exact_rows"] < q8_denoms["edit_exact_rows"]
        or f16_denoms["strict_noop_correct_rows"] < q8_denoms["strict_noop_correct_rows"]
        or f16_denoms["noop_false_positive_rows"] > q8_denoms["noop_false_positive_rows"]
    )
    protocol_or_cap_regression = (
        f16_denoms["protocol_rows"] < q8_denoms["protocol_rows"]
        or f16_denoms["cap_hit_rows"] > q8_denoms["cap_hit_rows"]
    )
    return {
        "profiles": [q8_review["label"], f16_review["label"]],
        "rows_compared": len(row_comparisons),
        "output_changed_rows": len(output_changed_ids),
        "output_changed_ids": output_changed_ids,
        "decision_changed_rows": len(decision_changed_ids),
        "decision_changed_ids": decision_changed_ids,
        "changed_field_counts": dict(sorted(field_counts.items())),
        "row_comparisons": row_comparisons,
        "aggregate_delta_f16_minus_q8_b256": delta,
        "by_operation": by_operation,
        "quality_buckets": quality_buckets,
        "quality_bucket_regression": quality_regression,
        "protocol_or_cap_regression": protocol_or_cap_regression,
        "interpretation": {
            "quality_buckets_match": not quality_regression,
            "f16_has_one_fewer_protocol_valid_row": f16_denoms["protocol_rows"] == q8_denoms["protocol_rows"] - 1,
            "f16_has_one_additional_cap_hit": f16_denoms["cap_hit_rows"] == q8_denoms["cap_hit_rows"] + 1,
            "native_diagnostic_outside_6000_source_budget": True,
            "promotion_owner": "root",
            "reviewer_does_not_promote": True,
        },
    }


def source_artifacts(base: Any, panel_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Inventory only source/run files needed by this review; never model bytes."""
    common = {
        "panel": PANEL,
        "panel_manifest": PANEL_MANIFEST,
        "protocol": PROTOCOL,
        "campaign_client": CAMPAIGN_CLIENT,
        "campaign_eval": CAMPAIGN_EVAL,
        "reference_scorer": REFERENCE_SCORER,
        "q8_controller": Q8_CONTROLLER,
        "q8_cap_adapter": Q8_CAP_ADAPTER,
        "q8_controller_manifest": Q8_CONTROLLER_MANIFEST,
        "reviewed_scorer": BASE_REVIEW_SCRIPT,
        "baseline_review_report": BASE_REVIEW_REPORT,
        "f16_controller": F16_CONTROLLER,
        "f16_cap_adapter": F16_CAP_ADAPTER,
        "f16_lead_manifest": F16_LEAD_MANIFEST,
    }
    source_paths = [
        {"name": name, "path": str(path), "sha256": base.sha256_file(path), "read": True}
        for name, path in common.items()
    ]
    run_files = [
        "quality.json", "terminal.json", "client-launch.json", "server-launch.json",
        "server-children.json", "server-env-23900.json", "server.log", "client.log",
        "server-maps-23900.txt",
    ]
    runs: dict[str, Any] = {}
    for label, run in (("q8_b256", Q8_RUN), ("f16_b256", F16_RUN)):
        entries = []
        for filename in run_files:
            path = run / filename
            if path.exists():
                entries.append({"path": str(path), "sha256": base.sha256_file(path), "read": True})
        runs[label] = entries
    if F16_ROOT_TERMINAL.exists():
        runs["f16_root_terminal"] = [{
            "path": str(F16_ROOT_TERMINAL),
            "sha256": base.sha256_file(F16_ROOT_TERMINAL),
            "read": True,
        }]
    return {
        "schema_version": "sepalith.run09.theta0-f16-dev256-review-inputs.v1",
        "purpose": "immutable input inventory for independent CPU-only F16 versus Q8 b256 review",
        "source_paths": source_paths,
        "runs": runs,
        "model": {
            "path": str(F16_MODEL_PATH),
            "expected_sha256": F16_MODEL_SHA256,
            "bytes_read_or_hashed": False,
            "identity_source": "quality.model.provenance and F16 lead manifest",
        },
        "q8_baseline_model": {
            "identity_source": "previously reviewed Q8 b256 report and saved quality provenance",
            "bytes_read_or_hashed": False,
        },
        "server_binary": {
            "path": str(SERVER_PATH),
            "expected_sha256": SERVER_SHA256,
            "bytes_read_or_hashed": False,
            "identity_source": "lead/controller manifest and saved server argv",
        },
        "panel_counts": {
            "rows": len(panel_items),
            "edit_rows": sum(int(item["row"]["operation"] == "replace") for item in panel_items),
            "noop_rows": sum(int(item["row"]["operation"] == "no_op") for item in panel_items),
        },
        "read_policy": [
            "CPU tokenizer, protocol, corrected panel, reviewed scorer, source controllers and completed run artifacts only",
            "no native launch, GPU, SSH, network, model-weight read or model-weight hash",
        ],
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_review()
    # The reviewed scorer's source pins are the common panel/protocol inputs.
    source_audit = base.source_artifact_checks()
    if not source_audit["all_checks_pass"]:
        raise ReviewError(f"common source checks failed: {source_audit['checks']}")
    protocol = base.load_protocol()
    tokenizer, tokenizer_audit = base.load_tokenizer()
    panel_items, panel_audit = base.load_panel(protocol, tokenizer)
    if len(panel_items) != PANEL_ROWS:
        raise ReviewError("corrected panel count is not 75")

    # Independently rerun the accepted Q8 b256/192 scorer from completed
    # artifacts.  Its model path is used only as an identity string by the
    # reviewed scorer; no model file is opened or hashed.
    q8_review = base.review_run(protocol, tokenizer, panel_items, Q8_RUN, CAP, "Q8_0-b256-out192")
    baseline_report = read_json(base, BASE_REVIEW_REPORT)
    baseline_profile = baseline_report.get("current_profiles", {}).get("b256", {})
    if baseline_profile.get("run_files", {}).get("quality_sha256") != q8_review["run_files"]["quality_sha256"]:
        raise ReviewError("Q8 baseline quality identity differs from reviewed baseline report")
    if baseline_profile.get("independent_denominators") != q8_review["independent_denominators"]:
        raise ReviewError("Q8 baseline denominators differ from reviewed baseline report")

    f16_review = review_f16(
        base, protocol, tokenizer, panel_items, panel_audit, tokenizer_audit,
    )
    comparison = compare_profiles(q8_review, f16_review)
    inputs = source_artifacts(base, panel_items)
    input_manifest_path = out_dir / "input-manifest.json"
    write_json(input_manifest_path, inputs)

    report = {
        "schema_version": "sepalith.run09.theta0-f16-dev256-independent-review.v1",
        "task": "RUN-09 independent theta0 F16 CUDA b256 DEV192 quality review",
        "status": "independent_review_complete_no_promotion",
        "review_policy": {
            "read_only_completed_artifacts": True,
            "cpu_tokenizer_and_protocol_only": True,
            "native_launch_performed": False,
            "gpu_or_ssh_or_network_used": False,
            "model_weight_read_or_hashed": False,
            "final_or_sealed_data_read": False,
        },
        "source_artifacts": source_audit,
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": base.sha256_file(input_manifest_path),
        },
        "panel": panel_audit,
        "tokenizer": tokenizer_audit,
        "model_identities": {
            "f16": {
                "label": "theta0-F16-CUDA-b256-out192",
                "path": str(F16_MODEL_PATH),
                "provenance_sha256": F16_MODEL_SHA256,
                "quality_provenance_sha256": f16_review["source_checks"]["model"]["provenance_saved"],
                "lead_manifest_sha256": f16_review["source_checks"]["model"]["lead_manifest_expected"],
                "bytes_read_or_hashed": False,
            },
            "q8_baseline": {
                "label": "Q8_0-b256-out192",
                "path": q8_review["source_checks"]["model"]["path"],
                "provenance_sha256": q8_review["source_checks"]["model"]["provenance_saved"],
                "bytes_read_or_hashed": False,
                "identity_source": "previously reviewed Q8 report and completed quality artifact",
            },
            "server": {
                "path": str(SERVER_PATH),
                "build_info": BUILD_INFO,
                "expected_sha256": SERVER_SHA256,
                "bytes_read_or_hashed": False,
            },
        },
        "q8_baseline_profile": q8_review,
        "f16_profile": f16_review,
        "paired_comparison_q8_b256_vs_f16_b256": comparison,
        "interpretation": {
            "same_75_case_ids": comparison["rows_compared"] == PANEL_ROWS,
            "paired_edit_noop_suggestion_buckets": comparison["quality_buckets"],
            "f16_quality_buckets_match_q8": comparison["quality_buckets"]["edit_exact"]["delta_f16_minus_q8"] == 0
            and comparison["quality_buckets"]["strict_noop_correct"]["delta_f16_minus_q8"] == 0
            and comparison["quality_buckets"]["noop_false_positive"]["delta_f16_minus_q8"] == 0,
            "f16_protocol_cap_tradeoff": {
                "protocol_rows_q8": q8_review["independent_denominators"]["protocol_rows"],
                "protocol_rows_f16": f16_review["independent_denominators"]["protocol_rows"],
                "cap_hits_q8": q8_review["independent_denominators"]["cap_hit_rows"],
                "cap_hits_f16": f16_review["independent_denominators"]["cap_hit_rows"],
            },
            "historical_f16_vulkan512_not_used": True,
            "native_diagnostic_outside_6000_source_budget": True,
            "promotion_owner": "root",
            "reviewer_does_not_promote": True,
        },
        "limits": [
            "The native response bodies are represented by hashes, lengths and independently decoded/parser metadata; raw model text is not repeated here.",
            "Native likelihood/NLL is unavailable from the endpoint.",
            "The paired comparison is one completed Q8 b256 and one completed F16 b256 run, each with 75 fresh output-192 cases; it is not a throughput benchmark or a historical F16 Vulkan512 comparison.",
            "A one-row protocol/cap regression is reported even though the three edit/no-op quality buckets are unchanged; root owns any promotion decision.",
        ],
    }
    report_path = out_dir / "review-report.json"
    write_json(report_path, report)
    report_sha = base.sha256_file(report_path)
    script_sha = base.sha256_file(Path(__file__).resolve())
    receipt = {
        "schema_version": "sepalith.receipt.run09-theta0-f16-dev256-independent-review.v1",
        "task": "RUN-09 independent theta0 F16 CUDA b256 DEV192 quality review",
        "status": report["status"],
        "review_script": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
        "input_manifest": {"path": str(input_manifest_path), "sha256": inputs and base.sha256_file(input_manifest_path)},
        "report": {"path": str(report_path), "sha256": report_sha},
        "scope": {
            "panel_rows": PANEL_ROWS,
            "edit_rows": EDIT_ROWS,
            "noop_rows": NOOP_ROWS,
            "context_size": CONTEXT_SIZE,
            "completion_cap": CAP,
            "batch": BATCH,
            "profiles": ["Q8_0-b256-out192", "theta0-F16-CUDA-b256-out192"],
            "historical_f16_vulkan512_used": False,
        },
        "identities": report["model_identities"],
        "results": {
            "q8_b256": q8_review["independent_denominators"],
            "f16_b256": f16_review["independent_denominators"],
            "output_changed_rows": comparison["output_changed_rows"],
            "output_changed_ids": comparison["output_changed_ids"],
            "decision_changed_rows": comparison["decision_changed_rows"],
            "decision_changed_ids": comparison["decision_changed_ids"],
            "quality_buckets": comparison["quality_buckets"],
            "aggregate_delta_f16_minus_q8_b256": comparison["aggregate_delta_f16_minus_q8_b256"],
        },
        "checks": {
            "all_75_f16_rows_independently_checked": f16_review["all_rows_independently_checked"],
            "all_75_q8_rows_independently_checked": q8_review["all_rows_independently_checked"],
            "all_f16_saved_decisions_match": f16_review["all_saved_decisions_match"],
            "all_q8_saved_decisions_match": q8_review["all_saved_decisions_match"],
            "f16_source_identity_and_props": f16_review["source_checks"]["all_checks_pass"],
            "f16_graph_opt0_and_native_diagnostics": f16_review["server_diagnostics"]["all_checks_pass"],
            "f16_clean_exit": f16_review["terminal_checks"]["clean_exit"],
            "q8_clean_exit": q8_review["terminal_checks"]["clean_exit"],
            "common_source_panel_tokenizer_checks": source_audit["all_checks_pass"] and panel_audit["target_geometry_checked"],
            "model_bytes_read_or_hashed": False,
            "promotion_left_to_root": True,
        },
        "read_policy": report["review_policy"],
    }
    write_json(RECEIPT_PATH, receipt)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "report_sha256": report_sha,
        "receipt": str(RECEIPT_PATH),
        "receipt_sha256": base.sha256_file(RECEIPT_PATH),
        "q8_b256": q8_review["independent_denominators"],
        "f16_b256": f16_review["independent_denominators"],
        "output_changed_rows": comparison["output_changed_rows"],
        "output_changed_ids": comparison["output_changed_ids"],
        "decision_changed_rows": comparison["decision_changed_rows"],
        "decision_changed_ids": comparison["decision_changed_ids"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
