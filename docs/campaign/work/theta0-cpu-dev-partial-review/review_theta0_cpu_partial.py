#!/usr/bin/env python3
"""Independent, bounded audit of the completed part of RUN-09 CPU DEV.

This reads saved case records and pinned protocol/tokenizer sources.  It does
not load model weights, start a server, or attempt the unobserved panel rows.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
RUN = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-theta0-q8-notebook-cpu-dev-a")
CUDA = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-theta0-q8-cuda-dev-a/quality.json")
VULKAN = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-theta0-q8-notebook-dev-a/quality.json")
PANEL = PLAN / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
PINNED_REVIEW = PLAN / "docs/campaign/work/theta0-q8-cuda-dev-review/review_theta0_q8_cuda_dev.py"
TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
EXPECTED = {
    "review_sha256": "6f69669f77fb02fdc85024b8256ad72c53067d8f07bbe163b0dc13501fa9af7d",
    "panel_sha256": "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035",
    "protocol_sha256": "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
    "tokenizer_json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
    "model_sha256": "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559",
    "runtime_sha256": "0569e1707d79bb3f8f5f39a9a9e63d1fcb1f6f5cf613e8a5f07bbae8ddb1968e",
    "runtime_binary_sha256": "e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6",
    "panel_case_ids_sha256": "6e0c4c5bd7de4f0d9aa6eab602796c13da110f56edab6b0dd8a9e04fb66ee6bb",
    "cuda_quality_sha256": "3d7bbd56c8f434b4bdc3de784b217feee5da338487aca3090d46fe7566a4a29f",
    "vulkan_quality_sha256": "e5d8c1d0119ae26047f5de343ec1adbff0fcbe50104550483d22e42178c7def2",
}
EXPECTED_COLLECTION = {
    "client.log": (1007110, "6b2f29ab4cddbbde3b17169a521018f7e777a9bbd4cfa2ffae4c6a14f610ebef"),
    "preflight.json": (3345, "3c90c9c04f1511d446c88c6bf6d68472ff4007d26f60fefa094b07747e931999"),
    "quality.json": (2228320, "5b5ba385de678dff1ddd50832241f7b0e78e032f73e63ea0658a165328f471d9"),
    "remote-launch.json": (1381, "6f4002c69a3090dbe1dd75d64f4ddd8ad81fefa1592202997cda63375aeaa34e"),
    "remote-live-device-audit.json": (126, "fc4a7ea71757ef683a5933176c8033b1c8507fa1648dda8aadc37f88efda6e94"),
    "remote-ready.json": (354, "4a862fe5b1c8e9f63b26b705a9a12bd7f6c0316e8747cb3140d750eafa6c2c87"),
    "remote-server.log": (273483, "ebecc1b65cee9f2a776cab9675ae3bb3d46d5731c19ae44f90b56577d97228ce"),
    "remote-terminal.json": (1599, "facd705cce36a5ad7009bab060c7c0f387c6cec71a5bfa56bd84ca1632f0b748"),
    "ssh.stderr": (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "terminal.json": (2860, "b65d513942b92e036ef95d379547bc800f5a83ffae741d01b725467a407454c9"),
}
IDENTITY_FIELDS = (
    "hf_prompt_ids_sha256", "native_prompt_ids_sha256",
    "hf_prompt_tokens_with_bos", "prompt_sha256", "target_sha256",
    "prompt_token_identity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def load_pinned_review() -> Any:
    actual = sha256_file(PINNED_REVIEW)
    if actual != EXPECTED["review_sha256"]:
        raise RuntimeError(f"pinned review hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("pinned_q8_review", PINNED_REVIEW)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned review")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_subset_metrics(quality: Mapping[str, Any], ids: list[str],
                         panel_by_id: Mapping[str, Any]) -> dict[str, int]:
    cases = {case.get("id"): case for case in quality.get("cases", [])
             if isinstance(case, Mapping)}
    counts = Counter()
    for row_id in ids:
        case = cases[row_id]
        q = case.get("quality", {})
        cap = case.get("cap", {})
        row = panel_by_id[row_id]
        valid = bool(q.get("protocol_valid"))
        exact = bool(q.get("exact_edit"))
        noop_correct = bool(q.get("strict_noop_correct"))
        counts["rows"] += 1
        counts["protocol_rows"] += int(valid)
        counts["protocol_error_rows"] += int(not valid)
        counts["cap_hit_rows"] += int(cap.get("hit") is True)
        counts["edit_exact_rows"] += int(exact and row.get("operation") != "no_op")
        counts["exact_region_rows"] += int(exact or noop_correct)
        counts["strict_noop_correct_rows"] += int(noop_correct)
        counts["noop_false_positive_rows"] += int(bool(q.get("noop_false_positive")))
    return dict(sorted(counts.items()))


def result_counts(results: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    counts = Counter()
    families: dict[str, Counter[str]] = {}
    for result in results:
        family = str(result["family"])
        counts["rows"] += 1
        counts["protocol_rows"] += int(result["protocol_valid_recomputed"])
        counts["protocol_error_rows"] += int(not result["protocol_valid_recomputed"])
        counts["cap_hit_rows"] += int(result["cap_hit"])
        counts["exact_region_rows"] += int(result["exact_recomputed"])
        if result["operation"] == "no_op":
            counts["strict_noop_correct_rows"] += int(
                result["protocol_valid_recomputed"] and result["predicted_noop_recomputed"])
            counts["noop_false_positive_rows"] += int(
                result["protocol_valid_recomputed"] and not result["predicted_noop_recomputed"])
        else:
            counts["edit_exact_rows"] += int(result["exact_recomputed"])
        fc = families.setdefault(family, Counter())
        fc["rows"] += 1
        fc["protocol_rows"] += int(result["protocol_valid_recomputed"])
        fc["cap_hit_rows"] += int(result["cap_hit"])
        fc["exact_region_rows"] += int(result["exact_recomputed"])
        if result["operation"] == "no_op":
            fc["strict_noop_correct_rows"] += int(
                result["protocol_valid_recomputed"] and result["predicted_noop_recomputed"])
            fc["noop_false_positive_rows"] += int(
                result["protocol_valid_recomputed"] and not result["predicted_noop_recomputed"])
    return dict(sorted(counts.items())), {
        name: dict(sorted(value.items())) for name, value in sorted(families.items())
    }


def compact_partial(case: Mapping[str, Any]) -> dict[str, Any]:
    timing = case.get("timing") if isinstance(case.get("timing"), Mapping) else {}
    server_timing = case.get("server_timing") if isinstance(case.get("server_timing"), Mapping) else {}
    return {
        "id": case.get("id"), "family": case.get("family"),
        "package_id": case.get("package_id"), "status": case.get("status"),
        "response_received": case.get("response_received"),
        "response_complete": case.get("response_complete"),
        "failure_class": case.get("failure_class"),
        "returned_tokens": len(case.get("returned_token_ids", []))
        if isinstance(case.get("returned_token_ids"), list) else None,
        "protocol_status": case.get("protocol", {}).get("valid")
        if isinstance(case.get("protocol"), Mapping) else None,
        "timing": {key: timing.get(key) for key in ("wall_ms", "deadline_seconds")},
        "server_timing_keys": sorted(server_timing.keys()),
    }


def compare_backend(cpu_cases: list[Mapping[str, Any]], other_quality: Mapping[str, Any],
                    other_label: str, panel_by_id: Mapping[str, Any]) -> dict[str, Any]:
    other = {case.get("id"): case for case in other_quality.get("cases", [])
             if isinstance(case, Mapping)}
    ids = [str(case["id"]) for case in cpu_cases]
    identity_mismatches = {field: [] for field in IDENTITY_FIELDS}
    token_mismatches: list[str] = []
    text_mismatches: list[str] = []
    protocol_mismatches: list[str] = []
    cap_mismatches: list[str] = []
    for case in cpu_cases:
        row_id = str(case["id"])
        counterpart = other.get(row_id)
        if counterpart is None:
            for field in IDENTITY_FIELDS:
                identity_mismatches[field].append(row_id)
            token_mismatches.append(row_id)
            text_mismatches.append(row_id)
            continue
        for field in IDENTITY_FIELDS:
            if case.get(field) != counterpart.get(field):
                identity_mismatches[field].append(row_id)
        if case.get("returned_token_ids") != counterpart.get("returned_token_ids"):
            token_mismatches.append(row_id)
        if case.get("raw_text") != counterpart.get("raw_text"):
            text_mismatches.append(row_id)
        if case.get("quality", {}).get("protocol_valid") != counterpart.get("quality", {}).get("protocol_valid"):
            protocol_mismatches.append(row_id)
        if case.get("cap", {}).get("hit") != counterpart.get("cap", {}).get("hit"):
            cap_mismatches.append(row_id)
    return {
        "label": other_label,
        "rows_compared": len(ids),
        "identity_mismatches": identity_mismatches,
        "identity_mismatch_counts": {field: len(value) for field, value in identity_mismatches.items()},
        "generated_token_id_matches": len(ids) - len(token_mismatches),
        "generated_token_id_mismatch_ids": token_mismatches,
        "generated_text_matches": len(ids) - len(text_mismatches),
        "generated_text_mismatch_ids": text_mismatches,
        "protocol_mismatch_ids": protocol_mismatches,
        "cap_mismatch_ids": cap_mismatches,
        "saved_subset_metrics": saved_subset_metrics(other_quality, ids, panel_by_id),
    }


def main() -> int:
    collection = load_json(RUN / "collection-manifest.json")
    declared = collection.get("files")
    if not isinstance(declared, Mapping) or set(declared) != set(EXPECTED_COLLECTION):
        raise RuntimeError("collection manifest does not declare the expected ten files")
    collection_checks: dict[str, Any] = {}
    for name, (expected_bytes, expected_sha) in EXPECTED_COLLECTION.items():
        path = RUN / name
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        manifest_entry = declared[name]
        collection_checks[name] = {
            "bytes": actual_bytes, "sha256": actual_sha,
            "matches_manifest": manifest_entry == {"bytes": expected_bytes, "sha256": expected_sha},
            "matches_expected": actual_bytes == expected_bytes and actual_sha == expected_sha,
        }
        if not collection_checks[name]["matches_expected"]:
            raise RuntimeError(f"collection mismatch: {name}")

    quality = load_json(RUN / "quality.json")
    terminal = load_json(RUN / "terminal.json")
    launch = load_json(RUN / "remote-launch.json")
    ready = load_json(RUN / "remote-ready.json")
    live = load_json(RUN / "remote-live-device-audit.json")
    remote_terminal = load_json(RUN / "remote-terminal.json")
    preflight = load_json(RUN / "preflight.json")
    pinned = load_pinned_review()
    rows, panel_by_id = pinned.load_panel()
    protocol = pinned.load_protocol()
    tokenizer, tokenizer_audit = pinned.load_tokenizer()
    ordered_panel_ids = [str(row["id"]) for row in rows]
    cases = quality.get("cases")
    if not isinstance(cases, list) or len(cases) != 35:
        raise RuntimeError("partial quality must contain 35 attempted records")
    case_ids = [str(case.get("id")) for case in cases]
    if case_ids != ordered_panel_ids[:35]:
        raise RuntimeError("partial case order is not the corrected DEV panel prefix")
    complete = [case for case in cases if case.get("response_complete") is True
                and case.get("response_received") is True]
    partial = [case for case in cases if case not in complete]
    if len(complete) != 34 or len(partial) != 1:
        raise RuntimeError("expected 34 complete cases and one partial case")
    complete_ids = [str(case["id"]) for case in complete]
    if complete_ids != ordered_panel_ids[:34]:
        raise RuntimeError("complete case order is not the corrected DEV panel prefix")
    independent = []
    saved_case_mismatches: list[str] = []
    for case in complete:
        result = pinned.independent_case(protocol, tokenizer, case, panel_by_id[case["id"]])
        independent.append(result)
        saved_cap = case.get("cap", {}).get("hit") if isinstance(case.get("cap"), Mapping) else None
        saved_q = case.get("quality", {})
        if (result["protocol_valid_recomputed"] != bool(saved_q.get("protocol_valid"))
                or result["cap_hit"] != bool(saved_cap)
                or result["exact_recomputed"] != bool(saved_q.get("exact_edit"))
                or result["predicted_noop_recomputed"] != bool(saved_q.get("predicted_noop"))):
            saved_case_mismatches.append(str(case["id"]))
    counts, family_counts = result_counts(independent)
    saved_denoms = quality.get("denominators", {})
    compare_fields = (
        "protocol_rows", "protocol_error_rows", "cap_hit_rows", "edit_exact_rows",
        "strict_noop_correct_rows", "noop_false_positive_rows", "exact_region_rows",
    )
    denominator_comparison = {
        field: {"independent": counts.get(field, 0), "saved_partial": saved_denoms.get(field),
                "match": counts.get(field, 0) == saved_denoms.get(field)}
        for field in compare_fields
    }
    if saved_case_mismatches or not all(item["match"] for item in denominator_comparison.values()):
        raise RuntimeError("saved CPU classifications do not match independent reclassification")

    cpu_ids = complete_ids
    cuda_quality = load_json(CUDA)
    vulkan_quality = load_json(VULKAN)
    backend_comparisons = [
        compare_backend(complete, cuda_quality, "CUDA Q8 graph0", panel_by_id),
        compare_backend(complete, vulkan_quality, "notebook Vulkan Q8", panel_by_id),
    ]
    long_prefill = []
    for result, case in zip(independent, complete):
        timings = case.get("server_timing", {}).get("timings", {})
        speed = timings.get("prompt_per_second") if isinstance(timings, Mapping) else None
        if isinstance(speed, (int, float)):
            long_prefill.append({
                "id": case["id"], "prompt_n": timings.get("prompt_n"),
                "prompt_ms": timings.get("prompt_ms"), "prompt_per_second": speed,
            })
    long_prefill.sort(key=lambda item: item["prompt_per_second"])
    identity_bindings = {
        "model_sha256": launch.get("model_sha256") == EXPECTED["model_sha256"],
        "remote_launch_model_sha256": launch.get("model_sha256"),
        "remote_terminal_model_sha256": remote_terminal.get("model_sha256"),
        "quality_model_provenance": quality.get("model", {}).get("provenance"),
        "runtime_identity_sha256": launch.get("identity", {}).get("sha256"),
        "runtime_binary_sha256": launch.get("identity", {}).get("binary_sha256"),
        "runtime_files": launch.get("identity", {}).get("files"),
        "runtime_sha_expected": launch.get("identity", {}).get("sha256") == EXPECTED["runtime_sha256"],
        "runtime_binary_sha_expected": launch.get("identity", {}).get("binary_sha256") == EXPECTED["runtime_binary_sha256"],
        "quality_panel_case_ids_sha256": quality.get("panel", {}).get("case_ids_sha256"),
        "panel_case_ids_sha256_expected": quality.get("panel", {}).get("case_ids_sha256") == EXPECTED["panel_case_ids_sha256"],
        "panel_order_prefix_verified": case_ids == ordered_panel_ids[:35],
        "unattempted_rows": len(quality.get("unattempted_ids", [])),
    }
    pid_binding = {
        "launch_pid": launch.get("pid"), "ready_server_pid": ready.get("server_pid"),
        "ready_child_pid": ready.get("server_child_pid"),
        "live_audit_pid": live.get("pid"),
        "remote_terminal_child_pid": remote_terminal.get("server_child_pid"),
        "all_declared_pids_consistent": (
            launch.get("pid") == ready.get("server_pid")
            and ready.get("server_child_pid") == live.get("pid") == remote_terminal.get("server_child_pid")
        ),
        "remote_terminal_status": remote_terminal.get("status"),
        "remote_server_exit_code": remote_terminal.get("server_exit_code"),
        "local_client_exit_code": terminal.get("client_exit_code"),
        "controlled_stop_note": "remote child and local scorer were stopped after the bounded partial run; this is not a successful full-panel exit",
    }
    report = {
        "schema_version": "sepalith.run09.theta0-q8-notebook-cpu-dev-partial-independent-review.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independent_partial_review_complete_no_admission",
        "scope": {
            "run": str(RUN), "panel_rows": 75, "attempted_rows": 35,
            "complete_rows": 34, "partial_rows": 1, "unattempted_rows": 40,
            "families_observed": {"roxygen_drafting": 8, "no_op": 26},
            "model_weights_or_generation": False,
            "ssh_network_server_gpu": False,
            "final_or_sealed_data": False,
        },
        "collection": {"manifest_path": str(RUN / "collection-manifest.json"),
                       "declared_files": 10, "verified_files": sum(int(v["matches_expected"]) for v in collection_checks.values()),
                       "checks": collection_checks},
        "pinned_inputs": {
            "review_source": {"path": str(PINNED_REVIEW), "sha256": EXPECTED["review_sha256"]},
            "panel": {"path": str(PANEL), "sha256": EXPECTED["panel_sha256"], "rows": 75,
                      "ordered_ids_sha256": canonical_sha(ordered_panel_ids)},
            "protocol": {"path": str(EXEC / "packages/sepalith/src/sepalith/campaign_protocol.py"),
                         "sha256": EXPECTED["protocol_sha256"]},
            "tokenizer": tokenizer_audit,
            "cuda_quality": {"path": str(CUDA), "sha256": sha256_file(CUDA)},
            "vulkan_quality": {"path": str(VULKAN), "sha256": sha256_file(VULKAN)},
        },
        "independent_reclassification": {
            "complete_case_ids_sha256": canonical_sha(cpu_ids),
            "counts": counts,
            "family_counts": family_counts,
            "denominator_comparison": denominator_comparison,
            "saved_case_classification_mismatches": saved_case_mismatches,
            "cases": [
                dict(result, prompt_sha256=case.get("prompt_sha256"), target_sha256=case.get("target_sha256"),
                     hf_prompt_ids_sha256=case.get("hf_prompt_ids_sha256"),
                     native_prompt_ids_sha256=case.get("native_prompt_ids_sha256"),
                     prompt_token_identity=case.get("prompt_token_identity"))
                for result, case in zip(independent, complete)
            ],
            "partial_record": compact_partial(partial[0]),
        },
        "matched_subset_backend_comparison": {
            "cpu_case_order_prefix_only": True,
            "comparisons": backend_comparisons,
            "interpretation": "Prompt/target geometry identity is exact across the 34-row prefix; generated output IDs/text are backend-dependent and do not establish equivalence or promotion.",
        },
        "runtime_binding": {"identity": identity_bindings, "pids": pid_binding,
                            "preflight_profile": preflight.get("profile"),
                            "remote_live_device_audit": {key: live.get(key) for key in ("backend", "ngl", "dri_fds", "vulkan_mappings")}},
        "timing": {
            "local_elapsed_seconds": terminal.get("elapsed_seconds"),
            "remote_elapsed_seconds": remote_terminal.get("elapsed_seconds"),
            "conservative_charge_seconds": max(float(terminal.get("elapsed_seconds", 0)), float(remote_terminal.get("elapsed_seconds", 0))),
            "observed_longest_prefill_sample": long_prefill[0] if long_prefill else None,
            "prefill_note": "The lowest saved complete-row server prompt rate is an observed native CPU prefill sample (about 53 tokens/s); it is not an editor end-to-end latency measurement or a universal rate.",
        },
        "decision": {
            "recommendation": "partial_diagnostic_only_no_admission",
            "reason": "Only 34 of 75 rows completed; one observed no_op transport partial and 40 panel rows were unattempted. The complete prefix is family-skewed (8 roxygen then 26 no_op), so no full-panel ranking or quality claim is supported.",
            "quality_scope": "Reclassification and matched identity audit only; no promotion, campaign-state update, rerun, or model claim.",
        },
        "limits": [
            "Only the 34 complete saved responses were parsed with the pinned protocol and local tokenizer.",
            "The partial row was not treated as a response or quality success; no unattempted row was opened.",
            "Prompt/target identity parity does not imply generated-token parity across CPU, CUDA, and Vulkan.",
            "No NLL is available from the native endpoint, and no editor end-to-end latency was measured.",
        ],
    }
    out = PLAN / "docs/campaign/work/theta0-cpu-dev-partial-review/review-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(out), "report_sha256": sha256_file(out),
        "counts": counts, "family_counts": family_counts,
        "partial": report["independent_reclassification"]["partial_record"],
        "backend": [{"label": item["label"], "token_matches": item["generated_token_id_matches"],
                     "token_mismatches": item["generated_token_id_mismatch_ids"],
                     "protocol_mismatches": item["protocol_mismatch_ids"]} for item in backend_comparisons],
        "prefill": long_prefill[0] if long_prefill else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
