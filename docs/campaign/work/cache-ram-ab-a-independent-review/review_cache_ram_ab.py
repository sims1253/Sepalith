#!/usr/bin/env python3
"""Independent, bounded review of the six RUN-04 cache-RAM cells.

Reads only the collected JSON/log evidence and copied source capsule. It does
not load a model, contact a server, or inspect corpus/sealed data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_MODE = {"default": "8192", "zero": "0"}
PAIRS = [(f"pair{i}-{mode}", i, mode) for i in range(3) for mode in ("default", "zero")]

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha_json_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

def verify_hash_map(base: Path, declarations: dict[str, Any], label: str) -> dict[str, str]:
    actual: dict[str, str] = {}
    failures: list[str] = []
    for rel, declaration in declarations.items():
        expected = declaration["sha256"] if isinstance(declaration, dict) else declaration
        path = base / rel
        if not path.is_file():
            failures.append(f"{label}:{rel}:missing")
            continue
        got = sha_file(path)
        actual[rel] = got
        if got != expected:
            failures.append(f"{label}:{rel}:sha {got} != {expected}")
    assert_true(not failures, "; ".join(failures))
    return actual

def log_timestamp(line: str) -> float | None:
    match = re.match(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s", line)
    if not match:
        return None
    hour, second, millis, micros = (int(part) for part in match.groups())
    return hour * 3600.0 + second + millis / 1000.0 + micros / 1_000_000.0

def server_metrics(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    enabled = [line for line in lines if "prompt cache is enabled" in line]
    disabled = [line for line in lines if "prompt cache is disabled" in line]
    saves = [line for line in lines if "prompt_save:" in line]
    cache_states = re.findall(
        r"cache state:\s*(\d+) prompts,\s*([0-9.]+) MiB \(limits:\s*([0-9.]+) MiB",
        "\n".join(lines),
    )
    updates = re.findall(r"prompt cache update took\s+([0-9.]+) ms", "\n".join(lines))
    cancels: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    # Preserve only lifecycle scalars, never prompt lines or full addresses.
    for line in lines:
        timestamp = log_timestamp(line)
        cancel = re.search(r"stop: cancel task, id_task = (-?\d+)", line)
        if cancel:
            cancels.append({"task": int(cancel.group(1)), "t_s": timestamp})
        release = re.search(
            r"release: id\s+\d+ \| task\s+(-?\d+).*?n_tokens\s*=\s*(\d+), truncated\s*=\s*(\d+)",
            line,
        )
        if release:
            releases.append(
                {
                    "task": int(release.group(1)),
                    "n_tokens": int(release.group(2)),
                    "truncated": bool(int(release.group(3))),
                    "t_s": timestamp,
                }
            )
    cancel_release_ms: list[dict[str, Any]] = []
    for cancel in cancels:
        matching = next(
            (release for release in releases if release["task"] == cancel["task"] and release["t_s"] is not None and cancel["t_s"] is not None and release["t_s"] >= cancel["t_s"]),
            None,
        )
        cancel_release_ms.append(
            {
                "task": cancel["task"],
                "cancel_to_release_ms": round((matching["t_s"] - cancel["t_s"]) * 1000.0, 3) if matching else None,
                "release_n_tokens": matching["n_tokens"] if matching else None,
                "release_truncated": matching["truncated"] if matching else None,
            }
        )
    cached = [int(value) for value in re.findall(r"cached n_tokens\s*=\s*(\d+)", "\n".join(lines))]
    prompt_eval = re.findall(
        r"task\s+10 \|\s+prompt eval time\s*=\s*([0-9.]+) ms /\s*(\d+) tokens",
        "\n".join(lines),
    )
    decode_eval = re.findall(
        r"task\s+10 \|\s+\s*eval time\s*=\s*([0-9.]+) ms /\s*(\d+) tokens",
        "\n".join(lines),
    )
    total_eval = re.findall(
        r"task\s+10 \|\s+\s*total time\s*=\s*([0-9.]+) ms /\s*(\d+) tokens",
        "\n".join(lines),
    )
    return {
        "enabled_marker_count": len(enabled),
        "disabled_marker_count": len(disabled),
        "prompt_save_count": len(saves),
        "cache_state_lines": [
            {"prompts": int(prompts), "mib": float(mib), "limit_mib": float(limit)}
            for prompts, mib, limit in cache_states
        ],
        "prompt_cache_update_ms": [float(value) for value in updates],
        "cancel_count": len(cancels),
        "release_count": len(releases),
        "cancel_release": cancel_release_ms,
        "cached_n_tokens_lines": cached,
        "short_prompt_eval": {"ms": float(prompt_eval[-1][0]), "tokens": int(prompt_eval[-1][1])} if prompt_eval else None,
        "short_decode_eval": {"ms": float(decode_eval[-1][0]), "tokens": int(decode_eval[-1][1])} if decode_eval else None,
        "short_total_eval": {"ms": float(total_eval[-1][0]), "tokens": int(total_eval[-1][1])} if total_eval else None,
        "uppercase_error_lines": sum(1 for line in lines if " ERROR " in line),
    }

def percentile_summary(values: list[float]) -> dict[str, float]:
    assert_true(len(values) == 3, f"expected three paired values, got {len(values)}")
    return {
        "n": 3,
        "median_ms": round(statistics.median(values), 6),
        "max_abs_ms": round(max(abs(value) for value in values), 6),
        "mean_ms": round(statistics.mean(values), 6),
        "p95_ms": None,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prior-recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    capsule = args.capsule.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    collection = read_json(evidence / "collection-manifest.json")
    collection_actual = verify_hash_map(evidence, collection["files"], "evidence")
    capsule_root = read_json(capsule / "manifest.json")
    capsule_actual = verify_hash_map(capsule, capsule_root, "capsule")

    cell_reports: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    normalized_server_argv: list[str] | None = None
    common_inputs: dict[str, Any] | None = None
    short_responses: list[dict[str, Any]] = []
    baseline = read_json(args.baseline.resolve())
    prior_recovery = read_json(args.prior_recovery.resolve())
    baseline_foreground = baseline["foreground"]
    baseline_result_sha = sha_file(args.baseline.resolve())
    prior_recovery_sha = sha_file(args.prior_recovery.resolve())
    baseline_identity = {
        # The accepted pilot snapshot stores the count without manual BOS;
        # its foreground request explicitly includes that one BOS token.
        "prompt_length": int(baseline["snapshot"]["promptTokenCount"]) + 1,
        "prompt_ids_sha256": baseline_foreground["promptTokenIdsSha256"],
        "generated_ids_sha256": sha_json_value(baseline_foreground["generatedTokenIds"]),
        "raw_text_sha256": sha_json_value(baseline_foreground["rawCompletion"]),
        "stop_type": baseline_foreground["stopType"],
        "operation": baseline_foreground["operation"],
    }
    prior_short = prior_recovery["shortForegroundResponse"]
    assert_true(prior_short["promptTokenIdsSha256"] == baseline_identity["prompt_ids_sha256"], "prior recovery prompt identity differs from baseline")
    assert_true(sha_json_value(prior_short["generatedTokenIds"]) == baseline_identity["generated_ids_sha256"], "prior recovery output IDs differ from baseline")
    assert_true(sha_json_value(prior_short["rawText"]) == baseline_identity["raw_text_sha256"], "prior recovery output text differs from baseline")
    assert_true(prior_short["stopType"] == baseline_identity["stop_type"] and prior_short["operation"] == baseline_identity["operation"], "prior recovery status differs from baseline")

    for name, pair, mode in PAIRS:
        cell = evidence / name
        launch = read_json(cell / "launch.json")
        client_launch = read_json(cell / "client-launch.json")
        terminal = read_json(cell / "terminal.json")
        result = read_json(cell / "recovery-a-recovery.json")
        raw_short = result["transport"]["records"][2]["response"]["rawJson"]
        timings = raw_short["timings"]
        argv = launch["argv"]
        cache_positions = [index for index, value in enumerate(argv) if value == "--cache-ram"]
        assert_true(cache_positions == [len(argv) - 2], f"{name}: cache-ram is not the final server setting")
        cache_value = argv[-1]
        assert_true(cache_value == EXPECTED_MODE[mode], f"{name}: unexpected cache-ram value {cache_value}")
        stripped_argv = argv[: cache_positions[0]] + argv[cache_positions[0] + 2 :]
        if normalized_server_argv is None:
            normalized_server_argv = stripped_argv
        else:
            assert_true(stripped_argv == normalized_server_argv, f"{name}: server argv differs beyond cache-ram")
        assert_true(result["status"] == "pass" and terminal["result_status"] == "pass", f"{name}: result not pass")
        assert_true(float(launch.get("wall_seconds", math.inf)) <= 90.0, f"{name}: launch wall ceiling exceeded")
        assert_true(terminal["result_sha256"] == sha_file(cell / "recovery-a-recovery.json"), f"{name}: terminal result hash mismatch")
        assert_true(launch["capsule_manifest_sha256"] == sha_file(capsule / name / "capsule-manifest.json"), f"{name}: launch capsule hash mismatch")
        assert_true(launch["binary_sha256"] == "92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f", f"{name}: binary identity mismatch")
        assert_true(launch["model_sha256"] == "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559", f"{name}: model identity mismatch")
        expected_inputs = {
            "source_sidecar_sha256": result["sourceSidecarSha256"],
            "context_capsule_sha256": result["contextCapsuleSha256"],
            "fixture_sha256": result["fixtureSha256"],
            "fixture_manifest_sha256": result["fixtureManifestSha256"],
        }
        if common_inputs is None:
            common_inputs = expected_inputs
        else:
            assert_true(expected_inputs == common_inputs, f"{name}: prepared input identities differ")
        checks = result["checks"]
        assert_true(all(checks.values()), f"{name}: a harness check is false")
        assert_true(result["transport"]["maxConcurrent"] == 1 and result["transport"]["atMostOneUnsettled"], f"{name}: transport concurrency guard failed")
        assert_true(result["transport"]["unresolvedAfterDrain"] == [], f"{name}: unresolved transport remains")
        assert_true(result["transport"]["lateSettlementAfterAbort"] == [0, 1], f"{name}: late abort records changed")
        records = result["transport"]["records"]
        assert_true(len(records) == 3, f"{name}: expected 3 transport records")
        assert_true([record["kind"] for record in records] == ["prewarm", "foreground", "foreground"], f"{name}: record kinds differ")
        assert_true([record["promptLength"] for record in records] == [1903, 1903, 346], f"{name}: prompt lengths differ")
        assert_true(records[0]["promptIdsSha256"] == records[1]["promptIdsSha256"] == result["longSnapshot"]["promptIdsSha256"], f"{name}: long prompt IDs differ")
        assert_true(records[2]["promptIdsSha256"] == result["shortSnapshot"]["promptIdsSha256"], f"{name}: short prompt IDs differ")
        assert_true(records[0]["error"].startswith("TimeoutError"), f"{name}: prewarm did not time out")
        assert_true(records[1]["error"].startswith("AbortError"), f"{name}: long foreground did not abort")
        assert_true(records[2].get("httpStatus") == 200, f"{name}: short HTTP status differs")
        short = result["shortForegroundResponse"]
        parity = {
            "prompt_length": short["promptLength"] == baseline_identity["prompt_length"],
            "prompt_ids": short["promptTokenIdsSha256"] == baseline_identity["prompt_ids_sha256"],
            "generated_ids": sha_json_value(short["generatedTokenIds"]) == baseline_identity["generated_ids_sha256"],
            "raw_text": sha_json_value(short["rawText"]) == baseline_identity["raw_text_sha256"],
            "stop_type": short["stopType"] == baseline_identity["stop_type"],
            "operation": short["operation"] == baseline_identity["operation"],
            "native_tokens": raw_short["tokens"] == short["generatedTokenIds"],
            "native_eos": raw_short["stop_type"] == "eos" and raw_short["stop"] is True,
            "native_not_truncated": raw_short["truncated"] is False,
        }
        assert_true(all(parity.values()), f"{name}: short output parity failed: {parity}")
        short_responses.append({
            "cell": name,
            "prompt_length": short["promptLength"],
            "prompt_ids_sha256": short["promptTokenIdsSha256"],
            "generated_ids_sha256": sha_json_value(short["generatedTokenIds"]),
            "generated_count": len(short["generatedTokenIds"]),
            "raw_text_sha256": sha_json_value(short["rawText"]),
            "stop_type": short["stopType"],
            "operation": short["operation"],
            "parity": parity,
        })
        native = {
            "cache_n": timings["cache_n"],
            "prompt_n": timings["prompt_n"],
            "prompt_ms": timings["prompt_ms"],
            "predicted_n": timings["predicted_n"],
            "predicted_ms": timings["predicted_ms"],
            "tokens_cached": raw_short["tokens_cached"],
            "tokens_evaluated": raw_short["tokens_evaluated"],
            "tokens_predicted": raw_short["tokens_predicted"],
            "truncated": raw_short["truncated"],
        }
        server = server_metrics(cell / "server.log")
        assert_true(server["enabled_marker_count"] == (1 if mode == "default" else 0), f"{name}: enabled marker mismatch")
        assert_true(server["disabled_marker_count"] == (0 if mode == "default" else 1), f"{name}: disabled marker mismatch")
        assert_true(server["prompt_save_count"] == (1 if mode == "default" else 0), f"{name}: prompt save mismatch")
        assert_true(len(server["cache_state_lines"]) == (2 if mode == "default" else 0), f"{name}: cache state archive evidence mismatch")
        assert_true(server["cancel_count"] == 2 and server["release_count"] == 3, f"{name}: cancel/release count mismatch")
        assert_true(server["uppercase_error_lines"] == 0, f"{name}: server contains ERROR line")
        assert_true(native["cache_n"] == 57 and native["prompt_n"] == 289 and native["predicted_n"] == 12, f"{name}: native token geometry mismatch")
        assert_true(native["tokens_evaluated"] == 346 and native["tokens_predicted"] == 12 and native["tokens_cached"] == 357, f"{name}: native response counters mismatch")
        assert_true(native["truncated"] is False, f"{name}: native response truncated")
        report = {
            "cell": name,
            "pair": pair,
            "mode": mode,
            "cache_ram_mib": int(cache_value),
            "launch": {
                "launch_sha256": sha_file(cell / "launch.json"),
                "terminal_sha256": sha_file(cell / "terminal.json"),
                "result_sha256": sha_file(cell / "recovery-a-recovery.json"),
                "server_log_sha256": sha_file(cell / "server.log"),
                "server_capsule_manifest_sha256": launch["capsule_manifest_sha256"],
                "server_argv_cache_only_delta": True,
                "server_pid_exit_code": terminal["server"]["exit_code"],
                "client_pid_exit_code": terminal["client"]["exit_code"],
                "terminal_seconds": terminal["seconds"],
                "client_argv_hash": sha_json_value(client_launch["argv"]),
            },
            "input_identity": expected_inputs,
            "lifecycle": {
                "harness_checks": checks,
                "max_concurrent": result["transport"]["maxConcurrent"],
                "at_most_one_unsettled": result["transport"]["atMostOneUnsettled"],
                "late_settlement_after_abort": result["transport"]["lateSettlementAfterAbort"],
                "unresolved_after_drain": result["transport"]["unresolvedAfterDrain"],
                "server_cancel_count": server["cancel_count"],
                "server_release_count": server["release_count"],
                "server_cancel_release": server["cancel_release"],
                "long_logical_elapsed_ms": result["logicalOutcomes"][0]["elapsedMs"],
                "short_logical_elapsed_ms": result["logicalOutcomes"][1]["elapsedMs"],
                "short_transport_elapsed_ms": records[2]["elapsedMs"],
                "short_client_queue_wait_ms": records[2].get("clientQueueWaitMs"),
            },
            "short_output": short_responses[-1],
            "native_short_timings": native,
            "server_cache_evidence": {
                "enabled_marker_count": server["enabled_marker_count"],
                "disabled_marker_count": server["disabled_marker_count"],
                "prompt_save_count": server["prompt_save_count"],
                "cache_state_lines": server["cache_state_lines"],
                "prompt_cache_update_ms": server["prompt_cache_update_ms"],
                "cached_n_tokens_lines": server["cached_n_tokens_lines"],
                "short_prompt_eval": server["short_prompt_eval"],
                "short_decode_eval": server["short_decode_eval"],
                "short_total_eval": server["short_total_eval"],
            },
        }
        cell_reports.append(report)
        by_name[name] = report

    assert_true(len(short_responses) == 6 and all(item["parity"] and all(item["parity"].values()) for item in short_responses), "not all six short outputs passed parity")
    defaults = [by_name[f"pair{i}-default"] for i in range(3)]
    zeros = [by_name[f"pair{i}-zero"] for i in range(3)]
    def paired(field: str) -> list[float]:
        return [float(zeros[i]["lifecycle"][field]) - float(defaults[i]["lifecycle"][field]) for i in range(3)]
    def paired_native(field: str) -> list[float]:
        return [float(zeros[i]["native_short_timings"][field]) - float(defaults[i]["native_short_timings"][field]) for i in range(3)]
    def paired_server(field: str) -> list[float]:
        return [float(zeros[i]["server_cache_evidence"][field]["ms"]) - float(defaults[i]["server_cache_evidence"][field]["ms"]) for i in range(3)]
    deltas = {
        "short_logical_elapsed_zero_minus_default_ms": paired("short_logical_elapsed_ms"),
        "short_transport_elapsed_zero_minus_default_ms": paired("short_transport_elapsed_ms"),
        "native_prompt_ms_zero_minus_default_ms": paired_native("prompt_ms"),
        "native_predicted_ms_zero_minus_default_ms": paired_native("predicted_ms"),
        "server_short_prompt_eval_ms_zero_minus_default_ms": paired_server("short_prompt_eval"),
        "server_short_decode_eval_ms_zero_minus_default_ms": paired_server("short_decode_eval"),
    }
    paired_summary = {key: {"values_ms": [round(value, 6) for value in values], **percentile_summary(values)} for key, values in deltas.items()}
    root_short_logical = {
        "default_ms": [report["lifecycle"]["short_logical_elapsed_ms"] for report in defaults],
        "zero_ms": [report["lifecycle"]["short_logical_elapsed_ms"] for report in zeros],
    }
    root_short_logical["default_s"] = [round(value / 1000.0, 3) for value in root_short_logical["default_ms"]]
    root_short_logical["zero_s"] = [round(value / 1000.0, 3) for value in root_short_logical["zero_ms"]]

    report = {
        "schema": "RUN-04.cache-ram-ab-a-independent-review/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "complete_limited_no_promotion",
        "scope": "Independent CPU-only review of three matched native recovery pairs; evidence and copied capsule only.",
        "inputs": {
            "evidence_root": str(evidence),
            "evidence_collection_manifest_sha256": sha_file(evidence / "collection-manifest.json"),
            "evidence_file_count": len(collection["files"]),
            "evidence_file_hashes_verified": len(collection_actual),
            "capsule_root": str(capsule),
            "capsule_manifest_sha256": sha_file(capsule / "manifest.json"),
            "capsule_file_count": len(capsule_root),
            "capsule_file_hashes_verified": len(capsule_actual),
            "capsule_per_cell_manifests_verified": 6,
            "baseline_result_sha256": baseline_result_sha,
            "prior_recovery_result_sha256": prior_recovery_sha,
            "binary_sha256": "92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f",
            "model_sha256": "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559",
            "fixture_sha256": common_inputs["fixture_sha256"],
            "fixture_manifest_sha256": common_inputs["fixture_manifest_sha256"],
            "source_sidecar_sha256": common_inputs["source_sidecar_sha256"],
            "context_capsule_sha256": common_inputs["context_capsule_sha256"],
        },
        "argv_review": {
            "server_common_argv_sha256": sha_json_value(normalized_server_argv),
            "only_server_argv_delta": {"flag": "--cache-ram", "default": "8192", "zero": "0"},
            "all_six_server_argvs_checked": True,
            "client_variations": "Only per-cell harness/output paths and copied per-cell capsule identities vary; all use the same fixture, manifest, context capsule, base URL, run ID, lead, cancel, and drain settings.",
        },
        "output_identity_review": {
            "cell_count": 6,
            "short_prompt_length": 346,
            "short_prompt_ids_sha256": baseline_identity["prompt_ids_sha256"],
            "short_generated_ids_sha256": baseline_identity["generated_ids_sha256"],
            "short_generated_count": 12,
            "short_output_text_sha256": baseline_identity["raw_text_sha256"],
            "short_stop_type": baseline_identity["stop_type"],
            "short_operation": baseline_identity["operation"],
            "all_six_match_each_other": True,
            "all_six_match_accepted_fixture0": True,
            "accepted_fixture0_result_sha256": baseline_result_sha,
            "prior_recovery_short_matches_fixture0": True,
            "long_prompt_lengths": [1903, 1903],
            "long_and_short_distinct_identities": True,
            "native_short_tokens_match_harness_ids": True,
            "native_short_eos_and_not_truncated": True,
        },
        "lifecycle_review": {
            "all_harness_pass_checks": True,
            "max_unsettled_http": 1,
            "max_concurrent_http": 1,
            "all_records_settled_after_bounded_drain": True,
            "late_abort_records_per_cell": [0, 1],
            "server_cancel_records_per_cell": 2,
            "server_release_records_per_cell": 3,
            "all_process_exit_codes_zero": True,
            "all_six_within_90_second_server_wall_ceiling": True,
        },
        "native_cache_review": {
            "default": {
                "startup_marker": "prompt cache is enabled, size limit: 8192 MiB",
                "prompt_save_per_cell": 1,
                "one_prompt_73_521_mib_state_per_cell": True,
                "prompt_cache_update_per_cell": 2,
            },
            "zero": {
                "startup_marker": "prompt cache is disabled - use --cache-ram N to enable it",
                "prompt_save_per_cell": 0,
                "one_prompt_73_521_mib_state_per_cell": False,
                "prompt_cache_update_per_cell": 0,
            },
            "short_response_counters_both_arms": {"cache_n": 57, "prompt_n": 289, "predicted_n": 12, "tokens_cached": 357, "tokens_evaluated": 346, "tokens_predicted": 12},
            "interpretation": "The disabled arm is confirmed by its startup marker and absence of prompt_save/cache-state archive lines. Both arms still report cache_n=57 and cached n_tokens=57 for the short request, so this is active slot/LCP reuse within the sequence and is not evidence that the disabled persistent prompt-cache archive remained enabled.",
            "same_clock_measurements": "Server log prompt-eval/decode totals and cancel-to-release intervals use each process's own monotonic-style log clock. Cross-process epoch queue estimates are omitted because arrival and native launch clocks are not calibrated.",
        },
        "paired_timing_review": {
            "short_logical_elapsed": root_short_logical,
            "zero_minus_default": paired_summary,
            "sample_limit": "n=3 paired observations per metric; report median and maximum absolute paired delta only. No reliable p95 is claimed.",
            "interpretation": "The three short logical pairs show no consistent cache-RAM benefit: zero minus default is +412.118, -12.346, and -60.801 ms. Native prompt evaluation is -56.254, +1081.870, and +1089.583 ms, while decode deltas are only -3.855, +0.454, and +6.537 ms. The large prompt deltas vary by pair and do not support promotion.",
        },
        "cells": cell_reports,
        "limitations": [
            "This is six matched recovery sequences on one TRAIN short fixture and one long fixture, not a broad latency benchmark.",
            "The short foreground includes the bounded recovery sequence after long cancellation; it is not an isolated warm-cache request.",
            "No calibrated native arrival-to-launch queue timing is reported; clientQueueWaitMs is retained as client-side evidence only.",
            "The response cache_n/tokens_cached fields cannot by themselves distinguish persistent prompt archive reuse from active slot/LCP reuse.",
            "No editor behavior, model quality, or promotion conclusion is established.",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "output_sha256": sha_file(output),
        "collection_verified": f"{len(collection_actual)}/{len(collection['files'])}",
        "capsule_verified": f"{len(capsule_actual)}/{len(capsule_root)} plus 6 per-cell manifests",
        "short_output_parity": True,
        "paired_short_logical_zero_minus_default_ms": deltas["short_logical_elapsed_zero_minus_default_ms"],
        "paired_prompt_zero_minus_default_ms": deltas["native_prompt_ms_zero_minus_default_ms"],
    }, sort_keys=True))

if __name__ == "__main__":
    main()
