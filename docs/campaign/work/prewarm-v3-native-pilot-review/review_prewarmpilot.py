#!/usr/bin/env python3
"""Independent, prompt-free review of the collected RUN-04 pilot cells.

This reads only the bounded root-collected evidence and the pinned four-row
source capsule.  It deliberately does not read a model, start a server, or
emit prompt text.  The output is a compact JSON review suitable for binding
from a receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EVIDENCE = PLAN / "docs/campaign/work/lead/prewarm-v3-pilot-a-evidence"
CAPSULE = PLAN / "docs/campaign/work/lead/prewarm-v3-pilot-a-capsule"
OUT_DIR = PLAN / "docs/campaign/work/prewarm-v3-native-pilot-review"
OUT = OUT_DIR / "prewarm-v3-native-pilot-review.json"

EXPECTED_MODEL_SHA = (
    "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559"
)
EXPECTED_BINARY_SHA = (
    "92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f"
)
EXPECTED_FIXTURE_SHA = (
    "4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008"
)
EXPECTED_FIXTURE_MANIFEST_SHA = (
    "0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3"
)
EXPECTED_CONTEXT_CAPSULE_SHA = (
    "eda08da64985e0547e4357588593b31db927eeca1d1de3a617353df859859ee0"
)
EXPECTED_SOURCE_SIDECAR_SHA = (
    "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f"
)
EXPECTED_BUILD = "b10453-3cb7ffb1a"

CELL_FILES = {
    "case0-off": "pilot-a-off-case0.json",
    "case0-on": "pilot-a-on-case0.json",
    "case3-off": "pilot-a-off-case3.json",
    "case3-on": "pilot-a-on-case3.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: Any) -> Any:
    """Keep review output bounded and avoid accidentally retaining prompt text."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def check_collection_manifest() -> dict[str, Any]:
    manifest_path = EVIDENCE / "collection-manifest.json"
    expected = read_json(manifest_path)
    listed = set(expected)
    actual = {
        str(p.relative_to(EVIDENCE))
        for p in EVIDENCE.rglob("*")
        if p.is_file() and p.name != "collection-manifest.json"
    }
    mismatches: list[dict[str, Any]] = []
    for rel in sorted(listed | actual):
        path = EVIDENCE / rel
        if rel not in expected:
            mismatches.append({"path": rel, "reason": "unlisted_file"})
        elif not path.exists():
            mismatches.append({"path": rel, "reason": "missing_file"})
        else:
            actual_sha = sha256_file(path)
            if actual_sha != expected[rel]:
                mismatches.append(
                    {
                        "path": rel,
                        "reason": "sha256_mismatch",
                        "expected": expected[rel],
                        "actual": actual_sha,
                    }
                )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "listed_file_count": len(listed),
        "actual_file_count_excluding_manifest": len(actual),
        "all_files_verified": not mismatches and len(listed) == len(actual),
        "mismatches": mismatches,
    }


def check_capsule_manifest() -> dict[str, Any]:
    manifest_path = CAPSULE / "capsule-manifest.json"
    expected = read_json(manifest_path)
    mismatches: list[dict[str, Any]] = []
    for rel, expected_sha in expected.items():
        path = CAPSULE / rel
        if not path.exists():
            mismatches.append({"path": rel, "reason": "missing_file"})
            continue
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            mismatches.append(
                {
                    "path": rel,
                    "reason": "sha256_mismatch",
                    "expected": expected_sha,
                    "actual": actual_sha,
                }
            )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "listed_file_count": len(expected),
        "all_listed_files_verified": not mismatches,
        "mismatches": mismatches,
    }


def native_timing(request: dict[str, Any]) -> dict[str, Any]:
    response = request.get("response") or {}
    raw = response.get("rawJson") or {}
    timing = raw.get("timings") or {}
    return {
        key: scalar(timing.get(key))
        for key in (
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "predicted_n",
            "predicted_ms",
        )
    }


def request_summary(request: dict[str, Any]) -> dict[str, Any]:
    response = request.get("response") or {}
    generated = response.get("generatedTokenIds")
    completion = response.get("rawCompletion")
    summary: dict[str, Any] = {
        "sequence": request.get("sequence"),
        "kind": request.get("kind"),
        "http_status": request.get("httpStatus"),
        "elapsed_ms": scalar(request.get("elapsedMs")),
        "server_queue_ms": scalar(request.get("serverQueueMs")),
        "client_queue_wait_ms": scalar(request.get("clientQueueWaitMs")),
        "prompt_length": request.get("promptLength"),
        "prompt_ids_sha256": request.get("promptIdsSha256"),
        "n_predict": request.get("nPredict"),
        "native_timings": native_timing(request),
        "request_started_epoch_ms": request.get("startedAtEpochMs"),
        "request_finished_epoch_ms": request.get("finishedAtEpochMs"),
    }
    if generated is not None:
        summary.update(
            {
                "generated_token_count": len(generated),
                "generated_token_ids_sha256": hash_json(generated),
                "stop_type": response.get("stopType"),
                "truncated": response.get("truncated"),
                "tokens_evaluated": response.get("tokensEvaluated"),
                # Native response tokens_cached is retained as an observed
                # slot-occupancy field; it is not used as cache proof.
                "response_tokens_cached_slot_occupancy": response.get("tokensCached"),
                "completion_sha256": sha256_bytes(completion.encode()) if completion is not None else None,
                "completion": completion if len(completion or "") <= 64 else None,
            }
        )
    else:
        error = request.get("error")
        if error is not None:
            summary["error"] = str(error)[:240]
    return summary


def log_markers(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    offload = []
    eval_lines = []
    processing = []
    cache_lines = []
    prompt_eval_geometry = []
    generation_geometry = []
    for line in lines:
        if "offloaded " in line and " layers to GPU" in line:
            offload.append(line[:240])
        if "prompt eval time =" in line or "eval time =" in line:
            eval_lines.append(line[:240])
        prompt_match = re.search(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", line)
        if prompt_match:
            prompt_eval_geometry.append(
                {"elapsed_ms": float(prompt_match.group(1)), "prompt_n": int(prompt_match.group(2))}
            )
        generation_match = re.search(r"eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", line)
        if generation_match:
            generation_geometry.append(
                {"elapsed_ms": float(generation_match.group(1)), "predicted_n": int(generation_match.group(2))}
            )
        if "prompt processing," in line:
            processing.append(line[:240])
        if "cached n_tokens" in line:
            cache_lines.append(line[:240])
    return {
        "sha256": sha256_file(path),
        "line_count": len(lines),
        "vulkan_seen": any("Vulkan0" in line for line in lines),
        "listening_seen": any("listening on http://127.0.0.1:18403" in line for line in lines),
        "build_seen": any("build 10453 (3cb7ffb1a)" in line for line in lines),
        "offload_markers": offload,
        "native_eval_marker_count": len(eval_lines),
        "native_eval_markers": eval_lines,
        "prompt_eval_geometry": prompt_eval_geometry,
        "generation_geometry": generation_geometry,
        "prompt_processing_marker_count": len(processing),
        "prompt_processing_markers": processing,
        "cache_marker_count": len(cache_lines),
        "cache_markers": cache_lines,
    }


def cell_review(cell: str, result_name: str, fixture_rows: list[dict[str, Any]], capsule_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell_dir = EVIDENCE / cell
    result_path = cell_dir / result_name
    result = read_json(result_path)
    terminal = read_json(cell_dir / "terminal.json")
    launch = read_json(cell_dir / "server-launch.json")
    client_launch = read_json(cell_dir / "client-launch.json")
    health = read_json(cell_dir / "health.json")
    fixture_index = result["fixtureIndex"]
    fixture = fixture_rows[fixture_index]
    capsule = capsule_rows[fixture_index]
    requests = [request_summary(request) for request in result.get("requests", [])]
    result_fg = result.get("foreground")
    generated = result_fg.get("generatedTokenIds") if isinstance(result_fg, dict) else None
    completion = result_fg.get("rawCompletion") if isinstance(result_fg, dict) else None
    props = health.get("props", {})
    terminal_server = terminal.get("server", {})
    terminal_client = terminal.get("client", {})
    prompt_length = requests[0].get("prompt_length") if requests else None
    log = log_markers(cell_dir / "server.log")
    completed_request_count = sum(1 for request in requests if request.get("http_status") == 200)
    checks = {
        "fixture_row_id_matches": result.get("rowId") == fixture.get("row_id") == capsule.get("row_id"),
        "fixture_is_train": fixture.get("split") == "train" and capsule.get("split") == "train",
        "fixture_prompt_length_plus_manual_bos": prompt_length == fixture.get("prompt_token_count", -1) + 1,
        "fixture_manifest_hash": result.get("manifestSha256") == EXPECTED_FIXTURE_MANIFEST_SHA,
        "fixture_hash": result.get("fixtureSha256") == EXPECTED_FIXTURE_SHA,
        "context_capsule_hash": result.get("contextCapsuleSha256") == EXPECTED_CONTEXT_CAPSULE_SHA,
        "source_sidecar_hash": result.get("sourceSidecarSha256") == EXPECTED_SOURCE_SIDECAR_SHA,
        "health_ok": health.get("health", {}).get("status") == "ok",
        "build_matches": props.get("build_info") == EXPECTED_BUILD,
        "model_path_pinned": props.get("model_path", "").endswith("SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf"),
        "server_launch_matches_health": launch.get("argv", [None, None, None])[2] == props.get("model_path"),
        "server_binary_path_pinned": launch.get("argv", [""])[0].endswith("build-b10453-vulkan-avx2/bin/llama-server"),
        "server_exited_cleanly": terminal_server.get("exit_code") == 0 and terminal_server.get("pid_exists") is False,
        "server_pid_binding": launch.get("pid") == terminal_server.get("pid"),
        "client_reaped": terminal_client.get("pid_exists") is False,
        "native_log_geometry_matches_completed_requests": len(log["prompt_eval_geometry"]) == completed_request_count,
    }
    return {
        "cell": cell,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "status": result.get("status"),
        "error": result.get("error"),
        "arm": result.get("arm"),
        "fixture_index": fixture_index,
        "row_id": result.get("rowId"),
        "family_for_audit": result.get("familyForAudit"),
        "fixture_prompt_token_count_without_manual_bos": fixture.get("prompt_token_count"),
        "request_prompt_length": prompt_length,
        "prompt_ids_sha256": requests[0].get("prompt_ids_sha256") if requests else None,
        "requests": requests,
        "result_timing": {
            key: scalar((result.get("timing") or {}).get(key))
            for key in (
                "totalRunElapsedMs",
                "warmLeadMs",
                "warmLeadElapsedMs",
                "foregroundBudgetMs",
                "foregroundCallToResolvedElapsedMs",
                "foregroundWithinBudget",
                "warmHttpElapsedMs",
                "foregroundHttpElapsedMs",
                "foregroundClientQueueWaitMs",
            )
        },
        "foreground": {
            "present": isinstance(result_fg, dict),
            "generated_token_count": len(generated) if generated is not None else 0,
            "generated_token_ids_sha256": hash_json(generated) if generated is not None else None,
            "completion_sha256": sha256_bytes(completion.encode()) if completion is not None else None,
            "stop_type": result_fg.get("stopType") if isinstance(result_fg, dict) else None,
            "operation": result_fg.get("operation") if isinstance(result_fg, dict) else None,
        },
        "server": {
            "pid": launch.get("pid"),
            "argv_sha256": hash_json(launch.get("argv")),
            "terminal_exit_code": terminal_server.get("exit_code"),
            "health_build_info": props.get("build_info"),
            "health_model_path": props.get("model_path"),
        },
        "client": {
            "pid": client_launch.get("pid"),
            "argv_sha256": hash_json(client_launch.get("argv")),
            "terminal_exit_code": terminal_client.get("exit_code"),
        },
        "server_log": log,
        "checks": checks,
        "all_cell_checks": all(checks.values()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    collection = check_collection_manifest()
    capsule_manifest = check_capsule_manifest()
    fixture_manifest_path = CAPSULE / "fixture-manifest.json"
    capsule_path = CAPSULE / "context-capsule.json"
    fixture_manifest = read_json(fixture_manifest_path)
    context_capsule = read_json(capsule_path)
    fixture_rows = fixture_manifest["selected_rows"]
    capsule_rows = context_capsule["rows"]

    capsule_row_ids = [row.get("row_id") for row in capsule_rows]
    fixture_row_ids = [row.get("row_id") for row in fixture_rows]
    input_checks = {
        "fixture_manifest_sha256": sha256_file(fixture_manifest_path),
        "fixture_sha256": fixture_manifest.get("fixture_sha256"),
        "context_capsule_sha256": sha256_file(capsule_path),
        "context_source_sidecar_sha256": context_capsule.get("source_sidecar_sha256"),
        "fixture_row_count": fixture_manifest.get("fixture_row_count"),
        "fixture_row_ids_order": fixture_row_ids,
        "capsule_row_ids_order": capsule_row_ids,
        "all_rows_train": all(row.get("split") == "train" for row in fixture_rows + capsule_rows),
        "all_context_rows_have_no_target_reward_keys": all(
            row.get("context_has_target_or_reward_keys") is False for row in fixture_rows
        ),
        "expected_pins_match": (
            sha256_file(fixture_manifest_path) == EXPECTED_FIXTURE_MANIFEST_SHA
            and fixture_manifest.get("fixture_sha256") == EXPECTED_FIXTURE_SHA
            and sha256_file(capsule_path) == EXPECTED_CONTEXT_CAPSULE_SHA
            and context_capsule.get("source_sidecar_sha256") == EXPECTED_SOURCE_SIDECAR_SHA
            and fixture_row_ids == capsule_row_ids
        ),
    }
    cells = [cell_review(cell, result_name, fixture_rows, capsule_rows) for cell, result_name in CELL_FILES.items()]
    server_pids = [cell["server"]["pid"] for cell in cells]

    short_off = next(cell for cell in cells if cell["cell"] == "case0-off")
    short_on = next(cell for cell in cells if cell["cell"] == "case0-on")
    long_off = next(cell for cell in cells if cell["cell"] == "case3-off")
    long_on = next(cell for cell in cells if cell["cell"] == "case3-on")
    short_off_fg = next(request for request in short_off["requests"] if request["kind"] == "foreground")
    short_on_warm = next(request for request in short_on["requests"] if request["kind"] == "prewarm")
    short_on_fg = next(request for request in short_on["requests"] if request["kind"] == "foreground")

    off_http = short_off["result_timing"]["foregroundHttpElapsedMs"]
    on_http = short_on["result_timing"]["foregroundHttpElapsedMs"]
    off_call = short_off["result_timing"]["foregroundCallToResolvedElapsedMs"]
    on_call = short_on["result_timing"]["foregroundCallToResolvedElapsedMs"]
    short_comparison = {
        "cell_pair": ["case0-off", "case0-on"],
        "completed_foreground_pair_count": 1,
        "same_prompt_ids": short_off["prompt_ids_sha256"] == short_on["prompt_ids_sha256"],
        "same_prompt_length": short_off["request_prompt_length"] == short_on["request_prompt_length"],
        "same_foreground_token_ids": short_off["foreground"]["generated_token_ids_sha256"] == short_on["foreground"]["generated_token_ids_sha256"],
        "same_foreground_text": short_off["foreground"]["completion_sha256"] == short_on["foreground"]["completion_sha256"],
        "same_foreground_stop_type": short_off["foreground"]["stop_type"] == short_on["foreground"]["stop_type"],
        "off_foreground_http_ms": off_http,
        "on_foreground_http_ms": on_http,
        "foreground_http_reduction_ms": round(off_http - on_http, 6),
        "foreground_http_reduction_percent": round((off_http - on_http) / off_http * 100, 4),
        "off_call_to_resolved_ms": off_call,
        "on_call_to_resolved_ms": on_call,
        "call_to_resolved_reduction_ms": round(off_call - on_call, 6),
        "off_native_timing": short_off_fg["native_timings"],
        "on_prewarm_native_timing": short_on_warm["native_timings"],
        "on_foreground_native_timing": short_on_fg["native_timings"],
        "cache_reuse_evidence": {
            "off_foreground_cache_n": short_off_fg["native_timings"].get("cache_n"),
            "off_foreground_prompt_n": short_off_fg["native_timings"].get("prompt_n"),
            "on_foreground_cache_n": short_on_fg["native_timings"].get("cache_n"),
            "on_foreground_prompt_n": short_on_fg["native_timings"].get("prompt_n"),
            "interpretation": "native timings and server log show 345 cached plus 1 prompt token on on foreground; response tokens_cached is final slot occupancy and is not used as reuse proof",
        },
    }

    long_outcomes = {
        "cell_count": 2,
        "case3_off": {
            "status": long_off["status"],
            "foreground_within_budget": long_off["result_timing"]["foregroundWithinBudget"],
            "foreground_call_to_resolved_ms": long_off["result_timing"]["foregroundCallToResolvedElapsedMs"],
            "foreground_http_elapsed_ms": long_off["result_timing"]["foregroundHttpElapsedMs"],
            "response_present": long_off["foreground"]["present"],
            "response_token_count": long_off["foreground"]["generated_token_count"],
            "client_queue_wait_ms": long_off["result_timing"]["foregroundClientQueueWaitMs"],
            "request_error": long_off["requests"][0].get("error"),
            "result_error": long_off["error"],
        },
        "case3_on": {
            "status": long_on["status"],
            "prewarm_elapsed_ms": long_on["requests"][0].get("elapsed_ms"),
            "prewarm_error": long_on["requests"][0].get("error"),
            "foreground_within_budget": long_on["result_timing"]["foregroundWithinBudget"],
            "foreground_call_to_resolved_ms": long_on["result_timing"]["foregroundCallToResolvedElapsedMs"],
            "foreground_http_elapsed_ms": long_on["result_timing"]["foregroundHttpElapsedMs"],
            "response_present": long_on["foreground"]["present"],
            "response_token_count": long_on["foreground"]["generated_token_count"],
            "foreground_client_queue_wait_ms": long_on["result_timing"]["foregroundClientQueueWaitMs"],
            "result_error": long_on["error"],
        },
        "interpretation": "Both 1903-token foreground attempts were caller-aborted without a response; no long-row prewarm benefit or output parity is inferred.",
    }

    all_cell_checks = all(cell["all_cell_checks"] for cell in cells)
    review = {
        "schema": "RUN-04-prewarm-v3-native-pilot-independent-review/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "limited_mechanical_signal_pending_replication",
        "recommendation": "Do not promote prewarm from this four-cell pilot. Retain the short-row cache signal for a matched multi-row replication; design a separate long-row cancellation/release test before making a general latency claim.",
        "scope": {
            "evidence_root": str(EVIDENCE),
            "capsule_root": str(CAPSULE),
            "model_sha256_recorded": EXPECTED_MODEL_SHA,
            "binary_sha256_recorded": EXPECTED_BINARY_SHA,
            "build_info": EXPECTED_BUILD,
            "warm_lead_ms": 1500,
            "foreground_budget_ms": 5000,
            "cell_count": 4,
            "completed_foreground_cell_count": 2,
            "completed_foreground_pair_count": 1,
            "canceled_or_incomplete_foreground_cell_count": 2,
        },
        "integrity": {
            "collection_manifest": collection,
            "capsule_manifest": capsule_manifest,
            "input_checks": input_checks,
            "fresh_server_per_cell": len(server_pids) == len(set(server_pids)),
            "server_pids": server_pids,
            "all_cell_checks": all_cell_checks,
        },
        "cells": cells,
        "short_row_comparison": short_comparison,
        "long_row_outcomes": long_outcomes,
        "limitations": [
            "The pilot has one completed off/on foreground pair; the long row is incomplete in both arms.",
            "serverQueueMs is null in the captured client records, so server arrival-to-launch delay is not separately measured.",
            "The on arm's prewarm request is intentionally one-token and incomplete; it is a cache-fill operation, not a quality response.",
            "Native tokens_cached is final slot occupancy. Cache reuse is attributed only to raw timings cache_n/prompt_n and matching server prompt-eval/cache log markers.",
            "Exact output parity here is a mechanical native response comparison and is not editor acceptance or model quality evidence.",
        ],
        "next_test": {
            "purpose": "replicate cache effect and test release after cancellation",
            "design": "Use several independently randomized short/medium/long TRAIN rows with fresh servers and matched lead. Keep one foreground slot, record native cache_n/prompt_n and prompt-eval logs, and after a canceled long prewarm issue a fresh foreground on the same server to verify slot/task release.",
            "success_condition": "Report paired completed foreground latency and exact IDs/text for every eligible row; classify timeout/cancel cells separately and require no unexplained queue starvation.",
        },
    }
    OUT.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "output_sha256": sha256_file(OUT),
        "collection_manifest_sha256": collection["sha256"],
        "capsule_manifest_sha256": capsule_manifest["sha256"],
        "all_integrity_checks": all_cell_checks and collection["all_files_verified"] and capsule_manifest["all_listed_files_verified"] and len(server_pids) == len(set(server_pids)),
        "short_http_reduction_ms": short_comparison["foreground_http_reduction_ms"],
        "short_http_reduction_percent": short_comparison["foreground_http_reduction_percent"],
        "short_output_parity": all(short_comparison[key] for key in ("same_prompt_ids", "same_prompt_length", "same_foreground_token_ids", "same_foreground_text", "same_foreground_stop_type")),
        "long_completed": 0,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
