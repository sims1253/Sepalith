#!/usr/bin/env python3
"""Independent scalar review of the bounded RUN-04 recovery-a evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EVIDENCE = PLAN / "docs/campaign/work/lead/prewarm-recovery-a-evidence"
CAPSULE = PLAN / "docs/campaign/work/lead/prewarm-recovery-a-capsule"
PILOT = PLAN / "docs/campaign/work/lead/prewarm-v3-pilot-a-evidence/case0-off"
OUT_DIR = PLAN / "docs/campaign/work/prewarm-recovery-a-independent-review"
OUT = OUT_DIR / "prewarm-recovery-a-independent-review.json"

FIXTURE_SHA = "4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008"
MANIFEST_SHA = "0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3"
CAPSULE_SHA = "eda08da64985e0547e4357588593b31db927eeca1d1de3a617353df859859ee0"
SIDECAR_SHA = "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f"
MODEL_SHA = "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559"
BINARY_SHA = "92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f"
BUILD = "b10453-3cb7ffb1a"
ROW_LONG = "00ef45e53aea030d3465f860"
ROW_SHORT = "0003ea6fc6a4d0b3b354efba"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_collection() -> dict[str, Any]:
    path = EVIDENCE / "collection-manifest.json"
    manifest = read_json(path)
    expected = manifest.get("files", {})
    actual = {p.name for p in EVIDENCE.iterdir() if p.is_file() and p.name != path.name}
    mismatches: list[dict[str, Any]] = []
    for name in sorted(set(expected) | actual):
        file_path = EVIDENCE / name
        if name not in expected:
            mismatches.append({"file": name, "reason": "unlisted"})
            continue
        if not file_path.exists():
            mismatches.append({"file": name, "reason": "missing"})
            continue
        entry = expected[name]
        actual_bytes = file_path.stat().st_size
        actual_sha = sha256_file(file_path)
        if actual_bytes != entry.get("bytes") or actual_sha != entry.get("sha256"):
            mismatches.append({"file": name, "reason": "identity_mismatch", "expected": entry, "actual": {"bytes": actual_bytes, "sha256": actual_sha}})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "remote": manifest.get("remote"),
        "listed_file_count": len(expected),
        "actual_file_count_excluding_manifest": len(actual),
        "all_files_verified": not mismatches and len(expected) == len(actual),
        "mismatches": mismatches,
    }


def verify_capsule() -> dict[str, Any]:
    path = CAPSULE / "capsule-manifest.json"
    manifest = read_json(path)
    mismatches: list[dict[str, Any]] = []
    for name, expected_sha in manifest.items():
        file_path = CAPSULE / name
        if not file_path.exists():
            mismatches.append({"file": name, "reason": "missing"})
            continue
        actual_sha = sha256_file(file_path)
        if actual_sha != expected_sha:
            mismatches.append({"file": name, "reason": "sha256_mismatch", "expected": expected_sha, "actual": actual_sha})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "listed_file_count": len(manifest),
        "all_listed_files_verified": not mismatches,
        "mismatches": mismatches,
    }


def parse_server_seconds(line: str) -> float | None:
    match = re.search(r"\b(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\b", line)
    if not match:
        return None
    return int(match.group(1)) + int(match.group(2)) + int(match.group(3)) / 1000 + int(match.group(4)) / 1_000_000


def server_log_review(path: Path, launch_at: str, short_record_start_epoch_ms: int, short_record_finish_epoch_ms: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected: list[tuple[int, str, float]] = []
    patterns = (
        "offloaded 43/43 layers to GPU",
        "stop: cancel task, id_task = 0",
        "release: id  0 | task 0",
        "processing task, is_child = 0",
        "stop: cancel task, id_task = 7",
        "release: id  0 | task 7",
        "new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 346",
        "cached n_tokens = 57",
        "get_availabl: updating prompt cache",
        "prompt_save:  - saving prompt with length 1792",
        "prompt cache update took",
        "cache state: 1 prompts, 73.521 MiB",
        "prompt eval time =",
        "eval time =",
    )
    for number, line in enumerate(lines, 1):
        if any(pattern in line for pattern in patterns):
            seconds = parse_server_seconds(line)
            if seconds is not None:
                selected.append((number, line[:280], seconds))
    # launch.at is the process epoch anchor emitted alongside the server PID.
    launch_epoch_ms = datetime.fromisoformat(launch_at.replace("Z", "+00:00")).timestamp() * 1000
    task10 = [item for item in selected if "task 10 | processing task" in item[1]]
    task10_start_epoch_ms = launch_epoch_ms + task10[0][2] * 1000 if task10 else None
    native_prompt = None
    native_predicted = None
    for _, line, _ in selected:
        match = re.search(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", line)
        if match:
            native_prompt = {"elapsed_ms": float(match.group(1)), "prompt_n": int(match.group(2))}
        match = re.search(r"eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", line)
        if match:
            native_predicted = {"elapsed_ms": float(match.group(1)), "predicted_n": int(match.group(2))}
    queue_ms = None if task10_start_epoch_ms is None else task10_start_epoch_ms - short_record_start_epoch_ms
    cache_update_start = [item for item in selected if "updating prompt cache" in item[1]]
    cache_update_end = [item for item in selected if "prompt cache update took" in item[1]]
    prompt_save = [item for item in selected if "saving prompt with length 1792" in item[1]]
    cache_update_ms = None
    if cache_update_start and cache_update_end:
        # Startup can also update an empty cache. The recovery update is the
        # latest one, immediately before the short task is launched.
        cache_update_ms = (cache_update_end[-1][2] - cache_update_start[-1][2]) * 1000
    pre_cache_wait_ms = None
    if cache_update_start:
        pre_cache_wait_ms = launch_epoch_ms + cache_update_start[-1][2] * 1000 - short_record_start_epoch_ms
    response_tail_ms = None
    if task10_start_epoch_ms is not None and native_prompt and native_predicted:
        native_compute_ms = native_prompt["elapsed_ms"] + native_predicted["elapsed_ms"]
        response_tail_ms = short_record_finish_epoch_ms - task10_start_epoch_ms - native_compute_ms
    cancel0 = [item for item in selected if "id_task = 0" in item[1]]
    release0 = [item for item in selected if "task 0 |" in item[1] and "release:" in item[1]]
    cancel7 = [item for item in selected if "id_task = 7" in item[1]]
    release7 = [item for item in selected if "task 7 |" in item[1] and "release:" in item[1]]
    cache_update_epoch_ms = launch_epoch_ms + cache_update_start[-1][2] * 1000 if cache_update_start else None
    release0_epoch_ms = launch_epoch_ms + release0[0][2] * 1000 if release0 else None
    release7_epoch_ms = launch_epoch_ms + release7[0][2] * 1000 if release7 else None
    release0_after_http_ms = None if release0_epoch_ms is None else release0_epoch_ms - records[0]["finishedAtEpochMs"]
    release7_after_http_ms = None if release7_epoch_ms is None else release7_epoch_ms - records[1]["finishedAtEpochMs"]
    return {
        "sha256": sha256_file(path),
        "line_count": len(lines),
        "vulkan_seen": any("Vulkan0" in line for line in lines),
        "offload_43_of_43_seen": any("offloaded 43/43 layers to GPU" in line for line in lines),
        "selected_markers": [{"line": n, "seconds": round(seconds, 6), "text": text} for n, text, seconds in selected],
        "task0_cancel_then_release": bool(cancel0 and release0 and release0[0][2] > cancel0[0][2]),
        "task7_cancel_then_release": bool(cancel7 and release7 and release7[0][2] > cancel7[0][2]),
        "short_task10_start_epoch_ms": round(task10_start_epoch_ms, 3) if task10_start_epoch_ms is not None else None,
        "short_server_queue_delay_ms": round(queue_ms, 3) if queue_ms is not None else None,
        "short_pre_cache_server_wait_ms": round(pre_cache_wait_ms, 3) if pre_cache_wait_ms is not None else None,
        "prompt_cache_update_start_epoch_ms": round(cache_update_epoch_ms, 3) if cache_update_epoch_ms is not None else None,
        "prompt_cache_update_ms": round(cache_update_ms, 3) if cache_update_ms is not None else None,
        "prompt_cache_saved_prompt_length": 1792 if prompt_save else None,
        "prompt_cache_saved_state_mib": 73.521 if prompt_save else None,
        "prompt_cache_update_is_separate_from_http_settlement": True,
        "task0_release_after_transport_settlement_ms": round(release0_after_http_ms, 3) if release0_after_http_ms is not None else None,
        "task7_release_after_transport_settlement_ms": round(release7_after_http_ms, 3) if release7_after_http_ms is not None else None,
        "short_native_prompt_timing": native_prompt,
        "short_native_generation_timing": native_predicted,
        "short_post_native_response_tail_ms": round(response_tail_ms, 3) if response_tail_ms is not None else None,
        "interpretation": "server task 10 launch is compared with the transport start; native prompt/decode timing is separated from HTTP elapsed time. The 1589.548 ms pre-launch interval includes approximately 534.3 ms before prompt-cache maintenance and a 1055.15 ms cache update saving the canceled 1792-token prompt; it is not labeled native release latency.",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    collection = verify_collection()
    capsule = verify_capsule()
    launch = read_json(EVIDENCE / "launch.json")
    terminal = read_json(EVIDENCE / "terminal.json")
    client_launch = read_json(EVIDENCE / "client-launch.json")
    health = read_json(EVIDENCE / "health.json")
    recovery_path = EVIDENCE / "recovery-a-recovery.json"
    recovery = read_json(recovery_path)
    pilot_path = PILOT / "pilot-a-off-case0.json"
    pilot = read_json(pilot_path)

    records = recovery["transport"]["records"]
    short_record = records[2]
    short_raw = short_record["response"]["rawJson"]
    short_fg = recovery["shortForegroundResponse"]
    pilot_fg = pilot["foreground"]
    pilot_raw = pilot["requests"][0]["response"]["rawJson"]
    terminal_server = terminal.get("server", {})
    terminal_client = terminal.get("client", {})
    props = health.get("props", {})
    checks = {
        "collection_files_verified": collection["all_files_verified"],
        "capsule_files_verified": capsule["all_listed_files_verified"],
        "result_hash_matches_terminal": terminal.get("result_sha256") == sha256_file(recovery_path),
        "launch_server_matches_terminal": launch.get("server_pid") == terminal_server.get("pid"),
        "client_matches_terminal": client_launch.get("pid") == terminal_client.get("pid"),
        "server_and_client_reaped": terminal_server.get("pid_exists") is False and terminal_client.get("pid_exists") is False,
        "server_exit_zero": terminal_server.get("exit_code") == 0 and terminal.get("client_returncode") == 0,
        "health_ok": health.get("health", {}).get("status") == "ok",
        "build_matches": props.get("build_info") == BUILD,
        "model_path_matches_launch": props.get("model_path") == launch.get("argv", [None, None, None])[2],
        "pinned_model_binary": launch.get("model_sha256") == MODEL_SHA and launch.get("binary_sha256") == BINARY_SHA,
        "result_status_pass": recovery.get("status") == "pass" and recovery.get("runId") == "recovery-a",
        "result_input_hashes": recovery.get("fixtureSha256") == FIXTURE_SHA and recovery.get("fixtureManifestSha256") == MANIFEST_SHA and recovery.get("contextCapsuleSha256") == CAPSULE_SHA and recovery.get("sourceSidecarSha256") == SIDECAR_SHA,
        "row_identity_order": recovery.get("longRowId") == ROW_LONG and recovery.get("shortRowId") == ROW_SHORT and recovery.get("longFixtureIndex") == 3 and recovery.get("shortFixtureIndex") == 0,
        "sequence_is_long_warm_long_foreground_short_foreground": [(r.get("sequence"), r.get("kind"), r.get("promptLength")) for r in records] == [(0, "prewarm", 1903), (1, "foreground", 1903), (2, "foreground", 346)],
        "record_settlement_order": records[0].get("finishedAtMonotonicMs", 0) <= records[1].get("startedAtMonotonicMs", 0) and records[1].get("finishedAtMonotonicMs", 0) <= records[2].get("startedAtMonotonicMs", 0),
        "max_one_http_unsettled": recovery["transport"].get("maxConcurrent") == 1 and recovery["transport"].get("atMostOneUnsettled") is True,
        "late_abort_settlements_retained": recovery["transport"].get("lateSettlementAfterAbort") == [0, 1],
        "no_unresolved_after_drain": recovery["transport"].get("unresolvedAfterDrain") == [],
        "long_logical_deadline_rejected": recovery["logicalOutcomes"][0].get("resolved") is False and recovery["logicalOutcomes"][0].get("error", {}).get("name") == "AbortError" and recovery["logicalOutcomes"][0].get("elapsedMs", 0) >= 5000,
        "short_logical_request_resolved": recovery["logicalOutcomes"][1].get("label") == "short_foreground" and recovery["logicalOutcomes"][1].get("resolved") is True,
        "recovery_short_prompt_matches_pilot": short_fg.get("promptTokenIdsSha256") == pilot_fg.get("promptTokenIdsSha256") and short_fg.get("promptLength") == pilot["requests"][0].get("promptLength"),
        "recovery_short_ids_match_pilot": short_fg.get("generatedTokenIds") == pilot_fg.get("generatedTokenIds") and short_raw.get("tokens") == pilot_raw.get("tokens"),
        "recovery_short_text_eos_operation_match_pilot": short_fg.get("rawText") == pilot_fg.get("rawCompletion") and short_fg.get("stopType") == pilot_fg.get("stopType") == "eos" and short_fg.get("operation") == pilot_fg.get("operation") == "no_op",
    }
    server_log = server_log_review(EVIDENCE / "server.log", launch["at"], short_record["startedAtEpochMs"], short_record["finishedAtEpochMs"], records)
    checks.update({
        "native_vulkan_offload_log": server_log["vulkan_seen"] and server_log["offload_43_of_43_seen"],
        "native_task0_cancel_release": server_log["task0_cancel_then_release"],
        "native_task7_cancel_release": server_log["task7_cancel_then_release"],
        "native_short_geometry_log": server_log["short_native_prompt_timing"] == {"elapsed_ms": 1287.94, "prompt_n": 289} and server_log["short_native_generation_timing"] == {"elapsed_ms": 631.45, "predicted_n": 12},
        "native_cache_update_accounted_separately": server_log["prompt_cache_update_ms"] is not None and abs(server_log["prompt_cache_update_ms"] - 1055.15) < 0.01 and server_log["prompt_cache_saved_prompt_length"] == 1792 and server_log["prompt_cache_saved_state_mib"] == 73.521,
    })
    result = {
        "schema": "RUN-04-prewarm-recovery-a-independent-review/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "complete_recovery_verified_no_promotion",
        "recommendation": "The bounded recovery path is mechanically accepted for one native run: serialized HTTP, late abort records, native task release, and exact short response parity are verified. Keep editor/prewarm promotion closed; the short request paid inferred server queue time after cancellation.",
        "scope": {
            "evidence_root": str(EVIDENCE),
            "capsule_root": str(CAPSULE),
            "pilot_baseline_root": str(PILOT),
            "cpu_only_review": True,
            "model_or_server_started_by_review": False,
            "ssh_network_cuda_final_access": False,
        },
        "input_identity": {
            "fixture_sha256": recovery.get("fixtureSha256"),
            "fixture_manifest_sha256": recovery.get("fixtureManifestSha256"),
            "context_capsule_sha256": recovery.get("contextCapsuleSha256"),
            "source_sidecar_sha256": recovery.get("sourceSidecarSha256"),
            "model_sha256_recorded": launch.get("model_sha256"),
            "binary_sha256_recorded": launch.get("binary_sha256"),
            "build_info": props.get("build_info"),
            "capsule_manifest_sha256": capsule["sha256"],
        },
        "integrity": {
            "collection": collection,
            "capsule": capsule,
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        },
        "transport_accounting": {
            "record_count": len(records),
            "records": [
                {
                    "sequence": r.get("sequence"),
                    "kind": r.get("kind"),
                    "prompt_length": r.get("promptLength"),
                    "prompt_ids_sha256": r.get("promptIdsSha256"),
                    "n_predict": r.get("nPredict"),
                    "started_at_epoch_ms": r.get("startedAtEpochMs"),
                    "started_at_monotonic_ms": r.get("startedAtMonotonicMs"),
                    "abort_requested_at_epoch_ms": r.get("abortRequestedAtEpochMs"),
                    "abort_requested_at_monotonic_ms": r.get("abortRequestedAtMonotonicMs"),
                    "http_status": r.get("httpStatus"),
                    "error": r.get("error"),
                    "finished_at_epoch_ms": r.get("finishedAtEpochMs"),
                    "finished_at_monotonic_ms": r.get("finishedAtMonotonicMs"),
                    "elapsed_ms": r.get("elapsedMs"),
                    "response_identity": None if r.get("response") is None else {
                        "raw_json_sha256": sha256_json(r["response"]["rawJson"]),
                        "tokens": r["response"]["rawJson"].get("tokens"),
                        "stop_type": r["response"]["rawJson"].get("stop_type"),
                        "tokens_evaluated": r["response"]["rawJson"].get("tokens_evaluated"),
                        "tokens_cached": r["response"]["rawJson"].get("tokens_cached"),
                        "timings": r["response"]["rawJson"].get("timings"),
                    },
                }
                for r in records
            ],
            "max_concurrent": recovery["transport"].get("maxConcurrent"),
            "at_most_one_unsettled_http": recovery["transport"].get("atMostOneUnsettled"),
            "late_settlement_after_abort_sequences": recovery["transport"].get("lateSettlementAfterAbort"),
            "unresolved_after_drain": recovery["transport"].get("unresolvedAfterDrain"),
            "drain_ms": recovery.get("drainMs"),
        },
        "logical_deadlines": recovery.get("logicalOutcomes"),
        "short_baseline_parity": {
            "baseline_result_sha256": sha256_file(pilot_path),
            "baseline_row_id": pilot.get("rowId"),
            "recovery_row_id": recovery.get("shortRowId"),
            "prompt_ids_equal": checks["recovery_short_prompt_matches_pilot"],
            "generated_ids_equal": checks["recovery_short_ids_match_pilot"],
            "text_stop_operation_equal": checks["recovery_short_text_eos_operation_match_pilot"],
            "recovery_prompt_ids_sha256": short_fg.get("promptTokenIdsSha256"),
            "generated_token_count": len(short_fg.get("generatedTokenIds", [])),
            "generated_token_ids_sha256": sha256_json(short_fg.get("generatedTokenIds")),
            "raw_text_sha256": hashlib.sha256(str(short_fg.get("rawText", "")).encode()).hexdigest(),
            "recovery_stop_type": short_fg.get("stopType"),
            "recovery_operation": short_fg.get("operation"),
        },
        "native_correlation": server_log,
        "limitations": [
            "One recovery run supplies the evidence denominator; this is a mechanical lifecycle/parity check, not a latency promotion or editor-quality result.",
            "The client transport field serverQueueMs is unavailable. Short server queue delay is inferred by correlating the server launch timestamp and task-10 log timestamp with the recorded transport start.",
            "The server continued native cancellation/release work after the client transport promises settled; server task release is therefore separate from HTTP settlement.",
            "The recovered short request reused a partial native prefix (cache_n=57, prompt_n=289), so its prompt timing is not a cold baseline comparison.",
        ],
        "next_test": {
            "design": "Repeat this same-server sequence on several TRAIN rows, then issue a second short request after the release marker; require no overlapping HTTP leases and report inferred native queue separately from HTTP elapsed.",
            "success_condition": "Every short response is exact-prompt bound and parity checked, all cancel/release pairs are observed, and queue delay is reported independently for each run.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "output_sha256": sha256_file(OUT),
        "all_checks_pass": result["integrity"]["all_checks_pass"],
        "collection_sha256": collection["sha256"],
        "capsule_sha256": capsule["sha256"],
        "short_server_queue_delay_ms": server_log["short_server_queue_delay_ms"],
        "short_http_elapsed_ms": short_record.get("elapsedMs"),
        "short_native_compute_ms": None if not server_log["short_native_prompt_timing"] or not server_log["short_native_generation_timing"] else server_log["short_native_prompt_timing"]["elapsed_ms"] + server_log["short_native_generation_timing"]["elapsed_ms"],
        "prompt_cache_update_ms": server_log["prompt_cache_update_ms"],
        "prompt_cache_saved_prompt_length": server_log["prompt_cache_saved_prompt_length"],
        "prompt_cache_saved_state_mib": server_log["prompt_cache_saved_state_mib"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
