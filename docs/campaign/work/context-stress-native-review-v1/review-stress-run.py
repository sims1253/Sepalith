#!/usr/bin/env python3
"""CPU-only independent review of the RUN-05 native stress traces.

The review reads the already completed trace artifacts and the immutable
synthetic fixture.  It never starts a native client or server.  Prompt
tokenization uses only the pinned local Hugging Face tokenizer and compares
all eighteen ``/tokenize`` responses with independently encoded fixture
prompts.  Native completion responses are checked from the captured trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
FIXTURE_PATH = PLAN_ROOT / "docs/campaign/work/context-stress-fixture-v1/context-stress-fixture.json"
TOKEN_REPORT_PATH = PLAN_ROOT / "docs/campaign/work/context-stress-fixture-v1/token-report.json"
NATIVE_MANIFEST_PATH = PLAN_ROOT / "docs/campaign/work/context-stress-native-v1/manifest.json"
NATIVE_WRAPPER_PATH = PLAN_ROOT / "docs/campaign/work/context-stress-native-v1/replay-context-stress.ts"
ROBUST_REPLAY_PATH = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-context-b/replay-context-trace.ts"
TOKENIZER_PATH = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
RUN_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-05-theta0-stress-a")
PROFILES = (2048, 4096, 8192)
MAX_OUTPUT_TOKENS = 192
MANUAL_BOS_ID = 0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    """Hash the compact JSON form used by the TypeScript trace code."""

    return sha256_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_equal(actual: Any, expected: Any, message: str) -> None:
    require(actual == expected, f"{message}: expected {expected!r}, got {actual!r}")


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if root is not None:
        record["relative"] = str(path.relative_to(root))
    return record


def event_source_checks(fixture: dict[str, Any]) -> dict[str, Any]:
    require_equal(fixture["schema"], "sepalith.serving.context-stress-fixture.v1", "fixture schema")
    require_equal(fixture["task"], "RUN-05", "fixture task")
    require(fixture["syntheticOnly"] is True, "fixture must be synthetic-only")
    require_equal(len(fixture["cases"]), 3, "fixture case count")
    require_equal(len(fixture["events"]), 6, "fixture event count")
    require_equal(fixture["selectorPolicy"]["productionDefaultBudgetUtf16Units"], 6000, "production selector budget")
    require(fixture["selectorPolicy"]["stressOnly"] is True, "stress budgets must be marked stress-only")
    require_equal(fixture["nativeProfile"]["maxOutputTokens"], MAX_OUTPUT_TOKENS, "fixture output budget")
    require_equal(fixture["nativeProfile"]["manualBosId"], MANUAL_BOS_ID, "fixture manual BOS")

    document = fixture["document"]
    document_bytes = document["text"].encode("utf-8")
    require_equal(sha256_bytes(document_bytes), document["contentSha256"], "complete synthetic document hash")
    lines = document["text"].splitlines()
    event_by_id = {event["eventId"]: event for event in fixture["events"]}
    require_equal(len(event_by_id), 6, "unique fixture event IDs")
    case_by_id = {case["id"]: case for case in fixture["cases"]}
    require_equal(set(case_by_id), {"near-2k", "near-4k", "near-8k"}, "fixture case IDs")

    checks = 0
    for event in fixture["events"]:
        checks += 1
        after = event["after"]
        require(after["promptText"].endswith("<[fim-middle]>\n"), f"{event['eventId']} FIM boundary")
        require_equal(sha256_bytes(after["promptText"].encode("utf-8")), after["promptSha256"], f"{event['eventId']} prompt hash")
        require_equal(after["promptText"], event_by_id[event["eventId"]]["after"]["promptText"], f"{event['eventId']} prompt identity")
        selection = after["selection"]
        require(selection["requiredOverflow"] is False, f"{event['eventId']} required source overflow")
        require(selection["usedUtf16Units"] <= selection["budgetUtf16Units"], f"{event['eventId']} selector budget")
        selected_lines: list[str] = []
        for span in selection["spans"]:
            require(0 <= span["startLine"] <= span["endLine"] < len(lines), f"{event['eventId']} source span bounds")
            selected_lines.extend(lines[span["startLine"] : span["endLine"] + 1])
        selected = selection["prefix"] + selection["region"] + selection["suffix"]
        require_equal(selected_lines, selected, f"{event['eventId']} complete source spans")
        require_equal(event["afterDocument"]["contentSha256"], document["contentSha256"], f"{event['eventId']} complete document identity")
        identity = after["identity"]
        context = after["context"]["replacement_range"]
        require_equal(identity["uri"], context["uri"], f"{event['eventId']} source URI")
        require_equal(identity["version"], context["document_version"], f"{event['eventId']} source version")
        require_equal(identity["contentSha256"], context["content_sha256"], f"{event['eventId']} source content hash")
        require_equal(identity["contentSha256"], document["contentSha256"], f"{event['eventId']} source document hash")

    for case_id in case_by_id:
        pair = [event for event in fixture["events"] if event["caseId"] == case_id]
        require_equal(len(pair), 2, f"{case_id} event pair")
        require_equal(pair[0]["kind"], "baseline", f"{case_id} baseline kind")
        require_equal(pair[1]["kind"], "unchanged_repeat_control", f"{case_id} repeat kind")
        require_equal(pair[0]["after"]["promptText"], pair[1]["after"]["promptText"], f"{case_id} repeat prompt")
        require_equal(pair[0]["cursorAfter"], pair[1]["cursorAfter"], f"{case_id} repeat cursor")

    return {
        "completeDocumentSha256": document["contentSha256"],
        "completeDocumentUtf8Bytes": len(document_bytes),
        "completeDocumentLineCount": len(lines),
        "events": len(fixture["events"]),
        "cases": len(fixture["cases"]),
        "sourceSpansChecked": checks,
        "productionDefaultSourceBudgetUtf16Units": 6000,
        "stressSourceBudgetsUtf16Units": [case["selectorBudgetUtf16Units"] for case in fixture["cases"]],
        "cursorAfter": fixture["cursor"],
        "replacementRangeUris": sorted({event["after"]["context"]["replacement_range"]["uri"] for event in fixture["events"]}),
        "unsupportedProfileLabels": {str(case["contextSize"]): case["supportClass"] for case in fixture["cases"]},
    }


def load_pinned_tokenizer() -> Any:
    # Import only after all path checks.  local_files_only prevents network use.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )


def tokenize_fixture_events(fixture: dict[str, Any], tokenizer: Any) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    ids_by_event: dict[str, list[int]] = {}
    records: list[dict[str, Any]] = []
    for event in fixture["events"]:
        prompt = event["after"]["promptText"]
        ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False, split_special_tokens=True)]
        require(all(0 <= token < 130560 for token in ids), f"{event['eventId']} tokenizer vocabulary range")
        require(0 not in ids, f"{event['eventId']} tokenizer unexpectedly added BOS")
        ids_by_event[event["eventId"]] = ids
        records.append(
            {
                "eventId": event["eventId"],
                "caseId": event["caseId"],
                "kind": event["kind"],
                "promptTextSha256": event["after"]["promptSha256"],
                "promptTokenCountWithoutBos": len(ids),
                "promptTokenIdsSha256WithoutBos": sha256_json(ids),
                "promptTokenCountWithManualBos": len(ids) + 1,
                "promptTokenIdsSha256WithManualBos": sha256_json([MANUAL_BOS_ID, *ids]),
            }
        )
    return ids_by_event, records


def row_transport_calls(row: dict[str, Any], path: str) -> list[dict[str, Any]]:
    return [call for call in row.get("transportCalls", []) if call.get("path") == path]


def profile_expectations(fixture: dict[str, Any], profile: int) -> tuple[set[str], set[str]]:
    support_class = {2048: "native_diagnostic_2048", 4096: "primary_editor_4096", 8192: "stress_only_8192"}[profile]
    accepted: set[str] = set()
    overflow: set[str] = set()
    for case in fixture["cases"]:
        if support_class in case["expectedOverflowProfiles"]:
            overflow.add(case["id"])
        else:
            accepted.add(case["id"])
    return accepted, overflow


def parse_graph_log(log: str) -> dict[str, Any]:
    graph_nodes = [int(value) for value in re.findall(r"graph nodes\s*=\s*(\d+)", log)]
    graph_splits = [int(value) for value in re.findall(r"graph splits\s*=\s*(\d+)", log)]
    graph_reuse = [int(value) for value in re.findall(r"graphs reused\s*=\s*(\d+)", log)]
    return {
        "useGraphs": bool(re.search(r"USE_GRAPHS\s*=\s*1", log)),
        "graphNodes": graph_nodes,
        "graphSplits": graph_splits,
        "graphsReused": graph_reuse,
        "cleanupBeforeExit": log.count("cleaning up before exit"),
    }


def inspect_native_profile(
    fixture: dict[str, Any],
    profile: int,
    expected_by_event: dict[str, list[int]],
    event_by_id: dict[str, dict[str, Any]],
    native_manifest: dict[str, Any],
) -> dict[str, Any]:
    profile_dir = RUN_ROOT / str(profile)
    trace_path = profile_dir / "trace.json"
    trace = load_json(trace_path)
    rows = trace["rows"]
    require_equal(len(rows), 6, f"{profile} row count")
    require_equal(trace["fixtureSha256"], native_manifest["fixtureSha256"], f"{profile} fixture identity")
    require_equal(trace["fixtureSha256"], sha256_file(FIXTURE_PATH), f"{profile} fixture hash")
    require_equal(trace["wrapperSha256"], native_manifest["wrapperSha256"], f"{profile} wrapper identity")
    require_equal(trace["manifestSha256"], sha256_file(NATIVE_MANIFEST_PATH), f"{profile} native manifest identity")
    require_equal(trace["requestedContextSize"], profile, f"{profile} requested context")
    require_equal(trace["serverProps"]["default_generation_settings"]["n_ctx"], profile, f"{profile} props context")
    require_equal(trace["serverProps"]["endpoint_props"], False, f"{profile} endpoint props mode")
    require_equal(trace["serverProps"]["model_ftype"], "Q8_0", f"{profile} model format")
    require_equal(trace["serverProps"]["build_info"], "b10453-3cb7ffb1a", f"{profile} llama build")

    accepted_cases, overflow_cases = profile_expectations(fixture, profile)
    completion_records: list[dict[str, Any]] = []
    token_records: list[dict[str, Any]] = []
    malformed_outputs = 0
    for row in rows:
        event_id = row["eventId"]
        event = event_by_id[event_id]
        expected_ids = expected_by_event[event_id]
        tokenize_calls = row_transport_calls(row, "/tokenize")
        completion_calls = row_transport_calls(row, "/completion")
        require_equal(len(tokenize_calls), 1, f"{profile}/{event_id} tokenize call count")
        tokenize_call = tokenize_calls[0]
        body = tokenize_call["body"]
        require_equal(body["content"], event["after"]["promptText"], f"{profile}/{event_id} tokenize source text")
        require_equal(body["add_special"], False, f"{profile}/{event_id} add_special flag")
        require_equal(body["parse_special"], False, f"{profile}/{event_id} parse_special flag")
        require_equal(body["with_pieces"], False, f"{profile}/{event_id} with_pieces flag")
        native_token_ids = tokenize_call["response"]["tokens"]
        require_equal(native_token_ids, expected_ids, f"{profile}/{event_id} exact HF prompt tokenization")
        token_records.append(
            {
                "profile": profile,
                "eventId": event_id,
                "caseId": event["caseId"],
                "kind": event["kind"],
                "promptTextSha256": event["after"]["promptSha256"],
                "tokenCountWithoutBos": len(expected_ids),
                "tokenIdsSha256WithoutBos": sha256_json(expected_ids),
                "nativeTokenizeResponseSha256": tokenize_call["responseSha256"],
                "exactHfTokenization": True,
            }
        )
        should_overflow = event["caseId"] in overflow_cases
        require_equal(row["contextOverflowRejected"], should_overflow, f"{profile}/{event_id} overflow classification")
        require_equal(row["duplicateWarmControl"], event["kind"] == "unchanged_repeat_control", f"{profile}/{event_id} duplicate marker")
        require(row["timeout"] is False and row["cancelled"] is False, f"{profile}/{event_id} timeout/cancel status")

        if should_overflow:
            require_equal(row["promptTextSha256"], None, f"{profile}/{event_id} overflow prompt text field")
            require_equal(len(completion_calls), 0, f"{profile}/{event_id} overflow completion dispatch")
            require_equal(row["responseApplicability"], "error", f"{profile}/{event_id} overflow applicability")
            require_equal(row["protocolEvidence"], "rejected", f"{profile}/{event_id} overflow protocol evidence")
            error = row["error"]
            require(isinstance(error, dict), f"{profile}/{event_id} overflow error object")
            require_equal(error["name"], "NativeServingError", f"{profile}/{event_id} overflow error name")
            require_equal(error["code"], "context_budget", f"{profile}/{event_id} overflow error code")
            expected_total = len(expected_ids) + 1 + MAX_OUTPUT_TOKENS
            require(f"prompt tokens {len(expected_ids) + 1} plus output budget {MAX_OUTPUT_TOKENS} exceeds managed context {profile}" in error["message"], f"{profile}/{event_id} overflow detail")
            require(row["promptTokenIds"] is None and row["promptTokenIdsSha256"] is None, f"{profile}/{event_id} overflow prompt transport fields")
            require(row["rawResponseSha256"] is None and row["generatedTokenIdsSha256"] is None, f"{profile}/{event_id} overflow response fields")
            require(row["parserStatus"] is None and row["stopType"] is None and row["planMode"] is None, f"{profile}/{event_id} overflow parser/plan fields")
            require_equal(row["completionDispatchStarted"], False, f"{profile}/{event_id} completion dispatch marker")
            require(expected_total > profile, f"{profile}/{event_id} arithmetic overflow control")
            continue

        require_equal(len(completion_calls), 1, f"{profile}/{event_id} completion call count")
        require_equal(row["promptTextSha256"], event["after"]["promptSha256"], f"{profile}/{event_id} prompt text hash")
        completion_call = completion_calls[0]
        completion_body = completion_call["body"]
        expected_prompt_ids = [MANUAL_BOS_ID, *expected_ids]
        require_equal(completion_body["prompt"], expected_prompt_ids, f"{profile}/{event_id} manual BOS prompt")
        require_equal(completion_body["n_predict"], MAX_OUTPUT_TOKENS, f"{profile}/{event_id} output budget")
        require_equal(completion_body["temperature"], 0, f"{profile}/{event_id} deterministic temperature")
        response = completion_call["response"]
        response_tokens = response["tokens"]
        response_content = response["content"]
        require_equal(row["promptTokenIds"], expected_prompt_ids, f"{profile}/{event_id} native prompt IDs")
        require_equal(row["promptTokenIdsSha256"], sha256_json(expected_prompt_ids), f"{profile}/{event_id} native prompt ID hash")
        require_equal(row["generatedTokenIds"], response_tokens, f"{profile}/{event_id} generated token capture")
        require_equal(row["generatedTokenIdsSha256"], sha256_json(response_tokens), f"{profile}/{event_id} generated token hash")
        require_equal(row["rawResponseSha256"], sha256_bytes(response_content.encode("utf-8")), f"{profile}/{event_id} raw response hash")
        require_equal(response["stop_type"], "eos", f"{profile}/{event_id} response stop type")
        require_equal(response["stopping_word"], "", f"{profile}/{event_id} response stopping word")
        require_equal(response["truncated"], False, f"{profile}/{event_id} response truncation")
        require_equal(row["stopType"], "eos", f"{profile}/{event_id} trace stop type")
        require_equal(row["parserStatus"], "accepted", f"{profile}/{event_id} parser status")
        require_equal(row["operation"], "no_op", f"{profile}/{event_id} operation")
        require_equal(row["responseApplicability"], "fresh", f"{profile}/{event_id} response applicability")
        require_equal(row["protocolEvidence"], "no_op", f"{profile}/{event_id} protocol evidence")
        require_equal(row["planMode"], "none", f"{profile}/{event_id} plan mode")
        require(isinstance(row["planHash"], str) and len(row["planHash"]) == 64, f"{profile}/{event_id} plan hash")
        require_equal(row["completionDispatchStarted"], True, f"{profile}/{event_id} completion dispatch marker")
        require(row["qualityDenominatorEligible"] is (event["kind"] == "baseline"), f"{profile}/{event_id} quality denominator marker")
        require_equal(row["error"], None, f"{profile}/{event_id} completed error")
        completion_records.append(
            {
                "profile": profile,
                "eventId": event_id,
                "caseId": event["caseId"],
                "kind": event["kind"],
                "duplicateWarmControl": row["duplicateWarmControl"],
                "elapsedMs": row["elapsedMs"],
                "promptTokenCountWithManualBos": len(expected_prompt_ids),
                "promptTokenIdsSha256": row["promptTokenIdsSha256"],
                "generatedTokenCount": len(response_tokens),
                "generatedTokenIdsSha256": row["generatedTokenIdsSha256"],
                "rawResponseSha256": row["rawResponseSha256"],
                "transportResponseSha256": completion_call["responseSha256"],
                "stopType": row["stopType"],
                "parserStatus": row["parserStatus"],
                "operation": row["operation"],
                "planMode": row["planMode"],
                "planHash": row["planHash"],
                "qualityDenominatorEligible": row["qualityDenominatorEligible"],
            }
        )

    expected_completed = {2048: 2, 4096: 4, 8192: 6}[profile]
    expected_overflows = {2048: 4, 4096: 2, 8192: 0}[profile]
    require_equal(len(completion_records), expected_completed, f"{profile} completion count")
    require_equal(sum(bool(row["contextOverflowRejected"]) for row in rows), expected_overflows, f"{profile} overflow count")
    denominator = trace["denominator"]
    require_equal(denominator["events"], 6, f"{profile} denominator events")
    require_equal(denominator["freshResponses"], expected_completed, f"{profile} fresh denominator")
    require_equal(denominator["duplicateWarmControlsExcluded"], 3, f"{profile} duplicate denominator")
    require_equal(denominator["contextOverflowsRejected"], expected_overflows, f"{profile} overflow denominator")
    require_equal(denominator["protocolRejected"], expected_overflows, f"{profile} generic protocol denominator")
    require_equal(denominator["protocolRejected"], denominator["contextOverflowsRejected"], f"{profile} protocol rejection split")
    require_equal(denominator["qualityClaims"], "none; this trace is latency/applicability/protocol evidence only", f"{profile} quality claim status")
    malformed_outputs += sum(
        1
        for record in completion_records
        if record["parserStatus"] != "accepted" or record["stopType"] != "eos" or record["operation"] != "no_op"
    )
    require_equal(malformed_outputs, 0, f"{profile} malformed model output count")

    terminal_path = profile_dir / "terminal.json"
    terminal = load_json(terminal_path)
    require_equal(terminal["context"], profile, f"{profile} terminal context")
    require_equal(terminal["trace_sha256"], sha256_file(trace_path), f"{profile} terminal trace hash")
    require_equal(terminal["client"]["exit_code"], 0, f"{profile} client clean exit")
    require_equal(terminal["server"]["exit_code"], 0, f"{profile} server clean exit")
    require(terminal["client"]["pid_exists"] is False and terminal["server"]["pid_exists"] is False, f"{profile} client/server process cleanup")
    require(all(child["exists"] is False for child in terminal["server"]["children"]), f"{profile} child process cleanup")

    launch = load_json(profile_dir / "server-launch.json")
    argv = launch["argv"]
    require_equal(argv[0:4], ["timeout", "--foreground", "--signal=TERM", "--kill-after=10s"], f"{profile} launch wrapper")
    require_equal(argv[argv.index("--parallel") + 1], "1", f"{profile} parallel launch")
    require_equal(argv[argv.index("-c") + 1], str(profile), f"{profile} native context launch")
    require_equal(argv[argv.index("-b") + 1], "256", f"{profile} batch launch")
    require_equal(argv[argv.index("-ub") + 1], "256", f"{profile} ubatch launch")
    require_equal(argv[argv.index("-ngl") + 1], "99", f"{profile} offload launch")
    require_equal(argv[argv.index("-lv") + 1], "4", f"{profile} log verbosity launch")
    env_paths = sorted(profile_dir.glob("server-env-*.json"))
    require_equal(len(env_paths), 1, f"{profile} server environment artifact count")
    env = load_json(env_paths[0])
    require_equal(env, ["GGML_CUDA_GRAPH_OPT=0"], f"{profile} CUDA graph opt environment")
    server_log_path = profile_dir / "server.log"
    graph = parse_graph_log(server_log_path.read_text(encoding="utf-8", errors="replace"))
    require(graph["useGraphs"], f"{profile} native graph use evidence")
    require(graph["graphNodes"] and all(nodes == 1308 for nodes in graph["graphNodes"]), f"{profile} graph node evidence")
    require(graph["graphSplits"] and all(splits == 2 for splits in graph["graphSplits"]), f"{profile} graph split evidence")
    require(graph["graphsReused"], f"{profile} graph reuse evidence")
    require_equal(graph["cleanupBeforeExit"], 1, f"{profile} server cleanup evidence")

    return {
        "profile": profile,
        "supportClass": trace["supportClass"],
        "deadlineMs": trace["deadlineMs"],
        "expectedAcceptedCaseIds": sorted(accepted_cases),
        "expectedOverflowCaseIds": sorted(overflow_cases),
        "rows": len(rows),
        "exactHfPromptTokenizations": token_records,
        "completionOutputs": completion_records,
        "denominator": denominator,
        "malformedModelOutputs": malformed_outputs,
        "terminal": {
            "path": str(terminal_path),
            "traceSha256": terminal["trace_sha256"],
            "clientExitCode": terminal["client"]["exit_code"],
            "serverExitCode": terminal["server"]["exit_code"],
            "clientPidExists": terminal["client"]["pid_exists"],
            "serverPidExists": terminal["server"]["pid_exists"],
            "childrenExist": [child["exists"] for child in terminal["server"]["children"]],
            "seconds": terminal["seconds"],
        },
        "launch": {
            "path": str(profile_dir / "server-launch.json"),
            "wrapper": argv[:4],
            "parallel": argv[argv.index("--parallel") + 1],
            "context": argv[argv.index("-c") + 1],
            "batch": argv[argv.index("-b") + 1],
            "ubatch": argv[argv.index("-ub") + 1],
            "ngl": argv[argv.index("-ngl") + 1],
            "logVerbosity": argv[argv.index("-lv") + 1],
        },
        "graphEvidence": {
            **graph,
            "environment": env,
            "serverLogSha256": sha256_file(server_log_path),
            "serverEnvironmentPath": str(env_paths[0]),
        },
        "artifactHashes": {
            name: sha256_file(profile_dir / name)
            for name in ("trace.json", "props.json", "terminal.json", "server-launch.json", "client-launch.json")
        },
    }


def compare_outputs(profiles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for profile in PROFILES:
        for output in profiles[profile]["completionOutputs"]:
            by_case.setdefault(output["caseId"], []).append(output)
    comparisons: dict[str, Any] = {}
    for case_id, outputs in sorted(by_case.items()):
        signatures = {
            (
                output["rawResponseSha256"],
                output["generatedTokenIdsSha256"],
                output["stopType"],
                output["parserStatus"],
                output["operation"],
                output["planMode"],
                output["planHash"],
            )
            for output in outputs
        }
        require_equal(len(signatures), 1, f"{case_id} same-case output signature")
        comparisons[case_id] = {
            "profilesSeen": sorted({output["profile"] for output in outputs}),
            "eventsSeen": [output["eventId"] for output in outputs],
            "outputCountIncludingRepeats": len(outputs),
            "exactAcrossAllocationsAndRepeats": True,
            "rawResponseSha256": outputs[0]["rawResponseSha256"],
            "generatedTokenIdsSha256": outputs[0]["generatedTokenIdsSha256"],
            "stopType": outputs[0]["stopType"],
            "parserStatus": outputs[0]["parserStatus"],
            "operation": outputs[0]["operation"],
            "planMode": outputs[0]["planMode"],
            "planHash": outputs[0]["planHash"],
        }
    require_equal(set(comparisons), {"near-2k", "near-4k", "near-8k"}, "same-case comparison coverage")
    return comparisons


def timing_summary(profiles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for profile in PROFILES:
        rows = profiles[profile]["completionOutputs"]
        fresh = [float(row["elapsedMs"]) for row in rows if not row["duplicateWarmControl"]]
        duplicate = [float(row["elapsedMs"]) for row in rows if row["duplicateWarmControl"]]
        summary[str(profile)] = {
            "freshCompletedElapsedMs": fresh,
            "duplicateCompletedElapsedMs": duplicate,
            "freshCompletedCount": len(fresh),
            "duplicateCompletedCount": len(duplicate),
            "freshCompletedMeanMs": statistics.fmean(fresh) if fresh else None,
            "duplicateCompletedMeanMs": statistics.fmean(duplicate) if duplicate else None,
            "freshCompletedMedianMs": statistics.median(fresh) if fresh else None,
            "duplicateCompletedMedianMs": statistics.median(duplicate) if duplicate else None,
            "qualityEligibleFreshCount": sum(bool(row["qualityDenominatorEligible"]) for row in rows),
            "duplicateControlsExcludedFromQuality": sum(not bool(row["qualityDenominatorEligible"]) for row in rows),
        }
    return summary


def build_input_manifest(fixture: dict[str, Any], native_manifest: dict[str, Any]) -> dict[str, Any]:
    tokenizer_files = [TOKENIZER_PATH / "tokenizer.json", TOKENIZER_PATH / "tokenizer_config.json"]
    require_equal(sha256_file(tokenizer_files[0]), fixture["source"]["tokenizerJsonSha256"], "tokenizer.json source hash")
    require_equal(sha256_file(tokenizer_files[1]), fixture["source"]["tokenizerConfigSha256"], "tokenizer_config.json source hash")
    profile_records: dict[str, Any] = {}
    names = ("trace.json", "props.json", "terminal.json", "server-launch.json", "client-launch.json", "server-children.json", "server.log")
    for profile in PROFILES:
        profile_dir = RUN_ROOT / str(profile)
        files = {name: file_record(profile_dir / name) for name in names}
        env_paths = sorted(profile_dir.glob("server-env-*.json"))
        require_equal(len(env_paths), 1, f"{profile} environment manifest count")
        files[env_paths[0].name] = file_record(env_paths[0])
        profile_records[str(profile)] = files
    return {
        "schema": "sepalith.serving.context-stress-native-review.input-manifest.v1",
        "immutableInputs": True,
        "readOnlyReview": True,
        "fixture": file_record(FIXTURE_PATH),
        "tokenReport": file_record(TOKEN_REPORT_PATH),
        "stressNativeManifest": file_record(NATIVE_MANIFEST_PATH),
        "stressNativeWrapper": file_record(NATIVE_WRAPPER_PATH),
        "robustReplay": file_record(ROBUST_REPLAY_PATH),
        "tokenizer": {
            "path": str(TOKENIZER_PATH),
            "localFilesOnly": True,
            "addSpecialTokens": False,
            "splitSpecialTokens": True,
            "manualBosId": MANUAL_BOS_ID,
            "tokenizerJson": file_record(tokenizer_files[0]),
            "tokenizerConfig": file_record(tokenizer_files[1]),
        },
        "nativeRunRoot": str(RUN_ROOT),
        "nativeProfiles": profile_records,
        "profileSupportClasses": {
            str(profile): native_manifest["profileExpectations"][str(profile)]["supportClass"] for profile in PROFILES
        },
        "noWeightReads": True,
        "noNativeExecution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture = load_json(FIXTURE_PATH)
    native_manifest = load_json(NATIVE_MANIFEST_PATH)
    require_equal(native_manifest["fixtureSha256"], sha256_file(FIXTURE_PATH), "native manifest fixture hash")
    require_equal(native_manifest["wrapperSha256"], sha256_file(NATIVE_WRAPPER_PATH), "native manifest wrapper hash")
    require_equal(native_manifest["robustReplaySha256"], sha256_file(ROBUST_REPLAY_PATH), "native manifest robust replay hash")
    fixture_source = event_source_checks(fixture)
    tokenizer = load_pinned_tokenizer()
    expected_by_event, independent_tokenization = tokenize_fixture_events(fixture, tokenizer)
    event_by_id = {event["eventId"]: event for event in fixture["events"]}

    profiles: dict[int, dict[str, Any]] = {}
    tokenization_records: list[dict[str, Any]] = []
    for profile in PROFILES:
        profiles[profile] = inspect_native_profile(fixture, profile, expected_by_event, event_by_id, native_manifest)
        tokenization_records.extend(profiles[profile]["exactHfPromptTokenizations"])
    require_equal(len(tokenization_records), 18, "exact HF tokenization denominator")
    require(all(record["exactHfTokenization"] for record in tokenization_records), "exact HF tokenization checks")

    all_outputs = [output for profile in PROFILES for output in profiles[profile]["completionOutputs"]]
    require_equal(len(all_outputs), 12, "completion output denominator")
    require_equal(sum(len(profiles[profile]["completionOutputs"]) for profile in PROFILES), 12, "completion output total")
    comparisons = compare_outputs(profiles)
    timing = timing_summary(profiles)
    input_manifest = build_input_manifest(fixture, native_manifest)
    input_manifest_path = out_dir / "input-manifest.json"
    input_manifest_path.write_text(json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")

    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema": "sepalith.serving.context-stress-native-review.v1",
        "task": "RUN-05",
        "reviewMode": "independent CPU-only artifact review",
        "reviewedAt": "2026-09-13",
        "sourcePolicy": {
            "productionDefaultSourceBudgetUtf16Units": 6000,
            "stressSourceBudgetIsDiagnosticOnly": True,
            "qualityPromotion": False,
            "nativeExecutionByReviewer": False,
        },
        "inputManifest": {
            "path": str(input_manifest_path),
            "sha256": sha256_file(input_manifest_path),
        },
        "immutableFixture": {
            "path": str(FIXTURE_PATH),
            "sha256": sha256_file(FIXTURE_PATH),
            "sourceChecks": fixture_source,
            "tokenizer": {
                "path": str(TOKENIZER_PATH),
                "localFilesOnly": True,
                "addSpecialTokens": False,
                "splitSpecialTokens": True,
                "manualBosId": MANUAL_BOS_ID,
                "independentPromptTokenizations": independent_tokenization,
            },
        },
        "nativeWrapper": {
            "path": str(NATIVE_WRAPPER_PATH),
            "sha256": sha256_file(NATIVE_WRAPPER_PATH),
            "robustReplayPath": str(ROBUST_REPLAY_PATH),
            "robustReplaySha256": sha256_file(ROBUST_REPLAY_PATH),
            "manifestPath": str(NATIVE_MANIFEST_PATH),
            "manifestSha256": sha256_file(NATIVE_MANIFEST_PATH),
        },
        "exactHfPromptTokenizations": {
            "count": len(tokenization_records),
            "expectedCount": 18,
            "allExact": True,
            "records": tokenization_records,
        },
        "completionReview": {
            "count": len(all_outputs),
            "expectedCount": 12,
            "expectedByProfile": {"2048": 2, "4096": 4, "8192": 6},
            "recordsByProfile": {str(profile): profiles[profile]["completionOutputs"] for profile in PROFILES},
            "sameCaseOutputComparisons": comparisons,
            "malformedModelOutputs": sum(profiles[profile]["malformedModelOutputs"] for profile in PROFILES),
            "rawResponseFieldsCaptured": ["rawResponseSha256", "transportResponseSha256", "generatedTokenIdsSha256", "stopType", "parserStatus", "operation", "planMode", "planHash"],
        },
        "overflowReview": {
            "expectedInputOverflowRejectionsByProfile": {"2048": 4, "4096": 2, "8192": 0},
            "genericProtocolRejectedIncludesInputOverflow": True,
            "malformedModelOutputsAreSeparate": True,
            "malformedModelOutputs": 0,
            "profileRows": {
                str(profile): {
                    "expectedAcceptedCaseIds": profiles[profile]["expectedAcceptedCaseIds"],
                    "expectedOverflowCaseIds": profiles[profile]["expectedOverflowCaseIds"],
                    "contextOverflowsRejected": profiles[profile]["denominator"]["contextOverflowsRejected"],
                    "protocolRejected": profiles[profile]["denominator"]["protocolRejected"],
                }
                for profile in PROFILES
            },
        },
        "timingReview": {
            "freshAndDuplicateControlsKeptSeparate": True,
            "descriptiveOnly": True,
            "profiles": timing,
        },
        "nativeProfiles": {str(profile): profiles[profile] for profile in PROFILES},
        "reviewScript": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "checks": [
            "immutable complete synthetic source spans and document identity",
            "18 exact local HF prompt tokenizations from fixture prompts",
            "manual BOS 0 added exactly once for completion prompt IDs",
            "12 captured completion outputs with stop/parser/operation/plan checks",
            "input context_budget rejections split from generic protocolRejected denominator",
            "same-case raw/generated output signatures equal across allocations and repeats",
            "fresh completion timing and duplicate warm controls reported separately",
            "GGML_CUDA_GRAPH_OPT=0 plus native graph log and clean-exit evidence",
        ],
        "result": "PASS_CPU_REVIEW_NATIVE_ARTIFACTS_CONSISTENT",
        "promotion": "none; 8192 remains stress_only_8192 and outside the 6000 UTF-16 production source budget",
    }
    report_path = out_dir / "review-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": report["result"],
        "fixtureSha256": report["immutableFixture"]["sha256"],
        "inputManifestSha256": report["inputManifest"]["sha256"],
        "reviewScriptSha256": report["reviewScript"]["sha256"],
        "exactHfPromptTokenizations": report["exactHfPromptTokenizations"]["count"],
        "completionOutputs": report["completionReview"]["count"],
        "overflowRejections": report["overflowReview"]["expectedInputOverflowRejectionsByProfile"],
        "malformedModelOutputs": report["completionReview"]["malformedModelOutputs"],
        "sameCaseComparisons": len(report["completionReview"]["sameCaseOutputComparisons"]),
    }, indent=2))


if __name__ == "__main__":
    main()
