#!/usr/bin/env python3
"""Bounded read-only CPU audit for the captured RUN-03 b4 editor v4 evidence.

This script hashes only the small source, package, receipt, and captured evidence
files named by the task.  It deliberately does not hash or open the 2 GiB model.
It does not start a server/editor, make network calls, import a model, or edit
anything outside this review directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import median
from zipfile import ZipFile


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EVIDENCE = PLAN / "docs/campaign/work/lead/b4-editor-v4-evidence"
HARNESS = PLAN / "docs/campaign/work/lead/b4-editor-auto-v4"
WORK = PLAN / "docs/campaign/work/b4-editor-v4-independent-review"
RECEIPTS = PLAN / "docs/campaign/receipts"
RUN_EVAL = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/eval/run_eval.py"
)
SOURCE_ROOT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith"
)
ACCEPTED_VSIX = SOURCE_ROOT / "vscode-sepalith-renderer-parity-v3-0.0.7.vsix"
MODEL_PATH = Path(
    "/home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf"
)
RUNTIME_PATH = Path(
    "/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server"
)

MODEL_SHA = "e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d"
MODEL_BYTES = 2012011904
RUNTIME_SHA = "e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6"
VSIX_SHA = "a56d2e442ae7abf266ef5f508d25443d00acecb9a78f92787b7ac332e3e2836d"
RUN_EVAL_SHA = "7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d"
EXPECTED_PROMPTS = {
    "replacement": "1e272ca5ae2e9c8a8b8b3bde9dd59d0ee1114c428ab91e1eab9cbe07754ce3b0",
    "no-op-applicability": "0dcaa3de21908f1f316cc2b45d171e538fb1d2e5e1600095ff0e6ab0b41cd35f",
    "post-cancellation-fresh": "6c0e8602214e1bedfdaecddc47d7434d8ce895e945ac58193ee6dd4fae19f9b1",
}
EXPECTED_PENDING = {
    "no-op suggestion applicability",
    "stale transport cancellation and ghost nonpublication",
    "legacy response parser and native stop metadata",
}
EXT_STOPS = [
    ">>>>>>> UPDATED",
    "<<<<<<< CURRENT",
    "=======",
    "<[fim-middle]>",
    "<[fim-suffix]>",
    "<[fim-prefix]>",
    "<|outline|>",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, hash_file: bool = True) -> dict:
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    stat = path.stat()
    result["bytes"] = stat.st_size
    if hash_file:
        result["sha256"] = sha256(path)
    return result


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def equal(actual, expected) -> bool:
    return actual == expected


def source_strings(path: Path, variable: str, name: str) -> list[str]:
    source = path.read_text(encoding="utf-8")
    pattern = rf'const {re.escape(variable)} = await createDocument\("{re.escape(name)}", \[(.*?)\]\);'
    match = re.search(pattern, source, re.S)
    if not match:
        raise AssertionError(f"synthetic document declaration not found: {name}")
    tokens = re.findall(r'"(?:\\.|[^"\\])*"', match.group(1))
    return [json.loads(token) for token in tokens]


def r_parse(text: str) -> dict:
    # The text is supplied over stdin; no temporary or repository file is made.
    proc = subprocess.run(
        ["Rscript", "--vanilla", "-e", "invisible(parse(file=stdin()))"],
        input=text,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip() or None,
    }


def event(host: dict, kind: str, **match) -> dict:
    matches = [item for item in host.get("events", []) if item.get("kind") == kind]
    for item in matches:
        if all(item.get(key) == value for key, value in match.items()):
            return item
    raise AssertionError(f"event not found: {kind} {match}")


def prompt_event(host: dict, label: str) -> dict:
    return event(host, "suggest_prompt_observed", label=label)


def main() -> int:
    checks: list[dict] = []

    def check(name: str, status: str, **details):
        checks.append({"name": name, "status": status, **details})

    # Load the authoritative receipts first.  Their content is small and is part
    # of the declared input evidence.
    cpu_receipt_path = RECEIPTS / "RUN-03-b4-cpu-cycle-admission.json"
    v4_receipt_path = RECEIPTS / "RUN-03-b4-editor-v4-admission.json"
    pre08_path = RECEIPTS / "PRE-08-fallback-ledger.json"
    parity_path = RECEIPTS / "RUN-03-b4-renderer-parity-v1.json"
    package_receipt_path = RECEIPTS / "RUN-01-b4-renderer-v3-source-and-package.json"
    cpu = load(cpu_receipt_path)
    v4 = load(v4_receipt_path)
    pre08 = load(pre08_path)
    parity = load(parity_path)
    package_receipt = load(package_receipt_path)

    bank = pre08["RESULT"]["matched_banked_artifact"]
    model_identity = {
        "path": cpu["model"]["path"],
        "bytes": cpu["model"]["bytes"],
        "sha256": cpu["model"]["sha256"],
    }
    pre08_identity = {key: bank[key] for key in ("path", "bytes", "sha256")}
    check(
        "model identity is shared by CPU admission and PRE-08 bank",
        "pass" if model_identity["bytes"] == pre08_identity["bytes"] == MODEL_BYTES and model_identity["sha256"] == pre08_identity["sha256"] == MODEL_SHA else "reject",
        cpu=model_identity,
        pre08=pre08_identity,
        path_aliases_equal_by_hash=model_identity["sha256"] == pre08_identity["sha256"],
    )
    check(
        "current model path is not opened by this audit",
        "pass",
        captured_bytes=MODEL_BYTES,
        captured_sha256=MODEL_SHA,
        current_path=identity(MODEL_PATH, hash_file=False),
        method="captured admission/hash only; no model read or tensor load",
    )

    runtime_identity_path = PLAN / "docs/campaign/work/lead/notebook-b4-cpu-runtime-identity.json"
    runtime_identity = load(runtime_identity_path)
    launch_path = EVIDENCE / "b4-editor-cycle-c/launch.json"
    launch = load(launch_path)
    dependency_match = runtime_identity["files"] == launch["runtime_dependencies"]
    binary_entry = next(item for item in runtime_identity["files"] if item["path"] == runtime_identity["binary"])
    runtime_ok = (
        runtime_identity["binary"] == launch["argv"][0]
        and binary_entry["sha256"] == launch["binary_sha256"] == RUNTIME_SHA
        and dependency_match
        and "build 10453" in runtime_identity["version"]
        and "3cb7ffb1a" in runtime_identity["version"]
    )
    check(
        "runtime identity and dependency manifest match server launch",
        "pass" if runtime_ok else "reject",
        binary_sha256=launch.get("binary_sha256"),
        dependency_count=len(launch.get("runtime_dependencies", [])),
        dependencies_match=dependency_match,
        version=runtime_identity.get("version"),
        current_path=identity(RUNTIME_PATH, hash_file=False),
    )

    # Hash the actual accepted source, package, and harness artifacts.  The
    # staged model/runtime/VSIX paths were cleaned after the run; the captured
    # preflight records the exact staged identities.
    file_records: dict[str, dict] = {}
    for path in [
        RUN_EVAL,
        ACCEPTED_VSIX,
        runtime_identity_path,
        cpu_receipt_path,
        v4_receipt_path,
        pre08_path,
        parity_path,
        package_receipt_path,
        launch_path,
        EVIDENCE / "b4-editor-cycle-c/server.log",
        EVIDENCE / "b4-editor-cycle-c/terminal.json",
        EVIDENCE / "b4-editor-cycle-c/lead-stop-request.json",
    ]:
        file_records[str(path)] = identity(path)

    expected_harness = v4["files"]
    harness_mismatches = []
    for name, expected in expected_harness.items():
        observed = identity(HARNESS / name)
        file_records[str(HARNESS / name)] = observed
        if not observed.get("exists") or observed.get("bytes") != (38855 if name == "primary.vsix" else None) and name == "primary.vsix":
            # The VSIX byte check below reports the useful mismatch detail.
            pass
        if not observed.get("exists") or observed.get("sha256") != expected:
            harness_mismatches.append({"file": name, "expected": expected, "observed": observed})
    check(
        "v4 harness file hashes match its admitted snapshot",
        "pass" if not harness_mismatches else "reject",
        file_count=len(expected_harness),
        mismatches=harness_mismatches,
    )

    source_mismatches = []
    for item in package_receipt.get("files", []):
        source_path = SOURCE_ROOT / item["path"]
        observed = identity(source_path)
        file_records[str(source_path)] = observed
        if not observed.get("exists") or observed.get("bytes") != item.get("bytes") or observed.get("sha256") != item.get("sha256"):
            source_mismatches.append({"expected": item, "observed": observed})
    check(
        "packaged source file identities match RUN-01 receipt",
        "pass" if not source_mismatches else "reject",
        file_count=len(package_receipt.get("files", [])),
        mismatches=source_mismatches,
    )

    run_eval_identity = file_records[str(RUN_EVAL)]
    check(
        "protected render_zeta2 source identity",
        "pass" if run_eval_identity.get("sha256") == RUN_EVAL_SHA else "reject",
        path=str(RUN_EVAL),
        observed=run_eval_identity,
        expected_sha256=RUN_EVAL_SHA,
        symbol="render_zeta2",
    )

    # Inspect the accepted VSIX itself, including the embedded product identity.
    vsix_identity = file_records[str(ACCEPTED_VSIX)]
    vsix_package = {}
    vsix_has_main = False
    try:
        with ZipFile(ACCEPTED_VSIX) as archive:
            vsix_has_main = "extension/dist/extension.js" in archive.namelist()
            vsix_package = json.loads(archive.read("extension/package.json"))
    except Exception as exc:  # pragma: no cover - diagnostic result is recorded
        vsix_package = {"error": str(exc)}
    vsix_ok = (
        vsix_identity.get("bytes") == 38855
        and vsix_identity.get("sha256") == VSIX_SHA
        and vsix_package.get("name") == "vscode-sepalith"
        and vsix_package.get("version") == "0.0.7"
        and vsix_package.get("main") == "./dist/extension.js"
        and vsix_has_main
    )
    check(
        "accepted VSIX identity and embedded extension package",
        "pass" if vsix_ok else "reject",
        observed=vsix_identity,
        expected={"bytes": 38855, "sha256": VSIX_SHA},
        package={key: vsix_package.get(key) for key in ("name", "publisher", "version", "main")},
        embedded_extension_js=vsix_has_main,
    )

    # Validate both captured runs, including exact status partition and cleanup.
    hosts: dict[str, dict] = {}
    run_records = {}
    supervision_records = {}
    evidence_names = [
        "host-result.json",
        "run-result.json",
        "inline-acceptance-evidence.json",
    ]
    for mode in ("fresh", "existing"):
        mode_dir = EVIDENCE / f"b4-editor-auto-v4-{mode}"
        host = load(mode_dir / "host-result.json")
        run = load(mode_dir / "run-result.json")
        inline = load(mode_dir / "inline-acceptance-evidence.json")
        hosts[mode] = host
        run_records[mode] = run
        for name in evidence_names:
            file_records[str(mode_dir / name)] = identity(mode_dir / name)
        pending = {item["name"] for item in host.get("checks", []) if item.get("status") == "pending"}
        failed = [item for item in host.get("checks", []) if item.get("status") == "fail"]
        all_statuses = {item.get("status") for item in host.get("checks", [])}
        expected_mode = f"{mode}_external"
        expected_identity = {
            "model_sha256_from_admission": MODEL_SHA,
            "runtime_sha256_from_admission": RUNTIME_SHA,
            "port": 18403,
            "backend": "cpu",
            "gpu_layers": 0,
            "threads": 6,
            "threads_batch": 6,
            "http_threads": 2,
            "context_size": 8192,
            "batch_size": 256,
            "ubatch_size": 256,
            "parallel": 1,
        }
        identity_observed = host.get("b4_identity", {})
        identity_ok = all(identity_observed.get(key) == value for key, value in expected_identity.items())
        run_ok = (
            run.get("status") == "b4_legacy_editor_route_exercised_acceptance_pending"
            and run.get("mode") == expected_mode
            and run.get("no_server_started_by_launcher") is True
            and run.get("install", {}).get("code") == 0
            and run.get("install", {}).get("timed_out") is False
            and run.get("preflight", {}).get("status") == "ready_for_root_editor_launch"
            and run.get("host_result", {}).get("status") == host.get("status")
        )
        check(
            f"{mode} run identity/status and finite check partition",
            "pass" if run_ok and identity_ok and not failed and all_statuses <= {"pass", "pending"} and pending == EXPECTED_PENDING else "reject",
            mode=mode,
            run_status=run.get("status"),
            host_status=host.get("status"),
            identity_ok=identity_ok,
            failed_checks=failed,
            status_counts={status: sum(item.get("status") == status for item in host.get("checks", [])) for status in sorted(all_statuses)},
            pending=sorted(pending),
        )
        # Keep the acceptance record in the report for the latency denominator.
        supervision_dir = EVIDENCE / f"b4-editor-auto-v4-{mode}-supervision"
        supervision_launch = load(supervision_dir / "launch.json")
        supervision_terminal = load(supervision_dir / "terminal.json")
        supervision_records[mode] = {"launch": supervision_launch, "terminal": supervision_terminal}
        file_records[str(supervision_dir / "launch.json")] = identity(supervision_dir / "launch.json")
        file_records[str(supervision_dir / "terminal.json")] = identity(supervision_dir / "terminal.json")
        supervision_ok = (
            supervision_terminal.get("status") == "completed_process_cleanup_verified"
            and supervision_terminal.get("child_exit_code") == 0
            and supervision_terminal.get("survivors") == []
            and supervision_terminal.get("reason") is None
            and supervision_launch.get("wall_seconds") == 180
            and supervision_launch.get("task") == "RUN-03"
        )
        check(
            f"{mode} supervisor process cleanup",
            "pass" if supervision_ok else "reject",
            launch={key: supervision_launch.get(key) for key in ("supervisor_pid", "child_pid", "wall_seconds", "task")},
            terminal={key: supervision_terminal.get(key) for key in ("status", "child_exit_code", "survivors", "reason")},
        )

    # Compare captured synthetic geometry against the protected Python renderer.
    extension_source = HARNESS / "extension.js"
    synthetic = {
        "replacement": {
            "variable": "a",
            "path": "run03-b4-replace.R",
            "line": 2,
        },
        "no-op-applicability": {
            "variable": "b",
            "path": "run03-b4-no-op.R",
            "line": 5,
        },
        "post-cancellation-fresh": {
            "variable": "c",
            "path": "run03-b4-cancel.R",
            "line": 2,
        },
    }
    source_geometry = {}
    for label, spec in synthetic.items():
        base_lines = source_strings(extension_source, spec["variable"], spec["path"])
        source_geometry[label] = {"path": spec["path"], "lines": base_lines}
    # The replacement prompt records the post-type document byte-for-byte.
    canonical = {mode: event(hosts[mode], "canonical_parity_input") for mode in hosts}
    replacement_docs = {mode: canonical[mode]["document"] for mode in hosts}
    geometry_checks = {}
    expected_prompt_text = {}
    for label, spec in synthetic.items():
        if label == "replacement":
            docs = replacement_docs
            captured_line = canonical["fresh"]["line"]
            captured_char = canonical["fresh"]["character"]
            if captured_line != spec["line"]:
                raise AssertionError(f"replacement line changed: {captured_line}")
            lines = docs["fresh"].split("\n")
            geometry_checks[label] = {
                "document_sha256": hashlib.sha256(docs["fresh"].encode()).hexdigest(),
                "captured_line": captured_line,
                "captured_character": captured_char,
                "source_base_lines_match_prefix": lines[:2] == source_geometry[label]["lines"][:2],
                "typed_suffix": lines[2].endswith("2"),
            }
        else:
            lines = list(source_geometry[label]["lines"])
            if label == "post-cancellation-fresh":
                lines[2] += event(hosts["fresh"], "cancellation_edit_applied")["deliberate_text"]
            docs = {mode: "\n".join(lines) + "\n" for mode in hosts}
            geometry_checks[label] = {
                "document_sha256": hashlib.sha256(docs["fresh"].encode()).hexdigest(),
                "captured_line": spec["line"],
                "captured_character": len(lines[spec["line"]]),
                "source_base_lines": source_geometry[label]["lines"],
            }
        for mode in hosts:
            if label == "replacement":
                doc = replacement_docs[mode]
            else:
                doc = docs[mode]
            lines = doc.split("\n")
            line = spec["line"]
            character = len(lines[line])
            ex = {
                "suffix": [""] + lines[line + 1 :],
                "prefix": lines[:line],
                "region_old": [lines[line]],
                "cursor_idx": 0,
                "path": spec["path"],
                "event_diff": "",
            }
            if label == "replacement":
                character = canonical[mode]["character"]
            if character != len(lines[line]):
                geometry_checks[label][f"{mode}_character_matches_line"] = False
            rendered = None
            module_spec = importlib.util.spec_from_file_location("protected_run_eval", RUN_EVAL)
            if module_spec is None or module_spec.loader is None:
                raise AssertionError("cannot load protected run_eval.py")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            rendered = module.render_zeta2(ex)
            expected_prompt_text.setdefault(label, {})[mode] = rendered
            observed = prompt_event(hosts[mode], label)
            observed_text = observed.get("prompt_text")
            observed_sha = hashlib.sha256(observed_text.encode()).hexdigest() if isinstance(observed_text, str) else None
            prompt_ok = (
                observed_text == rendered
                and observed_sha == EXPECTED_PROMPTS[label]
                and observed.get("prompt_sha256") == EXPECTED_PROMPTS[label]
                and observed.get("expected_path") == spec["path"]
                and not observed.get("has_prm03_task_marker")
            )
            geometry_checks[label][f"{mode}_prompt"] = {
                "status": "pass" if prompt_ok else "reject",
                "bytes": len(rendered),
                "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "observed_sha256": observed.get("prompt_sha256"),
            }
    # All three prompts must also be identical between fresh and existing.
    prompt_pair_equal = all(expected_prompt_text[label]["fresh"] == expected_prompt_text[label]["existing"] for label in synthetic)
    check(
        "captured prompts equal protected render_zeta2 for all synthetic geometries",
        "pass" if prompt_pair_equal and all(item[f"{mode}_prompt"]["status"] == "pass" for item in geometry_checks.values() for mode in hosts) else "reject",
        cases=geometry_checks,
        fresh_existing_byte_identical=prompt_pair_equal,
        renderer_source_sha256=RUN_EVAL_SHA,
        geometry_source_sha256=file_records[str(extension_source)]["sha256"] if str(extension_source) in file_records else sha256(extension_source),
    )

    # Inline commit evidence: hashes, versions, command, and an actual R parse.
    inline_results = {}
    for mode in hosts:
        inline_path = EVIDENCE / f"b4-editor-auto-v4-{mode}/inline-acceptance-evidence.json"
        inline = load(inline_path)
        before_hash = hashlib.sha256(inline["before_text"].encode()).hexdigest()
        after_hash = hashlib.sha256(inline["after_text"].encode()).hexdigest()
        parsed = r_parse(inline["after_text"])
        expected_inline = (
            inline.get("command") == "editor.action.inlineSuggest.commit"
            and inline.get("before_version") == 1
            and inline.get("after_version") == 2
            and inline.get("document_changed_after_commit") is True
            and inline.get("before_sha256") == before_hash
            and inline.get("after_sha256") == after_hash
            and inline.get("r_parse_ok") is True
            and parsed["ok"]
            and inline.get("elapsed_request_to_commit_observation_ms") in (5326, 5327)
            and inline.get("measurement_limit", "").startswith("The 5200ms quiet window is imposed")
        )
        inline_results[mode] = {
            "status": "pass" if expected_inline else "reject",
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "r_parse": parsed,
            "captured_elapsed_request_to_commit_observation_ms": inline.get("elapsed_request_to_commit_observation_ms"),
            "measurement_limit": inline.get("measurement_limit"),
        }
    check(
        "real inline commit hash/version/R parse evidence",
        "pass" if all(item["status"] == "pass" for item in inline_results.values()) else "reject",
        runs=inline_results,
    )

    # Cancellation and no-op evidence is deliberately split from the unresolved
    # ghost/provider checks.  The host only proves document authority and prompt
    # refresh, not private transport state or visual ghost publication.
    cancellation_results = {}
    for mode, host in hosts.items():
        issued = event(host, "cancellation_trigger_issued")
        applied = event(host, "cancellation_edit_applied")
        fresh = prompt_event(host, "post-cancellation-fresh")
        cancellation_results[mode] = {
            "issued_before_version": issued.get("before_version"),
            "cancel_after_ms": issued.get("cancel_after_ms"),
            "applied_version": applied.get("version"),
            "applied_deliberate_text": applied.get("deliberate_text"),
            "applied_sha256": applied.get("content_sha256"),
            "fresh_prompt_sha256": fresh.get("prompt_sha256"),
            "fresh_prompt_contains_user_cancel": "user-cancel" in fresh.get("prompt_text", ""),
            "document_authoritative_check": next(
                item for item in host["checks"] if item["name"] == "cancellation leaves the user document authoritative"
            ).get("status"),
        }
    check(
        "cancellation refresh and document-authority observations",
        "pass" if all(
            item["fresh_prompt_contains_user_cancel"]
            and item["document_authoritative_check"] == "pass"
            and item["applied_deliberate_text"] == " # user-cancel"
            for item in cancellation_results.values()
        ) else "reject",
        runs=cancellation_results,
        interpretation="host-side evidence only; stale ghost nonpublication remains pending",
    )
    no_op_results = {}
    for mode, host in hosts.items():
        item = next(item for item in host["checks"] if item["name"] == "no-op probe does not mutate the document automatically")
        no_op_results[mode] = {
            "status": item.get("status"),
            "before_sha256": item.get("before_sha256"),
            "after_sha256": item.get("after_sha256"),
            "same": item.get("before_sha256") == item.get("after_sha256"),
        }
    check(
        "no-op automatic document mutation observation",
        "pass" if all(item["status"] == "pass" and item["same"] for item in no_op_results.values()) else "reject",
        runs=no_op_results,
        interpretation="does not establish suggestion applicability or absence of a private ghost",
    )

    # Native server evidence is parsed from the captured log.  Tasks 0 and 52
    # are one-token health/warm-up probes; the other timed tasks are generation
    # requests.  No request body or completion stop reason is invented here.
    log_path = EVIDENCE / "b4-editor-cycle-c/server.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    prompt_timing: dict[int, dict] = {}
    eval_timing: dict[int, dict] = {}
    total_timing: dict[int, dict] = {}
    for line in log_text.splitlines():
        prompt_match = re.search(r"print_timing:.*?task\s+(\d+)\s+\| prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens", line)
        eval_match = re.search(r"print_timing:.*?task\s+(\d+)\s+\|\s+eval time =\s*([0-9.]+) ms /\s*(\d+) tokens", line)
        total_match = re.search(r"print_timing:.*?task\s+(\d+)\s+\|\s+total time =\s*([0-9.]+) ms /\s*(\d+) tokens", line)
        if prompt_match:
            prompt_timing[int(prompt_match.group(1))] = {"ms": float(prompt_match.group(2)), "tokens": int(prompt_match.group(3))}
        if eval_match:
            eval_timing[int(eval_match.group(1))] = {"ms": float(eval_match.group(2)), "tokens": int(eval_match.group(3))}
        if total_match:
            total_timing[int(total_match.group(1))] = {"ms": float(total_match.group(2)), "tokens": int(total_match.group(3))}
    release_entries = [
        {"task": int(match.group(1)), "n_tokens": int(match.group(2)), "truncated": int(match.group(3))}
        for match in re.finditer(r"release: id\s+\d+\s+\| task\s+(\d+).*?n_tokens = (\d+), truncated = (\d+)", log_text)
    ]
    cancel_tasks = [int(item) for item in re.findall(r"cancel task, id_task = (\d+)", log_text)]
    timed_tasks = sorted(set(prompt_timing) & set(eval_timing) & set(total_timing))
    generation_tasks = [task for task in timed_tasks if total_timing[task]["tokens"] > 2]
    health_tasks = [task for task in timed_tasks if total_timing[task]["tokens"] <= 2]
    generation_totals = [total_timing[task]["ms"] for task in generation_tasks]
    log_cpu_ok = (
        "build 10453 (3cb7ffb1a)" in log_text
        and "CPU     : AMD Ryzen" in log_text
        and "using 2 threads for HTTP server" in log_text
        and "loaded meta data with 32 key-value pairs and 320 tensors" in log_text
        and "general.architecture str              = qwen35" in log_text
        and "warning: no usable GPU found" in log_text
    )
    release_ok = len(release_entries) == 14 and all(item["truncated"] == 0 for item in release_entries)
    native_ok = len(generation_tasks) == 7 and len(health_tasks) == 2 and len(cancel_tasks) == 5 and release_ok
    check(
        "native CPU server timing and stop/cancellation log evidence",
        "pass" if log_cpu_ok and native_ok else "reject",
        build="10453/3cb7ffb1a",
        model_architecture="qwen35",
        tensors=320,
        gpu_warning_present="warning: no usable GPU found" in log_text,
        timed_task_count=len(timed_tasks),
        health_task_count=len(health_tasks),
        generation_task_count=len(generation_tasks),
        generation_tasks=generation_tasks,
        generation_total_ms=generation_totals,
        generation_total_ms_min=min(generation_totals) if generation_totals else None,
        generation_total_ms_median=median(generation_totals) if generation_totals else None,
        generation_total_ms_max=max(generation_totals) if generation_totals else None,
        explicit_cancel_task_ids=cancel_tasks,
        release_count=len(release_entries),
        release_truncated_values=sorted({item["truncated"] for item in release_entries}),
        interpretation="native backend compute timing and task cancellation only; no client-visible latency or stop reason claim",
    )
    check(
        "first-visible latency is not promoted from harness wait",
        "pending",
        captured_request_to_commit_ms={mode: inline_results[mode]["captured_elapsed_request_to_commit_observation_ms"] for mode in hosts},
        reason="the harness imposes a 5200 ms quiet wait; evidence does not timestamp first visible ghost/suggestion",
    )
    stop_terminal = load(EVIDENCE / "b4-editor-cycle-c/terminal.json")
    stop_request = load(EVIDENCE / "b4-editor-cycle-c/lead-stop-request.json")
    check(
        "root-owned server release terminal",
        "pass" if stop_terminal.get("exit_code") == 0 and stop_terminal.get("server_pid_exists") is False and stop_request.get("reason") == "Both editor phases terminal with no survivors; release bounded server after evidence capture" else "reject",
        stop_request=stop_request,
        terminal=stop_terminal,
        bind_free_observation="reported by root message; not serialized in terminal.json",
    )

    # The report is intentionally useful without approving the whole RUN-03.
    recommendations = [
        {
            "component": "source / protected renderer identity",
            "recommendation": "accept",
            "basis": "RUN-01 source hashes and current run_eval.render_zeta2 SHA match; prompt bytes match all three captured geometries",
        },
        {
            "component": "VSIX identity and route exercise",
            "recommendation": "accept",
            "basis": "accepted 0.0.7 VSIX bytes/SHA and embedded package identity match; fresh and existing host runs complete with no failed checks",
        },
        {
            "component": "model/runtime identity",
            "recommendation": "accept with captured-identity qualification",
            "basis": "model hash/bytes and runtime/dependency hashes match admissions and launch; current staged model/runtime were cleaned and not reopened",
        },
        {
            "component": "real inline commit and R parse",
            "recommendation": "accept",
            "basis": "both runs show editor.action.inlineSuggest.commit, version 1->2, before/after hashes, and independent Rscript parse success",
        },
        {
            "component": "fresh/existing supervision cleanup",
            "recommendation": "accept",
            "basis": "both supervisor terminals exit 0 with completed_process_cleanup_verified and no survivors; server release terminal exits 0 with PID absent",
        },
        {
            "component": "native CPU compute timing and task cancellation",
            "recommendation": "accept as bounded diagnostics",
            "basis": "server log has 7 generation timing records, 5 explicit cancellation task IDs, and 14 non-truncated releases under CPU/no-GPU launch",
        },
        {
            "component": "no-op suggestion applicability",
            "recommendation": "pending",
            "basis": "host only proves no automatic document mutation; visible ghost applicability remains unobserved",
        },
        {
            "component": "stale cancellation ghost nonpublication",
            "recommendation": "pending",
            "basis": "user edit and refreshed prompt are captured, but private provider lease/ghost state is not observable and no raw request correlation is retained",
        },
        {
            "component": "legacy response parser and stop metadata",
            "recommendation": "pending",
            "basis": "host does not capture raw completion body or stop reason; server release/truncated fields cannot establish parser output association",
        },
        {
            "component": "first-visible response latency",
            "recommendation": "reject as a claim for v4 evidence",
            "basis": "5200 ms quiet wait is imposed by the harness and is explicitly not time-to-first-visible suggestion",
        },
    ]
    missing_evidence = [
        "root-owned raw completion response bodies with stop metadata linked to each labeled editor request",
        "visible no-op ghost/applicability observation with acceptance left unperformed",
        "request cancellation correlation proving stale transport cannot publish or overwrite the deliberate edit/fresh result",
        "native client-side first-visible timestamp, if latency is required (the v4 harness wait cannot supply it)",
    ]

    report = {
        "schema": "RUN-03-b4-editor-v4-independent-review/v1",
        "task": "RUN-03",
        "at": "2026-09-13",
        "scope": {
            "plan": str(PLAN),
            "inputs": [str(EVIDENCE), str(HARNESS), str(parity_path), str(package_receipt_path), str(pre08_path)],
            "writes": [str(WORK)],
            "cpu_only": True,
            "model_opened": False,
            "server_started": False,
            "network_used": False,
        },
        "checks": checks,
        "recommendations": recommendations,
        "missing_evidence": missing_evidence,
        "captured_file_identities": file_records,
        "summary_counts": {
            "fresh_host_checks": len(hosts["fresh"].get("checks", [])),
            "existing_host_checks": len(hosts["existing"].get("checks", [])),
            "fresh_pending": sorted({item["name"] for item in hosts["fresh"].get("checks", []) if item.get("status") == "pending"}),
            "existing_pending": sorted({item["name"] for item in hosts["existing"].get("checks", []) if item.get("status") == "pending"}),
            "prompt_cases": 3,
            "prompt_runs": 2,
            "inline_runs": 2,
            "native_generation_timing_records": len(generation_tasks),
            "native_explicit_cancel_tasks": len(cancel_tasks),
            "native_release_records": len(release_entries),
        },
        "native_server": {
            "generation_timing_by_task": {str(task): total_timing[task] for task in generation_tasks},
            "health_timing_by_task": {str(task): total_timing[task] for task in health_tasks},
            "cancel_task_ids": cancel_tasks,
            "release_entries": release_entries,
        },
    }
    report_path = WORK / "report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "report": str(report_path),
        "checks": {status: sum(item["status"] == status for item in checks) for status in sorted({item["status"] for item in checks})},
        "prompt_cases": 3,
        "prompt_runs": 2,
        "native_generation_timing_records": len(generation_tasks),
        "native_explicit_cancel_tasks": len(cancel_tasks),
        "report_sha256": sha256(report_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
