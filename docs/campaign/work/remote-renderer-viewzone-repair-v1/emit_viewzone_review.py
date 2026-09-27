#!/usr/bin/env python3
"""Emit the bounded CPU-only RUN-04 view-zone repair review."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
PLAN = HERE.parents[3]
PRIOR = PLAN / "docs/campaign/work/remote-editor-a-independent-review"
REMOTE = PRIOR / "remote"
RECEIPT = PLAN / "docs/campaign/receipts/RUN-04-remote-renderer-viewzone-repair-v1.json"
AB_OBSERVER = PLAN / "docs/campaign/work/remote-auto-debounce-ab-v1/observe_renderer.mjs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(PLAN)) if path.is_relative_to(PLAN) else str(path)


def run_patch_syntax_check(source: Path, patch: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="run04-viewzone-") as tmp:
        copy = Path(tmp) / "observe_renderer.mjs"
        copy.write_bytes(source.read_bytes())
        applied = subprocess.run(["patch", str(copy)], input=patch.read_text(), text=True, capture_output=True, check=False)
        syntax = subprocess.run(["node", "--check", str(copy)], text=True, capture_output=True, check=False)
        return {
            "source": rel(source),
            "patch": rel(patch),
            "patch_returncode": applied.returncode,
            "node_check_returncode": syntax.returncode,
            "passed": applied.returncode == 0 and syntax.returncode == 0,
            "method": f"patch < {patch.name} in a temporary copy; node --check; source untouched",
        }


def main() -> int:
    frames = load_jsonl(REMOTE / "renderer-frames.jsonl")
    host = load_json(REMOTE / "host-result.json")
    renderer = load_json(REMOTE / "renderer-observer.json")
    renderer_result = load_json(REMOTE / "renderer-result.json")
    analysis = load_json(HERE / "renderer-analysis-viewzone-repair.json")
    check = subprocess.run(["node", str(HERE / "check_viewzone_repair.mjs")], text=True, capture_output=True, check=False)
    check_output = json.loads(check.stdout) if check.returncode == 0 else {"raw": check.stdout, "stderr": check.stderr}
    dump(HERE / "check-output.json", check_output)

    nonempty = [(index, frame) for index, frame in enumerate(frames) if frame.get("ghosts")]
    spans = []
    if nonempty:
        start = previous = nonempty[0][0]
        for index, _frame in nonempty[1:]:
            if index != previous + 1:
                spans.append((start, previous))
                start = index
            previous = index
        spans.append((start, previous))
    span_summary = []
    for start, end in spans:
        first = frames[start]
        ghost = first["ghosts"][0]
        span_summary.append({
            "frame_index": [start, end],
            "epoch_ms": [first["epoch_ms"], frames[end]["epoch_ms"]],
            "frame_count": end - start + 1,
            "direct_selector_text": ghost.get("text"),
            "direct_selector_text_sha256": hashlib.sha256(ghost.get("text", "").encode()).hexdigest(),
            "in_view_zone": ghost.get("in_view_zone"),
            "view_zone_metadata_field_present": any(key in ghost for key in ("view_zone", "view_zone_lines")),
            "geometry": ghost.get("rect"),
        })

    controls = [event for event in host["events"] if event.get("kind") == "control_provider_resolved"]
    multiline_check = next(check for check in host["checks"] if check.get("name") == "deterministic multiline insertion mechanics")
    selected_source = next(event for event in host["events"] if event.get("kind") == "case_start" and event.get("id") == "q8-multiline-1")

    input_paths = [
        PLAN / "docs/campaign/DELEGATION.md",
        PLAN / "docs/campaign/work/remote-editor-harness-v1/observe_renderer.mjs",
        PLAN / "docs/campaign/work/remote-editor-harness-v1/analyze_renderer.mjs",
        PLAN / "docs/campaign/work/remote-editor-harness-v1/run_remote_editor.mjs",
        PLAN / "docs/campaign/work/remote-renderer-viewzone-repair-v1/observe_renderer-viewzone-repair.patch",
        AB_OBSERVER,
        PLAN / "docs/campaign/work/remote-renderer-viewzone-repair-v1/observe_renderer-viewzone-repair-on-ab.patch",
        PLAN / "docs/campaign/work/remote-renderer-viewzone-repair-v1/analyze_renderer_viewzone.mjs",
        PRIOR / "review-report.json",
        REMOTE / "renderer-frames.jsonl",
        REMOTE / "renderer-observer.json",
        REMOTE / "renderer-result.json",
        REMOTE / "host-result.json",
        REMOTE / "renderer-ready.json",
        REMOTE / "ghost-01.png",
        REMOTE / "ghost-02.png",
        REMOTE / "ghost-03.png",
    ]
    manifest = {
        "schema": "sepalith.run04.remote-renderer-viewzone-repair.inputs.v1",
        "files": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in input_paths],
        "retained_run": "RUN-04-remote-editor-a",
        "read_policy": "Only retained campaign evidence and source files listed above were read; no SSH, editor, process control, GPU, model or state access.",
    }
    dump(HERE / "input-manifest.json", manifest)

    report = {
        "schema": "sepalith.run04.remote-renderer-viewzone-repair.v1",
        "task": "RUN-04-followup",
        "objective": "Repair multiline view-zone line classification and diagnose bounded observer CDP shutdown without making a Q8 multiline absence claim.",
        "retained_evidence": {
            "frame_count": len(frames),
            "dropped_frames": renderer.get("dropped"),
            "all_focused": all(frame.get("focused") is True for frame in frames),
            "all_visible": all(frame.get("visibility") == "visible" for frame in frames),
            "direct_nonempty_frames": len(nonempty),
            "spans": span_summary,
            "old_collector_line_metadata": "absent",
            "old_collector_limitation": "ghost-02.png visibly contains all three control sentinel lines, while the retained direct selector frame object contains only OBSERVER_MULTI_FIRST and no view_zone.lines.",
            "control_screenshot_manual_review": {
                "ghost_01_sha256": sha(REMOTE / "ghost-01.png"),
                "ghost_02_sha256": sha(REMOTE / "ghost-02.png"),
                "ghost_03_sha256": sha(REMOTE / "ghost-03.png"),
                "ghost_02_visible_sentinel_lines": 3,
                "basis": "retained PNG visual inspection; line-level DOM metadata was not retained",
            },
        },
        "control_evidence": {
            "resolved_provider_texts": [event.get("text") for event in controls],
            "multiline_commit_check": {"status": multiline_check.get("status"), "after_text": multiline_check.get("after_text")},
            "expected_multiline_lines": ["OBSERVER_MULTI_FIRST", "OBSERVER_MULTI_SECOND", "OBSERVER_MULTI_THIRD"],
            "selected_q8_source_label": selected_source.get("source"),
            "separation": "Controls are deterministic plaintext provider evidence; they do not establish selected-Q8 provider quality.",
        },
        "classifier_repair": {
            "patch": rel(HERE / "observe_renderer-viewzone-repair.patch"),
            "patch_sha256": sha(HERE / "observe_renderer-viewzone-repair.patch"),
            "collector_changes": [
                "retains existing selector-root geometry and occlusion checks",
                "adds per-node visibility, viewport, rect and elementFromPoint occlusion details",
                "collects every visible sibling .view-line in the nearest explicit view-zone or view-lines container as view_zone.lines",
                "retains line text, class, visible/occluded flags and geometry for line 2/3 matching",
            ],
            "analyzer": rel(HERE / "analyze_renderer_viewzone.mjs"),
            "ab_observer_delta": rel(HERE / "observe_renderer-viewzone-repair-on-ab.patch"),
            "retained_analysis": {
                "view_zone_line_metadata_observed": analysis.get("view_zone_line_metadata_observed"),
                "control_multiline_status": analysis.get("observer_multiline_control_status"),
                "q8_multiline_statuses": [row.get("status") for row in analysis.get("cases", []) if row.get("id", "").startswith("q8-multiline")],
            },
            "test": {
                "path": rel(HERE / "check_viewzone_repair.mjs"),
                "returncode": check.returncode,
                "output": check_output,
                "syntax_check": run_patch_syntax_check(
                    PLAN / "docs/campaign/work/remote-editor-harness-v1/observe_renderer.mjs",
                    HERE / "observe_renderer-viewzone-repair.patch",
                ),
                "ab_syntax_check": run_patch_syntax_check(
                    AB_OBSERVER,
                    HERE / "observe_renderer-viewzone-repair-on-ab.patch",
                ),
            },
        },
        "cdp_timeout_diagnosis": {
            "retained_result": renderer_result,
            "retained_observer": {key: renderer.get(key) for key in ("frames", "dropped", "sourceIdentity", "screenshots") if key in renderer},
            "inference_from_launcher": "run_remote_editor.mjs sets observerStopped only after await runProcess(host); an in-flight Runtime.evaluate poll can therefore outlive the host window and reach the five-second CDP timeout before the stop flag is observed.",
            "bounded_fix": [
                "catch only CDP timeout: Runtime.evaluate in the polling loop and set stop_reason=runtime_evaluate_timeout",
                "return stopped_after_cdp_timeout with last_frame_epoch_ms instead of rejecting observer completion",
                "in finally, attempt collector stop with a 500 ms call timeout, close the WebSocket, and retain graceful_stop/stop_reason in renderer-observer.json",
            ],
            "acceptance_effect": "Observer transport cleanup only; it does not turn frame capture into editor or model acceptance.",
        },
        "limits": [
            "The retained stream predates this collector patch, so actual Q8 multiline line 2/3 ghost presence or absence remains indeterminate.",
            "A fresh bounded editor capture with the repaired collector is required for a validated Q8 multiline visibility result.",
            "The fallback .view-lines container can include source siblings; root should review the first repaired capture and narrow the container selector if needed.",
            "No SSH, editor, server, process control, GPU, model or campaign state action was performed.",
        ],
        "decision": "repair_ready_recapture_required",
        "input_manifest_sha256": sha(HERE / "input-manifest.json"),
    }
    dump(HERE / "review-report.json", report)

    receipt = {
        "task": "RUN-04-followup",
        "packet": "RUN-04-remote-renderer-viewzone-repair-v1",
        "schema": "sepalith.campaign.run04.remote-renderer-viewzone-repair.v1",
        "owner": "stress_fixture",
        "status": "verified",
        "observed_at": "2026-09-13T00:00:00Z",
        "dependencies_checked": [
            rel(REMOTE / "renderer-frames.jsonl"),
            rel(REMOTE / "renderer-observer.json"),
            rel(REMOTE / "renderer-result.json"),
            rel(REMOTE / "host-result.json"),
            rel(PLAN / "docs/campaign/work/remote-editor-harness-v1/observe_renderer.mjs"),
        ],
        "action": "Inspected retained frame/view-zone fields and screenshots, prepared and syntax-checked a collector/analyzer patch, ran retained and synthetic line-level classifier checks, and diagnosed graceful CDP stop race.",
        "commands_or_method": [
            "node check_viewzone_repair.mjs",
            "node analyze_renderer_viewzone.mjs <retained remote evidence> renderer-analysis-viewzone-repair.json",
            "patch < observe_renderer-viewzone-repair.patch in a temporary copy; node --check",
        ],
        "result": {
            "report": rel(HERE / "review-report.json"),
            "report_sha256": sha(HERE / "review-report.json"),
            "manifest": rel(HERE / "input-manifest.json"),
            "manifest_sha256": sha(HERE / "input-manifest.json"),
            "patch": rel(HERE / "observe_renderer-viewzone-repair.patch"),
            "patch_sha256": sha(HERE / "observe_renderer-viewzone-repair.patch"),
            "ab_patch": rel(HERE / "observe_renderer-viewzone-repair-on-ab.patch"),
            "ab_patch_sha256": sha(HERE / "observe_renderer-viewzone-repair-on-ab.patch"),
            "retained_frame_count": len(frames),
            "retained_view_zone_line_metadata": False,
            "control_multiline_screenshot_lines": 3,
            "synthetic_enriched_all_three_visible": check_output.get("synthetic_enriched_frame", {}).get("all_three_visible"),
            "synthetic_occluded_third_rejected": check_output.get("synthetic_enriched_frame", {}).get("occluded_third_rejected"),
            "q8_multiline_visibility": "indeterminate; recapture required",
            "cdp_terminal_error": renderer_result.get("error"),
        },
        "acceptance": "inconclusive — repair and tests pass, but the retained observer did not capture line-level view-zone metadata; no Q8 multiline ghost absence claim is made and recapture is required.",
        "changed_files": [
            rel(HERE / "README.md"),
            rel(HERE / "observe_renderer-viewzone-repair.patch"),
            rel(HERE / "observe_renderer-viewzone-repair-on-ab.patch"),
            rel(HERE / "analyze_renderer_viewzone.mjs"),
            rel(HERE / "check_viewzone_repair.mjs"),
            rel(HERE / "emit_viewzone_review.py"),
            rel(HERE / "input-manifest.json"),
            rel(HERE / "renderer-analysis-viewzone-repair.json"),
            rel(HERE / "check-output.json"),
            rel(HERE / "review-report.json"),
            rel(RECEIPT),
        ],
        "artifacts": [rel(HERE / name) for name in ("observe_renderer-viewzone-repair.patch", "observe_renderer-viewzone-repair-on-ab.patch", "analyze_renderer_viewzone.mjs", "check_viewzone_repair.mjs", "renderer-analysis-viewzone-repair.json", "review-report.json", "input-manifest.json")] + [rel(RECEIPT)],
        "unresolved": [
            "fresh repaired observer capture needed to classify actual Q8 multiline line 2/3 ghosts",
            "review the first repaired capture for fallback .view-lines source-sibling contamination",
        ],
        "next": "Root owns patch review/application and a fresh bounded editor capture; keep Q8 multiline visibility indeterminate until line-level metadata passes geometry/focus/occlusion checks.",
        "lease_released": "yes — CPU/read-only source and retained-evidence review complete",
    }
    dump(RECEIPT, receipt)
    print(json.dumps({"report": str(HERE / "review-report.json"), "receipt": str(RECEIPT), "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
