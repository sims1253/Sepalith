#!/usr/bin/env python3
"""Materialize full DEV buffers for parse-only notebook replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from typing import Any


PANEL_SHA256 = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
PROTOCOL_PATH = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-task-v2/snapshots/8ec908a41904888af647de2ee8b40ba8f73837777509d91159051b25fc72dd3e/source/packages/sepalith/src/sepalith/campaign_protocol.py")
INPUTS = {
    "incumbent": Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-task-global500-native-a/dev-results/results.json"),
    "C250": Path("/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-c250-native-dev-v1/dev-results/results.json"),
    "D500": Path("/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-d500-native-dev-v1/dev-results/results.json"),
}
INPUT_SHA256 = {
    "incumbent": "a68190b6d87e175f01073021b16c1c8e16d57226b2d95977f2ad908e5a2fd2e3",
    "C250": "1f4e10ee118a98405cb9364157cbbdabf3033b6f9e7135cddf8c24177e6cb9e5",
    "D500": "ba909888aee088f3be4039eb186d2aad49ffa30c8199832d7ddbc70d4ca24ae9",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def utf16_index(line: str, units: int) -> int:
    require(units >= 0, "negative UTF-16 position")
    used = 0
    for index, char in enumerate(line):
        if used == units:
            return index
        used += len(char.encode("utf-16-le")) // 2
        require(used <= units, "UTF-16 position splits a code point")
    require(used == units, "UTF-16 position exceeds line")
    return len(line)


def position_offset(text: str, line_number: int, utf16_units: int) -> int:
    lines = text.splitlines(keepends=True)
    # Python has no final empty keepends line; LSP may point immediately after
    # the last newline.
    if line_number == len(lines) and text.endswith(("\n", "\r")):
        require(utf16_units == 0, "nonzero position after final EOL")
        return len(text)
    require(0 <= line_number < len(lines), "line position exceeds document")
    raw_line = lines[line_number]
    body = raw_line.rstrip("\r\n")
    return sum(len(item) for item in lines[:line_number]) + utf16_index(body, utf16_units)


def splice(text: str, range_value: dict[str, Any], replacement: str, eol: str,
           expected_old: str | None = None) -> str:
    start = range_value["start"]
    end = range_value["end"]
    left = position_offset(text, start["line"], start["character"])
    right = position_offset(text, end["line"], end["character"])
    require(left <= right, "replacement range is reversed")
    if expected_old is not None:
        require(text[left:right] == expected_old.replace("\n", eol),
                "history old_text differs at exact range")
    mapped = replacement.replace("\n", eol)
    return text[:left] + mapped + text[right:]


def load_protocol():
    name = "notebook_syntax_campaign_protocol"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot_candidate(row: dict[str, Any]) -> tuple[str, bytes] | None:
    provenance = row["source_provenance"]
    path_value = provenance.get("source_snapshot_path")
    if path_value:
        if "::" in path_value:
            archive, member = path_value.split("::", 1)
            with tarfile.open(archive) as bundle:
                handle = bundle.extractfile(member)
                require(handle is not None, "tar source snapshot member missing")
                return path_value, handle.read()
        path = Path(path_value)
        return (path_value, path.read_bytes()) if path.is_file() else None
    parent = provenance.get("parent_identity")
    if not isinstance(parent, dict):
        return None
    root = Path(parent["normalized_parent_path"])
    relative = Path(row["context"]["path"])
    candidates = [path for path in root.rglob(relative.name)
                  if path.is_file() and str(path).endswith(str(relative))]
    wanted = provenance.get("source_snapshot_sha256")
    exact = [path for path in candidates if sha_file(path) == wanted]
    require(len(exact) <= 1, "ambiguous exact normalized source snapshot")
    return (str(exact[0]), exact[0].read_bytes()) if exact else None


def reconstruct_baseline(row: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    context = row["context"]
    wanted = context["replacement_range"]["content_sha256"]
    eol = "\r\n" if context["document_eol"] == "crlf" else "\n"
    selection = row["selection"]
    if not selection.get("omissions"):
        raw = eol.join(selection["prefix"] + selection["region"] + selection["suffix"]).encode()
        require(sha_bytes(raw) == wanted, f"{row['id']}: complete selected document hash differs")
        return raw, {"method": "complete_selection", "history_events_replayed": 0}

    candidate = snapshot_candidate(row)
    require(candidate is not None, f"{row['id']}: omitted document has no source snapshot")
    source_name, raw = candidate
    provenance = row["source_provenance"]
    source_sha = provenance.get("source_snapshot_sha256") or provenance.get("before_snapshot_sha256")
    require(sha_bytes(raw) == source_sha, f"{row['id']}: source snapshot hash differs")
    text = raw.decode("utf-8", "strict")
    for event in context["history"]:
        range_value = event["range_utf16"]
        require(sha_bytes(text.encode()) == range_value["content_sha256"],
                f"{row['id']}: history predecessor hash differs")
        try:
            text = splice(text, range_value, event["new_text"], eol, event["old_text"])
        except ValueError as error:
            raise ValueError(f"{row['id']}: history splice rejected: {error}") from error
    raw = text.encode()
    require(sha_bytes(raw) == wanted, f"{row['id']}: reconstructed current document hash differs")
    return raw, {"method": "source_snapshot_plus_history", "source": source_name,
                 "source_sha256": source_sha, "history_events_replayed": len(context["history"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output directory must be fresh")
    require(sha_file(args.panel) == PANEL_SHA256, "corrected DEV75 panel hash differs")
    for arm, path in INPUTS.items():
        require(sha_file(path) == INPUT_SHA256[arm], f"{arm}: result input hash differs")

    panel = [json.loads(line) for line in args.panel.open()]
    require(len(panel) == 75 and len({row["id"] for row in panel}) == 75, "panel denominator or IDs differ")
    protocol = load_protocol()
    parsed_inputs = {arm: json.loads(path.read_text()) for arm, path in INPUTS.items()}
    args.output.mkdir(parents=True)
    buffers = args.output / "buffers"
    buffers.mkdir()
    rows_out: list[dict[str, Any]] = []
    reconstruction_counts: dict[str, int] = {}

    for index, panel_row in enumerate(panel):
        baseline, reconstruction = reconstruct_baseline(panel_row)
        reconstruction_counts[reconstruction["method"]] = reconstruction_counts.get(reconstruction["method"], 0) + 1
        case_dir = buffers / f"{index:03d}"
        case_dir.mkdir()
        baseline_path = case_dir / "baseline.R"
        baseline_path.write_bytes(baseline)
        context = protocol.PromptContext.from_dict(panel_row["context"])
        context_sha = panel_row["context"]["replacement_range"]["content_sha256"]
        rendered_prompt = protocol.render_prompt(context)
        require(sha_bytes(rendered_prompt.encode()) == panel_row["prompt_sha256"],
                f"{panel_row['id']}: rendered prompt hash differs")

        for arm, document in parsed_inputs.items():
            require(document["completed_case_ids"] == document["expected_case_ids"], f"{arm}: incomplete case list")
            require(document["expected_case_ids"] == [row["id"] for row in panel], f"{arm}: case order differs")
            native = document["results"][index]
            checks = {
                "id": native["id"] == panel_row["id"],
                "family": native["family"] == panel_row["family"],
                "expected_noop": native["expected_noop"] == (panel_row["operation"] == "no_op"),
                "operation_label": native["operation_label"] == panel_row["operation"],
                "prompt_sha256": native["prompt_sha256"] == panel_row["prompt_sha256"],
                "response_complete": native["response_complete"] is True,
            }
            require(all(checks.values()), f"{arm}/{panel_row['id']}: identity check differs")
            parsed = protocol.parse_output(native["raw_text"], context)
            require(parsed.status == native["protocol"]["parser_status"], f"{arm}/{panel_row['id']}: parser status differs")
            require(parsed.operation == native["protocol"]["parser_operation"], f"{arm}/{panel_row['id']}: parser operation differs")
            require((parsed.status == "accepted") == native["protocol"]["valid"], f"{arm}/{panel_row['id']}: protocol validity differs")
            output_path = None
            output_sha = None
            if parsed.status == "accepted":
                if parsed.operation == "no_op":
                    applied = baseline
                else:
                    eol = "\r\n" if context.document_eol == "crlf" else "\n"
                    applied_text = splice(baseline.decode("utf-8", "strict"),
                                          panel_row["context"]["replacement_range"], parsed.body_text, eol)
                    applied = applied_text.encode()
                output_path = case_dir / f"{arm}.R"
                output_path.write_bytes(applied)
                output_sha = sha_bytes(applied)
            rows_out.append({
                "arm": arm, "case_index": index, "id": panel_row["id"], "family": panel_row["family"],
                "expected_operation": panel_row["operation"], "prompt_sha256": panel_row["prompt_sha256"],
                "baseline_relative_path": str(baseline_path.relative_to(args.output)),
                "baseline_sha256": context_sha, "reconstruction": reconstruction,
                "protocol_valid": parsed.status == "accepted", "protocol_status": parsed.status,
                "protocol_operation": parsed.operation, "protocol_reason": parsed.reason,
                "raw_text_sha256": native["raw_text_sha256"], "identity_checks": checks,
                "applied_relative_path": str(output_path.relative_to(args.output)) if output_path else None,
                "applied_sha256": output_sha, "edit_exact": native["quality"]["edit_exact"],
                "strict_noop_correct": native["quality"]["strict_noop_correct"],
                "noop_false_positive": native["quality"]["noop_false_positive"],
            })

    with (args.output / "replay-inputs.jsonl").open("w") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": "sepalith.notebook-dev-syntax-input.v1", "status": "materialized_not_parsed",
        "panel": {"path": str(args.panel), "sha256": PANEL_SHA256, "rows": 75},
        "results": {arm: {"path": str(path), "sha256": INPUT_SHA256[arm], "rows": 75}
                    for arm, path in INPUTS.items()},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha_file(PROTOCOL_PATH)},
        "denominators": {"cases": 75, "arms": 3, "arm_cases": 225,
                         "protocol_valid": sum(row["protocol_valid"] for row in rows_out),
                         "protocol_invalid": sum(not row["protocol_valid"] for row in rows_out)},
        "reconstruction": reconstruction_counts,
        "safety": {"generated_r_executed": False, "candidate_application": "text splice only"},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
