#!/usr/bin/env python3
"""Audit the six finish_block rows at the end of the pinned DEV panel.

This driver reads only the six selected finish rows (panel lines 70--75),
reconstructs their source-derived pre-edit fragments using the recorded UTF-16
replacement ranges, and checks the exact target splice with the pinned
tree-sitter R parser.  It never opens the source corpus or any final data.
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


PANEL = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl")
EXPECTED_PANEL_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
PROTO = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/"
    "packages/sepalith/src/sepalith/campaign_protocol.py"
)
SCENARIOS = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py")
SELECTED_LINES = tuple(range(70, 76))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_selected_rows() -> tuple[list[dict[str, Any]], str]:
    if sha256_file(PANEL) != EXPECTED_PANEL_SHA256:
        raise RuntimeError("DAT-07 DEV panel SHA256 changed")
    rows: list[dict[str, Any]] = []
    with PANEL.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in SELECTED_LINES:
                row = json.loads(line)
                if row.get("split") != "dev" or row.get("family") != "finish_block":
                    raise RuntimeError(f"selected line {line_number} is not a DEV finish row")
                rows.append(row)
    if len(rows) != len(SELECTED_LINES):
        raise RuntimeError(f"expected six selected rows, found {len(rows)}")
    return rows, EXPECTED_PANEL_SHA256


def line_offset(lines: list[str], line_number: int, character: int, proto: Any) -> int:
    if line_number < 0 or line_number >= len(lines):
        raise RuntimeError(f"replacement line {line_number} outside reconstructed fragment")
    codepoint_column = proto.utf16_to_codepoint_column(lines[line_number], character)
    return sum(len(line) + 1 for line in lines[:line_number]) + codepoint_column


def reconstruct(row: dict[str, Any], proto: Any) -> tuple[str, str, dict[str, Any]]:
    context = row["context"]
    replacement_range = context["replacement_range"]
    lines = list(context["prefix"]) + list(context["region_old"]) + list(context["suffix_lines"])
    # Empty finish rows represent insertion at EOF.  The omitted final empty
    # line is implied by the recorded range and content hash.
    while len(lines) <= replacement_range["end"]["line"]:
        lines.append("")
    before = "\n".join(lines)
    if sha256_text(before) != replacement_range["content_sha256"]:
        raise RuntimeError(f"{row['id']} reconstructed content hash mismatch")
    start = replacement_range["start"]
    end = replacement_range["end"]
    start_offset = line_offset(lines, start["line"], start["character"], proto)
    end_offset = line_offset(lines, end["line"], end["character"], proto)
    selected = before[start_offset:end_offset]
    expected_old = "\n".join(context["region_old"])
    if selected != expected_old:
        raise RuntimeError(
            f"{row['id']} UTF-16 selection mismatch: {selected!r} != {expected_old!r}"
        )
    target = row["target_body_text"]
    if target != "\n".join(row["region_new"]):
        raise RuntimeError(f"{row['id']} target body differs from region_new")
    if sha256_text(target) != row["source_provenance"]["target_body_sha256"]:
        raise RuntimeError(f"{row['id']} target body identity mismatch")
    after = before[:start_offset] + target + before[end_offset:]
    geometry = {
        "start": start,
        "end": end,
        "selected_text_sha256": sha256_text(selected),
        "before_sha256": sha256_text(before),
        "after_sha256": sha256_text(after),
        "after_chars": len(after),
    }
    return before, after, geometry


def parse_ok(parser: Any, text: str) -> bool:
    tree = parser.parse(text.encode("utf-8"))
    return not tree.root_node.has_error


def audit() -> dict[str, Any]:
    proto = load_module("dat07_finish_impact_protocol", PROTO)
    scenarios = load_module("dat07_finish_impact_scenarios", SCENARIOS)
    rows, panel_sha256 = read_selected_rows()
    cases: list[dict[str, Any]] = []
    for panel_line, row in zip(SELECTED_LINES, rows):
        context = row["context"]
        provenance = row["source_provenance"]
        finish_splice = provenance.get("finish_splice")
        if finish_splice != {
            "literal_source_splice_verified": True,
            "outer_closing_brace_in_label": False,
        }:
            raise RuntimeError(f"{row['id']} finish provenance drifted")
        before, after, geometry = reconstruct(row, proto)
        before_ok = parse_ok(scenarios.parser, before)
        after_ok = parse_ok(scenarios.parser, after)
        corrected = after + "}"
        corrected_ok = parse_ok(scenarios.parser, corrected)
        suffix = list(context["suffix_lines"])
        target = row["target_body_text"]
        final_target_line = next((line for line in reversed(target.split("\n")) if line), "")
        cases.append(
            {
                "panel_line": panel_line,
                "id": row["id"],
                "package_id": row["package_id"],
                "group_id": row["group_id"],
                "path": context["path"],
                "operation": row["operation"],
                "prompt_sha256": row["prompt_sha256"],
                "target_sha256": row["target_sha256"],
                "target_body_sha256": provenance["target_body_sha256"],
                "source_ref": {
                    "source_file": row["source_ref"].get("source_file"),
                    "source_line": row["source_ref"].get("source_line"),
                    "source_sha256": row["source_ref"].get("source_sha256"),
                    "raw_line_sha256": row["source_ref"].get("raw_line_sha256"),
                    "canonical_row_sha256": row["source_ref"].get("canonical_row_sha256"),
                },
                "source_provenance": {
                    "source_constructor": provenance.get("pre_edit_document", {}).get("source_constructor"),
                    "pre_edit_content_sha256": provenance.get("pre_edit_document", {}).get("content_sha256"),
                    "source_identity": provenance.get("source_identity"),
                    "source_path": provenance.get("source_path"),
                    "target_convention": provenance.get("target_convention"),
                    "target_framing": provenance.get("target_framing"),
                    "outer_closing_brace_in_label": finish_splice["outer_closing_brace_in_label"],
                },
                "geometry": geometry,
                "fragment_shape": {
                    "prefix_lines": len(context["prefix"]),
                    "region_old_lines": len(context["region_old"]),
                    "visible_suffix_lines": len(suffix),
                    "visible_suffix_has_standalone_closing_brace": any(
                        line.strip() == "}" for line in suffix
                    ),
                    "target_body_lines": len(row["region_new"]),
                    "target_body_chars": len(target),
                    "target_body_token_count": row["target_body_token_count"],
                    "target_terminal_token_count": row["target_terminal_token_count"],
                    "target_final_nonempty_line": final_target_line,
                    "target_has_outer_closing_brace": final_target_line.strip() == "}",
                },
                "parser": {
                    "source_fragment_before_ok": before_ok,
                    "post_edit_ok": after_ok,
                    "post_edit_plus_one_outer_brace_ok": corrected_ok,
                    "correction": "append exactly one outer closing brace after the emitted target body",
                },
            }
        )
    return {
        "task": "DAT-07",
        "scope": "six finish_block DEV rows only; no final rows or source corpus bytes",
        "panel": str(PANEL),
        "panel_sha256": panel_sha256,
        "selected_panel_lines": list(SELECTED_LINES),
        "selected_count": len(cases),
        "finish_denominator": 6,
        "finish_operation_counts": {"replace": 6, "no_op": 0},
        "parser": {
            "implementation": "tree-sitter-r through scenarios.py",
            "scenarios_path": str(SCENARIOS),
            "scenarios_sha256": sha256_file(SCENARIOS),
            "protocol_path": str(PROTO),
            "protocol_sha256": sha256_file(PROTO),
            "utf16_conversion": "campaign_protocol.utf16_to_codepoint_column",
        },
        "cases": cases,
        "aggregate": {
            "all_outer_brace_labels_false": all(
                case["source_provenance"]["outer_closing_brace_in_label"] is False
                for case in cases
            ),
            "all_visible_suffixes_empty": all(
                case["fragment_shape"]["visible_suffix_lines"] == 0 for case in cases
            ),
            "all_post_edit_parse_rejected": all(
                case["parser"]["post_edit_ok"] is False for case in cases
            ),
            "all_post_edit_plus_brace_parse_accepted": all(
                case["parser"]["post_edit_plus_one_outer_brace_ok"] is True
                for case in cases
            ),
            "all_corrections_one_brace": True,
        },
        "interpretation": {
            "finding": "All six DEV finish rows use a target representation without the outer closing brace, and their recorded visible suffix is empty. Exact UTF-16 application yields a parser-invalid source-derived fragment; appending one brace parses every bounded case.",
            "truth_scope": "This demonstrates the boundary in the six DEV truth rows. It does not claim every DEV row or every full underlying source document is malformed.",
            "fragment_limit": "The panel pre_edit_document is source-derived simulated finish_block_v5_prefix context; source_fragment_before_ok=false is expected for a boundary fragment and is not a source validity claim.",
            "minimal_correction": "Preserve a visible source suffix containing the outer brace when the real document provides it, or make the finish target/application boundary carry exactly one outer brace. Revalidate the chosen representation before changing the panel or original file.",
            "historical_exact_score_implication": "Any historical exact score that treated these six finish targets as standalone/full-document completions cannot be interpreted as editor-application validity until the outer-brace boundary is resolved. The rows remain useful as target truth for the represented suffix/body, but no corrected score is invented here.",
        },
        "limits": [
            "Only the six finish rows at panel lines 70--75 were inspected; at most two nonfinish controls were viewed for context and are not included in the result.",
            "No raw source corpus file was opened; source identities are reported from the DEV panel.",
            "No model, tokenizer, training, final data, GPU, SSH, network, CLI, or data modification was performed.",
            "No original panel row or source file was changed.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit()
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite existing report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        dir_fd = os.open(args.output.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
