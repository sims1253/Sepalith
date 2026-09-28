#!/usr/bin/env python3
"""Audit five explicitly sampled converted TRAIN finish rows in memory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PACKET = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl")
PROTO = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/"
    "packages/sepalith/src/sepalith/campaign_protocol.py"
)
SCENARIOS = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module import unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_result(proto, result: dict) -> tuple[str, str]:
    context = result["context"]
    selection = result["selection_source"]
    before = selection["document_text"]
    assert sha256(before) == context["replacement_range"]["content_sha256"]
    lines = before.split("\n")
    replacement_range = context["replacement_range"]
    start = replacement_range["start"]
    end = replacement_range["end"]
    assert start["line"] == end["line"]
    line = lines[start["line"]]
    start_cp = proto.utf16_to_codepoint_column(line, start["character"])
    end_cp = proto.utf16_to_codepoint_column(line, end["character"])
    assert line[start_cp:end_cp] == context["region_old"][0] if context["region_old"] else start_cp == end_cp
    replacement = line[:start_cp] + "\n".join(result["target_body"]) + line[end_cp:]
    after = "\n".join(lines[:start["line"]] + replacement.split("\n") + lines[end["line"] + 1:])
    return before, after


def main() -> int:
    proto = load("dat08_finish_audit_protocol", PROTO)
    scenarios = load("dat08_finish_audit_scenarios", SCENARIOS)
    cases = []
    with PACKET.open(encoding="utf-8") as handle:
        for line_number, line in zip(range(1, 6), handle):
            packet = json.loads(line)
            assert packet["row_ref"]["split"] == "train_group"
            assert packet["family"] == "finish_block"
            result = packet["result"]
            provenance = result["provenance"]
            assert provenance["pre_edit_document"]["source_constructor"] == "finish_block_v5_prefix"
            assert provenance["target_convention"] == "suffix"
            assert provenance["finish_splice"] == {
                "literal_source_splice_verified": True,
                "outer_closing_brace_in_label": False,
            }
            before, after = apply_result(proto, result)
            after_parse_ok = not scenarios.parser.parse(after.encode("utf-8")).root_node.has_error
            after_plus_brace_parse_ok = not scenarios.parser.parse((after + "}").encode("utf-8")).root_node.has_error
            assert not after_parse_ok
            assert after_plus_brace_parse_ok
            cases.append(
                {
                    "packet_line": line_number,
                    "row_id": packet["row_ref"]["row_id"],
                    "kind": packet["row_ref"]["kind"],
                    "source_variant": packet["source_variant"],
                    "source_file": packet["row_ref"]["source_file"],
                    "source_line": packet["row_ref"]["source_line"],
                    "source_sha256": packet["row_ref"]["source_sha256"],
                    "raw_line_sha256": packet["row_ref"]["raw_line_sha256"],
                    "before_sha256": sha256(before),
                    "after_sha256": sha256(after),
                    "after_chars": len(after),
                    "operation": result["operation"],
                    "target_body_lines": len(result["target_body"]),
                    "target_body_sha256": provenance["target_body_sha256"],
                    "replacement_range": result["context"]["replacement_range"],
                    "after_parse_ok": after_parse_ok,
                    "after_plus_outer_brace_parse_ok": after_plus_brace_parse_ok,
                    "target_framing": provenance["target_framing"],
                }
            )
    print(
        json.dumps(
            {
                "assertions": "passed",
                "packet": str(PACKET),
                "sample_policy": "first five packet lines; each asserted train_group and finish_block",
                "cases": cases,
                "scope": "read five retained converted TRAIN packets only; no source corpus, DEV/final, model, network, SSH, GPU, or CLI access",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
