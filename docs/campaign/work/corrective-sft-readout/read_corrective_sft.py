#!/usr/bin/env python3
"""Stream and independently read a corrective SFT development evaluation.

The evaluator writes one JSON object containing a results array.  This driver
does not load a model or checkpoint: it validates the predeclared DEV panel,
walks that results array one case at a time, recomputes protocol/cap/strict
no-op counts, and audits the six source-bound finish rows.  The finish audit
applies each valid prediction to the recorded pre-edit document and parses the
result with the pinned tree-sitter R grammar.  Raw prompts and generated text
are never copied to the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Mapping


DEFAULT_PROTOCOL = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/"
    "prepared-state/snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/"
    "source/packages/sepalith/src/sepalith/campaign_protocol.py"
)
EXPECTED_FINISH_IDS = frozenset(
    {
        "e623a61b5a4c066358a477f2",
        "4f08633513b5c525240d2540",
        "d11581e9cfa4e3971aa1466e",
        "157517ba47dbab157f7c361a",
        "f43de3e77f2d92ed7b223464",
        "04834fef4fe59742f13677a9",
    }
)
EXPECTED_PANEL_COUNTS = {"cases": 75, "edits": 43, "strict_noop": 32, "finish_block": 6}
PARSER_PACKAGE = "tree_sitter_r"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_sha(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IncrementalJSON:
    """Small pull parser that keeps only one results object in memory."""

    def __init__(self, stream: Any, chunk_size: int = 64 * 1024) -> None:
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = ""
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self) -> None:
        if self.eof:
            return
        chunk = self.stream.read(self.chunk_size)
        if chunk == "":
            self.eof = True
        else:
            self.buffer += chunk

    def skip_ws(self) -> None:
        while True:
            stripped = self.buffer.lstrip()
            if stripped:
                self.buffer = stripped
                return
            if self.eof:
                return
            self.fill()

    def peek(self) -> str:
        self.skip_ws()
        if not self.buffer and self.eof:
            raise ValueError("unexpected end of JSON")
        return self.buffer[0]

    def punctuation(self, expected: str) -> None:
        self.skip_ws()
        while not self.buffer.startswith(expected):
            if self.eof:
                raise ValueError(f"expected {expected!r}")
            self.fill()
            self.skip_ws()
        self.buffer = self.buffer[len(expected) :]

    def value(self) -> Any:
        while True:
            self.skip_ws()
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()
                continue
            self.buffer = self.buffer[end:]
            return value


def stream_case_file(path: Path, on_result: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Read the evaluator object and call on_result for each result object."""
    meta: dict[str, Any] = {}
    with path.open(encoding="utf-8") as stream:
        reader = IncrementalJSON(stream)
        reader.punctuation("{")
        if reader.peek() == "}":
            reader.punctuation("}")
            return meta
        while True:
            key = reader.value()
            if not isinstance(key, str):
                raise ValueError("evaluation object key is not a string")
            reader.punctuation(":")
            if key == "results":
                reader.punctuation("[")
                if reader.peek() != "]":
                    while True:
                        result = reader.value()
                        if not isinstance(result, dict):
                            raise ValueError("evaluation result is not an object")
                        on_result(result)
                        separator = reader.peek()
                        if separator == "]":
                            break
                        reader.punctuation(",")
                reader.punctuation("]")
            else:
                meta[key] = reader.value()
            separator = reader.peek()
            if separator == "}":
                reader.punctuation("}")
                break
            reader.punctuation(",")
        reader.skip_ws()
        if reader.buffer or not reader.eof:
            # Permit only trailing whitespace.  fill once to distinguish it.
            reader.fill()
            if reader.buffer.strip():
                raise ValueError("trailing data after evaluation object")
    return meta


@dataclass
class PanelCase:
    row: dict[str, Any]
    context: Any
    expected_noop: bool


def read_panel(recipe: Mapping[str, Any], protocol: Any) -> tuple[list[PanelCase], dict[str, Any]]:
    panel_spec = recipe["development_panel"]
    panel = Path(panel_spec["path"])
    actual_sha = sha256_file(panel)
    expected_sha = panel_spec["sha256"]
    if actual_sha != expected_sha:
        raise ValueError(f"development panel SHA256 mismatch: {actual_sha} != {expected_sha}")
    expected_ids = list(recipe["development_case_ids"])
    cases: list[PanelCase] = []
    ids: set[str] = set()
    with panel.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"panel line {line_number} is not DEV")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise ValueError(f"invalid or duplicate DEV id at line {line_number}")
            if not isinstance(row.get("package_id"), str) or not isinstance(row.get("family"), str):
                raise ValueError(f"missing panel identity at line {line_number}")
            context = protocol.PromptContext.from_mapping(row["context"])
            expected_noop = list(context.region_old) == list(row["region_new"])
            if expected_noop != (row.get("operation") == "no_op"):
                raise ValueError(f"operation/region noop mismatch for {row_id}")
            target_text = row.get("target_body_text")
            expected_target_text = (
                "\n".join(row["region_new"])
                if row.get("operation") == "replace"
                else "[NO_EDIT]"
                if row.get("operation") == "no_op"
                else ""
            )
            if not isinstance(target_text, str) or target_text != expected_target_text:
                raise ValueError(f"target body mismatch for {row_id}")
            # The panel's target_sha256 is the token-row target identity; the
            # raw UTF-8 body hash is recorded only for the corrected
            # finish_block provenance.  Do not confuse those two domains.
            body_hash = row.get("source_provenance", {}).get("target_body_sha256")
            if body_hash is not None and sha256_text(target_text) != body_hash:
                raise ValueError(f"finish target body identity mismatch for {row_id}")
            ids.add(row_id)
            cases.append(PanelCase(row, context, expected_noop))
    if [case.row["id"] for case in cases] != expected_ids:
        raise ValueError("panel order or predeclared development identities changed")
    if len(cases) != EXPECTED_PANEL_COUNTS["cases"]:
        raise ValueError(f"expected 75 panel rows, found {len(cases)}")
    counts = Counter("strict_noop" if case.expected_noop else "edit" for case in cases)
    finish = {case.row["id"] for case in cases if case.row.get("family") == "finish_block"}
    if counts["edit"] != EXPECTED_PANEL_COUNTS["edits"] or counts["strict_noop"] != EXPECTED_PANEL_COUNTS["strict_noop"]:
        raise ValueError(f"panel denominator drift: {dict(counts)}")
    if finish != EXPECTED_FINISH_IDS:
        raise ValueError("finish_block identity set changed")
    return cases, {
        "path": str(panel),
        "sha256": actual_sha,
        "ordered_ids_sha256": canonical_sha(expected_ids),
        "cases": len(cases),
        "edits": counts["edit"],
        "strict_noop": counts["strict_noop"],
        "finish_block": len(finish),
    }


def valid_ids(protocol: Any, generated_ids: Any) -> bool:
    if not isinstance(generated_ids, list):
        return False
    try:
        return protocol.valid_generation_tokens(generated_ids)
    except (TypeError, ValueError):
        return False


def classify_result(
    result: Mapping[str, Any], case: PanelCase, protocol: Any, cap: int
) -> dict[str, Any]:
    generated_ids = result.get("generated_ids")
    raw = result.get("raw_output")
    parsed = None
    parse_reason = None
    if isinstance(raw, str):
        try:
            parsed = protocol.parse_output(raw, case.context)
        except Exception as error:  # protocol errors are an invalid prediction
            parse_reason = f"parser_exception:{type(error).__name__}"
    else:
        parse_reason = "output_not_string"
    token_valid = valid_ids(protocol, generated_ids)
    protocol_valid = bool(token_valid and parsed is not None and parsed.status == "accepted")
    cap_hit = (
        isinstance(generated_ids, list)
        and len(generated_ids) == cap
        and (not generated_ids or generated_ids[-1] != protocol.EOS_ID)
    )
    predicted_noop = protocol_valid and parsed.operation == "no_op"
    actual = list(case.context.region_old) if predicted_noop else (
        list(parsed.body) if protocol_valid and parsed is not None else []
    )
    exact_region = protocol_valid and actual == list(case.row["region_new"])
    if protocol_valid:
        failure = None
    elif cap_hit:
        failure = "generation_cap_without_canonical_eos"
    elif not token_valid:
        failure = "noncanonical_or_missing_eos"
    elif parsed is None:
        failure = parse_reason or "output_parse_failed"
    else:
        failure = parsed.reason or "output_not_accepted"
    issues: list[str] = []
    reported_cap = result.get("cap_hit")
    if isinstance(reported_cap, bool) and reported_cap != cap_hit:
        issues.append("embedded_cap_hit_mismatch")
    generated_count = result.get("generated_tokens")
    if isinstance(generated_count, int) and generated_count != (
        len(generated_ids) if isinstance(generated_ids, list) else -1
    ):
        issues.append("embedded_generated_tokens_mismatch")
    return {
        "protocol_valid": protocol_valid,
        "exact_region": bool(exact_region),
        "predicted_noop": predicted_noop,
        "suggestion": protocol_valid and not predicted_noop,
        "cap_hit": cap_hit,
        "failure": failure,
        "generated_tokens": len(generated_ids) if isinstance(generated_ids, list) else None,
        "raw_output_sha256": sha256_text(raw) if isinstance(raw, str) else None,
        "generated_ids_sha256": canonical_sha(generated_ids) if isinstance(generated_ids, list) else None,
        "issues": issues,
        "_parsed": parsed,
    }


def document_offset(document: str, line: int, character: int, protocol: Any) -> int:
    lines = document.split("\n")
    if type(line) is not int or type(character) is not int or line < 0 or character < 0:
        raise ValueError("replacement position is not a nonnegative integer")
    if line >= len(lines):
        raise ValueError("replacement line is outside bound document")
    column = protocol.utf16_to_codepoint_column(lines[line], character)
    return sum(len(item) + 1 for item in lines[:line]) + column


def apply_prediction(
    case: PanelCase, parsed: Any, protocol: Any
) -> tuple[str, str, dict[str, Any]]:
    provenance = case.row.get("source_provenance", {})
    selection = provenance.get("selection_source", {})
    document = selection.get("document_text")
    if not isinstance(document, str):
        raise ValueError("finish selection source has no bound document_text")
    replacement_range = case.context.replacement_range
    if sha256_text(document) != replacement_range.content_sha256:
        raise ValueError("bound document hash differs from replacement range")
    if selection.get("content_sha256") != replacement_range.content_sha256:
        raise ValueError("selection source hash differs from replacement range")
    pre_edit = provenance.get("pre_edit_document", {})
    if isinstance(pre_edit, dict) and pre_edit.get("content_sha256") not in (
        None,
        replacement_range.content_sha256,
    ):
        raise ValueError("pre-edit document hash differs from replacement range")
    start = replacement_range.start
    end = replacement_range.end
    start_offset = document_offset(document, start.line, start.character, protocol)
    end_offset = document_offset(document, end.line, end.character, protocol)
    if end_offset < start_offset:
        raise ValueError("replacement range is reversed")
    selected = document[start_offset:end_offset]
    expected_old = "\n".join(case.context.region_old)
    if selected != expected_old:
        raise ValueError("bound document selection differs from region_old")
    if parsed.operation == "no_op":
        replacement = selected
    elif parsed.operation == "delete":
        replacement = ""
    else:
        replacement = "\n".join(parsed.body)
    after = document[:start_offset] + replacement + document[end_offset:]
    return document, after, {
        "before_sha256": sha256_text(document),
        "after_sha256": sha256_text(after),
        "selected_sha256": sha256_text(selected),
        "replacement_sha256": sha256_text(replacement),
        "start": start.to_dict(),
        "end": end.to_dict(),
    }


def make_r_parser() -> Any:
    from tree_sitter import Language, Parser
    import tree_sitter_r

    return Parser(Language(tree_sitter_r.language()))


def parse_r(parser: Any, document: str) -> bool:
    tree = parser.parse(document.encode("utf-8"))
    return not tree.root_node.has_error


def readout(
    recipe_path: Path, cases_path: Path, step: int, output_path: Path
) -> dict[str, Any]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe_sha = sha256_file(recipe_path)
    protocol_path = Path(recipe.get("identity", {}).get("source_path", "")) / "packages/sepalith/src/sepalith/campaign_protocol.py"
    if not protocol_path.is_file():
        protocol_path = DEFAULT_PROTOCOL
    protocol = load_module("corrective_sft_readout_protocol", protocol_path)
    panel_cases, panel = read_panel(recipe, protocol)
    by_id = {case.row["id"]: case for case in panel_cases}
    results_seen: list[str] = []
    counts = Counter()
    families: dict[str, Counter] = {}
    issues: list[str] = []
    finish_reports: list[dict[str, Any]] = []
    loss_denominators = Counter()
    loss_sums = Counter()
    try:
        parser = make_r_parser()
        parser_status = "available"
    except Exception as error:
        parser = None
        parser_status = f"unavailable:{type(error).__name__}"

    def consume(result: dict[str, Any]) -> None:
        row_id = result.get("id")
        if not isinstance(row_id, str) or row_id not in by_id:
            issues.append("unknown_result_id")
            return
        if row_id in results_seen:
            issues.append(f"duplicate_result_id:{row_id}")
            return
        results_seen.append(row_id)
        case = by_id[row_id]
        if result.get("package_id") != case.row["package_id"]:
            issues.append(f"package_identity_mismatch:{row_id}")
        if result.get("family") != case.row["family"]:
            issues.append(f"family_identity_mismatch:{row_id}")
        if isinstance(result.get("expected_noop"), bool) and result["expected_noop"] != case.expected_noop:
            issues.append(f"expected_noop_identity_mismatch:{row_id}")
        outcome = classify_result(result, case, protocol, int(recipe["development_max_new_tokens"]))
        issues.extend(f"{issue}:{row_id}" for issue in outcome["issues"])
        for key in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit"):
            counts[key] += int(outcome[key])
        if case.expected_noop and outcome["predicted_noop"]:
            counts["strict_noop_correct"] += 1
        if case.expected_noop and outcome["suggestion"]:
            counts["strict_noop_false_suggestions"] += 1
        if not case.expected_noop and outcome["exact_region"]:
            counts["edit_exact"] += 1
        family_counter = families.setdefault(case.row["family"], Counter())
        family_counter["cases"] += 1
        family_counter["expected_noop"] += int(case.expected_noop)
        family_counter["exact_region"] += int(outcome["exact_region"])
        family_counter["protocol_valid"] += int(outcome["protocol_valid"])
        family_counter["cap_hit"] += int(outcome["cap_hit"])
        loss = result.get("loss")
        if isinstance(loss, dict):
            for name in ("prompt", "target"):
                tokens = loss.get(f"{name}_tokens")
                nll_sum = loss.get(f"{name}_nll_sum")
                if type(tokens) is int and tokens >= 0 and isinstance(nll_sum, (int, float)) and math.isfinite(nll_sum):
                    loss_denominators[name] += tokens
                    loss_sums[name] += float(nll_sum)
                else:
                    issues.append(f"invalid_{name}_loss:{row_id}")
        if case.row["id"] in EXPECTED_FINISH_IDS:
            finish: dict[str, Any] = {
                "id": row_id,
                "protocol_valid": outcome["protocol_valid"],
                "exact_region": outcome["exact_region"],
                "cap_hit": outcome["cap_hit"],
                "predicted_noop": outcome["predicted_noop"],
                "failure": outcome["failure"],
                "generated_tokens": outcome["generated_tokens"],
                "raw_output_sha256": outcome["raw_output_sha256"],
                "generated_ids_sha256": outcome["generated_ids_sha256"],
                "r_parse": "not_evaluated_invalid_protocol",
            }
            if outcome["protocol_valid"] and outcome["_parsed"] is not None:
                try:
                    before, after, geometry = apply_prediction(case, outcome["_parsed"], protocol)
                    finish["document_before_sha256"] = geometry["before_sha256"]
                    finish["post_document_sha256"] = geometry["after_sha256"]
                    finish["post_document_chars"] = len(after)
                    finish["r_parse"] = (
                        bool(parse_r(parser, after)) if parser is not None else "parser_unavailable"
                    )
                    expected_after = case.row.get("source_provenance", {}).get(
                        "correction", {}
                    ).get("new_post_document_sha256")
                    if outcome["exact_region"] and isinstance(expected_after, str):
                        finish["corrected_target_post_hash_matches"] = geometry["after_sha256"] == expected_after
                except Exception as error:
                    finish["r_parse"] = "not_evaluated_geometry_error"
                    finish["geometry_error"] = type(error).__name__
            finish_reports.append(finish)

    meta: dict[str, Any]
    if cases_path.is_file():
        meta = stream_case_file(cases_path, consume)
    else:
        meta = {"status": "missing"}
        issues.append("evaluation_cases_file_missing")
    complete = meta.get("status") == "complete"
    if meta.get("step") is not None and meta.get("step") != step:
        issues.append("evaluation_step_mismatch")
    order_matches = results_seen == [case.row["id"] for case in panel_cases]
    if complete and len(results_seen) != len(panel_cases):
        issues.append("complete_status_has_incomplete_result_count")
    if complete and not order_matches:
        issues.append("complete_result_order_differs_from_panel")
    if len(results_seen) == len(panel_cases) and set(results_seen) == set(by_id):
        result_identity_status = "exact_set"
    else:
        result_identity_status = "partial_or_mismatched"
    summary = meta.get("summary")
    embedded_mismatches: list[str] = []
    if isinstance(summary, dict):
        embedded_denoms = summary.get("denominators", {})
        for name, value in {
            "cases": len(results_seen),
            "strict_noop": EXPECTED_PANEL_COUNTS["strict_noop"],
            "edits": EXPECTED_PANEL_COUNTS["edits"],
        }.items():
            if isinstance(embedded_denoms, dict) and embedded_denoms.get(name) != value:
                embedded_mismatches.append(f"denominator:{name}")
        embedded_counts = summary.get("counts", {})
        for name in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit"):
            if isinstance(embedded_counts, dict) and embedded_counts.get(name) != counts[name]:
                embedded_mismatches.append(f"count:{name}")
    report = {
        "schema_version": 1,
        "status": "complete_readout" if complete and not issues else (
            "partial_readout" if cases_path.is_file() else "prepared_milestone_pending"
        ),
        "step": step,
        "recipe": {"path": str(recipe_path), "sha256": recipe_sha, "id": recipe.get("id"), "renderer_id": recipe.get("renderer_id")},
        "panel": panel,
        "evaluation": {
            "path": str(cases_path),
            "sha256": sha256_file(cases_path) if cases_path.is_file() else None,
            "file_status": meta.get("status"),
            "file_step": meta.get("step"),
            "result_count": len(results_seen),
            "result_identity": result_identity_status,
            "ordered_ids_match_panel": order_matches,
            "embedded_summary_mismatches": embedded_mismatches,
        },
        "denominators": {
            "cases": len(results_seen),
            "packages": len({by_id[row_id].row["package_id"] for row_id in results_seen}),
            "strict_noop": EXPECTED_PANEL_COUNTS["strict_noop"],
            "edits": EXPECTED_PANEL_COUNTS["edits"],
            "finish_block": EXPECTED_PANEL_COUNTS["finish_block"],
            "prompt_loss_tokens": loss_denominators["prompt"],
            "target_loss_tokens": loss_denominators["target"],
        },
        "counts": {name: counts[name] for name in (
            "protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit",
            "strict_noop_correct", "strict_noop_false_suggestions", "edit_exact",
        )},
        "family": {name: dict(counter) for name, counter in sorted(families.items())},
        "loss": {
            "prompt_nll": (
                loss_sums["prompt"] / loss_denominators["prompt"]
                if loss_denominators["prompt"] else None
            ),
            "target_nll": (
                loss_sums["target"] / loss_denominators["target"]
                if loss_denominators["target"] else None
            ),
        },
        "finish_block_audit": {
            "denominator": 6,
            "ids": sorted(EXPECTED_FINISH_IDS),
            "protocol_valid": sum(item["protocol_valid"] for item in finish_reports),
            "cap_hits": sum(item["cap_hit"] for item in finish_reports),
            "exact_region": sum(item["exact_region"] for item in finish_reports),
            "r_parse_ok": sum(item["r_parse"] is True for item in finish_reports),
            "r_parse_evaluated": sum(item["r_parse"] in (True, False) for item in finish_reports),
            "cases": finish_reports,
            "parser": {
                "package": PARSER_PACKAGE,
                "status": parser_status,
                "rule": "tree-sitter R root has_error must be false after exact UTF-16 replacement",
            },
        },
        "validation": {
            "issues": issues,
            "supported": (
                "complete or partial campaign_eval JSON; panel/result identity, protocol/EOS/cap, "
                "strict no-op/edit denominators, weighted loss denominators, and six finish document parses"
            ),
            "raw_prompt_or_output_in_report": False,
            "model_or_framework_loaded": False,
        },
        "scientific_scope": (
            "Preparation/readout only. Exact/protocol/no-op and bounded R parse evidence do not establish "
            "model quality, semantic correctness for non-finish edits, or promotion."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    cases = args.cases or (
        Path(recipe["archive_dir"]) / "evaluations" / f"cases-step-{args.step}.json"
    )
    report = readout(args.recipe, cases, args.step, args.out)
    print(json.dumps({
        "status": report["status"],
        "result_count": report["evaluation"]["result_count"],
        "denominators": report["denominators"],
        "counts": report["counts"],
        "finish": {
            "denominator": report["finish_block_audit"]["denominator"],
            "r_parse_ok": report["finish_block_audit"]["r_parse_ok"],
        },
        "issues": len(report["validation"]["issues"]),
    }, sort_keys=True))
    return 0 if report["status"] == "complete_readout" else 2


if __name__ == "__main__":
    raise SystemExit(main())
