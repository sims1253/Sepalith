#!/usr/bin/env python3
"""Parse generated edit buffers with base R, without evaluating them.

The TRAIN row stores the FIM prefix and suffix around the selected region.
For an accepted edit response this utility replaces the current region in
that stored FIM context with the response body, writes a temporary ``.R``
buffer, and runs ``Rscript --vanilla`` with ``parse(file=...)`` only.  It does
not call ``source`` or evaluate R code.  No-op rows are excluded from the
applied-edit denominator; a no-op response leaves the stored current region
unchanged and is recorded only as a context check when requested by a caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "sepalith.r2.serving.train-panel.v1"
OUTPUT_SCHEMA = "sepalith.r2.serving.train-panel-probe.v1"
TERMINAL = ">>>>>>> UPDATED"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_rows(panel_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected panel manifest schema")
    records = manifest.get("panel", {}).get("records", [])
    raw_lines = panel_path.read_bytes().splitlines(keepends=True)
    if not isinstance(records, list) or len(records) != len(raw_lines):
        raise ValueError("panel and manifest record counts differ")
    rows: list[dict[str, Any]] = []
    for raw, identity in zip(raw_lines, records):
        row = json.loads(raw)
        if not isinstance(row, dict) or not isinstance(identity, dict):
            raise ValueError("panel and manifest rows must be objects")
        if identity.get("row_id") != row.get("id"):
            raise ValueError(f"panel row identity mismatch for {row.get('id')!r}")
        if identity.get("row_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"panel row byte hash mismatch for {row.get('id')!r}")
        rows.append(row)
    if len(rows) != 8:
        raise ValueError("quant qualification requires exactly eight panel rows")
    return rows, manifest


def index_requests(result: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    if result.get("schema_version") != OUTPUT_SCHEMA:
        raise ValueError("probe output schema mismatch")
    requests = result.get("requests")
    if not isinstance(requests, list):
        raise ValueError("probe output has no requests")
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("probe request is not an object")
        key = (str(item.get("row_id")), str(item.get("phase")), int(item.get("rep", 0)))
        if key in indexed:
            raise ValueError(f"duplicate probe request {key}")
        indexed[key] = item
    return indexed


def fim_parts(prompt: str) -> tuple[str, str, str]:
    """Return prefix, current region, suffix from the stored FIM prompt."""
    selected = "<filename>selected_references"
    suffix_marker = "<[fim-suffix]>"
    current_marker = "<<<<<<< CURRENT"
    middle_marker = "======="
    if selected not in prompt or suffix_marker not in prompt or current_marker not in prompt:
        raise ValueError("prompt lacks FIM context markers")
    before_selected, _ = prompt.split(selected, 1)
    filename = before_selected.find("<filename>")
    if filename < 0:
        raise ValueError("prompt lacks filename marker")
    filename_end = before_selected.find("\n", filename)
    if filename_end < 0:
        raise ValueError("filename marker has no line ending")
    prefix = before_selected[filename_end + 1 :]
    suffix_start = prompt.index(suffix_marker) + len(suffix_marker)
    current_start = prompt.index(current_marker, suffix_start)
    suffix = prompt[suffix_start:current_start]
    current_end = prompt.find(middle_marker, current_start + len(current_marker))
    if current_end < 0:
        raise ValueError("prompt lacks current-region separator")
    current = prompt[current_start + len(current_marker):current_end]
    # The cursor marker is metadata, not R source.  Preserve all other bytes.
    current = current.replace("<|user_cursor|>", "")
    return prefix, current, suffix


def parse_file(rscript: str, path: Path) -> tuple[int, str, str]:
    expression = (
        "args <- commandArgs(trailingOnly=TRUE); "
        "parse(file=args[[1]], keep.source=FALSE); cat('PARSE_PASS\\n')"
    )
    completed = subprocess.run(
        [rscript, "--vanilla", "-e", expression, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return completed.returncode, completed.stdout[-2000:], completed.stderr[-2000:]


def parse_arm(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    arm: str,
    manifest_sha256: str,
    rscript: str,
) -> dict[str, Any]:
    if result.get("panel", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"{arm} probe manifest hash mismatch")
    indexed = index_requests(result)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="r2-quant-applied-") as temporary:
        temp_root = Path(temporary)
        for row in rows:
            if row.get("target_operation") == "no_op":
                continue
            key = (row["id"], "cold", 1)
            record = indexed.get(key)
            base = {
                "row_id": row["id"],
                "family": row.get("family"),
                "target_operation": row.get("target_operation"),
                "source_prompt_sha256": hashlib.sha256(row["prompt_text"].encode("utf-8")).hexdigest(),
                "phase": "cold",
                "rep": 1,
            }
            if record is None:
                cases.append({**base, "status": "not_attempted", "reason": "missing_cold_record"})
                continue
            if record.get("protocol_status") != "accepted":
                cases.append({
                    **base,
                    "status": "not_attempted",
                    "reason": "protocol_not_accepted",
                    "protocol_status": record.get("protocol_status"),
                })
                continue
            parsed = record.get("parsed_output")
            if not isinstance(parsed, dict) or parsed.get("status") != "accepted":
                cases.append({**base, "status": "not_attempted", "reason": "parsed_output_not_accepted"})
                continue
            prefix, current, suffix = fim_parts(row["prompt_text"])
            if parsed.get("operation") == "no_op":
                candidate = current
                candidate_source = "stored_current_region_after_noop"
            else:
                candidate = parsed.get("body_text")
                if not isinstance(candidate, str):
                    cases.append({**base, "status": "not_attempted", "reason": "replacement_body_missing"})
                    continue
                candidate_source = "accepted_model_replacement_body"
            applied = prefix + candidate + suffix
            path = temp_root / f"{row['id']}.R"
            path.write_text(applied, encoding="utf-8", newline="")
            try:
                exit_code, stdout, stderr = parse_file(rscript, path)
            except (OSError, subprocess.SubprocessError) as exc:
                cases.append({**base, "status": "parser_error", "error": str(exc)})
                continue
            cases.append({
                **base,
                "status": "pass" if exit_code == 0 else "fail",
                "protocol_status": record.get("protocol_status"),
                "model_operation": parsed.get("operation"),
                "candidate_source": candidate_source,
                "buffer_bytes": len(applied.encode("utf-8")),
                "buffer_sha256": hashlib.sha256(applied.encode("utf-8")).hexdigest(),
                "r_exit_code": exit_code,
                "r_stdout": stdout,
                "r_stderr": stderr,
            })
    attempted = [case for case in cases if case["status"] in {"pass", "fail"}]
    return {
        "arm": arm,
        "cases": len(cases),
        "attempted": len(attempted),
        "passed": sum(case["status"] == "pass" for case in cases),
        "failed": sum(case["status"] == "fail" for case in cases),
        "not_attempted": sum(case["status"] not in {"pass", "fail"} for case in cases),
        "status": "complete" if len(cases) == 4 else "partial",
        "cases_detail": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--f16", type=Path, required=True)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    try:
        rows, manifest = load_rows(args.panel, args.manifest)
        manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        f16_result, q8_result = load_json(args.f16), load_json(args.q8)
        f16 = parse_arm(rows, f16_result, "f16", manifest_sha256, args.rscript)
        q8 = parse_arm(rows, q8_result, "q8", manifest_sha256, args.rscript)
        result = {
            "schema_version": "sepalith.r2.final-quant.applied-r-parse.v1",
            "status": "measured_requires_root_review" if f16["status"] == q8["status"] == "complete" else "partial",
            "parser": f"{args.rscript} --vanilla; base parse(file=..., keep.source=FALSE); no eval/source",
            "scope": "four selected TRAIN edit rows per arm; FIM prefix/current/suffix reconstruction",
            "panel": {
                "path": str(args.panel),
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha256,
                "source_sha256": manifest["source"]["sha256"],
                "edit_rows": 4,
            },
            "f16": f16,
            "q8": q8,
            "same_parse_outcomes": [
                left["status"] == right["status"]
                for left, right in zip(f16["cases_detail"], q8["cases_detail"])
            ],
            "interpretation": {
                "parse_is_syntax_only": True,
                "model_quality_not_inferred": True,
                "no_op_rows_excluded_from_applied_edit_denominator": True,
                "full_source_file_execution": False,
            },
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": "sepalith.r2.final-quant.applied-r-parse.v1",
            "status": "failed",
            "error": str(exc),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "f16": [f16["passed"], f16["failed"], f16["not_attempted"]],
        "q8": [q8["passed"], q8["failed"], q8["not_attempted"]],
        "out": str(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
