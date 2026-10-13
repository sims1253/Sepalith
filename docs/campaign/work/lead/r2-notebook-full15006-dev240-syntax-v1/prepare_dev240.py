#!/usr/bin/env python3
"""Materialize context-aware DEV240 applied buffers for parse-only replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "r2-notebook-dev-syntax-v1"
sys.path.insert(0, str(PRIOR))
from prepare_replay import load_protocol, splice  # noqa: E402


PANEL = HERE.parent / "corrected-dev75-v1" / "dev75-corrected-finish-v1.jsonl"
PANEL_SHA = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
RESULT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-full15006-a/evaluations/cases-step-240.json")
RESULT_SHA = "ba052bf649a56143e01dcddd8800a1090542c15907c92cb3057602cae2785bbc"
PRIOR_MANIFEST_SHA = "d3695131bfe62003526777b275fc91cc49a330352990286fd149189086fe56b1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh output required")
    require(sha(PANEL) == PANEL_SHA and sha(RESULT) == RESULT_SHA, "input hash mismatch")
    require(sha(PRIOR / "packet" / "manifest.json") == PRIOR_MANIFEST_SHA, "reviewed prior packet identity mismatch")
    panel = [json.loads(line) for line in PANEL.open(encoding="utf-8")]
    document = json.loads(RESULT.read_text(encoding="utf-8"))
    ids = [row["id"] for row in panel]
    require(len(panel) == 75 and len(set(ids)) == 75, "panel denominator or IDs mismatch")
    require(document.get("status") == "complete" and document.get("step") == 240, "DEV240 result status/step mismatch")
    require(document["summary"]["panel_sha256"] == PANEL_SHA, "DEV240 panel binding mismatch")
    require(document["summary"]["case_ids"] == ids and len(document["results"]) == 75, "DEV240 case order mismatch")
    protocol = load_protocol()
    args.output.mkdir(parents=True)
    buffers = args.output / "buffers"
    buffers.mkdir()
    rows: list[dict] = []
    for index, (panel_row, native) in enumerate(zip(panel, document["results"])):
        require(native["id"] == panel_row["id"], "case ID mismatch")
        require(native["family"] == panel_row["family"], "case family mismatch")
        require(native["package_id"] == panel_row["package_id"], "case package mismatch")
        require(native["expected_noop"] == (panel_row["operation"] == "no_op"), "expected operation mismatch")
        case_dir = buffers / f"{index:03d}"
        case_dir.mkdir()
        prior_baseline = PRIOR / "packet" / "buffers" / f"{index:03d}" / "baseline.R"
        baseline = prior_baseline.read_bytes()
        context_sha = panel_row["context"]["replacement_range"]["content_sha256"]
        require(sha_bytes(baseline) == context_sha, "reviewed baseline identity mismatch")
        baseline_path = case_dir / "baseline.R"
        baseline_path.write_bytes(baseline)
        context = protocol.PromptContext.from_dict(panel_row["context"])
        rendered = protocol.render_prompt(context)
        require(sha_bytes(rendered.encode("utf-8")) == panel_row["prompt_sha256"], "prompt render mismatch")
        parsed = protocol.parse_output(native["raw_output"], context)
        require((parsed.status == "accepted") == native["protocol_valid"], "context parser validity mismatch")
        require((parsed.operation == "no_op") == native["predicted_noop"] if native["protocol_valid"] else parsed.operation is None,
                "context parser operation mismatch")
        applied_path = None
        applied_sha = None
        if parsed.status == "accepted":
            if parsed.operation == "no_op":
                applied = baseline
            else:
                eol = "\r\n" if context.document_eol == "crlf" else "\n"
                applied = splice(
                    baseline.decode("utf-8", "strict"),
                    panel_row["context"]["replacement_range"],
                    parsed.body_text,
                    eol,
                ).encode("utf-8")
            applied_path = case_dir / "full15006-step240.R"
            applied_path.write_bytes(applied)
            applied_sha = sha_bytes(applied)
        edit_exact = bool(native["exact_region"] and not native["expected_noop"])
        strict_noop_correct = bool(native["expected_noop"] and native["exact_region"] and not native["suggestion"])
        noop_false_positive = bool(native["expected_noop"] and native["suggestion"])
        rows.append({
            "arm": "full15006-step240", "case_index": index, "id": panel_row["id"],
            "family": panel_row["family"], "package_id": panel_row["package_id"],
            "expected_operation": panel_row["operation"], "prompt_sha256": panel_row["prompt_sha256"],
            "baseline_relative_path": str(baseline_path.relative_to(args.output)),
            "baseline_sha256": context_sha,
            "protocol_valid": parsed.status == "accepted", "protocol_status": parsed.status,
            "protocol_operation": parsed.operation, "protocol_reason": parsed.reason,
            "recorded_failure": native["failure"], "cap_hit": native["cap_hit"],
            "raw_text_sha256": sha_bytes(native["raw_output"].encode("utf-8")),
            "generated_ids_sha256": hashlib.sha256(json.dumps(native["generated_ids"], separators=(",", ":")).encode("ascii")).hexdigest(),
            "applied_relative_path": str(applied_path.relative_to(args.output)) if applied_path else None,
            "applied_sha256": applied_sha, "edit_exact": edit_exact,
            "strict_noop_correct": strict_noop_correct, "noop_false_positive": noop_false_positive,
        })
    require(sum(row["protocol_valid"] for row in rows) == 70, "protocol-valid denominator mismatch")
    require(sum(row["edit_exact"] for row in rows) == 27, "edit-exact denominator mismatch")
    require(sum(row["strict_noop_correct"] for row in rows) == 26, "strict no-op denominator mismatch")
    require(sum(row["noop_false_positive"] for row in rows) == 4, "no-op false-positive denominator mismatch")
    with (args.output / "replay-inputs.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": "sepalith.notebook-full15006-dev240-syntax-input.v1",
        "status": "materialized_not_parsed",
        "panel_sha256": PANEL_SHA, "result_sha256": RESULT_SHA,
        "reviewed_prior_packet_manifest_sha256": PRIOR_MANIFEST_SHA,
        "rows": 75, "protocol_valid": 70, "protocol_invalid": 5,
        "raw_wire_parsed_context_aware": True, "decoded_body_text_used": False,
        "baseline_content_hash_exact": 75, "prompt_render_hash_exact": 75,
        "generated_r_executed": False,
        "remote_root": str(args.remote_root),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, kind, subset in (
        ("parse-baseline.tsv", "baseline", rows),
        ("parse-full15006-step240.tsv", "applied", [row for row in rows if row["protocol_valid"]]),
    ):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["kind", "arm", "case_index", "id", "family", "path", "sha256"])
            for row in subset:
                relative = row["baseline_relative_path"] if kind == "baseline" else row["applied_relative_path"]
                digest = row["baseline_sha256"] if kind == "baseline" else row["applied_sha256"]
                writer.writerow([kind, row["arm"], row["case_index"], row["id"], row["family"],
                                 str(args.remote_root / "packet" / relative), digest])
    payload_files = [path for path in sorted(args.output.rglob("*"))
                     if path.is_file() and path.name != "payload.sha256"]
    with (args.output / "payload.sha256").open("w", encoding="utf-8") as stream:
        for path in payload_files:
            stream.write(f"{sha(path)}  {path.relative_to(args.output).as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
