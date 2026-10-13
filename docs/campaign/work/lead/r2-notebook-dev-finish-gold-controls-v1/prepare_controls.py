#!/usr/bin/env python3
"""Materialize immutable finish-block gold and candidate parse controls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEAD = HERE.parent
PANEL = LEAD / "corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
PANEL_SHA = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
BASE = LEAD / "r2-notebook-dev-syntax-v1"
E750 = LEAD / "r2-notebook-e750-dev-syntax-v1"
DEV240 = LEAD / "r2-notebook-full15006-dev240-syntax-v1"
sys.path.insert(0, str(BASE))
from prepare_replay import splice  # noqa: E402

FRAME = b"\n}"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def shab(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--remote-root", type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), "fresh output required")
    require(sha(PANEL) == PANEL_SHA, "panel hash differs")
    panel = load_jsonl(PANEL)
    finish = [(i, row) for i, row in enumerate(panel) if row["family"] == "finish_block"]
    require(len(finish) == 6, "finish denominator differs")
    arms = {
        "E750": E750 / "packet/replay-inputs.jsonl",
        "full15006-step240": DEV240 / "packet/replay-inputs.jsonl",
    }
    arm_rows = {name: {r["id"]: r for r in load_jsonl(path)} for name, path in arms.items()}
    args.output.mkdir(parents=True)
    buffers = args.output / "buffers"
    buffers.mkdir()
    controls = []
    candidates = []
    provenance = []
    for case_index, row in finish:
        cid = row["id"]
        case = buffers / f"{case_index:03d}"
        case.mkdir()
        baseline_src = BASE / f"packet/buffers/{case_index:03d}/baseline.R"
        baseline = baseline_src.read_bytes()
        require(shab(baseline) == row["context"]["replacement_range"]["content_sha256"],
                f"{cid}: baseline hash differs")
        eol = "\r\n" if row["context"]["document_eol"] == "crlf" else "\n"
        gold_text = splice(baseline.decode("utf-8"), row["context"]["replacement_range"],
                           row["target_body_text"], eol)
        gold = gold_text.encode("utf-8")
        correction = row["source_provenance"]["correction"]
        require(shab(gold) == correction["new_post_document_sha256"],
                f"{cid}: gold splice differs from corrected provenance")
        require(correction["new_document_parse_ok"] is True, f"{cid}: corrected gold not source-verified parseable")
        require(row["source_provenance"]["finish_splice"]["outer_closing_brace_in_label"] is True,
                f"{cid}: corrected label does not pin outer brace")
        raw_gold = case / "gold-raw.R"
        framed_gold = case / "gold-frame-plus-brace.R"
        raw_gold.write_bytes(gold)
        framed_gold.write_bytes(gold + FRAME)
        for mode, path in (("raw", raw_gold), ("frame_plus_brace", framed_gold)):
            controls.append({"kind": "gold", "arm": "gold", "case_index": case_index,
                             "id": cid, "family": "finish_block", "mode": mode,
                             "path": path, "sha256": sha(path)})
        provenance.append({
            "case_index": case_index, "id": cid,
            "source_file": row["source_provenance"]["source_identity"]["file"],
            "source_line": row["source_provenance"]["source_identity"]["line"],
            "source_raw_line_sha256": row["source_provenance"]["source_identity"]["raw_line_sha256"],
            "literal_source_splice_verified": True,
            "corrected_gold_parse_ok": True,
            "outer_closing_brace_already_in_label": True,
            "materialization": row["source_provenance"]["materialization"],
            "gold_raw_sha256": shab(gold),
            "gold_plus_frame_sha256": shab(gold + FRAME),
            "additional_frame_source_justified": False,
            "reason": "The frozen correction already appends exactly one proven source outer brace and pins the resulting document hash as parseable; a second brace is only a symmetric negative control.",
        })
        for arm, source in (("E750", E750), ("full15006-step240", DEV240)):
            record = arm_rows[arm][cid]
            require(record["case_index"] == case_index and record["family"] == "finish_block",
                    f"{arm}/{cid}: identity differs")
            candidate = {"arm": arm, "case_index": case_index, "id": cid,
                         "protocol_valid": record["protocol_valid"],
                         "protocol_status": record["protocol_status"],
                         "protocol_reason": record["protocol_reason"], "modes": []}
            if record["protocol_valid"]:
                source_path = source / "packet" / record["applied_relative_path"]
                raw = source_path.read_bytes()
                require(shab(raw) == record["applied_sha256"], f"{arm}/{cid}: applied hash differs")
                for mode, payload in (("raw", raw), ("frame_plus_brace", raw + FRAME)):
                    path = case / f"{arm}-{mode}.R"
                    path.write_bytes(payload)
                    controls.append({"kind": "candidate", "arm": arm, "case_index": case_index,
                                     "id": cid, "family": "finish_block", "mode": mode,
                                     "path": path, "sha256": sha(path)})
                    candidate["modes"].append({"mode": mode, "sha256": sha(path)})
            candidates.append(candidate)
    require(len(controls) == 24, "control row denominator differs")
    with (args.output / "control-inputs.jsonl").open("w") as f:
        for row in controls:
            rel = row["path"].relative_to(args.output)
            out = dict(row)
            out["path"] = str(rel)
            f.write(json.dumps(out, sort_keys=True) + "\n")
    with (args.output / "candidate-availability.jsonl").open("w") as f:
        for row in candidates:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "source-provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    with (args.output / "parse-controls.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["kind", "arm", "case_index", "id", "family", "path", "sha256"])
        for row in controls:
            w.writerow([f"{row['kind']}:{row['mode']}", row["arm"], row["case_index"], row["id"],
                        row["family"], args.remote_root / "packet" / row["path"].relative_to(args.output),
                        row["sha256"]])
    manifest = {
        "schema": "sepalith.run10.dev-finish-gold-controls.input.v1",
        "panel_sha256": PANEL_SHA,
        "finish_cases": 6,
        "control_rows": 24,
        "frame_bytes_hex": FRAME.hex(),
        "gold_raw_rows": 6,
        "gold_framed_rows": 6,
        "E750_candidate_valid": 3,
        "DEV240_candidate_valid": 3,
        "candidate_invalid_retained_separately": 6,
        "frame_source_justified": False,
        "frame_role": "symmetric_negative_control_only",
        "generated_r_executed": False,
        "remote_root": str(args.remote_root),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files = [p for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "payload.sha256"]
    with (args.output / "payload.sha256").open("w") as f:
        for p in files:
            f.write(f"{sha(p)}  {p.relative_to(args.output).as_posix()}\n")


if __name__ == "__main__":
    main()
