#!/usr/bin/env python3
"""Verify finish-block gold/candidate controls from fixed R parse outputs."""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
RESULTS = ROOT / "notebook-results"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def main() -> None:
    controls = read_jsonl(PACKET / "control-inputs.jsonl")
    availability = read_jsonl(PACKET / "candidate-availability.jsonl")
    with (RESULTS / "results-controls.tsv").open(newline="") as f:
        results = list(csv.DictReader(f, delimiter="\t", escapechar="\\"))
    assert len(controls) == len(results) == 24
    keyed = {(r["kind"], r["arm"], int(r["case_index"]), r["id"]): r for r in results}
    joined = []
    for c in controls:
        kind = f"{c['kind']}:{c['mode']}"
        r = keyed[(kind, c["arm"], c["case_index"], c["id"])]
        assert r["family"] == "finish_block" and r["sha256"] == c["sha256"]
        joined.append({**{k: c[k] for k in ("kind", "arm", "case_index", "id", "family", "mode", "sha256")},
                       "parse_ok": r["parse_ok"] == "true", "expressions": int(r["expressions"]) if r["expressions"] else None,
                       "parse_error": r["error"] or None})
    with (ROOT / "parse-results.jsonl").open("w") as f:
        for row in joined: f.write(json.dumps(row, sort_keys=True) + "\n")
    grouped = defaultdict(list)
    for r in joined: grouped[(r["kind"], r["arm"], r["mode"])].append(r)
    summary = {}
    for key, rows in sorted(grouped.items()):
        label = "/".join(key)
        summary[label] = {"rows": len(rows), "parse_ok": sum(r["parse_ok"] for r in rows),
                          "parse_failed": sum(not r["parse_ok"] for r in rows),
                          "failure_ids": [r["id"] for r in rows if not r["parse_ok"]]}
    provenance = json.loads((PACKET / "source-provenance.json").read_text())
    assert len(provenance) == 6
    assert all(p["literal_source_splice_verified"] and p["corrected_gold_parse_ok"] and
               p["outer_closing_brace_already_in_label"] and not p["additional_frame_source_justified"]
               for p in provenance)
    assert summary["gold/gold/raw"]["parse_ok"] == 6
    assert summary["gold/gold/frame_plus_brace"]["parse_ok"] == 0
    assert summary["candidate/E750/raw"]["parse_ok"] == 3
    assert summary["candidate/full15006-step240/raw"]["parse_ok"] == 1
    invalid = [r for r in availability if not r["protocol_valid"]]
    assert len(invalid) == 6
    readout = {
        "schema": "sepalith.run10.dev-finish-gold-controls.readout.v1",
        "status": "complete_parse_only_control",
        "identity": {
            "panel_sha256": "7c144bcd20570b1b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035",
            "finish_case_ids": [p["id"] for p in provenance],
            "gold_splice_hash_matches_corrected_provenance": 6,
            "literal_source_splice_verified": 6,
            "outer_closing_brace_already_in_corrected_label": 6,
            "candidate_valid": {"E750": 3, "full15006-step240": 3},
            "candidate_invalid_retained_unapplied": {"E750": 3, "full15006-step240": 3},
        },
        "frame": {
            "literal_bytes_hex": "0a7d",
            "description": "literal newline then one right brace",
            "source_justified_for_corrected_panel": False,
            "role": "symmetric negative control only",
            "finding": "Frozen correction provenance says one source outer brace was already appended to each label and pins each resulting gold document as parseable. Adding another brace is not a source-supported frame.",
        },
        "summary": summary,
        "candidate_invalid_protocol": invalid,
        "per_case": joined,
        "interpretation": [
            "All six raw gold-applied buffers parse, while all six gold buffers with an additional newline-plus-brace fail. The proposed fixed frame is therefore invalid for this corrected DEV panel.",
            "All three protocol-valid E750 finish buffers parse raw; the extra brace makes all three fail.",
            "One of three protocol-valid DEV240 finish buffers parses raw. The extra brace repairs the two incomplete candidates but breaks the already-complete third candidate, so it cannot define a uniform syntax test.",
            "The raw finish-buffer syntax difference remains supported against the corrected parseable gold controls. This is syntax evidence only and does not by itself establish a general quality regression.",
            "The six protocol-invalid arm/case combinations have no context-approved body to apply and remain separate from parse denominators.",
            "These DEV controls are evaluation evidence only and are not inputs to TRAIN policy or reward artifacts.",
        ],
        "notebook": {
            "host": "m0pad", "address": "192.168.178.40", "r_version": "4.6.1 (2026-06-24)",
            "affinity": [0, 2], "maximum_threads": 2, "timeout_seconds": 300,
            "elapsed_ns": json.loads((RESULTS / "timing-controls.json").read_text())["elapsed_ns"],
            "payload_rehash": "pass", "result_rehash": "pass", "owned_parse_processes_after": 0,
            "remote_owned_prefix_removed": True,
        },
        "safety": {"generated_r_executed": False, "harness": "base::parse(file=..., keep.source=FALSE)",
                   "source_or_eval_used": False, "gpu_cloud_training_state_touched": False,
                   "final_content_accessed": False, "train_artifacts_modified": False},
        "artifacts": {"parse_results_jsonl_sha256": sha(ROOT / "parse-results.jsonl"),
                      "raw_notebook_result_sha256": sha(RESULTS / "results-controls.tsv"),
                      "source_provenance_sha256": sha(PACKET / "source-provenance.json")},
    }
    (ROOT / "independent-readout.json").write_text(json.dumps(readout, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
