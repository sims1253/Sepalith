#!/usr/bin/env python3
"""Small TRAIN-independent structural controls for the conservative NSE audit."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent


def row(row_id: str, target: str, residual: str) -> dict:
    return {
        "row_id": row_id,
        "target_definition_name": target,
        "target_definition_span": [1, 1],
        "residual_noncall_names": [residual],
        "selected_context_tokens": 1,
        "target_body_tokens": 1,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sepalith-nse-test-") as directory:
        root = Path(directory)
        source = root / "fixture.R"
        source.write_text(
            'good <- function(dt, zoom) {\n'
            '  stopifnot(all(colnames(dt) %in% c("group", "value")))\n'
            '  n <- dt[group == zoom, .N]\n'
            '  ggforce::facet_zoom(x = group == zoom)\n'
            '}\n'
            'true_global <- function(dt) {\n'
            '  stopifnot(all(colnames(dt) %in% c("value")))\n'
            '  dt[value > threshold, .N]\n'
            '}\n'
            'name_only <- function() { message("group"); group + 1 }\n'
        )
        payload = {
            "source_groups": [{
                "source_path": str(source),
                "source_sha256": "test-only",
                "rows": [
                    row("positive", "good", "group"),
                    row("true-global", "true_global", "threshold"),
                    row("name-only", "name_only", "group"),
                ],
            }],
        }
        input_path = root / "input.json"
        output_path = root / "output.jsonl"
        input_path.write_text(json.dumps(payload))
        completed = subprocess.run(
            ["Rscript", "--vanilla", str(HERE / "nse_occurrence_audit.R"), str(input_path), str(output_path)],
            text=True, capture_output=True, timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        results = {item["row_id"]: item for item in map(json.loads, output_path.read_text().splitlines())}
        assert results["positive"]["status"] == "recoverable_occurrence_proven_nse"
        assert results["true-global"]["status"] == "hold_unresolved_or_insufficient_nse_evidence"
        assert results["name-only"]["status"] == "hold_unresolved_or_insufficient_nse_evidence"
        evidence = results["positive"]["evidence"]["residuals"]["group"]["occurrence_evidence"]
        assert {item["classification"] for item in evidence} == {
            "data_table_nse_with_local_schema_evidence",
            "facet_zoom_nse_with_same_name_column_proof",
        }
        assert all(len(item["expression_sha256"]) == 64 for item in evidence)
        threshold = results["true-global"]["evidence"]["residuals"]["threshold"]
        assert not threshold["all_occurrences_supported"]
        print("PASS: 3 structural NSE controls")


if __name__ == "__main__":
    main()
