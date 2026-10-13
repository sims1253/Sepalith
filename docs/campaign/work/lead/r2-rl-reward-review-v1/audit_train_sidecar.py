#!/usr/bin/env python3
"""Census only the hash-pinned TRAIN context sidecar; emit no row payload."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


SIDECAR = Path("docs/campaign/work/corrected-rl-admission-audit-v1/candidate-data/context-sidecar.jsonl")
PIN = "265b80762efc9544230f1aba09e492906760e98a34431593fdeb4813d6e169ac"


def main() -> None:
    digest = hashlib.sha256()
    families: Counter[str] = Counter()
    availability: Counter[str] = Counter()
    rows = finish = source_snapshot = finish_document_text = 0
    with SIDECAR.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            value = json.loads(raw)
            if value["split"] != "train":
                raise ValueError("non-TRAIN sidecar row refused")
            rows += 1
            family = value["family"]
            families[family] += 1
            availability[value["selection_geometry"]["availability"]] += 1
            provenance = value["source_identity"]["source_provenance"]
            if family == "finish_block":
                finish += 1
                if isinstance(provenance.get("selection_source", {}).get("document_text"), str):
                    finish_document_text += 1
            elif provenance.get("source_snapshot_path") and provenance.get("source_snapshot_sha256"):
                source_snapshot += 1
    if digest.hexdigest() != PIN:
        raise ValueError("frozen TRAIN context sidecar hash differs")
    result = {
        "schema": "sepalith.rl11.train-context-census.v1",
        "input": {"path": str(SIDECAR), "sha256": PIN},
        "rows": rows,
        "split": "train",
        "family_counts": dict(sorted(families.items())),
        "availability_counts": dict(sorted(availability.items())),
        "finish_completion_prefix_rows": finish,
        "finish_rows_with_selection_document_text": finish_document_text,
        "non_finish_rows_with_source_snapshot_pin": source_snapshot,
        "buffer_interpretation": {
            "full_snapshot": "availability label; reconstruct and hash-check complete source plus history before parse",
            "source_builder_window_finish_block": "completion prefix; baseline parse failure is not a model defect",
            "source_builder_window_other": "do not infer full-buffer parse validity from the prompt window alone",
        },
        "contains_prompt_or_target_payload": False,
    }
    Path("docs/campaign/work/lead/r2-rl-reward-review-v1/train-sidecar-census.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
