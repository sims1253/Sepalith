#!/usr/bin/env python3
"""Independent streaming verification for RL11 expanded context artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
sys.path.insert(0, str(EXEC / "packages/sepalith/src"))
from sepalith.campaign_protocol import PromptContext, render_prompt  # noqa: E402

ROWS = PLAN / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl"
PROVENANCE = PLAN / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl"
LEGACY = PLAN / "docs/campaign/work/corrected-rl-admission-audit-v1/candidate-data/context-sidecar.jsonl"
OUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-context-v1")
SIDECAR = OUT / "context-sidecar.jsonl"
IDS = OUT / "selected-train-ids.json"
REPORT = OUT / "materialization.json"
VERIFICATION = PLAN / "docs/campaign/work/lead/r2-expanded-rl-context-v1/independent-verification.json"
FORBIDDEN = {
    "reward", "score", "advantage", "return", "target", "target_text",
    "target_body", "target_tokens", "target_terminal_tokens", "region_new",
    "model_target", "corpus_target", "teacher", "generated", "completion",
    "gold", "reference_answer", "reference_target",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leak(value: Any, path: str = "context") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN:
                return f"{path}.{key}"
            found = leak(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = leak(child, f"{path}[{index}]")
            if found:
                return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(REPORT.read_text())
    expected_sidecar = report["outputs"]["context_sidecar"]
    expected_ids = report["outputs"]["selected_train_ids"]
    assert sha_file(ROWS) == report["inputs"]["expanded_rows"]["sha256"]
    assert sha_file(PROVENANCE) == report["inputs"]["expanded_provenance"]["sha256"]
    assert sha_file(SIDECAR) == expected_sidecar["sha256"]
    assert sha_file(IDS) == expected_ids["sha256"]

    legacy_raw: dict[str, str] = {}
    with LEGACY.open("rb") as handle:
        for raw in handle:
            value = json.loads(raw)
            legacy_raw[value["row_id"]] = hashlib.sha256(raw).hexdigest()

    counts: Counter[str] = Counter()
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    with ROWS.open("rb") as rows_handle, PROVENANCE.open("rb") as provenance_handle, SIDECAR.open("rb") as sidecar_handle:
        while True:
            row_raw = rows_handle.readline()
            provenance_raw = provenance_handle.readline()
            sidecar_raw = sidecar_handle.readline()
            if not row_raw:
                assert not provenance_raw and not sidecar_raw
                break
            assert provenance_raw and sidecar_raw
            row = json.loads(row_raw)
            provenance = json.loads(provenance_raw)
            sidecar = json.loads(sidecar_raw)
            row_id = row["id"]
            assert provenance["id"] == row_id == sidecar["row_id"]
            assert row_id not in seen_ids
            seen_ids.add(row_id)
            ordered_ids.append(row_id)
            assert row["split"] == "train"
            assert provenance["global_split"] == "train_group"
            assert provenance.get("cpt_partition") != "cpt_validation"
            assert len(row["input_ids"]) <= 4096
            assert len(row["input_ids"]) - row["target_start"] <= 1024
            assert provenance["prompt_sha256"] not in seen_prompts
            seen_prompts.add(provenance["prompt_sha256"])
            assert sidecar["split"] == "train"
            assert sidecar["prompt_sha256"] == provenance["prompt_sha256"]
            context_mapping = sidecar["context"]
            assert leak(context_mapping) is None
            context = PromptContext.from_mapping(context_mapping)
            assert render_prompt(context) == row["prompt_text"]
            geometry = sidecar["selection_geometry"]
            replacement = context.replacement_range
            assert geometry["document_sha256"] == replacement.content_sha256
            assert geometry["context_range"] == replacement.to_dict()
            assert geometry["overflow"] is False and geometry["required_overflow"] is False
            region_spans = [span for span in geometry["spans"] if span.get("kind") == "region"]
            assert len(region_spans) == 1
            assert region_spans[0]["start_line"] == replacement.start.line
            assert region_spans[0]["end_line"] == replacement.end.line
            if row_id in legacy_raw:
                assert hashlib.sha256(sidecar_raw).hexdigest() == legacy_raw[row_id]
                counts["legacy_byte_exact"] += 1
            else:
                source = sidecar.get("expanded_context_source")
                assert source in {"DAT05_candidate_envelope_revalidated", "accepted_short_packet_revalidated"}
                counts[source] += 1
            counts[f"family:{row['family']}"] += 1

    ids_value = json.loads(IDS.read_text())
    assert ids_value["split"] == "train"
    assert ids_value["row_ids"] == ordered_ids
    assert len(ordered_ids) == len(seen_ids) == 11505
    assert counts["legacy_byte_exact"] == 7910
    assert counts["DAT05_candidate_envelope_revalidated"] == 3184
    assert counts["accepted_short_packet_revalidated"] == 411
    assert len(set(legacy_raw) - seen_ids) == 336

    result = {
        "schema": "RL11-expanded-context-independent-verification-v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "rows": len(ordered_ids),
        "distinct_ids": len(seen_ids),
        "sidecar_sha256": expected_sidecar["sha256"],
        "selected_ids_sha256": expected_ids["sha256"],
        "source_counts": {key: value for key, value in sorted(counts.items()) if not key.startswith("family:")},
        "family_counts": {key.removeprefix("family:"): value for key, value in sorted(counts.items()) if key.startswith("family:")},
        "checks": {
            "expanded_rows_provenance_sidecar_exact_order_join": True,
            "selected_ids_exact_order": True,
            "all_train_and_not_cpt_validation": True,
            "no_duplicate_ids_or_prompts": True,
            "all_complete_rows_within_4096_1024": True,
            "render_prompt_exact_for_every_row": True,
            "replacement_range_geometry_exact_for_every_row": True,
            "recursive_context_target_reward_leak_scan": True,
            "legacy_overlap_raw_line_bytes_exact": True,
            "output_full_file_hashes_match_materialization": True,
        },
        "scope": {
            "cpu_only": True,
            "model_framework_or_weights": False,
            "dev_or_final_content": False,
            "training_or_launch": False,
        },
    }
    if args.output is not None:
        assert not args.output.exists()
        args.output.write_bytes(canonical(result) + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
