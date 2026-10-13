#!/usr/bin/env python3
"""Materialize exact, target-only TRAIN resource-probe rows without truncation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "752131a0ade6c1bd30248b9663046e027cd8ac9bdb38170a2b680f9dbae6d897"
SELECTED = {
    8192: (
        "1e5a0085e06c276308c15a70",  # rmarkdown, 8189 tokens
        "1db44de20ae09cc8b020fbf8",  # hahmmr, 8183 tokens
    ),
    16384: (
        "1a18a8535d1c5d86f4e55ee5",  # refund, 16322 tokens
        "1bd7e939e07b4618e7ff23c7",  # paws.analytics, 16283 tokens
    ),
    32768: (
        "06f074c4b3db8213a6e23cad",  # BayesTools, 32442 tokens
        "2232d76b5ba95a36a756c329",  # whitebox, 32394 tokens
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_and_convert(outer: dict) -> tuple[dict, dict]:
    token = outer["token_row"]
    ids = token["input_ids"]
    start = token["target_start"]
    body = token["target_body_tokens"]
    terminal = token["target_terminal_tokens"]
    if outer["row_id"] != token["id"]:
        raise ValueError("outer/token row ID mismatch")
    if token["split"] != "train" or outer["family"] != "roxygen_drafting":
        raise ValueError(f"row {outer['row_id']} is not TRAIN roxygen")
    if ids[0] != token["bos_token_id"] or start != token["prompt_token_count"] + 1:
        raise ValueError(f"row {outer['row_id']} prompt/BOS boundary differs")
    if len(body) != token["target_body_token_count"]:
        raise ValueError(f"row {outer['row_id']} target body count differs")
    if len(terminal) != token["target_terminal_token_count"]:
        raise ValueError(f"row {outer['row_id']} terminal count differs")
    if body + terminal != ids[start:-1] or ids[-1] != token["eos_token_id"]:
        raise ValueError(f"row {outer['row_id']} target suffix/EOS parity differs")
    if len(body) + len(terminal) != token["target_token_count"]:
        raise ValueError(f"row {outer['row_id']} target total differs")
    labels = [-100] * start + ids[start:]
    row = {
        "row_id": outer["row_id"],
        "input_ids": ids,
        "labels": labels,
        "attention_mask": [1] * len(ids),
        "supervised_tokens": len(ids) - start,
        "cpt_partition": "not_cpt_task_sft_train_resource_probe_only",
        "fixture_provenance": {
            "source_line": outer["source_line"],
            "source_path": outer["source_path"],
            "raw_line_sha256": outer["raw_line_sha256"],
            "normalized_after_source_sha256": outer["normalized_after_source_sha256"],
            "target_body_sha256": outer["target_body_sha256"],
            "tokenizer_json_sha256": token["tokenizer_json_sha256"],
            "tokenizer_revision": token["tokenizer_revision"],
            "renderer_id": token["renderer_id"],
            "target_start": start,
            "target_body_token_count": len(body),
            "target_terminal_token_count": len(terminal),
            "eos_supervised": True,
        },
    }
    evidence = {
        "row_id": outer["row_id"],
        "package_id": outer["package_id"],
        "source_path": outer["source_path"],
        "source_line": outer["source_line"],
        "sequence_tokens": len(ids),
        "prompt_plus_bos_tokens": start,
        "target_body_tokens": len(body),
        "target_terminal_tokens": len(terminal),
        "supervised_tokens_including_eos": len(ids) - start,
        "split": token["split"],
        "family": token["family"],
        "source_admission": outer["admission"],
        "input_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest(),
    }
    return row, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("source hash differs")
    expected = {row_id for values in SELECTED.values() for row_id in values}
    found: dict[str, dict] = {}
    with args.source.open(encoding="utf-8") as source:
        for line in source:
            outer = json.loads(line)
            if outer.get("row_id") in expected:
                if outer["row_id"] in found:
                    raise ValueError(f"duplicate selected ID {outer['row_id']}")
                found[outer["row_id"]] = outer
    if set(found) != expected:
        raise ValueError(f"selected IDs missing: {sorted(expected - set(found))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for cap, row_ids in SELECTED.items():
        path = args.output_dir / f"train-long-{cap}.jsonl"
        if path.exists():
            raise ValueError(f"fresh output required: {path}")
        converted, evidence = [], []
        for row_id in row_ids:
            row, row_evidence = validate_and_convert(found[row_id])
            if len(row["input_ids"]) > cap or len(row["input_ids"]) <= cap // 2:
                raise ValueError(f"row {row_id} is outside the {cap} cohort")
            converted.append(row); evidence.append(row_evidence)
        with path.open("x", encoding="utf-8") as output:
            for row in converted:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        manifest_rows.append({
            "cap": cap,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows": evidence,
            "minimum_sequence_tokens": min(x["sequence_tokens"] for x in evidence),
            "maximum_sequence_tokens": max(x["sequence_tokens"] for x in evidence),
        })
    manifest = {
        "schema": "sepalith.sft11.full-weight-long-context-fixtures.v1",
        "status": "resource_probe_only_not_training_admission",
        "source": {"path": str(args.source), "sha256": SOURCE_SHA256},
        "selection": "two longest intact TRAIN rows fitting each cap, with different packages per cohort",
        "cohorts": manifest_rows,
        "invariants": [
            "input_ids copied intact from full-file token row",
            "labels mask only tokens before target_start",
            "target body and wire terminal copied intact",
            "final EOS retained and supervised",
            "no padding and no truncation",
        ],
    }
    target = args.output_dir / "fixture-manifest.json"
    if target.exists():
        raise ValueError(f"fresh output required: {target}")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
