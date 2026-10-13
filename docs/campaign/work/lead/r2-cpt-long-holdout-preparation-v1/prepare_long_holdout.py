#!/usr/bin/env python3
"""Build lossless 8K/16K strata from the frozen TRAIN package holdout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


BOS, EOS, MASK = 0, 1, -100
VALIDATION = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl")
VALIDATION_SHA = "efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8"
RAW_CHUNKS = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-selected-context-rechunk-root-v3/source/raw_cpt_broader.py")
RAW_CHUNKS_SHA = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
VALIDATOR = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-selected-context-rechunk-root-v3/source/campaign_cpt_data.py")
VALIDATOR_SHA = "8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa"
STRATA = {
    "8k": {"context": 8192, "minimum": 8191, "maximum": 16382},
    "16k": {"context": 16384, "minimum": 16383, "maximum": 32766},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reassemble(rows: list[dict]) -> tuple[dict, list[int], list[str]]:
    rows = sorted(rows, key=lambda row: row["chunk_index"])
    first = rows[0]
    fields = ("document_id", "package", "group_id", "cpt_partition", "source_path", "source_sha256", "document_token_count")
    common = {field: first[field] for field in fields}
    if common["cpt_partition"] != "cpt_validation":
        raise ValueError("source row is not the frozen TRAIN validation partition")
    if common["document_id"] != common["source_sha256"]:
        raise ValueError("document/source identity differs")
    payload: list[int] = []
    expected_start = 0
    original_ids = []
    for index, row in enumerate(rows):
        if any(row[field] != common[field] for field in fields):
            raise ValueError(f"{common['document_id']}: provenance differs")
        accepted_ids = (f"{common['document_id']}:{index}", f"{common['document_id']}:ctx8192:{index}", f"{common['document_id']}:ctx16384:{index}")
        if row["chunk_index"] != index or row["row_id"] not in accepted_ids:
            raise ValueError(f"{common['document_id']}: chunk identity differs")
        if row["source_token_start"] != expected_start or row["token_start"] != expected_start:
            raise ValueError(f"{common['document_id']}: source range is not contiguous")
        carry = 0 if index == 0 else 1
        if row["overlap_context_tokens"] != carry:
            raise ValueError(f"{common['document_id']}: carry differs")
        body = row["input_ids"][1 + carry:-1]
        if row["input_ids"][0] != BOS or row["input_ids"][-1] != EOS:
            raise ValueError(f"{common['document_id']}: BOS/EOS differs")
        if row["labels"][:1 + carry] != [MASK] * (1 + carry) or row["labels"][1 + carry:-1] != body:
            raise ValueError(f"{common['document_id']}: labels differ")
        if carry and row["input_ids"][1] != payload[-1]:
            raise ValueError(f"{common['document_id']}: prior-token carry differs")
        end = row["source_token_end"]
        if row["token_end"] != end or end - expected_start != len(body):
            raise ValueError(f"{common['document_id']}: owned payload span differs")
        terminal = end == common["document_token_count"]
        if row["is_document_end"] is not terminal or row["labels"][-1] != (EOS if terminal else MASK):
            raise ValueError(f"{common['document_id']}: terminal label differs")
        payload.extend(body)
        expected_start = end
        original_ids.append(row["row_id"])
    if expected_start != common["document_token_count"] or not rows[-1]["is_document_end"]:
        raise ValueError(f"{common['document_id']}: original document is incomplete")
    return common, payload, original_ids


def token_sha(module, payload: list[int]) -> str:
    return module.canonical_token_stream_sha256(payload)


def prepare(output: Path) -> dict:
    for path, expected in ((VALIDATION, VALIDATION_SHA), (RAW_CHUNKS, RAW_CHUNKS_SHA), (VALIDATOR, VALIDATOR_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    if output.exists():
        raise FileExistsError("fresh output required")
    validator = load_module("long_holdout_validator", VALIDATOR)
    chunker = load_module("long_holdout_chunks", RAW_CHUNKS).chunks
    source_rows = [json.loads(line) for line in VALIDATION.read_text().splitlines() if line]
    source_validation = validator.validate_materialized_rows(source_rows, max_sequence_tokens=2048, require_complete_documents=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[row["document_id"]].append(row)
    documents = []
    for ident, rows in grouped.items():
        common, payload, original_ids = reassemble(rows)
        documents.append((common, payload, original_ids, token_sha(validator, payload)))
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    artifacts = {}
    stats = {}
    selected_ids = set()
    try:
        provenance_path = stage / "document-provenance.jsonl"
        with provenance_path.open("x") as provenance:
            for label, spec in STRATA.items():
                selected = [item for item in documents if spec["minimum"] <= len(item[1]) <= spec["maximum"]]
                if len(selected) < 2:
                    raise ValueError(f"insufficient complete documents for {label} stratum")
                panel_path = stage / f"cpt_validation_{label}.jsonl"
                output_rows = []
                with panel_path.open("x") as stream:
                    for common, payload, original_ids, stream_sha in sorted(selected, key=lambda item: item[0]["document_id"]):
                        if common["document_id"] in selected_ids:
                            raise ValueError("length strata overlap")
                        selected_ids.add(common["document_id"])
                        produced = []
                        for index, chunk in enumerate(chunker(payload, spec["context"])):
                            row = {
                                "schema": 1,
                                "row_id": f"{common['document_id']}:ctx{spec['context']}:{index}",
                                "document_id": common["document_id"],
                                "package": common["package"],
                                "group_id": common["group_id"],
                                "cpt_partition": "cpt_validation",
                                "source_path": common["source_path"],
                                "source_sha256": common["source_sha256"],
                                "chunk_index": index,
                                **chunk,
                            }
                            validator.validate_materialized_row(row, max_sequence_tokens=spec["context"])
                            stream.write(canonical(row) + "\n")
                            output_rows.append(row)
                            produced.append(row)
                        _, rebuilt, _ = reassemble(produced)
                        if rebuilt != payload:
                            raise ValueError("generated panel changed source tokens")
                        provenance.write(canonical({
                            "schema": "sepalith.cpt.long-holdout-document-provenance.v1",
                            "stratum": label,
                            "context_tokens": spec["context"],
                            "document_id": common["document_id"],
                            "package": common["package"],
                            "group_id": common["group_id"],
                            "cpt_partition": common["cpt_partition"],
                            "source_path": common["source_path"],
                            "source_sha256": common["source_sha256"],
                            "source_token_count": len(payload),
                            "token_stream_sha256": stream_sha,
                            "original_2k_row_ids": original_ids,
                            "generated_row_ids": [row["row_id"] for row in produced],
                            "frozen_input_sha256": VALIDATION_SHA,
                        }) + "\n")
                with panel_path.open("rb") as stream:
                    os.fsync(stream.fileno())
                validated = validator.validate_materialized_rows(output_rows, max_sequence_tokens=spec["context"], require_complete_documents=True)
                loss_tokens = sum(sum(token != MASK for token in row["labels"][1:]) for row in output_rows)
                stats[label] = {
                    "context_tokens": spec["context"],
                    "source_length_range": [spec["minimum"], spec["maximum"]],
                    "documents": len(selected),
                    "packages": len({item[0]["package"] for item in selected}),
                    "rows": len(output_rows),
                    "input_tokens": sum(len(row["input_ids"]) for row in output_rows),
                    "loss_tokens": loss_tokens,
                    "payload_tokens": sum(len(item[1]) for item in selected),
                    "terminal_rows": sum(row["is_document_end"] for row in output_rows),
                    "maximum_row_tokens": max(len(row["input_ids"]) for row in output_rows),
                    "validator_documents": validated["documents"],
                    "validator_loss_tokens": validated["loss_tokens"],
                }
                artifacts[panel_path.name] = {"bytes": panel_path.stat().st_size, "sha256": sha256(panel_path)}
            provenance.flush()
            os.fsync(provenance.fileno())
        artifacts[provenance_path.name] = {"bytes": provenance_path.stat().st_size, "sha256": sha256(provenance_path)}
        result = {
            "schema": "sepalith.sft11.cpt-long-holdout-preparation.v1",
            "status": "complete_root_review_required",
            "source": {
                "path": str(VALIDATION), "sha256": VALIDATION_SHA,
                "rows": len(source_rows), "documents": source_validation["documents"],
                "packages": len(source_validation["packages"]), "loss_tokens": source_validation["loss_tokens"],
                "all_documents_complete": True,
            },
            "strata": stats,
            "artifacts": artifacts,
            "pins": {"raw_chunks_sha256": RAW_CHUNKS_SHA, "validator_sha256": VALIDATOR_SHA},
            "guarantees": {
                "retokenized": False, "raw_source_read": False, "cross_document_context": False,
                "source_tokens_exact": True, "single_terminal_eos_per_document": True,
                "prior_token_overlap_masked": True, "partition": "TRAIN-derived cpt_validation",
                "strata_disjoint": True,
            },
        }
        result_path = stage / "manifest.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        with result_path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_fd)
        os.close(directory_fd)
        os.rename(stage, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(parent_fd)
        os.close(parent_fd)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(prepare(args.output)))


if __name__ == "__main__":
    main()
