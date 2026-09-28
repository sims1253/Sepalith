#!/usr/bin/env python3
"""Verify the RL-01 context-only theta0 fixture fingerprints.

This reads only the hashed RL-02 rows/sidecar and the pinned tokenizer. It
does not construct training rows, call a model, import CUDA, or assign labels
to counterfactual contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object, *, separators=(',', ':')) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=separators).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ids_sha256(value: list[int]) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_selected(path: Path, expected_sha: str) -> list[str]:
    if file_sha256(path) != expected_sha:
        raise ValueError(f"selected-ID file hash mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "sepalith.prm07.selected-train-ids.v1" or value.get("split") != "train":
        raise ValueError("selected-ID schema or split mismatch")
    ids = value.get("row_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("selected IDs are not a unique nonempty list")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912"),
        help="checkout containing the pinned campaign_protocol.py",
    )
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    inputs = fixture["inputs"]
    rows_path = Path(inputs["rows_path"])
    sidecar_path = Path(inputs["sidecar_path"])
    selected_path = Path(inputs["selected_ids_path"])
    for path, expected in ((rows_path, inputs["rows_sha256"]),
                           (sidecar_path, inputs["sidecar_sha256"])):
        if file_sha256(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    selected_ids = read_selected(selected_path, inputs["selected_ids_sha256"])
    if json_sha256(selected_ids, separators=(",", ":")) != inputs["ordered_ids_sha256"]:
        raise ValueError("ordered selected-ID hash mismatch")

    base_ids = [row["id"] for row in fixture["base_rows"]]
    needed = set(base_ids)
    for variant in fixture["variant_derivation"]["variants"]:
        needed.update(variant["source_row_ids"])
    rows = {}
    sidecars = {}
    for line in rows_path.open(encoding="utf-8"):
        value = json.loads(line)
        if value.get("id") in needed:
            rows[value["id"]] = value
    for line in sidecar_path.open(encoding="utf-8"):
        value = json.loads(line)
        if value.get("row_id") in needed:
            sidecars[value["row_id"]] = value
    if set(rows) != needed or set(sidecars) != needed:
        raise ValueError("fixture IDs are missing from hashed RL-02 inputs")

    for row in fixture["base_rows"]:
        row_id = row["id"]
        source = sidecars[row_id]["source_identity"]["source_ref"]
        expected_source = fixture["source_identity_by_row"][row_id]
        for source_key, expected_key in (("source_sha256", "source_sha256"),
                                         ("raw_line_sha256", "raw_line_sha256"),
                                         ("file", "source_file"),
                                         ("line", "source_line")):
            if source.get(source_key) != expected_source[expected_key]:
                raise ValueError(f"source identity mismatch for {row_id}: {source_key}")
        if sidecars[row_id]["source_identity"]["candidate_file_sha256"] != expected_source["candidate_file_sha256"]:
            raise ValueError(f"candidate source hash mismatch for {row_id}")

    protocol_path = args.execution_root / fixture["contract"]["protocol_path"]
    if file_sha256(protocol_path) != fixture["contract"]["protocol_sha256"]:
        raise ValueError(f"pinned protocol hash mismatch: {protocol_path}")
    sys.path.insert(0, str(args.execution_root / "packages" / "sepalith" / "src"))
    from transformers import AutoTokenizer
    from sepalith.campaign_protocol import PromptContext, encode_prompt, render_prompt

    tokenizer_path = fixture["contract"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True, trust_remote_code=False,
    )
    donors = fixture["variant_derivation"]["donor_cycle"]
    forbidden = {"target", "target_text", "target_body", "target_operation", "region_new", "reward"}
    checked = []
    for variant in fixture["variant_derivation"]["variants"]:
        base_id = variant["base_row_id"]
        context = json.loads(json.dumps(sidecars[base_id]["context"], ensure_ascii=False))
        donor = variant.get("donor_row_id")
        kind = variant["kind"]
        if kind in ("missing_history", "strict_noop_control", "local_only_control"):
            context["history"] = []
        elif kind == "irrelevant_history":
            if donor != donors.get(base_id) or donor not in sidecars:
                raise ValueError(f"invalid history donor for {variant['id']}")
            context["history"] = json.loads(json.dumps(sidecars[donor]["context"]["history"], ensure_ascii=False))
        elif kind != "evidence_supported":
            raise ValueError(f"unknown fixture variant kind: {kind}")

        def visit(value: object) -> None:
            if isinstance(value, dict):
                if forbidden.intersection(value):
                    raise ValueError(f"target/reward key entered context for {variant['id']}")
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(context)
        prompt_context = PromptContext.from_mapping(context)
        prompt = render_prompt(prompt_context)
        prompt_ids = encode_prompt(prompt_context, tokenizer)
        if json_sha256(prompt_context.to_dict()) != variant["context_sha256"]:
            raise ValueError(f"context hash mismatch for {variant['id']}")
        if text_sha256(prompt) != variant["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch for {variant['id']}")
        if ids_sha256(prompt_ids) != variant["prompt_ids_sha256"]:
            raise ValueError(f"token-ID hash mismatch for {variant['id']}")
        if len(prompt_ids) != variant["prompt_tokens_with_bos"] or len(prompt_ids) > 2048:
            raise ValueError(f"prompt geometry mismatch for {variant['id']}")
        if not prompt.endswith("\n") or len(prompt_ids) + 192 > 2240:
            raise ValueError(f"prompt boundary/cap mismatch for {variant['id']}")
        checked.append(variant["id"])
    print(json.dumps({
        "status": "verified",
        "variants": len(checked),
        "base_rows": len(base_ids),
        "selected_ids": len(selected_ids),
        "prompt_tokens_with_bos": {
            "min": min(item["prompt_tokens_with_bos"] for item in fixture["variant_derivation"]["variants"]),
            "max": max(item["prompt_tokens_with_bos"] for item in fixture["variant_derivation"]["variants"]),
        },
        "model_or_cuda_loaded": False,
        "target_or_reward_context_keys": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
