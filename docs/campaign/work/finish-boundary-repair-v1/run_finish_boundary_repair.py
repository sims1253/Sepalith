#!/usr/bin/env python3
"""CPU-only validation driver for the isolated DAT-04 finish repair.

The driver reads exactly the first five retained converted TRAIN packets and
one newly authored raw-source fixture.  It applies both the old and candidate
targets in memory with the pinned protocol's UTF-16 converter, checks the
actual post-application R parse, and measures the five repaired rows with the
pinned tokenizer.  It never starts a server, loads model weights, or writes a
training/data artifact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PACKET_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl")
TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
LINEAGE_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/DAT-04-finish-training-lineage-audit.json"
CANDIDATE_PATH = HERE / "finish_boundary_repair.py"

PROTOCOL_PATH = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
SCENARIOS_PATH = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py")
V3_PATH = PLAN_ROOT / "docs/campaign/work/final-constructor-families-v3/raw_source_families_v3.py"
FINISH_EXTRACTOR_PATH = EXEC_ROOT / "experiments/synthetic-data/finish_block.py"
BATCH_PATH = EXEC_ROOT / "experiments/training/campaign_completion_batch.py"
ADAPTER_PATH = EXEC_ROOT / "experiments/training/campaign_admission_completion.py"

PACKET_SHA256 = "42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25"
LINEAGE_RECEIPT_SHA256 = "cf9a8a85610bf64ff1afff77d7368f5cd35dbcbb90fe8184884d8a386a428273"
PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"
SCENARIOS_SHA256 = "cccf8ddfff0ae1f64a0113c9612386227df66f9fff4701608c2320bd8eb0250c"
V3_SHA256 = "dca1d2045a5b8354f22cb87df0bcdf95b590f5e462855f95fe67e0a6f84e04b6"
FINISH_EXTRACTOR_SHA256 = "47759356b85eb9b48e0461e9265a9fa5a272df14de87747a63e62304be48a3fb"
BATCH_SHA256 = "a7a989f491f168154a3274b6678595c15b4d434be6e7f346521483e0d76035f4"
ADAPTER_SHA256 = "fafb852f31e5e59f003c3ccc0fb34919752020fb6c231a09d77b3ea158f173cc"
TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_CONFIG_SHA256 = "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b"
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
TOKENIZER_CAP = 4096

EXPECTED_IDS = (
    "2b578e5b4936158eaab12ee3",
    "000b491aed3d64dbf6587c98",
    "0002bc91539cd90ea3089201",
    "0388bd30853cc3ce66901daa",
    "001093150ddd485f0255be66",
)

FINISH_SOURCE = (
    "#' Add values. 😀\n"
    "#' @param x Numeric values.\n"
    "#' @return Numeric values.\n"
    "add_values <- function(x = \"😀\") {\n"
    "  first <- x + 1\n"
    "  second <- first * 2\n"
    "  c(first, second)\n"
    "}"
 ).encode("utf-8")

NESTED_SOURCE = (
    "#' Summarise one value.\n"
    "#' @param x Numeric input.\n"
    "#' @return Numeric result.\n"
    "summarise_value <- function(x = 1) {\n"
    "  if (x > 0) {\n"
    "    value <- x + 1\n"
    "  } else {\n"
    "    value <- 0\n"
    "  }\n"
    "  value\n"
    "}"
).encode("utf-8")


def load_module(name: str, path: Path) -> Any:  # noqa: ANN401
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:  # noqa: ANN401
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(text)


def parser_ok(scenarios: Any, text: str) -> bool:  # noqa: ANN401
    return not scenarios.parser.parse(text.encode("utf-8")).root_node.has_error


def read_five_packets() -> list[dict[str, Any]]:
    """Read exactly five packet lines; the packet SHA is a prior pinned receipt."""

    packets: list[dict[str, Any]] = []
    with PACKET_PATH.open(encoding="utf-8") as handle:
        for _line_number in range(1, 6):
            line = handle.readline()
            if not line:
                raise AssertionError("packet ended before five retained rows")
            packets.append(json.loads(line))
    if tuple(packet["row_ref"]["row_id"] for packet in packets) != EXPECTED_IDS:
        raise AssertionError("the retained five packet IDs changed")
    if any(packet.get("family") != "finish_block" for packet in packets):
        raise AssertionError("sample contains a non-finish packet")
    return packets


def apply_target(candidate: Any, protocol: Any, case: Mapping[str, Any], target_text: str) -> str:  # noqa: ANN401
    result = case["result"]
    context = result["context"]
    selection = result["selection_source"]
    document = selection["document_text"]
    if sha256_text(document) != context["replacement_range"]["content_sha256"]:
        raise AssertionError("range does not bind represented document")
    return candidate.apply_document_replacement(
        document,
        context["replacement_range"],
        target_text,
        region_old=context["region_old"],
        utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
    )


def packet_case(candidate: Any, protocol: Any, scenarios: Any, packet: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    plan = candidate.prepare_repair(packet)
    result = packet["result"]
    original_text = "\n".join(result["target_body"])
    before = result["selection_source"]["document_text"]
    old_after = apply_target(candidate, protocol, packet, original_text)
    repaired_after = candidate.apply_plan(
        packet,
        plan,
        utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
    )
    if plan.repaired_target_text != original_text + "}":
        raise AssertionError("empty packet suffix did not receive exactly one brace")
    if repaired_after != old_after + "}":
        raise AssertionError("repair is not the exact known-framing reconstruction")
    if parser_ok(scenarios, old_after):
        raise AssertionError("unrepaired packet unexpectedly parses")
    if not parser_ok(scenarios, repaired_after):
        raise AssertionError("repaired packet does not parse")
    if result["target_body"] != list(plan.original_target_lines):
        raise AssertionError("candidate mutated the emitted target")
    return {
        "row_id": packet["row_ref"]["row_id"],
        "kind": packet["row_ref"]["kind"],
        "source_variant": packet["source_variant"],
        "source_file": packet["row_ref"]["source_file"],
        "source_line": packet["row_ref"]["source_line"],
        "source_sha256": packet["row_ref"]["source_sha256"],
        "raw_line_sha256": packet["row_ref"]["raw_line_sha256"],
        "represented_document_sha256": sha256_text(before),
        "old_post_sha256": sha256_text(old_after),
        "repaired_post_sha256": sha256_text(repaired_after),
        "old_post_chars": len(old_after),
        "repaired_post_chars": len(repaired_after),
        "old_post_parse_ok": False,
        "repaired_post_parse_ok": True,
        "exact_reconstruction": "old_post + one ASCII outer brace",
        "target_body_lines": len(plan.original_target_lines),
        "target_body_sha256": plan.evidence["target_body_sha256"],
        "repaired_target_sha256": plan.evidence["repaired_target_sha256"],
        "replacement_range": result["context"]["replacement_range"],
        "empty_region": not bool(result["context"]["region_old"]),
        "outer_brace_action": plan.outer_brace_action,
    }


def token_summary(protocol: Any, tokenizer: Any, context_mapping: Mapping[str, Any], lines: Sequence[str], row_id: str, package_id: str) -> dict[str, Any]:  # noqa: ANN401
    context = protocol.PromptContext.from_mapping(context_mapping)
    row = protocol.build_training_row(
        context,
        operation="replace",
        region_new=list(lines),
        tokenizer=tokenizer,
        row_id=row_id,
        family="finish_block",
        package_id=package_id,
        split="train_group",
    )
    protocol.validate_training_row(row)
    return {
        "prompt_token_count": row["prompt_token_count"],
        "target_body_token_count": row["target_body_token_count"],
        "target_terminal_token_count": row["target_terminal_token_count"],
        "target_token_count": row["target_token_count"],
        "input_token_count": len(row["input_ids"]),
        "input_ids_sha256": sha256_json(row["input_ids"]),
        "target_text_sha256": sha256_text(row["target_text"]),
        "target_start": row["target_start"],
        "bos_count": row["input_ids"].count(protocol.BOS_ID),
        "eos_count": row["input_ids"].count(protocol.EOS_ID),
        "truncated": False,
    }


def token_case(protocol: Any, tokenizer: Any, packet: Mapping[str, Any], plan: Any) -> dict[str, Any]:  # noqa: ANN401
    row_ref = packet["row_ref"]
    result = packet["result"]
    original = token_summary(
        protocol,
        tokenizer,
        result["context"],
        result["target_body"],
        row_ref["row_id"],
        row_ref["package"],
    )
    repaired = token_summary(
        protocol,
        tokenizer,
        result["context"],
        plan.repaired_target_lines,
        row_ref["row_id"],
        row_ref["package"],
    )
    if original["input_token_count"] > TOKENIZER_CAP or repaired["input_token_count"] > TOKENIZER_CAP:
        raise AssertionError("one of the five rows exceeded the pinned tokenizer cap")
    return {
        "row_id": row_ref["row_id"],
        "original": original,
        "repaired": repaired,
        "token_count_deltas": {
            key: repaired[key] - original[key]
            for key in ("prompt_token_count", "target_body_token_count", "target_token_count", "input_token_count")
        },
        "ids_changed": original["input_ids_sha256"] != repaired["input_ids_sha256"],
        "within_4096_without_truncation": True,
    }


def raw_case(v3: Any, source: bytes, *, package_id: str, path: str, group_id: str) -> dict[str, Any]:  # noqa: ANN401
    return v3.build_raw_family_case(
        source,
        package_id=package_id,
        path=path,
        group_id=group_id,
        family="finish_block",
        source_sha256=v3.sha256_bytes(source),
    )


def synthetic_case_checks(candidate: Any, protocol: Any, scenarios: Any, v3: Any) -> dict[str, Any]:  # noqa: ANN401
    finish = raw_case(v3, FINISH_SOURCE, package_id="synthetic", path="R/finish.R", group_id="g-finish")
    nested = raw_case(v3, NESTED_SOURCE, package_id="synthetic", path="R/nested.R", group_id="g-nested")
    finish_plan = candidate.prepare_repair(finish)
    nested_plan = candidate.prepare_repair(nested)
    finish_after = candidate.apply_plan(finish, finish_plan, utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
    nested_after = candidate.apply_plan(nested, nested_plan, utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
    finish_source = FINISH_SOURCE.decode("utf-8")
    nested_source = NESTED_SOURCE.decode("utf-8")
    if finish_after != finish_source or nested_after != nested_source:
        raise AssertionError("synthetic repaired document does not equal source")
    if not parser_ok(scenarios, finish_after) or not parser_ok(scenarios, nested_after):
        raise AssertionError("synthetic repaired source does not parse")
    if nested_plan.repaired_target_text.count("}") != nested_plan.original_target_text.count("}") + 1:
        raise AssertionError("nested repair did not add exactly one outer brace")

    # The raw-v3 case has an astral character before the UTF-16-selected line.
    # The pinned converter must map its range back to the exact source bytes.
    utf16_check = {
        "source_sha256": v3.sha256_bytes(FINISH_SOURCE),
        "replacement_utf16_start": finish["result"]["context"]["replacement_range"]["start"]["character"],
        "replacement_utf16_end": finish["result"]["context"]["replacement_range"]["end"]["character"],
        "post_equals_source": finish_after == finish_source,
    }
    if utf16_check["replacement_utf16_end"] <= 33 or not utf16_check["post_equals_source"]:
        raise AssertionError("UTF-16 astral-column repair check failed")

    # A zero-width range is an empty represented region.  This exercises the
    # application helper independently of finish target validation.
    empty_document = "😀header\n"
    empty_range = {
        "start": {"line": 1, "character": 0},
        "end": {"line": 1, "character": 0},
        "content_sha256": v3.sha256_bytes(empty_document.encode("utf-8")),
    }
    empty_after = candidate.apply_document_replacement(
        empty_document,
        empty_range,
        "inserted",
        region_old=[],
        utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
    )
    if empty_after != "😀header\ninserted":
        raise AssertionError("empty-range application changed surrounding bytes")

    # An explicit standalone visible suffix is authoritative.  It suppresses
    # the repair; the candidate never adds a second closing brace.
    suffix_plan = candidate.prepare_repair(finish, visible_suffix_lines=["  }"])
    if suffix_plan.appended_outer_brace or suffix_plan.repaired_target_text != suffix_plan.original_target_text:
        raise AssertionError("visible outer brace was duplicated")
    ambiguous_rejected = False
    try:
        candidate.prepare_repair(finish, visible_suffix_lines=["  body <- 1", "}"])
    except candidate.RepairRejected as exc:
        ambiguous_rejected = exc.code == "visible_suffix_outer_brace_position_ambiguous"
    if not ambiguous_rejected:
        raise AssertionError("ambiguous visible suffix was not rejected")

    other_family = deepcopy(finish)
    other_family["identity"]["family"] = "no_op"
    other_family_rejected = False
    try:
        candidate.prepare_repair(other_family)
    except candidate.RepairRejected as exc:
        other_family_rejected = exc.code == "family_not_finish_block"
    if not other_family_rejected:
        raise AssertionError("non-finish family was not refused")

    empty_target_rejected = False
    empty_target = deepcopy(finish)
    empty_target["result"]["target_body"] = []
    try:
        candidate.prepare_repair(empty_target)
    except candidate.RepairRejected as exc:
        empty_target_rejected = exc.code == "target_body_empty"
    if not empty_target_rejected:
        raise AssertionError("empty finish target was not refused")

    return {
        "finish_signature": {
            "source_sha256": v3.sha256_bytes(FINISH_SOURCE),
            "source_bytes": len(FINISH_SOURCE),
            "post_equals_source": True,
            "post_parse_ok": True,
            "outer_brace_added": finish_plan.appended_outer_brace,
            "target_body_inner_braces_preserved": True,
        },
        "finish_nested_body": {
            "source_sha256": v3.sha256_bytes(NESTED_SOURCE),
            "source_bytes": len(NESTED_SOURCE),
            "post_equals_source": True,
            "post_parse_ok": True,
            "outer_brace_added": nested_plan.appended_outer_brace,
            "inner_brace_count_preserved_plus_one_outer": True,
        },
        "utf16_astral_geometry": utf16_check,
        "empty_range_application": {
            "post_text_sha256": sha256_text(empty_after),
            "post_equals_expected": True,
        },
        "visible_suffix_brace": {
            "action": suffix_plan.outer_brace_action,
            "appended_outer_brace": suffix_plan.appended_outer_brace,
            "target_unchanged": True,
        },
        "ambiguous_suffix_rejected": ambiguous_rejected,
        "non_finish_family_rejected": other_family_rejected,
        "empty_target_rejected": empty_target_rejected,
    }


def build_manifest() -> dict[str, Any]:
    candidate = load_module("dat04_finish_boundary_repair_candidate", CANDIDATE_PATH)
    protocol = load_module("dat04_finish_boundary_repair_protocol", PROTOCOL_PATH)
    scenarios = load_module("dat04_finish_boundary_repair_scenarios", SCENARIOS_PATH)
    v3 = load_module("dat04_finish_boundary_repair_v3", V3_PATH)
    packets = read_five_packets()
    packet_results = [packet_case(candidate, protocol, scenarios, packet) for packet in packets]
    synthetic = synthetic_case_checks(candidate, protocol, scenarios, v3)

    # Tokenizer JSON/configuration are the only model-directory files read.
    # No model configuration, tensor, server, or native binary is loaded.
    from transformers import AutoTokenizer

    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    tokenizer_config = TOKENIZER_DIR / "tokenizer_config.json"
    if sha256_file(tokenizer_json) != TOKENIZER_JSON_SHA256:
        raise AssertionError("tokenizer.json hash mismatch")
    if sha256_file(tokenizer_config) != TOKENIZER_CONFIG_SHA256:
        raise AssertionError("tokenizer_config.json hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if len(tokenizer) != protocol.VOCAB_SIZE or tokenizer.bos_token_id != protocol.BOS_ID or tokenizer.eos_token_id != protocol.EOS_ID or tokenizer.pad_token_id != protocol.EOS_ID:
        raise AssertionError("pinned tokenizer identity mismatch")
    token_results = []
    for packet in packets:
        plan = candidate.prepare_repair(packet)
        token_results.append(token_case(protocol, tokenizer, packet, plan))

    pins = {
        "candidate": {"path": str(CANDIDATE_PATH), "sha256": sha256_file(CANDIDATE_PATH)},
        "driver": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "lineage_receipt": {"path": str(LINEAGE_RECEIPT), "sha256": LINEAGE_RECEIPT_SHA256},
        "packet": {"path": str(PACKET_PATH), "sha256": PACKET_SHA256, "read_lines": 5},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256},
        "canonical_scenarios": {"path": str(SCENARIOS_PATH), "sha256": SCENARIOS_SHA256},
        "raw_family_v3": {"path": str(V3_PATH), "sha256": V3_SHA256},
        "finish_extractor": {"path": str(FINISH_EXTRACTOR_PATH), "sha256": FINISH_EXTRACTOR_SHA256},
        "completion_batch": {"path": str(BATCH_PATH), "sha256": BATCH_SHA256},
        "completion_adapter": {"path": str(ADAPTER_PATH), "sha256": ADAPTER_SHA256},
        "tokenizer": {
            "directory": str(TOKENIZER_DIR),
            "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "vocab_size": protocol.VOCAB_SIZE,
            "bos_id": protocol.BOS_ID,
            "eos_id": protocol.EOS_ID,
            "native_eog_ids": list(protocol.NATIVE_EOG_IDS),
        },
    }
    return {
        "task": "DAT-04",
        "status": "prepared_finish_boundary_repair_candidate_validated",
        "observed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": {
            "owned_paths": [
                "docs/campaign/work/finish-boundary-repair-v1/",
                "docs/campaign/receipts/DAT-04-finish-boundary-repair-preparation.json",
            ],
            "sampled_train_packets": 5,
            "sample_ids": list(EXPECTED_IDS),
            "models_loaded": False,
            "model_tensors_read": False,
            "servers_or_native_launched": False,
            "network_or_ssh_used": False,
            "production_sources_edited": False,
            "final_or_dev_content_read": False,
            "training_or_data_artifacts_written": False,
        },
        "repair_contract": {
            "candidate_only": True,
            "families": ["finish_block"],
            "constructor": "finish_block_v5_prefix",
            "target_convention": "suffix",
            "required_finish_splice": {
                "literal_source_splice_verified": True,
                "outer_closing_brace_in_label": False,
            },
            "decision": "append one ASCII outer brace to the replacement only when context.suffix_lines is empty; use an explicit standalone visible suffix brace unchanged; reject ambiguous/nonempty suffixes",
            "application_semantics": "replace the exact UTF-16 range in the represented pre-edit document; preserve target bytes and nested body braces",
            "arbitrary_response_policy": "refuse non-finish families, non-replace operations, missing provenance, empty targets, and unproved suffix positions",
        },
        "pinned_inputs": pins,
        "train_sample_results": packet_results,
        "tokenization": {
            "policy": protocol.TOKENIZATION_POLICY,
            "rows": token_results,
            "denominator": 5,
            "truncated": 0,
            "native_tokenizer_invocation": "AutoTokenizer local fast backend; exact protocol build_training_row; no truncation",
        },
        "synthetic_checks": synthetic,
        "tests": {
            "driver": "passed",
            "five_train_postdocs_parse_after_repair": 5,
            "five_train_unrepaired_postdocs_parse": 0,
            "five_train_exact_framed_reconstructions": 5,
            "synthetic_finish_source_replays": 2,
            "utf16_astral_geometry": "passed",
            "empty_range_geometry": "passed",
            "nested_body_brace": "passed",
            "visible_suffix_not_duplicated": "passed",
            "non_finish_preserved_by_refusal": "passed",
        },
        "limitations": [
            "The five converted TRAIN packets carry source-derived simulated pre-edit windows, not complete source bytes. Their exact framed reference is the retained packet post-application plus the one constructor-proven outer brace; the raw-v3 synthetic cases provide the direct complete-source equality proof.",
            "The candidate proves the boundary only for packets with the exact finish_block_v5_prefix provenance. It does not establish that all 4346 converted packets have the same geometry or that every visible suffix is represented.",
            "A nonempty suffix is accepted only when its first nonblank line is exactly a standalone closing brace. Other suffix shapes are rejected; no arbitrary response is patched.",
            "Token counts and ID hashes are protocol measurements for five rows only. They are not model quality, generation, or latency evidence.",
            "No production adapter, schema, admission guard, editor integration, model, GPU, server, CLI, SSH, or network path was changed or launched.",
        ],
    }


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, help="create this new receipt after all checks pass")
    args = parser.parse_args(argv)
    manifest = build_manifest()
    if args.receipt is not None:
        write_new(args.receipt, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
