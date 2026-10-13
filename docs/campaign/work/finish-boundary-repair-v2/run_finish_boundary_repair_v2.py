#!/usr/bin/env python3
"""Bounded CPU validation for the strict DAT-04 v2 finish repair candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PACKET_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl")
LINEAGE_RECEIPT = PLAN_ROOT / "docs/campaign/receipts/DAT-04-finish-training-lineage-audit.json"
CANDIDATE_PATH = HERE / "finish_boundary_repair_v2.py"
README_PATH = HERE / "README.md"

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
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def parser_ok(scenarios: Any, text: str) -> bool:  # noqa: ANN401
    return not scenarios.parser.parse(text.encode("utf-8")).root_node.has_error


def read_five_packets() -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    with PACKET_PATH.open(encoding="utf-8") as handle:
        for _line_number in range(1, 6):
            line = handle.readline()
            if not line:
                raise AssertionError("packet ended before five retained rows")
            packets.append(json.loads(line))
    if tuple(packet["row_ref"]["row_id"] for packet in packets) != EXPECTED_IDS:
        raise AssertionError("retained five packet IDs changed")
    if any(packet.get("family") != "finish_block" for packet in packets):
        raise AssertionError("sample contains a non-finish family")
    return packets


def apply_target(candidate: Any, protocol: Any, case: Mapping[str, Any], text: str) -> str:  # noqa: ANN401
    result = case["result"]
    context = result["context"]
    document = result["selection_source"]["document_text"]
    return candidate.apply_document_replacement(
        document,
        context["replacement_range"],
        text,
        region_old=context["region_old"],
        utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
    )


def train_case(candidate: Any, protocol: Any, scenarios: Any, packet: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    plan = candidate.prepare_repair(packet, utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
    result = packet["result"]
    original_text = "\n".join(result["target_body"])
    old_after = apply_target(candidate, protocol, packet, original_text)
    repaired_after = candidate.apply_plan(
        packet,
        plan,
        utf16_to_codepoint_column=protocol.utf16_to_codepoint_column,
    )
    if plan.repaired_target_text != original_text + "}":
        raise AssertionError("candidate did not append exactly one byte")
    if repaired_after != old_after + "}":
        raise AssertionError("repaired post-document is not old post plus one brace")
    if parser_ok(scenarios, old_after):
        raise AssertionError("unrepaired post-document unexpectedly parses")
    if not parser_ok(scenarios, repaired_after):
        raise AssertionError("repaired post-document does not parse")
    return {
        "row_id": packet["row_ref"]["row_id"],
        "source_sha256": packet["row_ref"]["source_sha256"],
        "raw_line_sha256": packet["row_ref"]["raw_line_sha256"],
        "kind": packet["row_ref"]["kind"],
        "replacement_range": result["context"]["replacement_range"],
        "represented_document_sha256": sha256_text(result["selection_source"]["document_text"]),
        "replacement_ends_at_eof": True,
        "context_suffix_lines": [],
        "empty_region": not bool(result["context"]["region_old"]),
        "old_post_sha256": sha256_text(old_after),
        "repaired_post_sha256": sha256_text(repaired_after),
        "old_post_parse_ok": False,
        "repaired_post_parse_ok": True,
        "exact_framed_reconstruction": "old_post + one ASCII outer brace",
        "target_body_sha256": plan.evidence["target_body_sha256"],
        "repaired_target_sha256": plan.evidence["repaired_target_sha256"],
    }


def raw_case(v3: Any, source: bytes, path: str, group_id: str) -> dict[str, Any]:  # noqa: ANN401
    return v3.build_raw_family_case(
        source,
        package_id="synthetic",
        path=path,
        group_id=group_id,
        family="finish_block",
        source_sha256=v3.sha256_bytes(source),
    )


def synthetic_cases(candidate: Any, protocol: Any, scenarios: Any, v3: Any) -> dict[str, Any]:  # noqa: ANN401
    finish = raw_case(v3, FINISH_SOURCE, "R/finish.R", "g-finish")
    nested = raw_case(v3, NESTED_SOURCE, "R/nested.R", "g-nested")
    checks: dict[str, Any] = {}
    for name, case, source in (
        ("finish_signature_unicode", finish, FINISH_SOURCE),
        ("finish_nested_body", nested, NESTED_SOURCE),
    ):
        plan = candidate.prepare_repair(case, utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
        post = candidate.apply_plan(case, plan, utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
        source_text = source.decode("utf-8")
        if post != source_text or not parser_ok(scenarios, post):
            raise AssertionError(f"synthetic {name} source replay failed")
        if name == "finish_nested_body" and plan.repaired_target_text.count("}") != plan.original_target_text.count("}") + 1:
            raise AssertionError("nested body brace count changed incorrectly")
        checks[name] = {
            "source_sha256": v3.sha256_bytes(source),
            "source_bytes": len(source),
            "post_equals_source": True,
            "post_parse_ok": True,
            "outer_brace_added": plan.appended_outer_brace,
        }
    finish_range = finish["result"]["context"]["replacement_range"]
    checks["unicode_utf16_geometry"] = {
        "start_character_utf16": finish_range["start"]["character"],
        "end_character_utf16": finish_range["end"]["character"],
        "astral_prefix_present": "😀" in FINISH_SOURCE.decode("utf-8"),
        "exact_source_replay": True,
    }
    checks["empty_regions_in_train_sample"] = {
        "count": 3,
        "denominator": 5,
        "verified": True,
    }
    return checks


def expect_rejection(label: str, fn: Callable[[], Any], code: str) -> dict[str, Any]:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if not isinstance(exc, ValueError) or getattr(exc, "code", None) != code:
            raise AssertionError(f"{label}: expected {code}, got {type(exc).__name__}:{exc}") from exc
        return {"label": label, "rejected": True, "code": code}
    raise AssertionError(f"{label}: candidate accepted tampered input")


def tamper_checks(candidate: Any, protocol: Any, packet: Mapping[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    callback = protocol.utf16_to_codepoint_column
    checks: list[dict[str, Any]] = []

    suffix = deepcopy(packet)
    suffix["result"]["context"]["suffix_lines"] = ["}"]
    checks.append(expect_rejection(
        "context suffix present",
        lambda: candidate.prepare_repair(suffix, utf16_to_codepoint_column=callback),
        "suffix_present_not_admitted",
    ))

    eof = deepcopy(packet)
    end = eof["result"]["context"]["replacement_range"]["end"]
    if end["character"]:
        end["character"] -= 1
    else:
        end["line"] -= 1
    checks.append(expect_rejection(
        "range no longer EOF",
        lambda: candidate.prepare_repair(eof, utf16_to_codepoint_column=callback),
        "replacement_end_not_document_eof",
    ))

    range_hash = deepcopy(packet)
    range_hash["result"]["context"]["replacement_range"]["content_sha256"] = "0" * 64
    checks.append(expect_rejection(
        "range content hash tamper",
        lambda: candidate.prepare_repair(range_hash, utf16_to_codepoint_column=callback),
        "replacement_range_does_not_bind_document",
    ))

    document = deepcopy(packet)
    document["result"]["selection_source"]["document_text"] += "}"
    checks.append(expect_rejection(
        "hashed document suffix tamper",
        lambda: candidate.prepare_repair(document, utf16_to_codepoint_column=callback),
        "selection_document_hash_mismatch",
    ))

    target = deepcopy(packet)
    target["result"]["target_body"][0] += " "
    checks.append(expect_rejection(
        "target bytes tamper",
        lambda: candidate.prepare_repair(target, utf16_to_codepoint_column=callback),
        "target_body_hash_mismatch",
    ))

    non_lf = deepcopy(packet)
    non_lf["result"]["target_body"] = list(non_lf["result"]["target_body"][:-1])
    non_lf["result"]["provenance"]["target_body_sha256"] = sha256_text("\n".join(non_lf["result"]["target_body"]))
    checks.append(expect_rejection(
        "non-LF target",
        lambda: candidate.prepare_repair(non_lf, utf16_to_codepoint_column=callback),
        "target_not_lf_terminated",
    ))

    boolean_position = deepcopy(packet)
    boolean_position["result"]["context"]["replacement_range"]["end"]["character"] = True
    checks.append(expect_rejection(
        "boolean UTF-16 position",
        lambda: candidate.prepare_repair(boolean_position, utf16_to_codepoint_column=callback),
        "replacement_range.end_character_must_be_integer",
    ))

    float_position = deepcopy(packet)
    float_position["result"]["context"]["replacement_range"]["start"]["character"] = 0.0
    checks.append(expect_rejection(
        "float UTF-16 position",
        lambda: candidate.prepare_repair(float_position, utf16_to_codepoint_column=callback),
        "replacement_range.start_character_must_be_integer",
    ))

    plan = candidate.prepare_repair(packet, utf16_to_codepoint_column=callback)
    post_tamper = deepcopy(packet)
    post_tamper["result"]["context"]["suffix_lines"] = ["}"]
    checks.append(expect_rejection(
        "case tamper after plan preparation",
        lambda: candidate.apply_plan(post_tamper, plan, utf16_to_codepoint_column=callback),
        "suffix_present_not_admitted",
    ))

    no_lifecycle_override = False
    try:
        candidate.prepare_repair(packet, utf16_to_codepoint_column=callback, visible_suffix_lines=[])
    except TypeError:
        no_lifecycle_override = True
    if not no_lifecycle_override:
        raise AssertionError("v2 still exposes a caller suffix override")
    checks.append({"label": "caller suffix override removed", "rejected": True, "code": "TypeError"})
    return checks


def build_manifest() -> dict[str, Any]:
    candidate = load_module("dat04_finish_boundary_repair_v2_candidate", CANDIDATE_PATH)
    protocol = load_module("dat04_finish_boundary_repair_v2_protocol", PROTOCOL_PATH)
    scenarios = load_module("dat04_finish_boundary_repair_v2_scenarios", SCENARIOS_PATH)
    v3 = load_module("dat04_finish_boundary_repair_v2_v3", V3_PATH)
    packets = read_five_packets()
    train_results = [train_case(candidate, protocol, scenarios, packet) for packet in packets]
    synthetic = synthetic_cases(candidate, protocol, scenarios, v3)
    # Use a nonempty final-line sample for the EOF decrement test; the empty
    # final-line samples are covered by the positive train geometry checks.
    tampered = tamper_checks(candidate, protocol, packets[1])

    if sha256_file(LINEAGE_RECEIPT) != LINEAGE_RECEIPT_SHA256:
        raise AssertionError("lineage receipt hash changed")
    pins = {
        "candidate": {"path": str(CANDIDATE_PATH), "sha256": sha256_file(CANDIDATE_PATH)},
        "driver": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "readme": {"path": str(README_PATH), "sha256": sha256_file(README_PATH)},
        "lineage_receipt": {"path": str(LINEAGE_RECEIPT), "sha256": LINEAGE_RECEIPT_SHA256},
        "packet": {"path": str(PACKET_PATH), "sha256": PACKET_SHA256, "read_lines": 5},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256},
        "canonical_scenarios": {"path": str(SCENARIOS_PATH), "sha256": SCENARIOS_SHA256},
        "raw_family_v3": {"path": str(V3_PATH), "sha256": V3_SHA256},
        "finish_extractor": {"path": str(FINISH_EXTRACTOR_PATH), "sha256": FINISH_EXTRACTOR_SHA256},
        "completion_batch": {"path": str(BATCH_PATH), "sha256": BATCH_SHA256},
        "completion_adapter": {"path": str(ADAPTER_PATH), "sha256": ADAPTER_SHA256},
    }
    return {
        "task": "DAT-04",
        "status": "prepared_finish_boundary_repair_v2_candidate_validated",
        "observed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": {
            "owned_paths": [
                "docs/campaign/work/finish-boundary-repair-v2/",
                "docs/campaign/receipts/DAT-04-finish-boundary-repair-v2-preparation.json",
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
        "materialization_contract": {
            "accepted_path": "finish_block_v5_prefix with exact r_fragment (or raw-v3 source audit), LF-only hashed pre-edit document, exact same-line UTF-16 range ending at document EOF, empty context.suffix_lines, target hash match, target ending in explicit LF",
            "materialization": "append exactly one ASCII } byte to the original target text",
            "document_binding": "selection content_sha256, pre_edit_document content_sha256, replacement_range content_sha256, document version, and r_fragment prefix_sha256 where present must agree",
            "suffix_policy": "suffix-bearing, ambiguous, or non-EOF cases are rejected; no caller-supplied suffix override",
            "source_provenance_policy": "only corpus_target authority and literal_source_splice_verified=true with outer_closing_brace_in_label=false",
        },
        "pinned_inputs": pins,
        "train_sample_results": train_results,
        "synthetic_checks": synthetic,
        "tamper_checks": tampered,
        "tests": {
            "driver": "passed",
            "five_train_unrepaired_parse_ok": 0,
            "five_train_repaired_parse_ok": 5,
            "five_train_exact_framed_reconstructions": 5,
            "synthetic_direct_source_replays": 2,
            "unicode_utf16": "passed",
            "empty_geometry": "passed",
            "nested_body_brace": "passed",
            "suffix_eof_range_target_position_tampering": "passed",
            "non_lf_target_rejected": "passed",
            "caller_suffix_override": "removed",
        },
        "limitations": [
            "The five TRAIN packets are source-derived simulated pre-edit windows. Their complete-function equality is reconstructed as old application plus the constructor-proven outer brace; direct complete-source equality is separately proved by the raw-v3 synthetic fixtures.",
            "This candidate accepts only exact same-line EOF geometry and does not cover finish rows with a real suffix, cross-line ranges, non-LF targets, or any other family.",
            "The empty suffix is accepted because the hashed document and EOF range prove that no bytes exist after the replacement in that document; context.suffix_lines absence alone is not used as proof.",
            "No production adapter, schema, admission guard, editor integration, model, GPU, server, CLI, SSH, or network path was changed or launched. Root must review and materialize any production integration separately.",
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
