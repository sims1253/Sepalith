#!/usr/bin/env python3
"""Materialize source-supported context candidates for the frozen 10,017 IDs.

This is a CPU-only, review-only pass.  It reconstructs each complete
normalized TRAIN before-state in memory, selects deterministic AST-backed
source spans, and emits token IDs plus provenance metadata.  Source, prompt,
and target strings are never serialized.  Rows whose context or target is
long remain in explicit queues instead of being cropped or discarded.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Set process policy before importing the historical training modules.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
OUT_DEFAULT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1")
PACKET_ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1")
PACKET_REL = "structured-materialization-v1/candidate-packets.jsonl"
TOKEN_AUDIT_REL = "structured-materialization-v1/token-audit/candidate-token-rows.jsonl"
ROOT_IDS = PLAN / "docs/campaign/work/lead/r2-roxy10017-root-context-review-v1/candidate-ids.json"
ROOT_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-roxy10017-root-context-review.json"
SUPPORT_LEDGER = PLAN / "docs/campaign/work/lead/r2-source-walk-support-review-v1/support-ledger.jsonl"
SUPPORT_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-source-walk-support-review.json"
REPAIR_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-source-walk-support-repair-preparation.json"
FULL_TOKEN_PROFILES = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-token-profiles.jsonl")
FULL_TOKEN_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-full-file-token-rows.jsonl")
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json")
TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
SCENARIO_SOURCE = Path("/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl")
SCENARIO_SHA = "0d70ccc40a716a8a27cb508e39a16c9a5119a9bc9d0b5b8ee8362aa4b67f6193"
POLICY_PATH = PLAN / "docs/campaign/work/lead/r2-roxy-supported-context-v1/context_selector.py"

TRAINING_DIR = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training")
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module_missing:" + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREP = load_module(
    "dat10_roxy_context_prepare_repair",
    PLAN / "docs/campaign/work/lead/r2-source-walk-support-repair-v1/prepare_repair.py",
)
SELECTOR = load_module("dat10_roxy_context_selector", POLICY_PATH)

import campaign_structured_batch as structured_batch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tree_sitter import Language, Parser  # noqa: E402
import tree_sitter_r  # noqa: E402

review = PREP.review
adapter = PREP.adapter
token_audit = PREP.token_audit
# Keep the Language object alive for the lifetime of the parser.  This avoids
# the tree-sitter-r C binding's known use-after-GC crash during repeated
# source parses in a long materialization pass.
R_LANGUAGE = Language(tree_sitter_r.language())
R_PARSER = Parser(R_LANGUAGE)


class DirectPinnedTokenizer:
    def __init__(self, backend: Tokenizer) -> None:
        self.backend = backend

    def encode(self, text: str, *, add_special_tokens: bool = False,
               split_special_tokens: bool = True) -> list[int]:
        if add_special_tokens is not False or split_special_tokens is not True:
            raise ValueError("tokenizer_policy_violation")
        return self.backend.encode(text, add_special_tokens=False).ids


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class AtomicJsonl:
    """Stream a large candidate file and rename it only after fsync."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        self.stream = self.temporary.open("w", encoding="utf-8")
        self.rows = 0

    def write(self, value: dict[str, Any]) -> None:
        self.stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        self.rows += 1

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.temporary.replace(self.path)


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    root_receipt = json.loads(ROOT_RECEIPT.read_text(encoding="utf-8"))
    candidate_ids = set(json.loads(ROOT_IDS.read_text(encoding="utf-8")))
    if root_receipt.get("candidate_count") != len(candidate_ids) or len(candidate_ids) != 10017:
        raise ValueError("root_candidate_scope_mismatch")

    ledgers: dict[str, dict[str, Any]] = {}
    with SUPPORT_LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_id = str(row.get("row_id"))
            if row_id in candidate_ids:
                if row.get("status") not in ("supported_source_replay_pending_root_admission", "repair_required"):
                    raise ValueError("unsupported_frozen_status:" + row_id)
                if row.get("source_file") != str(SCENARIO_SOURCE):
                    raise ValueError("non_train_scenario_source:" + row_id)
                if not str(row.get("source_path", "")).startswith("/mnt/h/sepalith/normalized/"):
                    raise ValueError("non_normalized_train_source:" + row_id)
                ledgers[row_id] = row
    if set(ledgers) != candidate_ids:
        raise ValueError("support_ledger_scope_incomplete")

    packets: dict[str, dict[str, Any]] = {}
    packet_pins: list[dict[str, Any]] = []
    for shard_no in range(5):
        path = PACKET_ROOT / f"shard-{shard_no:04d}" / PACKET_REL
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                packet = json.loads(line)
                count += 1
                row_id = str(packet["row_ref"]["row_id"])
                if row_id in candidate_ids:
                    if row_id in packets:
                        raise ValueError("duplicate_packet:" + row_id)
                    packets[row_id] = packet
        packet_pins.append({"path": str(path), "rows": count, "sha256": sha_file(path)})
    if set(packets) != candidate_ids:
        raise ValueError("candidate_packet_scope_incomplete")

    # Parse only metadata from the prior full-file profiles.  The old full
    # token rows are hash-pinned but their token arrays are not loaded into
    # memory; this pass creates a new source-span tokenization.
    full_profiles: dict[str, dict[str, Any]] = {}
    with FULL_TOKEN_PROFILES.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            row_id = str(item["row_id"])
            if row_id in candidate_ids:
                full_profiles[row_id] = {
                    "context_profile": item["context_profiles"][-1],
                    "target_body_sha256": item["target_body_sha256"],
                    "target_semantics": item["target_semantics"],
                }
    if set(full_profiles) != candidate_ids:
        raise ValueError("full_profile_scope_incomplete")
    pins = {
        "root_context_receipt": {"path": str(ROOT_RECEIPT), "sha256": sha_file(ROOT_RECEIPT)},
        "root_candidate_ids": {"path": str(ROOT_IDS), "rows": len(candidate_ids), "sha256": sha_file(ROOT_IDS)},
        "support_ledger": {"path": str(SUPPORT_LEDGER), "rows": len(ledgers), "sha256": sha_file(SUPPORT_LEDGER)},
        "support_review_receipt": {"path": str(SUPPORT_RECEIPT), "sha256": sha_file(SUPPORT_RECEIPT)},
        "repair_preparation_receipt": {"path": str(REPAIR_RECEIPT), "sha256": sha_file(REPAIR_RECEIPT)},
        "candidate_packets": packet_pins,
        "full_file_profiles": {"path": str(FULL_TOKEN_PROFILES), "rows": len(full_profiles), "sha256": sha_file(FULL_TOKEN_PROFILES)},
        "full_file_token_rows": {"path": str(FULL_TOKEN_ROWS), "rows": len(candidate_ids), "sha256": sha_file(FULL_TOKEN_ROWS), "consumption": "hash_only"},
        "tokenizer": {"path": str(TOKENIZER), "sha256": sha_file(TOKENIZER)},
        "scenario_source": {"path": str(SCENARIO_SOURCE), "sha256": SCENARIO_SHA, "consumption": "path/hash guard only"},
    }
    if pins["tokenizer"]["sha256"] != TOKENIZER_SHA:
        raise ValueError("tokenizer_hash_mismatch")
    if pins["scenario_source"]["sha256"] != SCENARIO_SHA:
        raise ValueError("scenario_source_pin_mismatch")
    return ledgers, packets, full_profiles, pins


def reconstruct(ledger: dict[str, Any], packet: dict[str, Any]) -> tuple[dict[str, Any], str, int, str, list[str]]:
    raw, package = review.packet_raw(packet)
    source_path = Path(ledger["source_path"])
    source_bytes = source_path.read_bytes()
    if sha_bytes(source_bytes) != ledger["source_sha256"]:
        raise ValueError("normalized_after_source_hash_mismatch")
    source_text = source_bytes.decode("utf-8")
    if "\r" in source_text.replace("\r\n", ""):
        raise ValueError("mixed_or_lone_cr_source")
    eol = "crlf" if "\r\n" in source_text else "lf"
    source_lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    positions = review.locate_inverse_target(source_lines, raw["region_new"], raw["suffix"])
    if len(positions) != 1:
        raise ValueError("normalized_after_target_suffix_not_unique")
    position = positions[0]
    target_line = int(position["target_start_line"])
    removed = len(raw["region_new"]) + int(position["blank_after_target"])
    before_lines = source_lines[:target_line] + [""] + source_lines[target_line + removed:]
    before_text = ("\r\n" if eol == "crlf" else "\n").join(before_lines)
    before_hash = sha_bytes(before_text.encode("utf-8", "surrogatepass"))
    if before_hash != ledger.get("context", {}).get("full_snapshot_sha256"):
        raise ValueError("derived_before_hash_mismatch")
    ref = copy.deepcopy(packet["row_ref"])
    ref.update({
        "package_id": package,
        "source_snapshot_text": before_text,
        "source_snapshot_sha256": before_hash,
        "source_snapshot_provenance_path": "derived:normalized-after/" + str(source_path),
        "cursor_encoding": "scenario_char_offset",
        "workspace_revision_before": before_hash,
        "target_line": target_line,
        "parent_identity": {
            "package": package,
            "path": raw["path"],
            "normalized_after_source_path": str(source_path),
            "normalized_after_source_sha256": ledger["source_sha256"],
            "pre_edit_derivation": "remove_exact_mined_roxygen_target_retain_physical_blank_anchor",
        },
    })
    converted = adapter.convert_structured(raw, ref)
    if converted.get("status") != "converted":
        raise ValueError("structured_conversion_failed:" + str(converted.get("reason")))
    # This validates source hash, UTF-16 geometry, and post-edit identity.
    structured_batch.apply_result(converted)
    return converted, package, target_line, before_hash, source_lines


def length_profile(row: dict[str, Any]) -> dict[str, Any]:
    sequence = len(row["input_ids"])
    return {
        "prompt_with_bos": int(row["target_start"]),
        "prompt_without_bos": int(row["prompt_token_count"]),
        "target_body_tokens": int(row["target_body_token_count"]),
        "target_terminal_tokens": int(row["target_terminal_token_count"]),
        "target_token_count": int(row["target_token_count"]),
        "response_with_terminal_eos": int(row["target_token_count"] + 1),
        "sequence": sequence,
        "target_gt_1024": int(row["target_body_token_count"]) > 1024,
        "context_gt_4096": sequence > 4096,
        "context_gt_8192": sequence > 8192,
        "context_gt_16384": sequence > 16384,
        "context_gt_32768": sequence > 32768,
        "context_gt_131072": sequence > 131072,
    }


_TEXT_KEYS = {"prompt_text", "target_text", "target_body_text"}


def scrub_text(value: Any, key: str | None = None) -> Any:
    if key in _TEXT_KEYS:
        return None
    if isinstance(value, dict):
        return {name: scrub_text(item, name) for name, item in value.items() if name not in _TEXT_KEYS}
    if isinstance(value, list):
        return [scrub_text(item) for item in value]
    return value


def contains_text_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in _TEXT_KEYS or contains_text_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_text_key(item) for item in value)
    return False


def percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]


def common_metadata(ledger: dict[str, Any], packet: dict[str, Any], full_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": str(ledger["row_id"]),
        "family": ledger["family"],
        "group_id": ledger["group_id"],
        "package_id": ledger.get("package_id") or packet["row_ref"].get("package_id"),
        "source_file": ledger["source_file"],
        "source_line": ledger["source_line"],
        "source_path": ledger["source_path"],
        "normalized_after_source_sha256": ledger["source_sha256"],
        "raw_line_sha256": ledger["raw_line_sha256"],
        "original_review_reasons": ledger.get("reasons", []),
        "license_revalidated_by_frozen_review": ledger.get("license", {}).get("reason") is None,
        "license_review_reason": ledger.get("license", {}).get("reason"),
        "full_file_reference": full_profile["context_profile"],
        "admission": "review_only_unadmitted",
        "semantic_support_claim": False,
        "source_text_written": False,
        "target_text_written": False,
    }


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output_must_be_fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    try:
        os.nice(10)
    except OSError:
        pass

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    ledgers, packets, full_profiles, input_pins = load_inputs()
    row_ids = sorted(ledgers)
    if args.max_rows is not None:
        row_ids = row_ids[:args.max_rows]
    by_source: dict[str, list[str]] = defaultdict(list)
    for row_id in row_ids:
        by_source[ledgers[row_id]["source_path"]].append(row_id)
    for rows in by_source.values():
        rows.sort()
    write_json(args.output / "context-selector-policy.json", SELECTOR.policy_document())
    write_json(args.output / "status.json", {
        "schema": "sepalith.dat10.roxy_supported_context_materialization_status.v1",
        "status": "running",
        "started_at": started_at,
        "scope_rows": len(row_ids),
        "full_candidate_scope_rows": len(ledgers),
        "source_files_in_scope": len(by_source),
        "cpu_threads_max": 2,
        "cuda": False,
        "training": False,
        "admission": "review_only_unadmitted",
        "source_text_written": False,
        "target_text_written": False,
        "input_pins": input_pins,
    })
    profile_out = AtomicJsonl(args.output / "selected-context-profiles.jsonl")
    row_out = AtomicJsonl(args.output / "selected-context-token-rows.jsonl")
    error_out = AtomicJsonl(args.output / "context-errors.jsonl")
    long_out = AtomicJsonl(args.output / "long-context-queue.jsonl")
    tokenizer_backend = Tokenizer.from_file(str(TOKENIZER))
    tokenizer_backend.encode_special_tokens = True
    tokenizer = DirectPinnedTokenizer(tokenizer_backend)
    counts = Counter()
    selected_sequences: list[int] = []
    selected_targets: list[int] = []
    source_files_seen = 0
    source_license_cache: dict[str, dict[str, Any]] = {}
    parser_r = R_PARSER
    rows_done = 0

    def checkpoint(status: str = "running") -> None:
        write_json(args.output / "status.json", {
            "schema": "sepalith.dat10.roxy_supported_context_materialization_status.v1",
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scope_rows": len(row_ids),
            "full_candidate_scope_rows": len(ledgers),
            "rows_done": rows_done,
            "profiles_written": profile_out.rows,
            "token_rows_written": row_out.rows,
            "errors_written": error_out.rows,
            "long_queue_rows": long_out.rows,
            "source_files_seen": source_files_seen,
            "counts": dict(counts),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "cpu_threads_max": 2,
            "cuda": False,
            "training": False,
            "admission": "review_only_unadmitted",
            "source_text_written": False,
            "target_text_written": False,
            "input_pins": input_pins,
        })

    for source_path, source_row_ids in sorted(by_source.items()):
        source_files_seen += 1
        source_bytes: bytes | None = None
        source_lines: list[str] | None = None
        source_error: str | None = None
        try:
            source_bytes = Path(source_path).read_bytes()
            if sha_bytes(source_bytes) not in {ledgers[row_id]["source_sha256"] for row_id in source_row_ids}:
                source_error = "normalized_after_source_hash_mismatch"
            else:
                source_text = source_bytes.decode("utf-8")
                if "\r" in source_text.replace("\r\n", ""):
                    source_error = "mixed_or_lone_cr_source"
                source_lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        except (OSError, UnicodeError) as exc:
            source_error = type(exc).__name__ + ":" + str(exc)
        for row_id in source_row_ids:
            ledger = ledgers[row_id]
            packet = packets[row_id]
            full_profile = full_profiles[row_id]
            metadata = common_metadata(ledger, packet, full_profile)
            rows_done += 1
            try:
                license_info = ledger.get("license", {})
                license_path = str(license_info.get("path", ""))
                if license_path not in source_license_cache:
                    # The frozen support ledger stores the expected digest as
                    # ``expected_sha256``; the reviewed helper accepts the
                    # validation packet spelling ``sha256``.
                    expected_license = {"sha256": license_info.get("expected_sha256")}
                    source_license_cache[license_path] = review.license_check(
                        license_path, expected_license, source_license_cache,
                    )
                license_result = source_license_cache[license_path]
                if license_result.get("reason") is not None:
                    raise ValueError("license_revalidation:" + str(license_result["reason"]))
                if source_error is not None:
                    raise ValueError(source_error)
                assert source_bytes is not None and source_lines is not None
                converted, package, target_line, before_hash, _ = reconstruct(ledger, packet)
                target_lines = list(converted["target_body"])
                # The selector operates on the exact reconstructed before
                # state.  Use the conversion's selection source as the byte
                # authority after reconstruction, avoiding source text output.
                source_for_selector = converted["selection_source"].get("text", converted["selection_source"].get("document_text"))
                if source_for_selector is None:
                    source_for_selector = Path(converted["selection_source"]["path"]).read_bytes().decode("utf-8")
                source_for_selector_lines = source_for_selector.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                prefix, suffix, evidence = SELECTOR.select_source_spans(
                    source=source_for_selector.encode("utf-8", "surrogatepass"),
                    source_lines=source_for_selector_lines,
                    anchor_line=target_line,
                    target_lines=target_lines,
                    suffix_lines=list(packet["result"].get("context", {}).get("suffix_lines", [])),
                    parser=parser_r,
                )
                if evidence.get("status") in {"ast_parse_error", "target_function_not_found"}:
                    raise ValueError("context_selector:" + str(evidence.get("status")))
                base_context = token_audit.PromptContext.from_mapping(converted["context"])
                selected_context = replace(base_context, prefix=tuple(prefix), suffix_lines=tuple(suffix))
                row = token_audit.build_training_row(
                    selected_context,
                    operation=converted["operation"],
                    region_new=target_lines,
                    tokenizer=tokenizer,
                    row_id=row_id,
                    family=ledger["family"],
                    package_id=package,
                    split="train",
                )
                lengths = length_profile(row)
                pinned_lengths = full_profile["context_profile"]["lengths"]
                if lengths["target_body_tokens"] != int(pinned_lengths["target_body_tokens"]):
                    raise ValueError("target_token_count_differs_from_pinned_full_file")
                if lengths["target_body_tokens"] != int(full_profile["context_profile"]["lengths"]["target_body_tokens"]):
                    raise ValueError("target_body_profile_mismatch")
                evidence = {
                    **evidence,
                    "context_selector_input_before_sha256": before_hash,
                    "selected_prompt_sha256": sha_bytes(row["prompt_text"].encode("utf-8")),
                    "selected_context_prompt_tokens": lengths["prompt_without_bos"],
                    "selected_context_sequence_tokens": lengths["sequence"],
                    "selected_target_body_tokens": lengths["target_body_tokens"],
                    "full_file_sequence_tokens": int(pinned_lengths["sequence"]),
                    "full_file_prompt_tokens": int(pinned_lengths["prompt_without_bos"]),
                    "full_file_reference_is_prior_profile_only": True,
                }
                profile = {
                    **metadata,
                    "schema": "sepalith.dat10.roxy_supported_context_profile.v1",
                    "derived_before_sha256": before_hash,
                    "derived_before_target_line": target_line,
                    "target_body_line_count": len(target_lines),
                    "target_body_sha256": sha_bytes("\n".join(target_lines).encode("utf-8")),
                    "target_semantics": converted["operation"],
                    "target_rewritten": False,
                    "context_policy_id": SELECTOR.POLICY_ID,
                    "context_evidence": evidence,
                    "lengths": lengths,
                    "full_file_profile_comparison": {
                        "sequence_tokens": int(pinned_lengths["sequence"]),
                        "prompt_without_bos_tokens": int(pinned_lengths["prompt_without_bos"]),
                        "target_body_tokens": int(pinned_lengths["target_body_tokens"]),
                    },
                }
                profile_out.write(profile)
                clean_row = scrub_text(row)
                if contains_text_key(clean_row):
                    raise ValueError("serialized_token_row_contains_text_key")
                token_record = {
                    **metadata,
                    "schema": "sepalith.dat10.roxy_supported_context_token_row.v1",
                    "context_policy_id": SELECTOR.POLICY_ID,
                    "context_evidence": evidence,
                    "token_row": clean_row,
                    "token_ids_written": True,
                    "source_text_written": False,
                    "target_text_written": False,
                }
                row_out.write(token_record)
                counts["materialized"] += 1
                counts["selector_status:" + str(evidence.get("status"))] += 1
                selected_sequences.append(lengths["sequence"])
                selected_targets.append(lengths["target_body_tokens"])
                if lengths["context_gt_4096"]:
                    counts["context_gt_4096"] += 1
                if lengths["context_gt_8192"]:
                    counts["context_gt_8192"] += 1
                if lengths["context_gt_16384"]:
                    counts["context_gt_16384"] += 1
                if lengths["context_gt_32768"]:
                    counts["context_gt_32768"] += 1
                if lengths["context_gt_131072"]:
                    counts["context_gt_131072"] += 1
                if lengths["target_gt_1024"]:
                    counts["target_gt_1024"] += 1
                queue_reasons = []
                if lengths["context_gt_4096"]:
                    queue_reasons.append("context_sequence_gt_4096")
                if lengths["target_gt_1024"]:
                    queue_reasons.append("target_body_gt_1024")
                if lengths["context_gt_131072"]:
                    queue_reasons.append("context_sequence_gt_131072")
                if queue_reasons:
                    long_out.write({
                        "schema": "sepalith.dat10.roxy_supported_context_long_queue.v1",
                        "row_id": row_id,
                        "source_path": ledger["source_path"],
                        "group_id": ledger["group_id"],
                        "queue_reasons": queue_reasons,
                        "selected_lengths": lengths,
                        "full_file_lengths": pinned_lengths,
                        "target_truncation": "none",
                        "admission": "review_only_unadmitted",
                    })
            except Exception as exc:  # Every failed row remains in a named queue.
                counts["errors"] += 1
                error_out.write({
                    **metadata,
                    "schema": "sepalith.dat10.roxy_supported_context_error.v1",
                    "status": "context_selection_pending_repair",
                    "error": type(exc).__name__ + ":" + str(exc),
                    "target_truncation": "none",
                })
            if rows_done == 1 or rows_done % 100 == 0 or rows_done == len(row_ids):
                checkpoint()

    for stream in (profile_out, row_out, error_out, long_out):
        stream.close()
    summary = {
        "schema": "sepalith.dat10.roxy_supported_context_materialization_summary.v1",
        "status": "complete" if counts["errors"] == 0 and profile_out.rows == len(row_ids) else "complete_with_repair_queue",
        "scope_rows_requested": len(row_ids),
        "full_candidate_scope_rows": len(ledgers),
        "profiles_rows": profile_out.rows,
        "token_rows": row_out.rows,
        "error_rows": error_out.rows,
        "long_queue_rows": long_out.rows,
        "source_files": len(by_source),
        "counts": dict(counts),
        "lengths": {
            "sequence_tokens": {
                "min": min(selected_sequences) if selected_sequences else None,
                "median": percentile(selected_sequences, 0.50),
                "p95": percentile(selected_sequences, 0.95),
                "p99": percentile(selected_sequences, 0.99),
                "max": max(selected_sequences) if selected_sequences else None,
            },
            "target_body_tokens": {
                "min": min(selected_targets) if selected_targets else None,
                "median": percentile(selected_targets, 0.50),
                "p95": percentile(selected_targets, 0.95),
                "p99": percentile(selected_targets, 0.99),
                "max": max(selected_targets) if selected_targets else None,
            },
        },
        "input_pins": input_pins,
        "policy_id": SELECTOR.POLICY_ID,
        "cuda": False,
        "training": False,
        "admission": "review_only_unadmitted",
        "source_text_written": False,
        "target_text_written": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.output / "materialization-summary.json", summary)
    manifest = {
        "schema": "sepalith.dat10.roxy_supported_context_manifest.v1",
        "status": summary["status"],
        "scope_rows": len(row_ids),
        "policy": {"path": str(POLICY_PATH), "sha256": sha_file(POLICY_PATH)},
        "input_pins": input_pins,
        "artifacts": [],
    }
    for path in (
        args.output / "context-selector-policy.json",
        args.output / "selected-context-profiles.jsonl",
        args.output / "selected-context-token-rows.jsonl",
        args.output / "context-errors.jsonl",
        args.output / "long-context-queue.jsonl",
        args.output / "materialization-summary.json",
        POLICY_PATH,
        Path(__file__),
    ):
        item = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}
        if path.name == "selected-context-profiles.jsonl": item["rows"] = profile_out.rows
        if path.name == "selected-context-token-rows.jsonl": item["rows"] = row_out.rows
        if path.name == "context-errors.jsonl": item["rows"] = error_out.rows
        if path.name == "long-context-queue.jsonl": item["rows"] = long_out.rows
        manifest["artifacts"].append(item)
    write_json(args.output / "manifest.json", manifest)
    checkpoint(summary["status"])
    if args.max_rows is None:
        receipt = {
            "schema": "sepalith.dat10.roxy-supported-context-preparation.v1",
            "status": summary["status"],
            "decision": "review_only_source_supported_context_candidates",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "constraints": {
                "cpu_threads_max": 2, "cuda": False, "cloud": False,
                "dev_payloads": False, "final_payloads": False, "training": False,
                "source_text_written": False, "target_text_written": False,
            },
            "method": {
                "policy_id": SELECTOR.POLICY_ID,
                "selector": "complete top-level target function definition plus fixed-point same-file referenced functions/expressions",
                "docs_evidence": "roxygen tag/formal-name and same-file backtick-name evidence recorded; prose semantics remain pending",
                "long_context": "all selected targets retained; measured >4096/>131072 context and >1024 target queues",
                "target": "complete region_new passed unchanged; no target truncation or rewrite",
                "shared_identity": "context-selector-policy.json is the deterministic source-span identity for serving/training review",
                "semantic_limit": "AST/anchor preservation is not a semantic-support or admission claim",
            },
            "summary": summary,
            "manifest": {"path": str(args.output / "manifest.json"), "sha256": sha_file(args.output / "manifest.json")},
            "artifacts": manifest["artifacts"],
        }
        write_json(PLAN / "docs/campaign/receipts/DAT-10-roxy-supported-context-preparation.json", receipt)


if __name__ == "__main__":
    main()
