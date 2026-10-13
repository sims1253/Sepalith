#!/usr/bin/env python3
"""Bounded CPU audit for the DAT-04/SFT-06 corrected TRAIN token rows.

The audit streams the original and corrected JSONL files once, validates every
corrected row with the pinned protocol, and uses the pinned tokenizer only to
rebuild a small deterministic sample.  It does not load model tensors.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0")

MANIFEST = ROOT / "docs/campaign/work/lead/finish-corrected-train-v1/manifest.json"
ORIGINAL = DATA / "SFT-inputs-v1/train-token-rows.jsonl"
OUTPUT = ROOT / "docs/campaign/work/lead/finish-corrected-train-v1/train-token-rows.jsonl"
OVERLAY = ROOT / "docs/campaign/work/finish-target-overlay-v1/finish-target-overlay.jsonl"
PACKET = DATA / "DAT-04B-completion-batch.jsonl"
MATERIALIZER = ROOT / "docs/campaign/work/lead/materialize_corrected_train_v1.py"
PROTOCOL = EXEC / "packages/sepalith/src/sepalith/campaign_protocol.py"
ROOT_RECEIPT = ROOT / "docs/campaign/receipts/DAT-04-corrected-train-root-materialization.json"
OVERLAY_RECEIPT = ROOT / "docs/campaign/receipts/DAT-04-finish-overlay-lead-review.json"
REPORT = ROOT / "docs/campaign/work/corrected-train-independent-review/audit-report.json"

EXPECTED_INPUT_PINS = {
    str(PACKET): "42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25",
    str(ORIGINAL): "7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6",
    str(OVERLAY): "03cff340de0c2b0e641a0399ddc8de727c39c9aa5e51aeed984c6420117054cf",
    str(TOKENIZER / "tokenizer.json"): "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    str(TOKENIZER / "tokenizer_config.json"): "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
    str(PROTOCOL): "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=chunk_size) as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_manifest(manifest: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    require(manifest.get("status") == "corrected_train_rows_materialized_pending_independent_review",
            f"unexpected materialization status: {manifest.get('status')!r}")
    require(manifest.get("input_pins") == EXPECTED_INPUT_PINS, "manifest input pins differ from supplied pins")
    require(manifest.get("script_sha256") == sha256_file(MATERIALIZER), "materializer hash differs from manifest")
    output = manifest.get("output")
    require(isinstance(output, dict), "manifest output is not an object")
    require(output.get("path") == str(OUTPUT), "manifest output path differs")
    require(output.get("sha256") == "e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe",
            "manifest output hash differs from supplied hash")
    require(output.get("bytes") == 117900118 and output.get("rows") == 11526,
            "manifest output size/count differs from supplied values")
    counts = manifest.get("counts")
    require(counts == {
        "original": 11764,
        "changed": 4051,
        "excluded": 238,
        "unchanged_byte_identical": 7475,
        "noops": 1140,
    }, "manifest population counts differ")

    changed_entries = manifest.get("changed")
    excluded_entries = manifest.get("excluded")
    require(isinstance(changed_entries, list) and isinstance(excluded_entries, list),
            "manifest changed/excluded fields are not arrays")
    require(len(changed_entries) == 4051 and len(excluded_entries) == 238,
            "manifest changed/excluded list lengths differ")
    changed = {}
    excluded = {}
    for entry in changed_entries:
        ident = entry.get("id")
        require(isinstance(ident, str) and ident not in changed, f"duplicate/invalid changed id: {ident!r}")
        changed[ident] = entry
    for entry in excluded_entries:
        ident = entry.get("id")
        require(isinstance(ident, str) and ident not in excluded, f"duplicate/invalid excluded id: {ident!r}")
        excluded[ident] = entry
    require(not set(changed).intersection(excluded), "manifest changed/excluded IDs overlap")
    return changed, excluded


def inspect_approved_receipts(manifest: dict) -> dict:
    root_receipt = load_json(ROOT_RECEIPT)
    require(root_receipt.get("status") == manifest.get("status"), "root receipt status differs")
    require(root_receipt.get("output") == manifest.get("output"), "root receipt output differs")
    require(root_receipt.get("counts") == manifest.get("counts"), "root receipt counts differ")
    geometry_ref = manifest.get("geometry_review")
    require(geometry_ref == "docs/campaign/receipts/DAT-04-finish-overlay-lead-review.json",
            "manifest geometry receipt reference differs")
    overlay_receipt = load_json(OVERLAY_RECEIPT)
    require(overlay_receipt.get("status") == "independent_overlay_integrity_and_actual_application_accepted",
            "overlay lead receipt is not accepted")
    require(overlay_receipt.get("counts") == {
        "repaired_and_parse_verified": 4051,
        "strict_rejected_matched_train": 238,
        "absent_downstream": 711,
        "nonfinish_train_untouched": 7475,
        "original_train_total": 11764,
    }, "overlay lead receipt counts differ")
    return {
        "root_materialization_receipt": {
            "path": str(ROOT_RECEIPT),
            "sha256": sha256_file(ROOT_RECEIPT),
        },
        "overlay_lead_receipt": {
            "path": str(OVERLAY_RECEIPT),
            "sha256": sha256_file(OVERLAY_RECEIPT),
        },
    }


def inspect_overlay(changed: dict[str, dict], excluded: dict[str, dict]) -> tuple[str, dict[str, dict], dict]:
    expected = set(changed) | set(excluded)
    digest = hashlib.sha256()
    matched_ids: set[str] = set()
    overlay_selected: dict[str, dict] = {}
    all_ids: set[str] = set()
    rows = 0
    strict_true = 0
    strict_false = 0
    matched = 0
    unmatched = 0
    with OVERLAY.open("rb", buffering=8 * 1024 * 1024) as handle:
        for line_number, raw in enumerate(handle, 1):
            require(raw.endswith(b"\n"), f"overlay line {line_number} lacks LF")
            digest.update(raw)
            record = json.loads(raw)
            rows += 1
            key = record.get("key", {})
            ident = key.get("original_id")
            require(isinstance(ident, str) and ident not in all_ids,
                    f"overlay duplicate/invalid id at line {line_number}: {ident!r}")
            all_ids.add(ident)
            strict = record.get("eligibility", {}).get("strict_v2")
            if strict:
                strict_true += 1
            else:
                strict_false += 1
            if record.get("lineage", {}).get("matched_train_identity"):
                matched += 1
                matched_ids.add(ident)
            else:
                unmatched += 1
            if ident in expected:
                overlay_selected[ident] = record
                require(record["lineage"]["matched_train_identity"],
                        f"expected matched overlay row is absent from train identity: {ident}")

    require(rows == 5000, f"overlay row count {rows} != 5000")
    require(strict_true == 4051 and strict_false == 949, "overlay eligibility counts differ")
    require(matched == 4289 and unmatched == 711, "overlay train-match counts differ")
    require(matched_ids == expected, "overlay matched IDs differ from manifest changed+excluded IDs")
    require(set(overlay_selected) == expected, "overlay selected ID coverage differs")

    for ident, entry in changed.items():
        record = overlay_selected[ident]
        require(record["eligibility"]["strict_v2"] is True, f"changed overlay row is not strict_v2 eligible: {ident}")
        require(record["lineage"]["sft_line_sha256"] == entry["original_line_sha256"],
                f"changed overlay/SFT line hash mismatch: {ident}")
        require(record["original"]["target_body_sha256"] == entry["old_body_sha256"],
                f"changed overlay/original body hash mismatch: {ident}")
        repaired = record.get("repaired")
        require(isinstance(repaired, dict), f"changed overlay row lacks repair: {ident}")
        require(repaired.get("target_body_sha256") == entry["new_body_sha256"],
                f"changed overlay/repaired body hash mismatch: {ident}")
    for ident, entry in excluded.items():
        record = overlay_selected[ident]
        require(record["eligibility"]["strict_v2"] is False, f"excluded overlay row is strict_v2 eligible: {ident}")
        require(record["eligibility"]["rejection_reasons"] == entry["reasons"],
                f"excluded reason mismatch: {ident}")
        require(entry["reasons"] == ["strict_v2_rejected:target_not_lf_terminated"],
                f"unexpected exclusion reason: {ident}")
        require(record.get("repaired") is None, f"excluded overlay row has a repair: {ident}")

    return digest.hexdigest(), overlay_selected, {
        "rows": rows,
        "strict_v2_eligible": strict_true,
        "strict_v2_rejected": strict_false,
        "matched_train_identity": matched,
        "unmatched_train_identity": unmatched,
    }


def inspect_packet(changed: dict[str, dict], excluded: dict[str, dict], overlay: dict[str, dict], protocol) -> tuple[str, dict[str, dict], dict[str, object], dict]:
    expected = set(changed) | set(excluded)
    sample_ids = sorted(changed)[:8]
    digest = hashlib.sha256()
    packet_ids: set[str] = set()
    context_info: dict[str, dict] = {}
    sample_contexts: dict[str, object] = {}
    rows = 0
    with PACKET.open("rb", buffering=8 * 1024 * 1024) as handle:
        for line_number, raw in enumerate(handle, 1):
            require(raw.endswith(b"\n"), f"packet line {line_number} lacks LF")
            digest.update(raw)
            record = json.loads(raw)
            rows += 1
            row_ref = record.get("row_ref", {})
            ident = row_ref.get("row_id")
            require(isinstance(ident, str) and ident not in packet_ids,
                    f"packet duplicate/invalid id at line {line_number}: {ident!r}")
            packet_ids.add(ident)
            if ident not in expected:
                continue
            require(row_ref.get("split") == "train_group", f"packet row is not train_group: {ident}")
            require(sha256_bytes(raw) == overlay[ident]["key"]["packet_raw_line_sha256"],
                    f"packet raw-line hash mismatch: {ident}")
            context = protocol.PromptContext.from_mapping(record["result"]["context"])
            rendered = protocol.render_prompt(context)
            context_info[ident] = {
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                "rendered_prompt_bytes": len(rendered.encode("utf-8")),
                "replacement_content_sha256": context.replacement_range.content_sha256,
            }
            if ident in sample_ids:
                sample_contexts[ident] = context

    require(rows == 5000, f"packet row count {rows} != 5000")
    require(set(context_info) == expected, "packet context coverage differs from matched overlay IDs")
    require(set(sample_contexts) == set(sample_ids), "sample packet context coverage differs")
    return digest.hexdigest(), context_info, sample_contexts, {
        "rows": rows,
        "matched_contexts": len(context_info),
        "rendered_contexts": len(context_info),
        "bounded_sample_ids": sample_ids,
    }


def compare_rows(changed: dict[str, dict], excluded: dict[str, dict], overlay: dict[str, dict], context_info: dict[str, dict], protocol):
    digest_original = hashlib.sha256()
    digest_output = hashlib.sha256()
    original_ids: set[str] = set()
    output_ids: set[str] = set()
    changed_order: list[str] = []
    excluded_order: list[str] = []
    sample_ids = sorted(changed)[:8]
    sample_old: dict[str, dict] = {}
    sample_new: dict[str, dict] = {}
    original_rows = 0
    output_rows = 0
    original_bytes = 0
    output_bytes = 0
    unchanged_byte_identical = 0
    old_finish_rows = 0
    protocol_validated = 0
    max_total_tokens = 0
    families: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    lengths: Counter[str] = Counter()

    immutable_changed_keys = {
        "id", "target_start", "target_terminal_tokens", "target_terminal_token_count",
        "family", "package_id", "renderer_id", "prompt_text", "target_operation",
        "prompt_token_count", "bos_token_id", "eos_token_id", "tokenizer_revision",
        "tokenizer_json_sha256", "tokenization_policy", "split",
    }
    with ORIGINAL.open("rb", buffering=8 * 1024 * 1024) as old_handle, OUTPUT.open("rb", buffering=8 * 1024 * 1024) as new_handle:
        for old_line_number, old_raw in enumerate(old_handle, 1):
            require(old_raw.endswith(b"\n"), f"original line {old_line_number} lacks LF")
            digest_original.update(old_raw)
            original_bytes += len(old_raw)
            original_rows += 1
            old = json.loads(old_raw)
            ident = old.get("id")
            require(isinstance(ident, str) and ident not in original_ids,
                    f"original duplicate/invalid id at line {old_line_number}: {ident!r}")
            original_ids.add(ident)
            if old.get("family") == "finish_block":
                old_finish_rows += 1
            if ident in excluded:
                excluded_order.append(ident)
                require(old.get("family") == "finish_block" and old.get("split") == "train",
                        f"excluded source row metadata mismatch: {ident}")
                require(sha256_bytes(old_raw) == excluded[ident]["original_line_sha256"],
                        f"excluded source line hash mismatch: {ident}")
                continue

            new_raw = new_handle.readline()
            require(new_raw, f"corrected output ended before original id {ident}")
            require(new_raw.endswith(b"\n"), f"corrected output line for {ident} lacks LF")
            digest_output.update(new_raw)
            output_bytes += len(new_raw)
            output_rows += 1
            new = json.loads(new_raw)
            require(new.get("id") == ident, f"corrected ID/order mismatch: expected {ident}, got {new.get('id')!r}")
            require(ident not in output_ids, f"corrected duplicate id: {ident}")
            output_ids.add(ident)
            try:
                protocol.validate_training_row(new)
            except Exception as error:
                raise AuditError(f"validate_training_row failed for {ident}: {error}") from error
            protocol_validated += 1
            input_ids = new["input_ids"]
            body_ids = new["target_body_tokens"]
            terminal_ids = new["target_terminal_tokens"]
            target_start = new["target_start"]
            target_end = target_start + len(body_ids) + len(terminal_ids)
            require(input_ids[target_start:target_end] == body_ids + terminal_ids,
                    f"body+terminal token suffix reconstruction failed: {ident}")
            require(input_ids[target_end:] == [1], f"protocol EOS suffix reconstruction failed: {ident}")
            require(input_ids.count(0) == 1 and input_ids[0] == 0,
                    f"BOS cardinality/position failed: {ident}")
            require(input_ids.count(1) == 1 and input_ids[-1] == 1,
                    f"EOS cardinality/position failed: {ident}")
            require(len(input_ids) <= 4096, f"4096 ceiling exceeded: {ident}")
            max_total_tokens = max(max_total_tokens, len(input_ids))
            families[new["family"]] += 1
            operations[new["target_operation"]] += 1
            lengths["long" if len(input_ids) > 2048 else "short"] += 1

            if ident in changed:
                changed_order.append(ident)
                entry = changed[ident]
                require(old.get("family") == "finish_block" and old.get("split") == "train",
                        f"changed source row metadata mismatch: {ident}")
                require(sha256_bytes(old_raw) == entry["original_line_sha256"],
                        f"changed source line hash mismatch: {ident}")
                require(sha256_bytes(new_raw) == entry["new_line_sha256"],
                        f"changed corrected line hash mismatch: {ident}")
                require(sha256_bytes(old["target_body_text"].encode("utf-8")) == entry["old_body_sha256"],
                        f"changed old body hash mismatch: {ident}")
                require(sha256_bytes(new["target_body_text"].encode("utf-8")) == entry["new_body_sha256"],
                        f"changed new body hash mismatch: {ident}")
                require(entry["target_tokens_before"] == old["target_token_count"] and
                        entry["target_tokens_after"] == new["target_token_count"],
                        f"changed target token counts differ from manifest: {ident}")
                require(entry["prompt_ids_unchanged"] is True,
                        f"manifest does not assert unchanged prompt IDs: {ident}")
                require(new["target_body_text"] == old["target_body_text"] + "}",
                        f"changed target is not old body plus one ASCII closing brace: {ident}")
                require(new["target_body_text"].encode("utf-8").endswith(b"}"),
                        f"changed target does not terminate in ASCII closing brace: {ident}")
                require(new["prompt_text"] == old["prompt_text"], f"prompt text changed: {ident}")
                require(new["target_start"] == old["target_start"], f"target_start changed: {ident}")
                require(new["input_ids"][:new["target_start"]] == old["input_ids"][:old["target_start"]],
                        f"prompt token IDs changed: {ident}")
                require(new["prompt_token_count"] == old["prompt_token_count"], f"prompt token count changed: {ident}")
                for key in immutable_changed_keys:
                    require(new[key] == old[key], f"immutable changed-row field {key!r} differs: {ident}")
                require(context_info[ident]["rendered_prompt_sha256"] == sha256_bytes(old["prompt_text"].encode("utf-8")),
                        f"rendered packet prompt differs from stored prompt: {ident}")
                require(overlay[ident]["key"]["packet_raw_line_sha256"] == entry["packet_line_sha256"],
                        f"changed packet hash differs from manifest: {ident}")
                repaired = overlay[ident]["repaired"]
                require(repaired["target_body_text"] == new["target_body_text"],
                        f"corrected target differs from approved overlay repair: {ident}")
                require(context_info[ident]["replacement_content_sha256"] ==
                        repaired["strict_v2_evidence"]["replacement_content_sha256"],
                        f"packet replacement context differs from approved repair evidence: {ident}")
                if ident in sample_ids:
                    sample_old[ident] = old
                    sample_new[ident] = new
            else:
                require(ident not in changed and ident not in excluded, f"unlisted changed/excluded ID: {ident}")
                require(new_raw == old_raw, f"non-finish row is not byte-identical: {ident}")
                require(old.get("family") != "finish_block", f"unlisted finish row changed classification: {ident}")
                unchanged_byte_identical += 1

        require(not new_handle.readline(), "corrected output contains extra rows after original stream")

    require(original_rows == 11764 and output_rows == 11526, "original/corrected row counts differ")
    require(output_bytes == 117900118, f"corrected byte count {output_bytes} != 117900118")
    require(old_finish_rows == 4289, f"original finish row count {old_finish_rows} != 4289")
    require(len(changed_order) == 4051 and changed_order == list(changed), "changed order/coverage differs")
    require(len(excluded_order) == 238 and excluded_order == list(excluded), "excluded order/coverage differs")
    require(original_ids == output_ids | set(excluded), "corrected IDs are not original IDs minus exclusions")
    require(not output_ids.intersection(excluded), "excluded ID appeared in corrected output")
    require(unchanged_byte_identical == 7475, f"byte-identical non-finish rows {unchanged_byte_identical} != 7475")
    require(protocol_validated == 11526, f"validated rows {protocol_validated} != 11526")
    require(dict(families) == {
        "rename_propagation": 1578,
        "pipe_rewrite": 2027,
        "na_rm_propagation": 137,
        "format_propagation": 1719,
        "no_op": 1140,
        "roxygen_drafting": 874,
        "finish_block": 4051,
    }, f"family counts differ: {dict(families)!r}")
    require(dict(operations) == {"replace": 10386, "no_op": 1140},
            f"operation counts differ: {dict(operations)!r}")
    require(dict(lengths) == {"short": 10051, "long": 1475}, f"length buckets differ: {dict(lengths)!r}")
    require(max_total_tokens == 3064, f"maximum corrected total tokens {max_total_tokens} != 3064")
    return {
        "original_sha256": digest_original.hexdigest(),
        "corrected_sha256": digest_output.hexdigest(),
        "original_rows": original_rows,
        "corrected_rows": output_rows,
        "original_bytes": original_bytes,
        "corrected_bytes": output_bytes,
        "changed": len(changed_order),
        "excluded": len(excluded_order),
        "unchanged_byte_identical": unchanged_byte_identical,
        "finish_rows": old_finish_rows,
        "protocol_validated": protocol_validated,
        "families": dict(families),
        "operations": dict(operations),
        "length_buckets": dict(lengths),
        "max_total_tokens": max_total_tokens,
        "sample_old": sample_old,
        "sample_new": sample_new,
    }


def rebuild_bounded_sample(sample_old: dict[str, dict], sample_new: dict[str, dict], sample_contexts: dict[str, object], protocol) -> dict:
    sys.path.insert(0, str(EXEC / "packages/sepalith/src"))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    tokenizer_config = load_json(TOKENIZER / "tokenizer_config.json")
    require(tokenizer.get_vocab_size() == 130560,
            f"pinned tokenizer vocabulary length is {tokenizer.get_vocab_size()}")
    require(tokenizer.token_to_id("<s>") == 0 and tokenizer.token_to_id("</s>") == 1,
            "pinned tokenizer BOS/EOS vocabulary IDs differ")
    require(tokenizer_config.get("bos_token") == "<s>" and tokenizer_config.get("eos_token") == "</s>",
            "pinned tokenizer config BOS/EOS tokens differ")

    class DirectPinnedTokenizer:
        """Expose the pinned tokenizer JSON through the protocol encoder seam."""

        def encode(self, text: str, *, add_special_tokens: bool = False,
                   split_special_tokens: bool = True) -> list[int]:
            require(add_special_tokens is False and split_special_tokens is True,
                    "bounded tokenizer invoked outside the pinned encoding policy")
            return tokenizer.encode(text, add_special_tokens=False).ids

    encoder = DirectPinnedTokenizer()
    rebuilt = 0
    parsed = 0
    for ident in sorted(sample_new):
        old = sample_old[ident]
        new = sample_new[ident]
        context = sample_contexts[ident]
        require(protocol.render_prompt(context) == new["prompt_text"],
                f"bounded sample rendered prompt mismatch: {ident}")
        region_new = tuple(new["target_body_text"].split("\n"))
        parsed_output = protocol.parse_output(new["target_text"], context)
        require(parsed_output.status == "accepted" and parsed_output.operation == "replace" and
                parsed_output.body == region_new,
                f"bounded serialized output parse differs: {ident}")
        parsed += 1
        rebuilt_row = protocol.build_training_row(
            context,
            operation="replace",
            region_new=region_new,
            tokenizer=encoder,
            row_id=new["id"],
            family=new["family"],
            package_id=new["package_id"],
            split=new["split"],
        )
        require(rebuilt_row == new, f"pinned tokenizer/protocol rebuild differs: {ident}")
        require(new["prompt_text"] == old["prompt_text"] and
                new["input_ids"][:new["target_start"]] == old["input_ids"][:old["target_start"]],
                f"bounded sample prompt identity differs: {ident}")
        rebuilt += 1
    return {
        "rows_rebuilt": rebuilt,
        "serialized_outputs_parsed": parsed,
        "ids": sorted(sample_new),
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_json_sha256": sha256_file(TOKENIZER / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(TOKENIZER / "tokenizer_config.json"),
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_token_id": tokenizer.token_to_id("<s>"),
        "eos_token_id": tokenizer.token_to_id("</s>"),
        "backend": "tokenizers.Tokenizer.from_file",
        "full_corpus_retokenized": False,
    }


def main() -> None:
    started = time.monotonic()
    manifest = load_json(MANIFEST)
    require(isinstance(manifest, dict), "manifest is not an object")
    changed, excluded = check_manifest(manifest)
    receipt_hashes = inspect_approved_receipts(manifest)

    sys.path.insert(0, str(EXEC / "packages/sepalith/src"))
    from sepalith import campaign_protocol as protocol

    overlay_hash, overlay, overlay_counts = inspect_overlay(changed, excluded)
    packet_hash, context_info, sample_contexts, packet_counts = inspect_packet(
        changed, excluded, overlay, protocol
    )
    row_result = compare_rows(changed, excluded, overlay, context_info, protocol)
    sample_result = rebuild_bounded_sample(
        row_result.pop("sample_old"), row_result.pop("sample_new"), sample_contexts, protocol
    )

    require(row_result["original_sha256"] == EXPECTED_INPUT_PINS[str(ORIGINAL)],
            "original SFT input hash differs from pin")
    require(row_result["corrected_sha256"] == manifest["output"]["sha256"],
            "corrected output hash differs from manifest")
    require(overlay_hash == EXPECTED_INPUT_PINS[str(OVERLAY)], "overlay hash differs from pin")
    require(packet_hash == EXPECTED_INPUT_PINS[str(PACKET)], "packet hash differs from pin")
    source_hashes = {
        str(TOKENIZER / "tokenizer.json"): sha256_file(TOKENIZER / "tokenizer.json"),
        str(TOKENIZER / "tokenizer_config.json"): sha256_file(TOKENIZER / "tokenizer_config.json"),
        str(PROTOCOL): sha256_file(PROTOCOL),
    }
    require(source_hashes == {
        str(TOKENIZER / "tokenizer.json"): EXPECTED_INPUT_PINS[str(TOKENIZER / "tokenizer.json")],
        str(TOKENIZER / "tokenizer_config.json"): EXPECTED_INPUT_PINS[str(TOKENIZER / "tokenizer_config.json")],
        str(PROTOCOL): EXPECTED_INPUT_PINS[str(PROTOCOL)],
    }, "source/tokenizer hashes differ from pins")

    report = {
        "task": "DAT-04/SFT-06",
        "status": "independent_corrected_train_review_accepted",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "cpu_threads": 1,
            "ram_working_budget_mib": 768,
            "gpu": False,
            "model_tensors_loaded": False,
            "network": False,
            "ssh": False,
            "source_or_original_data_edits": False,
            "final_or_dev_opened": False,
            "full_corpus_retokenized": False,
        },
        "artifacts": {
            "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
            "materializer": {"path": str(MATERIALIZER), "sha256": sha256_file(MATERIALIZER)},
            "original_sft": {"path": str(ORIGINAL), "sha256": row_result["original_sha256"], "bytes": row_result["original_bytes"], "rows": row_result["original_rows"]},
            "corrected_sft": {"path": str(OUTPUT), "sha256": row_result["corrected_sha256"], "bytes": row_result["corrected_bytes"], "rows": row_result["corrected_rows"]},
            "overlay": {"path": str(OVERLAY), "sha256": overlay_hash},
            "packet": {"path": str(PACKET), "sha256": packet_hash},
            "protocol": {"path": str(PROTOCOL), "sha256": source_hashes[str(PROTOCOL)]},
            "tokenizer": {
                "path": str(TOKENIZER),
                "tokenizer_json_sha256": source_hashes[str(TOKENIZER / "tokenizer.json")],
                "tokenizer_config_sha256": source_hashes[str(TOKENIZER / "tokenizer_config.json")],
            },
            "approved_receipts": receipt_hashes,
        },
        "counts": {
            "original": row_result["original_rows"],
            "corrected": row_result["corrected_rows"],
            "changed": row_result["changed"],
            "excluded": row_result["excluded"],
            "unchanged_byte_identical": row_result["unchanged_byte_identical"],
            "finish_rows": row_result["finish_rows"],
            "protocol_validated": row_result["protocol_validated"],
            "overlay_rows": overlay_counts,
            "packet_rows": packet_counts,
        },
        "checks": {
            "manifest_and_root_receipt_consistent": True,
            "original_and_corrected_streamed_once": True,
            "id_order_exclusion_reconciliation": True,
            "no_duplicate_original_or_corrected_ids": True,
            "nonfinish_byte_identical": row_result["unchanged_byte_identical"] == 7475,
            "changed_target_is_old_plus_ascii_closing_brace": True,
            "prompt_text_and_prompt_ids_target_start_unchanged": True,
            "body_and_terminal_tokens_reconstruct_input_suffix": True,
            "one_bos_zero_and_one_eos_one": True,
            "no_truncation_and_max_total_tokens": row_result["max_total_tokens"] <= 4096,
            "family_counts": row_result["families"],
            "operation_counts": row_result["operations"],
            "length_buckets": row_result["length_buckets"],
            "bounded_pinned_tokenizer_rebuild": sample_result,
        },
        "limitations": [
            "The pinned tokenizer was rebuilt for eight deterministic changed rows; stored token arrays and protocol invariants were checked for all 11,526 output rows.",
            "No model weights, tensors, generation, semantic R execution, training, or admission decision was performed.",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(REPORT),
        "report_sha256": sha256_file(REPORT),
        "corrected_sha256": row_result["corrected_sha256"],
        "counts": report["counts"],
        "elapsed_seconds": report["elapsed_seconds"],
        "max_rss_KiB": report["max_rss_KiB"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
