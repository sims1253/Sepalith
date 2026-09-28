#!/usr/bin/env python3
"""Build prediction-only RL contexts for the admitted expanded TRAIN pool.

The expanded rows are the selector. Existing corrected-RL sidecar records are
copied byte-for-byte when their IDs overlap. Missing corrected rows are joined
to the admitted DAT-05 provenance and exact candidate envelope. Accepted short
rows are joined to their frozen candidate packets. Target and reward material
must never occur inside ``context``; source evidence remains out of band.

This is a CPU data preparation. It does not import a model framework, load
weights, access development/final contents, or authorize an RL launch.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, BinaryIO, Iterable, Mapping


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PROTOCOL_SRC = EXEC / "packages/sepalith/src"
if str(PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_SRC))

from sepalith.campaign_protocol import (  # noqa: E402
    PromptContext,
    RENDERER_ID,
    render_prompt,
    validate_training_row,
)


OUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-context-v1")
EXPANDED_ROWS = PLAN / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl"
EXPANDED_PROVENANCE = PLAN / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl"
EXPANDED_ADMISSION = PLAN / "docs/campaign/receipts/DAT-10-expanded-data-root-review.json"
LEGACY_SIDECAR = PLAN / "docs/campaign/work/corrected-rl-admission-audit-v1/candidate-data/context-sidecar.jsonl"
DAT05_PROVENANCE = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/provenance.jsonl")
DAT05_REPORT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/report.json")
SHORT_PACKETS = PLAN / "docs/campaign/work/r2-short-task-preparation-v1/candidate-packets.jsonl"

EXPECTED = {
    "expanded_rows": "fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889",
    "expanded_provenance": "317077b17afc38602309361396bb86d8897a6b823c5d7539f10b9733a0ad9a48",
    "legacy_sidecar": "265b80762efc9544230f1aba09e492906760e98a34431593fdeb4813d6e169ac",
    "dat05_provenance": "6883f0691301e79180164920414e4752f53edcfe555c73b1ea0b02dd599d2123",
    "dat05_report": "a0a72e101308ace5b3843c50196f0311134f055635de35a75955b06b324e422d",
    "short_packets": "a301ba4ed8d0355f4a2d3a83404c3bef41ee31b4820b60980af8afe201ebc7d7",
    "protocol": "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
}
EXPECTED_ROWS = 11_505
MAX_SEQUENCE_TOKENS = 4_096
MAX_TARGET_TOKENS = 1_024
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_CONTEXT_KEYS = frozenset({
    "reward", "score", "advantage", "return", "target", "target_text",
    "target_body", "target_tokens", "target_terminal_tokens", "region_new",
    "model_target", "corpus_target", "teacher", "generated", "completion",
    "gold", "reference_answer", "reference_target",
})


class PreparationError(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparationError(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[int, bytes, Mapping[str, Any]]]:
    with path.open("rb", buffering=4 * 1024 * 1024) as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            require(isinstance(value, Mapping), f"jsonl_object_required:{path}:{line_number}")
            yield line_number, raw, value


def forbidden_key(value: Any, path: str = "context") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_CONTEXT_KEYS:
                return f"{path}.{key_text}"
            found = forbidden_key(child, f"{path}.{key_text}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None


def static_hashes(value: Any, key: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            result.update(static_hashes(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            result.update(static_hashes(child, key))
    elif isinstance(value, str) and key.endswith("sha256") and SHA_RE.fullmatch(value):
        result.add(value)
    return result


def validate_row(row: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    row_id = str(row.get("id"))
    validate_training_row(row)
    require(row.get("split") == "train", f"row_not_train:{row_id}")
    require(row.get("renderer_id") == RENDERER_ID, f"renderer_mismatch:{row_id}")
    require(row.get("bos_token_id") == 0 and row.get("eos_token_id") == 1, f"special_tokens:{row_id}")
    require(row.get("tokenizer_json_sha256") == "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
            f"tokenizer_mismatch:{row_id}")
    input_ids = row.get("input_ids")
    target_start = row.get("target_start")
    require(isinstance(input_ids, list) and type(target_start) is int, f"row_geometry_missing:{row_id}")
    require(len(input_ids) <= MAX_SEQUENCE_TOKENS, f"sequence_over_4096:{row_id}")
    require(len(input_ids) - target_start <= MAX_TARGET_TOKENS, f"target_over_1024:{row_id}")
    require(input_ids[0] == 0 and input_ids[-1] == 1, f"terminal_ids_invalid:{row_id}")
    require(provenance.get("id") == row_id, f"expanded_provenance_id:{row_id}")
    require(provenance.get("global_split") == "train_group", f"global_split_not_train:{row_id}")
    require(provenance.get("cpt_partition") != "cpt_validation", f"cpt_validation_leak:{row_id}")
    require(provenance.get("repaired_binding_status") in {"task_train_cpt_train", "task_train_unmapped_cpt"},
            f"binding_not_train:{row_id}")
    require(provenance.get("total_tokens") == len(input_ids), f"total_tokens_mismatch:{row_id}")
    require(provenance.get("target_label_tokens_including_protocol_EOS") == len(input_ids) - target_start,
            f"target_tokens_mismatch:{row_id}")
    require(provenance.get("prompt_sha256") == sha_bytes(str(row["prompt_text"]).encode("utf-8")),
            f"prompt_hash_mismatch:{row_id}")
    require(provenance.get("target_sha256") == sha_bytes(str(row["target_text"]).encode("utf-8")),
            f"target_hash_mismatch:{row_id}")
    require(provenance.get("row_canonical_sha256") == sha_bytes(canonical(row)), f"canonical_row_mismatch:{row_id}")


def validate_context(
    row: Mapping[str, Any], context_mapping: Mapping[str, Any], selection: Mapping[str, Any],
    source_provenance: Mapping[str, Any], *, availability: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = str(row["id"])
    found = forbidden_key(context_mapping)
    require(found is None, f"forbidden_context_key:{row_id}:{found}")
    context = PromptContext.from_mapping(context_mapping)
    require(render_prompt(context) == row["prompt_text"], f"render_prompt_mismatch:{row_id}")
    replacement = context.replacement_range
    content_sha = replacement.content_sha256
    require(selection.get("document_sha256") == content_sha, f"selection_document_hash:{row_id}")
    require(selection.get("overflow") is False and selection.get("required_overflow") is False,
            f"selection_overflow:{row_id}")
    spans = selection.get("spans")
    require(isinstance(spans, list), f"selection_spans_missing:{row_id}")
    region_spans = [item for item in spans if isinstance(item, Mapping) and item.get("kind") == "region"]
    require(len(region_spans) == 1, f"selection_region_span:{row_id}")
    require(region_spans[0].get("start_line") == replacement.start.line, f"selection_start_line:{row_id}")
    require(region_spans[0].get("end_line") == replacement.end.line, f"selection_end_line:{row_id}")
    require(content_sha in static_hashes(source_provenance), f"static_source_hash_unproven:{row_id}")
    require(type(replacement.document_version) is int and replacement.document_version >= 0,
            f"document_version_invalid:{row_id}")
    geometry = {
        "availability": availability or selection.get("availability") or "full_snapshot",
        "document_sha256": content_sha,
        "policy_id": selection.get("policy_id"),
        "policy_id_combined": selection.get("policy_id_combined"),
        "budget_utf16_units": selection.get("budget_utf16_units"),
        "used_utf16_units": selection.get("used_utf16_units"),
        "required_utf16_units": selection.get("required_utf16_units"),
        "overflow": False,
        "required_overflow": False,
        "spans": spans,
        "region": selection.get("region"),
        "context_range": replacement.to_dict(),
        "document_version_policy": "offline_static_source; zero is valid; no live-editor freshness asserted",
    }
    return context.to_dict(), geometry


def index_legacy_sidecar() -> tuple[dict[str, tuple[int, int]], str]:
    index: dict[str, tuple[int, int]] = {}
    digest = hashlib.sha256()
    with LEGACY_SIDECAR.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            digest.update(raw)
            value = json.loads(raw)
            row_id = value.get("row_id")
            require(isinstance(row_id, str) and row_id not in index, f"legacy_sidecar_duplicate:{row_id}")
            index[row_id] = (offset, len(raw))
    observed = digest.hexdigest()
    require(observed == EXPECTED["legacy_sidecar"], "legacy_sidecar_hash_mismatch")
    return index, observed


def collect_expanded_ids(legacy_ids: set[str]) -> tuple[
    list[str], set[str], set[str], dict[str, Mapping[str, Any]], Counter[str]
]:
    ordered: list[str] = []
    missing_corrected: set[str] = set()
    missing_short: set[str] = set()
    provenance_by_id: dict[str, Mapping[str, Any]] = {}
    counts: Counter[str] = Counter()
    prompt_targets: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    row_iter = iter_jsonl(EXPANDED_ROWS)
    prov_iter = iter_jsonl(EXPANDED_PROVENANCE)
    for row_item, prov_item in zip(row_iter, prov_iter, strict=True):
        _, _, row = row_item
        _, _, provenance = prov_item
        row_id = str(row.get("id"))
        require(row_id not in seen, f"expanded_duplicate_id:{row_id}")
        seen.add(row_id)
        validate_row(row, provenance)
        prompt_sha = str(provenance["prompt_sha256"])
        target_sha = str(provenance["target_sha256"])
        require(prompt_sha not in prompt_targets, f"expanded_duplicate_or_contradictory_prompt:{row_id}")
        prompt_targets[prompt_sha] = (target_sha, row_id)
        ordered.append(row_id)
        provenance_by_id[row_id] = provenance
        source_class = str(provenance.get("source_class"))
        counts[f"source:{source_class}"] += 1
        counts[f"family:{row.get('family')}"] += 1
        if row_id in legacy_ids:
            counts["legacy_overlap"] += 1
        elif source_class == "corrected":
            missing_corrected.add(row_id)
        elif source_class == "short":
            missing_short.add(row_id)
        else:
            raise PreparationError(f"unsupported_source_class:{row_id}:{source_class}")
    require(len(ordered) == EXPECTED_ROWS, f"expanded_row_count:{len(ordered)}")
    return ordered, missing_corrected, missing_short, provenance_by_id, counts


def collect_dat05_provenance(wanted: set[str]) -> tuple[dict[str, Mapping[str, Any]], str]:
    selected: dict[str, Mapping[str, Any]] = {}
    digest = hashlib.sha256()
    with DAT05_PROVENANCE.open("rb", buffering=4 * 1024 * 1024) as handle:
        for raw in handle:
            digest.update(raw)
            value = json.loads(raw)
            row_id = value.get("id")
            if row_id in wanted:
                require(row_id not in selected, f"dat05_duplicate:{row_id}")
                require(value.get("decision") == "admitted" and value.get("row_split") == "train",
                        f"dat05_not_admitted_train:{row_id}")
                selected[str(row_id)] = value
    observed = digest.hexdigest()
    require(observed == EXPECTED["dat05_provenance"], "dat05_provenance_hash_mismatch")
    require(set(selected) == wanted, "dat05_missing_corrected_ids")
    return selected, observed


def collect_candidate_envelopes(
    provenance: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    by_path: dict[str, dict[int, str]] = {}
    expected_hashes: dict[str, str] = {}
    for row_id, item in provenance.items():
        path = str(item.get("candidate_file"))
        line = item.get("candidate_line")
        expected_sha = item.get("candidate_file_sha256")
        require(Path(path).is_absolute() and type(line) is int and line >= 1, f"candidate_ref:{row_id}")
        require(isinstance(expected_sha, str) and SHA_RE.fullmatch(expected_sha) is not None,
                f"candidate_hash_ref:{row_id}")
        require(path not in expected_hashes or expected_hashes[path] == expected_sha, f"candidate_hash_conflict:{path}")
        expected_hashes[path] = expected_sha
        require(line not in by_path.setdefault(path, {}), f"candidate_line_collision:{path}:{line}")
        by_path[path][line] = row_id

    envelopes: dict[str, Mapping[str, Any]] = {}
    file_results: dict[str, dict[str, Any]] = {}
    for path_text, lines in sorted(by_path.items()):
        path = Path(path_text)
        digest = hashlib.sha256()
        found: set[int] = set()
        with path.open("rb", buffering=4 * 1024 * 1024) as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                row_id = lines.get(line_number)
                if row_id is None:
                    continue
                value = json.loads(raw)
                require(isinstance(value, Mapping), f"candidate_envelope_shape:{row_id}")
                require(value.get("row", {}).get("id") == row_id if isinstance(value.get("row"), Mapping) else False,
                        f"candidate_line_id:{row_id}")
                envelopes[row_id] = value
                found.add(line_number)
        observed = digest.hexdigest()
        require(observed == expected_hashes[path_text], f"candidate_file_hash:{path}")
        require(found == set(lines), f"candidate_lines_missing:{path}")
        file_results[path_text] = {"sha256": observed, "selected_rows": len(found)}
    require(set(envelopes) == set(provenance), "candidate_envelope_id_set")
    return envelopes, file_results


def corrected_sidecar(
    row: Mapping[str, Any], expanded_provenance: Mapping[str, Any],
    dat05: Mapping[str, Any], envelope: Mapping[str, Any],
) -> dict[str, Any]:
    row_id = str(row["id"])
    require(dat05.get("id") == row_id, f"dat05_join:{row_id}")
    require(dat05.get("group_id") == expanded_provenance.get("group_id"), f"group_join:{row_id}")
    require(dat05.get("prompt_sha256") == expanded_provenance.get("prompt_sha256"), f"prompt_join:{row_id}")
    require(envelope.get("status") == "tokenizer_candidate_only" and envelope.get("admitted_for_training") is False,
            f"candidate_status:{row_id}")
    envelope_row = envelope.get("row")
    require(isinstance(envelope_row, Mapping), f"candidate_row_missing:{row_id}")
    require(envelope_row.get("id") == row_id and envelope_row.get("prompt_text") == row.get("prompt_text"),
            f"candidate_prompt_row_join:{row_id}")
    require(envelope_row.get("target_start") == row.get("target_start"), f"candidate_prompt_geometry:{row_id}")
    target_start = int(row["target_start"])
    require(envelope_row.get("input_ids", [])[:target_start] == row.get("input_ids", [])[:target_start],
            f"candidate_prompt_token_prefix:{row_id}")
    require(envelope.get("source_ref") == dat05.get("source_ref"), f"source_ref_join:{row_id}")
    require(envelope.get("source_provenance") == dat05.get("source_provenance"), f"source_provenance_join:{row_id}")
    require(envelope.get("selection") == dat05.get("selection"), f"selection_join:{row_id}")
    context_mapping = envelope.get("context")
    selection = envelope.get("selection")
    source_provenance = envelope.get("source_provenance")
    require(isinstance(context_mapping, Mapping) and isinstance(selection, Mapping) and isinstance(source_provenance, Mapping),
            f"candidate_context_evidence_missing:{row_id}")
    context, geometry = validate_context(row, context_mapping, selection, source_provenance)
    source_ref = envelope.get("source_ref")
    require(isinstance(source_ref, Mapping) and source_ref.get("split") == "train_group", f"source_ref_split:{row_id}")
    return {
        "row_id": row_id,
        "context": context,
        "source_identity": {
            "candidate_file": dat05["candidate_file"],
            "candidate_file_sha256": dat05["candidate_file_sha256"],
            "candidate_line": dat05["candidate_line"],
            "registry_provenance_id": dat05["id"],
            "registry_provenance_decision": dat05["decision"],
            "group_id": dat05["group_id"],
            "package_id": row["package_id"],
            "source_ref": dict(source_ref),
            "source_provenance": dict(source_provenance),
        },
        "selection_geometry": geometry,
        "family": row["family"],
        "package_id": row["package_id"],
        "split": "train",
        "prompt_sha256": expanded_provenance["prompt_sha256"],
        "context_has_target_or_reward_keys": False,
        "offline_static_source": True,
        "expanded_context_source": "DAT05_candidate_envelope_revalidated",
    }


def collect_short_packets(wanted: set[str]) -> tuple[dict[str, Mapping[str, Any]], str]:
    packets: dict[str, Mapping[str, Any]] = {}
    digest = hashlib.sha256()
    with SHORT_PACKETS.open("rb", buffering=4 * 1024 * 1024) as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            packet = json.loads(raw)
            row = packet.get("row")
            row_id = row.get("id") if isinstance(row, Mapping) else None
            if row_id in wanted:
                require(row_id not in packets, f"short_duplicate:{row_id}")
                packet = dict(packet)
                packet["_candidate_line"] = line_number
                packets[str(row_id)] = packet
    observed = digest.hexdigest()
    require(observed == EXPECTED["short_packets"], "short_packets_hash_mismatch")
    require(set(packets) == wanted, "short_packet_id_set")
    return packets, observed


def short_sidecar(
    row: Mapping[str, Any], expanded_provenance: Mapping[str, Any], packet: Mapping[str, Any],
) -> dict[str, Any]:
    row_id = str(row["id"])
    require(packet.get("candidate_status") == "source_and_token_checks_passed_not_training_admitted",
            f"short_status:{row_id}")
    packet_row = packet.get("row")
    require(isinstance(packet_row, Mapping) and canonical(packet_row) == canonical(row), f"short_row_join:{row_id}")
    source_provenance = packet.get("source_provenance")
    context_mapping = packet.get("context")
    require(isinstance(source_provenance, Mapping) and isinstance(context_mapping, Mapping), f"short_evidence:{row_id}")
    selection = source_provenance.get("selection")
    selection_source = source_provenance.get("selection_source")
    require(isinstance(selection, Mapping) and isinstance(selection_source, Mapping), f"short_selection:{row_id}")
    require(source_provenance.get("group_id") == expanded_provenance.get("group_id"), f"short_group:{row_id}")
    context, geometry = validate_context(
        row, context_mapping, selection, source_provenance,
        availability=str(selection_source.get("availability")),
    )
    return {
        "row_id": row_id,
        "context": context,
        "source_identity": {
            "candidate_file": str(SHORT_PACKETS),
            "candidate_file_sha256": EXPECTED["short_packets"],
            "candidate_line": packet["_candidate_line"],
            "registry_provenance_id": row_id,
            "registry_provenance_decision": "prior_short_source_and_token_checks_passed",
            "group_id": expanded_provenance["group_id"],
            "package_id": row["package_id"],
            "source_provenance": dict(source_provenance),
        },
        "selection_geometry": geometry,
        "family": row["family"],
        "package_id": row["package_id"],
        "split": "train",
        "prompt_sha256": expanded_provenance["prompt_sha256"],
        "context_has_target_or_reward_keys": False,
        "offline_static_source": True,
        "expanded_context_source": "accepted_short_packet_revalidated",
    }


def validate_legacy(row: Mapping[str, Any], provenance: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    row_id = str(row["id"])
    require(value.get("row_id") == row_id and value.get("split") == "train", f"legacy_join:{row_id}")
    require(value.get("prompt_sha256") == provenance.get("prompt_sha256"), f"legacy_prompt_hash:{row_id}")
    context_mapping = value.get("context")
    geometry = value.get("selection_geometry")
    source_identity = value.get("source_identity")
    require(isinstance(context_mapping, Mapping) and isinstance(geometry, Mapping) and isinstance(source_identity, Mapping),
            f"legacy_shape:{row_id}")
    source_provenance = source_identity.get("source_provenance")
    require(isinstance(source_provenance, Mapping), f"legacy_source_provenance:{row_id}")
    validate_context(row, context_mapping, geometry, source_provenance, availability=str(geometry.get("availability")))
    require(value.get("context_has_target_or_reward_keys") is False, f"legacy_leak_flag:{row_id}")


def atomic_output(path: Path) -> tuple[BinaryIO, Path]:
    require(not path.exists(), f"output_must_be_fresh:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"temporary_exists:{temporary}")
    return temporary.open("wb"), temporary


def publish(handle: BinaryIO, temporary: Path, final: Path) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(temporary, final)
    directory = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    started = datetime.now(timezone.utc)
    require(not OUT.exists(), f"output_directory_must_be_fresh:{OUT}")
    require(sha_file(EXPANDED_ROWS) == EXPECTED["expanded_rows"], "expanded_rows_hash")
    require(sha_file(EXPANDED_PROVENANCE) == EXPECTED["expanded_provenance"], "expanded_provenance_hash")
    require(sha_file(DAT05_REPORT) == EXPECTED["dat05_report"], "dat05_report_hash")
    require(sha_file(PROTOCOL_SRC / "sepalith/campaign_protocol.py") == EXPECTED["protocol"], "protocol_hash")
    admission = json.loads(EXPANDED_ADMISSION.read_text(encoding="utf-8"))
    require(admission.get("status") == "root_candidate_data_checks_passed_training_recipe_pending", "expanded_not_root_checked")
    require(admission.get("rows") == EXPECTED_ROWS, "expanded_admission_count")
    require(admission.get("data", {}).get("sha256") == EXPECTED["expanded_rows"], "expanded_admission_hash")

    legacy_index, legacy_sha = index_legacy_sidecar()
    ordered_ids, missing_corrected, missing_short, expanded_provenance, counts = collect_expanded_ids(set(legacy_index))
    require(counts["legacy_overlap"] == 7_910, f"legacy_overlap_count:{counts['legacy_overlap']}")
    require(len(missing_corrected) == 3_184, f"missing_corrected_count:{len(missing_corrected)}")
    require(len(missing_short) == 411, f"missing_short_count:{len(missing_short)}")

    dat05, dat05_sha = collect_dat05_provenance(missing_corrected)
    envelopes, candidate_results = collect_candidate_envelopes(dat05)
    short_packets, short_sha = collect_short_packets(missing_short)

    sidecar_path = OUT / "context-sidecar.jsonl"
    selected_ids_path = OUT / "selected-train-ids.json"
    sidecar_handle, sidecar_tmp = atomic_output(sidecar_path)
    sidecar_digest = hashlib.sha256()
    selected_ids: list[str] = []
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    maximums = Counter()
    legacy_handle = LEGACY_SIDECAR.open("rb")
    try:
        row_iter = iter_jsonl(EXPANDED_ROWS)
        prov_iter = iter_jsonl(EXPANDED_PROVENANCE)
        for row_item, prov_item in zip(row_iter, prov_iter, strict=True):
            _, _, row = row_item
            _, _, provenance = prov_item
            row_id = str(row["id"])
            if row_id in legacy_index:
                offset, length = legacy_index[row_id]
                legacy_handle.seek(offset)
                output_line = legacy_handle.read(length)
                value = json.loads(output_line)
                validate_legacy(row, provenance, value)
                source = "legacy_byte_exact"
            elif row_id in missing_corrected:
                value = corrected_sidecar(row, provenance, dat05[row_id], envelopes[row_id])
                output_line = canonical(value) + b"\n"
                source = "DAT05_candidate_envelope_revalidated"
            else:
                require(row_id in missing_short, f"unclassified_row:{row_id}")
                value = short_sidecar(row, provenance, short_packets[row_id])
                output_line = canonical(value) + b"\n"
                source = "accepted_short_packet_revalidated"
            sidecar_handle.write(output_line)
            sidecar_digest.update(output_line)
            selected_ids.append(row_id)
            source_counts[source] += 1
            family_counts[str(row["family"])] += 1
            maximums["sequence_tokens"] = max(maximums["sequence_tokens"], len(row["input_ids"]))
            maximums["prompt_tokens"] = max(maximums["prompt_tokens"], int(row["target_start"]))
            maximums["target_tokens"] = max(
                maximums["target_tokens"], len(row["input_ids"]) - int(row["target_start"])
            )
        require(selected_ids == ordered_ids, "output_order_mismatch")
        publish(sidecar_handle, sidecar_tmp, sidecar_path)
    except Exception:
        sidecar_handle.close()
        sidecar_tmp.unlink(missing_ok=True)
        raise
    finally:
        legacy_handle.close()

    ids_payload = canonical({
        "schema": "RL11-expanded-selected-train-ids-v1",
        "split": "train",
        "row_ids": selected_ids,
    }) + b"\n"
    ids_handle, ids_tmp = atomic_output(selected_ids_path)
    ids_handle.write(ids_payload)
    publish(ids_handle, ids_tmp, selected_ids_path)

    report_path = OUT / "materialization.json"
    report = {
        "schema": "RL11-expanded-context-materialization-v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_candidate_only_no_RL_launch_authorization",
        "limits": {
            "sequence_tokens": MAX_SEQUENCE_TOKENS,
            "target_tokens_including_EOS": MAX_TARGET_TOKENS,
            "truncation": False,
            "maximum_observed": dict(maximums),
        },
        "inputs": {
            "expanded_rows": {"path": str(EXPANDED_ROWS), "sha256": EXPECTED["expanded_rows"], "rows": EXPECTED_ROWS},
            "expanded_provenance": {"path": str(EXPANDED_PROVENANCE), "sha256": EXPECTED["expanded_provenance"], "rows": EXPECTED_ROWS},
            "expanded_root_review": {"path": str(EXPANDED_ADMISSION), "status": admission["status"]},
            "legacy_sidecar": {"path": str(LEGACY_SIDECAR), "sha256": legacy_sha, "available_rows": len(legacy_index)},
            "dat05_provenance": {"path": str(DAT05_PROVENANCE), "sha256": dat05_sha},
            "dat05_report": {"path": str(DAT05_REPORT), "sha256": EXPECTED["dat05_report"]},
            "short_packets": {"path": str(SHORT_PACKETS), "sha256": short_sha, "selected_rows": len(missing_short)},
            "candidate_files": candidate_results,
            "protocol": {"path": str(PROTOCOL_SRC / "sepalith/campaign_protocol.py"), "sha256": EXPECTED["protocol"]},
        },
        "coverage": {
            "expanded_rows": EXPECTED_ROWS,
            "sidecar_rows": len(selected_ids),
            "distinct_ids": len(set(selected_ids)),
            "source_counts": dict(sorted(source_counts.items())),
            "legacy_sidecar_rows_not_in_expanded_pool": len(set(legacy_index) - set(selected_ids)),
            "family_counts": dict(sorted(family_counts.items())),
        },
        "checks": {
            "expanded_rows_root_review_hash_bound": True,
            "expanded_rows_and_provenance_exact_order_join": True,
            "all_rows_train_split": True,
            "all_groups_global_train": True,
            "cpt_validation_rows": 0,
            "duplicate_or_contradictory_prompts": 0,
            "complete_rows_within_explicit_4096_1024_limits": True,
            "legacy_overlap_copied_byte_exact": True,
            "missing_corrected_joined_to_admitted_DAT05_provenance_and_candidate_envelope": True,
            "short_rows_joined_to_accepted_packet": True,
            "render_prompt_exact_match": True,
            "replacement_geometry_and_static_source_hash": True,
            "forbidden_target_or_reward_keys_inside_context": 0,
            "context_only_projection_required_by_RL_loader": True,
            "dev_or_final_content_accessed": False,
            "model_framework_or_weights_loaded": False,
            "cuda_cloud_ssh_or_training": False,
        },
        "outputs": {
            "context_sidecar": {
                "path": str(sidecar_path), "sha256": sidecar_digest.hexdigest(),
                "rows": len(selected_ids), "bytes": sidecar_path.stat().st_size,
            },
            "selected_train_ids": {
                "path": str(selected_ids_path), "sha256": sha_bytes(ids_payload),
                "rows": len(selected_ids), "bytes": len(ids_payload),
                "ordered_ids_sha256": sha_bytes(canonical(selected_ids)),
            },
        },
        "source_identity_note": "Source audit evidence can contain target-related fields out of band under source_identity. The prediction context itself is recursively checked against forbidden target/reward keys; RL loaders must project only context.",
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    report_payload = canonical(report) + b"\n"
    report_handle, report_tmp = atomic_output(report_path)
    report_handle.write(report_payload)
    publish(report_handle, report_tmp, report_path)
    print(json.dumps({
        "status": report["status"], "coverage": report["coverage"],
        "outputs": report["outputs"], "report": str(report_path),
        "report_sha256": sha_bytes(report_payload),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
