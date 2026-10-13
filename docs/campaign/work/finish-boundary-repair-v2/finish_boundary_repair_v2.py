#!/usr/bin/env python3
"""Strict EOF-bound materialization for the DAT-04 finish-block boundary.

The v5 finish constructor labels the literal source remainder before the
function's outer closing brace.  This candidate is deliberately narrower than
the v1 exploration: it accepts only a hashed, LF document whose exact
same-line replacement range ends at the document EOF and whose context carries
no suffix.  Under those facts, adding one ASCII ``}`` byte to the original
target is the direct materialization of the constructor's explicit framing
boundary.

There is no caller-supplied suffix override.  A suffix-bearing or otherwise
ambiguous case is rejected before a target is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Sequence


EXPECTED_FINISH_SPLICE = {
    "literal_source_splice_verified": True,
    "outer_closing_brace_in_label": False,
}
EXPECTED_SOURCE_CONSTRUCTOR = "finish_block_v5_prefix"
EXPECTED_TARGET_CONVENTION = "suffix"
EXPECTED_BOUNDARY = "raw_prefix_plus_corpus_target_before_outer_brace"


class RepairRejected(ValueError):
    """The case is outside the strict materialization contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RepairPlan:
    """An immutable, EOF-bound repaired replacement."""

    original_target_lines: tuple[str, ...]
    repaired_target_lines: tuple[str, ...]
    original_target_text: str
    repaired_target_text: str
    appended_outer_brace: bool
    evidence: Mapping[str, Any]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairRejected(f"{name}_not_mapping")
    return value


def _lines(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise RepairRejected(f"{name}_must_be_lines")
    if any("\n" in item or "\r" in item for item in value):
        raise RepairRejected(f"{name}_contains_line_break")
    return tuple(value)


def _strict_position(value: object, name: str) -> tuple[int, int]:
    position = _mapping(value, name)
    line = position.get("line")
    character = position.get("character")
    # bool is an int subclass.  Reject it, as well as float-like values, before
    # any conversion can silently change the editor geometry.
    if type(line) is not int or line < 0:
        raise RepairRejected(f"{name}_line_must_be_integer")
    if type(character) is not int or character < 0:
        raise RepairRejected(f"{name}_character_must_be_integer")
    return line, character


def _strict_document_version(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RepairRejected(f"{name}_must_be_integer")
    return value


def _case_family(case: Mapping[str, Any]) -> str:
    direct = case.get("family")
    if isinstance(direct, str):
        return direct
    identity = case.get("identity")
    if isinstance(identity, Mapping) and isinstance(identity.get("family"), str):
        return str(identity["family"])
    raise RepairRejected("case_family_missing")


def _check_finish_provenance(case: Mapping[str, Any], result: Mapping[str, Any], document: str) -> None:
    if _case_family(case) != "finish_block":
        raise RepairRejected("family_not_finish_block")
    if result.get("operation") != "replace":
        raise RepairRejected("operation_not_replace")
    provenance = _mapping(result.get("provenance"), "provenance")
    if provenance.get("source_family") != "finish_block":
        raise RepairRejected("provenance_source_family_mismatch")
    if provenance.get("target_convention") != EXPECTED_TARGET_CONVENTION:
        raise RepairRejected("provenance_target_convention_mismatch")
    if provenance.get("target_authority") != ["corpus_target"]:
        raise RepairRejected("target_authority_not_corpus_target")
    if provenance.get("finish_splice") != EXPECTED_FINISH_SPLICE:
        raise RepairRejected("finish_splice_evidence_mismatch")
    pre_edit = _mapping(provenance.get("pre_edit_document"), "pre_edit_document")
    if pre_edit.get("source_constructor") != EXPECTED_SOURCE_CONSTRUCTOR:
        raise RepairRejected("source_constructor_mismatch")
    if pre_edit.get("lineage") != "source_derived_simulated_pre_edit":
        raise RepairRejected("pre_edit_lineage_not_explicit")

    fragment_value = provenance.get("r_fragment")
    if fragment_value is not None:
        fragment = _mapping(fragment_value, "r_fragment")
        if fragment.get("passed") is not True:
            raise RepairRejected("r_fragment_not_verified")
        if fragment.get("body_fragment_boundary") != EXPECTED_BOUNDARY:
            raise RepairRejected("finish_boundary_mismatch")
        if fragment.get("outer_closing_brace_in_label") is not False:
            raise RepairRejected("outer_brace_label_not_excluded")
    else:
        # raw-family-v3 predates the packet's nested r_fragment audit field.
        # Accept that path only for its explicitly marked synthetic fixture.
        identity = _mapping(case.get("identity"), "identity")
        source_audit = _mapping(case.get("source_audit"), "source_audit")
        adapter_audit = _mapping(case.get("adapter_audit"), "adapter_audit")
        if identity.get("split") != "raw_source_fixture":
            raise RepairRejected("raw_fixture_identity_not_synthetic")
        if source_audit.get("finish_fragment_replay_verified") is not True:
            raise RepairRejected("raw_fixture_boundary_not_verified")
        if source_audit.get("outer_closing_brace_excluded_from_label") is not True:
            raise RepairRejected("raw_fixture_outer_brace_not_excluded")
        if adapter_audit.get("source_constructor") != EXPECTED_SOURCE_CONSTRUCTOR:
            raise RepairRejected("raw_fixture_constructor_mismatch")
        if adapter_audit.get("target_convention") != EXPECTED_TARGET_CONVENTION:
            raise RepairRejected("raw_fixture_convention_mismatch")

    framing = _mapping(provenance.get("target_framing"), "target_framing")
    if framing.get("finish_target_bytes_preserved") is not True:
        raise RepairRejected("finish_target_bytes_not_preserved")
    if framing.get("leading_framing_lf_removed") is not False:
        raise RepairRejected("finish_leading_lf_policy_changed")

    selection = _mapping(result.get("selection_source"), "selection_source")
    if selection.get("lineage") != "source_derived_simulated_pre_edit":
        raise RepairRejected("selection_lineage_not_explicit")
    document_hash = sha256_text(document)
    if selection.get("content_sha256") != document_hash:
        raise RepairRejected("selection_document_hash_mismatch")
    if pre_edit.get("content_sha256") != document_hash:
        raise RepairRejected("pre_edit_document_hash_mismatch")
    if fragment_value is not None and _mapping(fragment_value, "r_fragment").get("prefix_sha256") != document_hash:
        raise RepairRejected("fragment_prefix_hash_mismatch")

    selection_version = _strict_document_version(selection.get("document_version"), "selection.document_version")
    pre_version = _strict_document_version(pre_edit.get("document_version"), "pre_edit.document_version")
    if selection_version != pre_version:
        raise RepairRejected("document_version_bindings_disagree")
    if pre_edit.get("document_eol") != "lf":
        raise RepairRejected("pre_edit_document_not_lf")
    if "\r" in document:
        raise RepairRejected("document_contains_non_lf_eol")


def _validate_geometry(
    context: Mapping[str, Any],
    document: str,
    *,
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> tuple[Mapping[str, Any], tuple[str, ...], int, int]:
    if context.get("document_eol") != "lf":
        raise RepairRejected("context_document_not_lf")
    suffix_lines = _lines(context.get("suffix_lines"), "context.suffix_lines")
    if suffix_lines:
        raise RepairRejected("suffix_present_not_admitted")
    region_old = _lines(context.get("region_old"), "context.region_old")
    replacement_range = _mapping(context.get("replacement_range"), "replacement_range")
    if replacement_range.get("content_sha256") != sha256_text(document):
        raise RepairRejected("replacement_range_does_not_bind_document")
    range_version = _strict_document_version(replacement_range.get("document_version"), "replacement_range.document_version")
    start_line, start_character = _strict_position(replacement_range.get("start"), "replacement_range.start")
    end_line, end_character = _strict_position(replacement_range.get("end"), "replacement_range.end")
    if start_line != end_line:
        raise RepairRejected("multi_line_finish_geometry_not_admitted")
    lines = document.split("\n")
    if start_line >= len(lines):
        raise RepairRejected("replacement_start_line_out_of_bounds")
    try:
        start_cp = utf16_to_codepoint_column(lines[start_line], start_character)
        end_cp = utf16_to_codepoint_column(lines[end_line], end_character)
    except Exception as exc:  # noqa: BLE001
        raise RepairRejected("utf16_geometry_invalid") from exc
    if start_cp > end_cp:
        raise RepairRejected("replacement_range_reversed")
    start_offset = sum(len(line) + 1 for line in lines[:start_line]) + start_cp
    end_offset = sum(len(line) + 1 for line in lines[:end_line]) + end_cp
    if end_offset != len(document):
        raise RepairRejected("replacement_end_not_document_eof")
    if end_cp != len(lines[end_line]):
        raise RepairRejected("replacement_end_not_line_end")
    selected_text = document[start_offset:end_offset]
    if region_old:
        if selected_text != "\n".join(region_old):
            raise RepairRejected("replacement_region_old_mismatch")
    elif selected_text:
        raise RepairRejected("empty_region_not_empty")
    return replacement_range, region_old, range_version, end_offset


def _validated_parts(
    case: Mapping[str, Any],
    *,
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...], str, Mapping[str, Any]]:
    result = _mapping(case.get("result"), "result")
    context = _mapping(result.get("context"), "context")
    selection = _mapping(result.get("selection_source"), "selection_source")
    document = selection.get("document_text")
    if not isinstance(document, str):
        raise RepairRejected("selection_document_missing")
    _check_finish_provenance(case, result, document)
    replacement_range, region_old, range_version, _end_offset = _validate_geometry(
        context,
        document,
        utf16_to_codepoint_column=utf16_to_codepoint_column,
    )
    if range_version != _strict_document_version(selection.get("document_version"), "selection.document_version"):
        raise RepairRejected("replacement_version_does_not_bind_document")
    target_lines = _lines(result.get("target_body"), "target_body")
    if not target_lines or target_lines == ("",):
        raise RepairRejected("target_body_empty")
    target_text = "\n".join(target_lines)
    if "\r" in target_text:
        raise RepairRejected("target_contains_non_lf_eol")
    # The exact finish source framing is `post + b"}"`.  A target without its
    # final LF has no admitted line framing under this production geometry.
    if not target_text.endswith("\n"):
        raise RepairRejected("target_not_lf_terminated")
    provenance = _mapping(result.get("provenance"), "provenance")
    if provenance.get("target_body_sha256") != sha256_text(target_text):
        raise RepairRejected("target_body_hash_mismatch")
    return result, context, target_lines, target_text, replacement_range


def prepare_repair(
    case: Mapping[str, Any],
    *,
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> RepairPlan:
    """Create the only admitted materialization: ``original_target + '}'``."""

    _result, context, target_lines, target_text, replacement_range = _validated_parts(
        case,
        utf16_to_codepoint_column=utf16_to_codepoint_column,
    )
    repaired_text = target_text + "}"
    if repaired_text[:-1] != target_text or repaired_text[-1] != "}":
        raise RepairRejected("outer_brace_materialization_internal_error")
    return RepairPlan(
        original_target_lines=target_lines,
        repaired_target_lines=tuple(repaired_text.split("\n")),
        original_target_text=target_text,
        repaired_target_text=repaired_text,
        appended_outer_brace=True,
        evidence={
            "replacement_content_sha256": replacement_range["content_sha256"],
            "replacement_ends_at_document_eof": True,
            "context_suffix_lines": [],
            "target_body_sha256": sha256_text(target_text),
            "repaired_target_sha256": sha256_text(repaired_text),
            "outer_brace_materialization": "append exactly one ASCII } byte",
        },
    )


def apply_document_replacement(
    document_text: str,
    replacement_range: Mapping[str, Any],
    replacement_text: str,
    *,
    region_old: Sequence[str],
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> str:
    """Apply a strict same-line UTF-16 range, including its hash binding."""

    if not isinstance(document_text, str) or "\r" in document_text:
        raise RepairRejected("document_not_lf_text")
    if not isinstance(replacement_text, str) or "\r" in replacement_text:
        raise RepairRejected("replacement_not_lf_text")
    if "content_sha256" not in replacement_range:
        raise RepairRejected("replacement_range_hash_missing")
    if replacement_range.get("content_sha256") != sha256_text(document_text):
        raise RepairRejected("replacement_range_does_not_bind_document")
    start_line, start_character = _strict_position(replacement_range.get("start"), "replacement_range.start")
    end_line, end_character = _strict_position(replacement_range.get("end"), "replacement_range.end")
    if start_line != end_line:
        raise RepairRejected("multi_line_finish_geometry_not_admitted")
    lines = document_text.split("\n")
    if start_line >= len(lines):
        raise RepairRejected("replacement_start_line_out_of_bounds")
    try:
        start_cp = utf16_to_codepoint_column(lines[start_line], start_character)
        end_cp = utf16_to_codepoint_column(lines[end_line], end_character)
    except Exception as exc:  # noqa: BLE001
        raise RepairRejected("utf16_geometry_invalid") from exc
    if start_cp > end_cp:
        raise RepairRejected("replacement_range_reversed")
    selected = lines[start_line][start_cp:end_cp]
    old_lines = _lines(region_old, "region_old")
    if old_lines:
        if selected != "\n".join(old_lines):
            raise RepairRejected("replacement_region_old_mismatch")
    elif selected:
        raise RepairRejected("empty_region_not_empty")
    start_offset = sum(len(line) + 1 for line in lines[:start_line]) + start_cp
    end_offset = sum(len(line) + 1 for line in lines[:end_line]) + end_cp
    if end_offset != len(document_text):
        raise RepairRejected("replacement_end_not_document_eof")
    if end_cp != len(lines[end_line]):
        raise RepairRejected("replacement_end_not_line_end")
    return document_text[:start_offset] + replacement_text + document_text[end_offset:]


def apply_plan(
    case: Mapping[str, Any],
    plan: RepairPlan,
    *,
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> str:
    """Revalidate the case and apply the immutable plan to its hashed document."""

    result, context, target_lines, target_text, replacement_range = _validated_parts(
        case,
        utf16_to_codepoint_column=utf16_to_codepoint_column,
    )
    if (
        tuple(target_lines) != plan.original_target_lines
        or target_text != plan.original_target_text
        or plan.repaired_target_text != target_text + "}"
        or plan.appended_outer_brace is not True
    ):
        raise RepairRejected("plan_case_binding_mismatch")
    selection = _mapping(result.get("selection_source"), "selection_source")
    document = selection["document_text"]
    return apply_document_replacement(
        document,
        replacement_range,
        plan.repaired_target_text,
        region_old=context["region_old"],
        utf16_to_codepoint_column=utf16_to_codepoint_column,
    )


__all__ = [
    "RepairPlan",
    "RepairRejected",
    "apply_document_replacement",
    "apply_plan",
    "prepare_repair",
    "sha256_bytes",
    "sha256_text",
]
