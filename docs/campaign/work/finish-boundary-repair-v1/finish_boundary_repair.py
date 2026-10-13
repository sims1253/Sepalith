#!/usr/bin/env python3
"""A narrow, provenance-gated repair for the DAT-04 finish-block boundary.

The production finish-block conversion labels the literal source remainder
before the function's outer closing brace.  Its live replacement therefore
leaves a complete function only when the represented suffix already contains
that brace.  This module prepares a replacement that carries the brace in the
target only when the packet's exact finish provenance proves that the brace was
excluded and the represented visible suffix does not provide it.

This is an isolated candidate.  It does not change the production adapter or
invent a closing brace for arbitrary completion responses.
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
    """The explicit packet evidence is insufficient for this repair."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RepairPlan:
    """A replacement decision with its evidence and exact wire text."""

    original_target_lines: tuple[str, ...]
    repaired_target_lines: tuple[str, ...]
    original_target_text: str
    repaired_target_text: str
    outer_brace_action: str
    appended_outer_brace: bool
    visible_suffix_lines: tuple[str, ...]
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
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
        raise RepairRejected(f"{name}_must_be_lines")
    if any("\n" in x or "\r" in x for x in value):
        raise RepairRejected(f"{name}_contains_line_break")
    return tuple(value)


def _case_family(case: Mapping[str, Any]) -> str:
    direct = case.get("family")
    if isinstance(direct, str):
        return direct
    identity = case.get("identity")
    if isinstance(identity, Mapping) and isinstance(identity.get("family"), str):
        return str(identity["family"])
    raise RepairRejected("case_family_missing")


def _visible_suffix(context: Mapping[str, Any], supplied: Sequence[str] | None) -> tuple[str, ...]:
    value: object = context.get("suffix_lines") if supplied is None else supplied
    return _lines(value if value is not None else [], "visible_suffix_lines")


def _suffix_action(suffix_lines: tuple[str, ...]) -> tuple[str, bool]:
    """Use only an explicit complete-brace suffix; reject ambiguous suffixes.

    A suffix that starts with a standalone closing brace is already complete.
    An empty suffix proves that the finish fragment has no visible outer brace.
    Any other nonempty suffix is intentionally refused because guessing where
    its outer brace is would duplicate or corrupt source bytes.
    """

    nonempty = [(index, line) for index, line in enumerate(suffix_lines) if line.strip()]
    if not nonempty:
        return "append_outer_brace_to_replacement", True
    first_index, first_line = nonempty[0]
    if first_line.strip() == "}":
        return "use_visible_outer_brace", False
    if any(line.strip() == "}" for _index, line in nonempty):
        raise RepairRejected("visible_suffix_outer_brace_position_ambiguous")
    raise RepairRejected("visible_suffix_does_not_prove_outer_brace")


def _validated_finish_parts(case: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...]]:
    if _case_family(case) != "finish_block":
        raise RepairRejected("family_not_finish_block")
    result = _mapping(case.get("result"), "result")
    if result.get("operation") != "replace":
        raise RepairRejected("operation_not_replace")
    target_lines = _lines(result.get("target_body"), "target_body")
    if not target_lines or target_lines == ("",):
        raise RepairRejected("target_body_empty")
    context = _mapping(result.get("context"), "context")
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
        # The accepted raw-family-v3 fixture predates the packet's nested
        # ``r_fragment`` audit field.  Its source audit carries the same
        # constructor proof and is accepted only as a synthetic fixture path.
        source_audit = _mapping(case.get("source_audit"), "source_audit")
        adapter_audit = _mapping(case.get("adapter_audit"), "adapter_audit")
        identity = _mapping(case.get("identity"), "identity")
        if (
            identity.get("split") != "raw_source_fixture"
            or
            source_audit.get("finish_fragment_replay_verified") is not True
            or source_audit.get("outer_closing_brace_excluded_from_label") is not True
            or adapter_audit.get("source_constructor") != EXPECTED_SOURCE_CONSTRUCTOR
            or adapter_audit.get("target_convention") != EXPECTED_TARGET_CONVENTION
        ):
            raise RepairRejected("raw_fixture_boundary_not_verified")
    framing = _mapping(provenance.get("target_framing"), "target_framing")
    if framing.get("finish_target_bytes_preserved") is not True:
        raise RepairRejected("finish_target_bytes_not_preserved")
    if framing.get("leading_framing_lf_removed") is not False:
        raise RepairRejected("finish_leading_lf_policy_changed")
    selection = _mapping(result.get("selection_source"), "selection_source")
    document = selection.get("document_text")
    if not isinstance(document, str):
        raise RepairRejected("selection_document_missing")
    document_hash = sha256_text(document)
    replacement_range = _mapping(context.get("replacement_range"), "replacement_range")
    if replacement_range.get("content_sha256") != document_hash:
        raise RepairRejected("replacement_range_does_not_bind_document")
    owners = [
        ("content_sha256", selection),
        ("content_sha256", _mapping(pre_edit, "pre_edit_document")),
    ]
    if fragment_value is not None:
        owners.append(("prefix_sha256", _mapping(fragment_value, "r_fragment")))
    for name, owner in owners:
        if owner.get(name) != document_hash:
            raise RepairRejected(f"{name}_does_not_bind_document")
    expected_target_hash = sha256_text("\n".join(target_lines))
    if provenance.get("target_body_sha256") != expected_target_hash:
        raise RepairRejected("target_body_hash_mismatch")
    return result, context, target_lines


def prepare_repair(
    case: Mapping[str, Any],
    *,
    visible_suffix_lines: Sequence[str] | None = None,
) -> RepairPlan:
    """Prepare one explicit finish-block replacement.

    ``visible_suffix_lines`` is an optional caller-supplied observation of the
    lines immediately after the selected replacement.  When omitted, the
    packet's ``context.suffix_lines`` is used.  The argument is an observation,
    not an instruction to append or suppress a brace.
    """

    result, context, target_lines = _validated_finish_parts(case)
    suffix_lines = _visible_suffix(context, visible_suffix_lines)
    action, append_brace = _suffix_action(suffix_lines)
    original_text = "\n".join(target_lines)
    if append_brace:
        # Target rows conventionally retain a final LF as an explicit empty
        # line.  Append the brace byte to that LF; do not create an extra LF.
        repaired_text = original_text + ("}" if original_text.endswith("\n") else "\n}")
    else:
        repaired_text = original_text
    repaired_lines = tuple(repaired_text.split("\n"))
    evidence = {
        "source_constructor": EXPECTED_SOURCE_CONSTRUCTOR,
        "target_convention": EXPECTED_TARGET_CONVENTION,
        "finish_boundary": EXPECTED_BOUNDARY,
        "outer_closing_brace_in_label": False,
        "visible_suffix_empty": not any(line.strip() for line in suffix_lines),
        "visible_suffix_standalone_outer_brace": (
            bool(next((line for line in suffix_lines if line.strip()), ""))
            and next(line for line in suffix_lines if line.strip()).strip() == "}"
        ),
        "target_body_sha256": sha256_text(original_text),
        "repaired_target_sha256": sha256_text(repaired_text),
    }
    return RepairPlan(
        original_target_lines=target_lines,
        repaired_target_lines=repaired_lines,
        original_target_text=original_text,
        repaired_target_text=repaired_text,
        outer_brace_action=action,
        appended_outer_brace=append_brace,
        visible_suffix_lines=suffix_lines,
        evidence=evidence,
    )


def apply_document_replacement(
    document_text: str,
    replacement_range: Mapping[str, Any],
    replacement_text: str,
    *,
    region_old: Sequence[str],
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> str:
    """Apply an exact UTF-16 range to LF text using the pinned converter.

    The production protocol owns UTF-16 validation; this helper receives that
    converter so the candidate cannot silently substitute code-point columns.
    """

    if not isinstance(document_text, str) or not isinstance(replacement_text, str):
        raise RepairRejected("document_or_replacement_not_text")
    lines = document_text.split("\n")
    start = _mapping(replacement_range.get("start"), "replacement_start")
    end = _mapping(replacement_range.get("end"), "replacement_end")
    try:
        start_line = int(start["line"])
        end_line = int(end["line"])
        start_utf16 = int(start["character"])
        end_utf16 = int(end["character"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RepairRejected("replacement_range_position_invalid") from exc
    if min(start_line, end_line, start_utf16, end_utf16) < 0:
        raise RepairRejected("replacement_range_position_negative")
    if start_line > end_line or start_line >= len(lines) or end_line >= len(lines):
        raise RepairRejected("replacement_range_line_out_of_bounds")
    start_cp = utf16_to_codepoint_column(lines[start_line], start_utf16)
    end_cp = utf16_to_codepoint_column(lines[end_line], end_utf16)
    if start_line == end_line and start_cp > end_cp:
        raise RepairRejected("replacement_range_reversed")
    selected_lines = [lines[start_line][start_cp:]]
    if start_line < end_line:
        selected_lines.extend(lines[start_line + 1:end_line])
        selected_lines.append(lines[end_line][:end_cp])
    else:
        selected_lines = [lines[start_line][start_cp:end_cp]]
    old_lines = tuple(region_old)
    if old_lines:
        if tuple(selected_lines) != old_lines:
            raise RepairRejected("replacement_region_old_mismatch")
    elif any(selected_lines):
        raise RepairRejected("empty_region_not_empty")
    prefix_offset = sum(len(line) + 1 for line in lines[:start_line]) + start_cp
    end_offset = sum(len(line) + 1 for line in lines[:end_line]) + end_cp
    return document_text[:prefix_offset] + replacement_text + document_text[end_offset:]


def apply_plan(
    case: Mapping[str, Any],
    plan: RepairPlan,
    *,
    utf16_to_codepoint_column: Callable[[str, int], int],
) -> str:
    """Apply a prepared target to the case's represented pre-edit document."""

    result = _mapping(case.get("result"), "result")
    context = _mapping(result.get("context"), "context")
    selection = _mapping(result.get("selection_source"), "selection_source")
    document = selection.get("document_text")
    if not isinstance(document, str):
        raise RepairRejected("selection_document_missing")
    if selection.get("content_sha256") != sha256_text(document):
        raise RepairRejected("selection_document_hash_mismatch")
    replacement_range = _mapping(context.get("replacement_range"), "replacement_range")
    return apply_document_replacement(
        document,
        replacement_range,
        plan.repaired_target_text,
        region_old=_lines(context.get("region_old"), "region_old"),
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
