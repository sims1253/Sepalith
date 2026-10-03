"""Deterministic PRM-05 source and evidence selection.

The serving and training sides use this module's policy with the pure
TypeScript implementation in ``extensions/vscode-sepalith/src``.  Inputs are
already captured logical lines and provider records.  This module never reads
the target edit or fabricates history.  Required source spans are kept whole;
the bounded candidate policy reports an overflow instead of silently cutting
their only support.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal, Sequence


CAMPAIGN_SELECTION_POLICY_ID = "prm05-selection-dropout-v2"
SOURCE_SELECTION_POLICY_ID = "prm05-source-balanced-v1"
EVIDENCE_DROPOUT_POLICY_ID = "prm05-evidence-support-dropout-v2"
DEFAULT_SELECTION_UTF16_UNITS = 6000
DEFAULT_EVIDENCE_UTF16_UNITS = 6000
OPTIONAL_DROPOUT_RATE_PERCENT = 20


def _fail(message: str) -> None:
    raise ValueError(f"campaign_selection: {message}")


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def utf16_units(text: str) -> int:
    if not isinstance(text, str):
        _fail("text must be a string")
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def _line_cost(line: str) -> int:
    return utf16_units(line) + 1


def _lines_cost(lines: Sequence[str]) -> int:
    return sum(_line_cost(line) for line in lines)


def logical_lines(text: str) -> list[str]:
    """Split LF/CRLF text into wire lines and reject lone CR input."""
    if not isinstance(text, str):
        _fail("text must be a string")
    if any(text[index] == "\r" and (index + 1 == len(text) or text[index + 1] != "\n")
           for index in range(len(text))):
        _fail("lone CR is unsupported")
    return [line[:-1] if line.endswith("\r") else line
            for line in text.split("\n")]


@dataclass(frozen=True)
class SelectionScope:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class EvidenceCandidate:
    id: str
    content: str
    path: str | None
    required: bool
    source_sha256: str
    source_identity: str
    source_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "path": self.path,
            "required": self.required,
            "source_sha256": self.source_sha256,
            "source_identity": self.source_identity,
            "source_version": self.source_version,
        }


@dataclass(frozen=True)
class EvidenceSourceSnapshot:
    identity: str
    version: int
    content_sha256: str


@dataclass(frozen=True)
class EvidenceSupport:
    verified: bool
    target_supported: bool
    policy_id: str
    seed: int
    required_ids: tuple[str, ...]
    required_fingerprint: str


@dataclass(frozen=True)
class SourceSpan:
    kind: Literal["prefix", "scope_prefix", "region", "scope_suffix", "suffix"]
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "start_line": self.start_line, "end_line": self.end_line}


@dataclass(frozen=True)
class SourceOmission:
    side: Literal["prefix", "suffix"]
    start_line: int
    end_line: int
    reason: Literal["source_budget"] = "source_budget"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "source",
            "side": self.side,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceSelectionResult:
    policy_id: str
    budget_utf16_units: int
    used_utf16_units: int
    required_utf16_units: int
    overflow: bool
    required_overflow: bool
    prefix: tuple[str, ...]
    region: tuple[str, ...]
    suffix: tuple[str, ...]
    spans: tuple[SourceSpan, ...]
    omissions: tuple[SourceOmission, ...]
    document_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "budget_utf16_units": self.budget_utf16_units,
            "used_utf16_units": self.used_utf16_units,
            "required_utf16_units": self.required_utf16_units,
            "overflow": self.overflow,
            "required_overflow": self.required_overflow,
            "prefix": list(self.prefix),
            "region": list(self.region),
            "suffix": list(self.suffix),
            "spans": [span.to_dict() for span in self.spans],
            "omissions": [omission.to_dict() for omission in self.omissions],
            "document_sha256": self.document_sha256,
        }


@dataclass(frozen=True)
class EvidenceOmission:
    id: str
    required: bool
    reason: Literal["stale_evidence", "stale_required_evidence", "verified_support_dropout"]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "required": self.required, "reason": self.reason}


@dataclass(frozen=True)
class EvidenceSelectionResult:
    policy_id: str
    budget_utf16_units: int
    used_utf16_units: int
    overflow: bool
    required_overflow: bool
    required_complete: bool
    included: tuple[EvidenceCandidate, ...]
    omissions: tuple[EvidenceOmission, ...]
    stale_ids: tuple[str, ...]
    support_verified: bool
    dropout_enabled: bool
    seed: int | None
    required_ids: tuple[str, ...]
    required_fingerprint: str
    integrity_mismatch_ids: tuple[str, ...]
    dropout_rate_percent: int
    dropped_by_policy: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "budget_utf16_units": self.budget_utf16_units,
            "used_utf16_units": self.used_utf16_units,
            "overflow": self.overflow,
            "required_overflow": self.required_overflow,
            "required_complete": self.required_complete,
            "included": [candidate.to_dict() for candidate in self.included],
            "omissions": [omission.to_dict() for omission in self.omissions],
            "stale_ids": list(self.stale_ids),
            "support_verified": self.support_verified,
            "dropout_enabled": self.dropout_enabled,
            "seed": self.seed,
            "required_ids": list(self.required_ids),
            "required_fingerprint": self.required_fingerprint,
            "integrity_mismatch_ids": list(self.integrity_mismatch_ids),
            "dropout_rate_percent": self.dropout_rate_percent,
            "dropped_by_policy": self.dropped_by_policy,
        }


@dataclass(frozen=True)
class CampaignSelectionResult:
    policy_id: str
    source: SourceSelectionResult
    evidence: EvidenceSelectionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "source": self.source.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_surrounding(
    lines: Sequence[str],
    prefix_indices: Sequence[int],
    suffix_indices: Sequence[int],
    budget: int,
) -> tuple[list[int], list[int], int]:
    prefix_next = len(prefix_indices) - 1
    suffix_next = 0
    prefix_selected: list[int] = []
    suffix_selected: list[int] = []
    prefix_blocked = not prefix_indices
    suffix_blocked = not suffix_indices
    remaining = budget
    used = 0
    turn: Literal["prefix", "suffix"] = "prefix"
    while remaining > 0 and (not prefix_blocked or not suffix_blocked):
        order = (turn, "suffix" if turn == "prefix" else "prefix")
        selected_one = False
        for side in order:
            if side == "prefix":
                if prefix_blocked:
                    continue
                if prefix_next < 0:
                    prefix_blocked = True
                    continue
                index = prefix_indices[prefix_next]
                cost = _line_cost(lines[index])
                if cost > remaining:
                    prefix_blocked = True
                    continue
                prefix_selected.append(index)
                prefix_next -= 1
            else:
                if suffix_blocked:
                    continue
                if suffix_next >= len(suffix_indices):
                    suffix_blocked = True
                    continue
                index = suffix_indices[suffix_next]
                cost = _line_cost(lines[index])
                if cost > remaining:
                    suffix_blocked = True
                    continue
                suffix_selected.append(index)
                suffix_next += 1
            remaining -= cost
            used += cost
            turn = "suffix" if side == "prefix" else "prefix"
            selected_one = True
            break
        if not selected_one and prefix_blocked and suffix_blocked:
            break
    return sorted(prefix_selected), sorted(suffix_selected), used


def _append_omission(omissions: list[SourceOmission], side: Literal["prefix", "suffix"], index: int) -> None:
    if omissions and omissions[-1].side == side and omissions[-1].end_line + 1 == index:
        previous = omissions[-1]
        omissions[-1] = SourceOmission(side, previous.start_line, index)
    else:
        omissions.append(SourceOmission(side, index, index))


def select_source_window(
    *,
    lines: Sequence[str],
    region_start_line: int,
    region_end_line: int,
    scope: SelectionScope | None = None,
    max_utf16_units: int = DEFAULT_SELECTION_UTF16_UNITS,
    document_sha256: str | None = None,
) -> SourceSelectionResult:
    if not isinstance(lines, (list, tuple)) or any(
        not isinstance(line, str) or "\n" in line or "\r" in line for line in lines
    ):
        _fail("lines must be an array of LF-free, CR-free strings")
    source_lines = list(lines)
    region_start_line = _integer(region_start_line, "region_start_line")
    region_end_line = _integer(region_end_line, "region_end_line")
    if region_start_line > region_end_line or region_end_line >= len(source_lines):
        _fail("region lines are outside the source")
    max_utf16_units = _integer(max_utf16_units, "max_utf16_units")
    if scope is not None:
        scope_start = _integer(scope.start_line, "scope.start_line")
        scope_end = _integer(scope.end_line, "scope.end_line")
        if scope_start > scope_end or scope_end >= len(source_lines):
            _fail("scope lines are outside the source")
        valid_scope = scope_start <= region_start_line and scope_end >= region_end_line
    else:
        valid_scope = False
        scope_start = scope_end = 0
    region = source_lines[region_start_line:region_end_line + 1]
    required_prefix_indices = list(range(scope_start, region_start_line)) if valid_scope else []
    required_suffix_indices = list(range(region_end_line + 1, scope_end + 1)) if valid_scope else []
    optional_prefix_indices = list(range(0, scope_start if valid_scope else region_start_line))
    optional_suffix_start = scope_end + 1 if valid_scope else region_end_line + 1
    optional_suffix_indices = list(range(optional_suffix_start, len(source_lines)))
    required_lines = (
        [source_lines[index] for index in required_prefix_indices]
        + region
        + [source_lines[index] for index in required_suffix_indices]
    )
    required_units = _lines_cost(required_lines)
    prefix_selected, suffix_selected, surrounding_used = _select_surrounding(
        source_lines,
        optional_prefix_indices,
        optional_suffix_indices,
        max(0, max_utf16_units - required_units),
    )
    prefix_indices = sorted(prefix_selected + required_prefix_indices)
    suffix_indices = sorted(required_suffix_indices + suffix_selected)
    prefix = tuple(source_lines[index] for index in prefix_indices)
    suffix = tuple(source_lines[index] for index in suffix_indices)
    used_units = required_units + surrounding_used
    required_overflow = required_units > max_utf16_units
    omissions: list[SourceOmission] = []
    prefix_set = set(prefix_selected)
    suffix_set = set(suffix_selected)
    for index in optional_prefix_indices:
        if index not in prefix_set:
            _append_omission(omissions, "prefix", index)
    for index in optional_suffix_indices:
        if index not in suffix_set:
            _append_omission(omissions, "suffix", index)

    spans: list[SourceSpan] = []
    if prefix_indices:
        required_set = set(required_prefix_indices)
        first_required_position = next((i for i, index in enumerate(prefix_indices) if index in required_set), -1)
        if first_required_position > 0:
            spans.append(SourceSpan("prefix", prefix_indices[0], prefix_indices[first_required_position - 1]))
        if first_required_position >= 0:
            spans.append(SourceSpan("scope_prefix", prefix_indices[first_required_position], prefix_indices[-1]))
        else:
            spans.append(SourceSpan("prefix", prefix_indices[0], prefix_indices[-1]))
    spans.append(SourceSpan("region", region_start_line, region_end_line))
    if suffix_indices:
        required_set = set(required_suffix_indices)
        first_optional_position = next((i for i, index in enumerate(suffix_indices) if index not in required_set), -1)
        if first_optional_position > 0:
            spans.append(SourceSpan("scope_suffix", suffix_indices[0], suffix_indices[first_optional_position - 1]))
        if first_optional_position >= 0:
            spans.append(SourceSpan("suffix", suffix_indices[first_optional_position], suffix_indices[-1]))
        elif required_suffix_indices:
            spans.append(SourceSpan("scope_suffix", suffix_indices[0], suffix_indices[-1]))
    return SourceSelectionResult(
        policy_id=SOURCE_SELECTION_POLICY_ID,
        budget_utf16_units=max_utf16_units,
        used_utf16_units=used_units,
        required_utf16_units=required_units,
        overflow=required_overflow or used_units > max_utf16_units,
        required_overflow=required_overflow,
        prefix=prefix,
        region=tuple(region),
        suffix=suffix,
        spans=tuple(spans),
        omissions=tuple(omissions),
        document_sha256=document_sha256,
    )


def _evidence_fingerprint(candidates: Sequence[EvidenceCandidate]) -> str:
    rows = sorted(
        f"{candidate.id}\0{candidate.path or ''}\0{candidate.source_sha256}"
        for candidate in candidates
    )
    return _sha256_text("\n".join(rows))


def _evidence_cost(candidate: EvidenceCandidate) -> int:
    return utf16_units(candidate.content) + (utf16_units(candidate.path) + 1 if candidate.path is not None else 0) + 1


def _valid_evidence_shape(candidate: EvidenceCandidate) -> bool:
    return (
        isinstance(candidate.id, str)
        and bool(candidate.id)
        and isinstance(candidate.content, str)
        and (candidate.path is None or isinstance(candidate.path, str))
        and type(candidate.required) is bool
        and isinstance(candidate.source_sha256, str)
        and isinstance(candidate.source_identity, str)
        and bool(candidate.source_identity)
        and type(candidate.source_version) is int
        and candidate.source_version >= 0
    )


def _content_integrity(candidate: EvidenceCandidate) -> bool:
    return _valid_evidence_shape(candidate) and candidate.source_sha256 == _sha256_text(candidate.content)


def _current_source_matches(
    candidate: EvidenceCandidate,
    current_sources: Sequence[EvidenceSourceSnapshot],
) -> bool:
    current = next((snapshot for snapshot in current_sources if snapshot.identity == candidate.source_identity), None)
    return bool(
        current
        and type(current.version) is int
        and current.version >= 0
        and isinstance(current.content_sha256, str)
        and current.version == candidate.source_version
        and current.content_sha256 == candidate.source_sha256
    )


def select_evidence(
    *,
    required: Sequence[EvidenceCandidate] = (),
    optional: Sequence[EvidenceCandidate] = (),
    max_utf16_units: int = DEFAULT_EVIDENCE_UTF16_UNITS,
    support: EvidenceSupport | None = None,
    current_sources: Sequence[EvidenceSourceSnapshot] = (),
) -> EvidenceSelectionResult:
    required_items = [EvidenceCandidate(x.id, x.content, x.path, True, x.source_sha256, x.source_identity, x.source_version) for x in required]
    optional_items = [EvidenceCandidate(x.id, x.content, x.path, False, x.source_sha256, x.source_identity, x.source_version) for x in optional]
    all_items = required_items + optional_items
    ids: set[str] = set()
    for candidate in all_items:
        if candidate.id in ids:
            _fail(f"duplicate evidence id: {candidate.id}")
        ids.add(candidate.id)
    max_utf16_units = _integer(max_utf16_units, "evidence.max_utf16_units")
    source_identities: set[str] = set()
    for snapshot in current_sources:
        if (
            not isinstance(snapshot.identity, str)
            or not snapshot.identity
            or type(snapshot.version) is not int
            or snapshot.version < 0
            or not isinstance(snapshot.content_sha256, str)
        ):
            _fail("current evidence source snapshots are invalid")
        if snapshot.identity in source_identities:
            _fail(f"duplicate current evidence source: {snapshot.identity}")
        source_identities.add(snapshot.identity)
    integrity_mismatch_ids = tuple(candidate.id for candidate in all_items if not _content_integrity(candidate))
    stale_ids = tuple(
        candidate.id for candidate in all_items
        if not _content_integrity(candidate) or not _current_source_matches(candidate, current_sources)
    )
    stale_set = set(stale_ids)
    fresh_required = [candidate for candidate in required_items if candidate.id not in stale_set]
    fresh_optional = [candidate for candidate in optional_items if candidate.id not in stale_set]
    required_ids = tuple(sorted(candidate.id for candidate in fresh_required))
    required_fingerprint = _evidence_fingerprint(fresh_required)
    support_verified = bool(
        support
        and support.verified is True
        and support.target_supported is True
        and support.policy_id == EVIDENCE_DROPOUT_POLICY_ID
        and type(support.seed) is int
        and support.seed >= 0
        and tuple(sorted(support.required_ids)) == required_ids
        and support.required_fingerprint == required_fingerprint
        and not any(candidate.id in stale_set for candidate in required_items)
    )
    seed = support.seed if support_verified and support is not None else None
    required_used = sum(_evidence_cost(candidate) for candidate in fresh_required)
    omissions = [
        EvidenceOmission(
            candidate.id,
            candidate.id in {item.id for item in required_items},
            "stale_required_evidence" if candidate.id in {item.id for item in required_items} else "stale_evidence",
        )
        for candidate in all_items
        if candidate.id in stale_set
    ]
    included_optional = list(fresh_optional)
    dropout_enabled = False
    dropped_by_policy = 0
    used_units = required_used
    if support_verified and required_used <= max_utf16_units:
        ranked = sorted(
            fresh_optional,
            key=lambda candidate: (_sha256_text(f"{seed}\0{candidate.id}"), candidate.id),
        )
        kept: set[str] = set()
        remaining = max_utf16_units - required_used
        # Per-record Bernoulli threshold: a singleton must not be dropped
        # on every row through rounding a fractional quota up to one.
        threshold = (2**32 * OPTIONAL_DROPOUT_RATE_PERCENT) // 100
        policy_dropped = {
            candidate.id for candidate in ranked
            if int(_sha256_text(f"{seed}\0{candidate.id}")[:8], 16) < threshold
        }
        for candidate in ranked:
            if candidate.id in policy_dropped:
                dropped_by_policy += 1
                dropout_enabled = True
                omissions.append(EvidenceOmission(candidate.id, False, "verified_support_dropout"))
                continue
            cost = _evidence_cost(candidate)
            if cost <= remaining:
                kept.add(candidate.id)
                remaining -= cost
                used_units += cost
            else:
                dropout_enabled = True
                omissions.append(EvidenceOmission(candidate.id, False, "verified_support_dropout"))
        included_optional = [candidate for candidate in fresh_optional if candidate.id in kept]
    else:
        used_units += sum(_evidence_cost(candidate) for candidate in fresh_optional)
    included = tuple(fresh_required + included_optional)
    required_overflow = len(fresh_required) != len(required_items) or required_used > max_utf16_units
    return EvidenceSelectionResult(
        policy_id=EVIDENCE_DROPOUT_POLICY_ID,
        budget_utf16_units=max_utf16_units,
        used_utf16_units=used_units,
        overflow=required_overflow or used_units > max_utf16_units,
        required_overflow=required_overflow,
        required_complete=not required_overflow,
        included=included,
        omissions=tuple(omissions),
        stale_ids=stale_ids,
        support_verified=support_verified,
        dropout_enabled=dropout_enabled,
        seed=seed,
        required_ids=required_ids,
        required_fingerprint=required_fingerprint,
        integrity_mismatch_ids=integrity_mismatch_ids,
        dropout_rate_percent=OPTIONAL_DROPOUT_RATE_PERCENT,
        dropped_by_policy=dropped_by_policy,
    )


def select_campaign_selection(
    *,
    lines: Sequence[str],
    region_start_line: int,
    region_end_line: int,
    scope: SelectionScope | None = None,
    max_utf16_units: int = DEFAULT_SELECTION_UTF16_UNITS,
    document_sha256: str | None = None,
    required_evidence: Sequence[EvidenceCandidate] = (),
    optional_evidence: Sequence[EvidenceCandidate] = (),
    evidence_budget_utf16_units: int = DEFAULT_EVIDENCE_UTF16_UNITS,
    evidence_support: EvidenceSupport | None = None,
    current_evidence_sources: Sequence[EvidenceSourceSnapshot] = (),
) -> CampaignSelectionResult:
    return CampaignSelectionResult(
        policy_id=CAMPAIGN_SELECTION_POLICY_ID,
        source=select_source_window(
            lines=lines,
            region_start_line=region_start_line,
            region_end_line=region_end_line,
            scope=scope,
            max_utf16_units=max_utf16_units,
            document_sha256=document_sha256,
        ),
        evidence=select_evidence(
            required=required_evidence,
            optional=optional_evidence,
            max_utf16_units=evidence_budget_utf16_units,
            support=evidence_support,
            current_sources=current_evidence_sources,
        ),
    )


def selection_fixture_view(result: CampaignSelectionResult) -> dict[str, Any]:
    """Return the exact snake-case JSON shape emitted by the TS checker."""
    return result.to_dict()
