#!/usr/bin/env python3
"""Guard the post-freeze raw-source admission boundary.

The DAT-08 raw-source builder is deliberately a derivation component.  It
does not decide whether a path belongs to a sealed split, and it writes
through a replace-style helper.  This module is the small caller-side
boundary proposed for that component.

The boundary has two phases:

* metadata preflight validates the freeze receipt, positive source roots,
  split disjointness, semantic per-case bindings, and the output path;
* only after every preflight check succeeds are source bytes opened through
  no-follow directory descriptors and checksum-checked.

The output is a write-once metadata manifest.  It contains source/context/
target hashes and provenance, never source bytes or prompts.  The returned
'AdmissionResult.source_bytes' is an in-memory handoff for a caller that
will invoke the raw builder after admission; callers must not serialize it
as final content.

This is a preparation wrapper.  A local receipt proves that this code checked
the supplied values; it does not attest that a separate signer or storage
system trusted the values.  A production caller must pin the expected weight,
harness, source-lock, semantic-receipt, and component hashes in its campaign
recipe before invoking this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid


FREEZE_AT = _datetime.datetime(
    2026,
    9,
    14,
    10,
    0,
    0,
    tzinfo=_datetime.timezone.utc,
)
FREEZE_AT_TEXT = "2026-09-14T10:00:00Z"
GUARD_SCHEMA = "dat08.final-admission-guard.v1"
SEMANTIC_RECEIPT_SCHEMA = "dat08.semantic-source-case-receipt.v1"
SEMANTIC_RECEIPT_SPLIT = "final_locked_v1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
IDENTITY_FIELDS = ("row_id", "package_id", "group_id", "split_identity")
MEMBERSHIP_KEYS = {
    "row_ids": "row_id",
    "package_ids": "package_id",
    "group_ids": "group_id",
    "split_identities": "split_identity",
}
FORBIDDEN_PATH_COMPONENTS = frozenset(
    {
        "eval",
        "candidate-references",
        "candidate_references",
        "source-selection",
        "source_selection",
        "heldout",
        "held-out",
        "held_out",
    }
)


class AdmissionGuardError(ValueError):
    """Base class for a rejected admission request."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        message = code if detail is None else f"{code}:{detail}"
        super().__init__(message)


class AdmissionMetadataError(AdmissionGuardError):
    """A metadata or authorization preflight failed."""


class AdmissionSourceError(AdmissionGuardError):
    """A source path changed, escaped, or had unexpected bytes."""


class AdmissionOutputError(AdmissionGuardError):
    """The write-once manifest could not be durably created."""


BeforeSourceRead = Callable[["SourceCaseSpec"], None]


def canonical_json(value: Any) -> str:  # noqa: ANN401
    """Return the stable JSON representation used for identity hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:  # noqa: ANN401
    return sha256_bytes(canonical_json(value).encode("utf-8", "surrogatepass"))


def _required_text(value: Any, field: str) -> str:  # noqa: ANN401
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AdmissionMetadataError("metadata_text_required", field)
    return value.strip()


def _required_hash(value: Any, field: str) -> str:  # noqa: ANN401
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AdmissionMetadataError("sha256_required", field)
    return value.lower()


def _parse_timestamp(value: Any, field: str) -> _datetime.datetime:  # noqa: ANN401
    if isinstance(value, _datetime.datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = _datetime.datetime.fromisoformat(text)
        except ValueError as exc:
            raise AdmissionMetadataError("timestamp_invalid", field) from exc
    else:
        raise AdmissionMetadataError("timestamp_required", field)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdmissionMetadataError("timestamp_must_be_timezone_aware", field)
    return parsed.astimezone(_datetime.timezone.utc)


def _timestamp_text(value: _datetime.datetime) -> str:
    return value.astimezone(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_copy(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdmissionMetadataError("mapping_required", field)
    try:
        copied = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise AdmissionMetadataError("metadata_not_json", field) from exc
    if not isinstance(copied, dict):
        raise AdmissionMetadataError("mapping_required", field)
    return copied


def _path_components(value: str) -> set[str]:
    return {part.lower() for part in re.split(r"[/\\]+", value) if part}


def _reject_forbidden_path(value: str, field: str) -> None:
    if _path_components(value) & FORBIDDEN_PATH_COMPONENTS:
        raise AdmissionMetadataError("forbidden_source_path_component", field)


def _is_symlink_component(path: Path) -> bool:
    """Check every lexical component without reading file contents."""
    if not path.is_absolute():
        return False
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                return True
        except OSError as exc:
            raise AdmissionMetadataError("source_path_lstat_failed", str(current)) from exc
    return False


def _has_parent_component(path: Path) -> bool:
    return any(part in {".", ".."} for part in path.parts)


def _coerce_observed(value: _datetime.datetime | str) -> _datetime.datetime:
    return _parse_timestamp(value, "observed_at")


def _identity_tokens(case: "SourceCaseSpec") -> set[str]:
    """Produce conservative row/package/group/split collision tokens."""
    values = {
        "row_id": case.row_id,
        "package_id": case.package_id,
        "group_id": case.group_id,
        "split_identity": case.split_identity,
    }
    tokens: set[str] = set(values.values())
    tokens.update(f"{name}:{item}" for name, item in values.items())
    return tokens


def _membership_tokens(value: Any, field: str) -> set[str]:  # noqa: ANN401
    """Normalize a split-membership mapping without opening source paths.

    A mapping is preferred and records whether each item is a row, package,
    group, or split identity.  A flat sequence is accepted as a set of
    split-identity values for small callers and is also kept as raw values so
    a package/group collision cannot be hidden by an unnamespaced fixture.
    """
    if isinstance(value, str) or value is None:
        raise AdmissionMetadataError("identity_membership_required", field)
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, members in value.items():
            if key not in MEMBERSHIP_KEYS:
                raise AdmissionMetadataError("identity_membership_key_unknown", str(key))
            if isinstance(members, str) or not isinstance(members, (list, tuple, set, frozenset)):
                raise AdmissionMetadataError("identity_membership_values_required", str(key))
            identity_field = MEMBERSHIP_KEYS[key]
            for member in members:
                text = _required_text(member, f"{field}.{key}")
                result.add(text)
                result.add(f"{identity_field}:{text}")
        if not result:
            raise AdmissionMetadataError("identity_membership_empty", field)
        return result
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise AdmissionMetadataError("identity_membership_required", field)
    for member in value:
        text = _required_text(member, field)
        result.add(text)
        result.add(f"split_identity:{text}")
    if not result:
        raise AdmissionMetadataError("identity_membership_empty", field)
    return result


@dataclass(frozen=True)
class SourceCaseSpec:
    """Metadata required before one raw source file may be opened."""

    row_id: str
    split_identity: str
    package_id: str
    repository_id: str
    group_id: str
    family: str
    source_path: Path
    source_sha256: str
    context_sha256: str
    target_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceCaseSpec":
        if not isinstance(value, Mapping):
            raise AdmissionMetadataError("case_mapping_required")
        text_fields = (
            "row_id",
            "split_identity",
            "package_id",
            "repository_id",
            "group_id",
            "family",
        )
        values = {field: _required_text(value.get(field), field) for field in text_fields}
        source_path_value = value.get("source_path")
        source_path_text = _required_text(source_path_value, "source_path")
        if not Path(source_path_text).is_absolute():
            raise AdmissionMetadataError("source_path_must_be_absolute", source_path_text)
        if _has_parent_component(Path(source_path_text)):
            raise AdmissionMetadataError("source_path_parent_component", source_path_text)
        hashes = {
            field: _required_hash(value.get(field), field)
            for field in ("source_sha256", "context_sha256", "target_sha256")
        }
        return cls(
            **values,
            source_path=Path(source_path_text),
            **hashes,
        )


@dataclass(frozen=True)
class _PlannedCase:
    spec: SourceCaseSpec
    resolved_path: Path
    root: Path
    before_stat: os.stat_result


@dataclass(frozen=True)
class AdmissionResult:
    """The durable metadata result plus the post-gate builder handoff."""

    manifest: Mapping[str, Any]
    output_path: Path
    source_bytes: Mapping[str, bytes]


class AdmissionGuard:
    """Perform an all-metadata preflight, then admit checksum-pinned sources."""

    def __init__(
        self,
        *,
        allowed_source_roots: Sequence[Path],
        train_identities: Mapping[str, Iterable[str]] | Iterable[str],
        dev_identities: Mapping[str, Iterable[str]] | Iterable[str],
        freeze_receipt: Mapping[str, Any],
        semantic_receipt: Mapping[str, Any],
        expected_weights_sha256: str,
        expected_harness_sha256: str,
        expected_source_lock_sha256: str,
        observed_at: _datetime.datetime | str,
    ) -> None:
        self.allowed_source_roots = tuple(Path(root) for root in allowed_source_roots)
        self.train_identities = train_identities
        self.dev_identities = dev_identities
        self.freeze_receipt = _json_copy(freeze_receipt, "freeze_receipt")
        self.semantic_receipt = _json_copy(semantic_receipt, "semantic_receipt")
        self.expected_weights_sha256 = expected_weights_sha256
        self.expected_harness_sha256 = expected_harness_sha256
        self.expected_source_lock_sha256 = expected_source_lock_sha256
        self.observed_at = _coerce_observed(observed_at)

    def _validate_freeze_receipt(self) -> dict[str, Any]:
        receipt = self.freeze_receipt
        if receipt.get("status") != "frozen":
            raise AdmissionMetadataError("freeze_receipt_not_frozen")
        for field in ("weights_frozen", "harness_frozen", "final_access_unlocked"):
            if receipt.get(field) is not True:
                raise AdmissionMetadataError("freeze_authorization_missing", field)
        expected_weights = _required_hash(self.expected_weights_sha256, "expected_weights_sha256")
        expected_harness = _required_hash(self.expected_harness_sha256, "expected_harness_sha256")
        if _required_hash(receipt.get("weights_sha256"), "freeze_receipt.weights_sha256") != expected_weights:
            raise AdmissionMetadataError("weights_hash_not_pinned")
        if _required_hash(receipt.get("harness_sha256"), "freeze_receipt.harness_sha256") != expected_harness:
            raise AdmissionMetadataError("harness_hash_not_pinned")
        if self.observed_at < FREEZE_AT:
            raise AdmissionMetadataError("freeze_time_not_reached", FREEZE_AT_TEXT)
        for field in ("weights_frozen_at", "harness_frozen_at"):
            timestamp = _parse_timestamp(receipt.get(field), f"freeze_receipt.{field}")
            if timestamp > self.observed_at:
                raise AdmissionMetadataError("freeze_receipt_after_observation", field)
        return {
            "status": "frozen",
            "weights_sha256": expected_weights,
            "harness_sha256": expected_harness,
            "weights_frozen_at": _timestamp_text(
                _parse_timestamp(receipt["weights_frozen_at"], "freeze_receipt.weights_frozen_at")
            ),
            "harness_frozen_at": _timestamp_text(
                _parse_timestamp(receipt["harness_frozen_at"], "freeze_receipt.harness_frozen_at")
            ),
        }

    def _validate_semantic_receipt(
        self,
        cases: Sequence[SourceCaseSpec],
    ) -> dict[str, Any]:
        receipt = self.semantic_receipt
        if receipt.get("schema") != SEMANTIC_RECEIPT_SCHEMA:
            raise AdmissionMetadataError("semantic_receipt_schema_mismatch")
        if receipt.get("status") != "source_cases_verified":
            raise AdmissionMetadataError("semantic_receipt_not_verified")
        if receipt.get("split") != SEMANTIC_RECEIPT_SPLIT:
            raise AdmissionMetadataError("semantic_receipt_split_mismatch")
        expected_lock = _required_hash(
            self.expected_source_lock_sha256,
            "expected_source_lock_sha256",
        )
        if _required_hash(receipt.get("source_lock_sha256"), "semantic_receipt.source_lock_sha256") != expected_lock:
            raise AdmissionMetadataError("semantic_source_lock_mismatch")
        if receipt.get("content_access") != "post_freeze_source_bytes":
            raise AdmissionMetadataError("semantic_receipt_content_access_mismatch")
        constructor_hash = _required_hash(
            receipt.get("constructor_sha256"),
            "semantic_receipt.constructor_sha256",
        )
        case_ids = receipt.get("verified_case_ids")
        if isinstance(case_ids, str) or not isinstance(case_ids, (list, tuple, set, frozenset)):
            raise AdmissionMetadataError("semantic_case_ids_required")
        if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
            raise AdmissionMetadataError("semantic_case_ids_invalid")
        if len(set(case_ids)) != len(case_ids):
            raise AdmissionMetadataError("semantic_case_ids_duplicate")
        expected_ids = {case.row_id for case in cases}
        if set(case_ids) != expected_ids:
            raise AdmissionMetadataError("semantic_case_ids_mismatch")
        bindings = receipt.get("case_bindings")
        if not isinstance(bindings, Mapping):
            raise AdmissionMetadataError("semantic_case_bindings_required")
        if set(bindings) != expected_ids:
            raise AdmissionMetadataError("semantic_case_bindings_mismatch")
        for case in cases:
            binding = bindings.get(case.row_id)
            if not isinstance(binding, Mapping):
                raise AdmissionMetadataError("semantic_case_binding_required", case.row_id)
            expected_binding = {
                "split_identity": case.split_identity,
                "source_sha256": case.source_sha256,
                "context_sha256": case.context_sha256,
                "target_sha256": case.target_sha256,
            }
            if set(binding) != set(expected_binding):
                raise AdmissionMetadataError("semantic_case_binding_fields_mismatch", case.row_id)
            for field, expected in expected_binding.items():
                actual = binding.get(field)
                if field.endswith("_sha256"):
                    actual = _required_hash(actual, f"semantic_case_binding.{field}")
                elif not isinstance(actual, str) or actual != expected:
                    raise AdmissionMetadataError("semantic_binding_mismatch", case.row_id)
                if actual != expected:
                    raise AdmissionMetadataError("semantic_binding_mismatch", case.row_id)
        return {
            "status": "source_cases_verified",
            "split": SEMANTIC_RECEIPT_SPLIT,
            "source_lock_sha256": expected_lock,
            "content_access": "post_freeze_source_bytes",
            "verified_case_ids": sorted(expected_ids),
            "constructor_sha256": constructor_hash,
            "case_bindings": {
                case.row_id: {
                    "split_identity": case.split_identity,
                    "source_sha256": case.source_sha256,
                    "context_sha256": case.context_sha256,
                    "target_sha256": case.target_sha256,
                }
                for case in sorted(cases, key=lambda item: item.row_id)
            },
        }

    def _validate_roots(self) -> tuple[Path, ...]:
        if not self.allowed_source_roots:
            raise AdmissionMetadataError("source_allowlist_empty")
        roots: list[Path] = []
        for raw_root in self.allowed_source_roots:
            if not raw_root.is_absolute():
                raise AdmissionMetadataError("source_root_must_be_absolute", str(raw_root))
            if _has_parent_component(raw_root):
                raise AdmissionMetadataError("source_root_parent_component", str(raw_root))
            if _is_symlink_component(raw_root):
                raise AdmissionMetadataError("source_root_symlink_component", str(raw_root))
            try:
                resolved = raw_root.resolve(strict=True)
                stat = os.stat(resolved, follow_symlinks=False)
            except OSError as exc:
                raise AdmissionMetadataError("source_root_unavailable", str(raw_root)) from exc
            if not stat_is_directory(stat.st_mode):
                raise AdmissionMetadataError("source_root_not_directory", str(raw_root))
            if resolved in roots:
                raise AdmissionMetadataError("source_root_duplicate", str(resolved))
            roots.append(resolved)
        return tuple(roots)

    def _validate_source_path(
        self,
        case: SourceCaseSpec,
        roots: Sequence[Path],
    ) -> _PlannedCase:
        raw_path = case.source_path
        _reject_forbidden_path(str(raw_path), "source_path")
        if not raw_path.is_absolute():
            raise AdmissionMetadataError("source_path_must_be_absolute", str(raw_path))
        if _has_parent_component(raw_path):
            raise AdmissionMetadataError("source_path_parent_component", str(raw_path))
        if _is_symlink_component(raw_path):
            raise AdmissionMetadataError("source_symlink_component", str(raw_path))
        try:
            resolved = raw_path.resolve(strict=True)
            source_stat = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise AdmissionMetadataError("source_path_unavailable", str(raw_path)) from exc
        if not stat_is_regular(source_stat.st_mode):
            raise AdmissionMetadataError("source_path_not_regular", str(raw_path))
        if _path_components(str(resolved)) & FORBIDDEN_PATH_COMPONENTS:
            raise AdmissionMetadataError("forbidden_source_path_component", str(resolved))
        matching_roots = [
            root for root in roots if _is_relative_to(resolved, root)
        ]
        if not matching_roots:
            raise AdmissionMetadataError("source_outside_positive_allowlist", str(raw_path))
        root = min(matching_roots, key=lambda item: len(item.parts))
        return _PlannedCase(
            spec=case,
            resolved_path=resolved,
            root=root,
            before_stat=source_stat,
        )

    def _validate_output_path(self, output_path: Path) -> Path:
        if not output_path.is_absolute():
            raise AdmissionMetadataError("output_path_must_be_absolute", str(output_path))
        if _has_parent_component(output_path):
            raise AdmissionMetadataError("output_path_parent_component", str(output_path))
        parent = output_path.parent
        if _is_symlink_component(parent):
            raise AdmissionMetadataError("output_parent_symlink_component", str(parent))
        try:
            parent_resolved = parent.resolve(strict=True)
            parent_stat = os.stat(parent_resolved, follow_symlinks=False)
        except OSError as exc:
            raise AdmissionMetadataError("output_parent_unavailable", str(parent)) from exc
        if not stat_is_directory(parent_stat.st_mode):
            raise AdmissionMetadataError("output_parent_not_directory", str(parent))
        if os.path.lexists(output_path):
            raise AdmissionMetadataError("output_exists", str(output_path))
        return output_path

    def _preflight(
        self,
        case_values: Sequence[SourceCaseSpec | Mapping[str, Any]],
        output_path: Path,
    ) -> tuple[tuple[_PlannedCase, ...], dict[str, Any], dict[str, Any], Path]:
        # This method intentionally does not call read(), read_bytes(), or
        # open() on a source path.  It only examines metadata and lstat values.
        freeze = self._validate_freeze_receipt()
        roots = self._validate_roots()
        train = _membership_tokens(self.train_identities, "train_identities")
        dev = _membership_tokens(self.dev_identities, "dev_identities")
        if train & dev:
            raise AdmissionMetadataError("train_dev_identity_overlap")
        cases: list[SourceCaseSpec] = []
        for value in case_values:
            if isinstance(value, SourceCaseSpec):
                value = SourceCaseSpec.from_mapping(
                    {
                        "row_id": value.row_id,
                        "split_identity": value.split_identity,
                        "package_id": value.package_id,
                        "repository_id": value.repository_id,
                        "group_id": value.group_id,
                        "family": value.family,
                        "source_path": str(value.source_path),
                        "source_sha256": value.source_sha256,
                        "context_sha256": value.context_sha256,
                        "target_sha256": value.target_sha256,
                    }
                )
            else:
                value = SourceCaseSpec.from_mapping(value)
            cases.append(value)
        if not cases:
            raise AdmissionMetadataError("cases_empty")
        row_ids = [case.row_id for case in cases]
        if len(set(row_ids)) != len(row_ids):
            raise AdmissionMetadataError("case_row_id_duplicate")
        split_ids = [case.split_identity for case in cases]
        if len(set(split_ids)) != len(split_ids):
            raise AdmissionMetadataError("case_split_identity_duplicate")
        for case in cases:
            collisions = _identity_tokens(case) & (train | dev)
            if collisions:
                raise AdmissionMetadataError(
                    "case_split_identity_overlap",
                    sorted(collisions)[0],
                )
        semantic = self._validate_semantic_receipt(cases)
        planned = tuple(
            self._validate_source_path(case, roots)
            for case in sorted(cases, key=lambda item: item.row_id)
        )
        output = self._validate_output_path(Path(output_path))
        return planned, freeze, semantic, output

    def _read_source(
        self,
        plan: _PlannedCase,
        before_source_read: BeforeSourceRead | None,
    ) -> bytes:
        if before_source_read is not None:
            before_source_read(plan.spec)
        relative = plan.resolved_path.relative_to(plan.root)
        parts = relative.parts
        if not parts:
            raise AdmissionSourceError("source_relative_path_empty", plan.spec.row_id)
        directory_fds: list[int] = []
        file_fd = -1
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
            root_fd = os.open(plan.root, flags)
            directory_fds.append(root_fd)
            current_fd = root_fd
            for directory in parts[:-1]:
                next_fd = os.open(directory, flags, dir_fd=current_fd)
                directory_fds.append(next_fd)
                current_fd = next_fd
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            opened_stat = os.fstat(file_fd)
            if not stat_is_regular(opened_stat.st_mode):
                raise AdmissionSourceError("source_opened_nonregular", plan.spec.row_id)
            if not same_file_identity(plan.before_stat, opened_stat):
                raise AdmissionSourceError("source_replaced_after_gate", plan.spec.row_id)
            chunks: list[bytes] = []
            while True:
                block = os.read(file_fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            final_stat = os.fstat(file_fd)
            if not same_file_identity(opened_stat, final_stat):
                raise AdmissionSourceError("source_changed_during_read", plan.spec.row_id)
            source_bytes = b"".join(chunks)
        except AdmissionSourceError:
            raise
        except OSError as exc:
            raise AdmissionSourceError("source_open_or_read_failed", plan.spec.row_id) from exc
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            for descriptor in reversed(directory_fds):
                os.close(descriptor)
        actual_sha256 = sha256_bytes(source_bytes)
        if actual_sha256 != plan.spec.source_sha256:
            raise AdmissionSourceError("source_sha256_mismatch", plan.spec.row_id)
        return source_bytes

    def admit(
        self,
        cases: Sequence[SourceCaseSpec | Mapping[str, Any]],
        *,
        output_path: Path,
        before_source_read: BeforeSourceRead | None = None,
    ) -> AdmissionResult:
        planned, freeze, semantic, output = self._preflight(cases, Path(output_path))
        source_payloads: dict[str, bytes] = {}
        source_records: list[dict[str, Any]] = []
        for plan in planned:
            source_bytes = self._read_source(plan, before_source_read)
            source_payloads[plan.spec.row_id] = source_bytes
            source_records.append(
                {
                    "row_id": plan.spec.row_id,
                    "split_identity": plan.spec.split_identity,
                    "package_id": plan.spec.package_id,
                    "repository_id": plan.spec.repository_id,
                    "group_id": plan.spec.group_id,
                    "family": plan.spec.family,
                    "source_path": str(plan.resolved_path),
                    "source_sha256": plan.spec.source_sha256,
                    "context_sha256": plan.spec.context_sha256,
                    "target_sha256": plan.spec.target_sha256,
                    "source_bytes": len(source_bytes),
                }
            )
        manifest_body: dict[str, Any] = {
            "schema": GUARD_SCHEMA,
            "status": "source_admission_verified",
            "observed_at": _timestamp_text(self.observed_at),
            "freeze_gate": {
                "required_at": FREEZE_AT_TEXT,
                "observed_at_or_after": True,
                "explicit_receipt_required": True,
            },
            "freeze_receipt": {
                "sha256": sha256_json(self.freeze_receipt),
                "validated": freeze,
            },
            "semantic_receipt": {
                "sha256": sha256_json(self.semantic_receipt),
                "validated": semantic,
            },
            "source_allowlist": {
                "roots": sorted({str(plan.root) for plan in planned}),
                "positive": True,
                "symlinks_rejected": True,
            },
            "split_disjointness": {
                "train_and_dev_memberships_checked": True,
                "final_case_count": len(planned),
                "final_row_ids": [plan.spec.row_id for plan in planned],
                "final_split_identities": [plan.spec.split_identity for plan in planned],
            },
            "source_read_policy": {
                "preflight_before_bytes": True,
                "openat_directory_walk": True,
                "no_follow_symlinks": True,
                "regular_file_required": True,
                "expected_sha256_required": True,
            },
            "source_records": source_records,
            "final_content_opened": False,
            "manifest_local_attestation_only": True,
        }
        manifest_body["manifest_body_sha256"] = sha256_json(manifest_body)
        payload = (canonical_json(manifest_body) + "\n").encode("utf-8", "surrogatepass")
        _write_immutable_json(output, payload)
        return AdmissionResult(
            manifest=MappingProxyType(manifest_body),
            output_path=output,
            source_bytes=MappingProxyType(source_payloads),
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def stat_is_regular(mode: int) -> bool:
    return (mode & 0o170000) == 0o100000


def stat_is_directory(mode: int) -> bool:
    return (mode & 0o170000) == 0o040000


def same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and stat_is_regular(first.st_mode)
        and stat_is_regular(second.st_mode)
    )


def _write_immutable_json(path: Path, payload: bytes) -> None:
    """Create a durable output without replace semantics or error swallowing."""
    parent = path.parent
    parent_fd = -1
    temporary_path: Path | None = None
    temporary_fd = -1
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        temporary_path = parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temporary_fd = os.open(
            temporary_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError("short write while creating admission manifest")
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        os.link(temporary_path, path, follow_symlinks=False)
        os.fsync(parent_fd)
        os.unlink(temporary_path)
        temporary_path = None
        os.fsync(parent_fd)
    except AdmissionOutputError:
        raise
    except FileExistsError as exc:
        raise AdmissionOutputError("output_exists", str(path)) from exc
    except OSError as exc:
        raise AdmissionOutputError("durable_output_failed", str(exc)) from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        if parent_fd >= 0:
            os.close(parent_fd)


def admit_source_cases(
    cases: Sequence[SourceCaseSpec | Mapping[str, Any]],
    *,
    output_path: Path,
    allowed_source_roots: Sequence[Path],
    train_identities: Mapping[str, Iterable[str]] | Iterable[str],
    dev_identities: Mapping[str, Iterable[str]] | Iterable[str],
    freeze_receipt: Mapping[str, Any],
    semantic_receipt: Mapping[str, Any],
    expected_weights_sha256: str,
    expected_harness_sha256: str,
    expected_source_lock_sha256: str,
    observed_at: _datetime.datetime | str,
    before_source_read: BeforeSourceRead | None = None,
) -> AdmissionResult:
    """Functional wrapper for callers that do not need to retain the guard."""
    guard = AdmissionGuard(
        allowed_source_roots=allowed_source_roots,
        train_identities=train_identities,
        dev_identities=dev_identities,
        freeze_receipt=freeze_receipt,
        semantic_receipt=semantic_receipt,
        expected_weights_sha256=expected_weights_sha256,
        expected_harness_sha256=expected_harness_sha256,
        expected_source_lock_sha256=expected_source_lock_sha256,
        observed_at=observed_at,
    )
    return guard.admit(
        cases,
        output_path=output_path,
        before_source_read=before_source_read,
    )


def _load_json(path: Path) -> Any:  # noqa: ANN401
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionMetadataError("json_metadata_read_failed", str(path)) from exc


def _cli_cases(value: Any) -> list[Mapping[str, Any]]:  # noqa: ANN401
    if isinstance(value, Mapping):
        value = value.get("cases")
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise AdmissionMetadataError("cases_json_list_required")
    if any(not isinstance(item, Mapping) for item in value):
        raise AdmissionMetadataError("cases_json_item_mapping_required")
    return list(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of case metadata")
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--semantic-receipt", type=Path, required=True)
    parser.add_argument("--train-identities", type=Path, required=True)
    parser.add_argument("--dev-identities", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, action="append", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = admit_source_cases(
            _cli_cases(_load_json(args.cases)),
            output_path=args.output,
            allowed_source_roots=args.allowed_root,
            train_identities=_load_json(args.train_identities),
            dev_identities=_load_json(args.dev_identities),
            freeze_receipt=_load_json(args.freeze_receipt),
            semantic_receipt=_load_json(args.semantic_receipt),
            expected_weights_sha256=args.weights_sha256,
            expected_harness_sha256=args.harness_sha256,
            expected_source_lock_sha256=args.source_lock_sha256,
            observed_at=args.observed_at,
        )
    except AdmissionGuardError as exc:
        print(f"admission rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.manifest["status"],
                "case_count": len(result.source_bytes),
                "output": str(result.output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
