#!/usr/bin/env python3
"""Fail-closed PRM-03 data loading for the sustained RL runner.

The canonical PRM-03 training row deliberately contains the pre-tokenized
prompt/target sequence but not its PromptContext.  RL needs the context to
interpret a full-copy no-op, so this module joins rows to the separately
hashed, pre-edit-only RL-02 sidecar.  The sidecar carries source identity and
selection geometry out of band for each row; rows may point at different static
source documents.  A legacy context-manifest shape remains readable for old
CPU fixtures.  This module never discovers data, calls an admission routine,
or reconstructs a prompt from target text.

The module is framework-free at import time.  It is safe to use from a CPU
preflight before importing torch, Transformers, Unsloth, or TRL.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence



from sepalith.training.rl.campaign_rl_buffer import RewardBufferIndex

from sepalith.campaign_protocol import (
    BOS_ID,
    EOS_ID,
    RENDERER_ID,
    TOKENIZATION_POLICY,
    VOCAB_SIZE,
    PromptContext,
    ProtocolError,
    render_prompt,
    validate_training_row,
)


DATA_SCHEMA_VERSION = "sepalith.prm07.rl-train-data.v1"
CONTEXT_SCHEMA_VERSION = "sepalith.prm03.context-manifest.v1"
SELECTED_IDS_SCHEMA_VERSION = "sepalith.prm07.selected-train-ids.v1"
PROMPT_MAX_TOKENS = 2_048
COMPLETION_MAX_TOKENS = 192
CONTEXT_MAX_TOKENS = PROMPT_MAX_TOKENS + COMPLETION_MAX_TOKENS
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RLDataError(ValueError):
    """An explicit RL input failed closed validation."""


@dataclass(frozen=True)
class ContextCapture:
    """The immutable source identity attached to one pre-edit context."""

    uri: str
    version: int
    content_sha256: str
    source_snapshot_sha256: str

    @classmethod
    def from_mapping(cls, value: object, *, row_id: str) -> "ContextCapture":
        if not isinstance(value, Mapping) or set(value) != {
            "uri", "version", "content_sha256", "source_snapshot_sha256"
        }:
            raise RLDataError(f"context {row_id} capture fields must be explicit")
        uri = value["uri"]
        version = value["version"]
        content_sha256 = value["content_sha256"]
        source_snapshot_sha256 = value["source_snapshot_sha256"]
        if not isinstance(uri, str) or not uri:
            raise RLDataError(f"context {row_id} capture URI is invalid")
        if type(version) is not int or version < 0:
            raise RLDataError(f"context {row_id} capture version is invalid")
        for name, digest in (
            ("content_sha256", content_sha256),
            ("source_snapshot_sha256", source_snapshot_sha256),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise RLDataError(f"context {row_id} capture {name} is invalid")
        return cls(uri, version, content_sha256, source_snapshot_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "version": self.version,
            "content_sha256": self.content_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
        }


SIDECAR_ROW_KEYS = frozenset({
    "row_id", "context", "source_identity", "selection_geometry", "family",
    "package_id", "split", "prompt_sha256", "context_has_target_or_reward_keys",
    "offline_static_source",
})
SIDECAR_SOURCE_IDENTITY_KEYS = frozenset({
    "candidate_file", "candidate_file_sha256", "candidate_line",
    "registry_provenance_id", "registry_provenance_decision", "group_id",
    "package_id", "source_ref", "source_provenance",
})
SIDECAR_GEOMETRY_KEYS = frozenset({
    "document_sha256", "availability", "policy_id", "policy_id_combined",
    "budget_utf16_units", "used_utf16_units", "required_utf16_units",
    "overflow", "required_overflow", "spans", "region", "context_range",
    "document_version_policy",
})
ALLOWED_SIDECAR_AVAILABILITY = frozenset({"full_snapshot", "source_builder_window"})


@dataclass(frozen=True)
class RLTrainRecord:
    """One validated row with an exact stored prompt and pre-edit context."""

    row: Mapping[str, Any]
    context: PromptContext
    capture: ContextCapture
    # These are emitted by RL-02 outside PromptContext.  Keeping them on the
    # record prevents the data loader from silently dropping the source and
    # range evidence that made a sidecar row eligible.
    source_identity: Mapping[str, Any] | None = None
    selection_geometry: Mapping[str, Any] | None = None
    reward_buffer: Mapping[str, Any] | None = None

    @property
    def row_id(self) -> str:
        return str(self.row["id"])

    @property
    def prompt_ids(self) -> tuple[int, ...]:
        return tuple(self.row["input_ids"][: self.row["target_start"]])

    def envelope(self) -> dict[str, Any]:
        """Return the list-of-dicts shape consumed by TRL's identity collator."""
        value = {
            "prompt": {
                "text": self.row["prompt_text"],
                "ids": list(self.prompt_ids),
            },
            "context": self.context.to_dict(),
            "context_capture": self.capture.to_dict(),
            "target_operation": self.row["target_operation"],
            "target_body_text": self.row["target_body_text"],
            "family": self.row["family"],
            "package_id": self.row["package_id"],
            "id": self.row_id,
            "split": self.row["split"],
        }
        if self.source_identity is not None:
            value["source_identity"] = dict(self.source_identity)
        if self.selection_geometry is not None:
            value["selection_geometry"] = dict(self.selection_geometry)
        if self.reward_buffer is not None:
            value["reward_buffer"] = dict(self.reward_buffer)
        return value


@dataclass(frozen=True)
class RLDataManifest:
    """Byte identities and selection facts carried into the trainer identity."""

    rows_path: str
    rows_sha256: str
    context_path: str
    context_sha256: str
    selected_ids_path: str
    selected_ids_sha256: str
    selected_ids: tuple[str, ...]
    row_count: int
    # ``context_sha256`` is retained as the recipe field for compatibility;
    # for the RL-02 shape it is the complete sidecar artifact hash.  A single
    # source snapshot cannot represent a sidecar whose rows come from many
    # static source files, so the old field is optional and only populated for
    # the legacy context-manifest shape.
    context_snapshot_sha256: str | None
    admission_status: str
    sidecar_artifact_sha256: str
    row_identities: tuple[Mapping[str, Any], ...]
    reward_buffer_manifest_sha256: str
    reward_buffer_sidecar_sha256: str

    @property
    def row_identity_sha256(self) -> str:
        payload = json.dumps(
            list(self.row_identities), separators=(",", ":"),
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def ordered_ids_sha256(self) -> str:
        payload = json.dumps(list(self.selected_ids), separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_identity(self) -> dict[str, Any]:
        value = {
            "rows_sha256": self.rows_sha256,
            "context_sha256": self.context_sha256,
            "sidecar_artifact_sha256": self.sidecar_artifact_sha256,
            "selected_ids_sha256": self.selected_ids_sha256,
            "ordered_ids_sha256": self.ordered_ids_sha256,
            "selected_ids": list(self.selected_ids),
            "row_count": self.row_count,
            "split": "train",
            "admission_status": self.admission_status,
            "row_identity_sha256": self.row_identity_sha256,
            "row_identities": list(self.row_identities),
            "reward_buffer_manifest_sha256": self.reward_buffer_manifest_sha256,
            "reward_buffer_sidecar_sha256": self.reward_buffer_sidecar_sha256,
        }
        if self.context_snapshot_sha256 is not None:
            # Compatibility metadata for the old single-context-manifest
            # shape.  RL-02 sidecars intentionally omit this global field.
            value["context_snapshot_sha256"] = self.context_snapshot_sha256
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RLDataError(f"cannot read input file {path}: {error}") from error
    return digest.hexdigest()


def _verified_path(path: Path, expected_sha256: str, label: str) -> tuple[Path, str]:
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise RLDataError(f"{label} must be an absolute file: {path}")
    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise RLDataError(f"{label} SHA256 is invalid")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RLDataError(f"{label} SHA256 mismatch: {path}")
    return path, actual


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise RLDataError(f"cannot open {label}: {path}") from error
    rows: list[dict[str, Any]] = []
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise RLDataError(f"{label} has a blank line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RLDataError(f"{label} line {line_number} is not JSON") from error
            if not isinstance(value, dict):
                raise RLDataError(f"{label} line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise RLDataError(f"{label} is empty")
    return rows


def load_selected_ids(path: Path, expected_sha256: str) -> tuple[str, ...]:
    path, _ = _verified_path(path, expected_sha256, "selected_ids")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RLDataError(f"selected_ids is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "split", "row_ids"}:
        raise RLDataError("selected_ids must contain exactly schema_version, split, row_ids")
    if value["schema_version"] != SELECTED_IDS_SCHEMA_VERSION or value["split"] != "train":
        raise RLDataError("selected_ids schema or split mismatch")
    row_ids = value["row_ids"]
    if not isinstance(row_ids, list) or not row_ids:
        raise RLDataError("selected_ids.row_ids must be a nonempty array")
    if any(type(row_id) is not str or not row_id for row_id in row_ids):
        raise RLDataError("selected_ids.row_ids must contain nonempty strings")
    if len(set(row_ids)) != len(row_ids):
        raise RLDataError("selected_ids.row_ids must be unique")
    return tuple(row_ids)


def _validate_prompt_prefix(row: Mapping[str, Any]) -> tuple[int, ...]:
    values = row["input_ids"]
    start = row["target_start"]
    prompt = tuple(values[:start])
    if not prompt or prompt[0] != BOS_ID:
        raise RLDataError(f"row {row['id']} prompt lacks manual BOS {BOS_ID}")
    if len(prompt) > PROMPT_MAX_TOKENS:
        raise RLDataError(f"row {row['id']} prompt exceeds {PROMPT_MAX_TOKENS} IDs")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in prompt):
        raise RLDataError(f"row {row['id']} prompt has an invalid ID")
    if prompt.count(BOS_ID) != 1 or EOS_ID in prompt:
        raise RLDataError(f"row {row['id']} prompt violates one-BOS/no-EOS geometry")
    if row["target_start"] != row["prompt_token_count"] + 1:
        raise RLDataError(f"row {row['id']} target_start disagrees with prompt_token_count")
    return prompt


def _validate_context_record(value: Mapping[str, Any]) -> tuple[str, PromptContext, ContextCapture]:
    expected = {"schema_version", "id", "context", "capture"}
    if set(value) != expected:
        raise RLDataError("context record fields must be exactly schema_version, id, context, capture")
    row_id = value["id"]
    if not isinstance(row_id, str) or not row_id:
        raise RLDataError("context record ID is invalid")
    if value["schema_version"] != CONTEXT_SCHEMA_VERSION:
        raise RLDataError(f"context {row_id} schema mismatch")
    try:
        context = PromptContext.from_mapping(value["context"])
    except (ProtocolError, TypeError, ValueError) as error:
        raise RLDataError(f"context {row_id} is not a valid PromptContext: {error}") from error
    capture = ContextCapture.from_mapping(value["capture"], row_id=row_id)
    replacement = context.replacement_range
    if (capture.uri != replacement.uri or capture.version != replacement.document_version
            or capture.content_sha256 != replacement.content_sha256):
        raise RLDataError(f"context {row_id} capture does not match replacement range identity")
    # The canonical context constructor has already checked URI/version/range
    # geometry.  Requiring an exact source snapshot digest keeps the context
    # record tied to the immutable data panel rather than an old self-consistent
    # record from another source snapshot.
    return row_id, context, capture


def _validate_sidecar_source_identity(
    value: object, *, row_id: str,
) -> dict[str, Any]:
    """Validate the out-of-band source identity emitted by RL-02.

    The sidecar is already byte-hashed as a whole by ``load_training_records``.
    This check binds the row to the exact candidate/provenance references that
    RL-02 selected, while leaving each row free to point at a different static
    source document.
    """
    if not isinstance(value, Mapping) or set(value) != SIDECAR_SOURCE_IDENTITY_KEYS:
        raise RLDataError(f"sidecar row {row_id} source_identity fields are incomplete")
    candidate_file = value["candidate_file"]
    if not isinstance(candidate_file, str) or not Path(candidate_file).is_absolute():
        raise RLDataError(f"sidecar row {row_id} candidate file identity is invalid")
    candidate_sha = value["candidate_file_sha256"]
    if not isinstance(candidate_sha, str) or _SHA256_RE.fullmatch(candidate_sha) is None:
        raise RLDataError(f"sidecar row {row_id} candidate file hash is invalid")
    candidate_line = value["candidate_line"]
    if type(candidate_line) is not int or candidate_line < 1:
        raise RLDataError(f"sidecar row {row_id} candidate line is invalid")
    if (not isinstance(value["registry_provenance_id"], str)
            or value["registry_provenance_id"] != row_id):
        raise RLDataError(f"sidecar row {row_id} provenance ID is invalid")
    if value["registry_provenance_decision"] != "admitted":
        raise RLDataError(f"sidecar row {row_id} provenance is not admitted")
    if not isinstance(value["group_id"], str) or not value["group_id"]:
        raise RLDataError(f"sidecar row {row_id} group identity is invalid")
    if value["package_id"] is not None and not isinstance(value["package_id"], str):
        raise RLDataError(f"sidecar row {row_id} source package identity is invalid")

    for name in ("source_ref", "source_provenance"):
        nested = value[name]
        if not isinstance(nested, Mapping):
            raise RLDataError(f"sidecar row {row_id} {name} is missing")
        if "row_id" in nested and nested["row_id"] != row_id:
            raise RLDataError(f"sidecar row {row_id} {name} row ID mismatch")
        if "group_id" in nested and nested["group_id"] != value["group_id"]:
            raise RLDataError(f"sidecar row {row_id} {name} group mismatch")
        hash_names = ("source_sha256", "raw_line_sha256") if name == "source_ref" else (
            hash_name for hash_name in ("source_sha256", "raw_line_sha256") if hash_name in nested
        )
        for hash_name in hash_names:
            digest = nested.get(hash_name)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise RLDataError(f"sidecar row {row_id} {name}.{hash_name} is invalid")
    source_ref = value["source_ref"]
    source_provenance = value["source_provenance"]
    if source_ref.get("row_id") != row_id or source_ref.get("group_id") != value["group_id"]:
        raise RLDataError(f"sidecar row {row_id} source reference identity is incomplete")
    for name in ("file", "line", "split"):
        if name in source_provenance and source_provenance[name] != source_ref.get(name):
            raise RLDataError(f"sidecar row {row_id} source provenance {name} differs from source reference")
    for hash_name in ("source_sha256", "raw_line_sha256"):
        if (hash_name in source_provenance
                and source_provenance[hash_name] != source_ref.get(hash_name)):
            raise RLDataError(f"sidecar row {row_id} source provenance {hash_name} differs from source reference")
    if source_ref.get("split") != "train_group":
        raise RLDataError(f"sidecar row {row_id} source_ref split is invalid")
    source_file = source_ref.get("file")
    if not isinstance(source_file, str) or not Path(source_file).is_absolute():
        raise RLDataError(f"sidecar row {row_id} source_ref file is invalid")
    if type(source_ref.get("line")) is not int or source_ref["line"] < 1:
        raise RLDataError(f"sidecar row {row_id} source_ref line is invalid")
    if source_ref.get("family") is not None and not isinstance(source_ref.get("family"), str):
        raise RLDataError(f"sidecar row {row_id} source_ref family is invalid")
    source_name = source_ref.get("source") or source_ref.get("source_variant") or source_ref.get("family")
    if not isinstance(source_name, str) or not source_name:
        raise RLDataError(f"sidecar row {row_id} source_ref name is missing")
    if source_ref.get("package_id") not in (None, value["package_id"]):
        raise RLDataError(f"sidecar row {row_id} source_ref package differs from sidecar identity")
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _sha256_values(value: object) -> set[str]:
    """Collect declared hashes from nested sidecar provenance evidence."""
    values: set[str] = set()
    if isinstance(value, Mapping):
        for name, child in value.items():
            if isinstance(child, str) and str(name).endswith("sha256"):
                values.add(child)
            else:
                values.update(_sha256_values(child))
    elif isinstance(value, list):
        for child in value:
            values.update(_sha256_values(child))
    return values


def _validate_sidecar_geometry(
    value: object, *, row_id: str, context: PromptContext,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SIDECAR_GEOMETRY_KEYS:
        raise RLDataError(f"sidecar row {row_id} selection_geometry fields are incomplete")
    replacement = context.replacement_range
    if value["document_sha256"] != replacement.content_sha256:
        raise RLDataError(f"sidecar row {row_id} document geometry hash differs from context")
    if value["availability"] not in ALLOWED_SIDECAR_AVAILABILITY:
        raise RLDataError(f"sidecar row {row_id} source availability is invalid")
    if value["overflow"] is not False or value["required_overflow"] is not False:
        raise RLDataError(f"sidecar row {row_id} selection geometry overflow is not closed")
    spans = value["spans"]
    if not isinstance(spans, list):
        raise RLDataError(f"sidecar row {row_id} selection spans are missing")
    region_spans = [
        span for span in spans
        if isinstance(span, Mapping) and span.get("kind") == "region"
    ]
    if len(region_spans) != 1:
        raise RLDataError(f"sidecar row {row_id} selection region span is invalid")
    span = region_spans[0]
    if (span.get("start_line") != replacement.start.line
            or span.get("end_line") != replacement.end.line):
        raise RLDataError(f"sidecar row {row_id} selection region lines differ from context")
    if value["context_range"] != replacement.to_dict():
        raise RLDataError(f"sidecar row {row_id} context range differs from geometry")
    if value["document_version_policy"] != (
        "offline_static_source; zero is valid; no live-editor freshness asserted"
    ):
        raise RLDataError(f"sidecar row {row_id} document version policy is invalid")
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _validate_sidecar_record(
    value: Mapping[str, Any], row: Mapping[str, Any],
) -> tuple[str, PromptContext, ContextCapture, dict[str, Any], dict[str, Any]]:
    row_id = str(row["id"])
    if set(value) != SIDECAR_ROW_KEYS:
        raise RLDataError(f"sidecar row {row_id} fields do not match RL-02 output")
    if value["row_id"] != row_id:
        raise RLDataError(f"sidecar row {row_id} ID mismatch")
    if value["split"] != "train" or row.get("split") != "train":
        raise RLDataError(f"sidecar row {row_id} is not a train row")
    if value["family"] != row.get("family") or value["package_id"] != row.get("package_id"):
        raise RLDataError(f"sidecar row {row_id} row identity differs from admitted row")
    if value["context_has_target_or_reward_keys"] is not False:
        raise RLDataError(f"sidecar row {row_id} context label gate is not closed")
    if value["offline_static_source"] is not True:
        raise RLDataError(f"sidecar row {row_id} static source gate is not closed")
    prompt_sha = value["prompt_sha256"]
    if (not isinstance(prompt_sha, str) or _SHA256_RE.fullmatch(prompt_sha) is None
            or prompt_sha != hashlib.sha256(str(row["prompt_text"]).encode("utf-8")).hexdigest()):
        raise RLDataError(f"sidecar row {row_id} prompt hash differs from admitted row")
    try:
        context = PromptContext.from_mapping(value["context"])
        rendered = render_prompt(context)
    except (ProtocolError, TypeError, ValueError) as error:
        raise RLDataError(f"sidecar row {row_id} context is invalid: {error}") from error
    if rendered != row["prompt_text"]:
        raise RLDataError(f"sidecar row {row_id} prompt differs from its context")
    source_identity = _validate_sidecar_source_identity(
        value["source_identity"], row_id=row_id,
    )
    if source_identity["package_id"] not in (None, row.get("package_id")):
        raise RLDataError(f"sidecar row {row_id} source package differs from admitted row")
    source_ref = source_identity["source_ref"]
    if source_ref.get("family") not in (None, row.get("family")):
        raise RLDataError(f"sidecar row {row_id} source family differs from admitted row")
    geometry = _validate_sidecar_geometry(
        value["selection_geometry"], row_id=row_id, context=context,
    )
    if context.replacement_range.content_sha256 not in _sha256_values(source_identity["source_provenance"]):
        raise RLDataError(f"sidecar row {row_id} context hash is absent from source provenance")
    # Sidecar rows do not carry the legacy capture wrapper.  Derive a local
    # compatibility capture from the immutable range plus this row's static
    # source hash; it is intentionally kept per row and is never compared
    # across source documents.
    capture = ContextCapture(
        uri=context.replacement_range.uri,
        version=context.replacement_range.document_version,
        content_sha256=context.replacement_range.content_sha256,
        source_snapshot_sha256=source_ref["source_sha256"],
    )
    return row_id, context, capture, source_identity, geometry


def _validate_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[int, ...]]:
    try:
        normalized = validate_training_row(row)
    except (ProtocolError, TypeError, ValueError, KeyError) as error:
        raise RLDataError(f"row failed PRM-03 validation: {error}") from error
    if normalized["split"] != "train":
        raise RLDataError(f"row {normalized['id']} is not a train row")
    if normalized["renderer_id"] != RENDERER_ID or normalized["tokenization_policy"] != TOKENIZATION_POLICY:
        raise RLDataError(f"row {normalized['id']} renderer/tokenization identity mismatch")
    return normalized, _validate_prompt_prefix(normalized)


def _validate_semantics(row: Mapping[str, Any], context: PromptContext) -> None:
    operation = row["target_operation"]
    body = row["target_body_text"]
    if operation == "no_op":
        if body != "[NO_EDIT]":
            raise RLDataError(f"row {row['id']} no-op geometry is inconsistent")
    elif operation == "delete":
        if not context.region_old or body:
            raise RLDataError(f"row {row['id']} delete geometry is inconsistent")
    elif operation == "replace":
        if not body or tuple(body.split("\n")) == tuple(context.region_old):
            raise RLDataError(f"row {row['id']} replace geometry is inconsistent")
    else:  # validate_training_row currently catches this; keep this local guard explicit.
        raise RLDataError(f"row {row['id']} operation is invalid")


def load_training_records(
    rows_path: Path,
    rows_sha256: str,
    context_path: Path,
    context_sha256: str,
    selected_ids_path: Path,
    selected_ids_sha256: str,
    *,
    admission_status: str = "admitted",
    expected_context_snapshot_sha256: str | None = None,
    reward_buffer_manifest_path: Path,
    reward_buffer_manifest_sha256: str,
) -> tuple[list[RLTrainRecord], RLDataManifest]:
    """Load exactly the hashed, selected train rows and their pre-edit contexts.

    The admission status is a recipe assertion.  This function does not call or
    infer an admission process, and candidate-only envelopes fail canonical row
    validation instead of being upgraded implicitly.
    """
    if admission_status != "admitted":
        raise RLDataError("RL training requires an explicit admitted data status")
    rows_file, rows_actual = _verified_path(rows_path, rows_sha256, "rows")
    context_file, context_actual = _verified_path(context_path, context_sha256, "context")
    selected_file, selected_actual = _verified_path(selected_ids_path, selected_ids_sha256, "selected_ids")
    selected_ids = load_selected_ids(selected_file, selected_actual)
    buffer_index = RewardBufferIndex.load(
        reward_buffer_manifest_path, reward_buffer_manifest_sha256, selected_ids,
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    for raw in _jsonl(rows_file, "rows"):
        row, _prompt = _validate_row(raw)
        row_id = row["id"]
        if row_id in rows_by_id:
            raise RLDataError(f"rows repeat ID {row_id}")
        rows_by_id[row_id] = row
    missing = [row_id for row_id in selected_ids if row_id not in rows_by_id]
    if missing:
        raise RLDataError(f"selected IDs are absent from rows: {missing[:4]}")

    raw_contexts = _jsonl(context_file, "context")
    # RL-02's sidecar is the canonical integration shape.  Detect it by its
    # explicit row_id/geometry/source-identity fields.  Keep the older
    # schema_version/id/capture shape readable for already-authored CPU
    # fixtures; only that legacy shape has a common snapshot assertion.
    sidecar_mode = all("row_id" in value for value in raw_contexts)
    legacy_mode = all("schema_version" in value for value in raw_contexts)
    if sidecar_mode == legacy_mode:
        raise RLDataError("context input must contain one consistent RL-02 sidecar or legacy manifest shape")

    records: list[RLTrainRecord] = []
    row_identities: list[Mapping[str, Any]] = []
    snapshot_digests: set[str] = set()
    if sidecar_mode:
        sidecar_by_id: dict[str, Mapping[str, Any]] = {}
        for raw in raw_contexts:
            row_id = raw.get("row_id")
            if not isinstance(row_id, str) or not row_id:
                raise RLDataError("sidecar row ID is invalid")
            if row_id in sidecar_by_id:
                raise RLDataError(f"sidecar repeats ID {row_id}")
            sidecar_by_id[row_id] = raw
        missing_context = [row_id for row_id in selected_ids if row_id not in sidecar_by_id]
        if missing_context:
            raise RLDataError(f"selected IDs are absent from RL-02 sidecar: {missing_context[:4]}")
        extra_context = sorted(set(sidecar_by_id) - set(selected_ids))
        if extra_context:
            raise RLDataError(f"RL-02 sidecar contains IDs outside selected order: {extra_context[:4]}")
        # A common source snapshot is not a valid identity for this shape:
        # each sidecar row can come from a different static source file.  The
        # complete sidecar artifact hash and ordered row identities are the
        # bindings used by RL recipe/checkpoint identity instead.
        if expected_context_snapshot_sha256 is not None:
            raise RLDataError("RL-02 sidecars use per-row source identities; a common source snapshot is unsupported")
        for row_id in selected_ids:
            row = rows_by_id[row_id]
            try:
                (_row_id, context, capture, source_identity, geometry) = _validate_sidecar_record(
                    sidecar_by_id[row_id], row,
                )
            except RLDataError:
                raise
            _validate_semantics(row, context)
            records.append(RLTrainRecord(
                row=row, context=context, capture=capture,
                source_identity=source_identity, selection_geometry=geometry,
                reward_buffer=buffer_index.envelope_for(row_id),
            ))
            row_identities.append({
                "id": row_id,
                "source_identity": source_identity,
                "selection_geometry": geometry,
            })
        snapshot = None
    else:
        contexts: dict[str, tuple[PromptContext, ContextCapture]] = {}
        for raw in raw_contexts:
            row_id, context, capture = _validate_context_record(raw)
            if row_id in contexts:
                raise RLDataError(f"context repeats ID {row_id}")
            contexts[row_id] = (context, capture)
        missing_context = [row_id for row_id in selected_ids if row_id not in contexts]
        if missing_context:
            raise RLDataError(f"selected IDs are absent from context manifest: {missing_context[:4]}")
        for row_id in selected_ids:
            row = rows_by_id[row_id]
            context, capture = contexts[row_id]
            if render_prompt(context) != row["prompt_text"]:
                raise RLDataError(f"row {row_id} prompt_text differs from its pre-edit context")
            _validate_semantics(row, context)
            if expected_context_snapshot_sha256 is not None and capture.source_snapshot_sha256 != expected_context_snapshot_sha256:
                raise RLDataError(f"context {row_id} source snapshot differs from the admitted data identity")
            snapshot_digests.add(capture.source_snapshot_sha256)
            records.append(RLTrainRecord(
                row=row, context=context, capture=capture,
                reward_buffer=buffer_index.envelope_for(row_id),
            ))
            row_identities.append({
                "id": row_id,
                "capture": capture.to_dict(),
            })
        if len(snapshot_digests) != 1:
            raise RLDataError("selected contexts span multiple source snapshots")
        snapshot = next(iter(snapshot_digests))
        if expected_context_snapshot_sha256 is not None and snapshot != expected_context_snapshot_sha256:
            raise RLDataError("context snapshot identity mismatch")
    manifest = RLDataManifest(
        rows_path=str(rows_file), rows_sha256=rows_actual,
        context_path=str(context_file), context_sha256=context_actual,
        selected_ids_path=str(selected_file), selected_ids_sha256=selected_actual,
        selected_ids=selected_ids, row_count=len(records),
        context_snapshot_sha256=snapshot, admission_status=admission_status,
        sidecar_artifact_sha256=context_actual,
        row_identities=tuple(row_identities),
        reward_buffer_manifest_sha256=reward_buffer_manifest_sha256,
        reward_buffer_sidecar_sha256=buffer_index.sidecar_sha256,
    )
    return records, manifest


def records_to_dataset_rows(records: Sequence[RLTrainRecord]) -> list[dict[str, Any]]:
    """Return independent JSON-like rows for TRL's identity data collator."""
    if not records:
        raise RLDataError("cannot construct an empty RL dataset")
    return [record.envelope() for record in records]
