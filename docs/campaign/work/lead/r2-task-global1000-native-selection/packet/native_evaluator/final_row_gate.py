#!/usr/bin/env python3
"""Final-only evaluator gate and callable preparation component.

The execution campaign evaluator is development-only.  This module provides
an explicit final-row handoff seam without weakening that reader: it validates
an externally frozen row manifest and the exact immutable canonical-row file
before content is opened, then returns a callable that may later receive an
already-loaded model and tokenizer.  It never loads either at import or gate
construction time.
"""
from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import final_binding as binding


EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PROTOCOL_PATH = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
PROTOCOL_SHA256 = "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156"
PROTOCOL_MODULE_NAME = "sepalith.campaign_protocol"
CAMPAIGN_EVAL_PATH = EXEC_ROOT / "experiments/training/campaign_eval.py"
CAMPAIGN_EVAL_SHA256 = "7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f"
FREEZE_AT = _datetime.datetime(2026, 9, 14, 10, 0, tzinfo=_datetime.timezone.utc)
FREEZE_AT_TEXT = "2026-09-14T10:00:00Z"

ROWS_SCHEMA = "dat08.final-evaluator-input.v1"
ROWS_STATUS = "prepared_final_evaluator_rows"
MANIFEST_SCHEMA = "dat08.final-evaluator-manifest.v1"
OUTPUT_SCHEMA = "dat08.final-quality-evaluation.v1"
RENDERER_ID = "zeta2-prm03-v1"
PROTOCOL_SCHEMA = "sepalith.prompt.prm03.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ROW_FIELDS = {
    "id", "split", "package_id", "family", "operation", "region_new", "context",
    "source_provenance", "renderer_id", "schema_version", "prompt_sha256", "target_sha256",
}
_REQUIRED_ARTIFACT_FIELDS = {
    "schema", "status", "final_admission_allowed", "source_artifact", "row_count", "rows",
    "rows_sha256", "publication_policy",
}
_REQUIRED_SOURCE_ARTIFACT_FIELDS = {
    "schema", "status", "authorization_id", "guard_manifest_sha256",
    "semantic_receipt_sha256", "freeze_receipt_sha256", "trusted_revisions",
}
_REQUIRED_MANIFEST_FIELDS = {
    "schema", "rows_artifact_sha256", "row_count", "case_ids", "source_artifact", "manifest_sha256",
}

_PROTOCOL: Any | None = None
_EVAL: Any | None = None


class FinalEvaluatorError(ValueError):
    """A final evaluator gate, row, or evaluation operation was rejected."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(code if detail is None else f"{code}:{detail}")
        self.code = code


def canonical_json(value: Any) -> str:  # noqa: ANN401
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:  # noqa: ANN401
    return hashlib.sha256(canonical_json(value).encode("utf-8", "surrogatepass")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: Any, field: str) -> str:  # noqa: ANN401
    if not isinstance(value, str) or not value:
        raise FinalEvaluatorError("text_required", field)
    return value


def _sha(value: Any, field: str) -> str:  # noqa: ANN401
    value = _text(value, field)
    if _SHA256.fullmatch(value) is None:
        raise FinalEvaluatorError("sha256_required", field)
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:  # noqa: ANN401
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FinalEvaluatorError("mapping_required", field)
    return value


def _parse_timestamp(value: Any, field: str) -> _datetime.datetime:
    value = _text(value, field)
    try:
        parsed = _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalEvaluatorError("timestamp_invalid", field) from exc
    if parsed.tzinfo is None:
        raise FinalEvaluatorError("timestamp_must_be_utc", field)
    return parsed.astimezone(_datetime.timezone.utc)


def _verify_canonical_protocol_module(module: Any) -> Any:  # noqa: ANN401
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str):
        raise FinalEvaluatorError("protocol_module_file_missing")
    try:
        if Path(module_path).resolve() != PROTOCOL_PATH.resolve():
            raise FinalEvaluatorError("protocol_module_path_mismatch")
        observed_sha256 = sha256_bytes(Path(module_path).read_bytes())
    except OSError as exc:
        raise FinalEvaluatorError("protocol_module_metadata_unavailable") from exc
    if observed_sha256 != PROTOCOL_SHA256:
        raise FinalEvaluatorError("protocol_module_revision_mismatch")
    return module


def _load_protocol() -> Any:  # noqa: ANN401
    global _PROTOCOL
    preloaded = sys.modules.get(PROTOCOL_MODULE_NAME)
    if _PROTOCOL is not None:
        if preloaded is not _PROTOCOL:
            raise FinalEvaluatorError("protocol_module_identity_mismatch")
        _verify_canonical_protocol_module(_PROTOCOL)
        return _PROTOCOL
    if not PROTOCOL_PATH.is_file() or sha256_bytes(PROTOCOL_PATH.read_bytes()) != PROTOCOL_SHA256:
        raise FinalEvaluatorError("protocol_revision_mismatch")
    if preloaded is not None:
        # Never replace a preloaded module.  Its resolved file and bytes must
        # already be exactly the pinned execution protocol.
        _PROTOCOL = _verify_canonical_protocol_module(preloaded)
        return _PROTOCOL
    package_root = str(PROTOCOL_PATH.parent.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    try:
        module = importlib.import_module(PROTOCOL_MODULE_NAME)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FinalEvaluatorError("protocol_import_failed") from exc
    # import_module installed the canonical object; verify it before exposing
    # it to the row reader or campaign_eval's classifier.
    if sys.modules.get(PROTOCOL_MODULE_NAME) is not module:
        raise FinalEvaluatorError("protocol_module_identity_mismatch")
    _PROTOCOL = _verify_canonical_protocol_module(module)
    return module


def _load_campaign_eval() -> Any:  # noqa: ANN401
    global _EVAL
    if _EVAL is not None:
        return _EVAL
    if not CAMPAIGN_EVAL_PATH.is_file() or sha256_bytes(CAMPAIGN_EVAL_PATH.read_bytes()) != CAMPAIGN_EVAL_SHA256:
        raise FinalEvaluatorError("campaign_eval_revision_mismatch")
    training_root = str(CAMPAIGN_EVAL_PATH.parent)
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    spec = importlib.util.spec_from_file_location("dat08_final_quality_campaign_eval", CAMPAIGN_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise FinalEvaluatorError("campaign_eval_import_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FinalEvaluatorError("campaign_eval_import_failed") from exc
    _EVAL = module
    return module


def _safe_metadata_path(path: Path) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute():
        raise FinalEvaluatorError("rows_path_must_be_absolute")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise FinalEvaluatorError("rows_path_has_parent_component")
    # This checks path metadata only; it never opens row content.
    for index in range(1, len(path.parts) + 1):
        component = Path(*path.parts[:index])
        try:
            if component.is_symlink():
                raise FinalEvaluatorError("rows_path_symlink_component")
        except OSError as exc:
            raise FinalEvaluatorError("rows_path_metadata_unavailable") from exc
    return path


def _validate_freeze(
    freeze_receipt: Mapping[str, Any],
    *,
    expected_weights_sha256: str,
    expected_harness_sha256: str,
    now: _datetime.datetime,
) -> dict[str, Any]:
    receipt = _mapping(freeze_receipt, "freeze_receipt")
    if receipt.get("status") != "frozen":
        raise FinalEvaluatorError("freeze_receipt_not_frozen")
    for field in ("weights_frozen", "harness_frozen", "final_access_unlocked"):
        if receipt.get(field) is not True:
            raise FinalEvaluatorError("freeze_authorization_missing", field)
    if _sha(expected_weights_sha256, "expected_weights_sha256") != _sha(receipt.get("weights_sha256"), "freeze_receipt.weights_sha256"):
        raise FinalEvaluatorError("weights_hash_not_pinned")
    if _sha(expected_harness_sha256, "expected_harness_sha256") != _sha(receipt.get("harness_sha256"), "freeze_receipt.harness_sha256"):
        raise FinalEvaluatorError("harness_hash_not_pinned")
    if now < FREEZE_AT:
        raise FinalEvaluatorError("freeze_time_not_reached", FREEZE_AT_TEXT)
    timestamps = {}
    for field in ("weights_frozen_at", "harness_frozen_at"):
        timestamp = _parse_timestamp(receipt.get(field), f"freeze_receipt.{field}")
        if timestamp > now:
            raise FinalEvaluatorError("freeze_receipt_after_observation", field)
        timestamps[field] = timestamp.isoformat().replace("+00:00", "Z")
    return {
        "status": "frozen",
        "weights_sha256": expected_weights_sha256,
        "harness_sha256": expected_harness_sha256,
        **timestamps,
    }


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    return sha256_json(body)


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    freeze_sha256: str,
) -> dict[str, Any]:
    manifest = _mapping(manifest, "final_rows_manifest")
    if set(manifest) != _REQUIRED_MANIFEST_FIELDS:
        raise FinalEvaluatorError("final_rows_manifest_fields_mismatch")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FinalEvaluatorError("final_rows_manifest_schema_mismatch")
    if _sha(manifest.get("manifest_sha256"), "manifest_sha256") != _sha(expected_manifest_sha256, "expected_manifest_sha256"):
        raise FinalEvaluatorError("final_rows_manifest_hash_mismatch")
    if _manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise FinalEvaluatorError("final_rows_manifest_self_hash_mismatch")
    _sha(manifest.get("rows_artifact_sha256"), "rows_artifact_sha256")
    if type(manifest.get("row_count")) is not int or manifest["row_count"] < 1:
        raise FinalEvaluatorError("final_rows_manifest_count_invalid")
    ids = manifest.get("case_ids")
    if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids):
        raise FinalEvaluatorError("final_rows_manifest_ids_invalid")
    if ids != sorted(set(ids)):
        raise FinalEvaluatorError("final_rows_manifest_ids_not_sorted_unique")
    source_artifact = _mapping(manifest.get("source_artifact"), "final_rows_manifest.source_artifact")
    if set(source_artifact) != _REQUIRED_SOURCE_ARTIFACT_FIELDS:
        raise FinalEvaluatorError("final_rows_manifest_source_artifact_fields_mismatch")
    if source_artifact.get("schema") != "dat08.evaluator-input.v1" or source_artifact.get("status") != "prepared_evaluator_input":
        raise FinalEvaluatorError("final_rows_manifest_source_artifact_schema_mismatch")
    _text(source_artifact.get("authorization_id"), "source_artifact.authorization_id")
    for field in ("guard_manifest_sha256", "semantic_receipt_sha256", "freeze_receipt_sha256"):
        _sha(source_artifact.get(field), f"source_artifact.{field}")
    _mapping(source_artifact.get("trusted_revisions"), "source_artifact.trusted_revisions")
    if source_artifact.get("freeze_receipt_sha256") != freeze_sha256:
        raise FinalEvaluatorError("final_rows_manifest_freeze_binding_mismatch")
    return copy.deepcopy(dict(manifest))


def _validate_row(
    row: Mapping[str, Any], protocol: Any, expected_source_artifact: Mapping[str, Any]
) -> dict[str, Any]:  # noqa: ANN401
    row = _mapping(row, "final_row")
    if set(row) != _REQUIRED_ROW_FIELDS:
        raise FinalEvaluatorError("final_row_fields_mismatch", str(row.get("id", "")))
    row_id = _text(row.get("id"), "final_row.id")
    if row.get("split") != "final":
        raise FinalEvaluatorError("final_row_split_mismatch", row_id)
    _text(row.get("package_id"), f"{row_id}.package_id")
    _text(row.get("family"), f"{row_id}.family")
    if row.get("renderer_id") != RENDERER_ID or row.get("schema_version") != PROTOCOL_SCHEMA:
        raise FinalEvaluatorError("final_row_protocol_identity_mismatch", row_id)
    operation = row.get("operation")
    if operation not in {"no_op", "replace", "delete"}:
        raise FinalEvaluatorError("final_row_operation_invalid", row_id)
    region_new = row.get("region_new")
    if not isinstance(region_new, list) or any(not isinstance(item, str) or "\n" in item or "\r" in item for item in region_new):
        raise FinalEvaluatorError("final_row_region_invalid", row_id)
    if operation == "replace" and (not region_new or region_new == [""]):
        raise FinalEvaluatorError("final_row_replace_empty", row_id)
    if operation == "delete" and region_new:
        raise FinalEvaluatorError("final_row_delete_nonempty", row_id)
    try:
        context = protocol.PromptContext.from_mapping(copy.deepcopy(row["context"]))
        prompt_text = protocol.render_prompt(context)
        target_text = protocol.serialize_target(operation, region_new)
        parsed = protocol.parse_output(target_text, context)
    except Exception as exc:
        raise FinalEvaluatorError("final_row_protocol_geometry_invalid", row_id) from exc
    if parsed.status != "accepted" or parsed.operation != operation:
        raise FinalEvaluatorError("final_row_target_roundtrip_invalid", row_id)
    if operation == "replace" and list(parsed.body) != region_new:
        raise FinalEvaluatorError("final_row_target_region_roundtrip_invalid", row_id)
    if operation == "delete" and list(parsed.body):
        raise FinalEvaluatorError("final_row_delete_roundtrip_invalid", row_id)
    if _sha(row.get("prompt_sha256"), f"{row_id}.prompt_sha256") != sha256_bytes(prompt_text.encode("utf-8", "surrogatepass")):
        raise FinalEvaluatorError("final_row_prompt_hash_mismatch", row_id)
    if _sha(row.get("target_sha256"), f"{row_id}.target_sha256") != sha256_json(region_new):
        raise FinalEvaluatorError("final_row_target_hash_mismatch", row_id)
    provenance = _mapping(row.get("source_provenance"), f"{row_id}.source_provenance")
    artifact = _mapping(provenance.get("artifact"), f"{row_id}.source_provenance.artifact")
    if dict(artifact) != dict(expected_source_artifact):
        raise FinalEvaluatorError("final_row_provenance_artifact_binding_mismatch", row_id)
    request = _mapping(provenance.get("request"), f"{row_id}.source_provenance.request")
    source_audit = _mapping(provenance.get("source_audit"), f"{row_id}.source_provenance.source_audit")
    if request.get("family") != row["family"] or request.get("package_id") != row["package_id"]:
        raise FinalEvaluatorError("final_row_provenance_identity_mismatch", row_id)
    source_sha = _sha(provenance.get("source_sha256"), f"{row_id}.source_provenance.source_sha256")
    if source_audit.get("source_sha256") != source_sha:
        raise FinalEvaluatorError("final_row_provenance_source_hash_mismatch", row_id)
    if not isinstance(provenance.get("constructed_case"), Mapping):
        raise FinalEvaluatorError("final_row_constructed_case_missing", row_id)
    return {
        "id": row_id,
        "split": "final",
        "package_id": row["package_id"],
        "family": row["family"],
        "operation": operation,
        "region_new": list(region_new),
        "context": context.to_dict(),
        "source_provenance": copy.deepcopy(dict(provenance)),
        "renderer_id": RENDERER_ID,
        "schema_version": PROTOCOL_SCHEMA,
        "prompt_sha256": row["prompt_sha256"],
        "target_sha256": row["target_sha256"],
    }


def _validate_rows_file(path: Path, manifest: Mapping[str, Any], protocol: Any) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:  # noqa: ANN401
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FinalEvaluatorError("final_rows_artifact_read_failed") from exc
    if sha256_bytes(raw) != manifest["rows_artifact_sha256"]:
        raise FinalEvaluatorError("final_rows_artifact_hash_mismatch")
    try:
        artifact = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalEvaluatorError("final_rows_artifact_json_invalid") from exc
    artifact = _mapping(artifact, "final_rows_artifact")
    if set(artifact) != _REQUIRED_ARTIFACT_FIELDS:
        raise FinalEvaluatorError("final_rows_artifact_fields_mismatch")
    if artifact.get("schema") != ROWS_SCHEMA or artifact.get("status") != ROWS_STATUS:
        raise FinalEvaluatorError("final_rows_artifact_schema_mismatch")
    if artifact.get("final_admission_allowed") is not False:
        raise FinalEvaluatorError("final_rows_admission_flag_invalid")
    if artifact.get("source_artifact") != manifest["source_artifact"]:
        raise FinalEvaluatorError("final_rows_source_artifact_binding_mismatch")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or type(artifact.get("row_count")) is not int or artifact["row_count"] != len(rows):
        raise FinalEvaluatorError("final_rows_count_mismatch")
    if artifact["row_count"] != manifest["row_count"]:
        raise FinalEvaluatorError("final_rows_manifest_count_mismatch")
    if _sha(artifact.get("rows_sha256"), "final_rows.rows_sha256") != sha256_json(rows):
        raise FinalEvaluatorError("final_rows_digest_mismatch")
    row_ids = [row.get("id") if isinstance(row, Mapping) else None for row in rows]
    if row_ids != manifest["case_ids"]:
        raise FinalEvaluatorError("final_rows_ids_mismatch")
    normalized = [_validate_row(row, protocol, manifest["source_artifact"]) for row in rows]
    return normalized, {
        "schema": artifact["schema"],
        "status": artifact["status"],
        "row_count": len(normalized),
        "rows_sha256": artifact["rows_sha256"],
        "artifact_sha256": manifest["rows_artifact_sha256"],
    }, raw


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{__import__('os').getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# Only unchanged validation helpers; no public clock override or HF model evaluator.
