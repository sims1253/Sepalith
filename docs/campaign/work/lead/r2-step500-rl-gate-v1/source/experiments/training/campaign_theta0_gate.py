#!/usr/bin/env python3
"""Bounded, label-free theta0 context diagnostic for the RL-03 gate.

The diagnostic deliberately lives outside the RL recipe identity.  It renders
the sixteen context variants from the reviewed RL-01 fixture, runs a greedy
bounded decode, and records protocol/terminal/cap/resource evidence.  The
standard arm is one fresh request per context with ``use_cache=True``.  A
small, four-case ``use_cache=False`` arm is a decode-KV-cache numerical
control; it is not a cross-request prefix-cache test.

No framework or model is imported at module import time.  The command first
verifies the fixture inputs, the accepted merged-SFT parent, the output mount,
the one-device occupancy guard, and (when required) a root admission receipt.
Only then does it load a model.  Every case is persisted atomically, including
partial failure output.  This file does not score counterfactual contexts and
does not mutate RL data or training state.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Iterator, Mapping, Sequence


EXECUTION_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SRC = EXECUTION_ROOT / "packages" / "sepalith" / "src"
if str(PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_SRC))

from sepalith.campaign_protocol import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    NATIVE_EOG_IDS,
    PromptContext,
    encode_prompt,
    parse_output,
    render_prompt,
    valid_generation_tokens,
)


SCHEMA_VERSION = "sepalith.campaign.rl03.theta0-gate.v1"
FIXTURE_SCHEMA_VERSION = "sepalith.campaign.rl01.theta0-gate-fixture.v1"
RESERVE_SECONDS = 60
MAX_DEADLINE_SECONDS = 1_200
COMPLETION_CAP = 192
PROMPT_CAP = 2_048
CONTEXT_CAP = 2_240
STANDARD_ARM = "fresh_request_use_cache_true"
CONTROL_ARM = "decode_kv_cache_parity_use_cache_false"
FORBIDDEN_CONTEXT_KEYS = frozenset(
    ("target", "target_text", "target_body", "target_operation", "region_new", "reward")
)
FRAMEWORK_MODULES = frozenset(("torch", "transformers", "trl", "unsloth", "datasets"))


class Theta0GateError(ValueError):
    """A required preflight or case contract failed closed."""


class Theta0Unsupported(Theta0GateError):
    """The host/runtime cannot prove a requested diagnostic arm."""


class Theta0Deadline(Theta0GateError):
    """The bounded diagnostic reached its soft deadline reserve."""


def sha256_file(path: Path, *, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(block_size), b""):
                digest.update(block)
    except OSError as error:
        raise Theta0GateError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def sha256_json(value: object, *, sort_keys: bool = True) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=sort_keys,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically persist a JSON receipt and flush its file and directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # The file is already durable on filesystems that do not support a
            # directory fsync.  Native-ext4 preflight remains authoritative.
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Theta0GateError(f"{name} must be an object")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Theta0GateError(f"{name} must be a lowercase SHA256")
    return value


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Theta0GateError(f"cannot read {name}: {path}") from error
    return _mapping(value, name)


def _visit_forbidden(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        overlap = FORBIDDEN_CONTEXT_KEYS.intersection(value)
        if overlap:
            raise Theta0GateError(f"context {label} contains target/reward keys: {sorted(overlap)}")
        for child in value.values():
            _visit_forbidden(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _visit_forbidden(child, label=label)


def _read_jsonl_by_id(path: Path, key: str, needed: set[str]) -> tuple[dict[str, Mapping[str, Any]], int]:
    found: dict[str, Mapping[str, Any]] = {}
    count = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise Theta0GateError(f"blank line in {path} at {line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Theta0GateError(f"invalid JSON in {path} at {line_number}") from error
                value = _mapping(value, f"{path}:{line_number}")
                count += 1
                row_id = value.get(key)
                if isinstance(row_id, str) and row_id in needed:
                    if row_id in found:
                        raise Theta0GateError(f"duplicate {key} {row_id} in {path}")
                    found[row_id] = value
    except OSError as error:
        raise Theta0GateError(f"cannot read {path}: {error}") from error
    return found, count


def _selected_ids(path: Path, expected_sha: str, expected_ordered_sha: str) -> list[str]:
    if sha256_file(path) != _sha(expected_sha, "selected IDs SHA256"):
        raise Theta0GateError(f"selected-ID file hash mismatch: {path}")
    value = _read_json(path, "selected IDs")
    if value.get("schema_version") != "sepalith.prm07.selected-train-ids.v1" or value.get("split") != "train":
        raise Theta0GateError("selected-ID schema or split mismatch")
    ids = value.get("row_ids")
    if (not isinstance(ids, list) or not ids or any(type(item) is not str for item in ids)
            or len(set(ids)) != len(ids)):
        raise Theta0GateError("selected IDs must be a unique nonempty string list")
    if sha256_json(ids, sort_keys=False) != _sha(expected_ordered_sha, "ordered IDs SHA256"):
        raise Theta0GateError("ordered selected-ID hash mismatch")
    return list(ids)


def _verify_source_identity(fixture: Mapping[str, Any], sidecar: Mapping[str, Any], row_id: str) -> None:
    source = _mapping(_mapping(sidecar, f"sidecar {row_id}").get("source_identity"), f"source identity {row_id}")
    source_ref = _mapping(source.get("source_ref"), f"source ref {row_id}")
    expected = _mapping(_mapping(fixture.get("source_identity_by_row"), "fixture source identities").get(row_id),
                        f"fixture source identity {row_id}")
    for actual_key, expected_key in (
        ("source_sha256", "source_sha256"),
        ("raw_line_sha256", "raw_line_sha256"),
        ("file", "source_file"),
        ("line", "source_line"),
    ):
        if source_ref.get(actual_key) != expected.get(expected_key):
            raise Theta0GateError(f"source identity mismatch for {row_id}: {actual_key}")
    if source.get("candidate_file_sha256") != expected.get("candidate_file_sha256"):
        raise Theta0GateError(f"candidate source hash mismatch for {row_id}")


def validate_fixture_inputs(fixture_path: Path) -> dict[str, Any]:
    """Verify hashed RL-02 inputs and return only the eight needed base rows."""
    fixture = _read_json(Path(fixture_path), "theta0 fixture")
    if fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise Theta0GateError("theta0 fixture schema mismatch")
    inputs = _mapping(fixture.get("inputs"), "fixture.inputs")
    rows_path = Path(inputs["rows_path"])
    sidecar_path = Path(inputs["sidecar_path"])
    selected_path = Path(inputs["selected_ids_path"])
    for path_key, sha_key in (("rows_path", "rows_sha256"), ("sidecar_path", "sidecar_sha256")):
        path = Path(inputs[path_key])
        expected = _sha(inputs[sha_key], f"fixture.inputs.{sha_key}")
        if sha256_file(path) != expected:
            raise Theta0GateError(f"fixture input hash mismatch: {path}")

    selected = _selected_ids(selected_path, inputs["selected_ids_sha256"], inputs["ordered_ids_sha256"])
    base_rows = fixture.get("base_rows")
    if not isinstance(base_rows, list) or len(base_rows) != 8:
        raise Theta0GateError("theta0 fixture must contain exactly eight base rows")
    base_ids = [row.get("id") if isinstance(row, Mapping) else None for row in base_rows]
    if any(not isinstance(row_id, str) for row_id in base_ids) or len(set(base_ids)) != 8:
        raise Theta0GateError("theta0 base IDs are not unique strings")
    variants = _mapping(fixture.get("variant_derivation"), "fixture.variant_derivation").get("variants")
    if not isinstance(variants, list) or len(variants) != 16:
        raise Theta0GateError("theta0 fixture must contain exactly sixteen variants")
    needed = set(base_ids)
    for variant in variants:
        variant = _mapping(variant, "fixture variant")
        source_ids = variant.get("source_row_ids")
        if not isinstance(source_ids, list) or any(row_id not in selected for row_id in source_ids):
            raise Theta0GateError(f"variant {variant.get('id')} references an unselected source row")
        needed.update(source_ids)

    rows, row_count = _read_jsonl_by_id(rows_path, "id", needed)
    sidecars, sidecar_count = _read_jsonl_by_id(sidecar_path, "row_id", needed)
    if row_count != inputs.get("rows_count") or sidecar_count != inputs.get("sidecar_count"):
        raise Theta0GateError(
            f"input line counts changed: rows={row_count}, sidecar={sidecar_count}"
        )
    if set(rows) != needed or set(sidecars) != needed:
        raise Theta0GateError("fixture IDs are missing from hashed RL-02 inputs")

    for base in base_rows:
        row_id = base["id"]
        row = rows[row_id]
        sidecar = sidecars[row_id]
        if sidecar.get("row_id") != row_id:
            raise Theta0GateError(f"sidecar row identity mismatch for {row_id}")
        if sidecar.get("context_has_target_or_reward_keys") is not False:
            raise Theta0GateError(f"sidecar context is not target/reward-free for {row_id}")
        _verify_source_identity(fixture, sidecar, row_id)
        context = _mapping(sidecar.get("context"), f"context {row_id}")
        _visit_forbidden(context, label=row_id)
        prompt_text = row.get("prompt_text")
        if not isinstance(prompt_text, str) or sha256_text(prompt_text) != base.get("prompt_sha256"):
            raise Theta0GateError(f"base prompt hash mismatch for {row_id}")
        if sidecar.get("prompt_sha256") != base.get("prompt_sha256"):
            raise Theta0GateError(f"sidecar prompt hash mismatch for {row_id}")

    # The fixture itself is diagnostic-only.  Refuse a future fixture that
    # accidentally starts carrying labels in the context or admission block.
    admission = _mapping(fixture.get("admission"), "fixture.admission")
    if admission.get("target_or_reward_fields_in_payload") is not False:
        raise Theta0GateError("fixture does not prove target/reward-free payload")
    if admission.get("counterfactuals") and "diagnostic_only" not in str(admission["counterfactuals"]):
        raise Theta0GateError("fixture counterfactual status is not diagnostic-only")
    return {
        "fixture": fixture,
        "rows": rows,
        "sidecars": sidecars,
        "selected_ids": selected,
        "base_ids": base_ids,
        "variant_defs": variants,
        "rows_count": row_count,
        "sidecar_count": sidecar_count,
    }


def _derive_context(sidecars: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Any],
                    donor_cycle: Mapping[str, Any]) -> dict[str, Any]:
    base_id = variant.get("base_row_id")
    if base_id not in sidecars:
        raise Theta0GateError(f"variant {variant.get('id')} base row is absent")
    context = deepcopy(_mapping(sidecars[base_id].get("context"), f"context {base_id}"))
    kind = variant.get("kind")
    if kind in ("missing_history", "strict_noop_control", "local_only_control"):
        context["history"] = []
    elif kind == "irrelevant_history":
        donor = variant.get("donor_row_id")
        if donor != donor_cycle.get(base_id) or donor not in sidecars:
            raise Theta0GateError(f"invalid history donor for {variant.get('id')}")
        context["history"] = deepcopy(sidecars[donor]["context"]["history"])
    elif kind != "evidence_supported":
        raise Theta0GateError(f"unknown context variant kind: {kind}")
    _visit_forbidden(context, label=str(variant.get("id")))
    return context


def build_variant_cases(prepared: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    """Reconstruct fixture fingerprints with the pinned renderer/tokenizer."""
    fixture = prepared["fixture"]
    contract = _mapping(fixture.get("contract"), "fixture.contract")
    donor_cycle = _mapping(_mapping(fixture["variant_derivation"], "variant derivation").get("donor_cycle"),
                           "variant donor cycle")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for variant_raw in prepared["variant_defs"]:
        variant = _mapping(variant_raw, "fixture variant")
        variant_id = variant.get("id")
        if not isinstance(variant_id, str) or variant_id in seen:
            raise Theta0GateError("variant IDs must be unique nonempty strings")
        seen.add(variant_id)
        context_mapping = _derive_context(prepared["sidecars"], variant, donor_cycle)
        context = PromptContext.from_mapping(context_mapping)
        prompt = render_prompt(context)
        prompt_ids = encode_prompt(context, tokenizer, include_bos=True)
        if sha256_json(context.to_dict()) != variant.get("context_sha256"):
            raise Theta0GateError(f"context hash mismatch for {variant_id}")
        if sha256_text(prompt) != variant.get("prompt_sha256"):
            raise Theta0GateError(f"prompt hash mismatch for {variant_id}")
        prompt_id_hash = hashlib.sha256(
            json.dumps(prompt_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if prompt_id_hash != variant.get("prompt_ids_sha256"):
            raise Theta0GateError(f"token-ID hash mismatch for {variant_id}")
        if (not prompt.endswith("\n") or not prompt_ids or prompt_ids[0] != BOS_ID
                or len(prompt_ids) != variant.get("prompt_tokens_with_bos")
                or len(prompt_ids) > PROMPT_CAP or len(prompt_ids) + COMPLETION_CAP > CONTEXT_CAP):
            raise Theta0GateError(f"prompt geometry mismatch for {variant_id}")
        base_id = variant["base_row_id"]
        row = prepared["rows"][base_id]
        prompt_text = row.get("prompt_text") if variant["kind"] == "evidence_supported" else prompt
        # For every diagnostic variant, the request prompt is the freshly
        # rendered context.  The stored base prompt is only an out-of-band
        # tokenizer parity check for the four supported rows.
        if variant["kind"] == "evidence_supported" and prompt_text != prompt:
            raise Theta0GateError(f"stored base prompt differs from reconstructed prompt for {base_id}")
        case = {
            "variant_id": variant_id,
            "base_row_id": base_id,
            "kind": variant["kind"],
            "context": context,
            "prompt": prompt,
            "prompt_ids": list(prompt_ids),
            "context_sha256": variant["context_sha256"],
            "prompt_sha256": variant["prompt_sha256"],
            "prompt_ids_sha256": variant["prompt_ids_sha256"],
            "prompt_tokens_with_bos": len(prompt_ids),
            "row": row,
            "fixture_contract": contract,
        }
        cases.append(case)
    return cases


def _verify_admission(path: Path | None, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Require a root receipt that explicitly binds the candidate hashes."""
    fixture = prepared["fixture"]
    if path is None:
        source_status = str(_mapping(fixture.get("admission"), "fixture.admission").get("source_status", ""))
        if "candidate_only" in source_status:
            raise Theta0Unsupported("explicit root RL-02 admission receipt is required; fixture remains candidate-only")
        raise Theta0Unsupported("root RL-02 admission receipt was not supplied")
    value = _read_json(Path(path), "root RL-02 admission receipt")
    status = str(value.get("status", "")).lower()
    if value.get("launch_admitted") is False or "candidate" in status:
        raise Theta0Unsupported("root admission receipt still marks RL-02 candidate-only")
    if value.get("launch_admitted") is not True and value.get("admission_status") != "admitted" \
            and status not in {"accepted", "admitted", "live_admitted", "rl_data_admitted"}:
        raise Theta0Unsupported("root admission receipt lacks explicit admitted status")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    required_hashes = (
        fixture["inputs"]["rows_sha256"], fixture["inputs"]["sidecar_sha256"],
        fixture["inputs"]["selected_ids_sha256"], fixture["inputs"]["ordered_ids_sha256"],
        fixture["inputs"]["row_identity_sha256"],
    )
    missing = [digest for digest in required_hashes if digest not in encoded]
    if missing:
        raise Theta0Unsupported("root admission receipt does not bind all fixture input hashes")
    return {"path": str(Path(path).resolve()), "status": value.get("status"),
            "sha256": sha256_file(Path(path)), "hashes_bound": len(required_hashes)}


def _verify_protocol_source(fixture: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(fixture["contract"], "fixture.contract")
    path = EXECUTION_ROOT / str(contract["protocol_path"])
    expected = _sha(contract["protocol_sha256"], "protocol SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise Theta0GateError(f"pinned protocol hash mismatch: {path}")
    return {"path": str(path), "sha256": actual}


def _verify_tokenizer_files(fixture: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(fixture["contract"], "fixture.contract")
    root = Path(contract["tokenizer_path"])
    files = {
        "tokenizer.json": contract["tokenizer_json_sha256"],
        "tokenizer_config.json": contract["tokenizer_config_sha256"],
    }
    observed = {}
    for name, expected in files.items():
        path = root / name
        actual = sha256_file(path)
        if actual != _sha(expected, f"tokenizer {name} SHA256"):
            raise Theta0GateError(f"pinned tokenizer file hash mismatch: {path}")
        observed[name] = actual
    return {"path": str(root), "files": observed, "revision": contract["tokenizer_revision"]}


def _default_output_check(path: Path) -> Path:
    from campaign_rl_entry import assert_fresh_native_ext4_path
    return assert_fresh_native_ext4_path(path, "theta0 output")


def _default_parent_check(path: Path, digest: str) -> dict[str, Any]:
    from campaign_rl_entry import verify_merged_parent_manifest
    return verify_merged_parent_manifest({"path": str(path), "sha256": digest})


def _default_occupancy_check() -> dict[str, Any]:
    from campaign_rl_entry import check_single_cuda_occupancy
    return check_single_cuda_occupancy()


def _parent_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("status", "manifest_sha256", "merged_weights_sha256", "base_model_revision",
              "model_path", "sft_identity", "weight_inventory_sha256")
    return {name: audit[name] for name in fields if name in audit}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device(model: Any) -> Any:
    value = getattr(model, "device", None)
    if value is not None:
        return value
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, AttributeError, TypeError):
            pass
    return "cuda:0"


def _is_cuda_device(device: Any) -> bool:
    text = str(device).lower()
    return text.startswith("cuda")


def _sync(torch_module: Any, device: Any) -> None:
    if _is_cuda_device(device):
        cuda = getattr(torch_module, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(device)


def _resources(torch_module: Any, device: Any) -> dict[str, Any]:
    if not _is_cuda_device(device):
        return {"device": str(device), "cuda": False}
    cuda = getattr(torch_module, "cuda", None)
    result: dict[str, Any] = {"device": str(device), "cuda": True}
    for method, field in (("memory_allocated", "allocated_bytes"),
                          ("max_memory_allocated", "peak_allocated_bytes"),
                          ("memory_reserved", "reserved_bytes"),
                          ("max_memory_reserved", "peak_reserved_bytes")):
        fn = getattr(cuda, method, None)
        if callable(fn):
            try:
                result[field] = int(fn(device))
            except TypeError:
                result[field] = int(fn())
    return result


def _as_ids(value: Any) -> list[int]:
    if hasattr(value, "sequences"):
        value = value.sequences
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or any(type(item) is not int for item in value):
        raise Theta0GateError("model.generate did not return one integer token sequence")
    return list(value)


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise Theta0GateError("loaded tokenizer does not expose decode()")
    return str(decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False))


def _source_label(case: Mapping[str, Any], parsed: Any) -> dict[str, Any] | None:
    """Score only the source-authoritative base/strict-no-op cases."""
    if case["kind"] not in ("evidence_supported", "strict_noop_control"):
        return None
    row = _mapping(case["row"], f"base row {case['base_row_id']}")
    operation = row.get("target_operation", row.get("operation"))
    if operation not in ("no_op", "replace", "delete"):
        return None
    if operation == "no_op":
        exact = parsed.status == "accepted" and parsed.operation == "no_op"
    else:
        # This is an out-of-band source-authoritative comparison only.  The
        # target is never included in the model request or in the context
        # reconstruction.  RL-02 stores the text as target_body_text.
        region_new = row.get("region_new")
        if region_new is None:
            body_text = row.get("target_body_text")
            if isinstance(body_text, str):
                region_new = [] if body_text == "" else body_text.split("\n")
            else:
                region_new = None
        exact = (parsed.status == "accepted" and parsed.operation == operation
                 and isinstance(region_new, list) and list(parsed.body) == region_new)
    return {
        "operation": operation,
        "source_noop": operation == "no_op",
        "source_exact_region": bool(exact),
        "label_scope": "source-authoritative-base-only",
    }


def _case_record(case: Mapping[str, Any], arm: str, use_cache: bool, model: Any,
                 tokenizer: Any, reference_tokenizer: Any, fast_model: Any,
                 *, torch_module: Any = None, case_guard: Callable[[Any], Any] | None = None,
                 identity_checker: Callable[..., Mapping[str, Any]] | None = None,
                 resource_probe: Callable[[], Mapping[str, Any]] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Run one request and return a label-free durable record."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as error:  # pragma: no cover - live environment
            raise Theta0Unsupported(f"torch unavailable for live theta0 decode: {error}") from error
    device = _device(model)
    prompt_ids = list(case["prompt_ids"])
    try:
        input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
        attention_mask = torch_module.ones_like(input_ids)
    except Exception as error:
        # A fake-model test can deliberately use ordinary lists, but the live
        # path must always construct a device tensor.
        if getattr(model, "accepts_python_ids", False):
            input_ids, attention_mask = [prompt_ids], [[1] * len(prompt_ids)]
        else:
            raise Theta0GateError(f"cannot construct prompt tensor: {error}") from error

    if fast_model is not None:
        for_inference = getattr(fast_model, "for_inference", None)
        if callable(for_inference):
            for_inference(model)
    guard = case_guard(model) if case_guard is not None else nullcontext()
    before_resources = resource_probe() if resource_probe is not None else _resources(torch_module, device)
    _sync(torch_module, device)
    started = clock()
    try:
        with guard:
            returned = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=COMPLETION_CAP,
                do_sample=False,
                num_beams=1,
                eos_token_id=list(NATIVE_EOG_IDS),
                pad_token_id=EOS_ID,
                bos_token_id=BOS_ID,
                repetition_penalty=1.0,
                return_dict_in_generate=False,
                use_cache=use_cache,
            )
    except (NotImplementedError, TypeError) as error:
        if not use_cache:
            raise Theta0Unsupported(
                f"decode-KV-cache control arm unsupported by model.generate: {error}"
            ) from error
        raise
    except Exception:
        raise
    finally:
        _sync(torch_module, device)
    elapsed = clock() - started
    returned_ids = _as_ids(returned)
    if returned_ids[:len(prompt_ids)] != prompt_ids:
        raise Theta0GateError(f"model returned a sequence with a changed prompt for {case['variant_id']}")
    tail = returned_ids[len(prompt_ids):]
    terminal_index = next((index for index, token in enumerate(tail) if token in NATIVE_EOG_IDS), None)
    if terminal_index is None:
        generated_ids = list(tail)
        terminal_reason = "length" if len(generated_ids) == COMPLETION_CAP else "unknown"
        trailing_after_terminal = 0
        terminal_token = None
    else:
        generated_ids = list(tail[:terminal_index + 1])
        terminal_reason = "eos"
        trailing_after_terminal = len(tail) - len(generated_ids)
        terminal_token = generated_ids[-1]
    if len(tail) > COMPLETION_CAP:
        cap_status = "overflow"
    elif terminal_reason == "length":
        cap_status = "hit_without_terminal"
    else:
        cap_status = "within_cap"
    body_ids = generated_ids[:-1] if terminal_reason == "eos" else generated_ids
    raw = _decode(tokenizer, body_ids)
    try:
        parsed = parse_output(raw, case["context"])
        parser_status = parsed.status
        parser_operation = parsed.operation
        parser_reason = parsed.reason
        canonical_valid = valid_generation_tokens(generated_ids)
    except Exception as error:
        parsed = None
        parser_status = "error"
        parser_operation = None
        parser_reason = f"{type(error).__name__}: {error}"
        canonical_valid = False
    if identity_checker is not None:
        identity_audit = dict(identity_checker(model, tokenizer, reference_tokenizer, []))
    else:
        identity_audit = {"status": "not_checked"}
    after_resources = resource_probe() if resource_probe is not None else _resources(torch_module, device)
    return {
        "id": case["variant_id"],
        "base_row_id": case["base_row_id"],
        "variant_kind": case["kind"],
        "arm": arm,
        "fresh_request": True,
        "use_cache": use_cache,
        "cache_scope": "per-request-decode-only" if use_cache else "decode-KV-cache-disabled",
        "context_sha256": case["context_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "prompt_ids_sha256": case["prompt_ids_sha256"],
        "prompt_tokens_with_bos": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "returned_sequence_ids": returned_ids,
        "generated_token_ids_through_first_terminal": generated_ids,
        "raw_output": raw,
        "protocol": {
            "parser_status": parser_status,
            "parser_operation": parser_operation,
            "parser_reason": parser_reason,
            "canonical_generation_tokens_valid": canonical_valid,
        },
        "eos": {
            "native_eog_ids": list(NATIVE_EOG_IDS),
            "terminal_reason": terminal_reason,
            "terminal_token": terminal_token,
            "canonical_eos": terminal_reason == "eos" and terminal_token == EOS_ID,
            "noncanonical_native_eog": terminal_reason == "eos" and terminal_token == 130073,
            "first_terminal_index_in_tail": terminal_index,
            "trailing_after_first_terminal": trailing_after_terminal,
        },
        "cap": {
            "limit": COMPLETION_CAP,
            "returned_tail_tokens": len(tail),
            "generated_tokens_through_first_terminal": len(generated_ids),
            "status": cap_status,
            "inclusive_terminal_policy": True,
        },
        "timing": {
            "synchronized": _is_cuda_device(device),
            "synchronized_timing": _is_cuda_device(device),
            "generation_seconds": elapsed,
            "scope": "prompt-plus-greedy-generate",
        },
        "resources": {
            "before": dict(before_resources),
            "after": dict(after_resources),
            "peak": dict(after_resources),
        },
        "peak_allocation": dict(after_resources),
        "parent": dict(case.get("parent", {})),
        "parent_identity_checked": True,
        "post_generation_identity": identity_audit,
        "source_authoritative": _source_label(case, parsed) if parsed is not None else None,
        "counterfactual_target_score": None,
    }


def _model_loader(parent_audit: Mapping[str, Any], prompt_rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any, Any, Any]:
    from campaign_rl_entry import _load_live_model
    return _load_live_model(parent_audit, prompt_rows)


def _default_case_guard(model: Any):
    from campaign_eval import case_evaluation_guard
    return case_evaluation_guard(model)


def _default_identity_checker(model: Any, tokenizer: Any, reference: Any, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    from campaign_sft import assert_post_trainer_pinned_identity
    return assert_post_trainer_pinned_identity(model, tokenizer, reference, rows)


def _default_resource_probe(torch_module: Any, device: Any) -> Mapping[str, Any]:
    return _resources(torch_module, device)


def _transition_to_training(model: Any, fast_model: Any, identity_checker: Callable[..., Mapping[str, Any]],
                            tokenizer: Any, reference: Any) -> dict[str, Any]:
    hook = getattr(fast_model, "for_training", None) if fast_model is not None else None
    if not callable(hook):
        return {"status": "unsupported", "reason": "FastLanguageModel.for_training hook unavailable"}
    try:
        try:
            hook(model, use_gradient_checkpointing=False)
        except TypeError:
            hook(model)
        audit = dict(identity_checker(model, tokenizer, reference, []))
        return {"status": "checked", "hook": "for_training", "model_training": getattr(model, "training", None),
                "post_generation_identity": audit}
    except Exception as error:
        return {"status": "failed", "hook": "for_training", "error": f"{type(error).__name__}: {error}"}


def _case_failure_record(case: Mapping[str, Any], arm: str, use_cache: bool,
                         error: Exception) -> dict[str, Any]:
    """Keep the failing variant visible in the durable partial receipt."""
    return {
        "id": case["variant_id"],
        "base_row_id": case["base_row_id"],
        "variant_kind": case["kind"],
        "arm": arm,
        "fresh_request": True,
        "use_cache": use_cache,
        "cache_scope": "per-request-decode-only" if use_cache else "decode-KV-cache-disabled",
        "context_sha256": case["context_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "prompt_ids_sha256": case["prompt_ids_sha256"],
        "prompt_tokens_with_bos": case["prompt_tokens_with_bos"],
        "status": "case_failed",
        "error": {"type": type(error).__name__, "message": str(error)},
        "parent": dict(case.get("parent", {})),
        "counterfactual_target_score": None,
    }


def run_gate(
    *,
    fixture_path: Path,
    parent_manifest: Path,
    parent_manifest_sha256: str,
    output: Path,
    deadline_seconds: int,
    admission_receipt: Path | None = None,
    require_admission: bool = True,
    output_checker: Callable[[Path], Path] | None = None,
    parent_checker: Callable[[Path, str], Mapping[str, Any]] | None = None,
    occupancy_checker: Callable[[], Mapping[str, Any]] | None = None,
    model_loader: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], tuple[Any, Any, Any, Any, Any]] | None = None,
    tokenizer_override: Any = None,
    torch_override: Any = None,
    case_guard: Callable[[Any], Any] | None = None,
    identity_checker: Callable[..., Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if type(deadline_seconds) is not int or not 0 < deadline_seconds <= MAX_DEADLINE_SECONDS:
        raise Theta0GateError(f"deadline_seconds must be an integer in 1..{MAX_DEADLINE_SECONDS}")
    if deadline_seconds <= RESERVE_SECONDS:
        raise Theta0GateError("deadline_seconds leaves no 60-second soft-deadline reserve")
    output_path = (output_checker or _default_output_check)(Path(output))
    started = clock()
    deadline_at = started + deadline_seconds
    preflight_receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight_running",
        "started_at": datetime.fromtimestamp(wall_clock(), timezone.utc).isoformat(),
        "deadline_seconds": deadline_seconds,
        "soft_deadline_reserve_seconds": RESERVE_SECONDS,
        "output": str(Path(output_path).resolve()),
        "fixture_path": str(Path(fixture_path).resolve()),
        "parent_manifest": str(Path(parent_manifest).resolve()),
        "model_loaded": False,
        "cuda_model_loaded": False,
        "cases": [],
    }
    _write_json(output_path, preflight_receipt)
    try:
        prepared = validate_fixture_inputs(Path(fixture_path))
        protocol_audit = _verify_protocol_source(prepared["fixture"])
        tokenizer_file_audit = _verify_tokenizer_files(prepared["fixture"])
        admission = _verify_admission(admission_receipt, prepared) if require_admission else {
            "status": "test_override_or_lead_admission_external", "hashes_bound": 0
        }
        parent_audit = dict((parent_checker or _default_parent_check)(Path(parent_manifest), parent_manifest_sha256))
        occupancy = dict((occupancy_checker or _default_occupancy_check)())
    except Theta0Unsupported as error:
        preflight_receipt["status"] = "unsupported"
        preflight_receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        preflight_receipt["completed_at"] = _now_iso()
        _write_json(output_path, preflight_receipt)
        return preflight_receipt
    except Exception as error:
        preflight_receipt["status"] = "failed"
        preflight_receipt["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback_tail": traceback.format_exc(limit=4).splitlines()[-4:],
        }
        preflight_receipt["completed_at"] = _now_iso()
        _write_json(output_path, preflight_receipt)
        return preflight_receipt
    base_receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.fromtimestamp(wall_clock(), timezone.utc).isoformat(),
        "deadline_seconds": deadline_seconds,
        "soft_deadline_reserve_seconds": RESERVE_SECONDS,
        "soft_deadline_at_monotonic": deadline_at,
        "output": str(Path(output_path).resolve()),
        "fixture": {
            "path": str(Path(fixture_path).resolve()),
            "schema_version": prepared["fixture"]["schema_version"],
            "base_rows": len(prepared["base_ids"]),
            "variants": len(prepared["variant_defs"]),
            "rows_sha256": prepared["fixture"]["inputs"]["rows_sha256"],
            "sidecar_sha256": prepared["fixture"]["inputs"]["sidecar_sha256"],
            "selected_ids_sha256": prepared["fixture"]["inputs"]["selected_ids_sha256"],
            "ordered_ids_sha256": prepared["fixture"]["inputs"]["ordered_ids_sha256"],
            "row_identity_sha256": prepared["fixture"]["inputs"]["row_identity_sha256"],
        },
        "contract": {
            "renderer_id": prepared["fixture"]["contract"]["renderer_id"],
            "tokenization_policy": prepared["fixture"]["contract"]["tokenization_policy"],
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "native_eog_ids": list(NATIVE_EOG_IDS),
            "prompt_cap": PROMPT_CAP,
            "completion_cap": COMPLETION_CAP,
            "managed_context_cap": CONTEXT_CAP,
            "protocol": protocol_audit,
            "tokenizer": tokenizer_file_audit,
        },
        "admission": admission,
        "parent": _parent_summary(parent_audit),
        "occupancy": occupancy,
        "arms": {
            "standard": {"name": STANDARD_ARM, "use_cache": True, "cases": 16,
                          "cache_scope": "fresh full prompt per request; decode KV cache only"},
            "control": {"name": CONTROL_ARM, "use_cache": False, "cases": 4,
                         "base_scope": "four history-supported evidence_supported variants only",
                         "cache_scope": "decode KV-cache numerical control; not cross-request prefix cache"},
        },
        "base_source_authoritative": [],
        "cases": [],
        "transition_to_training": {"status": "not_run"},
        "model_loaded": False,
        "cuda_model_loaded": False,
    }
    base_meta = {
        row.get("id"): row for row in prepared["fixture"].get("base_rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    for row_id in prepared["base_ids"]:
        row = prepared["rows"][row_id]
        fixture_class = base_meta.get(row_id, {}).get("fixture_class", "unknown")
        source_operation = row.get("target_operation", row.get("operation"))
        base_receipt["base_source_authoritative"].append({
            "base_row_id": row_id,
            "fixture_class": fixture_class,
            "source_operation": source_operation,
            "source_noop": source_operation == "no_op",
            "exact_score_scope": (
                "source-authoritative-base-only"
                if fixture_class in ("history-supported-structured-edit", "history_supported_structured_edit",
                                     "strict_noop_control")
                else "diagnostic-only"
            ),
        })
    _write_json(output_path, base_receipt)
    try:
        if clock() + RESERVE_SECONDS >= deadline_at:
            raise Theta0Deadline("no diagnostic budget remains after the 60-second reserve")

        loader = model_loader or _model_loader
        prompt_rows = []
        for row_id in prepared["base_ids"]:
            row = prepared["rows"][row_id]
            prompt_rows.append({key: row[key] for key in ("id", "prompt_text", "input_ids", "target_start")})
        loaded = loader(parent_audit, prompt_rows)
        if len(loaded) != 5:
            raise Theta0GateError("live loader must return model, tokenizer, reference, FastLanguageModel, audit")
        model, tokenizer, reference_tokenizer, fast_model, tokenizer_audit = loaded
        base_receipt["model_loaded"] = True
        base_receipt["cuda_model_loaded"] = _is_cuda_device(_device(model))
        base_receipt["loaded_tokenizer_audit"] = tokenizer_audit
        tokenizer = tokenizer_override if tokenizer_override is not None else tokenizer
        cases = build_variant_cases(prepared, tokenizer)
        parent_for_case = _parent_summary(parent_audit)
        for case in cases:
            case["parent"] = parent_for_case
        supported_base_ids = {
            row["id"] for row in prepared["fixture"].get("base_rows", [])
            if isinstance(row, Mapping)
            and row.get("fixture_class") == "history_supported_structured_edit"
        }
        controls = {
            case["variant_id"] for case in cases
            if case["kind"] == "evidence_supported"
            and case["base_row_id"] in supported_base_ids
        }
        if prepared["fixture"].get("base_rows") and len(controls) != 4:
            raise Theta0GateError(
                f"expected four history-supported control cases, found {len(controls)}"
            )
        guard = case_guard or _default_case_guard
        checker = identity_checker or _default_identity_checker
        for case in cases:
            if clock() + RESERVE_SECONDS >= deadline_at:
                raise Theta0Deadline("soft deadline reached before all theta0 cases completed")
            try:
                record = _case_record(
                    case, STANDARD_ARM, True, model, tokenizer, reference_tokenizer, fast_model,
                    torch_module=torch_override, case_guard=guard, identity_checker=checker,
                    clock=time.perf_counter,
                )
            except Exception as error:
                base_receipt["cases"].append(_case_failure_record(case, STANDARD_ARM, True, error))
                _write_json(output_path, base_receipt)
                raise
            base_receipt["cases"].append(record)
            _write_json(output_path, base_receipt)
        for case in cases:
            if case["variant_id"] not in controls:
                continue
            if clock() + RESERVE_SECONDS >= deadline_at:
                raise Theta0Deadline("soft deadline reached before KV-cache control arm completed")
            try:
                record = _case_record(
                    case, CONTROL_ARM, False, model, tokenizer, reference_tokenizer, fast_model,
                    torch_module=torch_override, case_guard=guard, identity_checker=checker,
                    clock=time.perf_counter,
                )
            except Exception as error:
                base_receipt["cases"].append(_case_failure_record(case, CONTROL_ARM, False, error))
                _write_json(output_path, base_receipt)
                raise
            base_receipt["cases"].append(record)
            _write_json(output_path, base_receipt)
        base_receipt["transition_to_training"] = _transition_to_training(
            model, fast_model, checker, tokenizer, reference_tokenizer,
        )
        base_receipt["status"] = "complete"
        base_receipt["completed_at"] = _now_iso()
        _write_json(output_path, base_receipt)
        return base_receipt
    except Theta0Unsupported as error:
        base_receipt["status"] = "unsupported"
        base_receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    except Theta0Deadline as error:
        base_receipt["status"] = "deadline"
        base_receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        base_receipt["status"] = "failed"
        base_receipt["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback_tail": traceback.format_exc(limit=4).splitlines()[-4:],
        }
    base_receipt["completed_at"] = _now_iso()
    _write_json(output_path, base_receipt)
    return base_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, required=True)
    parser.add_argument("--admission-receipt", type=Path,
                        default=Path(os.environ["SEPALITH_RL_ADMISSION_RECEIPT"])
                        if os.environ.get("SEPALITH_RL_ADMISSION_RECEIPT") else None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_gate(
            fixture_path=args.fixture,
            parent_manifest=args.parent_manifest,
            parent_manifest_sha256=args.parent_manifest_sha256,
            output=args.output,
            deadline_seconds=args.deadline_seconds,
            admission_receipt=args.admission_receipt,
        )
    except Exception as error:
        result = {"schema_version": SCHEMA_VERSION, "status": "failed",
                  "failure": {"type": type(error).__name__, "message": str(error)}}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
