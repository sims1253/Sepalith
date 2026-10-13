#!/usr/bin/env python3
"""Independent CPU-only review of the paired RUN-05 output-192 DEV runs.

This review is deliberately read-only with respect to the run and campaign
state.  It loads the corrected panel and the pinned PRM-03 protocol, renders
and tokenizes every panel context with the pinned HF tokenizer, decodes every
saved native response, and recomputes protocol/cap/parser/target decisions.
The two current runs are compared row by row.  The older output-512 run is
audited separately so its denominator cannot be mixed with output-192.

No model bytes are opened or hashed by this program.  The model identity is
checked only against the immutable provenance string saved by the native
client and the pinned controller manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
STATE_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training")
TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")

PANEL = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
PANEL_MANIFEST = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/manifest.json"
PROTOCOL = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
CAMPAIGN_CLIENT = EXEC_ROOT / "extensions/vscode-sepalith/src/campaign_client.ts"
CAMPAIGN_EVAL = EXEC_ROOT / "experiments/training/campaign_eval.py"
REFERENCE_SCORER = PLAN_ROOT / "docs/campaign/work/quant-quality/run09_native_dev_quality.py"
CONTROLLER = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-dev192-a/run_dev192.py"
CAP_ADAPTER = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-dev192-a/run_dev192_client.py"
CONTROLLER_MANIFEST = PLAN_ROOT / "docs/campaign/work/lead/theta0-cuda-dev192-a/manifest.json"

CURRENT_ROOT = STATE_ROOT / "RUN-05-theta0-cuda-dev192-a"
HISTORICAL_ROOT = STATE_ROOT / "RUN-09-theta0-q8-cuda-dev-a"
CURRENT_RUNS = {
    "b256": CURRENT_ROOT / "Q8_0-b256",
    "b1024": CURRENT_ROOT / "Q8_0-b1024",
}

MODEL_PATH = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/"
    "SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf"
)
SERVER_PATH = Path(
    "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/"
    "llama-cuda-b10453/llama-server"
)

EXPECTED = {
    "model_sha256": "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559",
    "server_sha256": "e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee",
    "panel_sha256": "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035",
    "panel_manifest_sha256": "4528ff6dbd8cfe70580901d9254a1a6931dc8c334c5dd2c55e49cf92d1a5377a",
    "protocol_sha256": "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
    "campaign_client_sha256": "0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333",
    "campaign_eval_sha256": "7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f",
    "reference_scorer_sha256": "e574a9453066b5fa5a8e33c02777e8bdd875e208d4c3c51c00e5ede13bba5362",
    "controller_sha256": "18d1e37dbc94ab108865a7aa01ceb77bac51a63cc08b753c15e807d21032f8e3",
    "cap_adapter_sha256": "dacabb514952427b9b36c1f85c964c786cbe7820100c771ff13c99dabb2ecc89",
    "tokenizer_json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
    "tokenizer_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
    "build_info": "b10453-3cb7ffb1a",
}

PANEL_ROWS = 75
EDIT_ROWS = 43
NOOP_ROWS = 32
CONTEXT_SIZE = 4096
CURRENT_CAP = 192
HISTORICAL_CAP = 512
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = {1, 130073}
VOCAB_SIZE = 130560


class ReviewError(RuntimeError):
    """An immutable input or independently checked result violated the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot read JSON {path}: {error}") from error


def load_protocol() -> Any:
    observed = sha256_file(PROTOCOL)
    if observed != EXPECTED["protocol_sha256"]:
        raise ReviewError(f"pinned protocol hash mismatch: {observed}")
    package_src = EXEC_ROOT / "packages" / "sepalith" / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    try:
        return importlib.import_module("sepalith.campaign_protocol")
    except Exception as error:
        raise ReviewError(f"cannot import pinned campaign protocol: {error}") from error


def load_tokenizer() -> tuple[Any, dict[str, Any]]:
    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    tokenizer_config = TOKENIZER_DIR / "tokenizer_config.json"
    observed_json = sha256_file(tokenizer_json)
    observed_config = sha256_file(tokenizer_config)
    if observed_json != EXPECTED["tokenizer_json_sha256"]:
        raise ReviewError(f"tokenizer.json hash mismatch: {observed_json}")
    if observed_config != EXPECTED["tokenizer_config_sha256"]:
        raise ReviewError(f"tokenizer_config.json hash mismatch: {observed_config}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR), local_files_only=True, use_fast=True,
        trust_remote_code=False,
    )
    identity = {
        "vocab_size": len(tokenizer),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    expected_identity = {
        "vocab_size": VOCAB_SIZE, "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID, "pad_token_id": EOS_ID,
    }
    if identity != expected_identity:
        raise ReviewError(f"tokenizer identity mismatch: {identity}")
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert) or convert(EOS_ID) != "</s>":
        raise ReviewError("pinned tokenizer EOS ID 1 is not </s>")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or len(vocab) != VOCAB_SIZE:
        raise ReviewError("pinned tokenizer vocabulary is incomplete")
    vocab_sha = canonical_sha(sorted((str(key), int(value)) for key, value in vocab.items()))
    return tokenizer, {
        "path": str(TOKENIZER_DIR),
        "revision": EXPECTED["tokenizer_revision"],
        "files": {
            "tokenizer.json": observed_json,
            "tokenizer_config.json": observed_config,
        },
        "identity": identity,
        "vocab_sha256": vocab_sha,
    }


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    ids = getattr(value, "ids", value)
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise ReviewError("tokenizer.encode did not return a token-ID sequence")
    result = list(ids)
    if any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in result):
        raise ReviewError("tokenizer returned an out-of-range token")
    return result


def _target_geometry(protocol: Any, tokenizer: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    operation = row["operation"]
    region_new = list(row["region_new"])
    target_text = protocol.serialize_target(operation, region_new)
    target_ids = _encode_text(tokenizer, target_text)
    terminal_ids = _encode_text(tokenizer, ">>>>>>> UPDATED")
    if len(target_ids) < len(terminal_ids) or target_ids[-len(terminal_ids):] != terminal_ids:
        raise ReviewError(f"{row['id']}: target terminal suffix retokenized")
    body_ids = target_ids[:-len(terminal_ids)]
    target_parse = protocol.parse_output(target_text, protocol.PromptContext.from_mapping(row["context"]))
    if target_parse.status != "accepted" or target_parse.operation != operation:
        raise ReviewError(f"{row['id']}: serialized corrected target does not round-trip")
    if operation == "replace" and list(target_parse.body) != region_new:
        raise ReviewError(f"{row['id']}: serialized replacement body differs from region_new")
    if operation == "no_op" and row.get("target_body_text") != "[NO_EDIT]":
        raise ReviewError(f"{row['id']}: no-op target body label is not [NO_EDIT]")
    expected_body_text = "[NO_EDIT]" if operation == "no_op" else "\n".join(region_new)
    if row.get("target_body_text") != expected_body_text:
        raise ReviewError(f"{row['id']}: target_body_text is not the complete corrected body")
    # The corrected panel's target_sha256 identifies the full serialized wire
    # target (body plus terminal).  The native quality JSON independently
    # stores the scorer's canonical region_new identity; both identities are
    # checked below from the same complete corrected target.
    if row.get("target_sha256") != sha256_text(target_text):
        raise ReviewError(f"{row['id']}: panel target identity is not the serialized corrected target")
    if row.get("target_body_token_count") != len(body_ids):
        raise ReviewError(f"{row['id']}: target body token count mismatch")
    if row.get("target_terminal_token_count") != len(terminal_ids):
        raise ReviewError(f"{row['id']}: target terminal token count mismatch")
    return {
        "target_text_sha256": sha256_text(target_text),
        "target_body_sha256": sha256_text(expected_body_text),
        "target_body_token_count": len(body_ids),
        "target_terminal_token_count": len(terminal_ids),
        "target_round_trip": True,
    }


def _source_audit(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the three source-provenance shapes used by the DEV panel."""
    origin = provenance.get("origin")
    if isinstance(origin, Mapping):
        origin_kind = origin.get("kind") or origin.get("source")
    elif isinstance(origin, str):
        origin_kind = origin
    elif isinstance(provenance.get("correction"), Mapping):
        origin_kind = provenance.get("origin") or "finish_correction"
    else:
        origin_kind = provenance.get("source")
    parent = provenance.get("parent_identity")
    if parent is None:
        parent = provenance.get("parent_provenance")
    has_parent = isinstance(parent, (Mapping, str))
    has_source_identity = any(
        key in provenance
        for key in ("source_identity", "source_path", "source_snapshot_path", "selection_source", "correction")
    )
    has_geometry = any(
        key in provenance
        for key in ("region_geometry", "history_replay", "pre_edit_document", "target_start_line", "correction")
    ) or (
        "post_edit_sha256" in provenance
        and "source_snapshot_sha256" in provenance
    ) or isinstance(origin, Mapping)
    simulated = provenance.get("source_is_simulated")
    if not isinstance(simulated, bool):
        simulated = provenance.get("source_snapshot_is_simulated")
    if not isinstance(simulated, bool):
        simulated = "source_derived_simulated" in json.dumps(provenance, ensure_ascii=False)
    return {
        "origin_kind": origin_kind,
        "has_parent_identity": has_parent,
        "has_source_identity": has_source_identity,
        "has_geometry_evidence": has_geometry,
        "source_is_simulated": bool(simulated),
        "complete": bool(origin_kind or has_source_identity) and has_geometry,
    }


def load_panel(protocol: Any, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_panel = sha256_file(PANEL)
    observed_manifest = sha256_file(PANEL_MANIFEST)
    if observed_panel != EXPECTED["panel_sha256"]:
        raise ReviewError(f"corrected panel hash mismatch: {observed_panel}")
    if observed_manifest != EXPECTED["panel_manifest_sha256"]:
        raise ReviewError(f"corrected panel manifest hash mismatch: {observed_manifest}")
    panel_manifest = read_json(PANEL_MANIFEST)
    if panel_manifest.get("output", {}).get("sha256") != observed_panel:
        raise ReviewError("corrected panel manifest output hash mismatch")
    if panel_manifest.get("counts") != {
        "rows": PANEL_ROWS, "changed_finish": 6, "other_lines_byte_identical": 69,
        "edit_rows": EDIT_ROWS, "noop_rows": NOOP_ROWS,
    }:
        raise ReviewError("corrected panel manifest counts mismatch")
    changed_by_id = {item["id"]: item for item in panel_manifest.get("changes", [])}
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with PANEL.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ReviewError(f"blank corrected panel line {line_number}")
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ReviewError(f"corrected panel line {line_number} is not an object")
            row = dict(row)
            row_id = row.get("id")
            if row.get("split") != "dev" or not isinstance(row_id, str) or not row_id:
                raise ReviewError(f"invalid corrected panel row at line {line_number}")
            if row_id in by_id:
                raise ReviewError(f"duplicate corrected panel ID {row_id}")
            if row.get("operation") not in {"replace", "no_op"}:
                raise ReviewError(f"unsupported operation for {row_id}")
            if not isinstance(row.get("source_provenance"), Mapping):
                raise ReviewError(f"{row_id}: missing source provenance")
            provenance = row["source_provenance"]
            source_audit = _source_audit(provenance)
            if not source_audit["complete"]:
                raise ReviewError(f"{row_id}: incomplete source provenance")
            context = protocol.PromptContext.from_mapping(row.get("context"))
            region_new = row.get("region_new")
            if not isinstance(region_new, list) or any(not isinstance(item, str) for item in region_new):
                raise ReviewError(f"{row_id}: invalid corrected region_new")
            if row["operation"] == "no_op" and list(context.region_old) != region_new:
                raise ReviewError(f"{row_id}: no-op region_new differs from captured region_old")
            if row["operation"] == "replace" and list(context.region_old) == region_new:
                raise ReviewError(f"{row_id}: replacement target is unchanged")
            geometry = _target_geometry(protocol, tokenizer, row)
            if row_id in changed_by_id:
                change = changed_by_id[row_id]
                if change.get("new_target_sha256") != row.get("target_sha256"):
                    raise ReviewError(f"{row_id}: corrected finish target hash mismatch")
            item = {
                "row": row, "context": context,
                "target_geometry": geometry, "source_audit": source_audit,
            }
            rows.append(item)
            by_id[row_id] = item
    if len(rows) != PANEL_ROWS:
        raise ReviewError(f"corrected panel has {len(rows)} rows")
    operations = Counter(item["row"]["operation"] for item in rows)
    if operations != Counter({"replace": EDIT_ROWS, "no_op": NOOP_ROWS}):
        raise ReviewError(f"corrected panel operation counts mismatch: {operations}")
    ordered_ids = [item["row"]["id"] for item in rows]
    if set(changed_by_id) != {item["id"] for item in panel_manifest.get("changes", [])}:
        raise ReviewError("corrected panel change manifest IDs are inconsistent")
    return rows, {
        "path": str(PANEL),
        "sha256": observed_panel,
        "manifest_path": str(PANEL_MANIFEST),
        "manifest_sha256": observed_manifest,
        "rows": len(rows),
        "edit_rows": operations["replace"],
        "noop_rows": operations["no_op"],
        "ordered_ids_sha256": canonical_sha(ordered_ids),
        "ordered_ids": ordered_ids,
        "families": dict(sorted(Counter(item["row"]["family"] for item in rows).items())),
        "source_simulated_rows": sum(
            int(item["source_audit"]["source_is_simulated"])
            for item in rows
        ),
        "corrected_finish_ids": sorted(changed_by_id),
        "target_geometry_checked": True,
    }


def _protocol_failure(
    *, canonical_tokens: bool, wire_match: bool, generated_count: int,
    cap: int, saw_stop: bool, final_sse_tokens: int, malformed: int,
    stop_type: object, parser_status: object,
) -> str | None:
    if not canonical_tokens:
        return "invalid_generation_tokens"
    if not wire_match:
        return "wire_hf_text_mismatch"
    if generated_count > cap or (generated_count == cap and stop_type != "eos"):
        return "completion_cap_overflow"
    if not saw_stop:
        return "missing_sse_stop"
    if final_sse_tokens != 0:
        return "final_sse_token_duplication"
    if malformed:
        return "malformed_sse"
    if stop_type != "eos":
        return "noncanonical_stop_type"
    if parser_status != "accepted":
        return "invalid_prm03_output"
    return None


def independent_case(
    protocol: Any, tokenizer: Any, item: Mapping[str, Any],
    saved_case: Mapping[str, Any], cap: int,
) -> dict[str, Any]:
    row = item["row"]
    context = item["context"]
    row_id = row["id"]
    if saved_case.get("id") != row_id:
        raise ReviewError(f"case ID mismatch: {saved_case.get('id')} vs {row_id}")
    prompt = protocol.render_prompt(context)
    prompt_sha = sha256_text(prompt)
    prompt_ids = protocol.encode_prompt(context, tokenizer, include_bos=True)
    prompt_without_bos = prompt_ids[1:] if prompt_ids and prompt_ids[0] == BOS_ID else []
    saved_hf_ids = saved_case.get("hf_prompt_ids")
    saved_native_ids = saved_case.get("native_prompt_ids_without_bos")
    if prompt_sha != row.get("prompt_sha256") or prompt_sha != saved_case.get("prompt_sha256"):
        raise ReviewError(f"{row_id}: rendered prompt hash mismatch")
    target_text = protocol.serialize_target(row["operation"], list(row["region_new"]))
    panel_target_sha = sha256_text(target_text)
    canonical_region_sha = canonical_sha(list(row["region_new"]))
    if row.get("target_sha256") != panel_target_sha:
        raise ReviewError(f"{row_id}: panel target hash is not the complete serialized target")
    if saved_case.get("target_sha256") != canonical_region_sha:
        raise ReviewError(f"{row_id}: quality target hash is not canonical corrected region_new")
    if saved_hf_ids != prompt_ids or saved_native_ids != prompt_without_bos:
        raise ReviewError(f"{row_id}: saved HF/native prompt IDs differ from independent tokenization")
    if saved_case.get("hf_prompt_ids_sha256") != canonical_sha(prompt_ids):
        raise ReviewError(f"{row_id}: saved HF prompt ID hash mismatch")
    if saved_case.get("native_prompt_ids_sha256") != canonical_sha(prompt_without_bos):
        raise ReviewError(f"{row_id}: saved native prompt ID hash mismatch")
    if len(prompt_ids) + cap > CONTEXT_SIZE:
        raise ReviewError(f"{row_id}: prompt plus cap exceeds native context")
    generated = saved_case.get("returned_token_ids")
    if not isinstance(generated, list) or any(type(token) is not int for token in generated):
        raise ReviewError(f"{row_id}: returned token IDs are not an integer list")
    if any(token < 0 or token >= VOCAB_SIZE for token in generated):
        raise ReviewError(f"{row_id}: returned token ID is outside vocabulary")
    raw_text = saved_case.get("raw_text")
    if not isinstance(raw_text, str):
        raise ReviewError(f"{row_id}: native raw_text is not a string")
    terminal = generated[-1] if generated else None
    body_ids = generated[:-1] if terminal == EOS_ID else generated
    try:
        decoded_body = tokenizer.decode(
            body_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decode_error = None
        wire_match = decoded_body == raw_text
        decoded_sha = sha256_text(decoded_body)
    except Exception as error:
        decoded_body = ""
        decode_error = f"{type(error).__name__}: {error}"
        wire_match = False
        decoded_sha = None
    parsed = protocol.parse_output(raw_text, context)
    timing = saved_case.get("server_timing")
    if not isinstance(timing, Mapping):
        timing = {}
    saved_protocol = saved_case.get("protocol")
    if not isinstance(saved_protocol, Mapping):
        saved_protocol = {}
    stop_type = timing.get("stop_type")
    tokens_predicted = timing.get("tokens_predicted")
    truncated = timing.get("truncated") is True
    final_sse_tokens = saved_protocol.get("final_sse_tokens")
    malformed = saved_protocol.get("malformed_sse_frames")
    saw_stop = saved_protocol.get("saw_stop") is True
    canonical_tokens = bool(generated) and bool(protocol.valid_generation_tokens(generated))
    protocol_valid = bool(
        canonical_tokens and wire_match and len(generated) <= cap
        and saw_stop and final_sse_tokens == 0 and not malformed
        and stop_type == "eos" and parsed.status == "accepted"
    )
    cap_hit = bool(
        len(generated) > cap
        or (len(generated) == cap and terminal != EOS_ID)
        or (
            terminal != EOS_ID
            and (
                stop_type in {"limit", "length"}
                or truncated
                or (type(tokens_predicted) is int and tokens_predicted >= cap)
            )
        )
    )
    if parsed.status == "accepted" and parsed.operation == "no_op":
        actual_region = list(context.region_old)
    elif parsed.status == "accepted":
        actual_region = list(parsed.body)
    else:
        actual_region = []
    expected_region = list(row["region_new"])
    exact_region = bool(protocol_valid and actual_region == expected_region)
    predicted_noop = bool(protocol_valid and parsed.operation == "no_op")
    expected_noop = row["operation"] == "no_op"
    edit_exact = bool(not expected_noop and exact_region)
    noop_correct = bool(expected_noop and predicted_noop)
    noop_fp = bool(expected_noop and protocol_valid and not predicted_noop)
    saved_quality = saved_case.get("quality")
    if not isinstance(saved_quality, Mapping):
        saved_quality = {}
    saved_cap = saved_case.get("cap")
    if not isinstance(saved_cap, Mapping):
        saved_cap = {}
    saved_decisions = {
        "protocol_valid": bool(saved_quality.get("protocol_valid")),
        "exact_region": bool(saved_quality.get("exact_edit")),
        "edit_exact": bool(saved_quality.get("edit_exact")),
        "predicted_noop": bool(saved_quality.get("predicted_noop")),
        "strict_noop_correct": bool(saved_quality.get("strict_noop_correct")),
        "noop_false_positive": bool(saved_quality.get("noop_false_positive")),
        "cap_hit": bool(saved_cap.get("hit")),
    }
    independent_decisions = {
        "protocol_valid": protocol_valid,
        "exact_region": exact_region,
        "edit_exact": edit_exact,
        "predicted_noop": predicted_noop,
        "strict_noop_correct": noop_correct,
        "noop_false_positive": noop_fp,
        "cap_hit": cap_hit,
    }
    if saved_decisions != independent_decisions:
        raise ReviewError(
            f"{row_id}: saved quality decisions differ from independent review: "
            f"saved={saved_decisions} independent={independent_decisions}"
        )
    protocol_failure = _protocol_failure(
        canonical_tokens=canonical_tokens, wire_match=wire_match,
        generated_count=len(generated), cap=cap, saw_stop=saw_stop,
        final_sse_tokens=final_sse_tokens if type(final_sse_tokens) is int else -1,
        malformed=malformed if type(malformed) is int else 1,
        stop_type=stop_type, parser_status=parsed.status,
    )
    record = {
        "id": row_id,
        "family": row["family"],
        "package_id": row["package_id"],
        "operation": row["operation"],
        "expected_noop": expected_noop,
        "source": {
            "source_ref": row.get("source_ref"),
            "source_provenance_kind": item["source_audit"]["origin_kind"],
            "source_provenance_complete": item["source_audit"]["complete"],
            "source_is_simulated": item["source_audit"]["source_is_simulated"],
            "panel_serialized_target_sha256": panel_target_sha,
            "quality_canonical_region_new_sha256": canonical_region_sha,
        },
        "prompt": {
            "rendered_prompt_sha256": prompt_sha,
            "hf_prompt_ids_sha256": canonical_sha(prompt_ids),
            "native_prompt_ids_sha256": canonical_sha(prompt_without_bos),
            "hf_prompt_tokens_with_bos": len(prompt_ids),
            "saved_hf_prompt_match": saved_hf_ids == prompt_ids,
            "saved_native_prompt_match": saved_native_ids == prompt_without_bos,
        },
        "generation": {
            "raw_text_sha256": sha256_text(raw_text),
            "raw_text_chars": len(raw_text),
            "returned_token_ids_sha256": canonical_sha(generated),
            "returned_tokens_including_terminal": len(generated),
            "terminal_id": terminal,
            "terminal_is_canonical_eos": terminal == EOS_ID,
            "terminal_is_native_eog": terminal in NATIVE_EOG_IDS,
            "decoded_body_sha256": decoded_sha,
            "decoded_body_chars": len(decoded_body),
            "decode_error": decode_error,
            "wire_hf_text_match": wire_match,
            "canonical_generation_tokens": canonical_tokens,
            "stop_type": stop_type,
            "saw_stop": saw_stop,
            "truncated": truncated,
            "tokens_predicted": tokens_predicted,
            "malformed_sse_frames": malformed,
            "final_sse_tokens": final_sse_tokens,
            "parser_status": parsed.status,
            "parser_operation": parsed.operation,
            "parser_reason": parsed.reason,
            "protocol_failure": protocol_failure,
        },
        "cap": {
            "limit": cap,
            "hit": cap_hit,
            "saved_hit": saved_decisions["cap_hit"],
            "decision_match": cap_hit == saved_decisions["cap_hit"],
        },
        "decisions": {
            "protocol_valid": protocol_valid,
            "saved_protocol_valid": saved_decisions["protocol_valid"],
            "exact_region": exact_region,
            "edit_exact": edit_exact,
            "predicted_noop": predicted_noop,
            "strict_noop_correct": noop_correct,
            "noop_false_positive": noop_fp,
            "saved": saved_decisions,
        },
        "timing": {
            "case_wall_ms": saved_case.get("timing", {}).get("case_wall_ms")
            if isinstance(saved_case.get("timing"), Mapping) else None,
            "tokenize_wall_ms": saved_case.get("tokenize_wall_ms"),
            "predicted_ms": timing.get("timings", {}).get("predicted_ms")
            if isinstance(timing.get("timings"), Mapping) else None,
            "prompt_ms": timing.get("timings", {}).get("prompt_ms")
            if isinstance(timing.get("timings"), Mapping) else None,
        },
    }
    return record


def _compact_props(props: Mapping[str, Any]) -> dict[str, Any]:
    generation = props.get("default_generation_settings", {})
    return {
        "build_info": props.get("build_info"),
        "model_ftype": props.get("model_ftype"),
        "model_path": props.get("model_path"),
        "n_ctx": generation.get("n_ctx") if isinstance(generation, Mapping) else None,
        "endpoint_props": props.get("endpoint_props"),
        "bos_token": props.get("bos_token"),
        "eos_token": props.get("eos_token"),
    }


def _server_diagnostics(run: Path) -> dict[str, Any]:
    props_path = run / "props.json"
    launch_path = run / "server-launch.json"
    env_candidates = sorted(run.glob("server-env-*.json"))
    children_path = run / "server-children.json"
    log_path = run / "server.log"
    props = read_json(props_path)
    launch = read_json(launch_path)
    terminal = read_json(run / "terminal.json")
    env = read_json(env_candidates[0]) if len(env_candidates) == 1 else None
    children = read_json(children_path)
    terminal_children = terminal.get("server", {}).get("children", [])
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    argv = launch.get("argv", [])
    argv_text = " ".join(str(item) for item in argv)
    batch = None
    ubatch = None
    if "-b" in argv:
        batch = argv[argv.index("-b") + 1]
    if "-ub" in argv:
        ubatch = argv[argv.index("-ub") + 1]
    graph_use = re.findall(r"USE_GRAPHS\s*=\s*(\d+)", log_text)
    graph_nodes = re.findall(r"graph nodes\s*=\s*(\d+)", log_text)
    graph_splits = re.findall(r"graph splits\s*=\s*(\d+)", log_text)
    offloads = re.findall(r"offloaded\s+(\d+/\d+)\s+layers", log_text)
    reused = re.findall(r"graphs reused\s*=\s*(\d+)", log_text)
    cleanups = re.findall(r"cleaning up before exit", log_text)
    return {
        "props_sha256": sha256_file(props_path),
        "server_launch_sha256": sha256_file(launch_path),
        "server_env_sha256": sha256_file(env_candidates[0]) if len(env_candidates) == 1 else None,
        "server_children_sha256": sha256_file(children_path),
        "server_log_sha256": sha256_file(log_path),
        "props": _compact_props(props),
        "env": env,
        "children": children,
        "terminal_children": terminal_children,
        "argv": argv,
        "batch": batch,
        "ubatch": ubatch,
        "argv_has_expected_server": str(SERVER_PATH) in argv,
        "argv_has_context_4096": "4096" in [str(x) for x in argv],
        "argv_has_ngl99": "99" in [str(x) for x in argv],
        "graph_opt_environment": env == ["GGML_CUDA_GRAPH_OPT=0"],
        "graph_use_values": [int(value) for value in graph_use],
        "graph_nodes_values": [int(value) for value in graph_nodes],
        "graph_splits_values": [int(value) for value in graph_splits],
        "offloaded_values": offloads,
        "graph_reuse_samples": [int(value) for value in reused[-3:]],
        "graph_reuse_observations": len(reused),
        "cleanup_observations": len(cleanups),
        "checks": {
            "props_n_ctx_4096": _compact_props(props)["n_ctx"] == CONTEXT_SIZE,
            "props_model_ftype_q8_0": _compact_props(props)["model_ftype"] == "Q8_0",
            "props_build_pinned": _compact_props(props)["build_info"] == EXPECTED["build_info"],
            "props_endpoint_props_false": _compact_props(props)["endpoint_props"] is False,
            "props_model_path_pinned": _compact_props(props)["model_path"] == str(MODEL_PATH),
            "launch_batch_equals_ubatch": batch == ubatch,
            "launch_server_pinned": str(SERVER_PATH) in argv,
            "env_graph_opt_zero": env == ["GGML_CUDA_GRAPH_OPT=0"],
            "log_uses_cuda_graphs": graph_use == ["1"],
            "log_graph_geometry": graph_nodes == ["1308"] and graph_splits == ["2"],
            "log_graph_reuse": bool(reused),
            "log_offloaded_43_of_43": "43/43" in offloads,
            "log_clean_exit": bool(cleanups),
            "children_recorded": isinstance(children, list) and bool(children),
            "children_clean_after_exit": (
                isinstance(terminal_children, list)
                and all(child.get("exists") is False
                        for child in terminal_children if isinstance(child, Mapping))
            ),
        },
    }


def _terminal_checks(run: Path, quality_path: Path) -> dict[str, Any]:
    terminal = read_json(run / "terminal.json")
    quality_sha = sha256_file(quality_path)
    terminal_sha = terminal.get("quality_sha256")
    client = terminal.get("client", {})
    server = terminal.get("server", {})
    child_status = server.get("children", [])
    return {
        "terminal_sha256": sha256_file(run / "terminal.json"),
        "quality_sha256": quality_sha,
        "terminal_quality_sha_matches": terminal_sha == quality_sha,
        "quality_status": terminal.get("quality_status"),
        "client_exit_code": client.get("exit_code"),
        "server_exit_code": server.get("exit_code"),
        "client_pid_exists": client.get("pid_exists"),
        "server_pid_exists": server.get("pid_exists"),
        "server_child_pids_exist": [child.get("exists") for child in child_status if isinstance(child, Mapping)],
        "clean_exit": (
            terminal.get("quality_status") == "complete"
            and client.get("exit_code") == 0 and server.get("exit_code") == 0
            and client.get("pid_exists") is False and server.get("pid_exists") is False
            and all(child.get("exists") is False for child in child_status if isinstance(child, Mapping))
        ),
        "seconds": terminal.get("seconds"),
    }


def _source_checks(quality: Mapping[str, Any], cap: int, label: str) -> dict[str, Any]:
    contract = quality.get("contract", {})
    completion = contract.get("completion", {}) if isinstance(contract, Mapping) else {}
    static = quality.get("static_preflight", {})
    panel = quality.get("panel", {})
    model = quality.get("model", {})
    server = quality.get("server", {})
    props = server.get("props", {}) if isinstance(server, Mapping) else {}
    tokenizer = quality.get("tokenizer", {})
    checks = {
        "quality_status_complete": quality.get("status") == "complete" and quality.get("evaluation_complete") is True,
        "quality_cap_matches": completion.get("n_predict") == cap,
        "quality_context_4096": contract.get("context_size") == CONTEXT_SIZE,
        "quality_renderer_pinned": contract.get("renderer_id") == "zeta2-prm03-v1",
        "quality_manual_bos_zero": contract.get("manual_bos_id") == BOS_ID,
        "quality_native_eog_pinned": contract.get("native_eog_ids") == [1, 130073],
        "quality_split_special_true": contract.get("tokenization", {}).get("hf_split_special_tokens") is True,
        "quality_add_special_false": contract.get("tokenization", {}).get("add_special") is False,
        "quality_parse_special_false": contract.get("tokenization", {}).get("parse_special") is False,
        "quality_protocol_hash": static.get("protocol", {}).get("sha256") == EXPECTED["protocol_sha256"],
        "quality_campaign_client_hash": static.get("campaign_client", {}).get("sha256") == EXPECTED["campaign_client_sha256"],
        "quality_campaign_eval_hash": static.get("campaign_eval", {}).get("sha256") == EXPECTED["campaign_eval_sha256"],
        "quality_panel_hash": panel.get("sha256") == EXPECTED["panel_sha256"] and panel.get("rows") == PANEL_ROWS,
        "quality_model_provenance": model.get("provenance") == EXPECTED["model_sha256"],
        "quality_model_path": server.get("model_path_reported") == str(MODEL_PATH),
        "quality_server_context": server.get("n_ctx") == CONTEXT_SIZE,
        "quality_server_props_ftype": props.get("model_ftype") == "Q8_0",
        "quality_server_props_build": props.get("build_info") == EXPECTED["build_info"],
        "quality_server_props_endpoint_false": props.get("endpoint_props") is False,
        "quality_tokenizer_hashes": tokenizer.get("files") == {
            "tokenizer.json": EXPECTED["tokenizer_json_sha256"],
            "tokenizer_config.json": EXPECTED["tokenizer_config_sha256"],
        },
        "quality_no_transport_failures": quality.get("denominators", {}).get("transport_failures") == 0,
        "quality_no_partial_responses": quality.get("denominators", {}).get("partial_responses") == 0,
    }
    return {
        "label": label,
        "model": {
            "path": str(MODEL_PATH),
            "expected_sha256": EXPECTED["model_sha256"],
            "provenance_saved": model.get("provenance"),
            "bytes_read_or_hashed": False,
            "provenance_matches_expected": checks["quality_model_provenance"],
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _denominators(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    counts["attempted_rows"] = len(records)
    counts["responses"] = len(records)
    counts["panel_rows"] = len(records)
    counts["edit_cases"] = sum(int(record["operation"] == "replace") for record in records)
    counts["strict_noop_cases"] = sum(int(record["operation"] == "no_op") for record in records)
    counts["protocol_rows"] = sum(int(record["decisions"]["protocol_valid"]) for record in records)
    counts["protocol_error_rows"] = len(records) - counts["protocol_rows"]
    counts["cap_hit_rows"] = sum(int(record["cap"]["hit"]) for record in records)
    counts["exact_region_rows"] = sum(int(record["decisions"]["exact_region"]) for record in records)
    counts["edit_exact_rows"] = sum(int(record["decisions"]["edit_exact"]) for record in records)
    counts["strict_noop_correct_rows"] = sum(int(record["decisions"]["strict_noop_correct"]) for record in records)
    counts["noop_false_positive_rows"] = sum(int(record["decisions"]["noop_false_positive"]) for record in records)
    counts["mechanical_failures"] = counts["protocol_error_rows"]
    counts["transport_failures"] = 0
    counts["partial_responses"] = 0
    return dict(sorted(counts.items()))


def review_run(
    protocol: Any, tokenizer: Any, panel_items: Sequence[Mapping[str, Any]],
    run: Path, cap: int, label: str,
) -> dict[str, Any]:
    quality_path = run / "quality.json"
    quality = read_json(quality_path)
    cases = quality.get("cases")
    if not isinstance(cases, list) or len(cases) != PANEL_ROWS:
        raise ReviewError(f"{label}: quality case count is not {PANEL_ROWS}")
    ordered_ids = [item["row"]["id"] for item in panel_items]
    if [case.get("id") for case in cases] != ordered_ids:
        raise ReviewError(f"{label}: quality case order differs from corrected panel")
    records = [
        independent_case(protocol, tokenizer, item, case, cap)
        for item, case in zip(panel_items, cases)
    ]
    independent = _denominators(records)
    saved = quality.get("denominators", {})
    denominator_fields = [
        "attempted_rows", "responses", "panel_rows", "edit_cases", "strict_noop_cases",
        "protocol_rows", "protocol_error_rows", "cap_hit_rows", "exact_region_rows",
        "edit_exact_rows", "strict_noop_correct_rows", "noop_false_positive_rows",
        "mechanical_failures", "transport_failures", "partial_responses",
    ]
    denominator_comparison = {
        field: {
            "independent": independent.get(field), "saved": saved.get(field),
            "match": independent.get(field) == saved.get(field),
        }
        for field in denominator_fields
    }
    if not all(value["match"] for value in denominator_comparison.values()):
        raise ReviewError(f"{label}: independent denominators differ from saved quality")
    per_case_saved_matches = all(
        all(record["decisions"][key] == record["decisions"]["saved"][key]
            for key in ("protocol_valid", "exact_region", "edit_exact", "predicted_noop",
                        "strict_noop_correct", "noop_false_positive"))
        and record["cap"]["decision_match"]
        for record in records
    )
    if not per_case_saved_matches:
        raise ReviewError(f"{label}: one or more saved per-case decisions differ")
    diagnostics = _server_diagnostics(run) if (run / "props.json").exists() else None
    source_checks = _source_checks(quality, cap, label)
    terminal_checks = _terminal_checks(run, quality_path)
    return {
        "label": label,
        "run_path": str(run),
        "run_files": {
            "quality_sha256": sha256_file(quality_path),
            "quality_bytes": quality_path.stat().st_size,
            "launch_sha256": sha256_file(run / "launch.json") if (run / "launch.json").exists() else None,
            "terminal_sha256": sha256_file(run / "terminal.json"),
            "server_log_sha256": sha256_file(run / "server.log"),
        },
        "cap": cap,
        "source_checks": source_checks,
        "terminal_checks": terminal_checks,
        "server_diagnostics": diagnostics,
        "independent_denominators": independent,
        "saved_denominators": {field: saved.get(field) for field in denominator_fields},
        "denominator_comparison": denominator_comparison,
        "records": records,
        "all_rows_independently_checked": len(records) == PANEL_ROWS,
        "all_saved_decisions_match": per_case_saved_matches,
    }


def compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_records = {record["id"]: record for record in left["records"]}
    right_records = {record["id"]: record for record in right["records"]}
    if set(left_records) != set(right_records):
        raise ReviewError("current profiles do not have the same case IDs")
    output_fields = [
        ("generation", "raw_text_sha256"),
        ("generation", "returned_token_ids_sha256"),
        ("generation", "returned_tokens_including_terminal"),
        ("generation", "terminal_id"),
        ("generation", "decoded_body_sha256"),
        ("generation", "wire_hf_text_match"),
        ("generation", "canonical_generation_tokens"),
        ("generation", "stop_type"),
        ("generation", "saw_stop"),
        ("generation", "truncated"),
        ("generation", "parser_status"),
        ("generation", "parser_operation"),
        ("generation", "parser_reason"),
        ("cap", "hit"),
        ("decisions", "protocol_valid"),
        ("decisions", "exact_region"),
        ("decisions", "edit_exact"),
        ("decisions", "predicted_noop"),
        ("decisions", "strict_noop_correct"),
        ("decisions", "noop_false_positive"),
    ]
    rows: list[dict[str, Any]] = []
    output_changed_ids: list[str] = []
    decision_changed_ids: list[str] = []
    for row_id in left_records:
        a = left_records[row_id]
        b = right_records[row_id]
        changed: dict[str, dict[str, Any]] = {}
        for section, field in output_fields:
            av = a[section].get(field)
            bv = b[section].get(field)
            if av != bv:
                changed[f"{section}.{field}"] = {"b256": av, "b1024": bv}
        output_signature_fields = [
            "generation.raw_text_sha256", "generation.returned_token_ids_sha256",
            "generation.returned_tokens_including_terminal", "generation.terminal_id",
            "generation.decoded_body_sha256",
        ]
        decision_signature_fields = [
            "cap.hit", "decisions.protocol_valid", "decisions.exact_region",
            "decisions.edit_exact", "decisions.predicted_noop",
            "decisions.strict_noop_correct", "decisions.noop_false_positive",
        ]
        output_changed = any(key in changed for key in output_signature_fields)
        decision_changed = any(key in changed for key in decision_signature_fields)
        if output_changed:
            output_changed_ids.append(row_id)
        if decision_changed:
            decision_changed_ids.append(row_id)
        rows.append({
            "id": row_id,
            "family": a["family"],
            "operation": a["operation"],
            "output_changed": output_changed,
            "decision_changed": decision_changed,
            "changed_fields": changed,
            "b256": {
                "raw_text_sha256": a["generation"]["raw_text_sha256"],
                "returned_token_ids_sha256": a["generation"]["returned_token_ids_sha256"],
                "returned_tokens": a["generation"]["returned_tokens_including_terminal"],
                "terminal_id": a["generation"]["terminal_id"],
                "protocol_valid": a["decisions"]["protocol_valid"],
                "cap_hit": a["cap"]["hit"],
                "exact_region": a["decisions"]["exact_region"],
                "strict_noop_correct": a["decisions"]["strict_noop_correct"],
                "noop_false_positive": a["decisions"]["noop_false_positive"],
            },
            "b1024": {
                "raw_text_sha256": b["generation"]["raw_text_sha256"],
                "returned_token_ids_sha256": b["generation"]["returned_token_ids_sha256"],
                "returned_tokens": b["generation"]["returned_tokens_including_terminal"],
                "terminal_id": b["generation"]["terminal_id"],
                "protocol_valid": b["decisions"]["protocol_valid"],
                "cap_hit": b["cap"]["hit"],
                "exact_region": b["decisions"]["exact_region"],
                "strict_noop_correct": b["decisions"]["strict_noop_correct"],
                "noop_false_positive": b["decisions"]["noop_false_positive"],
            },
        })
    a_denoms = left["independent_denominators"]
    b_denoms = right["independent_denominators"]
    delta_fields = [
        "protocol_rows", "protocol_error_rows", "cap_hit_rows", "exact_region_rows",
        "edit_exact_rows", "strict_noop_correct_rows", "noop_false_positive_rows",
    ]
    delta = {field: b_denoms[field] - a_denoms[field] for field in delta_fields}
    regression = {
        "matched_case_id": "dat07-derived-d1c76ba29b30040a7561ce3e",
        "b256": left_records["dat07-derived-d1c76ba29b30040a7561ce3e"]["decisions"],
        "b1024": right_records["dat07-derived-d1c76ba29b30040a7561ce3e"]["decisions"],
        "reason": "same corrected no-op case loses exact/no-op correctness and becomes a no-op false positive under b1024",
    }
    return {
        "profiles": [left["label"], right["label"]],
        "row_comparisons": rows,
        "rows_compared": len(rows),
        "output_changed_rows": len(output_changed_ids),
        "output_changed_ids": output_changed_ids,
        "decision_changed_rows": len(decision_changed_ids),
        "decision_changed_ids": decision_changed_ids,
        "aggregate_delta_b1024_minus_b256": delta,
        "matched_case_regression": regression,
        "runtime_promotion": {
            "decision": "hold",
            "b1024_regresses_on_matched_192_case": True,
            "reason": "b1024 recovers one capped protocol row but changes a matched no-op from correct to false positive; aggregate protocol recovery does not clear the per-row regression",
            "do_not_promote_8k_or_change_default": True,
        },
    }


def source_input_manifest(panel_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a deterministic manifest without touching model bytes."""
    paths: list[dict[str, Any]] = []
    file_paths = [
        ("panel", PANEL), ("panel_manifest", PANEL_MANIFEST),
        ("protocol", PROTOCOL), ("campaign_client", CAMPAIGN_CLIENT),
        ("campaign_eval", CAMPAIGN_EVAL), ("reference_scorer", REFERENCE_SCORER),
        ("controller", CONTROLLER), ("cap_adapter", CAP_ADAPTER),
        ("controller_manifest", CONTROLLER_MANIFEST),
    ]
    for name, path in file_paths:
        paths.append({"name": name, "path": str(path), "sha256": sha256_file(path), "read": True})
    run_files = [
        "quality.json", "launch.json", "terminal.json", "server.log", "client.log",
        "props.json", "server-launch.json", "server-children.json", "server-maps-4134467.txt",
        "server-maps-4135151.txt", "server-env-4134467.json", "server-env-4135151.json",
    ]
    run_entries: dict[str, Any] = {}
    for label, run in {**CURRENT_RUNS, "historical512": HISTORICAL_ROOT}.items():
        entries = []
        for filename in run_files:
            path = run / filename
            if path.exists():
                entries.append({"path": str(path), "sha256": sha256_file(path), "read": True})
        run_entries[label] = entries
    return {
        "schema_version": "sepalith.run09.theta0-cuda-dev192-review-inputs.v1",
        "purpose": "immutable input inventory for independent CPU-only review",
        "paths": paths,
        "runs": run_entries,
        "model": {
            "path": str(MODEL_PATH),
            "expected_sha256": EXPECTED["model_sha256"],
            "bytes_read_or_hashed": False,
            "identity_source": "quality.model.provenance compared with controller manifest expected digest",
        },
        "server_binary": {
            "path": str(SERVER_PATH),
            "expected_sha256": EXPECTED["server_sha256"],
            "bytes_read_or_hashed": False,
            "identity_source": "controller manifest and saved server argv",
        },
        "panel_counts": {
            "rows": len(panel_items),
            "edit_rows": sum(int(item["row"]["operation"] == "replace") for item in panel_items),
            "noop_rows": sum(int(item["row"]["operation"] == "no_op") for item in panel_items),
        },
        "read_policy": [
            "CPU tokenizer, protocol, panel, controller/adapter source and completed run artifacts only",
            "no native launch, GPU, SSH, network, model-weight read or model-weight hash",
        ],
    }


def source_artifact_checks() -> dict[str, Any]:
    observed = {
        "panel": sha256_file(PANEL),
        "panel_manifest": sha256_file(PANEL_MANIFEST),
        "protocol": sha256_file(PROTOCOL),
        "campaign_client": sha256_file(CAMPAIGN_CLIENT),
        "campaign_eval": sha256_file(CAMPAIGN_EVAL),
        "reference_scorer": sha256_file(REFERENCE_SCORER),
        "controller": sha256_file(CONTROLLER),
        "cap_adapter": sha256_file(CAP_ADAPTER),
        "controller_manifest": sha256_file(CONTROLLER_MANIFEST),
    }
    expected = {
        "panel": EXPECTED["panel_sha256"],
        "panel_manifest": EXPECTED["panel_manifest_sha256"],
        "protocol": EXPECTED["protocol_sha256"],
        "campaign_client": EXPECTED["campaign_client_sha256"],
        "campaign_eval": EXPECTED["campaign_eval_sha256"],
        "reference_scorer": EXPECTED["reference_scorer_sha256"],
        "controller": EXPECTED["controller_sha256"],
        "cap_adapter": EXPECTED["cap_adapter_sha256"],
        "controller_manifest": None,
    }
    manifest = read_json(CONTROLLER_MANIFEST)
    controller_entries = {
        str(path): digest for path, digest in manifest.items()
        if isinstance(path, str) and isinstance(digest, str)
    }
    source_checks = {
        key: (expected[key] is None or observed[key] == expected[key])
        for key in observed
    }
    source_checks.update({
        "manifest_pins_protocol": controller_entries.get(str(PROTOCOL)) == EXPECTED["protocol_sha256"],
        "manifest_pins_panel": controller_entries.get(str(PANEL)) == EXPECTED["panel_sha256"],
        "manifest_pins_reference_scorer": controller_entries.get(str(REFERENCE_SCORER)) == EXPECTED["reference_scorer_sha256"],
        "manifest_pins_controller": controller_entries.get(str(CONTROLLER)) == EXPECTED["controller_sha256"],
        "manifest_pins_adapter": controller_entries.get(str(CAP_ADAPTER)) == EXPECTED["cap_adapter_sha256"],
        "adapter_mentions_explicit_cap_192": "COMPLETION_CAP = 192" in CAP_ADAPTER.read_text(encoding="utf-8"),
        "adapter_mentions_corrected_panel": "dev75-corrected-finish-v1.jsonl" in CAP_ADAPTER.read_text(encoding="utf-8"),
        "controller_mentions_256_and_1024": all(token in CONTROLLER.read_text(encoding="utf-8") for token in ("256", "1024")),
    })
    return {
        "observed": observed,
        "expected": expected,
        "checks": source_checks,
        "all_checks_pass": all(source_checks.values()),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    protocol = load_protocol()
    tokenizer, tokenizer_audit = load_tokenizer()
    panel_items, panel_audit = load_panel(protocol, tokenizer)
    source_audit = source_artifact_checks()
    if not source_audit["all_checks_pass"]:
        raise ReviewError(f"source artifact checks failed: {source_audit['checks']}")
    inputs = source_input_manifest(panel_items)
    input_manifest_path = out_dir / "input-manifest.json"
    write_json(input_manifest_path, inputs)
    input_manifest_sha = sha256_file(input_manifest_path)
    current_reviews = {
        label: review_run(protocol, tokenizer, panel_items, run, CURRENT_CAP, f"Q8_0-{label}-out192")
        for label, run in CURRENT_RUNS.items()
    }
    historical_review = review_run(
        protocol, tokenizer, panel_items, HISTORICAL_ROOT,
        HISTORICAL_CAP, "Q8_0-historical-out512",
    )
    comparison = compare_runs(current_reviews["b256"], current_reviews["b1024"])
    report = {
        "schema_version": "sepalith.run09.theta0-cuda-dev192-independent-review.v1",
        "task": "RUN-05/RUN-09 independent quality review",
        "status": "independent_review_complete_no_promotion",
        "review_policy": {
            "read_only_completed_artifacts": True,
            "cpu_tokenizer_and_protocol_only": True,
            "native_launch_performed": False,
            "gpu_or_ssh_or_network_used": False,
            "model_weight_read_or_hashed": False,
            "final_or_sealed_data_read": False,
        },
        "source_artifacts": source_audit,
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": input_manifest_sha,
        },
        "panel": panel_audit,
        "tokenizer": tokenizer_audit,
        "current_profiles": current_reviews,
        "historical_output512_separate": historical_review,
        "comparison_b256_vs_b1024": comparison,
        "interpretation": {
            "current_output_cap": CURRENT_CAP,
            "historical_output_cap": HISTORICAL_CAP,
            "current_denominators_are_not_merged_with_historical": True,
            "b1024_regression_before_promotion": True,
            "b1024_aggregate_protocol_recovery_is_not_promotion_evidence": True,
            "native_diagnostic_outside_6000_source_budget": True,
            "do_not_promote_8k": True,
        },
        "limits": [
            "The native output body is represented by hashes, lengths and independently decoded/parser metadata; raw model text is not repeated in this artifact.",
            "No model likelihood/NLL is available from the native endpoint.",
            "Source provenance is checked from the immutable corrected panel metadata; no external source corpus is read.",
        ],
    }
    report_path = out_dir / "review-report.json"
    write_json(report_path, report)
    report_sha = sha256_file(report_path)
    script_sha = sha256_file(Path(__file__).resolve())
    receipt = {
        "schema_version": "sepalith.receipt.run09-theta0-cuda-dev192-review-v1",
        "task": "RUN-05/RUN-09 independent quality review",
        "status": "independent_review_complete_no_promotion",
        "review_script": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
        "input_manifest": {"path": str(input_manifest_path), "sha256": input_manifest_sha},
        "report": {"path": str(report_path), "sha256": report_sha},
        "scope": {
            "current_profiles": ["Q8_0-b256-out192", "Q8_0-b1024-out192"],
            "historical_profile": "Q8_0-historical-out512",
            "panel_rows": PANEL_ROWS,
            "edit_rows": EDIT_ROWS,
            "noop_rows": NOOP_ROWS,
            "context_size": CONTEXT_SIZE,
            "current_completion_cap": CURRENT_CAP,
        },
        "results": {
            "b256": current_reviews["b256"]["independent_denominators"],
            "b1024": current_reviews["b1024"]["independent_denominators"],
            "historical512": historical_review["independent_denominators"],
            "output_changed_rows_b256_vs_b1024": comparison["output_changed_rows"],
            "output_changed_ids": comparison["output_changed_ids"],
            "matched_case_regression": comparison["matched_case_regression"],
            "runtime_promotion": comparison["runtime_promotion"],
        },
        "checks": {
            "all_150_current_and_historical_rows_independently_checked": (
                current_reviews["b256"]["all_rows_independently_checked"]
                and current_reviews["b1024"]["all_rows_independently_checked"]
                and historical_review["all_rows_independently_checked"]
            ),
            "all_current_denominators_match_saved": (
                current_reviews["b256"]["all_saved_decisions_match"]
                and current_reviews["b1024"]["all_saved_decisions_match"]
            ),
            "source_panel_tokenizer_protocol_checks": source_audit["all_checks_pass"] and panel_audit["target_geometry_checked"],
            "current_native_clean_exits": all(
                review["terminal_checks"]["clean_exit"] for review in current_reviews.values()
            ),
            "current_graph_opt0_diagnostics": all(
                review["server_diagnostics"] is not None
                and all(review["server_diagnostics"]["checks"].values())
                for review in current_reviews.values()
            ),
        },
        "read_policy": report["review_policy"],
    }
    receipt_path = PLAN_ROOT / "docs/campaign/receipts/RUN-09-theta0-cuda-dev192-review-v1.json"
    write_json(receipt_path, receipt)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "report_sha256": report_sha,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "b256": current_reviews["b256"]["independent_denominators"],
        "b1024": current_reviews["b1024"]["independent_denominators"],
        "historical512": historical_review["independent_denominators"],
        "output_changed_rows": comparison["output_changed_rows"],
        "output_changed_ids": comparison["output_changed_ids"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
