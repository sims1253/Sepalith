#!/usr/bin/env python3
"""Fail-closed MiniCPM tokenizer identity and pad-token repair.

The pinned MiniCPM tokenizer intentionally aliases ``pad`` to protocol EOS 1.
The installed Unsloth loader repairs an EOS pad by selecting an existing
``<unused_token_477>`` token.  That is a useful generic default for many
models, but it changes this campaign's identity and can change the padding
semantics used by the full-text collator.

This module is framework-free at import time.  The live loaders call
``restore_pinned_tokenizer_contract`` only after their resource guard and
model load.  Restoration assigns the already-vocabulary token ``</s>`` as
pad, verifies that the vocabulary mapping did not change, stamps both model
configs, and compares selected complete PRM-03 prompts against a separately
loaded local reference tokenizer.  It never adds, removes, or reorders a
vocabulary entry and never changes model weights.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Keep this helper importable from the training directory and from the
# campaign runner's explicit ``PYTHONPATH``.  No model/framework import occurs.
_EXECUTION_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_SRC = _EXECUTION_ROOT / "packages" / "sepalith" / "src"
if str(_PROTOCOL_SRC) not in sys.path:
    sys.path.insert(0, str(_PROTOCOL_SRC))

from sepalith.campaign_protocol import BOS_ID, EOS_ID, NATIVE_EOG_IDS, VOCAB_SIZE


EXPECTED_TOKENIZER_IDENTITY = (VOCAB_SIZE, BOS_ID, EOS_ID, EOS_ID)


class TokenizerContractError(ValueError):
    """A loaded tokenizer/model pair cannot satisfy the pinned identity."""


def _tokenizer_vocab(tokenizer: Any) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TokenizerContractError("loaded tokenizer does not expose get_vocab()")
    try:
        value = get_vocab()
    except Exception as error:  # pragma: no cover - backend-specific failure
        raise TokenizerContractError(f"cannot read loaded tokenizer vocabulary: {error}") from error
    if not isinstance(value, Mapping):
        raise TokenizerContractError("loaded tokenizer vocabulary is not a mapping")
    result: dict[str, int] = {}
    for token, token_id in value.items():
        if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
            raise TokenizerContractError("loaded tokenizer vocabulary contains an invalid token ID")
        result[token] = token_id
    return result


def _vocab_digest(vocab: Mapping[str, int]) -> str:
    encoded = json.dumps(sorted(vocab.items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _set_attr(value: Any, name: str, new_value: Any) -> None:
    if value is None:
        return
    try:
        setattr(value, name, new_value)
    except Exception:
        update = getattr(value, "update", None)
        if not callable(update):
            raise
        update({name: new_value})


def _id_list(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if type(value) is int:
        return (value,)
    if isinstance(value, (list, tuple)) and all(type(item) is int for item in value):
        return tuple(value)
    return None


def _token_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _tokenizer_summary(tokenizer: Any) -> dict[str, Any]:
    try:
        length = len(tokenizer)
    except Exception as error:
        raise TokenizerContractError(f"cannot determine tokenizer length: {error}") from error
    vocab = _tokenizer_vocab(tokenizer)
    summary = {
        "vocab_size": length,
        "vocab_entries": len(vocab),
        "vocab_sha256": _vocab_digest(vocab),
        "bos_token_id": _attr(tokenizer, "bos_token_id"),
        "eos_token_id": _attr(tokenizer, "eos_token_id"),
        "pad_token_id": _attr(tokenizer, "pad_token_id"),
        "pad_token": _token_text(_attr(tokenizer, "pad_token")),
        "eos_token": _token_text(_attr(tokenizer, "eos_token")),
    }
    return summary


def _validate_pinned_tokenizer(tokenizer: Any, *, label: str) -> dict[str, Any]:
    summary = _tokenizer_summary(tokenizer)
    identity = (
        summary["vocab_size"],
        summary["bos_token_id"],
        summary["eos_token_id"],
        summary["pad_token_id"],
    )
    if identity != EXPECTED_TOKENIZER_IDENTITY:
        raise TokenizerContractError(
            f"{label} tokenizer identity is not pinned: "
            f"{identity}; expected {EXPECTED_TOKENIZER_IDENTITY}"
        )
    token_for_eos = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(token_for_eos) or _token_text(token_for_eos(EOS_ID)) != "</s>":
        raise TokenizerContractError(f"{label} EOS ID 1 does not resolve to </s>")
    return summary


def load_pinned_reference_tokenizer(model_path: Path) -> Any:
    """Load only the local reference tokenizer; never load model weights.

    The import is deliberately inside this function so ``--preflight`` keeps
    the profile/RL modules free of Transformers, torch, Unsloth and TRL.
    """
    path = Path(model_path).resolve()
    if not path.is_absolute() or not path.is_dir():
        raise TokenizerContractError(f"reference tokenizer path is not a local directory: {path}")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, use_fast=True, trust_remote_code=False,
        )
    except Exception as error:  # pragma: no cover - exercised by live environment
        raise TokenizerContractError(f"cannot load local pinned reference tokenizer: {error}") from error
    _validate_pinned_tokenizer(tokenizer, label="reference")
    return tokenizer


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TokenizerContractError("loaded tokenizer does not expose encode()")
    try:
        values = encode(text, add_special_tokens=False, split_special_tokens=True)
    except TypeError as error:
        raise TokenizerContractError(
            "tokenizer encode() did not accept the required split_special_tokens=True mode"
        ) from error
    if not isinstance(values, (list, tuple)) or any(type(value) is not int for value in values):
        raise TokenizerContractError("tokenizer returned non-integer prompt IDs")
    return list(values)


def verify_selected_prompt_parity(
    loaded_tokenizer: Any,
    reference_tokenizer: Any,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare loaded/reference IDs with stored complete PRM-03 prompt IDs."""
    checked: list[dict[str, Any]] = []
    for row in prompt_rows:
        row_id = row.get("id")
        text = row.get("prompt_text")
        stored = row.get("input_ids")
        target_start = row.get("target_start")
        if (
            not isinstance(row_id, str)
            or not isinstance(text, str)
            or not isinstance(stored, (list, tuple))
            or type(target_start) is not int
            or not 1 <= target_start <= len(stored)
            or any(type(token) is not int for token in stored[:target_start])
        ):
            raise TokenizerContractError(f"row {row_id!r} lacks a complete PRM-03 prompt boundary")
        stored_prompt = list(stored[:target_start])
        if stored_prompt[0] != BOS_ID:
            raise TokenizerContractError(f"row {row_id} does not begin with manual BOS {BOS_ID}")
        expected_without_bos = stored_prompt[1:]
        loaded_ids = _encode_without_special_tokens(loaded_tokenizer, text)
        reference_ids = _encode_without_special_tokens(reference_tokenizer, text)
        if loaded_ids != reference_ids:
            raise TokenizerContractError(f"row {row_id} loaded/reference prompt token IDs differ")
        if loaded_ids != expected_without_bos:
            raise TokenizerContractError(f"row {row_id} does not match its stored PRM-03 whole prompt IDs")
        checked.append({
            "id": row_id,
            "prompt_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
            "stored_prompt_tokens": len(stored_prompt),
            "encoded_without_bos_tokens": len(loaded_ids),
            "manual_bos_id": BOS_ID,
            "exact": True,
        })
    return {"checked_rows": len(checked), "rows": checked}


def _repair_model_padding(model: Any, old_pad_id: Any) -> int:
    """Align embedding padding metadata when Unsloth stamped the old ID."""
    if model is None or type(old_pad_id) is not int:
        return 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0
    changed = 0
    for module in modules():
        if getattr(module, "padding_idx", None) == old_pad_id:
            _set_attr(module, "padding_idx", EOS_ID)
            changed += 1
    return changed


def restore_pinned_tokenizer_contract(
    model: Any,
    tokenizer: Any,
    *,
    reference_tokenizer: Any | None = None,
    prompt_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Restore PAD=EOS=1 and prove no vocabulary mapping changed.

    ``reference_tokenizer`` must be loaded directly from the pinned local
    tokenizer files.  Supplying it is required by the live loaders so the
    returned Unsloth tokenizer cannot silently become a self-consistent but
    stale tokenizer.  Prompt rows are complete PRM-03 rows selected by an
    explicit hashed train manifest; no prompt is synthesized from a target.
    """
    before = _tokenizer_summary(tokenizer)
    if before["vocab_size"] != VOCAB_SIZE or before["vocab_entries"] != VOCAB_SIZE:
        raise TokenizerContractError(
            f"loaded tokenizer vocabulary size is not pinned: {before['vocab_size']}/{before['vocab_entries']}"
        )
    if before["bos_token_id"] != BOS_ID or before["eos_token_id"] != EOS_ID:
        raise TokenizerContractError(
            f"loaded tokenizer changed BOS/EOS before repair: "
            f"{before['bos_token_id']}/{before['eos_token_id']}"
        )
    before_vocab = _tokenizer_vocab(tokenizer)
    if reference_tokenizer is not None:
        reference = _validate_pinned_tokenizer(reference_tokenizer, label="reference")
        reference_vocab = _tokenizer_vocab(reference_tokenizer)
        if before_vocab != reference_vocab:
            raise TokenizerContractError(
                "loaded tokenizer vocabulary mapping differs from pinned reference before pad repair"
            )
    else:
        reference = None

    eos_token = _token_text(_attr(tokenizer, "eos_token"))
    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert_ids_to_tokens) or _token_text(convert_ids_to_tokens(EOS_ID)) != eos_token:
        raise TokenizerContractError("loaded tokenizer cannot resolve its pinned EOS token")
    old_pad_id = before["pad_token_id"]
    try:
        # ``</s>`` already occupies ID 1 in the pinned mapping.  Assigning the
        # existing string updates special-token metadata only; it must not call
        # add_special_tokens or resize embeddings.
        tokenizer.pad_token = eos_token
    except Exception as error:
        raise TokenizerContractError(f"cannot restore tokenizer pad token to EOS: {error}") from error
    after = _tokenizer_summary(tokenizer)
    after_vocab = _tokenizer_vocab(tokenizer)
    if after["vocab_size"] != before["vocab_size"] or after_vocab != before_vocab:
        raise TokenizerContractError("pad repair added, removed, or reordered tokenizer vocabulary entries")
    if after["pad_token_id"] != EOS_ID:
        raise TokenizerContractError(
            f"pad repair did not restore pinned ID {EOS_ID}: {after['pad_token_id']}"
        )
    if after["bos_token_id"] != BOS_ID or after["eos_token_id"] != EOS_ID:
        raise TokenizerContractError("pad repair changed pinned BOS/EOS IDs")
    if reference is not None and after_vocab != _tokenizer_vocab(reference_tokenizer):
        raise TokenizerContractError("restored tokenizer vocabulary differs from pinned reference")

    model_config = _attr(model, "config")
    generation_config = _attr(model, "generation_config")
    config_before = {
        "bos_token_id": _attr(model_config, "bos_token_id"),
        "eos_token_id": _attr(model_config, "eos_token_id"),
        "pad_token_id": _attr(model_config, "pad_token_id"),
    }
    generation_before = {
        "bos_token_id": _attr(generation_config, "bos_token_id"),
        "eos_token_id": _attr(generation_config, "eos_token_id"),
        "pad_token_id": _attr(generation_config, "pad_token_id"),
    }
    for config in (model_config, generation_config):
        if config is None:
            continue
        if _attr(config, "bos_token_id") != BOS_ID:
            raise TokenizerContractError("model configuration changed pinned BOS ID")
        eos_ids = _id_list(_attr(config, "eos_token_id"))
        if eos_ids != tuple(NATIVE_EOG_IDS):
            raise TokenizerContractError(
                f"model configuration changed pinned EOG IDs: {eos_ids}; expected {NATIVE_EOG_IDS}"
            )
        _set_attr(config, "pad_token_id", EOS_ID)
    embedding_padding_updates = _repair_model_padding(model, old_pad_id)
    config_after = {
        "bos_token_id": _attr(model_config, "bos_token_id"),
        "eos_token_id": _attr(model_config, "eos_token_id"),
        "pad_token_id": _attr(model_config, "pad_token_id"),
    }
    generation_after = {
        "bos_token_id": _attr(generation_config, "bos_token_id"),
        "eos_token_id": _attr(generation_config, "eos_token_id"),
        "pad_token_id": _attr(generation_config, "pad_token_id"),
    }
    for name, config in (("model", config_after), ("generation", generation_after)):
        if config["pad_token_id"] is not None and config["pad_token_id"] != EOS_ID:
            raise TokenizerContractError(f"{name} configuration pad ID was not restored to {EOS_ID}")

    prompt_parity = None
    if reference_tokenizer is not None:
        prompt_parity = verify_selected_prompt_parity(tokenizer, reference_tokenizer, prompt_rows)
    return {
        "status": "verified",
        "contract": "pinned_minicpm_bos0_eos1_pad1_vocab130560_v1",
        "before": before,
        "after": after,
        "reference": reference,
        "vocab_mapping_unchanged": before_vocab == after_vocab,
        "model_config_before": config_before,
        "model_config_after": config_after,
        "generation_config_before": generation_before,
        "generation_config_after": generation_after,
        "embedding_padding_updates": embedding_padding_updates,
        "prompt_parity": prompt_parity,
        "added_tokens": False,
        "weights_changed": False,
    }
