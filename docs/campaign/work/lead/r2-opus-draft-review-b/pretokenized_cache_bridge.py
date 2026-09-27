"""Bridge exact target generations to DeepSpec's v2 cache wire contract.

The upstream data collator is chat-template based and expects ``conversations``.
R2 rows are already tokenized, so the bridge keeps ``input_ids`` and the
explicit target-generation ``loss_mask`` intact.  It also documents the v2
writer's actual behavior: EOS is not a cache requirement.  A cap-hit row may
therefore be cached as-is when it has at least one valid generated token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import struct
from typing import Any


CACHE_SCHEMA = "deepspec_target_cache_v2"
CACHE_INDEX_STRUCT = struct.Struct("<QIIQQQQQ")
CACHE_INDEX_BYTES = CACHE_INDEX_STRUCT.size
CACHE_WRITER_REQUIRES_EOS = False
TARGET_CACHE_SOURCE = (
    "https://raw.githubusercontent.com/deepseek-ai/DeepSpec/"
    "005e03b81cec38b7da6399833d609ee89a2587f2/"
    "scripts/data/prepare_target_cache.py"
)
DATASET_SOURCE = (
    "https://raw.githubusercontent.com/deepseek-ai/DeepSpec/"
    "005e03b81cec38b7da6399833d609ee89a2587f2/"
    "deepspec/data/target_cache_dataset.py"
)


def cache_payload_nbytes(
    sequence_length: int,
    *,
    hidden_size: int = 2048,
    target_taps: int = 5,
    hidden_dtype_bytes: int = 2,
) -> int:
    """Size of one v2 payload: int32 IDs + uint8 masks + BF16 hidden states."""
    if min(sequence_length, hidden_size, target_taps, hidden_dtype_bytes) <= 0:
        raise ValueError("cache dimensions must be positive")
    per_token = 4 + 1 + 1 + hidden_dtype_bytes * hidden_size * (target_taps + 1)
    return int(sequence_length) * per_token


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(dimension) for dimension in value.shape)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return (0,)
        return (len(value),) + _shape(value[0])
    return ()


def _int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must be a list of integers")
    return list(value)


def validate_cache_sample(
    record: Mapping[str, Any],
    target_hidden_states: Any,
    target_last_hidden_states: Any,
    *,
    hidden_size: int,
    target_taps: int,
) -> dict[str, Any]:
    """Validate one exact-token sample and return metadata for the writer."""
    input_ids = _int_list(record.get("input_ids"), "input_ids")
    loss_mask = _int_list(record.get("loss_mask"), "loss_mask")
    if not input_ids:
        raise ValueError("cache samples cannot have an empty input")
    if len(loss_mask) != len(input_ids) or any(mask not in (0, 1) for mask in loss_mask):
        raise ValueError("loss_mask must be binary and align with input_ids")
    hidden_shape = _shape(target_hidden_states)
    last_shape = _shape(target_last_hidden_states)
    expected_hidden = (len(input_ids), target_taps * hidden_size)
    expected_last = (len(input_ids), hidden_size)
    if hidden_shape != expected_hidden:
        raise ValueError(f"target hidden shape {hidden_shape} != {expected_hidden}")
    if last_shape != expected_last:
        raise ValueError(f"target last hidden shape {last_shape} != {expected_last}")
    return {
        "sequence_length": len(input_ids),
        "target_token_count": sum(loss_mask),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "loss_mask": loss_mask,
        "target_hidden_shape": list(hidden_shape),
        "target_last_hidden_shape": list(last_shape),
        "native_stop_token_id": record.get("native_stop_token_id"),
        "generation_status": record.get("generation_status"),
        "protocol_status": record.get("protocol_status"),
        "eos_was_synthesized": bool(record.get("eos_was_synthesized", False)),
    }


def build_cache_sample(
    record: Mapping[str, Any],
    target_hidden_states: Any,
    target_last_hidden_states: Any,
    *,
    hidden_size: int = 2048,
    target_taps: int = 5,
) -> dict[str, Any]:
    """Return writer-ready metadata while retaining exact tensor references."""
    metadata = validate_cache_sample(
        record,
        target_hidden_states,
        target_last_hidden_states,
        hidden_size=hidden_size,
        target_taps=target_taps,
    )
    metadata["target_hidden_states"] = target_hidden_states
    metadata["target_last_hidden_states"] = target_last_hidden_states
    metadata["cache_writer_requires_eos"] = CACHE_WRITER_REQUIRES_EOS
    metadata["eos_injection"] = "forbidden"
    return metadata


def collate_pretokenized(samples: Sequence[Mapping[str, Any]], *, pad_token_id: int = 1) -> dict[str, list[list[int]]]:
    """CPU-friendly list collator matching CacheCollator's padded fields."""
    if not samples:
        raise ValueError("at least one sample is required")
    lengths = [len(_int_list(sample.get("input_ids"), "input_ids")) for sample in samples]
    width = max(lengths)
    input_batch: list[list[int]] = []
    attention_batch: list[list[int]] = []
    loss_batch: list[list[int]] = []
    for sample, length in zip(samples, lengths):
        ids = _int_list(sample.get("input_ids"), "input_ids")
        mask = _int_list(sample.get("loss_mask"), "loss_mask")
        if len(mask) != length:
            raise ValueError("loss mask and input IDs must align")
        input_batch.append(ids + [pad_token_id] * (width - length))
        attention_batch.append([1] * length + [0] * (width - length))
        loss_batch.append(mask + [0] * (width - length))
    return {
        "input_ids": input_batch,
        "attention_mask": attention_batch,
        "loss_mask": loss_batch,
    }


def pack_index_record(
    *,
    sample_id: int,
    sequence_length: int,
    input_offset: int,
    attention_offset: int,
    loss_offset: int,
    target_hidden_offset: int,
    target_last_hidden_offset: int,
    shard_id: int = 0,
) -> bytes:
    """Pack the v2 index tuple; all offsets are byte offsets.

    The upstream order is ``sample_id, shard_id, seq_len,`` followed by the
    five tensor offsets.  Keep the keyword-friendly API while preserving that
    exact wire order.
    """
    values = (
        int(sample_id),
        int(shard_id),
        int(sequence_length),
        int(input_offset),
        int(attention_offset),
        int(loss_offset),
        int(target_hidden_offset),
        int(target_last_hidden_offset),
    )
    if any(value < 0 for value in values):
        raise ValueError("index fields must be non-negative")
    return CACHE_INDEX_STRUCT.pack(*values)


def unpack_index_record(raw: bytes) -> dict[str, int]:
    if len(raw) != CACHE_INDEX_BYTES:
        raise ValueError(f"index record must be {CACHE_INDEX_BYTES} bytes")
    sample_id, shard_id, seq_len, input_offset, attention_offset, loss_offset, target_hidden_offset, target_last_hidden_offset = CACHE_INDEX_STRUCT.unpack(raw)
    return {
        "sample_id": sample_id,
        "shard_id": shard_id,
        "seq_len": seq_len,
        "sequence_length": seq_len,
        "input_offset": input_offset,
        "attention_offset": attention_offset,
        "loss_offset": loss_offset,
        "target_hidden_offset": target_hidden_offset,
        "target_last_hidden_offset": target_last_hidden_offset,
    }


def cache_manifest(
    *,
    target_identity: Mapping[str, Any],
    source_rows_sha256: str,
    tokenizer_sha256: str,
    renderer: str,
    selected_row_ids: Sequence[str],
    hidden_size: int = 2048,
    target_taps: int = 5,
    target_layer_ids: Sequence[int] = (1, 10, 20, 30, 39),
    shards: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the provenance gate consumed before any training launch."""
    selected_digest = sha256("\n".join(selected_row_ids).encode("utf-8")).hexdigest()
    shard_list = [dict(shard) for shard in (shards or [])]
    return {
        # Canonical fields consumed by DeepSpec's validate_target_cache_manifest.
        "version": 2,
        "num_samples": len(selected_row_ids),
        "num_shards": len(shard_list),
        "target_layer_ids": [int(layer_id) for layer_id in target_layer_ids],
        "hidden_dtype": "bfloat16",
        "token_dtype": "int32",
        "mask_dtype": "uint8",
        "index_record_size": CACHE_INDEX_BYTES,
        "hidden_size": int(hidden_size),
        "shards": shard_list,
        # R2 provenance fields consumed by the campaign receipt.
        "schema": CACHE_SCHEMA,
        "target_identity": dict(target_identity),
        "target_model_name_or_path": target_identity.get("name"),
        "source_rows_sha256": source_rows_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "renderer": renderer,
        "selected_row_count": len(selected_row_ids),
        "selected_row_ids_sha256": selected_digest,
        "target_taps": int(target_taps),
        "cache_writer_requires_eos": CACHE_WRITER_REQUIRES_EOS,
        "native_serving_stop_ids": [1, 130073],
        "loss_mask_policy": "ones_on_exact_target_generated_tokens_only",
        "authored_tail_substitution": "forbidden",
        "target_cache_source": TARGET_CACHE_SOURCE,
        "dataset_source": DATASET_SOURCE,
    }


def manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


class PretokenizedTargetCollator:
    """Drop-in replacement for DeepSpec's conversation collator.

    It returns the exact tensor keys consumed by ``prepare_target_cache.py``
    and accepts a minimum of one generated token by default.  Set
    ``min_loss_tokens`` explicitly if a later quality policy wants to filter
    short rows; filtering is row-level and never mutates another row's tail.
    """

    def __init__(self, *, pad_token_id: int = 1, min_loss_tokens: int = 1) -> None:
        if min_loss_tokens <= 0:
            raise ValueError("min_loss_tokens must be positive")
        self.pad_token_id = int(pad_token_id)
        self.min_loss_tokens = int(min_loss_tokens)

    def __call__(self, records: Sequence[Mapping[str, Any]]):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("PretokenizedTargetCollator requires torch") from exc
        valid: list[tuple[list[int], list[int]]] = []
        for record in records:
            ids = _int_list(record.get("input_ids"), "input_ids")
            mask = _int_list(record.get("loss_mask"), "loss_mask")
            if len(ids) != len(mask) or not ids:
                continue
            if sum(mask) < self.min_loss_tokens:
                continue
            valid.append((ids, mask))
        if not valid:
            return None
        width = max(len(ids) for ids, _mask in valid)
        input_batch = []
        attention_batch = []
        loss_batch = []
        for ids, mask in valid:
            pad = width - len(ids)
            input_batch.append(ids + [self.pad_token_id] * pad)
            attention_batch.append([1] * len(ids) + [0] * pad)
            loss_batch.append(mask + [0] * pad)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.uint8),
            "loss_mask": torch.tensor(loss_batch, dtype=torch.uint8),
        }
