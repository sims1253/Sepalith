"""Metadata-only gate for the public MiniCPM5 DSpark warm start.

The public checkpoint is a candidate initialized draft, not evidence that it
matches a campaign target variant.  This module only compares small config
metadata.  It never downloads or opens the safetensors payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from minicpm5_dspark_config import DEFAULT_EOG_TOKEN_IDS


PUBLIC_MODEL_ID = "openbmb/MiniCPM5-2B-DSpark"
PUBLIC_REVISION = "114a20fdbf53220712c7fbdd7dccddbf1dedebb4"
PUBLIC_CONFIG_BLOB_OID = "5f2d826e2d137bc3e77a4b0263a9ce7942cd1018"
PUBLIC_CONFIG_URL = f"https://huggingface.co/{PUBLIC_MODEL_ID}/raw/{PUBLIC_REVISION}/config.json"
PUBLIC_WEIGHTS_SHA256 = "ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97"
PUBLIC_WEIGHTS_XET_HASH = "9d964ccffe1dc34b319be30a72f82e26328ac401c593736f589a343901e33bde"
PUBLIC_WEIGHTS_BYTES = 647558522
PUBLIC_DRAFT_PARAMETERS = 323776001
PUBLIC_TARGET_MODEL_ID = "openbmb/MiniCPM5-2B"
PUBLIC_TARGET_LAYER_IDS = (1, 10, 20, 30, 39)
PUBLIC_MASK_TOKEN_ID = 75982

# Names/prefixes follow Qwen3DSparkModel.__init__ in the pinned DeepSpec
# modeling source.  Exact shape/key enumeration still belongs to cloud
# admission after the public safetensors header is inspected.
EXPECTED_TENSOR_PREFIXES = (
    "embed_tokens.weight",
    "layers.",
    "norm.weight",
    "fc.weight",
    "hidden_norm.weight",
    "lm_head.weight",
    "markov_head.",
    "confidence_head.",
)


def public_config_contract() -> dict[str, Any]:
    return {
        "model_id": PUBLIC_MODEL_ID,
        "revision": PUBLIC_REVISION,
        "config_blob_oid": PUBLIC_CONFIG_BLOB_OID,
        "config_url": PUBLIC_CONFIG_URL,
        "weights_sha256": PUBLIC_WEIGHTS_SHA256,
        "weights_xet_hash": PUBLIC_WEIGHTS_XET_HASH,
        "weights_bytes": PUBLIC_WEIGHTS_BYTES,
        "draft_parameters": PUBLIC_DRAFT_PARAMETERS,
        "target_model_id": PUBLIC_TARGET_MODEL_ID,
        "draft_architecture": "Qwen3DSparkModel",
        "model_type": "qwen3",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "vocab_size": 130560,
        "draft_vocab_size": 130560,
        "num_target_layers": 42,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "block_size": 7,
        "target_layer_ids": list(PUBLIC_TARGET_LAYER_IDS),
        "mask_token_id": PUBLIC_MASK_TOKEN_ID,
        "eos_token_id": list(DEFAULT_EOG_TOKEN_IDS),
        "sliding_window": None,
        "tensor_prefix_inventory": list(EXPECTED_TENSOR_PREFIXES),
    }


def compare_target_config(target_config: Mapping[str, Any]) -> dict[str, Any]:
    """Check structural warm-start compatibility without reading weights."""
    checks = {
        "hidden_size": int(target_config.get("hidden_size", -1)) == 2048,
        "intermediate_size": int(target_config.get("intermediate_size", -1)) == 6144,
        "vocab_size": int(target_config.get("vocab_size", -1)) == 130560,
        "num_target_layers": int(target_config.get("num_hidden_layers", -1)) == 42,
        "num_attention_heads": int(target_config.get("num_attention_heads", -1)) == 16,
        "num_key_value_heads": int(target_config.get("num_key_value_heads", -1)) == 2,
        "head_dim": int(target_config.get("head_dim", -1)) == 128,
        "eos_token_id": tuple(int(x) for x in target_config.get("eos_token_id", ())) == DEFAULT_EOG_TOKEN_IDS,
        "tokenizer_required": True,
    }
    return {
        "structurally_compatible": all(checks.values()),
        "checks": checks,
        "target_model_type": target_config.get("model_type"),
        "target_identity_still_required": True,
        "teacher_cache_still_required": True,
    }
