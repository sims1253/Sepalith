"""Small, dependency-free DSpark config bridge for the MiniCPM5 target.

This module does not load model weights.  It captures the configuration seam
needed when DeepSpec's Qwen3 DSpark implementation is used with a MiniCPM5
LlamaConfig: Qwen3DSparkAttention reads ``config.sliding_window`` even when
all draft layers use full attention.  The upstream builder currently copies
fields from the target config without adding that optional field.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


DEEPSPEC_REPOSITORY = "https://github.com/deepseek-ai/DeepSpec"
DEEPSPEC_REVISION = "005e03b81cec38b7da6399833d609ee89a2587f2"
DEEPSPEC_MODEL_SOURCE = (
    "deepspec/modeling/dspark/qwen3/modeling.py"
)
DEEPSPEC_CONFIG_SOURCE = "deepspec/modeling/dspark/qwen3/config.py"

TARGET_CONFIG_PATH = (
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/"
    "minicpm5-2b-midtrain-native/config.json"
)
TARGET_CONFIG_SHA256 = "59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180"
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"

# This is the geometry requested by the R2 campaign.  The final target
# binding must be recorded again after the teacher checkpoint is selected.
DEFAULT_DRAFT_LAYERS = 5
DEFAULT_BLOCK_SIZE = 7
DEFAULT_TARGET_LAYER_IDS = (1, 10, 20, 30, 39)
DEFAULT_MASK_TOKEN_ID = 75982
DEFAULT_NUM_ANCHORS = 512
DEFAULT_MARKOV_RANK = 256
DEFAULT_EOG_TOKEN_IDS = (1, 130073)


def _mapping(config: Any) -> dict[str, Any]:
    """Return a mutable copy from a Transformers config or a mapping."""
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return deepcopy(dict(to_dict()))
    values = getattr(config, "__dict__", None)
    if values is not None:
        return deepcopy(dict(values))
    raise TypeError("target_config must be a mapping or config object")


def _get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    # Transformers sometimes serializes tuples as lists, which is fine.  Keep
    # None as None because it is the meaningful full-attention value here.
    return value


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def serving_stop_token_ids(target_config: Any) -> tuple[int, ...]:
    """Return native target stops, retaining MiniCPM5 EOG 1 and 130073."""
    config = _mapping(target_config)
    raw = _as_int_tuple(config.get("eos_token_id"))
    merged: list[int] = []
    for token_id in (*raw, *DEFAULT_EOG_TOKEN_IDS):
        if token_id not in merged:
            merged.append(token_id)
    return tuple(merged)


def validate_target_config(target_config: Any) -> dict[str, Any]:
    """Check fields consumed by the Qwen3 DSpark module and return a copy."""
    config = _mapping(target_config)
    required = ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads")
    missing = [name for name in required if config.get(name) is None]
    if missing:
        raise ValueError("target config missing fields: " + ", ".join(missing))
    if int(config["hidden_size"]) <= 0 or int(config["vocab_size"]) <= 0:
        raise ValueError("hidden_size and vocab_size must be positive")
    if int(config["num_attention_heads"]) <= 0:
        raise ValueError("num_attention_heads must be positive")
    kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    if kv_heads <= 0 or int(config["num_attention_heads"]) % kv_heads:
        raise ValueError("num_key_value_heads must divide num_attention_heads")
    head_dim = int(config.get("head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"])))
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return config


def build_draft_config(
    target_config: Any,
    *,
    draft_layers: int = DEFAULT_DRAFT_LAYERS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    target_layer_ids: tuple[int, ...] = DEFAULT_TARGET_LAYER_IDS,
    mask_token_id: int | None = None,
    num_anchors: int = DEFAULT_NUM_ANCHORS,
    markov_rank: int = DEFAULT_MARKOV_RANK,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimum Qwen3DSparkModel config from MiniCPM5 LlamaConfig.

    ``sliding_window`` is deliberately assigned even when the source config
    has no such attribute.  This one-field compatibility patch prevents the
    upstream attention constructor from raising AttributeError.
    """
    source = validate_target_config(target_config)
    if draft_layers <= 0 or block_size <= 0 or num_anchors <= 0:
        raise ValueError("draft_layers, block_size and num_anchors must be positive")
    if not target_layer_ids:
        raise ValueError("at least one target layer tap is required")
    target_layers = int(source["num_hidden_layers"])
    bad_taps = [tap for tap in target_layer_ids if tap < 0 or tap >= target_layers]
    if bad_taps:
        raise ValueError(f"target layer taps outside target depth: {bad_taps}")
    result = deepcopy(source)
    extra = dict(overrides or {})
    result.update(extra)

    result["architectures"] = ["Qwen3DSparkModel"]
    result["model_type"] = "qwen3_dspark"
    result["num_target_layers"] = target_layers
    result["num_hidden_layers"] = int(draft_layers)
    result["block_size"] = int(block_size)
    result["target_layer_ids"] = [int(tap) for tap in target_layer_ids]
    result["layer_types"] = ["full_attention"] * int(draft_layers)
    result["_attn_implementation"] = str(result.get("_attn_implementation", "flex_attention"))
    result["tie_word_embeddings"] = False
    result["sliding_window"] = source.get("sliding_window", None)
    selected_mask_token_id = (
        source.get("mask_token_id", DEFAULT_MASK_TOKEN_ID)
        if mask_token_id is None
        else mask_token_id
    )
    if int(selected_mask_token_id) < 0 or int(selected_mask_token_id) >= int(source["vocab_size"]):
        raise ValueError("mask_token_id must be inside target vocabulary")
    result["mask_token_id"] = int(selected_mask_token_id)
    result["num_anchors"] = int(num_anchors)
    result["enable_confidence_head"] = True
    result["confidence_head_with_markov"] = True
    result["markov_rank"] = int(markov_rank)
    result["markov_head_type"] = "vanilla"
    # These names match DeepSpec config/dspark/*.py and Qwen3DSparkTrainer.
    result["confidence_head_alpha"] = 1.0
    result["loss_decay_gamma"] = 4.0
    result["ce_loss_alpha"] = 0.1
    result["l1_loss_alpha"] = 0.9
    result["serving_stop_token_ids"] = list(serving_stop_token_ids(source))

    # MiniCPM's LlamaConfig may omit fields that Qwen3's implementation reads.
    # Supplying their documented defaults keeps the bridge explicit and makes
    # an eventual cloud run fail at config validation rather than deep in a
    # worker process.
    result.setdefault("attention_dropout", 0.0)
    result.setdefault("attention_bias", False)
    result.setdefault("rms_norm_eps", 1e-6)
    result.setdefault("pad_token_id", 1)
    return result


def config_contract(target_config: Any, draft_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact provenance/compatibility record for receipts."""
    target = validate_target_config(target_config)
    return {
        "target_model_type": target.get("model_type", "llama"),
        "target_layers": int(target["num_hidden_layers"]),
        "target_hidden_size": int(target["hidden_size"]),
        "target_vocab_size": int(target["vocab_size"]),
        "draft_architecture": draft_config.get("architectures"),
        "draft_layers": int(draft_config["num_hidden_layers"]),
        "block_size": int(draft_config["block_size"]),
        "target_layer_ids": list(draft_config["target_layer_ids"]),
        "sliding_window": draft_config.get("sliding_window"),
        "serving_stop_token_ids": list(draft_config["serving_stop_token_ids"]),
        "one_field_compatibility_patch": "sliding_window=getattr(target_config, 'sliding_window', None)",
    }


def apply_upstream_sliding_window_patch(target_config: Any) -> Any:
    """Mutate/return a Transformers config exactly at the upstream seam.

    DeepSpec's ``build_qwen3_draft_config`` returns a config object, whereas
    the pure bridge above returns JSON-compatible data.  A cloud integration
    can call this helper immediately after the upstream deepcopy and before
    constructing ``Qwen3DSparkModel``.
    """
    if not hasattr(target_config, "sliding_window"):
        setattr(target_config, "sliding_window", None)
    return target_config
