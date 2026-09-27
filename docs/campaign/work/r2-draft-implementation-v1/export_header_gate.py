"""Pre-export metadata gate for a target-bound DSpark checkpoint.

The gate runs against a small JSON sidecar emitted beside a Hugging Face
checkpoint.  It intentionally does not open the 652 MB released GGUF or any
campaign checkpoint.  A cloud job may call this gate before invoking
``convert_hf_to_gguf.py`` and before private persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from minicpm5_dspark_config import (
    DEEPSPEC_REVISION,
    TARGET_CONFIG_SHA256,
    TOKENIZER_SHA256,
)


REQUIRED_TENSORS = frozenset({"embed_tokens.weight", "fc.weight", "lm_head.weight"})
EXPECTED_TARGET_TAPS = (1, 10, 20, 30, 39)
EXPECTED_STOPS = (1, 130073)


def _as_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def validate_export_header(
    header: Mapping[str, Any],
    *,
    target_config_sha256: str = TARGET_CONFIG_SHA256,
    tokenizer_sha256: str = TOKENIZER_SHA256,
    deepspec_revision: str = DEEPSPEC_REVISION,
) -> dict[str, Any]:
    """Return gate status or raise with the first actionable mismatch."""
    required = (
        "format",
        "architecture",
        "model_type",
        "draft_vocab_size",
        "num_target_layers",
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "block_size",
        "target_layer_ids",
        "mask_token_id",
        "serving_stop_token_ids",
        "target_config_sha256",
        "tokenizer_sha256",
        "deepspec_revision",
        "tensor_names",
        "target_identity",
    )
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError("export header missing: " + ", ".join(missing))
    if str(header["format"]).lower() != "gguf":
        raise ValueError("export format must be gguf")
    if str(header["architecture"]).lower() not in {"dspark", "qwen3dsparkmodel", "dflash"}:
        raise ValueError("export architecture must identify the DSpark converter path")
    if str(header["model_type"]).lower() not in {"qwen3", "qwen3_dspark"}:
        raise ValueError("export model_type is not a supported Qwen3 DSpark config")
    expected_scalars = {
        "hidden_size": 2048,
        "vocab_size": 130560,
        "draft_vocab_size": 130560,
        "num_target_layers": 42,
        "num_hidden_layers": 5,
        "block_size": 7,
        "mask_token_id": 75982,
    }
    for key, expected in expected_scalars.items():
        if int(header[key]) != expected:
            raise ValueError(f"export {key}={header[key]!r}, expected {expected}")
    if _as_tuple(header["target_layer_ids"]) != EXPECTED_TARGET_TAPS:
        raise ValueError("export target layer taps do not match R2 geometry")
    if _as_tuple(header["serving_stop_token_ids"]) != EXPECTED_STOPS:
        raise ValueError("export serving stops must preserve native EOG [1, 130073]")
    if header["target_config_sha256"] != target_config_sha256:
        raise ValueError("export target config hash does not match bound target")
    if header["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError("export tokenizer hash does not match TRAIN rows")
    if header["deepspec_revision"] != deepspec_revision:
        raise ValueError("export DeepSpec revision is not pinned")
    tensor_names = set(str(name) for name in header["tensor_names"])
    missing_tensors = sorted(REQUIRED_TENSORS - tensor_names)
    if missing_tensors:
        raise ValueError("export missing tensors: " + ", ".join(missing_tensors))
    if not isinstance(header["target_identity"], Mapping) or not header["target_identity"].get("name"):
        raise ValueError("export target_identity.name is required")
    if header.get("authored_tail_substitution", "forbidden") != "forbidden":
        raise ValueError("authored target tails cannot be exported")
    if bool(header.get("eos_was_synthesized", False)):
        raise ValueError("synthetic EOS cannot pass export gate")
    return {
        "passed": True,
        "format": "gguf",
        "architecture": str(header["architecture"]).lower(),
        "geometry": {
            "hidden_size": 2048,
            "vocab_size": 130560,
            "draft_layers": 5,
            "block_size": 7,
            "target_layer_ids": list(EXPECTED_TARGET_TAPS),
        },
        "serving_stop_token_ids": list(EXPECTED_STOPS),
        "target_identity": dict(header["target_identity"]),
        "weights_opened": False,
    }


def synthetic_header() -> dict[str, Any]:
    return {
        "format": "gguf",
        "architecture": "dspark",
        "model_type": "qwen3",
        "draft_vocab_size": 130560,
        "num_target_layers": 42,
        "hidden_size": 2048,
        "vocab_size": 130560,
        "num_hidden_layers": 5,
        "block_size": 7,
        "target_layer_ids": list(EXPECTED_TARGET_TAPS),
        "mask_token_id": 75982,
        "serving_stop_token_ids": list(EXPECTED_STOPS),
        "target_config_sha256": TARGET_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "deepspec_revision": DEEPSPEC_REVISION,
        "tensor_names": sorted(REQUIRED_TENSORS | {"confidence_head.weight", "markov_head.weight"}),
        "target_identity": {"name": "bound-target-placeholder", "checkpoint_sha256": "record-at-launch"},
        "authored_tail_substitution": "forbidden",
        "eos_was_synthesized": False,
    }


def self_test() -> dict[str, Any]:
    passed = validate_export_header(synthetic_header())
    negative = deepcopy(synthetic_header())
    negative["serving_stop_token_ids"] = [1]
    try:
        validate_export_header(negative)
    except ValueError as exc:
        negative_result = {"rejected": True, "reason": str(exc)}
    else:  # pragma: no cover - protects the gate if validation regresses
        raise AssertionError("negative export gate case was accepted")
    return {"positive": passed, "negative": negative_result}


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), sort_keys=True))
