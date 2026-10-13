"""Narrow repair for the Transformers/Unsloth train-start EOS rewrite.

Copied from the reviewed CPT/SFT runner's ``restore_trainer_eog_alignment``
contract.  It accepts only the pinned native form or the one observed Trainer
alignment, restores the native stop set, and then re-runs the full tokenizer
contract.  It never adds tokens or resizes embeddings.
"""
from __future__ import annotations

from campaign_tokenizer_contract import (
    NATIVE_EOG_IDS,
    restore_pinned_tokenizer_contract,
)

_KNOWN_UNUSED_ID = 130559
_KNOWN_UNUSED_CONTENT = "<unused_token_477>"
_PINNED_UNUSED_STATE = {
    "content": _KNOWN_UNUSED_CONTENT, "single_word": False, "lstrip": False,
    "rstrip": False, "normalized": True, "special": False,
}
_TRAINER_UNUSED_STATE = {**_PINNED_UNUSED_STATE, "normalized": False, "special": True}


def _config_id_list(value):
    if type(value) is int:
        return (value,)
    if isinstance(value, (list, tuple)) and all(type(item) is int for item in value):
        return tuple(value)
    return None


def _added_token_states(tokenizer):
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not isinstance(decoder, dict):
        raise ValueError("Tokenizer has no inspectable added-token decoder")
    result = {}
    for token_id, token in decoder.items():
        state = token.__getstate__() if hasattr(token, "__getstate__") else None
        if not isinstance(state, dict):
            raise ValueError("Tokenizer added-token metadata is not inspectable")
        result[int(token_id)] = dict(state)
    return result


def assert_added_token_metadata(tokenizer, reference_tokenizer):
    loaded, reference = _added_token_states(tokenizer), _added_token_states(reference_tokenizer)
    if loaded != reference:
        raise ValueError("Tokenizer added-token metadata differs from pinned reference")
    return {"entries": len(loaded), "exact_reference": True}


def restore_known_trainer_added_token_mutation(tokenizer, reference_tokenizer):
    """Repair only Unsloth's observed ID-130559 flag rewrite."""
    before, reference = _added_token_states(tokenizer), _added_token_states(reference_tokenizer)
    mismatches = {key for key in set(before) | set(reference) if before.get(key) != reference.get(key)}
    if not mismatches:
        return {"changed": False, "mismatched_ids": []}
    if mismatches != {_KNOWN_UNUSED_ID}:
        raise ValueError("Unexpected tokenizer added-token mutation; refusing repair")
    if before[_KNOWN_UNUSED_ID] != _TRAINER_UNUSED_STATE or reference[_KNOWN_UNUSED_ID] != _PINNED_UNUSED_STATE:
        raise ValueError("Unexpected ID-130559 metadata; refusing repair")
    add_tokens = getattr(tokenizer, "add_tokens", None)
    if not callable(add_tokens):
        raise ValueError("Tokenizer cannot restore existing added-token metadata")
    vocab_before, size_before = tokenizer.get_vocab(), len(tokenizer)
    add_tokens([reference_tokenizer.added_tokens_decoder[_KNOWN_UNUSED_ID]], special_tokens=False)
    if len(tokenizer) != size_before or tokenizer.get_vocab() != vocab_before:
        raise ValueError("Added-token metadata repair changed tokenizer vocabulary")
    assert_added_token_metadata(tokenizer, reference_tokenizer)
    return {"changed": True, "mismatched_ids": [_KNOWN_UNUSED_ID], "restored": _PINNED_UNUSED_STATE}


def restore_trainer_eog_alignment(model, tokenizer, reference_tokenizer):
    """Undo only the pinned Trainer canonical-EOS alignment.

    Transformers 5.5 can turn model EOS ``[1, 130073]`` into ``1`` and
    prepend ``1`` to generation EOS.  Any other EOS rewrite is rejected.
    """
    model_ids = _config_id_list(model.config.eos_token_id)
    generation_ids = _config_id_list(model.generation_config.eos_token_id)
    native = tuple(NATIVE_EOG_IDS)
    if model_ids not in (native, (1,)) or generation_ids not in (native, (1, *native)):
        raise ValueError("Unexpected Trainer EOS alignment; refusing contract repair")
    before = {"model_eos": list(model_ids), "generation_eos": list(generation_ids)}
    backend_repair = restore_known_trainer_added_token_mutation(tokenizer, reference_tokenizer)
    model.config.eos_token_id = list(native)
    model.generation_config.eos_token_id = list(native)
    repair = restore_pinned_tokenizer_contract(
        model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=[],
    )
    return {"before": before, "backend_repair": backend_repair, "repair": repair}

def embedding_identity(model):
    result = {}
    for label, getter_name in (("input", "get_input_embeddings"), ("output", "get_output_embeddings")):
        getter = getattr(model, getter_name, None)
        module = getter() if callable(getter) else None
        weight = getattr(module, "weight", None)
        if weight is None:
            raise ValueError(f"{label} embedding weight missing")
        result[label] = {
            "object_id": id(weight), "data_ptr": int(weight.data_ptr()),
            "shape": list(weight.shape), "numel": int(weight.numel()), "dtype": str(weight.dtype),
        }
    return result


def assert_runtime_tokenizer(model, tokenizer, reference, expected_embeddings, stage):
    if len(tokenizer) != len(reference) or len(tokenizer) != 130560:
        raise ValueError(f"{stage}: tokenizer length differs")
    if tokenizer.get_vocab() != reference.get_vocab():
        raise ValueError(f"{stage}: tokenizer vocabulary differs")
    assert_added_token_metadata(tokenizer, reference)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (0, 1, 1):
        raise ValueError(f"{stage}: BOS/EOS/PAD differs")
    if tokenizer.convert_ids_to_tokens(1) != "</s>" or str(tokenizer.pad_token) != "</s>":
        raise ValueError(f"{stage}: EOS/PAD text differs")
    for label, config in (("model", model.config), ("generation", model.generation_config)):
        eos = _config_id_list(getattr(config, "eos_token_id", None))
        if getattr(config, "bos_token_id", None) != 0 or eos != tuple(NATIVE_EOG_IDS) or getattr(config, "pad_token_id", None) != 1:
            raise ValueError(f"{stage}: {label} special-token config differs")
    observed = embedding_identity(model)
    if observed != expected_embeddings:
        raise ValueError(f"{stage}: embedding objects/storage/shape/dtype changed")
    return {"stage": stage, "bos": 0, "eos": 1, "pad": 1, "vocab_entries": 130560,
            "embeddings": {label: {key: value[key] for key in ("shape", "numel", "dtype")} for label, value in observed.items()}}


def assert_serialized_tokenizer(checkpoint, reference, expected_tokenizer_sha256, sha256_file):
    import json
    from pathlib import Path
    from campaign_tokenizer_contract import load_pinned_reference_tokenizer

    checkpoint = Path(checkpoint)
    raw = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    if (raw.get("bos_token"), raw.get("eos_token"), raw.get("pad_token")) != ("<s>", "</s>", "</s>"):
        raise ValueError("raw serialized tokenizer specials differ")
    saved = load_pinned_reference_tokenizer(checkpoint)
    if saved.get_vocab() != reference.get_vocab() or sha256_file(checkpoint / "tokenizer.json") != expected_tokenizer_sha256:
        raise ValueError("serialized tokenizer identity differs")
    configs = {}
    for name in ("config.json", "generation_config.json"):
        value = json.loads((checkpoint / name).read_text(encoding="utf-8"))
        eos = _config_id_list(value.get("eos_token_id"))
        if value.get("bos_token_id") != 0 or eos != tuple(NATIVE_EOG_IDS) or value.get("pad_token_id") != 1:
            raise ValueError(f"serialized {name} token identity differs")
        configs[name] = {"bos": 0, "eog": list(eos), "pad": 1}
    return {"tokenizer_json_sha256": expected_tokenizer_sha256, "configs": configs}
