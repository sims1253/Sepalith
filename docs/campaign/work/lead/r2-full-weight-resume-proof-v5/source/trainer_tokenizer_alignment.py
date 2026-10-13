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


def _config_id_list(value):
    if type(value) is int:
        return (value,)
    if isinstance(value, (list, tuple)) and all(type(item) is int for item in value):
        return tuple(value)
    return None


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
    model.config.eos_token_id = list(native)
    model.generation_config.eos_token_id = list(native)
    repair = restore_pinned_tokenizer_contract(
        model, tokenizer, reference_tokenizer=reference_tokenizer, prompt_rows=[],
    )
    return {"before": before, "repair": repair}
