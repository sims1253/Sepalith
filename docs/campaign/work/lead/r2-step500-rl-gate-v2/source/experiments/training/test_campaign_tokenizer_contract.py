"""CPU-only tests for the pinned MiniCPM tokenizer loader contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from campaign_tokenizer_contract import (
    BOS_ID,
    EOS_ID,
    TokenizerContractError,
    restore_pinned_tokenizer_contract,
)


def _vocab() -> dict[str, int]:
    # Keep the fake at the pinned cardinality so an accidental add/reorder is
    # detected by the same checks used by the live loader.
    value = {"<s>": 0, "</s>": 1, "<unk>": 130074, "<unused_token_477>": 130559}
    for token_id in range(2, 130560):
        if token_id in {130074, 130559}:
            continue
        value[f"tok{token_id}"] = token_id
    assert len(value) == 130560
    return value


class FakeTokenizer:
    def __init__(self, *, pad_id: int, encoded: dict[str, list[int]]):
        self._vocab = _vocab()
        self._by_id = {token_id: token for token, token_id in self._vocab.items()}
        self._pad_id = pad_id
        self._encoded = encoded
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.eos_token = "</s>"

    def __len__(self):
        return 130560

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return self._by_id[token_id]

    def encode(self, text, *, add_special_tokens=False, split_special_tokens=True):
        assert add_special_tokens is False
        assert split_special_tokens is True
        return list(self._encoded[text])

    @property
    def pad_token_id(self):
        return self._pad_id

    @property
    def pad_token(self):
        return self._by_id[self._pad_id]

    @pad_token.setter
    def pad_token(self, value):
        self._pad_id = self._vocab[value]


class FakeModel:
    def __init__(self, old_pad_id: int):
        self.config = SimpleNamespace(bos_token_id=BOS_ID, eos_token_id=[1, 130073], pad_token_id=old_pad_id)
        self.generation_config = SimpleNamespace(
            bos_token_id=BOS_ID, eos_token_id=[1, 130073], pad_token_id=old_pad_id,
        )
        self._modules = [SimpleNamespace(padding_idx=old_pad_id)]

    def modules(self):
        return iter(self._modules)


def _row():
    return {"id": "selected", "prompt_text": "whole prompt\n", "input_ids": [0, 17, 18, 101], "target_start": 3}


class CampaignTokenizerContractTests(unittest.TestCase):
    def test_restores_existing_eos_without_vocab_change_and_checks_configs(self):
        encoded = {"whole prompt\n": [17, 18]}
        loaded = FakeTokenizer(pad_id=130559, encoded=encoded)
        reference = FakeTokenizer(pad_id=EOS_ID, encoded=encoded)
        model = FakeModel(old_pad_id=130559)

        audit = restore_pinned_tokenizer_contract(
            model, loaded, reference_tokenizer=reference, prompt_rows=[_row()],
        )

        self.assertEqual(audit["before"]["pad_token_id"], 130559)
        self.assertEqual(audit["after"]["pad_token_id"], EOS_ID)
        self.assertTrue(audit["vocab_mapping_unchanged"])
        self.assertEqual(audit["prompt_parity"]["checked_rows"], 1)
        self.assertEqual(model.config.pad_token_id, EOS_ID)
        self.assertEqual(model.generation_config.pad_token_id, EOS_ID)
        self.assertEqual(model._modules[0].padding_idx, EOS_ID)
        self.assertFalse(audit["added_tokens"])
        self.assertFalse(audit["weights_changed"])

    def test_rejects_stale_self_consistent_vocab_mapping(self):
        loaded = FakeTokenizer(pad_id=130559, encoded={"whole prompt\n": [17, 18]})
        reference = FakeTokenizer(pad_id=EOS_ID, encoded={"whole prompt\n": [17, 18]})
        loaded._vocab["tok2"] = 999
        with self.assertRaisesRegex(TokenizerContractError, "vocabulary mapping differs"):
            restore_pinned_tokenizer_contract(FakeModel(old_pad_id=130559), loaded, reference_tokenizer=reference)

    def test_rejects_prompt_tokenization_mismatch(self):
        loaded = FakeTokenizer(pad_id=130559, encoded={"whole prompt\n": [17, 19]})
        reference = FakeTokenizer(pad_id=EOS_ID, encoded={"whole prompt\n": [17, 18]})
        with self.assertRaisesRegex(TokenizerContractError, "loaded/reference prompt token IDs differ"):
            restore_pinned_tokenizer_contract(FakeModel(old_pad_id=130559), loaded,
                                               reference_tokenizer=reference, prompt_rows=[_row()])

    def test_rejects_non_pinned_model_eog_configuration(self):
        loaded = FakeTokenizer(pad_id=130559, encoded={"whole prompt\n": [17, 18]})
        reference = FakeTokenizer(pad_id=EOS_ID, encoded={"whole prompt\n": [17, 18]})
        model = FakeModel(old_pad_id=130559)
        model.config.eos_token_id = EOS_ID
        with self.assertRaisesRegex(TokenizerContractError, "changed pinned EOG IDs"):
            restore_pinned_tokenizer_contract(model, loaded, reference_tokenizer=reference)

    @unittest.skipUnless(
        os.environ.get("SEPALITH_RUN_PINNED_TOKENIZER_FIXTURE") == "1",
        "set SEPALITH_RUN_PINNED_TOKENIZER_FIXTURE=1 for the local artifact fixture",
    )
    def test_local_pinned_artifact_selected_prompt_fixture(self):
        """Exercise the repair against the real tokenizer files, without weights."""
        from transformers import AutoTokenizer

        model_path = Path(os.environ.get(
            "SEPALITH_PINNED_MODEL",
            "/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain",
        ))
        candidate_path = Path(os.environ.get(
            "SEPALITH_CANDIDATE_ROWS",
            "/mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-pilot-token-audit/candidate-token-rows.jsonl",
        ))
        selected_path = Path(os.environ.get(
            "SEPALITH_SELECTED_IDS",
            "/mnt/e/sepalith/campaign-20260915/data-work/PRM-07-lead-profile-selected.json",
        ))
        selected_ids = json.loads(selected_path.read_text(encoding="utf-8"))["row_ids"]
        by_id = {}
        with candidate_path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                row = item["row"] if "row" in item else item
                if row["id"] in selected_ids:
                    by_id[row["id"]] = row
        rows = [by_id[row_id] for row_id in selected_ids]
        self.assertEqual(len(rows), 6)

        reference = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, use_fast=True, trust_remote_code=False,
        )
        loaded = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, use_fast=True, trust_remote_code=False,
        )
        # This is the exact existing-token mutation seen in the archived
        # Unsloth profile log; no model or model weights are loaded here.
        loaded.pad_token = "<unused_token_477>"
        model = SimpleNamespace(
            config=SimpleNamespace(bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=130559),
            generation_config=SimpleNamespace(
                bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=130559,
            ),
        )
        audit = restore_pinned_tokenizer_contract(
            model, loaded, reference_tokenizer=reference, prompt_rows=rows,
        )
        self.assertEqual(audit["before"]["pad_token_id"], 130559)
        self.assertEqual(audit["after"]["pad_token_id"], 1)
        self.assertEqual(audit["prompt_parity"]["checked_rows"], 6)
        self.assertTrue(audit["vocab_mapping_unchanged"])
        self.assertEqual(model.config.pad_token_id, 1)
        self.assertEqual(model.generation_config.pad_token_id, 1)


if __name__ == "__main__":
    unittest.main()
