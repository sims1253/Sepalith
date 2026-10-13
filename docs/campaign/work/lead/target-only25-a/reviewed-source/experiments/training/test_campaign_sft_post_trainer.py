"""CPU checks for the strict post-SFTTrainer tokenizer contract."""

from __future__ import annotations

import unittest

import campaign_sft as sft
from test_campaign_tokenizer_contract import FakeModel, FakeTokenizer, _row


class CampaignSFTPostTrainerTests(unittest.TestCase):
    def _objects(self):
        encoded = {"whole prompt\n": [17, 18]}
        loaded = FakeTokenizer(pad_id=1, encoded=encoded)
        reference = FakeTokenizer(pad_id=1, encoded=encoded)
        return FakeModel(old_pad_id=1), loaded, reference

    def test_post_trainer_identity_and_prompt_parity_pass(self):
        model, loaded, reference = self._objects()
        audit = sft.assert_post_trainer_pinned_identity(model, loaded, reference, [_row()])
        self.assertEqual(audit["status"], "verified")
        self.assertEqual(audit["prompt_parity"]["checked_rows"], 1)
        self.assertTrue(audit["vocab_mapping_equal_reference"])

    def test_post_trainer_pad_mutation_fails_before_training(self):
        model, loaded, reference = self._objects()
        loaded.pad_token = "<unused_token_477>"
        with self.assertRaisesRegex(ValueError, "changed tokenizer identity"):
            sft.assert_post_trainer_pinned_identity(model, loaded, reference, [_row()])

    def test_post_trainer_config_mutation_fails_before_training(self):
        model, loaded, reference = self._objects()
        model.generation_config.pad_token_id = 130559
        with self.assertRaisesRegex(ValueError, "changed generation token identity"):
            sft.assert_post_trainer_pinned_identity(model, loaded, reference, [_row()])

    def test_actual_HF_train_alignment_restores_native_EOG(self):
        from transformers.trainer_utils import align_special_tokens
        model, loaded, reference = self._objects()
        align_special_tokens(model, loaded)
        self.assertEqual(model.config.eos_token_id, 1)
        self.assertEqual(model.generation_config.eos_token_id, [1, 1, 130073])
        audit = sft.restore_trainer_eog_alignment(model, loaded, reference)
        self.assertEqual(model.config.eos_token_id, [1, 130073])
        self.assertEqual(model.generation_config.eos_token_id, [1, 130073])
        self.assertEqual(audit["verified"]["status"], "verified")

    def test_unknown_train_alignment_fails_closed(self):
        model, loaded, reference = self._objects()
        model.generation_config.eos_token_id = [1, 77]
        with self.assertRaisesRegex(ValueError, "Unexpected Trainer EOS"):
            sft.restore_trainer_eog_alignment(model, loaded, reference)


if __name__ == "__main__":
    unittest.main()
