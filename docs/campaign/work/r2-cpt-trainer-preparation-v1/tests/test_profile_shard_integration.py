#!/usr/bin/env python3
"""CPU integration probe for the immutable 2K CPT profile shard."""
import hashlib
import unittest
from pathlib import Path

from datasets import Dataset

HERE = Path(__file__).resolve()
TRAINING = HERE.parents[1] / "source" / "experiments" / "training"
import sys
sys.path.insert(0, str(TRAINING))

from campaign_cpt_data import (  # noqa: E402
    causal_lm_collator,
    causal_loss_denominator,
    read_jsonl,
    validate_materialized_rows,
    validate_package_holdout,
    verify_causal_batch,
)


class FrozenProfileIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = HERE.parents[2] / "r2-corpus-preparation-v1" / "profile-shard-v2-2k"
        cls.train_path = cls.profile / "cpt_train.jsonl"
        cls.validation_path = cls.profile / "cpt_validation.jsonl"

    def test_frozen_rows_cross_the_dataset_collator_boundary(self):
        train = read_jsonl(self.train_path, max_sequence_tokens=2048, materialized=True)
        validation = read_jsonl(self.validation_path, max_sequence_tokens=2048, materialized=True)
        train_summary = validate_materialized_rows(train, max_sequence_tokens=2048)
        validation_summary = validate_materialized_rows(validation, max_sequence_tokens=2048)
        holdout = validate_package_holdout(
            train, validation, max_sequence_tokens=2048, materialized=True,
        )
        self.assertEqual(len(train), 2526)
        self.assertEqual(len(validation), 499)
        self.assertEqual(train_summary["documents"], 1055)
        self.assertEqual(validation_summary["documents"], 294)
        self.assertEqual(train_summary["loss_tokens"], 3867176)
        self.assertEqual(validation_summary["loss_tokens"], 661360)
        self.assertTrue(holdout["package_disjoint"])
        self.assertTrue(holdout["document_disjoint"])
        self.assertEqual(
            hashlib.sha256(self.train_path.read_bytes()).hexdigest(),
            "d707a61ccfabc058dc1215cd1b2345e17edf77e2797cc5fa6f6f53daebf31da9",
        )
        minimal = [
            {key: row[key] for key in ("input_ids", "labels", "attention_mask")}
            for row in train[:2]
        ]
        dataset = Dataset.from_list(minimal)
        self.assertEqual(dataset.column_names, ["input_ids", "labels", "attention_mask"])
        batch = causal_lm_collator([dataset[index] for index in range(2)], max_sequence_tokens=2048)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2048))
        self.assertEqual(causal_loss_denominator(batch), 3406)
        self.assertEqual(verify_causal_batch(batch, minimal), 3406)


if __name__ == "__main__":
    unittest.main()
