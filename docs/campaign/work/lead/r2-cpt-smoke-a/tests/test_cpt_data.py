#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
TRAINING = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-cpt-v1/snapshots/8df4c01e161fddb4f0e3c6d3b7b0899c6683b51971253806f9455dea66ec2b1d/source/experiments/training')
sys.path.insert(0, str(TRAINING))

from campaign_cpt_data import (  # noqa: E402
    CptDataError,
    causal_lm_collator,
    causal_loss_denominator,
    canonical_token_stream_sha256,
    validate_materialized_rows,
    validate_package_holdout,
    validate_draw_schedule,
)


def row(document, package, group, sha, chunk, start, end, count, overlap, final, owned):
    ids = [0] + ([owned[0] - 1] if overlap else []) + list(owned) + [1]
    labels = [-100] + ([-100] if overlap else []) + list(owned) + ([1] if final else [-100])
    return {
        "schema": 1, "document_id": document, "package": package, "group_id": group,
        "cpt_partition": "cpt_validation" if package.startswith("val") else "cpt_train",
        "source_path": "/synthetic/raw.R", "source_sha256": sha,
        "row_id": f"{document}:{chunk}", "chunk_index": chunk,
        "input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
        "source_token_start": start, "source_token_end": end,
        "token_start": start, "token_end": end, "document_token_count": count,
        "overlap_context_tokens": overlap, "is_document_end": final,
        "supervised_tokens": sum(x != -100 for x in labels),
    }


class CptDataTests(unittest.TestCase):
    def setUp(self):
        self.sha_a = hashlib.sha256(b"doc-a").hexdigest()
        self.sha_b = hashlib.sha256(b"doc-b").hexdigest()
        self.train = [
            row("doc-a", "train-pkg", "g-train", self.sha_a, 0, 0, 3, 6, 0, False, [10, 11, 12]),
            row("doc-a", "train-pkg", "g-train", self.sha_a, 1, 3, 6, 6, 1, True, [13, 14, 15]),
        ]
        self.validation = [
            row("doc-b", "val-pkg", "g-val", self.sha_b, 0, 0, 3, 3, 0, True, [20, 21, 22]),
        ]

    def test_contiguous_masks_and_holdout(self):
        train = validate_materialized_rows(self.train, max_sequence_tokens=2048)
        val = validate_materialized_rows(self.validation, max_sequence_tokens=2048)
        split = validate_package_holdout(self.train, self.validation, max_sequence_tokens=2048, materialized=True)
        self.assertEqual(train["payload_tokens"], 6)
        self.assertEqual(train["loss_tokens"], 7)
        self.assertEqual(val["loss_tokens"], 4)
        self.assertEqual(split["train_packages"], ["train-pkg"])
        self.assertEqual(split["validation_packages"], ["val-pkg"])
        self.assertTrue(split["document_disjoint"])
        self.assertEqual(self.train[0]["labels"][-1], -100)
        self.assertEqual(self.train[1]["labels"][-1], 1)
        self.assertEqual(self.train[1]["labels"][1], -100)

    def test_collator_preserves_explicit_masks_and_padding(self):
        batch = causal_lm_collator([self.train[0], self.train[1]], max_sequence_tokens=2048)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 6))
        self.assertEqual(batch["labels"][0, -1].item(), -100)
        self.assertEqual(batch["labels"][1, -1].item(), 1)
        self.assertEqual(batch["labels"][1, 1].item(), -100)
        self.assertEqual(batch["attention_mask"][0, 4].item(), 1)
        self.assertEqual(batch["attention_mask"][0, -1].item(), 0)
        self.assertEqual(causal_loss_denominator(batch), 7)

    def test_internal_eos_target_is_rejected(self):
        bad = copy.deepcopy(self.train)
        bad[0]["labels"][-1] = 1
        with self.assertRaisesRegex(CptDataError, "EOS label"):
            validate_materialized_rows(bad)

    def test_canonical_rows_mask_overlap_and_infer_terminal_chunk(self):
        source_common = {
            "source_kind": "raw_r_document", "source_id": "canonical-doc", "group_id": "canonical-group",
            "document_sha256": self.sha_a, "token_stream_sha256": canonical_token_stream_sha256([10, 11, 12, 13]),
            "document_token_count": 4, "chunk_count": 2,
            "tokenizer_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
            "builder_id": "b", "builder_sha256": self.sha_b,
        }
        canonical = [
            {"id": "canonical:0", "split": "train", "package_id": "train-pkg", "input_ids": [0, 10, 11, 1],
             "source": {**source_common, "token_start": 0, "token_end": 2, "chunk_index": 0,
                        "overlap_context_tokens": 0}},
            {"id": "canonical:1", "split": "train", "package_id": "train-pkg", "input_ids": [0, 11, 12, 13, 1],
             "source": {**source_common, "token_start": 2, "token_end": 4, "chunk_index": 1,
                        "overlap_context_tokens": 1}},
        ]
        from campaign_cpt_data import validate_rows
        checked = validate_rows(canonical)
        self.assertEqual(checked["loss_tokens"], 5)
        collated = causal_lm_collator(canonical)
        self.assertEqual(collated["labels"][1, 1].item(), -100)
        self.assertEqual(collated["labels"][1, 2].item(), 12)
        self.assertEqual(collated["labels"][1, -1].item(), 1)

    def test_supplied_code_label_mismatch_is_rejected(self):
        bad = copy.deepcopy(self.train)
        bad[0]["labels"][2] = 999
        with self.assertRaisesRegex(CptDataError, "code labels"):
            validate_materialized_rows(bad)

    def test_gap_and_package_document_leakage_rejected(self):
        bad = copy.deepcopy(self.train)
        bad[1]["source_token_start"] = bad[1]["token_start"] = 4
        bad[1]["source_token_end"] = bad[1]["token_end"] = 7
        bad[1]["document_token_count"] = 7
        with self.assertRaisesRegex(CptDataError, "gap or overlap"):
            validate_materialized_rows(bad)
        leaked = copy.deepcopy(self.validation)
        leaked[0]["source_sha256"] = self.sha_a
        with self.assertRaisesRegex(CptDataError, "document leakage"):
            validate_package_holdout(self.train, leaked, materialized=True)

    def test_schedule_wrapper_and_token_hash(self):
        rows = [
            {"id": "a", "input_ids": [0, 10, 1], "split": "train", "package_id": "p", "source": {
                "source_kind": "raw_r_document", "source_id": "d", "group_id": "g",
                "document_sha256": self.sha_a, "token_stream_sha256": canonical_token_stream_sha256([10]),
                "document_token_count": 1, "token_start": 0, "token_end": 1, "chunk_index": 0,
                "chunk_count": 1, "tokenizer_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
                "builder_id": "b", "builder_sha256": self.sha_b,
            }}
        ]
        schedule = {"schema_version": "sepalith.dat06.sampler.v1", "schedule": {
            "split_id": "split", "max_steps": 1, "effective_batch": 1,
            "token_rows_sha256": "rows", "row_ids": ["a"], "seed": 3407, "method": "sequential",
        }}
        check = validate_draw_schedule(schedule, rows, token_rows_sha256="rows", max_steps=1, effective_batch=1)
        self.assertEqual(check["draws"], 1)
        self.assertEqual(check["row_ids"], ["a"])


if __name__ == "__main__":
    unittest.main()
