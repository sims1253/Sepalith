#!/usr/bin/env python3
"""Synthetic tests for the DAT-10 repaired binding and cap gates."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dat10_expansion_under_test", HERE / "audit_expansion.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepairedBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = {
            "train-cpt": {"group_id": "train-cpt", "split": "train_group"},
            "train-task": {"group_id": "train-task", "split": "train_group"},
            "validation": {"group_id": "validation", "split": "train_group"},
            "dev": {"group_id": "dev", "split": "dev_group"},
            "final": {"group_id": "final", "split": "final_candidate_group"},
            "quarantine": {"group_id": "quarantine", "split": "quarantine_sft_v7"},
        }
        self.partition = {"train-cpt": "cpt_train", "validation": "cpt_validation"}

    def test_cpt_partition_is_not_a_task_train_prerequisite(self) -> None:
        self.assertEqual(
            MODULE.classify_group("train-cpt", self.groups, self.partition),
            "task_train_cpt_train",
        )
        self.assertEqual(
            MODULE.classify_group("train-task", self.groups, self.partition),
            "task_train_unmapped_cpt",
        )

    def test_validation_and_global_heldout_guards_remain_closed(self) -> None:
        self.assertEqual(
            MODULE.classify_group("validation", self.groups, self.partition),
            "cpt_validation_reserved",
        )
        self.assertEqual(
            MODULE.classify_group("dev", self.groups, self.partition),
            "global_dev_group_rejected",
        )
        self.assertEqual(
            MODULE.classify_group("final", self.groups, self.partition),
            "global_final_candidate_group_rejected",
        )
        self.assertEqual(
            MODULE.classify_group("quarantine", self.groups, self.partition),
            "global_nontrain_quarantine_sft_v7",
        )
        self.assertEqual(
            MODULE.classify_group("missing", self.groups, self.partition),
            "unknown_dat02_group",
        )

    def test_old_reason_is_binding_first(self) -> None:
        self.assertEqual(
            MODULE.old_binding_reason("task_train_unmapped_cpt", 700, 192),
            "group_not_in_cpt_partition",
        )
        self.assertEqual(
            MODULE.old_binding_reason("task_train_cpt_train", 193, 192),
            "target_over_192_including_EOS",
        )
        self.assertEqual(
            MODULE.old_binding_reason("cpt_validation_reserved", 20, 192),
            "cpt_validation_reserved",
        )


class GeometryTests(unittest.TestCase):
    def test_complete_boundary_and_no_truncation(self) -> None:
        row = {
            "id": "synthetic",
            "input_ids": [0, 22, 23, 24, 25, 1],
            "target_start": 3,
            "prompt_token_count": 2,
            "target_token_count": 2,
            "target_body_tokens": [24],
            "target_terminal_tokens": [25],
        }
        status, metrics = MODULE.row_contract_status(row)
        self.assertEqual(status, "ok")
        self.assertEqual(metrics, {"target_labels": 3, "total_tokens": 6})
        self.assertEqual(MODULE.cap_status(metrics), "within_current_task_cap")
        self.assertEqual(
            MODULE.cap_status({"target_labels": 193, "total_tokens": 200}),
            "target_over_192_including_EOS",
        )
        self.assertEqual(
            MODULE.cap_status({"target_labels": 30, "total_tokens": 4097}),
            "total_over_4096",
        )

    def test_boundary_corruption_is_rejected(self) -> None:
        row = {
            "id": "broken",
            "input_ids": [0, 22, 23, 24, 26, 1],
            "target_start": 3,
            "prompt_token_count": 2,
            "target_token_count": 2,
            "target_body_tokens": [24],
            "target_terminal_tokens": [25],
        }
        status, _metrics = MODULE.row_contract_status(row)
        self.assertEqual(status, "row_contract_boundary")


if __name__ == "__main__":
    unittest.main()
