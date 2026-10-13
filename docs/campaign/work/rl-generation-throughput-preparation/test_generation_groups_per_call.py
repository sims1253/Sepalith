"""Meaningful CPU-only tests for the RL-03 generation packing proposal."""
from __future__ import annotations

import unittest

from generation_groups_per_call_proposal import (
    GenerationGroupingError,
    flatten_call,
    materialize_batched_outputs,
    plan_generation_calls,
    validate_recipe_binding,
)


def _rows(group_count: int = 8, candidate_count: int = 4) -> list[list[int]]:
    return [
        [0, 1000 + group_index]
        for group_index in range(group_count)
        for _row_index in range(candidate_count)
    ]


class GenerationGroupsPerCallTests(unittest.TestCase):
    def test_g4_call_arms_preserve_exact_logical_order_and_counts(self):
        rows = _rows()
        for groups_per_call, expected_calls in ((1, 8), (2, 4), (4, 2), (8, 1)):
            with self.subTest(groups_per_call=groups_per_call):
                plan = plan_generation_calls(
                    rows, candidate_count=4, groups_per_call=groups_per_call,
                )
                self.assertEqual(plan["row_count"], 32)
                self.assertEqual(plan["call_count"], expected_calls)
                flattened = tuple(row for call in plan["calls"] for row in flatten_call(call))
                self.assertEqual(flattened, tuple(tuple(row) for row in rows))
                self.assertTrue(all(
                    len(group) == 4 and len({row for row in group}) == 1
                    for call in plan["calls"] for group in call
                ))

    def test_mocked_batched_outputs_keep_group_and_row_identity(self):
        plan = plan_generation_calls(_rows(), candidate_count=4, groups_per_call=2)
        fake_outputs = [
            [7000 + row_index, 1]
            for row_index in range(plan["row_count"])
        ]
        records = materialize_batched_outputs(plan, fake_outputs)
        self.assertEqual(len(records), 32)
        self.assertEqual(
            [(record["group_index"], record["group_row_index"], record["call_index"])
             for record in records[:8]],
            [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0),
             (1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 3, 0)],
        )
        self.assertEqual(records[8]["group_index"], 2)
        self.assertEqual(records[8]["call_index"], 1)
        self.assertEqual(
            [record["prompt_ids"] for record in records],
            [tuple(row) for row in _rows()],
        )
        self.assertEqual(
            [record["generated_ids"] for record in records],
            [tuple(row) for row in fake_outputs],
        )

    def test_resume_at_logical_group_boundary_has_exact_suffix_for_every_arm(self):
        rows = _rows()
        full_rows = tuple(tuple(row) for row in rows)
        for groups_per_call in (1, 2, 4, 8):
            with self.subTest(groups_per_call=groups_per_call):
                full = plan_generation_calls(
                    rows, candidate_count=4, groups_per_call=groups_per_call,
                )
                resumed = plan_generation_calls(
                    rows[4 * 4:], candidate_count=4, groups_per_call=groups_per_call,
                )
                full_suffix = tuple(
                    row for call in full["calls"][4 // groups_per_call:]
                    for row in flatten_call(call)
                )
                # If the cursor is inside a packed call, the real sampler
                # resumes at the source group and starts a fresh call.  This
                # is the exact suffix that must be regenerated.
                if 4 % groups_per_call:
                    full_suffix = tuple(
                        row for call in full["calls"]
                        for row in flatten_call(call)
                        if row in full_rows[16:]
                    )
                resumed_rows = tuple(
                    row for call in resumed["calls"] for row in flatten_call(call)
                )
                self.assertEqual(resumed_rows, full_rows[16:])
                self.assertEqual(resumed_rows, full_suffix)

    def test_recipe_binding_requires_finite_equal_top_level_and_policy_values(self):
        base = {
            "generation_groups_per_call": 2,
            "identity": {"policy": {
                "candidate_count": 4,
                "rollout_rows_per_update": 32,
                "generation_groups_per_call": 2,
            }},
        }
        self.assertEqual(validate_recipe_binding(base), 2)
        for mutation in (
            {"generation_groups_per_call": None},
            {"generation_groups_per_call": 3},
            {"generation_groups_per_call": 9},
        ):
            with self.subTest(mutation=mutation):
                bad = dict(base)
                bad.update(mutation)
                with self.assertRaises(GenerationGroupingError):
                    validate_recipe_binding(bad)
        bad_policy = {
            **base,
            "identity": {"policy": {
                "candidate_count": 4,
                "rollout_rows_per_update": 32,
                "generation_groups_per_call": 4,
            }},
        }
        with self.assertRaisesRegex(GenerationGroupingError, "differ"):
            validate_recipe_binding(bad_policy)

    def test_mixed_prompt_group_and_wrong_model_row_count_fail_closed(self):
        rows = _rows()
        rows[1] = [0, 9999]
        with self.assertRaises(GenerationGroupingError):
            plan_generation_calls(rows, candidate_count=4, groups_per_call=2)
        plan = plan_generation_calls(_rows(), candidate_count=4, groups_per_call=2)
        with self.assertRaises(GenerationGroupingError):
            materialize_batched_outputs(plan, [[1]] * 31)


if __name__ == "__main__":
    unittest.main()

