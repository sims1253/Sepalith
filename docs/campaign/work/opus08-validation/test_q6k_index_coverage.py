#!/usr/bin/env python3
"""Focused CPU tests for q6k_index_coverage.py."""

import unittest

from q6k_index_coverage import (
    BLOCK_BYTES,
    QK_K,
    canonical_terms,
    dispatch_plan,
    make_block,
    shader_row_terms,
    validate_width,
)


class Q6KIndexCoverageTests(unittest.TestCase):
    def test_target_widths_match_canonical_dequant(self):
        for width in (2048, 6144):
            result = validate_width(width)
            self.assertTrue(result["exact_scalar_coverage"])
            self.assertTrue(result["exact_storage_coverage"])
            self.assertEqual(result["terms"], width)

    def test_partial_workgroup_is_safe_and_explicitly_reference_only(self):
        width = 3 * QK_K
        self.assertEqual(dispatch_plan(width)["mode"], "generic_q6_partial_workgroup_reference")
        self.assertEqual(len(shader_row_terms([make_block(i) for i in range(3)], width)[0]), width)

    def test_unaligned_and_wrong_subgroup_fall_back(self):
        self.assertEqual(dispatch_plan(2050)["mode"], "fallback")
        self.assertEqual(dispatch_plan(2048, subgroup=32)["mode"], "fallback")
        self.assertEqual(dispatch_plan(2048, workgroup=32)["mode"], "fallback")

    def test_block_size_and_canonical_scalar_coverage(self):
        block = make_block(19)
        self.assertEqual(len(block), BLOCK_BYTES)
        terms = canonical_terms(block)
        self.assertEqual(len(terms), QK_K)
        self.assertEqual(len({(q, scale, index) for index, (q, scale, _value) in enumerate(terms)}), QK_K)

    def test_fault_injection_is_rejected(self):
        blocks = [make_block(17 + i) for i in range(2048 // QK_K)]
        expected = []
        for block in blocks:
            expected.extend(canonical_terms(block))
        observed, _ = shader_row_terms(blocks, 2048, inject_fault=True)
        self.assertNotEqual(observed, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
