"""CPU golden checks for bounded evaluation logits, without a trainer import."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.training.poc_twin.model import chunked_eval_ce


class ChunkedEvalCETest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(41)
        self.h = torch.randn(2, 7, 5, generator=generator)
        self.w = torch.randn(19, 5, generator=generator)
        self.targets = torch.randint(19, (2, 7), generator=generator)

    def reference(self, h, targets, chunk, initial_sum=0.0):
        logits = F.linear(h, self.w).reshape(-1, self.w.size(0))
        targets = targets.reshape(-1)
        total = initial_sum
        for c in range(0, targets.numel(), chunk):
            total += F.cross_entropy(logits[c:c + chunk].float(),
                                     targets[c:c + chunk], reduction="sum").item()
        return total

    def test_matches_full_projection_for_chunk_boundaries(self):
        for chunk in (1, 4, 7, 14, 4096):
            with self.subTest(chunk=chunk):
                total, count = chunked_eval_ce(self.h, self.w, self.targets, chunk)
                self.assertEqual(count, 14)
                self.assertAlmostEqual(total, self.reference(self.h, self.targets, chunk),
                                       delta=2e-5)

    def test_projection_is_bounded_with_short_final_chunk(self):
        original_linear = F.linear
        with patch("experiments.training.poc_twin.model.F.linear",
                   wraps=original_linear) as project:
            chunked_eval_ce(self.h, self.w, self.targets, chunk=4)
        self.assertEqual([call.args[0].shape[0] for call in project.call_args_list],
                         [4, 4, 4, 2])

    def test_running_sum_and_noncontiguous_targets(self):
        targets = self.targets[:, :-1]
        h = self.h[:, :-1]
        expected = actual = 100000000.125
        for _ in range(2):
            expected = self.reference(h, targets, 4, expected)
            actual, count = chunked_eval_ce(h, self.w, targets, 4, actual)
            self.assertEqual(count, 12)
        self.assertAlmostEqual(actual, expected, delta=4e-5)

    def test_ignored_targets_keep_legacy_count(self):
        self.targets[0, 0] = -100
        total, count = chunked_eval_ce(self.h, self.w, self.targets, chunk=4)
        self.assertEqual(count, 14)
        self.assertAlmostEqual(total, self.reference(self.h, self.targets, 4), delta=2e-5)

    def test_initial_sum_preserves_sequential_python_additions(self):
        initial = float(2 ** 53)
        with patch("experiments.training.poc_twin.model.F.cross_entropy",
                   return_value=torch.tensor(1.0)):
            total, count = chunked_eval_ce(self.h, self.w, self.targets,
                                           chunk=7, initial_sum=initial)
        self.assertEqual(count, 14)
        self.assertEqual(total, (initial + 1.0) + 1.0)
        self.assertNotEqual(total, initial + (1.0 + 1.0))

    def test_no_gradients_or_input_mutation(self):
        self.h.requires_grad_()
        self.w.requires_grad_()
        before = [tensor.detach().clone() for tensor in (self.h, self.w, self.targets)]
        total, _ = chunked_eval_ce(self.h, self.w, self.targets, chunk=4)
        self.assertIsInstance(total, float)
        self.assertIsNone(self.h.grad)
        self.assertIsNone(self.w.grad)
        for actual, expected in zip((self.h, self.w, self.targets), before):
            self.assertTrue(torch.equal(actual, expected))

    def test_invalid_chunk_and_mismatched_counts(self):
        with self.assertRaisesRegex(ValueError, "chunk must be positive"):
            chunked_eval_ce(self.h, self.w, self.targets, chunk=0)
        with self.assertRaisesRegex(ValueError, "equal token counts"):
            chunked_eval_ce(self.h, self.w, self.targets[:, :-1])


if __name__ == "__main__":
    unittest.main()
