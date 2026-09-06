"""CPU regressions for masked losses crossing vocabulary-projection chunks."""
import unittest

import torch
import torch.nn.functional as F

from experiments.training.poc_diff.objective import T_MIN, mdlm_loss


class ObjectiveChunksTest(unittest.TestCase):
    def test_loss_and_gradients_match_dense_reference_across_chunks(self):
        g = torch.Generator().manual_seed(31)
        features = torch.randn(3, 5, 4, generator=g)
        weights = torch.randn(7, 4, generator=g)
        targets = torch.randint(7, (3, 5), generator=g)
        # Unequal masked counts, plus a zero-contribution example in the mean.
        mask = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        rates = torch.tensor([0.7, 0.3, 0.9])
        spans = torch.tensor([4, 3, 2])
        hr = features.clone().requires_grad_()
        wr = weights.clone().requires_grad_()
        per_token = F.cross_entropy(F.linear(hr, wr).reshape(-1, 7),
                                    targets.reshape(-1), reduction="none").reshape(3, 5)
        ref = ((per_token * mask).sum(1) / rates.clamp(min=T_MIN) / spans).mean()
        ref.backward()
        # Equal-sized chunks, ragged final chunks, a single chunk, oversized chunk.
        for chunk in (1, 2, 3, 4, 5, 4096):
            with self.subTest(chunk=chunk):
                h = features.clone().requires_grad_()
                w = weights.clone().requires_grad_()
                got = mdlm_loss(h, targets, mask, rates, spans, w, chunk=chunk)
                got.backward()
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(h.grad, hr.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(w.grad, wr.grad, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
