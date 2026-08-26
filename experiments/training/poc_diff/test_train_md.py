"""Task 4 data-path tests: TripleData determinism + micro-batching, using
synthetic flat bins (no GPU, no real corpus)."""
import numpy as np
import pytest

from experiments.training.poc_diff.train_md import TripleData, micro_batches


class FakeData(TripleData):
    """TripleData over synthetic bins, bypassing file loading."""

    def __init__(self, prompt_lens, region_lens, seed=1273):
        total = [p + r for p, r in zip(prompt_lens, region_lens)]
        self.ids = np.arange(sum(total), dtype=np.int32)
        self.prompt_len = np.array(prompt_lens, dtype=np.int32)
        self.region_len = np.array(region_lens, dtype=np.int32)
        self.total_len = np.array(total, dtype=np.int32)
        self.offsets = np.concatenate([[0], np.cumsum(self.total_len)])
        self.n_rows = len(total)
        self.tokens_per_epoch = int(self.total_len.sum())
        self.seed = seed
        self._order_cache = {}


def test_step_rows_deterministic_and_sufficient():
    d = FakeData([10, 20, 15, 30, 25] * 40, [2, 5, 1, 8, 3] * 40)
    a, acc_a = d.rows_for_step(3, 500)
    b, acc_b = d.rows_for_step(3, 500)
    assert a == b, "same step must give the identical row list"
    assert acc_a >= 500 and acc_b >= 500


def test_step0_covers_from_epoch_start():
    d = FakeData([10, 20, 15, 30, 25] * 40, [2, 5, 1, 8, 3] * 40)
    rows, acc = d.rows_for_step(0, 400)
    order0 = d._order(0)
    assert rows[0] == int(order0[0])
    assert acc >= 400


def test_epoch_wraparound():
    """A step larger than one epoch wraps into the next permutation."""
    d = FakeData([10] * 20, [1] * 20)  # 220 tokens/epoch
    rows, acc = d.rows_for_step(1, 200)  # pos=200 -> near epoch end
    assert acc >= 200
    assert d.epoch_float(1, 200) > 0.9
    rows2, acc2 = d.rows_for_step(2, 200)  # pos=400 -> epoch 1
    assert acc2 >= 200


def test_micro_batches_respect_pad_budget():
    lens = [100, 900, 300, 700, 50]  # 900 forces a small micro
    d = FakeData(lens, [1] * 5)
    micros = micro_batches(list(range(5)), d)
    for rows, pad_len in micros:
        bucket = ((max(d.total_len[r] for r in rows) + 127) // 128) * 128
        assert pad_len == bucket
        assert len(rows) * pad_len <= 16384 or len(rows) == 1
    flat = [r for rows, _ in micros for r in rows]
    assert flat == list(range(5)), "micro-batching must preserve row order"
