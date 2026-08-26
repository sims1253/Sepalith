#!/usr/bin/env python3
"""Self-contained tests for data_prep_pos.py (run: python3 -m pytest
experiments/training/poc_ddot/test_data_prep_pos.py -q).

Implements the failing-test list from POC-DDOT Task 1
(docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md), verbatim:
  (a) slot coordinates strictly increasing, in [0,1], anchored at the cursor
      (0 = first span token);
  (b) [EMPTY] triples get a single sentinel slot;
  (c) round-trip: values + slots reconstruct the (prefix, span, suffix)
      layout — in the landed astfim PSM anatomy the suffix region is the
      terminator line (poc_diff data_prep.py row anatomy), so the
      reconstruction is prompt_ids + slot-ordered span region;
plus fp16-exactness (the plan stores slots as a float16 field) and
integration against the real eval triples when /tmp/poc_diff is present
(the writer artifacts: jsonl + float16 slots bin aligned with poc_diff's
row_lens.bin span regions).
"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_prep_pos as DP
from experiments.training.poc_diff import EMPTY_ID


def region_of(rec):
    return list(rec["span_ids"]) if rec["span_ids"] else [EMPTY_ID]


# --- (a) slot coordinate contract --------------------------------------------

def test_slots_strictly_increasing_in_unit_interval_anchored():
    for n in (1, 2, 3, 5, 17, 256):
        slots = DP.slot_coords(n)
        assert len(slots) == n
        assert all(0.0 <= s <= 1.0 for s in slots), f"n={n}"
        if n > 1:
            diffs = [b - a for a, b in zip(slots, slots[1:])]
            assert min(diffs) > 0, f"n={n} not strictly increasing"
        assert slots[0] == 0.0, f"n={n} not anchored at the cursor"
        if n > 1:
            assert slots[-1] == 1.0, f"n={n} last slot not normalized to 1"


def test_slots_empty_region():
    assert DP.slot_coords(0) == []


# --- (b) [EMPTY] sentinel slot -------------------------------------------------

def test_empty_triple_gets_single_sentinel_slot():
    rec = dict(prompt_ids=[7, 8, 9], span_ids=[], span_len=0, kind="x")
    out = DP.augment_triple(rec)
    assert out["slots"] == [0.0]
    assert out["span_ids"] == []  # augment must not mutate the region encoding


# --- augment: values + slots pairs ----------------------------------------------

def test_augment_slots_match_region_length():
    for span in ([], [5], [5, 6], list(range(256))):
        rec = dict(prompt_ids=[1], span_ids=span, span_len=len(span))
        out = DP.augment_triple(rec)
        assert len(out["slots"]) == len(region_of(rec))
        assert list(out.keys() - rec.keys()) == ["slots"]


def test_slots_field_is_float16_exact():
    # The plan stores slots as float16; every stored value must be exactly
    # representable in fp16 (round-trips through np.float16 unchanged).
    for n in (1, 2, 7, 256):
        for s in DP.slot_coords(n):
            assert float(np.float16(s)) == s, f"n={n} slot {s} not fp16-exact"


# --- (c) round-trip ---------------------------------------------------------------

def test_round_trip_reconstructs_layout():
    rec = dict(prompt_ids=[10, 11, 12], span_ids=[55, 56, 57], span_len=3)
    out = DP.augment_triple(rec)
    layout = DP.round_trip(rec["prompt_ids"], region_of(out), out["slots"])
    assert layout == rec["prompt_ids"] + region_of(rec)


def test_round_trip_from_shuffled_pairs():
    # Positions carry the order: a (value, slot) pair set in arbitrary
    # storage order still reconstructs the original sequence.
    values = [101, 102, 103, 104, 105]
    slots = DP.slot_coords(5)
    order = [3, 0, 4, 1, 2]  # storage permutation
    shuffled_vals = [values[i] for i in order]
    shuffled_slots = [slots[i] for i in order]
    assert DP.round_trip([], shuffled_vals, shuffled_slots) == values


# --- integration: real triples + writer --------------------------------------------

def _real_eval_path():
    return Path("/tmp/poc_diff/eval_triples.jsonl")


def test_augment_real_eval_triples():
    path = _real_eval_path()
    if not path.exists():
        import pytest
        pytest.skip("/tmp/poc_diff/eval_triples.jsonl not present (tmpfs)")
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out = DP.augment_triple(rec)
            assert len(out["slots"]) == len(region_of(rec))
            assert DP.round_trip(rec["prompt_ids"], region_of(out),
                                 out["slots"]) == rec["prompt_ids"] + region_of(rec)
            n += 1
    assert n == 216  # poc_diff's held-out eval count (board 23:29)


def test_writer_jsonl_and_slots_bin():
    recs = [
        dict(prompt_ids=[1, 2], span_ids=[10, 11, 12], span_len=3),
        dict(prompt_ids=[3], span_ids=[], span_len=0),          # [EMPTY]
        dict(prompt_ids=[4, 5, 6, 7], span_ids=[20], span_len=1),
    ]
    with tempfile.TemporaryDirectory() as td:
        prefix = str(Path(td) / "smoke_")
        counts = DP.write_pos_artifacts(iter(recs), prefix)
        assert counts == (3, 5)  # rows, total slot count (3 + 1 + 1)
        lines = [json.loads(l) for l in open(prefix + "triples_pos.jsonl")]
        assert [len(r["slots"]) for r in lines] == [3, 1, 1]
        assert lines[1]["slots"] == [0.0]
        raw = np.fromfile(prefix + "slots.bin", dtype=np.float16)
        assert raw.size == 5
        assert raw.tolist() == lines[0]["slots"] + lines[1]["slots"] + lines[2]["slots"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001 - standalone runner reports all
                failures += 1
                print(f"ERROR {name}: {e!r}")
    sys.exit(1 if failures else 0)
