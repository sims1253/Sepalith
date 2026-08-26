#!/usr/bin/env python3
"""Task 1: position-augmented span representation — POC-DDOT.

Wraps poc_diff's Task-1 renderer (experiments/training/poc_diff/data_prep.py)
and adds the slot-coordinate field the OT coupling consumes, per
docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md (Task 1, frozen
decisions):

  - slot = normalized token index within the span REGION (the values the
    model sees: the span's own tokens, or the single [EMPTY] class token
    for no-op rows — span_region() in poc_diff/__init__.py). Slots are
    strictly increasing, in [0,1], anchored at the cursor: slot 0.0 is the
    first span token, and for L >= 2 the last token normalizes to 1.0
    (slot_i = i / (L-1), so endpoint tokens of different-length spans share
    coordinates — the property DDOT's position coupling grows/shrinks on);
  - L == 1 (incl. [EMPTY]) gets the single sentinel slot [0.0];
  - slots are stored as float16 (json numbers quantized to fp16; and a
    float16 .bin stream aligned with poc_diff's flat_ids.bin span regions —
    row_lens.bin applies unchanged, since slot count == span_region_len).

Plan-vs-reality note (same spirit as pocdiff's row-native deviation): the
DDOT plan wrote the layout as (prefix, span, suffix); the landed astfim
PSM anatomy has no token-level suffix — text = prefix + span + "\\n<|end|>"
with the terminator as the prompt/target boundary. round_trip() therefore
reconstructs prompt_ids + slot-ordered span region; the suffix collapses
to the terminator line. Documented, not enforced.

Usage (from repo root):
  python3 -m experiments.training.poc_ddot.data_prep_pos            # both
  python3 -m experiments.training.poc_ddot.data_prep_pos --split eval
Reads /tmp/poc_diff/{train,eval}_triples.jsonl (pocdiff's tmpfs staging),
writes /tmp/poc_ddot/{split}_triples_pos.jsonl (+ train_slots.bin).
"""
import argparse
import json
import pathlib
import sys
from array import array

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.poc_diff import span_region  # noqa: E402

SRC_TMP = "/tmp/poc_diff"
OUT_TMP = "/tmp/poc_ddot"


def slot_coords(n: int) -> list:
    """fp16-quantized slot coordinates for an n-token span region.

    slot_i = i / max(n - 1, 1): strictly increasing for n > 1 (adjacent
    gap 1/(n-1) >= 1/255 >> fp16 spacing), anchored at 0.0, last token 1.0
    for n >= 2; n == 1 gets the single sentinel slot [0.0]; n == 0 -> [].
    """
    if n <= 0:
        return []
    return (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(
        np.float16).tolist()


def augment_triple(rec: dict) -> dict:
    """One poc_diff triple + the `slots` field (nothing else touched)."""
    out = dict(rec)
    out["slots"] = slot_coords(len(span_region(rec["span_ids"])))
    return out


def round_trip(prompt_ids, values, slots) -> list:
    """Reconstruct the row layout from (value, slot) pairs.

    Sorting the values by slot recovers the span's token order; prepending
    the prompt gives the layout the row encodes (prefix + span; the PSM
    terminator is a text-level boundary, not part of either ids field).
    """
    order = np.argsort(np.asarray(slots, dtype=np.float32), kind="stable")
    return list(prompt_ids) + [values[i] for i in order]


def write_pos_artifacts(triples_iter, out_prefix: str):
    """Augmented jsonl + float16 slots bin, slot stream aligned 1:1 with
    poc_diff's flat_ids.bin span regions (same row order, same iterator).

    Returns (n_rows, n_slots). Incremental — never buffers the stream.
    """
    slots = array("h")  # float16 bit pattern, int16 storage
    n = 0
    with open(out_prefix + "triples_pos.jsonl", "w", encoding="utf-8") as g:
        for rec in triples_iter:
            out = augment_triple(rec)
            g.write(json.dumps(out) + "\n")
            slots.extend(np.asarray(out["slots"], dtype=np.float16).view(np.int16))
            n += 1
    with open(out_prefix + "slots.bin", "wb") as f:
        slots.tofile(f)
    return n, len(slots)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "eval", "both"], default="both")
    args = ap.parse_args()

    splits = ["eval", "train"] if args.split == "both" else [args.split]
    for split in splits:
        src = f"{SRC_TMP}/{split}_triples.jsonl"
        path = pathlib.Path(src)
        if not path.exists():
            print(f"SKIP {split}: {src} missing (run poc_diff data_prep first)")
            continue
        pathlib.Path(OUT_TMP).mkdir(parents=True, exist_ok=True)

        def rows():
            with open(src, encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)

        n, n_slots = write_pos_artifacts(rows(), f"{OUT_TMP}/{split}_")
        print(f"{split}: {n} triples, {n_slots} slots -> {OUT_TMP}/{split}_"
              f"triples_pos.jsonl + {split}_slots.bin")


if __name__ == "__main__":
    main()
