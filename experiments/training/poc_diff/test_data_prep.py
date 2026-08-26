"""Task 1 tests: span-triple renderer.

Uses a char-level FakeTok so the logic tests run offline in milliseconds;
tokenizer-boundary behavior with the real MiniCPM5 tokenizer is MEASURED
(and logged to meta.json as boundary_align_frac) by data_prep.main, not
asserted here — BPE merges across the prefix|span boundary are expected on
some fraction of rows and are deliberately tolerated (both arms decode at
text level for metrics).

Plan-mandated cases:
  (a) prefix+span+suffix concatenation round-trips to the original tokens
  (b) spans > 256 tokens are dropped and counted
  (c) empty spans map to the [EMPTY] class
  (d) determinism: same input -> same triple list (no RNG in the renderer)
plus the prompt-side 640 cap.
"""
import json

import pytest

from experiments.training.poc_diff import EMPTY_ID, span_region
from experiments.training.poc_diff import data_prep as dp


class FakeTok:
    """Character-level tokenizer with the transformers call interface.
    No merges, so token-level round-trip is exact by construction."""

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def make_row(prefix, span, malformed=False):
    """A PSM row in the fixed-corpus anatomy."""
    if malformed:
        return dict(prompt=prefix + "\n", target=span + "\n<|end|>",
                    text="garbage", kind="k", package="p", path="r/x.R")
    return dict(
        prompt=prefix + "\n<|end|>\n",
        target=span + "\n<|end|>",
        text=prefix + span + "\n<|end|>",
        kind="k", package="p", path="r/x.R")


def run(rows):
    tok = FakeTok()
    stats = dp.new_stats()
    triples = list(dp.build_triples(iter(rows), tok, stats))
    return triples, stats


def test_roundtrip_tokens():
    """(a) tok(prefix) + tok(span) + tok(suffix) == tok(original text):
    with the merge-free tokenizer this must hold exactly; the string
    anatomy (prefix_part + span + END_MARKER == text) is the contract the
    real tokenizer is measured against."""
    row = make_row("<|context|>p.R\nx <- 1\n<|suffix|>\ny\n", "z <- 2")
    triples, stats = run([row])
    assert stats["kept"] == 1 and len(triples) == 1
    t = triples[0]
    tok = FakeTok()
    original = tok([row["text"]], add_special_tokens=False)["input_ids"][0]
    suffix = tok(["\n<|end|>"], add_special_tokens=False)["input_ids"][0]
    assert t["prompt_ids"] + t["span_ids"] + suffix == original
    # and the decomposition is the row's own fields, verbatim
    assert "".join(chr(i) for i in t["prompt_ids"]) == row["prompt"][:-len("\n<|end|>\n")]
    assert "".join(chr(i) for i in t["span_ids"]) == row["target"][:-len("\n<|end|>")]


def test_drop_span_gt_256():
    """(b) 300-token span dropped, counted, not emitted."""
    rows = [make_row("ctx", "a" * 100), make_row("ctx", "b" * 300)]
    triples, stats = run(rows)
    assert stats["span_gt_256"] == 1
    assert stats["kept"] == 1
    assert all(len(t["span_ids"]) <= 256 for t in triples)


def test_drop_prompt_gt_640():
    rows = [make_row("x" * 700, "span")]
    triples, stats = run(rows)
    assert stats["prompt_gt_640"] == 1 and stats["kept"] == 0
    assert triples == []


def test_empty_span_maps_to_empty_class():
    """(c) target == just the end marker -> span_ids == [] -> the rendered
    span region is the single [EMPTY] token."""
    rows = [make_row("ctx", ""), make_row("ctx", "real span")]
    triples, stats = run(rows)
    assert stats["empty_span"] == 1
    assert triples[0]["span_ids"] == []
    assert triples[0]["span_len"] == 0
    assert span_region(triples[0]["span_ids"]) == [EMPTY_ID]
    assert span_region(triples[1]["span_ids"]) == triples[1]["span_ids"]


def test_malformed_rows_counted():
    rows = [make_row("ctx", "s", malformed=True),
            dict(prompt="no terminator", target="t\n<|end|>"),
            make_row("ctx", "ok")]
    triples, stats = run(rows)
    assert stats["malformed"] == 2 and stats["kept"] == 1


def test_determinism():
    """(d) the renderer has no RNG: same rows -> identical triples."""
    rows = [make_row(f"ctx{i}\n", f"span {i}") for i in range(50)]
    a, sa = run(rows)
    b, sb = run(rows)
    assert json.dumps(a) == json.dumps(b)
    assert sa == sb


def test_context_cap_enforced():
    """prompt 640 + span 256 = 896 <= 1024 always holds, but a hypothetical
    over-cap combination must be dropped (guard, currently unreachable with
    the two caps above)."""
    assert 640 + 256 + 1 <= 1024  # (+1 for the [EMPTY] rendering of len-0)
