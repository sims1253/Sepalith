#!/usr/bin/env python3
"""A2-prime 32K R-vocab tokenizer (the spec's tied-vocab tokenizer).

Byte-level BPE, 32,768 entries incl. the six PSM specials, trained on a
mixture sample that mirrors the A2 draw: normalized R code (astfim_v1
corpus text) + SO R answers (stack v2 shard text) at roughly the R-draw's
own mix (code ~70% / prose 30%). The tokenizer must see the PSM marker
strings so they survive as specials; everything else is learned.

Output: /mnt/h/sepalith/datasets/a2_tokenizer_v1/ (tokenizer.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

OUT = "/mnt/h/sepalith/datasets/a2_tokenizer_v1/tokenizer.json"
ASTFIM = "/tmp/poc_twin/train.jsonl"                       # local R corpus
STACK = "/mnt/h/sepalith/stack_staging/files/shard-00000.jsonl"
SPECIALS = ["<|context|>", "<|history|>", "<|cursor|>", "<|suffix|>",
            "<|end|>", "<|endoftext|>"]
VOCAB = 32768


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-mb", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rng = random.Random(20260824)

    texts, nbytes = [], 0
    budget = args.target_mb * 1_000_000
    # ~70% R code from the astfim corpus (PSM text as-is: markers included
    # so their subwords are seen; they are ALSO forced specials)
    with open(ASTFIM) as fh:
        lines = fh.readlines()
    rng.shuffle(lines)
    for line in lines:
        if nbytes > budget * 0.7:
            break
        t = json.loads(line)["text"]
        texts.append(t)
        nbytes += len(t.encode("utf-8", "replace"))
    # ~30% SO R answers (code+prose) from the stack v2 shard
    with open(STACK, errors="replace") as fh:
        for line in fh:
            if nbytes > budget:
                break
            try:
                o = json.loads(line)
            except ValueError:
                continue
            t = (o.get("content") or o.get("text") or "")[:20000]
            texts.append(t)
            nbytes += len(t.encode("utf-8", "replace"))
    print(f"train sample: {len(texts)} docs, {nbytes/1e6:.0f} MB", flush=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=SPECIALS, show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    tok.save(OUT, pretty=True)
    n = tok.get_vocab_size()
    # round-trip + specials sanity
    enc = tok.encode("x <- mean(df$col, na.rm = TRUE)<|end|>")
    assert "<|end|>" in enc.tokens, enc.tokens[-5:]
    dec = tok.decode(enc.ids)
    assert dec.endswith("<|end|>")
    print(json.dumps(dict(vocab=n, out=OUT,
                          sample_tokens=enc.tokens[:12])), flush=True)


if __name__ == "__main__":
    main()
