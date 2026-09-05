"""
Masked-loss companion stream for the FIM-Replica arm (2026-09-05).

Builds /tmp/poc_twin/train_blocks_mask.npy — a uint8 loss-mask array whose
blocks are INDEX-ALIGNED with data_prep.py's train_blocks.npy (same
to_blocks packing math over the same token stream, rebuilt
deterministically from the same source jsonl):

  mask=1  on the trailing target tokens of each FIM doc (span + <|end|>,
          exactly design-A2 §5.2's "loss on span + <|end|> only")
  mask=0  on prompt/context tokens and on the doc-separator eos

The astfim text field is the PSM render and the target is its SUFFIX, so
the mask is placed by token-suffix alignment: assert
tok(text)[-len(tok(target)):] == tok(target); rows that fail the strict
suffix check fall back to longest-common-token-suffix alignment and are
counted. Run AFTER data_prep.py (needs /tmp/poc_twin/train.jsonl).
"""
import json, os, time
from array import array

import numpy as np
from transformers import AutoTokenizer

TMP = "/tmp/poc_twin"
SRC = os.path.join(TMP, "train.jsonl")
SEQ = 1025


def to_blocks(tokens):
    n_blocks = (len(tokens) - 1) // SEQ
    trimmed = np.asarray(tokens[: n_blocks * SEQ + 1])
    assert len(trimmed) == n_blocks * SEQ + 1
    blocks = np.lib.stride_tricks.as_strided(
        trimmed, shape=(n_blocks, SEQ),
        strides=(trimmed.strides[0] * 1024, trimmed.strides[0])).copy()
    return blocks


def main():
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    eos = tok.eos_token_id
    assert eos == 1
    mask = array("B")
    n_docs = n_exact = n_fallback = n_empty = 0
    tok_in_doc = tok_masked = 0
    batch_t, batch_g = [], []
    t0 = time.time()

    def flush(batch_t, batch_g):
        nonlocal n_docs, n_exact, n_fallback, n_empty, tok_in_doc, tok_masked
        enc_t = tok(batch_t, add_special_tokens=False)["input_ids"]
        enc_g = tok(batch_g, add_special_tokens=False)["input_ids"]
        for ids, g in zip(enc_t, enc_g):
            k = len(g)
            if k == 0 or k > len(ids):
                n_empty += 1
                mask.extend([0] * len(ids))
            elif ids[-k:] == g:
                n_exact += 1
                mask.extend([0] * (len(ids) - k))
                mask.extend([1] * k)
                tok_masked += k
            else:
                # longest common token suffix
                m = 0
                cap = min(len(ids), k)
                while m < cap and ids[len(ids) - 1 - m] == g[k - 1 - m]:
                    m += 1
                n_fallback += 1
                mask.extend([0] * (len(ids) - m))
                mask.extend([1] * m)
                tok_masked += m
            mask.append(0)          # doc-separator eos: no loss
            n_docs += 1
            tok_in_doc += len(ids) + 1

    with open(SRC, encoding="utf-8", errors="replace") as f:
        for line in f:
            r = json.loads(line)
            batch_t.append(r["text"])
            batch_g.append(r["target"])
            if len(batch_t) >= 2000:
                flush(batch_t, batch_g)
                batch_t, batch_g = [], []
                if n_docs % 40000 == 0:
                    print(f"  {n_docs} docs, {(time.time()-t0)/60:.1f} min, "
                          f"exact={n_exact} fallback={n_fallback} empty={n_empty}",
                          flush=True)
    if batch_t:
        flush(batch_t, batch_g)

    blocks = to_blocks(np.frombuffer(mask, dtype=np.uint8))
    np.save(os.path.join(TMP, "train_blocks_mask.npy"), blocks)
    meta = dict(docs=n_docs, exact_suffix=n_exact, fallback=n_fallback,
                empty_or_bad=n_empty, stream_tokens=int(tok_in_doc),
                masked_tokens=int(tok_masked),
                masked_frac=round(tok_masked / max(1, tok_in_doc), 4),
                blocks=int(len(blocks)))
    with open(os.path.join(TMP, "meta_mask.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
