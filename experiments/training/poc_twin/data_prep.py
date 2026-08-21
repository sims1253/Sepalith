"""
Tokenize + pack the astfim_v1 PSM-format R corpus for the twin POC.

- Copies the jsonl from drvfs (/mnt/h) to /tmp once, then tokenizes locally.
- 'text' field, MiniCPM5 tokenizer, add_special_tokens=False, eos (id 1)
  appended between documents; stream packed into overlapping 1025-token
  blocks (block k = tokens[k*1024 : k*1024+1025]; input=[:-1], target=[1:]).
- Eval: first 500 rows of eval.jsonl, packed the same way; total UTF-8
  bytes recorded for BPB.
- Output: /tmp/poc_twin/{train_blocks.npy,eval_blocks.npy,meta.json}
"""
import json, os, shutil, sys, time
from array import array

import numpy as np
from transformers import AutoTokenizer

TMP = "/tmp/poc_twin"
SRC_TRAIN = "/mnt/h/sepalith/datasets/astfim_v1/fixed/train.jsonl"
SRC_EVAL = "/mnt/h/sepalith/datasets/astfim_v1/fixed/eval.jsonl"
SEQ = 1025


def pack_file(path, n_rows=None, report_every=20000):
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    eos = tok.eos_token_id
    assert eos == 1
    stream = array("i")
    n_docs, n_bytes = 0, 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f
        if n_rows is not None:
            lines = (next(f) for _ in range(n_rows))
        batch, batch_bytes = [], []
        for line in lines:
            batch.append(json.loads(line)["text"])
            batch_bytes.append(0)
            if len(batch) >= 2000:
                enc = tok(batch, add_special_tokens=False)["input_ids"]
                for ids, txt in zip(enc, batch):
                    stream.extend(ids)
                    stream.append(eos)
                n_docs += len(batch)
                if n_docs % report_every == 0:
                    mr = len(stream) / 1e6
                    print(f"  {n_docs} docs, {mr:.1f}M tokens, "
                          f"{mr/max(1,(time.time()-t0))/60:.2f}M tok/min", flush=True)
                batch = []
        if batch:
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                stream.extend(ids)
                stream.append(eos)
            n_docs += len(batch)
    return np.frombuffer(stream, dtype=np.int32), n_docs


def to_blocks(tokens):
    n_blocks = (len(tokens) - 1) // SEQ
    trimmed = np.asarray(tokens[: n_blocks * SEQ + 1])
    assert len(trimmed) == n_blocks * SEQ + 1
    blocks = np.lib.stride_tricks.as_strided(
        trimmed, shape=(n_blocks, SEQ),
        strides=(trimmed.strides[0] * 1024, trimmed.strides[0])).copy()
    return blocks


def main():
    os.makedirs(TMP, exist_ok=True)
    local_train = os.path.join(TMP, "train.jsonl")
    local_eval = os.path.join(TMP, "eval500.jsonl")
    if not os.path.exists(local_train):
        print("copying train.jsonl to /tmp (drvfs is slow)...", flush=True)
        shutil.copy(SRC_TRAIN, local_train)
    if not os.path.exists(local_eval):
        with open(SRC_EVAL, "r", encoding="utf-8", errors="replace") as f, \
                open(local_eval, "w", encoding="utf-8") as g:
            for i, line in enumerate(f):
                if i >= 500:
                    break
                g.write(line)

    print("tokenizing eval(500)...", flush=True)
    eval_tokens, eval_docs = pack_file(local_eval, n_rows=500, report_every=10 ** 9)
    eval_bytes = sum(len(json.loads(l)["text"].encode("utf-8"))
                     for l in open(local_eval, encoding="utf-8", errors="replace"))
    eval_blocks = to_blocks(eval_tokens)
    np.save(os.path.join(TMP, "eval_blocks.npy"), eval_blocks)
    print(f"eval: {eval_docs} docs, {len(eval_tokens)/1e6:.2f}M tokens, "
          f"{eval_bytes/1e6:.2f}MB, {len(eval_blocks)} blocks", flush=True)

    print("tokenizing train (276k docs)...", flush=True)
    train_tokens, train_docs = pack_file(local_train)
    train_blocks = to_blocks(train_tokens)
    np.save(os.path.join(TMP, "train_blocks.npy"), train_blocks)
    meta = dict(
        train_docs=train_docs, train_tokens=int(len(train_tokens)),
        train_blocks=int(len(train_blocks)),
        eval_docs=eval_docs, eval_tokens=int(len(eval_tokens)),
        eval_blocks=int(len(eval_blocks)), eval_bytes=int(eval_bytes),
        seq=SEQ, eos_id=1)
    with open(os.path.join(TMP, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)

    # 20 greedy-generation prompts: first 120 tokens of eval rows 0..19
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    prompts = []
    with open(local_eval, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"][:120]
            prompts.append(ids)
    with open(os.path.join(TMP, "gen_prompts.json"), "w") as f:
        json.dump(prompts, f)
    print("wrote gen_prompts.json", flush=True)


if __name__ == "__main__":
    main()
