"""A2-prime POC data prep (2026-08-23 night) — the FIRST twin-rig run of
the A2 mixture structure (user GO).

Scope (honest): the twin rig is R-only, so this POC validates the R-side
structure of A2-prime — the transfer strata (28% of the real draw) and
the edit-diff stratum are NOT in it; they belong to the real data build.
What it DOES test, at 350.2M tokens on the 206.5M TinyGQA with the
adopted recipe unchanged:

  - dose 0.30 PSM-FIM doc-rate (center of the adopted 20-35% band at
    R-only scale; in the real 72%-R mixture this is ~24% of total draw)
  - the FIM pool is a MIX, the v8/A2 novelty:
      ~83% astfim_v1 semantic spans (PSM text as-is, the ladder stream)
      ~12% astfim_random_v1 random-cursor cuts (text = prompt+target,
        the v8-noop-random-cursor.md layer-2 stream, first time trained)
      ~5%  no_op PSM renders (empty-span targets, 6 stop geometries)
  - everything else identical to the ladder's dose-0.30 arm would be
    (same causal stream, same seed, same optimizer/schedule)

Also packs a random-cut held-out eval slice (rc rows not in the train
sample) for a per-stratum BPB read alongside the standard causal/PSM
slices.

Output: /tmp/poc_twin/a2/train_blocks_fim_mixed.npy, eval_blocks_rc.npy
"""
import json, os, random, shutil, time
from array import array

import numpy as np
from transformers import AutoTokenizer

TMP = "/tmp/poc_twin"
OUT = os.path.join(TMP, "a2")
SEQ = 1025
RC_SRC = "/mnt/h/sepalith/datasets/astfim_random_v1/train-000.jsonl"
NOOP_SRC = "/mnt/h/sepalith/datasets/scenarios_v1/no_op.jsonl"
AST_SRC = os.path.join(TMP, "train.jsonl")
CONTEXT, HISTORY, CURSOR, SUFFIX, END = (
    "<|context|>", "<|history|>", "<|cursor|>", "<|suffix|>", "<|end|>")


def noop_psm_text(row):
    """Mirror build_astfim.render exactly, with an EMPTY span (mid='')."""
    pre = "\n".join(list(row["prefix"]) + list(row["region_old"]))\
        .replace("\r\n", "\n").rstrip("\n")
    suf = "\n".join(row.get("suffix") or []).replace("\r\n", "\n")\
        .strip("\n")
    prompt = (f"{CONTEXT}{row['package']}/{row['path']}\n{pre}\n{HISTORY}"
              f"\n\n{CURSOR}{SUFFIX}\n{suf}\n{END}\n")
    target = f"\n{END}"          # mid="" -> f"{{mid}}\n{END}"
    return prompt + target


def pack_texts(texts, tok, eos):
    stream = array("i")
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) >= 2000:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                stream.extend(ids)
                stream.append(eos)
            batch = []
    for ids in tok(batch, add_special_tokens=False)["input_ids"] if batch else []:
        stream.extend(ids)
        stream.append(eos)
    return np.frombuffer(stream, dtype=np.int32)


def to_blocks(tokens):
    n_blocks = (len(tokens) - 1) // SEQ
    trimmed = np.asarray(tokens[: n_blocks * SEQ + 1])
    blocks = np.lib.stride_tricks.as_strided(
        trimmed, shape=(n_blocks, SEQ),
        strides=(trimmed.strides[0] * 1024, trimmed.strides[0])).copy()
    return blocks


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(20260823)
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    eos = tok.eos_token_id
    assert eos == 1

    # 1. astfim PSM docs (local file, the ladder stream source)
    ast = []
    with open(AST_SRC) as f:
        for line in f:
            ast.append(json.loads(line)["text"])
    print(f"astfim docs: {len(ast)}", flush=True)

    # 2. random-cut docs (copy locally once, drvfs), 12% of ast count;
    #    hold out the LAST 400 rows as the rc eval slice
    rc_local = os.path.join(OUT, "rc.jsonl")
    if not os.path.exists(rc_local):
        shutil.copy(RC_SRC, rc_local)
    rc_rows = [json.loads(l) for l in open(rc_local)]
    rc_eval = rc_rows[-400:]
    rc_train_pool = rc_rows[:-400]
    n_rc = round(0.12 * len(ast))
    rng.shuffle(rc_train_pool)
    rc_texts = [r["prompt"] + r["target"] for r in rc_train_pool[:n_rc]]
    print(f"random-cut docs: {len(rc_texts)} (of {len(rc_train_pool)} pool)",
          flush=True)

    # 3. no_op PSM docs — all of them (short docs; token share is small)
    noop_rows = [json.loads(l) for l in open(NOOP_SRC)]
    noop_texts = [noop_psm_text(r) for r in noop_rows]
    print(f"no_op docs: {len(noop_texts)}", flush=True)

    # 4. mix + pack (seeded shuffle: the pool order defines the stream)
    pool = [("ast", t) for t in ast] + [("rc", t) for t in rc_texts] \
        + [("noop", t) for t in noop_texts]
    rng.shuffle(pool)
    t0 = time.time()
    tokens = pack_texts([t for _, t in pool], tok, eos)
    blocks = to_blocks(tokens)
    np.save(os.path.join(OUT, "train_blocks_fim_mixed.npy"), blocks)
    from collections import Counter
    shares = Counter(k for k, _ in pool)
    print(json.dumps(dict(
        docs={k: shares[k] for k in shares},
        doc_shares={k: round(shares[k] / len(pool), 4) for k in shares},
        tokens=int(len(tokens)), blocks=int(blocks.shape[0]),
        elapsed_s=round(time.time() - t0, 1))), flush=True)

    # 5. rc eval slice
    ev_tokens = pack_texts([r["prompt"] + r["target"] for r in rc_eval],
                           tok, eos)
    np.save(os.path.join(OUT, "eval_blocks_rc.npy"), to_blocks(ev_tokens))
    print("rc eval blocks saved", flush=True)


if __name__ == "__main__":
    main()
