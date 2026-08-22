"""
FIM dose-ladder data prep (2026-08-22, ladder agent).

The ladder needs TWO renderings of the SAME astfim_v1 fixed-corpus rows so
that only the objective mix (PSM span-FIM format vs plain causal LM) differs
across dose arms:

  FIM stream    : row["text"] as-is — the PSM surround rendering
                  (<|context|>path+prefix<|history|>\n\n<|cursor|><|suffix|>\n
                  {suffix}\n{span}\n<|end|>) — i.e. EXACTLY what the twin POC
                  trained on. REUSED as /tmp/poc_twin/train_blocks.npy
                  (byte-identical construction, do not rebuild).
  causal stream : the same rows re-rendered as plain documents in natural
                  order (path header + prefix + span + suffix, all PSM
                  markers stripped). Built here.

Both streams are packed with the parent data_prep.py discipline: MiniCPM5
tokenizer, add_special_tokens=False, eos (id 1) between docs, overlap-1
1025-token blocks.

Also emits:
  - eval_causal_blocks.npy + eval bytes (held-out plain text, causal BPB)
  - fim_eval_rows.json: held-out PSM rows selected for the served span-F1
    eval, CAPPED so prompt+generation stays inside the 1024-token trained
    position range (prompt<=640 tok, target<=320 tok, per-row max_tokens)
  - canary prompt files (causal/FIM in-corpus + causal held-out, first 120
    tokens of the respective renderings)

Output dir: /tmp/poc_twin/ladder/
"""
import json, os, time
from array import array

import numpy as np
from transformers import AutoTokenizer

TMP = "/tmp/poc_twin"
OUT = "/tmp/poc_twin/ladder"
TRAIN = os.path.join(TMP, "train.jsonl")
EVAL500 = os.path.join(TMP, "eval500.jsonl")
SEQ = 1025
FIM_PROMPT_TOK_MAX = 640   # keep prompt+gen inside trained positions
FIM_TARGET_TOK_MAX = 320
GEN_TOK_CAP = 384

CTX = "<|context|>"
HIST = "<|history|>"
SUF = "<|suffix|>\n"
END = "\n<|end|>"


def causal_from_row(r):
    """Plain-document rendering: path header + prefix + span + suffix."""
    p, t = r["prompt"], r["target"]
    i_ctx = p.find(CTX)
    i_hist = p.find(HIST)
    i_sf = p.find(SUF)
    if i_ctx != 0 or i_hist == -1 or i_sf == -1:
        return None
    ctx = p[len(CTX):i_hist]                      # path + "\n" + prefix
    i_end = p.rfind(END)
    s0 = i_sf + len(SUF)
    suffix = p[s0:i_end] if i_end != -1 else p[s0:]
    span = t
    if span.endswith(END):
        span = span[:-len(END)]
    elif span.endswith("<|end|>"):
        span = span[:-len("<|end|>")]
    return ctx + span + "\n" + suffix


def iter_causal(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            r = json.loads(line)
            c = causal_from_row(r)
            if c is None or "<|" in c:
                continue
            yield c


def pack_stream(texts, tok, eos, report_every=20000):
    stream = array("i")
    n_docs = 0
    t0 = time.time()
    batch = []
    for txt in texts:
        batch.append(txt)
        if len(batch) >= 2000:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                stream.extend(ids)
                stream.append(eos)
            n_docs += len(batch)
            if n_docs % report_every == 0:
                print(f"  {n_docs} docs, {len(stream)/1e6:.1f}M tokens, "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            batch = []
    if batch:
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            stream.extend(ids)
            stream.append(eos)
        n_docs += len(batch)
    return np.frombuffer(stream, dtype=np.int32), n_docs


def to_blocks(tokens):
    n_blocks = (len(tokens) - 1) // SEQ
    trimmed = np.asarray(tokens[: n_blocks * SEQ + 1])
    blocks = np.lib.stride_tricks.as_strided(
        trimmed, shape=(n_blocks, SEQ),
        strides=(trimmed.strides[0] * 1024, trimmed.strides[0])).copy()
    return blocks


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    eos = tok.eos_token_id
    assert eos == 1

    # ---- train causal stream ----
    print("tokenizing train CAUSAL stream (276k docs)...", flush=True)
    t_train, n_docs = pack_stream(iter_causal(TRAIN), tok, eos)
    train_blocks = to_blocks(t_train)
    np.save(os.path.join(OUT, "train_blocks_causal.npy"), train_blocks)
    print(f"train causal: {n_docs} docs, {len(t_train)/1e6:.2f}M tokens, "
          f"{len(train_blocks)} blocks", flush=True)

    # ---- eval causal stream (same 500 held-out rows, plain rendering) ----
    print("tokenizing eval CAUSAL stream (500 rows)...", flush=True)
    eval_rows = [json.loads(l) for l in open(EVAL500, encoding="utf-8",
                                             errors="replace")]
    causal_texts = [c for c in (causal_from_row(r) for r in eval_rows)
                    if c is not None and "<|" not in c]
    eval_bytes = sum(len(t.encode("utf-8")) for t in causal_texts)
    e_train, _ = pack_stream(iter(causal_texts), tok, eos)
    eval_blocks = to_blocks(e_train)
    np.save(os.path.join(OUT, "eval_blocks_causal.npy"), eval_blocks)
    print(f"eval causal: {len(causal_texts)} docs, {len(e_train)/1e6:.2f}M tokens, "
          f"{eval_bytes} bytes, {len(eval_blocks)} blocks", flush=True)

    # ---- served span-F1 eval row selection (in-distribution positions) ----
    pr = tok([r["prompt"] for r in eval_rows], add_special_tokens=False)["input_ids"]
    tg = tok([r["target"] for r in eval_rows], add_special_tokens=False)["input_ids"]
    sel = []
    for i, (r, p, t) in enumerate(zip(eval_rows, pr, tg)):
        if len(p) > FIM_PROMPT_TOK_MAX or len(t) > FIM_TARGET_TOK_MAX:
            continue
        tt = r["target"]
        if tt.endswith(END):
            tt = tt[:-len(END)]
        elif tt.endswith("<|end|>"):
            tt = tt[:-len("<|end|>")]
        sel.append(dict(i=i, prompt=r["prompt"], target=tt.strip("\n"),
                        prompt_tok=len(p), target_tok=len(t),
                        max_tokens=int(min(GEN_TOK_CAP, 1024 - len(p)))))
    with open(os.path.join(OUT, "fim_eval_rows.json"), "w") as f:
        json.dump(sel, f)
    print(f"fim eval rows selected: {len(sel)}/500 "
          f"(prompt<={FIM_PROMPT_TOK_MAX}tok, target<={FIM_TARGET_TOK_MAX}tok)", flush=True)

    # ---- canary prompts: first 120 tokens of the respective renderings ----
    with open(TRAIN, encoding="utf-8", errors="replace") as f:
        train_head = [json.loads(next(f)) for _ in range(10)]
    fim_prompts = tok([r["text"] for r in train_head],
                      add_special_tokens=False)["input_ids"]
    causal_prompts = tok([causal_from_row(r) for r in train_head],
                         add_special_tokens=False)["input_ids"]
    heldout_causal = tok([causal_from_row(r) for r in eval_rows[10:20]],
                         add_special_tokens=False)["input_ids"]
    json.dump([p[:120] for p in fim_prompts],
              open(os.path.join(OUT, "canary_prompts_fim_train.json"), "w"))
    json.dump([p[:120] for p in causal_prompts],
              open(os.path.join(OUT, "canary_prompts_causal_train.json"), "w"))
    json.dump([p[:120] for p in heldout_causal],
              open(os.path.join(OUT, "canary_prompts_causal_heldout.json"), "w"))

    meta = dict(
        train_docs=n_docs, train_tokens_causal=int(len(t_train)),
        train_blocks_causal=int(len(train_blocks)),
        eval_docs_causal=len(causal_texts), eval_tokens_causal=int(len(e_train)),
        eval_blocks_causal=int(len(eval_blocks)), eval_bytes_causal=int(eval_bytes),
        fim_eval_rows=len(sel), seq=SEQ, eos_id=eos,
        fim_stream="REUSED /tmp/poc_twin/train_blocks.npy (parent data_prep.py)")
    with open(os.path.join(OUT, "meta_ladder.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
