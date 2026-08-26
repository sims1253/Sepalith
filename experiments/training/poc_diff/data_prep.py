"""Task 1: span-triple data renderer (shared by both arms).

Turns astfim_v1 PSM rows into (prefix, span, suffix) triples with ONE
tokenizer pass, so the diffusion twin and the AR twin consume byte-identical
token streams:

  context ids  = tok(prefix_part)   — the row's prompt minus its terminator
                                        line; exactly what the AR arm sees
                                        before it must emit the span.
  span ids     = tok(span_text)     — the row's target minus "\n<|end|>".
                                        [] encodes the [EMPTY] no-op class
                                        (rendered as the single [EMPTY]
                                        token by poc_diff.span_region).

Row anatomy (verified on astfim_v1/fixed, 2026-08-26; see __init__.py):
  text   = prefix_part + span_text + "\n<|end|>"
  prompt = prefix_part + "\n<|end|>\n"
  target = span_text + "\n<|end|>"

Triples are ROW-NATIVE (the corpus' own edit spans), per the companion
survey nse-ot-flows-survey-2026-08.md §4 ("edit pairs as-is; both twins see
identical triples") — NOT synthetic uniform-start spans as an early plan
draft suggested. Deviation is deliberate: the paired-arm discipline
(identical data, only head/objective differs) outranks it.

Pre-registered caps (frozen): span > 256 tokens dropped (both arms), prompt
(prompt side) > 640 tokens dropped (ladder pattern), prompt+span <= 1024
asserted. Outputs land in /tmp/poc_diff (tmpfs; drvfs staging rule).

Output artifacts:
  train_triples.jsonl  — one triple per kept row (ids only + provenance)
  train_flat_ids.bin   — int32 stream: [prompt_ids span_region] per row,
                         concatenated; for RAM-lean mmap batching
  train_row_lens.bin   — int32 per row: (prompt_len, span_region_len) pairs
  eval_triples.jsonl   — held-out triples from eval.jsonl docs only, with
                         the raw text fields for text-level metrics
  span_length_hist.json, meta.json
"""
import json
import os
import shutil
import sys
import time
from array import array

import numpy as np

from experiments.training.poc_diff import (
    CTX_TOK, END_MARKER, MAX_SPAN_TOK, PROMPT_TERMINATOR, PROMPT_TOK_MAX,
    SRC_EVAL, SRC_TRAIN, TMP, span_region,
)

EVAL_ROWS = 500          # house convention: first 500 eval rows = held-out
BATCH = 2000             # tokenizer batch (parent data_prep.py pattern)


def split_row(row):
    """(prefix_part, span_text) from a PSM row, or (None, None) if the row
    does not follow the fixed-corpus anatomy."""
    p, t = row["prompt"], row["target"]
    if not isinstance(p, str) or not isinstance(t, str):
        return None, None
    # ladder tolerance: some targets end "<|end|>" without the newline
    if t.endswith(END_MARKER):
        t = t[: -len(END_MARKER)]
    elif t.endswith("<|end|>"):
        t = t[: -len("<|end|>")]
    else:
        return None, None
    if not p.endswith(PROMPT_TERMINATOR):
        return None, None
    return p[: -len(PROMPT_TERMINATOR)], t


def new_stats():
    return dict(rows=0, kept=0, malformed=0, span_gt_256=0, prompt_gt_640=0,
                empty_span=0, ctx_gt_1024=0, boundary_checked=0,
                boundary_aligned=0, span_tok_total=0)


def build_triples(rows, tok, stats, keep_text=False, boundary_every=100):
    """Streaming generator of triple dicts. Mutates `stats` (drop counters).
    `rows` yields raw jsonl dicts. Deterministic: no RNG anywhere."""
    end_ids = tok(END_MARKER, add_special_tokens=False)["input_ids"]
    n_since_boundary = boundary_every  # check the first batch too
    batch = []

    def flush():
        nonlocal n_since_boundary
        if not batch:
            return
        pre = tok([r["prefix"] for r in batch], add_special_tokens=False)["input_ids"]
        spn = tok([r["span"] for r in batch], add_special_tokens=False)["input_ids"]
        out = []
        for meta, p_ids, s_ids in zip(batch, pre, spn):
            stats["rows"] += 1
            if len(s_ids) > MAX_SPAN_TOK:
                stats["span_gt_256"] += 1
                continue
            if len(p_ids) > PROMPT_TOK_MAX:
                stats["prompt_gt_640"] += 1
                continue
            region = span_region(s_ids)
            if len(p_ids) + len(region) > CTX_TOK:
                stats["ctx_gt_1024"] += 1
                continue
            stats["kept"] += 1
            stats["span_tok_total"] += len(s_ids)
            if not s_ids:
                stats["empty_span"] += 1
            rec = dict(prompt_ids=p_ids, span_ids=s_ids,
                       span_len=len(s_ids), kind=meta.get("kind"),
                       package=meta.get("package"), path=meta.get("path"))
            if keep_text:
                rec["prefix_text"] = meta["prefix"]
                rec["span_text"] = meta["span"]
            out.append(rec)
        # BPE boundary check ~1 row per `boundary_every`: does the concat of
        # the parts' tokenizations equal the joint tokenization of text?
        # (tok(a)+tok(b) == tok(a+b) fails where BPE merges across the
        # prefix|span boundary; measured, not enforced)
        if n_since_boundary >= boundary_every and out:
            n_since_boundary = 0
            meta = batch[0]
            p_ids, s_ids = out[0]["prompt_ids"], out[0]["span_ids"]
            if s_ids:  # empty-span rows have no boundary to check
                joint = tok(meta["prefix"] + meta["span"] + END_MARKER,
                            add_special_tokens=False)["input_ids"]
                stats["boundary_checked"] += 1
                if list(p_ids) + list(s_ids) + end_ids == joint:
                    stats["boundary_aligned"] += 1
        n_since_boundary += len(out)
        batch.clear()
        return out

    for row in rows:
        prefix, span = split_row(row)
        if prefix is None:
            stats["rows"] += 1
            stats["malformed"] += 1
            continue
        batch.append(dict(prefix=prefix, span=span, kind=row.get("kind"),
                          package=row.get("package"), path=row.get("path")))
        if len(batch) >= BATCH:
            yield from flush()
    yield from flush()


def iter_jsonl(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield json.loads(line)


def write_train_artifacts(triples_iter, out_prefix, stats):
    """jsonl triples + flat int32 token stream + (prompt_len, span_region_len)
    per row. The flat stream is written incrementally (never buffered in
    RAM: a2-transfer holds ~20GB and earlyoom is load-bearing history)."""
    flat = array("i")
    lens = array("i")
    n = 0
    t0 = time.time()
    with open(out_prefix + "triples.jsonl", "w", encoding="utf-8") as g:
        for rec in triples_iter:
            region = span_region(rec["span_ids"])
            g.write(json.dumps(rec) + "\n")
            flat.extend(rec["prompt_ids"])
            flat.extend(region)
            lens.append(len(rec["prompt_ids"]))
            lens.append(len(region))
            n += 1
            if n % 20000 == 0:
                print(f"  {n} triples, {(len(flat)/1e6):.1f}M tokens, "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    with open(out_prefix + "flat_ids.bin", "wb") as f:
        flat.tofile(f)
    with open(out_prefix + "row_lens.bin", "wb") as f:
        lens.tofile(f)
    return n, len(flat)


def length_histogram(path):
    """Span-length buckets from a triples jsonl (token lengths)."""
    counts = {"0_empty": 0, "1-10": 0, "11-50": 0, "51-100": 0,
              "101-256": 0}
    all_lens = array("i")
    for line in open(path, encoding="utf-8"):
        n = json.loads(line)["span_len"]
        all_lens.append(n)
        if n == 0:
            counts["0_empty"] += 1
        elif n <= 10:
            counts["1-10"] += 1
        elif n <= 50:
            counts["11-50"] += 1
        elif n <= 100:
            counts["51-100"] += 1
        else:
            counts["101-256"] += 1
    arr = np.frombuffer(all_lens, dtype=np.int32)
    return counts, dict(mean=float(arr.mean()), p50=float(np.percentile(arr, 50)),
                        p95=float(np.percentile(arr, 95)), max=int(arr.max()))


def main():
    os.makedirs(TMP, exist_ok=True)
    local_train = os.path.join(TMP, "train.jsonl")
    local_eval = os.path.join(TMP, "eval500.jsonl")
    if not os.path.exists(local_train):
        print("copying train.jsonl to /tmp (drvfs is slow)...", flush=True)
        shutil.copy(SRC_TRAIN, local_train)
    if not os.path.exists(local_eval):
        with open(SRC_EVAL, encoding="utf-8", errors="replace") as f, \
                open(local_eval, "w", encoding="utf-8") as g:
            for i, line in enumerate(f):
                if i >= EVAL_ROWS:
                    break
                g.write(line)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

    print("building EVAL triples (held-out, first 500 eval docs)...", flush=True)
    st_e = new_stats()
    with open(os.path.join(TMP, "eval_triples.jsonl"), "w", encoding="utf-8") as g:
        for rec in build_triples(iter_jsonl(local_eval), tok, st_e, keep_text=True):
            g.write(json.dumps(rec) + "\n")
    print(f"eval: {json.dumps(st_e)}", flush=True)

    print("building TRAIN triples (276k docs)...", flush=True)
    st_t = new_stats()
    n_train, n_tok = write_train_artifacts(
        build_triples(iter_jsonl(local_train), tok, st_t),
        os.path.join(TMP, "train_"), st_t)
    print(f"train: {json.dumps(st_t)}", flush=True)

    hist, stats = length_histogram(os.path.join(TMP, "train_triples.jsonl"))
    with open(os.path.join(TMP, "span_length_hist.json"), "w") as f:
        json.dump(dict(buckets=hist, stats=stats,
                       dropped_gt_256=st_t["span_gt_256"]), f, indent=1)

    meta = dict(
        train=st_t, eval=st_e, train_triples=n_train,
        train_prompt_plus_span_tokens=int(n_tok),
        boundary_align_frac=(st_t["boundary_aligned"] /
                             max(1, st_t["boundary_checked"])),
        caps=dict(max_span_tok=MAX_SPAN_TOK, prompt_tok_max=PROMPT_TOK_MAX,
                  ctx_tok=CTX_TOK),
        tokenizer="openbmb/MiniCPM5-1B", triples="row-native (survey §4)")
    with open(os.path.join(TMP, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
