#!/usr/bin/env python3
"""A2 transfer-strata acquisition (stack-edu + fineweb-edu -> 32K blocks).

Per the A2 §3.2 table's transfer strata, single-epoch draws. stack-edu
(HuggingFaceTB, ungated, permissive content policy) covers most; gaps
are DOCUMENTED in the plan (Python under-target vs the full 2.8B draw,
SQL short, Julia/MATLAB absent — alternatives in the acquisition plan).

Pipeline per stratum: download parquet(s) -> read content column ->
char-budget sample -> tokenize with the A2 32K vocab + eos-sep docs ->
1025-blocks npy -> /mnt/h/sepalith/a2_transfers/<stratum>/blocks.npy +
stats.json. Resume: strata with blocks.npy are skipped.

Usage:
  python3 ingest_transfers.py [--strata python,cpp,english,...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from array import array
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TOK = "/mnt/h/sepalith/datasets/a2_tokenizer_v1/tokenizer.json"
OUT = Path("/mnt/h/sepalith/a2_transfers")
SEQ = 1025

# stratum -> (repo, files, char budget). Char budgets are ~3.6 chars/tok
# at the 32K vocab (measured on R; code similar) -> tokens ≈ budget/3.6.
STACK_EDU = "HuggingFaceTB/stack-edu"
FWEDU = "HuggingFaceFW/fineweb-edu"
STRATA = {
    # full-draw targets (A2 table) in parens; budgets staged to what the
    # sources hold / the 13B checkpoint needs
    "python": (STACK_EDU, None, 3_600_000_000),     # target 2.8B tok; source ~1B
    "c_cpp": (STACK_EDU, ["C", "Cpp"], 2_200_000_000),
    "js_ts": (STACK_EDU, ["JavaScript", "TypeScript"], 1_800_000_000),
    "sql": (STACK_EDU, ["SQL"], 1_000_000_000),      # target 0.4B tok; short
    "english": (FWEDU, None, 6_600_000_000),         # target 1.85B tok
}


def to_blocks(tokens: array) -> np.ndarray:
    n_blocks = (len(tokens) - 1) // SEQ
    trimmed = np.asarray(tokens[: n_blocks * SEQ + 1])
    return np.lib.stride_tricks.as_strided(
        trimmed, shape=(n_blocks, SEQ),
        strides=(trimmed.strides[0] * 1024, trimmed.strides[0])).copy()


def ingest_stratum(name: str, repo: str, dirs, budget: int, tok: Tokenizer):
    out_dir = OUT / name
    if (out_dir / "blocks.npy").exists():
        print(f"[{name}] blocks present — skip", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    eos = tok.token_to_id("<|endoftext|>")
    rng = random.Random(20260824)

    # resolve parquet files
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(repo)
    files = []
    for s in info.siblings:
        if not s.rfilename.endswith(".parquet"):
            continue
        top = s.rfilename.split("/")[0]
        if dirs is None or top in dirs:
            files.append(s.rfilename)
    print(f"[{name}] {len(files)} parquets", flush=True)

    content_col = "text" if repo == FWEDU else "content"
    # fineweb-edu: 3036 parquets; sample a spread (single-epoch draw
    # means we only need budget-worth of unique text)
    if repo == FWEDU and len(files) > 40:
        rng2 = random.Random(11)
        rng2.shuffle(files)
        files = files[:40]
        print(f"[{name}] sampled {len(files)} fineweb parquets", flush=True)
    texts, nbytes = [], 0
    t0 = time.time()
    for i, rf in enumerate(files):
        if nbytes > budget:
            break
        local = hf_hub_download(repo, rf, repo_type="dataset")
        import pyarrow.parquet as pq
        tbl = pq.read_table(local, columns=[content_col])
        col = tbl.column(content_col).to_pylist()
        rng.shuffle(col)
        for t in col:
            if not t:
                continue
            if nbytes > budget:
                break
            t = t[:50_000]
            texts.append(t)
            nbytes += len(t.encode("utf-8", "replace"))
        os.remove(local)  # drvfs space: parquet consumed streaming
        print(f"[{name}] {i+1}/{len(files)} files, {nbytes/1e9:.1f} GB text, "
              f"{time.time()-t0:.0f}s", flush=True)

    # tokenize + pack
    stream = array("i")
    B = 2000
    for i in range(0, len(texts), B):
        for e in tok.encode_batch(texts[i:i + B]):
            stream.extend(e.ids)
            stream.append(eos)
    blocks = to_blocks(stream)
    np.save(out_dir / "blocks.npy", blocks)
    stats = dict(stratum=name, docs=len(texts), chars=nbytes,
                 tokens=len(stream), blocks=int(blocks.shape[0]),
                 est_tokens_32k=len(stream), repo=repo,
                 files_used=min(i + 1, len(files)))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[{name}] DONE {json.dumps(stats)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", default=",".join(STRATA))
    args = ap.parse_args()
    tok = Tokenizer.from_file(TOK)
    for name in args.strata.split(","):
        if name not in STRATA:
            print(f"[{name}] unknown stratum — skip", flush=True)
            continue
        repo, dirs, budget = STRATA[name]
        try:
            ingest_stratum(name, repo, dirs, budget, tok)
        except Exception as e:
            print(f"[{name}] ERROR {type(e).__name__}: {str(e)[:200]}",
                  flush=True)


if __name__ == "__main__":
    main()
