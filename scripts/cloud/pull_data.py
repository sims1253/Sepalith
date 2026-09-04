#!/usr/bin/env python3
"""Pull the B-series SFT mixture from the private HF dataset repo.

Cloud-side counterpart of push_sft_v7.py. Downloads {train,eval}.jsonl.gz +
stats.json and gunzips to DATA_DIR — byte-identical to
/mnt/h/sepalith/datasets/sft_v7 on the NAS (gzip is lossless; the trainer's
shuffle(seed=42)+48k-cap row selection therefore matches local runs).

Usage:  python pull_data.py [DATA_DIR=/root/data/sft_v7]
Env:    HF_TOKEN (read-access to SFT_DATA_REPO), SFT_DATA_REPO (default
        scholzmx/sepalith-sft-v7)
"""
import gzip
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = os.environ.get("SFT_DATA_REPO", "scholzmx/sepalith-sft-v7")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/data/sft_v7")
OUT.mkdir(parents=True, exist_ok=True)

for fname in ("train.jsonl", "eval.jsonl", "stats.json"):
    remote = fname + ".gz" if fname.endswith(".jsonl") else fname
    src = hf_hub_download(repo_id=REPO, filename=remote, repo_type="dataset")
    dst = OUT / fname
    if remote.endswith(".gz"):
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 22)
    else:
        shutil.copyfile(src, dst)
    print(f"pulled {remote} -> {dst} ({dst.stat().st_size:,} bytes)", flush=True)

print(f"DATA READY at {OUT}", flush=True)
