#!/usr/bin/env python3
"""Pull SFT data files from a private HF dataset repo into DATA_DIR.

Cloud-side counterpart of the push scripts (push_sft_v7.py /
push_hf_folder.py). Default = the B-series mixture (sft_v7): downloads
{train,eval}.jsonl.gz + stats.json and gunzips — byte-identical to
/mnt/h/sepalith/datasets/sft_v7 on the NAS (gzip is lossless; the
trainer's shuffle(seed=42)+48k-cap row selection therefore matches local
runs).

Arms/other datasets: set SFT_DATA_REPO + SFT_DATA_FILES, a comma list of
`repo/path>local/name` (or just `repo/path`, local name = basename).

Usage:  python pull_data.py [DATA_DIR=/tmp/data/sft_v7]
Env:    HF_TOKEN (read-access to SFT_DATA_REPO),
        SFT_DATA_REPO (default scholzmx/sepalith-sft-v7),
        SFT_DATA_FILES (default "train.jsonl>train.jsonl,
        eval.jsonl>eval.jsonl, stats.json>stats.json"; .gz variants are
        tried first for jsonl and gunzipped transparently)
"""
import gzip
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = os.environ.get("SFT_DATA_REPO", "scholzmx/sepalith-sft-v7")
DEFAULT_FILES = "train.jsonl>train.jsonl,eval.jsonl>eval.jsonl,stats.json>stats.json"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/data/sft_v7")
OUT.mkdir(parents=True, exist_ok=True)

for spec in os.environ.get("SFT_DATA_FILES", DEFAULT_FILES).split(","):
    spec = spec.strip()
    if ">" in spec:
        remote, local = spec.split(">", 1)
    else:
        remote, local = spec, spec.rsplit("/", 1)[-1]
    src = None
    for candidate in ([remote + ".gz", remote] if remote.endswith(".jsonl")
                      else [remote]):
        try:
            src = hf_hub_download(repo_id=REPO, filename=candidate,
                                  repo_type="dataset")
            remote = candidate
            break
        except Exception:
            continue
    if src is None:
        raise SystemExit(f"pull_data: {remote} not found in {REPO} "
                         "(neither plain nor .gz)")
    dst = OUT / local
    dst.parent.mkdir(parents=True, exist_ok=True)
    if remote.endswith(".gz"):
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 22)
    else:
        shutil.copyfile(src, dst)
    print(f"pulled {remote} -> {dst} ({dst.stat().st_size:,} bytes)", flush=True)

print(f"DATA READY at {OUT}", flush=True)
