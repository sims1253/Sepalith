#!/usr/bin/env python3
"""Stage the B-series SFT mixture (sft_v7) to the private HF dataset repo.

Run LOCALLY, CPU-only (huggingface_hub + gzip; never imports torch, so it is
safe while the 5090 is busy). Gzips /mnt/h/sepalith/datasets/sft_v7 with pigz
when available and uploads train/eval.jsonl.gz + stats.json.

Usage:  python push_sft_v7.py [--repo scholzmx/sepalith-sft-v7]
Env:    HF_TOKEN (dataset write; see ~/.zshrc)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SRC = Path("/mnt/h/sepalith/datasets/sft_v7")
REPO = "scholzmx/sepalith-sft-v7"
if "--repo" in sys.argv:
    REPO = sys.argv[sys.argv.index("--repo") + 1]

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)

with tempfile.TemporaryDirectory(prefix="sft_v7_push_") as tmp:
    tmp = Path(tmp)
    for fname in ("train.jsonl", "eval.jsonl"):
        gz = tmp / (fname + ".gz")
        pigz = subprocess.run(["pigz", "-k", "-c", str(SRC / fname)],
                              stdout=open(gz, "wb"), stderr=subprocess.DEVNULL)
        if pigz.returncode != 0:  # fall back to gzip(1)
            subprocess.run(["gzip", "-c", str(SRC / fname)], stdout=open(gz, "wb"),
                           check=True)
        print(f"gzipped {fname}: {SRC} {gz.stat().st_size:,} bytes", flush=True)
        api.upload_file(path_or_fileobj=str(gz), path_in_repo=fname + ".gz",
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"sft_v7 snapshot: {fname} (gzip)")
    api.upload_file(path_or_fileobj=str(SRC / "stats.json"),
                    path_in_repo="stats.json", repo_id=REPO, repo_type="dataset",
                    commit_message="sft_v7 snapshot: stats.json")
print(f"pushed -> https://huggingface.co/datasets/{REPO} (private)", flush=True)
