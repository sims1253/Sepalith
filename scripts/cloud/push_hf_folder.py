#!/usr/bin/env python3
"""Push a local folder to a private HF hub repo (model or dataset).

Generic cloud-staging counterpart used for base weights and small arm
datasets. Run LOCALLY, CPU-only (huggingface_hub; never imports torch).

Usage:  push_hf_folder.py <local_dir> <repo_id> <model|dataset> [path_in_repo]
Env:    HF_TOKEN (write access)
"""
import os
import sys

from huggingface_hub import HfApi

src, repo, kind = sys.argv[1], sys.argv[2], sys.argv[3]
path_in_repo = sys.argv[4] if len(sys.argv) > 4 else "."

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
api.upload_folder(folder_path=src, path_in_repo=path_in_repo, repo_id=repo,
                  repo_type=kind,
                  commit_message=f"staged from {src} ({path_in_repo})")
print(f"pushed {src} -> https://huggingface.co/"
      f"{'datasets/' if kind == 'dataset' else ''}{repo}", flush=True)
