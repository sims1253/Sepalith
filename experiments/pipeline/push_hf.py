#!/usr/bin/env python3
"""Push the sepalith CRAN dataset to the private HF hub repo (incremental).

Usage: push_hf.py [repo_id]   (default <username>/sepalith-cran)
Reads HF_TOKEN from the environment (see ~/.zshrc).
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path("/mnt/h/sepalith")
REPO = sys.argv[1] if len(sys.argv) > 1 else None

api = HfApi(token=os.environ["HF_TOKEN"])
if REPO is None:
    REPO = api.whoami()["name"] + "/sepalith-cran"
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)

card = """---
language: [code]
license: other               # per-package licenses in provenance/ and per-record fields
size_categories: unknown
---
# Sepalith CRAN corpus (research, private)

Top-downloaded CRAN packages, current releases, normalized with `air format` +
`jarl check --fix`. For training an R next-edit-suggestion model.

## Provenance & takedown
- `provenance/<pkg>.json` — license, authors, upstream, tarball sha256, ingest date.
- `datasets/packages/<pkg>.jsonl` — one shard per package; **removal = deleting that shard**.
- `datasets/manifest.jsonl` — append-only ingest log.
- Raw tarballs retained offline; `.tarball_sha256` links shards to originals.
- Removal requests: delete the package's shard + provenance record and rebuild.

## Field notes
- `area` = top-level dir (`R`, `tests`, `man`, `vignettes`, ...). `man/` (.Rd) is
  DERIVED from roxygen comments (generated at build time) — exclude or downweight
  it in corpus assembly to avoid double-counting documentation text.
- `normalized: true` — content is post air/jarl, not the raw original.
"""
(ROOT / "datasets" / "README.md").write_text(card)
api.upload_file(path_or_fileobj=str(ROOT / "datasets" / "README.md"),
                path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
api.upload_folder(folder_path=str(ROOT / "provenance"), path_in_repo="provenance",
                  repo_id=REPO, repo_type="dataset",
                  commit_message="provenance + license texts snapshot",
                  allow_patterns=["*.json", "*.license.txt"])
api.upload_folder(folder_path=str(ROOT / "datasets"), path_in_repo="datasets",
                  repo_id=REPO, repo_type="dataset",
                  commit_message="shards + manifest snapshot")
print(f"pushed -> https://huggingface.co/datasets/{REPO} (private)")
