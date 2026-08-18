#!/usr/bin/env python3
"""Push the sepalith CRAN dataset to the private HF hub repo (incremental).

Usage:
  push_hf.py                 # full snapshot: card + provenance/ + datasets/
  push_hf.py <repo_id>       # same, explicit repo
  push_hf.py --finish-block  # targeted: card + the provenance-enriched
                             # finish-block sample as finish-block/sample_top2000.jsonl
Reads HF_TOKEN from the environment (see ~/.zshrc).
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path("/mnt/h/sepalith")
FINISH_BLOCK_LOCAL = Path("/home/m0hawk/Documents/Sepalith/experiments"
                          "/synthetic/finish_block_sample.jsonl")
FINISH_BLOCK_REPO = "finish-block/sample_top2000.jsonl"
TARGETED = "--finish-block" in sys.argv
REPO = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None

api = HfApi(token=os.environ["HF_TOKEN"])
if REPO is None:
    REPO = api.whoami()["name"] + "/sepalith"
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

## Provenance fields (2026-08-19 schema, backfilled by enrich_provenance.py)
Every record in every derived dataset carries:
- `source_url` — exact source artifact: CRAN tarball link
  (`https://cran.r-project.org/src/contrib/<pkg>_<ver>.tar.gz`) for
  finish-block / scenarios_v1 / sft_ablation records (joined from
  `provenance/<pkg>.json`); `{repo_url}/commit/{sha_full}` for edit_pairs_v1;
  `null` for pure-synthetic records (synthetic_analyst_v1).
- `license` — SPDX-ish string from the source: the CRAN DESCRIPTION license
  for package-derived records; for edit_pairs_v1 the repo DESCRIPTION
  `License:` line if present, else the LICENSE/LICENCE/COPYING file content
  mapped heuristically (MIT / Apache-2.0 / GPL-3 / ... ), else the explicit
  string `"unknown"`; `null` for pure-synthetic records.
- CRAN-derived records also carry `version` and `upstream` (from
  `provenance/<pkg>.json`) and a `derivation` line naming the extractor
  (e.g. `finish_block.py tree-sitter-r extraction from <tarball_sha256>`,
  `<family> constructor (scenarios.py|comment_to_code.py)`,
  `format_sft_types.py ... ry dump-types (ry-worktrees/dump-types @ a907e10)`
  for the types-variant ablation files).
- LLM-generated records additionally carry `full_prompt` (the literal prompt
  text sent — reconstructed from the generator's template + grid cell / code
  block, flagged `prompt_reconstructed: true`), `model` (exact id),
  `generator` (script), and `generated_at`. In comment_to_code_synthetic the
  pre-existing `generator` field holds the endpoint/model tag and is mirrored
  into `model`.

## Field notes
- `area` = top-level dir (`R`, `tests`, `man`, `vignettes`, ...). `man/` (.Rd) is
  DERIVED from roxygen comments (generated at build time) — exclude or downweight
  it in corpus assembly to avoid double-counting documentation text.
- `normalized: true` — content is post air/jarl, not the raw original.
- `finish-block/sample_top2000.jsonl` — finish-block sample with the provenance
  fields above (enriched copy of the local experiments sample).
"""
(ROOT / "datasets" / "README.md").write_text(card)
api.upload_file(path_or_fileobj=str(ROOT / "datasets" / "README.md"),
                path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
if TARGETED:
    api.upload_file(path_or_fileobj=str(FINISH_BLOCK_LOCAL),
                    path_in_repo=FINISH_BLOCK_REPO,
                    repo_id=REPO, repo_type="dataset",
                    commit_message="finish-block sample: provenance-enriched "
                                   "(source_url/license/version/derivation)")
    print(f"pushed card + {FINISH_BLOCK_REPO} -> "
          f"https://huggingface.co/datasets/{REPO} (private)")
    sys.exit(0)
api.upload_folder(folder_path=str(ROOT / "provenance"), path_in_repo="provenance",
                  repo_id=REPO, repo_type="dataset",
                  commit_message="provenance + license texts snapshot",
                  allow_patterns=["*.json", "*.license.txt"])
api.upload_folder(folder_path=str(ROOT / "datasets"), path_in_repo="datasets",
                  repo_id=REPO, repo_type="dataset",
                  commit_message="shards + manifest snapshot")
print(f"pushed -> https://huggingface.co/datasets/{REPO} (private)")
