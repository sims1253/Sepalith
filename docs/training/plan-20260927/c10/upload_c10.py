#!/usr/bin/env python3
"""C10: one batched upload of TRAIN-derived artifacts to scholzmx/sepalith.

Explicit allow-list only: the admission ledgers and receipt, and the v4 arm-B
cohort, length profile and schedule. Row text appears only in the cohort file,
which holds the 15,006 TRAIN rows (already public under
campaign-20260915/sft/DAT10-finish-source-repair-v4-braceindent/) and the 3,861
admitted TRAIN-group roxygen rows. The review dump, sampled source files and the
superseded v3-based cohort stay local. The token comes from HF_TOKEN or the
huggingface_hub login cache; it is never printed.

    upload_c10.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = "scholzmx/sepalith"
PREFIX = "campaign-20260915/C10"
ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/C10-roxygen-admission-v1")
ALLOW = (
    "admission/admitted-ids.json",
    "admission/held-ids.json",
    "admission/receipt.json",
    "admission/checks.json",
    "admission/mechanical.jsonl",
    "admission/sample-checks.jsonl",
    "admission/semantic-classifications.json",
    "admission/recommended-context-ids.json",
    "arm-b-v4/arm-b-cohort-rows.jsonl",
    "arm-b-v4/cohort-receipt.json",
    "arm-b-v4/length-profile.json",
    "arm-b-v4/draw-schedule-v2-armb.json",
    "arm-b-v4/draw-schedule-v2-armb.summary.json",
)
FORBIDDEN = ("DAT-07", "final", "dev75", "dev250", "sealed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files = []
    for rel in ALLOW:
        if any(word in rel.lower() for word in FORBIDDEN):
            raise SystemExit(f"forbidden path: {rel}")
        path = ROOT / rel
        files.append({"local": str(path), "repo_path": f"{PREFIX}/{rel}", "bytes": path.stat().st_size, "sha256": sha256(path)})
    if args.dry_run:
        print(json.dumps(files, indent=1))
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    # HF_TOKEN from the environment, else the huggingface_hub login cache.
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    operations = [CommitOperationAdd(path_in_repo=f["repo_path"], path_or_fileobj=f["local"]) for f in files]
    info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations,
                             commit_message="C10: roxygen admission (3,861 rows) and arm-B v4 cohort and schedule",
                             commit_description="TRAIN-derived only. See docs/training/plan-20260927/status/C10.json in github.com/sims1253/Sepalith.")
    print(json.dumps({"commit_url": info.commit_url, "oid": info.oid, "files": files}, indent=1))


if __name__ == "__main__":
    main()
