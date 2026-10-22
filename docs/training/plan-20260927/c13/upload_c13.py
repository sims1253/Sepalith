#!/usr/bin/env python3
"""C13: one batched upload of the TRAIN-derived offline data to scholzmx/sepalith.

Explicit allow-list only: the RFT cohort (verified census samples as
TRAIN-shaped rows plus original TRAIN rows; iteration 1: 4,413 + 1,891), its
selection ledger, TRAIN-mix ids, schedule v2 and length profile; the DPO pairs
(iteration 1: 18,268) and their pair schedule; and the data manifest with every
hash. The commit message takes its counts from the manifest. Everything derives from
the C12 census on TRAIN prompts and the public TRAIN rows; nothing from DEV or
the sealed final set. The token comes from HF_TOKEN or the huggingface_hub
login cache and is never printed.

    upload_c13.py [--model-id sepalith-2b-edit-sft-arm-a | sepalith-2b-edit-sft-arm-a-t12] [--dry-run] [--receipt PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = "scholzmx/sepalith"
DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/C13-offline-v1")
ALLOW = (
    "manifest.json",
    "rft/rft-cohort-rows.jsonl",
    "rft/rft-selection.jsonl",
    "rft/train-mix-ids.json",
    "rft/draw-schedule-v2-rft.json",
    "rft/length-profile.json",
    "dpo/dpo-pairs.jsonl",
    "dpo/pair-schedule.json",
)
FORBIDDEN = ("dat-07", "final", "dev75", "dev250", "sealed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="sepalith-2b-edit-sft-arm-a")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = DATA / args.model_id
    prefix = f"campaign-20260915/C13/{args.model_id}"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("split") != "train" or manifest.get("final_set_access") is not False:
        raise SystemExit("the data manifest must be TRAIN-only without final-set access")
    recorded = {Path(r["path"]).relative_to(root).as_posix(): r["sha256"] for r in manifest["outputs"].values()}
    files = []
    for rel in ALLOW:
        if any(word in rel.lower() for word in FORBIDDEN):
            raise SystemExit(f"forbidden path: {rel}")
        path = root / rel
        digest = sha256(path)
        if rel in recorded and recorded[rel] != digest:
            raise SystemExit(f"{rel} differs from the data manifest")
        files.append({"local": str(path), "repo_path": f"{prefix}/{rel}", "bytes": path.stat().st_size, "sha256": digest})
    if args.dry_run:
        print(json.dumps(files, indent=1))
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    operations = [CommitOperationAdd(path_in_repo=f["repo_path"], path_or_fileobj=f["local"]) for f in files]
    info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations,
                             commit_message=(f"C13: RFT cohort ({manifest['cohort']['rft_rows']:,} verified samples + "
                                             f"{manifest['cohort']['train_mix_rows']:,} TRAIN rows) and {manifest['dpo']['pairs']:,} "
                                             f"DPO pairs ({args.model_id})"),
                             commit_description="TRAIN-derived only, from the C12 census. See docs/training/plan-20260927/status/C13.json "
                                                "in github.com/sims1253/Sepalith.")
    receipt = {"commit_url": info.commit_url, "oid": info.oid, "repo": f"datasets/{REPO}", "path": f"{prefix}/",
               "files": [{k: f[k] for k in ("repo_path", "bytes", "sha256")} for f in files]}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
