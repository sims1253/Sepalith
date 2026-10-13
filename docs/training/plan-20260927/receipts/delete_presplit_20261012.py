#!/usr/bin/env python3
"""C16 follow-up (user decision 2026-10-12): remove the LoRA-era case families
copied from NAS datasets/cases_v1/ and every mixtures/sft_v*/ file from the
public dataset in ONE commit, together with the updated README. History is
not rewritten.

Usage: delete_presplit_20261012.py PLAN.json README.md RECEIPT.json

PLAN.json lists {"repo", "nas", "hub_size"} rows and the base revision. Every
row is re-checked before the commit: the repo path must be under families/
or mixtures/sft_v*, still exist at the base revision with the planned size,
and its NAS source must still exist, so the only copies left are local.
"""
import json
import os
import sys

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO = "scholzmx/sepalith"
plan_path, readme, receipt_path = sys.argv[1:4]
plan = json.load(open(plan_path))
api = HfApi(token=os.environ.get("HF_TOKEN"))
head = api.dataset_info(REPO).sha
if head != plan["base_revision"]:
    sys.exit(f"Hub moved: {head} != planned base {plan['base_revision']}; rebuild the plan")
remote = {i.path: i.size for i in api.list_repo_tree(REPO, repo_type="dataset", recursive=True,
                                                    revision=head) if hasattr(i, "size")}
ops, rows, skipped = [], [], []
for r in plan["delete"]:
    p = r["repo"]
    if not (p.startswith("families/") or p.startswith("mixtures/sft_v")):
        sys.exit(f"refusing path outside the approved scope: {p}")
    if remote.get(p) != r["hub_size"]:
        sys.exit(f"Hub copy changed or missing: {p}")
    if not os.path.isfile(r["nas"]):
        skipped.append(r)
        continue
    ops.append(CommitOperationDelete(path_in_repo=p))
    rows.append(r)
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme))
info = api.create_commit(
    repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head,
    commit_message=(f"C16: remove LoRA-era mixtures and cases_v1 families "
                    f"({len(rows)} files; user decision 2026-10-12; history kept)"))
out = {"base_revision": head, "commit": info.oid, "deleted_files": len(rows),
       "deleted_bytes": sum(r["hub_size"] for r in rows), "skipped_no_nas_copy": skipped}
json.dump(out, open(receipt_path, "w"), indent=1)
print(json.dumps(out, indent=1))
