#!/usr/bin/env python3
"""Independently verify the row-free DAT-02 v2 split manifest.

The verifier reads only the emitted metadata manifest. It checks group and
source-parent disjointness, the historical precedence policy, and count
consistency without opening any raw source or target record.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        default="/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    path = Path(args.manifest)
    data = json.loads(path.read_text())
    groups = data["groups"]
    assert len({g["group_id"] for g in groups}) == len(groups)
    assert all(g["split"] for g in groups)

    # A parent token must map to exactly one assigned split. This is checked
    # from the emitted token lists independently of the builder's in-memory
    # DSU check.
    parent_splits: dict[str, set[str]] = collections.defaultdict(set)
    for group in groups:
        for parent in group.get("parent_tokens", []):
            parent_splits[parent].add(group["split"])
    parent_crossings = {
        parent: sorted(splits)
        for parent, splits in parent_splits.items()
        if len(splits) > 1
    }

    by_split = collections.Counter(g["split"] for g in groups)
    candidate_groups = sum(g["candidate_rows"] > 0 for g in groups)
    candidate_rows_by_split = collections.Counter()
    final_rows = 0
    final_groups = 0
    final_named_packages = 0
    final_named_repositories = 0
    for group in groups:
        candidate_rows_by_split[group["split"]] += group["candidate_rows"]
        if group["split"] == "final_candidate_group" and group["final_capable_rows"] > 0:
            final_groups += 1
            final_rows += group["final_capable_rows"]
            final_named_packages += group["named_package_count"]
            final_named_repositories += group["named_repository_count"]

    # Historical overlap precedence: candidate-bearing sft_v7 identities are
    # train-only unless an explicit historical/TU3 held-out flag overrides.
    sft7_train_policy = all(
        not ("sft_v7" in g["flags"] and g["candidate_rows"] > 0)
        or "historical_eval" in g["flags"]
        or "tu3" in g["flags"]
        or g["split"] == "train_group"
        for g in groups
    )
    sft7_marker_policy = all(
        not ("sft_v7" in g["flags"] and g["candidate_rows"] > 0)
        or "final/dev_ineligible:sft_v7" in g["flags"]
        for g in groups
    )
    final_exclusion_policy = all(
        g["split"] != "final_candidate_group"
        or not ({"sft_v7", "historical_eval", "tu3"} & set(g["flags"]))
        for g in groups
    )
    source_only_sft7_policy = all(
        not ("sft_v7" in g["flags"] and g["candidate_rows"] == 0)
        or g["split"] == "quarantine_sft_v7"
        or "historical_eval" in g["flags"]
        or "tu3" in g["flags"]
        for g in groups
    )

    expected = data["counts"]
    checks = {
        "group_ids_unique": len({g["group_id"] for g in groups}) == len(groups),
        "group_split_disjoint": all(g["split"] for g in groups),
        "source_parent_no_cross_split": not parent_crossings,
        "sft_v7_candidate_train_only_or_held_out_override": sft7_train_policy,
        "sft_v7_candidate_groups_flagged_final_dev_ineligible": sft7_marker_policy,
        "sft_v7_source_only_quarantined": source_only_sft7_policy,
        "final_groups_historical_excluded": final_exclusion_policy,
        "candidate_group_count_consistent": candidate_groups == expected["candidate_identity_groups"],
        "candidate_rows_split_counts_consistent": {
            key: value for key, value in candidate_rows_by_split.items() if value
        } == expected["candidate_rows_by_split"],
        "final_group_count_consistent": final_groups == expected["actual_clean_final_candidate_packages"],
        "final_row_count_consistent": final_rows == expected["actual_clean_final_candidate_rows"],
        "final_named_package_count_consistent": final_named_packages == expected["actual_clean_final_candidate_named_packages"],
        "final_named_repository_count_consistent": final_named_repositories == expected["actual_clean_final_candidate_named_repositories"],
        "final_records_opened_false": data["inputs"]["final_records_opened"] is False and data["checks"]["final_records_opened"] is False,
        "raw_inputs_changed_false": data["checks"]["raw_inputs_changed"] is False,
    }
    assert all(checks.values()), checks
    result = {
        "task": "DAT-02",
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "group_count": len(groups),
        "named_package_identity_tokens_total": expected["named_package_identity_tokens_total"],
        "named_repository_identity_tokens_total": expected["named_repository_identity_tokens_total"],
        "named_package_identity_groups_total": expected["named_package_identity_groups_total"],
        "named_repository_identity_groups_total": expected["named_repository_identity_groups_total"],
        "parent_token_count": len(parent_splits),
        "parent_crossings": parent_crossings,
        "groups_by_split": dict(by_split),
        "candidate_rows_by_split": dict(candidate_rows_by_split),
        "checks": checks,
        "raw_inputs_opened": False,
        "targets_opened": False,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
