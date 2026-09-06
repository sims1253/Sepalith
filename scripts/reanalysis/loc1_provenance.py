#!/usr/bin/env python3
"""Bounded read-only git-object checks for the 60 saved LOC1 evaluation rows."""

import argparse
import hashlib
import subprocess
from pathlib import Path

from common import Evidence, dump, keyed


def git(repo, *args):
    p = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, timeout=20, check=False
    )
    return p.returncode, p.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("inventory", "snapshot", "mirror", "out", "private-out"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    e = Evidence(a.inventory, a.snapshot)
    builder = e.module("/experiments/data-mining/loc1_build_set.py")
    source = keyed(e.rows("/loc1_s0_r/set.jsonl"), "qid")
    results = []
    private = []
    for s in e.json("/loc1_s0_results/results.json")["rows"]:
        row = source[s["qid"]]
        repo = Path(a.mirror) / row["repo"]
        sha = row["sha"]
        r = {
            "qid": row["qid"],
            "repo": row["repo"],
            "child": sha,
            "source_path": row["src_path"],
        }
        rc, parent = git(repo, "rev-parse", "--verify", sha + "^")
        r["parent_available"] = rc == 0
        r["parent"] = parent.decode().strip() if rc == 0 else None
        rc, tree = git(repo, "rev-parse", "--verify", sha + "^{tree}")
        r["child_tree"] = tree.decode().strip() if rc == 0 else None
        rc, msg = git(repo, "show", "-s", "--format=%s%n%b", sha)
        subject, _, body = msg.decode(errors="replace").partition("\n")
        r["query_matches_commit"] = (
            rc == 0 and builder.clean_query(subject, body) == row["query"]
        )
        r["message_sha256"] = hashlib.sha256(msg).hexdigest() if rc == 0 else None
        rc, blob = git(repo, "show", sha + "^:" + row["src_path"])
        r["parent_source_available"] = rc == 0
        r["parent_source_sha256"] = (
            hashlib.sha256(blob).hexdigest() if rc == 0 else None
        )
        names = (
            {
                f[0]
                for f in builder.parse_top_level_functions(
                    blob.decode(errors="replace")
                )
            }
            if rc == 0
            else set()
        )
        r["gold_names_in_parent_source"] = sum(g["name"] in names for g in row["gold"])
        r["n_gold"] = len(row["gold"])
        r["missing_parent_gold_names"] = [
            g["name"] for g in row["gold"] if g["name"] not in names
        ]
        results.append(r)
        private.append(
            r
            | {
                "message": msg.decode(errors="replace"),
                "parent_source": blob.decode(errors="replace") if rc == 0 else None,
            }
        )
    dump(
        a.out,
        {
            "rows": results,
            "method": "Parent-source name availability using frozen historical parser; not a parent-corpus retrieval evaluation.",
        },
    )
    dump(a.private_out, private)


if __name__ == "__main__":
    main()
