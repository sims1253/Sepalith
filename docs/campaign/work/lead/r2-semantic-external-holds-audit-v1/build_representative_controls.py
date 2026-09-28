#!/usr/bin/env python3
"""Independently re-hash four source-backed TRAIN examples; never read targets."""
import hashlib
import json
from pathlib import Path

DECISIONS = Path("/mnt/e/sepalith/campaign-20260915/data-work/Semantic-external-holds-audit-v1/decisions.jsonl")
SEMANTIC = Path("/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-v2/shard5-root-03/shard-0005/semantic-ledger.jsonl")
IDS = [
    "22899cb124702c6d2df03b26",  # one stats origin
    "22b9d6c89c2911234027cb1c",  # five origins in stats + Matrix
    "231158f0e529c8c815c60cd9",  # operator import
    "236056feb230505782626bc9",  # three ggplot2 origins
]


def records(path):
    return {x["row_id"]: x for x in map(json.loads, path.read_text().splitlines())}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


decision = records(DECISIONS)
semantic = records(SEMANTIC)
rows = []
for row_id in IDS:
    d, s = decision[row_id], semantic[row_id]
    source_sha = sha(s["source_path"])
    namespace_sha = sha(s["namespace"]["path"])
    assert source_sha == s["source_sha256"]
    assert namespace_sha == s["namespace"]["sha256"]
    direct = set(s["reference_resolution"]["direct_references"])
    visited = set(s["reference_resolution"]["visited_dependency_names"])
    assert all(dep in direct or dep in visited for dep in d["dependencies"])
    rows.append({
        "row_id": row_id,
        "source_path": s["source_path"],
        "source_sha256": source_sha,
        "namespace_path": s["namespace"]["path"],
        "namespace_sha256": namespace_sha,
        "dependencies": d["dependencies"],
        "origins": {name: value["package"] for name, value in d["dependency_origin"].items()},
        "dependencies_visible_in_preedit_reference_inventory": True,
        "target_or_gold_read": False,
    })
out = Path(__file__).with_name("representative-controls.json")
out.write_text(json.dumps({
    "schema": "sepalith.dat10.namespace_representative_controls.v1",
    "rows": rows,
    "count": len(rows),
    "target_or_gold_read": False,
}, indent=2, sort_keys=True) + "\n")
print(out)
