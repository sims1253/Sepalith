"""Independently join the frozen 12--26 semantic records.

This review validates record identity and the source/provenance joins. It does
not reread raw source files, render targets, or admit any row for training.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-semantic-streaming-queue-shards12plus-v1"
)
SHARDS = list(range(12, 27))


def rows(record: dict):
    """Read a pinned JSONL record and verify its declared hash and count."""
    path = Path(record["path"])
    hasher = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            hasher.update(line)
            count += 1
            yield json.loads(line)
    assert hasher.hexdigest() == record["sha256"], path
    assert count == record["rows"], (path, count, record["rows"])


def main() -> int:
    counts = collections.Counter()
    all_ids: set[str] = set()
    shard_summaries = []

    for shard in SHARDS:
        manifest_path = ROOT / f"shard-{shard:04d}/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == "sepalith.dat10.sourcewalk_roxy_semantic.v6"
        assert manifest["status"] == "complete_review_only"
        assert manifest["training_admission"] is False
        assert manifest["exact_id_closure"] is True

        binding = manifest["streaming_binding"]
        assert binding["shard"] == shard
        assert binding["queued_rows"] == len(binding["queued_ids"])
        selected = set(binding["queued_ids"])
        assert len(selected) == binding["queued_rows"]

        provenance = {}
        for item in rows(binding["provenance_ledger"]):
            rid = item["row_id"]
            assert rid not in provenance
            provenance[rid] = item
            assert item["shard"] == shard
        expected = {
            rid
            for rid, item in provenance.items()
            if item["status"] == "provenance_pass_semantic_analyzer_queued"
        }
        assert selected == expected, (shard, len(selected), len(expected))

        packet_refs = collections.defaultdict(list)
        for packet in rows(binding["candidate_packet"]):
            ref = packet["row_ref"]
            packet_refs[ref["row_id"]].append(ref)

        output = manifest["output_binding"]
        observed = set()
        joined_statuses = collections.Counter()
        for semantic in rows(output):
            rid = semantic["row_id"]
            assert rid in selected
            assert rid not in observed
            assert rid not in all_ids
            observed.add(rid)
            all_ids.add(rid)

            original = provenance[rid]
            assert original["source_path_sha256"] == semantic["source_sha256"]
            for flag in (
                "global_train",
                "global_group_source_membership",
                "protected_disjoint",
                "source_line_group_join",
                "source_parse_ok",
                "source_stat_stable",
                "description_stat_stable",
                "strict_protocol_ok",
            ):
                assert original[flag] is True, (rid, flag, original[flag])
            assert original["license_decision"]["ok"] is True

            matching = [
                ref
                for ref in packet_refs[rid]
                if ref["family"] == original["family"]
                and ref["package_id"] == original["package_id"]
                and ref["group_id"] == original["group_id"]
                and ref["line"] == original["raw_source_line"]
                and ref["split"] == "train_group"
            ]
            assert matching, rid
            joined_statuses[semantic["status"]] += 1

        assert observed == selected, (shard, len(observed), len(selected))
        counts.update(joined_statuses)
        shard_summaries.append(
            {
                "shard": shard,
                "provenance_rows": len(provenance),
                "queued_rows": len(selected),
                "joined_rows": len(observed),
                "status_counts": dict(joined_statuses),
            }
        )

    print(
        json.dumps(
            {
                "schema": "sepalith.dat10.semantic_provenance_join.v1",
                "shards": SHARDS,
                "joined_rows": len(all_ids),
                "status_counts": dict(counts),
                "shard_summaries": shard_summaries,
                "source_record_join": "pass",
                "training_admission": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
