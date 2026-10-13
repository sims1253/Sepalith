#!/usr/bin/env python3
"""Deterministic integrity checks for the DAT10 materialisation increments."""
from __future__ import annotations
import ast
import hashlib
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[5]
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
OWN = ROOT / "docs/campaign/work/lead/r2-novel-materialization-v1"
BASE = E / "DAT10-novel-v1"
CPT = json.loads((ROOT / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json").read_text())
CPT_GROUPS = {str(k): str(v) for k, v in CPT["groups"].items()}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    scripts = [
        OWN / "repair_mixed_eol.py", OWN / "build_repaired_gate_v3.py",
        OWN / "inventory_all_metadata.py", OWN / "materialize_increment_na_rm.py",
        OWN / "prepare_source_walk_shard.py",
    ]
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    repair = BASE / "completion-mixed-eol-repair-v1"
    repair_packets = list(jsonl(repair / "candidate-packets.jsonl"))
    if len(repair_packets) != 976 or any(x.get("result", {}).get("status") != "converted" for x in repair_packets):
        raise AssertionError("mixed-EOL queue did not convert exactly 976 rows")
    for packet in repair_packets:
        repair_meta = packet["result"]["provenance"].get("mixed_eol_repair")
        if not isinstance(repair_meta, dict) or repair_meta.get("geometry_preserved") is not True or repair_meta.get("target_bytes_unchanged") is not True:
            raise AssertionError("mixed-EOL provenance/geometry marker missing")
    repair_report = json.loads((repair / "token-audit/report.json").read_text())
    if repair_report["converted_inputs"] != 976 or repair_report["tokenizer_candidates"] != 976 or repair_report["excluded"]:
        raise AssertionError("mixed-EOL token audit failed")

    gate = json.loads((BASE / "candidate-gate-v3/candidate-gate-summary-v3.json").read_text())
    if gate["candidate_total"] != 5466 or gate["all_global_train_excluding_cpt_validation"] != 5109:
        raise AssertionError("v3 gate arithmetic changed")
    if gate["cpt_partition_counts"] != {"cpt_partition_missing": 112, "cpt_train": 4997, "cpt_validation": 357}:
        raise AssertionError("v3 CPT gate counts changed")
    gate_rows = list(jsonl(BASE / "candidate-gate-v3/candidate-gate-ledger-v3.jsonl"))
    if len(gate_rows) != 5466 or len({x["row_id"] for x in gate_rows}) != 5466:
        raise AssertionError("v3 ledger identity failure")

    inventory = BASE / "all-data-inventory-v1/all-metadata-inventory-v1.jsonl"
    count = 0
    source_categories = Counter()
    forbidden = {"prompt_text", "target_text", "document_text", "full_prompt", "target_body"}
    for row in jsonl(inventory):
        count += 1
        source_categories[row["source_category"]] += 1
        if forbidden & set(row):
            raise AssertionError("content-bearing field leaked into metadata inventory")
    if count != 376830:
        raise AssertionError(f"metadata inventory row count {count}")
    expected_categories = {
        "existing_parent_materialized": 11094,
        "genuine_metadata_exclusion_pending_review": 69485,
        "heldout_global_or_cpt_validation": 20422,
        "source_checked_excluded_or_repair_queue": 1163,
        "unattempted_supported_global_train": 200753,
        "unsupported_or_unrouted_metadata": 68804,
        "verified_eligible_materialized_candidate": 5109,
    }
    if dict(source_categories) != expected_categories:
        raise AssertionError(f"source categories changed: {source_categories}")
    for name, expected in (("support-review-noop-450-or-more.jsonl", 450), ("support-review-documentation-579-or-more.jsonl", 579)):
        if sum(1 for _ in jsonl(BASE / "all-data-inventory-v1" / name)) != expected:
            raise AssertionError(f"support queue count changed: {name}")

    inc = BASE / "expansion-increment-na-rm-v1"
    inc_rows = list(jsonl(inc / "candidate-token-rows.jsonl"))
    if len(inc_rows) != 136 or any((x.get("source_ref") or {}).get("split") != "train_group" for x in inc_rows):
        raise AssertionError("na_rm global-TRAIN increment failure")
    if any(CPT_GROUPS.get(str((x.get("source_ref") or {}).get("group_id"))) == "cpt_validation" for x in inc_rows):
        raise AssertionError("CPT validation content entered na_rm increment")
    if sum(1 for _ in jsonl(inc / "source-exclusions.jsonl")) != 2 or sum(1 for _ in jsonl(inc / "cpt-validation-heldout-metadata.jsonl")) != 1:
        raise AssertionError("na_rm hold/exclusion counts changed")
    review = list(jsonl(inc / "root-review-packet-v4-na-rm.jsonl"))
    if len(review) != 5245 or len({(x.get("source_ref") or {}).get("row_id") for x in review}) != 5245:
        raise AssertionError("v4 review packet identity/count failure")
    if any(CPT_GROUPS.get(str((x.get("source_ref") or {}).get("group_id"))) == "cpt_validation" for x in review):
        raise AssertionError("CPT validation content entered v4 review packet")

    plan = json.loads((BASE / "all-data-inventory-v1/source-walk-plan-v1.json").read_text())
    if plan["total_metadata_rows"] != 376830 or plan["unattempted_supported_global_train_rows"] != 200753:
        raise AssertionError("source-walk coverage changed")
    if plan["io_shards"]["purpose"] != "resume/I-O only, no data exclusion":
        raise AssertionError("source-walk shard policy changed")
    shard = BASE / "source-walk-shards-v1/shard-0000"
    shard_manifest = json.loads((shard / "manifest.json").read_text())
    if shard_manifest["rows_copied"] != 5000 or shard_manifest["policy"]["shard_is_io_unit_only"] is not True:
        raise AssertionError("source-walk shard 0 count/policy failure")
    for path in (shard / "audit-rows.jsonl", shard / "structured-audit-rows.jsonl"):
        for row in jsonl(path):
            if forbidden & set(row):
                raise AssertionError("content-bearing field leaked into source-walk metadata input")
    print(json.dumps({"status": "pass", "metadata_rows": count, "v3_review_rows": 5109, "v4_review_rows": 5245, "source_categories": dict(source_categories), "hashes": {"inventory": sha(inventory), "v3_gate": sha(BASE / "candidate-gate-v3/candidate-gate-summary-v3.json"), "v4": sha(inc / "root-review-packet-v4-na-rm.jsonl")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
