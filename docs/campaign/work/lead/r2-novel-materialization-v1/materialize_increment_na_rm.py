#!/usr/bin/env python3
"""Freeze the complete supported ``na_rm_propagation`` increment.

DAT-03 has 139 rows for this family.  The reviewed broad source pass converted
137 and explicitly excluded 2.  This script filters that already token-audited
source output, verifies global TRAIN/CPT boundaries and the all-row inventory,
and publishes a new review packet.  It uses no family quota: the family is
complete in this increment; the all-family source walk remains resumable.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
from collections import Counter
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[5]
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
BROAD = E / "DAT-04-structured-broad-token-audit/candidate-token-rows.jsonl"
BROAD_REPORT = E / "DAT-04-structured-batch-broad-v2/report.json"
INVENTORY = E / "DAT10-novel-v1/all-data-inventory-v1/all-metadata-inventory-v1.jsonl"
CPT = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
ROSTER = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/novel-train-audit-roster.jsonl"
CURRENT = E / "DAT10-novel-v1/candidate-gate-v3/all-global-train-token-rows-v3.jsonl"
OUT = E / "DAT10-novel-v1/expansion-increment-na-rm-v1"


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


def key(rec: dict[str, Any]) -> str:
    return str((rec.get("source_ref") or rec).get("row_id") or "")


def main() -> int:
    for p in (BROAD, BROAD_REPORT, INVENTORY, CPT, ROSTER, CURRENT):
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)
    cpt_doc = json.loads(CPT.read_text(encoding="utf-8"))
    cpt = {str(k): str(v) for k, v in cpt_doc.get("groups", {}).items()}
    roster_ids = {str(x.get("row_id")) for x in jsonl(ROSTER)}
    current_ids = {key(x) for x in jsonl(CURRENT)}
    inventory: dict[str, dict[str, Any]] = {}
    for x in jsonl(INVENTORY):
        inventory[str(x["row_id"])] = x

    converted_all = [x for x in jsonl(BROAD) if (x.get("row", {}).get("family") or x.get("source_ref", {}).get("family")) == "na_rm_propagation"]
    if len(converted_all) != 137:
        raise RuntimeError(f"expected complete token-audited na_rm increment of 137, got {len(converted_all)}")
    heldout = [x for x in converted_all if inventory.get(key(x), {}).get("cpt_partition") == "cpt_validation"]
    converted = [x for x in converted_all if inventory.get(key(x), {}).get("cpt_partition") != "cpt_validation"]
    if len(heldout) != 1 or len(converted) != 136:
        raise RuntimeError(f"expected 136 global-train and 1 CPT-validation na_rm token rows, got {len(converted)} and {len(heldout)}")
    ids = [key(x) for x in converted]
    all_ids = [key(x) for x in converted_all]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("duplicate na_rm row ids")
    if set(all_ids) & roster_ids:
        raise RuntimeError("na_rm increment overlaps historical 6690 roster")
    if set(all_ids) & current_ids:
        raise RuntimeError("na_rm increment overlaps current candidate gate")

    inventory_rows = []
    for x in converted_all:
        rid = key(x)
        if rid not in inventory:
            raise RuntimeError(f"na_rm row absent from complete inventory: {rid}")
        inv = inventory[rid]
        if inv.get("family") != "na_rm_propagation":
            raise RuntimeError(f"inventory family mismatch: {rid}")
        if inv.get("global_registry_split") != "train_group":
            raise RuntimeError(f"non-global-train na_rm row: {rid}")
        if inv.get("cpt_partition") != "cpt_validation":
            inventory_rows.append(inv)

    report = json.loads(BROAD_REPORT.read_text(encoding="utf-8"))
    excluded = [x for x in report.get("excluded", []) if x.get("family") == "na_rm_propagation"]
    if len(excluded) != 2:
        raise RuntimeError(f"expected 2 explicit na_rm source exclusions, got {len(excluded)}")
    excluded_ids = {str(x.get("id")) for x in excluded}
    if excluded_ids & set(ids):
        raise RuntimeError("source exclusion overlaps converted na_rm row")
    for rid in excluded_ids:
        if rid not in inventory:
            raise RuntimeError(f"excluded na_rm row absent from complete inventory: {rid}")

    token_path = OUT / "candidate-token-rows.jsonl"
    with token_path.open("w", encoding="utf-8", newline="\n") as f:
        for x in converted:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    exclusion_path = OUT / "source-exclusions.jsonl"
    with exclusion_path.open("w", encoding="utf-8", newline="\n") as f:
        for x in sorted(excluded, key=lambda r: str(r.get("id"))):
            inv = inventory[str(x["id"])]
            f.write(json.dumps({
                "row_id": x.get("id"), "family": x.get("family"),
                "reason": x.get("reason"), "inventory": inv,
                "status": "source_checked_excluded_unadmitted",
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    heldout_path = OUT / "cpt-validation-heldout-metadata.jsonl"
    with heldout_path.open("w", encoding="utf-8", newline="\n") as f:
        for x in heldout:
            rid = key(x)
            f.write(json.dumps({
                "row_id": rid, "family": "na_rm_propagation",
                "cpt_partition": "cpt_validation", "global_registry_split": inventory[rid]["global_registry_split"],
                "status": "heldout_cpt_validation_not_in_review_packet",
                "target_sha256": inventory[rid]["target_sha256"],
                "canonical_row_sha256": inventory[rid]["canonical_row_sha256"],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    ledger_path = OUT / "increment-ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8", newline="\n") as f:
        for inv in sorted(inventory_rows, key=lambda r: str(r["row_id"])):
            f.write(json.dumps({
                "row_id": inv["row_id"], "group_id": inv["group_id"],
                "family": inv["family"], "source": inv["source"],
                "source_file": inv["source_file"], "source_line": inv["source_line"],
                "global_registry_split": inv["global_registry_split"],
                "cpt_partition": inv["cpt_partition"],
                "source_category_before_increment": inv["source_category"],
                "status": "source_verified_token_audited_review_only",
                "target_sha256": inv["target_sha256"],
                "canonical_row_sha256": inv["canonical_row_sha256"],
                "raw_line_sha256": inv["raw_line_sha256"],
                "license_status": inv["license_status"],
                "target_chars": inv["target_chars"],
                "target_truncation": False,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    # Add this increment to a fresh root-review packet.  The current packet is
    # copied as input and remains untouched.
    current_rows = list(jsonl(CURRENT))
    combined_ids = {key(x) for x in current_rows}
    if combined_ids & set(ids):
        raise RuntimeError("combined packet identity collision")
    review_path = OUT / "root-review-packet-v4-na-rm.jsonl"
    with review_path.open("w", encoding="utf-8", newline="\n") as f:
        for x in current_rows + converted:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    summary = {
        "schema": "DAT-10-novel-expansion-increment-na-rm-v1",
        "status": "review_only_unadmitted",
        "family": "na_rm_propagation",
        "family_metadata_total": 139,
        "converted_token_audited_rows": len(converted_all),
        "global_train_token_audited_rows": len(converted),
        "cpt_validation_token_audited_rows_heldout": len(heldout),
        "source_checked_excluded_rows": len(excluded),
        "source_exclusion_reasons": dict(Counter(str(x.get("reason")) for x in excluded)),
        "global_train_rows": len(converted),
        "cpt_partition_counts": dict(Counter(inventory[rid]["cpt_partition"] for rid in ids)),
        "historical_roster_overlap": 0,
        "current_candidate_gate_overlap": 0,
        "broad_source_report": str(BROAD_REPORT),
        "broad_source_report_sha256": sha(BROAD_REPORT),
        "broad_token_input": str(BROAD),
        "broad_token_input_sha256": sha(BROAD),
        "token_rows": {"path": str(token_path), "sha256": sha(token_path), "rows": len(converted)},
        "increment_ledger": {"path": str(ledger_path), "sha256": sha(ledger_path), "rows": len(inventory_rows)},
        "source_exclusions": {"path": str(exclusion_path), "sha256": sha(exclusion_path), "rows": len(excluded)},
        "cpt_validation_heldout": {"path": str(heldout_path), "sha256": sha(heldout_path), "rows": len(heldout)},
        "review_packet": {"path": str(review_path), "sha256": sha(review_path), "rows": len(current_rows) + len(converted)},
        "policy": {
            "all_rows_in_family_processed": True,
            "family_row_time_quota": False,
            "cpt_validation_reserved": True,
            "train_content_only": True,
            "target_truncation": False,
            "admission": "none; root source/license/duplicate/quality review remains mandatory",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = OUT / "increment-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "DAT-10-novel-expansion-increment-na-rm-manifest",
        "summary": str(summary_path), "summary_sha256": sha(summary_path),
        "outputs": {k: v for k, v in summary.items() if k in ("token_rows", "increment_ledger", "source_exclusions", "cpt_validation_heldout", "review_packet")},
        "inputs": {
            "inventory": {"path": str(INVENTORY), "sha256": sha(INVENTORY)},
            "cpt": {"path": str(CPT), "sha256": sha(CPT)},
            "historical_roster": {"path": str(ROSTER), "sha256": sha(ROSTER)},
            "current_review_packet": {"path": str(CURRENT), "sha256": sha(CURRENT)},
        },
    }
    manifest_path = OUT / "increment-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"family": "na_rm_propagation", "converted": len(converted), "excluded": len(excluded), "review_rows": len(current_rows) + len(converted), "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
