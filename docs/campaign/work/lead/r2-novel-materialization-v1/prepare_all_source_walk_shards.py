#!/usr/bin/env python3
"""Prepare all remaining DAT10 source-walk shard inputs in one audit pass.

The precomputed inventory assigns every supported global-TRAIN row to an I/O
shard.  This command reads DAT-03 exactly once, writes each original metadata
row to its assigned shard, and routes it to completion/structured inputs.  It
never applies family/package/row/time quotas; shard size is only a resumable
I/O unit.  Shard 0000 is preserved when already prepared.
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[5]
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
AUDIT = E / "DAT-03-row-audit.jsonl"
INVENTORY = E / "DAT10-novel-v1/all-data-inventory-v1/all-metadata-inventory-v1.jsonl"
SHARD_PLAN = E / "DAT10-novel-v1/all-data-inventory-v1/source-walk-shards-v1.jsonl"
SHARD_ROOT = E / "DAT10-novel-v1/source-walk-shards-v1"
PROGRESS = SHARD_ROOT / "source-walk-preparation-progress-v1.json"
FORBIDDEN = {"prompt", "prompt_text", "target", "target_text", "document_text", "full_prompt", "full_text"}
EXPECTED_AUDIT_SHA = "9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd"
EXPECTED_INVENTORY_SHA = "e17e501b6742374ffd5d441cbd0fe644f4b7e1e61aeafe45b515281a201b5e72"
EXPECTED_PLAN_SHA = "bc4b673e2c3f761457ff08037586993c1f7adff023e411f19bcade7dff8dc039"
SUPPORTED = {"roxygen_drafting", "no_op", "rename_propagation", "format_propagation", "pipe_rewrite", "na_rm_propagation"}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    for p in (AUDIT, INVENTORY, SHARD_PLAN):
        if not p.exists():
            raise FileNotFoundError(p)
    if sha(INVENTORY) != EXPECTED_INVENTORY_SHA or sha(SHARD_PLAN) != EXPECTED_PLAN_SHA:
        raise RuntimeError("inventory_or_shard_plan_hash_mismatch")
    expected_rows: dict[int, int] = {}
    for line in SHARD_PLAN.open(encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            expected_rows[int(x["shard"])] = int(x["rows"])
    if not expected_rows:
        raise RuntimeError("empty_shard_plan")
    wanted: dict[str, int] = {}
    for line in INVENTORY.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        shard = row.get("io_shard")
        if shard is None:
            continue
        rid = str(row["row_id"])
        if rid in wanted:
            raise RuntimeError(f"duplicate_inventory_row:{rid}")
        wanted[rid] = int(shard)
    if sum(expected_rows.values()) != len(wanted):
        raise RuntimeError(f"inventory/plan count mismatch: {len(wanted)} vs {sum(expected_rows.values())}")

    # Existing completed/prepared shard 0000 is never mutated.  Remaining
    # shards must be fresh so an interrupted run cannot silently append.
    handles: dict[int, dict[str, Any]] = {}
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    written: Counter[int] = Counter()
    for shard, expected in sorted(expected_rows.items()):
        out = SHARD_ROOT / f"shard-{shard:04d}"
        final_audit = out / "audit-rows.jsonl"
        final_manifest = out / "manifest.json"
        if final_manifest.exists() and final_audit.exists():
            if shard == 0:
                continue
            raise ValueError(f"shard_already_prepared:{shard}")
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("audit-rows.jsonl", "completion-audit-rows.jsonl", "structured-audit-rows.jsonl"):
            partial = out / (name + ".partial")
            if partial.exists():
                raise ValueError(f"partial_shard_exists:{shard}:{name}")
            files[name] = partial.open("w", encoding="utf-8", newline="\n")
        handles[shard] = {"out": out, "files": files}

    progress = {"schema": "DAT-10-source-walk-preparation-progress-v1", "status": "running", "rows_scanned": 0, "selected_rows": 0, "shards": {}, "updated_at": datetime.now(timezone.utc).isoformat(), "policy": "progress_only; no family/row/time omission"}
    atomic_json(PROGRESS, progress)
    audit_digest = hashlib.sha256()
    selected_ids: set[str] = set()
    scanned = 0
    try:
        with AUDIT.open("rb", buffering=4 * 1024 * 1024) as source:
            for raw_line in source:
                audit_digest.update(raw_line)
                scanned += 1
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                rid = str(row.get("row_id") or "")
                shard = wanted.get(rid)
                if shard is None or shard not in handles:
                    continue
                if rid in selected_ids:
                    raise RuntimeError(f"duplicate_audit_row:{rid}")
                selected_ids.add(rid)
                if FORBIDDEN & set(row):
                    raise RuntimeError(f"content_field_in_metadata:{FORBIDDEN & set(row)}")
                family = str(row.get("family") or "")
                if family not in SUPPORTED and not family.startswith("finish_block"):
                    raise RuntimeError(f"unsupported_planned_family:{family}")
                encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                handles[shard]["files"]["audit-rows.jsonl"].write(encoded)
                if family.startswith("finish_block"):
                    route = "completion"
                    handles[shard]["files"]["completion-audit-rows.jsonl"].write(encoded)
                else:
                    route = "structured"
                    handles[shard]["files"]["structured-audit-rows.jsonl"].write(encoded)
                written[shard] += 1
                counts[shard][family] += 1
                counts[shard][route] += 1
                if scanned % 50000 == 0:
                    progress.update({"rows_scanned": scanned, "selected_rows": len(selected_ids), "updated_at": datetime.now(timezone.utc).isoformat()})
                    progress["shards"] = {str(k): {"rows": written[k], "expected": expected_rows[k]} for k in sorted(written)}
                    atomic_json(PROGRESS, progress)
    finally:
        for state in handles.values():
            for stream in state["files"].values():
                stream.close()
    if audit_digest.hexdigest() != EXPECTED_AUDIT_SHA:
        raise RuntimeError(f"audit_hash_mismatch:{audit_digest.hexdigest()}")
    if len(selected_ids) != len(wanted) - (expected_rows.get(0, 0) if 0 not in handles else 0):
        # If shard 0000 was preserved, its assigned rows are intentionally not
        # selected in this pass; otherwise every planned row must be copied.
        expected_selected = sum(expected_rows[s] for s in handles)
        if len(selected_ids) != expected_selected:
            raise RuntimeError(f"selected_count_mismatch:{len(selected_ids)} vs {expected_selected}")
    manifests = []
    for shard, state in sorted(handles.items()):
        if written[shard] != expected_rows[shard]:
            raise RuntimeError(f"shard_count_mismatch:{shard}:{written[shard]}:{expected_rows[shard]}")
        out = state["out"]
        files = state["files"]
        for name in ("audit-rows.jsonl", "completion-audit-rows.jsonl", "structured-audit-rows.jsonl"):
            partial = out / (name + ".partial")
            os.replace(partial, out / name)
        outputs = {}
        for name, count in (("audit-rows.jsonl", written[shard]), ("completion-audit-rows.jsonl", counts[shard]["completion"]), ("structured-audit-rows.jsonl", counts[shard]["structured"])):
            path = out / name
            outputs[name.replace(".jsonl", "")]= {"path": str(path), "sha256": sha(path), "rows": count}
        manifest = {
            "schema": "DAT-10-source-walk-shard-input-v1", "status": "source_walk_input_prepared_unmaterialized",
            "shard": shard, "planned_rows": expected_rows[shard], "rows_copied": written[shard],
            "family_counts": {k: v for k, v in sorted(counts[shard].items()) if k not in ("completion", "structured")},
            "route_counts": {"completion": counts[shard]["completion"], "structured": counts[shard]["structured"]},
            "inputs": {"audit": {"path": str(AUDIT), "sha256": EXPECTED_AUDIT_SHA}, "inventory": {"path": str(INVENTORY), "sha256": EXPECTED_INVENTORY_SHA}, "shard_plan": {"path": str(SHARD_PLAN), "sha256": EXPECTED_PLAN_SHA}},
            "outputs": outputs,
            "policy": {"selection": "all rows assigned to this resume shard", "family_row_time_quotas": False, "shard_is_io_unit_only": True, "train_content_only": True, "dev_final_content": False, "cpt_validation_content": False, "target_truncation": False, "admission": "none"},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(out / "manifest.json", manifest)
        manifests.append(manifest)
    progress = {"schema": "DAT-10-source-walk-preparation-progress-v1", "status": "complete", "rows_scanned": scanned, "selected_rows": len(selected_ids), "shards_prepared": sorted(handles), "updated_at": datetime.now(timezone.utc).isoformat(), "policy": "complete source audit pass; no rows dropped"}
    atomic_json(PROGRESS, progress)
    aggregate = {"schema": "DAT-10-source-walk-all-shard-inputs-v1", "status": "all_remaining_shards_prepared", "audit": {"path": str(AUDIT), "sha256": EXPECTED_AUDIT_SHA, "rows": scanned}, "inventory": {"path": str(INVENTORY), "sha256": EXPECTED_INVENTORY_SHA}, "shard_plan": {"path": str(SHARD_PLAN), "sha256": EXPECTED_PLAN_SHA}, "shards_prepared": sorted(handles), "rows_prepared": len(selected_ids), "family_counts": {str(f): sum(counts[s][f] for s in handles) for f in sorted({f for s in handles for f in counts[s] if f not in ("completion", "structured")})}, "outputs": manifests, "policy": {"all_assigned_rows": True, "family_row_time_quotas": False, "shard_is_io_unit_only": True, "heldout_content_included": False, "target_truncation": False}}
    atomic_json(SHARD_ROOT / "all-shard-inputs-manifest-v1.json", aggregate)
    print(json.dumps({"status": aggregate["status"], "rows_prepared": len(selected_ids), "shards": sorted(handles), "family_counts": aggregate["family_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
