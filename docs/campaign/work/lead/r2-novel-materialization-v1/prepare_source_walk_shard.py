#!/usr/bin/env python3
"""Prepare a quota-free DAT10 source-walk shard from the complete audit.

A shard is only an I/O/resume unit.  It is selected by the precomputed
``io_shard`` assignment in the all-row inventory, and every selected original
DAT-03 metadata row is copied to completion/structured input files.  This step
contains no prompt/target text and performs no admission.  The downstream
completion adapter can consume its completion file (<=5,000 rows by shard
construction); structured conversion must use a quota-free reviewed adapter
runner rather than the historical per-family/package-capped wrapper.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
from collections import Counter
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[5]
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
AUDIT = E / "DAT-03-row-audit.jsonl"
INVENTORY = E / "DAT10-novel-v1/all-data-inventory-v1/all-metadata-inventory-v1.jsonl"
SHARDS = E / "DAT10-novel-v1/all-data-inventory-v1/source-walk-shards-v1.jsonl"
OUT_ROOT = E / "DAT10-novel-v1/source-walk-shards-v1"
FORBIDDEN_CONTENT_KEYS = {"prompt", "prompt_text", "target", "target_text", "document_text", "full_prompt", "full_text"}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard < 0:
        raise ValueError("shard_must_be_nonnegative")
    for p in (AUDIT, INVENTORY, SHARDS):
        if not p.exists():
            raise FileNotFoundError(p)
    shard_plan = next((x for x in load_jsonl(SHARDS) if int(x["shard"]) == args.shard), None)
    if shard_plan is None:
        raise ValueError(f"unknown shard {args.shard}")
    wanted: dict[str, dict] = {}
    for row in load_jsonl(INVENTORY):
        if row.get("io_shard") == args.shard:
            wanted[str(row["row_id"])] = row
    expected = int(shard_plan["rows"])
    if len(wanted) != expected:
        raise RuntimeError(f"inventory shard count {len(wanted)} != plan {expected}")
    out = OUT_ROOT / f"shard-{args.shard:04d}"
    if out.exists():
        raise ValueError("output_shard_must_be_fresh")
    out.mkdir(parents=True)
    all_path = out / "audit-rows.jsonl"
    completion_path = out / "completion-audit-rows.jsonl"
    structured_path = out / "structured-audit-rows.jsonl"
    counts = Counter()
    seen: set[str] = set()
    with AUDIT.open(encoding="utf-8") as source, all_path.open("w", encoding="utf-8", newline="\n") as all_out, completion_path.open("w", encoding="utf-8", newline="\n") as completion_out, structured_path.open("w", encoding="utf-8", newline="\n") as structured_out:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("row_id") or "")
            if rid not in wanted:
                continue
            if rid in seen:
                raise RuntimeError(f"duplicate selected row {rid}")
            seen.add(rid)
            if FORBIDDEN_CONTENT_KEYS & set(row):
                raise RuntimeError(f"content-bearing DAT-03 key encountered: {FORBIDDEN_CONTENT_KEYS & set(row)}")
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            all_out.write(encoded)
            family = str(row.get("family") or "")
            if family.startswith("finish_block"):
                completion_out.write(encoded)
                counts["completion"] += 1
            elif family in {"roxygen_drafting", "no_op", "rename_propagation", "format_propagation", "pipe_rewrite", "na_rm_propagation"}:
                structured_out.write(encoded)
                counts["structured"] += 1
            else:
                raise RuntimeError(f"unsupported family in planned shard: {family}")
            counts[family] += 1
    if seen != set(wanted):
        raise RuntimeError(f"audit rows missing from shard: {len(set(wanted) - seen)}")
    if counts["completion"] + counts["structured"] != expected:
        raise RuntimeError("route count mismatch")
    outputs = {
        "audit_rows": {"path": str(all_path), "sha256": sha(all_path), "rows": expected},
        "completion_audit_rows": {"path": str(completion_path), "sha256": sha(completion_path), "rows": counts["completion"]},
        "structured_audit_rows": {"path": str(structured_path), "sha256": sha(structured_path), "rows": counts["structured"]},
    }
    manifest = {
        "schema": "DAT-10-source-walk-shard-input-v1",
        "status": "source_walk_input_prepared_unmaterialized",
        "shard": args.shard,
        "planned_rows": expected,
        "rows_copied": len(seen),
        "family_counts": dict(sorted((str(k), int(v)) for k, v in counts.items() if k not in ("completion", "structured"))),
        "route_counts": {"completion": counts["completion"], "structured": counts["structured"]},
        "inputs": {
            "audit": {"path": str(AUDIT), "sha256": sha(AUDIT)},
            "inventory": {"path": str(INVENTORY), "sha256": sha(INVENTORY)},
            "shard_plan": {"path": str(SHARDS), "sha256": sha(SHARDS)},
        },
        "outputs": outputs,
        "policy": {
            "selection": "all rows assigned to this resume shard",
            "family_row_time_quotas": False,
            "shard_is_io_unit_only": True,
            "train_content_only": True,
            "dev_final_content": False,
            "cpt_validation_content": False,
            "target_truncation": False,
            "admission": "none",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"shard": args.shard, "rows": expected, "routes": dict(counts), "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
