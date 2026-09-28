#!/usr/bin/env python3
"""Audit document identities after the main CPT and alias passes finish.

The audit reads document metadata only.  It does not read cpt_train payload
JSONL, admit rows, or rewrite either producer output.  It is deliberately
run after the main materializer reaches all 8,092 scheduled groups and after
the alias recovery has been rerun against that terminal main snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MAIN = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1")
ALIAS = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-license-alias-recovery-v1")
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
BASE = PLAN / "docs/campaign/work/r2-corpus-preparation-v1"
GLOBAL = PLAN / "docs/campaign/work/r2-cpt-global-shard-v1"
EXPECTED_GROUPS = 8092


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def add_document(seen: dict[str, str], duplicate_rows: list[dict[str, str]], row: dict[str, Any], origin: str) -> None:
    identity = row.get("sha256") or row.get("document_id")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError(f"document_identity_missing:{origin}")
    prior = seen.get(identity)
    if prior is not None:
        duplicate_rows.append({"sha256": identity, "first_origin": prior, "duplicate_origin": origin})
    else:
        seen[identity] = origin


def load_base(seen: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    paths = [BASE / "broader-shard-v1-2k/documents.jsonl", GLOBAL / "shard/documents.jsonl"]
    profile = BASE / "profile-shard-v1/documents.jsonl"
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                add_document(seen, [], row, f"base:{path.name}")
                counts["base_documents"] += 1
    with profile.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("cpt_partition") == "cpt_validation":
                add_document(seen, [], row, "reserved:cpt_validation")
                counts["reserved_validation_documents"] += 1
    return counts


def load_main(seen: dict[str, str], duplicate_rows: list[dict[str, str]]) -> Counter[str]:
    progress = json.loads((MAIN / "progress.json").read_text(encoding="utf-8"))
    if progress.get("groups_committed") != EXPECTED_GROUPS or progress.get("groups_remaining") != 0:
        raise ValueError("main_materializer_not_terminal")
    if progress.get("status") not in {"complete", "all_groups_inventoried_repairs_pending"}:
        raise ValueError("main_materializer_status_not_terminal")
    counts: Counter[str] = Counter()
    for folder in sorted((MAIN / "groups").iterdir()):
        if not folder.is_dir() or not folder.name[:6].isdigit():
            continue
        receipt_path = folder / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        docs_path = folder / "documents.jsonl"
        expected = receipt.get("artifacts", {}).get("documents.jsonl", {})
        if docs_path.stat().st_size != expected.get("bytes") or sha_file(docs_path) != expected.get("sha256"):
            raise ValueError(f"main_documents_artifact_changed:{folder.name}")
        with docs_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                add_document(seen, duplicate_rows, json.loads(line), f"main:{folder.name}:{line_number}")
                counts["main_documents"] += 1
    return counts


def load_alias(seen: dict[str, str], duplicate_rows: list[dict[str, str]]) -> Counter[str]:
    progress_path = ALIAS / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("captured_main_groups") != EXPECTED_GROUPS:
        raise ValueError("alias_rescan_did_not_capture_terminal_main")
    counts: Counter[str] = Counter()
    for folder in sorted((ALIAS / "groups").iterdir()):
        if not folder.is_dir():
            continue
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        docs_path = folder / "documents.jsonl"
        expected = receipt.get("artifacts", {}).get("documents.jsonl", {})
        if docs_path.stat().st_size != expected.get("bytes") or sha_file(docs_path) != expected.get("sha256"):
            raise ValueError(f"alias_documents_artifact_changed:{folder.name}")
        with docs_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row.get("provisional_pending_terminal_main_dedup") is not True:
                    raise ValueError(f"alias_row_not_provisional:{folder.name}:{line_number}")
                add_document(seen, duplicate_rows, row, f"alias:{folder.name}:{line_number}")
                counts["alias_documents"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output_must_be_fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    duplicate_rows: list[dict[str, str]] = []
    counts = load_base(seen)
    counts.update(load_main(seen, duplicate_rows))
    counts.update(load_alias(seen, duplicate_rows))
    with (args.output / "duplicate-ledger.jsonl").open("x", encoding="utf-8") as stream:
        for row in duplicate_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    main_progress = MAIN / "progress.json"
    alias_progress = ALIAS / "progress.json"
    result = {
        "schema": "sepalith.dat10.cpt_terminal_global_dedup_audit.v1",
        "status": "pass_pending_root_admission" if not duplicate_rows else "duplicates_require_root_resolution",
        "main_progress": {"path": str(main_progress), "sha256": sha_file(main_progress)},
        "alias_progress": {"path": str(alias_progress), "sha256": sha_file(alias_progress)},
        "counts": dict(counts),
        "unique_document_identities": len(seen),
        "duplicate_records": len(duplicate_rows),
        "duplicate_ledger": str(args.output / "duplicate-ledger.jsonl"),
        "duplicate_ledger_sha256": sha_file(args.output / "duplicate-ledger.jsonl"),
        "payloads_read": False,
        "training_admission": False,
        "heldout_content_read": False,
    }
    write_json(args.output / "global-dedup-audit.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
