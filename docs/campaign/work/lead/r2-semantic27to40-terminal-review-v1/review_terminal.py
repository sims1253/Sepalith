#!/usr/bin/env python3
"""Independent, read-only terminal review for semantic shards 27 through 40."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = list(range(27, 41))
DEFAULT_ROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Sourcewalk-semantic-streaming-queue-shards27to40-v1"
)
DEFAULT_TERMINAL = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/"
    "work/lead/r2-semantic27to40-resume-root-v1/terminal.json"
)


class ReviewError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ReviewError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_ids(ids: list[str]) -> str:
    encoded = json.dumps(
        sorted(ids), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checked_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing_file:{path}")
    digest = sha256(path)
    if expected is not None:
        require(digest == expected, f"sha256_mismatch:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def run(root: Path, terminal_path: Path) -> dict[str, Any]:
    terminal = json.loads(terminal_path.read_text())
    require(terminal.get("exit_code") == 0, "producer_terminal_not_exit_zero")
    require(terminal.get("training_admission") is False, "producer_terminal_admission")

    top_path = root / "streaming-manifest.json"
    top_pin = checked_file(top_path)
    top = json.loads(top_path.read_text())
    require(top.get("schema") == "sepalith.dat10.sourcewalk_semantic_streaming_queue.v2", "top_schema")
    require(top.get("status") == "partial_review_only", "top_status")
    require(top.get("training_admission") is False, "top_admission")
    require(top.get("requested_shards") == EXPECTED, "top_requested_shards")
    require(top.get("source_pool_closed") is False, "source_pool_closed_claim")

    shard_entries = top.get("shards")
    require(isinstance(shard_entries, list) and len(shard_entries) == 14, "top_shard_count")
    by_shard = {entry.get("shard"): entry for entry in shard_entries}
    require(sorted(by_shard) == EXPECTED and len(by_shard) == 14, "top_shard_identity")

    visible_dirs = sorted(
        int(path.name.rsplit("-", 1)[1])
        for path in root.glob("shard-*")
        if path.is_dir()
    )
    require(visible_dirs == EXPECTED, "physical_shard_directories")
    quarantined_partials = sorted(
        str(path)
        for path in root.rglob("*")
        if "quarantine" in path.parts
        and (
            path.name.startswith(".")
            or path.name.endswith((".tmp", ".partial", ".incomplete"))
        )
    )
    active_partials = sorted(
        str(path)
        for path in root.rglob("*")
        if "quarantine" not in path.parts
        and (
            path.name.startswith(".")
            or path.name.endswith((".tmp", ".partial", ".incomplete"))
        )
    )
    require(not active_partials, "active_partial_artifacts_present")

    global_ids: set[str] = set()
    all_statuses: collections.Counter[str] = collections.Counter()
    all_reasons: collections.Counter[str] = collections.Counter()
    reviewed: list[dict[str, Any]] = []
    for shard in EXPECTED:
        entry = by_shard[shard]
        child = root / f"shard-{shard:04d}"
        manifest_path = child / "manifest.json"
        output_path = child / "semantic-ledger.jsonl"
        manifest_pin = checked_file(manifest_path, entry.get("manifest_sha256"))
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get("schema") == "sepalith.dat10.sourcewalk_roxy_semantic.v6", f"manifest_schema:{shard}")
        require(manifest.get("status") == "complete_review_only", f"manifest_status:{shard}")
        require(manifest.get("training_admission") is False, f"manifest_admission:{shard}")
        require(manifest.get("exact_id_closure") is True, f"manifest_id_closure:{shard}")
        require(manifest.get("output") == manifest.get("output_binding"), f"output_binding:{shard}")

        output_decl = manifest["output"]
        require(output_decl.get("path") == str(output_path), f"output_path:{shard}")
        output_pin = checked_file(output_path, output_decl.get("sha256"))
        require(output_pin["bytes"] == output_decl.get("bytes") == entry.get("semantic_output_bytes"), f"output_bytes:{shard}")

        ids: list[str] = []
        statuses: collections.Counter[str] = collections.Counter()
        reasons: collections.Counter[str] = collections.Counter()
        with output_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                require(bool(line.strip()), f"blank_output_line:{shard}:{line_number}")
                row = json.loads(line)
                row_id = row.get("row_id")
                status = row.get("status")
                require(isinstance(row_id, str) and row_id, f"bad_row_id:{shard}:{line_number}")
                require(isinstance(status, str) and status, f"bad_status:{shard}:{line_number}")
                ids.append(row_id)
                statuses[status] += 1
                row_reasons = row.get("reasons", [])
                require(isinstance(row_reasons, list), f"bad_reasons:{shard}:{line_number}")
                for reason in row_reasons:
                    require(isinstance(reason, str) and reason, f"bad_reason:{shard}:{line_number}")
                    reasons[reason] += 1
        require(len(ids) == len(set(ids)), f"duplicate_id_within_shard:{shard}")
        overlap = global_ids.intersection(ids)
        require(not overlap, f"duplicate_id_across_shards:{shard}:{next(iter(overlap), '')}")
        global_ids.update(ids)
        require(len(ids) == manifest.get("rows") == output_decl.get("rows") == entry.get("rows") == entry.get("semantic_output_rows"), f"row_count:{shard}")
        ids_digest = canonical_ids(ids)
        require(ids_digest == manifest.get("queued_ids_sha256"), f"manifest_queued_ids:{shard}")
        require(ids_digest == output_decl.get("row_ids_sha256"), f"output_ids:{shard}")
        require(ids_digest == entry.get("queued_ids_sha256") == entry.get("semantic_output_row_ids_sha256"), f"top_ids:{shard}")
        require(dict(statuses) == manifest.get("status_counts"), f"status_counts:{shard}")
        require(dict(reasons) == manifest.get("reason_counts"), f"reason_counts:{shard}")

        inputs: list[dict[str, Any]] = []
        for candidate in manifest.get("inputs", {}).get("candidate_packets", []):
            inputs.append(checked_file(Path(candidate["path"]), candidate.get("sha256")))
        provenance = manifest.get("inputs", {}).get("provenance_ledger", {})
        provenance_pin = checked_file(Path(provenance["path"]), provenance.get("sha256"))
        require(provenance_pin["sha256"] == entry.get("provenance_ledger_sha256"), f"top_provenance:{shard}")

        all_statuses.update(statuses)
        all_reasons.update(reasons)
        reviewed.append(
            {
                "shard": shard,
                "execution": entry.get("status"),
                "rows": len(ids),
                "status_counts": dict(statuses),
                "manifest": manifest_pin,
                "semantic_output": {**output_pin, "row_ids_sha256": ids_digest},
                "candidate_packets": inputs,
                "provenance_ledger": provenance_pin,
                "source_binding_sha256": manifest.get("source_provenance_binding_sha256"),
            }
        )

    semantic_rows = sum(item["rows"] for item in reviewed)
    require(semantic_rows == top.get("semantic_rows") == len(global_ids), "top_semantic_rows")
    provenance_rows = top.get("provenance_rows")
    require(isinstance(provenance_rows, int) and provenance_rows >= semantic_rows, "top_provenance_rows")
    reused = [item["shard"] for item in reviewed if item["execution"] == "reused"]
    computed = [item["shard"] for item in reviewed if item["execution"] == "computed"]
    require(reused == list(range(27, 35)), "reused_scope")
    require(computed == list(range(35, 41)), "computed_scope")

    return {
        "schema": "sepalith.dat10.semantic27to40_terminal_review.v1",
        "status": "verified_partial_review_only_not_training_admitted",
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "producer_terminal": checked_file(terminal_path),
        "streaming_manifest": top_pin,
        "requested_shards": EXPECTED,
        "reused_shards": reused,
        "computed_shards": computed,
        "shard_count": len(reviewed),
        "provenance_rows": provenance_rows,
        "semantic_rows": semantic_rows,
        "provenance_not_emitted_as_semantic_rows": provenance_rows - semantic_rows,
        "unique_semantic_row_ids": len(global_ids),
        "status_counts": dict(all_statuses),
        "reason_counts": dict(all_reasons),
        "partial_artifacts_counted": 0,
        "active_partial_artifacts": active_partials,
        "quarantined_partial_artifacts": quarantined_partials,
        "training_admission": False,
        "source_pool_closed": False,
        "closure": top.get("closure"),
        "shards": reviewed,
        "next": "Root-reviewed rendering/materialization of semantic_supported_context_closure_root_review_required rows using the pinned provider/renderer, with hold rows retained and all target/provenance joins revalidated. This review does not authorize that render or training.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.terminal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "shard_count", "provenance_rows", "semantic_rows",
        "provenance_not_emitted_as_semantic_rows", "status_counts"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
