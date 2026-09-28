#!/usr/bin/env python3
"""Token-audit and freeze one completed DAT10 source-walk shard."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
try:
    os.sched_setaffinity(0, {0, 1})
except (AttributeError, OSError):
    pass

ROOT = pathlib.Path(__file__).resolve().parents[5]
EXEC = pathlib.Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
TOKEN_AUDIT = EXEC / "experiments/training/campaign_token_audit.py"
TOKENIZER = pathlib.Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work")
SHARDS = E / "DAT10-novel-v1/source-walk-shards-v1"
PRIOR_REVIEW = E / "DAT10-novel-v1/expansion-increment-na-rm-v1/root-review-packet-v4-na-rm.jsonl"


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: pathlib.Path, values: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    shard = SHARDS / f"shard-{args.shard:04d}" / "structured-materialization-v1"
    packet_path = shard / "candidate-packets.jsonl"
    material_manifest = shard / "manifest.json"
    if not packet_path.exists() or not material_manifest.exists():
        raise FileNotFoundError(packet_path)
    if not PRIOR_REVIEW.exists():
        raise FileNotFoundError(PRIOR_REVIEW)
    material = json.loads(material_manifest.read_text(encoding="utf-8"))
    if material.get("status") != "frozen_review_increment_unadmitted":
        raise RuntimeError("source materialization is not frozen")
    packets = list(jsonl(packet_path))
    if len(packets) != int(material["outputs"]["candidate_packets"]["rows"]):
        raise RuntimeError("candidate packet count differs from material manifest")
    if sha(packet_path) != material["outputs"]["candidate_packets"]["sha256"]:
        raise RuntimeError("candidate packet hash differs from material manifest")
    ids = [str(p.get("row_ref", {}).get("row_id")) for p in packets]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate source-walk candidate identity")
    if any(p.get("row_ref", {}).get("split") != "train_group" for p in packets):
        raise RuntimeError("non-train source-walk candidate")

    token_input = shard / "token-input.jsonl"
    with token_input.with_name(token_input.name + ".partial").open("w", encoding="utf-8", newline="\n") as f:
        for p in packets:
            ref = dict(p["row_ref"])
            ref["file"] = ref.get("file") or ref.get("source_file")
            ref["line"] = ref.get("line") or ref.get("source_line")
            ref["package_id"] = ref.get("package_id") or ref.get("package")
            normalized = dict(p)
            normalized["row_ref"] = ref
            f.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(token_input.with_name(token_input.name + ".partial"), token_input)
    input_sha = sha(token_input)
    token_dir = shard / "token-audit"
    if token_dir.exists():
        raise ValueError("token audit output must be fresh")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EXEC / "packages/sepalith/src")
    command = [
        sys.executable, str(TOKEN_AUDIT), "--input-jsonl", str(token_input),
        "--input-sha256", input_sha, "--tokenizer", str(TOKENIZER),
        "--output", str(token_dir),
    ]
    completed = subprocess.run(command, env=env, check=False, text=True, capture_output=True)
    (shard / "token-audit.stdout.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"token audit failed ({completed.returncode}): {completed.stdout[-1000:]}{completed.stderr[-1000:]}")
    report_path = token_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    token_rows_path = token_dir / "candidate-token-rows.jsonl"
    token_rows = list(jsonl(token_rows_path))
    if len(token_rows) != len(packets) or report.get("excluded"):
        raise RuntimeError("token audit did not retain every converted source row")
    lengths = Counter()
    long_rows: list[dict[str, Any]] = []
    for item in token_rows:
        l = item["lengths"]
        response = int(l["response_with_terminal_eos"])
        sequence = int(l["sequence"])
        for limit in (192, 512, 1024):
            if response <= limit:
                lengths[f"response_le_{limit}"] += 1
        if response > 1024:
            lengths["response_gt_1024"] += 1
        for limit in (2048, 4096):
            if sequence <= limit:
                lengths[f"sequence_le_{limit}"] += 1
        if sequence > 4096:
            lengths["sequence_gt_4096"] += 1
        if response > 1024 or sequence > 4096:
            ref = item["source_ref"]
            long_rows.append({
                "row_id": ref["row_id"], "group_id": ref.get("group_id"),
                "family": item["row"]["family"], "source": ref.get("source"),
                "source_file": ref.get("file"), "source_line": ref.get("line"),
                "package_id": ref.get("package_id"), "split": ref.get("split"),
                "prompt_sha256": item.get("prompt_sha256"), "target_sha256": item.get("target_sha256"),
                "prompt_tokens_with_bos": l.get("prompt_with_bos"),
                "target_tokens_including_eos": response, "sequence_tokens": sequence,
                "long_context_reasons": [x for x, ok in (("target_over_1024", response > 1024), ("sequence_over_4096", sequence > 4096)) if ok],
                "status": "long_context_queue_review_only",
            })
    long_path = shard / "long-context-queue.jsonl"
    atomic_jsonl(long_path, long_rows)
    summary = {
        "schema": "DAT-10-source-walk-token-audit-v1",
        "status": "frozen_review_increment_unadmitted",
        "shard": args.shard, "source_converted_rows": len(packets),
        "tokenizer_candidates": len(token_rows), "tokenizer_excluded": len(report.get("excluded", [])),
        "length_buckets": dict(sorted(lengths.items())),
        "max_target_tokens_including_eos": max(int(x["lengths"]["response_with_terminal_eos"]) for x in token_rows) if token_rows else None,
        "max_sequence_tokens": max(int(x["lengths"]["sequence"]) for x in token_rows) if token_rows else None,
        "long_context_rows": len(long_rows),
        "long_context_queue": {"path": str(long_path), "sha256": sha(long_path), "rows": len(long_rows)},
        "token_input": {"path": str(token_input), "sha256": input_sha, "rows": len(packets)},
        "token_report": {"path": str(report_path), "sha256": sha(report_path)},
        "token_rows": {"path": str(token_rows_path), "sha256": sha(token_rows_path), "rows": len(token_rows)},
        "target_truncation": False, "heldout_content_included": False, "cuda_started": False,
        "admission": "none; root quality/source/license/duplicate/CPT review remains mandatory",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = shard / "token-audit-summary.json"
    atomic_json(summary_path, summary)

    prior = list(jsonl(PRIOR_REVIEW))
    prior_ids = {str(p.get("row", {}).get("id") or p.get("source_ref", {}).get("row_id")) for p in prior}
    if prior_ids & set(ids):
        raise RuntimeError("source-walk candidate overlaps frozen prior review packet")
    review = shard / "root-review-packet-v5-shard0000.jsonl"
    atomic_jsonl(review, prior + token_rows)
    manifest = {
        "schema": "DAT-10-source-walk-token-audit-manifest-v1",
        "status": "frozen_review_increment_unadmitted",
        "shard": args.shard,
        "source_material_manifest": {"path": str(material_manifest), "sha256": sha(material_manifest)},
        "token_summary": {"path": str(summary_path), "sha256": sha(summary_path)},
        "token_input": {"path": str(token_input), "sha256": input_sha, "rows": len(packets)},
        "token_rows": {"path": str(token_rows_path), "sha256": sha(token_rows_path), "rows": len(token_rows)},
        "long_context_queue": {"path": str(long_path), "sha256": sha(long_path), "rows": len(long_rows)},
        "root_review_packet": {"path": str(review), "sha256": sha(review), "rows": len(prior) + len(token_rows)},
        "length_buckets": dict(sorted(lengths.items())),
        "policy": {
            "all_source_converted_rows_token_audited": True,
            "target_truncation": False, "long_targets_preserved": True,
            "long_context_queue_is_not_exclusion": True,
            "family_row_time_quotas": False, "heldout_content_included": False,
            "admission": "none",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = shard / "token-audit-manifest.json"
    atomic_json(manifest_path, manifest)
    progress_path = SHARDS / "source-walk-progress-v1.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"schema": "DAT-10-source-walk-progress-v1"}
    progress.update({"status": "complete_token_audited", "current_shard": args.shard, "updated_at": datetime.now(timezone.utc).isoformat(), "shard": {**progress.get("shard", {}), "tokenized": True, "token_audit_manifest": str(manifest_path), "token_audit_manifest_sha256": sha(manifest_path)}})
    atomic_json(progress_path, progress)
    print(json.dumps({"shard": args.shard, "source_rows": len(packets), "token_rows": len(token_rows), "long_context_rows": len(long_rows), "review_rows": len(prior) + len(token_rows), "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
