#!/usr/bin/env python3
"""Independent target-free pre-edit/source reapplication verification."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import argparse


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(folder: Path, manifest: dict, name: str) -> dict[str, dict]:
    path = folder / name
    raw = path.read_bytes()
    expected = manifest["outputs"][name]
    assert sha_bytes(raw) == expected["sha256"], path
    assert len(raw) == expected["bytes"], path
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == expected["rows"], path
    result = {row["row_id"]: row for row in rows}
    assert len(result) == len(rows), path
    return result


def verify_shard(root: Path, shard: int) -> dict:
    folder = root / f"shard-{shard:04d}"
    manifest = json.loads((folder / "preparation-manifest.json").read_text())
    predictions = load_rows(folder, manifest, "prediction-inputs.jsonl")
    targets = load_rows(folder, manifest, "training-sidecar.jsonl")
    holds = load_rows(folder, manifest, "preparation-holds.jsonl")
    assert set(predictions) == set(targets)
    assert not set(predictions) & set(holds)
    assert len(predictions) + len(holds) == manifest["semantic_supported"]
    eols = collections.Counter()
    character_buckets = collections.Counter()
    nonascii = 0
    for rid, prediction in predictions.items():
        expected_prediction_keys = {
            "schema", "row_id", "path", "preedit_text", "preedit_sha256", "cursor",
            "document_eol", "required_helper_names", "external_import_dependencies",
        }
        assert set(prediction) == expected_prediction_keys
        assert prediction["schema"] == "sepalith.dat10.semantic763.prediction_input.v1"
        before = prediction["preedit_text"]
        assert sha_bytes(before.encode()) == prediction["preedit_sha256"]
        assert prediction["document_eol"] in ("lf", "crlf")
        separator = "\r\n" if prediction["document_eol"] == "crlf" else "\n"
        lines = before.split(separator)
        cursor = prediction["cursor"]
        assert cursor["character"] == 0 and 0 <= cursor["line"] < len(lines)
        assert lines[cursor["line"]] == ""
        target = targets[rid]
        assert target["schema"] == "sepalith.dat10.semantic763.training_sidecar.v1"
        target_lines = target["target_lines"]
        assert target_lines and all(
            isinstance(line, str) and "\r" not in line and "\n" not in line for line in target_lines
        )
        target_bytes = ("\n".join(target_lines) + "\n").encode()
        assert sha_bytes(target_bytes) == target["target_sha256"]
        lines[cursor["line"]] = separator.join(target_lines)
        assert sha_bytes(separator.join(lines).encode()) == target["postedit_source_sha256"]
        assert target["postedit_source_sha256"] == target["identity"]["source_sha256"]
        assert target["identity"]["split"] == "train_group"
        assert isinstance(target["required_helper_spans"], list)
        character_buckets[min(len(before) // 10000, 999)] += 1
        eols[prediction["document_eol"]] += 1
        nonascii += not before.isascii()
    for item in holds.values():
        assert item["silent_drop"] is False
        assert isinstance(item.get("reason"), str) and item["reason"]
    return {
        "shard": shard,
        "prepared_rows": len(predictions),
        "hold_rows": len(holds),
        "eols": dict(eols),
        "nonascii_preedits": nonascii,
        "character_length_buckets_10k": dict(character_buckets),
        "prediction_target_free": True,
        "full_target_sidecar": True,
        "exact_source_reapplication": True,
        "training_admission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    reports = [verify_shard(args.root, shard) for shard in range(12, 27)]
    hold_reasons = collections.Counter()
    for shard in range(12, 27):
        path = args.root / f"shard-{shard:04d}/preparation-holds.jsonl"
        with path.open() as stream:
            for line in stream:
                hold_reasons[json.loads(line)["reason"]] += 1
    result = {
        "schema": "sepalith.dat10.semantic10952.independent_geometry_verify.v1",
        "shards": reports,
        "prepared_rows": sum(item["prepared_rows"] for item in reports),
        "hold_rows": sum(item["hold_rows"] for item in reports),
        "status": "pass",
        "hold_reason_counts": dict(hold_reasons),
        "prediction_target_free": True,
        "full_target_sidecar": True,
        "exact_source_reapplication": True,
        "geometry_basis": "cursor/EOL/full-preedit reapplication; no profile sidecar is emitted by the reviewed preparer",
        "training_admission": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
