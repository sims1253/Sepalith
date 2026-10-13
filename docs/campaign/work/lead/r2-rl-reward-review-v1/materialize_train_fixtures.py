#!/usr/bin/env python3
"""Extract four pinned TRAIN-only reward fixtures from the frozen RL inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


SIDECAR = Path("docs/campaign/work/corrected-rl-admission-audit-v1/candidate-data/context-sidecar.jsonl")
ROWS = Path("docs/campaign/work/corrected-rl-admission-audit-v1/candidate-data/eligible-train-rows.jsonl")
PINS = {
    "sidecar": "265b80762efc9544230f1aba09e492906760e98a34431593fdeb4813d6e169ac",
    "rows": "7e9cf35e8ecbf1af151df78bfc42c07d7b1d24f6ab82ec47770463d967fe9465",
}
IDS = {
    "complete_edit": "000a6aaa48aee4291dbcb0cb",
    "no_op": "0003ea6fc6a4d0b3b354efba",
    "roxygen_window": "0002afb77102acedfe97c1b9",
    "finish_prefix": "2b578e5b4936158eaab12ee3",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def utf16_index(line: str, units: int) -> int:
    used = 0
    for index, char in enumerate(line):
        if used == units:
            return index
        used += len(char.encode("utf-16-le")) // 2
    require(used == units, "UTF-16 position differs")
    return len(line)


def offset(text: str, position: dict) -> int:
    lines = text.splitlines(keepends=True)
    line = position["line"]
    return sum(map(len, lines[:line])) + utf16_index(lines[line].rstrip("\r\n"), position["character"])


def source_after_history(sidecar: dict) -> str:
    provenance = sidecar["source_identity"]["source_provenance"]
    source_path = Path(provenance["source_snapshot_path"])
    raw = source_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == provenance["source_snapshot_sha256"], "TRAIN source snapshot differs")
    text = raw.decode("utf-8")
    eol = "\r\n" if sidecar["context"]["document_eol"] == "crlf" else "\n"
    for event in sidecar["context"]["history"]:
        rr = event["range_utf16"]
        require(hashlib.sha256(text.encode()).hexdigest() == rr["content_sha256"], "TRAIN history predecessor differs")
        left, right = offset(text, rr["start"]), offset(text, rr["end"])
        require(text[left:right] == event["old_text"].replace("\n", eol), "TRAIN history old text differs")
        text = text[:left] + event["new_text"].replace("\n", eol) + text[right:]
    return text


def main() -> None:
    output = Path(sys.argv[1])
    require(not output.exists(), "fixture output must be fresh")
    require(sha(SIDECAR) == PINS["sidecar"] and sha(ROWS) == PINS["rows"], "frozen TRAIN input differs")
    selected = set(IDS.values())
    sidecars = {}
    for line in SIDECAR.open():
        value = json.loads(line)
        if value["row_id"] in selected:
            sidecars[value["row_id"]] = value
    rows = {}
    for line in ROWS.open():
        value = json.loads(line)
        if value["id"] in selected:
            rows[value["id"]] = value
    require(set(sidecars) == selected and set(rows) == selected, "TRAIN fixture IDs missing")
    fixtures = {}
    for role, row_id in IDS.items():
        sidecar, row = sidecars[row_id], rows[row_id]
        require(sidecar["split"] == row["split"] == "train", "non-TRAIN fixture refused")
        context = sidecar["context"]
        baseline = None
        buffer_mode = "unverified"
        if role == "complete_edit":
            baseline = source_after_history(sidecar)
            buffer_mode = "complete_document"
        elif role == "finish_prefix":
            provenance = sidecar["source_identity"]["source_provenance"]
            baseline = provenance["selection_source"]["document_text"]
            buffer_mode = "completion_prefix"
        if baseline is not None:
            require(hashlib.sha256(baseline.encode()).hexdigest() == context["replacement_range"]["content_sha256"],
                    f"{role}: prepared baseline hash differs")
        fixtures[role] = {
            "id": row_id, "split": "train", "family": row["family"], "package_id": row["package_id"],
            "target_operation": row["target_operation"], "target_body_text": row["target_body_text"],
            "context": context, "selection_geometry": sidecar["selection_geometry"],
            "source_ref": sidecar["source_identity"]["source_ref"],
            "buffer_mode": buffer_mode, "baseline_text": baseline,
        }
    output.mkdir()
    (output / "train-fixtures.json").write_text(json.dumps({
        "schema": "sepalith.rl11.train-reward-fixtures.v1", "split": "train",
        "source_pins": PINS, "fixtures": fixtures,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
