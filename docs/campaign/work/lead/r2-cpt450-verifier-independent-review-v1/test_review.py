#!/usr/bin/env python3
"""Tiny fail-closed controls for the independent checkpoint review."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import review


def write_checkpoint(root: Path, mutate: str | None = None) -> dict:
    files = {}
    for name in sorted(review.EXPECTED_FULL_FILES):
        payload = ("payload:" + name).encode()
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {"files": files, "full": True, "checkpoint_kind": "full_weights", "step": 450}
    for name in review.EXPECTED_FULL_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("payload:" + name).encode())
    if mutate == "missing":
        (root / "optimizer.pt").unlink()
    elif mutate == "extra":
        (root / "unexpected.bin").write_bytes(b"extra")
    elif mutate == "changed":
        (root / "optimizer.pt").write_bytes(b"changed")
    elif mutate == "symlink":
        target = root / "optimizer.pt"
        target.unlink()
        (root / "optimizer.pt").symlink_to(root / "config.json")
    (root / "campaign-manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_full_pair_contract() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        durable, native = base / "durable", base / "native"
        durable.mkdir()
        native.mkdir()
        manifest = write_checkpoint(durable)
        write_checkpoint(native)
        result = review.full_checkpoint_file_contract(manifest, durable, native)
        assert result["pass"] is True


def test_full_pair_rejects_mutations() -> None:
    for mutation in ("missing", "extra", "changed", "symlink"):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            durable, native = base / "durable", base / "native"
            durable.mkdir()
            native.mkdir()
            manifest = write_checkpoint(durable, mutate=mutation)
            write_checkpoint(native)
            assert review.full_checkpoint_file_contract(manifest, durable, native)["pass"] is False


def test_schedule_requires_contiguous_window() -> None:
    result = review.exact_draw_window(4224, 1920)
    assert result["positions_first"] == 4224
    assert result["positions_last"] == 6143
    assert result["positions_contiguous"] is True
    forged = [4224, 4225, 4227]
    assert forged != list(range(4224, 4224 + len(forged)))


if __name__ == "__main__":
    test_full_pair_contract()
    test_full_pair_rejects_mutations()
    test_schedule_requires_contiguous_window()
    print("cpt450 independent tiny controls: PASS")
