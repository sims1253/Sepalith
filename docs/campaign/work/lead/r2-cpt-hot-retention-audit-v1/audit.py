#!/usr/bin/env python3
"""Read-only hot-checkpoint retention audit.

This audit hashes only the named inactive ordinary-330 C/E checkpoint pair.
It never deletes, renames, writes, or opens a model from any other checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

BLOCK = 8 * 1024 * 1024

ORDINARY_C = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
    "SFT11-native-varlen-canary-322-ordinary_reference-v1/runtime/checkpoint-330"
)
ORDINARY_E = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/"
    "SFT11-native-varlen-canary-322-ordinary_reference-v1/full/checkpoint-330"
)

NAMED_PATHS = {
    "reusable_data_bundle_c": Path(
        "/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/"
        "cpt-prefix-extension-reusable-data-v1"
    ),
    "checkpoint_322_c": Path(
        "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
        "SFT11-native-CPT-prefix-extension-v3-from194/runtime/checkpoint-322"
    ),
    "ordinary_330_c": ORDINARY_C,
    "packed_330_c": Path(
        "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
        "SFT11-native-varlen-canary-322-varlen_candidate-v1/runtime/checkpoint-330"
    ),
    "ordinary_330_e": ORDINARY_E,
    "packed_330_e": Path(
        "/mnt/e/sepalith/campaign-20260915/checkpoints/"
        "SFT11-native-varlen-canary-322-varlen_candidate-v1/full/checkpoint-330"
    ),
    "future_450_e": Path(
        "/mnt/e/sepalith/campaign-20260915/checkpoints/"
        "SFT11-full-weight-CPT-production-from-selected-packed330-v1/full/checkpoint-450"
    ),
}


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(BLOCK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def fingerprint(path: Path) -> dict[str, int]:
    st = path.stat()
    return {
        "dev": st.st_dev,
        "inode": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def tree_stat(path: Path) -> dict:
    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "exists": False}
    st = path.lstat()
    if not path.is_dir():
        return {
            "path": str(path),
            "exists": True,
            "type": stat.filemode(st.st_mode),
            "symlink": path.is_symlink(),
            "dev": st.st_dev,
            "inode": st.st_ino,
            "bytes": st.st_size,
        }
    total = 0
    files = 0
    symlinks = []
    for item in path.rglob("*"):
        item_st = item.lstat()
        if item.is_symlink():
            symlinks.append(str(item))
        elif item.is_file():
            files += 1
            total += item_st.st_size
    return {
        "path": str(path),
        "exists": True,
        "type": stat.filemode(st.st_mode),
        "symlink": path.is_symlink(),
        "dev": st.st_dev,
        "inode": st.st_ino,
        "apparent_regular_file_bytes": total,
        "regular_file_count": files,
        "symlink_count": len(symlinks),
        "symlinks": symlinks[:20],
    }


def audit_pair() -> dict:
    started = time.time()
    result = {
        "ordinary_c": str(ORDINARY_C),
        "ordinary_e": str(ORDINARY_E),
        "status": "started",
    }
    if not ORDINARY_C.is_dir() or not ORDINARY_E.is_dir():
        raise RuntimeError("ordinary-330 C/E pair is not present as directories")
    if ORDINARY_C.is_symlink() or ORDINARY_E.is_symlink():
        raise RuntimeError("ordinary-330 checkpoint root is a symlink")

    c_manifest_path = ORDINARY_C / "campaign-manifest.json"
    e_manifest_path = ORDINARY_E / "campaign-manifest.json"
    c_manifest_sha, c_manifest_bytes = sha256_file(c_manifest_path)
    e_manifest_sha, e_manifest_bytes = sha256_file(e_manifest_path)
    c_manifest = json.loads(c_manifest_path.read_text(encoding="utf-8"))
    e_manifest = json.loads(e_manifest_path.read_text(encoding="utf-8"))
    if c_manifest != e_manifest:
        raise RuntimeError("ordinary-330 C and E campaign manifests differ")
    files = c_manifest.get("files")
    if not isinstance(files, dict) or len(files) != 12:
        raise RuntimeError("ordinary-330 manifest does not contain exactly 12 payload files")

    expected_names = set(files) | {"campaign-manifest.json"}
    pair = {}
    all_match = True
    for name, expected in [("campaign-manifest.json", {"sha256": c_manifest_sha, "bytes": c_manifest_bytes})] + sorted(files.items()):
        if name != "campaign-manifest.json":
            expected = {"sha256": expected["sha256"], "bytes": expected["bytes"]}
        c_path = ORDINARY_C / name
        e_path = ORDINARY_E / name
        if c_path.is_symlink() or e_path.is_symlink():
            raise RuntimeError(f"payload symlink: {name}")
        c_before = fingerprint(c_path)
        e_before = fingerprint(e_path)
        c_sha, c_bytes = sha256_file(c_path)
        e_sha, e_bytes = sha256_file(e_path)
        c_after = fingerprint(c_path)
        e_after = fingerprint(e_path)
        if c_before != c_after or e_before != e_after:
            raise RuntimeError(f"payload changed during hash: {name}")
        expected_ok = c_sha == expected["sha256"] and e_sha == expected["sha256"] and c_bytes == expected["bytes"] and e_bytes == expected["bytes"]
        pair[name] = {
            "expected_bytes": expected["bytes"],
            "c_bytes": c_bytes,
            "e_bytes": e_bytes,
            "expected_sha256": expected["sha256"],
            "c_sha256": c_sha,
            "e_sha256": e_sha,
            "match": c_sha == e_sha and c_bytes == e_bytes and expected_ok,
        }
        all_match = all_match and pair[name]["match"]
    c_names = {str(p.relative_to(ORDINARY_C)) for p in ORDINARY_C.rglob("*") if p.is_file() or p.is_symlink()}
    e_names = {str(p.relative_to(ORDINARY_E)) for p in ORDINARY_E.rglob("*") if p.is_file() or p.is_symlink()}
    if c_names != expected_names or e_names != expected_names:
        raise RuntimeError("ordinary-330 payload file set differs from the 12-file manifest closure")
    result.update(
        {
            "status": "pass" if all_match else "fail",
            "manifest_sha256": c_manifest_sha,
            "manifest_bytes": c_manifest_bytes,
            "file_count": len(pair),
            "payload_bytes": sum(v["expected_bytes"] for v in pair.values()),
            "pair": pair,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    if not all_match:
        raise RuntimeError("ordinary-330 C/E payload hash comparison failed")
    return result


def main() -> int:
    try:
        output = {
            "schema": "sepalith.sft11.cpt-hot-retention-audit.v1",
            "audit": audit_pair(),
            "named_paths": {name: tree_stat(path) for name, path in NAMED_PATHS.items()},
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # pragma: no cover - exercised by negative invocations
        print(json.dumps({"schema": "sepalith.sft11.cpt-hot-retention-audit.v1", "status": "error", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
