#!/usr/bin/env python3
"""Describe the immutable code, recipe, and streaming cache needed beside checkpoint90."""
from __future__ import annotations

import json
from pathlib import Path

from transport_common import atomic_json, require, sha256

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
CACHE = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-streaming-input-v1/final-union-cache")
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-representative-lr3e-6-v1/full/checkpoint-66")
PACKET = PLAN / "docs/campaign/work/lead/r2-a100-fullweight-trainer-v1"
RECIPE = PLAN / "docs/campaign/work/lead/r2-cpt-full-corpus-expandable-pilot-root-v1/bound-recipe.json"
CACHE_MANIFEST_SHA = "ac17fe1ced73efe37767bf28619b47c120f57be121c4648a1bf5fb0c426084ec"
SOURCE_MANIFEST_SHA = "5b66cab643b55259814c2dbfb9252dec391c563cc29099d5ce578c2c14483698"
PARENT_MANIFEST_SHA = "6ae1d9cef615d179d4e55b7da7f68cc67b92239ffce5daeb8852c547c6dfc15d"
PARENT_LOAD_FILES = {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}


def main() -> None:
    cache_manifest_path = CACHE / "manifest.json"
    source_manifest_path = PACKET / "source-manifest.json"
    require(sha256(cache_manifest_path) == CACHE_MANIFEST_SHA, "cache manifest pin differs")
    require(sha256(source_manifest_path) == SOURCE_MANIFEST_SHA, "source manifest pin differs")
    cache_manifest = json.loads(cache_manifest_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    parent_manifest_path = PARENT / "campaign-manifest.json"
    require(sha256(parent_manifest_path) == PARENT_MANIFEST_SHA, "parent bootstrap manifest pin differs")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    files: dict[str, dict] = {}
    for name, row in sorted(cache_manifest["files"].items()):
        p = CACHE / name
        st = p.stat()
        require(st.st_size == row["bytes"], f"cache file size differs:{name}")
        files[f"cache/{name}"] = {
            "source_path": str(p), "bytes": st.st_size, "sha256": row["sha256"],
            "source_mode": st.st_mode & 0o777, "restore_mode": 0o600,
        }
    st = cache_manifest_path.stat()
    files["cache/manifest.json"] = {
        "source_path": str(cache_manifest_path), "bytes": st.st_size,
        "sha256": CACHE_MANIFEST_SHA, "source_mode": st.st_mode & 0o777, "restore_mode": 0o600,
    }
    for row in source_manifest["files"]:
        relative = row["path"].removeprefix("source/")
        p = PACKET / row["path"]
        require(p.stat().st_size == row["bytes"] and sha256(p) == row["sha256"], f"source pin differs:{relative}")
        files[f"source/{relative}"] = {
            "source_path": str(p), "bytes": row["bytes"], "sha256": row["sha256"],
            "source_mode": p.stat().st_mode & 0o777, "restore_mode": 0o600,
        }
    for name in sorted(PARENT_LOAD_FILES):
        row = parent_manifest["files"][name]
        p = PARENT / name
        require(p.stat().st_size == row["bytes"], f"parent bootstrap size differs:{name}")
        files[f"parent-checkpoint-66/{name}"] = {
            "source_path": str(p), "bytes": row["bytes"], "sha256": row["sha256"],
            "source_mode": p.stat().st_mode & 0o777, "restore_mode": 0o600,
        }
    st = parent_manifest_path.stat()
    files["parent-checkpoint-66/campaign-manifest.json"] = {
        "source_path": str(parent_manifest_path), "bytes": st.st_size,
        "sha256": PARENT_MANIFEST_SHA, "source_mode": st.st_mode & 0o777, "restore_mode": 0o600,
    }
    controls = {
        "controls/source-manifest.json": (source_manifest_path, SOURCE_MANIFEST_SHA),
        "controls/bound-recipe.json": (RECIPE, sha256(RECIPE)),
        "controls/a100-recipe.template.json": (PACKET / "recipe.template.json", sha256(PACKET / "recipe.template.json")),
        "controls/a100-findings.json": (PACKET / "findings.json", sha256(PACKET / "findings.json")),
    }
    for remote, (p, digest) in controls.items():
        files[remote] = {
            "source_path": str(p), "bytes": p.stat().st_size, "sha256": digest,
            "source_mode": p.stat().st_mode & 0o777, "restore_mode": 0o600,
        }
    result = {
        "schema": "sepalith.pre04.full_weight_runtime_bundle_spec.v1",
        "status": "prepared_upload_not_authorized",
        "repo": "scholzmx/sepalith-lora",
        "prefix": f"full-weight-runtime/{CACHE_MANIFEST_SHA[:24]}",
        "cache_manifest_sha256": CACHE_MANIFEST_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA,
        "recipe_sha256": sha256(RECIPE),
        "files": files,
        "total_bytes": sum(x["bytes"] for x in files.values()),
        "cache_files": 5,
        "source_files": len(source_manifest["files"]),
        "control_files": 4,
        "parent_bootstrap_files": 7,
        "resume_world_size_required": 1,
    }
    atomic_json(Path(__file__).with_name("runtime-bundle-spec.json"), result)


if __name__ == "__main__":
    main()
