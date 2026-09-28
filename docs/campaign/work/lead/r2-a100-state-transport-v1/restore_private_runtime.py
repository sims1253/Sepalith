#!/usr/bin/env python3
"""Restore a pinned runtime closure into a fresh directory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from transport_common import atomic_json, require, safe_nested, sha256


def client(repo):
    from huggingface_hub import HfApi, get_token, hf_hub_download
    token = get_token()
    require(bool(token), "cached private token unavailable")
    api = HfApi(token=token)
    require(api.repo_info(repo, repo_type="model").private is True, "repository is not private")
    return api, hf_hub_download, token


def copy_verified(source, target, expected):
    import hashlib
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    count = 0
    with Path(source).open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(8 << 20), b""):
            dst.write(block); h.update(block); count += len(block)
        dst.flush(); os.fsync(dst.fileno())
    require(count == expected["bytes"] and h.hexdigest() == expected["sha256"], "runtime restored file differs:" + str(target))
    os.chmod(target, expected["restore_mode"])


def run(repo, prefix, revision, closure_sha, destination, *, api=None, download=None, token=None):
    destination = Path(destination)
    require(not destination.exists(), "fresh runtime destination required")
    if api is None:
        api, download, token = client(repo)
    closure_file = Path(download(repo, filename=prefix + "/closure.json", revision=revision, repo_type="model", token=token))
    require(sha256(closure_file) == closure_sha, "runtime closure hash differs")
    closure = json.loads(closure_file.read_text())
    require(closure.get("schema") == "sepalith.pre04.full_weight_runtime_closure.v1" and closure.get("status") == "complete" and closure.get("repo") == repo and closure.get("prefix") == prefix and closure.get("resume_world_size_required") == 1, "runtime closure identity differs")
    files = closure.get("files")
    require(isinstance(files, dict) and files and all(safe_nested(x) for x in files), "runtime closure paths unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=destination.parent))
    try:
        for name, expected in sorted(files.items()):
            cached = download(repo, filename=prefix + "/files/" + name, revision=revision, repo_type="model", token=token)
            copy_verified(cached, temporary / name, expected)
        require(sha256(temporary / "cache/manifest.json") == closure["cache_manifest_sha256"], "cache manifest differs")
        require(sha256(temporary / "controls/source-manifest.json") == closure["source_manifest_sha256"], "source manifest differs")
        require(sha256(temporary / "controls/bound-recipe.json") == closure["recipe_sha256"], "bound recipe differs")
        require(sha256(temporary / "parent-checkpoint-66/campaign-manifest.json") == closure["parent_manifest_sha256"], "parent bootstrap manifest differs")
        for directory in sorted({p.parent for p in temporary.rglob("*") if p.is_file()}, key=lambda p: len(p.parts), reverse=True):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        os.rename(temporary, destination)
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    receipt = {"schema": "sepalith.pre04.runtime_restore_receipt.v1", "status": "complete", "repo": repo, "prefix": prefix, "revision": revision, "closure_sha256": closure_sha, "destination": str(destination), "files": len(files), "bytes": closure["total_bytes"], "credential_persisted": False}
    atomic_json(destination.parent / (destination.name + ".restore-receipt.json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True); parser.add_argument("--prefix", required=True)
    parser.add_argument("--revision", required=True); parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.prefix, args.revision, args.closure_sha256, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
