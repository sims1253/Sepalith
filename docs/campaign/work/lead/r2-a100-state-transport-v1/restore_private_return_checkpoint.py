#!/usr/bin/env python3
"""Restore an exact DDP8-produced checkpoint for portable rank-0 local resume."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from return_checkpoint_common import RETURN_EXTRA
from restore_private_checkpoint import client, stream_copy
from transport_common import PAYLOAD_REQUIRED, atomic_json, require, safe_relative, sha256


def run(repo, prefix, revision, closure_sha, destination, *, api=None, download=None, token=None):
    destination = Path(destination); require(not destination.exists(), "fresh return destination required")
    if api is None: api, download, token = client(repo)
    closure_file = Path(download(repo, filename=prefix + "/closure.json", revision=revision, repo_type="model", token=token))
    require(sha256(closure_file) == closure_sha, "return closure hash differs")
    closure = json.loads(closure_file.read_text()); files = closure.get("files")
    require(closure.get("schema") == "sepalith.pre04.full_weight_return_closure.v1" and closure.get("status") == "complete" and closure.get("repo") == repo and closure.get("prefix") == prefix and closure.get("source_world_size") == 8 and closure.get("portable_world_size") == 1, "return closure identity differs")
    require(isinstance(files, dict) and set(files) == PAYLOAD_REQUIRED | RETURN_EXTRA and all(safe_relative(x) for x in files), "return closure files differ")
    require(files["rng_state.pth"]["sha256"] == files["rng_state_0.pth"]["sha256"], "return canonical RNG differs from rank0")
    destination.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=destination.parent))
    try:
        for name, expected in sorted(files.items()):
            cached = download(repo, filename=prefix + "/files/" + name, revision=revision, repo_type="model", token=token); stream_copy(cached, temporary / name, expected)
        require(sha256(temporary / "campaign-manifest.json") == closure["campaign_manifest_sha256"], "return campaign manifest differs")
        manifest = json.loads((temporary / "campaign-manifest.json").read_text()); require(manifest.get("files") == {k: {"bytes": files[k]["bytes"], "sha256": files[k]["sha256"]} for k in files if k != "campaign-manifest.json"}, "return internal manifest differs")
        fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd); os.rename(temporary, destination); fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    receipt = {"schema": "sepalith.pre04.return_restore_receipt.v1", "status": "complete", "repo": repo, "prefix": prefix, "revision": revision, "closure_sha256": closure_sha, "destination": str(destination), "step": closure["step"], "files": len(files), "bytes": closure["total_bytes"], "local_resume_world_size": 1, "credential_persisted": False}
    atomic_json(destination.parent / (destination.name + ".restore-receipt.json"), receipt); return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--repo", required=True); p.add_argument("--prefix", required=True); p.add_argument("--revision", required=True); p.add_argument("--closure-sha256", required=True); p.add_argument("--destination", type=Path, required=True); a = p.parse_args(); print(json.dumps(run(a.repo, a.prefix, a.revision, a.closure_sha256, a.destination), sort_keys=True))


if __name__ == "__main__": main()
