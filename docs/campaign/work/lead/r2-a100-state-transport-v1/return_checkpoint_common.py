#!/usr/bin/env python3
"""Validate and describe a future DDP8 checkpoint for portable local restore."""
from __future__ import annotations

import json
import stat
from pathlib import Path

from transport_common import PAYLOAD_REQUIRED, atomic_json, require, safe_relative, sha256

RANK_RNG = {f"rng_state_{rank}.pth" for rank in range(8)}
RETURN_EXTRA = RANK_RNG | {"distributed-runtime.json"}


def build(checkpoint: Path, expected_manifest_sha256: str, repo: str, prefix: str) -> dict:
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "campaign-manifest.json"
    require(manifest_path.is_file() and sha256(manifest_path) == expected_manifest_sha256, "accepted return manifest differs")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("full") is True and manifest.get("checkpoint_kind") == "full_weights" and type(manifest.get("step")) is int and manifest["step"] > 90, "return checkpoint identity differs")
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == (PAYLOAD_REQUIRED - {"campaign-manifest.json"}) | RETURN_EXTRA, "return checkpoint file closure differs")
    state = json.loads((checkpoint / "campaign-state.json").read_text())
    sampler = state.get("sampler", {})
    require(state.get("step") == manifest["step"] and sampler.get("global_step") == manifest["step"], "return checkpoint step differs")
    require(sampler.get("global_optimizer_step_offset") == 66 and sampler.get("effective_batch") == 16, "return sampler policy differs")
    require(sampler.get("draw_schedule_sha256") == "78bc2f3ec17ece7ad56a6dd79d4a1369d58c2329b5fb6a282d5a613135517937", "return draw schedule differs")
    require(sampler.get("cursor") == sampler.get("stage_cursor") == (manifest["step"] - 66) * 16, "return stage cursor differs")
    distributed = json.loads((checkpoint / "distributed-runtime.json").read_text())
    require(distributed.get("world_size") == 8 and distributed.get("portable_single_device") is True and distributed.get("single_device_resume_rng") == "rng_state.pth == rank0", "return distributed portability differs")
    physical = {}
    for name, row in sorted(files.items()):
        require(safe_relative(name), "unsafe return filename")
        path = checkpoint / name
        require(path.is_file() and not path.is_symlink(), "return file missing/nonregular:" + name)
        st = path.stat()
        require(st.st_size == row.get("bytes") and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64, "return file metadata differs:" + name)
        physical[name] = {**row, "source_mode": stat.S_IMODE(st.st_mode), "restore_mode": 0o600}
    require(files["rng_state.pth"]["sha256"] == files["rng_state_0.pth"]["sha256"], "canonical return RNG is not rank0")
    physical["campaign-manifest.json"] = {"bytes": manifest_path.stat().st_size, "sha256": expected_manifest_sha256, "source_mode": stat.S_IMODE(manifest_path.stat().st_mode), "restore_mode": 0o600}
    return {"schema": "sepalith.pre04.full_weight_return_transport_spec.v1", "status": "prepared_upload_not_authorized", "repo": repo, "prefix": prefix, "checkpoint_path": str(checkpoint.resolve()), "campaign_manifest_sha256": expected_manifest_sha256, "step": manifest["step"], "files": physical, "total_bytes": sum(x["bytes"] for x in physical.values()), "source_world_size": 8, "portable_world_size": 1}
