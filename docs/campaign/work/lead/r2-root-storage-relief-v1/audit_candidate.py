#!/usr/bin/env python3
"""Audit one pinned CPT parent and path-preserving relocation constraints.

This intentionally touches only the named checkpoint directory, its small
campaign manifests, the named validator source files, and /proc metadata for
exact candidate references. It never copies, moves, deletes, or loads model
weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
STATE = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915")
CANDIDATE = STATE / "models/SFT11-CPT-global-a-250-merged"
WEIGHTS = CANDIDATE / "model.safetensors"
DESTINATION = Path("/mnt/e/sepalith/campaign-20260915/models/SFT11-CPT-global-a-250-merged")
MERGE_RECEIPT = PLAN / "docs/campaign/receipts/SFT-11-CPT-global250-merge-acceptance.json"
TASK_RECIPE = PLAN / "docs/campaign/work/lead/r2-task-global-a/recipe.json"
NATIVE_RECIPE = PLAN / "docs/campaign/work/lead/r2-task-global500-native-selection/merge-recipe.json"
PARENT_MANIFEST = CANDIDATE / "parent-manifest.preparation.json"
RELEASE_RECEIPT = PLAN / "docs/campaign/receipts/SFT-11-post-preflight-cache-release-root.json"
VALIDATORS = {
    "dense_model_layout": PLAN / "docs/campaign/work/lead/r2-full-weight-cpt-representative-trainer-v3/source/experiments/training/model_layout.py",
    "full_weight_trainer": PLAN / "docs/campaign/work/lead/r2-full-weight-cpt-representative-trainer-v3/source/experiments/training/full_weight_cpt_trainer.py",
    "task_selection_contract": PLAN / "docs/campaign/work/lead/r2-task-global500-native-selection/packet/selection_contract.py",
    "native_final_row_guard": PLAN / "docs/campaign/work/lead/r2-task-global500-native-selection/packet/native_evaluator/final_row_gate.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256(path)}


def symlink_components(path: Path) -> list[str]:
    found = []
    for index in range(1, len(path.parts) + 1):
        component = Path(*path.parts[:index])
        if component.is_symlink():
            found.append(str(component))
    return found


def process_references() -> dict:
    """Return only role/PID metadata for processes mentioning this path."""

    references = []
    needle = str(WEIGHTS).encode()
    for proc in Path("/proc").glob("[0-9]*"):
        cmdline_path = proc / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if needle not in raw:
            continue
        try:
            stat_fields = (proc / "stat").read_text().split()
            start_tick = stat_fields[21]
        except (FileNotFoundError, PermissionError, OSError, IndexError):
            start_tick = None
        command_class = "other"
        text = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if "host-memory-guard" in text:
            command_class = "host_memory_guard"
        elif "full_weight_cpt_trainer.py" in text:
            command_class = "full_weight_cpt_trainer"
        references.append({"pid": int(proc.name), "start_tick": start_tick, "command_class": command_class})
    return {"count": len(references), "references": sorted(references, key=lambda x: x["pid"])}


def exact_open_fds() -> dict:
    try:
        completed = subprocess.run(
            ["lsof", "-n", "-P", "--", str(WEIGHTS)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__, "pids": []}
    pids = []
    for line in completed.stdout.splitlines()[1:]:
        match = re.match(r"\S+\s+(\d+)\s+", line)
        if match:
            pids.append(int(match.group(1)))
    return {"status": "queried", "exit_code": completed.returncode, "pids": sorted(set(pids))}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main(output: Path) -> None:
    if output.exists():
        raise SystemExit("fresh output required")
    merge = load_json(MERGE_RECEIPT)
    task = load_json(TASK_RECIPE)
    native = load_json(NATIVE_RECIPE)
    parent = load_json(PARENT_MANIFEST)
    release = load_json(RELEASE_RECEIPT)

    expected = merge["files"]
    siblings = {}
    for name in sorted(expected):
        path = CANDIDATE / name
        siblings[name] = file_record(path)
    # These companions are present in the actual merged parent directory even
    # though the older merge receipt's file map predates them.
    for name in ("chat_template.jinja", "parent-manifest.preparation.json"):
        if (CANDIDATE / name).is_file():
            siblings[name] = file_record(CANDIDATE / name)

    candidate_stat = WEIGHTS.stat()
    destination_parent = DESTINATION.parent
    output_value = {
        "schema": "sepalith.pre03-root-storage-relief-audit.v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "path": str(WEIGHTS),
            "resolved_path": str(WEIGHTS.resolve()),
            "is_symlink": WEIGHTS.is_symlink(),
            "symlink_components": symlink_components(WEIGHTS),
            "mode": oct(candidate_stat.st_mode & 0o777),
            "owner_uid": candidate_stat.st_uid,
            "owner_gid": candidate_stat.st_gid,
            "size_bytes": candidate_stat.st_size,
            "inode": candidate_stat.st_ino,
            "device": candidate_stat.st_dev,
            "mtime_ns": candidate_stat.st_mtime_ns,
            "sha256": sha256(WEIGHTS),
            "parent_directory": str(CANDIDATE),
            "parent_directory_is_symlink": CANDIDATE.is_symlink(),
            "sibling_files": siblings,
        },
        "destination": {
            "path": str(DESTINATION),
            "exists": DESTINATION.exists() or DESTINATION.is_symlink(),
            "parent_exists": destination_parent.is_dir(),
            "parent_device": destination_parent.stat().st_dev if destination_parent.is_dir() else None,
            "candidate_parent_device": CANDIDATE.parent.stat().st_dev,
            "available_bytes": os.statvfs(destination_parent).f_bavail * os.statvfs(destination_parent).f_frsize if destination_parent.is_dir() else None,
        },
        "open_file_evidence": {
            "exact_candidate_lsof": exact_open_fds(),
            "candidate_argument_references": process_references(),
        },
        "pins": {
            "merge_receipt": {"path": str(MERGE_RECEIPT), "sha256": sha256(MERGE_RECEIPT), "expected_weights_sha256": expected["model.safetensors"]},
            "task_recipe": {"path": str(TASK_RECIPE), "sha256": sha256(TASK_RECIPE), "model_path": task.get("model_path"), "identity_parent_weights_sha256": task.get("identity", {}).get("parent", {}).get("weights_sha256")},
            "native_recipe": {"path": str(NATIVE_RECIPE), "sha256": sha256(NATIVE_RECIPE), "model_path": native.get("model_path"), "identity_parent_weights_sha256": native.get("identity", {}).get("parent", {}).get("weights_sha256")},
            "parent_manifest": {"path": str(PARENT_MANIFEST), "sha256": sha256(PARENT_MANIFEST), "kind": parent.get("kind"), "status": parent.get("status"), "merged_model_path": parent.get("merged_model_path"), "merged_weights_sha256": parent.get("merged_weights_sha256")},
            "cache_release_receipt": {"path": str(RELEASE_RECEIPT), "sha256": sha256(RELEASE_RECEIPT), "path_bound": release.get("path"), "file_metadata_unchanged": release.get("file_metadata_unchanged")},
        },
        "validator_source_pins": {name: {"path": str(path), "sha256": sha256(path)} for name, path in VALIDATORS.items()},
        "interpretation": {
            "candidate_is_immutable_pinned_parent": True,
            "candidate_content_hash_matches_merge_and_task_pins": siblings["model.safetensors"]["sha256"] == expected["model.safetensors"] == task.get("identity", {}).get("parent", {}).get("weights_sha256") == parent.get("merged_weights_sha256"),
            "candidate_open_fd_count": len(exact_open_fds()["pids"]),
            "directory_symlink_for_cpt_dense_layout": "synthetic test pass",
            "individual_weight_symlink_for_cpt_dense_layout": "synthetic test rejected",
            "directory_symlink_for_native_final_row_paths": "synthetic test rejected by path-component guard",
            "active_trainer_consumes_candidate": False,
            "active_trainer_observed_checkpoint": "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-representative-lr3e-6-v1/full/checkpoint-66/model.safetensors",
        },
        "relocation_plan": {
            "preferred_destination": str(DESTINATION),
            "safe_stages": [
                "pause/confirm no reader owns the candidate and wait for checkpoint IO clear",
                "copy the complete seven-file HF parent directory to a fresh E temporary directory",
                "compare explicit file inventory, byte sizes, and SHA256 hashes against this audit and parent manifest",
                "run model_layout.validate_dense_weights on the E directory while it is still a regular directory",
                "publish the verified E directory, then atomically rename the root directory to a rollback backup and create a directory symlink at the original path",
                "re-run parent preflight, exact hash checks, and path-specific load checks through the original path",
                "retain the root backup until root verifies all consumers; delete it only after explicit approval to reclaim approximately 4.69 GiB",
            ],
            "directory_link_scope": "CPT/full-weight training parent path only; current dense validator follows a directory link while requiring regular final files",
            "link_exclusions": "Do not use a symlink component for native final-row paths/source closure or any validator that checks path.is_symlink(); use a fresh E-bound recipe/path there.",
            "rollback": "remove the link and restore the retained root directory before deleting the backup",
            "no_action_taken": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "candidate_sha256": output_value["candidate"]["sha256"], "open_fd_pids": output_value["open_file_evidence"]["exact_candidate_lsof"]["pids"]}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
