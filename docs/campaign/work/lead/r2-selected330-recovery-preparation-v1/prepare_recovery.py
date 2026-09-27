#!/usr/bin/env python3
"""Bind a cadence-24 recovery recipe without reading checkpoint payloads."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = Path(__file__).resolve().parent
BASE = PLAN / "docs/campaign/work/lead/r2-selected330-ordinary-root-v1/runtime-recipe.json"
BASE_SHA = "eff266cdb287e8b099990f265760539e024bc20c14dd0707b6e225abce1a6981"
SOURCE_CHECKPOINT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-varlen-canary-322-varlen_candidate-v1/runtime/checkpoint-330")
SOURCE_MANIFEST_SHA = "2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951"
SOURCE_STATE_SHA = "fb6d4d456e2318c70301a8b39f21e54069570c48c9c90de80e3f09155361a8e8"
STAGE_RECEIPT_SHA = "a72c1b699fe931e0db0fa2ddfc96a43486e74fc962802e8a541540e16596b88d"
SCIENTIFIC_SOURCE_SHA = "8dd7ecf617e37e1778dabee6721857e1b79074e0314f17e0e3f55b57a809a98c"
CANONICAL_RECIPE_SHA = "579269f0f203ae1eed9cdb96af0f535f19c5953e2cf3e19c52615b7545e10f70"
FIRST_STOP = 354
CADENCE = 24


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_trainer():
    source = PACKET / "source/experiments/training/full_weight_cpt_trainer.py"
    spec = importlib.util.spec_from_file_location("recovery_trainer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(source_manifest, migration_admission, trainer_root, archive_root, output_dir):
    require(sha(BASE) == BASE_SHA, "base ordinary recipe differs")
    source_manifest = Path(source_manifest).resolve()
    migration_admission = Path(migration_admission).resolve()
    source_sha = sha(source_manifest)
    source = json.loads(source_manifest.read_text())
    require(source.get("schema") == "sepalith.sft11.native-cpt-trainer-source.v2", "runtime source schema differs")
    migration = json.loads(migration_admission.read_text())
    require(migration.get("schema") == "sepalith.sft11.native-runtime-source-migration-admission.v1", "migration schema differs")
    require(migration.get("status") == "admitted" and migration.get("launch_authorized") is True, "runtime source migration is not root-admitted")
    require(migration.get("runtime_source_manifest_sha256") == source_sha, "migration runtime source differs")
    require(migration.get("scientific_source_manifest_sha256") == SCIENTIFIC_SOURCE_SHA, "migration scientific source differs")
    require(migration.get("canonical_bound_recipe_sha256") == CANONICAL_RECIPE_SHA, "migration canonical recipe differs")
    require(migration.get("stage_receipt_sha256") == STAGE_RECEIPT_SHA, "migration staged data differs")
    require(migration.get("allowed_changes") == [
        "selected packed330 identity-transition validation",
        "immutable packed canary and selection lineage propagation",
        "fresh ordinary-production runtime output roots",
        "recovery checkpoint cadence 24 with first mandatory stop 354",
    ], "migration allowed changes differ")

    trainer_root = Path(trainer_root)
    archive_root = Path(archive_root)
    require(str(trainer_root).startswith("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"), "trainer root is not native campaign storage")
    require(str(archive_root).startswith("/mnt/e/sepalith/campaign-20260915/checkpoints/"), "archive root is not durable E storage")
    require(not trainer_root.exists() and not archive_root.exists(), "recovery output roots must be fresh")

    recipe = copy.deepcopy(json.loads(BASE.read_text()))
    recipe["runtime_source"] = {"manifest_path": str(source_manifest), "manifest_sha256": source_sha}
    recipe["runtime_source_migration"] = {"admission": str(migration_admission), "admission_sha256": sha(migration_admission)}
    recipe["outputs"]["trainer"] = str(trainer_root)
    recipe["outputs"]["archive"] = str(archive_root)
    recipe["outputs"]["graceful_stop"] = "/home/m0hawk/.local/state/sepalith/campaign-20260915/control/" + trainer_root.parent.name + "-save-stop.json"
    recipe["checkpoint_storage"]["trainer_root"] = str(trainer_root)
    recipe["checkpoint_storage"]["archive_root"] = str(archive_root)
    recipe["runtime"]["checkpoint_every"] = CADENCE
    recipe["runtime"]["mandatory_stop_step"] = FIRST_STOP
    recipe["runtime"]["evaluation_steps"] = sorted(set(recipe["runtime"]["evaluation_steps"] + [FIRST_STOP]))
    recipe["runtime"]["selected_milestones"] = sorted(set(recipe["runtime"]["selected_milestones"] + [FIRST_STOP]))

    trainer = load_trainer()
    trainer.validate_stage_schedule(recipe)
    old_identity = trainer.identity(json.loads(BASE.read_text()))
    new_identity = trainer.identity(recipe)
    changed = []
    for key in old_identity:
        if old_identity[key] != new_identity[key]:
            changed.append(key)
    require(changed == ["schedule"], "recovery changed non-schedule scientific identity")
    old_schedule = old_identity["schedule"]
    new_schedule = new_identity["schedule"]
    require({k: v for k, v in old_schedule.items() if k not in {"checkpoint_every", "mandatory_stop_step"}} == {k: v for k, v in new_schedule.items() if k not in {"checkpoint_every", "mandatory_stop_step"}}, "recovery changed optimizer horizon or batch schedule")
    require((old_schedule["checkpoint_every"], old_schedule["mandatory_stop_step"]) == (128, 194), "base schedule differs")
    require((new_schedule["checkpoint_every"], new_schedule["mandatory_stop_step"]) == (24, 354), "recovery schedule differs")

    output_dir = Path(output_dir)
    recipe_path = output_dir / "runtime-recipe.json"
    write_json(recipe_path, recipe)
    recipe_sha = sha(recipe_path)
    destination_identity_sha = canonical_sha(new_identity)
    selected = json.loads((PACKET / "selected-packed330-transition.template.json").read_text())
    selected.update({"bound_recipe_sha256": recipe_sha, "destination_identity_sha256": destination_identity_sha})
    continuation = json.loads((PACKET / "continuation-330.template.json").read_text())
    continuation["bound_recipe_sha256"] = recipe_sha
    execution_stop = json.loads((PACKET / "execution-stop-354.template.json").read_text())
    execution_stop["bound_recipe_sha256"] = recipe_sha
    write_json(output_dir / "selected-packed330-transition.review.json", selected)
    write_json(output_dir / "continuation-330.review.json", continuation)
    write_json(output_dir / "execution-stop-354.review.json", execution_stop)
    python = "/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python"
    attestation = str((PACKET / "source/experiments/training/native_launch_attestation.py").resolve())
    trainer_entry = str((PACKET / "source/experiments/training/full_weight_cpt_trainer.py").resolve())
    wrapper = str((PACKET / "run_with_allocator.py").resolve())
    common = [
        "--recipe", str(recipe_path.resolve()),
        "--resume", str(SOURCE_CHECKPOINT),
        "--continuation-admission", str((output_dir / "continuation-330.admitted.json").resolve()),
        "--execution-stop-admission", str((output_dir / "execution-stop-354.admitted.json").resolve()),
        "--ordinary-canary-transition-admission", str((output_dir / "selected-packed330-transition.admitted.json").resolve()),
    ]
    attest = [attestation, "--base-receipt", "/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json", "--base-receipt-sha256", STAGE_RECEIPT_SHA, "--resume", str(SOURCE_CHECKPOINT), "--resume-manifest-sha256", SOURCE_MANIFEST_SHA, "--"]
    cpu = ["env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=2", python, "-B", *attest, python, "-B", trainer_entry, "preflight-resume", *common]
    gpu = ["env", "CUDA_VISIBLE_DEVICES=0", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=4", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "UNSLOTH_RETURN_LOGITS=0", "TORCHINDUCTOR_CACHE_DIR=/mnt/e/sepalith/campaign-20260915/cache/full-weight-smoke/torchinductor", "TRITON_CACHE_DIR=/mnt/e/sepalith/campaign-20260915/cache/full-weight-smoke/triton", "CUDA_CACHE_PATH=/mnt/e/sepalith/campaign-20260915/cache/full-weight-smoke/cuda", "TMPDIR=/mnt/e/sepalith/campaign-20260915/cache/full-weight-smoke/tmp", "PYTORCH_ALLOC_CONF=backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8", f"SEPALITH_ALLOCATOR_REPORT={output_dir.resolve()}/allocator.json", python, "-B", *attest, python, "-B", wrapper, "run", *common]
    guard = [python, "-B", str((PLAN / "docs/campaign/work/lead/host-memory-guard-v4/cuda_host_guard.py").resolve()), "--command-json", str((output_dir / "training-command.json").resolve()), "--output", "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-recovery-cadence24-v1-host-supervision", "--seconds", "10800", "--minimum-free-mib", "6144", "--admission-free-mib", "14336"]
    write_json(output_dir / "cpu-preflight-command.json", cpu)
    write_json(output_dir / "training-command.json", gpu)
    write_json(output_dir / "guard-command.json", guard)
    result = {
        "schema": "sepalith.sft11.selected330-recovery-binding.v1",
        "status": "prepared_requires_root_admission_no_launch",
        "runtime_recipe": str(recipe_path.resolve()),
        "runtime_recipe_sha256": recipe_sha,
        "source_manifest_sha256": source_sha,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_checkpoint_state_sha256": SOURCE_STATE_SHA,
        "destination_identity_sha256": destination_identity_sha,
        "resume_step": 330,
        "resume_cursor": 4224,
        "checkpoint_cadence": 24,
        "first_stop": 354,
        "first_stop_cursor": (354 - 66) * 16,
        "payload_bytes_read": 0,
        "root_must_create_empty_output_roots_after_binding_before_run": [str(trainer_root), str(archive_root)],
    }
    write_json(output_dir / "binding-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--runtime-source-migration-admission", required=True)
    parser.add_argument("--trainer-root", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_manifest, args.runtime_source_migration_admission, args.trainer_root, args.archive_root, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
