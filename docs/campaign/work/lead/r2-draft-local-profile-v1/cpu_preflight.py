"""Metadata-only preflight for the selected step-500 local DSpark profile.

This script deliberately does not open model weights, public draft weights, or
TRAIN.  It hashes only the copied Python/config source files listed in the
refreshed inventory; files larger than 1 MiB are rejected from the inventory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


PACKET = Path(__file__).resolve().parent
SOURCE_ROOT = PACKET / "source"
INVENTORY_PATH = SOURCE_ROOT / "docs/campaign/work/r2-draft-cloud-profile-v2/source-inventory.json"
TARGET_MANIFEST_PATH = PACKET / "target-manifest.step500.json"
REPORT_PATH = PACKET / "cpu-preflight.json"
MAX_INVENTORY_HASH_BYTES = 1 << 20

EXPECTED_TARGET = {
    "model_dir": "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged",
    "config_file": "config.json",
    "config_sha256": "f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991",
    "weights_file": "model.safetensors",
    "weights_sha256": "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c",
    "merged_weights_sha256": "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c",
    "generation_config_sha256": "7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269",
    "tokenizer_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
    "deepspec_revision": "005e03b81cec38b7da6399833d609ee89a2587f2",
}
EXPECTED_PUBLIC = {
    "path": "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-hf-v1/model.safetensors",
    "bytes": 647558522,
    "sha256": "ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97",
    "config_sha256": "bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b",
}
FIXED_RUNTIME_SHA256 = "76c3d3ca89397a4a120f0c3e11fa24602123851fe9165872c7578d0ed00a644a"
FIXED_RUNTIME_RELATIVE = "docs/campaign/work/r2-draft-target-runtime-v3/target_runtime.py"
TRAIN_SHA256 = "85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e"


def sha256_small(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_INVENTORY_HASH_BYTES:
        raise ValueError(f"inventory file exceeds CPU preflight hash bound: {path} ({size} bytes)")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def main() -> None:
    target = read_json(TARGET_MANIFEST_PATH)
    for key, expected in EXPECTED_TARGET.items():
        require(target.get(key) == expected, f"target manifest {key} differs from selected-500 metadata")
    require(target.get("expected_target_cache_identity") == target["model_dir"], "cache identity is not the resolved selected target path")
    require(target.get("source_manifest", {}).get("sha256") == "92157a4a52a4ed928f76df9f1f77a6aa39235eefd6a859eb9c2b3d4118ee18db", "selected parent manifest pin differs")
    contract = target.get("tokenizer_contract")
    require(contract == {"vocab_size": 130560, "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1, "native_eog_ids": [1, 130073]}, "tokenizer contract differs")
    binding = target.get("profile_binding")
    require(binding == {"rows": 8, "steps": 8, "block_size": 7, "num_draft_layers": 5, "target_layer_ids": [1, 10, 20, 30, 39], "num_anchors": 32, "precision": "bf16", "max_context_tokens": 4096, "resume": False}, "profile binding differs")

    inventory = read_json(INVENTORY_PATH)
    require(inventory.get("portable") is True, "source inventory is not portable")
    require(inventory.get("excluded") == [".git", "dependency-overlay"], "source inventory exclusions differ")
    verified_files = 0
    inventory_entries = list(inventory.get("files", [])) + list(inventory.get("requirements", []))
    require(inventory_entries, "source inventory is empty")
    for entry in inventory_entries:
        relative = entry.get("path")
        require(isinstance(relative, str) and not Path(relative).is_absolute(), f"inventory path is not relative: {relative}")
        path = SOURCE_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"inventory source is missing: {relative}")
        actual_bytes = path.stat().st_size
        require(actual_bytes == int(entry["bytes"]), f"inventory byte count differs: {relative}")
        actual_sha = sha256_small(path)
        require(actual_sha == entry["sha256"], f"inventory SHA differs: {relative}")
        verified_files += 1

    runtime_path = SOURCE_ROOT / FIXED_RUNTIME_RELATIVE
    require(sha256_small(runtime_path) == FIXED_RUNTIME_SHA256, "fixed target runtime SHA differs")
    profile_path = SOURCE_ROOT / "docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py"
    profile_text = profile_path.read_text(encoding="utf-8")
    require(EXPECTED_TARGET["weights_sha256"] in profile_text, "local profile target pin is not selected step-500")
    require("b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4" not in profile_text, "local profile retains v6 a-250 target pin")
    require(re.search(r"PROFILE_ROWS\s*=\s*8", profile_text), "profile row bound is not eight")
    require(re.search(r"PROFILE_STEPS\s*=\s*8", profile_text), "profile step bound is not eight")
    require(re.search(r"PROFILE_LIMIT_SECONDS\s*=\s*3600", profile_text), "profile cumulative deadline is not 3600 seconds")
    require('"target_model_name_or_path": _target_cache_model_name(target_identity)' in runtime_path.read_text(encoding="utf-8"), "cache identity repair is not staged")

    config_path = SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/profile_config.py"
    config_text = config_path.read_text(encoding="utf-8")
    require('exp_name = "dspark_minicpm5_2b_step500_profile"' in config_text, "step-500 profile experiment name is missing")
    require("max_train_steps=8" in config_text, "trainer max_train_steps is not eight")
    require("checkpointing_steps=8" in config_text, "trainer checkpointing_steps is not eight")
    require('precision="bf16"' in config_text, "BF16 training precision is missing")
    require("allow_resume=False" in config_text, "implicit resume is enabled")

    warmstart_path = SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/warmstart_trainer.py"
    warmstart_text = warmstart_path.read_text(encoding="utf-8")
    require(EXPECTED_PUBLIC["sha256"] in warmstart_text, "public draft weight pin differs")
    require(str(EXPECTED_PUBLIC["bytes"]) in warmstart_text, "public draft byte pin differs")
    require("FROZEN_TARGET_KEYS" in warmstart_text, "frozen target matrix gate is missing")

    report = {
        "schema": "sepalith.run08.r2-draft-local-profile.cpu-preflight.v1",
        "status": "PASS",
        "cpu_only": True,
        "model_loaded": False,
        "model_or_train_content_read": False,
        "model_weights_hashed": False,
        "public_weights_hashed": False,
        "train_rows_read": False,
        "gpu_or_provider_used": False,
        "network_used": False,
        "selected_parent": {
            "alias": "NATIVE/models/SFT11-task-global-b-500-merged",
            "manifest_sha256": target["source_manifest"]["sha256"],
            "merged_weights_sha256": target["merged_weights_sha256"],
            "status": target["source_manifest"]["status"],
        },
        "public_draft_metadata": EXPECTED_PUBLIC,
        "source_inventory": {
            "path": str(INVENTORY_PATH),
            "entries_verified": verified_files,
            "fixed_runtime_sha256": FIXED_RUNTIME_SHA256,
        },
        "profile_contract": binding,
        "cache_identity": {
            "producer_value": target["expected_target_cache_identity"],
            "trainer_value_required": target["expected_target_cache_identity"],
            "mismatch_policy": "official validate_train_cache must fail before trainer step",
        },
        "evidence_limits": [
            "Hashes were computed only for inventory source/config files at or below 1 MiB.",
            "Target model, public draft weight payload, and TRAIN were not opened or hashed.",
            "No teacher rows, cache shards, CUDA forward/backward, optimizer step, or checkpoint was produced.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
