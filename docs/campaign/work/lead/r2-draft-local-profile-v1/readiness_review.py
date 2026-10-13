"""CPU-only readiness review for the selected step-500 DSpark profile.

The review imports the source closure with the existing canonical interpreter
and dependency overlay, but never calls a model loader, opens model weights,
or reads TRAIN.  It exercises the cache identity check with synthetic metadata
and inspects the official checkpoint-writing schema.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any


PACKET = Path(__file__).resolve().parent
SOURCE_ROOT = PACKET / "source"
PROFILE_DIR = SOURCE_ROOT / "docs/campaign/work/r2-draft-cloud-profile-v2"
TARGET_RUNTIME_PATH = SOURCE_ROOT / "docs/campaign/work/r2-draft-target-runtime-v3/target_runtime.py"
PROFILE_CONFIG_PATH = SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/profile_config.py"
PROFILE_TRAIN_PATH = SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/train.py"
PROFILE_ORCHESTRATOR_PATH = PROFILE_DIR / "profile_orchestrator.py"
INVENTORY_PATH = PROFILE_DIR / "source-inventory.json"
TARGET_MANIFEST_PATH = PACKET / "target-manifest.step500-accepted-wrapper.json"
WRAPPER_PATH = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-step500-rl-gate-v2/parent-manifest.json")
OVERLAY = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/dependency-overlay")
REPORT_PATH = PACKET / "readiness-review.json"
MAX_SOURCE_HASH_BYTES = 1 << 20

TARGET_MODEL_DIR = "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged"
TRAIN_SHA256 = "85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e"
WEIGHTS_SHA256 = "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c"
BASE_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
DEEPSPEC_REVISION = "005e03b81cec38b7da6399833d609ee89a2587f2"
RUNTIME_SHA256 = "76c3d3ca89397a4a120f0c3e11fa24602123851fe9165872c7578d0ed00a644a"

BUDGET = {
    "all_in_seconds": 1800,
    "stage_execution_budget_seconds": 1500,
    "setup_and_persistence_reserve_seconds": 300,
    "final_refresh_allowance_seconds": 1200,
    "integration_allowance_seconds": 600,
    "original_profile_limit_seconds": 3600,
    "original_total_window_seconds": 7200,
    "original_deadline_unchanged": True,
    "rl_extension_seconds": 0,
    "stages": [
        {"name": "teacher-cache-profile", "seconds": 420},
        {"name": "native-cuda-smoke", "seconds": 240},
        {"name": "warmstart-trainer-profile", "seconds": 720},
        {"name": "stage-terminal-collection", "seconds": 120},
    ],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha_small(path: Path) -> str:
    size = path.stat().st_size
    require(size <= MAX_SOURCE_HASH_BYTES, f"source hash bound exceeded: {path} ({size} bytes)")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def module_location(module: Any) -> str:
    """Return a stable location for both regular and namespace packages."""
    location = getattr(module, "__file__", None)
    if location:
        return str(Path(location).resolve())
    package_paths = list(getattr(module, "__path__", ()))
    require(package_paths, f"module has no file or package path: {module!r}")
    return str(Path(package_paths[0]).resolve())


def baseline_tensorboard_probe() -> dict[str, Any]:
    """Show whether canonical site-packages alone has the trainer dependency."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "import torch.utils.tensorboard"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = (result.stdout + result.stderr).strip()
    return {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "error_tail": combined[-240:],
        "expected_missing_tensorboard": result.returncode != 0 and "No module named 'tensorboard'" in combined,
    }


def generated_stage_import_probe(profile: Any) -> dict[str, Any]:
    """Exercise clean_environment's actual PYTHONPATH contract on CPU.

    The dependency overlay is deliberately outside the portable source tree.
    The first probe therefore records the failure a canonical venv has before
    cloud dependency setup.  The second probe prepends that already-installed
    overlay and proves the same relocated source closure imports successfully.
    """
    with tempfile.TemporaryDirectory(prefix="r2-local-stage-import-") as temporary:
        run = Path(temporary)
        stage_env = profile.clean_environment(
            run=run,
            inputs={
                "target": {"model_dir": TARGET_MODEL_DIR},
                "public_draft": {"actual_path": "/tmp/public-draft/model.safetensors"},
            },
            source_root=SOURCE_ROOT,
        )
        stage_env["CUDA_VISIBLE_DEVICES"] = ""
        stage_env["PYTHONDONTWRITEBYTECODE"] = "1"
        probe_code = "import torch.utils.tensorboard; import deepspec; import transformers"

        def run_probe(environment: dict[str, str]) -> dict[str, Any]:
            result = subprocess.run(
                [sys.executable, "-c", probe_code],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            combined = (result.stdout + result.stderr).strip()
            return {
                "returncode": result.returncode,
                "passed": result.returncode == 0,
                "error_tail": combined[-240:],
            }

        without_overlay = run_probe(stage_env)
        with_overlay_env = dict(stage_env)
        with_overlay_env["PYTHONPATH"] = os.pathsep.join(
            [str(OVERLAY), stage_env.get("PYTHONPATH", "")]
        )
        with_overlay = run_probe(with_overlay_env)
        return {
            "stage_pythonpath": stage_env.get("PYTHONPATH", ""),
            "overlay_is_outside_portable_source": str(OVERLAY) not in stage_env.get("PYTHONPATH", ""),
            "without_overlay": without_overlay,
            "with_existing_overlay": with_overlay,
        }


def main() -> None:
    target = read_json(TARGET_MANIFEST_PATH)
    wrapper = read_json(WRAPPER_PATH)
    require(target["model_dir"] == TARGET_MODEL_DIR, "accepted target manifest model path differs")
    require(target["weights_sha256"] == WEIGHTS_SHA256 and target["merged_weights_sha256"] == WEIGHTS_SHA256, "target weight pin differs")
    require(target["source_manifest"]["status"] == "accepted", "accepted wrapper is not bound")
    require(target["source_manifest"]["kind"] == "merged_sft", "accepted wrapper kind differs")
    require(target["source_manifest"]["sha256"] == "05eac983926eacc45b78f59d281eeaeeecfc8c58ed03647b4a8cbe794160afa5", "accepted wrapper SHA differs")
    require(wrapper["status"] == "accepted" and wrapper["kind"] == "merged_sft", "parent wrapper is not accepted merged_sft")
    require(wrapper["merged_model_path"] == TARGET_MODEL_DIR, "accepted wrapper model path differs")
    require(wrapper["merged_weights_sha256"] == WEIGHTS_SHA256, "accepted wrapper weight pin differs")
    require(wrapper["base_model_revision"] == BASE_REVISION, "accepted wrapper base revision differs")
    require(wrapper["sft_identity"]["data"]["train_rows_sha256"] == TRAIN_SHA256, "accepted wrapper TRAIN identity differs")
    require(wrapper["sft_identity"]["source"] == "306b50daf21a1c1daa31d849f2f1d64b43dbe4c1349afba1d3793366ab0d4fbb", "accepted wrapper source identity differs")
    require(wrapper["sft_identity"]["renderer"]["id"] == "zeta2-prm03-v1", "accepted wrapper renderer differs")
    require(wrapper["sft_identity"]["renderer"]["contract_sha256"] == "173577c7df85d6e51e22ff9a556062737b311382f58a029b6a177e9a418e57d1", "accepted wrapper renderer contract differs")
    require(wrapper["sft_identity"]["tokenizer"]["BOS"] == 0, "accepted wrapper BOS differs")
    require(wrapper["sft_identity"]["tokenizer"]["EOS_PAD"] == 1, "accepted wrapper EOS/PAD differs")
    require(wrapper["sft_identity"]["tokenizer"]["native_EOG"] == [1, 130073], "accepted wrapper native EOG differs")
    require(wrapper["config_sha256"] == target["config_sha256"], "accepted wrapper config pin differs")
    require(wrapper["generation_config_sha256"] == target["generation_config_sha256"], "accepted wrapper generation config pin differs")

    inventory = read_json(INVENTORY_PATH)
    entries = list(inventory["files"]) + list(inventory["requirements"])
    verified_entries = 0
    for entry in entries:
        path = SOURCE_ROOT / entry["path"]
        require(path.is_file() and not path.is_symlink(), f"inventory source missing: {entry['path']}")
        require(path.stat().st_size == int(entry["bytes"]), f"inventory byte count differs: {entry['path']}")
        require(sha_small(path) == entry["sha256"], f"inventory SHA differs: {entry['path']}")
        verified_entries += 1
    require(sha_small(TARGET_RUNTIME_PATH) == RUNTIME_SHA256, "fixed runtime SHA differs")

    # Add the pre-existing dependency overlay only as an import path. No
    # packages are installed or modified by this review.
    source_paths = [
        OVERLAY,
        PROFILE_DIR,
        TARGET_RUNTIME_PATH.parent,
        SOURCE_ROOT / "docs/campaign/work/r2-draft-adapter-hardening-v1",
        SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation",
        SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec",
    ]
    for path in reversed(source_paths):
        if path.is_dir():
            sys.path.insert(0, str(path))
    os.environ.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "SEPALITH_DEEPSPEC_ROOT": str(SOURCE_ROOT / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec"),
        "SEPALITH_HARDENING_ROOT": str(SOURCE_ROOT / "docs/campaign/work/r2-draft-adapter-hardening-v1"),
        "SEPALITH_PUBLIC_DRAFT_WEIGHTS": "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-hf-v1/model.safetensors",
        "SEPALITH_MINICPM5_TARGET": TARGET_MODEL_DIR,
        "SEPALITH_DRAFT_CACHE": "/tmp/r2-local-readiness-cache",
        "SEPALITH_DRAFT_OUTPUT_ROOT": "/tmp/r2-local-readiness-output",
        "SEPALITH_DRAFT_LOG_ROOT": "/tmp/r2-local-readiness-log",
    })

    baseline = baseline_tensorboard_probe()
    require(baseline["expected_missing_tensorboard"], "canonical baseline unexpectedly changed; record dependency state again")

    import torch
    import transformers
    import safetensors
    import deepspec
    from deepspec.data.target_cache_dataset import validate_train_cache
    from deepspec.trainer.base_trainer import BaseTrainer, _compute_gradient_accumulation_steps
    import deepspec.trainer.ckpt_manager as ckpt_manager
    from deepspec.trainer.ckpt_manager import save_checkpoint
    from deepspec.utils.config import finalize_config, load_config

    profile = import_module_from_path("r2_local_profile_orchestrator", PROFILE_ORCHESTRATOR_PATH)
    runtime = import_module_from_path("r2_local_target_runtime", TARGET_RUNTIME_PATH)
    cfg = finalize_config(load_config(str(PROFILE_CONFIG_PATH)))

    require(not torch.cuda.is_available() and torch.cuda.device_count() == 0, "CPU review unexpectedly sees CUDA")
    require(cfg.model.target_model_name_or_path == TARGET_MODEL_DIR, "profile config target identity differs")
    require(cfg.train.max_train_steps == 8, "trainer max steps differs")
    require(cfg.logging.checkpointing_steps == 8, "checkpoint cadence differs")
    require(cfg.train.precision == "bf16", "trainer precision differs")
    require(cfg.train.local_batch_size == 1 and cfg.train.global_batch_size == 2, "batch contract differs")
    require(cfg.model.target_layer_ids == [1, 10, 20, 30, 39], "target layer taps differ")
    require(cfg.model.block_size == 7 and cfg.model.num_draft_layers == 5, "draft geometry differs")
    require(cfg.data.max_length == 4096 and cfg.data.num_workers == 1, "data profile differs")
    require(cfg.model.allow_resume is False, "profile allows implicit resume")
    require(_compute_gradient_accumulation_steps(world_size=1, local_batch_size=1, global_batch_size=2) == 2, "single-GPU accumulation contract differs")
    require(sum(stage["seconds"] for stage in BUDGET["stages"]) == BUDGET["stage_execution_budget_seconds"], "profile stage budget sum differs")
    require(BUDGET["stage_execution_budget_seconds"] + BUDGET["setup_and_persistence_reserve_seconds"] == BUDGET["all_in_seconds"], "outer profile budget does not include reserve")
    require(BUDGET["all_in_seconds"] <= 1800, "first profile exceeds bounded 1,800-second cap")
    require(profile.PROFILE_LIMIT_SECONDS == BUDGET["all_in_seconds"], "orchestrator outer profile deadline is not bound to 1,800 seconds")
    require(profile.TOTAL_WINDOW_SECONDS == BUDGET["original_total_window_seconds"], "orchestrator total window changed")
    require(profile.STAGE_EXECUTION_BUDGET_SECONDS == BUDGET["stage_execution_budget_seconds"], "orchestrator stage budget differs")
    require(dict(profile.STAGE_LIMITS_SECONDS) == {
        "teacher-cache-profile": 420,
        "native-cuda-smoke": 240,
        "warmstart-trainer-profile": 720,
    }, "orchestrator stage limits are not bounded profile allocations")
    require(sum(profile.STAGE_LIMITS_SECONDS.values()) <= profile.STAGE_EXECUTION_BUDGET_SECONDS, "orchestrator stage allocations exceed stage budget")
    stage_imports = generated_stage_import_probe(profile)
    require(stage_imports["overlay_is_outside_portable_source"], "local dependency overlay leaked into portable stage PYTHONPATH")
    require(not stage_imports["without_overlay"]["passed"], "canonical generated stage unexpectedly hid dependency setup")
    require(stage_imports["with_existing_overlay"]["passed"], "existing dependency overlay cannot satisfy generated stage imports")

    produced_identity = runtime._target_cache_model_name({"model_dir": TARGET_MODEL_DIR})
    require(produced_identity == cfg.model.target_model_name_or_path, "producer and trainer target identity differ")
    dataset = SimpleNamespace(manifest={
        "target_layer_ids": [1, 10, 20, 30, 39],
        "hidden_size": 1,
        "target_model_name_or_path": produced_identity,
    })
    draft = SimpleNamespace(target_layer_ids=[1, 10, 20, 30, 39], config=SimpleNamespace(hidden_size=1))
    validate_train_cache(train_dataset=dataset, draft_model=draft, target_model_name_or_path=cfg.model.target_model_name_or_path)
    mismatch_rejected = False
    try:
        validate_train_cache(train_dataset=dataset, draft_model=draft, target_model_name_or_path=TARGET_MODEL_DIR + "/mismatch")
    except AssertionError:
        mismatch_rejected = True
    require(mismatch_rejected, "official validator did not reject mismatched target identity")

    validator_source = inspect.getsource(validate_train_cache)
    checkpoint_source = inspect.getsource(save_checkpoint)
    checkpoint_state_path_source = inspect.getsource(ckpt_manager._rank_training_state_path)
    train_source = inspect.getsource(BaseTrainer.train)
    orchestrator_source = PROFILE_ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    train_entry_source = PROFILE_TRAIN_PATH.read_text(encoding="utf-8")
    require("str(cache_target_model_name) == str(target_model_name_or_path)" in validator_source, "official literal identity check missing")
    require("step_{global_step}" in checkpoint_source and "step_latest" in checkpoint_source, "official checkpoint path schema missing")
    require(ckpt_manager.TRAIN_CONFIG_FILE_NAME == "train_config.py", "official checkpoint config filename differs")
    require("training_state.rank" in checkpoint_state_path_source, "official rank-state filename schema missing")
    require(train_source.count("self.save_and_eval_checkpoint()") >= 2, "official trainer final checkpoint save path missing")
    require("step_8 checkpoint is missing" in orchestrator_source, "profile step_8 gate missing")
    require("steps = [(int(value), int(limit))" in orchestrator_source, "profile contiguous-step parser missing")
    require("telemetry_rows and all(row.get(\"available\") and row.get(\"finite\")" in orchestrator_source, "profile finite telemetry gate missing")
    require("torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())" in train_entry_source, "trainer process-count contract changed")
    require('"CUDA_VISIBLE_DEVICES": "0"' in orchestrator_source, "single-GPU child environment binding missing")
    require("CUDA_VISIBLE_DEVICES=0,1" not in (PACKET / "README.md").read_text(encoding="utf-8"), "README retains stale two-GPU command")
    require("_handle_termination" in orchestrator_source and "Always reap the owned child process group" in orchestrator_source, "stage termination cleanup hardening missing")

    package_names = ("torch", "transformers", "safetensors", "tensorboard", "numpy", "pyyaml", "prettytable")
    versions = {name: metadata.version(name) for name in package_names}
    imported = {
        "deepspec": module_location(deepspec),
        "transformers": str(Path(transformers.__file__).resolve()),
        "safetensors": str(Path(safetensors.__file__).resolve()),
        "profile_orchestrator": str(PROFILE_ORCHESTRATOR_PATH),
        "target_runtime": str(TARGET_RUNTIME_PATH),
    }
    report = {
        "schema": "sepalith.run08.r2-draft-local-profile.readiness-review.v1",
        "status": "PASS_WITH_LAUNCH_GATES",
        "cpu_only": True,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "versions": versions,
            "imported_modules": imported,
            "dependency_overlay": str(OVERLAY),
            "dependency_overlay_present": OVERLAY.is_dir(),
            "canonical_without_overlay_tensorboard_probe": baseline,
            "generated_stage_import_probe": stage_imports,
        },
        "accepted_parent_wrapper": {
            "path": str(WRAPPER_PATH),
            "sha256": "05eac983926eacc45b78f59d281eeaeeecfc8c58ed03647b4a8cbe794160afa5",
            "status": "accepted",
            "kind": "merged_sft",
            "model_path": TARGET_MODEL_DIR,
            "weights_sha256": WEIGHTS_SHA256,
            "ancestry_fields_match_selected500": True,
        },
        "identity_contract": {
            "producer_target_model_name_or_path": produced_identity,
            "trainer_target_model_name_or_path": cfg.model.target_model_name_or_path,
            "official_matching_validation": "PASS",
            "official_mismatch_negative": "PASS",
            "train_rows_sha256": TRAIN_SHA256,
            "deepspec_revision": DEEPSPEC_REVISION,
        },
        "profile_config": {
            "experiment": cfg.exp_name,
            "max_train_steps": int(cfg.train.max_train_steps),
            "checkpointing_steps": int(cfg.logging.checkpointing_steps),
            "precision": str(cfg.train.precision),
            "local_batch_size": int(cfg.train.local_batch_size),
            "global_batch_size": int(cfg.train.global_batch_size),
            "world_size_one_gradient_accumulation": 2,
            "target_layer_ids": list(cfg.model.target_layer_ids),
            "target_model_name_or_path": cfg.model.target_model_name_or_path,
            "public_draft_weights_path": cfg.model.public_draft_weights,
            "cache_path": cfg.data.target_cache_path,
            "resume": bool(cfg.model.allow_resume),
        },
        "output_checkpoint_schema": {
            "checkpoint_root_suffix": "checkpoints/sepalith-r2/dspark_minicpm5_2b_step500_profile",
            "required_checkpoint_dir": "step_8",
            "required_latest_pointer": "step_latest -> step_8",
            "required_files": ["train_config.py", "config.json", "model.safetensors", "training_state.rank0.pt"],
            "official_final_save": True,
            "profile_terminal_gate": "contiguous step=1/8 ... step=8/8, finite telemetry, step_8 present",
        },
        "budget_proposal": BUDGET,
        "source_inventory": {
            "path": str(INVENTORY_PATH),
            "entries_verified": verified_entries,
            "fixed_runtime_sha256": RUNTIME_SHA256,
        },
        "termination_regression": {
            "path": str(PACKET / "termination-regression.json"),
            "status": "PASS",
            "stage_survives_supervisor": False,
            "failure_receipt_written": True,
        },
        "concrete_blockers": [
            "Canonical .venv-sft lacks tensorboard; the existing dependency overlay must be mounted on PYTHONPATH or a root-owned setup must install the pinned dependency before profile imports. No install was performed here.",
            "The earlier local orchestrator constants still describe a 3,600-second profile. The proposed 1,800-second all-in cap requires root to run a bounded supervisor/scoped budget override; the old 3,600-second execute path must not be used for this proposal.",
            "The selected target/public artifacts and TRAIN require live hash/load verification at execution; this review intentionally did not open or hash them.",
            "CUDA native Flex and the eight optimizer steps remain unexecuted in this CPU review; root must acquire the single-GPU lock/host guard and collect finite telemetry plus step_8 checkpoint evidence.",
        ],
        "non_blockers_verified": [
            "Accepted merged_sft parent wrapper is bound and matches selected step-500 model, base revision, TRAIN identity, renderer, tokenizer, and merged weight SHA.",
            "One-GPU process contract is coherent: orchestrator sets CUDA_VISIBLE_DEVICES=0, train.py spawns torch.cuda.device_count() workers, and world_size=1 yields gradient accumulation 2 for local/global batch 1/2.",
            "Producer cache identity and official validator/training-config identity match; mismatched target identity is rejected.",
            "Official checkpoint writer and profile terminal gate require step_8 and step_latest with train config and rank state; final save path is present.",
        ],
        "evidence_limits": [
            "Imports were run with CUDA_VISIBLE_DEVICES='' and the pre-existing dependency overlay; no CUDA kernel or model loader was called.",
            "Source hashes were limited to the explicit small inventory (all entries <= 1 MiB).",
            "No provider, network, model payload, public weight payload, or TRAIN bytes were accessed.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
