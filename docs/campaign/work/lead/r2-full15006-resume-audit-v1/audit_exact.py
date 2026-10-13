#!/usr/bin/env python3
"""Read-only structural and runtime-path audit for full15006 checkpoint 480."""
from __future__ import annotations
import copy, hashlib, importlib, inspect, json, os, stat, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
SOURCE = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-full15006-v1/attempts/28a06420683d4035b60941347955bb3e/source")
RECIPE = PLAN / "docs/campaign/work/lead/r2-expanded-sft-full15008-v1/recipe.json"
SOURCE_MANIFEST = PLAN / "docs/campaign/work/lead/r2-expanded-sft-full15008-v1/source-manifest.json"
CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-full15006-a/full/checkpoint-480")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    os.nice(10)
    available = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(available)})
    recipe = json.loads(RECIPE.read_text())
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_checks = []
    for record in source_manifest["files"]:
        path = SOURCE / record["path"]
        mode = stat.S_IMODE(path.stat().st_mode)
        source_checks.append({"path": record["path"], "sha256": sha(path), "expected_sha256": record["sha256"],
                              "mode": mode, "expected_mode": record["mode"],
                              "pass": sha(path) == record["sha256"] and mode == record["mode"]})
    assert len(source_checks) == 12 and all(x["pass"] for x in source_checks)
    sys.path.insert(0, str(SOURCE / "experiments/training"))
    import campaign_checkpoint
    import campaign_expanded_sft
    started = time.monotonic_ns()
    manifest = campaign_checkpoint.verify_checkpoint(CHECKPOINT, recipe["identity"], require_full=True)
    verify_ns = time.monotonic_ns() - started
    state = json.loads((CHECKPOINT / "campaign-state.json").read_text())
    trainer = json.loads((CHECKPOINT / "trainer_state.json").read_text())
    assert manifest["step"] == state["step"] == trainer["global_step"] == 480
    assert manifest["full"] is state["full"] is True
    assert manifest["identity"] == state["identity"] == recipe["identity"]
    assert state["sampler"] == {"method": "frozen sequential draw schedule",
        "split_id": "DAT-10-expanded-15006-v1", "schedule_sha256": recipe["draw_schedule"]["sha256"],
        "consumed_draws": 7680}
    import torch, transformers
    scheduler = torch.load(CHECKPOINT / "scheduler.pt", map_location="cpu", weights_only=True)
    assert scheduler["last_epoch"] == 480 and scheduler["_step_count"] == 481
    trace = {}
    for name in ("train", "_init_training_state", "_prepare_for_training", "_load_optimizer_and_scheduler",
                 "_load_rng_state", "_run_epoch"):
        fn = getattr(transformers.Trainer, name)
        body = inspect.getsource(fn)
        trace[name] = {"path": inspect.getsourcefile(fn), "start_line": inspect.getsourcelines(fn)[1],
                       "source_sha256": hashlib.sha256(body.encode()).hexdigest()}
    # Demonstrate the exact active adapter's legacy 250/500 check without loading TRAIN rows.
    overlay = copy.deepcopy(recipe)
    overlay.update(resume_from=str(CHECKPOINT), resume_milestone=480)
    original_rows = campaign_expanded_sft.validate_expanded_rows
    original_parent = campaign_expanded_sft.validate_sft_merged_parent
    campaign_expanded_sft.validate_expanded_rows = lambda _r, _rows: {"rows": 15006}
    campaign_expanded_sft.validate_sft_merged_parent = lambda _r: None
    active_gate_error = None
    try:
        campaign_expanded_sft.validate_expanded_admission(overlay, [])
    except ValueError as error:
        active_gate_error = str(error)
    finally:
        campaign_expanded_sft.validate_expanded_rows = original_rows
        campaign_expanded_sft.validate_sft_merged_parent = original_parent
    assert active_gate_error == "Task SFT resume step/full identity is inconsistent"
    compatibility = copy.deepcopy(overlay)
    source_id = recipe["identity"]["source"]
    compatibility.update(mandatory_stop_steps=[720], resume_identity_compatibility={
        "schema": "sepalith.sft.resume-source-compatibility.v1",
        "mode": "exact_same_source", "reason": "external_interruption",
        "checkpoint_step": 480,
        "checkpoint_manifest_sha256": sha(CHECKPOINT / "campaign-manifest.json"),
        "predecessor_source": source_id, "current_source": source_id,
    })
    campaign_expanded_sft.validate_expanded_rows = lambda _r, _rows: {"rows": 15006}
    campaign_expanded_sft.validate_sft_merged_parent = lambda _r: None
    compatibility_gate_error = None
    try:
        campaign_expanded_sft.validate_expanded_admission(compatibility, [])
    except KeyError as error:
        compatibility_gate_error = repr(error)
    finally:
        campaign_expanded_sft.validate_expanded_rows = original_rows
        campaign_expanded_sft.validate_sft_merged_parent = original_parent
    assert compatibility_gate_error == "KeyError('split_id')"
    active_entry = SOURCE / "experiments/training/campaign_sft.py"
    entry_body = active_entry.read_text()
    assert 'trainer.train(resume_from_checkpoint=recipe.get("resume_from"))' in entry_body
    result = {
        "schema": "sepalith.sft11.full15006.resume-audit.exact.v1", "status": "audited_read_only",
        "recipe": {"path": str(RECIPE), "sha256": sha(RECIPE), "resume_from": recipe["resume_from"],
                   "full_every": recipe["checkpoint"]["full_every"],
                   "declared_target_resume_steps": recipe["target_resume_steps"]},
        "source": {"path": str(SOURCE), "runner_snapshot_identity": source_manifest["id"],
                   "manifest_sha256": sha(SOURCE_MANIFEST), "files_verified": len(source_checks),
                   "all_manifest_hashes_and_modes_exact": True, "checks": source_checks,
                   "ignored_runtime_pycache_not_in_runner_identity": True},
        "checkpoint": {"path": str(CHECKPOINT), "manifest_sha256": sha(CHECKPOINT / "campaign-manifest.json"),
                       "step": 480, "full": True, "manifest_files_verified": len(manifest["files"]),
                       "all_file_hashes_and_bytes_exact": True, "verify_elapsed_ns": verify_ns,
                       "state_identity_exact": True, "trainer_global_step": trainer["global_step"],
                       "sampler": state["sampler"]},
        "learning_rate": {"trainer_log_step_480": trainer["log_history"][-1]["learning_rate"],
                          "scheduler_last_epoch": scheduler["last_epoch"],
                          "scheduler_step_count": scheduler["_step_count"],
                          "scheduler_next_restored_lr": scheduler["_last_lr"],
                          "base_lrs": scheduler["base_lrs"]},
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda_initialized": torch.cuda.is_initialized(), "trainer_source_trace": trace,
                    "campaign_train_call": "trainer.train(resume_from_checkpoint=recipe.get('resume_from'))",
                    "save_only_model": False, "ignore_data_skip": False,
                    "sampler_class": "torch.utils.data.SequentialSampler"},
        "restore_trace": [
            "Preflight verifies the sealed full checkpoint inventory and exact identity before model construction.",
            "Trainer.train loads adapter/model state and trainer_state.json from resume_from_checkpoint.",
            "Trainer._prepare_for_training creates optimizer/scheduler and calls _load_optimizer_and_scheduler on optimizer.pt and scheduler.pt.",
            "Trainer._run_epoch skips global_step*gradient_accumulation deterministic sequential microbatches because ignore_data_skip is false, then reloads rng_state.pth at the resume boundary.",
            "campaign-state sampler metadata is validated against split, schedule hash, and consumed_draws=step*16; it is not deserialized as a custom sampler object.",
        ],
        "active_gate": {"checkpoint_480_structurally_valid": True,
                        "expanded_target_only_policy_declares_480": 480 in recipe["target_resume_steps"],
                        "expanded_admission_accepts_without_patch": False,
                        "observed_error": active_gate_error,
                        "compatibility_route_error": compatibility_gate_error,
                        "causes": [
                            "validate_expanded_admission retains legacy step membership (250, 500, compatibility_step), inconsistent with this recipe's 240/480/720 list.",
                            "After exact_same_source compatibility admits step 480, the same validator indexes schedule['split_id'], but the pinned draw manifest has no top-level split_id; the bound split lives in identity.data.split_id.",
                        ]},
        "limitations": ["No checkpoint tensor was compared to a full 5090 continuation in this CPU audit.",
                        "optimizer.pt and rng_state.pth were hash-verified but not unpickled by the audit; the Trainer restore path is covered by the toy interruption test."],
    }
    (HERE / "exact-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
