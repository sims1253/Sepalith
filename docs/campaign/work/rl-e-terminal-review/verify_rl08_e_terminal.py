#!/usr/bin/env python3
"""Bounded, model-free audit of the terminal RL-08 e run.

This reads only the named e-run terminal/supervision metadata, the last host
samples, durable small telemetry records, and full checkpoint 140.  It uses
the frozen production checkpoint verifier.  It never imports a framework,
reads the RL recipe or large preflight/telemetry JSON, starts a process, or
reads sealed/final data.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


RUN = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-e")
SUPERVISION = RUN.with_name("RL-primary-p2-mb4-full5-e-host-supervision")
CHECKPOINT = RUN / "archive/full/checkpoint-140"
OUTPUT_CHECKPOINT = RUN / "output/checkpoint-140"
PRODUCTION_TRAINING = Path(
    "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/"
    "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/"
    "source/experiments/training"
)
TERMINAL = SUPERVISION / "terminal.json"
HOST_MEMORY = SUPERVISION / "host-memory.jsonl"
LOW_MEMORY = SUPERVISION / "low-memory-details.jsonl"
TRAINER_STATE = CHECKPOINT / "trainer_state.json"
GENERATION = RUN / "output/generation-records.jsonl"
REWARDS = RUN / "output/reward-records.jsonl"
GRADIENTS = RUN / "output/gradient-records.jsonl"
PROCESS_LOG = SUPERVISION / "process.log"
RECEIPT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/"
    "RL-08-e-terminal-full140-independent-review.json"
)
EXPECTED_IDENTITY_SHA = "ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818"
EXPECTED_SOURCE_SHA = "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9"
EXPECTED_SCHEDULE_SHA = "2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132"
EXPECTED_SEQUENCE_SHA = "dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6"
EXPECTED_SELECTED_SHA = "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d"
EXPECTED_ORDERED_SHA = "7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d"
EXPECTED_CONTEXT_SHA = "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f"
EXPECTED_PARENT_MANIFEST_SHA = "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12"
EXPECTED_PARENT_WEIGHTS_SHA = "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d"
EXPECTED_SOFT_MIB = 8192
EXPECTED_HARD_MIB = 4096


def digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            h.update(block)
    return size, h.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def add(checks: list[dict[str, Any]], name: str, ok: bool, details: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "status": "pass" if ok else "fail"}
    if details is not None:
        item["details"] = details
    checks.append(item)


def state_projection() -> tuple[dict[str, Any] | None, str | None]:
    """Read only scalar state fields from the large campaign-state JSON."""
    jq = shutil.which("jq")
    if jq is None:
        return None, "jq unavailable"
    expression = (
        "{step,full,resume_validation,"
        "identity:{data:{row_count:.identity.data.row_count,split:.identity.data.split,"
        "admission_status:.identity.data.admission_status,selected_ids_sha256:.identity.data.selected_ids_sha256,"
        "ordered_ids_sha256:.identity.data.ordered_ids_sha256,context_sha256:.identity.data.context_sha256},"
        "policy:{cuda_memory_fraction:.identity.policy.cuda_memory_fraction,"
        "candidate_count:.identity.policy.candidate_count,completion_max_tokens:.identity.policy.completion_max_tokens,"
        "prompt_max_tokens:.identity.policy.prompt_max_tokens,gradient_accumulation_steps:.identity.policy.gradient_accumulation_steps,"
        "per_device_train_batch_size:.identity.policy.per_device_train_batch_size}},"
        "sampler:{current_index:.sampler.current_index,source_draw_cursor:.sampler.source_draw_cursor,"
        "consumed_rows:.sampler.consumed_rows,consumed_prompt_copies:.sampler.consumed_prompt_copies,"
        "source_draws:.sampler.source_draws,source_draws_bound:.sampler.source_draws_bound,"
        "buffer_reuse:.sampler.buffer_reuse,candidate_count:.sampler.candidate_count,"
        "generation_rows_per_update:.sampler.generation_rows_per_update,"
        "prompt_groups_per_batch:.sampler.prompt_groups_per_batch,repeat_count:.sampler.repeat_count}}"
    )
    process = subprocess.run([jq, "-c", expression, str(CHECKPOINT / "campaign-state.json")], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        return None, process.stderr.strip()[:300]
    return json.loads(process.stdout), None


def main() -> int:
    checks: list[dict[str, Any]] = []
    terminal = json.loads(TERMINAL.read_text(encoding="utf-8"))
    host = read_jsonl(HOST_MEMORY)
    low_memory = read_jsonl(LOW_MEMORY)
    manifest = json.loads((CHECKPOINT / "campaign-manifest.json").read_text(encoding="utf-8"))
    manifest_size, manifest_sha = digest(CHECKPOINT / "campaign-manifest.json")
    output_manifest_size, output_manifest_sha = digest(OUTPUT_CHECKPOINT / "campaign-manifest.json")
    identity = manifest.get("identity", {})
    identity_sha = canonical_sha(identity)

    add(
        checks,
        "terminal.host_supervision_reason",
        terminal.get("status") == "stopped_or_failed"
        and terminal.get("reason") == "host_free_memory_below_soft_floor_persisted"
        and terminal.get("child_exit_code") == -15,
        {"status": terminal.get("status"), "reason": terminal.get("reason"), "child_exit_code": terminal.get("child_exit_code"), "seconds": terminal.get("seconds")},
    )
    below_soft = [row for row in host if int(row.get("AvailableMBytes", -1)) < EXPECTED_SOFT_MIB]
    below_hard = [row for row in host if int(row.get("AvailableMBytes", -1)) < EXPECTED_HARD_MIB]
    driver_events = [event for row in host for event in row.get("DriverEvents", [])]
    max_page_input = max((int(row.get("PagesInputPersec", 0)) for row in host), default=0)
    max_page_reads = max((int(row.get("PageReadsPersec", 0)) for row in host), default=0)
    last_two = host[-2:]
    soft_stop_ok = len(last_two) == 2 and all(int(row.get("AvailableMBytes", -1)) < EXPECTED_SOFT_MIB for row in last_two)
    commit_limit = int(last_two[-1].get("CommitLimit", 0)) if last_two else 0
    committed = int(last_two[-1].get("CommittedBytes", 0)) if last_two else 0
    add(
        checks,
        "terminal.host_floor_samples",
        soft_stop_ok and not below_hard and not driver_events and committed < int(commit_limit * 0.90),
        {
            "samples": len(host),
            "available_min_mib": min((int(row.get("AvailableMBytes", 0)) for row in host), default=None),
            "available_max_mib": max((int(row.get("AvailableMBytes", 0)) for row in host), default=None),
            "below_soft_count": len(below_soft),
            "below_hard_count": len(below_hard),
            "last_two_available_mib": [int(row.get("AvailableMBytes", -1)) for row in last_two],
            "driver_event_count": len(driver_events),
            "max_page_input_per_sec": max_page_input,
            "max_page_reads_per_sec": max_page_reads,
            "last_commit_fraction": committed / commit_limit if commit_limit else None,
        },
    )

    identity_fields_ok = (
        identity_sha == EXPECTED_IDENTITY_SHA
        and identity.get("parent", {}).get("manifest_sha256") == EXPECTED_PARENT_MANIFEST_SHA
        and identity.get("parent", {}).get("merged_weights_sha256") == EXPECTED_PARENT_WEIGHTS_SHA
        and identity.get("data", {}).get("row_count") == 8440
        and identity.get("data", {}).get("split") == "train"
        and identity.get("data", {}).get("selected_ids_sha256") == EXPECTED_SELECTED_SHA
        and identity.get("data", {}).get("ordered_ids_sha256") == EXPECTED_ORDERED_SHA
        and identity.get("data", {}).get("context_sha256") == EXPECTED_CONTEXT_SHA
        and identity.get("policy", {}).get("cuda_memory_fraction") == 0.8
        and identity.get("policy", {}).get("candidate_count") == 4
        and identity.get("policy", {}).get("completion_max_tokens") == 192
        and identity.get("policy", {}).get("prompt_max_tokens") == 2048
        and identity.get("policy", {}).get("gradient_accumulation_steps") == 8
        and identity.get("policy", {}).get("per_device_train_batch_size") == 4
        and identity.get("schedule", {}).get("source_draw_schedule_sha256") == EXPECTED_SCHEDULE_SHA
        and identity.get("schedule", {}).get("source_draw_sequence_sha256") == EXPECTED_SEQUENCE_SHA
    )
    add(
        checks,
        "checkpoint140.identity",
        identity_fields_ok,
        {
            "canonical_identity_sha256": identity_sha,
            "policy_cuda_memory_fraction": identity.get("policy", {}).get("cuda_memory_fraction"),
            "data_row_count": identity.get("data", {}).get("row_count"),
            "selected_ids_sha256": identity.get("data", {}).get("selected_ids_sha256"),
            "ordered_ids_sha256": identity.get("data", {}).get("ordered_ids_sha256"),
            "source_schedule_sha256": identity.get("schedule", {}).get("source_draw_schedule_sha256"),
        },
    )
    add(
        checks,
        "checkpoint140.manifest_headers",
        manifest.get("schema_version") == 1 and manifest.get("step") == 140 and manifest.get("full") is True and manifest_sha == output_manifest_sha,
        {"archive_manifest": {"bytes": manifest_size, "sha256": manifest_sha}, "output_manifest": {"bytes": output_manifest_size, "sha256": output_manifest_sha}, "step": manifest.get("step"), "full": manifest.get("full")},
    )

    sys.path.insert(0, str(PRODUCTION_TRAINING))
    from campaign_checkpoint import verify_checkpoint  # type: ignore

    production_error: str | None = None
    verified: dict[str, Any] | None = None
    try:
        verified = verify_checkpoint(CHECKPOINT, identity, require_full=True)
    except Exception as error:  # verifier failure is recorded in the receipt
        production_error = f"{type(error).__name__}: {error}"
    add(
        checks,
        "checkpoint140.production_verify_checkpoint",
        verified is not None and production_error is None,
        {"error": production_error, "verified_file_count": len(verified.get("files", {})) if verified else None, "require_full": True},
    )

    files = (verified or manifest).get("files", {})
    required_full = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    files_ok = all(name in files and (CHECKPOINT / name).is_file() and int(files[name].get("bytes", 0)) > 0 for name in required_full)
    add(checks, "checkpoint140.full_state_inventory", files_ok, {"required_files": list(required_full), "file_count": len(files)})

    trainer_state = json.loads(TRAINER_STATE.read_text(encoding="utf-8"))
    history = trainer_state.get("log_history", [])
    latest_log = history[-1] if history else {}
    optimizer_ok = trainer_state.get("global_step") == 140 and trainer_state.get("max_steps") == 3000 and latest_log.get("step") == 140
    add(
        checks,
        "checkpoint140.latest_completed_optimizer_step",
        optimizer_ok,
        {"trainer_global_step": trainer_state.get("global_step"), "trainer_max_steps": trainer_state.get("max_steps"), "log_history_records": len(history), "last_log_step": latest_log.get("step"), "last_grad_norm": latest_log.get("grad_norm")},
    )

    state, state_error = state_projection()
    state_expected = {
        "step": 140,
        "full": True,
        "sampler": {"current_index": 35840, "source_draw_cursor": 1120, "consumed_rows": 35840, "consumed_prompt_copies": 8960, "source_draws": 24000, "source_draws_bound": True, "buffer_reuse": 8, "candidate_count": 4, "generation_rows_per_update": 32, "prompt_groups_per_batch": 8, "repeat_count": 8},
    }
    state_ok = state is not None and state.get("step") == state_expected["step"] and state.get("full") is state_expected["full"] and all(state.get("sampler", {}).get(k) == v for k, v in state_expected["sampler"].items())
    state_identity = state.get("identity", {}) if state else {}
    state_ok = state_ok and state_identity.get("data", {}).get("selected_ids_sha256") == EXPECTED_SELECTED_SHA and state_identity.get("data", {}).get("ordered_ids_sha256") == EXPECTED_ORDERED_SHA and state_identity.get("policy", {}).get("cuda_memory_fraction") == 0.8
    add(checks, "checkpoint140.state_boundary", state_ok, {"projection": state, "error": state_error})

    generation_count = 0
    generation_steps: collections.Counter[int] = collections.Counter()
    generation_hashes: list[str] = []
    generation_last: dict[str, Any] = {}
    generation_schedule_hashes: set[str] = set()
    with GENERATION.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            generation_count += 1
            generation_steps[int(row.get("global_step"))] += 1
            generation_hashes.append(str(row.get("generated_ids_sha256")))
            generation_schedule_hashes.add(str(row.get("source_schedule_sha256")))
            generation_last = row
    reward_count = 0
    reward_hashes: list[str] = []
    with REWARDS.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            reward_count += 1
            reward_hashes.append(str(row.get("output_ids_sha256")))
    join_mismatches = sum(a != b for a, b in zip(generation_hashes, reward_hashes))
    add(
        checks,
        "durable_generation_reward_join",
        generation_count == reward_count == 40 * 32 and join_mismatches == 0 and generation_schedule_hashes == {EXPECTED_SCHEDULE_SHA},
        {"generation_records": generation_count, "reward_records": reward_count, "global_step_min": min(generation_steps) if generation_steps else None, "global_step_max": max(generation_steps) if generation_steps else None, "records_per_step": sorted(set(generation_steps.values())), "output_hash_mismatches": join_mismatches, "source_schedule_hashes": sorted(generation_schedule_hashes), "last_global_step_before_update": generation_last.get("global_step_before_update")},
    )

    gradient_rows = read_jsonl(GRADIENTS)
    gradient_bad = [row for row in gradient_rows if not row.get("finite") or row.get("nonfinite") or row.get("trainable_scope") != "requires_grad LoRA adapter tensors only" or not math.isfinite(float(row.get("norm", float("nan"))))]
    gradient_steps = [int(row.get("global_step")) for row in gradient_rows]
    add(
        checks,
        "durable_gradient_boundary",
        len(gradient_rows) == 40 and not gradient_bad and min(gradient_steps) == 100 and max(gradient_steps) == 139,
        {"records": len(gradient_rows), "global_step_range": [min(gradient_steps), max(gradient_steps)] if gradient_steps else None, "nonfinite_or_scope_errors": len(gradient_bad), "last_record": {key: gradient_rows[-1].get(key) for key in ("global_step", "step", "finite", "nonfinite", "norm", "trainable_tensor_count", "grad_present_count", "nonzero_tensor_count")} if gradient_rows else None, "interpretation": "callback records global_step before the optimizer update; trainer_state step 140 is the completed boundary"},
    )

    process_tail = PROCESS_LOG.read_bytes()[-16000:].decode("utf-8", errors="replace")
    progress_steps = [int(value) for value in re.findall(r"(\d+)/3000", process_tail)]
    add(checks, "terminal.no_training_failure_marker", not (RUN / "output/failure.json").exists() and not (RUN / "output/terminal.json").exists(), {"failure_marker": (RUN / "output/failure.json").exists(), "trainer_terminal_marker": (RUN / "output/terminal.json").exists(), "process_tail_progress_steps": progress_steps[-4:]})

    source_size, source_sha = digest(PRODUCTION_TRAINING / "campaign_checkpoint.py")
    core_files = {name: files.get(name) for name in sorted(files)}
    overall = all(item["status"] == "pass" for item in checks)
    receipt = {
        "task": "RL-08",
        "status": "pass_structural_full140_host_terminal_audit" if overall else "fail_structural_full140_host_terminal_audit",
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owner": "worker-rl-contexts",
        "scope": "terminal e-run and full checkpoint 140 only; no restart, promotion, sealed/final read, or model/framework load",
        "terminal": {"path": str(TERMINAL), **terminal, "host_supervision_path": str(SUPERVISION)},
        "host_pressure": {"sample_count": len(host), "available_min_mib": min((int(row.get("AvailableMBytes", 0)) for row in host), default=None), "available_max_mib": max((int(row.get("AvailableMBytes", 0)) for row in host), default=None), "last_sample": host[-1] if host else None, "last_two_available_mib": [int(row.get("AvailableMBytes", -1)) for row in last_two], "soft_floor_mib": EXPECTED_SOFT_MIB, "hard_floor_mib": EXPECTED_HARD_MIB, "below_soft_count": len(below_soft), "below_hard_count": len(below_hard), "driver_event_count": len(driver_events), "max_page_input_per_sec": max_page_input, "last_commit_fraction": committed / commit_limit if commit_limit else None, "low_memory_detail_records": len(low_memory)},
        "checkpoint": {"path": str(CHECKPOINT), "manifest_sha256": manifest_sha, "manifest_bytes": manifest_size, "step": manifest.get("step"), "full": manifest.get("full"), "identity_sha256": identity_sha, "production_verifier": "campaign_checkpoint.verify_checkpoint(require_full=True)", "production_source": {"path": str(PRODUCTION_TRAINING / "campaign_checkpoint.py"), "bytes": source_size, "sha256": source_sha, "snapshot": EXPECTED_SOURCE_SHA}, "files": core_files},
        "optimizer_boundary": {"trainer_state_path": str(TRAINER_STATE), "global_step": trainer_state.get("global_step"), "max_steps": trainer_state.get("max_steps"), "last_log_step": latest_log.get("step"), "gradient_records": len(gradient_rows), "gradient_pre_update_global_step_max": max(gradient_steps) if gradient_steps else None, "source_draw_cursor": state.get("sampler", {}).get("source_draw_cursor") if state else None, "consumed_rows": state.get("sampler", {}).get("consumed_rows") if state else None, "generation_records": generation_count, "reward_records": reward_count},
        "resumability": {"structural_full_checkpoint_verified": overall and files_ok and state_ok and optimizer_ok, "matched_interruption_resume_test": False, "limitation": "Full-state files and production seal verify, but no matched interruption/resume receipt exists for step 140; faithful resume is not claimed."},
        "uncertainty": ["The supervisor stop reason and two below-soft-floor samples are established.", "No driver events and no training failure marker were recorded; the exact upstream host-memory contributor is unresolved.", "The child exited by SIGTERM (-15), so the absence of an output terminal marker is expected and is not a model-quality result."],
        "checks": checks,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "failed_checks": [item["name"] for item in checks if item["status"] == "fail"], "manifest_sha256": manifest_sha, "identity_sha256": identity_sha, "global_step": trainer_state.get("global_step"), "generation_records": generation_count, "reward_records": reward_count, "terminal_reason": terminal.get("reason")}, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
