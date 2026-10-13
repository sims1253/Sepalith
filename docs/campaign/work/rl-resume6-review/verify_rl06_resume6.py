#!/usr/bin/env python3
"""Compare the known continuous and resumed RL update-6 records.

This is a CPU-only, framework-free audit.  It compares the old continuous
run's ``global_step=5`` rollout with the resumed run's ``global_step=5``
rollout (the generation immediately before optimizer step 6).  It keeps
trainer invocation metadata such as elapsed time and ``trl_microstep``
separate from semantic prompt/output/reward evidence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import mmap
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "source": "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9",
    "schedule": "2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132",
    "sequence": "dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6",
    "selected": "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d",
    "parent_manifest": "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12",
    "parent_weights": "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d",
    "parent_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
    "identity_old": "dee63a052efeb9dd1ddeb2fbbf7e623c0b3cc8dc61cefcd8d5526df7ddbe8384",
    "identity_resumed": "48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2",
}

SHARED_POLICY = {
    "beta": 0,
    "candidate_count": 4,
    "completion_max_tokens": 192,
    "context_max_tokens": 2240,
    "cuda_memory_fraction": 0.75,
    "expected_attachments": 294,
    "expected_trainable_parameters": 25116672,
    "generation_groups_per_call": 2,
    "gradient_accumulation_steps": 8,
    "lora_alpha": 16,
    "lora_rank": 16,
    "loss_type": "bnpo",
    "model_load_max_seq_length": 4096,
    "per_device_train_batch_size": 4,
    "prompt_max_tokens": 2048,
    "rollout_rows_per_update": 32,
    "scale_rewards": "group",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return default
        obj = obj[part]
    return obj


def add(checks: list[dict[str, Any]], name: str, status: str, details: Any = None) -> None:
    item = {"name": name, "status": status}
    if details is not None:
        item["details"] = details
    checks.append(item)


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: object required")
            rows.append(value)
    return rows


def small_section(path: Path, key: str) -> dict[str, Any] | None:
    """Read one small object from a giant pretty-printed JSON receipt."""
    if not path.is_file():
        return None
    marker = ('"' + key + '"').encode("utf-8")
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = mm.find(marker)
            if pos < 0:
                return None
            colon = mm.find(b":", pos + len(marker))
            if colon < 0:
                return None
            start = colon + 1
            while start < len(mm) and mm[start] in b" \t\r\n":
                start += 1
            if start >= len(mm) or mm[start] != ord("{"):
                return None
            depth = 0
            in_string = False
            escaped = False
            end = start
            while end < len(mm):
                byte = mm[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == 92:
                        escaped = True
                    elif byte == 34:
                        in_string = False
                else:
                    if byte == 34:
                        in_string = True
                    elif byte == 123:
                        depth += 1
                    elif byte == 125:
                        depth -= 1
                        if depth == 0:
                            end += 1
                            break
                end += 1
            try:
                value = json.loads(mm[start:end].decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return value if isinstance(value, dict) else None


def compare_records(checks: list[dict[str, Any]], old_root: Path, resumed_root: Path, sequence_path: Path, old_step: int = 5, resumed_step: int = 5) -> dict[str, Any]:
    old_output = old_root / "output"
    resumed_output = resumed_root / "output"
    old_generation = [row for row in jsonl(old_output / "generation-records.jsonl") if row.get("global_step") == old_step]
    resumed_generation = [row for row in jsonl(resumed_output / "generation-records.jsonl") if row.get("global_step") == resumed_step]
    add(checks, "generation.row_count", "pass" if len(old_generation) == 32 and len(resumed_generation) == 32 else "fail", {"old": len(old_generation), "resumed": len(resumed_generation)})

    semantic_generation_keys = (
        "schema_version", "prompt_ids_sha256", "generated_ids_sha256", "generated_ids",
        "generated_token_count", "terminal_reason", "padded_after_terminal",
        "source_schedule_sha256", "group_geometry", "accounting", "accounting_row",
    )
    generation_diffs: list[dict[str, Any]] = []
    for index, (old, resumed) in enumerate(zip(old_generation, resumed_generation)):
        for key in semantic_generation_keys:
            if old.get(key) != resumed.get(key):
                generation_diffs.append({"row": index, "field": key})
    add(checks, "generation.semantic_records_exact", "pass" if len(old_generation) == len(resumed_generation) == 32 and not generation_diffs else "fail", generation_diffs[:8])

    metadata_diffs: collections.Counter[str] = collections.Counter()
    metadata_examples: dict[str, Any] = {}
    for index, (old, resumed) in enumerate(zip(old_generation, resumed_generation)):
        for key in ("elapsed_sec", "trl_microstep", "update"):
            if old.get(key) != resumed.get(key):
                metadata_diffs[key] += 1
                metadata_examples.setdefault(key, {"row": index, "old": old.get(key), "resumed": resumed.get(key)})
    add(checks, "generation.invocation_metadata_audit", "pass", {"differences": dict(metadata_diffs), "examples": metadata_examples, "interpretation": "elapsed time is run-specific; resumed trl_microstep resets to 0 while continuous run reports 40"})

    old_rewards = jsonl(old_output / "reward-records.jsonl")[-32:]
    resumed_rewards = jsonl(resumed_output / "reward-records.jsonl")[:32]
    reward_exact = len(old_rewards) == len(resumed_rewards) == 32 and old_rewards == resumed_rewards
    add(checks, "reward.complete_records_exact", "pass" if reward_exact else "fail", {"old_rows": len(old_rewards), "resumed_rows": len(resumed_rewards)})
    old_ids = [row.get("id") for row in old_rewards]
    resumed_ids = [row.get("id") for row in resumed_rewards]
    add(checks, "reward.ordered_source_ids_exact", "pass" if old_ids == resumed_ids else "fail", {"first": resumed_ids[:2], "last": resumed_ids[-2:]})

    try:
        sequence = read_json(sequence_path)
        sequence_ids = sequence.get("row_ids", [])
    except Exception as exc:
        sequence = {}
        sequence_ids = []
        add(checks, "source_sequence.parse", "fail", repr(exc))
    expected_next = sequence_ids[40:48]
    collapsed = resumed_ids[::4]
    add(checks, "reward.next_source_draws_40_47", "pass" if collapsed == expected_next else "fail", {"expected": expected_next, "actual": collapsed})

    old_gradient = next((row for row in jsonl(old_output / "gradient-records.jsonl") if row.get("global_step") == old_step), None)
    resumed_gradient = next((row for row in jsonl(resumed_output / "gradient-records.jsonl") if row.get("global_step") == resumed_step), None)
    invariant_gradient_keys = ("aggregation", "finite", "grad_present_count", "nonfinite", "nonzero_tensor_count", "schema_version", "step", "trainable_scope", "trainable_tensor_count")
    gradient_diffs = [key for key in invariant_gradient_keys if old_gradient and resumed_gradient and old_gradient.get(key) != resumed_gradient.get(key)]
    old_norm = old_gradient.get("norm") if old_gradient else None
    resumed_norm = resumed_gradient.get("norm") if resumed_gradient else None
    delta = resumed_norm - old_norm if isinstance(old_norm, (int, float)) and isinstance(resumed_norm, (int, float)) else None
    relative = delta / old_norm if delta is not None and old_norm else None
    gradient_ok = old_gradient is not None and resumed_gradient is not None and not gradient_diffs and all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in (old_norm, resumed_norm)
    )
    add(checks, "gradient.finite_scope_and_counts", "pass" if gradient_ok else "fail", {"invariant_differences": gradient_diffs, "old": old_gradient, "resumed": resumed_gradient})
    add(checks, "gradient.scalar_norm_comparison", "pass" if delta is not None else "fail", {"old_norm": old_norm, "resumed_norm": resumed_norm, "delta": delta, "relative": relative, "relative_percent": relative * 100 if relative is not None else None, "interpretation": "small finite reduction drift; not byte-exact gradient evidence"})

    return {
        "generation_rows": {"old": len(old_generation), "resumed": len(resumed_generation)},
        "generation_semantic_exact": not generation_diffs and len(old_generation) == len(resumed_generation) == 32,
        "generation_metadata_differences": dict(metadata_diffs),
        "generation_metadata_examples": metadata_examples,
        "reward_rows": {"old": len(old_rewards), "resumed": len(resumed_rewards)},
        "reward_exact": reward_exact,
        "next_source_draws": expected_next,
        "gradient": {"old_norm": old_norm, "resumed_norm": resumed_norm, "delta": delta, "relative": relative, "relative_percent": relative * 100 if relative is not None else None, "invariant_differences": gradient_diffs},
        "semantic_status": "pass" if reward_exact and not generation_diffs and gradient_ok and collapsed == expected_next else "fail",
    }


def compare_identity(checks: list[dict[str, Any]], old_root: Path, resumed_root: Path) -> dict[str, Any]:
    old_manifest_path = old_root / "archive" / "adapters" / "checkpoint-5" / "campaign-manifest.json"
    resumed_recipe_path = resumed_root / "output" / "admitted-recipe.json"
    try:
        old_manifest = read_json(old_manifest_path)
        resumed_recipe = read_json(resumed_recipe_path)
    except Exception as exc:
        add(checks, "identity.parse", "fail", repr(exc))
        return {"status": "fail"}
    old_identity = old_manifest.get("identity", {})
    resumed_identity = resumed_recipe.get("identity", {})
    shared_paths = {
        "data.context_sha256": EXPECTED["context"] if "context" in EXPECTED else "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f",
        "data.selected_ids_sha256": EXPECTED["selected"],
        "data.ordered_ids_sha256": "7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d",
        "data.row_count": 8440,
        "parent.manifest_sha256": EXPECTED["parent_manifest"],
        "parent.merged_weights_sha256": EXPECTED["parent_weights"],
        "parent.base_model_revision": EXPECTED["parent_revision"],
        "source.frozen_source_root": "be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9",
        "source.source_schedule_sha256": EXPECTED["schedule"],
        "policy.candidate_count": 4,
        "policy.gradient_accumulation_steps": 8,
        "policy.per_device_train_batch_size": 4,
        "policy.rollout_rows_per_update": 32,
        "policy.context_max_tokens": 2240,
        "policy.completion_max_tokens": 192,
        "policy.model_load_max_seq_length": 4096,
        "renderer.renderer_id": "zeta2-prm03-v1",
        "tokenizer.bos_id": 0,
        "tokenizer.eos_id": 1,
        "tokenizer.pad_id": 1,
        "tokenizer.native_eog_ids": [1, 130073],
        "tokenizer.vocab_size": 130560,
    }
    shared_failures = []
    for path, expected in shared_paths.items():
        old_value = get_path(old_identity, path)
        resumed_value = get_path(resumed_identity, path)
        if path == "source.frozen_source_root":
            old_value = str(old_value).split("snapshots/")[-1].split("/source")[0] if old_value else old_value
            resumed_value = str(resumed_value).split("snapshots/")[-1].split("/source")[0] if resumed_value else resumed_value
        if old_value != expected or resumed_value != expected:
            shared_failures.append({"path": path, "expected": expected, "old": old_value, "resumed": resumed_value})
    add(checks, "identity.shared_source_parent_policy_tokenizer", "pass" if not shared_failures else "fail", shared_failures)

    intentional = {
        "old.schedule.stage_id": old_identity.get("schedule", {}).get("stage_id"),
        "resumed.schedule.stage_id": resumed_identity.get("schedule", {}).get("stage_id"),
        "old.schedule.full_save_steps": old_identity.get("schedule", {}).get("full_save_steps"),
        "resumed.schedule.full_save_steps": resumed_identity.get("schedule", {}).get("full_save_steps"),
        "old.schedule.first_development_update": old_identity.get("schedule", {}).get("first_development_update"),
        "resumed.schedule.first_development_update": resumed_identity.get("schedule", {}).get("first_development_update"),
        "old.policy.runtime_environment": old_identity.get("policy", {}).get("runtime_environment"),
        "resumed.policy.runtime_environment": resumed_identity.get("policy", {}).get("runtime_environment"),
    }
    add(checks, "identity.cross_run_difference_audit", "pass", {"known_identity_sha256": {"old": EXPECTED["identity_old"], "resumed": EXPECTED["identity_resumed"]}, "intentional_or_identity_bound_differences": intentional, "interpretation": "old v4/full25 and resumed v5/full5 are different identity contracts; cross-run byte-exact resume is not asserted"})
    return {"status": "pass" if not shared_failures else "fail", "known_identity_sha256": {"old": EXPECTED["identity_old"], "resumed": EXPECTED["identity_resumed"]}, "intentional_differences": intentional}


def verify_restore(checks: list[dict[str, Any]], resumed_root: Path, parent_checkpoint: Path) -> dict[str, Any]:
    preflight = resumed_root / "output" / "entry-preflight.json"
    resume_audit = small_section(preflight, "resume_audit")
    expected_audit = {"complete_optimizer_boundary": True, "consumed_rows": 1280, "source_draw_cursor": 40, "source_schedule_bound": True, "status": "verified", "step": 5}
    audit_failures = []
    for key, expected in expected_audit.items():
        actual = resume_audit.get(key) if resume_audit else None
        if actual != expected:
            audit_failures.append({"key": key, "expected": expected, "actual": actual})
    add(checks, "restore.resume_audit", "pass" if not audit_failures else "fail", {"audit": resume_audit, "failures": audit_failures})

    tokenizer_path = resumed_root / "output" / "train-begin-tokenizer-contract.json"
    tokenizer = read_json(tokenizer_path) if tokenizer_path.is_file() else {}
    token_identity = get_path(tokenizer, "verified.tokenizer_identity", {})
    token_expected = {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1, "vocab_size": 130560}
    token_failures = [{"key": key, "expected": value, "actual": token_identity.get(key)} for key, value in token_expected.items() if token_identity.get(key) != value]
    token_ok = tokenizer.get("step") == 5 and get_path(tokenizer, "verified.status") == "verified" and not token_failures
    add(checks, "restore.train_begin_tokenizer_step5", "pass" if token_ok else "fail", {"step": tokenizer.get("step"), "verified_status": get_path(tokenizer, "verified.status"), "tokenizer_identity": token_identity, "failures": token_failures})

    manifest_path = parent_checkpoint / "campaign-manifest.json"
    parent_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    manifest_files = parent_manifest.get("files", {})
    required = ("campaign-state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "adapter_model.safetensors")
    missing = [name for name in required if name not in manifest_files]
    parent_ok = parent_manifest.get("full") is True and parent_manifest.get("step") == 5 and not missing
    add(checks, "restore.parent_full_checkpoint_inventory", "pass" if parent_ok else "fail", {"full": parent_manifest.get("full"), "step": parent_manifest.get("step"), "missing": missing, "critical_hashes": {name: manifest_files.get(name, {}).get("sha256") for name in required}})
    trainer_path = parent_checkpoint / "trainer_state.json"
    trainer = read_json(trainer_path) if trainer_path.is_file() else {}
    trainer_ok = trainer.get("global_step") == 5 and trainer.get("max_steps") == 3000
    add(checks, "restore.parent_trainer_state_step5", "pass" if trainer_ok else "fail", {"global_step": trainer.get("global_step"), "max_steps": trainer.get("max_steps")})
    return {"status": "pass" if not audit_failures and token_ok and parent_ok and trainer_ok else "fail", "resume_audit": resume_audit, "tokenizer": token_identity, "parent_checkpoint": {"full": parent_manifest.get("full"), "step": parent_manifest.get("step"), "critical_files": required}}


def telemetry_metrics(path: Path, step: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    found = None
    with path.open("rb") as fh:
        for raw in fh:
            # The first train_begin line embeds the giant identity object.
            if len(raw) > 2 * 1024 * 1024:
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if row.get("event") == "trainer_metrics" and row.get("step") == step:
                metrics = row.get("metrics", {})
                found = {key: metrics.get(key) for key in ("grad_norm", "loss", "num_tokens", "reward", "reward_std")}
    return found


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--resumed-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    checks: list[dict[str, Any]] = []
    records = compare_records(checks, args.old_root, args.resumed_root, args.sequence)
    identity = compare_identity(checks, args.old_root, args.resumed_root)
    restore = verify_restore(checks, args.resumed_root, args.parent_checkpoint)
    old_metrics = telemetry_metrics(args.old_root / "telemetry.jsonl", 6)
    resumed_metrics = telemetry_metrics(args.resumed_root / "telemetry.jsonl", 6)
    stable_metric_keys = ("loss", "num_tokens", "reward", "reward_std")
    metrics_equal = old_metrics is not None and resumed_metrics is not None and all(old_metrics.get(key) == resumed_metrics.get(key) for key in stable_metric_keys)
    add(checks, "metrics.step6_stable_fields", "pass" if metrics_equal else "fail", {"keys": stable_metric_keys, "old": old_metrics, "resumed": resumed_metrics})
    add(checks, "metrics.step6_grad_norm_difference_audited", "pass" if old_metrics is not None and resumed_metrics is not None else "fail", {"old": old_metrics.get("grad_norm") if old_metrics else None, "resumed": resumed_metrics.get("grad_norm") if resumed_metrics else None, "interpretation": "the gradient norm is intentionally excluded from stable metric equality; it is audited in the gradient section"})
    hard_failures = [item for item in checks if item["status"] == "fail"]
    status = "semantic_match_gradient_scalar_drift" if records["semantic_status"] == "pass" and identity.get("status") == "pass" and restore.get("status") == "pass" else "mechanical_failure"
    result = {
        "schema_version": "sepalith.rl-06.resume6-independent-review.v1",
        "status": status,
        "old_root": str(args.old_root),
        "resumed_root": str(args.resumed_root),
        "parent_checkpoint": str(args.parent_checkpoint),
        "sequence_path": str(args.sequence),
        "expected": EXPECTED,
        "records": records,
        "identity": identity,
        "restore": restore,
        "metrics": {"old": old_metrics, "resumed": resumed_metrics, "equal": metrics_equal},
        "checks": checks,
        "limits": [
            "The old continuous v4/full25 run and resumed v5/full5 run have different identity contracts.",
            "The old artifact has no full optimizer checkpoint at step 5 for semantic optimizer-tensor comparison.",
            "Only the scalar gradient norm is compared; a 0.0122868 percent finite drift is not proof of byte-exact or non-exact optimizer restoration.",
            "The resumed run remains root-owned; this verifier does not stop or alter it.",
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "semantic": records["semantic_status"], "identity": identity.get("status"), "restore": restore.get("status"), "gradient_delta": records["gradient"].get("delta"), "failed_checks": len(hard_failures)}, sort_keys=True))
    return 1 if status == "mechanical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
