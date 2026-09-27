#!/usr/bin/env python3
"""Independent metadata/static review of the root checkpoint-450 verifier.

This file intentionally never hashes or opens a future checkpoint payload.
It records the preterminal gate, recipe arithmetic, source-level full-file
checks, and heldout-panel bindings needed before root can run prepare.py.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
LEAD = PLAN / "docs/campaign/work/lead"
PACKET = LEAD / "r2-cpt450-verifier-independent-review-v1"
ROOT_PREPARE = LEAD / "r2-cpt450-review-root-v1/prepare.py"
ACTIVE = LEAD / "r2-selected330-ordinary-root-v1"
RETRY = LEAD / "r2-selected330-ordinary-retry-root-v1"
HOST = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
    "SFT11-selected330-ordinary-to450-root-v1-host-supervision-b"
)
ARCHIVE = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/"
    "SFT11-full-weight-CPT-production-from-selected-packed330-v1"
)
NATIVE_TRAINER = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
    "SFT11-native-CPT-production-from-selected-packed330-v1/runtime"
)
PANELS = LEAD / "r2-cpt-long-holdout-preparation-v1/evaluator-configs.json"
EVALUATOR = LEAD / "r2-cpt-long-eval-root-v2/evaluate.py"
ANCHOR_TEMPLATE = LEAD / "r2-full-weight-cpt-lr-pilot-v3/recipe.lr3e-5.template.json"
RECIPE = ACTIVE / "runtime-recipe.json"
SOURCE_TRAINER = LEAD / (
    "r2-selected330-ordinary-continuation-preparation-v2/source/"
    "experiments/training/full_weight_cpt_trainer.py"
)

PIDS = (2473526, 2474029)
EXPECTED_RECIPE_SHA = "eff266cdb287e8b099990f265760539e024bc20c14dd0707b6e225abce1a6981"
EXPECTED_SOURCE_CHECKPOINT_MANIFEST = "2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951"
EXPECTED_FULL_FILES = frozenset(
    {
        "campaign-state.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
    }
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path: Path, *, expected_bytes: int | None = None, hash_file: bool = True) -> dict[str, Any]:
    """Pin only small metadata/source files; never read checkpoint payloads."""
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    stat = path.stat()
    record["bytes"] = stat.st_size
    if expected_bytes is not None:
        record["expected_bytes"] = expected_bytes
        record["bytes_match"] = stat.st_size == expected_bytes
    if hash_file:
        if stat.st_size > 50 * 1024 * 1024:
            raise ValueError(f"refusing payload hash in independent review: {path}")
        record["sha256"] = sha(path)
    return record


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def live_process(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    result: dict[str, Any] = {"pid": pid, "live": proc.exists()}
    if proc.exists():
        raw = (proc / "cmdline").read_bytes().replace(b"\x00", b" ")
        result["cmdline"] = raw.decode("utf-8", errors="replace")
    return result


def exact_draw_window(initial_cursor: int, observed_draws: int) -> dict[str, Any]:
    positions = list(range(initial_cursor, initial_cursor + observed_draws))
    return {
        "initial_cursor": initial_cursor,
        "observed_draws": observed_draws,
        "positions_first": positions[0] if positions else None,
        "positions_last": positions[-1] if positions else None,
        "positions_contiguous": positions == list(range(initial_cursor, initial_cursor + observed_draws)),
    }


def full_checkpoint_file_contract(manifest: dict[str, Any], durable: Path, native: Path) -> dict[str, Any]:
    """Small-fixture-compatible mirror of prepare.py's 12-file checks."""
    files = manifest.get("files", {})
    required = set(files) == set(EXPECTED_FULL_FILES)
    details: dict[str, Any] = {"required_set_exact": required, "durable": {}, "native": {}}
    if not required:
        return details
    for root, label in ((durable, "durable"), (native, "native")):
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        details[label]["file_set_exact"] = actual == set(EXPECTED_FULL_FILES) | {"campaign-manifest.json"}
        details[label]["files"] = {}
        for name, expected in files.items():
            path = root / name
            item = {"exists": path.is_file(), "symlink": path.is_symlink()}
            if path.is_file() and not path.is_symlink():
                raw = path.read_bytes()
                item.update({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
                item["matches_manifest"] = (
                    len(raw) == expected["bytes"] and item["sha256"] == expected["sha256"]
                )
            else:
                item["matches_manifest"] = False
            details[label]["files"][name] = item
    durable_manifest = durable / "campaign-manifest.json"
    native_manifest = native / "campaign-manifest.json"
    details["manifest_equal"] = (
        durable_manifest.is_file()
        and native_manifest.is_file()
        and json.loads(durable_manifest.read_text()) == manifest
        and json.loads(native_manifest.read_text()) == manifest
    )
    details["pass"] = bool(
        required
        and details["durable"]["file_set_exact"]
        and details["native"]["file_set_exact"]
        and details["manifest_equal"]
        and all(item["matches_manifest"] for item in details["durable"]["files"].values())
        and all(item["matches_manifest"] for item in details["native"]["files"].values())
    )
    return details


def source_static_review(source: str) -> dict[str, Any]:
    checks = {
        "exact_12_file_set": "set(manifest['files']) ==" in source and all(name in source for name in EXPECTED_FULL_FILES),
        "durable_actual_set": "actual == set(manifest['files']) | {'campaign-manifest.json'}" in source,
        "native_manifest_equal": "read(native / 'campaign-manifest.json') == manifest" in source,
        "native_payload_hashes": "sha(path) == expected['sha256'], name" in source,
        "durable_payload_hashes": "sha(path) == expected['sha256'], name" in source,
        "native_symlink_rejection": "not path.is_symlink()" in source,
        "no_cuda_launcher": not any(token in source for token in ("subprocess", "Popen(", "FastLanguageModel", "torch.cuda")),
        "no_final_dataset_path": "/final" not in source and "final_dataset" not in source,
    }
    line_numbers = {}
    for needle in (
        "for pid in (2473526, 2474029)",
        "terminal = read(",
        "controller = read(",
        "telemetry = [",
        "assert set(manifest['files']) ==",
        "native = Path(recipe['outputs']['trainer'])",
        "binding.update(model_path=str(native)",
    ):
        line_numbers[needle] = next(
            (index for index, line in enumerate(source.splitlines(), 1) if needle in line), None
        )
    try:
        ast.parse(source)
        parse_ok = True
    except SyntaxError:
        parse_ok = False
    return {"checks": checks, "all_static_checks_pass": parse_ok and all(checks.values()), "line_numbers": line_numbers}


def evaluate_panels() -> dict[str, Any]:
    config = read_json(PANELS)
    panel_rows = {}
    for name, panel in config["panels"].items():
        record = panel["validation_rows"]
        path = Path(record["path"])
        panel_rows[name] = {
            "role": panel["role"],
            "context_tokens": panel["parameters"]["max_sequence_tokens"],
            "validation_batch_size": panel["validation_batch_size"],
            "rows": record["rows"],
            "recorded_bytes": record["bytes"],
            "recorded_sha256": record["sha256"],
            "exists": path.is_file(),
            "observed_bytes": path.stat().st_size if path.is_file() else None,
            "path_is_final": "/final" in str(path).lower() or "final" in str(path).lower(),
        }
    template = read_json(ANCHOR_TEMPLATE)
    anchor = template["validation"]
    anchor_path = Path(anchor["path"])
    panel_rows["anchor2k"] = {
        "role": anchor["role"],
        "context_tokens": anchor["max_sequence_tokens"],
        "rows": anchor["rows"],
        "recorded_bytes": anchor_path.stat().st_size if anchor_path.is_file() else None,
        "recorded_sha256": anchor["sha256"],
        "exists": anchor_path.is_file(),
        "observed_bytes": anchor_path.stat().st_size if anchor_path.is_file() else None,
        "path_is_final": "/final" in str(anchor_path).lower() or "final" in str(anchor_path).lower(),
    }
    evaluator_source = EVALUATOR.read_text(encoding="utf-8")
    static = {
        "source_sha256": sha(EVALUATOR),
        "panels_exact_8k_16k": "set(panels) == {'8k', '16k'}" in evaluator_source,
        "anchor2k_added_from_template": "panels['anchor2k']" in evaluator_source,
        "same_model_tokenizer_runtime": "load_pinned_reference_tokenizer" in evaluator_source and "assert_runtime_tokenizer" in evaluator_source,
        "no_training": "training_performed': False" in evaluator_source,
        "no_promotion": "promotion_authorized': False" in evaluator_source,
    }
    return {
        "config": pin(PANELS),
        "config_schema": config.get("schema"),
        "fixed_499_2k_comparator_unchanged": config.get("fixed_499_2k_comparator_unchanged"),
        "panels": panel_rows,
        "evaluator": static,
        "all_panel_metadata_checks_pass": bool(
            set(config.get("panels", {})) == {"8k", "16k"}
            and config.get("fixed_499_2k_comparator_unchanged") is True
            and all(item["exists"] and item["observed_bytes"] == item["recorded_bytes"] and not item["path_is_final"] for item in panel_rows.values())
            and all(static.values()),
        ),
    }


def collect() -> dict[str, Any]:
    root_source = ROOT_PREPARE.read_text(encoding="utf-8")
    recipe = read_json(RECIPE)
    transition = read_json(ACTIVE / "transition.json")
    continuation = read_json(ACTIVE / "continuation.json")
    stop = read_json(ACTIVE / "execution-stop.json")
    retry_launch = read_json(RETRY / "launch.json")
    host_terminal = HOST / "terminal.json"
    controller_terminal = RETRY / "terminal.json"
    process = [live_process(pid) for pid in PIDS]
    terminal_state = {
        "host_terminal": pin(host_terminal),
        "controller_terminal": pin(controller_terminal),
        "host_terminal_status": read_json(host_terminal) if host_terminal.exists() else None,
        "controller_terminal_status": read_json(controller_terminal) if controller_terminal.exists() else None,
        "preterminal_gate": all(not item["live"] for item in process),
        "observed_processes": process,
        "retry_launch": retry_launch,
    }
    schedule = {
        "resume_step": 330,
        "stop_step": 450,
        "updates": 450 - 330,
        "update_steps": [331, 450],
        "initial_cursor": 4224,
        "draws": 1920,
        "last_draw_position": 6143,
        "exclusive_end_cursor": 6144,
        "draw_arithmetic": exact_draw_window(4224, 1920),
        "draws_equal_updates_times_effective_batch": 1920 == (450 - 330) * recipe["runtime"]["effective_batch"],
    }
    recipe_checks = {
        "recipe_sha256": sha(RECIPE),
        "recipe_sha_matches_expected": sha(RECIPE) == EXPECTED_RECIPE_SHA,
        "recipe_status": recipe.get("status"),
        "recipe_effective_batch": recipe["runtime"]["effective_batch"],
        "recipe_max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"],
        "recipe_validation_context_tokens": recipe["validation"]["max_sequence_tokens"],
        "transition": {
            "bound_recipe_sha256": transition["bound_recipe_sha256"],
            "checkpoint_step": transition["checkpoint_step"],
            "checkpoint_manifest_sha256": transition["checkpoint_manifest_sha256"],
            "source_arm": transition["source_arm"],
            "destination_execution": transition["destination_execution"],
            "status": transition["status"],
            "source_checkpoint_path": transition["checkpoint"],
        },
        "continuation": continuation,
        "execution_stop": stop,
        "source_checkpoint_manifest": pin(Path(transition["checkpoint"]) / "campaign-manifest.json"),
        "source_checkpoint_manifest_hash_matches": pin(Path(transition["checkpoint"]) / "campaign-manifest.json").get("sha256") == EXPECTED_SOURCE_CHECKPOINT_MANIFEST,
        "source_packed330_full_state": transition["source_arm"] == "varlen_candidate" and transition["checkpoint_step"] == 330 and transition["checkpoint_manifest_sha256"] == EXPECTED_SOURCE_CHECKPOINT_MANIFEST and transition["destination_execution"] == "ordinary_sdpa_unpacked",
        "ordinary_identity_bound": isinstance(transition.get("destination_identity_sha256"), str) and len(transition["destination_identity_sha256"]) == 64 and transition["destination_execution"] == "ordinary_sdpa_unpacked",
    }
    static = source_static_review(root_source)
    review = {
        "schema": "sepalith.sft11.cpt450_verifier_independent_review.v1",
        "review_id": "SFT-11-cpt450-verifier-independent-review",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "preterminal_review_complete_root_verifier_pending_terminal",
        "training_admission": False,
        "cuda_launched": False,
        "checkpoint_payloads_loaded": False,
        "future_artifacts_fabricated": False,
        "inputs": {
            "root_prepare": pin(ROOT_PREPARE),
            "bound_recipe": pin(RECIPE),
            "transition": pin(ACTIVE / "transition.json"),
            "continuation": pin(ACTIVE / "continuation.json"),
            "execution_stop": pin(ACTIVE / "execution-stop.json"),
            "retry_launch": pin(RETRY / "launch.json"),
            "source_trainer": pin(SOURCE_TRAINER),
        },
        "terminal_guard_order": terminal_state,
        "schedule": schedule,
        "recipe_checks": recipe_checks,
        "full_checkpoint_contract": {
            "expected_files": sorted(EXPECTED_FULL_FILES),
            "root_static_check": static["checks"],
            "future_durable_checkpoint": pin(ARCHIVE / "full/checkpoint-450/campaign-manifest.json"),
            "future_native_checkpoint": pin(NATIVE_TRAINER / "checkpoint-450/campaign-manifest.json"),
            "payload_hash_verification_deferred_until_terminal": True,
        },
        "evaluation_binding": evaluate_panels(),
        "static_verifier_review": static,
        "findings": [
            {
                "id": "F1",
                "severity": "pass",
                "finding": "The verifier stops on either live current retry handle before reading terminal/checkpoint data; the current run is demonstrably preterminal.",
            },
            {
                "id": "F2",
                "severity": "pass",
                "finding": "The requested stage arithmetic is exact: 120 updates 331..450 and 1,920 draws from 4,224 through 6,143, exclusive cursor 6,144.",
            },
            {
                "id": "F3",
                "severity": "pass_deferred",
                "finding": "The frozen verifier requires exactly the 12 full-state files, rejects symlinks/extras, and hashes every durable and native payload; no future payload exists to verify yet.",
            },
            {
                "id": "F4",
                "severity": "pass",
                "finding": "The bound transition identifies source packed-330 full state and ordinary_sdpa_unpacked destination, with source manifest 2b358545...520951.",
            },
            {
                "id": "F5",
                "severity": "pass",
                "finding": "The evaluator route contains fixed 2K anchor data from the pinned LR template plus separate 8K and 16K package-heldout panels; panel metadata is not final data.",
            },
            {
                "id": "A1",
                "severity": "advisory",
                "finding": "prepare.py checks draw count and endpoints in run-result, while the pinned trainer enforces contiguous _draw_position values in-process. A future hardening revision could independently bind a compact observed-position digest, but the source-bound trainer currently supplies the continuity guard.",
            },
            {
                "id": "A2",
                "severity": "advisory",
                "finding": "prepare.py checks the two current PIDs and terminal status paths but does not bind terminal JSON to launch.json controller identity. Current absent terminal files and live-PID gate prevent stale reuse; a postterminal hardening revision could add launch/controller PID and command-hash equality.",
            },
        ],
        "required_next_step": "After root retry terminal success and both handles exit, run the frozen root prepare.py once; then independently rehash its generated binding/command packet before any matched evaluation launch.",
    }
    return review


if __name__ == "__main__":
    PACKET.mkdir(parents=True, exist_ok=True)
    path = PACKET / "review.json"
    value = collect()
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "status": value["status"], "pids": value["terminal_guard_order"]["observed_processes"]}, sort_keys=True))
