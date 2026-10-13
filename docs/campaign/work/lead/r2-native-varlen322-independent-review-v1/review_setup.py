#!/usr/bin/env python3
"""Read-only, small-file review of the root checkpoint-322 varlen canary setup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-native-varlen322-root-v1"
SOURCE = PLAN / "docs/campaign/work/lead/r2-native-varlen-canary-preparation-v1"
POLICY = PLAN / "docs/campaign/work/lead/r2-native-varlen-canary-root-review-v1/measurement-policy.json"
DECISION = PLAN / "docs/campaign/receipts/SFT-11-cpt322-root-decision.json"
GUARD = PLAN / "docs/campaign/work/lead/host-memory-guard-v4/cuda_host_guard.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arg(command, flag):
    index = command.index(flag)
    return command[index + 1]


def main():
    policy = load(POLICY)
    decision = load(DECISION)
    require(decision["checkpoint_manifest_sha256"] == "81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf", "checkpoint-322 decision differs")
    require(all(panel["relative_change"] < 0 for panel in decision["panels"].values()), "checkpoint-322 did not improve every bound panel")

    records = {}
    for arm in ("ordinary_reference", "varlen_candidate"):
        root = PACKET / arm
        recipe_path = root / "recipe.json"
        recipe, admission = load(recipe_path), load(root / "admission.json")
        preflight = load(root / "preflight-result.json")
        preflight_log = load(root / "preflight.log")
        guard = load(root / "guard-command.json")
        child = load(root / "child-command.json")
        env = load(root / "environment.json")
        canary = recipe["varlen_canary"]

        require(admission["bound_recipe_sha256"] == sha(recipe_path), f"{arm}: admission recipe hash differs")
        require(admission["arm"] == canary["arm"] == arm, f"{arm}: arm differs")
        require(admission["source_checkpoint"] == canary["source_checkpoint"], f"{arm}: source checkpoint differs")
        require(admission["source_checkpoint_manifest_sha256"] == canary["source_checkpoint_manifest_sha256"] == decision["checkpoint_manifest_sha256"], f"{arm}: source manifest differs")
        require((admission["source_global_step"], admission["target_global_step"]) == (322, 330), f"{arm}: step window differs")
        require((admission["source_cursor"], admission["target_cursor"]) == (4096, 4224), f"{arm}: cursor window differs")
        require(admission["updates"] == policy["updates_per_arm"] == 8 and admission["logical_rows"] == 16, f"{arm}: logical update shape differs")
        require(preflight["exit_code"] == 0 and preflight_log["status"] == "pass_preparation_canary_only", f"{arm}: CPU preflight did not pass")
        require(preflight_log["loss_denominators"] == [39690, 26479, 45634, 69499, 55665, 33799, 60212, 47586], f"{arm}: denominator vector differs")
        require(preflight_log["physical_packs"] == [3, 2, 4, 7, 5, 3, 6, 4], f"{arm}: physical pack vector differs")
        require(preflight_log["source_cursor"] == 4096 and preflight_log["target_cursor"] == 4224, f"{arm}: preflight cursor differs")
        require(recipe["runtime"]["optimizer"] == records.get("optimizer", recipe["runtime"]["optimizer"]), f"{arm}: optimizer differs")
        require(recipe["seed"] == 3407, f"{arm}: seed differs")
        require(recipe["cohort"]["draw_schedule"]["sha256"] == "5ea4a04f4082f93ad8e7211a139e19e7ffffd33f4ef2661d93e42134933a3cac", f"{arm}: schedule differs")
        require(recipe["cohort"]["streaming_cache"]["manifest_sha256"] == "0e2bad85de9926d1949aa17397144e58f86f7604d7b72f41e1fcfefec7b27fb8", f"{arm}: cache differs")
        require(recipe["parent"]["files"]["tokenizer.json"] == "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81", f"{arm}: tokenizer differs")
        require(recipe["runtime_source"]["manifest_sha256"] == "faa7dd822910c2ad81154f6b5f9b7b8165952adcd7a38e41d3c3a90210a8b2b0", f"{arm}: runtime source differs")
        require(int(arg(guard, "--seconds")) == policy["resource_policy"]["guard_seconds_per_arm"], f"{arm}: guard deadline differs")
        require(int(arg(guard, "--minimum-free-mib")) == policy["resource_policy"]["host_soft_mib"], f"{arm}: soft floor differs")
        require(int(arg(guard, "--admission-free-mib")) == policy["resource_policy"]["host_admission_mib"], f"{arm}: admission floor differs")
        require(arg(child, "--resume") == admission["source_checkpoint"] and arg(child, "--canary-admission") == str((root / "admission.json").resolve()), f"{arm}: child handoff differs")
        require(env["PYTORCH_ALLOC_CONF"].startswith("backend:native,"), f"{arm}: allocator differs")
        records[arm] = {"recipe_sha256": sha(recipe_path), "admission_sha256": sha(root / "admission.json"), "preflight_log_sha256": sha(root / "preflight.log"), "outputs": recipe["outputs"], "environment": env}
        records["optimizer"] = recipe["runtime"]["optimizer"]

    left, right = records["ordinary_reference"], records["varlen_candidate"]
    require(set(left["outputs"].values()).isdisjoint(set(right["outputs"].values())), "arm output paths overlap")
    ordinary_env, packed_env = left["environment"], right["environment"]
    require({k: v for k, v in ordinary_env.items() if k not in {"PYTHONPATH", "SEPALITH_ALLOCATOR_REPORT"}} == {k: v for k, v in packed_env.items() if k not in {"PYTHONPATH", "SEPALITH_ALLOCATOR_REPORT"}}, "arm environments differ beyond backend/report")
    require(ordinary_env["PYTHONPATH"] == "" and packed_env["PYTHONPATH"].endswith("/cu130-overlay"), "backend environment differs")
    guard_text = GUARD.read_text()
    require("hard_mib=4096" in guard_text and policy["resource_policy"]["host_hard_mib"] == 4096, "guard hard floor differs")

    result = {
        "schema": "sepalith.sft11.native-varlen322-independent-review.v1",
        "status": "pass_no_launch_blocker",
        "checkpoint_manifest_sha256": decision["checkpoint_manifest_sha256"],
        "source_cursor": 4096,
        "target_cursor": 4224,
        "source_global_step": 322,
        "target_global_step": 330,
        "logical_rows_each": 128,
        "loss_denominators": [39690, 26479, 45634, 69499, 55665, 33799, 60212, 47586],
        "candidate_physical_packs": [3, 2, 4, 7, 5, 3, 6, 4],
        "arms": {k: {x: y for x, y in records[k].items() if x != "environment"} for k in ("ordinary_reference", "varlen_candidate")},
        "source_manifest_sha256": sha(SOURCE / "source-manifest.json"),
        "measurement_policy_sha256": sha(POLICY),
        "checkpoint322_decision_sha256": sha(DECISION),
        "guard_source_sha256": sha(GUARD),
        "findings": [],
        "reservations": [
            "Matched 2K/8K/16K evaluation and the root production-packing decision remain post-run gates.",
            "MKL_NUM_THREADS is 4 in both arm environments; it is paired and therefore does not bias the arm comparison, but exceeds the campaign's usual two-thread CPU convention.",
            "The arm-specific full checkpoints intentionally receive distinct execution identities; source checkpoint validation uses the unchanged ordinary scientific identity before either arm runs."
        ],
        "large_payload_rehashed": False,
        "cuda_started_by_review": False
    }
    out = Path(__file__).with_name("review-result.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
