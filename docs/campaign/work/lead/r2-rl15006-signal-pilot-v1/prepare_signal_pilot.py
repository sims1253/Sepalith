#!/usr/bin/env python3
"""Prepare a deterministic TRAIN-only signal pilot over the eligible 15,006 pool."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PACKET = Path(__file__).resolve().parent
ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl")
CONTEXT = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl")
SELECTED_IDS = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/selected-train-ids.json")
BUFFER = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-buffer-v5/reward-buffer-sidecar.jsonl")
BUFFER_MANIFEST = BUFFER.parent / "materialization.json"
EXPECTED = {
    "rows": "65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7",
    "context": "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a",
    "selected_ids": "986a1f7910c44423e50988e76abb6671dedf8cd70f5542ff9efd7a03ce504462",
    "buffer": "126a9654b9d7d788c6cd60a4c90c9e0bf7f88fa718232be3f209473c94a81c4f",
    "buffer_manifest": "a49b1ef464773817d7a18f76cdec68b82eaa4abb960a10a7d919c7535d7add84",
}
HELD = {"8451310ff8e0c5d9f3e77bbc", "dd65bd2cd11f38e729a9c712"}
PARSER_IDENTITY = {
    "command": "Rscript --vanilla parse_only.R",
    "generated_r_executed": False,
    "operation": "base::parse(file=...,keep.source=FALSE)",
    "r_version": "R 4.6.1 (2026-06-24)",
}
SEED = "sepalith-rl11-signal-pilot-v1"
GROUPS_PER_FAMILY = 16
CANDIDATES_PER_GROUP = 4
LONG_TARGET_THRESHOLD = 192
FAMILY_ORDER = (
    "finish_block", "format_propagation", "na_rm_propagation", "no_op",
    "pipe_rewrite", "rename_propagation", "roxygen_drafting",
)
MODE_ORDER = ("unverified", "framed_fragment", "completion_prefix", "complete_document")
BAND_ORDER = ("long_gt_192", "standard_le_192")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_key(row_id: str) -> str:
    return hashlib.sha256((SEED + "\0" + row_id).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> int:
    actual = {name: sha256(path) for name, path in {
        "rows": ROWS, "context": CONTEXT, "selected_ids": SELECTED_IDS,
        "buffer": BUFFER, "buffer_manifest": BUFFER_MANIFEST,
    }.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"input identity mismatch: {actual}")
    rows = read_jsonl(ROWS)
    selected = json.loads(SELECTED_IDS.read_text(encoding="utf-8"))["row_ids"]
    if len(rows) != len(selected) != 15006 or [row["id"] for row in rows] != selected:
        raise RuntimeError("eligible row/selected order mismatch")
    if HELD.intersection(selected):
        raise RuntimeError("held contradictory IDs entered eligible pool")
    buffers = {row["row_id"]: row for row in read_jsonl(BUFFER) if row["row_id"] not in HELD}
    if set(buffers) != set(selected):
        raise RuntimeError("reward-buffer coverage differs from eligible rows")

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    pool_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    pool_max_prompt = 0
    pool_max_target = 0
    pool_max_sequence = 0
    for position, row in enumerate(rows):
        row_id = row["id"]
        evidence = buffers[row_id]
        if evidence["supported"] and evidence.get("parser_identity") != PARSER_IDENTITY:
            raise RuntimeError(f"parser identity mismatch: {row_id}")
        mode = evidence["buffer_mode"] if evidence["supported"] else "unverified"
        target_tokens = len(row["input_ids"]) - row["target_start"]
        pool_max_prompt = max(pool_max_prompt, row["target_start"])
        pool_max_target = max(pool_max_target, target_tokens)
        pool_max_sequence = max(pool_max_sequence, len(row["input_ids"]))
        if target_tokens != evidence["target_tokens_including_protocol_eos"]:
            raise RuntimeError(f"target length mismatch: {row_id}")
        band = "long_gt_192" if target_tokens > LONG_TARGET_THRESHOLD else "standard_le_192"
        family = row["family"]
        if family not in FAMILY_ORDER:
            raise RuntimeError(f"unexpected family: {family}")
        item = {
            "row_id": row_id,
            "pool_position": position,
            "family": family,
            "package_id": row["package_id"],
            "syntax_evidence_mode": mode,
            "syntax_available": bool(evidence["supported"]),
            "repair_reason": evidence["repair_reason"],
            "target_band": band,
            "prompt_tokens": row["target_start"],
            "target_tokens": target_tokens,
            "prompt_ids_sha256": hashlib.sha256(canonical(row["input_ids"][:row["target_start"]])).hexdigest(),
            "selection_key": stable_key(row_id),
        }
        strata[(family, mode, band)].append(item)
        family_counts[family] += 1
        pool_counts[f"{family}|{mode}|{band}"] += 1
    if tuple(family for family in FAMILY_ORDER if family_counts[family]) != FAMILY_ORDER:
        raise RuntimeError("one or more task families is absent")
    for values in strata.values():
        values.sort(key=lambda item: (item["selection_key"], item["row_id"]))

    chosen: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()
    for family in FAMILY_ORDER:
        active = [
            (family, mode, band) for mode in MODE_ORDER for band in BAND_ORDER
            if strata.get((family, mode, band))
        ]
        cursors = {key: 0 for key in active}
        family_selected: list[dict[str, Any]] = []
        while len(family_selected) < GROUPS_PER_FAMILY:
            progressed = False
            for key in active:
                cursor = cursors[key]
                if cursor < len(strata[key]) and len(family_selected) < GROUPS_PER_FAMILY:
                    item = dict(strata[key][cursor])
                    item["family_pilot_index"] = len(family_selected)
                    family_selected.append(item)
                    cursors[key] += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"family {family} has fewer than {GROUPS_PER_FAMILY} rows")
        chosen.extend(family_selected)
        for item in family_selected:
            selection_counts[f"{family}|{item['syntax_evidence_mode']}|{item['target_band']}"] += 1

    for group_index, item in enumerate(chosen):
        item["group_index"] = group_index
        item["candidate_count"] = CANDIDATES_PER_GROUP
    if len(chosen) != len(FAMILY_ORDER) * GROUPS_PER_FAMILY or len({x["row_id"] for x in chosen}) != len(chosen):
        raise RuntimeError("pilot selection cardinality or uniqueness mismatch")
    if not any(item["target_band"] == "long_gt_192" for item in chosen):
        raise RuntimeError("pilot lacks long-target coverage")
    if not any(item["syntax_evidence_mode"] == "unverified" for item in chosen):
        raise RuntimeError("pilot lacks unverified-syntax coverage")

    groups_path = PACKET / "pilot-groups.jsonl"
    with groups_path.open("w", encoding="utf-8") as stream:
        for item in chosen:
            stream.write(canonical(item).decode("utf-8") + "\n")
    pool_census = {
        "schema": "sepalith.rl11.signal-pilot-pool-census.v1",
        "pool_rows": len(rows),
        "family_counts": dict(family_counts),
        "atomic_strata_counts": dict(sorted(pool_counts.items())),
        "pilot_atomic_strata_counts": dict(sorted(selection_counts.items())),
        "all_nonempty_family_mode_length_strata_represented": all(selection_counts[key] > 0 for key in pool_counts),
        "pilot_rows": len(chosen),
        "pilot_rollouts": len(chosen) * CANDIDATES_PER_GROUP,
        "pool_max_prompt_tokens": pool_max_prompt,
        "pool_max_target_tokens": pool_max_target,
        "pool_max_sequence_tokens": pool_max_sequence,
        "pool_max_prompt_plus_pilot_cap": pool_max_prompt + 1024,
    }
    (PACKET / "pool-census.json").write_text(json.dumps(pool_census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prompt_tokens = sum(item["prompt_tokens"] for item in chosen) * CANDIDATES_PER_GROUP
    actual_target_tokens = sum(item["target_tokens"] for item in chosen) * CANDIDATES_PER_GROUP
    spec = {
        "schema": "sepalith.rl11.signal-pilot-spec.v1",
        "status": "prepared_model_binding_and_root_admission_required",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Measure reward signal and telemetry behavior on TRAIN only; this is not a training subset or a quality evaluation.",
        "pool": {
            "eligible_rows": 15006,
            "eventual_rl_pool_preserved": True,
            "pilot_selection_is_not_training_cap": True,
            "arbitrary_pool_exclusions": 0,
            "held_contradictory_ids": sorted(HELD),
            "inputs": {name: {"path": str(path), "sha256": EXPECTED[name]} for name, path in {
                "rows": ROWS, "context": CONTEXT, "selected_ids": SELECTED_IDS,
                "buffer": BUFFER, "buffer_manifest": BUFFER_MANIFEST,
            }.items()},
        },
        "selection": {
            "seed": SEED,
            "algorithm": "per-family round-robin over every nonempty (syntax_mode,target_band) stratum; within-stratum SHA256(seed+NUL+row_id)",
            "families": list(FAMILY_ORDER),
            "groups_per_family": GROUPS_PER_FAMILY,
            "prompt_groups": len(chosen),
            "candidate_count": CANDIDATES_PER_GROUP,
            "rollouts": len(chosen) * CANDIDATES_PER_GROUP,
            "long_target_threshold_tokens": LONG_TARGET_THRESHOLD,
            "groups_path": str(groups_path),
            "groups_sha256": sha256(groups_path),
        },
        "generation": {
            "model_snapshot": None,
            "model_binding_policy": "Root binds one exact post-SFT merged snapshot and tokenizer only after the SFT milestone decision.",
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "candidate_count": CANDIDATES_PER_GROUP,
            "max_new_tokens": 1024,
            "prompt_max_tokens": 3072,
            "context_max_tokens": 4096,
            "generation_groups_per_call": 1,
            "full_stored_prompt_ids": True,
            "full_pool_max_prompt_tokens": pool_max_prompt,
            "full_pool_prompts_truncated": 0,
        },
        "resume": {
            "unit": "complete_prompt_group_of_four",
            "commit_order": "fsync four generation records and four reward records, then atomically create one group receipt",
            "deduplication_key": "pilot_identity_sha256 + row_id + candidate_index",
            "resume_rule": "verify immutable config/model/input hashes and every committed group; resume at first missing group; reject partial or duplicate group",
            "output_must_be_fresh_or_matching_resume": True,
        },
        "metrics": {
            "group_reward_variance": "population variance across exactly four rewards; denominator 112 prompt groups; report by family and stratum",
            "raw_output": "448 expected candidate records; report persisted/nonempty/hash/cap-hit denominators",
            "protocol": "448 expected; report valid, invalid, unterminated, operation and canonical-EOS counts",
            "false_noop": "report count over all 448 and rate over no_op gold rows separately",
            "repetition": "report severe-repetition count over all protocol-valid non-exact candidates",
            "applied_syntax": "report checked/pass/fail only where syntax_available; unverified is a separate denominator and must not call parse",
            "exact_region": "report exact target equality separately from syntax; no semantic correctness claim beyond exact equality",
        },
        "finish_block_diagnostics": {
            "train_only_prompt_groups": 16,
            "candidate_denominator": 64,
            "selected_evidence": {
                "completion_prefix_long": 4,
                "completion_prefix_standard": 4,
                "framed_fragment_long": 4,
                "framed_fragment_standard": 4,
            },
            "automatic_reports": [
                "protocol-valid, non-exact, applied-syntax-passing candidates (syntactically valid wrong implementations)",
                "severe repeated blocks among protocol-valid non-exact candidates",
                "cap hits and missing terminal outcomes",
                "applied-syntax failures, including possible extra delimiters against immutable suffix/frame",
            ],
            "review_only_reports": [
                "available-context return-type or functional contract apparently ignored",
                "unrelated validation boilerplate",
            ],
            "review_policy": "Use only each selected TRAIN PromptContext and candidate output. Preserve raw evidence and mark unknown when the behavior cannot be established deterministically. These labels do not change pilot rewards.",
            "parse_success_semantic_claim": False,
            "dev_examples_or_targets_used": False,
        },
        "parser": {
            "identity": PARSER_IDENTITY,
            "identity_sha256": hashlib.sha256(canonical(PARSER_IDENTITY)).hexdigest(),
            "syntax_success_exit": 0,
            "syntax_failure_exit": 1,
            "all_other_exits_signals_timeouts": "infrastructure_failure_no_reward_no_group_commit",
            "generated_r_executed": False,
        },
        "estimates": {
            "prompt_token_presentations": prompt_tokens,
            "gold_target_token_presentations_for_reference_only": actual_target_tokens,
            "maximum_generated_tokens": len(chosen) * CANDIDATES_PER_GROUP * 1024,
            "historical_mean_generated_tokens_per_rollout": 14567 / 320,
            "historical_expected_generated_tokens": round(len(chosen) * CANDIDATES_PER_GROUP * (14567 / 320)),
            "kv_cache_formula_bytes": "2 * layers * kv_heads * head_dim * bytes_per_element * concurrent_sequences * (padded_prompt_tokens + generated_tokens); bind after model snapshot",
            "conditional_known_base_architecture": {
                "base_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
                "reference_config_sha256": "f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991",
                "layers": 42,
                "kv_heads": 2,
                "head_dim": 128,
                "bf16_bytes": 2,
                "concurrent_sequences": 4,
                "kv_bytes_per_token_per_sequence": 43008,
                "kv_upper_bound_bytes_at_4024_tokens": 692256768,
                "kv_upper_bound_gib": 0.645,
                "condition": "Recompute and reject mismatch after root binds the exact post-SFT model config; excludes model weights, allocator workspace, activations and framework overhead.",
            },
            "model_weights_and_optimizer_bytes": "not estimable until root binds snapshot; pilot performs no optimizer update",
        },
        "continuation_gates": {
            "operational": [
                "112 complete unique prompt-group receipts and 448 unique candidates",
                "zero hash/order/model-binding mismatches and zero partial groups",
                "all seven families and every selected nonempty syntax/length stratum reported",
            ],
            "signal": [
                "report finite within-group variance for every group and nonzero-variance fractions by family",
                "do not begin a long RL run for a family with fewer than two nonzero-variance pilot groups without root review or a revised reward/sampling plan",
                "report zero-variance exact/all-invalid/all-neutral groups separately; a parseable wrong freeform score of zero is not semantic evidence",
                "any cap hits, parse service failures, or unverified-syntax parse calls require repair/review before continuation",
            ],
            "quality_claim_permitted": False,
        },
        "launch": {"authorized": False, "performed": False, "blocked_on": ["root model snapshot decision", "root pilot admission"]},
    }
    (PACKET / "pilot-spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "pilot_groups": len(chosen), "rollouts": len(chosen) * 4,
        "groups_sha256": sha256(groups_path), "pool_census": pool_census,
        "estimates": spec["estimates"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
