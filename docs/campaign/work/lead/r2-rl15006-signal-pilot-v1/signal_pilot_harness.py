#!/usr/bin/env python3
"""CPU-safe identity, atomic-group resume, and signal summary harness.

The live rollout driver supplies four generation and reward records after one
complete prompt group. This module never loads a model and never executes R.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


class PilotError(ValueError):
    pass


def require(value: bool, reason: str) -> None:
    if not value:
        raise PilotError(reason)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_spec(path: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    require(spec.get("schema") == "sepalith.rl11.signal-pilot-spec.v1", "spec schema mismatch")
    require(spec["pool"]["eligible_rows"] == 15006, "spec does not retain eligible15006 pool")
    require(spec["pool"]["eventual_rl_pool_preserved"] is True, "pilot is presented as a training cap")
    require(spec["selection"]["prompt_groups"] == 112 and spec["selection"]["candidate_count"] == 4, "pilot geometry mismatch")
    require(spec["generation"]["max_new_tokens"] == 1024, "pilot output cap mismatch")
    require(spec["generation"]["context_max_tokens"] == 4096, "pilot context window mismatch")
    for entry in spec["pool"]["inputs"].values():
        candidate = Path(entry["path"])
        require(candidate.is_file() and sha256(candidate) == entry["sha256"], "spec input identity mismatch")
    groups_path = Path(spec["selection"]["groups_path"])
    require(groups_path.is_file() and sha256(groups_path) == spec["selection"]["groups_sha256"], "pilot group identity mismatch")
    groups = [json.loads(line) for line in groups_path.read_text(encoding="utf-8").splitlines()]
    require(len(groups) == 112 and len({row["row_id"] for row in groups}) == 112, "pilot group coverage mismatch")
    require([row["group_index"] for row in groups] == list(range(112)), "pilot group order mismatch")
    require(set(row["family"] for row in groups) == set(spec["selection"]["families"]), "pilot family coverage mismatch")
    require(any(row["target_band"] == "long_gt_192" for row in groups), "pilot lacks long target")
    require(any(row["syntax_evidence_mode"] == "unverified" for row in groups), "pilot lacks unverified syntax")
    require(all(row["prompt_tokens"] <= spec["generation"]["prompt_max_tokens"] for row in groups), "pilot prompt cap mismatch")
    return spec, sha256(path), groups


def require_runnable(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = spec["generation"].get("model_snapshot")
    require(isinstance(binding, Mapping), "root model snapshot binding is absent")
    require(set(binding) == {"manifest_path", "manifest_sha256", "merged_weights_sha256", "tokenizer_json_sha256"}, "model binding fields mismatch")
    for key in ("manifest_sha256", "merged_weights_sha256", "tokenizer_json_sha256"):
        value = binding[key]
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "model binding hash invalid")
    manifest = Path(binding["manifest_path"])
    require(manifest.is_file() and sha256(manifest) == binding["manifest_sha256"], "model manifest identity mismatch")
    return binding


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_candidate_records(group: Mapping[str, Any], generations: Sequence[Mapping[str, Any]], rewards: Sequence[Mapping[str, Any]]) -> None:
    require(len(generations) == len(rewards) == 4, "group must contain exactly four candidates")
    row_id = group["row_id"]
    group_index = group["group_index"]
    expected = list(range(4))
    for records, label in ((generations, "generation"), (rewards, "reward")):
        require([record.get("candidate_index") for record in records] == expected, f"{label} candidate order mismatch")
        require(all(record.get("row_id") == row_id and record.get("group_index") == group_index for record in records), f"{label} group identity mismatch")
    for generation in generations:
        ids = generation.get("generated_ids")
        require(isinstance(ids, list) and ids and all(type(token) is int and token >= 0 for token in ids), "generated IDs invalid")
        require(generation.get("generated_token_count") == len(ids), "generated token count mismatch")
        require(isinstance(generation.get("generated_ids_sha256"), str), "generated ID hash missing")
        actual = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()
        require(actual == generation["generated_ids_sha256"], "generated ID hash mismatch")
        require(type(generation.get("cap_hit")) is bool, "cap-hit flag missing")
    for reward in rewards:
        value = reward.get("reward")
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)), "reward is not finite")
        require(type(reward.get("protocol_valid")) is bool, "protocol denominator missing")
        require(reward.get("parser_infrastructure_failure") is False, "parser infrastructure failure cannot become a reward record")
        require(type(reward.get("exact_region")) is bool, "exact denominator missing")
        require(type(reward.get("false_noop_edit")) is bool, "false-noop denominator missing")
        repetition = reward.get("repetition")
        require(isinstance(repetition, Mapping) and type(repetition.get("detected")) is bool, "repetition denominator missing")
        parse_status = reward.get("candidate_parse")
        mode = reward.get("syntax_evidence_mode")
        require(mode == group["syntax_evidence_mode"], "reward syntax mode differs from selected group")
        if mode == "unverified":
            require(parse_status == "unavailable_unverified_buffer", "unverified syntax row called or claimed parser evidence")
        else:
            require(parse_status in {"passed", "failed", "not_checked"}, "syntax parse status invalid")


def record_infrastructure_failure(output: Path, *, spec_sha256: str,
                                  model_identity: Mapping[str, Any],
                                  group: Mapping[str, Any], candidate_index: int,
                                  error: BaseException) -> Path:
    """Persist parser/service failure separately; the group stays uncommitted."""
    require(type(candidate_index) is int and 0 <= candidate_index < 4, "failure candidate index invalid")
    root = output / "infrastructure-failures"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{int(group['group_index']):06d}-{candidate_index}.json"
    require(not path.exists(), "infrastructure failure record already exists")
    payload = {
        "schema": "sepalith.rl11.signal-pilot-infrastructure-failure.v1",
        "spec_sha256": spec_sha256,
        "model_identity": dict(model_identity),
        "group_index": group["group_index"],
        "row_id": group["row_id"],
        "candidate_index": candidate_index,
        "failure_class": type(error).__name__,
        "failure": str(error),
        "reward_emitted": False,
        "group_committed": False,
    }
    _write_fsynced(path, canonical(payload) + b"\n")
    return path


def commit_group(output: Path, *, spec_sha256: str, model_identity: Mapping[str, Any], group: Mapping[str, Any], generations: Sequence[Mapping[str, Any]], rewards: Sequence[Mapping[str, Any]]) -> Path:
    """Publish one group atomically; an interrupted attempt is never resumed as complete."""
    _validate_candidate_records(group, generations, rewards)
    root = output / "groups"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{int(group['group_index']):06d}"
    require(not final.exists(), "group already committed")
    attempt = Path(tempfile.mkdtemp(prefix=f".attempt-{int(group['group_index']):06d}-", dir=root))
    try:
        generation_payload = b"".join(canonical(record) + b"\n" for record in generations)
        reward_payload = b"".join(canonical(record) + b"\n" for record in rewards)
        _write_fsynced(attempt / "generation-records.jsonl", generation_payload)
        _write_fsynced(attempt / "reward-records.jsonl", reward_payload)
        receipt = {
            "schema": "sepalith.rl11.signal-pilot-group-receipt.v1",
            "spec_sha256": spec_sha256,
            "model_identity": dict(model_identity),
            "group_index": group["group_index"],
            "row_id": group["row_id"],
            "candidate_count": 4,
            "generation_sha256": hashlib.sha256(generation_payload).hexdigest(),
            "reward_sha256": hashlib.sha256(reward_payload).hexdigest(),
        }
        _write_fsynced(attempt / "receipt.json", canonical(receipt) + b"\n")
        os.replace(attempt, final)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return final
    except BaseException:
        if attempt.exists():
            shutil.rmtree(attempt)
        raise


def validate_committed(output: Path, *, spec_sha256: str, model_identity: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    root = output / "groups"
    if not root.exists():
        return {}
    committed: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for path in sorted(root.iterdir()):
        if path.name.startswith(".attempt-"):
            continue
        require(path.is_dir() and path.name.isdigit(), "unexpected pilot output entry")
        index = int(path.name)
        require(index < len(groups) and index not in committed, "committed group index invalid")
        receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
        require(receipt.get("spec_sha256") == spec_sha256 and receipt.get("model_identity") == dict(model_identity), "resume identity mismatch")
        require(receipt.get("group_index") == index and receipt.get("row_id") == groups[index]["row_id"], "group receipt identity mismatch")
        generation_path = path / "generation-records.jsonl"
        reward_path = path / "reward-records.jsonl"
        require(sha256(generation_path) == receipt.get("generation_sha256") and sha256(reward_path) == receipt.get("reward_sha256"), "committed group hash mismatch")
        generations = [json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()]
        rewards = [json.loads(line) for line in reward_path.read_text(encoding="utf-8").splitlines()]
        _validate_candidate_records(groups[index], generations, rewards)
        committed[index] = (generations, rewards)
    present = sorted(committed)
    require(present == list(range(len(present))), "resume has a gap in committed prompt groups")
    return committed


def summarize(committed: Mapping[int, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    family_groups: dict[str, Counter[str]] = defaultdict(Counter)
    variances: list[dict[str, Any]] = []
    for index, (generations, rewards) in sorted(committed.items()):
        group = groups[index]
        values = [float(record["reward"]) for record in rewards]
        mean = sum(values) / 4
        variance = sum((value - mean) ** 2 for value in values) / 4
        nonzero = variance > 0.0
        family_groups[group["family"]]["groups"] += 1
        family_groups[group["family"]]["nonzero_variance_groups"] += int(nonzero)
        counters["prompt_groups"] += 1
        counters["nonzero_variance_groups"] += int(nonzero)
        variances.append({"group_index": index, "row_id": group["row_id"], "family": group["family"], "syntax_evidence_mode": group["syntax_evidence_mode"], "target_band": group["target_band"], "reward_mean": mean, "reward_population_variance": variance})
        for generation, reward in zip(generations, rewards):
            counters["raw_outputs"] += 1
            counters["raw_nonempty"] += int(bool(generation["generated_ids"]))
            counters["cap_hits"] += int(generation["cap_hit"])
            counters["protocol_valid"] += int(reward["protocol_valid"])
            counters["protocol_invalid"] += int(not reward["protocol_valid"])
            counters["protocol_unterminated"] += int(reward.get("protocol_failure") in {"missing_terminal", "unterminated", "cap_hit"})
            counters["exact_region"] += int(reward["exact_region"])
            counters["false_noop_all"] += int(reward["false_noop_edit"])
            if group["family"] == "no_op":
                counters["noop_gold_candidates"] += 1
                counters["false_noop_on_noop_gold"] += int(reward["false_noop_edit"])
            if group["family"] == "finish_block":
                counters["finish_candidates"] += 1
                counters["finish_cap_hits"] += int(generation["cap_hit"])
                counters["finish_protocol_unterminated"] += int(
                    reward.get("protocol_failure") in {"missing_terminal", "unterminated", "cap_hit"}
                )
                counters["finish_valid_nonexact_syntax_passed"] += int(
                    reward["protocol_valid"] and not reward["exact_region"]
                    and reward["candidate_parse"] == "passed"
                )
                counters["finish_applied_syntax_failed"] += int(
                    reward["candidate_parse"] == "failed"
                )
                counters["finish_severe_repetition"] += int(
                    bool(reward["repetition"]["detected"])
                )
            if reward["protocol_valid"] and not reward["exact_region"]:
                counters["repetition_eligible"] += 1
                counters["severe_repetition"] += int(reward["repetition"]["detected"])
            status = reward["candidate_parse"]
            if group["syntax_evidence_mode"] == "unverified":
                counters["syntax_unverified"] += 1
            elif status in {"passed", "failed"}:
                counters["syntax_checked"] += 1
                counters[f"syntax_{status}"] += 1
            else:
                counters["syntax_not_checked_due_prior_outcome"] += 1
    return {
        "schema": "sepalith.rl11.signal-pilot-summary.v1",
        "complete": len(committed) == len(groups),
        "denominators": dict(counters),
        "group_reward_variance": variances,
        "family_group_signal": {family: dict(values) for family, values in sorted(family_groups.items())},
        "semantic_claim": "exact target-region equality only; syntax validity is separate and freeform semantic correctness is not measured",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-spec", "next-group", "summarize"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    spec, spec_sha, groups = load_spec(args.spec)
    if args.command == "validate-spec":
        print(json.dumps({"status": "pass", "spec_sha256": spec_sha, "groups": len(groups), "runnable": spec["generation"]["model_snapshot"] is not None}, sort_keys=True))
        return 0
    require(args.output is not None, "--output is required")
    binding = require_runnable(spec)
    committed = validate_committed(args.output, spec_sha256=spec_sha, model_identity=binding, groups=groups)
    if args.command == "next-group":
        print(json.dumps({"committed_groups": len(committed), "next_group": None if len(committed) == len(groups) else groups[len(committed)]}, sort_keys=True))
    else:
        print(json.dumps(summarize(committed, groups), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
