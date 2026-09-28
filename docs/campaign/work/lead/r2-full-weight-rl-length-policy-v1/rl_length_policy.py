#!/usr/bin/env python3
"""Fail-closed, data-bound length policy for TRAIN-only RL rows."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA = "sepalith.rl11.full-weight-rl-length-policy.v1"
VOCAB_SIZE = 130_560
BOS_ID = 0
EOS_ID = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LengthPolicyError(ValueError):
    pass


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise LengthPolicyError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class LengthPolicy:
    policy_id: str
    input_rows_sha256: str
    model_context_tokens: int
    prompt_max_tokens: int
    completion_max_tokens: int
    disposition: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LengthPolicy":
        if set(value) != {
            "schema", "policy_id", "input_rows_sha256", "model_context_tokens",
            "prompt_max_tokens", "completion_max_tokens", "disposition",
        }:
            raise LengthPolicyError("length policy fields are not exact")
        if value["schema"] != SCHEMA:
            raise LengthPolicyError("length policy schema mismatch")
        policy_id = value["policy_id"]
        if not isinstance(policy_id, str) or not policy_id:
            raise LengthPolicyError("policy_id must be a nonempty string")
        digest = value["input_rows_sha256"]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise LengthPolicyError("input_rows_sha256 is invalid")
        context = _integer(value["model_context_tokens"], "model_context_tokens")
        prompt = _integer(value["prompt_max_tokens"], "prompt_max_tokens")
        completion = _integer(value["completion_max_tokens"], "completion_max_tokens")
        if prompt + completion > context:
            raise LengthPolicyError("prompt_max_tokens + completion_max_tokens exceeds model context")
        disposition = value["disposition"]
        if disposition not in {"audit_only", "candidate_pending_resource_admission", "root_admitted"}:
            raise LengthPolicyError("disposition is invalid")
        return cls(policy_id, digest, context, prompt, completion, disposition)


def validate_complete_train_row(row: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return prompt/completion/sequence lengths; never modify or slice a row."""
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise LengthPolicyError("row id is invalid")
    if row.get("split") != "train":
        raise LengthPolicyError(f"row {row_id} is not TRAIN")
    ids = row.get("input_ids")
    if not isinstance(ids, list) or not ids:
        raise LengthPolicyError(f"row {row_id} input_ids is invalid")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids):
        raise LengthPolicyError(f"row {row_id} contains an invalid token ID")
    start = row.get("target_start")
    prompt_count = row.get("prompt_token_count")
    target_count = row.get("target_token_count")
    if type(start) is not int or type(prompt_count) is not int or type(target_count) is not int:
        raise LengthPolicyError(f"row {row_id} length fields must be integers")
    if start != prompt_count + 1:
        raise LengthPolicyError(f"row {row_id} target_start does not include exactly one manual BOS")
    if len(ids) != 1 + prompt_count + target_count + 1:
        raise LengthPolicyError(f"row {row_id} stored counts do not cover the complete sequence")
    prompt = ids[:start]
    completion = ids[start:]
    if prompt[0] != BOS_ID or prompt.count(BOS_ID) != 1 or EOS_ID in prompt:
        raise LengthPolicyError(f"row {row_id} prompt violates one-BOS/no-EOS geometry")
    if not completion or completion[-1] != EOS_ID or completion.count(EOS_ID) != 1:
        raise LengthPolicyError(f"row {row_id} target lacks one terminal EOS")
    return len(prompt), len(completion), len(ids)


def assess_row(row: Mapping[str, Any], policy: LengthPolicy) -> dict[str, Any]:
    prompt, completion, sequence = validate_complete_train_row(row)
    reasons = []
    if prompt > policy.prompt_max_tokens:
        reasons.append("prompt_over_limit")
    if completion > policy.completion_max_tokens:
        reasons.append("completion_over_limit")
    if sequence > policy.model_context_tokens:
        reasons.append("sequence_over_limit")
    return {
        "row_id": row["id"], "prompt_tokens": prompt,
        "completion_tokens_including_eos": completion,
        "sequence_tokens": sequence, "accepted_complete": not reasons,
        "reasons": reasons,
    }


def require_row(row: Mapping[str, Any], policy: LengthPolicy) -> Mapping[str, Any]:
    result = assess_row(row, policy)
    if not result["accepted_complete"]:
        raise LengthPolicyError(
            f"row {result['row_id']} rejected without truncation: {','.join(result['reasons'])}"
        )
    return row


def _quantile(values: list[int], numerator: int, denominator: int) -> int:
    values = sorted(values)
    index = max(0, (len(values) * numerator + denominator - 1) // denominator - 1)
    return values[index]


def audit_rows(path: Path, policies: Iterable[LengthPolicy]) -> dict[str, Any]:
    policies = tuple(policies)
    if not policies:
        raise LengthPolicyError("at least one policy is required")
    expected = policies[0].input_rows_sha256
    if any(policy.input_rows_sha256 != expected for policy in policies):
        raise LengthPolicyError("policies bind different input rows")
    before = path.stat()
    digest = hashlib.sha256()
    lengths: dict[str, list[int]] = {"prompt": [], "completion_including_eos": [], "sequence": []}
    families: dict[str, int] = {}
    rejected: dict[str, list[str]] = {policy.policy_id: [] for policy in policies}
    reasons: dict[str, dict[str, int]] = {policy.policy_id: {} for policy in policies}
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LengthPolicyError(f"invalid JSON at line {line_number}") from error
            row_id = row.get("id")
            if row_id in seen:
                raise LengthPolicyError(f"duplicate row id {row_id}")
            seen.add(row_id)
            prompt, completion, sequence = validate_complete_train_row(row)
            lengths["prompt"].append(prompt)
            lengths["completion_including_eos"].append(completion)
            lengths["sequence"].append(sequence)
            family = row.get("family")
            if not isinstance(family, str) or not family:
                raise LengthPolicyError(f"row {row_id} family is invalid")
            families[family] = families.get(family, 0) + 1
            for policy in policies:
                result = assess_row(row, policy)
                if not result["accepted_complete"]:
                    rejected[policy.policy_id].append(row_id)
                    for reason in result["reasons"]:
                        reasons[policy.policy_id][reason] = reasons[policy.policy_id].get(reason, 0) + 1
    after = path.stat()
    actual = digest.hexdigest()
    if actual != expected:
        raise LengthPolicyError(f"input rows hash mismatch: {actual}")
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise LengthPolicyError("input rows changed during streaming audit")
    profiles = {}
    for name, values in lengths.items():
        profiles[name] = {
            "min": min(values), "p50": _quantile(values, 50, 100),
            "p90": _quantile(values, 90, 100), "p95": _quantile(values, 95, 100),
            "p99": _quantile(values, 99, 100), "max": max(values),
        }
    policy_results = {}
    for policy in policies:
        ids = rejected[policy.policy_id]
        policy_results[policy.policy_id] = {
            "policy": policy.__dict__, "accepted_complete_rows": len(seen) - len(ids),
            "rejected_without_truncation_rows": len(ids),
            "rejected_reason_counts": reasons[policy.policy_id],
            "ordered_rejected_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    return {
        "schema": "sepalith.rl11.length-census.v1", "rows_path": str(path.resolve()),
        "rows_sha256": actual, "rows": len(seen), "families": families,
        "lengths": profiles, "policies": policy_results,
        "no_tokens_truncated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.policies.read_text(encoding="utf-8"))
    policies = [LengthPolicy.from_mapping(item) for item in value["policies"]]
    result = audit_rows(args.rows, policies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
