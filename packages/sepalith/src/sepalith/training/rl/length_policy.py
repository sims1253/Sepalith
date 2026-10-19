"""Fail-closed, data-bound length policy for TRAIN-only RL rows (card C11).

Ported from the reviewed RL-11 packet
``docs/campaign/work/lead/r2-full-weight-rl-length-policy-v1/rl_length_policy.py``
(sha256 ``6ea7015b...0ccb``) with type annotations added. The behaviour is
unchanged: a complete stored row (prompt including the manual BOS, target
including the terminal EOS) is either admitted whole or rejected with named
reasons. Nothing is ever truncated.

``REVIEWED_POLICY`` is the policy the RL-11 root review accepted
(``RL-11-length-policy-root-review.json``): 3,072 prompt tokens including BOS,
1,024 completion tokens including EOS, 4,096 context. It replaces the 2,048 and
192 limits the RL driver used to hardcode, which rejected 4,106 of the 15,006
TRAIN rows. Full-weight GRPO memory fit at this geometry is still unproven;
card C14 runs that probe.

Run the census:

    python -m sepalith.training.rl.length_policy --rows TRAIN_ROWS.jsonl --output CENSUS.json
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "sepalith.rl11.full-weight-rl-length-policy.v1"
CENSUS_SCHEMA = "sepalith.rl11.length-census.v1"
VOCAB_SIZE = 130_560
BOS_ID = 0
EOS_ID = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DISPOSITIONS = frozenset({"audit_only", "candidate_pending_resource_admission", "root_admitted"})
TRAIN_ROWS_SHA256 = "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"
# DAT10 v4-braceindent: the v3 rows with the final closing brace of 486 repaired
# finish_block targets re-indented (card C04). Prompts and ranges are unchanged.
# Arm A was trained on these rows and the C12 census samples their prompts.
TRAIN_ROWS_V4_BRACEINDENT_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"


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
    def from_mapping(cls, value: Mapping[str, object]) -> LengthPolicy:
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
        if not isinstance(disposition, str) or disposition not in DISPOSITIONS:
            raise LengthPolicyError("disposition is invalid")
        return cls(policy_id, digest, context, prompt, completion, disposition)

    def to_mapping(self) -> dict[str, object]:
        return {"schema": SCHEMA, **asdict(self)}

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


LEGACY_POLICY = LengthPolicy.from_mapping({
    "schema": SCHEMA,
    "policy_id": "legacy-2048-prompt-192-completion-audit-only",
    "input_rows_sha256": TRAIN_ROWS_SHA256,
    "model_context_tokens": 2240,
    "prompt_max_tokens": 2048,
    "completion_max_tokens": 192,
    "disposition": "audit_only",
})
REVIEWED_POLICY = LengthPolicy.from_mapping({
    "schema": SCHEMA,
    "policy_id": "current-15006-complete-4096-candidate",
    "input_rows_sha256": TRAIN_ROWS_SHA256,
    "model_context_tokens": 4096,
    "prompt_max_tokens": 3072,
    "completion_max_tokens": 1024,
    "disposition": "candidate_pending_resource_admission",
})
# The reviewed limits bound to the v4-braceindent rows (census:
# receipts/C12-length-census-v4.json, every row accepted).
REVIEWED_POLICY_V4 = LengthPolicy.from_mapping({
    **REVIEWED_POLICY.to_mapping(),
    "policy_id": "v4-braceindent-15006-complete-4096-candidate",
    "input_rows_sha256": TRAIN_ROWS_V4_BRACEINDENT_SHA256,
})


def validate_complete_train_row(row: Mapping[str, object]) -> tuple[int, int, int]:
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
    tokens = [token for token in ids if isinstance(token, int)]  # all ints, checked above
    prompt = tokens[:start]
    completion = tokens[start:]
    if prompt[0] != BOS_ID or prompt.count(BOS_ID) != 1 or EOS_ID in prompt:
        raise LengthPolicyError(f"row {row_id} prompt violates one-BOS/no-EOS geometry")
    if not completion or completion[-1] != EOS_ID or completion.count(EOS_ID) != 1:
        raise LengthPolicyError(f"row {row_id} target lacks one terminal EOS")
    return len(prompt), len(completion), len(tokens)


def assess_row(row: Mapping[str, object], policy: LengthPolicy) -> dict[str, object]:
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


def require_row(row: Mapping[str, object], policy: LengthPolicy) -> Mapping[str, object]:
    result = assess_row(row, policy)
    if not result["accepted_complete"]:
        reasons = result["reasons"]
        assert isinstance(reasons, list)
        raise LengthPolicyError(
            f"row {result['row_id']} rejected without truncation: {','.join(str(reason) for reason in reasons)}"
        )
    return row


def _quantile(values: list[int], numerator: int, denominator: int) -> int:
    values = sorted(values)
    index = max(0, (len(values) * numerator + denominator - 1) // denominator - 1)
    return values[index]


def audit_rows(path: Path, policies: Iterable[LengthPolicy]) -> dict[str, object]:
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
                    found = result["reasons"]
                    assert isinstance(found, list)
                    for reason in map(str, found):
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
            "policy": asdict(policy), "policy_sha256": policy.sha256,
            "accepted_complete_rows": len(seen) - len(ids),
            "rejected_without_truncation_rows": len(ids),
            "rejected_reason_counts": reasons[policy.policy_id],
            "ordered_rejected_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    return {
        "schema": CENSUS_SCHEMA, "rows_path": str(path.resolve()),
        "rows_sha256": actual, "rows": len(seen), "families": families,
        "lengths": profiles, "policies": policy_results,
        "no_tokens_truncated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--policies", type=Path,
                        help="policy-set JSON; default: the legacy and reviewed policies")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.policies is None:
        policies = [LEGACY_POLICY, REVIEWED_POLICY]
    else:
        value = json.loads(args.policies.read_text(encoding="utf-8"))
        policies = [LengthPolicy.from_mapping(item) for item in value["policies"]]
    result = audit_rows(args.rows, policies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
