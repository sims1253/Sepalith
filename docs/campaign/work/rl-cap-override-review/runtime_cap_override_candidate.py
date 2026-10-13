"""Isolated RL-08 runtime-cap contract candidate.

This module is a review fixture only. It is not imported by EXEC and does not
load a model, checkpoint, training framework, or campaign data.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

SCHEMA = "sepalith.rl.runtime-resource-override.v1"
MAX_FRACTION = 0.80
_OVERRIDE_KEYS = frozenset({
    "schema",
    "identity_sha256",
    "from_cuda_memory_fraction",
    "to_cuda_memory_fraction",
    "review_receipt_sha256",
    "reason",
})


class CapOverrideError(ValueError):
    """The explicit resource override is not admissible."""


def canonical_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Match the canonical identity serialization used by the preparation receipt."""
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapOverrideError(f"{name} must be finite and in (0, {MAX_FRACTION}]")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= MAX_FRACTION:
        raise CapOverrideError(f"{name} must be finite and in (0, {MAX_FRACTION}]")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CapOverrideError(f"{name} must be a lowercase SHA256")
    return value


def resolve_runtime_cap(recipe: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve .75 by default or one explicit reviewed .80 override.

    The identity object is never mutated. The override is deliberately narrow:
    it is bound to the canonical identity, must start at the configured value,
    and can only select the already-coded hard maximum .80.
    """
    configured = _fraction(recipe.get("cuda_memory_fraction"), "recipe.cuda_memory_fraction")
    policy = identity.get("policy")
    if not isinstance(policy, Mapping):
        raise CapOverrideError("identity.policy must be an object")
    identity_fraction = _fraction(
        policy.get("cuda_memory_fraction"), "identity.policy.cuda_memory_fraction"
    )
    if configured != identity_fraction:
        raise CapOverrideError(
            "recipe.cuda_memory_fraction must equal identity.policy.cuda_memory_fraction"
        )

    identity_sha = canonical_identity_sha256(identity)
    override = recipe.get("runtime_resource_override")
    if override is None:
        return {
            "configured_identity_cuda_memory_fraction": configured,
            "runtime_cuda_memory_fraction": configured,
            "identity_sha256": identity_sha,
            "override": None,
        }
    if not isinstance(override, Mapping):
        raise CapOverrideError("runtime_resource_override must be an object")
    if set(override) != _OVERRIDE_KEYS:
        raise CapOverrideError("runtime_resource_override keys are not exact")
    if override.get("schema") != SCHEMA:
        raise CapOverrideError("runtime_resource_override schema mismatch")
    if _sha(override.get("identity_sha256"), "runtime_resource_override.identity_sha256") != identity_sha:
        raise CapOverrideError("runtime_resource_override identity does not match recipe identity")
    from_fraction = _fraction(
        override.get("from_cuda_memory_fraction"),
        "runtime_resource_override.from_cuda_memory_fraction",
    )
    to_fraction = _fraction(
        override.get("to_cuda_memory_fraction"),
        "runtime_resource_override.to_cuda_memory_fraction",
    )
    if from_fraction != configured:
        raise CapOverrideError("runtime resource override starts at the configured fraction")
    if to_fraction != MAX_FRACTION:
        raise CapOverrideError("runtime resource override may select only the .80 hard maximum")
    receipt_sha = _sha(
        override.get("review_receipt_sha256"),
        "runtime_resource_override.review_receipt_sha256",
    )
    reason = override.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
        raise CapOverrideError("runtime_resource_override.reason must be a bounded nonempty string")
    return {
        "configured_identity_cuda_memory_fraction": configured,
        "runtime_cuda_memory_fraction": to_fraction,
        "identity_sha256": identity_sha,
        "override": {
            "schema": SCHEMA,
            "identity_sha256": identity_sha,
            "from_cuda_memory_fraction": from_fraction,
            "to_cuda_memory_fraction": to_fraction,
            "review_receipt_sha256": receipt_sha,
            "reason": reason,
        },
    }


def derive_explicit_cap_identity(
    identity: Mapping[str, Any], *, from_fraction: float, to_fraction: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Prepare a new checkpoint identity for the no-source-change migration option."""
    current_hash = canonical_identity_sha256(identity)
    current = _fraction(identity.get("policy", {}).get("cuda_memory_fraction"), "identity.policy.cuda_memory_fraction")
    if current != from_fraction:
        raise CapOverrideError("migration source fraction does not match identity")
    target = _fraction(to_fraction, "target_cuda_memory_fraction")
    if target != MAX_FRACTION:
        raise CapOverrideError("migration target must be the .80 hard maximum")
    derived = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    derived["policy"]["cuda_memory_fraction"] = target
    derived_hash = canonical_identity_sha256(derived)
    return derived, {
        "source_identity_sha256": current_hash,
        "derived_identity_sha256": derived_hash,
    }
