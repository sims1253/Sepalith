"""CPU-only tests for the isolated RL-08 cap contract candidate."""
from __future__ import annotations

import copy
import unittest

from runtime_cap_override_candidate import (
    CapOverrideError,
    MAX_FRACTION,
    SCHEMA,
    canonical_identity_sha256,
    derive_explicit_cap_identity,
    resolve_runtime_cap,
)


IDENTITY = {
    "parent": {"manifest_sha256": "a" * 64, "merged_weights_sha256": "b" * 64},
    "tokenizer": {"vocab_size": 130560, "bos_id": 0, "eos_id": 1},
    "renderer": {"renderer_id": "prm03"},
    "data": {"rows_sha256": "c" * 64},
    "source": {"snapshot": "d" * 64},
    "policy": {
        "cuda_memory_fraction": 0.75,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
    },
    "schedule": {"source_draw_schedule_sha256": "e" * 64},
}


def recipe(identity=IDENTITY):
    return {"cuda_memory_fraction": 0.75, "identity": identity}


def override(identity=IDENTITY, **changes):
    value = {
        "schema": SCHEMA,
        "identity_sha256": canonical_identity_sha256(identity),
        "from_cuda_memory_fraction": 0.75,
        "to_cuda_memory_fraction": 0.80,
        "review_receipt_sha256": "f" * 64,
        "reason": "reviewed bounded retry after update-104 allocator OOM",
    }
    value.update(changes)
    return value


class RuntimeCapOverrideTests(unittest.TestCase):
    def test_default_keeps_identity_and_runtime_at_point_seven_five(self):
        original = copy.deepcopy(IDENTITY)
        result = resolve_runtime_cap(recipe(), IDENTITY)
        self.assertEqual(result["configured_identity_cuda_memory_fraction"], 0.75)
        self.assertEqual(result["runtime_cuda_memory_fraction"], 0.75)
        self.assertIsNone(result["override"])
        self.assertEqual(IDENTITY, original)

    def test_explicit_override_selects_only_point_eight_without_mutating_identity(self):
        original = copy.deepcopy(IDENTITY)
        value = recipe()
        value["runtime_resource_override"] = override()
        result = resolve_runtime_cap(value, IDENTITY)
        self.assertEqual(result["runtime_cuda_memory_fraction"], MAX_FRACTION)
        self.assertEqual(result["configured_identity_cuda_memory_fraction"], 0.75)
        self.assertEqual(result["identity_sha256"], canonical_identity_sha256(IDENTITY))
        self.assertEqual(IDENTITY, original)

    def test_override_identity_binding_rejects_wrong_identity(self):
        value = recipe()
        value["runtime_resource_override"] = override(identity=IDENTITY, identity_sha256="0" * 64)
        with self.assertRaisesRegex(CapOverrideError, "identity does not match"):
            resolve_runtime_cap(value, IDENTITY)

    def test_override_rejects_non_maximum_fraction(self):
        for bad in (0.79, 0.800001, 0.0, float("nan")):
            value = recipe()
            value["runtime_resource_override"] = override(to_cuda_memory_fraction=bad)
            with self.subTest(bad=bad), self.assertRaises(CapOverrideError):
                resolve_runtime_cap(value, IDENTITY)

    def test_override_rejects_wrong_start_fraction(self):
        value = recipe()
        value["runtime_resource_override"] = override(from_cuda_memory_fraction=0.70)
        with self.assertRaisesRegex(CapOverrideError, "starts at"):
            resolve_runtime_cap(value, IDENTITY)

    def test_base_policy_mismatch_still_rejects_before_override(self):
        value = recipe()
        value["runtime_resource_override"] = override()
        value["cuda_memory_fraction"] = 0.80
        with self.assertRaisesRegex(CapOverrideError, "must equal"):
            resolve_runtime_cap(value, IDENTITY)

    def test_override_schema_keys_and_receipt_binding_are_strict(self):
        for change, expected in (({"extra": 1}, "keys are not exact"),
                                 ({"schema": "wrong"}, "schema mismatch"),
                                 ({"review_receipt_sha256": "not-a-sha"}, "lowercase SHA256")):
            value = recipe()
            value["runtime_resource_override"] = override(**change)
            with self.subTest(change=change), self.assertRaisesRegex(CapOverrideError, expected):
                resolve_runtime_cap(value, IDENTITY)

    def test_identity_cap_migration_is_explicit_and_original_hash_survives(self):
        original = copy.deepcopy(IDENTITY)
        old_hash = canonical_identity_sha256(IDENTITY)
        derived, hashes = derive_explicit_cap_identity(IDENTITY, from_fraction=0.75, to_fraction=0.80)
        self.assertEqual(hashes["source_identity_sha256"], old_hash)
        self.assertNotEqual(hashes["derived_identity_sha256"], old_hash)
        self.assertEqual(derived["policy"]["cuda_memory_fraction"], 0.80)
        self.assertEqual(set(derived["policy"]) - {"cuda_memory_fraction"}, set(IDENTITY["policy"]) - {"cuda_memory_fraction"})
        self.assertEqual(IDENTITY, original)

    def test_identity_cap_migration_rejects_wrong_source_or_target(self):
        with self.assertRaises(CapOverrideError):
            derive_explicit_cap_identity(IDENTITY, from_fraction=0.70, to_fraction=0.80)
        with self.assertRaises(CapOverrideError):
            derive_explicit_cap_identity(IDENTITY, from_fraction=0.75, to_fraction=0.79)


if __name__ == "__main__":
    unittest.main()
