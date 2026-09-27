"""CPU-only functional checks for the DAT-06 finite sampler.

All rows in this file are synthetic admission metadata.  They intentionally do
not contain prompt text, target text, or token IDs, and they are not observed
editor coverage.
"""

from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from campaign_sampling import (  # noqa: E402
    SamplerInputError,
    build_draw_manifest,
    canonical_token_rows_sha256,
    validate_admitted_rows,
    validate_draw_manifest,
)


def row(
    row_id: str,
    family: str,
    *,
    source_id: str | None = None,
    package_id: str = "synthetic-package",
    noop: bool = False,
    prompt_tokens: int = 80,
    target_tokens: int = 20,
    length: str = "short",
    source_kind: str = "ordinary",
    split: str = "train",
    **extra,
):
    total = prompt_tokens + target_tokens
    if length == "long" and total <= 2048:
        prompt_tokens = 2500 - target_tokens
        total = prompt_tokens + target_tokens
    return {
        "row_id": row_id,
        "family": family,
        "source_id": source_id or f"source-{row_id}",
        "package_id": package_id,
        "split": split,
        "semantic_noop": noop,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "total_tokens": total,
        "length_bucket": length,
        "naturally_long": length == "long",
        "source_kind": source_kind,
        **extra,
    }


def balanced_rows(
    *,
    families: int = 5,
    rows_per_family: int = 20,
    noops: int = 20,
    source_kind: str = "ordinary",
):
    result = [
        row(f"noop-{index:03d}", "no_op", source_id=f"noop-source-{index % 4}", noop=True)
        for index in range(noops)
    ]
    for family_index in range(families):
        family = f"family-{family_index}"
        result.extend(
            row(
                f"{family}-row-{index:03d}", family,
                source_id=f"{family}-source-{index % 4}", source_kind=source_kind,
            )
            for index in range(rows_per_family)
        )
    return result


class CampaignSamplingTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_hash_covers_exposure(self):
        rows = balanced_rows()
        kwargs = dict(
            max_steps=10, effective_batch=10, split_id="synthetic-train-v1", seed=101,
            ordinary_replay_cap=4,
        )
        first = build_draw_manifest(rows, **kwargs)
        second = build_draw_manifest(copy.deepcopy(rows), **kwargs)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["schedule_sha256"], second["schedule_sha256"])
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first["row_ids"], second["row_ids"])
        self.assertEqual(first["token_rows_sha256"], canonical_token_rows_sha256(rows))
        self.assertEqual(first["draw_count"], 100)
        self.assertEqual(sum(first["exposure"]["row"].values()), 100)
        self.assertEqual(sum(first["exposure"]["family"].values()), 100)
        validate_draw_manifest(first)
        altered = build_draw_manifest(rows, **{**kwargs, "seed": 102})
        self.assertNotEqual(first["schedule_sha256"], altered["schedule_sha256"])

    def test_noop_reservation_family_ceiling_and_finite_replay_cap(self):
        manifest = build_draw_manifest(
            balanced_rows(), max_steps=10, effective_batch=10,
            split_id="synthetic-train-v1", seed=7, ordinary_replay_cap=4,
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["achieved_mixture"]["semantic_noop"]["achieved"], 10)
        self.assertEqual(manifest["achieved_mixture"]["semantic_noop"]["reserved"], 10)
        for family, count in manifest["exposure"]["family"].items():
            if family != "no_op":
                self.assertLessEqual(count, manifest["policy"]["effective_family_ceiling"])
        for row_id, count in manifest["exposure"]["row"].items():
            self.assertLessEqual(
                count,
                manifest["exposure"]["row_eligibility"][row_id]["capacity"],
            )
        self.assertEqual(manifest["quota_ledger"]["backfill_draws"], 0)

    def test_small_pack_rows_rotate_before_replay_and_stop_at_three(self):
        rows = [
            row(f"small-{index}", "small-family", source_id="one-source", source_kind="small_pack")
            for index in range(4)
        ]
        rows.extend(
            row(f"ordinary-{family}-{index}", family, source_id=f"{family}-source-{index % 2}")
            for family in ("family-a", "family-b", "family-c")
            for index in range(16)
        )
        rows.extend(row(f"noop-{index}", "no_op", noop=True) for index in range(8))
        manifest = build_draw_manifest(
            rows, max_steps=4, effective_batch=10, split_id="small-pack-v1", seed=33,
            ordinary_replay_cap=5,
        )
        self.assertEqual(manifest["status"], "complete")
        small_draws = [draw for draw in manifest["draws"] if draw["source_kind"] == "small_pack"]
        self.assertGreaterEqual(len(small_draws), 5)
        self.assertEqual(len({draw["row_id"] for draw in small_draws[:4]}), 4)
        self.assertTrue(any(draw["presentation"] == 2 for draw in small_draws))
        for draw in small_draws:
            self.assertLessEqual(draw["presentation"], 3)
        self.assertLessEqual(
            max(manifest["exposure"]["row"][draw["row_id"]] for draw in small_draws),
            3,
        )

    def test_rare_family_quota_shortage_is_backfilled_without_global_deficit(self):
        rows = [row(f"noop-{i}", "no_op", noop=True) for i in range(20)]
        for family in ("a", "b", "c", "d", "e"):
            rows.extend(row(f"{family}-{i}", family) for i in range(20))
        rows.append(row("rare-only", "rare"))
        manifest = build_draw_manifest(
            rows, max_steps=10, effective_batch=10, split_id="scarce-v1", seed=3,
            ordinary_replay_cap=2,
        )
        self.assertEqual(manifest["status"], "complete")
        rare = manifest["quota_ledger"]["families"]["rare"]
        self.assertGreater(rare["remaining_deficit"], 0)
        self.assertGreater(manifest["quota_ledger"]["backfill_draws"], 0)
        self.assertEqual(manifest["deficits"]["global"], 0)
        self.assertLessEqual(manifest["exposure"]["row"]["rare-only"], 2)
        for family, ledger in manifest["quota_ledger"]["families"].items():
            if ledger["backfill"]:
                family_draws = [draw for draw in manifest["draws"] if draw["family"] == family]
                # Backfill must consume a new row after the initial rotation,
                # rather than restarting at the first row with presentation 2.
                self.assertTrue(all(draw["presentation"] == 1 for draw in family_draws))

    def test_mixed_noop_and_edit_family_obeys_ceiling_and_prefix_is_interleaved(self):
        rows = [row(f"a-noop-{i}", "a", source_id=f"noop-source-{i % 3}", noop=True)
                for i in range(30)]
        rows.extend(row(f"a-edit-{i}", "a", source_id=f"a-source-{i % 3}") for i in range(30))
        for family in ("b", "c", "d"):
            rows.extend(row(f"{family}-{i}", family, source_id=f"{family}-source-{i % 3}")
                        for i in range(30))
        manifest = build_draw_manifest(
            rows, max_steps=10, effective_batch=10, split_id="mixed-family-v2", seed=9,
            ordinary_replay_cap=3,
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertLessEqual(manifest["exposure"]["family"]["a"], 25)
        self.assertEqual(manifest["achieved_mixture"]["semantic_noop"]["achieved"], 10)
        # Every non-final prefix carries both semantic classes.  This catches
        # the old all-no-op-then-family-block ordering.
        for milestone in manifest["prefix_milestones"][:-1]:
            self.assertGreater(milestone["semantic_noop"], 0)
            self.assertGreater(sum(milestone["families"].values()) - milestone["semantic_noop"], 0)
        validate_draw_manifest(manifest)

    def test_global_scarcity_is_infeasible_and_never_replays_forever(self):
        rows = [row(f"noop-{i}", "no_op", noop=True) for i in range(20)]
        for family in ("a", "b"):
            rows.extend(row(f"{family}-{i}", family) for i in range(20))
        manifest = build_draw_manifest(
            rows, max_steps=10, effective_batch=10, split_id="scarce-v1", seed=4,
            ordinary_replay_cap=2,
        )
        self.assertEqual(manifest["status"], "infeasible")
        self.assertGreater(manifest["deficits"]["global"], 0)
        self.assertEqual(len(manifest["draws"]), manifest["draw_count"])
        self.assertIn("finite_row_or_family_capacity", manifest["deficits"]["reasons"])
        validate_draw_manifest(manifest)

    def test_heldout_and_raw_target_metadata_are_rejected(self):
        with self.assertRaisesRegex(SamplerInputError, "not admitted train"):
            validate_admitted_rows([row("heldout", "a", split="dev")])
        with self.assertRaisesRegex(SamplerInputError, "raw prompt/target"):
            validate_admitted_rows([row("raw", "a", target_text="future target")])
        bad = row("truncated", "a", target_truncated=True)
        with self.assertRaisesRegex(SamplerInputError, "truncated"):
            validate_admitted_rows([bad])

    def test_long_fraction_ceiling_and_no_truncation_admission(self):
        rows = [row(f"noop-{i}", "no_op", noop=True) for i in range(20)]
        rows.extend(row(f"short-a-{i}", "a") for i in range(20))
        rows.extend(row(f"short-b-{i}", "b") for i in range(20))
        rows.extend(row(f"short-c-{i}", "c") for i in range(20))
        rows.extend(row(f"short-d-{i}", "d") for i in range(20))
        rows.extend(row(f"long-{i}", "long-family", length="long") for i in range(20))
        manifest = build_draw_manifest(
            rows, max_steps=10, effective_batch=10, split_id="length-v1", seed=8,
            ordinary_replay_cap=3, naturally_long_fraction=0.20,
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertLessEqual(manifest["achieved_mixture"]["length"]["long"], 20)
        self.assertLessEqual(manifest["achieved_mixture"]["length"]["long_fraction"], 0.20)
        with self.assertRaisesRegex(SamplerInputError, "exceeds 4096"):
            validate_admitted_rows([row("too-long", "a", length="long", prompt_tokens=4090, target_tokens=20)])

    def test_custom_natural_length_bounds_are_used_for_metadata_digest(self):
        custom = row("long-5000", "long-family", length="long", prompt_tokens=4980, target_tokens=20)
        self.assertEqual(
            len(validate_admitted_rows([custom], short_max_tokens=4096, long_max_tokens=8192)),
            1,
        )
        manifest = build_draw_manifest(
            [row("noop", "no_op", noop=True), custom],
            max_steps=1, effective_batch=1, split_id="custom-bounds-v2", seed=1,
            short_max_tokens=4096, long_max_tokens=8192, naturally_long_fraction=1.0,
        )
        self.assertEqual(manifest["token_rows_sha256"], canonical_token_rows_sha256(
            [row("noop", "no_op", noop=True), custom],
            short_max_tokens=4096, long_max_tokens=8192,
        ))

    def test_manual_row_cannot_bypass_geometry_or_identity_checks(self):
        with self.assertRaisesRegex(SamplerInputError, "total_tokens"):
            validate_admitted_rows([
                row("bad-total", "a", total_tokens=999),
            ])
        with self.assertRaisesRegex(SamplerInputError, "duplicate"):
            validate_admitted_rows([row("same", "a"), row("same", "b")])

    def test_48000_draw_manifest_is_reproducible_with_exact_schedule_fields(self):
        # 6000 synthetic rows x ordinary replay ceiling 8 = exactly 48000 finite
        # presentations.  The rows are metadata-only and carry no model target.
        rows = [
            row(
                f"noop-{index:04d}", "no_op", source_id=f"noop-source-{index % 30}",
                noop=True,
            ) for index in range(600)
        ]
        for family in ("alpha", "beta", "gamma", "delta", "epsilon"):
            rows.extend(
                row(
                    f"{family}-{index:04d}", family,
                    source_id=f"{family}-source-{index % 30}",
                    length="long" if index < 120 else "short",
                ) for index in range(1080)
            )
        kwargs = dict(
            max_steps=3000, effective_batch=16, split_id="synthetic-48k-v1", seed=20260912,
            ordinary_replay_cap=8, naturally_long_fraction=0.20,
        )
        first = build_draw_manifest(rows, **kwargs)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(len(first["row_ids"]), 48000)
        self.assertEqual(first["max_steps"], 3000)
        self.assertEqual(first["effective_batch"], 16)
        self.assertEqual(first["split_id"], "synthetic-48k-v1")
        self.assertEqual(first["token_rows_sha256"], canonical_token_rows_sha256(rows))
        self.assertEqual(first["achieved_mixture"]["semantic_noop"]["fraction"], 0.1)
        self.assertLessEqual(first["achieved_mixture"]["length"]["long_fraction"], 0.2)
        self.assertEqual(first["exposure"]["distinct_rows"], 6000)
        self.assertEqual(sum(first["exposure"]["row"].values()), 48000)
        self.assertTrue(all(count <= 8 for count in first["exposure"]["row"].values()))
        validate_draw_manifest(first)
        second = build_draw_manifest(rows, **kwargs)
        self.assertEqual(first["schedule_sha256"], second["schedule_sha256"])
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
