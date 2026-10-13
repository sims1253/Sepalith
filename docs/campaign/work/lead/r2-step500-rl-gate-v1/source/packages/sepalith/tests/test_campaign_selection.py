"""CPU-only parity fixtures for the PRM-05 selection/dropout seam."""
from __future__ import annotations

import hashlib
import json
import unittest

from sepalith.campaign_selection import (
    EVIDENCE_DROPOUT_POLICY_ID,
    EvidenceCandidate,
    EvidenceSourceSnapshot,
    EvidenceSupport,
    SelectionScope,
    logical_lines,
    select_campaign_selection,
    select_evidence,
    selection_fixture_view,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item(identifier: str, content: str, required: bool) -> EvidenceCandidate:
    return EvidenceCandidate(identifier, content, None, required, digest(content), f"fixture/{identifier}", 1)


def current_sources(candidates: list[EvidenceCandidate]) -> list[EvidenceSourceSnapshot]:
    return [EvidenceSourceSnapshot(candidate.source_identity, candidate.source_version, candidate.source_sha256)
            for candidate in candidates]


def support_for(required: list[EvidenceCandidate], seed: int) -> EvidenceSupport:
    baseline = select_evidence(required=required, max_utf16_units=6000, current_sources=current_sources(required))
    return EvidenceSupport(
        verified=True,
        target_supported=True,
        policy_id=EVIDENCE_DROPOUT_POLICY_ID,
        seed=seed,
        required_ids=baseline.required_ids,
        required_fingerprint=baseline.required_fingerprint,
    )


def fixtures() -> list[tuple[str, dict[str, object]]]:
    unicode_text = "\r\n".join(
        [
            "function_header <- function(x) {",
            '  value <- "😀"',
            "  current <- x",
            "}",
            "trailer <- TRUE",
        ]
    )
    required = [item("required-contract", "required", True)]
    optional = [
        item("optional-a", "optional-a", False),
        item("optional-b", "optional-b", False),
        item("optional-c", "optional-c", False),
        item("optional-d", "optional-d", False),
        item("optional-e", "optional-e", False),
    ]
    support = support_for(required, 17)
    large_scope_lines = [
        "before <- 1",
        "before2 <- 2",
        "big <- function(x) {",
        *[f"  body_{index} <- {'x' * 18}" for index in range(14)],
        "}",
        "after <- 3",
    ]
    return [
        (
            "unsaved_unicode_crlf",
            dict(
                lines=logical_lines(unicode_text), region_start_line=2, region_end_line=2,
                scope=SelectionScope(0, 3), max_utf16_units=80,
                document_sha256=digest(unicode_text), evidence_budget_utf16_units=8,
            ),
        ),
        (
            "large_required_scope_overflow",
            dict(
                lines=large_scope_lines, region_start_line=8, region_end_line=8,
                scope=SelectionScope(2, 17), max_utf16_units=48,
                document_sha256=digest("\n".join(large_scope_lines)), evidence_budget_utf16_units=10,
            ),
        ),
        (
            "missing_provider_balanced_source",
            dict(
                lines=["prefix far", "prefix near", "cursor", "suffix near", "suffix far"],
                region_start_line=2, region_end_line=2, max_utf16_units=32,
                document_sha256=digest("\n".join(["prefix far", "prefix near", "cursor", "suffix near", "suffix far"])),
                evidence_budget_utf16_units=10,
            ),
        ),
        (
            "required_only_evidence",
            dict(
                lines=["header", "cursor", "tail"], region_start_line=1, region_end_line=1,
                max_utf16_units=20, required_evidence=required, evidence_budget_utf16_units=20,
                current_evidence_sources=current_sources(required),
            ),
        ),
        (
            "verified_optional_dropout",
            dict(
                lines=["header", "cursor", "tail"], region_start_line=1, region_end_line=1,
                max_utf16_units=20, required_evidence=required, optional_evidence=optional,
                evidence_budget_utf16_units=70, evidence_support=support,
                current_evidence_sources=current_sources(required + optional),
            ),
        ),
        (
            "unverified_optional_overflow",
            dict(
                lines=["header", "cursor", "tail"], region_start_line=1, region_end_line=1,
                max_utf16_units=20, required_evidence=required, optional_evidence=optional,
                evidence_budget_utf16_units=8,
                current_evidence_sources=current_sources(required + optional),
            ),
        ),
        (
            "stale_required_evidence",
            dict(
                lines=["header", "cursor", "tail"], region_start_line=1, region_end_line=1,
                max_utf16_units=20,
                required_evidence=[EvidenceCandidate("required-contract", "required", None, True, "0" * 64, "fixture/required-contract", 1)],
                optional_evidence=[item("fresh-optional", "fresh", False)], evidence_budget_utf16_units=20,
                evidence_support=support,
                current_evidence_sources=current_sources([required[0], item("fresh-optional", "fresh", False)]),
            ),
        ),
        (
            "stale_self_consistent_capture",
            dict(
                lines=["header", "cursor", "tail"], region_start_line=1, region_end_line=1,
                max_utf16_units=20,
                required_evidence=[item("old-capture", "old excerpt", True)],
                evidence_budget_utf16_units=20,
                current_evidence_sources=[EvidenceSourceSnapshot("fixture/old-capture", 2, digest("old excerpt"))],
            ),
        ),
    ]


def fixture_payload() -> dict[str, object]:
    return {
        "policy_id": "prm05-selection-dropout-v1",
        "cases": [
            {"name": name, "result": selection_fixture_view(select_campaign_selection(**request))}
            for name, request in fixtures()
        ],
    }


class CampaignSelectionTests(unittest.TestCase):
    def test_unsaved_unicode_crlf_preserves_region_and_scope_header(self):
        result = select_campaign_selection(**fixtures()[0][1])
        self.assertEqual(result.source.region, ("  current <- x",))
        self.assertEqual(result.source.prefix[0], "function_header <- function(x) {")
        self.assertEqual(result.source.suffix[0], "}")
        self.assertTrue(any(span.kind == "scope_prefix" for span in result.source.spans))
        self.assertTrue(any(span.kind == "scope_suffix" for span in result.source.spans))
        self.assertNotIn("\r", "\n".join(result.source.prefix + result.source.region + result.source.suffix))

    def test_large_required_scope_is_complete_and_overflow_is_explicit(self):
        result = select_campaign_selection(**fixtures()[1][1])
        self.assertTrue(result.source.required_overflow)
        self.assertEqual(result.source.region, ("  body_5 <- xxxxxxxxxxxxxxxxxx",))
        self.assertEqual(len(result.source.suffix), 9)
        self.assertIn("}", result.source.suffix)
        self.assertEqual([(item.side, item.start_line, item.end_line) for item in result.source.omissions], [("prefix", 0, 1), ("suffix", 18, 18)])

    def test_missing_provider_balances_nearest_source_sides(self):
        result = select_campaign_selection(**fixtures()[2][1])
        self.assertEqual(result.source.prefix, ("prefix near",))
        self.assertEqual(result.source.suffix, ("suffix near",))
        self.assertEqual(result.evidence.included, ())
        self.assertEqual(result.evidence.omissions, ())

    def test_required_evidence_is_retained_and_has_provenance(self):
        result = select_campaign_selection(**fixtures()[3][1])
        self.assertEqual([candidate.id for candidate in result.evidence.included], ["required-contract"])
        self.assertTrue(result.evidence.required_complete)
        self.assertEqual(result.evidence.required_ids, ("required-contract",))
        self.assertEqual(result.evidence.omissions, ())

    def test_verified_dropout_is_seeded_and_leaves_required_evidence_unchanged(self):
        request = fixtures()[4][1]
        first = select_campaign_selection(**request)
        second = select_campaign_selection(**request)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(first.evidence.support_verified)
        self.assertTrue(first.evidence.dropout_enabled)
        self.assertEqual([candidate.id for candidate in first.evidence.included if candidate.required], ["required-contract"])
        self.assertTrue(any(item.reason == "verified_support_dropout" for item in first.evidence.omissions))

    def test_optional_evidence_cannot_drop_without_verified_support(self):
        result = select_campaign_selection(**fixtures()[5][1])
        self.assertFalse(result.evidence.support_verified)
        self.assertFalse(result.evidence.dropout_enabled)
        self.assertTrue(result.evidence.overflow)
        self.assertEqual(len(result.evidence.included), 6)

    def test_stale_required_evidence_is_explicitly_omitted(self):
        result = select_campaign_selection(**fixtures()[6][1])
        self.assertFalse(result.evidence.required_complete)
        self.assertEqual(result.evidence.stale_ids, ("required-contract",))
        self.assertTrue(any(item.reason == "stale_required_evidence" for item in result.evidence.omissions))
        self.assertEqual([candidate.id for candidate in result.evidence.included], ["fresh-optional"])

    def test_self_consistent_old_excerpt_is_stale_against_current_source_version(self):
        result = select_campaign_selection(**fixtures()[7][1])
        self.assertEqual(result.evidence.stale_ids, ("old-capture",))
        self.assertEqual(result.evidence.integrity_mismatch_ids, ())
        self.assertFalse(result.evidence.required_complete)

    def test_utf16_budget_never_cuts_a_line(self):
        result = select_campaign_selection(
            lines=["😀😀", "cursor", "tail"], region_start_line=1, region_end_line=1,
            max_utf16_units=12, evidence_budget_utf16_units=10,
        )
        self.assertEqual(result.source.prefix, ("😀😀",))
        self.assertEqual(result.source.used_utf16_units, 12)

    def test_lone_cr_is_rejected(self):
        with self.assertRaises(ValueError):
            logical_lines("a\rb")

    def test_single_optional_record_is_not_always_dropped(self):
        required = [item("required", "observed signature", True)]
        optional = [item("optional", "observed documentation", False)]
        dropped = 0
        for seed in range(1000):
            result = select_evidence(
                required=required, optional=optional, max_utf16_units=1000,
                support=support_for(required, seed),
                current_sources=current_sources(required + optional),
            )
            self.assertTrue(result.support_verified)
            self.assertEqual([x.id for x in result.included if x.required], ["required"])
            dropped += result.dropped_by_policy
        self.assertGreater(dropped, 100)
        self.assertLess(dropped, 300)

    def test_fixture_bytes_match_typescript_checker(self):
        encoded = json.dumps(fixture_payload(), ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(digest(encoded), "cae8e84e87a5c3f06479b6d49f156a0616fbff166576c9fe81d9a0ee47a14fa3")


if __name__ == "__main__":
    unittest.main()
