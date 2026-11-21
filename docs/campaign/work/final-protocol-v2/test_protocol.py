"""Synthetic-only tests of the sealed-final v2 entry point."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import protocol as v2
import runtime
from admission_guard import AdmissionMetadataError, AdmissionOutputError
from sepalith.campaign_protocol import Cursor, Position, PromptContext, ReplacementRange, serialize_target

SOURCE = b"x <- 1\n"
TARGET = ["x <- 2"]
HARNESS_SHA = "3" * 64


class TextTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False, split_special_tokens: bool = True) -> list[int]:
        return [ord(char) + 512 for char in text]

    def decode(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return "".join(chr(token - 512) for token in ids)


class SyntheticRuntime:
    def __init__(self, plan: v2.Json, *, replay: bool) -> None:
        self.tokenizer = TextTokenizer()

    @contextmanager
    def generator(self, label: str) -> Iterator[v2.Generate]:
        def generate(prompt: list[int]) -> list[int]:
            assert prompt[0] == 0
            if label == "arm-a-803":
                return [512] * v2.CAP
            return [*self.tokenizer.encode(serialize_target("replace", TARGET)), 1]

        yield generate


def save(path: Path, value: v2.Json) -> v2.Json:
    payload = v2.canonical_json(value).encode()
    path.write_bytes(payload)
    return {"path": str(path), "sha256": v2.sha256_bytes(payload)}


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.state = self.root / "state"
        self.state.mkdir()
        self.rows: list[v2.Json] = []
        self.payloads: v2.Json = {}
        for i in range(20):
            self.add_case(i)
        self.memberships = {
            stage: {
                "row_ids": [], "package_ids": [f"unrelated-{stage}"], "group_ids": [],
                "split_identities": [], "function_ids": [f"outside:{stage}"]
            }
            for stage in v2.STAGES
        }
        harness = patch.object(v2, "harness_hash", return_value=HARNESS_SHA)
        harness.start()
        self.addCleanup(harness.stop)
        fake_runtime = patch.object(runtime, "FrozenRuntime", SyntheticRuntime)
        fake_runtime.start()
        self.addCleanup(fake_runtime.stop)

    def add_case(self, index: int) -> None:
        source = self.sources / f"test-{index}.R"
        source.write_bytes(SOURCE)
        context = PromptContext(
            path="R/test.R", prefix=(), selected_references=(), history=(), diagnostics=(), retrieval=(),
            scope_mode="off", scope_lines=(), suffix_lines=(), region_old=("x <- 1",),
            cursor=Cursor(0, 0, 0), replacement_range=ReplacementRange(
                "file:///R/test.R", 1, v2.sha256_bytes(SOURCE), Position(0, 0), Position(0, 6),
            ),
        ).to_dict()
        target = {"operation": "replace", "region_new": TARGET.copy()}
        row_id = f"row-{index}"
        self.payloads[row_id] = {"context": context, "target": target}
        self.rows.append({
            "row_id": row_id, "package_id": f"package-{index}", "repository_id": "synthetic",
            "group_id": f"group-{index}", "split_identity": f"synthetic:{row_id}", "family": "finish_block",
            "function_ids": [f"fn-{index}"], "source_path": str(source), "source_sha256": v2.sha256_bytes(SOURCE),
            "context_sha256": v2.sha256_json(context), "target_sha256": v2.sha256_json(target),
        })

    def freeze(self, **overrides: object) -> argparse.Namespace:
        cases = save(self.root / "cases.json", {"split": "final_locked_v1", "cases": self.rows})
        semantic = save(self.root / "semantic.json", {
            "schema": "dat08.semantic-source-case-receipt.v1", "status": "source_cases_verified",
            "split": "final_locked_v1", "source_lock_sha256": cases["sha256"],
            "constructor_sha256": "7" * 64, "content_access": "post_freeze_source_bytes",
            "verified_case_ids": [row["row_id"] for row in self.rows],
            "case_bindings": {
                row["row_id"]: {key: row[key] for key in (
                    "split_identity", "source_sha256", "context_sha256", "target_sha256",
                )}
                for row in self.rows
            },
            **getattr(self, "semantic_extra", {}),
        })
        save(self.root / "payloads.json", self.payloads)
        models = {label: {} for label in v2.MODELS}
        plan = save(self.root / "plan.json", {
            "schema": v2.SCHEMA + ".plan", "models": models, "source_roots": [str(self.sources)],
            "cases": cases, "semantic_receipt": semantic, "case_count": len(self.rows),
            "state_dir": str(self.state), "payloads_path": str(self.root / "payloads.json"),
            "memberships": save(self.root / "memberships.json", self.memberships),
        })
        frozen = save(self.root / "freeze.json", {
            "schema": v2.SCHEMA + ".freeze", "status": "frozen", "plan_sha256": plan["sha256"],
            "weights_frozen": True, "harness_frozen": True, "final_access_unlocked": True,
            "weights_sha256": v2.sha256_json(models), "harness_sha256": HARNESS_SHA,
            "weights_frozen_at": "2026-09-13T00:00:00Z", "harness_frozen_at": "2026-09-13T00:00:00Z",
            **overrides,
        })
        return argparse.Namespace(
            plan=Path(plan["path"]), freeze_receipt=Path(frozen["path"]), freeze_sha256=frozen["sha256"],
            resume=None, resume_sha256=None,
        )

    def test_all_hashes_checked_case_excluded_at_exact_ceiling(self) -> None:
        Path(self.rows[0]["source_path"]).write_bytes(b"changed")
        self.payloads["row-0"]["context"]["prefix"] = ["changed"]
        self.payloads["row-0"]["target"]["region_new"] = ["changed"]
        report = v2.run(self.freeze())
        self.assertEqual(report["audit"]["hash_mismatches"], {"cases": 1, "issuance": 0, "context": 1, "target": 1, "source": 1})
        self.assertEqual(report["clean"]["cases"], 19)
        comparison = report["clean"]["comparisons"]["c19-100-bf16_minus_arm-a-803"]
        self.assertEqual(comparison, {"delta": 1.0, "ci95": [1.0, 1.0]})
        output = (self.state / "result.json").read_text()
        for value in ("row-0", "package-0", "x <-", "generated_ids"):
            self.assertNotIn(value, output)

    def test_issuance_mismatch_stays_excluded_and_counts_once(self) -> None:
        self.semantic_extra = {"issuance_mismatches": ["row-0"]}  # matches now, mismatched at issuance
        report = v2.run(self.freeze())
        self.assertEqual(report["audit"]["hash_mismatches"]["cases"], 1)
        self.assertEqual(report["audit"]["hash_mismatches"]["issuance"], 1)
        self.assertEqual(report["clean"]["cases"], 19)

    def test_package_clusters_keep_frozen_ids(self) -> None:
        self.rows[0]["package_id"] = "foo.bar"
        self.rows[1]["package_id"] = "foobar"
        report = v2.run(self.freeze())
        self.assertEqual(report["clean"]["packages"], 20)

    def test_hash_mismatch_over_five_percent_has_no_scoring(self) -> None:
        self.payloads["row-0"]["target"]["region_new"] = ["changed"]
        self.payloads["row-1"]["context"]["prefix"] = ["changed"]
        with patch.object(v2, "score") as score:
            report = v2.run(self.freeze())
            score.assert_not_called()
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["reason"], "mismatch_ceiling_exceeded")
        self.assertEqual(report["audit"]["hash_mismatches"]["cases"], 2)

    def test_contamination_by_stage_and_group_at_exact_ceiling(self) -> None:
        self.memberships["cpt"]["package_ids"].append("package-0")
        self.memberships["c14"]["row_ids"].append("row-1")
        self.memberships["synth"]["group_ids"].append("group-2")
        self.memberships["exec_train"]["package_ids"].append("exec:package-3")
        self.memberships["exec_validation"]["function_ids"].append("fn-1")
        self.memberships["exec_holdout"]["function_ids"].append("fn-3")
        self.memberships["train"]["split_identities"].append("synthetic:row-0")
        self.memberships["dev"]["function_ids"].append("fn-2")
        self.add_case(20)
        self.rows[-1]["group_id"] = "group-3"
        report = v2.run(self.freeze())
        self.assertEqual(report["audit"]["overlap_groups"], 4)
        self.assertEqual(report["audit"]["overlap_groups_by_stage"], dict.fromkeys(v2.STAGES, 1))
        self.assertEqual(report["clean"]["cases"], 16)
        self.assertEqual(report["overlapping"]["cases"], 5)

    def test_package_forms_normalize_across_inventories(self) -> None:
        self.memberships["cpt"]["package_ids"].append("pkg:Package_4")
        report = v2.run(self.freeze())
        self.assertEqual(report["audit"]["overlap_groups"], 1)

    def test_unfrozen_receipt_rejected_before_plan_is_read(self) -> None:
        args = self.freeze(status="draft")
        args.plan = self.root / "missing-plan.json"
        with self.assertRaisesRegex(v2.ProtocolError, "freeze_not_authorized"):
            v2.run(args)

    def test_overlap_abort_does_not_open_payloads_or_sources(self) -> None:
        self.memberships["train"]["row_ids"].extend(f"row-{i}" for i in range(5))
        args = self.freeze()
        (self.root / "payloads.json").unlink()
        for source in self.sources.iterdir():
            source.unlink()
        report = v2.run(args)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["reason"], "overlap_ceiling_exceeded")
        self.assertEqual(report["audit"]["overlap_groups"], 5)

    def test_write_once_outputs_and_completed_run(self) -> None:
        args = self.freeze()
        v2.run(args)
        result = self.state / "result.json"
        original = result.read_bytes()
        with self.assertRaisesRegex(v2.ProtocolError, "run_already_closed"):
            v2.run(args)
        with self.assertRaisesRegex(AdmissionOutputError, "output_exists"):
            v2.write_once(result, {"replacement": True})
        self.assertEqual(result.read_bytes(), original)

    def test_missing_or_invalid_freeze_blocks_access(self) -> None:
        args = self.freeze(final_access_unlocked=False)
        (self.root / "cases.json").unlink()
        with self.assertRaisesRegex(v2.ProtocolError, "freeze_not_authorized"):
            v2.run(args)
        args.freeze_receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            v2.run(args)
        self.assertEqual(list(self.state.iterdir()), [])

    def test_resume_uses_only_matching_completed_artifacts(self) -> None:
        args = self.freeze()
        with patch.object(v2, "score", side_effect=OSError):
            with self.assertRaises(OSError):
                v2.run(args)
        started = (self.state / "started.json").read_bytes()
        plan_sha = json.loads(started)["plan_sha256"]
        with self.assertRaisesRegex(v2.ProtocolError, "run_requires_empty_state_directory"):
            v2.run(args)
        decoded = serialize_target("replace", TARGET)
        output = TextTokenizer().encode(decoded) + [1]
        completions = {
            label: save(self.root / f"{label}-completions.json", {
                "schema": v2.SCHEMA + ".completions", "model": label, "plan_sha256": plan_sha,
                "completions": {row["row_id"]: output for row in self.rows},
            })
            for label in v2.MODELS
        }
        resume = save(self.root / "resume.json", {
            "schema": v2.SCHEMA + ".resume", "authorization": "user-approved", "plan_sha256": plan_sha,
            "started_sha256": v2.sha256_bytes(started), "completions": completions,
        })
        args.resume, args.resume_sha256 = Path(resume["path"]), resume["sha256"]
        with patch.object(SyntheticRuntime, "generator", side_effect=AssertionError("replay must not generate")):
            report = v2.run(args)
        self.assertEqual(report["clean"]["models"]["arm-a-803"]["rate"], 1.0)
        self.assertEqual((self.state / "started.json").read_bytes(), started)

    def test_resume_rejects_changed_artifact(self) -> None:
        args = self.freeze()
        with patch.object(v2, "score", side_effect=OSError):
            with self.assertRaises(OSError):
                v2.run(args)
        started = (self.state / "started.json").read_bytes()
        plan_sha = json.loads(started)["plan_sha256"]
        path = self.root / "completions.json"
        pin = save(path, {"original": True})
        save(path, {"forged": True})
        resume = save(self.root / "resume.json", {
            "schema": v2.SCHEMA + ".resume", "authorization": "user-approved", "plan_sha256": plan_sha,
            "started_sha256": v2.sha256_bytes(started),
            "completions": dict.fromkeys(v2.MODELS, pin),
        })
        args.resume, args.resume_sha256 = Path(resume["path"]), resume["sha256"]
        with self.assertRaisesRegex(v2.ProtocolError, "input_hash_mismatch"):
            v2.run(args)
        self.assertFalse((self.state / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
