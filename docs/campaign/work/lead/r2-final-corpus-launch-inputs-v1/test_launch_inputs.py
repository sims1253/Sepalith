"""Payload-free tests for terminal gating and all-row schedule arithmetic."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PACKET = Path(__file__).resolve().parent
TRAINER_SOURCE = PACKET.parents[4] / "docs/campaign/work/lead/r2-full-weight-cpt-full-corpus-trainer-v3/source/experiments/training"
import sys
sys.path.insert(0, str(TRAINER_SOURCE))
from cpt_streaming_cache import StreamingCptDataset, build, sha256  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load_module("pre04_gate_test", PACKET / "verify_terminal_union.py")
SCHEDULE = load_module("pre04_schedule_test", PACKET / "make_one_pass_schedule.py")


def row(index: int) -> dict:
    source = hashlib.sha256(f"doc-{index}".encode()).hexdigest()
    token = 100 + index
    return {
        "schema": 1,
        "row_id": f"{source}:0",
        "document_id": source,
        "package": f"package-{index}",
        "group_id": f"group-{index}",
        "cpt_partition": "cpt_train",
        "source_path": f"TRAIN/package-{index}/R/doc-{index}.R",
        "source_sha256": source,
        "chunk_index": 0,
        "input_ids": [0, token, 1],
        "labels": [-100, token, 1],
        "attention_mask": [1, 1, 1],
        "source_token_start": 0,
        "source_token_end": 1,
        "overlap_context_tokens": 0,
        "is_document_end": True,
        "supervised_tokens": 2,
        "document_token_count": 1,
    }


class LaunchInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pre04-launch-inputs-", dir="/mnt/e/sepalith/campaign-20260915/data-work")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schedule_has_unique_first_pass_and_minimal_tail(self):
        rows_path = self.root / "cpt_train_ctx16384.jsonl"
        rows = [row(i) for i in range(15)]
        rows_path.write_text("".join(json.dumps(value, separators=(",", ":")) + "\n" for value in rows))
        result = self.root / "result.json"
        result.write_text(json.dumps({
            "schema": "sepalith.cpt.lossless-rechunk-result.v1",
            "status": "complete",
            "totals": {"documents": 15, "payload_tokens": 15},
            "outputs": {"16384": {"rows": 15, "payload_tokens": 15, "terminal_eos": 15}},
            "artifacts": {"cpt_train_ctx16384.jsonl": {"bytes": rows_path.stat().st_size, "sha256": sha256(rows_path)}},
        }))
        schedule_path = self.root / "draw-schedule.json"
        report = SCHEDULE.make_schedule(rows_path, result, schedule_path, seed=3407, split_id="synthetic")
        value = json.loads(schedule_path.read_text())
        self.assertEqual(value["method"], "one_pass_plus_named_replay_v1")
        self.assertEqual(value["coverage"]["unique_rows"], 15)
        self.assertEqual(value["replay_count"], 1)
        self.assertEqual(value["max_steps"], 1)
        self.assertEqual(len(value["row_ids"]), 16)
        self.assertEqual(len(set(value["row_ids"][:15])), 15)
        self.assertEqual(value["replay_row_ids"], value["row_ids"][-1:])
        self.assertEqual(value["stage_transition"]["initial_stage_cursor"], 0)
        self.assertEqual(value["stage_transition"]["global_optimizer_step_offset"], 66)
        cli_schedule = self.root / "draw-schedule-cli.json"
        cli = subprocess.run([
            sys.executable, str(PACKET / "make_one_pass_schedule.py"),
            "--rows", str(rows_path), "--rechunk-result", str(result),
            "--output", str(cli_schedule), "--seed", "3407", "--split-id", "synthetic-cli",
        ], check=True, capture_output=True, text=True)
        self.assertNotIn("row_ids", cli.stdout)
        self.assertIn('"split_id": "synthetic-cli"', cli.stdout)
        cache = self.root / "cache"
        manifest = build(rows_path, schedule_path, cache, 16384, sha256(rows_path), sha256(schedule_path))
        self.assertEqual(manifest["counts"]["rows"], 15)
        self.assertEqual(manifest["counts"]["draws"], 16)
        self.assertEqual(manifest["counts"]["named_replays"], 1)
        dataset = StreamingCptDataset(cache, sha256(cache / "manifest.json"), initial_cursor=0)
        self.assertEqual(dataset[0]["attention_mask"], [1, 1, 1])
        dataset.close()

    def test_terminal_gate_reads_only_small_metadata(self):
        union = self.root / "union"
        union.mkdir()
        counts = {"documents": 15, "rows": 15, "payload_tokens": 15}
        input_manifest = {
            "schema": "sepalith.cpt.lossless-rechunk-input.v1",
            "status": "final_candidate_pending_root_admission",
            "context_sizes": [16384],
            "training_admission": False,
            "truncation": False,
            "tokenizer": {"retokenized": False},
            "inputs": [{"path": str(union / "cpt_train.jsonl"), "bytes": 1, "sha256": "a" * 64}],
            "expected_totals": counts,
        }
        source_manifest_path = union / "source-manifest.jsonl"
        source_manifest_path.write_text("metadata-only\n")
        (union / "cpt_train.jsonl").write_bytes(b"x")
        (union / "input-manifest.json").write_text(json.dumps(input_manifest))
        manifest = {
            "schema": "sepalith.dat10.cpt_final_union_candidate.v1",
            "status": "complete_candidate_pending_root_global_dedup_review_and_training_admission",
            "mode": "final",
            "training_admission": False,
            "truncation": False,
            "retokenized": False,
            "heldout_content_read": False,
            "main_terminal_gate": {"progress": {"groups_committed": 8092, "groups_remaining": 0, "first_uncommitted_seeded_index": 8867, "status": "complete"}},
            "alias_terminal_capture_gate": {"progress": {"captured_main_groups": 8092, "admitted": False}},
            "artifacts": {"cpt_train.jsonl": {"bytes": 1, "sha256": "a" * 64}},
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            "input_manifest": str(union / "input-manifest.json"),
            "input_manifest_sha256": __import__("hashlib").sha256((union / "input-manifest.json").read_bytes()).hexdigest(),
            "counts": counts,
        }
        (union / "manifest.json").write_text(json.dumps(manifest))
        # The bulk cpt_train.jsonl is intentionally absent: this proves the
        # gate can validate terminal metadata without opening payload bytes.
        result = GATE.verify(union)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["bulk_payload_opened"])
        manifest["main_terminal_gate"]["progress"]["groups_remaining"] = 1
        (union / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "main_groups_remaining"):
            GATE.verify(union)


if __name__ == "__main__":
    unittest.main(verbosity=2)
