from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("queue_v2", HERE / "source/run_streaming_semantic_queue.py")
assert spec is not None and spec.loader is not None
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def lines(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


class TestQueueV2(unittest.TestCase):
    def fixture(self) -> tuple[Path, dict, dict, Path, dict]:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        replay = root / "replay"
        base = root / "base"
        q.BASE = base

        index_path = replay / "index/shard-0005.jsonl"
        lines(index_path, [{"selected": {"row_id": "a"}}, {"selected": {"row_id": "b"}}])
        token_pin = {"shard": 5, "sha256": "t" * 64, "manifest_sha256": "m" * 64}
        index = {
            "schema": "sepalith.dat10.sourcewalk_raw_index.v3",
            "status": "complete",
            "requested_shards": [5],
            "input_inventory": {"shard_pins": [token_pin]},
            "index_files": [{
                "shard": 5, "sha256": q.sha(index_path), "bytes": index_path.stat().st_size,
                "rows": 2, "path": str(index_path),
            }],
        }
        index_manifest_path = replay / "index/manifest.json"
        write(index_manifest_path, index)
        index_sha = q.sha(index_manifest_path)

        packet = base / "shard-0005/structured-materialization-v1/candidate-packets.jsonl"
        packet_rows = [{"row_ref": {"row_id": "a"}}, {"row_ref": {"row_id": "b"}}]
        lines(packet, packet_rows)
        packet_manifest = packet.parent / "manifest.json"
        write(packet_manifest, {"outputs": {"candidate_packets": {
            "path": str(packet), "sha256": q.sha(packet), "bytes": packet.stat().st_size, "rows": 2,
        }}})

        ledger = replay / "shards/shard-0005/ledger.jsonl"
        provenance_rows = [
            {"row_id": "a", "shard": 5, "family": "roxygen_drafting", "status": q.QUEUED},
            {"row_id": "b", "shard": 5, "family": "no_op", "status": "provenance_supported_candidate_root_review_required"},
        ]
        lines(ledger, provenance_rows)
        binding = {
            "driver_sha256": q.DRIVER_SHA, "index_manifest_sha256": index_sha,
            "index_shard_sha256": q.sha(index_path), "token_rows_sha256": token_pin["sha256"],
            "token_manifest_sha256": token_pin["manifest_sha256"], "candidate_packets_sha256": q.sha(packet),
            "candidate_packet_manifest_sha256": q.sha(packet_manifest), "global_sha256": q.GLOBAL_SHA,
            "cpt_sha256": q.CPT_SHA, "hold_ledger_sha256": q.HOLD_SHA,
            "strict_validator_sha256": q.STRICT_SHA, "license_parser_sha256": q.LICENSE_SHA,
        }
        receipt = {
            "schema": "sepalith.dat10.sourcewalk_provenance_shard.v3", "status": "complete", "shard": 5,
            "binding": binding, "rows": 2,
            "outputs": [{"path": str(ledger), "rows": 2, "bytes": ledger.stat().st_size, "sha256": q.sha(ledger)}],
        }
        receipt_path = ledger.parent / "receipt.json"
        write(receipt_path, receipt)
        source = q.validate_receipt(replay, 5, index, {5: index["index_files"][0]}, index_sha)
        return replay, index, source, receipt_path, receipt

    def test_source_binding_contains_content_digests(self) -> None:
        _, _, source, _, _ = self.fixture()
        self.assertEqual(source["queued_ids"], ["a"])
        self.assertEqual(source["provenance_ledger"]["rows"], 2)
        self.assertEqual(source["provenance_ledger"]["sha256"], q.sha(Path(source["provenance_ledger"]["path"])))
        self.assertEqual(source["queued_ids_sha256"], q.digest_ids(["a"]))
        self.assertTrue(source["binding_sha256"])

    def test_same_ids_with_tampered_provenance_bytes_are_rejected(self) -> None:
        replay, index, _, receipt_path, receipt = self.fixture()
        ledger = replay / "shards/shard-0005/ledger.jsonl"
        rows = list(q.lines(ledger))
        rows[0]["status"] = "tampered_but_same_row_id"
        lines(ledger, rows)
        with self.assertRaisesRegex(q.QueueError, "provenance ledger changed"):
            # The row-ID set is unchanged; the receipt content hash rejects it.
            q.validate_receipt(replay, 5, index, {5: index["index_files"][0]}, q.sha(replay / "index/manifest.json"))
        self.assertEqual(json.loads(receipt_path.read_text()), receipt)

    def test_reuse_requires_source_and_output_content_binding(self) -> None:
        _, _, source, _, _ = self.fixture()
        code = {"analyzer_sha256": "a", "scope_helper_sha256": "b", "namespace_helper_sha256": "c"}
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "shard-0005"
            ledger = target / "semantic-ledger.jsonl"
            semantic_rows = [{"row_id": "a", "decision": "supported"}]
            lines(ledger, semantic_rows)
            output = q.semantic_output(ledger, semantic_rows)
            manifest = {
                "schema": "sepalith.dat10.sourcewalk_roxy_semantic.v6", "status": "complete_review_only",
                "streaming_binding": source, "source_provenance_binding_sha256": source["binding_sha256"],
                "queued_ids_sha256": source["queued_ids_sha256"], "code": code, "exact_id_closure": True,
                "rows": 1, "output": output, "output_binding": output,
            }
            write(target / "manifest.json", manifest)
            self.assertTrue(q.reusable(target, source, code))
            lines(ledger, [{"row_id": "a", "decision": "same_id_different_bytes"}])
            self.assertFalse(q.reusable(target, source, code))

    def test_aggregate_rejects_id_only_child_result(self) -> None:
        _, _, source, _, _ = self.fixture()
        result = {
            "shard": 5, "rows": 1, "queued_ids": ["a"], "queued_ids_sha256": source["queued_ids_sha256"],
            "source_binding_sha256": source["binding_sha256"],
            "provenance_ledger_sha256": source["provenance_ledger"]["sha256"],
            "semantic_output_rows": 1, "semantic_output_sha256": "x" * 64,
            # Deliberately absent output row-ID digest: same IDs alone are insufficient.
        }
        with self.assertRaisesRegex(q.QueueError, "semantic output ID digest differs"):
            q.aggregate_status([result], [source], None)

    def test_partial_never_claims_global_terminal(self) -> None:
        _, _, source, _, _ = self.fixture()
        result = {
            "shard": 5, "rows": 1, "queued_ids": ["a"], "queued_ids_sha256": source["queued_ids_sha256"],
            "source_binding_sha256": source["binding_sha256"],
            "provenance_ledger_sha256": source["provenance_ledger"]["sha256"],
            "semantic_output_rows": 1, "semantic_output_sha256": "x" * 64,
            "semantic_output_row_ids_sha256": source["queued_ids_sha256"],
        }
        status, closure = q.aggregate_status([result], [source], None)
        self.assertEqual(status, "partial_review_only")
        self.assertFalse(any(closure.values()))

    def test_terminal_requires_validated_receipt_hash_and_rows(self) -> None:
        replay, _, source, receipt_path, _ = self.fixture()
        terminal = replay / "manifest.json"
        write(terminal, {"status": "complete_review_only_no_admission", "shard_receipts": [
            {"shard": 5, "receipt_sha256": q.sha(receipt_path), "rows": source["provenance_ledger"]["rows"]},
        ]})
        result = {
            "shard": 5, "rows": 1, "queued_ids": ["a"], "queued_ids_sha256": source["queued_ids_sha256"],
            "source_binding_sha256": source["binding_sha256"],
            "provenance_ledger_sha256": source["provenance_ledger"]["sha256"],
            "semantic_output_rows": 1, "semantic_output_sha256": "x" * 64,
            "semantic_output_row_ids_sha256": source["queued_ids_sha256"],
        }
        status, closure = q.aggregate_status([result], [source], terminal)
        self.assertEqual(status, "complete_review_only")
        self.assertTrue(all(closure.values()))
        bad_terminal = copy.deepcopy(json.loads(terminal.read_text()))
        bad_terminal["shard_receipts"][0]["rows"] = 999
        write(terminal, bad_terminal)
        with self.assertRaisesRegex(q.QueueError, "row count differs"):
            q.aggregate_status([result], [source], terminal)

    def test_shard_parser_rejects_duplicates_and_unsorted_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            replay = Path(td)
            self.assertEqual(q.parse_shards("1,3", replay), [1, 3])
            for bad in ("3,1", "1,1", ""):
                with self.assertRaises(q.QueueError):
                    q.parse_shards(bad, replay)


if __name__ == "__main__":
    unittest.main()
