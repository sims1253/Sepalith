#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("repair_geometry_tested", HERE / "repair_geometry.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class GeometryTests(unittest.TestCase):
    def test_mixed_eol_boundary_map_and_utf8(self):
        raw = "α\r\n#' x\r\nf <- function() {}\n".encode()
        normalized, mapping = module.normalized_with_boundaries(raw)
        self.assertEqual(normalized.decode(), "α\n#' x\nf <- function() {}\n")
        target = b"#' x\n"
        start = normalized.index(target)
        self.assertEqual(raw[mapping[start] : mapping[start + len(target)]].replace(b"\r\n", b"\n"), target)

    def test_lone_cr_rejected(self):
        with self.assertRaisesRegex(Exception, "lone_cr"):
            module.normalized_with_boundaries(b"a\rb")

    def test_occurrence_is_not_guessed(self):
        self.assertEqual(module.exact_occurrences(b"x\nx\n", b"x\n"), [0, 2])

    def test_actual_three_rows_close_and_predictions_are_target_free(self):
        sem = Path("/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1/shard-0011/semantic-ledger.jsonl")
        prov = Path("/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards/shard-0011/ledger.jsonl")
        cand = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0011/structured-materialization-v1/candidate-packets.jsonl")
        holds = Path("/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1/shard-0011/preparation-holds.jsonl")
        S, P, C, H = map(module.read_selected, (sem, prov, cand, holds))
        for rid in module.REQUIRED_IDS:
            prediction, sidecar = module.checked_row(S[rid], P[rid], C[rid], H[rid])
            self.assertNotIn("\n".join(sidecar["target_lines"]), prediction["preedit_text"])
            self.assertTrue(sidecar["source"]["full_source_reapplication_exact"])
            self.assertTrue(sidecar["source"]["non_target_bytes_preserved"])
            self.assertGreater(sidecar["source"]["source_crlf_count"], 0)
            self.assertGreater(sidecar["source"]["source_lf_count"], sidecar["source"]["source_crlf_count"])

    def test_wrong_span_rejected(self):
        paths = [
            Path("/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1/shard-0011/semantic-ledger.jsonl"),
            Path("/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards/shard-0011/ledger.jsonl"),
            Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0011/structured-materialization-v1/candidate-packets.jsonl"),
            Path("/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1/shard-0011/preparation-holds.jsonl"),
        ]
        S, P, C, H = map(module.read_selected, paths)
        rid = module.REQUIRED_IDS[1]
        mutated = json.loads(json.dumps(S[rid]))
        mutated["context_closure"]["target_definition_span"] = [169, 172]
        mutated["scope"]["target_definition_span"] = [169, 172]
        with self.assertRaisesRegex(Exception, "immediately_attached"):
            module.checked_row(mutated, P[rid], C[rid], H[rid])


if __name__ == "__main__":
    unittest.main()
