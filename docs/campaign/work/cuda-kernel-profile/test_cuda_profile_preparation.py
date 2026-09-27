#!/usr/bin/env python3
"""CPU-only checks for the redacted profiling packet."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class PreparationPacketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aggregate = json.loads((HERE / "RUN-05-cuda-baseline-aggregate.json").read_text())
        cls.plan = json.loads((HERE / "nsys-profile-plan.json").read_text())

    def test_baseline_counts_and_acceptance(self) -> None:
        contract = self.aggregate["contract"]
        self.assertEqual(contract["fixture_rows"], 4)
        self.assertEqual(contract["requests"], 8)
        self.assertEqual(contract["cold_requests"], 4)
        self.assertEqual(contract["warm_requests"], 4)
        self.assertEqual(contract["accepted_requests"], 8)
        self.assertEqual(self.aggregate["phases"]["cold"]["protocol_status"], {"accepted": 4})
        self.assertEqual(self.aggregate["phases"]["warm"]["protocol_status"], {"accepted": 4})
        self.assertEqual(self.aggregate["phases"]["cold"]["truncated"], {"false": 4})
        self.assertEqual(self.aggregate["phases"]["warm"]["truncated"], {"false": 4})

    def test_aggregate_contains_only_safe_scalar_surface(self) -> None:
        encoded = json.dumps(self.aggregate)
        for forbidden in ("raw_text", "row_id", "returned_token_ids", "prompt_sha256", "prompt_ids_sha256"):
            self.assertNotIn(forbidden, self.aggregate)
            self.assertNotIn(forbidden, encoded)
        self.assertNotIn("/home/", encoded)
        self.assertNotIn("/mnt/", encoded)

    def test_plan_has_paired_arms_and_validated_node_trace(self) -> None:
        arms = self.plan["profile_argv_templates"]
        self.assertEqual([arm["arm"] for arm in arms], ["graph_opt_0", "graph_opt_1"])
        for arm in arms:
            argv = arm["argv"]
            self.assertIn("--cuda-graph-trace=node:host-only", argv)
            self.assertIn("--cuda-event-trace=true", argv)
            self.assertIn("--cuda-trace-all-apis=true", argv)
            self.assertIn("--stats=true", argv)
            self.assertIn("python3", argv)
        self.assertFalse(self.plan["authorization_boundary"]["this_task_executed_gpu_work"])


if __name__ == "__main__":
    unittest.main()
