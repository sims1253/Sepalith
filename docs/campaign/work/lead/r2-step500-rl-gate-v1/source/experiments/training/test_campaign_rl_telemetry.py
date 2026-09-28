"""CPU tests for durable RL generation and gradient telemetry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parent))

import campaign_rl_train  # noqa: E402
from campaign_rl_profile import account_generation_records  # noqa: E402
from campaign_rl_train import (  # noqa: E402
    DurableTelemetrySink,
    FixedIDGRPOTrainerMixin,
    RLGenerationError,
    RLTrainError,
    _emit_generation_telemetry,
    collect_gradient_record,
    gradient_telemetry_callback,
)


class AdapterOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2))
        self.lora_B = torch.nn.Parameter(torch.ones(2))
        self.frozen_weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)


class FailingSink:
    def write(self, _record):
        raise OSError("injected sink failure")


class TelemetryTests(unittest.TestCase):
    def generation_records(self):
        return [
            {
                "prompt_tokens": [0, 200 + index],
                "generated_tokens": [300 + index, 1],
                "terminal_reason": "eos",
                "padded_after_terminal": 0,
                "group_index": 0,
                "group_row_index": index,
            }
            for index in range(4)
        ]

    def test_fsynced_jsonl_generation_events_match_g4_rows(self):
        records = self.generation_records()
        accounting = account_generation_records(records)
        trainer = SimpleNamespace(
            state=SimpleNamespace(global_step=7),
            _step=12,
            num_generations=4,
            _campaign_source_draw_schedule_sha256="a" * 64,
        )
        groups = [{"group_index": 0, "start": 0, "end": 4, "candidate_count": 4}]
        with tempfile.TemporaryDirectory() as directory:
            sink = DurableTelemetrySink(Path(directory) / "generation-records.jsonl")
            trainer._campaign_generation_record_sink = sink
            with mock.patch.object(campaign_rl_train.os, "fsync", wraps=os.fsync) as fsync:
                _emit_generation_telemetry(trainer, records, accounting, groups, 0.125)
                self.assertEqual(fsync.call_count, len(records))
            lines = (Path(directory) / "generation-records.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 4)
        events = [json.loads(line) for line in lines]
        self.assertEqual([event["group_geometry"]["candidate_count"] for event in events], [4] * 4)
        self.assertEqual([event["group_geometry"]["group_row_index"] for event in events], list(range(4)))
        self.assertEqual([event["generated_ids"] for event in events], [[300 + i, 1] for i in range(4)])
        self.assertEqual([event["update"]["global_step_before_update"] for event in events], [7] * 4)
        self.assertEqual([event["trl_microstep"] for event in events], [12] * 4)
        self.assertEqual([event["source_schedule_sha256"] for event in events], ["a" * 64] * 4)
        self.assertEqual([event["accounting"]["status"] for event in events], ["accounted"] * 4)
        self.assertTrue(all("prompt_text" not in event for event in events))

    def test_generation_sink_failure_propagates(self):
        trainer = SimpleNamespace(
            state=SimpleNamespace(global_step=0), _step=0, num_generations=4,
            _campaign_source_draw_schedule_sha256=None,
            _campaign_generation_record_sink=FailingSink(),
        )
        records = self.generation_records()
        with self.assertRaisesRegex(RLGenerationError, "telemetry sink failed"):
            _emit_generation_telemetry(
                trainer, records, account_generation_records(records),
                [{"group_index": 0, "start": 0, "end": 4, "candidate_count": 4}], 0.0,
            )

    def test_finite_and_zero_lora_gradients_are_logged_without_zero_rejection(self):
        model = AdapterOnly()
        model.lora_A.grad = torch.tensor([1.0, -2.0])
        model.lora_B.grad = torch.zeros(2)
        record = collect_gradient_record(model, step=3)
        self.assertTrue(record["finite"])
        self.assertEqual(record["trainable_tensor_count"], 2)
        self.assertEqual(record["grad_present_count"], 2)
        self.assertEqual(record["nonzero_tensor_count"], 1)
        self.assertAlmostEqual(record["norm"], 5**0.5)
        with tempfile.TemporaryDirectory() as directory:
            sink = DurableTelemetrySink(Path(directory) / "gradient-records.jsonl")
            callback = gradient_telemetry_callback(sink)
            callback.on_pre_optimizer_step(
                None, SimpleNamespace(global_step=3), SimpleNamespace(), model=model,
            )
            saved = json.loads((Path(directory) / "gradient-records.jsonl").read_text())
        self.assertTrue(saved["finite"])
        self.assertEqual(saved["nonzero_tensor_count"], 1)

    def test_nonfinite_lora_gradient_is_durable_then_rejected(self):
        model = AdapterOnly()
        model.lora_A.grad = torch.tensor([float("nan"), 0.0])
        model.lora_B.grad = torch.ones(2)
        with tempfile.TemporaryDirectory() as directory:
            sink = DurableTelemetrySink(Path(directory) / "gradient-records.jsonl")
            callback = gradient_telemetry_callback(sink)
            with self.assertRaisesRegex(RLTrainError, "nonfinite LoRA gradient"):
                callback.on_pre_optimizer_step(
                    None, SimpleNamespace(global_step=11), SimpleNamespace(), model=model,
                )
            saved = json.loads((Path(directory) / "gradient-records.jsonl").read_text())
        self.assertFalse(saved["finite"])
        self.assertTrue(saved["nonfinite"])
        self.assertEqual(saved["step"], 11)
        self.assertIsNone(saved["norm"])

    def test_full_weight_trainable_tensor_is_rejected(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RLTrainError, "non-LoRA trainable"):
            collect_gradient_record(model, step=1)

    def test_fixture_runtime_sink_defaults_to_none(self):
        obj = FixedIDGRPOTrainerMixin.__new__(FixedIDGRPOTrainerMixin)
        obj.configure_campaign_runtime(generation_guard_factory=None)
        self.assertIsNone(obj._campaign_generation_record_sink)


if __name__ == "__main__":
    unittest.main()
