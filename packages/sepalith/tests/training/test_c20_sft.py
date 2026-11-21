"""CPU coverage for the frozen C20 dose, clustered gates, and replay masking."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sepalith.training.sft import c20_data, c20_recipe, c20_run
from sepalith.training.sft.campaign_edit_data import validate_row

HAS_TORCH = importlib.util.find_spec("torch") is not None


class C20Tests(unittest.TestCase):
    def test_mix_draws_are_deterministic_with_exact_composition(self):
        with patch.object(c20_data, "UPDATES", 2):
            groups = {group: [{"id": f"{group}-{i}", "family": group, "input_ids": [0, 10, 1],
                               "target_start": 1} for i in range(2 * dose)]
                      for group, dose in c20_data.DOSE.items()}
            rows, draws = c20_data.draw_batches(groups)
            self.assertEqual((rows, draws), c20_data.draw_batches(groups))
            self.assertEqual(len(rows), 32)
            for update in (1, 2):
                ids = [r["row_id"] for r in draws if r["update"] == update]
                self.assertEqual(len(ids), 16)
                for group, dose in c20_data.DOSE.items():
                    self.assertEqual(sum(i.startswith(group + "-") for i in ids), dose)

    def test_expansion_requires_existing_function_and_package(self):
        split = ({"f"}, {"known"})
        self.assertTrue(c20_data.belongs({"function_id": "f", "package": "KNOWN"}, split))
        self.assertFalse(c20_data.belongs({"function_id": "unknown", "package": "known"}, split))
        self.assertFalse(c20_data.belongs({"function_id": "f", "package": "unknown"}, split))

    def test_clustered_gate_requires_a_positive_lower_bound(self):
        start, candidate = self.pair()
        self.assertTrue(c20_run.promote(start, candidate, 2, 0.03)["accepted"])
        for row in candidate:
            if row["family"] == "finish_block" and row["function_id"] != "f-0":
                row["all_pass"] = False
        decision = c20_run.promote(start, candidate, 0, 0)
        self.assertGreaterEqual(decision["finish_difference"]["mean"], 0.02)
        self.assertEqual(decision["finish_difference"]["ci95"][0], 0)
        self.assertFalse(decision["accepted"])

    def test_guardrails_and_complete_pairing(self):
        start, candidate = self.pair()
        self.assertFalse(c20_run.promote(start, candidate, 3, 0.03)["accepted"])
        self.assertFalse(c20_run.promote(start, candidate, 2, 0.031)["accepted"])
        candidate[-1]["all_pass"] = False  # Exactly 1 percentage point.
        self.assertTrue(c20_run.promote(start, candidate, 2, 0.03)["accepted"])
        candidate[-2]["all_pass"] = False
        self.assertFalse(c20_run.promote(start, candidate, 2, 0.03)["accepted"])
        with self.assertRaisesRegex(ValueError, "pairs differ"):
            c20_run.promote(start, candidate[:-1], 0, 0)

    @staticmethod
    def pair() -> tuple[list[dict], list[dict]]:
        start = [{"task_id": f"target-{i}-{j}", "function_id": f"f-{i}", "family": "finish_block",
                  "target_lines": 9, "all_pass": False} for i in range(20) for j in range(2)]
        start += [{"task_id": f"rename-{i}", "function_id": f"r-{i}", "family": "rename_propagation",
                   "target_lines": 1, "all_pass": True} for i in range(100)]
        candidate = [dict(r, all_pass=True) for r in start]
        return start, candidate

    def test_failed_pilot_cannot_read_final_spec_or_start_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "checkpoints" / c20_recipe.JOB_IDS["pilot"]
            archive.mkdir(parents=True)
            (archive / "pilot-summary.json").write_text(json.dumps({"status": "keep_c19"}))
            with patch.object(c20_recipe, "CAMPAIGN", root), patch.object(c20_run, "exec_run") as run:
                self.assertEqual(c20_run.final_job(root / "unreadable-final-spec.json"), 0)
                run.assert_not_called()

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_plain_cpt_keeps_source_eos_and_does_not_create_one(self):
        from sepalith.training.sft.campaign_edit_data import target_only_collator
        from sepalith.training.sft.night_controls import verify_microbatch

        for suffix in ([], [1]):
            row = c20_data.token_row("cpt:fixture", "cpt_replay", "R", [0], [20, 21], [], suffix)
            with self.assertRaises(ValueError):
                validate_row(row, 32)
            validated = validate_row(row, 32, allow_cpt=True)
            batch = target_only_collator([validated], 32)
            self.assertEqual(batch["labels"].tolist(), [[-100, 20, 21, *suffix]])
            self.assertEqual(verify_microbatch(batch, [validated])["eos_supervised_rows"], len(suffix))

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_cpu_resident_optimizer_roundtrip_preserves_fp32_and_next_update(self):
        import torch

        from sepalith.training.optim.full_weight_optimizer import OptimizerConfig
        from sepalith.training.sft.c20_optimizer import build_offloaded_optimizer

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)

        config = OptimizerConfig(arm="adamw_anchor", hidden_lr=2.5e-6, side_lr=2.5e-7)
        model = Tiny()
        optimizer, _ = build_offloaded_optimizer(model, config)
        param = model.model.embed_tokens.weight
        param.grad = torch.ones_like(param)
        optimizer.step()
        other = copy.deepcopy(model)
        resumed, _ = build_offloaded_optimizer(other, config)
        resumed.load_state_dict(optimizer.state_dict())
        tracked = resumed.state[other.model.embed_tokens.weight]["exp_avg"]
        self.assertEqual((tracked.device.type, tracked.dtype), ("cpu", torch.float32))
        rng = torch.get_rng_state()
        optimizer.step()
        torch.set_rng_state(rng)
        other.model.embed_tokens.weight.grad = torch.ones_like(other.model.embed_tokens.weight)
        resumed.step()
        torch.testing.assert_close(param, other.model.embed_tokens.weight, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
