"""CPU coverage for the frozen C20 dose, clustered gates, and replay masking."""

import copy
import importlib.util
import json
import signal
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sepalith.training.sft.c20 import data as c20_data, recipe as c20_recipe, run as c20_run, trainer
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

    def test_sigterm_during_training_waits_for_verified_checkpoint_and_defers(self):
        from sepalith.training.sft.c20.night_controls import DeadlineStop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "pilot.json"
            save_stop = root / "save-stop.json"
            archive = root / "archive"
            recipe = {"mode": "pilot", "outputs": {
                "archive": str(archive), "evidence": str(root / "evidence"), "graceful_stop": str(save_stop),
            }}
            trainer.write_json(recipe_path, recipe)
            recipe_hash = c20_run.sha256(recipe_path)
            cp = c20_run.checkpoint(recipe, 7)
            env = {"SEPALITH_NIGHT_ATTEMPT_DIR": str(root / "attempt"),
                   "SEPALITH_HARD_STOP_EPOCH": str(time.time() + 3000),
                   "SEPALITH_DEADLINE_EPOCH": str(time.time() + 2000),
                   "SEPALITH_NIGHT_JOB_ID": c20_recipe.JOB_IDS["pilot"]}
            argv = ["c20", "pilot", "--recipe", str(recipe_path), "--sha256", recipe_hash]

            with (
                patch.dict("os.environ", env, clear=True),
                patch("sys.argv", argv),
                patch.object(c20_run, "queue_stop", c20_run.QueueStop()),
                patch.object(c20_run, "guard"),
                patch.object(c20_run, "verify_smoke"),
                patch.object(c20_run, "gate", return_value={}) as gate,
                patch.object(trainer, "load_bound", return_value=recipe),
                patch.object(trainer, "identity", return_value={"recipe": recipe_hash}),
                patch.object(c20_recipe, "require_disk"),
                patch.object(c20_run.exec_jobs, "CUDA_LOCK", root / "cuda.lock"),
                patch.object(c20_run.subprocess, "Popen") as popen,
                patch.object(c20_run, "verify_checkpoint", return_value={"step": 7}) as verify,
            ):
                process = popen.return_value.__enter__.return_value

                def wait(timeout: int) -> int:
                    command = popen.call_args.args[0]
                    if process.wait.call_count == 1:
                        signal.raise_signal(signal.SIGTERM)
                        raise subprocess.TimeoutExpired(command, timeout)
                    control = SimpleNamespace(should_save=False, should_training_stop=False)
                    self.assertEqual(DeadlineStop(None, None, save_stop, recipe_hash).apply(7, control), "graceful_stop")
                    self.assertTrue(control.should_save and control.should_training_stop)
                    trainer.write_json(cp / "campaign-manifest.json", {"step": 7})
                    trainer.write_json(archive / "run-result.json", {
                        "global_step": 7, "initial_cursor": 0, "observed_draws": 7 * 16,
                        "terminal_checkpoint": str(cp), "peak_cuda_reserved_bytes": 1024,
                    })
                    out = Path(command[command.index("--output") + 1])
                    trainer.write_json(out / "terminal.json", {"status": "completed", "child_exit_code": 0, "reason": None})
                    return 0

                process.wait.side_effect = wait
                handler = signal.getsignal(signal.SIGTERM)
                try:
                    self.assertEqual(c20_run.main(), 75)
                finally:
                    signal.signal(signal.SIGTERM, handler)
                process.terminate.assert_not_called()
                verify.assert_called_once_with(cp, {"recipe": recipe_hash}, require_full=True,
                                               expected_checkpoint_kind="full_weights")
                gate.assert_called_once_with(recipe, recipe_path, 0)
                self.assertTrue((c20_run.evidence(recipe, 7) / "train-segment.json").is_file())

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_queue_stop_during_guard_preflight_defers_without_receipt(self):
        class Guard:
            returncode = -signal.SIGTERM
            calls = 0
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def terminate(self): self.terminated = True
            def wait(self, timeout=None):
                Guard.calls += 1
                if Guard.calls == 1:
                    raise subprocess.TimeoutExpired("guard", timeout or 1)
                return self.returncode

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"SEPALITH_NIGHT_ATTEMPT_DIR": str(root), "SEPALITH_HARD_STOP_EPOCH": str(time.time() + 3000)}
            with patch.dict(c20_run.os.environ, env), \
                    patch.object(c20_run.exec_jobs, "CUDA_LOCK", root / "lock" / "cuda.lock"), \
                    patch.object(c20_run.queue_stop, "requested", return_value=True), \
                    patch.object(c20_run.subprocess, "Popen", lambda *a, **k: Guard()):
                self.assertEqual(c20_run.guarded_launch("x", "val"), c20_run.EXIT_DEFERRED)
            Guard.calls, Guard.returncode = 0, 1
            with patch.dict(c20_run.os.environ, env), \
                    patch.object(c20_run.exec_jobs, "CUDA_LOCK", root / "lock" / "cuda.lock"), \
                    patch.object(c20_run.queue_stop, "requested", return_value=True), \
                    patch.object(c20_run.subprocess, "Popen", lambda *a, **k: Guard()):
                with self.assertRaises(FileNotFoundError):
                    c20_run.guarded_launch("x", "val")

    def test_nonfinite_raw_bf16_loss_is_rejected_even_with_finite_gradient(self):
        import torch

        parameter = torch.tensor(2.0, dtype=torch.bfloat16, requires_grad=True)
        offset = torch.tensor(float("nan"), dtype=torch.bfloat16)
        self.assertTrue(torch.isfinite(torch.autograd.grad(parameter + offset, parameter)[0]))
        backward_calls = []
        parameter.register_hook(lambda grad: backward_calls.append(grad))
        optimizer = torch.optim.SGD([parameter], lr=1)
        with self.assertRaisesRegex(ValueError, "nonfinite raw C20 loss"):
            trainer.checked_loss(parameter + offset).backward()
            optimizer.step()
        self.assertEqual(backward_calls, [])
        self.assertEqual(parameter.item(), 2.0)
        finite = (parameter.square(), {"auxiliary": None})
        self.assertIs(trainer.checked_loss(finite), finite)

    def test_pilot_rejects_stale_or_changed_smoke_evidence_before_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / c20_recipe.JOB_IDS["smoke"]
            receipt_path = state / "smoke-pass.json"
            smoke_path = root / "smoke.json"
            smoke = {"mode": "smoke", "source": {}, "outputs": {
                "archive": str(root / "smoke-checkpoints"), "evidence": str(state),
            }}
            trainer.write_json(smoke_path, smoke)
            pilot = {"mode": "pilot", "source": {}, "smoke_recipe": c20_run.record(smoke_path),
                     "outputs": {"archive": str(root / "pilot-checkpoints")}}
            pilot_path = root / "pilot.json"

            for step in (1, 2):
                cp = c20_run.checkpoint(smoke, step)
                trainer.write_json(cp / "campaign-manifest.json", {"step": step})
                trainer.write_json(c20_run.evidence(smoke, step) / "train-segment.json", {
                    "binding": c20_run.binding(smoke, smoke_path, step), "initial_cursor": (step - 1) * 16,
                    "observed_draws": 16, "terminal_checkpoint": str(cp), "peak_cuda_reserved_bytes": 1024,
                })
            out = c20_run.evidence(smoke, 2)
            identity = c20_run.binding(smoke, smoke_path, 2)
            label = c20_recipe.SELECTED
            tasks = out / "smoke-tasks.jsonl"
            tasks.write_text("{}\n")
            val = {"coverage": {"tasks_scored": 24, "tasks_total": 24}}
            paths = {"validation": out / f"validation/records-{label}.jsonl",
                     "validation_metrics": out / "validation/metrics.json",
                     "validation_receipt": out / "validation/receipt.json",
                     "dev75": out / "dev75.json", "cpt": out / "cpt.json"}
            trainer.write_json(paths["validation_metrics"], {label: val})
            trainer.write_json(paths["validation_receipt"], {
                "status": "complete", "tasks": {"sha256": c20_run.sha256(tasks)},
                "models": {label: {"path": str(c20_run.checkpoint(smoke, 2))}},
            })
            paths["validation"].write_text("{}\n")
            dev75, cpt = {"nofp": 2}, {"nll": 1.5}
            trainer.write_json(paths["dev75"], {"binding": identity, "metrics": dev75})
            trainer.write_json(paths["cpt"], {"binding": identity, "metrics": cpt})
            trainer.write_json(out / "gate.json", {
                "binding": identity, "inputs": {name: c20_run.record(path) for name, path in paths.items()},
                "dev75": dev75, "cpt": cpt, "validation": val,
            })

            with (
                patch.object(c20_recipe, "STATE", root),
                patch.object(trainer, "load_bound", return_value=pilot),
                patch.object(c20_run, "verify_saved_checkpoint"),
                patch.object(c20_run, "gate", return_value=None) as gate,
            ):
                receipt = c20_run.smoke_receipt(smoke, smoke_path)
                trainer.write_json(receipt_path, {**receipt, "binding": {**identity, "recipe_sha256": "stale"}})
                with self.assertRaisesRegex(ValueError, "belongs to a different recipe"):
                    c20_run.train_job(pilot_path)
                gate.assert_not_called()

                trainer.write_json(receipt_path, receipt)
                c20_run.verify_smoke(pilot)
                manifest = c20_run.checkpoint(smoke, 2) / "campaign-manifest.json"
                original = manifest.read_text()
                manifest.write_text("{}")
                with self.assertRaisesRegex(ValueError, "smoke did not resume for a full update"):
                    c20_run.train_job(pilot_path)
                gate.assert_not_called()
                manifest.write_text(original)

                paths["dev75"].write_text("{}")
                with self.assertRaisesRegex(ValueError, "input differs"):
                    c20_run.train_job(pilot_path)
                gate.assert_not_called()

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_plain_cpt_keeps_source_eos_and_does_not_create_one(self):
        from sepalith.training.sft.c20.night_controls import verify_microbatch
        from sepalith.training.sft.c20.rows import target_only_collator, validate_row as validate_c20_row

        for suffix in ([], [1]):
            row = c20_data.token_row("cpt:fixture", "cpt_replay", "R", [0], [20, 21], [], suffix)
            with self.assertRaises(ValueError):
                validate_row(row, 32)
            validated = validate_c20_row(row, 32)
            batch = target_only_collator([validated], 32)
            self.assertEqual(batch["labels"].tolist(), [[-100, 20, 21, *suffix]])
            self.assertEqual(verify_microbatch(batch, [validated])["eos_supervised_rows"], len(suffix))

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_cpu_resident_optimizer_roundtrip_preserves_fp32_and_next_update(self):
        import torch

        from sepalith.training.optim.full_weight_optimizer import OptimizerConfig
        from sepalith.training.sft.c20.optimizer import build_offloaded_optimizer

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
