"""Exec-eval night jobs: resumable generation, exit codes and metrics, with CUDA mocked (no GPU needed)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from sepalith.exec_eval import jobs


def fake_task(i: int, family: str = "finish_block", bucket: str = "short", function: str | None = None) -> dict:
    return {"id": f"t{i:03d}", "function_id": function or f"f{i % 5}", "family": family,
            "prompt_ids": [0] + list(range(2, 2 + 5 + i % 7)), "curriculum": {"bucket": bucket, "target_lines": 3},
            "exec": {"before": "a\nb", "after": ""}}


HAS_TORCH = importlib.util.find_spec("torch") is not None


class GenerationTests(unittest.TestCase):
    def test_resume_and_deadline(self):
        tasks = [fake_task(i) for i in range(10)]
        calls = []

        def generate(prompts, number):
            calls.append(len(prompts))
            return [[[5, 6, 1]] for _ in prompts]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c.jsonl"
            with mock.patch.dict(os.environ, {"SEPALITH_DEADLINE_EPOCH": str(time.time() - 1)}):
                self.assertEqual(jobs.generate_model("m", tasks, out, generate, samples=1, prompts_per_batch=4)["status"],
                                 "deferred")
            self.assertEqual(calls, [])
            first = jobs.generate_model("m", tasks[:6], out, generate, samples=1, prompts_per_batch=4)
            self.assertEqual((first["status"], first["generated"]), ("complete", 6))
            with out.open("a") as stream:
                stream.write('{"model": "m", "task_id": "t0')  # a torn line from a crash
            second = jobs.generate_model("m", tasks, out, generate, samples=1, prompts_per_batch=4)
            self.assertEqual(second["generated"], 4)  # only the remaining tasks
            self.assertEqual(len(jobs.read_jsonl(out)), 10)

    @unittest.skipUnless(HAS_TORCH, "needs torch")
    def test_sampling_regroups_and_truncates(self):
        import torch

        class FakeModel:
            def generate(self, input_ids, attention_mask, **kwargs):
                self.kwargs = kwargs
                rows = input_ids.shape[0]
                tail = torch.tensor([[7, 1, 1, 1]] * rows)
                return torch.cat([input_ids, tail], dim=1)

        model = FakeModel()
        out = jobs.batched_sample_generate(model, [[0, 2, 3], [0, 4]], k=3, cap=4, seed=1, device="cpu")
        self.assertEqual(out, [[[7, 1]] * 3, [[7, 1]] * 3])
        self.assertEqual((model.kwargs["do_sample"], model.kwargs["temperature"], model.kwargs["top_k"]), (True, 1.0, 0))


@unittest.skipUnless(HAS_TORCH, "the inner job imports torch")
class InnerExitCodeTests(unittest.TestCase):
    """The GPU entry point with CUDA, model loading and scoring mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.tasks = root / "tasks.jsonl"
        self.tasks.write_text("".join(json.dumps(fake_task(i)) + "\n" for i in range(6)))
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"x")
        self.models = root / "models.json"
        self.models.write_text(json.dumps({"models": [{"label": "a", "name": "A", "path": str(model_dir)},
                                                      {"label": "b", "name": "B", "path": str(model_dir)}]}))
        self.manifest = root / "manifest.json"
        self.manifest.write_text("{}")
        self.attempt = root / "work" / "attempts" / "x"
        self.attempt.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **extra) -> argparse.Namespace:
        base = dict(kind="holdout", job_id="T", work_dir=Path(self.tmp.name) / "work", tasks=self.tasks,
                    tasks_sha256=jobs.sha256(self.tasks), models=self.models, k=8, batch=4, cap=8, score_workers=1,
                    model_budget_minutes=0, attempt_dir=self.attempt)
        base.update(extra)
        return argparse.Namespace(**base)

    def run_inner(self, args, deadline=None):
        import torch

        patches = [
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(torch.cuda, "init"),
            mock.patch.object(torch.cuda, "get_device_name", return_value="fake"),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            mock.patch.object(torch.cuda, "empty_cache"),
            mock.patch.object(torch, "zeros"),
            mock.patch.object(jobs, "load_model", return_value=(object(), {"contract": "mocked"})),
            mock.patch("sepalith.training.eval.campaign_eval.batched_greedy_generate",
                       side_effect=lambda model, prompts, **kw: [[9, 1] for _ in prompts]),
            mock.patch.object(jobs, "score"),
            mock.patch.object(jobs, "MANIFEST", self.manifest),
            mock.patch.object(jobs, "MANIFEST_SHA256", jobs.hashlib.sha256(self.manifest.read_bytes()).hexdigest()),
            mock.patch("sepalith.exec_env.gate.gate_version",
                       return_value={"name": "v3", "commit": "x", "exec_env_modified": False,
                                     "fingerprint": jobs.GATE_FINGERPRINT}),
        ]
        env = {"SEPALITH_DEADLINE_EPOCH": str(deadline)} if deadline else {}
        with mock.patch.dict(os.environ, env):
            for p in patches:
                p.start()
            try:
                with mock.patch.object(jobs, "sha256", side_effect=lambda p: jobs.hashlib.sha256(Path(p).read_bytes()).hexdigest()):
                    return jobs.inner(args)
            finally:
                for p in reversed(patches):
                    p.stop()

    def test_census_samples_k_per_task_with_the_first_model_only(self):
        args = self.args(kind="census", k=2)
        with mock.patch.object(jobs, "batched_sample_generate",
                               side_effect=lambda model, prompts, k, **kw: [[[9, 1]] * k for _ in prompts]):
            self.assertEqual(self.run_inner(args), 0)
        self.assertEqual(len(jobs.read_jsonl(args.work_dir / "completions-a.jsonl")), 12)
        self.assertFalse((args.work_dir / "completions-b.jsonl").exists())

    def test_greedy_complete_then_status(self):
        args = self.args()
        self.assertEqual(self.run_inner(args), 0)
        self.assertEqual(len(jobs.read_jsonl(args.work_dir / "completions-a.jsonl")), 6)
        self.assertEqual(len(jobs.read_jsonl(args.work_dir / "completions-b.jsonl")), 6)
        receipt = json.loads((args.work_dir / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(json.loads((self.attempt / "inner-status.json").read_text())["exit"], 0)

    def test_deadline_defers(self):
        args = self.args()
        self.assertEqual(self.run_inner(args, deadline=time.time() + 10), 75)
        receipt = json.loads((args.work_dir / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "partial")

    def test_bad_tasks_hash_needs_user(self):
        args = self.args(tasks_sha256="0" * 64)
        self.assertEqual(self.run_inner(args), 3)


class MetricsTests(unittest.TestCase):
    def test_holdout_and_census(self):
        records = []
        for i in range(40):
            records.append({"task_id": f"t{i}", "function_id": f"f{i % 8}", "family": "finish_block",
                            "bucket": ["short", "medium", "long"][i % 3], "reward": (i % 4) / 3, "all_pass": i % 4 == 3,
                            "exact_match": i % 8 == 3, "protocol_valid": True, "cap_hit": False, "outcome": "partial"})
        m = jobs.holdout_metrics(records, "a", 50)
        self.assertEqual(m["coverage"], {"tasks_scored": 40, "tasks_total": 50})
        lo, hi = m["overall"]["mean_reward"]["ci95"]
        self.assertTrue(lo <= m["overall"]["mean_reward"]["mean"] <= hi)
        self.assertEqual(m["overall"]["functions"], 8)
        diff = jobs.paired_difference(records, [{**r, "reward": float(r["reward"]) + 0.1} for r in records], "reward", "s")
        self.assertAlmostEqual(diff["mean"], 0.1, places=4)
        census = []
        for t in range(6):
            for s in range(4):
                census.append({"task_id": f"t{t}", "function_id": f"f{t}", "family": "finish_block", "bucket": "short",
                               "reward": 1.0 if (t < 2 or (t < 4 and s == 0)) else 0.0,
                               "all_pass": t < 2 or (t < 4 and s == 0)})
        c = jobs.census_metrics(census, 4)
        self.assertEqual((c["overall"]["all_right_share"], c["overall"]["signal_share_pass"], c["overall"]["all_wrong_share"]),
                         (round(2 / 6, 4), round(2 / 6, 4), round(2 / 6, 4)))

    def test_select_tasks_is_stratified_and_deterministic(self):
        tasks = [fake_task(i, bucket=b, function="f") for i, b in enumerate(["short"] * 6 + ["long"] * 2)]
        picked = jobs.select_tasks(tasks, 4, "f")
        self.assertEqual(sum(1 for t in picked if t["curriculum"]["bucket"] == "long"), 2)
        self.assertEqual([t["id"] for t in picked], [t["id"] for t in jobs.select_tasks(tasks, 4, "f")])

    def test_remainder_lines(self):
        self.assertEqual(jobs.remainder_lines({"exec": {"before": "a\nb\nc", "after": ""}}, 10), (4, 10))
        self.assertEqual(jobs.remainder_lines({"exec": {"before": "a", "after": "x\ny"}}, 10), (2, 8))
        self.assertEqual(jobs.expand_slices("S000..S002,H001"), ["S000", "S001", "S002", "H001"])


if __name__ == "__main__":
    unittest.main()
