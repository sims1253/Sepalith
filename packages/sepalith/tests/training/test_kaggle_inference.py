"""C03 Kaggle inference harness on tiny fp32 Llamas on CPU.

Covers the batch-granular work split (every batch keeps its PC composition),
the entry's worker path in prompts and panel mode (two workers reproduce one
worker exactly), input refusals, the TRAIN parity prompt selection, the parity
comparison and the driver's kernel packaging. The real-weight checks are the
Kaggle smoke and the night reference job.
"""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch

from sepalith.training.eval import campaign_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_batched_dev_generation as batched_harness
import test_dev_generation_runtime as harness

ROOT = Path(__file__).resolve().parents[4]
C03 = ROOT / "docs/training/plan-20260927/c03"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load("c03_infer_entry", ROOT / "scripts/kaggle/infer_entry.py")
driver = load("c03_infer_push", ROOT / "scripts/kaggle/infer_push.py")
train_prompts = load("c03_train_prompts", C03 / "train_prompts.py")
compare = load("c03_compare_parity", C03 / "compare_parity.py")
CAP = 12


def worker_args(**overrides):
    values = {"max_new_tokens": CAP, "batch_size": 3, "kv_cache": "dynamic", "temperature": 0.0, "samples": 1,
              "top_p": 1.0, "seed": 3407}
    values.update(overrides)
    return Namespace(**values)


def run_plan(model, tokenizer, plan, args, root):
    plan = {**plan, "output_dir": str(root)}
    summaries = [entry.run_worker(args, plan, model, tokenizer, worker, Path(root) / f"worker-{worker}")
                 for worker in range(int(plan["workers"]))]
    return entry.merge(args, plan, summaries), [json.loads(line) for line in (Path(root) / "generations.jsonl").read_text().splitlines()]


class BatchPlanTests(unittest.TestCase):
    def test_work_split_keeps_every_global_batch_intact(self):
        rng = random.Random(3407)
        items = [{"id": f"c{i:03d}", "prompt_tokens": rng.randrange(5, 60)} for i in range(75)]
        reference = [[items[i]["id"] for i in batch] for batch in campaign_eval.generation_batches(
            [(item["id"], item["prompt_tokens"]) for item in items], 32)]
        for shards in (1, 2, 3):
            for workers in (1, 2):
                seen = []
                for shard in range(shards):
                    plan = entry.batch_plan(items, 32, shard_index=shard, shard_count=shards, workers=workers)
                    for worker in range(workers):
                        mine = [b for b in plan["batches"] if b["worker"] == worker]
                        ids = {i for b in mine for i in b["ids"]}
                        entry.check_rebatching([item for item in items if item["id"] in ids], mine, 32)
                    seen.extend(b["ids"] for b in plan["batches"])
                self.assertEqual(sorted(map(tuple, seen)), sorted(map(tuple, reference)))

    def test_prompt_rows_outside_train_and_dev_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(json.dumps({"id": "a", "split": "final", "prompt_token_ids": [0, 5]}) + "\n")
            with self.assertRaisesRegex(ValueError, "not TRAIN or DEV"):
                entry.prompt_items(path, CAP)
            path.write_text(json.dumps({"id": "a", "split": "dev", "prompt_token_ids": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                entry.prompt_items(path, CAP)

    def test_sources(self):
        with self.assertRaisesRegex(ValueError, "scheme"):
            entry.resolve_source("scholzmx/x", "/tmp")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(entry.resolve_source(f"path:{tmp}", "/tmp"), Path(tmp))
        self.assertEqual(entry.split_revision("a/b/c@abc"), ("a/b/c", "abc"))


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, torch.get_num_threads()))
        cls.model = batched_harness.tiny_model(initializer_range=0.5)
        cls.tokenizer = harness.TextTokenizer()

    def prompts_plan(self, tmp, workers, count=10):
        rng = random.Random(7)
        rows = []
        for i in range(count):
            length = rng.randrange(2, 30)
            rows.append({"id": f"p{i:02d}", "split": "train", "family": "f",
                         "prompt_token_ids": [0] + [rng.randrange(2, 1024) for _ in range(length)]})
        path = Path(tmp) / "prompts.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        items = entry.prompt_items(path, CAP)
        return {"mode": "prompts", "prompts_path": str(path), "items": items, "workers": workers,
                **entry.batch_plan(items, 3, workers=workers)}, rows

    def test_prompts_mode_two_workers_equal_one_worker_and_the_evaluator_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan1, rows = self.prompts_plan(tmp, 1)
            plan2, _ = self.prompts_plan(tmp, 2)
            one, records1 = run_plan(self.model, self.tokenizer, plan1, worker_args(), Path(tmp) / "one")
            two, records2 = run_plan(self.model, self.tokenizer, plan2, worker_args(), Path(tmp) / "two")
            def strip(rows):
                return [(r["id"], r["generated_ids"], r["finish_reason"], r["cap_hit"]) for r in rows]

            self.assertEqual(strip(records1), strip(records2))
            self.assertEqual({r["worker"] for r in records2}, {0, 1})
            expected = batched_harness.batched(self.model, [(r["id"], r["prompt_token_ids"]) for r in rows], 3, CAP,
                                               (1, 130073))
            self.assertEqual([r["generated_ids"] for r in records1], expected)
            self.assertEqual(one["calls_with_nonfinite_logits"], 0)
            self.assertEqual(two["records"], 10)

    def test_sampling_returns_k_samples_per_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, _ = self.prompts_plan(tmp, 1, count=4)
            args = worker_args(temperature=0.8, samples=3)
            summary, records = run_plan(self.model, self.tokenizer, plan, args, Path(tmp) / "s")
            self.assertEqual(len(records), 12)
            self.assertEqual(sorted({r["sample"] for r in records}), [0, 1, 2])
            _, again = run_plan(self.model, self.tokenizer, plan, args, Path(tmp) / "s2")
            self.assertEqual([r["generated_ids"] for r in records], [r["generated_ids"] for r in again])

    def test_panel_mode_two_workers_equal_one_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = harness.write_panel(Path(tmp) / "panel.jsonl", 11, "dev")
            items = entry.panel_items(panel, self.tokenizer, CAP)
            results = []
            for workers in (1, 2):
                plan = {"mode": "panel", "panel": "dev75", "model_path": tmp, "items": items, "workers": workers,
                        **entry.batch_plan(items, 4, workers=workers)}
                summary, records = run_plan(self.model, self.tokenizer, plan, worker_args(batch_size=4),
                                            Path(tmp) / f"w{workers}")
                results.append((summary, records))
            (one, rec1), (two, rec2) = results
            self.assertEqual([r["generated_ids"] for r in rec1], [r["generated_ids"] for r in rec2])
            self.assertEqual(one["counts"], two["counts"])
            self.assertEqual(one["cases"], 11)
            self.assertTrue(one["nll_finite"])


class TrainPromptTests(unittest.TestCase):
    def test_allocation_matches_the_documented_quota(self):
        sizes = {"finish_block": 7785, "pipe_rewrite": 1983, "format_propagation": 1615, "rename_propagation": 1578,
                 "no_op": 1094, "roxygen_drafting": 815, "na_rm_propagation": 136}
        self.assertEqual(train_prompts.allocate(sizes, 200),
                         {"finish_block": 104, "pipe_rewrite": 26, "format_propagation": 21, "rename_propagation": 21,
                          "no_op": 15, "roxygen_drafting": 11, "na_rm_propagation": 2})

    def test_selection_is_order_independent_and_train_only(self):
        rows = [{"id": f"r{i:03d}", "split": "train", "family": "ab"[i % 2], "package_id": "p",
                 "input_ids": [0, 5, 6, 7], "target_start": 3} for i in range(40)]
        first, _ = train_prompts.select(rows, 10, 3407)
        second, _ = train_prompts.select(list(reversed(rows)), 10, 3407)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertEqual(train_prompts.prompt_record(first[0])["prompt_token_ids"], [0, 5, 6])
        with self.assertRaises(SystemExit):
            train_prompts.select(rows + [{**rows[0], "id": "x", "split": "dev"}], 10, 3407)


class CompareTests(unittest.TestCase):
    def write(self, folder, records, cases, batch=32):
        folder.mkdir(parents=True)
        (folder / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (folder / "cases.json").write_text(json.dumps({"results": cases}))
        (folder / "summary.json").write_text(json.dumps({"generation": {"batch_size": batch, "kv_cache": "dynamic",
                                                                         "max_new_tokens": 1024},
                                                         "result": {"calls_with_nonfinite_logits": 0, "nll_finite": True}}))

    def fixture(self, tmp, changed=0, batch=32):
        rng = random.Random(1)
        generated = [[rng.randrange(9) for _ in range(20)] for _ in range(20)]
        records = [{"id": f"c{i}", "sample": 0, "generated_ids": generated[i], "prompt_sha256": f"s{i}"} for i in range(20)]
        cases = [{"id": f"c{i}", "protocol_valid": i % 2 == 0, "cap_hit": i % 5 == 0, "expected_noop": i % 3 == 0,
                  "exact_region": i % 4 == 0, "suggestion": i % 2 == 0} for i in range(20)]
        ref = Path(tmp) / "ref"
        ref.mkdir()
        (ref / "dev75-generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (ref / "dev75.cases").mkdir()
        (ref / "dev75.cases/cases.json").write_text(json.dumps({"results": cases}))
        receipt = Path(tmp) / "references.json"
        receipt.write_text(json.dumps({"settings": {"batch_size": 32, "kv_cache": "dynamic", "cap": 1024}}))
        perturbed = [ids[:7] + [99] + ids[8:] if i < changed else ids for i, ids in enumerate(generated)]
        candidate = [{**r, "generated_ids": perturbed[i]} for i, r in enumerate(records)]
        self.write(Path(tmp) / "kaggle", candidate, cases, batch)
        return ["--reference-dir", str(ref), "--reference-receipt", str(receipt), "--dev75", str(Path(tmp) / "kaggle"),
                "--output", str(Path(tmp) / "out.json")]

    def test_identical_passes_and_divergence_is_located(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(compare.main(self.fixture(tmp)), 0)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(compare.main(self.fixture(tmp, changed=3)), 1)
            out = json.loads((Path(tmp) / "out.json").read_text())
            self.assertEqual(out["sets"]["dev75"]["identity"]["identical"], 17)
            self.assertEqual(out["sets"]["dev75"]["identity"]["first_divergence"]["max"], 7)
            self.assertFalse(out["checks"]["identity_at_least_90pct"])

    def test_different_batching_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            compare.main(self.fixture(tmp, batch=5))
            self.assertEqual(json.loads((Path(tmp) / "out.json").read_text())["verdict"], "not_comparable_batching_differs")


class DriverTests(unittest.TestCase):
    def test_kernel_has_no_secret_and_attaches_kaggle_sources(self):
        args = driver.parse_args(["--run-id", "c03-test", "--estimate-hours", "1", "--", "--model",
                                  "kaggle:m0hawk/sepalith-arm-a/full", "--panel", "dev75"])
        source, metadata = driver.kernel_files(args, "a" * 40, "b" * 64, "m0hawk/sepalith-infer-c03-test")
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["dataset_sources"], ["m0hawk/sepalith-arm-a"])
        self.assertTrue(metadata["is_private"] and metadata["enable_internet"])
        self.assertFalse(any(p.search(source) for p in driver.SECRET_PATTERNS))
        self.assertNotIn("HF_TOKEN", source)
        self.assertNotIn("KAGGLE_API_TOKEN", source)
        compile(source, "kernel.py", "exec")
        self.assertTrue(any(p.search("x = 'KGAT_" + "0123456789abcdef'") for p in driver.SECRET_PATTERNS))

    def test_run_id_and_reserved_arguments(self):
        with self.assertRaises(SystemExit):
            driver.parse_args(["--run-id", "Bad_ID", "--estimate-hours", "1", "--", "--panel", "dev75"])
        with self.assertRaises(SystemExit):
            driver.parse_args(["--run-id", "ok-id", "--estimate-hours", "1", "--", "--output-dir", "/x"])


if __name__ == "__main__":
    unittest.main()
