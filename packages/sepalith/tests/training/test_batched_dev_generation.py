"""Batched greedy DEV generation (C07 follow-up) on tiny fp32 Llamas on CPU.

The batched path must reproduce the per-case path: same generated ids for every
case, in panel order, with per-row stops and the generation cap.  fp32 on CPU
makes that an exact comparison; bf16 on the GPU is measured by
``docs/training/plan-20260927/c07/validate_batched_eval.py``.
"""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import torch
import transformers

from sepalith import campaign_protocol
from sepalith.training.eval import campaign_eval, milestone_gate
from sepalith.training.eval import run_dev_generation as driver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dev_generation_runtime as harness

CAP = 24


def tiny_model(initializer_range=0.02):
    config = transformers.LlamaConfig(
        vocab_size=1024, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=4096, bos_token_id=0, eos_token_id=1, pad_token_id=1,
        tie_word_embeddings=False, initializer_range=initializer_range)
    torch.manual_seed(3407)
    return transformers.LlamaForCausalLM(config).eval()


def sensitive_checkpoint(directory):
    """A tiny Llama whose greedy output depends on the whole prompt (the harness model's barely does)."""
    tiny_model(initializer_range=0.5).save_pretrained(directory)
    return Path(directory)


def per_case(model, prompt, cap, stop_ids):
    """The original evaluator call: one unpadded case."""
    ids = torch.tensor([prompt])
    with torch.no_grad():
        sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  **campaign_eval.greedy_generate_kwargs(cap, stop_ids))
    return sequence[0, len(prompt):].tolist()


def batched(model, cases, batch_size, cap, stop_ids, kv_cache="dynamic"):
    """Batch ``cases`` ((id, prompt) pairs) as the evaluator does; return ids in the input order."""
    out = {}
    for indexes in campaign_eval.generation_batches([(case_id, len(prompt)) for case_id, prompt in cases], batch_size):
        with torch.no_grad():
            rows = campaign_eval.batched_greedy_generate(model, [cases[i][1] for i in indexes], cap=cap,
                                                         stop_ids=stop_ids, kv_cache=kv_cache)
        out.update(zip(indexes, rows, strict=True))
    return [out[index] for index in range(len(cases))]


class BatchedGreedyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, torch.get_num_threads()))
        cls.model = tiny_model()
        rng = random.Random(3407)
        # Mixed prompt lengths, including ties, so batches need left padding.
        lengths = [3, 7, 7, 12, 19, 25, 31, 40, 40, 2, 16, 9, 28, 5]
        cls.cases = [(f"case-{i:02d}", [0] + [rng.randrange(2, 1024) for _ in range(n - 1)]) for i, n in enumerate(lengths)]
        free = [per_case(cls.model, prompt, CAP, (1,)) for _, prompt in cls.cases]
        # A second stop id the model actually emits mid-sequence, so some rows stop early.
        common = Counter(token for row in free for token in row[2:CAP - 4])
        cls.stop_ids = (1, common.most_common(1)[0][0])
        cls.reference = [per_case(cls.model, prompt, CAP, cls.stop_ids) for _, prompt in cls.cases]
        unused = sorted(set(range(2, 1024)) - {token for row in free for token in row})
        cls.never = (unused[0],)

    def test_the_fixture_mixes_early_stops_and_cap_hits(self):
        lengths = [len(row) for row in self.reference]
        self.assertTrue(any(n < CAP for n in lengths) and any(n == CAP for n in lengths), lengths)
        self.assertGreater(len({len(prompt) for _, prompt in self.cases}), 5)

    def test_batched_outputs_equal_per_case_outputs_in_fp32(self):
        for kv_cache in ("dynamic", "static"):
            for batch_size in (1, 2, 4, 5, len(self.cases)):
                with self.subTest(kv_cache=kv_cache, batch_size=batch_size):
                    self.assertEqual(batched(self.model, self.cases, batch_size, CAP, self.stop_ids, kv_cache), self.reference)

    def test_finished_rows_do_not_affect_the_others(self):
        # The same row gives the same output alone, with rows that stop at once, and with rows that run to the cap.
        early = [i for i, row in enumerate(self.reference) if len(row) < CAP]
        late = [i for i, row in enumerate(self.reference) if len(row) == CAP]
        target = late[0]
        for partners in ([], early, late[1:3], early + late[1:]):
            cases = [self.cases[target]] + [self.cases[i] for i in partners]
            with torch.no_grad():
                rows = campaign_eval.batched_greedy_generate(self.model, [prompt for _, prompt in cases], cap=CAP,
                                                             stop_ids=self.stop_ids)
            self.assertEqual(rows[0], self.reference[target], partners)

    def test_the_cap_is_kept_when_no_row_stops(self):
        reference = [per_case(self.model, prompt, CAP, self.never) for _, prompt in self.cases]
        self.assertTrue(all(len(row) == CAP for row in reference))
        self.assertEqual(batched(self.model, self.cases, 4, CAP, self.never), reference)
        for cap in (1, 5):
            rows = batched(self.model, self.cases, 6, cap, self.never)
            self.assertEqual([len(row) for row in rows], [cap] * len(self.cases))

    def test_stop_id_handling(self):
        rows = batched(self.model, self.cases, 5, CAP, self.stop_ids)
        for row in rows:
            stops = [i for i, token in enumerate(row) if token in self.stop_ids]
            # Either one stop token, at the end, or none and the row ran to the cap.
            self.assertTrue(stops == [len(row) - 1] or (not stops and len(row) == CAP), row)
        self.assertTrue(any(row[-1] == self.stop_ids[1] for row in rows))
        # Rows that finished early are padded with id 1 (itself a stop id); the cut removes the fill.
        self.assertEqual(campaign_eval.truncate_at_stop([5, 9, 7, 1, 1, 1], (1, 7), 6), [5, 9, 7])
        self.assertEqual(campaign_eval.truncate_at_stop([5, 1, 1, 1], (1, 130073), 4), [5, 1])
        self.assertEqual(campaign_eval.truncate_at_stop([130073, 1], (1, 130073), 2), [130073])
        self.assertEqual(campaign_eval.truncate_at_stop([5, 6, 8], (1,), 3), [5, 6, 8])
        with self.assertRaisesRegex(ValueError, "without a stop token"):
            campaign_eval.truncate_at_stop([5, 6], (1,), 3)
        with self.assertRaisesRegex(ValueError, "non-empty prompts"):
            campaign_eval.batched_greedy_generate(self.model, [[0, 5], []], cap=2, stop_ids=(1,))

    def test_batches_depend_only_on_prompt_lengths_and_ids(self):
        cases = [("b", 5), ("a", 5), ("c", 9), ("d", 1), ("e", 9)]
        self.assertEqual(campaign_eval.generation_batches(cases, 2), [[2, 4], [1, 0], [3]])
        names = lambda order, batches: [[order[i][0] for i in batch] for batch in batches]  # noqa: E731
        expected = names(cases, campaign_eval.generation_batches(cases, 2))
        self.assertEqual(expected, [["c", "e"], ["a", "b"], ["d"]])
        for seed in range(5):
            shuffled = list(cases)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(names(shuffled, campaign_eval.generation_batches(shuffled, 2)), expected)
        with self.assertRaisesRegex(ValueError, "unique case ids"):
            campaign_eval.generation_batches([("a", 1), ("a", 2)], 2)

    def test_order_is_restored_whatever_the_batching(self):
        shuffled = list(self.cases)
        random.Random(11).shuffle(shuffled)
        expected = [self.reference[self.cases.index(case)] for case in shuffled]
        self.assertEqual(batched(self.model, shuffled, 3, CAP, self.stop_ids), expected)

    def test_generation_batching_record(self):
        self.assertIsNone(campaign_eval.generation_batching(1, "dynamic"))
        record = campaign_eval.generation_batching(16)
        self.assertEqual({k: record[k] for k in ("batch_size", "ordering", "padding_side", "pad_token_id", "kv_cache")},
                         {"batch_size": 16, "ordering": "prompt_tokens_descending_then_case_id_ascending",
                          "padding_side": "left", "pad_token_id": 1, "kv_cache": "dynamic"})
        self.assertEqual(campaign_eval.generation_batching(1, "static")["batch_size"], 1)
        for size, cache in ((0, "dynamic"), (257, "dynamic"), (True, "dynamic"), (2.0, "dynamic"), (4, "paged")):
            with self.assertRaises(ValueError):
                campaign_eval.generation_batching(size, cache)


def varied_panel(path, count, prefix):
    """DEV-like rows whose prompts differ in length (the harness panel repeats one prompt)."""
    rows = []
    for index in range(count):
        value = "x <- " + "1" * (1 + (index * 7) % 23)
        noop = index % 3 == 0
        rows.append({"id": f"{prefix}-{index:03d}", "split": "dev", "family": "no_op" if noop else "rename_propagation",
                     "package_id": f"pkg-{index % 7}", "operation": "no_op" if noop else "replace",
                     "region_new": [value] if noop else [f"y <- {index}"], "context": harness.context(value).to_dict()})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class BatchedEvaluatorTests(unittest.TestCase):
    """End to end through ``run_dev_generation.main`` with the harness of the C05 runtime tests."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, torch.get_num_threads()))
        cls.tmp = tempfile.TemporaryDirectory(prefix="sepalith-c07-batched-")
        cls.tiny = sensitive_checkpoint(Path(cls.tmp.name) / "tiny")
        fx = harness.build_fixture(Path(cls.tmp.name) / "fixture")
        varied_panel(fx.dev75, 75, "d75")
        fx.recipe["development"]["panel"]["sha256"] = harness.sha(fx.dev75)
        fx.recipe_path.write_text(json.dumps(fx.recipe, sort_keys=True) + "\n")
        cls.fx = fx
        # Pick a second stop id the tiny model emits, so some cases stop before the cap of 8.
        free = cls.drive("free", [])
        # The token that splits the cases most evenly into "emits it before the cap" and "does not".
        outputs = [case["generated_ids"][:harness.CAP - 1] for case in free["files"]["dev75.json"]["cases"]["results"]]
        rows_with = Counter(token for ids in outputs for token in set(ids) if token != 1)
        cls.stop_ids = (1, max(sorted(rows_with), key=lambda token: min(rows_with[token], len(outputs) - rows_with[token])))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @classmethod
    def drive(cls, name, argv, stop_ids=None) -> dict:
        fx = cls.fx
        run = harness.new_run(fx, name)
        admission = harness.write_admission(fx, panel_sha=harness.sha(fx.dev75), output=run.output)
        with mock.patch.object(campaign_protocol, "NATIVE_EOG_IDS", stop_ids or campaign_protocol.NATIVE_EOG_IDS):
            record = harness.run_driver(driver, fx, run, cls.tiny, ["--dtype", "fp32", *argv], admission=admission)
        record["dir"] = run.dir
        return record

    def test_batched_driver_matches_the_per_case_driver(self):
        single = self.drive("single", [], self.stop_ids)
        batch = self.drive("batched", ["--generation-batch-size", "4"], self.stop_ids)
        cases = {name: rec["files"]["dev75.json"]["cases"]["results"] for name, rec in (("single", single), ("batch", batch))}
        lengths = [len(c["generated_ids"]) for c in cases["single"]]
        self.assertTrue(any(n < harness.CAP for n in lengths) and any(n == harness.CAP for n in lengths))
        self.assertTrue(any(c["stop_token_id"] == self.stop_ids[1] for c in cases["single"]))
        self.assertTrue(any(c["cap_hit"] for c in cases["single"]))
        # Same per-case records, field for field, in the panel's order (wall-clock fields aside).
        strip = lambda rows: [{k: v for k, v in row.items() if k != "generation_seconds"} for row in rows]  # noqa: E731
        self.assertEqual(strip(cases["batch"]), strip(cases["single"]))
        panel_ids = [json.loads(line)["id"] for line in self.fx.dev75.read_text().splitlines()]
        self.assertEqual([c["id"] for c in cases["batch"]], panel_ids)
        # One generate call per batch, left-padded (the masks are not all ones).
        generate = [call for call in batch["calls"] if call[0] == "generate"]
        self.assertEqual(len(generate), math.ceil(75 / 4))
        self.assertFalse(all(call[3] for call in generate))
        self.assertEqual([call for call in single["calls"] if call[0] == "forward"],
                         [call for call in batch["calls"] if call[0] == "forward"])
        # The summary and so the R1 counts are unchanged; the batching is recorded.
        results = {name: rec["files"]["dev75.json"]["result"] for name, rec in (("single", single), ("batch", batch))}
        summaries = {name: {k: v for k, v in result["summary"].items() if k not in ("cases_path", "generation_runtime")}
                     for name, result in results.items()}
        self.assertEqual(summaries["batch"], summaries["single"])
        self.assertNotIn("generation_runtime", results["single"]["summary"])
        batching = campaign_eval.generation_batching(4)
        self.assertEqual(results["batch"]["summary"]["generation_runtime"], batching)
        self.assertEqual(results["batch"]["generation_runtime"], {"dtype": "fp32", "cuda_visible_devices": "0", **batching})
        self.assertEqual(results["single"]["generation_runtime"], {"dtype": "fp32", "cuda_visible_devices": "0"})
        self.assertEqual(milestone_gate.generation_settings(results["batch"]),
                         milestone_gate.requested_generation_settings(harness.CAP, 4, dtype="fp32"))
        self.assertNotEqual(milestone_gate.generation_settings(results["batch"]),
                            milestone_gate.generation_settings(results["single"]))

    def test_the_recipe_selects_batching_and_the_cli_overrides_it(self):
        fx = self.fx
        original = fx.recipe_path.read_text()
        try:
            recipe = json.loads(original)
            recipe["dev_gate"]["generation_batching"] = {"batch_size": 16, "kv_cache": "static"}
            fx.recipe_path.write_text(json.dumps(recipe, sort_keys=True) + "\n")
            bound = self.drive("bound", [])
            self.assertEqual(bound["files"]["dev75.json"]["result"]["generation_runtime"]["batch_size"], 16)
            self.assertEqual(bound["files"]["dev75.json"]["result"]["generation_runtime"]["kv_cache"], "static")
            self.assertEqual(sum(call[0] == "generate" for call in bound["calls"]), math.ceil(75 / 16))
            per_case = self.drive("cli-per-case", ["--generation-batch-size", "1", "--generation-kv-cache", "dynamic"])
            self.assertEqual(per_case["files"]["dev75.json"]["result"]["generation_runtime"],
                             {"dtype": "fp32", "cuda_visible_devices": "0"})
            self.assertEqual(sum(call[0] == "generate" for call in per_case["calls"]), 75)
            with self.assertRaisesRegex(ValueError, "batch size"):
                self.drive("refused", ["--generation-batch-size", "0"])
        finally:
            fx.recipe_path.write_text(original)


class ParentBaselineBatchingTests(unittest.TestCase):
    """``parent_baseline`` passes the recipe's batching to the shared evaluator."""

    def test_parent_baseline_uses_the_batched_path(self):
        from sepalith.training.contracts import campaign_tokenizer_contract, trainer_tokenizer_alignment
        from sepalith.training.sft import parent_baseline

        with tempfile.TemporaryDirectory(prefix="sepalith-c07-baseline-") as tmp:
            root = Path(tmp)
            tiny = harness.tiny_checkpoint(root / "tiny")
            parent = root / "parent"
            parent.mkdir()
            (parent / "campaign-manifest.json").write_text("{}\n")
            panel = varied_panel(root / "dev75.jsonl", 75, "d75")
            recipe = {"parent": {"path": str(parent), "files": {"campaign-manifest.json": harness.sha(parent / "campaign-manifest.json")}},
                      "dev_gate": {"panels": {"dev75": {"path": str(panel), "sha256": harness.sha(panel), "cases": 75}},
                                   "generation_max_new_tokens": harness.CAP,
                                   "generation_batching": {"batch_size": 8, "kv_cache": "dynamic"}},
                      "cohort": {"max_sequence_tokens": 4096}}
            recipe_path = root / "bound.json"
            recipe_path.write_text(json.dumps(recipe))
            records = {}
            for name, argv in (("bound", []), ("per-case", ["--generation-batch-size", "1"])):
                output = root / name / "baseline-dev75.json"
                admission = root / f"{name}-admission.json"
                admission.write_text(json.dumps({
                    "schema": "sepalith.sft11.full-weight-edit-dev-generation-admission.v1", "status": "admitted",
                    "purpose": "development_generation", "cuda_authorized": True, "final_set_access": False,
                    "bound_recipe_sha256": harness.sha(recipe_path),
                    "checkpoint_manifest_sha256": harness.sha(parent / "campaign-manifest.json"),
                    "panel_sha256": harness.sha(panel), "output": str(output), "deadline_seconds": 3600}))
                log = []
                original = transformers.AutoModelForCausalLM.from_pretrained

                def from_pretrained(path, log=log, original=original, **kwargs):
                    return harness.DenseView(original(str(tiny), torch_dtype=torch.float32, local_files_only=True), log)

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}))
                    stack.enter_context(mock.patch.object(parent_baseline, "load_bound_v2", lambda path: json.loads(Path(path).read_text())))
                    stack.enter_context(mock.patch.object(torch.cuda, "is_available", lambda: True))
                    stack.enter_context(mock.patch.object(torch.cuda, "device_count", lambda: 1))
                    stack.enter_context(mock.patch.object(campaign_tokenizer_contract, "load_pinned_reference_tokenizer", lambda path: "ref"))
                    stack.enter_context(mock.patch.object(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: harness.TextTokenizer()))
                    stack.enter_context(mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained))
                    stack.enter_context(mock.patch.object(trainer_tokenizer_alignment, "restore_trainer_eog_alignment", lambda *a: {}))
                    stack.enter_context(mock.patch.object(trainer_tokenizer_alignment, "embedding_identity", lambda model: {}))
                    stack.enter_context(mock.patch.object(trainer_tokenizer_alignment, "assert_runtime_tokenizer", lambda *a: {}))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    self.assertEqual(parent_baseline.main(["--bound-recipe", str(recipe_path), "--checkpoint", str(parent),
                                                           "--dev-admission", str(admission), "--output", str(output), *argv]), 0)
                result = json.loads(output.read_text())
                records[name] = (result, json.loads(Path(result["cases"]["path"]).read_text())["results"], log)
            result, cases, log = records["bound"]
            self.assertEqual(result["generation_runtime"], {"dtype": "bf16", "cuda_visible_devices": "0",
                                                            **campaign_eval.generation_batching(8)})
            self.assertEqual(sum(call[0] == "generate" for call in log), math.ceil(75 / 8))
            self.assertEqual(milestone_gate.generation_settings(result),
                             milestone_gate.requested_generation_settings(harness.CAP, 8))
            single, single_cases, single_log = records["per-case"]
            self.assertNotIn("generation_runtime", single)
            self.assertEqual(sum(call[0] == "generate" for call in single_log), 75)
            strip = lambda rows: [{k: v for k, v in row.items() if k != "generation_seconds"} for row in rows]  # noqa: E731
            self.assertEqual(strip(cases), strip(single_cases))


class ValidationJobTests(unittest.TestCase):
    """The GPU validation script's comparisons and its night-queue job (no GPU here)."""

    ROOT = Path(__file__).resolve().parents[4]
    C07 = ROOT / "docs/training/plan-20260927/c07"

    def load(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("validate_batched_eval", self.C07 / "validate_batched_eval.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_identity_and_first_divergence(self):
        module = self.load()
        self.assertIsNone(module.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(module.first_divergence([1, 2, 3], [1, 5, 3]), 1)
        self.assertEqual(module.first_divergence([1, 2, 3], [1, 2]), 2)
        report = module.compare({"a": [1, 2], "b": [3, 4, 5], "c": [7]}, {"a": [1, 2], "b": [3, 9, 5], "c": [8]})
        self.assertEqual((report["identical"], report["identity_rate"]), (1, 1 / 3))
        self.assertEqual(report["first_divergence_positions"], [0, 1])

    def test_job_file_runs_the_validation_under_the_night_queue_rules(self):
        from sepalith.ops import night_queue

        job = json.loads((self.C07 / "jobs/C07-batched-eval-validation.json").read_text())
        night_queue.validate_job(dict(job))
        self.assertEqual(job["cwd"], "/home/m0hawk/sandbox/Sepalith")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", job["command"])
        self.assertEqual(job["command"][6], "/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python")
        self.assertFalse(job["resumable"])
        script = job["command"].index("-B") + 1
        self.assertTrue(job["command"][script].endswith("docs/training/plan-20260927/c07/validate_batched_eval.py"))
        args = self.load().parse_args(job["command"][script + 1:])
        self.assertFalse(args.inner or args.no_guard)
        self.assertTrue(str(args.recipe).endswith("c04/bound/sft-v2-lr1e-5.json"))
        self.assertTrue(str(args.output_dir).startswith("/home/m0hawk/.local/state/sepalith/campaign-20260915/"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
