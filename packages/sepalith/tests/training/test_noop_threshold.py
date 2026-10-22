"""Decode-time no-op threshold job (docs/training/plan-20260927/decode-time/noop_threshold.py).

CPU only. The job file under the night-queue rules and its command line, the threshold
curves and the TRAIN calibration on synthetic records, crash-safe record files, attempt
resumption and the outer exit codes, the first-step suppression of the no-op token, and
the GPU path (``inner``) end to end on a tiny Llama with ``torch.cuda`` mocked: every CUDA
call must name device 0 and ``set_device``/``synchronize``/``mem_get_info`` must never be
called (2026-10-22: a job crashed on ``torch.cuda.set_device("cuda")``). The real model
on the CPU is exercised by the script's ``--dry-run --cpu-smoke`` (receipt in
``docs/training/plan-20260927/receipts/DT-noop-threshold-dry-run.json``).
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "docs/training/plan-20260927/decode-time"
JOB = HERE / "jobs/DT-noop-threshold-arm-a-536.json"
NOOP = 80


def load():
    spec = importlib.util.spec_from_file_location("noop_threshold", HERE / "noop_threshold.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(case_id, expected_noop, p_noop, *, argmax=None, exact=False, suggestion=True, noop=False, valid=True,
           cap=False, family="f", batch=0):
    return {"id": case_id, "family": family, "expected_noop": expected_noop, "batch": batch,
            "batch_forward_seconds": 1.0, "batch_generation_seconds": 2.0,
            "first": {"p_noop": p_noop, "margin_logit": p_noop - 0.5, "argmax_id": argmax if argmax is not None else
                      (NOOP if p_noop > 0.5 else 7), "noop_chain_greedy": True},
            "forced_edit": {"protocol_valid": valid, "exact_region": exact, "predicted_noop": noop, "suggestion": suggestion,
                            "cap_hit": cap, "first_token": argmax if argmax not in (None, NOOP) else 7}}


class JobFileTests(unittest.TestCase):
    def test_job_file_runs_under_the_night_queue_rules_and_parses(self):
        from sepalith.ops import night_queue

        module = load()
        job = json.loads(JOB.read_text())
        night_queue.validate_job(dict(job))
        self.assertEqual(job["id"], "DT-noop-threshold-arm-a-536")
        self.assertEqual(job["cwd"], "/home/m0hawk/sandbox/Sepalith")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", job["command"])
        self.assertEqual(job["command"][6], "/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python")
        self.assertTrue(job["resumable"])
        script = job["command"].index("-B") + 1
        self.assertEqual(job["command"][script],
                         "/home/m0hawk/sandbox/Sepalith/docs/training/plan-20260927/decode-time/noop_threshold.py")
        args = module.parse_args(job["command"][script + 1:])
        self.assertFalse(args.inner or args.no_guard or args.dry_run or args.fresh)
        self.assertEqual(args.panels, ["dev75", "dev250", "train1000"])
        self.assertTrue(str(args.output_dir).startswith("/home/m0hawk/.local/state/sepalith/campaign-20260915/decode-time/"))
        self.assertTrue(str(args.dry_run_output).endswith("receipts/DT-noop-threshold-dry-run.json"))
        self.assertTrue(str(args.summary_dir).endswith("plan-20260927/decode-time/results"))
        self.assertEqual(args.train_subset, module.DEFAULT_SUBSET)
        # The job stays under 2.5 h: run budget, guard and estimate.
        self.assertLessEqual(args.budget_minutes, 140)
        self.assertLessEqual(args.guard_seconds, 9000)
        self.assertLessEqual(job["estimated_minutes"], args.budget_minutes)

    def test_cli_rules(self):
        module = load()
        with self.assertRaises(SystemExit):
            module.parse_args([])
        with self.assertRaises(SystemExit):
            module.parse_args(["--output-dir", "/tmp/x", "--cpu-smoke"])
        with self.assertRaises(SystemExit):
            module.parse_args(["--output-dir", "/tmp/x", "--panels", "dev75,final"])
        with self.assertRaises(SystemExit):
            module.parse_args(["--output-dir", "/tmp/x", "--budget-minutes", "200"])
        self.assertEqual(module.parse_args(["--output-dir", "/tmp/x", "--panels", "train1000,dev75"]).panels,
                         ["dev75", "train1000"])

    def test_dry_run_receipt_passed_and_pins_the_subset(self):
        module = load()
        dry = json.loads(module.DEFAULT_DRY_RUN.read_text())
        self.assertEqual(dry["schema"], module.DRY_RUN_SCHEMA)
        self.assertEqual(dry["status"], "pass")
        self.assertFalse(dry["final_set_access"])
        self.assertEqual(dry["checkpoint"]["weights_sha256"], module.WEIGHTS_SHA256)
        self.assertEqual(dry["train_subset"]["path"], str(module.DEFAULT_SUBSET))
        self.assertEqual(dry["train_subset"]["families"]["no_op"], 200)
        self.assertEqual(dry["noop_target_ids"], list(module.NOOP_TARGET))
        self.assertEqual(dry["cpu_smoke"]["status"], "complete")
        self.assertEqual(set(dry["cpu_smoke"]["resume_new_cases"].values()), {0})


class CurveTests(unittest.TestCase):
    def setUp(self):
        self.m = load()
        self.records = [
            record("n1", True, 0.95),                       # confident no-op case
            record("n2", True, 0.30, suggestion=True),      # unsure no-op case; forced edit is a false positive
            record("n3", True, 0.10, noop=True, suggestion=False),  # forced edit echoes the region -> parsed no-op
            record("e1", False, 0.60, exact=True),          # greedy said no-op, forced edit is exact
            record("e2", False, 0.05, exact=True),
            record("e3", False, 0.02, exact=False, cap=True, valid=False, suggestion=False),
        ]

    def test_threshold_semantics(self):
        always = self.m.metrics(self.records, self.m.decide_tau(None))
        self.assertEqual((always["predicted_noop"], always["noop_correct"], always["nofp"], always["edit_exact"]), (1, 1, 2, 2))
        self.assertEqual((always["cap"], always["valid"]), (1, 5))
        silent = self.m.metrics(self.records, self.m.decide_tau(0.0))
        self.assertEqual((silent["predicted_noop"], silent["noop_correct"], silent["edit_exact"], silent["missed_edit"]), (6, 3, 0, 3))
        half = self.m.metrics(self.records, self.m.decide_tau(0.5))
        self.assertEqual((half["noop_correct"], half["nofp"], half["edit_exact"], half["missed_edit"], half["score"]), (2, 1, 1, 1, 3))
        self.assertEqual(half["noop_recall"], round(2 / 3, 4))
        greedy = self.m.metrics(self.records, self.m.decide_greedy)
        self.assertEqual(greedy, half)

    def test_calibration_prefers_more_proposals_on_ties(self):
        curve = self.m.curve(self.records)
        self.assertEqual(len(curve), len(self.m.TAU_GRID))
        scores = {row["tau"]: row["score"] for row in curve}
        best = max(scores.values())
        tau = self.m.calibrate(curve)
        self.assertEqual(scores[tau], best)
        tied = [t for t in self.m.TAU_GRID if scores[t] == best]
        self.assertEqual(tau, tied[-1])
        # 0.7 to 0.95 score 4 (e1's forced edit is exact, n1 stays silent); the highest of them wins the tie.
        self.assertEqual(tau, 0.95)
        self.assertEqual(scores[None], 3)

    def test_auc_and_greedy_check(self):
        self.assertEqual(self.m.roc_auc(self.records), round(7 / 9, 4))
        gate = {r["id"]: {"protocol_valid": True, "exact_region": r["id"] in ("n1", "e2"), "predicted_noop": r["id"] in ("n1", "e1")}
                for r in self.records}
        check = self.m.greedy_check(self.records, gate)
        self.assertEqual(check["gate_case_agreement"]["of"], 6)
        self.assertIn("e3", check["gate_case_agreement"]["differ"])  # invalid in the records, valid in the fake gate

    def test_summary_and_markdown_from_partial_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "20261022T120000"
            (attempt / "records").mkdir(parents=True)
            self.m.write_json(attempt / "receipt.json", {"status": "stopped", "panels": {
                "dev75": {"cases": 6}, "train1000": {"cases": 10}}})
            for name in ("dev75", "train1000"):
                self.m.append_records(attempt / "records" / f"{name}.jsonl", self.records)
            summary = self.m.summarize(attempt, Path(tmp) / "out", ["dev75", "train1000"])
            self.assertTrue(summary["panels"]["dev75"]["complete"])
            self.assertFalse(summary["panels"]["train1000"]["complete"])
            self.assertIn("warning", summary["calibration"])
            self.assertIn("dev75_at_tau", summary["calibration"])
            text = (Path(tmp) / "out/noop-threshold-20261022T120000.md").read_text()
            self.assertIn("| always propose |", text)
            self.assertIn("provisional", text)
            json.loads((Path(tmp) / "out/noop-threshold-20261022T120000.json").read_text())


class RecordFileTests(unittest.TestCase):
    def test_torn_trailing_line_is_dropped(self):
        m = load()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            m.append_records(path, [{"id": "a"}, {"id": "b"}])
            with path.open("a") as stream:
                stream.write('{"id": "c", "fir')
            self.assertEqual([r["id"] for r in m.read_records(path)], ["a", "b"])
            m.append_records(path, [{"id": "c"}])
            self.assertEqual([r["id"] for r in m.read_records(path)], ["a", "b", "c"])

    def test_latest_unfinished_attempt_is_resumed(self):
        m = load()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first, resumed = m.find_attempt(out)
            self.assertFalse(resumed)
            again, resumed = m.find_attempt(out)
            self.assertEqual((again, resumed), (first, True))
            m.write_json(first / "receipt.json", {"status": "complete"})
            with mock.patch.object(m.time, "strftime", return_value="29991231T235959"):
                fresh, resumed = m.find_attempt(out)
            self.assertNotEqual(fresh, first)
            self.assertFalse(resumed)


class OuterExitCodeTests(unittest.TestCase):
    """The outer process maps the inner receipt to 0 / 75 / 1 and writes the job result."""

    def run_outer(self, receipt, records):
        m = load()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"

            class FakeProcess:
                def __init__(self, launch, **_):
                    inner = m.parse_args(launch[3:])  # the inner command line the outer builds must parse
                    assert inner.inner and inner.panels == ["dev75"] and inner.budget_minutes == 140
                    attempt = Path(launch[launch.index("--attempt-dir") + 1])
                    (attempt / "records").mkdir(parents=True, exist_ok=True)
                    m.write_json(attempt / "receipt.json", receipt)
                    for name, count in records.items():
                        m.append_records(attempt / "records" / f"{name}.jsonl", [{"id": str(i)} for i in range(count)])

                def wait(self):
                    return 0 if receipt["status"] == "complete" else 1

                def terminate(self):
                    pass

            result = Path(tmp) / "result.json"
            args = m.parse_args(["--output-dir", str(out), "--no-guard", "--panels", "dev75",
                                 "--summary-dir", str(Path(tmp) / "s")])
            with mock.patch.object(m.subprocess, "Popen", FakeProcess), \
                    mock.patch.dict(os.environ, {"SEPALITH_JOB_RESULT": str(result)}):
                code = m.outer(args)
            return code, json.loads(result.read_text())

    def test_complete_stopped_and_nothing_measured(self):
        code, result = self.run_outer({"status": "complete", "panels": {"dev75": {"cases": 3}}}, {"dev75": 3})
        self.assertEqual(code, 0)
        self.assertIsNone(result["needs_user"])
        code, _ = self.run_outer({"status": "stopped", "panels": {"dev75": {"cases": 3}}}, {"dev75": 2})
        self.assertEqual(code, 75)
        code, result = self.run_outer({"status": "failed", "panels": {"dev75": {"cases": 3}}}, {})
        self.assertEqual(code, 1)
        self.assertIn("0 cases recorded", result["needs_user"])
        code, _ = self.run_outer({"status": "failed", "panels": {"dev75": {"cases": 3}}}, {"dev75": 1})
        self.assertEqual(code, 1)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=130560, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, bos_token_id=0, eos_token_id=[1, 130073],
                         pad_token_id=1, tie_word_embeddings=False)
    model = LlamaForCausalLM(config).eval()

    class Head(torch.nn.Module):
        """Token 80 (no-op) is always the argmax, token 81 second: the suppression is visible."""

        def forward(self, hidden):
            logits = torch.zeros(*hidden.shape[:-1], 130560, dtype=hidden.dtype)
            logits[..., NOOP] = 9.0
            logits[..., 81] = 5.0
            return logits + 1e-3 * hidden.sum(-1, keepdim=True)

    model.lm_head = Head()
    return model


class SuppressionAndMarginTests(unittest.TestCase):
    def test_no_op_token_is_banned_only_at_the_first_step(self):
        from sepalith.training.eval import campaign_eval

        m = load()
        model = tiny_model()
        with torch.no_grad():
            plain = campaign_eval.batched_greedy_generate(model, [[0, 5, 6], [0, 7]], cap=4, stop_ids=(1, 130073))
            banned = campaign_eval.batched_greedy_generate(m.SuppressNoopFirst(model), [[0, 5, 6], [0, 7]], cap=4,
                                                           stop_ids=(1, 130073))
        self.assertEqual(plain, [[NOOP] * 4, [NOOP] * 4])
        self.assertEqual(banned, [[81, NOOP, NOOP, NOOP], [81, NOOP, NOOP, NOOP]])

    def test_first_token_record(self):
        m = load()
        model = tiny_model()
        with torch.no_grad():
            first = m.first_token_record(model, [0, 5, 6, 7], torch.device("cpu"))
            logits = model(input_ids=torch.tensor([[0, 5, 6, 7]])).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        self.assertAlmostEqual(first["lp_noop"], lp[NOOP].item(), places=4)
        self.assertEqual(first["argmax_id"], NOOP)
        self.assertEqual(first["best_other_id"], 81)
        self.assertAlmostEqual(first["p_noop"] + torch.exp(torch.tensor(first["lse_other"])).item(), 1.0, places=3)
        self.assertAlmostEqual(first["margin_max"], 4.0, places=2)
        self.assertEqual(len(first["noop_chain_lps"]), len(m.NOOP_TARGET))
        self.assertFalse(first["noop_chain_greedy"])  # token 80 is the argmax everywhere, not the no-op string

    def test_gpu_device_has_an_explicit_index(self):
        m = load()
        self.assertEqual(m.gpu_device(), torch.device("cuda", 0))
        self.assertEqual(str(m.gpu_device()), "cuda:0")


def strict_index(name, value):
    def call(*args, **kwargs):
        if args != (0,) or kwargs:
            raise AssertionError(f"torch.cuda.{name} must be called with the explicit device index 0, got {args} {kwargs}")
        return value
    return mock.Mock(side_effect=call)


def forbidden(name):
    return mock.Mock(side_effect=AssertionError(f"torch.cuda.{name} must not be called"))


@unittest.skipUnless(Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586").is_dir()
                     and load().DEFAULT_SUBSET.is_file(), "needs the local panels, tokenizer and TRAIN subset")
class MockedCudaInnerTests(unittest.TestCase):
    """``inner`` end to end with torch.cuda mocked; only the tensors' device is the CPU."""

    def setUp(self):
        self.m = load()
        tiny_model()  # import the Transformers model code before torch.cuda is mocked
        self.tmp = tempfile.TemporaryDirectory()
        self.attempt = Path(self.tmp.name) / "attempt"
        self.cuda = {
            "is_available": mock.Mock(return_value=True), "device_count": mock.Mock(return_value=1),
            "get_device_name": strict_index("get_device_name", "Mock GPU"),
            "max_memory_allocated": strict_index("max_memory_allocated", 2**30),
            "max_memory_reserved": strict_index("max_memory_reserved", 2**31),
            "reset_peak_memory_stats": strict_index("reset_peak_memory_stats", None),
            "empty_cache": mock.Mock(return_value=None), "set_device": forbidden("set_device"),
            "synchronize": forbidden("synchronize"), "mem_get_info": forbidden("mem_get_info"),
        }
        recipe = self.m.load_recipe(self.m.DEFAULT_RECIPE)
        self.recipe = {**recipe, "cap": 3}
        self.loads = []

        def fake_load(parent, reference, device_name):
            self.loads.append(device_name)
            from transformers import AutoTokenizer

            return tiny_model(), AutoTokenizer.from_pretrained(str(parent), local_files_only=True), {"tokenizer_contract": "mock"}

        real_dev, real_train = self.m.dev_cases, self.m.train_cases
        self.patches = [
            mock.patch.multiple(torch.cuda, **self.cuda),
            mock.patch.object(self.m, "load_recipe", lambda path: self.recipe),
            mock.patch.object(self.m, "load_model_and_tokenizer", fake_load),
            mock.patch.object(self.m, "gpu_device", lambda: torch.device("cpu")),
            mock.patch.object(self.m, "verify_weights", lambda dry: {"method": "mock"}),
            mock.patch.object(self.m, "dev_cases", lambda *a: real_dev(*a)[:3]),
            mock.patch.object(self.m, "train_cases", lambda *a: real_train(*a)[-3:]),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_inner(self, env=None):
        argv = ["--inner", "--attempt-dir", str(self.attempt), "--output-dir", str(self.attempt.parent),
                "--summary-dir", str(Path(self.tmp.name) / "summary")]
        with mock.patch.dict(os.environ, env or {}, clear=False):
            return self.m.main(argv)

    def test_inner_measures_every_panel_with_explicit_cuda_indices(self):
        self.assertEqual(self.run_inner({"SEPALITH_DEADLINE_EPOCH": ""}), 0)
        self.assertEqual(self.loads, ["cpu"])  # str(gpu_device()); "cuda:0" on the GPU
        receipt = json.loads((self.attempt / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["runs"][0]["cuda"], {"name": "Mock GPU", "peak_allocated_mib": 1024, "peak_reserved_mib": 2048})
        self.cuda["reset_peak_memory_stats"].assert_called_with(0)
        self.cuda["empty_cache"].assert_called()
        for name in ("dev75", "dev250", "train1000"):
            rows = self.m.read_records(self.attempt / "records" / f"{name}.jsonl")
            self.assertEqual(len(rows), 3)
            for r in rows:
                self.assertEqual(r["first"]["argmax_id"], NOOP)
                self.assertEqual(r["forced_edit"]["first_token"], 81)
                self.assertTrue(r["forced_edit"]["cap_hit"])
        summary = json.loads((self.attempt / "summary.json").read_text())
        self.assertIn("recommended_tau", summary["calibration"])
        self.assertIn("gate_case_agreement", summary["panels"]["dev75"]["greedy"])
        # A second run resumes and measures nothing new.
        self.assertEqual(self.run_inner(), 0)
        self.assertEqual(len(self.m.read_records(self.attempt / "records" / "dev75.jsonl")), 3)

    def test_deadline_stops_at_a_batch_boundary_and_the_next_run_resumes(self):
        self.assertEqual(self.run_inner({"SEPALITH_DEADLINE_EPOCH": "1"}), 75)
        receipt = json.loads((self.attempt / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "stopped")
        self.assertEqual(receipt["runs"][0]["stop_reason"], "window_deadline")
        self.assertEqual(self.m.read_records(self.attempt / "records" / "dev75.jsonl"), [])
        self.assertEqual(self.run_inner({"SEPALITH_DEADLINE_EPOCH": ""}), 0)
        self.assertEqual(json.loads((self.attempt / "receipt.json").read_text())["status"], "complete")

    def test_a_crash_is_recorded_and_raised(self):
        with mock.patch.object(self.m, "first_token_record", side_effect=RuntimeError("boom")), self.assertRaises(RuntimeError):
            self.run_inner({"SEPALITH_DEADLINE_EPOCH": ""})
        receipt = json.loads((self.attempt / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("boom", receipt["runs"][0]["error"])
        self.assertTrue((self.attempt / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
