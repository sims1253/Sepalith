"""Card C14 step 2: the GRPO memory probe's CPU paths.

Chunked log-probabilities against the plain log-softmax (values and
gradients), the hidden-state path against the model's own logits on a tiny
Llama, the fit rule, the card's ladder order and skips, TRAIN prompt
selection, and a whole probe run on CPU with one configuration that completes
and one that runs out of memory (an injected OOM is a recorded result, not a
failure).
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sepalith.training.rl import grpo_memory_probe as probe

OPTIMIZER = {"adam_betas": [0.9, 0.999], "adam_eps": 1e-08, "arm": "aurora_mix", "aurora_K": 2, "aurora_beta": 0.5,
             "bf16_update_policy": "stochastic_round", "hidden_lr": 1e-05, "momentum": 0.95, "ns_steps": 5,
             "rms_scale": 0.2, "side_lr": 1e-06, "state_dtype": "float32",
             "stochastic_round_chunk_elements": 1048576, "weight_decay": 0.0}


def tiny_model(dtype=torch.float32):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False, bos_token_id=0,
                         eos_token_id=1, pad_token_id=1)
    return LlamaForCausalLM(config).to(dtype)


def reference_logprobs(hidden, weight, targets, temperature):
    logits = (hidden @ weight.T) / temperature
    picked = torch.log_softmax(logits, dim=-1).gather(1, targets.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(targets >= 0, picked, torch.zeros_like(picked))


def train_row(index, prompt_body, target_body, split="train"):
    prompt = [0, *prompt_body]
    ids = [*prompt, *target_body, 1]
    return {"id": f"row{index:03d}", "split": split, "family": "finish_block", "input_ids": ids,
            "target_start": len(prompt), "prompt_token_count": len(prompt) - 1,
            "target_token_count": len(target_body)}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ResolveDeviceTests(unittest.TestCase):
    # 2026-10-22: every configuration crashed on torch.cuda.set_device("cuda") (no index).
    def test_bare_cuda_gets_index_zero(self):
        self.assertEqual(probe.resolve_device("cuda"), torch.device("cuda", 0))
        self.assertEqual(probe.resolve_device("cuda:1"), torch.device("cuda", 1))
        self.assertEqual(probe.resolve_device("cpu"), torch.device("cpu"))

    def test_all_crashed_is_a_job_failure(self):
        self.assertEqual(probe.probe_exit_code({"attempts": [{"status": "crashed"}, {"status": "crashed"}]}), 1)
        self.assertEqual(probe.probe_exit_code({"attempts": [{"status": "crashed"}, {"status": "oom"}]}), 0)
        self.assertEqual(probe.probe_exit_code({"attempts": []}), 0)


class ChunkedLogProbTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(1)
        self.hidden = torch.randn(11, 6, generator=generator, dtype=torch.float64)
        self.weight = torch.randn(23, 6, generator=generator, dtype=torch.float64)
        self.targets = torch.randint(0, 23, (11,), generator=generator)
        self.targets[[2, 7]] = -100

    def test_values_match_log_softmax_for_uneven_chunks(self):
        expected = reference_logprobs(self.hidden, self.weight, self.targets, 0.7)
        for seq_chunk, vocab_chunk in ((3, 7), (11, 23), (1, 1), (64, 5)):
            got = probe.chunked_token_logprobs(self.hidden, self.weight, self.targets, temperature=0.7,
                                               seq_chunk=seq_chunk, vocab_chunk=vocab_chunk)
            torch.testing.assert_close(got, expected, rtol=0, atol=1e-10)
        self.assertEqual(float(got[2]), 0.0)

    def test_gradients_match_autograd(self):
        upstream = torch.randn(11, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        h1, w1 = self.hidden.clone().requires_grad_(), self.weight.clone().requires_grad_()
        (reference_logprobs(h1, w1, self.targets, 0.7) * upstream).sum().backward()
        h2, w2 = self.hidden.clone().requires_grad_(), self.weight.clone().requires_grad_()
        (probe.chunked_token_logprobs(h2, w2, self.targets, temperature=0.7, seq_chunk=4, vocab_chunk=6)
         * upstream).sum().backward()
        torch.testing.assert_close(h2.grad, h1.grad, rtol=0, atol=1e-10)
        torch.testing.assert_close(w2.grad, w1.grad, rtol=0, atol=1e-10)
        self.assertTrue(torch.all(h2.grad[[2, 7]] == 0))

    def test_gradcheck(self):
        hidden = self.hidden[:5].clone().requires_grad_()
        weight = self.weight[:9].clone().requires_grad_()
        targets = torch.tensor([0, 3, -100, 8, 4])
        self.assertTrue(torch.autograd.gradcheck(
            lambda h, w: probe.chunked_token_logprobs(h, w, targets, temperature=0.9, seq_chunk=2, vocab_chunk=4),
            (hidden, weight)))

    def test_bf16_close_to_fp32(self):
        hidden, weight = self.hidden.float(), self.weight.float()
        expected = reference_logprobs(hidden, weight, self.targets, 1.0)
        got = probe.chunked_token_logprobs(hidden.bfloat16(), weight.bfloat16(), self.targets, seq_chunk=4, vocab_chunk=8)
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, expected, rtol=0, atol=0.15)

    def test_shape_errors(self):
        with self.assertRaises(probe.ProbeError):
            probe.chunked_token_logprobs(self.hidden, self.weight[:, :5], self.targets)


class ModelPathTests(unittest.TestCase):
    rows = [([0, 30, 31, 32, 33], [40, 41, 42]), ([0, 34, 35], [43, 44, 45, 46, 1]), ([0, 36, 37, 38], [47])]

    def test_hidden_state_path_matches_model_logits(self):
        model = tiny_model().eval()
        with torch.no_grad():
            chunked = probe.model_completion_logprobs(model, self.rows, temperature=0.7, seq_chunk=2, vocab_chunk=5)
            full = probe.model_completion_logprobs(model, self.rows, temperature=0.7, seq_chunk=2, vocab_chunk=5,
                                                   impl="full_logits")
        self.assertEqual([len(value) for value in chunked], [3, 5, 1])
        for a, b in zip(chunked, full, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_gradients_match_model_logits_with_checkpointing(self):
        grads = []
        for impl in ("chunked", "full_logits"):
            model = tiny_model()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            for row in self.rows:
                (values,) = probe.model_completion_logprobs(model, [row], temperature=0.7, seq_chunk=2, vocab_chunk=5,
                                                            impl=impl)
                (-values.sum() / 9).backward()
            grads.append({name: p.grad.clone() for name, p in model.named_parameters()})
        self.assertEqual(set(grads[0]), set(grads[1]))
        for name in grads[0]:
            torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-4, atol=1e-6, msg=name)

    def test_group_advantages(self):
        values = probe.group_advantages([1.0, 0.0, 1.0, 0.0, 0.5, 0.5], 2)
        self.assertAlmostEqual(values[0], -values[1])
        self.assertEqual(values[4:], [0.0, 0.0])


def fake_result(config, peaks_mib, *, status="completed", failed=None, total=32607.0, context=500.0, other=4000.0):
    phases = {}
    for name in ["load", "warm_step", *config.phases]:
        if name not in peaks_mib:
            continue
        phases[name] = {"status": "oom" if name == failed else "ok", "peak_reserved_bytes": int(peaks_mib[name] * probe.MIB)}
    return {"config": config.to_mapping(), "status": status, "phases": phases,
            "device": {"total_mib": total, "context_mib": context, "other_used_mib_at_start": other}}


class AssessTests(unittest.TestCase):
    config = probe.ProbeConfig(name="c", candidate_count=8, rows_per_update=32, groups_per_call=1)

    def peaks(self, **override):
        values = {"load": 4800.0, "warm_step": 17000.0, "rollout": 21000.0, "scoring": 18000.0, "backward": 23000.0,
                  "step": 22000.0, "natural_rollout": 20000.0}
        values.update(override)
        return values

    def test_fits_with_margin_at_desktop_budget(self):
        verdict = probe.assess(fake_result(self.config, self.peaks()))
        self.assertTrue(verdict["fits"])
        self.assertEqual(verdict["process_peak_mib"], 23500.0)
        self.assertEqual(verdict["limit_mib"], 32607.0 - 6144 - 1536)
        self.assertEqual(verdict["headroom_at_desktop_budget_mib"], 32607.0 - 6144 - 23500.0)
        self.assertEqual(verdict["headroom_at_observed_desktop_mib"], 32607.0 - 4000 - 23500.0)
        self.assertIsNone(verdict["binding_phase"])
        self.assertFalse(verdict["physical_overflow"])

    def test_over_budget_names_binding_phase(self):
        verdict = probe.assess(fake_result(self.config, self.peaks(rollout=25000.0)))
        self.assertFalse(verdict["fits"])
        self.assertEqual(verdict["binding_phase"], "rollout")

    def test_oom_phase_is_binding(self):
        peaks = {"load": 4800.0, "warm_step": 17000.0, "rollout": 28000.0}
        verdict = probe.assess(fake_result(self.config, peaks, status="oom", failed="rollout"))
        self.assertFalse(verdict["fits"])
        self.assertFalse(verdict["completed"])
        self.assertEqual(verdict["binding_phase"], "rollout")

    def test_cpu_result_is_unmeasured(self):
        result = {"config": self.config.to_mapping(), "status": "completed", "phases": {"load": {"status": "ok"}}}
        verdict = probe.assess(result)
        self.assertFalse(verdict["measured"])
        self.assertIsNone(verdict["fits"])


class LadderTests(unittest.TestCase):
    def test_card_order(self):
        names = [config.name for config in probe.ladder(8)]
        self.assertEqual(names, ["g8-r32-resident", "g8-r16-resident", "g8-r8-resident",
                                 "g8-r32-offload", "g8-r16-offload", "g8-r8-offload"])
        tricks = [config.trick for config in probe.ladder(8)]
        self.assertEqual(tricks[:3], ["baseline", "smaller_rollout_microbatch", "smaller_rollout_microbatch"])
        plan = dict(probe.default_plan())
        self.assertEqual(plan["g8"][-1].name, "g6-r6-offload")
        self.assertEqual(plan["g8"][-1].rows_per_update, 30)
        self.assertEqual([c.rows_per_call for c in plan["g4"]], [32, 16, 8, 4, 32, 16, 8, 4])
        self.assertEqual(plan["informational"][0].logprob_impl, "full_logits")

    def test_config_validation(self):
        with self.assertRaises(probe.ProbeError):
            probe.ProbeConfig(name="x", candidate_count=8, rows_per_update=30, groups_per_call=1)
        with self.assertRaises(probe.ProbeError):
            probe.ProbeConfig(name="x", candidate_count=8, rows_per_update=32, groups_per_call=3)
        config = probe.ProbeConfig(name="x", candidate_count=4, rows_per_update=32, groups_per_call=2, phases=("backward",))
        self.assertEqual(probe.ProbeConfig.from_mapping(json.loads(json.dumps(config.to_mapping()))), config)

    def executor(self, outcomes):
        """outcomes: config name -> peaks override (or 'oom:<phase>')."""
        ran = []

        def execute(config):
            ran.append(config.name)
            spec = outcomes.get(config.name, {})
            peaks = AssessTests.peaks(AssessTests(), **(spec if isinstance(spec, dict) else {}))
            if isinstance(spec, str):
                phase = spec.split(":")[1]
                order = ["load", "warm_step", *config.phases]
                peaks = {name: peaks[name] for name in order[:order.index(phase) + 1]}
                peaks[phase] = 30000.0
                result = fake_result(config, peaks, status="oom", failed=phase)
            else:
                result = fake_result(config, peaks)
            return {"name": config.name, "config": config.to_mapping(), "status": result["status"],
                    "assessment": probe.assess(result)}
        return execute, ran

    def test_stops_at_first_fit_and_skips_dominated(self):
        execute, ran = self.executor({"g8-r32-resident": "oom:rollout", "g8-r16-resident": {"rollout": 25500.0}})
        attempts = []
        plan = dict(probe.default_plan())
        g8 = probe.run_ladder(plan["g8"], execute, attempts, can_start=lambda: None)
        self.assertEqual(g8["first_fit"], "g8-r8-resident")
        self.assertEqual(ran, ["g8-r32-resident", "g8-r16-resident", "g8-r8-resident"])
        g4 = probe.run_ladder(plan["g4"], execute, attempts, can_start=lambda: None)
        self.assertEqual(g4["first_fit"], "g4-r8-resident")
        statuses = {a["name"]: a["status"] for a in attempts}
        self.assertEqual(statuses["g4-r32-resident"], "skipped")
        self.assertEqual(statuses["g4-r16-resident"], "skipped")
        self.assertIn("dominated by g8-r16-resident", next(a["reason"] for a in attempts if a["name"] == "g4-r16-resident"))
        self.assertEqual(ran[-1], "g4-r8-resident")

    def test_backward_bound_failure_skips_rollout_tricks(self):
        execute, ran = self.executor({"g8-r32-resident": "oom:backward"})
        attempts = []
        outcome = probe.run_ladder(dict(probe.default_plan())["g8"], execute, attempts, can_start=lambda: None)
        self.assertIsNone(outcome["first_fit"])
        self.assertEqual(ran, ["g8-r32-resident"])
        self.assertEqual(len(attempts), 7)
        self.assertTrue(all(a["status"] == "skipped" and "phase backward" in a["reason"] for a in attempts[1:]))

    def test_offload_after_resident_rollout_failures(self):
        execute, ran = self.executor({name: "oom:rollout" for name in
                                      ("g8-r32-resident", "g8-r16-resident", "g8-r8-resident")})
        attempts = []
        outcome = probe.run_ladder(dict(probe.default_plan())["g8"], execute, attempts, can_start=lambda: None)
        self.assertEqual(outcome["first_fit"], "g8-r32-offload")
        self.assertEqual(ran[-1], "g8-r32-offload")

    def test_time_stop_marks_not_run(self):
        execute, ran = self.executor({})
        attempts = []
        outcome = probe.run_ladder(probe.ladder(4), execute, attempts, can_start=lambda: "not started: deadline")
        self.assertEqual(ran, [])
        self.assertEqual(outcome["stopped"], "not started: deadline")
        self.assertTrue(all(a["status"] == "not_run" for a in attempts))


class PromptSelectionTests(unittest.TestCase):
    def test_longest_distinct_train_prompts(self):
        rows = [train_row(0, [30, 31], [40]), train_row(1, [30, 31, 32, 33], [41]),
                train_row(2, [30, 31, 32, 33], [42, 43]),  # same prompt as row 1
                train_row(3, [34, 35, 36], [44])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_rows(path, rows)
            prompts, source = probe.select_longest_prompts(path, None, 2)
            self.assertEqual([p["id"] for p in prompts], ["row001", "row003"])
            self.assertEqual(prompts[0]["prompt_tokens"], [0, 30, 31, 32, 33])
            self.assertEqual(source["distinct_prompts"], 3)
            with self.assertRaises(probe.ProbeError):
                probe.select_longest_prompts(path, "0" * 64, 2)
            with self.assertRaises(probe.ProbeError):
                probe.select_longest_prompts(path, None, 4)

    def test_refuses_non_train_and_overlong(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_rows(path, [train_row(0, [30], [40], split="dev")])
            with self.assertRaises(probe.ProbeError):
                probe.select_longest_prompts(path, None, 1)
            write_rows(path, [train_row(0, [30] * 3072, [40])])
            with self.assertRaises(ValueError):
                probe.select_longest_prompts(path, None, 1)


class EndToEndCPUTests(unittest.TestCase):
    def test_probe_run_records_completion_and_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            tiny_model().save_pretrained(model_dir)
            rows = root / "rows.jsonl"
            write_rows(rows, [train_row(i, [22 + (i + k) % 40 for k in range(6 + i)], [40 + i, 41]) for i in range(6)])
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps({"runtime": {"optimizer": OPTIMIZER}}))
            phases = ("rollout", "scoring", "backward", "step", "natural_rollout")
            ok = probe.ProbeConfig(name="tiny-ok", candidate_count=2, rows_per_update=4, groups_per_call=1,
                                   offload_optimizer=True, phases=phases, seq_chunk=3, vocab_chunk=10)
            oom = probe.ProbeConfig(name="tiny-oom", candidate_count=2, rows_per_update=4, groups_per_call=2,
                                    phases=phases, inject_oom_phase="backward")
            settings = probe.ProbeSettings(
                run_dir=root / "run", model_path=model_dir, model_sha256=None, rows_path=rows, rows_sha256=None,
                sft_recipe_path=recipe, device="cpu", dtype="float32", completion_tokens=5, eog_ids=(1,),
                guard=False, python=sys.executable, docs_dir=root / "docs", poll_seconds=0.2,
                plan=[("first", [ok]), ("second", [oom])], job_result=root / "job-result.json")
            receipt = json.loads(json.dumps(probe.Probe(settings).run()))
            self.assertEqual(receipt["status"], "complete")
            first, second = receipt["attempts"]
            self.assertEqual(first["status"], "completed", first)
            self.assertEqual(second["status"], "oom", second)
            self.assertEqual(second["failed_phase"], "backward")
            self.assertEqual(second["exit_code"], 0)
            result = json.loads(Path(first["result"]).read_text())
            self.assertEqual(result["phase_order"], ["load", "warm_step", *phases])
            rollout = result["phases"]["rollout"]
            self.assertTrue(rollout["forced_to_cap"])
            self.assertEqual(rollout["rows"], 4)
            self.assertEqual(rollout["generated_tokens"], 20)
            self.assertGreater(rollout["offloaded_bytes"], 0)
            self.assertEqual(result["phases"]["backward"]["micro_batches"], 4)
            self.assertTrue(math.isfinite(result["phases"]["backward"]["grad_norm"]))
            self.assertGreater(result["phases"]["warm_step"]["optimizer_state_bytes"], 0)
            self.assertIn("update_seconds_natural_rollout", first["timing"])
            self.assertFalse(first["assessment"]["measured"])  # CPU: no GPU memory figures
            inputs = json.loads((root / "run/inputs.json").read_text())
            self.assertEqual(len(inputs["prompts"]), 2)
            self.assertNotIn("prompt_tokens", json.dumps(receipt["inputs"]["prompts"]))
            summary = (root / "run/summary.md").read_text()
            self.assertIn("tiny-oom", summary)
            self.assertTrue((root / "docs/C14-memory-probe.json").exists())
            self.assertTrue((root / "docs/C14-memory-probe.md").exists())
            self.assertIn("summary", json.loads((root / "job-result.json").read_text()))


if __name__ == "__main__":
    unittest.main()
