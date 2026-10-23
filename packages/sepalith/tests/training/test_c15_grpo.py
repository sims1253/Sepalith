"""Card C15: the GRPO trainer, loss, CUDA discipline and night driver on CPU (no GPU).

- The DAPO loss: token normalisation equal to TRL 0.24 ``loss_type="dapo"``,
  asymmetric clipping and its counts, truncated completions masked.
- A tiny Llama with the real 130,560-token vocabulary: chunked completion
  log-probabilities against the model's logits, rollout trimming, and whole
  trainer segments (dynamic sampling, refill, warmup, on- and off-policy
  updates, telemetry) including a save at a rollout-buffer boundary and a
  resume that reproduces the uninterrupted run bit for bit.
- ``DeviceOps`` against a fake ``torch.cuda``: explicit ``cuda:0``, the context
  before any memory statistic, and a scan of the package for raw CUDA calls.
- The night driver with a fake launcher: R6 decisions at DEV75 and DEV250
  evaluations, continue records, overrides, rollback, deadline deferral,
  trainer exit codes, and the smoke receipt.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sepalith.training.eval import campaign_eval
from sepalith.training.grpo import night, recipe as recipe_module, rules, state, trainer
from sepalith.training.grpo.loss import ClipStats, row_surrogate, update_token_count
from sepalith.training.offline import rules as offline_rules
from sepalith.training.sft import night_arm

REPO = Path(__file__).resolve().parents[4]
PILOT = REPO / "docs/training/plan-20260927/c15/bound/grpo-pilot-1-arm-a-536.json"
SMOKE = REPO / "docs/training/plan-20260927/c15/bound/grpo-smoke-arm-a-536.json"
VOCAB = 130560
BATCH = campaign_eval.generation_batching(32, "dynamic")


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=256, tie_word_embeddings=False, bos_token_id=0,
                         eos_token_id=1, pad_token_id=1)
    return LlamaForCausalLM(config)


def plain(value):
    return json.loads(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Loss ---------------------------------------------------------------------------------------------

class LossTests(unittest.TestCase):
    def test_token_normalisation_matches_trl_dapo_with_truncation_masked(self):
        generator = torch.Generator().manual_seed(0)
        lengths, truncated = [3, 5, 2], [False, True, False]
        advantages = [0.7, -0.4, -0.3]
        current = [torch.randn(n, generator=generator, dtype=torch.float64).requires_grad_() for n in lengths]
        old = [value.detach() + 0.3 * torch.randn(value.shape, generator=generator, dtype=torch.float64) for value in current]
        # TRL 0.24: padded (B, T) tensors, completion mask zeroed for truncated rows, sum / num_items_in_batch.
        width = max(lengths)
        logp = torch.zeros(3, width, dtype=torch.float64)
        old_pad = torch.zeros(3, width, dtype=torch.float64)
        mask = torch.zeros(3, width, dtype=torch.float64)
        for i, n in enumerate(lengths):
            logp[i, :n], old_pad[i, :n] = current[i], old[i]
            mask[i, :n] = 0.0 if truncated[i] else 1.0
        coef1 = torch.exp(logp - old_pad)
        coef2 = torch.clamp(coef1, 0.8, 1.28)
        adv = torch.tensor(advantages, dtype=torch.float64)[:, None]
        trl = (-torch.minimum(coef1 * adv, coef2 * adv) * mask).sum() / mask.sum()
        trl_grads = torch.autograd.grad(trl, current, allow_unused=True)
        tokens = update_token_count(lengths, truncated)
        self.assertEqual(tokens, 5)
        ours = sum(row_surrogate(current[i], old[i], advantages[i])[0] / tokens for i in range(3) if not truncated[i])
        torch.testing.assert_close(ours, trl)
        our_grads = torch.autograd.grad(ours, [current[0], current[2]])
        torch.testing.assert_close(our_grads[0], trl_grads[0])
        torch.testing.assert_close(our_grads[1], trl_grads[2])
        self.assertTrue(trl_grads[1] is None or bool(torch.all(trl_grads[1] == 0)))

    def test_asymmetric_clipping_and_counts(self):
        old = torch.zeros(4, dtype=torch.float64)
        ratios = torch.tensor([1.5, 1.1, 0.7, 0.95], dtype=torch.float64)
        logp = torch.log(ratios).requires_grad_()
        loss, stats = row_surrogate(logp, old, 1.0)
        loss.backward()
        self.assertEqual((stats.tokens, stats.clipped_high, stats.clipped_low), (4, 1, 0))
        self.assertEqual(float(logp.grad[0]), 0.0)  # above 1 + 0.28 with A > 0: clipped, no gradient
        self.assertNotEqual(float(logp.grad[2]), 0.0)  # below 1 - 0.2 with A > 0: min keeps the unclipped term
        logp2 = torch.log(ratios).requires_grad_()
        loss2, stats2 = row_surrogate(logp2, old, -1.0)
        loss2.backward()
        self.assertEqual((stats2.clipped_high, stats2.clipped_low), (0, 1))
        self.assertEqual(float(logp2.grad[2]), 0.0)
        self.assertNotEqual(float(logp2.grad[0]), 0.0)
        torch.testing.assert_close(loss, -(torch.tensor([1.28, 1.1, 0.7, 0.95], dtype=torch.float64)).sum())

    def test_on_policy_ratio_is_exactly_one(self):
        logp = torch.randn(6, dtype=torch.float32, requires_grad=True)
        _, stats = row_surrogate(logp, logp.detach(), 0.5)
        self.assertEqual((stats.ratio_max, stats.ratio_min), (1.0, 1.0))
        merged = ClipStats()
        merged.add(stats)
        self.assertEqual(merged.summary()["clip_fraction"], 0.0)


# Tiny model paths -----------------------------------------------------------------------------------

class TinyModelTests(unittest.TestCase):
    def test_chunked_completion_logprobs_match_model_logits(self):
        model = tiny_model().eval()
        row = trainer.Row("p", (0, 500, 600, 700), (800, 130000, 1), "eos")
        with torch.no_grad():
            got = trainer.row_logprobs(model, row, seq_chunk=2, vocab_chunk=40000)
            logits = model(input_ids=torch.tensor([list(row.prompt_ids + row.completion)])).logits[0].float()
        expected = torch.log_softmax(logits[3:6], dim=-1).gather(1, torch.tensor(row.completion)[:, None]).squeeze(1)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    def test_rollout_with_real_generate_keeps_prompts_and_trims(self):
        model = tiny_model()
        model.generation_config.top_p = 0.95  # as in the arm A checkpoint; the explicit kwargs must win
        recipe = json.loads(PILOT.read_text())
        kwargs = {**trainer.generation_kwargs(recipe["runtime"], recipe["pre_registered"]), "max_new_tokens": 3}
        self.assertEqual((kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]), (1.0, 1.0, 0))
        self.assertEqual(kwargs["eos_token_id"], [1, 130073])
        prompts = [("a", (0, 300, 301)), ("b", (0, 400, 401, 402, 403))]
        groups = trainer.rollout(model, prompts, group_size=2, rows_per_call=2, kwargs=kwargs, device=torch.device("cpu"))
        self.assertEqual([[row.prompt_id for row in group] for group in groups], [["a", "a"], ["b", "b"]])
        for group, (_, prompt) in zip(groups, prompts, strict=True):
            for row in group:
                self.assertEqual(row.prompt_ids, prompt)
                self.assertIn(row.terminal_reason, ("eos", "length"))
                self.assertLessEqual(len(row.completion), 3)
        self.assertTrue(model.training)

    def test_rollout_trims_at_the_first_end_of_generation(self):
        def generate(input_ids, attention_mask, **kwargs):
            tails = torch.tensor([[50, 51, 1, 1], [60, 130073, 1, 1], [70, 71, 72, 73], [80, 1, 1, 1]])
            return torch.cat([input_ids, tails], dim=1)

        model = tiny_model()
        kwargs = {"max_new_tokens": 4}
        groups = trainer.rollout(model, [("a", (0, 9)), ("b", (0, 8, 7))], group_size=2, rows_per_call=4, kwargs=kwargs,
                                 device=torch.device("cpu"), generate=generate)
        rows = [row for group in groups for row in group]
        self.assertEqual([row.completion for row in rows], [(50, 51, 1), (60, 130073), (70, 71, 72, 73), (80, 1)])
        self.assertEqual([row.truncated for row in rows], [False, False, True, False])


# Trainer segments ------------------------------------------------------------------------------------

ZERO = {"p00", "p01", "p02"}  # prompts whose groups always have equal rewards


def fake_generate(input_ids, attention_mask, **kwargs):
    """Random completions from the global RNG: EOS at a random position, or none (truncated)."""
    cap = kwargs["max_new_tokens"]
    rows = []
    for _ in range(input_ids.shape[0]):
        length = int(torch.randint(1, cap + 1, (1,)))
        body = torch.randint(22, 129000, (cap,))
        if length < cap or bool(torch.rand(()) < 0.5):
            body[length - 1] = 1
            body[length:] = 1
        rows.append(body)
    return torch.cat([input_ids, torch.stack(rows)], dim=1)


def fake_reward(ids, completions):
    out = []
    for identifier, completion in zip(ids, completions, strict=True):
        if completion[-1] != 1:
            reward, exact = -0.5, False
        elif identifier in ZERO:
            reward, exact = 0.5, False
        else:
            exact = sum(completion) % 3 == 0
            reward = 1.0 if exact else 0.0
        out.append({"reward": reward, "exact": exact, "outcome": "exact" if exact else "near_miss"})
    return out


def trainer_recipe(root):
    recipe = json.loads(PILOT.read_text())
    parent = root / "parent"
    parent.mkdir(parents=True, exist_ok=True)
    for name in trainer.TOKENIZER_FILES:
        (parent / name).write_text(name)
    recipe["parent"]["path"] = str(parent)
    recipe["outputs"] = {"archive": str(root / "archive"), "graceful_stop": str(root / "control" / "stop.json"),
                         "trainer": str(root / "archive" / "runtime")}
    recipe["dev_gate"]["evidence_directory"] = str(root / "gates")
    runtime = recipe["runtime"]
    runtime.update({"completion_max_tokens": 6, "seq_chunk": 3, "vocab_chunk": 50000, "min_free_disk_bytes": 0,
                    "checkpoint_bytes_estimate": 0})
    runtime["optimizer"] = {**runtime["optimizer"], "hidden_lr": 0.02, "side_lr": 0.002}
    return recipe


def pool_prompts():
    generator = torch.Generator().manual_seed(5)
    prompts = {}
    for index in range(12):
        body = torch.randint(22, 129000, (3 + index % 4,), generator=generator).tolist()
        prompts[f"p{index:02d}"] = (0, *body)
    return prompts


def build_trainer(recipe, *, model, update=0, cursor=0, resumed_from=None, stop_after=None, settings=None):
    optimizer = trainer.build_optimizer(model, recipe["runtime"], check_dispatch=False)
    prompts = pool_prompts()
    return trainer.GRPOTrainer(recipe, "r" * 64, model=model, optimizer=optimizer, ops=trainer.DeviceOps(torch.device("cpu")),
                               reward=fake_reward, pool_ids=sorted(prompts), prompts=prompts,
                               settings=settings or trainer.Settings(stop_after=stop_after), update=update, cursor=cursor,
                               resumed_from=resumed_from, generate=fake_generate)


def events(archive, kind="grpo_update"):
    return [json.loads(line) for line in (archive / "telemetry.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == kind]


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_segment(self, root, *, stop_after, resume=None):
        recipe = trainer_recipe(root)
        if resume is None:
            model = tiny_model()
            torch.manual_seed(recipe["runtime"]["seed"])
            segment = build_trainer(recipe, model=model, stop_after=stop_after)
        else:
            saved = plain(trainer.verify_checkpoint(resume, "r" * 64))
            model = LlamaForCausalLM.from_pretrained(str(resume))
            segment = build_trainer(recipe, model=model, update=saved["update"], cursor=saved["sampler"]["cursor"],
                                    resumed_from=saved["update"], stop_after=stop_after)
            trainer.load_optimizer_state(segment.optimizer, resume / "optimizer.pt")
            rng = torch.load(resume / "rng_state.pt", weights_only=True)
            torch.set_rng_state(rng["cpu"])
            self.assertIsNone(rng["cuda"])
        report = segment.run()
        return segment, report

    def test_segment_dynamic_sampling_warmup_masking_and_telemetry(self):
        segment, report = self.run_segment(self.root / "a", stop_after=4)
        self.assertEqual((report["status"], report["update"], report["reason"]), ("saved", 4, "stop_after_update"))
        updates = events(self.root / "a" / "archive")
        self.assertEqual([e["update"] for e in updates], [1, 2, 3, 4])
        self.assertAlmostEqual(updates[0]["lr_factor"], 0.1)
        self.assertAlmostEqual(updates[0]["lr"][0], 0.002)
        self.assertEqual([e["buffer_position"] for e in updates], [0, 1, 0, 1])
        for event in updates:
            for name in ("zero_variance_share_before", "zero_variance_share_after", "mean_abs_advantage", "reward_mean_trained",
                         "exact_rate_trained", "completion_tokens_mean_trained", "clip_fraction", "reward_mean_drawn",
                         "completion_tokens_mean_drawn"):
                self.assertIn(name, event)
            self.assertLessEqual(event["rows_backward"], event["rows_trained"] - event["rows_masked_truncated"])
            self.assertEqual(event["rows_trained"], 8 * event["groups_trained"])
        for event in updates[::2]:
            self.assertEqual((event["ratio_max"], event["ratio_min"]), (1.0, 1.0))  # first update of a buffer: on-policy
        self.assertTrue(any(e["ratio_max"] != 1.0 for e in updates[1::2] if e["rows_backward"]))  # later: old log-probs
        drawn = events(self.root / "a" / "archive", "segment_start")
        self.assertEqual(drawn[0]["target"], 4)
        checkpoint = self.root / "a" / "archive" / "full" / "checkpoint-4"
        saved = plain(trainer.verify_checkpoint(checkpoint, "r" * 64))
        self.assertEqual(saved["update"], 4)
        self.assertEqual(saved["sampler"]["cursor"], segment.sampler.cursor)
        self.assertTrue((checkpoint / "model.safetensors").is_file() and (checkpoint / "optimizer.pt").is_file())

    def test_zero_variance_prompts_never_train(self):
        recipe = trainer_recipe(self.root)
        segment = build_trainer(recipe, model=tiny_model(), stop_after=2)
        torch.manual_seed(1)
        plan_groups = []
        original = segment.draw

        def spy(count):
            groups = original(count)
            plan_groups.extend(groups)
            return groups
        segment.draw = spy
        segment.run_buffer()
        zero = [g for g in plan_groups if g.prompt_id in ZERO]
        self.assertTrue(all(g.zero_variance or any(r.truncated for r in g.rows) for g in zero))
        update = events(self.root / "archive")[0]
        self.assertGreaterEqual(update["groups_drawn"], 12)

    def test_resume_at_a_buffer_boundary_reproduces_the_uninterrupted_run(self):
        whole, _ = self.run_segment(self.root / "whole", stop_after=6)
        self.run_segment(self.root / "split", stop_after=4)
        resumed, report = self.run_segment(self.root / "split", stop_after=6,
                                           resume=self.root / "split" / "archive" / "full" / "checkpoint-4")
        self.assertEqual(report["from_update"], 4)
        for (name, a), (_, b) in zip(whole.model.state_dict().items(), resumed.model.state_dict().items(), strict=True):
            self.assertTrue(torch.equal(a, b), name)
        later = [e for e in events(self.root / "split" / "archive") if e["update"] > 4]
        self.assertEqual([e["resumed_from"] for e in later], [4, 4])
        self.assertEqual(resumed.sampler.cursor, whole.sampler.cursor)

    def test_stop_must_end_a_rollout_buffer(self):
        segment = build_trainer(trainer_recipe(self.root), model=tiny_model(), stop_after=3)
        with self.assertRaises(trainer.TrainerError):
            segment.run()

    def test_deadline_stops_before_a_buffer_that_would_not_finish(self):
        clock = {"now": 1000.0}
        settings = trainer.Settings(deadline=1000.0 + 180 + 360 + 90 - 1, clock=lambda: clock["now"])
        segment = build_trainer(trainer_recipe(self.root), model=tiny_model(), settings=settings)
        report = segment.run()
        self.assertEqual(report["status"], "no_progress")
        settings = trainer.Settings(deadline=1000.0 + 10_000, clock=lambda: clock["now"])
        root = self.root / "b"
        segment = build_trainer(trainer_recipe(root), model=tiny_model(), settings=settings)
        original = segment.ops.reset_peak

        def tick():  # each buffer now takes 4,000 s on the fake clock
            original()
            clock["now"] += 4000.0
        segment.ops.reset_peak = tick
        report = segment.run()
        self.assertEqual((report["reason"], report["update"]), ("deadline", 4))

    def test_graceful_stop_file_is_bound_to_the_recipe(self):
        recipe = trainer_recipe(self.root)
        stop = Path(recipe["outputs"]["graceful_stop"])
        stop.parent.mkdir(parents=True)
        stop.write_text(json.dumps({"action": "save_and_stop", "bound_recipe_sha256": "r" * 64}))
        self.assertEqual(build_trainer(recipe, model=tiny_model()).run()["status"], "no_progress")
        stop.write_text(json.dumps({"action": "save_and_stop", "bound_recipe_sha256": "x" * 64}))
        with self.assertRaises(trainer.TrainerError):
            build_trainer(recipe, model=tiny_model()).run()

    def test_disk_floor_stops_before_writing(self):
        recipe = trainer_recipe(self.root)
        recipe["runtime"]["min_free_disk_bytes"] = 10**18
        segment = build_trainer(recipe, model=tiny_model(), stop_after=2)
        with self.assertRaises(trainer.DiskLow):
            segment.run()
        self.assertFalse(any((self.root / "archive" / "full").glob("*")))

    def test_continue_record_gates_training_past_an_evaluation(self):
        recipe = trainer_recipe(self.root)
        checkpoint = self.root / "archive" / "full" / "checkpoint-50"
        checkpoint.mkdir(parents=True)
        (checkpoint / "campaign-manifest.json").write_text("{}")
        with self.assertRaises(state.StateError):
            state.check_continue(recipe, "r" * 64, checkpoint, 50)
        decision = self.root / "decision.json"
        decision.write_text("{}")
        path = state.continue_record_path(recipe, 50)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(state.continue_record("r" * 64, 50, checkpoint, decision)))
        state.check_continue(recipe, "r" * 64, checkpoint, 50)
        with self.assertRaises(state.StateError):
            state.check_continue(recipe, "s" * 64, checkpoint, 50)


# CUDA discipline -----------------------------------------------------------------------------------

class FakeCuda:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            self.calls.append((name, args))
            if name == "mem_get_info":
                return (20 * 2**30, 32 * 2**30)
            if name == "get_device_name":
                return "fake"
            if name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
                return 1
            return None
        return call


class DeviceOpsTests(unittest.TestCase):
    def test_bare_cuda_is_pinned_to_index_zero(self):
        self.assertEqual(trainer.resolve_device("cuda"), torch.device("cuda", 0))
        with self.assertRaises(trainer.TrainerError):
            trainer.DeviceOps(torch.device("cuda"), FakeCuda())

    def test_context_before_memory_statistics(self):
        cuda = FakeCuda()
        touched = []
        ops = trainer.DeviceOps(torch.device("cuda", 0), cuda, touch=touched.append)
        for call in (ops.reset_peak, ops.memory, ops.rng_state):
            with self.assertRaises(trainer.TrainerError):
                call()
        self.assertEqual(cuda.calls, [])
        info, allocator = plain(trainer.start_cuda(ops, {"allocator_safety_mib": 256}))
        names = [name for name, _ in cuda.calls]
        self.assertEqual(names[:2], ["set_device", "synchronize"])
        self.assertEqual(touched, [torch.device("cuda", 0)])
        self.assertLess(names.index("synchronize"), names.index("reset_peak_memory_stats"))
        self.assertLess(names.index("set_per_process_memory_fraction"), names.index("reset_peak_memory_stats"))
        for _, args in cuda.calls:
            devices = [a for a in args if isinstance(a, torch.device)]
            self.assertTrue(all(d.index == 0 for d in devices))
            self.assertTrue(devices, f"a CUDA call without an explicit device: {args}")
        self.assertEqual(info["device"], "cuda:0")
        self.assertAlmostEqual(allocator["allocator_fraction"], (20 * 1024 - 256) / (32 * 1024), places=4)
        self.assertEqual(set(ops.memory()), {"memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"})

    def test_package_routes_cuda_calls_through_device_ops(self):
        package = REPO / "packages/sepalith/src/sepalith/training/grpo"
        allowed = {"is_available", "device_count", "set_device"}
        for path in package.glob("*.py"):
            text = path.read_text()
            self.assertIsNone(re.search(r"""set_device\(\s*["']cuda["']\s*\)""", text), path.name)
            self.assertIsNone(re.search(r"""device_map=\{"": "cuda"\}""", text), path.name)
            for match in re.finditer(r"torch\.cuda\.(\w+)", text):
                self.assertIn(match.group(1), allowed, f"{path.name}: torch.cuda.{match.group(1)} outside DeviceOps")
        dev = (package / "dev_generation.py").read_text()
        self.assertIn('DEVICE = "cuda:0"', dev)


@unittest.skipUnless(Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536/config.json").is_file(),
                     "arm A checkpoint not present")
class DispatchTests(unittest.TestCase):
    def test_recipe_dispatch_hash_matches_the_plain_bf16_load(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        from sepalith.training.optim.full_weight_optimizer import inventory_full_weight

        config = AutoConfig.from_pretrained("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536")
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        _, manifest = inventory_full_weight(model, "aurora_mix")
        self.assertEqual(manifest["ordered_rows_sha256"], recipe_module.OPTIMIZER_DISPATCH_SHA256)
        self.assertEqual(manifest["counts"], {"adamw": 87, "muon": 210, "aurora": 84})


@unittest.skipUnless(Path("/mnt/e/sepalith/campaign-20260915/data-work/C14-grpo-pool-v1").is_dir(), "pool bundle not present")
class RealRewardTests(unittest.TestCase):
    def test_pool_reward_scores_gold_completions_exact_with_the_pinned_tools(self):
        from sepalith.training.grpo.pool import load_bundle, read_pool
        from sepalith.training.rl.reward_v21 import DATA_ROOT, TRAIN_INPUTS

        recipe = plain(recipe_module.load(PILOT, root_override=REPO, check_source=False))
        pool_record = recipe["pool"]
        rows = plain(read_pool(REPO / pool_record["pool"]["path"], pool_record["pool"]["sha256"]))
        ids = [str(row["id"]) for row in rows]
        bundle = load_bundle(Path(pool_record["bundle"]["path"]), pool_record["bundle"]["sha256"], ids)
        families = {}
        for row in rows:
            families.setdefault(row["family"], str(row["id"]))
        wanted = {families[name] for name in ("finish_block", "format_propagation", "no_op", "rename_propagation")}
        gold = {}
        with (DATA_ROOT / TRAIN_INPUTS["rows"][0]).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in wanted:
                    gold[row["id"]] = tuple(row["input_ids"][row["target_start"]:])
                    self.assertEqual(tuple(row["input_ids"][:row["target_start"]]), bundle[row["id"]].prompt_ids)
        reward = trainer.make_pool_reward(recipe, bundle)
        order = sorted(gold)
        records = reward(order + order[:1], [gold[i] for i in order] + [(500,) * 1024])
        self.assertEqual([r["exact"] for r in records[:-1]], [True] * len(order))
        self.assertEqual([r["reward"] for r in records[:-1]], [1.0] * len(order))
        self.assertEqual((records[-1]["reward"], records[-1]["cap_hit"]), (-0.5, True))


# Night driver --------------------------------------------------------------------------------------

def counts(valid=74, cap=1, edit=27, nofp=8, score=117, noedit=1):
    dev75 = {"protocol_valid": valid, "cap_hit": cap, "edit_exact": edit, "strict_noop_false_suggestions": nofp,
             "predicted_noop": 24, "strict_noop_correct": 24}
    dev250 = {"edit_exact": score - 38, "strict_noop_correct": 38, "predicted_noop": 38 + noedit}
    return dev75, dev250


def write_result(path, step, recipe_path, manifest, panel, values):
    value = {"schema": night_arm.RESULT_SCHEMA, "status": "complete", "step": step,
             "bound_recipe": {"path": str(recipe_path), "sha256": sha(recipe_path) if Path(recipe_path).is_file() else "0" * 64},
             "checkpoint": {"path": str(Path(manifest).parent), "campaign_manifest_sha256": sha(manifest)},
             "panel": {"path": panel["path"], "sha256": panel["sha256"]},
             "summary": {"denominators": {"cases": panel["cases"]}, "counts": values, "panel_sha256": panel["sha256"],
                         "generation_max_new_tokens": 1024, "generation_runtime": BATCH},
             "generation_runtime": {"dtype": "bf16", "cuda_visible_devices": "0", **BATCH}, "final_set_access": False}
    Path(path).write_text(json.dumps(value))


def write_checkpoint(archive, step, recipe_sha, cursor=0):
    directory = archive / "full" / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"weights %d" % step)
    (directory / "trainer-state.json").write_text(json.dumps({"schema": state.STATE_SCHEMA, "update": step,
                                                              "sampler": {"cursor": cursor}}))
    files = {path.name: state.file_entry(path) for path in directory.iterdir()}
    (directory / "campaign-manifest.json").write_text(json.dumps({"schema": state.CHECKPOINT_SCHEMA, "step": step,
                                                                  "recipe_sha256": recipe_sha, "files": files}))


def update_event(update, resumed_from=None, **override):
    event = {"event": "grpo_update", "update": update, "buffer_position": (update - 1) % 2, "resumed_from": resumed_from,
             "groups_trained": 4, "zero_variance_share_before": 0.3, "zero_variance_share_after": 0.0, "mean_abs_advantage": 0.4,
             "reward_mean_drawn": 0.4, "exact_rate_drawn": 0.3, "completion_tokens_mean_drawn": 45.0, "clip_fraction": 0.01,
             "loss": 0.02, "optimizer_step": True, "update_seconds": 10.0, "rollout_seconds": 30.0, "max_memory_reserved": 2**34}
    event.update(override)
    return event


class FakeLauncher(night_arm.Launcher):
    def __init__(self, test, root):
        self.test, self.root = test, root
        self.calls = []
        self.counts = {}
        self.trainer_codes = []
        self.deadline_at = None  # save at this update instead of the target
        self.event = lambda update: {}
        self.on_train = None

    def run(self, label, module_args, *, budget, on_stop):
        module = module_args[0]
        self.calls.append((module, module_args))
        args = dict(zip(module_args[1::2], module_args[2::2], strict=False))
        if module == night.TRAINER:
            return self.train(module_args)
        self.test.assertEqual(module, night.EVALUATOR)
        recipe = json.loads(Path(args["--bound-recipe"]).read_text())
        panel = recipe["dev_gate"]["panels"][args["--panel"]]
        step = int(Path(args["--checkpoint"]).name.removeprefix("checkpoint-"))
        dev75, dev250 = self.counts.get(step, counts())
        write_result(args["--output"], step, args["--bound-recipe"], Path(args["--checkpoint"]) / "campaign-manifest.json", panel,
                     dev75 if args["--panel"] == "dev75" else dev250)
        return 0

    def train(self, module_args):
        recipe_path = Path(module_args[module_args.index("--recipe") + 1])
        recipe = json.loads(recipe_path.read_text())
        if self.trainer_codes:
            code = self.trainer_codes.pop(0)
            if code != 0:
                return code
        start = int(Path(module_args[module_args.index("--resume") + 1]).name.removeprefix("checkpoint-")) if "--resume" in module_args else 0
        stop_after = int(module_args[module_args.index("--stop-after-update") + 1]) if "--stop-after-update" in module_args else None
        target = min(step for step in rules.evaluation_steps() if step > start)
        if stop_after is not None:
            target = min(target, stop_after)
        if self.deadline_at is not None and start < self.deadline_at < target:
            target = self.deadline_at
        archive = Path(recipe["outputs"]["archive"])
        archive.mkdir(parents=True, exist_ok=True)
        with (archive / "telemetry.jsonl").open("a") as stream:
            for update in range(start + 1, target + 1):
                stream.write(json.dumps({**update_event(update, start or None), **self.event(update)}) + "\n")
            stream.write(json.dumps({"event": "checkpoint_durable", "step": target, "seconds": 120.0}) + "\n")
        write_checkpoint(archive, target, sha(recipe_path), cursor=target * 6)
        if self.on_train is not None:
            self.on_train(target)
        return 0


def driver_recipe(root, source):
    recipe = json.loads(source.read_text())
    recipe["outputs"] = {"archive": str(root / "archive"), "graceful_stop": str(root / "control" / "stop.json"),
                         "trainer": str(root / "archive" / "runtime")}
    recipe["dev_gate"]["evidence_directory"] = str(root / "gates")
    recipe["source"]["repository_root"] = str(REPO)
    start = root / "start" / "checkpoint-536"
    start.mkdir(parents=True)
    (start / "campaign-manifest.json").write_text("{}")
    recipe["parent"]["path"] = str(start)
    start75, start250 = counts()
    results = {}
    for name, values in (("dev75", start75), ("dev250", start250)):
        path = root / "start" / f"{name}.json"
        write_result(path, 536, root / "sft-recipe.json", start / "campaign-manifest.json", recipe["dev_gate"]["panels"][name], values)
        results[name] = {"path": str(path), "sha256": sha(path)}
    recipe["parent_reference"]["results"] = results
    recipe["parent_reference"]["metrics"] = offline_rules.offline_metrics({"counts": start75}, {"counts": start250})
    path = root / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe_path = driver_recipe(self.root, PILOT)
        self.launcher = FakeLauncher(self, self.root)
        self.clock = {"now": 1000.0}
        self.deadline = 1000.0 + 10**6

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, eval_budget_seconds=100.0, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: self.clock["now"],
                                      deadline=self.deadline, stop_request_file=self.root / "stop-request")
        return night.GRPOArm(settings, lambda s, w, stop: self.launcher, verify_files=False)

    def decision(self, step):
        return json.loads(self.arm().decisions(step)[-1].read_text())

    def stop_after(self, step):
        """End the job after the trainer reaches ``step``: the deadline passes."""
        def hook(target):
            if target >= step:
                self.clock["now"] = self.deadline + 1
        self.launcher.on_train = hook

    def override(self, step, outcome, reason="test override"):
        arm = self.arm()
        path = self.root / "gates" / f"override-step-{step}.json"
        path.write_text(json.dumps({"schema": night_arm.OVERRIDE_SCHEMA, "step": step, "bound_recipe_sha256": arm.recipe_sha,
                                    "outcome": outcome, "reason": reason}))

    def test_dev75_and_dev250_evaluations_continue_then_stop_for_user_and_override(self):
        self.launcher.counts[200] = counts(nofp=10)
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        for step in (50, 100, 150):
            decision = self.decision(step)
            self.assertEqual(decision["outcome"], rules.CONTINUE)
            self.assertTrue((self.root / "gates" / f"continue-step-{step}.json").is_file())
        self.assertEqual(self.decision(100)["panels"], ["dev75"])
        self.assertEqual(self.decision(150)["panels"], ["dev250", "dev75"])
        names = [row["name"] for row in self.decision(150)["rule"]["rows"]]
        self.assertIn("score250_drop", names)
        self.assertIn("reward_hacking", names)
        self.assertNotIn("score250_drop", [row["name"] for row in self.decision(100)["rule"]["rows"]])
        self.assertEqual(self.decision(200)["outcome"], rules.STOP_FOR_USER)
        self.assertEqual(self.decision(150)["r1_versus_start"]["outcome"], rules.CONTINUE)
        resumes = [args[args.index("--resume") + 1] for module, args in self.launcher.calls if module == night.TRAINER and "--resume" in args]
        self.assertEqual([Path(p).name for p in resumes], ["checkpoint-50", "checkpoint-100", "checkpoint-150"])
        result = json.loads((self.root / "job-result.json").read_text())
        self.assertIn("override-step-200.json", result["needs_user"])
        self.assertIn("R8 nightly candidate", result["summary"])
        self.override(200, rules.CONTINUE)
        self.stop_after(250)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)
        self.assertEqual(self.decision(200)["decided_by"], "human_override")
        self.assertTrue((self.root / "gates" / "continue-step-200.json").is_file())
        self.assertIn(250, self.arm().durable_steps())

    def test_rollback_ends_the_pilot_with_the_last_good_checkpoint(self):
        self.launcher.counts[100] = counts(nofp=12)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        summary = json.loads((self.root / "archive" / "pilot-summary.json").read_text())
        self.assertEqual(summary["status"], "rolled_back")
        self.assertTrue(summary["result_checkpoint"].endswith("checkpoint-50"))
        self.assertEqual(self.decision(100)["last_good_checkpoint"], summary["result_checkpoint"])
        self.assertEqual(summary["start"]["metrics"]["score250"], 117)

    def test_first_evaluation_rollback_returns_the_starting_model(self):
        self.launcher.counts[50] = counts(valid=60)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        summary = json.loads((self.root / "archive" / "pilot-summary.json").read_text())
        self.assertTrue(summary["result_checkpoint"].endswith("start/checkpoint-536"))

    def test_zero_variance_streak_stops_at_the_third_evaluation(self):
        self.launcher.event = lambda update: {"zero_variance_share_after": 0.75}
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertEqual([self.decision(s)["outcome"] for s in (50, 100, 150)], [rules.CONTINUE, rules.CONTINUE, rules.STOP_FOR_USER])
        row = next(r for r in self.decision(150)["rule"]["rows"] if r["name"] == "zero_variance_streak")
        self.assertEqual(row["outcome"], rules.STOP_FOR_USER)

    def test_completion_length_doubling_stops(self):
        self.launcher.event = lambda update: {"completion_tokens_mean_drawn": 45.0 if update <= 10 else 120.0}
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        row = next(r for r in self.decision(50)["rule"]["rows"] if r["name"] == "length_factor")
        self.assertEqual(row["outcome"], rules.STOP_FOR_USER)

    def test_reward_hacking_rolls_back_at_a_dev250_evaluation(self):
        self.launcher.event = lambda update: {"reward_mean_drawn": 0.4 if update <= 10 else 0.6}
        self.launcher.counts[150] = counts(score=107)
        self.launcher.counts[100] = counts()
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        rows = {r["name"]: r["outcome"] for r in self.decision(150)["rule"]["rows"]}
        self.assertEqual(rows["reward_hacking"], rules.ROLLBACK)

    def test_deadline_save_defers_and_the_next_night_resumes(self):
        self.launcher.deadline_at = 20

        def hook(target):
            self.clock["now"] = self.deadline + 1
        self.launcher.on_train = hook
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)
        self.assertEqual(self.arm().durable_steps(), [20])
        self.launcher.on_train = None
        self.launcher.deadline_at = None
        self.clock["now"] = 1000.0
        self.stop_after(50)
        self.launcher.counts[50] = counts()
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)  # 50 reached; deadline before its evaluation
        module, args = [c for c in self.launcher.calls if c[0] == night.TRAINER][-1]
        self.assertEqual(Path(args[args.index("--resume") + 1]).name, "checkpoint-20")

    def test_resumed_segment_supersedes_lost_telemetry(self):
        archive = self.root / "archive"
        archive.mkdir(parents=True)
        with (archive / "telemetry.jsonl").open("w") as stream:
            for update in range(1, 31):
                stream.write(json.dumps(update_event(update, completion_tokens_mean_drawn=500.0 if update > 20 else 45.0)) + "\n")
        write_checkpoint(archive, 20, sha(self.recipe_path))
        self.stop_after(100)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)
        self.assertEqual(self.decision(50)["outcome"], rules.CONTINUE)

    def test_trainer_exit_codes(self):
        self.launcher.trainer_codes = [night.TRAINER_DISK]
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertIn("R7", json.loads((self.root / "job-result.json").read_text())["summary"])
        self.launcher.trainer_codes = [night.TRAINER_NO_PROGRESS]
        self.assertEqual(self.arm().run(), night_arm.EXIT_FAILED)
        (self.root / "stop-request").write_text("stop")
        self.launcher.trainer_codes = [night.TRAINER_NO_PROGRESS]
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)

    def test_trainer_without_progress_is_a_failure(self):
        class Idle(FakeLauncher):
            def train(self, module_args):
                return 0
        self.launcher = Idle(self, self.root)
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: 1000.0, deadline=10**9)
        arm = night.GRPOArm(settings, lambda s, w, stop: self.launcher, verify_files=False)
        with self.assertRaises(night_arm.DriverError):
            arm.run()

    def test_evaluator_crash_stops_for_the_user(self):
        class Broken(FakeLauncher):
            def run(self, label, module_args, *, budget, on_stop):
                if module_args[0] == night.EVALUATOR:
                    return 1
                return super().run(label, module_args, budget=budget, on_stop=on_stop)
        self.launcher = Broken(self, self.root)
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertIn("evaluator crash", self.decision(50)["rule"]["problems"][0])

    def test_run_refuses_the_smoke_recipe(self):
        self.recipe_path = driver_recipe(self.root / "s", SMOKE)
        with self.assertRaises(night_arm.DriverError):
            self.arm().run()


class SmokeDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe_path = driver_recipe(self.root, SMOKE)
        self.launcher = FakeLauncher(self, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: 1000.0, deadline=10**9)
        return night.GRPOArm(settings, lambda s, w, stop: self.launcher, verify_files=False)

    def test_smoke_trains_saves_evaluates_and_resumes(self):
        self.assertEqual(night.run_smoke(self.arm()), night_arm.EXIT_DONE)
        receipt = json.loads((self.root / "archive" / "smoke-receipt.json").read_text())
        self.assertEqual((receipt["status"], receipt["updates"]), ("pass", [1, 2, 3, 4, 5, 6]))
        self.assertEqual(receipt["dev75"]["counts"]["edit"], 27)
        trains = [args for module, args in self.launcher.calls if module == night.TRAINER]
        self.assertEqual([args[args.index("--stop-after-update") + 1] for args in trains], ["4", "6"])
        self.assertNotIn("--resume", trains[0])
        self.assertEqual(Path(trains[1][trains[1].index("--resume") + 1]).name, "checkpoint-4")
        evaluations = [args for module, args in self.launcher.calls if module == night.EVALUATOR]
        self.assertEqual(len(evaluations), 1)
        self.assertTrue(evaluations[0][evaluations[0].index("--checkpoint") + 1].endswith("checkpoint-4"))

    def test_smoke_fails_when_nothing_was_trained(self):
        self.launcher.event = lambda update: {"optimizer_step": False}
        self.assertEqual(night.run_smoke(self.arm()), night_arm.EXIT_FAILED)
        receipt = json.loads((self.root / "archive" / "smoke-receipt.json").read_text())
        self.assertIn("no optimizer step ran", " ".join(receipt["problems"]))

    def test_smoke_trainer_failure_is_a_failure(self):
        self.launcher.trainer_codes = [1]
        self.assertEqual(night.run_smoke(self.arm()), night_arm.EXIT_FAILED)


class JobCommandTests(unittest.TestCase):
    def test_job_commands_parse(self):
        for name in ("C15-grpo-smoke", "C15-grpo-pilot-1"):
            job = json.loads((REPO / f"docs/training/plan-20260927/c15/jobs/{name}.json").read_text())
            argv = job["command"]
            args = night.build_parser().parse_args(argv[argv.index("sepalith.training.grpo.night") + 1:])
            self.assertEqual(args.command, "smoke" if "smoke" in name else "run")
            self.assertFalse(args.no_guard)
            local = REPO / Path(str(args.recipe)).relative_to(recipe_module.DEPLOY_ROOT)
            self.assertEqual(json.loads(local.read_text())["kind"], "smoke" if "smoke" in name else "pilot")
            copy.deepcopy(job)


if __name__ == "__main__":
    unittest.main()
