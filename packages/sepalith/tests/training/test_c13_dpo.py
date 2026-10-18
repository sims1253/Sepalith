"""Card C13: the in-repo DPO path on a tiny CPU model.

The harness mirrors ``offline.trainer.run`` for ``dpo``: the real aurora_mix
optimizer mixin with bf16 stochastic rounding, the v2 TrainingArguments, one
pair per micro-batch in a frozen order with a cursor check, stored reference
log-probabilities, the pair-batch gate at train begin, the reference-alignment
gate at the first optimizer boundary and ``DeadlineStop``.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments

from sepalith.training.offline import dpo
from sepalith.training.optim.full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
from sepalith.training.sft import full_weight_edit_sft as trainer_module
from sepalith.training.sft.night_controls import DeadlineStop

MAX_STEPS = 6
ACCUMULATION = 2
PAIRS = 4


def pairs(count=PAIRS):
    out = []
    for index in range(count):
        prompt = [0] + [2 + (index + k) % 40 for k in range(5 + index % 3)]
        chosen = [42 + (index + k) % 9 for k in range(2 + index % 2)] + [60, 61, 1]
        rejected = [52 + (index * 2 + k) % 7 for k in range(3)] + [60, 61, 1]
        out.append({"pair_id": f"r{index}:p0", "split": "train", "prompt_ids": prompt, "chosen_ids": chosen, "rejected_ids": rejected})
    return out


def model(dtype=torch.bfloat16):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False, bos_token_id=0,
                         eos_token_id=1, pad_token_id=1)
    return LlamaForCausalLM(config).to(dtype)


def references(net, data):
    """What offline.reference stores: completion log-probabilities of the starting model under no_grad."""
    net.train()
    values = []
    with torch.no_grad():
        for pair in data:
            prompt = pair["prompt_ids"]
            values.append(tuple(float(dpo.completion_logp(net, torch.tensor([prompt + pair[name]]), len(prompt)))
                                for name in ("chosen_ids", "rejected_ids")))
    return values


def recipe():
    return {"seed": 3407, "runtime": {"micro_batch": 1, "gradient_accumulation": ACCUMULATION, "max_steps": MAX_STEPS,
                                      "learning_rate": 1e-3, "scheduler": "cosine_with_min_lr",
                                      "lr_scheduler_kwargs": {"min_lr_rate": 0.1}, "warmup_steps": 2,
                                      "checkpoint_every": MAX_STEPS}}


def run(output, *, resume=None, night=None, refs=None):
    data = pairs()
    net = model()
    stored = refs if refs is not None else references(model(), data)
    draws = [i % PAIRS for i in range(MAX_STEPS * ACCUMULATION)]
    dataset = dpo.PairDataset(dpo.PairRows(data, draws, stored))
    initial = 0 if resume is None else int(Path(resume).name.removeprefix("checkpoint-"))
    observed, updates, gates, alignments = [], [], [], []
    holder = {}

    class Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                expected = [dataset[i] for i in range(ACCUMULATION)]
                gates.append(dpo.pair_batch_gate(holder["trainer"], kwargs["train_dataloader"], expected,
                                                 accumulation=ACCUMULATION, global_step=0))

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if state.global_step == 0 and initial == 0:
                alignments.append(dpo.reference_alignment(holder["trainer"].dpo_records))

        def on_step_end(self, args, state, control, **kwargs):
            updates.append(dpo.update_summary(holder["trainer"].dpo_records))
            holder["trainer"].dpo_records.clear()
            if night is not None:
                night.apply(state.global_step, control)

    class Tiny(FullWeightOptimizerTrainerMixin, dpo.DPOLossMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset)

        def dpo_positions(self, positions):
            start = initial * ACCUMULATION + len(observed)
            if positions != list(range(start, start + len(positions))):
                raise AssertionError("draw cursor differs")
            observed.extend(positions)

    kwargs = trainer_module.training_arguments_kwargs(recipe(), output)
    kwargs.update(use_cpu=True, bf16=False, gradient_checkpointing=False, save_total_limit=None, logging_first_step=False)
    trainer = Tiny(model=net, args=TrainingArguments(**kwargs), train_dataset=dataset, data_collator=dpo.collate_pair,
                   callbacks=[Callback()])
    trainer.full_weight_optimizer_config = OptimizerConfig(arm="aurora_mix", hidden_lr=1e-3, side_lr=1e-4)
    holder["trainer"] = trainer
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    params = {name: p.detach().clone() for name, p in net.named_parameters()}
    return {"params": params, "observed": observed, "updates": updates, "gates": gates, "alignments": alignments,
            "step": trainer.state.global_step}


class CompletionLogpTests(unittest.TestCase):
    def test_sums_exactly_the_completion_tokens(self):
        net = model(torch.float32).eval()
        pair = pairs()[1]
        prompt, completion = pair["prompt_ids"], pair["chosen_ids"]
        ids = torch.tensor([prompt + completion])
        start = len(prompt)
        with torch.no_grad():
            value = float(dpo.completion_logp(net, ids, start))
            full = net(input_ids=ids).logits.float().log_softmax(-1)
            manual = sum(float(full[0, t - 1, ids[0, t]]) for t in range(start, ids.shape[1]))
            labels = ids.clone()
            labels[:, :start] = -100  # prompt masked: the SFT target-only convention
            fused = -float(net(input_ids=ids, labels=labels).loss) * len(completion)
        self.assertAlmostEqual(value, manual, places=4)
        self.assertAlmostEqual(value, fused, places=4)
        # The prompt never enters the sum: one more prompt-side position changes the count, not the terms used.
        with torch.no_grad():
            shorter = float(dpo.completion_logp(net, ids, start + 1))
        self.assertAlmostEqual(value - shorter, float(full[0, start - 1, ids[0, start]]), places=4)
        with self.assertRaises(dpo.DPOGateError):
            dpo.completion_logp(net, ids, ids.shape[1])

    def test_sigmoid_dpo_loss(self):
        zero = torch.tensor(-5.0)
        loss, stats = dpo.dpo_loss(zero, zero, zero, zero)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        self.assertEqual(stats["accuracy"], 0.0)
        better, stats = dpo.dpo_loss(torch.tensor(-4.0), torch.tensor(-6.0), zero, zero, beta=0.1)
        # Log-ratios +1 (chosen) and -1 (rejected): margin 2, logits beta * 2 = 0.2.
        self.assertAlmostEqual(float(better), -math.log(1 / (1 + math.exp(-0.2))), places=6)
        self.assertEqual(stats["accuracy"], 1.0)
        self.assertAlmostEqual(json.loads(json.dumps(stats))["margin"], 2.0, places=6)
        worse, _ = dpo.dpo_loss(torch.tensor(-6.0), torch.tensor(-4.0), zero, zero, beta=0.1)
        self.assertGreater(float(worse), math.log(2))

    def test_pairs_must_be_train_rows_with_eos_terminated_completions(self):
        good = pairs()[0]
        dpo.validate_pair(good)
        for change, message in (({"split": "dev"}, "TRAIN"), ({"chosen_ids": good["rejected_ids"]}, "chosen equals rejected"),
                                ({"rejected_ids": [5, 60, 61]}, "EOS"), ({"prompt_ids": [3, 4]}, "BOS")):
            with self.assertRaisesRegex(dpo.DPOGateError, message):
                dpo.validate_pair({**good, **change})


class DPOTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.full = run(Path(cls.tmp.name) / "full")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_loss_decreases_from_log_two(self):
        losses = [u["loss"] for u in self.full["updates"]]
        self.assertEqual(len(losses), MAX_STEPS)
        self.assertAlmostEqual(losses[0], math.log(2), places=5)  # policy equals reference before update 1
        self.assertLess(losses[-1], losses[0] - 0.005)
        self.assertTrue(all(b < a for a, b in zip(losses[1:], losses[2:], strict=False)))  # falls after the first warmup update
        self.assertGreater(self.full["updates"][-1]["accuracy"], 0.5)
        self.assertTrue(all(math.isfinite(u["chosen_log_ratio"]) for u in self.full["updates"]))

    def test_gates_pass_on_the_live_first_update(self):
        gate = self.full["gates"][0]
        self.assertEqual((gate["status"], gate["micro_batches_checked"]), ("pass", ACCUMULATION))
        alignment = self.full["alignments"][0]
        self.assertEqual((alignment["status"], alignment["pairs"]), ("pass", ACCUMULATION))
        self.assertEqual(alignment["max_abs_nats"], 0.0)  # stored references load exactly

    def test_misaligned_references_abort_before_update_one(self):
        stored = references(model(), pairs())
        shifted = stored[1:] + stored[:1]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(dpo.DPOGateError, "do not match the starting policy"):
            run(Path(tmp), refs=shifted)
        self.assertFalse(any(Path(tmp).glob("checkpoint-*")))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(dpo.DPOGateError, "reference log-probabilities"):
            run(Path(tmp), refs=[(1.0, 1.0)] * PAIRS)  # not log-probabilities: the pair gate refuses before update 1

    def test_deadline_saves_full_state_and_resume_is_exact(self):
        root = Path(self.tmp.name) / "interrupted"
        first = run(root, night=DeadlineStop(stop_after_step=3))
        self.assertEqual(first["step"], 3)
        checkpoint = root / "checkpoint-3"
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "model.safetensors"):
            self.assertTrue((checkpoint / name).is_file(), name)
        second = run(root, resume=checkpoint)
        self.assertEqual(second["step"], MAX_STEPS)
        self.assertEqual(first["observed"] + second["observed"], self.full["observed"])
        self.assertEqual([u["loss"] for u in first["updates"] + second["updates"]], [u["loss"] for u in self.full["updates"]])
        for name, value in self.full["params"].items():
            self.assertTrue(torch.equal(value, second["params"][name]), name)
        self.assertEqual((second["gates"], second["alignments"]), ([], []))


if __name__ == "__main__":
    unittest.main()
