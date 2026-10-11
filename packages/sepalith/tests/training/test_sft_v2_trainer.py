"""Card C04: deadline save and exact resume, and the pre-update masking gate, on a tiny CPU model.

The harness mirrors ``full_weight_edit_sft.run``: the real aurora_mix optimizer
mixin with bf16 stochastic rounding, the v2 TrainingArguments (32-step-style
warmup then cosine to 10%), the target-only collator, a sequential frozen draw
order with a cursor check, ``DeadlineStop`` at the optimizer boundary and the
masking gate at train begin.
"""
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments

from sepalith.training.optim.full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
from sepalith.training.sft import full_weight_edit_sft as trainer_module
from sepalith.training.sft.campaign_edit_data import target_only_collator, verify_batch
from sepalith.training.sft.night_controls import DeadlineStop, MaskingGateError, pre_update_masking_gate, verify_microbatch

MAX_STEPS = 6
ACCUMULATION = 2


def rows(count=MAX_STEPS * ACCUMULATION):
    out = []
    for index in range(count):
        prompt = [2 + (index + k) % 40 for k in range(5 + index % 4)]
        body = [42 + (index * 3 + k) % 15 for k in range(2 + index % 3)]
        terminal = [60, 61]
        ids = [0] + prompt + body + terminal + [1]
        out.append({"id": f"r{index}", "input_ids": ids, "target_start": 1 + len(prompt), "target_body_tokens": body,
                    "target_terminal_tokens": terminal, "family": "finish_block", "package_id": "p"})
    return out


class Draws(torch.utils.data.Dataset):
    def __init__(self, data):
        self.rows = data
        self.draws = list(range(len(data)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = dict(self.rows[position])
        row["_draw_position"] = position
        return row


def recipe():
    return {"seed": 3407, "runtime": {"micro_batch": 1, "gradient_accumulation": ACCUMULATION, "max_steps": MAX_STEPS,
                                      "learning_rate": 1e-3, "scheduler": "cosine_with_min_lr",
                                      "lr_scheduler_kwargs": {"min_lr_rate": 0.1}, "warmup_steps": 2,
                                      "checkpoint_every": MAX_STEPS}}


def model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False, bos_token_id=0,
                         eos_token_id=1, pad_token_id=1)
    return LlamaForCausalLM(config).to(torch.bfloat16)


def run(output, *, resume=None, night=None, collate_mutation=None, gate=True):
    dataset = Draws(rows())
    net = model()
    observed, gates, losses, lrs = [], [], [], []
    initial = 0 if resume is None else int(Path(resume).name.removeprefix("checkpoint-"))
    holder = {}

    def collate(items):
        positions = [item.pop("_draw_position") for item in items]
        batch = target_only_collator([dict(item) for item in items], max_sequence_tokens=64)
        verify_batch(batch, items)
        if collate_mutation:
            collate_mutation(batch)
        batch["_draw_position"] = torch.tensor(positions, dtype=torch.long)
        return batch

    class Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if gate and state.global_step == 0:
                expected = [dataset.rows[i] for i in range(ACCUMULATION)]
                gates.append(pre_update_masking_gate(holder["trainer"], kwargs["train_dataloader"], expected,
                                                     accumulation=ACCUMULATION, global_step=state.global_step))

        def on_step_end(self, args, state, control, **kwargs):
            lrs.append(holder["trainer"].lr_scheduler.get_last_lr()[0])
            if night is not None:
                night.apply(state.global_step, control)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                losses.append((state.global_step, logs["loss"]))

    class Tiny(FullWeightOptimizerTrainerMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, *args, **kwargs):
            positions = inputs.pop("_draw_position").tolist()
            start = initial * ACCUMULATION + len(observed)
            if positions != list(range(start, start + len(positions))):
                raise AssertionError("draw cursor differs")
            observed.extend(positions)
            return super().compute_loss(model, inputs, *args, **kwargs)

    kwargs = trainer_module.training_arguments_kwargs(recipe(), output)
    kwargs.update(use_cpu=True, bf16=False, gradient_checkpointing=False, save_total_limit=None, logging_first_step=False)
    trainer = Tiny(model=net, args=TrainingArguments(**kwargs), train_dataset=dataset, data_collator=collate,
                   callbacks=[Callback()])
    trainer.full_weight_optimizer_config = OptimizerConfig(arm="aurora_mix", hidden_lr=1e-3, side_lr=1e-4)
    holder["trainer"] = trainer
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    params = {name: p.detach().clone() for name, p in net.named_parameters()}
    return {"params": params, "observed": observed, "gates": gates, "losses": losses, "lrs": lrs,
            "step": trainer.state.global_step, "optimizer": trainer.optimizer}


class DeadlineResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.full = run(Path(cls.tmp.name) / "full")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_v2_schedule_warms_up_then_decays_to_ten_percent(self):
        lrs = self.full["lrs"]
        self.assertEqual(len(lrs), MAX_STEPS)
        self.assertAlmostEqual(lrs[0], 0.5e-3)  # update 2 uses lambda(1) of a 2-update warmup
        self.assertAlmostEqual(lrs[1], 1e-3)
        self.assertAlmostEqual(lrs[-1], 1e-4, places=9)  # 10% of peak at the horizon
        self.assertTrue(all(a >= b for a, b in zip(lrs[1:], lrs[2:], strict=False)))

    def test_deadline_saves_full_state_and_resume_is_exact(self):
        root = Path(self.tmp.name) / "interrupted"
        # A deadline already in the past stops after the first update; stop_after_step gives a fixed update.
        self.assertEqual(DeadlineStop(deadline_epoch=1.0).reason(1, now=2.0), "deadline")
        first = run(root, night=DeadlineStop(stop_after_step=3))
        self.assertEqual(first["step"], 3)
        checkpoint = root / "checkpoint-3"
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "model.safetensors"):
            self.assertTrue((checkpoint / name).is_file(), name)
        second = run(root, resume=checkpoint)
        self.assertEqual(second["step"], MAX_STEPS)
        # Sampler cursor: the resumed run starts exactly at draw 3 * 2.
        self.assertEqual(first["observed"] + second["observed"], self.full["observed"])
        self.assertEqual(second["observed"][0], 3 * ACCUMULATION)
        # Scheduler and losses continue exactly; weights match bit for bit (RNG drives stochastic rounding).
        self.assertEqual(first["lrs"] + second["lrs"], self.full["lrs"])
        self.assertEqual(first["losses"] + second["losses"], self.full["losses"])
        for name, value in self.full["params"].items():
            self.assertTrue(torch.equal(value, second["params"][name]), name)
        # Optimizer moments match too.
        full_state = self.full["optimizer"].state_dict()["state"]
        resumed_state = second["optimizer"].state_dict()["state"]
        self.assertEqual(sorted(full_state), sorted(resumed_state))
        for key, state in full_state.items():
            for field, value in state.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, resumed_state[key][field]), (key, field))
        self.assertEqual(second["gates"], [])  # the masking gate runs only before update 1

    def test_masking_gate_passes_on_the_live_first_batch_without_changing_training(self):
        gate = self.full["gates"][0]
        self.assertEqual((gate["status"], gate["micro_batches_checked"], gate["prompt_supervised_tokens"]), ("pass", ACCUMULATION, 0))
        expected = sum(len(r["input_ids"]) - r["target_start"] for r in rows()[:ACCUMULATION])
        self.assertEqual(gate["target_denominator"], expected)
        ungated = run(Path(self.tmp.name) / "ungated", gate=False)
        for name, value in self.full["params"].items():
            self.assertTrue(torch.equal(value, ungated["params"][name]), name)

    def test_masking_gate_aborts_before_update_one(self):
        def supervise_prompt(batch):
            batch["labels"][0, 1] = batch["input_ids"][0, 1]

        def drop_eos(batch):
            lengths = batch["attention_mask"].sum(dim=1)
            batch["labels"][0, int(lengths[0]) - 1] = -100

        for mutation, message in ((supervise_prompt, "prompt tokens are supervised"), (drop_eos, "EOS is not supervised")):
            # verify_batch inside the collator runs first; bypass it by mutating after it, as a buggy collator would.
            with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(MaskingGateError, message):
                run(Path(tmp), collate_mutation=mutation)
            self.assertFalse(any((Path(tmp)).glob("checkpoint-*")))

    def test_verify_microbatch_checks_body_terminal_and_positions(self):
        data = rows()[:1]
        batch = target_only_collator([dict(r) for r in data], max_sequence_tokens=64)
        batch["_draw_position"] = torch.tensor([0])
        self.assertEqual(verify_microbatch(batch, data, [0])["eos_supervised_rows"], 1)
        with self.assertRaisesRegex(MaskingGateError, "draw positions"):
            verify_microbatch(batch, data, [1])
        start = data[0]["target_start"]
        broken = {k: v.clone() for k, v in batch.items()}
        broken["labels"][0, start] = -100
        with self.assertRaisesRegex(MaskingGateError, "body is not supervised"):
            verify_microbatch(broken, data, [0])
        self.assertTrue(math.isfinite(self.full["losses"][-1][1]))


if __name__ == "__main__":
    unittest.main()
