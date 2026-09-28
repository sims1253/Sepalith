"""CPU checks for EOS supervision and the exact production SFT draw sampler."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, TrainerCallback
from trl import SFTConfig

from campaign_checkpoint import checkpoint_callback, digest, verify_checkpoint
from campaign_sft import sequential_sft_trainer_class
from campaign_sft_data import full_text_collator, inspect_training_data
from test_campaign_checkpoint import IDENTITY, evaluation, make_model

torch.set_num_threads(1)


def rows():
    return [{"id": f"source-{i}", "split": "train", "renderer_id": "fixture-v1",
             "family": "no_op" if i == 0 else "edit", "package_id": f"package-{i}",
             "input_ids": [0, 11 + i, 5, 7] + [8] * i + [1],
             "target_start": 3, "target_body_tokens": [7] + [8] * i} for i in range(5)]


def fixture_files(root, data=None):
    data = rows() if data is None else data
    tokens, draws = root / "tokens.jsonl", root / "draws.json"
    tokens.write_text("".join(json.dumps(row) + "\n" for row in data))
    draw_ids = [f"source-{[2, 0, 3, 1, 4][i % 5]}" for i in range(64)]
    draws.write_text(json.dumps({"max_steps": 4, "effective_batch": 16,
                                "split_id": "synthetic-train-only-v1",
                                "token_rows_sha256": digest(tokens), "row_ids": draw_ids}))
    return ({"path": str(tokens), "sha256": digest(tokens)},
            {"path": str(draws), "sha256": digest(draws)})


def inspect(files):
    return inspect_training_data(*files, renderer_id="fixture-v1", max_sequence_tokens=16,
                                 max_steps=4, effective_batch=16, vocab_size=32)


def tokenizer():
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 32)}}
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
                                  bos_token="<s>", eos_token="</s>", pad_token="</s>", unk_token="<unk>")


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


def trainer(root, data, *, stop=False):
    class RecordingTrainer(sequential_sft_trainer_class()):
        def compute_loss(self, model, inputs, *args, **kwargs):
            self.observed_draws.extend(inputs["input_ids"][:, 1].tolist())
            return super().compute_loss(model, inputs, *args, **kwargs)

    hook = checkpoint_callback(identity=IDENTITY, archive_root=root / "archive", tokenizer=None,
                               light_every=1, full_every=2, evaluator=evaluation, evaluation_steps=[2])
    result = RecordingTrainer(
        model=make_model(), processing_class=tokenizer(), train_dataset=Dataset.from_list(data),
        data_collator=full_text_collator, callbacks=[hook] + ([StopAtTwo()] if stop else []),
        args=SFTConfig(output_dir=str(root / "trainer"), use_cpu=True, max_steps=4,
                       per_device_train_batch_size=2, gradient_accumulation_steps=8,
                       learning_rate=0.003, lr_scheduler_type="cosine", seed=77, data_seed=77,
                       bf16=False, fp16=False, gradient_checkpointing=False,
                       dataset_kwargs={"skip_prepare_dataset": True}, packing=False,
                       max_length=16, completion_only_loss=False,
                       save_strategy="steps", save_steps=2, save_total_limit=1,
                       logging_strategy="steps", logging_steps=1, report_to="none", disable_tqdm=True,
                       optim="adamw_torch", dataloader_num_workers=0),
    )
    result.observed_draws = []
    return result


class SFTDataTests(unittest.TestCase):
    def test_shared_pad_eos_id_has_distinct_loss_and_gradient_behavior(self):
        batch = full_text_collator([{"input_ids": [0, 5, 9, 1]}, {"input_ids": [0, 9, 10, 11, 1]}])
        self.assertEqual(batch["labels"].tolist(), [[0, 5, 9, 1, -100], [0, 9, 10, 11, 1]])
        logits = torch.zeros((2, 5, 32), requires_grad=True)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32),
                                                batch["labels"][:, 1:].reshape(-1), reduction="sum")
        loss.backward()
        self.assertLess(logits.grad[0, 2, 1].item(), 0)  # genuine EOS receives supervision
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)  # padding does not
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])

    def test_provenance_schedule_and_token_guards_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = fixture_files(root)
            data, draws, schedule, exposure = inspect(files)
            self.assertEqual(len(draws), 64)
            self.assertEqual(sum(x["draws"] for x in exposure), 64)
            self.assertEqual(sum(x["prompt_loss_tokens"] + x["target_loss_tokens"] for x in exposure),
                             sum(len(data[i]["input_ids"]) - 1 for i in draws))
            for key, value, error in (("split", "final", "Wrong split"),
                                      ("renderer_id", "old-renderer", "Wrong split"),
                                      ("input_ids", [0, 11, 1, 8, 1], "BOS/EOS"),
                                      ("input_ids", [0, 11, 8, 32, 1], "Invalid token"),
                                      ("id", "source-1", "duplicate")):
                bad = copy.deepcopy(rows())
                bad[0][key] = value
                with self.assertRaisesRegex(ValueError, error):
                    inspect(fixture_files(root, bad))
            files = fixture_files(root)
            Path(files[0]["path"]).write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                inspect(files)
            files = fixture_files(root)
            schedule["row_ids"][0] = "unknown-row"
            Path(files[1]["path"]).write_text(json.dumps(schedule))
            files[1]["sha256"] = digest(files[1]["path"])
            with self.assertRaisesRegex(ValueError, "absent training"):
                inspect(files)

    def test_real_sft_resume_consumes_frozen_draws_and_preserves_lora_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, indices, _, _ = inspect(fixture_files(root))
            draws = [{"input_ids": data[i]["input_ids"]} for i in indices]
            expected = [row["input_ids"][1] for row in draws]
            uninterrupted = trainer(root / "full", draws)
            uninterrupted.train()
            partial = trainer(root / "partial", draws, stop=True)
            partial.train()
            checkpoint = root / "partial/archive/full/checkpoint-2"
            verify_checkpoint(checkpoint, IDENTITY, require_full=True)
            resumed = trainer(root / "resumed", draws)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(uninterrupted.observed_draws, expected)
            self.assertEqual(partial.observed_draws, expected[:32])
            self.assertEqual(resumed.observed_draws, expected[32:])
            self.assertEqual(uninterrupted.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            for name, tensor in uninterrupted.model.state_dict().items():
                if "lora_" in name:
                    torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
            first = json.loads((root / "partial/archive/evaluations/step-2.json").read_text())
            last = json.loads((root / "resumed/archive/evaluations/step-4.json").read_text())
            self.assertEqual(first["status"], "evaluated")
            self.assertTrue(last["status"].startswith("archived:"))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
