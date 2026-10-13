"""CPU-only exact admission and toy Trainer resume checks."""
from __future__ import annotations
import copy, datetime, importlib.util, json, os, sys, tempfile, unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SOURCE = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-full15006-v1/attempts/28a06420683d4035b60941347955bb3e/source")
CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-full15006-a/full/checkpoint-480")
sys.path.insert(0, str(SOURCE / "experiments/training"))
import campaign_checkpoint
import campaign_control
import campaign_expanded_sft as active_expanded
import campaign_sft
import torch
torch.set_num_threads(2)

spec = importlib.util.spec_from_file_location("patched_expanded", WORK / "source-patch/experiments/training/campaign_expanded_sft.py")
patched_expanded = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(patched_expanded)


class ExactAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb") /
                                  "docs/campaign/work/lead/r2-expanded-sft-full15008-v1/recipe.json").read_text())

    @staticmethod
    def call_validator(module, recipe):
        old_rows, old_parent = module.validate_expanded_rows, module.validate_sft_merged_parent
        module.validate_expanded_rows = lambda _recipe, _rows: {"rows": 15006}
        module.validate_sft_merged_parent = lambda _recipe: None
        try:
            return module.validate_expanded_admission(recipe, [])
        finally:
            module.validate_expanded_rows, module.validate_sft_merged_parent = old_rows, old_parent

    def test_active_source_rejects_exact_full480_due_to_legacy_step_gate(self):
        recipe = copy.deepcopy(self.recipe)
        recipe.update(resume_from=str(CHECKPOINT), resume_milestone=480)
        with self.assertRaisesRegex(ValueError, "resume step/full identity"):
            self.call_validator(active_expanded, recipe)

    def test_active_compatibility_route_then_reads_missing_schedule_split_id(self):
        recipe = json.loads((WORK / "resume480-proposed-recipe.json").read_text())
        with self.assertRaises(KeyError):
            self.call_validator(active_expanded, recipe)

    def test_patch_accepts_exact_full480_and_all_full_cadences(self):
        recipe = copy.deepcopy(self.recipe)
        steps = list(range(40, 1160, 40))
        recipe["target_resume_steps"] = steps
        recipe["target_gate_start_steps"] = [0] + steps
        recipe.update(resume_from=str(CHECKPOINT), resume_milestone=480)
        self.assertTrue(patched_expanded.expanded_target_only_policy(recipe))
        self.assertEqual(self.call_validator(patched_expanded, recipe)["rows"], 15006)
        # A reviewed source migration must preserve the same all-cadence startup gates.
        manifest_hash = __import__("hashlib").sha256((CHECKPOINT / "campaign-manifest.json").read_bytes()).hexdigest()
        source_id = recipe["identity"]["source"]
        recipe["resume_identity_compatibility"] = {
            "schema": "sepalith.sft.resume-source-compatibility.v1", "mode": "exact_same_source",
            "reason": "external_interruption", "checkpoint_step": 480,
            "checkpoint_manifest_sha256": manifest_hash,
            "predecessor_source": source_id, "current_source": source_id,
        }
        recipe["mandatory_stop_steps"] = [720]
        self.assertTrue(patched_expanded.expanded_target_only_policy(recipe))

    def test_patch_rejects_incomplete_cadence_and_adapter_only(self):
        recipe = copy.deepcopy(self.recipe)
        steps = list(range(40, 1160, 40))
        recipe["target_resume_steps"] = steps[:-1]
        recipe["target_gate_start_steps"] = [0] + steps[:-1]
        with self.assertRaisesRegex(ValueError, "every pre-terminal full cadence"):
            patched_expanded.expanded_target_only_policy(recipe)
        adapter = CHECKPOINT.parents[1] / "adapters/checkpoint-480"
        if adapter.exists():
            with self.assertRaisesRegex(ValueError, "lightweight adapter"):
                campaign_checkpoint.verify_checkpoint(adapter, self.recipe["identity"], require_full=True)


from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers import PretrainedConfig
from trl import SFTConfig
from campaign_sft_data import target_only_collator


def rows(count=1296):
    return [{"input_ids": [0, 2 + i, 20, 31, 41, 42, 1], "target_start": 3,
             "target_body_tokens": [31], "target_terminal_tokens": [41, 42]} for i in range(count)]


def tokenizer():
    tok = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="63"))
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="0", eos_token="1", pad_token="1", unk_token="63")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(0.1))
        self.config = PretrainedConfig(vocab_size=64, bos_token_id=0, eos_token_id=1, pad_token_id=1)
        self.seen = []

    def forward(self, input_ids, **_kwargs):
        self.seen.extend(input_ids[:, 1].tolist())
        target = input_ids[:, 1].float().mean() / 20000
        # Random multiplier makes exact terminal equality depend on RNG restore.
        loss = (self.scalar - target).square() * (0.5 + torch.rand(()))
        return {"loss": loss, "logits": None}


def make_trainer(path, model):
    return campaign_sft.sequential_sft_trainer_class()(
        model=model, processing_class=tokenizer(), train_dataset=Dataset.from_list(rows()),
        data_collator=target_only_collator,
        args=SFTConfig(output_dir=str(path), use_cpu=True, bf16=False, fp16=False,
            gradient_checkpointing=False, per_device_train_batch_size=2, gradient_accumulation_steps=8,
            max_steps=81, learning_rate=.001, warmup_ratio=.03, lr_scheduler_type="cosine",
            optim="adamw_torch", seed=3407, data_seed=3407, remove_unused_columns=False,
            dataloader_num_workers=0, dataset_kwargs={"skip_prepare_dataset": True}, packing=False,
            completion_only_loss=False, max_length=4096, save_steps=40, save_strategy="steps",
            save_only_model=False, ignore_data_skip=False, logging_strategy="no", report_to="none",
            disable_tqdm=True))


class ToyContinuityTest(unittest.TestCase):
    def test_interrupted40_resume_to81_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            base = Path(tmp)
            baseline_model = Tiny()
            baseline = make_trainer(base / "baseline", baseline_model)
            baseline.train()
            first_model = Tiny()
            first = make_trainer(base / "first", first_model)
            first.add_callback(campaign_control.control_callback(
                telemetry_path=base / "first-telemetry.jsonl", identity={"synthetic_cpu_resume_only": True},
                deadline=(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)).isoformat(),
                reserve_seconds=5, stop_steps=[40]))
            first.train()
            checkpoint40 = base / "first/checkpoint-40"
            self.assertEqual(first.state.global_step, 40)
            resumed_model = Tiny()
            resumed = make_trainer(base / "resumed", resumed_model)
            resumed.train(resume_from_checkpoint=str(checkpoint40))
            self.assertEqual(resumed.state.global_step, baseline.state.global_step, 81)
            self.assertEqual(first_model.seen, list(range(2, 642)))
            self.assertEqual(resumed_model.seen, list(range(642, 1298)))
            self.assertTrue(torch.equal(baseline_model.scalar, resumed_model.scalar))
            self.assertEqual(baseline.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            left, right = baseline.optimizer.state_dict(), resumed.optimizer.state_dict()
            self.assertEqual(left["param_groups"], right["param_groups"])
            for key, value in left["state"][0].items():
                self.assertTrue(torch.equal(value, right["state"][0][key]) if torch.is_tensor(value)
                                else value == right["state"][0][key], key)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                self.assertGreater((checkpoint40 / name).stat().st_size, 0)
            evidence = {"status": "PASS", "synthetic_only": True, "checkpoint_step": 40,
                        "terminal_step": 81, "effective_batch": 16, "microbatch": 2,
                        "gradient_accumulation": 8, "exact_draw_order": True,
                        "exact_parameter_optimizer_scheduler_equality": True,
                        "rng_sensitive_objective": True,
                        "limitations": ["Tiny CPU scalar model with adamw_torch, not MiniCPM LoRA or fused CUDA AdamW.",
                                        "This exercises the installed Trainer restore/data-skip path but does not prove full 5090 equivalence."]}
            (WORK / "cpu-toy-resume-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__": unittest.main(verbosity=2)
