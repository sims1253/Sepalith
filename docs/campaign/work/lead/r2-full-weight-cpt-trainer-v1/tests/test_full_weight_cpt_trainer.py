import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


PACKET = Path(__file__).resolve().parents[1]
TRAINING = PACKET / "source" / "experiments" / "training"
sys.path.insert(0, str(TRAINING))
import bind_full_weight_cpt as binder
import full_weight_cpt_trainer as trainer
import request_graceful_stop as stop_request
from trainer_tokenizer_alignment import restore_trainer_eog_alignment


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullWeightCptPreparationTests(unittest.TestCase):
    def valid_admission(self):
        common = json.loads(Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-full-weight-optimizer-v1/smoke-configs.json").read_text())
        optimizer = {**common["common"], **common["smoke_arms"]["aurora_mix"]}
        return {
            "schema": "sepalith.sft11.full-weight-cpt-root-admission.v1",
            "status": "admitted", "launch_authorized": True,
            "template_sha256": sha(PACKET / "recipe.template.json"),
            "selected": {
                "parent_candidate_id": "global_cpt250_merged", "optimizer": optimizer,
                "micro_batch": 1, "gradient_accumulation": 16,
                "learning_rate": 1e-4, "scheduler": "cosine", "warmup_ratio": 0.03,
                "checkpoint_every": 317, "evaluation_steps": [317, 951, 1585, 1902],
                "selected_milestones": [317, 951, 1585, 1902], "telemetry_every": 1,
            },
        }

    def test_root_admission_binds_effective_batch_and_scientific_selection(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); admission = root / "admission.json"; output = root / "bound.json"
            admission.write_text(json.dumps(self.valid_admission()))
            result = binder.bind(PACKET / "recipe.template.json", admission, output)
            self.assertEqual(result["runtime"]["effective_batch"], 16)
            self.assertEqual(result["runtime"]["max_steps"], 1902)
            self.assertEqual(result["parent"]["kind"], "merged_cpt_parent")

    def test_binding_rejects_wrong_effective_batch(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); admission = root / "admission.json"
            record = self.valid_admission(); record["selected"]["gradient_accumulation"] = 8
            admission.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "effective batch"):
                binder.bind(PACKET / "recipe.template.json", admission, root / "bound.json")

    def test_binding_rejects_evaluation_without_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); admission = root / "admission.json"
            record = self.valid_admission(); record["selected"]["evaluation_steps"] = [318, 1902]
            record["selected"]["selected_milestones"] = [318, 1902]
            admission.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "evaluation step lacks"):
                binder.bind(PACKET / "recipe.template.json", admission, root / "bound.json")

    def test_frozen_dataset_requires_every_unique_row_before_replay_and_preserves_masks(self):
        source = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-remaining-v1/cpt_train.jsonl")
        with source.open() as stream:
            rows = [json.loads(stream.readline()), json.loads(stream.readline())]
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); rows_path = root / "rows.jsonl"
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            draws = [row["row_id"] for row in rows] + [rows[0]["row_id"], rows[1]["row_id"]] * 7
            schedule = {"token_rows_sha256": sha(rows_path), "row_ids": draws}
            schedule_path = root / "draws.json"; schedule_path.write_text(json.dumps(schedule))
            recipe = {"cohort": {"rows": {"path": str(rows_path), "sha256": sha(rows_path)}, "draw_schedule": {"path": str(schedule_path)}, "max_sequence_tokens": 2048, "unique_rows": 2, "input_tokens": sum(len(r["input_ids"]) for r in rows), "loss_tokens": sum(sum(x != -100 for x in r["labels"][1:]) for r in rows), "updates": 1, "named_replays": 14}}
            dataset = trainer.FrozenTokenRowDataset(recipe)
            self.assertEqual(len(dataset), 16)
            self.assertEqual(dataset[0]["labels"], rows[0]["labels"])
            schedule["row_ids"][:2] = list(reversed(schedule["row_ids"][:2]))
            schedule_path.write_text(json.dumps(schedule))
            with self.assertRaisesRegex(ValueError, "every unique row"):
                trainer.FrozenTokenRowDataset(recipe)

    def test_malformed_target_mask_is_rejected(self):
        source = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-remaining-v1/cpt_train.jsonl")
        with source.open() as stream: row = json.loads(stream.readline())
        row["labels"][1] = -100
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); rows = root / "rows.jsonl"; rows.write_text(json.dumps(row) + "\n")
            schedule = root / "draws.json"; schedule.write_text(json.dumps({"token_rows_sha256": sha(rows), "row_ids": [row["row_id"]] * 16}))
            recipe = {"cohort": {"rows": {"path": str(rows)}, "draw_schedule": {"path": str(schedule)}, "max_sequence_tokens": 2048, "unique_rows": 1, "input_tokens": len(row["input_ids"]), "loss_tokens": 1, "updates": 1, "named_replays": 15}}
            with self.assertRaises(Exception): trainer.FrozenTokenRowDataset(recipe)

    def test_retention_preserves_named_milestones_and_latest_two(self):
        with tempfile.TemporaryDirectory() as value:
            full = Path(value) / "full"; full.mkdir()
            for step in (10, 20, 30, 40): (full / f"checkpoint-{step}").mkdir()
            result = trainer.retain_archives(Path(value), {10}, latest=2)
            self.assertEqual(result["preserved"], [10, 30, 40])
            self.assertEqual(result["retired"], [20])

    def test_actual_transformers_55_accepts_runtime_arguments(self):
        from transformers import TrainingArguments
        recipe = {"seed": 3407, "runtime": {"micro_batch": 1, "gradient_accumulation": 16, "max_steps": 1902, "learning_rate": 1e-4, "scheduler": "cosine", "warmup_ratio": 0.03, "checkpoint_every": 317}}
        with tempfile.TemporaryDirectory() as value:
            kwargs = trainer.training_arguments_kwargs(recipe, Path(value)); kwargs["use_cpu"] = True
            args = TrainingArguments(**kwargs)
        self.assertEqual(args.max_steps, 1902); self.assertEqual(args.save_total_limit, 2)
        self.assertEqual(args.warmup_steps, 57)

    def test_source_closure_imports_tokenizer_contract_and_protocol(self):
        code = "import campaign_tokenizer_contract as c; import sepalith.campaign_protocol as p; assert c.EOS_ID==p.EOS_ID==1"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=TRAINING, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_resume_and_external_stop_are_fail_closed_in_live_source(self):
        source = (TRAINING / "full_weight_cpt_trainer.py").read_text()
        self.assertIn('expected_checkpoint_kind="full_weights"', source)
        self.assertIn('"action": "save_and_stop", "bound_recipe_sha256": recipe_hash', source)
        self.assertIn('control.should_save = True; control.should_training_stop = True', source)
        self.assertLess(source.index('from unsloth import FastLanguageModel'), source.index('import torch', source.index('from unsloth import FastLanguageModel')))
        self.assertIn('restore_trainer_eog_alignment(model, tokenizer, reference)', source)
        self.assertIn('assert_serialized_tokenizer(source, reference', source)
        self.assertIn('verify_checkpoint(terminal, identity(recipe)', source)

    def test_known_trainer_alignment_repairs_only_observed_form(self):
        vocab = {f"t{i}": i for i in range(130560)}
        del vocab["t1"]; del vocab["t130000"]
        vocab["</s>"] = 1; vocab["<unused_token_477>"] = 130000
        class Tokenizer:
            bos_token_id=0; eos_token_id=1; eos_token="</s>"; pad_token_id=130000; _pad_token="<unused_token_477>"; padding_side="right"
            def __len__(self): return 130560
            def get_vocab(self): return dict(vocab)
            def convert_ids_to_tokens(self, value): return "</s>" if value == 1 else f"t{value}"
            @property
            def pad_token(self): return self._pad_token
            @pad_token.setter
            def pad_token(self, value): self._pad_token=value; self.pad_token_id=self.get_vocab()[value]
        class Model:
            config=SimpleNamespace(bos_token_id=0,eos_token_id=1,pad_token_id=130000)
            generation_config=SimpleNamespace(bos_token_id=0,eos_token_id=[1,1,130073],pad_token_id=130000)
            def modules(self): return []
        loaded, reference, model = Tokenizer(), Tokenizer(), Model(); reference.pad_token="</s>"
        report=restore_trainer_eog_alignment(model,loaded,reference)
        self.assertEqual(report["before"]["generation_eos"],[1,1,130073]); self.assertEqual(loaded.pad_token_id,1)
        model.generation_config.eos_token_id=[1,2,130073]
        with self.assertRaisesRegex(ValueError,"Unexpected Trainer EOS"): restore_trainer_eog_alignment(model,loaded,reference)

    def test_atomic_publish_exposes_only_complete_sealed_directory(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); source=root/"runtime"/"checkpoint-7"; destination=root/"archive"/"full"/"checkpoint-7"
            source.mkdir(parents=True); (source/"payload.bin").write_bytes(b"payload")
            manifest={"step":7,"checkpoint_kind":"full_weights","files":{"payload.bin":{"bytes":7,"sha256":hashlib.sha256(b"payload").hexdigest()}}}
            (source/"campaign-manifest.json").write_text(json.dumps(manifest))
            real_rename=os.rename
            def checked_rename(old,new):
                self.assertFalse(destination.exists()); self.assertTrue(source.exists()); real_rename(old,new)
            with mock.patch.object(trainer.os,"rename",side_effect=checked_rename):
                result=trainer.publish_sealed_checkpoint(source,destination,manifest)
            self.assertEqual(result,destination); self.assertFalse(source.exists()); self.assertEqual((destination/"payload.bin").read_bytes(),b"payload")

    def test_graceful_stop_request_is_exclusive_and_recipe_bound(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); recipe=root/"bound.json"; recipe.write_text('{"bound":true}\n')
            target=root/"stop.json"; bound={"outputs":{"graceful_stop":str(target)}}
            with mock.patch.object(stop_request,"load_bound",return_value=bound):
                result=stop_request.request(recipe)
                self.assertEqual(json.loads(target.read_text())["bound_recipe_sha256"],sha(recipe))
                with self.assertRaises(FileExistsError): stop_request.request(recipe)


if __name__ == "__main__": unittest.main(verbosity=2)
