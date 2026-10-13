"""CPU-only checks for the supervised RL entry and launcher boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from campaign_rl_entry import (
    DEFAULT_CUDA_MEMORY_FRACTION,
    DEVELOPMENT_EVALUATOR_FACTORY,
    ENTRY_SCHEMA_VERSION,
    MODEL_LOAD_MAX_SEQ_LENGTH,
    PARENT_MANIFEST_SCHEMA_VERSION,
    RLEntryError,
    assert_fresh_native_ext4_path,
    check_single_cuda_occupancy,
    main as entry_main,
    preflight_entry,
    run,
    _resolve_stop_reason,
    _observe_loaded_capacity,
    _validate_entry_contract,
    _validate_resume_checkpoint,
    _validate_schedule,
    verify_merged_parent_manifest,
    weight_inventory_sha256,
)
from campaign_rl_launch import entry_result_receipt_path, main as launch_main, supervised_command
from campaign_rl_train import TRAIN_SCHEMA_VERSION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParentFixture(unittest.TestCase):
    def parent(self, root: Path) -> tuple[dict, dict, Path]:
        model = root / "merged-sft"
        model.mkdir()
        (model / "config.json").write_text("{}\n")
        (model / "generation_config.json").write_text("{\"eos_token_id\": [1, 130073]}\n")
        (model / "tokenizer.json").write_text("{\"version\": 1}\n")
        (model / "tokenizer_config.json").write_text("{\"pad_token\": \"</s>\"}\n")
        weights = model / "model.safetensors"
        weights.write_bytes(b"merged-weight-fixture")
        inventory = [{"path": "model.safetensors", "bytes": weights.stat().st_size, "sha256": _sha(weights)}]
        tokenizer = {
            "tokenizer_json_sha256": _sha(model / "tokenizer.json"),
            "tokenizer_config_sha256": _sha(model / "tokenizer_config.json"),
            "vocab_size": 130560,
            "bos_id": 0,
            "eos_id": 1,
            "pad_id": 1,
            "native_eog_ids": [1, 130073],
        }
        manifest = {
            "schema_version": PARENT_MANIFEST_SCHEMA_VERSION,
            "status": "accepted",
            "kind": "merged_sft",
            "merged_model_path": str(model),
            "base_model_revision": "base-revision-fixture",
            "config_sha256": _sha(model / "config.json"),
            "generation_config_sha256": _sha(model / "generation_config.json"),
            "merged_weights_sha256": weight_inventory_sha256(inventory),
            "weight_inventory": inventory,
            "weight_inventory_sha256": weight_inventory_sha256(inventory),
            "tokenizer": tokenizer,
            "sft_identity": {"recipe_id": "sft-fixture-v1", "steps": 50},
        }
        path = root / "merged-sft-parent.json"
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        wrapper = {"path": str(path), "sha256": _sha(path)}
        parent_identity = {
            "kind": "merged_sft",
            "manifest_sha256": wrapper["sha256"],
            "merged_weights_sha256": manifest["merged_weights_sha256"],
            "base_model_revision": manifest["base_model_revision"],
            "sft_identity": manifest["sft_identity"],
        }
        return wrapper, parent_identity, model

    def test_parent_manifest_binds_every_weight_and_tokenizer_file(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper, _identity, model = self.parent(Path(directory))
            audit = verify_merged_parent_manifest(wrapper)
            self.assertEqual(audit["status"], "verified")
            self.assertEqual(audit["model_path"], str(model.resolve()))
            (model / "config.json").write_text("{\"drifted\": true}\n")
            with self.assertRaisesRegex(RLEntryError, "config.json SHA256"):
                verify_merged_parent_manifest(wrapper)
            (model / "config.json").write_text("{}\n")
            (model / "generation_config.json").write_text("{\"eos_token_id\": [1]}\n")
            with self.assertRaisesRegex(RLEntryError, "generation_config.json SHA256"):
                verify_merged_parent_manifest(wrapper)
            (model / "generation_config.json").write_text("{\"eos_token_id\": [1, 130073]}\n")
            (model / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(RLEntryError, "weight (byte count|SHA256) mismatch"):
                verify_merged_parent_manifest(wrapper)

    def test_parent_midtrain_and_incomplete_inventory_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, model = self.parent(root)
            manifest_path = Path(wrapper["path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["kind"] = "minicpm_midtrain"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            wrapper["sha256"] = _sha(manifest_path)
            with self.assertRaisesRegex(RLEntryError, "kind=merged_sft"):
                verify_merged_parent_manifest(wrapper)
            # Restore the accepted kind and list a file that does not exist.
            manifest["kind"] = "merged_sft"
            manifest["weight_inventory"].append({"path": "second.safetensors", "bytes": 1, "sha256": "0" * 64})
            manifest["weight_inventory"] = sorted(manifest["weight_inventory"], key=lambda item: item["path"])
            manifest["merged_weights_sha256"] = weight_inventory_sha256(manifest["weight_inventory"])
            manifest["weight_inventory_sha256"] = manifest["merged_weights_sha256"]
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            wrapper["sha256"] = _sha(manifest_path)
            with self.assertRaisesRegex(RLEntryError, "absent"):
                verify_merged_parent_manifest(wrapper)


class EntryBoundaryTests(ParentFixture):
    def _mock_admission(self, root: Path, parent_identity: dict, wrapper: dict):
        output = root / "output"
        archive = root / "archive"
        telemetry = root / "telemetry.jsonl"
        data_identity = {
            "rows_sha256": "a" * 64, "context_sha256": "b" * 64,
            "sidecar_artifact_sha256": "b" * 64, "selected_ids_sha256": "c" * 64,
            "ordered_ids_sha256": "d" * 64, "row_identity_sha256": "e" * 64,
            "row_identities": [{"id": "r1"}], "selected_ids": ["r1"],
            "row_count": 1, "split": "train", "admission_status": "admitted",
        }
        identity = {
            "parent": parent_identity,
            "tokenizer": {"revision": "rev", "json_sha256": "f" * 64,
                          "config_sha256": "0" * 64, "vocab_size": 130560,
                          "bos_id": 0, "eos_id": 1, "pad_id": 1,
                          "native_eog_ids": [1, 130073]},
            "renderer": {"renderer_id": "zeta2-prm03-v1",
                         "tokenization_policy": "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1",
                         "terminal": ">>>>>>> UPDATED", "no_edit": "[NO_EDIT]"},
            "data": data_identity,
            "source": {"protocol_sha256": "1" * 64, "rl_trainer_sha256": "2" * 64, "trl_version": "0.24.0"},
            "policy": {"lora_rank": 16, "lora_alpha": 16,
                       "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "expected_attachments": 294, "expected_trainable_parameters": 25116672,
                       "prompt_max_tokens": 2048, "completion_max_tokens": 192,
                       "context_max_tokens": 2240, "model_load_max_seq_length": 4096,
                       "cuda_memory_fraction": DEFAULT_CUDA_MEMORY_FRACTION,
                       "candidate_count": 2, "rollout_rows_per_update": 2,
                       "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                       "loss_type": "bnpo", "scale_rewards": "group", "beta": 0,
                       "sampling": {"do_sample": True, "temperature": 0.7, "top_p": 0.95,
                                    "repetition_penalty": 1.0}},
            "schedule": {"seed": 3407, "sampler_id": "campaign-repeat-manifest-order-v1",
                         "generation_batch_size": 2, "steps_per_generation": 1, "num_iterations": 1},
        }
        geometry = SimpleNamespace(candidate_count=2, prompt_groups_per_update=1,
                                   generation_batch_size=2, per_device_train_batch_size=2,
                                   gradient_accumulation_steps=1, steps_per_generation=1,
                                   to_dict=lambda: {
            "candidate_count": 2, "rollout_rows_per_update": 2, "prompt_groups_per_update": 1,
            "per_device_train_batch_size": 2, "generation_batch_size": 2,
            "gradient_accumulation_steps": 1, "steps_per_generation": 1, "num_iterations": 1,
        })
        manifest = SimpleNamespace(to_identity=lambda: data_identity, selected_ids=("r1",))
        records = [SimpleNamespace(row={"id": "r1"})]
        recipe = {
            "entry_schema_version": ENTRY_SCHEMA_VERSION,
            "schema_version": TRAIN_SCHEMA_VERSION,
            "identity": identity,
            "parent_manifest": wrapper,
            "data": {"rows_path": str(root / "rows.jsonl"), "rows_sha256": "a" * 64,
                     "sidecar_path": str(root / "sidecar.jsonl"), "sidecar_artifact_sha256": "b" * 64,
                     "selected_ids_path": str(root / "selected.json"), "selected_ids_sha256": "c" * 64,
                     "admission_status": "admitted"},
            "output_dir": str(output), "archive_root": str(archive), "telemetry_path": str(telemetry),
            "model_load_max_seq_length": 4096,
            "cuda_memory_fraction": DEFAULT_CUDA_MEMORY_FRACTION,
            "max_steps": 1, "full_save_steps": 1, "light_save_steps": 1,
            "evaluation_steps": [1], "decision_steps": [1],
            "checkpoint_reserve_seconds": 10, "max_attempt_seconds": 100,
            "termination_grace_seconds": 5, "deadline": "2099-01-01T00:00:00Z",
            "allow_pending_evaluation": True,
            "generation_kwargs": {"do_sample": True, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.0},
        }
        return recipe, identity, geometry, records, manifest

    def test_entry_binds_4k_model_load_and_frozen_rl_context_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, identity, _geometry, _records, _manifest = self._mock_admission(
                root, parent_identity, wrapper,
            )
            contract = _validate_entry_contract(recipe, identity)
            self.assertEqual(contract["model_load_max_seq_length"], MODEL_LOAD_MAX_SEQ_LENGTH)
            self.assertEqual(contract["cuda_memory_fraction"], DEFAULT_CUDA_MEMORY_FRACTION)
            self.assertEqual(contract["rl_prompt_max_tokens"], 2048)
            self.assertEqual(contract["rl_completion_max_tokens"], 192)
            self.assertEqual(contract["rl_context_max_tokens"], 2240)

            identity["policy"]["context_max_tokens"] = 2048
            with self.assertRaisesRegex(RLEntryError, "context_max_tokens"):
                _validate_entry_contract(recipe, identity)
            identity["policy"]["context_max_tokens"] = 2240
            recipe["model_load_max_seq_length"] = 2240
            with self.assertRaisesRegex(RLEntryError, "model_load_max_seq_length"):
                _validate_entry_contract(recipe, identity)

    def test_cuda_memory_fraction_is_required_bounded_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, identity, _geometry, _records, _manifest = self._mock_admission(
                root, parent_identity, wrapper,
            )

            recipe.pop("cuda_memory_fraction")
            with self.assertRaisesRegex(RLEntryError, "recipe.cuda_memory_fraction"):
                _validate_entry_contract(recipe, identity)
            recipe["cuda_memory_fraction"] = DEFAULT_CUDA_MEMORY_FRACTION

            identity["policy"].pop("cuda_memory_fraction")
            with self.assertRaisesRegex(RLEntryError, "identity.policy.cuda_memory_fraction"):
                _validate_entry_contract(recipe, identity)
            identity["policy"]["cuda_memory_fraction"] = DEFAULT_CUDA_MEMORY_FRACTION

            for invalid in (float("nan"), float("inf"), 0, -0.1, 0.800001):
                recipe["cuda_memory_fraction"] = invalid
                with self.assertRaisesRegex(RLEntryError, "finite and in"):
                    _validate_entry_contract(recipe, identity)
            recipe["cuda_memory_fraction"] = DEFAULT_CUDA_MEMORY_FRACTION

            for invalid in (float("nan"), 0, 0.800001):
                identity["policy"]["cuda_memory_fraction"] = invalid
                with self.assertRaisesRegex(RLEntryError, "identity.policy.cuda_memory_fraction"):
                    _validate_entry_contract(recipe, identity)
            identity["policy"]["cuda_memory_fraction"] = DEFAULT_CUDA_MEMORY_FRACTION

            recipe["cuda_memory_fraction"] = 0.8
            with self.assertRaisesRegex(RLEntryError, "must equal"):
                _validate_entry_contract(recipe, identity)

    def test_development_evaluator_wrapper_binds_panel_ids_renderer_and_4k_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, identity, _geometry, _records, _manifest = self._mock_admission(
                root, parent_identity, wrapper,
            )
            panel = root / "development.jsonl"
            panel.write_text(
                json.dumps({"id": "dev-1", "split": "dev", "package_id": "pkg-1", "family": "finish_block"})
                + "\n"
                + json.dumps({"id": "dev-2", "split": "dev", "package_id": "pkg-2", "family": "roxygen_drafting"})
                + "\n"
            )
            recipe.update({
                "evaluator_factory": DEVELOPMENT_EVALUATOR_FACTORY,
                "renderer_id": "zeta2-prm03-v1",
                "parameters": {"max_sequence_tokens": 4096},
                "development_panel": {"path": str(panel), "sha256": _sha(panel)},
                "development_case_ids": ["dev-1", "dev-2"],
                "development_max_new_tokens": 192,
            })
            result = _validate_entry_contract(recipe, identity)
            self.assertEqual(result["development_evaluator"]["rows"], 2)
            self.assertEqual(result["development_evaluator"]["parameters"]["max_sequence_tokens"], 4096)

            recipe["development_max_new_tokens"] = 512
            with self.assertRaisesRegex(RLEntryError, "development_max_new_tokens"):
                _validate_entry_contract(recipe, identity)
            recipe["development_max_new_tokens"] = 192
            recipe["development_case_ids"] = ["dev-1"]
            with self.assertRaisesRegex(RLEntryError, "panel IDs"):
                _validate_entry_contract(recipe, identity)

    def test_explicit_evaluation_steps_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, _identity, geometry, _records, _manifest = self._mock_admission(
                root, parent_identity, wrapper,
            )
            recipe.pop("evaluation_steps")
            with self.assertRaisesRegex(RLEntryError, "explicitly declared"):
                _validate_schedule(recipe, geometry)

    def test_loaded_capacity_is_recorded_and_rejects_short_model(self):
        model = SimpleNamespace(
            max_seq_length=4096,
            config=SimpleNamespace(max_position_embeddings=4096),
        )
        result = _observe_loaded_capacity(model, MODEL_LOAD_MAX_SEQ_LENGTH)
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["observed_capacity_tokens"], 4096)
        with self.assertRaisesRegex(RLEntryError, "below requested"):
            _observe_loaded_capacity(
                SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2240)),
                MODEL_LOAD_MAX_SEQ_LENGTH,
            )

    def test_loader_passes_4k_capacity_and_records_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = {}
            events = []
            model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=4096),
            )
            tokenizer = object()

            class FastLanguageModel:
                @staticmethod
                def from_pretrained(**kwargs):
                    events.append("model_allocation")
                    loaded.update(kwargs)
                    return model, tokenizer

            fake_torch = ModuleType("torch")
            fake_torch.bfloat16 = object()
            fake_torch.cuda = SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                set_per_process_memory_fraction=lambda fraction, device: events.append(
                    ("set_allocator_fraction", fraction, device)
                ),
                get_device_properties=lambda device: (
                    events.append("allocator_properties")
                    or SimpleNamespace(total_memory=16 * 1024**3)
                ),
            )
            fake_unsloth = ModuleType("unsloth")
            fake_unsloth.FastLanguageModel = FastLanguageModel
            fake_contract = ModuleType("campaign_tokenizer_contract")
            fake_contract.load_pinned_reference_tokenizer = lambda _path: object()
            fake_contract.restore_pinned_tokenizer_contract = lambda *_args, **_kwargs: {"status": "verified"}
            with mock.patch.dict("sys.modules", {
                "torch": fake_torch,
                "unsloth": fake_unsloth,
                "campaign_tokenizer_contract": fake_contract,
            }):
                result = __import__("campaign_rl_entry")._load_live_model(
                    {"model_path": str(root / "merged-sft")}, [],
                    model_load_max_seq_length=MODEL_LOAD_MAX_SEQ_LENGTH,
                )
            self.assertEqual(loaded["max_seq_length"], 4096)
            self.assertIs(loaded["dtype"], fake_torch.bfloat16)
            self.assertFalse(loaded["load_in_4bit"])
            self.assertFalse(loaded["trust_remote_code"])
            self.assertEqual(result[4]["model_load"]["observed_capacity_tokens"], 4096)
            self.assertEqual(
                result[4]["cuda_allocator"],
                {
                    "cuda_allocator_fraction": DEFAULT_CUDA_MEMORY_FRACTION,
                    "cuda_allocator_total_bytes": 16 * 1024**3,
                    "cuda_allocator_cap_bytes": int(16 * 1024**3 * DEFAULT_CUDA_MEMORY_FRACTION),
                    "device_index": 0,
                },
            )
            self.assertEqual(events[0], ("set_allocator_fraction", 0.75, 0))
            self.assertEqual(events[1], "allocator_properties")
            self.assertEqual(events[2], "model_allocation")

    def test_resume_sampler_state_requires_identity_cursor_and_complete_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, identity, geometry, _records, manifest = self._mock_admission(
                root, parent_identity, wrapper,
            )
            identity["schedule"].update({
                "source_draw_schedule_sha256": "s" * 64,
                "source_draw_sequence_sha256": "q" * 64,
                "source_draws": 2,
                "source_draws_per_update": 1,
                "buffer_reuse": 1,
            })
            schedule = {"max_steps": 2}
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            sampler = {
                "sampler_id": "campaign-repeat-manifest-order-v1", "seed": 3407, "shuffle": False,
                "num_samples": 1, "candidate_count": 2, "prompt_groups_per_batch": 1,
                "repeat_count": 1, "consumed_rows": 2, "current_index": 2,
                "consumed_prompt_copies": 1, "selected_id_index": 0, "epoch": 0,
                "source_draws_bound": True, "source_draws": 2, "source_draw_cursor": 1,
                "buffer_reuse": 1, "source_draw_sequence_sha256": "q" * 64,
                "source_draw_schedule_sha256": "s" * 64,
                "geometry": {"generation_batch_size": 2, "per_device_train_batch_size": 2,
                              "gradient_accumulation_steps": 1, "steps_per_generation": 1},
                "generation_rows_per_update": 2, "sampler_rows_per_update": 2,
                "data_identity": identity["data"],
            }
            (checkpoint / "campaign-state.json").write_text(json.dumps({
                "identity": identity, "step": 1, "full": True, "sampler": sampler,
            }))
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
            resume_manifest = {"step": 1, "full": True}
            source_draw_schedule = {"row_ids": ["r1", "r1"]}
            result = _validate_resume_checkpoint(
                checkpoint, resume_manifest, identity, geometry, schedule,
                data_identity=manifest.to_identity(), source_draw_schedule=source_draw_schedule,
            )
            self.assertEqual(result["status"], "verified")
            sampler["source_draw_cursor"] = 0
            (checkpoint / "campaign-state.json").write_text(json.dumps({
                "identity": identity, "step": 1, "full": True, "sampler": sampler,
            }))
            with self.assertRaisesRegex(RLEntryError, "source_draw_cursor"):
                _validate_resume_checkpoint(
                    checkpoint, resume_manifest, identity, geometry, schedule,
                    data_identity=manifest.to_identity(), source_draw_schedule=source_draw_schedule,
                )
            sampler["source_draw_cursor"] = 1
            sampler["consumed_rows"] = 1
            (checkpoint / "campaign-state.json").write_text(json.dumps({
                "identity": identity, "step": 1, "full": True, "sampler": sampler,
            }))
            with self.assertRaisesRegex(RLEntryError, "consumed_rows"):
                _validate_resume_checkpoint(
                    checkpoint, resume_manifest, identity, geometry, schedule,
                    data_identity=manifest.to_identity(), source_draw_schedule=source_draw_schedule,
                )

    def test_preflight_requires_merged_parent_wrapper_and_stays_framework_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, identity, geometry, records, manifest = self._mock_admission(root, parent_identity, wrapper)
            before = set(__import__("sys").modules)
            train_packet = {"identity": identity, "geometry": geometry.to_dict()}
            with mock.patch("campaign_rl_entry.preflight_rl_recipe", return_value=train_packet), \
                 mock.patch("campaign_rl_entry.load_training_records", return_value=(records, manifest)):
                result = preflight_entry(recipe)
            self.assertEqual(result["status"], "preflight_pass")
            self.assertEqual(result["runtime"]["model_load_max_seq_length"], 4096)
            self.assertEqual(result["runtime"]["rl_context_max_tokens"], 2240)
            after = set(__import__("sys").modules)
            self.assertFalse(
                {"torch", "transformers", "trl", "unsloth", "datasets"} & (after - before)
            )
            recipe.pop("parent_manifest")
            with mock.patch("campaign_rl_entry.preflight_rl_recipe", return_value=train_packet), \
                 mock.patch("campaign_rl_entry.load_training_records", return_value=(records, manifest)):
                with self.assertRaisesRegex(RLEntryError, "parent_manifest"):
                    preflight_entry(recipe)

    def test_resume_requires_an_existing_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper, parent_identity, _model = self.parent(root)
            recipe, identity, geometry, records, manifest = self._mock_admission(root, parent_identity, wrapper)
            recipe["resume_from"] = str(root / "missing-checkpoint")
            with mock.patch("campaign_rl_entry.preflight_rl_recipe", return_value={"identity": identity, "geometry": geometry.to_dict()}), \
                 mock.patch("campaign_rl_entry.load_training_records", return_value=(records, manifest)):
                with self.assertRaisesRegex(RLEntryError, "existing directory"):
                    preflight_entry(recipe)

    def test_native_ext4_and_occupancy_gates_reject_network_or_busy_lane(self):
        if Path("/mnt/h").is_dir():
            with self.assertRaisesRegex(RLEntryError, "native ext4"):
                assert_fresh_native_ext4_path("/mnt/h/sepalith-rl-test-new", "output_dir")
        class Result:
            stdout = "9000\n"
        with self.assertRaisesRegex(RLEntryError, "exceeds"):
            check_single_cuda_occupancy(runner=lambda *args, **kwargs: Result())
        class Good:
            stdout = "42\n"
        self.assertEqual(check_single_cuda_occupancy(runner=lambda *args, **kwargs: Good())["gpu_count"], 1)

    def test_launcher_binds_soft_and_hard_deadlines_with_checkpoint_reserve(self):
        recipe = {
            "deadline": "1970-01-01T00:01:40Z",
            "max_attempt_seconds": 100,
            "termination_grace_seconds": 5,
            "checkpoint_reserve_seconds": 10,
        }
        argv, record = supervised_command(recipe, ["entry.py", "recipe.json"], now=70)
        self.assertEqual(record["soft_deadline"], "1970-01-01T00:01:35+00:00")
        self.assertEqual(record["hard_deadline"], "1970-01-01T00:01:40+00:00")
        self.assertIn("25.000000s", argv)
        with self.assertRaisesRegex(RLEntryError, "no attempt budget"):
            supervised_command(recipe, ["entry.py"], now=85)


class RunFlowTests(unittest.TestCase):
    def test_stop_reason_skips_default_callbacks_and_requires_terminal_step(self):
        trainer = SimpleNamespace(
            callbacks=[SimpleNamespace(), SimpleNamespace(stop_reason="lead_decision")],
            callback_handler=SimpleNamespace(callbacks=[SimpleNamespace(stop_reason="deadline")]),
        )
        self.assertEqual(_resolve_stop_reason(trainer, 10, 20), "lead_decision")
        trainer.callbacks = [SimpleNamespace()]
        trainer.callback_handler.callbacks = []
        self.assertEqual(_resolve_stop_reason(trainer, 20, 20), "schedule_complete")
        with self.assertRaisesRegex(RLEntryError, "before max_steps"):
            _resolve_stop_reason(trainer, 19, 20)

    def test_stop_reason_classifies_deadline_from_handler(self):
        trainer = SimpleNamespace(
            callbacks=[SimpleNamespace()],
            callback_handler=SimpleNamespace(callbacks=[SimpleNamespace(stop_reason="deadline")]),
        )
        self.assertEqual(_resolve_stop_reason(trainer, 7, 20), "deadline")

    def test_run_trains_fresh_attempt_and_verifies_terminal_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {
                "output_dir": str(root / "output"), "archive_root": str(root / "archive"),
                "telemetry_path": str(root / "telemetry.jsonl"),
                "deadline": "2099-01-01T00:00:00Z", "checkpoint_reserve_seconds": 10,
                "resume_from": None, "identity": {"parent": {"kind": "merged_sft"}},
                "data": {"rows_path": str(root / "rows.jsonl"), "rows_sha256": "a" * 64,
                         "sidecar_path": str(root / "sidecar.jsonl"), "sidecar_artifact_sha256": "b" * 64,
                         "selected_ids_path": str(root / "selected.json"), "selected_ids_sha256": "c" * 64,
                         "admission_status": "admitted"},
            }
            admission = {
                "parent": {"manifest_sha256": "a" * 64, "merged_weights_sha256": "b" * 64},
                "data": {}, "geometry": {}, "schedule": {"checkpoint_reserve_seconds": 10, "max_steps": 1},
                "runtime": {"model_load_max_seq_length": MODEL_LOAD_MAX_SEQ_LENGTH,
                            "cuda_memory_fraction": DEFAULT_CUDA_MEMORY_FRACTION},
            }
            class Reward:
                event_sink = None
                def __call__(self, *args, **kwargs): return []
            class Trainer:
                _campaign_reward = Reward()
                _campaign_post_trainer_contract = {"status": "verified"}
                processing_class = object()
                state = SimpleNamespace(global_step=1)
                def __init__(self): self.callbacks = []
                def add_callback(self, callback): self.callbacks.append(callback)
                def train(self, *, resume_from_checkpoint):
                    self.resume = resume_from_checkpoint
                    control = SimpleNamespace()
                    for callback in self.callbacks:
                        callback.on_train_begin(None, self.state, control)
            trainer = Trainer()
            fake_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="bfloat16")]))
            fast_model = SimpleNamespace(for_training=lambda **kwargs: None)
            records = [SimpleNamespace(row={"id": "r1"})]
            manifest = SimpleNamespace(to_identity=lambda: {}, selected_ids=("r1",))
            env = {
                "SEPALITH_CAMPAIGN_SOFT_DEADLINE": "2098-12-31T00:00:00+00:00",
                "SEPALITH_CAMPAIGN_HARD_DEADLINE": "2098-12-31T00:10:00+00:00",
            }
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch("campaign_rl_entry.preflight_entry", return_value=admission), \
                 mock.patch("campaign_rl_entry.check_single_cuda_occupancy", return_value={"gpu_count": 1}), \
                 mock.patch("campaign_rl_entry.load_training_records", return_value=(records, manifest)), \
                 mock.patch("campaign_rl_entry._load_live_model", return_value=(fake_model, object(), object(), fast_model, {"status": "verified"})), \
                 mock.patch("campaign_rl_entry._attach_rl_adapter", return_value=(fake_model, {"trainable_parameters": 25116672})), \
                 mock.patch("campaign_rl_entry.build_live_trainer", return_value=trainer) as build_trainer, \
                 mock.patch("campaign_sft.assert_post_trainer_pinned_identity", return_value={"status": "verified"}), \
                 mock.patch("campaign_sft.restore_trainer_eog_alignment", return_value={"status": "verified"}), \
                 mock.patch("campaign_rl_entry.verify_checkpoint", return_value={"full": True}):
                result = run(recipe)
            self.assertEqual(result["status"], "schedule_complete")
            self.assertEqual(trainer.resume, None)
            self.assertEqual(
                build_trainer.call_args.args[0]["supervised_deadline"],
                env["SEPALITH_CAMPAIGN_SOFT_DEADLINE"],
            )
            self.assertTrue((root / "output" / "terminal.json").is_file())
            self.assertTrue((root / "output" / "train-begin-tokenizer-contract.json").is_file())
            # The callback is installed before train; the fake invokes the
            # same train-begin boundary used by a real Trainer.
            self.assertEqual(len(trainer.callbacks), 1)

    def test_entry_success_and_failure_write_distinct_result_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipe.json"
            recipe.write_text("{}\n")
            supervision = root / "supervision.json"
            supervision.write_text('{"status": "supervisor_configured", "pid": 17}\n')
            result_receipt = entry_result_receipt_path(supervision)
            env = {"SEPALITH_CAMPAIGN_LAUNCH_RECEIPT": str(supervision)}
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch("campaign_rl_entry.run", return_value={"status": "schedule_complete"}):
                self.assertEqual(entry_main([str(recipe), "--receipt", str(result_receipt)]), 0)
            self.assertEqual(json.loads(supervision.read_text())["pid"], 17)
            self.assertEqual(json.loads(result_receipt.read_text())["status"], "schedule_complete")

            failed_result = root / "failed-entry.json"
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch("campaign_rl_entry.run", side_effect=RLEntryError("synthetic failure")):
                self.assertEqual(entry_main([str(recipe), "--receipt", str(failed_result)]), 2)
            self.assertEqual(json.loads(supervision.read_text())["status"], "supervisor_configured")
            self.assertEqual(json.loads(failed_result.read_text())["status"], "entry_failed")

    def test_launcher_records_distinct_entry_result_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipe.json"
            recipe.write_text("{}\n")
            supervision = root / "supervision.json"
            entry_result = entry_result_receipt_path(supervision)
            argv = ["/usr/bin/timeout", "10s", "entry.py", "--receipt", str(entry_result)]
            preflight = {"status": "preflight_pass", "parent": {"manifest_sha256": "a" * 64},
                         "data": {}, "geometry": {}}
            supervision_record = {"soft_deadline": "2099-01-01T00:00:00+00:00",
                                  "hard_deadline": "2099-01-01T00:01:00+00:00",
                                  "checkpoint_reserve_seconds": 10}
            with mock.patch("campaign_rl_launch.preflight_file", return_value=preflight), \
                 mock.patch("campaign_rl_launch.supervised_command", return_value=(argv, supervision_record)), \
                 mock.patch("campaign_rl_launch.os.execve", side_effect=SystemExit(0)):
                with self.assertRaises(SystemExit):
                    launch_main([str(recipe), "--receipt", str(supervision)])
            saved = json.loads(supervision.read_text())
            self.assertEqual(saved["entry_result_receipt"], str(entry_result))
            self.assertEqual(saved["argv"], argv)


if __name__ == "__main__":
    unittest.main()
