"""CPU and pinned-TRL checks for the PRM-03 RL runner preparation."""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from campaign_rl_data import (
    CONTEXT_SCHEMA_VERSION,
    SELECTED_IDS_SCHEMA_VERSION,
    RLDataError,
    load_training_records,
    records_to_dataset_rows,
)
from campaign_rl_train import (
    BOS_ID,
    EOS_ID,
    NATIVE_EOG_IDS,
    CampaignPRM03Reward,
    CampaignRepeatSampler,
    DEFAULT_SEED,
    RLTrainError,
    TRAIN_SCHEMA_VERSION,
    line_f1,
    make_grpo_config,
    preflight_rl_recipe,
    DEFAULT_GENERATION_KWARGS,
    campaign_grpo_trainer_class,
    resolve_trl_geometry,
    load_source_draw_schedule,
    source_draw_sequence_sha256,
)
from sepalith.campaign_protocol import (
    Cursor,
    Position,
    PromptContext,
    ReplacementRange,
    RENDERER_ID,
    SCHEMA_VERSION,
    TOKENIZATION_POLICY,
    build_training_row,
)


class CharEncoder:
    def __init__(self):
        self.forward = {}
        self.reverse = {}
        self.next_id = 200

    def encode(self, text, *, add_special_tokens=False, split_special_tokens=True):
        del add_special_tokens, split_special_tokens
        ids = []
        for char in text:
            if char not in self.forward:
                self.forward[char] = self.next_id
                self.reverse[self.next_id] = char
                self.next_id += 1
            ids.append(self.forward[char])
        return ids

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.reverse[int(token)] for token in ids)


class RLFixtures(unittest.TestCase):
    def setUp(self):
        self.encoder = CharEncoder()
        self.snapshot = hashlib.sha256(b"immutable-source-snapshot").hexdigest()

    def context(self, row_id, region=("old",), *, version=7, empty=False):
        content = ("old\n" if region else "")
        content_sha = hashlib.sha256(content.encode()).hexdigest()
        if empty:
            region = ()
            start = end = Position(0, 0)
            cursor = Cursor(-1, None, None)
        else:
            start = Position(0, 0)
            end = Position(0, len(region[0]))
            cursor = Cursor(0, 1, 1)
        context = PromptContext(
            path=f"src/{row_id}.py",
            prefix=("def f():",),
            selected_references=(), history=(), diagnostics=(), retrieval=(),
            scope_mode="outline", scope_lines=("def f():",), suffix_lines=(),
            region_old=tuple(region), cursor=cursor,
            replacement_range=ReplacementRange(
                uri=f"file:///workspace/{row_id}.py", document_version=version,
                content_sha256=content_sha, start=start, end=end,
            ),
            document_eol="lf", schema_version=SCHEMA_VERSION,
        )
        return context

    def row(self, row_id, operation, region_new, *, empty=False):
        context = self.context(row_id, empty=empty)
        row = build_training_row(
            context, operation=operation, region_new=region_new,
            tokenizer=self.encoder, row_id=row_id, family="rename", package_id="pkg",
            split="train",
        )
        capture = {
            "uri": context.replacement_range.uri,
            "version": context.replacement_range.document_version,
            "content_sha256": context.replacement_range.content_sha256,
            "source_snapshot_sha256": self.snapshot,
        }
        context_record = {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "id": row_id,
            "context": context.to_dict(),
            "capture": capture,
        }
        return row, context, context_record

    def files(self, rows):
        root = Path(tempfile.mkdtemp())
        rows_path = root / "admitted.jsonl"
        context_path = root / "contexts.jsonl"
        selected_path = root / "selected.json"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row, _, _ in rows))
        context_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for _, _, record in rows))
        selected_path.write_text(json.dumps({
            "schema_version": SELECTED_IDS_SCHEMA_VERSION,
            "split": "train",
            "row_ids": [row["id"] for row, _, _ in rows],
        }, sort_keys=True))
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        return root, rows_path, context_path, selected_path, digest(rows_path), digest(context_path), digest(selected_path)

    def sidecar_files(self, rows):
        """Write the exact RL-02 sidecar shape, with per-row source hashes."""
        root = Path(tempfile.mkdtemp())
        rows_path = root / "admitted.jsonl"
        sidecar_path = root / "contexts-sidecar.jsonl"
        selected_path = root / "selected.json"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row, _, _ in rows))
        row_values = []
        for index, (row, context, _legacy_record) in enumerate(rows, 1):
            replacement = context.replacement_range
            source_bytes = f"static-source-{row['id']}".encode()
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            source_ref = {
                "row_id": row["id"], "group_id": f"group-{row['id']}",
                "split": "train_group", "file": str(root / f"source-{row['id']}.jsonl"),
                "line": index, "source_sha256": source_sha,
                "raw_line_sha256": hashlib.sha256(source_bytes + b"\n").hexdigest(),
                "family": row["family"], "source": "fixture", "source_variant": "fixture",
            }
            source_provenance = {
                "row_id": row["id"], "group_id": f"group-{row['id']}",
                "split": "train_group", "file": source_ref["file"], "line": index,
                "source_sha256": source_sha, "raw_line_sha256": source_ref["raw_line_sha256"],
                "after_snapshot_sha256": replacement.content_sha256,
            }
            selection = {
                "document_sha256": replacement.content_sha256,
                "availability": "full_snapshot", "policy_id": "fixture",
                "policy_id_combined": None, "budget_utf16_units": None,
                "used_utf16_units": None, "required_utf16_units": None,
                "overflow": False, "required_overflow": False,
                "spans": [{"kind": "region", "start_line": replacement.start.line,
                           "end_line": replacement.end.line}],
                "region": list(context.region_old),
                "context_range": replacement.to_dict(),
                "document_version_policy": (
                    "offline_static_source; zero is valid; no live-editor freshness asserted"
                ),
            }
            row_values.append({
                "row_id": row["id"], "context": context.to_dict(),
                "source_identity": {
                    "candidate_file": str(root / "candidate.jsonl"),
                    "candidate_file_sha256": hashlib.sha256(b"candidate").hexdigest(),
                    "candidate_line": index, "registry_provenance_id": row["id"],
                    "registry_provenance_decision": "admitted",
                    "group_id": f"group-{row['id']}", "package_id": row["package_id"],
                    "source_ref": source_ref, "source_provenance": source_provenance,
                },
                "selection_geometry": selection,
                "family": row["family"], "package_id": row["package_id"],
                "split": "train",
                "prompt_sha256": hashlib.sha256(row["prompt_text"].encode()).hexdigest(),
                "context_has_target_or_reward_keys": False,
                "offline_static_source": True,
            })
        sidecar_path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in row_values))
        selected_path.write_text(json.dumps({
            "schema_version": SELECTED_IDS_SCHEMA_VERSION,
            "split": "train", "row_ids": [row["id"] for row, _, _ in rows],
        }, sort_keys=True))
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        return root, rows_path, sidecar_path, selected_path, digest(rows_path), digest(sidecar_path), digest(selected_path)

    def identity(self, rows_sha, context_sha, selected_sha, ordered_sha, count, rows=None):
        rows = rows or []
        row_identities = [
            {"id": row["id"], "capture": capture}
            for row, _context, record in rows
            for capture in [record["capture"]]
        ]
        row_identity_sha = hashlib.sha256(json.dumps(
            row_identities, separators=(",", ":"), ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        microbatch = 4 if count == 1 else 8
        accumulation = 1 if count == 1 else 4
        rollout = 4 if count == 1 else 32
        return {
            "parent": {
                "kind": "merged_sft",
                "manifest_sha256": "1" * 64,
                "merged_weights_sha256": "2" * 64,
            },
            "tokenizer": {
                "revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
                "json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
                "config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
                "vocab_size": 130560, "bos_id": 0, "eos_id": 1, "pad_id": 1,
                "native_eog_ids": [1, 130073],
            },
            "renderer": {
                "schema_version": SCHEMA_VERSION, "renderer_id": RENDERER_ID,
                "tokenization_policy": TOKENIZATION_POLICY,
                "terminal": ">>>>>>> UPDATED", "no_edit": "[NO_EDIT]",
            },
            "data": {
                "rows_sha256": rows_sha, "context_sha256": context_sha,
                "sidecar_artifact_sha256": context_sha,
                "selected_ids_sha256": selected_sha, "ordered_ids_sha256": ordered_sha,
                "row_identity_sha256": row_identity_sha,
                "row_identities": row_identities,
                "context_snapshot_sha256": self.snapshot,
                "selected_ids": ["r1"] if count == 1 else ["r1", "r2"],
                "split": "train", "admission_status": "admitted", "row_count": count,
            },
            "source": {"protocol_sha256": "4" * 64, "rl_trainer_sha256": "5" * 64, "trl_version": "0.24.0"},
            "policy": {
                "lora_rank": 16, "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "expected_attachments": 294, "expected_trainable_parameters": 25116672,
                "prompt_max_tokens": 2048, "completion_max_tokens": 192,
                "candidate_count": 4, "rollout_rows_per_update": rollout,
                "per_device_train_batch_size": microbatch,
                "gradient_accumulation_steps": accumulation,
                "generation_groups_per_call": 1,
                "loss_type": "bnpo", "scale_rewards": "group", "beta": 0,
                "sampling": dict(DEFAULT_GENERATION_KWARGS),
            },
            "schedule": {
                "seed": DEFAULT_SEED, "sampler_id": CampaignRepeatSampler.SAMPLER_ID,
                "generation_batch_size": rollout,
                "steps_per_generation": accumulation, "num_iterations": 1,
                "full_save_steps": 1, "evaluation_steps": [1], "decision_steps": [1],
            },
        }

    def test_data_loader_requires_admission_and_preserves_selected_order(self):
        rows = [self.row("r1", "replace", ("new",)), self.row("r2", "delete", ())]
        root, rows_path, context_path, selected_path, rows_sha, context_sha, selected_sha = self.files(rows)
        loaded, manifest = load_training_records(
            rows_path, rows_sha, context_path, context_sha, selected_path, selected_sha,
            admission_status="admitted", expected_context_snapshot_sha256=self.snapshot,
        )
        self.assertEqual([record.row_id for record in loaded], ["r1", "r2"])
        self.assertEqual(manifest.row_count, 2)
        self.assertEqual(manifest.context_snapshot_sha256, self.snapshot)
        self.assertEqual(records_to_dataset_rows(loaded)[0]["prompt"]["ids"][0], BOS_ID)
        self.assertEqual(records_to_dataset_rows(loaded)[0]["split"], "train")
        try:
            from datasets import Dataset
        except ImportError:
            Dataset = None
        if Dataset is not None:
            dataset_row = Dataset.from_list(records_to_dataset_rows(loaded))[0]
            self.assertEqual(dataset_row["prompt"]["ids"][0], BOS_ID)
            self.assertEqual(dataset_row["context"]["path"], "src/r1.py")
            self.assertEqual(dataset_row["context_capture"]["source_snapshot_sha256"], self.snapshot)
            self.assertEqual(dataset_row["target_operation"], "replace")
        with self.assertRaisesRegex(RLDataError, "explicit admitted"):
            load_training_records(
                rows_path, rows_sha, context_path, context_sha, selected_path, selected_sha,
                admission_status="tokenizer_candidate_only",
            )
        # Keep the temporary root alive through all reads above.
        self.assertTrue(root.exists())

    def test_data_loader_rejects_stale_self_consistent_snapshot(self):
        rows = [self.row("r1", "replace", ("new",))]
        root, rows_path, context_path, selected_path, rows_sha, context_sha, selected_sha = self.files(rows)
        with self.assertRaisesRegex(RLDataError, "source snapshot"):
            load_training_records(
                rows_path, rows_sha, context_path, context_sha, selected_path, selected_sha,
                expected_context_snapshot_sha256="6" * 64,
            )
        self.assertTrue(root.exists())

    def test_empty_range_noop_context_is_admitted(self):
        rows = [self.row("empty", "no_op", (), empty=True)]
        root, rows_path, context_path, selected_path, rows_sha, context_sha, selected_sha = self.files(rows)
        loaded, _ = load_training_records(
            rows_path, rows_sha, context_path, context_sha, selected_path, selected_sha,
            expected_context_snapshot_sha256=self.snapshot,
        )
        self.assertEqual(loaded[0].context.region_old, ())
        self.assertTrue(root.exists())

    def test_rl02_sidecar_accepts_mixed_static_sources_and_binds_row_evidence(self):
        rows = [self.row("r1", "replace", ("new",)), self.row("r2", "delete", ())]
        root, rows_path, sidecar_path, selected_path, rows_sha, sidecar_sha, selected_sha = self.sidecar_files(rows)
        loaded, manifest = load_training_records(
            rows_path, rows_sha, sidecar_path, sidecar_sha, selected_path, selected_sha,
        )
        self.assertEqual([record.row_id for record in loaded], ["r1", "r2"])
        self.assertEqual(manifest.sidecar_artifact_sha256, sidecar_sha)
        self.assertIsNone(manifest.context_snapshot_sha256)
        self.assertEqual(len(manifest.row_identities), 2)
        self.assertNotEqual(
            manifest.row_identities[0]["source_identity"]["source_ref"]["source_sha256"],
            manifest.row_identities[1]["source_identity"]["source_ref"]["source_sha256"],
        )
        dataset_rows = records_to_dataset_rows(loaded)
        self.assertEqual(
            loaded[0].capture.source_snapshot_sha256,
            loaded[0].source_identity["source_ref"]["source_sha256"],
        )
        self.assertEqual(dataset_rows[0]["selection_geometry"]["document_sha256"],
                         loaded[0].context.replacement_range.content_sha256)
        self.assertEqual(dataset_rows[1]["source_identity"]["source_ref"]["row_id"], "r2")
        self.assertNotIn("target_operation", dataset_rows[0]["prompt"])
        self.assertNotIn("target_body_text", dataset_rows[0]["prompt"])
        self.assertTrue(root.exists())

    def test_rl02_sidecar_rejects_row_evidence_that_disagrees_internally(self):
        rows = [self.row("r1", "replace", ("new",))]
        root, rows_path, sidecar_path, selected_path, rows_sha, _sidecar_sha, selected_sha = self.sidecar_files(rows)
        value = json.loads(sidecar_path.read_text().splitlines()[0])
        value["source_identity"]["source_provenance"]["source_sha256"] = "f" * 64
        sidecar_path.write_text(json.dumps(value, sort_keys=True) + "\n")
        sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(RLDataError, "differs from source reference"):
            load_training_records(
                rows_path, rows_sha, sidecar_path, sidecar_sha, selected_path, selected_sha,
            )
        self.assertTrue(root.exists())


class RewardTests(RLFixtures):
    def setUp(self):
        super().setUp()
        self.row_replace, self.replace_context, _ = self.row("replace", "replace", ("new",))
        self.row_noop, self.noop_context, _ = self.row("noop", "no_op", self.context("noop").region_old)
        self.row_delete, self.delete_context, _ = self.row("delete", "delete", ())
        self.row_empty, self.empty_context, _ = self.row("empty", "no_op", (), empty=True)
        self.reward = CampaignPRM03Reward(decoder=self.encoder.decode)

    def generated(self, text, terminal=EOS_ID):
        return self.encoder.encode(text) + [terminal]

    def test_exact_replace_and_whitespace_shaping(self):
        exact, record = self.reward.score_one(
            self.replace_context, "replace", "new", self.generated("new\n>>>>>>> UPDATED"), row_id="replace",
        )
        self.assertEqual(exact, 1.2)
        self.assertTrue(record["protocol_valid"])
        shaped, mismatch = self.reward.score_one(
            self.replace_context, "replace", "new", self.generated("new \n>>>>>>> UPDATED"), row_id="replace",
        )
        self.assertEqual(shaped, 0.0)
        self.assertFalse(mismatch["exact_region"])
        self.assertEqual(mismatch["line_f1"], 1.0)

    def test_noop_delete_and_empty_range_semantics(self):
        no_edit, no_edit_record = self.reward.score_one(
            self.noop_context, "no_op", "[NO_EDIT]", self.generated("[NO_EDIT]\n>>>>>>> UPDATED"),
        )
        full_copy, full_copy_record = self.reward.score_one(
            self.noop_context, "no_op", "[NO_EDIT]", self.generated("old\n>>>>>>> UPDATED"),
        )
        deleted, delete_record = self.reward.score_one(
            self.delete_context, "delete", "", self.generated(">>>>>>> UPDATED"),
        )
        empty_noop, empty_record = self.reward.score_one(
            self.empty_context, "no_op", "[NO_EDIT]", self.generated("[NO_EDIT]\n>>>>>>> UPDATED"),
        )
        self.assertEqual([no_edit, full_copy, deleted, empty_noop], [1.2] * 4)
        self.assertTrue(all(record["protocol_valid"] for record in (no_edit_record, full_copy_record, delete_record, empty_record)))
        self.assertEqual(delete_record["operation"], "delete")

    def test_invalid_terminal_eog_control_and_duplicate_marker_are_zero(self):
        cases = [
            (self.generated("new\n>>>>>>> UPDATED", terminal=130073), "noncanonical_eog"),
            ([0] + self.generated("new\n>>>>>>> UPDATED"), "control_before_terminal"),
            (self.encoder.encode("new\n>>>>>>> UPDATED"), "missing_canonical_eos"),
            (self.encoder.encode("new\n>>>>>>> UPDATED") + [EOS_ID, EOS_ID], "premature_canonical_eos"),
            (self.generated("new\n>>>>>>> UPDATED\n>>>>>>> UPDATED"), "duplicate_exact_terminal"),
        ]
        for ids, reason in cases:
            value, record = self.reward.score_one(
                self.replace_context, "replace", "new", ids,
            )
            self.assertEqual(value, 0.0, reason)
            self.assertEqual(record["failure"], reason)

    def test_reward_call_aligns_metadata_and_emits_records(self):
        events = []
        reward = CampaignPRM03Reward(decoder=self.encoder.decode, event_sink=events.append)
        values = reward(
            prompts=[{"ids": list(self.row_replace["input_ids"][: self.row_replace["target_start"]])}],
            completions=["ignored"],
            completion_ids=[self.generated("new\n>>>>>>> UPDATED")],
            context=[self.replace_context.to_dict()],
            target_operation=["replace"], target_body_text=["new"],
            id=["replace"], family=["rename"], package_id=["pkg"],
        )
        self.assertEqual(values, [1.2])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["output_ids_sha256"], reward.last_records[0]["output_ids_sha256"])

    def test_line_f1_matches_legacy_empty_and_trailing_space_behavior(self):
        self.assertEqual(line_f1([], []), 1.0)
        self.assertEqual(line_f1(["x  ", ""], ["x"]), 1.0)
        self.assertEqual(line_f1(["x"], ["y"]), 0.0)


class GeometryTests(unittest.TestCase):
    def test_geometry_preserves_legacy_32_completion_rows_and_one_update(self):
        geometry = resolve_trl_geometry(4)
        self.assertEqual(geometry.to_dict(), {
            "candidate_count": 4, "rollout_rows_per_update": 32,
            "prompt_groups_per_update": 8, "per_device_train_batch_size": 8,
            "generation_batch_size": 32, "gradient_accumulation_steps": 4,
            "steps_per_generation": 4, "num_iterations": 1,
        })
        microbatch4 = resolve_trl_geometry(
            4, per_device_train_batch_size=4, gradient_accumulation_steps=8,
        )
        self.assertEqual(microbatch4.generation_batch_size, 32)
        self.assertEqual(microbatch4.steps_per_generation, 8)
        self.assertEqual(microbatch4.per_device_train_batch_size, 4)
        self.assertEqual(microbatch4.gradient_accumulation_steps, 8)
        one_group = resolve_trl_geometry(2, rollout_rows_per_update=2)
        self.assertEqual(one_group.prompt_groups_per_update, 1)
        with self.assertRaisesRegex(RLTrainError, "divisible"):
            resolve_trl_geometry(4, rollout_rows_per_update=6)
        with self.assertRaisesRegex(RLTrainError, "one process"):
            resolve_trl_geometry(4, world_size=2)

    def test_sampler_groups_are_contiguous_and_drops_partial_prompt_batch(self):
        sampler = CampaignRepeatSampler(
            list(range(5)), candidate_count=2, prompt_groups_per_batch=2,
        )
        self.assertEqual(list(sampler), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(len(sampler), 8)
        self.assertEqual(sampler.state(), {
            "sampler_id": "campaign-repeat-manifest-order-v1", "seed": 3407,
            "shuffle": False, "num_samples": 5, "candidate_count": 2,
            "prompt_groups_per_batch": 2, "repeat_count": 1,
            "consumed_rows": 0, "current_index": 0, "consumed_prompt_copies": 0,
            "selected_id_index": 0,
            "epoch": 0,
        })
        with self.assertRaisesRegex(RLTrainError, "shuffle"):
            CampaignRepeatSampler(list(range(4)), candidate_count=2, prompt_groups_per_batch=2, shuffle=True)

    def test_sampler_checkpoint_boundary_reconstructs_next_generation_batch(self):
        sampler = CampaignRepeatSampler(
            list(range(18)), candidate_count=4, prompt_groups_per_batch=8,
            repeat_count=4, generation_batch_size=32,
            per_device_train_batch_size=8, gradient_accumulation_steps=4,
            steps_per_generation=4,
        )
        first = sampler.indices_from(0, 32)
        resumed = sampler.indices_from(32, 32)
        self.assertEqual(first, [index for index in range(8) for _ in range(4)])
        self.assertEqual(resumed, first)
        self.assertEqual(
            sampler.indices_from(128, 32), [index for index in range(8, 16) for _ in range(4)]
        )
        state = sampler.state(consumed_rows=32, epoch=1)
        self.assertEqual(state["consumed_rows"], 32)
        self.assertEqual(state["selected_id_index"], 0)
        self.assertEqual(state["geometry"], {
            "generation_batch_size": 32,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "steps_per_generation": 4,
        })
        self.assertEqual(state["generation_rows_per_update"], 32)
        self.assertEqual(state["sampler_rows_per_update"], 128)
        self.assertEqual(sampler.state(consumed_rows=128)["selected_id_index"], 8)
        with self.assertRaisesRegex(RLTrainError, "inside a prompt group batch"):
            sampler.indices_from(4, 32)
        one_group = CampaignRepeatSampler(
            [0], candidate_count=4, prompt_groups_per_batch=1, repeat_count=8,
            generation_batch_size=32, per_device_train_batch_size=4,
            gradient_accumulation_steps=8, steps_per_generation=8,
        )
        with self.assertRaisesRegex(RLTrainError, "inside a generation buffer"):
            one_group.indices_from(4, 4)

    def test_frozen_source_draw_schedule_drives_g4_update_and_resume_prefix_suffix(self):
        schedule = list(range(16))
        sampler = CampaignRepeatSampler(
            list(range(32)), candidate_count=4, prompt_groups_per_batch=8,
            repeat_count=4, generation_batch_size=32,
            per_device_train_batch_size=8, gradient_accumulation_steps=4,
            steps_per_generation=4, source_draw_sequence=schedule,
            source_draw_sequence_sha256="a" * 64,
            source_draw_schedule_sha256="b" * 64,
        )
        expected_prefix = [index for index in range(8) for _ in range(4)] * 4
        expected_suffix = [index for index in range(8, 16) for _ in range(4)] * 4
        self.assertEqual(len(sampler), 256)
        self.assertEqual(sampler.indices_from(0, 128), expected_prefix)
        self.assertEqual(sampler.indices_from(128, 128), expected_suffix)
        state = sampler.state(consumed_rows=128)
        self.assertEqual(state["source_draw_cursor"], 8)
        self.assertEqual(state["source_draws"], 16)
        self.assertEqual(state["source_draw_sequence_sha256"], "a" * 64)
        self.assertEqual(state["source_draw_schedule_sha256"], "b" * 64)
        # One optimizer update: 8 source prompts, G=4 => 32 generated rows,
        # then four TRL buffer-reuse copies => 128 dataloader rows.
        self.assertEqual(len(expected_prefix) // 4, 32)
        self.assertEqual(len(expected_prefix), 128)

    def test_frozen_source_draw_schedule_drives_g2_geometry(self):
        schedule = list(range(16))
        sampler = CampaignRepeatSampler(
            list(range(32)), candidate_count=2, prompt_groups_per_batch=16,
            repeat_count=4, generation_batch_size=32,
            per_device_train_batch_size=8, gradient_accumulation_steps=4,
            steps_per_generation=4, source_draw_sequence=schedule,
            source_draw_sequence_sha256=source_draw_sequence_sha256(
                [f"row-{index}" for index in schedule]
            ),
            source_draw_schedule_sha256="c" * 64,
        )
        expected = [index for index in schedule for _ in range(2)] * 4
        self.assertEqual(len(expected), 128)
        self.assertEqual(list(sampler), expected)
        self.assertEqual(sampler.state(consumed_rows=128)["source_draw_cursor"], 16)
        self.assertEqual(sampler.state(consumed_rows=128)["sampler_rows_per_update"], 128)

    def test_frozen_source_draw_schedule_drives_g4_four_by_eight_and_rejects_mismatch(self):
        schedule = list(range(16))
        sampler = CampaignRepeatSampler(
            list(range(32)), candidate_count=4, prompt_groups_per_batch=8,
            repeat_count=8, generation_batch_size=32,
            per_device_train_batch_size=4, gradient_accumulation_steps=8,
            steps_per_generation=8, source_draw_sequence=schedule,
            source_draw_sequence_sha256="d" * 64,
            source_draw_schedule_sha256="e" * 64,
        )
        expected_prefix = [index for index in range(8) for _ in range(4)] * 8
        expected_suffix = [index for index in range(8, 16) for _ in range(4)] * 8
        self.assertEqual(len(sampler), 512)
        self.assertEqual(sampler.indices_from(0, 256), expected_prefix)
        self.assertEqual(sampler.indices_from(256, 256), expected_suffix)
        state = sampler.state(consumed_rows=256)
        self.assertEqual(state["source_draw_cursor"], 8)
        self.assertEqual(state["sampler_rows_per_update"], 256)
        with self.assertRaisesRegex(RLTrainError, "buffer reuse"):
            CampaignRepeatSampler(
                list(range(32)), candidate_count=4, prompt_groups_per_batch=8,
                repeat_count=4, generation_batch_size=32,
                per_device_train_batch_size=4, gradient_accumulation_steps=8,
                steps_per_generation=8, source_draw_sequence=schedule,
                source_draw_sequence_sha256="d" * 64,
                source_draw_schedule_sha256="e" * 64,
            )

    def test_source_draw_schedule_loader_allows_replays_but_rejects_id_leaks(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draws.json"
            row_ids = ["r0", "r1", "r0", "r1"]
            value = {
                "schema_version": "sepalith.dat09.source-row-draw-sequence.v1",
                "selected_ids_sha256": "a" * 64,
                "ordered_ids_sha256": "b" * 64,
                "row_identity_sha256": "c" * 64,
                "candidate_count": 2,
                "completions_per_update": 4,
                "prompt_groups_per_update": 2,
                "buffer_reuse": 1,
                "gradient_accumulation_steps": 1,
                "steps_per_generation": 1,
                "source_draws_per_update": 2,
                "source_draws": len(row_ids),
                "sequence_sha256": source_draw_sequence_sha256(row_ids),
                "row_ids": row_ids,
            }
            path.write_text(json.dumps(value, sort_keys=True))
            loaded = load_source_draw_schedule(
                path, hashlib.sha256(path.read_bytes()).hexdigest(),
                selected_ids=["r0", "r1"], selected_ids_sha256="a" * 64,
                ordered_ids_sha256="b" * 64, row_identity_sha256="c" * 64,
                candidate_count=2, source_draws_per_update=2, buffer_reuse=1,
            )
            self.assertEqual(loaded["source_draws"], 4)
            self.assertEqual(loaded["row_ids"], row_ids)
            value["gradient_accumulation_steps"] = 2
            path.write_text(json.dumps(value, sort_keys=True))
            with self.assertRaisesRegex(RLTrainError, "gradient_accumulation_steps"):
                load_source_draw_schedule(
                    path, hashlib.sha256(path.read_bytes()).hexdigest(),
                    selected_ids=["r0", "r1"], selected_ids_sha256="a" * 64,
                    ordered_ids_sha256="b" * 64, row_identity_sha256="c" * 64,
                    candidate_count=2, source_draws_per_update=2, buffer_reuse=1,
                )
            value["gradient_accumulation_steps"] = 1
            value["row_ids"] = ["r0", "leaked", "r0", "r1"]
            value["sequence_sha256"] = source_draw_sequence_sha256(value["row_ids"])
            path.write_text(json.dumps(value, sort_keys=True))
            with self.assertRaisesRegex(RLTrainError, "outside unique selected"):
                load_source_draw_schedule(
                    path, hashlib.sha256(path.read_bytes()).hexdigest(),
                    selected_ids=["r0", "r1"], selected_ids_sha256="a" * 64,
                    ordered_ids_sha256="b" * 64, row_identity_sha256="c" * 64,
                    candidate_count=2, source_draws_per_update=2, buffer_reuse=1,
                )

    def test_actual_pinned_trl_config_derives_steps_without_conflicting_fields(self):
        try:
            config, geometry = make_grpo_config(
                candidate_count=4, rollout_rows_per_update=4,
                output_dir=Path(tempfile.mkdtemp()) / "trainer",
                max_steps=1, save_steps=1, use_cpu=True,
            )
        except RLTrainError as error:
            if "cannot import the pinned TRL" in str(error):
                self.skipTest(str(error))
            raise
        self.assertEqual(geometry.steps_per_generation, 1)
        self.assertEqual(config.generation_batch_size, 4)
        self.assertEqual(config.steps_per_generation, 1)
        self.assertEqual(config.per_device_train_batch_size, 4)
        self.assertEqual(config.loss_type, "bnpo")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.95)
        self.assertFalse(config.remove_unused_columns)
        self.assertFalse(config.shuffle_dataset)
        config32, geometry32 = make_grpo_config(
            candidate_count=4, rollout_rows_per_update=32,
            per_device_train_batch_size=4, gradient_accumulation_steps=8,
            output_dir=Path(tempfile.mkdtemp()) / "trainer32",
            max_steps=1, save_steps=1, use_cpu=True,
        )
        self.assertEqual(geometry32.to_dict()["generation_batch_size"], 32)
        self.assertEqual(config32.per_device_train_batch_size, 4)
        self.assertEqual(config32.gradient_accumulation_steps, 8)
        self.assertEqual(config32.steps_per_generation, 8)
        self.assertEqual(config32.generation_batch_size, 32)
        with self.assertRaisesRegex(RLTrainError, "cannot override"):
            make_grpo_config(candidate_count=4, rollout_rows_per_update=4, use_cpu=True,
                             max_steps=1, save_steps=1, extra={"steps_per_generation": 1})
        with self.assertRaisesRegex(RLTrainError, "cannot override"):
            make_grpo_config(candidate_count=4, rollout_rows_per_update=4, use_cpu=True,
                             max_steps=1, save_steps=1, extra={"num_iterations": 2})

    def test_reward_object_has_pinned_trl_callable_name(self):
        reward = CampaignPRM03Reward(decoder=lambda ids, **_: "")
        self.assertEqual(reward.__name__, "campaign_prm03_reward")

    def test_fixed_id_padding_preserves_manual_bos_and_response_shape(self):
        import torch
        from campaign_rl_train import FixedIDGRPOTrainerMixin, RLGenerationError

        input_ids, attention, width = FixedIDGRPOTrainerMixin._padded_prompt_tensors(
            [[0, 200], [0, 201, 202]], torch, torch.device("cpu"), pad_id=1,
        )
        self.assertEqual(width, 3)
        self.assertEqual(input_ids.tolist(), [[1, 0, 200], [0, 201, 202]])
        self.assertEqual(attention.tolist(), [[0, 1, 1], [1, 1, 1]])
        obj = FixedIDGRPOTrainerMixin.__new__(FixedIDGRPOTrainerMixin)
        obj._campaign_prompt_max_tokens = 3
        obj._campaign_completion_max_tokens = 2
        obj._campaign_context_max_tokens = 5
        obj._campaign_generation_kwargs = {}
        self.assertEqual(obj._checked_generation_kwargs()["eos_token_id"], [1, 130073])
        obj._campaign_generation_kwargs = {"eos_token_id": [1]}
        with self.assertRaises(RLGenerationError):
            obj._checked_generation_kwargs()

    def test_fixed_id_generation_uses_stored_ids_and_exact_eog_trim(self):
        try:
            from trl.models import unwrap_model_for_generation  # noqa: F401
        except Exception as error:
            self.skipTest(f"pinned TRL generation seam unavailable: {error}")
        import torch
        from campaign_rl_train import FixedIDGRPOTrainerMixin

        class FakeModel(torch.nn.Module):
            is_gradient_checkpointing = False

            def __init__(self):
                super().__init__()
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                input_ids = kwargs["input_ids"]
                self.assert_shape = tuple(input_ids.shape)
                # Both rows are one contiguous G=2 prompt group. Returned
                # rows include the prompt width before their generated tail.
                return SimpleNamespace(sequences=torch.tensor([
                    [0, 200, 300, EOS_ID],
                    [0, 200, 301, EOS_ID],
                ], dtype=torch.long))

        class FakeAccelerator:
            device = torch.device("cpu")
            state = SimpleNamespace(deepspeed_plugin=None)

            @staticmethod
            def unwrap_model(model):
                return model

        model = FakeModel()
        obj = FixedIDGRPOTrainerMixin.__new__(FixedIDGRPOTrainerMixin)
        obj.model = model
        obj.model_wrapped = model
        obj.accelerator = FakeAccelerator()
        obj.args = SimpleNamespace(ds3_gather_for_generation=False)
        obj.processing_class = object()
        obj.num_generations = 2
        obj._campaign_generation_kwargs = {}
        obj.configure_campaign_runtime(
            generation_guard_factory=lambda _model: nullcontext(),
            prompt_max_tokens=2048,
            completion_max_tokens=192,
            context_max_tokens=2240,
        )
        prompt_ids, completions, logprobs, forward_kwargs = obj._generate_single_turn([
            {"text": "ignored", "ids": [0, 200]},
            {"text": "ignored", "ids": [0, 200]},
        ])
        self.assertEqual(prompt_ids, [[0, 200], [0, 200]])
        self.assertEqual(completions, [[300, EOS_ID], [301, EOS_ID]])
        self.assertIsNone(logprobs)
        self.assertEqual(forward_kwargs, {})
        self.assertEqual(model.assert_shape, (2, 2))
        self.assertEqual(model.calls[0]["input_ids"].tolist(), [[0, 200], [0, 200]])
        self.assertEqual(model.calls[0]["eos_token_id"], list(NATIVE_EOG_IDS))
        self.assertEqual(model.calls[0]["max_new_tokens"], 192)
        self.assertEqual(obj._campaign_last_generation["accounting"]["canonical_eos"], 2)

    def test_fixed_id_generation_chunks_32_row_buffer_into_ordered_g_groups(self):
        try:
            from trl.models import unwrap_model_for_generation  # noqa: F401
        except Exception as error:
            self.skipTest(f"pinned TRL generation seam unavailable: {error}")
        import torch
        from campaign_rl_train import FixedIDGRPOTrainerMixin

        class FakeModel(torch.nn.Module):
            is_gradient_checkpointing = False

            def __init__(self):
                super().__init__()
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                input_ids = kwargs["input_ids"]
                base = int(input_ids[0, -1])
                return SimpleNamespace(sequences=torch.tensor([
                    [0, base, 300 + index, EOS_ID]
                    for index in range(int(input_ids.shape[0]))
                ], dtype=torch.long))

        class FakeAccelerator:
            device = torch.device("cpu")
            state = SimpleNamespace(deepspeed_plugin=None)

            @staticmethod
            def unwrap_model(model):
                return model

        model = FakeModel()
        obj = FixedIDGRPOTrainerMixin.__new__(FixedIDGRPOTrainerMixin)
        obj.model = model
        obj.model_wrapped = model
        obj.accelerator = FakeAccelerator()
        obj.args = SimpleNamespace(ds3_gather_for_generation=False)
        obj.processing_class = object()
        obj.num_generations = 4
        obj._campaign_generation_kwargs = {}

        transitions = []

        class GenerationGuard:
            def __enter__(self):
                transitions.append("enter")
                return self

            def __exit__(self, *_args):
                transitions.append("exit")

        obj.configure_campaign_runtime(
            generation_guard_factory=lambda _model: GenerationGuard(),
            post_generation_restore=lambda _model, _tokenizer: transitions.append("restore"),
            prompt_max_tokens=2048, completion_max_tokens=192, context_max_tokens=2240,
        )
        prompts = ([{"text": "ignored", "ids": [0, 200]}] * 4
                   + [{"text": "ignored", "ids": [0, 201]}] * 4)
        prompt_ids, completions, _logprobs, _forward_kwargs = obj._generate_single_turn(prompts)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([tuple(call["input_ids"][0].tolist()) for call in model.calls], [(0, 200), (0, 201)])
        self.assertEqual(prompt_ids, [[0, 200]] * 4 + [[0, 201]] * 4)
        self.assertEqual(completions, [[300, EOS_ID], [301, EOS_ID], [302, EOS_ID], [303, EOS_ID],
                                       [300, EOS_ID], [301, EOS_ID], [302, EOS_ID], [303, EOS_ID]])
        self.assertEqual([record["group_index"] for record in obj._campaign_last_generation["records"]],
                         [0] * 4 + [1] * 4)
        self.assertEqual(obj._campaign_last_generation["generation_chunk_size"], 4)
        self.assertEqual(transitions, ["enter", "exit", "restore"])

    def test_dynamic_class_is_actual_pinned_trl_subclass_when_available(self):
        try:
            cls = campaign_grpo_trainer_class()
        except RLTrainError as error:
            if "cannot import the pinned TRL" in str(error):
                self.skipTest(str(error))
            raise
        from trl import GRPOTrainer
        self.assertTrue(issubclass(cls, GRPOTrainer))
        self.assertIn("_generate_single_turn", dir(cls))
        self.assertIn("_get_train_sampler", cls.__dict__)

    def test_actual_pinned_trl_constructor_works_without_optional_mergekit(self):
        """Exercise the class constructor after the optional-import boundary."""
        try:
            cls = campaign_grpo_trainer_class()
            from datasets import Dataset
            from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
        except (ImportError, ModuleNotFoundError, RLTrainError) as error:
            if "trl" in str(error).lower() or "datasets" in str(error).lower():
                self.skipTest(str(error))
            raise
        config, geometry = make_grpo_config(
            candidate_count=4, rollout_rows_per_update=32,
            per_device_train_batch_size=4, gradient_accumulation_steps=8,
            output_dir=Path(tempfile.mkdtemp()) / "constructor",
            max_steps=1, save_steps=1, use_cpu=True,
        )
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=130560, n_embd=16, n_layer=1, n_head=1, n_positions=64,
        ))
        model.config._name_or_path = "rl-constructor-fixture"
        model.warnings_issued = {}
        tokenizer_backend = Tokenizer(WordLevel(
            vocab={"<bos>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>",
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            bos_token="<bos>", eos_token="<eos>", pad_token="<eos>", unk_token="<unk>",
        )
        dataset = Dataset.from_list([{
            "prompt": {"text": "ignored", "ids": [0, 2]},
            "context": {}, "context_capture": {}, "target_operation": "replace",
            "target_body_text": "x", "family": "fixture", "package_id": "pkg",
            "id": f"r{index}", "split": "train",
        } for index in range(8)])
        trainer = cls(
            model=model, processing_class=tokenizer,
            reward_funcs=CampaignPRM03Reward(decoder=lambda _ids, **_kwargs: ""),
            train_dataset=dataset, args=config,
        )
        self.assertEqual(trainer.args.generation_batch_size, 32)
        self.assertEqual(trainer.args.steps_per_generation, 8)
        self.assertEqual(trainer._train_batch_size, geometry.per_device_train_batch_size)
        sampler = trainer._get_train_sampler(dataset)
        self.assertEqual(len(sampler), 32 * 8)
        checkpoint_state = sampler.state(consumed_rows=256)
        self.assertEqual(checkpoint_state["geometry"]["steps_per_generation"], 8)
        self.assertEqual(checkpoint_state["sampler_rows_per_update"], 256)
        trainer.state.global_step = 1
        self.assertEqual(trainer.campaign_sampler_state()["consumed_rows"], 256)


class RecipeTests(RLFixtures):
    def test_preflight_requires_future_merged_parent_and_exact_manifest(self):
        rows = [self.row("r1", "replace", ("new",))]
        root, rows_path, context_path, selected_path, rows_sha, context_sha, selected_sha = self.files(rows)
        # Build the ordered hash exactly as RLDataManifest does.
        ordered_sha = hashlib.sha256(json.dumps(["r1"], separators=(",", ":")).encode()).hexdigest()
        recipe = {
            "schema_version": TRAIN_SCHEMA_VERSION,
            "generation_groups_per_call": 1,
            "identity": self.identity(rows_sha, context_sha, selected_sha, ordered_sha, 1, rows=rows),
            "data": {
                "rows_path": str(rows_path), "rows_sha256": rows_sha,
                "context_path": str(context_path), "context_sha256": context_sha,
                "selected_ids_path": str(selected_path), "selected_ids_sha256": selected_sha,
                "admission_status": "admitted", "context_snapshot_sha256": self.snapshot,
            },
        }
        result = preflight_rl_recipe(recipe)
        self.assertEqual(result["status"], "preflight_pass")
        self.assertEqual(result["geometry"]["steps_per_generation"], 1)
        self.assertEqual(result["data"]["ordered_ids_sha256"], ordered_sha)
        recipe["identity"]["parent"]["kind"] = "minicpm_midtrain"
        with self.assertRaisesRegex(RLTrainError, "merged_sft"):
            preflight_rl_recipe(recipe)
        self.assertTrue(root.exists())


if __name__ == "__main__":
    unittest.main()
