from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from export_header_gate import self_test, validate_export_header, synthetic_header
from minicpm5_dspark_config import (
    apply_upstream_sliding_window_patch,
    build_draft_config,
    serving_stop_token_ids,
)
from pretokenized_cache_bridge import (
    CACHE_INDEX_BYTES,
    build_cache_sample,
    cache_manifest,
    cache_payload_nbytes,
    collate_pretokenized,
    PretokenizedTargetCollator,
    reconcile_written_records,
    pack_index_record,
    unpack_index_record,
)
from prepare_teacher_cache import (
    TRAIN_SOURCE_PATH,
    TRAIN_SOURCE_SHA256,
    _generation_parts,
    regenerate_train_rows,
)
from public_dspark_warm_start import PUBLIC_REVISION, compare_target_config, public_config_contract
from teacher_rows import build_teacher_record, classify_generated_tail


TARGET = {
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "vocab_size": 64,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "bos_token_id": 0,
    "pad_token_id": 1,
    "eos_token_id": [1, 63],
}


class ConfigBridgeTest(unittest.TestCase):
    def test_missing_sliding_window_is_explicitly_patched(self) -> None:
        draft = build_draft_config(
            TARGET,
            draft_layers=2,
            block_size=2,
            target_layer_ids=(0, 1, 2, 3, 4),
            num_anchors=1,
            markov_rank=0,
            mask_token_id=1,
        )
        self.assertIsNone(draft["sliding_window"])
        self.assertEqual(draft["layer_types"], ["full_attention", "full_attention"])
        self.assertEqual(draft["serving_stop_token_ids"], [1, 63, 130073])

    def test_object_patch_matches_upstream_constructor_seam(self) -> None:
        class Config:
            pass

        config = Config()
        self.assertFalse(hasattr(config, "sliding_window"))
        self.assertIs(apply_upstream_sliding_window_patch(config), config)
        self.assertIsNone(config.sliding_window)
        self.assertEqual(serving_stop_token_ids(TARGET), (1, 63, 130073))


class TeacherRowsTest(unittest.TestCase):
    ROW = {
        "id": "train-1",
        "input_ids": [0, 10, 11, 100, 101, 1],
        "target_start": 3,
        "target_body": [100],
        "target_terminal": 101,
    }

    def test_exact_generated_prefix_and_no_synthetic_eos(self) -> None:
        record = build_teacher_record(self.ROW, [22, 1], protocol_status="valid")
        self.assertEqual(record["input_ids"], [0, 10, 11, 22, 1])
        self.assertEqual(record["loss_mask"], [0, 0, 0, 1, 1])
        self.assertEqual(record["target_generated_ids"], [22, 1])
        self.assertFalse(record["eos_was_synthesized"])
        self.assertTrue(record["cacheable"])

    def test_cap_and_invalid_protocol_are_retained(self) -> None:
        outcome = classify_generated_tail([22, 23, 24], generation_cap=3, protocol_status="invalid")
        self.assertTrue(outcome["cap_hit"])
        self.assertTrue(outcome["protocol_invalid"])
        self.assertTrue(outcome["cacheable"])
        self.assertEqual(outcome["target_generated_ids"], [22, 23, 24])

    def test_native_eog_is_preserved(self) -> None:
        record = build_teacher_record(self.ROW, [22, 1], protocol_status="valid")
        self.assertEqual(record["input_ids"], [0, 10, 11, 22, 1])
        self.assertEqual(record["native_stop_token_id"], 1)
        self.assertFalse(record["cap_hit"])

    def test_post_eog_tail_is_truncated_and_over_cap_rejected(self) -> None:
        outcome = classify_generated_tail([22, 1, 1, 23], generation_cap=4, vocab_size=64)
        self.assertEqual(outcome["target_generated_ids"], [22, 1])
        self.assertEqual(outcome["post_eog_tail_count"], 2)
        self.assertFalse(outcome["cacheable"])
        self.assertEqual(outcome["cache_rejection_reason"], "post_eog_tail")
        self.assertEqual(outcome["raw_target_generated_ids"], [22, 1, 1, 23])
        self.assertFalse(
            classify_generated_tail([22, 130073, 130560], vocab_size=130560)["cacheable"]
        )
        outcome = classify_generated_tail([22, 23, 24, 25, 1], generation_cap=4, vocab_size=64)
        self.assertTrue(outcome["over_cap"])
        self.assertFalse(outcome["cacheable"])
        self.assertEqual(outcome["cache_rejection_reason"], "generation_over_cap")
        record = build_teacher_record(self.ROW, [22, 1, 1, 23], vocab_size=64)
        self.assertEqual(record["input_ids"], [0, 10, 11, 22, 1])
        self.assertEqual(record["loss_mask"], [0, 0, 0, 1, 1])
        self.assertFalse(record["cacheable"])

    def test_length_terminated_generation_is_rejected(self) -> None:
        outcome = classify_generated_tail([22, 23], generation_cap=4, vocab_size=64)
        self.assertEqual(outcome["generation_status"], "length_terminated")
        self.assertFalse(outcome["cacheable"])

    def test_generator_requires_explicit_suffix_and_exact_full_sequence(self) -> None:
        prefix = [0, 10, 11]
        with self.assertRaisesRegex(ValueError, "mapping"):
            _generation_parts([22, 1], prefix)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "equal"):
            _generation_parts({"generated_ids": [22, 1], "input_ids": [0, 10, 11, 22]}, prefix)
        self.assertEqual(
            _generation_parts(
                {"generated_ids": [22, 1], "input_ids": [0, 10, 11, 22, 1]},
                prefix,
            )[0],
            [22, 1],
        )

    def test_systemic_generator_error_propagates(self) -> None:
        rows = [dict(self.ROW, id=f"train-{index}") for index in range(3)]

        def generator(prompt: list[int], cap: int):
            raise RuntimeError("simulated target service error")

        with self.assertRaisesRegex(RuntimeError, "target generator failed"):
            regenerate_train_rows(
                rows,
                generator,
                limit=3,
                generation_cap=4,
                vocab_size=64,
                source_path=TRAIN_SOURCE_PATH,
                source_sha256=TRAIN_SOURCE_SHA256,
            )

    def test_invalid_edit_protocol_remains_cacheable_when_tokens_are_valid(self) -> None:
        rows = [dict(self.ROW, id="train-invalid")]

        def generator(prompt: list[int], cap: int):
            return {"generated_ids": [22, 1], "protocol_status": "invalid"}

        records, summary = regenerate_train_rows(
            rows,
            generator,
            limit=1,
            generation_cap=4,
            vocab_size=64,
            source_path=TRAIN_SOURCE_PATH,
            source_sha256=TRAIN_SOURCE_SHA256,
        )
        self.assertTrue(records[0]["cacheable"])
        self.assertEqual(summary["protocol_invalid"], 1)

    def test_prompt_requires_bos_and_vocab(self) -> None:
        with self.assertRaisesRegex(ValueError, "BOS"):
            build_teacher_record({"id": "bad", "input_ids": [9, 10], "target_start": 1}, [1])
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            build_teacher_record(
                {"id": "bad", "input_ids": [0, 130560], "target_start": 2},
                [1],
            )

    def test_train_source_binding_and_split_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_path"):
            regenerate_train_rows(
                [self.ROW],
                lambda prompt, cap: {"generated_ids": [22, 1]},
                limit=1,
                vocab_size=64,
            )
        heldout = dict(self.ROW, id="heldout", split="validation")
        records, summary = regenerate_train_rows(
            [heldout],
            lambda prompt, cap: {"generated_ids": [22, 1]},
            limit=1,
            vocab_size=64,
            source_path=TRAIN_SOURCE_PATH,
            source_sha256=TRAIN_SOURCE_SHA256,
        )
        self.assertFalse(records[0]["cacheable"])
        self.assertEqual(summary["generation_error"], 1)


class CacheBridgeTest(unittest.TestCase):
    def test_payload_geometry_and_sample(self) -> None:
        # For H=2 and 2 target taps: 4+1+1+2*2*3=18 bytes/token.
        self.assertEqual(cache_payload_nbytes(3, hidden_size=2, target_taps=2), 54)
        self.assertEqual(cache_payload_nbytes(1), 24582)
        generated = build_teacher_record(
            {"id": "r", "input_ids": [0, 10, 11, 100], "target_start": 3},
            [22, 23],
        )
        import torch

        sample = build_cache_sample(
            generated,
            torch.zeros((5, 4), dtype=torch.bfloat16),
            torch.zeros((5, 2), dtype=torch.bfloat16),
            hidden_size=2,
            target_taps=2,
        )
        self.assertEqual(sample["sequence_length"], 5)
        self.assertEqual(sample["target_token_count"], 2)
        self.assertFalse(sample["cache_writer_requires_eos"])
        self.assertEqual(sample["eos_injection"], "forbidden")

    def test_bf16_and_written_row_gates(self) -> None:
        import torch

        generated = build_teacher_record(
            {"id": "r", "input_ids": [0, 10, 11, 100], "target_start": 3},
            [22, 1],
        )
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            build_cache_sample(
                generated,
                torch.zeros((5, 4), dtype=torch.float32),
                torch.zeros((5, 2), dtype=torch.bfloat16),
                hidden_size=2,
                target_taps=2,
            )
        synthetic = dict(generated, eos_was_synthesized=True)
        with self.assertRaisesRegex(ValueError, "synthetic EOS"):
            build_cache_sample(
                synthetic,
                torch.zeros((5, 4), dtype=torch.bfloat16),
                torch.zeros((5, 2), dtype=torch.bfloat16),
                hidden_size=2,
                target_taps=2,
            )
        self.assertEqual(
            reconcile_written_records([{"id": "a"}, {"id": "b"}], ["a", "b"])[
                "written_row_count"
            ],
            2,
        )
        with self.assertRaisesRegex(ValueError, "written row IDs"):
            reconcile_written_records([{"id": "a"}, {"id": "b"}], ["b", "a"])

    def test_collator_and_index_are_padded_without_eos(self) -> None:
        first = {"input_ids": [0, 2, 3], "loss_mask": [0, 1, 1]}
        second = {"input_ids": [0, 2], "loss_mask": [0, 1]}
        batch = collate_pretokenized([first, second], pad_token_id=1)
        self.assertEqual(batch["input_ids"], [[0, 2, 3], [0, 2, 1]])
        self.assertEqual(batch["loss_mask"], [[0, 1, 1], [0, 1, 0]])
        tensor_batch = PretokenizedTargetCollator(min_loss_tokens=1)([first, second])
        self.assertEqual(tuple(tensor_batch["input_ids"].shape), (2, 3))
        self.assertEqual(str(tensor_batch["attention_mask"].dtype), "torch.uint8")
        self.assertEqual(tensor_batch["loss_mask"].tolist(), [[0, 1, 1], [0, 1, 0]])
        raw = pack_index_record(
            sample_id=7,
            shard_id=2,
            sequence_length=3,
            input_ids_offset=100,
            attention_mask_offset=112,
            loss_mask_offset=115,
            target_hidden_states_offset=118,
            target_last_hidden_states_offset=130,
        )
        self.assertEqual(len(raw), CACHE_INDEX_BYTES)
        unpacked = unpack_index_record(raw)
        self.assertEqual(unpacked["sample_id"], 7)
        self.assertEqual(unpacked["shard_id"], 2)
        self.assertEqual(unpacked["seq_len"], 3)

    def test_manifest_records_train_only_and_target_identity(self) -> None:
        manifest = cache_manifest(
            target_identity={"name": "control250", "checkpoint_sha256": "pending"},
            source_rows_sha256="rows",
            tokenizer_sha256="tokenizer",
            renderer="zeta2-prm03-v1",
            selected_row_ids=["a", "b"],
            hidden_size=2,
            target_taps=2,
            target_layer_ids=(1, 2),
            shards=({"shard_id": 0, "file_name": "shard-00000.bin"},),
        )
        self.assertEqual(manifest["schema"], "deepspec_target_cache_v2")
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["index_record_size"], 56)
        self.assertEqual(manifest["target_layer_ids"], [1, 2])
        self.assertEqual(manifest["written_row_count"], 2)
        self.assertFalse(manifest["cache_writer_requires_eos"])
        self.assertEqual(manifest["authored_tail_substitution"], "forbidden")

        filtered = cache_manifest(
            target_identity={"name": "control250"},
            source_rows_sha256="rows",
            tokenizer_sha256="tokenizer",
            renderer="pretokenized",
            selected_row_ids=["a", "b", "c"],
            written_row_ids=["a", "c"],
            hidden_size=2,
            target_taps=2,
            target_layer_ids=(1, 2),
            shards=({"shard_id": 0, "file_name": "shard-00000.bin"},),
        )
        self.assertEqual(filtered["selected_row_count"], 3)
        self.assertEqual(filtered["written_row_count"], 2)
        self.assertEqual(filtered["num_samples"], 2)

    def test_official_writer_reader_roundtrip_preserves_eog_length(self) -> None:
        import torch

        vendor = "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-draft-upstream-smoke-v1/vendor/deepspec"
        sys.path.insert(0, vendor)
        from deepspec.data.target_cache_dataset import (
            CacheCollator,
            CacheDataset,
            INDEX_RECORD_SIZE,
            LocalCacheWriteSummary,
            LocalTargetCacheWriter,
            TARGET_CACHE_VERSION,
            atomic_json_dump,
            build_global_target_cache_shard_map,
            build_target_cache_manifest,
            cleanup_target_cache_tmp_dir,
            finalize_target_cache_index,
            prepare_target_cache_output_dir,
            rename_local_target_cache_shards,
            write_target_cache_manifest,
        )

        root = Path(tempfile.mkdtemp(prefix="sepalith-hardening-cache-", dir="/tmp"))
        cache_dir = root / "cache"
        rank_dir = cache_dir / "_tmp" / "rank_0"
        writer = None
        dataset = None
        try:
            prepare_target_cache_output_dir(str(cache_dir))
            rank_dir.mkdir(parents=True)
            ids = torch.tensor([0, 2, 3, 1], dtype=torch.int32)
            attention = torch.ones(4, dtype=torch.uint8)
            loss = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
            hidden = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4).bfloat16()
            last = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2).bfloat16()
            writer = LocalTargetCacheWriter(rank_dir=str(rank_dir), max_shard_bytes=1 << 20)
            writer.write_sample(
                sample_id=0,
                input_ids=ids,
                attention_mask=attention,
                loss_mask=loss,
                target_hidden_states=hidden,
                target_last_hidden_states=last,
            )
            local_files = list(writer.local_shard_files)
            writer.close()
            writer = None
            summary = LocalCacheWriteSummary(
                global_rank=0,
                source_sample_start=0,
                source_sample_end=1,
                num_local_samples=1,
                num_local_shards=len(local_files),
                local_shard_files=local_files,
            )
            atomic_json_dump(summary.to_json(), rank_dir / "summary.json")
            summaries = [summary.to_json()]
            shard_map, shards = build_global_target_cache_shard_map(summaries)
            rename_local_target_cache_shards(
                output_dir=str(cache_dir),
                rank_dir=str(rank_dir),
                summary=summary.to_json(),
                shard_map=shard_map,
            )
            count = finalize_target_cache_index(
                output_dir=str(cache_dir), summaries=summaries, shard_map=shard_map
            )
            write_target_cache_manifest(
                output_dir=str(cache_dir),
                manifest=build_target_cache_manifest(
                    num_samples=count,
                    shards=shards,
                    target_layer_ids=[1, 2],
                    hidden_size=2,
                ),
            )
            cleanup_target_cache_tmp_dir(str(cache_dir))
            dataset = CacheDataset(str(cache_dir))
            item = dataset[0]
            collated = CacheCollator()([item])
            self.assertEqual(TARGET_CACHE_VERSION, 2)
            self.assertEqual(INDEX_RECORD_SIZE, 56)
            self.assertEqual(dataset._read_record(0)["seq_len"], 4)
            self.assertEqual(collated["attention_mask"].tolist(), [[1, 1, 1, 1]])
            self.assertTrue(torch.equal(item["input_ids"], ids))
            self.assertTrue(torch.equal(item["target_hidden_states"], hidden))
            self.assertTrue(torch.equal(item["target_last_hidden_states"], last))
        finally:
            if dataset is not None:
                dataset.close()
            if writer is not None:
                writer.close()
            shutil.rmtree(root, ignore_errors=True)


class PublicWarmStartTest(unittest.TestCase):
    def test_pinned_public_geometry_matches_campaign_target(self) -> None:
        contract = public_config_contract()
        self.assertEqual(PUBLIC_REVISION, "114a20fdbf53220712c7fbdd7dccddbf1dedebb4")
        self.assertEqual(contract["target_layer_ids"], [1, 10, 20, 30, 39])
        self.assertEqual(contract["mask_token_id"], 75982)
        compatibility = compare_target_config({
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "vocab_size": 130560,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "eos_token_id": [1, 130073],
            "model_type": "llama",
        })
        self.assertTrue(compatibility["structurally_compatible"])
        self.assertTrue(compatibility["target_identity_still_required"])
        self.assertTrue(compatibility["teacher_cache_still_required"])

class ExportGateTest(unittest.TestCase):
    def test_positive_and_negative_headers(self) -> None:
        report = self_test()
        self.assertTrue(report["positive"]["passed"])
        self.assertTrue(report["negative"]["rejected"])
        bad = copy.deepcopy(synthetic_header())
        bad["eos_was_synthesized"] = True
        with self.assertRaisesRegex(ValueError, "synthetic EOS"):
            validate_export_header(bad)


class TinyModelSmokeTest(unittest.TestCase):
    def test_tiny_llama_forward_and_dspark_loss(self) -> None:
        from tiny_dspark_smoke import run_smoke

        result = run_smoke()
        if result.get("skipped"):
            self.skipTest(str(result["reason"]))
        self.assertTrue(result["target_forward"])
        self.assertTrue(result["draft_forward"])
        self.assertTrue(result["loss_finite_and_gradients_finite"])
        self.assertEqual(result["compat_sliding_window"], None)
        self.assertEqual(result["serving_stop_token_ids"], [1, 63, 130073])


if __name__ == "__main__":
    unittest.main(verbosity=2)
