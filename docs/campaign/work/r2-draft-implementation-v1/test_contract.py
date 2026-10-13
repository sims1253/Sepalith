from __future__ import annotations

import copy
import unittest

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
    pack_index_record,
    unpack_index_record,
)
from prepare_teacher_cache import regenerate_train_rows
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
        record = build_teacher_record(self.ROW, [22, 23], protocol_status="valid")
        self.assertEqual(record["input_ids"], [0, 10, 11, 22, 23])
        self.assertEqual(record["loss_mask"], [0, 0, 0, 1, 1])
        self.assertEqual(record["target_generated_ids"], [22, 23])
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

    def test_generation_error_does_not_abort_other_rows(self) -> None:
        rows = [dict(self.ROW, id=f"train-{index}") for index in range(3)]

        def generator(prompt: list[int], cap: int):
            if prompt[-1] == 11:
                # Every row has this prompt; the first call's behavior is
                # enough to verify row-level error handling in this fixture.
                generator.calls += 1
                if generator.calls == 1:
                    raise RuntimeError("simulated target service error")
            return {"generated_ids": [22, 1], "protocol_status": "invalid"}

        generator.calls = 0
        records, summary = regenerate_train_rows(rows, generator, limit=3, generation_cap=4, vocab_size=64)
        self.assertEqual(len(records), 3)
        self.assertEqual(summary["selected_rows"], 3)
        self.assertEqual(summary["abort_on_protocol_invalid"], False)
        self.assertGreaterEqual(summary["protocol_invalid"], 1)
        self.assertTrue(any(record["generation_status"] == "generation_error" for record in records))


class CacheBridgeTest(unittest.TestCase):
    def test_payload_geometry_and_sample(self) -> None:
        # For H=2 and 2 target taps: 4+1+1+2*2*3=18 bytes/token.
        self.assertEqual(cache_payload_nbytes(3, hidden_size=2, target_taps=2), 54)
        self.assertEqual(cache_payload_nbytes(1), 24582)
        generated = build_teacher_record(
            {"id": "r", "input_ids": [0, 10, 11, 100], "target_start": 3},
            [22, 23],
        )
        sample = build_cache_sample(
            generated,
            [[0.0] * 4 for _ in range(5)],
            [[0.0] * 2 for _ in range(5)],
            hidden_size=2,
            target_taps=2,
        )
        self.assertEqual(sample["sequence_length"], 5)
        self.assertEqual(sample["target_token_count"], 2)
        self.assertFalse(sample["cache_writer_requires_eos"])
        self.assertEqual(sample["eos_injection"], "forbidden")

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
        self.assertFalse(manifest["cache_writer_requires_eos"])
        self.assertEqual(manifest["authored_tail_substitution"], "forbidden")


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
