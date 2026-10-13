"""CPU checks for the bounded PRM-07 profile input and measurement contract."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("campaign_profile.py")
SPEC = importlib.util.spec_from_file_location("campaign_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
sys.modules["campaign_profile"] = profile
SPEC.loader.exec_module(profile)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, length: int, *, split: str = "train") -> dict[str, object]:
    if length < 8:
        raise ValueError(length)
    prompt_count = length - 5
    prompt_ids = [100] * prompt_count
    body_ids = [101, 102]
    terminal_ids = [103]
    ids = [profile.BOS_ID, *prompt_ids, *body_ids, *terminal_ids, profile.EOS_ID]
    assert len(ids) == length
    return {
        "id": row_id,
        "input_ids": ids,
        "target_start": 1 + prompt_count,
        "target_body_tokens": body_ids,
        "target_terminal_tokens": terminal_ids,
        "target_body_token_count": len(body_ids),
        "target_terminal_token_count": len(terminal_ids),
        "family": "fixture",
        "package_id": "fixture-package",
        "renderer_id": "zeta2-prm03-v1",
        "prompt_text": "fixture prompt\n",
        "target_text": "x\n>>>>>>> UPDATED",
        "target_body_text": "x",
        "target_operation": "replace",
        "prompt_token_count": prompt_count,
        "target_token_count": len(body_ids) + len(terminal_ids),
        "bos_token_id": profile.BOS_ID,
        "eos_token_id": profile.EOS_ID,
        "tokenizer_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
        "tokenizer_json_sha256": profile.EXPECTED_TOKENIZER_JSON_SHA256,
        "tokenization_policy": "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1",
        "split": split,
    }


class CampaignProfileTests(unittest.TestCase):
    def _files(self) -> tuple[profile.ProfileInput, tuple[str, ...], list[dict[str, object]]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        candidate = directory / "candidate.jsonl"
        rows = [_row("short-a", 12), _row("short-b", 2_048), _row("long-a", 2_049)]
        candidate.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        selected = directory / "selected.json"
        ids = {"schema_version": profile.SELECTED_IDS_SCHEMA_VERSION, "split": "train", "row_ids": ["long-a", "short-a", "short-b"]}
        selected.write_text(json.dumps(ids, sort_keys=True) + "\n", encoding="utf-8")
        return profile.ProfileInput(
            candidate_file=candidate, candidate_sha256=_sha(candidate),
            selected_ids_file=selected, selected_ids_sha256=_sha(selected),
            model_path=profile.EXPECTED_MODEL_PATH,
        ), tuple(ids["row_ids"]), rows

    def test_module_import_has_no_cuda_framework_import(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(name == "torch" or name.startswith(("torch.", "unsloth", "transformers", "trl")) for name in imported))

    def test_explicit_hashes_and_train_ids_select_natural_buckets(self):
        input_spec, ids, _rows = self._files()
        selected_ids, rows, summary = profile.load_profile_rows(input_spec)
        self.assertEqual(selected_ids, ids)
        self.assertEqual([row["id"] for row in rows], list(ids))
        self.assertEqual(summary["natural_buckets"], {"short": 2, "long": 1, "out_of_profile": 0})
        with self.assertRaises(profile.ProfileInputError):
            profile.load_profile_rows(profile.ProfileInput(
                input_spec.candidate_file, "0" * 64,
                input_spec.selected_ids_file, input_spec.selected_ids_sha256,
            ))

    def test_candidate_only_envelope_checks_audited_lengths_without_admission(self):
        input_spec, ids, rows = self._files()
        envelopes = []
        for row in rows:
            envelopes.append({
                "status": "tokenizer_candidate_only",
                "row": row,
                "source_ref": {"split": "train_group"},
                "lengths": {
                    "prompt_with_bos": row["prompt_token_count"] + 1,
                    "prompt_without_bos": row["prompt_token_count"],
                    "target_body": len(row["target_body_tokens"]),
                    "response_with_terminal_eos": row["target_token_count"] + 1,
                    "sequence": len(row["input_ids"]),
                },
                "admitted_for_training": False,
            })
        input_spec.candidate_file.write_text(
            "".join(json.dumps(entry) + "\n" for entry in envelopes), encoding="utf-8",
        )
        wrapped = profile.ProfileInput(
            input_spec.candidate_file, _sha(input_spec.candidate_file),
            input_spec.selected_ids_file, input_spec.selected_ids_sha256,
        )
        _selected, _selected_rows, summary = profile.load_profile_rows(wrapped)
        self.assertEqual(summary["entry_kinds"], {"tokenizer_candidate_only": 3})
        envelopes[0]["lengths"]["sequence"] += 1
        input_spec.candidate_file.write_text(
            "".join(json.dumps(entry) + "\n" for entry in envelopes), encoding="utf-8",
        )
        stale = profile.ProfileInput(
            input_spec.candidate_file, _sha(input_spec.candidate_file),
            input_spec.selected_ids_file, input_spec.selected_ids_sha256,
        )
        with self.assertRaisesRegex(profile.ProfileInputError, "length audit mismatch"):
            profile.load_profile_rows(stale)

    def test_dev_row_cannot_enter_candidate_registry(self):
        input_spec, _ids, rows = self._files()
        raw = input_spec.candidate_file.read_text(encoding="utf-8")
        bad = _row("dev", 12, split="dev")
        input_spec.candidate_file.write_text(raw + json.dumps(bad) + "\n", encoding="utf-8")
        input_spec = profile.ProfileInput(
            input_spec.candidate_file, _sha(input_spec.candidate_file),
            input_spec.selected_ids_file, input_spec.selected_ids_sha256,
        )
        with self.assertRaisesRegex(profile.ProfileInputError, "dev/final"):
            profile.load_profile_rows(input_spec)

    def test_padding_is_eos_but_never_a_useful_loss_position(self):
        import torch

        batch = profile.collate_profile_rows([_row("a", 10), _row("b", 12)], torch)
        self.assertEqual(batch["input_ids"].shape, (2, 12))
        self.assertEqual(batch["input_ids"][0, 10:].tolist(), [profile.EOS_ID, profile.EOS_ID])
        self.assertEqual(batch["attention_mask"][0, 10:].tolist(), [0, 0])
        self.assertEqual(batch["labels"][0, 10:].tolist(), [-100, -100])
        self.assertEqual(batch["labels"].ne(-100).sum().item(), 22)
        self.assertEqual(batch["prompt_tokens"], 5 + 7)
        self.assertEqual(batch["target_tokens"], 4 + 4)

    def test_fake_forward_backward_profile_reports_actual_denominators(self):
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, attention_mask, labels):
                valid = labels.ne(-100)
                loss = (self.scale * input_ids.float() * valid).sum() / valid.sum()
                return type("Output", (), {"loss": loss})()

        rows = [_row("s1", 10), _row("s2", 12), _row("l1", 2_049)]
        model = FakeModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        result = profile.measure_sft_profile(
            model, optimizer, torch, rows, warmup_steps=1, timed_steps=1, batch_size=2,
        )
        self.assertEqual(result["short"]["rows"], 2)
        self.assertEqual(result["short"]["actual_sequence_tokens"], 22)
        self.assertEqual(result["short"]["full_text_label_tokens"], 22)
        self.assertEqual(result["short"]["padded_slots_excluded_from_denominator"], 2)
        self.assertEqual(result["long"]["rows"], 1)
        self.assertEqual(result["long"]["actual_sequence_tokens"], 2_049)
        self.assertIsNone(result["short"]["peak_allocated_bytes"])

    def test_rl_accounting_exposes_eos_cap_and_control_counts(self):
        records = [
            {"prompt_tokens": [profile.BOS_ID, 100], "generated_tokens": [100, profile.EOS_ID], "terminal_reason": "eos"},
            {"prompt_tokens": [profile.BOS_ID, 100], "generated_tokens": [100] * 192, "terminal_reason": "length"},
            {"prompt_tokens": [profile.BOS_ID, 100], "generated_tokens": [10, profile.EOS_ID], "terminal_reason": "eos"},
            {"prompt_tokens": [profile.BOS_ID, 100], "generated_tokens": [100, 130073], "terminal_reason": "eos"},
        ]
        counts = profile.validate_rl_candidate_records(records)
        self.assertEqual(counts["records"], 4)
        self.assertEqual(counts["canonical_eos"], 2)
        self.assertEqual(counts["noncanonical_eog"], 1)
        self.assertEqual(counts["cap_hits"], 1)
        self.assertEqual(counts["control_before_terminal"], 1)
        with self.assertRaises(profile.ProfileInputError):
            profile.validate_rl_candidate_records([{
                "prompt_tokens": [profile.BOS_ID] + [100] * 2_048,
                "generated_tokens": [profile.EOS_ID],
                "terminal_reason": "eos",
            }])

    def test_non_finite_loss_cannot_be_reported_as_completed_update(self):
        loss = mock.Mock()
        loss.detach.return_value.item.return_value = float('nan')
        model = mock.Mock(return_value=type('Output', (), {'loss': loss})())
        optimizer = mock.Mock()
        batch = {'input_ids': mock.Mock(), 'attention_mask': mock.Mock(), 'labels': mock.Mock()}
        with self.assertRaisesRegex(RuntimeError, 'non_finite_profile_loss'):
            profile._optimizer_step(model, optimizer, batch, 'cpu')
        loss.backward.assert_not_called()
        optimizer.step.assert_not_called()

    def test_wall_timeout_and_preflight_are_framework_free_paths(self):
        with self.assertRaises(profile.ProfileTimeout):
            with profile.wall_clock_limit(0.02) as deadline:
                time.sleep(0.1)
                deadline.check()
        input_spec, _ids, _rows = self._files()
        fake_identity = {"path": str(profile.EXPECTED_MODEL_PATH), "revision": profile.EXPECTED_MODEL_REVISION}
        with mock.patch.object(profile, "model_identity", return_value=fake_identity), \
                mock.patch.object(profile, "_frameworks_imported", return_value=[]):
            result = profile.preflight_profile(input_spec)
        self.assertFalse(result["CUDA_started"])
        self.assertEqual(result["natural_buckets"]["long"]["rows"], 1)

    def test_live_guard_requires_clear_single_device_before_framework_import(self):
        clear_processes = mock.Mock(stdout="")
        clear_devices = mock.Mock(stdout="0, 20, 32768")
        with mock.patch.object(profile, "_frameworks_imported", return_value=[]), \
                mock.patch.object(profile.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(profile.subprocess, "run", side_effect=[clear_processes, clear_devices]):
            result = profile.live_resource_guard(max_existing_vram_mib=1024)
        self.assertEqual(result["status"], "clear")
        self.assertEqual(result["gpu"]["memory_used_mib"], 20)

        busy_processes = mock.Mock(stdout="1234, 2048")
        with mock.patch.object(profile, "_frameworks_imported", return_value=[]), \
                mock.patch.object(profile.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(profile.subprocess, "run", side_effect=[busy_processes, clear_devices]):
            with self.assertRaisesRegex(profile.ProfileGuardError, "active compute"):
                profile.live_resource_guard(max_existing_vram_mib=1024)

    def test_preflight_failure_writes_clean_receipt(self):
        input_spec, _ids, _rows = self._files()
        receipt = input_spec.candidate_file.with_name("failed-receipt.json")
        code = profile.main([
            "--preflight",
            "--candidate-file", str(input_spec.candidate_file), "--candidate-sha256", "0" * 64,
            "--selected-train-ids", str(input_spec.selected_ids_file),
            "--selected-ids-sha256", input_spec.selected_ids_sha256,
            "--receipt", str(receipt),
        ])
        self.assertEqual(code, 1)
        value = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "input_failed")
        self.assertEqual(value["error"]["type"], "ProfileInputError")


if __name__ == "__main__":
    unittest.main()
